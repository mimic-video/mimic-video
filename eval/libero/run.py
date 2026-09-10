"""Run LIBERO evaluation with the Video Action Model (VAM) policy."""

from __future__ import annotations

import json
import os
import pathlib
import random
from collections import deque
from collections.abc import Iterable
from pathlib import Path

import imageio
import numpy as np
import torch
import tqdm
import tyro
from einops import rearrange
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from scipy.spatial.transform import Rotation

from cosmos_predict2.configs.config import make_config
from cosmos_predict2.data.action.utils import extract_normalization_types
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.pipelines.video2world2action import Video2World2ActionPipeline
from cosmos_predict2.pipelines.world2action import World2ActionPipeline
from imaginaire.lazy_config import instantiate
from imaginaire.utils.config_helper import override

LIBERO_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
}

CAMERA_HEIGHT = 480
CAMERA_WIDTH = 640

DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def set_seed_everywhere(seed: int) -> None:
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_video2world2action_pipeline(
    experiment_name: str,
    video_model_path: str,
    action_model_path: str,
    dataset_statistics_path: pathlib.Path,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> Video2World2ActionPipeline:
    """Instantiate the video-to-world-to-action pipeline and load normalizer statistics."""
    config = make_config()
    config = override(config, ["--", f"experiment={experiment_name}"])

    # all libero task descriptions have been verified to be unproblematic
    config.model.config.video_pipe_config.guardrail_config.enabled = False

    video2world_pipe = Video2WorldPipeline.from_config(
        config=config.model.config.video_pipe_config,
        dit_path=video_model_path,
        device=device,
        torch_dtype=dtype,
        load_ema_to_reg=False,
    )

    world2action_pipe = World2ActionPipeline.from_config(
        config.model.config.pipe_config,
        dit_path=action_model_path,
        device=device,
        dtype=dtype,
    )

    data_config = instantiate(config.data_config)

    with dataset_statistics_path.open("rb") as stats_file:
        stats = json.load(stats_file)
    world2action_pipe.normalizer.build_from_stats(
        stats,
        normalization_types=extract_normalization_types(data_config.policy_io.policy_io),
        concat_groups=data_config.policy_io.concat_groups,
        device=device,
        dtype=dtype,
    )

    return Video2World2ActionPipeline(video2world_pipe, world2action_pipe).to(device)


class VAMInference:
    """Helper class that maintains temporal context and queries the VAM policy."""

    def __init__(
        self,
        experiment_name: str,
        video_model_path: str,
        action_model_path: str,
        dataset_statistics_path: pathlib.Path,
        img_horizon: int,
        lowdim_horizon: int,
        stop_video_denoising_step: int,
        num_execute_actions: int,
        rollout_dir: pathlib.Path,
        device: str = "cuda",
    ):
        self.device = device
        self.model = load_video2world2action_pipeline(
            experiment_name,
            video_model_path,
            action_model_path,
            dataset_statistics_path,
            device=device,
        )
        self._image_horizon = img_horizon
        self._lowdim_horizon = lowdim_horizon
        self.stop_video_denoising_step = stop_video_denoising_step
        self.num_execute_actions = num_execute_actions
        self.num_sampling_steps = 35
        self.rollout_dir = rollout_dir
        self.reset(task_description="")

    def reset(self, task_description: str) -> None:
        """Reset internal state for a new task/episode."""
        self.task_description = task_description
        self._image_history: deque[np.ndarray] = deque(maxlen=(self._image_horizon - 1) * 4 + 1)
        self._lowdim_history: deque[np.ndarray] = deque(maxlen=self._lowdim_horizon)
        self.action_buffer: np.ndarray | None = None
        self.action_buffer_idx = 0
        self._execute_horizon = 0

    def step(
        self,
        image: np.ndarray,
        task_description: str,
        obs: dict,
    ) -> np.ndarray:
        """Return the next action for the given observation."""
        if image.dtype != np.uint8:
            raise ValueError(f"Expected image dtype uint8, received {image.dtype}.")

        if task_description != self.task_description:
            self.reset(task_description)

        processed_image = self._process_image(image)
        self._add_image_to_history(processed_image)

        state_vec = self._state_from_observation(obs)
        self._add_lowdim_to_history(state_vec)

        if self.action_buffer is None:
            self._query_policy(task_description)

        current_action = self.action_buffer[self.action_buffer_idx]
        self.action_buffer_idx += 1
        if self.action_buffer_idx >= self._execute_horizon:
            self.action_buffer = None

        return self._convert_action(current_action)

    def _query_policy(self, task_description: str) -> None:
        """Query the model and cache the planned action sequence."""
        images = np.concatenate(list(self._image_history)[::4], axis=1)  # downsample from 20 fps to 5
        lowdims = np.stack(list(self._lowdim_history), axis=0)

        input_vid = torch.from_numpy(images[None]).to(device=self.device, dtype=torch.bfloat16)
        state_tensor = torch.from_numpy(lowdims[None]).to(device=self.device, dtype=torch.bfloat16)

        with torch.no_grad():
            pred_actions = self.model(
                input_vid=input_vid,
                state_B_HO_O=state_tensor,
                prompt=task_description,
                num_sampling_step=self.num_sampling_steps,
                stop_after_step=self.stop_video_denoising_step,
                use_cuda_graphs=True,
            )

        self.action_buffer = pred_actions[0].float().cpu().numpy()
        self._execute_horizon = self.num_execute_actions
        self.action_buffer_idx = 0

    def _process_image(self, image: np.ndarray) -> np.ndarray:
        tensor = rearrange(image, "h w c -> c h w")[:, None, :, :]
        return 2.0 * (tensor.astype(np.float32) / 255.0 - 0.5)

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self._image_history.append(image)
        while len(self._image_history) < self._image_history.maxlen:
            self._image_history.append(image.copy())

    def _add_lowdim_to_history(self, lowdim: np.ndarray) -> None:
        self._lowdim_history.append(lowdim)
        while len(self._lowdim_history) < self._lowdim_horizon:
            self._lowdim_history.append(lowdim.copy())

    @staticmethod
    def _state_from_observation(obs: dict[str, np.ndarray]) -> np.ndarray:
        rot_6d = Rotation.from_quat(obs["robot0_eef_quat"]).as_matrix()[:2].reshape((6,))
        return np.concatenate((obs["robot0_eef_pos"], rot_6d, obs["robot0_gripper_qpos"][0][None]), axis=0)

    @staticmethod
    def _matrix_from_6d(orient6: np.ndarray) -> np.ndarray:
        r1 = orient6[:3]
        r2 = orient6[3:]
        r1_norm = r1 / (np.linalg.norm(r1) + 1e-9)
        r2_orth = r2 - np.dot(r2, r1_norm) * r1_norm
        r2_norm = r2_orth / (np.linalg.norm(r2_orth) + 1e-9)
        r3 = np.cross(r1_norm, r2_norm)
        return np.stack([r1_norm, r2_norm, r3], axis=0)

    def _convert_action(self, action: np.ndarray) -> np.ndarray:
        delta_pos = action[:3]
        rot_matrix = self._matrix_from_6d(action[3:9])
        rot_vec = Rotation.from_matrix(rot_matrix).as_rotvec()
        gripper = np.sign(action[9][None])
        return np.concatenate([delta_pos, rot_vec, gripper], axis=0)


def get_libero_env(task) -> tuple[OffScreenRenderEnv, str]:
    """Initializes and returns the LIBERO environment alongside the task description."""
    task_description = task.language.replace("black bowl", "bowl")
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": CAMERA_HEIGHT,
        "camera_widths": CAMERA_WIDTH,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


def get_libero_image(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Extract the agentview image and check that it matches the expected resolution."""
    image = obs["agentview_image"][::-1, ::-1]
    if image.shape != (CAMERA_HEIGHT, CAMERA_WIDTH, 3):
        raise ValueError(f"Unexpected agentview image shape {image.shape}.")
    return image


def save_rollout_video(
    rollout_images: Iterable[np.ndarray],
    idx: int,
    success: bool,
    task_description: str,
    rollout_dir: Path,
) -> Path:
    """Save an MP4 replay of the episode."""
    rollout_dir.mkdir(parents=True, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")
    mp4_path = rollout_dir / f"episode{idx}_{'success' if success else 'failure'}_{processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=20)
    try:
        for img in rollout_images:
            video_writer.append_data(img)
    finally:
        video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    return mp4_path


def run_episode(
    env: OffScreenRenderEnv,
    policy: VAMInference,
    task_description: str,
    initial_observation: dict[str, np.ndarray],
    max_steps: int,
    num_steps_wait: int,
) -> tuple[bool, list[np.ndarray]]:
    """Execute a single episode and return success flag along with captured frames."""
    obs = initial_observation
    replay_images: list[np.ndarray] = []
    success = False

    for step_idx in range(max_steps + num_steps_wait):
        if step_idx < num_steps_wait:
            obs, _, done, info = env.step(DUMMY_ACTION)
            if done:
                success = True
                break
            continue

        image = get_libero_image(obs)
        replay_images.append(image)

        action = policy.step(image, task_description, obs)

        obs, _, done, info = env.step(action.tolist())
        if done:
            success = True
            break

    return success, replay_images


def eval_vam_libero(
    vam_experiment_name: str,
    vam_video_model_path: str,
    vam_action_model_path: pathlib.Path,
    vam_dataset_statistics_path: pathlib.Path,
    vam_img_horizon: int,
    vam_lowdim_horizon: int,
    vam_stop_video_denoising_step: int,
    vam_num_execute_actions: int,
    task_suite_name: str,
    num_trials_per_task: int = 50,
    eval_rank: int = 0,
    eval_world_size: int = 1,
    num_steps_wait: int = 10,
    seed: int = 0,
    device: str = "cuda",
) -> None:
    set_seed_everywhere(seed)

    run_label = (
        f"{vam_action_model_path.stem}_stopafter{vam_stop_video_denoising_step}_execute{vam_num_execute_actions}"
    )
    rollout_dir = Path("./results") / run_label / task_suite_name
    rollout_dir.mkdir(parents=True, exist_ok=True)

    policy = VAMInference(
        vam_experiment_name,
        vam_video_model_path,
        str(vam_action_model_path),
        vam_dataset_statistics_path,
        vam_img_horizon,
        vam_lowdim_horizon,
        vam_stop_video_denoising_step,
        vam_num_execute_actions,
        rollout_dir,
        device=device,
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    if task_suite_name not in benchmark_dict:
        raise ValueError(f"Task suite {task_suite_name} not available.")
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks

    max_steps = LIBERO_SUITE_MAX_STEPS[task_suite_name]

    total_episodes = 0
    total_successes = 0

    for task_id in tqdm.tqdm(range(num_tasks), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task)

        if len(initial_states) == 0:
            raise ValueError(f"No initial states provided for task {task_id}.")

        task_successes = 0
        task_episodes = 0

        try:
            for episode_idx in tqdm.tqdm(range(num_trials_per_task), desc="Episodes", leave=False):
                task_episodes += 1
                total_episodes += 1

                if total_episodes % eval_world_size != eval_rank:
                    continue
                should_skip = False
                for ep in map(str, rollout_dir.iterdir()):
                    if f"episode{total_episodes}_" not in ep:
                        continue
                    should_skip = True
                    if "success" in ep:
                        task_successes += 1
                        total_successes += 1
                    break
                if should_skip:
                    continue

                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])

                policy.reset(task_description)

                success, replay_images = run_episode(
                    env,
                    policy,
                    task_description,
                    obs,
                    max_steps,
                    num_steps_wait,
                )

                if success:
                    task_successes += 1
                    total_successes += 1

                save_rollout_video(
                    replay_images,
                    total_episodes,
                    success,
                    task_description,
                    rollout_dir,
                )

                success_rate = total_successes / max(total_episodes, 1)
                print(
                    f"Task {task_id} | Episode {episode_idx + 1} | Success: {success} "
                    f"| Total Success Rate: {success_rate:.3f}\n"
                )
        finally:
            env.close()

        task_success_rate = task_successes / max(task_episodes, 1)
        print(f"Task {task_id} success rate: {task_success_rate:.3f}")

    overall_success_rate = total_successes / max(total_episodes, 1)
    print(
        f"Completed {total_episodes} episodes | "
        f"Total successes: {total_successes} | "
        f"Overall success rate: {overall_success_rate:.3f}\n"
    )


if __name__ == "__main__":
    tyro.cli(eval_vam_libero)
