import collections
import json
import pathlib
from collections.abc import Sequence

import einops
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from scipy.spatial.transform import Rotation
from torchvision import transforms

from cosmos_predict2.configs.config import make_config
from cosmos_predict2.data.action.utils import extract_normalization_types
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.pipelines.video2world2action import Video2World2ActionPipeline
from cosmos_predict2.pipelines.video2world2action_gtvid import (
    Video2World2ActionPipeline as HILVideo2World2ActionPipeline,
)
from cosmos_predict2.pipelines.world2action import World2ActionPipeline
from imaginaire.lazy_config import instantiate
from imaginaire.utils.config_helper import override


def load_video2world2action_pipeline(
    experiment_name: str,
    video_model_path: str,
    action_model_path: str,
    dataset_statistics_path: pathlib.Path,
    is_hil: bool,
    device: str = "cuda",
):
    config = make_config()
    config = override(config, ["--", f"experiment={experiment_name}"])

    # all simpler-bridge task descriptions have been verified to be unproblematic
    config.model.config.video_pipe_config.guardrail_config.enabled = False

    video2world_pipe = Video2WorldPipeline.from_config(
        config=config.model.config.video_pipe_config,
        dit_path=video_model_path,
        device=device,
        torch_dtype=torch.bfloat16,
        load_ema_to_reg=False,
    )

    world2action_pipe = World2ActionPipeline.from_config(
        config.model.config.pipe_config,
        dit_path=action_model_path,
        device=device,
        dtype=torch.bfloat16,
    )
    data_config = instantiate(config.data_config)

    with dataset_statistics_path.open("rb") as stats_file:
        stats = json.load(stats_file)

    world2action_pipe.normalizer.build_from_stats(
        stats,
        normalization_types=extract_normalization_types(data_config.policy_io.policy_io),
        concat_groups=data_config.policy_io.concat_groups,
        device=device,
        dtype=torch.bfloat16,
    )
    if is_hil:
        return HILVideo2World2ActionPipeline(video2world_pipe, world2action_pipe).to(device)

    return Video2World2ActionPipeline(video2world_pipe, world2action_pipe).to(device)


class VAMInference:
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
        is_hil: bool,
        device: str = "cuda",
    ):
        self.device = device
        self.model = load_video2world2action_pipeline(
            experiment_name,
            video_model_path,
            action_model_path,
            dataset_statistics_path,
            is_hil,
            device=device,
        )

        self._image_horizon = img_horizon
        self._lowdim_horizon = lowdim_horizon
        self.stop_video_denoising_step = stop_video_denoising_step
        self.num_execute_actions = num_execute_actions

        self.is_hil = is_hil

    def reset(self, task_description):
        self._image_history = None
        self._lowdim_history = None
        self.task_description = task_description
        self.action_buffer = None
        self.action_buffer_idx = 0
        self._viz_records = []  # each: {"obs": lowdim(10,), "pred_gripper": (H,1)}
        self._obs_hist = []  # per-step proprio: list of dict(pos, rot, gripper)
        self._plan_abs_Ts = None  # (H,4,4) absolute target poses in world
        self._last_target_p = None
        self._global_step = 0

        self._future_vid = None

    def _process_image(self, image: np.ndarray) -> np.ndarray:
        image = np.array(
            transforms.Resize((480, 640))(
                Image.fromarray(image, "RGB"),
            )
        )
        image = einops.rearrange(image, "h w c -> c h w")[:, None, :, :]
        return 2.0 * (image.astype(np.float32) / 255.0 - 0.5)

    def _add_image_to_history(self, image: np.ndarray) -> None:
        if self._image_history is None:
            self._image_history = collections.deque(maxlen=self._image_horizon)
        self._image_history.append(image)

    def _add_lowdim_to_history(self, lowdim: np.ndarray) -> None:
        if self._lowdim_history is None:
            self._lowdim_history = collections.deque(maxlen=self._lowdim_horizon)
            self._lowdim_history.extend([lowdim] * self._lowdim_horizon)
        else:
            self._lowdim_history.append(lowdim)

    def ingest_video(self, hil_images: np.ndarray) -> None:
        self._future_vid = np.take(
            np.concatenate([self._process_image(image) for image in hil_images], axis=1),
            indices=np.arange(61 - self._image_horizon),
            axis=1,
            mode="clip",
        )

    def step(
        self,
        image: np.ndarray,
        task_description: str,
        ee_pose_proprio,
        gripper_proprio,
    ) -> dict[str, np.ndarray]:
        assert image.dtype == np.uint8
        image = self._process_image(image)
        self._add_image_to_history(image)

        if task_description != self.task_description:
            self.reset(task_description)

        cur_rot = Rotation.from_quat(ee_pose_proprio.q, scalar_first=True)

        self._obs_hist.append(
            {
                "pos": ee_pose_proprio.p.copy(),
                "rot": cur_rot.as_euler(seq="XYZ").astype(np.float64),
                "gripper": float(gripper_proprio),
            }
        )

        cur_pos = ee_pose_proprio.p.astype(np.float64)

        ee_rot = cur_rot.as_matrix()[:2].reshape(6)
        lowdim_concat = np.concatenate((cur_pos, ee_rot, [gripper_proprio]))
        self._add_lowdim_to_history(lowdim_concat)

        if self.action_buffer is None:
            if len(self._image_history) == self._image_horizon:
                images = np.concatenate(self._image_history, axis=1)
            else:
                images = np.repeat(self._image_history[-1], self._image_horizon, axis=1)

            lowdims = np.stack(self._lowdim_history, axis=0)

            if self.is_hil:
                pred_actions = self.model(
                    input_vid=torch.from_numpy(images[None]).to(device=self.device, dtype=torch.bfloat16),
                    gt_future_vid=torch.from_numpy(self._future_vid[None]).to(device=self.device, dtype=torch.bfloat16),
                    state_B_HO_O=torch.from_numpy(lowdims[None]).to(device=self.device, dtype=torch.bfloat16),
                    prompt=task_description,
                    num_sampling_step=35,
                    stop_after_step=self.stop_video_denoising_step,
                    use_cuda_graphs=True,
                )
                self._future_vid = None
            else:
                pred_actions = self.model(
                    input_vid=torch.from_numpy(images[None]).to(device=self.device, dtype=torch.bfloat16),
                    state_B_HO_O=torch.from_numpy(lowdims[None]).to(device=self.device, dtype=torch.bfloat16),
                    prompt=task_description,
                    num_sampling_step=35,
                    stop_after_step=self.stop_video_denoising_step,
                    use_cuda_graphs=True,
                )
            self.action_buffer = pred_actions[0].float().cpu().numpy()
            self.action_buffer_idx = 0

            # store absolute pose targets for closed-loop deltas
            obs_T = self._pose_from_lowdim(lowdims[-1].astype(np.float64))
            pred_Ts_delta = [self._pose_from_lowdim(a.astype(np.float64)) for a in self.action_buffer]

            self._plan_abs_Ts = np.stack([obs_T @ Td for Td in pred_Ts_delta], axis=0)  # (H,4,4)

            if self._last_target_p is None:
                self._last_target_p = ee_pose_proprio.p

            self._viz_records.append(
                {
                    "pred_gripper": self.action_buffer[:, 9],  # (H,10)
                    "abs_Ts": self._plan_abs_Ts.copy(),  # (H,4,4) absolute targets
                    "start_t": self._global_step,
                }
            )

        T_target = self._plan_abs_Ts[self.action_buffer_idx]
        current_gripper_action = self.action_buffer[self.action_buffer_idx][9]

        self.action_buffer_idx += 1

        if self.action_buffer_idx >= self.num_execute_actions:
            self.action_buffer = None
        self._global_step += 1

        sim_actions = {}

        p_tgt = T_target[:3, 3]
        R_tgt = T_target[:3, :3]

        R_cur = Rotation.from_quat(ee_pose_proprio.q, scalar_first=True).as_matrix().astype(np.float32)
        R_err = R_tgt @ R_cur.T
        omega = Rotation.from_matrix(R_err).as_rotvec().astype(np.float32)
        u_r = omega / np.float32(np.pi / 2)
        n = np.linalg.norm(u_r)
        if n > 1.0:
            u_r /= n

        R_applied = Rotation.from_rotvec(np.float32(np.pi / 2) * u_r).as_matrix()
        p_cur = ee_pose_proprio.p.astype(np.float32)
        p_prev = self._last_target_p
        t_cmd = p_tgt - R_applied @ p_prev + (R_applied - np.eye(3, dtype=np.float32)) @ p_cur

        sim_actions["world_vector"] = t_cmd
        sim_actions["rot_axangle"] = u_r

        sim_actions["gripper"] = 2.0 * current_gripper_action[None] - 1.0

        sim_actions["terminate_episode"] = np.array([0.0], dtype=np.float32)

        self._last_target_p = R_applied @ p_prev + p_cur + t_cmd - R_applied @ p_cur

        return sim_actions

    @staticmethod
    def _pose_from_lowdim(lowdim: np.ndarray) -> np.ndarray:
        pos = lowdim[:3]
        r = lowdim[3:9].reshape(2, 3)

        r1 = r[0]
        r1_n = r1 / (np.linalg.norm(r1) + 1e-9)

        r2 = r[1]
        r2_orth = r2 - np.dot(r2, r1_n) * r1_n
        r2_n = r2_orth / (np.linalg.norm(r2_orth) + 1e-9)

        r3_n = np.cross(r1_n, r2_n)

        R = np.stack([r1_n, r2_n, r3_n], axis=0)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = pos
        return T

    # completely vibecoded months ago
    def visualize_epoch(
        self,
        _images_ignored: Sequence[np.ndarray],
        save_path: str,
    ) -> None:
        rot_bound = np.float32(np.pi / 2)

        T = len(self._obs_hist)
        if T == 0:
            plt.figure(figsize=(6, 2))
            plt.text(0.5, 0.5, "No data", ha="center", va="center")
            plt.savefig(save_path, dpi=150)
            plt.close()
            return

        obs_pos = np.stack([o["pos"] for o in self._obs_hist], axis=0)  # (T,3)
        obs_euler = np.stack([o["rot"] for o in self._obs_hist], axis=0)  # (T,3) radians, XYZ
        obs_Rs = Rotation.from_euler("XYZ", obs_euler).as_matrix()  # (T,3,3)
        obs_grip = np.array([o["gripper"] for o in self._obs_hist], dtype=float)  # (T,)

        plan_pos_traces = []  # list of (t_idx, y) pairs
        plan_eul_traces = []
        step_angle_traces = []  # per-step planned angle (between consecutive plan rotations), for saturation viz
        pos_err_traces = []  # ||p_obs - p_plan|| over time indices
        rot_err_traces = []  # geodesic angle ||log(R_plan^T R_obs)|| over time indices

        for rec in self._viz_records:
            abs_Ts = rec["abs_Ts"].astype(np.float64)  # (H,4,4)
            H = abs_Ts.shape[0]
            if H == 0:
                continue
            t0 = rec["start_t"] + 1  # first plan step corresponds to obs at t0
            t_idx = np.arange(t0, t0 + H, dtype=int)

            # plan pose
            R_plan = abs_Ts[:, :3, :3]
            p_plan = abs_Ts[:, :3, 3]
            eul_plan = Rotation.from_matrix(R_plan).as_euler("XYZ")  # radians

            # align to valid obs timesteps
            valid = (t_idx >= 0) & (t_idx < T)
            if not np.any(valid):
                continue
            t_idx = t_idx[valid]
            R_plan = R_plan[valid]
            p_plan = p_plan[valid]
            eul_plan = eul_plan[valid]

            # store plan traces (for plotting)
            plan_pos_traces.append((t_idx, p_plan))
            plan_eul_traces.append((t_idx, eul_plan))

            # position error
            p_obs = obs_pos[t_idx]
            pos_err = np.linalg.norm(p_obs - p_plan, axis=1)
            pos_err_traces.append((t_idx, pos_err))

            # rotation geodesic error
            R_obs = obs_Rs[t_idx]
            R_rel = np.einsum("tij,tjk->tik", np.transpose(R_plan, (0, 2, 1)), R_obs)  # R_plan^T * R_obs
            rot_err = np.linalg.norm(Rotation.from_matrix(R_rel).as_rotvec(), axis=1)
            rot_err_traces.append((t_idx, rot_err))

            # planned step angle (diagnostic vs rot_bound)
            # angle between consecutive planned rotations
            if R_plan.shape[0] >= 2:
                Re = np.einsum("tij,tjk->tik", R_plan[1:], np.transpose(R_plan[:-1], (0, 2, 1)))
                step_ang = np.linalg.norm(Rotation.from_matrix(Re).as_rotvec(), axis=1)
                step_angle_traces.append((t_idx[1:], step_ang))

        # === Plotting ===
        DOF_NAME = {
            "pos": {0: "x", 1: "y", 2: "z"},
            "rot": {0: "roll", 1: "pitch", 2: "yaw"},
            "gripper": {0: "gripper"},
        }

        # panels: pos(3) + rot euler(3) + pos error(1) + rot geodesic error(1) + planned step angle(1) + gripper(1)
        n_panels = 3 + 3 + 1 + 1 + 1 + 1
        fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3 * n_panels), sharex=True)
        axes = np.atleast_1d(axes)
        ax_i = 0

        # --- position ---
        for j, name in DOF_NAME["pos"].items():
            ax = axes[ax_i]
            ax.plot(np.arange(T), obs_pos[:, j], label="obs", color="blue")
            first = True
            for t_idx, p_plan in plan_pos_traces:
                ax.plot(t_idx, p_plan[:, j], label=("plan" if first else "_nolegend_"), color="red", alpha=0.5)
                first = False
            ax.set_title(f"pos/{name}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax_i += 1

        # --- rotation (Euler XYZ) ---
        # unwrap obs and plan separately to avoid jump illusions
        for j, name in DOF_NAME["rot"].items():
            ax = axes[ax_i]
            ax.plot(np.arange(T), obs_euler[:, j], label="obs", color="blue")
            first = True
            for t_idx, eul_plan in plan_eul_traces:
                ax.plot(t_idx, eul_plan[:, j], label=("plan" if first else "_nolegend_"), color="red", alpha=0.5)
                first = False
            ax.set_title(f"rot/{name} (Euler XYZ)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax_i += 1

        # --- position error norm ---
        ax = axes[ax_i]
        first = True
        for t_idx, pos_err in pos_err_traces:
            ax.plot(t_idx, pos_err, label=("||p_obs - p_plan||" if first else "_nolegend_"), color="black")
            first = False
        ax.set_title("position error (meters)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax_i += 1

        # --- rotation geodesic error ---
        ax = axes[ax_i]
        first = True
        for t_idx, rot_err in rot_err_traces:
            ax.plot(t_idx, rot_err, label=("angle(R_plan^T R_obs))" if first else "_nolegend_"), color="purple")
            first = False
        ax.set_title("rotation geodesic error (radians)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax_i += 1

        # --- planned step angle vs rot_bound (diagnostic) ---
        ax = axes[ax_i]
        first = True
        for t_idx, step_ang in step_angle_traces:
            ax.plot(t_idx, step_ang, label=("planned step angle" if first else "_nolegend_"), alpha=0.6)
            first = False
        ax.axhline(float(rot_bound), color="gray", linestyle="--", label="rot_bound")
        ax.set_title("planned step angle vs controller bound")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax_i += 1

        # --- gripper (unchanged) ---
        ax = axes[ax_i]
        ax.plot(np.arange(T), obs_grip, label="obs", color="blue")
        first = True
        for rec in self._viz_records:
            y = rec["pred_gripper"].reshape(-1)
            t0 = rec["start_t"] + 1
            t = np.arange(t0, t0 + y.shape[0])
            ax.plot(t, y, label=("plan" if first else "_nolegend_"), color="red", alpha=0.5)
            first = False
        ax.set_title("gripper")
        ax.grid(True, alpha=0.3)
        ax.legend()

        for ax in axes:
            ax.set_xlabel("sim step")
            ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
