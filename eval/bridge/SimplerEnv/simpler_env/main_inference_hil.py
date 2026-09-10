import os

import numpy as np
from simpler_env.policies.vam.video_action_model import VAMInference

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_hil_evaluator import maniskill2_hil_evaluator


if __name__ == "__main__":
    args = get_args()

    os.environ["DISPLAY"] = ""

    model = VAMInference(
        args.vam_experiment_name,
        args.vam_video_model_path,
        args.vam_action_model_path,
        args.vam_dataset_statistics_path,
        args.vam_img_horizon,
        args.vam_lowdim_horizon,
        args.vam_stop_video_denoising_step,
        args.vam_num_execute_actions,
        is_hil=True,
        device=args.device,
    )
    success_arr = maniskill2_hil_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
