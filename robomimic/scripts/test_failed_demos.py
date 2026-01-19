import argparse
import json
import h5py
import imageio
import numpy as np
import sys
import os
from copy import deepcopy

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.log_utils as LogUtils
from robomimic.algo import RolloutPolicy
from robomimic.utils.log_utils import PrintLogger

from copy import deepcopy

def rollout_on_failed_state(policy, env, initial_state, horizon, render=False, video_writer=None, video_skip=5, camera_names=None):
    policy.start_episode()
    env.reset()
    obs = env.reset_to({"states": initial_state})

    video_count = 0
    total_reward = 0.
    success = False

    try:
        for step_i in range(horizon):
            act = policy(ob=obs)
            next_obs, r, done, _ = env.step(act)

            total_reward += r
            success = env.is_success()["task"]
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1)
                    video_writer.append_data(video_img)
                video_count += 1

            if done or success:
                break       
            obs = deepcopy(next_obs)
    except env.rollout_exceptions as e:
        print(f"WARNING: got rollout exception {e}")

    stats = dict(
        Return=total_reward, 
        Horizon=(step_i + 1), 
        Success_Rate=float(success))
    return stats

def run_failed_test(args):

    if args.output_to_txt_path is not None:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(args.output_to_txt_path, 'failed_test_log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    write_video = (args.video_path is not None)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True) 

    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=args.agent, device=device, verbose=True)

    print("Loading failed demos from: {}".format(args.failed_hdf5))
    f = h5py.File(args.failed_hdf5, "r")
    demo_keys = list(f["data"].keys())

    if args.n_cases is not None:
        demo_keys = demo_keys[:args.n_cases]

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, 
        env_name=args.env, 
        render=args.render, 
        render_offscreen=write_video, 
        verbose=True,
    )

    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    rollout_stats = []

    print(f"Starting evaluation on {len(demo_keys)} failed cases...")
    pbar = LogUtils.custom_tqdm(demo_keys)
    success_count = 0
    total_tested = 0
    for demo_key in pbar:
        initial_state = f[f"data/{demo_key}/states"][0]
        
        stats = rollout_on_failed_state(
            policy=policy, 
            env=env, 
            initial_state=initial_state,
            horizon=args.horizon,
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip, 
            camera_names=args.camera_names,
        )
        rollout_stats.append(stats)

        total_tested += 1
        if stats["Success_Rate"] > 0:
            success_count += 1

        current_success_rate = (success_count / total_tested) * 100
        pbar.set_postfix(ordered_dict={
                "Success": f"{success_count}/{total_tested}",
                "Rate": f"{current_success_rate:.1f}%"
            }, refresh=True)
    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Total_Cases"] = len(demo_keys)
    avg_rollout_stats["Success_Count"] = int(np.sum(rollout_stats["Success_Rate"]))

    print("\n--- Evaluation Results on Failed MimicGen Envs ---")
    print(json.dumps(avg_rollout_stats, indent=4))

    if write_video:
        video_writer.close()
    f.close()
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained policy on failed MimicGen environments.")

    parser.add_argument("--agent", type=str, required=True, help="Path to your trained .pth checkpoint")
    parser.add_argument("--failed_hdf5", type=str, required=True, help="Path to the failed_demo.hdf5 file")
    parser.add_argument("--n_cases", type=int, default=None, help="Number of failed cases to test (default: all)")
    parser.add_argument("--horizon", type=int, default=None, help="Override maximum horizon")
    parser.add_argument("--env", type=str, default=None, help="Override env name")
    parser.add_argument("--render", action='store_true', help="On-screen rendering")
    parser.add_argument("--video_path", type=str, default=None, help="Path to save output video")
    parser.add_argument("--video_skip", type=int, default=5, help="Render every n steps")
    parser.add_argument("--camera_names", type=str, nargs='+', default=["agentview"], help="Cameras for rendering")
    parser.add_argument("--output_to_txt_path", type=str, default=None, help="Output the log into txt")

    args = parser.parse_args()
    run_failed_test(args)