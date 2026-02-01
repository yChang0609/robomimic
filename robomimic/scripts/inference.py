import os
import sys
import argparse
import h5py
import json
import imageio
import numpy as np
import torch
from collections import deque

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.log_utils as LogUtils
from robomimic.algo import RolloutPolicy
from robomimic.utils.log_utils import PrintLogger
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper

def run_rollout(solvers, env, horizon, initial_state=None, render=False, video_writer=None, video_skip=5, camera_names=None):
    num_subtasks = len(solvers)
    is_multi_task = (num_subtasks > 1)     

    for key in solvers.keys():
        solvers[key][0].start_episode()
    
    if initial_state is not None:
        obs = env.reset_to({"states": initial_state})
    else:
        env.reset()
        state_dict = env.get_state()
        obs = env.reset_to(state_dict)

    video_count = 0
    total_reward = 0.
    success = False
    current_subtask_id = 0

    if not is_multi_task:
        single_policy = solvers["task_0"][0]

    try:
        for step_i in range(horizon):
            current_policy = solvers[f"task_{current_subtask_id}"][0] if is_multi_task else single_policy
            act = current_policy(ob=obs)
            obs, r, done, info = env.step(act)
            total_reward += r

            if is_multi_task:
                if check_subtask_finished(env, current_subtask_id):
                    if current_subtask_id < num_subtasks - 1:
                        current_subtask_id += 1
                    else:
                        success = env.is_success()["task"]
                        if success: break
            else:
                success = env.is_success()["task"]
                if success: break

            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = [env.render(mode="rgb_array", height=512, width=512, camera_name=cam) for cam in camera_names]
                    video_img = np.concatenate(video_img, axis=1)
                    video_writer.append_data(video_img)
                video_count += 1

            if done: break
    except Exception as e:
        print(f"WARNING: rollout exception {e}")

    return {
        "Return": total_reward, 
        "Horizon": (step_i + 1), 
        "Success_Rate": float(success)
    }

def check_subtask_finished(env, subtask_id):
    base_env = env.env.env
    if subtask_id == 0:
        nut_idx = base_env.nut_id
        is_grasped = base_env._check_grasp(base_env.robots[0].gripper, base_env.nuts[nut_idx])
        nut_pos = base_env.sim.data.body_xpos[base_env.obj_body_id[base_env.nuts[nut_idx].name]]
        is_lifted = nut_pos[2] > (base_env.table_offset[2] + 0.1)
        return is_grasped and is_lifted
    elif subtask_id == 1:
        return env.is_success()["task"]
    return False

def inference(args):
    # Inferences from failed cases or new environments
    assert (args.env is not None) or (args.failed_hdf5 is not None), "Need give failed case or env"

    if args.output_to_txt_path is not None:
        # log stdout and stderr to a text file
        log_name = 'env_test.txt' if args.env else 'failed_case_test.txt'
        os.makedirs(args.output_to_txt_path, exist_ok=True)
        sys.stdout = sys.stderr = PrintLogger(os.path.join(args.output_to_txt_path, log_name))

    write_video = (args.video_path is not None)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True) 

    # Load models for each
    task_solver = {f"task_{i}": FileUtils.policy_from_checkpoint(ckpt_path=p, device=device, verbose=True) for i, p in enumerate(args.agents)}
        
    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # maybe create video writer
    video_writer = None
    if write_video:
        os.makedirs(os.path.dirname(args.video_path), exist_ok=True)
        video_writer = imageio.get_writer(args.video_path, fps=20) 
    # create environment from saved checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=task_solver[f"task_0"][1], 
        env_name=args.env, 
        render=args.render, 
        render_offscreen=(args.video_path is not None), 
        verbose=True,
    )
    # Env testing
    if args.env:
        print(f"Starting standard env rollout: {args.env}")
        test_keys = range(args.n_rollouts)
        failed_file = None
    else:
        print(f"Evaluating failed cases from: {args.failed_hdf5}")
        failed_file = h5py.File(args.failed_hdf5, "r")
        test_keys = list(failed_file["data"].keys())
        if args.n_rollouts: test_keys = test_keys[:args.n_rollouts]

    results = []
    pbar = LogUtils.custom_tqdm(test_keys)
    for key in pbar:
        initial_state = failed_file[f"data/{key}/states"][0] if failed_file else None
        stats = run_rollout(
            task_solver, env, 
            args.horizon, initial_state, 
            args.render, video_writer, 
            args.video_skip, args.camera_names)
        results.append(stats)
        success_count = sum([r["Success_Rate"] for r in results])
        pbar.set_postfix(Success=f"{int(success_count)}/{len(results)}", Rate=f"{(success_count/len(results))*100:.1f}%")

    final_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(results)
    summary = {
        "Exp_Metadata": {
            "Type": "Standard_Env" if args.env else "Failed_Cases",
            "Model_Count": len(task_solver),
            "Model_Paths": args.agents
        },
        "Statistics": {
            "Total_Cases": len(results),
            "Success_Count": int(np.sum(final_stats["Success_Rate"])),
            "Avg_Success_Rate": float(np.mean(final_stats["Success_Rate"])),
            "Avg_Return": float(np.mean(final_stats["Return"])),
            "Avg_Horizon": float(np.mean(final_stats["Horizon"]))
        }
    }

    print("\n" + "="*30 + "\nFinal Experiment Summary\n" + "="*30)
    print(json.dumps(summary, indent=4))
    
    if failed_file: failed_file.close()
    if video_writer: video_writer.close()
    env.env.env.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained policy on failed MimicGen environments.")

    parser.add_argument("--agents", type=str, nargs='+', required=True, help="Path to your trained .pth checkpoint")
    parser.add_argument("--failed_hdf5", type=str, default=None, help="Path to the failed_demo.hdf5 file")
    parser.add_argument("--env", type=str, default=None, help="Testing env name")
    parser.add_argument("--n_rollouts", type=int, default=None, help="Number of failed cases to test (default: all)")
    parser.add_argument("--horizon", type=int, required=True, help="Maximum horizon")
    parser.add_argument("--render", action='store_true', help="On-screen rendering")
    parser.add_argument("--video_path", type=str, default=None, help="Path to save output video")
    parser.add_argument("--video_skip", type=int, default=5, help="Render every n steps")
    parser.add_argument("--camera_names", type=str, nargs='+', default=["agentview"], help="Cameras for rendering")
    parser.add_argument("--output_to_txt_path", type=str, default=None, help="Output the log into txt")
    parser.add_argument("--seed",type=int,default=None,help="(optional) set seed for rollouts")

    args = parser.parse_args()
    inference(args)