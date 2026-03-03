import os
from collections import OrderedDict

import numpy as np
import torch

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.algo import algo_factory
from robomimic.utils.log_utils import flush_warnings
from robomimic.utils.python_utils import deep_update

def set_seed(config):
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

def print_config(config):
    print("\n============= New Training Run with Config =============")
    print(config)
    print("")

def create_envs_from_dataset(config):
    envs = OrderedDict()
    env_meta_list = []
    shape_meta_list = []

    if isinstance(config.train.data, str):
        with config.values_unlocked():
            config.train.data = [{"path": config.train.data}]
            
    if config.train.data is not None:
        for dataset_cfg in config.train.data:
            dataset_path = os.path.expanduser(dataset_cfg["path"])
            if not os.path.exists(dataset_path):
                raise Exception("Dataset at provided path {} not found!".format(dataset_path))
            env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
            env_meta["lang"] = dataset_cfg.get("lang", "dummy")
            deep_update(env_meta, config.experiment.env_meta_update_dict)
            env_meta_list.append(env_meta)

            shape_meta = FileUtils.get_shape_metadata_from_dataset(
                dataset_config=dataset_cfg,
                action_keys=config.train.action_keys,
                all_obs_keys=config.all_obs_keys
            )
            shape_meta_list.append(shape_meta)

    for env_i in range(len(env_meta_list)):
        dataset_cfg = config.train.data[env_i]
        env_meta = env_meta_list[env_i]
        shape_meta = shape_meta_list[env_i]
        env_names = [env_meta["env_name"]]
        if (env_i == 0) and (config.experiment.additional_envs is not None):
            for name in config.experiment.additional_envs:
                env_names.append(name)
        for env_name in env_names:
            env = create_env(config, env_name, env_meta, shape_meta)
            env_key = os.path.splitext(os.path.basename(dataset_cfg["path"]))[0] if not dataset_cfg.get("key", None) else dataset_cfg["key"]
            envs[env_key] = env
            print(env)
    return envs, env_meta_list, shape_meta_list

def _dataset_cfg_to_env_key(dataset_cfg):
    return os.path.splitext(os.path.basename(dataset_cfg["path"]))[0] if not dataset_cfg.get("key", None) else dataset_cfg["key"]


def _extract_demo_init_state(demo_group):
    if "states" not in demo_group:
        return None

    states = demo_group["states"]
    try:
        first_state = states[0]
    except Exception:
        # Some datasets store states as an HDF5 group of per-field arrays.
        try:
            first_state = {k: states[k][0] for k in states.keys()}
        except Exception:
            return None

    init_state = {"states": first_state}
    if "model_file" in demo_group.attrs:
        init_state["model"] = demo_group.attrs["model_file"]
    if "ep_meta" in demo_group.attrs:
        init_state["ep_meta"] = demo_group.attrs["ep_meta"]
    return init_state


def build_rollout_init_states_from_trainset(config, trainset):
    """
    Build rollout init-state pools per env key from the train dataset(s).
    """
    rollout_init_states = OrderedDict()
    if config.train.data is None:
        return rollout_init_states

    datasets = trainset.datasets if hasattr(trainset, "datasets") else [trainset]
    if len(datasets) != len(config.train.data):
        print(
            "WARNING: dataset count mismatch between trainset ({}) and config.train.data ({}). "
            "Using the first {} entries.".format(
                len(datasets), len(config.train.data), min(len(datasets), len(config.train.data))
            )
        )

    num_entries = min(len(datasets), len(config.train.data))
    for ds_idx in range(num_entries):
        dataset_cfg = config.train.data[ds_idx]
        env_key = _dataset_cfg_to_env_key(dataset_cfg)
        dataset = datasets[ds_idx]
        rollout_init_states[env_key] = []

        if (not hasattr(dataset, "demos")) or (not hasattr(dataset, "hdf5_file_opened")):
            print(
                "WARNING: dataset for env_key '{}' does not expose demos / hdf5 access. "
                "Skipping init-state extraction.".format(env_key)
            )
            continue

        with dataset.hdf5_file_opened() as hdf5_file:
            for demo_id in dataset.demos:
                demo_path = "data/{}".format(demo_id)
                if demo_path not in hdf5_file:
                    print("WARNING: missing '{}' in dataset for env_key '{}', skipping.".format(demo_path, env_key))
                    continue

                init_state = _extract_demo_init_state(hdf5_file[demo_path])
                if init_state is None:
                    print(
                        "WARNING: demo '{}' for env_key '{}' has no valid states[0], skipping.".format(
                            demo_id, env_key
                        )
                    )
                    continue
                rollout_init_states[env_key].append(init_state)

        if len(rollout_init_states[env_key]) == 0:
            print(
                "WARNING: env_key '{}' has no valid init states. Rollouts will fallback to env.reset().".format(
                    env_key
                )
            )
        else:
            print(
                "Loaded {} rollout init states for env_key '{}'.".format(
                    len(rollout_init_states[env_key]), env_key
                )
            )

    return rollout_init_states

# create environment for each env_name
def create_env(config, env_name, env_meta, shape_meta):
    env_kwargs = dict(
        env_meta=env_meta,
        env_name=env_name,
        render=False,
        render_offscreen=config.experiment.render_video,
        use_image_obs=shape_meta["use_images"] or shape_meta["use_depths"],
    )
    env = EnvUtils.create_env_from_metadata(**env_kwargs)
    # handle environment wrappers
    env = EnvUtils.wrap_env_from_config(env, config=config)  # apply environment warpper, if applicable
    return env

def create_model(config, shape_meta, device, resume=None):
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device
    )
    
    if resume is not None:
        # load ckpt dict
        print("*" * 50)
        print("resuming from ckpt at {}".format(resume["latest_model_path"]))
        try:
            ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=resume["latest_model_path"])
        except Exception as e:
            print("got error: {} when loading from {}".format(e, resume["latest_model_path"]))
            print("trying backup path {}".format(resume["latest_model_backup_path"]))
            ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=resume["latest_model_backup_path"])
        # load model weights and optimizer state
        model.deserialize(ckpt_dict["model"], load_optimizers=True)
        print("*" * 50)

    # if checkpoint is specified, load in model weights;
    # will not use ckpt_path if resuming training
    ckpt_path = config.experiment.ckpt_path
    if (ckpt_path is not None) and (not resume):
        print("LOADING MODEL WEIGHTS FROM " + ckpt_path)
        from robomimic.utils.file_utils import maybe_dict_from_checkpoint
        ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
        model.deserialize(ckpt_dict["model"])

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # print all warnings before training begins
    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
    flush_warnings()
    print("*" * 50)
    print("")

    return model, ckpt_dict["variable_state"] if resume is not None else None

def env_close(env):
    base_env = env
    while True:
        next_env = getattr(base_env, "env", None)
        if (next_env is None) or (next_env is base_env):
            break
        base_env = next_env
    close_fn = getattr(base_env, "close", None)
    
    # if the base env has a close function, call it. Otherwise, return False to indicate that we were not able to close the env.
    if callable(close_fn):
        close_fn()
    else:
        return False
    
    return True