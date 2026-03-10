"""
The main entry point for training policies.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    debug (bool): set this flag to run a quick training run for debugging purposes
"""

import argparse
import json
import traceback

import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.scripts.training.il_train import il_train
from robomimic.scripts.training.rl_train import rl_train


def main(args):

    if args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)
    
    if args.dataset is not None:
        config.train.data = [{"path": args.dataset}]

    if args.name is not None:
        config.experiment.name = args.name

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # maybe modify config for debugging purposes
    if args.debug:
        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        # train and validate (if enabled) for 3 gradient steps, for 2 epochs
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2

        # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10

        # send output to a temporary directory
        config.train.output_dir = "/tmp/tmp_trained_models"

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()
    
    if "training_mode" not in config.train:
        with config.values_unlocked():
            config.train.training_mode = "IL"

    train_func = il_train if config.train.training_mode == "IL" else rl_train        
    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        train_func(config, device=device, resume=args.resume, auto_remove_exp_dir=args.auto_remove_exp)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    # Algorithm Name
    parser.add_argument(
        "--algo",
        type=str,
        help="(optional) name of algorithm to run. Only needs to be provided if --config is not provided",
    )

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    # debug mode
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes"
    )

    # resume training from latest checkpoint
    parser.add_argument(
        "--resume",
        action='store_true',
        help="set this flag to resume training from latest checkpoint",
    )

    # 
    parser.add_argument(
        "--auto-remove-exp",
        action='store_true',
        help="force delete the experiment folder if it exists"
    )

    args = parser.parse_args()
    main(args)
