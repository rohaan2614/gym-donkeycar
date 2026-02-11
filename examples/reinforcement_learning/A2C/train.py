# TODO: populate this
"""
Do not forgot to run tensorboard like:
tensorboard --logdir runs --host 127.0.0.1 --port 6006
"""

import argparse
import os
import uuid
from datetime import datetime

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Change these env variables to reduce warning clutter
new_env_vars = {"TF_ENABLE_ONEDNN_OPTS": "0"}
# change env var
for k,v in new_env_vars.items():
    os.environ[k] = v
# confirm changes
for k,v in new_env_vars.items():
    if os.getenv(k) != v:
        print('Env Var Change Failed!', k)


import gym
import gym_donkeycar  # registers donkey envs into gym
# from sb3_contrib import A2
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback, CallbackList, CheckpointCallback

from funny_helpers import CTETrainingLogger, EpisodeRewardLogger, ActionStatsCallback
from names_generator import generate_name

import traceback

import imageio
import json


if __name__ == "__main__":
    env_list = [
        "donkey-warehouse-v0",
        "donkey-generated-roads-v0",
        "donkey-avc-sparkfun-v0",
        "donkey-generated-track-v0",
        "donkey-roboracingleague-track-v0",
        "donkey-waveshare-v0",
        "donkey-minimonaco-track-v0",
        "donkey-warren-track-v0",
        "donkey-thunderhill-track-v0",
        "donkey-circuit-launch-track-v0",
    ]
    
    model_name = 'a2c'
    parser = argparse.ArgumentParser(description=f"{model_name}_train")

    #####################
    parser.add_argument("--sim", type=str, default="/home/rn7823/projects/DonkeySimLinux/donkey_sim.x86_64",
        help="path to unity simulator. maybe be left at manual if you would like to start the sim on your own.")
    parser.add_argument("--port", type=int, default=9091, help="port to use for tcp")
    #####################

    #####################
    #TODO: fix/build this
    # parser.add_argument("--multi", action="store_true", help="start multiple sims at once")
    #####################

    #####################
    parser.add_argument("--env_name", type=str, default="donkey-warren-track-v0", help="name of donkey sim environment",
    choices=env_list)
    
    #####################
    parser.add_argument("-t", "--timesteps", type=int, default=50000,
        help="number of timesteps the agent interacts with the environment in training" )
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate of the model')
    #####################
    
    #####################
    parser.add_argument("--eval_freq", type=int, default=-1, help="run evaluation every N training timesteps")
    parser.add_argument("--n_evaluation_episodes", type=int, default=1, help="run evaluation for N episodes")
    #####################
    
    #####################
    parser.add_argument("--early_stopping_threshold", type=float, default=1000.0, help="threshold for early stopping")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping.")
    #####################

    #####################
    parser.add_argument("--max_cte", type=int, default=10, help="maximum cross-track error to fail the episode")
    #####################

    #####################
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name, otherwise auto-generated.")
    #####################
    
    #####################
    parser.add_argument("--save-metadata", dest="save_metadata", action="store_true", default=True,
                    help="Save image frames + JSON metadata (default: on)")
    parser.add_argument("--disable-metadata-saving", dest="save_metadata", action="store_false",
                        help="Disable saving image frames + JSON metadata")
    #####################

    #####################
    parser.add_argument("--force-cpu", dest="force_cpu_training", action="store_true", default=True,
                    help="Force to train on CPU (a2c is intended for cpu trainging)")
    #####################


    args = parser.parse_args()

    # Constants
    training_timesteps = args.timesteps
    learning_rate = args.lr
    env_id = args.env_name
    run_name = args.run_name or generate_name()
    # TODO: evaluate after every few training runs (fix eval_freq & n_eval_episodes)
    eval_freq = args.eval_freq
    n_eval_episodes = args.n_evaluation_episodes
    stop_early_at = float(args.early_stopping_threshold)

    conf = {
        "exe_path": args.sim,
        "host": "127.0.0.1",
        "port": args.port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "font_size": 100,
        "racer_name": f"{run_name}_{model_name}",
        "guid": str(uuid.uuid4()),
        "max_cte": args.max_cte,
    }

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>> TRAIN MODE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    if int(eval_freq) > -1:
        error_message = 'eval freq is problematic and needs to be fixed. Please use eval_freq = -1'
        print(error_message)
        raise(error_message)
    
    conf["car_name"] = f'{model_name}_{run_name}'

    run_id = f"env_{env_id}/model_{model_name}/train_{training_timesteps}-{datetime.now().strftime('%Y%m%d_%H%M%S')}/{run_name}"
    log_dir = os.path.join("runs", run_id)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make(args.env_name, conf=conf)
    print('Environment created...')

    stop_callback = None

    try:
        model = A2C(policy='CnnPolicy', #Mlp policy is for vector inputs
                    env=env,
                    verbose=1,
                    device='cpu' if args.force_cpu_training else 'auto')

        cte_cb = CTETrainingLogger(tb_every_steps=50, print_every_steps=500, verbose=0)
        ep_reward_logger = EpisodeRewardLogger()
        act_cb = ActionStatsCallback(print_every_steps=100)

         # TODO: evaluate after every few training runs (fix eval_freq & n_eval_episodes)

        checkpoint_callback = CheckpointCallback(save_freq=20_000,
                                                 save_path=log_dir,
                                                 name_prefix="training_checkpoint",
                                                 verbose=1)

        callback = CallbackList([cte_cb, ep_reward_logger, checkpoint_callback, act_cb])
        crash_path = os.path.join(log_dir, "crash_checkpoint")
        final_path = os.path.join(log_dir, "final")

        try:
            # model.train(+)
            model.learn(total_timesteps=training_timesteps,
                        callback=callback,
                        log_interval=100,
                        tb_log_name=f'A2C_{run_name}')
            print(f'[TRAIN] training completed...')
        except Exception as e:
            print("\n[TRAIN] CRASHED:", repr(e))
            traceback.print_exc()
            
            # last-ditch save effort
            try:
                model.save(crash_path)
                print(f'Model Saved at: {crash_path}.zip')
            except Exception as e2:
                print('Failed to save model because of exception:', str(e2))
            raise
        else:
            model.save(final_path)
            print(f'Model (final) Saved at: {final_path}.zip')

    finally:
        # Close envs cleanly
        try:
            print(f'Process Completed. Closing Env.')
            env.close()
        except Exception:
            pass

