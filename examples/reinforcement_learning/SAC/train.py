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
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback, CallbackList, CheckpointCallback

from funny_helpers import EpisodeRewardLogger
from stats_callback import StatsCallback
from coolname import generate_slug

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
    
    model_name = 'sac'
    parser = argparse.ArgumentParser(description=f"{model_name}_train")

    #####################
    parser.add_argument("--sim", type=str, default="/home/rn7823/projects/DonkeySimLinux/donkey_sim.x86_64",
        help="path to unity simulator. maybe be left at manual if you would like to start the sim on your own.")
    parser.add_argument("--manual-sim", action="store_true", default=False,
        help="Do not auto-launch Unity from Python; assume you started it yourself")
    
    default_port = 9091
    parser.add_argument("-p", "--port", type=int, default=default_port, help="port to use for tcp")
    parser.add_argument("--seed", type=int, default=1947, help="seed for stochasticity")
    #####################

    #####################
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint (without .zip) to resume training from")
    #####################

    #####################
    parser.add_argument("--env_name", type=str, default="donkey-warren-track-v0", help="name of donkey sim environment",
    choices=env_list)
    #####################
    parser.add_argument("-t", "--timesteps", type=int, default=50000,
        help="number of timesteps the agent interacts with the environment in training" )
    #####################
    
    
    #####################
    parser.add_argument("--early_stopping_threshold", type=float, default=1000.0, help="threshold for early stopping")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping.")
    #####################

    #####################
    parser.add_argument("--max-cte", type=int, default=10, help="maximum cross-track error to fail the episode")
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
    parser.add_argument("--force-cpu", dest="force_cpu_training", action="store_true", default=False,
                    help="Force to train on CPU (a2c is intended for cpu trainging)")
    #####################
    
    args = parser.parse_args()

    # -------------------------------------------------
    # Core constants
    # -------------------------------------------------
    training_timesteps = args.timesteps
    seed = args.seed
    env_id = args.env_name

    # -------------------------------------------------
    # Determine run identity (Resume or Fresh)
    # -------------------------------------------------
    if args.resume_from is not None:

        checkpoint_path = args.resume_from

        if not os.path.exists(checkpoint_path + ".zip"):
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}.zip"
            )

        log_dir = os.path.dirname(checkpoint_path)
        run_name = os.path.basename(log_dir)
        run_id = log_dir.replace("runs/", "")
        reset_timesteps = False

        print(f"[INFO] Resuming from: {checkpoint_path}.zip")
        print(f"[INFO] Extracted run_name: {run_name}")
        print(f"[INFO] Reusing log_dir: {log_dir}")

    else:

        run_name = args.run_name or generate_slug(2).split("-")[-1]
        run_id = (
            f"env_{env_id}/model_{model_name}/"
            f"train_{training_timesteps}-"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
            f"{run_name}"
        )

        log_dir = os.path.join("runs", run_id)
        os.makedirs(log_dir, exist_ok=True)
        reset_timesteps = True

        print("[INFO] Training from scratch.")
        print(f"[INFO] New run_id: {run_id}")

    # -------------------------------------------------
    # Environment Configuration
    # -------------------------------------------------
    conf = {
        "host": "127.0.0.1",
        "port": args.port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "font_size": 100,
        "racer_name": f"{run_name}_{model_name}",
        "guid": str(uuid.uuid4()),
        "max_cte": args.max_cte,
    }

    conf["car_name"] = f"{model_name}_{run_name}"
    
    if args.manual_sim:
        conf["exe_path"] = ""
    else:
        conf["exe_path"] = args.sim
        
    # -------------------------------------------------
    # Create Environment
    # -------------------------------------------------
    print('ENV CONF:', conf)
    env = gym.make(env_id, conf=conf)

    print("Environment created.")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # -------------------------------------------------
    # Training
    # -------------------------------------------------
    try:

        if args.resume_from is not None:
            model = SAC.load(
                args.resume_from,
                env=env,
                seed=seed,
                tensorboard_log="runs"
            )
            print("[INFO] Continuing training (reset_num_timesteps=False)")

        else:
            model = SAC(
                policy="CnnPolicy",
                env=env,
                learning_rate=3e-4,
                buffer_size=int(3e5),
                learning_starts=500,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                seed=seed,
                tensorboard_log="runs",
            )

        # Callbacks
        ep_reward_logger = EpisodeRewardLogger()
        stats_callback = StatsCallback(print_every_steps=100)

        checkpoint_callback = CheckpointCallback(
            save_freq=20_000,
            save_path=log_dir,
            name_prefix="training_checkpoint",
            verbose=0
        )

        callback = CallbackList([
            ep_reward_logger,
            checkpoint_callback,
            stats_callback
        ])

        crash_path = os.path.join(log_dir, "crash_checkpoint")
        final_path = os.path.join(log_dir, "final")

        # Train
        model.learn(
            total_timesteps=training_timesteps,
            callback=callback,
            log_interval=100,
            tb_log_name=run_id,
            reset_num_timesteps=reset_timesteps
        )

        model.save(final_path)
        print(f"[TRAIN] Completed. Final model saved at {final_path}.zip")

    except Exception as e:
        print("\n[TRAIN] CRASHED:", repr(e))
        traceback.print_exc()

        try:
            model.save(crash_path)
            print(f"[TRAIN] Crash checkpoint saved at {crash_path}.zip")
        except Exception as e2:
            print("Failed to save crash checkpoint:", str(e2))

        raise

    finally:
        print("Closing environment.")
        env.close()