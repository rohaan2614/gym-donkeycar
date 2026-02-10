"""
file: ppo_train.py
author: Tawn Kramer
date: 13 October 2018
notes: ppo2 test from stable-baselines here:
https://github.com/hill-a/stable-baselines

Changes by Rohaan:
- Register Donkey Envs in Gym
- Shimmy Installation (requirements.txt)
- Parser Args for training/testing timesteps
- Tensorboard
- Callback & Evaluation Frequency


Do not forgot to run tensorboard like:
tensorboard --logdir runs --host 127.0.0.1 --port 6006
"""

# TODO: Handle this issue: "UserWarning: Training and eval env are not of the same type<stable_baselines3.common.vec_env.vec_transpose.VecTransposeImage object at 0x78a4a1a9f450> != <stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv object at 0x78a47ae3d7d0> warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")"

# import socket
# socket.setdefaulttimeout(10.0) # socket timeout when Donkey Simulator stops responding


import argparse
import os
import uuid
from datetime import datetime

import gym
import gym_donkeycar  # registers donkey envs into gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback, CallbackList, CheckpointCallback

from funny_helpers import (extract_cte, CTETrainingLogger, EpisodeRewardLogger, 
                           ActionStatsCallback, find_model_dir, resolve_model_path_from_run_dir)
from names_generator import generate_name

import traceback


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

    parser = argparse.ArgumentParser(description="ppo_train")
    parser.add_argument(
        "--sim",
        type=str,
        default="sim_path",
        help="path to unity simulator. maybe be left at manual if you would like to start the sim on your own.",
    )
    parser.add_argument("--port", type=int, default=9091, help="port to use for tcp")
    parser.add_argument("--test", action="store_true", help="load the trained model and play")
    parser.add_argument("--multi", action="store_true", help="start multiple sims at once")
    parser.add_argument(
        "--env_name",
        type=str,
        default="donkey-warehouse-v0",
        help="name of donkey sim environment",
        choices=env_list,
    )
    parser.add_argument(
        "--training_timesteps",
        type=int,
        default=50000,
        help="number of timesteps the agent interacts with the environment in training",
    )
    parser.add_argument(
        "--eval-timesteps",
        type=int,
        default=1000,
        help="(used only in --test mode) number of timesteps to roll out deterministically",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=50000,
        help="run evaluation every N training timesteps",
    )
    parser.add_argument(
        "--n_evaluation_episodes",
        type=int,
        default=1,
        help="run evaluation for N episodes",
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=1000.0,
        help="threshold for early stopping",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a trained PPO model (.zip). Required in --test mode.",
    )
    parser.add_argument("--max_cte", type=int, default=10, help="maximum cross-track error to fail the episode")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name, otherwise auto-generated.",
    )
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping.")

    parser.add_argument(
        "--model-nick",
        type=str,
        default=None,
        help="Optional instead of model path, provide the shorter model-nick.",
    )

    args = parser.parse_args()

    if args.sim == "sim_path" and args.multi:
        print("you must supply the sim path with --sim when running multiple environments")
        raise SystemExit(1)

    env_id = args.env_name

    run_name = args.run_name or generate_name()

    conf = {
        "exe_path": args.sim,
        "host": "127.0.0.1",
        "port": args.port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "font_size": 100,
        "racer_name": f"{run_name}_PPO",
        "country": "USA",
        "bio": "Learning to drive 1 step at a time",
        "guid": str(uuid.uuid4()),
        "max_cte": args.max_cte,
    }

    training_timesteps = args.training_timesteps
    eval_timesteps = args.eval_timesteps
    eval_freq = args.eval_freq
    n_eval_episodes = args.n_evaluation_episodes

    EARLY_STOPPING_THRESHOLD = float(args.early_stopping_threshold)

    if args.test:
        if args.model_path is None:
            if args.model_nick:
                model_dir = find_model_dir(keyword=args.model_nick)
                if not model_dir:
                    raise ValueError(f"Could not find a run directory for nick: {args.model_nick}")
                else:
                    model_path = resolve_model_path_from_run_dir(model_dir)
                    if not model_path:
                        raise ValueError(f"Found run dir but no model zip inside: {model_dir}")
            else:
                raise ValueError("--test requires --model-path (e.g. runs/.../best_model.zip)")
        else:
            model_path = args.model_path
        
        if args.model_nick:
            conf["car_name"] = args.model_nick
        else:
            conf["car_name"] = f'TEST'
        
        env = gym.make(args.env_name, conf=conf)
        try:
            model = PPO.load(model_path)

            obs = env.reset()
            for _ in range(eval_timesteps):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)

                cte = extract_cte(info)
                if (_ % 20 == 0) and (cte is not None):
                    print(f"[TEST t={_:06d}] cte={cte:+.3f}, steer={action[0]}, thr={action[1]}, reward={reward:.3f}")

                # env.render() # Donkey Sim is already rendering so this is likely redundant.
                if done:
                    obs = env.reset()
        finally:
            env.close()

    else:
        if int(eval_freq) > -1:
            error_message = 'eval freq is problematic and needs to be fixed. Please use eval_freq = -1'
            print(error_message)
            raise(error_message)
        
        conf["car_name"] = run_name

        run_id = f"env_{env_id}/train_{training_timesteps}/{datetime.now().strftime('%Y%m%d_%H%M%S')}/{run_name}"
        log_dir = os.path.join("runs", run_id)
        os.makedirs(log_dir, exist_ok=True)

        env = gym.make(args.env_name, conf=conf)

        stop_callback = None

        try:
            model = PPO(
                policy="CnnPolicy",
                env=env,
                verbose=1,
                tensorboard_log=log_dir,
            )

            cte_cb = CTETrainingLogger(tb_every_steps=50, print_every_steps=500, verbose=0)
            ep_reward_logger = EpisodeRewardLogger()
            act_cb = ActionStatsCallback(print_every_steps=100)

            if not args.no_early_stop:
                stop_callback = StopTrainingOnRewardThreshold(reward_threshold=EARLY_STOPPING_THRESHOLD, verbose=1)

            eval_callback = EvalCallback(
                env,  # same env to avoid second sim
                callback_after_eval=stop_callback,
                best_model_save_path=log_dir,
                log_path=log_dir,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False,  # Donkey Sim is already rendering so rendering here is likely redundant.
            )

            checkpoint_callback = CheckpointCallback(save_freq=20_000,
                                                     save_path=log_dir,
                                                     name_prefix="training_checkpoint",
                                                     # PPO does not use replay buffer
                                                     save_replay_buffer=False,
                                                     save_vecnormalize=False)

            callback = CallbackList([cte_cb, eval_callback, ep_reward_logger, checkpoint_callback, act_cb])
            crash_path = os.path.join(log_dir, "crash_checkpoint")
            final_path = os.path.join(log_dir, "final")

            try:
                model.learn(
                    total_timesteps=training_timesteps,
                    tb_log_name="PPO",
                    callback=callback,
                )
                print(f'training completed...')
            except Exception as e:
                print("\n[TRAIN] CRASHED:", repr(e))
                traceback.print_exc()
                # last-ditch save so we don't lose progress
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
