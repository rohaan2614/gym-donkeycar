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

import argparse
import os
import uuid
from datetime import datetime

import gym
import gym_donkeycar  # registers donkey envs into gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback, CallbackList

from funny_helpers import extract_cte, CTETrainingLogger

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
        "--evaluation_timesteps",
        type=int,
        default=1000,
        help="(used only in --test mode) number of timesteps to roll out deterministically",
    )
    parser.add_argument(
        "--evaluation_frequency",
        type=int,
        default=10000,
        help="run evaluation every N training timesteps",
    )
    parser.add_argument(
        "--n_evaluation_episodes",
        type=int,
        default=5,
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

    args = parser.parse_args()

    if args.sim == "sim_path" and args.multi:
        print("you must supply the sim path with --sim when running multiple environments")
        raise SystemExit(1)

    env_id = args.env_name

    conf = {
        "exe_path": args.sim,
        "host": "127.0.0.1",
        "port": args.port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "me",
        "font_size": 100,
        "racer_name": "PPO",
        "country": "USA",
        "bio": "Learning to drive w PPO RL",
        "guid": str(uuid.uuid4()),
        "max_cte": args.max_cte,
    }

    training_timesteps = args.training_timesteps
    evaluation_timesteps = args.evaluation_timesteps
    eval_freq = args.evaluation_frequency
    n_eval_episodes = args.n_evaluation_episodes

    EARLY_STOPPING_THRESHOLD = float(args.early_stopping_threshold)

    if args.test:
        if args.model_path is None:
            raise ValueError("--test requires --model-path (e.g. runs/.../best_model.zip)")
        env = gym.make(args.env_name, conf=conf)
        try:
            model = PPO.load(args.model_path)

            obs = env.reset()
            for _ in range(evaluation_timesteps):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)

                cte = extract_cte(info)
                if (_ % 20 == 0) and (cte is not None):
                    print(f"[TEST t={_:06d}] cte={cte:+.3f} reward={reward:.3f}")

                env.render()
                if done:
                    obs = env.reset()
        finally:
            env.close()

    else:
        run_id = (
            f"env_{env_id}/max_cte_{args.max_cte}/train_{training_timesteps}/" f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        log_dir = os.path.join("runs", run_id)
        os.makedirs(log_dir, exist_ok=True)

        env = gym.make(args.env_name, conf=conf)

        try:
            model = PPO(
                policy="CnnPolicy",
                env=env,
                verbose=1,
                tensorboard_log=log_dir,
            )

            stop_callback = StopTrainingOnRewardThreshold(reward_threshold=EARLY_STOPPING_THRESHOLD, verbose=1)

            cte_cb = CTETrainingLogger(tb_every_steps=50, print_every_steps=500, verbose=0)
            
            eval_callback = EvalCallback(
                env,  # same env to avoid second sim
                callback_after_eval=stop_callback,
                best_model_save_path=log_dir,
                log_path=log_dir,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=True,
            )

            callback = CallbackList([cte_cb, eval_callback])

            model.learn(
                total_timesteps=training_timesteps,
                tb_log_name="PPO",
                callback=callback,
            )

            # Save the agent (keeps your original behavior)
            model.save("ppo_donkey")
            # Optional: also save into the run directory so each run keeps its model
            model.save(os.path.join(log_dir, "ppo_donkey"))

        finally:
            # Close envs cleanly
            try:
                env.close()
            except Exception:
                pass

