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

from funny_helpers import (extract_cte, CTETrainingLogger, EpisodeRewardLogger, 
                     ActionStatsCallback, find_model_dir, resolve_model_path_from_run_dir)
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
    parser.add_argument("--model-path", type=str, help="Path to a trained model (.zip).")
    parser.add_argument("--model-nick", type=str, help="Optional; instead of model path, provide the shorter model-nick.")
    #####################

    #####################
    #TODO: fix/build this
    # parser.add_argument("--multi", action="store_true", help="start multiple sims at once")
    #####################

    #####################
    parser.add_argument("--env_name", type=str, default="donkey-warren-track-v0", help="name of donkey sim environment",
                        choices=env_list)
    #####################
    parser.add_argument("-t", "--timesteps", type=int, default=10_000,
                        help="number of timesteps to roll out deterministically" )
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

    # #####################
    # parser.add_argument("--force-cpu", dest="force_cpu_training", action="store_true", default=True,
    #                 help="Force to train on CPU (a2c is intended for cpu trainging)")
    # #####################


    args = parser.parse_args()

    # Constants
    timesteps = args.timesteps
    env_id = args.env_name
    run_name = args.run_name or generate_name()

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
            raise ValueError("--test requires --model-path (e.g. runs/.../best_model.zip) or at least a valid model-nick")
    else:
        model_path = args.model_path
    
    conf = {
        "exe_path": args.sim,
        "host": "127.0.0.1",
        "port": args.port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "font_size": 100,
        "guid": str(uuid.uuid4()),
        "max_cte": args.max_cte,
    }

    if args.model_nick:
        conf["car_name"] = args.model_nick
    else:
        conf["car_name"] = f'TEST_{model_name}'

    env = gym.make(args.env_name, conf=conf)
    print('Environment created...')

    try:
        model = A2C.load(model_path)

        obs = env.reset()
        for _ in range(timesteps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            cte = extract_cte(info)

            if cte is not None:
                print(f"[TEST t={_:06d}] cte={cte:+.3f}, steer={action[0]:.3f}, thr={action[1]:.3f}, reward={reward:.3f}")

            if done:
                print(f"[TEST t={_:06d}] done={done}, hit={info['hit']}, cte_crossed={abs(info['cte']) > args.max_cte}")
                
            # if ((_ % save_frame_every == 0) or (done)) and (save_metadata):
            #     frame_path = os.path.join(frame_root, f'frame_{_:06d}.png')
            #     meta_path = os.path.join(frame_root, f'frame_{_:06d}.json')

            #     imageio.imwrite(frame_path, obs)

            #     meta = {
            #         "timestep": _,
            #         "steer": float(action[0]),
            #         "throttle": float(action[1]),
            #         "reward": float(reward),
            #         "cte": float(cte) if cte is not None else None,
            #         "done": bool(done),
            #         "hit": info.get("hit", None),
            #         "speed": info.get("speed", None),
            #         "cte_crossed": (
            #             cte is not None and abs(cte) > args.max_cte
            #         ),
            #     }

            #     with open(meta_path, "w") as f:
            #         json.dump(meta, f, indent=2)

            if done: obs = env.reset()
            
    finally:
        env.close()