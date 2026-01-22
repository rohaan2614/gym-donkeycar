import os
import gym
import gym_donkeycar
import numpy as np


from rohaans_funny_functions import is_port_in_use

# Set up env
exe_path = f'/home/rn7823/projects/DonkeySimLinux/donkey_sim.x86_64'
port = 9091



if (is_port_in_use(port=9091)):
    print(f'Port {port} is in use. Simulation is probably already running. Connecting...')

    env = gym.make("donkey-warren-track-v0")

    obs = env.reset()
    try:
        for _ in range(100):
            # drive straight with small speed
            action = np.array([0.0, 0.5])  
            # execute the action
            obs, reward, done, info = env.step(action)
    except KeyboardInterrupt:
        # You can kill the program using ctrl+c
        pass

else:
    print(f'Opening Simulation on port {port}...')
    conf = {
        'exe_path' : exe_path,
        'port' : port
    }

    env = gym.make('donkey-generated-track-v0',
               conf=conf)
    
    # PLAY
    obs = env.reset()



for t in range(100):
    action = np.array([0.0, 0.5])
    # execute action
    obs, reward, done, info = env.step(action)
env.close()