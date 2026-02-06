# Reinforcement Learning

## ppo_train.py
* To train, run: 

    `(donkey) $ python examples/reinforcement_learning/ppo_train.py --sim /home/rn7823/projects/DonkeySimLinux/donkey_sim.x86_64 --env_name donkey-warren-track-v0 --training_timesteps 100000 --early_stopping_threshold 800`

* To run the trained agent (i.e to exploit), run: 
    
    `(donkey) $ python examples/reinforcement_learning/ppo_train.py --sim /home/rn7823/projects/DonkeySimLinux/donkey_sim.x86_64 --env_name donkey-warren-track-v0 --test --model-path runs/env_donkey-warren-track-v0/max_cte_10/train_100000/20260122_153939/best_model.zip`


# Tensorboard Server

`tensorboard --logdir runs/`
<!-- ## ppo_train.py

An example using stable-baselines to train a PPO agent using the gym-donkeycar environment

* follow [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) install
* ```python gym-donkeycar/examples/reinforcement_learning/ppo_train.py --sim <path to simulator>```

## ddqn.py

An example training a [deep double Q-learning](https://arxiv.org/abs/1509.06461) agent using the gym-donkeycar environment

* ```python gym-donkeycar/examples/reinforcement_learning/ddqn.py --sim <path to simulator>``` -->
