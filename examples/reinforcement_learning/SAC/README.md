# Reinforcement Learning

## ppo_train.py
* To train, run: 

    `(donkey) $ python examples/reinforcement_learning/A2C/train.py -t 1_000_000`

* To run the trained agent, run: 
    
    `(donkey) $ python examples/reinforcement_learning/A2C/test.py --model-path runs/env_donkey-warren-track-v0/model_a2c/train_50000-20260211_144544/hardcore_solomon/training_checkpoint_40000_steps.zip`


* Can also use model nick if you don't want to ype out the full model path:

    `$ python examples/reinforcement_learning/A2C/test.py --model-nick hardcore_solomon`


# Tensorboard Server

`tensorboard --logdir runs/`
<!-- ## ppo_train.py

An example using stable-baselines to train a PPO agent using the gym-donkeycar environment

* follow [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) install
* ```python gym-donkeycar/examples/reinforcement_learning/ppo_train.py --sim <path to simulator>```

## ddqn.py

An example training a [deep double Q-learning](https://arxiv.org/abs/1509.06461) agent using the gym-donkeycar environment

* ```python gym-donkeycar/examples/reinforcement_learning/ddqn.py --sim <path to simulator>``` -->
