# Reinforcement Learning

## train.py
* To train, run: 

    `(donkey) $ python examples/reinforcement_learning/SAC/train.py -t 2_000_000 --max-cte 15 --run-name owl-12 -p 9097`

* Can also resume training from a checkpoint (checkpoints after the provided checkpoint will be overwritten):

    `(donkey) $ python examples/reinforcement_learning/SAC/train.py --port 9091 --resume-from runs/env_donkey-warren-track-v0/model_sac/train_2000000-20260213_110310/owl-24/training_checkpoint_340000_steps`



## test.py
* To run the trained agent, run: 
    
    


* Can also use model nick if you don't want to type out the full model path:

    


# Tensorboard Server

`tensorboard --logdir runs/`
<!-- ## ppo_train.py

An example using stable-baselines to train a PPO agent using the gym-donkeycar environment

* follow [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) install
* ```python gym-donkeycar/examples/reinforcement_learning/ppo_train.py --sim <path to simulator>```

## ddqn.py

An example training a [deep double Q-learning](https://arxiv.org/abs/1509.06461) agent using the gym-donkeycar environment

* ```python gym-donkeycar/examples/reinforcement_learning/ddqn.py --sim <path to simulator>``` -->
