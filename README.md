# Awesome Hyperparams

## Contributors 

Please provide citations (e.g., arxiv link, blog post, github repo, etc). Any info on the hyperparameter search process taken in the original work (if available) is a bonus.

Example contribution: 

### Original DQN

* [paper](https://arxiv.org/pdf/1312.5602.pdf)
* [original Torch code](https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner/blob/master/run_gpu)

| hyperparam name | default value |
| --- | --- |
| lr | 25e-5 |
| RMSprop momentum | 0.95 |
| RMSprop epsilon | 0.01 | 
| discount factor | 0.99 |
| epsilon(-greedy) | 1 annealed to 0.1 over 1 million frames |
| minibatch size | 32 |
| replay memory size | 1000000 |
| weight init | Xavier (Torch default) |

## Computer Vision

## Natural Language Processing 

## Deep Reinforcement Learning

### Deep Deterministic Policy Gradient

* [paper](https://arxiv.org/pdf/1509.02971v2.pdf)

In the paper, the actor and critic learning rates are reversed. However, to help stabilize the actor network during training, you generally want to encourage the critic network to converge faster; hence the larger initial lr for the critic is suggested here.

| hyperparam name | default value |
| --- | --- |
| actor lr | 10-4 |
| critic lr | 10-3 |
| critic L2 weight decay | 10-2 |
| discount factor | 0.99 |
| target network update tau | 0.001 |
| Ornstein-Uhlenbeck theta | 0.15 |
| Ornstein-Uhlenbeck sigma | 0.3 |
| minibatch size | 64 on low-dim input, 16 on pixel-input | 
| replay memory size | 1000000 |
| weight init | final layer of actor & critic are uniform(-3 * 10-3, 3 * 10-3) for low-dim input and uniform(-3 * 10-4, 3 * 10-4) for pixel-input; other layers => Xavier |

### A3C

* [paper](https://arxiv.org/pdf/1602.01783.pdf)
* [Correspondence by @muupan with author on hyperparams](https://github.com/muupan/async-rl/wiki)
    * See for more training details 
    
| hyperparam name | default value |
| --- | --- |
| discount factor | 0.99 |
| shared RMSprop eta | 7e-4 |
| shared RMSprop alpha | 0.99 |
| shared RMSprop epsilon | 0.1 |
| A3C entropy regularization beta | 0.01 |
| V-network gradients multiplied by | 0.5 |
| Weight init | Xavier (Torch default) |
| Reward clipping | [-1, 1] on Atari |
| # of threads w/ best performance | 16 |

## General


