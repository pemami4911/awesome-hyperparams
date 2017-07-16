# Awesome Hyperparams

## Contributors 

Please provide citations (e.g., arxiv link, blog post, github repo, etc). Any info on the hyperparameter search process taken in the original work (if available) is a bonus. Please use scientific "e" notation (10e5 instead of 1000000).

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
| replay memory size | 10e5 |
| weight init | Xavier (Torch default) |

## Computer Vision

### DCGAN

* [paper](https://arxiv.org/abs/1511.06434)
* [ganhacks from "How To Train a GAN" at NIPS 2016](https://github.com/soumith/ganhacks)

| hyperparam name | default value |
| --- | --- |
| ADAM lr | 2e-4 |
| ADAM momentum beta1 | 0.5 |
| minibatch size | 64 or 128 |
| image scaling | [-1, 1] |
| LeakyReLU slope | 0.2 |
| Real labels (label smoothing) | 1 -> [0.7, 1.2] |
| Fake labels (label smoothing | 0 -> [0.0, 0.3] |
| Weight init | N(0, 0.02) |
| Z distribution| n-dim uniform or gaussian (e.g., uniform (-0.2, 0.2) from [this](https://github.com/carpedm20/DCGAN-tensorflow/blob/master/utils.py#L200) implementation |

For Z, sampling from a uniform distribution is simpler, but see the discussion here about [interpolation in the latent space](https://github.com/soumith/dcgan.torch/issues/14); [current recommendation](https://github.com/soumith/ganhacks#3-use-a-spherical-z) is to use a spherical Z and interpolate via a [great circle](https://en.wikipedia.org/wiki/Great_circle)

## Natural Language Processing 

## Deep Reinforcement Learning

### Deep Deterministic Policy Gradient

* [paper](https://arxiv.org/pdf/1509.02971v2.pdf)

In the paper, the actor and critic learning rates are reversed. However, to help stabilize the actor network during training, you generally want to encourage the critic network to converge faster; hence the larger initial lr for the critic is suggested here.

| hyperparam name | default value |
| --- | --- |
| actor lr | 10e-4 |
| critic lr | 10e-3 |
| critic L2 weight decay | 10e-2 |
| discount factor | 0.99 |
| target network update tau | 10e-4 |
| Ornstein-Uhlenbeck theta | 0.15 |
| Ornstein-Uhlenbeck sigma | 0.3 |
| minibatch size | 64 on low-dim input, 16 on pixel-input | 
| replay memory size | 10e5 |
| weight init | final layer of actor & critic are uniform(-3 * 10-3, 3 * 10-3) for low-dim input and uniform(-3 * 10-4, 3 * 10-4) for pixel-input; other layers -> Xavier |

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


