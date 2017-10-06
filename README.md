# Awesome Hyperparams

These hyperparams should provide generally good starting points. Please check the original paper/code/etc (esp. in Deep RL) as the hyperparams for a specific model can vary based on your task or environment. 

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

See the [OpenAI baselines](https://github.com/openai/baselines) repo for solid implementations of many of these algorithms

### Deep Deterministic Policy Gradient

* [paper](https://arxiv.org/pdf/1509.02971v2.pdf)
* [hyperparam analysis](https://arxiv.org/pdf/1709.06560v1.pdf)

In the original paper, the actor and critic learning rates are reversed. However, to help stabilize the actor network during training, you generally want to encourage the critic network to converge faster; hence the larger initial lr for the critic is suggested here.

| hyperparam name | default value |
| --- | --- |
| policy network | 400/64 -> relu -> 300/64 -> relu -> tanh | 
| critic network |  400/64 -> relu -> 300/64 -> relu -> linear |
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

### TRPO

* [paper](https://people.eecs.berkeley.edu/~pabbeel/papers/2015-ICML-TRPO.pdf)
* [hyperparam analysis](https://arxiv.org/pdf/1709.06560v1.pdf)
    * TRPO appears to be quite robust to variations in the sizes of hidden layers

| hyperparam name | default value |
| --- | --- |
| policy network | 400/64 -> tanh -> 300/64 -> tanh -> linear, + std dev | 
| value network |  400/64 -> tanh -> 300/64 -> tanh -> linear |
| timesteps per batch | 5000 |
| max KL | 0.01 |
| conjugate gradient iters | 20 |
| conjugate gradient damping | 0.1 |
| value function (VF) optimizer | Adam |
| VF iters | 3-5 | 
| VF batch size | 64 |
| VF step size | 1e-3 |
| discount | 0.995 |
| entropy coeff | 0.0 |
| GAE lambda | 0.97 |

### PPO

* [paper](https://arxiv.org/abs/1707.06347)
    * More hyperparams are reported in the appendix
* [hyperparam analysis](https://arxiv.org/pdf/1709.06560v1.pdf)
* reported below are hyperparams for the Mujoco environment

| hyperparam name | default value |
| --- | --- |
| policy network | 64 -> tanh -> 64 -> tanh -> linear, + std dev | 
| value network |  64 -> tanh -> 64 -> tanh -> linear |
| timesteps per batch | 2048 |
| clip param | 0.2 |
| optimizer | Adam |
| optimizer epochs per iter | 10 |
| optimizer step size | 3e-4 |
| optimizer batch size | 64 |
| lr schedule | linear |
| discount | 0.995 |
| entropy coeff | 0.0 |
| GAE lambda | 0.97 |


## General


