# Reinforcement Learning for Robots Locomotion Learning

Authors: Killian SUSINI - Zakariae EL ASRI


## Summary

In this project, we provide the details of implementing a reinforcement learning (RL) algorithm for controlling a multi-legged robot. In particular, we describe the implementation of TD3 (twin-delayed deep deterministic policy gradient) concept to train a Walker-2D to move across a line. In the process, we used OpenAI/Gym environments and Pytorch utilities.

A tutorial is availible in this google colab: https://colab.research.google.com/drive/1sJuJ-as_bK9le_CpF41EV2jBVG383V6m?usp=sharing



## Random agent

A random agent chooses actions Randomly. It will quickly fall since its movements are not synchronous.

<img src="images\random_agent.gif" width="384" height="256" />


## TD3 Agent During training

Here, we display three stage from training. 

At the beginning, when the agent chooses actions based on normal distribution. 

At the middle, when the agent starts to learn greedily. 

At the end of training, the agent learned to walk with a quite good gait.

<img src="images\Training.gif" width="384" height="256" />


## Testing the trained TD3 Agent

Here, we apply the policy learned during training to a new test episode.

<img src="images\Testing.gif" width="384" height="256" />


## Installation

To install the dependencies use pip 

```
    $ pip install pybullet
    $ pip install matplotlib
    $ pip install numpy
    $ pip install gym
```

## How to Run

To run the model :


For training:
```
    $ python td3_learning_main.py --save_model --render --seed #number# --env "MinitaurBulletEnv-v0"/"Walker2DBulletEnv-v0"
```
For testing:
```
    $ python td3_learning_main.py --load_model "default" --testing --render --seed #number_on_save# --env "MinitaurBulletEnv-v0"/"Walker2DBulletEnv-v0"
```    
The seed in the test args must be the same used in the training.



## Files

* ``td3_learning_main.py``          : This is the main code .py
* ``TD3_Walker2D_Tutorial.ipynb``   : A tutorial from Google Collab
* ``Report_TD3_Walker2D.pdf``       : The final Report for the project.

## Further Reading

This model is closely based on (TD3) paper and an implementation for Minitaur:

[1] S. Fujimoto, H. van Hoof, and D. Meger. Addressing function approximation error in actorcritic methods. 2018
https://arxiv.org/abs/1802.09477


[2] https://github.com/liu-qingzhen/Minitaur-Pybullet-TD3-Reinforcement-Learning
