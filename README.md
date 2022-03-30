# Reinforcement Learning for Robots Locomotion Learning

Killian SUSINI - Zakariae EL ASRI


## Summary

In this project, we provide the details of implementing a reinforcement learning (RL) algorithm for controlling a multi-legged robot. In particular, we describe the implementation of TD3 (twin-delayed deep deterministic policy gradient) concept to train a Walker-2D to move across a line. In the process, we used OpenAI/Gym environments and Pytorch utilities.



## Random agent

<img src="images\random_agent.gif" width="384" height="256" />

## TD3 Agent During training

<img src="images\Training.gif" width="384" height="256" />

## Testing the trained TD3 Agent

<img src="images\Testing.gif" width="384" height="256" />

## Installation

To install the dependencies use pip and the requirements.txt in this directory. e.g.

```
    $ pip install pybullet
    $ pip install matplotlib
    $ pip install numpy
    $ pip install gym
```

## How to Run

To run the model interactively, run ``mesa runserver`` in this directory. e.g.

```
For training:
    $ python td3_learning_main.py --save_model --render --seed #number# --env "MinitaurBulletEnv-v0"/"Walker2DBulletEnv-v0"
For testing:
    $ python td3_learning_main.py --load_model "default" --testing --render --seed #number_on_save# --env "MinitaurBulletEnv-v0"/"Walker2DBulletEnv-v0"
    
The seed in the test args must be the same used in the training.
```


## Files

* ``td3_learning_main.py``: This is the main code .py
* ``TD3_Walker2D_Tutorial.ipynb``: A tutorial from Google Collab
* ``Report_TD3_Walker2D.pdf``: The final Report for the project.

## Further Reading

This model is closely based on (TD3) paper and an implementation for Minitaur:

S. Fujimoto, H. van Hoof, and D. Meger. Addressing function approximation error in actorcritic methods. arXiv preprint arXiv:1802.09477, 2018.![image](https://user-images.githubusercontent.com/72689460/160902855-c303f9ce-4b94-4fd4-a61f-388aabc841a6.png)


https://github.com/liu-qingzhen/Minitaur-Pybullet-TD3-Reinforcement-Learning
