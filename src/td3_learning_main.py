import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import argparse
import os

import matplotlib.pyplot as plt

import pybullet_envs
import pybullet


from time import sleep

# code mostly taken from the Twin Delayed Deep Deterministic Policy Gradients (TD3) paper code https://arxiv.org/abs/1802.09477
# and modified, design choices inspired by the work done at https://github.com/liu-qingzhen/Minitaur-Pybullet-TD3-Reinforcement-Learning


print("using ", "cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    # This is basically taken straight from TD3 original papers' code, it's just a ciruclar buffer on parallel arrays
    def __init__(self, state_dim, action_dim, max_size):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        # essentially a circular buffer
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.done[ind]).to(device)
        )
        
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        return self.max_action * torch.tanh(self.fc3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        q = F.relu(self.fc1(state_action))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)

        return q


class TD3(object):
    # Adapted from both the paper and the git

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        lr=3e-4
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)

        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)


        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        ret_actor_loss = None

        # Sample replay buffer 
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1 = self.critic1_target(next_state, next_action)
            target_Q2 = self.critic2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1-done) * self.discount * target_Q

        # Get current Q estimates
        current_Q1 = self.critic1(state, action)
        current_Q2 = self.critic2(state, action)
        
        # Compute critics loss
        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)
        ret_critics_loss = critic1_loss.item() + critic2_loss.item()

        # Optimize the critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic1(state, self.actor(state)).mean()
            ret_actor_loss = actor_loss.item()
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        
        return ret_actor_loss, ret_critics_loss


    def save(self, filename):
        # Saving everything (otherwise the training on load is buggy)
        torch.save(self.critic1.state_dict(), filename + "_critic1")
        torch.save(self.critic1_target.state_dict(), filename + "_critic1_target")
        torch.save(self.critic1_optimizer.state_dict(), filename + "_critic1_optimizer")
        
        torch.save(self.critic2.state_dict(), filename + "_critic2")
        torch.save(self.critic2_target.state_dict(), filename + "_critic2_target")
        torch.save(self.critic2_optimizer.state_dict(), filename + "_critic2_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_target.state_dict(), filename + "_actor_target")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic1.load_state_dict(torch.load(filename + "_critic1"))
        self.critic1_target.load_state_dict(torch.load(filename + "_critic1_target"))
        self.critic1_optimizer.load_state_dict(torch.load(filename + "_critic1_optimizer"))

        self.critic2.load_state_dict(torch.load(filename + "_critic2"))
        self.critic2_target.load_state_dict(torch.load(filename + "_critic2_target"))
        self.critic2_optimizer.load_state_dict(torch.load(filename + "_critic2_optimizer"))

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_target.load_state_dict(torch.load(filename + "_actor_target"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes, shake_weight, drift_weight):
    if env_name == "Walker2DBulletEnv-v0":
        eval_env = gym.make(env_name)
    else:
        eval_env = gym.make(env_name, shake_weight=shake_weight, drift_weight=drift_weight)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")

    eval_env.close()

    return avg_reward

def test_agent(policy, env):
    test_env = env
    test_reward = 0.
    for _ in range(100):    
        state, done = test_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            sleep(1./240.)
            #print(test_env._cam_dist)
            state, reward, done, _ = test_env.step(action)
            
            #pybullet.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=0.0, cameraPitch=-20, cameraTargetPosition=[0, 0, 1.0])
            test_reward += reward

    print("---------------------------------------")
    print(f"Test reward over the episode: {test_reward:.3f}")
    print("---------------------------------------")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Walker2DBulletEnv-v0")    # gym environment name (only MinitaurBulletEnv-v0 and Walker2DBulletEnv-v0)
    parser.add_argument("--seed", default=-1, type=int)             # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=25e3, type=int)      # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=15e5, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--learning_rate", default=3e-4)            # Learning rate for the critics and actor
    parser.add_argument("--shake_weight", default=0.2)              # (minitaur only) control the penality caused by the robot shaking
    parser.add_argument("--drift_weight", default=0.1)              # (minitaur only) control the penality caused by the robot not staying on course
    parser.add_argument("--render", action="store_true")            # render the training (it doesn't affect the learning speed)
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--testing", action="store_true")           # (needs --load_model)Test the model for a run (needs --render too)
    
    args = parser.parse_args()

    if args.seed == -1:
        seed = np.random.randint(0, 10000)
    else:
        seed = args.seed

    file_name = f"{args.env}_Agent_seed_{seed}"

    if args.save_model and not os.path.exists(f"./saved_models/{file_name}"):
        os.makedirs(f"./saved_models/{file_name}")

    if args.env == "Walker2DBulletEnv-v0":
        env = gym.make(args.env, render=args.render)
    else:
        env = gym.make(args.env, render=args.render, shake_weight=args.shake_weight, drift_weight=args.drift_weight)

    
    # Set seeds
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])
    print(max_action)

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "lr": args.learning_rate,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
    }
    policy = TD3(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./saved_models/{policy_file}/{policy_file}")    
        if args.testing:
            test_agent(policy, env)
            exit(0)
    else:
        print("please specify the name of the folder, or \"default\" for the autogenerated name used in training with that seed")
    
    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(1e5)) # larger buffers are good, to a point
    
    # for plotting
    y = [eval_policy(policy, args.env, seed, 10, args.shake_weight, args.drift_weight)] # cumul timesteps
    x = [0] # cumul timesteps
    train_reward = [] # per episode
    train_timesteps = [] # per episode
    train_actor_loss = [] # per episode
    train_critics_loss = [] # per episode

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    episode_actor_loss = 0.0
    episode_critics_loss = 0.0


    for t in range(int(args.max_timesteps)):
        
        episode_timesteps += 1

        # Select action randomly for the first args.start_timesteps (to load some data in the replay buffer)
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            # Add some random noise to the selected actions (for exploration)
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action) 
        
        # This is env specific, but for our purpose env._max_episode_steps is 1000
        # we don't want to flag the step as an ending step if it was stopped by going over the limit
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0.0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Now we can train
        if t >= args.start_timesteps:
            actor_loss, critics_loss = policy.train(replay_buffer, args.batch_size)
            if actor_loss:
                episode_actor_loss += actor_loss
            episode_critics_loss += critics_loss

        if done: 
            train_reward.append(episode_reward)
            train_timesteps.append(episode_timesteps)
            train_actor_loss.append(episode_actor_loss/episode_timesteps)
            train_critics_loss.append(episode_critics_loss/episode_timesteps)
            print(f"cumulated timesteps: {t+1} Episode: {episode_num+1} timesteps: {episode_timesteps} reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_actor_loss = 0.0
            episode_critics_loss = 0.0
            episode_num += 1 
            
        if (t + 1) % args.eval_freq == 0:
            y.append(eval_policy(policy, args.env, seed, 10, args.shake_weight, args.drift_weight))
            x.append(t)
            if args.save_model: policy.save(f"./saved_models/{file_name}/{file_name}")
            fig1, ax1 = plt.subplots()
            ax1.plot(x, y)
            plt.savefig(f"./saved_models/{file_name}/eval_avg_reward.png")
            plt.close(fig1)

            fig2, ax2 = plt.subplots()
            ax2.plot(list(range(0,len(train_reward))), train_reward)
            plt.savefig(f"./saved_models/{file_name}/train_reward.png")
            plt.close(fig2)

            fig3, ax3 = plt.subplots()
            ax3.plot(list(range(0,len(train_timesteps))), train_timesteps)
            plt.savefig(f"./saved_models/{file_name}/train_timesteps.png")
            plt.close(fig3)

            fig4, ax4 = plt.subplots()
            ax4.plot(list(range(0,len(train_actor_loss))), train_actor_loss)
            plt.savefig(f"./saved_models/{file_name}/train_actor_loss.png")
            plt.close(fig4)

            fig5, ax5 = plt.subplots()
            ax5.plot(list(range(0,len(train_critics_loss))), train_critics_loss)
            plt.savefig(f"./saved_models/{file_name}/train_critics_loss.png")
            plt.close(fig5)
