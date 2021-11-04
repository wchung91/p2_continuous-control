from unityagents import UnityEnvironment
import numpy as np
from ddpg_agent import Agent
import random
import torch
from collections import deque
import matplotlib.pyplot as plt

# select this option to load version 1 (with a single agent) of the environment
#env = UnityEnvironment(file_name='./oneAgent/Reacher_Linux_NoVis/Reacher.x86_64')
env = UnityEnvironment(file_name='./Reacher_Linux_NoVis/Reacher.x86_64')

# select this option to load version 2 (with 20 agents) of the environment
# env = UnityEnvironment(file_name='/data/Reacher_Linux_NoVis/Reacher.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


env_info = env.reset(train_mode=True)[brain_name]      # reset the environment
states = env_info.vector_observations                  # get the current state (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)


"""
    Mode to Run
    Change this in order to Train or Test the agent
    Train = 1 for training the agent
    Train = 0 for testing the agent
"""
Train = 1
train_n_episode = 2000
test_n_episode = 100

"""
    Code for training
"""
if Train == 1:
    #Initialize agent
    agent = Agent(state_size=33, action_size=4, random_seed=2)

    #method containing algorithm for ddpg (training)
    def ddpg(n_episodes=train_n_episode, max_t=10000, print_every=100):
        scores_deque = deque(maxlen=print_every)                   # queue for averaging 100 episodes
        scores_history = []                                        # list with score history
        for i_episode in range(1, n_episodes+1):                   # Start episode
            env_info = env.reset(train_mode=True)[brain_name]      # reset environment
            states = env_info.vector_observations                  # Get initial observation
            scores = np.zeros(num_agents)                          # initialize the score (for each agent)
            agent.reset()                                          # reset agent
            score = 0                                              # initialize score
            for t in range(max_t):                                 # start step
                #actions = np.random.randn(num_agents, action_size)
                actions = agent.act(states[0], add_noise=False)                     # select an action (for each agent)
                actions = np.array(actions).reshape(1,4)           # convert to np array

                env_info = env.step(actions)[brain_name]           # send action to tne environment
                next_states = env_info.vector_observations         # get next state (for each agent)
                rewards = env_info.rewards                         # get reward (for each agent)
                dones = env_info.local_done                        # see if episode finished
                agent.step(states[0], actions[0], rewards[0], next_states[0], dones[0]) #Update information to the agent
                score += env_info.rewards[0]                       # update the score (for each agent)
                states = next_states                               # roll over states to next time step
                if np.any(dones):                                  # exit loop if episode finished
                    break

            scores_deque.append(score)                             # Append to queue
            scores_history.append(score)                           # Append to history list

            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')      #Save actor
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')    #Save critic

            #Print average score every 100 step
            if i_episode % print_every == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        return scores_history

    #code for saving figure
    scores = ddpg()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    #plt.show()
    plt.savefig("ScoresTraining")


"""
    Code for testing
"""
if Train == 0:
    agent = Agent(state_size=33, action_size=4, random_seed=2)     # initialize agent

    #load actor and critic network
    agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
    agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

    #method containing algorithm for ddpg (testing)
    def ddpg(n_episodes=test_n_episode, max_t=10000, print_every=100):
        scores_deque = deque(maxlen=print_every)                   # queue used for averaging 100 scores
        scores_history = []                                        # contains history of scores
        for i_episode in range(1, n_episodes+1):                   # start of an episode
            env_info = env.reset(train_mode=True)[brain_name]      # reset environment
            states = env_info.vector_observations                  # get state
            scores = np.zeros(num_agents)                          # initialize the score (for each agent)

            agent.reset()                                          # reset agent
            score = 0                                              # reset score
            for t in range(max_t):                                 # start of step
                actions = agent.act(states[0])                     # select an action
                actions = np.array(actions).reshape(1,4)           # change action to np array

                env_info = env.step(actions)[brain_name]           # send all actions to tne environment
                next_states = env_info.vector_observations         # get next state (for each agent)
                rewards = env_info.rewards                         # get reward (for each agent)
                dones = env_info.local_done                        # see if episode finished

                score += env_info.rewards[0]                       # update the score (for each agent)
                states = next_states                               # roll over states to next time step

                if np.any(dones):                                  # exit loop if episode finished
                    break

            scores_deque.append(score)                             # append to queue
            scores_history.append(score)                           # append to score history

            print('\rEpisode {}\ Score: {:.2f}'.format(i_episode, score))   #print score every episode

            #print average score every 100 episode
            if i_episode % print_every == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        return scores_history

    #plot figure and save
    scores = ddpg()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    #plt.show()
    plt.savefig("TestScores")
env.close()
