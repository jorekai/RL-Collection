from collections import namedtuple

import gym
import numpy as np
import tensorflow as tf

# we define a experience tuple for easy convention of MDP notation
# sometimes in literature people use the equivalent word observation for state
# state == observation
from ddqn_c51.agent import C51Agent
from ddqn_c51.memory import ReplayMemory
from ddqn_c51.nn import NN

Experience = namedtuple("Experience", "state action reward next_state done")


# helper method for reshaping the cartpole observation
def reshape(state):
    return np.reshape(state, [1, 4])


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    max_score = 0

    n_episodes = 5000
    max_env_steps = 1000

    env = gym.make('CartPole-v0')
    agent = C51Agent(env=env,
                      net=NN(env=env, alpha=0.001, decay=0.0001, atoms=51),
                      target_net=NN(env=env, alpha=0.001, decay=0.0001, atoms=51),
                      memory=ReplayMemory(size=100000))

    if max_env_steps is not None:
        env._max_episode_steps = max_env_steps

    for e in range(n_episodes):
        # reset the env
        state = reshape(env.reset())
        done = False
        score = 0
        # play until env done
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # env.render()
            next_state = reshape(next_state)
            agent.memory.append(Experience(state, action, reward, next_state, done))
            state = next_state
            score += 1
        for i in range(10):
            # replay experience and decay exploration factor
            agent.replay(batch_size=64)
            agent.decay_epsilon()
        if score >= max_score:
            max_score = score
            print(f"Score in episode: {e} is: {score} --- eps: {agent.epsilon}")
