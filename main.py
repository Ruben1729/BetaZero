import numpy as np
from BetaZeroAgent import Agent
from Timer import Timer
from BetaZeroPlot import plot_learning_curve

import gym
import gym_chess

if __name__ == '__main__':
    env = gym.make("ChessAlphaZero-v0")
    env.reset()

    games = 500
    load_checkpoint = False
    best_score = -np.inf

    agent = Agent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=env.observation_space.shape,
                  n_actions=env.action_space.n, mem_size=5000, eps_min=0.1,
                  batch_size=32, replace=1000, eps_dec=1e-5, chkpt_dir='models/', algo='DQN', env_name='BetaZero-v0')

    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' + str(games) + 'games'
    ffigure = 'plots/' + fname + '.png'

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    t = Timer()
    t.start()

    for i in range(games):
        done = False
        score = 0
        observation = env.reset()

        counter = 0
        while not done:
            action = agent.choose_action(observation, env.legal_actions)

            # Each state and action
            observation_, reward, done, info = env.step(action)

            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action, reward, observation_, done)
                agent.learn()

            observation = observation_
            n_steps += 1

        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print(f"{t.get_elapsed():0.4f}", 'episode', i, 'score: ', score,
              'average score %.1f best score %.1f epsilon %.2f' % (avg_score, best_score, agent.epsilon),
              'steps ', n_steps)

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

    plot_learning_curve(steps_array, scores, eps_history, ffigure)
