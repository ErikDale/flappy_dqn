from agent import Agent, ActorCriticAgent
from flappy_game import flappyGame
import random
import matplotlib.pyplot as plt
from pre_processing import pre_process_cnn_input, pre_process_dnn_input
import pickle

"""
@author: Erik Dale
@date: 22.03.23
"""


def save_agent(agent, filename, actor):
    """
    Saves an agent's model to file
    :param agent: the agent that has the model that is to be saved
    :param filename: what file to save the model to
    :param actor: a bool that says if the agent is an actor critic agent or not
    """
    with open(filename, 'wb') as f:
        if actor:
            pickle.dump(agent.actor_model, f)
        else:
            pickle.dump(agent.model, f)


def plotGraph(x, y, title, x_label, y_label):
    """
    Plots a 2D graph
    :param x: x-axis values
    :param y: y-axis values
    :param title: title of the graph
    :param x_label: name of x-axis
    :param y_label: name of y-axis
    """

    # Create a plot
    plt.plot(x, y)

    # Add labels to the plot
    plt.xlabel(str(x_label))
    plt.ylabel(str(y_label))
    plt.title(str(title))

    # Display the plot
    plt.show()
    plt.savefig('./plots/plot.png')

# Epsilon starting value
epsilon = 1

# Number of episodes to train
num_episodes = 500

# Makes epsilon go from 1 to 0.1 in num_episodes episodes
epsilon_decay = 0.9 / num_episodes


def random_output():
    """
    Makes a random action with 90% chance of taking the action 0
    and 10% chance of taking the action 1
    :return: either 0 or 1 (actions)
    """
    if random.random() < 0.9:
        return 0
    else:
        return 1


def train_agent(epsilon, actor_critic, cnn):
    """
    Trains the different actors with their different models (cnn models, dnn models, actor and critic models)
    :param epsilon: starting value of epsilon
    :param actor_critic: if the agent that is to be trained is an actor critic agent or not
    :param cnn: if the agent that is to be trained uses a cnn model or not
    """
    best_score = 0  # best overall score
    has_learned = False  # tells if the training has started or not
    scores = []  # list of scores for plotting
    if actor_critic:
        agent = ActorCriticAgent(num_actions=2)  # initialization of actor critic agent
    else:
        agent = Agent(n_actions=2, cnn_model=cnn)  # initialization of cnn agent

    game = flappyGame(cnn_model=cnn)  # initialization of game

    num = 0

    for i in range(num_episodes):
        score = 0
        state = game.main()  # return initial state
        done = False

        # implement epsilon greedy action selection
        exploration_bool = False
        random_float = random.uniform(0, 1)
        if random_float < epsilon:
            exploration_bool = True

        while not done:
            num += 1
            if not exploration_bool:
                action = agent.choose_action(state)
                print("Exploitation")
            else:
                action = random_output()
                # agent.store_action(tf.convert_to_tensor(action, 1))
                print("Exploration")

            # perform action and get new state, reward and done bool
            state_reward_struct = game.take_step(action)

            # checks if the agent utilizes a cnn model or not
            # it has to check this to see what pre-processing to apply
            if cnn:
                next_state = pre_process_cnn_input(state_reward_struct['state'])
            else:
                next_state = pre_process_dnn_input(state_reward_struct)

            reward = state_reward_struct['reward']
            done = state_reward_struct['done']

            # convert done bool to 1 or 0
            if done:
                done = 1
            else:
                done = 0

            # fill up the experience replay memory
            agent.remember(state, action, reward, next_state, done)

            state = next_state  # update state

            score += reward

            '''if len(agent.replay_buffer) > 3000:
                agent.learn2(target_train=(num == 100))
                has_learned = True
                num = 0'''

            if done:
                if has_learned:
                    # Save the model if the current score is better than the best
                    # overall score
                    if score > best_score and not exploration_bool:
                        save_agent(agent, "models/dnn_medium", actor=actor_critic)
                        best_score = score  # Update the best score

                # append score to be plotted if it was exploitation
                if not exploration_bool:
                    scores.append(score)

                epsilon -= epsilon_decay

                # start learning when the memory is bigger than 3000
                if len(agent.replay_buffer) > 3000:
                    agent.learn()
                    has_learned = True

                print(f'episode done: {i + 1}\t score recieved: {score}')

    x = list(range(1, len(scores) + 1))
    plotGraph(x, scores, "Rewards over episodes", "Episode", "Score")
    print(best_score)


def train_agent_test(epsilon, actor_critic, cnn):
    """
    Trains the different actors with their different models (cnn models, dnn models, actor and critic models)
    :param epsilon: starting value of epsilon
    :param actor_critic: if the agent that is to be trained is an actor critic agent or not
    :param cnn: if the agent that is to be trained uses a cnn model or not
    """
    best_score = 0  # best overall score
    has_learned = False  # tells if the training has started or not
    scores = []  # list of scores for plotting
    if actor_critic:
        agent = ActorCriticAgent(num_actions=2)  # initialization of actor critic agent
    else:
        agent = Agent(n_actions=2, cnn_model=cnn)  # initialization of cnn agent

    game = flappyGame(cnn_model=cnn)  # initialization of game

    num = 0

    for i in range(num_episodes):
        action = 0
        states = []

        score = 0
        state = game.main()  # return initial state

        states.append(state)

        for j in range(3):
            state_reward_struct = game.take_step(action)
            state = pre_process_cnn_input(state_reward_struct['state'])
            states.append(state)

        done = False

        # implement epsilon greedy action selection
        exploration_bool = False
        random_float = random.uniform(0, 1)
        if random_float < epsilon:
            exploration_bool = True

        while not done:
            next_states = []

            num += 1
            if not exploration_bool:
                action = agent.choose_action(states)
                print("Exploitation")
            else:
                action = random_output()
                # agent.store_action(tf.convert_to_tensor(action, 1))
                print("Exploration")

            for j in range(4):
                if not done:
                    # perform action and get new state, reward and done bool
                    state_reward_struct = game.take_step(action)

                    # checks if the agent utilizes a cnn model or not
                    # it has to check this to see what pre-processing to apply
                    if cnn:
                        next_state = pre_process_cnn_input(state_reward_struct['state'])
                    else:
                        next_state = pre_process_dnn_input(state_reward_struct)

                    next_states.append(next_state)

                    if j == 0:
                        first_action = action
                        reward = state_reward_struct['reward']
                        done = state_reward_struct['done']

                    # convert done bool to 1 or 0
                    if done:
                        done = 1
                    else:
                        done = 0
                    action = 0

            if len(next_states) == 4:
                # fill up the experience replay memory
                agent.remember(states, first_action, reward, next_states, done)

            states = next_states  # update states

            score += reward

            '''if len(agent.replay_buffer) > 3000:
                agent.learn2(target_train=(num == 100))
                has_learned = True
                num = 0'''

            if done:
                if has_learned:
                    # Save the model if the current score is better than the best
                    # overall score
                    if score > best_score and not exploration_bool:
                        save_agent(agent, "models/cnn_medium_new_model", actor=actor_critic)
                        best_score = score  # Update the best score

                # append score to be plotted if it was exploitation
                if not exploration_bool:
                    scores.append(score)

                epsilon -= epsilon_decay

                # start learning when the memory is bigger than 3000
                if len(agent.replay_buffer) > 100:
                    agent.learn()
                    has_learned = True

                print(f'episode done: {i + 1}\t score recieved: {score}')

    x = list(range(1, len(scores) + 1))
    plotGraph(x, scores, "Rewards over episodes", "Episode", "Score")
    print(best_score)

train_agent_test(epsilon, actor_critic=False, cnn=True)
# train_agent(epsilon, False, False)
