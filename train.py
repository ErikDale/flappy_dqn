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


# Epsilon starting value
epsilon = 1

# Number of episodes to train
num_episodes = 1000

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
            if not exploration_bool:
                action = agent.act(state)
                print("Exploitation")
            else:
                action = random_output()
                # agent.store_action(tf.convert_to_tensor(action, 1))
                print("Exploration")

            # perform action and get new state, reward and done bool
            state_reward_struct = game.takeStep(action)

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

            if done:
                if has_learned:
                    # Save the model if the current score is better than the best
                    # overall score
                    if score > best_score:
                        save_agent(agent, "models/dnn_model_best", actor=actor_critic)
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


'''def train_cnnq_model(epsilon):
    """
    Trains the agent that uses the cnn model
    :param epsilon: starting value of epsilon
    """
    scores = []
    best_score = 0
    has_learned = False
    agent = Agent(n_actions=2, cnn_model=True)
    game = flappyGame(cnn_model=True)

    for i in range(num_episodes):
        score = 0
        state = game.main()
        done = False

        exploration_bool = False
        random_float = random.uniform(0, 1)
        if random_float > epsilon:
            exploration_bool = True

        while not done:
            # get move
            if not exploration_bool:
                action = agent.act(state)
                print("Exploitation")
            else:
                action = random_output()
                # agent.store_action(tf.convert_to_tensor(action, 1))
                print("Exploration")

            # perform move and get new state
            state_reward_struct = game.takeStep(action)

            next_state = pre_process_cnn_input(state_reward_struct['state'])

            reward = state_reward_struct['reward']

            done = state_reward_struct['done']

            if done:
                done = 1
            else:
                done = 0
            # remember
            agent.remember(state, action, reward, next_state, done)

            state = next_state

            score += reward

            if done:
                if has_learned:
                    # Save the model if the current score is better than the best
                    # overall score
                    if score > best_score:
                        save_agent(agent, "./models/cnn_model", actor=True)
                        best_score = score  # Update the best score

                if not exploration_bool:
                    scores.append(score)

                if agent.replay_buffer > 3000:
                    agent.learn()
                    has_learned = True

                print(f'episode done: {i + 1}\t score recieved: {score}')

    x = list(range(1, len(scores) + 1))
    plotGraph(x, scores, "Rewards over episodes", "Episode", "Score")


def train_dnnq_model(exploration):
    scores = []
    best_score = 0
    has_learned = False
    agent = Agent(n_actions=2, cnn_model=False)
    game = flappyGame(cnn_model=False)
    for i in range(num_episodes):
        score = 0
        state = game.main()
        done = False

        exploration_bool = False
        random_float = random.uniform(0, 1)
        if random_float > epsilon:
            exploration_bool = True

        while not done:
            # get move
            random_float = random.uniform(0, 1)
            if random_float > exploration:
                action = agent.act(state)
                print("Exploitation")
            else:
                action = random_output()
                # agent.store_action(tf.convert_to_tensor(action, 1))
                print("Exploration")

            if not exploration_bool:
                action = agent.act(state)
                print("Exploitation")
            else:
                action = random_output()
                # agent.store_action(tf.convert_to_tensor(action, 1))
                print("Exploration")

            state_reward_struct = game.takeStep(action)

            next_state = pre_process_dnn_input(state_reward_struct)

            reward = state_reward_struct['reward']

            done = state_reward_struct['done']

            if done:
                done = 1
            else:
                done = 0
            # remember
            agent.remember(state, action, reward, next_state, done)

            state = next_state

            score += reward

            if done:
                if has_learned:
                    # Save the model if the current score is better than the best
                    # overall score
                    if score > best_score:
                        save_agent(agent, "./models/dnn_model", actor=True)
                        best_score = score  # Update the best score

                if not exploration_bool:
                    scores.append(score)

                if agent.replay_buffer > 3000:
                    agent.learn()
                    has_learned = True
                print(f'episode done: {i + 1}\t score recieved: {score}')

    x = list(range(1, len(scores) + 1))
    plotGraph(x, scores, "Rewards over episodes", "Episode", "Score")
    # Saving agent's model
    # save_agent(agent, "./models/model1", actor=False)


def train_cnnq_agent2(exploration):
    scores = []

    agent = Agent2(cnn_model=True)
    game = flappyGame(cnn_model=True)
    for i in range(num_episodes):
        score = 0
        state_old = game.main()
        state_old = [state_old]
        done = False

        exploration_bool = False
        random_float = random.uniform(0, 1)
        if random_float > epsilon:
            exploration_bool = True

        while not done:
            # get move
            random_float = random.uniform(0, 1)
            if not exploration_bool:
                final_move = agent.get_action(state_old)
                print("Exploitation")
            else:
                final_move = random_output()
                # agent.store_action(tf.convert_to_tensor(action, 1))
                print("Exploration")

            # perform move and get new state
            state_reward_struct = game.takeStep(final_move)

            state_new = pre_process_cnn_input(state_reward_struct['state'])

            state_new = [state_new]

            reward = state_reward_struct['reward']

            done = state_reward_struct['done']
            # train short memory
            agent.train_short_memory(state_old, [final_move], [reward], state_new, [done])

            # remember
            agent.remember(state_old, [final_move], [reward], state_new, [done])

            state_old = state_new

            score += reward

            if done:
                # train long memory, plot result

                scores.append(score)

                agent.train_long_memory()
                print(f'episode done: {i + 1}\t score recieved: {score}')

    x = list(range(1, len(scores) + 1))
    plotGraph(x, scores, "Rewards over episodes", "Episode", "Score")
    # Saving agent's model
    save_agent(agent, "./models/cnn_model_agent2", actor=False)


def train_dnnq_agent2(exploration):
    scores = []

    agent = Agent2(cnn_model=False)
    game = flappyGame(cnn_model=False)
    for i in range(num_episodes):
        score = 0
        state_old = game.main()
        state_old = [state_old]
        done = False
        while not done:
            # get move
            random_float = random.uniform(0, 1)
            if random_float > exploration:
                final_move = agent.get_action(state_old)
                print("Exploitation")
            else:
                final_move = random_output()
                # agent.store_action(tf.convert_to_tensor(action, 1))
                print("Exploration")

            # perform move and get new state
            state_reward_struct = game.takeStep(final_move)
            state_new = pre_process_dnn_input(state_reward_struct)
            state_new = [state_new]

            reward = state_reward_struct['reward']

            done = state_reward_struct['done']
            # train short memory
            agent.train_short_memory(state_old, [final_move], [reward], state_new, [done])

            # remember
            agent.remember(state_old, [final_move], [reward], state_new, [done])

            state_old = state_new

            score += reward

            if done:
                # train long memory, plot result

                scores.append(score)

                agent.train_long_memory()
                print(f'episode done: {i + 1}\t score recieved: {score}')

    x = list(range(1, len(scores) + 1))
    plotGraph(x, scores, "Rewards over episodes", "Episode", "Score")
    # Saving agent's model
    save_agent(agent, "./models/model_agent2", actor=False)'''


# train_dnnq_agent2(exploration)

# train_dnnq_model(exploration)

# train_cnnq_model(exploration)

# train_cnnq_agent2(exploration)

train_agent(epsilon, False, False)
