import flappy_game
from agent import Agent, Agent2
from flappy_game import flappyGame
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from pre_processing import pre_process_cnn_input, pre_process_dnn_input
from PIL import Image
import pickle


# Save agent's model parameters to disk
def save_agent(agent, filename):
    with open(filename, 'wb') as f:
        pickle.dump(agent.model, f)


def plotGraph(x, y, title, x_label, y_label):
    # Create a plot
    plt.plot(x, y)

    # Add labels to the plot
    plt.xlabel(str(x_label))
    plt.ylabel(str(y_label))
    plt.title(str(title))

    # Display the plot
    plt.show()


exploration = 1.0
exploration_decay = 0.01

num_episodes = 150


def random_output():
    if random.random() < 0.9:
        return 0
    else:
        return 1

def train_cnnq_model(exploration):
    agent = Agent(cnn_model=True)
    # Maybe add exploration rate and exploration decay
    scores = []
    flappyGameObj = flappyGame(cnn_model=True)
    for i in range(num_episodes):
        # Initialize the environment (game)
        state = flappyGameObj.main()
        score = 0
        done = False
        while not done:
            random_float = random.uniform(0, 1)
            if random_float > exploration:
                action = agent.choose_action(state)
            else:
                action = random_output()
                agent.store_action(tf.convert_to_tensor(action, 1))

            state_reward_struct = flappyGameObj.takeStep(action)

            # Perform pre-processing on the image
            image = pre_process_cnn_input(state_reward_struct['state'])

            state_ = image

            agent.store_reward(state_reward_struct['reward'])
            agent.store_state(state_)
            state = state_
            score += state_reward_struct['reward']

            done = state_reward_struct['done']
            if done:
                scores.append(score)
                agent.learn()
                print(f'episode done: {i + 1}\t score recieved: {score}')
                exploration -= exploration_decay

    x = list(range(1, len(scores) + 1))
    plotGraph(x, scores, "Rewards over episodes", "Episode", "Score")


def train_dnnq_model(exploration):
    agent = Agent(cnn_model=False)
    # Maybe add exploration rate and exploration decay
    scores = []
    flappyGameObj = flappyGame(cnn_model=False)
    for i in range(num_episodes):
        state = flappyGameObj.main()
        score = 0
        done = False
        while not done:
            random_float = random.uniform(0, 1)
            if random_float > exploration:
                action = agent.choose_action(state)
                print("Exploitation")
            else:
                action = random_output()
                agent.store_action(tf.convert_to_tensor(action, 1))
                print("Exploration")

            state_reward_struct = flappyGameObj.takeStep(action)

            # state_,reward,done,_ = env.step(action)
            # Perform normalization on the image

            state_ = pre_process_dnn_input(state_reward_struct)

            agent.store_reward(state_reward_struct['reward'])

            agent.store_state(state_)
            state = state_
            score += state_reward_struct['reward']

            done = state_reward_struct['done']
            if done:
                scores.append(score)
                agent.learn()
                exploration -= exploration_decay
                print(f'episode done: {i + 1}\t score recieved: {score}')

    x = list(range(1, len(scores) + 1))
    plotGraph(x, scores, "Rewards over episodes", "Episode", "Score")
    # Saving agent's model
    save_agent(agent, "./models/model1")


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
                exploration -= exploration_decay

                agent.train_long_memory()
                print(f'episode done: {i + 1}\t score recieved: {score}')

    x = list(range(1, len(scores) + 1))
    plotGraph(x, scores, "Rewards over episodes", "Episode", "Score")
    # Saving agent's model
    save_agent(agent, "./models/model_agent2")



train_dnnq_agent2(exploration)

#train_dnnq_model(exploration)

#train_cnnq_model(exploration)
