from agent import Agent
from flappy_game import flappyGame
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from pre_processing import pre_process_cnn_input, pre_process_dnn_input
from PIL import Image


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
exploration_decay = 0.001

num_episodes = 500


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
            else:
                action = random_output()
                agent.store_action(tf.convert_to_tensor(action, 1))

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


train_dnnq_model(exploration)
#train_cnnq_model(exploration)
