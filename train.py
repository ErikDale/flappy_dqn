from agent import Agent
from flappy_game import flappyGame
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from pre_processing import pre_process
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


# Maybe add exploration rate and exploration decay
exploration = 1.0
exploration_decay = 0.05

agent = Agent()
num_episodes = 100

scores = []

for i in range(num_episodes):
    # Initialize the environment (game)
    flappyGameObj = flappyGame()
    state = flappyGameObj.main()
    score = 0
    rewards = []
    states = []
    actions = []
    done = False
    while not done:
        random_float = random.uniform(0, 1)
        if random_float > exploration:
            action = agent.choose_action(state)
        else:
            action = random.randint(0, 1)
            exploration -= exploration_decay
            agent.store_action(tf.convert_to_tensor(action, 1))

        state_reward_struct = flappyGameObj.takeStep(action)

        # Perform pre-processing on the image
        image = pre_process(state_reward_struct['state'])

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

x = list(range(1, len(scores) + 1))
plotGraph(x, scores, "Rewards over episodes", "Episode", "Score")
