from agent import Agent, Agent2, ActorCriticAgent, ActorCriticAgent2
from flappy_game import flappyGame
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from pre_processing import pre_process_cnn_input, pre_process_dnn_input
from PIL import Image
import pickle


# Save agent's model parameters to disk
def save_agent(agent, filename, actor):
    with open(filename, 'wb') as f:
        if actor:
            pickle.dump(agent.actor_model, f)
        else:
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


epsilon = 1



exploration = 1.0
exploration_decay = 0.001

num_episodes = 1000

# Makes epsilon go from 1 to 0.1 in 1000 episodes
epsilon_decay = 0.9 / 1000


def random_output():
    if random.random() < 0.9:
        return 0
    else:
        return 1


def train_cnnq_actor_critic_agent(exploration, epsilon):
    scores = []
    agent = ActorCriticAgent2(num_actions=2)
    game = flappyGame(cnn_model=True)
    for i in range(num_episodes):
        score = 0
        state = game.main()
        done = False

        exploration_bool = False
        random_float = random.uniform(0, 1)
        if random_float < epsilon:
            exploration_bool = True

        while not done:
            # get move
            '''random_float = random.uniform(0, 1)
            if random_float > exploration:
                action = agent.act(state)
                print("Exploitation")
            else:
                action = random_output()
                # agent.store_action(tf.convert_to_tensor(action, 1))
                print("Exploration")'''

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
                # train long memory, plot result
                if not exploration_bool:
                    scores.append(score)

                exploration -= exploration_decay

                # Stop decreasing episolon after it has hit a value of 0.1
                if epsilon > 0.1:
                    epsilon -= epsilon_decay

                if len(agent.replay_buffer) > 3000:
                    agent.learn()

                print(f'episode done: {i + 1}\t score recieved: {score}')

    x = list(range(1, len(scores) + 1))
    plotGraph(x, scores, "Rewards over episodes", "Episode", "Score")
    # Saving agent's model
    save_agent(agent, "./models/actor_critic_model2", actor=True)


def train_cnnq_model(exploration):
    scores = []

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
            '''random_float = random.uniform(0, 1)
            if random_float > exploration:
                action = agent.act(state)
                print("Exploitation")
            else:
                action = random_output()
                # agent.store_action(tf.convert_to_tensor(action, 1))
                print("Exploration")'''

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
                # train long memory, plot result
                if not exploration_bool:
                    scores.append(score)

                exploration -= exploration_decay

                agent.learn()
                print(f'episode done: {i + 1}\t score recieved: {score}')

    x = list(range(1, len(scores) + 1))
    plotGraph(x, scores, "Rewards over episodes", "Episode", "Score")
    # Saving agent's model
    save_agent(agent, "./models/cnn_model", actor=False)


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
                exploration -= exploration_decay

                agent.train_long_memory()
                print(f'episode done: {i + 1}\t score recieved: {score}')

    x = list(range(1, len(scores) + 1))
    plotGraph(x, scores, "Rewards over episodes", "Episode", "Score")
    # Saving agent's model
    save_agent(agent, "./models/cnn_model_agent2", actor=False)


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
    save_agent(agent, "./models/model1", actor=False)


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
    save_agent(agent, "./models/model_agent2", actor=False)


# train_dnnq_agent2(exploration)

# train_dnnq_model(exploration)

#train_cnnq_model(exploration)

# train_cnnq_agent2(exploration)

train_cnnq_actor_critic_agent(exploration, epsilon)
