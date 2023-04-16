from agent import Agent, ActorCriticAgent
from flappy_game import flappyGame
from pre_processing import pre_process_cnn_input, pre_process_dnn_input
import pickle


"""
@author: Erik Dale
@date: 28.03.23
"""


def load_agent(agent, filename, actor):
    """
    Load a saved agent or actor model from a file and update the agent's or actor's parameters.
    :param agent: the agent or actor model to be updated.
    :param filename: the filename of the saved agent or actor model.
    :param actor: a boolean indicating whether to update the agent's actor model or not.
    :return: the updated agent or actor model.
    """
    with open(filename, 'rb') as f:
        if actor:
            agent.actor_model = pickle.load(f)
        else:
            agent.model = pickle.load(f)
    return agent


def test_agent(actor_critic, cnn):
    """
    Tests a trained agent in the Flappy Bird game environment.
    :param actor_critic: a boolean indicating whether to use an actor-critic agent or not.
    :param cnn: a boolean indicating whether to use a convolutional neural network or not.
    """
    # Initialize the agent
    if actor_critic:
        agent = ActorCriticAgent(num_actions=2)
    else:
        agent = Agent(cnn_model=cnn)

    # Load the saved model parameters
    agent = load_agent(agent, "models/actor_critic_model_best", actor=actor_critic)

    # Initialize the Flappy Bird game environment
    flappyGameObj = flappyGame(cnn_model=cnn)

    # Get the initial state
    state = flappyGameObj.main()

    # Play the game until it's over
    done = False
    while not done:
        # Get the action from the agent
        action = agent.act(state)

        # Take a step in the environment
        state_reward_struct = flappyGameObj.takeStep(action)

        # Pre-process the next state
        if cnn:
            state = pre_process_cnn_input(state_reward_struct['state'])
        else:
            state = pre_process_dnn_input(state_reward_struct)

        # Check if the game is over
        done = state_reward_struct['done']
        if done:
            print("Game over")


test_agent(actor_critic=True, cnn=True)