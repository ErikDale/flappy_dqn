from agent import Agent, ActorCriticAgent2
from flappy_game import flappyGame
from pre_processing import pre_process_cnn_input, pre_process_dnn_input
import pickle


# Load agent's model parameters from disk
def load_agent(agent, filename, actor):
    with open(filename, 'rb') as f:
        if actor:
            agent.actor_model = pickle.load(f)
        else:
            agent.model = pickle.load(f)
    return agent


def test_actor_critic_model():
    agent = ActorCriticAgent2(num_actions=2)

    agent = load_agent(agent, "./models/actor_critic_model2", actor=True)
    # Maybe add exploration rate and exploration decay
    flappyGameObj = flappyGame(cnn_model=True)

    # Initialize the environment (game)
    state = flappyGameObj.main()
    done = False
    while not done:
        action = agent.act(state)

        state_reward_struct = flappyGameObj.takeStep(action)

        state = pre_process_cnn_input(state_reward_struct['state'])

        done = state_reward_struct['done']
        if done:
            print("Game over")

def test_cnnq_model():
    agent = Agent(cnn_model=True)

    agent = load_agent(agent, "./models/model1", actor=False)
    # Maybe add exploration rate and exploration decay
    flappyGameObj = flappyGame(cnn_model=True)

    # Initialize the environment (game)
    state = flappyGameObj.main()
    done = False
    while not done:
        action = agent.choose_action(state)

        state_reward_struct = flappyGameObj.takeStep(action)

        state = pre_process_cnn_input(state_reward_struct['state'])

        done = state_reward_struct['done']
        if done:
            print("Game over")


def test_dnnq_model():
    agent = Agent(cnn_model=False)
    agent = load_agent(agent, "./models/model1", actor=False)
    # Maybe add exploration rate and exploration decay
    scores = []
    flappyGameObj = flappyGame(cnn_model=False)

    # Initialize the environment (game)
    state = flappyGameObj.main()
    done = False
    while not done:
        action = agent.choose_action(state)

        state_reward_struct = flappyGameObj.takeStep(action)

        state = pre_process_dnn_input(state_reward_struct)

        done = state_reward_struct['done']
        if done:
            print("Game over")

test_actor_critic_model()