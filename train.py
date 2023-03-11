from agent import Agent
from flappy_game import flappyGame


agent = Agent()
num_episodes = 100

for i in range(num_episodes):
    flappyGameObj = flappyGame()
    state = flappyGameObj.main()
    score = 0
    rewards = []
    states = []
    actions = []
    done = False
    while not done:
        action = agent.choose_action(state)
        state_reward_struct = flappyGameObj.takeStep(action)

        # state_,reward,done,_ = env.step(action)

        agent.store_reward(state_reward_struct['reward'])
        agent.store_state(state_reward_struct['state'])
        state = state_reward_struct['state']
        score += state_reward_struct['reward']

        # env.render()
        done = state_reward_struct['done']
        if done:
            agent.learn()
            print(f'episode done: {i+1}\t score recieved: {score}')