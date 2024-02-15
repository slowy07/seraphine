import gym

NUM_INPUT = 4
NUM_ACTIONS = 2
GAMMA = 0.9

Kx = [0.5, 1]
Ka = [5, 10]

environment = gym.make("CartPole-v0")
environment.reset()

for iteration_episode in range(20):
    observation = environment.reset()
    for t in range(100):
        environment.render()
        print(observation)
        action = environment.action_space.sample()
        ux = Kx[0] * (0 - observation[0]) - Kx[1] * observation[1]
        ua = Ka[0] * (ux - observation[2]) - Ka[1] * observation[3]
        action = 1 if ua < 0 else 0
        observation, reward, done, info = environment.step(action)
