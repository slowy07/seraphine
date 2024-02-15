import gym
from gym import wrappers
import numpy as np
import random
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop

EPISODE_DURATION = 200


def train(environment, model, config):
    replay = []
    epsilon = config["epsilon"]
    max_duration = 0
    for episode in range(config["episode"]):
        state = environment.reset()
        duration = 0
        done = False
        while not done:
            duration += 1
            environment.render()
            action, epsilon = pick_action(state, model, epsilon, config)
            prev_state = state
            state, reward, done, info = environment.step(action)
            terminal = done and not (duration == EPISODE_DURATION)

            store_experience(
                replay, (prev_state, action, reward, state, terminal), config["memory"]
            )
            minibatch = sample_replay_memory(replay, config["batchSize"])

            if minibatch:
                X_train, y_train = process_minibatch(minibatch, model, config)
                model.fit(
                    X_train,
                    y_train,
                    batch_size=config["batchSize"],
                    nb_epoch=1,
                    verbose=0,
                )

        if duration > max_duration:
            max_duration = duration
            model.save("last_best.h5")
        print(
            f"episode: {episode} duration {duration} (max: {max_duration})\t epsilon {epsilon}"
        )


def pick_action(state, model, epsilon, config):
    epsilon -= epsilon / config["tau"]
    if random.random() < epsilon:
        action = np.random.randint(config["numActions"])
    else:
        action = np.argmax(model.predict(np.array([state]), batch_size=1))
    return action, epsilon


def store_experience(replay, sample, memory):
    replay.append(sample)
    if len(replay) > memory:
        replay.pop(0)


def sample_replay_memory(replay, batch_size):
    if len(replay) > batch_size:
        return random.sample(replay, batch_size)
    else:
        return None


def process_minibatch(minibatch, model, config):
    X_train = []
    y_train = []
    for memory in minibatch:
        state, action, reward, next_state, terminal = memory

        qvalues = model.predict(np.array([state]), batch_size=1)
        y = np.zeros((1, config["numActions"]))
        y[:] = qvalues[:]

        nextQ = model.predict(np.array([next_state]), batch_size=1)
        if not terminal:
            value = reward + config["gamma"] * np.max(nextQ)
        else:
            value = reward
        y[0][action] = value
        X_train.append(
            state.reshape(
                config["numStates"],
            )
        )
        y_train.append(y.reshape(config["numActions"]))
    return np.array(X_train), np.array(y_train)


def build_neural_net(layers):
    model = Sequential()

    model.add(Dense(layers[1], init="lecun_uniform", input_shape=(layers[0],)))
    model.add(Activation("tanh"))
    model.add(Dropout(0.2))

    for layer in layers[2:-1]:
        print(layer)
        model.add(Dense(layer, init="lecun_uniform"))
        model.add(Activation("tanh"))
        model.add(Dropout(0.2))
    model.add(Dense(layers[-1], init="lecun_uniform"))
    model.add(Activation("linear"))

    rms = RMSprop()
    model.compile(loss="mse", optimizers=rms)
    return model


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = wrappers.Monitor(env, "tmp/cartpole-experiment-1", force=True)
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    config = {
        "numStates": num_states,
        "numActions": num_actions,
        "neuralNet": [num_states, 100, 100, num_actions],
        "memory": 10000,
        "batchSize": 200,
        "episodes": 150,
        "gamma": 0.9,
        "epsilon": 0.5,
        "tau": 500,
    }

    model = build_neural_net(config["neuralNet"])
    train(env, model, config)
