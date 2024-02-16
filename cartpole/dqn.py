import numpy as np
import random
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam


class DQN:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.config["numStates"] = env.observation_space.shape[0]
        self.config["numActions"] = env.action_space.n
        self.config["neuralNet"] = (
            [self.config["numStates"]]
            + self.config["neuralNet"]
            + [self.config["numActions"]]
        )
        self.model = None
        self.replay = []
        self.epsilon = self.config["epsilon"][0]
        self.build_neural_net()

    def train(self):
        max_score = -float("inf")

        for episode in range(self.config["episodes"]):
            state = self.env.reset()
            score = 0
            done = False
            while not done:
                self.env.render()
                action = self.pick_action(state)
                prev_state = state
                state, reward, done, info = self.env.step(action)
                score += reward
                self.store_experience((prev_state, action, reward, state, done))
                minibatch = self.sample_replay_memory()
                if minibatch:
                    X_train, y_train = self.process_minibatch(minibatch)
                    self.mode.fit(
                        X_train,
                        y_train,
                        batch_size=self.config["batchSize"],
                        nb_epoch=1,
                        verbose=0,
                    )

            if score > max_score:
                max_score = score
                self.model.save("last_best.h5")
            print(
                f"episode {episode}, score {score} (max {max_score})\t epsilon {self.epsilon}"
            )

    def pick_action(self, state):
        if self.epsilon > self.config["epsilon"][1]:
            self.epsilon -= (
                self.config["epsilon"][0]
                - self.config["epsilon"][1] / self.config["tau"]
            )
        if random.random() < self.epsilon:
            action = np.random.randint(self.config["numActions"])
        else:
            action = np.argmax(self.model.predict(np.array([state]), batch_size=1))
        return action

    def store_experience(self, sample):
        self.replay.append(sample)
        if len(self.replay) > self.config["memory"]:
            self.replay.pop(0)

    def sample_replay_memory(self):
        if len(self.replay) > self.config["batchSize"]:
            return random.sample(self.replay, self.config["batchSize"])
        else:
            return None

    def process_minibatch(self, minibatch):
        X_train = []
        y_train = []
        for memory in minibatch:
            state, action, reward, next_state, terminal = memory

            qvalues = self.model.predict(np.array([state]), batch_size=1)
            y = np.zeros((1, self.config["numActions"]))
            y[:] = qvalues[:]
            nextQ = self.model.predict(np.array([next_state]), batch_size=1)
            if not terminal:
                value = reward + self.config["gamma"] * np.max(nextQ)
            else:
                value = reward
            y[0][action] = value

            X_train.append(
                state.reshape(
                    self.config["numStates"],
                )
            )
            y_train.append(
                y.reshape(
                    self.config["numActions"],
                )
            )
        return np.array(X_train), np.array(y_train)

    def build_neural_net(self):
        self.model = Sequential()
        layers = self.config["neuralNet"]

        self.model.add(Dense(layers[1], init="lecun_uniform", input_shape=(layers[0],)))
        self.model.add(Activation("tanh"))
        self.model.add(Dropout(0.2))

        for layer in layers[2:-1]:
            self.model.add(Dense(layer, init="lecun_uniform"))
            self.model.add(Activation("tanh"))
            self.model.add(Dropout(0.2))

        self.model.add(Dense(layers[-1], init="lecun_uniform"))
        self.model.add(Activation("linear"))
        optim = Adam(lr=5e-4)
        self.model.compiler(loss="mse", optimizer=optim)
