import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random

# Create the CartPole environment
env = gym.make("CartPole-v1")

# Define the DQN model
def build_model(state_size, action_size):
    model = keras.Sequential([
        keras.layers.Dense(24, activation='relu', input_shape=(state_size,)),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# Hyperparameters
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
gamma = 0.95  # Discount factor
batch_size = 32
memory = deque(maxlen=2000)

# Build the model
model = build_model(state_size, action_size)

def act(state):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    q_values = model.predict(state, verbose=0)
    return np.argmax(q_values[0])

def replay():
    global epsilon
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target += gamma * np.amax(model.predict(next_state, verbose=0)[0])
        target_f = model.predict(state, verbose=0)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Train the agent
episodes = 50
for episode in range(episodes):
    state = env.reset()[0].reshape(1, state_size)
    total_reward = 0
    for time in range(200):  # Max steps per episode
        action = act(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = next_state.reshape(1, state_size)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        if done:
            print(f"Episode: {episode+1}/{episodes}, Score: {total_reward}, Epsilon: {epsilon:.2f}")
            break
    replay()

# Test the trained model
def test_agent():
    state = env.reset()[0].reshape(1, state_size)
    total_reward = 0
    for _ in range(200):
        action = np.argmax(model.predict(state, verbose=0)[0])
        next_state, reward, done, _, _ = env.step(action)
        next_state = next_state.reshape(1, state_size)
        state = next_state
        total_reward += reward
        env.render()
        if done:
            break
    print(f"Test Score: {total_reward}")

test_agent()
env.close()
