import numpy as np
import matplotlib.pyplot as plt

gamma = 0.9  # Discount factor
threshold = 1e-4  # Convergence threshold
grid_size = 4  # 4x4 GridWorld
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

def is_valid(x, y):
    return 0 <= x < grid_size and 0 <= y < grid_size

def policy_evaluation(policy, V):
    while True:
        delta = 0
        for x in range(grid_size):
            for y in range(grid_size):
                state = (x, y)
                v = V[state]
                action = policy[state]
                (dx, dy) = actions[action]
                next_state = (x + dx, y + dy) if is_valid(x + dx, y + dy) else state
                reward = -1 if next_state != (grid_size - 1, grid_size - 1) else 0
                V[state] = reward + gamma * V[next_state]
                delta = max(delta, abs(v - V[state]))
        if delta < threshold:
            break
    return V

def policy_improvement(V):
    policy = {}
    for x in range(grid_size):
        for y in range(grid_size):
            state = (x, y)
            action_values = []
            for i, (dx, dy) in enumerate(actions):
                next_state = (x + dx, y + dy) if is_valid(x + dx, y + dy) else state
                reward = -1 if next_state != (grid_size - 1, grid_size - 1) else 0
                action_values.append((reward + gamma * V[next_state], i))
            policy[state] = max(action_values, key=lambda x: x[0])[1]
    return policy

def policy_iteration():
    V = {(x, y): 0 for x in range(grid_size) for y in range(grid_size)}
    policy = {(x, y): np.random.choice(len(actions)) for x in range(grid_size) for y in range(grid_size)}
    while True:
        V = policy_evaluation(policy, V)
        new_policy = policy_improvement(V)
        if new_policy == policy:
            break
        policy = new_policy
    return policy, V

def visualize_policy(policy):
    direction_map = ['↑', '↓', '←', '→']
    grid = np.array([[direction_map[policy[(x, y)]] for y in range(grid_size)] for x in range(grid_size)])
    print("Optimal Policy:")
    print(grid)

optimal_policy, optimal_value = policy_iteration()
visualize_policy(optimal_policy)
print("Optimal Value Function:")
for x in range(grid_size):
    print([round(optimal_value[(x, y)], 2) for y in range(grid_size)])
