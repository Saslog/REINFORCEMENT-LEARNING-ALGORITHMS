import numpy as np
import matplotlib.pyplot as plt

gamma = 0.9  # Discount factor
threshold = 1e-4  # Convergence threshold
capital = 100  # Maximum capital
ph = 0.4  # Probability of winning the bet

def value_iteration():
    V = np.zeros(capital + 1)
    policy = np.zeros(capital + 1)
    
    while True:
        delta = 0
        for s in range(1, capital):
            actions = np.arange(1, min(s, capital - s) + 1)
            action_values = [ph * (1 if s + a == capital else gamma * V[s + a]) + (1 - ph) * (gamma * V[s - a]) for a in actions]
            max_value = max(action_values)
            delta = max(delta, abs(V[s] - max_value))
            V[s] = max_value
            policy[s] = actions[np.argmax(action_values)]
        if delta < threshold:
            break
    return policy, V

def visualize_gambler(policy, V):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(V, label='Value Function')
    plt.xlabel('Capital')
    plt.ylabel('Value')
    plt.title('Optimal Value Function')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(policy, label='Policy')
    plt.xlabel('Capital')
    plt.ylabel('Stake')
    plt.title('Optimal Policy')
    plt.legend()
    
    plt.show()

# Solve gambler's problem
optimal_policy, optimal_value = value_iteration()
print("Optimal Policy:", optimal_policy)
print("Optimal Value Function:", optimal_value)
visualize_gambler(optimal_policy, optimal_value)