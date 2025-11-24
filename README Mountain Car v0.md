# Reinforcement Learning MountainCar-v0

This project implements a **neural network policy** using the **REINFORCE algorithm** to solve the OpenAI Gym `MountainCar-v0` environment. The agent learns to drive the car up the hill.

## Project Overview

### Environment
- **MountainCar-v0** (Gymnasium)
- **State space:** 2-dimensional (position, velocity)
- **Action space:** 3 discrete actions (push left, no push, push right)
- **Goal:** Reach the flag at position ≥ 0.5

### Policy Network
A neural network \(\pi_\theta(a|s)\) mapping states to action probabilities:
```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # Output probabilities
        )
Input: 2 state features

Two hidden layers with 128 neurons each

Output: Softmax probabilities over 3 actions

Action Selection
def select_action(policy, state):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action_probs = policy(state).squeeze(0)
    action_dist = torch.distributions.Categorical(probs=action_probs)
    action = action_dist.sample()
    return action.item(), action_dist.log_prob(action)


Samples actions from the policy

Returns the action and its log-probability

Trajectory Collection

Rewards are shaped to encourage reaching the goal and moving forward

Collected states, actions, log-probs, and rewards per episode

states, actions, log_probs, rewards = collect_trajectory(policy, env)

Return Computation

Discounted returns with normalization

def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = np.array(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns.tolist()

Policy Update (REINFORCE)
loss = -(log_probs * returns_tensor).sum()
optimizer.zero_grad()
loss.backward()
optimizer.step()


Updates policy parameters using the gradient of expected returns

Training

Number of episodes: 1000

Batch size for update: 4 episodes

Average return per episode is tracked and plotted

Example output (every 50 episodes):

Episode  50 | Average return (last 50): 220.3
Episode 100 | Average return (last 50): 310.5
...

Visualization

Plot shows moving average of returns over episodes:

plt.plot(moving_avg)
plt.xlabel("Episode")
plt.ylabel("Average return (per episode)")
plt.title("Training Progress")
plt.show()

Requirements

Python 3.8+

Gymnasium

PyTorch

NumPy

Matplotlib

How to Run

Install dependencies:

pip install gymnasium torch numpy matplotlib


Run the training script:

python reinforce_mountaincar.py


Observe training progress and moving average of returns.
