# Reinforcement Learning MountainCar-v0

This project implements a **neural network policy** using the **REINFORCE algorithm** to solve the OpenAI Gym `MountainCar-v0` environment. The agent learns to drive the car up the hill.

## Project Overview

1. **Environment**
- **MountainCar-v0** (Gymnasium)
- **State space:** 2-dimensional (position, velocity)
- **Action space:** 3 discrete actions (push left, no push, push right)
- **Goal:** Reach the flag at position ≥ 0.5

2.** Policy Network**
A neural network \(\pi_\theta(a|s)\) mapping states to action probabilities:
-**Input**: 2 state features
-**Two hidden layers** with 128 neurons each
-**Output**: Softmax probabilities over 3 actions

3.**Action Selection**
-Samples actions from the policy
-Returns the action and its log-probability

4.**Trajectory Collection**
-Rewards are shaped to encourage reaching the goal and moving forward
-Collected states, actions, log-probs, and rewards per episode

5.**Return Computation**
-Discounted returns with normalization

6.**Policy Update (REINFORCE)**
-Updates policy parameters using the gradient of expected returns

7.**Training**
-Number of episodes: 1000
-Batch size for update: 4 episodes
-Average return per episode is tracked and plotted

8.**Visualization**
-Plot shows moving average of returns over episodes

**Requirements**
-Python 3.8+
-Gymnasium
-PyTorch
-NumPy
-Matplotlib

**Install dependencies:**
pip install gymnasium torch numpy matplotlib


**Run the script:**
python reinforce_mountaincar.py


Observe training progress and moving average of returns.
