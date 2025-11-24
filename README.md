# MountainCar REINFORCE Agent

This project trains an AI agent to solve the OpenAI Gym `MountainCar-v0` environment using **REINFORCE**, a policy gradient algorithm.

---

## Overview

- **Environment**: MountainCar-v0  
  The car must reach the top of the hill.  
  **State**: `[position, velocity]`  
  **Actions**: push left, no push, push right  

- **Agent**: Neural network policy with two hidden layers and softmax output  
- **Algorithm**: REINFORCE (Monte Carlo policy gradient)  
- **Reward Shaping**: Added bonuses to encourage reaching the goal and building momentum  

---

## How to Run

1. Clone the repo:  
   ```bash
   git clone https://github.com/yourusername/mountaincar-reinforce.git
   cd mountaincar-reinforce

Install dependencies:

pip install gymnasium torch matplotlib numpy


Run training:

python train_mountaincar.py


The script will train the agent and plot the average return per episode.

Results

The agent learns to reach the goal, and the average return improves as training progresses.
