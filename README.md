# Collaboration and Competition — Deep Reinforcement Learning

**Author**: Tim Wilcoxson

A self-play DDPG agent that learns to play Tennis in a Unity ML-Agents environment, where two agents cooperate to rally a ball over a net.

## Environment

- **State space**: 24 dimensions per agent (3 stacked frames × 8 variables: ball/racket position and velocity)
- **Action space**: 2 continuous actions per agent (movement toward/away from net, jumping), each in [-1, 1]
- **Agents**: 2 agents sharing one actor, one critic, and one replay buffer
- **Reward**: +0.1 for hitting the ball over the net, -0.01 if ball hits ground or goes out of bounds
- **Solve condition**: Average of max(agent1_score, agent2_score) >= 0.5 over 100 consecutive episodes

## Project Structure

| File | Description |
|---|---|
| `Tennis.ipynb` | Main training notebook with results |
| `model.py` | Actor and Critic network architectures (128-unit layers, no BatchNorm) |
| `maddpg_agent.py` | Self-play DDPG agent with exploration noise decay, OU noise, replay buffer |
| `train.py` | Standalone training script |
| `Report.md` | Detailed report: algorithm, architecture, hyperparameters, plot, future work |
| `checkpoint_actor.pth` | Trained actor weights |
| `checkpoint_critic.pth` | Trained critic weights |
| `scores.npy` | Raw per-episode scores |
| `scores_plot.png` | Training rewards plot |
| `python/` | Bundled Unity ML-Agents Python package (v0.4) |
| `Tennis.app/` | macOS Unity environment (gitignored) |

## Setup

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- macOS (this project uses the macOS Tennis.app environment)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/trwilcoxson/self-play-ddpg-tennis.git
   cd self-play-ddpg-tennis
   ```

2. Create and activate the conda environment:
   ```bash
   conda create -n drlnd-nav python=3.10 -y
   conda activate drlnd-nav
   ```

3. Install dependencies (includes PyTorch, NumPy, Jupyter, and all other required packages):
   ```bash
   cd python
   pip install .
   cd ..
   ```

4. Install the Jupyter kernel:
   ```bash
   python -m ipykernel install --user --name drlnd-nav --display-name "Python (drlnd-nav)"
   ```

5. Download the Tennis environment for your OS, unzip it, and place it in the project root:
   - [macOS](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
   - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
   - [Windows 64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
   - [Windows 32-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)

   **macOS users**: After unzipping, remove the quarantine attribute so the app can launch:
   ```bash
   xattr -cr Tennis.app
   ```

   **Linux/Windows users**: After unzipping, update the `file_name` path in the notebook's environment initialization cell to match your extracted binary (e.g., `Tennis_Linux/Tennis.x86_64` or `Tennis_Windows_x86_64/Tennis.exe`).

## Training

```bash
conda activate drlnd-nav
jupyter notebook Tennis.ipynb
```

Select the **"Python (drlnd-nav)"** kernel and run all cells. The agent typically solves the environment in 1000–2000 episodes.

## Results

See [Report.md](Report.md) for the full learning algorithm description, architecture details, training plot, and ideas for future work.
