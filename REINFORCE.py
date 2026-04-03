import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import webbrowser
import time
from datetime import datetime

# ── Hyperparameters ──
learning_rate  = 0.0002
gamma          = 0.98
n_episodes     = 300
print_interval = 20
video_interval = 300
hidden_size    = 128
env_name       = "CartPole-v1"

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []

def record_video(pi, episode, writer):
    env = gym.make(env_name, render_mode='rgb_array')
    s, _ = env.reset()
    done = False
    frames = []

    while not done:
        frame = env.render()
        frames.append(frame)
        with torch.no_grad():
            prob = pi(torch.from_numpy(s).float())
        a = Categorical(prob).sample().item()
        s, r, done, truncated, _ = env.step(a)
        done = done or truncated

    env.close()

    # TensorBoard video: (1, T, 3, H, W)
    video = np.array(frames).transpose(0, 3, 1, 2)
    video = np.expand_dims(video, 0)
    writer.add_video('cartpole', torch.from_numpy(video), episode, fps=30)

def main():
    env = gym.make(env_name)
    pi = Policy()
    score = 0.0
    scores = []

    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(os.path.expanduser('~'), 'tb_logs', 'REINFORCE', run_name)
    writer = SummaryWriter(log_dir)

    for n_epi in range(n_episodes):
        s, _ = env.reset()
        done = False
        ep_score = 0.0

        while not done:
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done, truncated, info = env.step(a.item())
            pi.put_data((r, prob[a]))
            s = s_prime
            ep_score += r
            done = done or truncated

        pi.train_net()
        score += ep_score

        # 에피소드 score 기록
        writer.add_scalar('score/episode', ep_score, n_epi)

        if n_epi % print_interval == 0 and n_epi != 0:
            avg = score / print_interval
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, avg))
            writer.add_scalar('score/avg_20', avg, n_epi)
            scores.append(avg)
            score = 0.0

        # CartPole 영상 녹화
        if n_epi % video_interval == 0:
            record_video(pi, n_epi, writer)
            print(f"  → video recorded at episode {n_epi}")

    env.close()
    writer.close()

    # 학습 곡선 이미지
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, len(scores)*print_interval, print_interval), scores, linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Avg Score')
    plt.title('REINFORCE on CartPole-v1')
    plt.grid(True, alpha=0.3)
    plt.savefig('reinforce_result.png', dpi=150)
    plt.close()

    print("\nsaved: reinforce_result.png")

    # TensorBoard
    print(f"\nTensorBoard 서버를 시작합니다... (run: {run_name})")
    tb_process = subprocess.Popen(
        ['tensorboard', '--logdir_spec', f'{run_name}:{log_dir}'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    webbrowser.open('http://localhost:6006')
    print("브라우저에서 http://localhost:6006 이 열렸습니다.")
    print("종료하려면 Ctrl+C 를 누르세요.")

    try:
        tb_process.wait()
    except KeyboardInterrupt:
        tb_process.terminate()
        print("\nTensorBoard 종료.")

if __name__ == '__main__':
    main()
