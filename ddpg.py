import gymnasium as gym
import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import time
import webbrowser

# ── Hyperparameters ──
lr_mu          = 0.0005    # Actor (정책 네트워크) 학습률
lr_q           = 0.001     # Critic (Q 네트워크) 학습률
gamma          = 0.99
batch_size     = 32
buffer_limit   = 50000
tau            = 0.005     # target network soft update 비율
warm_up        = 2000      # buffer에 이만큼 쌓일 때까지 학습 안 함
train_repeat   = 10        # 에피소드 종료 후 buffer에서 몇 번 샘플링할지
n_episodes     = 10000
print_interval = 20
video_interval = 300
env_name       = "Pendulum-v1"

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
               torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)

class MuNet(nn.Module):
    """Actor: state → continuous action [-2, 2]"""
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) * 2  # Pendulum의 action space: [-2, 2]
        return mu

class QNet(nn.Module):
    """Critic: (state, action) → Q-value"""
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(1, 64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q

class OrnsteinUhlenbeckNoise:
    """탐색을 위한 mean-reverting noise (Gaussian 대비 관성 있는 탐색)"""
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    s, a, r, s_prime, done_mask = memory.sample(batch_size)

    # Critic update: target = r + γ * Q_target(s', μ_target(s'))
    target = r + gamma * q_target(s_prime, mu_target(s_prime)) * done_mask
    q_loss = F.smooth_l1_loss(q(s, a), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()

    # Actor update: maximize Q(s, μ(s)) → minimize -Q(s, μ(s))
    mu_loss = -q(s, mu(s)).mean()
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()

def soft_update(net, net_target):
    """target network를 천천히 따라가도록 soft update (τ=0.005)"""
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

def record_video(mu, episode, writer):
    """학습된 Actor로 Pendulum을 실행하고 TensorBoard에 기록"""
    env = gym.make(env_name, render_mode='rgb_array')
    s, _ = env.reset()
    done = False
    frames = []
    count = 0

    while count < 200 and not done:
        frame = env.render()
        frames.append(frame)
        with torch.no_grad():
            a = mu(torch.from_numpy(s).float()).item()
        s, r, done, truncated, _ = env.step([a])
        done = done or truncated
        count += 1

    env.close()

    video = np.array(frames).transpose(0, 3, 1, 2)
    video = np.expand_dims(video, 0)
    writer.add_video('pendulum', torch.from_numpy(video), episode, fps=30)

def main():
    env = gym.make(env_name, max_episode_steps=200)
    memory = ReplayBuffer()

    q, q_target = QNet(), QNet()
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(), MuNet()
    mu_target.load_state_dict(mu.state_dict())

    score = 0.0
    scores = []

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer  = optim.Adam(q.parameters(), lr=lr_q)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    log_dir = os.path.join(os.path.expanduser('~'), 'tb_logs', f'DDPG_{time.time():.0f}')
    writer = SummaryWriter(log_dir)

    for n_epi in range(n_episodes):
        s, _ = env.reset()
        done = False
        ep_score = 0.0
        count = 0

        while count < 200 and not done:
            a = mu(torch.from_numpy(s).float())
            a = a.item() + ou_noise()[0]
            s_prime, r, done, truncated, info = env.step([a])
            memory.put((s, a, r/100.0, s_prime, done))
            score += r
            ep_score += r
            s = s_prime
            count += 1

        if memory.size() > warm_up:
            for i in range(train_repeat):
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                soft_update(mu, mu_target)
                soft_update(q, q_target)

        writer.add_scalar('score/episode', ep_score, n_epi)
        writer.add_scalar('buffer_size', memory.size(), n_epi)

        if n_epi % print_interval == 0 and n_epi != 0:
            avg = score / print_interval
            print("# of episode :{}, avg score : {:.1f}, buffer : {}".format(
                n_epi, avg, memory.size()))
            writer.add_scalar('score/avg_20', avg, n_epi)
            scores.append(avg)
            score = 0.0

        if n_epi % video_interval == 0:
            record_video(mu, n_epi, writer)
            print(f"  → video recorded at episode {n_epi}")

    env.close()
    writer.close()

    # 학습 곡선 이미지 저장
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, len(scores)*print_interval, print_interval), scores, linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Avg Score')
    plt.title('DDPG on Pendulum-v1')
    plt.grid(True, alpha=0.3)
    plt.savefig('ddpg_result.png', dpi=150)
    plt.close()

    print("\nsaved: ddpg_result.png")

    print(f"\nTensorBoard 서버를 시작합니다... (logdir: {log_dir})")
    tb_process = subprocess.Popen(
        ['tensorboard', '--logdir', log_dir],
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