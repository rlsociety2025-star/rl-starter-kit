import gymnasium as gym
import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import time
import webbrowser

# ── Hyperparameters ──
lr_pi          = 0.0005    # Actor (정책 네트워크) 학습률
lr_q           = 0.001     # Critic (Q 네트워크) 학습률
lr_alpha       = 0.001     # alpha (entropy 가중치) 학습률
init_alpha     = 0.01      # alpha 초기값
gamma          = 0.98
batch_size     = 32
buffer_limit   = 50000
tau            = 0.01      # target network soft update 비율
target_entropy = -1.0      # 목표 entropy (action dim = 1)
warm_up        = 1000      # buffer에 이만큼 쌓일 때까지 학습 안 함
train_repeat   = 20        # 에피소드 종료 후 buffer에서 몇 번 샘플링할지
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

class PolicyNet(nn.Module):
    """Actor: state → stochastic action (Gaussian) + log_prob, 그리고 자동 alpha 튜닝"""
    def __init__(self, learning_rate):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc_mu = nn.Linear(128, 1)
        self.fc_std = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # 자동 entropy 가중치 (alpha) 튜닝
        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        action = dist.rsample()  # reparameterization trick
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action)
        # tanh squashing 보정
        real_log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s, a), q2(s, a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy  # gradient ascent를 위해 음수
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        # alpha 자동 업데이트
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

class QNet(nn.Module):
    """Critic: (state, action) → Q-value"""
    def __init__(self, learning_rate):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(1, 64)
        self.fc_cat = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def train_net(self, target, mini_batch):
        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a), target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

def calc_target(pi, q1, q2, mini_batch):
    """Soft Bellman target: r + γ(min Q - α·log π)"""
    s, a, r, s_prime, done = mini_batch

    with torch.no_grad():
        a_prime, log_prob = pi(s_prime)
        entropy = -pi.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(s_prime, a_prime), q2(s_prime, a_prime)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        target = r + gamma * done * (min_q + entropy)

    return target

def record_video(pi, episode, writer):
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
            a, _ = pi(torch.from_numpy(s).float())
        s, r, done, truncated, _ = env.step([2.0 * a.item()])
        done = done or truncated
        count += 1

    env.close()

    video = np.array(frames).transpose(0, 3, 1, 2)
    video = np.expand_dims(video, 0)
    writer.add_video('pendulum', torch.from_numpy(video), episode, fps=30)

def main():
    env = gym.make(env_name, max_episode_steps=200)
    memory = ReplayBuffer()

    q1, q2 = QNet(lr_q), QNet(lr_q)
    q1_target, q2_target = QNet(lr_q), QNet(lr_q)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())
    pi = PolicyNet(lr_pi)

    score = 0.0
    scores = []

    log_dir = os.path.join(os.path.expanduser('~'), 'tb_logs', f'SAC_{time.time():.0f}')
    writer = SummaryWriter(log_dir)

    for n_epi in range(n_episodes):
        s, _ = env.reset()
        done = False
        ep_score = 0.0
        count = 0

        while count < 200 and not done:
            a, log_prob = pi(torch.from_numpy(s).float())
            s_prime, r, done, truncated, info = env.step([2.0 * a.item()])
            memory.put((s, a.item(), r/10.0, s_prime, done))
            score += r
            ep_score += r
            s = s_prime
            count += 1

        if memory.size() > warm_up:
            for i in range(train_repeat):
                mini_batch = memory.sample(batch_size)
                td_target = calc_target(pi, q1_target, q2_target, mini_batch)
                q1.train_net(td_target, mini_batch)
                q2.train_net(td_target, mini_batch)
                pi.train_net(q1, q2, mini_batch)
                q1.soft_update(q1_target)
                q2.soft_update(q2_target)

        writer.add_scalar('score/episode', ep_score, n_epi)
        writer.add_scalar('buffer_size', memory.size(), n_epi)
        writer.add_scalar('alpha', pi.log_alpha.exp().item(), n_epi)

        if n_epi % print_interval == 0 and n_epi != 0:
            avg = score / print_interval
            print("# of episode :{}, avg score : {:.1f}, buffer : {}, alpha : {:.4f}".format(
                n_epi, avg, memory.size(), pi.log_alpha.exp().item()))
            writer.add_scalar('score/avg_20', avg, n_epi)
            scores.append(avg)
            score = 0.0

        if n_epi % video_interval == 0:
            record_video(pi, n_epi, writer)
            print(f"  → video recorded at episode {n_epi}")

    env.close()
    writer.close()

    # 학습 곡선 이미지 저장
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, len(scores)*print_interval, print_interval), scores, linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Avg Score')
    plt.title('SAC on Pendulum-v1')
    plt.grid(True, alpha=0.3)
    plt.savefig('sac_result.png', dpi=150)
    plt.close()

    print("\nsaved: sac_result.png")

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