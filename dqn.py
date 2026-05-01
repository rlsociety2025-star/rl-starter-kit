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
import time

# ── Hyperparameters ──
learning_rate  = 0.0005
gamma          = 0.98
buffer_limit   = 50000
batch_size     = 32
train_repeat   = 10        # 에피소드 종료 후 buffer에서 몇 번 샘플링할지
warm_up        = 2000      # buffer에 이만큼 쌓일 때까지 학습 안 함
target_update  = 20        # target network hard copy 주기 (에피소드)
n_episodes     = 3000
print_interval = 20
video_interval = 300
env_name       = "CartPole-v1"

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()

def train(q, q_target, memory, optimizer):
    for i in range(train_repeat):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def record_video(q, episode, writer):
    """학습된 Q-network로 CartPole을 실행하고 TensorBoard에 기록"""
    env = gym.make(env_name, render_mode='rgb_array')
    s, _ = env.reset()
    done = False
    frames = []

    while not done:
        frame = env.render()
        frames.append(frame)
        with torch.no_grad():
            a = q.sample_action(torch.from_numpy(s).float(), epsilon=0.0)
        s, r, done, truncated, _ = env.step(a)
        done = done or truncated

    env.close()

    video = np.array(frames).transpose(0, 3, 1, 2)
    video = np.expand_dims(video, 0)
    writer.add_video('cartpole', torch.from_numpy(video), episode, fps=30)

def main():
    env = gym.make(env_name)
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    score = 0.0
    scores = []
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    log_dir = os.path.join(os.path.expanduser('~'), 'tb_logs', f'DQN_{time.time():.0f}')
    writer = SummaryWriter(log_dir)

    for n_epi in range(n_episodes):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200))
        s, _ = env.reset()
        done = False
        ep_score = 0.0

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, truncated, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r/100.0, s_prime, done_mask))
            s = s_prime
            ep_score += r
            done = done or truncated

        if memory.size() > warm_up:
            train(q, q_target, memory, optimizer)

        score += ep_score
        writer.add_scalar('score/episode', ep_score, n_epi)
        writer.add_scalar('epsilon', epsilon, n_epi)
        writer.add_scalar('buffer_size', memory.size(), n_epi)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            avg = score / print_interval
            print("# of episode :{}, avg score : {:.1f}, buffer : {}, eps : {:.1f}%".format(
                n_epi, avg, memory.size(), epsilon*100))
            writer.add_scalar('score/avg_20', avg, n_epi)
            scores.append(avg)
            score = 0.0

        if n_epi % video_interval == 0:
            record_video(q, n_epi, writer)
            print(f"  → video recorded at episode {n_epi}")

    env.close()
    writer.close()

    # 학습 곡선 이미지 저장
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, len(scores)*print_interval, print_interval), scores, linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Avg Score')
    plt.title('DQN on CartPole-v1')
    plt.grid(True, alpha=0.3)
    plt.savefig('dqn_result.png', dpi=150)
    plt.close()

    print("\nsaved: dqn_result.png")
    print("\nTensorBoard 실행:")
    print(f"  tensorboard --logdir={log_dir}")
    print("  브라우저에서 http://localhost:6006 접속")

if __name__ == '__main__':
    main()