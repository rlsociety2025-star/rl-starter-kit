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
import time

# ── Hyperparameters ──
learning_rate  = 0.0005
gamma          = 0.98
lmbda          = 0.95      # GAE lambda
eps_clip       = 0.1       # PPO clipping range
K_epoch        = 3         # 같은 데이터로 몇 번 재사용
T_horizon      = 20        # n-step rollout 길이
n_episodes     = 300
print_interval = 20
video_interval = 300
hidden_size    = 256
env_name       = "CartPole-v1"

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc1   = nn.Linear(4, hidden_size)
        self.fc_pi = nn.Linear(hidden_size, 2)
        self.fc_v  = nn.Linear(hidden_size, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                               torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

def record_video(model, episode, writer):
    """학습된 policy로 CartPole을 실행하고 TensorBoard에 기록"""
    env = gym.make(env_name, render_mode='rgb_array')
    s, _ = env.reset()
    done = False
    frames = []

    while not done:
        frame = env.render()
        frames.append(frame)
        with torch.no_grad():
            prob = model.pi(torch.from_numpy(s).float())
        a = Categorical(prob).sample().item()
        s, r, done, truncated, _ = env.step(a)
        done = done or truncated

    env.close()

    video = np.array(frames).transpose(0, 3, 1, 2)
    video = np.expand_dims(video, 0)
    writer.add_video('cartpole', torch.from_numpy(video), episode, fps=30)

def main():
    env = gym.make(env_name)
    model = PPO()
    score = 0.0
    scores = []

    log_dir = os.path.join(os.path.expanduser('~'), 'tb_logs', f'PPO_{time.time():.0f}')
    writer = SummaryWriter(log_dir)

    for n_epi in range(n_episodes):
        s, _ = env.reset()
        done = False
        ep_score = 0.0

        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, truncated, info = env.step(a)

                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                s = s_prime
                ep_score += r
                done = done or truncated

                if done:
                    break

            model.train_net()

        score += ep_score
        writer.add_scalar('score/episode', ep_score, n_epi)

        if n_epi % print_interval == 0 and n_epi != 0:
            avg = score / print_interval
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, avg))
            writer.add_scalar('score/avg_20', avg, n_epi)
            scores.append(avg)
            score = 0.0

        if n_epi % video_interval == 0:
            record_video(model, n_epi, writer)
            print(f"  → video recorded at episode {n_epi}")

    env.close()
    writer.close()

    # 학습 곡선 이미지 저장
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, len(scores)*print_interval, print_interval), scores, linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Avg Score')
    plt.title('PPO on CartPole-v1')
    plt.grid(True, alpha=0.3)
    plt.savefig('ppo_result.png', dpi=150)
    plt.close()

    print("\nsaved: ppo_result.png")
    print("\nTensorBoard 실행:")
    print(f"  tensorboard --logdir={log_dir}")
    print("  브라우저에서 http://localhost:6006 접속")

if __name__ == '__main__':
    main()