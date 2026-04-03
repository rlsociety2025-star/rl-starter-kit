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
n_rollout      = 10
n_episodes     = 300
print_interval = 20
video_interval = 300
env_name       = "CartPole-v1"

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 0):
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
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s,a,r,s_prime,done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r/100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                                               torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                                                               torch.tensor(done_lst, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)

        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1,a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

def record_video(model, episode, writer):
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
    model = ActorCritic()
    score = 0.0
    scores = []

    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(os.path.expanduser('~'), 'tb_logs', 'ActorCritic', run_name)
    writer = SummaryWriter(log_dir)

    for n_epi in range(n_episodes):
        done = False
        s, _ = env.reset()
        ep_score = 0.0

        while not done:
            for _ in range(n_rollout):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, truncated, info = env.step(a)
                model.put_data((s, a, r, s_prime, done))

                s = s_prime
                ep_score += r

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

    # 학습 곡선 이미지
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, len(scores)*print_interval, print_interval), scores, linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Avg Score')
    plt.title('Actor-Critic on CartPole-v1')
    plt.grid(True, alpha=0.3)
    plt.savefig('actor_critic_result.png', dpi=150)
    plt.close()
    print("\nsaved: actor_critic_result.png")

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
