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
lr_alpha       = 0.001     # entropy alpha 학습률
init_alpha     = 0.01
gamma          = 0.99
batch_size     = 32
tau            = 0.01      # target network soft update 비율
target_entropy = -1.0      # 목표 entropy (action dim = 1)
cql_alpha      = 1.0       # CQL 정규화 항의 가중치 (CQL의 핵심 파라미터)
cql_n_samples  = 10        # logsumexp 추정을 위한 action 샘플 수
n_steps        = 50000     # offline 학습 step 수 (환경 상호작용 없음)
print_interval = 1000      # 몇 step마다 평가 출력할지
video_interval = 5000      # 몇 step마다 영상 녹화할지
env_name       = "Pendulum-v1"
action_bound   = 2.0

# offline 데이터셋 수집 설정
collect_episodes = 300     # behavior policy로 수집할 에피소드 수
sac_checkpoint   = 'sac_behavior.pt'  # 학습된 SAC behavior policy 캐시 (없으면 자동 학습)

# SAC behavior policy pretraining 설정 (캐시 없을 때만 사용)
sac_pretrain_episodes = 1500
sac_warm_up           = 1000
sac_train_repeat      = 20

class ReplayBuffer():
    """미리 채워두고 '고정된 데이터셋'으로 사용 (offline RL)"""
    def __init__(self):
        self.buffer = collections.deque(maxlen=200000)

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

    def train_net(self, target, mini_batch, pi):
        s, a, r, s_prime, done = mini_batch

        # ── 1. 표준 critic loss (SAC와 동일) ──
        td_loss = F.smooth_l1_loss(self.forward(s, a), target)

        # ── 2. CQL 정규화 항 (CQL의 핵심) ──
        # 강의노트: α·( logΣexp Q(s,a) - E_{(s,a)~D}[Q(s,a)] )
        #   logsumexp 항 : 모든 행동의 Q를 누른다 (특히 OOD 행동)
        #   data 항      : 데이터에 있는 행동의 Q는 올린다
        batch_n = s.shape[0]

        # (a) 랜덤 action들의 Q값 (OOD 포함, 행동 공간 전체를 훑음)
        rand_actions = torch.empty(batch_n, cql_n_samples, 1).uniform_(-action_bound, action_bound)
        # (b) 현재 정책이 뽑은 action들의 Q값
        s_rep = s.unsqueeze(1).repeat(1, cql_n_samples, 1)          # (batch, n, 3)
        pi_actions, _ = pi(s_rep.reshape(-1, 3))
        pi_actions = pi_actions.reshape(batch_n, cql_n_samples, 1)

        # 각 후보 action에 대한 Q값 계산
        q_rand = self.forward(s_rep.reshape(-1, 3), rand_actions.reshape(-1, 1)).reshape(batch_n, cql_n_samples)
        q_pi   = self.forward(s_rep.reshape(-1, 3), pi_actions.reshape(-1, 1)).reshape(batch_n, cql_n_samples)

        # logsumexp로 "모든 행동의 Q를 누르는" 항
        cat_q = torch.cat([q_rand, q_pi], dim=1)                    # (batch, 2n)
        logsumexp_q = torch.logsumexp(cat_q, dim=1, keepdim=True)

        # 데이터에 있는 행동의 Q는 올리는 항
        data_q = self.forward(s, a)

        cql_term = (logsumexp_q - data_q).mean()

        # ── 최종 loss = 표준 critic + α·CQL ──
        loss = td_loss + cql_alpha * cql_term

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

def train_sac_behavior_policy():
    """
    cql.py가 self-contained로 동작하도록, 내장된 SAC 구성요소(PolicyNet/QNet/calc_target)를
    재사용해 behavior policy를 직접 학습한다.
    CQL 정규화는 빼고 평범한 SAC critic loss(smooth_l1)만 사용.
    """
    print(f"  SAC behavior policy 학습 시작 ({sac_pretrain_episodes} episodes, ~5-10분)")
    env = gym.make(env_name, max_episode_steps=200)
    sac_buf = ReplayBuffer()  # SAC 자체 버퍼 (offline 데이터셋과 분리)

    q1, q2 = QNet(lr_q), QNet(lr_q)
    q1_target, q2_target = QNet(lr_q), QNet(lr_q)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())
    pi = PolicyNet(lr_pi)

    def sac_critic_step(q_net, target, batch):
        """평범한 SAC critic 학습 step (CQL 정규화 없음)"""
        s, a, _, _, _ = batch
        loss = F.smooth_l1_loss(q_net(s, a), target)
        q_net.optimizer.zero_grad()
        loss.mean().backward()
        q_net.optimizer.step()

    score_window = 0.0
    for n_epi in range(sac_pretrain_episodes):
        s, _ = env.reset()
        done = False
        count = 0
        ep_score = 0.0
        while count < 200 and not done:
            a, _ = pi(torch.from_numpy(s).float())
            s_prime, r, done, truncated, _ = env.step([2.0 * a.item()])
            sac_buf.put((s, a.item(), r/10.0, s_prime, done))
            ep_score += r
            s = s_prime
            count += 1

        score_window += ep_score
        if sac_buf.size() > sac_warm_up:
            for _ in range(sac_train_repeat):
                mb = sac_buf.sample(batch_size)
                td_target = calc_target(pi, q1_target, q2_target, mb)
                sac_critic_step(q1, td_target, mb)
                sac_critic_step(q2, td_target, mb)
                pi.train_net(q1, q2, mb)
                q1.soft_update(q1_target)
                q2.soft_update(q2_target)

        if (n_epi + 1) % 100 == 0:
            avg = score_window / 100.0
            print(f"  SAC pretrain epi {n_epi+1}/{sac_pretrain_episodes}, avg score : {avg:.1f}, alpha : {pi.log_alpha.exp().item():.4f}")
            score_window = 0.0

    env.close()
    return pi


def collect_offline_data(memory, sac_pi=None):
    """
    behavior policy로 Pendulum을 돌려서 offline 데이터셋을 만든다.
    sac_pi가 None이면 무작위 정책, 아니면 학습된 SAC 정책을 사용.
    offline RL은 이 '미리 고정된 데이터'만 가지고 학습한다.
    """
    env = gym.make(env_name, max_episode_steps=200)
    total_r = 0.0
    for ep in range(collect_episodes):
        s, _ = env.reset()
        done = False
        count = 0
        while count < 200 and not done:
            if sac_pi is None:
                a_scaled = np.random.uniform(-action_bound, action_bound)
            else:
                with torch.no_grad():
                    a, _ = sac_pi(torch.from_numpy(s).float())
                a_scaled = 2.0 * a.item()  # SAC tanh 출력 [-1,1] → [-2,2]
            s_prime, r, done, truncated, _ = env.step([a_scaled])
            memory.put((s, a_scaled, r/10.0, s_prime, done or truncated))
            total_r += r
            s = s_prime
            count += 1
    env.close()
    behavior_name = "SAC" if sac_pi is not None else "random"
    print(f"offline 데이터 수집 완료: {memory.size()} transitions ({collect_episodes} episodes)")
    print(f"{behavior_name} behavior policy 평균 reward/step: {total_r/memory.size():.3f}")

def record_video(pi, step, writer):
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
    writer.add_video('pendulum', torch.from_numpy(video), step, fps=30)

def evaluate(pi, n_eval=5):
    """환경에서 정책 성능 평가 (학습엔 사용 안 함, 모니터링용)"""
    env = gym.make(env_name, max_episode_steps=200)
    total = 0.0
    for _ in range(n_eval):
        s, _ = env.reset()
        done = False
        count = 0
        while count < 200 and not done:
            with torch.no_grad():
                a, _ = pi(torch.from_numpy(s).float())
            s, r, done, truncated, _ = env.step([2.0 * a.item()])
            done = done or truncated
            total += r
            count += 1
    env.close()
    return total / n_eval

def main():
    memory = ReplayBuffer()

    # ── Step 0a: SAC behavior policy 준비 (캐시 우선, 없으면 직접 학습) ──
    if os.path.exists(sac_checkpoint):
        print(f"[1/3] SAC behavior policy 캐시 로드: {sac_checkpoint}")
        sac_pi = PolicyNet(lr_pi)
        sac_pi.load_state_dict(torch.load(sac_checkpoint))
    else:
        print(f"[1/3] SAC behavior policy 캐시 없음 → 직접 학습 (최초 1회만)")
        sac_pi = train_sac_behavior_policy()
        torch.save(sac_pi.state_dict(), sac_checkpoint)
        print(f"  SAC checkpoint saved: {sac_checkpoint}")

    # ── Step 0b: offline 데이터셋 수집 (한 번만, 이후 고정) ──
    print(f"\n[2/3] 학습된 SAC로 offline 데이터 수집 ({collect_episodes} epi)...")
    collect_offline_data(memory, sac_pi=sac_pi)
    print(f"\n[3/3] CQL 학습 시작 ({n_steps} steps)...")

    q1, q2 = QNet(lr_q), QNet(lr_q)
    q1_target, q2_target = QNet(lr_q), QNet(lr_q)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())
    pi = PolicyNet(lr_pi)

    scores = []

    log_dir = os.path.join(os.path.expanduser('~'), 'tb_logs', f'CQL_{time.time():.0f}')
    writer = SummaryWriter(log_dir)

    # ── Step 1~: 고정된 데이터셋으로만 학습 (환경 상호작용 없음) ──
    for step in range(n_steps):
        mini_batch = memory.sample(batch_size)
        td_target = calc_target(pi, q1_target, q2_target, mini_batch)
        q1.train_net(td_target, mini_batch, pi)   # CQL 정규화 포함
        q2.train_net(td_target, mini_batch, pi)
        pi.train_net(q1, q2, mini_batch)
        q1.soft_update(q1_target)
        q2.soft_update(q2_target)

        if step % print_interval == 0 and step != 0:
            score = evaluate(pi)
            print("# of step :{}, eval score : {:.1f}, alpha : {:.4f}".format(
                step, score, pi.log_alpha.exp().item()))
            writer.add_scalar('score/eval', score, step)
            writer.add_scalar('alpha', pi.log_alpha.exp().item(), step)
            scores.append(score)

        if step % video_interval == 0:
            record_video(pi, step, writer)
            print(f"  → video recorded at step {step}")

    writer.close()

    # 학습 곡선 이미지 저장
    plt.figure(figsize=(10, 5))
    plt.plot(range(print_interval, print_interval*(len(scores)+1), print_interval), scores, linewidth=2)
    plt.xlabel('Training Step')
    plt.ylabel('Eval Score')
    plt.title('CQL (Offline) on Pendulum-v1')
    plt.grid(True, alpha=0.3)
    plt.savefig('cql_result.png', dpi=150)
    plt.close()

    print("\nsaved: cql_result.png")

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