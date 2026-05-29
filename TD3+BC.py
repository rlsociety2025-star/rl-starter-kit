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
lr_mu          = 0.0005    # Actor (정책 네트워크) 학습률
lr_q           = 0.001     # Critic (Q 네트워크) 학습률
gamma          = 0.99
batch_size     = 32
tau            = 0.005     # target network soft update 비율
policy_delay   = 2         # TD3: delayed policy update
policy_noise   = 0.2       # TD3: target policy smoothing
noise_clip     = 0.5
alpha_bc       = 2.5       # BC 항의 가중치 (TD3+BC의 핵심 파라미터)
n_steps        = 50000     # offline 학습 step 수 (환경 상호작용 없음)
print_interval = 1000      # 몇 step마다 평가 출력할지
video_interval = 5000      # 몇 step마다 영상 녹화할지
env_name       = "Pendulum-v1"
action_bound   = 2.0

# offline 데이터셋 수집 설정
collect_episodes = 300     # behavior policy로 수집할 에피소드 수

# SAC behavior policy pretraining 설정
sac_pretrain_episodes = 1500
sac_lr_pi          = 0.0005
sac_lr_q           = 0.001
sac_lr_alpha       = 0.001
sac_init_alpha     = 0.01
sac_gamma          = 0.98
sac_tau            = 0.01
sac_target_entropy = -1.0
sac_train_repeat   = 20
sac_warm_up        = 1000
sac_checkpoint     = 'sac_behavior.pt'  # 학습된 SAC policy 캐시 파일

class ReplayBuffer():
    """SAC와 동일한 버퍼. 단, 여기서는 미리 채워두고 '고정된 데이터셋'으로 사용"""
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
        mu = torch.tanh(self.fc_mu(x)) * action_bound  # Pendulum action space: [-2, 2]
        return mu

class QNet(nn.Module):
    """Critic: (state, action) → Q-value"""
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(1, 64)
        self.fc_cat = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

def soft_update(net, net_target):
    """target network를 천천히 따라가도록 soft update (τ=0.005)"""
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

class SACPolicyNet(nn.Module):
    """SAC actor: behavior policy 학습용 (sac.py와 동일 구조)"""
    def __init__(self, learning_rate):
        super().__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc_mu = nn.Linear(128, 1)
        self.fc_std = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.log_alpha = torch.tensor(np.log(sac_init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=sac_lr_alpha)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(s, a), q2(s, a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        loss = -min_q - entropy
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + sac_target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


class SACQNet(nn.Module):
    """SAC critic: behavior policy 학습용"""
    def __init__(self, learning_rate):
        super().__init__()
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
            param_target.data.copy_(param_target.data * (1.0 - sac_tau) + param.data * sac_tau)


def _sac_calc_target(pi, q1, q2, mini_batch):
    s, a, r, s_prime, done = mini_batch
    with torch.no_grad():
        a_prime, log_prob = pi(s_prime)
        entropy = -pi.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(s_prime, a_prime), q2(s_prime, a_prime)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        target = r + sac_gamma * done * (min_q + entropy)
    return target


def _sac_sample(buf, n):
    mini = random.sample(buf, n)
    s_l, a_l, r_l, sp_l, dm_l = [], [], [], [], []
    for s, a, r, sp, done in mini:
        s_l.append(s); a_l.append([a]); r_l.append([r])
        sp_l.append(sp); dm_l.append([0.0 if done else 1.0])
    return (torch.tensor(s_l, dtype=torch.float), torch.tensor(a_l, dtype=torch.float),
            torch.tensor(r_l, dtype=torch.float), torch.tensor(sp_l, dtype=torch.float),
            torch.tensor(dm_l, dtype=torch.float))


def train_sac_behavior_policy():
    """sac.py와 동일한 방식으로 SAC를 학습해 behavior policy를 얻는다."""
    env = gym.make(env_name, max_episode_steps=200)
    buf = collections.deque(maxlen=50000)

    q1 = SACQNet(sac_lr_q); q2 = SACQNet(sac_lr_q)
    q1_target = SACQNet(sac_lr_q); q2_target = SACQNet(sac_lr_q)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())
    pi = SACPolicyNet(sac_lr_pi)

    score_window = 0.0
    for n_epi in range(sac_pretrain_episodes):
        s, _ = env.reset()
        done = False
        count = 0
        ep_score = 0.0
        while count < 200 and not done:
            a, _ = pi(torch.from_numpy(s).float())
            s_prime, r, done, truncated, _ = env.step([2.0 * a.item()])
            buf.append((s, a.item(), r/10.0, s_prime, done))
            ep_score += r
            s = s_prime
            count += 1

        score_window += ep_score
        if len(buf) > sac_warm_up:
            for _ in range(sac_train_repeat):
                mb = _sac_sample(buf, batch_size)
                td_target = _sac_calc_target(pi, q1_target, q2_target, mb)
                q1.train_net(td_target, mb)
                q2.train_net(td_target, mb)
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
    학습된 SAC policy(deterministic mean)로 Pendulum을 돌려 offline 데이터셋을 만든다.
    sac_pi=None이면 무작위 정책으로 폴백.
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
            memory.put((s, a_scaled, r/100.0, s_prime, done or truncated))
            total_r += r
            s = s_prime
            count += 1
    env.close()
    behavior_name = "SAC" if sac_pi is not None else "random"
    print(f"offline 데이터 수집 완료: {memory.size()} transitions ({collect_episodes} episodes)")
    print(f"{behavior_name} behavior policy 평균 reward/step: {total_r/memory.size():.3f}")

def train(mu, mu_target, q1, q1_target, q2, q2_target,
          memory, q1_optimizer, q2_optimizer, mu_optimizer, step):
    s, a, r, s_prime, done_mask = memory.sample(batch_size)

    # ── Critic update (TD3와 동일: Twin Q + target smoothing) ──
    with torch.no_grad():
        noise = (torch.randn_like(a) * policy_noise).clamp(-noise_clip, noise_clip)
        a_target = (mu_target(s_prime) + noise).clamp(-action_bound, action_bound)
        target_q = torch.min(q1_target(s_prime, a_target), q2_target(s_prime, a_target))
        target = r + gamma * target_q * done_mask

    q1_loss = F.smooth_l1_loss(q1(s, a), target)
    q1_optimizer.zero_grad(); q1_loss.backward(); q1_optimizer.step()
    q2_loss = F.smooth_l1_loss(q2(s, a), target)
    q2_optimizer.zero_grad(); q2_loss.backward(); q2_optimizer.step()

    # ── Actor update (TD3+BC의 핵심) ──
    # 강의노트 수식: π = argmax E[ λ·Q(s,π(s)) - (π(s)-a)² ]
    #   λ·Q(s,π(s)) : 기존 TD3 (Q를 높이는 방향)
    #   (π(s)-a)²   : BC 항 (데이터의 행동 a에서 벗어나지 못하게 제약)
    if step % policy_delay == 0:
        pi = mu(s)
        q_val = q1(s, pi)
        # lambda normalization (Fujimoto 2021): Q 스케일에 맞춰 BC 가중치 자동 조정
        lmbda = alpha_bc / q_val.abs().mean().detach()
        mu_loss = -lmbda * q_val.mean() + F.mse_loss(pi, a)

        mu_optimizer.zero_grad(); mu_loss.backward(); mu_optimizer.step()
        soft_update(mu, mu_target)
        soft_update(q1, q1_target)
        soft_update(q2, q2_target)

def record_video(mu, step, writer):
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
    writer.add_video('pendulum', torch.from_numpy(video), step, fps=30)

def evaluate(mu, n_eval=5):
    """환경에서 정책 성능 평가 (학습엔 사용 안 함, 모니터링용)"""
    env = gym.make(env_name, max_episode_steps=200)
    total = 0.0
    for _ in range(n_eval):
        s, _ = env.reset()
        done = False
        count = 0
        while count < 200 and not done:
            with torch.no_grad():
                a = mu(torch.from_numpy(s).float()).item()
            s, r, done, truncated, _ = env.step([a])
            done = done or truncated
            total += r
            count += 1
    env.close()
    return total / n_eval

def main():
    memory = ReplayBuffer()

    # ── Step 0a: SAC behavior policy 준비 (캐시 우선) ──
    sac_pi = SACPolicyNet(sac_lr_pi)
    if os.path.exists(sac_checkpoint):
        print(f"[1/3] SAC behavior policy 캐시 로드: {sac_checkpoint}")
        sac_pi.load_state_dict(torch.load(sac_checkpoint))
    else:
        print(f"[1/3] SAC behavior policy 학습 ({sac_pretrain_episodes} epi)...")
        sac_pi = train_sac_behavior_policy()
        torch.save(sac_pi.state_dict(), sac_checkpoint)
        print(f"  SAC checkpoint saved: {sac_checkpoint}")

    # ── Step 0b: offline 데이터셋 수집 (한 번만, 이후 고정) ──
    print(f"\n[2/3] 학습된 SAC로 offline 데이터 수집 ({collect_episodes} epi)...")
    collect_offline_data(memory, sac_pi=sac_pi)
    print(f"\n[3/3] TD3+BC 학습 시작 ({n_steps} steps)...")

    q1, q1_target = QNet(), QNet()
    q1_target.load_state_dict(q1.state_dict())
    q2, q2_target = QNet(), QNet()
    q2_target.load_state_dict(q2.state_dict())
    mu, mu_target = MuNet(), MuNet()
    mu_target.load_state_dict(mu.state_dict())

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q1_optimizer = optim.Adam(q1.parameters(), lr=lr_q)
    q2_optimizer = optim.Adam(q2.parameters(), lr=lr_q)

    scores = []

    log_dir = os.path.join(os.path.expanduser('~'), 'tb_logs', f'TD3BC_{time.time():.0f}')
    writer = SummaryWriter(log_dir)

    # ── Step 1~: 고정된 데이터셋으로만 학습 (환경 상호작용 없음) ──
    for step in range(n_steps):
        train(mu, mu_target, q1, q1_target, q2, q2_target,
              memory, q1_optimizer, q2_optimizer, mu_optimizer, step)

        if step % print_interval == 0 and step != 0:
            score = evaluate(mu)
            print("# of step :{}, eval score : {:.1f}".format(step, score))
            writer.add_scalar('score/eval', score, step)
            scores.append(score)

        if step % video_interval == 0:
            record_video(mu, step, writer)
            print(f"  → video recorded at step {step}")

    writer.close()

    # 학습 곡선 이미지 저장
    plt.figure(figsize=(10, 5))
    plt.plot(range(print_interval, print_interval*(len(scores)+1), print_interval), scores, linewidth=2)
    plt.xlabel('Training Step')
    plt.ylabel('Eval Score')
    plt.title('TD3+BC (Offline) on Pendulum-v1')
    plt.grid(True, alpha=0.3)
    plt.savefig('td3bc_result.png', dpi=150)
    plt.close()

    print("\nsaved: td3bc_result.png")

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