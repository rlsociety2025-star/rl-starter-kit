# rl-starter-kit

강화학회 모각코 스타터 킷

## 원본 참고

강화학회 모각코의 알고리즘 코드는 노승은님(바닥부터 배우는 강화 학습의 저자, Georgia Tech PhD 과정)이 만든 교육용 RL 스크립트 기반 입니다.

## 구조

```
rl-starter-kit/
├── README.md
├── 개발환경.md
├── REINFORCE.py      ← REINFORCE 알고리즘 (CartPole-v1)
├── actor_critic.py   ← Actor-Critic 알고리즘 (CartPole-v1)
├── ppo.py            ← PPO 알고리즘 (CartPole-v1)
├── dqn.py            ← DQN 알고리즘 (CartPole-v1)
├── ddpg.py           ← DDPG 알고리즘 (Pendulum-v1)
└── td3.py            ← TD3 알고리즘 (Pendulum-v1)
```

## 시작하기

```bash
pip install gymnasium torch matplotlib numpy tensorboard pygame moviepy
```

## 실행

### REINFORCE

```bash
python REINFORCE.py
```

학습 완료 후 TensorBoard로 결과 확인:

```bash
tensorboard --logdir=~/tb_logs/REINFORCE
```

브라우저에서 http://localhost:6006 접속

- **SCALARS 탭**: 학습 곡선
- **IMAGES 탭**: CartPole 영상 (학습 전/후 비교)

### Actor-Critic

```bash
python actor_critic.py
```

학습 완료 후 TensorBoard로 결과 확인:

```bash
tensorboard --logdir=~/tb_logs/ActorCritic
```

### PPO

```bash
python ppo.py
```

학습 완료 후 TensorBoard로 결과 확인:

```bash
tensorboard --logdir=~/tb_logs/PPO
```

### DQN

```bash
python dqn.py
```

학습 완료 후 TensorBoard로 결과 확인:

```bash
tensorboard --logdir=~/tb_logs/DQN
```

### DDPG

```bash
python ddpg.py
```

학습 완료 후 TensorBoard로 결과 확인:

```bash
tensorboard --logdir=~/tb_logs/DDPG
```

브라우저에서 http://localhost:6006 접속

- **SCALARS 탭**: 학습 곡선 (Pendulum reward는 음수, -200에 가까울수록 잘 학습됨)
- **IMAGES 탭**: Pendulum 영상 (학습 후엔 막대가 위쪽에 서 있어야 함)

### TD3

```bash
python td3.py
```

학습 완료 후 TensorBoard로 결과 확인:

```bash
tensorboard --logdir=~/tb_logs/TD3
```

DDPG와 같은 Pendulum 환경에서 동작합니다. DDPG와 겹쳐서 보면 학습 안정성의 차이가 명확합니다:

```bash
tensorboard --logdir=~/tb_logs
```

### 알고리즘 비교

| | REINFORCE | Actor-Critic | PPO | DQN | DDPG | TD3 |
|---|---|---|---|---|---|---|
| 환경 | CartPole | CartPole | CartPole | CartPole | Pendulum | Pendulum |
| Action space | Discrete (2) | Discrete (2) | Discrete (2) | Discrete (2) | Continuous [-2, 2] | Continuous [-2, 2] |
| 계열 | Policy Gradient | Policy Gradient | Policy Gradient | Value-based | Actor-Critic (Off-policy) | Actor-Critic (Off-policy) |
| Critic 수 | - | 1 | 1 | 1 (Q) | 1 | **2 (Twin Q)** |
| 네트워크 | Policy만 | Policy + Value | Policy + Value | Q-network + Target Q | Actor + Critic (각각 Target) | Actor + Twin Critic (각각 Target) |
| 학습 방식 | On-policy | On-policy | On-policy | Off-policy (Replay Buffer) | Off-policy (Replay Buffer) | Off-policy (Replay Buffer) |
| 탐험 | 확률적 정책 | 확률적 정책 | 확률적 정책 | ε-greedy | OU Noise | Gaussian Noise |
| Target update | - | - | - | Hard copy (20 epi) | Soft update (τ=0.005) | Soft update (τ=0.005, delayed) |
| Actor 업데이트 | 매번 | 매번 | 매번 | - | 매번 | **2 step마다 (Delayed)** |
| 데이터 재사용 | 1회 | 1회 | K회 (K_epoch=3) | Buffer 반복 | Buffer 반복 | Buffer 반복 |
| 핵심 추가 | - | TD error as baseline | GAE + clipped ratio | Replay Buffer + Target Network | Deterministic Policy + OU Noise | Twin Critic + Delayed Update + Target Smoothing |

여섯 알고리즘의 TensorBoard 로그를 동시에 비교하려면:

```bash
tensorboard --logdir=~/tb_logs
```

## 개발환경

[개발환경.md](개발환경.md) 참조

## 라이선스

알고리즘 코드 원본: [minimalRL](https://github.com/seungeunrho/minimalRL) (MIT License, Copyright (c) 2019 seungeunrho)

## 링크

참고 도서: [바닥부터 배우는 강화학습](https://github.com/seungeunrho/RLfrombasics)
