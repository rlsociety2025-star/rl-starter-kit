# rl-starter-kit

강화학회 모각코 스타터 킷

## 원본 참고

강화학회 모각코의 알고리즘 코드는 노승은님(바닥부터 배우는 강화 학습의 저자, Georgia Tech PhD 과정)이 만든 교육용 RL 스크립트 기반 입니다.
(Week 8 이후 Offline RL 알고리즘은 별도 강의노트를 기반으로 합니다.)

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
├── td3.py            ← TD3 알고리즘 (Pendulum-v1)
├── sac.py            ← SAC 알고리즘 (Pendulum-v1)
└── td3_bc.py         ← TD3+BC 알고리즘 (Offline, Pendulum-v1)
```

## Online RL / Offline RL

| | Online RL (Week 1~7) | Offline RL (Week 8~) |
|---|---|---|
| 데이터 | 환경과 상호작용하며 직접 수집 | 미리 모아둔 고정 데이터셋 |
| 학습 중 환경 접근 | O | X (샘플링만) |
| 대표 알고리즘 | REINFORCE ~ SAC | TD3+BC, ... |

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

### SAC

```bash
python sac.py
```

학습 완료 후 TensorBoard로 결과 확인:

```bash
tensorboard --logdir=~/tb_logs/SAC
```

연속 제어 3부작(DDPG → TD3 → SAC)의 마지막 단계입니다. TD3의 Twin Critic을 그대로 가져오되, 탐색을 entropy로 자동 조절합니다.

- **SCALARS 탭**: 학습 곡선 + `alpha` 그래프 (탐색량이 자동 조절되는 과정)

### TD3+BC (Offline RL)

```bash
python td3_bc.py
```

학습 완료 후 TensorBoard로 결과 확인:

```bash
tensorboard --logdir=~/tb_logs/TD3BC
```

여기서부터 **Offline RL**입니다. 환경과 상호작용하지 않고, 미리 모아둔 데이터셋만으로 학습합니다.

실행은 세 단계로 진행됩니다:
1. behavior policy 학습 (SAC를 돌려 좋은 정책을 만듦, sac_behavior.pt에 캐싱)
2. 그 정책으로 Pendulum을 돌려 데이터셋 수집 (환경 상호작용은 여기까지)
3. 고정된 데이터셋만으로 TD3+BC 학습 (학습 루프에 env.step() 없음)

TD3에 BC(Behavior Cloning) 항을 더한 것이 핵심입니다. 정책이 데이터에 있던 행동에서 너무 벗어나지 못하게 제약해서, 데이터에 없는 행동(OOD)을 과대평가하는 문제를 막습니다.

- **SCALARS 탭**: `score/eval` (training step에 따른 평가 점수, x축이 episode가 아닌 step)
- **IMAGES 탭**: Pendulum 영상

데이터 품질이 좋으면(학습된 SAC로 수집) eval score가 -150 근처까지 도달해, 환경과 직접 상호작용한 SAC와 거의 동등합니다. 데이터를 무작위로 모으면 -1,200 수준에 머뭅니다. 즉 **데이터 품질이 성능의 상한을 결정**합니다.

### 알고리즘 비교

| | REINFORCE | Actor-Critic | PPO | DQN | DDPG | TD3 | SAC | TD3+BC |
|---|---|---|---|---|---|---|---|---|
| 환경 | CartPole | CartPole | CartPole | CartPole | Pendulum | Pendulum | Pendulum | Pendulum |
| Action space | Discrete | Discrete | Discrete | Discrete | Continuous | Continuous | Continuous | Continuous |
| 학습 방식 | Online | Online | Online | Online | Online | Online | Online | **Offline** |
| 계열 | Policy Gradient | Policy Gradient | Policy Gradient | Value-based | Actor-Critic | Actor-Critic | Actor-Critic | Actor-Critic |
| 정책 | 확률적 | 확률적 | 확률적 | - | 결정적 | 결정적 | 확률적 | 결정적 |
| Critic 수 | - | 1 | 1 | 1 | 1 | 2 (Twin) | 2 (Twin) | 2 (Twin) |
| 탐험 | 확률적 정책 | 확률적 정책 | 확률적 정책 | ε-greedy | OU Noise | Gaussian Noise | Entropy (자동) | (탐험 없음) |
| 핵심 추가 | - | TD error as baseline | GAE + clipped ratio | Replay Buffer + Target Network | Deterministic Policy + OU Noise | Twin Critic + Delayed + Smoothing | Max Entropy + 자동 α | **TD3 + BC 항 (정책 제약)** |

전체 TensorBoard 로그를 동시에 비교하려면:

```bash
tensorboard --logdir=~/tb_logs
```

## 개발환경

[개발환경.md](개발환경.md) 참조

## 라이선스

알고리즘 코드 원본: [minimalRL](https://github.com/seungeunrho/minimalRL) (MIT License, Copyright (c) 2019 seungeunrho)

## 링크

참고 도서: [바닥부터 배우는 강화학습](https://github.com/seungeunrho/RLfrombasics)
