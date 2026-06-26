# rl-starter-kit

강화학회 모각코 스타터 킷 (시즌 1)

## 원본 참고

강화학회 모각코의 알고리즘 코드는 노승은님(바닥부터 배우는 강화 학습의 저자, Georgia Tech PhD 과정)이 만든 교육용 RL 스크립트 기반 입니다.
(Week 8 이후 Offline RL 알고리즘은 별도 강의노트를 기반으로 합니다.)

## 커리큘럼 (10주)

| Week | 알고리즘 | 환경 | 분류 |
|---|---|---|---|
| 1 | REINFORCE | CartPole | Online · Policy Gradient |
| 2 | Actor-Critic | CartPole | Online · Policy Gradient |
| 3 | PPO | CartPole | Online · Policy Gradient |
| 4 | DQN | CartPole | Online · Value-based |
| 5 | DDPG | Pendulum | Online · Actor-Critic (연속) |
| 6 | TD3 | Pendulum | Online · Actor-Critic (연속) |
| 7 | SAC | Pendulum | Online · Actor-Critic (연속) |
| 8 | TD3+BC | Pendulum | **Offline** · 정책 제약 |
| 9 | CQL | Pendulum | **Offline** · 가치 제약 |
| 10 | IQL | Pendulum | **Offline** · OOD 회피 |

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
├── td3_bc.py         ← TD3+BC 알고리즘 (Offline, Pendulum-v1)
├── cql.py            ← CQL 알고리즘 (Offline, Pendulum-v1)
└── iql.py            ← IQL 알고리즘 (Offline, Pendulum-v1)
```

## Online RL / Offline RL

| | Online RL (Week 1~7) | Offline RL (Week 8~10) |
|---|---|---|
| 데이터 | 환경과 상호작용하며 직접 수집 | 미리 모아둔 고정 데이터셋 |
| 학습 중 환경 접근 | O | X (샘플링만) |
| 대표 알고리즘 | REINFORCE ~ SAC | TD3+BC, CQL, IQL |

Offline RL은 OOD(데이터에 없는 행동) 과대평가가 핵심 난제이며, 해결 방식에 따라 나뉩니다:
- **TD3+BC**: 정책을 제약 (Actor 쪽) — 데이터 행동에서 벗어나지 못하게
- **CQL**: 가치를 제약 (Critic 쪽) — OOD 행동의 Q값을 낮게 누름
- **IQL**: OOD를 회피 — 데이터에 있는 행동만 평가, OOD는 쳐다보지 않음

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

```bash
tensorboard --logdir=~/tb_logs/ActorCritic
```

### PPO

```bash
python ppo.py
```

```bash
tensorboard --logdir=~/tb_logs/PPO
```

### DQN

```bash
python dqn.py
```

```bash
tensorboard --logdir=~/tb_logs/DQN
```

### DDPG

```bash
python ddpg.py
```

```bash
tensorboard --logdir=~/tb_logs/DDPG
```

- **SCALARS 탭**: 학습 곡선 (Pendulum reward는 음수, -200에 가까울수록 잘 학습됨)
- **IMAGES 탭**: Pendulum 영상 (학습 후엔 막대가 위쪽에 서 있어야 함)

### TD3

```bash
python td3.py
```

```bash
tensorboard --logdir=~/tb_logs/TD3
```

DDPG와 같은 Pendulum 환경에서 동작합니다. DDPG와 겹쳐서 보면 학습 안정성의 차이가 명확합니다.

### SAC

```bash
python sac.py
```

```bash
tensorboard --logdir=~/tb_logs/SAC
```

연속 제어 3부작(DDPG → TD3 → SAC)의 마지막 단계입니다. TD3의 Twin Critic을 그대로 가져오되, 탐색을 entropy로 자동 조절합니다.

- **SCALARS 탭**: 학습 곡선 + `alpha` 그래프 (탐색량이 자동 조절되는 과정)

### TD3+BC (Offline RL)

```bash
python td3_bc.py
```

```bash
tensorboard --logdir=~/tb_logs/TD3BC
```

여기서부터 **Offline RL**입니다. 환경과 상호작용하지 않고, 미리 모아둔 데이터셋만으로 학습합니다.

실행은 세 단계로 진행됩니다:
1. behavior policy 학습 (SAC를 돌려 좋은 정책을 만듦, sac_behavior.pt에 캐싱)
2. 그 정책으로 Pendulum을 돌려 데이터셋 수집 (환경 상호작용은 여기까지)
3. 고정된 데이터셋만으로 TD3+BC 학습 (학습 루프에 env.step() 없음)

TD3에 BC(Behavior Cloning) 항을 더해, 정책이 데이터 행동에서 벗어나지 못하게 제약합니다. (정책 제약 방식)
데이터 품질이 성능의 상한을 결정합니다 (SAC 데이터 -150, 무작위 데이터 -1,200).

### CQL (Offline RL)

```bash
python cql.py
```

```bash
tensorboard --logdir=~/tb_logs/CQL
```

TD3+BC가 **정책**을 제약했다면, CQL은 **가치(Q함수)**를 제약합니다.
TD3+BC와 동일한 3단계(SAC behavior policy로 데이터 수집)로 진행되며, 같은 데이터셋을 써서 정책 제약 vs 가치 제약을 같은 조건에서 비교할 수 있습니다.

베이스는 SAC이고, Q-loss에 CQL 정규화 항(logsumexp 항으로 OOD 행동 Q를 누르고, 데이터 행동 Q를 올림)을 더한 것이 핵심입니다. `cql_alpha`가 보수성의 강도를 결정합니다.

### IQL (Offline RL)

```bash
python iql.py
```

```bash
tensorboard --logdir=~/tb_logs/IQL
```

오프라인 RL의 완성형입니다. TD3+BC(정책 제약), CQL(가치 누르기)과 달리, IQL은 **OOD 행동을 아예 평가하지 않습니다**. 데이터에 있는 행동만으로 학습합니다.

TD3+BC / CQL과 동일한 3단계(SAC behavior policy로 데이터 수집)로 진행되어, 세 방법을 같은 데이터셋에서 비교할 수 있습니다.

세 네트워크가 협력합니다 (Q1, Q2, V, Policy):
1. **Value 학습 (Expectile regression)**: 비대칭 손실로 "데이터 안 행동 중 좋은 쪽"을 추정. OOD를 보지 않고 잠재 최대치를 가늠.
2. **Q 학습**: 다음 상태의 V(s')만 사용 → OOD 평가를 원천 차단 (CQL이 min Q를 쓴 것과 대비)
3. **정책 추출 (AWR)**: advantage(Q-V)가 큰 행동을 더 강하게 모방

`expectile`(0.5=평균, 1=낙관)과 `beta`(AWR 온도)가 튜닝 포인트입니다. 강의노트에서 실전 성능이 가장 좋다고 평가한 방법입니다.

### 알고리즘 비교

| | TD3+BC | CQL | IQL |
|---|---|---|---|
| 학습 방식 | Offline | Offline | Offline |
| 베이스 | TD3 | SAC | (V + Q + AWR) |
| 정책 학습 | TD3 + BC 항 | SAC | AWR (advantage 가중) |
| OOD 처리 | 정책 제약 | 가치 누르기 | **평가 회피** |
| 추가 네트워크 | - | - | **V network** |
| 튜닝 포인트 | alpha_bc | cql_alpha | expectile, beta |
| 핵심 | `(π(s)-a)²` | `L_CQL` (logsumexp) | Expectile + AWR |

전체 TensorBoard 로그를 동시에 비교하려면:

```bash
tensorboard --logdir=~/tb_logs
```

## 시즌 1 마무리

REINFORCE부터 IQL까지, 10주간 온라인 RL과 오프라인 RL의 핵심 알고리즘을 코드로 직접 돌려보며 달려왔습니다. 이 스타터킷은 강화학습의 기본기를 다지는 토대입니다.

**시즌 2 예고**: 이 토대 위에서, 응용된 버전의 강화학습 모각코로 찾아뵙겠습니다. gym 환경을 직접 만들고, 실제 산업 문제에 RL을 적용하는 방향으로 나아갈 예정입니다.

## 라이선스

알고리즘 코드 원본: [minimalRL](https://github.com/seungeunrho/minimalRL) (MIT License, Copyright (c) 2019 seungeunrho)

## 링크

참고 도서: [바닥부터 배우는 강화학습](https://github.com/seungeunrho/RLfrombasics)
