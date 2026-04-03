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
└── actor_critic.py   ← Actor-Critic 알고리즘 (CartPole-v1)
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

브라우저에서 http://localhost:6006 접속

- **SCALARS 탭**: 학습 곡선
- **IMAGES 탭**: CartPole 영상 (학습 전/후 비교)

### REINFORCE vs Actor-Critic 비교

| | REINFORCE | Actor-Critic |
|---|---|---|
| 네트워크 | Policy만 사용 | Policy(Actor) + Value(Critic) |
| 업데이트 | 에피소드 끝난 후 | n-step(10) rollout마다 |
| 분산 | 높음 (Monte Carlo) | 낮음 (TD 기반 baseline) |

두 알고리즘의 TensorBoard 로그를 동시에 비교하려면:

```bash
tensorboard --logdir=~/tb_logs
```

## 개발환경

[개발환경.md]참조

## 라이선스

알고리즘 코드 원본: [minimalRL](https://github.com/seungeunrho/minimalRL) (MIT License, Copyright (c) 2019 Seungeun Rho)

## 링크

참고 도서: [바닥부터 배우는 강화학습](https://github.com/seungeunrho/RLfrombasics)
