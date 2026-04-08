# Alchemist 3-Agent Harness — 예비 설계 검토서 (PDR)

> 최신 모델 탐색 + 사용자 태스크 맞춤 오토 리서치 + 통제 에이전트 프레임워크

---

## 1. 과제 배경

### 1.1 문제

| 문제 | 현상 |
|------|------|
| **모델 선택 병목** | 수백 개 모델 중 태스크에 맞는 최적 모델을 수동으로 찾아야 함 |
| **HP 탐색 수동** | LR, freeze, adapter 등을 연구자가 직접 실험 설계 |
| **벤치마크 단절** | 모델 성능 비교와 태스크 최적화가 분리되어 있음 |
| **재현 불가** | 실험 설정/결과가 체계적으로 기록되지 않음 |

### 1.2 목표

**3개의 전문화된 Agent가 협력하여 모델 탐색 → 순위화 → 사용자 태스크 최적화를 자동화한다.**

### 1.3 Agent 역할

| Agent | 역할 | 핵심 산출물 |
|-------|------|-----------|
| **Benchmark Agent** | 최신 논문/모델 탐색 + 다중 벤치마크 성능 측정 + 모델 순위화 | **Leaderboard** (모델별 랭킹) |
| **Research Agent** | Leaderboard 최고 모델을 가져와 사용자 태스크에 맞게 HP/아키텍처 자동 수정 | **최적화된 모델 + 성능 리포트** |
| **Controller Agent** | 1,2 통제 — 파이프라인 오케스트레이션, Safety, 결과 판정 | **실험 이력 + Ship 판정** |

---

## 2. 전체 흐름

```
사용자 입력: "태스크 설명 + 데이터 경로"
         │
    ┌────▼─────┐
    │Controller │ ── 파이프라인 시작
    └────┬─────┘
         │
   ┌─────▼───────────────────────────────────┐
   │ Phase 1: Benchmark Agent                 │
   │  1. 최신 모델 탐색 (HuggingFace, timm)   │
   │  2. 후보 모델 벤치마크 실행                │
   │  3. 벤치마크별 순위표 생성                 │
   │  4. 사용자 태스크에 최적 모델 추천          │
   └─────┬───────────────────────────────────┘
         │ 추천 모델 + Leaderboard
   ┌─────▼───────────────────────────────────┐
   │ Phase 2: Research Agent                  │
   │  1. 추천 모델을 base로 가져오기           │
   │  2. HP 탐색 (LR, freeze, adapter 등)    │
   │  3. 아키텍처 수정 + Fine-tuning          │
   │  4. 반복 실험 → 최적 설정 도출            │
   │  5. 최종 성능 리포트 반환                 │
   └─────┬───────────────────────────────────┘
         │ 최적화된 모델 + 결과
   ┌─────▼─────┐
   │Controller  │ ── 결과 등록, Ship 판정
   └───────────┘
```

---

## 3. 설계

### 3.1 Benchmark Agent (AD-1)

**내부 구조:**

```
Benchmark Agent
├── Model Scout (LLM + HuggingFace Hub)
│   └── 후보 모델 목록 생성 (이름, backend, 파라미터)
├── Benchmark Runner (결정적)
│   ├── Linear Probe (CIFAR-100 / ImageNet)
│   ├── kNN
│   ├── Detection (COCO)
│   └── 통계 프로토콜
├── Ranker (결정적)
│   ├── 벤치마크별 순위
│   └── 종합 순위 (median rank)
└── Recommender (LLM)
    └── 사용자 태스크 + 제약조건 기반 추천
```

**핵심 스키마:**

```python
@dataclass
class LeaderboardEntry:
    model_name: str
    backend: str
    params_m: float
    scores: dict[str, float]    # benchmark → score
    ranks: dict[str, int]       # benchmark → rank
    overall_rank: int           # median rank

@dataclass
class Leaderboard:
    entries: list[LeaderboardEntry]
    benchmarks: list[str]
    recommendation: str
    recommendation_reason: str
```

### 3.2 Research Agent (AD-2)

**사용자 입력:**

```python
@dataclass
class UserTask:
    name: str                # "pet_action_recognition"
    description: str         # "실내 반려동물 행동 인식"
    data_path: str           # "/data/pet_actions/"
    num_classes: int
    eval_metric: str         # "top1_accuracy"
    constraints: dict        # {"max_params_m": 10}
```

**내부 구조:**

```
Research Agent
├── Experiment Designer (LLM)
│   └── 탐색 공간 설계 (HP 범위, 아키텍처 변형)
├── HP Searcher (결정적)
│   └── LR / batch size / freeze / adapter 조합 탐색
├── Trainer (결정적)
│   └── Fine-tuning 실행 + 평가
└── Report Generator (LLM)
    └── 최종 리포트 + 최적 설정 문서화
```

**핵심 스키마:**

```python
@dataclass
class ResearchResult:
    base_model: str
    task: UserTask
    best_config: dict
    best_score: float
    baseline_score: float
    improvement: float
    trials: list[TrialResult]
    checkpoint_path: str
    report: str
```

### 3.3 Controller Agent (AD-3)

```
Controller Agent
├── Pipeline Orchestrator
│   └── Benchmark → Research 순차 실행
├── Safety Guard
│   ├── Budget 추적, 성능 하락 감지
│   └── Human Escalation Gate
├── Registry
│   └── Leaderboard + Research 이력 관리
└── DECIDE
    ├── 모델 추천 승인/거부
    └── Ship 판정
```

### 3.4 Agent 간 통신

**AgentMessage 프로토콜 (JSON):**

```python
@dataclass
class AgentMessage:
    msg_id: str
    from_agent: Literal["controller", "benchmark", "research"]
    to_agent: Literal["controller", "benchmark", "research"]
    msg_type: Literal["directive", "response", "status", "escalation"]
    payload: dict
    episode: int
    budget_remaining: float
    trace_id: str
    timestamp: str
```

모든 메시지는 **MessageBus**를 통해 전달되며, JSONL 감사 로그에 자동 기록.

### 3.5 LLM 격리

| Agent | LLM 영역 | 결정적 영역 | Fallback |
|-------|---------|-----------|---------|
| Benchmark | 모델 추천 근거 | 벤치마크 실행, 순위 산출 | 규칙 기반 추천 |
| Research | 실험 설계, 실패 분석 | HP 탐색, 학습, 평가 | 기본 config grid |
| Controller | 결과 해석 | 예산 추적, Ship 판정 | 규칙 기반 |

---

## 4. 안전 장치

| Level | 주체 | 동작 |
|-------|------|------|
| L0 | Agent 내부 | OOM 감지, Compile Gate, 통계 이상치 |
| L1 | Controller | 예산 추적, 성능 하락 감지, 에스컬레이션 |
| L2 | 인간 | Ship 최종 판정, 예산 80% 소진 시 |

---

## 5. 검증된 결과

### CIFAR-100 분류

| 단계 | 결과 |
|------|------|
| Benchmark Agent | 7개 모델 실측, DINOv2 ViT-B/14 #1 (87.50%) |
| Research Agent | MLP head + SGD lr=0.1로 89.57% 달성 (+2.07%) |

### COCO Detection

| 단계 | 결과 |
|------|------|
| Benchmark Agent | 5개 모델 실측, Faster R-CNN #1 (AP50=70.7%) |

---

## 6. 리스크

| 리스크 | 심각도 | 완화 |
|--------|--------|------|
| LLM 추천 품질 | Medium | Tool 결과 기반 판단, 규칙 fallback |
| GPU OOM (RTX 3050) | High | 대형 모델 자동 감지 → frozen only |
| HP 탐색 수렴 실패 | Medium | 조기 종료 + 탐색 횟수 상한 |
| 에이전트 간 통신 오류 | Low | JSON 직렬화 + 감사 로그 |

---

## 관련 문서

- [TDD.md](TDD.md) — 테스트 주도 설계
- Obsidian 상세: `Obsidian/Alchemist/Alchemist-3Agent-PDR.md`
