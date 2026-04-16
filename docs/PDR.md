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
| **Benchmark Agent** | 최신 논문/모델 탐색 + 다중 벤치마크 성능 측정 + 모델 순위화 + **SoTA 현황 조사** | **Leaderboard + SoTA Standing** |
| **Research Agent** | Leaderboard 최고 모델을 가져와 **SoTA를 넘기 위한** 기법 탐색 + HP/아키텍처 자율 수정 | **최적화된 모델 + 성능 리포트** |
| **Controller Agent** | 1,2 통제 — 파이프라인 오케스트레이션, Safety, 결과 판정 | **실험 이력 + Ship 판정** |

**에이전트간 지식 역할 분리 (v3에서 명확화):**
- **Benchmark = "무엇이 최고 점수인가?"** (what number to beat)
- **Research = "어떻게 그 점수를 뛰어넘는가?"** (how to beat it)

---

## 2. 전체 흐름

```
사용자 입력: "태스크 설명 + 데이터 경로"
         │
    ┌────▼─────┐
    │Controller │ ── 파이프라인 시작
    └────┬─────┘
         │
   ┌─────▼────────────────────────────────────────┐
   │ Phase 1: Benchmark Agent                      │
   │  1. 최신 모델 탐색 (HF Hub API, live query)   │
   │  2. SoTA 현황 조사 (PwC archive snapshot)      │
   │  3. 후보 모델 벤치마크 실행                    │
   │  4. 벤치마크별 순위표 생성 (SoTA 대비 표시)    │
   │  5. 사용자 태스크에 최적 모델 추천              │
   └─────┬────────────────────────────────────────┘
         │ 추천 모델 + Leaderboard + SoTA Standing
   ┌─────▼────────────────────────────────────────┐
   │ Phase 2: Research Agent                       │
   │  1. 기법 탐색 (arXiv API, 2023–2025 논문)     │
   │  2. 추천 모델을 base로 가져오기               │
   │  3. SoTA Standing(Benchmark) + 기법(arXiv)로  │
   │     실험 설계                                  │
   │  4. HP 탐색 + 아키텍처 탐색 (NAS)             │
   │  5. 자체 분석 + SoTA gap 분석                 │
   │  6. 부족한 기법 자동 도입 (SAM 등)            │
   │  7. 반복 개선 → 최적 설정 도출                │
   │  8. 최종 성능 리포트 + SoTA 대비 포지셔닝     │
   └─────┬────────────────────────────────────────┘
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
├── Model Scout (LLM + HFHubRetriever)            ← v3: HF Hub API
│   ├── search_imagenet1k_models() 라이브 발견
│   ├── pretrain source 자동 분류 (in1k/in21k/jft/...)
│   └── KNOWN_MODELS seed + LLM 추가 제안 융합
├── SoTA Standing Lookup (HFHubRetriever) ← v3 NEW
│   ├── search_sota_standing(task)
│   ├── pwc-archive parquet 조회 (137MB cache)
│   └── top model + 점수 + 'extra-data' flag 추출
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

**External knowledge sources (v3에서 추가):**
- **HuggingFace Hub API** (`huggingface_hub`) — 모델 라이브 검색·메타데이터
- **PwC Archive** (`pwc-archive/evaluation-tables` HF parquet) — 최종 SoTA 스냅샷 (2025-09)
- 모두 **무료 공개 API**, 인증 불필요, 7일 file cache

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

# v3 NEW — Benchmark Agent의 SoTA 조사 결과
SoTAStanding = dict  # {
#     "task": str,
#     "entries": list[dict],          # PwC top entries
#     "top_score_pct": float | None,  # ground-truth 최고 점수
#     "top_model": str | None,
#     "top_paper": str | None,
#     "summary": str,                  # LLM-friendly text block
# }
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
├── Technique Retrieval (ArxivRetriever) ← v3: arXiv API
│   ├── search() 벤치마크/기법 키워드 → 최근 3년 논문
│   ├── summarize_for_llm() LLM 프롬프트 주입용 텍스트
│   └── 7일 file cache (~/.cache/alchemist/retrievers/)
│
├── SoTA Synthesis (LLM + Benchmark Standing + arXiv) ← v3 재구성
│   ├── Benchmark Agent의 sota_standing(PwC) 입력 사용
│   ├── arXiv 기법 논문 evidence 결합
│   └── search_sota(task, sota_standing) → 기법 요약
│
├── Gap Analysis & Technique Suggestion (LLM)
│   ├── Gap Analysis: 현재 결과 vs SoTA 차이 분석
│   └── Technique Suggestion: 부족한 기법 자동 제안
│       (optimizer: SAM, pretrained: IN-22K/JFT, 학습 전략 등)
│
├── Experiment Designer (LLM + SoTA Knowledge)
│   ├── Round 1: 탐색 공간 설계 (HP 범위, 아키텍처 변형)
│   └── Round 2+: SoTA gap 기반 targeted refinement
│
├── NAS (Architecture Search)
│   ├── Phase 1: 다중 backbone 탐색 (CNN/ViT/Swin/Mamba)
│   └── Phase 2: Top-K backbone 집중 HP 최적화
│
├── Self-Analysis Loop (LLM)
│   ├── 실험 결과 패턴 분석 (어떤 HP가 효과적인지)
│   ├── SoTA 대비 gap 분석 (부족한 기법 식별)
│   └── 계속/중단 자율 판단
│
├── HP Searcher (결정적)
│   └── LR / batch size / freeze / adapter / optimizer 조합 탐색
│
├── Trainer (결정적, Local or AWS GPU)
│   └── Fine-tuning / Linear Probe 실행 + 평가
│
├── Research Log
│   └── 전 과정 기록 (SoTA 검색, 분석, 설계, 실행, 판단)
│
└── Report Generator (LLM)
    └── 최종 리포트 + SoTA 대비 포지셔닝 문서화
```

**External knowledge sources (v3에서 추가):**
- **arXiv API** (`arxiv` 라이브러리) — 기법 논문 검색, 연도 필터, free public
- (Benchmark Agent로부터) **PwC archive 기반 sota_standing** — "어느 점수를 넘어야 하는가"
- 모두 무료, 인증 불필요, 7일 file cache

**자율 개선 흐름:**

```
                    ┌─────────────────────────────────────────┐
                    │  Benchmark Agent: SoTA Standing (PwC)   │
                    │  "Top=96.81%, EffNet-L2(SAM), 2020"    │
                    │  flag=[uses additional data]            │
                    └────────────────┬────────────────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────────────┐
                    │  Research: Technique Synthesis (arXiv) │
                    │  "SAM, longer training, IN-22K …"      │
                    │  - extra-data 항목은 fair-comparison 제외│
                    └────────────────┬───────────────────────┘
                                     │
                                     ▼
    ┌─────────────────────────────────────────────────────────┐
    │  Round 1: 실험 설계 (SoTA 지식 반영)                     │
    │  → 실행 → 자체 분석 + SoTA gap 분석                      │
    │  → "현재 93%, SoTA 96.81%, GAP: SAM optimizer 미사용"   │
    │  → 계속 판단: Yes (gap 존재)                             │
    ├─────────────────────────────────────────────────────────┤
    │  Round 2: SoTA gap 기반 재설계                           │
    │  → SAM optimizer 추가, 더 긴 학습, LR 조정               │
    │  → 실행 → 분석 → gap 재평가                              │
    │  → "94.5%, gap 줄어듦. Pretrained 크기가 병목"            │
    ├─────────────────────────────────────────────────────────┤
    │  Round N: 더 이상 개선 불가 또는 예산 소진 → 종료         │
    └─────────────────────────────────────────────────────────┘
```

**핵심 메서드:**

| 메서드 | 역할 | 입력 | 출력 |
|--------|------|------|------|
| `search_sota(task, sota_standing)` | arXiv 기법 evidence + Benchmark의 SoTA standing 결합 → 기법 종합 | task, (optional) sota_standing | 기법 요약 텍스트 |
| `analyze_sota_gap()` | 현재 결과 vs SoTA gap 분석 | score, sota, trials | gap 분석 + 기법 제안 |
| `suggest_techniques()` | 미사용 기법 자동 제안 | sota, history | config 리스트 (JSON) |
| `design_experiment()` | SoTA 반영 실험 설계 | sota, analysis | TrialConfig 리스트 |
| `analyze_results()` | 자체 결과 패턴 분석 | trials | 분석 텍스트 |
| `should_continue_research()` | 계속/중단 판단 | analysis, gap | bool |

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

**Benchmark → Research payload 확장 (v3):** Benchmark 응답 payload에
`sota_standing` 키 추가. Research Agent가 `search_sota(task, sota_standing)`로
이를 직접 사용 (재조회 X).

### 3.5 External Knowledge Retrievers (v3 NEW)

`alchemist/core/retrievers/` 패키지 — 두 에이전트가 공용으로 사용.

| Retriever | API | 사용 에이전트 | 주 용도 |
|---|---|---|---|
| `ArxivRetriever` | arxiv.org public API | Research | 기법 논문 검색 |
| `HFHubRetriever` | huggingface_hub | Benchmark (주) / Research | 모델 라이브 검색, **PwC archive 조회** |

**공통 정책:**
- 인증/유료 키 불필요 (Alchemist의 zero-API-cost 원칙 유지)
- 7일 file cache (`~/.cache/alchemist/retrievers/`)
- Graceful fallback — 네트워크/소스 실패 시 빈 결과 반환, 에이전트는 LLM 내재 지식으로 진행
- ImageNet-1K 사전학습 모델만 후보로 인정 (`classify_pretrain_source`)
- "uses additional data" 플래그된 SoTA는 fair-comparison 모드에서 자동 제외

**`PwC archive` 데이터 출처:**
- PaperWithCode 본 서비스는 2025-07 종료
- 최종 스냅샷이 `pwc-archive/evaluation-tables` HF parquet 으로 보존됨 (~138MB)
- `HFHubRetriever.search_pwc_leaderboard()`로 조회

### 3.6 LLM 격리

| Agent | LLM 영역 | 결정적 영역 | Retrieval | Fallback |
|-------|---------|-----------|-----------|---------|
| Benchmark | 모델 추천 근거 | 벤치마크 실행, 순위 산출 | HF Hub + PwC | 규칙 기반 추천 |
| Research | 실험 설계, 실패 분석 | HP 탐색, 학습, 평가 | arXiv | 기본 config grid |
| Controller | 결과 해석 | 예산 추적, Ship 판정 | — | 규칙 기반 |

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
