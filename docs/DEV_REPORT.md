# Alchemist 개발 리포트

> 최종 업데이트: 2026-04-01

---

## 세션 1 — 2026-03-31

### 개요

| 항목 | 값 |
|------|-----|
| 작업 시간 | ~60분 |
| 변경/생성 파일 | 8개 |
| 추가된 코드 | ~2,472줄 (89.5KB) |
| 테스트 결과 | 54/54 통과 |
| LLM 백엔드 | Claude Code CLI (Opus 4.6 1M) |

### 토큰 사용량 (추정)

| 구분 | Input 토큰 (추정) | Output 토큰 (추정) |
|------|-------------------|-------------------|
| 프로젝트 분석 | ~30,000 | ~8,000 |
| LLM CLI 클라이언트 구현 | ~15,000 | ~5,000 |
| AWS Executor 구현 | ~20,000 | ~12,000 |
| 에이전트 상호작용 구현 | ~25,000 | ~10,000 |
| Research Agent 리팩토링 | ~35,000 | ~18,000 |
| 테스트 및 검증 | ~10,000 | ~3,000 |
| **합계** | **~135,000** | **~56,000** |

> 참고: Claude CLI 호출로 소비된 에이전트 테스트 토큰은 별도 (sonnet 모델 약 6회 호출, 각 ~2,000 토큰)

---

### 개발 내역

#### 1. LLM CLI 클라이언트 구현

**파일:** `alchemist/core/llm.py` (기존 99줄 → 193줄)

| 클래스 | 설명 |
|--------|------|
| `ClaudeCLIClient` | `claude -p --model {model}` subprocess 호출 |
| `CodexCLIClient` | `codex exec --skip-git-repo-check` subprocess 호출 + 출력 파싱 |

- API 키 없이 로컬에 인증된 CLI를 그대로 사용
- `generate()` → 텍스트 응답, `generate_json()` → JSON 파싱 (상위 클래스에서 상속)
- 타임아웃, 모델 선택, 에러 핸들링 포함

**검증:**
```
ClaudeCLIClient.generate("Say hello") → "Hello!"  ✓
ClaudeCLIClient.generate_json(...)    → {"status": "ok"}  ✓
CodexCLIClient.generate("Say hello")  → "Hello"  ✓
CodexCLIClient.generate_json(...)     → {"status": "ok"}  ✓
```

---

#### 2. Training Executor 분리 (로컬/AWS)

**새 파일:** `alchemist/core/executor.py` (270줄)

| 클래스 | 설명 |
|--------|------|
| `TrainingExecutor` | 추상 인터페이스 (`run_trial`, `evaluate_baseline`) |
| `LocalExecutor` | 기존 시뮬레이션 로직 추출 (테스트용) |
| `AWSExecutor` | SSH로 원격 GPU 인스턴스에 학습 작업 제출 + 폴링 |

**AWSExecutor 동작 흐름:**
```
로컬: job JSON 생성 → SCP 전송 → SSH로 nohup 실행
원격: train_worker.py가 학습 → result JSON 생성
로컬: 10초 간격 SSH 폴링 → result 수신 → TrialResult 반환
```

**새 파일:** `train_worker.py` (305줄)
- AWS GPU에서 실행되는 독립 학습 스크립트
- PyTorch + timm 기반 실제 학습 (feature extraction / full fine-tuning)
- job JSON 입력 → result JSON 출력

---

#### 3. 에이전트 간 LLM 기반 상호작용

**변경 파일:** `alchemist/agents/benchmark.py`, `controller.py`, `research.py`, `harness.py`

**구현된 상호작용 흐름:**
```
BenchmarkAgent
  └─ recommend(): LLM으로 모델 강점/약점/fine-tuning 전략 제안 생성
        │
        ▼
Controller._context에 축적
  └─ validate_recommendation(): LLM으로 추천 모델 적합성 검증
        │
        ▼
ResearchAgent
  └─ design_experiment(upstream_context=...): Controller의 축적된 컨텍스트 반영
```

---

#### 4. Research Agent 자율 분석 루프 리팩토링

**변경 파일:** `alchemist/agents/research.py` (기존 308줄 → 432줄)

**변경 전 (역할 혼재):**
```
Research: 실험만 실행 → Controller: 결과 분석 + 피드백 → Harness: 재실험 루프
```

**변경 후 (역할 명확 분리):**
```
Research: 설계 → 실행 → 자체 분석 → 자체 개선 (내부 루프, 최대 max_rounds)
Controller: 예산/안전/ship 기준 판정만
```

**Research Agent 내부 루프 구조:**
```
Round 1: design_experiment() → run_trials() → analyze_results() → should_continue_research()
  │        (broad exploration)     (6 trials)    (LLM 자체 분석)     (LLM 계속 여부 판단)
  ↓ analysis: "linear_head + lr=0.0003 최고, partial unfreeze 시도 필요"
Round 2: design_experiment(prior_analysis) → run_trials() → analyze_results() → ...
  │        (분석 기반 targeted refinement)
  ↓
...최대 max_rounds까지 또는 자체 판단으로 중단
```

**새 클래스:** `ResearchLog`
- 전체 연구 과정을 구조화된 JSON으로 기록
- phase별: init → baseline → design → execution → analysis → decision → final
- `logs/research_log_{task}_{trace_id}.json`에 자동 저장

**연구 로그 예시 (실제 출력):**
```json
{
  "round": 1,
  "phase": "analysis",
  "action": "self_analysis",
  "detail": {
    "analysis": "linear_head adapter is clearly better (+4.3pp avg over none),
                 lr=0.0003 is the sweet spot. Unexplored: partial unfreeze,
                 lr range [0.0002-0.0005], extended epochs.",
    "current_best": 82.94,
    "improvement": 11.69
  }
}
```

---

#### 5. CLI 옵션 확장

**변경 파일:** `main.py` (기존 138줄 → 218줄)

**추가된 CLI 옵션:**

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--llm` | LLM 백엔드 선택 (mock/claude/codex) | mock |
| `--llm-model` | CLI 모델 지정 | sonnet (claude), 기본 (codex) |
| `--llm-timeout` | CLI 타임아웃 (초) | 120 |
| `--executor` | 학습 실행 위치 (local/aws) | local |
| `--aws-host` | AWS SSH 호스트 | - |
| `--aws-key` | SSH 키 경로 | - |
| `--aws-work-dir` | 원격 작업 디렉토리 | /home/ubuntu/alchemist |
| `--aws-python` | 원격 Python 명령 | python |
| `--max-rounds` | Research 반복 라운드 수 | 3 |

---

### 통합 테스트 결과

**Mock LLM 테스트:**
```
54/54 passed (0.05s)
```

**Claude CLI (sonnet) 실제 파이프라인 테스트:**
```
Benchmark → 7 모델 탐색, dinov2_vitb14 추천
Controller → LLM 기반 추천 검증: "dinov2_vitb14 appropriate for task"
Research → 6 trials, 자체 분석 수행
  Round 1: baseline 71.2% → best 82.9% (+11.7%)
  Analysis: "linear_head adapter +4.3pp, lr=0.0003 optimal, partial unfreeze unexplored"
  Decision: 자체 판단으로 1라운드에서 중단
Controller → Ship 판정: Approved
연구 로그 저장: research_log_classification_22173192-ab9.json (7 entries)
```

---

### 파일 변경 요약

| 파일 | 상태 | 줄 수 | 설명 |
|------|------|-------|------|
| `alchemist/core/llm.py` | 수정 | 193 | ClaudeCLIClient, CodexCLIClient 추가 |
| `alchemist/core/executor.py` | **신규** | 270 | TrainingExecutor, LocalExecutor, AWSExecutor |
| `alchemist/agents/research.py` | 수정 | 432 | 자율 분석 루프, ResearchLog 추가 |
| `alchemist/agents/controller.py` | 수정 | 233 | 결과 분석 제거, ship 기준 판정만 |
| `alchemist/agents/benchmark.py` | 수정 | 249 | LLM 추천 상세화 |
| `alchemist/harness.py` | 수정 | 322 | 피드백 루프 제거, 연구 로그 저장 |
| `main.py` | 수정 | 218 | CLI 옵션 확장 |
| `train_worker.py` | **신규** | 305 | AWS GPU 학습 워커 |
| **합계** | | **2,222** | |

---

### 아키텍처 현황

```
┌─────────────────────────────────────────────────────────┐
│                    로컬 (Agent Layer)                     │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Benchmark    │  │ Research     │  │ Controller   │   │
│  │ Agent        │  │ Agent        │  │ Agent        │   │
│  │              │  │              │  │              │   │
│  │ 모델 탐색    │  │ 실험 설계    │  │ 안전 관리    │   │
│  │ 벤치마크     │→ │ 실행+분석    │→ │ Ship 판정    │   │
│  │ 추천         │  │ 자율 반복    │  │              │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘   │
│         │                 │                              │
│         │    LLM (Claude CLI / Codex CLI)                │
│         │    ┌─────────────────────────┐                 │
│         └────│ claude -p / codex exec  │─────────────────│
│              └─────────────────────────┘                 │
│                                                          │
│  Executor: LocalExecutor (시뮬레이션)                     │
│            AWSExecutor (SSH → 원격 GPU)                   │
│                        │                                 │
└────────────────────────│─────────────────────────────────┘
                         │ SSH + SCP
                         ▼
              ┌──────────────────────┐
              │   AWS GPU Instance   │
              │                      │
              │   train_worker.py    │
              │   PyTorch + timm     │
              │                      │
              └──────────────────────┘
```

---

---

## 세션 2 — 2026-04-01

### 개요

| 항목 | 값 |
|------|-----|
| 작업 시간 | ~30분 |
| 변경 파일 | 1개 (`alchemist/agents/research.py`) |
| 테스트 결과 | 54/54 통과 |
| LLM 백엔드 | Claude Code CLI (Opus 4.6 1M) |

### 토큰 사용량 (추정)

| 구분 | Input 토큰 (추정) | Output 토큰 (추정) |
|------|-------------------|-------------------|
| 3-Agent 상호작용 검증 | ~25,000 | ~5,000 |
| 다중 라운드 버그 진단 | ~15,000 | ~3,000 |
| should_continue 프롬프트 개선 | ~10,000 | ~4,000 |
| trial 예산 로직 수정 | ~8,000 | ~2,000 |
| 3라운드 통합 테스트 | ~12,000 | ~5,000 |
| **합계** | **~70,000** | **~19,000** |

> Claude CLI (sonnet) 에이전트 테스트 호출: ~40회 (3라운드 × 각 ~8 LLM 호출 + 벤치마크/컨트롤러)

### 개발 내역

#### 1. 다중 라운드 Research 루프 버그 수정

**문제:** Research Agent가 항상 Round 1에서 중단됨

**근본 원인:** `max_trials`가 **전체 누적** 기준이라 Round 1에서 모든 trial을 소진 → `remaining_trials=0` → `should_continue_research()`가 LLM 호출 전에 `False` 반환

**수정:**
- `max_trials`를 **라운드당 trial 수**로 변경
- 각 라운드가 독립적으로 `max_trials`개 trial을 실행 가능
- 총 trial 수 제한은 예산(GPU-hours)으로만 관리

```python
# 변경 전
remaining_trials = self.max_trials - len(all_trials)  # 누적되면 0이 됨

# 변경 후 — 각 라운드 독립
configs = self.design_experiment(..., remaining_trials=self.max_trials, ...)
```

#### 2. should_continue_research 프롬프트 개선

**문제:** LLM이 분석에서 "unexplored directions 있음, Room for improvement: Yes"라고 하면서도 `continue: false` 반환

**수정:** 프롬프트에 명확한 판단 기준 제시

```
You should continue if ANY of these are true:
- Your analysis identified unexplored promising directions
- Key HP dimensions remain untested
- The improvement trend suggests further gains
- You have remaining budget and trials

You should stop only if:
- All major directions have been explored
- Scores have plateaued across diverse configurations
- Budget or trial count is exhausted
```

### 통합 테스트 결과 — 3-Agent 상호작용 검증

**실행:** `python main.py run --llm claude --max-trials 4 --max-rounds 3`

#### 에이전트 간 상호작용 흐름 (검증 완료)

```
1. BenchmarkAgent
   └─ 7개 모델 탐색, dinov2_vitb14 추천
   └─ LLM: "87.3% linear probe, 5pt margin, strong features"
        │
        ▼ (upstream_context: 1,287자)
2. Controller
   └─ LLM 검증: "appropriate for task, no constraints disqualify it"
        │
        ▼ (upstream_context → Research directive)
3. ResearchAgent (자율 3라운드 루프)
   │
   ├─ Round 1: 4 trials → best 80.0% (+5.2%)
   │   분석: "linear_head +4.59pp, freeze=False untested"
   │   판단: continue=True ✓
   │
   ├─ Round 2: 4 trials → best 82.9% (+8.1%)
   │   분석: "freeze=False with LoRA best, lr sweep needed"
   │   판단: continue=True ✓
   │
   └─ Round 3: 4 trials → best 85.8% (+11.1%)
       분석: "LoRA + freeze=False dominant, convergence nearing"
       판단: continue=False (max_rounds 도달)
        │
        ▼
4. Controller
   └─ Ship 판정: "Approved: 74.7% → 85.8% (+11.1%)" ✓
```

#### 연구 로그 (실제 출력)

| Round | Trials | Best | 개선 | 핵심 발견 | 다음 방향 |
|-------|--------|------|------|----------|----------|
| 1 | 4 | 80.0% | +5.2% | linear_head +4.59pp > none | freeze=False, MLP head 시도 |
| 2 | 4 | 82.9% | +8.1% | freeze=False +5.9pp 도약 | LoRA + lr sweep, epochs=30 |
| 3 | 4 | 85.8% | +11.1% | LoRA + freeze=False 최적 | lr fine-tune, LoRA rank 조정 |

**최종 결과:** baseline 74.7% → **85.8%** (+11.1%), 12 trials / 3 rounds

### 파일 변경 요약

| 파일 | 변경 내용 |
|------|----------|
| `alchemist/agents/research.py` | trial 예산 라운드당으로 변경, should_continue 프롬프트 개선 |

---

---

## 세션 2 (추가) — Dashboard 연동

### 개요

| 항목 | 값 |
|------|-----|
| 작업 | Alchemist → Alchemist_Dashboard 데이터 이식 |
| 생성 파일 | `export_dashboard.py` (208줄) |
| 출력 | `Alchemist_Dashboard/data/alchemist_v2.json` (13 models) |

### 토큰 사용량 (추정)

| 구분 | Input 토큰 (추정) | Output 토큰 (추정) |
|------|-------------------|-------------------|
| 양 프로젝트 구조 분석 | ~40,000 | ~15,000 |
| export_dashboard.py 구현 | ~10,000 | ~8,000 |
| 파이프라인 실행 + 변환 | ~15,000 | ~5,000 |
| 대시보드 검증 | ~5,000 | ~1,000 |
| **합계** | **~70,000** | **~29,000** |

### 개발 내역

#### Dashboard 데이터 변환 모듈 (`export_dashboard.py`)

**핵심:** Alchemist 3-Agent 파이프라인 결과 → Dashboard JSON 스키마 변환

**데이터 매핑:**

| Dashboard 필드 | Alchemist 소스 |
|---------------|---------------|
| `models[0]` (v001) | Leaderboard baseline (linear_probe + knn 점수) |
| `models[1+]` (v002~) | 각 TrialResult (score, config, elapsed) |
| `scores[].value` | `trial.score / 100.0` (0-1 범위로 변환) |
| `training.*` | `TrialConfig` (lr, batch_size, epochs, scheduler) |
| `training.extra` | freeze_backbone, adapter, weight_decay 등 |
| `notes` | ResearchLog의 라운드별 자체 분석 결과 포함 |
| `proposal_reason` | "Round N: {adapter}, {freeze}, lr={lr}" |
| `efficiency.params_m` | KNOWN_MODELS 레지스트리에서 조회 |

**실행 결과:**

```
Framework: Alchemist_v2 (3 agents)
Models: 13 (1 baseline + 12 trials across 3 rounds)
Best score: 87.30% (dinov2_vitb14 baseline linear_probe)
Best trial: 85.80% (LoRA + unfrozen backbone)
Dashboard 로드: 성공 ✓
```

**사용법:**

```bash
# Mock LLM (빠른 테스트)
python export_dashboard.py

# Claude CLI로 실제 에이전트 실행
python export_dashboard.py --llm claude --llm-model sonnet

# 커스텀 출력 경로
python export_dashboard.py --output /path/to/dashboard/data/alchemist_v2.json
```

---

---

## 세션 3 — 2026-04-01 (AWS GPU 학습)

### 개요

| 항목 | 값 |
|------|-----|
| 작업 | AWS EC2(A10G) 인프라 구축 + ResNet-50 CIFAR-100 최적화 |
| EC2 Instance | g5.xlarge (A10G 24GB, us-east-1) |
| S3 Bucket | alchemist-vp-851336487511 |
| 목표 | ResNet-50 CIFAR-100 최고 성능 도출 |

### 토큰 사용량 (추정)

| 구분 | Input 토큰 (추정) | Output 토큰 (추정) |
|------|-------------------|-------------------|
| AWS 인프라 구축 (S3, EC2, IAM, SG) | ~20,000 | ~8,000 |
| train_worker.py 고급 기법 구현 | ~25,000 | ~15,000 |
| ResNet-50 아키텍처 수정 기능 구현 | ~15,000 | ~10,000 |
| EC2 환경 설정 + 검증 | ~15,000 | ~5,000 |
| 학습 실험 실행 + 모니터링 | ~10,000 | ~3,000 |
| **합계** | **~85,000** | **~41,000** |

### AWS 인프라 구축

| 리소스 | 값 | 비용 |
|--------|-----|------|
| EC2 g5.xlarge | i-07e444c5bdc43b0d3 | ~$1.006/hr |
| S3 Bucket | alchemist-vp-851336487511 | ~$0.025/GB/월 |
| SSH Key | ~/.ssh/alchemist-gpu-key-use1.pem | - |
| Security Group | sg-038b3c7d2c309e0ca (SSH) | - |
| IAM Role | alchemist-ec2-role (S3 접근) | - |
| Region | us-east-1 | - |

### train_worker.py 고급 기법 추가

| 기법 | 설명 |
|------|------|
| Mixup | α=0.2~0.3, 이미지 혼합 정규화 |
| CutMix | α=1.0, 패치 혼합 정규화 |
| RandAugment | num_ops=2, magnitude=9 |
| RandomErasing | p=0.25 |
| Label Smoothing | 0.1 |
| Warmup + Cosine LR | warmup_epochs=5 + cosine decay |
| Differential LR | backbone 0.05~0.1x, head 1x |
| Mixed Precision | FP16 (GradScaler) |
| Gradient Clipping | max_norm=1.0 |
| Best Checkpoint | epoch별 best val score 저장 |

### ResNet-50 아키텍처 수정 기능

| 수정 | 설명 |
|------|------|
| SE Block | Squeeze-and-Excitation, 모든 layer에 주입 |
| Self-Attention | layer4 마지막 블록에 MultiheadAttention 추가 |
| MLP Head | 2048→1024→512→100, BatchNorm+GELU+Dropout |
| Drop Path | Stochastic depth (drop_path_rate) |

### 실험 결과 (진행 중)

**이전 인스턴스 (i-027fda1f27afcbcd2) 결과:**

| Trial | Config | Score | 시간 |
|-------|--------|-------|------|
| Baseline | frozen, linear_head, 3ep | 71.55% | 112s |
| T1 | frozen, mlp_head, 30ep | 73.54% | 158s |
| T2 | fine-tune, none, lr=3e-4, 30ep, RandAug+Erasing | **83.79%** | 2,772s |

**현재 인스턴스 (i-07e444c5bdc43b0d3) — 실행 중:**

| Trial | Config | Score | 상태 |
|-------|--------|-------|------|
| R1 | fine-tune, mlp_head, Mixup, 40ep | - | 실행 중 |
| R2 | fine-tune, mlp_head, CutMix+SE, 50ep | - | 대기 |
| R3 | fine-tune, mlp_head, Mixup+CutMix+Attn, 50ep | - | 대기 |
| R4 | fine-tune, mlp_head, SE+Attn+all, 60ep | - | 대기 |

> 학습 완료 후 EC2 일시정지 + 대시보드 업데이트 예정

---

### EC2 비용 추적

| 인스턴스 | 시작 | 종료 | 비용(추정) |
|---------|------|------|-----------|
| i-027fda1f27afcbcd2 | 08:12 UTC | 09:15 UTC (terminated) | ~$1.06 |
| i-07e444c5bdc43b0d3 | 09:17 UTC | 진행 중 | - |

---

---

## 세션 3 (완료) — AWS GPU 학습 최종 결과

### 실험 결과 (ResNet-50 + CIFAR-100)

| Trial | Config | LR 전략 | Best Score | 시간 | 인사이트 |
|-------|--------|---------|-----------|------|---------|
| Baseline | frozen, linear head | cosine | 71.55% | 112s | - |
| **R1** | MLP+Mixup+RandAug+Erasing | warmup+cosine | **86.03%** | 81min | 강력한 기본 레시피 |
| **R2** | MLP+Mixup+CutMix+RandAug | **OneCycleLR** | **86.77%** | 81min | **최고 성능, OneCycleLR > Cosine** |
| R3 | MLP+**SE**+Mixup+CutMix | CosineWarmRestarts | 82.80% | 95min | SE 초기화가 pretrained weights 파괴 |
| R4 | MLP+Mixup+CutMix+**EMA** | OneCycleLR | 1.46% | 81min | EMA 초기 apply 버그 (수정됨) |
| R5 | MLP+Mixup+CutMix+**EMA**(수정) | OneCycleLR | 80.45% | 97min | EMA shadow 수렴 부족 |

### 최종 Best: **86.77%** (R2)

**Best 설정:**
- ResNet-50 (ImageNet pretrained) → full fine-tuning
- MLP Head: 2048→1024→512→100 (BN+GELU+Dropout 0.3)
- **OneCycleLR**: max_lr=0.001, pct_start=0.3, div_factor=25
- Differential LR: backbone 0.1x
- Mixup(α=0.2) + CutMix(α=1.0)
- RandAugment + RandomErasing
- Label Smoothing(0.1), Weight Decay(0.05)
- 50 epochs, batch 64, AdamW, Mixed Precision

### 핵심 인사이트

1. **OneCycleLR > Warmup+Cosine**: +0.74%p 차이 (86.77% vs 86.03%)
2. **Mixup+CutMix 조합이 효과적**: 단독 Mixup보다 조합이 더 좋음
3. **SE Block 주의**: random 초기화된 SE가 pretrained weights를 파괴, warmup이 필요
4. **EMA는 epoch 단위 update로는 부족**: batch 단위 update 필요, 또는 충분한 warmup 후 apply
5. **MLP Head가 linear보다 훨씬 우수**: frozen+MLP(73.54%) vs frozen+linear(71.55%)

### EC2 비용 추적

| 인스턴스 | 시작 | 종료 | 비용(추정) |
|---------|------|------|-----------|
| i-027fda1f27afcbcd2 | 08:12 | 09:15 (terminated) | ~$1.06 |
| i-07e444c5bdc43b0d3 | 09:17 | 09:40 (terminated) | ~$0.38 |
| i-099063a9d2735575f | 09:47 | 10:25 (terminated) | ~$0.63 |
| i-0af09a7dd632cf070 | 10:55 | 11:45 (terminated) | ~$0.84 |
| i-018d47c53fa11619c | 11:38 | 11:48 (terminated) | ~$0.17 |
| **i-08747276c99e6f69f** | 11:49 | **20:10 (stopped)** | **~$8.39** |
| **합계** | | | **~$11.47** |

### 토큰 사용량 (추정, 세션 3 전체)

| 구분 | Input 토큰 (추정) | Output 토큰 (추정) |
|------|-------------------|-------------------|
| AWS 인프라 구축 | ~20,000 | ~8,000 |
| train_worker 구현 (고급 기법 + 아키텍처 수정 + EMA) | ~40,000 | ~25,000 |
| 학습 실행 + 모니터링 + 디버깅 | ~60,000 | ~15,000 |
| 대시보드 업데이트 | ~10,000 | ~5,000 |
| **합계** | **~130,000** | **~53,000** |

### 대시보드 업데이트

`Alchemist_Dashboard/data/alchemist_v2.json` 업데이트:
- 5 models (1 baseline + 4 trials)
- Best: v003 = 86.77%

---

---

## 세션 4 — 2026-04-07 (NAS Architecture Search)

### 개요

| 항목 | 값 |
|------|-----|
| 작업 | CIFAR-100 Neural Architecture Search (8시간) |
| EC2 | i-0e35db2d675713696 (incheon-alchemist-nas, g5.xlarge) |
| 실행 시간 | 7시간 57분 (02:02~09:59 UTC) |
| 총 trials | 10 (Phase 1: 7, Phase 2: 3) |
| **Best Score** | **90.87%** (ConvNeXt-Tiny + MLP head, 40ep) |

### 토큰 사용량 (추정)

| 구분 | Input 토큰 (추정) | Output 토큰 (추정) |
|------|-------------------|-------------------|
| NAS worker + batch script 구현 | ~20,000 | ~15,000 |
| EC2 설정 + 배포 | ~12,000 | ~5,000 |
| 모니터링 (~8시간, 15회 체크) | ~35,000 | ~12,000 |
| 결과 분석 + 대시보드 업데이트 | ~15,000 | ~8,000 |
| **합계** | **~82,000** | **~40,000** |

### EC2 비용

| 인스턴스 | 시간 | 비용(추정) |
|---------|------|-----------|
| i-0e35db2d675713696 | 8시간 | ~$8.05 |

### NAS 3단계 전략

| Phase | 시간 | 내용 | Trials |
|-------|------|------|--------|
| Phase 1 | 4.5h | 7개 backbone 탐색 (20ep) | 7 |
| Phase 2 | 3.5h | Top backbone(ConvNeXt-Tiny) HP 최적화 (40ep) | 3 |
| Phase 3 | - | 시간 초과로 미실행 | 0 |

### Phase 1: Architecture Exploration (20ep)

| Rank | Backbone | Params | Score |
|------|----------|--------|-------|
| **1** | **ConvNeXt-Tiny** | 28M | **89.57%** |
| 2 | RegNetY-032 | 19M | 88.97% |
| 3 | WideResNet-50 | 69M | 88.14% |
| 4 | ResNeXt-50 | 25M | 88.03% |
| 5 | EfficientNet-B2 | 9M | 88.03% |
| 6 | ResNet-101 | 45M | 87.98% |
| 7 | ResNet-50 | 26M | 85.23% |

### Phase 2: HP Optimization (40ep, ConvNeXt-Tiny)

| Trial | Head | SE/CBAM | Score |
|-------|------|---------|-------|
| **P2-T11** | **mlp** | **-** | **90.87%** |
| P2-T13 | mlp | SE | 90.85% |
| P2-T12 | mlp_deep | - | 90.76% |

### Best Config: 90.87%

- **Backbone:** ConvNeXt-Tiny (fb_in1k pretrained)
- **Head:** MLP (768→512→100, BN+GELU+Dropout 0.3)
- **LR:** OneCycleLR, max_lr=0.001, backbone 0.1x
- **Augmentation:** Mixup(α=0.2) + CutMix(α=1.0) + RandAugment + RandomErasing
- **Regularization:** Label Smoothing(0.1), Weight Decay(0.05), Dropout(0.3)
- **Training:** 40 epochs, batch 128, AdamW, Mixed Precision

### 핵심 인사이트

1. **ConvNeXt-Tiny가 모든 backbone 중 압도적 1위** — ViT 설계 철학을 CNN으로 구현한 아키텍처가 CIFAR-100 transfer learning에 가장 효과적
2. **RegNetY-032이 효율성 최고** — 19M params로 88.97% (params 대비 성능)
3. **SE block은 ConvNeXt에서 거의 효과 없음** (90.87% → 90.85%) — ConvNeXt 자체가 이미 충분한 channel mixing 보유
4. **mlp_deep(3-layer)보다 mlp(2-layer)가 약간 우수** — 과적합 가능성
5. **20ep→40ep 확장 시 +1.3%p 향상** (89.57% → 90.87%) — 더 긴 학습 여지 있음

### 성능 진화 추적

| 세션 | Best Model | Best Score | 대비 |
|------|-----------|-----------|------|
| 세션 3 (ResNet-50) | ResNet-50 + OneCycleLR | 86.77% | baseline |
| **세션 4 (NAS)** | **ConvNeXt-Tiny + MLP** | **90.87%** | **+4.1%p** |

---

*이후 개발 내역은 이 문서에 세션별로 추가됩니다.*
