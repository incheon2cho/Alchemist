# Alchemist: Vision-Specialized Multi-Agent Framework for Autonomous Model Selection and Self-Refinement

> 논문 초안 — v5 (2026-04-20)

---

## Abstract

대규모 비전 모델 zoo(timm 1,266+, HuggingFace 수만 모델)와 학습 기법의 조합 폭발로 인해, 특정 task에 최적인 모델·기법 조합을 찾는 과정이 실무 연구의 주요 병목이다. 이를 자동화하려면 **(1) 유망 모델의 외부 증거 기반 탐색**, **(2) 추천 모델의 실측 검증**, **(3) SoTA 기법의 자율 적용·개선** 세 단계가 유기적으로 연결되어야 하나, 기존 AutoML-Agent류 프레임워크는 LLM 생성 코드의 불안정성(35/35 CUDA 실패)과 에이전트 간 검증 메커니즘 부재로 이를 달성하지 못한다.

본 논문은 **Alchemist**를 제안한다: **Benchmark Agent**(4-source retrieval-grounded 모델 탐색)가 유망 후보를 발굴하고, **Controller Agent**(다중 registry 실측 검증 + vision-aware 실시간 감시 + 품질 판정)가 파이프라인 신뢰성을 보장하며, **Research Agent**(26개 기법 자율 적용 + cross-task 경험 누적)가 성능을 극대화하는 3-agent 협업 하네스이다. 핵심은 선형 pipeline이 아닌 **validate-fail-fallback 루프**로, 각 단계의 실패가 chain 전체를 중단시키지 않고 자동 대안 탐색으로 이어진다.

CIFAR-100 / Butterfly / Shopee-IET 3개 데이터셋에서 동일 제약 하에 Alchemist는 AutoML-Agent plan 대비 2/3 dataset 우위(Butterfly 98.1% vs 96.3%, Shopee 98.8% vs 98.1%), CIFAR-100에서 SwinV2-Base + SAM으로 94.32%를 달성하였다. Controller의 early-stop은 R1 compute 40%+를 절감하고, cross-task experience로 후속 task cold-start가 가속됨을 실증하였다.

---

## 1. Introduction

### 1.1 배경

비전 모델 최적화는 세 단계의 의사결정이 유기적으로 맞물려야 하는 복합 문제이다: **(1) 어떤 모델을 출발점으로 삼을 것인가** (timm, HuggingFace, GitHub 등 수천 개 후보), **(2) 추천된 모델이 현재 파이프라인에서 실제로 로드·학습 가능한지 어떻게 검증할 것인가** (LLM 추론만으로는 모델의 존재 여부, registry 호환성, 해상도 적합성을 판별할 수 없음), **(3) 어떤 기법 조합으로 성능을 극대화할 것인가** (SAM, Mixup, CutMix, EMA, LLRD 등 수십 가지 조합). 기존 연구는 이 세 단계를 개별적으로 다루거나 단일 LLM에 일임하여, 단계 간 정보 단절과 실행 레이어의 silent failure가 발생한다.

AutoML-Agent(ICML 2025)는 LLM 기반 multi-agent로 이를 통합 시도하지만, planning → code generation → execution의 선형 파이프라인에서 **(i) LLM 생성 코드의 런타임 불안정성**(CUDA 초기화 오류, 경로 참조 실패, 원격 환경 비호환 등으로 35/35 실행 실패), **(ii) 성능 기반 피드백 부재**(n_attempts=5의 반복은 코드 에러 수정만 수행하며, validation accuracy 등 metric-driven refinement는 없음)라는 한계를 보인다.

### 1.2 문제 정의: 세 단계의 자율화

본 연구는 모델 탐색 · 검증 · 최적화 세 단계를 **전문화된 에이전트에 분담하고, 에이전트 간 협업 프로토콜로 단절을 해소**하는 접근을 취한다. 각 단계에서 해결해야 할 핵심 과제는:

**(Stage 1 — 탐색) 어떤 모델이 이 task에 가장 유망한가?**
LLM 기반 추천은 두 가지 위험을 내포한다: (a) 존재하지 않는 모델명을 생성하는 **hallucination** (e.g., 어떤 registry에도 없는 이름), (b) 실제로 존재하는 모델이지만 현재 파이프라인에서 로드할 수 없는 **registry 호환성 불일치** (e.g., PwC에 실측 결과가 있는 Astroformer가 timm에는 미등록). 전자는 외부 증거 기반 grounding으로, 후자는 **다중 registry 탐색(timm → torch.hub → GitHub clone)과 실측 baseline 평가**로 해결해야 한다.

**(Stage 2 — 검증) 추천된 모델이 실제로 작동하는가?**
LLM 기반 모델 추천은 다양한 실패 모드를 내포한다: PwC 리더보드의 모델명이 특정 프레임워크(e.g., timm)의 모델 ID와 일치하지 않을 수 있고, 모델마다 요구하는 입력 해상도가 다르며(224/256/384px 등), GitHub에 공개 코드가 있더라도 프레임워크 호환이 안 되거나 pretrained 가중치 로딩 방식이 다를 수 있다. 실제로 Alchemist 개발 과정에서 PwC의 "Astroformer"(GitHub에 구현이 존재하나 timm에 미등록 — **단일 모델 registry 의존의 한계**)나 SwinV2의 256px 해상도 불일치 등을 경험하였다. 이는 다중 registry 탐색(timm + GitHub + HF Hub)의 필요성과, LLM의 symbolic reasoning만으로는 실행 가능성을 판별할 수 없어 **실측 baseline 평가**로 교차 검증해야 함을 보여준다.

**(Stage 3 — 최적화) 어떤 기법을 적용해야 성능이 오르는가?**
에이전트가 "SAM을 쓰라"고 제안해도 실행 레이어에서 silent drop되면 무의미하다. 기법이 **실제로 적용됐는지 검증**하고, 실패 시 다음 trial을 **자동 조정**하며, 이전 task 경험을 **축적·전이**하는 closed-loop 자기 개선이 필요하다.

### 1.3 제안: Alchemist — 3-Agent 협업 하네스

Alchemist는 세 단계를 각각 전문화된 에이전트에 배정하고, **Controller Agent의 실측 검증 + 실시간 감시 프로토콜**로 에이전트 간 협업을 orchestrate하는 하네스이다.

#### Benchmark Agent — Multi-Source Retrieval-Grounded 모델 탐색 (Stage 1)

단일 LLM 판단이 아닌 **4-source 증거 융합**으로 후보를 발굴한다:
- **HuggingFace Hub**: timm 공식 ImageNet top-1 CSV(1,266모델)에서 실측 성능 기반 랭킹. 모델별 입력 해상도 · 파라미터 수 · 사전학습 소스 메타데이터 자동 추출.
- **Papers-with-Code**: task-specific 리더보드(e.g., CIFAR-100 top-1 accuracy)에서 실제 SoTA 수치와 사용 기법 수집.
- **GitHub**: 공개 모델 저장소 검색 — `torch.hub` 호환 여부(`hubconf.py` 존재), pretrained 가중치 가용성, star 수를 기준으로 논문 공식 구현체·커뮤니티 재구현 탐색. timm에 미등록된 최신 모델도 발견 가능.
- **arXiv**: 2023-2025 최신 논문에서 유망 아키텍처·기법 컨텍스트 확보.

네 소스의 증거��� **실�� 가능성 우선**(HF Hub timm ID 보유 → GitHub torch.hub/clone 로드 가능 → PwC 참조) 순위로 융합���다. Task 제약에 위반되는 후보를 자동 제��한 ���, **Top-K 후보 리스트**를 Controller에 전달한다. 이를 통해 (a) hallucinated 모델은 4-source 어디에도 증거가 없어 자연스럽게 탈락하고, (b) PwC에 실적이 있지만 특정 registry에 미등록인 모델(e.g., Astroformer)도 **GitHub clone + 자동 아키텍처 탐지**를 통해 파이프라인에 편입할 수 있다.

#### Controller Agent — Empirical 검증 + 실시간 감시 (Stage 2 + 전체 orchestration)

Controller는 파이프라인의 **중앙 통제자**로서, 세 가지 핵심 역할을 수행한다:

**① 추천 모델의 실행 가능성 검증.** Benchmark Agent가 추천한 모델이 실제로 로드·학습 가능한지를 다층으로 검증한다. 먼저 LLM 기반으로 제약 적합성(사전학습 소스, HW 예산, 입력 해상도)을 점검하고, timm 모델 registry 또는 GitHub `torch.hub` 호환성을 확인한다. 추천명이 현재 지원하는 registry(timm, GitHub torch.hub)에 매칭되지 않으면(e.g., "Astroformer" — GitHub 코드 존재하나 timm 미등록), 자동으로 리더보드를 탐색하여 **즉시 실행 가능한 대안 후보**로 fallback한다. 최종적으로 Top-K 후보를 **실제 GPU에서 frozen baseline 평가**하여 LLM 추론의 정확성을 실측으로 교차 검증하고, 최고 성능 후보를 Winner로 선정한다.

**② 학습 중 실시간 감시와 선제 개입 (Vision-Aware Early-Stop).** Research Agent가 실행하는 각 trial의 매 epoch 진행 상황(validation accuracy, train loss)을 모니터링하고, 네 가지 vision-specific 이상 징후를 감지하면 trial을 선제 종료한다:
- *Catastrophic forgetting*: val < baseline − 10%p (epoch ≥ 3) — 과도한 learning rate로 pretrained 가중치가 파괴되는 현상
- *Hopeless trial*: val + 5%p < best_so_far (epoch ≥ 5) — 잔여 epoch으로 gap을 회복할 수 없는 경우
- *Optimizer divergence*: train_loss > 3.0 (epoch ≥ 3) — 옵티마이저 불안정
- *Mid-training collapse*: score < 5% with epochs > 10 — 정밀도 overflow 등으로 인한 학습 붕괴

이를 통해 hopeless trial에 소비되는 GPU 시간을 **40% 이상 절감**하고, 동일 budget 내에서 유효한 실험 수를 극대화한다.

**③ 최종 품질 판정.** Research Agent의 최종 결과를 baseline 대비 개선폭, 절대 score, budget 사용률 기준으로 평가하여 ship/no-ship 판정을 내린다.

#### Research Agent — Autonomous Vision Expert (Stage 3)

검증된 Winner 모델을 받아 **자율적으로 최고 성능 기법 조합을 탐색**한다:

- **Vision Technique Catalog (26개 SoTA 기법):** 기법명을 train_worker가 이해하는 concrete config override로 매핑(e.g., "stochastic_depth" → `{drop_path_rate: 0.1}`, "sam_aggressive" → `{optimizer: "sam", sam_rho: 0.1}`). LLM이 제안한 텍스트와 실행 레이어 사이의 semantic gap을 해소하여, **제안된 기법이 실제 학습 코드에 정확히 반영**되게 한다.
- **Closed-loop technique verification:** 매 trial 완료 후 제안된 기법(proposed config)과 실제 적용된 기법(train_worker의 applied_techniques)을 비교. 기법이 실행 레이어에서 silent drop된 경우 즉시 감지하여 에이전트의 잘못된 판단("SAM을 적용했는데 효과 없었다")을 방지.
- **Within-round adaptive tuning:** 실패 모드별 후속 trial 자동 조정 (catastrophic → lr cap, OOM → batch÷2 with 연속 성공 시 forgiveness rule, collapse → epochs cap + warmup 강화).
- **Cross-task VisionExperienceStore:** 완료된 task의 winning config · 기법 효과 · 실패 경험을 persistent memory에 기록. 다음 task 시작 시 유사 경험(num_classes · task 키워드 기반)을 검색하여 LLM 프롬프트에 "prior experience" 섹션으로 주입 → cold-start 가속 및 에이전트의 점진적 전문화.

#### 하네스의 핵심: Validate-Fail-Fallback 루프

Alchemist의 3-agent 협업은 **선형 파이프라인이 아닌 검증-실패-대안 탐색 루프**로 동작한다:

```
Benchmark 추천 "Astroformer" (PwC 93.4%, GitHub 코드 존재)
  → Controller 검증: timm 미등록, torch.hub 비호환 → 현재 파이프라인에서 즉시 로드 불가 → FAIL
  → Controller: leaderboard walk → 다음 후보 "SwinV2-Base" 시도
  → Controller 실측: EC2 baseline 87.8% + img_size=256 자동 감지 → PASS → Winner
  → Research: SAM(rho=0.05) trial 시작
  → Controller 감시: epoch 3 val=70% < baseline-10% → KILL (catastrophic)
  → Research Adapt: lr cap 적용 → 다음 trial 안정적 수렴 → 94.32% 달성
```

각 단계에서 **실패가 chain 전체를 중단시키지 않고, 자동으로 대안을 탐색**한다. 이것이 AutoML-Agent의 선형 pipeline(plan → codegen → execute → error → retry same code)과의 구조적 차별점이다.

### 1.4 Contributions

1. **3-Agent 협업 하네스와 validate-fail-fallback 프로토콜**: Benchmark(4-source retrieval-grounded 탐색: HF Hub + PwC + GitHub + arXiv) → Controller(다중 registry 검증 + 실측 baseline 평가 + vision-aware 실시간 감시) → Research(자율 최적화)의 역할 분담과, 검증-실패-대안 탐색 루프에 의한 robust orchestration을 제안한다. Controller의 실측 검증(top-K baseline eval)과 실시간 감시(epoch-level early-stop, 40%+ compute 절감)가 LLM 추론의 한계를 empirical barrier로 보완한다.

2. **Vision Technique Catalog + Closed-loop verification**: 26개 SoTA 비전 기법을 executable config로 매핑하고, 제안-실행 간 일치를 검증하는 피드백 루프를 구축하여 에이전트의 기법 적용 신뢰성을 보장한다.

3. **Cross-task experience accumulation + adaptive tuning**: 실패 모드별 within-round config 자동 조정과, task 간 persistent memory 축적으로 에이전트가 점진적으로 비전 전문가로 성장한다.

4. **3개 benchmark에서 실증**: 동일 제약 하에 Alchemist의 자율 탐색(SwinV2+SAM 94.32% / Butterfly 98.1% / Shopee 98.8%)이 AutoML-Agent plan 대비 우위 또는 동등 성능을 달성하며, 각 에이전트·하네스 구성요소의 ablation 효과를 보인다.

---

## 2. Method

### 2.1 Framework Overview

Alchemist는 **Controller Agent**를 중심축으로, 좌측의 **Benchmark Agent**(모델 탐색·추천)와 우측의 **Research Agent**(실험·자율 개선)가 typed schema 메시지로 협업하는 3-agent 하네스이다. 모든 학습 실행은 AWS EC2 GPU에 위임되며, Controller가 SSH fire-and-forget 제출 + per-epoch polling으로 원격 학습을 관리한다.

### 2.2 Benchmark Agent — Retrieval-Grounded Model Scouting

**3-source 증거 수집:**
- *PwC Scout*: `search_pwc_leaderboard(task_name)` → 해당 task의 real top-1 accuracy와 uses_additional_data 플래그 수집. CIFAR-100에서 15개 SoTA 엔트리, Butterfly에서 0개(task별 가용 여부에 적응).
- *HF Hub + timm CSV Scout*: 1,266모델의 공식 ImageNet top-1 · param_count · img_size 메타데이터 로딩. `classify_pretrain_source`로 6-tier 분류(IN-1K/21K/JFT/LAION/LVD-142M/CLIP). HF-first → PwC-second 소스 우선순위 랭킹.
- *arXiv Scout*: 2023-2025 year-filtered 최신 논문 5개 수집. Research Agent의 SoTA 기법 탐색에 context로 전달.

**Constraint-aware ranking**: task.description의 자연어 제약("ImageNet-21K allowed, NOT JFT/LAION")을 파싱하여 non-compliant 후보를 자동 제외. CIFAR-100 실험에서 13/15 PwC 엔트리가 JFT/LAION 사전학습으로 제외됨.

**Top-K candidate 리스트**: 최종 순위 상위 3개 후보를 Leaderboard.candidates에 저장하여 Controller에 전달.

### 2.3 Controller Agent — 7-Verb Empirical Guardian

| Verb | 역할 | 비전 특화 요소 |
|------|------|---------------|
| **Formalize** | 자연어 task를 typed 제약으로 파싱 | Pretrain corpus tier, img_size, HW budget |
| **Invoke** | Benchmark Agent에 지시 전달 | Search query + benchmark 종류 지정 |
| **Scrutinize** | 추천의 제약 적합성 LLM 검증 | Pretrain corpus 준수, VRAM fit, img_size 호환 |
| **Assay** | Top-K 후보 실측 baseline 평가 | EC2에서 frozen linear-probe + timm ID resolution |
| **Instantiate** | Winner를 Research Agent에 위임 | Upstream context(benchmark 근거, SoTA standing) 전달 |
| **Surveil** | 매 epoch progress.json 모니터링 + 4-rule early-stop | Catastrophic/hopeless/divergence/collapse 감지 |
| **Adjudicate** | 최종 결과의 ship/no-ship 판정 | Baseline 대비 improvement + 절대 score 기준 |

**Vision-aware early-stop (Surveil) 세부:**
1. `val < baseline − 10%p` at epoch ≥ 3 → **catastrophic forgetting** (lr 과대 시 pretrained 가중치 파괴)
2. `val + 5%p < best_so_far` at epoch ≥ 5 → **hopeless** (잔여 epoch으로 gap 회복 불가)
3. `train_loss > 3.0` at epoch ≥ 3 → **optimizer divergence** (SAM rho 과대 등)
4. `score < 5%` with `epochs > 10` → **mid-training collapse** (fp16 overflow, EMA 오염 등)

**실측 효과**: CIFAR-100 R1에서 lr=1e-3 unfreeze trials 2개를 epoch 3에서 kill → 각 ~60분 × 2 = **120분 절감**.

### 2.4 Research Agent — Vision-Expert Self-Refinement Loop

#### 2.4.1 R0: Baseline 측정
Frozen linear-probe로 base model의 기초 전이 성능 측정. SwinV2-Base(22K) baseline = 87.2%.

#### 2.4.2 R1: SAM-focused HP Grid
**SAM(Sharpness-Aware Minimization)** 중심 탐색. Freeze trial은 baseline으로 이미 측정되었으므로 R1에서 제외하고, **unfreeze + SAM sweep**에 12 trial 전부 집중:

| 탐색 축 | 범위 | 비고 |
|---------|------|------|
| lr | {1e-4, 3e-4, 1e-3, 3e-3} | Controller가 catastrophic lr 자동 kill |
| SAM rho | {0.02, 0.05, 0.1} | Conservative / standard / aggressive |
| Advanced tech | Mixup α=0.3, CutMix α=0.5, RandAug, EMA 0.9999, LLRD 0.7, label_smoothing 0.1 | SAM + 약화된 regularization |

**Model-size-aware batch scaling**: ≥50M params → batch=32, <50M → batch=64 (A10G 24GB 기준).

**img_size 자동 감지**: `timm.data.resolve_model_data_config()` → SwinV2(256px), MaxViT(256px), ViT(224px) 등 모델별 native 해상도 자동 적용.

#### 2.4.3 Within-Round Adaptive Tuning
매 trial 실패 시 failure 모드를 분류하고 후속 trial config를 자동 조정:
- **catastrophic** → 후속 unfreeze lr을 1e-3으로 cap
- **divergence** → Mixup α 0.4, CutMix α 0.5로 약화 + lr 3e-4로 하향
- **OOM** → batch_size 절반 (consecutive success 2회 후 용서 규칙으로 원복)
- **collapse** → epochs를 10으로 cap + warmup 15% + batch 64 하한

#### 2.4.4 Vision Technique Catalog (26개 기법)
SoTA 기법명 → concrete config override 매핑 사전:

| 카테고리 | 기법 예시 | Config override |
|---------|----------|----------------|
| Optimizer | sam, sam_conservative, sam_aggressive | `optimizer="sam", sam_rho=0.02/0.05/0.1` |
| Augmentation | mixup_light, cutmix, randaugment, random_erasing | `mixup_alpha=0.2, cutmix_alpha=1.0, ...` |
| Regularization | stochastic_depth, label_smoothing, ema, llrd | `drop_path_rate=0.1, backbone_lr_scale=0.7` |
| Schedule | cosine_restarts, onecycle, longer_training_30ep | `lr_schedule="cosine_restarts", epochs=30` |
| Architecture | se_attention, cbam_attention | `add_se=True, add_attention=True` |

`suggest_techniques()`가 catalog-driven(미시도 기법 자동 추출) + LLM-driven(catalog을 menu로 제시하여 조합 제안) 2-path 방식으로 config 생성.

#### 2.4.5 Closed-Loop Technique Verification
train_worker의 result에 `applied_techniques` dict를 포함하여 **실제 적용된 기법을 기록**. Research Agent가 매 trial 후 proposed config vs applied_techniques를 비교하여 silent drop(기법이 제안됐으나 실행 레이어에서 무시됨)을 감지.

이전 실험에서 suggest_techniques가 `optimizer="sam"`을 제안했으나 train_worker에 SAM 미구현으로 silent AdamW fallback → **agent가 "SAM 적용" 착각** 문제를 이 메커니즘으로 해결.

#### 2.4.6 Cross-Task Experience Store
`VisionExperienceStore`가 완료된 task의 winning config · techniques · score를 persistent JSONL에 기록. 다음 task 시작 시 `retrieve_similar(task_name, num_classes, top_k=3)`로 유사 경험을 검색하여 `search_sota()` 프롬프트의 "Prior experience" 섹션에 주입.

**유사도 메트릭**: num_classes bucket(tiny/small/medium/large) × keyword Jaccard (task name + description).

**collapse 경고 전파**: 이전 task에서 mid-training collapse 발생 시 "WARNING: keep epochs≤10 or increase warmup to 15%" 메시지가 experience에 기록되어 후속 task가 같은 실수를 방지.

### 2.5 Training Infrastructure

**train_worker.py 주요 기능:**
- **bfloat16 mixed precision**: A10G(Ampere) native bf16으로 fp16 overflow 근원 차단
- **SAM optimizer**: 2-step optimization (ascend + descend) with GradScaler + gradient clipping
- **EMA per-batch update**: epoch 당 1회 → batch 당 1회로 수정하여 shadow 가중치 안정화 (EMA shadow의 NaN 오염 방어 포함)
- **Warmup 자동 스케일링**: epochs의 10% 이상 보장 + LR floor(max(1e-7, lr×0.01))
- **NaN 즉시 감지**: train_loss NaN/Inf 시 즉시 halt
- **img_size 자동 감지**: timm.data.resolve_model_data_config()로 모델별 native 해상도 적용
- **Per-epoch progress.json**: Controller Surveil을 위한 실시간 모니터링 파일

---

## 3. Experiments

### 3.1 실험 설정

| 항목 | 값 |
|------|-----|
| Hardware | AWS g5.2xlarge (NVIDIA A10G 24GB VRAM, 8 vCPU, 32GB RAM) |
| LLM | Claude Sonnet 4.6 via Claude CLI |
| Pretrain 제약 | ImageNet-1K + ImageNet-21K 허용, JFT/LAION/LVD-142M 불허 |
| Datasets | CIFAR-100 (50K train, 100 classes), Butterfly (9.3K, 75 classes), Shopee-IET (640, 4 classes) |
| Budget | CIFAR 8h, Butterfly 4h, Shopee 2h GPU wall-clock |

### 3.2 비교 대상

**Hybrid (AutoML-Agent plan):** AutoML-Agent(ICML'25)의 plan-only output(backbone, lr, wd, Mixup α, CutMix α, EMA, LLRD 등)을 Alchemist의 안정적 train_worker로 실행. AutoML-Agent의 자체 코드 생성은 35/35 CUDA assert 실패로 불가 → plan 품질만 공정 비교.

**Alchemist (autonomous):** Benchmark Agent 모델 추천 → Controller top-K eval → Research Agent SAM sweep + adaptive tuning.

### 3.3 주요 결과

#### Table 1: 3-Dataset Comparison (ImageNet-21K 허용)

| Dataset | Hybrid (AMLA plan) | Alchemist (ours) | Δ |
|---------|-------------------|-------------------|---|
| CIFAR-100 | 93.95% | **94.32%** (SwinV2+SAM) | Alchemist **+0.37%p** |
| Butterfly | 96.27% | **98.10%** | Alchemist **+1.83%p** |
| Shopee-IET | 98.12% | **98.80%** | Alchemist **+0.68%p** |

#### Table 2: Alchemist 자율 탐색 상세

| Dataset | Model | Baseline | Best | Trials | Key Technique |
|---------|-------|----------|------|--------|--------------|
| CIFAR-100 | SwinV2-Base (22K) | 87.3% | **94.32%** | 30ep | SAM(rho=0.05) + Mixup + EMA + LLRD |
| Butterfly | ConvNeXtV2-Base (22K) | 95.7% | **98.1%** | 17 | AdamW + advanced tech |
| Shopee-IET | ConvNeXtV2-Base (22K) | 97.5% | **98.8%** | 16 | AdamW + advanced tech |

### 3.4 Ablation Studies

#### Table 3: Controller Early-Stop 효과

| 설정 | R1 walltime | Kill 횟수 | 절감 시간 | Best score 영향 |
|------|-------------|----------|----------|---------------|
| Early-stop OFF | ~12h | 0 | 0 | 동일 (hopeless trial도 끝까지 실행) |
| Early-stop ON | ~7h | 5 | **~4h (40%)** | 동일 (kill된 trial은 어차피 저성능) |

#### Table 4: Adaptive Tuning 효과

| 설정 | OOM 후 동작 | 반복 catastrophic | Best score |
|------|------------|------------------|-----------|
| Adapt OFF | crash → chain 종료 | lr=1e-3 반복 실패 | N/A |
| Adapt ON | batch÷2 자동 → 학습 계속 | lr cap 적용 → 회피 | 93.8%+ |

#### Table 5: Cross-Task Experience Transfer

| Task 순서 | Experience 입력 | R1 첫 trial val | Best score |
|-----------|---------------|-----------------|-----------|
| CIFAR (1번째) | 없음 | 88.3% (scratch) | 93.8% |
| Butterfly (2번째) | CIFAR 경험 | 95.7% (baseline↑) | 98.1% |
| Shopee (3번째) | CIFAR+Butterfly | 97.5% (baseline↑) | 98.8% |

### 3.5 AutoML-Agent와의 질적 비교

| 관점 | AutoML-Agent | Alchemist |
|------|-------------|-----------|
| 실행 방식 | LLM이 학습 코드 직접 생성 → 런타임 불안정(35/35 CUDA 실패) | Pre-validated train_worker + config 전달 → 코드 생성 불필요 |
| 성능 피드백 | ❌ 에러 수정만 (n_attempts=5, metric 미참조) | ✅ metric-driven self-refinement + SoTA gap 분석 |
| 기법 탐색 | Plan에 기법 명시 가능하나, 생성 코드 품질에 의존 | 26-technique catalog → config로 자동 매핑 + closed-loop 검증 |
| 모델 탐색 | LLM 내부 지식 기반 (외부 검증 없음) | 4-source retrieval (HF+PwC+GitHub+arXiv) + 실측 baseline 검증 |
| 경험 학습 | ❌ 매 task fresh start | ✅ cross-task experience 누적 |
| Compute 효율 | 코드 에러 시 재시도 반복 (동일 budget 소모) | Early-stop + adaptive tuning으로 hopeless trial 선제 종료 |

> **주의**: AutoML-Agent의 plan 품질 자체는 우수하였다 (ConvNeXt-Base + LLRD 0.7 + Mixup 0.8 + EMA 등 적절한 조합 제안). 한계는 plan을 실행 가능한 코드로 안정적으로 변환하는 과정과, 실행 결과를 기반으로 plan을 개선하는 피드백 루프에 있다.

---

## 4. Analysis & Discussion

### 4.1 비전 특화의 필요성

Alchemist의 비전 특화 요소를 제거한 ablation:
- Pretrain ontology 없이 → dinov2(LVD-142M) 선택 → 제약 위반 (validator 필요)
- img_size auto-detect 없이 → SwinV2(256px) 실패 → 차선 모델로 fallback (성능↓)
- Technique catalog 없이 → LLM이 "SAM 써라" 제안하지만 config에 매핑 안 됨 → silent drop

### 4.2 Self-Refinement의 closed-loop 중요성

이전 구현에서 suggest_techniques()가 optimizer="sam"을 제안했으나 train_worker에 SAM 미구현 → AdamW로 실행 → agent가 "SAM 적용" 착각. Closed-loop verification 도입 후 MISMATCH 즉시 감지.

### 4.3 EMA per-batch update 발견

EMA를 epoch 당 1회 update(기존)에서 batch 당 1회(수정)로 변경함으로써, epoch 11에서 발생하던 학습 붕괴(93%→1%) 완전 해소. 원인: per-epoch EMA는 10 epochs 후에도 shadow가 99.9% 초기(random head) 가중치 → validation 시 random 예측.

### 4.4 bf16 vs fp16 Mixed Precision

A10G에서 fp16 autocast는 10 epoch 이상의 장기 학습에서 activation overflow(max 65504) → NaN 전파. bf16(fp32 동일 exponent 범위)으로 전환하여 근원적 해결.

### 4.5 한계 및 향후 연구

- **LLM 타임아웃**: Claude CLI 120초 제한으로 suggest_techniques가 자주 fallback 의존. Async LLM call 또는 batch reasoning으로 개선 가능.
- **단일 GPU 제약**: A10G 24GB에서 ViT-Large 이상 모델은 batch=8 이하로 제한. Multi-GPU 또는 gradient accumulation 통합 필요.
- **Experience store 유사도**: 현재 keyword Jaccard + num_classes bucket으로 단순. Task embedding 기반 유사도로 정교화 가능.
- **Architecture search 미통합**: 현재 Benchmark Agent가 선택한 단일 아키텍처를 최적화. 복수 아키텍처 병렬 탐색으로 확장 가능.

---

## 5. Conclusion

본 논문은 비전 모델 선택의 domain-specific 난제(pretrain ambiguity, heterogeneous zoo, transfer-learning search space, compute waste)를 **retriever-grounded 3-agent collaboration** + **controller-driven mid-trial guardrails** + **research agent의 adaptive tuning + cross-task experience accumulation** + **26-technique vision catalog with closed-loop verification**으로 통합 해결하는 Alchemist 프레임워크를 제안하였다.

Alchemist는 AutoML-Agent 대비 (1) 코드 안정성(pre-validated worker), (2) 성능 기반 자율 개선(metric-driven refinement), (3) 비전 도메인 지식(pretrain ontology + technique catalog), (4) compute 효율(early-stop 40%+ 절감), (5) 경험 축적(cross-task transfer)에서 우위를 보이며, 3개 benchmark에서 이를 실증하였다.

향후 연구로는 (1) task embedding 기반 cross-task transfer 정교화, (2) multi-GPU distributed training 통합, (3) 에이전트 간 negotiation protocol 고도화, (4) 비전 외 도메인(NLP, speech)으로의 technique catalog 확장을 계획한다.

---

## Appendix

### A. AutoML-Agent 실패 분석

AutoML-Agent(Claude CLI + EC2 실행)의 CIFAR-100 실험에서:
- 35/35 execute_script 호출이 CUDA assert 실패
- 원인: LLM 생성 코드의 `device = "cuda" if torch.cuda.is_available() else "cpu"` 패턴이 원격 실행 환경에서 CUDA 초기화 레이스 컨디션 유발
- code-level iteration(n_attempts=5)이 동일 패턴 반복 → 해결 불가
- Plan 자체는 양호(ConvNeXt-Base + LLRD 0.7 + Mixup 0.8 + EMA 0.9999) → Hybrid 실험의 근거

### B. EC2 비용 분석

| 기간 | GPU 시간 | 비용 |
|------|---------|------|
| 세션 1-10 (이전 실험) | ~210h | ~$260 |
| 세션 11 (본 비교 실험) | ~77h | ~$63 |
| 세션 11+ (SAM/SwinV2 추가) | ~10h+ | ~$12+ |
| **합계** | **~297h** | **~$335** |

### C. Vision Technique Catalog 전체 목록 (26개)

Optimizer (3): sam, sam_conservative, sam_aggressive
Augmentation (6): mixup_light, mixup_standard, cutmix, cutmix_light, randaugment, random_erasing
Regularization (7): stochastic_depth, stochastic_depth_strong, label_smoothing, ema, llrd, weight_decay_light, weight_decay_heavy
Schedule (4): cosine_restarts, onecycle, longer_training_30ep, longer_training_50ep
Architecture (6): se_attention, cbam_attention, self_attention_2d, lora_attn, lora_qkv, adapter_houlsby
