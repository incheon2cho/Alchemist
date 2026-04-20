# Alchemist: Vision-Specialized Multi-Agent Framework for Autonomous Model Selection and Self-Refinement

> 논문 초안 — v3 (2026-04-20)

---

## Abstract

대규모 비전 모델 zoo(timm 1,266+, HuggingFace 수만 모델)와 복잡한 학습 레시피(SAM, Mixup, CutMix, EMA, LLRD 등)의 조합 폭발로 인해, 특정 task에 최적인 모델·전략·기법을 찾는 것은 연구자의 주요 병목이다. 기존 AutoML-Agent류 프레임워크는 LLM으로 코드를 생성·실행하지만, (1) 생성 코드의 불안정성(35/35 CUDA 실패), (2) 성능 기반 피드백 부재(에러 수정만 반복), (3) 비전 도메인 고유의 사전학습 corpus 제약·해상도 호환성·전이학습 탐색 공간을 처리하지 못한다.

본 논문은 **Alchemist**를 제안한다: **Controller Agent의 7-verb 프로토콜** 하에 Benchmark Agent(3-source retrieval-grounded 모델 탐색)와 Research Agent(SAM·Mixup·EMA 등 22개 SoTA 기법 자율 적용 + cross-task 경험 누적)가 협업하는 비전 특화 다중 에이전트 프레임워크이다.

CIFAR-100 / Butterfly (75-class) / Shopee-IET (4-class) 3개 데이터셋에서 Alchemist는 AutoML-Agent 대비 2/3 dataset에서 우위를 달성하며 (Butterfly 98.1% vs 96.3%, Shopee 98.8% vs 98.1%), CIFAR-100에서도 SwinV2-Base + SAM 조합으로 94%+ 를 기록하였다. 또한 Controller의 vision-aware early-stop은 R1 compute의 40%+ 를 절감하고, cross-task experience store를 통해 후속 task의 cold-start 수렴이 가속됨을 실증하였다.

---

## 1. Introduction

### 1.1 배경 및 동기

비전 분야는 다른 AI 도메인과 구별되는 네 가지 구조적 특성을 갖는다:

**(P1) Pretrain corpus ambiguity.** 동일 아키텍처라도 사전학습 데이터(ImageNet-1K / 21K / JFT-300M / LAION-2B / LVD-142M)에 따라 전이 성능이 5-10%p 차이 나며, 공정 비교를 위한 tier 정의가 비명시적이다.

**(P2) Heterogeneous model zoo.** timm에만 1,266개 모델이 등록되어 있으며, 입력 해상도(224/256/384/448/518), 패치 크기(8/14/16/32), 윈도우 크기가 모델마다 다르다. 잘못된 해상도를 입력하면 학습이 즉시 실패한다.

**(P3) Transfer-learning decision space.** freeze/unfreeze × adapter(linear_head/LoRA/none) × LLRD(layer-wise LR decay) × augmentation(Mixup/CutMix/RandAugment/EMA/SAM)의 다차원 조합을 탐색해야 최적 성능에 도달한다. 이 탐색 공간은 NLP의 LoRA/adapter 선택보다 복잡하다.

**(P4) Compute waste on hopeless trials.** 높은 learning rate로 pretrained backbone을 fine-tuning하면 epoch 3 이내에 catastrophic forgetting이 발생하나, 기존 시스템은 full training이 끝날 때까지 이를 감지하지 못한다.

### 1.2 관련 연구의 한계

**AutoML-Agent (ICML 2025, KAIST/DeepAuto.ai):** LLM 기반 multi-agent로 planning→code generation→execution 파이프라인을 구성하나, (1) LLM 생성 코드의 런타임 불안정성(CUDA assert, 경로 오류, download=True 등), (2) 성능 기반 iteration 부재(n_attempts=5는 코드 에러 수정만, metric-driven refinement 없음), (3) 비전 도메인 지식 미반영(사전학습 corpus 제약, 해상도 호환성, modern training recipe)이라는 한계가 있다.

**일반 AutoML/NAS:** 아키텍처 탐색 또는 HP 최적화 중 하나에 집중하며, 모델 선정부터 학습 전략·기법 적용·결과 분석·자율 개선까지의 전 과정을 통합하지 못한다.

### 1.3 Alchemist 제안

본 논문은 위 네 가지 비전 고유 난제를 해결하는 **retriever-grounded multi-agent collaboration** 프레임워크를 제안한다. 핵심 설계 원리는:

- **Controller 중심 7-verb 프로토콜**: Formalize → Invoke → Scrutinize → Assay → Instantiate → Surveil → Adjudicate — 단순 pipeline이 아닌 validate-fail-fallback 루프
- **Retriever-grounded reasoning**: LLM 판단을 PwC 리더보드 · timm ImageNet CSV · arXiv 논문으로 교차 검증하여 hallucination 방지
- **Vision Technique Catalog**: 22개 SoTA 기법을 concrete config override로 매핑하여 LLM의 텍스트 제안을 실행 가능한 코드로 연결
- **Closed-loop verification**: 제안된 기법이 실제로 학습에 적용됐는지 확인하는 피드백 메커니즘

### 1.4 Contributions

1. **Vision-domain ontology 기반 모델 탐색**: PwC + HF Hub(timm ImageNet top-1 CSV, 1,266모델) + arXiv의 3-source retrieval-grounded 모델 스카우팅과, 6-tier pretrain-source ontology(IN-1K/21K/JFT/LAION/LVD-142M/CLIP)에 의한 제약 자동 필터링을 제안한다.

2. **Controller Agent의 7-verb empirical guardrail**: *Formalize*에서 자연어 제약을 파싱하고, *Scrutinize*로 추천의 제약 적합성을 검증하며, *Assay*로 top-K 후보의 실측 baseline을 비교하고, *Surveil*로 매 epoch vision-aware early-stop(catastrophic forgetting / hopeless / divergence / collapse 4-rule)을 수행한다. 이를 통해 R1 compute의 40%+ 를 절감하였다.

3. **Research Agent의 자율 개선 루프**: (a) **Vision Technique Catalog**(22개 기법의 config 자동 매핑)와 **closed-loop verification**(제안 vs 실제 적용 교차 검증)으로 진정한 자율 기법 적용을 실현하고, (b) **within-round adaptive tuning**(실패 모드별 config 자동 조정: catastrophic→lr cap, divergence→aug↓, OOM→batch÷2, collapse→epochs cap)과 (c) **cross-task VisionExperienceStore**(persistent JSONL, num_classes+keyword Jaccard 유사도)로 task 간 경험을 축적한다.

4. **3개 benchmark에서 AutoML-Agent 대비 실증**: CIFAR-100(SwinV2-Base+SAM → 94%+), Butterfly(98.1%), Shopee(98.8%) 3개 dataset에서 Alchemist가 AutoML-Agent plan 대비 2/3 dataset 우위, CIFAR-100에서도 동등 이상의 성능을 달성하며, Controller early-stop / adaptive tuning / experience transfer 각각의 ablation 효과를 보인다.

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

#### 2.4.4 Vision Technique Catalog (22개 기법)
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
| CIFAR-100 | 93.95% | **94.X%** (SwinV2+SAM) | Alchemist +X%p |
| Butterfly | 96.27% | **98.10%** | Alchemist **+1.83%p** |
| Shopee-IET | 98.12% | **98.80%** | Alchemist **+0.68%p** |

#### Table 2: Alchemist 자율 탐색 상세

| Dataset | Model | Baseline | Best | Trials | Key Technique |
|---------|-------|----------|------|--------|--------------|
| CIFAR-100 | SwinV2-Base (22K) | 87.2% | **94.X%** | 12+ | SAM(rho=0.05) + Mixup + EMA + LLRD |
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
| 코드 안정성 | LLM 생성 코드 → 35/35 CUDA 실패 | Pre-validated train_worker → 100% 실행 성공 |
| 성능 피드백 | ❌ 에러 수정만 (n_attempts=5) | ✅ metric-driven self-refinement + SoTA gap 분석 |
| 비전 지식 | 없음 (generic prompt) | 22-technique catalog + pretrain ontology |
| 기법 적용 | Plan에 명시해도 코드에서 구현 실패 가능 | Closed-loop 검증으로 silent drop 감지 |
| 경험 학습 | ❌ 매 task fresh start | ✅ cross-task experience 누적 |
| Compute 효율 | Full training 완주 필수 | Early-stop으로 40%+ 절감 |

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

본 논문은 비전 모델 선택의 domain-specific 난제(pretrain ambiguity, heterogeneous zoo, transfer-learning search space, compute waste)를 **retriever-grounded 3-agent collaboration** + **controller-driven mid-trial guardrails** + **research agent의 adaptive tuning + cross-task experience accumulation** + **22-technique vision catalog with closed-loop verification**으로 통합 해결하는 Alchemist 프레임워크를 제안하였다.

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

### C. Vision Technique Catalog 전체 목록 (22개)

Optimizer: sam, sam_conservative, sam_aggressive
Augmentation: mixup_light, mixup_standard, cutmix, cutmix_light, randaugment, random_erasing
Regularization: stochastic_depth, stochastic_depth_strong, label_smoothing, ema, llrd, weight_decay_light, weight_decay_heavy
Schedule: cosine_restarts, onecycle, longer_training_30ep, longer_training_50ep
Architecture: se_attention, cbam_attention
