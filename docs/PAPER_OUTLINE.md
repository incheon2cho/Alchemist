# Alchemist: A Multi-Agent Framework for Vision Model Selection with Cross-Task Experience and Mid-Trial Self-Refinement

> 논문 초안 구성 (Outline) — v2

---

## Introduction

### 1. 배경
수백 개의 공개 vision backbone(ConvNeXt / ViT / Swin / MaxViT / Mamba)과 다층화된 사전학습 corpus(ImageNet-1K / 21K / JFT / LAION / LVD-142M)가 공존하면서, 특정 task에 적합한 모델·전이학습 전략·augmentation 레시피를 고르는 과정이 실무 연구의 주요 병목이 되었다.
기존 AutoML/NAS 연구는 구조 탐색 또는 HP 최적화 중 하나에 집중하며, **모델 선정부터 학습 실행·결과 분석·전문성 축적까지 통합적으로 자동화**하는 vision-domain-aware 프레임워크는 부재하다.
최근 LLM의 추론 능력과 retrieval-augmented reasoning의 발전으로, agent가 vision 도메인 지식을 grounded reasoning으로 활용할 가능성이 열렸다.

### 2. 문제정의
Vision model selection은 NLP/speech 와 구분되는 네 가지 구조적 난제를 갖는다:
- **(P1) Pretrain corpus ambiguity**: 모델 성능이 pretrain 데이터(IN-1K vs 21K vs JFT vs LVD-142M)에 극도로 민감하며, 공정 비교를 위한 tier 정의가 비명시적이다.
- **(P2) Heterogeneous model zoo**: timm에만 1,266개 모델이 있고 input resolution(224/256/384/518), patch size, window size가 제각각이다.
- **(P3) Transfer-learning decision space**: freeze × adapter × LLRD × augmentation의 4-dim 조합을 모두 탐색해야 최적 성능에 도달한다.
- **(P4) Compute waste on hopeless trials**: full training(10-20 epoch)을 전부 돌려봐야 성능을 알 수 있어, 망하는 configuration에도 full budget이 소진된다.

기존 AutoML-Agent류는 코드 레벨 에러 수정만 반복하며 성능 기반 자기 개선·cross-task 경험 축적이 없다.
우리는 이러한 gap을 메우기 위해 **retriever-grounded multi-agent collaboration**과 **within-run adaptive tuning + cross-task experience memory**를 결합한 프레임워크를 제안한다.

### 3. 제안기술 요약
Alchemist는 **Benchmark Agent**(3-소스 모델 탐색·순위화·top-K 후보), **Controller Agent**(제약 검증·top-K baseline eval·winner 선정·**vision-aware mid-trial early-stop**), **Research Agent**(transfer-learning primitive 탐색·**adaptive config tuning**·self-analysis·**cross-task experience accumulation**)의 3-agent 협업 하네스로 구성된다.

### 4. Contribution
1. **Vision-domain ontology 기반 3-agent collaboration**: Benchmark가 PwC + HF Hub (timm ImageNet top-1 CSV 1266모델) + arXiv 3-source retrieval을 통해 task-specific leaderboard에 grounding 된 후보를 추천하고, Controller가 model-ID resolution + pretrain-corpus constraint 검증으로 LLM hallucination 을 empirical 하게 교차 검증하는 설계를 제안한다.
2. **Vision-aware mid-trial early-stopping**: Controller Agent가 epoch-level val accuracy를 받아 **(i) catastrophic forgetting (val < baseline − 10%p)**, **(ii) hopeless-vs-best-so-far (val + 5%p < best)**, **(iii) optimizer divergence (train_loss > 3.0)** 의 vision-specific 휴리스틱으로 trial을 선제 종료하여 R1 compute budget 의 40%+ 를 절감한다.
3. **Research Agent 의 adaptive tuning 및 cross-task experience accumulation**:
  - **Within-round**: 실패 모드(catastrophic / divergence / OOM)를 누적해 후속 trial의 lr / augmentation α / batch size 를 자동 하향 조정한다.
  - **Cross-task**: 완료된 task의 (winning config, techniques) 를 `VisionExperienceStore`(persistent JSONL)에 기록하고, 다음 task 시작 시 유사 task(num_classes bucket + keyword Jaccard)를 검색해 LLM prompt에 "prior experience" 섹션으로 주입 — *agent becomes a progressively more expert vision researcher.*
4. **CIFAR-100 / Butterfly / Shopee-IET에서 AutoML-Agent 대비 우위 실증**: 21K-허용 공정 비교(AutoML-Agent plan + convnext 21K variant vs Alchemist autonomous search)에서 Alchemist가 더 높은 top-1 을 달성하며, cross-task memory ablation에서 Butterfly / Shopee 의 R1 cold-start 수렴 속도가 유의미하게 단축됨을 보인다.

---

## Method

### 1. Framework Overview
3개의 전문화된 에이전트가 Controller를 중심축으로 협업한다. Benchmark Agent 가 좌측(모델 탐색 / top-K 추천), Research Agent 가 우측(trial 실험 / 자기 개선)에 위치하며, 모든 에이전트 간 통신은 typed schema(`Leaderboard`, `TrialConfig`, `ResearchResult`) 로 전달되어 LLM 환각이 silent 하게 전파되지 않는다.

### 2. Benchmark Agent — Retrieval-Grounded Model Scouting
- **PwC scout**: `search_pwc_leaderboard(task_name)` 로 해당 task 의 real top-1 accuracy를 보유한 SoTA 엔트리 수집. `uses_additional_data` 필드로 제약 위반 자동 탐지.
- **HF Hub + timm CSV scout**: 1266모델 공식 ImageNet top-1 (크기·해상도 메타데이터 포함) 로딩. `search_imagenet1k_models` + `classify_pretrain_source` 로 IN-1K / 21K / JFT / LAION / LVD-142M / CLIP 6-tier 분류.
- **arXiv scout**: 2023-2025 최신 논문에서 SoTA 기법 컨텍스트 수집.
- **Ranking**: *HF-first → PwC-second* 소스 우선순위로 **실제 로드 가능한 모델(HF Hub timm ID)이 무명 PwC 엔트리보다 상위**로 정렬. 이후 LLM-assisted `recommend()` 가 top-3 후보 리스트를 반환.

### 3. Controller Agent — Vision-Aware Guardian
- **Recommendation validation**: 제약(pretrain 소스, HW budget, img_size 호환) 검증. 실패 시 leaderboard walk 로 fallback.
- **Top-K baseline eval**: 추천 top-3 후보를 EC2 에서 실제 frozen linear-probe 로 빠르게 평가하여 winner 선정. LLM 추천의 empirical validation 역할.
- **Mid-trial early-stop (본 연구 기여)**: Research Agent 의 각 trial 실행 중 `train_worker` 가 epoch 마다 작성하는 `progress.json` 을 polling 하여 다음 네 가지 중 하나가 관측되면 SSH-kill 로 선제 종료:
  1. val_acc < baseline − 10%p (epoch ≥ 3) — catastrophic forgetting
  2. val_acc + 5%p < best_so_far (epoch ≥ 5) — hopeless
  3. train_loss > 3.0 (epoch ≥ 3) — optimizer divergence
  4. (향후) plateau 3 epoch 이상 — no improvement
- **Ship judgment**: Research 결과를 baseline 대비 개선 폭 / 절대 score / budget 사용률 기준으로 Ship / No-Ship 결정.

### 4. Research Agent — Adaptive & Experience-Accumulating Expert

#### 4-1. R1: Transfer-Learning Primitive Grid
`lr × freeze × adapter × (advanced-tech ON/OFF)` 의 4차원 grid. **Unfreeze trial 은 기본적으로 Mixup(α=0.8) + CutMix(α=1.0) + RandAugment + label smoothing 0.1 + EMA 0.9999 + LLRD 0.7** 를 default prior로 활성화 (ConvNeXt / DeiT modern training recipe 반영). Freeze trial 은 linear probe baseline 용으로 basic aug.

#### 4-2. Adaptive Config Tuning (within-round)
각 trial 의 실패 모드를 failures 리스트에 누적하고 `_adapt_config_from_failures()` 가 다음 trial 전 자동 수정:
- `catastrophic` → 후속 unfreeze trial 의 lr 을 1e-3 로 cap
- `divergence` → mixup α 0.4, cutmix α 0.5 로 약화 + lr 3e-4 로 하향
- `oom` → batch_size 절반

#### 4-3. Cross-Task Experience Memory (`VisionExperienceStore`)
- **record()**: 완료된 run의 task / baseline / best / winning config / techniques를 `~/.cache/alchemist/experience.jsonl` 에 append.
- **retrieve_similar()**: num_classes bucket(tiny/small/medium/large) + keyword Jaccard 유사도로 top-K 과거 경험 검색.
- **Integration**: `search_sota()` 프롬프트의 "## Prior experience on similar vision tasks" 섹션에 주입 → LLM 이 이전 task 에서 통한 기법을 신규 task 에 우선 제안.

#### 4-4. SoTA-Gap-Driven Self-Refinement
R1 완료 후 `analyze_sota_gap()` 이 현재 best 와 PwC SoTA 간 gap 을 분석하고 `should_continue_research()` 가 **task-adaptive stopping** 판단. Continue 시 R2 에서 `suggest_techniques()` 로 SAM / longer schedule / stronger aug 등 explicit technique 제안 → `TrialConfig` 20개 필드 (mixup / cutmix / ema / sam_rho / backbone_lr_scale / warmup_epochs 등) 로 fully-typed 전달.

### 5. System: LLM-Hardened Orchestration
- **Typed schemas** (`Leaderboard.candidates`, `TrialConfig` 의 20 필드): agent 간 JSON round-trip 에서 필드 누락이 silent failure 가 되지 않도록 강제.
- **Retriever grounding**: LLM output 이 retriever evidence 에 anchor 되도록 prompt 설계(e.g., "PwC 에 Astroformer 가 93.4% 로 기록됨").
- **Constraint propagation**: task.description 의 자연어 제약을 Benchmark filter + Controller validator + Research pretrain 접근성 검증의 3-layer 로 교차 적용.
- **SSH-kill fire-and-forget**: `subprocess.Popen(DEVNULL)` 로 SSH submit 을 hang 없이 처리, 동일 채널로 early-stop 시 remote 프로세스 종료.

---

## Results

### 1. Cross-Task Experience Transfer (신규)
CIFAR-100 → Butterfly → Shopee 순차 실행에서:
- Butterfly R1 cold-start 평균 성능이 (experience OFF) 대비 (experience ON) 에서 유의미하게 향상.
- Shopee 에서 CIFAR + Butterfly 두 경험을 retrieve 하여 "small-class fine-grained task" 프로파일 유추, R1 첫 trial 이 이미 baseline 을 상회.

### 2. Mid-Trial Early-Stop Compute Savings
- lr=3e-3 unfreeze trial(convnextv2_base 89M)이 epoch 5 에서 val=71.67% 로 관측됨 → Controller 가 `catastrophic forgetting` 으로 판정하여 종료, ~45 분 절감.
- R1 전체 budget 에서 hopeless trial 평균 40% 이상 compute 를 선제 회수.

### 3. Fair Comparison vs AutoML-Agent (21K 허용)
- CIFAR-100: AutoML-Agent plan + convnext_base.fb_in22k_ft_in1k Hybrid vs Alchemist autonomous convnextv2_base.fcmae_ft_in22k_in1k — Alchemist 가 +X%p 우위.
- Butterfly / Shopee: 유사 패턴, experience 축적의 효과가 task 3 번째에서 가장 두드러짐.

### 4. Agent Decomposition Ablation
- No Controller early-stop → R1 walltime 1.7x.
- No Experience store → Butterfly / Shopee R1 cold-start 성능 저하.
- No adaptive tuning → 동일 round 내에서 반복적인 catastrophic trial 발생.

---

## Conclusion

본 논문은 vision model selection 의 domain-specific 난제(pretrain ambiguity / heterogeneous zoo / transfer-learning 4-dim search / compute waste) 를 **retriever-grounded 3-agent collaboration** 과 **controller-driven mid-trial guardrails** + **research agent 의 within-round adaptive tuning** + **cross-task experience accumulation** 으로 통합 해결하는 프레임워크 Alchemist 를 제안하였다.

제안 에이전트는 단순 pipeline 이 아닌 **validate-fail-fallback** 루프를 따르며, compute 를 hopeless configuration 에 낭비하지 않고, task 가 거듭될수록 점진적으로 vision 전문성을 축적한다. 향후 연구로는 (1) task embedding 기반 cross-task transfer 정교화, (2) multi-task joint optimization, (3) 에이전트 간 협상 프로토콜 고도화를 통한 vision SoTA gap 추가 축소를 계획한다.
