# Alchemist — Figure 1 Captions

Candidate captions for the paper's Figure 1 (Alchemist framework with
agent-level modules and vision-specific priors). Three length variants
in both English and Korean are provided; pick the one that matches the
target venue's style and page budget.

Legend used in the figure and referenced by every caption:
- **★ / ⭐** marks this paper's primary contributions:
  (1) Controller's vision-aware mid-trial early-stop (`evaluate_trial_progress`)
  (2) Research Agent's failure-mode-adaptive tuning (`_adapt_config_from_failures`)
  (3) Cross-task `VisionExperienceStore` (persistent JSONL)

---

## English

### A. Concise — conference paper (~60 words)

> **Figure 1. Alchemist architecture with vision-specific modules.** Three agents collaborate under a Controller-centric six-verb orchestration (*Formalize → Invoke → Scrutinize → Assay → Instantiate → Surveil → Adjudicate*). The Benchmark Agent grounds model selection in three vision-specific sources (timm ImageNet top-1 CSV, Papers-with-Code task leaderboards, arXiv); the Research Agent explores the transfer-learning primitive space with a modern vision training recipe (Mixup/CutMix/EMA/LLRD) as prior. Starred modules (★) denote our primary contributions: Controller's vision-aware mid-trial early-stop (`evaluate_trial_progress`), Research Agent's failure-mode-adaptive tuning (`_adapt_config_from_failures`), and its cross-task `VisionExperienceStore`.

### B. Standard — journal paper (~120 words)

> **Figure 1. Alchemist framework: agent-level modules and vision-specific priors.** User queries enter through the Controller Agent (slate), which enforces a six-stage protocol (*Formalize, Invoke, Scrutinize, Assay, Instantiate, Surveil, Adjudicate*) against the Benchmark Agent (indigo, left) and Research Agent (teal, right). The Benchmark Agent performs retrieval-grounded model scouting across three vision corpora — timm's ImageNet top-1 CSV (1266 models), Papers-with-Code task leaderboards, and arXiv — with a six-tier pretrain-source ontology (IN-1K, IN-21K, JFT, LAION, LVD-142M, CLIP) that filters constraint-violating candidates. The Research Agent searches the vision transfer-learning primitive space (*freeze × adapter × LLRD × modern-recipe-on/off*) and, upon completion, records winning configurations to a persistent cross-task memory. **Starred modules (⭐) highlight our primary contributions**: the Controller's vision-aware mid-trial early-stop with four heuristics (catastrophic forgetting, hopeless-vs-best, optimizer divergence, plateau), the Research Agent's failure-mode-aware adaptive tuning (OOM-forgiveness rule after consecutive successes), and its `VisionExperienceStore` that turns the agent into a progressively-more-expert vision researcher across tasks.

### C. Detailed — journal / thesis (~200 words, module-level)

> **Figure 1. Alchemist: multi-agent framework for vision model selection with empirical guardrails and accumulating expertise.** The framework comprises three specialized agents coordinated by a Controller-centric six-verb orchestration. **Benchmark Agent** (indigo) fuses evidence from three vision-native retrievers — *HFHubRetriever* (timm ImageNet top-1 CSV, 1266 models), *PwC Scout* (task-specific leaderboards), and *ArxivRetriever* (2023–25 papers) — and classifies each candidate's pretrain corpus into a six-tier ontology (ImageNet-1K/21K, JFT, LAION, LVD-142M, CLIP) to filter constraint violations; candidates are ranked HF-first, PwC-second to ensure timm-loadability. **Controller Agent** (slate) executes the pipeline through six stages: it *Formalizes* the task into typed constraints, *Invokes* the Benchmark Agent, *Scrutinizes* the returned recommendation symbolically, *Assays* the top-K candidates via empirical baseline evaluation on AWS GPUs to select a winner, *Instantiates* the Research Agent with the winner, *Surveils* each trial through per-epoch progress (★ vision-aware four-rule early-stop), and finally *Adjudicates* the output for shipping. **Research Agent** (teal) iterates over the transfer-learning primitive space (freeze × adapter × LLRD × advanced-tech) with Mixup, CutMix, RandAugment, EMA, and LLRD activated as modern-recipe priors, learns from failures via ★ *adaptive tuning* (catastrophic→lr cap, divergence→augmentation↓, OOM→batch÷2 with consecutive-success forgiveness), and persists winning configurations to a ★ *VisionExperienceStore* (JSONL), enabling cross-task expertise accumulation across subsequent tasks. Stars (★) mark this paper's primary contributions.

---

## 한국어 (Korean)

### A. 간결형 — 학회 논문 (~60 단어)

> **그림 1. Alchemist 프레임워크의 비전 특화 모듈 구성.** 세 에이전트가 Controller 중심의 6-단계 프로토콜(*Formalize → Invoke → Scrutinize → Assay → Instantiate → Surveil → Adjudicate*)에 따라 협업한다. Benchmark Agent는 비전 특화 세 소스(timm ImageNet top-1 CSV, Papers-with-Code 태스크 리더보드, arXiv)에 기반해 모델을 탐색하고, Research Agent는 전이학습 프리미티브 공간을 현대 비전 학습 레시피(Mixup/CutMix/EMA/LLRD)를 기본 prior로 삼아 탐색한다. 별표(★) 모듈은 본 논문의 핵심 기여로, Controller의 비전 인지형 mid-trial early-stop (`evaluate_trial_progress`), Research Agent의 실패 모드 기반 적응형 튜닝 (`_adapt_config_from_failures`), 그리고 태스크 간 경험을 누적하는 `VisionExperienceStore`를 나타낸다.

### B. 표준형 — 저널 논문 (~120 단어)

> **그림 1. Alchemist 프레임워크: 에이전트별 세부 모듈과 비전 도메인 특화 prior.** 사용자 쿼리는 중앙의 Controller Agent(슬레이트)로 진입하여 Benchmark Agent(인디고, 좌측)와 Research Agent(틸, 우측)를 상대로 6-단계 프로토콜(*Formalize, Invoke, Scrutinize, Assay, Instantiate, Surveil, Adjudicate*)을 수행한다. Benchmark Agent는 세 개의 비전 corpus — timm 공식 ImageNet top-1 CSV(1,266 모델), Papers-with-Code 태스크 리더보드, arXiv — 를 retrieval-grounded 방식으로 종합하며, 6-tier 사전학습 소스 ontology(IN-1K / IN-21K / JFT / LAION / LVD-142M / CLIP)로 제약 위반 후보를 자동 필터링한다. Research Agent는 비전 전이학습의 프리미티브 공간(*freeze × adapter × LLRD × advanced-tech*)을 탐색하고, 완료 시 winning configuration을 영구 메모리에 기록한다. **별표(⭐) 모듈은 본 논문의 핵심 기여**를 나타낸다: Controller의 비전 인지형 mid-trial early-stop(catastrophic forgetting / hopeless-vs-best / optimizer divergence / plateau의 4-rule 휴리스틱), Research Agent의 실패 모드 기반 적응형 config 튜닝(연속 성공 후 OOM 제약 해제 포함), 그리고 태스크 간 경험 축적을 가능케 하는 `VisionExperienceStore`이다. 이를 통해 Alchemist는 **태스크가 반복될수록 점진적으로 비전 전문성을 누적**하는 학습형 에이전트로 동작한다.

### C. 상세형 — 학위논문·저널 (~200 단어, 세부 모듈 언급)

> **그림 1. Alchemist: 실측 기반 가드레일과 경험 누적 기반의 비전 모델 선택을 위한 다중 에이전트 프레임워크.** 본 프레임워크는 Controller를 중심으로 3개의 특화된 에이전트가 6-단계 프로토콜로 협업한다. **Benchmark Agent**(인디고)는 3개의 비전 특화 retriever — *HFHubRetriever*(timm ImageNet top-1 CSV, 1,266 모델), *PwC Scout*(태스크별 리더보드), *ArxivRetriever*(2023–25 논문) — 로부터 증거를 결합하며, 각 후보 모델의 사전학습 corpus를 6-tier ontology(ImageNet-1K / 21K, JFT, LAION, LVD-142M, CLIP)로 분류하여 제약 위반 모델을 자동 제외한다. 후보는 timm-loadability 보장을 위해 **HF-first, PwC-second** 순위로 정렬된다. **Controller Agent**(슬레이트)는 파이프라인을 6단계로 실행한다: 태스크를 typed 제약으로 *Formalize*, Benchmark에 *Invoke*, 추천 결과를 symbolic하게 *Scrutinize*, top-K 후보의 **실제 baseline을 AWS GPU에서 *Assay***하여 Winner 선정, Research Agent를 *Instantiate*, 각 trial을 epoch 단위로 ★ *Surveil*(비전 특화 4-rule early-stop), 최종 결과를 *Adjudicate*로 출시 판정한다. **Research Agent**(틸)는 비전 전이학습 프리미티브 공간(freeze × adapter × LLRD × advanced-tech)을 탐색하며, Mixup·CutMix·RandAugment·EMA·LLRD가 modern recipe prior로 기본 활성화된다. 실패로부터 ★ *adaptive tuning*(catastrophic → lr 상한, divergence → augmentation α 감쇠, OOM → batch 절반화 with 연속 성공 시 용서 규칙)으로 학습하며, 완료된 태스크의 winning configuration을 ★ *VisionExperienceStore*(JSONL persistent)에 기록하여 후속 태스크에서 유사 경험을 retrieve한다. 별표(★)는 본 논문의 핵심 기여를 표시한다.

---

## Usage guide

| Venue | Recommended |
|---|---|
| ICLR / NeurIPS / CVPR (2-column) | **A (concise)** |
| AAAI / IJCAI / TPAMI (journal) | **B (standard)** ⭐ |
| PhD thesis / TOG / long-form | **C (detailed)** |

Bilingual (KR + EN) caption recommended for domestic venues (e.g.,
KIISE, KIISS, KSIC). International submissions should use English only
unless the venue specifies otherwise.

Color references (indigo / slate / teal / amber) are preserved in all
captions so the figure remains interpretable in grayscale print.
