# A Multi-Agent Collaboration Framework for Vision Model Selection and Structural Self-Refinement

> 논문 초안 구성 (Outline)

---

## Introduction

### 1. 배경
대규모 비전 모델의 빠른 발전으로 인해 수백 개의 사전학습 모델(ViT, Swin, ConvNeXt, Mamba 등)이 공개되고 있으나, 특정 태스크에 최적인 모델과 학습 전략을 찾는 과정은 여전히 연구자의 수동 실험에 의존하고 있다.
기존 AutoML/NAS 연구는 아키텍처 탐색 또는 하이퍼파라미터 최적화 중 하나에 집중하며, 모델 선정부터 학습 전략 설계, 결과 분석, 자율 개선까지의 전 과정을 통합적으로 자동화하는 프레임워크는 부재하다.
최근 LLM의 추론 능력이 향상됨에 따라, 이를 연구 자동화의 의사결정 엔진으로 활용할 가능성이 대두되고 있다.

### 2. 문제정의
비전 모델 최적화는 (1) 다양한 아키텍처 중 태스크에 적합한 모델 선정, (2) 하이퍼파라미터 및 학습 전략 탐색, (3) 실험 결과 분석 및 개선 방향 도출이 순환적으로 이루어져야 하나, 현재는 각 단계가 분절되어 있어 비효율적이다.
또한 연구자가 최신 SoTA 기법을 파악하고 실험에 반영하는 데 상당한 시간이 소요되며, 실험 이력의 체계적 관리와 재현이 어렵다.
본 논문은 이러한 문제를 해결하기 위해 모델 탐색-최적화-자율 개선의 전 과정을 자동화하는 다중 에이전트 협업 프레임워크를 제안한다.

### 3. 제안기술 요약
본 논문은 Benchmark Agent(모델 탐색·순위화), Research Agent(자율 실험·분석·개선), Controller Agent(안전·품질 관리)로 구성된 3-Agent 협업 하네스를 제안한다.
Research Agent는 LLM을 활용하여 외부 SoTA 지식을 탐색하고, 현재 결과와의 gap을 분석하여 부족한 기법(SAM optimizer 등)을 자율적으로 다음 실험에 도입하는 self-refinement 루프를 수행한다.
제안 프레임워크는 CNN, Transformer, Mamba 등 이종 아키텍처를 단일 파이프라인에서 탐색하며, AWS GPU와 연동하여 실제 학습을 수행하고 결과를 대시보드에 자동 배포한다.

### 4. Contribution
- LLM 기반 3-Agent 협업 하네스를 통해 비전 모델의 선정·탐색·최적화·자율 개선을 end-to-end로 자동화하는 프레임워크를 최초로 제안한다.
- Research Agent에 SoTA 지식 탐색과 gap 분석 기반 self-refinement 메커니즘을 도입하여, 에이전트가 스스로 부족한 기법을 식별하고 실험에 반영하는 자율 개선 능력을 구현하였다.
- CIFAR-100에서 5개 계열 18개 backbone을 자동 탐색하여 94.00%(Swin-Base + SAM)를 달성하였으며, 이는 수동 실험 대비 탐색 시간을 대폭 단축하면서 SoTA 대비 2.08%p 이내의 성능을 확보한 결과이다.

---

## Method

### 1. Multi-Agent 자동 탐색 및 자율 학습 하네스
제안 프레임워크는 3개의 전문화된 에이전트가 구조화된 메시지 프로토콜을 통해 협업하며, Benchmark → Research → Controller의 파이프라인으로 동작한다.
각 에이전트는 LLM(Claude CLI/Codex CLI)을 의사결정 엔진으로 활용하며, 학습 실행은 AWS EC2 GPU에 위임하여 에이전트 로직과 학습 인프라를 분리한다.
전체 실험 과정은 ResearchLog에 구조화된 JSON으로 기록되어 재현성과 감사 추적이 가능하다.

### 2-1. Benchmark Agent
Benchmark Agent는 timm, HuggingFace 등에서 후보 모델을 탐색하고, 다중 벤치마크(linear probe, kNN)에서 성능을 측정하여 종합 순위표(Leaderboard)를 생성한다.
LLM을 활용하여 사용자 태스크의 제약조건(파라미터 수, 지연시간 등)에 맞는 최적 모델을 추천하며, 추천 근거를 자연어로 생성하여 Controller와 Research Agent에 전달한다.
본 연구에서는 CNN 7종, Transformer 5종, Mamba 계열 3종 등 총 18개 backbone을 단일 파이프라인에서 탐색하였다.

### 2-2. Research Agent
Research Agent는 추천 모델을 기반으로 실험 설계 → 학습 실행 → 자체 분석 → 계속/중단 판단의 자율 반복 루프를 수행한다.
핵심 기능으로 외부 SoTA 지식 탐색(`search_sota`), 현재 결과 vs SoTA gap 분석(`analyze_sota_gap`), 부족한 기법 자동 제안(`suggest_techniques`)을 포함하며, 이를 통해 SAM optimizer, 더 긴 학습, stochastic depth 등의 기법을 에이전트가 자율적으로 도입한다.
NAS 전략으로 Phase 1(다중 backbone 탐색) → Phase 2(Top-K HP 최적화) → Phase 3(최종 long training)의 3단계 점진적 탐색을 수행한다.

### 2-3. Controller Agent
Controller Agent는 파이프라인 오케스트레이션, 예산(GPU 시간) 관리, 안전 가드(성능 하락 감지, OOM 핸들링)를 담당한다.
Benchmark Agent의 모델 추천을 LLM 기반으로 검증하고, Research Agent의 최종 결과에 대해 품질 기준(baseline 대비 개선 여부)에 따른 Ship 판정을 수행한다.
에이전트 간 축적된 컨텍스트(벤치마크 결과, 추천 근거, 분석 이력)를 Research Agent에 전달하여 정보 흐름의 연속성을 보장한다.

---

## Results

### 1. 모델 탐색 효율성
18개 이종 backbone(CNN/ViT/Swin/Mamba)을 약 8시간의 자동 탐색으로 전수 평가하였으며, 수동 실험 대비 탐색 시간과 인적 비용을 대폭 절감하였다.
3단계 NAS 전략을 통해 Phase 1에서 ConvNeXt-Tiny(89.57%)를 조기 발견하고, Phase 2에서 40ep HP 최적화로 90.87%까지 향상시키는 점진적 성능 개선을 자동 달성하였다.
전체 탐색·최적화 과정이 약 $25(EC2 ~25시간)의 비용으로 완료되어, 제한된 연구 예산에서도 대규모 모델 탐색이 가능함을 입증하였다.

### 2. 자율 개선 효과 (Self-Refinement)
Research Agent의 SoTA gap 분석이 SAM optimizer를 자동 제안하였으며, 이를 적용한 결과 기존 best(93.82%) → 94.00%로 +0.18%p 기록 갱신을 달성하였다.
이는 에이전트가 외부 지식을 탐색하고 부족한 기법을 스스로 식별·도입하는 self-refinement 메커니즘의 실효성을 실증한 결과이다.
SoTA(96.08%, JFT-300M pretrained) 대비 gap이 2.26%p → 2.08%p로 축소되었으며, 추가 실험(rho 변형, longer training)을 통한 개선 여지가 남아 있다.

### 3. SoTA 대비 포지셔닝
동일 CIFAR-100 벤치마크에서 ViT-B/16 논문 결과(91.48%)를 본 프레임워크(93.46%)가 +1.98%p 상회하여, 자동화된 학습 전략 설계의 우수성을 확인하였다.
ImageNet-1K pretrained 모델 기준으로 Vision Mamba(91.13%)가 유사 크기 모델(DeiT-Small 89.59%) 대비 우수한 전이 성능을 보여, 에이전트의 다양한 아키텍처 탐색이 새로운 발견으로 이어질 수 있음을 시사한다.
Linear Probe 평가에서도 Swin-Base(85.17%)가 ViT-Base(80.37%)를 상회하여, fine-tuning 성능과 representation quality 간의 상관관계 및 pretrained 데이터 규모의 영향을 체계적으로 분석하였다.

---

## Conclusion

본 논문에서는 LLM 기반 3-Agent 협업 하네스를 통해 비전 모델의 선정·탐색·최적화·자율 개선을 end-to-end로 자동화하는 프레임워크를 제안하였다.
Research Agent의 SoTA 지식 탐색과 gap 분석 기반 self-refinement를 통해, 에이전트가 SAM optimizer와 같은 최신 기법을 스스로 도입하여 기존 성능을 갱신(93.82% → 94.00%)하는 자율 개선 능력을 실증하였다.
향후 연구로는 더 큰 pretrained 모델(JFT-300M) 활용, 다중 GPU 분산 학습 지원, 그리고 에이전트 간 협상 메커니즘 고도화를 통해 SoTA와의 gap을 추가로 축소할 계획이다.
