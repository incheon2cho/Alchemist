# AutoML-Agent vs Alchemist — 차이 분석

> 논문 Related Work 및 Positioning 작성용 비교 자료
>
> 비교 대상: **AutoML-Agent** (Trirat et al., ICML 2025, [arXiv:2410.02958](https://arxiv.org/abs/2410.02958), [code](https://github.com/DeepAuto-AI/automl-agent))

---

## 1. 한눈에 보는 비교 (Executive Summary)

| 축 | **AutoML-Agent** (ICML 2025) | **Alchemist** (본 프로젝트) |
|---|---|---|
| 범위 | 7 modalities × 14 datasets (범용) | **Vision 특화** (심화) |
| 에이전트 수 | 5 (`data`, `model`, `operation`, `prompt`, `manager`) | 3 (Benchmark, Research, Controller) |
| LLM 백엔드 | GPT-4 / Mixtral API | **Claude CLI + Codex CLI (로컬)** |
| 인터페이스 | Python API (Jupyter 권장) | Shell + AWS EC2 오케스트레이션 |
| 핵심 독창성 | Retrieval-augmented **planning** + 다단계 검증 | **SoTA-gap 분석 기반 self-refinement 루프** |
| 학습 실행 | 로컬 · 같은 프로세스 | **AWS EC2 분리 오케스트레이션** |
| 출시 상태 | 논문 발표 + 코드 공개 | 프로토타입, 논문 준비 중 |

---

## 2. 에이전트 아키텍처 상세 비교

### 2.1 AutoML-Agent 구조 (5-agent)

```
                   [agent_manager]
                         │
          ┌──────┬───────┼───────┬─────────┐
          ▼      ▼       ▼       ▼         ▼
  [data_agent][model_agent][operation_agent][prompt_agent][prompt_pool]
```

역할 분담:
- `data_agent`: 데이터 수집 및 전처리
- `model_agent`: 모델 선정 및 HP 설정
- `operation_agent`: 파이프라인 실행
- `prompt_agent`: LLM 프롬프트 생성·정제
- `agent_manager`: 전체 오케스트레이션

### 2.2 Alchemist 구조 (3-agent)

```
  [Controller Agent]
         │ (오케스트레이션 · 예산 · 안전)
         ├─── [Benchmark Agent] ── 모델 탐색·순위
         │
         └─── [Research Agent] ── SoTA-gap 분석
                                  + 기법 제안 (SAM 등)
                                  + Self-refinement loop
```

### 2.3 핵심 차이

| 축 | AutoML-Agent | Alchemist |
|---|---|---|
| 설계 철학 | **기능별 분업** (data/model/op/prompt) | **역할별 협업** (탐색/개선/통제) |
| 상위 오케스트레이터 | `agent_manager` (단일) | `Controller` (안전·품질 게이트 포함) |
| Prompt 전담 에이전트 | ✅ 별도 존재 (`prompt_agent`) | ❌ 각 에이전트가 자체 처리 |
| 자기개선 루프 | 다단계 검증 중심 (verify → refine) | **외부 SoTA 지식 → gap → 기법 자동 도입** |

---

## 3. 탐색 공간 & 도메인

### 3.1 AutoML-Agent
- **범용 ML** 전체 파이프라인 자동화
- 지원 도메인: Image, Text, Tabular, Graph, Time Series (7 modalities)
- 14+ 데이터셋: Butterfly, Shopee-IET, e-commerce text, Banana Quality, Cora, Citeseer, Weather 등
- "Full-pipeline AutoML" — 데이터 수집부터 배포까지

### 3.2 Alchemist
- **Vision 도메인 집중** (이미지 분류·검출)
- Backbone 탐색: **CNN 7종 + Transformer 5종 + Mamba 3종 + DINO + I-JEPA** (총 18+)
- 태스크: CIFAR-100 심층 실증, COCO detection(부분), ImageNet(시도)
- 아키텍처 다양성이 **이종(heterogeneous)** — ViT · Swin · MambaOut · Vision Mamba 동일 파이프라인

### 3.3 핵심 차이

| 축 | AutoML-Agent | Alchemist |
|---|---|---|
| **폭 vs 깊이** | 폭 넓음 (범용) | **깊이 (Vision 특화)** |
| Mamba/JEPA 등 최신 vision 모델 | ❌ 언급 없음 | ✅ 내장 |
| Tabular/Text/Graph | ✅ 지원 | ❌ 범위 밖 |

---

## 4. Self-Refinement 메커니즘

### 4.1 AutoML-Agent — "계획 단계 retrieval + 실행 후 verification"

```
사용자 task → [retrieval-augmented planning]
            ↓ (관련 논문/예제 검색해서 plan 생성)
            → sub-task 분해 → 병렬 실행
            ↓
            [multi-stage verification]
            ↓ (코드 에러 / 성능 기준 검증)
            → 재생성 또는 종료
```

특징: 검증이 **개별 step 단위**로 작동. SoTA와의 거리 관점은 아님.

### 4.2 Alchemist — "실행 후 SoTA-gap 분석 → 기법 도입"

```python
# Research Agent 내부 로직 (alchemist/agents/research.py)
def self_refine_loop():
    result   = run_experiment()
    sota     = search_sota(task)             # 외부 SoTA 지식 탐색
    gap      = analyze_sota_gap(result, sota)
    if gap > threshold:
        techniques = suggest_techniques(gap)  # SAM, longer train, etc.
        propose_next_experiment(techniques)
```

특징: **결과와 외부 SoTA의 거리**가 루프의 트리거. "부족한 기법"을 동적으로 도입.

### 4.3 핵심 차이

| 축 | AutoML-Agent | Alchemist |
|---|---|---|
| 개선 트리거 | 실행 에러 / 단계별 검증 실패 | **외부 SoTA 대비 성능 gap** |
| 도입 지식 | RAG로 plan 생성 초기에 | **결과 분석 후 부족 기법 동적 도입** |
| 예시 | 데이터 전처리 코드 정정 | **SAM optimizer 자동 도입 → +0.18%p** |

→ Alchemist의 self-refinement는 **"meta-learning on search process"** 색채이며,
   AutoML-Agent는 **"robust execution with verification"** 색채.

---

## 5. LLM 활용 방식

| 축 | AutoML-Agent | Alchemist |
|---|---|---|
| 주 LLM | GPT-4, GPT-3.5, Mixtral-8x7B | **Claude CLI + Codex CLI** |
| 접근 방식 | OpenAI API 또는 vLLM 서빙 | **Subprocess로 로컬 CLI 호출** (`claude -p`, `codex exec`) |
| API 비용 | 있음 (유료) | **실질 $0** (CLI 구독) |
| 로컬 실행 난이도 | Mixtral vLLM 필요 (GPU 인프라) | CLI 사용으로 간편 |
| 프롬프트 관리 | `prompt_agent` + `prompt_pool` 체계화 | 에이전트 내부 직접 관리 |

→ Alchemist의 **CLI 기반 접근은 산업/연구 배포에서 실용적 장점** — API 비용·vLLM 인프라 없이 개인/소규모 팀 사용 가능.

---

## 6. 실험 인프라

### 6.1 AutoML-Agent
- 학습 실행도 Python 프로세스 내부 (Jupyter 중심)
- GPU는 로컬/서버에 있어야 함
- 작은~중간 규모 태스크 중심

### 6.2 Alchemist
- **에이전트 로직(로컬) ↔ 학습 실행(AWS EC2) 분리**
- Executor 추상화: `LocalExecutor` + `AWSExecutor` (SSH 기반)
- Spot 인스턴스, checkpoint auto-sync, S3 결과 보관 등 **production-grade infra**
- `ResearchLog`로 모든 실험 JSON 감사 기록

**의미**: Alchemist는 **대규모 · 장시간 실험**에 적합한 설계. 반면 AutoML-Agent는 **쉬운 재현 실행**(Jupyter) 우선.

---

## 7. 실증 결과 비교

### 7.1 AutoML-Agent
- 14 datasets × 7 modalities **성공률** 중심 평가
- 논문이 accuracy 단일 수치보다 "**얼마나 많은 태스크를 완주하는가**"를 강조
- 구체 수치(Top-1 등)는 논문 표에 존재하지만 태스크별 상이

### 7.2 Alchemist
- **CIFAR-100 심층 결과**:
  - Swin-Base + SAM: **94.00%** (자체 best)
  - ViT-B/16: **93.46%** — 원 논문 91.48% **초과** (+1.98%p)
  - Vision Mamba: 91.13% (동급 DeiT-Small 89.59% 대비 우수)
  - SoTA gap: 96.08% → 2.08%p 차
- COCO: Faster R-CNN 기반 실패 → YOLO 실험에서 부분 성공
- ImageNet-1K: EVA-02-L 90.05% baseline 확인, ensemble 시 89.84%로 초과 실패

### 7.3 핵심 차이

| 축 | AutoML-Agent | Alchemist |
|---|---|---|
| 평가 목표 | 태스크 완주율(범용성) | **Top-1 / SoTA gap** (심층성) |
| 최고 달성 | 14 dataset 분산 성공 | **CIFAR-100 94.00%, 논문 수치 상회** |
| 한계 | Image 태스크 상세 결과 부족 | 대형 데이터셋(ImageNet) 실증 미완 |

---

## 8. Alchemist의 차별화 포인트 (논문 작성 관점)

### 8.1 강점
1. **Vision-specific depth** — Mamba, DINO, I-JEPA, MambaOut 등 **최신 아키텍처 통합 탐색** (AutoML-Agent 미포함)
2. **SoTA-gap-driven self-refinement** — 계획 단계 RAG와 본질적으로 다른 루프. "결과 분석 → 기법 자동 도입" 실증 (SAM → +0.18%p)
3. **Production-grade infrastructure** — AWS EC2 분리, Spot 중단 핸들링, S3 체크포인트, ResearchLog 감사
4. **CLI-based LLM economics** — API 비용 ~$0로 장시간 에이전트 루프 운영
5. **Heterogeneous backbone** — CNN/Transformer/Mamba **한 파이프라인**에서 공정 비교

### 8.2 약점 (리뷰어 지적 가능성)
1. **범위 좁음** — AutoML-Agent가 7 modalities 다룬 반면 Alchemist는 Vision만 → "AutoML" 주장 취약
2. **직접 정량 비교 부재** — AutoML-Agent 코드로 같은 CIFAR-100 돌려본 결과 없음
3. **대형 데이터셋 미완** — ImageNet 실험 실패 기록만
4. **Precedence** — AutoML-Agent가 ICML 2025 선행 발표 → "최초 제안" 주장 약화
5. Agent 수 차이(3 vs 5)는 "더 간결" or "기능 부족" 양쪽으로 해석 가능

---

## 9. 논문에서의 포지셔닝 권장

### 9.1 Title 수정 제안
- 기존: "A Multi-Agent Collaboration Framework for Vision Model Selection and Structural Self-Refinement"
- 권장: **"Vision-Specialized Multi-Agent Framework with SoTA-Gap-Driven Self-Refinement"**
- 이유: "Vision 특화" + "SoTA-gap loop" **두 가지를 title에서 드러내야** AutoML-Agent와 다름을 각인.

### 9.2 Contribution 재정의
- ~~"비전 모델의 선정·탐색·최적화·자율 개선을 end-to-end로 자동화하는 프레임워크를 **최초로** 제안"~~ → AutoML-Agent 선행 사례로 취약
- ✅ "Vision **heterogeneous backbone** (CNN/ViT/Mamba/JEPA) 단일 파이프라인 탐색 framework 제안"
- ✅ "결과 사후 **SoTA-gap 분석** 기반 self-refinement 메커니즘 (SAM auto-injection 실증)"
- ✅ "**Local CLI LLM** (Claude/Codex) 기반 zero-API-cost 에이전트 아키텍처"

### 9.3 Related Work 비교 테이블 (필수)

| 축 | AutoKaggle | **AutoML-Agent** | MLCopilot | **Alchemist** |
|---|---|---|---|---|
| Agents | 6 | 5 | 2 | **3** |
| Domain | Kaggle tab | 7 modalities | General | **Vision 심화** |
| Self-refinement | ❌ | Verify-only | ❌ | **SoTA-gap** |
| LLM cost | API | API/local | API | **Local CLI** |
| Infra scale | Local | Local | Local | **Cloud-native** |

### 9.4 Experiments 권장
- **필수**: AutoML-Agent 공개 코드 clone → 같은 CIFAR-100에서 동일 예산(GPU-h)으로 실행 → Top-1 비교
- **Ablation**:
  - Research Agent 제거 vs 포함
  - SoTA-gap 모듈 제거 vs 포함
  - CLI LLM vs API LLM (Claude vs GPT-4)

---

## 10. 결론

두 프레임워크는 **외형은 유사하지만 설계 철학과 기여 포인트가 다르다**:

- **AutoML-Agent**: "**넓게, 견고하게**" — 범용 AutoML의 실행 안정성 확보
- **Alchemist**: "**깊게, 자율적으로**" — Vision 특화 + SoTA-gap driven 자율 개선

Alchemist가 논문으로 인정받으려면:
1. Title/Contribution을 **Vision + SoTA-gap**으로 좁혀 포지셔닝
2. AutoML-Agent와 **같은 벤치마크에서 직접 비교** (필수)
3. **ImageNet 수준 실증** 완성 (현 약점)
4. SoTA-gap self-refinement를 **정량적으로 ablation** (단일 사례보다 체계적으로)

---

## 부록 A — AutoML-Agent 참고 링크

- Paper (arXiv): https://arxiv.org/abs/2410.02958
- Code (GitHub): https://github.com/DeepAuto-AI/automl-agent
- Project page: https://deepauto-ai.github.io/automl-agent/
- ICML 2025 Poster: https://icml.cc/virtual/2025/poster/44029
- OpenReview: https://openreview.net/forum?id=p1UBWkOvZm

## 부록 B — 추가 경쟁자 (참고)

| 시스템 | arXiv | 코드 |
|---|---|---|
| AutoKaggle | 2410.20424 | 일부 공개 |
| KompeteAI | 2508.10177 | 불명 |
| I-MCTS | 2502.14693 | 불명 |
| MLCopilot | — | 부분 공개 |
