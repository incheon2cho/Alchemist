# Alchemist 3-Agent Harness — 테스트 주도 설계 (TDD)

---

## 1. 테스트 개요

### 모듈 구조

```
alchemist/
├── core/
│   ├── schemas.py        — UserTask, Leaderboard, ResearchResult, ExperimentState
│   ├── llm.py            — LLMClient ABC, MockLLMClient, safe_llm_call
│   └── utils.py          — safe_asdict (enum 직렬화)
├── agents/
│   ├── protocol.py       — AgentMessage, MessageBus, make_directive/response
│   ├── benchmark.py      — BenchmarkAgent (모델 탐색 + 순위화)
│   ├── research.py       — ResearchAgent (HP 탐색 + 오토 리서치)
│   └── controller.py     — ControllerAgent (통제 + Safety + Registry)
├── harness.py            — ThreeAgentHarness (통합 오케스트레이터)
└── main.py               — CLI 엔트리포인트
```

### 테스트 레벨

| 레벨 | 목적 | 비율 |
|------|------|------|
| Unit | 스키마, 프로토콜, 각 Agent 독립 | ~50% |
| Integration | Agent 간 연동, 파이프라인 | ~35% |
| E2E | CLI 실행, 전체 흐름 | ~15% |

---

## 2. 테스트 매트릭스

| 모듈 | 테스트 파일 | 테스트 종류 |
|------|-----------|-----------|
| core/schemas.py | test_agents.py, test_edge_cases.py | Unit |
| core/llm.py | test_edge_cases.py::TestLLMEdgeCases | Unit |
| agents/protocol.py | test_protocol.py | Unit |
| agents/benchmark.py | test_agents.py::TestBenchmarkAgent | Unit |
| agents/research.py | test_agents.py::TestResearchAgent | Unit |
| agents/controller.py | test_agents.py::TestControllerAgent | Unit |
| harness.py | test_harness.py | Integration + E2E |
| (edge cases) | test_edge_cases.py | Unit + Integration |

---

## 3. Unit Tests

### test_protocol.py — Agent 통신 프로토콜

| Test Case | 기대 결과 |
|-----------|----------|
| `test_agent_message_roundtrip` | to_dict → from_dict 왕복 일치 |
| `test_message_bus_logging` | send → get_log에 1건 기록 |
| `test_message_bus_trace` | trace_id로 필터링 정상 |
| `test_message_bus_disk_logging` | JSONL 파일 기록 확인 |
| `test_make_helpers` | directive/response/escalation 헬퍼 정상 |

### test_agents.py — 개별 Agent

**Benchmark Agent:**

| Test Case | 입력 | 기대 결과 |
|-----------|------|----------|
| `test_handle_directive_returns_leaderboard` | directive | leaderboard payload 포함 |
| `test_scout_models_returns_known` | "vision encoder" | dinov2, vit_s16 등 포함 |
| `test_build_leaderboard_ranks_correctly` | 3 models, 2 benchmarks | 점수 높은 모델이 상위 rank |
| `test_recommend_respects_constraints` | max_params_m=30 | 86M 모델 제외, 22M 추천 |
| `test_recommend_no_constraint_picks_best` | 제약 없음 | overall_rank 1위 추천 |

**Research Agent:**

| Test Case | 입력 | 기대 결과 |
|-----------|------|----------|
| `test_handle_directive_returns_result` | base_model + task | best_score > 0, trials > 0 |
| `test_baseline_score_exists` | "dinov2_vitb14" | 60 ≤ score ≤ 85 |
| `test_design_experiment_returns_configs` | max_trials=6 | len ≤ 6, 각 config에 lr 존재 |
| `test_trials_respect_budget` | budget=0.01 hr | trials < 100 |

**Controller Agent:**

| Test Case | 입력 | 기대 결과 |
|-----------|------|----------|
| `test_validate_recommendation_ok` | 22M model, limit=30M | ok=True |
| `test_validate_recommendation_exceeds_params` | 86M model, limit=30M | ok=False |
| `test_judge_result_improved` | best=85, baseline=70 | ok=True |
| `test_judge_result_no_improvement` | best=68, baseline=70 | ok=False |
| `test_safety_budget_exhausted` | used=100/100 | budget_exhausted=True |
| `test_safety_budget_ok` | used=10/100 | len(issues)=0 |

---

## 4. Integration Tests

### test_harness.py — 통합 파이프라인

| Test Case | 시나리오 | 기대 결과 |
|-----------|---------|----------|
| `test_run_produces_result` | full pipeline | best_score > 0, base_model 존재, trials > 0 |
| `test_run_uses_recommended_model` | auto 추천 | 7개 known models 중 하나 |
| `test_run_with_explicit_base_model` | base_model="vit_s16_dino" | base_model 일치 |
| `test_run_with_constraints` | max_params_m=25 | 86M 모델 미선택 |
| `test_benchmark_returns_leaderboard` | benchmark only | entries ≥ 5, sorted |
| `test_benchmark_with_task_gives_recommendation` | task 포함 | recommendation != "" |
| `test_leaderboard_has_scores` | 2 benchmarks | 각 entry에 scores ≥ 2 |
| `test_research_returns_result` | research only | best_score > baseline |
| `test_research_improves_over_baseline` | 8 trials | improvement > 0 |

### test_harness.py — 감사 로그

| Test Case | 기대 결과 |
|-----------|----------|
| `test_audit_log_populated` | ≥ 4 메시지 |
| `test_audit_log_has_correct_flow` | 첫 메시지: controller → benchmark |
| `test_disk_logging` | JSONL 파일 존재 |

---

## 5. Edge Cases

### test_edge_cases.py

| Test Case | 입력 | 기대 결과 |
|-----------|------|----------|
| `test_llm_returns_invalid_json` | "{{{" | fallback 반환 |
| `test_llm_raises_exception` | ConnectionError | fallback 반환 |
| `test_mock_llm_call_count` | 2회 호출 | count=2 |
| `test_empty_benchmarks_list` | benchmarks=[] | 에러 없음 |
| `test_leaderboard_single_model` | 1 model | rank=1 |
| `test_recommend_empty_leaderboard` | entries=[] | recommendation="" |
| `test_zero_budget_stops_early` | budget=0.001 | trials < max |
| `test_baseline_unknown_model` | "unknown_xyz" | 50~70 범위 |
| `test_validate_missing_recommendation` | recommendation="" | ok=False |
| `test_validate_recommendation_not_in_lb` | "B" not in entries | ok=False |
| `test_judge_exact_equal` | best=baseline=70 | ok=False |
| `test_judge_with_min_improvement` | +2% < min=5% | ok=False |
| `test_zero_budget_halts` | budget=0 | report="Budget exhausted" |
| `test_deterministic_with_seed` | seed=12345 × 2 | 동일 결과 |
| `test_state_updated_after_run` | full run | phase=completed, leaderboard 존재 |
| `test_audit_log_json_serializable` | full run | json.dumps 에러 없음 |
| `test_message_roundtrip` | AgentMessage | to_dict → from_dict 일치 |
| `test_experiment_state_budget` | 100-75 | remaining=25 |
| `test_experiment_state_overspent` | 100-150 | remaining=0 |
| `test_user_task_defaults` | UserTask() | eval_metric="top1_accuracy" |

---

## 6. 검증된 버그 수정 이력

| # | 버그 | 테스트 | 수정 |
|---|------|--------|------|
| 1 | `asdict()`가 Enum을 문자열로 변환 안 함 | JSON serialization test | `safe_asdict()` 유틸 도입 |
| 2 | 빈 metrics에서 규칙 엔진이 0과 target 비교 | test_decide_with_empty_metrics | `"key" in metrics` 존재 확인 |
| 3 | Research Agent OOM 시 crash | test_zero_budget_stops_early | OOM catch + graceful skip |
| 4 | DINOv2 img_size 518 강제 | 실측 벤치마크 | `img_size=224` 문서화 |

---

## 7. 현재 테스트 현황

| 파일 | 항목 수 | 커버리지 |
|------|--------|---------|
| test_protocol.py | 5 | 프로토콜 100% |
| test_agents.py | 15 | 3 Agent 단위 |
| test_harness.py | 12 | 통합 파이프라인 |
| test_edge_cases.py | 22 | 엣지 케이스 |
| **합계** | **54** | **ALL PASS** |

---

## 8. 테스트 실행

```bash
cd alchemist
python -m pytest tests/ -v          # 전체 테스트
python -m pytest tests/ -q          # 요약만
python -m pytest tests/test_agents.py -v    # Agent 단위만
python -m pytest tests/test_harness.py -v   # 통합만
```

---

## 관련 문서

- [PDR.md](PDR.md) — 아키텍처 설계
- Obsidian 상세: `Obsidian/Alchemist/TDD/M1~M6-*.md`
