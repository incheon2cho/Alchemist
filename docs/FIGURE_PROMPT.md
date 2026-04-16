# Alchemist Framework — Figure Generation Prompts

Prompts for generating the paper's framework overview figure using
**Nano Banana** (Google Gemini 2.5 Flash Image).

## How to use

Paste a prompt verbatim into Nano Banana and upload no reference image.
Aim for a **landscape 16:9** aspect at paper-column width (~1800×1000 px).
After generation, export as PDF/SVG via vector tracing if crisper text is
needed.

---

## Prompt v1 — Full system overview (recommended)

```
A clean technical diagram for an academic ML paper (IEEE/NeurIPS style),
horizontal landscape layout, white background, thin gray strokes, flat modern
colors with a restrained palette (indigo, teal, amber, slate), sans-serif
labels (similar to Inter or Helvetica). Title at top: "Alchemist: Multi-Agent
Vision Model Selection and Self-Refinement".

Layout: Controller Agent at CENTER, commanding Benchmark (left) and
Research (right). User task enters from top-left, final output exits
bottom-right.

(1) USER TASK (top-left, amber accent):
   - Small clipboard icon labeled "User Task"
   - Chips: "dataset", "classes", "metric", "budget"
   - Arrow down-right into the Controller

(2) CONTROLLER AGENT (center, large slate rounded rectangle):
   - Title: "Controller Agent" (largest box, central authority)
   - Inside, FIVE stacked actions top-to-bottom:
       "1. Direct Benchmark Agent",
       "2. Validate & pick Winner (top-K baseline eval)",
       "3. Direct Research Agent",
       "4. Mid-trial early-stop guardrail (vision-aware)",
       "5. Judge final result (ship / iterate)"
   - Small pill-shaped badge next to action 4 reading
     "val < baseline − 10%p  |  val + 5%p < best  |  train_loss > 3"
   - Left bidirectional arrow to Benchmark Agent
   - Right bidirectional arrow to Research Agent
   - Small dashed arrow down to AWS EC2 GPU icon labeled
     "SSH + train_worker (baseline eval & trials)"
   - A small "LLM (Claude)" cloud icon attached with dotted line

(3) BENCHMARK AGENT (left column, indigo accent):
   - Rounded rectangle labeled "Benchmark Agent"
   - Three incoming arrows from three small database cylinders ABOVE it:
       "HuggingFace Hub (timm top-1)",
       "Papers-with-Code (task SoTA)",
       "arXiv (recent papers)"
   - Inside: "Scout → Score → Rank → Recommend top-K"
   - Bidirectional arrow to Controller labeled "directive / leaderboard"

(4) RESEARCH AGENT (right column, teal accent):
   - Large rounded rectangle labeled "Research Agent"
   - Inside, a circular self-refinement loop with FOUR nodes connected
     by curved arrows:
       R0 "Baseline eval",
       R1 "Grid search (lr × freeze × adapter × advanced-tech)",
       R2 "LLM-guided refinement + SoTA-gap analysis",
       ADAPT "Adaptive tuning from failures"
   - Small side label near ADAPT: "catastrophic→lr cap | divergence→aug↓ | OOM→batch÷2"
   - Loop arrow from R2 back to R1 labeled "continue if gap remains"
   - Bidirectional arrow to Controller labeled "directive / result / progress"
   - A small cylinder icon labeled "Experience Store" below Research Agent,
     connected by a read-write arrow labeled "record / retrieve similar task"
   - Final output arrow exiting bottom-right to a box:
     "Trained checkpoint + report" with a small trophy icon

Global:
- Controller is visually the LARGEST and CENTRAL element
- Benchmark and Research are SUBORDINATE, flanking left and right
- All arrows show Controller initiating (solid) and receiving (dashed)
- A horizontal dashed band between Controller and Research shows the
  "per-epoch progress.json" stream feeding Controller's early-stop rule
- Experience Store cylinder below Research, with persistent-memory icon
- Tiny footer: "ImageNet-1K / 21K allowed · JFT, LAION, LVD-142M blocked ·
  Cross-task memory accumulates as tasks complete"
- No 3D, no drop shadows, crisp labels, IEEE-paper-friendly
```

---

## Prompt v2 — Compact single-panel (alternative)

Use when the full 4-lane layout is too busy.

```
A clean flat-style technical diagram for an academic paper, landscape 16:9,
white background, thin gray strokes, limited palette (indigo, teal, amber),
sans-serif labels. Title: "Alchemist Framework".

Three stacked horizontal bands:

TOP BAND — "Evidence gathering":
   Three small database icons side by side labeled
   "HuggingFace Hub", "Papers-with-Code", "arXiv".
   Arrows merge downward into a single box.

MIDDLE BAND — "Benchmark → Controller → Research" pipeline:
   Three connected rounded rectangles in a row:
     [Benchmark Agent] → [Controller Agent + Harness] → [Research Agent]
   Under the Benchmark box: small text "Top-K compliant candidates".
   Under the Controller box: small text "Baseline eval on AWS EC2, pick
   Winner" and a tiny AWS GPU icon.
   Under the Research box: small text "R1 grid → R2 LLM refinement"
   with a circular arrow icon.

BOTTOM BAND — "Output":
   A single rounded rectangle labeled "Trained Model + Results" with a
   small chart icon on the side.

No 3D effects, no drop shadows, crisp text, IEEE-paper-friendly style.
```

---

## Prompt v3 — Comparison figure (supporting, Alchemist vs AutoML-Agent)

For the "Related Work" / "Comparison" figure.

```
Clean side-by-side comparison diagram, academic style, landscape 16:9,
white background, sans-serif labels, two color-coded lanes.

LEFT LANE (salmon accent) titled "AutoML-Agent (ICML '25)":
   Vertical stack of boxes from top to bottom:
     "User task" → "Planning (LLM plan)" → "Code generation (LLM writes
     training script)" → "Execute on GPU" → "Output"
   Small red cross icon and label beside "Code generation" saying
   "error-correction iteration only (n_attempts=5, no performance feedback)"

RIGHT LANE (indigo accent) titled "Alchemist (ours)":
   Vertical stack:
     "User task" → "Benchmark Agent (HF + PwC + arXiv)" → "Controller +
     Harness (top-K baseline eval, pick winner)" → "Research Agent (R1
     grid → R2 LLM refinement, SoTA-gap analysis)" → "Output"
   Small green check icon beside Research Agent saying
   "performance-driven self-refinement"

Thin horizontal arrow at bottom connecting both lanes to shared final box:
   "Benchmark comparison: CIFAR-100 / Butterfly / Shopee-IET"

No 3D, crisp text labels, IEEE-paper-friendly style.
```

---

## Prompt v4 — Two-Level Memory Figure (supporting claim)

Use this as a Figure 3 illustrating Alchemist's distinguishing memory
architecture: within-round adaptive tuning + cross-task experience.

```
A minimal, clean diagram for an academic ML paper, landscape 16:9, white
background, thin gray strokes, restrained palette (teal + amber), sans-serif
labels. Title: "Alchemist Research Agent: Two-Level Memory".

LEFT PANEL — "Within-Round Adaptive Tuning" (teal):
   Circular loop with three nodes:
     "Trial N" → "Outcome classifier" → "_adapt_config_from_failures()"
     → back to "Trial N+1"
   Three small branch labels from the classifier:
     "catastrophic (val < baseline−10%p) → cap lr",
     "divergence (train_loss > 3) → mixup α ↓ + lr ↓",
     "OOM → batch ÷ 2"
   Caption: "Agent avoids repeating the same failure within a round"

RIGHT PANEL — "Cross-Task Experience Memory" (amber):
   Horizontal pipeline of three tasks:
     [CIFAR-100] → [Butterfly] → [Shopee-IET]
   Below each task, a small cylinder icon connects upward:
     Cylinder labeled "Experience Store
                       (~/.cache/alchemist/experience.jsonl)"
   After CIFAR completes: arrow from CIFAR → cylinder labeled "record"
   Before Butterfly starts: arrow from cylinder → Butterfly labeled
     "retrieve similar (num_classes bucket + keyword Jaccard)"
   Small side label: "CIFAR winning config (lr, mixup, ema) → Butterfly LLM
   prompt as prior"
   Caption: "Agent becomes a progressively more expert vision researcher"

Connecting insight text between panels:
   "Within-round: fast failure avoidance.
    Cross-task: persistent expertise accumulation."

No 3D, crisp text, IEEE-paper-friendly.
```

---

## Guidance for refinement

If the first generation is too cluttered:
- Simplify to 3 lanes (drop the Input column, merge into Benchmark).
- Reduce text inside boxes to ≤ 3 words each.
- Remove icons if they crowd.

If labels render illegibly:
- Regenerate with the instruction "all labels must be clearly readable at
  1200px width".
- Or export as SVG and relabel in Inkscape/Figma for final paper.

## Notes for the paper

- Use **v1** as the main Figure 1 (system overview).
- Use **v3** as Figure 2 ("Alchemist vs AutoML-Agent comparison").
- Keep the final rendered file at `docs/alchemist_framework.png` (current
  placeholder file can be replaced).
