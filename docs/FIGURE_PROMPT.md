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

Layout from left to right in four stacked lanes:

(1) INPUT (leftmost column, amber accent):
   - Icon of a clipboard labeled "User Task"
   - Three small bullet chips underneath: "dataset path", "num classes",
     "metric + budget"

(2) BENCHMARK AGENT (second column, indigo accent):
   - A rounded rectangle labeled "Benchmark Agent"
   - Three incoming arrows from three database cylinders on the top:
       "HuggingFace Hub (timm ImageNet top-1)",
       "Papers-with-Code (task leaderboard)",
       "arXiv (recent SoTA)"
   - Inside the box: three small stacked steps: "Scout", "Score (HF-first,
     PwC-second)", "Filter external corpora"
   - Output arrow to the right labeled "Top-K compliant candidates"

(3) CONTROLLER + HARNESS (middle column, slate accent):
   - Rounded rectangle labeled "Controller Agent"
   - Inside: "Validate recommendation", "Resolve timm ID", "Baseline eval
     top-K", "Pick Winner"
   - Small icon of AWS EC2 GPU on the right with a dashed bidirectional
     arrow labeled "SSH + train_worker"
   - Output arrow to the right labeled "Winner base model"

(4) RESEARCH AGENT LOOP (rightmost column, teal accent):
   - Large rounded rectangle labeled "Research Agent"
   - Inside, a circular self-refinement loop with three nodes:
       R1 "Grid search (lr × freeze × adapter)",
       R2 "LLM-guided refinement",
       R3 "SoTA-gap analysis"
   - A small "Claude CLI" LLM bubble attached with a dotted line
   - Final output box on the far right labeled "Trained checkpoint + report"
     with a small trophy or medal icon

Global details:
- Thin arrows between the four lanes
- Tiny footer text at the bottom center: "ImageNet-1K / ImageNet-21K
  pretraining allowed · external corpora filtered (JFT, LAION, LVD-142M)"
- Everything aligned horizontally, no 3D effects, no drop shadows
- High resolution, crisp text labels, legible at paper-column width
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
