---
layout: pub
title: "Lazy Rudder"
page_id: main
extra_js:
  - /assets/js/figA.js
  - /assets/js/figB.js
  - /assets/js/figE.js
---

<div class="pub-section" id="abstract">

## Abstract & TL;DR

<div class="abstract-card">
<div class="tldr">TL;DR</div>
We train LoRA adapters on Pythia 70M–1B under DPO and CLM, then ask: does scale matter for <em>geometry</em>? The answer is no. Stable rank is ~3.6 regardless of model width — set by preference-learning complexity, not parameter count. We prove this is consistent with formal bounds (Lean 4), and show the γ subspace aligns 3–5× above chance across seeds and objectives.
</div>

**Full abstract.** We present an axiomatic framework decoupling energy from geometry in LoRA adapters under Direct Preference Optimization. Using Lean 4 formal verification, we prove that α-scaling bounds Frobenius energy but leaves stable rank unchanged (load-bearing theorem: `rsLoraUpdate_frob_bounded`). Empirically, across three orders of model width (Pythia 70M → 1B), we observe a task-intrinsic stable rank of $\text{srank} \approx 3.6 \pm 0.5$, independent of model capacity.

<button class="expand-trigger" onclick="this.nextElementSibling.classList.toggle('open'); this.textContent = this.nextElementSibling.classList.contains('open') ? '▲ Hide background' : '▼ New to LoRA? Start here'">▼ New to LoRA? Start here</button>
<div class="expandable">

**What is a LoRA adapter?** Large language models have billions of parameters. Fine-tuning all of them is expensive. LoRA (Low-Rank Adaptation) instead adds a small adapter — a pair of thin matrices $A$ and $B$ — beside each weight matrix. During training, only $A$ and $B$ update. The effective weight change is $\Delta W = BA$, which is low-rank.

**What is DPO?** Direct Preference Optimization is an alignment technique. Given pairs of (preferred, rejected) responses, DPO tunes the model to prefer the "good" response. The LoRA adapter absorbs the DPO signal.

**What is stable rank?** The rank of a matrix counts linearly independent directions. Stable rank is a smooth version: $\text{srank}(M) = \|M\|_F^2 / \|M\|^2$. It measures effective dimensionality of the adapter. Our finding: this number is ~3.6 no matter how large the model.

</div>

</div>

---

<div class="pub-section" id="srank-floor">

## The Task-Intrinsic Srank Floor

Across four model scales, the DPO adapter geometry converges to $\text{srank} \approx 3.6 \pm 0.5$. Scaling model width from 512 to 2048 does not widen the alignment manifold.

<div id="widget-figA" class="widget" aria-label="Srank floor scatter across model scales"></div>

**Reading the chart.** Each dot is one model (hover for exact value). Blue = DPO adapter; teal = CLM (available for 410M and 1B). The dashed line is the empirical floor at 3.6. Random LoRA matrices at $r=128$ would have $\text{srank} \approx 128$ — these adapters are ~35× lower-dimensional.

</div>

---

<div class="pub-section" id="gamma-rudder">

## The γ-Rudder: Subspace Overlap

The DPO adapter's top-$k$ singular vectors align with CLM's at 3–5× above random expectation. This is the γ-rudder signal: a shared geometric rudder that both DPO and CLM discover.

<div id="widget-figB" class="widget" aria-label="Bonus_R per layer at 1B, DPO vs CLM, two seeds"></div>

**Reading the chart.** Y-axis is $\text{bonus\_R}(k=5)$ — actual overlap divided by random baseline. A value of 3× means the top-5 singular vectors overlap 3× more than chance. Results for two independent seeds (42 and 117) track closely.

### Module universality

The γ signal appears across all four LoRA target modules, not just QKV.

<div id="widget-figE" class="widget" aria-label="Gamma bonus_R across four adapter modules"></div>

</div>

---

<div class="pub-section" id="falsifications">

## Falsification Results

Three hypotheses tested and rejected.

**BitFit gauge theory (falsified).** Bias-only DPO drives loss from 0.487 → 0.370 by step 800 — partial reduction, but the geometric signal is absent. Gauge theory dead.

<figure>
  <img src="{{ '/assets/img/figD.png' | relative_url }}" alt="BitFit vs LoRA loss trajectories">
  <figcaption>Fig D. BitFit-only DPO (orange) vs LoRA-DPO (blue) loss trajectories. BitFit captures gauge-accessible loss but not the geometric signal.</figcaption>
</figure>

**δ/δ′ quasiparticle (falsified).** No depth-wise structure in the DPO adapter. Layer-depth correlator $C(L, L+k)$ is flat (Pearson ≈ 0.97 — no trend). Angular-Fourier probe shows flat energy distribution.

**Bias-autopsy LN-γ (negative).** 99.97% of DPO adapter energy lies outside the LayerNorm gain subspace.

</div>

---

<div class="pub-section" id="lean-bounds">

## Lean Formal Bounds

Proven in Lean 4 (Mathlib v4.28.0):

1. **Frobenius energy decays with r** — $\|\Delta W\|_F^2 \leq \alpha^2 c$ (`rsLoraUpdate_frob_bounded`)
2. **α-scaling is rank-invariant** — scaling α while reducing r preserves $\alpha/\sqrt{r}$ (`stableRank_smul_invariant`)
3. **Stable rank is α-invariant** — $\text{srank}(\lambda M) = \text{srank}(M)$ for all $\lambda \neq 0$

See the [Lean page]({{ '/lean' | relative_url }}) for the full theorem status table.

</div>

---

<div class="pub-section" id="methods">

## Methods

**Models.** Pythia 70M, 160M, 410M, 1B (EleutherAI).

**LoRA config.** $r=128$, $\alpha=256$ for all primary runs. Target modules: `query_key_value`, `dense`, `dense_h_to_4h`, `dense_4h_to_h`.

**Training.** 800 steps DPO / CLM, LR=5e-6, cosine schedule, warmup=50, batch=8, fp16. Dataset: Anthropic/hh-rlhf (2000 samples).

**Seeds.** Primary: seed 42. Independent replication: seed 117 (independent data draw).

**Checkpoints.** All LoRA adapter checkpoints: [`d3banjan/lazy-rudder-checkpoints`](https://huggingface.co/d3banjan/lazy-rudder-checkpoints) (~1.9 GB).

<figure>
  <img src="{{ '/assets/img/figC.png' | relative_url }}" alt="Per-layer srank bars for 70M and 160M">
  <figcaption>Fig C. Per-layer stable rank for Pythia-70M and Pythia-160M DPO adapters.</figcaption>
</figure>

</div>

---

<div class="pub-section" id="references">

## References & Citation

<div class="citation-box">@misc{basu2026lazyrugder,
  title  = {Axiomatic Bounds on {LoRA} Alignment Geometry:
             A Task-Intrinsic Dimensional Floor Across {Pythia} 70M--1B},
  author = {Basu, Debanjan},
  year   = {2026},
  url    = {https://github.com/d3banjan/lazy-rudder-paper}
}</div>

**Key references:** Hu et al. (2022) LoRA; Rafailov et al. (2023) DPO; Kalajdzievski (2023) RsLoRA; EleutherAI Pythia suite; Lean 4 + Mathlib.

</div>
