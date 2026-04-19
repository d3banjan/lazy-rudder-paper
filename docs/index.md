---
layout: pub
title: "Low Stable-Rank Structure in LoRA-DPO Adapters on Pythia 70M–1B: Empirical Scaling and Formal Invariants"
page_id: main
extra_js:
  - /assets/js/figA.js
  - /assets/js/figB.js
  - /assets/js/figE.js
---

<div class="pub-section hero" id="hero">

# Low Stable-Rank Structure in LoRA-DPO Adapters on Pythia 70M–1B: Empirical Scaling and Formal Invariants

<p class="byline" markdown="0">Debanjan Basu &nbsp;&middot;&nbsp; 2026 &nbsp;&middot;&nbsp; Pythia 70M&ndash;1B &nbsp;&middot;&nbsp; <a href="https://github.com/d3banjan/lazy-rudder-paper" class="byline-link">code &amp; data</a></p>

<p class="hook" markdown="0">We made a model four times bigger. The adapter stable rank stayed the same — under this recipe, on this dataset.</p>

When you fine-tune a language model with LoRA under Direct Preference Optimization, the adapter learns to move in a tiny number of directions — roughly 3 to 4, across Pythia models from 70M to 1B parameters, trained on hh-rlhf with a fixed LoRA configuration. We call this **width-stable low-rank structure under fixed recipe**. The observed floor appears consistent with the preference signal complexity, though we have not varied the dataset or LoRA configuration to isolate causes.

<div id="widget-figA" class="widget" aria-label="Srank floor scatter across model scales"></div>

<p class="aside" markdown="0">Each dot is one model. Hover for exact values. The dashed line is the empirical floor at srank&nbsp;&asymp;&nbsp;3.6. A random LoRA matrix at r=128 would have srank&nbsp;&asymp;&nbsp;128 &mdash; these adapters are ~35&times; lower-dimensional.</p>

<div class="callout-row" markdown="0">
  <div class="callout"><div class="num">3.6</div><div class="label">average stable rank<br>across all models</div></div>
  <div class="callout"><div class="num">4&times;</div><div class="label">width increase<br>with no geometry change</div></div>
  <div class="callout"><div class="num">3&ndash;5&times;</div><div class="label">above-chance subspace<br>alignment (&gamma; signal)</div></div>
</div>

</div>

---

<div class="pub-section" id="what-is-srank">

Background
{:.section-label}

## What does "stable rank 3.6" mean?

A LoRA adapter is a matrix — a grid of numbers that shifts a weight in the model. That matrix can be "wide" (spread across many directions in space) or "narrow" (concentrated in just a few). **Stable rank** measures this width: a value of 3.6 means the adapter's energy is effectively spread across ~3–4 directions, even though the matrix formally has 128 dimensions.

Think of a ship's rudder. The ship is enormous. The rudder is tiny. But the rudder has one job — deflect the flow — and it does that job in a small number of geometric directions. Our finding: DPO always uses a "rudder" of about 3–4 directions, no matter how large the ship.

<div class="finding" markdown="0"><strong>The surprising part:</strong> we expected larger models to learn richer, higher-dimensional alignment geometry. Under this fixed recipe and dataset, they don't. A 1B-parameter model uses the same number of effective directions as a 70M-parameter model fine-tuned on the same data. Whether retraining with different (r, α) or a different dataset would change this remains an open empirical question.</div>

<button class="expand-trigger" onclick="this.nextElementSibling.classList.toggle('open');this.textContent=this.nextElementSibling.classList.contains('open')?'▲ Less detail':'▼ More detail: what is stable rank exactly?'">▼ More detail: what is stable rank exactly?</button>
<div class="expandable">

Formally: $\text{srank}(M) = \|M\|_F^2\, /\, \|M\|_2^2$. The numerator is total energy (sum of all squared singular values). The denominator is the energy in the single largest direction. The ratio tells you how many directions carry the energy. If all energy is in one direction, srank = 1. If energy is spread equally across 128 directions, srank = 128.

Crucially, srank is **invariant to scaling**: multiplying the matrix by any scalar leaves it unchanged. This matters for LoRA, where the $\alpha$ hyperparameter scales the adapter. We prove this formally in Lean 4 (`stableRank_smul_invariant`). It means the geometry we measure is real — not an artifact of how we initialized the adapter.

</div>

</div>

---

<div class="pub-section" id="gamma-signal">

Key Finding
{:.section-label}

## DPO and CLM find the same directions

Here is the second surprise. When we train on preference data (DPO) vs. plain language modelling (CLM), we get different adapters — but their top singular vectors overlap far more than chance. We call this the **γ-rudder signal**.

The chart below shows, for each layer of a 1B-parameter model, how much DPO and CLM agree on the important directions. A value of 3× means their top-5 singular vectors share 3× more subspace than the **analytic Haar-random expectation** — the closed-form baseline E[overlap] = k/d for two uniformly-random k-frames in R^d (not a sampled approximation; exact for any finite d).

<div id="widget-figB" class="widget" aria-label="Bonus_R per layer at 1B, DPO vs CLM, two seeds"></div>

<p class="aside" markdown="0">Four traces: DPO and CLM at two independent seeds (42 and 117). The seeds use completely different data draws &mdash; yet the curves track closely. The signal is not a random artifact.</p>

This alignment isn't specific to one part of the model. It appears in all four LoRA target modules: the attention projections and both MLP layers.

<div id="widget-figE" class="widget" aria-label="Gamma bonus_R across four adapter modules"></div>

<div class="finding" markdown="0"><strong>What this means:</strong> DPO top subspaces overlap with CLM top subspaces above the analytic Haar-random baseline (E[bonus] = 1 exactly; E[p(k)] = k/d) on hh-rlhf. The update concentrates in directions the base model already uses — a structural geometric fact. Crucially, a behavioral cross-check (T1.2, Fig G) shows that higher γ does <em>not</em> predict larger reward margin; the correlation is negative. Geometric structure and behavioral outcome are decoupled on this dataset.</div>

</div>

---

<div class="pub-section" id="falsifications">

What we ruled out
{:.section-label}

## Three explanations that didn't survive

Before concluding the srank floor is real, we tried to explain it away with three targeted alternatives. All three failed, though they do not exhaust the space of possible confounds.

**Attempt 1: maybe it's just the biases.** If LayerNorm gain vectors (γ parameters) span the adapter's subspace, then the geometry would be trivially determined by initialization — not preference learning. We tested this by projecting each DPO adapter onto the LayerNorm gain subspace. Result: **99.97% of the energy lies outside it.** This rules out the LayerNorm-gain subspace as an explanation. It does NOT rule out a broader class of pretrained-anisotropy explanations (weight curvature, weight-tying, token-frequency bias, optimizer-induced anisotropy).

**Attempt 2: maybe bias-only fine-tuning reproduces it.** BitFit trains only bias parameters — no weight matrices at all. If BitFit-DPO produces the same loss reduction as LoRA-DPO, the geometric signal would be "gauge-accessible" (reachable without learning any subspace). We ran BitFit-DPO to 800 steps and beyond.

<figure markdown="0">
  <img src="/lazy-rudder-paper/assets/img/figD.png" alt="BitFit vs LoRA loss trajectories">
  <figcaption>BitFit-DPO (orange) reduces loss — but plateaus well above LoRA-DPO (blue). The loss gap is consistent with LoRA learning geometry that biases alone cannot replicate in this setting.</figcaption>
</figure>

**Attempt 3: maybe depth carries structure.** If DPO adapters form a "quasiparticle" — a correlated pattern that travels across layers — we'd expect the layer-depth correlator $C(L, L+k)$ to decay with $k$. It doesn't. The correlator is flat (Pearson ≈ 0.97 regardless of depth gap). No depth-wise structure. No quasiparticle.

</div>

---

<div class="pub-section" id="lean-bounds">

Formal grounding
{:.section-label}

## Ruling out a measurement artifact from α or r

One might worry: maybe we're measuring an artifact of hyperparameter choice. If we change the LoRA scaling $\alpha$, does srank change?

We prove it doesn't — formally, in Lean 4:

- **`stableRank_smul_invariant`** — scaling a fixed learned matrix by any $\lambda \neq 0$ leaves srank unchanged
- **`rsLoraUpdate_frob_bounded`** — Frobenius energy obeys $\|\Delta W\|_F^2 \leq \alpha^2 c$, while srank is invariant to the scalar

The proofs use Mathlib's linear algebra library and are machine-checked. This rules out a trivial measurement artifact: the observed srank ≈ 3.6 is not an artifact of our α setting. **Important caveat:** this does NOT imply that retraining with different (α, r) would converge to the same geometry. What happens when you retrain under a different configuration is an open empirical question.

<figure markdown="0">
  <img src="/lazy-rudder-paper/assets/img/figF.png" alt="Conceptual schematic: the lazy rudder">
  <figcaption>The pretrained model occupies a large subspace (blue ellipse). The DPO adapter (red arrow) is concentrated in ~3–4 directions — a small rudder on a large ship. The α hyperparameter controls the length of the arrow; the formal invariant shows direction is unchanged by α scaling of a fixed matrix.</figcaption>
</figure>

See the [Lean proof status →]({{ '/lean' | relative_url }})

</div>

---

<div class="pub-section" id="methods">

For the curious
{:.section-label}

## How the experiments work

**Models.** Pythia 70M, 160M, 410M, 1B (EleutherAI). Pre-trained, no instruction tuning.

**LoRA.** $r=128$, $\alpha=256$. Targets: `query_key_value`, `dense`, `dense_h_to_4h`, `dense_4h_to_h`. 800 training steps, LR=5e-6, cosine schedule, fp16.

**Data.** Anthropic/hh-rlhf, 2000 samples. DPO uses preference pairs. CLM uses chosen responses only (no rejection signal).

**Reproducibility.** All adapter checkpoints are mirrored at [`d3banjan/lazy-rudder-checkpoints`](https://huggingface.co/d3banjan/lazy-rudder-checkpoints) (~1.9 GB). Every figure regenerates from `make analysis && make paper`.

<figure markdown="0">
  <img src="/lazy-rudder-paper/assets/img/figC.png" alt="Per-layer srank for 70M and 160M">
  <figcaption>Per-layer stable rank for Pythia-70M and Pythia-160M. Smaller models show a slight upward drift in early layers — a weak "acoustic scaling" trend that co-occurs with lower width, and flattens at 410M and above.</figcaption>
</figure>

</div>

---

<div class="pub-section" id="behavior-geometry">

Reviewer M9 response
{:.section-label}

## Geometry–behavior decoupling (T1.2)

We measured **reward margin** and **KL-to-base** on 495 clean held-out `Anthropic/hh-rlhf` test examples for all five DPO checkpoints (70M, 160M, 410M, 1B×2 seeds), β=0.1, fp16.

**Reward margin** = β · [log π\_θ(y\_win|x)/π\_ref(y\_win|x) − log π\_θ(y\_los|x)/π\_ref(y\_los|x)]. **KL-to-base** = mean per-token log-ratio of DPO vs base on chosen response (teacher-forced; negative = DPO keeps probability close to or below base, consistent with β-regularization).

<figure markdown="0">
  <img src="/lazy-rudder-paper/assets/img/figG.png" alt="Geometry-behavior decoupling scatter grid">
  <figcaption><strong>Fig G — decoupling evidence.</strong> 2×2 scatter: (srank, γ) × (reward margin, KL-to-base). Each point is one DPO checkpoint. Error bars = ±1 SE over 495 examples. Pearson r with 95% bootstrap CI in each panel. n=5; all CIs are wide and one pins at ±1 — treat as suggestive, not definitive. The γ–reward-margin Pearson is −0.35 (Spearman −0.60): higher subspace overlap predicts <em>lower</em> reward margin, opposite to the benign-steering prior. The γ–KL-to-base Pearson is +0.49: higher overlap co-occurs with greater divergence from base. The srank–reward-margin Pearson is near zero. T2.1 and T1.3 are the adjudicating experiments.</figcaption>
</figure>

**Primary finding: structural geometry and behavioral outcome are decoupled.** Geometric alignment (γ) does not predict reward; it predicts KL drift. Tighter overlap with pretrained right-subspaces correlates with *more* divergence from the base distribution and *lower* reward margin — not efficient alignment. Stable rank (srank) is uncorrelated with reward margin across this scale range: it measures the rigid geometric constraint under which learning is forced to happen (the width of the pipe), not how successful that learning turns out to be. Adapters with r=128, α=256 on hh-rlhf converge to srank ≈ 3.6 regardless of model size; reward margin varies independently. This decoupling — not a confirmation of benign steering — is the paper's primary empirical result.

**Caveat.** n=5 checkpoints; bootstrap CIs are wide and one pins at ±1. The most parsimonious reading of the current data is decoupling, but confirmation requires T2.1 (non-Pythia replication) and T1.3 (additional seeds).

</div>

---

<div class="pub-section" id="transient-regime">

Open question
{:.section-label}

## Does the srank floor persist, or is it a transient early-training artefact?

Reviewer M7 asked whether the observed srank ≈ 3–4 at step 800 reflects convergence or just an early-training lazy regime. The right test is a **srank-vs-training-step trajectory** for each model size.

<figure markdown="0">
  <img src="/lazy-rudder-paper/assets/img/fig_F_progression.png" alt="[Pending] srank vs training step">
  <figcaption><strong>[Pending data]</strong> This plot will show stable rank vs. training step for Pythia 70M, 160M, 410M, and 1B (LoRA-DPO, r=128). A plateau in the last 30–50% of training would rebut the transient-regime critique. Currently a placeholder — no intermediate LoRA checkpoints were saved during the original training runs. Pending re-training with <code>save_steps ≤ 100</code> or a SrankCallback. See <code>scripts/generate_fig_F.py</code> for the data schema.</figcaption>
</figure>

**What we know now.** The loss curve (logged at 10-step intervals in `trainer_state.json`) decreases monotonically through step 800 with no plateau, suggesting training was active throughout. The srank floor is consistent across DPO and CLM objectives and across two independent seeds at 1B scale — harder to explain as a transient coincidence. These are supportive signals but not a substitute for a trajectory plot. We flag this as a Tier-1 follow-up.

</div>

---

<div class="pub-section" id="references">

Citation
{:.section-label}

## Cite this work

<div class="citation-box" markdown="0">@misc{basu2026lazyrugder,
  title  = {Low Stable-Rank Structure in {LoRA}-{DPO} Adapters on {Pythia} 70M--1B:
             Empirical Scaling and Formal Invariants},
  author = {Basu, Debanjan},
  year   = {2026},
  url    = {https://github.com/d3banjan/lazy-rudder-paper}
}</div>

**References.** Hu et al. (2022) LoRA; Rafailov et al. (2023) DPO; Kalajdzievski (2023) RsLoRA; Biderman et al. (2023) Pythia; Lean 4 + Mathlib.

</div>
