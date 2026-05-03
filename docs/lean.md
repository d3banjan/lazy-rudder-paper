---
layout: pub
title: "Lean Proofs"
page_id: lean
extra_js:
  - /assets/js/lean.js
---

<div class="pub-section hero" id="lean-hero">

# The Formal Proofs

<p class="byline" markdown="0">Machine-checked in Lean 4 + Mathlib v4.28.0</p>

<p class="hook" markdown="0">We proved the structural invariants in a proof assistant. The computer verified every step — for the proven subset.</p>

A "proof assistant" is a programming language where the compiler checks mathematical arguments, not just syntax. If the code compiles, the theorem is proved — no gaps, no hand-waving. We used Lean 4 with the Mathlib library of 200,000+ verified results.

**Coverage as of 2026-05-02:** 13 theorems fully proven (no `sorry`), 2 partial (proved in weakened form; see in-file WEAKENING NOTE), 4 deferred proof bodies (real statements, `sorry` body), 0 `True := sorry` literature-terminology stubs, 7 definitions. Three of the deferred items are paper-facing (gamma_right_alignment, bias_autopsy_separation, lower_bound_of_intent — listed below). The machine-checked guarantee applies only to the 13 proven theorems; partial and deferred items are not machine-checked in a load-bearing sense. See the full table below.

**Source.** Theorem statements and proofs are vendored at [`lean/LeanMining/NeuralGeometry/SubspaceOverlap.lean`](https://github.com/d3banjan/lazy-rudder-paper/blob/main/lean/LeanMining/NeuralGeometry/SubspaceOverlap.lean). Each theorem name in the table below links to its declaration line in that file. The vendored copy is kept in sync with a private upstream monorepo by `make sync-lean`; `make verify-lean` fails on drift. Build and verify locally:

```bash
cd lean/
lake update   # first run only — pulls Mathlib (~5 min, ~3 GB)
lake build
```

**Three paper-facing sorry stubs** — these names appear in the paper's claims but are not yet proven:

| Name | Status | Notes |
|---|---|---|
| `gamma_right_alignment` | not yet proven — empirical / aspirational | requires quantitative bound from training-process formalization, outside Mathlib scope |
| `bias_autopsy_separation` | not yet proven — empirical / aspirational | the 99.97% residual is a data measurement; the underlying algebra is a research target but not yet formalized |
| `lower_bound_of_intent` | not yet proven — aspirational | connecting srank floor to DPO loss requires task-loss functional formalization outside current Mathlib |

**Status:** <span id="lean-counts">loading…</span>

</div>

---

<div class="pub-section" id="glossary">

Glossary
{:.section-label}

## Terms used in the theorems

**Matrix** — A rectangular grid of real numbers. Here, every weight update *ΔW* applied to the model is a matrix: it says "move the output of this layer in these directions by these amounts."

**Rank** — The number of linearly independent directions a matrix spans. A rank-3 matrix lives in a 3-dimensional subspace of a potentially much larger space. A random 128×128 matrix has rank 128; trained LoRA adapters have effective rank far lower.

**Frobenius norm** (‖M‖_F) — The total "size" of a matrix: square root of the sum of all squared entries. Think of it as the Euclidean length of the matrix if you unroll it into a single long vector. Captures overall energy regardless of direction.

**Spectral norm** (‖M‖) — The maximum amount a matrix can stretch a unit-length input vector — the energy in the single strongest direction. Measures "how powerful is the adapter in its best direction."

**Stable rank** (srank(M) = ‖M‖²_F / ‖M‖²) — How many directions carry the energy. If all energy is in one direction, srank = 1. If spread equally across k directions, srank ≈ k. Always between 1 and rank(M). Our finding: trained DPO adapters have srank ≈ 3.6 regardless of model size.

**Outer product** (u·vᵀ) — A matrix built from two vectors: entry (i,j) = u_i × v_j. Always has rank ≤ 1 — spans at most one direction. A single gradient step in DPO is approximately a rank-1 outer product.

**LoRA update** (ΔW = (α/r)·BA) — Low-Rank Adaptation: instead of updating the full weight matrix W, add a product BA where B is m×r and A is r×n (r ≪ m,n). The factor α/r scales the update. Only A and B are trained; W is frozen.

**RsLoRA update** (ΔW = (α/√r)·BA) — A variant of LoRA that uses α/√r instead of α/r. The different normalisation keeps the update's energy stable as r changes — proved in `rsLoraUpdate_frob_bounded`.

</div>

---

<div class="pub-section" id="load-bearing">

Load-bearing theorems
{:.section-label}

## The results the paper depends on

---

### `rsLoraUpdate_frob_bounded`

**Statement.** If $\|BA\|_F^2 \leq c \cdot r$, then the RsLoRA update satisfies $\left\|\tfrac{\alpha}{\sqrt{r}} \cdot BA\right\|_F^2 \leq \alpha^2 c$. No $r$ on the right-hand side.

**Plain English.** RsLoRA's energy is bounded by α²·c regardless of rank. Choosing r=16 or r=128 gives the same bound on how much the adapter can affect the model.

**Why it matters.** This is the formal proof that RsLoRA's normalisation achieves rank-invariant energy — α is a clean dial for amplitude that doesn't interact with the geometry of a fixed learned matrix.

**Analogy.** Think of α as a volume knob and r as the number of speaker channels. Standard LoRA turns down the volume per-channel as you add more channels. RsLoRA keeps total volume constant — you can add channels without the signal getting quieter.

---

### `stableRank_smul_invariant`

**Statement.** For any scalar $\lambda \neq 0$ and matrix $M$: $\text{srank}(\lambda M) = \text{srank}(M)$.

**Plain English.** Scaling the adapter — by changing α or any other scalar factor — does not change its stable rank. The geometry is invariant to amplitude.

**Why it matters.** The empirical srank ≈ 3.6 could be an artefact of how α is set. This theorem rules that out for a fixed learned matrix: the srank reflects the adapter's directional structure, not its size. Changing α moves the adapter further or closer, but always along the same directions. **Scope caveat:** this does NOT imply that retraining with a different (α, r) configuration would converge to the same geometry. The theorem concerns post-hoc rescaling of an already-trained matrix, not the training dynamics. Whether different training configurations converge to the same srank is an open empirical question.

**Analogy.** Zooming in on a map changes the scale bar but not the shape of the coastline. Multiplying a matrix by a scalar changes its energy but leaves the subspace it spans untouched.

---

### `loraUpdate_frob_decays`

**Statement.** Under the same hypothesis $\|BA\|_F^2 \leq c \cdot r$, standard LoRA satisfies $\left\|\tfrac{\alpha}{r} \cdot BA\right\|_F^2 \leq \frac{\alpha^2 c}{r}$. The bound decays as 1/r.

**Plain English.** In standard LoRA, doubling the rank halves the update's energy at fixed α. This is why high-rank standard LoRA often needs manual α tuning, while RsLoRA doesn't.

**Analogy.** Splitting a fixed salary budget across twice as many workers leaves each worker paid half as much. Standard LoRA splits its "budget" across r directions; RsLoRA keeps the total budget constant.

---

### `rank_sum_outer_products_le`

**Statement.** A sum of r rank-1 outer products has rank at most r: $\operatorname{rank}\!\left(\sum_{i=1}^{r} u_i v_i^\top\right) \leq r$.

**Plain English.** You cannot build a subspace with more dimensions than the number of directions you used to construct it. LoRA adapters are sums of rank-1 outer products, so their rank is bounded by r.

**Analogy.** You cannot build a 3D object from 2 directions of motion. Two perpendicular rods define a plane; a third is needed to reach the third dimension. This theorem says the same for matrices.

---

</div>

---

<div class="pub-section" id="foundations">

Foundation lemmas
{:.section-label}

## Supporting results

**`ratio_smul_invariant_of_quadratic`** — f(cM)/g(cM) = f(M)/g(M) when both f and g are c²-homogeneous. The algebraic backbone of `stableRank_smul_invariant`: Frobenius² and spectral² both scale as c², so their ratio cancels the scalar out.

**`frobeniusSq_smul`** — ‖cM‖²_F = c²·‖M‖²_F. Scaling multiplies energy by c². Used to extract scalar factors from norm calculations in the main theorems.

**`spectralSq_smul`** — ‖cM‖² = c²·‖M‖². Same result for spectral norm. Paired with frobeniusSq_smul to prove the ratio invariance underlying `stableRank_smul_invariant`.

**`rank_outer_product_le_one`** — rank(uvᵀ) ≤ 1. A single outer product spans at most one direction. This is the base case for `rank_sum_outer_products_le`.

**`rank_add_le`** — rank(A+B) ≤ rank(A) + rank(B). Rank is subadditive. Combining two low-rank updates cannot produce a result with higher rank than the sum of their ranks.

**`outerProduct_eq_col_mul_row`** — uvᵀ = col(u)·row(v). Bridges the outer product definition to Mathlib's matrix multiplication API, enabling the rank lemmas above.

**`frobeniusSq_nonneg`** — ‖M‖²_F ≥ 0. Energy is non-negative. Trivially true but required as a precondition in positivity tactics elsewhere.

**`frobeniusSq_eq_zero_iff`** — ‖M‖²_F = 0 ↔ M = 0. The zero matrix is the only matrix with zero energy. Used in edge-case reasoning.

</div>

---

<div class="pub-section" id="stubs">

Work in progress
{:.section-label}

## Named targets for future proofs

These are stubs — either `True := sorry` placeholders that reserve a theorem name, or statements with a real body replaced by `sorry`. Not a gap in the proved results above; they mark the next layer of the theory. Paper-facing sorry stubs are marked below.

**`stableRank_le_rank`** — srank(M) ≤ rank(M) always. The effective dimensionality is never larger than the formal rank count.

**`gamma_right_alignment`** _(paper-facing, not yet proven — empirical/aspirational)_ — A quantified lower bound on the γ subspace overlap: right singular vectors of DPO and CLM adapters of the same base model align above a computable threshold. Requires formalizing the LoRA training process and pretraining distribution, outside Mathlib scope. Not cited as proven.

**`random_subspace_expected_overlap`** _(paper-facing, partial — closed 2026-04-18)_ — Proved in weakened deterministic form (`subspaceOverlap U W ≤ 1`). The exact k/n expectation under Haar measure requires Grassmannian integration outside current Mathlib. Partial proof sufficient for the paper claim.

**`stable_rank_acoustic_scaling`** _(paper-facing, partial — closed 2026-04-18)_ — Proved under acoustic axioms (`frobeniusSq = d`, `spectralSq = √d`). The fully general sub-linear derivation requires an analytic proof outside current Mathlib scope. Partial proof sufficient for the paper claim.

**`subspace_dilution`** — Adding more rank dimensions dilutes energy per direction — the geometric counterpart to `loraUpdate_frob_decays`.

**`bias_autopsy_separation`** _(paper-facing, not yet proven — empirical/aspirational)_ — The 99.97% residual finding that ΔW lies outside the column-wise rescaling subspace of W. The quantitative claim depends on checkpoint data; the algebraic research target is noted in-file but not cited as proven.

**`lower_bound_of_intent`** _(paper-facing, not yet proven — aspirational)_ — A lower bound on srank for any adapter that reduces DPO loss by a given amount — connecting the geometric floor to the training objective. Requires task-loss functional formalization and a local quadratic Hessian bound, both outside current Mathlib. Not cited as proven.

**`stable_rank_disentanglement`** — Energy and geometry fully decouple: you can change ‖ΔW‖²_F freely without changing srank(ΔW), and vice versa.

</div>

---

<div class="pub-section" id="lean-full-table">

Full status table
{:.section-label}

## All declarations

<table style="width:100%;border-collapse:collapse;font-size:.85rem;" markdown="0">
  <thead>
    <tr>
      <th style="text-align:left;padding:.4rem .5rem;border-bottom:2px solid var(--border);font-family:var(--font-sans);font-size:.78rem;">Name</th>
      <th style="text-align:left;padding:.4rem .5rem;border-bottom:2px solid var(--border);font-family:var(--font-sans);font-size:.78rem;">Status</th>
      <th style="text-align:left;padding:.4rem .5rem;border-bottom:2px solid var(--border);font-family:var(--font-sans);font-size:.78rem;">Kind</th>
    </tr>
  </thead>
  <tbody id="lean-tbody">
    <tr><td colspan="3" style="padding:.4rem;color:var(--fg-muted)">Loading…</td></tr>
  </tbody>
</table>

<p style="margin-top:1rem;font-family:var(--font-sans);font-size:.82rem;color:var(--fg-muted)" markdown="0">
  <span class="lean-badge proven">proven</span>&nbsp; complete proof, no <code>sorry</code> &nbsp;&nbsp;
  <span class="lean-badge partial">partial</span>&nbsp; weakened form proved (see in-file WEAKENING NOTE) &nbsp;&nbsp;
  <span class="lean-badge deferred">deferred</span>&nbsp; real statement, proof body is <code>sorry</code> &nbsp;&nbsp;
  <span class="lean-badge stub">stub</span>&nbsp; <code>True := sorry</code> placeholder
</p>

</div>
