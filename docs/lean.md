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

<p class="hook" markdown="0">We proved the key claims in a proof assistant. The computer verified every step.</p>

A "proof assistant" is a programming language where the compiler checks mathematical arguments, not just syntax. If the code compiles, the theorem is proved — no gaps, no hand-waving. We used [Lean 4](https://leanprover.github.io/) with the [Mathlib](https://leanprover-community.github.io/mathlib4_docs/) library of 200,000+ verified results.

**Status:** <span id="lean-counts">loading…</span>

</div>

---

<div class="pub-section" id="glossary">

Glossary
{:.section-label}

## Terms used in the theorems

<dl class="glossary">

<dt>Matrix</dt>
<dd>A rectangular grid of real numbers. Here, every weight update <em>ΔW</em> is a matrix — it says "move the output of this layer in these directions by these amounts."</dd>

<dt>Rank</dt>
<dd>The number of linearly independent directions a matrix spans. A rank-3 matrix lives in a 3-dimensional subspace of a potentially much larger space. A random 128×128 matrix has rank 128; trained LoRA adapters have effective rank far lower.</dd>

<dt>Frobenius norm</dt>
<dd markdown="0">Written ‖M‖<sub>F</sub>. The total "size" of a matrix: square root of the sum of all squared entries. Think of it as the Euclidean length of the matrix if you unroll it into a single long vector. Captures overall energy regardless of direction.</dd>

<dt>Spectral norm</dt>
<dd markdown="0">Written ‖M‖. The maximum amount a matrix can stretch a unit-length input vector — the energy in the single strongest direction. Measures "how powerful is the adapter in its best direction."</dd>

<dt>Stable rank</dt>
<dd markdown="0">srank(M) = ‖M‖²<sub>F</sub> / ‖M‖². How many directions carry the energy. srank = 1 means all energy is in one direction. srank ≈ k means energy is spread across about k directions. Always between 1 and rank(M). Our finding: DPO adapters have srank ≈ 3.6 regardless of model size.</dd>

<dt>Outer product</dt>
<dd markdown="0">Written u·vᵀ. A matrix built from two vectors: entry (i,j) = u<sub>i</sub> × v<sub>j</sub>. Always has rank ≤ 1 — spans at most one direction. A single gradient step in DPO is approximately a rank-1 outer product.</dd>

<dt>LoRA update</dt>
<dd markdown="0">ΔW = (α/r)·BA, where B is m×r and A is r×n (r ≪ m,n). Only A and B are trained; the base weight W is frozen. The factor α/r scales the update.</dd>

<dt>RsLoRA update</dt>
<dd markdown="0">ΔW = (α/√r)·BA. Same structure as LoRA but uses α/√r instead of α/r. The different normalization keeps update energy stable as r changes — proved in <code>rsLoraUpdate_frob_bounded</code>.</dd>

</dl>

</div>

---

<div class="pub-section" id="load-bearing">

Load-bearing theorems
{:.section-label}

## The results the paper depends on

---

### `rsLoraUpdate_frob_bounded`

**Statement.** If $\|BA\|_F^2 \leq c \cdot r$, then:

$$\left\|\tfrac{\alpha}{\sqrt{r}} \cdot BA\right\|_F^2 \leq \alpha^2 c$$

No $r$ on the right-hand side.

**Plain English.** RsLoRA's energy is bounded by $\alpha^2 \cdot c$ regardless of rank. Choosing $r=16$ or $r=128$ gives the same bound — provided the inner matrices scale appropriately with $r$.

**Why it matters for the paper.** This is the formal proof that the RsLoRA normalisation ($\alpha/\sqrt{r}$) achieves rank-invariant energy. It means α is a clean dial for amplitude that doesn't interact with the geometry.

**Analogy.** Think of α as a volume knob and r as the number of speaker channels. Standard LoRA turns down the volume per-channel as you add more channels. RsLoRA keeps total volume constant — you can add channels without the signal getting quieter.

---

### `stableRank_smul_invariant`

**Statement.** For any scalar $\lambda \neq 0$ and any matrix $M$:

$$\text{srank}(\lambda M) = \text{srank}(M)$$

**Plain English.** Scaling the adapter — by any non-zero amount — does not change its stable rank. The *geometry* of the adapter is invariant to its *amplitude*.

**Why it matters for the paper.** The empirical srank ≈ 3.6 could be an artefact of how α is set. This theorem rules that out: the srank you measure reflects the adapter's directional structure, not its size. Changing α moves the adapter further or closer, but always along the same directions.

**Analogy.** Zooming in on a map changes the scale bar but not the shape of the coastline. Multiplying a matrix by a scalar changes its energy but leaves the subspace it spans untouched.

---

### `loraUpdate_frob_decays`

**Statement.** Under the same hypothesis $\|BA\|_F^2 \leq c \cdot r$, standard LoRA satisfies:

$$\left\|\tfrac{\alpha}{r} \cdot BA\right\|_F^2 \leq \frac{\alpha^2 c}{r}$$

The bound *decays as $1/r$* — energy shrinks as rank grows.

**Plain English.** In standard LoRA, doubling the rank halves the update's energy at fixed α. This is why high-rank standard LoRA often needs manual α tuning, while RsLoRA doesn't.

**Analogy.** Splitting a fixed payroll budget across twice as many workers leaves each worker paid half as much. Standard LoRA splits its "update budget" across r directions; as r grows the per-direction budget shrinks.

---

### `rank_sum_outer_products_le`

**Statement.** A sum of $r$ rank-1 outer products has rank at most $r$:

$$\operatorname{rank}\!\!\left(\sum_{i=1}^{r} u_i v_i^\top\right) \leq r$$

**Plain English.** You cannot build a subspace with more dimensions than the number of directions you used to construct it. LoRA adapters are sums of rank-1 outer products, so their rank is provably bounded by $r$.

**Analogy.** You cannot build a 3D object from 2 directions of motion. Two perpendicular rods define a plane; a third rod is needed to reach the third dimension. This theorem says the same thing for matrices.

---

</div>

---

<div class="pub-section" id="foundations">

Foundation lemmas
{:.section-label}

## Supporting results

<table class="lean-table">
  <thead>
    <tr>
      <th>Name</th>
      <th>What it says</th>
      <th>Role</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>ratio_smul_invariant_of_quadratic</code></td>
      <td markdown="0">f(cM)/g(cM) = f(M)/g(M) when f and g are both c²-homogeneous</td>
      <td>The algebraic backbone of <code>stableRank_smul_invariant</code>: Frobenius² and spectral² both scale as c², so their ratio cancels the scalar out</td>
    </tr>
    <tr>
      <td><code>frobeniusSq_smul</code></td>
      <td markdown="0">‖cM‖²_F = c²·‖M‖²_F</td>
      <td>Scaling multiplies energy by c² — used to extract scalar factors from norm calculations</td>
    </tr>
    <tr>
      <td><code>spectralSq_smul</code></td>
      <td markdown="0">‖cM‖² = c²·‖M‖²</td>
      <td>Same result for spectral norm — paired with frobeniusSq_smul to prove the ratio invariance</td>
    </tr>
    <tr>
      <td><code>rank_outer_product_le_one</code></td>
      <td markdown="0">rank(uvᵀ) ≤ 1</td>
      <td>A single outer product spans at most one direction. Base case for rank_sum_outer_products_le</td>
    </tr>
    <tr>
      <td><code>rank_add_le</code></td>
      <td markdown="0">rank(A+B) ≤ rank(A) + rank(B)</td>
      <td>Rank is subadditive. Combining two low-rank updates cannot produce a result of higher rank than the sum</td>
    </tr>
    <tr>
      <td><code>outerProduct_eq_col_mul_row</code></td>
      <td markdown="0">uvᵀ = col(u)·row(v)</td>
      <td>Bridges the outer product definition to Mathlib's matrix multiplication, enabling rank lemmas</td>
    </tr>
    <tr>
      <td><code>frobeniusSq_nonneg</code></td>
      <td markdown="0">‖M‖²_F ≥ 0</td>
      <td>Energy is non-negative. Trivially true but needed as a hypothesis in positivity tactics</td>
    </tr>
    <tr>
      <td><code>frobeniusSq_eq_zero_iff</code></td>
      <td markdown="0">‖M‖²_F = 0 ↔ M = 0</td>
      <td>The zero matrix is the only matrix with zero energy — used in edge-case reasoning</td>
    </tr>
  </tbody>
</table>

</div>

---

<div class="pub-section" id="stubs">

Work in progress
{:.section-label}

## Named targets for future proofs

These are `True := sorry` stubs — the name is reserved, the real statement comes later. They are not mathematical claims; they mark the next layer of the theory.

<table class="lean-table">
  <thead>
    <tr>
      <th>Name</th>
      <th>What it would say when proved</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>stableRank_le_rank</code></td>
      <td>srank(M) ≤ rank(M) always. The effective dimensionality is never larger than the formal rank count.</td>
    </tr>
    <tr>
      <td><code>gamma_right_alignment</code></td>
      <td>A quantified lower bound on the γ subspace overlap: right singular vectors of DPO and CLM adapters trained on the same base model align above a computable threshold.</td>
    </tr>
    <tr>
      <td><code>random_subspace_expected_overlap</code></td>
      <td>The expected overlap of two random k-dimensional subspaces of ℝⁿ is k/n. This is the formal baseline that the empirical bonus_R metric is measured against.</td>
    </tr>
    <tr>
      <td><code>stable_rank_acoustic_scaling</code></td>
      <td>A bound on how srank grows sub-linearly with model width d_model — the formal version of the acoustic scaling trend in Figure C.</td>
    </tr>
    <tr>
      <td><code>subspace_dilution</code></td>
      <td>Adding more rank dimensions dilutes energy per direction — the geometric counterpart to <code>loraUpdate_frob_decays</code>.</td>
    </tr>
    <tr>
      <td><code>bias_autopsy_separation</code></td>
      <td>Under mild assumptions, the LayerNorm gain subspace and the DPO adapter subspace are nearly orthogonal — the formal version of the 99.97% result.</td>
    </tr>
    <tr>
      <td><code>lower_bound_of_intent</code></td>
      <td>A lower bound on srank for any adapter that reduces DPO loss by a given amount — connecting the geometric floor to the training objective directly.</td>
    </tr>
    <tr>
      <td><code>stable_rank_disentanglement</code></td>
      <td>Energy and geometry fully decouple: you can change ‖ΔW‖_F² freely without changing srank(ΔW), and vice versa.</td>
    </tr>
  </tbody>
</table>

</div>

---

<div class="pub-section" id="lean-full-table">

Full status table
{:.section-label}

## All declarations

<table style="width:100%;border-collapse:collapse;font-size:.88rem;">
  <thead>
    <tr>
      <th style="text-align:left;padding:.4rem .5rem;border-bottom:2px solid var(--border)">Name</th>
      <th style="text-align:left;padding:.4rem .5rem;border-bottom:2px solid var(--border)">Status</th>
      <th style="text-align:left;padding:.4rem .5rem;border-bottom:2px solid var(--border)">Kind</th>
    </tr>
  </thead>
  <tbody id="lean-tbody">
    <tr><td colspan="3" style="padding:.4rem;color:var(--fg-muted)">Loading…</td></tr>
  </tbody>
</table>

<p markdown="0" style="margin-top:1rem;font-family:var(--font-sans);font-size:.82rem;color:var(--fg-muted)">
  <span class="lean-badge proven">proven</span>&nbsp; complete proof, no <code>sorry</code> &nbsp;&nbsp;
  <span class="lean-badge deferred">deferred</span>&nbsp; real statement, proof body is <code>sorry</code> &nbsp;&nbsp;
  <span class="lean-badge stub">stub</span>&nbsp; <code>True := sorry</code> placeholder
</p>

</div>
