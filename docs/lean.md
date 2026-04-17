---
layout: pub
title: "Lean Proofs"
page_id: lean
extra_js:
  - /assets/js/lean.js
---

<div class="pub-section" id="lean-intro">

## Lean 4 Formal Verification

We use Lean 4 + Mathlib (v4.28.0) to formally verify key algebraic properties of LoRA adapter geometry. The load-bearing theorem is `rsLoraUpdate_frob_bounded`: Frobenius² ≤ α²c, rank-invariant under r.

**Status:** <span id="lean-counts">loading…</span>

**Three categories:**
- <span class="lean-badge proven">proven</span> — complete Lean 4 proof, no `sorry`
- <span class="lean-badge deferred">deferred</span> — real mathematical statement; proof body is `sorry` with known proof sketch
- <span class="lean-badge stub">stub</span> — `theorem foo : True := sorry` literature-terminology placeholder

</div>

---

<div class="pub-section" id="lean-table">

## Theorem Status

Source: [`lean/SubspaceOverlap.lean`](https://github.com/d3banjan/lazy-rudder-paper/blob/main/lean/SubspaceOverlap.lean)

<table style="width:100%;border-collapse:collapse;font-size:.9rem;">
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

</div>
