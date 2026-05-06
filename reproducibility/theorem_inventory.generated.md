# Theorem Inventory Snapshot — lazy-rudder

Generated: 2026-05-06
Root commit: `0f9fdc9`
Source: `status/theorem-status.yaml` + `status/papers/lazy-rudder.yaml`

Do not edit here. Regenerate from root repo:

```bash
uv run python scripts/export_paper_snapshot.py --paper lazy-rudder
```

## Theorems

- `global_plus_delta_error_bound` — **proved**
  - G_4 error bound: approximation error of global+delta scheme ≤ global error + delta error (triangle inequality). Closed via SubspaceOverlap bridge. (2026-04-26)
- `subspace_project_complement_le_norm` — **partial** (1)
  - orthogonal projection residual ‖x - P_U x‖² ≤ ‖x‖²; Pythagoras argument; statement clean; proof sorry (2026-04-25)
- `stableRank_implies_low_dim_approximation` — **partial** (1)
  - Eckart-Young existence: stableRank W ≤ k ⇒ ∃ rank-k approx with bounded Frobenius error; needs SVD theory; sorry (2026-04-25)
