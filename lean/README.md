# Lean source

Vendored copy of `LeanMining/NeuralGeometry/SubspaceOverlap.lean` from the
private `lean-mining` monorepo. Frozen here so the public paper repository
is self-contained: any reader can `lake build` and verify the load-bearing
theorems without access to the upstream monorepo.

**Authoritative source:** `LeanMining/NeuralGeometry/SubspaceOverlap.lean`
in the parent monorepo. This vendored copy is overwritten by
`make sync-lean` (in this directory's grandparent) on every drift-cure run,
so it cannot diverge from canonical without the next sync surfacing the diff.

**Toolchain:** Lean 4 v4.28.0, Mathlib pinned to commit
`8f9d9cff6bd728b17a24e163c9402775d9e6a365` (matches monorepo).

**Build:**

```bash
cd lean/
lake update
lake build
```

First build pulls Mathlib (~5 min and ~3 GB). Subsequent builds are
incremental.
