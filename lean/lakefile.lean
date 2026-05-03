import Lake
open Lake DSL

package «lazy-rudder» where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4"
    @ "8f9d9cff6bd728b17a24e163c9402775d9e6a365"

@[default_target]
lean_lib «LeanMining» where
  roots := #[`LeanMining.NeuralGeometry.SubspaceOverlap]
