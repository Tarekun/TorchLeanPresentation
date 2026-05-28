import Lake
open Lake DSL

package «lean-mlp» where
  version := v!"0.1.0"
  description := "Minimal MLP demo using TorchLean — showcase of compile-time shape checking."
  leanOptions := #[⟨`autoImplicit, false⟩]

-- Point at the local TorchLean clone one directory above.
-- Lake will resolve TorchLean's own transitive dependencies (mathlib, etc.) automatically.
require TorchLean from "../TorchLean"

-- Library that registers MlpDemo.* so submodule imports resolve correctly.
lean_lib MlpDemo where

-- The runnable executable.  `lake exe mlp_demo [flags...]` builds and runs it.
lean_exe mlp_demo where
  root := `MlpDemo
