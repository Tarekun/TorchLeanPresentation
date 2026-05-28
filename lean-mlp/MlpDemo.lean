/-
  MlpDemo.lean — entry point.  Ties together Model, Data, and Training.

  Mirrors pytorch/main.py.

  Run:
    lake exe mlp_demo
    lake exe mlp_demo -- --steps 200 --dtype float --backend eager
    lake exe mlp_demo -- --steps 50 --seed 42

  Module layout (parallel to pytorch/):
    MlpDemo/Model.lean    — architecture (σ, τ, mkModel)
    MlpDemo/Data.lean     — synthetic dataset (buildX, buildY, buildDataset)
    MlpDemo/Training.lean — train + eval loop (runExperiment)
    MlpDemo.lean          — CLI entry point (this file)
-/
import MlpDemo.Training   -- transitively brings Model and Data too

open NN.API   -- CLI.*, Common.*, nn.*, train.*, …

def main (args : List String) : IO Unit := do
  -- Drop the `--` separator inserted by `lake exe ... -- <flags>`.
  let args := CLI.dropDashDash args
  -- Parse optional --seed (default 0) and --steps (default 100).
  let (seed, args) ← Common.orThrow "mlp_demo" <| CLI.takeSeed args 0
  let (steps?, args) ← Common.orThrow "mlp_demo" <| CLI.takeNatFlagOnce args "steps"
  let steps : Nat := steps?.getD 100

  -- `train.regression` wraps the model with MSE loss → Task σ τ.
  -- `nn.build seed mkModel` initialises weights deterministically from `seed`.
  let task := train.regression (nn.build seed mkModel)

  IO.println "══════════════════════════════════════════"
  IO.println " MLP Demo  10 → 30 → 1  (regression)"
  IO.println "══════════════════════════════════════════"
  IO.println s!"  seed   = {seed}"
  IO.println s!"  steps  = {steps}"
  IO.println s!"  data   = {nSamples} samples,  y = (Σxᵢ)² + noise"

  -- `train.run` parses --dtype / --backend, picks scalar type α at runtime,
  -- and calls the callback with a compiled Runner.  The four `_` bind the
  -- typeclass instances that `runExperiment` (in Training.lean) requires.
  train.run task args (fun {_} _ _ _ _ runner rest => do
    Common.orThrow "mlp_demo" <| CLI.requireNoArgs rest
    runExperiment (task := task) runner steps)
