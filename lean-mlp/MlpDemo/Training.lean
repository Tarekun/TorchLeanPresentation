/-
  MlpDemo/Training.lean — training and evaluation logic.

  Mirrors pytorch/training.py.

  `runExperiment` is the Lean equivalent of `train()` + `evaluate()`:
  it takes a live `Runner` (TorchLean's stateful wrapper around the compiled
  model), runs Adam for `steps` updates on the in-memory dataset, then
  evaluates MSE loss in eval mode.
-/
import MlpDemo.Data   -- brings buildDataset, nSamples (and Model transitively)

open NN.API   -- train.*, optim.*, sample.*, …

/--
Train for `steps` Adam steps, then report MSE loss before and after.

`train.Runner α task` holds the compiled train/eval predictors, the loss,
and a mutable mode flag.  The four typeclass constraints match exactly what
`train.run` injects into its callback (see MlpDemo.lean):
  - `Semantics.Scalar α` : scalar arithmetic semantics for `α`
  - `DecidableEq Spec.Shape` : needed internally by the Runner
  - `ToString α` : for loss logging
  - `Runtime.Scalar α` : concrete runtime representation of `α`
-/
def runExperiment {α : Type}
    [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [Runtime.Scalar α]
    (task : train.Task σ τ) (runner : train.Runner α task) (steps : Nat) : IO Unit := do
  let dataset := buildDataset (α := α)
  IO.println s!"  dataset size = {dataset.size}"
  IO.println ""
  -- Baseline MSE before any gradient updates.
  train.Report.reportMeanLoss (task := task) runner dataset "before"
  -- Adam with lr=0.01, log every 25 steps.
  let trainCfg := train.steps steps (optimizer := optim.adam 0.01) (logEvery := 25)
  let _report ← train.fitDataset (task := task) runner trainCfg dataset
  -- Switch to eval mode (disables dropout etc.) then report final MSE.
  train.evalMode (task := task) runner
  train.Report.reportMeanLoss (task := task) runner dataset "after"
