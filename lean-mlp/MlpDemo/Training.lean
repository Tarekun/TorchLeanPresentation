-- MlpDemo/Training.lean — training and evaluation.
-- Mirrors pytorch/training.py.
import MlpDemo.Data
open NN.API

-- ── Hyperparameters ────────────────────────────────────────────────
def nSteps       : Nat   := 1
def learningRate : Float := 0.5
-- ───────────────────────────────────────────────────────────────────

-- α is the runtime scalar type injected by train.run (Float32 by default).
-- The variable block makes it implicit in trainModel and evaluateModel below.
variable {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [Runtime.Scalar α]

def trainModel (task : train.Task σ τ) (runner : train.Runner α task) : IO Unit := do
  let _ ← train.fitDataset (task := task) runner
    (train.steps nSteps (optimizer := optim.adam learningRate) (logEvery := 25))
    generateData

def evaluateModel (task : train.Task σ τ) (runner : train.Runner α task) : IO Unit := do
  train.evalMode (task := task) runner
  train.Report.reportMeanLoss (task := task) runner generateData "eval"
