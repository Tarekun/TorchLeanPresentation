-- MlpDemo.lean — entry point.  Mirrors pytorch/main.py.
-- Run:  lake exe mlp_demo
import MlpDemo.Training
open NN.API

def main : IO Unit := do
  let task := train.regression (nn.build 0 mkModel)
  train.run task [] (fun {_} _ _ _ _ runner _ => do
    trainModel task runner
    evaluateModel task runner)
