/-
  MlpDemo/Model.lean — MLP architecture definition.

  Mirrors pytorch/mlp.py.

  TYPE-ERROR DEMO
  ───────────────
  TorchLean encodes layer shapes in Lean's type system.  Change the second
  `30` in `mkModel` (the `nn.linear 30 outDim` line) to any other number,
  e.g. `15`, and save.  The Lean language server marks it immediately:

      "type mismatch … expected Shape.Vec 30, got Shape.Vec 15"

  This fires at elaboration time — before building, before running, before
  any tensor is ever allocated.  The elaborator checks that every consecutive
  pair of layers has matching input/output shapes.

  In Python the equivalent bug (nn.Linear(29, 1) instead of nn.Linear(30, 1))
  raises a RuntimeError only when the first forward pass executes.
-/
import NN

open NN.API   -- nn.*, train.*, …

def inDim  : Nat := 10
def outDim : Nat := 1

-- Input shape: one 10-D feature vector per sample.
abbrev σ : Spec.Shape := NN.Tensor.Shape.Vec inDim

-- Output shape: length-1 vector (TorchLean never uses bare scalars as layer outputs).
abbrev τ : Spec.Shape := NN.Tensor.Shape.Vec outDim

-- MLP with 1 hidden layer of 30 neurons that uses ReLU activation function
def mkModel : nn.M (nn.Sequential σ τ) :=
  nn.sequential![
    nn.linear  inDim  30      Spec.Shape.scalar,
    -- nn.linear  inDim  25      Spec.Shape.scalar,
    nn.relu,
    nn.linear  30     outDim  Spec.Shape.scalar
  ]
