-- MlpDemo/Data.lean — synthetic dataset generation.
-- Mirrors pytorch/data.py.
import MlpDemo.Model
open Spec Tensor NN.API

-- x_{i,j} = sin(i·0.31 + j·0.70)  ∈ (−1, 1)
private def buildX (n : Nat) : Spec.Tensor Float (.dim n (.dim inDim .scalar)) :=
  Spec.Tensor.dim fun i => Spec.Tensor.dim fun j =>
    Spec.Tensor.scalar (Float.sin (Float.ofNat i.val * 0.31 + Float.ofNat j.val * 0.70))

private def rowSum (row : Spec.Tensor Float (.dim inDim .scalar)) : Float :=
  (List.finRange inDim).foldl (fun s idx => s + Spec.Tensor.toScalar (Spec.get row idx)) 0.0

-- y_i = (Σⱼ x_{i,j})²  +  sin(i·7.13)·√10   (pseudo-noise, σ ≈ √10)
private def buildY (n : Nat) (X : Spec.Tensor Float (.dim n (.dim inDim .scalar))) :
    Spec.Tensor Float (.dim n (.dim outDim .scalar)) :=
  Spec.Tensor.dim fun i =>
    let s := rowSum (Spec.get X i)
    Spec.Tensor.dim fun _ => Spec.Tensor.scalar
      (s * s + Float.sin (Float.ofNat i.val * 7.13) * Float.sqrt 10.0)

-- α is the runtime scalar type (resolved by train.run — Float32 by default).
variable {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]

-- Returns n (x, y) pairs.  Mirrors generate_batch() in pytorch/data.py.
def generateData (n : Nat) : _root_.Runtime.Autograd.Train.Dataset (sample.Supervised α σ τ) :=
  Data.supervisedDim0F (α := α) (buildX n) (buildY n (buildX n))
