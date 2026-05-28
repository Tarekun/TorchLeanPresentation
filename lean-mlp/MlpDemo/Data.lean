/-
  MlpDemo/Data.lean — synthetic dataset generation.

  Mirrors pytorch/data.py.

  Produces 50 in-memory (x, y) pairs:
    x  : 10-D vector, entries = sin(i·0.31 + j·0.70) ∈ (−1, 1)  (no IO needed)
    y  = (Σᵢ xᵢ)²  +  ε,   where ε = sin(i·7.13)·√10  (σ ≈ √10 ≈ 3.16)

  True Gaussian sampling requires IO; we use a deterministic sin-based
  surrogate so the whole dataset can be built in pure code.
-/
import MlpDemo.Model   -- brings inDim, outDim, nSamples, σ, τ (and NN transitively)

open Spec     -- Spec.Shape as Shape, Spec.Tensor as Tensor, Spec.get, …
open Tensor   -- Tensor.dim, Tensor.scalar, Tensor.toScalar
open NN.API   -- Data.supervisedDim0F, sample.Supervised, Semantics.Scalar, Runtime.Scalar, …

def nSamples : Nat := 50

/--
Input tensor X : shape (nSamples × inDim).

`Spec.Tensor.dim (fun i => ...)` is the type-safe constructor for a leading
dimension; `i : Fin nSamples` and `j : Fin inDim` are bounds-checked `Fin`
values — out-of-range access is a type error, not a runtime panic.
-/
def buildX : Spec.Tensor Float (.dim nSamples (.dim inDim .scalar)) :=
  Spec.Tensor.dim (fun i =>
    Spec.Tensor.dim (fun j =>
      Spec.Tensor.scalar (Float.sin (Float.ofNat i.val * 0.31 + Float.ofNat j.val * 0.70))))

/-- Sum all `inDim` elements of a single feature vector (used by `buildY`). -/
def rowSum (row : Spec.Tensor Float (.dim inDim .scalar)) : Float :=
  (List.finRange inDim).foldl
    (fun acc idx => acc + Spec.Tensor.toScalar (Spec.get row idx))
    0.0

/--
Target tensor Y : shape (nSamples × 1).

y_i = (Σⱼ x_{i,j})²  +  sin(i·7.13)·√10
-/
def buildY (X : Spec.Tensor Float (.dim nSamples (.dim inDim .scalar))) :
    Spec.Tensor Float (.dim nSamples (.dim outDim .scalar)) :=
  Spec.Tensor.dim (fun i =>
    let row  := Spec.get X i
    let s    := rowSum row
    let yval := s * s + Float.sin (Float.ofNat i.val * 7.13) * Float.sqrt 10.0
    Spec.Tensor.dim (fun (_ : Fin outDim) => Spec.Tensor.scalar yval))

/--
Pack X and Y into a typed `Dataset` — the TorchLean analogue of
`torch.utils.data.TensorDataset(X, Y)`.

`Data.supervisedDim0F` slices along axis 0; each element is a
`sample.Supervised α σ τ` = `TList α [Vec inDim, Vec outDim]`.
The cast from `Float` literals to the runtime scalar `α` happens here,
so the same function works for `Float`, `IEEE32Exec`, and other backends.
-/
def buildDataset {α : Type} [Semantics.Scalar α] [Runtime.Scalar α] :
    _root_.Runtime.Autograd.Train.Dataset (sample.Supervised α σ τ) :=
  Data.supervisedDim0F (α := α) buildX (buildY buildX)
