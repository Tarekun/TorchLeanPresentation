/-
  MlpDemo.lean — minimal TorchLean MLP from scratch.

  Network:  Linear(10 → 30) → ReLU → Linear(30 → 1)
  Task:     Regression  y = (Σᵢ xᵢ)² + noise
  Training: Adam, ~100 steps, in-memory synthetic dataset

  Run:
    lake exe mlp_demo
    lake exe mlp_demo -- --steps 200 --dtype float --backend eager
    lake exe mlp_demo -- --steps 50 --seed 42

  TYPE-ERROR DEMO
  ──────────────
  TorchLean encodes layer shapes in Lean's type system.  To trigger a
  compile-time error before running anything, open this file and change
  the second `30` inside `mkModel` below to a different number, e.g.:

      nn.linear 15 outDim ...   -- was 30

  Lean will immediately report:
    "type mismatch … expected Shape.Vec 30, got Shape.Vec 15"

  No runtime, no inference — pure elaboration-time rejection.
-/

-- `import NN` pulls in the curated library (via NN.Library → NN.Entrypoint.API).
-- `import NN.API.Models.Mlp` is a separately-gated module; it is NOT re-exported
-- by `import NN` and must be imported explicitly when needed.
import NN
import NN.API.Models.Mlp

-- `open Spec` brings Spec.Shape (as Shape), Spec.Tensor (as Tensor), Spec.get, …
-- We deliberately do NOT also `open NN.Tensor` because NN.Tensor.Shape is just
-- `abbrev Shape := Spec.Shape`.  Opening both names would make `Shape` ambiguous.
open Spec      -- Shape, Tensor, get, toScalar, …
open Tensor    -- Tensor.dim, Tensor.scalar  (Spec.Tensor opened via Spec)
open NN.API    -- nn.*, train.*, optim.*, Data.*, sample.*, CLI.*, Common.*, …


-- ─────────────────────────────────────────────────────────────
-- 1.  Architecture: shapes and model
-- ─────────────────────────────────────────────────────────────

/-- Number of input features per sample. -/
def inDim  : Nat := 10

/-- Number of neurons in the single hidden layer. -/
def hidDim : Nat := 30

/-- Single scalar regression output per sample. -/
def outDim : Nat := 1

/--
Input shape: one 10-D feature vector per sample.

`Shape.Vec n` is syntactic sugar for `Spec.Shape.dim n Spec.Shape.scalar`.
TorchLean carries this in the *type* of the forward function, so feeding
data of the wrong shape is a compile-time error.
-/
abbrev σ : Spec.Shape := NN.Tensor.Shape.Vec inDim   -- Vec 10

/--
Output shape: one scalar per sample, wrapped in a length-1 vector
(TorchLean always uses vector shapes for outputs, never bare scalars).
-/
abbrev τ : Spec.Shape := NN.Tensor.Shape.Vec outDim  -- Vec 1

/--
The one-hidden-layer MLP:  Linear(10→30) → ReLU → Linear(30→1).

`nn.M (nn.Sequential σ τ)` is the type of a *model builder* that will
produce a typed `Sequential` from σ to τ.  The elaborator checks at
compile time that the intermediate shapes align — here both linear layers
share the `30`-wide hidden dimension.

── TYPE-ERROR DEMO ──────────────────────────────────────────────────────
Change the second `30` (second `nn.linear`) to any other number, e.g.:

    nn.linear 15 outDim (pfx := Spec.Shape.scalar)

Lean elaborates `nn.sequential!` left-to-right and immediately finds
that the first layer's output shape `Vec 30` does not unify with the
second layer's expected input shape `Vec 15`.  Error fires before any C
code is generated, before any tensor is allocated, before any run.
─────────────────────────────────────────────────────────────────────────
-/
def mkModel : nn.M (nn.Sequential σ τ) :=
  nn.sequential![
    nn.linear inDim   30      (pfx := Spec.Shape.scalar),  -- 10 → 30  (hidden in)
    nn.relu,
    nn.linear 30      outDim  (pfx := Spec.Shape.scalar)   -- 30 → 1   (hidden out)
  ]


-- ─────────────────────────────────────────────────────────────
-- 2.  Synthetic dataset  y = (Σᵢ xᵢ)² + ε
-- ─────────────────────────────────────────────────────────────

/-- Total number of training samples (fixed at compile time for shape safety). -/
def nSamples : Nat := 50

/--
Build input tensor X of shape (nSamples × inDim).

Each entry is purely deterministic — no IO required:
  x_{i,j} = sin(i · 0.31 + j · 0.70)  ∈ (−1, 1)

`Spec.Tensor.dim (fun i => ...)` is the type-safe constructor for a leading
batch dimension.  Both `i : Fin nSamples` and `j : Fin inDim` are Lean
`Fin` values — access is bounds-checked by the type system.
-/
def buildX : Spec.Tensor Float (.dim nSamples (.dim inDim .scalar)) :=
  Spec.Tensor.dim (fun i =>
    Spec.Tensor.dim (fun j =>
      Spec.Tensor.scalar (Float.sin (Float.ofNat i.val * 0.31 + Float.ofNat j.val * 0.70))))

/--
Sum all `inDim` elements of a single feature vector.

`List.finRange inDim` = `[⟨0,_⟩, ⟨1,_⟩, …, ⟨9,_⟩]` — bounds-checked indices.
`Spec.get row idx` fetches element `idx` as a `.scalar` tensor; `toScalar` unwraps it.
-/
def rowSum (row : Spec.Tensor Float (.dim inDim .scalar)) : Float :=
  (List.finRange inDim).foldl
    (fun acc idx => acc + Spec.Tensor.toScalar (Spec.get row idx))
    0.0

/--
Build target tensor Y of shape (nSamples × 1).

For each sample i:
  y_i = (Σⱼ x_{i,j})²  +  sin(i · 7.13) · √10

`sin(i · 7.13) · √10` is a deterministic surrogate for Gaussian noise
with standard deviation ≈ √10 ≈ 3.16 (as requested).  True Gaussian
sampling requires IO, which we avoid to keep the demo self-contained.
-/
def buildY (X : Spec.Tensor Float (.dim nSamples (.dim inDim .scalar))) :
    Spec.Tensor Float (.dim nSamples (.dim outDim .scalar)) :=
  Spec.Tensor.dim (fun i =>
    let row  := Spec.get X i
    let s    := rowSum row
    -- quadratic target + pseudo-noise (σ ≈ √10)
    let yval := s * s + Float.sin (Float.ofNat i.val * 7.13) * Float.sqrt 10.0
    Spec.Tensor.dim (fun (_ : Fin outDim) => Spec.Tensor.scalar yval))

/--
Pack X and Y into a typed supervised `Dataset` — TorchLean's `TensorDataset(X, Y)`.

`Data.supervisedDim0F` slices along axis 0 and pairs rows: each element of the
resulting dataset is a `TList α [Vec inDim, Vec outDim]` (one (x, y) pair).
The cast from Float literals to the chosen runtime scalar `α` happens here, so
the same function works for `Float`, `IEEE32Exec`, and other backends.
-/
def buildDataset {α : Type} [Semantics.Scalar α] [Runtime.Scalar α] :
    _root_.Runtime.Autograd.Train.Dataset (sample.Supervised α σ τ) :=
  Data.supervisedDim0F (α := α) buildX (buildY buildX)


-- ─────────────────────────────────────────────────────────────
-- 3.  Training entry point
-- ─────────────────────────────────────────────────────────────

/--
CLI entry point.

`train.run` parses `--dtype` / `--backend` flags, instantiates the compiled
module at the chosen scalar type `α`, and calls our callback.

Inside the callback we:
  1. Print baseline MSE loss on the full 50-sample in-memory dataset.
  2. Train with Adam (lr = 0.01) for `steps` updates, logging every 25.
  3. Switch to eval mode, then print final MSE loss.

Note on namespace resolution: after `open NN.API`, `NN.API.CLI` resolves
to `CLI` and `NN.API.Common` resolves to `Common`.  The examples bundled
inside TorchLean write `API.CLI.*` because they live inside `namespace NN`,
where `API` refers to `NN.API`.  This standalone file has no enclosing
namespace so we use the shorter form.
-/
def main (args : List String) : IO Unit := do
  -- Drop the `--` separator inserted by `lake exe ... -- <flags>`.
  let args := CLI.dropDashDash args

  -- Parse optional flags (default to seed=0, steps=100).
  let (seed, args) ← Common.orThrow "mlp_demo" <| CLI.takeSeed args 0
  let (steps?, args) ← Common.orThrow "mlp_demo" <| CLI.takeNatFlagOnce args "steps"
  let steps : Nat := steps?.getD 100

  -- `train.regression` wraps the compiled model with an MSE loss function,
  -- producing a `SeqTask σ τ` that `train.run` can dispatch.
  let task := train.regression (nn.build seed mkModel)

  IO.println "══════════════════════════════════════════"
  IO.println " MLP Demo  10 → 30 → 1  (regression)"
  IO.println "══════════════════════════════════════════"
  IO.println s!"  seed   = {seed}"
  IO.println s!"  steps  = {steps}"
  IO.println s!"  data   = {nSamples} samples,  y = (Σxᵢ)² + noise"

  -- `train.run` dispatches on --dtype/--backend flags.
  -- Pass the remaining `args` (after --seed and --steps are stripped above).
  -- The four `_` in the callback are inferred typeclass args for α.
  train.run task args (fun {α} _ _ _ _ runner rest => do
    Common.orThrow "mlp_demo" <| CLI.requireNoArgs rest

    let dataset := buildDataset (α := α)
    IO.println s!"  dataset size = {dataset.size}"
    IO.println ""

    -- Baseline loss before any gradient updates.
    train.Report.reportMeanLoss (task := task) runner dataset "before"

    -- Run Adam for `steps` updates; print loss every 25 steps.
    let trainCfg := train.steps steps (optimizer := optim.adam 0.01) (logEvery := 25)
    let _report ← train.fitDataset (task := task) runner trainCfg dataset

    -- Eval mode then final loss.
    train.evalMode (task := task) runner
    train.Report.reportMeanLoss (task := task) runner dataset "after"

    pure ())
