# ternary-svm

Support Vector Machines for ternary feature spaces {-1, 0, +1}, with ternary-aware kernels, SMO training, and one-vs-rest 3-class classification.

## The Problem

You have feature vectors where every dimension is a trit: {-1, 0, +1}. Maybe they came from quantized neural network activations, maybe from a sensor that only reports under/normal/over. You want to classify them.

Standard SVMs work — technically. You can feed ternary vectors to a linear kernel or an RBF kernel with Euclidean distance and get answers. But those kernels don't understand the discrete structure of your data. In Euclidean space, the distance between +1 and 0 is 1.0, and the distance between +1 and -1 is 2.0. That ratio (2:1) doesn't capture the fact that +1 and -1 are *opposites* — qualitatively different from +1 and 0 (which is just "signal present" vs "signal absent").

You need kernels that know the difference between disagreement and opposition.

## The Insight

The ternary dot product `x · y` is a natural kernel because it captures three relationships simultaneously:

| x | y | x·y | Meaning |
|---|---|-----|---------|
| +1 | +1 | +1 | Agreement |
| -1 | -1 | +1 | Agreement (same direction) |
| +1 | -1 | -1 | Opposition |
| ±1 | 0 | 0 | One signal is silent |
| 0 | 0 | 0 | Both silent |

The zeros act as a built-in soft attention mechanism — dimensions where one vector is silent contribute nothing to the similarity score. This is something Euclidean RBF can't replicate.

The ternary distance metric encodes three levels: agreement (d=0), soft disagreement (d=1, e.g., +1 vs 0), and opposition (d=2, i.e., +1 vs -1). Plugging this into an RBF kernel `K(x,y) = exp(-γ · d(x,y))` gives you a similarity function that correctly treats opposition as fundamentally different from mere disagreement.

## How It Works

### Kernel functions

- **Linear**: `K(x, y) = x · y` — the ternary dot product directly.
- **Ternary polynomial**: `K(x, y) = (x · y + c)^d` — raises the dot product to a power, capturing interactions between trit positions.
- **Ternary RBF**: `K(x, y) = exp(-γ · d(x,y))` where d uses the three-level distance. The bandwidth γ controls how quickly similarity decays across the three levels.

### SMO training

The binary SVM implements Platt's Simplified Sequential Minimal Optimization. The full kernel matrix K[i][j] is precomputed once during `fit` (O(n²·d)). Then SMO iterates over pairs of Lagrange multipliers (αᵢ, αⱼ), optimizing each pair analytically while clamping to the box constraint [0, C]. Convergence is measured by consecutive passes with no constraint violations.

The algorithm: pick αᵢ that violates KKT conditions. Pick αⱼ (currently the next index). Compute η = 2K[i][j] - K[i][i] - K[j][j]. Update αⱼ with a constrained step. Update αᵢ to maintain the linear constraint. Update bias b from the KKT complementarity conditions. Repeat.

### One-vs-rest multiclass

`TernarySVM` trains three binary SVMs — one for each class {-1, 0, +1}. Each binary SVM treats its class as +1 and the other two as -1. At prediction time, all three decision functions are evaluated and the class with the highest score wins.

## Code Example

```rust
use ternary_svm::{BinarySVM, TernarySVM, Kernel, trit_distance_f64};

// ── Binary classification with a linear kernel ──
let x = vec![
    vec![1, 1, 1], vec![1, 1, 0], vec![1, 0, 1],    // class +1
    vec![-1, -1, -1], vec![-1, -1, 0], vec![-1, 0, -1], // class -1
];
let y = vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0];

let mut svm = BinarySVM::new(Kernel::Linear, 10.0);
svm.fit(&x, &y, 1000).unwrap();

assert_eq!(svm.predict(&[1, 1, 0]).unwrap(), 1.0);
assert_eq!(svm.predict(&[-1, -1, 0]).unwrap(), -1.0);

// Inspect the learned model
let margin = svm.margin();              // minimum functional margin
let sv_indices = svm.support_vectors(); // which training points are support vectors
let alphas = svm.alphas();              // Lagrange multipliers
let bias = svm.bias();                  // decision boundary offset

// ── Ternary RBF kernel (recommended for non-linear boundaries) ──
let mut rbf_svm = BinarySVM::new(Kernel::TernaryRBF { gamma: 1.0 }, 100.0);
rbf_svm.fit(&x, &y, 1000).unwrap();

// ── Polynomial kernel for interaction features ──
let mut poly_svm = BinarySVM::new(
    Kernel::TernaryPolynomial { degree: 3.0, constant: 1.0 },
    100.0,
);

// ── 3-class classification (one-vs-rest over {-1, 0, +1}) ──
let x3 = vec![
    vec![1, 1, 1], vec![1, 1, 0], vec![1, 0, 1], vec![1, 1, 1],    // class +1
    vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0],    // class  0
    vec![-1, -1, -1], vec![-1, -1, 0], vec![-1, 0, -1], vec![-1, -1, -1], // class -1
];
let y3 = vec![1i8, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1];

let mut ternary = TernarySVM::new(Kernel::TernaryRBF { gamma: 1.0 }, 100.0);
ternary.fit(&x3, &y3, 1000).unwrap();

assert_eq!(ternary.predict(&[1, 1, 1]).unwrap(), 1);
assert_eq!(ternary.predict(&[0, 0, 0]).unwrap(), 0);
assert_eq!(ternary.predict(&[-1, -1, -1]).unwrap(), -1);

// Inspect per-class classifiers
let class_pos_svm = ternary.classifier(1).unwrap();
println!("Support vectors for class +1: {:?}", class_pos_svm.support_vectors());

// ── Ternary distance directly ──
let d = trit_distance_f64(&[1, -1, 0], &[1, -1, 0]); // 0.0 (identical)
let d = trit_distance_f64(&[1, 1, 1], &[-1, -1, -1]); // 6.0 (full opposition)
```

## Module Map

Everything in `src/lib.rs`.

```
Trit                  — type alias for i8 (values -1, 0, 1)
validate_ternary      — check all elements are in {-1, 0, +1}

Kernel                — enum { Linear, TernaryPolynomial{degree,constant}, TernaryRBF{gamma} }
  .apply(a, b)        — kernel function on two ternary vectors

BinarySVM             — single-hyperplane classifier
  .new(kernel, c)     — C is the soft-margin penalty
  .fit(x, y, max_passes) — SMO training, precomputes full kernel matrix
  .predict(x)         — ±1.0
  .decision_function(x) — signed distance to hyperplane
  .margin()           — minimum functional margin over support vectors
  .support_vectors()  — indices of training points with α > 0
  .alphas()           — Lagrange multipliers
  .bias()             — learned offset b

TernarySVM            — one-vs-rest for 3-class {-1, 0, +1}
  .new(kernel, c)
  .fit(x, y, max_passes) — y: i8 in {-1, 0, 1}
  .predict(x)         — i8, class with highest decision function
  .classifier(class)  — access the underlying BinarySVM

dot(a, b)             — ternary dot product (private)
trit_distance_f64(a, b) — three-level distance: 0=agree, 1=soft, 2=oppose
```

## Design Decisions

**Full kernel matrix materialization.** During `fit`, the entire n×n kernel matrix is computed and stored in memory. This makes each SMO iteration O(n) instead of O(n·d), but it means training uses O(n²) memory. For n > 10K, you'll need to add kernel caching or approximate methods. This was a deliberate tradeoff: correctness and clarity over scalability for the initial implementation.

**j selection is sequential, not heuristic.** Platt's original SMO paper recommends selecting j to maximize |Eᵢ - Eⱼ| for faster convergence. This crate uses `(i + 1) % n` instead. Simpler code, same eventual convergence, potentially more SMO passes needed. For the typical use case (small to medium ternary datasets), the difference is negligible.

**`Trit` is `i8`, not an enum.** The `ternary-quantize` crate defines `Trit` as an enum `{Neg, Zero, Pos}`. This crate uses `type Trit = i8`. The two can't be interchanged directly. The `i8` choice makes the math natural (multiply, dot product) at the cost of not catching invalid values at the type level — `validate_ternary()` exists as a runtime check instead.

**No serialization.** Trained models can't be saved to disk. You'd need to manually extract the support vectors, alphas, and bias, and reconstruct the SVM yourself. This is a real gap for production use.

**Labels are `f64` for binary, `i8` for ternary.** The binary SVM takes `y: &[f64]` with values ±1.0. The ternary SVM takes `y: &[i8]` with values {-1, 0, 1}. This inconsistency is because the binary SVM uses `y[i]` as a multiplier in the SMO update (needs to be f64), while the ternary SVM maps class labels to binary targets internally.

## Status

- **12 tests passing.** All three kernels on known values, linearly separable classification, RBF on non-linear data, margin computation, support vector identification, ternary 3-class classification, polynomial classification, decision function sign correctness.
- **Functional for small to medium datasets.** The SMO implementation is correct and converges. It's not optimized for large-scale problems.
- **Known gaps:**
  - O(n²) memory for the kernel matrix — no kernel caching
  - No model serialization or persistence
  - Sequential j selection (not Platt's heuristic)
  - No probability estimates (Platt scaling not implemented)
  - No online/incremental learning — SMO is batch-only
  - `Trit` type (`i8`) doesn't match `ternary-quantize`'s `Trit` enum

## Ecosystem

- [`ternary-quantize`](https://github.com/SuperInstance/ternary-quantize) — produces the ternary features this crate classifies
- [`ternary-optimizer`](https://github.com/SuperInstance/ternary-optimizer) — sign-based training for ternary networks
- [`ternary-em`](https://github.com/SuperInstance/ternary-em) — cluster analysis of ternary distributions

## References

- Platt, J. C. (1998). *Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines*.
- Bernstein, J. et al. (2018). *signSGD: Compressed Optimisation for Non-Convex Problems*. [arXiv:1802.04434](https://arxiv.org/abs/1802.04434)

## License

MIT
