# ternary-svm

Support vector machines for ternary feature spaces {-1, 0, +1} — with ternary-aware kernels, SMO training, and one-vs-rest 3-class classification.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Why this exists

SVMs are geometric classifiers — they find the hyperplane that maximizes the margin between classes. Standard kernels (RBF with Euclidean distance, polynomial with dot products) assume continuous features. When your data is ternary, those kernels miss the discrete structure: the distance between +1 and −1 isn't 2.0 like Euclidean says, it's an *opposition* — a qualitatively different relationship than +1 vs 0.

This crate provides kernels that respect the three-level ternary distance metric directly, yielding tighter margins and better generalization on quantized data.

## The key insight

The ternary dot product is a natural kernel. When two trit vectors are aligned (+1×+1 = +1, −1×−1 = +1), the dot product is high. When they oppose (+1×−1 = −1), it's negative. When one is silent (×0 = 0), it contributes nothing — a built-in soft attention mechanism. The RBF kernel on ternary distance amplifies this: `K(x,y) = exp(−γ · d(x,y))` where d costs 0 for agreement, 1 for soft disagreement, and 2 for opposition.

## Quick Start

```rust
use ternary_svm::{BinarySVM, TernarySVM, Kernel};

// ── Binary classification with linear kernel ──
let x = vec![
    vec![1, 1, 1], vec![1, 1, 0],     // class +1
    vec![-1, -1, -1], vec![-1, -1, 0], // class -1
];
let y = vec![1.0, 1.0, -1.0, -1.0];

let mut svm = BinarySVM::new(Kernel::Linear, 10.0);
svm.fit(&x, &y, 1000).unwrap();

let pred = svm.predict(&[1, 1, 0]).unwrap(); // → 1.0
let decision = svm.decision_function(&[1, 0, 0]); // signed distance to hyperplane
let margin = svm.margin();         // minimum functional margin
let sv = svm.support_vectors();    // indices of support vectors

// ── Ternary RBF kernel (recommended for non-linear ternary data) ──
let mut rbf_svm = BinarySVM::new(Kernel::TernaryRBF { gamma: 1.0 }, 10.0);
rbf_svm.fit(&x, &y, 1000).unwrap();

// ── Polynomial kernel ──
let mut poly_svm = BinarySVM::new(
    Kernel::TernaryPolynomial { degree: 3.0, constant: 1.0 }, 10.0,
);

// ── 3-class classification (one-vs-rest) ──
let x = vec![
    vec![1, 1, 1], vec![1, 1, 0], vec![1, 0, 1], vec![1, 1, 1],   // class +1
    vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0],   // class 0
    vec![-1, -1, -1], vec![-1, -1, 0], vec![-1, 0, -1], vec![-1, -1, -1], // class -1
];
let y = vec![1i8, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1];

let mut ternary = TernarySVM::new(Kernel::TernaryRBF { gamma: 1.0 }, 100.0);
ternary.fit(&x, &y, 1000).unwrap();

let class = ternary.predict(&[1, 1, 1]).unwrap(); // → 1
```

## Architecture

```
                    ┌──────────────────────────┐
  Training data ──→ │   Kernel precomputation   │
  (ternary vecs)    │   K[i][j] for all pairs   │
                    └──────────┬────────────────┘
                               ▼
                    ┌──────────────────────────┐
                    │   Simplified SMO          │
                    │   (Platt's algorithm)     │
                    │   Optimize α₁, α₂ pairs  │
                    │   until KKT satisfied     │
                    └──────────┬────────────────┘
                               ▼
               ┌───────────────┴───────────────┐
               ▼                               ▼
    ┌──────────────────┐            ┌──────────────────┐
    │  BinarySVM       │            │  TernarySVM       │
    │  Single hyperplane│           │  3 × BinarySVM   │
    │  predict: ±1     │            │  One-vs-rest      │
    │  margin, SVs     │            │  predict: {-1,0,1}│
    └──────────────────┘            └──────────────────┘
```

## Kernel Functions

### Linear: `K(x, y) = x · y`

The ternary dot product naturally captures alignment: matching signs contribute +1, opposing signs −1, zeros are neutral. No normalization needed — the output is bounded by the dimension.

### Polynomial: `K(x, y) = (x · y + c)^d`

Amplifies the ternary dot product. Higher degrees capture interactions between trit positions — useful when the decision boundary depends on *combinations* of features, not individual ones.

### Ternary RBF: `K(x, y) = exp(−γ · d(x, y))`

The recommended kernel for non-linearly separable ternary data. Uses the three-level ternary distance directly: agreement (d=0) gives K=1, opposition (d=2) gives K=exp(−2γ). The bandwidth γ controls how quickly similarity decays.

## API Reference

### Kernel

```rust
pub enum Kernel {
    Linear,
    TernaryPolynomial { degree: f64, constant: f64 },
    TernaryRBF { gamma: f64 },
}
// kernel.apply(a: &[Trit], b: &[Trit]) -> f64
```

### BinarySVM

```rust
let mut svm = BinarySVM::new(kernel: Kernel, c: f64);
svm.fit(x: &[Vec<Trit>], y: &[f64], max_passes: usize) -> Result<(), String>;
svm.predict(x: &[Trit]) -> Result<f64, String>;           // → ±1.0
svm.decision_function(x: &[Trit]) -> f64;                  // signed distance
svm.margin() -> f64;                                        // min functional margin
svm.support_vectors() -> &[usize];                          // SV indices
svm.alphas() -> &[f64];                                    // Lagrange multipliers
svm.bias() -> f64;                                         // bias term b
```

### TernarySVM (one-vs-rest)

```rust
let mut svm = TernarySVM::new(kernel: Kernel, c: f64);
svm.fit(x: &[Vec<Trit>], y: &[i8], max_passes: usize) -> Result<(), String>;
svm.predict(x: &[Trit]) -> Result<i8, String>;             // → {-1, 0, 1}
svm.classifier(class: i8) -> Option<&BinarySVM>;           // inspect per-class SVM
```

### Utilities

```rust
fn trit_distance_f64(a: &[Trit], b: &[Trit]) -> f64;  // ternary distance
fn validate_ternary(vec: &[Trit]) -> Result<(), String>;
```

## Real-world example

A data center monitors server health as ternary vectors: each dimension is CPU load (−1: underutilized, 0: normal, +1: overloaded), memory pressure, disk IO, network congestion — 16 features per server. You want to classify servers into three states: healthy (0), degraded (+1), critical (−1).

With 1000 labeled servers and the TernaryRBF kernel (γ=0.5), the SVM finds non-linear decision boundaries that separate the three classes. The ternary RBF kernel outperforms a standard RBF because it treats the +1-to-−1 gap as fundamentally different from the +1-to-0 gap — overloaded vs underutilized is a harder signal mismatch than overloaded vs normal.

## Ecosystem connections

- **[`ternary-quantize`](https://github.com/SuperInstance/ternary-quantize)** — produces the ternary features this crate classifies
- **[`ternary-knn`](https://github.com/SuperInstance/ternary-knn)** — non-parametric alternative (no training, slower inference)
- **[`ternary-hmm`](https://github.com/SuperInstance/ternary-hmm)** — for temporal sequences where SVMs ignore ordering
- **[`ternary-transformer`](https://github.com/SuperInstance/ternary-transformer)** — produces ternary embeddings for downstream classification

## Performance

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Kernel precomputation | O(n²·d) | Done once during `fit` |
| SMO iteration | O(n²) per pass | n = training samples |
| `predict` | O(|SV|·d) | Only support vectors matter |
| `decision_function` | O(|SV|·d) | Same as predict without sign |

The full kernel matrix is materialized during training. For n > 10K, consider SMO with kernel caching or approximate methods.

## Open questions

- **Kernel selection**: Is there a theoretically optimal γ for ternary RBF, or must it always be cross-validated?
- **Multiclass beyond 3**: One-vs-rest with 3 classes is natural for ternary labels. For more classes, would an error-correcting output code scheme work better?
- **Online learning**: SMO is batch-only. Can ternary SVMs support incremental updates when new labeled vectors arrive?
- **Sparse ternary kernels**: When >50% of trits are zero, can we skip zero positions in the dot product for a 2× speedup?

## Testing

```bash
cargo test
```

10 tests: all 3 kernels (linear, polynomial, RBF) on known values, linearly separable classification, margin computation, support vector identification, RBF on non-linear data, ternary 3-class classification, decision function sign correctness.

## License

MIT
