# ternary-svm

**Support Vector Machines for Ternary Feature Spaces**

An SVM implementation designed for ternary data — vectors whose elements live in {-1, 0, +1}. Provides binary SVM classification with ternary-specific kernels, plus a one-vs-rest ternary SVM for 3-class problems.

---

## Why Ternary SVM?

Ternary representations appear in balanced ternary computing, quantized neural networks (where weights are constrained to {-1, 0, +1}), and certain post-quantum cryptographic schemes. Standard SVM kernels treat features as continuous values, missing the discrete structure of ternary space.

This crate provides:

- **Ternary-aware kernels** that respect the {-1, 0, +1} geometry
- **Simplified SMO optimization** (Platt's algorithm) for training
- **Binary SVM** for two-class problems with ternary features
- **Ternary SVM** using one-vs-rest for 3-class classification
- **Margin computation** and **support vector identification**

---

## Kernel Functions

### Linear Kernel
```
K(x, y) = x · y = Σ x_i · y_i
```
Standard dot product. Since trits are -1, 0, or +1, the dot product naturally captures alignment: matching signs contribute positively, opposing signs negatively, zeros are neutral.

### Ternary Polynomial Kernel
```
K(x, y) = (x · y + c)^d
```
Amplifies the ternary dot product through polynomial expansion. Higher degrees capture more complex interactions between trit positions.

### Ternary RBF Kernel
```
K(x, y) = exp(-γ · d(x, y))
```
Where `d(x, y)` is the ternary distance (0 for same, 1 for zero-vs-nonzero, 2 for opposite). This is the recommended kernel for non-linearly separable ternary data — it directly uses the ternary distance metric rather than Euclidean distance.

---

## Quick Start

```rust
use ternary_svm::{BinarySVM, TernarySVM, Kernel};

// --- Binary classification ---
let x = vec![
    vec![1, 1, 1],
    vec![1, 1, 0],
    vec![-1, -1, -1],
    vec![-1, -1, 0],
];
let y = vec![1.0, 1.0, -1.0, -1.0];

let mut svm = BinarySVM::new(Kernel::Linear, 10.0);
svm.fit(&x, &y, 1000).unwrap();

let pred = svm.predict(&[1, 1, 0]).unwrap(); // → 1.0
let decision = svm.decision_function(&[1, 0, 0]); // signed distance to hyperplane
let margin = svm.margin(); // minimum functional margin
let sv = svm.support_vectors(); // indices of support vectors

// --- RBF kernel ---
let mut rbf_svm = BinarySVM::new(Kernel::TernaryRBF { gamma: 1.0 }, 10.0);
rbf_svm.fit(&x, &y, 1000).unwrap();

// --- Polynomial kernel ---
let mut poly_svm = BinarySVM::new(
    Kernel::TernaryPolynomial { degree: 3.0, constant: 1.0 },
    10.0,
);
poly_svm.fit(&x, &y, 1000).unwrap();

// --- Ternary 3-class classification ---
let x = vec![
    vec![1, 1, 1], vec![1, 1, 0], vec![1, 0, 1], vec![1, 1, 1],
    vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0],
    vec![-1, -1, -1], vec![-1, -1, 0], vec![-1, 0, -1], vec![-1, -1, -1],
];
let y = vec![1i8, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1];

let mut ternary = TernarySVM::new(Kernel::TernaryRBF { gamma: 1.0 }, 100.0);
ternary.fit(&x, &y, 1000).unwrap();

let class = ternary.predict(&[1, 1, 1]).unwrap(); // → 1
```

---

## Algorithm Details

### Simplified SMO (Sequential Minimal Optimization)

Training uses Platt's simplified SMO algorithm:

1. **Initialization**: All Lagrange multipliers α = 0
2. **Outer loop**: Iterate over training examples, checking KKT conditions
3. **Inner loop**: For each violating example, select a partner and jointly optimize their α values
4. **Convergence**: When no α changes for `max_passes` consecutive iterations

The algorithm pre-computes the full kernel matrix for efficiency on small-to-medium datasets.

### One-vs-Rest for Ternary Classification

For 3-class problems (labels -1, 0, +1):
- Train 3 binary SVMs: one per class
- Each SVM distinguishes "this class" (label +1) vs "all others" (label -1)
- Prediction: choose the class whose SVM outputs the highest decision function value

### Margin Computation

The functional margin for a support vector `(x_i, y_i)` is:
```
y_i · f(x_i) = y_i · (Σ α_j y_j K(x_j, x_i) + b)
```
The overall margin is the minimum over all support vectors.

---

## Research Applications

- **Ternary weight networks**: classify patterns in quantized neural network weights
- **Balanced ternary ALU design**: decision boundaries for ternary arithmetic operations
- **Cryptanalysis**: distinguish ternary sequences from random using SVM classifiers
- **Signal processing**: classify ternary-valued sensor readings
- **Natural language processing**: sentiment analysis with ternary features (negative/neutral/positive)

---

## API Reference

| Item | Description |
|---|---|
| `Kernel` | Enum: Linear, TernaryPolynomial, TernaryRBF |
| `BinarySVM` | Two-class SVM with SMO training |
| `TernarySVM` | Three-class SVM via one-vs-rest |
| `trit_distance_f64` | Ternary distance function |
| `validate_ternary` | Validate {-1, 0, +1} elements |

---

## License

MIT
