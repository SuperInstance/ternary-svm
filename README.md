# ternary-svm

**Lightweight Support Vector Machine for ternary feature spaces {-1, 0, +1}**

Trained with the PEGASOS algorithm — stochastic sub-gradient descent that converges in O(1/ε) iterations with no quadratic programming.

## The Problem

Feature vectors where every dimension is a trit {-1, 0, +1}. Maybe they came from quantized neural network activations, maybe from a sensor that only reports under/normal/over. You want to classify them — fast.

Standard SVMs require quadratic programming (QPs) for training. PEGASOS replaces that with pure SGD: iterate over random examples, take gradient steps, regularize. No kernel matrix to store, no QP solver to call.

## Quick Start

```rust
use ternary_svm::TernSVM;

// ── Binary classification with PEGASOS ──
let x = vec![
    vec![1, 1, 1], vec![1, 1, 0],
    vec![-1, -1, -1], vec![-1, -1, 0],
];
let y = vec![1.0, 1.0, -1.0, -1.0];

let mut model = TernSVM::new(1.0, 0.01);
model.fit(&x, &y).unwrap();

println!("Prediction: {}", model.predict_label(&[1, 0, 0]).unwrap());
println!("Weights: {:?}", model.w);
println!("Bias: {}", model.b);
println!("Converged: {} after {} epochs", model.converged, model.epochs_trained);
println!("Loss history: {:?}", model.loss_history);

// ── Multi-class (one-vs-one) for {-1, 0, +1} ──
use ternary_svm::OvOTernSVM;

let x3 = vec![
    vec![1, 1, 1], vec![1, 1, 0],  // class +1
    vec![0, 0, 0], vec![0, 0, 0],  // class 0
    vec![-1, -1, -1], vec![-1, -1, 0], // class -1
];
let y3 = vec![1, 1, 0, 0, -1, -1];

let mut ovo = OvOTernSVM::new(1.0, 0.01).with_max_epochs(200);
ovo.fit(&x3, &y3).unwrap();

assert_eq!(ovo.predict(&[1, 1, 1]).unwrap(), 1);
assert_eq!(ovo.predict(&[0, 0, 0]).unwrap(), 0);
```

## CLI Training

```bash
# Binary classification
cargo run --features cli --bin train-cli -- \
  --train data/train.csv \
  --test data/test.csv \
  --lambda 1.0 \
  --learning-rate 0.01 \
  --epochs 200 \
  --output weights.json

# Multi-class classification (-1, 0, +1 labels)
cargo run --features cli --bin train-cli -- \
  --multiclass \
  --train data/train_3class.csv \
  --test data/test_3class.csv
```

CSV format (no header, one sample per row):
```
feat_1,feat_2,...,feat_d,label
1,0,-1,...,1,1.0
-1,1,0,...,-1,-1.0
```

## Architecture

### PEGASOS Training

The core algorithm (Shalev-Shwartz et al., 2011):

```
Initialize w = 0, b = 0
For epoch = 1..T:
  η_t = η₀ / (1 + λ·η₀·t)
  Shuffle training data
  For each (x_i, y_i):
    If y_i·(w·x_i + b) < 1:    // margin violation
      w ← (1 - η_t·λ)·w + η_t·y_i·x_i
      b ← b + η_t·y_i
    Else:
      w ← (1 - η_t·λ)·w        // only regularize
```

Key properties:
- **O(d·n·T)** time complexity (d = features, n = samples, T = epochs)
- **O(d)** memory — no kernel matrix
- **O(1/ε)** iteration bound for ε-accurate solution
- **Decreasing step size** η_t / (1 + λ·η₀·t) ensures convergence

### Early Stopping

Training tracks the average hinge loss each epoch. If the relative change in loss stays below `tol` for `patience` consecutive epochs, training stops. This prevents overfitting and unnecessary computation.

### One-vs-One Multi-Class

For {-1, 0, +1} labels, `OvOTernSVM` trains three binary SVMs:
- +1 vs 0
- +1 vs -1
- 0 vs -1

Prediction uses weighted majority vote: each classifier casts a vote for its winner, weighted by the absolute decision value.

## API Reference

### `TernSVM` — Binary SVM

| Method | Description |
|--------|-------------|
| `new(lambda, learning_rate)` | Create with regularization and step size |
| `fit(x, y)` | PEGASOS training, returns `Result` |
| `predict(x)` | Raw decision value w·x + b |
| `predict_label(x)` | ±1.0 class prediction |
| `score(x, y)` | Accuracy on test set |
| `with_seed(seed)` | Reproducible shuffling |
| `with_max_epochs(n)` | Override max epochs |
| `with_early_stop(tol, patience)` | Configure convergence |
| `with_kernel_hint(s)` | Future-use marker |

**Fields:** `w`, `b`, `loss_history`, `weight_norm_history`, `epochs_trained`, `converged`, `n_features`, `kernel_hint`

### `OvOTernSVM` — Multi-Class (one-vs-one)

| Method | Description |
|--------|-------------|
| `new(lambda, learning_rate)` | Create |
| `fit(x, y)` | Train 3 binary classifiers |
| `predict(x)` | i8 class {-1, 0, +1} |
| `score(x, y)` | Accuracy |
| `with_max_epochs(n)` | Override epochs per sub-classifier |
| `with_seed(s)` | Reproducible |

### `validate_ternary(vec)` — Input validation

Returns `Ok(())` if all elements are {-1, 0, +1}, `Err` otherwise.

## Dependencies

- `rand` — training shuffling
- `csv`, `clap`, `serde`, `serde_json` — CLI features only (`features = ["cli"]`)

## Testing

```bash
# All tests
cargo test

# With doc tests
cargo test --doc

# CLI tests
cargo test --features cli
```

The test suite covers:
- **Validation:** valid values, invalid values, empty vectors
- **Edge cases:** empty training set, length mismatch, untrained model, 0-dim features
- **Perfect separation:** 1D and 3D linearly separable data
- **XOR-like:** linear SVM beats chance on 2D XOR (n=4)
- **Regularization:** high λ shrinks weights, low λ allows larger weights
- **Noise dimension:** high λ penalizes irrelevant features more
- **Convergence:** tracking and early stopping
- **Score:** accuracy computation, empty sets, error conditions
- **Multi-class:** OvO training, accuracy, invalid labels, untrained error
- **Kernel hint:** doc marker field

## Design Decisions

**PEGASOS over SMO.** The previous version used Platt's SMO with a full kernel matrix (O(n²) memory). PEGASOS uses O(d) memory and O(d·n·T) time. For ternary data with many samples but few features, this is dramatically more scalable.

**i8 feature vectors.** Values are raw i8, not enums. This makes dot products trivial (`wi * xi as f64`) at the cost of runtime validation. The `validate_ternary()` function checks inputs at `fit()` time.

**Decreasing step size.** The standard PEGASOS schedule η_t = η₀ / (1 + λ·η₀·t) avoids manual tuning of the step count.

**No serialization (yet).** Weights w and bias b are public fields — extract them manually or use the CLI's `--output` flag for JSON export.

## Status

- **25 tests passing**
- **Binary and multi-class classification**
- **Convergence tracking with early stopping**
- **Regularization via L2 penalty**
- **CLI for CSV training and JSON weight export**

## References

- Shalev-Shwartz, S., Singer, Y., Srebro, N., & Cotter, A. (2011). *PEGASOS: Primal Estimated sub-GrAdient SOlver for SVM*. Mathematical Programming, 127(1), 3–30.
- Platt, J. C. (1998). *Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines*.

## License

MIT
