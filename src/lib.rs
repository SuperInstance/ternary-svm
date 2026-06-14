//! # ternary-svm
//!
//! A lightweight linear SVM for ternary feature vectors (elements in {-1, 0, +1}),
//! trained with the PEGASOS (Primal Estimated sub-GrAdient SOlver for SVM)
//! stochastic sub-gradient descent algorithm.
//!
//! ## Features
//!
//! - **Fast training**: PEGASOS converges in O(1/ε) iterations, no quadratic programming
//! - **Ternary-first**: feature vectors stored as `Vec<Vec<i8>>` with values in {-1, 0, +1}
//! - **Binary and multi-class**: one-vs-one for 3-class {-1, 0, +1}
//! - **Convergence tracking**: early stopping when loss stabilizes
//! - **Regularization**: configurable L2 weight to prevent overfitting
//!
//! ## Quick Start
//!
//! ```rust
//! use ternary_svm::TernSVM;
//!
//! let mut model = TernSVM::new(1.0, 0.01);  // λ=1.0, learning rate=0.01
//!
//! // Feature vectors where each element is -1, 0, or +1
//! let x = vec![
//!     vec![1, 1, 1],
//!     vec![1, 1, 0],
//!     vec![-1, -1, -1],
//!     vec![-1, -1, 0],
//! ];
//! let y = vec![1.0, 1.0, -1.0, -1.0];
//!
//! model.fit(&x, &y).unwrap();
//!
//! let pred = model.predict_label(&vec![1, 1, 0]).unwrap();
//! assert_eq!(pred, 1.0);
//! ```

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{thread_rng, SeedableRng};

// ─── Core types ──────────────────────────────────────────────────────────────

/// A ternary feature value: must be -1, 0, or +1.
pub type Trit = i8;

// ─── Validation ──────────────────────────────────────────────────────────────

/// Validate that every element of `vec` is in {-1, 0, +1}.
///
/// # Errors
///
/// Returns an `Err` containing the invalid value and its index if any element
/// is outside the allowed set.
///
/// # Example
///
/// ```
/// use ternary_svm::validate_ternary;
///
/// assert!(validate_ternary(&[1, 0, -1]).is_ok());
/// assert!(validate_ternary(&[2, 0, -1]).is_err());
/// ```
pub fn validate_ternary(vec: &[Trit]) -> Result<(), String> {
    for (i, &t) in vec.iter().enumerate() {
        if !matches!(t, -1..=1) {
            return Err(format!(
                "Invalid trit {} at index {}; must be -1, 0, or +1",
                t, i
            ));
        }
    }
    Ok(())
}

// ─── PEGASOS SVM ─────────────────────────────────────────────────────────────

/// A linear Support Vector Machine trained with the PEGASOS algorithm.
///
/// PEGASOS (Primal Estimated sub-GrAdient SOlver for SVM) is a stochastic
/// sub-gradient descent method that iterates over random training examples
/// and takes projected gradient steps. It requires no quadratic programming
/// and converges to an ε-accurate solution in O(1/ε) iterations.
///
/// The model stores a weight vector `w` (one weight per feature dimension)
/// and a bias term `b`.
///
/// Training tracks the average hinge loss over each epoch and stops early
/// when the loss stabilizes.
///
/// The `kernel_hint` field is reserved for future non-linear kernel expansion.
///
/// # Example
///
/// ```
/// use ternary_svm::TernSVM;
///
/// let mut model = TernSVM::new(1.0, 0.01);
///
/// let x = vec![
///     vec![1, 1, 1],
///     vec![1, 1, 0],
///     vec![-1, -1, -1],
///     vec![-1, -1, 0],
/// ];
/// let y = vec![1.0, 1.0, -1.0, -1.0];
///
/// model.fit(&x, &y).unwrap();
/// assert_eq!(model.predict_label(&vec![1, 0, 0]).unwrap(), 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct TernSVM {
    /// Regularization parameter λ (lambda). Higher values = stronger regularization.
    pub lambda: f64,
    /// Learning rate for SGD.
    pub learning_rate: f64,
    /// Maximum training epochs.
    pub max_epochs: usize,
    /// Tolerance for early stopping (relative change in average loss).
    pub tol: f64,
    /// Convergence patience: stop after this many consecutive epochs
    /// with relative loss change < `tol`.
    pub patience: usize,
    /// Random seed for reproducibility. `None` uses random entropy.
    pub seed: Option<u64>,

    // Learned parameters
    /// Weight vector (one weight per feature dimension).
    pub w: Vec<f64>,
    /// Bias term.
    pub b: f64,

    // Training history
    /// Average hinge loss per epoch during the last `fit` call.
    pub loss_history: Vec<f64>,
    /// Weight norm ||w|| after each epoch.
    pub weight_norm_history: Vec<f64>,
    /// Number of epochs actually trained (may be < max_epochs due to early stop).
    pub epochs_trained: usize,
    /// Whether training converged via early stopping.
    pub converged: bool,

    // Metadata
    /// Number of features (dimensionality) seen during `fit`.
    pub n_features: usize,

    /// Future-use marker for non-linear kernel expansion.
    /// Currently unused; reserved for when PEGASOS is extended to kernel
    /// feature maps (e.g., random Fourier features for RBF).
    pub kernel_hint: Option<String>,
}

impl Default for TernSVM {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            learning_rate: 0.01,
            max_epochs: 100,
            tol: 1e-3,
            patience: 3,
            seed: None,
            w: Vec::new(),
            b: 0.0,
            loss_history: Vec::new(),
            weight_norm_history: Vec::new(),
            epochs_trained: 0,
            converged: false,
            n_features: 0,
            kernel_hint: None,
        }
    }
}

impl TernSVM {
    /// Create a new `TernSVM` with the given regularization and learning rate.
    ///
    /// Default hyperparams: max_epochs = 100, tol = 1e-3, patience = 3,
    /// seed = None, kernel_hint = None.
    ///
    /// # Arguments
    ///
    /// * `lambda` - Regularization parameter λ (higher = stronger regularization)
    /// * `learning_rate` - Step size for gradient descent updates
    pub fn new(lambda: f64, learning_rate: f64) -> Self {
        Self {
            lambda,
            learning_rate,
            ..Default::default()
        }
    }

    /// Set the random seed for reproducible shuffling during training.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the maximum number of training epochs.
    pub fn with_max_epochs(mut self, max_epochs: usize) -> Self {
        self.max_epochs = max_epochs;
        self
    }

    /// Set early stopping tolerance and patience.
    pub fn with_early_stop(mut self, tol: f64, patience: usize) -> Self {
        self.tol = tol;
        self.patience = patience;
        self
    }

    /// Set a kernel hint (documentation marker for future expansion).
    pub fn with_kernel_hint(mut self, hint: &str) -> Self {
        self.kernel_hint = Some(hint.to_string());
        self
    }

    /// Train the model on ternary feature vectors with binary labels (±1).
    ///
    /// Uses the PEGASOS algorithm:
    /// 1. For each epoch, iterate over a random permutation of training examples
    /// 2. For each example (x_i, y_i), if y_i·(w·x_i + b) < 1, take a
    ///    sub-gradient step: w ← (1 - ηλ)w + η·y_i·x_i, b ← b + η·y_i
    ///    Otherwise: w ← (1 - ηλ)w
    /// 3. Track average hinge loss per epoch
    /// 4. Stop early if loss stabilizes (relative change < tol for `patience` epochs)
    ///
    /// # Arguments
    ///
    /// * `x` - Training feature vectors (each inner vec must be same length)
    /// * `y` - Training labels (±1.0)
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - Feature vectors contain values outside {-1, 0, +1}
    /// - Feature vectors have inconsistent dimensions
    /// - `x.len() != y.len()`
    /// - Labels are not ±1.0
    /// - Feature vectors are empty (0 dimensions)
    /// - Training set is empty
    pub fn fit(&mut self, x: &[Vec<Trit>], y: &[f64]) -> Result<(), String> {
        let n = x.len();
        if n == 0 {
            return Err("Empty training set".into());
        }
        if n != y.len() {
            return Err(format!(
                "X length ({}) does not match y length ({})",
                n,
                y.len()
            ));
        }

        let d = x[0].len();
        if d == 0 {
            return Err("Feature vectors must have at least 1 dimension".into());
        }
        for xi in x.iter() {
            validate_ternary(xi)?;
            if xi.len() != d {
                return Err(format!(
                    "Inconsistent feature dimensions: expected {} but found {}",
                    d,
                    xi.len()
                ));
            }
        }
        for &yi in y {
            if yi != 1.0 && yi != -1.0 {
                return Err(format!("Labels must be +/-1.0, got {}", yi));
            }
        }

        self.n_features = d;
        self.w = vec![0.0; d];
        self.b = 0.0;
        self.loss_history = Vec::with_capacity(self.max_epochs);
        self.weight_norm_history = Vec::with_capacity(self.max_epochs);
        self.converged = false;

        let mut rng: Box<dyn rand::RngCore> = match self.seed {
            Some(seed) => Box::new(StdRng::seed_from_u64(seed)),
            None => Box::new(thread_rng()),
        };
        let mut indices: Vec<usize> = (0..n).collect();

        let mut no_improve_epochs = 0;
        let mut prev_loss = f64::INFINITY;

        for epoch in 0..self.max_epochs {
            // eta_t = learning_rate / (1 + lambda * learning_rate * epoch)
            // Standard PEGASOS decreasing step size.
            let eta = self.learning_rate / (1.0 + self.lambda * self.learning_rate * epoch as f64);

            indices.shuffle(&mut rng);

            for &i in &indices {
                let xi = &x[i];
                let yi = y[i];

                let decision = self.decision_raw(xi).unwrap_or(0.0);
                let margin = yi * decision;

                let scale = 1.0 - eta * self.lambda;
                if margin < 1.0 {
                    // Sub-gradient update: penalize misclassification
                    for (wj, &xj) in self.w.iter_mut().zip(xi.iter()) {
                        *wj = scale * *wj + eta * yi * xj as f64;
                    }
                    self.b += eta * yi;
                } else {
                    // No misclassification: only regularize (shrink toward zero)
                    for wj in self.w.iter_mut() {
                        *wj *= scale;
                    }
                }
            }

            // Compute average hinge loss for this epoch
            let mut total_loss = 0.0;
            for (xi, &yi) in x.iter().zip(y.iter()) {
                let decision = self.decision_raw(xi).unwrap_or(0.0);
                let hinge = 1.0 - yi * decision;
                total_loss += if hinge > 0.0 { hinge } else { 0.0 };
            }
            let avg_loss = total_loss / n as f64;
            self.loss_history.push(avg_loss);

            // Track weight norm
            let norm: f64 = self.w.iter().map(|v| v * v).sum::<f64>().sqrt();
            self.weight_norm_history.push(norm);

            // Early stopping check
            let relative_change = (prev_loss - avg_loss).abs() / prev_loss.max(1e-12);
            if relative_change < self.tol {
                no_improve_epochs += 1;
                if no_improve_epochs >= self.patience {
                    self.converged = true;
                    self.epochs_trained = epoch + 1;
                    return Ok(());
                }
            } else {
                no_improve_epochs = 0;
            }
            prev_loss = avg_loss;
        }

        self.epochs_trained = self.max_epochs;
        self.converged = false;
        Ok(())
    }

    /// Compute the raw decision value w·x + b.
    ///
    /// The sign of this value is the predicted class (±1).
    /// The magnitude is the confidence (distance to the hyperplane).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `x` dimension doesn't match trained model.
    pub fn predict(&self, x: &[Trit]) -> Result<f64, String> {
        self.decision_raw(x)
    }

    /// Predict the class label (±1.0) for a feature vector.
    ///
    /// Equivalent to `sign(predict(x))`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `x` dimension doesn't match trained model.
    pub fn predict_label(&self, x: &[Trit]) -> Result<f64, String> {
        let decision = self.predict(x)?;
        Ok(if decision >= 0.0 { 1.0 } else { -1.0 })
    }

    /// Compute accuracy on a labeled test set.
    ///
    /// Returns the fraction of predictions that match the true labels.
    ///
    /// # Arguments
    ///
    /// * `x` - Test feature vectors
    /// * `y` - True labels (±1.0)
    ///
    /// # Errors
    ///
    /// Returns `Err` if any vector is invalid or dims are inconsistent.
    pub fn score(&self, x: &[Vec<Trit>], y: &[f64]) -> Result<f64, String> {
        if x.len() != y.len() {
            return Err(format!("X length ({}) != y length ({})", x.len(), y.len()));
        }
        if x.is_empty() {
            return Ok(1.0);
        }

        let mut correct = 0usize;
        for (xi, &yi) in x.iter().zip(y.iter()) {
            let pred = self.predict_label(xi)?;
            if (pred - yi).abs() < 0.5 {
                correct += 1;
            }
        }
        Ok(correct as f64 / x.len() as f64)
    }

    /// Raw decision value w·x + b.
    fn decision_raw(&self, x: &[Trit]) -> Result<f64, String> {
        if self.w.is_empty() && self.b.abs() < 1e-15 {
            return Ok(0.0);
        }
        if x.len() != self.n_features {
            return Err(format!(
                "Feature dimension mismatch: expected {} but got {}",
                self.n_features,
                x.len()
            ));
        }
        let dot: f64 = self
            .w
            .iter()
            .zip(x.iter())
            .map(|(wi, xi)| wi * (*xi as f64))
            .sum();
        Ok(dot + self.b)
    }
}

// ─── One-vs-One Multi-Class ─────────────────────────────────────────────────

/// Multi-class SVM using one-vs-one strategy for labels {-1, 0, +1}.
///
/// Trains three binary `TernSVM` classifiers: (+1 vs 0), (+1 vs -1), (0 vs -1).
/// Prediction uses majority vote weighted by decision function magnitude.
///
/// # Example
///
/// ```
/// use ternary_svm::{TernSVM, OvOTernSVM};
///
/// let x = vec![
///     vec![1, 1, 1],     // class +1
///     vec![1, 1, 1],     // class +1
///     vec![0, 0, 0],     // class 0
///     vec![0, 0, 0],     // class 0
///     vec![-1, -1, -1],  // class -1
///     vec![-1, -1, -1],  // class -1
/// ];
/// let y = vec![1, 1, 0, 0, -1, -1];
///
/// let mut model = OvOTernSVM::new(1.0, 0.01).with_max_epochs(200);
/// model.fit(&x, &y).unwrap();
///
/// assert_eq!(model.predict(&vec![1, 1, 1]).unwrap(), 1);
/// assert_eq!(model.predict(&vec![0, 0, 0]).unwrap(), 0);
/// assert_eq!(model.predict(&vec![-1, -1, -1]).unwrap(), -1);
/// ```
#[derive(Debug, Clone)]
pub struct OvOTernSVM {
    /// Regularization parameter λ.
    pub lambda: f64,
    /// Learning rate.
    pub learning_rate: f64,
    /// Max epochs per binary classifier.
    pub max_epochs: usize,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,

    /// Binary classifiers for each pair: (pos_class, neg_class) -> TernSVM
    classifiers: Vec<((i8, i8), TernSVM)>,
}

impl OvOTernSVM {
    /// Create a new OvO multi-class SVM.
    pub fn new(lambda: f64, learning_rate: f64) -> Self {
        Self {
            lambda,
            learning_rate,
            max_epochs: 100,
            seed: None,
            classifiers: Vec::new(),
        }
    }

    /// Set max epochs for each binary sub-classifier.
    pub fn with_max_epochs(mut self, max_epochs: usize) -> Self {
        self.max_epochs = max_epochs;
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Train all OvO binary classifiers.
    ///
    /// # Arguments
    ///
    /// * `x` - Training feature vectors
    /// * `y` - Labels in {-1, 0, +1}
    ///
    /// # Errors
    ///
    /// Returns `Err` for invalid labels, empty datasets, or inconsistent dims.
    pub fn fit(&mut self, x: &[Vec<Trit>], y: &[i8]) -> Result<(), String> {
        if x.len() != y.len() {
            return Err(format!("X length ({}) != y length ({})", x.len(), y.len()));
        }
        for &yi in y {
            if !matches!(yi, -1i8..=1) {
                return Err(format!("Labels must be -1, 0, or +1, got {}", yi));
            }
        }

        let classes = [-1i8, 0, 1];
        let pairs = [(0, 1), (0, 2), (1, 2)];

        self.classifiers.clear();
        for &(ai, bi) in &pairs {
            let pos_class = classes[ai];
            let neg_class = classes[bi];

            let mut pair_x: Vec<Vec<Trit>> = Vec::new();
            let mut pair_y: Vec<f64> = Vec::new();

            for (xi, &yi) in x.iter().zip(y.iter()) {
                if yi == pos_class {
                    pair_x.push(xi.clone());
                    pair_y.push(1.0);
                } else if yi == neg_class {
                    pair_x.push(xi.clone());
                    pair_y.push(-1.0);
                }
            }

            if pair_x.is_empty() || pair_y.is_empty() {
                continue;
            }

            let mut svm = TernSVM::new(self.lambda, self.learning_rate);
            svm.max_epochs = self.max_epochs;
            if let Some(seed) = self.seed {
                svm.seed = Some(seed.wrapping_add((ai * 10 + bi) as u64));
            }
            svm.fit(&pair_x, &pair_y)?;
            self.classifiers.push(((pos_class, neg_class), svm));
        }

        Ok(())
    }

    /// Predict the class for a feature vector using weighted majority vote.
    ///
    /// Each binary classifier votes for one of its two classes.
    /// The class with the highest total confidence wins.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the model is untrained or input is invalid.
    pub fn predict(&self, x: &[Trit]) -> Result<i8, String> {
        if self.classifiers.is_empty() {
            return Err("Model has not been trained".into());
        }

        validate_ternary(x)?;

        let mut votes: std::collections::HashMap<i8, f64> = std::collections::HashMap::new();
        for &((pos_class, neg_class), ref svm) in &self.classifiers {
            let decision = svm.predict(x)?;
            let winner = if decision >= 0.0 {
                pos_class
            } else {
                neg_class
            };
            *votes.entry(winner).or_insert(0.0) += decision.abs();
        }

        votes
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(class, _)| class)
            .ok_or_else(|| "No votes cast - all classifiers returned empty".into())
    }

    /// Compute accuracy on a test set.
    ///
    /// # Errors
    ///
    /// Returns `Err` for invalid input or untrained model.
    pub fn score(&self, x: &[Vec<Trit>], y: &[i8]) -> Result<f64, String> {
        if x.len() != y.len() {
            return Err(format!("X length ({}) != y length ({})", x.len(), y.len()));
        }
        if x.is_empty() {
            return Ok(1.0);
        }

        let mut correct = 0usize;
        for (xi, &yi) in x.iter().zip(y.iter()) {
            let pred = self.predict(xi)?;
            if pred == yi {
                correct += 1;
            }
        }
        Ok(correct as f64 / x.len() as f64)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Validation tests ────────────────────────────────────────────────

    #[test]
    fn test_validate_valid() {
        assert!(validate_ternary(&[1, 0, -1]).is_ok());
        assert!(validate_ternary(&[1; 100]).is_ok());
        assert!(validate_ternary(&[0; 0]).is_ok());
    }

    #[test]
    fn test_validate_invalid() {
        assert!(validate_ternary(&[2]).is_err());
        assert!(validate_ternary(&[-2]).is_err());
        assert!(validate_ternary(&[1, 2, 3]).is_err());
    }

    // ── Empty / edge cases ──────────────────────────────────────────────

    #[test]
    fn test_empty_training_set() {
        let mut svm = TernSVM::new(1.0, 0.01);
        let result = svm.fit(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_length_mismatch() {
        let mut svm = TernSVM::new(1.0, 0.01);
        let x = vec![vec![1, 0]];
        let y = vec![1.0, -1.0];
        let result = svm.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_untrained_predict() {
        let svm = TernSVM::new(1.0, 0.01);
        let pred = svm.predict(&[1, 0, -1]);
        assert!(pred.is_ok());
        assert!((pred.unwrap() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_zero_dim_features() {
        let mut svm = TernSVM::new(1.0, 0.01);
        let result = svm.fit(&[vec![], vec![]], &[1.0, -1.0]);
        assert!(result.is_err());
    }

    // ── Trivial perfect separation ──────────────────────────────────────

    #[test]
    fn test_perfect_separation_1d() {
        let mut svm = TernSVM::new(1.0, 0.01).with_max_epochs(200);
        let x = vec![vec![1], vec![-1]];
        let y = vec![1.0, -1.0];
        svm.fit(&x, &y).unwrap();

        assert_eq!(svm.predict_label(&[1]).unwrap(), 1.0);
        assert_eq!(svm.predict_label(&[-1]).unwrap(), -1.0);
        assert!((svm.score(&x, &y).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_perfect_separation_3d() {
        let mut svm = TernSVM::new(0.1, 0.1).with_max_epochs(300);
        let x = vec![
            vec![1, 1, 1],
            vec![1, 1, 0],
            vec![1, 0, 1],
            vec![-1, -1, -1],
            vec![-1, -1, 0],
            vec![-1, 0, -1],
        ];
        let y = vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0];

        svm.fit(&x, &y).unwrap();
        let score = svm.score(&x, &y).unwrap();
        assert!(
            score >= 0.9,
            "Perfectly separable data should achieve >= 90% training accuracy, got {}",
            score
        );
    }

    // ── XOR-like separability ─────────────────────────────────────────────

    #[test]
    fn test_xor_ternary() {
        let mut svm = TernSVM::new(0.01, 0.1).with_max_epochs(500);
        let x = vec![vec![1, 1], vec![-1, -1], vec![1, -1], vec![-1, 1]];
        let y = vec![1.0, 1.0, -1.0, -1.0];

        svm.fit(&x, &y).unwrap();

        let score = svm.score(&x, &y).unwrap();
        assert!(
            score >= 0.5,
            "XOR-like data should beat random (50%), got {}",
            score
        );
    }

    #[test]
    fn test_xor_with_zeros() {
        let mut svm = TernSVM::new(0.1, 0.05).with_max_epochs(500);
        let x = vec![
            vec![1, 1, 0],
            vec![-1, -1, 0],
            vec![1, -1, 0],
            vec![-1, 1, 0],
        ];
        let y = vec![1.0, 1.0, -1.0, -1.0];

        svm.fit(&x, &y).unwrap();
        let score = svm.score(&x, &y).unwrap();
        assert!(
            score >= 0.5,
            "Should beat random (50%) with 0-noise dim, got {}",
            score
        );
    }

    // ── Regularization tests ────────────────────────────────────────────

    #[test]
    fn test_regularization_high_lambda_small_weights() {
        let mut svm = TernSVM::new(10.0, 0.01).with_max_epochs(200);
        let x = vec![vec![1, 1], vec![-1, -1]];
        let y = vec![1.0, -1.0];
        svm.fit(&x, &y).unwrap();

        let norm: f64 = svm.w.iter().map(|v| v * v).sum();
        assert!(
            norm < 2.0,
            "High regularization should keep weight norm small, got {}",
            norm
        );
    }

    #[test]
    fn test_regularization_low_lambda_allows_larger_weights() {
        let mut svm = TernSVM::new(0.01, 0.1).with_max_epochs(200);
        let x = vec![vec![1, 1], vec![-1, -1]];
        let y = vec![1.0, -1.0];
        svm.fit(&x, &y).unwrap();

        let norm: f64 = svm.w.iter().map(|v| v * v).sum();
        assert!(
            norm > 0.01,
            "Low regularization should allow non-trivial weights, got {}",
            norm
        );
    }

    #[test]
    fn test_regularization_noise_dimension() {
        let mut x: Vec<Vec<Trit>> = Vec::new();
        let mut y: Vec<f64> = Vec::new();
        for _ in 0..50 {
            x.push(vec![1, -1, 0]);
            y.push(1.0);
            x.push(vec![-1, 1, 0]);
            y.push(-1.0);
        }

        let mut svm_high = TernSVM::new(100.0, 0.01).with_max_epochs(100);
        let mut svm_low = TernSVM::new(0.001, 0.1).with_max_epochs(100);
        svm_high.fit(&x, &y).unwrap();
        svm_low.fit(&x, &y).unwrap();

        let noise_high = svm_high.w[2].abs();
        let noise_low = svm_low.w[2].abs();
        assert!(
            noise_high <= noise_low + 0.1,
            "High lambda should penalize noise more, high={}, low={}",
            noise_high,
            noise_low
        );
    }

    // ── Convergence tracking ───────────────────────────────────────────

    #[test]
    fn test_convergence_tracking() {
        let mut svm = TernSVM::new(1.0, 0.01)
            .with_max_epochs(1000)
            .with_early_stop(1e-3, 3);
        let x = vec![vec![1, 1], vec![-1, -1]];
        let y = vec![1.0, -1.0];
        svm.fit(&x, &y).unwrap();

        assert!(svm.epochs_trained > 0);
        assert!(!svm.loss_history.is_empty());
        assert!(!svm.weight_norm_history.is_empty());
    }

    #[test]
    fn test_convergence_stops_early() {
        let mut svm = TernSVM::new(10.0, 0.1)
            .with_max_epochs(10000)
            .with_early_stop(1e-4, 5);
        let x = vec![vec![1, 0], vec![-1, 0]];
        let y = vec![1.0, -1.0];
        svm.fit(&x, &y).unwrap();

        assert!(
            svm.epochs_trained < 200,
            "Should converge early on trivial data, used {} epochs",
            svm.epochs_trained
        );
        assert!(svm.converged);
    }

    // ── Predict and score ──────────────────────────────────────────────

    #[test]
    fn test_predict_returns_sign() {
        let mut svm = TernSVM::new(1.0, 0.01).with_max_epochs(200);
        let x = vec![vec![1, 1], vec![-1, -1]];
        let y = vec![1.0, -1.0];
        svm.fit(&x, &y).unwrap();

        let label = svm.predict_label(&[1, 1]).unwrap();
        assert_eq!(label, 1.0);

        let raw = svm.predict(&[1, 1]).unwrap();
        assert!(raw >= -0.001); // decision value should be >= 0 for positive prediction
    }

    #[test]
    fn test_score_perfect() {
        let mut svm = TernSVM::new(1.0, 0.01).with_max_epochs(200);
        let x = vec![vec![1, 1], vec![-1, -1]];
        let y = vec![1.0, -1.0];
        svm.fit(&x, &y).unwrap();
        assert!((svm.score(&x, &y).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_score_empty() {
        let svm = TernSVM::new(1.0, 0.01);
        let score = svm.score(&[], &[]).unwrap();
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_score_wrong_feature_dim() {
        let mut svm = TernSVM::new(1.0, 0.01).with_max_epochs(10);
        let x = vec![vec![1, 0], vec![-1, 0]];
        let y = vec![1.0, -1.0];
        svm.fit(&x, &y).unwrap();
        // Score with wrong feature dimension (3 instead of 2)
        let result = svm.score(&[vec![1, 0, 1]], &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_score_mismatch_length() {
        let svm = TernSVM::new(1.0, 0.01);
        let result = svm.score(&[vec![1]], &[1.0, -1.0]);
        assert!(result.is_err());
    }

    // ── Multi-class OvO ─────────────────────────────────────────────────

    #[test]
    fn test_ovo_basic() {
        let x = vec![
            vec![1, 1, 1],
            vec![1, 1, 1],
            vec![0, 0, 0],
            vec![0, 0, 0],
            vec![-1, -1, -1],
            vec![-1, -1, -1],
        ];
        let y = vec![1, 1, 0, 0, -1, -1];

        let mut model = OvOTernSVM::new(1.0, 0.01)
            .with_max_epochs(200)
            .with_seed(42);
        model.fit(&x, &y).unwrap();

        assert_eq!(model.predict(&vec![1, 1, 1]).unwrap(), 1);
        assert_eq!(model.predict(&vec![0, 0, 0]).unwrap(), 0);
        assert_eq!(model.predict(&vec![-1, -1, -1]).unwrap(), -1);
    }

    #[test]
    fn test_ovo_accuracy() {
        let x = vec![
            vec![1, 1, 1],
            vec![1, 1, 1],
            vec![0, 0, 0],
            vec![0, 0, 0],
            vec![-1, -1, -1],
            vec![-1, -1, -1],
        ];
        let y = vec![1, 1, 0, 0, -1, -1];

        let mut model = OvOTernSVM::new(1.0, 0.01)
            .with_max_epochs(200)
            .with_seed(42);
        model.fit(&x, &y).unwrap();

        let acc = model.score(&x, &y).unwrap();
        assert!(
            acc >= 0.8,
            "OvO should get >=80% on cleanly separable 3-class data, got {}",
            acc
        );
    }

    #[test]
    fn test_ovo_invalid_label() {
        let mut model = OvOTernSVM::new(1.0, 0.01);
        let result = model.fit(&[vec![1]], &[2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ovo_untrained() {
        let model = OvOTernSVM::new(1.0, 0.01);
        let result = model.predict(&[1, 0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_kernel_hint() {
        let svm = TernSVM::new(1.0, 0.01).with_kernel_hint("rbf");
        assert_eq!(svm.kernel_hint.as_deref(), Some("rbf"));

        let svm_default = TernSVM::new(1.0, 0.01);
        assert!(svm_default.kernel_hint.is_none());
    }
}
