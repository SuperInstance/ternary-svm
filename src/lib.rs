//! # ternary-svm
//!
//! Support Vector Machine classifiers for ternary feature spaces
//! (elements in {-1, 0, +1}), with ternary-specific kernel functions
//! and simplified SMO-style optimization.

use std::collections::HashMap;

/// A ternary value.
pub type Trit = i8;

/// Validate ternary vector.
pub fn validate_ternary(vec: &[Trit]) -> Result<(), String> {
    for (i, &t) in vec.iter().enumerate() {
        if t != -1 && t != 0 && t != 1 {
            return Err(format!("Invalid trit {} at index {}", t, i));
        }
    }
    Ok(())
}

// ─── Kernel Functions ────────────────────────────────────────────────────────

/// Available kernel types.
#[derive(Debug, Clone)]
pub enum Kernel {
    /// Linear kernel: x·y
    Linear,
    /// Ternary polynomial: (x·y + c)^d
    TernaryPolynomial { degree: f64, constant: f64 },
    /// Ternary RBF: exp(-gamma * trit_distance(x, y))
    TernaryRBF { gamma: f64 },
}

impl Kernel {
    /// Apply the kernel function to two ternary vectors.
    pub fn apply(&self, a: &[Trit], b: &[Trit]) -> f64 {
        match self {
            Kernel::Linear => dot(a, b),
            Kernel::TernaryPolynomial { degree, constant } => {
                (dot(a, b) + constant).powf(*degree)
            }
            Kernel::TernaryRBF { gamma } => {
                (-gamma * trit_distance_f64(a, b)).exp()
            }
        }
    }
}

fn dot(a: &[Trit], b: &[Trit]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (*x as f64) * (*y as f64)).sum()
}

/// Compute ternary distance.
pub fn trit_distance_f64(a: &[Trit], b: &[Trit]) -> f64 {
    let mut dist = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        if x == y {
        } else if (*x == -1 && *y == 1) || (*x == 1 && *y == -1) {
            dist += 2.0;
        } else {
            dist += 1.0;
        }
    }
    dist
}

// ─── Binary SVM ──────────────────────────────────────────────────────────────

/// Binary SVM using simplified SMO (Platt's algorithm).
#[derive(Debug)]
pub struct BinarySVM {
    support_indices: Vec<usize>,
    alphas: Vec<f64>,
    bias: f64,
    train_x: Vec<Vec<Trit>>,
    train_y: Vec<f64>,
    kernel: Kernel,
    c: f64,
    tol: f64,
}

impl BinarySVM {
    pub fn new(kernel: Kernel, c: f64) -> Self {
        Self {
            support_indices: Vec::new(),
            alphas: Vec::new(),
            bias: 0.0,
            train_x: Vec::new(),
            train_y: Vec::new(),
            kernel,
            c,
            tol: 1e-3,
        }
    }

    /// Compute the decision function f(x) = sum_i alpha_i * y_i * K(x_i, x) + b
    fn f(&self, x: &[Trit], alphas: &[f64], bias: f64, train_x: &[Vec<Trit>], train_y: &[f64]) -> f64 {
        let mut sum = bias;
        for i in 0..train_x.len() {
            if alphas[i] > 1e-12 {
                sum += alphas[i] * train_y[i] * self.kernel.apply(&train_x[i], x);
            }
        }
        sum
    }

    /// Train using simplified SMO.
    pub fn fit(&mut self, x: &[Vec<Trit>], y: &[f64], max_passes: usize) -> Result<(), String> {
        for xi in x.iter() {
            validate_ternary(xi)?;
        }
        let n = x.len();
        if n != y.len() {
            return Err("X and y length mismatch".into());
        }
        for &yi in y {
            if yi != 1.0 && yi != -1.0 {
                return Err(format!("Labels must be ±1, got {}", yi));
            }
        }

        let mut alphas = vec![0.0; n];
        let mut bias = 0.0;

        // Precompute kernel matrix
        let mut k = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                k[i][j] = self.kernel.apply(&x[i], &x[j]);
            }
        }

        // Compute E(i) = f(x_i) - y_i using the kernel matrix
        let compute_f = |alphas: &[f64], bias: f64, i: usize, k: &[Vec<f64>], y: &[f64], n: usize| -> f64 {
            let mut s = bias;
            for j in 0..n {
                s += alphas[j] * y[j] * k[j][i];
            }
            s
        };

        let mut passes = 0;
        while passes < max_passes {
            let mut num_changed = 0;
            for i in 0..n {
                let fi = compute_f(&alphas, bias, i, &k, y, n);
                let ei = fi - y[i];

                let ri = y[i] * ei;
                if !((ri < -self.tol && alphas[i] < self.c) || (ri > self.tol && alphas[i] > 0.0)) {
                    continue;
                }

                // Select j randomly (but different from i)
                let j = (i + 1) % n;
                let fj = compute_f(&alphas, bias, j, &k, y, n);
                let ej = fj - y[j];

                // Save old alphas
                let ai_old = alphas[i];
                let aj_old = alphas[j];

                // Compute L and H
                let (l, h) = if y[i] != y[j] {
                    (0.0_f64.max(aj_old - ai_old), self.c.min(self.c + aj_old - ai_old))
                } else {
                    (0.0_f64.max(aj_old + ai_old - self.c), self.c.min(aj_old + ai_old))
                };

                if (h - l).abs() < 1e-10 {
                    continue;
                }

                let eta = 2.0 * k[i][j] - k[i][i] - k[j][j];
                if eta >= 0.0 {
                    continue;
                }

                // Update alpha_j
                alphas[j] = aj_old - y[j] * (ei - ej) / eta;
                alphas[j] = alphas[j].clamp(l, h);

                if (alphas[j] - aj_old).abs() < 1e-5 {
                    continue;
                }

                // Update alpha_i
                alphas[i] = ai_old + y[i] * y[j] * (aj_old - alphas[j]);

                // Update bias
                let b1 = bias - ei
                    - y[i] * (alphas[i] - ai_old) * k[i][i]
                    - y[j] * (alphas[j] - aj_old) * k[i][j];
                let b2 = bias - ej
                    - y[i] * (alphas[i] - ai_old) * k[i][j]
                    - y[j] * (alphas[j] - aj_old) * k[j][j];

                bias = if 0.0 < alphas[i] && alphas[i] < self.c {
                    b1
                } else if 0.0 < alphas[j] && alphas[j] < self.c {
                    b2
                } else {
                    (b1 + b2) / 2.0
                };

                num_changed += 1;
            }

            if num_changed == 0 {
                passes += 1;
            } else {
                passes = 0;
            }
        }

        self.train_x = x.to_vec();
        self.train_y = y.to_vec();
        self.alphas = alphas;
        self.bias = bias;
        self.support_indices = (0..n).filter(|&i| self.alphas[i] > 1e-6).collect();

        Ok(())
    }

    /// Predict class (+1 or -1).
    pub fn predict(&self, x: &[Trit]) -> Result<f64, String> {
        validate_ternary(x)?;
        Ok(if self.decision_function(x) >= 0.0 { 1.0 } else { -1.0 })
    }

    /// Decision function value.
    pub fn decision_function(&self, x: &[Trit]) -> f64 {
        self.f(x, &self.alphas, self.bias, &self.train_x, &self.train_y)
    }

    /// Compute margin.
    pub fn margin(&self) -> f64 {
        if self.support_indices.is_empty() {
            return 0.0;
        }
        let mut min_margin = f64::MAX;
        for &i in &self.support_indices {
            let f_val = self.decision_function(&self.train_x[i]);
            let margin = f_val * self.train_y[i];
            if margin < min_margin {
                min_margin = margin;
            }
        }
        min_margin
    }

    /// Support vector indices.
    pub fn support_vectors(&self) -> &[usize] {
        &self.support_indices
    }

    /// Lagrange multipliers.
    pub fn alphas(&self) -> &[f64] {
        &self.alphas
    }

    /// Bias term.
    pub fn bias(&self) -> f64 {
        self.bias
    }
}

// ─── Ternary SVM (One-vs-Rest) ──────────────────────────────────────────────

/// Ternary SVM: one-vs-rest for labels {-1, 0, +1}.
pub struct TernarySVM {
    classifiers: HashMap<i8, BinarySVM>,
    kernel: Kernel,
    c: f64,
}

impl TernarySVM {
    pub fn new(kernel: Kernel, c: f64) -> Self {
        Self {
            classifiers: HashMap::new(),
            kernel,
            c,
        }
    }

    /// Train one-vs-rest.
    pub fn fit(&mut self, x: &[Vec<Trit>], y: &[i8], max_passes: usize) -> Result<(), String> {
        for &yi in y {
            if yi != -1 && yi != 0 && yi != 1 {
                return Err(format!("Labels must be -1/0/+1, got {}", yi));
            }
        }

        for &cls in &[-1i8, 0, 1] {
            let binary_y: Vec<f64> = y.iter().map(|&yi| if yi == cls { 1.0 } else { -1.0 }).collect();
            let mut svm = BinarySVM::new(self.kernel.clone(), self.c);
            svm.fit(x, &binary_y, max_passes)?;
            self.classifiers.insert(cls, svm);
        }
        Ok(())
    }

    /// Predict class with highest decision function value.
    pub fn predict(&self, x: &[Trit]) -> Result<i8, String> {
        validate_ternary(x)?;
        let mut best_class = 0i8;
        let mut best_score = f64::NEG_INFINITY;
        for &cls in &[-1i8, 0, 1] {
            if let Some(svm) = self.classifiers.get(&cls) {
                let score = svm.decision_function(x);
                if score > best_score {
                    best_score = score;
                    best_class = cls;
                }
            }
        }
        Ok(best_class)
    }

    /// Get binary classifier for a class.
    pub fn classifier(&self, class: i8) -> Option<&BinarySVM> {
        self.classifiers.get(&class)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_kernel() {
        let k = Kernel::Linear;
        assert!((k.apply(&[1, -1, 0], &[1, -1, 1]) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_polynomial_kernel() {
        let k = Kernel::TernaryPolynomial { degree: 2.0, constant: 1.0 };
        // dot=2, (2+1)^2 = 9
        assert!((k.apply(&[1, 1], &[1, 1]) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_rbf_kernel_same() {
        let k = Kernel::TernaryRBF { gamma: 1.0 };
        assert!((k.apply(&[1, -1, 0], &[1, -1, 0]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rbf_kernel_different() {
        let k = Kernel::TernaryRBF { gamma: 0.5 };
        let val = k.apply(&[1, 0, 0], &[-1, 0, 0]);
        assert!((val - (-1.0f64).exp()).abs() < 1e-6);
    }

    #[test]
    fn test_trit_distance() {
        assert_eq!(trit_distance_f64(&[1, -1, 0], &[1, -1, 0]), 0.0);
        assert_eq!(trit_distance_f64(&[1, 1, 1], &[-1, -1, -1]), 6.0);
    }

    #[test]
    fn test_linearly_separable() {
        let x = vec![
            vec![1, 1, 1],
            vec![1, 1, 0],
            vec![1, 0, 1],
            vec![-1, -1, -1],
            vec![-1, -1, 0],
            vec![-1, 0, -1],
        ];
        let y = vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0];

        let mut svm = BinarySVM::new(Kernel::Linear, 100.0);
        svm.fit(&x, &y, 1000).unwrap();

        // Check training data
        for (xi, &yi) in x.iter().zip(y.iter()) {
            let pred = svm.predict(xi).unwrap();
            assert_eq!(pred, yi, "Failed for {:?}", xi);
        }

        // New points
        assert_eq!(svm.predict(&[1, 1, 0]).unwrap(), 1.0);
        assert_eq!(svm.predict(&[-1, -1, 0]).unwrap(), -1.0);
    }

    #[test]
    fn test_margin_computation() {
        let x = vec![vec![1, 0], vec![-1, 0]];
        let y = vec![1.0, -1.0];

        let mut svm = BinarySVM::new(Kernel::Linear, 100.0);
        svm.fit(&x, &y, 1000).unwrap();

        let margin = svm.margin();
        assert!(margin > 0.0, "Margin should be positive, got {}", margin);
    }

    #[test]
    fn test_support_vector_identification() {
        let x = vec![
            vec![1, 1], vec![1, 0], vec![0, 1],
            vec![-1, -1], vec![-1, 0], vec![0, -1],
        ];
        let y = vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0];

        let mut svm = BinarySVM::new(Kernel::Linear, 10.0);
        svm.fit(&x, &y, 1000).unwrap();

        assert!(!svm.support_vectors().is_empty(), "Should have support vectors");
    }

    #[test]
    fn test_rbf_classification() {
        let x = vec![
            vec![1, 0], vec![0, 1],
            vec![-1, 0], vec![0, -1],
        ];
        let y = vec![1.0, 1.0, -1.0, -1.0];

        let mut svm = BinarySVM::new(Kernel::TernaryRBF { gamma: 1.0 }, 100.0);
        svm.fit(&x, &y, 1000).unwrap();

        for (xi, &yi) in x.iter().zip(y.iter()) {
            let pred = svm.predict(xi).unwrap();
            assert_eq!(pred, yi, "RBF SVM failed for {:?}", xi);
        }
    }

    #[test]
    fn test_ternary_classification() {
        let x = vec![
            vec![1, 1, 1], vec![1, 1, 0], vec![1, 0, 1], vec![1, 1, 1],  // class +1
            vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0],  // class 0
            vec![-1, -1, -1], vec![-1, -1, 0], vec![-1, 0, -1], vec![-1, -1, -1], // class -1
        ];
        let y = vec![1i8, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1];

        let mut svm = TernarySVM::new(Kernel::TernaryRBF { gamma: 1.0 }, 100.0);
        svm.fit(&x, &y, 1000).unwrap();

        assert_eq!(svm.predict(&[1, 1, 1]).unwrap(), 1);
        assert_eq!(svm.predict(&[0, 0, 0]).unwrap(), 0);
        assert_eq!(svm.predict(&[-1, -1, -1]).unwrap(), -1);
    }

    #[test]
    fn test_polynomial_classification() {
        let x = vec![vec![1, 1], vec![-1, -1]];
        let y = vec![1.0, -1.0];

        let mut svm = BinarySVM::new(
            Kernel::TernaryPolynomial { degree: 2.0, constant: 1.0 },
            100.0,
        );
        svm.fit(&x, &y, 1000).unwrap();

        assert_eq!(svm.predict(&[1, 1]).unwrap(), 1.0);
        assert_eq!(svm.predict(&[-1, -1]).unwrap(), -1.0);
    }

    #[test]
    fn test_decision_function_sign() {
        let x = vec![vec![1, 0], vec![-1, 0]];
        let y = vec![1.0, -1.0];

        let mut svm = BinarySVM::new(Kernel::Linear, 100.0);
        svm.fit(&x, &y, 1000).unwrap();

        let f_pos = svm.decision_function(&[1, 0]);
        let f_neg = svm.decision_function(&[-1, 0]);
        assert!(f_pos > 0.0, "f_pos = {} should be > 0", f_pos);
        assert!(f_neg < 0.0, "f_neg = {} should be < 0", f_neg);
    }
}
