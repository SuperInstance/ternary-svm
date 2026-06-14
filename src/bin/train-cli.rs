//! # train-cli — Ternary SVM training CLI
//!
//! Read a CSV of ternary features, train a PEGASOS SVM, and output the
//! learned weights. Optionally report accuracy on a held-out test set.
//!
//! ## Usage
//!
//! ```text
//! cargo run --features cli --bin train-cli -- --train data.csv --test test.csv
//! ```
//!
//! CSV format: each row has d feature columns (values -1, 0, or 1)
//! followed by a label column (±1 for binary, or -1/0/+1 for multi-class).

use clap::Parser;
use std::error::Error;
use std::fs::File;
use std::path::PathBuf;

use ternary_svm::{OvOTernSVM, TernSVM, Trit};

/// Alias for CSV parse result: feature vectors + label vector.
type CsvResult<T> = Result<(Vec<Vec<Trit>>, T), Box<dyn Error>>;

/// Train a ternary SVM from CSV data using PEGASOS.
#[derive(Parser, Debug)]
#[command(name = "train-cli", about = "Train a ternary SVM", version)]
struct Args {
    /// Path to training CSV (features + labels, no header)
    #[arg(short, long)]
    train: PathBuf,

    /// Path to test CSV (features + labels, no header)
    #[arg(short, long)]
    test: Option<PathBuf>,

    /// Regularization parameter lambda (higher = stronger regularization)
    #[arg(long, default_value = "1.0")]
    lambda: f64,

    /// Learning rate for PEGASOS
    #[arg(long, default_value = "0.01")]
    learning_rate: f64,

    /// Number of training epochs
    #[arg(long, default_value = "200")]
    epochs: usize,

    /// Early stopping tolerance
    #[arg(long, default_value = "1e-4")]
    tol: f64,

    /// Enable OvO multi-class (labels -1,0,+1 instead of binary +/-1)
    #[arg(long, default_value_t = false)]
    multiclass: bool,

    /// Random seed
    #[arg(long)]
    seed: Option<u64>,

    /// Output weights to JSON file
    #[arg(short, long)]
    output: Option<PathBuf>,
}

/// Parse a CSV file into feature vectors and binary labels.
fn read_csv(path: &PathBuf) -> CsvResult<Vec<f64>> {
    let file = File::open(path)?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(false)
        .from_reader(file);

    let mut x: Vec<Vec<Trit>> = Vec::new();
    let mut y: Vec<f64> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        if record.len() < 2 {
            return Err("Each row needs at least 1 feature + 1 label".into());
        }

        let n_features = record.len() - 1;
        let mut features = Vec::with_capacity(n_features);
        for val_str in record.iter().take(n_features) {
            let val: i8 = val_str.parse()?;
            if !matches!(val, -1i8..=1) {
                return Err(format!(
                    "Feature values must be -1, 0, or 1, got {} at row {}",
                    val,
                    rdr.position().line()
                )
                .into());
            }
            features.push(val);
        }
        let label: f64 = record[n_features].parse()?;

        x.push(features);
        y.push(label);
    }

    Ok((x, y))
}

/// Parse CSV with i8 labels (for multi-class).
fn read_csv_multiclass(path: &PathBuf) -> CsvResult<Vec<i8>> {
    let file = File::open(path)?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file);

    let mut x: Vec<Vec<Trit>> = Vec::new();
    let mut y: Vec<i8> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        if record.len() < 2 {
            return Err("Each row needs at least 1 feature + 1 label".into());
        }

        let n_features = record.len() - 1;
        let mut features = Vec::with_capacity(n_features);
        for val_str in record.iter().take(n_features) {
            let val: i8 = val_str.parse()?;
            if !matches!(val, -1i8..=1) {
                return Err(format!(
                    "Feature values must be -1, 0, or 1, got {} at row {}",
                    val,
                    rdr.position().line()
                )
                .into());
            }
            features.push(val);
        }
        let label: i8 = record[n_features].parse()?;
        if !matches!(label, -1i8..=1) {
            return Err(format!(
                "Labels must be -1, 0, or 1 (multi-class mode), got {}",
                label
            )
            .into());
        }

        x.push(features);
        y.push(label);
    }

    Ok((x, y))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    if args.multiclass {
        run_multiclass(&args)?;
    } else {
        run_binary(&args)?;
    }

    Ok(())
}

fn run_binary(args: &Args) -> Result<(), Box<dyn Error>> {
    let (train_x, train_y) = read_csv(&args.train)?;
    eprintln!(
        "Loaded {} training samples with {} features each",
        train_x.len(),
        if train_x.is_empty() {
            0
        } else {
            train_x[0].len()
        }
    );

    let mut model = TernSVM::new(args.lambda, args.learning_rate)
        .with_max_epochs(args.epochs)
        .with_early_stop(args.tol, 5);
    if let Some(seed) = args.seed {
        model = model.with_seed(seed);
    }

    model.fit(&train_x, &train_y)?;

    let train_acc = model.score(&train_x, &train_y)?;
    eprintln!("Training accuracy: {:.4}", train_acc);
    eprintln!(
        "Trained for {} epochs (converged: {})",
        model.epochs_trained, model.converged
    );
    eprintln!("Weight vector: {:?}", model.w);
    eprintln!("Bias: {:.6}", model.b);

    if let Some(ref weights_path) = args.output {
        let output = serde_json::json!({
            "w": model.w,
            "b": model.b,
            "n_features": model.n_features,
            "epochs_trained": model.epochs_trained,
            "converged": model.converged,
            "train_accuracy": train_acc,
            "loss_history": model.loss_history,
            "weight_norm_history": model.weight_norm_history,
        });
        let json_str = serde_json::to_string_pretty(&output)?;
        std::fs::write(weights_path, json_str)?;
        eprintln!("Weights written to {}", weights_path.display());
    }

    if let Some(ref test_path) = args.test {
        let (test_x, test_y) = read_csv(test_path)?;
        let test_acc = model.score(&test_x, &test_y)?;
        println!("Test accuracy: {:.4}", test_acc);
    }

    Ok(())
}

fn run_multiclass(args: &Args) -> Result<(), Box<dyn Error>> {
    let (train_x, train_y) = read_csv_multiclass(&args.train)?;
    eprintln!(
        "Loaded {} training samples with {} features each (multi-class mode)",
        train_x.len(),
        if train_x.is_empty() {
            0
        } else {
            train_x[0].len()
        }
    );

    let mut model = OvOTernSVM::new(args.lambda, args.learning_rate).with_max_epochs(args.epochs);
    if let Some(seed) = args.seed {
        model = model.with_seed(seed);
    }

    model.fit(&train_x, &train_y)?;

    let train_acc = model.score(&train_x, &train_y)?;
    eprintln!("Training accuracy: {:.4}", train_acc);

    if let Some(ref test_path) = args.test {
        let (test_x, test_y) = read_csv_multiclass(test_path)?;
        let test_acc = model.score(&test_x, &test_y)?;
        println!("Test accuracy: {:.4}", test_acc);
    }

    Ok(())
}
