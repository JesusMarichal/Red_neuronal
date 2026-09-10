//! Entrenamiento por mini-lotes con normalización, weight decay y
//! parada temprana (early stopping) sobre un conjunto de validación.

use burn::{
    module::AutodiffModule,
    nn::loss::CrossEntropyLossConfig,
    optim::{decay::WeightDecayConfig, AdamConfig, GradientsParams, Optimizer},
    tensor::{
        activation::softmax,
        backend::{AutodiffBackend, Backend},
        Int, Tensor,
    },
};

use crate::dataset::Normalizer;
use crate::features::{Sample, NUM_FEATURES};
use crate::model::BaseballPredictor;

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub epochs: usize,
    pub lr: f64,
    pub batch_size: usize,
    pub hidden1: usize,
    pub hidden2: usize,
    pub dropout: f64,
    pub weight_decay: f32,
    pub patience: usize,
    pub seed: u64,
    pub verbose: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 120,
            lr: 3e-3,
            batch_size: 256,
            hidden1: 32,
            hidden2: 16,
            dropout: 0.25,
            weight_decay: 1e-4,
            patience: 15,
            seed: 42,
            verbose: false,
        }
    }
}

/// Modelo entrenado listo para inferencia, junto con su normalizador.
pub struct Trained<B: Backend> {
    pub model: BaseballPredictor<B>,
    pub norm: Normalizer,
    pub device: B::Device,
    pub epochs_run: usize,
    pub best_val_logloss: f32,
}

/// PRNG xorshift: barajado reproducible sin depender de la API de `rand`.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

fn labels_of(samples: &[Sample]) -> Vec<i32> {
    samples.iter().map(|s| s.label as i32).collect()
}

/// Entrena y devuelve el modelo en su backend de inferencia (sin autodiff).
pub fn train<B: AutodiffBackend>(
    device: &B::Device,
    train_set: &[Sample],
    valid_set: &[Sample],
    cfg: &TrainConfig,
) -> Trained<B::InnerBackend> {
    assert!(!train_set.is_empty(), "conjunto de entrenamiento vacío");
    // Sin esto, la inicialización de pesos usa el RNG global de Burn y cada
    // ejecución del backtest daría números distintos.
    B::seed(device, cfg.seed);

    let norm = Normalizer::fit(train_set);
    let x_all = norm.transform(train_set);
    let y_all = labels_of(train_set);

    let mut model: BaseballPredictor<B> =
        BaseballPredictor::new(device, cfg.hidden1, cfg.hidden2, cfg.dropout);
    let mut optim = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(cfg.weight_decay)))
        .init();
    let loss_cfg = CrossEntropyLossConfig::new();

    // Tensores de validación (una sola vez, sin autodiff).
    let has_valid = !valid_set.is_empty();
    let val_x = norm.transform(valid_set);
    let val_y: Vec<f32> = valid_set.iter().map(|s| s.label as f32).collect();

    let mut order: Vec<usize> = (0..train_set.len()).collect();
    let mut rng = Rng(cfg.seed | 1);

    let mut best_logloss = f32::INFINITY;
    let mut best_model = model.clone().valid();
    let mut since_best = 0usize;
    let mut epochs_run = 0usize;

    for epoch in 1..=cfg.epochs {
        epochs_run = epoch;
        rng.shuffle(&mut order);
        let mut epoch_loss = 0.0f32;
        let mut batches = 0usize;

        for chunk in order.chunks(cfg.batch_size) {
            let n = chunk.len();
            let mut xb = Vec::with_capacity(n * NUM_FEATURES);
            let mut yb = Vec::with_capacity(n);
            for &i in chunk {
                xb.extend_from_slice(&x_all[i * NUM_FEATURES..(i + 1) * NUM_FEATURES]);
                yb.push(y_all[i]);
            }

            let x = Tensor::<B, 1>::from_floats(xb.as_slice(), device).reshape([n, NUM_FEATURES]);
            let y = Tensor::<B, 1, Int>::from_ints(yb.as_slice(), device);

            let output = model.forward(x);
            let loss = loss_cfg.init(&output.device()).forward(output, y);
            epoch_loss += loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            batches += 1;

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(cfg.lr, model, grads);
        }

        // --- Validación / early stopping ---
        if has_valid {
            let inner = model.valid();
            let probs = predict_raw::<B::InnerBackend>(&inner, &val_x, valid_set.len(), device);
            let ll = log_loss(&probs, &val_y);
            if ll + 1e-5 < best_logloss {
                best_logloss = ll;
                best_model = inner;
                since_best = 0;
            } else {
                since_best += 1;
            }
            if cfg.verbose && (epoch % 10 == 0 || epoch == 1) {
                println!(
                    "  época {epoch:3}  loss_train={:.4}  logloss_val={ll:.4}  (mejor={best_logloss:.4})",
                    epoch_loss / batches.max(1) as f32
                );
            }
            if since_best >= cfg.patience {
                if cfg.verbose {
                    println!("  parada temprana en la época {epoch}");
                }
                break;
            }
        } else {
            best_model = model.valid();
            best_logloss = epoch_loss / batches.max(1) as f32;
            if cfg.verbose && (epoch % 20 == 0 || epoch == 1) {
                println!("  época {epoch:3}  loss_train={best_logloss:.4}");
            }
        }
    }

    Trained {
        model: best_model,
        norm,
        device: device.clone(),
        epochs_run,
        best_val_logloss: best_logloss,
    }
}

/// Probabilidad de victoria local para un vector de features ya normalizado.
fn predict_raw<B: Backend>(
    model: &BaseballPredictor<B>,
    x_flat: &[f32],
    n: usize,
    device: &B::Device,
) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    let x = Tensor::<B, 1>::from_floats(x_flat, device).reshape([n, NUM_FEATURES]);
    let probs = softmax(model.forward(x), 1);
    let flat = probs.into_data().to_vec::<f32>().unwrap();
    // Columna 1 = "gana el local".
    (0..n).map(|i| flat[i * 2 + 1]).collect()
}

impl<B: Backend> Trained<B> {
    /// Probabilidad de que gane el equipo local, para cada partido.
    pub fn predict(&self, samples: &[Sample]) -> Vec<f32> {
        let x = self.norm.transform(samples);
        predict_raw(&self.model, &x, samples.len(), &self.device)
    }
}

// ------------------------------------------------------------ Métricas
pub fn log_loss(probs: &[f32], labels: &[f32]) -> f32 {
    if probs.is_empty() {
        return f32::NAN;
    }
    let mut s = 0.0f32;
    for (p, y) in probs.iter().zip(labels) {
        let p = p.clamp(1e-7, 1.0 - 1e-7);
        s += -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
    }
    s / probs.len() as f32
}

pub fn brier(probs: &[f32], labels: &[f32]) -> f32 {
    if probs.is_empty() {
        return f32::NAN;
    }
    probs
        .iter()
        .zip(labels)
        .map(|(p, y)| (p - y) * (p - y))
        .sum::<f32>()
        / probs.len() as f32
}

pub fn accuracy(probs: &[f32], labels: &[f32]) -> f32 {
    if probs.is_empty() {
        return f32::NAN;
    }
    let hits = probs
        .iter()
        .zip(labels)
        .filter(|(p, y)| ((**p >= 0.5) as i32 as f32 - **y).abs() < 0.5)
        .count();
    hits as f32 / probs.len() as f32
}
