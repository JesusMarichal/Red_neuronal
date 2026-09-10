//! Validación walk-forward (hacia adelante en el tiempo).
//!
//! Para cada bloque de prueba se entrena un modelo NUEVO usando solo los
//! partidos anteriores a ese bloque, y se predicen los partidos del bloque.
//! Es exactamente la idea de "entreno con noviembre y predigo diciembre":
//! el modelo nunca ve el futuro.

use burn::tensor::backend::AutodiffBackend;

use crate::features::Sample;
use crate::training::{accuracy, brier, log_loss, train, TrainConfig};

/// Descarta partidos de inicio de temporada, donde las estadísticas
/// acumuladas todavía no dicen nada.
pub fn usable(s: &Sample, min_gp: usize) -> bool {
    s.home_gp >= min_gp && s.away_gp >= min_gp
}

#[derive(Debug, Default, Clone)]
pub struct Metrics {
    pub n: usize,
    pub acc: f32,
    pub logloss: f32,
    pub brier: f32,
}

pub fn evaluate(probs: &[f32], labels: &[f32]) -> Metrics {
    Metrics {
        n: probs.len(),
        acc: accuracy(probs, labels),
        logloss: log_loss(probs, labels),
        brier: brier(probs, labels),
    }
}

/// Baselines contra los que hay que comparar la red.
pub struct Baselines {
    pub home: Metrics,   // apostar siempre al local
    pub elo: Metrics,    // usar la probabilidad Elo
    pub record: Metrics, // el equipo con mejor récord
}

pub fn baselines(samples: &[Sample], labels: &[f32]) -> Baselines {
    let home: Vec<f32> = vec![1.0; samples.len()];
    let elo: Vec<f32> = samples.iter().map(|s| s.elo_prob).collect();
    // features[2] = home_wpct, features[3] = away_wpct
    let record: Vec<f32> = samples
        .iter()
        .map(|s| if s.features[2] >= s.features[3] { 0.9 } else { 0.1 })
        .collect();
    Baselines {
        home: evaluate(&home, labels),
        elo: evaluate(&elo, labels),
        record: evaluate(&record, labels),
    }
}

/// Resultado de un bloque temporal (un mes).
pub struct Block {
    pub label: String,
    pub train_n: usize,
    pub metrics: Metrics,
    pub elo_metrics: Metrics,
    pub home_metrics: Metrics,
    pub probs: Vec<f32>,
    pub labels: Vec<f32>,
    pub samples: Vec<Sample>,
}

/// Ejecuta la validación walk-forward mes a mes.
///
/// `test_from` es una fecha "YYYY-MM-DD": todo lo anterior es historia
/// inicial de entrenamiento; de ahí en adelante se evalúa mes por mes.
pub fn walk_forward<B: AutodiffBackend>(
    device: &B::Device,
    all: &[Sample],
    test_from: &str,
    min_gp: usize,
    cfg: &TrainConfig,
) -> Vec<Block> {
    // Meses distintos presentes en el periodo de prueba.
    let mut months: Vec<String> = all
        .iter()
        .filter(|s| s.date.as_str() >= test_from && usable(s, min_gp))
        .map(|s| s.date[..7].to_string())
        .collect();
    months.sort();
    months.dedup();

    let mut blocks = Vec::new();

    for month in months {
        let month_start = format!("{month}-01");

        // Entrenamiento: TODO lo estrictamente anterior al mes.
        let train_pool: Vec<Sample> = all
            .iter()
            .filter(|s| s.date < month_start && usable(s, min_gp))
            .cloned()
            .collect();
        // Prueba: los partidos de ese mes.
        let test: Vec<Sample> = all
            .iter()
            .filter(|s| s.date[..7] == month && usable(s, min_gp))
            .cloned()
            .collect();

        if train_pool.len() < 500 || test.is_empty() {
            continue;
        }

        // Validación interna = último 10% del histórico (también anterior en
        // el tiempo al bloque de prueba), para la parada temprana.
        let cut = (train_pool.len() as f32 * 0.9) as usize;
        let (tr, va) = train_pool.split_at(cut);

        let trained = train::<B>(device, tr, va, cfg);
        let probs = trained.predict(&test);
        let labels: Vec<f32> = test.iter().map(|s| s.label as f32).collect();

        let elo: Vec<f32> = test.iter().map(|s| s.elo_prob).collect();
        let home: Vec<f32> = vec![1.0; test.len()];

        println!(
            "  {month}  entren.={:>6}  prueba={:>4}  aciertos_red={:.1}%  Elo={:.1}%  local={:.1}%",
            train_pool.len(),
            test.len(),
            accuracy(&probs, &labels) * 100.0,
            accuracy(&elo, &labels) * 100.0,
            accuracy(&home, &labels) * 100.0,
        );

        blocks.push(Block {
            label: month,
            train_n: train_pool.len(),
            metrics: evaluate(&probs, &labels),
            elo_metrics: evaluate(&elo, &labels),
            home_metrics: evaluate(&home, &labels),
            probs,
            labels,
            samples: test,
        });
    }

    blocks
}

/// Junta todos los bloques en una métrica global.
pub fn aggregate(blocks: &[Block]) -> (Metrics, Metrics, Metrics) {
    let mut p = Vec::new();
    let mut y = Vec::new();
    let mut e = Vec::new();
    let mut h = Vec::new();
    for b in blocks {
        p.extend_from_slice(&b.probs);
        y.extend_from_slice(&b.labels);
        e.extend(b.samples.iter().map(|s| s.elo_prob));
        h.extend(std::iter::repeat(1.0f32).take(b.probs.len()));
    }
    (evaluate(&p, &y), evaluate(&e, &y), evaluate(&h, &y))
}

/// Tabla de calibración: ¿cuando la red dice 70%, gana el local el 70%?
pub fn calibration(blocks: &[Block]) -> Vec<(f32, f32, f32, usize)> {
    let edges = [0.0f32, 0.35, 0.45, 0.50, 0.55, 0.65, 1.01];
    let mut rows = Vec::new();
    for w in edges.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        let mut sum_p = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut n = 0usize;
        for b in blocks {
            for (p, y) in b.probs.iter().zip(&b.labels) {
                if *p >= lo && *p < hi {
                    sum_p += p;
                    sum_y += y;
                    n += 1;
                }
            }
        }
        if n > 0 {
            rows.push((lo, sum_p / n as f32, sum_y / n as f32, n));
        }
    }
    rows
}

/// Precisión según cuán segura está la red (|p - 0.5|).
pub fn accuracy_by_confidence(blocks: &[Block]) -> Vec<(&'static str, f32, usize)> {
    let bands: [(&str, f32, f32); 4] = [
        ("baja    (50-55%)", 0.00, 0.05),
        ("media   (55-60%)", 0.05, 0.10),
        ("alta    (60-65%)", 0.10, 0.15),
        ("muy alta ( >65%)", 0.15, 1.00),
    ];
    let mut out = Vec::new();
    for (name, lo, hi) in bands {
        let mut hits = 0usize;
        let mut n = 0usize;
        for b in blocks {
            for (p, y) in b.probs.iter().zip(&b.labels) {
                let conf = (p - 0.5).abs();
                if conf >= lo && conf < hi {
                    n += 1;
                    if (*p >= 0.5) == (*y > 0.5) {
                        hits += 1;
                    }
                }
            }
        }
        if n > 0 {
            out.push((name, hits as f32 / n as f32, n));
        }
    }
    out
}

/// Demo pedida explícitamente: entrena con todo lo anterior a `cutoff` y
/// predice, partido por partido, la primera jornada posterior.
pub fn next_day_demo<B: AutodiffBackend>(
    device: &B::Device,
    all: &[Sample],
    cutoff: &str,
    min_gp: usize,
    cfg: &TrainConfig,
) {
    let train_pool: Vec<Sample> = all
        .iter()
        .filter(|s| s.date.as_str() < cutoff && usable(s, min_gp))
        .cloned()
        .collect();

    let Some(day) = all
        .iter()
        .filter(|s| s.date.as_str() >= cutoff && usable(s, min_gp))
        .map(|s| s.date.clone())
        .min()
    else {
        println!("No hay partidos posteriores a {cutoff}.");
        return;
    };

    let test: Vec<Sample> = all
        .iter()
        .filter(|s| s.date == day && usable(s, min_gp))
        .cloned()
        .collect();

    if train_pool.len() < 500 {
        println!("Historia insuficiente antes de {cutoff}.");
        return;
    }

    println!(
        "\nEntrenando con {} partidos anteriores a {cutoff} ...",
        train_pool.len()
    );
    let cut = (train_pool.len() as f32 * 0.9) as usize;
    let (tr, va) = train_pool.split_at(cut);
    let trained = train::<B>(device, tr, va, cfg);

    let probs = trained.predict(&test);
    let labels: Vec<f32> = test.iter().map(|s| s.label as f32).collect();

    println!("\nJORNADA DEL {day}  ({} partidos)", test.len());
    println!(
        "{:<26} {:<26} {:>9} {:>9} {:>10} {:>6}",
        "VISITANTE", "LOCAL", "P(local)", "Elo", "RESULTADO", "OK"
    );
    println!("{}", "-".repeat(92));

    let mut hits = 0;
    for (i, s) in test.iter().enumerate() {
        let p = probs[i];
        let pick_home = p >= 0.5;
        let ok = pick_home == (s.label == 1);
        if ok {
            hits += 1;
        }
        println!(
            "{:<26} {:<26} {:>8.1}% {:>8.1}% {:>4}-{:<5} {:>6}",
            trim(&s.away_name),
            trim(&s.home_name),
            p * 100.0,
            s.elo_prob * 100.0,
            s.away_score,
            s.home_score,
            if ok { "SI" } else { "no" }
        );
    }
    println!("{}", "-".repeat(92));
    let m = evaluate(&probs, &labels);
    let elo: Vec<f32> = test.iter().map(|s| s.elo_prob).collect();
    println!(
        "Aciertos de la red: {hits}/{}  ({:.1}%)   |  Elo: {:.1}%  |  siempre local: {:.1}%",
        test.len(),
        m.acc * 100.0,
        accuracy(&elo, &labels) * 100.0,
        labels.iter().sum::<f32>() / labels.len() as f32 * 100.0
    );
}

fn trim(s: &str) -> String {
    if s.chars().count() > 25 {
        s.chars().take(24).collect::<String>() + "."
    } else {
        s.to_string()
    }
}
