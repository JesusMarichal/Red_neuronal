//! Persistencia del dataset ya procesado (CSV) y utilidades de normalización.

use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use crate::features::{Sample, FEATURE_NAMES, NUM_FEATURES};

/// Guarda las filas en CSV, con una columna por característica.
pub fn save_csv(path: &str, samples: &[Sample]) -> Result<(), Box<dyn Error>> {
    let mut w = BufWriter::new(File::create(path)?);

    write!(w, "game_pk,date,season,home,away,home_score,away_score,label,home_gp,away_gp,elo_baseline")?;
    for name in FEATURE_NAMES {
        write!(w, ",{name}")?;
    }
    writeln!(w)?;

    for s in samples {
        write!(
            w,
            "{},{},{},\"{}\",\"{}\",{},{},{},{},{},{:.6}",
            s.game_pk, s.date, s.season, s.home_name, s.away_name,
            s.home_score, s.away_score, s.label, s.home_gp, s.away_gp, s.elo_prob
        )?;
        for f in &s.features {
            write!(w, ",{f:.6}")?;
        }
        writeln!(w)?;
    }
    Ok(())
}

/// Relee el CSV producido por `save_csv`.
pub fn load_csv(path: &str) -> Result<Vec<Sample>, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut out = Vec::new();

    for rec in reader.records() {
        let r = rec?;
        if r.len() != 11 + NUM_FEATURES {
            return Err(format!(
                "CSV con {} columnas, se esperaban {}. Regenera el dataset con `fetch`.",
                r.len(),
                11 + NUM_FEATURES
            )
            .into());
        }
        let mut features = Vec::with_capacity(NUM_FEATURES);
        for i in 0..NUM_FEATURES {
            features.push(r[11 + i].parse::<f32>()?);
        }
        out.push(Sample {
            game_pk: r[0].parse()?,
            date: r[1].to_string(),
            season: r[2].parse()?,
            home_name: r[3].to_string(),
            away_name: r[4].to_string(),
            home_score: r[5].parse()?,
            away_score: r[6].parse()?,
            label: r[7].parse()?,
            home_gp: r[8].parse()?,
            away_gp: r[9].parse()?,
            elo_prob: r[10].parse()?,
            features,
        });
    }
    Ok(out)
}

/// Normalizador z-score. Se ajusta SOLO con datos de entrenamiento y luego
/// se aplica tal cual al conjunto de prueba (si no, habría fuga de datos).
#[derive(Debug, Clone)]
pub struct Normalizer {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl Normalizer {
    pub fn fit(samples: &[Sample]) -> Self {
        let n = samples.len().max(1) as f32;
        let mut mean = vec![0.0f32; NUM_FEATURES];
        let mut std = vec![0.0f32; NUM_FEATURES];

        for s in samples {
            for i in 0..NUM_FEATURES {
                mean[i] += s.features[i];
            }
        }
        for m in mean.iter_mut() {
            *m /= n;
        }
        for s in samples {
            for i in 0..NUM_FEATURES {
                let d = s.features[i] - mean[i];
                std[i] += d * d;
            }
        }
        for v in std.iter_mut() {
            *v = (*v / n).sqrt().max(1e-6);
        }
        Self { mean, std }
    }

    /// Aplana las filas a un vector [n * NUM_FEATURES] ya normalizado.
    pub fn transform(&self, samples: &[Sample]) -> Vec<f32> {
        let mut out = Vec::with_capacity(samples.len() * NUM_FEATURES);
        for s in samples {
            for i in 0..NUM_FEATURES {
                out.push((s.features[i] - self.mean[i]) / self.std[i]);
            }
        }
        out
    }
}
