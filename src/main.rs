mod backtest;
mod dataset;
mod features;
mod live;
mod mlb;
mod model;
mod stats;
mod training;
mod ui;

use std::path::Path;
use std::time::Duration;

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};

use backtest::usable;
use features::{FeatureBuilder, Sample};
use mlb::Freshness;
use training::TrainConfig;

type Backend = Autodiff<NdArray>;

const DATA_CSV: &str = "baseball_data.csv";
/// 2020 se excluye: temporada de 60 partidos sin público, no es comparable.
const DEFAULT_SEASONS: [i32; 8] = [2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026];
const MIN_GP: usize = 15;
const HTML_OUT: &str = "dashboard.html";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("all");
    let device = NdArrayDevice::default();

    match cmd {
        "fetch" => {
            let seasons = parse_seasons(&args[1..]);
            build_dataset(&seasons, Freshness::Full);
        }
        "train" => {
            let samples = ensure_dataset();
            holdout::<Backend>(&device, &samples);
        }
        "backtest" => {
            let from = args
                .get(1)
                .cloned()
                .unwrap_or_else(|| "2024-05-01".to_string());
            let samples = ensure_dataset();
            run_backtest::<Backend>(&device, &samples, &from);
        }
        "predict" => {
            let Some(date) = args.get(1) else {
                eprintln!("uso: predict YYYY-MM-DD");
                return;
            };
            let samples = ensure_dataset();
            backtest::next_day_demo::<Backend>(&device, &samples, date, MIN_GP, &cfg());
        }
        "ui" | "serve" => {
            let days = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(live::DEFAULT_DAYS);
            let port = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(8080);
            let cada = args.get(3).and_then(|a| a.parse().ok()).unwrap_or(60u64);

            let mut engine = new_engine::<Backend>(device, days);
            let inicial = engine.tick().expect("no se pudo construir el panel");
            if let Err(e) = ui::write_file(HTML_OUT, &inicial) {
                eprintln!("no se pudo escribir {HTML_OUT}: {e}");
            }
            if let Err(e) = ui::serve_live(inicial, port, Duration::from_secs(cada), move || {
                engine.tick()
            }) {
                eprintln!("no se pudo abrir el puerto {port}: {e}");
            }
        }
        "export" => {
            let days = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(live::DEFAULT_DAYS);
            let mut engine = new_engine::<Backend>(device, days);
            let payload = engine.tick().expect("no se pudo construir el panel");
            match ui::write_file(HTML_OUT, &payload) {
                Ok(()) => println!("Panel guardado en {HTML_OUT}"),
                Err(e) => eprintln!("no se pudo escribir {HTML_OUT}: {e}"),
            }
        }
        "all" => {
            let samples = ensure_dataset();
            holdout::<Backend>(&device, &samples);
            run_backtest::<Backend>(&device, &samples, "2025-05-01");
            header("PREDICCION DIA A DIA (entrenado solo con el pasado)");
            backtest::next_day_demo::<Backend>(&device, &samples, "2026-06-01", MIN_GP, &cfg());
        }
        other => {
            eprintln!("comando desconocido: {other}\n");
            eprintln!("comandos disponibles:");
            eprintln!("  ui [dias] [puerto]     panel web con las proximas fechas (10 dias, puerto 8080)");
            eprintln!("  export [dias]          genera {HTML_OUT} sin levantar el servidor");
            eprintln!("  fetch [temporadas...]  descarga datos reales y regenera el CSV");
            eprintln!("  train                  corte temporal: pasado -> ultima temporada");
            eprintln!("  backtest [YYYY-MM-DD]  validacion walk-forward mes a mes");
            eprintln!("  predict YYYY-MM-DD     jornada pasada, partido a partido");
            eprintln!("  all                    train + backtest + predict");
        }
    }
}

/// Arranca el motor en vivo: histórico, entrenamiento y estadísticas.
fn new_engine<B: burn::tensor::backend::AutodiffBackend>(
    device: B::Device,
    days: i64,
) -> live::LiveEngine<B> {
    header("PANEL EN VIVO - TEMPORADA EN CURSO");
    live::LiveEngine::new(device, &DEFAULT_SEASONS, MIN_GP, days, cfg())
        .expect("no se pudo arrancar el motor en vivo")
}

fn cfg() -> TrainConfig {
    TrainConfig::default()
}

fn parse_seasons(args: &[String]) -> Vec<i32> {
    let parsed: Vec<i32> = args.iter().filter_map(|a| a.parse().ok()).collect();
    if parsed.is_empty() {
        DEFAULT_SEASONS.to_vec()
    } else {
        parsed
    }
}

fn header(title: &str) {
    println!("\n{}", "=".repeat(78));
    println!("  {title}");
    println!("{}", "=".repeat(78));
}

/// Descarga los partidos reales y genera el CSV de características.
fn build_dataset(seasons: &[i32], fresh: Freshness) -> Vec<Sample> {
    header("DESCARGA DE DATOS REALES (MLB Stats API)");
    let games = mlb::fetch_games(seasons, fresh).expect("fallo la descarga de partidos");
    println!("Total: {} partidos reales finalizados.", games.len());

    println!("\nConstruyendo caracteristicas cronologicas (sin fuga de datos)...");
    let mut fb = FeatureBuilder::new();
    let samples = fb.build(&games);
    dataset::save_csv(DATA_CSV, &samples).expect("no se pudo escribir el CSV");
    println!(
        "{} filas guardadas en {DATA_CSV} ({} caracteristicas por partido).",
        samples.len(),
        features::NUM_FEATURES
    );
    samples
}

/// Carga el CSV; si no existe o tiene formato viejo, lo regenera.
fn ensure_dataset() -> Vec<Sample> {
    if Path::new(DATA_CSV).exists() {
        match dataset::load_csv(DATA_CSV) {
            Ok(s) if !s.is_empty() => {
                println!("Dataset cargado: {} partidos reales.", s.len());
                return s;
            }
            Ok(_) => println!("El CSV esta vacio, regenerando..."),
            Err(e) => println!("CSV no utilizable ({e}), regenerando..."),
        }
    }
    build_dataset(&DEFAULT_SEASONS, Freshness::Normal)
}

/// Entrenamiento único: pasado -> última temporada completa como prueba.
fn holdout<B: burn::tensor::backend::AutodiffBackend>(device: &B::Device, all: &[Sample]) {
    header("ENTRENAMIENTO UNICO (corte temporal)");

    let last_season = all.iter().map(|s| s.season).max().unwrap_or(0);
    let train_pool: Vec<Sample> = all
        .iter()
        .filter(|s| s.season < last_season && usable(s, MIN_GP))
        .cloned()
        .collect();
    let test: Vec<Sample> = all
        .iter()
        .filter(|s| s.season == last_season && usable(s, MIN_GP))
        .cloned()
        .collect();

    println!(
        "Entrenamiento: temporadas < {last_season} -> {} partidos",
        train_pool.len()
    );
    println!("Prueba (nunca vista): temporada {last_season} -> {} partidos\n", test.len());

    let cut = (train_pool.len() as f32 * 0.9) as usize;
    let (tr, va) = train_pool.split_at(cut);

    let mut c = cfg();
    c.verbose = true;
    let trained = training::train::<B>(device, tr, va, &c);
    println!(
        "Epocas ejecutadas: {}  |  mejor logloss de validacion: {:.4}",
        trained.epochs_run, trained.best_val_logloss
    );

    let probs = trained.predict(&test);
    let labels: Vec<f32> = test.iter().map(|s| s.label as f32).collect();
    let m = backtest::evaluate(&probs, &labels);
    let b = backtest::baselines(&test, &labels);

    println!("\nRESULTADOS SOBRE LA TEMPORADA {last_season} (datos nunca vistos)");
    println!("{:<28} {:>8} {:>10} {:>9}", "MODELO", "ACIERTO", "LOGLOSS", "BRIER");
    println!("{}", "-".repeat(60));
    row("Red neuronal", &m);
    row("Baseline: Elo", &b.elo);
    row("Baseline: mejor record", &b.record);
    row("Baseline: siempre local", &b.home);
}

fn row(name: &str, m: &backtest::Metrics) {
    println!(
        "{:<28} {:>7.2}% {:>10.4} {:>9.4}",
        name,
        m.acc * 100.0,
        m.logloss,
        m.brier
    );
}

fn run_backtest<B: burn::tensor::backend::AutodiffBackend>(
    device: &B::Device,
    all: &[Sample],
    from: &str,
) {
    header(&format!("VALIDACION WALK-FORWARD MES A MES (desde {from})"));
    println!("Cada mes se reentrena desde cero solo con partidos anteriores.\n");

    let blocks = backtest::walk_forward::<B>(device, all, from, MIN_GP, &cfg());
    if blocks.is_empty() {
        println!("No hubo bloques evaluables.");
        return;
    }

    let (net, elo, home) = backtest::aggregate(&blocks);
    println!("\nAGREGADO DE TODOS LOS MESES");
    println!("{:<28} {:>8} {:>10} {:>9}", "MODELO", "ACIERTO", "LOGLOSS", "BRIER");
    println!("{}", "-".repeat(60));
    row("Red neuronal", &net);
    row("Baseline: Elo", &elo);
    row("Baseline: siempre local", &home);
    println!("Partidos evaluados: {}", net.n);

    println!("\nCALIBRACION (la probabilidad que dice, ¿se cumple?)");
    println!("{:<16} {:>12} {:>12} {:>8}", "BANDA", "PREDICHO", "REAL", "N");
    println!("{}", "-".repeat(52));
    for (lo, pred, real, n) in backtest::calibration(&blocks) {
        println!(
            "{:<16} {:>11.1}% {:>11.1}% {:>8}",
            format!("p >= {:.2}", lo),
            pred * 100.0,
            real * 100.0,
            n
        );
    }

    let mejor = blocks
        .iter()
        .max_by(|a, b| a.metrics.acc.total_cmp(&b.metrics.acc))
        .unwrap();
    let peor = blocks
        .iter()
        .min_by(|a, b| a.metrics.acc.total_cmp(&b.metrics.acc))
        .unwrap();
    println!(
        "Mejor mes: {} ({:.1}%, {} partidos, entrenado con {})",
        mejor.label,
        mejor.metrics.acc * 100.0,
        mejor.metrics.n,
        mejor.train_n
    );
    println!(
        "Peor  mes: {} ({:.1}%, {} partidos, entrenado con {})",
        peor.label,
        peor.metrics.acc * 100.0,
        peor.metrics.n,
        peor.train_n
    );
    println!(
        "Meses en que la red supero al Elo: {}/{}",
        blocks.iter().filter(|b| b.metrics.acc > b.elo_metrics.acc).count(),
        blocks.len()
    );
    println!(
        "Meses en que la red supero al 'siempre local': {}/{}",
        blocks.iter().filter(|b| b.metrics.acc > b.home_metrics.acc).count(),
        blocks.len()
    );

    println!("\nACIERTO SEGUN CONFIANZA DE LA RED");
    println!("{:<20} {:>9} {:>8}", "CONFIANZA", "ACIERTO", "N");
    println!("{}", "-".repeat(40));
    for (name, acc, n) in backtest::accuracy_by_confidence(&blocks) {
        println!("{:<20} {:>8.1}% {:>8}", name, acc * 100.0, n);
    }
}
