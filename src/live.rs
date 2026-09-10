//! Une todas las piezas para la interfaz: entrena el modelo con el histórico
//! completo, baja el calendario y las estadísticas de la temporada en curso,
//! y produce el JSON que consume el panel web.

use burn::tensor::backend::AutodiffBackend;
use chrono::{Duration, Local};
use serde_json::{json, Value};

use crate::backtest::{evaluate, usable};
use crate::features::{FeatureBuilder, Sample};
use crate::mlb::{self, ScheduledGame};
use crate::stats::{self, League, Pitcher};
use crate::training::{accuracy, train, TrainConfig};

/// Días hacia adelante que se muestran por defecto.
pub const DEFAULT_DAYS: i64 = 10;

/// Proporción -> porcentaje, en precisión completa.
/// El redondeo a 2 decimales lo hace la interfaz al mostrarlo; aquí y en el
/// modelo el valor viaja tal cual.
fn pct(x: f32) -> f64 {
    (x as f64) * 100.0
}

fn pitcher_json(p: Option<&Pitcher>, fallback_name: &str) -> Value {
    match p {
        Some(p) => json!({
            "name": p.name,
            "era": p.era,
            "whip": p.whip,
            "wins": p.wins,
            "losses": p.losses,
            "innings": p.innings,
            "strikeouts": p.strikeouts,
            "walks": p.walks,
            "starts": p.starts,
            "known": true,
        }),
        None => json!({
            "name": if fallback_name.is_empty() { "Por anunciar" } else { fallback_name },
            "era": "-", "whip": "-", "wins": 0, "losses": 0,
            "innings": "-", "strikeouts": 0, "walks": 0, "starts": 0,
            "known": false,
        }),
    }
}

fn team_json(info: Option<&stats::TeamInfo>, snap: Option<&crate::features::TeamSnapshot>, name: &str) -> Value {
    let hitters: Vec<Value> = info
        .map(|t| {
            t.hitters
                .iter()
                .take(12)
                .map(|h| {
                    json!({
                        "name": h.name, "pos": h.pos, "avg": h.avg, "obp": h.obp,
                        "ops": h.ops, "hr": h.hr, "rbi": h.rbi,
                        "hits": h.hits, "ab": h.at_bats, "games": h.games,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let rotation: Vec<Value> = info
        .map(|t| {
            t.pitchers
                .iter()
                .filter(|p| p.starts > 0)
                .take(6)
                .map(|p| {
                    json!({
                        "name": p.name, "era": p.era, "whip": p.whip,
                        "wins": p.wins, "losses": p.losses,
                        "innings": p.innings, "strikeouts": p.strikeouts,
                        "starts": p.starts,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    json!({
        "id": snap.map(|s| s.id).unwrap_or(0),
        "name": info.map(|t| t.name.clone()).unwrap_or_else(|| name.to_string()),
        "abbrev": info.map(|t| t.abbrev.clone()).unwrap_or_default(),
        "wins": snap.map(|s| s.wins).unwrap_or(0),
        "losses": snap.map(|s| s.losses).unwrap_or(0),
        "elo": snap.map(|s| s.elo.round() as i64).unwrap_or(1500),
        "rsPg": snap.map(|s| (s.rs_pg * 100.0).round() / 100.0).unwrap_or(0.0),
        "raPg": snap.map(|s| (s.ra_pg * 100.0).round() / 100.0).unwrap_or(0.0),
        "form10": snap.map(|s| s.form10 * 100.0).unwrap_or(50.0),
        "batAvg": info.map(|t| t.bat_avg.clone()).unwrap_or_else(|| "-".into()),
        "obp": info.map(|t| t.obp.clone()).unwrap_or_else(|| "-".into()),
        "slg": info.map(|t| t.slg.clone()).unwrap_or_else(|| "-".into()),
        "ops": info.map(|t| t.ops.clone()).unwrap_or_else(|| "-".into()),
        "runs": info.map(|t| t.runs).unwrap_or(0),
        "homeRuns": info.map(|t| t.home_runs).unwrap_or(0),
        "era": info.map(|t| t.era.clone()).unwrap_or_else(|| "-".into()),
        "whip": info.map(|t| t.whip.clone()).unwrap_or_else(|| "-".into()),
        "hitters": hitters,
        "rotation": rotation,
    })
}

/// Construye el payload completo del panel.
pub fn build_payload<B: AutodiffBackend>(
    device: &B::Device,
    samples: &[Sample],
    games: &[mlb::RawGame],
    min_gp: usize,
    days: i64,
    cfg: &TrainConfig,
) -> Result<Value, String> {
    let season = mlb::current_year();

    // --- 1. Estado cronológico hasta hoy (mismo pipeline del backtest) ---
    let mut fb = FeatureBuilder::new();
    fb.build(games);
    let snaps = fb.snapshot();

    // --- 2. Modelo de evaluación: honesto sobre los últimos 30 días ---
    let today = Local::now().date_naive();
    let cutoff = (today - Duration::days(30)).format("%Y-%m-%d").to_string();

    let hist: Vec<Sample> = samples.iter().filter(|s| usable(s, min_gp)).cloned().collect();
    let eval_train: Vec<Sample> = hist.iter().filter(|s| s.date < cutoff).cloned().collect();
    let eval_test: Vec<Sample> = hist.iter().filter(|s| s.date >= cutoff).cloned().collect();

    let recent = if eval_train.len() > 500 && !eval_test.is_empty() {
        let cut = (eval_train.len() as f32 * 0.9) as usize;
        let (tr, va) = eval_train.split_at(cut);
        let m = train::<B>(device, tr, va, cfg);
        let probs = m.predict(&eval_test);
        let labels: Vec<f32> = eval_test.iter().map(|s| s.label as f32).collect();
        let elo: Vec<f32> = eval_test.iter().map(|s| s.elo_prob).collect();
        let met = evaluate(&probs, &labels);
        json!({
            "desde": cutoff,
            "partidos": met.n,
            "aciertoRed": pct(met.acc),
            "aciertoElo": pct(accuracy(&elo, &labels)),
            "logloss": met.logloss as f64,
            "brier": met.brier as f64,
        })
    } else {
        json!(null)
    };

    // --- 3. Modelo de producción: entrenado con TODO el histórico ---
    let cut = (hist.len() as f32 * 0.9) as usize;
    let (tr, va) = hist.split_at(cut);
    let model = train::<B>(device, tr, va, cfg);

    // --- 4. Estadísticas de la temporada en curso ---
    let league: League = stats::fetch_league(season)?;

    // --- 5. Calendario por venir ---
    let from = today.format("%Y-%m-%d").to_string();
    let scheduled = mlb::fetch_scheduled(&from, days)?;

    // Un solo lote de predicción para todos los partidos programados.
    let mut usable_games: Vec<(&ScheduledGame, Sample)> = Vec::new();
    for g in &scheduled {
        let Some((feats, elo_p)) =
            fb.features_for(g.home_id, g.away_id, &g.date, g.home_sp_id, g.away_sp_id)
        else {
            continue;
        };
        // Sample sintético: solo se usan `features` para inferir.
        let s = Sample {
            game_pk: g.game_pk,
            date: g.date.clone(),
            season,
            home_name: g.home_name.clone(),
            away_name: g.away_name.clone(),
            home_score: 0,
            away_score: 0,
            label: 0,
            features: feats,
            home_gp: 0,
            away_gp: 0,
            elo_prob: elo_p,
        };
        usable_games.push((g, s));
    }

    let batch: Vec<Sample> = usable_games.iter().map(|(_, s)| s.clone()).collect();
    let probs = model.predict(&batch);

    // --- 6. Agrupar por fecha ---
    let mut dates: Vec<Value> = Vec::new();
    let mut current_date = String::new();
    let mut current_games: Vec<Value> = Vec::new();

    for (idx, (g, s)) in usable_games.iter().enumerate() {
        if g.date != current_date {
            if !current_date.is_empty() {
                dates.push(json!({ "date": current_date, "games": current_games }));
            }
            current_date = g.date.clone();
            current_games = Vec::new();
        }

        let p_home = probs[idx];
        let hp = league.pitchers.get(&g.home_sp_id);
        let ap = league.pitchers.get(&g.away_sp_id);

        current_games.push(json!({
            "gamePk": g.game_pk,
            "start": g.start_utc,
            "venue": g.venue,
            "status": g.status,
            "confirmed": g.pitchers_confirmed,
            "pHome": pct(p_home),
            // El complemento se deriva para que la barra sume siempre 100.
            "pAway": 100.0 - pct(p_home),
            "eloHome": pct(s.elo_prob),
            "final": g.is_final,
            "homeScore": g.home_score,
            "awayScore": g.away_score,
            "hit": match (g.is_final, g.home_score, g.away_score) {
                (true, Some(hs), Some(as_)) if hs != as_ => Some((p_home >= 0.5) == (hs > as_)),
                _ => None,
            },
            "pick": if p_home >= 0.5 { g.home_name.clone() } else { g.away_name.clone() },
            "home": team_json(league.teams.get(&g.home_id), snaps.get(&g.home_id), &g.home_name),
            "away": team_json(league.teams.get(&g.away_id), snaps.get(&g.away_id), &g.away_name),
            "homePitcher": pitcher_json(hp, &g.home_sp_name),
            "awayPitcher": pitcher_json(ap, &g.away_sp_name),
        }));
    }
    if !current_date.is_empty() {
        dates.push(json!({ "date": current_date, "games": current_games }));
    }

    // --- 7. Tabla de posiciones por Elo ---
    let mut standings: Vec<Value> = snaps
        .values()
        .filter(|s| s.wins + s.losses > 0)
        .map(|s| {
            let info = league.teams.get(&s.id);
            json!({
                "id": s.id,
                "name": s.name,
                "abbrev": info.map(|t| t.abbrev.clone()).unwrap_or_default(),
                "wins": s.wins,
                "losses": s.losses,
                "elo": s.elo.round() as i64,
                "rsPg": (s.rs_pg * 100.0).round() / 100.0,
                "raPg": (s.ra_pg * 100.0).round() / 100.0,
                "form10": s.form10 * 100.0,
                "batAvg": info.map(|t| t.bat_avg.clone()).unwrap_or_else(|| "-".into()),
                "ops": info.map(|t| t.ops.clone()).unwrap_or_else(|| "-".into()),
                "era": info.map(|t| t.era.clone()).unwrap_or_else(|| "-".into()),
            })
        })
        .collect();
    standings.sort_by(|a, b| b["elo"].as_i64().unwrap_or(0).cmp(&a["elo"].as_i64().unwrap_or(0)));

    let confirmadas: usize = usable_games.iter().filter(|(g, _)| g.pitchers_confirmed).count();

    // Último partido ya terminado que entró al entrenamiento.
    let ultimo = games.last().map(|g| g.date.clone()).unwrap_or_default();

    Ok(json!({
        "generado": Local::now().format("%Y-%m-%d %H:%M").to_string(),
        "ultimoPartido": ultimo,
        "temporada": season,
        "modelo": {
            "partidosEntrenamiento": hist.len(),
            "caracteristicas": crate::features::NUM_FEATURES,
            "epocas": model.epochs_run,
            "loglossValidacion": model.best_val_logloss as f64,
            "reciente": recent,
        },
        "resumen": {
            "partidos": usable_games.len(),
            "confirmados": confirmadas,
            "sinConfirmar": usable_games.len() - confirmadas,
            "dias": dates.len(),
        },
        "fechas": dates,
        "tabla": standings,
    }))
}
