//! Motor en vivo del panel.
//!
//! Mantiene en memoria el histórico, el estado cronológico de los equipos y el
//! modelo entrenado. Cada `tick()` vuelve a bajar solo lo que cambia de verdad
//! (el calendario de los próximos días, con marcadores y abridores anunciados)
//! y reconstruye el payload. El modelo se reentrena únicamente cuando entran
//! partidos nuevos ya terminados, no en cada refresco.

use std::time::Instant;

use burn::tensor::backend::AutodiffBackend;
use chrono::{Duration, Local};
use serde_json::{json, Value};

use crate::backtest::{evaluate, usable};
use crate::features::{FeatureBuilder, Sample};
use crate::mlb::{self, Freshness, RawGame, ScheduledGame};
use crate::stats::{self, League, Pitcher};
use crate::training::{accuracy, train, TrainConfig, Trained};

/// Días hacia adelante que se muestran por defecto.
pub const DEFAULT_DAYS: i64 = 10;
/// Cada cuánto se rebajan plantillas y estadísticas de jugadores (minutos).
const LEAGUE_TTL_MIN: u64 = 30;
/// Cada cuánto se rebaja el histórico completo por si algo se escapó (minutos).
const HISTORY_TTL_MIN: u64 = 60;

/// Proporción -> porcentaje, en precisión completa.
/// El redondeo a 2 decimales lo hace la interfaz al mostrarlo.
fn pct(x: f32) -> f64 {
    (x as f64) * 100.0
}

/// `id` viene del calendario: puede haber abridor anunciado aunque no esté en
/// nuestro índice de plantillas (un traspaso reciente, por ejemplo). Sirve para
/// pedir su foto aunque no tengamos sus números.
fn pitcher_json(p: Option<&Pitcher>, fallback_name: &str, id: i64) -> Value {
    match p {
        Some(p) => json!({
            "id": p.id, "name": p.name, "era": p.era, "whip": p.whip,
            "wins": p.wins, "losses": p.losses, "innings": p.innings,
            "strikeouts": p.strikeouts, "walks": p.walks, "starts": p.starts,
            "known": true,
        }),
        None => json!({
            "id": id,
            "name": if fallback_name.is_empty() { "Por anunciar" } else { fallback_name },
            "era": "-", "whip": "-", "wins": 0, "losses": 0,
            "innings": "-", "strikeouts": 0, "walks": 0, "starts": 0,
            "known": false,
        }),
    }
}

fn team_json(
    info: Option<&stats::TeamInfo>,
    snap: Option<&crate::features::TeamSnapshot>,
    name: &str,
) -> Value {
    let hitters: Vec<Value> = info
        .map(|t| {
            t.hitters.iter().take(12).map(|h| json!({
                "name": h.name, "pos": h.pos, "avg": h.avg, "obp": h.obp,
                "ops": h.ops, "hr": h.hr, "rbi": h.rbi,
                "hits": h.hits, "ab": h.at_bats, "games": h.games,
            })).collect()
        })
        .unwrap_or_default();

    let rotation: Vec<Value> = info
        .map(|t| {
            t.pitchers.iter().filter(|p| p.starts > 0).take(6).map(|p| json!({
                "id": p.id, "name": p.name, "era": p.era, "whip": p.whip,
                "wins": p.wins, "losses": p.losses, "innings": p.innings,
                "strikeouts": p.strikeouts, "starts": p.starts,
            })).collect()
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
        // Récord separado por condición (toda la temporada).
        "homeW": snap.map(|s| s.home_wins).unwrap_or(0),
        "homeL": snap.map(|s| s.home_losses).unwrap_or(0),
        "awayW": snap.map(|s| s.away_wins).unwrap_or(0),
        "awayL": snap.map(|s| s.away_losses).unwrap_or(0),
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

// ------------------------------------------------------------------ motor
pub struct LiveEngine<B: AutodiffBackend> {
    device: B::Device,
    cfg: TrainConfig,
    min_gp: usize,
    days: i64,
    seasons: Vec<i32>,

    history: Vec<RawGame>,
    samples: Vec<Sample>,
    fb: FeatureBuilder,
    model: Trained<B::InnerBackend>,
    recent: Value,

    league: League,
    league_at: Instant,
    history_at: Instant,
}

impl<B: AutodiffBackend> LiveEngine<B> {
    /// Arranque: baja el histórico, entrena y trae las estadísticas.
    pub fn new(
        device: B::Device,
        seasons: &[i32],
        min_gp: usize,
        days: i64,
        cfg: TrainConfig,
    ) -> Result<Self, String> {
        println!("Descargando historico...");
        let history = mlb::fetch_games(seasons, Freshness::Live)?;
        let ultimo = history.last().map(|g| g.date.as_str()).unwrap_or("?");
        println!("{} partidos reales (ultimo: {ultimo}).", history.len());

        let mut fb = FeatureBuilder::new();
        let samples = fb.build(&history);

        println!("Entrenando la red...");
        let (model, recent) = Self::fit(&device, &samples, min_gp, &cfg);

        let league = stats::fetch_league(mlb::current_year())?;

        Ok(Self {
            device,
            cfg,
            min_gp,
            days,
            seasons: seasons.to_vec(),
            history,
            samples,
            fb,
            model,
            recent,
            league,
            league_at: Instant::now(),
            history_at: Instant::now(),
        })
    }

    /// Entrena el modelo de producción y, de paso, mide su acierto reciente
    /// con un segundo modelo que no vio los últimos 30 días.
    fn fit(
        device: &B::Device,
        samples: &[Sample],
        min_gp: usize,
        cfg: &TrainConfig,
    ) -> (Trained<B::InnerBackend>, Value) {
        let hist: Vec<Sample> = samples.iter().filter(|s| usable(s, min_gp)).cloned().collect();

        let cutoff = (Local::now().date_naive() - Duration::days(30))
            .format("%Y-%m-%d")
            .to_string();
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

            // Acierto real por banda de confianza: es la referencia honesta
            // para juzgar una selección "solo por encima del X %".
            let bandas: Vec<Value> = [55.0f32, 60.0, 65.0, 70.0]
                .iter()
                .map(|&u| {
                    let (mut n, mut ok) = (0usize, 0usize);
                    for (p, y) in probs.iter().zip(&labels) {
                        if p.max(1.0 - p) * 100.0 >= u {
                            n += 1;
                            if (*p >= 0.5) == (*y > 0.5) {
                                ok += 1;
                            }
                        }
                    }
                    json!({
                        "umbral": u,
                        "partidos": n,
                        "acierto": if n > 0 { (ok as f64 / n as f64) * 100.0 } else { 0.0 },
                    })
                })
                .collect();

            json!({
                "desde": cutoff,
                "partidos": met.n,
                "aciertoRed": pct(met.acc),
                "aciertoElo": pct(accuracy(&elo, &labels)),
                "logloss": met.logloss as f64,
                "brier": met.brier as f64,
                "bandas": bandas,
            })
        } else {
            json!(null)
        };

        let cut = (hist.len() as f32 * 0.9) as usize;
        let (tr, va) = hist.split_at(cut);
        (train::<B>(device, tr, va, cfg), recent)
    }

    fn rebuild_state(&mut self) {
        self.history
            .sort_by(|a, b| a.date.cmp(&b.date).then(a.game_pk.cmp(&b.game_pk)));
        self.fb = FeatureBuilder::new();
        self.samples = self.fb.build(&self.history);
    }

    fn retrain(&mut self) {
        let (m, r) = Self::fit(&self.device, &self.samples, self.min_gp, &self.cfg);
        self.model = m;
        self.recent = r;
    }

    /// Incorpora al histórico los partidos de la ventana que ya terminaron.
    /// Devuelve cuántos entraron nuevos.
    fn absorb_finals(&mut self, window: &[ScheduledGame]) -> usize {
        let known: std::collections::HashSet<i64> =
            self.history.iter().map(|g| g.game_pk).collect();
        let mut added = 0;

        for g in window {
            if !g.is_final || known.contains(&g.game_pk) {
                continue;
            }
            let (Some(hs), Some(aws)) = (g.home_score, g.away_score) else {
                continue;
            };
            if hs == aws {
                continue; // un empate no sirve como etiqueta binaria
            }
            self.history.push(RawGame {
                game_pk: g.game_pk,
                date: g.date.clone(),
                season: mlb::current_year(),
                home_id: g.home_id,
                home_name: g.home_name.clone(),
                away_id: g.away_id,
                away_name: g.away_name.clone(),
                home_score: hs,
                away_score: aws,
                home_sp_id: g.home_sp_id,
                away_sp_id: g.away_sp_id,
            });
            added += 1;
        }
        added
    }

    /// Un ciclo de refresco. Barato salvo que hayan terminado partidos.
    pub fn tick(&mut self) -> Result<Value, String> {
        let from = Local::now().date_naive().format("%Y-%m-%d").to_string();

        // 1. Lo único que se baja siempre: la ventana de próximos días.
        //    Trae marcadores en vivo y abridores recién anunciados.
        let window = mlb::fetch_scheduled(&from, self.days)?;

        // 2. Cada tanto, el histórico completo por si algo se escapó.
        if self.history_at.elapsed().as_secs() > HISTORY_TTL_MIN * 60 {
            if let Ok(h) = mlb::fetch_games(&self.seasons, Freshness::Live) {
                let crecio = h.len() > self.history.len();
                self.history = h;
                self.history_at = Instant::now();
                if crecio {
                    self.rebuild_state();
                    self.retrain();
                }
            }
        }

        // 3. Partidos que acaban de terminar -> al histórico, y reentrenar.
        let nuevos = self.absorb_finals(&window);
        if nuevos > 0 {
            println!("  {nuevos} partido(s) terminado(s): reentrenando la red...");
            self.rebuild_state();
            self.retrain();
        }

        // 4. Plantillas y estadísticas de jugadores, de vez en cuando.
        if self.league_at.elapsed().as_secs() > LEAGUE_TTL_MIN * 60 {
            if let Ok(l) = stats::fetch_league(mlb::current_year()) {
                self.league = l;
                self.league_at = Instant::now();
            }
        }

        Ok(self.payload(&window))
    }

    fn payload(&self, window: &[ScheduledGame]) -> Value {
        let season = mlb::current_year();
        let snaps = self.fb.snapshot();

        // Un solo lote de inferencia para toda la ventana.
        let mut rows: Vec<(&ScheduledGame, Sample)> = Vec::new();
        for g in window {
            let Some((feats, elo_p)) =
                self.fb
                    .features_for(g.home_id, g.away_id, &g.date, g.home_sp_id, g.away_sp_id)
            else {
                continue;
            };
            rows.push((
                g,
                Sample {
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
                },
            ));
        }

        let batch: Vec<Sample> = rows.iter().map(|(_, s)| s.clone()).collect();
        let probs = self.model.predict(&batch);

        let mut dates: Vec<Value> = Vec::new();
        let mut cur_date = String::new();
        let mut cur_games: Vec<Value> = Vec::new();
        let (mut n_conf, mut n_final, mut n_live) = (0usize, 0usize, 0usize);

        for (idx, (g, s)) in rows.iter().enumerate() {
            if g.date != cur_date {
                if !cur_date.is_empty() {
                    dates.push(json!({ "date": cur_date, "games": cur_games }));
                }
                cur_date = g.date.clone();
                cur_games = Vec::new();
            }

            if g.pitchers_confirmed { n_conf += 1; }
            if g.is_final { n_final += 1; }
            if g.is_live { n_live += 1; }

            // Sin ambos abridores confirmados NO se publica probabilidad: el
            // lanzador que abre mueve el pronóstico lo bastante como para que
            // dar un número sería engañoso.
            let show = g.pitchers_confirmed;
            let p_home = probs[idx];

            let hit = match (show, g.is_final, g.home_score, g.away_score) {
                (true, true, Some(hs), Some(aws)) if hs != aws => {
                    Some((p_home >= 0.5) == (hs > aws))
                }
                _ => None,
            };

            cur_games.push(json!({
                "gamePk": g.game_pk,
                "start": g.start_utc,
                "venue": g.venue,
                "status": g.status,
                "confirmed": g.pitchers_confirmed,
                "final": g.is_final,
                "live": g.is_live,
                "homeScore": g.home_score,
                "awayScore": g.away_score,
                "hit": hit,
                "pHome": if show { json!(pct(p_home)) } else { json!(null) },
                "pAway": if show { json!(100.0 - pct(p_home)) } else { json!(null) },
                "eloHome": if show { json!(pct(s.elo_prob)) } else { json!(null) },
                "home": team_json(self.league.teams.get(&g.home_id), snaps.get(&g.home_id), &g.home_name),
                "away": team_json(self.league.teams.get(&g.away_id), snaps.get(&g.away_id), &g.away_name),
                "homePitcher": pitcher_json(
                    self.league.pitchers.get(&g.home_sp_id), &g.home_sp_name, g.home_sp_id),
                "awayPitcher": pitcher_json(
                    self.league.pitchers.get(&g.away_sp_id), &g.away_sp_name, g.away_sp_id),
            }));
        }
        if !cur_date.is_empty() {
            dates.push(json!({ "date": cur_date, "games": cur_games }));
        }

        let mut standings: Vec<Value> = snaps
            .values()
            .filter(|s| s.wins + s.losses > 0)
            .map(|s| {
                let info = self.league.teams.get(&s.id);
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
                    "homeW": s.home_wins,
                    "homeL": s.home_losses,
                    "awayW": s.away_wins,
                    "awayL": s.away_losses,
                    "batAvg": info.map(|t| t.bat_avg.clone()).unwrap_or_else(|| "-".into()),
                    "ops": info.map(|t| t.ops.clone()).unwrap_or_else(|| "-".into()),
                    "era": info.map(|t| t.era.clone()).unwrap_or_else(|| "-".into()),
                })
            })
            .collect();
        standings
            .sort_by(|a, b| b["elo"].as_i64().unwrap_or(0).cmp(&a["elo"].as_i64().unwrap_or(0)));

        json!({
            "generado": Local::now().format("%H:%M:%S").to_string(),
            "hoy": Local::now().date_naive().format("%Y-%m-%d").to_string(),
            "ultimoPartido": self.history.last().map(|g| g.date.clone()).unwrap_or_default(),
            "temporada": season,
            "modelo": {
                "partidosEntrenamiento": self.samples.len(),
                "caracteristicas": crate::features::NUM_FEATURES,
                "epocas": self.model.epochs_run,
                "reciente": self.recent,
            },
            "resumen": {
                "partidos": rows.len(),
                "confirmados": n_conf,
                "sinConfirmar": rows.len() - n_conf,
                "jugados": n_final,
                "enVivo": n_live,
                "dias": dates.len(),
            },
            "fechas": dates,
            "tabla": standings,
        })
    }
}
