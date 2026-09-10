//! Ingeniería de características estrictamente cronológica.
//!
//! Regla de oro: para predecir el partido del día D solo se usa información
//! disponible ANTES de que empiece ese partido. El estado se actualiza
//! únicamente después de emitir la fila, así que no hay fuga de datos
//! (data leakage) por construcción.

use std::collections::{HashMap, VecDeque};
use chrono::NaiveDate;
use crate::mlb::RawGame;

pub const FEATURE_NAMES: [&str; 25] = [
    "elo_diff",        "elo_prob",       "home_wpct",      "away_wpct",     "wpct_diff",
    "home_rs_pg",      "home_ra_pg",     "away_rs_pg",     "away_ra_pg",    "home_rundiff_pg",
    "away_rundiff_pg", "home_form10",    "away_form10",    "home_rs10",     "home_ra10",
    "away_rs10",       "away_ra10",      "home_pyth",      "away_pyth",     "home_rest",
    "away_rest",       "home_sp_ra",     "away_sp_ra",     "home_sp_exp",   "away_sp_exp",
];
pub const NUM_FEATURES: usize = FEATURE_NAMES.len();

/// Una fila lista para entrenar: características previas + resultado real.
#[derive(Debug, Clone)]
pub struct Sample {
    pub game_pk: i64,
    pub date: String,
    pub season: i32,
    pub home_name: String,
    pub away_name: String,
    pub home_score: i32,
    pub away_score: i32,
    pub label: usize,           // 1 = ganó el local
    pub features: Vec<f32>,
    pub home_gp: usize,         // partidos jugados por el local antes de este
    pub away_gp: usize,
    pub elo_prob: f32,          // guardado aparte para el baseline Elo
}

// ---------------------------------------------------------------- Elo
const ELO_START: f64 = 1500.0;
const ELO_K: f64 = 6.0;
const ELO_HFA: f64 = 24.0;       // ventaja de jugar en casa, en puntos Elo
const ELO_CARRYOVER: f64 = 0.72; // regresión a la media entre temporadas

fn elo_expected(home: f64, away: f64) -> f64 {
    1.0 / (1.0 + 10f64.powf(-((home + ELO_HFA) - away) / 400.0))
}

// ------------------------------------------------------------- Estado
#[derive(Clone)]
struct TeamState {
    name: String,
    elo: f64,
    season: i32,
    gp: usize,
    wins: usize,
    rs: f64,
    ra: f64,
    last_results: VecDeque<f32>,
    last_rs: VecDeque<f32>,
    last_ra: VecDeque<f32>,
    last_date: Option<NaiveDate>,
}

impl TeamState {
    fn new() -> Self {
        Self {
            name: String::new(),
            elo: ELO_START,
            season: 0,
            gp: 0,
            wins: 0,
            rs: 0.0,
            ra: 0.0,
            last_results: VecDeque::new(),
            last_rs: VecDeque::new(),
            last_ra: VecDeque::new(),
            last_date: None,
        }
    }

    /// Nueva temporada: se reinician los acumulados y el Elo regresa a la media.
    fn roll_season(&mut self, season: i32) {
        if self.season != season {
            if self.season != 0 {
                self.elo = ELO_START + ELO_CARRYOVER * (self.elo - ELO_START);
            }
            self.season = season;
            self.gp = 0;
            self.wins = 0;
            self.rs = 0.0;
            self.ra = 0.0;
            self.last_results.clear();
            self.last_rs.clear();
            self.last_ra.clear();
        }
    }

    /// Porcentaje de victorias con shrinkage bayesiano hacia .500.
    fn wpct(&self) -> f64 {
        (self.wins as f64 + 8.0 * 0.5) / (self.gp as f64 + 8.0)
    }
    fn rs_pg(&self) -> f64 {
        if self.gp == 0 { 4.4 } else { self.rs / self.gp as f64 }
    }
    fn ra_pg(&self) -> f64 {
        if self.gp == 0 { 4.4 } else { self.ra / self.gp as f64 }
    }
    fn pyth(&self) -> f64 {
        if self.gp == 0 { return 0.5; }
        let (rs, ra) = (self.rs.max(0.1), self.ra.max(0.1));
        let e = 1.83;
        rs.powf(e) / (rs.powf(e) + ra.powf(e))
    }
    fn mean(dq: &VecDeque<f32>, default: f64) -> f64 {
        if dq.is_empty() { default } else { dq.iter().sum::<f32>() as f64 / dq.len() as f64 }
    }
    fn rest_days(&self, date: NaiveDate) -> f64 {
        match self.last_date {
            Some(d) => ((date - d).num_days() as f64).clamp(0.0, 7.0),
            None => 3.0,
        }
    }

    fn update(&mut self, date: NaiveDate, won: bool, scored: i32, allowed: i32) {
        self.gp += 1;
        if won { self.wins += 1; }
        self.rs += scored as f64;
        self.ra += allowed as f64;
        push_capped(&mut self.last_results, if won { 1.0 } else { 0.0 }, 10);
        push_capped(&mut self.last_rs, scored as f32, 10);
        push_capped(&mut self.last_ra, allowed as f32, 10);
        self.last_date = Some(date);
    }
}

fn push_capped(dq: &mut VecDeque<f32>, v: f32, cap: usize) {
    dq.push_back(v);
    while dq.len() > cap { dq.pop_front(); }
}

/// Forma reciente del pitcher abridor: carreras que permitió su equipo
/// en las últimas aperturas de ese lanzador.
#[derive(Clone)]
struct PitcherState {
    starts: usize,
    recent_ra: VecDeque<f32>,
}

impl PitcherState {
    fn new() -> Self { Self { starts: 0, recent_ra: VecDeque::new() } }
    fn ra(&self) -> f64 { TeamState::mean(&self.recent_ra, 4.4) }
}

// --------------------------------------------------------- Constructor

/// Vector de características a partir del estado previo de ambos equipos.
/// Lo usan tanto el histórico como la predicción de partidos futuros.
fn feature_vector(
    h: &TeamState,
    a: &TeamState,
    hp: &PitcherState,
    ap: &PitcherState,
    date: NaiveDate,
) -> Vec<f32> {
    let feats = vec![
        (((h.elo + ELO_HFA) - a.elo) / 400.0) as f32,
        elo_expected(h.elo, a.elo) as f32,
        h.wpct() as f32,
        a.wpct() as f32,
        (h.wpct() - a.wpct()) as f32,
        h.rs_pg() as f32,
        h.ra_pg() as f32,
        a.rs_pg() as f32,
        a.ra_pg() as f32,
        (h.rs_pg() - h.ra_pg()) as f32,
        (a.rs_pg() - a.ra_pg()) as f32,
        TeamState::mean(&h.last_results, 0.5) as f32,
        TeamState::mean(&a.last_results, 0.5) as f32,
        TeamState::mean(&h.last_rs, 4.4) as f32,
        TeamState::mean(&h.last_ra, 4.4) as f32,
        TeamState::mean(&a.last_rs, 4.4) as f32,
        TeamState::mean(&a.last_ra, 4.4) as f32,
        h.pyth() as f32,
        a.pyth() as f32,
        h.rest_days(date) as f32,
        a.rest_days(date) as f32,
        hp.ra() as f32,
        ap.ra() as f32,
        (hp.starts.min(30) as f32) / 30.0,
        (ap.starts.min(30) as f32) / 30.0,
    ];
    debug_assert_eq!(feats.len(), NUM_FEATURES);
    feats
}

/// Foto del estado actual de un equipo, para mostrar en la interfaz.
#[derive(Debug, Clone)]
pub struct TeamSnapshot {
    pub id: i64,
    pub name: String,
    pub elo: f64,
    pub wins: usize,
    pub losses: usize,
    pub rs_pg: f64,
    pub ra_pg: f64,
    pub form10: f64,
}

pub struct FeatureBuilder {
    teams: HashMap<i64, TeamState>,
    pitchers: HashMap<i64, PitcherState>,
}

impl FeatureBuilder {
    pub fn new() -> Self {
        Self { teams: HashMap::new(), pitchers: HashMap::new() }
    }

    /// Recorre los partidos en orden y devuelve una fila por partido.
    pub fn build(&mut self, games: &[RawGame]) -> Vec<Sample> {
        let mut out = Vec::with_capacity(games.len());

        for g in games {
            let date = match NaiveDate::parse_from_str(&g.date, "%Y-%m-%d") {
                Ok(d) => d,
                Err(_) => continue,
            };

            {
                let t = self.teams.entry(g.home_id).or_insert_with(TeamState::new);
                t.name = g.home_name.clone();
                t.roll_season(g.season);
            }
            {
                let t = self.teams.entry(g.away_id).or_insert_with(TeamState::new);
                t.name = g.away_name.clone();
                t.roll_season(g.season);
            }
            self.pitchers.entry(g.home_sp_id).or_insert_with(PitcherState::new);
            self.pitchers.entry(g.away_sp_id).or_insert_with(PitcherState::new);

            // --- Lectura del estado PREVIO al partido ---
            let h = self.teams[&g.home_id].clone();
            let a = self.teams[&g.away_id].clone();
            let hp = self.pitchers[&g.home_sp_id].clone();
            let ap = self.pitchers[&g.away_sp_id].clone();

            let elo_p = elo_expected(h.elo, a.elo);
            let feats = feature_vector(&h, &a, &hp, &ap, date);
            debug_assert_eq!(feats.len(), NUM_FEATURES);

            let home_won = g.home_wins();
            out.push(Sample {
                game_pk: g.game_pk,
                date: g.date.clone(),
                season: g.season,
                home_name: g.home_name.clone(),
                away_name: g.away_name.clone(),
                home_score: g.home_score,
                away_score: g.away_score,
                label: if home_won { 1 } else { 0 },
                features: feats,
                home_gp: h.gp,
                away_gp: a.gp,
                elo_prob: elo_p as f32,
            });

            // --- Solo AHORA se incorpora el resultado al estado ---
            let margin = (g.home_score - g.away_score).abs() as f64;
            let elo_diff_winner = if home_won {
                h.elo + ELO_HFA - a.elo
            } else {
                a.elo - h.elo - ELO_HFA
            };
            let mov = (margin + 1.0).ln() * (2.2 / (elo_diff_winner * 0.001 + 2.2));
            let delta = ELO_K * mov * ((if home_won { 1.0 } else { 0.0 }) - elo_p);
            self.teams.get_mut(&g.home_id).unwrap().elo += delta;
            self.teams.get_mut(&g.away_id).unwrap().elo -= delta;

            self.teams.get_mut(&g.home_id).unwrap()
                .update(date, home_won, g.home_score, g.away_score);
            self.teams.get_mut(&g.away_id).unwrap()
                .update(date, !home_won, g.away_score, g.home_score);

            let hps = self.pitchers.get_mut(&g.home_sp_id).unwrap();
            hps.starts += 1;
            push_capped(&mut hps.recent_ra, g.away_score as f32, 8);
            let aps = self.pitchers.get_mut(&g.away_sp_id).unwrap();
            aps.starts += 1;
            push_capped(&mut aps.recent_ra, g.home_score as f32, 8);
        }

        out
    }

    /// Características de un partido que TODAVÍA no se juega, usando el
    /// estado actual (después de todos los partidos ya procesados).
    ///
    /// Un abridor desconocido (`sp_id == 0` o sin historial) recibe el
    /// prior de liga en lugar de la bolsa común de pitchers sin identificar.
    pub fn features_for(
        &self,
        home_id: i64,
        away_id: i64,
        date: &str,
        home_sp: i64,
        away_sp: i64,
    ) -> Option<(Vec<f32>, f32)> {
        let date = NaiveDate::parse_from_str(date, "%Y-%m-%d").ok()?;
        let h = self.teams.get(&home_id)?;
        let a = self.teams.get(&away_id)?;

        let prior = PitcherState::new();
        let pick = |id: i64| -> PitcherState {
            if id == 0 {
                return prior.clone();
            }
            self.pitchers.get(&id).cloned().unwrap_or_else(PitcherState::new)
        };

        let feats = feature_vector(h, a, &pick(home_sp), &pick(away_sp), date);
        Some((feats, elo_expected(h.elo, a.elo) as f32))
    }

    /// Estado actual de cada equipo, para mostrarlo en la interfaz.
    pub fn snapshot(&self) -> HashMap<i64, TeamSnapshot> {
        self.teams
            .iter()
            .map(|(&id, t)| {
                (
                    id,
                    TeamSnapshot {
                        id,
                        name: t.name.clone(),
                        elo: t.elo,
                        wins: t.wins,
                        losses: t.gp - t.wins,
                        rs_pg: t.rs_pg(),
                        ra_pg: t.ra_pg(),
                        form10: TeamState::mean(&t.last_results, 0.5),
                    },
                )
            })
            .collect()
    }
}
