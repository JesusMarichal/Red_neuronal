//! Estadísticas reales de la temporada en curso: totales por equipo,
//! bateadores del roster activo y lanzadores con sus números.
//!
//! Todo sale de la MLB Stats API y se cachea 6 horas, así que la interfaz
//! muestra datos actualizados sin martillar el servidor.

use std::collections::HashMap;
use serde_json::Value;
use crate::mlb::get_json;

const MAX_AGE_H: Option<u64> = Some(6);

#[derive(Debug, Clone, Default)]
pub struct Hitter {
    pub name: String,
    pub pos: String,
    pub avg: String,
    pub obp: String,
    pub ops: String,
    pub hr: i64,
    pub rbi: i64,
    pub hits: i64,
    pub at_bats: i64,
    pub games: i64,
}

#[derive(Debug, Clone, Default)]
pub struct Pitcher {
    pub id: i64,
    pub name: String,
    pub era: String,
    pub whip: String,
    pub wins: i64,
    pub losses: i64,
    pub innings: String,
    pub strikeouts: i64,
    pub walks: i64,
    pub starts: i64,
}

#[derive(Debug, Clone, Default)]
pub struct TeamInfo {
    pub name: String,
    pub abbrev: String,
    // Bateo del equipo
    pub bat_avg: String,
    pub obp: String,
    pub slg: String,
    pub ops: String,
    pub runs: i64,
    pub home_runs: i64,
    // Pitcheo del equipo
    pub era: String,
    pub whip: String,
    pub strikeouts: i64,
    // Plantilla
    pub hitters: Vec<Hitter>,
    pub pitchers: Vec<Pitcher>,
}

#[derive(Debug, Default)]
pub struct League {
    pub teams: HashMap<i64, TeamInfo>,
    /// Índice global de lanzadores por id, para resolver abridores rápido.
    pub pitchers: HashMap<i64, Pitcher>,
}

// ------------------------------------------------------------- helpers
fn s(v: &Value, key: &str) -> String {
    v.get(key)
        .and_then(|x| x.as_str())
        .unwrap_or("-")
        .to_string()
}

fn i(v: &Value, key: &str) -> i64 {
    v.get(key).and_then(|x| x.as_i64()).unwrap_or(0)
}

/// Busca el bloque de stats de un grupo ("hitting" / "pitching").
fn split_of<'a>(person: &'a Value, group: &str) -> Option<&'a Value> {
    person
        .get("stats")?
        .as_array()?
        .iter()
        .find(|b| {
            b.pointer("/group/displayName").and_then(|g| g.as_str()) == Some(group)
                && b.pointer("/type/displayName").and_then(|t| t.as_str()) == Some("season")
        })?
        .get("splits")?
        .as_array()?
        .first()?
        .get("stat")
}

// ------------------------------------------------------------- fetching
fn fetch_team_totals(season: i32, group: &str) -> Result<HashMap<i64, Value>, String> {
    let url = format!(
        "https://statsapi.mlb.com/api/v1/teams/stats?season={season}&group={group}\
         &stats=season&sportIds=1"
    );
    let json = get_json(&url, &format!("team_{group}_{season}.json"), MAX_AGE_H)?;

    let mut out = HashMap::new();
    if let Some(splits) = json.pointer("/stats/0/splits").and_then(|v| v.as_array()) {
        for sp in splits {
            if let Some(id) = sp.pointer("/team/id").and_then(|v| v.as_i64()) {
                out.insert(id, sp.clone());
            }
        }
    }
    Ok(out)
}

fn fetch_roster(team_id: i64, season: i32) -> Result<(Vec<Hitter>, Vec<Pitcher>), String> {
    let url = format!(
        "https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active\
         &hydrate=person(stats(type=season,season={season}))"
    );
    let json = get_json(&url, &format!("roster_{team_id}_{season}.json"), MAX_AGE_H)?;

    let mut hitters = Vec::new();
    let mut pitchers = Vec::new();

    let Some(roster) = json.get("roster").and_then(|r| r.as_array()) else {
        return Ok((hitters, pitchers));
    };

    for entry in roster {
        let person = &entry["person"];
        let name = s(person, "fullName");
        let pos = entry
            .pointer("/position/abbreviation")
            .and_then(|v| v.as_str())
            .unwrap_or("-")
            .to_string();
        let pid = person.get("id").and_then(|v| v.as_i64()).unwrap_or(0);

        if let Some(st) = split_of(person, "hitting") {
            let ab = i(st, "atBats");
            // Sin turnos suficientes el promedio no dice nada.
            if ab >= 50 {
                hitters.push(Hitter {
                    name: name.clone(),
                    pos: pos.clone(),
                    avg: s(st, "avg"),
                    obp: s(st, "obp"),
                    ops: s(st, "ops"),
                    hr: i(st, "homeRuns"),
                    rbi: i(st, "rbi"),
                    hits: i(st, "hits"),
                    at_bats: ab,
                    games: i(st, "gamesPlayed"),
                });
            }
        }

        if let Some(st) = split_of(person, "pitching") {
            pitchers.push(Pitcher {
                id: pid,
                name: name.clone(),
                era: s(st, "era"),
                whip: s(st, "whip"),
                wins: i(st, "wins"),
                losses: i(st, "losses"),
                innings: s(st, "inningsPitched"),
                strikeouts: i(st, "strikeOuts"),
                walks: i(st, "baseOnBalls"),
                starts: i(st, "gamesStarted"),
            });
        }
    }

    // Mejor promedio de bateo primero.
    hitters.sort_by(|a, b| {
        b.avg
            .trim_start_matches('.')
            .parse::<f32>()
            .unwrap_or(0.0)
            .total_cmp(&a.avg.trim_start_matches('.').parse::<f32>().unwrap_or(0.0))
    });
    // Abridores primero, luego por ERA.
    pitchers.sort_by(|a, b| {
        b.starts.cmp(&a.starts).then(
            a.era
                .parse::<f32>()
                .unwrap_or(99.0)
                .total_cmp(&b.era.parse::<f32>().unwrap_or(99.0)),
        )
    });

    Ok((hitters, pitchers))
}

/// Descarga todo lo de la temporada: equipos, totales y plantillas.
pub fn fetch_league(season: i32) -> Result<League, String> {
    println!("Descargando estadisticas de la temporada {season}...");

    let teams_json = get_json(
        &format!("https://statsapi.mlb.com/api/v1/teams?sportId=1&season={season}"),
        &format!("teams_{season}.json"),
        MAX_AGE_H,
    )?;
    let hitting = fetch_team_totals(season, "hitting")?;
    let pitching = fetch_team_totals(season, "pitching")?;

    let mut league = League::default();

    let empty = Vec::new();
    let teams = teams_json
        .get("teams")
        .and_then(|t| t.as_array())
        .unwrap_or(&empty);

    println!("  {} equipos, bajando plantillas...", teams.len());

    for t in teams {
        let Some(id) = t.get("id").and_then(|v| v.as_i64()) else {
            continue;
        };
        let mut info = TeamInfo {
            name: s(t, "name"),
            abbrev: s(t, "abbreviation"),
            ..Default::default()
        };

        if let Some(st) = hitting.get(&id).and_then(|v| v.get("stat")) {
            info.bat_avg = s(st, "avg");
            info.obp = s(st, "obp");
            info.slg = s(st, "slg");
            info.ops = s(st, "ops");
            info.runs = i(st, "runs");
            info.home_runs = i(st, "homeRuns");
        }
        if let Some(st) = pitching.get(&id).and_then(|v| v.get("stat")) {
            info.era = s(st, "era");
            info.whip = s(st, "whip");
            info.strikeouts = i(st, "strikeOuts");
        }

        match fetch_roster(id, season) {
            Ok((h, p)) => {
                for pit in &p {
                    league.pitchers.insert(pit.id, pit.clone());
                }
                info.hitters = h;
                info.pitchers = p;
            }
            Err(e) => eprintln!("  aviso: plantilla de {} no disponible ({e})", info.name),
        }

        league.teams.insert(id, info);
    }

    let n_hit: usize = league.teams.values().map(|t| t.hitters.len()).sum();
    println!(
        "  listo: {} equipos, {} bateadores con >=50 turnos, {} lanzadores",
        league.teams.len(),
        n_hit,
        league.pitchers.len()
    );

    Ok(league)
}
