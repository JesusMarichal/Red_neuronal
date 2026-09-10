//! Cliente de la MLB Stats API (statsapi.mlb.com).
//!
//! Descarga resultados REALES de partidos, temporada por temporada, y los
//! cachea en disco para no volver a pedirlos. No requiere API key.

use std::fs;
use std::path::{Path, PathBuf};
use serde_json::Value;

/// Un partido real, ya normalizado y sin ruido de la API.
#[derive(Debug, Clone)]
pub struct RawGame {
    pub game_pk: i64,
    pub date: String,        // "YYYY-MM-DD"
    pub season: i32,
    pub home_id: i64,
    pub home_name: String,
    pub away_id: i64,
    pub away_name: String,
    pub home_score: i32,
    pub away_score: i32,
    pub home_sp_id: i64,     // pitcher abridor local (0 si desconocido)
    pub away_sp_id: i64,
}

impl RawGame {
    pub fn home_wins(&self) -> bool {
        self.home_score > self.away_score
    }
}

pub const CACHE_DIR: &str = "data/raw";

/// Lee un archivo de cache si existe y es más reciente que `max_age_hours`.
/// Con `max_age_hours = None` vale cualquier antigüedad.
pub fn cache_read(path: &Path, max_age_hours: Option<u64>) -> Option<String> {
    let meta = fs::metadata(path).ok()?;
    if let Some(max) = max_age_hours {
        let age = meta.modified().ok()?.elapsed().ok()?;
        if age.as_secs() > max * 3600 {
            return None;
        }
    }
    fs::read_to_string(path).ok()
}

/// GET con cache en disco. Devuelve el JSON ya parseado.
pub fn get_json(url: &str, cache_file: &str, max_age_hours: Option<u64>) -> Result<Value, String> {
    let path = Path::new(CACHE_DIR).join(cache_file);
    if let Some(txt) = cache_read(&path, max_age_hours) {
        if let Ok(v) = serde_json::from_str(&txt) {
            return Ok(v);
        }
    }

    let mut resp = ureq::get(url)
        .call()
        .map_err(|e| format!("fallo HTTP en {url}: {e}"))?;
    let body = resp
        .body_mut()
        .with_config()
        .limit(256 * 1024 * 1024)
        .read_to_string()
        .map_err(|e| format!("fallo leyendo {url}: {e}"))?;

    fs::create_dir_all(CACHE_DIR).map_err(|e| e.to_string())?;
    fs::write(&path, &body).map_err(|e| e.to_string())?;
    serde_json::from_str(&body).map_err(|e| format!("JSON invalido de {url}: {e}"))
}

fn cache_path(season: i32) -> PathBuf {
    Path::new(CACHE_DIR).join(format!("schedule_{season}.json"))
}

/// Descarga (o lee del cache) el calendario completo de temporada regular.
fn fetch_season_json(season: i32, refresh: bool) -> Result<Value, String> {
    let url = format!(
        "https://statsapi.mlb.com/api/v1/schedule?sportId=1&gameType=R\
         &startDate={season}-02-01&endDate={season}-11-30&hydrate=probablePitcher"
    );
    // Las temporadas cerradas ya no cambian; la actual se refresca cada 6 h.
    let max_age = if refresh {
        Some(0)
    } else if season >= current_year() {
        Some(6)
    } else {
        None
    };
    if cache_read(&cache_path(season), max_age).is_none() {
        println!("  -> GET temporada {season} ...");
    }
    get_json(&url, &format!("schedule_{season}.json"), max_age)
}

/// Año en curso según el reloj del sistema.
pub fn current_year() -> i32 {
    chrono::Local::now().format("%Y").to_string().parse().unwrap_or(2026)
}

fn pitcher_id(side: &Value) -> i64 {
    side.get("probablePitcher")
        .and_then(|p| p.get("id"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
}

/// Convierte el JSON del calendario en partidos finalizados.
fn parse_season(json: &Value, season: i32) -> Vec<RawGame> {
    let mut out = Vec::new();
    let Some(dates) = json.get("dates").and_then(|d| d.as_array()) else {
        return out;
    };

    for date_entry in dates {
        let Some(games) = date_entry.get("games").and_then(|g| g.as_array()) else {
            continue;
        };
        for g in games {
            // Solo partidos terminados de verdad (F = Final, O = Game Over).
            let state = g
                .pointer("/status/codedGameState")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if state != "F" && state != "O" {
                continue;
            }

            let (Some(home), Some(away)) =
                (g.pointer("/teams/home"), g.pointer("/teams/away"))
            else {
                continue;
            };

            let (Some(hs), Some(aws)) = (
                home.get("score").and_then(|v| v.as_i64()),
                away.get("score").and_then(|v| v.as_i64()),
            ) else {
                continue;
            };
            // Un empate no sirve como etiqueta binaria (rarísimo, pero existe).
            if hs == aws {
                continue;
            }

            out.push(RawGame {
                game_pk: g.get("gamePk").and_then(|v| v.as_i64()).unwrap_or(0),
                date: g
                    .get("officialDate")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                season,
                home_id: home.pointer("/team/id").and_then(|v| v.as_i64()).unwrap_or(0),
                home_name: home
                    .pointer("/team/name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?")
                    .to_string(),
                away_id: away.pointer("/team/id").and_then(|v| v.as_i64()).unwrap_or(0),
                away_name: away
                    .pointer("/team/name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?")
                    .to_string(),
                home_score: hs as i32,
                away_score: aws as i32,
                home_sp_id: pitcher_id(home),
                away_sp_id: pitcher_id(away),
            });
        }
    }
    out
}

/// Trae todas las temporadas pedidas, ordenadas cronológicamente.
pub fn fetch_games(seasons: &[i32], refresh: bool) -> Result<Vec<RawGame>, String> {
    let mut all = Vec::new();
    for &s in seasons {
        let json = fetch_season_json(s, refresh)?;
        let games = parse_season(&json, s);
        println!("  temporada {s}: {} partidos finalizados", games.len());
        all.extend(games);
    }
    // Orden cronológico estricto: es la base de todo el resto del pipeline.
    all.sort_by(|a, b| a.date.cmp(&b.date).then(a.game_pk.cmp(&b.game_pk)));
    Ok(all)
}

/// Un partido programado (todavía sin resultado firme).
#[derive(Debug, Clone)]
pub struct ScheduledGame {
    pub game_pk: i64,
    pub date: String,
    pub start_utc: String,
    pub home_id: i64,
    pub home_name: String,
    pub away_id: i64,
    pub away_name: String,
    pub home_sp_id: i64,
    pub home_sp_name: String,
    pub away_sp_id: i64,
    pub away_sp_name: String,
    pub venue: String,
    pub status: String,
    /// Ambos abridores anunciados por los equipos.
    pub pitchers_confirmed: bool,
    /// Marcador real, si el partido ya terminó.
    pub home_score: Option<i32>,
    pub away_score: Option<i32>,
    pub is_final: bool,
}

fn pitcher_name(side: &Value) -> String {
    side.get("probablePitcher")
        .and_then(|p| p.get("fullName"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

/// Calendario de los próximos `days` días desde `from` (formato YYYY-MM-DD).
pub fn fetch_scheduled(from: &str, days: i64) -> Result<Vec<ScheduledGame>, String> {
    let start = chrono::NaiveDate::parse_from_str(from, "%Y-%m-%d")
        .map_err(|e| format!("fecha invalida {from}: {e}"))?;
    let end = start + chrono::Duration::days(days.max(1) - 1);
    let end = end.format("%Y-%m-%d").to_string();

    let url = format!(
        "https://statsapi.mlb.com/api/v1/schedule?sportId=1&gameType=R\
         &startDate={from}&endDate={end}&hydrate=probablePitcher,venue,team"
    );
    // Los abridores se anuncian con poca antelación: cache corta.
    let json = get_json(&url, &format!("upcoming_{from}_{days}.json"), Some(1))?;

    let mut out = Vec::new();
    let Some(dates) = json.get("dates").and_then(|d| d.as_array()) else {
        return Ok(out);
    };

    for entry in dates {
        let Some(games) = entry.get("games").and_then(|g| g.as_array()) else {
            continue;
        };
        for g in games {
            let (Some(home), Some(away)) = (g.pointer("/teams/home"), g.pointer("/teams/away"))
            else {
                continue;
            };
            let hs = pitcher_id(home);
            let aws = pitcher_id(away);
            let state = g
                .pointer("/status/codedGameState")
                .and_then(|v| v.as_str())
                .unwrap_or("S");
            let is_final = state == "F" || state == "O";
            out.push(ScheduledGame {
                game_pk: g.get("gamePk").and_then(|v| v.as_i64()).unwrap_or(0),
                date: g.get("officialDate").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                start_utc: g.get("gameDate").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                home_id: home.pointer("/team/id").and_then(|v| v.as_i64()).unwrap_or(0),
                home_name: home.pointer("/team/name").and_then(|v| v.as_str()).unwrap_or("?").to_string(),
                away_id: away.pointer("/team/id").and_then(|v| v.as_i64()).unwrap_or(0),
                away_name: away.pointer("/team/name").and_then(|v| v.as_str()).unwrap_or("?").to_string(),
                home_sp_id: hs,
                home_sp_name: pitcher_name(home),
                away_sp_id: aws,
                away_sp_name: pitcher_name(away),
                venue: g.pointer("/venue/name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                status: g
                    .pointer("/status/detailedState")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Scheduled")
                    .to_string(),
                pitchers_confirmed: hs != 0 && aws != 0,
                home_score: home.get("score").and_then(|v| v.as_i64()).map(|v| v as i32),
                away_score: away.get("score").and_then(|v| v.as_i64()).map(|v| v as i32),
                is_final,
            });
        }
    }

    out.sort_by(|a, b| a.date.cmp(&b.date).then(a.start_utc.cmp(&b.start_utc)));
    Ok(out)
}
