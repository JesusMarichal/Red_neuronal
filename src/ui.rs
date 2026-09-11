//! Interfaz web: genera el panel HTML y lo sirve en localhost, refrescándolo
//! en segundo plano cada cierto tiempo.

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

use serde_json::Value;

const TEMPLATE: &str = include_str!("dashboard.html");

/// Inyecta el payload en la plantilla. El HTML sin datos también es válido,
/// así que el marcador es un `null` dentro de un comentario.
pub fn render(payload: &Value) -> String {
    TEMPLATE.replace("/*__DATA__*/null", &payload.to_string())
}

pub fn write_file(path: &str, payload: &Value) -> std::io::Result<()> {
    fs::write(path, render(payload))
}

/// Lo que sirve el servidor en un momento dado.
struct Snapshot {
    html: String,
    json: String,
}

impl Snapshot {
    fn of(payload: &Value) -> Self {
        Self {
            html: render(payload),
            json: payload.to_string(),
        }
    }
}

fn respond(stream: &mut TcpStream, status: &str, ctype: &str, body: &[u8]) {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\n\
         Cache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

fn handle(stream: &mut TcpStream, state: &RwLock<Snapshot>) {
    let mut line = String::new();
    if BufReader::new(&*stream).read_line(&mut line).is_err() {
        return;
    }
    let path = line.split_whitespace().nth(1).unwrap_or("/");

    // Se clona lo mínimo para no retener el lock mientras se escribe al socket.
    let body = {
        let snap = match state.read() {
            Ok(s) => s,
            Err(e) => e.into_inner(),
        };
        match path {
            "/" | "/index.html" => Some((snap.html.clone(), "text/html; charset=utf-8")),
            "/api" | "/api.json" => Some((snap.json.clone(), "application/json; charset=utf-8")),
            _ => None,
        }
    };

    match body {
        Some((b, ctype)) => respond(stream, "200 OK", ctype, b.as_bytes()),
        None => respond(stream, "404 Not Found", "text/plain; charset=utf-8", b"no existe"),
    }
}

fn open_browser(url: &str) {
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("cmd").args(["/C", "start", "", url]).spawn();
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(url).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let _ = std::process::Command::new("xdg-open").arg(url).spawn();
}

/// Sirve el panel y lo refresca en segundo plano cada `interval`.
///
/// `refresh` se ejecuta en su propio hilo; si falla, se avisa por consola y se
/// sigue sirviendo la última versión buena.
pub fn serve_live<F>(
    initial: Value,
    port: u16,
    interval: Duration,
    mut refresh: F,
) -> std::io::Result<()>
where
    F: FnMut() -> Result<Value, String> + Send + 'static,
{
    let state = Arc::new(RwLock::new(Snapshot::of(&initial)));
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    let url = format!("http://127.0.0.1:{port}");

    println!("\n  Panel disponible en {url}");
    println!("  JSON crudo en {url}/api");
    println!("  Se actualiza solo cada {} s. Ctrl+C para detener.\n", interval.as_secs());
    open_browser(&url);

    let bg = Arc::clone(&state);
    thread::spawn(move || loop {
        thread::sleep(interval);
        match refresh() {
            Ok(v) => {
                let resumen = v["resumen"].clone();
                let snap = Snapshot::of(&v);
                match bg.write() {
                    Ok(mut w) => *w = snap,
                    Err(e) => *e.into_inner() = snap,
                }
                println!(
                    "  [{}] actualizado: {} partidos, {} en vivo, {} ya jugados",
                    v["generado"].as_str().unwrap_or("?"),
                    resumen["partidos"],
                    resumen["enVivo"],
                    resumen["jugados"],
                );
            }
            Err(e) => eprintln!("  fallo al actualizar (se mantiene lo anterior): {e}"),
        }
    });

    // Un hilo por conexión: si un cliente abre el socket y no lee (el navegador
    // hace conexiones especulativas), no debe bloquear al resto. Los timeouts
    // evitan que una conexión muerta deje el hilo colgado para siempre.
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let st = Arc::clone(&state);
                thread::spawn(move || {
                    let mut s = s;
                    let _ = s.set_read_timeout(Some(Duration::from_secs(10)));
                    let _ = s.set_write_timeout(Some(Duration::from_secs(30)));
                    handle(&mut s, &st);
                });
            }
            Err(e) => eprintln!("conexion fallida: {e}"),
        }
    }
    Ok(())
}
