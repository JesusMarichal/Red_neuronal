//! Interfaz web: genera el panel HTML y lo sirve en localhost.

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
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

fn handle(stream: &mut TcpStream, html: &str, json: &str) {
    let mut line = String::new();
    if BufReader::new(&*stream).read_line(&mut line).is_err() {
        return;
    }
    let path = line.split_whitespace().nth(1).unwrap_or("/");

    match path {
        "/" | "/index.html" => respond(stream, "200 OK", "text/html; charset=utf-8", html.as_bytes()),
        "/api" | "/api.json" => respond(
            stream,
            "200 OK",
            "application/json; charset=utf-8",
            json.as_bytes(),
        ),
        _ => respond(stream, "404 Not Found", "text/plain; charset=utf-8", b"no existe"),
    }
}

/// Sirve el panel hasta que se corte con Ctrl+C.
pub fn serve(payload: &Value, port: u16) -> std::io::Result<()> {
    let html = render(payload);
    let json = payload.to_string();

    let listener = TcpListener::bind(("127.0.0.1", port))?;
    let url = format!("http://127.0.0.1:{port}");

    println!("\n  Panel disponible en {url}");
    println!("  JSON crudo en {url}/api");
    println!("  Ctrl+C para detener.\n");

    // Abrir el navegador por defecto (si falla, no pasa nada: la URL ya se imprimió).
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("cmd").args(["/C", "start", "", &url]).spawn();
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(&url).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let _ = std::process::Command::new("xdg-open").arg(&url).spawn();

    for stream in listener.incoming() {
        match stream {
            Ok(mut s) => handle(&mut s, &html, &json),
            Err(e) => eprintln!("conexion fallida: {e}"),
        }
    }
    Ok(())
}
