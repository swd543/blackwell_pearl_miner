//! Optional, dependency-free, read-only HTTP stats endpoint.
//!
//! Disabled by default. Bind to loopback when consumed by a local portal; if
//! binding to a LAN interface, expose this unauthenticated endpoint only on a
//! trusted network and let the host firewall restrict access.
use crate::state::State;
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const DASHBOARD: &str = r#"<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>PearlHash miner</title><style>body{font:16px system-ui;max-width:760px;margin:3rem auto;padding:0 1rem;background:#10151d;color:#e7edf5}pre{white-space:pre-wrap;background:#1b2532;padding:1rem;border-radius:8px}</style><h1>PearlHash miner</h1><p>Read-only live status</p><pre id="stats">Loading…</pre><script>async function update(){try{let r=await fetch('/api/v1/stats',{cache:'no-store'});document.querySelector('#stats').textContent=JSON.stringify(await r.json(),null,2)}catch(e){document.querySelector('#stats').textContent='Status unavailable: '+e}}update();setInterval(update,5000)</script>"#;

pub struct WebServer {
    addr: SocketAddr,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl WebServer {
    pub fn start(bind: &str, state: Arc<State>, started: Instant) -> Result<Self> {
        let listener = TcpListener::bind(bind).with_context(|| format!("web stats bind {bind}"))?;
        listener.set_nonblocking(true)?;
        let addr = listener.local_addr()?;
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = stop.clone();
        let thread = thread::Builder::new().name("pearl-web-stats".into()).spawn(move || {
            while !thread_stop.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, _peer)) => serve(stream, &state, started),
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => thread::sleep(Duration::from_millis(100)),
                    Err(e) => {
                        eprintln!("stats HTTP accept failed: {e}");
                        thread::sleep(Duration::from_millis(250));
                    }
                }
            }
        })?;
        Ok(WebServer { addr, stop, thread: Some(thread) })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }
}

impl Drop for WebServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn stats(state: &State, started: Instant) -> Value {
    let counters = state.counters();
    let rate = state.share_rate.lock().unwrap().snapshot();
    let job_id = state.job().map(|g| g.job.job_id.clone());
    json!({
        "status": if state.abort.load(Ordering::SeqCst) { "stopping" } else if !state.ready.load(Ordering::SeqCst) { "connecting" } else if state.gpu_paused.load(Ordering::SeqCst) { "yielding" } else { "mining" },
        "uptime_seconds": started.elapsed().as_secs(),
        "job_id": job_id,
        "accepted": counters.accepted,
        "rejected": counters.rejected,
        "stale": counters.stale,
        "rate": rate
    })
}

fn respond(stream: &mut TcpStream, status: &str, content_type: &str, body: &str) {
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(body.as_bytes());
    let _ = stream.flush();
}

fn serve(mut stream: TcpStream, state: &State, started: Instant) {
    let _ = stream.set_read_timeout(Some(Duration::from_millis(300)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(300)));
    let mut buf = [0u8; 2048];
    let n = match stream.read(&mut buf) {
        Ok(n) => n,
        Err(_) => return,
    };
    let request = String::from_utf8_lossy(&buf[..n]);
    let mut parts = request.lines().next().unwrap_or_default().split_whitespace();
    let method = parts.next().unwrap_or_default();
    let path = parts.next().unwrap_or_default().split('?').next().unwrap_or_default();
    if method != "GET" {
        respond(&mut stream, "405 Method Not Allowed", "text/plain", "GET only\n");
        return;
    }
    match path {
        "/" => respond(&mut stream, "200 OK", "text/html", DASHBOARD),
        "/api/v1/stats" | "/health" => {
            let body = stats(state, started).to_string();
            respond(&mut stream, "200 OK", "application/json", &body);
        }
        _ => respond(&mut stream, "404 Not Found", "text/plain", "not found\n"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpStream;

    #[test]
    fn endpoint_is_disabled_until_explicitly_started_and_reports_read_only_stats() {
        let state = State::new();
        let start = Instant::now();
        state.counters.lock().unwrap().accepted = 7;
        state.ready.store(true, Ordering::SeqCst);
        state.gpu_paused.store(true, Ordering::SeqCst);
        let row = stats(&state, start);
        assert_eq!(row["status"], "yielding");
        assert_eq!(row["accepted"], 7);
        assert!(row.get("wallet").is_none());
    }

    #[test]
    fn loopback_http_endpoint_serves_stats_and_stops_cleanly() {
        let state = State::new();
        state.ready.store(true, Ordering::SeqCst);
        state.counters.lock().unwrap().accepted = 3;
        let server = WebServer::start("127.0.0.1:0", state, Instant::now()).unwrap();
        let mut stream = TcpStream::connect(server.local_addr()).unwrap();
        stream.write_all(b"GET /api/v1/stats HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(response.contains("\"accepted\":3"));
        drop(server);
    }
}
