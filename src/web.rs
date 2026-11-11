//! Optional, dependency-free, read-only HTTP stats endpoint.
//!
//! Disabled by default. Network clients must belong to an explicitly configured
//! LAN CIDR; loopback is always allowed. The endpoint has no authentication.
use crate::state::State;
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const DASHBOARD: &str = r##"<!doctype html><html><head><meta charset="utf-8"><title>axiom mining</title>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"><style>
:root{color-scheme:dark}*{box-sizing:border-box}
html,body{margin:0;background:#0b0d10;color:#e8eaed;font-family:system-ui,'Segoe UI',sans-serif}
body{min-height:100dvh;display:flex;flex-direction:column}
.card{flex:1;width:100%;max-width:640px;background:#14171d;border:1px solid #262b36;border-radius:0;padding:16px;display:flex;flex-direction:column}
@media(min-width:700px){.card{flex:1;max-width:none;margin:0;padding:28px}.grid{grid-template-columns:repeat(4,1fr)}.kv{grid-template-columns:auto 1fr auto 1fr auto 1fr}pre{font-size:.78rem;line-height:1.7}}
.top{display:flex;align-items:center;gap:10px;margin:0 0 4px}h1{margin:0;font-size:1.15rem;font-weight:650;letter-spacing:.01em}.sub{color:#8a93a5;font-size:.85rem;font-weight:400}
.badge{margin-left:auto;display:inline-block;padding:.2rem .55rem;border-radius:6px;font-weight:600;font-size:.72rem;letter-spacing:.05em;background:#3a2d12;color:#f5a623;border:1px solid #6b4d17}
.badge.up{background:#12351f;color:#57d98a;border-color:#1d5c38}.badge.bad{background:#421b1b;color:#ff8d8d;border-color:#703030}
.reason{margin:6px 0 12px;font-size:.82rem;color:#a7aebb}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}
.tile{background:#1a1e26;border:1px solid #262b36;border-radius:12px;padding:8px 10px;min-width:0}.tile.wide{grid-column:span 2}
.tile .k{font-size:.66rem;letter-spacing:.07em;text-transform:uppercase;color:#8a93a5}.tile .v{font-size:1.22rem;font-weight:650;margin-top:2px;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.tile .v small{font-size:.68rem;color:#8a93a5;font-weight:400}.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 12px;font-size:.76rem;color:#8a93a5;margin-top:12px;border-top:1px solid #262b36;padding-top:10px}
.kv b{color:#c6cbd4;font-weight:500;overflow-wrap:anywhere}pre{color:#9aa2ad;font-size:.72rem;line-height:1.6;white-space:pre-wrap;font-family:ui-monospace,monospace;margin:12px 0 0;flex:1;min-height:90px;overflow-y:auto;padding-right:4px}
.hint{color:#6f7683;font-size:.7rem;margin:10px 0 0}
</style></head><body><div class="card">
<div class="top"><h1>axiom <span class="sub">Pearl / Kryptex</span></h1><span id="badge" class="badge">CONNECTING</span></div>
<p id="reason" class="reason">Waiting for miner stats…</p>
<div class="grid">
<div class="tile wide"><div class="k">Submitted rate · 30m window</div><div class="v"><span id="rate">—</span> <small>TH/s</small></div></div>
<div class="tile"><div class="k">Accepted</div><div class="v" id="accepted">—</div></div>
<div class="tile"><div class="k">Rejected</div><div class="v" id="rejected">—</div></div>
<div class="tile"><div class="k">Stale</div><div class="v" id="stale">—</div></div>
<div class="tile"><div class="k">Uptime</div><div class="v" id="uptime">—</div></div>
</div>
<div class="kv"><span>Job</span><b id="job">—</b><span>Updated</span><b id="updated">—</b></div>
<pre id="detail">Read-only · stats refresh every 5 seconds.</pre>
<p class="hint">Read-only · Pearl / Kryptex · no credentials are displayed</p>
</div><script>
const el=id=>document.getElementById(id), fmt=n=>Number.isFinite(Number(n))?Number(n).toLocaleString():"—";
function duration(s){s=Math.max(0,Math.floor(Number(s)||0));let h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h?`${h}h ${m}m`:`${m}m ${s%60}s`}
async function update(){try{const r=await fetch('/api/v1/stats',{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);const s=await r.json();const b=el('badge');
const mode={mining:['MINING','up','Mining'],yielding:['YIELDING','','Yielding GPU to other work'],connecting:['WAITING','','Connecting to pool'],stopping:['STOPPING','bad','Miner is stopping']};const v=mode[s.status]||['UNKNOWN','bad','Miner status unavailable'];b.textContent=v[0];b.className='badge '+v[1];el('reason').textContent=v[2];
el('rate').textContent=fmt(s.rate?.submitted_TH_s_30m);el('accepted').textContent=fmt(s.accepted);el('rejected').textContent=fmt(s.rejected);el('stale').textContent=fmt(s.stale);el('uptime').textContent=duration(s.uptime_seconds);el('job').textContent=s.job_id||'—';el('updated').textContent=new Date().toLocaleTimeString();el('detail').textContent=`Shares in rate window: ${fmt(s.rate?.submitted_shares_30m)}\nRate window: ${fmt(s.rate?.submitted_window_s)} seconds\nAPI: /api/v1/stats`;
}catch(e){el('badge').textContent='OFFLINE';el('badge').className='badge bad';el('reason').textContent='Stats unavailable: '+e;}}
update();setInterval(update,5000);
</script></body></html>"##;

#[derive(Clone, Copy, Debug)]
struct AllowedNet {
    address: IpAddr,
    prefix: u8,
}

impl AllowedNet {
    fn parse(value: &str) -> Result<Self> {
        let (address, prefix) = match value.split_once('/') {
            Some((address, prefix)) => (address.parse::<IpAddr>().with_context(|| format!("invalid web allow-net address: {value}"))?, prefix.parse::<u8>().with_context(|| format!("invalid web allow-net prefix: {value}"))?),
            None => {
                let address = value.parse::<IpAddr>().with_context(|| format!("invalid web allow-net address: {value}"))?;
                let prefix = if address.is_ipv4() { 32 } else { 128 };
                (address, prefix)
            }
        };
        let max = if address.is_ipv4() { 32 } else { 128 };
        if prefix > max {
            bail!("web allow-net prefix out of range: {value}");
        }
        Ok(AllowedNet { address, prefix })
    }

    fn contains(&self, peer: IpAddr) -> bool {
        match (self.address, peer) {
            (IpAddr::V4(network), IpAddr::V4(peer)) => {
                let mask = if self.prefix == 0 { 0 } else { u32::MAX << (32 - self.prefix) };
                u32::from(network) & mask == u32::from(peer) & mask
            }
            (IpAddr::V6(network), IpAddr::V6(peer)) => {
                let mask = if self.prefix == 0 { 0 } else { u128::MAX << (128 - self.prefix) };
                u128::from(network) & mask == u128::from(peer) & mask
            }
            _ => false,
        }
    }
}

pub struct WebServer {
    addr: SocketAddr,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl WebServer {
    pub fn start(bind: &str, state: Arc<State>, started: Instant, allow_nets: &[String]) -> Result<Self> {
        let allowed_nets = allow_nets.iter().map(|net| AllowedNet::parse(net)).collect::<Result<Vec<_>>>()?;
        let listener = TcpListener::bind(bind).with_context(|| format!("web stats bind {bind}"))?;
        listener.set_nonblocking(true)?;
        let addr = listener.local_addr()?;
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = stop.clone();
        let thread = thread::Builder::new().name("pearl-web-stats".into()).spawn(move || {
            while !thread_stop.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, peer)) => {
                        if peer.ip().is_loopback() || allowed_nets.iter().any(|net| net.contains(peer.ip())) {
                            serve(stream, &state, started);
                        }
                    }
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
    fn cidr_acl_matches_only_configured_subnets() {
        let lan = AllowedNet::parse("192.168.50.0/24").unwrap();
        assert!(lan.contains("192.168.50.42".parse().unwrap()));
        assert!(!lan.contains("192.168.51.42".parse().unwrap()));
        assert!(!lan.contains("::1".parse().unwrap()));
        assert!(AllowedNet::parse("10.1.2.3").unwrap().contains("10.1.2.3".parse().unwrap()));
        assert!(AllowedNet::parse("2001:db8::/32").unwrap().contains("2001:db8::9".parse().unwrap()));
    }

    #[test]
    fn endpoint_is_read_only_and_reports_stats() {
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
    fn loopback_http_endpoint_serves_dashboard_and_json_stats() {
        let state = State::new();
        state.ready.store(true, Ordering::SeqCst);
        state.counters.lock().unwrap().accepted = 3;
        let server = WebServer::start("127.0.0.1:0", state.clone(), Instant::now(), &[]).unwrap();
        let mut stream = TcpStream::connect(server.local_addr()).unwrap();
        stream.write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(response.contains("axiom mining"));
        let mut stream = TcpStream::connect(server.local_addr()).unwrap();
        stream.write_all(b"GET /api/v1/stats HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        assert!(response.contains("\"accepted\":3"));
        drop(server);
    }
}
