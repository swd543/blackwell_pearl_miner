//! Optional, dependency-free, read-only HTTP stats endpoint.
//!
//! Disabled by default. Network clients must belong to an explicitly configured
//! LAN CIDR; loopback is always allowed. The endpoint has no authentication.
use crate::state::State;
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
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
.tile .v small{font-size:.68rem;color:#8a93a5;font-weight:400}.tile.hot .v{color:#ffb36b}.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 12px;font-size:.76rem;color:#8a93a5;margin-top:12px;border-top:1px solid #262b36;padding-top:10px}
.kv b{color:#c6cbd4;font-weight:500;overflow-wrap:anywhere}pre{color:#9aa2ad;font-size:.72rem;line-height:1.6;white-space:pre-wrap;font-family:ui-monospace,monospace;margin:12px 0 0;flex:1;min-height:90px;overflow-y:auto;padding-right:4px}
.hint{color:#6f7683;font-size:.7rem;margin:10px 0 0}
.chart{margin-top:14px;background:#14171d;border:1px solid #262b36;border-radius:12px;padding:10px}.chart-head{display:flex;justify-content:space-between;color:#a7aebb;font-size:.78rem}.chart canvas{display:block;width:100%;height:230px}.legend{display:flex;flex-wrap:wrap;gap:12px;color:#8a93a5;font-size:.7rem}.legend span:before{content:"";display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;background:var(--c)}
</style></head><body><div class="card">
<div class="top"><h1>axiom <span class="sub">Pearl / Kryptex</span></h1><span id="badge" class="badge">CONNECTING</span></div>
<p id="reason" class="reason">Waiting for miner stats…</p>
<div class="grid">
<div class="tile wide"><div class="k">Submitted rate · 30m window</div><div class="v"><span id="rate">—</span> <small>TH/s</small></div></div>
<div class="tile"><div class="k">Accepted</div><div class="v" id="accepted">—</div></div>
<div class="tile"><div class="k">Rejected</div><div class="v" id="rejected">—</div></div>
<div class="tile"><div class="k">Stale</div><div class="v" id="stale">—</div></div>
<div class="tile"><div class="k">Uptime</div><div class="v" id="uptime">—</div></div>
<div class="tile"><div class="k">GPU temp</div><div class="v" id="temperature">—</div></div>
<div class="tile"><div class="k">Power limit</div><div class="v" id="power-limit">—</div></div>
</div>
<div class="kv"><span>Job</span><b id="job">—</b><span>Updated</span><b id="updated">—</b></div>
<div class="chart"><div class="chart-head"><span>GPU temperature &amp; hashrate</span><span id="trend-window">36h</span></div><canvas id="trend" aria-label="GPU temperature and hashrate trend"></canvas><div class="legend"><span style="--c:#ffb36b">GPU °C</span><span style="--c:#6aa9ff">Compute TMAC/s</span><span style="--c:#57d98a">Submitted TH/s</span></div></div>
<div class="sub" style="margin-top:12px">Recent miner events</div>
<pre id="detail">Loading recent events…</pre>
<p class="hint">Read-only · Pearl / Kryptex · no credentials are displayed</p>
</div><script>
const el=id=>document.getElementById(id), fmt=n=>Number.isFinite(Number(n))?Number(n).toLocaleString():"—";
function duration(s){s=Math.max(0,Math.floor(Number(s)||0));let h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h?`${h}h ${m}m`:`${m}m ${s%60}s`}
function shortSpan(s){return s<60?`${Math.max(1,Math.round(s))}s`:s<3600?`${Math.round(s/60)}m`:`${(s/3600).toFixed(s%3600?1:0)}h`}
const trend={cursor:null,temperature:[],hashrate:[],windowSeconds:129600,lastTrim:0};
function drawTrend(){const canvas=el('trend'),rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=Math.max(320,rect.width),h=230;if(!rect.width)return;canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,w,h);const m={l:42,r:46,t:12,b:25},pw=w-m.l-m.r,ph=h-m.t-m.b,now=Date.now(),firstTemp=trend.temperature[0]?.[0],firstRate=trend.hashrate[0]?.[0],first=firstTemp==null?firstRate:firstRate==null?firstTemp:Math.min(firstTemp,firstRate),available=first==null?trend.windowSeconds:Math.max(60,(now-first)/1000),visibleSeconds=Math.min(trend.windowSeconds,available),start=now-visibleSeconds*1000,tmin=40,tmax=100;el('trend-window').textContent=first==null?`No data · ${shortSpan(trend.windowSeconds)}`:visibleSeconds<trend.windowSeconds?`${shortSpan(visibleSeconds)} of ${shortSpan(trend.windowSeconds)}`:`Last ${shortSpan(trend.windowSeconds)}`;let peakRate=0;for(const p of trend.hashrate)peakRate=Math.max(peakRate,p[1],p[2]);const rmax=Math.max(50,Math.ceil(peakRate*1.1/50)*50);c.font='10px system-ui';c.lineWidth=1;c.strokeStyle='#303743';c.fillStyle='#858e9d';c.textBaseline='middle';for(let i=0;i<=4;i++){let y=m.t+ph*i/4,temp=Math.round(tmax-(tmax-tmin)*i/4),rate=Math.round(rmax*(1-i/4));c.beginPath();c.moveTo(m.l,y);c.lineTo(m.l+pw,y);c.stroke();c.textAlign='right';c.fillText(`${temp}°`,m.l-6,y);c.textAlign='left';c.fillText(`${rate}`,m.l+pw+6,y)}c.textBaseline='alphabetic';c.textAlign='center';for(let i=0;i<=2;i++){let x=m.l+pw*i/2;c.fillText(new Date(start+visibleSeconds*1000*i/2).toLocaleString([],{month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit',...(visibleSeconds<3600?{second:'2-digit'}:{})}),x,h-5)}
const line=(points,value,lo,hi,color)=>{c.save();c.beginPath();c.rect(m.l,m.t,pw,ph);c.clip();c.beginPath();let began=false;for(const p of points){if(p[0]<start||p[0]>now)continue;let x=m.l+(p[0]-start)/(visibleSeconds*1000)*pw,y=m.t+(hi-p[value])/(hi-lo)*ph;if(!began){c.moveTo(x,y);began=true}else c.lineTo(x,y)}c.strokeStyle=color;c.lineWidth=1.6;c.stroke();c.restore()};line(trend.temperature,1,tmin,tmax,'#ffb36b');line(trend.hashrate,1,0,rmax,'#6aa9ff');line(trend.hashrate,2,0,rmax,'#57d98a');if(!trend.temperature.length&&!trend.hashrate.length){c.fillStyle='#858e9d';c.textAlign='center';c.fillText('Collecting samples…',m.l+pw/2,m.t+ph/2)}}
async function updateTrend(){try{const url='/api/v1/history'+(trend.cursor===null?'':`?after=${trend.cursor}`),r=await fetch(url,{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);const d=await r.json();if(d.reset){trend.temperature=[];trend.hashrate=[]}trend.temperature.push(...d.temperature);trend.hashrate.push(...d.hashrate);trend.cursor=d.cursor;trend.windowSeconds=d.window_seconds;const now=Date.now();if(now-trend.lastTrim>30000){const cutoff=now-trend.windowSeconds*1000;let ti=0,hi=0;while(ti<trend.temperature.length&&trend.temperature[ti][0]<cutoff)ti++;while(hi<trend.hashrate.length&&trend.hashrate[hi][0]<cutoff)hi++;if(ti)trend.temperature.splice(0,ti);if(hi)trend.hashrate.splice(0,hi);trend.lastTrim=now}drawTrend()}catch(e){}}
async function update(){try{const r=await fetch('/api/v1/stats',{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);const s=await r.json();const b=el('badge');
const mode={mining:['MINING','up','Mining'],yielding:['YIELDING','','Yielding GPU to other work'],connecting:['WAITING','','Connecting to pool'],stopping:['STOPPING','bad','Miner is stopping']};const v=mode[s.status]||['UNKNOWN','bad','Miner status unavailable'];b.textContent=v[0];b.className='badge '+v[1];el('reason').textContent=v[2];
el('rate').textContent=fmt(s.rate?.submitted_TH_s_30m);el('accepted').textContent=fmt(s.accepted);el('rejected').textContent=fmt(s.rejected);el('stale').textContent=fmt(s.stale);el('uptime').textContent=duration(s.uptime_seconds);el('job').textContent=s.job_id||'—';el('updated').textContent=new Date().toLocaleTimeString();
const temp=s.gpu?.temperature_c;el('temperature').textContent=temp==null?'—':`${fmt(temp)} °C`;el('temperature').closest('.tile').classList.toggle('hot',Number(temp)>=85);const limit=s.gpu?.power_limit_w;el('power-limit').textContent=limit==null?'—':`${fmt(limit)} W`;
el('detail').textContent=(s.events||[]).join('\n')||'No recent miner events.';
}catch(e){el('badge').textContent='OFFLINE';el('badge').className='badge bad';el('reason').textContent='Stats unavailable: '+e;}}
async function refresh(){await update();await updateTrend()}refresh();setInterval(refresh,5000);window.addEventListener('resize',drawTrend);
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
    pub fn start(bind: &str, state: Arc<State>, started: Instant, allow_nets: &[String], log_path: PathBuf) -> Result<Self> {
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
                            serve(stream, &state, started, &log_path);
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

fn stats(state: &State, started: Instant, log_path: &Path) -> Value {
    let counters = state.counters();
    let rate = state.share_rate.lock().unwrap().snapshot();
    let telemetry = state.gpu_telemetry();
    let job_id = state.job().map(|g| g.job.job_id.clone());
    json!({
        "status": if state.abort.load(Ordering::SeqCst) { "stopping" } else if !state.ready.load(Ordering::SeqCst) { "connecting" } else if state.gpu_paused.load(Ordering::SeqCst) { "yielding" } else { "mining" },
        "uptime_seconds": started.elapsed().as_secs(),
        "job_id": job_id,
        "accepted": counters.accepted,
        "rejected": counters.rejected,
        "stale": counters.stale,
        "rate": rate,
        "gpu": {"temperature_c":telemetry.temperature_c,"power_limit_w":telemetry.power_limit_w},
        "events": recent_events(log_path)
    })
}

const EVENT_TAIL_BYTES: u64 = 131_072;
const EVENT_TAIL_COUNT: usize = 20;
const SAFE_EVENTS: &[&str] = &["authorized", "job", "mining", "share", "stats", "complete", "share_discarded", "round_discarded", "priority_pause", "priority_resume", "disconnected", "profile_fallback", "error"];

/// Format a bounded tail of existing JSONL events; never includes the startup row (which has the wallet).
fn recent_events(path: &Path) -> Vec<String> {
    let Ok(mut file) = File::open(path) else { return Vec::new() };
    let Ok(len) = file.metadata().map(|m| m.len()) else { return Vec::new() };
    let start = len.saturating_sub(EVENT_TAIL_BYTES);
    if file.seek(SeekFrom::Start(start)).is_err() { return Vec::new(); }
    let mut reader = BufReader::new(file);
    if start > 0 {
        let mut partial = String::new();
        if reader.read_line(&mut partial).is_err() { return Vec::new(); }
    }
    let mut tail = VecDeque::with_capacity(EVENT_TAIL_COUNT);
    for line in reader.lines().flatten() {
        let Ok(row) = serde_json::from_str::<Value>(&line) else { continue };
        let event = row["event"].as_str().unwrap_or_default();
        if !SAFE_EVENTS.contains(&event) { continue; }
        let formatted = crate::console::format_event(&row);
        if !formatted.is_empty() {
            tail.push_back(formatted);
            if tail.len() > EVENT_TAIL_COUNT { tail.pop_front(); }
        }
    }
    tail.into_iter().collect()
}

fn trend_payload(state: &State, after: Option<u64>) -> Value {
    let snapshot = state.trend_snapshot(after);
    let temperature: Vec<(u64, u32)> = snapshot.temperature.iter().map(|p| (p.timestamp_ms, p.temperature_c)).collect();
    let hashrate: Vec<(u64, f32, f32)> = snapshot.hashrate.iter().map(|p| (p.timestamp_ms, p.compute_tmac_s, p.submitted_th_s)).collect();
    json!({"reset":snapshot.reset,"cursor":snapshot.cursor,"window_seconds":snapshot.window_seconds,"temperature":temperature,"hashrate":hashrate})
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

fn serve(mut stream: TcpStream, state: &State, started: Instant, log_path: &Path) {
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
    let target = parts.next().unwrap_or_default();
    let mut target_parts = target.splitn(2, '?');
    let path = target_parts.next().unwrap_or_default();
    let query = target_parts.next();
    if method != "GET" {
        respond(&mut stream, "405 Method Not Allowed", "text/plain", "GET only\n");
        return;
    }
    match path {
        "/" => respond(&mut stream, "200 OK", "text/html", DASHBOARD),
        "/api/v1/stats" | "/health" => {
            let body = stats(state, started, log_path).to_string();
            respond(&mut stream, "200 OK", "application/json", &body);
        }
        "/api/v1/history" => {
            let after = query.and_then(|q| q.split('&').find_map(|part| part.strip_prefix("after=")));
            let after = match after.map(str::parse::<u64>).transpose() {
                Ok(value) => value,
                Err(_) => { respond(&mut stream, "400 Bad Request", "text/plain", "invalid after cursor\n"); return; }
            };
            let body = trend_payload(state, after).to_string();
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
        let state = State::with_history_window_hours(crate::state::DEFAULT_HISTORY_WINDOW_HOURS);
        let start = Instant::now();
        state.counters.lock().unwrap().accepted = 7;
        state.ready.store(true, Ordering::SeqCst);
        state.gpu_paused.store(true, Ordering::SeqCst);
        state.update_gpu_telemetry(74, 250.0);
        let row = stats(&state, start, Path::new("/dev/null"));
        assert_eq!(row["status"], "yielding");
        assert_eq!(row["accepted"], 7);
        assert_eq!(row["gpu"]["temperature_c"], 74);
        assert_eq!(row["gpu"]["power_limit_w"], 250.0);
        assert!(row.get("wallet").is_none());
    }

    #[test]
    fn log_tail_is_bounded_to_safe_formatted_events() {
        let path = std::env::temp_dir().join(format!("bpm-web-log-{}.jsonl", std::process::id()));
        let startup = json!({"time":1.0,"event":"startup","wallet":"do-not-display"});
        let share = json!({"time":2.0,"event":"share","accepted":true,"accepted_total":1,"rejected_total":0,"roundtrip_ms":42.0});
        let unknown = json!({"time":3.0,"event":"private_event","secret":"do-not-display"});
        std::fs::write(&path, format!("{startup}\n{share}\n{unknown}\n")).unwrap();
        let events = recent_events(&path);
        assert_eq!(events.len(), 1);
        assert!(events[0].contains("SHARE ACCEPTED"));
        assert!(!events.join("\n").contains("do-not-display"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn history_endpoint_returns_initial_series_then_only_new_samples() {
        let state = State::with_history_window_hours(1);
        state.update_gpu_telemetry(80, 250.0);
        state.record_hashrate(300.0, 275.0);
        let initial = trend_payload(&state, None);
        assert_eq!(initial["reset"], true);
        assert_eq!(initial["window_seconds"], 3600);
        assert_eq!(initial["temperature"].as_array().unwrap().len(), 1);
        assert_eq!(initial["hashrate"].as_array().unwrap().len(), 1);
        let cursor = initial["cursor"].as_u64().unwrap();
        assert_eq!(trend_payload(&state, Some(cursor))["temperature"].as_array().unwrap().len(), 0);
        state.update_gpu_telemetry(81, 250.0);
        let delta = trend_payload(&state, Some(cursor));
        assert_eq!(delta["reset"], false);
        assert_eq!(delta["temperature"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn loopback_http_endpoint_serves_dashboard_and_json_stats() {
        let state = State::with_history_window_hours(crate::state::DEFAULT_HISTORY_WINDOW_HOURS);
        state.ready.store(true, Ordering::SeqCst);
        state.counters.lock().unwrap().accepted = 3;
        let server = WebServer::start("127.0.0.1:0", state.clone(), Instant::now(), &[], PathBuf::from("/dev/null")).unwrap();
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
