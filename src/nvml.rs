//! Read-only NVML (libnvidia-ml.so.1, shipped with the driver): telemetry guard,
//! GPU process list and per-process SM utilisation.
use crate::state::State;
use anyhow::{bail, Result};
use std::collections::{HashMap, HashSet};
use std::ffi::{c_void, CString};
use std::os::raw::{c_int, c_uint};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

const NOT_FOUND: c_int = 6;
const INSUFFICIENT_SIZE: c_int = 7;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ProcessInfo {
    pid: c_uint,
    used_gpu_memory: u64,
    gpu_instance_id: c_uint,
    compute_instance_id: c_uint,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct UtilSample {
    pid: c_uint,
    time_stamp: u64,
    sm_util: c_uint,
    mem_util: c_uint,
    enc_util: c_uint,
    dec_util: c_uint,
}

type Dev = *mut c_void;

pub struct Nvml {
    lib: *mut c_void,
    dev: Dev,
    temperature: unsafe extern "C" fn(Dev, c_uint, *mut c_uint) -> c_int,
    power_limit: unsafe extern "C" fn(Dev, *mut c_uint) -> c_int,
    compute: unsafe extern "C" fn(Dev, *mut c_uint, *mut ProcessInfo) -> c_int,
    graphics: unsafe extern "C" fn(Dev, *mut c_uint, *mut ProcessInfo) -> c_int,
    utilization: unsafe extern "C" fn(Dev, *mut UtilSample, *mut c_uint, u64) -> c_int,
    shutdown: unsafe extern "C" fn() -> c_int,
    last_timestamp: u64,
}
unsafe impl Send for Nvml {}

fn check(rc: c_int, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("NVML {what} failed: status {rc}");
    }
    Ok(())
}

impl Nvml {
    /// Device by PCI bus id (as reported by CUDA), so CUDA and NVML numbering may differ.
    pub fn open(pci_bus_id: &str) -> Result<Self> {
        unsafe {
            let name = CString::new("libnvidia-ml.so.1")?;
            let lib = libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
            if lib.is_null() {
                bail!("libnvidia-ml.so.1 is unavailable");
            }
            let sym = |n: &str| -> Result<*mut c_void> {
                let c = CString::new(n)?;
                let p = libc::dlsym(lib, c.as_ptr());
                if p.is_null() {
                    bail!("missing NVML symbol {n}");
                }
                Ok(p)
            };
            let init: unsafe extern "C" fn() -> c_int = std::mem::transmute(sym("nvmlInit_v2")?);
            check(init(), "init")?;
            let handle: unsafe extern "C" fn(*const libc::c_char, *mut Dev) -> c_int = std::mem::transmute(sym("nvmlDeviceGetHandleByPciBusId_v2")?);
            let mut dev = std::ptr::null_mut();
            let bus = CString::new(pci_bus_id)?;
            check(handle(bus.as_ptr(), &mut dev), "device handle")?;
            let mut n = Nvml {
                lib,
                dev,
                temperature: std::mem::transmute(sym("nvmlDeviceGetTemperature")?),
                power_limit: std::mem::transmute(sym("nvmlDeviceGetPowerManagementLimit")?),
                compute: std::mem::transmute(sym("nvmlDeviceGetComputeRunningProcesses_v3")?),
                graphics: std::mem::transmute(sym("nvmlDeviceGetGraphicsRunningProcesses_v3")?),
                utilization: std::mem::transmute(sym("nvmlDeviceGetProcessUtilization")?),
                shutdown: std::mem::transmute(sym("nvmlShutdown")?),
                last_timestamp: 0,
            };
            n.active()?; // consume samples older than the start
            Ok(n)
        }
    }

    pub fn temperature_c(&self) -> Result<u32> {
        let mut v = 0;
        check(unsafe { (self.temperature)(self.dev, 0, &mut v) }, "temperature")?;
        Ok(v)
    }
    pub fn power_limit_w(&self) -> Result<f64> {
        let mut v = 0;
        check(unsafe { (self.power_limit)(self.dev, &mut v) }, "power limit")?;
        Ok(v as f64 / 1000.0)
    }
    /// PIDs with a context on the GPU (compute and graphics).
    pub fn pids(&self) -> Result<HashSet<u32>> {
        let mut found = HashSet::new();
        for f in [self.compute, self.graphics] {
            let mut size = 64usize;
            loop {
                let mut rows = vec![ProcessInfo::default(); size];
                let mut count = size as c_uint;
                let rc = unsafe { f(self.dev, &mut count, rows.as_mut_ptr()) };
                if rc == INSUFFICIENT_SIZE {
                    size = (size * 2).max(count as usize + 8);
                    continue;
                }
                check(rc, "process list")?;
                found.extend(rows[..count as usize].iter().map(|r| r.pid));
                break;
            }
        }
        Ok(found)
    }

    /// {pid: max SM utilisation %} over the samples since the previous call.
    pub fn active(&mut self) -> Result<HashMap<u32, u32>> {
        let mut size: c_uint = 0;
        let rc = unsafe { (self.utilization)(self.dev, std::ptr::null_mut(), &mut size, self.last_timestamp) };
        if rc == NOT_FOUND {
            return Ok(HashMap::new());
        }
        if rc != 0 && rc != INSUFFICIENT_SIZE {
            check(rc, "process utilization")?;
        }
        let mut rows = vec![UtilSample::default(); size as usize + 16];
        let mut n = rows.len() as c_uint;
        let rc = unsafe { (self.utilization)(self.dev, rows.as_mut_ptr(), &mut n, self.last_timestamp) };
        if rc == NOT_FOUND {
            return Ok(HashMap::new());
        }
        check(rc, "process utilization")?;
        let mut found = HashMap::new();
        for r in &rows[..n as usize] {
            let e = found.entry(r.pid).or_insert(0);
            *e = (*e).max(r.sm_util);
            self.last_timestamp = self.last_timestamp.max(r.time_stamp);
        }
        Ok(found)
    }
}

impl Drop for Nvml {
    fn drop(&mut self) {
        unsafe {
            (self.shutdown)();
            libc::dlclose(self.lib);
        }
    }
}

/// Telemetry guard: stops mining at `max_temp_c` or, if requested, at a power limit other than the required one.
pub struct Guard {
    stop: Arc<AtomicBool>,
    error: Arc<Mutex<Option<String>>>,
    handle: Option<JoinHandle<()>>,
}

impl Guard {
    pub fn start(pci_bus_id: &str, power_limit_w: Option<f64>, max_temp_c: u32, state: Arc<State>) -> Result<Self> {
        let nvml = Nvml::open(pci_bus_id)?;
        let sample = move |n: &Nvml, state: &State| -> Result<()> {
            let (t, p) = (n.temperature_c()?, n.power_limit_w()?);
            state.update_gpu_telemetry(t, p);
            if t >= max_temp_c {
                bail!("mining stopped: GPU temperature {t} °C (limit {max_temp_c} °C)");
            }
            if let Some(required) = power_limit_w {
                if (p - required).abs() > 0.5 {
                    bail!("mining stopped: power limit {p} W, required {required} W");
                }
            }
            Ok(())
        };
        sample(&nvml, &state)?;
        let stop = Arc::new(AtomicBool::new(false));
        let error = Arc::new(Mutex::new(None));
        let (s, e) = (stop.clone(), error.clone());
        let handle = std::thread::Builder::new().name("pearl-telemetry".into()).spawn(move || {
            while !s.load(Ordering::SeqCst) {
                for _ in 0..100 {
                    if s.load(Ordering::SeqCst) {
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
                if let Err(err) = sample(&nvml, &state) {
                    *e.lock().unwrap() = Some(format!("{err:#}"));
                    return;
                }
            }
        })?;
        Ok(Guard { stop, error, handle: Some(handle) })
    }
    pub fn check(&self) -> Result<()> {
        if let Some(e) = &*self.error.lock().unwrap() {
            bail!("{e}");
        }
        Ok(())
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
