//! The CUDA kernel library (built by build.rs), embedded in the binary and loaded from
//! memory with memfd_create + dlopen.
use anyhow::{bail, Result};
use std::ffi::{c_void, CString};
use std::os::raw::c_int;

type P = *mut c_void;
type C = *const c_void;

static LIBRARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/libpearlkernels.so"));

pub struct Kernels {
    _handle: *mut c_void,
    pub k780: unsafe extern "C" fn(C, C, C, C, c_int, c_int, c_int, C, C, P, P, c_int, P) -> c_int,
    pub k448: unsafe extern "C" fn(C, C, C, C, P, P, c_int, P) -> c_int,
    pub merkle_init: unsafe extern "C" fn(C, P, c_int, P) -> c_int,
    pub merkle_root: unsafe extern "C" fn(C, C, C, P, c_int, P) -> c_int,
    pub advance_nonce: unsafe extern "C" fn(P, P, c_int, P) -> c_int,
    pub build_indices: unsafe extern "C" fn(C, P, c_int, P) -> c_int,
    pub pack_basis: unsafe extern "C" fn(C, P, P) -> c_int,
    pub prepare_packed: unsafe extern "C" fn(C, C, C, P, P, c_int, c_int, P) -> c_int,
    pub commit: unsafe extern "C" fn(C, C, C, P, P, c_int, c_int, c_int, P) -> c_int,
    pub noise: unsafe extern "C" fn(P, P, P, P, C, C, c_int, c_int, c_int, c_int, P) -> c_int,
}
unsafe impl Send for Kernels {}
unsafe impl Sync for Kernels {}

fn open() -> Result<*mut c_void> {
    unsafe {
        let name = CString::new("pearl-kernels")?;
        let fd = libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC);
        if fd < 0 {
            bail!("memfd_create failed");
        }
        let mut written = 0;
        while written < LIBRARY.len() {
            let n = libc::write(fd, LIBRARY[written..].as_ptr().cast(), LIBRARY.len() - written);
            if n <= 0 {
                libc::close(fd);
                bail!("cannot write the embedded kernel library");
            }
            written += n as usize;
        }
        // The memfd stays open: the loader identifies libraries by path.
        let path = CString::new(format!("/proc/self/fd/{fd}"))?;
        let handle = libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        if handle.is_null() {
            libc::close(fd);
            let err = std::ffi::CStr::from_ptr(libc::dlerror()).to_string_lossy().into_owned();
            bail!("cannot load the kernel library: {err}");
        }
        Ok(handle)
    }
}

fn sym<T: Copy>(handle: *mut c_void, name: &str) -> Result<T> {
    let c = CString::new(name)?;
    let p = unsafe { libc::dlsym(handle, c.as_ptr()) };
    if p.is_null() {
        bail!("missing kernel symbol {name}");
    }
    Ok(unsafe { std::mem::transmute_copy(&p) })
}

impl Kernels {
    pub fn load() -> Result<Self> {
        let h = open()?;
        Ok(Kernels {
            _handle: h,
            k780: sym(h, "pm_k780_launch")?,
            k448: sym(h, "pm_k448_launch")?,
            merkle_init: sym(h, "pm_merkle_init")?,
            merkle_root: sym(h, "pm_merkle_root")?,
            advance_nonce: sym(h, "pm_advance_nonce")?,
            build_indices: sym(h, "pm_build_indices")?,
            pack_basis: sym(h, "pm_pack_basis")?,
            prepare_packed: sym(h, "pm_prepare_packed")?,
            commit: sym(h, "pm_commit")?,
            noise: sym(h, "pm_noise")?,
        })
    }
}

static CELL: std::sync::OnceLock<std::result::Result<Kernels, String>> = std::sync::OnceLock::new();

/// The process-wide kernel library (loaded once).
pub fn get() -> Result<&'static Kernels> {
    match CELL.get_or_init(|| Kernels::load().map_err(|e| format!("{e:#}"))) {
        Ok(k) => Ok(k),
        Err(e) => bail!("{e}"),
    }
}
