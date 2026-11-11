//! Optional GPU power policy through a user-supplied, trusted helper.
//!
//! The miner never invokes sudo or changes GPU power policy itself. A caller may
//! supply a tightly controlled helper; it receives only the approved 250 W / 300 W
//! values. The 300 W limit is restored on yield and exit; 250 W is reapplied before
//! mining resumes when the configured mining limit is 250 W.
use anyhow::{ensure, Context, Result};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub const MINING_LIMIT_W: u32 = 250;
pub const YIELDED_LIMIT_W: u32 = 300;

pub struct RestorePowerLimit(Option<PathBuf>);

impl RestorePowerLimit {
    pub fn new(helper: Option<PathBuf>) -> Self {
        Self(helper)
    }

    pub fn has_helper(&self) -> bool {
        self.0.is_some()
    }

    pub fn set_limit(&self, watts: u32) -> Result<()> {
        ensure!(watts == MINING_LIMIT_W || watts == YIELDED_LIMIT_W, "refusing unapproved power limit: {watts} W");
        let Some(helper) = self.0.as_deref() else { return Ok(()) };
        run_helper(helper, watts)
    }

    pub fn manages_yield(&self, mining_limit_w: Option<f64>) -> bool {
        self.has_helper() && mining_limit_w.is_some_and(|limit| (limit - f64::from(MINING_LIMIT_W)).abs() <= 0.5)
    }
}

fn run_helper(helper: &Path, watts: u32) -> Result<()> {
    let status = Command::new(helper)
        .arg(watts.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .with_context(|| format!("could not run power-limit helper {}", helper.display()))?;
    ensure!(status.success(), "power-limit helper {} exited with {status}", helper.display());
    Ok(())
}

impl Drop for RestorePowerLimit {
    fn drop(&mut self) {
        let Some(helper) = self.0.take() else { return };
        if let Err(error) = run_helper(&helper, YIELDED_LIMIT_W) {
            eprintln!("could not restore default GPU power limit: {error:#}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn calls_trusted_helper_with_default_limit_on_drop() {
        let dir = std::env::temp_dir().join(format!(
            "bpm-power-restore-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let marker = dir.join("limit.txt");
        let helper = dir.join("helper.sh");
        std::fs::write(&helper, format!("#!/bin/sh\nprintf '%s' \"$1\" > '{}'\n", marker.display())).unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o700)).unwrap();
        drop(RestorePowerLimit::new(Some(helper)));
        assert_eq!(std::fs::read_to_string(marker).unwrap(), "300");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn changes_limit_on_yield_and_resume_then_restores_on_drop() {
        let dir = std::env::temp_dir().join(format!(
            "bpm-power-transition-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let marker = dir.join("limits.txt");
        let helper = dir.join("helper.sh");
        std::fs::write(&helper, format!("#!/bin/sh\nprintf '%s\\n' \"$1\" >> '{}'\n", marker.display())).unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o700)).unwrap();
        let controller = RestorePowerLimit::new(Some(helper));
        controller.set_limit(YIELDED_LIMIT_W).unwrap();
        controller.set_limit(MINING_LIMIT_W).unwrap();
        drop(controller);
        assert_eq!(std::fs::read_to_string(marker).unwrap(), "300\n250\n300\n");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn manages_yield_only_for_the_supported_mining_policy() {
        let no_helper = RestorePowerLimit::new(None);
        let helper = RestorePowerLimit::new(Some(PathBuf::from("helper")));
        assert!(!no_helper.manages_yield(Some(250.0)));
        assert!(!helper.manages_yield(None));
        assert!(!helper.manages_yield(Some(200.0)));
        assert!(helper.manages_yield(Some(250.0)));
    }

    #[test]
    fn does_nothing_when_no_helper_is_configured() {
        drop(RestorePowerLimit::new(None));
        assert!(RestorePowerLimit::new(None).set_limit(275).is_err());
    }
}
