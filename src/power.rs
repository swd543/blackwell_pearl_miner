//! Optional power-limit reset through a user-supplied, trusted policy helper.
//!
//! The miner never invokes sudo or changes GPU power policy itself. A caller may
//! supply an existing controlled helper (for example mining/plimit.sh); the
//! helper receives the reset limit as its only argument when this process exits.
use std::path::PathBuf;
use std::process::{Command, Stdio};

pub struct RestorePowerLimit(Option<PathBuf>);

impl RestorePowerLimit {
    pub fn new(helper: Option<PathBuf>) -> Self {
        Self(helper)
    }
}

impl Drop for RestorePowerLimit {
    fn drop(&mut self) {
        let Some(helper) = self.0.take() else { return };
        let result = Command::new(&helper)
            .arg("300")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        match result {
            Ok(status) if status.success() => {}
            Ok(status) => eprintln!("power-limit restore helper {} exited with {status}", helper.display()),
            Err(error) => eprintln!("could not run power-limit restore helper {}: {error}", helper.display()),
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
    fn does_nothing_when_no_helper_is_configured() {
        drop(RestorePowerLimit::new(None));
    }
}
