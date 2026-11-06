//! /proc helpers: command lines and user names.

use std::collections::HashMap;
use std::fs;

#[derive(Clone, Debug, Default)]
pub struct ProcMeta {
    pub user: String,
    pub cmd: String,
}

#[derive(Default)]
pub struct ProcCache {
    users: HashMap<u32, String>,
    meta: HashMap<u32, ProcMeta>,
}

impl ProcCache {
    pub fn new() -> Self {
        let mut users = HashMap::new();
        if let Ok(s) = fs::read_to_string("/etc/passwd") {
            for line in s.lines() {
                let mut it = line.split(':');
                let (Some(name), _, Some(uid)) = (it.next(), it.next(), it.next()) else {
                    continue;
                };
                if let Ok(uid) = uid.parse() {
                    users.insert(uid, name.to_string());
                }
            }
        }
        Self {
            users,
            meta: HashMap::new(),
        }
    }

    pub fn get(&mut self, pid: u32, hint: Option<&str>) -> ProcMeta {
        if let Some(m) = self.meta.get(&pid) {
            return m.clone();
        }
        let cmd = fs::read(format!("/proc/{pid}/cmdline"))
            .ok()
            .map(|b| {
                String::from_utf8_lossy(&b)
                    .split('\0')
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .filter(|s| !s.is_empty())
            .or_else(|| hint.map(str::to_string))
            .unwrap_or_else(|| "?".into());
        let user = fs::read_to_string(format!("/proc/{pid}/status"))
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("Uid:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|u| u.parse::<u32>().ok())
            })
            .map(|uid| self.users.get(&uid).cloned().unwrap_or_else(|| uid.to_string()))
            .unwrap_or_else(|| "-".into());
        let m = ProcMeta { user, cmd };
        // Only cache real lookups; a hint-only answer may improve later.
        if std::path::Path::new(&format!("/proc/{pid}")).exists() {
            self.meta.insert(pid, m.clone());
        }
        m
    }

    /// Drop entries for pids that are no longer present.
    pub fn retain(&mut self, alive: &[u32]) {
        self.meta.retain(|pid, _| alive.contains(pid));
    }
}

pub fn hostname() -> String {
    fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "localhost".into())
}
