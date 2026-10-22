//! Flux AI bridge CLAP plugin (skeleton).
//! Real-time rule: the process() callback must never block or call into Python.
//! Jobs are submitted to a worker; results come back through a lock-free queue.

pub mod worker;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn has_version() {
        assert!(!super::VERSION.is_empty());
    }
}
