//! Flux AI bridge CLAP plugin.
//! Real-time rule: process() must never block, allocate, or call into Python.
//! Audio is captured into a lock-free ring; a background task writes it to disk
//! and runs the `flux-ai --json` worker.

pub mod capture;
#[cfg(feature = "plugin")]
mod plugin;
pub mod worker;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn has_version() {
        assert!(!super::VERSION.is_empty());
    }
}
