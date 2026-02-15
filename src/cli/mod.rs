//! CLI command handlers for the `vox` binary.

pub mod listen;
pub mod models;
pub mod speak;

#[cfg(feature = "server")]
pub mod serve;
