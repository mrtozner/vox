//! CLI command handlers for the `vox` binary.

pub mod benchmark;
pub mod chat;
pub mod config;
pub mod listen;
pub mod models;
pub mod speak;
pub mod test;

#[cfg(feature = "server")]
pub mod serve;
