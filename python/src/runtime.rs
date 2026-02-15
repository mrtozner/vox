//! Shared tokio runtime for all Python wrapper methods.
//!
//! Every sync Python method that needs to call async Rust code
//! blocks on this runtime via `RUNTIME.block_on(...)`.

use once_cell::sync::Lazy;
use tokio::runtime::Runtime;

pub(crate) static RUNTIME: Lazy<Runtime> =
    Lazy::new(|| Runtime::new().expect("failed to create tokio runtime"));
