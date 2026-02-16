use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let lib_dir = find_sherpa_lib_dir().unwrap_or_else(|| {
        eprintln!(
            "\n\
            ============================================================\n\
            sherpa-onnx shared libraries not found.\n\
            \n\
            Download prebuilt libraries from:\n\
              https://github.com/k2-fsa/sherpa-onnx/releases\n\
            \n\
            Then either:\n\
              1. Set SHERPA_ONNX_DIR=/path/to/sherpa-onnx-vX.Y.Z\n\
              2. Copy libs to vendor/sherpa-sys/lib/\n\
              3. Run: vox models download sherpa-libs\n\
            ============================================================\n"
        );
        std::process::exit(1);
    });

    let lib_dir_str = lib_dir.display().to_string();

    println!("cargo:rustc-link-search=native={lib_dir_str}");
    println!("cargo:rustc-link-lib=dylib=sherpa-onnx-c-api");
    println!("cargo:rustc-link-lib=dylib=onnxruntime");

    // Platform-specific linker flags.
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir_str}");
        println!("cargo:rustc-link-lib=dylib=c++");
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Foundation");
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    // Re-run if the env var changes or the lib directory contents change.
    println!("cargo:rerun-if-env-changed=SHERPA_ONNX_DIR");
    println!("cargo:rerun-if-changed={lib_dir_str}");
}

/// Search for the sherpa-onnx shared library directory in priority order:
///
/// 1. `SHERPA_ONNX_DIR` environment variable (user-provided path)
/// 2. `vendor/sherpa-sys/lib/` (local copy next to this crate)
/// 3. Platform-specific application data directory:
///    - macOS: `~/Library/Application Support/vox/lib/`
///    - Linux: `~/.local/share/vox/lib/`
fn find_sherpa_lib_dir() -> Option<PathBuf> {
    // 1. SHERPA_ONNX_DIR env var
    if let Ok(dir) = env::var("SHERPA_ONNX_DIR") {
        let lib_path = Path::new(&dir).join("lib");
        if has_sherpa_lib(&lib_path) {
            return Some(lib_path);
        }
        // Also check the directory itself (user might point directly at the lib dir)
        let direct = PathBuf::from(&dir);
        if has_sherpa_lib(&direct) {
            return Some(direct);
        }
    }

    // 2. vendor/sherpa-sys/lib/ (relative to this crate's manifest dir)
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let local_lib = Path::new(&manifest_dir).join("lib");
    if has_sherpa_lib(&local_lib) {
        return Some(local_lib);
    }

    // 3. Platform-specific application data directory
    if let Some(data_dir) = app_data_lib_dir() {
        if has_sherpa_lib(&data_dir) {
            return Some(data_dir);
        }
    }

    None
}

/// Check whether the given directory contains a sherpa-onnx-c-api shared library.
fn has_sherpa_lib(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }

    let lib_name = if cfg!(target_os = "macos") {
        "libsherpa-onnx-c-api.dylib"
    } else if cfg!(target_os = "windows") {
        "sherpa-onnx-c-api.dll"
    } else {
        "libsherpa-onnx-c-api.so"
    };

    dir.join(lib_name).exists()
}

/// Return the platform-specific vox lib directory.
fn app_data_lib_dir() -> Option<PathBuf> {
    let home = env::var("HOME").ok()?;

    if cfg!(target_os = "macos") {
        Some(
            Path::new(&home)
                .join("Library/Application Support/vox/lib"),
        )
    } else {
        // Linux and other Unix-like systems
        Some(Path::new(&home).join(".local/share/vox/lib"))
    }
}
