use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// sherpa-onnx release used for auto-download. Must have matching Linux
/// shared-library archives on GitHub releases.
const SHERPA_ONNX_VERSION: &str = "v1.12.20";
const SHERPA_ONNX_RELEASE_BASE: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download";

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_SHERPA");
    println!("cargo:rerun-if-env-changed=SHERPA_ONNX_DIR");
    println!("cargo:rerun-if-env-changed=VOX_SHERPA_NO_AUTODOWNLOAD");
    println!("cargo:rerun-if-env-changed=HOME");

    if env::var_os("CARGO_FEATURE_SHERPA").is_none()
        || !(cfg!(target_os = "macos") || cfg!(target_os = "linux"))
    {
        return;
    }

    let Some(runtime_dir) = app_runtime_dir() else {
        return;
    };

    // 1. Try already-staged libs, explicit SHERPA_ONNX_DIR, or the vendored
    //    directory first. This is how existing macOS / hand-built Linux
    //    installations work today.
    let mut source_dir = sherpa_source_dirs().into_iter().find(|dir| dir.is_dir());

    // 2. On Linux, if nothing usable was found OR if the vendor libs are the
    //    wrong architecture (e.g., the repo ships macOS dylibs), fall back to
    //    auto-download for the current target.
    if cfg!(target_os = "linux") {
        let needs_download = match source_dir.as_ref() {
            None => true,
            Some(dir) => !has_linux_libs(dir),
        };

        if needs_download && !download_disabled() {
            match autodownload_sherpa() {
                Ok(path) => {
                    println!(
                        "cargo:warning=vox: auto-downloaded sherpa-onnx {SHERPA_ONNX_VERSION} to {}",
                        path.display()
                    );
                    source_dir = Some(path);
                }
                Err(err) => {
                    println!(
                        "cargo:warning=vox: sherpa-onnx auto-download failed: {err}. See docs/raspberry_pi.md for manual instructions."
                    );
                }
            }
        }
    }

    if let Some(source_dir) = source_dir {
        let _ = stage_runtime_libs(&source_dir, &runtime_dir);
    }

    if runtime_dir.is_dir() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", runtime_dir.display());
        // Also expose the library search path so the linker can find
        // libsherpa-onnx-c-api / libonnxruntime at link time on Linux.
        println!("cargo:rustc-link-search=native={}", runtime_dir.display());
    }
}

fn sherpa_source_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    if let Some(configured) = configured_sherpa_dir() {
        dirs.push(configured);
    }

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is not set"));
    dirs.push(manifest_dir.join("vendor").join("sherpa-sys").join("lib"));

    if let Some(runtime_dir) = app_runtime_dir() {
        dirs.push(runtime_dir);
    }

    dirs
}

fn configured_sherpa_dir() -> Option<PathBuf> {
    let configured = env::var_os("SHERPA_ONNX_DIR")?;
    let path = PathBuf::from(configured);
    let lib_dir = path.join("lib");

    if lib_dir.is_dir() {
        Some(lib_dir)
    } else if path.is_dir() {
        Some(path)
    } else {
        None
    }
}

fn app_runtime_dir() -> Option<PathBuf> {
    let home = env::var_os("HOME")?;

    if cfg!(target_os = "macos") {
        Some(
            PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join("vox")
                .join("lib"),
        )
    } else {
        Some(
            PathBuf::from(home)
                .join(".local")
                .join("share")
                .join("vox")
                .join("lib"),
        )
    }
}

fn stage_runtime_libs(source_dir: &Path, runtime_dir: &Path) -> std::io::Result<()> {
    fs::create_dir_all(runtime_dir)?;

    for entry in fs::read_dir(source_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };

        if !should_stage(name) {
            continue;
        }

        let dest = runtime_dir.join(name);
        let should_copy = match (fs::metadata(&path), fs::metadata(&dest)) {
            (Ok(src), Ok(dst)) => src.len() != dst.len(),
            (Ok(_), Err(_)) => true,
            _ => false,
        };

        if should_copy {
            fs::copy(&path, dest)?;
        }
    }

    Ok(())
}

fn should_stage(name: &str) -> bool {
    if cfg!(target_os = "macos") || cfg!(target_os = "linux") {
        name.starts_with("libsherpa-onnx") || name.starts_with("libonnxruntime")
    } else {
        false
    }
}

/// Linux target detection + autodownload support.
///
/// Returns the name of the sherpa-onnx release asset for the current Linux
/// target, or `None` if the target is unsupported.
fn linux_archive_name() -> Option<String> {
    let arch = env::var("CARGO_CFG_TARGET_ARCH").ok()?;
    let env_abi = env::var("CARGO_CFG_TARGET_ENV").ok()?;

    let suffix = match (arch.as_str(), env_abi.as_str()) {
        // Raspberry Pi 4/5 64-bit, any modern aarch64 Linux (Debian, Ubuntu,
        // Raspberry Pi OS 64-bit, etc.). We use the -cpu variant since we
        // don't depend on CUDA at runtime.
        ("aarch64", "gnu") | ("aarch64", "musl") => "linux-aarch64-shared-cpu",
        // Raspberry Pi Zero 2 W / Pi 3 on 32-bit OS, other armv7 boards.
        ("arm", "gnueabihf") | ("armv7", "gnueabihf") => "linux-arm-gnueabihf-shared",
        // Linux x86_64 servers / desktops.
        ("x86_64", "gnu") | ("x86_64", "musl") => "linux-x64-shared",
        _ => return None,
    };

    Some(format!("sherpa-onnx-{SHERPA_ONNX_VERSION}-{suffix}"))
}

/// Returns true if the given directory contains Linux sherpa-onnx shared
/// libraries (so we don't re-download when the user has already staged them).
fn has_linux_libs(dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };

    let mut has_sherpa = false;
    let mut has_onnx = false;
    for entry in entries.flatten() {
        if let Some(name) = entry.file_name().to_str() {
            if name.starts_with("libsherpa-onnx-c-api.so") {
                has_sherpa = true;
            }
            if name.starts_with("libonnxruntime.so") {
                has_onnx = true;
            }
        }
    }
    has_sherpa && has_onnx
}

fn download_disabled() -> bool {
    // Explicit opt-out (CI, offline builds).
    if env::var("VOX_SHERPA_NO_AUTODOWNLOAD").is_ok() {
        return true;
    }
    // If CARGO_NET_OFFLINE=true, respect that too.
    if env::var("CARGO_NET_OFFLINE")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false)
    {
        return true;
    }
    false
}

/// Download + extract the appropriate sherpa-onnx Linux archive into
/// `~/.cache/vox/sherpa-onnx/<version>/<target>/lib` and return that lib dir.
fn autodownload_sherpa() -> Result<PathBuf, String> {
    let archive_name = linux_archive_name().ok_or_else(|| {
        format!(
            "unsupported Linux target ({}/{})",
            env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default(),
            env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default()
        )
    })?;

    let cache_root = cache_dir()?;
    let target_dir = cache_root
        .join(SHERPA_ONNX_VERSION.trim_start_matches('v'))
        .join(&archive_name);
    let lib_dir = target_dir.join("lib");

    // Already extracted from a previous build — reuse it.
    if has_linux_libs(&lib_dir) {
        return Ok(lib_dir);
    }

    fs::create_dir_all(&target_dir)
        .map_err(|e| format!("failed to create cache dir {}: {e}", target_dir.display()))?;

    let archive_file = format!("{archive_name}.tar.bz2");
    let url = format!("{SHERPA_ONNX_RELEASE_BASE}/{SHERPA_ONNX_VERSION}/{archive_file}");
    let archive_path = target_dir.join(&archive_file);

    println!("cargo:warning=vox: downloading {url}");
    let status = Command::new("curl")
        .arg("-sSL")
        .arg("--fail")
        .arg("-o")
        .arg(&archive_path)
        .arg(&url)
        .status()
        .map_err(|e| format!("failed to run curl (is it installed?): {e}"))?;
    if !status.success() {
        return Err(format!(
            "curl returned non-zero status ({status}) when fetching {url}"
        ));
    }

    println!("cargo:warning=vox: extracting {archive_file}");
    let status = Command::new("tar")
        .arg("xjf")
        .arg(&archive_path)
        .arg("-C")
        .arg(&target_dir)
        .status()
        .map_err(|e| format!("failed to run tar: {e}"))?;
    if !status.success() {
        return Err(format!("tar returned non-zero status ({status})"));
    }

    // Archives extract into a single subdirectory named <archive_name>/. Move
    // its `lib` child into our canonical target_dir/lib so that the caller can
    // just point at target_dir/lib.
    let nested_lib = target_dir.join(&archive_name).join("lib");
    if nested_lib.is_dir() && !lib_dir.is_dir() {
        // Copy files out (rather than rename) so we don't disturb the rest of
        // the extracted tree — users may want the headers for manual rebuilds.
        fs::create_dir_all(&lib_dir).map_err(|e| {
            format!("failed to create lib cache dir {}: {e}", lib_dir.display())
        })?;
        for entry in fs::read_dir(&nested_lib)
            .map_err(|e| format!("failed to read {}: {e}", nested_lib.display()))?
        {
            let entry = entry.map_err(|e| format!("failed to read entry: {e}"))?;
            let src = entry.path();
            if !src.is_file() {
                continue;
            }
            let Some(name) = src.file_name() else {
                continue;
            };
            let dst = lib_dir.join(name);
            fs::copy(&src, &dst)
                .map_err(|e| format!("failed to copy {} -> {}: {e}", src.display(), dst.display()))?;
        }
    }

    // Clean up the archive to save disk space; extracted tree stays cached.
    let _ = fs::remove_file(&archive_path);

    if has_linux_libs(&lib_dir) {
        Ok(lib_dir)
    } else {
        Err(format!(
            "sherpa-onnx libs not found in extracted archive at {}",
            lib_dir.display()
        ))
    }
}

fn cache_dir() -> Result<PathBuf, String> {
    // Prefer XDG_CACHE_HOME, fall back to $HOME/.cache, fall back to OUT_DIR.
    if let Some(xdg) = env::var_os("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(xdg).join("vox").join("sherpa-onnx"));
    }
    if let Some(home) = env::var_os("HOME") {
        return Ok(PathBuf::from(home)
            .join(".cache")
            .join("vox")
            .join("sherpa-onnx"));
    }
    // Last resort: use cargo's OUT_DIR so at least the build succeeds, at the
    // cost of re-downloading on every `cargo clean`.
    let out_dir = env::var("OUT_DIR")
        .map_err(|_| "no HOME, XDG_CACHE_HOME, or OUT_DIR set".to_string())?;
    Ok(PathBuf::from(out_dir).join("sherpa-onnx-cache"))
}
