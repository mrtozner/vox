//! Egress control: where vox is allowed to connect, and a record of where it did.
//!
//! vox makes two kinds of outbound connection, and nothing else:
//!
//! 1. requests to the configured LLM runtime, and
//! 2. model downloads (`vox models download`, `vox serve --download`, the
//!    first-run prompts of the other commands).
//!
//! This module is the single place both go through:
//!
//! - [`resolve_ollama_host`] decides which Ollama address is used
//!   (flag, then `VOX_OLLAMA_HOST`, then the config file, then the default).
//! - The shared request builder prepares each LLM request. Offline requests
//!   connect directly to loopback addresses and do not follow redirects.
//! - [`download_request`] gates every model download. In offline mode it
//!   refuses all of them.
//! - Both return a [`Connection`] that records destination, purpose and
//!   payload bytes in the process-wide [`PrivacyLedger`], which the server
//!   exposes at `GET /v1/privacy/ledger`.
//!
//! Offline mode is switched on with `--offline` or `VOX_OFFLINE=1`.
//!
//! ## What the byte counts mean
//!
//! Counts are HTTP body bytes as seen by vox (request JSON, response body).
//! Headers, TLS and TCP overhead are not included.
//!
//! ## What is not covered
//!
//! Backends that fetch their own weights through the Hugging Face hub client
//! (`pocket`, `chatterbox`, `qwen3` features) do not go through this module.
//! They are not part of the default release feature set.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::net::IpAddr;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use serde::Serialize;

/// Ollama address used when nothing else is configured.
pub const DEFAULT_OLLAMA_HOST: &str = "localhost:11434";
/// Environment variable naming the Ollama address.
pub const OLLAMA_HOST_ENV: &str = "VOX_OLLAMA_HOST";
/// Environment variable that switches offline mode on (`1`, `true`, `yes`, `on`).
pub const OFFLINE_ENV: &str = "VOX_OFFLINE";

/// How many individual connections the ledger keeps before dropping the
/// oldest. Per-destination totals are kept for the life of the process.
pub const LEDGER_CAPACITY: usize = 500;

// ---------------------------------------------------------------------------
// Host resolution
// ---------------------------------------------------------------------------

/// Where a resolved Ollama address came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HostSource {
    /// `--ollama-host`.
    Flag,
    /// `VOX_OLLAMA_HOST`.
    Env,
    /// `ollama_host` in the config file.
    Config,
    /// [`DEFAULT_OLLAMA_HOST`].
    Default,
}

impl fmt::Display for HostSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            HostSource::Flag => "--ollama-host",
            HostSource::Env => OLLAMA_HOST_ENV,
            HostSource::Config => "config file",
            HostSource::Default => "default",
        })
    }
}

/// An Ollama address as bare `host:port`, and where it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedHost {
    pub host: String,
    pub source: HostSource,
}

/// Reduce an Ollama address to `host:port`: callers build
/// `http://{host}/api/...` URLs, so a scheme, path or trailing slash typed by
/// the operator would otherwise produce a broken URL. Empty input yields
/// [`DEFAULT_OLLAMA_HOST`].
pub fn normalize_ollama_host(raw: &str) -> String {
    let s = raw.trim();
    let s = s
        .strip_prefix("http://")
        .or_else(|| s.strip_prefix("https://"))
        .unwrap_or(s);
    let s = s.split('/').next().unwrap_or(s).trim();
    if s.is_empty() {
        DEFAULT_OLLAMA_HOST.to_string()
    } else {
        s.to_string()
    }
}

/// Pick the Ollama address: `flag`, then `env`, then `config`, then the
/// default. Blank values count as absent. The result is normalized.
pub fn resolve_ollama_host_from(
    flag: Option<&str>,
    env: Option<&str>,
    config: Option<&str>,
) -> ResolvedHost {
    fn present(v: Option<&str>) -> Option<&str> {
        v.map(str::trim).filter(|s| !s.is_empty())
    }
    let (raw, source) = if let Some(v) = present(flag) {
        (v, HostSource::Flag)
    } else if let Some(v) = present(env) {
        (v, HostSource::Env)
    } else if let Some(v) = present(config) {
        (v, HostSource::Config)
    } else {
        (DEFAULT_OLLAMA_HOST, HostSource::Default)
    };
    ResolvedHost {
        host: normalize_ollama_host(raw),
        source,
    }
}

/// [`resolve_ollama_host_from`] with `VOX_OLLAMA_HOST` read from the
/// environment.
pub fn resolve_ollama_host(flag: Option<&str>, config: Option<&str>) -> ResolvedHost {
    let env = std::env::var(OLLAMA_HOST_ENV).ok();
    resolve_ollama_host_from(flag, env.as_deref(), config)
}

/// Split `host`, `host:port`, `[v6]` or `[v6]:port`. Returns `None` for
/// anything that is not exactly that, in particular for strings carrying
/// userinfo (`localhost:1@evil.example`), a path or a non-numeric port.
fn split_host_port(addr: &str) -> Option<(&str, Option<u16>)> {
    let addr = addr.trim();
    if addr.is_empty()
        || addr
            .chars()
            .any(|c| c.is_whitespace() || matches!(c, '@' | '/' | '?' | '#' | '\\' | '%'))
    {
        return None;
    }
    let parse_port = |p: &str| -> Option<u16> {
        (!p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
            .then(|| p.parse().ok())
            .flatten()
    };
    if let Some(rest) = addr.strip_prefix('[') {
        let (host, tail) = rest.split_once(']')?;
        return match tail {
            "" => Some((host, None)),
            _ => Some((host, Some(parse_port(tail.strip_prefix(':')?)?))),
        };
    }
    match addr.matches(':').count() {
        0 => Some((addr, None)),
        1 => {
            let (host, port) = addr.split_once(':')?;
            (!host.is_empty()).then_some((host, Some(parse_port(port)?)))
        }
        // A bare IPv6 address.
        _ => Some((addr, None)),
    }
}

/// Whether `addr` (`host` or `host:port`) names this machine: `localhost` or
/// a loopback IP address. Other names are not resolved, so a hostname that
/// happens to point at 127.0.0.1 is not loopback. Malformed addresses are
/// never loopback.
pub fn is_loopback_host(addr: &str) -> bool {
    let Some((host, _)) = split_host_port(addr) else {
        return false;
    };
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    host == "localhost" || host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

// ---------------------------------------------------------------------------
// Offline mode
// ---------------------------------------------------------------------------

const OFFLINE_UNSET: u8 = 0;
const OFFLINE_OFF: u8 = 1;
const OFFLINE_ON: u8 = 2;

static OFFLINE: AtomicU8 = AtomicU8::new(OFFLINE_UNSET);

/// Whether a `VOX_OFFLINE` value switches offline mode on.
pub fn offline_env_value_enabled(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Switch offline mode on or off for this process (`--offline`). Overrides
/// `VOX_OFFLINE`.
pub fn set_offline(on: bool) {
    OFFLINE.store(if on { OFFLINE_ON } else { OFFLINE_OFF }, Ordering::Relaxed);
}

/// Whether offline mode is on: [`set_offline`] if it was called, otherwise
/// `VOX_OFFLINE` (read once).
pub fn offline() -> bool {
    match OFFLINE.load(Ordering::Relaxed) {
        OFFLINE_ON => true,
        OFFLINE_OFF => false,
        _ => {
            let on = std::env::var(OFFLINE_ENV).is_ok_and(|v| offline_env_value_enabled(&v));
            // Only fill in the value if nobody set it in the meantime.
            let _ = OFFLINE.compare_exchange(
                OFFLINE_UNSET,
                if on { OFFLINE_ON } else { OFFLINE_OFF },
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
            offline()
        }
    }
}

/// An outbound connection refused by offline mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EgressDenied {
    /// The LLM address is not loopback.
    NonLoopbackHost { host: String },
    /// Model downloads are off.
    Download { target: String },
}

impl fmt::Display for EgressDenied {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EgressDenied::NonLoopbackHost { host } => write!(
                f,
                "offline mode: refusing to connect to the LLM at '{host}' because it is not a \
                 loopback address. Use a runtime on this machine, or drop \
                 --offline / {OFFLINE_ENV} to allow it."
            ),
            EgressDenied::Download { target } => write!(
                f,
                "offline mode: refusing to download '{target}'. Drop --offline / {OFFLINE_ENV} \
                 to download it, or copy the file into the models directory by hand \
                 (see `vox models path`)."
            ),
        }
    }
}

impl std::error::Error for EgressDenied {}

/// Check an LLM address against an explicit offline setting.
pub fn check_host(host: &str, offline: bool) -> Result<(), EgressDenied> {
    if offline && !is_loopback_host(host) {
        return Err(EgressDenied::NonLoopbackHost {
            host: host.to_string(),
        });
    }
    Ok(())
}

/// Check a download against an explicit offline setting.
pub fn check_download(target: &str, offline: bool) -> Result<(), EgressDenied> {
    if offline {
        return Err(EgressDenied::Download {
            target: target.to_string(),
        });
    }
    Ok(())
}

/// Fail if offline mode forbids talking to `host`. Call this at startup so a
/// misconfiguration is reported before any model is loaded;
/// the shared request builder enforces the same rule on every request.
pub fn ensure_ollama_host_allowed(host: &str) -> Result<(), EgressDenied> {
    check_host(host, offline())
}

// ---------------------------------------------------------------------------
// Ledger
// ---------------------------------------------------------------------------

/// One outbound connection (or refused attempt).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LedgerEntry {
    /// Position in the ledger, starting at 1.
    pub seq: u64,
    /// When the connection was opened, in milliseconds since the Unix epoch.
    pub started_unix_ms: u64,
    /// `host:port` the request was sent to.
    pub destination: String,
    /// Host that actually served the response when a redirect led elsewhere
    /// (model downloads are redirected to a CDN).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redirected_to: Option<String>,
    /// Why the connection was made, e.g. `llm chat` or `model download: kokoro`.
    pub purpose: String,
    /// Whether the destination is this machine.
    pub loopback: bool,
    /// Refused by offline mode; nothing was sent.
    pub blocked: bool,
    pub bytes_sent: u64,
    pub bytes_received: u64,
}

/// Totals for one (destination, purpose) pair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LedgerSummary {
    pub destination: String,
    pub purpose: String,
    pub loopback: bool,
    /// Connections made (refused attempts are counted in `blocked`).
    pub connections: u64,
    pub blocked: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub last_unix_ms: u64,
}

/// Totals over everything recorded.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct LedgerTotals {
    pub connections: u64,
    /// Connections that left this machine.
    pub non_loopback_connections: u64,
    pub blocked: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    /// Bytes sent to destinations other than this machine.
    pub non_loopback_bytes_sent: u64,
}

/// A point-in-time copy of the ledger.
#[derive(Debug, Clone, Serialize)]
pub struct LedgerSnapshot {
    /// Whether offline mode is on.
    pub offline: bool,
    pub totals: LedgerTotals,
    /// One row per (destination, purpose), sorted by destination.
    pub destinations: Vec<LedgerSummary>,
    /// The most recent connections, oldest first (at most [`LEDGER_CAPACITY`]).
    pub recent: Vec<LedgerEntry>,
    /// Individual entries dropped from `recent` because the ledger was full.
    pub dropped_entries: u64,
}

#[derive(Default)]
struct LedgerInner {
    next_seq: u64,
    entries: VecDeque<LedgerEntry>,
    summaries: BTreeMap<(String, String), LedgerSummary>,
    totals: LedgerTotals,
    dropped: u64,
}

/// Record of outbound connections. Use [`ledger`] for the process-wide one.
#[derive(Default)]
pub struct PrivacyLedger {
    inner: Mutex<LedgerInner>,
}

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

impl PrivacyLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// A panicking recorder must not take the ledger down with it.
    fn lock(&self) -> MutexGuard<'_, LedgerInner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn push(&self, destination: &str, purpose: &str, bytes_sent: u64, blocked: bool) -> u64 {
        let loopback = is_loopback_host(destination);
        let now = now_unix_ms();
        let mut inner = self.lock();
        inner.next_seq += 1;
        let seq = inner.next_seq;

        let summary = inner
            .summaries
            .entry((destination.to_string(), purpose.to_string()))
            .or_insert_with(|| LedgerSummary {
                destination: destination.to_string(),
                purpose: purpose.to_string(),
                loopback,
                connections: 0,
                blocked: 0,
                bytes_sent: 0,
                bytes_received: 0,
                last_unix_ms: now,
            });
        summary.last_unix_ms = now;
        if blocked {
            summary.blocked += 1;
            inner.totals.blocked += 1;
        } else {
            summary.connections += 1;
            summary.bytes_sent += bytes_sent;
            inner.totals.connections += 1;
            inner.totals.bytes_sent += bytes_sent;
            if !loopback {
                inner.totals.non_loopback_connections += 1;
                inner.totals.non_loopback_bytes_sent += bytes_sent;
            }
        }

        if inner.entries.len() >= LEDGER_CAPACITY {
            inner.entries.pop_front();
            inner.dropped += 1;
        }
        inner.entries.push_back(LedgerEntry {
            seq,
            started_unix_ms: now,
            destination: destination.to_string(),
            redirected_to: None,
            purpose: purpose.to_string(),
            loopback,
            blocked,
            bytes_sent,
            bytes_received: 0,
        });
        seq
    }

    /// Record a connection to `destination` (`host:port`) that is about to be
    /// made, with `bytes_sent` bytes of request body. Report the response
    /// through the returned handle.
    pub fn begin(&self, destination: &str, purpose: &str, bytes_sent: u64) -> Connection<'_> {
        let seq = self.push(destination, purpose, bytes_sent, false);
        Connection {
            ledger: self,
            seq,
            key: (destination.to_string(), purpose.to_string()),
        }
    }

    /// Record an attempt that offline mode refused.
    pub fn record_blocked(&self, destination: &str, purpose: &str) {
        self.push(destination, purpose, 0, true);
    }

    pub fn snapshot(&self) -> LedgerSnapshot {
        let inner = self.lock();
        LedgerSnapshot {
            offline: offline(),
            totals: inner.totals.clone(),
            destinations: inner.summaries.values().cloned().collect(),
            recent: inner.entries.iter().cloned().collect(),
            dropped_entries: inner.dropped,
        }
    }

    /// Whether nothing has been recorded.
    pub fn is_empty(&self) -> bool {
        self.lock().next_seq == 0
    }
}

/// Handle for one recorded connection; reports what came back.
pub struct Connection<'a> {
    ledger: &'a PrivacyLedger,
    seq: u64,
    key: (String, String),
}

impl Connection<'_> {
    fn update(&self, f: impl Fn(&mut LedgerEntry)) {
        let mut inner = self.ledger.lock();
        // Entries are ordered by seq; the one we want is usually near the end.
        if let Some(entry) = inner.entries.iter_mut().rev().find(|e| e.seq == self.seq) {
            f(entry);
        }
    }

    /// Add `bytes` of response body.
    pub fn add_received(&self, bytes: u64) {
        if bytes == 0 {
            return;
        }
        self.update(|e| e.bytes_received += bytes);
        let mut inner = self.ledger.lock();
        inner.totals.bytes_received += bytes;
        if let Some(summary) = inner.summaries.get_mut(&self.key) {
            summary.bytes_received += bytes;
        }
    }

    /// Note the URL that served the response. Recorded only when its host
    /// differs from the destination the request was sent to.
    pub fn served_by(&self, final_url: &str) {
        let served = destination_of(final_url);
        if served != self.key.0 {
            self.update(|e| e.redirected_to = Some(served.clone()));
        }
    }
}

/// The process-wide ledger.
pub fn ledger() -> &'static PrivacyLedger {
    static LEDGER: OnceLock<PrivacyLedger> = OnceLock::new();
    LEDGER.get_or_init(PrivacyLedger::new)
}

/// `host:port` of a URL (the default port is filled in for http/https).
/// Input without a scheme is returned up to the first `/`.
pub fn destination_of(url: &str) -> String {
    let (default_port, rest) = if let Some(r) = url.strip_prefix("https://") {
        (Some(443), r)
    } else if let Some(r) = url.strip_prefix("http://") {
        (Some(80), r)
    } else {
        (None, url)
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or(rest);
    let authority = authority.rsplit('@').next().unwrap_or(authority);
    match (split_host_port(authority), default_port) {
        (Some((_, None)), Some(port)) => format!("{authority}:{port}"),
        _ => authority.to_string(),
    }
}

/// Size of `body` once serialized as JSON, for [`ollama_request`].
pub fn json_len<T: Serialize + ?Sized>(body: &T) -> u64 {
    serde_json::to_vec(body)
        .map(|v| v.len() as u64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// The two egress points
// ---------------------------------------------------------------------------

/// An Ollama request could not be prepared.
#[derive(Debug, thiserror::Error)]
pub enum OllamaRequestError {
    #[error(transparent)]
    Denied(#[from] EgressDenied),
    #[error("failed to initialize the offline LLM client: {0}")]
    Client(#[source] &'static reqwest::Error),
}

/// Prepare and record an Ollama request. The supplied client is used online.
/// Offline requests use Vox's own client to enforce loopback transport.
pub fn ollama_http_request(
    online_client: &reqwest::Client,
    method: reqwest::Method,
    host: &str,
    path: &str,
    purpose: &str,
    bytes_sent: u64,
) -> Result<(reqwest::RequestBuilder, Connection<'static>), OllamaRequestError> {
    llm_http_request(online_client, method, host, path, purpose, bytes_sent)
}

/// Prepare and record an LLM request with the shared offline transport.
pub(crate) fn llm_http_request(
    online_client: &reqwest::Client,
    method: reqwest::Method,
    host: &str,
    path: &str,
    purpose: &str,
    bytes_sent: u64,
) -> Result<(reqwest::RequestBuilder, Connection<'static>), OllamaRequestError> {
    let offline = offline();
    let ledger = ledger();
    let url = checked_llm_url(ledger, offline, host, path, purpose)?;
    let client = if offline {
        offline_llm_client()?
    } else {
        online_client
    };
    Ok((
        client.request(method, url),
        ledger.begin(host, purpose, bytes_sent),
    ))
}

struct LoopbackResolver;

impl reqwest::dns::Resolve for LoopbackResolver {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let is_localhost = name
            .as_str()
            .trim_end_matches('.')
            .eq_ignore_ascii_case("localhost");
        Box::pin(async move {
            if !is_localhost {
                return Err(std::io::Error::other(
                    "offline LLM connections only resolve localhost",
                )
                .into());
            }
            let addresses: reqwest::dns::Addrs = Box::new(
                [
                    std::net::SocketAddr::from((std::net::Ipv6Addr::LOCALHOST, 0)),
                    std::net::SocketAddr::from((std::net::Ipv4Addr::LOCALHOST, 0)),
                ]
                .into_iter(),
            );
            Ok(addresses)
        })
    }
}

fn offline_llm_client() -> Result<&'static reqwest::Client, OllamaRequestError> {
    static CLIENT: OnceLock<Result<reqwest::Client, reqwest::Error>> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .dns_resolver(std::sync::Arc::new(LoopbackResolver))
                .connect_timeout(std::time::Duration::from_secs(5))
                .build()
        })
        .as_ref()
        .map_err(OllamaRequestError::Client)
}

/// Start a request to Ollama: the one place that turns an Ollama address
/// into a URL.
///
/// `path` is the API path (`/api/chat`), `purpose` a short label for the
/// ledger, `bytes_sent` the size of the request body (see [`json_len`]).
/// In offline mode a non-loopback `host` is refused and the attempt is
/// recorded as blocked.
pub fn ollama_request(
    host: &str,
    path: &str,
    purpose: &str,
    bytes_sent: u64,
) -> Result<(String, Connection<'static>), EgressDenied> {
    ollama_request_in(ledger(), offline(), host, path, purpose, bytes_sent)
}

/// [`ollama_request`] against an explicit ledger and offline setting.
pub fn ollama_request_in<'a>(
    ledger: &'a PrivacyLedger,
    offline: bool,
    host: &str,
    path: &str,
    purpose: &str,
    bytes_sent: u64,
) -> Result<(String, Connection<'a>), EgressDenied> {
    let url = checked_llm_url(ledger, offline, host, path, purpose)?;
    Ok((url, ledger.begin(host, purpose, bytes_sent)))
}

fn checked_llm_url(
    ledger: &PrivacyLedger,
    offline: bool,
    host: &str,
    path: &str,
    purpose: &str,
) -> Result<String, EgressDenied> {
    if let Err(denied) = check_host(host, offline) {
        ledger.record_blocked(host, purpose);
        return Err(denied);
    }
    let url = if offline && host.parse::<std::net::Ipv6Addr>().is_ok() {
        format!("http://[{host}]{path}")
    } else {
        format!("http://{host}{path}")
    };
    // IP literals bypass DNS, so also check the authority of the final URL.
    if offline
        && !reqwest::Url::parse(&url).is_ok_and(|url| url.host_str().is_some_and(is_loopback_host))
    {
        ledger.record_blocked(host, purpose);
        return Err(EgressDenied::NonLoopbackHost {
            host: host.to_string(),
        });
    }
    Ok(url)
}

/// Start a model download from `url`. Refused in offline mode.
pub fn download_request(url: &str, model: &str) -> Result<Connection<'static>, EgressDenied> {
    download_request_in(ledger(), offline(), url, model)
}

/// [`download_request`] against an explicit ledger and offline setting.
pub fn download_request_in<'a>(
    ledger: &'a PrivacyLedger,
    offline: bool,
    url: &str,
    model: &str,
) -> Result<Connection<'a>, EgressDenied> {
    let destination = destination_of(url);
    let purpose = format!("model download: {model}");
    if let Err(denied) = check_download(model, offline) {
        ledger.record_blocked(&destination, &purpose);
        return Err(denied);
    }
    Ok(ledger.begin(&destination, &purpose, 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_resolution_priority_is_flag_env_config_default() {
        let r = resolve_ollama_host_from(Some("flag:1"), Some("env:2"), Some("cfg:3"));
        assert_eq!((r.host.as_str(), r.source), ("flag:1", HostSource::Flag));

        let r = resolve_ollama_host_from(None, Some("env:2"), Some("cfg:3"));
        assert_eq!((r.host.as_str(), r.source), ("env:2", HostSource::Env));

        let r = resolve_ollama_host_from(None, None, Some("cfg:3"));
        assert_eq!((r.host.as_str(), r.source), ("cfg:3", HostSource::Config));

        let r = resolve_ollama_host_from(None, None, None);
        assert_eq!(
            (r.host.as_str(), r.source),
            (DEFAULT_OLLAMA_HOST, HostSource::Default)
        );
    }

    #[test]
    fn blank_sources_fall_through_and_values_are_normalized() {
        let r = resolve_ollama_host_from(Some("  "), Some(""), Some(" http://box:1234/api "));
        assert_eq!(
            (r.host.as_str(), r.source),
            ("box:1234", HostSource::Config)
        );

        let r = resolve_ollama_host_from(Some(""), Some("https://gpu.lan:443/"), None);
        assert_eq!(
            (r.host.as_str(), r.source),
            ("gpu.lan:443", HostSource::Env)
        );

        assert_eq!(normalize_ollama_host(""), DEFAULT_OLLAMA_HOST);
        assert_eq!(normalize_ollama_host("localhost:11434"), "localhost:11434");
    }

    #[test]
    fn loopback_detection() {
        for host in [
            "localhost",
            "localhost:11434",
            "LOCALHOST:11434",
            "localhost.:11434",
            "127.0.0.1",
            "127.0.0.1:11434",
            "127.8.9.10:1",
            "::1",
            "[::1]",
            "[::1]:11434",
        ] {
            assert!(is_loopback_host(host), "{host} should be loopback");
        }
        for host in [
            "",
            "gpu-box:11434",
            "10.0.0.5:11434",
            "192.168.1.2",
            "0.0.0.0:11434",
            "example.com",
            "localhost.example.com:11434",
            "[2001:db8::1]:11434",
            // Would be sent to evil.example with "localhost:11434" as userinfo.
            "localhost:11434@evil.example",
            "localhost@evil.example",
            "127.0.0.1:11434@evil.example:80",
            "localhost:notaport",
            "localhost:",
            "localhost:11434/path",
            "localhost :11434",
            "[::1]x",
            "[::1]:x",
        ] {
            assert!(!is_loopback_host(host), "{host} should not be loopback");
        }
    }

    #[test]
    fn offline_refuses_non_loopback_hosts_only() {
        assert!(check_host("localhost:11434", true).is_ok());
        assert!(check_host("[::1]:11434", true).is_ok());
        assert_eq!(
            check_host("gpu-box:11434", true),
            Err(EgressDenied::NonLoopbackHost {
                host: "gpu-box:11434".into()
            })
        );
        assert!(check_host("gpu-box:11434", false).is_ok());

        let msg = check_host("10.0.0.5:11434", true).unwrap_err().to_string();
        assert!(msg.contains("offline mode"), "{msg}");
        assert!(msg.contains("10.0.0.5:11434"), "{msg}");
    }

    #[test]
    fn offline_refuses_every_download() {
        assert!(check_download("kokoro", false).is_ok());
        let err = check_download("kokoro", true).unwrap_err();
        assert_eq!(
            err,
            EgressDenied::Download {
                target: "kokoro".into()
            }
        );
        assert!(err.to_string().contains("offline mode"));
        assert!(err.to_string().contains("kokoro"));
    }

    #[test]
    fn offline_env_values() {
        for v in ["1", "true", "TRUE", " yes ", "on"] {
            assert!(offline_env_value_enabled(v), "{v}");
        }
        for v in ["", "0", "false", "no", "off", "2"] {
            assert!(!offline_env_value_enabled(v), "{v}");
        }
    }

    #[test]
    fn ledger_records_a_request_and_its_response() {
        let ledger = PrivacyLedger::new();
        assert!(ledger.is_empty());

        let (url, conn) = ollama_request_in(
            &ledger,
            false,
            "localhost:11434",
            "/api/chat",
            "llm chat",
            120,
        )
        .unwrap();
        assert_eq!(url, "http://localhost:11434/api/chat");
        conn.add_received(300);
        conn.add_received(50);

        let snap = ledger.snapshot();
        assert_eq!(snap.recent.len(), 1);
        let entry = &snap.recent[0];
        assert_eq!(entry.seq, 1);
        assert_eq!(entry.destination, "localhost:11434");
        assert_eq!(entry.purpose, "llm chat");
        assert!(entry.loopback);
        assert!(!entry.blocked);
        assert_eq!((entry.bytes_sent, entry.bytes_received), (120, 350));
        assert!(entry.started_unix_ms > 0);

        assert_eq!(snap.destinations.len(), 1);
        assert_eq!(snap.destinations[0].connections, 1);
        assert_eq!(snap.destinations[0].bytes_received, 350);
        assert_eq!(snap.totals.connections, 1);
        assert_eq!(snap.totals.non_loopback_connections, 0);
        assert_eq!(snap.totals.bytes_sent, 120);
        assert_eq!(snap.totals.bytes_received, 350);
    }

    #[test]
    fn ledger_separates_remote_destinations_and_aggregates_by_purpose() {
        let ledger = PrivacyLedger::new();
        for _ in 0..3 {
            let (_, c) =
                ollama_request_in(&ledger, false, "gpu-box:11434", "/api/chat", "llm chat", 10)
                    .unwrap();
            c.add_received(5);
        }
        ollama_request_in(
            &ledger,
            false,
            "gpu-box:11434",
            "/api/tags",
            "llm model list",
            0,
        )
        .unwrap();
        ollama_request_in(
            &ledger,
            false,
            "127.0.0.1:11434",
            "/api/tags",
            "llm model list",
            0,
        )
        .unwrap();

        let snap = ledger.snapshot();
        assert_eq!(snap.recent.len(), 5);
        assert_eq!(snap.destinations.len(), 3);
        let chat = snap
            .destinations
            .iter()
            .find(|s| s.destination == "gpu-box:11434" && s.purpose == "llm chat")
            .unwrap();
        assert_eq!(
            (chat.connections, chat.bytes_sent, chat.bytes_received),
            (3, 30, 15)
        );
        assert!(!chat.loopback);
        assert_eq!(snap.totals.connections, 5);
        assert_eq!(snap.totals.non_loopback_connections, 4);
        assert_eq!(snap.totals.non_loopback_bytes_sent, 30);
    }

    #[test]
    fn offline_requests_are_refused_and_recorded_as_blocked() {
        let ledger = PrivacyLedger::new();
        let err = ollama_request_in(&ledger, true, "gpu-box:11434", "/api/chat", "llm chat", 99)
            .err()
            .expect("non-loopback host must be refused offline");
        assert!(matches!(err, EgressDenied::NonLoopbackHost { .. }));

        let err = download_request_in(
            &ledger,
            true,
            "https://huggingface.co/x/resolve/abc/model.onnx",
            "kokoro",
        )
        .err()
        .expect("downloads must be refused offline");
        assert!(matches!(err, EgressDenied::Download { .. }));

        // Loopback Ollama keeps working offline.
        assert!(
            ollama_request_in(&ledger, true, "localhost:11434", "/api/chat", "llm chat", 7).is_ok()
        );

        let snap = ledger.snapshot();
        assert_eq!(snap.totals.blocked, 2);
        assert_eq!(snap.totals.connections, 1);
        assert_eq!(snap.totals.bytes_sent, 7, "blocked attempts send nothing");
        let blocked: Vec<_> = snap.recent.iter().filter(|e| e.blocked).collect();
        assert_eq!(blocked.len(), 2);
        assert_eq!(blocked[0].destination, "gpu-box:11434");
        assert_eq!(blocked[0].bytes_sent, 0);
        assert_eq!(blocked[1].destination, "huggingface.co:443");
        assert_eq!(blocked[1].purpose, "model download: kokoro");
    }

    #[test]
    fn download_records_destination_redirect_and_bytes() {
        let ledger = PrivacyLedger::new();
        let conn = download_request_in(
            &ledger,
            false,
            "https://huggingface.co/x/resolve/abc/model.onnx",
            "kokoro",
        )
        .unwrap();
        conn.served_by("https://cdn-lfs.hf.co/repos/aa/bb?sig=1");
        conn.add_received(1024);
        // Same host: not a redirect worth recording.
        let same = download_request_in(&ledger, false, "http://127.0.0.1:9/f", "test").unwrap();
        same.served_by("http://127.0.0.1:9/other");

        let snap = ledger.snapshot();
        assert_eq!(snap.recent[0].destination, "huggingface.co:443");
        assert_eq!(
            snap.recent[0].redirected_to.as_deref(),
            Some("cdn-lfs.hf.co:443")
        );
        assert_eq!(snap.recent[0].bytes_received, 1024);
        assert!(!snap.recent[0].loopback);
        assert_eq!(snap.recent[1].destination, "127.0.0.1:9");
        assert_eq!(snap.recent[1].redirected_to, None);
        assert!(snap.recent[1].loopback);
    }

    #[test]
    fn ledger_is_bounded_but_totals_are_not() {
        let ledger = PrivacyLedger::new();
        let first = ledger.begin("localhost:11434", "llm chat", 1);
        for _ in 0..LEDGER_CAPACITY + 9 {
            ledger.begin("localhost:11434", "llm chat", 1);
        }
        // The first entry was evicted; reporting on it must still reach the totals.
        first.add_received(40);

        let snap = ledger.snapshot();
        assert_eq!(snap.recent.len(), LEDGER_CAPACITY);
        assert_eq!(snap.dropped_entries, 10);
        assert_eq!(snap.recent[0].seq, 11);
        assert_eq!(snap.totals.connections, LEDGER_CAPACITY as u64 + 10);
        assert_eq!(snap.totals.bytes_received, 40);
        assert_eq!(snap.destinations[0].bytes_received, 40);
    }

    #[test]
    fn destination_of_urls() {
        assert_eq!(
            destination_of("https://huggingface.co/a/b"),
            "huggingface.co:443"
        );
        assert_eq!(destination_of("http://example.com"), "example.com:80");
        assert_eq!(destination_of("http://127.0.0.1:9/unused"), "127.0.0.1:9");
        assert_eq!(
            destination_of("https://user:pw@host.example/x"),
            "host.example:443"
        );
        assert_eq!(destination_of("http://[::1]:8080/x"), "[::1]:8080");
        assert_eq!(destination_of("localhost:11434"), "localhost:11434");
    }

    #[test]
    fn snapshot_serializes_to_the_documented_shape() {
        let ledger = PrivacyLedger::new();
        ledger
            .begin("localhost:11434", "llm chat", 3)
            .add_received(4);
        let json = serde_json::to_value(ledger.snapshot()).unwrap();
        assert!(json["offline"].is_boolean());
        assert_eq!(json["totals"]["connections"], 1);
        assert_eq!(json["destinations"][0]["destination"], "localhost:11434");
        assert_eq!(json["recent"][0]["purpose"], "llm chat");
        assert_eq!(json["recent"][0]["bytes_sent"], 3);
        assert_eq!(json["recent"][0]["bytes_received"], 4);
        assert!(json["recent"][0].get("redirected_to").is_none());
    }

    #[test]
    fn json_len_matches_serialized_size() {
        let body = serde_json::json!({"model": "m", "stream": true});
        assert_eq!(
            json_len(&body),
            serde_json::to_vec(&body).unwrap().len() as u64
        );
    }
}
