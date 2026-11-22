//! Access control for a server that has no authentication.
//!
//! vox listens on loopback and trusts whoever can reach it, so two browser
//! attacks have to be closed off:
//!
//! - **Cross-site requests.** Any web page can `fetch()` or open a WebSocket
//!   to `http://127.0.0.1:3000`. Browsers attach an `Origin` header to those
//!   requests; we reject every origin that is not the server itself.
//! - **DNS rebinding.** A page on `evil.example` re-points its own name at
//!   127.0.0.1 and becomes "same-origin" with the server. The `Host` header
//!   still says `evil.example`, so we reject every host name we do not know.
//!
//! Requests without an `Origin` header (curl, SDKs, same-origin `GET`s) are
//! allowed: they do not come from a cross-site script.

use std::net::IpAddr;
use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, Method, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use tower_http::cors::{AllowOrigin, Any, CorsLayer};

use super::ServerState;
use super::error::ServerError;

/// Which `Host` and `Origin` header values the server accepts.
#[derive(Debug, Clone)]
pub struct AccessPolicy {
    /// Port the server is bound to.
    port: u16,
    /// Lower-case host names accepted on `port`: the loopback names plus the
    /// bind host.
    hosts: Vec<String>,
    /// The server is bound to `0.0.0.0` / `::`, so clients reach it through
    /// an address we cannot enumerate. Any IP literal is accepted then: a
    /// rebinding attack needs a DNS *name* in the `Host` header.
    any_ip_literal: bool,
    /// Operator-supplied hosts (`name`, optional port).
    extra_hosts: Vec<(String, Option<u16>)>,
    /// Operator-supplied origins, accepted verbatim.
    extra_origins: Vec<String>,
}

impl AccessPolicy {
    /// Policy for a server bound to `bind_host:port`.
    pub fn new(bind_host: &str, port: u16) -> Self {
        let mut hosts = vec![
            "localhost".to_string(),
            "127.0.0.1".to_string(),
            "[::1]".to_string(),
        ];
        let bind = normalize_host_name(bind_host);
        let any_ip_literal = bind
            .trim_matches(['[', ']'])
            .parse::<IpAddr>()
            .is_ok_and(|ip| ip.is_unspecified());
        if !any_ip_literal && !bind.is_empty() && !hosts.contains(&bind) {
            hosts.push(bind);
        }
        Self {
            port,
            hosts,
            any_ip_literal,
            extra_hosts: Vec::new(),
            extra_origins: Vec::new(),
        }
    }

    /// Also accept these `Host` values (`name` matches any port, `name:port`
    /// only that port). Meant for a reverse proxy's public name.
    pub fn with_extra_hosts(mut self, hosts: impl IntoIterator<Item = String>) -> Self {
        for h in hosts {
            if let Some((name, port)) = split_host_port(&h) {
                self.extra_hosts.push((name, port));
            }
        }
        self
    }

    /// Also accept these `Origin` values verbatim (`https://app.example`).
    pub fn with_extra_origins(mut self, origins: impl IntoIterator<Item = String>) -> Self {
        for o in origins {
            let o = o.trim().trim_end_matches('/').to_ascii_lowercase();
            if !o.is_empty() {
                self.extra_origins.push(o);
            }
        }
        self
    }

    fn name_port_allowed(&self, name: &str, port: u16) -> bool {
        if self
            .extra_hosts
            .iter()
            .any(|(n, p)| n == name && p.is_none_or(|p| p == port))
        {
            return true;
        }
        if port != self.port {
            return false;
        }
        self.hosts.iter().any(|h| h == name)
            || (self.any_ip_literal && name.trim_matches(['[', ']']).parse::<IpAddr>().is_ok())
    }

    /// Whether a `Host` header value names this server.
    pub fn host_allowed(&self, host: &str) -> bool {
        match split_host_port(host) {
            // We only speak plain HTTP, so a missing port means 80.
            Some((name, port)) => self.name_port_allowed(&name, port.unwrap_or(80)),
            None => false,
        }
    }

    /// Whether an `Origin` header value is the server itself (or an origin
    /// the operator added).
    pub fn origin_allowed(&self, origin: &str) -> bool {
        let origin = origin.trim().trim_end_matches('/').to_ascii_lowercase();
        if self.extra_origins.contains(&origin) {
            return true;
        }
        let (default_port, rest) = if let Some(rest) = origin.strip_prefix("http://") {
            (80, rest)
        } else if let Some(rest) = origin.strip_prefix("https://") {
            (443, rest)
        } else {
            // "null", file://, chrome-extension://, ...
            return false;
        };
        match split_host_port(rest) {
            Some((name, port)) => self.name_port_allowed(&name, port.unwrap_or(default_port)),
            None => false,
        }
    }

    /// Check a request's headers. `uri_host` is the authority of an
    /// absolute-form / HTTP/2 request, used when there is no `Host` header.
    pub fn check(&self, headers: &HeaderMap, uri_host: Option<&str>) -> Result<(), ServerError> {
        let host = match headers.get(header::HOST) {
            Some(v) => Some(v.to_str().map_err(|_| {
                ServerError::forbidden("Host header is not valid").with_code("host_not_allowed")
            })?),
            None => uri_host,
        };
        // A request with no host at all cannot come from a browser, and a
        // rebinding attack needs one.
        if let Some(host) = host
            && !self.host_allowed(host)
        {
            return Err(ServerError::forbidden(format!(
                "Host '{host}' is not allowed. vox only answers to localhost and its bind \
                     address; set VOX_ALLOWED_HOSTS to add a name."
            ))
            .with_code("host_not_allowed"));
        }

        if let Some(origin) = headers.get(header::ORIGIN) {
            let allowed = origin.to_str().is_ok_and(|o| self.origin_allowed(o));
            if !allowed {
                return Err(ServerError::forbidden(
                    "cross-origin requests are not allowed. Set VOX_ALLOWED_ORIGINS to trust \
                     another origin.",
                )
                .with_code("origin_not_allowed"));
            }
        }
        Ok(())
    }
}

/// Lower-case a host name, drop a trailing dot, bracket a bare IPv6 address.
fn normalize_host_name(name: &str) -> String {
    let name = name.trim().trim_end_matches('.').to_ascii_lowercase();
    if !name.starts_with('[') && name.contains(':') && name.parse::<IpAddr>().is_ok() {
        format!("[{name}]")
    } else {
        name
    }
}

/// Split `name`, `name:port`, `[v6]` or `[v6]:port` into a normalized name
/// and an optional port. Returns `None` for anything malformed.
pub fn split_host_port(raw: &str) -> Option<(String, Option<u16>)> {
    let raw = raw.trim();
    if raw.is_empty() || raw.contains(['/', '@', ' ', '?', '#']) {
        return None;
    }
    let (name, port) = if let Some(rest) = raw.strip_prefix('[') {
        let (addr, tail) = rest.split_once(']')?;
        let port = match tail {
            "" => None,
            t => Some(t.strip_prefix(':')?.parse::<u16>().ok()?),
        };
        (format!("[{addr}]"), port)
    } else {
        match raw.rsplit_once(':') {
            // A bare IPv6 address without brackets has several colons.
            Some((n, _)) if n.contains(':') => (raw.to_string(), None),
            Some((n, p)) => (n.to_string(), Some(p.parse::<u16>().ok()?)),
            None => (raw.to_string(), None),
        }
    };
    let name = normalize_host_name(&name);
    if name.is_empty() {
        None
    } else {
        Some((name, port))
    }
}

/// Whether `host` (a bind address) only accepts connections from this machine.
pub fn is_loopback_host(host: &str) -> bool {
    let name = normalize_host_name(host);
    name == "localhost"
        || name
            .trim_matches(['[', ']'])
            .parse::<IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

/// Middleware: reject requests whose `Host` or `Origin` fails the policy.
pub async fn enforce(
    State(state): State<Arc<ServerState>>,
    request: Request,
    next: Next,
) -> Response {
    let uri_host = request.uri().authority().map(|a| a.as_str().to_string());
    if let Err(err) = state.access.check(request.headers(), uri_host.as_deref()) {
        tracing::warn!(
            path = %request.uri().path(),
            host = ?request.headers().get(header::HOST),
            origin = ?request.headers().get(header::ORIGIN),
            "rejected request: {}",
            err.message()
        );
        return err.into_response();
    }
    let mut response = next.run(request).await;
    response.headers_mut().insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    response
}

/// CORS that mirrors the access policy: only origins the policy accepts get
/// `Access-Control-Allow-Origin`. Replaces the old allow-everything layer.
pub fn cors_layer(policy: &AccessPolicy) -> CorsLayer {
    let policy = policy.clone();
    CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(move |origin, _| {
            origin.to_str().is_ok_and(|o| policy.origin_allowed(o))
        }))
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any)
}

/// Content-Security-Policy for the embedded web UI.
///
/// The UI is a single HTML file with inline `<script>` and `<style>`, builds
/// its AudioWorklet from a `blob:` URL, plays audio from `blob:`/`data:` URLs
/// and talks only to this server. `host` (the validated `Host` header) is
/// spelled out for WebSockets because older Safari does not let `'self'`
/// cover `ws:`.
pub fn content_security_policy(host: Option<&str>) -> String {
    let ws = host
        .filter(|h| split_host_port(h).is_some())
        .map(|h| format!(" ws://{h} wss://{h}"))
        .unwrap_or_default();
    format!(
        "default-src 'self'; \
         script-src 'self' 'unsafe-inline' blob:; \
         style-src 'self' 'unsafe-inline'; \
         img-src 'self' data: blob:; \
         font-src 'self' data:; \
         media-src 'self' blob: data:; \
         worker-src 'self' blob:; \
         connect-src 'self'{ws}; \
         object-src 'none'; \
         base-uri 'none'; \
         form-action 'none'; \
         frame-ancestors 'none'"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(
                axum::http::HeaderName::from_bytes(k.as_bytes()).unwrap(),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        h
    }

    #[test]
    fn splits_hosts_and_ports() {
        assert_eq!(
            split_host_port("LocalHost:3000"),
            Some(("localhost".into(), Some(3000)))
        );
        assert_eq!(
            split_host_port("127.0.0.1"),
            Some(("127.0.0.1".into(), None))
        );
        assert_eq!(
            split_host_port("[::1]:3000"),
            Some(("[::1]".into(), Some(3000)))
        );
        assert_eq!(split_host_port("[::1]"), Some(("[::1]".into(), None)));
        assert_eq!(split_host_port("::1"), Some(("[::1]".into(), None)));
        assert_eq!(
            split_host_port("localhost.:80"),
            Some(("localhost".into(), Some(80)))
        );
        assert_eq!(split_host_port("host:notaport"), None);
        assert_eq!(split_host_port("[::1"), None);
        assert_eq!(split_host_port("user@host"), None);
        assert_eq!(split_host_port(""), None);
    }

    #[test]
    fn loopback_hosts_on_the_bind_port_are_allowed() {
        let p = AccessPolicy::new("127.0.0.1", 3000);
        for host in [
            "localhost:3000",
            "127.0.0.1:3000",
            "[::1]:3000",
            "LOCALHOST:3000",
        ] {
            assert!(p.host_allowed(host), "{host}");
        }
        for host in [
            "localhost:3001",
            "localhost",
            "evil.example:3000",
            "127.0.0.1.evil.example:3000",
            "192.168.1.5:3000",
            "",
        ] {
            assert!(!p.host_allowed(host), "{host}");
        }
    }

    #[test]
    fn bind_host_is_allowed() {
        let p = AccessPolicy::new("192.168.1.5", 8080);
        assert!(p.host_allowed("192.168.1.5:8080"));
        assert!(p.host_allowed("localhost:8080"));
        assert!(!p.host_allowed("192.168.1.6:8080"));
        assert!(p.origin_allowed("http://192.168.1.5:8080"));
    }

    #[test]
    fn port_80_matches_a_host_without_port() {
        let p = AccessPolicy::new("127.0.0.1", 80);
        assert!(p.host_allowed("localhost"));
        assert!(p.origin_allowed("http://localhost"));
        assert!(!p.origin_allowed("https://localhost"));
    }

    #[test]
    fn unspecified_bind_accepts_ip_literals_but_not_names() {
        let p = AccessPolicy::new("0.0.0.0", 3000);
        assert!(p.host_allowed("192.168.1.5:3000"));
        assert!(p.host_allowed("[fe80::1]:3000"));
        assert!(p.host_allowed("localhost:3000"));
        assert!(!p.host_allowed("0.0.0.0.evil.example:3000"));
        assert!(!p.host_allowed("evil.example:3000"));
        assert!(!p.host_allowed("192.168.1.5:9999"));
    }

    #[test]
    fn origins_must_be_the_server_itself() {
        let p = AccessPolicy::new("127.0.0.1", 3000);
        for origin in [
            "http://127.0.0.1:3000",
            "http://localhost:3000",
            "http://[::1]:3000",
            "https://localhost:3000",
            "http://LOCALHOST:3000/",
        ] {
            assert!(p.origin_allowed(origin), "{origin}");
        }
        for origin in [
            "https://evil.example",
            "http://evil.example:3000",
            "http://localhost:5173",
            "http://localhost",
            "null",
            "file://",
            "chrome-extension://abcdef",
            "http://localhost:3000@evil.example",
            "",
        ] {
            assert!(!p.origin_allowed(origin), "{origin}");
        }
    }

    #[test]
    fn operator_can_add_hosts_and_origins() {
        let p = AccessPolicy::new("127.0.0.1", 3000)
            .with_extra_hosts(["vox.lan".to_string(), "proxy.lan:8443".to_string()])
            .with_extra_origins(["https://app.example/".to_string()]);
        assert!(p.host_allowed("vox.lan"));
        assert!(p.host_allowed("vox.lan:1234"));
        assert!(p.host_allowed("proxy.lan:8443"));
        assert!(!p.host_allowed("proxy.lan:3000"));
        assert!(p.origin_allowed("https://app.example"));
        assert!(p.origin_allowed("https://vox.lan"));
        assert!(!p.origin_allowed("https://other.example"));
    }

    #[test]
    fn check_combines_host_and_origin() {
        let p = AccessPolicy::new("127.0.0.1", 3000);
        // No Host, no Origin: a non-browser client.
        assert!(p.check(&headers(&[]), None).is_ok());
        // Origin-less request on loopback.
        assert!(
            p.check(&headers(&[("host", "127.0.0.1:3000")]), None)
                .is_ok()
        );
        assert!(
            p.check(
                &headers(&[
                    ("host", "127.0.0.1:3000"),
                    ("origin", "http://127.0.0.1:3000")
                ]),
                None
            )
            .is_ok()
        );
        let cross = p
            .check(
                &headers(&[
                    ("host", "127.0.0.1:3000"),
                    ("origin", "https://evil.example"),
                ]),
                None,
            )
            .unwrap_err();
        assert_eq!(cross.status(), axum::http::StatusCode::FORBIDDEN);
        assert_eq!(cross.code(), "origin_not_allowed");
        // DNS rebinding: the page is "same-origin" with its own name.
        let rebind = p
            .check(
                &headers(&[
                    ("host", "evil.example:3000"),
                    ("origin", "http://evil.example:3000"),
                ]),
                None,
            )
            .unwrap_err();
        assert_eq!(rebind.code(), "host_not_allowed");
        // HTTP/2 style authority instead of a Host header.
        assert!(p.check(&headers(&[]), Some("evil.example:3000")).is_err());
        assert!(p.check(&headers(&[]), Some("localhost:3000")).is_ok());
    }

    #[test]
    fn loopback_detection() {
        for h in ["127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.2"] {
            assert!(is_loopback_host(h), "{h}");
        }
        for h in ["0.0.0.0", "::", "192.168.1.5", "example.com"] {
            assert!(!is_loopback_host(h), "{h}");
        }
    }

    #[test]
    fn csp_allows_what_the_ui_needs_and_nothing_remote() {
        let csp = content_security_policy(Some("127.0.0.1:3000"));
        assert!(csp.contains("default-src 'self'"));
        assert!(csp.contains("script-src 'self' 'unsafe-inline' blob:"));
        assert!(csp.contains("connect-src 'self' ws://127.0.0.1:3000 wss://127.0.0.1:3000"));
        assert!(csp.contains("frame-ancestors 'none'"));
        assert!(!csp.contains("http://"));
        assert!(!csp.contains("https://"));
        assert!(!csp.contains('*'));
        // A malformed host never reaches the header.
        let csp = content_security_policy(Some("evil.example; script-src *"));
        assert!(csp.contains("connect-src 'self';"));
    }
}
