//! A tiny blocking HTTP/1.1 client over `std::net::TcpStream` (no TLS). Used by the CLI
//! to talk to the daemon and by `OmlxService` for its read-only loopback GETs.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct Response {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Response {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

/// (host, port, path-with-query) of an `http://` URL.
pub fn parse_url(url: &str) -> Result<(String, u16, String), String> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| format!("only http:// URLs are supported: {url}"))?;
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    if authority.contains('@') {
        return Err("user info in URLs is not supported".into());
    }
    let (host, port) = if let Some(stripped) = authority.strip_prefix('[') {
        let end = stripped.find(']').ok_or("bad IPv6 host")?;
        let host = &stripped[..end];
        let after = &stripped[end + 1..];
        let port = match after.strip_prefix(':') {
            Some(p) => p.parse::<u16>().map_err(|_| format!("bad port in {url}"))?,
            None => 80,
        };
        (host.to_string(), port)
    } else {
        match authority.rsplit_once(':') {
            Some((h, p)) => (h.to_string(), p.parse::<u16>().map_err(|_| format!("bad port in {url}"))?),
            None => (authority.to_string(), 80),
        }
    };
    if host.is_empty() {
        return Err(format!("no host in {url}"));
    }
    Ok((host, port, path.to_string()))
}

pub fn request(
    method: &str,
    url: &str,
    headers: &[(&str, &str)],
    body: Option<&[u8]>,
    timeout: Duration,
) -> Result<Response, String> {
    let (host, port, path) = parse_url(url)?;
    let deadline = Instant::now() + timeout;
    let addrs: Vec<_> = (host.as_str(), port)
        .to_socket_addrs()
        .map_err(|e| format!("cannot resolve {host}: {e}"))?
        .collect();
    let mut last_err = format!("no address for {host}");
    let mut stream = None;
    for a in addrs {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            last_err = "timed out".into();
            break;
        }
        match TcpStream::connect_timeout(&a, left) {
            Ok(s) => {
                stream = Some(s);
                break;
            }
            Err(e) => last_err = e.to_string(),
        }
    }
    let mut s = stream.ok_or(last_err)?;
    let host_hdr = if host.contains(':') { format!("[{host}]:{port}") } else { format!("{host}:{port}") };
    let mut req = format!("{method} {path} HTTP/1.1\r\nHost: {host_hdr}\r\nConnection: close\r\nUser-Agent: shutterd\r\n");
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    if let Some(b) = body {
        req.push_str(&format!("Content-Length: {}\r\n", b.len()));
    } else if method != "GET" && method != "HEAD" {
        req.push_str("Content-Length: 0\r\n");
    }
    req.push_str("\r\n");
    let io = |e: std::io::Error| {
        if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) {
            "timed out".to_string()
        } else {
            e.to_string()
        }
    };
    let left = deadline.saturating_duration_since(Instant::now()).max(Duration::from_millis(1));
    s.set_write_timeout(Some(left)).map_err(io)?;
    s.write_all(req.as_bytes()).map_err(io)?;
    if let Some(b) = body {
        s.write_all(b).map_err(io)?;
    }
    let mut raw = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err("timed out".into());
        }
        s.set_read_timeout(Some(left)).map_err(io)?;
        match s.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                raw.extend_from_slice(&buf[..n]);
                if let Some(done) = complete_len(&raw) {
                    raw.truncate(done);
                    break;
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(io(e)),
        }
    }
    parse_response(&raw)
}

fn head_end(raw: &[u8]) -> Option<usize> {
    raw.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 4)
}

/// When the response carries a Content-Length, the total length once it is all here.
fn complete_len(raw: &[u8]) -> Option<usize> {
    let he = head_end(raw)?;
    let head = String::from_utf8_lossy(&raw[..he]);
    for line in head.lines().skip(1) {
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                let n: usize = v.trim().parse().ok()?;
                return (raw.len() >= he + n).then_some(he + n);
            }
        }
    }
    None
}

pub fn parse_response(raw: &[u8]) -> Result<Response, String> {
    let he = head_end(raw).ok_or("incomplete HTTP response")?;
    let head = String::from_utf8_lossy(&raw[..he - 4]).into_owned();
    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap_or("");
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("bad status line: {status_line:?}"))?;
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect();
    let mut body = raw[he..].to_vec();
    let chunked = headers.iter().any(|(k, v)| {
        k.eq_ignore_ascii_case("transfer-encoding") && v.to_ascii_lowercase().contains("chunked")
    });
    if chunked {
        body = dechunk(&body)?;
    } else if let Some((_, v)) = headers.iter().find(|(k, _)| k.eq_ignore_ascii_case("content-length")) {
        if let Ok(n) = v.parse::<usize>() {
            body.truncate(n);
        }
    }
    Ok(Response { status, headers, body })
}

fn dechunk(mut b: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    loop {
        let le = b.windows(2).position(|w| w == b"\r\n").ok_or("bad chunked body")?;
        let size_s = String::from_utf8_lossy(&b[..le]);
        let size_s = size_s.split(';').next().unwrap_or("").trim().to_string();
        let n = usize::from_str_radix(&size_s, 16).map_err(|_| "bad chunk size")?;
        b = &b[le + 2..];
        if n == 0 {
            return Ok(out);
        }
        if b.len() < n {
            // truncated: keep what arrived
            out.extend_from_slice(b);
            return Ok(out);
        }
        out.extend_from_slice(&b[..n]);
        b = &b[n..];
        if b.starts_with(b"\r\n") {
            b = &b[2..];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls() {
        assert_eq!(parse_url("http://127.0.0.1:8000").unwrap(), ("127.0.0.1".into(), 8000, "/".into()));
        assert_eq!(parse_url("http://[::1]:9/v1?x=1").unwrap(), ("::1".into(), 9, "/v1?x=1".into()));
        assert_eq!(parse_url("http://h/p").unwrap(), ("h".into(), 80, "/p".into()));
        assert!(parse_url("https://h").is_err());
    }

    #[test]
    fn responses() {
        let r = parse_response(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhiXX").unwrap();
        assert_eq!((r.status, r.text().as_str()), (200, "hi"));
        let r = parse_response(b"HTTP/1.1 404 NF\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n2\r\nde\r\n0\r\n\r\n").unwrap();
        assert_eq!((r.status, r.text().as_str()), (404, "abcde"));
    }
}
