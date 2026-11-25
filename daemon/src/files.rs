//! Uploaded input files (POST /v1/files) and helpers the upload and image routes need:
//! multipart/form-data parsing, base64, sha256. Same layout and rules as
//! `shutter/core/files.py`: an upload lives at `<work_dir>/uploads/<id>/<name>`, and a request's
//! `inputs` may name it by its path or by its bare 16-hex id.

use crate::util::now;
use serde_json::{json, Value};
use std::io::Read;
use std::path::{Path, PathBuf};

pub fn content_type_for(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref() {
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        Some("gif") => "image/gif",
        Some("mp4") => "video/mp4",
        Some("mov") => "video/quicktime",
        Some("webm") => "video/webm",
        Some("wav") => "audio/wav",
        Some("mp3") => "audio/mpeg",
        Some("flac") => "audio/flac",
        Some("m4a") => "audio/mp4",
        Some("json") => "application/json",
        Some("txt") => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

fn ext_for_type(ct: &str) -> Option<&'static str> {
    let ct = ct.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
    Some(match ct.as_str() {
        "image/png" => ".png",
        "image/jpeg" => ".jpg",
        "image/webp" => ".webp",
        "image/gif" => ".gif",
        "video/mp4" => ".mp4",
        "video/quicktime" => ".mov",
        "video/webm" => ".webm",
        "audio/wav" | "audio/x-wav" | "audio/wave" => ".wav",
        "audio/mpeg" => ".mp3",
        "audio/flac" => ".flac",
        "audio/mp4" => ".m4a",
        "application/json" => ".json",
        "text/plain" => ".txt",
        _ => return None,
    })
}

/// A plain file name: last path component, [A-Za-z0-9._-] only, at most 100 characters,
/// never empty or hidden; the extension comes from the content type when missing.
pub fn safe_name(name: Option<&str>, content_type: Option<&str>) -> String {
    let n = name.unwrap_or("").replace('\\', "/");
    let base = n.rsplit('/').next().unwrap_or("").trim();
    let mut out = String::new();
    let mut last_us = false;
    for c in base.chars() {
        if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
            out.push(c);
            last_us = false;
        } else if !last_us {
            out.push('_');
            last_us = true;
        }
    }
    let mut out: String = out.trim_start_matches('.').chars().take(100).collect();
    if out.is_empty() {
        out = "upload".into();
    }
    if !out.contains('.') {
        if let Some(e) = content_type.and_then(ext_for_type) {
            out.push_str(e);
        }
    }
    out
}

fn random_hex(n: usize) -> String {
    let mut buf = vec![0u8; n];
    if std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut buf)).is_err() {
        let t = now().to_bits() ^ ((std::process::id() as u64) << 20);
        for (i, b) in buf.iter_mut().enumerate() {
            *b = (t >> ((i % 8) * 8)) as u8 ^ (i as u8).wrapping_mul(31);
        }
    }
    buf.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn is_upload_id(s: &str) -> bool {
    s.len() == 16 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Store one upload; Err((status, message)). Returns the POST /v1/files response.
pub fn save_upload(
    uploads: &Path,
    name: Option<&str>,
    data: &[u8],
    content_type: Option<&str>,
    max_bytes: usize,
) -> Result<Value, (u16, String)> {
    if data.is_empty() {
        return Err((400, "the upload is empty".into()));
    }
    if data.len() > max_bytes {
        return Err((
            413,
            format!("the upload is {} bytes, more than the {max_bytes} byte limit (max_upload_mb)", data.len()),
        ));
    }
    let id = random_hex(8);
    let fname = safe_name(name, content_type);
    let dir = uploads.join(&id);
    std::fs::create_dir_all(&dir).map_err(|e| (500, format!("cannot store the upload: {e}")))?;
    let path = dir.join(&fname);
    let tmp = dir.join(format!("{fname}.part"));
    std::fs::write(&tmp, data).map_err(|e| (500, format!("cannot store the upload: {e}")))?;
    std::fs::rename(&tmp, &path).map_err(|e| (500, format!("cannot store the upload: {e}")))?;
    let path = path.canonicalize().unwrap_or(path);
    let ct = content_type.map(|c| c.split(';').next().unwrap_or("").trim().to_string()).unwrap_or_default();
    let ct = if ct.is_empty() || ct == "application/octet-stream" { content_type_for(&path).to_string() } else { ct };
    Ok(json!({
        "id": id, "path": path.to_string_lossy(), "name": fname, "size": data.len(),
        "content_type": ct, "sha256": sha256_hex(data), "created_at": now(),
    }))
}

/// An `inputs` value -> an absolute path: a bare upload id is looked up under `uploads`.
pub fn resolve_input(uploads: &Path, value: &str) -> PathBuf {
    if is_upload_id(value) {
        let d = uploads.join(value);
        if let Ok(rd) = std::fs::read_dir(&d) {
            let mut files: Vec<PathBuf> = rd
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.is_file() && !p.to_string_lossy().ends_with(".part"))
                .collect();
            files.sort();
            if let Some(f) = files.into_iter().next() {
                return f.canonicalize().unwrap_or(f);
            }
        }
    }
    crate::util::abspath(value)
}

// ---- multipart/form-data ----------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Part {
    pub name: String,
    pub filename: Option<String>,
    pub content_type: Option<String>,
    pub data: Vec<u8>,
}

impl Part {
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.data).into_owned()
    }
}

/// The boundary of a `multipart/form-data; boundary=...` content type.
pub fn boundary(content_type: &str) -> Option<String> {
    let (kind, rest) = content_type.split_once(';')?;
    if !kind.trim().eq_ignore_ascii_case("multipart/form-data") {
        return None;
    }
    for p in rest.split(';') {
        let (k, v) = p.split_once('=')?;
        if k.trim().eq_ignore_ascii_case("boundary") {
            let v = v.trim().trim_matches('"');
            return (!v.is_empty()).then(|| v.to_string());
        }
    }
    None
}

fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || from > hay.len() {
        return None;
    }
    hay[from..].windows(needle.len()).position(|w| w == needle).map(|i| i + from)
}

fn header_param(h: &str, key: &str) -> Option<String> {
    // name="x"; filename="y.png" (quoted or bare)
    for seg in h.split(';').skip(1) {
        let (k, v) = seg.split_once('=')?;
        if k.trim().eq_ignore_ascii_case(key) {
            let v = v.trim();
            let v = v.strip_prefix('"').and_then(|x| x.strip_suffix('"')).unwrap_or(v);
            return Some(v.replace("\\\"", "\""));
        }
    }
    None
}

/// Parse a multipart/form-data body. Err with a plain message when it is malformed.
pub fn parse_multipart(body: &[u8], boundary: &str) -> Result<Vec<Part>, String> {
    let delim = format!("--{boundary}").into_bytes();
    let mut parts = Vec::new();
    let mut pos = find(body, &delim, 0).ok_or("multipart body without its boundary")?;
    loop {
        pos += delim.len();
        if body[pos..].starts_with(b"--") {
            break; // closing delimiter
        }
        if body[pos..].starts_with(b"\r\n") {
            pos += 2;
        }
        let head_end = find(body, b"\r\n\r\n", pos).ok_or("multipart part without headers")?;
        let head = String::from_utf8_lossy(&body[pos..head_end]).into_owned();
        let data_start = head_end + 4;
        let mut next_delim = delim.clone();
        next_delim.splice(0..0, b"\r\n".iter().copied());
        let data_end = find(body, &next_delim, data_start).ok_or("multipart part is not terminated")?;
        let mut name = String::new();
        let mut filename = None;
        let mut ctype = None;
        for line in head.split("\r\n") {
            let Some((k, v)) = line.split_once(':') else { continue };
            if k.trim().eq_ignore_ascii_case("content-disposition") {
                name = header_param(v, "name").unwrap_or_default();
                filename = header_param(v, "filename");
            } else if k.trim().eq_ignore_ascii_case("content-type") {
                ctype = Some(v.trim().to_string());
            }
        }
        parts.push(Part { name, filename, content_type: ctype, data: body[data_start..data_end].to_vec() });
        pos = data_end + 2; // at the next delimiter
    }
    Ok(parts)
}

// ---- base64 -------------------------------------------------------------------------------

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn b64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        out.push(B64[(n >> 18) as usize & 63] as char);
        out.push(B64[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 { B64[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { B64[n as usize & 63] as char } else { '=' });
    }
    out
}

/// Standard or URL-safe alphabet, padding optional, whitespace ignored.
pub fn b64_decode(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut acc = 0u32;
    let mut bits = 0u32;
    for b in s.bytes() {
        let v = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => break,
            b' ' | b'\n' | b'\r' | b'\t' => continue,
            _ => return None,
        };
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
}

/// `data:<type>;base64,<data>` -> (bytes, type).
pub fn decode_data_url(s: &str) -> Option<(Vec<u8>, Option<String>)> {
    let rest = s.trim().strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    if !meta.split(';').any(|x| x.eq_ignore_ascii_case("base64")) {
        return None;
    }
    let ct = meta.split(';').next().filter(|t| !t.is_empty()).map(String::from);
    Some((b64_decode(data)?, ct))
}

// ---- sha256 -------------------------------------------------------------------------------

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
    0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
    0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
];

pub fn sha256_hex(data: &[u8]) -> String {
    let mut h: [u32; 8] =
        [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(y);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_values() {
        assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        let long = vec![b'a'; 1000];
        assert_eq!(sha256_hex(&long), "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
    }

    #[test]
    fn base64_roundtrip() {
        for s in [&b""[..], b"f", b"fo", b"foo", b"foob", b"fooba", b"foobar", &[0u8, 255, 128, 7][..]] {
            assert_eq!(b64_decode(&b64_encode(s)).unwrap(), s);
        }
        assert_eq!(b64_encode(b"foobar"), "Zm9vYmFy");
        assert_eq!(b64_decode("Zm9v\nYg").unwrap(), b"foob");
        assert!(b64_decode("a*b").is_none());
        let (d, t) = decode_data_url("data:image/png;base64,iVBO").unwrap();
        assert_eq!(t.as_deref(), Some("image/png"));
        assert_eq!(d, vec![0x89, 0x50, 0x4e]);
        assert!(decode_data_url("https://x/y.png").is_none());
    }

    #[test]
    fn names_and_ids() {
        assert_eq!(safe_name(Some("../../etc/pass wd"), None), "pass_wd");
        assert_eq!(safe_name(Some(".hidden"), Some("image/png")), "hidden.png");
        assert_eq!(safe_name(None, Some("audio/wav")), "upload.wav");
        assert_eq!(safe_name(Some("a.png"), Some("image/jpeg")), "a.png");
        assert!(is_upload_id("0123456789abcdef") && !is_upload_id("0123456789abcdeg") && !is_upload_id("abc"));
    }

    #[test]
    fn multipart() {
        assert_eq!(boundary("multipart/form-data; boundary=XyZ").as_deref(), Some("XyZ"));
        assert_eq!(boundary("multipart/form-data; boundary=\"q q\"").as_deref(), Some("q q"));
        assert!(boundary("application/json").is_none());
        let body = b"--XyZ\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\na cat\r\n--XyZ\r\n\
Content-Disposition: form-data; name=\"file\"; filename=\"x.png\"\r\nContent-Type: image/png\r\n\r\n\x89PNG\r\n\r\n--XyZ--\r\n";
        let parts = parse_multipart(body, "XyZ").unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].name, "prompt");
        assert_eq!(parts[0].text(), "a cat");
        assert_eq!(parts[1].filename.as_deref(), Some("x.png"));
        assert_eq!(parts[1].content_type.as_deref(), Some("image/png"));
        assert_eq!(parts[1].data, b"\x89PNG\r\n");
        assert!(parse_multipart(b"garbage", "XyZ").is_err());
    }

    #[test]
    fn upload_and_resolve() {
        let d = tempfile::tempdir().unwrap();
        let v = save_upload(d.path(), Some("a b.png"), b"abc", Some("image/png"), 10).unwrap();
        assert_eq!(v["name"], "a_b.png");
        assert_eq!(v["size"], 3);
        assert_eq!(v["sha256"], "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        let id = v["id"].as_str().unwrap();
        assert_eq!(resolve_input(d.path(), id).to_string_lossy(), v["path"].as_str().unwrap());
        assert_eq!(save_upload(d.path(), None, b"", None, 10).unwrap_err().0, 400);
        assert_eq!(save_upload(d.path(), None, b"0123456789a", None, 10).unwrap_err().0, 413);
    }
}
