import json
import math
import signal
import struct
import subprocess
import tempfile
import threading
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from werkzeug.serving import make_server

from .client import IrisClient
from .config import SystemConfig, real_eye_config
from .io_utils import read_visible
from .server import create_app
from .tls import ensure_certificate, server_context

LINK_HEADERS = {1: 14, 113: 16, 276: 20, 0: 4}
TLS_RECORD_TYPES = {20: "change_cipher_spec", 21: "alert", 22: "handshake", 23: "application_data"}
TLS_VERSIONS = {0x0301: "TLS 1.0", 0x0302: "TLS 1.1", 0x0303: "TLS 1.2", 0x0304: "TLS 1.3"}
CIPHER_NAMES = {
    0x1301: "TLS_AES_128_GCM_SHA256",
    0x1302: "TLS_AES_256_GCM_SHA384",
    0x1303: "TLS_CHACHA20_POLY1305_SHA256",
    0xC02F: "ECDHE-RSA-AES128-GCM-SHA256",
    0xC02B: "ECDHE-ECDSA-AES128-GCM-SHA256",
    0xC030: "ECDHE-RSA-AES256-GCM-SHA384",
    0xC02C: "ECDHE-ECDSA-AES256-GCM-SHA384",
}


def start_capture(interface: str, port: int, target: Path) -> subprocess.Popen:
    process = subprocess.Popen(
        ["tcpdump", "-i", interface, "--immediate-mode", "-U", "-s", "0", "-w", str(target), f"tcp port {port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if target.exists():
            break
        time.sleep(0.05)
    time.sleep(0.5)
    return process


def stop_capture(process: subprocess.Popen) -> None:
    time.sleep(1.0)
    process.send_signal(signal.SIGINT)
    try:
        process.wait(10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def read_pcap(path: Path) -> Tuple[int, List[bytes]]:
    data = Path(path).read_bytes()
    magic = data[:4]
    if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
        order = "<"
    elif magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
        order = ">"
    else:
        raise ValueError("not a classic pcap file")
    link = struct.unpack(order + "I", data[20:24])[0]
    frames: List[bytes] = []
    offset = 24
    while offset + 16 <= len(data):
        included = struct.unpack(order + "I", data[offset + 8: offset + 12])[0]
        offset += 16
        frames.append(data[offset: offset + included])
        offset += included
    return link, frames


def tcp_streams(link: int, frames: Sequence[bytes], server_port: int) -> Tuple[bytes, bytes, Dict[str, int]]:
    skip = LINK_HEADERS.get(link)
    if skip is None:
        raise ValueError(f"unsupported link type {link}")
    toward = bytearray()
    away = bytearray()
    stats = {"packets": 0, "payload_packets": 0, "syn": 0}
    for frame in frames:
        packet = frame[skip:]
        if len(packet) < 20 or packet[0] >> 4 != 4 or packet[9] != 6:
            continue
        header = (packet[0] & 0x0F) * 4
        segment = packet[header:]
        if len(segment) < 20:
            continue
        source, destination = struct.unpack("!HH", segment[:4])
        flags = segment[13]
        payload = segment[(segment[12] >> 4) * 4:]
        stats["packets"] += 1
        if flags & 0x02:
            stats["syn"] += 1
        if not payload:
            continue
        stats["payload_packets"] += 1
        if destination == server_port:
            toward.extend(payload)
        elif source == server_port:
            away.extend(payload)
    return bytes(toward), bytes(away), stats


def entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    return -sum(count / total * math.log2(count / total) for count in counts.values())


def tls_records(stream: bytes) -> List[Tuple[int, int, bytes]]:
    records = []
    offset = 0
    while offset + 5 <= len(stream):
        kind, version, length = struct.unpack("!BHH", stream[offset: offset + 5])
        if kind not in TLS_RECORD_TYPES or (version >> 8) != 3 or offset + 5 + length > len(stream):
            break
        records.append((kind, version, stream[offset + 5: offset + 5 + length]))
        offset += 5 + length
    return records


def parse_server_hello(body: bytes) -> Dict[str, str]:
    if len(body) < 42 or body[0] != 2:
        return {}
    cursor = 4 + 2 + 32
    session = body[cursor]
    cursor += 1 + session
    cipher = struct.unpack("!H", body[cursor: cursor + 2])[0]
    cursor += 3
    selected = struct.unpack("!H", body[4:6])[0]
    if cursor + 2 <= len(body):
        total = struct.unpack("!H", body[cursor: cursor + 2])[0]
        cursor += 2
        end = min(cursor + total, len(body))
        while cursor + 4 <= end:
            kind, size = struct.unpack("!HH", body[cursor: cursor + 4])
            cursor += 4
            if kind == 43 and size == 2:
                selected = struct.unpack("!H", body[cursor: cursor + 2])[0]
            cursor += size
    return {"version": TLS_VERSIONS.get(selected, hex(selected)), "cipher": CIPHER_NAMES.get(cipher, hex(cipher))}


def analyse_capture(path: Path, port: int, markers: Dict[str, bytes]) -> Dict[str, Any]:
    link, frames = read_pcap(path)
    toward, away, stats = tcp_streams(link, frames, port)
    combined = toward + away
    found = {name: combined.count(value) for name, value in markers.items()}
    report: Dict[str, Any] = {
        "file": Path(path).name,
        "packets": stats["packets"],
        "payload_packets": stats["payload_packets"],
        "client_to_server_bytes": len(toward),
        "server_to_client_bytes": len(away),
        "entropy_client_to_server": round(entropy(toward), 3),
        "entropy_server_to_client": round(entropy(away), 3),
        "marker_hits": found,
        "readable_http": combined.count(b"HTTP/1.1") > 0,
    }
    records = tls_records(toward)
    if records and records[0][0] == 22:
        server_records = tls_records(away)
        hello = {}
        for kind, _, body in server_records:
            if kind == 22 and body[:1] == b"\x02":
                hello = parse_server_hello(body)
                break
        counts = Counter(TLS_RECORD_TYPES[kind] for kind, _, _ in records + server_records)
        report["tls"] = {"records": dict(counts), **hello}
    return report


def format_report(plain: Dict[str, Any], secure: Dict[str, Any]) -> List[str]:
    lines = []
    for title, report in (("Plain HTTP", plain), ("HTTPS (TLS)", secure)):
        lines.append(f"{title}: {report['file']}")
        lines.append(
            f"  packets {report['packets']}, with payload {report['payload_packets']}, "
            f"client->server {report['client_to_server_bytes']} B, server->client {report['server_to_client_bytes']} B"
        )
        lines.append(
            f"  byte entropy client->server {report['entropy_client_to_server']} bits/byte, server->client {report['entropy_server_to_client']} bits/byte"
        )
        lines.append(f"  readable HTTP in capture: {'yes' if report['readable_http'] else 'no'}")
        hits = ", ".join(f"{name} x{count}" for name, count in report["marker_hits"].items())
        lines.append(f"  sensitive strings found in the capture: {hits}")
        if "tls" in report:
            tls = report["tls"]
            lines.append(f"  negotiated {tls.get('version', '?')} with {tls.get('cipher', '?')}, records {tls['records']}")
    return lines


def run_flow(base_url: str, ca_file: Optional[Path], api_key: str, admin_key: str, enroll: Sequence, probe, user: str, signing_key: Optional[bytes] = None) -> Dict[str, Any]:
    client = IrisClient(base_url, api_key, ca_file, admin_key, signing_key=signing_key)
    outcome: Dict[str, Any] = {}
    outcome["enroll"] = client.enroll(user, enroll, replace=True)[0]
    status, body = client.verify(user, probe)
    outcome["verify"] = {"status": status, "accepted": body.get("accepted")}
    token = body.get("token")
    status, body = client.vault(user, probe)
    outcome["vault"] = {"status": status, "accepted": body.get("accepted"), "secret": body.get("secret")}
    outcome["token"] = token
    outcome["session"] = client.session(token)[0] if token else None
    return outcome


def serve_in_thread(app, port: int, context=None) -> Tuple[Any, threading.Thread]:
    server = make_server("127.0.0.1", port, app, threaded=True, ssl_context=context)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def run_demo(
    enroll_images: Sequence[Path],
    probe_image: Path,
    out_dir: Path,
    interface: str = "lo",
    plain_port: int = 18080,
    tls_port: int = 18443,
    user: str = "demo-user",
    passphrase: str = "network-demo-passphrase",
    config: Optional[SystemConfig] = None,
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    api_key = "device-key-" + "demo0123456789"
    admin_key = "admin-key-" + "demo0123456789"
    signing_key = b"capture-signing-" + b"demo0123456789"
    settings = config or real_eye_config()
    settings = replace(settings, pad=replace(settings.pad, enabled=False))
    enrol = [read_visible(path) for path in enroll_images]
    probe = read_visible(probe_image)
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        app = create_app(root / "store", passphrase, api_key, admin_key, settings, signing_key=signing_key)
        cert, key = ensure_certificate(root / "tls")
        plain_server, plain_thread = serve_in_thread(app, plain_port)
        tls_server, tls_thread = serve_in_thread(app, tls_port, server_context(cert, key))
        try:
            plain_pcap = out / "capture_plain_http.pcap"
            tls_pcap = out / "capture_tls.pcap"
            capture = start_capture(interface, plain_port, plain_pcap)
            plain_flow = run_flow(f"http://127.0.0.1:{plain_port}", None, api_key, admin_key, enrol, probe, user, signing_key)
            stop_capture(capture)
            capture = start_capture(interface, tls_port, tls_pcap)
            tls_flow = run_flow(f"https://localhost:{tls_port}", cert, api_key, admin_key, enrol, probe, user, signing_key)
            stop_capture(capture)
        finally:
            plain_server.shutdown()
            tls_server.shutdown()
            plain_thread.join(5)
            tls_thread.join(5)
        markers = {
            "user id": user.encode(),
            "API key": api_key.encode(),
            "admin key": admin_key.encode(),
            "request line": b"POST /api/verify",
            "JPEG image (base64)": b"/9j/",
            "released secret": b"protected record of",
            "capture signing key": signing_key,
        }
        if plain_flow.get("token"):
            markers["session token"] = plain_flow["token"].encode()
        plain_report = analyse_capture(plain_pcap, plain_port, markers)
        tls_markers = {name: value for name, value in markers.items() if name != "session token"}
        if tls_flow.get("token"):
            tls_markers["session token"] = tls_flow["token"].encode()
        tls_report = analyse_capture(tls_pcap, tls_port, tls_markers)
    results = {"plain": plain_report, "tls": tls_report, "flows": {"plain": plain_flow, "tls": tls_flow}}
    for item in results["flows"].values():
        item.pop("token", None)
    summary = format_report(plain_report, tls_report)
    results["summary"] = summary
    (out / "network_capture_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out / "network_capture_report.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return results
