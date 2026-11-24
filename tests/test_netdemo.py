import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iris_auth.netdemo import analyse_capture, entropy, parse_server_hello, read_pcap, tcp_streams, tls_records

SERVER_PORT = 8443


def frame(source: int, destination: int, payload: bytes, flags: int = 0x18) -> bytes:
    ethernet = b"\x00" * 12 + b"\x08\x00"
    total = 20 + 20 + len(payload)
    ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, total, 1, 0, 64, 6, 0, bytes([127, 0, 0, 1]), bytes([127, 0, 0, 1]))
    tcp = struct.pack("!HHIIBBHHH", source, destination, 0, 0, 5 << 4, flags, 65535, 0, 0)
    return ethernet + ip + tcp + payload


def pcap(frames) -> bytes:
    header = struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
    body = b"".join(struct.pack("<IIII", 0, 0, len(item), len(item)) + item for item in frames)
    return header + body


def record(kind: int, payload: bytes) -> bytes:
    return struct.pack("!BHH", kind, 0x0303, len(payload)) + payload


class PcapTests(unittest.TestCase):
    def write(self, frames) -> Path:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        path = Path(self.directory.name) / "x.pcap"
        path.write_bytes(pcap(frames))
        return path

    def test_plain_http_is_readable(self):
        request = b"POST /api/verify HTTP/1.1\r\nX-Api-Key: secret-key\r\n\r\n{\"user\": \"alice\"}"
        path = self.write([frame(50000, SERVER_PORT, request), frame(SERVER_PORT, 50000, b"HTTP/1.1 200 OK\r\n\r\n{}")])
        report = analyse_capture(path, SERVER_PORT, {"key": b"secret-key", "user": b"alice", "absent": b"zzz"})
        self.assertTrue(report["readable_http"])
        self.assertEqual(report["marker_hits"], {"key": 1, "user": 1, "absent": 0})
        self.assertEqual(report["client_to_server_bytes"], len(request))
        self.assertNotIn("tls", report)

    def test_tls_stream_hides_content_and_is_summarised(self):
        hello = bytes([2, 0, 0, 0]) + b"\x03\x03" + b"\x00" * 32 + b"\x00" + b"\x13\x02" + b"\x00" + struct.pack("!H", 6) + struct.pack("!HHH", 43, 2, 0x0304)
        client = record(22, b"\x01" + b"\x00" * 40) + record(23, bytes(range(256)) * 4)
        server = record(22, hello) + record(23, bytes(range(255, -1, -1)) * 2)
        path = self.write([frame(50000, SERVER_PORT, client), frame(SERVER_PORT, 50000, server)])
        report = analyse_capture(path, SERVER_PORT, {"key": b"secret-key"})
        self.assertFalse(report["readable_http"])
        self.assertEqual(report["marker_hits"]["key"], 0)
        self.assertEqual(report["tls"]["version"], "TLS 1.3")
        self.assertEqual(report["tls"]["cipher"], "TLS_AES_256_GCM_SHA384")
        self.assertEqual(report["tls"]["records"]["application_data"], 2)

    def test_entropy_bounds(self):
        self.assertEqual(entropy(b""), 0.0)
        self.assertEqual(entropy(b"aaaa"), 0.0)
        self.assertAlmostEqual(entropy(bytes(range(256))), 8.0)

    def test_pcap_parsing_and_unsupported_formats(self):
        path = self.write([frame(1, SERVER_PORT, b"x", flags=0x02)])
        link, frames = read_pcap(path)
        toward, away, stats = tcp_streams(link, frames, SERVER_PORT)
        self.assertEqual((link, len(frames), toward, away, stats["syn"]), (1, 1, b"x", b"", 1))
        path.write_bytes(b"\x0a\x0d\x0d\x0a" + b"\x00" * 40)
        with self.assertRaises(ValueError):
            read_pcap(path)
        with self.assertRaises(ValueError):
            tcp_streams(147, frames, SERVER_PORT)

    def test_tls_record_parser_stops_on_garbage(self):
        self.assertEqual(tls_records(b"GET / HTTP/1.1"), [])
        self.assertEqual(parse_server_hello(b"\x01\x00"), {})


if __name__ == "__main__":
    unittest.main()
