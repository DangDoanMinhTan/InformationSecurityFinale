import argparse
import getpass
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from .attacks import AttackReport, evaluate_attacks
from .audit import AuditLog
from .authenticator import EnrollmentError, IrisAuthenticator
from .config import SystemConfig, real_eye_config
from .evaluation import evaluate_records, save_report
from .io_utils import load_dataset, read_visible, write_image
from .paths import app_root
from .pipeline import IrisPipeline
from .segmentation import SegmentationError
from .templatestore import TemplateStore, TemplateStoreError
from .visualization import compose_panel

ROOT = app_root()
DEFAULT_DATABASE = ROOT / "database_real"
DEFAULT_OUTPUT = ROOT / "output"


def resolve_passphrase() -> str:
    value = os.environ.get("IRIS_DB_PASSPHRASE")
    if value:
        return value
    if sys.stdin.isatty():
        return getpass.getpass("Template store passphrase: ")
    raise SystemExit("set the IRIS_DB_PASSPHRASE environment variable or run from an interactive terminal")


def build_config(args: argparse.Namespace) -> SystemConfig:
    config = real_eye_config()
    matching = config.matching
    if getattr(args, "threshold", None) is not None:
        matching = replace(matching, threshold=args.threshold)
    if getattr(args, "max_shift", None) is not None:
        matching = replace(matching, rotation_shifts=2 * args.max_shift + 1)
    pad = replace(
        config.pad,
        enabled=getattr(args, "pad", False),
        require_pupil_reaction=getattr(args, "require_liveness", False),
    )
    security = replace(config.security, expose_diagnostics=getattr(args, "debug", False))
    return replace(config, matching=matching, pad=pad, security=security)


def open_system(args: argparse.Namespace) -> IrisAuthenticator:
    config = build_config(args)
    directory = Path(args.db)
    store = TemplateStore(directory / "templates.enc", resolve_passphrase())
    return IrisAuthenticator(store, config, audit=AuditLog(directory / "audit.log"))


def add_matching_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", default=str(DEFAULT_DATABASE), help="database directory")
    parser.add_argument("--threshold", type=float, help="Hamming distance threshold (default 0.36 for real photos)")
    parser.add_argument("--max-shift", type=int, help="rotation compensation in code steps each way (default 5)")
    parser.add_argument("--pad", action="store_true", help="enable presentation attack detection (needs NIR-quality images)")
    parser.add_argument("--debug", action="store_true", help="show scores and diagnostics")


def add_probe_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image", required=True, help="probe eye image")
    parser.add_argument("--liveness", help="second frame captured under bright light (pupil reaction check)")
    parser.add_argument("--require-liveness", action="store_true", help="reject probes without a liveness frame")
    parser.add_argument("--save-panel", help="save an annotated PNG of segmentation, normalized iris and IrisCode")


def print_result(result, panel_path: Optional[str], title: str) -> None:
    print(result.message)
    print(f"outcome: {result.outcome.value}")
    if result.user_id:
        print(f"user: {result.user_id}")
    if result.diagnostics:
        for key, value in result.diagnostics.items():
            if key == "features":
                continue
            print(f"  {key}: {value}")
    if panel_path and result.diagnostics and "features" in result.diagnostics:
        lines = [title, f"{result.message}  HD {result.diagnostics.get('distance', float('nan')):.3f}"]
        write_image(panel_path, compose_panel(result.diagnostics["features"], lines))
        print(f"panel saved: {panel_path}")


def command_enroll(args: argparse.Namespace) -> int:
    system = open_system(args)
    images = [read_visible(path) for path in args.images]
    try:
        features = system.enroll(args.user, images, replace=args.replace)
    except EnrollmentError as error:
        print(f"enrollment failed: {error}")
        return 1
    print(f"enrolled {args.user} with {len(features)} template(s)")
    return 0


def command_verify(args: argparse.Namespace) -> int:
    system = open_system(args)
    liveness = read_visible(args.liveness) if args.liveness else None
    result = system.verify(args.user, read_visible(args.image), liveness)
    print_result(result, args.save_panel, f"Probe: {Path(args.image).name}  claimed: {args.user}")
    return 0 if result.accepted else 2


def command_identify(args: argparse.Namespace) -> int:
    system = open_system(args)
    liveness = read_visible(args.liveness) if args.liveness else None
    result = system.identify(read_visible(args.image), liveness)
    print_result(result, args.save_panel, f"Probe: {Path(args.image).name}  1:N identification")
    return 0 if result.accepted else 2


def command_encode(args: argparse.Namespace) -> int:
    config = build_config(args)
    pipeline = IrisPipeline(config)
    try:
        features = pipeline.extract(read_visible(args.image))
    except SegmentationError as error:
        print(f"segmentation failed: {error}")
        return 1
    code = features.code
    segmentation = features.segmentation
    print(f"pupil  : x={segmentation.pupil.x:.1f} y={segmentation.pupil.y:.1f} r={segmentation.pupil.r:.1f}")
    print(f"iris   : x={segmentation.iris.x:.1f} y={segmentation.iris.y:.1f} r={segmentation.iris.r:.1f}")
    print(f"quality: sharpness={features.quality.sharpness:.2f} usable={features.quality.usable_fraction:.2f} passed={features.quality.passed}")
    print(f"IrisCode: {code.size} bits, {int(code.mask.sum())} usable")
    print(f"hex     : {code.to_hex()}")
    output = Path(args.out) if args.out else DEFAULT_OUTPUT / f"{Path(args.image).stem}_iriscode.png"
    write_image(output, compose_panel(features, [f"Probe: {Path(args.image).name}", f"IrisCode {code.size} bits"]))
    print(f"panel saved: {output}")
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    config = build_config(args)
    roots = [Path(item) for item in args.dataset]
    records = load_dataset(roots)
    print(f"{len(records)} images, {len({label for label, _ in records})} classes")

    def progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"  processed {done}/{total}", flush=True)

    result = evaluate_records(records, config, progress, reader=read_visible)
    paths = save_report(result, config, Path(args.out))
    print(f"acquired {result.acquired}/{result.images}, genuine pairs {result.genuine.size}, impostor pairs {result.impostor.size}")
    print(f"genuine HD  mean {result.genuine.mean():.3f} std {result.genuine.std():.3f}")
    print(f"impostor HD mean {result.impostor.mean():.3f} std {result.impostor.std():.3f}")
    print(f"EER {result.eer * 100:.2f}% at threshold {result.eer_threshold:.3f}")
    for point in result.operating_points:
        print(f"  threshold {point['threshold']:.2f}: FAR {point['far']:.6f}  FRR {point['frr']:.4f}  model FAR {point['model_far']:.3e}")
    print(f"saved {paths['figure']} and {paths['json']}")
    return 0


def command_vault(args: argparse.Namespace) -> int:
    system = open_system(args)
    liveness = read_visible(args.liveness) if args.liveness else None
    result = system.unlock_vault(args.user, read_visible(args.image), liveness)
    print_result(result, None, "")
    if result.secret:
        print(f"released secret: {result.secret}")
    return 0 if result.accepted else 2


def command_revoke(args: argparse.Namespace) -> int:
    system = open_system(args)
    system.store.revoke(args.user)
    print(f"revoked {args.user}: the stored template and its transform are gone, re-enroll to issue a new one")
    return 0


def command_attacks(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.merge:
        parts = [AttackReport.from_dict(json.loads(Path(item).read_text(encoding="utf-8"))) for item in args.merge]
        report = AttackReport.merge(parts)
        target = out / "attack_report.json"
        target.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print("\n".join(report.summary()))
        print(f"saved {target}")
        return 0
    config = build_config(args)
    records = load_dataset([Path(item) for item in args.dataset])
    scratch = Path(args.scratch) if args.scratch else out / "attack_store.enc"
    if scratch.exists():
        scratch.unlink()

    def progress(done: int, total: int) -> None:
        print(f"  eyes processed {done}/{total}", flush=True)

    report = evaluate_attacks(
        records,
        config,
        scratch,
        resolve_passphrase(),
        repeats=args.repeats,
        seed=args.seed,
        reader=read_visible,
        progress=progress,
        offset=args.offset,
        limit=args.limit,
    )
    target = out / (args.name or "attack_report.json")
    target.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print("\n".join(report.summary()))
    print(f"saved {target}")
    return 0


def command_serve(args: argparse.Namespace) -> int:
    from .server import serve

    api_key = os.environ.get("IRIS_API_KEY")
    admin_key = os.environ.get("IRIS_ADMIN_KEY")
    if not api_key or not admin_key:
        raise SystemExit("set the IRIS_API_KEY and IRIS_ADMIN_KEY environment variables")
    signing = os.environ.get("IRIS_SIGNING_KEY")
    serve(
        Path(args.db),
        resolve_passphrase(),
        api_key,
        admin_key,
        signing_key=signing.encode("utf-8") if signing else None,
        host=args.host,
        port=args.port,
        tls=not args.no_tls,
        cert_dir=Path(args.cert_dir) if args.cert_dir else None,
    )
    return 0


def command_netdemo(args: argparse.Namespace) -> int:
    from .netdemo import run_demo

    results = run_demo(
        [Path(item) for item in args.enroll],
        Path(args.probe),
        Path(args.out),
        interface=args.interface,
    )
    print("\n".join(results["summary"]))
    return 0


def command_webcam_sim(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    from .degradation import CHANNELS, PROFILES, evaluate_webcam

    config = build_config(args)
    records = load_dataset([Path(item) for item in args.dataset])
    images = []
    for label, path in records:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is not None:
            images.append((label, image))
    names = args.profile or list(PROFILES)
    channels = args.channel or list(CHANNELS)
    lines = []
    results = []
    for channel in channels:
        for name in names:
            result = evaluate_webcam(images, config, PROFILES[name], channel, seed=args.seed)
            results.append(result.__dict__)
            lines.append(result.summary())
            print(result.summary(), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / (args.name or "webcam_simulation.json")
    target.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {target}")
    return 0


def command_vault_eval(args: argparse.Namespace) -> int:
    from .vault_eval import evaluate_vault, extract_codes

    config = build_config(args)
    codes = extract_codes(load_dataset([Path(item) for item in args.dataset]), config, read_visible)
    results = []
    for symbols in args.key_symbols:
        result = evaluate_vault(codes, config, symbols, impostors_per_eye=args.impostors)
        results.append({**result.__dict__, "frr": result.frr, "far": result.far})
        print(result.summary(), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / (args.name or "vault_evaluation.json")
    target.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {target}")
    return 0


def command_list(args: argparse.Namespace) -> int:
    system = open_system(args)
    users = system.store.users()
    print(f"{len(users)} enrolled user(s), {system.store.template_count()} template(s)")
    for user in users:
        print(f"  {user}")
    return 0


def command_remove(args: argparse.Namespace) -> int:
    system = open_system(args)
    system.store.remove_user(args.user)
    print(f"removed {args.user}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description="Iris recognition: Daugman IrisCode pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    enroll = commands.add_parser("enroll", help="enroll a user from one or more eye images")
    add_matching_options(enroll)
    enroll.add_argument("--user", required=True)
    enroll.add_argument("--images", nargs="+", required=True)
    enroll.add_argument("--replace", action="store_true")
    enroll.set_defaults(handler=command_enroll)

    verify = commands.add_parser("verify", help="1:1 verification of a claimed identity")
    add_matching_options(verify)
    add_probe_options(verify)
    verify.add_argument("--user", required=True)
    verify.set_defaults(handler=command_verify)

    identify = commands.add_parser("identify", help="1:N identification against the database")
    add_matching_options(identify)
    add_probe_options(identify)
    identify.set_defaults(handler=command_identify)

    encode = commands.add_parser("encode", help="extract and display the IrisCode of an image")
    encode.add_argument("--image", required=True)
    encode.add_argument("--out", help="annotated PNG output path")
    encode.set_defaults(handler=command_encode)

    evaluate = commands.add_parser("evaluate", help="compute FAR, FRR and EER on a labelled dataset")
    evaluate.add_argument("--dataset", nargs="+", required=True, help="dataset directories")
    evaluate.add_argument("--out", default=str(DEFAULT_OUTPUT))
    evaluate.add_argument("--threshold", type=float)
    evaluate.add_argument("--max-shift", type=int)
    evaluate.set_defaults(handler=command_evaluate)

    vault = commands.add_parser("vault-unlock", help="release the key-bound secret with a live iris (fuzzy commitment)")
    add_matching_options(vault)
    vault.add_argument("--user", required=True)
    vault.add_argument("--image", required=True)
    vault.add_argument("--liveness")
    vault.set_defaults(handler=command_vault)

    revoke = commands.add_parser("revoke", help="revoke a user's cancelable template")
    revoke.add_argument("--db", default=str(DEFAULT_DATABASE))
    revoke.add_argument("--user", required=True)
    revoke.set_defaults(handler=command_revoke)

    attacks = commands.add_parser("attacks", help="simulated presentation-attack evaluation (print, halftone, screen replay)")
    attacks.add_argument("--dataset", nargs="*", default=[], help="dataset directories")
    attacks.add_argument("--out", default=str(DEFAULT_OUTPUT))
    attacks.add_argument("--name", help="report file name")
    attacks.add_argument("--scratch", help="temporary template store path")
    attacks.add_argument("--repeats", type=int, default=3)
    attacks.add_argument("--seed", type=int, default=7)
    attacks.add_argument("--offset", type=int, default=0)
    attacks.add_argument("--limit", type=int)
    attacks.add_argument("--merge", nargs="*", help="merge saved partial reports instead of running")
    attacks.add_argument("--threshold", type=float)
    attacks.add_argument("--max-shift", type=int)
    attacks.set_defaults(handler=command_attacks)

    server = commands.add_parser("serve", help="run the HTTPS authentication server (HTTP only with --no-tls)")
    server.add_argument("--db", default=str(DEFAULT_DATABASE))
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8443)
    server.add_argument("--no-tls", action="store_true", help="serve plain HTTP (for the packet-capture comparison only)")
    server.add_argument("--cert-dir", help="directory holding server.crt and server.key (created if missing)")
    server.set_defaults(handler=command_serve)

    netdemo = commands.add_parser("netdemo", help="capture plain HTTP versus TLS traffic with tcpdump and analyse it (needs root)")
    netdemo.add_argument("--enroll", nargs="+", required=True, help="enrollment images")
    netdemo.add_argument("--probe", required=True, help="probe image")
    netdemo.add_argument("--out", default=str(DEFAULT_OUTPUT / "network"))
    netdemo.add_argument("--interface", default="lo")
    netdemo.set_defaults(handler=command_netdemo)

    webcam = commands.add_parser("webcam-sim", help="measure accuracy on degraded webcam-like captures (visible light and a near-infrared proxy)")
    webcam.add_argument("--dataset", nargs="+", required=True)
    webcam.add_argument("--profile", nargs="*", choices=["good", "typical", "poor"])
    webcam.add_argument("--channel", nargs="*", choices=["visible", "nir_proxy"])
    webcam.add_argument("--out", default=str(DEFAULT_OUTPUT))
    webcam.add_argument("--name")
    webcam.add_argument("--seed", type=int, default=11)
    webcam.add_argument("--threshold", type=float)
    webcam.add_argument("--max-shift", type=int)
    webcam.set_defaults(handler=command_webcam_sim)

    vaulteval = commands.add_parser("vault-eval", help="measure key-release rate and false releases of the fuzzy commitment")
    vaulteval.add_argument("--dataset", nargs="+", required=True)
    vaulteval.add_argument("--key-symbols", nargs="+", type=int, default=[20])
    vaulteval.add_argument("--impostors", type=int, help="impostor probes per eye (default: all)")
    vaulteval.add_argument("--out", default=str(DEFAULT_OUTPUT))
    vaulteval.add_argument("--name")
    vaulteval.add_argument("--threshold", type=float)
    vaulteval.add_argument("--max-shift", type=int)
    vaulteval.set_defaults(handler=command_vault_eval)

    listing = commands.add_parser("list-users", help="list enrolled users")
    listing.add_argument("--db", default=str(DEFAULT_DATABASE))
    listing.set_defaults(handler=command_list)

    remove = commands.add_parser("remove-user", help="remove an enrolled user")
    remove.add_argument("--db", default=str(DEFAULT_DATABASE))
    remove.add_argument("--user", required=True)
    remove.set_defaults(handler=command_remove)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (TemplateStoreError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}")
        return 1
