import json
import os
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .attacks import simulate_attack
from .audit import AuditLog
from .authenticator import IDENTIFY_KEY, UNKNOWN_KEY, AuthResult, EnrollmentError, IrisAuthenticator, Outcome
from .config import SystemConfig, real_eye_config
from .evaluation import evaluate_records, plot_evaluation, save_report
from .io_utils import load_dataset, read_visible, write_image
from .matching import hamming_distance
from .pad import PresentationAttackDetector
from .pipeline import IrisFeatures, IrisPipeline
from .protection import FuzzyVault
from .segmentation import SegmentationError
from .templatestore import TemplateStore
from .visualization import compose_panel, draw_segmentation

PASSPHRASE = os.environ.get("IRIS_DB_PASSPHRASE") or "tdtu-demo-passphrase"

NAMES = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Isabel", "Jack", "Karen", "Liam",
    "Maria", "Nathan", "Olivia", "Peter", "Quinn", "Rachel", "Samuel", "Tina", "Victor", "Wendy", "Xavier", "Yara",
]
EYE_LABEL = re.compile(r"^(\d+)([LR])$")
STRANGERS = 4


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str


SCENARIOS = [
    Scenario("genuine", "1. Genuine user, correct eye"),
    Scenario("impostor", "2. Impostor: someone else's eye"),
    Scenario("stranger", "3. Stranger (never enrolled) claims an identity"),
    Scenario("other_eye", "4. Same person, wrong eye (left vs right)"),
    Scenario("identify_known", "5. Identify (1:N): enrolled person"),
    Scenario("identify_stranger", "6. Identify (1:N): stranger"),
    Scenario("blur", "7. Out-of-focus capture is denied"),
    Scenario("lockout", "8. Brute force: 5 wrong tries lock the account"),
]


SECURITY_SCENARIOS = [
    Scenario("vault", "9. Key release: the owner's iris unlocks the protected secret"),
    Scenario("vault_impostor", "10. Key release: another person's iris gets nothing"),
    Scenario("print_attack", "11. Printed copy of the eye, attack detection ON"),
    Scenario("print_attack_open", "12. Same printed copy, attack detection OFF"),
    Scenario("revocation", "13. Revocation: a stolen template becomes useless"),
]

ALL_SCENARIOS = SCENARIOS + SECURITY_SCENARIOS


def display_name(label: str) -> str:
    match = EYE_LABEL.match(label)
    if not match:
        return label
    index = int(match.group(1)) - 1
    base = NAMES[index % len(NAMES)]
    suffix = "" if index < len(NAMES) else str(index // len(NAMES) + 1)
    return f"{base}{suffix}_{match.group(2)}"


def person_key(label: str) -> str:
    match = EYE_LABEL.match(label)
    return match.group(1) if match else label


def other_eye_label(label: str) -> Optional[str]:
    match = EYE_LABEL.match(label)
    if not match:
        return None
    return match.group(1) + ("R" if match.group(2) == "L" else "L")


TONES = {
    Outcome.ACCEPTED: "good",
    Outcome.NO_MATCH: "bad",
    Outcome.UNKNOWN_USER: "bad",
    Outcome.ATTACK_DETECTED: "warn",
    Outcome.QUALITY_REJECTED: "info",
    Outcome.SEGMENTATION_FAILED: "info",
    Outcome.LOCKED_OUT: "locked",
}


@dataclass
class RunView:
    banner: str
    tone: str
    image: np.ndarray
    details: List[str]
    result: Optional[AuthResult] = None


def to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return cleaned or "capture"


class AppController:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.database_dir = self.root / "database_real"
        self.output_dir = self.root / "output"
        self.capture_dir = self.root / "captures"
        self.base: SystemConfig = real_eye_config()
        self.store: Optional[TemplateStore] = None
        self.plan_path = self.database_dir / "scenarios.json"

    def prepare(self) -> None:
        self.open_store()

    def open_store(self) -> None:
        self.store = TemplateStore(self.database_dir / "templates.enc", PASSPHRASE)

    def default_settings(self) -> Tuple[float, bool]:
        return self.base.matching.threshold, self.base.pad.enabled

    def reset_database(self) -> None:
        target = self.database_dir / "templates.enc"
        if target.exists():
            target.unlink()
        if self.plan_path.exists():
            self.plan_path.unlink()
        self.open_store()

    def clear_lockouts(self) -> None:
        self.store.clear_all_failures()

    def load_plan(self) -> Optional[Dict]:
        if not self.plan_path.exists():
            return None
        try:
            plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        for entry in plan.values():
            probe = entry.get("probe") if isinstance(entry, dict) else None
            if probe and not Path(probe).exists():
                return None
        enrolled = set(self.users())
        for entry in plan.values():
            if isinstance(entry, dict) and entry.get("claimed") and entry["claimed"] not in enrolled:
                return None
        return plan

    def scenarios_ready(self) -> bool:
        plan = self.load_plan()
        return bool(plan) and "genuine" in plan

    def users(self) -> List[str]:
        return self.store.users() if self.store else []

    def read(self, path: Path) -> np.ndarray:
        return read_visible(path)

    def make_system(self, pad_enabled: bool, require_reaction: bool, threshold: float) -> IrisAuthenticator:
        config = replace(
            self.base,
            matching=replace(self.base.matching, threshold=threshold),
            pad=replace(self.base.pad, enabled=pad_enabled, require_pupil_reaction=require_reaction),
            security=replace(self.base.security, expose_diagnostics=True),
        )
        return IrisAuthenticator(self.store, config, audit=AuditLog(self.database_dir / "audit.log"))

    def features_for(self, image: np.ndarray, system: IrisAuthenticator, result: AuthResult) -> Optional[IrisFeatures]:
        if result.diagnostics and "features" in result.diagnostics:
            return result.diagnostics["features"]
        try:
            return system.pipeline.extract(image)
        except SegmentationError:
            return None

    def describe(self, result: AuthResult, claimed: Optional[str]) -> List[str]:
        details = result.diagnostics or {}
        lines = [f"Outcome            : {result.outcome.value}"]
        if claimed:
            lines.append(f"Claimed identity   : {claimed}")
        if "best_candidate" in details:
            lines.append(f"Closest enrolled   : {details['best_candidate']}")
        if "distance" in details:
            lines.append(
                f"Hamming distance   : {details['distance']:.3f}  (accept if <= {details['threshold']:.3f})"
            )
        if "shift" in details:
            lines.append(f"Best rotation shift: {details['shift']}   compared bits: {details.get('overlap', 0)}")
        quality = details.get("quality")
        if quality is not None:
            lines.append(
                f"Image quality      : sharpness {quality.sharpness:.1f}, usable iris {quality.usable_fraction * 100:.0f}%, "
                f"pupil/iris {quality.pupil_iris_ratio:.2f}"
            )
            if quality.failures:
                lines.append(f"Quality problems   : {'; '.join(quality.failures)}")
        pad = details.get("pad")
        if pad is not None:
            reaction = "n/a" if pad.pupil_constriction is None else f"{pad.pupil_constriction * 100:.0f}% constriction"
            lines.append(f"Attack detection   : print pattern score {pad.spectral_peak:.2f} (limit {self.base.pad.max_spectral_peak}), pupil reaction {reaction}")
        elif result.outcome != Outcome.LOCKED_OUT and "quality" in details:
            lines.append("Attack detection   : OFF")
        if details.get("pad_reasons"):
            lines.append(f"Attack reasons     : {'; '.join(details['pad_reasons'])}")
        if "pairing_distance" in details:
            lines.append(f"Liveness frame HD  : {details['pairing_distance']:.3f}")
        if "error" in details:
            lines.append(f"Error              : {details['error']}")
        return lines

    def banner_for(self, result: AuthResult) -> str:
        outcome = result.outcome
        if outcome == Outcome.ACCEPTED:
            return f"ACCESS GRANTED  -  {result.user_id}"
        if outcome == Outcome.ATTACK_DETECTED:
            return "PRESENTATION ATTACK DETECTED"
        if outcome == Outcome.LOCKED_OUT:
            return "TOO MANY FAILED ATTEMPTS - LOCKED"
        if outcome in (Outcome.QUALITY_REJECTED, Outcome.SEGMENTATION_FAILED):
            return "IMAGE REJECTED - CAPTURE AGAIN"
        return "ACCESS DENIED"

    def build_view(
        self, result: AuthResult, image: np.ndarray, features: Optional[IrisFeatures], title: str, claimed: Optional[str]
    ) -> RunView:
        distance = (result.diagnostics or {}).get("distance")
        second = result.message if distance is None else f"{result.message}   HD {distance:.3f}"
        panel = compose_panel(features, [title, second]) if features is not None else to_bgr(image)
        return RunView(
            banner=self.banner_for(result),
            tone=TONES[result.outcome],
            image=panel,
            details=self.describe(result, claimed),
            result=result,
        )

    def verify_image(
        self,
        claimed: str,
        image: np.ndarray,
        liveness: Optional[np.ndarray],
        pad_enabled: bool,
        require_reaction: bool,
        threshold: float,
        title: str,
    ) -> RunView:
        system = self.make_system(pad_enabled, require_reaction, threshold)
        result = system.verify(claimed, image, liveness)
        features = self.features_for(image, system, result)
        return self.build_view(result, image, features, title, claimed)

    def vault_image(
        self, claimed: str, image: np.ndarray, liveness: Optional[np.ndarray], threshold: float, title: str
    ) -> RunView:
        system = self.make_system(False, False, threshold)
        result = system.unlock_vault(claimed, image, liveness)
        view = self.build_view(result, image, self.features_for(image, system, result), title, claimed)
        if result.secret:
            view.banner = f"SECRET RELEASED  -  {claimed}"
            view.details.append(f"Released secret    : {result.secret}")
            view.details.append("The key was rebuilt from the iris itself; it is not stored anywhere.")
        else:
            view.banner = "NO KEY RELEASED"
            view.details.append("The error-correcting decoder could not rebuild the key from this iris.")
        return view

    def revocation_view(self, claimed: str, image: np.ndarray, threshold: float, title: str) -> RunView:
        system = self.make_system(False, False, threshold)
        features = system.pipeline.extract(image)
        matching = self.base.matching
        stolen = self.store.templates(claimed)
        current = self.store.distance(claimed, features.code, matching)
        replacement, meta = self.store.new_transform(claimed)
        reissued = replacement.apply(features.code)
        cross = min(hamming_distance(reissued, template, matching).distance for template in stolen)
        details = [
            f"Protection version : {self.store.protection_version(claimed)} (stored templates are transformed, not raw IrisCodes)",
            f"Live iris vs stored template      : HD {current.distance:.3f}  (accept if <= {threshold:.3f})",
            f"After revocation (version {meta['version']}), the same iris is re-enrolled under a new key.",
            f"Stolen old template vs new template: HD {cross:.3f}  (about 0.5 means unrelated)",
            "The leaked template can no longer be matched, and the old and new records cannot be linked.",
        ]
        return RunView(
            banner="STOLEN TEMPLATE IS USELESS AFTER REVOCATION",
            tone="warn",
            image=compose_panel(features, [title, f"stolen vs reissued template  HD {cross:.3f}"]),
            details=details,
            result=None,
        )

    def identify_image(
        self,
        image: np.ndarray,
        liveness: Optional[np.ndarray],
        pad_enabled: bool,
        require_reaction: bool,
        threshold: float,
        title: str,
    ) -> RunView:
        system = self.make_system(pad_enabled, require_reaction, threshold)
        result = system.identify(image, liveness)
        features = self.features_for(image, system, result)
        return self.build_view(result, image, features, title, None)

    def verify(
        self,
        claimed: str,
        probe: Path,
        liveness: Optional[Path],
        pad_enabled: bool,
        require_reaction: bool,
        threshold: float,
        title: Optional[str] = None,
    ) -> RunView:
        frame = self.read(liveness) if liveness else None
        return self.verify_image(
            claimed, self.read(probe), frame, pad_enabled, require_reaction, threshold,
            title or f"Probe: {Path(probe).name}   claimed: {claimed}",
        )

    def identify(
        self,
        probe: Path,
        liveness: Optional[Path],
        pad_enabled: bool,
        require_reaction: bool,
        threshold: float,
        title: Optional[str] = None,
    ) -> RunView:
        frame = self.read(liveness) if liveness else None
        return self.identify_image(
            self.read(probe), frame, pad_enabled, require_reaction, threshold,
            title or f"Probe: {Path(probe).name}   1:N identification",
        )

    def encode_only(self, probe: Path) -> RunView:
        pipeline = IrisPipeline(self.base)
        image = self.read(probe)
        try:
            features = pipeline.extract(image)
        except SegmentationError as error:
            return RunView("SEGMENTATION FAILED", "info", to_bgr(image), [f"Error: {error}"])
        code = features.code
        panel = compose_panel(features, [f"Probe: {Path(probe).name}", f"IrisCode {code.size} bits"])
        details = [
            f"IrisCode           : {code.size} bits, {int(code.mask.sum())} usable",
            f"Pupil              : x={features.segmentation.pupil.x:.1f} y={features.segmentation.pupil.y:.1f} r={features.segmentation.pupil.r:.1f}",
            f"Iris               : x={features.segmentation.iris.x:.1f} y={features.segmentation.iris.y:.1f} r={features.segmentation.iris.r:.1f}",
            f"Image quality      : sharpness {features.quality.sharpness:.1f}, usable iris {features.quality.usable_fraction * 100:.0f}%",
            "Hex (first 96 digits): " + code.to_hex()[:96] + "...",
        ]
        return RunView("IRISCODE EXTRACTED", "info", panel, details)

    def analyze_capture(self, gray: np.ndarray) -> Tuple[bool, str, np.ndarray]:
        pipeline = IrisPipeline(self.base)
        try:
            features = pipeline.extract(gray)
        except SegmentationError as error:
            return False, f"No iris found ({error}). Move closer, center the eye in the circle, reduce glare.", to_bgr(gray)
        overlay = draw_segmentation(features.segmentation)
        quality = features.quality
        if not quality.passed:
            return False, "Retake: " + "; ".join(quality.failures), overlay
        return True, f"Iris found. Usable iris area {quality.usable_fraction * 100:.0f}%, sharpness {quality.sharpness:.1f}.", overlay

    def save_captures(self, tag: str, images: Sequence[np.ndarray]) -> List[Path]:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        paths = []
        for index, image in enumerate(images, start=1):
            path = self.capture_dir / f"{safe_name(tag)}_{stamp}_{index:02d}.png"
            write_image(path, image)
            paths.append(path)
        return paths

    def enroll_images(self, user: str, images: Sequence[np.ndarray], pad_enabled: bool, threshold: float) -> str:
        system = self.make_system(pad_enabled, False, threshold)
        try:
            features = system.enroll(user, list(images))
        except EnrollmentError as error:
            return f"Enrollment failed: {error}"
        return f"Enrolled {user} with {len(features)} template(s)"

    def enroll(self, user: str, paths: Sequence[Path], pad_enabled: bool, threshold: float) -> str:
        return self.enroll_images(user, [self.read(path) for path in paths], pad_enabled, threshold)

    def import_dataset(
        self, root: Path, threshold: float, max_users: int = 60, per_user: int = 3, strangers: int = STRANGERS
    ) -> Tuple[str, Optional[Path], Optional[str]]:
        groups: Dict[str, List[Path]] = {}
        for label, path in load_dataset([root]):
            groups.setdefault(label, []).append(path)
        labels = sorted(groups)
        held_out = labels[-strangers:] if len(labels) > 2 * strangers else []
        candidates = [label for label in labels if label not in held_out]
        system = self.make_system(False, False, threshold)
        enrolled: Dict[str, Tuple[str, List[Path]]] = {}
        skipped: List[str] = []
        for label in candidates:
            if len(enrolled) >= max_users:
                break
            name = display_name(label)
            paths = sorted(groups[label])
            take = max(1, min(per_user, len(paths) - 1))
            if self.store.has_user(name):
                enrolled[label] = (name, paths[take:])
                continue
            try:
                system.enroll(name, [self.read(path) for path in paths[:take]])
            except (EnrollmentError, ValueError, FileNotFoundError) as error:
                skipped.append(f"{name} ({error})")
                continue
            enrolled[label] = (name, paths[take:])
        stranger_paths = {label: sorted(groups[label]) for label in held_out}
        plan = self.build_plan(threshold, enrolled, stranger_paths, groups)
        self.plan_path.parent.mkdir(parents=True, exist_ok=True)
        self.plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        message = f"Imported {len(enrolled)} user(s) from {Path(root).name}, named {', '.join(display_name(label) for label in list(enrolled)[:3])}..."
        if held_out:
            message += f" Kept {len(held_out)} eyes un-enrolled to play strangers."
        if skipped:
            message += f" Skipped {len(skipped)}: " + "; ".join(skipped[:3])
        example = None
        example_user = None
        for label, (name, probes) in enrolled.items():
            if probes:
                example, example_user = probes[0], name
                break
        if example is not None:
            message += f" Example probe selected: {example.name} for {example_user}."
        return message, example, example_user

    def build_plan(
        self,
        threshold: float,
        enrolled: Dict[str, Tuple[str, List[Path]]],
        strangers: Dict[str, List[Path]],
        groups: Dict[str, List[Path]],
    ) -> Dict:
        system = self.make_system(False, False, threshold)
        pipeline = system.pipeline
        matching = self.base.matching
        identify_limit = min(threshold, system.identification_threshold())
        names = [name for name, _ in enrolled.values()]

        def best(code, user: str) -> float:
            return self.store.distance(user, code, matching).distance

        def code_for(path: Path):
            try:
                features = pipeline.extract(self.read(path))
            except (SegmentationError, ValueError, FileNotFoundError):
                return None
            return features.code if features.quality.passed else None

        eligible = []
        for label, (name, probes) in enrolled.items():
            if not probes:
                continue
            code = code_for(probes[0])
            if code is None:
                continue
            distances = {user: best(code, user) for user in names}
            own = distances.pop(name)
            if own <= identify_limit and all(value > identify_limit for value in distances.values()):
                eligible.append((label, name, probes[0], code))
        plan: Dict = {}
        if not eligible:
            return plan
        label_a, user_a, probe_a, code_a = eligible[0]
        plan["genuine"] = {"claimed": user_a, "probe": str(probe_a), "label": label_a}
        for label_b, user_b, probe_b, code_b in eligible[1:]:
            if person_key(label_b) != person_key(label_a) and best(code_b, user_a) > threshold:
                plan["impostor"] = {"claimed": user_a, "probe": str(probe_b), "from": user_b}
                break
        known = [item for item in eligible if item[0] != label_a]
        if known:
            pick = known[len(known) // 2]
            plan["identify_known"] = {"probe": str(pick[2]), "expected": pick[1]}
        other = other_eye_label(label_a)
        if other in enrolled and enrolled[other][1]:
            probe = enrolled[other][1][0]
            code = code_for(probe)
            if code is not None and best(code, user_a) > threshold:
                plan["other_eye"] = {"claimed": user_a, "probe": str(probe), "from": enrolled[other][0]}
        usable = []
        for label, paths in strangers.items():
            for path in paths:
                code = code_for(path)
                if code is not None and all(best(code, user) > identify_limit for user in names):
                    usable.append((label, path))
                    break
        if usable:
            label, path = usable[0]
            plan["stranger"] = {"claimed": user_a, "probe": str(path), "from": display_name(label)}
            other_people = [item for item in usable if person_key(item[0]) != person_key(label)]
            label2, path2 = (other_people or usable)[0]
            plan["identify_stranger"] = {"probe": str(path2), "from": display_name(label2)}
        plan["blur"] = {"claimed": user_a, "probe": str(probe_a)}
        vault = FuzzyVault(self.base.security.vault_key_symbols)

        def opens(user: str, code) -> bool:
            return any(
                vault.unlock(record, code, matching) is not None for record in self.store.vault_records(user)
            )

        for label, user, probe, code in eligible:
            if not self.store.vault_records(user) or not opens(user, code):
                continue
            plan["vault"] = {"claimed": user, "probe": str(probe), "label": label}
            for label_b, user_b, probe_b, code_b in eligible:
                if person_key(label_b) != person_key(label) and not opens(user, code_b):
                    plan["vault_impostor"] = {"claimed": user, "probe": str(probe_b), "from": user_b}
                    break
            break
        detector = PresentationAttackDetector(self.base.pad)
        original = self.read(probe_a)
        for seed in range(1, 25):
            attack = simulate_attack("halftone_print", original, seed)
            try:
                features = pipeline.extract(attack)
            except (SegmentationError, ValueError):
                continue
            if not features.quality.passed or best(features.code, user_a) > threshold:
                continue
            if detector.assess(features.segmentation).passed:
                continue
            entry = {"claimed": user_a, "probe": str(probe_a), "kind": "halftone_print", "seed": seed}
            plan["print_attack"] = dict(entry)
            plan["print_attack_open"] = dict(entry)
            break
        plan["revocation"] = {"claimed": user_a, "probe": str(probe_a)}
        if "impostor" in plan:
            plan["lockout"] = {
                "claimed": user_a,
                "probe": str(probe_a),
                "impostor_probe": plan["impostor"]["probe"],
                "from": plan["impostor"]["from"],
            }
        return plan

    def run_scenario(self, key: str, threshold: float) -> RunView:
        plan = self.load_plan()
        if not plan or key not in plan:
            raise ValueError("This scenario needs imported dataset users. Open the Dataset tab and click Import dataset users.")
        entry = plan[key]
        claimed = entry.get("claimed")
        for failure_key in (claimed, UNKNOWN_KEY, IDENTIFY_KEY):
            if failure_key:
                self.store.clear_failures(failure_key)
        title = next(item.title for item in ALL_SCENARIOS if item.key == key)
        probe = Path(entry["probe"])
        if key == "genuine":
            return self.verify(claimed, probe, None, False, False, threshold, f"{title}: {claimed}")
        if key == "impostor":
            return self.verify(claimed, probe, None, False, False, threshold, f"{entry['from']}'s eye claims to be {claimed}")
        if key == "stranger":
            return self.verify(claimed, probe, None, False, False, threshold, f"Stranger ({entry['from']}) claims to be {claimed}")
        if key == "other_eye":
            return self.verify(claimed, probe, None, False, False, threshold, f"{entry['from']} presented as {claimed}")
        if key == "identify_known":
            return self.identify(probe, None, False, False, threshold, f"Who is this? (really {entry['expected']})")
        if key == "identify_stranger":
            return self.identify(probe, None, False, False, threshold, f"Who is this? (stranger {entry['from']})")
        if key == "vault":
            return self.vault_image(claimed, self.read(probe), None, threshold, f"{title}: {claimed}")
        if key == "vault_impostor":
            return self.vault_image(claimed, self.read(probe), None, threshold, f"{entry['from']}'s eye tries to open {claimed}'s record")
        if key in ("print_attack", "print_attack_open"):
            attack = simulate_attack(entry["kind"], self.read(probe), int(entry["seed"]))
            return self.verify_image(claimed, attack, None, key == "print_attack", False, threshold, f"{title}: {claimed}")
        if key == "revocation":
            return self.revocation_view(claimed, self.read(probe), threshold, title)
        if key == "blur":
            image = self.read(probe)
            blurred = cv2.GaussianBlur(image, (0, 0), image.shape[1] / 40.0)
            return self.verify_image(claimed, blurred, None, False, False, threshold, f"Out-of-focus capture of {claimed}")
        attempts = 0
        for _ in range(self.base.security.max_failed_attempts):
            self.verify(claimed, Path(entry["impostor_probe"]), None, False, False, threshold, f"Attempt {attempts + 1}: {entry['from']}'s eye")
            attempts += 1
        final = self.verify(claimed, probe, None, False, False, threshold, f"Attempt {attempts + 1}: the real {claimed}")
        self.store.clear_failures(claimed)
        final.details.append(f"Attempts 1-{attempts} used another person's eye and were denied.")
        final.details.append(f"Attempt {attempts + 1} is the real owner, and is still blocked until the lockout ends.")
        final.details.append("The lockout was cleared after this demo.")
        return final

    def evaluate_dataset(self, root: Path, threshold: float) -> Tuple[Path, List[str]]:
        config = replace(self.base, matching=replace(self.base.matching, threshold=threshold))
        records = load_dataset([root])
        result = evaluate_records(records, config, reader=read_visible)
        save_report(result, config, self.output_dir)
        figure = self.output_dir / "evaluation_gui.png"
        plot_evaluation(result, config, figure, dpi=72)
        summary = [
            f"Dataset {Path(root).name}: images used {result.acquired}/{result.images}   genuine pairs {result.genuine.size}   impostor pairs {result.impostor.size}",
            f"Genuine HD mean {result.genuine.mean():.3f}   impostor HD mean {result.impostor.mean():.3f}   EER {result.eer * 100:.2f}% at threshold {result.eer_threshold:.3f}",
        ]
        for point in result.operating_points:
            summary.append(
                f"threshold {point['threshold']:.2f}:  FAR {point['far']:.6f}   FRR {point['frr']:.4f}   model FAR {point['model_far']:.2e}"
            )
        return figure, summary
