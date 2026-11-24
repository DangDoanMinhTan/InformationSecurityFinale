import base64
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Callable, List, Optional

import cv2
import numpy as np

from .app_logic import SCENARIOS, SECURITY_SCENARIOS, AppController, RunView
from .paths import app_root
from .webcam import Camera, preview_frame, process_frame

TONE_COLORS = {
    "good": "#1f8f5f",
    "bad": "#c62839",
    "warn": "#d9730d",
    "info": "#4a5568",
    "locked": "#6b3fa0",
    "idle": "#2d3748",
}

FILE_TYPES = [("Eye images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.pgm"), ("All files", "*.*")]
WEBCAM_HELP = (
    "Hold ONE eye close to the camera and look straight at the lens. Move until the iris fills the "
    "orange circle. Use bright, even light from the front and avoid reflections. Use the same eye every time."
)


def make_photo(image: np.ndarray, max_width: int, max_height: int, max_scale: float = 1.0) -> tk.PhotoImage:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, max_scale)
    if abs(scale - 1.0) > 0.01:
        method = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=method)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("cannot encode image for display")
    return tk.PhotoImage(data=base64.b64encode(encoded.tobytes()))


class WebcamDialog(tk.Toplevel):
    def __init__(self, parent: tk.Misc, camera_factory: Callable, analyze: Callable, needed: int, title: str):
        super().__init__(parent)
        self.title(title)
        self.configure(bg="#edf0f4")
        self.transient(parent)
        self.camera_factory = camera_factory
        self.analyze = analyze
        self.needed = needed
        self.collected: List[np.ndarray] = []
        self.result: Optional[List[np.ndarray]] = None
        self.frozen: Optional[np.ndarray] = None
        self.frozen_ok = False
        self.last_frame: Optional[np.ndarray] = None
        self.camera = None
        self.running = True
        self.photo: Optional[tk.PhotoImage] = None
        self.fraction = tk.DoubleVar(value=0.35)
        self.index = tk.IntVar(value=0)
        self.message = tk.StringVar(value="")
        self.progress = tk.StringVar()
        self.update_progress()
        ttk.Label(self, text=WEBCAM_HELP, wraplength=640, padding=(10, 8)).pack(fill="x")
        self.preview = tk.Label(self, bg="#202020", width=80, height=24)
        self.preview.pack(padx=10)
        ttk.Label(self, textvariable=self.progress, font=("Segoe UI", 10, "bold"), padding=(10, 6, 10, 0)).pack(fill="x")
        self.message_label = tk.Label(self, textvariable=self.message, bg="#edf0f4", fg="#333", wraplength=640, justify="left", anchor="w")
        self.message_label.pack(fill="x", padx=10, pady=(0, 6))
        controls = ttk.Frame(self, padding=(10, 0))
        controls.pack(fill="x")
        ttk.Label(controls, text="Guide size").pack(side="left")
        ttk.Scale(controls, from_=0.15, to=0.70, variable=self.fraction, length=160).pack(side="left", padx=(6, 16))
        ttk.Label(controls, text="Camera").pack(side="left")
        ttk.Spinbox(controls, from_=0, to=5, width=3, textvariable=self.index, command=self.start_camera).pack(side="left", padx=(6, 6))
        buttons = ttk.Frame(self, padding=10)
        buttons.pack(fill="x")
        self.capture_button = ttk.Button(buttons, text="Capture (Space)", command=self.capture)
        self.capture_button.pack(side="left")
        self.use_button = ttk.Button(buttons, text="Use this capture (Enter)", command=self.accept, state="disabled")
        self.use_button.pack(side="left", padx=6)
        self.retake_button = ttk.Button(buttons, text="Retake", command=self.retake, state="disabled")
        self.retake_button.pack(side="left")
        ttk.Button(buttons, text="Cancel (Esc)", command=self.cancel).pack(side="right")
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.bind("<space>", lambda _: self.capture())
        self.bind("<Return>", lambda _: self.accept())
        self.bind("<Escape>", lambda _: self.cancel())
        self.start_camera()
        self.after(30, self.tick)
        try:
            self.wait_visibility()
            self.grab_set()
        except tk.TclError:
            pass
        self.focus_set()

    def update_progress(self) -> None:
        self.progress.set(f"Capture {min(len(self.collected) + 1, self.needed)} of {self.needed}")

    def start_camera(self) -> None:
        if self.camera is not None:
            self.camera.release()
        self.camera = self.camera_factory(int(self.index.get()))
        if not self.camera.is_open():
            self.message.set(
                f"Camera {self.index.get()} could not be opened. Close other apps using it, check Windows camera "
                "privacy settings, or try another camera number."
            )
            self.capture_button.state(["disabled"])
        else:
            self.message.set("")
            self.capture_button.state(["!disabled"])

    def show(self, image: np.ndarray) -> None:
        self.photo = make_photo(image, 640, 600, 2.0)
        self.preview.configure(image=self.photo, width=self.photo.width(), height=self.photo.height())

    def tick(self) -> None:
        if not self.running:
            return
        if self.frozen is None and self.camera is not None and self.camera.is_open():
            frame = self.camera.read()
            if frame is not None:
                self.last_frame = frame
                self.show(preview_frame(frame, float(self.fraction.get())))
        self.after(33, self.tick)

    def capture(self) -> None:
        if self.frozen is not None or self.last_frame is None:
            return
        gray = process_frame(self.last_frame.copy(), float(self.fraction.get()))
        ok, message, overlay = self.analyze(gray)
        self.frozen = gray
        self.frozen_ok = ok
        self.show(cv2.resize(overlay, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC))
        self.message.set(message)
        self.message_label.configure(fg="#1f8f5f" if ok else "#c62839")
        self.capture_button.state(["disabled"])
        self.use_button.state(["!disabled"] if ok else ["disabled"])
        self.retake_button.state(["!disabled"])

    def accept(self) -> None:
        if self.frozen is None or not self.frozen_ok:
            return
        self.collected.append(self.frozen)
        self.frozen = None
        if len(self.collected) >= self.needed:
            self.result = list(self.collected)
            self.close()
            return
        self.update_progress()
        self.reset_buttons()
        self.message.set(f"Captured {len(self.collected)} of {self.needed}. Capture the next image of the same eye.")
        self.message_label.configure(fg="#333")

    def retake(self) -> None:
        self.frozen = None
        self.reset_buttons()
        self.message.set("")

    def reset_buttons(self) -> None:
        self.capture_button.state(["!disabled"])
        self.use_button.state(["disabled"])
        self.retake_button.state(["disabled"])

    def cancel(self) -> None:
        self.result = None
        self.close()

    def close(self) -> None:
        self.running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        try:
            self.grab_release()
        except tk.TclError:
            pass
        self.destroy()


class IrisApp(tk.Tk):
    def __init__(self, controller: AppController, camera_factory: Optional[Callable] = None):
        super().__init__()
        self.controller = controller
        self.camera_factory = camera_factory or (lambda index: Camera(index))
        self.title("Iris Recognition - real eyes only (Daugman IrisCode)")
        height = min(880, max(640, self.winfo_screenheight() - 90))
        self.geometry(f"1380x{height}")
        self.minsize(1200, 640)
        self.configure(bg="#edf0f4")
        self.queue: "queue.Queue" = queue.Queue()
        self.busy = False
        self.photo: Optional[tk.PhotoImage] = None
        self.plot_photo: Optional[tk.PhotoImage] = None
        self.webcam_dialog: Optional[WebcamDialog] = None
        self.actions: List[ttk.Button] = []
        self.scenario_buttons: List[ttk.Button] = []
        self.probe_path = tk.StringVar()
        self.liveness_path = tk.StringVar()
        self.probe_label = tk.StringVar(value="(none chosen)")
        self.liveness_label = tk.StringVar(value="(none)")
        self.probe_path.trace_add("write", lambda *_: self.probe_label.set(Path(self.probe_path.get()).name or "(none chosen)"))
        self.liveness_path.trace_add("write", lambda *_: self.liveness_label.set(Path(self.liveness_path.get()).name or "(none)"))
        self.claimed = tk.StringVar()
        self.pad_enabled = tk.BooleanVar(value=False)
        self.require_reaction = tk.BooleanVar(value=False)
        self.threshold = tk.DoubleVar(value=0.36)
        self.threshold_text = tk.StringVar(value="0.36")
        self.status = tk.StringVar(value="Opening encrypted database...")
        self.configure_style()
        self.build_widgets()
        self.set_busy(True)
        self.after(100, self.poll)
        self.run_async(controller.prepare, self.on_prepared)

    def configure_style(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background="#edf0f4")
        style.configure("TLabelframe", background="#edf0f4")
        style.configure("TLabelframe.Label", background="#edf0f4", font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background="#edf0f4", font=("Segoe UI", 9))
        style.configure("TCheckbutton", background="#edf0f4", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 9), padding=4)
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=6)
        style.configure("TNotebook", background="#edf0f4")

    def section(self, parent: tk.Misc, title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, padding=6)
        frame.pack(fill="x", pady=(0, 5))
        return frame

    def button(self, parent: tk.Misc, text: str, command: Callable[[], None], primary: bool = False) -> ttk.Button:
        widget = ttk.Button(parent, text=text, command=command, style="Primary.TButton" if primary else "TButton")
        self.actions.append(widget)
        return widget

    def build_widgets(self) -> None:
        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w", padding=(8, 2)).pack(side="bottom", fill="x")
        left = ttk.Frame(self, padding=(10, 10, 5, 10))
        left.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=(5, 10, 10, 10))
        right.pack(side="left", fill="both", expand=True)

        database = self.section(left, "Database (real eyes only)")
        self.user_box = ttk.Combobox(database, textvariable=self.claimed, state="readonly", width=36)
        self.user_box.pack(fill="x")
        row = ttk.Frame(database)
        row.pack(fill="x", pady=(6, 0))
        self.button(row, "Enroll from photos...", self.do_enroll).pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.button(row, "Reset database", self.do_reset).pack(side="left", expand=True, fill="x", padx=(3, 0))
        self.button(database, "Clear lockouts", self.do_clear_lockouts).pack(fill="x", pady=(4, 0))

        self.tabs = ttk.Notebook(left)
        self.tabs.pack(fill="x", pady=(0, 5))
        scenario_tab = ttk.Frame(self.tabs, padding=6)
        webcam_tab = ttk.Frame(self.tabs, padding=6)
        image_tab = ttk.Frame(self.tabs, padding=6)
        dataset_tab = ttk.Frame(self.tabs, padding=6)
        security_tab = ttk.Frame(self.tabs, padding=6)
        self.tabs.add(scenario_tab, text="Scenarios")
        self.tabs.add(security_tab, text="Security")
        self.tabs.add(webcam_tab, text="Webcam")
        self.tabs.add(image_tab, text="Photo file")
        self.tabs.add(dataset_tab, text="Dataset")

        ttk.Label(
            scenario_tab,
            text="Real-eye scenarios from the imported dataset.\nImport it first on the Dataset tab.",
            foreground="#444", justify="left",
        ).pack(fill="x", pady=(0, 4))
        for scenario in SCENARIOS:
            widget = self.button(scenario_tab, scenario.title, lambda item=scenario: self.do_scenario(item.key))
            widget.pack(fill="x", pady=1)
            self.scenario_buttons.append(widget)

        ttk.Label(
            security_tab,
            text="Protection layer on the imported users:\nfuzzy-commitment key release, simulated print attack,\ncancelable templates.",
            foreground="#444", justify="left",
        ).pack(fill="x", pady=(0, 4))
        for scenario in SECURITY_SCENARIOS:
            widget = self.button(security_tab, scenario.title, lambda item=scenario: self.do_scenario(item.key))
            widget.pack(fill="x", pady=1)
            self.scenario_buttons.append(widget)

        self.button(webcam_tab, "Enroll myself with webcam...", self.do_webcam_enroll, True).pack(fill="x", pady=2)
        self.button(webcam_tab, "Verify with webcam (1:1)", self.do_webcam_verify, True).pack(fill="x", pady=2)
        self.button(webcam_tab, "Identify with webcam (1:N)", self.do_webcam_identify).pack(fill="x", pady=2)
        ttk.Label(webcam_tab, text="Captures are saved in the captures folder.", foreground="#444").pack(fill="x", pady=(6, 0))

        probe = ttk.LabelFrame(image_tab, text="Eye photo", padding=6)
        probe.pack(fill="x", pady=(0, 5))
        ttk.Label(probe, textvariable=self.probe_label, foreground="#444").pack(fill="x")
        self.button(probe, "Choose photo...", self.browse_probe).pack(fill="x", pady=(4, 0))
        ttk.Label(probe, text="Optional bright-light photo (pupil reaction):").pack(fill="x", pady=(6, 0))
        ttk.Label(probe, textvariable=self.liveness_label, foreground="#444").pack(fill="x")
        row = ttk.Frame(probe)
        row.pack(fill="x", pady=(4, 0))
        self.button(row, "Choose...", self.browse_liveness).pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.button(row, "Clear", lambda: self.liveness_path.set("")).pack(side="left", expand=True, fill="x", padx=(3, 0))
        self.button(image_tab, "Verify claimed user (1:1)", self.do_verify).pack(fill="x", pady=2)
        self.button(image_tab, "Identify in database (1:N)", self.do_identify).pack(fill="x", pady=2)
        self.button(image_tab, "Extract IrisCode only", self.do_encode).pack(fill="x", pady=2)

        ttk.Label(
            dataset_tab,
            text="Use a folder of real iris images, one sub-folder per eye\n(for example CASIA-Iris-Interval, IITD, UBIRIS).",
            foreground="#444", justify="left",
        ).pack(fill="x", pady=(0, 6))
        self.button(dataset_tab, "Import dataset users...", self.do_import).pack(fill="x", pady=2)
        self.button(dataset_tab, "Measure FAR / FRR / EER...", self.do_evaluate, True).pack(fill="x", pady=2)

        self.banner = tk.Label(
            right, text="Choose a scenario or use the webcam", font=("Segoe UI", 22, "bold"), fg="white",
            bg=TONE_COLORS["idle"], pady=14,
        )
        self.banner.pack(fill="x")
        self.display = tk.Label(right, bg="#f5f5f5", relief="flat")
        self.display.pack(fill="both", expand=True, pady=(8, 8))
        self.details = tk.Text(right, height=10, font=("Consolas", 10), bg="#ffffff", relief="flat", state="disabled", wrap="none")
        self.details.pack(fill="x")

    def set_busy(self, busy: bool) -> None:
        self.busy = busy
        for widget in self.actions:
            widget.state(["disabled"] if busy else ["!disabled"])
        self.configure(cursor="watch" if busy else "")

    def run_async(self, work: Callable, done: Callable, label: str = "") -> None:
        if label:
            self.status.set(label)
        self.set_busy(True)

        def target() -> None:
            try:
                self.queue.put((done, "ok", work()))
            except Exception as error:
                self.queue.put((done, "error", error))

        threading.Thread(target=target, daemon=True).start()

    def poll(self) -> None:
        try:
            while True:
                done, state, value = self.queue.get_nowait()
                self.set_busy(False)
                if state == "ok":
                    done(value)
                else:
                    self.status.set("Error")
                    messagebox.showerror("Error", str(value))
        except queue.Empty:
            pass
        self.after(100, self.poll)

    def on_prepared(self, _: None) -> None:
        self.apply_defaults()
        self.tabs.select(0 if self.controller.scenarios_ready() else 1)
        self.refresh_users()
        self.status.set(f"Ready. {len(self.controller.users())} user(s) enrolled in the encrypted database.")

    def refresh_users(self) -> None:
        users = self.controller.users()
        self.user_box["values"] = users
        if users and self.claimed.get() not in users:
            self.claimed.set(users[0])
        if not users:
            self.claimed.set("")

    def apply_defaults(self) -> None:
        threshold, pad = self.controller.default_settings()
        self.threshold.set(threshold)
        self.threshold_text.set(f"{threshold:.2f}")
        self.pad_enabled.set(pad)
        self.require_reaction.set(False)

    def current_threshold(self) -> float:
        return round(float(self.threshold.get()), 2)

    def browse_probe(self) -> None:
        initial = self.controller.capture_dir if self.controller.capture_dir.exists() else app_root()
        path = filedialog.askopenfilename(title="Choose eye photo", initialdir=str(initial), filetypes=FILE_TYPES)
        if path:
            self.probe_path.set(path)

    def browse_liveness(self) -> None:
        initial = self.controller.capture_dir if self.controller.capture_dir.exists() else app_root()
        path = filedialog.askopenfilename(title="Choose bright-light frame", initialdir=str(initial), filetypes=FILE_TYPES)
        if path:
            self.liveness_path.set(path)

    def require_probe(self) -> Optional[Path]:
        if not self.probe_path.get():
            messagebox.showinfo("Image", "Choose an eye photo first.")
            return None
        return Path(self.probe_path.get())

    def liveness(self) -> Optional[Path]:
        return Path(self.liveness_path.get()) if self.liveness_path.get() else None

    def do_verify(self) -> None:
        probe = self.require_probe()
        if probe is None:
            return
        if not self.claimed.get():
            messagebox.showinfo("Claimed identity", "Select an enrolled user to verify against.")
            return
        claimed, liveness = self.claimed.get(), self.liveness()
        pad = reaction = liveness is not None
        threshold = self.current_threshold()
        self.run_async(
            lambda: self.controller.verify(claimed, probe, liveness, pad, reaction, threshold),
            self.show_view,
            "Verifying...",
        )

    def do_identify(self) -> None:
        probe = self.require_probe()
        if probe is None:
            return
        liveness = self.liveness()
        pad = reaction = liveness is not None
        threshold = self.current_threshold()
        self.run_async(
            lambda: self.controller.identify(probe, liveness, pad, reaction, threshold),
            self.show_view,
            "Identifying...",
        )

    def do_encode(self) -> None:
        probe = self.require_probe()
        if probe is None:
            return
        self.run_async(lambda: self.controller.encode_only(probe), self.show_view, "Extracting IrisCode...")

    def do_enroll(self) -> None:
        name = simpledialog.askstring("Enroll user", "User name:", parent=self)
        if not name or not name.strip():
            return
        paths = filedialog.askopenfilenames(title="Choose 1 to 5 real eye photos of this user", filetypes=FILE_TYPES)
        if not paths:
            return
        files = [Path(item) for item in paths]
        pad, threshold = self.pad_enabled.get(), self.current_threshold()
        user = name.strip()

        def finished(message: str) -> None:
            self.refresh_users()
            if self.controller.store.has_user(user):
                self.claimed.set(user)
            self.status.set(message)
            messagebox.showinfo("Enrollment", message)

        self.run_async(lambda: self.controller.enroll(user, files, pad, threshold), finished, "Enrolling...")

    def do_scenario(self, key: str) -> None:
        plan = self.controller.load_plan()
        if not plan or key not in plan:
            messagebox.showinfo(
                "Scenario",
                "This scenario needs imported dataset users.\nOpen the Dataset tab and click Import dataset users.",
            )
            return
        entry = plan[key]
        threshold = self.current_threshold()
        self.probe_path.set(entry["probe"])
        self.liveness_path.set("")
        if entry.get("claimed"):
            self.claimed.set(entry["claimed"])
        self.pad_enabled.set(False)
        self.require_reaction.set(False)
        self.run_async(lambda: self.controller.run_scenario(key, threshold), self.show_view, "Running scenario...")

    def do_clear_lockouts(self) -> None:
        self.controller.clear_lockouts()
        self.status.set("Lockouts cleared.")

    def do_reset(self) -> None:
        if not messagebox.askyesno("Reset database", "Delete ALL enrolled users?"):
            return

        def finished(_: None) -> None:
            self.refresh_users()
            self.status.set("Database reset. Lockouts cleared.")

        self.run_async(self.controller.reset_database, finished, "Resetting database...")

    def do_evaluate(self) -> None:
        folder = filedialog.askdirectory(title="Choose a labelled dataset folder (one sub-folder per eye)")
        if not folder:
            return
        threshold = self.current_threshold()

        def finished(value) -> None:
            figure, summary = value
            self.status.set("Evaluation finished.")
            self.show_plot(figure, summary, Path(folder).name)

        self.run_async(
            lambda: self.controller.evaluate_dataset(Path(folder), threshold),
            finished,
            "Measuring FAR / FRR on the dataset (can take a few minutes)...",
        )

    def capture_frames(self, needed: int, title: str) -> Optional[List[np.ndarray]]:
        dialog = WebcamDialog(self, self.camera_factory, self.controller.analyze_capture, needed, title)
        self.webcam_dialog = dialog
        self.wait_window(dialog)
        self.webcam_dialog = None
        return dialog.result

    def do_webcam_enroll(self) -> None:
        self.webcam_enroll()

    def webcam_enroll(self) -> None:
        name = simpledialog.askstring("Enroll with webcam", "Your name:", parent=self)
        if not name or not name.strip():
            return
        user = name.strip()
        frames = self.capture_frames(3, f"Enroll {user} - capture 3 images of the same eye")
        if not frames:
            return
        self.controller.save_captures(user, frames)
        pad, threshold = self.pad_enabled.get(), self.current_threshold()

        def finished(message: str) -> None:
            self.refresh_users()
            if self.controller.store.has_user(user):
                self.claimed.set(user)
            self.status.set(message)
            messagebox.showinfo("Enrollment", message)

        self.run_async(lambda: self.controller.enroll_images(user, frames, pad, threshold), finished, "Enrolling...")

    def do_webcam_verify(self) -> None:
        self.webcam_verify()

    def webcam_verify(self) -> None:
        if not self.controller.users():
            messagebox.showinfo("Verify", "Enroll at least one user first.")
            return
        claimed = self.claimed.get()
        frames = self.capture_frames(1, f"Verify as {claimed} - capture 1 image")
        if not frames:
            return
        self.controller.save_captures(f"probe_{claimed}", frames)
        pad, threshold = self.pad_enabled.get(), self.current_threshold()
        self.run_async(
            lambda: self.controller.verify_image(
                claimed, frames[0], None, pad, False, threshold, f"Webcam probe   claimed: {claimed}"
            ),
            self.show_view,
            "Verifying...",
        )

    def do_webcam_identify(self) -> None:
        self.webcam_identify()

    def webcam_identify(self) -> None:
        if not self.controller.users():
            messagebox.showinfo("Identify", "Enroll at least one user first.")
            return
        frames = self.capture_frames(1, "Identify - capture 1 image")
        if not frames:
            return
        self.controller.save_captures("probe_identify", frames)
        pad, threshold = self.pad_enabled.get(), self.current_threshold()
        self.run_async(
            lambda: self.controller.identify_image(frames[0], None, pad, False, threshold, "Webcam probe   1:N identification"),
            self.show_view,
            "Identifying...",
        )

    def do_import(self) -> None:
        self.import_dataset()

    def import_dataset(self) -> None:
        folder = filedialog.askdirectory(title="Choose the dataset folder (for example CASIA-Iris-Interval)")
        if not folder:
            return
        threshold = self.current_threshold()

        def finished(value) -> None:
            message, example, user = value
            self.refresh_users()
            if example is not None:
                self.probe_path.set(str(example))
                self.claimed.set(user)
            self.status.set(message)
            self.tabs.select(0 if self.controller.scenarios_ready() else 3)
            messagebox.showinfo("Import", message)

        self.run_async(
            lambda: self.controller.import_dataset(Path(folder), threshold),
            finished,
            "Importing dataset (about 1 second per user, up to 60 users)...",
        )

    def show_view(self, view: RunView) -> None:
        self.banner.configure(text=view.banner, bg=TONE_COLORS[view.tone])
        width = max(600, self.display.winfo_width() - 8)
        height = max(300, self.display.winfo_height() - 8)
        self.photo = make_photo(view.image, width, height, 1.5)
        self.display.configure(image=self.photo)
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", "\n".join(view.details))
        self.details.configure(state="disabled")
        self.status.set(view.banner)

    def show_plot(self, figure: Path, summary: List[str], name: str) -> None:
        window = tk.Toplevel(self)
        window.title(f"FAR / FRR / EER on {name}")
        window.configure(bg="#edf0f4")
        image = cv2.imdecode(np.fromfile(str(figure), dtype=np.uint8), cv2.IMREAD_COLOR)
        self.plot_photo = make_photo(image, 1300, 420)
        tk.Label(window, image=self.plot_photo, bg="#edf0f4").pack(padx=10, pady=10)
        tk.Label(window, text="\n".join(summary), font=("Consolas", 10), justify="left", bg="#edf0f4").pack(padx=10, pady=(0, 10), anchor="w")


def main() -> None:
    controller = AppController(app_root())
    IrisApp(controller).mainloop()
