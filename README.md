# Iris Recognition System (real eyes only)

Final project for Introduction to Information Security, Topic 1: Iris Recognition Systems.
Daugman-style pipeline in Python with OpenCV: segmentation, rubber-sheet normalization, 2D Gabor phase encoding into a 2,048-bit IrisCode, masked Hamming matching with rotation compensation, plus the defenses proposed in the midterm report.

The program contains no eye generator and no sample data. Every eye it processes comes from your webcam, a photo you choose, or a real iris dataset folder. The attack and webcam simulations in `iris_auth/attacks.py` and `iris_auth/degradation.py` only transform those real photos (print, halftone, screen replay, webcam noise); they never create eyes.

## Setup

```
py -m pip install -r requirements.txt
```

`flask` is only needed for the network server (`serve`, `netdemo`); the window and the other commands do not import it.

## Window version

Double-click `run_gui.bat`, or run `py gui.py`.

To make a single `.exe`, double-click `build_exe.bat`. It creates `dist\IrisRecognition.exe`; copy that file anywhere and double-click it. The database (`database_real`), webcam captures (`captures`) and reports (`output`) are created next to the exe.

### Scenarios tab

After you import a dataset (Dataset tab), eight real-eye scenarios run with one click, each showing the segmentation, the IrisCode and the Hamming distance: genuine user, impostor (another person's eye), stranger who was never enrolled, same person's other eye, 1:N identification of an enrolled person, 1:N identification of a stranger, an out-of-focus capture, and a brute-force attempt that locks the account after 5 wrong tries. The probes are real photos held back from enrollment, and the four last eyes of the dataset are kept un-enrolled to play the strangers. **Clear lockouts** removes any lockout.

### Security tab

Five more one-click scenarios exercise the protection layer on the same imported users: (9) the owner's iris rebuilds the key and releases the protected secret, (10) another person's iris gets nothing, (11) a simulated halftone print of the owner's photo is stopped by attack detection, (12) the same print is accepted when attack detection is off, and (13) revocation: a stolen template no longer matches the re-issued one. The print is computed from the real probe photo by `iris_auth/attacks.py`; it is a simulation, not a photographed print.

For UPOL-style labels such as `001L` (person 001, left eye), users get readable names: `001L` becomes `Alice_L`, `002R` becomes `Bob_R`, and so on. Each eye is its own user. Other datasets keep their folder names.

### Webcam tab

- **Enroll myself with webcam**: type a name, then capture 3 images of the same eye. Each capture is checked immediately and you can retake it.
- **Verify with webcam (1:1)**: choose the enrolled name in the user box, capture once.
- **Identify with webcam (1:N)**: capture once and the system finds who it is.

Tips: hold one eye about 10 cm from the lens, look straight at the camera, fill the orange circle with the iris (Guide size slider; camera number box if you have several cameras), use bright even light from the front, avoid window or lamp reflections on the eye, and use the same eye every time. Enroll and verify under similar lighting.

### Photo file tab

Choose any real eye photo (for example a tight close-up from a phone camera or an image from a dataset), then Verify claimed user, Identify in database, or Extract IrisCode only. An optional second photo taken under bright light enables the pupil reaction check.

### Dataset tab

- **Import dataset users**: choose a folder such as CASIA-Iris-Interval. Up to 60 classes are enrolled (each with all but its last image, at most 3), and the held-back image of the first class is selected as the probe. Each class is one eye, so a person with a left and a right eye is two users.
- **Measure FAR / FRR / EER**: choose a labelled dataset folder. The program compares every pair of images, then shows the genuine and impostor distance histograms, the FAR/FRR curves, the EER and the FAR/FRR at several thresholds. This is the way to get honest accuracy numbers for the report.

Accepted dataset layouts: `<class>/<image>`, `<subject>/<L|R>/<image>`, or flat files named `<class><number>.<ext>`.

### Fixed settings

There is no settings panel; the values are fixed. The real-eye preset uses threshold 0.36 and 11 rotation shifts. Attack detection is off for ordinary webcam and photo attempts, because webcams capture visible light and the images are noisier, and it switches on automatically (with the pupil-reaction check) whenever you choose a second, bright-light photo on the Photo file tab. The Security tab scenarios turn it on or off themselves. The Hamming distance is shown for every attempt.

## Command line

| Command | Purpose |
| --- | --- |
| `py main.py enroll --user alice --images a1.jpg a2.jpg a3.jpg` | enroll a user |
| `py main.py verify --user alice --image probe.jpg --debug` | 1:1 verification, `--debug` shows the Hamming distance |
| `py main.py identify --image probe.jpg` | 1:N identification |
| `py main.py encode --image eye.jpg` | print the IrisCode and save an annotated panel |
| `py main.py evaluate --dataset path/to/dataset` | FAR, FRR, EER, plots |
| `py main.py list-users` / `remove-user --user alice` | database management |
| `py main.py vault-unlock --user alice --image probe.jpg` | release the key-bound secret with a live iris (fuzzy commitment) |
| `py main.py revoke --user alice` | revoke the cancelable template, then enroll again to issue a new one |
| `py main.py vault-eval --dataset path/to/dataset --key-symbols 12 20 28` | key-release rate and false releases of the fuzzy commitment |
| `py main.py attacks --dataset path/to/dataset` | simulated print, halftone and screen-replay attacks, with and without attack detection |
| `py main.py webcam-sim --dataset path/to/dataset` | accuracy on degraded webcam-like captures, visible light and a near-infrared proxy |
| `py main.py serve --port 8443` | HTTPS authentication server (self-signed certificate is created in `database_real\tls`); needs `IRIS_API_KEY` and `IRIS_ADMIN_KEY`; with `IRIS_SIGNING_KEY` set, every capture request must carry an HMAC signature over the challenge and image |
| `py main.py netdemo --enroll a1.png a2.png --probe a3.png` | capture plain HTTP versus TLS traffic with tcpdump and analyse it (needs root and tcpdump) |

The command line and the window share the same encrypted database (`database_real`). Database commands read the passphrase from the `IRIS_DB_PASSPHRASE` environment variable or prompt for it; the window uses the same variable, or a built-in default if it is not set. Optional flags: `--liveness bright_frame.jpg`, `--require-liveness`, `--pad`, `--threshold 0.30`, `--save-panel out.png`.

## Structure

```
main.py                  command line entry point
gui.py                   window version
run_gui.bat              installs requirements and opens the window
build_exe.bat            builds dist\IrisRecognition.exe with PyInstaller
iris_auth/
  segmentation.py        pupil and limbus (integro-differential operator), eyelids, lashes, reflections
  normalization.py       rubber-sheet model to a 64x512 polar image
  encoding.py            2D Gabor filters, phase quantization, 2,048-bit code and mask
  matching.py            masked Hamming distance, rotation shifts, binomial FAR model, 1:N threshold
  quality.py             image quality gate
  pad.py                 print/screen pattern detection, detail-loss check, pupil reaction check
  protection.py          Reed-Solomon and Hadamard codes, fuzzy commitment, cancelable template transform
  templatestore.py       AES-256-GCM encrypted store, scrypt key, persistent lockout state, revocation
  authenticator.py       enroll, verify, identify, key release, retry limit, hidden scores
  attacks.py             print, halftone and screen-replay simulation, pupil-reflex simulation, attack evaluation
  degradation.py         webcam-like degradation profiles and the visible/near-infrared-proxy comparison
  vault_eval.py          key-release and false-release measurement
  server.py, client.py   Flask API (challenge nonces, API keys, session tokens) and its client
  tls.py                 self-signed certificate generation and TLS contexts
  signing.py             HMAC signature that binds a captured image to its challenge
  netdemo.py             tcpdump capture of plain HTTP versus TLS and pcap analysis
  evaluation.py          FAR/FRR/EER and plots on a labelled dataset
  io_utils.py            image and dataset loading, visible-light preprocessing
  webcam.py              camera access, guide box, visible-light preprocessing
  gui.py, app_logic.py   window and the logic behind it
  cli.py                 commands
tests/                   unit tests (py -m unittest discover -s tests)
```

The unit tests need known inputs, so `tests/synthetic_eyes.py` renders computer-generated eye images used only as test fixtures. It is not imported by the program and not needed to run it.

## Mapping to the midterm report

- Localization, normalization, encoding, matching and decision follow section 3.2 (masks, rotation compensation, Hamming distance).
- FAR is modeled as the binomial tail with 249 degrees of freedom times the number of rotations, about 1 in 35 million at 0.32; the 1:N threshold tightens with database size (section 7.3).
- Section 7 defenses implemented: print and screen pattern detection, detail-loss check, pupil reaction, encrypted templates, cancelable templates with revocation, fuzzy commitment key release, accept or reject only (score hidden), retry limit of 5 failures (10 with PAD), image quality gate, HTTPS transport with single-use challenges and signed captures.

## Measured results (UPOL, 48 eyes, 144 photos, real photos only)

- Clean photos: EER 2.3% at threshold 0.427.
- Fuzzy commitment, 2 enrollment photos locked to the same key: the 140-bit key was released for 45 of 48 genuine probes (FRR 6.2%) and for 0 of 2,880 impostor probes; 84-bit and 196-bit keys gave the same figures.
- Simulated attacks (45 enrolled eyes, 135 attempts per attack type, made from a held-out photo of the victim). Accepted without attack detection: halftone print 74.8%, screen replay 92.6%, continuous-tone photo print 94.8%. Accepted with attack detection: 0.0%, 2.2%, 22.2%. Attack detection also rejected 5.9% of genuine photos. Adding the simulated pupil-reflex challenge brought all three to 0.0%, and genuine acceptance fell from 97.8% (no detection) to 86.7%.
- Webcam-like captures through the real webcam code path, visible light: EER 4.2% (good), 8.0% (typical), 41% with only 48% of captures usable (poor). The red-channel near-infrared proxy gave 2.8%, 9.4% and 45%.
- Network: tcpdump of plain HTTP shows the user id, API key, base64 JPEG and released secret in clear; the TLS 1.3 capture shows none of them (byte entropy 7.999 bits/byte).

## Limitations

- The attacks are simulated from digital photos (printer dots, screen grid, recapture blur and noise). No physical print, screen or contact lens was photographed, so the numbers show the defenses working against these models, not against a real fabricated eye. The continuous-tone photo print is the weak point of the passive detector (22.2% pass); the pupil-reflex challenge is what stops it, and that challenge was evaluated on a simulated pupil response because UPOL has no bright-light frames.
- The webcam results use UPOL photos pushed through a webcam model (distance, blur, noise, JPEG, uneven light, glare) and the program's real webcam preprocessing. No real webcam and no real volunteers were used. The "near-infrared proxy" is the red channel of the same colour photos; no infrared hardware was used, so real near-infrared accuracy is not measured here.
- The network demo runs the client and server on one machine and captures the loopback interface with tcpdump (the .pcap files open in Wireshark). It shows the effect of TLS on a sniffer; it is not a test across a real network or a virtual machine.
- The fuzzy commitment released the key for 93.8% of genuine probes in this visible-light dataset, lower than published near-infrared figures.
- The attack detection thresholds were tuned on the same simulated attacks and the same 144 photos (held-out halves gave 3-8% genuine rejection), so treat them as a starting point rather than an ISO/IEC 30107-3 result.
- The program cannot tell a real eye from a computer-generated or printed image except through the quality and attack-detection checks.
