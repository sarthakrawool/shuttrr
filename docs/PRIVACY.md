# Privacy and "Help The Dev Improve"

Shuttrr runs models on your Mac. It goes online for three things only:

1. downloading the models you choose (Hugging Face);
2. checking GitHub for a new Shuttrr version (Settings > Check for updates);
3. **Help The Dev Improve**, only if you turn it on.

## Help The Dev Improve (off by default)

You are asked once: in the first-run window, or (if you set up Shuttrr before 0.3.3) in a small window after the
update. The switch starts off. You can change it any time in Settings > Privacy.

When it is on, the engine (`shutter/core/telemetry.py`) sends a report to
`https://telemetry.shuttrr.sarthakrawool.com/v1/report`:

| What | When |
| --- | --- |
| For each model: requests and finished requests (catalog id, or "custom" for any other model) | at most once a week |
| Crashes and failures: engine crashes (it stopped without shutting down), failed requests, Shuttrr app crashes (from macOS crash reports): error type, short message, stack trace with Shuttrr's module names | at most once an hour |
| Shuttrr and engine version, macOS major version, chip family (e.g. M5), memory size rounded to a Mac size (e.g. 16 GB) | with each report |

Before anything is kept, every message and stack trace is cleaned: file paths, your home folder, your user name,
file names, e-mail addresses and long quoted text are replaced (`<path>`, `<file>`, `<user>`, `<email>`, `<text>`).
Prompts are never part of a report.

Never sent: prompts, pictures, videos, audio, file or folder names, model paths, your Mac's name or serial number,
your accounts, or any identifier. There is no install ID, so reports cannot be linked to each other or to a Mac.
The receiving Worker keeps only the fields above; it does not store IP addresses or request headers.

Turning it off stops collection at once and deletes anything collected but not yet sent. Settings > Privacy >
See What's Sent shows the last report exactly as it was sent. Nothing is sent while it is off (tested in
`tests/test_telemetry.py`).

API: `GET /v1/telemetry`, `POST /v1/telemetry {"enabled": true|false}`, `POST /v1/telemetry/app-crash`. The Rust
`shutterd` engine does not have these routes (the app hides the setting there).
