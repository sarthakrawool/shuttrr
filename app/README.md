# Shutter for macOS

A menu bar app and dashboard for the Shutter engine (the Python daemon on
`http://127.0.0.1:8484`). Swift Package, SwiftUI + AppKit, macOS 14+. One dependency: Sparkle 2 for in-app updates (`Updater.swift`); releases are made with `scripts/release.sh`.

## Build

```sh
chmod +x app/scripts/build_app.sh
app/scripts/build_app.sh          # swift build -c release, then app/build/Shutter.app
open app/build/Shutter.app
```

For development, `cd app/Shutter && swift run` also works. Launch at login only
works from the bundled app.

The app runs as a menu bar agent (`LSUIElement`, set in the generated Info.plist). The
engine is started from the panel (`Start Engine`) with the command set in Settings
(default `~/.shutter/venv/bin/shutter serve`), or you can start it yourself. The app never
stops an engine it did not start.

## Files

| File | What it does |
| --- | --- |
| `Shutter/Package.swift` | Package with one executable target, macOS 14. |
| `Sources/Shutter/ShutterApp.swift` | `@main` app: `MenuBarExtra` (window style) + `Window` "Shutter" (dashboard); `AppDelegate` starts polling and asks on quit whether to keep a busy engine running. |
| `Models.swift` | Codable types for `/v1/system`, `/v1/models`, jobs, events. Every field is optional with a default. |
| `APIClient.swift` | Async URLSession client. Base URL from UserDefaults key `engineURL`. |
| `EngineStore.swift` | `@MainActor` store: polls every 2 s, backs off to 30 s while the engine is down, follows `/v1/events` (SSE), holds the memory history and actions (submit, cancel, unload, pause). |
| `EngineProcess.swift` | Starts/stops the daemon as a child `Process` via a login shell; output goes to `~/Library/Logs/Shutter/engine.log`, and the last 200 lines are kept. |
| `Formatting.swift` | Bytes as GB (one decimal), durations, percentages. |
| `Views/MenuPanelView.swift` | The menu bar panel: memory gauge, protected-service line, current job, queue counts, buttons. |
| `Views/MemoryGaugeView.swift` | Used vs total memory bar (a headroom line only when the user set one). |
| `Views/CurrentJobView.swift` | Running job: progress, stage, ETA; or why it waits for memory. |
| `Views/StateStyle.swift` | State colours/symbols (accent = running, orange = waiting for memory, red = failed). |
| `Views/DashboardView.swift` | Main window with sidebar: Queue, Models, Memory, Settings. |
| `Views/QueueView.swift` | Jobs table (state, model, framing, progress, duration, peak memory, output, cancel). |
| `Views/NewJobSheet.swift` | New job form: model, prompt, image/audio pickers, framing, seed, duration. |
| `Views/ModelsView.swift` | Registry models from `/v1/models`. |
| `Views/MemoryView.swift` | Swift Charts rolling chart of available memory and swap; loaded components with Unload. |
| `Views/SettingsView.swift` | Engine URL, daemon command, launch at login (`SMAppService`), engine log. |
| `scripts/build_app.sh` | Builds, assembles `Shutter.app`, writes Info.plist, ad-hoc signs. |

## What the app expects from the engine beyond the documented API

- `job.result.output` (path) and a peak memory number in `job.result.peak_bytes`
  (also accepted: `peak_memory_bytes`, `peak_memory`, `peak`).
- Optional `eta_s` on a job or in `job.progress` event data; otherwise the ETA is
  extrapolated from progress and elapsed time.
- `job.progress` events with `data.progress` and `data.stage`.
- Optional queue pause: `queue.paused` (bool) in `/v1/system`, plus
  `POST /v1/queue/pause` and `POST /v1/queue/resume`. The button is shown only when
  `paused` is present.
