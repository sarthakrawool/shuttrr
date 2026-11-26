import AVFoundation
import SwiftUI

/// Plays a finished voice request's WAV in the app (one at a time). Shown as a play button on audio rows.
@MainActor
final class AudioPlayer: NSObject, ObservableObject, AVAudioPlayerDelegate {
    static let shared = AudioPlayer()
    @Published private(set) var playing: String?
    private var player: AVAudioPlayer?

    func toggle(_ path: String) {
        if playing == path {
            stop()
            return
        }
        stop()
        guard let p = try? AVAudioPlayer(contentsOf: URL(fileURLWithPath: path)) else { return }
        p.delegate = self
        p.play()
        player = p
        playing = path
    }

    func stop() {
        player?.stop()
        player = nil
        playing = nil
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in self.playing = nil; self.player = nil }
    }
}

extension Job {
    /// Every audio file this request wrote (the first take, then the others), for voice models.
    var audioOutputs: [String] {
        let list = (result["outputs"]?.arrayValue ?? []).compactMap(\.stringValue)
        let all = list.isEmpty ? [outputPath].compactMap { $0 } : list
        return all.filter { ["wav", "mp3", "m4a", "aiff", "flac"].contains(URL(fileURLWithPath: $0).pathExtension.lowercased()) }
    }
}

/// Play / stop for one file; a menu of takes when a request made several.
struct PlayButton: View {
    @ObservedObject var player = AudioPlayer.shared
    let files: [String]

    var body: some View {
        if files.count > 1 {
            Menu {
                ForEach(Array(files.enumerated()), id: \.offset) { i, f in
                    Button((player.playing == f ? "Stop take " : "Play take ") + "\(i + 1)") { player.toggle(f) }
                }
            } label: {
                Image(systemName: files.contains(where: { player.playing == $0 }) ? "stop.fill" : "play.fill")
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .help("Play one of the \(files.count) takes")
        } else if let f = files.first {
            Button { player.toggle(f) } label: {
                Image(systemName: player.playing == f ? "stop.fill" : "play.fill")
            }
            .buttonStyle(.kit(.line, small: true))
            .help(player.playing == f ? "Stop" : "Play")
            .accessibilityLabel(player.playing == f ? "Stop" : "Play")
        }
    }
}
