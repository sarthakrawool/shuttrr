import AppKit
import SwiftUI

/// Asked once before a model under its own licence downloads (Fish Audio Drama 3): the licence in plain points,
/// a link to the full text, and an "I agree" box. Nothing downloads until the box is ticked and Download pressed.
struct LicenceSheet: View {
    @EnvironmentObject private var store: EngineStore
    let model: CatalogModel
    @State private var agreed = false

    var body: some View {
        let lic = model.licence ?? CatalogLicence()
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: Theme.Space.s2) {
                Text("Before you download \(model.name)")
                    .font(Theme.Fonts.section)
                    .foregroundStyle(Theme.Palette.ink)
                Text("It is under the \(lic.name). In short:")
                    .font(Theme.Fonts.row)
                    .foregroundStyle(Theme.Palette.ink2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(Theme.Space.s5)
            VStack(alignment: .leading, spacing: Theme.Space.s3) {
                ForEach(lic.points, id: \.self) { p in
                    HStack(alignment: .firstTextBaseline, spacing: Theme.Space.s2) {
                        Text("•").foregroundStyle(Theme.Palette.muted)
                        Text(p)
                            .font(Theme.Fonts.row)
                            .foregroundStyle(Theme.Palette.ink)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                if let url = URL(string: lic.url), !lic.url.isEmpty {
                    Button("Read the full licence") { NSWorkspace.shared.open(url) }
                        .buttonStyle(.link)
                        .font(Theme.Fonts.row)
                }
                Toggle(isOn: $agreed) {
                    Text("I have read the licence and agree to it. I will only use voices of people who agreed.")
                        .font(Theme.Fonts.row)
                        .foregroundStyle(Theme.Palette.ink)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .toggleStyle(.checkbox)
                .padding(.top, Theme.Space.s2)
            }
            .padding(.horizontal, Theme.Space.s5)
            .padding(.bottom, Theme.Space.s5)
            Hairline()
            HStack {
                Spacer()
                Button("Cancel") { store.licenceAsk = nil }
                    .buttonStyle(.kit(.line))
                    .keyboardShortcut(.cancelAction)
                Button("Download") { store.acceptLicenceAndDownload(model.id) }
                    .buttonStyle(.kit(.ink))
                    .disabled(!agreed)
                    .keyboardShortcut(.defaultAction)
            }
            .padding(Theme.Space.s4)
        }
        .frame(width: 520)
        .background(Theme.Palette.content)
        .preferredColorScheme(.light)
    }
}
