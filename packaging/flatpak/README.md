# Flatpak

The GTK4 front-end is packaged as a Flatpak. This is the portable Linux
distribution: it bundles GTK 4.14+ / libadwaita 1.5+ and does not depend on the
host distribution's glibc or GTK version.

## Build and install

From the repository root:

```bash
flatpak-builder --user --install --force-clean build-dir \
    packaging/flatpak/io.github.pvtc.pointimg-gtk.yml
flatpak run io.github.pvtc.pointimg-gtk photo.jpg
```

Build requirements: `flatpak`, `flatpak-builder`, the `org.gnome.Platform//48`
and `org.gnome.Sdk//48` runtimes, and the `rust-stable` SDK extension.

## Regenerating the vendored Rust sources

Flatpak builds are offline. Every crate from `Cargo.lock` is listed in
`cargo-sources.json`, generated from the official tool:

```bash
curl -fsSLO https://raw.githubusercontent.com/flatpak/flatpak-builder-tools/master/cargo/flatpak-cargo-generator.py
python3 -m pip install --user aiohttp toml   # only needed once
python3 flatpak-cargo-generator.py Cargo.lock \
    -o packaging/flatpak/cargo-sources.json
```

`cargo-sources.json` must be regenerated whenever `Cargo.lock` changes; a stale
file fails the Flatpak build with a missing crate, it does not ship a wrong
binary.

## Notes

- The manifest pins `runtime-version: "48"`. Move it together with the SDK and
  the `rust-stable` extension when a newer GNOME runtime is adopted.
- `--filesystem=home` is deliberate: the recent-files list and drag & drop need
  direct access to the user's files. The file chooser itself also works through
  the document portal.
- The `.desktop`, AppStream metadata and hicolor icons installed here are the
  same files as the tarball, so the app appears in software centres with its
  screenshot and translations.
