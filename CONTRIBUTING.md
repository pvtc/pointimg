# Contributing to pointimg

Thank you for your interest in this project!

## Reporting Bugs

Use [GitHub Issues](https://github.com/pvtc/pointimg/issues) with the "Bug Report" template.

## Suggesting Features

Open an issue with the "Feature Request" template.

## Pull Requests

1. Fork the repository
2. Create a branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Before Submitting

- [ ] Code compiles (`cargo build`)
- [ ] Tests pass (`cargo test --features gui,gpu,avif`)
- [ ] Clippy is happy (`cargo clippy --features gui,gpu,avif --all-targets -- -D warnings`)
- [ ] Code is formatted (`cargo fmt`)
- [ ] Fuzz targets compile (`cargo check --manifest-path fuzz/Cargo.toml`)
- [ ] Shipped metadata validates (`desktop-file-validate`, `appstreamcli`, `flatpak-builder --show-manifest`)

## Development

```bash
# Build the CLI (the default: no GUI dependency)
cargo build --release

# Build the egui GUI
cargo build --release --features gui --bin pointimg-gui

# Tests (the default covers the library and CLI; the features add the GUIs)
cargo test
cargo test --features gui,gpu,avif

# Linting
cargo clippy --features gui,gpu,avif --all-targets -- -D warnings

# Formatting
cargo fmt

# Shipped metadata
desktop-file-validate packaging/*.desktop
appstreamcli validate --no-net packaging/*.metainfo.xml
flatpak-builder --show-manifest packaging/flatpak/io.github.pvtc.pointimg-gtk.yml > /dev/null

# Fuzzing (install cargo-fuzz first)
cargo fuzz run preset --fuzz-dir fuzz
cargo fuzz run image --fuzz-dir fuzz
```

The fuzz targets cover TOML parsing and untrusted image decoding. They apply
dimension and allocation limits so a malicious corpus cannot turn fuzzing into
a denial-of-service test.

If you touch user-facing strings, see the i18n workflow below.

## Dependencies

- **MSRV**: Rust 1.100 (`rust-version` in `Cargo.toml`), checked by the `msrv`
  CI job with `cargo +1.100`.
- **Cooldown**: `.cargo/config.toml` makes the resolver ignore releases
  published less than 3 days ago (`registry.global-min-publish-age` plus
  `resolver.incompatible-publish-age = "deny"`). A compromised release is
  almost always yanked within hours. Versions already in `Cargo.lock` stay
  valid, but a blanket `cargo update` will therefore pick versions at least
  3 days old; target one crate with `cargo update -p <crate>` when you need a
  fresher release. For an urgent fix, or a release you already tested by hand,
  override the limit for one command:

  ```bash
  CARGO_RESOLVER_INCOMPATIBLE_PUBLISH_AGE=allow cargo update -p <crate> --precise <version>
  ```

  Dependabot applies the same 3-day delay to version updates; **security
  updates are never delayed**.
- **Unused dependencies**: `[lints.cargo] unused_dependencies = "deny"` fails
  the build when a declared dependency is no longer used.

## Translations (i18n)

User-facing strings are French msgids passed through `pointimg::frontend::gettext`.
The English catalogue lives in `locales/en/LC_MESSAGES/pointimg.po` and is
regenerated with:

```bash
sh scripts/update-i18n.sh
```

The script extracts every `gettext("...")` call site, merges new msgids into the
`.po`, drops obsolete ones and rebuilds the embedded `.mo`. Fill in the new
`msgstr` entries between two runs. The CI fails if the committed catalogue is
not in sync with the code.

## Questions?

Open a GitHub Discussion or contact the maintainers.
