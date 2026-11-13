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
- [ ] Tests pass (`cargo test`)
- [ ] Clippy is happy (`cargo clippy -- -D warnings`)
- [ ] Code is formatted (`cargo fmt`)
- [ ] Fuzz targets compile (`cargo check --manifest-path fuzz/Cargo.toml`)

## Development

```bash
# Build
cargo build --release

# Tests
cargo test --lib

# Linting
cargo clippy -- -D warnings

# Formatting
cargo fmt

# Fuzzing (install cargo-fuzz first)
cargo fuzz run preset --fuzz-dir fuzz
cargo fuzz run image --fuzz-dir fuzz
```

The fuzz targets cover TOML parsing and untrusted image decoding. They apply
dimension and allocation limits so a malicious corpus cannot turn fuzzing into
a denial-of-service test.

If you touch user-facing strings, see the i18n workflow below.

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
