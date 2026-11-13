#!/usr/bin/env sh
# Regenerate the English gettext catalogue from the Rust sources.
# Equivalent to: python3 scripts/sync-i18n.py
set -eu
cd "$(dirname "$0")/.."
exec python3 scripts/sync-i18n.py "$@"
