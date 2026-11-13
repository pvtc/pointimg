#!/usr/bin/env python3
"""Sync locales/en/LC_MESSAGES/pointimg.po with the gettext() calls in src/.

- Extracts msgids from plain and raw Rust string literals (handles the
  common escapes and the backslash-newline continuation).
- Keeps the existing English translation for msgids still in use, drops
  obsolete entries, appends new ones with an empty msgstr.
- Recompiles pointimg.mo with msgfmt when available, otherwise with a
  minimal built-in writer.
- Exits nonzero when a msgid still has no English translation, so CI can
  gate on it.
"""
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
PO = ROOT / "locales" / "en" / "LC_MESSAGES" / "pointimg.po"
MO = PO.with_suffix(".mo")

# Matches gettext(  "  or  gettext(  r#"  -- the opening quote of the literal.
CALL = re.compile(r'gettext\(\s*(r)?"')

ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "0": "\0"}


def read_literal(text, i, raw):
    """Parse a Rust string literal starting at the quote index.

    Returns (value, index just past the closing quote).
    """
    out = []
    i += 1
    while i < len(text):
        c = text[i]
        if c == '"':
            return "".join(out), i + 1
        if c == "\\" and not raw:
            nxt = text[i + 1]
            if nxt == "\n":
                # Line continuation: skip the newline and leading whitespace.
                i += 2
                while i < len(text) and text[i] in " \t":
                    i += 1
                continue
            out.append(ESCAPES.get(nxt, "\\" + nxt))
            i += 2
            continue
        out.append(c)
        i += 1
    raise ValueError("unterminated string literal")


def extract_msgids():
    """Collect every literal passed to gettext(), with source references."""
    msgids = {}  # msgid -> set of "path:line"
    for path in sorted(SRC.rglob("*.rs")):
        text = path.read_text(encoding="utf-8")
        for m in CALL.finditer(text):
            try:
                value, _ = read_literal(text, m.end() - 1, bool(m.group(1)))
            except ValueError as e:
                line = text.count("\n", 0, m.start()) + 1
                sys.exit(f"{path}:{line}: {e}")
            ref = f"{path.relative_to(ROOT)}:{text.count(chr(10), 0, m.start()) + 1}"
            msgids.setdefault(value, set()).add(ref)
    return msgids


def po_parse(path):
    """Parse a .po file into an ordered dict-like {msgid: msgstr}."""
    entries, order = {}, []
    msgid = msgstr = None
    state = None
    for line in path.read_text(encoding="utf-8").splitlines() + [""]:
        if line.startswith("msgid "):
            if msgid is not None and msgid != "":
                entries[msgid] = msgstr or ""
                order.append(msgid)
            msgid, state = po_unquote(line[6:]), "id"
            msgstr = ""
        elif line.startswith("msgstr "):
            msgstr, state = po_unquote(line[7:]), "str"
        elif line.startswith('"') and state:
            value = po_unquote(line.strip())
            if state == "id":
                msgid += value
            else:
                msgstr += value
    if msgid:
        entries[msgid] = msgstr or ""
        order.append(msgid)
    return entries


def po_unquote(s):
    return read_literal(s.strip() + '"', 0, False)[0]


def po_quote(s):
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    if "\n" in escaped:
        parts = escaped.split("\n")
        return '""\n' + "\n".join(f'"{p}\\n"' for p in parts[:-1]) + (
            f'\n"{parts[-1]}"' if parts[-1] else "")
    return f'"{escaped}"'


def compile_mo(entries, path):
    if shutil.which("msgfmt"):
        subprocess.run(["msgfmt", "--check", "-o", str(path), str(PO)], check=True)
        return "msgfmt"
    items = sorted((k, v or k) for k, v in entries.items())
    keys = b"".join(k.encode() + b"\0" for k, _ in items)
    vals = b"".join(v.encode() + b"\0" for _, v in items)
    n = len(items)
    koff, voff = 28 + 16 * n, 28 + 16 * n + len(keys)
    with open(path, "wb") as f:
        f.write(struct.pack("<7I", 0x950412DE, 0, n, 28, 28 + 8 * n, 0, 0))
        off = koff
        for k, _ in items:
            b = k.encode()
            f.write(struct.pack("<2I", len(b), off))
            off += len(b) + 1
        off = voff
        for _, v in items:
            b = v.encode()
            f.write(struct.pack("<2I", len(b), off))
            off += len(b) + 1
        f.write(keys)
        f.write(vals)
    return "built-in writer"


def main():
    msgids = extract_msgids()
    old = po_parse(PO) if PO.exists() else {}
    missing = sorted(m for m in msgids if not old.get(m))
    dropped = sorted(set(old) - set(msgids))

    lines = [
        'msgid ""',
        'msgstr ""',
        r'"Project-Id-Version: pointimg\n"',
        r'"PO-Revision-Date: 2026-11-13 00:00+0000\n"',
        r'"Last-Translator: pointimg contributors\n"',
        r'"Language-Team: English\n"',
        r'"MIME-Version: 1.0\n"',
        r'"Content-Type: text/plain; charset=UTF-8\n"',
        r'"Content-Transfer-Encoding: 8bit\n"',
        r'"Language: en\n"',
        r'"Plural-Forms: nplurals=2; plural=(n != 1);\n"',
        "",
    ]
    for msgid in sorted(msgids, key=str.lower):
        for ref in sorted(msgids[msgid]):
            lines.append(f"#: {ref}")
        lines.append(f"msgid {po_quote(msgid)}")
        lines.append(f"msgstr {po_quote(old.get(msgid, ''))}")
        lines.append("")
    PO.write_text("\n".join(lines) + "\n", encoding="utf-8")
    how = compile_mo({m: old.get(m, "") for m in msgids}, MO)
    print(
        f"{PO.relative_to(ROOT)}: {len(msgids)} msgids "
        f"({len(msgids) - len(missing)} translated kept, "
        f"{len(missing)} new, {len(dropped)} dropped)"
    )
    print(f"{MO.relative_to(ROOT)}: compiled with {how}")
    if missing:
        print(f"\n{len(missing)} msgid(s) need an English msgstr:", file=sys.stderr)
        for m in missing:
            print(f"  {m!r}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
