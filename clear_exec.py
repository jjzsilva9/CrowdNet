#!/usr/bin/env python3
"""Clear the executable-stack flag (PF_X in PT_GNU_STACK) on ELF64 shared objects.
A dependency-free stand-in for `execstack -c` when execstack/patchelf aren't available.

Usage:
    python clear_execstack.py <file1.so> [file2.so ...]
    python clear_execstack.py --dry <file.so>     # report only, no changes

Stdlib only. Edits files in place; touches only the single flag bit.
"""
import sys, struct

PT_GNU_STACK = 0x6474e551
PF_X = 0x1

def clear(path, dry=False):
    with open(path, "rb") as f:
        data = bytearray(f.read())
    if data[:4] != b"\x7fELF":
        return f"skip (not ELF): {path}"
    if data[4] != 2:
        return f"skip (not ELF64): {path}"
    # ELF64 header: e_phoff @0x20 (Q), e_phentsize @0x36 (H), e_phnum @0x38 (H)
    e_phoff     = struct.unpack_from("<Q", data, 0x20)[0]
    e_phentsize = struct.unpack_from("<H", data, 0x36)[0]
    e_phnum     = struct.unpack_from("<H", data, 0x38)[0]
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type = struct.unpack_from("<I", data, off)[0]        # ELF64: p_type @+0
        if p_type == PT_GNU_STACK:
            p_flags = struct.unpack_from("<I", data, off + 4)[0]  # ELF64: p_flags @+4
            if p_flags & PF_X:
                new = p_flags & ~PF_X
                if not dry:
                    struct.pack_into("<I", data, off + 4, new)
                    with open(path, "wb") as f:
                        f.write(data)
                return f"{path}: GNU_STACK {p_flags:#x} -> {new:#x}" + ("  (dry)" if dry else "")
            return f"{path}: already non-exec ({p_flags:#x})"
    return f"{path}: no GNU_STACK header found"

if __name__ == "__main__":
    dry = "--dry" in sys.argv
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not files:
        print(__doc__); sys.exit(1)
    for p in files:
        print(clear(p, dry=dry))