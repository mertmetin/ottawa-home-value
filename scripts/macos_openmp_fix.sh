#!/usr/bin/env bash
# macOS only. PyTorch's wheel bundles its own libomp; LightGBM's wheel loads
# Homebrew's. Two OpenMP runtimes in one process segfault as soon as both are
# used. This points LightGBM at PyTorch's copy, so there is only one.
# Linux wheels share the system libgomp and don't need this.
#
#   scripts/macos_openmp_fix.sh            # uses .venv
#   scripts/macos_openmp_fix.sh /path/to/venv
set -euo pipefail
[ "$(uname)" = Darwin ] || { echo "not macOS: nothing to do"; exit 0; }
VENV=${1:-.venv}
SITE=$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
LGB="$SITE/lightgbm/lib/lib_lightgbm.dylib"
OMP="$SITE/torch/lib/libomp.dylib"
[ -f "$LGB" ] && [ -f "$OMP" ] || { echo "need lightgbm and torch installed in $VENV" >&2; exit 1; }
if otool -L "$LGB" | grep -q "$OMP"; then echo "already fixed"; exit 0; fi
install_name_tool -change @rpath/libomp.dylib "$OMP" "$LGB" 2>/dev/null
codesign --force --sign - "$LGB" 2>/dev/null
echo "LightGBM now uses $OMP"
