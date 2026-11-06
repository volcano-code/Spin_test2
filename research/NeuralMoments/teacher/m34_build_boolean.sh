#!/usr/bin/env bash
# Build the W2 offline verifier with the registered libigl and CGAL trees.
set -euo pipefail
if [ "$#" -ne 4 ]; then
  echo "usage: bash m34_build_boolean.sh W2_REPO LIBIGL_REPO CGAL_REPO BUILD_DIR" >&2
  exit 2
fi
root=$(realpath "$1")
igl=$(realpath "$2")
cgal=$(realpath "$3")
out=$(realpath -m "$4")
test "$(git -C "$igl" rev-parse HEAD)" = f06fb13014aadff7eb1d7423ac018620441a3657
test "$(git -C "$cgal" rev-parse HEAD)" = f7c3c8212b56c0d6dae63787efc99093f4383415
mkdir -p "$out"
# Compile only the stock transport parser, without linking the unrelated
# boundary auditor or replacing its strict n/d validation by fractional floats.
python - "$root/src/verification/ExactBoundaryAudit.cpp" "$out/parse.cpp" <<'PY'
from pathlib import Path
import sys
s = Path(sys.argv[1]).read_text()
a = s.index('CGAL::Gmpq parseExactRationalCoordinate')
b = s.index('\n}', a) + 2
Path(sys.argv[2]).write_text('#include "verification/ExactRationalCoordinates.h"\n#include <algorithm>\n'
                            '#include <cctype>\n#include <stdexcept>\nnamespace verification {\n'+s[a:b]+'\n}\n')
PY
includes=()
for h in "$cgal"/*/include; do includes+=(-isystem "$h"); done
"${CXX:-c++}" -std=c++17 -O1 -g0 -DCGAL_HEADER_ONLY \
    -I"$root/include" -I"$igl/include" -isystem "${EIGEN_INCLUDE_DIR:-/usr/include/eigen3}" \
    "${includes[@]}" "$root/experiments/faster_core_repair_20261012/exact_input_boolean.cpp" \
    "$out/parse.cpp" -lgmp -lmpfr -lboost_thread -o "$out/exact-input-boolean"
