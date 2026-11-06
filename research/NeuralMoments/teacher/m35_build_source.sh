#!/usr/bin/env bash
# Build the two native finite-map adapters. Use the same dependency trees as M34.
set -euo pipefail
if [ "$#" -ne 4 ]; then
    echo "usage: m35_build_source.sh W2_REPO LIBIGL_REPO CGAL_REPO BUILD_DIR" >&2
    exit 2
fi
root=$(realpath "$1")
igl=$(realpath "$2")
cgal=$(realpath "$3")
out=$(realpath -m "$4")
test "$(git -C "$igl" rev-parse HEAD)" = f06fb13014aadff7eb1d7423ac018620441a3657
test "$(git -C "$cgal" rev-parse HEAD)" = f7c3c8212b56c0d6dae63787efc99093f4383415
mkdir -p "$out"
common=(-std=c++17 -g0 -ffunction-sections -fdata-sections -DCGAL_HEADER_ONLY
        -I"$root/include" -I"$root/src/spin" -isystem "${EIGEN_INCLUDE_DIR:-/usr/include/eigen3}")
for h in "$cgal"/*/include; do common+=(-isystem "$h"); done
src="$root/research/NeuralMoments/teacher"
"${CXX:-c++}" "${common[@]}" -O1 -Wl,--gc-sections "$src/m35_ellipsoid_source.cpp" \
    "$root"/src/spin/{QuadricInverseGaussMap,QuadricCoefficients}.cpp -lgmp -lmpfr -o "$out/source"
# Native represented finite-map mode; rational D' is sent separately to M34 exact-input-boolean.
sources=("$src/m35_general_source.cpp" "$root"/src/spin/{QuadricSourceMotion,QuadricTriangulator,QuadricInverseGaussMap,QuadricCoefficients,TruncationRootCertificate,QuadricStratumTangent,QuadricMomentEvaluator}.cpp
         "$root/src/manufacturing/SlicerValidation.cpp" "$root/src/physics/MassProperties.cpp")
objects=()
for f in "${sources[@]}"; do
    obj="$out/$(basename "$f" .cpp).o"
    "${CXX:-c++}" "${common[@]}" -O0 -DMAKE_IT_STAND_HAS_EXACT_GEOMETRY_CHECKS=0 -I"$igl/include" -c "$f" -o "$obj"
    objects+=("$obj")
done
"${CXX:-c++}" -Wl,--gc-sections "${objects[@]}" -lgmp -lmpfr -lboost_thread -o "$out/general-source"
