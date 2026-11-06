#!/usr/bin/env bash
# External headers and linker paths can be provided through CPATH and LIBRARY_PATH.
# Usage: EIGEN_INCLUDE=/path/to/eigen3 ./build_m35_native_kernel.sh /tmp/m35-native
set -euo pipefail
out=${1:?build directory required}
mkdir -p "$out"
contracts=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$contracts" rev-parse --show-toplevel)
compiler=${CXX:-c++}
flags=(-O1 -g0 -std=c++17 -ffunction-sections -fdata-sections -DCGAL_HEADER_ONLY "-I$repo/include" "-I${EIGEN_INCLUDE:-/usr/include/eigen3}")
sources=(spin/QuadricHollowingSolver spin/SpinMomentProblem spin/DualMonomialCoordinates spin/QuadricCoefficients spin/LegacyPredictionTransaction spin/SpinDesignRequest manufacturing/SlicerValidation)
objects=()
for src in "${sources[@]}"; do
    object="$out/${src##*/}.o"
    "$compiler" "${flags[@]}" -c "$repo/src/$src.cpp" -o "$object"
    objects+=("$object")
done
"$compiler" "${flags[@]}" -c "$contracts/m35_native_kernel.cpp" -o "$out/driver.o"
"$compiler" -Wl,--gc-sections "$out/driver.o" "${objects[@]}" -lgmp -lmpfr -o "$out/kernel"
