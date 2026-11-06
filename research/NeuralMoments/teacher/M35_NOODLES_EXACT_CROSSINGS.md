# M35 · Noodles M32 的可挖域与运动二次曲面首次实交

输入为 M34 原生有理数可挖域 D′，46,679 个顶点、94,150 个三角面。相交运算逐字节使用操作数 `c3633c74…db99d622`。曲面生成与速度由 V7 的代数有限映射和二级固定法线球提供。系数和速度使用支点坐标系，输出顶点经 binary64 支点平移后按其精确 dyadic 值输入 CGAL。精确性对应此离散有向三角曲面。原生 V7 截断器另用支点系浮点顶点确定包围范围。该文件固定 SHA-256，生成时还逐个以分数验证设计顶点处于登记的物理坐标 AABB 中。[M34 原始实体构造](M34_NOODLES_PARTITION.md) · [逐调用回执、操作数摘要及计时](M35_NOODLES_EXACT_CROSSINGS.json)

| 实验 | 采样 | 原生裁刀 | 输出及来源 |
|---|---|---|---|
| 局部椭球横穿 | 基点及 c0、c1、c8 三方向，各用两步长和正负扰动，共 13 点 | 162 点、320 面 | 每点 243 点、482 面，217 面来自 D′、265 面来自裁刀 |
| 原始确定性 seed0 | 1 个不同的真实初值 | 双曲面，76 点、144 面，其中 50 面为截断封盖 | 6,948 点、14,092 面，12,414 面来自 D′、1,678 面来自曲面、0 面来自封盖 |

椭球中心取 D′ 上 binary64 面积最大的三角形编号 81227 的重心，半轴生成矩阵对角为 0.11、0.079、0.054，非零剪切为 0.013、−0.009。步骤由生成脚本与采样登记固定。三组原始原生系数先取 ±2⁻¹⁹ 与 ±2⁻²⁰ 的扰动，再经 `fromUnnormalized`。对全部 13 点，轻量椭球适配器与完整 `evaluateQuadricSourceMotion` 重建的全部顶点、4,860 项速度及有向面逐项相等。两种几何的二次系数均直接交给 V7 原生选边逻辑；两例保留分支都是交集。seed0 的 50 张封盖各自至少存在一个坐标轴将其与整个 D′ 的精确 AABB 严格分开。

## 裁剪缓存对照

对基点与同一 c0 方向的四次扰动，先用保守 dyadic 盒裁剪精确 D′，再对 1,326 面局部域复用同一布尔程序。逐个检查裁刀所有顶点严格在盒内、输出不含盒面。五份输出的精确有向三角形多重集及原始操作数来源面逐项相等。五次直接布尔用时合计 326.23 秒。裁剪初始化 99.13 秒，五次缓存布尔合计 3.03 秒，含初始化共 102.17 秒，观测比值 3.19。计时范围为本机并发负载下的原生布尔子进程；该范围不含构建、训练或整个 Newton 调用。这些数值来自一个固定局部工作集。PINO 收益需在相同裁剪基线下另行测量。

## 复算

构建要求与 M34 相同，使用 C++17、Eigen、GMP、MPFR、Boost.Thread、固定版本的 libigl 与 CGAL。非系统路径可通过 EIGEN_INCLUDE_DIR、CPLUS_INCLUDE_PATH、LIBRARY_PATH、LD_LIBRARY_PATH 提供。teacher 脚本需要 Python 3.10+ 及 NumPy。

```bash
p=research/NeuralMoments/teacher
out=/path/to/m35-run
build=/path/to/build
design=/path/to/m34/Dprime.txt
bash "$p/m34_build_boolean.sh" "$PWD" /path/to/libigl /path/to/cgal "$build"
bash "$p/m35_build_source.sh" "$PWD" /path/to/libigl /path/to/cgal "$build"
python "$p/m35_exact_crossing.py" --design "$design" --executable "$build/exact-input-boolean" \
    --source "$build/source" --output "$out"
# Select seed0.json from the packet or regenerate original SplitMix64 seed0 via the W0 runner.
python "$p/m35_general_crossing.py" --design "$design" --q "$out/seed0.json" \
    --csg "$build/exact-input-boolean" --source "$build/general-source" \
    --bounds "$out/design-bounds.txt" --output "$out/seed0"
python "$p/m35_replay_source.py" --source "$build/general-source" --bounds "$out/design-bounds.txt" \
    "$out/base" "$out"/c?_* "$out/seed0"
python "$p/m35_crop_control.py" --run --design "$design" --csg "$build/exact-input-boolean" --directory "$out"
python "$p/m35_verify_crossing.py" --directory "$out" --output "$out/replayed.json"
```

归档 `Spin_test2_M35_W2_exact_crossing_packets.tar.gz` 含 84 个文件，SHA-256 为 `9ce3ab218674744af3ced17e2cf04d720a2fd59dd53c858d696f01184b9f7267`。内含所有 14 份精确输出、源面、原生运动、裁刀、回执、逐项完整源映射复查，以及五组缓存对照。原始 seed0 捕获使用归档中的早期通用适配器；仓库版本另输出 Eigen 实际归一化后的 q。14 份归档样本的系数、几何、运动与原始索引都已逐项复查。时间与编译器相关的可执行程序摘要是本次运行记录。M35 研究后端采用精确输入 CGAL/libigl 布尔。Cherchi 教师准入、交叉样本的 Newton 结果、制造约束及端到端加速分别由其对应实验判断。
