# M34 · M32 的固定件、原始整件、可挖域与固定材料

本次从与 M32 相同的 94 份源 OBJ 重建实体。93 个固定件按登记顺序做 92 次平衡树精确并集得到 F，随后计算 A = E ∪ F、D′ = E′ 减 F、F′ = A 减 D′。E′ 逐字节使用 M33 的原生有理数操作数。全程保留输入精确分数和中间布尔结果；未做平滑、补洞、采样或二进制浮点网格往返。

| 实体 | 顶点数 | 三角面数 |
|---|---:|---:|
| 原固定件并集 F | 57,365 | 114,942 |
| 原始完整实体 A | 18,495 | 36,986 |
| M32 可挖域 D′ | 46,679 | 94,150 |
| M32 固定材料 F′ | 65,174 | 131,136 |

本次 95 次原生调用用时合计 248.23 秒，仅计精确输入构造。使用 [既有原生程序](../../../experiments/faster_core_repair_20261012/exact_input_boolean.cpp)、GCC 13.3.0、libigl `f06fb130`、CGAL `f7c3c821`，程序 SHA-256 为 `6d47d4aed9b5ae29f365eaae26a1fdd0c1d59829f1799d71a6546e1991adef81`。源坐标先按历史 OBJ 读取为 binary64，再转为其表示的精确有理数。所有后续中间顶点均直接以有理数传递。 [逐节点身份与输出摘要](M34_NOODLES_CONSTRUCTION.json) · [原登记壁厚](M32_NOODLES_INSET_WALL.json) · [E′ 输入证书](M33_NOODLES_EPRIME.md)

## 复算

使用包含 Noodles V7 归档的 Spin_test2 main 工作副本和本 W2 工作副本。构建需要 C++17、Eigen、GMP、MPFR、Boost.Thread 和上述两个版本的依赖源码。非系统头文件或库路径可通过编译器的 CPLUS_INCLUDE_PATH、LIBRARY_PATH 与 LD_LIBRARY_PATH 指定。几何生成另需 Python、NumPy 与 trimesh。

```bash
p=research/NeuralMoments/teacher
out=$(mktemp -d)
bash "$p/m34_build_boolean.sh" "$PWD" /path/to/libigl /path/to/cgal "$out/build"
python "$p/m34_noodles_partition.py" --main-tree /path/to/Spin_test2-main \
    --executable "$out/build/exact-input-boolean" --out "$out/solids" --set-checks
```

每个成功节点保存原生输出 JSON（顶点、面、来源面）、下次可直接输入的精确 n/d 文本和输入/输出回执。中断后用同一程序和目录重跑，缓存摘要必须一致。不同编译环境需新建结果目录；运行耗时会改变回执文件的字节。交付数据包 `Spin_test2_M34_M32_exact_solids.tar.gz` 的 SHA-256 为 `98dd033ed8679246ae477e7fe8157e0469f7a3a503ed9f4661e82fd6249d7511`；包内四个原生输出文件可按本次构造回执逐一复核。

本轮构造用于后续独立实体与物理验收。原始整件的指定轴惯量、四项约束的共同可行性、最终整件连通性、Newton 收敛及减重需各自验证。

## 正则实体集合复查

五项集合复查共执行六次原生调用。U 减 A 在并行调度时重复执行了一次，两次结果字节一致。将 F′ 和 D′ 重新求并得到 U，其精确有向三角网格输入字节与原始整件 A 完全一致，操作数 SHA-256 均为 `05bcae303ca7d7b2b478e1a9f66293ca07f9ef2cc275cda966ded34c62dc5e74`。F′ 与 D′ 的正则交集、A 减 U、U 减 A、原固定件 F 减 F′ 四项均返回零顶点及零三角面。两域的公共界面允许共享边界；交集结果支持零正体积重叠。该复查复用同一精确布尔后端，独立实现的有向边界与物理复核由 W1 验收。 [集合复查回执](M34_NOODLES_SET_CHECKS.json)
