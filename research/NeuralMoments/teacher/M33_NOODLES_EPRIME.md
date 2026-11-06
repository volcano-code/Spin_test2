# M33 精确的 Noodles 内缩包络

M32 登记的可挖域为 D′ = E′ 减 F，其中 E′ 是 9/10 等比内缩的包络，F 是 93 个原固定件的并集。本次交付单个 E′ 操作数。程序重建 94 个原始网格并核对各 SHA，从源包络的实际 binary64 值恢复精确分数，以 M32 登记中心作有理变换后写出 642 个顶点和 1,280 个原连接的有向三角面。共有 1,698/1,926 个内缩坐标值不能无损往返 binary64。保留分数可避免为后续贴边判定引入舍入误差。

文件为 [E′ 原生操作数](M33_NOODLES_EPRIME.txt)（82,487 字节），外部 SHA-256 为 `b68f810266a94e0c4e2489f2a1005ae0be1f7f321e9d9f092b047e374dd12133`。[回执](M33_NOODLES_EPRIME.json) 包含全部十个原始矩、来源和坐标系。对 E′ 表面积分与对 E 的矩施加仿射变换的结果逐项完全相等。E 减 E′ 的体积为 E 体积的 271/1000，绕登记 z 轴的惯量为 E 对同一轴惯量的 40951/100000。两项比值的分母均只包含包络 E。

W1 的 GMP 三角形精确相交程序在输出的字节上检查了 7,866/7,866 对候选（共边或共点 7,650，对无共同顶点 216），未发现非法相交。以下命令可在 W2 checkout 根目录重现。另备包含 25c5905977a69fcbbdd7e47557127f2f2ebb64d5 的 main 工作副本及 [W1 检查器提交](https://github.com/volcano-code/Spin_test2/commit/15db682052e9a97779c34ba02000095b78cf4b91) 的 Git 历史。需要 Python、NumPy、trimesh、C++17、Boost 头文件与 GMP 开发库。

```bash
p=research/NeuralMoments/teacher
work=$(mktemp -d)
python "$p/m33_noodles_inset_operand.py" --main-tree /path/to/Spin_test2-main --output "$work/Eprime.txt" > "$work/receipt.json"
cmp "$work/Eprime.txt" "$p/M33_NOODLES_EPRIME.txt"
cmp "$work/receipt.json" "$p/M33_NOODLES_EPRIME.json"
git show 15db682052e9a97779c34ba02000095b78cf4b91:research/NeuralMoments/performance/m26_exact_intersection.cpp > "$work/checker.cpp"
c++ -std=c++17 -O2 "$work/checker.cpp" -o "$work/checker" -lgmpxx -lgmp
"$work/checker" --max-seconds 60 < "$work/Eprime.txt"
```

[原生精确布尔程序](../../../experiments/faster_core_repair_20261012/exact_input_boolean.cpp) 按顺序读取两条操作数记录。将 E′ 文件与同格式的认证 F 记录直接拼接即可作为 difference 的双操作数输入。每条记录首行为顶点数和面数，随后为逐点的三列十进制整数分数 n/d 以及从零起算的面索引。旧 V7 的六份布尔输出在现有归档中有摘要；仍需取得并绑定 F 并集的真材，或从 93 个固定件以新运行号重建。原始外形 A、D′、F′ 的集合验收、整件参考惯量、平衡能力和原生 Newton 均待后续验证。
