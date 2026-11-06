# M32 精确域分割与质量、惯量验收

W2 从 93 个固定件构造 F，再以 E 和线性内缩至 9/10 的 E′ 构造 A = E ∪ F、D′ = E′ 减 F、F′ = A 减 D′。W1 对这些新生成的有理数网格单独积分并复核边界。计算以单位密度和原模型坐标为准，一单位等于 25 mm，指定轴为经过固定尖端的 z 轴。各份结果绑定相应输入网格的 SHA-256。

| 边界 | 三角面 | 闭合且互不相交的曲面分量 |
|---|---:|---:|
| 初始整件 A | 36,986 | 1 |
| 原固定域 F | 114,942 | 1 |
| 新可挖域 D′ | 94,150 | 1 |
| 新固定域 F′ | 131,136 | 2 |

GMP 检查每张边界的面朝向、边流形与顶点连通、面积和潜在自交。每个闭合分量面向外法向一侧的总绕数为 0，跨入材料侧的绕数为 1。两份域的有向表面相加恰好等于初始整件的表面，精确消去后剩余三角面数为 0。有限支撑实体的有向边界决定其内部绕数，与 0/1 占用共同验证内部无重叠且完整覆盖。全部十个原始矩也由有理四面体分别积分并逐项验证了可加性。

## 约束与质量范围

横向一阶矩由固定材料和可保留设计材料共同决定。W1 在 x 和 y 两个平面分别精确裁剪边界，对两侧积分并与完整设计域矩核对。两侧独立选择允许的总一阶矩区间在 x 方向约为 [−0.127960, 0.128346]，在 y 方向约为 [−0.109590, 0.109485]。四项等式的共同构造另行检验。

在实际 A 内移除九个互不接触的精确长方体。每个盒的外包立方体与 D′ 的三角面严格分离，中心的精确绕数为 1。四个体积由新 A 的横向一阶矩和 xz、yz 交叉矩通过有理消元确定；其余五个体积固定。保留体的四个对应矩严格为零，指定轴惯量约为原件的 99.9746864%，质量约为 99.9857768%。

还可移除关于 z 轴成对对称的立方体，使四项等式继续成立。逐个排除可能触及边界三角面的完整网格单元，再以精确绕数判定各个剩余连通区域的内外。使用一个有限网格扫描，得到 6,879 个完整的设计域内部单元；排除九个原盒及其缓冲区，取其中 1,720 对。所得 3,449 个空腔的保留质量约为 95.09997%，轴向惯量约为 95.04532%。初始域有单个连通的实体内部；去除有限多个严格位于内部并保持正间隙的闭合盒后仍可绕盒连通。相邻附加立方体面间隙可小至 0.09765625 mm，内部薄结构、封闭空腔的制造可达性、强度和实物旋转均待验证。

由 D′ 的最大轴距平方和固定域惯量还可推出简单的下界。达到 90% 惯量所需的质量至少为固定质量加上正的惯量缺口除以设计域的最大轴距平方。对 M32 等于初始整件质量的约 61.19938%。这是一条偏保守的界；上述构造仅给出可行质量上界。单二次曲面、原生 Newton 最优性及实测成本另行验证。

## 复算

W2 M34 的四个精确输出作为输入。原始矩按 mass、x、y、z、xx、yy、zz、yz、xz、xy 排列。独立积分所需的大整数以十六进制分数保存在科学证据包。建议 Python 3.12、NumPy、SciPy、gmpy2 2.3；嵌入检查还需 C++17、GMP 和 Boost 头文件。以下命令在本目录执行。

```bash
c++ -O2 -std=c++17 m25_independent_gmp_tetra.cpp -lgmpxx -lgmp -o w1-ten
c++ -O2 -std=c++17 m26_exact_intersection.cpp -lgmpxx -lgmp -o w1-embedding
for part in A F Dprime Fprime; do
  python m34_noodles_partition.py mesh "$INPUT_DIR" "$AUDIT_DIR" "$part" --ten ./w1-ten --embedding ./w1-embedding --seconds 1500
done
python m34_noodles_partition.py collect "$INPUT_DIR" "$AUDIT_DIR" > "$AUDIT_DIR/partition.json"
python m34_noodles_cuboid_witness.py "$INPUT_DIR" "$AUDIT_DIR" --volume-packet "$AUDIT_DIR/nine-cuboid-volumes.hex" > "$AUDIT_DIR/witness-final.json"
python m34_noodles_symmetric_carve.py "$INPUT_DIR" "$AUDIT_DIR" "$AUDIT_DIR/witness-final.json" --pair-centers "$AUDIT_DIR/lattice-pairs.json" --eligible "$AUDIT_DIR/all-eligible-cells.txt" > "$AUDIT_DIR/lighter-final-v2.json"
python -m unittest test_m34_noodles_partition -q
```

[分割与矩结果](evidence/m34-noodles-partition.json) · [九盒共同可行构造](evidence/m34-noodles-nine-cuboids.json) · [对称减重结果](evidence/m34-noodles-symmetric-cubes.json)
