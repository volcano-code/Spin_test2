# M32 的精确实体、平衡构造与首次减重结果

北京时间 2026 年 11 月 6 日晚至 7 日凌晨。对象为固定原轴、原尖端及原外形的 Noodles M32 工程输入。质量均相对于完整初始实体 A，惯量均相对于 A 绕登记的同一条 z 轴的惯量。

## 实体与求值

从已认证的 94 个源网格构造了 F（93 个原固定件的并集）、A（初始整件）、D′（可挖域）、F′（固定材料）。95 次原生精确布尔调用累计约 248 秒，属于一次性输入构造。W2 检查了 F′ 与 D′ 的正则交集、两者并集与 A 的双向差集，以及 F 被 F′ 完整保留；四个差异/重叠结果均为空。W1 独立完成四份网格的精确嵌入、朝向及有向边界核验。十个有理数质量矩逐项满足 A = F′ + D′。初始整件保留唯一最低尖端。 [W2 原生集合检查](https://github.com/volcano-code/Spin_test2/blob/2f9cd722fa024b0144a73809933743831d5fe6b3/research/NeuralMoments/teacher/M34_NOODLES_SET_CHECKS.json) · [W1 独立复算](https://github.com/volcano-code/Spin_test2/blob/e0ceffd2595f2adff27a97af0058c8531f6918e7/research/NeuralMoments/performance/M34_NOODLES_EXACT_PARTITION.md)

W3 在 D′ 的一个光滑平面切割点核验十矩和全部 100 个环境导数，三个原生方向、两种步长的中心差分最大绝对差约为 2.37×10⁻¹⁰。将 D′ 直接经过现有 Cherchi binary64 设计域入口时出现 1,877 个退化三角面，故该输入身份尚未放行。 [平面参考](https://github.com/volcano-code/Spin_test2/blob/bc6183b3c539a82567ffbbd13b6d6079d86a94a2/research/Noodles_M32/M32_PLANE_REFERENCE.md) · [坐标传输失败记录](https://github.com/volcano-code/Spin_test2/blob/1d088ccdaecc93646bce4da6bcde74a79803b393/research/NeuralMoments/teacher/M34_DPRIME_BINARY64_TRANSPORT.md)

对 W4 构造的满秩单椭球，W3 另外核验了所有参数在半径 2⁻³⁰ 的原生系数邻域内扰动时，空腔整体仍位于 D′ 内。连续椭球的解析十矩及其全部十列导数经独立面积求积和三步中心差分复测，最大中心差分绝对差约为 5.0×10⁻⁹。把取整后的参数按历史 Noodles 目标及尖端坐标复算，连续模型的初始乘子约为 −16.5414、原方程残差约为 5.02163；四项平衡条件可以满足，原始正乘子初值门槛仍未通过。该复算与全域离散 Cherchi 原生求解分别登记。 [满秩局部求值](https://github.com/volcano-code/Spin_test2/blob/05d5f255e3f32ffb7ec4d929638ef468ca14d375/research/Noodles_M32/M32_ELLIPSOID_LOCAL.md)

## 轻量化与单二次曲面

| 保留材料的构造 | 保留质量 | 保留轴向惯量 | 验收范围 |
|---|---:|---:|---|
| 严格内含的一个椭球空腔 | 99.97812% | 99.98343% | 真实单二次曲面共同可行；四项平衡等式精确为零，轴向主惯量占优。 |
| 6,879 个已认证单元上的最优高度 | 90.32951% | 91.96643% | 有理数原始/对偶证书，最优性限定在该离散族。6,811 个非空腔体，最近间隔约 0.09766 mm。 |
| 每轴每三个单元取一个的稀疏族 | 99.68279% | 99.76531% | 242 个候选、226 个非空腔体，该离散族内最优，腔体间距至少 1.66016 mm。 |

上述构造在相同固定外形、固定材料及均匀密度下达到原轴的横向质心和两项交叉惯量等式。两种单元族均独立核验了轴向主惯量占优；部分空腔高度约 0.02 mm。物理打印与实测旋转仍待验证。 [单椭球的构造和分离证明](https://github.com/volcano-code/Spin_test2/blob/83f37953594673bd347ef027cbb795e3968ed82e/research/Noodles_M32/M32_SINGLE_ELLIPSOID.md) · [受限质量优化与重放](https://github.com/volcano-code/Spin_test2/blob/8ec241fc1d5fded042c324f64838c4f9cd9bad1e/research/Noodles_M32/M32_CELL_MASS.md)

W0 另给出允许任意挖空时的保留质量下界。令 S 为从 A 移除的材料，单位密度下其轴向惯量最多为初始值的 10%。S 位于内缩包络高度 H 的水平层内。取任意正乘子 η，对每个点使用 1 ≤ ηr² + max(0,1−ηr²)。将第二项放宽至整层后积分为 πH/(2η)。以 η=5、π<22/7 并对初始质量和惯量作向外舍入，得到最终质量至少为 **67.1118586%**。当要求单二次曲面或附加连通条件时，该下界依然有效。它对任意挖空给出一侧界；有限单元族的 90.32951% 是可行质量上界。单椭球的可行质量上界为 99.97812%。 [数值证书](SPIN_TEST2_M32_MASS_BOUND.json) · [证明与复算](../NeuralMoments/contracts/m32_mass_bound.py)

## 学习器与比较状态

Spin-PINO 已迁入 Spin_test2 并保留目的分支梯度修复。六个解析模型、三个训练种子和使用相同留出旋转的测试已复现。新增解析测试采用此前未用的 24 个双椭球几何、48 个查询，共用相同样本和训练预算。原 W5 比较器在该数据上的 r/J 相对 L2 为 0.7077/0.5027。Logistic-atom 在相同主干下，默认节点加权为 0.3114/0.4110；已有占据率的有效密度只计权一次后为 0.1611/0.2866。显式物理损失开关几乎没有影响。该指标只描述此解析族。 [W6 迁移及新实验](https://github.com/volcano-code/Spin_test2/blob/9832a0ebac17f50ca20f02e274851df33fc8fa12/research/NeuralMoments/models/pcraft_ufno/spin_pino/SUPPORT_HOLDOUT.md)

W5 已预登记同起点和同预算的 A/B/C/D 比较，包括冷启动、几何缓存及 95%→90% 惯量阈值延续，先核验物理和目标质量，再评价耗时。配对试验预设每项每组每轮 600 秒及 120 次精确求值共用上限，包含诊断点、重启、校正和回退，原生 seed=0 的最多八个初值按相同顺序尝试。正式比较尚未运行，速度比为空；真实几何诊断数据与获准训练或评估的数据分别计数。 [配对协议](https://github.com/volcano-code/Spin_test2/blob/bea8dc6604ca5d71f686ac07f6bbfeb9e37d250a/research/NeuralMoments/models/deeponet/M32_PAIRED_PILOT.json)

单二次曲面可行证书已经取得。下一项是让与真实 D′ 边界相交的二次曲面完成保留有理输入身份的求值，核验其交线运动、十矩及导数；随后用共同初值运行原生优化，从最终整件独立复测。已认证的实物减重和真实同精度时间对照均待取得；当前原型复现计数仍为 0/5。
