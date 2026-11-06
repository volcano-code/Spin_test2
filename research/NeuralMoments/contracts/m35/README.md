# M35 · M32 Noodles 上的原生 Newton 与 Spin-PINO 对照

目标是在固定外壳、陀尖和旋转轴的单材料设计中，用 Spin-It Faster 的二次曲面方法完成 Spin-It 的任务。项目另要求保留至少 90% 的原始轴向惯量，并尽量降低整模质量和体积。

## 真实切割和求解

输入来自 M32 的精确 D′ 和 F′。求解器使用本仓库的 C++ `solveSpinMomentSystem`、原始初值生成器、`DualMonomialRawV1` 和原线搜索。每次查询用原离散二次曲面运动生成器、多面体精确布尔交、W1 的带符号十矩和 W3 的移动面通量导数形成响应。布尔后端为 libigl/CGAL。此轮 Cherchi 调用数为零。两次诊断均只允许一个起点和两次 Newton 更新，原生目标使用 `InertiaAndLowering`，90% 惯量在此轮按查询做独立可行性检查。

| 初值 | 原生查询 | 接受步长 | 初末方程残差 | 终点减重 | 终点固定轴惯量保留 |
|---|---:|---|---|---:|---:|
| 预选的局部椭球 | 6 | 0.25、0.125 | 7.518883 → 7.493579 | 53.2363% | 55.7799% |
| seed=0 的第一个原生初值，双曲面 | 3 | 0.5、0.5 | 17.498348 → 13.486670 | 44.7541% | 66.8590% |

两条轨迹均以 `RESOURCE_LIMIT_REACHED` 结束。椭球有一次试探走出负定椭球接口，另外两次试探触发 libigl 的输入 winding 断言，求解器回退后继续。双曲面首次回调用已绑定输入及可执行文件哈希的几何结果重放，后续两个布尔均为新计算。三个双曲面切割结果都显式检查了截断盖面未进入结果。原生方程残差不能代替惯量、质量或制造验证。

W1 分别重新装配了两个椭球接受点和双曲面终点。双曲面终点有 61,796 个顶点和 124,352 个面。其有向边界链、与部件之和一致的十矩、最低点等于原唯一陀尖这三项均通过独立核验。整模的自相交、完整制造条件和严格收敛仍需单独验证。

## 导数和数值对照

W2 在完整精确 D′ 上跑了基点和三个系数方向的双尺度中心差分，共 13 次。W1 核对输入曲面交线、保留区域、面来源及十矩。W3 通过移动面通量和交点运动导出的带符号四面体微分两路计算 10×10 Jacobian。椭球基点两者最大绝对差小于 2.7e-40，双曲面初值小于 2.2e-39。这些数描述相同离散几何和源速度上的数值一致性。

对严格包住局部椭球的固定裁剪盒，W2 再以 41 个点覆盖全部 10 个系数和两个步长。13 个重合点逐一核对了有向精确三角面和原始父面。200 个差分分量的最大缩放偏差为 7.111e-7，预定容差为 1e-6。该结论限于这个椭球邻域。源归一化最后一位的差异记在 `crop_source_normalization.json`，数值差分使用登记的原始系数增量。

五次局部查询的完整 D′ 布尔耗时合计 326.23 秒。裁剪预计算耗时 99.13 秒，同五次裁剪查询耗时合计 3.03 秒，计入预计算后实测 3.19 倍。机器上有并行任务，这组时间只代表这批布尔子进程。此次未计入神经网络，也未测量端到端加速。

## W6 的真实标签离线评估

输入是同一个 D′ 的 8³ 个采样点，其中 16 点属于材料。评估采用预先确定的 3 个模型训练种子，在原解析训练集上重跑，按原验证规则选步。独立的真实离散十矩和 Jacobian 仅用于评分。几何平移变换同时作用于原生系数、十矩和 Jacobian。各点均针对可变设计区，固定壳层另行核算。

| 真实查询 | 十矩相对 L2 的种子均值 | Jacobian 相对 L2 的种子均值 |
|---|---:|---:|
| 初始小椭球 | 70.1984 | 0.5989 |
| 第二次更新后的小椭球 | 79.2029 | 0.6238 |
| 原始双曲面初值 | 0.4689 | 0.7426 |

此处误差是比值，1.0 对应 100%。初始小切块质量为 0.000736，三种子预测约 0.064–0.076。216 个求积原子中只有 25 个得到非零几何支持。在该几何采样、固定温度和共享正权重头下，第一椭球与双曲面这对查询至少有一项相对质量误差达 84.81%。真实输入包含一个模型的三个相关查询，尚不能估计跨模型泛化。神经网络参与原生求解的调用数为零。

## 复现

W0 保留两个完整 `*_trace.json`、基于原始 A 十矩再算出的 `*_states.json` 和 W6 输入的占据采样。完整原始材料包有 258 个带 SHA-256 的条目。W1 和 W2 的原始材料各自在子归档内。`crop_source_normalization.json` 是归档后补充的源坐标核查。

```bash
# 在 W0 分支根目录执行。需要 Eigen、CGAL、libigl、GMP/MPFR 头文件。
# 非系统安装可使用 EIGEN_INCLUDE、CPATH、LIBRARY_PATH。
bash research/NeuralMoments/contracts/build_m35_native_kernel.sh /tmp/m35-kernel
python -m research.NeuralMoments.contracts.m35_replay_kernel /tmp/m35-kernel/kernel \
  research/NeuralMoments/contracts/m35/ellipsoid_trace.json \
  research/NeuralMoments/contracts/m35/stock_seed0_trace.json
python -m pytest -q research/NeuralMoments/contracts/test_m35_crossing_newton.py
```

重编译后的内核逐项重现了全部九个原生候选系数、接受或回退事件、终止状态及事件摘要。重新执行布尔可使用 `python -m research.NeuralMoments.contracts.m35_crossing_newton --help` 模块入口，并提供 W1/W2/W3 工作区与 M32 精确输入。椭球原轨迹使用 `--moment-method exact`，双曲面使用 `--chart full-rank --start-index 0`。首次几何回放是可选项，必须满足历史输入和可执行文件指纹匹配。

原始材料包 SHA-256 为 `df9581cfcf69881eae02bcc4b1b1cb6481d5b9aa68a3657980ac421a088e9b86`，文件名为 `Spin_test2_M35_Noodles_raw_evidence.tar.gz`。

## 与初始目标的差距

目前可行的多盒基准仍为减重 9.6705%、保留 91.9664% 固定轴惯量。两条真实 Newton 轨迹展示了从切割到壳层重组的过程，但两个终点均低于惯量门槛。下一轮应从满足 90% 惯量的切割开始，在线搜索时检查整模可行性，同时用保留未见几何的固定评估集检验局部细化的 PINO，在精确复算终态相同且满足约束时再比较总耗时。

## 窗口交接

W0 逐项核对了 W6 三个输入的几何指纹、原生系数、可变区十矩、100 个导数分量和完整的 8³ 占据数组，并核对评分文件的输入摘要。

| 窗口 | 本轮公开证据 |
|---|---|
| W1 | [精确区域、装配与十矩](https://github.com/volcano-code/Spin_test2/blob/5e1396ba95f274dd997209094f386b883bd10232/research/NeuralMoments/performance/M35_NOODLES_CROSSING.md) |
| W2 | [原始切割与计时](https://github.com/volcano-code/Spin_test2/blob/9c8693ed8ea46efea78ec73d9069dfc29911013c/research/NeuralMoments/teacher/M35_NOODLES_EXACT_CROSSINGS.md) |
| W3 | [离散导数和双尺度差分](https://github.com/volcano-code/Spin_test2/blob/618e194b6c53b3860afe239ef87b215e95c094ce/research/Noodles_M32/M35_CROSSING_DERIVATIVE.md) |
| W6 | [模型预测与逐样本真值对比](https://github.com/volcano-code/Spin_test2/blob/b4bd3ba04270d853a80a1bd3b06c5ecd52a5b634/research/NeuralMoments/models/pcraft_ufno/spin_pino/M35_REAL_SHADOW.json) |
