# Heart 基点十列的采样验收与 Newton 初值条件

对象为 H8 非对称重建 Heart。M23 基点、固定部分质量矩、密度、原生模块和输入请求均沿用上轮。W2 在同一冻结构建上为七列分别采集四份材料，再为 c[7] 缩步试验采集另外四份。W0 校验每组外部运行回执，从同次保存的有向边界独立重算十矩，用实际 binary64 横坐标生成精确有理差分权重。W2 对每份材料另行重建顶点、核验每个三角面对应的输入父面及朝向。

| 原生列 | 最大绝对扰动 | 外层三点最大缩放误差 | 内层三点 | 五点 | 采样构造与数值 |
|---|---:|---:|---:|---:|---|
| c[3] | 3e-06 | 8.390e-11 | 2.001e-11 | 4.623e-12 | 通过 |
| c[4] | 3e-06 | 2.434e-10 | 7.788e-11 | 3.710e-11 | 通过 |
| c[5] | 3e-06 | 1.308e-11 | 4.924e-11 | 6.362e-11 | 通过 |
| c[6] | 3e-06 | 5.476e-10 | 1.369e-10 | 1.385e-11 | 通过 |
| c[7] | 3e-06 | 1.818e-10 | 4.746e-11 | 3.856e-12 | 外侧构造改变，拒绝 |
| c[8] | 3e-06 | 1.672e-09 | 4.368e-10 | 2.649e-11 | 通过 |
| c[9] | 3e-06 | 1.907e-11 | 4.061e-12 | 9.421e-13 | 通过 |
| c[7] | 1.5e-06 | 4.746e-11 | 1.404e-11 | 2.899e-12 | 通过 |

每格汇总十个分量，逐分量门槛为 10⁻⁶，误差分母为 max(1, |原生分量|)。c[7] 的 +3×10⁻⁶ 样本替换了两个构造顶点身份，删除和增加各 17 个有向三角面。宽步长虽然通过三组数值检查，仍按事前约定保留拒绝。缩步后四个新样本都与基点的构造和源连接关系相符。连同先前 c[0]、c[1]、c[2]，十列现在各有一组通过相同误差与几何门槛的中心差分。这里记录的范围限于取样位置和列方向。多方向联合扰动的连续邻域仍待验证。

[七列预登记](https://github.com/volcano-code/Spin_test2/commit/5796a425b25ba8b4261eaa9b4a494a687182affe) · [缩步预登记](https://github.com/volcano-code/Spin_test2/commit/f6fb66e56d583f2abd218ecf489b2e8799f27d60) · [七列逐分量结果](SPIN_TEST2_HEART_REMAINING_COLUMNS.json) · [缩步逐分量结果](SPIN_TEST2_HEART_C7_SMALLER_BASE.json) · [W2 采集与操作数回放](https://github.com/volcano-code/Spin_test2/blob/4d6b3d92d7b7e1ba4d57d8168dfd2dc98d5dee98/research/NeuralMoments/teacher/M27_HEART_BASE_CAPTURE.md)

## 运行账本

首轮 [42189847862](https://github.com/volcano-code/Spin_test2/actions/runs/42189847862) 的七个作业均在参数解析期间退出，完整新材料为零。修复启动器后，[42190295793](https://github.com/volcano-code/Spin_test2/actions/runs/42190295793) 的 c[3] 与 c[4] 在首次尝试中排队取消，随后补跑；其余五列保留首次成功的材料，每列的实际尝试编号见逐分量结果。按准入条件触发的 [42191401306](https://github.com/volcano-code/Spin_test2/actions/runs/42191401306) 独立记录四份缩步材料。W0 回放不调用原生布尔求值。

## 下一次原生求解

严格 Newton 初值要求 μ > 0。利用保存的基点十矩、固定物理目标、四个等式约束和 C++ 的系数变换重放乘子求解，得到 μ ≈ −16.8120304342。把此球形探针作为初值时，会在求解第一步方向前触发 `initial_multiplier_nonpositive`。探针的原生 c[6] 为正，冻结求解器给旋转陀螺生成的八个初值该项均为负。默认初值序列的收敛状况仍需实测。下一次应使用原求解器已登记的确定性初值序列，保留所有初值的成功或拒绝原因，并检查新求值的试步、原始残差下降和构造切换处理。[W3 严格初值复核](https://github.com/volcano-code/Spin_test2/blob/4725ca00545e559693c7131e7f67f4ea60929cc0/research/Heart_v1_20261017/Heart_H8_Asymmetric_CAD_20261018/reports/H9_POINT000_STRICT_INIT.md) · [c[3] 源运动复核](https://github.com/volcano-code/Spin_test2/blob/4725ca00545e559693c7131e7f67f4ea60929cc0/research/Heart_v1_20261017/Heart_H8_Asymmetric_CAD_20261018/reports/H9_Z_SOURCE_TANGENT.md)

W5 已采用 W0 审查的 v2 读取器。W2/W5 的真实教师训练样本仍为 0，正式发布的 `J_D` 仍为空。项目尚需取得 Heart 的被接受 Newton 步、终态收敛和保存网格的独立复测，再进行五物体复现、同预算学习器比较及保留 90% 轴向惯量的最小质量试验。[W5 源码回读与训练准入](SPIN_TEST2_W5_V2_READBACK.json)
