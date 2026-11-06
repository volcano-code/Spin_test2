# M37 · 解析标定能否改善冻结 Spin-PINO 的真实切割误差？

以 M36 的 131 号训练种子和第 20 步验证检查点为起点，在其正权重与单位权重之间增加单一系数。允许负系数，所以校正后的权重可以超过 1，同时保持非负。训练用新增的 32 个解析盒面半椭球切割，在十矩和原生十维 Jacobian 两项相对平方误差上等权。三路求积节点、温度、输入及回退规则相同。[冻结协议](M37_CALIBRATION_PROTOCOL.json)

标定的最优系数落在上界 1，与原冻结算子逐项相同。下表先在每个查询上计算十矩相对 L2 和 Jacobian 相对 Frobenius 误差，再对同行查询取算术平均。

| 评估集合 | 查询数 | 单位权重，十矩 / Jacobian | 标定后与原冻结算子，十矩 / Jacobian |
|---|---:|---:|---:|
| 解析盒面半椭球 | 8 | 0.481 / 0.476 | 0.453 / 0.449 |
| Noodles | 3 | 0.771 / 0.784 | 0.804 / 0.815 |
| Bunny | 3 | 0.079 / 0.091 | 0.445 / 0.402 |

这组解析标签偏好的修正在两个真实模型上均未改善平均误差。它将新增可调自由度限制为一个标量，未检验其他几何表示或训练分布。两件真实模型已在 M36 检查过，本轮作开发诊断。所有真实标签只参与评分。本轮未运行或计时闭环求解器。[逐例结果与输入摘要](M37_CALIBRATION.json)

从 W6 仓库根目录复现：

```bash
p=research/NeuralMoments/models/pcraft_ufno/spin_pino
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m research.NeuralMoments.models.pcraft_ufno.spin_pino.residual_calibration --output /tmp/m37-calibration.json
python -m pytest -q "$p/tests"
```

## 在实际 Newton 轨迹上检查惯量门槛

对 W0 三条已完成轨迹中的每个实际系数 q 分别调用两个 M36 冻结算子。沿用原生首步对照的种子 131、24³ 节点、温度 0.0075 及 8³ 几何输入。算子在网格系预测二次多项式正侧的保留变量材料十矩，沿用 W6 的原生系数和矩平移至尖端坐标。按 W0 约定，将固定材料 F′ 加入后用 (xx+yy)/Izz(A) 计算相对固定轴惯量。每个精确结果同时核对 W0 保存的整模十矩和惯量区间，两个算子在共享锚点的预测质量也与原始 M36 记录一致。

三条轨迹合计 14 次求值，其中同一锚点出现三次。单独统计锚点之后的 11 个不同试探，精确几何通过 4 个、拒绝 7 个。共享锚点的精确保留率为 98.231%，学习和数值算子的直接预测分别为 69.993% 和 71.855%。观察到这一偏差后，补做了事后锚点校正：每个候选的修正值等于该锚点的精确保留率加上代理从锚点至候选的保留率预测变化。计算这个修正值只用了共同锚点的精确值。所有四组均使用原 90% 门槛。

| 预测方式 | 正确放行 | 正确拒绝 | 误放 | 误拒 |
|---|---:|---:|---:|---:|
| PINO 直接预测 | 0 | 7 | 0 | 4 |
| 单位权重直接预测 | 0 | 7 | 0 | 4 |
| PINO 锚点校正 | 4 | 0 | 7 | 0 |
| 单位权重锚点校正 | 4 | 1 | 6 | 0 |

直接筛选会丢掉全部通过惯量检查的试探，锚点校正后仍放行大部分未达标试探。这里的放行仅指惯量检查。事后校正公式没有给出误差界。这次回放未测量在线筛选时间。[逐次结果、真实几何摘要与轨迹摘要](M37_TRAJECTORY_INERTIA.json)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m research.NeuralMoments.models.pcraft_ufno.spin_pino.trajectory_inertia \
  --traces /path/to/m37/runs --fixed-ten /path/to/Fprime.ten --original-ten /path/to/A.ten --output /tmp/m37-inertia.json
```
