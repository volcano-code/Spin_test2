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
