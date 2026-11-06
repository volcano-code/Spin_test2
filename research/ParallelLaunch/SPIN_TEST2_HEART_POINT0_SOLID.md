# Heart point-000 的实体差集验收

同日后续的新原生运行已经完成，实体有向曲面与本记录一致，十个精确矩逐项相等。新运行拥有独立的完整材料身份。[新运行记录](https://github.com/volcano-code/Spin_test2/blob/6aa9db4f4f4286cb5551ab8a4571b7159b781134/research/NeuralMoments/teacher/M23_HEART_POINT000_FRESH.md)

2026-11-06。在 H8f 的原始物理模型坐标下，设 D 为冻结设计网格围成的实体，C 为该评估点使用的 320 面切割体。保存网格围成的实体等于正则化差集，即 D 去除 C 后，取内部再取闭包所得的实体。验收绑定到归档 `99a050f4…` 和历史完整材料 `6be4faf5…`。[机器可读结果](SPIN_TEST2_HEART_POINT0_SOLID.json)

| 验证 | 实测结果 |
|---|---|
| 设计侧预期边界 | 26,244 面中，14,900 面完整保留、10,648 面完整移除、696 面部分切割。 |
| 切割体侧预期边界 | 320 面中，48 面完整保留、187 面完整移除、85 面部分保留。4,763 个开分区的精确遗漏、额外与重叠面积均为零。 |
| 全部输出面的归属 | 从两份输入推导出的 16,910 个输出面分配，与独立重建的逐面编号全部一致；16,034 面沿用设计体朝向，876 面使用切割体反向。 |
| 接缝与开区域 | 独立枚举 7,711 对输入三角面候选，用精确有理运算得到的 784 段有向交线与输出接缝逐段一致；两个闭环分别长 102 和 682 段。320 个切割体面的内部锚点射线检查均与输出相符。 |
| 实体与体积 | 输入两体和输出均为闭合、一致向外且无非法自交的单分量曲面。独立检查共覆盖 382,627 对候选三角面。输出精确有向体积等于归档分数，约为 0.4638104336177978 模型体积单位。 |

逐面检查给出了差集应有的全部有向边界。独立接缝检查排除了这次输入的共面重合和孤立相切，并验证边界片在相交处闭合。对这些有界、嵌入且一致定向的实体，无界外部的绕数为零。因此，两边占据相同的开区域，取闭包后得到相同的正则化实体。[W1 双侧覆盖](https://github.com/volcano-code/Spin_test2/blob/15db682052e9a97779c34ba02000095b78cf4b91/research/NeuralMoments/performance/M30_HEART_CUTTER_FACES.md) · [W2 原坐标曲面检查](https://github.com/volcano-code/Spin_test2/blob/519906bc1447eddc74fb3ca0c2117b1d3c9eca09/research/NeuralMoments/teacher/M22_P4_POINT000_SOLID_BOUNDARIES.md)

## 复现

在 `Spin_test2` 的 W0 分支根目录运行。使用 Python 3.12 或更高版本、NumPy、Git、C++17 编译器、Boost 头文件和 GMP 的 C/C++ 开发库。`H8F_ZIP` 是原始 H8f 归档，`M21_PILOT_DIR` 是含已保存 spec、label 的原始运行目录。`NEW_RESULT_DIR` 是此前不存在的目录。单分支克隆可先用下列 fetch 取得另外两个研究分支。程序从固定 Git 提交导出源文件并重新运行各项检验，最后写出 result.json。

```sh
git fetch origin research/neuralmoments-w1-native-20261019 research/neuralmoments-w2-teacher-20261019
python -m research.NeuralMoments.contracts.heart_point0_solid \
  "$H8F_ZIP" "$M21_PILOT_DIR" "$NEW_RESULT_DIR"
```

本次实测使用历史多面体切割体。M21 的完整材料描述哈希是 `b136cbcc…`，求值映射也与 H8f 不同。当前新增原生布尔调用为 0，真实训练样本为 0，`J_D=null`。下一项任务是在独立的新运行中同时留存请求、原生模块身份、求值映射、完整材料和输出，再验证该新样本；可用的导数和 Newton 步还需要邻域证据。
