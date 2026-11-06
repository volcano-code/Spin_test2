# Heart 首轮验证 · H8f point-000

执行日期为 2026-11-06。H8f 冻结请求与 M21 的请求数据一致，求值映射与完整材料哈希仍有差异。本次所有几何结果属于 H8f 的已保存输出。

| 判定 | 可重现的结果 |
|---|---|
| [W1 设计侧覆盖](https://github.com/volcano-code/Spin_test2/blob/e14573948261806d51cf0e8ce8793aa8942379e7/research/NeuralMoments/performance/M29_HEART_CUT_FACES.md) | 全部 26,244 个设计面的内部均已核验。完整保留 14,900 面，完整移除 10,648 面，部分切割 696 面。在划分出的 3,200 个开单元中，1,134 个保存的输出三角面具有预期朝向，遗漏、额外覆盖与重叠面积均为零。|
| [W2 输入与交点](https://github.com/volcano-code/Spin_test2/blob/e702344d4f9037fd8f8f55bbdc785a1f494ddaeb/research/NeuralMoments/teacher/M22_P3_POINT000_OPERAND_REPLAY.md) | 在物理和目标坐标中精确重建 8,455 个顶点。784 个交点来自已验证的输入边与对侧三角面。输出的 16,910 个三角面各自匹配构造记录中的输入父面，其中 16,034 个设计侧面同向，876 个切割体侧面反向。另提供新运行的完整材料输出选项。|
| W0 独立截面复算 | 在设计面 3456 所处的平面与切割体的 320 个三角面逐一求交，共得到 42 段。第 125 号切割面产生唯一穿过父面内部的线段。对父面三角形三个角点作射线测试，交数为 2、1、2。按奇偶性去掉中间角点一侧后，保留部分的有向三角形恰好是输出面 14948 和 14949，精确投影面积残差为零。|

W0 从原始输入独立求平面截面和射线交数；W1 使用凸切割体的支持半空间。可在 W0 分支根目录使用原始 H8f ZIP 复算第一个面。

```sh
python -m research.NeuralMoments.contracts.heart_point0_first_face "$H8F_ZIP"
```

W0 比较了两个程序的逐面输出，全部 16,034 个设计侧输出面及对应的父面编号逐项一致。第一组逐面清单 SHA-256 为 `fd3c67802be26e267d41f2bbe823b557fb13de18b9bc5907ecc2e6ea90076783`，第二组映射的规范 JSON SHA-256 为 `c1de258442374ba9678da93b3f12e4cfc150d82dabcbefcdc966a1d0ea2e0270`。

下一步要从输入独立验证切割体侧哪些区域应出现，查验该侧 876 个输出面是否完整并覆盖恰好一次，再判定整体区域归属。同次调用完整材料的归档与重放也须验证。当前物理训练样本为 0，雅可比保持未准入，尚无 Newton 步或求解加速结果。W1 旧归档中评估点 4、116、117 的 1,555 个切割父面继续单独计数。

同轮工程验证见[新仓库备份工作流修复](https://github.com/volcano-code/Spin_test2/commit/25c5905977a69fcbbdd7e47557127f2f2ebb64d5)及 [W6 梯度回归](https://github.com/volcano-code/Spin_test2/commit/a3fe82ddcc5b418630411017b90cc8502de1ae18)。
