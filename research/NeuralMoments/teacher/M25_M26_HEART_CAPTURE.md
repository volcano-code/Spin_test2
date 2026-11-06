# Heart 线性项与切换两侧的 14 次原生采集

两个批次在 Spin_test2 的运行器上串行执行，复用了 M23 基点已核验的三个 ELF。每个样本有独立运行编号、请求、标签、完整同次材料和资源记录。逐文件哈希已跨环境读回复核。[捕获记录与 SHA-256](M25_M26_HEART_FRESH_RECEIPTS.json)

| 批次 | 原生调用 | 逐面复核 |
|---|---:|---|
| [M25](https://github.com/volcano-code/Spin_test2/actions/runs/42185824300) | 8 | 分别扰动原生线性系数 c[1]、c[2]；每列四点与基点的父面归属一致。[逐点结果](M25_HEART_OPERAND_REPLAY.json) |
| [M26](https://github.com/volcano-code/Spin_test2/actions/runs/42186214516) | 6 | 分别采集原生交叉项 c[7] 事件两侧各三点。左侧三点的父面归属与基点一致，右侧三点具有相同的新归属。[逐点结果](M26_HEART_OPERAND_REPLAY.json) |

每份材料均从输入操作数重建 8,455 个顶点，并在两个坐标系中核验。16,910 个输出三角面的归属和朝向均通过差集检查，包含 16,034 个设计侧同向面和 876 个切割体侧反向面。逐面支持证据不单独证明完整布尔实体或连续可微邻域。所有材料保留 `split=excluded`、`J_D=null`。

在 W2 根目录复现时，将 PUBLISHED_SHA 设为运行器发布的对应 stencil-result.json 的 SHA-256，可从上述捕获记录取得。线性批次使用 `--linear`，事件批次使用 `--c7`。

```sh
python research/NeuralMoments/teacher/m24_c0_operand_replay.py \
  "$GROUP_DIR" "$GROUP_ID" "$PUBLISHED_SHA" --linear
```
