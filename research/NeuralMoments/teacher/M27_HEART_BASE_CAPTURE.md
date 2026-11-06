# Heart 基点剩余系数的原生采集

M23 基点的准备请求和三个原生 ELF 身份均已核对。下表的每个样本对应一次新的 Cherchi 差集调用。W2 从原始操作数重建顶点，复查了模型与目标两个坐标系中的位置，以及输出三角面在父操作数上的归属和朝向。[逐样本身份及结果](M27_HEART_BASE_CAPTURE_RECEIPTS.json)

| 批次 | 原生列 | 运行与尝试 | 新材料 | 逐点顶点数 | 逐点面数 |
|---|---:|---|---:|---:|---:|
| M27 | c[3] | [42190295793 / 2](https://github.com/volcano-code/Spin_test2/actions/runs/42190295793/attempts/2) | 4 | 8455 | 16910 |
| M27 | c[4] | [42190295793 / 2](https://github.com/volcano-code/Spin_test2/actions/runs/42190295793/attempts/2) | 4 | 8455 | 16910 |
| M27 | c[5] | [42190295793 / 1](https://github.com/volcano-code/Spin_test2/actions/runs/42190295793/attempts/1) | 4 | 8455 | 16910 |
| M27 | c[6] | [42190295793 / 1](https://github.com/volcano-code/Spin_test2/actions/runs/42190295793/attempts/1) | 4 | 8455 | 16910 |
| M27 | c[7] | [42190295793 / 1](https://github.com/volcano-code/Spin_test2/actions/runs/42190295793/attempts/1) | 4 | 8455 | 16910 |
| M27 | c[8] | [42190295793 / 1](https://github.com/volcano-code/Spin_test2/actions/runs/42190295793/attempts/1) | 4 | 8455 | 16910 |
| M27 | c[9] | [42190295793 / 1](https://github.com/volcano-code/Spin_test2/actions/runs/42190295793/attempts/1) | 4 | 8455 | 16910 |
| M28 | c[7] | [42191401306 / 1](https://github.com/volcano-code/Spin_test2/actions/runs/42191401306/attempts/1) | 4 | 8455 | 16910 |

初次运行的七组均在解析参数时以代码 2 退出。将系数列绑定至组编号后，重试沿用了原请求、预登记步长和冻结的原生模块。修正后首轮有 2 个列任务在排队阶段取消，表中保留其实际采集尝试编号。[初次运行](https://github.com/volcano-code/Spin_test2/actions/runs/42189847862)

按列复现来源审计时，以该组对应的公开结果 SHA 为输入。将 GROUP 设为材料组目录，ID 设为组编号，K 设为列号。复查半径减半的 c[7] 时另加 --half。

```sh
python research/NeuralMoments/teacher/m24_c0_operand_replay.py \
  "$GROUP" "$ID" "$PUBLIC_RESULT_SHA256" --base-column "$K"
```

全部样本保持 split=excluded、J_D=null。预登记差分与构造一致性由 W0 对独立保存的材料评定。

较小模板的两个端点与宽模板的两个内点坐标相同，四次调用均有各自的来源记录。两组重复坐标的精确边界和父面映射相同。逐字段比较只观察到诊断雅可比的浮点差别，最大绝对差分别为 1.39e-17 和 2.78e-17。
