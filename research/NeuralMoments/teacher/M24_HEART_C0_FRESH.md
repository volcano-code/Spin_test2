# Heart 第一个系数的四次新采集

运行 `heart-c0-42184121466-1` 使用上一轮基点的三个原生模块。模块归档和每个实际导入的 ELF 文件均按基点记录的 SHA-256 核验。从仓库 CAD 重建的请求字节与冻结基线一致。四个独立编号分别记录原生常数项的 −3×10⁻⁶、−1.5×10⁻⁶、+1.5×10⁻⁶、+3×10⁻⁶ 名义扰动。实际二进制浮点对应的精确有理偏移见运行方案。各点只执行一次原生布尔调用，共四次。[运行记录](https://github.com/volcano-code/Spin_test2/actions/runs/42184121466) · [捕获与下载哈希](M24_HEART_C0_FRESH_RECEIPT.json)

下载证据包后独立核验了 38 个文件条目，逐一重建每份材料的 8,455 个顶点，包括 784 个边与对侧三角面的交点。四份材料各有 16,034 个设计侧同向输出面、876 个切割体侧反向输出面，父面分配与基点一致。[逐点复核](M24_HEART_C0_OPERAND_CHECK.json)

用运行器归档的 `runtime/stencil-result.json` 计算 SHA-256。以下命令从解压的完整四点材料重算构造与父面归属。

```sh
python research/NeuralMoments/teacher/m24_c0_operand_replay.py \
  "$C0_STENCIL_DIR" heart-c0-42184121466-1 "$PUBLISHED_STENCIL_RESULT_SHA256"
```

采集保持 `J_D=null` 和样本排除标记。四个扰动的逐面归属检查支持比较这些采样点的边界构造，完整布尔实体验收与连续区间事件排除分别记账。
