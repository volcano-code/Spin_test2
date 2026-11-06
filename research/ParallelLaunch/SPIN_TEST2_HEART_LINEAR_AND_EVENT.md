# Heart 线性项与构造切换验收

2026-11-06。以已验收的 M23 原生运行为基点，固定请求、三个原生模块及求值设置，启动两个独立批次。线性两列进行了 8 次新布尔调用，交叉项切换两侧进行了 6 次。保存材料全部带独立编号。在另一环境解密后，从目标坐标系的有向网格重新作有理数积分，以实际 binary64 系数的精确有理表示构造差分权重。[W2 采集与逐面来源](https://github.com/volcano-code/Spin_test2/blob/a4a206bf8618662b25edaf4d652be021c2a5052c/research/NeuralMoments/teacher/M25_M26_HEART_CAPTURE.md)

## 基点的两个线性系数

对十个变量域质量矩分别计算误差，分母为 max(1, |原生雅可比分量|)。每列、每个分量、每种差分均须不超过 10⁻⁶。三种差分各检查十个分量，两列共 60 项通过。八个扰动点与基点之间均无归一化构造身份或有向三角面增删。[事前协议](https://github.com/volcano-code/Spin_test2/blob/c52376ceca25776f4f85322c4fcc7bf3fb243faf/research/NeuralMoments/contracts/HEART_LINEAR_COLUMNS_PROTOCOL.md) · [逐分量数值与样本身份](SPIN_TEST2_HEART_LINEAR_COLUMNS.json)

| 原生系数 | 三点差分，h=3×10⁻⁶ | 三点差分，h/2 | 合并五点差分 |
|---|---:|---:|---:|
| c[1] | 1.381×10⁻¹⁰ | 1.595×10⁻¹⁰ | 1.666×10⁻¹⁰ |
| c[2] | 8.617×10⁻¹² | 2.385×10⁻¹² | 3.203×10⁻¹³ |

在这个球形基点上，两个线性项分别改变球心；c[1] 另产生一阶半径变化。W3 独立传播该运动到交点和十矩，条件导数与原生列的最大绝对差分别为 5.50×10⁻¹⁶ 和 1.31×10⁻¹⁷。[W3 源顶点运动复核](https://github.com/volcano-code/Spin_test2/blob/82a7f989897da97a4fa980061051ab4ce658a8ea/research/Heart_v1_20261017/Heart_H8_Asymmetric_CAD_20261018/reports/H9_LINEAR_SOURCE_TANGENTS.md)

## 交叉项的分支切换

六个采样点均只改原生对称矩阵项 c[7]=A_yz=A_zy。多项式 yz 的系数为 2c[7]。冻结的边事件行列式在左侧三点为正，在右侧三点为负。同侧三点的构造相同；两侧最近的两个点之间有两个构造身份被替换，17 个有向三角面移除、17 个新增。切割体连接关系保持一致。[两侧事前协议](https://github.com/volcano-code/Spin_test2/blob/e7080583d993f62c793d6ee3f50fd06a5e58d483/research/Heart_v1_20261017/Heart_H8_Asymmetric_CAD_20261018/reports/H9_C7_TWO_SIDE_PROTOCOL.md) · [六份新材料的验收数据](SPIN_TEST2_HEART_C7_SIDES.json)

| 近端点的原生 c[7] | 同侧差分 | 最大缩放误差 | 通过分量 |
|---:|---|---:|---:|
| 2.2705862283444327×10⁻⁶ | 向后三点 | 2.331×10⁻⁸ | 10/10 |
| 2.2905862283444326×10⁻⁶ | 向前三点 | 2.476×10⁻⁸ | 10/10 |

原生采样把构造变化定位到宽约 2×10⁻⁸ 的区间。尚未认证区间内的精确根、根数或事件处的单侧极限。历史数据中跨事件的中心差分虽然只有 3.86×10⁻¹² 的误差，构造检查仍将其拒绝。[历史对照](https://github.com/volcano-code/Spin_test2/blob/82a7f989897da97a4fa980061051ab4ce658a8ea/research/Heart_v1_20261017/Heart_H8_Asymmetric_CAD_20261018/reports/H9_C7_GUARDED_STENCILS.md)

## 复现与里程碑

在 W0 检出中，用以下入口复算私有完整材料。LINEAR_SHA 和 C7_SHA 为两个 GitHub 运行分别公开的 stencil-result.json 的 SHA-256，可从 W2 采集记录取得。程序也会检查下载包内的运行编号、文件哈希、模块身份、构造与积分值。

```sh
python -m research.NeuralMoments.contracts.heart_linear_columns \
  --base "$BASE" --group "$LINEAR" --expected-group-id heart-linear-42185824300-1 \
  --published-result-sha256 "$LINEAR_SHA"
python -m research.NeuralMoments.contracts.heart_c7_event \
  --base "$BASE" --group "$C7" --expected-group-id heart-c7-42186214516-1 \
  --result-sha256 "$C7_SHA"
```

基点已有 c[0]、c[1]、c[2] 三列通过同一套采样检验。c[7] 的近端点单独登记。后续优先筛查其余基点列，对跨事件的步长缩步或按侧求导；在原目标、约束和冻结容差下记录可接受的 Newton 步，再以收敛后的网格复算十矩和约束。当前真实训练集仍为空，首月首个端到端案例、学习器加速比较和 90% 惯量约束最小质量试验均待完成。
