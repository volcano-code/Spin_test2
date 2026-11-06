# W0｜数学合同、接口与总控

**GitHub 仓库**　`volcano-code/Spin_test2`

**唯一写入分支：** `research/neuralmoments-w0-contract-20261019`  
**冻结源版本：** `0bcc187b3005478baa586a6d6f450e73811ca342`  
**独占文件范围：** research/NeuralMoments/contracts/ 与 research/ParallelLaunch/  
**依赖：** 无；所有教师和神经实验的前置合同

## 当前对话的开发入口

W0 在当前对话调度 W0–W6。各窗口在新仓库已有分支提交代码和证据，使用输入哈希、复现命令和提交号交接。[首月计划](SPIN_TEST2_30_DAY_PLAN.md)

## 具体工作

1. 审计 `src/spin/QuadricCoefficients.cpp`、`python/make_it_stand/verification/faster_material.py`，冻结原生二次系数 c 与十个普通原始质量矩 r 的顺序，尤其后三个交叉项的 2 倍映射 B；不得直接将原生 dr/dc 当作对称阵。
2. 定义 `MomentContractV1`：输入几何身份、固定 F / 设计 D、坐标系/密度/尺度、c 的 projective 正比例等价、正负号切换、r 的顺序、J 类型、计算状态与误差口径。固定材料矩只计算一次。
3. 编写 `moment_coordinates.py`、`physical_transforms.py`、`schema_v1.json` 及 pytest；盒、球、椭球等解析参考，严格检查旋转、平移、归一化、Hessian 适用条件与离散例外。
4. 完成一个不依赖大型网络的 `contract_smoke.json`，归档通过与未通过条件；维护 W0–W6 分支/证据看板。

## 当期验收清单

- 含非零 xy/xz/yz 的二次曲面值与 C++ 一致。
- 真实质量矩在尺度、旋转、平移下正确；P0 解析样本通过数值对照。
- 不能把 J(c) 直接错验为对称；应在一致坐标中检查 JB^{-1} 的适用性质。
- 不定义人为的统一 1% 错误门槛，合同不冒充证明。

## 所有窗口都必须遵守的规则

- 本任务是独立 GitHub **代码开发任务**，不是让对话在后台自行持续执行。每轮必须实际检查文件、编写代码、运行可用的测试并提交证据；未经执行不能声称通过。
- 仓库 `volcano-code/Spin_test2`。禁止修改其 `main`、历史 `volcano-code/Spin`、`volcano-code/test/main`、他人 W 分支；不覆盖此前保存的失败 CI、历史报告、原始 CAD。
- 与其他窗口沟通通过 GitHub 的合同版本、提交 SHA 和文件链接；等待依赖时先完成自己的解析/纯代码 smoke，不得捏造依赖已完成。
- 每个里程碑提交到对应研究分支，反馈更改文件、通过/未通过的真实测试、误差/性能和下一步；所有优化必须保留完整材料与强真值独立验收。
- 按先前资源协议，大型原生作业串行、24GiB/swap0、每次≤60分钟、编译最多4并发；GPU 训练避免与其他窗口冲突。

