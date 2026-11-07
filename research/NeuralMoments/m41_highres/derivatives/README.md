# 有限三角切割面的十矩与局部导数

M38 面板包括 Noodles 原生已接受的 q1 与 q6，每状态三个冻结方向为两状态之间的大圆切向、单位球切空间上的常数系数方向、交叉项 xy 系数方向。四个步长为 10⁻³、10⁻⁴、10⁻⁵、10⁻⁶ 弧度，每次使用真实参数大圆上的正负两个端点，固定原生有限三角面片连通性。不同方向或步长不能视作独立物体。

从首个原生有限椭球的顶点恢复固定法线和面序。随后仅用 q 与固定法线解逆高斯映射，产生正负扰动后的顶点。两个起点的顶点和速度分别与原生数据交叉检查。分析 Jacobian 也可以直接由 q 更新顶点速度。参考是独立于读出估计的原生移动面几何 J_tip。未重跑原生布尔端点或完整优化器。

二值对照按体素中心判包含并积整个小立方体。每个未扰动点的十个矩与独立 VTK 结果作一致性断言。分数体积算子保留同一组 y,z 射线和固定材料区间，求它们与有限三角面片的真实交点并精确积分小棱柱在 x 方向的连续端点。所有有序交点成对处理，同一射线可以有多段。y,z 向离散误差以及固定设计域的体素化误差仍存在。

通过面片法向和三个顶点速度的重心插值，可以对连续端点做解析求导，单次表面积分组装完整十乘十矩阵。体素边界、轮廓线或三角边正好穿过采样射线时，导数可能切换。中心差分一致性是独立于原生 Jacobian 误差的指标。

## 本地重放

解压上一轮 Spin_5090_HighRes_Kit.zip。将 KIT 设为该包内含 geometry 和 inputs 两个子目录的根目录，LAB 设为本 README 所在的 derivatives 目录。采用 NumPy 与 VTK 9.3.1。以下 CPU 命令不需 PyTorch 或 GPU。

```bash
export KIT=/absolute/path/to/Spin_5090_HighRes_Kit
export LAB=/absolute/path/to/derivatives
export PYTHONPATH="$KIT/geometry"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VTK_SMP_MAX_THREADS=1
python "$LAB/native_jv.py" --handoff "$LAB/fixtures" --design "$KIT/inputs/Dprime.txt" --sizes 128 256 512 1024 --output local-jv.json
python "$LAB/analytic_jacobian.py" --handoff "$LAB/fixtures" --design "$KIT/inputs/Dprime.txt" --sizes 128 256 512 1024 --output local-analytic.json
python "$LAB/m37_crosscheck.py" --runs "$LAB/fixtures/runs" --design "$KIT/inputs/Dprime.txt" --sizes 512 1024 --output local-m37.json
```

results.json 给出 192 个逐方法记录（4 个网格 × 2 个状态 × 3 个方向 × 4 个半步长 × 2 种读出）。analytic.json 给出同一 M38 两点的完整原生 Jacobian 对照，以及对分数体积算子自身全部十列的中心差分检查。表中单次 CPU 耗时仅含源顶点与速度求值、切割面交点遍历与矩阵组装。原生切割、固定几何预处理、学习网络、优化器均不计入，精度和时延不可外推为端到端加速。

m37.json 在同一物体上追加五个不同的已接受状态。三例椭球由 q 独立重建，另两例有限截断双曲面差集采用原生顶点与**源顶点速度**，仅独立检验移动面到质量矩 Jacobian 的积分。该数据的截断端盖没有出现在原生保留物体的边界。M37 的 exact/call-001 是明显较难的样例，1024³ 完整 J 相对误差仍为 3.34%。其他对象和拓扑切换需要另行实验。
