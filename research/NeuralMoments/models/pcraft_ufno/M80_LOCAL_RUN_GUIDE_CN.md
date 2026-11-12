# M80 本地操作指南（只使用 Spin_test2）

所有代码更新都在你现有的 Spin_test2 工作目录内完成。不需要重新 clone，不创建第二个本地仓库。

## 1. 更新现有 Spin_test2

~~~bash
cd /data/xvxinyao/Spin_test2

git status --short
git fetch origin
git checkout research/m80-pnmo-large-train-20261112
git pull --ff-only origin research/m80-pnmo-large-train-20261112

git log -5 --oneline
git status --short
~~~

如果 git status 显示你有未提交修改，先停止并备份/提交自己的修改，不要用 reset 或 clean 覆盖它们。

## 2. 设置环境

~~~bash
export SPIN_ROOT=/data/xvxinyao/Spin_test2
export SPIN_PY=/home/lin/anaconda3/bin/python
export M72_ROOT=/data/xvxinyao/Spin_M68_Trellis_Data_Kit

export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
unset MKL_SERVICE_FORCE_INTEL KMP_DUPLICATE_LIB_OK

cd "$SPIN_ROOT"
export PYTHONPATH="$SPIN_ROOT"
~~~

M72 的大体积几何和标签仍留在原 M72 数据目录，不复制进 Git 仓库。Spin_test2 只保存模型、适配器、测试和训练代码。

## 3. 运行回归与 M80 合同测试

~~~bash
python -m compileall -q research/NeuralMoments/models/pcraft_ufno/spin_pino

python -m pytest -q \\
  research/NeuralMoments/models/pcraft_ufno/spin_pino/tests/test_native_pcraft.py \\
  research/NeuralMoments/models/pcraft_ufno/spin_pino/tests/test_native_train.py \\
  research/NeuralMoments/models/pcraft_ufno/spin_pino/tests/test_spin_pino.py \\
  research/NeuralMoments/models/pcraft_ufno/spin_pino/tests/test_dataset.py

python -m research.NeuralMoments.models.pcraft_ufno.spin_pino.native_train --help
~~~

## 4. 生成 M72 正式语料审计收据

先使用当前已经存在的 M69/M72 原始语料做模型 preflight。假设 manifest 位于：

$M72_ROOT/data/combined70-v2/operator-manifest.json

~~~bash
export M72_MANIFEST="$M72_ROOT/data/combined70-v2/operator-manifest.json"
export M72_AUDIT="$M72_ROOT/data/combined70-v2/m80-audit.json"

test -f "$M72_MANIFEST"

cd "$M72_ROOT"
"$SPIN_PY" src/corpus69.py audit \\
  --manifest "$M72_MANIFEST" \\
  --out "$M72_AUDIT"

test -s "$M72_AUDIT"
~~~

审计失败或 ready_for_training 不为 true 时停止。不要手工修改 manifest fingerprint、状态或审计阈值。

## 5. 先跑 5090D GPU preflight

先运行较小的 large 配置，确认 CUDA、完整反向、AdamW 和 checkpoint 往返：

~~~bash
cd "$SPIN_ROOT"
mkdir -p runs/m80

"$SPIN_PY" -m research.NeuralMoments.models.pcraft_ufno.spin_pino.native_train preflight \\
  --manifest "$M72_MANIFEST" \\
  --audit "$M72_AUDIT" \\
  --out "runs/m80/preflight-large-$(date -u +%Y%m%dT%H%M%SZ).json" \\
  --preset large \\
  --device cuda \\
  --amp bf16 \\
  --batch-queries 16 \\
  --chunk 4096 \\
  --threads 1 \\
  --cache-mib 512
~~~

large 通过后，再测 XL：

~~~bash
"$SPIN_PY" -m research.NeuralMoments.models.pcraft_ufno.spin_pino.native_train preflight \\
  --manifest "$M72_MANIFEST" \\
  --audit "$M72_AUDIT" \\
  --out "runs/m80/preflight-xl-$(date -u +%Y%m%dT%H%M%SZ).json" \\
  --preset xl \\
  --device cuda \\
  --amp bf16 \\
  --batch-queries 16 \\
  --chunk 2048 \\
  --threads 1 \\
  --cache-mib 512
~~~

重点检查：
- status = passed
- 最大训练对象的 native interval 数量
- 完整 Jacobian loss backward 和 AdamW update
- peak allocated / reserved
- checkpoint round-trip 最大绝对误差不超过 10^-10

这两步是 GPU 资源与训练图验收，不代表模型已经学好，也不代表 Newton 已经接通。

## 6. 当前 32-query M72 语料只允许短 pilot

当前正式基线如果每个对象仍然只有 32 条查询，可做 200 步工程 pilot，但必须显式降低查询门槛：

~~~bash
"$SPIN_PY" -m research.NeuralMoments.models.pcraft_ufno.spin_pino.native_train train \\
  --manifest "$M72_MANIFEST" \\
  --audit "$M72_AUDIT" \\
  --out "runs/m80/pilot-32q-$(date -u +%Y%m%dT%H%M%SZ)" \\
  --preset large \\
  --device cuda \\
  --amp bf16 \\
  --max-steps 200 \\
  --max-seconds 1800 \\
  --min-queries-per-record 32 \\
  --min-train-groups 50 \\
  --min-val-groups 7 \\
  --min-test-groups 10 \\
  --batch-queries 16 \\
  --chunk 4096 \\
  --threads 1
~~~

这只是 M80 适配 pilot，不得作为 256-query 正式长训结果。

## 7. 256-query 正式语料通过后才启动较大训练

当 M72 R2 导数审计、8 对象 Pilot、全量 67 对象 ×256 查询生成和正式 corpus audit 全部通过后，设置新的 manifest/audit 路径：

~~~bash
export M72_256_MANIFEST=/绝对路径/到/正式256查询语料/operator-manifest.json
export M72_256_AUDIT=/绝对路径/到/正式256查询语料/audit-recheck.json
~~~

先核对：

~~~bash
test -f "$M72_256_MANIFEST"
test -f "$M72_256_AUDIT"
~~~

再启动 XL 10k pilot：

~~~bash
"$SPIN_PY" -m research.NeuralMoments.models.pcraft_ufno.spin_pino.native_train train \\
  --manifest "$M72_256_MANIFEST" \\
  --audit "$M72_256_AUDIT" \\
  --out "runs/m80/xl-10k-seed800071" \\
  --preset xl \\
  --device cuda \\
  --amp bf16 \\
  --max-steps 10000 \\
  --max-seconds 14400 \\
  --min-queries-per-record 256 \\
  --min-train-groups 50 \\
  --min-val-groups 7 \\
  --min-test-groups 10 \\
  --batch-queries 16 \\
  --chunk 2048 \\
  --threads 1 \\
  --cache-mib 512 \\
  --seed 800071
~~~

中断后只有在源码、manifest、audit、运行环境和训练配置合同完全相同的情况下，才使用：

~~~bash
--resume runs/m80/xl-10k-seed800071/last.pt
~~~

不要对原运行目录换配置续训。

## 8. 本地结果回传

优先回传：
- 两份 preflight JSON；
- 200 步 pilot 的 report.json；
- best.pt、last.pt 的 SHA-256（不必上传大型权重）；
- nvidia-smi 峰值显存；
- M72 manifest/audit 的 fingerprint 和查询数量。

不要上传原始大体积 NPZ/OBJ 或所有 checkpoint。

## 当前边界

- M72 原始 32-query 语料可用于 preflight/短 pilot；正式 10k 训练默认要求每个合格对象至少 256 查询。
- M72 R2 导数审计修复尚需在真实本地数据上确认；本指南不放宽阈值。
- M80 当前实现的是 P-CRAFT + M72 native interval moment/Jacobian 训练模型。Spin-It Faster Newton bridge、自适应精确校正和完整端到端速度实验仍是后续研发阶段。
