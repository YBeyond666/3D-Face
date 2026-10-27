# SMIRK／DECA 双数据集实验代码 v0.1

本工程用于自采数据 A 与 Multiface B 的训练效用对比。已经实现数据工具、官方模型适配、实验运行与基础评价代码。当前是待目标电脑联调的初版，不包含真实训练结果。

## 已实现与验证边界

| 部分 | 初版功能 | 验证状态 |
| --- | --- | --- |
| 数据接入 | Multiface PNG／OBJ／KRT 索引；自采 MP4＋帧记录＋点位索引 | 人工构造文件测试通过；真实采集格式待核对 |
| 数据组织 | 按人划分、防重复／泄漏检查、等人物等帧数抽样、共同验证集 | 自动测试通过 |
| 预处理 | FAN＋MediaPipe Tasks 检测、关键点缓存、凸包掩码、失败清单 | 缓存及失败逻辑已测；真实检测器待资产与环境验证 |
| SMIRK | 官方重建／循环两路径训练、编码器与生成器加载、推理和可视化 | 接口按固定上游提交实现；未运行本体 |
| DECA | 官方 coarse 训练损失、同人 K 图像分组、推理和可视化 | 接口按固定上游提交实现；未运行本体；不含 detail 微调 |
| 实验管理 | 四组配置生成、顺序运行、验证选模、配置／数据哈希、优化器／调度器／随机状态保存 | CPU 小型替代网络测试通过，包括中断恢复与连续训练一致 |
| 三维评价 | 固定重心对应与校准变换、按 marker ID 配对、毫米点误差、覆盖率、失败率、人物等权汇总 | 数值测试通过；真实参考、单位、点位对应待核验 |
| 环境 | Windows 信息采集；WSL2 候选安装脚本；CUDA 与 PyTorch3D 梯度检查 | 当前 Mac CPU 测试；RTX 5080／WSL2 安装尚未验证 |

没有真实样本也能开发和测试上述逻辑。真实数据到位后重点核验文件字段、编号、同步、坐标与标定。自采32点不是 FAN 68 点或 MediaPipe 点，禁止按数组顺序替换。几何评价未校准时会报错，不输出伪造毫米结果。

## 1. 先在 Windows 收集环境信息

解压工程后，在 PowerShell 中进入工程目录，运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\environments\检查Windows环境.ps1
```

报告位于 `outputs/environment/windows.json`，包含系统、内存、GPU 显存、驱动、WSL 和磁盘空间。该脚本不安装或修改驱动。

## 2. 安装数据工具或准备模型环境

仅使用数据和评价工具：Python 3.10+，在工程目录执行：

```bash
python -m pip install -e .
python -m facebench --help
```

MP4 读取另装 `python -m pip install -e '.[video]'`。训练建议使用 WSL2 Ubuntu，两个模型分别建环境。候选脚本要求预先准备 Python 3.10、venv、Git、C++ 编译工具及可支持 CUDA 12.8 的 `nvcc`：

```bash
bash environments/bootstrap_wsl.sh smirk
bash environments/bootstrap_wsl.sh deca
```

脚本采用 PyTorch 2.7.1／CUDA 12.8 候选组合，并从源码编译 PyTorch3D，不沿用原项目 CUDA 11.x 二进制轮子。**该组合尚未在本机型实测，安装或模型运行报错应保留日志后修复，不视为已完成环境验收。**验证通过后保存 pip freeze 与 PyTorch3D 提交号，正式实验固定版本。

激活对应环境后检查：

```bash
source .venv-smirk/bin/activate
python -m facebench doctor --render-test --require-cuda --out outputs/environment/smirk.json
```

此检查包含 CUDA 张量反向与三角网格栅格化梯度；通过后还需验证完整模型。

## 3. 准备上游代码与模型资产

安装脚本会克隆固定提交的官方代码，也可以单独执行：

```bash
python -m facebench fetch-upstream --model smirk --destination third_party/smirk
python -m facebench fetch-upstream --model deca --destination third_party/deca
```

SMIRK：`c7de404c4389f073906a6db1adabf62efcea3f35`。

DECA：`a11554ae2a2b0f3998cf1fa94dd4db03babb34a2`。

按各项目官方流程获取本人有权使用的模型文件，保持官方目录结构：

- SMIRK：FLAME2020、SMIRK 完整编码器／生成器权重、面部标记嵌入、模板与掩码；训练另需表情模板和 VGG 感知损失资源。
- DECA：FLAME、DECA 权重、纹理／UV／模板资源；coarse 原损失还需 VGGFace2 身份模型。
- 预处理：MediaPipe `face_landmarker.task` 和 FAN 模型。

初版 SMIRK 配置关闭额外 emotion 与 MICA loss，保留重建／循环路径；不会启动原数据集的完整预训练流程。DECA 保留 coarse 默认损失。两者初版属于本实验微调适配，不能称作原论文全数据复现。

若已核实使用的是可信官方旧格式资产，可在本机配置中将 `trusted_legacy_assets` 设为 `true`，仅在上游调用的作用域内兼容旧 `torch.load` 默认行为。不要使用来源不明的 pickle 权重。

## 4. 建立原始数据索引

Multiface 先按官方 mini 配置检查下载和接入：

```bash
git clone https://github.com/facebookresearch/multiface third_party/multiface
# 先按该仓库说明准备下载器依赖；此操作会实际下载约16.2 GB。
python scripts/download_multiface_mini.py --repository third_party/multiface --destination data/raw/multiface-mini
python -m facebench index-multiface --root data/raw/multiface-mini --out data/multiface/raw.jsonl --frame-step 3
```

正式子集再选人物、表情、视角及配套网格／头姿。`--cameras`、`--sequences` 可筛选；`--require-mesh` 会拒绝缺失网格的样本。网格单位默认未知，**不会假定为毫米**。帧名存在间隔时不推断曝光时间。mini 若只有一人，无法划分未见人物训练／验证／测试集，只用于接入检查。

自采：复制 `configs/own_session.example.json`，填写当前采集目录与真实字段后执行：

```bash
python -m facebench index-own --spec configs/own_session.local.json --out data/own/raw.jsonl
```

每个 session 明确 RGB 视频、人物、任务、机位。时间来自 `frames.jsonl` 中真实记录，不以 MP4 标称帧率代替。点位可以暂缺；已有点位时声明哪一字段对应 RGB 帧索引，例如旧版单机重建的 `source_indices.rgb`。融合版本的字段嵌套不同，应按实际文件填写。多条点位记录对应同一 RGB 帧时会报错，需先确定选择策略。

## 5. 提取辅助标注并按人划分

```bash
python -m facebench preprocess --manifest data/own/raw.jsonl --out data/own/preprocessed --mediapipe-asset third_party/smirk/assets/face_landmarker.task --device cuda:0
python -m facebench preprocess --manifest data/multiface/raw.jsonl --out data/multiface/preprocessed --mediapipe-asset third_party/smirk/assets/face_landmarker.task --device cuda:0
```

预处理写出 `accepted.jsonl` 和 `failures.jsonl`；检测失败和多人图像不会被随机替换。初版生成的是 MediaPipe 面部**凸包掩码**，不是精细皮肤分割；DECA 此项与原版分割标注不同，需在实验报告注明。A/B 统一采用相同生成规则。

下面人数只是命令示例，需替换为实际人数，总和必须等于对应清单中身份总数：

```bash
python -m facebench split --manifest data/own/preprocessed/accepted.jsonl --out data/own/split --counts 6 2 2
python -m facebench split --manifest data/multiface/preprocessed/accepted.jsonl --out data/multiface/split --counts 9 2 2
python -m facebench merge --manifests data/own/split/val.jsonl data/multiface/split/val.jsonl --out data/common_val.jsonl
```

全部机位和片段按人物归入同一划分。使用 `subset` 匹配正式训练的身份数与有效帧数，重新指定输出清单：

```bash
python -m facebench subset --manifest data/own/split/train.jsonl --subjects 6 --frames-per-subject 1000 --seed 17 --out data/own/train_budget.jsonl
```

它是按现有帧的等额抽样，尚不自动匹配表情比例、有效时长或时间间隔；相邻帧冗余需在制定正式数据预算时处理。样本不足会报错，不隐藏重复采样。

## 6. 生成四组配置并先跑短训练

复制 `configs/local.example.json` 为 `configs/local.json`。全部相对路径以该 JSON 所在目录为基准；两个来源使用同一共同验证集。初版 `max_steps=100` 仅供短训练，正式预算需验证后另定。

```bash
python -m facebench plan --local configs/local.json --out configs/runs --seeds 17
python -m facebench check-run --config configs/runs/smirk_own_s17.json
python -m facebench train --config configs/runs/smirk_own_s17.json
```

DECA 组切换到 `.venv-deca`。运行前会检查数据可读、人物隔离、同人分组所需帧数和上游代码版本。SMIRK 默认只更新表情编码器与生成器，冻结身份／姿态分支及其 BN 缓冲；DECA 更新 coarse 编码器。固定总步数调度属于适配选择，均保存在配置中。

运行输出：`run.json`、`effective_upstream_config.txt`、`metrics.jsonl`、`validation_selection.jsonl`、`last.pt`、`best.pt`。

**初版最佳权重按共同验证集的二维关键点 MAE 选择**：先人物等权，再数据集等权。它不是三维精度或论文主指标；三维评价单独执行。每个验证人物默认抽固定8帧作为开发配置，可调整后冻结。训练不读取 test 清单。

恢复同一实验：

```bash
python -m facebench train --config configs/runs/smirk_own_s17.json --resume outputs/smirk_own_s17/last.pt
```

恢复会检查配置、训练／验证清单和初始权重的哈希，恢复优化器、调度器及随机状态。不会静默覆盖已有运行目录。当前固定单进程取样，无梯度累积或混合精度，batch 和显存需在5080实测后设定。

四组均跑通后，生成12次配置并顺序执行：

```bash
python -m facebench plan --local configs/local.json --out configs/formal --seeds 17 29 43
python scripts/run_plan.py configs/formal/plan.json
python scripts/run_plan.py configs/formal/plan.json --execute
```

默认分别使用 `.venv-smirk` 和 `.venv-deca` 的解释器。可用 `--python-smirk`、`--python-deca` 明确指定环境；遇到失败停止。正式运行前更换输出根目录与经过验证的训练预算。

## 7. 统一推理与三维评价

官方预训练参考组省略 `--checkpoint`；微调模型指定最佳权重。每个组分别在同一份 A-test、B-test 上运行：

```bash
python -m facebench infer --config configs/runs/smirk_own_s17.json --manifest data/own/split/test.jsonl --checkpoint outputs/smirk_own_s17/best.pt --out outputs/predictions/smirk_own_s17_A
```

输出每帧网格 NPZ、参数、输入／渲染／叠加图，以及成功与失败清单。顶点先标为 `model_native_unverified`；单目模型相机参数和 FLAME 原生空间不能自动当成实测世界坐标。初版保留这些信息供对齐，不宣称已恢复绝对头部平移。

自采原始观测可按明确的 track→F编号表导出：

```bash
python -m facebench export-own-points --manifest data/own/raw.jsonl --unit mm --coordinate-frame rig --out data/reference/own
```

对应单位／坐标必须已写入清单且与参数一致；目前此转换器只支持已核查的 `markers[].xyz_mm` 原始结构。未核实稳定编号时会拒绝，实际格式不同则补对应读取器。

复制 `configs/correspondence.example.json`，填入真实固定三角形顶点、重心坐标及个体校准变换，再将模型网格转换为相同语义的测量点。模板没有虚构对应，不能直接计分。Multiface 的网格与 F32 也不直接同拓扑，仍需单独建立参考映射；初版未提供自动稠密扫描评测。

```bash
python -m facebench project-points --manifest outputs/predictions/smirk_own_s17_A/predictions.jsonl --mapping configs/smirk_mapping.local.json --out outputs/aligned/smirk_own_s17_A
python -m facebench evaluate --reference data/reference/own/reference.jsonl --predictions outputs/aligned/smirk_own_s17_A/predicted_points.jsonl --protocol configs/evaluation_protocol.local.json --out outputs/results/smirk_own_s17_A.json
python -m facebench summarize --results outputs/results/smirk_own_s17_A.json --out outputs/results/comparison.csv
```

使用固定的个体中性校准，不对每个测试帧自由拟合后再声称绝对精度。评价包括：逐帧有效点误差、人物宏平均、覆盖率、包含失败的阈值成功率、全失败人物数量。低覆盖下的低误差不能单独作为好结果。速度与固定中性位移的计算函数位于 `evaluation.py`，使用前须确认共同头部坐标与真实时间间隔。

## 8. 自动验证与已知限制

```bash
python -m unittest discover -s tests -v
```

基础测试只依赖 NumPy／Pillow；安装 PyTorch 后增加训练状态测试。训练测试明确使用小型替代网络，验证调度、优化器和随机状态，不冒充两套人脸模型测试。所有本地验证中间文件保存于工作区的“工作暂存/代码开发验证/facebench”。

仍需目标机器与实际资产验证：5080／WSL2 环境安装、完整模型 GPU 前后向、真实 FAN／MediaPipe 检测、自采当前版本字段、多机同步与真实三维点对应。DECA detail 微调、自动三维配准、自动稠密表面误差和完整消融矩阵不在 v0.1 范围内。

## 官方依据

适配接口依据 [SMIRK 官方代码](https://github.com/georgeretsi/smirk) 与 [DECA 官方代码](https://github.com/yfeng95/DECA)；Multiface 目录依据[官方格式文档](https://github.com/facebookresearch/multiface/blob/main/documentation/DATA_STRUCTURE.md)。环境候选参考[PyTorch Blackwell 支持说明](https://pytorch.org/blog/pytorch-2-7/)、[PyTorch3D 安装说明](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)及[Microsoft WSL CUDA 说明](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)。
