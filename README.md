# IRAUL · Weibo21 参数敏感性实验

**IRAUL: Inductive Relation-Aware Evidence Utility Learning for Multimodal Fake News Detection**

本仓库可以在任意 Linux GPU 服务器上，从零完整复现 IRAUL（7B Full）在 **Weibo21full_version** 数据集上的参数敏感性实验。流程是：先训练 reference，再训练学生，最后做一次 TEST，生成 3×3 敏感性矩阵。实验配置与 A10 服务器上的论文运行完全一致。

```bash
python train.py --data /path/to/Weibo21full_version
```

---

## 1. 实验内容

扫描三个超参数，每次只改动默认配置中的对应项，其余配置都与论文默认点相同：

- **τ**：`model.r1.tau`，效用 softmax 的温度
- **λrel**：`loss.lambda_relation`，关系监督损失权重
- **λutil**：`loss.lambda_utility`，效用监督损失权重

**3×3 矩阵：λrel × λutil（τ = 1.0）**

| λrel ＼ λutil | 0.5 | 1.0 | 2.0 |
|---|---|---|---|
| **0.1** | rel_0p1_util_0p5 | rel_0p1 | rel_0p1_util_2p0 |
| **0.2** | util_0p5 | **default（默认点）** | util_2p0 |
| **0.4** | rel_0p4_util_0p5 | rel_0p4 | rel_0p4_util_2p0 |

**τ 行（λrel = 0.2，λutil = 1.0）**：tau_0p5、default（τ=1.0）、tau_2p0。

规模如下：

- 共 11 组设置 × 5 个种子（200404、200405、200406、200407、200408）= **55 个学生任务**。
- 每个种子先训练 1 份独立的三折 reference，共 **5 个 reference 任务**；同一种子的 11 个学生共用它。reference 只依赖数据和种子，不依赖 τ、λrel 和 λutil。
- 网格定义在 `configs/grid.json`，与 GossipCop 敏感性实验使用同一组取值。

## 2. 目录结构

```
train.py                    唯一入口：检查 → reference → 学生 → TEST → 汇总
configs/grid.json           11 组设置与 5 个种子
configs/templates/          A10 论文运行的 reference / 学生配置（路径在运行时填写）
experiment/job.py           单个任务的执行器（reference / train / test）
experiment/collect.py       汇总 results.csv、matrix.md 与结果包
iraul/                      IRAUL 源码（与 A10 运行逐字节一致，未做修改）
scripts/download_models.sh  按固定 revision 下载两个预训练骨干并校验哈希
scripts/models.sha256       骨干模型文件的 sha256
requirements.txt            与 A10 完全相同的 Python 依赖
```

## 3. 环境

- Linux，NVIDIA GPU：单卡显存 ≥ 24 GB；推荐 H20/A100/H100。
- 驱动支持 CUDA 12.4（驱动版本 ≥ 550）。
- Python 3.10。

```bash
conda create -y -n iraul python=3.10
conda activate iraul
pip install -r requirements.txt
```

`train.py` 开跑前会核对 torch 2.4.1+cu124、transformers 4.46.3、peft 0.13.2、Pillow 12.3.0、tokenizers 0.20.3。版本不同会直接退出；如果确实要用别的版本，需要显式加 `--allow-other-versions`。

## 4. 预训练模型

```bash
bash scripts/download_models.sh
```

脚本会下载到 `./models/`，然后逐个文件核对 sha256。国内网络可以先执行 `export HF_ENDPOINT=https://hf-mirror.com`。

| 模型 | Hugging Face | 固定 revision |
|---|---|---|
| 文本骨干 | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` |
| 视觉骨干 | `google/siglip-base-patch16-224` | `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` |

模型放在别处时，运行时用 `--models /path/to/models` 指定；该目录下需要有 `Qwen2.5-7B-Instruct/` 和 `siglip-base-patch16-224/`。

## 5. 数据集

数据集单独提供：`Weibo21full_version.zip`（1.5 GB，9600 个文件，已清洗，可直接运行）。

zip 的 sha256 为 `0d2d1d409ef0e67bb7a626fefc18bbb4bd4aa91de2169fc01761b9d3ae23d861`，下载后可以用 `sha256sum Weibo21full_version.zip` 核对。

解压后把目录路径传给 `--data`：

```bash
unzip Weibo21full_version.zip -d /path/to/data
python train.py --data /path/to/data/Weibo21full_version
```

解压后的目录结构：

```
Weibo21full_version/
  fake_release_all_picture/   real_release_all_picture/    图片
  fake_release_all.json       real_release_all.json        原始数据
  splits_811/train.jsonl  val.jsonl  test.jsonl            固定 8:1:1 划分（图片为相对路径）
  splits_811/dataset_manifest.json                         划分清单（image_root 为 ..）
  README_IRAUL.md                                          数据集说明
```

| 集合 | 条数 | 假新闻 | 有图样本 |
|---|---|---|---|
| train | 7302 | 3590 | 5046 |
| val | 913 | 449 | 643 |
| test | 913 | 449 | 632 |

划分说明：

- 原始数据没有官方划分。这里用固定种子 200404 做分组分层 80/10/10 划分：同一事件、同一来源、近重复文本放在同一个集合里。
- 保留全部 9128 条记录，包括无图样本。缺图样本通过可用性 mask 处理，不伪造图片。
- 每张图片的 sha256 记录在清单里，`train.py` 开跑前会逐张核对。

## 6. 运行

建议在 `tmux` 或 `screen` 里运行：

```bash
python train.py --data /path/to/Weibo21full_version
```

实验流程：

1. **检查**：依赖版本、模型文件、划分文件哈希、全部图片哈希。
2. **reference**：5 个种子各训练一份三折 reference（`iraul/scripts/continue_masked_r1_reference.py`），输出 `outputs/references/seed<S>/targets.pt`。
3. **学生训练**：某个种子的 reference 一完成，该种子的 11 个学生就开始排队。每个学生调用原始 `iraul/train.py`，并使用同种子 reference 的 OOF 目标。
4. **选择与冻结**：在验证集上按 Macro-F1 取 top-5 检查点的平均模型，并在验证集上选定阈值，然后冻结。
5. **TEST**：用冻结的模型和阈值在 913 条测试样本上测试一次，测试集不参与任何选择。
6. **汇总**：每完成一个任务，就更新 `outputs/results.csv`、`outputs/matrix.md` 和 `outputs/results_bundle.tar.gz`。

默认使用所有可见 GPU，每卡并行 3 个任务。中途中断（Ctrl-C、断线、重启）后，再执行同一条命令即可：

- 学生从最近的断点继续（每 20 步保存一次）；
- reference 从已完成的折继续；
- 已测试的任务不会重跑，也不会重复读取 TEST。

| 参数 | 作用 |
|---|---|
| `--data DIR` | 解压后的 Weibo21full_version 目录（必填） |
| `--models DIR` | 预训练模型目录（默认 `./models`） |
| `--out DIR` | 输出目录（默认 `./outputs`）。reference 和学生必须在同一台机器、同一输出目录下运行 |
| `--gpus 0,1,2,3` | 指定 GPU |
| `--procs-per-gpu 3` | 每卡并行任务数。单个任务峰值不超过 24 GB；96 GB 的卡用 3，40 GB 的卡用 1 |
| `--settings default,rel_0p4` | 只跑部分设置（名称见 `configs/grid.json`） |
| `--seeds 200404,200405` | 只跑部分种子 |
| `--num-workers N` | 每个任务的 DataLoader 进程数（默认 2，与 A10 一致；不影响结果） |
| `--no-prune` | TEST 后保留断点和 top-k 权重。默认只保留实际测试的 `final_averaged.pth` 和 `best.pth`，每个任务约 0.2 GB |
| `--skip-data-check` | 跳过图片哈希检查（已通过时） |
| `--collect-only` | 只重新汇总结果 |

## 7. 输出

```
outputs/
  references/seed<S>/        三折 reference（targets.pt、fold_*.pth、reference_manifest.json）
  runs/<setting>/seed<S>/    学生运行：checkpoints/、history.json、frozen.json、evaluation/final/test/
  configs/                   每个任务实际使用的配置
  logs/                      每个任务的日志
  status.json                实时进度
  results.csv                每个任务一行：TEST Acc / Macro-F1 / AUC / fake F1 / real F1 / NLL / Brier / ECE、阈值、验证集最佳 F1、训练轮数
  matrix.md, matrix.csv      3×3 矩阵与 τ 行：5 个种子均值 ± 样本标准差，括号内为已完成种子数
  results_bundle.tar.gz      汇总表 + 每个任务的指标、逐条预测、冻结记录、训练历史与配置（几 MB，便于转交统计）
```

## 8. 实验配置（与 A10 论文运行相同）

| 项目 | 取值 |
|---|---|
| 文本骨干 | Qwen2.5-7B-Instruct（bf16），LoRA r=8、α=16、dropout 0.05，作用于 q_proj / v_proj |
| 视觉骨干 | SigLIP base patch16-224，只微调最后 2 层 |
| 证据维度 | D=384，8 头；关系判定使用共享 Qwen 的最后 2 层，严格可见性 mask |
| 视图 | single_view，alternate_epoch（每条记录每轮使用一个版本） |
| batch | 微批 2 × 梯度累积 8 = 有效 batch 16，单卡；Qwen 梯度检查点开启 |
| 优化 | LoRA 学习率 1e-4，视觉 2e-5，head 4e-5，weight decay 0.02，warmup 10%，梯度裁剪 1.0 |
| 训练 | 最多 30 轮，早停耐心 8 轮（按验证集 Macro-F1），top-5 检查点平均 |
| 损失 | λrel=0.2、λutil=1.0（默认），EDL β=0.01，EDL warmup 3 轮，效用目标为 Shapley |
| 阈值 | 在验证集 [0.2, 0.8] 范围内以 0.01 为步长选取（目标 Macro-F1），然后冻结 |
| reference | 三折（分组分层），每折 3 轮，仅从预训练初始化，覆盖 clean 与 aug1 两个版本 |
| 种子 | 200404–200408，每个种子独立训练 reference |

- **配置模板来源**：`configs/templates/` 就是 A10 上种子 200404 的配置。各种子之间只有 `seed` 和 `data.replay_seed` 不同，由 `experiment/job.py` 自动填写。
- **源码来源**：`iraul/` 是 A10 运行所用源码的原样拷贝。

## 9. 资源与用时

- A10 实测：每个 reference 约 3.4 小时；每个学生约 6–9 小时，大多在第 20–27 轮早停，每轮约 18 分钟。合计约 430 A10 GPU 小时。
- 4 × H20、每卡 3 个任务并行：预计约 2 天，以第一批任务的实际速度为准。
- 磁盘：模型 16 GB，数据 1.5 GB，输出约 15 GB（默认清理）。

## 10. 常见问题

- **`... differs from the published 8:1:1 split` 或图片哈希不一致**：`--data` 没有指向解压后的 `Weibo21full_version` 目录，或者数据文件被改动过。重新解压 zip 即可。
- **`reference of seed ... is not complete`**：该种子的 reference 还没训完，或者训练失败，请查看 `outputs/logs/reference_seed<S>.reference.log`。
- **TEST 中途失败后不自动重测**：这是有意设计，防止反复读取测试集。请人工检查原因后，删除对应的 `evaluation/final/test/test_started.json` 再重跑。
- **显存不足**：把 `--procs-per-gpu` 调小。
