# DSTP

面向 ViT-B/16 的独立训练工程，支持 FGVC 五个数据集和 VTAB-1k 全部 19 个任务。通用训练设施从原项目中提取，原目录保留不变。

目前提供 **普通 AdamW 全参数微调基线**，用于检查数据、训练及恢复流程。**DSTP 算法尚未实现**；后续可在 `dstp/baseline.py` 的 `build_optimizer` 及 `dstp/model.py` 中接入新方法。学习率统一从 `2e-4` 起步，FGVC 为 30 轮含 3 轮 warmup，VTAB 为 50 轮含 5 轮 warmup；这些是工程起始配置，不代表新算法的推荐参数或实验结论。

## 目录

```text
dstp/             ViT 加载、数据加载、训练、评估、恢复、AdamW 基线
configs/fgvc/     CUB、NABirds、Flowers、Dogs、Cars 配置
configs/vtab/     VTAB 19 个任务的配置
configs/shared_seeds.json  共享种子 42、43、44
scripts/FGVC/     单任务、全部任务、多个种子和数据检查入口
scripts/VTAB/     单任务、全部任务、多个种子和数据检查入口
data_splits/      15 个 FGVC JSON、76 个 VTAB TXT 划分文件
tests/            训练恢复与目录兼容性测试
```

工程不携带图片、数据压缩包、模型权重、虚拟环境、旧日志或旧结果，也未迁入旧算法的优化器、评分、稀疏参数划分、搜索、消融和分割/检测工程。

## 外部资源

配置继续使用迁移前的绝对路径，数据和预训练权重均在原位置读取：

| 资源 | 配置路径 | 当前实际位置 |
| --- | --- | --- |
| FGVC | `/home/pc/data/PEFT-method/GPS-main/datasets/FGVC` | `/home/pc/data/PEFT-method/MCST/datasets/FGVC` |
| VTAB-1k | `/home/pc/data/PEFT-method/MCST/datasets/vtab-1k` | 同左 |
| ViT NPZ | `/home/pc/data/PEFT-method/pretrained-model/VFM/ViT/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz` | `/home/pc/data/PEFT-method/MCST/model/vit/` 中的同名文件 |

FGVC 和权重的配置路径通过已有符号链接访问实际位置。训练代码不导入原项目或 GPS 工程的 Python 模块；外部 Python 包见 `requirements.txt`。

默认读取原数据目录中的划分文件；`data_splits/` 保留相同的划分副本。用 `--split-root data_splits/FGVC` 或 `--split-root data_splits/vtab-1k` 可改为读取项目内的划分，图片仍从 `--data-root` 读取。相对的 `--split-root` 以本项目根目录为基准。

VTAB 优先使用本地的 `cifar100`、`flowers102`、`pets` 目录；同时兼容 `cifar`、`oxford_flowers102`、`oxford_iiit_pet` 别名。

## 使用

默认解释器为现有 `/home/pc/anaconda3/envs/GPS/bin/python`，可设置 `PYTHON_BIN` 覆盖；不自动安装依赖或下载模型。

```bash
cd /home/pc/data/Project/DSTP

# 只解析配置，不训练。
bash scripts/FGVC/cub.sh --print-config
bash scripts/VTAB/cifar100.sh --print-config

# 检查所有数据路径、划分、类别及首张图像。
bash scripts/FGVC/check_data.sh --verify-files
bash scripts/VTAB/check_data.sh --verify-files

# 显式指定可见 GPU 后，启动单任务基线训练。
CUDA_VISIBLE_DEVICES=0 bash scripts/FGVC/cub.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/VTAB/cifar100.sh

# 顺序运行同一套数据集；不自动分配 GPU。
CUDA_VISIBLE_DEVICES=0 bash scripts/FGVC/run_all.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/VTAB/run_all.sh

# 预览多个种子的命令；去掉 DRY_RUN=1 后才会训练。
DRY_RUN=1 bash scripts/FGVC/run_seeds.sh cub
DRY_RUN=1 bash scripts/VTAB/run_seeds.sh cifar100

# 使用新项目内的划分副本、原位置的图像。
bash scripts/FGVC/cub.sh --split-root data_splits/FGVC --print-config

# 恢复本项目基线的 checkpoint；继续使用原来的训练配置。
CUDA_VISIBLE_DEVICES=0 bash scripts/FGVC/cub.sh --resume /absolute/run/last.pt

# CPU 小模型测试；临时数据和输出写入系统临时目录。
/home/pc/anaconda3/envs/GPS/bin/python -m unittest discover -s tests -v
```

所有任务脚本都转发尾部 CLI 参数；`DATA_ROOT`、`SPLIT_ROOT`、`PRETRAINED`、`LR`、`HEAD_LR`、`BATCH_SIZE`、`SEED`、`RUN_DIR`、`RESUME` 等环境变量也可覆盖对应参数。优先级为公共默认值、数据集 JSON、按顺序的 `--config`、CLI；脚本环境变量先转为 CLI，用户尾部参数优先。

保留原数据评估流程：FGVC 合并 train/val 训练；VTAB 使用 `train800val200.txt` 的 1000 张图训练；每轮评估 test，以 test Top-1 保存 `best.pt`。FGVC 默认启用 EMA，VTAB 默认关闭；启用时按 EMA Top-1 选取。该指标属于 best-test 选择口径。新运行输出写入本项目 `result/`，不会写入原工程；旧工程 checkpoint 不在本基线的恢复范围内。

当前工程准备完成后没有启动正式 GPU 训练。
