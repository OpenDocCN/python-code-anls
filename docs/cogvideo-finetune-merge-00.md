# CogVideo & CogVideoX 微调代码源码解析（一）



#  Raise valuable PR / 提出有价值的PR

## Caution / 注意事项:
Users should keep the following points in mind when submitting PRs:

1. Ensure that your code meets the requirements in the [specification](../../resources/contribute.md).
2. the proposed PR should be relevant, if there are multiple ideas and optimizations, they should be assigned to different PRs.

用户在提交PR时候应该注意以下几点:

1. 确保您的代码符合 [规范](../../resources/contribute_zh.md) 中的要求。
2. 提出的PR应该具有针对性，如果具有多个不同的想法和优化方案，应该分配到不同的PR中。

## 不应该提出的PR / PRs that should not be proposed

If a developer proposes a PR about any of the following, it may be closed or Rejected.

1. those that don't describe improvement options.
2. multiple issues of different types combined in one PR.
3. The proposed PR is highly duplicative of already existing PRs.

如果开发者提出关于以下方面的PR，则可能会被直接关闭或拒绝通过。

1. 没有说明改进方案的。
2. 多个不同类型的问题合并在一个PR中的。
3. 提出的PR与已经存在的PR高度重复的。


# 检查您的PR
- [ ] Have you read the Contributor Guidelines, Pull Request section? / 您是否阅读了贡献者指南、Pull Request 部分？
- [ ] Has this been discussed/approved via a Github issue or forum? If so, add a link. / 是否通过 Github 问题或论坛讨论/批准过？如果是，请添加链接。
- [ ] Did you make sure you updated the documentation with your changes? Here are the Documentation Guidelines, and here are the Documentation Formatting Tips. /您是否确保根据您的更改更新了文档？这里是文档指南，这里是文档格式化技巧。
- [ ] Did you write new required tests? / 您是否编写了新的必要测试？
- [ ]  Are your PRs for only one issue / 您的PR是否仅针对一个问题

# CogVideoX diffusers Fine-tuning Guide

[中文阅读](./README_zh.md)

[日本語で読む](./README_ja.md)

This feature is not fully complete yet. If you want to check the fine-tuning for the SAT version, please
see [here](../sat/README_zh.md). The dataset format is different from this version.

## Hardware Requirements

+ CogVideoX-2B / 5B LoRA: 1 * A100 (5B need to use `--use_8bit_adam`)
+ CogVideoX-2B SFT:  8 * A100 (Working)
+ CogVideoX-5B-I2V is not supported yet.

## Install Dependencies

Since the related code has not been merged into the diffusers release, you need to base your fine-tuning on the
diffusers branch. Please follow the steps below to install dependencies:

```py
git clone https://github.com/huggingface/diffusers.git
cd diffusers # Now in Main branch
pip install -e .
```

## Prepare the Dataset

First, you need to prepare the dataset. The dataset format should be as follows, with `videos.txt` containing the list
of videos in the `videos` directory:

```py
.
├── prompts.txt
├── videos
└── videos.txt
```

You can download
the [Disney Steamboat Willie](https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset) dataset from
here.

This video fine-tuning dataset is used as a test for fine-tuning.

## Configuration Files and Execution

The `accelerate` configuration files are as follows:

+ `accelerate_config_machine_multi.yaml`: Suitable for multi-GPU use
+ `accelerate_config_machine_single.yaml`: Suitable for single-GPU use

The configuration for the `finetune` script is as follows:

```py
accelerate launch --config_file accelerate_config_machine_single.yaml --multi_gpu \  # Use accelerate to launch multi-GPU training with the config file accelerate_config_machine_single.yaml
  train_cogvideox_lora.py \  # Training script train_cogvideox_lora.py for LoRA fine-tuning on CogVideoX model
  --gradient_checkpointing \  # Enable gradient checkpointing to reduce memory usage
  --pretrained_model_name_or_path $MODEL_PATH \  # Path to the pretrained model, specified by $MODEL_PATH
  --cache_dir $CACHE_PATH \  # Cache directory for model files, specified by $CACHE_PATH
  --enable_tiling \  # Enable tiling technique to process videos in chunks, saving memory
  --enable_slicing \  # Enable slicing to further optimize memory by slicing inputs
  --instance_data_root $DATASET_PATH \  # Dataset path specified by $DATASET_PATH
  --caption_column prompts.txt \  # Specify the file prompts.txt for video descriptions used in training
  --video_column videos.txt \  # Specify the file videos.txt for video paths used in training
  --validation_prompt "" \  # Prompt used for generating validation videos during training
  --validation_prompt_separator ::: \  # Set ::: as the separator for validation prompts
  --num_validation_videos 1 \  # Generate 1 validation video per validation round
  --validation_epochs 100 \  # Perform validation every 100 training epochs
  --seed 42 \  # Set random seed to 42 for reproducibility
  --rank 128 \  # Set the rank for LoRA parameters to 128
  --lora_alpha 64 \  # Set the alpha parameter for LoRA to 64, adjusting LoRA learning rate
  --mixed_precision bf16 \  # Use bf16 mixed precision for training to save memory
  --output_dir $OUTPUT_PATH \  # Specify the output directory for the model, defined by $OUTPUT_PATH
  --height 480 \  # Set video height to 480 pixels
  --width 720 \  # Set video width to 720 pixels
  --fps 8 \  # Set video frame rate to 8 frames per second
  --max_num_frames 49 \  # Set the maximum number of frames per video to 49
  --skip_frames_start 0 \  # Skip 0 frames at the start of the video
  --skip_frames_end 0 \  # Skip 0 frames at the end of the video
  --train_batch_size 4 \  # Set training batch size to 4
  --num_train_epochs 30 \  # Total number of training epochs set to 30
  --checkpointing_steps 1000 \  # Save model checkpoint every 1000 steps
  --gradient_accumulation_steps 1 \  # Accumulate gradients for 1 step, updating after each batch
  --learning_rate 1e-3 \  # Set learning rate to 0.001
  --lr_scheduler cosine_with_restarts \  # Use cosine learning rate scheduler with restarts
  --lr_warmup_steps 200 \  # Warm up the learning rate for the first 200 steps
  --lr_num_cycles 1 \  # Set the number of learning rate cycles to 1
  --optimizer AdamW \  # Use the AdamW optimizer
  --adam_beta1 0.9 \  # Set Adam optimizer beta1 parameter to 0.9
  --adam_beta2 0.95 \  # Set Adam optimizer beta2 parameter to 0.95
  --max_grad_norm 1.0 \  # Set maximum gradient clipping value to 1.0
  --allow_tf32 \  # Enable TF32 to speed up training
  --report_to wandb  # Use Weights and Biases (wandb) for logging and monitoring the training
```

## Running the Script to Start Fine-tuning

Single Node (One GPU or Multi GPU) fine-tuning:

```py
bash finetune_single_rank.sh
```

Multi-Node fine-tuning:

```py
bash finetune_multi_rank.sh # Needs to be run on each node
```

## Loading the Fine-tuned Model

+ Please refer to [cli_demo.py](../inference/cli_demo.py) for how to load the fine-tuned model.

## Best Practices

+ Includes 70 training videos with a resolution of `200 x 480 x 720` (frames x height x width). By skipping frames in
  the data preprocessing, we created two smaller datasets with 49 and 16 frames to speed up experimentation, as the
  maximum frame limit recommended by the CogVideoX team is 49 frames. We split the 70 videos into three groups of 10,
  25, and 50 videos, with similar conceptual nature.
+ Using 25 or more videos works best when training new concepts and styles.
+ It works better to train using identifier tokens specified with `--id_token`. This is similar to Dreambooth training,
  but regular fine-tuning without such tokens also works.
+ The original repository used `lora_alpha` set to 1. We found this value ineffective across multiple runs, likely due
  to differences in the backend and training setup. Our recommendation is to set `lora_alpha` equal to rank or rank //
    2.
+ We recommend using a rank of 64 or higher.


# CogVideoX diffusers 微調整方法

[Read this in English.](./README_zh)

[中文阅读](./README_zh.md)


この機能はまだ完全に完成していません。SATバージョンの微調整を確認したい場合は、[こちら](../sat/README_ja.md)を参照してください。本バージョンとは異なるデータセット形式を使用しています。

## ハードウェア要件

+ CogVideoX-2B / 5B T2V LORA: 1 * A100  (5B need to use `--use_8bit_adam`)
+ CogVideoX-2B SFT:  8 * A100 （動作確認済み）
+ CogVideoX-5B-I2V まだサポートしていません

## 依存関係のインストール

関連コードはまだdiffusersのリリース版に統合されていないため、diffusersブランチを使用して微調整を行う必要があります。以下の手順に従って依存関係をインストールしてください：

```py
git clone https://github.com/huggingface/diffusers.git
cd diffusers # Now in Main branch
pip install -e .
```

## データセットの準備

まず、データセットを準備する必要があります。データセットの形式は以下のようになります。

```py
.
├── prompts.txt
├── videos
└── videos.txt
```

[ディズニースチームボートウィリー](https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset)をここからダウンロードできます。

ビデオ微調整データセットはテスト用として使用されます。

## 設定ファイルと実行

`accelerate` 設定ファイルは以下の通りです:

+ accelerate_config_machine_multi.yaml 複数GPU向け
+ accelerate_config_machine_single.yaml 単一GPU向け

`finetune` スクリプト設定ファイルの例：

```py
accelerate launch --config_file accelerate_config_machine_single.yaml --multi_gpu \  # accelerateを使用してmulti-GPUトレーニングを起動、設定ファイルはaccelerate_config_machine_single.yaml
  train_cogvideox_lora.py \  # LoRAの微調整用のトレーニングスクリプトtrain_cogvideox_lora.pyを実行
  --gradient_checkpointing \  # メモリ使用量を減らすためにgradient checkpointingを有効化
  --pretrained_model_name_or_path $MODEL_PATH \  # 事前学習済みモデルのパスを$MODEL_PATHで指定
  --cache_dir $CACHE_PATH \  # モデルファイルのキャッシュディレクトリを$CACHE_PATHで指定
  --enable_tiling \  # メモリ節約のためにタイル処理を有効化し、動画をチャンク分けして処理
  --enable_slicing \  # 入力をスライスしてさらにメモリ最適化
  --instance_data_root $DATASET_PATH \  # データセットのパスを$DATASET_PATHで指定
  --caption_column prompts.txt \  # トレーニングで使用する動画の説明ファイルをprompts.txtで指定
  --video_column videos.txt \  # トレーニングで使用する動画のパスファイルをvideos.txtで指定
  --validation_prompt "" \  # トレーニング中に検証用の動画を生成する際のプロンプト
  --validation_prompt_separator ::: \  # 検証プロンプトの区切り文字を:::に設定
  --num_validation_videos 1 \  # 各検証ラウンドで1本の動画を生成
  --validation_epochs 100 \  # 100エポックごとに検証を実施
  --seed 42 \  # 再現性を保証するためにランダムシードを42に設定
  --rank 128 \  # LoRAのパラメータのランクを128に設定
  --lora_alpha 64 \  # LoRAのalphaパラメータを64に設定し、LoRAの学習率を調整
  --mixed_precision bf16 \  # bf16混合精度でトレーニングし、メモリを節約
  --output_dir $OUTPUT_PATH \  # モデルの出力ディレクトリを$OUTPUT_PATHで指定
  --height 480 \  # 動画の高さを480ピクセルに設定
  --width 720 \  # 動画の幅を720ピクセルに設定
  --fps 8 \  # 動画のフレームレートを1秒あたり8フレームに設定
  --max_num_frames 49 \  # 各動画の最大フレーム数を49に設定
  --skip_frames_start 0 \  # 動画の最初のフレームを0スキップ
  --skip_frames_end 0 \  # 動画の最後のフレームを0スキップ
  --train_batch_size 4 \  # トレーニングのバッチサイズを4に設定
  --num_train_epochs 30 \  # 総トレーニングエポック数を30に設定
  --checkpointing_steps 1000 \  # 1000ステップごとにモデルのチェックポイントを保存
  --gradient_accumulation_steps 1 \  # 1ステップの勾配累積を行い、各バッチ後に更新
  --learning_rate 1e-3 \  # 学習率を0.001に設定
  --lr_scheduler cosine_with_restarts \  # リスタート付きのコサイン学習率スケジューラを使用
  --lr_warmup_steps 200 \  # トレーニングの最初の200ステップで学習率をウォームアップ
  --lr_num_cycles 1 \  # 学習率のサイクル数を1に設定
  --optimizer AdamW \  # AdamWオプティマイザーを使用
  --adam_beta1 0.9 \  # Adamオプティマイザーのbeta1パラメータを0.9に設定
  --adam_beta2 0.95 \  # Adamオプティマイザーのbeta2パラメータを0.95に設定
  --max_grad_norm 1.0 \  # 勾配クリッピングの最大値を1.0に設定
  --allow_tf32 \  # トレーニングを高速化するためにTF32を有効化
  --report_to wandb  # Weights and Biasesを使用してトレーニングの記録とモニタリングを行う
```

## 微調整を開始

単一マシン (シングルGPU、マルチGPU) での微調整:

```py
bash finetune_single_rank.sh
```

複数マシン・マルチGPUでの微調整：

```py
bash finetune_multi_rank.sh # 各ノードで実行する必要があります。
```

## 微調整済みモデルのロード

+ 微調整済みのモデルをロードする方法については、[cli_demo.py](../inference/cli_demo.py) を参照してください。

## ベストプラクティス

+ 解像度が `200 x 480 x 720`（フレーム数 x 高さ x 幅）のトレーニングビデオが70本含まれています。データ前処理でフレームをスキップすることで、49フレームと16フレームの小さなデータセットを作成しました。これは実験を加速するためのもので、CogVideoXチームが推奨する最大フレーム数制限は49フレームです。
+ 25本以上のビデオが新しい概念やスタイルのトレーニングに最適です。
+ 現在、`--id_token` を指定して識別トークンを使用してトレーニングする方が効果的です。これはDreamboothトレーニングに似ていますが、通常の微調整でも機能します。
+ 元のリポジトリでは `lora_alpha` を1に設定していましたが、複数の実行でこの値が効果的でないことがわかりました。モデルのバックエンドやトレーニング設定によるかもしれません。私たちの提案は、lora_alphaをrankと同じか、rank // 2に設定することです。
+ Rank 64以上の設定を推奨します。


# CogVideoX diffusers 微调方案

[Read this in English](./README_zh.md)

[日本語で読む](./README_ja.md)

本功能尚未完全完善，如果您想查看SAT版本微调，请查看[这里](../sat/README_zh.md)。其数据集格式与本版本不同。

## 硬件要求

+ CogVideoX-2B / 5B T2V LORA: 1 * A100  (5B need to use `--use_8bit_adam`)
+ CogVideoX-2B SFT:  8 * A100 (制作中)
+ CogVideoX-5B-I2V 暂未支持

## 安装依赖

由于相关代码还没有被合并到diffusers发行版，你需要基于diffusers分支进行微调。请按照以下步骤安装依赖：

```py
git clone https://github.com/huggingface/diffusers.git
cd diffusers # Now in Main branch
pip install -e .
```

## 准备数据集

首先，你需要准备数据集，数据集格式如下，其中，videos.txt 存放 videos 中的视频。

```py
.
├── prompts.txt
├── videos
└── videos.txt
```

你可以从这里下载 [迪士尼汽船威利号](https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset)

视频微调数据集作为测试微调。

## 配置文件和运行

`accelerate` 配置文件如下:

+ accelerate_config_machine_multi.yaml 适合多GPU使用
+ accelerate_config_machine_single.yaml 适合单GPU使用

`finetune` 脚本配置文件如下：

```py

accelerate launch --config_file accelerate_config_machine_single.yaml --multi_gpu \  # 使用 accelerate 启动多GPU训练，配置文件为 accelerate_config_machine_single.yaml
  train_cogvideox_lora.py \  # 运行的训练脚本为 train_cogvideox_lora.py，用于在 CogVideoX 模型上进行 LoRA 微调
  --gradient_checkpointing \  # 启用梯度检查点功能，以减少显存使用
  --pretrained_model_name_or_path $MODEL_PATH \  # 预训练模型路径，通过 $MODEL_PATH 指定
  --cache_dir $CACHE_PATH \  # 模型缓存路径，由 $CACHE_PATH 指定
  --enable_tiling \  # 启用tiling技术，以分片处理视频，节省显存
  --enable_slicing \  # 启用slicing技术，将输入切片，以进一步优化内存
  --instance_data_root $DATASET_PATH \  # 数据集路径，由 $DATASET_PATH 指定
  --caption_column prompts.txt \  # 指定用于训练的视频描述文件，文件名为 prompts.txt
  --video_column videos.txt \  # 指定用于训练的视频路径文件，文件名为 videos.txt
  --validation_prompt "" \  # 验证集的提示语 (prompt)，用于在训练期间生成验证视频
  --validation_prompt_separator ::: \  # 设置验证提示语的分隔符为 :::
  --num_validation_videos 1 \  # 每个验证回合生成 1 个视频
  --validation_epochs 100 \  # 每 100 个训练epoch进行一次验证
  --seed 42 \  # 设置随机种子为 42，以保证结果的可复现性
  --rank 128 \  # 设置 LoRA 参数的秩 (rank) 为 128
  --lora_alpha 64 \  # 设置 LoRA 的 alpha 参数为 64，用于调整LoRA的学习率
  --mixed_precision bf16 \  # 使用 bf16 混合精度进行训练，减少显存使用
  --output_dir $OUTPUT_PATH \  # 指定模型输出目录，由 $OUTPUT_PATH 定义
  --height 480 \  # 视频高度为 480 像素
  --width 720 \  # 视频宽度为 720 像素
  --fps 8 \  # 视频帧率设置为 8 帧每秒
  --max_num_frames 49 \  # 每个视频的最大帧数为 49 帧
  --skip_frames_start 0 \  # 跳过视频开头的帧数为 0
  --skip_frames_end 0 \  # 跳过视频结尾的帧数为 0
  --train_batch_size 4 \  # 训练时的 batch size 设置为 4
  --num_train_epochs 30 \  # 总训练epoch数为 30
  --checkpointing_steps 1000 \  # 每 1000 步保存一次模型检查点
  --gradient_accumulation_steps 1 \  # 梯度累计步数为 1，即每个 batch 后都会更新梯度
  --learning_rate 1e-3 \  # 学习率设置为 0.001
  --lr_scheduler cosine_with_restarts \  # 使用带重启的余弦学习率调度器
  --lr_warmup_steps 200 \  # 在训练的前 200 步进行学习率预热
  --lr_num_cycles 1 \  # 学习率周期设置为 1
  --optimizer AdamW \  # 使用 AdamW 优化器
  --adam_beta1 0.9 \  # 设置 Adam 优化器的 beta1 参数为 0.9
  --adam_beta2 0.95 \  # 设置 Adam 优化器的 beta2 参数为 0.95
  --max_grad_norm 1.0 \  # 最大梯度裁剪值设置为 1.0
  --allow_tf32 \  # 启用 TF32 以加速训练
  --report_to wandb  # 使用 Weights and Biases 进行训练记录与监控
```

## 运行脚本，开始微调

单机(单卡，多卡)微调：

```py
bash finetune_single_rank.sh
```

多机多卡微调：

```py
bash finetune_multi_rank.sh #需要在每个节点运行
```

## 载入微调的模型

+ 请关注[cli_demo.py](../inference/cli_demo.py) 以了解如何加载微调的模型。

## 最佳实践

+ 包含70个分辨率为 `200 x 480 x 720`（帧数 x 高 x
  宽）的训练视频。通过数据预处理中的帧跳过，我们创建了两个较小的49帧和16帧数据集，以加快实验速度，因为CogVideoX团队建议的最大帧数限制是49帧。我们将70个视频分成三组，分别为10、25和50个视频。这些视频的概念性质相似。
+ 25个及以上的视频在训练新概念和风格时效果最佳。
+ 现使用可以通过 `--id_token` 指定的标识符token进行训练效果更好。这类似于 Dreambooth 训练，但不使用这种token的常规微调也可以工作。
+ 原始仓库使用 `lora_alpha` 设置为 1。我们发现这个值在多次运行中效果不佳，可能是因为模型后端和训练设置的不同。我们的建议是将
  lora_alpha 设置为与 rank 相同或 rank // 2。
+ 建议使用 rank 为 64 及以上的设置。



# `.\cogvideo-finetune\finetune\train_cogvideox_image_to_video_lora.py`

```
# 版权声明，标明版权归属于CogView团队、清华大学、ZhipuAI和HuggingFace团队，所有权利保留。
# 
# 根据Apache许可证第2.0版（“许可证”）授权；
# 除非遵循该许可证，否则不得使用此文件。
# 可以在以下网址获取许可证的副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，软件在许可证下以“按原样”方式分发，
# 不提供任何明示或暗示的担保或条件。
# 有关许可证下权限和限制的具体信息，请参见许可证。

# 导入命令行参数解析库
import argparse
# 导入日志记录库
import logging
# 导入数学库
import math
# 导入操作系统接口库
import os
# 导入随机数生成库
import random
# 导入文件和目录操作库
import shutil
# 导入时间处理库
from datetime import timedelta
# 导入路径操作库
from pathlib import Path
# 导入类型注解相关的库
from typing import List, Optional, Tuple, Union

# 导入PyTorch库
import torch
# 导入transformers库，用于处理预训练模型
import transformers
# 从accelerate库导入加速器类
from accelerate import Accelerator
# 从accelerate库导入日志记录函数
from accelerate.logging import get_logger
# 从accelerate库导入分布式数据并行相关参数
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, ProjectConfiguration, set_seed
# 从huggingface_hub库导入创建和上传模型库的函数
from huggingface_hub import create_repo, upload_folder
# 从peft库导入Lora配置及模型状态字典相关函数
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
# 从torch.utils.data库导入数据加载器和数据集类
from torch.utils.data import DataLoader, Dataset
# 从torchvision库导入图像变换函数
from torchvision import transforms
# 从tqdm库导入进度条显示
from tqdm.auto import tqdm
# 从transformers库导入自动标记器和模型
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

# 导入diffusers库
import diffusers
# 从diffusers库导入不同模型和调度器
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
# 从diffusers.models.embeddings库导入3D旋转位置嵌入函数
from diffusers.models.embeddings import get_3d_rotary_pos_embed
# 从diffusers.optimization库导入调度器获取函数
from diffusers.optimization import get_scheduler
# 从diffusers.pipelines.cogvideo库导入图像缩放裁剪区域获取函数
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
# 从diffusers.training_utils库导入训练参数处理和内存释放函数
from diffusers.training_utils import cast_training_params, free_memory
# 从diffusers.utils库导入多种工具函数
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    export_to_video,
    is_wandb_available,
    load_image,
)
# 从diffusers.utils.hub_utils库导入模型卡加载和填充函数
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
# 从diffusers.utils.torch_utils库导入检查模块编译状态的函数
from diffusers.utils.torch_utils import is_compiled_module
# 从torchvision.transforms.functional库导入中心裁剪和调整大小函数
from torchvision.transforms.functional import center_crop, resize
# 从torchvision.transforms库导入插值模式
from torchvision.transforms import InterpolationMode
# 导入torchvision.transforms库
import torchvision.transforms as TT
# 导入NumPy库
import numpy as np
# 从diffusers.image_processor库导入图像处理器
from diffusers.image_processor import VaeImageProcessor

# 如果WandB库可用，则导入WandB
if is_wandb_available():
    import wandb

# 检查是否安装了最小版本的diffusers库，如果没有，将会报错。风险自负。
check_min_version("0.31.0.dev0")

# 获取日志记录器实例，使用当前模块的名称
logger = get_logger(__name__)

# 定义获取命令行参数的函数
def get_args():
    # 创建参数解析器，描述为CogVideoX的训练脚本示例
    parser = argparse.ArgumentParser(description="Simple example of a training script for CogVideoX.")

    # 添加模型信息的命令行参数
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        # 帮助信息，说明该参数是预训练模型的路径或Hugging Face模型标识符
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    # 添加命令行参数，指定预训练模型的修订版
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        # 帮助信息：来自 huggingface.co/models 的预训练模型标识符的修订版
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    # 添加命令行参数，指定预训练模型的变体
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        # 帮助信息：预训练模型标识符的模型文件的变体，例如 fp16
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    # 添加命令行参数，指定缓存目录
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        # 帮助信息：下载的模型和数据集将存储的目录
        help="The directory where the downloaded models and datasets will be stored.",
    )

    # 数据集信息
    # 添加命令行参数，指定数据集名称
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        # 帮助信息：包含实例图像训练数据的数据集名称，可以是本地数据集路径
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    # 添加命令行参数，指定数据集配置名称
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        # 帮助信息：数据集的配置，如果只有一个配置则保留为 None
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    # 添加命令行参数，指定实例数据根目录
    parser.add_argument(
        "--instance_data_root",
        type=str,
        default=None,
        # 帮助信息：包含训练数据的文件夹
        help=("A folder containing the training data."),
    )
    # 添加命令行参数，指定视频列名
    parser.add_argument(
        "--video_column",
        type=str,
        default="video",
        # 帮助信息：数据集中包含视频的列名，或包含视频数据路径的文件名
        help="The column of the dataset containing videos. Or, the name of the file in `--instance_data_root` folder containing the line-separated path to video data.",
    )
    # 添加命令行参数，指定提示文本列名
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        # 帮助信息：数据集中每个视频的实例提示的列名，或包含实例提示的文件名
        help="The column of the dataset containing the instance prompt for each video. Or, the name of the file in `--instance_data_root` folder containing the line-separated instance prompts.",
    )
    # 添加命令行参数，指定标识符令牌
    parser.add_argument(
        "--id_token", type=str, default=None, 
        # 帮助信息：如果提供，将附加到每个提示的开头的标识符令牌
        help="Identifier token appended to the start of each prompt if provided."
    )
    # 添加命令行参数，指定数据加载器的工作进程数
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        # 帮助信息：用于数据加载的子进程数量。0 表示在主进程中加载数据
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    # 验证
    # 添加命令行参数，指定验证提示
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        # 帮助信息：在验证期间使用的一个或多个提示，以验证模型是否在学习
        help="One or more prompt(s) that is used during validation to verify that the model is learning. Multiple validation prompts should be separated by the '--validation_prompt_seperator' string.",
    )
    # 添加一个命令行参数用于指定验证图像路径
    parser.add_argument(
        "--validation_images",
        # 参数类型为字符串
        type=str,
        # 默认值为 None
        default=None,
        # 参数帮助信息说明用途
        help="One or more image path(s) that is used during validation to verify that the model is learning. Multiple validation paths should be separated by the '--validation_prompt_seperator' string. These should correspond to the order of the validation prompts.",
    )
    # 添加一个命令行参数用于指定验证提示分隔符
    parser.add_argument(
        "--validation_prompt_separator",
        # 参数类型为字符串
        type=str,
        # 默认值为 ':::'
        default=":::",
        # 参数帮助信息说明用途
        help="String that separates multiple validation prompts",
    )
    # 添加一个命令行参数用于指定生成验证视频的数量
    parser.add_argument(
        "--num_validation_videos",
        # 参数类型为整数
        type=int,
        # 默认值为 1
        default=1,
        # 参数帮助信息说明用途
        help="Number of videos that should be generated during validation per `validation_prompt`.",
    )
    # 添加一个命令行参数用于指定每 X 个训练周期进行验证
    parser.add_argument(
        "--validation_epochs",
        # 参数类型为整数
        type=int,
        # 默认值为 50
        default=50,
        # 参数帮助信息说明用途
        help=(
            "Run validation every X epochs. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_videos`."
        ),
    )
    # 添加一个命令行参数用于指定引导尺度
    parser.add_argument(
        "--guidance_scale",
        # 参数类型为浮点数
        type=float,
        # 默认值为 6
        default=6,
        # 参数帮助信息说明用途
        help="The guidance scale to use while sampling validation videos.",
    )
    # 添加一个命令行参数用于指定是否使用动态配置
    parser.add_argument(
        "--use_dynamic_cfg",
        # 参数类型为布尔值，设置为真时启用动态配置
        action="store_true",
        # 默认值为 False
        default=False,
        # 参数帮助信息说明用途
        help="Whether or not to use the default cosine dynamic guidance schedule when sampling validation videos.",
    )

    # 训练信息
    # 添加一个命令行参数用于指定随机种子
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    # 添加一个命令行参数用于指定 LoRA 更新矩阵的维度
    parser.add_argument(
        "--rank",
        # 参数类型为整数
        type=int,
        # 默认值为 128
        default=128,
        # 参数帮助信息说明用途
        help=("The dimension of the LoRA update matrices."),
    )
    # 添加一个命令行参数用于指定 LoRA 的缩放因子
    parser.add_argument(
        "--lora_alpha",
        # 参数类型为浮点数
        type=float,
        # 默认值为 128
        default=128,
        # 参数帮助信息说明用途
        help=("The scaling factor to scale LoRA weight update. The actual scaling factor is `lora_alpha / rank`"),
    )
    # 添加一个命令行参数用于指定混合精度设置
    parser.add_argument(
        "--mixed_precision",
        # 参数类型为字符串
        type=str,
        # 默认值为 None
        default=None,
        # 可选值包括 "no", "fp16", "bf16"
        choices=["no", "fp16", "bf16"],
        # 参数帮助信息说明用途
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    # 添加一个命令行参数用于指定输出目录
    parser.add_argument(
        "--output_dir",
        # 参数类型为字符串
        type=str,
        # 默认值为 'cogvideox-i2v-lora'
        default="cogvideox-i2v-lora",
        # 参数帮助信息说明用途
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    # 添加一个命令行参数用于指定输入视频的高度
    parser.add_argument(
        "--height",
        # 参数类型为整数
        type=int,
        # 默认值为 480
        default=480,
        # 参数帮助信息说明用途
        help="All input videos are resized to this height.",
    )
    # 添加一个命令行参数用于指定输入视频的宽度
    parser.add_argument(
        "--width",
        # 参数类型为整数
        type=int,
        # 默认值为 720
        default=720,
        # 参数帮助信息说明用途
        help="All input videos are resized to this width.",
    )
    # 添加一个参数用于设置视频重塑模式，接受的值有 ['center', 'random', 'none']
    parser.add_argument(
        "--video_reshape_mode",
        type=str,
        default="center",
        help="All input videos are reshaped to this mode. Choose between ['center', 'random', 'none']",
    )
    # 添加一个参数用于设置输入视频的帧率，默认为 8
    parser.add_argument("--fps", type=int, default=8, help="All input videos will be used at this FPS.")
    # 添加一个参数用于设置输入视频的最大帧数，默认为 49
    parser.add_argument(
        "--max_num_frames", type=int, default=49, help="All input videos will be truncated to these many frames."
    )
    # 添加一个参数用于设置从每个输入视频开始跳过的帧数，默认为 0
    parser.add_argument(
        "--skip_frames_start",
        type=int,
        default=0,
        help="Number of frames to skip from the beginning of each input video. Useful if training data contains intro sequences.",
    )
    # 添加一个参数用于设置从每个输入视频结束跳过的帧数，默认为 0
    parser.add_argument(
        "--skip_frames_end",
        type=int,
        default=0,
        help="Number of frames to skip from the end of each input video. Useful if training data contains outro sequences.",
    )
    # 添加一个参数用于设置是否随机水平翻转视频
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip videos horizontally",
    )
    # 添加一个参数用于设置训练数据加载器的批处理大小，默认为 4
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    # 添加一个参数用于设置训练的总周期数，默认为 1
    parser.add_argument("--num_train_epochs", type=int, default=1)
    # 添加一个参数用于设置总训练步骤数，默认为 None，覆盖周期设置
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides `--num_train_epochs`.",
    )
    # 添加一个参数用于设置每 X 次更新保存训练状态检查点的步数，默认为 500
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    # 添加一个参数用于设置存储的最大检查点数量，默认为 None
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    # 添加一个参数用于设置是否从先前的检查点恢复训练，默认为 None
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    # 添加一个参数用于设置在执行反向传播/更新前要累积的更新步骤数，默认为 1
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    # 添加一个参数用于设置是否使用梯度检查点来节省内存，默认为 False
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    # 添加一个参数用于设置初始学习率，默认为 1e-4
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    # 添加命令行参数 --scale_lr，作为布尔标志，默认值为 False
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    # 添加命令行参数 --lr_scheduler，指定学习率调度器的类型，默认值为 "constant"
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    # 添加命令行参数 --lr_warmup_steps，指定学习率调度器的预热步骤数，默认值为 500
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    # 添加命令行参数 --lr_num_cycles，指定在 cosine_with_restarts 调度器中学习率的硬重置次数，默认值为 1
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    # 添加命令行参数 --lr_power，指定多项式调度器的幂因子，默认值为 1.0
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    # 添加命令行参数 --enable_slicing，作为布尔标志，默认值为 False，表示是否使用 VAE 切片以节省内存
    parser.add_argument(
        "--enable_slicing",
        action="store_true",
        default=False,
        help="Whether or not to use VAE slicing for saving memory.",
    )
    # 添加命令行参数 --enable_tiling，作为布尔标志，默认值为 False，表示是否使用 VAE 瓷砖以节省内存
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        default=False,
        help="Whether or not to use VAE tiling for saving memory.",
    )
    # 添加命令行参数 --noised_image_dropout，指定图像条件的丢弃概率，默认值为 0.05
    parser.add_argument(
        "--noised_image_dropout",
        type=float,
        default=0.05,
        help="Image condition dropout probability.",
    )

    # 添加命令行参数 --optimizer，指定优化器类型，默认值为 "adam"
    parser.add_argument(
        "--optimizer",
        type=lambda s: s.lower(),
        default="adam",
        choices=["adam", "adamw", "prodigy"],
        help=("The optimizer type to use."),
    )
    # 添加命令行参数 --use_8bit_adam，作为布尔标志，表示是否使用 bitsandbytes 的 8 位 Adam
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    # 添加命令行参数 --adam_beta1，指定 Adam 和 Prodigy 优化器的 beta1 参数，默认值为 0.9
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    # 添加命令行参数 --adam_beta2，指定 Adam 和 Prodigy 优化器的 beta2 参数，默认值为 0.95
    parser.add_argument(
        "--adam_beta2", type=float, default=0.95, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    # 添加命令行参数 --prodigy_beta3，指定 Prodigy 优化器的步长系数，默认值为 None
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="Coefficients for computing the Prodigy optimizer's stepsize using running averages. If set to None, uses the value of square root of beta2.",
    )
    # 添加命令行参数 --prodigy_decouple，作为布尔标志，表示是否使用 AdamW 风格的解耦权重衰减
    parser.add_argument("--prodigy_decouple", action="store_true", help="Use AdamW style decoupled weight decay")
    # 添加命令行参数 --adam_weight_decay，指定 UNet 参数使用的权重衰减，默认值为 1e-04
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    # 添加命令行参数 --adam_epsilon，指定 Adam 和 Prodigy 优化器的 epsilon 值，默认值为 1e-08
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    # 添加命令行参数 --max_grad_norm，指定最大梯度范数，默认值为 1.0
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # 添加命令行参数，用于开启 Adam 的偏差修正
        parser.add_argument("--prodigy_use_bias_correction", action="store_true", help="Turn on Adam's bias correction.")
        # 添加命令行参数，用于在热身阶段移除学习率，以避免 D 估计的问题
        parser.add_argument(
            "--prodigy_safeguard_warmup",
            action="store_true",
            help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage.",
        )
    
        # 添加项目追踪器名称的命令行参数，类型为字符串，默认为 None
        parser.add_argument("--tracker_name", type=str, default=None, help="Project tracker name")
        # 添加命令行参数，指定是否将模型推送到 Hub
        parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
        # 添加 Hub 访问令牌的命令行参数，类型为字符串，默认为 None
        parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
        # 添加命令行参数，指定要与本地输出目录同步的存储库名称
        parser.add_argument(
            "--hub_model_id",
            type=str,
            default=None,
            help="The name of the repository to keep in sync with the local `output_dir`.",
        )
        # 添加命令行参数，指定日志文件存储目录，默认为 "logs"
        parser.add_argument(
            "--logging_dir",
            type=str,
            default="logs",
            help="Directory where logs are stored.",
        )
        # 添加命令行参数，指定是否允许在 Ampere GPU 上使用 TF32，以加速训练
        parser.add_argument(
            "--allow_tf32",
            action="store_true",
            help=(
                "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
                " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
            ),
        )
        # 添加命令行参数，指定将结果和日志报告到的集成平台
        parser.add_argument(
            "--report_to",
            type=str,
            default=None,
            help=(
                'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
            ),
        )
        # 添加 NCCL 后端超时的命令行参数，单位为秒，默认为 600
        parser.add_argument("--nccl_timeout", type=int, default=600, help="NCCL backend timeout in seconds.")
    
        # 解析命令行参数并返回结果
        return parser.parse_args()
# 定义一个视频数据集类，继承自 Dataset 基类
class VideoDataset(Dataset):
    # 初始化方法，接受多个参数以配置数据集
    def __init__(
        # 数据根目录，可选
        self,
        instance_data_root: Optional[str] = None,
        # 数据集名称，可选
        dataset_name: Optional[str] = None,
        # 数据集配置名称，可选
        dataset_config_name: Optional[str] = None,
        # 用于文本描述的列名
        caption_column: str = "text",
        # 视频列名
        video_column: str = "video",
        # 视频高度，默认 480
        height: int = 480,
        # 视频宽度，默认 720
        width: int = 720,
        # 视频重塑模式，默认使用中心模式
        video_reshape_mode: str = "center",
        # 帧率，默认 8 帧每秒
        fps: int = 8,
        # 最大帧数，默认 49
        max_num_frames: int = 49,
        # 开始跳过的帧数，默认 0
        skip_frames_start: int = 0,
        # 结束跳过的帧数，默认 0
        skip_frames_end: int = 0,
        # 缓存目录，可选
        cache_dir: Optional[str] = None,
        # ID 令牌，可选
        id_token: Optional[str] = None,
    ) -> None:
        # 调用父类的初始化方法
        super().__init__()

        # 将数据根目录转换为 Path 对象，如果没有提供则为 None
        self.instance_data_root = Path(instance_data_root) if instance_data_root is not None else None
        # 设置数据集名称
        self.dataset_name = dataset_name
        # 设置数据集配置名称
        self.dataset_config_name = dataset_config_name
        # 设置文本描述列名
        self.caption_column = caption_column
        # 设置视频列名
        self.video_column = video_column
        # 设置视频高度
        self.height = height
        # 设置视频宽度
        self.width = width
        # 设置视频重塑模式
        self.video_reshape_mode = video_reshape_mode
        # 设置帧率
        self.fps = fps
        # 设置最大帧数
        self.max_num_frames = max_num_frames
        # 设置开始跳过的帧数
        self.skip_frames_start = skip_frames_start
        # 设置结束跳过的帧数
        self.skip_frames_end = skip_frames_end
        # 设置缓存目录
        self.cache_dir = cache_dir
        # 设置 ID 令牌，默认为空字符串
        self.id_token = id_token or ""

        # 如果提供了数据集名称，则从 hub 加载数据集
        if dataset_name is not None:
            self.instance_prompts, self.instance_video_paths = self._load_dataset_from_hub()
        # 否则，从本地路径加载数据集
        else:
            self.instance_prompts, self.instance_video_paths = self._load_dataset_from_local_path()

        # 将 ID 令牌添加到每个提示前
        self.instance_prompts = [self.id_token + prompt for prompt in self.instance_prompts]

        # 计算实例视频的数量
        self.num_instance_videos = len(self.instance_video_paths)
        # 确保视频和提示数量匹配，不匹配则引发错误
        if self.num_instance_videos != len(self.instance_prompts):
            raise ValueError(
                f"Expected length of instance prompts and videos to be the same but found {len(self.instance_prompts)=} and {len(self.instance_video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )

        # 预处理数据并存储处理后的实例视频
        self.instance_videos = self._preprocess_data()

    # 返回数据集中的实例数量
    def __len__(self):
        return self.num_instance_videos

    # 根据索引获取数据项
    def __getitem__(self, index):
        return {
            # 返回对应的实例提示
            "instance_prompt": self.instance_prompts[index],
            # 返回对应的实例视频
            "instance_video": self.instance_videos[index],
        }
    # 从数据集中加载数据的私有方法
        def _load_dataset_from_hub(self):
            # 尝试导入 datasets 库以加载数据集
            try:
                from datasets import load_dataset
            # 如果导入失败，则抛出 ImportError
            except ImportError:
                raise ImportError(
                    "You are trying to load your data using the datasets library. If you wish to train using custom "
                    "captions please install the datasets library: `pip install datasets`. If you wish to load a "
                    "local folder containing images only, specify --instance_data_root instead."
                )
    
            # 从数据集中心下载并加载数据集，更多信息请参见文档链接
            dataset = load_dataset(
                self.dataset_name,  # 数据集名称
                self.dataset_config_name,  # 数据集配置名称
                cache_dir=self.cache_dir,  # 缓存目录
            )
            # 获取训练集的列名
            column_names = dataset["train"].column_names
    
            # 如果未指定视频列，则默认为列名的第一个
            if self.video_column is None:
                video_column = column_names[0]
                logger.info(f"`video_column` defaulting to {video_column}")
            else:
                video_column = self.video_column
                # 检查指定的视频列是否存在于列名中
                if video_column not in column_names:
                    raise ValueError(
                        f"`--video_column` value '{video_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
    
            # 如果未指定字幕列，则默认为列名的第二个
            if self.caption_column is None:
                caption_column = column_names[1]
                logger.info(f"`caption_column` defaulting to {caption_column}")
            else:
                caption_column = self.caption_column
                # 检查指定的字幕列是否存在于列名中
                if self.caption_column not in column_names:
                    raise ValueError(
                        f"`--caption_column` value '{self.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
    
            # 获取训练集中的实例提示（字幕）
            instance_prompts = dataset["train"][caption_column]
            # 获取训练集中视频文件路径的列表
            instance_videos = [Path(self.instance_data_root, filepath) for filepath in dataset["train"][video_column]]
    
            # 返回实例提示和视频路径
            return instance_prompts, instance_videos
    # 从本地路径加载数据集
        def _load_dataset_from_local_path(self):
            # 检查实例数据根目录是否存在
            if not self.instance_data_root.exists():
                # 抛出错误，指明根文件夹不存在
                raise ValueError("Instance videos root folder does not exist")
    
            # 构建提示文本文件路径
            prompt_path = self.instance_data_root.joinpath(self.caption_column)
            # 构建视频文件路径
            video_path = self.instance_data_root.joinpath(self.video_column)
    
            # 检查提示文件路径是否存在且为文件
            if not prompt_path.exists() or not prompt_path.is_file():
                # 抛出错误，指明提示文件路径不正确
                raise ValueError(
                    "Expected `--caption_column` to be path to a file in `--instance_data_root` containing line-separated text prompts."
                )
            # 检查视频文件路径是否存在且为文件
            if not video_path.exists() or not video_path.is_file():
                # 抛出错误，指明视频文件路径不正确
                raise ValueError(
                    "Expected `--video_column` to be path to a file in `--instance_data_root` containing line-separated paths to video data in the same directory."
                )
    
            # 读取提示文本文件，按行去除空白并返回列表
            with open(prompt_path, "r", encoding="utf-8") as file:
                instance_prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
            # 读取视频文件，按行去除空白并构建视频路径列表
            with open(video_path, "r", encoding="utf-8") as file:
                instance_videos = [
                    self.instance_data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0
                ]
    
            # 检查视频路径列表中是否存在无效文件路径
            if any(not path.is_file() for path in instance_videos):
                # 抛出错误，指明至少一个路径不是有效文件
                raise ValueError(
                    "Expected '--video_column' to be a path to a file in `--instance_data_root` containing line-separated paths to video data but found atleast one path that is not a valid file."
                )
    
            # 返回提示文本和视频路径列表
            return instance_prompts, instance_videos
    
        # 根据长宽调整数组以适应矩形裁剪
        def _resize_for_rectangle_crop(self, arr):
            # 获取目标图像尺寸
            image_size = self.height, self.width
            # 获取重塑模式
            reshape_mode = self.video_reshape_mode
            # 检查数组宽高比与目标图像宽高比
            if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
                # 调整数组尺寸以匹配目标图像宽度
                arr = resize(
                    arr,
                    size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                    interpolation=InterpolationMode.BICUBIC,
                )
            else:
                # 调整数组尺寸以匹配目标图像高度
                arr = resize(
                    arr,
                    size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                    interpolation=InterpolationMode.BICUBIC,
                )
    
            # 获取调整后数组的高度和宽度
            h, w = arr.shape[2], arr.shape[3]
            # 去掉数组的第一维
            arr = arr.squeeze(0)
    
            # 计算高度和宽度的差值
            delta_h = h - image_size[0]
            delta_w = w - image_size[1]
    
            # 根据重塑模式计算裁剪的起始点
            if reshape_mode == "random" or reshape_mode == "none":
                # 随机生成裁剪起始点
                top = np.random.randint(0, delta_h + 1)
                left = np.random.randint(0, delta_w + 1)
            elif reshape_mode == "center":
                # 计算中心裁剪起始点
                top, left = delta_h // 2, delta_w // 2
            else:
                # 抛出错误，指明重塑模式未实现
                raise NotImplementedError
            # 裁剪数组到指定的高度和宽度
            arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
            # 返回裁剪后的数组
            return arr
    # 数据预处理函数
    def _preprocess_data(self):
        # 尝试导入 decord 库
        try:
            import decord
        # 如果导入失败，则抛出 ImportError 异常，并提示用户安装 decord
        except ImportError:
            raise ImportError(
                "The `decord` package is required for loading the video dataset. Install with `pip install decord`"
            )

        # 设置 decord 使用 PyTorch 作为桥接库
        decord.bridge.set_bridge("torch")

        # 创建一个进度条，显示视频加载、调整大小和裁剪的进度
        progress_dataset_bar = tqdm(
            range(0, len(self.instance_video_paths)),
            desc="Loading progress resize and crop videos",
        )

        # 初始化视频列表，用于存储处理后的视频帧
        videos = []

        # 遍历每个视频文件的路径
        for filename in self.instance_video_paths:
            # 使用 decord.VideoReader 读取视频文件
            video_reader = decord.VideoReader(uri=filename.as_posix())
            # 获取视频帧的数量
            video_num_frames = len(video_reader)

            # 确定开始和结束帧的索引
            start_frame = min(self.skip_frames_start, video_num_frames)
            end_frame = max(0, video_num_frames - self.skip_frames_end)
            # 如果结束帧索引小于等于开始帧索引，则只获取开始帧
            if end_frame <= start_frame:
                frames = video_reader.get_batch([start_frame])
            # 如果帧数在开始和结束帧之间小于等于最大帧数，则获取全部帧
            elif end_frame - start_frame <= self.max_num_frames:
                frames = video_reader.get_batch(list(range(start_frame, end_frame)))
            # 否则，均匀选择帧的索引
            else:
                indices = list(range(start_frame, end_frame, (end_frame - start_frame) // self.max_num_frames))
                frames = video_reader.get_batch(indices)

            # 确保不超过最大帧数限制
            frames = frames[: self.max_num_frames]
            # 获取选中的帧数
            selected_num_frames = frames.shape[0]

            # 选择前 (4k + 1) 帧，确保帧数满足 VAE 的要求
            remainder = (3 + (selected_num_frames % 4)) % 4
            # 如果有多余帧，去掉这些帧
            if remainder != 0:
                frames = frames[:-remainder]
            # 更新选中的帧数
            selected_num_frames = frames.shape[0]

            # 断言选中的帧数减去 1 能被 4 整除
            assert (selected_num_frames - 1) % 4 == 0

            # 进行训练变换，将帧值归一化到 [-1, 1]
            frames = (frames - 127.5) / 127.5
            # 调整帧的维度顺序为 [F, C, H, W]
            frames = frames.permute(0, 3, 1, 2) # [F, C, H, W]
            # 更新进度条描述，显示当前视频的尺寸
            progress_dataset_bar.set_description(
                f"Loading progress Resizing video from {frames.shape[2]}x{frames.shape[3]} to {self.height}x{self.width}"
            )
            # 调整帧的尺寸以适应矩形裁剪
            frames = self._resize_for_rectangle_crop(frames)
            # 将处理后的帧添加到视频列表中
            videos.append(frames.contiguous())  # [F, C, H, W]
            # 更新进度条
            progress_dataset_bar.update(1)

        # 关闭进度条
        progress_dataset_bar.close()

        # 返回处理后的所有视频帧
        return videos
# 保存模型卡片，包含模型信息和视频验证
def save_model_card(
    # 仓库标识
    repo_id: str,
    # 视频列表，默认值为 None
    videos=None,
    # 基础模型名称，默认值为 None
    base_model: str = None,
    # 验证提示，默认值为 None
    validation_prompt=None,
    # 仓库文件夹路径，默认值为 None
    repo_folder=None,
    # 帧率，默认值为 8
    fps=8,
):
    # 初始化小部件字典
    widget_dict = []
    # 检查是否提供视频
    if videos is not None:
        # 遍历视频列表及其索引
        for i, video in enumerate(videos):
            # 为每个视频生成文件名
            video_path = f"final_video_{i}.mp4"
            # 导出视频到指定路径
            export_to_video(video, os.path.join(repo_folder, video_path, fps=fps))
            # 将视频信息添加到小部件字典中
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": video_path}},
            )

    # 定义模型描述文本
    model_description = f"""
# CogVideoX LoRA - {repo_id}

<Gallery />

## Model description

These are {repo_id} LoRA weights for {base_model}.

The weights were trained using the [CogVideoX Diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/cogvideo/train_cogvideox_image_to_video_lora.py).

Was LoRA for the text encoder enabled? No.

## Download model

[Download the *.safetensors LoRA]({repo_id}/tree/main) in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)


import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import load_image, export_to_video

pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("{repo_id}", weight_name="pytorch_lora_weights.safetensors", adapter_name=["cogvideox-i2v-lora"])

# The LoRA adapter weights are determined by what was used for training.
# In this case, we assume `--lora_alpha` is 32 and `--rank` is 64.
# It can be made lower or higher from what was used in training to decrease or amplify the effect
# of the LoRA upto a tolerance, beyond which one might notice no effect at all or overflows.
pipe.set_adapters(["cogvideox-i2v-lora"], [32 / 64])

image = load_image("/path/to/image")
video = pipe(image=image, "{validation_prompt}", guidance_scale=6, use_dynamic_cfg=True).frames[0]
export_to_video(video, "output.mp4", fps=8)


For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/THUDM/CogVideoX-5b-I2V/blob/main/LICENSE).
"""
    # 加载或创建模型卡片
    model_card = load_or_create_model_card(
        # 仓库 ID 或路径
        repo_id_or_path=repo_id,
        # 指示是否从训练中创建
        from_training=True,
        # 许可证类型
        license="other",
        # 基础模型名称
        base_model=base_model,
        # 验证提示
        prompt=validation_prompt,
        # 模型描述
        model_description=model_description,
        # 小部件信息
        widget=widget_dict,
    )
    # 定义标签列表
    tags = [
        "image-to-video",
        "diffusers-training",
        "diffusers",
        "lora",
        "cogvideox",
        "cogvideox-diffusers",
        "template:sd-lora",
    ]

    # 填充模型卡片的标签
    model_card = populate_model_card(model_card, tags=tags)
    # 保存模型卡片到指定路径
    model_card.save(os.path.join(repo_folder, "README.md"))


# 记录验证过程
def log_validation(
    # 管道对象
    pipe,
    # 参数
    args,
    # 加速器对象
    accelerator,
    # 管道参数，用于配置和管理数据处理流程
        pipeline_args,
        # 当前训练的轮次，通常用于控制训练过程
        epoch,
        # 指示是否进行最终验证的布尔值，默认为 False
        is_final_validation: bool = False,
# 日志记录当前验证运行的信息，包括生成视频的数量和提示内容
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_videos} videos with prompt: {pipeline_args['prompt']}."
    )
    # 初始化调度器参数字典
    scheduler_args = {}

    # 检查调度器配置中是否包含方差类型
    if "variance_type" in pipe.scheduler.config:
        # 获取方差类型
        variance_type = pipe.scheduler.config.variance_type

        # 如果方差类型是已学习的类型，设置为固定小
        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        # 更新调度器参数字典中的方差类型
        scheduler_args["variance_type"] = variance_type

    # 使用调度器配置和参数初始化调度器
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
    # 将管道移动到指定的加速器设备上
    pipe = pipe.to(accelerator.device)
    # 关闭进度条配置（注释掉，表示不使用进度条）

    # 运行推理
    # 创建生成器并设置随机种子，如果未指定种子，则为 None
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    # 初始化视频列表
    videos = []
    # 循环生成指定数量的视频
    for _ in range(args.num_validation_videos):
        # 通过管道生成视频帧
        pt_images = pipe(**pipeline_args, generator=generator, output_type="pt").frames[0]
        # 将生成的帧堆叠成一个张量
        pt_images = torch.stack([pt_images[i] for i in range(pt_images.shape[0])])

        # 将 PyTorch 图像转换为 NumPy 数组
        image_np = VaeImageProcessor.pt_to_numpy(pt_images)
        # 将 NumPy 数组转换为 PIL 图像
        image_pil = VaeImageProcessor.numpy_to_pil(image_np)

        # 将生成的 PIL 图像添加到视频列表中
        videos.append(image_pil)

    # 遍历加速器的跟踪器
    for tracker in accelerator.trackers:
        # 确定当前阶段名称
        phase_name = "test" if is_final_validation else "validation"
        # 检查是否为 WandB 跟踪器
        if tracker.name == "wandb":
            # 初始化视频文件名列表
            video_filenames = []
            # 遍历生成的视频
            for i, video in enumerate(videos):
                # 格式化提示内容并替换特殊字符
                prompt = (
                    pipeline_args["prompt"][:25]
                    .replace(" ", "_")
                    .replace(" ", "_")
                    .replace("'", "_")
                    .replace('"', "_")
                    .replace("/", "_")
                )
                # 生成视频文件名
                filename = os.path.join(args.output_dir, f"{phase_name}_video_{i}_{prompt}.mp4")
                # 将视频导出为文件
                export_to_video(video, filename, fps=8)
                # 将文件名添加到列表中
                video_filenames.append(filename)

            # 记录视频到跟踪器
            tracker.log(
                {
                    phase_name: [
                        wandb.Video(filename, caption=f"{i}: {pipeline_args['prompt']}")
                        for i, filename in enumerate(video_filenames)
                    ]
                }
            )

    # 删除管道对象以释放内存
    del pipe
    # 释放内存资源
    free_memory()

    # 返回生成的视频列表
    return videos


# 定义获取 T5 提示嵌入的函数
def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    # 如果提示是字符串，则将其转换为列表
    prompt = [prompt] if isinstance(prompt, str) else prompt
    # 获取批处理大小，即提示的数量
    batch_size = len(prompt)
    # 检查 tokenizer 是否被指定
        if tokenizer is not None:
            # 使用 tokenizer 对提示文本进行编码，生成张量形式的输入
            text_inputs = tokenizer(
                prompt,
                padding="max_length",  # 填充至最大长度
                max_length=max_sequence_length,  # 设置最大序列长度
                truncation=True,  # 允许截断超出最大长度的输入
                add_special_tokens=True,  # 添加特殊标记
                return_tensors="pt",  # 返回 PyTorch 张量
            )
            # 提取编码后的输入 ID
            text_input_ids = text_inputs.input_ids
        else:
            # 如果未提供 tokenizer，检查输入 ID 是否为 None
            if text_input_ids is None:
                # 引发错误，提示必须提供 text_input_ids
                raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")
    
        # 使用文本编码器生成提示的嵌入
        prompt_embeds = text_encoder(text_input_ids.to(device))[0]
        # 将嵌入转换为指定的数据类型和设备
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    
        # 为每个提示生成重复的文本嵌入，使用兼容 MPS 的方法
        _, seq_len, _ = prompt_embeds.shape  # 获取嵌入的形状
        # 重复嵌入以匹配生成数量
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        # 调整嵌入的形状以适应批处理
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
    
        # 返回处理后的文本嵌入
        return prompt_embeds
# 定义一个函数用于编码提示文本，参数包括分词器、文本编码器、提示内容等
def encode_prompt(
    tokenizer: T5Tokenizer,  # 用于将文本转换为模型输入格式的分词器
    text_encoder: T5EncoderModel,  # 文本编码器模型
    prompt: Union[str, List[str]],  # 提示文本，可以是字符串或字符串列表
    num_videos_per_prompt: int = 1,  # 每个提示生成的视频数量，默认为1
    max_sequence_length: int = 226,  # 输入序列的最大长度，默认为226
    device: Optional[torch.device] = None,  # 指定运行设备（如GPU），默认为None
    dtype: Optional[torch.dtype] = None,  # 指定数据类型（如float32），默认为None
    text_input_ids=None,  # 预先提供的文本输入ID，默认为None
):
    # 如果提示是字符串，则将其转换为单元素列表
    prompt = [prompt] if isinstance(prompt, str) else prompt
    # 获取提示的嵌入表示，调用自定义函数
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,  # 分词器
        text_encoder,  # 文本编码器
        prompt=prompt,  # 提示文本
        num_videos_per_prompt=num_videos_per_prompt,  # 每个提示生成的视频数量
        max_sequence_length=max_sequence_length,  # 最大序列长度
        device=device,  # 运行设备
        dtype=dtype,  # 数据类型
        text_input_ids=text_input_ids,  # 文本输入ID
    )
    # 返回提示嵌入表示
    return prompt_embeds


# 定义一个函数用于计算提示的嵌入表示，接受多个参数
def compute_prompt_embeddings(
    tokenizer,  # 分词器
    text_encoder,  # 文本编码器
    prompt,  # 提示文本
    max_sequence_length,  # 最大序列长度
    device,  # 运行设备
    dtype,  # 数据类型
    requires_grad: bool = False  # 是否需要计算梯度，默认为False
):
    # 如果需要计算梯度
    if requires_grad:
        # 调用 encode_prompt 函数获取提示嵌入
        prompt_embeds = encode_prompt(
            tokenizer,  # 分词器
            text_encoder,  # 文本编码器
            prompt,  # 提示文本
            num_videos_per_prompt=1,  # 每个提示生成的视频数量
            max_sequence_length=max_sequence_length,  # 最大序列长度
            device=device,  # 运行设备
            dtype=dtype,  # 数据类型
        )
    else:
        # 如果不需要计算梯度，使用上下文管理器禁止梯度计算
        with torch.no_grad():
            # 调用 encode_prompt 函数获取提示嵌入
            prompt_embeds = encode_prompt(
                tokenizer,  # 分词器
                text_encoder,  # 文本编码器
                prompt,  # 提示文本
                num_videos_per_prompt=1,  # 每个提示生成的视频数量
                max_sequence_length=max_sequence_length,  # 最大序列长度
                device=device,  # 运行设备
                dtype=dtype,  # 数据类型
            )
    # 返回计算得到的提示嵌入
    return prompt_embeds


# 定义一个函数用于准备旋转位置嵌入，接受多个参数
def prepare_rotary_positional_embeddings(
    height: int,  # 输入图像的高度
    width: int,  # 输入图像的宽度
    num_frames: int,  # 帧数
    vae_scale_factor_spatial: int = 8,  # VAE空间缩放因子，默认为8
    patch_size: int = 2,  # 每个补丁的大小，默认为2
    attention_head_dim: int = 64,  # 注意力头的维度，默认为64
    device: Optional[torch.device] = None,  # 指定运行设备（如GPU），默认为None
    base_height: int = 480,  # 基础高度，默认为480
    base_width: int = 720,  # 基础宽度，默认为720
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回两个张量的元组
    # 计算网格高度
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    # 计算网格宽度
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    # 计算基础宽度
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    # 计算基础高度
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    # 获取网格的裁剪区域坐标
    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    # 获取旋转位置嵌入的正弦和余弦频率
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,  # 嵌入维度
        crops_coords=grid_crops_coords,  # 网格裁剪坐标
        grid_size=(grid_height, grid_width),  # 网格大小
        temporal_size=num_frames,  # 时间维度大小
    )

    # 将余弦频率张量移动到指定设备
    freqs_cos = freqs_cos.to(device=device)
    # 将正弦频率张量移动到指定设备
    freqs_sin = freqs_sin.to(device=device)
    # 返回余弦和正弦频率张量
    return freqs_cos, freqs_sin


# 定义一个函数用于获取优化器，接受多个参数
def get_optimizer(args, params_to_optimize, use_deepspeed: bool = False):
    # 如果使用 DeepSpeed 优化器
    if use_deepspeed:
        # 从 accelerate 库导入 DummyOptim
        from accelerate.utils import DummyOptim

        # 返回 DeepSpeed 优化器的实例
        return DummyOptim(
            params_to_optimize,  # 待优化的参数
            lr=args.learning_rate,  # 学习率
            betas=(args.adam_beta1, args.adam_beta2),  # Adam优化器的动量参数
            eps=args.adam_epsilon,  # Adam优化器的 epsilon
            weight_decay=args.adam_weight_decay,  # 权重衰减
        )

    # 优化器创建
    # 定义支持的优化器类型列表
    supported_optimizers = ["adam", "adamw", "prodigy"]
    # 检查用户选择的优化器是否在支持的列表中
    if args.optimizer not in supported_optimizers:
        # 记录不支持的优化器警告信息
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        # 将优化器默认设置为 "adamw"
        args.optimizer = "adamw"

    # 检查是否使用 8 位 Adam 优化器，并且当前优化器不是 Adam 或 AdamW
    if args.use_8bit_adam and args.optimizer.lower() not in ["adam", "adamw"]:
        # 记录警告，说明使用 8 位 Adam 时优化器必须为 Adam 或 AdamW
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    # 如果用户选择使用 8 位 Adam 优化器
    if args.use_8bit_adam:
        try:
            # 尝试导入 bitsandbytes 库
            import bitsandbytes as bnb
        except ImportError:
            # 如果导入失败，抛出错误提示用户安装 bitsandbytes 库
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    # 如果用户选择的优化器是 AdamW
    if args.optimizer.lower() == "adamw":
        # 根据是否使用 8 位 Adam 选择相应的优化器类
        optimizer_class = bnb.optim.AdamW8bit if args.use_8bit_adam else torch.optim.AdamW

        # 创建优化器实例，传入优化参数和相关超参数
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    # 如果用户选择的优化器是 Adam
    elif args.optimizer.lower() == "adam":
        # 根据是否使用 8 位 Adam 选择相应的优化器类
        optimizer_class = bnb.optim.Adam8bit if args.use_8bit_adam else torch.optim.Adam

        # 创建优化器实例，传入优化参数和相关超参数
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    # 如果用户选择的优化器是 Prodigy
    elif args.optimizer.lower() == "prodigy":
        try:
            # 尝试导入 prodigyopt 库
            import prodigyopt
        except ImportError:
            # 如果导入失败，抛出错误提示用户安装 prodigyopt 库
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        # 设置 Prodigy 优化器类
        optimizer_class = prodigyopt.Prodigy

        # 检查学习率是否过低，并记录警告
        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        # 创建优化器实例，传入优化参数和相关超参数
        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # 返回创建的优化器实例
    return optimizer
# 主函数，接收命令行参数
def main(args):
    # 检查是否同时使用 wandb 和 hub_token，若是则抛出错误
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    # 检查 MPS 是否可用且混合精度为 bf16，若是则抛出错误
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # 生成日志目录的路径
    logging_dir = Path(args.output_dir, args.logging_dir)

    # 初始化项目配置
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    # 设置分布式数据并行的参数
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # 初始化进程组的参数
    init_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=args.nccl_timeout))
    # 创建加速器实例
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_kwargs],
    )

    # 如果 MPS 可用，禁用自动混合精度
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # 检查是否使用 wandb 进行报告
    if args.report_to == "wandb":
        # 如果 wandb 不可用，则抛出导入错误
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # 配置日志记录以便于调试
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # 记录加速器的状态信息
    logger.info(accelerator.state, main_process_only=False)
    # 如果是本地主进程，设置不同的日志详细级别
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # 如果提供了种子，则设置训练种子
    if args.seed is not None:
        set_seed(args.seed)

    # 处理仓库创建
    if accelerator.is_main_process:
        # 如果输出目录不为空，创建该目录
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        # 如果需要推送到 Hub，创建仓库
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # 准备模型和调度器
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

    # CogVideoX-2b 权重以 float16 存储
    # CogVideoX-5b 和 CogVideoX-5b-I2V 的权重以 bfloat16 存储
    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    # 从预训练模型路径加载 3D Transformer 模型
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",  # 指定子文件夹为 transformer
        torch_dtype=load_dtype,  # 设置加载的权重数据类型
        revision=args.revision,  # 使用指定的修订版本
        variant=args.variant,  # 使用指定的变体
    )
    
    # 从预训练模型路径加载 VAE 模型
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    
    # 从预训练模型路径加载调度器
    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    # 如果启用了切片，则启用 VAE 的切片功能
    if args.enable_slicing:
        vae.enable_slicing()
    # 如果启用了平铺，则启用 VAE 的平铺功能
    if args.enable_tiling:
        vae.enable_tiling()
    
    # 仅训练附加的适配器 LoRA 层
    text_encoder.requires_grad_(False)  # 禁用文本编码器的梯度计算
    transformer.requires_grad_(False)  # 禁用 Transformer 的梯度计算
    vae.requires_grad_(False)  # 禁用 VAE 的梯度计算
    
    # 对于混合精度训练，将所有不可训练权重（vae、text_encoder 和 transformer）转换为半精度
    weight_dtype = torch.float32  # 默认权重数据类型为 float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed 处理精度，使用 DeepSpeed 配置中的设置
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16  # 启用 fp16 时设置权重数据类型为 float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.float16  # 启用 bf16 时也设置为 float16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16  # 如果使用 fp16，设置权重数据类型为 float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16  # 如果使用 bf16，设置为 bfloat16
    
    # 检查 MPS 是否可用，且权重数据类型为 bfloat16
    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # 由于 pytorch#99272，MPS 尚不支持 bfloat16。
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    
    # 将文本编码器、Transformer 和 VAE 转移到加速器设备，指定数据类型
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    
    # 如果启用了梯度检查点，则启用 Transformer 的梯度检查点功能
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    
    # 现在我们将新 LoRA 权重添加到注意力层
    transformer_lora_config = LoraConfig(
        r=args.rank,  # 设置 LoRA 的秩
        lora_alpha=args.lora_alpha,  # 设置 LoRA 的 alpha 值
        init_lora_weights=True,  # 初始化 LoRA 权重
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # 目标模块列表
    )
    # 将 LoRA 适配器添加到 Transformer
    transformer.add_adapter(transformer_lora_config)
    # 解包模型，以便于处理
        def unwrap_model(model):
            # 使用加速器解包模型
            model = accelerator.unwrap_model(model)
            # 如果是编译的模块，获取原始模型，否则返回当前模型
            model = model._orig_mod if is_compiled_module(model) else model
            # 返回处理后的模型
            return model
    
        # 创建自定义保存和加载钩子，以便加速器以良好格式序列化状态
        def save_model_hook(models, weights, output_dir):
            # 检查当前进程是否为主进程
            if accelerator.is_main_process:
                # 初始化待保存的层为 None
                transformer_lora_layers_to_save = None
    
                # 遍历模型列表
                for model in models:
                    # 检查模型类型是否与解包后的 transformer 相同
                    if isinstance(model, type(unwrap_model(transformer))):
                        # 获取模型的状态字典
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    else:
                        # 抛出异常以处理意外模型类型
                        raise ValueError(f"unexpected save model: {model.__class__}")
    
                    # 确保从权重中移除已处理的模型
                    weights.pop()
    
                # 保存 LoRA 权重
                CogVideoXImageToVideoPipeline.save_lora_weights(
                    output_dir,
                    transformer_lora_layers=transformer_lora_layers_to_save,
                )
    
        # 创建加载模型的钩子
        def load_model_hook(models, input_dir):
            # 初始化 transformer 为 None
            transformer_ = None
    
            # 当模型列表不为空时
            while len(models) > 0:
                # 从模型列表中弹出模型
                model = models.pop()
    
                # 检查模型类型
                if isinstance(model, type(unwrap_model(transformer))):
                    # 将 transformer 设置为当前模型
                    transformer_ = model
                else:
                    # 抛出异常以处理意外模型类型
                    raise ValueError(f"Unexpected save model: {model.__class__}")
    
            # 从输入目录获取 LoRA 状态字典
            lora_state_dict = CogVideoXImageToVideoPipeline.lora_state_dict(input_dir)
    
            # 创建转换后的 transformer 状态字典
            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
            }
            # 将状态字典转换为适合 PEFT 的格式
            transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
            # 设置 PEFT 模型的状态字典，并获取不兼容的键
            incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
            # 如果存在不兼容的键，检查意外的键
            if incompatible_keys is not None:
                # 获取意外的键
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    # 记录警告信息
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )
    
            # 确保可训练参数为 float32 类型
            if args.mixed_precision == "fp16":
                # 仅将可训练参数（LoRA）转为 fp32 类型
                cast_training_params([transformer_])
    
        # 注册保存状态前钩子
        accelerator.register_save_state_pre_hook(save_model_hook)
        # 注册加载状态前钩子
        accelerator.register_load_state_pre_hook(load_model_hook)
    
        # 启用 TF32 以加速 Ampere GPU 的训练
        if args.allow_tf32 and torch.cuda.is_available():
            # 允许使用 TF32
            torch.backends.cuda.matmul.allow_tf32 = True
    # 如果指定了缩放学习率的标志
    if args.scale_lr:
        # 根据梯度累积步骤、训练批量大小和进程数缩放学习率
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # 确保可训练参数为 float32 类型
    if args.mixed_precision == "fp16":
        # 仅将可训练参数（LoRA）提升为 fp32 类型
        cast_training_params([transformer], dtype=torch.float32)

    # 获取所有可训练的 LoRA 参数
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # 优化参数
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    # 将参数放入待优化的列表中
    params_to_optimize = [transformer_parameters_with_lr]

    # 检查是否使用 DeepSpeed 优化器
    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    # 检查是否使用 DeepSpeed 调度器
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    # 获取优化器
    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    # 创建数据集和数据加载器
    train_dataset = VideoDataset(
        # 实例数据的根目录
        instance_data_root=args.instance_data_root,
        # 数据集名称
        dataset_name=args.dataset_name,
        # 数据集配置名称
        dataset_config_name=args.dataset_config_name,
        # 描述性文本列名称
        caption_column=args.caption_column,
        # 视频列名称
        video_column=args.video_column,
        # 视频高度
        height=args.height,
        # 视频宽度
        width=args.width,
        # 视频重塑模式
        video_reshape_mode=args.video_reshape_mode,
        # 帧率
        fps=args.fps,
        # 最大帧数
        max_num_frames=args.max_num_frames,
        # 开始跳过的帧数
        skip_frames_start=args.skip_frames_start,
        # 结束跳过的帧数
        skip_frames_end=args.skip_frames_end,
        # 缓存目录
        cache_dir=args.cache_dir,
        # 身份令牌
        id_token=args.id_token,
    )

    # 定义编码视频的函数
    def encode_video(video, bar):
        # 更新进度条
        bar.update(1)
        # 将视频转换为指定设备并增加一个维度
        video = video.to(accelerator.device, dtype=vae.dtype).unsqueeze(0)
        # 调整视频维度顺序为 [B, C, F, H, W]
        video = video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        # 克隆第一个帧作为图像
        image = video[:, :, :1].clone()

        # 编码视频以获取潜在分布
        latent_dist = vae.encode(video).latent_dist

        # 生成图像噪声标准差
        image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=image.device)
        # 取指数并转换为图像数据类型
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=image.dtype)
        # 生成与图像大小相同的噪声图像
        noisy_image = torch.randn_like(image) * image_noise_sigma[:, None, None, None, None]
        # 对噪声图像进行编码以获取潜在分布
        image_latent_dist = vae.encode(noisy_image).latent_dist

        # 返回潜在分布
        return latent_dist, image_latent_dist

    # 计算实例提示的嵌入
    train_dataset.instance_prompts = [
        compute_prompt_embeddings(
            tokenizer,
            text_encoder,
            [prompt],
            transformer.config.max_text_seq_length,
            accelerator.device,
            weight_dtype,
            requires_grad=False,
        )
        for prompt in train_dataset.instance_prompts
    ]

    # 创建进度条以显示编码视频的加载进度
    progress_encode_bar = tqdm(
        range(0, len(train_dataset.instance_videos)),
        desc="Loading Encode videos",
    )
    # 对训练数据集中的每个实例视频进行编码，并更新数据集的实例视频列表
    train_dataset.instance_videos = [encode_video(video, progress_encode_bar) for video in train_dataset.instance_videos]
    # 关闭进度编码条
    progress_encode_bar.close()

    # 定义用于合并样本的函数
    def collate_fn(examples):
        # 初始化视频和图像的列表
        videos = []
        images = []
        # 遍历所有示例
        for example in examples:
            # 获取实例视频的潜在分布和图像潜在分布
            latent_dist, image_latent_dist = example["instance_video"]

            # 从潜在分布中采样，并应用缩放因子
            video_latents = latent_dist.sample() * vae.config.scaling_factor
            image_latents = image_latent_dist.sample() * vae.config.scaling_factor
            # 调整视频潜在表示的维度顺序
            video_latents = video_latents.permute(0, 2, 1, 3, 4)
            # 调整图像潜在表示的维度顺序
            image_latents = image_latents.permute(0, 2, 1, 3, 4)

            # 计算填充的形状，以便为视频潜在表示保留时间步长
            padding_shape = (video_latents.shape[0], video_latents.shape[1] - 1, *video_latents.shape[2:])
            # 创建新的零填充张量
            latent_padding = image_latents.new_zeros(padding_shape)
            # 将填充张量附加到图像潜在表示
            image_latents = torch.cat([image_latents, latent_padding], dim=1)

            # 根据随机值决定是否将图像潜在表示置为零（添加噪声）
            if random.random() < args.noised_image_dropout:
                image_latents = torch.zeros_like(image_latents)

            # 将视频和图像潜在表示添加到列表中
            videos.append(video_latents)
            images.append(image_latents)

        # 将视频和图像列表合并成单一张量
        videos = torch.cat(videos)
        images = torch.cat(images)
        # 将张量转换为连续格式并转为浮点型
        videos = videos.to(memory_format=torch.contiguous_format).float()
        images = images.to(memory_format=torch.contiguous_format).float()

        # 提取每个示例的提示信息
        prompts = [example["instance_prompt"] for example in examples]
        # 将提示信息合并为一个张量
        prompts = torch.cat(prompts)

        # 返回包含视频、图像和提示的字典
        return {
            "videos": (videos, images),
            "prompts": prompts,
        }

    # 创建数据加载器以便于批量加载训练数据
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,  # 设置批量大小
        shuffle=True,  # 打乱数据
        collate_fn=collate_fn,  # 使用自定义的合并函数
        num_workers=args.dataloader_num_workers,  # 设置工作进程数
    )

    # 计算训练步骤数的调度器及相关数学
    overrode_max_train_steps = False  # 初始化标志，表示是否覆盖最大训练步骤
    # 计算每个训练周期的更新步骤数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # 如果没有设置最大训练步骤，则根据训练周期和更新步骤计算最大训练步骤
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # 根据是否使用 DeepSpeed 调度器选择相应的学习率调度器
    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        # 创建一个虚拟调度器
        lr_scheduler = DummyScheduler(
            name=args.lr_scheduler,  # 学习率调度器名称
            optimizer=optimizer,  # 关联的优化器
            total_num_steps=args.max_train_steps * accelerator.num_processes,  # 总训练步骤
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,  # 预热步骤数
        )
    else:
        # 创建标准学习率调度器
        lr_scheduler = get_scheduler(
            args.lr_scheduler,  # 学习率调度器类型
            optimizer=optimizer,  # 关联的优化器
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,  # 预热步骤数
            num_training_steps=args.max_train_steps * accelerator.num_processes,  # 总训练步骤
            num_cycles=args.lr_num_cycles,  # 循环次数
            power=args.lr_power,  # 学习率调整的指数
        )

    # 使用加速器准备所有组件
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer,  # 转换器模型
        optimizer,  # 优化器
        train_dataloader,  # 数据加载器
        lr_scheduler  # 学习率调度器
    )
    # 由于训练数据加载器的大小可能已经改变，我们需要重新计算总的训练步骤
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)  
    # 如果覆盖了最大训练步骤，则更新最大训练步骤为训练轮数乘以每轮的更新步骤
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch  
    # 之后我们重新计算训练轮数
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)  

    # 我们需要初始化追踪器，并存储我们的配置
    # 追踪器会在主进程中自动初始化
    if accelerator.is_main_process:
        # 获取追踪器名称，如果未指定则使用默认名称
        tracker_name = args.tracker_name or "cogvideox-i2v-lora"  
        # 初始化追踪器，并传入配置参数
        accelerator.init_trackers(tracker_name, config=vars(args))  

    # 开始训练！
    # 计算每个设备上的总批量大小
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps  
    # 计算可训练参数的总数
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])  

    # 记录训练信息
    logger.info("***** Running training *****")  
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")  # 记录可训练参数数量
    logger.info(f"  Num examples = {len(train_dataset)}")  # 记录训练样本数量
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")  # 记录每轮的批次数
    logger.info(f"  Num epochs = {args.num_train_epochs}")  # 记录总训练轮数
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")  # 记录每个设备的即时批量大小
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")  # 记录总批量大小
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")  # 记录梯度累积步骤数
    logger.info(f"  Total optimization steps = {args.max_train_steps}")  # 记录总优化步骤数
    global_step = 0  # 初始化全局步骤
    first_epoch = 0  # 初始化第一轮

    # 可能加载来自之前保存的权重和状态
    if not args.resume_from_checkpoint:
        initial_global_step = 0  # 如果不从检查点恢复，初始全局步骤设为0
    else:
        # 如果指定的检查点不是"latest"，则提取路径
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)  
        else:
            # 获取最近的检查点
            dirs = os.listdir(args.output_dir)  # 列出输出目录中的文件
            dirs = [d for d in dirs if d.startswith("checkpoint")]  # 过滤出检查点文件
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))  # 按检查点编号排序
            path = dirs[-1] if len(dirs) > 0 else None  # 获取最新的检查点路径，如果没有则设为None

        # 检查点路径为空，打印错误信息并开始新的训练
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )  
            args.resume_from_checkpoint = None  # 将恢复检查点设为None
            initial_global_step = 0  # 初始化全局步骤为0
        else:
            # 从检查点恢复训练
            accelerator.print(f"Resuming from checkpoint {path}")  
            # 加载检查点状态
            accelerator.load_state(os.path.join(args.output_dir, path))  
            # 提取全局步骤数
            global_step = int(path.split("-")[1])  

            initial_global_step = global_step  # 初始化全局步骤为当前步骤
            first_epoch = global_step // num_update_steps_per_epoch  # 计算第一轮
    # 创建进度条，范围为最大训练步骤，初始值为全局步数
        progress_bar = tqdm(
            range(0, args.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # 仅在每台机器上显示一次进度条
            disable=not accelerator.is_local_main_process,
        )
        # 计算 VAE 空间缩放因子，根据块输出通道的数量
        vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
    
        # 获取模型配置，支持 DeepSpeed 训练
        model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config
    
        # 等待所有进程准备完毕
        accelerator.wait_for_everyone()
        # 结束训练
        accelerator.end_training()
# 判断当前模块是否是主程序入口
if __name__ == "__main__":
    # 获取命令行参数
    args = get_args()
    # 调用主函数并传递参数
    main(args)
```

# `.\cogvideo-finetune\finetune\train_cogvideox_lora.py`

```
# 版权声明，标明代码的版权所有者及相关信息
# Copyright 2024 The CogView team, Tsinghua University & ZhipuAI and The HuggingFace Team. All rights reserved.
#
# 按照 Apache 2.0 许可协议进行授权
# Licensed under the Apache License, Version 2.0 (the "License");
# 你不得在未遵循许可的情况下使用此文件
# you may not use this file except in compliance with the License.
# 你可以在以下网址获取许可证的副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，软件根据许可证分发是基于“按现状”原则
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何明示或暗示的保证或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 请参见许可证以获取特定语言适用的权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入命令行参数解析模块
import argparse
# 导入日志记录模块
import logging
# 导入数学运算模块
import math
# 导入操作系统相关的模块
import os
# 导入文件和目录操作模块
import shutil
# 导入路径处理模块
from pathlib import Path
# 导入类型提示相关的模块
from typing import List, Optional, Tuple, Union

# 导入 PyTorch 库
import torch
# 导入 Transformers 库
import transformers
# 从 accelerate 库导入加速器
from accelerate import Accelerator
# 从 accelerate.logging 导入获取日志记录器
from accelerate.logging import get_logger
# 从 accelerate.utils 导入分布式数据并行相关的工具
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
# 从 huggingface_hub 导入创建和上传模型的工具
from huggingface_hub import create_repo, upload_folder
# 从 peft 库导入 Lora 配置和模型状态字典处理工具
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
# 从 PyTorch 的工具数据集模块导入 DataLoader 和 Dataset
from torch.utils.data import DataLoader, Dataset
# 从 torchvision 导入数据预处理工具
from torchvision import transforms
# 导入进度条工具
from tqdm.auto import tqdm
# 从 Transformers 库导入自动标记器和 T5 模型
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

# 导入 Diffusers 库及其相关模块
import diffusers
# 导入 CogVideoX 相关的模型和调度器
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler, CogVideoXPipeline, CogVideoXTransformer3DModel
# 从 Diffusers 导入获取 3D 旋转位置嵌入的工具
from diffusers.models.embeddings import get_3d_rotary_pos_embed
# 从 Diffusers 导入获取调度器的工具
from diffusers.optimization import get_scheduler
# 从 Diffusers 的 CogVideoX 管道导入调整区域的工具
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
# 从 Diffusers 导入训练相关的工具
from diffusers.training_utils import (
    cast_training_params,  # 转换训练参数的工具
    free_memory,           # 释放内存的工具
)
# 从 Diffusers 导入工具集
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft, export_to_video, is_wandb_available
# 从 Diffusers 的 Hub 工具导入模型卡加载与创建工具
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
# 从 Diffusers 导入与 PyTorch 相关的工具
from diffusers.utils.torch_utils import is_compiled_module

# 如果可用，导入 Weights & Biases 库
if is_wandb_available():
    import wandb

# 检查是否安装了最小版本的 Diffusers，如果未安装将引发错误
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

# 获取日志记录器，记录当前模块的日志信息
logger = get_logger(__name__)


# 定义获取命令行参数的函数
def get_args():
    # 创建一个解析器，描述训练脚本的简单示例
    parser = argparse.ArgumentParser(description="Simple example of a training script for CogVideoX.")

    # 添加预训练模型信息参数
    parser.add_argument(
        "--pretrained_model_name_or_path",  # 参数名
        type=str,                            # 参数类型为字符串
        default=None,                        # 默认值为 None
        required=True,                       # 该参数为必需项
        help="Path to pretrained model or model identifier from huggingface.co/models.",  # 参数帮助信息
    )
    # 添加模型修订版本参数
    parser.add_argument(
        "--revision",                        # 参数名
        type=str,                            # 参数类型为字符串
        default=None,                        # 默认值为 None
        required=False,                      # 该参数为可选项
        help="Revision of pretrained model identifier from huggingface.co/models.",  # 参数帮助信息
    )
    # 添加模型变体参数
    parser.add_argument(
        "--variant",                         # 参数名
        type=str,                            # 参数类型为字符串
        default=None,                        # 默认值为 None
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",  # 参数帮助信息
    )
    # 添加命令行参数 --cache_dir，指定缓存目录
        parser.add_argument(
            "--cache_dir",
            type=str,
            default=None,
            help="The directory where the downloaded models and datasets will be stored.",
        )
    
        # 数据集信息
        # 添加命令行参数 --dataset_name，指定数据集名称
        parser.add_argument(
            "--dataset_name",
            type=str,
            default=None,
            help=(
                "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
                " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
                " or to a folder containing files that 🤗 Datasets can understand."
            ),
        )
        # 添加命令行参数 --dataset_config_name，指定数据集配置名称
        parser.add_argument(
            "--dataset_config_name",
            type=str,
            default=None,
            help="The config of the Dataset, leave as None if there's only one config.",
        )
        # 添加命令行参数 --instance_data_root，指定训练数据根目录
        parser.add_argument(
            "--instance_data_root",
            type=str,
            default=None,
            help=("A folder containing the training data."),
        )
        # 添加命令行参数 --video_column，指定包含视频的列名称
        parser.add_argument(
            "--video_column",
            type=str,
            default="video",
            help="The column of the dataset containing videos. Or, the name of the file in `--instance_data_root` folder containing the line-separated path to video data.",
        )
        # 添加命令行参数 --caption_column，指定包含提示文本的列名称
        parser.add_argument(
            "--caption_column",
            type=str,
            default="text",
            help="The column of the dataset containing the instance prompt for each video. Or, the name of the file in `--instance_data_root` folder containing the line-separated instance prompts.",
        )
        # 添加命令行参数 --id_token，指定标识符令牌
        parser.add_argument(
            "--id_token", type=str, default=None, help="Identifier token appended to the start of each prompt if provided."
        )
        # 添加命令行参数 --dataloader_num_workers，指定数据加载的子进程数量
        parser.add_argument(
            "--dataloader_num_workers",
            type=int,
            default=0,
            help=(
                "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
            ),
        )
    
        # 验证
        # 添加命令行参数 --validation_prompt，指定验证时使用的提示
        parser.add_argument(
            "--validation_prompt",
            type=str,
            default=None,
            help="One or more prompt(s) that is used during validation to verify that the model is learning. Multiple validation prompts should be separated by the '--validation_prompt_seperator' string.",
        )
        # 添加命令行参数 --validation_prompt_separator，指定验证提示的分隔符
        parser.add_argument(
            "--validation_prompt_separator",
            type=str,
            default=":::",
            help="String that separates multiple validation prompts",
        )
        # 添加命令行参数 --num_validation_videos，指定每个验证提示生成的视频数量
        parser.add_argument(
            "--num_validation_videos",
            type=int,
            default=1,
            help="Number of videos that should be generated during validation per `validation_prompt`.",
        )
        # 添加命令行参数 --validation_epochs，指定每隔多少个周期进行一次验证
        parser.add_argument(
            "--validation_epochs",
            type=int,
            default=50,
            help=(
                "Run validation every X epochs. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_videos`."
            ),
        )
    # 添加参数，指定指导尺度，用于采样验证视频
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6,
        help="The guidance scale to use while sampling validation videos.",
    )
    # 添加参数，指定是否使用动态配置标志
    parser.add_argument(
        "--use_dynamic_cfg",
        action="store_true",
        default=False,
        help="Whether or not to use the default cosine dynamic guidance schedule when sampling validation videos.",
    )

    # 训练信息
    # 添加参数，指定随机种子以确保训练的可重复性
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    # 添加参数，指定LoRA更新矩阵的维度
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    # 添加参数，指定LoRA权重更新的缩放因子
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=128,
        help=("The scaling factor to scale LoRA weight update. The actual scaling factor is `lora_alpha / rank`"),
    )
    # 添加参数，指定是否使用混合精度训练
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    # 添加参数，指定模型预测和检查点的输出目录
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cogvideox-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    # 添加参数，指定所有输入视频的高度
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="All input videos are resized to this height.",
    )
    # 添加参数，指定所有输入视频的宽度
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="All input videos are resized to this width.",
    )
    # 添加参数，指定所有输入视频的帧率
    parser.add_argument("--fps", type=int, default=8, help="All input videos will be used at this FPS.")
    # 添加参数，指定所有输入视频将被截断到的帧数
    parser.add_argument(
        "--max_num_frames", type=int, default=49, help="All input videos will be truncated to these many frames."
    )
    # 添加参数，指定从每个输入视频开始跳过的帧数
    parser.add_argument(
        "--skip_frames_start",
        type=int,
        default=0,
        help="Number of frames to skip from the beginning of each input video. Useful if training data contains intro sequences.",
    )
    # 添加参数，指定从每个输入视频末尾跳过的帧数
    parser.add_argument(
        "--skip_frames_end",
        type=int,
        default=0,
        help="Number of frames to skip from the end of each input video. Useful if training data contains outro sequences.",
    )
    # 添加参数，指定是否随机水平翻转视频
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip videos horizontally",
    )
    # 添加参数，指定训练数据加载器的批量大小（每个设备）
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    # 添加参数，指定训练的周期数
    parser.add_argument("--num_train_epochs", type=int, default=1)
    # 添加参数 `--max_train_steps`，用于指定训练的最大步数
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        # 帮助信息，说明该参数的用途
        help="Total number of training steps to perform. If provided, overrides `--num_train_epochs`.",
    )
    # 添加参数 `--checkpointing_steps`，用于指定每 X 次更新保存一次检查点
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        # 帮助信息，说明该参数的用途
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    # 添加参数 `--checkpoints_total_limit`，用于指定要存储的最大检查点数量
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        # 帮助信息，说明该参数的用途
        help=("Max number of checkpoints to store."),
    )
    # 添加参数 `--resume_from_checkpoint`，用于指定是否从之前的检查点恢复训练
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        # 帮助信息，说明该参数的用途
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    # 添加参数 `--gradient_accumulation_steps`，用于指定在执行反向传播和更新之前积累的更新步数
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        # 帮助信息，说明该参数的用途
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    # 添加参数 `--gradient_checkpointing`，用于指定是否使用梯度检查点以节省内存
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        # 帮助信息，说明该参数的用途
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    # 添加参数 `--learning_rate`，用于指定初始学习率
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        # 帮助信息，说明该参数的用途
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    # 添加参数 `--scale_lr`，用于指定是否按 GPU 数量、梯度积累步数和批量大小缩放学习率
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        # 帮助信息，说明该参数的用途
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    # 添加参数 `--lr_scheduler`，用于指定学习率调度器的类型
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        # 帮助信息，说明该参数的可选值
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    # 添加参数 `--lr_warmup_steps`，用于指定学习率调度器的预热步数
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, 
        # 帮助信息，说明该参数的用途
        help="Number of steps for the warmup in the lr scheduler."
    )
    # 添加参数 `--lr_num_cycles`，用于指定在 `cosine_with_restarts` 调度器中的硬重置次数
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        # 帮助信息，说明该参数的用途
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    # 添加参数 `--lr_power`，用于指定多项式调度器的幂因子
    parser.add_argument("--lr_power", type=float, default=1.0, 
        # 帮助信息，说明该参数的用途
        help="Power factor of the polynomial scheduler."
    )
    # 添加参数 `--enable_slicing`，用于指定是否使用 VAE 切片以节省内存
    parser.add_argument(
        "--enable_slicing",
        action="store_true",
        default=False,
        # 帮助信息，说明该参数的用途
        help="Whether or not to use VAE slicing for saving memory.",
    )
    # 添加一个命令行参数，用于启用或禁用 VAE 瓦片功能以节省内存
    parser.add_argument(
        "--enable_tiling",
        action="store_true",  # 指定该参数为布尔类型，默认值为 False
        default=False,
        help="Whether or not to use VAE tiling for saving memory.",  # 参数说明
    )

    # 优化器配置
    # 添加一个命令行参数，选择优化器类型
    parser.add_argument(
        "--optimizer",
        type=lambda s: s.lower(),  # 将输入转为小写
        default="adam",  # 默认使用 Adam 优化器
        choices=["adam", "adamw", "prodigy"],  # 可选的优化器类型
        help=("The optimizer type to use."),  # 参数说明
    )
    # 添加一个命令行参数，决定是否使用 8-bit Adam 优化器
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",  # 指定该参数为布尔类型
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",  # 参数说明
    )
    # 添加一个命令行参数，设置 Adam 优化器的 beta1 参数
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    # 添加一个命令行参数，设置 Adam 优化器的 beta2 参数
    parser.add_argument(
        "--adam_beta2", type=float, default=0.95, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    # 添加一个命令行参数，设置 Prodigy 优化器的 beta3 参数
    parser.add_argument(
        "--prodigy_beta3",
        type=float,  # 参数类型为浮点数
        default=None,  # 默认值为 None
        help="Coefficients for computing the Prodigy optimizer's stepsize using running averages. If set to None, uses the value of square root of beta2.",  # 参数说明
    )
    # 添加一个命令行参数，决定是否使用 AdamW 风格的解耦权重衰减
    parser.add_argument("--prodigy_decouple", action="store_true", help="Use AdamW style decoupled weight decay")
    # 添加一个命令行参数，设置 Adam 优化器的权重衰减
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    # 添加一个命令行参数，设置 Adam 和 Prodigy 优化器的 epsilon 值
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,  # 默认 epsilon 值
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",  # 参数说明
    )
    # 添加一个命令行参数，设置最大梯度范数
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # 添加一个命令行参数，决定是否开启 Adam 的偏差修正
    parser.add_argument("--prodigy_use_bias_correction", action="store_true", help="Turn on Adam's bias correction.")
    # 添加一个命令行参数，决定是否在暖启动阶段移除 lr 在 D 估计的分母中
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        action="store_true",  # 指定该参数为布尔类型
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage.",  # 参数说明
    )

    # 其他信息
    # 添加一个命令行参数，设置项目追踪器名称
    parser.add_argument("--tracker_name", type=str, default=None, help="Project tracker name")
    # 添加一个命令行参数，决定是否将模型推送到 Hub
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    # 添加一个命令行参数，设置推送到模型 Hub 时使用的 token
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    # 添加一个命令行参数，设置与本地 output_dir 同步的仓库名称
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,  # 默认值为 None
        help="The name of the repository to keep in sync with the local `output_dir`.",  # 参数说明
    )
    # 添加一个命令行参数，设置日志存储目录
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",  # 默认日志目录
        help="Directory where logs are stored.",  # 参数说明
    )
    # 添加一个命令行参数，决定是否允许在 Ampere GPU 上使用 TF32
    parser.add_argument(
        "--allow_tf32",
        action="store_true",  # 指定该参数为布尔类型
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"  # 参数说明
        ),
    )
    # 添加命令行参数 '--report_to' 的配置
        parser.add_argument(
            # 参数名称
            "--report_to",
            # 参数类型为字符串
            type=str,
            # 默认值为 None
            default=None,
            # 参数帮助信息，解释该参数的用途
            help=(
                # 提供支持的平台和默认值说明
                'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                # 继续帮助信息，说明可选的平台
                ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
            ),
        )
    
    # 解析命令行参数并返回结果
        return parser.parse_args()
# 定义一个视频数据集类，继承自 Dataset
class VideoDataset(Dataset):
    # 初始化方法，设置数据集的各种参数
    def __init__(
        self,
        instance_data_root: Optional[str] = None,  # 实例数据根目录，可选
        dataset_name: Optional[str] = None,  # 数据集名称，可选
        dataset_config_name: Optional[str] = None,  # 数据集配置名称，可选
        caption_column: str = "text",  # 描述文本所在列名，默认为 "text"
        video_column: str = "video",  # 视频数据所在列名，默认为 "video"
        height: int = 480,  # 视频高度，默认为 480
        width: int = 720,  # 视频宽度，默认为 720
        fps: int = 8,  # 每秒帧数，默认为 8
        max_num_frames: int = 49,  # 最大帧数，默认为 49
        skip_frames_start: int = 0,  # 开始跳过的帧数，默认为 0
        skip_frames_end: int = 0,  # 结束跳过的帧数，默认为 0
        cache_dir: Optional[str] = None,  # 缓存目录，可选
        id_token: Optional[str] = None,  # ID 令牌，可选
    ) -> None:
        super().__init__()  # 调用父类初始化方法

        # 如果提供了实例数据根目录，则将其转换为 Path 对象
        self.instance_data_root = Path(instance_data_root) if instance_data_root is not None else None
        # 设置数据集名称
        self.dataset_name = dataset_name
        # 设置数据集配置名称
        self.dataset_config_name = dataset_config_name
        # 设置描述文本所在列名
        self.caption_column = caption_column
        # 设置视频数据所在列名
        self.video_column = video_column
        # 设置视频高度
        self.height = height
        # 设置视频宽度
        self.width = width
        # 设置每秒帧数
        self.fps = fps
        # 设置最大帧数
        self.max_num_frames = max_num_frames
        # 设置开始跳过的帧数
        self.skip_frames_start = skip_frames_start
        # 设置结束跳过的帧数
        self.skip_frames_end = skip_frames_end
        # 设置缓存目录
        self.cache_dir = cache_dir
        # 设置 ID 令牌，如果未提供则为空字符串
        self.id_token = id_token or ""

        # 如果提供了数据集名称，则从数据集中加载实例提示和视频路径
        if dataset_name is not None:
            self.instance_prompts, self.instance_video_paths = self._load_dataset_from_hub()
        # 否则从本地路径加载实例提示和视频路径
        else:
            self.instance_prompts, self.instance_video_paths = self._load_dataset_from_local_path()

        # 计算实例视频的数量
        self.num_instance_videos = len(self.instance_video_paths)
        # 检查实例提示和视频路径数量是否匹配
        if self.num_instance_videos != len(self.instance_prompts):
            raise ValueError(
                # 抛出错误，提示实例提示和视频数量不匹配
                f"Expected length of instance prompts and videos to be the same but found {len(self.instance_prompts)=} and {len(self.instance_video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )

        # 预处理数据以获取实例视频
        self.instance_videos = self._preprocess_data()

    # 返回数据集的长度
    def __len__(self):
        return self.num_instance_videos

    # 根据索引获取实例数据
    def __getitem__(self, index):
        return {
            # 返回组合后的实例提示
            "instance_prompt": self.id_token + self.instance_prompts[index],
            # 返回对应的实例视频
            "instance_video": self.instance_videos[index],
        }
    # 从数据集中心加载数据集的私有方法
    def _load_dataset_from_hub(self):
        try:
            # 尝试导入 datasets 库
            from datasets import load_dataset
        except ImportError:
            # 如果导入失败，抛出 ImportError，并提供安装提示
            raise ImportError(
                "You are trying to load your data using the datasets library. If you wish to train using custom "
                "captions please install the datasets library: `pip install datasets`. If you wish to load a "
                "local folder containing images only, specify --instance_data_root instead."
            )

        # 从数据集中心下载并加载数据集，关于如何加载自定义图像的信息见链接
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
        dataset = load_dataset(
            self.dataset_name,  # 数据集名称
            self.dataset_config_name,  # 数据集配置名称
            cache_dir=self.cache_dir,  # 缓存目录
        )
        # 获取训练集的列名
        column_names = dataset["train"].column_names

        # 如果没有指定视频列
        if self.video_column is None:
            # 默认使用列名列表中的第一个列名作为视频列
            video_column = column_names[0]
            # 记录使用默认视频列的信息
            logger.info(f"`video_column` defaulting to {video_column}")
        else:
            # 如果已指定视频列，则使用指定的列名
            video_column = self.video_column
            # 检查指定的视频列是否在列名中
            if video_column not in column_names:
                # 如果不在，抛出 ValueError
                raise ValueError(
                    f"`--video_column` value '{video_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        # 如果没有指定字幕列
        if self.caption_column is None:
            # 默认使用列名列表中的第二个列名作为字幕列
            caption_column = column_names[1]
            # 记录使用默认字幕列的信息
            logger.info(f"`caption_column` defaulting to {caption_column}")
        else:
            # 如果已指定字幕列，则使用指定的列名
            caption_column = self.caption_column
            # 检查指定的字幕列是否在列名中
            if self.caption_column not in column_names:
                # 如果不在，抛出 ValueError
                raise ValueError(
                    f"`--caption_column` value '{self.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        # 从训练集中提取实例提示（字幕）
        instance_prompts = dataset["train"][caption_column]
        # 根据视频列的文件路径创建视频实例列表
        instance_videos = [Path(self.instance_data_root, filepath) for filepath in dataset["train"][video_column]]

        # 返回实例提示和视频实例列表
        return instance_prompts, instance_videos
    # 从本地路径加载数据集
        def _load_dataset_from_local_path(self):
            # 检查实例数据根目录是否存在
            if not self.instance_data_root.exists():
                # 如果不存在，则抛出值错误
                raise ValueError("Instance videos root folder does not exist")
    
            # 构建提示文件路径
            prompt_path = self.instance_data_root.joinpath(self.caption_column)
            # 构建视频文件路径
            video_path = self.instance_data_root.joinpath(self.video_column)
    
            # 检查提示文件是否存在且是文件
            if not prompt_path.exists() or not prompt_path.is_file():
                # 如果不是，则抛出值错误
                raise ValueError(
                    "Expected `--caption_column` to be path to a file in `--instance_data_root` containing line-separated text prompts."
                )
            # 检查视频文件是否存在且是文件
            if not video_path.exists() or not video_path.is_file():
                # 如果不是，则抛出值错误
                raise ValueError(
                    "Expected `--video_column` to be path to a file in `--instance_data_root` containing line-separated paths to video data in the same directory."
                )
    
            # 打开提示文件并读取每行，去除首尾空白，形成提示列表
            with open(prompt_path, "r", encoding="utf-8") as file:
                instance_prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
            # 打开视频文件并读取每行，去除首尾空白，形成视频路径列表
            with open(video_path, "r", encoding="utf-8") as file:
                instance_videos = [
                    self.instance_data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0
                ]
    
            # 检查视频路径列表中是否有无效文件路径
            if any(not path.is_file() for path in instance_videos):
                # 如果有，则抛出值错误
                raise ValueError(
                    "Expected '--video_column' to be a path to a file in `--instance_data_root` containing line-separated paths to video data but found atleast one path that is not a valid file."
                )
    
            # 返回实例提示和视频路径列表
            return instance_prompts, instance_videos
    # 定义数据预处理的方法
    def _preprocess_data(self):
        # 尝试导入 decord 库
        try:
            import decord
        # 如果导入失败，则抛出错误，提示需要安装 decord
        except ImportError:
            raise ImportError(
                "The `decord` package is required for loading the video dataset. Install with `pip install decord`"
            )

        # 设置 decord 使用 PyTorch 作为后端
        decord.bridge.set_bridge("torch")

        # 初始化一个空列表，用于存储视频数据
        videos = []
        # 定义训练时的转换操作
        train_transforms = transforms.Compose(
            [
                # 将像素值归一化到 [-1, 1] 范围
                transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0),
            ]
        )

        # 遍历每个视频文件的路径
        for filename in self.instance_video_paths:
            # 使用 decord 读取视频，指定宽度和高度
            video_reader = decord.VideoReader(uri=filename.as_posix(), width=self.width, height=self.height)
            # 获取视频的帧数
            video_num_frames = len(video_reader)

            # 计算开始帧和结束帧的索引
            start_frame = min(self.skip_frames_start, video_num_frames)
            end_frame = max(0, video_num_frames - self.skip_frames_end)
            # 如果结束帧小于等于开始帧，只获取开始帧的帧数据
            if end_frame <= start_frame:
                frames = video_reader.get_batch([start_frame])
            # 如果要获取的帧数量在允许的最大范围内
            elif end_frame - start_frame <= self.max_num_frames:
                frames = video_reader.get_batch(list(range(start_frame, end_frame)))
            # 如果要获取的帧数量超过最大限制，则按步长获取
            else:
                indices = list(range(start_frame, end_frame, (end_frame - start_frame) // self.max_num_frames))
                frames = video_reader.get_batch(indices)

            # 确保帧数量不超过最大限制
            frames = frames[: self.max_num_frames]
            # 获取当前选择的帧数量
            selected_num_frames = frames.shape[0]

            # 选择前 (4k + 1) 帧，以满足 VAE 的需求
            remainder = (3 + (selected_num_frames % 4)) % 4
            # 如果帧数量不是 4 的倍数，去掉多余的帧
            if remainder != 0:
                frames = frames[:-remainder]
            # 更新选择的帧数量
            selected_num_frames = frames.shape[0]

            # 确保选择的帧数量减 1 是 4 的倍数
            assert (selected_num_frames - 1) % 4 == 0

            # 应用训练转换操作
            frames = frames.float()
            # 将每一帧应用转换并堆叠成一个新的张量
            frames = torch.stack([train_transforms(frame) for frame in frames], dim=0)
            # 将处理后的帧按照 [F, C, H, W] 的顺序排列并存入视频列表
            videos.append(frames.permute(0, 3, 1, 2).contiguous())  # [F, C, H, W]

        # 返回处理后的视频数据
        return videos
# 保存模型卡片信息
def save_model_card(
    repo_id: str,  # 模型仓库的标识
    videos=None,  # 可选的视频列表
    base_model: str = None,  # 基础模型的名称，默认为 None
    validation_prompt=None,  # 验证时使用的提示语
    repo_folder=None,  # 模型存储的文件夹路径
    fps=8,  # 视频帧率，默认为 8
):
    widget_dict = []  # 初始化小部件字典，用于存储视频信息
    if videos is not None:  # 检查视频列表是否不为空
        for i, video in enumerate(videos):  # 遍历视频列表，获取索引和视频对象
            # 将视频导出到指定路径，并设置帧率
            export_to_video(video, os.path.join(repo_folder, f"final_video_{i}.mp4", fps=fps))
            # 将视频信息添加到小部件字典中
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"video_{i}.mp4"}}
            )

    # 定义模型描述信息，包括模型 ID 和基础模型名称
    model_description = f"""
# CogVideoX LoRA - {repo_id}

<Gallery />

## Model description

These are {repo_id} LoRA weights for {base_model}.

The weights were trained using the [CogVideoX Diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/cogvideo/train_cogvideox_lora.py).

Was LoRA for the text encoder enabled? No.

## Download model

[Download the *.safetensors LoRA]({repo_id}/tree/main) in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)


from diffusers import CogVideoXPipeline  # 导入 CogVideoXPipeline 类
import torch  # 导入 PyTorch 库

# 从预训练模型中加载管道，设置数据类型为 bfloat16，并将其转移到 CUDA
pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16).to("cuda")
# 加载 LoRA 权重，指定权重文件名和适配器名称
pipe.load_lora_weights("{repo_id}", weight_name="pytorch_lora_weights.safetensors", adapter_name=["cogvideox-lora"])

# LoRA 适配器权重是基于训练时使用的参数确定的。
# 在这种情况下，假设 `--lora_alpha` 是 32，`--rank` 是 64。
# 可以根据训练中使用的值进行调整，以减小或放大 LoRA 的效果
# 超过一定的容忍度，可能会注意到没有效果或溢出。
pipe.set_adapters(["cogvideox-lora"], [32 / 64])

# 使用管道生成视频，传入验证提示，设置指导比例，并启用动态配置
video = pipe("{validation_prompt}", guidance_scale=6, use_dynamic_cfg=True).frames[0]


# 更多细节，包括权重、合并和融合 LoRA，请查看 [diffusers 中加载 LoRA 的文档](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## License

请遵守 [此处](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE) 和 [此处](https://huggingface.co/THUDM/CogVideoX-2b/blob/main/LICENSE) 中描述的许可条款。
"""
    # 加载或创建模型卡片，传入必要参数
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,  # 模型 ID 或路径
        from_training=True,  # 指示从训练生成
        license="other",  # 设置许可证类型
        base_model=base_model,  # 基础模型名称
        prompt=validation_prompt,  # 验证提示
        model_description=model_description,  # 模型描述
        widget=widget_dict,  # 小部件信息
    )
    # 定义标签列表，用于标识模型特性
    tags = [
        "text-to-video",  # 文本转视频
        "diffusers-training",  # Diffusers 训练
        "diffusers",  # Diffusers
        "lora",  # LoRA
        "cogvideox",  # CogVideoX
        "cogvideox-diffusers",  # CogVideoX Diffusers
        "template:sd-lora",  # 模板类型
    ]

    # 填充模型卡片的标签
    model_card = populate_model_card(model_card, tags=tags)
    # 保存模型卡片到指定路径
    model_card.save(os.path.join(repo_folder, "README.md"))


# 记录验证结果
def log_validation(
    pipe,  # 视频生成管道
    args,  # 其他参数
    accelerator,  # 加速器实例
    pipeline_args,  # 管道参数
    epoch,  # 当前训练的轮次
    is_final_validation: bool = False,  # 是否为最终验证
):
    # 记录正在运行验证的信息，包括生成视频的数量和提示内容
        logger.info(
            f"Running validation... \n Generating {args.num_validation_videos} videos with prompt: {pipeline_args['prompt']}."
        )
        # 创建一个空字典，用于存储调度器的参数
        scheduler_args = {}
    
        # 检查调度器配置中是否包含方差类型
        if "variance_type" in pipe.scheduler.config:
            # 获取方差类型
            variance_type = pipe.scheduler.config.variance_type
    
            # 如果方差类型是“learned”或“learned_range”，则将其更改为“fixed_small”
            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"
    
            # 将方差类型添加到调度器参数中
            scheduler_args["variance_type"] = variance_type
    
        # 根据调度器配置和参数创建新的调度器
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
        # 将管道移动到指定的设备上
        pipe = pipe.to(accelerator.device)
        # 关闭进度条配置（注释掉）
        # pipe.set_progress_bar_config(disable=True)
    
        # 运行推理，创建随机数生成器，设置种子
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    
        # 初始化一个空列表以存储生成的视频
        videos = []
        # 根据需要生成指定数量的视频
        for _ in range(args.num_validation_videos):
            # 调用管道生成视频，获取第一帧
            video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]
            # 将生成的视频添加到列表中
            videos.append(video)
    
        # 遍历所有跟踪器
        for tracker in accelerator.trackers:
            # 根据是否为最终验证选择阶段名称
            phase_name = "test" if is_final_validation else "validation"
            # 检查跟踪器名称是否为“wandb”
            if tracker.name == "wandb":
                # 初始化视频文件名列表
                video_filenames = []
                # 遍历生成的视频列表
                for i, video in enumerate(videos):
                    # 处理提示文本以创建安全的文件名
                    prompt = (
                        pipeline_args["prompt"][:25]
                        .replace(" ", "_")
                        .replace(" ", "_")
                        .replace("'", "_")
                        .replace('"', "_")
                        .replace("/", "_")
                    )
                    # 创建视频文件的完整路径
                    filename = os.path.join(args.output_dir, f"{phase_name}_video_{i}_{prompt}.mp4")
                    # 将视频导出为文件
                    export_to_video(video, filename, fps=8)
                    # 将文件名添加到列表中
                    video_filenames.append(filename)
    
                # 记录视频到 wandb
                tracker.log(
                    {
                        phase_name: [
                            wandb.Video(filename, caption=f"{i}: {pipeline_args['prompt']}")
                            for i, filename in enumerate(video_filenames)
                        ]
                    }
                )
    
        # 释放内存
        free_memory()
    
        # 返回生成的视频列表
        return videos
# 获取 T5 模型的提示嵌入
def _get_t5_prompt_embeds(
    # 定义 T5 令牌化器
    tokenizer: T5Tokenizer,
    # 定义 T5 编码器模型
    text_encoder: T5EncoderModel,
    # 提示文本，字符串或字符串列表
    prompt: Union[str, List[str]],
    # 每个提示生成视频的数量，默认为 1
    num_videos_per_prompt: int = 1,
    # 最大序列长度，默认为 226
    max_sequence_length: int = 226,
    # 指定设备（如 GPU），可选
    device: Optional[torch.device] = None,
    # 指定数据类型（如 float32），可选
    dtype: Optional[torch.dtype] = None,
    # 预先提供的文本输入 ID，可选
    text_input_ids=None,
):
    # 如果提示是字符串，则将其转换为列表
    prompt = [prompt] if isinstance(prompt, str) else prompt
    # 获取提示的批处理大小
    batch_size = len(prompt)

    # 如果提供了令牌化器
    if tokenizer is not None:
        # 使用令牌化器对提示进行编码，生成张量
        text_inputs = tokenizer(
            prompt,
            padding="max_length",  # 填充到最大长度
            max_length=max_sequence_length,  # 最大长度
            truncation=True,  # 超过最大长度时截断
            add_special_tokens=True,  # 添加特殊令牌
            return_tensors="pt",  # 返回 PyTorch 张量
        )
        # 获取文本输入 ID
        text_input_ids = text_inputs.input_ids
    else:
        # 如果没有令牌化器且未提供文本输入 ID，抛出错误
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    # 将文本输入 ID 输入编码器以获取提示嵌入
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    # 将嵌入转换为指定的数据类型和设备
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # 为每个提示生成的每个视频复制文本嵌入，使用适合 MPS 的方法
    _, seq_len, _ = prompt_embeds.shape  # 获取嵌入的形状
    # 重复嵌入以匹配每个提示的视频数量
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    # 将嵌入调整为新的形状
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    # 返回最终的提示嵌入
    return prompt_embeds


# 编码提示，生成其嵌入
def encode_prompt(
    # 定义 T5 令牌化器
    tokenizer: T5Tokenizer,
    # 定义 T5 编码器模型
    text_encoder: T5EncoderModel,
    # 提示文本，字符串或字符串列表
    prompt: Union[str, List[str]],
    # 每个提示生成视频的数量，默认为 1
    num_videos_per_prompt: int = 1,
    # 最大序列长度，默认为 226
    max_sequence_length: int = 226,
    # 指定设备（如 GPU），可选
    device: Optional[torch.device] = None,
    # 指定数据类型（如 float32），可选
    dtype: Optional[torch.dtype] = None,
    # 预先提供的文本输入 ID，可选
    text_input_ids=None,
):
    # 如果提示是字符串，则将其转换为列表
    prompt = [prompt] if isinstance(prompt, str) else prompt
    # 调用内部函数获取提示嵌入
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    # 返回提示嵌入
    return prompt_embeds


# 计算提示的嵌入
def compute_prompt_embeddings(
    # 定义 T5 令牌化器
    tokenizer, 
    # 定义 T5 编码器模型
    text_encoder, 
    # 提示文本
    prompt, 
    # 最大序列长度
    max_sequence_length, 
    # 指定设备（如 GPU）
    device, 
    # 指定数据类型（如 float32）
    dtype, 
    # 是否需要梯度计算，默认为 False
    requires_grad: bool = False
):
    # 如果需要计算梯度
    if requires_grad:
        # 调用 encode_prompt 函数获取提示嵌入
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompt,
            num_videos_per_prompt=1,  # 默认为 1
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    else:
        # 如果不需要梯度计算，使用 no_grad 上下文管理器
        with torch.no_grad():
            # 调用 encode_prompt 函数获取提示嵌入
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompt,
                num_videos_per_prompt=1,  # 默认为 1
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
    # 返回计算得到的提示嵌入
    return prompt_embeds


# 准备旋转位置嵌入
def prepare_rotary_positional_embeddings(
    # 嵌入的高度
    height: int,
    # 嵌入的宽度
    width: int,
    # 帧的数量
    num_frames: int,
    # 空间 VAE 缩放因子，默认为 8
    vae_scale_factor_spatial: int = 8,
    # 贴片大小，默认为 2
    patch_size: int = 2,
    # 注意力头的维度，默认为 64
    attention_head_dim: int = 64,
    # 可选参数，指定设备类型（如 CPU 或 GPU），默认为 None
    device: Optional[torch.device] = None,
    # 基础高度，默认为 480 像素
    base_height: int = 480,
    # 基础宽度，默认为 720 像素
    base_width: int = 720,
# 函数返回一个包含两个张量的元组
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 计算网格的高度
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    # 计算网格的宽度
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    # 计算基础宽度
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    # 计算基础高度
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    # 获取网格的裁剪区域坐标
    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    # 计算3D旋转位置嵌入的余弦和正弦频率
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )

    # 将余弦频率张量移动到指定设备
    freqs_cos = freqs_cos.to(device=device)
    # 将正弦频率张量移动到指定设备
    freqs_sin = freqs_sin.to(device=device)
    # 返回余弦和正弦频率张量
    return freqs_cos, freqs_sin


# 创建优化器的函数，接受参数和优化参数
def get_optimizer(args, params_to_optimize, use_deepspeed: bool = False):
    # 使用 DeepSpeed 优化器
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        # 返回一个虚拟优化器以供使用
        return DummyOptim(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )

    # 优化器创建部分
    supported_optimizers = ["adam", "adamw", "prodigy"]
    # 检查所选优化器是否受支持
    if args.optimizer not in supported_optimizers:
        # 记录不支持的优化器警告
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        # 默认优化器设置为 AdamW
        args.optimizer = "adamw"

    # 检查8位Adam的使用条件
    if args.use_8bit_adam and not (args.optimizer.lower() not in ["adam", "adamw"]):
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    # 检查是否使用8位Adam优化器
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            # 如果未安装bitsandbytes，抛出导入错误
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    # 创建AdamW优化器
    if args.optimizer.lower() == "adamw":
        optimizer_class = bnb.optim.AdamW8bit if args.use_8bit_adam else torch.optim.AdamW

        # 初始化优化器
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    # 创建Adam优化器
    elif args.optimizer.lower() == "adam":
        optimizer_class = bnb.optim.Adam8bit if args.use_8bit_adam else torch.optim.Adam

        # 初始化优化器
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    # 检查优化器参数是否为 "prodigy"（不区分大小写）
    elif args.optimizer.lower() == "prodigy":
        # 尝试导入 prodigyopt 库
        try:
            import prodigyopt
        # 如果导入失败，抛出 ImportError 并提示安装命令
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        # 设置优化器类为 Prodigy
        optimizer_class = prodigyopt.Prodigy

        # 检查学习率是否过低，并发出警告
        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        # 初始化优化器对象，传入所需参数
        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # 返回创建的优化器对象
    return optimizer
# 主函数，接收命令行参数
def main(args):
    # 如果报告目标是 "wandb" 且提供了 hub_token，则抛出错误
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    # 检查 MPS 后端是否可用，并且混合精度设置为 bf16，若是则抛出错误
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # 由于 pytorch#99272，MPS 目前不支持 bfloat16。
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # 创建日志目录的路径
    logging_dir = Path(args.output_dir, args.logging_dir)

    # 创建项目配置，包括项目目录和日志目录
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    # 配置分布式数据并行的参数
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # 创建加速器实例，配置其参数
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # 禁用 MPS 的自动混合精度
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # 如果报告目标是 "wandb"，检查其是否可用
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # 配置日志，确保每个进程都能记录调试信息
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # 记录加速器的状态，所有进程都能看到
    logger.info(accelerator.state, main_process_only=False)
    # 设置主进程的日志级别
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # 如果提供了种子，则设置随机种子
    if args.seed is not None:
        set_seed(args.seed)

    # 处理仓库的创建
    if accelerator.is_main_process:
        # 如果输出目录不为空，创建输出目录
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        # 如果需要推送到 hub，创建仓库
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # 准备模型和调度器
    # 从预训练模型路径加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    # 从预训练模型路径加载文本编码器
    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

    # CogVideoX-2b 权重存储为 float16
    # CogVideoX-5b 和 CogVideoX-5b-I2V 权重存储为 bfloat16
    # 根据预训练模型名称选择加载的数据类型，支持 bfloat16 或 float16
    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    # 从预训练模型中加载 3D 变换器模型
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,  # 预训练模型的路径
        subfolder="transformer",  # 指定子文件夹
        torch_dtype=load_dtype,  # 设置数据类型
        revision=args.revision,  # 指定版本
        variant=args.variant,  # 指定变体
    )

    # 从预训练模型中加载 VAE 模型
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )

    # 从预训练模型中加载调度器
    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # 如果启用了切片功能，则启用 VAE 的切片
    if args.enable_slicing:
        vae.enable_slicing()
    # 如果启用了平铺功能，则启用 VAE 的平铺
    if args.enable_tiling:
        vae.enable_tiling()

    # 只训练额外的适配器 LoRA 层
    text_encoder.requires_grad_(False)  # 禁用文本编码器的梯度计算
    transformer.requires_grad_(False)  # 禁用变换器的梯度计算
    vae.requires_grad_(False)  # 禁用 VAE 的梯度计算

    # 对于混合精度训练，将所有非可训练权重（VAE、文本编码器和变换器）转换为半精度
    # 因为这些权重仅用于推理，因此不需要保持全精度
    weight_dtype = torch.float32  # 默认权重数据类型为 float32
    if accelerator.state.deepspeed_plugin:  # 如果使用 DeepSpeed
        # DeepSpeed 处理精度，使用 DeepSpeed 配置中的设置
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16  # 如果启用 fp16，则设置权重为 float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.float16  # 如果启用 bf16，则设置权重为 float16
    else:  # 如果不使用 DeepSpeed
        if accelerator.mixed_precision == "fp16":  # 如果混合精度为 fp16
            weight_dtype = torch.float16  # 设置权重为 float16
        elif accelerator.mixed_precision == "bf16":  # 如果混合精度为 bf16
            weight_dtype = torch.bfloat16  # 设置权重为 bfloat16

    # 如果 MPS 可用且权重类型为 bfloat16，抛出错误
    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # 由于 pytorch#99272，MPS 目前不支持 bfloat16。
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # 将文本编码器、变换器和 VAE 转移到加速器设备，并设置权重数据类型
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # 如果启用了梯度检查点，则启用变换器的梯度检查点
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # 现在将新的 LoRA 权重添加到注意力层
    transformer_lora_config = LoraConfig(
        r=args.rank,  # LoRA 的秩
        lora_alpha=args.lora_alpha,  # LoRA 的 alpha 值
        init_lora_weights=True,  # 初始化 LoRA 权重
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # 目标模块
    )
    # 将 LoRA 适配器添加到变换器中
    transformer.add_adapter(transformer_lora_config)

    # 定义一个函数，用于解包模型
    def unwrap_model(model):
        # 解包加速器中的模型
        model = accelerator.unwrap_model(model)
        # 如果是编译模块，则返回其原始模块
        model = model._orig_mod if is_compiled_module(model) else model
        return model  # 返回解包后的模型
    # 创建自定义的保存和加载钩子，以便 `accelerator.save_state(...)` 可以序列化为良好的格式
    def save_model_hook(models, weights, output_dir):
        # 检查当前进程是否为主进程
        if accelerator.is_main_process:
            # 初始化要保存的变换器 LoRA 层变量
            transformer_lora_layers_to_save = None
    
            # 遍历所有模型
            for model in models:
                # 检查模型是否是变换器的实例
                if isinstance(model, type(unwrap_model(transformer))):
                    # 获取变换器模型的状态字典
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    # 如果模型类型不匹配，抛出异常
                    raise ValueError(f"unexpected save model: {model.__class__}")
    
                # 确保从权重中移除相应的权重，以避免重复保存
                weights.pop()
    
            # 保存 LoRA 权重到指定输出目录
            CogVideoXPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )
    
        # 定义加载模型的钩子
        def load_model_hook(models, input_dir):
            # 初始化变换器变量
            transformer_ = None
    
            # 当模型列表非空时持续执行
            while len(models) > 0:
                # 弹出模型
                model = models.pop()
    
                # 检查模型是否是变换器的实例
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_ = model
                else:
                    # 如果模型类型不匹配，抛出异常
                    raise ValueError(f"Unexpected save model: {model.__class__}")
    
            # 从指定输入目录获取 LoRA 状态字典
            lora_state_dict = CogVideoXPipeline.lora_state_dict(input_dir)
    
            # 创建变换器状态字典，仅保留以 "transformer." 开头的键
            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
            }
            # 转换 UNet 状态字典为 PEFT 格式
            transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
            # 设置 PEFT 模型状态字典并获取不兼容的键
            incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
            # 检查是否存在意外的键
            if incompatible_keys is not None:
                # 获取意外的键
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    # 记录警告日志
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )
    
            # 确保可训练参数为 float32 类型
            if args.mixed_precision == "fp16":
                # 仅将可训练参数（LoRA）提升为 fp32
                cast_training_params([transformer_])
    
        # 注册保存状态前钩子
        accelerator.register_save_state_pre_hook(save_model_hook)
        # 注册加载状态前钩子
        accelerator.register_load_state_pre_hook(load_model_hook)
    
        # 如果允许使用 TF32，则在 Ampere GPU 上启用更快的训练
        if args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
    
        # 如果需要缩放学习率，则进行相应调整
        if args.scale_lr:
            args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
            )
    # 确保可训练参数为 float32 类型
    if args.mixed_precision == "fp16":
        # 仅将可训练参数（LoRA）上升为 fp32 类型
        cast_training_params([transformer], dtype=torch.float32)

    # 过滤出需要梯度更新的参数
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # 优化器的参数字典
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    # 将优化参数放入列表中
    params_to_optimize = [transformer_parameters_with_lr]

    # 判断是否使用 DeepSpeed 优化器
    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    # 判断是否使用 DeepSpeed 调度器
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    # 获取优化器实例
    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    # 创建数据集和数据加载器
    train_dataset = VideoDataset(
        instance_data_root=args.instance_data_root,  # 实例数据根目录
        dataset_name=args.dataset_name,  # 数据集名称
        dataset_config_name=args.dataset_config_name,  # 数据集配置名称
        caption_column=args.caption_column,  # 描述列名称
        video_column=args.video_column,  # 视频列名称
        height=args.height,  # 视频高度
        width=args.width,  # 视频宽度
        fps=args.fps,  # 帧率
        max_num_frames=args.max_num_frames,  # 最大帧数
        skip_frames_start=args.skip_frames_start,  # 开始跳过的帧数
        skip_frames_end=args.skip_frames_end,  # 结束跳过的帧数
        cache_dir=args.cache_dir,  # 缓存目录
        id_token=args.id_token,  # ID 令牌
    )

    # 定义编码视频的函数
    def encode_video(video):
        # 将视频转移到设备并增加维度
        video = video.to(accelerator.device, dtype=vae.dtype).unsqueeze(0)
        # 调整视频维度顺序
        video = video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        # 使用 VAE 编码视频并获取潜在分布
        latent_dist = vae.encode(video).latent_dist
        return latent_dist

    # 对数据集中每个实例视频进行编码
    train_dataset.instance_videos = [encode_video(video) for video in train_dataset.instance_videos]

    # 定义整理函数以组合数据
    def collate_fn(examples):
        # 提取视频样本并进行缩放
        videos = [example["instance_video"].sample() * vae.config.scaling_factor for example in examples]
        # 提取对应的提示文本
        prompts = [example["instance_prompt"] for example in examples]

        # 将视频张量合并
        videos = torch.cat(videos)
        # 确保视频张量连续并转换为 float 类型
        videos = videos.to(memory_format=torch.contiguous_format).float()

        return {
            "videos": videos,  # 返回视频张量
            "prompts": prompts,  # 返回提示文本
        }

    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,  # 使用的数据集
        batch_size=args.train_batch_size,  # 每批次的大小
        shuffle=True,  # 是否打乱数据
        collate_fn=collate_fn,  # 自定义整理函数
        num_workers=args.dataloader_num_workers,  # 使用的工作线程数
    )

    # 调度器和训练步骤的数学计算
    overrode_max_train_steps = False  # 标记是否覆盖最大训练步数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)  # 每个 epoch 的更新步骤数
    if args.max_train_steps is None:
        # 如果未指定最大训练步数，则根据训练周期计算
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True  # 设置标记为 True
    # 检查是否使用 DeepSpeed 调度器
    if use_deepspeed_scheduler:
        # 从 accelerate.utils 导入 DummyScheduler 类
        from accelerate.utils import DummyScheduler

        # 创建一个 DummyScheduler 实例，用于学习率调度
        lr_scheduler = DummyScheduler(
            # 设置调度器名称
            name=args.lr_scheduler,
            # 传入优化器
            optimizer=optimizer,
            # 设置总训练步数
            total_num_steps=args.max_train_steps * accelerator.num_processes,
            # 设置预热步数
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        )
    else:
        # 如果不使用 DeepSpeed，调用 get_scheduler 函数获取学习率调度器
        lr_scheduler = get_scheduler(
            # 传入调度器名称
            args.lr_scheduler,
            # 传入优化器
            optimizer=optimizer,
            # 设置预热步数
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            # 设置总训练步数
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            # 设置学习率循环次数
            num_cycles=args.lr_num_cycles,
            # 设置学习率衰减的幂
            power=args.lr_power,
        )

    # 使用 accelerator 准备所有组件
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        # 准备变换器、优化器、训练数据加载器和学习率调度器
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # 需要重新计算总训练步数，因为训练数据加载器的大小可能已经改变
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # 如果覆盖了最大训练步数，则重新计算
    if overrode_max_train_steps:
        # 根据训练周期和更新步骤计算最大训练步数
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # 随后重新计算训练周期数
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # 初始化跟踪器并存储配置
    # 跟踪器在主进程中自动初始化
    if accelerator.is_main_process:
        # 获取跟踪器名称，如果未指定则使用默认名称
        tracker_name = args.tracker_name or "cogvideox-lora"
        # 初始化跟踪器，并传入配置
        accelerator.init_trackers(tracker_name, config=vars(args))

    # 开始训练！
    # 计算总批量大小
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    # 计算可训练参数的数量
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    # 记录训练开始的信息
    logger.info("***** Running training *****")
    # 记录可训练参数的数量
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    # 记录样本数量
    logger.info(f"  Num examples = {len(train_dataset)}")
    # 记录每个周期的批次数
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    # 记录训练周期数
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    # 记录每个设备的瞬时批量大小
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    # 记录总批量大小（包括并行、分布式和积累）
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    # 记录梯度积累步数
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    # 记录总优化步骤
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # 初始化全局步数
    global_step = 0
    # 初始化首个周期
    first_epoch = 0

    # 可能从之前的保存中加载权重和状态
    if not args.resume_from_checkpoint:
        # 如果未从检查点恢复，设置初始全局步数为0
        initial_global_step = 0
    else:  # 如果前面的条件不满足，执行以下代码
        if args.resume_from_checkpoint != "latest":  # 检查是否指定了非最新的检查点
            path = os.path.basename(args.resume_from_checkpoint)  # 获取指定检查点的基本文件名
        else:  # 如果没有指定非最新检查点
            # 获取最近的检查点
            dirs = os.listdir(args.output_dir)  # 列出输出目录中的所有文件和目录
            dirs = [d for d in dirs if d.startswith("checkpoint")]  # 过滤出以 "checkpoint" 开头的目录
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))  # 根据检查点的数字部分排序
            path = dirs[-1] if len(dirs) > 0 else None  # 如果有检查点，取最新的一个，否则为 None

        if path is None:  # 如果没有找到有效的检查点
            accelerator.print(  # 输出信息
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."  # 提示检查点不存在，开始新的训练
            )
            args.resume_from_checkpoint = None  # 将恢复检查点参数设置为 None
            initial_global_step = 0  # 初始化全局步骤为 0
        else:  # 如果找到了有效的检查点
            accelerator.print(f"Resuming from checkpoint {path}")  # 输出恢复检查点的信息
            accelerator.load_state(os.path.join(args.output_dir, path))  # 加载指定检查点的状态
            global_step = int(path.split("-")[1])  # 从检查点的文件名中提取全局步骤

            initial_global_step = global_step  # 将初始全局步骤设置为提取的值
            first_epoch = global_step // num_update_steps_per_epoch  # 计算当前是第几个 epoch

    progress_bar = tqdm(  # 创建一个进度条
        range(0, args.max_train_steps),  # 设置进度条的范围
        initial=initial_global_step,  # 设置进度条的初始值
        desc="Steps",  # 设置进度条的描述
        # 仅在每台机器上显示一次进度条。
        disable=not accelerator.is_local_main_process,  # 如果不是本地主进程，则禁用进度条
    )
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)  # 计算 VAE 的空间缩放因子

    # 用于 DeepSpeed 训练
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config  # 获取模型配置，考虑模块属性

    # 保存 LoRA 层
    accelerator.wait_for_everyone()  # 等待所有进程完成
    # 检查当前进程是否为主进程
        if accelerator.is_main_process:
            # 解包模型以获取主模型
            transformer = unwrap_model(transformer)
            # 根据混合精度设置选择数据类型
            dtype = (
                torch.float16
                if args.mixed_precision == "fp16"
                else torch.bfloat16
                if args.mixed_precision == "bf16"
                else torch.float32
            )
            # 将模型转换为所选的数据类型
            transformer = transformer.to(dtype)
            # 获取模型的 LoRA 层状态字典
            transformer_lora_layers = get_peft_model_state_dict(transformer)
    
            # 保存 LoRA 权重到指定目录
            CogVideoXPipeline.save_lora_weights(
                save_directory=args.output_dir,
                transformer_lora_layers=transformer_lora_layers,
            )
    
            # 最终测试推理
            pipe = CogVideoXPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )
            # 使用配置创建调度器
            pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
    
            # 如果启用切片功能，则启用 VAE 的切片
            if args.enable_slicing:
                pipe.vae.enable_slicing()
            # 如果启用平铺功能，则启用 VAE 的平铺
            if args.enable_tiling:
                pipe.vae.enable_tiling()
    
            # 加载 LoRA 权重
            lora_scaling = args.lora_alpha / args.rank
            # 从输出目录加载 LoRA 权重
            pipe.load_lora_weights(args.output_dir, adapter_name="cogvideox-lora")
            # 设置适配器及其缩放因子
            pipe.set_adapters(["cogvideox-lora"], [lora_scaling])
    
            # 运行推理并进行验证
            validation_outputs = []
            # 如果有验证提示且数量大于零，则进行验证
            if args.validation_prompt and args.num_validation_videos > 0:
                validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
                # 遍历每个验证提示
                for validation_prompt in validation_prompts:
                    # 准备推理参数
                    pipeline_args = {
                        "prompt": validation_prompt,
                        "guidance_scale": args.guidance_scale,
                        "use_dynamic_cfg": args.use_dynamic_cfg,
                        "height": args.height,
                        "width": args.width,
                    }
    
                    # 记录验证输出
                    video = log_validation(
                        pipe=pipe,
                        args=args,
                        accelerator=accelerator,
                        pipeline_args=pipeline_args,
                        epoch=epoch,
                        is_final_validation=True,
                    )
                    # 扩展验证输出列表
                    validation_outputs.extend(video)
    
            # 如果需要上传到中心
            if args.push_to_hub:
                # 保存模型卡信息到指定的库
                save_model_card(
                    repo_id,
                    videos=validation_outputs,
                    base_model=args.pretrained_model_name_or_path,
                    validation_prompt=args.validation_prompt,
                    repo_folder=args.output_dir,
                    fps=args.fps,
                )
                # 上传输出目录到指定的库
                upload_folder(
                    repo_id=repo_id,
                    folder_path=args.output_dir,
                    commit_message="End of training",
                    ignore_patterns=["step_*", "epoch_*"],
                )
    
        # 结束训练过程
        accelerator.end_training()
# 如果该脚本是主程序，则执行以下代码块
if __name__ == "__main__":
    # 获取命令行参数
    args = get_args()
    # 调用主函数，并将参数传递给它
    main(args)
```