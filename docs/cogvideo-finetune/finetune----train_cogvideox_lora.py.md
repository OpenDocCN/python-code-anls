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