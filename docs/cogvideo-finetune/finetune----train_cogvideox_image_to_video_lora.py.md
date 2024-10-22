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