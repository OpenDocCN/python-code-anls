# `.\transformers\training_args.py`

```
# 版权声明及许可信息
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入所需的库
import contextlib
import io
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# 从huggingface_hub导入获取完整仓库名称的函数
from huggingface_hub import get_full_repo_name
# 导入版本信息处理库
from packaging import version

# 导入调试工具函数
from .debug_utils import DebugOption
# 导入训练器工具函数
from .trainer_utils import (
    EvaluationStrategy,
    FSDPOption,
    HubStrategy,
    IntervalStrategy,
    SchedulerType,
)
# 导入工具函数
from .utils import (
    ACCELERATE_MIN_VERSION,
    ExplicitEnum,
    cached_property,
    is_accelerate_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_available,
    is_torch_bf16_cpu_available,
    is_torch_bf16_gpu_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_tf32_available,
    is_torch_tpu_available,
    is_torch_xpu_available,
    logging,
    requires_backends,
)
# 导入通用工具函数
from .utils.generic import strtobool
# 导入模型优化工具函数
from .utils.import_utils import is_optimum_neuron_available

# 获取日志记录器
logger = logging.get_logger(__name__)
# 复制日志级别字典，作为日志记录器的日志级别
log_levels = logging.get_log_levels_dict().copy()
# 更新训练器日志级别字典
trainer_log_levels = dict(**log_levels, passive=-1)

# 如果torch库可用
if is_torch_available():
    # 导入torch库
    import torch
    # 导入torch分布式库
    import torch.distributed as dist

# 如果加速库可用
if is_accelerate_available():
    # 从加速库导入状态和部分状态
    from accelerate.state import AcceleratorState, PartialState
    # 从加速库导入分布式类型
    from accelerate.utils import DistributedType

# 如果存在torch_tpu库可用（检查设备时不要求）
if is_torch_tpu_available(check_device=False):
    # 导入torch_xla.core.xla_model库
    import torch_xla.core.xla_model as xm

# 如果存在torch_neuroncore库可用（检查设备时不要求）
if is_torch_neuroncore_available(check_device=False):
    # 导入torchrun支持库
    # https://github.com/pytorch/xla/pull/3609
```  
    # 检查是否存在名为 "TORCHELASTIC_RUN_ID" 的环境变量
    if os.environ.get("TORCHELASTIC_RUN_ID"):
        # 如果可用的最佳神经元资源可用
        if is_optimum_neuron_available():
            # 输出信息提示，建议使用 optimum[neuron] 的 TrainiumTrainer 进行训练，否则会失败
            logger.info(
                "Make sure that you are performing the training with the TrainiumTrainer from optimum[neuron], this "
                "will fail otherwise."
            )
        else:
            # 输出警告信息，建议使用 optimum[neuron] 的 TrainiumTrainer 代替 Transformers 库在 AWS Trainium 实例上进行训练
            logger.warning(
                "Please use the TrainiumTrainer from optimum[neuron] instead of the Transformers library to perform "
                "training on AWS Trainium instances. More information here: "
                "https://github.com/huggingface/optimum-neuron"
            )
            # 导入 torch_xla.distributed.xla_backend 模块
            import torch_xla.distributed.xla_backend as xbn

            # 如果 dist.group.WORLD 不是 xbn.ProcessGroupXla 类型的实例
            if not isinstance(dist.group.WORLD, xbn.ProcessGroupXla):
                # 使用 XLA 后端初始化 torch.distributed 进程组
                dist.init_process_group(backend="xla")
                # 如果 dist.group.WORLD 仍然不是 xbn.ProcessGroupXla 类型的实例
                if not isinstance(dist.group.WORLD, xbn.ProcessGroupXla):
                    # 抛出异常，表示使用 XLA 后端初始化 torch.distributed 进程组失败
                    raise AssertionError("Failed to initialize torch.distributed process group using XLA backend.")
```  
# 如果 SageMaker 多模型并行训练被启用
if is_sagemaker_mp_enabled():
    # 导入 SageMaker 多模型并行训练 Torch 库
    import smdistributed.modelparallel.torch as smp
    # 初始化多模型并行训练
    smp.init()

# 默认日志目录函数，返回一个字符串路径
def default_logdir() -> str:
    """
    Same default as PyTorch
    """
    # 导入 socket 模块
    import socket
    # 导入 datetime 模块中的 datetime 函数
    from datetime import datetime

    # 获取当前时间的字符串表示，格式为月份和日期_小时-分钟-秒
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    # 返回默认日志目录，结合当前时间和主机名
    return os.path.join("runs", current_time + "_" + socket.gethostname())

# 从环境变量中获取整数值函数
def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    # 遍历环境变量键列表
    for e in env_keys:
        # 获取环境变量值并转换为整数，如果不存在则默认为 -1
        val = int(os.environ.get(e, -1))
        # 如果值大于等于 0，则返回该值
        if val >= 0:
            return val
    # 如果所有环境变量的值都小于 0，则返回默认值
    return default

# 获取 XLA 设备类型函数，参数为 torch 设备对象，返回字符串类型或 None
def get_xla_device_type(device: "torch.device") -> Optional[str]:
    """
    Returns the xla device type (CPU|GPU|TPU) or None if the device is a non-xla device.
    """
    # 如果当前可用的是 TPU
    if is_torch_tpu_available():
        # 返回 XLA 设备类型（CPU|GPU|TPU）或 None
        return xm.xla_real_devices([device])[0].split(":")[0]
    # 如果不是 XLA 设备，则返回 None
    return None

# 优化器名称枚举类，存储优化器的可接受字符串标识符
class OptimizerNames(ExplicitEnum):
    """
    Stores the acceptable string identifiers for optimizers.
    """

    ADAMW_HF = "adamw_hf"
    ADAMW_TORCH = "adamw_torch"
    ADAMW_TORCH_FUSED = "adamw_torch_fused"
    ADAMW_TORCH_XLA = "adamw_torch_xla"
    ADAMW_TORCH_NPU_FUSED = "adamw_torch_npu_fused"
    ADAMW_APEX_FUSED = "adamw_apex_fused"
    ADAFACTOR = "adafactor"
    ADAMW_ANYPRECISION = "adamw_anyprecision"
    SGD = "sgd"
    ADAGRAD = "adagrad"
    ADAMW_BNB = "adamw_bnb_8bit"
    ADAMW_8BIT = "adamw_8bit"  # just an alias for adamw_bnb_8bit
    LION_8BIT = "lion_8bit"
    LION = "lion_32bit"
    PAGED_ADAMW = "paged_adamw_32bit"
    PAGED_ADAMW_8BIT = "paged_adamw_8bit"
    PAGED_LION = "paged_lion_32bit"
    PAGED_LION_8BIT = "paged_lion_8bit"
    RMSPROP = "rmsprop"

# 训练参数数据类，用于指定训练相关的参数
# TODO: `TrainingArguments` users rely on it being fully mutable. In the future see if we can narrow this to a few keys: https://github.com/huggingface/transformers/pull/25903
@dataclass
class TrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use in our example scripts **which relate to the training loop
    itself**.

    Using [`HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    """

    # 框架类型，默认为 "pt"（PyTorch）
    framework = "pt"
    # 输出目录，用于存储模型预测和检查点
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    # 是否覆盖输出目录的内容，默认为 False
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    # 是否运行训练，默认为 False
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    # 是否在开发集上运行评估，默认为 False
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    # 是否运行测试集上的预测
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    
    # 评估策略
    evaluation_strategy: Union[IntervalStrategy, str] = field(
        default="no",
        metadata={"help": "The evaluation strategy to use."},
    )
    
    # 仅返回损失值
    prediction_loss_only: bool = field(
        default=False,
        metadata={"help": "When performing evaluation and predictions, only returns the loss."},
    )

    # 训练时每个 GPU/TPU/MPS/NPU core/CPU 的批量大小
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
    )
    
    # 评估时每个 GPU/TPU/MPS/NPU core/CPU 的批量大小
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation."}
    )

    # 训练时每个 GPU/TPU core/CPU 的批量大小（已弃用）
    per_gpu_train_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Deprecated, the use of `--per_device_train_batch_size` is preferred. "
                "Batch size per GPU/TPU core/CPU for training."
            )
        },
    )
    
    # 评估时每个 GPU/TPU core/CPU 的批量大小（已弃用）
    per_gpu_eval_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Deprecated, the use of `--per_device_eval_batch_size` is preferred. "
                "Batch size per GPU/TPU core/CPU for evaluation."
            )
        },
    )

    # 累积梯度更新步数
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    
    # 累积评估步数
    eval_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of predictions steps to accumulate before moving the tensors to the CPU."},
    )

    # 延迟评估的时间
    eval_delay: Optional[float] = field(
        default=0,
        metadata={
            "help": (
                "Number of epochs or steps to wait for before the first evaluation can be performed, depending on the"
                " evaluation_strategy."
            )
        },
    )

    # AdamW 的初始学习率
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    
    # AdamW 的权重衰减
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    
    # AdamW 优化器的 Beta1
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    
    # AdamW 优化器的 Beta2
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    
    # AdamW 优化器的 Epsilon
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    
    # 最大梯度范数
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    # 总训练轮数
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    
    # 最大训练步数
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    
    # 使用的调度器类型
    lr_scheduler_type: Union[SchedulerType, str] = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )
    lr_scheduler_kwargs: Optional[Dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Extra parameters for the lr_scheduler such as {'num_cycles': 1} for the cosine with hard restarts"
            )
        },
    )
    # 学习率调度器的额外参数，例如 {'num_cycles': 1} 用于余弦退火重启的参数

    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    # 线性预热，占总步数的比例

    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    # 线性预热的步数

    log_level: Optional[str] = field(
        default="passive",
        metadata={
            "help": (
                "Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug',"
                " 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and"
                " lets the application set the level. Defaults to 'passive'."
            ),
            "choices": trainer_log_levels.keys(),
        },
    )
    # 主节点上要使用的记录器日志级别

    log_level_replica: Optional[str] = field(
        default="warning",
        metadata={
            "help": "Logger log level to use on replica nodes. Same choices and defaults as ``log_level``",
            "choices": trainer_log_levels.keys(),
        },
    )
    # 副本节点上要使用的记录器日志级别，与 log_level 相同

    log_on_each_node: bool = field(
        default=True,
        metadata={
            "help": (
                "When doing a multinode distributed training, whether to log once per node or just once on the main"
                " node."
            )
        },
    )
    # 在多节点分布式训练时，是否每个节点都记录日志

    logging_dir: Optional[str] = field(default=None, metadata={"help": "Tensorboard log dir."})
    # Tensorboard 日志目录

    logging_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    # 使用的记录策略

    logging_first_step: bool = field(default=False, metadata={"help": "Log the first global_step"})
    # 记录第一个全局步骤

    logging_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    # 每 X 步记录一次日志

    logging_nan_inf_filter: bool = field(default=True, metadata={"help": "Filter nan and inf losses for logging."})
    # 过滤记录中的 nan 和 inf 损失

    save_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    # 使用的检查点保存策略

    save_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    # 每 X 步保存一次检查点
    # 保存检查点的总数限制，如果传入值，则限制检查点的总数。删除`output_dir`中较旧的检查点。当启用`load_best_model_at_end`时，根据`metric_for_best_model`始终保留“最佳”检查点，以及最近的检查点。例如，对于`save_total_limit=5`和`load_best_model_at_end=True`，最后四个检查点将始终与最佳模型一起保留。当`save_total_limit=1`和`load_best_model_at_end=True`时，可能保存两个检查点：最后一个和最佳一个（如果它们不同）。默认为无限制检查点
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in"
                " `output_dir`. When `load_best_model_at_end` is enabled, the 'best' checkpoint according to"
                " `metric_for_best_model` will always be retained in addition to the most recent ones. For example,"
                " for `save_total_limit=5` and `load_best_model_at_end=True`, the four last checkpoints will always be"
                " retained alongside the best model. When `save_total_limit=1` and `load_best_model_at_end=True`,"
                " it is possible that two checkpoints are saved: the last one and the best one (if they are different)."
                " Default is unlimited checkpoints"
            )
        },
    )
    
    # 使用safetensors保存和加载状态字典，而不是默认的torch.load和torch.save
    save_safetensors: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Use safetensors saving and loading for state dicts instead of default torch.load and torch.save."
        },
    )
    
    # 在进行多节点分布式训练时，是否在每个节点上保存模型和检查点，还是仅在主节点上保存
    save_on_each_node: bool = field(
        default=False,
        metadata={
            "help": (
                "When doing multi-node distributed training, whether to save models and checkpoints on each node, or"
                " only on the main one"
            )
        },
    )
    
    # 在检查点时，是否仅保存模型，还是同时保存优化器、调度器和随机数生成器状态。注意，当此选项为真时，您将无法从检查点恢复训练。这样可以通过不存储优化器、调度器和随机数生成器状态来节省存储空间。您只能使用设置为True的from_pretrained加载模型。
    save_only_model: bool = field(
        default=False,
        metadata={
            "help": (
                "When checkpointing, whether to only save the model, or also the optimizer, scheduler & rng state."
                "Note that when this is true, you won't be able to resume training from checkpoint."
                "This enables you to save storage by not storing the optimizer, scheduler & rng state."
                "You can only load the model using from_pretrained with this option set to True."
            )
        },
    )
    
    # 此参数已弃用。在🤗 Transformers的5.0版本中将被移除。
    no_cuda: bool = field(
        default=False,
        metadata={"help": "This argument is deprecated. It will be removed in version 5.0 of 🤗 Transformers."},
    )
    
    # 是否使用cpu。如果设置为False，将使用cuda/tpu/mps/npu设备（如果可用）。
    use_cpu: bool = field(
        default=False,
        metadata={
            "help": " Whether or not to use cpu. If set to False, we will use cuda/tpu/mps/npu device if available."
        },
    )
    
    # 此参数已弃用。将使用`mps`设备（如果可用），类似于`cuda`设备。在🤗 Transformers的5.0版本中将被移除。
    use_mps_device: bool = field(
        default=False,
        metadata={
            "help": "This argument is deprecated. `mps` device will be used if available similar to `cuda` device."
            " It will be removed in version 5.0 of 🤗 Transformers"
        },
    )
    
    # 在训练开始时设置的随机种子
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    
    # 用于数据采样器的随机种子
    data_seed: Optional[int] = field(default=None, metadata={"help": "Random seed to be used with data samplers."})
    
    # 是否在推断时使用PyTorch jit跟踪
    jit_mode_eval: bool = field(
        default=False, metadata={"help": "Whether or not to use PyTorch jit trace for inference"}
    )
    use_ipex: bool = field(  # 是否使用 Intel PyTorch 扩展，如果可用
        default=False,  # 默认为 False
        metadata={  # 元数据，提供帮助信息
            "help": (
                "Use Intel extension for PyTorch when it is available, installation:"  # 使用 Intel PyTorch 扩展，安装链接
                " 'https://github.com/intel/intel-extension-for-pytorch'"
            )
        },
    )
    bf16: bool = field(  # 是否使用 bf16（混合）精度代替 32 位精度
        default=False,  # 默认为 False
        metadata={  # 元数据，提供帮助信息
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"  # 是否使用 bf16（混合）精度代替 32 位精度，需要 Ampere 或更高版本的 NVIDIA
                " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."  # 或使用 CPU（use_cpu）或 Ascend NPU。这是一个实验性的 API，可能会发生变化
            )
        },
    )
    fp16: bool = field(  # 是否使用 fp16（混合）精度代替 32 位精度
        default=False,  # 默认为 False
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},  # 元数据，提供帮助信息
    )
    fp16_opt_level: str = field(  # fp16 优化级别
        default="O1",  # 默认为 O1
        metadata={  # 元数据，提供帮助信息
            "help": (
                "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. "  # fp16 的 Apex AMP 优化级别，可选值为 ['O0', 'O1', 'O2', 'O3']
                "See details at https://nvidia.github.io/apex/amp.html"  # 详情请参阅链接
            )
        },
    )
    half_precision_backend: str = field(  # 使用半精度的后端
        default="auto",  # 默认为 auto
        metadata={  # 元数据，提供帮助信息
            "help": "The backend to be used for half precision.",  # 用于半精度的后端
            "choices": ["auto", "apex", "cpu_amp"],  # 可选值为 auto、apex、cpu_amp
        },
    )
    bf16_full_eval: bool = field(  # 是否使用完整的 bf16 评估代替 32 位精度
        default=False,  # 默认为 False
        metadata={  # 元数据，提供帮助信息
            "help": (
                "Whether to use full bfloat16 evaluation instead of 32-bit. This is an experimental API and it may"  # 是否使用完整的 bf16 评估代替 32 位精度。这是一个实验性的 API，可能会发生变化
                " change."
            )
        },
    )
    fp16_full_eval: bool = field(  # 是否使用完整的 fp16 评估代替 32 位精度
        default=False,  # 默认为 False
        metadata={"help": "Whether to use full float16 evaluation instead of 32-bit"},  # 元数据，提供帮助信息
    )
    tf32: Optional[bool] = field(  # 是否启用 tf32 模式，仅在 Ampere 和更新的 GPU 架构上可用
        default=None,  # 默认为 None
        metadata={  # 元数据，提供帮助信息
            "help": (
                "Whether to enable tf32 mode, available in Ampere and newer GPU architectures. This is an experimental"  # 是否启用 tf32 模式，仅在 Ampere 和更新的 GPU 架构上可用。这是一个实验性的 API，可能会发生变化
                " API and it may change."
            )
        },
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})  # 用于分布式训练的本地排名
    ddp_backend: Optional[str] = field(  # 用于分布式训练的后端
        default=None,  # 默认为 None
        metadata={  # 元数据，提供帮助信息
            "help": "The backend to be used for distributed training",  # 用于分布式训练的后端
            "choices": ["nccl", "gloo", "mpi", "ccl", "hccl"],  # 可选值为 nccl、gloo、mpi、ccl、hccl
        },
    )
    tpu_num_cores: Optional[int] = field(  # TPU 的核心数
        default=None, metadata={"help": "TPU: Number of TPU cores (automatically passed by launcher script)"}  # TPU：TPU 核心数（由启动脚本自动传递）
    )
    tpu_metrics_debug: bool = field(  # TPU：是否打印调试指标
        default=False,  # 默认为 False
        metadata={  # 元数据，提供帮助信息
            "help": (
                "Deprecated, the use of `--debug tpu_metrics_debug` is preferred. TPU: Whether to print debug metrics"  # 不推荐使用，优先使用 `--debug tpu_metrics_debug`。TPU：是否打印调试指标
            )
        },
    )
    debug: Union[str, List[DebugOption]] = field(
        default="",
        metadata={
            "help": (
                "Whether or not to enable debug mode. Current options: "
                "`underflow_overflow` (Detect underflow and overflow in activations and weights), "
                "`tpu_metrics_debug` (print debug metrics on TPU)."
            )
        },
    )
    # 调试模式选项，可以是字符串或DebugOption列表，默认为空字符串
    # 可选项包括：`underflow_overflow`（检测激活和权重中的下溢和上溢），
    # `tpu_metrics_debug`（在TPU上打印调试指标）

    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    # 是否丢弃最后一个不完整的批次，如果不是批次大小的整数倍，则丢弃

    eval_steps: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    # 每隔X步运行一次评估，应为范围`[0,1)`内的整数或浮点数
    # 如果小于1，则将解释为总训练步数的比率

    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                " in the main process."
            )
        },
    )
    # 用于数据加载的子进程数（仅适用于PyTorch）。0表示数据将在主进程中加载

    past_index: int = field(
        default=-1,
        metadata={"help": "If >=0, uses the corresponding part of the output as the past state for next step."},
    )
    # 如果>=0，则使用输出的相应部分作为下一步的过去状态

    run_name: Optional[str] = field(
        default=None, metadata={"help": "An optional descriptor for the run. Notably used for wandb logging."}
    )
    # 运行的可选描述符。主要用于wandb日志记录

    disable_tqdm: Optional[bool] = field(
        default=None, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )
    # 是否禁用tqdm进度条

    remove_unused_columns: Optional[bool] = field(
        default=True, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    # 在使用nlp.Dataset时，是否删除模型不需要的列

    label_names: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )
    # 输入字典中对应标签的键列表

    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to load the best model found during training at the end of training. When this option"
                " is enabled, the best checkpoint will always be saved. See `save_total_limit` for more."
            )
        },
    )
    # 是否在训练结束时加载找到的最佳模型
    # 启用此选项时，将始终保存最佳检查点

    metric_for_best_model: Optional[str] = field(
        default=None, metadata={"help": "The metric to use to compare two different models."}
    )
    # 用于比较两个不同模型的指标

    greater_is_better: Optional[bool] = field(
        default=None, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )
    # `metric_for_best_model`是否应该最大化

    ignore_data_skip: bool = field(
        default=False,
        metadata={
            "help": (
                "When resuming training, whether or not to skip the first epochs and batches to get to the same"
                " training data."
            )
        },
    )
    # 恢复训练时，是否跳过第一个周期和批次以获取相同的训练数据
    # 定义一个可选的字段 fsdp，类型为 Optional[Union[List[FSDPOption], str]]，默认为空字符串
    fsdp: Optional[Union[List[FSDPOption], str]] = field(
        default="",
        metadata={
            "help": (
                "是否使用 PyTorch Fully Sharded Data Parallel (FSDP) 训练（仅在分布式训练中使用）。"
                "基本选项应为 `full_shard`、`shard_grad_op` 或 `no_shard`，您可以使用以下方式添加 CPU-offload 到 `full_shard` 或 `shard_grad_op`："
                " `full_shard offload` 或 `shard_grad_op offload`。您可以使用相同语法为 `full_shard` 或 `shard_grad_op` 添加自动包装："
                " `full_shard auto_wrap` 或 `shard_grad_op auto_wrap`。"
            ),
        },
    )
    
    # 定义一个整数字段 fsdp_min_num_params，默认值为 0，用于 FSDP 的默认自动包装的最小参数数量（仅当传递了 `fsdp` 字段时有效）
    fsdp_min_num_params: int = field(
        default=0,
        metadata={
            "help": (
                "此参数已弃用。FSDP 的默认自动包装的最小参数数量（仅当传递了 `fsdp` 字段时有效）。"
            )
        },
    )
    
    # 定义一个可选的字符串字段 fsdp_config，默认值为 None，用于指定 FSDP 的配置文件
    fsdp_config: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "用于 FSDP（PyTorch Fully Sharded Data Parallel）的配置。值可以是一个 fsdp json 配置文件（例如 `fsdp_config.json`）"
                " 或已加载的 json 文件作为 `dict`。"
            )
        },
    )
    
    # 定义一个可选的字符串字段 fsdp_transformer_layer_cls_to_wrap，默认值为 None，用于指定要包装的 Transformer 层类名（大小写敏感）
    fsdp_transformer_layer_cls_to_wrap: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "此参数已弃用。要包装的 Transformer 层类名（大小写敏感），例如 `BertLayer`、`GPTJBlock`、`T5Block` ......（仅当传递了 `fsdp` 标志时有效）。"
            )
        },
    )
    
    # 定义一个可选的字符串字段 deepspeed，默认值为 None，用于启用 deepspeed 并传递 deepspeed json 配置文件的路径
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "启用 deepspeed 并传递 deepspeed json 配置文件的路径（例如 `ds_config.json`）"
                " 或已加载的 json 文件作为 dict"
            )
        },
    )
    
    # 定义一个浮点数字段 label_smoothing_factor，默认值为 0.0，用于应用标签平滑度（零表示不进行标签平滑）
    label_smoothing_factor: float = field(
        default=0.0, metadata={"help": "应用的标签平滑度（零表示不进行标签平滑）。"}
    )
    
    # 默认优化器为 "adamw_torch"
    default_optim = "adamw_torch"
    
    # XXX: 当 pytorch==2.0.1 发布时启用 - 我们希望给它足够的时间来解决所有的 bug
    # if is_torch_available() and version.parse(version.parse(torch.__version__).base_version) >= version.parse("2.1.0"):
    #     default_optim = "adamw_torch_fused"
    # 并更新上面的文档为:
    # optim (`str` or [`training_args.OptimizerNames`], *optional*, 默认为 `"adamw_torch_fused"`（对于 torch<2.1.0 为 `"adamw_torch"`）:
    # 定义一个联合类型字段 optim，类型为 Union[OptimizerNames, str]，默认为 default_optim，用于指定要使用的优化器
    optim: Union[OptimizerNames, str] = field(
        default=default_optim,
        metadata={"help": "要使用的优化器。"},
    )
    # 定义一个可选的字符串参数，用于传递给优化器的可选参数
    optim_args: Optional[str] = field(default=None, metadata={"help": "Optional arguments to supply to optimizer."})
    # 是否使用 Adafactor 替代 AdamW
    adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    # 是否在批处理时将大致相同长度的样本分组在一起
    group_by_length: bool = field(
        default=False,
        metadata={"help": "Whether or not to group samples of roughly the same length together when batching."},
    )
    # 用于分组长度的预先计算长度的列名
    length_column_name: Optional[str] = field(
        default="length",
        metadata={"help": "Column name with precomputed lengths to use when grouping by length."},
    )
    # 报告结果和日志的集成列表
    report_to: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of integrations to report the results and logs to."}
    )
    # 在使用分布式训练时，传递给 DistributedDataParallel 的 find_unused_parameters 标志的值
    ddp_find_unused_parameters: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `find_unused_parameters` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    # 在使用分布式训练时，传递给 DistributedDataParallel 的 bucket_cap_mb 标志的值
    ddp_bucket_cap_mb: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `bucket_cap_mb` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    # 在使用分布式训练时，传递给 DistributedDataParallel 的 broadcast_buffers 标志的值
    ddp_broadcast_buffers: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `broadcast_buffers` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    # 是否为 DataLoader 固定内存
    dataloader_pin_memory: bool = field(
        default=True, metadata={"help": "Whether or not to pin memory for DataLoader."}
    )
    # 是否保持 DataLoader 的 worker 进程持久化
    dataloader_persistent_workers: bool = field(
        default=False,
        metadata={
            "help": "If True, the data loader will not shut down the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive. Can potentially speed up training, but will increase RAM usage."
        },
    )
    # 是否跳过将内存分析报告添加到指标中
    skip_memory_metrics: bool = field(
        default=True, metadata={"help": "Whether or not to skip adding of memory profiler reports to metrics."}
    )
    # 是否使用传统的 prediction_loop
    use_legacy_prediction_loop: bool = field(
        default=False, metadata={"help": "Whether or not to use the legacy prediction_loop in the Trainer."}
    )
    # 是否在训练后将训练好的模型上传到模型中心
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )
    # 从检查点恢复训练的路径
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )
    # 与本地 output_dir 同步的存储库名称
    hub_model_id: Optional[str] = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    # 定义一个变量 hub_strategy，类型为 Union[HubStrategy, str]，默认为"every_save"，当`--push_to_hub`被激活时使用的 Hub 策略
    hub_strategy: Union[HubStrategy, str] = field(
        default="every_save",
        metadata={"help": "The hub strategy to use when `--push_to_hub` is activated."},
    )
    # 定义一个变量 hub_token，类型为 Optional[str]，默认为 None，用于推送到 Model Hub 的令牌
    hub_token: Optional[str] = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})
    # 定义一个变量 hub_private_repo，类型为 bool，默认为 False，指示模型仓库是否为私有
    hub_private_repo: bool = field(default=False, metadata={"help": "Whether the model repository is private or not."})
    # 定义一个变量 hub_always_push，类型为 bool，默认为 False，除非为 True，否则 Trainer 将跳过推送，如果上一个推送尚未完成
    hub_always_push: bool = field(
        default=False,
        metadata={"help": "Unless `True`, the Trainer will skip pushes if the previous one wasn't finished yet."},
    )
    # 定义一个变量 gradient_checkpointing，类型为 bool，默认为 False，如果为 True，则使用梯度检查点以节省内存，但会减慢反向传播速度
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    # 定义一个变量 gradient_checkpointing_kwargs，类型为 Optional[dict]，默认为 None，梯度检查点关键字参数，如`use_reentrant`，将通过`model.gradient_checkpointing_enable`传递给`torch.utils.checkpoint.checkpoint`
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Gradient checkpointing key word arguments such as `use_reentrant`. Will be passed to `torch.utils.checkpoint.checkpoint` through `model.gradient_checkpointing_enable`."
        },
    )
    # 定义一个变量 include_inputs_for_metrics，类型为 bool，默认为 False，指示是否将输入传递给`compute_metrics`函数
    include_inputs_for_metrics: bool = field(
        default=False, metadata={"help": "Whether or not the inputs will be passed to the `compute_metrics` function."}
    )
    # Deprecated arguments
    # 定义一个变量 fp16_backend，类型为 str，默认为"auto"，已弃用，使用 half_precision_backend 替代
    fp16_backend: str = field(
        default="auto",
        metadata={
            "help": "Deprecated. Use half_precision_backend instead",
            "choices": ["auto", "apex", "cpu_amp"],
        },
    )
    # 定义一个变量 push_to_hub_model_id，类型为 Optional[str]，默认为 None，要推送到的`Trainer`的仓库名称
    push_to_hub_model_id: Optional[str] = field(
        default=None, metadata={"help": "The name of the repository to which push the `Trainer`."}
    )
    # 定义一个变量 push_to_hub_organization，类型为 Optional[str]，默认为 None，要推送到的组织名称
    push_to_hub_organization: Optional[str] = field(
        default=None, metadata={"help": "The name of the organization in with to which push the `Trainer`."}
    )
    # 定义一个变量 push_to_hub_token，类型为 Optional[str]，默认为 None，用于推送到 Model Hub 的令牌
    push_to_hub_token: Optional[str] = field(
        default=None, metadata={"help": "The token to use to push to the Model Hub."}
    )
    # 定义一个变量 _n_gpu，类型为 int，初始化为 -1，不可表示，用于表示 GPU 数量
    _n_gpu: int = field(init=False, repr=False, default=-1)
    # 定义一个变量 mp_parameters，类型为 str，默认为空字符串，由 SageMaker 启动器使用以发送 mp-specific 参数，在 Trainer 中被忽略
    mp_parameters: str = field(
        default="",
        metadata={"help": "Used by the SageMaker launcher to send mp-specific args. Ignored in Trainer"},
    )

    # 定义一个变量 auto_find_batch_size，类型为 bool，默认为 False，是否自动减少批量大小并重新运行训练循环，每次达到 CUDA 内存不足时
    auto_find_batch_size: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to automatically decrease the batch size in half and rerun the training loop again each time"
                " a CUDA Out-of-Memory was reached"
            )
        },
    )
    # 定义一个变量 full_determinism，类型为 bool，默认为 False，是否调用 enable_full_determinism 而不是 set_seed 以实现在分布式训练中的可重现性，重要：这会对性能产生负面影响，仅用于调试
    full_determinism: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to call enable_full_determinism instead of set_seed for reproducibility in distributed"
                " training. Important: this will negatively impact the performance, so only use it for debugging."
            )
        },
    )
    torchdynamo: Optional[str] = field(
        default=None,
        metadata={
            "help": "This argument is deprecated, use `--torch_compile_backend` instead.",
        },
    )

# `torchdynamo`是一个可选的字符串类型字段，用于设置Torch Dynamo的参数。默认值为None。
# 元数据(metadata)提供了帮助信息，指出该参数已弃用，应使用`--torch_compile_backend`代替。

    ray_scope: Optional[str] = field(
        default="last",
        metadata={
            "help": (
                'The scope to use when doing hyperparameter search with Ray. By default, `"last"` will be used. Ray'
                " will then use the last checkpoint of all trials, compare those, and select the best one. However,"
                " other options are also available. See the Ray documentation"
                " (https://docs.ray.io/en/latest/tune/api_docs/analysis.html"
                "#ray.tune.ExperimentAnalysis.get_best_trial)"
                " for more options."
            )
        },
    )

# `ray_scope`是一个可选的字符串类型字段，用于设置Ray的超参数搜索范围。
# 默认值为"last"，表示使用最后一个检查点进行超参数搜索。
# 元数据(metadata)提供了帮助信息，说明了不同选项的含义，并提供了链接到Ray文档的详细信息。

    ddp_timeout: Optional[int] = field(
        default=1800,
        metadata={
            "help": "Overrides the default timeout for distributed training (value should be given in seconds)."
        },
    )

# `ddp_timeout`是一个可选的整数类型字段，用于设置分布式训练的超时时间。
# 默认值为1800秒。
# 元数据(metadata)提供了帮助信息，指出了超时时间的单位为秒。

    torch_compile: bool = field(
        default=False, metadata={"help": "If set to `True`, the model will be wrapped in `torch.compile`."}
    )

# `torch_compile`是一个布尔类型字段，用于设置是否使用`torch.compile`对模型进行封装。
# 默认值为False。
# 元数据(metadata)提供了帮助信息，说明了当设置为`True`时的行为。

    torch_compile_backend: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which backend to use with `torch.compile`, passing one will trigger a model compilation.",
        },
    )

# `torch_compile_backend`是一个可选的字符串类型字段，用于设置`torch.compile`使用的后端。
# 默认值为None。
# 元数据(metadata)提供了帮助信息，说明了如何触发模型编译以及可能的后端选项。

    torch_compile_mode: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which mode to use with `torch.compile`, passing one will trigger a model compilation.",
        },
    )

# `torch_compile_mode`是一个可选的字符串类型字段，用于设置`torch.compile`的模式。
# 默认值为None。
# 元数据(metadata)提供了帮助信息，说明了如何触发模型编译以及可能的模式选项。

    dispatch_batches: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to dispatch batches across devices in distributed training. If set to `True`, the dataloader prepared by the Accelerator is only iterated through on the main process "
            "and then the batches are split and broadcast to each process. Will default to `True` for `DataLoader` whose"
            "underlying dataset is an `IterableDataset`, `False` otherwise."
        },
    )

# `dispatch_batches`是一个可选的布尔类型字段，用于设置在分布式训练中是否跨设备分发批次。
# 如果设置为`True`，则由加速器准备的数据加载器只在主进程上进行迭代，
# 然后将批次拆分并广播到每个进程。
# 对于底层数据集为`IterableDataset`的`DataLoader`，默认值为`True`，否则为`False`。
# 元数据(metadata)提供了帮助信息，说明了该选项的含义。

    split_batches: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether or not the accelerator should split the batches yielded by the dataloaders across the devices during distributed training. If"
            "set to `True`, the actual batch size used will be the same on any kind of distributed processes, but it must be a"
            "round multiple of the number of processes you are using (such as GPUs)."
        },
    )

# `split_batches`是一个可选的布尔类型字段，用于设置加速器在分布式训练期间是否应该跨设备拆分数据加载器产生的批次。
# 如果设置为`True`，则在任何类型的分布式进程上使用的实际批次大小将相同，
# 但它必须是您使用的进程数量的圆整倍数（例如GPU数量）。
# 元数据(metadata)提供了帮助信息，说明了该选项的含义。

    include_tokens_per_second: Optional[bool] = field(
        default=False,
        metadata={"help": "If set to `True`, the speed metrics will include `tgs` (tokens per second per device)."},
    )

# `include_tokens_per_second`是一个可选的布尔类型字段，用于设置是否在速度指标中包含`tgs`（每个设备的每秒标记数）。
# 默认值为False。
# 元数据(metadata)提供了帮助信息，说明了该选项的含义。
    # 定义一个可选的布尔类型变量，用于指示是否包含观察到的输入标记数。默认为 False。
    include_num_input_tokens_seen: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If set to `True`, will track the number of input tokens seen throughout training. (May be slower in distributed training)"
        },
    )

    # 定义一个浮点型变量，用于激活 NEFTune 噪声嵌入到模型中。NEFTune 已被证明可以显著提高指令微调的模型性能。
    # 只支持 `PreTrainedModel` 和 `PeftModel` 类。
    # 请参阅原始论文：https://arxiv.org/abs/2310.05914，原始代码：https://github.com/neelsjain/NEFTune。
    neftune_noise_alpha: float = field(
        default=None,
        metadata={
            "help": "Activates neftune noise embeddings into the model. NEFTune has been proven to drastically improve model performances for instrcution fine-tuning. Check out the original paper here: https://arxiv.org/abs/2310.05914 and the original code here: https://github.com/neelsjain/NEFTune. Only supported for `PreTrainedModel` and `PeftModel` classes."
        },
    )

    # 定义 __str__ 方法，返回该对象的字符串表示形式
    def __str__(self):
        # 将对象转换为字典形式
        self_as_dict = asdict(self)

        # 移除不推荐使用的参数。一旦这些不推荐使用的参数从 TrainingArguments 中移除，这段代码就应该被移除。（TODO: v5）
        del self_as_dict["per_gpu_train_batch_size"]
        del self_as_dict["per_gpu_eval_batch_size"]

        # 将所有 token 结尾的键的值改为大写形式
        self_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in self_as_dict.items()}

        # 将字典键值对转换为字符串列表
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        # 返回类名和属性的字符串表示形式
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    # 将 __repr__ 方法指向 __str__ 方法
    __repr__ = __str__

    # 定义 train_batch_size 属性，返回实际的训练批量大小（在分布式训练中可能与 per_gpu_train_batch_size 不同）
    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training (may differ from `per_gpu_train_batch_size` in distributed training).
        """
        if self.per_gpu_train_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_train_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_train_batch_size` is preferred."
            )
        # 获取每个设备的批量大小
        per_device_batch_size = self.per_gpu_train_batch_size or self.per_device_train_batch_size
        # 计算实际的训练批量大小
        train_batch_size = per_device_batch_size * max(1, self.n_gpu)
        return train_batch_size

    # 定义 eval_batch_size 属性，返回实际的评估批量大小（在分布式训练中可能与 per_gpu_eval_batch_size 不同）
    @property
    def eval_batch_size(self) -> int:
        """
        The actual batch size for evaluation (may differ from `per_gpu_eval_batch_size` in distributed training).
        """
        if self.per_gpu_eval_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_eval_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_eval_batch_size` is preferred."
            )
        # 获取每个设备的批量大小
        per_device_batch_size = self.per_gpu_eval_batch_size or self.per_device_eval_batch_size
        # 计算实际的评估批量大小
        eval_batch_size = per_device_batch_size * max(1, self.n_gpu)
        return eval_batch_size

    # 定义 ddp_timeout_delta 属性，返回 torch.distributed.init_process_group 的实际超时时间，因为它期望一个 timedelta 变量。
    @property
    def ddp_timeout_delta(self) -> timedelta:
        """
        The actual timeout for torch.distributed.init_process_group since it expects a timedelta variable.
        """
        return timedelta(seconds=self.ddp_timeout)

    # 定义 cached_property 属性
    @cached_property
    @property
    def device(self) -> "torch.device":
        """
        返回当前进程使用的设备。

        Returns:
            torch.device: 当前进程使用的设备。
        """
        # 确保需要的后端已经加载
        requires_backends(self, ["torch"])
        # 返回设备设置
        return self._setup_devices

    @property
    def n_gpu(self):
        """
        本进程使用的 GPU 数量。

        注意:
            当有多个 GPU 可用但不使用分布式训练时，此值将大于 1。
            对于分布式训练，它将始终为 1。
        """
        # 确保需要的后端已经加载
        requires_backends(self, ["torch"])
        # 确保 `_n_gpu` 属性正确设置
        if not hasattr(self, "_n_gpu"):
            _ = self._setup_devices
        # 返回 GPU 数量
        return self._n_gpu

    @property
    def parallel_mode(self):
        """
        如果有多个 GPU/TPU 核心可用，则返回当前使用的并行模式之一。

        - `ParallelMode.NOT_PARALLEL`: 无并行（CPU 或一个 GPU）。
        - `ParallelMode.NOT_DISTRIBUTED`: 单个进程中有多个 GPU（使用 `torch.nn.DataParallel`）。
        - `ParallelMode.DISTRIBUTED`: 多个 GPU，每个 GPU 有自己的进程（使用 `torch.nn.DistributedDataParallel`）。
        - `ParallelMode.TPU`: 多个 TPU 核心。
        """
        # 确保需要的后端已经加载
        requires_backends(self, ["torch"])
        if is_torch_tpu_available():
            return ParallelMode.TPU
        elif is_sagemaker_mp_enabled():
            return ParallelMode.SAGEMAKER_MODEL_PARALLEL
        elif is_sagemaker_dp_enabled():
            return ParallelMode.SAGEMAKER_DATA_PARALLEL
        elif (
            self.distributed_state is not None and self.distributed_state.distributed_type != DistributedType.NO
        ) or (self.distributed_state is None and self.local_rank != -1):
            return ParallelMode.DISTRIBUTED
        elif self.n_gpu > 1:
            return ParallelMode.NOT_DISTRIBUTED
        else:
            return ParallelMode.NOT_PARALLEL

    @property
    def world_size(self):
        """
        并行使用的进程数。
        """
        # 确保需要的后端已经加载
        requires_backends(self, ["torch"])
        if self.distributed_state is not None:
            return self.distributed_state.num_processes
        elif is_sagemaker_mp_enabled():
            return smp.dp_size() if not smp.state.cfg.prescaled_batch else smp.rdp_size()
        return 1

    @property
    def process_index(self):
        """
        当前使用的进程索引。
        """
        # 确保需要的后端已经加载
        requires_backends(self, ["torch"])
        if self.distributed_state is not None:
            return self.distributed_state.process_index
        elif is_sagemaker_mp_enabled():
            return smp.dp_rank() if not smp.state.cfg.prescaled_batch else smp.rdp_rank()
        return 0

    @property
    def local_process_index(self):
        """
        The index of the local process used.
        """
        # 检查是否需要后端支持
        requires_backends(self, ["torch"])

        # 如果存在分布式状态，则返回本地进程索引
        if self.distributed_state is not None:
            return self.distributed_state.local_process_index
        # 如果启用了 SageMaker 多进程，则返回本地进程索引
        elif is_sagemaker_mp_enabled():
            return smp.local_rank()
        # 默认返回索引 0
        return 0

    @property
    def should_log(self):
        """
        Whether or not the current process should produce log.
        """
        # 如果设置了在每个节点上记录日志，则返回当前进程是否为索引 0
        if self.log_on_each_node:
            return self.local_process_index == 0
        else:
            # 如果启用了 SageMaker 多进程，则返回当前进程是否为索引 0
            if is_sagemaker_mp_enabled():
                return smp.rank() == 0
            else:
                return self.process_index == 0

    @property
    def should_save(self):
        """
        Whether or not the current process should write to disk, e.g., to save models and checkpoints.
        """
        # 如果设置了在每个节点上保存模型，则返回当前进程是否为索引 0
        if self.save_on_each_node:
            return self.local_process_index == 0
        else:
            # 如果启用了 SageMaker 多进程，则返回当前进程是否为索引 0
            if is_sagemaker_mp_enabled():
                return smp.rank() == 0
            else:
                return self.process_index == 0

    def get_process_log_level(self):
        """
        Returns the log level to be used depending on whether this process is the main process of node 0, main process
        of node non-0, or a non-main process.

        For the main process the log level defaults to the logging level set (`logging.WARNING` if you didn't do
        anything) unless overridden by `log_level` argument.

        For the replica processes the log level defaults to `logging.WARNING` unless overridden by `log_level_replica`
        argument.

        The choice between the main and replica process settings is made according to the return value of `should_log`.
        """
        # 将日志级别转换为整数
        log_level = trainer_log_levels[self.log_level]
        log_level_replica = trainer_log_levels[self.log_level_replica]

        # 获取主节点和副本节点的日志级别
        log_level_main_node = logging.get_verbosity() if log_level == -1 else log_level
        log_level_replica_node = logging.get_verbosity() if log_level_replica == -1 else log_level_replica
        # 根据是否应记录日志返回相应的日志级别
        return log_level_main_node if self.should_log else log_level_replica_node

    @property
    def place_model_on_device(self):
        """
        Can be subclassed and overridden for some specific integrations.
        """
        # 如果未启用 SageMaker 多进程，则返回 True
        return not is_sagemaker_mp_enabled()

    @property
    def _no_sync_in_gradient_accumulation(self):
        """
        Whether or not to use no_sync for the gradients when doing gradient accumulation.
        """
        # 如果未使用 DeepSpeed、SageMaker DP、SageMaker MP 或 Torch NeuronCore，则返回 True
        return not (
            self.deepspeed or is_sagemaker_dp_enabled() or is_sagemaker_mp_enabled() or is_torch_neuroncore_available()
        )

    @contextlib.contextmanager
    def main_process_first(self, local=True, desc="work"):
        """
        A context manager for torch distributed environment where one needs to do something on the main process, while
        blocking replicas, and when it's finished releasing the replicas.

        One such use is for `datasets`'s `map` feature which to be efficient should be run once on the main process,
        which upon completion saves a cached version of results and which then automatically gets loaded by the
        replicas.

        Args:
            local (`bool`, *optional*, defaults to `True`):
                if `True` first means process of rank 0 of each node if `False` first means process of rank 0 of node
                rank 0 In multi-node environment with a shared filesystem you most likely will want to use
                `local=False` so that only the main process of the first node will do the processing. If however, the
                filesystem is not shared, then the main process of each node will need to do the processing, which is
                the default behavior.
            desc (`str`, *optional*, defaults to `"work"`):
                a work description to be used in debug logs

        """
        # Check if torch is available and the world size is greater than 1 (i.e., distributed environment)
        if is_torch_available() and self.world_size > 1:
            # Define the description for the main process based on the value of 'local'
            main_process_desc = "main local process" if local else "main process"
            # Check if distributed state is available
            if self.distributed_state is not None:
                # Determine if the current process is the main process based on 'local' value
                is_main_process = (
                    self.distributed_state.is_local_main_process if local else self.distributed_state.is_main_process
                )
            # Check if SageMaker multi-processing is enabled
            elif is_sagemaker_mp_enabled():
                is_main_process = smp.rank() == 0

            try:
                # If the current process is not the main process, wait for the main process to finish its task
                if not is_main_process:
                    # Tell all replicas to wait
                    logger.debug(f"{self.process_index}: waiting for the {main_process_desc} to perform {desc}")

                    # If running on TPU, synchronize all processes
                    if is_torch_tpu_available():
                        xm.rendezvous(desc)
                    else:
                        dist.barrier()
                # Yield control to the block of code where this context manager is used
                yield
            finally:
                # If the current process is the main process, signal that it has completed its task
                if is_main_process:
                    # The wait is over
                    logger.debug(f"{self.process_index}: {main_process_desc} completed {desc}, releasing all replicas")
                    # If running on TPU, synchronize all processes
                    if is_torch_tpu_available():
                        xm.rendezvous(desc)
                    else:
                        dist.barrier()
        else:
            # If torch is not available or world size <= 1, yield control without synchronization
            yield

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        # Calculate the number of warmup steps based on provided warmup steps or warmup ratio
        warmup_steps = (
            self.warmup_steps if self.warmup_steps > 0 else math.ceil(num_training_steps * self.warmup_ratio)
        )
        # Return the calculated warmup steps
        return warmup_steps
    # 将实例序列化为字典，替换`Enum`为其值（用于 JSON 序列化支持），并移除令牌值以混淆
    def to_dict(self):
        # 过滤掉定义为 field(init=False) 的字段
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        for k, v in d.items():
            # 如果值是 Enum 类型，则替换为其值
            if isinstance(v, Enum):
                d[k] = v.value
            # 如果值是列表且第一个元素是 Enum 类型，则替换为其值列表
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            # 如果键以 "_token" 结尾，则替换为特定格式的字符串
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

    # 将实例序列化为 JSON 字符串
    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2)

    # 用于 TensorBoard 的 hparams 的序列化，返回经过处理的字典
    def to_sanitized_dict(self) -> Dict[str, Any]:
        d = self.to_dict()
        # 添加额外的字段到字典中
        d = {**d, **{"train_batch_size": self.train_batch_size, "eval_batch_size": self.eval_batch_size}}

        valid_types = [bool, int, float, str]
        # 如果有 torch 库可用，则添加 torch.Tensor 类型到有效类型列表中
        if is_torch_available():
            valid_types.append(torch.Tensor)

        # 根据值的类型进行处理，如果不在有效类型列表中，则转换为字符串
        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}

    # 下面的方法用于简化 `TrainingArguments` 的实例化
    def set_training(
        self,
        learning_rate: float = 5e-5,
        batch_size: int = 8,
        weight_decay: float = 0,
        num_epochs: float = 3,
        max_steps: int = -1,
        gradient_accumulation_steps: int = 1,
        seed: int = 42,
        gradient_checkpointing: bool = False,
        """
        A method that regroups all basic arguments linked to the training.

        <Tip>

        Calling this method will automatically set `self.do_train` to `True`.

        </Tip>

        Args:
            learning_rate (`float`, *optional*, defaults to 5e-5):
                The initial learning rate for the optimizer.
            batch_size (`int` *optional*, defaults to 8):
                The batch size per device (GPU/TPU core/CPU...) used for training.
            weight_decay (`float`, *optional*, defaults to 0):
                The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in the
                optimizer.
            num_train_epochs(`float`, *optional*, defaults to 3.0):
                Total number of training epochs to perform (if not an integer, will perform the decimal part percents
                of the last epoch before stopping training).
            max_steps (`int`, *optional*, defaults to -1):
                If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`.
                For a finite dataset, training is reiterated through the dataset (if all data is exhausted) until
                `max_steps` is reached.
            gradient_accumulation_steps (`int`, *optional*, defaults to 1):
                Number of updates steps to accumulate the gradients for, before performing a backward/update pass.

                <Tip warning={true}>

                When using gradient accumulation, one step is counted as one step with backward pass. Therefore,
                logging, evaluation, save will be conducted every `gradient_accumulation_steps * xxx_step` training
                examples.

                </Tip>

            seed (`int`, *optional*, defaults to 42):
                Random seed that will be set at the beginning of training. To ensure reproducibility across runs, use
                the [`~Trainer.model_init`] function to instantiate the model if it has some randomly initialized
                parameters.
            gradient_checkpointing (`bool`, *optional*, defaults to `False`):
                If True, use gradient checkpointing to save memory at the expense of slower backward pass.

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_training(learning_rate=1e-4, batch_size=32)
        >>> args.learning_rate
        1e-4
        ```
        """
        # 设置 self.do_train 为 True
        self.do_train = True
        # 设置学习率
        self.learning_rate = learning_rate
        # 设置每个设备的���练批次大小
        self.per_device_train_batch_size = batch_size
        # 设置权重衰减
        self.weight_decay = weight_decay
        # 设置训练的总轮数
        self.num_train_epochs = num_epochs
        # 设置最大训练步数
        self.max_steps = max_steps
        # 设置梯度累积步数
        self.gradient_accumulation_steps = gradient_accumulation_steps
        # 设置随机种子
        self.seed = seed
        # 设置是否使用梯度检查点
        self.gradient_checkpointing = gradient_checkpointing
        # 返回 self
        return self
    def set_evaluate(
        self,
        strategy: Union[str, IntervalStrategy] = "no",  # 定义评估策略，默认为"no"
        steps: int = 500,  # 每次评估之间的更新步数，默认为500步
        batch_size: int = 8,  # 用于评估的每个设备（GPU/TPU核心/CPU等）的批量大小，默认为8
        accumulation_steps: Optional[int] = None,  # 在将输出张量移动到 CPU 之前累积输出张量的预测步数。如果未设置，则在将整个预测累积在 GPU/TPU 上之后将其移动到 CPU（速度更快但需要更多内存）。
        delay: Optional[float] = None,  # 在第一次评估之前需要等待的周期或步数，取决于评估策略。
        loss_only: bool = False,  # 仅考虑损失，忽略除损失之外的所有输出。
        jit_mode: bool = False,  # 是否使用 PyTorch jit 跟踪进行推理。
    ):
        """
        A method that regroups all arguments linked to evaluation.

        Args:
            strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"no"`):
                The evaluation strategy to adopt during training. Possible values are:

                    - `"no"`: No evaluation is done during training.
                    - `"steps"`: Evaluation is done (and logged) every `steps`.
                    - `"epoch"`: Evaluation is done at the end of each epoch.

                Setting a `strategy` different from `"no"` will set `self.do_eval` to `True`.
            steps (`int`, *optional*, defaults to 500):
                Number of update steps between two evaluations if `strategy="steps"`.
            batch_size (`int` *optional*, defaults to 8):
                The batch size per device (GPU/TPU core/CPU...) used for evaluation.
            accumulation_steps (`int`, *optional*):
                Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU.
                If left unset, the whole predictions are accumulated on GPU/TPU before being moved to the CPU (faster
                but requires more memory).
            delay (`float`, *optional*):
                Number of epochs or steps to wait for before the first evaluation can be performed, depending on the
                evaluation_strategy.
            loss_only (`bool`, *optional*, defaults to `False`):
                Ignores all outputs except the loss.
            jit_mode (`bool`, *optional*):
                Whether or not to use PyTorch jit trace for inference.

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_evaluate(strategy="steps", steps=100)
        >>> args.eval_steps
        100
        ```
        """
        self.evaluation_strategy = IntervalStrategy(strategy)  # 设置评估策略
        if self.evaluation_strategy == IntervalStrategy.STEPS and steps == 0:  # 如果评估策略为步数且步数为0，则抛出值错误
            raise ValueError("Setting `strategy` as 'steps' requires a positive value for `steps`.")
        self.do_eval = self.evaluation_strategy != IntervalStrategy.NO  # 设置是否执行评估
        self.eval_steps = steps  # 设置评估步数
        self.per_device_eval_batch_size = batch_size  # 设置每个设备的评估批量大小
        self.eval_accumulation_steps = accumulation_steps  # 设置累积步数
        self.eval_delay = delay  # 设置延迟
        self.prediction_loss_only = loss_only  # 设置是否仅损失
        self.jit_mode_eval = jit_mode  # 设置 JIT 模式是否启用
        return self

    def set_testing(
        self,
        batch_size: int = 8,  # 用于测试的每个设备的批量大小，默认为8
        loss_only: bool = False,  # 是否仅考虑损失，默认为 False
        jit_mode: bool = False,  # 是否使用 PyTorch jit 跟踪进行推理，默认为 False
```  
        """
        一个方法，重新组织所有与在保留数据集上进行测试相关的基本参数。

        <提示>

        调用此方法将自动将 `self.do_predict` 设置为 `True`。

        </提示>

        Args:
            batch_size (`int` *可选*, 默认为 8):
                用于测试的每个设备（GPU/TPU 核心/CPU...）的批量大小。
            loss_only (`bool`, *可选*, 默认为 `False`):
                仅忽略损失以外的所有输出。
            jit_mode (`bool`, *可选*):
                是否使用 PyTorch jit trace 进行推断。

        示例:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_testing(batch_size=32)
        >>> args.per_device_eval_batch_size
        32
        ```
        """
        # 设置 self.do_predict 为 True
        self.do_predict = True
        # 设置每个设备的评估批量大小
        self.per_device_eval_batch_size = batch_size
        # 设置是否仅预测损失
        self.prediction_loss_only = loss_only
        # 设置是否使用 jit 模式进行评估
        self.jit_mode_eval = jit_mode
        # 返回 self
        return self

    def set_save(
        self,
        strategy: Union[str, IntervalStrategy] = "steps",
        steps: int = 500,
        total_limit: Optional[int] = None,
        on_each_node: bool = False,
    ):
        """
        A method that regroups all arguments linked to checkpoint saving.

        Args:
            strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
                The checkpoint save strategy to adopt during training. Possible values are:

                    - `"no"`: No save is done during training.
                    - `"epoch"`: Save is done at the end of each epoch.
                    - `"steps"`: Save is done every `save_steps`.

            steps (`int`, *optional*, defaults to 500):
                Number of updates steps before two checkpoint saves if `strategy="steps"`.
            total_limit (`int`, *optional*):
                If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
                `output_dir`.
            on_each_node (`bool`, *optional*, defaults to `False`):
                When doing multi-node distributed training, whether to save models and checkpoints on each node, or
                only on the main one.

                This should not be activated when the different nodes use the same storage as the files will be saved
                with the same names for each node.

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_save(strategy="steps", steps=100)
        >>> args.save_steps
        100
        ```
        """
        # 设置保存策略
        self.save_strategy = IntervalStrategy(strategy)
        # 若保存策略为步数间隔且步数为0，则引发值错误
        if self.save_strategy == IntervalStrategy.STEPS and steps == 0:
            raise ValueError("Setting `strategy` as 'steps' requires a positive value for `steps`.")
        # 设置保存步数
        self.save_steps = steps
        # 设置保存总数限制
        self.save_total_limit = total_limit
        # 设置是否在每个节点保存
        self.save_on_each_node = on_each_node
        # 返回设置后的参数对象
        return self

    def set_logging(
        self,
        strategy: Union[str, IntervalStrategy] = "steps",
        steps: int = 500,
        report_to: Union[str, List[str]] = "none",
        level: str = "passive",
        first_step: bool = False,
        nan_inf_filter: bool = False,
        on_each_node: bool = False,
        replica_level: str = "passive",
    def set_push_to_hub(
        self,
        model_id: str,
        strategy: Union[str, HubStrategy] = "every_save",
        token: Optional[str] = None,
        private_repo: bool = False,
        always_push: bool = False,
    def set_optimizer(
        self,
        name: Union[str, OptimizerNames] = "adamw_torch",
        learning_rate: float = 5e-5,
        weight_decay: float = 0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        args: Optional[str] = None,
        """
        一个方法，重新组织所有与优化器及其超参数相关的参数。

        Args:
            name (`str` or [`training_args.OptimizerNames`], *optional*, defaults to `"adamw_torch"`):
                要使用的优化器："adamw_hf"、"adamw_torch"、"adamw_torch_fused"、"adamw_apex_fused"、
                "adamw_anyprecision"或"adafactor"。
            learning_rate (`float`, *optional*, defaults to 5e-5):
                初始学习率。
            weight_decay (`float`, *optional*, defaults to 0):
                应用的权重衰减（如果不为零）到除所有偏置和LayerNorm权重之外的所有层。
            beta1 (`float`, *optional*, defaults to 0.9):
                adam优化器或其变体的beta1超参数。
            beta2 (`float`, *optional*, defaults to 0.999):
                adam优化器或其变体的beta2超参数。
            epsilon (`float`, *optional*, defaults to 1e-8):
                adam优化器或其变体的epsilon超参数。
            args (`str`, *optional*):
                提供给AnyPrecisionAdamW的可选参数（仅在`optim="adamw_anyprecision"`时有用）。

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_optimizer(name="adamw_torch", beta1=0.8)
        >>> args.optim
        'adamw_torch'
        ```
        """
        # 设置优化器名称
        self.optim = OptimizerNames(name)
        # 设置学习率
        self.learning_rate = learning_rate
        # 设置权重衰减
        self.weight_decay = weight_decay
        # 设置adam优化器的beta1超参数
        self.adam_beta1 = beta1
        # 设置adam优化器的beta2超参数
        self.adam_beta2 = beta2
        # 设置adam优化器的epsilon超参数
        self.adam_epsilon = epsilon
        # 设置优化器参数
        self.optim_args = args
        # 返回当前实例
        return self

    def set_lr_scheduler(
        self,
        name: Union[str, SchedulerType] = "linear",
        num_epochs: float = 3.0,
        max_steps: int = -1,
        warmup_ratio: float = 0,
        warmup_steps: int = 0,
        """
        将所有与学习率调度器及其超参数相关的参数重新组合的方法。

        Args:
            name (`str` 或 [`SchedulerType`]，*可选*，默认为 `"linear"`):
                要使用的调度器类型。参见 [`SchedulerType`] 文档以获取所有可能的值。
            num_epochs (`float`，*可选*，默认为 3.0):
                要执行的总训练轮数（如果不是整数，将在停止训练之前执行最后一个周期的小数部分百分比）。
            max_steps (`int`，*可选*，默认为 -1):
                如果设置为正数，则要执行的总训练步数。覆盖 `num_train_epochs`。对于有限数据集，
                训练将在数据集上重复进行（如果所有数据都已耗尽），直到达到 `max_steps`。
            warmup_ratio (`float`，*可选*，默认为 0.0):
                用于从 0 线性预热到 `learning_rate` 的总训练步骤的比率。
            warmup_steps (`int`，*可选*，默认为 0):
                用于从 0 线性预热到 `learning_rate` 的步骤数。覆盖任何 `warmup_ratio` 的效果。

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_lr_scheduler(name="cosine", warmup_ratio=0.05)
        >>> args.warmup_ratio
        0.05
        ```
        """
        # 设置学习率调度器类型
        self.lr_scheduler_type = SchedulerType(name)
        # 设置总训练轮数
        self.num_train_epochs = num_epochs
        # 设置总训练步数
        self.max_steps = max_steps
        # 设置预热比率
        self.warmup_ratio = warmup_ratio
        # 设置预热步数
        self.warmup_steps = warmup_steps
        # 返回修改后的参数对象
        return self

    def set_dataloader(
        self,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        drop_last: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        auto_find_batch_size: bool = False,
        ignore_data_skip: bool = False,
        sampler_seed: Optional[int] = None,
``` 
    ):
        """
        A method that regroups all arguments linked to the dataloaders creation.

        Args:
            drop_last (`bool`, *optional*, defaults to `False`):
                Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch
                size) or not.
            num_workers (`int`, *optional*, defaults to 0):
                Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in
                the main process.
            pin_memory (`bool`, *optional*, defaults to `True`):
                Whether you want to pin memory in data loaders or not. Will default to `True`.
            persistent_workers (`bool`, *optional*, defaults to `False`):
                If True, the data loader will not shut down the worker processes after a dataset has been consumed
                once. This allows to maintain the workers Dataset instances alive. Can potentially speed up training,
                but will increase RAM usage. Will default to `False`.
            auto_find_batch_size (`bool`, *optional*, defaults to `False`)
                Whether to find a batch size that will fit into memory automatically through exponential decay,
                avoiding CUDA Out-of-Memory errors. Requires accelerate to be installed (`pip install accelerate`)
            ignore_data_skip (`bool`, *optional*, defaults to `False`):
                When resuming training, whether or not to skip the epochs and batches to get the data loading at the
                same stage as in the previous training. If set to `True`, the training will begin faster (as that
                skipping step can take a long time) but will not yield the same results as the interrupted training
                would have.
            sampler_seed (`int`, *optional*):
                Random seed to be used with data samplers. If not set, random generators for data sampling will use the
                same seed as `self.seed`. This can be used to ensure reproducibility of data sampling, independent of
                the model seed.

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_dataloader(train_batch_size=16, eval_batch_size=64)
        >>> args.per_device_train_batch_size
        16
        ```

        Set the training and evaluation batch sizes along with other dataloader related options.
        """
        # Set the training batch size per device
        self.per_device_train_batch_size = train_batch_size
        # Set the evaluation batch size per device
        self.per_device_eval_batch_size = eval_batch_size
        # Set whether to drop the last incomplete batch
        self.dataloader_drop_last = drop_last
        # Set the number of worker subprocesses for data loading
        self.dataloader_num_workers = num_workers
        # Set whether to pin memory in data loaders
        self.dataloader_pin_memory = pin_memory
        # Set whether to keep worker processes alive after dataset consumption
        self.dataloader_persistent_workers = persistent_workers
        # Set whether to automatically find a batch size that fits into memory
        self.auto_find_batch_size = auto_find_batch_size
        # Set whether to ignore data skipping when resuming training
        self.ignore_data_skip = ignore_data_skip
        # Set the random seed for data samplers
        self.data_seed = sampler_seed
        # Return the modified TrainingArguments object
        return self
# 定义一个枚举类，表示并行模式，包括以下选项：
class ParallelMode(Enum):
    # 未并行，指示不使用并行模式
    NOT_PARALLEL = "not_parallel"
    # 非分布式，并行，但不涉及分布式计算
    NOT_DISTRIBUTED = "not_distributed"
    # 分布式，并行，涉及到分布式计算
    DISTRIBUTED = "distributed"
    # SageMaker 模型并行，指示使用 Amazon SageMaker 进行模型并行计算
    SAGEMAKER_MODEL_PARALLEL = "sagemaker_model_parallel"
    # SageMaker 数据并行，指示使用 Amazon SageMaker 进行数据并行计算
    SAGEMAKER_DATA_PARALLEL = "sagemaker_data_parallel"
    # TPU，并行，指示使用 Google 的 TPU（张量处理单元）进行并行计算
    TPU = "tpu"
```