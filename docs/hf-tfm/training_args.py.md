# `.\training_args.py`

```py
# 导入必要的库和模块，这些库和模块用于整个程序的功能实现
import contextlib  # 上下文管理工具，用于创建上下文管理器和支持上下文管理协议的对象
import io  # 提供了用于处理流的核心工具，如文本、二进制和内存缓冲区
import json  # 处理 JSON 格式数据的库
import math  # 数学函数库，提供了标准的数学运算函数
import os  # 操作系统相关功能的库，提供了与操作系统交互的方法
import warnings  # 警告处理工具，用于控制警告的显示方式

from dataclasses import asdict, dataclass, field, fields  # 数据类相关功能，用于创建和操作数据类
from datetime import timedelta  # 处理时间间隔的类和函数
from enum import Enum  # 枚举类型的支持
from pathlib import Path  # 处理路径的类和函数
from typing import Any, Dict, List, Optional, Union  # 类型提示相关功能

from huggingface_hub import get_full_repo_name  # Hugging Face Hub 相关功能，用于获取完整仓库名
from packaging import version  # 版本号处理工具，用于比较和操作版本号

from .debug_utils import DebugOption  # 自定义模块中的调试选项
from .trainer_utils import (  # 自定义模块中的训练器相关工具
    EvaluationStrategy,
    FSDPOption,
    HubStrategy,
    IntervalStrategy,
    SchedulerType,
)
from .utils import (  # 自定义模块中的实用工具集合
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
    is_torch_xla_available,
    is_torch_xpu_available,
    logging,
    requires_backends,
)
from .utils.generic import strtobool  # 自定义模块中的通用工具，如字符串转布尔值
from .utils.import_utils import is_optimum_neuron_available  # 自定义模块中的导入工具，检查神经核是否可用

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)
# 复制日志级别字典，以便在训练器日志级别中添加 passsive 级别
log_levels = logging.get_log_levels_dict().copy()
trainer_log_levels = dict(**log_levels, passive=-1)

# 如果 Torch 可用，导入相关模块
if is_torch_available():
    import torch  # 导入 PyTorch 库
    import torch.distributed as dist  # 导入 PyTorch 分布式训练支持模块

    from .pytorch_utils import is_torch_greater_or_equal_than_2_0  # 导入自定义的 PyTorch 工具函数

# 如果 Accelerate 可用，导入相关模块
if is_accelerate_available():
    from accelerate.state import AcceleratorState, PartialState  # 导入加速器状态相关模块
    from accelerate.utils import DistributedType  # 导入分布式类型枚举

    from .trainer_pt_utils import AcceleratorConfig  # 导入自定义的加速器配置类

# 如果 Torch XLA 可用，导入相关模块
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # 导入 Torch XLA 核心模块

# 如果 Torch NeuronCore 可用，导入相关模块
if is_torch_neuroncore_available(check_device=False):
    # 支持 Torchrun 的特定导入，参考：https://github.com/pytorch/xla/pull/3609
    pass
    # 检查是否设置了环境变量 TORCHELASTIC_RUN_ID
    if os.environ.get("TORCHELASTIC_RUN_ID"):
        # 检查是否有最佳神经元可用
        if is_optimum_neuron_available():
            # 如果有最佳神经元可用，记录信息提示用户使用 TrainiumTrainer 进行训练
            logger.info(
                "Make sure that you are performing the training with the TrainiumTrainer from optimum[neuron], this "
                "will fail otherwise."
            )
        else:
            # 如果没有最佳神经元可用，警告用户使用 optimum[neuron] 的 TrainiumTrainer 替代 Transformers 库进行训练
            logger.warning(
                "Please use the TrainiumTrainer from optimum[neuron] instead of the Transformers library to perform "
                "training on AWS Trainium instances. More information here: "
                "https://github.com/huggingface/optimum-neuron"
            )
            # 导入 torch_xla.distributed.xla_backend 并使用其 ProcessGroupXla
            import torch_xla.distributed.xla_backend as xbn
            
            # 如果当前的分布式组不是 ProcessGroupXla 类型，则尝试使用 XLA 后端初始化分布式进程组
            if not isinstance(dist.group.WORLD, xbn.ProcessGroupXla):
                dist.init_process_group(backend="xla")
                # 再次检查分布式组是否成功初始化为 ProcessGroupXla 类型，否则抛出断言错误
                if not isinstance(dist.group.WORLD, xbn.ProcessGroupXla):
                    raise AssertionError("Failed to initialize torch.distributed process group using XLA backend.")
if is_sagemaker_mp_enabled():
    # 如果在SageMaker中启用了模型并行，则导入相应的模型并行库
    import smdistributed.modelparallel.torch as smp
    # 初始化模型并行
    smp.init()


def default_logdir() -> str:
    """
    Same default as PyTorch
    """
    # 导入所需的库
    import socket
    from datetime import datetime

    # 获取当前时间并格式化
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    # 构建默认的日志目录路径
    return os.path.join("runs", current_time + "_" + socket.gethostname())


def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    # 遍历环境变量列表
    for e in env_keys:
        # 获取环境变量值，并尝试转换为整数，如果无法转换则返回默认值
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    # 如果所有环境变量都不符合要求，则返回默认值
    return default


def get_xla_device_type(device: "torch.device") -> Optional[str]:
    """
    Returns the xla device type (CPU|GPU|TPU) or None if the device is a non-xla device.
    """
    # 检查是否支持PyTorch XLA
    if is_torch_xla_available():
        # 如果设备类型为CPU，则返回"CPU"
        if device.type == "cpu":
            return "CPU"
        # 否则返回XLA真实设备列表中第一个设备类型
        return xm.xla_real_devices([device])[0].split(":")[0]
    # 如果不支持PyTorch XLA，则返回None
    return None


class OptimizerNames(ExplicitEnum):
    """
    Stores the acceptable string identifiers for optimizers.
    """

    # 枚举优化器的可接受字符串标识
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
    RMSPROP_BNB = "rmsprop_bnb"
    RMSPROP_8BIT = "rmsprop_bnb_8bit"
    RMSPROP_32BIT = "rmsprop_bnb_32bit"
    GALORE_ADAMW = "galore_adamw"
    GALORE_ADAMW_8BIT = "galore_adamw_8bit"
    GALORE_ADAFACTOR = "galore_adafactor"
    GALORE_ADAMW_LAYERWISE = "galore_adamw_layerwise"
    GALORE_ADAMW_8BIT_LAYERWISE = "galore_adamw_8bit_layerwise"
    GALORE_ADAFACTOR_LAYERWISE = "galore_adafactor_layerwise"


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

    # 指定框架为PyTorch
    framework = "pt"
    # 定义输出目录路径，用于存储模型预测和检查点
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    # 是否覆盖输出目录的内容，默认为False
    # 当output_dir指向检查点目录时，设置为True以继续训练

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    # 是否运行训练，默认为False

    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    # 是否在开发集上运行评估，默认为False

    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    # 是否在测试集上运行预测，默认为False

    evaluation_strategy: Union[IntervalStrategy, str] = field(
        default="no",
        metadata={"help": "The evaluation strategy to use."},
    )
    # 使用的评估策略，默认为"no"

    prediction_loss_only: bool = field(
        default=False,
        metadata={"help": "When performing evaluation and predictions, only returns the loss."},
    )
    # 在执行评估和预测时，是否只返回损失，默认为False

    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
    )
    # 每个GPU/TPU/MPS/NPU core/CPU的训练批次大小，默认为8

    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation."}
    )
    # 每个GPU/TPU/MPS/NPU core/CPU的评估批次大小，默认为8

    per_gpu_train_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Deprecated, the use of `--per_device_train_batch_size` is preferred. "
                "Batch size per GPU/TPU core/CPU for training."
            )
        },
    )
    # 每个GPU/TPU core/CPU的训练批次大小（已弃用），建议使用`--per_device_train_batch_size`

    per_gpu_eval_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Deprecated, the use of `--per_device_eval_batch_size` is preferred. "
                "Batch size per GPU/TPU core/CPU for evaluation."
            )
        },
    )
    # 每个GPU/TPU core/CPU的评估批次大小（已弃用），建议使用`--per_device_eval_batch_size`

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    # 执行反向传播/更新步骤之前累积的更新步骤数，默认为1

    eval_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of predictions steps to accumulate before moving the tensors to the CPU."},
    )
    # 在将张量移动到CPU之前累积的预测步骤数，默认为None

    eval_delay: Optional[float] = field(
        default=0,
        metadata={
            "help": (
                "Number of epochs or steps to wait for before the first evaluation can be performed, depending on the"
                " evaluation_strategy."
            )
        },
    )
    # 在第一次评估之前等待的时期或步骤数，取决于评估策略，默认为0

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    # AdamW优化器的初始学习率，默认为5e-5

    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    # 如果应用的话，AdamW的权重衰减率，默认为0.0

    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    # AdamW优化器的Beta1参数，默认为0.9

    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    # AdamW优化器的Beta2参数，默认为0.999

    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    # AdamW优化器的Epsilon参数，默认为1e-8
    # 定义最大梯度范数，默认为1.0，用于梯度裁剪
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    # 定义总的训练周期数，默认为3.0
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    
    # 定义最大训练步数，默认为-1，如果大于0，则设置总的训练步数，覆盖num_train_epochs的设定
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    
    # 定义学习率调度器的类型，默认为"linear"
    lr_scheduler_type: Union[SchedulerType, str] = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )
    
    # 学习率调度器的额外参数设定，默认为空字典，例如{'num_cycles': 1}用于余弦退火重启时的参数设置
    lr_scheduler_kwargs: Optional[Dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Extra parameters for the lr_scheduler such as {'num_cycles': 1} for the cosine with hard restarts"
            )
        },
    )
    
    # 线性预热的比例，默认为0.0，表示在总步数的这一部分上进行线性预热
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    
    # 线性预热的步数，默认为0，表示固定的线性预热步数
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    # 主节点日志记录级别，默认为"passive"，允许应用程序设定日志级别
    log_level: Optional[str] = field(
        default="passive",
        metadata={
            "help": (
                "Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug',"
                " 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and"
                " lets the application set the level. Defaults to 'passive'."
            ),
            "choices": trainer_log_levels.keys(),  # 可选的日志级别
        },
    )
    
    # 复制节点日志记录级别，默认为"warning"，与主节点日志记录级别相同
    log_level_replica: Optional[str] = field(
        default="warning",
        metadata={
            "help": "Logger log level to use on replica nodes. Same choices and defaults as ``log_level``",
            "choices": trainer_log_levels.keys(),  # 可选的日志级别
        },
    )
    
    # 多节点分布式训练时，是否在每个节点记录日志，默认为True表示每个节点都记录日志
    log_on_each_node: bool = field(
        default=True,
        metadata={
            "help": (
                "When doing a multinode distributed training, whether to log once per node or just once on the main"
                " node."
            )
        },
    )
    
    # Tensorboard日志目录，默认为None
    logging_dir: Optional[str] = field(default=None, metadata={"help": "Tensorboard log dir."})
    
    # 训练过程中的日志记录策略，默认为"steps"，表示每隔一定步数记录一次日志
    logging_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    
    # 是否记录第一个全局步数，默认为False
    logging_first_step: bool = field(default=False, metadata={"help": "Log the first global_step"})
    
    # 每隔多少步记录一次日志，默认为500，可以是整数或小于1的浮点数，表示比例
    logging_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    
    # 是否过滤掉记录中的NaN和Inf损失，默认为True
    logging_nan_inf_filter: bool = field(default=True, metadata={"help": "Filter nan and inf losses for logging."})
    
    # 检查点保存策略，默认为"steps"，表示每隔一定步数保存一次检查点
    save_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    # 定义一个浮点类型的字段 `save_steps`，默认值为 500
    save_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )

    # 定义一个可选整数类型的字段 `save_total_limit`，默认值为 None
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

    # 定义一个可选布尔类型的字段 `save_safetensors`，默认值为 True
    save_safetensors: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Use safetensors saving and loading for state dicts instead of default torch.load and torch.save."
        },
    )

    # 定义一个布尔类型的字段 `save_on_each_node`，默认值为 False
    save_on_each_node: bool = field(
        default=False,
        metadata={
            "help": (
                "When doing multi-node distributed training, whether to save models and checkpoints on each node, or"
                " only on the main one"
            )
        },
    )

    # 定义一个布尔类型的字段 `save_only_model`，默认值为 False
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

    # 定义一个布尔类型的字段 `no_cuda`，默认值为 False
    no_cuda: bool = field(
        default=False,
        metadata={"help": "This argument is deprecated. It will be removed in version 5.0 of 🤗 Transformers."},
    )

    # 定义一个布尔类型的字段 `use_cpu`，默认值为 False
    use_cpu: bool = field(
        default=False,
        metadata={
            "help": " Whether or not to use cpu. If set to False, we will use cuda/tpu/mps/npu device if available."
        },
    )

    # 定义一个布尔类型的字段 `use_mps_device`，默认值为 False
    use_mps_device: bool = field(
        default=False,
        metadata={
            "help": "This argument is deprecated. `mps` device will be used if available similar to `cuda` device."
            " It will be removed in version 5.0 of 🤗 Transformers"
        },
    )
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    # 设置随机种子，用于训练开始时的随机性
    data_seed: Optional[int] = field(default=None, metadata={"help": "Random seed to be used with data samplers."})
    # 数据采样器使用的随机种子，可选参数
    jit_mode_eval: bool = field(
        default=False, metadata={"help": "Whether or not to use PyTorch jit trace for inference"}
    )
    # 是否使用 PyTorch jit 追踪进行推断
    use_ipex: bool = field(
        default=False,
        metadata={
            "help": (
                "Use Intel extension for PyTorch when it is available, installation:"
                " 'https://github.com/intel/intel-extension-for-pytorch'"
            )
        },
    )
    # 在可用时是否使用 Intel 扩展进行 PyTorch 加速
    bf16: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
            )
        },
    )
    # 是否使用 bf16（混合）精度替代 32 位精度
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )
    # 是否使用 fp16（混合）精度替代 32 位精度
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. "
                "See details at https://nvidia.github.io/apex/amp.html"
            )
        },
    )
    # fp16 使用的优化级别，选择在 ['O0', 'O1', 'O2', 'O3'] 中的一个
    half_precision_backend: str = field(
        default="auto",
        metadata={
            "help": "The backend to be used for half precision.",
            "choices": ["auto", "apex", "cpu_amp"],
        },
    )
    # 用于半精度计算的后端选择，可选值为 ['auto', 'apex', 'cpu_amp']
    bf16_full_eval: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use full bfloat16 evaluation instead of 32-bit. This is an experimental API and it may"
                " change."
            )
        },
    )
    # 是否使用 bf16（完整）评估替代 32 位精度
    fp16_full_eval: bool = field(
        default=False,
        metadata={"help": "Whether to use full float16 evaluation instead of 32-bit"},
    )
    # 是否使用 fp16（完整）评估替代 32 位精度
    tf32: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to enable tf32 mode, available in Ampere and newer GPU architectures. This is an experimental"
                " API and it may change."
            )
        },
    )
    # 是否启用 tf32 模式，仅适用于 Ampere 及更新的 GPU 架构
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})
    # 分布式训练中的本地排名
    ddp_backend: Optional[str] = field(
        default=None,
        metadata={
            "help": "The backend to be used for distributed training",
            "choices": ["nccl", "gloo", "mpi", "ccl", "hccl"],
        },
    )
    # 分布式训练使用的后端选择，可选值为 ['nccl', 'gloo', 'mpi', 'ccl', 'hccl']
    tpu_num_cores: Optional[int] = field(
        default=None, metadata={"help": "TPU: Number of TPU cores (automatically passed by launcher script)"}
    )
    # TPU 使用的核心数
    tpu_metrics_debug: bool = field(
        default=False,
        metadata={
            "help": (
                "已弃用，推荐使用 `--debug tpu_metrics_debug`。TPU：是否打印调试指标"
            )
        },
    )
    debug: Union[str, List[DebugOption]] = field(
        default="",
        metadata={
            "help": (
                "是否启用调试模式。当前选项："
                "`underflow_overflow`（检测激活和权重中的下溢和上溢），"
                "`tpu_metrics_debug`（在TPU上打印调试指标）。"
            )
        },
    )

    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "如果不是批量大小的整数倍，丢弃最后不完整的批次。"}
    )
    eval_steps: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "每隔X步运行一次评估。应为整数或范围为`[0,1)`的浮点数。"
                "如果小于1，将解释为总训练步数的比例。"
            )
        },
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": (
                "用于数据加载的子进程数（仅适用于PyTorch）。"
                "0表示数据将在主进程中加载。"
            )
        },
    )
    dataloader_prefetch_factor: Optional[int] = field(
        default=None if not is_torch_available() or is_torch_greater_or_equal_than_2_0 else 2,
        metadata={
            "help": (
                "每个工作进程预加载的批次数。"
                "2表示每个工作进程预加载2 * num_workers批次。"
                "对于PyTorch < 2.0.0，默认为2，否则为None。"
            )
        },
    )
    past_index: int = field(
        default=-1,
        metadata={"help": "如果 >= 0，则使用输出的相应部分作为下一步的过去状态。"},
    )

    run_name: Optional[str] = field(
        default=None, metadata={"help": "运行的可选描述符。主要用于wandb日志记录。"}
    )
    disable_tqdm: Optional[bool] = field(
        default=None, metadata={"help": "是否禁用tqdm进度条。"}
    )

    remove_unused_columns: Optional[bool] = field(
        default=True, metadata={"help": "在使用nlp.Dataset时，移除模型不需要的列。"}
    )
    label_names: Optional[List[str]] = field(
        default=None, metadata={"help": "输入字典中与标签对应的键列表。"}
    )
    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to load the best model found during training at the end of training. When this option"
                " is enabled, the best checkpoint will always be saved. See `save_total_limit` for more."
            )
        },
    )
    # 是否在训练结束时加载找到的最佳模型。启用此选项时，始终保存最佳检查点。详见 `save_total_limit`。
    
    metric_for_best_model: Optional[str] = field(
        default=None, metadata={"help": "The metric to use to compare two different models."}
    )
    # 用于比较两个不同模型的度量标准。

    greater_is_better: Optional[bool] = field(
        default=None, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )
    # 是否应最大化 `metric_for_best_model`。

    ignore_data_skip: bool = field(
        default=False,
        metadata={
            "help": (
                "When resuming training, whether or not to skip the first epochs and batches to get to the same"
                " training data."
            )
        },
    )
    # 在恢复训练时，是否跳过初始的若干轮次和批次，以达到相同的训练数据。

    fsdp: Optional[Union[List[FSDPOption], str]] = field(
        default="",
        metadata={
            "help": (
                "Whether or not to use PyTorch Fully Sharded Data Parallel (FSDP) training (in distributed training"
                " only). The base option should be `full_shard`, `shard_grad_op` or `no_shard` and you can add"
                " CPU-offload to `full_shard` or `shard_grad_op` like this: full_shard offload` or `shard_grad_op"
                " offload`. You can add auto-wrap to `full_shard` or `shard_grad_op` with the same syntax: full_shard"
                " auto_wrap` or `shard_grad_op auto_wrap`."
            ),
        },
    )
    # 是否使用 PyTorch 完全分片数据并行（FSDP）训练（仅限分布式训练）。基本选项应为 `full_shard`、`shard_grad_op` 或 `no_shard`，
    # 可以如下方式添加 CPU-offload 到 `full_shard` 或 `shard_grad_op`：`full_shard offload` 或 `shard_grad_op offload`。
    # 可以使用相同的语法为 `full_shard` 或 `shard_grad_op` 添加自动包装：`full_shard auto_wrap` 或 `shard_grad_op auto_wrap`。

    fsdp_min_num_params: int = field(
        default=0,
        metadata={
            "help": (
                "This parameter is deprecated. FSDP's minimum number of parameters for Default Auto Wrapping. (useful"
                " only when `fsdp` field is passed)."
            )
        },
    )
    # 此参数已弃用。FSDP 的默认自动包装最小参数数量。（仅当传递 `fsdp` 字段时有效）。

    # Do not touch this type annotation or it will stop working in CLI
    fsdp_config: Optional[Union[dict, str]] = field(
        default=None,
        metadata={
            "help": (
                "Config to be used with FSDP (Pytorch Fully Sharded  Data Parallel). The value is either a "
                "fsdp json config file (e.g., `fsdp_config.json`) or an already loaded json file as `dict`."
            )
        },
    )
    # 用于 FSDP（Pytorch 完全分片数据并行）的配置。值可以是 fsdp 的 JSON 配置文件（例如 `fsdp_config.json`）或已加载的 `dict`。

    fsdp_transformer_layer_cls_to_wrap: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "This parameter is deprecated. Transformer layer class name (case-sensitive) to wrap, e.g,"
                " `BertLayer`, `GPTJBlock`, `T5Block` .... (useful only when `fsdp` flag is passed)."
            )
        },
    )
    # 此参数已弃用。要包装的 Transformer 层类名（区分大小写），例如 `BertLayer`、`GPTJBlock`、`T5Block` ...... （仅当传递 `fsdp` 标志时有效）。

    # Do not touch this type annotation or it will stop working in CLI
    accelerator_config: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Config to be used with the internal Accelerator object initializtion. The value is either a "
                "accelerator json config file (e.g., `accelerator_config.json`) or an already loaded json file as `dict`."
            )
        },
    )
    # accelerator_config参数，用于内部加速器对象初始化的配置
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Enable deepspeed and pass the path to deepspeed json config file (e.g. `ds_config.json`) or an already"
                " loaded json file as a dict"
            )
        },
    )
    # deepspeed参数，用于启用deepspeed并传递deepspeed json配置文件的路径或已加载的json文件作为字典
    label_smoothing_factor: float = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (zero means no label smoothing)."}
    )
    # label_smoothing_factor参数，用于应用标签平滑的ε值（零表示不进行标签平滑）

    default_optim = "adamw_torch"
    # 默认优化器设定为"adamw_torch"
    # XXX: enable when pytorch==2.0.1 comes out - we want to give it time to get all the bugs sorted out
    # if is_torch_available() and version.parse(version.parse(torch.__version__).base_version) >= version.parse("2.1.0"):
    #     default_optim = "adamw_torch_fused"
    # and update the doc above to:
    # optim (`str` or [`training_args.OptimizerNames`], *optional*, defaults to `"adamw_torch_fused"` (for torch<2.1.0 `"adamw_torch"`):
    # 当pytorch版本为2.0.1时启用，我们希望给它足够的时间来解决所有的bug
    # 如果torch可用且版本大于等于2.1.0，则将默认优化器更新为"adamw_torch_fused"，否则为"adamw_torch"
    optim: Union[OptimizerNames, str] = field(
        default=default_optim,
        metadata={"help": "The optimizer to use."},
    )
    # optim参数，用于指定要使用的优化器
    optim_args: Optional[str] = field(default=None, metadata={"help": "Optional arguments to supply to optimizer."})
    # optim_args参数，用于传递给优化器的可选参数
    adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    # adafactor参数，用于指定是否使用Adafactor替代AdamW
    group_by_length: bool = field(
        default=False,
        metadata={"help": "Whether or not to group samples of roughly the same length together when batching."},
    )
    # group_by_length参数，用于指定是否在批处理时将大致相同长度的样本分组在一起
    length_column_name: Optional[str] = field(
        default="length",
        metadata={"help": "Column name with precomputed lengths to use when grouping by length."},
    )
    # length_column_name参数，用于指定在按长度分组时使用的预计算长度的列名
    report_to: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of integrations to report the results and logs to."}
    )
    # report_to参数，用于指定要报告结果和日志的集成列表
    ddp_find_unused_parameters: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `find_unused_parameters` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    # ddp_find_unused_parameters参数，用于在使用分布式训练时传递给`DistributedDataParallel`的`find_unused_parameters`标志的值
    ddp_bucket_cap_mb: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `bucket_cap_mb` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    # ddp_bucket_cap_mb参数，用于在使用分布式训练时传递给`DistributedDataParallel`的`bucket_cap_mb`标志的值
    # 用于分布式训练中，指定是否将 `broadcast_buffers` 标志传递给 `DistributedDataParallel`。
    ddp_broadcast_buffers: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `broadcast_buffers` passed to "
                "`DistributedDataParallel`."
            )
        },
    )

    # 是否为 DataLoader 固定内存。
    dataloader_pin_memory: bool = field(
        default=True, metadata={"help": "Whether or not to pin memory for DataLoader."}
    )

    # 是否保持 DataLoader 的 worker 进程持久化，不在每次数据集使用完后关闭。
    dataloader_persistent_workers: bool = field(
        default=False,
        metadata={
            "help": "If True, the data loader will not shut down the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive. Can potentially speed up training, but will increase RAM usage."
        },
    )

    # 是否跳过将内存分析报告添加到指标中。
    skip_memory_metrics: bool = field(
        default=True, metadata={"help": "Whether or not to skip adding of memory profiler reports to metrics."}
    )

    # 是否使用旧版的 prediction_loop 在 Trainer 中。
    use_legacy_prediction_loop: bool = field(
        default=False, metadata={"help": "Whether or not to use the legacy prediction_loop in the Trainer."}
    )

    # 是否在训练结束后上传训练好的模型到模型中心。
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )

    # 从检查点恢复训练的路径。
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )

    # 与本地 `output_dir` 保持同步的模型中心的名称。
    hub_model_id: Optional[str] = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )

    # 在 `--push_to_hub` 激活时使用的模型中心策略。
    hub_strategy: Union[HubStrategy, str] = field(
        default="every_save",
        metadata={"help": "The hub strategy to use when `--push_to_hub` is activated."},
    )

    # 用于推送模型到模型中心的令牌。
    hub_token: Optional[str] = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})

    # 模型存储库是否是私有的。
    hub_private_repo: bool = field(default=False, metadata={"help": "Whether the model repository is private or not."})

    # 如果为 `False`，则如果上一个推送未完成，Trainer 将跳过推送。
    hub_always_push: bool = field(
        default=False,
        metadata={"help": "Unless `True`, the Trainer will skip pushes if the previous one wasn't finished yet."},
    )

    # 是否使用梯度检查点来节省内存，尽管会导致反向传播速度变慢。
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )

    # 梯度检查点的关键字参数，例如 `use_reentrant`，将传递给 `torch.utils.checkpoint.checkpoint` 通过 `model.gradient_checkpointing_enable`。
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Gradient checkpointing key word arguments such as `use_reentrant`. Will be passed to `torch.utils.checkpoint.checkpoint` through `model.gradient_checkpointing_enable`."
        },
    )

    # 是否将输入传递给 `compute_metrics` 函数以计算指标。
    include_inputs_for_metrics: bool = field(
        default=False, metadata={"help": "Whether or not the inputs will be passed to the `compute_metrics` function."}
    )
    # 已弃用的参数
    fp16_backend: str = field(
        default="auto",
        metadata={
            "help": "Deprecated. Use half_precision_backend instead",
            "choices": ["auto", "apex", "cpu_amp"],
        },
    )
    # 初始化一个字符串字段，表示混合精度计算的后端选择，默认为"auto"，可选值为["auto", "apex", "cpu_amp"]。
    push_to_hub_model_id: Optional[str] = field(
        default=None, metadata={"help": "The name of the repository to which push the `Trainer`."}
    )
    # 可选的字符串字段，用于指定要推送到的模型仓库的名称。
    push_to_hub_organization: Optional[str] = field(
        default=None, metadata={"help": "The name of the organization in with to which push the `Trainer`."}
    )
    # 可选的字符串字段，用于指定要推送到的组织的名称。
    push_to_hub_token: Optional[str] = field(
        default=None, metadata={"help": "The token to use to push to the Model Hub."}
    )
    # 可选的字符串字段，用于指定用于推送到模型中心的令牌。
    _n_gpu: int = field(init=False, repr=False, default=-1)
    # 不可初始化和不可显示的整数字段，表示GPU的数量，默认为-1。
    mp_parameters: str = field(
        default="",
        metadata={"help": "Used by the SageMaker launcher to send mp-specific args. Ignored in Trainer"},
    )
    # 字符串字段，默认为空字符串，用于SageMaker启动器发送特定的多进程参数，Trainer中被忽略。

    auto_find_batch_size: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to automatically decrease the batch size in half and rerun the training loop again each time"
                " a CUDA Out-of-Memory was reached"
            )
        },
    )
    # 布尔字段，默认为False，控制是否在每次CUDA内存溢出时自动减少批量大小并重新运行训练循环。
    full_determinism: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to call enable_full_determinism instead of set_seed for reproducibility in distributed"
                " training. Important: this will negatively impact the performance, so only use it for debugging."
            )
        },
    )
    # 布尔字段，默认为False，控制是否在分布式训练中使用enable_full_determinism而不是set_seed来实现可重复性。
    torchdynamo: Optional[str] = field(
        default=None,
        metadata={
            "help": "This argument is deprecated, use `--torch_compile_backend` instead.",
        },
    )
    # 可选的字符串字段，已废弃，建议使用`--torch_compile_backend`代替。
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
    # 可选的字符串字段，默认为"last"，用于在使用Ray进行超参数搜索时指定作用域。
    ddp_timeout: Optional[int] = field(
        default=1800,
        metadata={
            "help": "Overrides the default timeout for distributed training (value should be given in seconds)."
        },
    )
    # 可选的整数字段，默认为1800，用于覆盖分布式训练的默认超时时间（以秒为单位）。
    torch_compile: bool = field(
        default=False, metadata={"help": "If set to `True`, the model will be wrapped in `torch.compile`."}
    )
    # 布尔字段，默认为False，如果设置为True，模型将被包装在torch.compile中。
    torch_compile_backend: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which backend to use with `torch.compile`, passing one will trigger a model compilation.",
        },
    )
    # 可选的字符串字段，用于指定在torch.compile中使用的后端。
    torch_compile_mode: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which mode to use with `torch.compile`, passing one will trigger a model compilation.",
        },
    )
    # 可选的编译模式，用于指定 `torch.compile` 的模式，传入一个值将触发模型编译。

    dispatch_batches: Optional[bool] = field(
        default=None,
        metadata={"help": "Deprecated. Pass {'dispatch_batches':VALUE} to `accelerator_config`."},
    )
    # 已弃用。通过将 {'dispatch_batches':VALUE} 传递给 `accelerator_config` 来代替。

    split_batches: Optional[bool] = field(
        default=None,
        metadata={"help": "Deprecated. Pass {'split_batches':True} to `accelerator_config`."},
    )
    # 已弃用。通过将 {'split_batches':True} 传递给 `accelerator_config` 来代替。

    include_tokens_per_second: Optional[bool] = field(
        default=False,
        metadata={"help": "If set to `True`, the speed metrics will include `tgs` (tokens per second per device)."},
    )
    # 如果设置为 `True`，速度指标将包括 `tgs`（每设备每秒标记数）。

    include_num_input_tokens_seen: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If set to `True`, will track the number of input tokens seen throughout training. (May be slower in distributed training)"
        },
    )
    # 如果设置为 `True`，将跟踪训练过程中看到的输入标记数量。（在分布式训练中可能会变慢）

    neftune_noise_alpha: Optional[float] = field(
        default=None,
        metadata={
            "help": "Activates neftune noise embeddings into the model. NEFTune has been proven to drastically improve model performances for instrcution fine-tuning. Check out the original paper here: https://arxiv.org/abs/2310.05914 and the original code here: https://github.com/neelsjain/NEFTune. Only supported for `PreTrainedModel` and `PeftModel` classes."
        },
    )
    # 激活 NEFTune 噪声嵌入到模型中。NEFTune 已被证明可以显著改善指令微调的模型性能。查看原始论文：https://arxiv.org/abs/2310.05914 和原始代码：https://github.com/neelsjain/NEFTune。仅支持 `PreTrainedModel` 和 `PeftModel` 类。

    optim_target_modules: Union[None, str, List[str]] = field(
        default=None,
        metadata={
            "help": "Target modules for the optimizer defined in the `optim` argument. Only used for the GaLore optimizer at the moment."
        },
    )
    # 用于优化器中 `optim` 参数定义的目标模块。目前仅用于 GaLore 优化器。

    def __str__(self):
        self_as_dict = asdict(self)

        # Remove deprecated arguments. That code should be removed once
        # those deprecated arguments are removed from TrainingArguments. (TODO: v5)
        del self_as_dict["per_gpu_train_batch_size"]
        del self_as_dict["per_gpu_eval_batch_size"]

        self_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in self_as_dict.items()}

        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__
    # 将对象转换为字符串表示形式的方法和其 `__repr__` 方法的重写。

    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training (may differ from `per_gpu_train_batch_size` in distributed training).
        """
        # 如果定义了 per_gpu_train_batch_size，则发出警告信息，因为这个参数在将来版本中将被移除
        if self.per_gpu_train_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_train_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_train_batch_size` is preferred."
            )
        # 根据是否设置了 per_gpu_train_batch_size 来确定每个设备的批处理大小
        per_device_batch_size = self.per_gpu_train_batch_size or self.per_device_train_batch_size
        # 计算实际的训练批处理大小，考虑到 GPU 数量
        train_batch_size = per_device_batch_size * max(1, self.n_gpu)
        return train_batch_size

    @property
    def eval_batch_size(self) -> int:
        """
        The actual batch size for evaluation (may differ from `per_gpu_eval_batch_size` in distributed training).
        """
        # 如果定义了 per_gpu_eval_batch_size，则发出警告信息，因为这个参数在将来版本中将被移除
        if self.per_gpu_eval_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_eval_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_eval_batch_size` is preferred."
            )
        # 根据是否设置了 per_gpu_eval_batch_size 来确定每个设备的批处理大小
        per_device_batch_size = self.per_gpu_eval_batch_size or self.per_device_eval_batch_size
        # 计算实际的评估批处理大小，考虑到 GPU 数量
        eval_batch_size = per_device_batch_size * max(1, self.n_gpu)
        return eval_batch_size

    @property
    def ddp_timeout_delta(self) -> timedelta:
        """
        The actual timeout for torch.distributed.init_process_group since it expects a timedelta variable.
        """
        # 返回用于 torch.distributed.init_process_group 的超时时间，作为 timedelta 变量
        return timedelta(seconds=self.ddp_timeout)

    @cached_property
    @property
    def device(self) -> "torch.device":
        """
        The device used by this process.
        """
        # 确保 torch 被正确加载
        requires_backends(self, ["torch"])
        # 返回当前进程使用的设备对象
        return self._setup_devices

    @property
    def n_gpu(self):
        """
        The number of GPUs used by this process.

        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        # 确保 torch 被正确加载
        requires_backends(self, ["torch"])
        # 确保 self._n_gpu 被正确设置
        if not hasattr(self, "_n_gpu"):
            _ = self._setup_devices
        # 返回当前进程使用的 GPU 数量
        return self._n_gpu
    def parallel_mode(self):
        """
        The current mode used for parallelism if multiple GPUs/TPU cores are available. One of:

        - `ParallelMode.NOT_PARALLEL`: no parallelism (CPU or one GPU).
        - `ParallelMode.NOT_DISTRIBUTED`: several GPUs in one single process (uses `torch.nn.DataParallel`).
        - `ParallelMode.DISTRIBUTED`: several GPUs, each having its own process (uses
          `torch.nn.DistributedDataParallel`).
        - `ParallelMode.TPU`: several TPU cores.
        """
        # 确保所需后端库存在，此处需要 "torch"
        requires_backends(self, ["torch"])
        # 如果当前环境支持 TPU，则返回 TPU 并行模式
        if is_torch_xla_available():
            return ParallelMode.TPU
        # 如果使用 SageMaker 并启用了模型并行，则返回 SageMaker 模型并行模式
        elif is_sagemaker_mp_enabled():
            return ParallelMode.SAGEMAKER_MODEL_PARALLEL
        # 如果使用 SageMaker 并启用了数据并行，则返回 SageMaker 数据并行模式
        elif is_sagemaker_dp_enabled():
            return ParallelMode.SAGEMAKER_DATA_PARALLEL
        # 如果分布式状态存在且不是未分布式类型，或者本地排名不为 -1，则返回分布式并行模式
        elif (
            self.distributed_state is not None and self.distributed_state.distributed_type != DistributedType.NO
        ) or (self.distributed_state is None and self.local_rank != -1):
            return ParallelMode.DISTRIBUTED
        # 如果 GPU 数量大于 1，则返回非分布式并行模式
        elif self.n_gpu > 1:
            return ParallelMode.NOT_DISTRIBUTED
        # 否则返回非并行模式
        else:
            return ParallelMode.NOT_PARALLEL

    @property
    def world_size(self):
        """
        The number of processes used in parallel.
        """
        # 确保所需后端库存在，此处需要 "torch"
        requires_backends(self, ["torch"])
        # 如果分布式状态存在，则返回并行使用的进程数
        if self.distributed_state is not None:
            return self.distributed_state.num_processes
        # 如果使用 SageMaker 并且未启用批次预调整，则返回数据并行的大小
        elif is_sagemaker_mp_enabled():
            return smp.dp_size() if not smp.state.cfg.prescaled_batch else smp.rdp_size()
        # 否则返回默认值 1
        return 1

    @property
    def process_index(self):
        """
        The index of the current process used.
        """
        # 确保所需后端库存在，此处需要 "torch"
        requires_backends(self, ["torch"])
        # 如果分布式状态存在，则返回当前进程的索引
        if self.distributed_state is not None:
            return self.distributed_state.process_index
        # 如果使用 SageMaker 并且未启用批次预调整，则返回数据并行的排名
        elif is_sagemaker_mp_enabled():
            return smp.dp_rank() if not smp.state.cfg.prescaled_batch else smp.rdp_rank()
        # 否则返回默认值 0
        return 0

    @property
    def local_process_index(self):
        """
        The index of the local process used.
        """
        # 确保所需后端库存在，此处需要 "torch"
        requires_backends(self, ["torch"])

        # 如果分布式状态存在，则返回本地进程的索引
        if self.distributed_state is not None:
            return self.distributed_state.local_process_index
        # 如果使用 SageMaker 并启用了本地排名，则返回本地排名
        elif is_sagemaker_mp_enabled():
            return smp.local_rank()
        # 否则返回默认值 0
        return 0

    @property
    def should_log(self):
        """
        Whether or not the current process should produce log.
        """
        # 如果设置为在每个节点上记录日志，则仅当本地进程索引为 0 时返回 True
        if self.log_on_each_node:
            return self.local_process_index == 0
        else:
            # 如果使用 SageMaker 并且当前进程排名为 0，则返回 True
            if is_sagemaker_mp_enabled():
                return smp.rank() == 0
            # 否则仅当当前进程索引为 0 时返回 True
            else:
                return self.process_index == 0
    def should_save(self):
        """
        Whether or not the current process should write to disk, e.g., to save models and checkpoints.
        """
        # 如果设置为在每个节点保存，则仅在本地进程索引为0时返回True
        if self.save_on_each_node:
            return self.local_process_index == 0
        else:
            # 如果在SageMaker多进程环境中启用了多进程，则仅在排名为0的进程返回True
            if is_sagemaker_mp_enabled():
                return smp.rank() == 0
            else:
                # 否则，仅在进程索引为0时返回True
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
        # 将log_level和log_level_replica转换为整数
        log_level = trainer_log_levels[self.log_level]
        log_level_replica = trainer_log_levels[self.log_level_replica]

        # 如果log_level为-1，则使用当前日志级别设置的详细程度
        log_level_main_node = logging.get_verbosity() if log_level == -1 else log_level
        # 如果log_level_replica为-1，则使用默认的WARNING日志级别
        log_level_replica_node = logging.get_verbosity() if log_level_replica == -1 else log_level_replica
        # 根据should_log方法的返回值选择主进程或副本进程的日志级别设置
        return log_level_main_node if self.should_log else log_level_replica_node

    @property
    def place_model_on_device(self):
        """
        Can be subclassed and overridden for some specific integrations.
        """
        # 如果未启用SageMaker多进程，则返回True；否则返回False
        return not is_sagemaker_mp_enabled()

    @property
    def _no_sync_in_gradient_accumulation(self):
        """
        Whether or not to use no_sync for the gradients when doing gradient accumulation.
        """
        # 当不使用DeepSpeed、SageMaker分布式训练、SageMaker多进程或Torch NeuronCore时返回True，否则返回False
        return not (
            self.deepspeed or is_sagemaker_dp_enabled() or is_sagemaker_mp_enabled() or is_torch_neuroncore_available()
        )

    @contextlib.contextmanager
    # 定义一个上下文管理器，用于在 Torch 分布式环境中执行主进程的操作，
    # 阻塞副本进程，并在完成后释放副本。
    def main_process_first(self, local=True, desc="work"):
        """
        A context manager for torch distributed environment where on needs to do something on the main process, while
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
        # 检查当前环境是否支持 Torch，并且是否处于分布式环境中
        if is_torch_available() and self.world_size > 1:
            # 根据参数确定主进程的描述信息
            main_process_desc = "main local process" if local else "main process"
            # 根据当前的分布式状态确定是否为主进程
            if self.distributed_state is not None:
                is_main_process = (
                    self.distributed_state.is_local_main_process if local else self.distributed_state.is_main_process
                )
            elif is_sagemaker_mp_enabled():
                is_main_process = smp.rank() == 0

            try:
                if not is_main_process:
                    # 告知所有副本进程等待
                    logger.debug(f"{self.process_index}: waiting for the {main_process_desc} to perform {desc}")

                    # 如果支持 Torch XLA，则使用其同步方法
                    if is_torch_xla_available():
                        xm.rendezvous(desc)
                    else:
                        # 否则使用 Torch 的分布式 barrier
                        dist.barrier()
                # 使用 yield 将控制权交给调用者，允许在主进程完成后继续执行
                yield
            finally:
                if is_main_process:
                    # 主进程完成任务，释放所有副本
                    logger.debug(f"{self.process_index}: {main_process_desc} completed {desc}, releasing all replicas")
                    if is_torch_xla_available():
                        xm.rendezvous(desc)
                    else:
                        dist.barrier()
        else:
            # 如果不满足分布式条件，则直接 yield
            yield

    # 获取线性预热所需的步数
    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.warmup_steps if self.warmup_steps > 0 else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps
    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # 创建一个空字典 `d`，用于存储实例的序列化数据，仅包含可以初始化的字段
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        # 遍历字典 `d` 中的每个键值对
        for k, v in d.items():
            # 如果值 `v` 是枚举类型 `Enum`，则将其替换为其值
            if isinstance(v, Enum):
                d[k] = v.value
            # 如果值 `v` 是列表且第一个元素是枚举类型 `Enum`，则将列表中所有枚举元素替换为其值
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            # 如果键 `k` 以 "_token" 结尾，将其值 `v` 替换为 `<K_UPPERCASE>` 形式的字符串
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
            # 如果加速器配置可用且值 `v` 是 `AcceleratorConfig` 类型，则将其序列化为字典形式
            if is_accelerate_available() and isinstance(v, AcceleratorConfig):
                d[k] = v.to_dict()
        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        # 将实例序列化为 JSON 字符串，使用两个空格缩进
        return json.dumps(self.to_dict(), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        # 获取原始的字典表示形式
        d = self.to_dict()
        # 将训练批次大小和评估批次大小添加到字典 `d` 中
        d = {**d, **{"train_batch_size": self.train_batch_size, "eval_batch_size": self.eval_batch_size}}

        # 定义有效的数据类型列表
        valid_types = [bool, int, float, str]
        if is_torch_available():
            valid_types.append(torch.Tensor)

        # 返回字典，其中值的类型在有效类型列表中，否则转换为字符串形式
        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}

    # The following methods are there to simplify the instantiation of `TrainingArguments`
    # 下面的方法用于简化 `TrainingArguments` 的实例化设置
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
    ):
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

        ```
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_training(learning_rate=1e-4, batch_size=32)
        >>> args.learning_rate
        1e-4
        ```
        """
        # 设置 self.do_train 为 True，表明将执行训练过程
        self.do_train = True
        # 设置初始学习率
        self.learning_rate = learning_rate
        # 设置每个设备上的训练批次大小
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
        # 返回设置好的参数对象 self
        return self
    # 定义一个方法，用于设置评估相关的所有参数
    def set_evaluate(
        self,
        strategy: Union[str, IntervalStrategy] = "no",
        steps: int = 500,
        batch_size: int = 8,
        accumulation_steps: Optional[int] = None,
        delay: Optional[float] = None,
        loss_only: bool = False,
        jit_mode: bool = False,
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

        ```
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_evaluate(strategy="steps", steps=100)
        >>> args.eval_steps
        100
        ```
        """
        # 将传入的评估策略转换为IntervalStrategy枚举类型
        self.evaluation_strategy = IntervalStrategy(strategy)
        # 如果评估策略为STEPS，并且steps设置为0，则抛出数值错误
        if self.evaluation_strategy == IntervalStrategy.STEPS and steps == 0:
            raise ValueError("Setting `strategy` as 'steps' requires a positive value for `steps`.")
        # 根据评估策略是否为NO来设置是否进行评估
        self.do_eval = self.evaluation_strategy != IntervalStrategy.NO
        # 设置评估步数
        self.eval_steps = steps
        # 设置每个设备的评估批量大小
        self.per_device_eval_batch_size = batch_size
        # 设置评估累积步数
        self.eval_accumulation_steps = accumulation_steps
        # 设置评估延迟
        self.eval_delay = delay
        # 设置是否只计算损失
        self.prediction_loss_only = loss_only
        # 设置是否启用JIT模式用于评估
        self.jit_mode_eval = jit_mode
        # 返回当前对象的引用
        return self
    ):
        """
        A method that regroups all basic arguments linked to testing on a held-out dataset.

        <Tip>

        Calling this method will automatically set `self.do_predict` to `True`.

        </Tip>

        Args:
            batch_size (`int` *optional*, defaults to 8):
                The batch size per device (GPU/TPU core/CPU...) used for testing.
            loss_only (`bool`, *optional*, defaults to `False`):
                Ignores all outputs except the loss.
            jit_mode (`bool`, *optional*):
                Whether or not to use PyTorch jit trace for inference.

        Example:

        ```
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_testing(batch_size=32)
        >>> args.per_device_eval_batch_size
        32
        ```
        """
        # 将self.do_predict设置为True，表示在调用此方法后进行预测
        self.do_predict = True
        # 设置每个设备（GPU/TPU核心/CPU...）用于测试的批处理大小
        self.per_device_eval_batch_size = batch_size
        # 设置是否仅计算预测损失，忽略所有其他输出
        self.prediction_loss_only = loss_only
        # 设置是否使用PyTorch的jit追踪进行推断
        self.jit_mode_eval = jit_mode
        # 返回设置后的对象本身，以支持方法链式调用
        return self
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

        ```
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_save(strategy="steps", steps=100)
        >>> args.save_steps
        100
        ```
        """
        self.save_strategy = IntervalStrategy(strategy)
        # 设置保存策略为指定的策略类型
        if self.save_strategy == IntervalStrategy.STEPS and steps == 0:
            raise ValueError("Setting `strategy` as 'steps' requires a positive value for `steps`.")
        # 如果保存策略为步数，并且步数设置为0，则抛出数值错误
        self.save_steps = steps
        # 设置保存步数
        self.save_total_limit = total_limit
        # 设置总共保存的最大数量限制
        self.save_on_each_node = on_each_node
        # 设置是否在每个节点上保存模型和检查点
        return self
        # 返回当前实例化对象，以便支持链式调用
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
    ):
        """
        A method that regroups all arguments linked to the optimizer and its hyperparameters.

        Args:
            name (`str` or [`training_args.OptimizerNames`], *optional*, defaults to `"adamw_torch"`):
                The optimizer to use: `"adamw_hf"`, `"adamw_torch"`, `"adamw_torch_fused"`, `"adamw_apex_fused"`,
                `"adamw_anyprecision"` or `"adafactor"`.
            learning_rate (`float`, *optional*, defaults to 5e-5):
                The initial learning rate.
            weight_decay (`float`, *optional*, defaults to 0):
                The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights.
            beta1 (`float`, *optional*, defaults to 0.9):
                The beta1 hyperparameter for the adam optimizer or its variants.
            beta2 (`float`, *optional*, defaults to 0.999):
                The beta2 hyperparameter for the adam optimizer or its variants.
            epsilon (`float`, *optional*, defaults to 1e-8):
                The epsilon hyperparameter for the adam optimizer or its variants.
            args (`str`, *optional*):
                Optional arguments that are supplied to AnyPrecisionAdamW (only useful when
                `optim="adamw_anyprecision"`).

        Example:

        ```
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_optimizer(name="adamw_torch", beta1=0.8)
        >>> args.optim
        'adamw_torch'
        ```
        """
        # 设置优化器名称，将输入的名称转换为 OptimizerNames 对象
        self.optim = OptimizerNames(name)
        # 设置初始学习率
        self.learning_rate = learning_rate
        # 设置权重衰减率，应用于除所有偏置和 LayerNorm 权重以外的所有层
        self.weight_decay = weight_decay
        # 设置 adam 优化器及其变体的 beta1 参数
        self.adam_beta1 = beta1
        # 设置 adam 优化器及其变体的 beta2 参数
        self.adam_beta2 = beta2
        # 设置 adam 优化器及其变体的 epsilon 参数
        self.adam_epsilon = epsilon
        # 设置优化器的额外参数
        self.optim_args = args
        # 返回当前对象，以支持方法链调用
        return self
    ):
        """
        A method that regroups all arguments linked to the learning rate scheduler and its hyperparameters.

        Args:
            name (`str` or [`SchedulerType`], *optional*, defaults to `"linear"`):
                The scheduler type to use. See the documentation of [`SchedulerType`] for all possible values.
            num_epochs(`float`, *optional*, defaults to 3.0):
                Total number of training epochs to perform (if not an integer, will perform the decimal part percents
                of the last epoch before stopping training).
            max_steps (`int`, *optional*, defaults to -1):
                If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`.
                For a finite dataset, training is reiterated through the dataset (if all data is exhausted) until
                `max_steps` is reached.
            warmup_ratio (`float`, *optional*, defaults to 0.0):
                Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.
            warmup_steps (`int`, *optional*, defaults to 0):
                Number of steps used for a linear warmup from 0 to `learning_rate`. Overrides any effect of
                `warmup_ratio`.

        Example:

        ```
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_lr_scheduler(name="cosine", warmup_ratio=0.05)
        >>> args.warmup_ratio
        0.05
        ```
        """
        # 设置学习率调度器类型
        self.lr_scheduler_type = SchedulerType(name)
        # 设置训练的总轮次
        self.num_train_epochs = num_epochs
        # 设置最大训练步数
        self.max_steps = max_steps
        # 设置线性预热的比例
        self.warmup_ratio = warmup_ratio
        # 设置线性预热的步数
        self.warmup_steps = warmup_steps
        # 返回设置后的对象本身
        return self

    def set_dataloader(
        self,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        drop_last: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
        auto_find_batch_size: bool = False,
        ignore_data_skip: bool = False,
        sampler_seed: Optional[int] = None,
# 定义一个枚举类 ParallelMode，用于表示并行计算模式的选项
class ParallelMode(Enum):
    # 表示非并行模式
    NOT_PARALLEL = "not_parallel"
    # 表示非分布式模式
    NOT_DISTRIBUTED = "not_distributed"
    # 表示分布式模式
    DISTRIBUTED = "distributed"
    # 表示使用Sagemaker的模型并行计算模式
    SAGEMAKER_MODEL_PARALLEL = "sagemaker_model_parallel"
    # 表示使用Sagemaker的数据并行计算模式
    SAGEMAKER_DATA_PARALLEL = "sagemaker_data_parallel"
    # 表示使用TPU进行计算
    TPU = "tpu"
```