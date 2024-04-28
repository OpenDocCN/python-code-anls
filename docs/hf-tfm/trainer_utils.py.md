# `.\transformers\trainer_utils.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本使用此文件，除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制
"""
# 用于 Trainer 类的独立于 PyTorch 的实用程序
"""

# 导入所需的库
import copy
import functools
import gc
import inspect
import os
import random
import re
import threading
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np

# 导入自定义的工具函数
from .utils import (
    ExplicitEnum,
    is_psutil_available,
    is_tf_available,
    is_torch_available,
    is_torch_cuda_available,
    is_torch_mps_available,
    is_torch_npu_available,
    is_torch_tpu_available,
    is_torch_xpu_available,
    requires_backends,
)

# 如果 PyTorch 可用，则导入 torch 库
if is_torch_available():
    import torch

# 定义一个函数，用于设置 worker 的种子
def seed_worker(_):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    # 计算 worker 的种子
    worker_seed = torch.initial_seed() % 2**32
    # 调用 set_seed 函数设置种子
    set_seed(worker_seed)

# 定义一个函数，用于启用完全确定性
def enable_full_determinism(seed: int, warn_only: bool = False):
    """
    Helper function for reproducible behavior during distributed training. See
    - https://pytorch.org/docs/stable/notes/randomness.html for pytorch
    - https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism for tensorflow
    """
    # 首先设置种子
    set_seed(seed)

    # 如果 PyTorch 可用
    if is_torch_available():
        # 启用 PyTorch 的确定性模式
        # 这可能需要设置环境变量 'CUDA_LAUNCH_BLOCKING' 或 'CUBLAS_WORKSPACE_CONFIG'，
        # 取决于 CUDA 版本，因此在这里同时设置它们
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True, warn_only=warn_only)

        # 启用 CUDNN 的确定性模式
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 如果 TensorFlow 可用
    if is_tf_available():
        import tensorflow as tf

        # 启用 TensorFlow 的操作确定性
        tf.config.experimental.enable_op_determinism()

# 定义一个函数，用于设置种子
def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    # 设置随机数种子
    random.seed(seed)
    np.random.seed(seed)
    # 如果 PyTorch 可用
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ 即使 cuda 不可用也可以安全调用此函数
    # 如果 PyTorch NPU 可用
    if is_torch_npu_available():
        torch.npu.manual_seed_all(seed)
    # 如果存在可用的 Torch XPU（加速处理单元），则设置所有 XPU 的随机种子为给定的种子值
    if is_torch_xpu_available():
        torch.xpu.manual_seed_all(seed)
    # 如果 TensorFlow 可用，则导入 TensorFlow 模块，并设置 TensorFlow 的随机种子为给定的种子值
    if is_tf_available():
        import tensorflow as tf
        tf.random.set_seed(seed)
def neftune_post_forward_hook(module, input, output):
    """
    Implements the NEFTune forward pass for the model using forward hooks. Note this works only for torch.nn.Embedding
    layers. This method is slightly adapted from the original source code that can be found here:
    https://github.com/neelsjain/NEFTune Simply add it to your model as follows:
    ```python
    model = ...
    model.embed_tokens.neftune_noise_alpha = 0.1
    model.embed_tokens.register_forward_hook(neftune_post_forward_hook)
    ```
    Args:
        module (`torch.nn.Module`):
            The embedding module where the hook is attached. Note that you need to set `module.neftune_noise_alpha` to
            the desired noise alpha value.
        input (`torch.Tensor`):
            The input tensor to the model.
        output (`torch.Tensor`):
            The output tensor of the model (i.e. the embeddings).
    """
    # 检查模型是否处于训练模式
    if module.training:
        # 计算输出张量的维度乘积
        dims = torch.tensor(output.size(1) * output.size(2))
        # 计算噪声幅度的归一化值
        mag_norm = module.neftune_noise_alpha / torch.sqrt(dims)
        # 对输出张量添加服从均匀分布的噪声
        output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
    # 返回处理后的输出张量
    return output


class EvalPrediction:
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
        inputs (`np.ndarray`, *optional*):
    """

    def __init__(
        self,
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        label_ids: Union[np.ndarray, Tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
    ):
        # 初始化对象实例的预测值、标签和输入数据
        self.predictions = predictions
        self.label_ids = label_ids
        self.inputs = inputs

    def __iter__(self):
        # 如果输入数据不为空，则返回预测值、标签和输入数据的迭代器
        if self.inputs is not None:
            return iter((self.predictions, self.label_ids, self.inputs))
        # 如果输入数据为空，则返回预测值和标签的迭代器
        else:
            return iter((self.predictions, self.label_ids))

    def __getitem__(self, idx):
        # 检查索引是否在范围内
        if idx < 0 or idx > 2:
            raise IndexError("tuple index out of range")
        # 如果索引为2且输入数据为空，则抛出索引超出范围的错误
        if idx == 2 and self.inputs is None:
            raise IndexError("tuple index out of range")
        # 根据索引返回相应的值
        if idx == 0:
            return self.predictions
        elif idx == 1:
            return self.label_ids
        elif idx == 2:
            return self.inputs


class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]


class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float
    metrics: Dict[str, float]


PREFIX_CHECKPOINT_DIR = "checkpoint"
# 编译正则表达式，用于匹配检查点目录的格式
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")


# 获取给定文件夹中最新的检查点路径
def get_last_checkpoint(folder):
    # 列出文件夹中的内容
    content = os.listdir(folder)
    # 从内容中筛选出所有符合检查点格式的路径
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    # 如果没有找到符合格式的检查点路径，则返回 None
    if len(checkpoints) == 0:
        return
    # 返回最新的检查点路径
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


# 枚举类：间隔策略
class IntervalStrategy(ExplicitEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


# 枚举类：评估策略
class EvaluationStrategy(ExplicitEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


# 枚举类：Hub 策略
class HubStrategy(ExplicitEnum):
    END = "end"
    EVERY_SAVE = "every_save"
    CHECKPOINT = "checkpoint"
    ALL_CHECKPOINTS = "all_checkpoints"


# 命名元组：最佳运行
class BestRun(NamedTuple):
    """
    The best run found by a hyperparameter search (see [`~Trainer.hyperparameter_search`]).

    Parameters:
        run_id (`str`):
            The id of the best run (if models were saved, the corresponding checkpoint will be in the folder ending
            with run-{run_id}).
        objective (`float`):
            The objective that was obtained for this run.
        hyperparameters (`Dict[str, Any]`):
            The hyperparameters picked to get this run.
        run_summary (`Optional[Any]`):
            A summary of tuning experiments. `ray.tune.ExperimentAnalysis` object for Ray backend.
    """

    run_id: str
    objective: Union[float, List[float]]
    hyperparameters: Dict[str, Any]
    run_summary: Optional[Any] = None


# 默认计算目标函数
def default_compute_objective(metrics: Dict[str, float]) -> float:
    """
    The default objective to maximize/minimize when doing an hyperparameter search. It is the evaluation loss if no
    metrics are provided to the [`Trainer`], the sum of all metrics otherwise.

    Args:
        metrics (`Dict[str, float]`): The metrics returned by the evaluate method.

    Return:
        `float`: The objective to minimize or maximize
    """
    # 深拷贝度量
    metrics = copy.deepcopy(metrics)
    # 提取评估损失
    loss = metrics.pop("eval_loss", None)
    _ = metrics.pop("epoch", None)
    # 删除速度度量
    speed_metrics = [
        m
        for m in metrics.keys()
        if m.endswith("_runtime") or m.endswith("_per_second") or m.endswith("_compilation_time")
    ]
    for sm in speed_metrics:
        _ = metrics.pop(sm, None)
    # 如果没有其他度量，则返回评估损失；否则返回所有度量的总和
    return loss if len(metrics) == 0 else sum(metrics.values())


# 默认 Optuna 超参数空间
def default_hp_space_optuna(trial) -> Dict[str, float]:
    from .integrations import is_optuna_available

    # 确保已安装 Optuna
    assert is_optuna_available(), "This function needs Optuna installed: `pip install optuna`"
    # 返回默认的超参数空间字典
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
        "seed": trial.suggest_int("seed", 1, 40),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32, 64]),
    }
```  
    }



# 结束当前的代码块，该代码块可能是函数、循环、条件语句或其他代码结构的一部分，对应着一个开放的花括号
# 定义一个函数，用于生成默认的超参数空间，以供 Ray 进行超参数优化
def default_hp_space_ray(trial) -> Dict[str, float]:
    # 导入检查 Ray Tune 是否可用的函数
    from .integrations import is_ray_tune_available

    # 检查是否安装了 Ray Tune，如果未安装，则抛出 AssertionError
    assert is_ray_tune_available(), "This function needs ray installed: `pip install ray[tune]`"
    # 导入 Ray Tune 模块
    from ray import tune

    # 返回超参数空间的字典
    return {
        "learning_rate": tune.loguniform(1e-6, 1e-4),  # 学习率的取值范围为对数均匀分布
        "num_train_epochs": tune.choice(list(range(1, 6))),  # 训练轮数的取值范围为 1 到 5 之间的整数
        "seed": tune.uniform(1, 40),  # 随机种子的取值范围为 1 到 40 之间的均匀分布
        "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),  # 单设备训练批量大小的取值范围为给定的选择列表中
    }


# 定义一个函数，用于生成默认的超参数空间，以供 SigOpt 进行超参数优化
def default_hp_space_sigopt(trial):
    # 返回超参数空间的列表
    return [
        {"bounds": {"min": 1e-6, "max": 1e-4}, "name": "learning_rate", "type": "double", "transformamtion": "log"},
        # 学习率的取值范围为对数变换的 double 类型
        {"bounds": {"min": 1, "max": 6}, "name": "num_train_epochs", "type": "int"},  # 训练轮数的取值范围为整数
        {"bounds": {"min": 1, "max": 40}, "name": "seed", "type": "int"},  # 随机种子的取值范围为整数
        {
            "categorical_values": ["4", "8", "16", "32", "64"],
            "name": "per_device_train_batch_size",
            "type": "categorical",  # 单设备训练批量大小的取值范围为给定的类别值列表中
        },
    ]


# 定义一个函数，用于生成默认的超参数空间，以供 Wandb 进行超参数优化
def default_hp_space_wandb(trial) -> Dict[str, float]:
    # 导入检查 Wandb 是否可用的函数
    from .integrations import is_wandb_available

    # 如果 Wandb 不可用，则抛出 ImportError
    if not is_wandb_available():
        raise ImportError("This function needs wandb installed: `pip install wandb`")

    # 返回超参数空间的字典
    return {
        "method": "random",  # 采用随机搜索方法
        "metric": {"name": "objective", "goal": "minimize"},  # 优化目标为最小化
        "parameters": {
            "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-4},  # 学习率的取值范围为均匀分布
            "num_train_epochs": {"distribution": "int_uniform", "min": 1, "max": 6},  # 训练轮数的取值范围为整数均匀分布
            "seed": {"distribution": "int_uniform", "min": 1, "max": 40},  # 随机种子的取值范围为整数均匀分布
            "per_device_train_batch_size": {"values": [4, 8, 16, 32, 64]},  # 单设备训练批量大小的取值范围为给定的值列表中
        },
    }


# 定义一个枚举类，表示超参数搜索后端
class HPSearchBackend(ExplicitEnum):
    OPTUNA = "optuna"
    RAY = "ray"
    SIGOPT = "sigopt"
    WANDB = "wandb"


# 定义一个函数，判断当前进程是否为主进程
def is_main_process(local_rank):
    """
    Whether or not the current process is the local process, based on `xm.get_ordinal()` (for TPUs) first, then on
    `local_rank`.
    """
    # 如果是在 TPU 上，则通过 `xm.get_ordinal()` 判断是否为主进程
    if is_torch_tpu_available(check_device=True):
        import torch_xla.core.xla_model as xm

        return xm.get_ordinal() == 0
    # 否则，根据 `local_rank` 判断是否为主进程
    return local_rank in [-1, 0]


# 定义一个函数，返回并行启动的进程数
def total_processes_number(local_rank):
    """
    Return the number of processes launched in parallel. Works with `torch.distributed` and TPUs.
    """
    # 如果是在 TPU 上，则返回并行启动的进程数
    if is_torch_tpu_available(check_device=True):
        import torch_xla.core.xla_model as xm

        return xm.xrt_world_size()
    # 如果 `local_rank` 不为 -1 且可用 PyTorch，则返回并行启动的进程数
    elif local_rank != -1 and is_torch_available():
        import torch

        return torch.distributed.get_world_size()
    # 否则，返回默认的并行启动的进程数为 1
    return 1


# 定义一个函数，用于测量和返回速度性能指标
def speed_metrics(split, start_time, num_samples=None, num_steps=None, num_tokens=None):
    """
    Measure and return speed performance metrics.

    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.

    Args:
    - split: name to prefix metric (like train, eval, test...)
    """
    # 计算操作的运行时间
    runtime = time.time() - start_time
    # 初始化结果字典，包含操作运行时间的信息
    result = {f"{split}_runtime": round(runtime, 4)}
    # 若运行时间为0，直接返回结果字典
    if runtime == 0:
        return result
    # 如果提供了样本数，则计算每秒处理的样本数，并添加到结果字典中
    if num_samples is not None:
        samples_per_second = num_samples / runtime
        result[f"{split}_samples_per_second"] = round(samples_per_second, 3)
    # 如果提供了步数，则计算每秒处理的步数，并添加到结果字典中
    if num_steps is not None:
        steps_per_second = num_steps / runtime
        result[f"{split}_steps_per_second"] = round(steps_per_second, 3)
    # 如果提供了标记数，则计算每秒处理的标记数，并添加到结果字典中
    if num_tokens is not None:
        tokens_per_second = num_tokens / runtime
        result[f"{split}_tokens_per_second"] = round(tokens_per_second, 3)
    # 返回结果字典
    return result
class SchedulerType(ExplicitEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    INVERSE_SQRT = "inverse_sqrt"
    REDUCE_ON_PLATEAU = "reduce_lr_on_plateau"

class TrainerMemoryTracker:
    """
    A helper class that tracks cpu and gpu memory.

    This class will silently skip unless `psutil` is available. Install with `pip install psutil`.

    When a stage completes, it can pass metrics dict to update with the memory metrics gathered during this stage.

    Example :

    ```python
    self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
    self._memory_tracker.start()
    # code ...
    metrics = {"train_runtime": 10.5}
    self._memory_tracker.stop_and_update_metrics(metrics)
    ```

    At the moment GPU tracking is only for `pytorch`, but can be extended to support `tensorflow`.

    To understand this class' intricacies please read the documentation of [`~Trainer.log_metrics`].
    """

    # map trainer methods to metrics prefix
    stages = {
        "__init__": "init",
        "train": "train",
        "_inner_training_loop": "train",
        "evaluate": "eval",
        "predict": "test",
    }

    def __init__(self, skip_memory_metrics=False):
        # 初始化函数，用于设置跟踪内存的参数
        self.skip_memory_metrics = skip_memory_metrics

        if not is_psutil_available():
            # 如果没有安装 psutil 库，则跳过内存跟踪
            self.skip_memory_metrics = True

        if self.skip_memory_metrics:
            return

        import psutil  # noqa

        if is_torch_cuda_available():
            import torch

            self.torch = torch
            self.gpu = {}
        elif is_torch_mps_available():
            import torch

            self.torch = torch
            self.gpu = {}
        elif is_torch_xpu_available():
            import torch

            self.torch = torch
            self.gpu = {}
        elif is_torch_npu_available():
            import torch

            self.torch = torch
            self.gpu = {}
        else:
            self.torch = None

        self.process = psutil.Process()

        self.cur_stage = None
        self.cpu = {}
        self.init_reported = False

    def derive_stage(self):
        """derives the stage/caller name automatically"""
        # 推导当前阶段/调用者的名称
        caller = inspect.currentframe().f_back.f_back.f_code.co_name
        if caller in self.stages:
            return self.stages[caller]
        else:
            raise ValueError(
                f"was called from {caller}, but only expect to be called from one of {self.stages.keys()}"
            )

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        # 获取当前进程的驻留集大小内存
        return self.process.memory_info().rss
    def peak_monitor_func(self):
        # 初始化 CPU 和内存使用峰值为 -1
        self.cpu_mem_used_peak = -1

        # 循环监视内存使用峰值
        while True:
            # 更新 CPU 和内存使用峰值
            self.cpu_mem_used_peak = max(self.cpu_mem_used(), self.cpu_mem_used_peak)

            # 无法睡眠，否则将无法捕获到正确的内存峰值（此注释是有意放在这里的）
            # time.sleep(0.001) # 1msec

            # 如果不再监视内存使用峰值，则退出循环
            if not self.peak_monitoring:
                break

    # 启动内存追踪
    def start(self):
        """start tracking for the caller's stage"""
        # 如果跳过内存指标，则直接返回
        if self.skip_memory_metrics:
            return

        # 推导当前阶段
        stage = self.derive_stage()
        # 处理在训练期间 eval 的嵌套调用 - 简单地忽略这些
        if self.cur_stage is not None and self.cur_stage != stage:
            return

        # 设置当前阶段
        self.cur_stage = stage

        # 收集垃圾
        gc.collect()

        # 如果存在 torch
        if self.torch is not None:
            # 如果 CUDA 可用
            if torch.cuda.is_available():
                # 重置 CUDA 内存峰值统计信息并清空缓存
                self.torch.cuda.reset_peak_memory_stats()
                self.torch.cuda.empty_cache()
            # 如果存在 XPU
            elif is_torch_xpu_available():
                # 重置 XPU 内存峰值统计信息并清空缓存
                self.torch.xpu.reset_peak_memory_stats()
                self.torch.xpu.empty_cache()
            # 如果存在 NPU
            elif is_torch_npu_available():
                # 重置 NPU 内存峰值统计信息并清空缓存
                self.torch.npu.reset_peak_memory_stats()
                self.torch.npu.empty_cache()

        # GPU
        # 如果存在 torch
        if self.torch is not None:
            # 如果 CUDA 可用
            if torch.cuda.is_available():
                # 记录 GPU 在启动时的内存使用量
                self.gpu_mem_used_at_start = self.torch.cuda.memory_allocated()
            # 如果存在 XPU
            elif is_torch_xpu_available():
                # 记录 GPU 在启动时的内存使用量
                self.gpu_mem_used_at_start = self.torch.xpu.memory_allocated()
            # 如果存在 NPU
            elif is_torch_npu_available():
                # 记录 GPU 在启动时的内存使用量
                self.gpu_mem_used_at_start = self.torch.npu.memory_allocated()

        # CPU
        # 记录 CPU 在启动时的内存使用量
        self.cpu_mem_used_at_start = self.cpu_mem_used()

        # 开始监视内存使用峰值
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
    # 停止跟踪指定阶段的内存使用情况

    # 处理在训练期间 eval 的嵌套调用 - 简单地忽略这些情况
    if self.cur_stage is not None and self.cur_stage != stage:
        return

    # 发送信号给 peak_monitor_func 完成其循环
    self.peak_monitoring = False

    # 首先确保所有对象被收集并且它们的内存被释放
    gc.collect()

    if self.torch is not None:
        if torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        elif is_torch_xpu_available():
            self.torch.xpu.empty_cache()
        elif is_torch_npu_available():
            self.torch.npu.empty_cache()

    # 概念:
    # - alloc_delta: 分配内存的差值，即结束时与开始时的内存差
    # - peaked_delta: 峰值内存与当前内存的差值
    # 为了知道测量代码消耗了多少内存，需要将这两者相加

    # GPU
    if self.torch is not None:
        if torch.cuda.is_available():
            self.gpu_mem_used_now = self.torch.cuda.memory_allocated()
            self.gpu_mem_used_peak = self.torch.cuda.max_memory_allocated()
        elif is_torch_xpu_available():
            self.gpu_mem_used_now = self.torch.xpu.memory_allocated()
            self.gpu_mem_used_peak = self.torch.xpu.max_memory_allocated()
        elif is_torch_npu_available():
            self.gpu_mem_used_now = self.torch.npu.memory_allocated()
            self.gpu_mem_used_peak = self.torch.npu.max_memory_allocated()
        else:
            raise ValueError("No available GPU device found!")

        self.gpu[self.cur_stage] = {
            "begin": self.gpu_mem_used_at_start,
            "end": self.gpu_mem_used_now,
            "alloc": (self.gpu_mem_used_now - self.gpu_mem_used_at_start),
            "peaked": max(0, self.gpu_mem_used_peak - self.gpu_mem_used_now),
        }

    # CPU
    self.cpu_mem_used_now = self.cpu_mem_used()
    self.cpu[self.cur_stage] = {
        "begin": self.cpu_mem_used_at_start,
        "end": self.cpu_mem_used_now,
        "alloc": (self.cpu_mem_used_now - self.cpu_mem_used_at_start),
        "peaked": max(0, self.cpu_mem_used_peak - self.cpu_mem_used_now),
    }

    # 重置 - 循环结束
    self.cur_stage = None
    # 更新指标信息
    def update_metrics(self, stage, metrics):
        """updates the metrics"""
        # 如果跳过内存指标，直接返回
        if self.skip_memory_metrics:
            return

        # 处理在训练过程中 eval 的嵌套调用 - 简单地忽略这些情况
        if self.cur_stage is not None and self.cur_stage != stage:
            return

        # 如果没有返回初始指标，将其放入第一个训练/验证/预测中
        stages = [stage]
        if not self.init_reported:
            stages.insert(0, "init")
            self.init_reported = True

        # 遍历阶段
        for stage in stages:
            # 遍历不同类型的内存指标
            for t in ["alloc", "peaked"]:
                # 如果 CPU 内存指标可用，则添加到指标字典中
                if stage in self.cpu and t in self.cpu[stage]:
                    metrics[f"{stage}_mem_cpu_{t}_delta"] = self.cpu[stage][t]
                # 如果 GPU 内存指标可用，并且 torch 可用，则添加到指标字典中
                if self.torch is not None and stage in self.gpu and t in self.gpu[stage]:
                    metrics[f"{stage}_mem_gpu_{t}_delta"] = self.gpu[stage][t]
            # 如果需要额外的调试信息，启用下面的代码
            # for t in ["begin", "end"]:
            #     if stage in self.cpu and t in self.cpu[stage]:
            #         metrics[f"{stage}_mem_cpu_{t}"] = self.cpu[stage][t]
            #     if self.torch is not None and stage in self.gpu and t in self.gpu[stage]:
            #         metrics[f"{stage}_mem_gpu_{t}"] = self.gpu[stage][t]

        # 由于内存可能在初始化之前分配，并且跟踪整体内存使用情况可能很困难，尤其是对于 GPU，所以在调用 init 时报告内存使用情况
        if stages[0] == "init":
            # 报告初始化时的 CPU 内存使用情况
            metrics["before_init_mem_cpu"] = self.cpu["init"]["begin"]
            # 如果 torch 可用，报告初始化时的 GPU 内存使用情况
            if self.torch is not None:
                metrics["before_init_mem_gpu"] = self.gpu["init"]["begin"]
            # 如果我们还想报告初始化和下一个阶段之间的任何额外内存分配，我们也可以这样报告
            # if self.cpu["init"]["end"] != self.cpu[stage]["begin"]:
            #     metrics[f"after_init_mem_cpu_delta"] = self.cpu[stage]["begin"] - self.cpu["init"]["end"]
            # if self.torch is not None and self.gpu["init"]["end"] != self.gpu[stage]["begin"]:
            #     metrics[f"after_init_mem_gpu_delta"] = self.gpu[stage]["begin"] - self.gpu["init"]["end"]

    # 停止并更新指标信息，合并为一次调用以简化代码
    def stop_and_update_metrics(self, metrics=None):
        """combine stop and metrics update in one call for simpler code"""
        # 如果跳过内存指标，直接返回
        if self.skip_memory_metrics:
            return

        # 推断当前阶段
        stage = self.derive_stage()
        # 停止当前阶段
        self.stop(stage)

        # 初始化没有要更新的指标，所以我们只保存数据以供后续阶段检索
        if metrics is not None:
            # 更新指标信息
            self.update_metrics(stage, metrics)
# 检查数据集是否实现了 __len__() 方法，并且调用该方法不会引发错误
def has_length(dataset):
    try:
        # 返回数据集的长度是否不为 None
        return len(dataset) is not None
    except TypeError:
        # 捕获 TypeError 异常，表示数据集为不可计算长度的对象
        # 返回 False
        return False


# 递归调用字典中元素的 `.item()` 方法
def denumpify_detensorize(metrics):
    if isinstance(metrics, (list, tuple)):
        # 如果 metrics 是列表或元组，则递归调用 denumpify_detensorize 函数
        return type(metrics)(denumpify_detensorize(m) for m in metrics)
    elif isinstance(metrics, dict):
        # 如果 metrics 是字典，则递归调用 denumpify_detensorize 函数
        return type(metrics)({k: denumpify_detensorize(v) for k, v in metrics.items()})
    elif isinstance(metrics, np.generic):
        # 如果 metrics 是 numpy 的标量类型，则返回其 item() 方法的结果
        return metrics.item()
    elif is_torch_available() and isinstance(metrics, torch.Tensor) and metrics.numel() == 1:
        # 如果 torch 可用且 metrics 是 torch.Tensor 类型且元素个数为 1，则返回其 item() 方法的结果
        return metrics.item()
    # 其他情况直接返回 metrics
    return metrics


# 返回传入函数的参数个数，即使是 partial 函数
def number_of_arguments(func):
    if isinstance(func, functools.partial):
        # 如果 func 是 partial 函数，则计算其实际参数个数
        total_args = len(inspect.signature(func.func).parameters)
        return total_args - len(func.args) - len(func.keywords)
    # 如果不是 partial 函数，则直接返回其参数个数
    return len(inspect.signature(func).parameters)


# 查找可执行的批量大小
def find_executable_batch_size(
    function: callable = None, starting_batch_size: int = 128, auto_find_batch_size: bool = False
):
    if function is None:
        # 如果 function 为 None，则返回一个带有默认参数的 partial 函数
        return functools.partial(
            find_executable_batch_size,
            starting_batch_size=starting_batch_size,
            auto_find_batch_size=auto_find_batch_size,
        )

    if auto_find_batch_size:
        # 如果 auto_find_batch_size 为 True，则调用 accelerate 包中的函数
        requires_backends(find_executable_batch_size, "accelerate")
        from accelerate.utils import find_executable_batch_size as accelerate_find_executable_batch_size

        return accelerate_find_executable_batch_size(function=function, starting_batch_size=starting_batch_size)

    # 如果 auto_find_batch_size 为 False，则返回一个带有默认批量大小参数的 partial 函数
    return functools.partial(function, batch_size=starting_batch_size)


# 枚举类，定义了一些 FSDP 选项
class FSDPOption(ExplicitEnum):
    FULL_SHARD = "full_shard"
    SHARD_GRAD_OP = "shard_grad_op"
    NO_SHARD = "no_shard"
    HYBRID_SHARD = "hybrid_shard"
    HYBRID_SHARD_ZERO2 = "hybrid_shard_zero2"
    OFFLOAD = "offload"
    AUTO_WRAP = "auto_wrap"


# 数据收集器，用于在传递给收集器之前移除未使用的列
class RemoveColumnsCollator:
    """Wrap the data collator to remove unused columns before they are passed to the collator."""
    # 初始化方法，用于实例化对象时进行初始化操作
    def __init__(
        self,
        data_collator,                           # 数据收集器，用于收集数据
        signature_columns,                      # 签名列，即模型所期望的输入特征列
        logger=None,                            # 日志记录器，可选参数，默认为 None
        model_name: Optional[str] = None,       # 模型名称，可选参数，默认为 None
        description: Optional[str] = None,      # 描述信息，可选参数，默认为 None
    ):
        # 初始化数据收集器
        self.data_collator = data_collator
        # 初始化签名列
        self.signature_columns = signature_columns
        # 初始化日志记录器
        self.logger = logger
        # 初始化描述信息
        self.description = description
        # 初始化模型名称
        self.model_name = model_name
        # 初始化消息记录标志，默认为 False
        self.message_logged = False

    # 私有方法，用于移除特征中的不需要的列
    def _remove_columns(self, feature: dict) -> dict:
        # 如果特征不是字典类型，则直接返回特征
        if not isinstance(feature, dict):
            return feature
        # 如果消息未记录且日志记录器和模型名称都存在
        if not self.message_logged and self.logger and self.model_name:
            # 计算被忽略的列，即特征中存在但模型不期望的列
            ignored_columns = list(set(feature.keys()) - set(self.signature_columns))
            # 如果有被忽略的列
            if len(ignored_columns) > 0:
                # 构建数据集描述信息
                dset_description = "" if self.description is None else f"in the {self.description} set"
                # 记录消息到日志中
                self.logger.info(
                    f"The following columns {dset_description} don't have a corresponding argument in "
                    f"`{self.model_name}.forward` and have been ignored: {', '.join(ignored_columns)}."
                    f" If {', '.join(ignored_columns)} are not expected by `{self.model_name}.forward`, "
                    " you can safely ignore this message."
                )
                # 将消息记录标志设置为 True，表示已记录消息
                self.message_logged = True
        # 返回仅包含签名列的特征字典
        return {k: v for k, v in feature.items() if k in self.signature_columns}

    # 调用对象时执行的方法，用于处理特征列表
    def __call__(self, features: List[dict]):
        # 对特征列表中的每个特征应用 _remove_columns 方法，移除不需要的列
        features = [self._remove_columns(feature) for feature in features]
        # 使用数据收集器对处理后的特征列表进行处理，并返回结果
        return self.data_collator(features)
```