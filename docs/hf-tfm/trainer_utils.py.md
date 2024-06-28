# `.\trainer_utils.py`

```
# 设置文件编码为 UTF-8

# 版权声明和许可证信息

"""
PyTorch-独立工具类，为 Trainer 类提供支持。
"""

# 导入标准库和第三方库
import copy  # 导入 copy 模块，用于复制对象
import functools  # 导入 functools 模块，用于高阶函数（higher-order functions）
import gc  # 导入 gc 模块，用于垃圾回收
import inspect  # 导入 inspect 模块，用于检查对象
import os  # 导入 os 模块，用于操作系统相关功能
import random  # 导入 random 模块，用于生成随机数
import re  # 导入 re 模块，用于正则表达式操作
import threading  # 导入 threading 模块，用于线程支持
import time  # 导入 time 模块，用于时间操作

from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union  # 导入类型提示相关的类和函数

import numpy as np  # 导入 NumPy 库

# 导入自定义模块和函数
from .utils import (
    ExplicitEnum,  # 导入 ExplicitEnum 枚举类
    is_psutil_available,  # 导入检查 psutil 库是否可用的函数
    is_tf_available,  # 导入检查 TensorFlow 是否可用的函数
    is_torch_available,  # 导入检查 PyTorch 是否可用的函数
    is_torch_cuda_available,  # 导入检查 PyTorch CUDA 是否可用的函数
    is_torch_mps_available,  # 导入检查 PyTorch MPS 是否可用的函数
    is_torch_npu_available,  # 导入检查 PyTorch NPU 是否可用的函数
    is_torch_xla_available,  # 导入检查 PyTorch XLA 是否可用的函数
    is_torch_xpu_available,  # 导入检查 PyTorch XPU 是否可用的函数
    requires_backends,  # 导入装饰器函数 requires_backends
)


if is_torch_available():  # 如果 PyTorch 可用
    import torch  # 导入 PyTorch 库


def seed_worker(_: Any):  # 定义一个用于设置 worker 种子的辅助函数
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    worker_seed = torch.initial_seed() % 2**32  # 获取当前 PyTorch 种子并进行处理
    set_seed(worker_seed)  # 调用全局设置种子的函数


def enable_full_determinism(seed: int, warn_only: bool = False):  # 定义一个函数启用全确定性
    """
    Helper function for reproducible behavior during distributed training. See
    - https://pytorch.org/docs/stable/notes/randomness.html for pytorch
    - https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism for tensorflow
    """
    # 首先设置种子
    set_seed(seed)

    if is_torch_available():  # 如果 PyTorch 可用
        # 启用 PyTorch 的确定性模式，可能需要设置环境变量 'CUDA_LAUNCH_BLOCKING' 或 'CUBLAS_WORKSPACE_CONFIG'
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True, warn_only=warn_only)

        # 启用 CUDNN 的确定性模式
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if is_tf_available():  # 如果 TensorFlow 可用
        import tensorflow as tf  # 导入 TensorFlow 库

        tf.config.experimental.enable_op_determinism()  # 启用 TensorFlow 的确定性操作


def set_seed(seed: int):  # 定义一个设置种子的辅助函数
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)  # 设置 Python 内置随机数生成器的种子
    np.random.seed(seed)  # 设置 NumPy 库的种子
    if is_torch_available():  # 如果 PyTorch 可用
        torch.manual_seed(seed)  # 设置 PyTorch 随机数生成器的种子
        torch.cuda.manual_seed_all(seed)  # 设置所有 CUDA 设备的种子
        # ^^ 即使 CUDA 不可用也可以安全调用这个函数
    if is_torch_npu_available():  # 如果 PyTorch NPU 可用
        torch.npu.manual_seed_all(seed)  # 设置所有 NPU 设备的种子
    # 如果当前环境支持 Torch XPU（加速处理单元），则设置所有 XPU 的随机种子为指定的种子值
    if is_torch_xpu_available():
        torch.xpu.manual_seed_all(seed)
    
    # 如果当前环境支持 TensorFlow，则设置 TensorFlow 的随机种子为指定的种子值
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
    # Check if the module is in training mode
    if module.training:
        # Calculate the total number of elements in the output tensor
        dims = torch.tensor(output.size(1) * output.size(2))
        # Calculate magnitude normalization factor based on neftune_noise_alpha
        mag_norm = module.neftune_noise_alpha / torch.sqrt(dims)
        # Add uniform noise to the output tensor
        output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
    # Return the modified output tensor
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
        # Initialize with predictions, label_ids, and optional inputs
        self.predictions = predictions
        self.label_ids = label_ids
        self.inputs = inputs

    def __iter__(self):
        # Return an iterator over predictions, label_ids, and inputs if available
        if self.inputs is not None:
            return iter((self.predictions, self.label_ids, self.inputs))
        else:
            return iter((self.predictions, self.label_ids))

    def __getitem__(self, idx):
        # Return the item corresponding to the given index
        if idx < 0 or idx > 2:
            raise IndexError("tuple index out of range")
        if idx == 2 and self.inputs is None:
            raise IndexError("tuple index out of range")
        if idx == 0:
            return self.predictions
        elif idx == 1:
            return self.label_ids
        elif idx == 2:
            return self.inputs


class EvalLoopOutput(NamedTuple):
    """
    NamedTuple for evaluation loop output, containing predictions, label_ids, metrics, and num_samples.

    Attributes:
        predictions (Union[np.ndarray, Tuple[np.ndarray]]): Predictions from the model.
        label_ids (Optional[Union[np.ndarray, Tuple[np.ndarray]]]): Target labels.
        metrics (Optional[Dict[str, float]]): Metrics computed during evaluation.
        num_samples (Optional[int]): Number of samples evaluated.
    """
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]


class PredictionOutput(NamedTuple):
    """
    NamedTuple for prediction output, containing predictions, label_ids, and metrics.

    Attributes:
        predictions (Union[np.ndarray, Tuple[np.ndarray]]): Predictions from the model.
        label_ids (Optional[Union[np.ndarray, Tuple[np.ndarray]]]): Target labels.
        metrics (Optional[Dict[str, float]]): Metrics computed during prediction.
    """
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]


class TrainOutput(NamedTuple):
    """
    NamedTuple for training output, containing global_step, training_loss, and metrics.

    Attributes:
        global_step (int): Current global step of training.
        training_loss (float): Loss computed during training.
        metrics (Dict[str, float]): Metrics computed during training.
    """
    global_step: int
    training_loss: float
    metrics: Dict[str, float]


PREFIX_CHECKPOINT_DIR = "checkpoint"
# 编译正则表达式，用于匹配检查点目录名称的格式，预期格式为 PREFIX_CHECKPOINT_DIR-数字
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")


def get_last_checkpoint(folder):
    # 获取指定文件夹中的所有内容列表
    content = os.listdir(folder)
    # 筛选出是有效检查点目录的路径列表，即名称符合正则表达式要求且是目录
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    # 如果没有有效的检查点目录，则返回 None
    if len(checkpoints) == 0:
        return
    # 返回最新的检查点目录的完整路径
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


class IntervalStrategy(ExplicitEnum):
    # 定义枚举类 `IntervalStrategy`，包含 NO、STEPS 和 EPOCH 三个枚举值
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class EvaluationStrategy(ExplicitEnum):
    # 定义枚举类 `EvaluationStrategy`，包含 NO、STEPS 和 EPOCH 三个枚举值
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class HubStrategy(ExplicitEnum):
    # 定义枚举类 `HubStrategy`，包含 END、EVERY_SAVE、CHECKPOINT 和 ALL_CHECKPOINTS 四个枚举值
    END = "end"
    EVERY_SAVE = "every_save"
    CHECKPOINT = "checkpoint"
    ALL_CHECKPOINTS = "all_checkpoints"


class BestRun(NamedTuple):
    """
    通过超参数搜索找到的最佳运行结果的命名元组。

    Parameters:
        run_id (`str`):
            最佳运行的 ID （如果模型被保存，则对应的检查点将位于以 run-{run_id} 结尾的文件夹中）。
        objective (`float`):
            获得此运行的目标值。
        hyperparameters (`Dict[str, Any]`):
            用于此运行的超参数。
        run_summary (`Optional[Any]`):
            调优实验的摘要。对于 Ray 后端，为 `ray.tune.ExperimentAnalysis` 对象。
    """

    run_id: str
    objective: Union[float, List[float]]
    hyperparameters: Dict[str, Any]
    run_summary: Optional[Any] = None


def default_compute_objective(metrics: Dict[str, float]) -> float:
    """
    在进行超参数搜索时最大化/最小化的默认目标函数。如果没有提供任何指标，则为评估损失；否则为所有指标之和。

    Args:
        metrics (`Dict[str, float]`): evaluate 方法返回的指标。

    Return:
        `float`: 最小化或最大化的目标值。
    """
    # 深拷贝指标字典
    metrics = copy.deepcopy(metrics)
    # 移除评估损失指标
    loss = metrics.pop("eval_loss", None)
    _ = metrics.pop("epoch", None)
    # 移除速度相关指标
    speed_metrics = [
        m
        for m in metrics.keys()
        if m.endswith("_runtime") or m.endswith("_per_second") or m.endswith("_compilation_time")
    ]
    for sm in speed_metrics:
        _ = metrics.pop(sm, None)
    # 如果指标字典为空，则返回评估损失，否则返回所有指标之和
    return loss if len(metrics) == 0 else sum(metrics.values())


def default_hp_space_optuna(trial) -> Dict[str, float]:
    from .integrations import is_optuna_available

    # 确保 Optuna 已安装
    assert is_optuna_available(), "This function needs Optuna installed: `pip install optuna`"
    # 返回一个包含超参数的字典，用于 Optuna 的超参数搜索
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
        "seed": trial.suggest_int("seed", 1, 40),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32, 64]),
    }
    }


注释：


    # 函数定义结束
# 默认超参数空间设置函数，用于Ray调优
def default_hp_space_ray(trial) -> Dict[str, float]:
    # 检查是否安装了Ray Tune库
    from .integrations import is_ray_tune_available
    assert is_ray_tune_available(), "This function needs ray installed: `pip install ray[tune]`"
    
    # 导入Ray Tune库
    from ray import tune
    
    # 返回超参数字典
    return {
        "learning_rate": tune.loguniform(1e-6, 1e-4),  # 学习率在对数均匀分布中取值
        "num_train_epochs": tune.choice(list(range(1, 6))),  # 训练周期在1到5之间的选择
        "seed": tune.uniform(1, 40),  # 种子值在1到40之间均匀分布
        "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),  # 每设备训练批量大小的选择
    }


# 默认超参数空间设置函数，用于SigOpt调优
def default_hp_space_sigopt(trial):
    # 返回超参数列表
    return [
        {"bounds": {"min": 1e-6, "max": 1e-4}, "name": "learning_rate", "type": "double", "transformamtion": "log"},  # 学习率在对数变换下的双精度边界
        {"bounds": {"min": 1, "max": 6}, "name": "num_train_epochs", "type": "int"},  # 训练周期在1到6之间的整数边界
        {"bounds": {"min": 1, "max": 40}, "name": "seed", "type": "int"},  # 种子值在1到40之间的整数边界
        {
            "categorical_values": ["4", "8", "16", "32", "64"],  # 每设备训练批量大小的类别值列表
            "name": "per_device_train_batch_size",
            "type": "categorical",
        },
    ]


# 默认超参数空间设置函数，用于W&B调优
def default_hp_space_wandb(trial) -> Dict[str, float]:
    # 检查是否安装了W&B库
    from .integrations import is_wandb_available
    if not is_wandb_available():
        raise ImportError("This function needs wandb installed: `pip install wandb`")
    
    # 返回超参数字典
    return {
        "method": "random",
        "metric": {"name": "objective", "goal": "minimize"},  # 优化目标为最小化目标函数
        "parameters": {
            "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-4},  # 学习率在均匀分布中取值
            "num_train_epochs": {"distribution": "int_uniform", "min": 1, "max": 6},  # 训练周期在均匀整数分布中取值
            "seed": {"distribution": "int_uniform", "min": 1, "max": 40},  # 种子值在均匀整数分布中取值
            "per_device_train_batch_size": {"values": [4, 8, 16, 32, 64]},  # 每设备训练批量大小的固定值列表
        },
    }


# 超参数搜索后端类型枚举类
class HPSearchBackend(ExplicitEnum):
    OPTUNA = "optuna"
    RAY = "ray"
    SIGOPT = "sigopt"
    WANDB = "wandb"


# 是否为主进程函数
def is_main_process(local_rank):
    """
    Whether or not the current process is the local process, based on `xm.get_ordinal()` (for TPUs) first, then on
    `local_rank`.
    """
    if is_torch_xla_available():
        # 如果是在TPU上，使用torch_xla库来判断当前进程是否为主进程
        import torch_xla.core.xla_model as xm
        return xm.get_ordinal() == 0
    return local_rank in [-1, 0]


# 总进程数量函数
def total_processes_number(local_rank):
    """
    Return the number of processes launched in parallel. Works with `torch.distributed` and TPUs.
    """
    if is_torch_xla_available():
        # 如果是在TPU上，使用torch_xla库来获取并行启动的总进程数
        import torch_xla.core.xla_model as xm
        return xm.xrt_world_size()
    elif local_rank != -1 and is_torch_available():
        # 如果不是TPU且使用了torch分布式，使用torch库来获取并行启动的总进程数
        import torch
        return torch.distributed.get_world_size()
    return 1


# 速度指标函数
def speed_metrics(split, start_time, num_samples=None, num_steps=None, num_tokens=None):
    """
    Measure and return speed performance metrics.

    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.

    Args:
    - split: name to prefix metric (like train, eval, test...)
    - start_time: operation start time
    """
    # 计算代码运行时长
    runtime = time.time() - start_time
    # 初始化结果字典，存储运行时间信息
    result = {f"{split}_runtime": round(runtime, 4)}
    # 若运行时间为零，直接返回结果字典
    if runtime == 0:
        return result
    # 如果有指定的样本数量，计算每秒处理的样本数
    if num_samples is not None:
        samples_per_second = num_samples / runtime
        result[f"{split}_samples_per_second"] = round(samples_per_second, 3)
    # 如果有指定的步骤数量，计算每秒处理的步骤数
    if num_steps is not None:
        steps_per_second = num_steps / runtime
        result[f"{split}_steps_per_second"] = round(steps_per_second, 3)
    # 如果有指定的标记数量，计算每秒处理的标记数
    if num_tokens is not None:
        tokens_per_second = num_tokens / runtime
        result[f"{split}_tokens_per_second"] = round(tokens_per_second, 3)
    # 返回包含运行时间信息的结果字典
    return result
# 定义枚举类 SchedulerType，表示调度器类型，继承自 ExplicitEnum
class SchedulerType(ExplicitEnum):
    LINEAR = "linear"  # 线性调度器类型
    COSINE = "cosine"  # 余弦退火调度器类型
    COSINE_WITH_RESTARTS = "cosine_with_restarts"  # 带重启的余弦退火调度器类型
    POLYNOMIAL = "polynomial"  # 多项式调度器类型
    CONSTANT = "constant"  # 恒定调度器类型
    CONSTANT_WITH_WARMUP = "constant_with_warmup"  # 带预热的恒定调度器类型
    INVERSE_SQRT = "inverse_sqrt"  # 倒数平方根调度器类型
    REDUCE_ON_PLATEAU = "reduce_lr_on_plateau"  # 在平台上减少学习率调度器类型

# 定义 TrainerMemoryTracker 类，用于跟踪 CPU 和 GPU 内存
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

    # 将训练器方法映射到指标前缀的字典
    stages = {
        "__init__": "init",  # 初始化阶段
        "train": "train",  # 训练阶段
        "_inner_training_loop": "train",  # 内部训练循环阶段
        "evaluate": "eval",  # 评估阶段
        "predict": "test",  # 预测阶段
    }

    def __init__(self, skip_memory_metrics=False):
        self.skip_memory_metrics = skip_memory_metrics  # 是否跳过内存指标的标志

        if not is_psutil_available():
            # 如果 psutil 不可用，则跳过内存指标的收集
            self.skip_memory_metrics = True

        if self.skip_memory_metrics:
            return  # 如果跳过内存指标，则直接返回

        import psutil  # 导入 psutil 模块，用于内存和系统进程的检测

        # 根据不同的 GPU 类型检测并导入相应的 torch 模块
        if is_torch_cuda_available():
            import torch
            self.torch = torch  # 导入 torch 库
            self.gpu = {}  # 初始化 GPU 字典
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
            self.torch = None  # 如果没有可用的 GPU，将 torch 设为 None

        self.process = psutil.Process()  # 获取当前进程的 psutil 进程对象

        self.cur_stage = None  # 当前阶段名称
        self.cpu = {}  # CPU 内存字典
        self.init_reported = False  # 初始化报告标志

    def derive_stage(self):
        """自动推断阶段/调用者名称"""
        caller = inspect.currentframe().f_back.f_back.f_code.co_name  # 获取调用者函数名
        if caller in self.stages:
            return self.stages[caller]  # 返回对应的阶段名称
        else:
            raise ValueError(
                f"was called from {caller}, but only expect to be called from one of {self.stages.keys()}"
            )  # 如果调用者不在预期的阶段中，则抛出异常

    def cpu_mem_used(self):
        """获取当前进程的驻留集大小内存"""
        return self.process.memory_info().rss  # 返回当前进程的内存使用情况
    # 定义一个方法，用于监视 CPU 和内存的峰值使用情况
    def peak_monitor_func(self):
        # 初始化 CPU 和内存的峰值使用为 -1
        self.cpu_mem_used_peak = -1

        # 无限循环，持续监视峰值使用情况
        while True:
            # 更新当前的 CPU 和内存使用的峰值
            self.cpu_mem_used_peak = max(self.cpu_mem_used(), self.cpu_mem_used_peak)

            # 不能使用 sleep，否则可能无法捕获到正确的峰值（此注释是有意为之的）
            # time.sleep(0.001) # 1msec

            # 如果停止监视峰值使用，则退出循环
            if not self.peak_monitoring:
                break

    # 启动方法，开始跟踪调用者的阶段
    def start(self):
        """开始跟踪调用者的阶段"""
        # 如果设置为跳过内存指标，则直接返回
        if self.skip_memory_metrics:
            return

        # 推断当前阶段
        stage = self.derive_stage()

        # 处理在训练期间 eval 的嵌套调用 - 简单地忽略这些情况
        if self.cur_stage is not None and self.cur_stage != stage:
            return

        # 设置当前阶段
        self.cur_stage = stage

        # 执行垃圾回收
        gc.collect()

        # 如果存在 torch 对象
        if self.torch is not None:
            # 如果 CUDA 可用，重置 CUDA 的峰值内存统计并清空缓存
            if torch.cuda.is_available():
                self.torch.cuda.reset_peak_memory_stats()
                self.torch.cuda.empty_cache()
            # 如果支持 Torch XPU，重置 XPU 的峰值内存统计并清空缓存
            elif is_torch_xpu_available():
                self.torch.xpu.reset_peak_memory_stats()
                self.torch.xpu.empty_cache()
            # 如果支持 Torch NPU，重置 NPU 的峰值内存统计并清空缓存
            elif is_torch_npu_available():
                self.torch.npu.reset_peak_memory_stats()
                self.torch.npu.empty_cache()
            # 如果支持 Torch MPS，清空 MPS 的缓存
            elif is_torch_mps_available():
                self.torch.mps.empty_cache()

        # GPU 内存使用情况
        if self.torch is not None:
            if torch.cuda.is_available():
                self.gpu_mem_used_at_start = self.torch.cuda.memory_allocated()
            elif is_torch_xpu_available():
                self.gpu_mem_used_at_start = self.torch.xpu.memory_allocated()
            elif is_torch_npu_available():
                self.gpu_mem_used_at_start = self.torch.npu.memory_allocated()
            elif is_torch_mps_available():
                self.gpu_mem_used_at_start = self.torch.mps.current_allocated_memory()

        # CPU 内存使用情况
        self.cpu_mem_used_at_start = self.cpu_mem_used()

        # 开启峰值监视
        self.peak_monitoring = True
        # 创建一个线程用于运行峰值监视方法
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
    # 更新指定阶段的度量指标
    def update_metrics(self, stage, metrics):
        """updates the metrics"""
        # 如果设置跳过内存度量，则直接返回
        if self.skip_memory_metrics:
            return

        # 处理在训练期间嵌套调用 eval 的情况，简单忽略这些调用
        if self.cur_stage is not None and self.cur_stage != stage:
            return

        # 如果未报告过初始化阶段的度量指标，则在 train/val/predict 之前插入 "init" 阶段
        stages = [stage]
        if not self.init_reported:
            stages.insert(0, "init")
            self.init_reported = True

        # 遍历所有阶段，更新内存相关的度量指标
        for stage in stages:
            for t in ["alloc", "peaked"]:
                # 更新 CPU 内存的增量度量指标
                if stage in self.cpu and t in self.cpu[stage]:
                    metrics[f"{stage}_mem_cpu_{t}_delta"] = self.cpu[stage][t]
                # 如果存在 GPU，更新 GPU 内存的增量度量指标
                if self.torch is not None and stage in self.gpu and t in self.gpu[stage]:
                    metrics[f"{stage}_mem_gpu_{t}_delta"] = self.gpu[stage][t]

            # 如果需要额外的调试信息，可以启用以下代码
            # for t in ["begin", "end"]:
            #     if stage in self.cpu and t in self.cpu[stage]:
            #         metrics[f"{stage}_mem_cpu_{t}"] = self.cpu[stage][t]
            #     if self.torch is not None and stage in self.gpu and t in self.gpu[stage]:
            #         metrics[f"{stage}_mem_gpu_{t}"] = self.gpu[stage][t]

        # 在 init 阶段报告内存使用情况
        if stages[0] == "init":
            # 报告 init 阶段 CPU 内存的初始使用量
            metrics["before_init_mem_cpu"] = self.cpu["init"]["begin"]
            # 如果存在 GPU，则报告 init 阶段 GPU 内存的初始使用量
            if self.torch is not None:
                metrics["before_init_mem_gpu"] = self.gpu["init"]["begin"]

            # 如果希望在 init 和下一个阶段之间报告额外的内存分配情况，可以启用以下代码
            # if self.cpu["init"]["end"] != self.cpu[stage]["begin"]:
            #     metrics[f"after_init_mem_cpu_delta"] = self.cpu[stage]["begin"] - self.cpu["init"]["end"]
            # if self.torch is not None and self.gpu["init"]["end"] != self.gpu[stage]["begin"]:
            #     metrics[f"after_init_mem_gpu_delta"] = self.gpu[stage]["begin"] - self.gpu["init"]["end"]

    # 结合停止和度量更新的函数调用，简化代码逻辑
    def stop_and_update_metrics(self, metrics=None):
        """combine stop and metrics update in one call for simpler code"""
        # 如果设置跳过内存度量，则直接返回
        if self.skip_memory_metrics:
            return

        # 推导当前阶段
        stage = self.derive_stage()
        # 执行停止操作
        self.stop(stage)

        # init 阶段没有度量需要更新，所以只保存数据以供后续阶段检索
        if metrics is not None:
            # 更新度量指标
            self.update_metrics(stage, metrics)
# 检查数据集是否实现了 __len__() 方法，并且调用该方法不会引发错误
def has_length(dataset):
    try:
        # 返回数据集的长度是否不为 None
        return len(dataset) is not None
    except TypeError:
        # 如果调用 len() 方法时出现 TypeError，则返回 False
        # TypeError: len() of unsized object
        return False


# 递归地调用 `.item()` 方法，将字典中的每个元素转换为其对应的标量值
def denumpify_detensorize(metrics):
    if isinstance(metrics, (list, tuple)):
        # 如果 metrics 是列表或元组，则递归调用 denumpify_detensorize() 处理每个元素
        return type(metrics)(denumpify_detensorize(m) for m in metrics)
    elif isinstance(metrics, dict):
        # 如果 metrics 是字典，则递归调用 denumpify_detensorize() 处理每对键值对
        return type(metrics)({k: denumpify_detensorize(v) for k, v in metrics.items()})
    elif isinstance(metrics, np.generic):
        # 如果 metrics 是 numpy 标量类型，则调用 .item() 方法获取其标量值
        return metrics.item()
    elif is_torch_available() and isinstance(metrics, torch.Tensor) and metrics.numel() == 1:
        # 如果 PyTorch 可用且 metrics 是包含单个元素的 Tensor，则调用 .item() 方法获取其标量值
        return metrics.item()
    # 其他情况直接返回 metrics
    return metrics


# 返回函数的参数数量，即使函数是 partial function 也可以
def number_of_arguments(func):
    if isinstance(func, functools.partial):
        # 如果 func 是 functools.partial 类型，则获取其真实函数的参数数量
        total_args = len(inspect.signature(func.func).parameters)
        # 减去 partial function 自身的参数数量得到真实的函数参数数量
        return total_args - len(func.args) - len(func.keywords)
    # 直接返回函数的参数数量
    return len(inspect.signature(func).parameters)


# 尝试执行函数，如果因内存不足或 CUDNN 相关异常失败，则减半批处理大小继续执行
def find_executable_batch_size(
    function: callable = None, starting_batch_size: int = 128, auto_find_batch_size: bool = False
):
    """
    Args:
    A basic decorator that will try to execute `function`. If it fails from exceptions related to out-of-memory or
    CUDNN, the batch size is cut in half and passed to `function`. `function` must take in a `batch_size` parameter as
    its first argument.
        function (`callable`, *optional*)
            A function to wrap
        starting_batch_size (`int`, *optional*)
            The batch size to try and fit into memory
        auto_find_batch_size (`bool`, *optional*)
            If False, will just execute `function`
    """
    if function is None:
        # 如果 function 参数为 None，则返回一个带有默认参数的 functools.partial 对象
        return functools.partial(
            find_executable_batch_size,
            starting_batch_size=starting_batch_size,
            auto_find_batch_size=auto_find_batch_size,
        )

    if auto_find_batch_size:
        # 如果 auto_find_batch_size 为 True，则引入 accelerate 模块，调用其内部的批处理大小搜索函数
        requires_backends(find_executable_batch_size, "accelerate")
        from accelerate.utils import find_executable_batch_size as accelerate_find_executable_batch_size

        return accelerate_find_executable_batch_size(function=function, starting_batch_size=starting_batch_size)

    # 如果 auto_find_batch_size 为 False，则直接调用传入的 function，并传递起始批处理大小参数
    return functools.partial(function, batch_size=starting_batch_size)


# 枚举类定义了一些特定的选项
class FSDPOption(ExplicitEnum):
    FULL_SHARD = "full_shard"
    SHARD_GRAD_OP = "shard_grad_op"
    NO_SHARD = "no_shard"
    HYBRID_SHARD = "hybrid_shard"
    HYBRID_SHARD_ZERO2 = "hybrid_shard_zero2"
    OFFLOAD = "offload"
    AUTO_WRAP = "auto_wrap"


class RemoveColumnsCollator:
    """Wrap the data collator to remove unused columns before they are passed to the collator."""
    # 数据整理器类，用于在传递给整理器之前移除未使用的列
    # 初始化函数，用于实例化对象时进行初始化操作
    def __init__(
        self,
        data_collator,                   # 数据合并器，用于合并特征
        signature_columns,               # 特征签名列，用于指定模型的输入特征
        logger=None,                     # 日志记录器，可选
        model_name: Optional[str] = None, # 模型名称，可选参数
        description: Optional[str] = None, # 描述信息，可选参数
    ):
        self.data_collator = data_collator  # 将数据合并器赋值给对象属性
        self.signature_columns = signature_columns  # 将特征签名列赋值给对象属性
        self.logger = logger                # 将日志记录器赋值给对象属性
        self.description = description      # 将描述信息赋值给对象属性
        self.model_name = model_name        # 将模型名称赋值给对象属性
        self.message_logged = False         # 初始化消息日志标记为 False，用于记录是否已经输出过消息

    # 私有方法，用于移除特征中不属于特征签名列的列
    def _remove_columns(self, feature: dict) -> dict:
        if not isinstance(feature, dict):  # 如果 feature 不是字典类型，则直接返回 feature
            return feature
        if not self.message_logged and self.logger and self.model_name:
            # 计算被忽略的列，即特征字典中存在但不属于特征签名列的列
            ignored_columns = list(set(feature.keys()) - set(self.signature_columns))
            if len(ignored_columns) > 0:
                # 构造日志信息，记录被忽略的列以及相关的描述信息和模型名称
                dset_description = "" if self.description is None else f"in the {self.description} set"
                self.logger.info(
                    f"The following columns {dset_description} don't have a corresponding argument in "
                    f"`{self.model_name}.forward` and have been ignored: {', '.join(ignored_columns)}."
                    f" If {', '.join(ignored_columns)} are not expected by `{self.model_name}.forward`, "
                    " you can safely ignore this message."
                )
                self.message_logged = True  # 设置消息日志标记为 True，表示已经输出过消息
        # 返回仅包含特征签名列的特征字典
        return {k: v for k, v in feature.items() if k in self.signature_columns}

    # 对象可调用方法，用于处理特征列表
    def __call__(self, features: List[dict]):
        # 对每个特征调用 _remove_columns 方法，移除不属于特征签名列的列
        features = [self._remove_columns(feature) for feature in features]
        return self.data_collator(features)  # 使用数据合并器合并处理后的特征列表并返回
def check_target_module_exists(optim_target_modules, key: str, return_is_regex: bool = False):
    """A helper method to check if the passed module's key name matches any of the target modules in the optim_target_modules.

    Args:
        optim_target_modules (`Union[str, List[str]]`):
            A list of strings to try to match. Can be also a full string.
        key (`str`):
            A key to search any matches in optim_target_modules
        return_is_regex (`bool`):
            If set to `True`, the method will return whether the passed `optim_target_modules`
            is a regex or not.

    Returns:
        `bool` : True of match object if key matches any target modules from config, False or
        None if no match found
        `bool` : If the matched target module is a regex to silence out the warnings in Trainer
        for extra modules being found (only if `target_module_found=True` for an array of regex).
    """
    # Initialize variables to track if target module is found and if it's a regex
    target_module_found = False
    is_regex = False

    # Check if optim_target_modules is a single string
    if isinstance(optim_target_modules, str):
        # Check if key matches the entire optim_target_modules string as a regex
        target_module_found = bool(re.fullmatch(optim_target_modules, key))
        # Determine if optim_target_modules is a regex based on whether it exactly matches key
        is_regex = True if not optim_target_modules == key else False
    # Check if key is directly in the list of optim_target_modules
    elif key in optim_target_modules:
        target_module_found = True
    # Check if key contains any substring that matches elements in optim_target_modules
    elif any(target_key in key for target_key in optim_target_modules):
        target_module_found = True
    # Check if key matches any element in optim_target_modules as a regex
    elif any(bool(re.fullmatch(optim_target_module, key)) for optim_target_module in optim_target_modules):
        target_module_found = True
        is_regex = True

    # If return_is_regex is True, return both target_module_found and is_regex
    if return_is_regex:
        return target_module_found, is_regex

    # Otherwise, return only target_module_found
    return target_module_found
```