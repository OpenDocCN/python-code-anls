# `.\training_args_tf.py`

```py
# 导入警告模块，用于处理警告信息
import warnings
# 导入 dataclass 模块中的 dataclass 和 field 函数
from dataclasses import dataclass, field
# 导入 Optional 和 Tuple 类型提示
from typing import Optional, Tuple
# 导入 TrainingArguments 类，该类用于存储训练相关的参数
from .training_args import TrainingArguments
# 导入 cached_property 函数、is_tf_available 函数、logging 函数和 requires_backends 函数
from .utils import cached_property, is_tf_available, logging, requires_backends

# 获取当前模块的 logger
logger = logging.get_logger(__name__)

# 如果 TensorFlow 可用，导入 TensorFlow 模块
if is_tf_available():
    import tensorflow as tf
    # 导入 keras 子模块
    from .modeling_tf_utils import keras

@dataclass
class TFTrainingArguments(TrainingArguments):
    """
    TFTrainingArguments 是我们在示例脚本中使用的 TrainingArguments 的子集，用于与训练循环本身相关的参数。

    使用 HfArgumentParser，我们可以将这个类转换成可以在命令行上指定的 argparse 参数。

    """

    # 指定框架为 "tf"
    framework = "tf"
    # TPU 的名称
    tpu_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of TPU"},
    )
    # TPU 的区域
    tpu_zone: Optional[str] = field(
        default=None,
        metadata={"help": "Zone of TPU"},
    )
    # Cloud TPU 启用项目的名称
    gcp_project: Optional[str] = field(
        default=None,
        metadata={"help": "Name of Cloud TPU-enabled project"},
    )
    # 多项式衰减学习率调度器的幂次方
    poly_power: float = field(
        default=1.0,
        metadata={"help": "Power for the Polynomial decay LR scheduler."},
    )
    # 是否激活 XLA 编译
    xla: bool = field(default=False, metadata={"help": "Whether to activate the XLA compilation or not"})

    @cached_property
    def _setup_strategy(self) -> Tuple["tf.distribute.Strategy", int]:
        # 确保所需的后端包括 TensorFlow
        requires_backends(self, ["tf"])
        # 记录信息：Tensorflow 正在设置策略
        logger.info("Tensorflow: setting up strategy")

        # 获取物理设备中的 GPU 列表
        gpus = tf.config.list_physical_devices("GPU")

        # 如果设置为 fp16，则将全局策略设置为 mixed_float16
        if self.fp16:
            keras.mixed_precision.set_global_policy("mixed_float16")

        # 如果禁用 CUDA，则使用 OneDeviceStrategy 在 CPU 上运行
        if self.no_cuda:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        else:
            try:
                if self.tpu_name:
                    # 尝试连接指定名称的 TPU，设置区域和项目信息
                    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
                        self.tpu_name, zone=self.tpu_zone, project=self.gcp_project
                    )
                else:
                    # 尝试连接默认的 TPU
                    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            except ValueError:
                if self.tpu_name:
                    # 如果指定的 TPU 连接失败，则抛出运行时错误
                    raise RuntimeError(f"Couldn't connect to TPU {self.tpu_name}!")
                else:
                    # 如果没有指定 TPU 名称，则将 tpu 设为 None
                    tpu = None

            if tpu:
                # 如果使用 TPU，则根据设置的 fp16 将全局策略设置为 mixed_bfloat16
                if self.fp16:
                    keras.mixed_precision.set_global_policy("mixed_bfloat16")

                # 将当前进程连接到 TPU 群集
                tf.config.experimental_connect_to_cluster(tpu)
                # 初始化 TPU 系统
                tf.tpu.experimental.initialize_tpu_system(tpu)

                # 使用 TPUStrategy 进行分布式策略设置
                strategy = tf.distribute.TPUStrategy(tpu)

            elif len(gpus) == 0:
                # 如果没有 GPU 可用，则使用 OneDeviceStrategy 在 CPU 上运行
                strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
            elif len(gpus) == 1:
                # 如果只有一个 GPU 可用，则使用 OneDeviceStrategy 在该 GPU 上运行
                strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            elif len(gpus) > 1:
                # 如果有多个 GPU 可用，则使用 MirroredStrategy 在所有可见 GPU 上进行镜像策略
                # 如果只想使用特定的 GPU 子集，请使用 `CUDA_VISIBLE_DEVICES=0`。
                strategy = tf.distribute.MirroredStrategy()
            else:
                # 如果无法找到适当的策略，则抛出值错误
                raise ValueError("Cannot find the proper strategy, please check your environment properties.")

        # 返回设置好的策略及其关联的副本数
        return strategy

    @property
    def strategy(self) -> "tf.distribute.Strategy":
        """
        The strategy used for distributed training.
        """
        # 确保所需的后端包括 TensorFlow
        requires_backends(self, ["tf"])
        # 返回由 _setup_strategy 方法设置的策略对象
        return self._setup_strategy

    @property
    def n_replicas(self) -> int:
        """
        The number of replicas (CPUs, GPUs or TPU cores) used in this training.
        """
        # 确保所需的后端包括 TensorFlow
        requires_backends(self, ["tf"])
        # 返回当前策略中同步的副本数
        return self._setup_strategy.num_replicas_in_sync

    @property
    def should_log(self):
        """
        Whether or not the current process should produce log.
        """
        # 返回 False，因为 TF 的日志由 Keras 处理而不是 Trainer
        return False
    def train_batch_size(self) -> int:
        """
        训练时的实际批大小（在分布式训练中可能与 `per_gpu_train_batch_size` 不同）。
        """
        if self.per_gpu_train_batch_size:
            # 如果使用了过时的 `--per_gpu_train_batch_size` 参数，发出警告并建议使用 `--per_device_train_batch_size`。
            logger.warning(
                "Using deprecated `--per_gpu_train_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_train_batch_size` is preferred."
            )
        # 确定每个设备的批大小，优先选择 `per_gpu_train_batch_size`，否则选择 `per_device_train_batch_size`。
        per_device_batch_size = self.per_gpu_train_batch_size or self.per_device_train_batch_size
        # 返回每个设备的批大小乘以副本数（n_replicas），作为总的训练批大小。
        return per_device_batch_size * self.n_replicas

    @property
    def eval_batch_size(self) -> int:
        """
        评估时的实际批大小（在分布式训练中可能与 `per_gpu_eval_batch_size` 不同）。
        """
        if self.per_gpu_eval_batch_size:
            # 如果使用了过时的 `--per_gpu_eval_batch_size` 参数，发出警告并建议使用 `--per_device_eval_batch_size`。
            logger.warning(
                "Using deprecated `--per_gpu_eval_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_eval_batch_size` is preferred."
            )
        # 确定每个设备的评估批大小，优先选择 `per_gpu_eval_batch_size`，否则选择 `per_device_eval_batch_size`。
        per_device_batch_size = self.per_gpu_eval_batch_size or self.per_device_eval_batch_size
        # 返回每个设备的评估批大小乘以副本数（n_replicas），作为总的评估批大小。
        return per_device_batch_size * self.n_replicas

    @property
    def n_gpu(self) -> int:
        """
        用于训练的副本数（CPU、GPU 或 TPU 核心）。
        """
        # 确保使用的后端是 TensorFlow。
        requires_backends(self, ["tf"])
        # 发出警告，提示 `n_gpu` 参数已经过时，建议使用 `n_replicas` 代替。
        warnings.warn(
            "The n_gpu argument is deprecated and will be removed in a future version, use n_replicas instead.",
            FutureWarning,
        )
        # 返回当前设置的策略中的副本数（即同步中的副本数）。
        return self._setup_strategy.num_replicas_in_sync
```