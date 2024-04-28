# `.\transformers\training_args_tf.py`

```py
# 导入警告模块，用于处理警告信息
import warnings
# 导入 dataclass 模块的 dataclass 装饰器和 field 函数
from dataclasses import dataclass, field
# 导入 Optional 和 Tuple 类型
from typing import Optional, Tuple
# 从当前目录中的 training_args 模块导入 TrainingArguments 类
from .training_args import TrainingArguments
# 从当前目录中的 utils 模块导入 cached_property, is_tf_available, logging, requires_backends 函数
from .utils import cached_property, is_tf_available, logging, requires_backends

# 从 logging 模块中获取 logger 对象
logger = logging.get_logger(__name__)

# 如果 TensorFlow 可用
if is_tf_available():
    # 导入 TensorFlow 模块
    import tensorflow as tf

# TFTrainingArguments 类继承自 TrainingArguments 类
@dataclass
class TFTrainingArguments(TrainingArguments):
    """
    TrainingArguments 是我们在示例脚本中使用的与训练循环相关的参数的子集。

    使用 [`HfArgumentParser`] 我们可以将这个类转换为可以在命令行上指定的 [argparse](https://docs.python.org/3/library/argparse#module-argparse) 参数。

    """

    # 设置框架为 TensorFlow
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
    # GCP 项目的名称
    gcp_project: Optional[str] = field(
        default=None,
        metadata={"help": "Name of Cloud TPU-enabled project"},
    )
    # 多项式衰减学习率调度器的幂
    poly_power: float = field(
        default=1.0,
        metadata={"help": "Power for the Polynomial decay LR scheduler."},
    )
    # 是否激活 XLA 编译
    xla: bool = field(default=False, metadata={"help": "Whether to activate the XLA compilation or not"})

    # 缓存的属性
    @cached_property
    # 设置分布式策略，并返回策略对象和副本数量
    def _setup_strategy(self) -> Tuple["tf.distribute.Strategy", int]:
        # 检查所需后端是否可用，此处需要 TensorFlow 后端
        requires_backends(self, ["tf"])
        # 打印信息，表示正在设置 TensorFlow 策略
        logger.info("Tensorflow: setting up strategy")

        # 获取物理设备列表中的 GPU 设备
        gpus = tf.config.list_physical_devices("GPU")

        # 如果开启混合精度，则设置为 float16
        if self.fp16:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")

        # 如果禁用 GPU，则使用 OneDeviceStrategy 在 CPU 上执行
        if self.no_cuda:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        else:
            try:
                if self.tpu_name:
                    # 尝试连接到指定名称的 TPU，如果未指定名称，则连接默认 TPU
                    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
                        self.tpu_name, zone=self.tpu_zone, project=self.gcp_project
                    )
                else:
                    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            except ValueError:
                if self.tpu_name:
                    # 如果连接失败，则抛出异常
                    raise RuntimeError(f"Couldn't connect to TPU {self.tpu_name}!")
                else:
                    tpu = None

            # 如果存在 TPU，则设置为 bfloat16
            if tpu:
                if self.fp16:
                    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")

                # 连接到 TPU 集群并初始化 TPU 系统
                tf.config.experimental_connect_to_cluster(tpu)
                tf.tpu.experimental.initialize_tpu_system(tpu)

                # 使用 TPUStrategy
                strategy = tf.distribute.TPUStrategy(tpu)

            elif len(gpus) == 0:
                # 如果没有 GPU，则使用 OneDeviceStrategy 在 CPU 上执行
                strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
            elif len(gpus) == 1:
                # 如果只有一个 GPU，则使用 OneDeviceStrategy 在该 GPU 上执行
                strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            elif len(gpus) > 1:
                # 如果有多个 GPU，则使用 MirroredStrategy 在所有 GPU 上执行
                # 如果只想使用特定的一组 GPU，请使用 `CUDA_VISIBLE_DEVICES=0`
                strategy = tf.distribute.MirroredStrategy()
            else:
                # 如果找不到适当的策略，则抛出 ValueError
                raise ValueError("Cannot find the proper strategy, please check your environment properties.")

        # 返回设置好的策略对象和副本数量
        return strategy

    @property
    def strategy(self) -> "tf.distribute.Strategy":
        """
        The strategy used for distributed training.
        """
        # 检查所需后端是否可用，此处需要 TensorFlow 后端
        requires_backends(self, ["tf"])
        # 返回设置好的策略对象
        return self._setup_strategy

    @property
    def n_replicas(self) -> int:
        """
        The number of replicas (CPUs, GPUs or TPU cores) used in this training.
        """
        # 检查所需后端是否可用，此处需要 TensorFlow 后端
        requires_backends(self, ["tf"])
        # 返回设置好的策略对象中的副本数量
        return self._setup_strategy.num_replicas_in_sync

    @property
    def should_log(self):
        """
        Whether or not the current process should produce log.
        """
        # 返回 False，因为 TF 的日志由 Keras 处理而不是 Trainer
        return False  # TF Logging is handled by Keras not the Trainer

    @property
    def train_batch_size(self) -> int:
        """
        训练时实际的批量大小（在分布式训练中可能与`per_gpu_train_batch_size`不同）。
        """
        # 如果存在`per_gpu_train_batch_size`参数，则发出警告
        if self.per_gpu_train_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_train_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_train_batch_size` is preferred."
            )
        # 获取每个设备的批量大小
        per_device_batch_size = self.per_gpu_train_batch_size or self.per_device_train_batch_size
        # 返回每个设备的批量大小乘以副本数
        return per_device_batch_size * self.n_replicas

    @property
    def eval_batch_size(self) -> int:
        """
        评估时实际的批量大小（在分布式训练中可能与`per_gpu_eval_batch_size`不同）。
        """
        # 如果存在`per_gpu_eval_batch_size`参数，则发出警告
        if self.per_gpu_eval_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_eval_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_eval_batch_size` is preferred."
            )
        # 获取每个设备的批量大小
        per_device_batch_size = self.per_gpu_eval_batch_size or self.per_device_eval_batch_size
        # 返回每个设备的批量大小乘以副本数
        return per_device_batch_size * self.n_replicas

    @property
    def n_gpu(self) -> int:
        """
        用于训练的副本数（CPU、GPU或TPU核心）。
        """
        # 检查是否需要后端支持
        requires_backends(self, ["tf"])
        # 发出警告，提示参数`n_gpu`将在未来版本中被移除，建议使用`n_replicas`代替
        warnings.warn(
            "The n_gpu argument is deprecated and will be removed in a future version, use n_replicas instead.",
            FutureWarning,
        )
        # 返回设置好的策略中的副本数
        return self._setup_strategy.num_replicas_in_sync
```