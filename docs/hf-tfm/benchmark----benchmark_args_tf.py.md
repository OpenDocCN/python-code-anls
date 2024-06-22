# `.\transformers\benchmark\benchmark_args_tf.py`

```py
# 导入必要的模块
from dataclasses import dataclass, field  # 导入 dataclass 类和 field 函数
from typing import Tuple  # 导入 Tuple 类型

# 导入工具函数
from ..utils import cached_property, is_tf_available, logging, requires_backends  # 从上级目录的 utils 模块导入 cached_property、is_tf_available、logging 和 requires_backends 函数
from .benchmark_args_utils import BenchmarkArguments  # 从 benchmark_args_utils 模块导入 BenchmarkArguments 类

# 如果 TensorFlow 可用，则导入 TensorFlow 模块
if is_tf_available():
    import tensorflow as tf  # 导入 TensorFlow 模块

# 获取日志记录器
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# 定义 TensorFlowBenchmarkArguments 类，继承自 BenchmarkArguments 类
@dataclass
class TensorFlowBenchmarkArguments(BenchmarkArguments):
    # 定义过时的参数列表
    deprecated_args = [
        "no_inference",
        "no_cuda",
        "no_tpu",
        "no_speed",
        "no_memory",
        "no_env_print",
        "no_multi_process",
    ]

    def __init__(self, **kwargs):
        """
        This __init__ is there for legacy code. When removing deprecated args completely, the class can simply be
        deleted
        """
        # 遍历过时的参数列表
        for deprecated_arg in self.deprecated_args:
            # 如果参数存在于 kwargs 中
            if deprecated_arg in kwargs:
                # 构造对应的正向参数名称
                positive_arg = deprecated_arg[3:]
                # 将正向参数设置为与原参数相反的值，并从 kwargs 中删除原参数
                kwargs[positive_arg] = not kwargs.pop(deprecated_arg)
                # 发出警告，建议使用正向参数或者设定对应的正向参数值
                logger.warning(
                    f"{deprecated_arg} is depreciated. Please use --no-{positive_arg} or"
                    f" {positive_arg}={kwargs[positive_arg]}"
                )
        # 从 kwargs 中获取 TPU 名称，如果不存在则使用默认值
        self.tpu_name = kwargs.pop("tpu_name", self.tpu_name)
        # 从 kwargs 中获取设备索引，如果不存在则使用默认值
        self.device_idx = kwargs.pop("device_idx", self.device_idx)
        # 从 kwargs 中获取是否启用 eager 模式，如果不存在则使用默认值
        self.eager_mode = kwargs.pop("eager_mode", self.eager_mode)
        # 从 kwargs 中获取是否使用 XLA 编译，如果不存在则使用默认值
        self.use_xla = kwargs.pop("use_xla", self.use_xla)
        # 调用父类的 __init__ 方法初始化对象
        super().__init__(**kwargs)

    # 定义 TPU 名称属性，默认为 None
    tpu_name: str = field(
        default=None,
        metadata={"help": "Name of TPU"},
    )
    # 定义设备索引属性，默认为 0
    device_idx: int = field(
        default=0,
        metadata={"help": "CPU / GPU device index. Defaults to 0."},
    )
    # 定义是否启用 eager 模式属性，默认为 False
    eager_mode: bool = field(default=False, metadata={"help": "Benchmark models in eager model."})
    # 定义是否使用 XLA 编译属性，默认为 False
    use_xla: bool = field(
        default=False,
        metadata={
            "help": "Benchmark models using XLA JIT compilation. Note that `eager_model` has to be set to `False`."
        },
    )

    # 定义缓存属性装饰器
    @cached_property
    # 设置 TPU 环境
    def _setup_tpu(self) -> Tuple["tf.distribute.cluster_resolver.TPUClusterResolver"]:
        # 检查是否需要使用 TensorFlow 后端
        requires_backends(self, ["tf"])
        tpu = None
        # 如果存在 TPU
        if self.tpu:
            try:
                # 尝试创建 TPUClusterResolver 对象
                if self.tpu_name:
                    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(self.tpu_name)
                else:
                    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            except ValueError:
                tpu = None
        return tpu

    # 设置分布式策略
    @cached_property
    def _setup_strategy(self) -> Tuple["tf.distribute.Strategy", "tf.distribute.cluster_resolver.TPUClusterResolver"]:
        # 检查是否需要使用 TensorFlow 后端
        requires_backends(self, ["tf"])
        if self.is_tpu:
            # 连接到 TPU 集群
            tf.config.experimental_connect_to_cluster(self._setup_tpu)
            # 初始化 TPU 系统
            tf.tpu.experimental.initialize_tpu_system(self._setup_tpu)

            # 创建 TPUStrategy 对象
            strategy = tf.distribute.TPUStrategy(self._setup_tpu)
        else:
            # 当前不支持多 GPU
            if self.is_gpu:
                # 设置可见的 GPU 设备
                tf.config.set_visible_devices(self.gpu_list[self.device_idx], "GPU")
                # 创建 OneDeviceStrategy 对象
                strategy = tf.distribute.OneDeviceStrategy(device=f"/gpu:{self.device_idx}")
            else:
                # 禁用 GPU
                tf.config.set_visible_devices([], "GPU")
                # 创建 OneDeviceStrategy 对象
                strategy = tf.distribute.OneDeviceStrategy(device=f"/cpu:{self.device_idx}")

        return strategy

    # 判断是否存在 TPU
    @property
    def is_tpu(self) -> bool:
        # 检查是否需要使用 TensorFlow 后端
        requires_backends(self, ["tf"])
        return self._setup_tpu is not None

    # 返回分布式策略
    @property
    def strategy(self) -> "tf.distribute.Strategy":
        # 检查是否需要使用 TensorFlow 后端
        requires_backends(self, ["tf"])
        return self._setup_strategy

    # 返回 GPU 列表
    @property
    def gpu_list(self):
        # 检查是否需要使用 TensorFlow 后端
        requires_backends(self, ["tf"])
        return tf.config.list_physical_devices("GPU")

    # 返回 GPU 数量
    @property
    def n_gpu(self) -> int:
        # 检查是否需要使用 TensorFlow 后端
        requires_backends(self, ["tf"])
        if self.cuda:
            return len(self.gpu_list)
        return 0

    # 判断是否存在 GPU
    @property
    def is_gpu(self) -> bool:
        return self.n_gpu > 0
```