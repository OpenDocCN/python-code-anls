# `.\benchmark\benchmark_args_tf.py`

```
# 设置编码格式为 UTF-8
# 版权声明，版权归 HuggingFace Inc. 团队和 NVIDIA CORPORATION 所有
# 根据 Apache License, Version 2.0 许可证使用本文件，除非符合许可证的条款，否则不得使用本文件
# 获取许可证的副本，请访问 http://www.apache.org/licenses/LICENSE-2.0
# 根据适用法律要求或书面同意，本软件按“原样”分发，无任何明示或暗示的担保或条件
# 有关许可证的详细信息，请参阅许可证文档

# 导入必要的模块和库
from dataclasses import dataclass, field  # 导入 dataclass 类型和 field 函数
from typing import Tuple  # 导入 Tuple 类型

# 从自定义的 utils 模块中导入 cached_property, is_tf_available, logging, requires_backends 函数
from ..utils import cached_property, is_tf_available, logging, requires_backends
# 从 benchmark_args_utils 模块中导入 BenchmarkArguments 类
from .benchmark_args_utils import BenchmarkArguments

# 如果 TensorFlow 可用，导入 TensorFlow 模块
if is_tf_available():
    import tensorflow as tf

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义 TensorFlowBenchmarkArguments 类，继承自 BenchmarkArguments 类
@dataclass
class TensorFlowBenchmarkArguments(BenchmarkArguments):
    # 已弃用的参数列表
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
        初始化方法用于处理已弃用的参数。在完全移除弃用参数后，可以删除此方法和相应代码。
        """
        # 遍历已弃用的参数列表
        for deprecated_arg in self.deprecated_args:
            # 如果传入的参数中包含已弃用的参数
            if deprecated_arg in kwargs:
                # 根据约定将参数名处理为正向的名称
                positive_arg = deprecated_arg[3:]
                # 将已弃用参数的值设置为相反的值，并移除原有的已弃用参数
                kwargs[positive_arg] = not kwargs.pop(deprecated_arg)
                # 记录警告日志，提示用户使用正确的参数或标志
                logger.warning(
                    f"{deprecated_arg} is deprecated. Please use --no-{positive_arg} or "
                    f"{positive_arg}={kwargs[positive_arg]}"
                )
        
        # 将 TPU 名称从传入的参数中提取出来，如果不存在则使用默认值
        self.tpu_name = kwargs.pop("tpu_name", self.tpu_name)
        # 将设备索引号从传入的参数中提取出来，如果不存在则使用默认值
        self.device_idx = kwargs.pop("device_idx", self.device_idx)
        # 将 eager 模式标志从传入的参数中提取出来，如果不存在则使用默认值
        self.eager_mode = kwargs.pop("eager_mode", self.eager_mode)
        # 将使用 XLA 编译的标志从传入的参数中提取出来，如果不存在则使用默认值
        self.use_xla = kwargs.pop("use_xla", self.use_xla)
        
        # 调用父类 BenchmarkArguments 的初始化方法，传入剩余的参数
        super().__init__(**kwargs)

    # TPU 名称，支持使用帮助文档
    tpu_name: str = field(
        default=None,
        metadata={"help": "Name of TPU"},
    )
    
    # 设备索引号，默认为 0，支持使用帮助文档
    device_idx: int = field(
        default=0,
        metadata={"help": "CPU / GPU device index. Defaults to 0."},
    )
    
    # 是否启用 eager 模式的标志，默认为 False，支持使用帮助文档
    eager_mode: bool = field(default=False, metadata={"help": "Benchmark models in eager model."})
    
    # 是否使用 XLA JIT 编译的标志，默认为 False，支持使用帮助文档
    use_xla: bool = field(
        default=False,
        metadata={
            "help": "Benchmark models using XLA JIT compilation. Note that `eager_model` has to be set to `False`."
        },
    )

    @cached_property
    # 设置用于处理 TPU 的函数，返回一个 TPUClusterResolver 对象或 None
    def _setup_tpu(self) -> Tuple["tf.distribute.cluster_resolver.TPUClusterResolver"]:
        # 要求当前对象支持 TensorFlow 后端
        requires_backends(self, ["tf"])
        tpu = None
        # 如果已经配置了 TPU
        if self.tpu:
            try:
                # 如果指定了 TPU 名称，使用指定名称创建 TPUClusterResolver 对象
                if self.tpu_name:
                    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(self.tpu_name)
                # 否则创建默认 TPUClusterResolver 对象
                else:
                    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            except ValueError:
                tpu = None
        return tpu

    # 设置分布策略的缓存属性，返回一个包含策略和 TPUClusterResolver 对象的元组
    @cached_property
    def _setup_strategy(self) -> Tuple["tf.distribute.Strategy", "tf.distribute.cluster_resolver.TPUClusterResolver"]:
        # 要求当前对象支持 TensorFlow 后端
        requires_backends(self, ["tf"])
        # 如果是 TPU 环境
        if self.is_tpu:
            # 连接到 TPU 集群
            tf.config.experimental_connect_to_cluster(self._setup_tpu)
            # 初始化 TPU 系统
            tf.tpu.experimental.initialize_tpu_system(self._setup_tpu)

            # 创建 TPUStrategy 对象
            strategy = tf.distribute.TPUStrategy(self._setup_tpu)
        else:
            # 当前不允许多 GPU 情况
            if self.is_gpu:
                # TODO: 目前仅支持单 GPU
                # 设置可见的 GPU 设备为指定索引的 GPU
                tf.config.set_visible_devices(self.gpu_list[self.device_idx], "GPU")
                # 创建 OneDeviceStrategy 对象，指定设备为指定索引的 GPU
                strategy = tf.distribute.OneDeviceStrategy(device=f"/gpu:{self.device_idx}")
            else:
                # 禁用 GPU，设置可见的设备为空列表
                tf.config.set_visible_devices([], "GPU")
                # 创建 OneDeviceStrategy 对象，指定设备为指定索引的 CPU
                strategy = tf.distribute.OneDeviceStrategy(device=f"/cpu:{self.device_idx}")

        return strategy

    # 返回当前是否配置了 TPU
    @property
    def is_tpu(self) -> bool:
        # 要求当前对象支持 TensorFlow 后端
        requires_backends(self, ["tf"])
        return self._setup_tpu is not None

    # 返回当前的分布策略
    @property
    def strategy(self) -> "tf.distribute.Strategy":
        # 要求当前对象支持 TensorFlow 后端
        requires_backends(self, ["tf"])
        return self._setup_strategy

    # 返回当前可用的 GPU 列表
    @property
    def gpu_list(self):
        # 要求当前对象支持 TensorFlow 后端
        requires_backends(self, ["tf"])
        return tf.config.list_physical_devices("GPU")

    # 返回当前可用的 GPU 数量
    @property
    def n_gpu(self) -> int:
        # 要求当前对象支持 TensorFlow 后端
        requires_backends(self, ["tf"])
        # 如果支持 CUDA，则返回 GPU 列表的长度
        if self.cuda:
            return len(self.gpu_list)
        # 否则返回 0
        return 0

    # 返回当前是否配置了 GPU
    @property
    def is_gpu(self) -> bool:
        return self.n_gpu > 0
```