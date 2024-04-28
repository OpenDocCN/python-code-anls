# `.\transformers\benchmark\benchmark_tf.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 HuggingFace Inc. 团队所有
# 版权声明，版权归 2018 年 NVIDIA 公司所有，保留所有权利
# 根据 Apache 许可证 2.0 版本使用此文件
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，不提供任何保证或条件
# 请参阅许可证以了解许可证下的特定语言和权限

"""
    在 PyTorch 中对库进行推理和训练的基准测试。
"""

# 导入必要的模块
import random  # 导入随机模块
import timeit  # 导入计时模块
from functools import wraps  # 导入装饰器模块
from typing import Callable, Optional  # 导入类型提示模块

# 导入库中的其他模块和函数
from ..configuration_utils import PretrainedConfig  # 从库的配置工具中导入 PretrainedConfig 类
from ..models.auto.modeling_tf_auto import TF_MODEL_MAPPING, TF_MODEL_WITH_LM_HEAD_MAPPING  # 从自动模型中导入 TensorFlow 模型映射
from ..utils import is_py3nvml_available, is_tf_available, logging  # 导入判断 TensorFlow 和 py3nvml 是否可用以及日志记录的工具函数
from .benchmark_utils import (  # 导入基准测试工具模块
    Benchmark,  # 导入基准测试类
    Memory,  # 导入内存类
    MemorySummary,  # 导入内存摘要类
    measure_peak_memory_cpu,  # 导入测量 CPU 峰值内存的函数
    start_memory_tracing,  # 导入开始内存追踪的函数
    stop_memory_tracing,  # 导入停止内存追踪的函数
)

# 如果 TensorFlow 可用
if is_tf_available():
    import tensorflow as tf  # 导入 TensorFlow 库
    from tensorflow.python.framework.errors_impl import ResourceExhaustedError  # 导入资源耗尽错误类

    from .benchmark_args_tf import TensorFlowBenchmarkArguments  # 从 TensorFlow 基准测试参数中导入 TensorFlowBenchmarkArguments 类

# 如果 py3nvml 可用
if is_py3nvml_available():
    import py3nvml.py3nvml as nvml  # 导入 py3nvml 库

# 获取日志记录器
logger = logging.get_logger(__name__)


# 装饰器函数，用于根据优化参数选择 TensorFlow 运行模式
def run_with_tf_optimizations(do_eager_mode: bool, use_xla: bool):
    def run_func(func):
        @wraps(func)
        def run_in_eager_mode(*args, **kwargs):
            return func(*args, **kwargs)  # 在 eager 模式下运行函数

        @wraps(func)
        @tf.function(experimental_compile=use_xla)  # 使用 TensorFlow 的函数图模式，可选择是否使用 XLA 编译
        def run_in_graph_mode(*args, **kwargs):
            return func(*args, **kwargs)  # 在图模式下运行函数

        # 如果设置了 eager 模式为 True，抛出 ValueError
        if do_eager_mode is True:
            if use_xla is not False:
                raise ValueError(
                    "Cannot run model in XLA, if `args.eager_mode` is set to `True`. Please set `args.eager_mode=False`."
                )
            return run_in_eager_mode  # 返回在 eager 模式下运行的函数
        else:
            return run_in_graph_mode  # 返回在图模式下运行的函数

    return run_func  # 返回运行函数的装饰器


# 生成随机输入 ID 的函数
def random_input_ids(batch_size: int, sequence_length: int, vocab_size: int) -> ["tf.Tensor"]:
    rng = random.Random()  # 创建随机数生成器
    values = [rng.randint(0, vocab_size - 1) for i in range(batch_size * sequence_length)]  # 生成随机输入 ID
    return tf.constant(values, shape=(batch_size, sequence_length), dtype=tf.int32)  # 返回 TensorFlow 张量形式的输入 ID


# TensorFlow 基准测试类
class TensorFlowBenchmark(Benchmark):
    args: TensorFlowBenchmarkArguments  # TensorFlow 基准测试参数
    configs: PretrainedConfig  # 预训练配置
    framework: str = "TensorFlow"  # 使用的框架为 TensorFlow

    @property
    def framework_version(self):  # 获取框架版本的属性方法
        return tf.__version__  # 返回 TensorFlow 的版本
    # 计算推理速度
    def _inference_speed(self, model_name: str, batch_size: int, sequence_length: int) -> float:
        # 初始化单独进程上的 GPU
        strategy = self.args.strategy
        if strategy is None:
            raise ValueError("A device strategy has to be initialized before using TensorFlow.")
        # 准备推理函数
        _inference = self._prepare_inference_func(model_name, batch_size, sequence_length)
        # 测量速度
        return self._measure_speed(_inference)

    # 计算训练速度
    def _train_speed(self, model_name: str, batch_size: int, sequence_length: int) -> float:
        strategy = self.args.strategy
        if strategy is None:
            raise ValueError("A device strategy has to be initialized before using TensorFlow.")
        # 准备训练函数
        _train = self._prepare_train_func(model_name, batch_size, sequence_length)
        # 测量速度
        return self._measure_speed(_train)

    # 计算推理内存占用
    def _inference_memory(
        self, model_name: str, batch_size: int, sequence_length: int
    ) -> [Memory, Optional[MemorySummary]]:
        # 初始化单独进程上的 GPU
        if self.args.is_gpu:
            tf.config.experimental.set_memory_growth(self.args.gpu_list[self.args.device_idx], True)
        strategy = self.args.strategy
        if strategy is None:
            raise ValueError("A device strategy has to be initialized before using TensorFlow.")
        # 准备推理函数
        _inference = self._prepare_inference_func(model_name, batch_size, sequence_length)
        # 测量内存占用
        return self._measure_memory(_inference)

    # 计算训练内存占用
    def _train_memory(
        self, model_name: str, batch_size: int, sequence_length: int
    ) -> [Memory, Optional[MemorySummary]]:
        if self.args.is_gpu:
            tf.config.experimental.set_memory_growth(self.args.gpu_list[self.args.device_idx], True)
        strategy = self.args.strategy
        if strategy is None:
            raise ValueError("A device strategy has to be initialized before using TensorFlow.")
        # 准备训练函数
        _train = self._prepare_train_func(model_name, batch_size, sequence_length)
        # 测量内存占用
        return self._measure_memory(_train)
    # 准备推理函数，根据模型名称、批处理大小和序列长度返回一个无参数函数
    def _prepare_inference_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        # 获取模型配置
        config = self.config_dict[model_name]

        # 检查是否启用了混合精度
        if self.args.fp16:
            raise NotImplementedError("Mixed precision is currently not supported.")

        # 检查配置中是否包含模型类
        has_model_class_in_config = (
            hasattr(config, "architectures")
            and isinstance(config.architectures, list)
            and len(config.architectures) > 0
        )
        # 如果不仅仅是预训练模型且配置中包含模型类
        if not self.args.only_pretrain_model and has_model_class_in_config:
            try:
                # 获取模型类名并在前面加上 'TF' 以表示是 TensorFlow 模型
                model_class = "TF" + config.architectures[0]
                # 动态导入 transformers 模块并获取模型类
                transformers_module = __import__("transformers", fromlist=[model_class])
                model_cls = getattr(transformers_module, model_class)
                # 根据模型类实例化模型
                model = model_cls(config)
            except ImportError:
                raise ImportError(
                    f"{model_class} does not exist. If you just want to test the pretrained model, you might want to"
                    " set `--only_pretrain_model` or `args.only_pretrain_model=True`."
                )
        else:
            # 根据配置选择相应的 TensorFlow 模型
            model = TF_MODEL_MAPPING[config.__class__](config)

        # 对于编码器-解码器模型，词汇表大小保存在不同的位置
        vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size
        # 生成随机输入 ID
        input_ids = random_input_ids(batch_size, sequence_length, vocab_size)

        # 定义编码器-解码器前向推理函数
        @run_with_tf_optimizations(self.args.eager_mode, self.args.use_xla)
        def encoder_decoder_forward():
            return model(input_ids, decoder_input_ids=input_ids, training=False)

        # 定义编码器前向推理函数
        @run_with_tf_optimizations(self.args.eager_mode, self.args.use_xla)
        def encoder_forward():
            return model(input_ids, training=False)

        # 根据模型类型选择相应的推理函数
        _inference = encoder_decoder_forward if config.is_encoder_decoder else encoder_forward

        return _inference
    # 准备训练函数，返回一个无参数的函数，用于训练模型
    def _prepare_train_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        # 获取模型配置
        config = self.config_dict[model_name]

        # 检查是否启用了即时执行模式
        if self.args.eager_mode is not False:
            # 如果启用了即时执行模式，抛出数值错误
            raise ValueError("Training cannot be done in eager mode. Please make sure that `args.eager_mode = False`.")

        # 检查是否启用了混合精度
        if self.args.fp16:
            # 抛出未实现错误，因为当前不支持混合精度训练
            raise NotImplementedError("Mixed precision is currently not supported.")

        # 检查模型配置中是否包含模型类，并且模型类的列表长度大于0
        has_model_class_in_config = (
            hasattr(config, "architectures")
            and isinstance(config.architectures, list)
            and len(config.architectures) > 0
        )
        # 如果不仅仅是预训练模型，并且配置中包含模型类
        if not self.args.only_pretrain_model and has_model_class_in_config:
            try:
                # 获取模型类名，并在其前添加'TF'以表示tensorflow模型
                model_class = "TF" + config.architectures[0]
                # 动态导入transformers模块
                transformers_module = __import__("transformers", fromlist=[model_class])
                # 获取模型类对象
                model_cls = getattr(transformers_module, model_class)
                # 根据配置创建模型实例
                model = model_cls(config)
            except ImportError:
                # 如果导入失败，抛出导入错误
                raise ImportError(
                    f"{model_class} does not exist. If you just want to test the pretrained model, you might want to"
                    " set `--only_pretrain_model` or `args.only_pretrain_model=True`."
                )
        else:
            # 如果只是预训练模型或者配置中没有包含模型类，则根据配置创建模型实例
            model = TF_MODEL_WITH_LM_HEAD_MAPPING[config.__class__](config)

        # 对于编码器-解码器模型，词汇表大小存储在不同的地方
        vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size
        # 生成随机输入id
        input_ids = random_input_ids(batch_size, sequence_length, vocab_size)

        # 定义编码器-解码器训练函数
        @run_with_tf_optimizations(self.args.eager_mode, self.args.use_xla)
        def encoder_decoder_train():
            # 计算损失
            loss = model(input_ids, decoder_input_ids=input_ids, labels=input_ids, training=True)[0]
            # 计算梯度
            gradients = tf.gradients(loss, model.trainable_variables)
            return gradients

        # 定义编码器训练函数
        @run_with_tf_optimizations(self.args.eager_mode, self.args.use_xla)
        def encoder_train():
            # 计算损失
            loss = model(input_ids, labels=input_ids, training=True)[0]
            # 计算梯度
            gradients = tf.gradients(loss, model.trainable_variables)
            return gradients

        # 根据模型类型选择对应的训练函数
        _train = encoder_decoder_train if config.is_encoder_decoder else encoder_train

        # 返回选定的训练函数
        return _train
    # 定义一个私有方法，用于测量函数执行速度，返回一个浮点数表示执行时间
    def _measure_speed(self, func) -> float:
        # 使用指定的策略进行上下文管理
        with self.args.strategy.scope():
            # 尝试执行以下代码块，捕获异常
            try:
                # 如果正在使用 TPU 或启用了 XLA
                if self.args.is_tpu or self.args.use_xla:
                    # 在 TPU 上运行额外的 10 次以稳定编译
                    logger.info("Do inference on TPU. Running model 5 times to stabilize compilation")
                    timeit.repeat(func, repeat=1, number=5)

                # 如 https://docs.python.org/2/library/timeit.html#timeit.Timer.repeat 中所述，应取最小值而不是平均值
                # 使用 timeit 库多次运行指定的函数，返回每次运行的时间
                runtimes = timeit.repeat(
                    func,
                    repeat=self.args.repeat,  # 重复运行次数
                    number=10,  # 每次运行的执行次数
                )

                # 返回执行时间最短的一次执行时间除以执行次数的结果
                return min(runtimes) / 10.0
            # 捕获 ResourceExhaustedError 异常
            except ResourceExhaustedError as e:
                # 打印错误信息
                self.print_fn(f"Doesn't fit on GPU. {e}")
```