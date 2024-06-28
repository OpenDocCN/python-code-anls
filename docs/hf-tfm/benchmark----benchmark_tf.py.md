# `.\benchmark\benchmark_tf.py`

```
# coding=utf-8
# 设置编码格式为 UTF-8

# Copyright 2018 The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# 版权声明

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache 许可证版本 2.0 进行许可

# you may not use this file except in compliance with the License.
# 除非符合许可证要求，否则不得使用此文件

# You may obtain a copy of the License at
# 可以在以下网址获取许可证副本

#     http://www.apache.org/licenses/LICENSE-2.0
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非适用法律要求或书面同意，本软件按"原样"分发，不提供任何形式的担保或条件

# See the License for the specific language governing permissions and
# limitations under the License.
# 请查看许可证以了解特定的语言授权和限制

"""
    Benchmarking the library on inference and training in PyTorch.
"""
# 此模块用于在 PyTorch 中进行推断和训练的性能基准测试

import random  # 导入随机数模块
import timeit  # 导入计时模块
from functools import wraps  # 导入 wraps 装饰器
from typing import Callable, Optional  # 导入类型提示

from ..configuration_utils import PretrainedConfig  # 导入预训练配置
from ..models.auto.modeling_tf_auto import TF_MODEL_MAPPING, TF_MODEL_WITH_LM_HEAD_MAPPING  # 导入 TensorFlow 模型映射
from ..utils import is_py3nvml_available, is_tf_available, logging  # 导入工具函数和日志模块
from .benchmark_utils import (  # 导入性能基准测试相关工具
    Benchmark,
    Memory,
    MemorySummary,
    measure_peak_memory_cpu,
    start_memory_tracing,
    stop_memory_tracing,
)

# 如果 TensorFlow 可用，则导入 TensorFlow 模块和相关错误类
if is_tf_available():
    import tensorflow as tf
    from tensorflow.python.framework.errors_impl import ResourceExhaustedError

    from .benchmark_args_tf import TensorFlowBenchmarkArguments  # 导入 TensorFlow 的性能基准测试参数

# 如果 py3nvml 可用，则导入 py3nvml 模块
if is_py3nvml_available():
    import py3nvml.py3nvml as nvml

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def run_with_tf_optimizations(do_eager_mode: bool, use_xla: bool):
    """
    返回一个装饰器函数，根据参数决定以急切模式还是图模式运行 TensorFlow 函数。

    Args:
        do_eager_mode (bool): 是否使用急切执行模式
        use_xla (bool): 是否使用 XLA 加速

    Returns:
        Callable: 装饰器函数，用于在急切模式或图模式下运行给定函数
    """
    def run_func(func):
        @wraps(func)
        def run_in_eager_mode(*args, **kwargs):
            return func(*args, **kwargs)

        @wraps(func)
        @tf.function(experimental_compile=use_xla)
        def run_in_graph_mode(*args, **kwargs):
            return func(*args, **kwargs)

        if do_eager_mode is True:
            if use_xla is not False:
                raise ValueError(
                    "Cannot run model in XLA, if `args.eager_mode` is set to `True`. Please set `args.eager_mode=False`."
                )
            return run_in_eager_mode
        else:
            return run_in_graph_mode

    return run_func


def random_input_ids(batch_size: int, sequence_length: int, vocab_size: int) -> ["tf.Tensor"]:
    """
    生成指定形状和范围内随机整数张量作为输入 ID。

    Args:
        batch_size (int): 批量大小
        sequence_length (int): 序列长度
        vocab_size (int): 词汇表大小

    Returns:
        tf.Tensor: 随机整数张量，形状为 (batch_size, sequence_length)
    """
    rng = random.Random()
    values = [rng.randint(0, vocab_size - 1) for i in range(batch_size * sequence_length)]
    return tf.constant(values, shape=(batch_size, sequence_length), dtype=tf.int32)


class TensorFlowBenchmark(Benchmark):
    """
    TensorFlow 的性能基准测试类，继承自 Benchmark 类。
    """
    args: TensorFlowBenchmarkArguments  # TensorFlow 的性能基准测试参数
    configs: PretrainedConfig  # 预训练模型的配置
    framework: str = "TensorFlow"  # 框架名称为 TensorFlow

    @property
    def framework_version(self):
        """
        返回当前 TensorFlow 的版本号。

        Returns:
            str: TensorFlow 的版本号字符串
        """
        return tf.__version__  # 返回 TensorFlow 的版本号
    # 计算推理速度的私有方法，返回模型推理速度（每秒推理样本数）
    def _inference_speed(self, model_name: str, batch_size: int, sequence_length: int) -> float:
        # 获取设备策略
        strategy = self.args.strategy
        # 如果策略为空，则抛出数值错误异常
        if strategy is None:
            raise ValueError("A device strategy has to be initialized before using TensorFlow.")
        # 准备推理函数
        _inference = self._prepare_inference_func(model_name, batch_size, sequence_length)
        # 测量推理函数的速度并返回
        return self._measure_speed(_inference)

    # 计算训练速度的私有方法，返回模型训练速度（每秒训练样本数）
    def _train_speed(self, model_name: str, batch_size: int, sequence_length: int) -> float:
        # 获取设备策略
        strategy = self.args.strategy
        # 如果策略为空，则抛出数值错误异常
        if strategy is None:
            raise ValueError("A device strategy has to be initialized before using TensorFlow.")
        # 准备训练函数
        _train = self._prepare_train_func(model_name, batch_size, sequence_length)
        # 测量训练函数的速度并返回
        return self._measure_speed(_train)

    # 计算推理内存占用的私有方法，返回模型推理时内存信息
    def _inference_memory(
        self, model_name: str, batch_size: int, sequence_length: int
    ) -> [Memory, Optional[MemorySummary]]:
        # 如果使用 GPU，则设置 GPU 内存增长策略
        if self.args.is_gpu:
            tf.config.experimental.set_memory_growth(self.args.gpu_list[self.args.device_idx], True)
        # 获取设备策略
        strategy = self.args.strategy
        # 如果策略为空，则抛出数值错误异常
        if strategy is None:
            raise ValueError("A device strategy has to be initialized before using TensorFlow.")
        # 准备推理函数
        _inference = self._prepare_inference_func(model_name, batch_size, sequence_length)
        # 测量推理函数的内存占用并返回
        return self._measure_memory(_inference)

    # 计算训练内存占用的私有方法，返回模型训练时内存信息
    def _train_memory(
        self, model_name: str, batch_size: int, sequence_length: int
    ) -> [Memory, Optional[MemorySummary]]:
        # 如果使用 GPU，则设置 GPU 内存增长策略
        if self.args.is_gpu:
            tf.config.experimental.set_memory_growth(self.args.gpu_list[self.args.device_idx], True)
        # 获取设备策略
        strategy = self.args.strategy
        # 如果策略为空，则抛出数值错误异常
        if strategy is None:
            raise ValueError("A device strategy has to be initialized before using TensorFlow.")

        # 准备训练函数
        _train = self._prepare_train_func(model_name, batch_size, sequence_length)
        # 测量训练函数的内存占用并返回
        return self._measure_memory(_train)
    # 准备推断函数，用于模型推断
    def _prepare_inference_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        # 获取指定模型配置
        config = self.config_dict[model_name]

        # 如果启用了混合精度，则抛出未实现错误
        if self.args.fp16:
            raise NotImplementedError("Mixed precision is currently not supported.")

        # 检查配置中是否包含模型类信息
        has_model_class_in_config = (
            hasattr(config, "architectures")
            and isinstance(config.architectures, list)
            and len(config.architectures) > 0
        )
        # 如果不仅仅是预训练模型且配置中有模型类信息，则尝试初始化模型
        if not self.args.only_pretrain_model and has_model_class_in_config:
            try:
                # 构建模型类名，以'TF'开头表示使用TensorFlow模型
                model_class = "TF" + config.architectures[0]
                # 动态导入transformers库中的模型类
                transformers_module = __import__("transformers", fromlist=[model_class])
                model_cls = getattr(transformers_module, model_class)
                # 使用配置初始化模型
                model = model_cls(config)
            except ImportError:
                # 如果导入失败，则抛出导入错误，提示用户设置`--only_pretrain_model`参数测试预训练模型
                raise ImportError(
                    f"{model_class} does not exist. If you just want to test the pretrained model, you might want to"
                    " set `--only_pretrain_model` or `args.only_pretrain_model=True`."
                )
        else:
            # 如果仅仅是预训练模型或配置中没有模型类信息，则使用预定义的映射创建模型
            model = TF_MODEL_MAPPING[config.__class__](config)

        # 对于编码器-解码器模型，vocab_size的保存方式有所不同
        vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size
        # 生成随机输入ID，用于模型推断
        input_ids = random_input_ids(batch_size, sequence_length, vocab_size)

        # 定义编码器-解码器模型推断函数，根据是否是编码器-解码器模型选择不同的输入方式
        @run_with_tf_optimizations(self.args.eager_mode, self.args.use_xla)
        def encoder_decoder_forward():
            return model(input_ids, decoder_input_ids=input_ids, training=False)

        # 定义编码器模型推断函数
        @run_with_tf_optimizations(self.args.eager_mode, self.args.use_xla)
        def encoder_forward():
            return model(input_ids, training=False)

        # 根据配置选择推断函数是编码器-解码器推断还是编码器推断
        _inference = encoder_decoder_forward if config.is_encoder_decoder else encoder_forward

        # 返回选择的推断函数
        return _inference
    # 定义一个私有方法，用于准备训练函数，该函数返回一个无参数的可调用对象
    def _prepare_train_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        # 从配置字典中获取特定模型名称对应的配置
        config = self.config_dict[model_name]

        # 如果参数中设置了 eager_mode 不为 False，抛出数值错误
        if self.args.eager_mode is not False:
            raise ValueError("Training cannot be done in eager mode. Please make sure that `args.eager_mode = False`.")

        # 如果参数中启用了 fp16，抛出未实现错误，暂不支持混合精度训练
        if self.args.fp16:
            raise NotImplementedError("Mixed precision is currently not supported.")

        # 检查配置中是否包含模型类信息
        has_model_class_in_config = (
            hasattr(config, "architectures")  # 检查配置是否包含 architectures 属性
            and isinstance(config.architectures, list)  # architectures 属性是否为列表类型
            and len(config.architectures) > 0  # architectures 列表长度大于 0
        )
        # 如果不仅是预训练模型，并且配置中包含模型类信息，则尝试加载模型类
        if not self.args.only_pretrain_model and has_model_class_in_config:
            try:
                # 构建模型类名称，以 'TF' 开头表示 TensorFlow 模型
                model_class = "TF" + config.architectures[0]
                # 动态导入 transformers 模块中的指定模型类
                transformers_module = __import__("transformers", fromlist=[model_class])
                model_cls = getattr(transformers_module, model_class)
                # 使用模型类和配置创建模型实例
                model = model_cls(config)
            except ImportError:
                # 如果导入失败，抛出导入错误，提醒用户检查模型类是否存在
                raise ImportError(
                    f"{model_class} does not exist. If you just want to test the pretrained model, you might want to"
                    " set `--only_pretrain_model` or `args.only_pretrain_model=True`."
                )
        else:
            # 如果仅加载预训练模型或配置中不包含模型类信息，则使用默认的语言模型和配置创建模型
            model = TF_MODEL_WITH_LM_HEAD_MAPPING[config.__class__](config)

        # 对于 encoder-decoder 类型的模型，需要特殊处理词汇表大小的设置
        vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size
        # 生成随机的输入 ID，用于模型训练
        input_ids = random_input_ids(batch_size, sequence_length, vocab_size)

        # 定义 encoder-decoder 模型训练函数，根据 eager_mode 和 use_xla 参数优化执行方式
        @run_with_tf_optimizations(self.args.eager_mode, self.args.use_xla)
        def encoder_decoder_train():
            # 计算模型在给定输入下的损失值，并获取损失相对于可训练变量的梯度
            loss = model(input_ids, decoder_input_ids=input_ids, labels=input_ids, training=True)[0]
            gradients = tf.gradients(loss, model.trainable_variables)
            return gradients

        # 定义 encoder 模型训练函数，根据 eager_mode 和 use_xla 参数优化执行方式
        @run_with_tf_optimizations(self.args.eager_mode, self.args.use_xla)
        def encoder_train():
            # 计算模型在给定输入下的损失值，并获取损失相对于可训练变量的梯度
            loss = model(input_ids, labels=input_ids, training=True)[0]
            gradients = tf.gradients(loss, model.trainable_variables)
            return gradients

        # 根据模型配置决定返回 encoder-decoder 训练函数还是 encoder 训练函数
        _train = encoder_decoder_train if config.is_encoder_decoder else encoder_train

        return _train
    def _measure_speed(self, func) -> float:
        # 使用给定的策略作用域执行以下代码块
        with self.args.strategy.scope():
            try:
                if self.args.is_tpu or self.args.use_xla:
                    # 如果使用 TPU 或者启用 XLA，则额外运行 5 次以稳定编译过程
                    logger.info("Do inference on TPU. Running model 5 times to stabilize compilation")
                    timeit.repeat(func, repeat=1, number=5)

                # 根据文档建议，使用最小值而非平均值来计算时间
                runtimes = timeit.repeat(
                    func,
                    repeat=self.args.repeat,  # 重复测量次数
                    number=10,  # 每次测量执行的次数
                )

                # 返回最小运行时间的平均值
                return min(runtimes) / 10.0
            except ResourceExhaustedError as e:
                # 如果资源不足错误，则打印相关信息
                self.print_fn(f"Doesn't fit on GPU. {e}")
```