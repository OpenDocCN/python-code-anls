# `.\transformers\benchmark\benchmark.py`

```py
# 指定编码格式为 UTF-8
# 版权声明，版权归 HuggingFace Inc. 团队和 NVIDIA 公司所有
# 根据 Apache 许可证 2.0 版本使用本文件，除非遵循许可证，否则不得使用此文件
# 可在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证副本
# 除非适用法律要求或书面同意，否则本软件按“原样”分发，不附带任何明示或暗示的保证或条件
# 有关详细信息，请参阅许可证
"""
    在 PyTorch 中对库进行推理和训练的基准测试。
"""

# 导入计时器模块
import timeit
# 导入类型提示模块
from typing import Callable, Optional
# 从库中导入相关配置
from ..configuration_utils import PretrainedConfig
from ..models.auto.modeling_auto import MODEL_MAPPING, MODEL_WITH_LM_HEAD_MAPPING
from ..utils import is_py3nvml_available, is_torch_available, logging
# 导入内存相关的模块
from .benchmark_utils import (
    Benchmark,
    Memory,
    MemorySummary,
    measure_peak_memory_cpu,
    start_memory_tracing,
    stop_memory_tracing,
)

# 如果 PyTorch 可用，则导入 PyTorch 模块
if is_torch_available():
    import torch

    # 从 benchmark_args 模块中导入 PyTorchBenchmarkArguments 类
    from .benchmark_args import PyTorchBenchmarkArguments

# 如果 py3nvml 可用，则导入 py3nvml 模块
if is_py3nvml_available():
    import py3nvml.py3nvml as nvml

# 获取日志记录器
logger = logging.get_logger(__name__)

# PyTorchBenchmark 类继承自 Benchmark 类
class PyTorchBenchmark(Benchmark):
    # PyTorchBenchmark 类的属性包括参数和配置
    args: PyTorchBenchmarkArguments
    configs: PretrainedConfig
    # 框架名称为 PyTorch
    framework: str = "PyTorch"

    # 框架版本属性返回当前 PyTorch 版本
    @property
    def framework_version(self):
        return torch.__version__

    # 内部方法，用于测量推理速度
    def _inference_speed(self, model_name: str, batch_size: int, sequence_length: int) -> float:
        # 准备推理函数
        _inference = self._prepare_inference_func(model_name, batch_size, sequence_length)
        # 测量推理速度
        return self._measure_speed(_inference)

    # 内部方法，用于测量推理内存占用
    def _inference_memory(
        self, model_name: str, batch_size: int, sequence_length: int
    ) -> [Memory, Optional[MemorySummary]]:
        # 准备推理函数
        _inference = self._prepare_inference_func(model_name, batch_size, sequence_length)
        # 测量内存占用
        return self._measure_memory(_inference)

    # 内部方法，用于测量训练速度
    def _train_speed(self, model_name: str, batch_size: int, sequence_length: int) -> float:
        # 准备训练函数
        _train = self._prepare_train_func(model_name, batch_size, sequence_length)
        # 测量训练速度
        return self._measure_speed(_train)

    # 内部方法，用于测量训练内存占用
    def _train_memory(
        self, model_name: str, batch_size: int, sequence_length: int
    ) -> [Memory, Optional[MemorySummary]]:
        # 准备训练函数
        _train = self._prepare_train_func(model_name, batch_size, sequence_length)
        # 测量内存占用
        return self._measure_memory(_train)
    # 准备推断函数，根据给定的模型名称、批量大小和序列长度
    def _prepare_inference_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        # 获取模型配置
        config = self.config_dict[model_name]

        # 如果启用了 TorchScript
        if self.args.torchscript:
            # 设置配置中的 TorchScript 为 True
            config.torchscript = True

        # 检查配置中是否包含模型类信息
        has_model_class_in_config = (
            hasattr(config, "architectures")
            and isinstance(config.architectures, list)
            and len(config.architectures) > 0
        )
        # 如果不仅仅是预训练模型并且配置中存在模型类信息
        if not self.args.only_pretrain_model and has_model_class_in_config:
            try:
                # 获取模型类
                model_class = config.architectures[0]
                # 动态导入 transformers 模块，并获取模型类
                transformers_module = __import__("transformers", fromlist=[model_class])
                model_cls = getattr(transformers_module, model_class)
                # 使用模型类和配置创建模型
                model = model_cls(config)
            except ImportError:
                # 如果导入失败，抛出 ImportError
                raise ImportError(
                    f"{model_class} does not exist. If you just want to test the pretrained model, you might want to"
                    " set `--only_pretrain_model` or `args.only_pretrain_model=True`."
                )
        else:
            # 如果仅使用预训练模型或者配置中不存在模型类信息，则根据配置中的类映射创建模型
            model = MODEL_MAPPING[config.__class__](config)

        # 将模型设置为评估模式
        model.eval()
        # 将模型移到指定设备上
        model.to(self.args.device)

        # 对于 encoder-decoder 模型，词汇表大小保存在不同的地方
        vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size
        # 生成指定大小的随机输入张量
        input_ids = torch.randint(vocab_size, (batch_size, sequence_length), dtype=torch.long, device=self.args.device)

        # 如果启用了混合精度
        if self.args.fp16:
            # 打印日志，提示使用混合精度进行训练
            logger.info("Running training in Mixed Precision...")
            # 如果不是 GPU，则抛出 ValueError
            if not self.args.is_gpu:
                raise ValueError("Mixed precision is possible only for GPU.")
            # 将模型转换为半精度
            model.half()

        # 如果启用了 TorchScript
        if self.args.torchscript:
            # 使用 torch.jit.trace 进行模型追踪
            with torch.no_grad():
                inference_model = torch.jit.trace(model, input_ids)
        else:
            # 否则使用原始模型
            inference_model = model

        # 定义编码器-解码器前向函数
        def encoder_decoder_forward():
            # 在不计算梯度的情况下进行推断
            with torch.no_grad():
                # 使用推断模型进行前向传播，同时提供解码器输入张量
                outputs = inference_model(input_ids, decoder_input_ids=input_ids)
            return outputs

        # 定义编码器前向函数
        def encoder_forward():
            # 在不计算梯度的情况下进行推断
            with torch.no_grad():
                # 使用推断模型进行编码器前向传播
                outputs = inference_model(input_ids)
            return outputs

        # 根据模型是否为编码器-解码器类型选择前向函数
        _forward = encoder_decoder_forward if config.is_encoder_decoder else encoder_forward
        # 返回选择的前向函数
        return _forward
    # 定义一个准备训练函数，接受模型名称、批量大小和序列长度作为参数，返回一个无参数函数
    def _prepare_train_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        # 从配置字典中获取给定模型名称的配置
        config = self.config_dict[model_name]

        # 检查配置中是否包含模型类，并且该模型类是列表中的第一个元素
        has_model_class_in_config = (
            hasattr(config, "architectures")
            and isinstance(config.architectures, list)
            and len(config.architectures) > 0
        )

        # 如果不是仅使用预训练模型，并且配置中包含模型类
        if not self.args.only_pretrain_model and has_model_class_in_config:
            try:
                # 获取模型类名称
                model_class = config.architectures[0]
                # 动态导入transformers模块并获取模型类
                transformers_module = __import__("transformers", fromlist=[model_class])
                model_cls = getattr(transformers_module, model_class)
                # 根据配置创建模型实例
                model = model_cls(config)
            except ImportError:
                # 抛出导入错误，说明模型类不存在
                raise ImportError(
                    f"{model_class} does not exist. If you just want to test the pretrained model, you might want to"
                    " set `--only_pretrain_model` or `args.only_pretrain_model=True`."
                )
        else:
            # 如果仅使用预训练模型或者配置中没有模型类，则使用默认的模型类
            model = MODEL_WITH_LM_HEAD_MAPPING[config.__class__](config)

        # 如果使用torchscript进行训练，抛出未实现错误
        if self.args.torchscript:
            raise NotImplementedError("Training for torchscript is currently not implemented")
        else:
            # 否则，将模型设置为训练模式
            train_model = model

        # 将模型移动到指定的设备（GPU或CPU）
        model.train()
        model.to(self.args.device)

        # 对于encoder-decoder模型，保存不同的词汇大小
        vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size
        # 生成指定大小的随机输入张量
        input_ids = torch.randint(vocab_size, (batch_size, sequence_length), dtype=torch.long, device=self.args.device)

        # 如果启用了混合精度训练
        if self.args.fp16:
            # 打印日志信息
            logger.info("Running training in Mixed Precision...")
            # 如果不是GPU，抛出错误
            if not self.args.is_gpu:
                raise ValueError("Mixed precision is possible only for GPU.")

            # 将模型转换为半精度浮点数类型
            model.half()

        # 定义计算损失和反向传播函数（仅编码器模型）
        def compute_loss_and_backprob_encoder():
            # 计算输入数据的损失并执行反向传播
            loss = train_model(input_ids, labels=input_ids)[0]
            loss.backward()
            return loss

        # 定义计算损失和反向传播函数（编码器-解码器模型）
        def compute_loss_and_backprob_encoder_decoder():
            # 计算输入数据的损失并执行反向传播
            loss = train_model(input_ids, decoder_input_ids=input_ids, labels=input_ids)[0]
            loss.backward()
            return loss

        # 根据模型是否为编码器-解码器模型选择不同的计算损失和反向传播函数
        _train = (
            compute_loss_and_backprob_encoder_decoder
            if config.is_encoder_decoder
            else compute_loss_and_backprob_encoder
        )
        # 返回选择的计算损失和反向传播函数
        return _train
    # 定义一个函数，用于测量给定函数的执行速度，并返回最小执行时间的十分之一
    def _measure_speed(self, func) -> float:
        try:
            # 如果是在 TPU 上执行或者开启了 torchscript 模式，则额外运行 5 次以稳定编译
            if self.args.is_tpu or self.args.torchscript:
                logger.info("Do inference on TPU or torchscript. Running model 5 times to stabilize compilation")
                timeit.repeat(
                    func,
                    repeat=1,
                    number=5,
                )

            # 根据参数设定的重复次数和执行次数，进行函数的计时运行，并返回各次运行时间的最小值
            runtimes = timeit.repeat(
                func,
                repeat=self.args.repeat,
                number=10,
            )

            # 如果同时在 TPU 上执行且设置了打印 TPU 指标，则导入 torch_xla.debug.metrics 并打印指标报告
            if self.args.is_tpu and self.args.torch_xla_tpu_print_metrics:
                import torch_xla.debug.metrics as met
                self.print_fn(met.metrics_report())

            # 返回最小执行时间的十分之一
            return min(runtimes) / 10.0
        # 捕获运行时错误，并打印错误信息
        except RuntimeError as e:
            self.print_fn(f"Doesn't fit on GPU. {e}")
            return "N/A"
    # 定义一个方法，用于测量内存使用情况
    def _measure_memory(self, func: Callable[[], None]) -> [Memory, MemorySummary]:
        try:
            # 如果需要逐行跟踪内存使用情况
            if self.args.trace_memory_line_by_line:
                # 开始跟踪内存使用情况
                trace = start_memory_tracing("transformers")

            # 如果是在 TPU 上运行
            if self.args.is_tpu:
                # 抛出未实现的错误
                raise NotImplementedError(
                    "Memory Benchmarking is currently not implemented for TPU. Please disable memory benchmarking with"
                    " `--no-memory` or `args.memory=False`"
                )
            # 如果是在 GPU 上运行
            elif self.args.is_gpu:
                # 如果没有安装 py3nvml
                if not is_py3nvml_available():
                    # 发出警告
                    logger.warning(
                        "py3nvml not installed, we won't log GPU memory usage. "
                        "Install py3nvml (pip install py3nvml) to log information about GPU."
                    )
                    # 设置内存为 N/A
                    memory = "N/A"
                else:
                    # 记录 GPU 内存使用情况
                    logger.info(
                        "Measuring total GPU usage on GPU device. Make sure to not have additional processes running"
                        " on the same GPU."
                    )
                    # 初始化 nvml
                    nvml.nvmlInit()
                    # 执行传入的函数
                    func()
                    # 获取 GPU 设备句柄
                    handle = nvml.nvmlDeviceGetHandleByIndex(self.args.device_idx)
                    # 获取 GPU 内存信息
                    meminfo = nvml.nvmlDeviceGetMemoryInfo(handle)
                    # 记录当前使用的最大字节数
                    max_bytes_in_use = meminfo.used
                    # 创建 Memory 对象
                    memory = Memory(max_bytes_in_use)
                    # 关闭 nvml
                    nvml.nvmlShutdown()
            # 如果是在 CPU 上运行
            else:
                # 测量 CPU 内存使用情况
                memory_bytes = measure_peak_memory_cpu(func)
                # 如果内存使用情况是整数类型，则创建 Memory 对象
                memory = Memory(memory_bytes) if isinstance(memory_bytes, int) else memory_bytes

            # 如果需要逐行跟踪内存使用情况
            if self.args.trace_memory_line_by_line:
                # 停止内存跟踪
                summary = stop_memory_tracing(trace)
            else:
                summary = None

            # 返回内存使用情况和摘要信息
            return memory, summary
        except RuntimeError as e:
            # 打印错误信息
            self.print_fn(f"Doesn't fit on GPU. {e}")
            return "N/A", None
```