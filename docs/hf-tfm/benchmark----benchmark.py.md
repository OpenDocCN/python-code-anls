# `.\benchmark\benchmark.py`

```
# coding=utf-8
# 声明文件编码格式为 UTF-8

# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 版权归 NVIDIA 公司所有，保留所有权利

# 根据 Apache 许可证 2.0 版本，除非符合许可证的要求，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0

# 如果适用法律要求或书面同意，本软件按 "原样" 分发，不提供任何明示或暗示的保证或条件
# 请参阅许可证，了解详细的法律规定

"""
    在 PyTorch 中对库进行推理和训练的基准测试。
"""

# 导入计时模块
import timeit
# 导入类型提示模块
from typing import Callable, Optional

# 导入配置工具模块
from ..configuration_utils import PretrainedConfig
# 导入模型映射和带语言模型头部的模型映射
from ..models.auto.modeling_auto import MODEL_MAPPING, MODEL_WITH_LM_HEAD_MAPPING
# 导入工具模块，包括检测 Py3nvml 和 Torch 是否可用，以及日志记录
from ..utils import is_py3nvml_available, is_torch_available, logging
# 导入基准测试工具模块，包括内存、内存摘要、CPU 最高内存测量、内存跟踪等
from .benchmark_utils import (
    Benchmark,
    Memory,
    MemorySummary,
    measure_peak_memory_cpu,
    start_memory_tracing,
    stop_memory_tracing,
)

# 如果 Torch 可用
if is_torch_available():
    # 导入 Torch 模块
    import torch

    # 导入 PyTorch 基准测试参数
    from .benchmark_args import PyTorchBenchmarkArguments

# 如果 Py3nvml 可用
if is_py3nvml_available():
    # 导入 Py3nvml 模块
    import py3nvml.py3nvml as nvml

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 PyTorch 基准测试类，继承自 Benchmark 类
class PyTorchBenchmark(Benchmark):
    # 声明 PyTorch 基准测试类的参数
    args: PyTorchBenchmarkArguments
    # 声明预训练配置
    configs: PretrainedConfig
    # 框架名称为 PyTorch
    framework: str = "PyTorch"

    # 框架版本属性，返回 Torch 的版本号
    @property
    def framework_version(self):
        return torch.__version__

    # 推理速度方法，返回推理的速度
    def _inference_speed(self, model_name: str, batch_size: int, sequence_length: int) -> float:
        # 准备推理函数
        _inference = self._prepare_inference_func(model_name, batch_size, sequence_length)
        # 测量推理速度
        return self._measure_speed(_inference)

    # 推理内存方法，返回内存占用
    def _inference_memory(
        self, model_name: str, batch_size: int, sequence_length: int
    ) -> [Memory, Optional[MemorySummary]]:
        # 准备推理函数
        _inference = self._prepare_inference_func(model_name, batch_size, sequence_length)
        # 测量内存占用
        return self._measure_memory(_inference)

    # 训练速度方法，返回训练的速度
    def _train_speed(self, model_name: str, batch_size: int, sequence_length: int) -> float:
        # 准备训练函数
        _train = self._prepare_train_func(model_name, batch_size, sequence_length)
        # 测量训练速度
        return self._measure_speed(_train)

    # 训练内存方法，返回内存占用
    def _train_memory(
        self, model_name: str, batch_size: int, sequence_length: int
    ) -> [Memory, Optional[MemorySummary]]:
        # 准备训练函数
        _train = self._prepare_train_func(model_name, batch_size, sequence_length)
        # 测量内存占用
        return self._measure_memory(_train)
    # 定义一个方法，用于准备推理函数，接受模型名称、批大小和序列长度作为参数，并返回一个无参数的回调函数
    def _prepare_inference_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        # 从配置字典中获取指定模型名称的配置信息
        config = self.config_dict[model_name]

        # 如果设置了 torchscript 标志，则将配置中的 torchscript 属性设置为 True
        if self.args.torchscript:
            config.torchscript = True

        # 检查配置中是否包含模型类信息，并且列表不为空
        has_model_class_in_config = (
            hasattr(config, "architectures")
            and isinstance(config.architectures, list)
            and len(config.architectures) > 0
        )

        # 如果不仅仅是预训练模型且配置中包含模型类信息，则尝试实例化指定的模型类
        if not self.args.only_pretrain_model and has_model_class_in_config:
            try:
                # 获取配置中的第一个模型类名称
                model_class = config.architectures[0]
                # 动态导入 transformers 模块，并从中获取指定的模型类
                transformers_module = __import__("transformers", fromlist=[model_class])
                model_cls = getattr(transformers_module, model_class)
                # 使用模型类和配置信息实例化模型
                model = model_cls(config)
            except ImportError:
                # 抛出 ImportError 如果指定的模型类不存在
                raise ImportError(
                    f"{model_class} does not exist. If you just want to test the pretrained model, you might want to"
                    " set `--only_pretrain_model` or `args.only_pretrain_model=True`."
                )
        else:
            # 根据配置中的类信息从 MODEL_MAPPING 中选择相应的模型，并实例化
            model = MODEL_MAPPING[config.__class__](config)

        # 将模型设置为评估模式
        model.eval()
        # 将模型移动到指定的设备上（GPU 或 CPU）
        model.to(self.args.device)

        # 对于 encoder-decoder 模型，词汇表大小可能会以不同方式保存
        vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size
        # 创建一个随机的输入张量 input_ids，形状为 (batch_size, sequence_length)，数据类型为长整型，放置在指定的设备上
        input_ids = torch.randint(vocab_size, (batch_size, sequence_length), dtype=torch.long, device=self.args.device)

        # 如果设置了 fp16 标志，则将模型转换为半精度浮点数运行
        if self.args.fp16:
            logger.info("Running training in Mixed Precision...")
            if not self.args.is_gpu:
                # 如果不是 GPU，抛出 ValueError，因为混合精度计算只支持 GPU
                raise ValueError("Mixed precision is possible only for GPU.")
            # 将模型转换为半精度浮点数
            model.half()

        # 如果设置了 torchscript 标志，则使用 torch.jit.trace 对模型进行跟踪
        if self.args.torchscript:
            with torch.no_grad():
                inference_model = torch.jit.trace(model, input_ids)
        else:
            # 否则，直接使用原始模型
            inference_model = model

        # 定义 encoder-decoder 模型和 encoder 模型的前向推理函数
        def encoder_decoder_forward():
            with torch.no_grad():
                # 对输入数据 input_ids 进行推理，同时提供 decoder_input_ids 作为输入
                outputs = inference_model(input_ids, decoder_input_ids=input_ids)
            return outputs

        def encoder_forward():
            with torch.no_grad():
                # 对输入数据 input_ids 进行推理
                outputs = inference_model(input_ids)
            return outputs

        # 根据配置信息中是否为 encoder-decoder 模型选择对应的推理函数
        _forward = encoder_decoder_forward if config.is_encoder_decoder else encoder_forward
        return _forward
    def _prepare_train_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        # 获取指定模型名称对应的配置信息
        config = self.config_dict[model_name]

        # 检查配置中是否包含模型类信息
        has_model_class_in_config = (
            hasattr(config, "architectures")
            and isinstance(config.architectures, list)
            and len(config.architectures) > 0
        )

        # 如果不仅仅是预训练模型，并且配置中包含模型类信息
        if not self.args.only_pretrain_model and has_model_class_in_config:
            try:
                # 从配置中获取模型类名
                model_class = config.architectures[0]
                # 动态导入 transformers 模块中的模型类
                transformers_module = __import__("transformers", fromlist=[model_class])
                model_cls = getattr(transformers_module, model_class)
                # 使用配置创建模型实例
                model = model_cls(config)
            except ImportError:
                # 抛出导入错误，指示模型类不存在
                raise ImportError(
                    f"{model_class} does not exist. If you just want to test the pretrained model, you might want to"
                    " set `--only_pretrain_model` or `args.only_pretrain_model=True`."
                )
        else:
            # 如果仅使用预定义的语言模型头部映射来创建模型
            model = MODEL_WITH_LM_HEAD_MAPPING[config.__class__](config)

        # 如果设置了 torchscript，目前还未实现 torchscript 的训练
        if self.args.torchscript:
            raise NotImplementedError("Training for torchscript is currently not implemented")
        else:
            # 将模型设置为训练模式
            train_model = model

        # 将模型切换到指定的设备（GPU 或 CPU）
        model.train()
        model.to(self.args.device)

        # 对于 encoder-decoder 模型，词汇表大小可能以不同方式保存
        vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size
        # 生成随机输入 ID，形状为 (batch_size, sequence_length)，数据类型为 long，放置在指定设备上
        input_ids = torch.randint(vocab_size, (batch_size, sequence_length), dtype=torch.long, device=self.args.device)

        # 如果启用混合精度训练
        if self.args.fp16:
            logger.info("Running training in Mixed Precision...")
            if not self.args.is_gpu:
                # 如果不是 GPU，不能使用混合精度
                raise ValueError("Mixed precision is possible only for GPU.")

            # 使用半精度浮点数进行训练，以减少内存使用
            model.half()

        # 定义计算损失和反向传播的函数，针对 encoder 模型
        def compute_loss_and_backprob_encoder():
            loss = train_model(input_ids, labels=input_ids)[0]
            loss.backward()
            return loss

        # 定义计算损失和反向传播的函数，针对 encoder-decoder 模型
        def compute_loss_and_backprob_encoder_decoder():
            loss = train_model(input_ids, decoder_input_ids=input_ids, labels=input_ids)[0]
            loss.backward()
            return loss

        # 根据配置是否为 encoder-decoder 模型，选择不同的训练函数
        _train = (
            compute_loss_and_backprob_encoder_decoder
            if config.is_encoder_decoder
            else compute_loss_and_backprob_encoder
        )
        return _train
    # 定义一个方法，用于测量函数执行速度，返回一个浮点数表示执行时间
    def _measure_speed(self, func) -> float:
        try:
            # 如果使用 TPU 或者需要 torchscript 编译，额外运行10次以稳定编译过程
            logger.info("Do inference on TPU or torchscript. Running model 5 times to stabilize compilation")
            timeit.repeat(
                func,
                repeat=1,
                number=5,
            )

            # 根据参数设定重复运行 func 函数，记录运行时间
            runtimes = timeit.repeat(
                func,
                repeat=self.args.repeat,
                number=10,
            )

            # 如果使用 TPU 并且开启了 torch_xla_tpu_print_metrics，则打印性能指标
            if self.args.is_tpu and self.args.torch_xla_tpu_print_metrics:
                import torch_xla.debug.metrics as met

                self.print_fn(met.metrics_report())

            # 返回最小运行时间除以10的结果，以获得平均每次运行的时间
            return min(runtimes) / 10.0
        except RuntimeError as e:
            # 如果运行时出现异常，打印错误信息并返回 "N/A"
            self.print_fn(f"Doesn't fit on GPU. {e}")
            return "N/A"
    # 定义一个方法 `_measure_memory`，接收一个不接受参数并不返回任何内容的函数作为参数，
    # 返回一个元组，包含一个 `Memory` 对象和一个 `MemorySummary` 对象
    def _measure_memory(self, func: Callable[[], None]) -> [Memory, MemorySummary]:
        try:
            # 如果设置了逐行追踪内存使用情况
            if self.args.trace_memory_line_by_line:
                # 启动以 `transformers` 为标识的内存追踪
                trace = start_memory_tracing("transformers")

            # 如果程序运行在 TPU 上
            if self.args.is_tpu:
                # 抛出未实现错误，因为目前尚未实现 TPU 的内存基准测试
                raise NotImplementedError(
                    "Memory Benchmarking is currently not implemented for TPU. Please disable memory benchmarking with"
                    " `--no-memory` or `args.memory=False`"
                )
            # 如果程序运行在 GPU 上
            elif self.args.is_gpu:
                # 如果没有安装 py3nvml 库
                if not is_py3nvml_available():
                    # 发出警告，提示未安装 py3nvml 库，无法记录 GPU 内存使用情况
                    logger.warning(
                        "py3nvml not installed, we won't log GPU memory usage. "
                        "Install py3nvml (pip install py3nvml) to log information about GPU."
                    )
                    # 将 memory 设为字符串 "N/A"
                    memory = "N/A"
                else:
                    # 记录日志，提示正在测量 GPU 设备的总体使用情况
                    logger.info(
                        "Measuring total GPU usage on GPU device. Make sure to not have additional processes running"
                        " on the same GPU."
                    )
                    # 初始化 nvml 库
                    nvml.nvmlInit()
                    # 执行传入的函数 func
                    func()
                    # 获取指定索引的 GPU 设备句柄
                    handle = nvml.nvmlDeviceGetHandleByIndex(self.args.device_idx)
                    # 获取 GPU 设备的内存信息
                    meminfo = nvml.nvmlDeviceGetMemoryInfo(handle)
                    # 获取已使用的最大字节数
                    max_bytes_in_use = meminfo.used
                    # 创建 Memory 对象，表示已使用的最大字节数
                    memory = Memory(max_bytes_in_use)
                    # 关闭 nvml 库
                    nvml.nvmlShutdown()
            # 如果程序运行在 CPU 上
            else:
                # 测量 CPU 的峰值内存使用情况
                memory_bytes = measure_peak_memory_cpu(func)
                # 如果 memory_bytes 是整数，则创建 Memory 对象，表示测量到的内存字节数，否则直接使用 memory_bytes
                memory = Memory(memory_bytes) if isinstance(memory_bytes, int) else memory_bytes

            # 如果设置了逐行追踪内存使用情况
            if self.args.trace_memory_line_by_line:
                # 停止内存追踪，并获取追踪结果的汇总信息
                summary = stop_memory_tracing(trace)
            else:
                # 否则，汇总信息设为 None
                summary = None

            # 返回内存对象和汇总信息对象的元组
            return memory, summary
        # 捕获 RuntimeError 异常
        except RuntimeError as e:
            # 打印异常信息，指出 GPU 不适合执行当前任务
            self.print_fn(f"Doesn't fit on GPU. {e}")
            # 返回 "N/A" 表示不适合 GPU 执行
            return "N/A", None
```