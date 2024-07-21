# `.\pytorch\torch\_dynamo\backends\common.py`

```py
# 忽略类型检查错误，这通常用于在类型检查工具（如mypy）中禁用特定的错误报告
# 模块导入：导入需要的Python模块
import contextlib  # 上下文管理器，用于创建轻量级的上下文管理器
import functools  # 函数装饰器和其他高阶功能的工具
import logging  # Python的日志记录系统
from unittest.mock import patch  # 用于在单元测试中模拟对象的标准库导入

import torch  # PyTorch深度学习框架
from torch._dynamo import disable  # TorchDynamo中用于禁用特定功能的函数
from torch._dynamo.utils import counters, defake, flatten_graph_inputs  # TorchDynamo的一些实用工具函数
from torch._functorch.aot_autograd import aot_module_simplified  # 异步运算图自动微分的简化模块
from torch.utils._python_dispatch import _disable_current_modes  # Python分发机制中用于禁用当前模式的函数

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class AotAutograd:
    def __init__(self, **kwargs):
        self.__name__ = "compiler_fn"  # 设置对象的特殊名称属性
        self.kwargs = kwargs  # 初始化对象的关键字参数

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        # 如果示例输入包含列表、元组或字典中的任何一种，将其展平以便处理
        if any(isinstance(x, (list, tuple, dict)) for x in example_inputs):
            return flatten_graph_inputs(
                gm,
                example_inputs,
                self,
            )

        # 用于解决与aot_eager_decomp_partition循环导入问题的hack

        if callable(self.kwargs.get("decompositions")):
            self.kwargs["decompositions"] = self.kwargs["decompositions"]()

        # 注意：不要删除计数器的增加操作
        counters["aot_autograd"]["total"] += 1
        use_fallback = False

        if use_fallback:
            # 如果需要回退，则记录调试信息并增加相应计数器
            log.debug("Unable to use AOT Autograd because graph has mutation")
            counters["aot_autograd"]["not_ok"] += 1
            return gm

        # 尝试进行编译

        def _wrapped_bw_compiler(*args, **kwargs):
            # 阻止TorchDynamo尝试编译生成的反向传播
            return disable(disable(bw_compiler)(*args, **kwargs))

        bw_compiler = self.kwargs.get("bw_compiler") or self.kwargs["fw_compiler"]
        self.kwargs["bw_compiler"] = _wrapped_bw_compiler
        self.kwargs["inference_compiler"] = (
            self.kwargs.get("inference_compiler") or self.kwargs["fw_compiler"]
        )

        from functorch.compile import nop  # 导入编译时的空操作

        from torch._inductor.debug import enable_aot_logging  # 启用AOT日志记录的调试模块

        # 调试断言会显著降低编译时间，因此仅在使用aot_eager后端时默认启用它们
        if self.kwargs.get("fw_compiler", None) == nop:
            patch_config = patch("functorch.compile.config.debug_assert", True)
        else:
            patch_config = contextlib.nullcontext()

        try:
            # 注意：此处未克隆！
            with enable_aot_logging(), patch_config:
                cg = aot_module_simplified(gm, example_inputs, **self.kwargs)
                counters["aot_autograd"]["ok"] += 1
                return disable(cg)
        except Exception:
            counters["aot_autograd"]["not_ok"] += 1
            raise


def aot_autograd(**kwargs):
    return AotAutograd(**kwargs)  # 返回AotAutograd类的实例化对象，传递关键字参数


def mem_efficient_fusion_kwargs(use_decomps):
    from functorch.compile import (
        default_decompositions,  # 导入默认的分解策略
        min_cut_rematerialization_partition,  # 导入最小割重建分区策略
        ts_compile,  # 导入Tensor编译策略
    )
    # 创建一个关键字参数字典 kwargs，用于传递给函数或方法
    kwargs = {
        # 使用内存高效融合函数 memory_efficient_fusion() 的结果作为 "fw_compiler" 参数
        "fw_compiler": ts_compile,
        # 使用内存高效融合函数 memory_efficient_fusion() 的结果作为 "bw_compiler" 参数
        "bw_compiler": ts_compile,
        # 使用最小割重组分区函数 min_cut_rematerialization_partition 作为 "partition_fn" 参数
        "partition_fn": min_cut_rematerialization_partition,
    }
    
    # 如果 use_decomps 为真，则添加默认的分解方法 default_decompositions 到 kwargs 中
    if use_decomps:
        kwargs["decompositions"] = default_decompositions
    
    # 返回填充后的 kwargs 字典作为函数的结果
    return kwargs
# 用于不支持虚拟张量的后端的装饰器。它将虚拟张量替换为零张量。
def fake_tensor_unsupported(fn):
    # 使用 functools.wraps 来确保 wrapper 函数与原始函数具有相同的元数据
    @functools.wraps(fn)
    # 包装器函数，接受模型、输入和额外的关键字参数
    def wrapper(model, inputs, **kwargs):
        # 使用 _disable_current_modes() 上下文管理器禁用当前模式
        with _disable_current_modes():
            # 将 inputs 中的每个元素映射为其 defake 版本（假设是一个函数）
            inputs = list(map(defake, inputs))
            # 调用原始函数 fn，传入模型、处理后的输入和其他关键字参数
            return fn(model, inputs, **kwargs)

    # 返回包装后的函数
    return wrapper


# 从示例输入中获取设备信息的函数，返回一个 torch.device 对象
def device_from_inputs(example_inputs) -> torch.device:
    # 遍历示例输入的元素
    for x in example_inputs:
        # 如果当前元素 x 具有 "device" 属性
        if hasattr(x, "device"):
            # 返回当前元素 x 的设备信息（torch.device 对象）
            return x.device


# 从示例输入中获取数据类型信息的函数，返回一个 torch.dtype 对象
def dtype_from_inputs(example_inputs) -> torch.dtype:
    # 遍历示例输入的元素
    for x in example_inputs:
        # 如果当前元素 x 具有 "dtype" 属性
        if hasattr(x, "dtype"):
            # 返回当前元素 x 的数据类型信息（torch.dtype 对象）
            return x.dtype
```