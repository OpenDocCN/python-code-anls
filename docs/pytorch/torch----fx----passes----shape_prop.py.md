# `.\pytorch\torch\fx\passes\shape_prop.py`

```py
# 忽略 mypy 类型检查错误
mypy: ignore-errors

# 导入 PyTorch 库及相关模块
import torch
import torch.fx
import traceback

# 导入用于 Python 分发的模块
from torch._dispatch.python import enable_python_dispatcher

# 导入 FX 图节点相关模块
from torch.fx.node import Node, map_aggregate

# 导入类型提示相关模块
from typing import Any, Tuple, NamedTuple, Optional, Dict

# 导入向后兼容性相关模块
from torch.fx._compatibility import compatibility

# 导入检测假模式相关模块
from torch._guards import detect_fake_mode

# 定义公开的类名列表
__all__ = ['TensorMetadata', 'ShapeProp']

# 定义向后兼容的 TensorMetadata 命名元组
@compatibility(is_backward_compatible=True)
class TensorMetadata(NamedTuple):
    """
    TensorMetadata 结构体包含 PyTorch 程序中关于张量的重要信息。
    """

    # 通用的张量元数据
    shape: torch.Size
    dtype: torch.dtype
    requires_grad: bool
    stride: Tuple[int, ...]
    memory_format: Optional[torch.memory_format]

    # 量化元数据
    is_quantized: bool
    qparams: Dict[str, Any]

# 定义函数用于提取张量的元数据信息并返回一个 TensorMetadata 结构体
def _extract_tensor_metadata(result: torch.Tensor, include_contiguity=True) -> TensorMetadata:
    """
    提取描述 `result` 的 TensorMetadata 命名元组。
    """
    shape = result.shape  # 获取张量的形状信息
    dtype = result.dtype  # 获取张量的数据类型信息
    requires_grad = result.requires_grad  # 获取张量的梯度需求信息
    stride = result.stride()  # 获取张量的步幅信息

    memory_format = None  # 初始化内存格式信息为 None

    # 如果包含内存连续性信息，检查不同的内存格式
    if include_contiguity:
        memory_formats = {
            torch.contiguous_format,
            torch.channels_last,
            torch.channels_last_3d,
        }
        for query_format in memory_formats:
            if result.is_contiguous(memory_format=query_format):
                memory_format = query_format
                break

    is_quantized = result.is_quantized  # 获取张量是否量化的信息
    qparams: Dict[str, Any] = {}  # 初始化量化参数字典

    # 如果张量量化，进一步获取量化参数
    if is_quantized:
        qscheme = result.qscheme()
        qparams["qscheme"] = qscheme
        if qscheme in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            qparams["scale"] = result.q_scale()  # 获取量化缩放因子
            qparams["zero_point"] = result.q_zero_point()  # 获取量化零点
        elif qscheme in {
            torch.per_channel_affine,
            torch.per_channel_affine_float_qparams,
            torch.per_channel_symmetric,
        }:
            # 对于每通道量化的情况，存储 scale 和 zero_point 作为不可变列表
            qparams["scale"] = result.q_per_channel_scales().tolist()  # 获取每通道量化的缩放因子列表
            qparams["zero_point"] = result.q_per_channel_zero_points().tolist()  # 获取每通道量化的零点列表
            qparams["axis"] = result.q_per_channel_axis()  # 获取每通道量化的轴信息

    # 返回 TensorMetadata 结构体
    return TensorMetadata(
        shape, dtype, requires_grad, stride, memory_format, is_quantized, qparams
    )

# 定义向后兼容的 ShapeProp 类，继承自 torch.fx.Interpreter 类
@compatibility(is_backward_compatible=True)
class ShapeProp(torch.fx.Interpreter):
    """
    逐节点执行 FX 图并记录结果的形状和类型到相应节点中。
    """
    pass  # 类体为空，只有文档字符串提供了类的描述
    def __init__(self, gm, fake_mode=None):
        # 调用父类的构造函数初始化对象
        super().__init__(gm)
        
        # 检查是否传入了 fake_mode 参数，如果没有则使用默认的检测方式
        if fake_mode is None:
            fake_mode = detect_fake_mode()
        
        # 如果检测到了 fake_mode，则从 torch._dynamo.utils 模块导入函数 deepcopy_to_fake_tensor
        if fake_mode is not None:
            from torch._dynamo.utils import deepcopy_to_fake_tensor
            
            # 注意:
            # 我们需要进行伪执行，因为输入是伪造的，但我们不能对模块进行伪装，
            # 因为我们需要写入真实模块的 tensor_meta。因此，我们先进行伪装操作以生成结果(L131下面)，
            # 以提取张量元数据，然后继续执行。
            #
            # 如果我们进行了伪装，我们会将数据写入错误的节点，然后下游的融合过程会缺少 tensor_meta。
            #
            # 请查看 torch/_inductor/overrides.py，这是在融合的上游位置被调用的地方。
            
            # 使用 deepcopy_to_fake_tensor 函数对模块进行伪装，以处理 fake_mode
            self.fake_module = deepcopy_to_fake_tensor(self.module, fake_mode)
            self.fake_mode = fake_mode
        else:
            # 如果没有检测到 fake_mode，则将 fake_module 和 fake_mode 设置为 None
            self.fake_module = None
            self.fake_mode = None
        
        # 将真实的模块对象保存在 real_module 中
        self.real_module = self.module
    # 运行给定节点 `n`，返回执行结果
    def run_node(self, n: Node) -> Any:
        try:
            if self.fake_module is not None:
                # 如果存在 `fake_module`，则暂时替换当前模块 `module` 为 `fake_module`
                # 这是一种权宜之计。作为替代方案，我们也可以通过重写 `call_module` 和 `get_attr` 方法来实现。
                self.module = self.fake_module
            try:
                if self.fake_mode is not None:
                    # 如果存在 `fake_mode`，则在 `fake_mode` 的上下文中启用 Python 调度器，运行节点 `n`
                    with self.fake_mode, enable_python_dispatcher():
                        result = super().run_node(n)
                else:
                    # 否则直接运行节点 `n`
                    result = super().run_node(n)
            finally:
                # 恢复真实的模块 `module`
                self.module = self.real_module
        except Exception as e:
            # 捕获异常并打印堆栈信息
            traceback.print_exc()
            # 抛出运行时错误，提示出错的节点信息和元数据
            raise RuntimeError(
                f"ShapeProp error for: node={n.format_node()} with "
                f"meta={n.meta}"
            ) from e

        found_tensor = False

        def extract_tensor_meta(obj):
            # 如果 `obj` 是 `torch.Tensor` 类型，则提取其元数据
            if isinstance(obj, torch.Tensor):
                nonlocal found_tensor
                found_tensor = True
                return _extract_tensor_metadata(obj)
            else:
                return obj

        # 将 `result` 中的每个对象应用 `extract_tensor_meta` 函数，获取所有 tensor 的元数据
        meta = map_aggregate(result, extract_tensor_meta)
        # 如果找到了 tensor，则将其元数据存储在节点的 `meta` 中
        if found_tensor:
            n.meta['tensor_meta'] = meta

        # 记录结果的类型到节点的 `meta` 中
        n.meta['type'] = type(result)
        # 返回执行结果 `result`
        return result

    def propagate(self, *args):
        """
        通过解释运行 `module`，返回结果并记录每个节点的形状和类型。

        Args:
            *args (Tensor): 输入样本.

        Returns:
            Any: 执行模块后返回的值
        """
        if self.fake_mode is not None:
            # 如果存在 `fake_mode`，则将输入参数中的每个 `torch.Tensor` 转换为 `fake_mode` 中的对应类型
            fake_args = [self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t for t in args]
        else:
            # 否则直接使用原始输入参数 `args`
            fake_args = args
        # 调用父类的 `run` 方法执行模块，并返回结果
        return super().run(*fake_args)
```