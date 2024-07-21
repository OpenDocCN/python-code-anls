# `.\pytorch\torch\autograd\_functions\tensor.py`

```py
# mypy: allow-untyped-defs
# 引入operator模块中的函数reduce和模块functools中的函数reduce
import operator
from functools import reduce
# 引入typing_extensions模块中的deprecated装饰器
from typing_extensions import deprecated

# 引入PyTorch库
import torch
# 引入torch._utils模块
import torch._utils
# 从当前目录的上一级目录中引入function模块
from ..function import Function

# 定义一个类Type，继承自Function类
class Type(Function):
    # 声明静态方法forward，接收输入参数i和目标数据类型dest_type，ctx是上下文对象
    @staticmethod
    # 使用deprecated装饰器，给出警告信息，并标记为FutureWarning
    @deprecated(
        "`torch.autograd._functions.Type` is deprecated as of PyTorch 2.1, "
        "please use `torch.tensor.to(dtype=dtype)` instead.",
        category=FutureWarning,
    )
    # forward方法的实现
    def forward(ctx, i, dest_type):
        # 记录输入张量i的类型
        ctx.input_type = type(i)
        # 如果i不在GPU上，则设定输入设备为-1，否则设定为i所在GPU设备编号
        ctx.input_device = -1 if not i.is_cuda else i.get_device()
        # 返回将输入张量i转换为目标数据类型dest_type的结果
        return i.type(dest_type)

    # 声明静态方法backward，接收梯度输出grad_output和上下文对象ctx
    @staticmethod
    # backward方法的实现
    def backward(ctx, grad_output):
        # 如果输入设备为-1，则返回grad_output转换为ctx.input_type的结果和None
        if ctx.input_device == -1:
            return grad_output.type(ctx.input_type), None
        else:
            # 在指定的GPU设备上，返回grad_output转换为ctx.input_type的结果和None
            with torch.cuda.device(ctx.input_device):
                return grad_output.type(ctx.input_type), None


# TODO: deprecate this
# 定义一个类Resize，继承自Function类
class Resize(Function):
    # 声明静态方法forward，接收输入参数tensor和目标大小sizes，ctx是上下文对象
    @staticmethod
    # forward方法的实现
    def forward(ctx, tensor, sizes):
        # 记录目标大小sizes
        ctx.sizes = sizes
        # 记录输入张量tensor的元素个数
        ctx.numel = reduce(operator.mul, sizes, 1)
        # 如果输入张量tensor的元素个数与目标大小sizes的元素个数不匹配，则抛出异常
        if tensor.numel() != ctx.numel:
            raise RuntimeError(
                (
                    "requested resize to {} ({} elements in total), "
                    "but the given tensor has a size of {} ({} elements). "
                    "autograd's resize can only change the shape of a given "
                    "tensor, while preserving the number of elements. "
                ).format(
                    "x".join(map(str, sizes)),
                    ctx.numel,
                    "x".join(map(str, tensor.size())),
                    tensor.numel(),
                )
            )
        # 记录输入张量tensor的原始大小
        ctx.input_sizes = tensor.size()
        # 如果输入张量tensor是量化的，则复制并返回一个连续的视图
        if tensor.is_quantized:
            tensor.copy_(tensor)
            return tensor.contiguous().view(*sizes)
        # 如果输入张量tensor是连续的，则创建一个新的连续视图并返回
        if tensor.is_contiguous():
            result = tensor.new(tensor).contiguous().view(*sizes)
            return result
        else:
            # 否则返回一个连续的视图
            return tensor.contiguous().view(*sizes)

    # 声明静态方法backward，接收梯度输出grad_output和上下文对象ctx
    @staticmethod
    # backward方法的实现
    def backward(ctx, grad_output):
        # 断言grad_output的元素个数与ctx.numel相同
        assert grad_output.numel() == ctx.numel
        # 返回grad_output的连续视图，并返回None
        return grad_output.contiguous().view(ctx.input_sizes), None
```