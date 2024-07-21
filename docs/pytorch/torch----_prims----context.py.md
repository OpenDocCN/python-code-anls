# `.\pytorch\torch\_prims\context.py`

```
# 指定允许未类型化的函数定义，用于类型检查器mypy
# 导入functools模块，用于LRU缓存功能
import functools
# 导入nullcontext，提供一个空的上下文管理器
from contextlib import nullcontext
# 导入Any、Callable、Dict、Optional和Sequence类型
from typing import Any, Callable, Dict, Optional, Sequence

# 导入torch库
import torch

# 导入torch的_decomp和_prims子模块
import torch._decomp
import torch._prims

# 导入torch的_refs和其下的子模块nn、nn.functional、special
import torch._refs
import torch._refs.nn
import torch._refs.nn.functional
import torch._refs.special

# 导入torch的overrides模块
import torch.overrides

# 从torch._prims_common导入torch_function_passthrough函数
from torch._prims_common import torch_function_passthrough


@functools.lru_cache(None)
# 定义函数torch_to_refs_map，返回一个字典，将torch API函数映射到torch._refs函数
def torch_to_refs_map():
    """
    Mapping of torch API functions to torch._refs functions.
    E.g. torch_to_refs_map()[torch.add] == torch._refs.add
    """
    # 定义模块列表，将torch模块和对应的torch._refs子模块组合成元组
    modules = [
        (torch, torch._refs),
        (torch.nn, torch._refs.nn),
        (torch.nn.functional, torch._refs.nn.functional),
        (torch.special, torch._refs.special),
        (torch.fft, torch._refs.fft),
        (torch.linalg, torch._refs.linalg),
    ]
    # 创建一个空字典r
    r: Dict[Any, Any] = {
        # 将torch.Tensor的位反转映射到torch._refs的bitwise_not函数
        torch.Tensor.__invert__: torch._refs.bitwise_not,
        # 将torch.Tensor的位异或映射到torch._refs的bitwise_xor函数
        torch.Tensor.__xor__: torch._refs.bitwise_xor,
        # 将torch.Tensor的位与映射到torch._refs的bitwise_and函数
        torch.Tensor.__and__: torch._refs.bitwise_and,
        # 将torch.Tensor的位或映射到torch._refs的bitwise_or函数
        torch.Tensor.__or__: torch._refs.bitwise_or,
        # 将torch.Tensor的相等性比较映射到torch._refs的eq函数
        torch.Tensor.__eq__: torch._refs.eq,
        # 将torch.Tensor的右减法映射到torch._refs的rsub函数
        torch.Tensor.__rsub__: torch._refs.rsub,
        # 将torch.Tensor的右除法映射到torch._refs的rtruediv函数
        torch.Tensor.__rtruediv__: torch._refs.rtruediv,
        # 将torch.Tensor的整除映射到torch._refs的floor_divide函数
        torch.Tensor.__floordiv__: torch._refs.floor_divide,
        # 将torch.Tensor的右整除映射到torch._refs的rfloordiv函数
        torch.Tensor.__rfloordiv__: torch._refs.rfloordiv,
        # 将torch.Tensor的幂运算映射到torch._refs的pow函数
        torch.Tensor.__pow__: torch._refs.pow,
        # 将torch.Tensor的右幂运算映射到torch._refs的rpow函数
        torch.Tensor.__rpow__: torch._refs.rpow,
        # 将torch.Tensor的新建空张量映射到torch._refs的new_empty函数
        torch.Tensor.new_empty: torch._refs.new_empty,
        # 将torch.Tensor的新建全1张量映射到torch._refs的new_full函数
        torch.Tensor.new_full: torch._refs.new_full,
        # 将torch.Tensor的新建全0张量映射到torch._refs的new_zeros函数
        torch.Tensor.new_zeros: torch._refs.new_zeros,
        # 将torch.Tensor的新建全1张量映射到torch._refs的new_ones函数
        torch.Tensor.new_ones: torch._refs.new_ones,
        # 将torch.Tensor的填充操作映射到torch._refs的fill_函数
        torch.Tensor.fill_: torch._refs.fill_,
        # 将torch.Tensor的零填充映射到torch._refs的zero_函数
        torch.Tensor.zero_: torch._refs.zero_,
        # 将torch.Tensor的类型转换映射到torch._refs的to函数
        torch.Tensor.to: torch._refs.to,
        # 将torch.Tensor的大小调整到指定大小映射到torch._refs的sum_to_size函数
        torch.Tensor.sum_to_size: torch._refs.sum_to_size,
        # TODO: 这些方法是否应以其他方式映射？
        # 将torch.Tensor的复制操作映射到torch._prims的copy_to函数
        torch.Tensor.copy_: torch._prims.copy_to,
        # 将torch.Tensor的调整大小映射到torch._prims的resize函数
        torch.Tensor.resize: torch._prims.resize,
    }
    # 遍历模块列表，将每个模块的函数添加到字典r中
    for mod_torch, mod_refs in modules:
        for s in mod_refs.__all__:  # type: ignore[attr-defined]
            r[mod_torch.__dict__.get(s)] = mod_refs.__dict__.get(s)

    # 支持将torch.Tensor.foo映射到_refs.foo
    for s in dir(torch.Tensor):
        if s in torch._refs.__all__:
            r[getattr(torch.Tensor, s)] = torch._refs.__dict__.get(s)

    # 支持转换操作
    for s in torch._refs._conversions.__all__:
        tensor_attr = getattr(torch.Tensor, s, None) or getattr(torch, s)
        r[tensor_attr] = torch._refs._conversions.__dict__.get(s)

    return r


@functools.lru_cache(None)
# 定义函数all_prims，返回所有原始函数的集合，例如在all_prims()中返回torch._prims.add
def all_prims():
    """
    Set of all prim functions, e.g., torch._prims.add in all_prims()
    """
    return {torch._prims.__dict__.get(s) for s in torch._prims.__all__}


# 定义TorchRefsMode类，继承自torch.overrides.TorchFunctionMode
class TorchRefsMode(torch.overrides.TorchFunctionMode):
    """
    Switches the interpretation of torch.* functions and Tensor methods to
    # 设置用于控制是否严格模式的初始化方法
    def __init__(
        self,
        strict=False,  # 是否启用严格模式，默认为False
        should_fallback_fn=lambda *_: False,  # 控制是否回退到torch.*的函数，默认为不回退
        prims_mode_cls=nullcontext,  # 控制是否启用primitive模式的上下文管理器，默认为nullcontext
    ):
        self.strict = strict  # 将传入的strict参数保存到实例变量self.strict中
        self.should_fallback_fn = should_fallback_fn  # 将传入的should_fallback_fn保存到实例变量self.should_fallback_fn中
        self.prims_mode_cls = prims_mode_cls  # 将传入的prims_mode_cls保存到实例变量self.prims_mode_cls中

    # 对torch函数的重载方法，用于处理torch函数的调用
    def __torch_function__(
        self,
        orig_func: Callable,  # 原始调用的torch函数
        types: Sequence,  # 参数的类型序列
        args: Sequence[Any] = (),  # 传递给函数的位置参数序列，默认为空
        kwargs: Optional[Dict] = None,  # 传递给函数的关键字参数字典，可选，默认为None
    ):
        if kwargs is None:
            kwargs = {}
        
        # 对于原始操作，直接执行而不拦截
        # 除非我们在prims_mode下，这时我们希望使用nvprims
        if orig_func in torch_function_passthrough or orig_func in all_prims():
            with self.prims_mode_cls():  # 使用self.prims_mode_cls上下文管理器
                return orig_func(*args, **kwargs)
        
        mapping = torch_to_refs_map()  # 获取torch函数到引用函数的映射
        func = mapping.get(orig_func, None)  # 获取orig_func对应的引用函数，如果没有则为None

        # 对于torch.ops.aten.*，使用从torch._decomp.decomposition_table中注册的分解
        # torch._decomp.decomposition_table提供了从torch.ops.aten.*到torch._refs或torch._decomp.decompositions的映射
        if func is None and isinstance(orig_func, torch._ops.OpOverload):
            func = torch._decomp.decomposition_table.get(orig_func, None)
        
        if func is not None:
            # 如果引用函数存在，则查询是否应该使用它
            if self.should_fallback_fn(self, orig_func, func, args, kwargs):
                return orig_func(*args, **kwargs)
            # 在引用函数存在时，torch内部调用应被解释为引用调用
            with self:  # 使用当前对象的上下文管理器
                return func(*args, **kwargs)
        
        if self.strict:
            # 如果启用了严格模式但没有找到对应的_refs支持，抛出运行时错误
            raise RuntimeError(
                f"no _refs support for {torch.overrides.resolve_name(orig_func)}"
            )
        
        return orig_func(*args, **kwargs)  # 默认情况下直接调用原始函数
```