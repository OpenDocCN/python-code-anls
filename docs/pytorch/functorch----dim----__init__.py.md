# `.\pytorch\functorch\dim\__init__.py`

```py
import dis
import inspect
from typing import Sequence, Union

import functorch._C  # 导入functorch._C模块

import torch  # 导入torch模块
from functorch._C import dim as _C  # 从functorch._C模块导入dim并命名为_C
from .tree_map import tree_flatten, tree_map  # 导入当前目录下的tree_flatten和tree_map模块
from .wrap_type import wrap_type  # 导入当前目录下的wrap_type模块

_C._patch_tensor_class()  # 调用_C模块中的_patch_tensor_class函数

# 导入op_properties模块中的内容，并使用字典存储在pointwise变量中，避免为集合编写C++绑定
pointwise = dict.fromkeys(op_properties.pointwise, True)

use_c = True  # 设置use_c变量为True
if not use_c:  # 如果use_c为False
    from . import reference  # 从当前目录下导入reference模块

class _Tensor:
    # 快速路径，用于简单查询实现中绕过缓慢的封装/解封装逻辑...
    
    @property
    def dims(self):
        return tuple(d for d in self._levels if isinstance(d, Dim))  # 返回包含所有维度的元组
    
    def dim(self):
        return self.ndim  # 返回张量的维度数

    if use_c:
        __torch_function__ = classmethod(_C.__torch_function__)  # 如果use_c为True，则使用_C模块中的__torch_function__方法
        expand = _C._instancemethod(_C.expand)  # 如果use_c为True，则使用_C模块中的expand方法
    else:
        __torch_function__ = reference.__torch_function__  # 如果use_c为False，则使用reference模块中的__torch_function__方法
        expand = reference.expand  # 如果use_c为False，则使用reference模块中的expand方法

    index = _C._instancemethod(_C.index)  # 使用_C模块中的index方法

    def __repr__(self):
        tensor, levels, ndim = self._tensor, self._levels, self.ndim
        return f"{tensor}\nwith dims={tuple(l + ndim if isinstance(l, int) else l for l in levels)} sizes={tuple(tensor.size())}"  # 返回张量的字符串表示形式

TensorLike = (_Tensor, torch.Tensor)  # 定义TensorLike为_Tensor和torch.Tensor的元组

class Dim(_C.Dim, _Tensor):
    # 注意_C.Dim位于_Tensor之前，因为我们希望像size这样的Dim API优先于Tensor定义的格式化，但我们希望用特殊格式打印Dims
    __format__ = object.__format__  # 设置__format__属性为object.__format__

class Tensor(_Tensor, _C.Tensor):
    if not use_c:
        from_batched = staticmethod(_C.Tensor_from_batched)  # 如果use_c为False，则使用_C模块中的Tensor_from_batched方法
    from_positional = staticmethod(_C.Tensor_from_positional)  # 使用_C模块中的Tensor_from_positional方法
    sum = _C._instancemethod(_C.Tensor_sum)  # 使用_C模块中的Tensor_sum方法

def cat(tensors, dim, new_dim):
    n = dims()  # 调用dims函数，返回dims对象
    return stack(tensors, n, dim).index([n, dim], new_dim)  # 返回使用stack函数对张量进行堆叠后的索引结果

if use_c:
    _wrap = _C._wrap  # 如果use_c为True，则使用_C模块中的_wrap方法

    def _def(name, *args, **kwargs):
        orig = getattr(torch.Tensor, name)  # 获取torch.Tensor中的指定名称的属性或方法
        setattr(_Tensor, name, _C._instancemethod(_wrap(orig, *args, **kwargs)))  # 设置_Tensor类中的属性或方法为使用_wrap包装后的orig属性或方法

    t__getitem__ = _C._instancemethod(_C.__getitem__)  # 使用_C模块中的__getitem__方法
    stack = _C.stack  # 使用_C模块中的stack方法
    split = _C._instancemethod(_C.split)  # 使用_C模块中的split方法
else:
    _wrap, _def = reference._wrap, reference._def  # 如果use_c为False，则使用reference模块中的_wrap和_def方法
    t__getitem__ = reference.t__getitem__  # 如果use_c为False，则使用reference模块中的t__getitem__方法
    stack = reference.stack  # 如果use_c为False，则使用reference模块中的stack方法
    split = reference.split  # 如果use_c为False，则使用reference模块中的split方法

t__setitem__ = _C._instancemethod(_C.__setitem__)  # 使用_C模块中的__setitem__方法
_Tensor.__getitem__ = t__getitem__  # 设置_Tensor类的__getitem__方法为t__getitem__
_Tensor.__setitem__ = t__setitem__  # 设置_Tensor类的__setitem__方法为t__setitem__

torch.Tensor.split = split  # 设置torch.Tensor的split方法为split
_Tensor.split = split  # 设置_Tensor类的split方法为split
torch.Tensor.expand = _C._instancemethod(_C.expand)  # 设置torch.Tensor的expand方法为_C模块中的expand方法
torch.Tensor.index = _C._instancemethod(_C.index)  # 设置torch.Tensor的index方法为_C模块中的index方法
# 根据 use_c 变量选择性地设置 _Tensor 类的属性或方法
wrap_type(use_c, _Tensor, torch.Tensor, _Tensor.__torch_function__)
# 删除 _Tensor 类的 ndim 属性
del _Tensor.ndim

# 如果 use_c 为 True，则设置 _Tensor 类的 order 属性为 _C.order 方法
if use_c:
    _Tensor.order = _C._instancemethod(_C.order)
# 否则，设置 _Tensor 类的 order 属性为 reference.positional
else:
    _Tensor.order = reference.positional

# 定义一系列函数的装饰器 _def，每个函数名作为参数
_def("mean")
_def("sum")
_def("all")
_def("amax")
_def("amin")
_def("aminmax")
_def("any")
_def("count_nonzero")
_def("logsumexp")
_def("nanmean")
_def("nansum")
_def("prod")
_def("std", keepdim_offset=2)
_def("var", keepdim_offset=2)
_def("max", single_dim=True)
_def("min", single_dim=True)
_def("argmax", single_dim=True)
_def("argmin", single_dim=True)
_def("kthvalue", single_dim=True)
_def("median", single_dim=True)
_def("nanmedian", single_dim=True)
_def("mode", single_dim=True)
_def("sort", reduce=False)
_def("argsort", reduce=False)
_def("unbind", single_dim=True)
_def("chunk", dim_offset=1, reduce=False)
_def("cummax", single_dim=True, reduce=False)
_def("cummin", single_dim=True, reduce=False)
_def("cumprod", single_dim=True, reduce=False)
_def("cumprod_", single_dim=True, reduce=False)
_def("cumsum", single_dim=True, reduce=False)
_def("cumsum_", single_dim=True, reduce=False)
_def("logcumsumexp", single_dim=True, reduce=False)
_def("renorm", dim_offset=1, single_dim=True, reduce=False)
_def("softmax", single_dim=True, reduce=False)

# 使用 _wrap 函数包装 torch.nn.functional.softmax 函数，设定参数 single_dim=True, reduce=False
softmax = _wrap(torch.nn.functional.softmax, single_dim=True, reduce=False)

# 在未来需要处理的内容，因为它们需要特殊的维度绑定逻辑
# cross
# diag_embed
# diagonal
# diagonal_scatter
# diff
# nanquantile
# quantile
# roll
# rot90
# topk (输出中的新维度)
# 这些操作是否都可以通过原地索引实现？
# index_add_
# index_add
# index_copy
# index_copy_
# index_fill
# index_fill_
# index_select
# scatter
# scatter_
# scatter_add
# scatter_add_
# scatter_reduce
```