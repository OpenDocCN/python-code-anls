# `.\pytorch\torch\_numpy\_ndarray.py`

```py
# 忽略类型检查错误
# 从将来的模块导入注解支持
import builtins
import math
import operator
from typing import Sequence

# 导入 PyTorch 库
import torch

# 导入内部模块和函数
from . import _dtypes, _dtypes_impl, _funcs, _ufuncs, _util
# 导入正常化相关模块和类
from ._normalizations import (
    ArrayLike,
    normalize_array_like,
    normalizer,
    NotImplementedType,
)

# 定义常量 newaxis
newaxis = None

# 定义标志位列表
FLAGS = [
    "C_CONTIGUOUS",
    "F_CONTIGUOUS",
    "OWNDATA",
    "WRITEABLE",
    "ALIGNED",
    "WRITEBACKIFCOPY",
    "FNC",
    "FORC",
    "BEHAVED",
    "CARRAY",
    "FARRAY",
]

# 定义标志位的缩写到完整名称的映射
SHORTHAND_TO_FLAGS = {
    "C": "C_CONTIGUOUS",
    "F": "F_CONTIGUOUS",
    "O": "OWNDATA",
    "W": "WRITEABLE",
    "A": "ALIGNED",
    "X": "WRITEBACKIFCOPY",
    "B": "BEHAVED",
    "CA": "CARRAY",
    "FA": "FARRAY",
}

# 定义 Flags 类，用于处理标志位
class Flags:
    def __init__(self, flag_to_value: dict):
        # 断言所有传入的标志位都在 FLAGS 列表中，确保数据完整性
        assert all(k in FLAGS for k in flag_to_value.keys())  # sanity check
        self._flag_to_value = flag_to_value

    def __getattr__(self, attr: str):
        # 如果属性名为小写且对应的大写在 FLAGS 中，则返回该属性对应的值
        if attr.islower() and attr.upper() in FLAGS:
            return self[attr.upper()]
        else:
            raise AttributeError(f"No flag attribute '{attr}'")

    def __getitem__(self, key):
        # 如果 key 在 SHORTHAND_TO_FLAGS 中，则将其转换为完整名称
        if key in SHORTHAND_TO_FLAGS.keys():
            key = SHORTHAND_TO_FLAGS[key]
        # 如果 key 在 FLAGS 中，则返回对应的值；否则抛出 KeyError
        if key in FLAGS:
            try:
                return self._flag_to_value[key]
            except KeyError as e:
                raise NotImplementedError(f"{key=}") from e
        else:
            raise KeyError(f"No flag key '{key}'")

    def __setattr__(self, attr, value):
        # 如果属性名为小写且对应的大写在 FLAGS 中，则设置该属性对应的值
        if attr.islower() and attr.upper() in FLAGS:
            self[attr.upper()] = value
        else:
            super().__setattr__(attr, value)

    def __setitem__(self, key, value):
        # 禁止修改标志位，抛出 NotImplementedError
        if key in FLAGS or key in SHORTHAND_TO_FLAGS.keys():
            raise NotImplementedError("Modifying flags is not implemented")
        else:
            raise KeyError(f"No flag key '{key}'")

# 创建方法的辅助函数
def create_method(fn, name=None):
    name = name or fn.__name__

    # 创建并返回一个函数，其调用传入的 fn 函数
    def f(*args, **kwargs):
        return fn(*args, **kwargs)

    # 设置函数的名称和限定名称
    f.__name__ = name
    f.__qualname__ = f"ndarray.{name}"
    return f

# 定义方法的映射字典
# 每个方法对应的值为 None，表示与其对应的 np 函数名相同
methods = {
    "clip": None,
    "nonzero": None,
    "repeat": None,
    "round": None,
    "squeeze": None,
    "swapaxes": None,
    "ravel": None,
    # linalg
    "diagonal": None,
    "dot": None,
    "trace": None,
    # sorting
    "argsort": None,
    "searchsorted": None,
    # reductions
    "argmax": None,
    "argmin": None,
    "any": None,
    "all": None,
    "max": None,
    "min": None,
    "ptp": None,
    "sum": None,
    "prod": None,
    "mean": None,
    "var": None,
    "std": None,
    # scans
    "cumsum": None,
    "cumprod": None,
    # advanced indexing
    "take": None,
    "choose": None,
}

# 定义双下划线方法的映射字典
dunder = {
    "abs": "absolute",
    "invert": None,
    "pos": "positive",
    "neg": "negative",
}
    "gt": "greater",  # 定义键 'gt'，对应的值是字符串 'greater'
    "lt": "less",     # 定义键 'lt'，对应的值是字符串 'less'
    "ge": "greater_equal",  # 定义键 'ge'，对应的值是字符串 'greater_equal'
    "le": "less_equal",     # 定义键 'le'，对应的值是字符串 'less_equal'
}

# 定义一个包含右向和原地操作变体的双下划线方法字典
ri_dunder = {
    "add": None,                   # '__add__'方法，无原地操作变体
    "sub": "subtract",             # '__sub__'方法，原地操作变体为'__rsub__'
    "mul": "multiply",             # '__mul__'方法，原地操作变体为'__rmul__'
    "truediv": "divide",           # '__truediv__'方法，原地操作变体为'__rtruediv__'
    "floordiv": "floor_divide",    # '__floordiv__'方法，原地操作变体为'__rfloordiv__'
    "pow": "power",                # '__pow__'方法，原地操作变体为'__rpow__'
    "mod": "remainder",            # '__mod__'方法，原地操作变体为'__rmod__'
    "and": "bitwise_and",          # '__and__'方法，原地操作变体为'__rand__'
    "or": "bitwise_or",            # '__or__'方法，原地操作变体为'__ror__'
    "xor": "bitwise_xor",          # '__xor__'方法，原地操作变体为'__rxor__'
    "lshift": "left_shift",        # '__lshift__'方法，原地操作变体为'__rlshift__'
    "rshift": "right_shift",       # '__rshift__'方法，原地操作变体为'__rrshift__'
    "matmul": None,                # '__matmul__'方法，无原地操作变体
}


def _upcast_int_indices(index):
    # 如果索引是torch.Tensor类型
    if isinstance(index, torch.Tensor):
        # 如果索引的数据类型是int8、int16、int32或uint8之一
        if index.dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
            # 将索引类型提升为int64
            return index.to(torch.int64)
    # 如果索引是tuple类型，递归地处理每个元素
    elif isinstance(index, tuple):
        return tuple(_upcast_int_indices(i) for i in index)
    # 其他情况直接返回索引本身
    return index


# 用于指示参数未指定（而不是显式地为`None`）
class _Unspecified:
    pass


# 将_Unspecified类的静态属性unspecified设置为_Unspecified实例
_Unspecified.unspecified = _Unspecified()

###############################################################
#                      ndarray class                          #
###############################################################


class ndarray:
    def __init__(self, t=None):
        if t is None:
            # 如果未提供参数t，则初始化一个空的torch.Tensor
            self.tensor = torch.Tensor()
        elif isinstance(t, torch.Tensor):
            # 如果参数t是torch.Tensor类型，则直接使用它初始化
            self.tensor = t
        else:
            # 如果参数t既不为None也不是torch.Tensor类型，则抛出错误
            raise ValueError(
                "ndarray constructor is not recommended; prefer"
                "either array(...) or zeros/empty(...)"
            )

    # 将NumPy函数注册为类方法
    for method, name in methods.items():
        fn = getattr(_funcs, name or method)
        vars()[method] = create_method(fn, method)

    # 从ufuncs中继承的常规方法
    conj = create_method(_ufuncs.conjugate, "conj")
    conjugate = create_method(_ufuncs.conjugate)

    # 为dunder字典中的方法注册特殊方法
    for method, name in dunder.items():
        fn = getattr(_ufuncs, name or method)
        method = f"__{method}__"
        vars()[method] = create_method(fn, method)

    # 为ri_dunder字典中的方法注册特殊方法和原地操作方法
    for method, name in ri_dunder.items():
        fn = getattr(_ufuncs, name or method)
        plain = f"__{method}__"
        vars()[plain] = create_method(fn, plain)
        rvar = f"__r{method}__"
        vars()[rvar] = create_method(lambda self, other, fn=fn: fn(other, self), rvar)
        ivar = f"__i{method}__"
        vars()[ivar] = create_method(
            lambda self, other, fn=fn: fn(self, other, out=self), ivar
        )

    # 由于没有__idivmod__方法，因此需要单独注册
    __divmod__ = create_method(_ufuncs.divmod, "__divmod__")
    __rdivmod__ = create_method(
        lambda self, other: _ufuncs.divmod(other, self), "__rdivmod__"
    )

    # 防止循环变量泄漏到ndarray类命名空间
    del ivar, rvar, name, plain, fn, method

    @property
    def shape(self):
        # 返回tensor的形状作为元组
        return tuple(self.tensor.shape)

    @property
    def size(self):
        # 返回tensor中元素的总数
        return self.tensor.numel()

    @property
    def ndim(self):
        # 返回tensor的维度数
        return self.tensor.ndim

    @property
    def dtype(self):
        # 返回tensor的数据类型
        return _dtypes.dtype(self.tensor.dtype)

    @property
    # 计算张量每个维度的步长乘以元素大小，返回一个元组
    def strides(self):
        elsize = self.tensor.element_size()
        return tuple(stride * elsize for stride in self.tensor.stride())

    @property
    def itemsize(self):
        # 返回张量中单个元素的字节大小
        return self.tensor.element_size()

    @property
    def flags(self):
        # 返回张量的标志信息，包括是否C连续、是否Fortran连续、是否拥有独立数据、是否可写
        # 注意：在PyTorch中，假定连续表示C风格的连续性
        return Flags(
            {
                "C_CONTIGUOUS": self.tensor.is_contiguous(),
                "F_CONTIGUOUS": self.T.tensor.is_contiguous(),
                "OWNDATA": self.tensor._base is None,
                "WRITEABLE": True,  # PyTorch没有只读张量
            }
        )

    @property
    def data(self):
        # 返回张量数据的指针
        return self.tensor.data_ptr()

    @property
    def nbytes(self):
        # 返回张量存储器的总字节数
        return self.tensor.storage().nbytes()

    @property
    def T(self):
        # 返回张量的转置
        return self.transpose()

    @property
    def real(self):
        # 返回张量的实部
        return _funcs.real(self)

    @real.setter
    def real(self, value):
        # 设置张量的实部
        self.tensor.real = asarray(value).tensor

    @property
    def imag(self):
        # 返回张量的虚部
        return _funcs.imag(self)

    @imag.setter
    def imag(self, value):
        # 设置张量的虚部
        self.tensor.imag = asarray(value).tensor

    # 构造函数
    def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
        # 检查是否支持指定的参数，如果不支持则抛出异常
        if order != "K":
            raise NotImplementedError(f"astype(..., order={order} is not implemented.")
        if casting != "unsafe":
            raise NotImplementedError(
                f"astype(..., casting={casting} is not implemented."
            )
        if not subok:
            raise NotImplementedError(f"astype(..., subok={subok} is not implemented.")
        if not copy:
            raise NotImplementedError(f"astype(..., copy={copy} is not implemented.")
        # 将张量转换为指定类型的张量，并返回新的ndarray对象
        torch_dtype = _dtypes.dtype(dtype).torch_dtype
        t = self.tensor.to(torch_dtype)
        return ndarray(t)

    @normalizer
    def copy(self: ArrayLike, order: NotImplementedType = "C"):
        # 返回张量的深拷贝副本
        return self.clone()

    @normalizer
    def flatten(self: ArrayLike, order: NotImplementedType = "C"):
        # 返回张量的扁平化表示
        return torch.flatten(self)
    def resize(self, *new_shape, refcheck=False):
        # NB: differs from np.resize: fills with zeros instead of making repeated copies of input.
        # 如果 refcheck 参数为 True，则抛出未实现的异常
        if refcheck:
            raise NotImplementedError(
                f"resize(..., refcheck={refcheck} is not implemented."
            )
        # 如果 new_shape 为空或者为 None，则直接返回，不进行 resize 操作
        if new_shape in [(), (None,)]:
            return

        # 支持两种调用方式：x.resize((2, 2)) 和 x.resize(2, 2)
        if len(new_shape) == 1:
            new_shape = new_shape[0]
        # 如果 new_shape 是整数，则转换为元组形式
        if isinstance(new_shape, int):
            new_shape = (new_shape,)

        # 检查 new_shape 的每个元素是否为非负数
        if builtins.any(x < 0 for x in new_shape):
            raise ValueError("all elements of `new_shape` must be non-negative")

        # 计算新 tensor 的元素个数和旧 tensor 的元素个数
        new_numel, old_numel = math.prod(new_shape), self.tensor.numel()

        # 调整 tensor 的大小
        self.tensor.resize_(new_shape)

        # 如果新的元素个数大于等于旧的元素个数，则填充多出的元素为零
        if new_numel >= old_numel:
            # 确保 tensor 是连续的
            assert self.tensor.is_contiguous()
            # 获取扁平化的 tensor 视图，并将多出的元素置零
            b = self.tensor.flatten()  # 不会复制数据
            b[old_numel:].zero_()

    def view(self, dtype=_Unspecified.unspecified, type=_Unspecified.unspecified):
        # 如果 dtype 未指定，则使用对象本身的 dtype
        if dtype is _Unspecified.unspecified:
            dtype = self.dtype
        # 如果指定了 type 参数，则抛出未实现异常
        if type is not _Unspecified.unspecified:
            raise NotImplementedError(f"view(..., type={type} is not implemented.")
        # 转换为 Torch 的数据类型
        torch_dtype = _dtypes.dtype(dtype).torch_dtype
        # 使用 Torch 的 view 方法创建视图
        tview = self.tensor.view(torch_dtype)
        # 返回 ndarray 对象的视图
        return ndarray(tview)

    @normalizer
    def fill(self, value: ArrayLike):
        # 填充 tensor 的所有元素为指定的值
        self.tensor.fill_(value)

    def tolist(self):
        # 将 tensor 转换为 Python 列表
        return self.tensor.tolist()

    def __iter__(self):
        # 返回 tensor 的迭代器，每个元素转换为 ndarray 对象
        return (ndarray(x) for x in self.tensor.__iter__())

    def __str__(self):
        # 返回 tensor 的字符串表示，替换 tensor 为 torch.ndarray，dtype=torch. 替换为 dtype=
        return (
            str(self.tensor)
            .replace("tensor", "torch.ndarray")
            .replace("dtype=torch.", "dtype=")
        )

    __repr__ = create_method(__str__)

    def __eq__(self, other):
        try:
            # 使用 _ufuncs.equal 比较两个对象是否相等
            return _ufuncs.equal(self, other)
        except (RuntimeError, TypeError):
            # 转换失败时，返回一个全为 False 的数组作为结果
            falsy = torch.full(self.shape, fill_value=False, dtype=bool)
            return asarray(falsy)

    def __ne__(self, other):
        # 使用 __eq__ 方法的结果取反，判断两个对象是否不相等
        return ~(self == other)

    def __index__(self):
        try:
            # 尝试将 tensor 的单个元素转换为整数索引
            return operator.index(self.tensor.item())
        except Exception as exc:
            # 转换失败时，抛出类型错误
            raise TypeError(
                "only integer scalar arrays can be converted to a scalar index"
            ) from exc

    def __bool__(self):
        # 判断 tensor 是否为真
        return bool(self.tensor)

    def __int__(self):
        # 将 tensor 转换为整数
        return int(self.tensor)

    def __float__(self):
        # 将 tensor 转换为浮点数
        return float(self.tensor)

    def __complex__(self):
        # 将 tensor 转换为复数
        return complex(self.tensor)
    def is_integer(self):
        try:
            # 获取张量的单个元素值
            v = self.tensor.item()
            # 检查该值是否为整数
            result = int(v) == v
        except Exception:
            result = False
        return result

    def __len__(self):
        # 返回张量的第一维度大小作为长度
        return self.tensor.shape[0]

    def __contains__(self, x):
        # 检查张量是否包含元素 x
        return self.tensor.__contains__(x)

    def transpose(self, *axes):
        # 调用 _funcs.transpose 函数进行张量的转置操作
        return _funcs.transpose(self, axes)

    def reshape(self, *shape, order="C"):
        # 调用 _funcs.reshape 函数进行张量的形状重塑操作
        return _funcs.reshape(self, shape, order=order)

    def sort(self, axis=-1, kind=None, order=None):
        # 使用 _funcs.sort 函数对张量进行排序，该操作是原地的
        _funcs.copyto(self, _funcs.sort(self, axis, kind, order))

    def item(self, *args):
        # 模仿 NumPy 的实现，处理三种特殊情况（无参数、平坦索引和多索引）
        if args == ():
            return self.tensor.item()
        elif len(args) == 1:
            # 整数参数
            return self.ravel()[args[0]]
        else:
            return self.__getitem__(args)

    def __getitem__(self, index):
        tensor = self.tensor

        def neg_step(i, s):
            # 处理负步长的切片索引，翻转相应轴并调整切片
            if not (isinstance(s, slice) and s.step is not None and s.step < 0):
                return s

            nonlocal tensor
            tensor = torch.flip(tensor, (i,))

            # 考虑切片包含起始但不包含结束的特性
            assert isinstance(s.start, int) or s.start is None
            assert isinstance(s.stop, int) or s.stop is None
            start = s.stop + 1 if s.stop else None
            stop = s.start + 1 if s.start else None

            return slice(start, stop, -s.step)

        if isinstance(index, Sequence):
            # 对序列中的每个索引应用 neg_step 函数
            index = type(index)(neg_step(i, s) for i, s in enumerate(index))
        else:
            # 对单个索引应用 neg_step 函数
            index = neg_step(0, index)
        # 将索引转换为张量类型
        index = _util.ndarrays_to_tensors(index)
        # 将整数索引向上转型
        index = _upcast_int_indices(index)
        # 返回通过张量索引创建的新的 ndarray 对象
        return ndarray(tensor.__getitem__(index))

    def __setitem__(self, index, value):
        # 将索引转换为张量类型
        index = _util.ndarrays_to_tensors(index)
        # 将整数索引向上转型
        index = _upcast_int_indices(index)

        # 如果 value 不是标量，则规范化成类似数组
        if not _dtypes_impl.is_scalar(value):
            value = normalize_array_like(value)
            # 根据需要进行类型转换
            value = _util.cast_if_needed(value, self.tensor.dtype)

        # 调用 tensor 的 __setitem__ 方法设置值
        return self.tensor.__setitem__(index, value)

    take = _funcs.take
    put = _funcs.put

    def __dlpack__(self, *, stream=None):
        # 调用 tensor 的 __dlpack__ 方法
        return self.tensor.__dlpack__(stream=stream)

    def __dlpack_device__(self):
        # 调用 tensor 的 __dlpack_device__ 方法
        return self.tensor.__dlpack_device__()
# 递归地将张量转换为列表。
def _tolist(obj):
    a1 = []  # 创建空列表a1，用于存放转换后的结果
    for elem in obj:  # 遍历obj中的每个元素
        if isinstance(elem, (list, tuple)):  # 如果elem是列表或元组类型
            elem = _tolist(elem)  # 递归调用_tolist函数，将elem转换为列表
        if isinstance(elem, ndarray):  # 如果elem是ndarray类型
            a1.append(elem.tensor.tolist())  # 将elem的tensor属性转换为列表，并添加到a1中
        else:
            a1.append(elem)  # 将elem直接添加到a1中
    return a1  # 返回转换后的列表a1


# 这是唯一直接与ndarray交互的地方。
# 其余部分通过asarray（首选）或array处理。
```