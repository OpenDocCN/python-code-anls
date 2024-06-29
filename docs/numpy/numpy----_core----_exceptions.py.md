# `.\numpy\numpy\_core\_exceptions.py`

```
"""
Various richly-typed exceptions, that also help us deal with string formatting
in python where it's easier.

By putting the formatting in `__str__`, we also avoid paying the cost for
users who silence the exceptions.
"""
from .._utils import set_module  # 导入相对路径中的 set_module 函数


def _unpack_tuple(tup):
    if len(tup) == 1:
        return tup[0]
    else:
        return tup


def _display_as_base(cls):
    """
    A decorator that makes an exception class look like its base.

    We use this to hide subclasses that are implementation details - the user
    should catch the base type, which is what the traceback will show them.

    Classes decorated with this decorator are subject to removal without a
    deprecation warning.
    """
    assert issubclass(cls, Exception)
    cls.__name__ = cls.__base__.__name__  # 将异常类的名称设为其基类的名称
    return cls


class UFuncTypeError(TypeError):
    """ Base class for all ufunc exceptions """
    def __init__(self, ufunc):
        self.ufunc = ufunc  # 初始化 ufunc 属性


@_display_as_base
class _UFuncNoLoopError(UFuncTypeError):
    """ Thrown when a ufunc loop cannot be found """
    def __init__(self, ufunc, dtypes):
        super().__init__(ufunc)
        self.dtypes = tuple(dtypes)  # 初始化 dtypes 属性为类型元组

    def __str__(self):
        return (
            "ufunc {!r} did not contain a loop with signature matching types "
            "{!r} -> {!r}"
        ).format(
            self.ufunc.__name__,
            _unpack_tuple(self.dtypes[:self.ufunc.nin]),  # 格式化字符串，显示 ufunc 名称和输入类型
            _unpack_tuple(self.dtypes[self.ufunc.nin:])  # 显示输出类型
        )


@_display_as_base
class _UFuncBinaryResolutionError(_UFuncNoLoopError):
    """ Thrown when a binary resolution fails """
    def __init__(self, ufunc, dtypes):
        super().__init__(ufunc, dtypes)  # 调用父类初始化方法
        assert len(self.dtypes) == 2  # 断言 dtypes 长度为 2

    def __str__(self):
        return (
            "ufunc {!r} cannot use operands with types {!r} and {!r}"
        ).format(
            self.ufunc.__name__, *self.dtypes  # 格式化字符串，显示 ufunc 名称和操作数类型
        )


@_display_as_base
class _UFuncCastingError(UFuncTypeError):
    def __init__(self, ufunc, casting, from_, to):
        super().__init__(ufunc)  # 调用父类初始化方法
        self.casting = casting  # 初始化 casting 属性
        self.from_ = from_  # 初始化 from_ 属性
        self.to = to  # 初始化 to 属性


@_display_as_base
class _UFuncInputCastingError(_UFuncCastingError):
    """ Thrown when a ufunc input cannot be casted """
    def __init__(self, ufunc, casting, from_, to, i):
        super().__init__(ufunc, casting, from_, to)  # 调用父类初始化方法
        self.in_i = i  # 初始化 in_i 属性

    def __str__(self):
        # only show the number if more than one input exists
        i_str = "{} ".format(self.in_i) if self.ufunc.nin != 1 else ""  # 根据输入数量确定显示的输入编号字符串
        return (
            "Cannot cast ufunc {!r} input {}from {!r} to {!r} with casting "
            "rule {!r}"
        ).format(
            self.ufunc.__name__, i_str, self.from_, self.to, self.casting  # 格式化字符串，显示 ufunc 名称、输入编号、源类型、目标类型和转换规则
        )


@_display_as_base
class _UFuncOutputCastingError(_UFuncCastingError):
    """ Thrown when a ufunc output cannot be casted """
    # 初始化方法，继承父类的构造方法，并添加了一个额外的实例变量 self.out_i
    def __init__(self, ufunc, casting, from_, to, i):
        super().__init__(ufunc, casting, from_, to)
        self.out_i = i

    # 返回描述对象的字符串表示，根据输出个数决定是否显示输出索引号
    def __str__(self):
        # 如果输出的个数不是1，则在字符串中包含输出索引号
        i_str = "{} ".format(self.out_i) if self.ufunc.nout != 1 else ""
        return (
            "Cannot cast ufunc {!r} output {}from {!r} to {!r} with casting "
            "rule {!r}"
        ).format(
            self.ufunc.__name__, i_str, self.from_, self.to, self.casting
        )
# 将 _ArrayMemoryError 类声明为一个装饰器的展示基类
@_display_as_base
# _ArrayMemoryError 类继承自内置的 MemoryError 类，用于表示无法分配数组时抛出的异常
class _ArrayMemoryError(MemoryError):
    
    """ Thrown when an array cannot be allocated"""
    # _ArrayMemoryError 类的初始化方法，接收 shape 和 dtype 两个参数
    def __init__(self, shape, dtype):
        self.shape = shape  # 将 shape 参数赋值给实例变量 self.shape
        self.dtype = dtype  # 将 dtype 参数赋值给实例变量 self.dtype

    # 定义 _total_size 属性方法，计算数组总大小的字节数
    @property
    def _total_size(self):
        num_bytes = self.dtype.itemsize  # 获取单个数组元素的字节数
        for dim in self.shape:
            num_bytes *= dim  # 根据数组各维度大小计算总字节数
        return num_bytes  # 返回计算得到的总字节数

    # 定义静态方法 _size_to_string，将字节数转换成合适的二进制大小字符串
    @staticmethod
    def _size_to_string(num_bytes):
        """ Convert a number of bytes into a binary size string """

        # https://en.wikipedia.org/wiki/Binary_prefix
        LOG2_STEP = 10
        STEP = 1024
        units = ['bytes', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB']

        # 计算应该使用的单位索引
        unit_i = max(num_bytes.bit_length() - 1, 1) // LOG2_STEP
        unit_val = 1 << (unit_i * LOG2_STEP)
        n_units = num_bytes / unit_val
        del unit_val

        # 确保选择的单位在四舍五入后是正确的
        if round(n_units) == STEP:
            unit_i += 1
            n_units /= STEP

        # 处理超出已定义单位范围的大小
        if unit_i >= len(units):
            new_unit_i = len(units) - 1
            n_units *= 1 << ((unit_i - new_unit_i) * LOG2_STEP)
            unit_i = new_unit_i

        unit_name = units[unit_i]  # 获取相应的单位名称
        # 格式化字符串，以合适的数字格式显示大小
        if unit_i == 0:
            # bytes 单位不显示小数点
            return '{:.0f} {}'.format(n_units, unit_name)
        elif round(n_units) < 1000:
            # 如果小于 1000，则显示三个有效数字
            return '{:#.3g} {}'.format(n_units, unit_name)
        else:
            # 否则显示所有数字
            return '{:#.0f} {}'.format(n_units, unit_name)

    # 定义 __str__ 方法，返回描述性字符串，说明无法分配数组内存的原因
    def __str__(self):
        size_str = self._size_to_string(self._total_size)  # 获取总大小的字符串表示
        return (
            "Unable to allocate {} for an array with shape {} and data type {}"
            .format(size_str, self.shape, self.dtype)
        )
```