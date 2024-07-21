# `.\pytorch\torch\_numpy\_dtypes.py`

```
# mypy: ignore-errors
# 忽略 mypy 类型检查中的错误信息

""" Define analogs of numpy dtypes supported by pytorch.
Define the scalar types and supported dtypes and numpy <--> torch dtype mappings.
"""
# 导入内置模块 builtins
import builtins

# 导入 torch 库
import torch

# 导入 _dtypes_impl 模块（假设是当前目录下的一个模块）
from . import _dtypes_impl


# ### Scalar types ###

# 定义一个通用的类 generic
class generic:
    name = "generic"

    def __new__(cls, value):
        # NumPy scalars are modelled as 0-D arrays
        # so a call to np.float32(4) produces a 0-D array.

        # 导入 asarray 和 ndarray 函数
        from ._ndarray import asarray, ndarray

        # 如果 value 是字符串并且在 ["inf", "nan"] 中，则转换成 torch 中对应的值
        if isinstance(value, str) and value in ["inf", "nan"]:
            value = {"inf": torch.inf, "nan": torch.nan}[value]

        # 如果 value 是 ndarray 类型，则转换成指定 dtype 的 ndarray
        if isinstance(value, ndarray):
            return value.astype(cls)
        else:
            return asarray(value, dtype=cls)


##################
# abstract types #
##################

# 定义 number 类，继承自 generic
class number(generic):
    name = "number"

# 定义 integer 类，继承自 number
class integer(number):
    name = "integer"

# 定义 inexact 类，继承自 number
class inexact(number):
    name = "inexact"

# 定义 signedinteger 类，继承自 integer
class signedinteger(integer):
    name = "signedinteger"

# 定义 unsignedinteger 类，继承自 integer
class unsignedinteger(integer):
    name = "unsignedinteger"

# 定义 floating 类，继承自 inexact
class floating(inexact):
    name = "floating"

# 定义 complexfloating 类，继承自 inexact
class complexfloating(inexact):
    name = "complexfloating"

# 抽象类型列表
_abstract_dtypes = [
    "generic",
    "number",
    "integer",
    "signedinteger",
    "unsignedinteger",
    "inexact",
    "floating",
    "complexfloating",
]

# ##### concrete types

# signed integers

# 定义 int8 类，继承自 signedinteger
class int8(signedinteger):
    name = "int8"
    typecode = "b"
    torch_dtype = torch.int8

# 定义 int16 类，继承自 signedinteger
class int16(signedinteger):
    name = "int16"
    typecode = "h"
    torch_dtype = torch.int16

# 定义 int32 类，继承自 signedinteger
class int32(signedinteger):
    name = "int32"
    typecode = "i"
    torch_dtype = torch.int32

# 定义 int64 类，继承自 signedinteger
class int64(signedinteger):
    name = "int64"
    typecode = "l"
    torch_dtype = torch.int64

# unsigned integers

# 定义 uint8 类，继承自 unsignedinteger
class uint8(unsignedinteger):
    name = "uint8"
    typecode = "B"
    torch_dtype = torch.uint8

# 定义 uint16 类，继承自 unsignedinteger
class uint16(unsignedinteger):
    name = "uint16"
    typecode = "H"
    torch_dtype = torch.uint16

# 定义 uint32 类，继承自 signedinteger
class uint32(signedinteger):
    name = "uint32"
    typecode = "I"
    torch_dtype = torch.uint32

# 定义 uint64 类，继承自 signedinteger
class uint64(signedinteger):
    name = "uint64"
    typecode = "L"
    torch_dtype = torch.uint64

# floating point

# 定义 float16 类，继承自 floating
class float16(floating):
    name = "float16"
    typecode = "e"
    torch_dtype = torch.float16

# 定义 float32 类，继承自 floating
class float32(floating):
    name = "float32"
    typecode = "f"
    torch_dtype = torch.float32

# 定义 float64 类，继承自 floating
class float64(floating):
    name = "float64"
    typecode = "d"
    torch_dtype = torch.float64

# 定义 complex64 类，继承自 complexfloating
class complex64(complexfloating):
    name = "complex64"
    typecode = "F"
    torch_dtype = torch.complex64

# 定义 complex128 类，继承自 complexfloating
class complex128(complexfloating):
    name = "complex128"
    typecode = "D"
    torch_dtype = torch.complex128

# 定义 bool_ 类，继承自 generic
class bool_(generic):
    name = "bool_"
    typecode = "?"
    torch_dtype = torch.bool

# 名称别名字典
_name_aliases = {
    "intp": int64,
    "int_": int64,
    "intc": int32,
    "byte": int8,
    "short": int16,
    "longlong": int64,  # 定义名为 'longlong' 的变量，其类型为 int64，表示有符号64位整数
    "ulonglong": uint64,  # 定义名为 'ulonglong' 的变量，其类型为 uint64，表示无符号64位整数
    "ubyte": uint8,  # 定义名为 'ubyte' 的变量，其类型为 uint8，表示无符号8位整数（字节）
    "half": float16,  # 定义名为 'half' 的变量，其类型为 float16，表示半精度浮点数
    "single": float32,  # 定义名为 'single' 的变量，其类型为 float32，表示单精度浮点数
    "double": float64,  # 定义名为 'double' 的变量，其类型为 float64，表示双精度浮点数
    "float_": float64,  # 定义名为 'float_' 的变量，其类型为 float64，表示双精度浮点数
    "csingle": complex64,  # 定义名为 'csingle' 的变量，其类型为 complex64，表示单精度复数（实部和虚部均为32位浮点数）
    "singlecomplex": complex64,  # 定义名为 'singlecomplex' 的变量，其类型为 complex64，表示单精度复数
    "cdouble": complex128,  # 定义名为 'cdouble' 的变量，其类型为 complex128，表示双精度复数
    "cfloat": complex128,  # 定义名为 'cfloat' 的变量，其类型为 complex128，表示双精度复数
    "complex_": complex128,  # 定义名为 'complex_' 的变量，其类型为 complex128，表示双精度复数
# 循环结束标记
}

# 注册浮点类型别名，例如 float_ = float32 等等
for name, obj in _name_aliases.items():
    vars()[name] = obj

# 模仿 NumPy 中定义的方法，按照标量类型分组，详见 tests/core/test_scalar_methods.py
sctypes = {
    "int": [int8, int16, int32, int64],
    "uint": [uint8, uint16, uint32, uint64],
    "float": [float16, float32, float64],
    "complex": [complex64, complex128],
    "others": [bool_],
}

# 支持映射和函数

# 创建名称到标量类型的字典 _names
_names = {st.name: st for cat in sctypes for st in sctypes[cat]}

# 创建类型码到标量类型的字典 _typecodes
_typecodes = {st.typecode: st for cat in sctypes for st in sctypes[cat]}

# 创建 PyTorch 数据类型到标量类型的字典 _torch_dtypes
_torch_dtypes = {st.torch_dtype: st for cat in sctypes for st in sctypes[cat]}

# 创建别名到标量类型的字典 _aliases
_aliases = {
    "u1": uint8,
    "i1": int8,
    "i2": int16,
    "i4": int32,
    "i8": int64,
    "b": int8,  # XXX: srsly?
    "f2": float16,
    "f4": float32,
    "f8": float64,
    "c8": complex64,
    "c16": complex128,
    # numpy-specific trailing underscore
    "bool_": bool_,
}

# 创建 Python 类型到标量类型的字典 _python_types
_python_types = {
    int: int64,
    float: float64,
    complex: complex128,
    builtins.bool: bool_,
    # 允许 Python 类型的字符串化名称
    int.__name__: int64,
    float.__name__: float64,
    complex.__name__: complex128,
    builtins.bool.__name__: bool_,
}

def sctype_from_string(s):
    """规范化字符串值：类型名称、类型码或宽度别名。"""
    if s in _names:
        return _names[s]
    if s in _name_aliases.keys():
        return _name_aliases[s]
    if s in _typecodes:
        return _typecodes[s]
    if s in _aliases:
        return _aliases[s]
    if s in _python_types:
        return _python_types[s]
    raise TypeError(f"data type {s!r} not understood")

def sctype_from_torch_dtype(torch_dtype):
    return _torch_dtypes[torch_dtype]

# ### DTypes. ###

def dtype(arg):
    """返回与给定参数对应的数据类型对象。如果参数为 None，则返回默认的浮点数据类型对象。"""
    if arg is None:
        arg = _dtypes_impl.default_dtypes().float_dtype
    return DType(arg)

class DType:
    """表示特定标量类型的数据类型对象。"""
    def __init__(self, arg):
        # 如果参数是 PyTorch 对象
        if isinstance(arg, torch.dtype):
            sctype = _torch_dtypes[arg]
        # 如果参数是 PyTorch 张量
        elif isinstance(arg, torch.Tensor):
            sctype = _torch_dtypes[arg.dtype]
        # 如果参数是标量类型的子类
        elif issubclass_(arg, generic):
            sctype = arg
        # 如果参数已经是 DType 对象
        elif isinstance(arg, DType):
            sctype = arg._scalar_type
        # 如果参数有 dtype 属性
        elif hasattr(arg, "dtype"):
            sctype = arg.dtype._scalar_type
        else:
            sctype = sctype_from_string(arg)
        self._scalar_type = sctype

    @property
    def name(self):
        return self._scalar_type.name

    @property
    def type(self):
        return self._scalar_type

    @property
    def kind(self):
        """返回数据类型的种类符号，参考 https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html"""
        return _torch_dtypes[self.torch_dtype].name[0]

    @property
    def typecode(self):
        return self._scalar_type.typecode
    # 比较函数，用于判断是否相等
    def __eq__(self, other):
        # 如果other是DType类的实例，则比较其标量类型是否相等
        if isinstance(other, DType):
            return self._scalar_type == other._scalar_type
        # 如果other不是DType类的实例，尝试用other创建一个新的DType实例
        try:
            other_instance = DType(other)
        except TypeError:
            # 如果创建失败，则返回False
            return False
        # 比较self的标量类型与新创建的DType实例的标量类型是否相等
        return self._scalar_type == other_instance._scalar_type

    # 属性方法，返回与torch数据类型相关联的标量类型
    @property
    def torch_dtype(self):
        return self._scalar_type.torch_dtype

    # 哈希函数，返回标量类型名称的哈希值
    def __hash__(self):
        return hash(self._scalar_type.name)

    # 返回DType对象的字符串表示形式
    def __repr__(self):
        return f'dtype("{self.name}")'

    # 将__str__方法重载为__repr__方法
    __str__ = __repr__

    # 属性方法，返回标量类型的字节大小
    @property
    def itemsize(self):
        # 创建一个标量类型为1的元素，并返回其张量的元素大小
        elem = self.type(1)
        return elem.tensor.element_size()

    # 序列化方法，返回对象的状态信息（标量类型）
    def __getstate__(self):
        return self._scalar_type

    # 反序列化方法，设置对象的状态信息（标量类型）
    def __setstate__(self, value):
        self._scalar_type = value
# 定义一个字典，用于存储不同类型的类型码
typecodes = {
    "All": "efdFDBbhil?",
    "AllFloat": "efdFD",
    "AllInteger": "Bbhil",
    "Integer": "bhil",
    "UnsignedInteger": "B",
    "Float": "efd",
    "Complex": "FD",
}

# 设置默认的数据类型（dtype）和复杂数据类型（complex dtype）
def set_default_dtype(fp_dtype="numpy", int_dtype="numpy"):
    """Set the (global) defaults for fp, complex, and int dtypes.

    The complex dtype is inferred from the float (fp) dtype. It has
    a width at least twice the width of the float dtype,
    i.e., it's complex128 for float64 and complex64 for float32.

    Parameters
    ----------
    fp_dtype : str
        Allowed values are "numpy", "pytorch" or dtype_like things which
        can be converted into a DType instance.
        Default is "numpy" (i.e. float64).
    int_dtype : str
        Allowed values are "numpy", "pytorch" or dtype_like things which
        can be converted into a DType instance.
        Default is "numpy" (i.e. int64).

    Returns
    -------
    namedtuple
        The old default dtype state: a namedtuple with attributes ``float_dtype``,
        ``complex_dtypes`` and ``int_dtype``. These attributes store *pytorch*
        dtypes.

    Notes
    ------------
    This functions has a side effect: it sets the global state with the provided dtypes.

    The complex dtype has bit width of at least twice the width of the float
    dtype, i.e. it's complex128 for float64 and complex64 for float32.

    """
    # 将非标准的数据类型转换为对应的 torch dtype
    if fp_dtype not in ["numpy", "pytorch"]:
        fp_dtype = dtype(fp_dtype).torch_dtype
    if int_dtype not in ["numpy", "pytorch"]:
        int_dtype = dtype(int_dtype).torch_dtype

    # 根据 fp_dtype 设置 float_dtype
    if fp_dtype == "numpy":
        float_dtype = torch.float64
    elif fp_dtype == "pytorch":
        float_dtype = torch.float32
    else:
        float_dtype = fp_dtype

    # 根据 float_dtype 设置对应的 complex_dtype
    complex_dtype = {
        torch.float64: torch.complex128,
        torch.float32: torch.complex64,
        torch.float16: torch.complex64,
    }[float_dtype]

    # 将 int_dtype 转换为 torch.int64
    if int_dtype in ["numpy", "pytorch"]:
        int_dtype = torch.int64
    else:
        int_dtype = int_dtype

    # 创建新的默认 dtype 状态
    new_defaults = _dtypes_impl.DefaultDTypes(
        float_dtype=float_dtype, complex_dtype=complex_dtype, int_dtype=int_dtype
    )

    # 设置新的全局状态并返回旧的状态
    old_defaults = _dtypes_impl.default_dtypes
    _dtypes_impl._default_dtypes = new_defaults
    return old_defaults


# 检查 arg 是否为 klass 的子类
def issubclass_(arg, klass):
    try:
        return issubclass(arg, klass)
    except TypeError:
        return False


# 检查 arg1 是否为 arg2 的子类型
def issubdtype(arg1, arg2):
    # cf https://github.com/numpy/numpy/blob/v1.24.0/numpy/core/numerictypes.py#L356-L420

    # 将字符串转换为对应的抽象数据类型
    def str_to_abstract(t):
        if isinstance(t, str) and t in _abstract_dtypes:
            return globals()[t]
        return t

    arg1 = str_to_abstract(arg1)
    arg2 = str_to_abstract(arg2)

    # 如果 arg1 不是 generic 的子类，则将其转换为 dtype 类型
    if not issubclass_(arg1, generic):
        arg1 = dtype(arg1).type
    # 如果参数 arg2 不是 generic 的子类，则将 arg2 转换为其数据类型的类型
    if not issubclass_(arg2, generic):
        arg2 = dtype(arg2).type
    # 检查 arg1 是否是 arg2 的子类，并返回结果
    return issubclass(arg1, arg2)
# 定义一个包含所有公开导出对象名称的列表
__all__ = ["dtype", "DType", "typecodes", "issubdtype", "set_default_dtype", "sctypes"]
# 将私有字典 _names 的所有键添加到 __all__ 列表中，用于导出
__all__ += list(_names.keys())  # noqa: PLE0605
# 将私有字典 _name_aliases 的所有键添加到 __all__ 列表中，用于导出
__all__ += list(_name_aliases.keys())  # noqa: PLE0605
# 将私有列表 _abstract_dtypes 的所有元素添加到 __all__ 列表中，用于导出
__all__ += _abstract_dtypes  # noqa: PLE0605
```