# `.\numpy\numpy\ma\core.pyi`

```py
# 导入必要的模块和类型声明
from collections.abc import Callable
from typing import Any, TypeVar
from numpy import ndarray, dtype, float64

# 导入特定的 numpy 函数并重新命名以简化使用
from numpy import (
    amax as amax,
    amin as amin,
    bool as bool,
    expand_dims as expand_dims,
    clip as clip,
    indices as indices,
    ones_like as ones_like,
    squeeze as squeeze,
    zeros_like as zeros_like,
    angle as angle
)

# 定义一个公共的导出列表
__all__: list[str]

# 定义几个类型别名和全局变量
MaskType = bool
nomask: bool

# 定义几个自定义异常类
class MaskedArrayFutureWarning(FutureWarning): ...
class MAError(Exception): ...
class MaskError(MAError): ...

# 下面是一系列函数的声明，它们用于操作掩码数组
# 具体实现在后续定义中

def default_fill_value(obj): ...
def minimum_fill_value(obj): ...
def maximum_fill_value(obj): ...
def set_fill_value(a, fill_value): ...
def common_fill_value(a, b): ...
def filled(a, fill_value=...): ...
def getdata(a, subok=...): ...
get_data = getdata

def fix_invalid(a, mask=..., copy=..., fill_value=...): ...

# 下面是一系列类的定义，它们用于定义掩码操作的行为和逻辑
# 具体实现在后续定义中

class _MaskedUFunc:
    f: Any
    __doc__: Any
    __name__: Any
    def __init__(self, ufunc): ...

class _MaskedUnaryOperation(_MaskedUFunc):
    fill: Any
    domain: Any
    def __init__(self, mufunc, fill=..., domain=...): ...
    def __call__(self, a, *args, **kwargs): ...

class _MaskedBinaryOperation(_MaskedUFunc):
    fillx: Any
    filly: Any
    def __init__(self, mbfunc, fillx=..., filly=...): ...
    def __call__(self, a, b, *args, **kwargs): ...
    def reduce(self, target, axis=..., dtype=...): ...
    def outer(self, a, b): ...
    def accumulate(self, target, axis=...): ...

class _DomainedBinaryOperation(_MaskedUFunc):
    domain: Any
    fillx: Any
    filly: Any
    def __init__(self, dbfunc, domain, fillx=..., filly=...): ...
    def __call__(self, a, b, *args, **kwargs): ...

# 下面是一系列特定数学函数的定义，它们用于数组运算
# 具体实现在后续定义中

exp: _MaskedUnaryOperation
conjugate: _MaskedUnaryOperation
sin: _MaskedUnaryOperation
cos: _MaskedUnaryOperation
arctan: _MaskedUnaryOperation
arcsinh: _MaskedUnaryOperation
sinh: _MaskedUnaryOperation
cosh: _MaskedUnaryOperation
tanh: _MaskedUnaryOperation
abs: _MaskedUnaryOperation
absolute: _MaskedUnaryOperation
fabs: _MaskedUnaryOperation
negative: _MaskedUnaryOperation
floor: _MaskedUnaryOperation
ceil: _MaskedUnaryOperation
around: _MaskedUnaryOperation
logical_not: _MaskedUnaryOperation
sqrt: _MaskedUnaryOperation
log: _MaskedUnaryOperation
log2: _MaskedUnaryOperation
log10: _MaskedUnaryOperation
tan: _MaskedUnaryOperation
arcsin: _MaskedUnaryOperation
arccos: _MaskedUnaryOperation
arccosh: _MaskedUnaryOperation
arctanh: _MaskedUnaryOperation

# 下面是一系列特定数学操作的定义，它们用于数组的二元运算
# 具体实现在后续定义中

add: _MaskedBinaryOperation
subtract: _MaskedBinaryOperation
multiply: _MaskedBinaryOperation
arctan2: _MaskedBinaryOperation
equal: _MaskedBinaryOperation
not_equal: _MaskedBinaryOperation
less_equal: _MaskedBinaryOperation
greater_equal: _MaskedBinaryOperation
less: _MaskedBinaryOperation
greater: _MaskedBinaryOperation
logical_and: _MaskedBinaryOperation
# 定义 _MaskedBinaryOperation 类型别名为 alltrue
alltrue: _MaskedBinaryOperation
# 定义 _MaskedBinaryOperation 类型别名为 logical_or
logical_or: _MaskedBinaryOperation
# sometrue 是接受任意参数并返回任意类型的可调用对象
sometrue: Callable[..., Any]
# 定义 _MaskedBinaryOperation 类型别名为 logical_xor
logical_xor: _MaskedBinaryOperation
# 定义 _MaskedBinaryOperation 类型别名为 bitwise_and
bitwise_and: _MaskedBinaryOperation
# 定义 _MaskedBinaryOperation 类型别名为 bitwise_or
bitwise_or: _MaskedBinaryOperation
# 定义 _MaskedBinaryOperation 类型别名为 bitwise_xor
bitwise_xor: _MaskedBinaryOperation
# 定义 _MaskedBinaryOperation 类型别名为 hypot
hypot: _MaskedBinaryOperation
# 定义 _MaskedBinaryOperation 类型别名为 divide
divide: _MaskedBinaryOperation
# 定义 _MaskedBinaryOperation 类型别名为 true_divide
true_divide: _MaskedBinaryOperation
# 定义 _MaskedBinaryOperation 类型别名为 floor_divide
floor_divide: _MaskedBinaryOperation
# 定义 _MaskedBinaryOperation 类型别名为 remainder
remainder: _MaskedBinaryOperation
# 定义 _MaskedBinaryOperation 类型别名为 fmod
fmod: _MaskedBinaryOperation
# 定义 _MaskedBinaryOperation 类型别名为 mod
mod: _MaskedBinaryOperation

# 定义函数 make_mask_descr，用于生成掩码描述符
def make_mask_descr(ndtype): ...

# 定义函数 getmask，用于获取对象的掩码
def getmask(a): ...
# get_mask 是 getmask 的别名
get_mask = getmask

# 定义函数 getmaskarray，用于从数组获取掩码
def getmaskarray(arr): ...
# 定义函数 is_mask，用于检查对象是否为掩码
def is_mask(m): ...
# 定义函数 make_mask，用于创建掩码
def make_mask(m, copy=..., shrink=..., dtype=...): ...
# 定义函数 make_mask_none，用于创建空掩码
def make_mask_none(newshape, dtype=...): ...
# 定义函数 mask_or，用于对两个掩码进行逻辑或运算
def mask_or(m1, m2, copy=..., shrink=...): ...
# 定义函数 flatten_mask，用于将掩码扁平化
def flatten_mask(mask): ...
# 定义函数 masked_where，根据条件创建掩码数组
def masked_where(condition, a, copy=...): ...
# 定义一系列比较函数，根据条件对掩码数组进行元素级比较
def masked_greater(x, value, copy=...): ...
def masked_greater_equal(x, value, copy=...): ...
def masked_less(x, value, copy=...): ...
def masked_less_equal(x, value, copy=...): ...
def masked_not_equal(x, value, copy=...): ...
def masked_equal(x, value, copy=...): ...
def masked_inside(x, v1, v2, copy=...): ...
def masked_outside(x, v1, v2, copy=...): ...
def masked_object(x, value, copy=..., shrink=...): ...
def masked_values(x, value, rtol=..., atol=..., copy=..., shrink=...): ...
# 定义函数 masked_invalid，用于标记数组中的无效值
def masked_invalid(a, copy=...): ...

# 定义类 _MaskedPrintOption，用于控制掩码数组的打印选项
class _MaskedPrintOption:
    # 构造方法，接受显示选项
    def __init__(self, display): ...
    # 返回当前显示选项
    def display(self): ...
    # 设置显示选项
    def set_display(self, s): ...
    # 判断是否启用显示选项
    def enabled(self): ...
    # 启用显示选项，并选择是否压缩
    def enable(self, shrink=...): ...

# masked_print_option 是 _MaskedPrintOption 类的实例
masked_print_option: _MaskedPrintOption

# 定义函数 flatten_structured_array，用于扁平化结构化数组
def flatten_structured_array(a): ...

# 定义 MaskedIterator 类，用于迭代掩码数组
class MaskedIterator:
    # ma 是任意类型的成员数组
    ma: Any
    # dataiter 是任意类型的数据迭代器
    dataiter: Any
    # maskiter 是任意类型的掩码迭代器
    maskiter: Any
    # 构造方法，初始化掩码迭代器
    def __init__(self, ma): ...
    # 返回迭代器自身
    def __iter__(self): ...
    # 根据索引获取迭代器元素
    def __getitem__(self, indx): ...
    # 根据索引设置迭代器元素
    def __setitem__(self, index, value): ...
    # 获取迭代器的下一个元素
    def __next__(self): ...

# 定义 MaskedArray 类，继承自 ndarray
class MaskedArray(ndarray[_ShapeType, _DType_co]):
    # __array_priority__ 是任意类型的数组优先级
    __array_priority__: Any
    # 构造方法，创建掩码数组
    def __new__(cls, data=..., mask=..., dtype=..., copy=..., subok=..., ndmin=..., fill_value=..., keep_mask=..., hard_mask=..., shrink=..., order=...): ...
    # 用于在数组终结时调用的方法
    def __array_finalize__(self, obj): ...
    # 用于包装数组的方法
    def __array_wrap__(self, obj, context=..., return_scalar=...): ...
    # 返回视图数组
    def view(self, dtype=..., type=..., fill_value=...): ...
    # 根据索引获取数组元素
    def __getitem__(self, indx): ...
    # 根据索引设置数组元素
    def __setitem__(self, indx, value): ...
    # 返回数组的数据类型
    @property
    def dtype(self): ...
    # 设置数组的数据类型
    @dtype.setter
    def dtype(self, dtype): ...
    # 返回数组的形状
    @property
    def shape(self): ...
    # 设置数组的形状
    @shape.setter
    def shape(self, shape): ...
    # 设置数组的掩码
    def __setmask__(self, mask, copy=...): ...
    # 返回数组的掩码
    @property
    def mask(self): ...
    # 设置数组的掩码
    @mask.setter
    def mask(self, value): ...
    # 返回记录掩码
    @property
    def recordmask(self): ...
    # 设置记录掩码
    @recordmask.setter
    def recordmask(self, mask): ...
    # 强化掩码
    def harden_mask(self): ...
    # 软化掩码
    def soften_mask(self): ...
    # 返回硬掩码
    @property
    def hardmask(self): ...
    # 分离掩码
    def unshare_mask(self): ...
    # 返回共享掩码
    @property
    def sharedmask(self): ...
    # 缩小掩码
    def shrink_mask(self): ...
    # 返回
    @property
    # 定义一个空的方法 baseclass，占位用
    def baseclass(self): ...

    # 声明一个类型为 Any 的属性 data
    data: Any

    # 定义一个名为 flat 的属性方法，用于获取数据
    @property
    def flat(self): ...

    # 定义 flat 属性的 setter 方法，用于设置数据
    @flat.setter
    def flat(self, value): ...

    # 定义一个名为 fill_value 的属性方法，用于获取填充值
    @property
    def fill_value(self): ...

    # 定义 fill_value 属性的 setter 方法，用于设置填充值，默认为 ...
    @fill_value.setter
    def fill_value(self, value=...): ...

    # 声明两个 Any 类型的方法，get_fill_value 和 set_fill_value，功能不详
    get_fill_value: Any
    set_fill_value: Any

    # 定义一个名为 filled 的方法，用于返回使用指定填充值填充后的副本
    def filled(self, fill_value=...): ...

    # 定义一个名为 compressed 的方法，功能不详
    def compressed(self): ...

    # 定义一个名为 compress 的方法，用于按条件压缩数据
    def compress(self, condition, axis=..., out=...): ...

    # 定义相等比较运算符重载方法 __eq__
    def __eq__(self, other): ...

    # 定义不等比较运算符重载方法 __ne__
    def __ne__(self, other): ...

    # 定义大于等于比较运算符重载方法 __ge__
    def __ge__(self, other): ...

    # 定义大于比较运算符重载方法 __gt__
    def __gt__(self, other): ...

    # 定义小于等于比较运算符重载方法 __le__
    def __le__(self, other): ...

    # 定义小于比较运算符重载方法 __lt__
    def __lt__(self, other): ...

    # 定义加法运算符重载方法 __add__
    def __add__(self, other): ...

    # 定义反向加法运算符重载方法 __radd__
    def __radd__(self, other): ...

    # 定义减法运算符重载方法 __sub__
    def __sub__(self, other): ...

    # 定义反向减法运算符重载方法 __rsub__
    def __rsub__(self, other): ...

    # 定义乘法运算符重载方法 __mul__
    def __mul__(self, other): ...

    # 定义反向乘法运算符重载方法 __rmul__
    def __rmul__(self, other): ...

    # 定义除法运算符重载方法 __div__ （Python 2 中使用），功能不详
    def __div__(self, other): ...

    # 定义真除法运算符重载方法 __truediv__
    def __truediv__(self, other): ...

    # 定义反向真除法运算符重载方法 __rtruediv__
    def __rtruediv__(self, other): ...

    # 定义地板除法运算符重载方法 __floordiv__
    def __floordiv__(self, other): ...

    # 定义反向地板除法运算符重载方法 __rfloordiv__
    def __rfloordiv__(self, other): ...

    # 定义幂运算符重载方法 __pow__
    def __pow__(self, other): ...

    # 定义反向幂运算符重载方法 __rpow__
    def __rpow__(self, other): ...

    # 定义增强赋值加法运算符重载方法 __iadd__
    def __iadd__(self, other): ...

    # 定义增强赋值减法运算符重载方法 __isub__
    def __isub__(self, other): ...

    # 定义增强赋值乘法运算符重载方法 __imul__
    def __imul__(self, other): ...

    # 定义增强赋值除法运算符重载方法 __idiv__ （Python 2 中使用），功能不详
    def __idiv__(self, other): ...

    # 定义增强赋值地板除法运算符重载方法 __ifloordiv__
    def __ifloordiv__(self, other): ...

    # 定义增强赋值真除法运算符重载方法 __itruediv__
    def __itruediv__(self, other): ...

    # 定义增强赋值幂运算符重载方法 __ipow__
    def __ipow__(self, other): ...

    # 定义浮点数转换运算符重载方法 __float__
    def __float__(self): ...

    # 定义整数转换运算符重载方法 __int__
    def __int__(self): ...

    # 定义一个名为 imag 的属性方法，功能不详，类型为 ignore[misc]
    @property  # type: ignore[misc]
    def imag(self): ...

    # 声明一个 Any 类型的方法 get_imag，功能不详
    get_imag: Any

    # 定义一个名为 real 的属性方法，功能不详，类型为 ignore[misc]
    @property  # type: ignore[misc]
    def real(self): ...

    # 声明一个 Any 类型的方法 get_real，功能不详
    get_real: Any

    # 定义一个计数方法 count，用于统计元素个数
    def count(self, axis=..., keepdims=...): ...

    # 定义一个展平方法 ravel，用于将数组展平
    def ravel(self, order=...): ...

    # 定义一个重塑方法 reshape，用于改变数组形状
    def reshape(self, *s, **kwargs): ...

    # 定义一个重新调整大小方法 resize，用于改变数组大小
    def resize(self, newshape, refcheck=..., order=...): ...

    # 定义一个放置方法 put，用于根据索引放置值
    def put(self, indices, values, mode=...): ...

    # 定义一个 ids 方法，功能不详
    def ids(self): ...

    # 定义一个连续性检查方法 iscontiguous，用于检查数组是否连续存储
    def iscontiguous(self): ...

    # 定义一个所有元素是否为真方法 all，用于检查数组所有元素是否为真
    def all(self, axis=..., out=..., keepdims=...): ...

    # 定义一个任意元素是否为真方法 any，用于检查数组任意元素是否为真
    def any(self, axis=..., out=..., keepdims=...): ...

    # 定义一个非零元素索引方法 nonzero，用于返回非零元素的索引
    def nonzero(self): ...

    # 定义一个迹方法 trace，用于计算数组的迹
    def trace(self, offset=..., axis1=..., axis2=..., dtype=..., out=...): ...

    # 定义一个点积方法 dot，用于计算数组的点积
    def dot(self, b, out=..., strict=...): ...

    # 定义一个求和方法 sum，用于计算数组元素的和
    def sum(self, axis=..., dtype=..., out=..., keepdims=...): ...

    # 定义一个累积和方法 cumsum，用于计算数组元素的累积和
    def cumsum(self, axis=..., dtype=..., out=...): ...

    # 定义一个求积方法 prod，用于计算数组元素的乘积
    def prod(self, axis=..., dtype=..., out=..., keepdims=...): ...

    # 声明一个 Any 类型的属性 product，功能不详
    product: Any

    # 定义一个累积乘积方法 cumprod，用于计算数组元素的累积乘积
    def cumprod(self, axis=..., dtype=..., out=...): ...

    # 定义一个平均值方法 mean，用于计算数组元素的平均值
    def mean(self, axis=..., dtype=..., out=..., keepdims=...): ...

    # 定义一个异常方法 anom，功能不详
    def anom(self, axis=
    # 定义一个排序方法，可选参数包括轴向(axis)，排序类型(kind)，排序顺序(order)，以及结尾行为(endwith)，填充值(fill_value)，稳定性(stable)
    def sort(self, axis=..., kind=..., order=..., endwith=..., fill_value=..., stable=...): ...

    # 定义一个计算最小值的方法，可选参数包括轴向(axis)，输出(out)，填充值(fill_value)，保持维度(keepdims)
    def min(self, axis=..., out=..., fill_value=..., keepdims=...): ...

    # 提示：已弃用
    # def tostring(self, fill_value=..., order=...): ...

    # 定义一个计算最大值的方法，可选参数包括轴向(axis)，输出(out)，填充值(fill_value)，保持维度(keepdims)
    def max(self, axis=..., out=..., fill_value=..., keepdims=...): ...

    # 定义一个计算最大值与最小值差值的方法，可选参数包括轴向(axis)，输出(out)，填充值(fill_value)，保持维度(keepdims)
    def ptp(self, axis=..., out=..., fill_value=..., keepdims=...): ...

    # 定义一个分区方法，使用*args和**kwargs接受任意位置参数和关键字参数
    def partition(self, *args, **kwargs): ...

    # 定义一个参数分区方法，使用*args和**kwargs接受任意位置参数和关键字参数
    def argpartition(self, *args, **kwargs): ...

    # 定义一个按照给定索引取值的方法，可选参数包括索引(indices)，轴向(axis)，输出(out)，模式(mode)
    def take(self, indices, axis=..., out=..., mode=...): ...

    # 拷贝属性
    copy: Any

    # 返回对角线元素
    diagonal: Any

    # 返回扁平化的数组
    flatten: Any

    # 返回重复的数组
    repeat: Any

    # 去除维度为1的轴
    squeeze: Any

    # 交换指定轴的位置
    swapaxes: Any

    # 转置数组
    T: Any

    # 转置数组
    transpose: Any

    # 属性方法，返回自身的转置
    @property  # type: ignore[misc]
    def mT(self): ...

    # 将数组转换为列表形式，可选参数包括填充值(fill_value)
    def tolist(self, fill_value=...): ...

    # 将数组转换为字节形式，可选参数包括填充值(fill_value)，顺序(order)
    def tobytes(self, fill_value=..., order=...): ...

    # 将数组内容写入文件中，可选参数包括文件标识符(fid)，分隔符(sep)，格式(format)
    def tofile(self, fid, sep=..., format=...): ...

    # 转换为灵活类型
    def toflex(self): ...

    # 转换为记录类型
    torecords: Any

    # 序列化对象
    def __reduce__(self): ...

    # 深度拷贝对象，可选参数包括备忘录(memo)
    def __deepcopy__(self, memo=...): ...
# 定义一个名为 mvoid 的类，继承自 MaskedArray 类型，带有泛型参数 _ShapeType 和 _DType_co
class mvoid(MaskedArray[_ShapeType, _DType_co]):
    # __new__ 方法，用于创建新的对象实例，接受多个参数并返回实例
    def __new__(
        self,
        data,
        mask=...,
        dtype=...,
        fill_value=...,
        hardmask=...,
        copy=...,
        subok=...,
    ): ...

    # __getitem__ 方法，用于获取实例中指定索引处的值
    def __getitem__(self, indx): ...

    # __setitem__ 方法，用于设置实例中指定索引处的值
    def __setitem__(self, indx, value): ...

    # __iter__ 方法，使实例可迭代
    def __iter__(self): ...

    # __len__ 方法，返回实例中元素的数量
    def __len__(self): ...

    # filled 方法，返回填充了指定值的副本
    def filled(self, fill_value=...): ...

    # tolist 方法，返回实例的 Python 列表表示
    def tolist(self): ...

# 定义函数 isMaskedArray，用于检查对象是否为 MaskedArray 类型
def isMaskedArray(x): ...

# isarray 和 isMA 都是 isMaskedArray 函数的别名

# 定义一个名为 MaskedConstant 的类，继承自 MaskedArray 类型，具有泛型参数 Any 和 dtype[float64]
# 用于表示0维的 float64 数组常量
class MaskedConstant(MaskedArray[Any, dtype[float64]]):
    # __new__ 方法，用于创建新的对象实例
    def __new__(cls): ...

    __class__: Any

    # __array_finalize__ 方法，用于终结数组创建时的操作
    def __array_finalize__(self, obj): ...

    # __array_wrap__ 方法，用于包装数组操作的结果
    def __array_wrap__(self, obj, context=..., return_scalar=...): ...

    # __format__ 方法，用于格式化数组的输出
    def __format__(self, format_spec): ...

    # __reduce__ 方法，用于序列化数组对象
    def __reduce__(self): ...

    # __iop__ 方法，用于处理 in-place 操作
    def __iop__(self, other): ...

    __iadd__: Any
    __isub__: Any
    __imul__: Any
    __ifloordiv__: Any
    __itruediv__: Any
    __ipow__: Any

    # copy 方法，返回对象的副本
    def copy(self, *args, **kwargs): ...

    # __copy__ 方法，返回对象的浅层副本
    def __copy__(self): ...

    # __deepcopy__ 方法，返回对象的深层副本
    def __deepcopy__(self, memo): ...

    # __setattr__ 方法，设置对象的属性值
    def __setattr__(self, attr, value): ...

# masked 和 masked_singleton 是 MaskedConstant 类的实例，表示特定的屏蔽常量
masked: MaskedConstant
masked_singleton: MaskedConstant
masked_array = MaskedArray

# array 函数，用于创建数组
def array(
    data,
    dtype=...,
    copy=...,
    order=...,
    mask=...,
    fill_value=...,
    keep_mask=...,
    hard_mask=...,
    shrink=...,
    subok=...,
    ndmin=...,
): ...

# is_masked 函数，用于检查对象是否被屏蔽
def is_masked(x): ...

# 定义一个名为 _extrema_operation 的类，继承自 _MaskedUFunc 类型
class _extrema_operation(_MaskedUFunc):
    compare: Any
    fill_value_func: Any

    # __init__ 方法，初始化操作
    def __init__(self, ufunc, compare, fill_value): ...

    # __call__ 方法，用于调用对象实例
    # 注意：在实践中，`b` 有默认值，但是用户应显式提供此处的值，因为默认值已被弃用
    def __call__(self, a, b): ...

    # reduce 方法，用于减少操作
    def reduce(self, target, axis=...): ...

    # outer 方法，用于外积操作
    def outer(self, a, b): ...

# min 函数，计算对象中的最小值
def min(obj, axis=..., out=..., fill_value=..., keepdims=...): ...

# max 函数，计算对象中的最大值
def max(obj, axis=..., out=..., fill_value=..., keepdims=...): ...

# ptp 函数，计算对象中数值范围的峰值
def ptp(obj, axis=..., out=..., fill_value=..., keepdims=...): ...

# 定义一个名为 _frommethod 的类
class _frommethod:
    __name__: Any
    __doc__: Any
    reversed: Any

    # __init__ 方法，初始化操作
    def __init__(self, methodname, reversed=...): ...

    # getdoc 方法，返回对象的文档字符串
    def getdoc(self): ...

    # __call__ 方法，用于调用对象实例的方法
    def __call__(self, a, *args, **params): ...

# 下列名称都是 _frommethod 类的实例，表示特定的方法
all: _frommethod
anomalies: _frommethod
anom: _frommethod
any: _frommethod
compress: _frommethod
cumprod: _frommethod
cumsum: _frommethod
copy: _frommethod
diagonal: _frommethod
harden_mask: _frommethod
ids: _frommethod
mean: _frommethod
nonzero: _frommethod
prod: _frommethod
product: _frommethod
ravel: _frommethod
repeat: _frommethod
soften_mask: _frommethod
std: _frommethod
sum: _frommethod
swapaxes: _frommethod
trace: _frommethod
var: _frommethod
count: _frommethod
argmin: _frommethod
argmax: _frommethod

# minimum 和 maximum 是 _extrema_operation 类的实例，表示最小和最大值的计算操作
minimum: _extrema_operation
maximum: _extrema_operation

# take 函数，从数组中取出指定索引的元素
def take(a, indices, axis=..., out=..., mode=...): ...

# power 函数，计算指数幂运算
def power(a, b, third=...): ...

# argsort 函数，对数组进行排序并返回索引
def argsort(a, axis=..., kind=..., order=..., endwith=..., fill_value=..., stable=...): ...
# 定义一个排序函数，用于对数组进行排序
def sort(a, axis=..., kind=..., order=..., endwith=..., fill_value=..., stable=...): ...

# 返回一个压缩版本的数组，可能是对输入数组进行压缩操作
def compressed(x): ...

# 沿指定轴连接数组序列，返回连接后的结果
def concatenate(arrays, axis=...): ...

# 返回一个由向量 v 组成的对角矩阵或者从对角线中提取的元素
def diag(v, k=...): ...

# 返回数组 a 的所有元素按位左移 n 位后的结果
def left_shift(a, n): ...

# 返回数组 a 的所有元素按位右移 n 位后的结果
def right_shift(a, n): ...

# 将给定的值 values 按照 indices 数组中的索引放入到数组 a 中
def put(a, indices, values, mode=...): ...

# 根据 mask 数组的条件，将 values 中的值放入数组 a 中
def putmask(a, mask, values): ...

# 返回数组 a 的轴的置换结果，根据 axes 指定的轴顺序
def transpose(a, axes=...): ...

# 返回数组 a 的新形状 new_shape 的重塑结果，根据 order 指定的顺序
def reshape(a, new_shape, order=...): ...

# 将数组 x 调整为新形状 new_shape 的结果
def resize(x, new_shape): ...

# 返回对象 obj 的维度数
def ndim(obj): ...

# 返回对象 obj 的形状
def shape(obj): ...

# 返回对象 obj 指定轴的大小
def size(obj, axis=...): ...

# 返回数组 a 沿指定轴的第 n 阶差分结果
def diff(a, /, n=..., axis=..., prepend=..., append=...): ...

# 根据 condition 数组返回 x 或 y 中对应位置的值
def where(condition, x=..., y=...): ...

# 根据 indices 数组返回 choices 数组中对应位置的值
def choose(indices, choices, out=..., mode=...): ...

# 返回数组 a 中所有元素按指定精度 decimals 进行舍入的结果
def round(a, decimals=..., out=...): ...

# 返回数组 a 和 b 的内积结果
def inner(a, b): ...
innerproduct = inner

# 返回数组 a 和 b 的外积结果
def outer(a, b): ...
outerproduct = outer

# 返回数组 a 和 v 的相关性计算结果
def correlate(a, v, mode=..., propagate_mask=...): ...

# 返回数组 a 和 v 的卷积计算结果
def convolve(a, v, mode=..., propagate_mask=...): ...

# 判断数组 a 和 b 是否全部相等，填充值 fill_value 用于缺失值比较
def allequal(a, b, fill_value=...): ...

# 判断数组 a 和 b 是否在指定容差范围内全部相等
def allclose(a, b, masked_equal=..., rtol=..., atol=...): ...

# 将输入转换为数组，可以指定数据类型 dtype 和存储顺序 order
def asarray(a, dtype=..., order=...): ...

# 将输入转换为任意数组，可以指定数据类型 dtype
def asanyarray(a, dtype=...): ...

# 根据灵活数组 fxarray 返回一个对象
def fromflex(fxarray): ...

# _convert2ma 类的定义，包含函数名 funcname 和参数 params
class _convert2ma:
    __doc__: Any
    # 初始化方法，接受函数名和参数 params
    def __init__(self, funcname, params=...): ...
    # 返回类的文档
    def getdoc(self): ...
    # 调用实例时执行的方法，接受任意数量的位置参数和关键字参数 params
    def __call__(self, *args, **params): ...

# 函数 arange 被 _convert2ma 类对象 _convert2ma 包装
arange: _convert2ma

# 函数 empty 被 _convert2ma 类对象 _convert2ma 包装
empty: _convert2ma

# 函数 empty_like 被 _convert2ma 类对象 _convert2ma 包装
empty_like: _convert2ma

# 函数 frombuffer 被 _convert2ma 类对象 _convert2ma 包装
frombuffer: _convert2ma

# 函数 fromfunction 被 _convert2ma 类对象 _convert2ma 包装
fromfunction: _convert2ma

# 函数 identity 被 _convert2ma 类对象 _convert2ma 包装
identity: _convert2ma

# 函数 ones 被 _convert2ma 类对象 _convert2ma 包装
ones: _convert2ma

# 函数 zeros 被 _convert2ma 类对象 _convert2ma 包装
zeros: _convert2ma

# 返回沿指定轴追加数组 b 到数组 a 的结果
def append(a, b, axis=...): ...

# 返回数组 a 和 b 的点积结果
def dot(a, b, strict=..., out=...): ...

# 在数组 a 上按照指定轴 axis 屏蔽行或列
def mask_rowcols(a, axis=...): ...
```