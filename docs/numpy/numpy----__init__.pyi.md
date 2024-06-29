# `D:\src\scipysrc\numpy\numpy\__init__.pyi`

```py
# 导入内置模块和第三方库
import builtins              # 内置模块，提供Python的内置函数和异常
import sys                   # 内置模块，提供与Python解释器相关的功能
import os                    # 内置模块，提供与操作系统交互的功能
import mmap                  # 内置模块，提供内存映射文件的支持
import ctypes as ct          # 内置模块，提供与C语言兼容的数据类型定义和函数调用接口
import array as _array       # 内置模块，提供高效的数值数组操作支持
import datetime as dt        # 内置模块，提供日期和时间处理的支持
import enum                  # 内置模块，提供枚举类型的支持
from abc import abstractmethod  # 从abc模块中导入abstractmethod装饰器，用于抽象方法的声明

import numpy as np           # 导入NumPy库，提供快速数组操作和数学函数
from numpy._pytesttester import PytestTester  # 导入NumPy的测试工具
from numpy._core._internal import _ctypes  # 导入NumPy内部使用的_ctypes模块

from numpy._typing import (
    # Arrays
    ArrayLike,                 # 定义数组类似类型的类型提示
    NDArray,                   # 定义NumPy数组类型的类型提示
    _SupportsArray,            # 定义支持数组操作的类型提示
    _NestedSequence,           # 定义嵌套序列类型的类型提示
    _FiniteNestedSequence,     # 定义有限嵌套序列类型的类型提示
    _SupportsArray,            # 定义支持数组操作的类型提示（重复定义，可能是误导性注释）
    _ArrayLikeBool_co,         # 定义布尔类型的数组类似类型的协变类型提示
    _ArrayLikeUInt_co,         # 定义无符号整数类型的数组类似类型的协变类型提示
    _ArrayLikeInt_co,          # 定义有符号整数类型的数组类似类型的协变类型提示
    _ArrayLikeFloat_co,        # 定义浮点数类型的数组类似类型的协变类型提示
    _ArrayLikeComplex_co,      # 定义复数类型的数组类似类型的协变类型提示
    _ArrayLikeNumber_co,       # 定义数字类型的数组类似类型的协变类型提示
    _ArrayLikeTD64_co,         # 定义时间日期类型的数组类似类型的协变类型提示
    _ArrayLikeDT64_co,         # 定义时间日期类型的数组类似类型的协变类型提示
    _ArrayLikeObject_co,       # 定义对象类型的数组类似类型的协变类型提示
    _ArrayLikeStr_co,          # 定义字符串类型的数组类似类型的协变类型提示
    _ArrayLikeBytes_co,        # 定义字节类型的数组类似类型的协变类型提示
    _ArrayLikeUnknown,         # 定义未知类型的数组类似类型提示
    _UnknownType,              # 定义未知类型的类型提示

    # DTypes
    DTypeLike,                 # 定义数据类型类似类型的类型提示
    _DTypeLike,                # 定义数据类型类似类型的类型提示
    _DTypeLikeVoid,            # 定义空类型的数据类型类似类型的类型提示
    _SupportsDType,            # 定义支持数据类型操作的类型提示
    _VoidDTypeLike,            # 定义空类型的数据类型类似类型的类型提示

    # Shapes
    _Shape,                    # 定义形状类型的类型提示
    _ShapeLike,                # 定义形状类似类型的类型提示

    # Scalars
    _CharLike_co,              # 定义字符类似类型的协变类型提示
    _IntLike_co,               # 定义整数类似类型的协变类型提示
    _FloatLike_co,             # 定义浮点数类似类型的协变类型提示
    _TD64Like_co,              # 定义时间日期类型类似类型的协变类型提示
    _NumberLike_co,            # 定义数字类似类型的协变类型提示
    _ScalarLike_co,            # 定义标量类似类型的协变类型提示

    # `number` precision
    NBitBase,                  # 定义N位数的基础类型提示
    # NOTE: Do not remove the extended precision bit-types even if seemingly unused;
    # they're used by the mypy plugin
    _256Bit, _128Bit, _96Bit, _80Bit, _64Bit, _32Bit, _16Bit, _8Bit,  # 定义不同位数的整数和浮点数类型提示
    _NBitByte, _NBitShort, _NBitIntC, _NBitIntP, _NBitInt, _NBitLong,   # 续定义不同位数的整数类型提示
    _NBitLongLong, _NBitHalf, _NBitSingle, _NBitDouble, _NBitLongDouble,  # 续定义不同位数的浮点数类型提示

    # Character codes
    _BoolCodes,                # 定义布尔类型的字符代码类型提示
    _UInt8Codes,               # 定义无符号8位整数类型的字符代码类型提示
    _UInt16Codes,              # 定义无符号16位整数类型的字符代码类型提示
    _UInt32Codes,              # 定义无符号32位整数类型的字符代码类型提示
    _UInt64Codes,              # 定义无符号64位整数类型的字符代码类型提示
    _Int8Codes,                # 定义有符号8位整数类型的字符代码类型提示
    _Int16Codes,               # 定义有符号16位整数类型的字符代码类型提示
    _Int32Codes,               # 定义有符号32位整数类型的字符代码类型提示
    _Int64Codes,               # 定义有符号64位整数类型的字符代码类型提示
    _Float16Codes,             # 定义16位浮点数类型的字符代码类型提示
    _Float32Codes,             # 定义32位浮点数类型的字符代码类型提示
    _Float64Codes,             # 定义64位浮点数类型的字符代码类型提示
    _Complex64Codes,           # 定义64位复数类型的字符代码类型提示
    _Complex128Codes,          # 定义128位复数类型的字符代码类型提示
    _ByteCodes,                # 定义字节类型的字符代码类型提示
    _ShortCodes,               # 定义短整数类型的字符代码类型提示
    _IntCCodes,                # 定义C语言整数类型的字符代码类型提示
    _IntPCodes,                # 定义平台相关的整数类型的字符代码类型提示
    _LongCodes,                # 定义长整数类型的字符代码类型提示
    _LongLongCodes,            # 定义长长整数类型的字符代码类型提示
    _UByteCodes,               # 定义无符号字节类型的字符代码类型提示
    _UShortCodes,              # 定义无符号短整数类型的字符代码类型提示
    _UIntCCodes,               # 定义无符号C语言整数类型的字符代码类型提示
    _UIntPCodes,               # 定义无符号平台相关整数类型的字符代码类型提示
    _ULongCodes,               # 定义无符号长整数类型的字符代码类型提示
    _ULongLongCodes,           # 定义无符号长长整数类型的字符代码类型提示
    _HalfCodes,                # 定义半精度浮点数类型的字符代码类型提示
    _SingleCodes,              # 定义单精度浮点数类型的字符代码类型提示
    _DoubleCodes,              # 定义双精度浮点数类型的字符代码类型提示
    _LongDoubleCodes,          # 定义长双精度浮点数类型的字符代码类型
    # 定义不同精度的数值类型别名，如 uint256 表示无符号 256 位整数，int128 表示有符号 128 位整数
    uint256 as uint256,
    int128 as int128,
    int256 as int256,
    float80 as float80,
    float96 as float96,
    float128 as float128,
    float256 as float256,
    complex160 as complex160,
    complex192 as complex192,
    complex256 as complex256,
    complex512 as complex512,
# 导入必要的模块和函数
from numpy._array_api_info import __array_namespace_info__ as __array_namespace_info__
# 导入抽象基类集合，用于声明类型约束
from collections.abc import (
    Callable,       # 可调用对象
    Iterable,       # 可迭代对象
    Iterator,       # 迭代器
    Mapping,        # 映射类型
    Sequence,       # 序列类型
)
# 导入类型提示，用于声明变量和函数的参数类型
from typing import (
    Literal as L,                  # 字面量类型别名
    Any,                          # 任意类型
    Generator,                    # 生成器类型
    Generic,                      # 泛型类型
    NoReturn,                     # 表示函数没有返回值
    overload,                     # 函数重载的装饰器
    SupportsComplex,              # 支持复数类型
    SupportsFloat,                # 支持浮点数类型
    SupportsInt,                  # 支持整数类型
    TypeVar,                      # 泛型变量
    Protocol,                     # 协议类型
    SupportsIndex,                # 支持索引类型
    Final,                        # 最终变量声明
    final,                        # 最终方法声明
    ClassVar,                     # 类变量声明
)

# 确保各个子模块能被正确引用
from numpy import (
    ctypeslib as ctypeslib,        # C 类型库
    exceptions as exceptions,      # 异常模块
    fft as fft,                    # 快速傅里叶变换
    lib as lib,                    # 核心库
    linalg as linalg,              # 线性代数函数
    ma as ma,                      # 缺失值处理
    polynomial as polynomial,      # 多项式操作
    random as random,              # 随机数生成
    testing as testing,            # 测试框架
    version as version,            # NumPy 版本信息
    exceptions as exceptions,      # 异常模块（再次导入，可能是错误）
    dtypes as dtypes,              # 数据类型
    rec as rec,                    # 记录数组操作
    char as char,                  # 字符串操作
    strings as strings,            # 字符串函数
)

# 导入记录数组相关模块和类
from numpy._core.records import (
    record as record,              # 记录数组
    recarray as recarray,          # 记录数组
)

# 导入字符数组相关模块和类
from numpy._core.defchararray import (
    chararray as chararray,        # 字符数组
)

# 导入基本函数操作模块
from numpy._core.function_base import (
    linspace as linspace,          # 线性间隔数组
    logspace as logspace,          # 对数间隔数组
    geomspace as geomspace,        # 几何间隔数组
)

# 导入数组操作函数
from numpy._core.fromnumeric import (
    take as take,                  # 从数组中获取元素
    reshape as reshape,            # 重新塑形数组
    choose as choose,              # 从数组中选择元素
    repeat as repeat,              # 重复数组元素
    put as put,                    # 将元素放置到数组中
    swapaxes as swapaxes,          # 交换数组的轴
    transpose as transpose,        # 转置数组
    matrix_transpose as matrix_transpose,  # 矩阵转置
    partition as partition,        # 对数组进行分区操作
    argpartition as argpartition,  # 对数组进行分区的索引
    sort as sort,                  # 对数组进行排序
    argsort as argsort,            # 对数组进行排序并返回索引
    argmax as argmax,              # 返回数组中最大元素的索引
    argmin as argmin,              # 返回数组中最小元素的索引
    searchsorted as searchsorted,  # 在排序数组中搜索元素
    resize as resize,              # 调整数组的形状和大小
    squeeze as squeeze,            # 压缩数组中的单维度
    diagonal as diagonal,          # 获取数组的对角线元素
    trace as trace,                # 计算数组的迹
    ravel as ravel,                # 展平多维数组
    nonzero as nonzero,            # 返回数组中非零元素的索引
    shape as shape,                # 返回数组的形状
    compress as compress,          # 压缩数组，根据给定条件返回选择的元素
    clip as clip,                  # 将数组中的元素限制在给定范围内
    sum as sum,                    # 计算数组元素的总和
    all as all,                    # 判断数组中所有元素是否为真
    any as any,                    # 判断数组中是否有任意一个元素为真
    cumsum as cumsum,              # 计算数组元素的累积和
    ptp as ptp,                    # 计算数组元素的峰-峰值
    max as max,                    # 返回数组中的最大值
    min as min,                    # 返回数组中的最小值
    amax as amax,                  # 返回数组中的最大值（同 max）
    amin as amin,                  # 返回数组中的最小值（同 min）
    prod as prod,                  # 计算数组元素的乘积
    cumprod as cumprod,            # 计算数组元素的累积乘积
    ndim as ndim,                  # 返回数组的维度数
    size as size,                  # 返回数组中元素的总数
    around as around,              # 对数组进行四舍五入
    round as round,                # 对数组进行四舍五入（同 around）
    mean as mean,                  # 计算数组元素的平均值
    std as std,                    # 计算数组元素的标准差
    var as var,                    # 计算数组元素的方差
)

# 导入数组转换相关函数
from numpy._core._asarray import (
    require as require,            # 将输入转换为数组
)

# 导入标量类型字典
from numpy._core._type_aliases import (
    sctypeDict as sctypeDict,      # 标量类型字典
)

# 导入通用函数配置
from numpy._core._ufunc_config import (
    seterr as seterr,              # 设置浮点错误处理
    geterr as geterr,              # 获取浮点错误处理的当前设置
    setbufsize as setbufsize,      # 设置通用函数缓冲区大小
    getbufsize as getbufsize,      # 获取通用函数缓冲区大小
    seterrcall as seterrcall,      # 设置浮点错误处理的回调函数
    geterrcall as geterrcall,      # 获取浮点错误处理的回调函数
    _ErrKind,                      # 错误类型枚举
    _ErrFunc,                      # 错误处理函数类型
)

# 导入数组打印相关函数和设置
from numpy._core.arrayprint import (
    set_printoptions as set_printoptions,      # 设置数组打印选项
    get_printoptions as get_printoptions,      # 获取数组打印选项
    array2string as array2string,              # 将数组转换为字符串
    format_float_scientific as format_float_scientific,  # 科学计数法格式化浮点数
    format_float_positional as format_float_positional,      # 位置计数法格式化浮点数
    array_repr as array_repr,                  # 返回数组的字符串表示形式
    array_str as array_str,                    # 返回数组的字符串表示形式
    printoptions as printoptions,              # 打印选项类
)

# 导入 einsum 函数及相关路径计算函数
from numpy._core.einsumfunc import (
    einsum as einsum,                          # 构造张量乘积的字符串表示
    einsum_path as einsum_path,                # 高效计算 einsum 的路径
)

# 导入多维数组相关操作
from numpy._core.multiarray import (
    array as array,                            # 创建数组
    empty_like as empty_like,                  # 创建与给定数组具有相同形状的空数组
    empty as empty
    # 导入需要的函数和对象
    from numpy import (
        zeros,                 # 创建全零数组
        concatenate,           # 拼接数组
        inner,                 # 计算内积
        where,                 # 条件查找
        lexsort,               # 对数组进行间接排序
        can_cast,              # 检查类型转换是否安全
        min_scalar_type,       # 查找最小标量类型
        result_type,           # 查找结果的数据类型
        dot,                   # 矩阵乘积
        vdot,                  # 向量内积
        bincount,              # 计算整数数组中每个非负整数的出现次数
        copyto,                # 将源数组复制到目标数组
        putmask,               # 根据掩码向数组赋值
        packbits,              # 将布尔数组打包为字节
        unpackbits,            # 将字节解包为布尔数组
        shares_memory,         # 检查两个数组是否共享内存
        may_share_memory,      # 检查两个数组是否可能共享内存
        asarray,               # 将输入转换为数组
        asanyarray,            # 将输入转换为任意数组
        ascontiguousarray,     # 返回内存连续的数组
        asfortranarray,        # 返回Fortran（列主序）数组
        arange,                # 创建等差数组
        busday_count,          # 计算工作日数量
        busday_offset,         # 计算工作日偏移量
        datetime_as_string,    # 将日期时间数组转换为字符串数组
        datetime_data,         # 访问日期时间数据
        frombuffer,            # 从缓冲区创建数组
        fromfile,              # 从文件中读取数据创建数组
        fromiter,              # 从可迭代对象创建数组
        is_busday,             # 检查日期是否为工作日
        promote_types,         # 推广两个数据类型
        fromstring,            # 从字符串创建数组
        frompyfunc,            # 从Python函数创建通用函数
        nested_iters,          # 生成多层嵌套迭代器
        flagsobj               # 标志对象，用于传递参数
    )
# 导入所需的函数和类，来自 numpy 库的不同子模块

from numpy._core.numeric import (
    zeros_like as zeros_like,            # 导入 zeros_like 函数，并命名为 zeros_like
    ones as ones,                        # 导入 ones 函数，并命名为 ones
    ones_like as ones_like,              # 导入 ones_like 函数，并命名为 ones_like
    full as full,                        # 导入 full 函数，并命名为 full
    full_like as full_like,              # 导入 full_like 函数，并命名为 full_like
    count_nonzero as count_nonzero,      # 导入 count_nonzero 函数，并命名为 count_nonzero
    isfortran as isfortran,              # 导入 isfortran 函数，并命名为 isfortran
    argwhere as argwhere,                # 导入 argwhere 函数，并命名为 argwhere
    flatnonzero as flatnonzero,          # 导入 flatnonzero 函数，并命名为 flatnonzero
    correlate as correlate,              # 导入 correlate 函数，并命名为 correlate
    convolve as convolve,                # 导入 convolve 函数，并命名为 convolve
    outer as outer,                      # 导入 outer 函数，并命名为 outer
    tensordot as tensordot,              # 导入 tensordot 函数，并命名为 tensordot
    roll as roll,                        # 导入 roll 函数，并命名为 roll
    rollaxis as rollaxis,                # 导入 rollaxis 函数，并命名为 rollaxis
    moveaxis as moveaxis,                # 导入 moveaxis 函数，并命名为 moveaxis
    cross as cross,                      # 导入 cross 函数，并命名为 cross
    indices as indices,                  # 导入 indices 函数，并命名为 indices
    fromfunction as fromfunction,        # 导入 fromfunction 函数，并命名为 fromfunction
    isscalar as isscalar,                # 导入 isscalar 函数，并命名为 isscalar
    binary_repr as binary_repr,          # 导入 binary_repr 函数，并命名为 binary_repr
    base_repr as base_repr,              # 导入 base_repr 函数，并命名为 base_repr
    identity as identity,                # 导入 identity 函数，并命名为 identity
    allclose as allclose,                # 导入 allclose 函数，并命名为 allclose
    isclose as isclose,                  # 导入 isclose 函数，并命名为 isclose
    array_equal as array_equal,          # 导入 array_equal 函数，并命名为 array_equal
    array_equiv as array_equiv,          # 导入 array_equiv 函数，并命名为 array_equiv
    astype as astype,                    # 导入 astype 函数，并命名为 astype
)

from numpy._core.numerictypes import (
    isdtype as isdtype,                  # 导入 isdtype 函数，并命名为 isdtype
    issubdtype as issubdtype,            # 导入 issubdtype 函数，并命名为 issubdtype
    cast as cast,                        # 导入 cast 函数，并命名为 cast
    ScalarType as ScalarType,            # 导入 ScalarType 类，并命名为 ScalarType
    typecodes as typecodes,              # 导入 typecodes 变量，并命名为 typecodes
)

from numpy._core.shape_base import (
    atleast_1d as atleast_1d,            # 导入 atleast_1d 函数，并命名为 atleast_1d
    atleast_2d as atleast_2d,            # 导入 atleast_2d 函数，并命名为 atleast_2d
    atleast_3d as atleast_3d,            # 导入 atleast_3d 函数，并命名为 atleast_3d
    block as block,                      # 导入 block 函数，并命名为 block
    hstack as hstack,                    # 导入 hstack 函数，并命名为 hstack
    stack as stack,                      # 导入 stack 函数，并命名为 stack
    vstack as vstack,                    # 导入 vstack 函数，并命名为 vstack
)

from numpy.lib import (
    scimath as emath,                    # 导入 scimath 模块，并命名为 emath
)

from numpy.lib._arraypad_impl import (
    pad as pad,                          # 导入 pad 函数，并命名为 pad
)

from numpy.lib._arraysetops_impl import (
    ediff1d as ediff1d,                  # 导入 ediff1d 函数，并命名为 ediff1d
    intersect1d as intersect1d,          # 导入 intersect1d 函数，并命名为 intersect1d
    isin as isin,                        # 导入 isin 函数，并命名为 isin
    setdiff1d as setdiff1d,              # 导入 setdiff1d 函数，并命名为 setdiff1d
    setxor1d as setxor1d,                # 导入 setxor1d 函数，并命名为 setxor1d
    union1d as union1d,                  # 导入 union1d 函数，并命名为 union1d
    unique as unique,                    # 导入 unique 函数，并命名为 unique
    unique_all as unique_all,            # 导入 unique_all 函数，并命名为 unique_all
    unique_counts as unique_counts,      # 导入 unique_counts 函数，并命名为 unique_counts
    unique_inverse as unique_inverse,    # 导入 unique_inverse 函数，并命名为 unique_inverse
    unique_values as unique_values,      # 导入 unique_values 函数，并命名为 unique_values
)

from numpy.lib._function_base_impl import (
    select as select,                    # 导入 select 函数，并命名为 select
    piecewise as piecewise,              # 导入 piecewise 函数，并命名为 piecewise
    trim_zeros as trim_zeros,            # 导入 trim_zeros 函数，并命名为 trim_zeros
    copy as copy,                        # 导入 copy 函数，并命名为 copy
    iterable as iterable,                # 导入 iterable 函数，并命名为 iterable
    percentile as percentile,            # 导入 percentile 函数，并命名为 percentile
    diff as diff,                        # 导入 diff 函数，并命名为 diff
    gradient as gradient,                # 导入 gradient 函数，并命名为 gradient
    angle as angle,                      # 导入 angle 函数，并命名为 angle
    unwrap as unwrap,                    # 导入 unwrap 函数，并命名为 unwrap
    sort_complex as sort_complex,        # 导入 sort_complex 函数，并命名为 sort_complex
    disp as disp,                        # 导入 disp 函数，并命名为 disp
    flip as flip,                        # 导入 flip 函数，并命名为 flip
    rot90 as rot90,                      # 导入 rot90 函数，并命名为 rot90
    extract as extract,                  # 导入 extract 函数，并命名为 extract
    place as place,                      # 导入 place 函数，并命名为 place
    asarray_chkfinite as asarray_chkfinite,  # 导入 asarray_chkfinite 函数，并命名为 asarray_chkfinite
    average as average,                  # 导入 average 函数，并命名为 average
    bincount as bincount,                # 导入 bincount 函数，并命名为 bincount
    digitize as digitize,                # 导入 digitize 函数，并命名为 digitize
    cov as cov,                          # 导入 cov 函数，并命名为 cov
    corrcoef as corrcoef,                # 导入 corrcoef 函数，并命名为 corrcoef
    median as median,                    # 导入 median 函数，并命名为 median
    sinc as sinc,                        # 导入 sinc 函数，并命名为 sinc
    hamming as hamming,                  # 导入 hamming 函数，并命名为 hamming
    hanning as hanning,                  # 导入 hanning 函数，并命名为 hanning
    bartlett as bartlett,                # 导入 bartlett 函数，并命名为 bartlett
    blackman as blackman,                # 导入 blackman 函数，并命名为 blackman
    kaiser as kaiser,                    # 导入 kaiser 函数，并命名为 kaiser
    i0 as i0,                            # 导入 i0 函数，并命名为 i0
    meshgrid as meshgrid,                # 导入 meshgrid 函数，并命名为 meshgrid
    delete as delete,                    # 导入 delete 函数，并命名为 delete
    insert as insert,                    # 导入 insert 函数，并命名为 insert
    append as append,                    # 导入 append 函数
    # 导入 diag_indices 和 diag_indices_from 函数并重命名，使其可以直接使用
    diag_indices as diag_indices,
    diag_indices_from as diag_indices_from,
# 导入以下来自numpy库的一组函数，这些函数涵盖了_nanfunctions_impl模块
from numpy.lib._nanfunctions_impl import (
    nansum as nansum,  # 导入nansum函数并重命名为nansum
    nanmax as nanmax,  # 导入nanmax函数并重命名为nanmax
    nanmin as nanmin,  # 导入nanmin函数并重命名为nanmin
    nanargmax as nanargmax,  # 导入nanargmax函数并重命名为nanargmax
    nanargmin as nanargmin,  # 导入nanargmin函数并重命名为nanargmin
    nanmean as nanmean,  # 导入nanmean函数并重命名为nanmean
    nanmedian as nanmedian,  # 导入nanmedian函数并重命名为nanmedian
    nanpercentile as nanpercentile,  # 导入nanpercentile函数并重命名为nanpercentile
    nanvar as nanvar,  # 导入nanvar函数并重命名为nanvar
    nanstd as nanstd,  # 导入nanstd函数并重命名为nanstd
    nanprod as nanprod,  # 导入nanprod函数并重命名为nanprod
    nancumsum as nancumsum,  # 导入nancumsum函数并重命名为nancumsum
    nancumprod as nancumprod,  # 导入nancumprod函数并重命名为nancumprod
    nanquantile as nanquantile,  # 导入nanquantile函数并重命名为nanquantile
)

# 导入以下来自numpy库的一组函数，这些函数涵盖了_npyio_impl模块
from numpy.lib._npyio_impl import (
    savetxt as savetxt,  # 导入savetxt函数并重命名为savetxt
    loadtxt as loadtxt,  # 导入loadtxt函数并重命名为loadtxt
    genfromtxt as genfromtxt,  # 导入genfromtxt函数并重命名为genfromtxt
    load as load,  # 导入load函数并重命名为load
    save as save,  # 导入save函数并重命名为save
    savez as savez,  # 导入savez函数并重命名为savez
    savez_compressed as savez_compressed,  # 导入savez_compressed函数并重命名为savez_compressed
    packbits as packbits,  # 导入packbits函数并重命名为packbits
    unpackbits as unpackbits,  # 导入unpackbits函数并重命名为unpackbits
    fromregex as fromregex,  # 导入fromregex函数并重命名为fromregex
)

# 导入以下来自numpy库的一组函数，这些函数涵盖了_polynomial_impl模块
from numpy.lib._polynomial_impl import (
    poly as poly,  # 导入poly函数并重命名为poly
    roots as roots,  # 导入roots函数并重命名为roots
    polyint as polyint,  # 导入polyint函数并重命名为polyint
    polyder as polyder,  # 导入polyder函数并重命名为polyder
    polyadd as polyadd,  # 导入polyadd函数并重命名为polyadd
    polysub as polysub,  # 导入polysub函数并重命名为polysub
    polymul as polymul,  # 导入polymul函数并重命名为polymul
    polydiv as polydiv,  # 导入polydiv函数并重命名为polydiv
    polyval as polyval,  # 导入polyval函数并重命名为polyval
    polyfit as polyfit,  # 导入polyfit函数并重命名为polyfit
)

# 导入以下来自numpy库的一组函数，这些函数涵盖了_shape_base_impl模块
from numpy.lib._shape_base_impl import (
    column_stack as column_stack,  # 导入column_stack函数并重命名为column_stack
    dstack as dstack,  # 导入dstack函数并重命名为dstack
    array_split as array_split,  # 导入array_split函数并重命名为array_split
    split as split,  # 导入split函数并重命名为split
    hsplit as hsplit,  # 导入hsplit函数并重命名为hsplit
    vsplit as vsplit,  # 导入vsplit函数并重命名为vsplit
    dsplit as dsplit,  # 导入dsplit函数并重命名为dsplit
    apply_over_axes as apply_over_axes,  # 导入apply_over_axes函数并重命名为apply_over_axes
    expand_dims as expand_dims,  # 导入expand_dims函数并重命名为expand_dims
    apply_along_axis as apply_along_axis,  # 导入apply_along_axis函数并重命名为apply_along_axis
    kron as kron,  # 导入kron函数并重命名为kron
    tile as tile,  # 导入tile函数并重命名为tile
    take_along_axis as take_along_axis,  # 导入take_along_axis函数并重命名为take_along_axis
    put_along_axis as put_along_axis,  # 导入put_along_axis函数并重命名为put_along_axis
)

# 导入以下来自numpy库的一组函数，这些函数涵盖了_stride_tricks_impl模块
from numpy.lib._stride_tricks_impl import (
    broadcast_to as broadcast_to,  # 导入broadcast_to函数并重命名为broadcast_to
    broadcast_arrays as broadcast_arrays,  # 导入broadcast_arrays函数并重命名为broadcast_arrays
    broadcast_shapes as broadcast_shapes,  # 导入broadcast_shapes函数并重命名为broadcast_shapes
)

# 导入以下来自numpy库的一组函数，这些函数涵盖了_twodim_base_impl模块
from numpy.lib._twodim_base_impl import (
    diag as diag,  # 导入diag函数并重命名为diag
    diagflat as diagflat,  # 导入diagflat函数并重命名为diagflat
    eye as eye,  # 导入eye函数并重命名为eye
    fliplr as fliplr,  # 导入fliplr函数并重命名为fliplr
    flipud as flipud,  # 导入flipud函数并重命名为flipud
    tri as tri,  # 导入tri函数并重命名为tri
    triu as triu,  # 导入triu函数并重命名为triu
    tril as tril,  # 导入tril函数并重命名为tril
    vander as vander,  # 导入vander函数并重命名为vander
    histogram2d as histogram2d,  # 导入histogram2d函数并重命名为histogram2d
    mask_indices as mask_indices,  # 导入mask_indices函数并重命名为mask_indices
    tril_indices as tril_indices,  # 导入tril_indices函数并重命名为tril_indices
    tril_indices_from as tril_indices_from,  # 导入tril_indices_from函数并重命名为tril_indices_from
    triu_indices as triu_indices,  # 导入triu_indices函数并重命名为triu_indices
    triu_indices_from as triu_indices_from,  # 导入triu_indices_from函数并重命名为triu_indices_from
)

# 导入以下来自numpy库的一组函数，这些函数涵盖了_type_check_impl模块
from numpy.lib._type_check_impl import (
    mintypecode as mintypecode,  # 导入mintypecode函数并重命名为mintypecode
    real as real,  # 导入real函数并重命名为real
    imag as imag,  # 导入imag函数并重命名为imag
    iscomplex as iscomplex,  # 导入iscomplex函数并重命名为iscomplex
    isreal as isreal,  # 导入isreal函数并重命名为isreal
    iscomplexobj as iscomplexobj,  # 导入iscomplexobj函数并重命名为iscomplexobj
    isrealobj as isrealobj,  # 导入isrealobj函数并重命名为isrealobj
    nan_to_num as
    # 定义一个方法 tell，返回类型为 SupportsIndex
    def tell(self) -> SupportsIndex: ...
    
    # 定义一个方法 seek，接受 offset（偏移量）和 whence（起始位置），返回类型为 object
    def seek(self, offset: int, whence: int, /) -> object: ...
# NOTE: `seek`, `write` and `flush` are technically only required
# for `readwrite`/`write` modes
# 定义了一个协议 `_MemMapIOProtocol`，包含了文件映射 IO 操作需要的方法
class _MemMapIOProtocol(Protocol):
    # 刷新缓冲区，返回一个对象
    def flush(self) -> object: ...
    # 获取文件描述符号，返回一个支持索引的对象
    def fileno(self) -> SupportsIndex: ...
    # 返回当前文件指针位置
    def tell(self) -> int: ...
    # 移动文件指针到指定位置，支持偏移量和寻址方式
    def seek(self, offset: int, whence: int, /) -> object: ...
    # 写入字节流到文件，返回一个对象
    def write(self, s: bytes, /) -> object: ...
    # 属性注解，表示可以读取对象
    @property
    def read(self) -> object: ...

# 定义了一个协议 `_SupportsWrite`，指定了支持写操作的方法
class _SupportsWrite(Protocol[_AnyStr_contra]):
    # 写入方法，参数为泛型字符串，返回一个对象
    def write(self, s: _AnyStr_contra, /) -> object: ...

# 声明模块的导出符号列表
__all__: list[str]
# 声明模块的所有属性名列表
__dir__: list[str]
# 声明模块的版本号
__version__: str
# 声明 Git 版本号
__git_version__: str
# 声明数组 API 的版本号
__array_api_version__: str
# Pytest 测试器
test: PytestTester

# TODO: Move placeholders to their respective module once
# their annotations are properly implemented
#
# Placeholders for classes

# 定义一个函数 `show_config`，无返回值
def show_config() -> None: ...

# 定义类型变量 `_NdArraySubClass`，表示数组的子类
_NdArraySubClass = TypeVar("_NdArraySubClass", bound=NDArray[Any])
# 定义类型变量 `_DTypeScalar_co`，表示标量的类型
_DTypeScalar_co = TypeVar("_DTypeScalar_co", covariant=True, bound=generic)
# 定义字节顺序类型 `_ByteOrder`，包含了多种可能的字节顺序
_ByteOrder = L["S", "<", ">", "=", "|", "L", "B", "N", "I", "little", "big", "native"]

# 类型修饰符，表示该类是最终类，不能被继承
@final
# 类 `dtype`，泛型类型 `_DTypeScalar_co`
class dtype(Generic[_DTypeScalar_co]):
    # 属性 `names`，可以为 `None` 或者字符串元组
    names: None | tuple[builtins.str, ...]
    # 哈希函数，返回一个整数
    def __hash__(self) -> int: ...
    # 重载方法，用于创建泛型的子类
    @overload
    def __new__(
        cls,
        dtype: type[_DTypeScalar_co],
        align: builtins.bool = ...,
        copy: builtins.bool = ...,
        metadata: dict[builtins.str, Any] = ...,
    ) -> dtype[_DTypeScalar_co]: ...
    # 重载方法，针对布尔类型
    @overload
    def __new__(cls, dtype: type[builtins.bool], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[np.bool]: ...
    # 重载方法，针对整数类型
    @overload
    def __new__(cls, dtype: type[int], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[int_]: ...
    # 重载方法，针对浮点数类型
    @overload
    def __new__(cls, dtype: None | type[float], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[float64]: ...
    # 重载方法，针对复数类型
    @overload
    def __new__(cls, dtype: type[complex], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[complex128]: ...
    # 重载方法，针对字符串类型
    @overload
    def __new__(cls, dtype: type[builtins.str], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[str_]: ...
    # 重载方法，针对字节类型
    @overload
    def __new__(cls, dtype: type[bytes], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[bytes_]: ...
    # 无符号整数的字符串表示和 ctypes
    @overload
    # 定义一个特殊的方法 `__new__()`，用于创建新的实例对象
    def __new__(cls, dtype: _UInt8Codes | type[ct.c_uint8], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[uint8]:
        ...
    
    # 使用装饰器 `@overload`，表示下面的函数是 `__new__()` 方法的重载版本
    @overload
    def __new__(cls, dtype: _UInt16Codes | type[ct.c_uint16], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[uint16]:
        ...
    
    # 同上，定义不同参数类型的 `__new__()` 方法的重载版本
    @overload
    def __new__(cls, dtype: _UInt32Codes | type[ct.c_uint32], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[uint32]:
        ...
    
    @overload
    def __new__(cls, dtype: _UInt64Codes | type[ct.c_uint64], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[uint64]:
        ...
    
    @overload
    def __new__(cls, dtype: _UByteCodes | type[ct.c_ubyte], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[ubyte]:
        ...
    
    @overload
    def __new__(cls, dtype: _UShortCodes | type[ct.c_ushort], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[ushort]:
        ...
    
    @overload
    def __new__(cls, dtype: _UIntCCodes | type[ct.c_uint], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[uintc]:
        ...
    
    # 根据传入的参数类型不同，定义 `__new__()` 方法的重载版本
    @overload
    def __new__(cls, dtype: _UIntPCodes | type[ct.c_void_p] | type[ct.c_size_t], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[uintp]:
        ...
    
    @overload
    def __new__(cls, dtype: _ULongCodes | type[ct.c_ulong], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[ulong]:
        ...
    
    @overload
    def __new__(cls, dtype: _ULongLongCodes | type[ct.c_ulonglong], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[ulonglong]:
        ...
    
    # 使用装饰器 `@overload`，表示下面的函数是 `__new__()` 方法的重载版本
    @overload
    def __new__(cls, dtype: _Int8Codes | type[ct.c_int8], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[int8]:
        ...
    
    @overload
    def __new__(cls, dtype: _Int16Codes | type[ct.c_int16], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[int16]:
        ...
    
    @overload
    def __new__(cls, dtype: _Int32Codes | type[ct.c_int32], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[int32]:
        ...
    
    @overload
    def __new__(cls, dtype: _Int64Codes | type[ct.c_int64], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[int64]:
        ...
    def __new__(cls, dtype: _ByteCodes | type[ct.c_byte], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[byte]: ...

这是一个特殊方法 `__new__` 的定义，用于创建新的实例。它使用了类型注解来指定参数和返回类型。参数包括 `dtype`，它可以是 `_ByteCodes` 或 `ct.c_byte` 类型之一；`align` 是一个布尔型参数，表示是否对齐；`copy` 也是一个布尔型参数，表示是否复制；`metadata` 是一个字典，包含字符串键和任意类型的值。返回类型根据 `dtype` 的不同而不同，可以是 `dtype[byte]` 类型。


    @overload
    def __new__(cls, dtype: _ShortCodes | type[ct.c_short], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[short]: ...

这是 `__new__` 方法的重载，处理 `_ShortCodes` 或 `ct.c_short` 类型的 `dtype`，具有相同的 `align`、`copy` 和 `metadata` 参数，返回类型为 `dtype[short]`。


    @overload
    def __new__(cls, dtype: _IntCCodes | type[ct.c_int], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[intc]: ...

这是 `__new__` 方法的另一个重载，处理 `_IntCCodes` 或 `ct.c_int` 类型的 `dtype`，参数和返回类型分别指定为 `dtype[intc]`。


    @overload
    def __new__(cls, dtype: _IntPCodes | type[ct.c_ssize_t], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[intp]: ...

这是 `__new__` 方法的重载，处理 `_IntPCodes` 或 `ct.c_ssize_t` 类型的 `dtype`，参数和返回类型分别指定为 `dtype[intp]`。


    @overload
    def __new__(cls, dtype: _LongCodes | type[ct.c_long], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[long]: ...

这是 `__new__` 方法的重载，处理 `_LongCodes` 或 `ct.c_long` 类型的 `dtype`，参数和返回类型分别指定为 `dtype[long]`。


    @overload
    def __new__(cls, dtype: _LongLongCodes | type[ct.c_longlong], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[longlong]: ...

这是 `__new__` 方法的重载，处理 `_LongLongCodes` 或 `ct.c_longlong` 类型的 `dtype`，参数和返回类型分别指定为 `dtype[longlong]`。


    @overload
    def __new__(cls, dtype: _Float16Codes, align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[float16]: ...

这是 `__new__` 方法的重载，处理 `_Float16Codes` 类型的 `dtype`，参数和返回类型分别指定为 `dtype[float16]`。


    @overload
    def __new__(cls, dtype: _Float32Codes, align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[float32]: ...

这是 `__new__` 方法的重载，处理 `_Float32Codes` 类型的 `dtype`，参数和返回类型分别指定为 `dtype[float32]`。


    @overload
    def __new__(cls, dtype: _Float64Codes, align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[float64]: ...

这是 `__new__` 方法的重载，处理 `_Float64Codes` 类型的 `dtype`，参数和返回类型分别指定为 `dtype[float64]`。


    @overload
    def __new__(cls, dtype: _HalfCodes, align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[half]: ...

这是 `__new__` 方法的重载，处理 `_HalfCodes` 类型的 `dtype`，参数和返回类型分别指定为 `dtype[half]`。


    @overload
    def __new__(cls, dtype: _SingleCodes | type[ct.c_float], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[single]: ...

这是 `__new__` 方法的重载，处理 `_SingleCodes` 或 `ct.c_float` 类型的 `dtype`，参数和返回类型分别指定为 `dtype[single]`。


    @overload
    def __new__(cls, dtype: _DoubleCodes | type[ct.c_double], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[double]: ...

这是 `__new__` 方法的重载，处理 `_DoubleCodes` 或 `ct.c_double` 类型的 `dtype`，参数和返回类型分别指定为 `dtype[double]`。


    @overload
    def __new__(cls, dtype: _LongDoubleCodes | type[ct.c_longdouble], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[longdouble]: ...

这是 `__new__` 方法的重载，处理 `_LongDoubleCodes` 或 `ct.c_longdouble` 类型的 `dtype`，参数和返回类型分别指定为 `dtype[longdouble]`。


    @overload
    def __new__(cls, dtype: _Complex64Codes, align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[complex64]: ...

这是 `__new__` 方法的重载，处理 `_Complex64Codes` 类型的 `dtype`，参数和返回类型分别指定为 `dtype[complex64]`。


    @overload
    def __new__(cls, dtype: _Complex128Codes, align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[complex128]: ...

这是 `__new__` 方法的重载，处理 `_Complex128Codes` 类型的 `dtype`，参数和返回类型分别指定为 `dtype[complex128]`。
    def __new__(cls, dtype: _CSingleCodes, align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[csingle]: ...
    # 创建一个特定类型的新实例，接受单精度数据类型参数，返回值的类型是由参数决定的 csingle

    @overload
    def __new__(cls, dtype: _CDoubleCodes, align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[cdouble]: ...
    # 重载 __new__ 方法，处理双精度数据类型参数，返回值类型为 cdouble

    @overload
    def __new__(cls, dtype: _CLongDoubleCodes, align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[clongdouble]: ...
    # 重载 __new__ 方法，处理长双精度数据类型参数，返回值类型为 clongdouble

    # Miscellaneous string-based representations and ctypes

    @overload
    def __new__(cls, dtype: _BoolCodes | type[ct.c_bool], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[np.bool]: ...
    # 处理布尔类型或者 ctypes 的类型参数，返回值类型为 np.bool

    @overload
    def __new__(cls, dtype: _TD64Codes, align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[timedelta64]: ...
    # 处理时间增量类型参数，返回值类型为 timedelta64

    @overload
    def __new__(cls, dtype: _DT64Codes, align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[datetime64]: ...
    # 处理日期时间类型参数，返回值类型为 datetime64

    @overload
    def __new__(cls, dtype: _StrCodes, align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[str_]: ...
    # 处理字符串类型参数，返回值类型为 str_

    @overload
    def __new__(cls, dtype: _BytesCodes | type[ct.c_char], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[bytes_]: ...
    # 处理字节串类型或者 ctypes 的字符类型参数，返回值类型为 bytes_

    @overload
    def __new__(cls, dtype: _VoidCodes, align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[void]: ...
    # 处理空类型参数，返回值类型为 void

    @overload
    def __new__(cls, dtype: _ObjectCodes | type[ct.py_object[Any]], align: builtins.bool = ..., copy: builtins.bool = ..., metadata: dict[builtins.str, Any] = ...) -> dtype[object_]: ...
    # 处理对象类型参数或者 ctypes 的 Python 对象类型参数，返回值类型为 object_

    # dtype of a dtype is the same dtype

    @overload
    def __new__(
        cls,
        dtype: dtype[_DTypeScalar_co],
        align: builtins.bool = ...,
        copy: builtins.bool = ...,
        metadata: dict[builtins.str, Any] = ...,
    ) -> dtype[_DTypeScalar_co]: ...
    # 处理 dtype 的 dtype 类型参数，返回值类型与参数类型相同 _DTypeScalar_co

    @overload
    def __new__(
        cls,
        dtype: _SupportsDType[dtype[_DTypeScalar_co]],
        align: builtins.bool = ...,
        copy: builtins.bool = ...,
        metadata: dict[builtins.str, Any] = ...,
    ) -> dtype[_DTypeScalar_co]: ...
    # 处理支持 dtype 参数的类型，返回值类型为 _DTypeScalar_co

    # Handle strings that can't be expressed as literals; i.e. s1, s2, ...

    @overload
    def __new__(
        cls,
        dtype: builtins.str,
        align: builtins.bool = ...,
        copy: builtins.bool = ...,
        metadata: dict[builtins.str, Any] = ...,
    ) -> dtype[Any]: ...
    # 处理无法表示为字面量的字符串类型参数，返回值类型为任意类型

    # Catchall overload for void-likes

    @overload
    def __new__(
        cls,
        dtype: _VoidDTypeLike,
        align: builtins.bool = ...,
        copy: builtins.bool = ...,
        metadata: dict[builtins.str, Any] = ...,
    ) -> dtype[void]: ...
    # 处理类似空类型的参数，返回值类型为 void

    # Catchall overload for object-likes
    # 定义一个方法重载，用于创建新的对象实例。
    @overload
    def __new__(
        cls,
        dtype: type[object],
        align: builtins.bool = ...,
        copy: builtins.bool = ...,
        metadata: dict[builtins.str, Any] = ...,
    ) -> dtype[object_]: ...

    # 定义一个类方法 __class_getitem__，用于获取类的类型参数。
    def __class_getitem__(self, item: Any) -> GenericAlias: ...

    # 定义一个方法重载，处理对 void 类型的索引操作。
    @overload
    def __getitem__(self: dtype[void], key: list[builtins.str]) -> dtype[void]: ...
    @overload
    def __getitem__(self: dtype[void], key: builtins.str | SupportsIndex) -> dtype[Any]: ...

    # NOTE: 将来基于 1 的乘法也将产生 `flexible` 类型的数据。
    # 定义一个方法重载，处理对象与 L[1] 类型的乘法操作。
    @overload
    def __mul__(self: _DType, value: L[1]) -> _DType: ...
    # 定义一个方法重载，处理对象与 SupportsIndex 类型的乘法操作。
    @overload
    def __mul__(self: _FlexDType, value: SupportsIndex) -> _FlexDType: ...
    # 定义一个方法重载，处理一般情况下对象与 SupportsIndex 类型的乘法操作。
    @overload
    def __mul__(self, value: SupportsIndex) -> dtype[void]: ...

    # NOTE: 当与字面量一起使用时，`__rmul__` 在 mypy 0.902 版本中存在问题。
    # 暂时将非 `flexible` 类型的返回类型设置为 `dtype[Any]`。
    # 定义一个方法重载，处理 `__rmul__` 方法的修复问题。
    @overload
    def __rmul__(self: _FlexDType, value: SupportsIndex) -> _FlexDType: ...
    # 定义一个方法重载，处理一般情况下 `__rmul__` 方法的修复问题。
    @overload
    def __rmul__(self, value: SupportsIndex) -> dtype[Any]: ...

    # 定义一个方法 __gt__，用于处理对象的大于比较操作。
    def __gt__(self, other: DTypeLike) -> builtins.bool: ...
    # 定义一个方法 __ge__，用于处理对象的大于等于比较操作。
    def __ge__(self, other: DTypeLike) -> builtins.bool: ...
    # 定义一个方法 __lt__，用于处理对象的小于比较操作。
    def __lt__(self, other: DTypeLike) -> builtins.bool: ...
    # 定义一个方法 __le__，用于处理对象的小于等于比较操作。
    def __le__(self, other: DTypeLike) -> builtins.bool: ...

    # 明确定义 `__eq__` 和 `__ne__` 方法，以绕过 mypy 的 `strict_equality` 选项。
    # 尽管它们的签名与基于 `object` 的对应方法相同。
    # 定义一个方法 __eq__，用于处理对象的相等比较操作。
    def __eq__(self, other: Any) -> builtins.bool: ...
    # 定义一个方法 __ne__，用于处理对象的不等比较操作。
    def __ne__(self, other: Any) -> builtins.bool: ...

    # 定义一个属性方法 alignment，返回对象的对齐方式。
    @property
    def alignment(self) -> int: ...
    # 定义一个属性方法 base，返回对象的基本数据类型。
    @property
    def base(self) -> dtype[Any]: ...
    # 定义一个属性方法 byteorder，返回对象的字节顺序。
    @property
    def byteorder(self) -> builtins.str: ...
    # 定义一个属性方法 char，返回对象的字符表示。
    @property
    def char(self) -> builtins.str: ...
    # 定义一个属性方法 descr，返回对象的描述信息列表。
    @property
    def descr(self) -> list[tuple[builtins.str, builtins.str] | tuple[builtins.str, builtins.str, _Shape]]: ...
    # 定义一个属性方法 fields，返回对象的字段信息或空值。
    @property
    def fields(
        self,
    ) -> None | MappingProxyType[builtins.str, tuple[dtype[Any], int] | tuple[dtype[Any], int, Any]]: ...
    # 定义一个属性方法 flags，返回对象的标志位。
    @property
    def flags(self) -> int: ...
    # 定义一个属性方法 hasobject，返回对象是否含有对象类型。
    @property
    def hasobject(self) -> builtins.bool: ...
    # 定义一个属性方法 isbuiltin，返回对象是否是内置类型。
    @property
    def isbuiltin(self) -> int: ...
    # 定义一个属性方法 isnative，返回对象是否是原生类型。
    @property
    def isnative(self) -> builtins.bool: ...
    # 定义一个属性方法 isalignedstruct，返回对象是否是对齐结构。
    @property
    def isalignedstruct(self) -> builtins.bool: ...
    # 定义一个属性方法 itemsize，返回对象的每个元素的大小。
    @property
    def itemsize(self) -> int: ...
    # 定义一个属性方法 kind，返回对象的类型。
    @property
    def kind(self) -> builtins.str: ...
    # 定义一个属性方法 metadata，返回对象的元数据或空值。
    @property
    def metadata(self) -> None | MappingProxyType[builtins.str, Any]: ...
    # 定义一个属性方法 name，返回对象的名称。
    @property
    def name(self) -> builtins.str: ...
    # 定义一个属性方法 num，返回对象的编号。
    @property
    def num(self) -> int: ...
    # 定义一个属性方法 shape，返回对象的形状。
    @property
    def shape(self) -> _Shape: ...
    # 定义一个属性方法 ndim，返回对象的维度数。
    @property
    def ndim(self) -> int: ...
    # 定义一个属性方法 subdtype，返回对象的子数据类型或空值。
    @property
    def subdtype(self) -> None | tuple[dtype[Any], _Shape]: ...
    # 定义方法 newbyteorder，用于改变数据类型的字节顺序
    def newbyteorder(self: _DType, __new_order: _ByteOrder = ...) -> _DType:
        # 方法签名说明：self 是 _DType 类型的实例，__new_order 是可选参数，表示新的字节顺序，返回一个新的 _DType 对象
        ...
    
    # 定义属性 str，返回该数据类型的名称字符串
    @property
    def str(self) -> builtins.str:
        # 属性签名说明：self 是当前对象的实例，返回类型为内置的字符串类型 str
        ...
    
    # 定义属性 type，返回该数据类型的 Python 类型
    @property
    def type(self) -> type[_DTypeScalar_co]:
        # 属性签名说明：self 是当前对象的实例，返回类型为 _DTypeScalar_co 表示的类型
        ...
# 定义 `_ArrayLikeInt` 类型别名，可以是单个整数、整数的列表、整数的序列、递归序列（待支持）、NDArray 中的任意类型
_ArrayLikeInt = (
    int
    | integer[Any]
    | Sequence[int | integer[Any]]
    | Sequence[Sequence[Any]]  # TODO: 等待对递归类型的支持
    | NDArray[Any]
)

# 定义 `_FlatIterSelf` 类型变量，用于泛型类 `flatiter`，限定为 `_NdArraySubClass` 的子类
_FlatIterSelf = TypeVar("_FlatIterSelf", bound=flatiter[Any])

# 定义 `flatiter` 泛型类，用于扁平迭代多维数组 `_NdArraySubClass`
@final
class flatiter(Generic[_NdArraySubClass]):
    __hash__: ClassVar[None]  # 类属性，表示对象不可哈希
    @property
    def base(self) -> _NdArraySubClass: ...  # 返回迭代器所基于的 `_NdArraySubClass` 对象
    @property
    def coords(self) -> _Shape: ...  # 返回迭代器当前位置的坐标 `_Shape`
    @property
    def index(self) -> int: ...  # 返回迭代器当前位置的索引
    def copy(self) -> _NdArraySubClass: ...  # 复制迭代器对象并返回 `_NdArraySubClass` 类型
    def __iter__(self: _FlatIterSelf) -> _FlatIterSelf: ...  # 返回迭代器自身 `_FlatIterSelf`
    def __next__(self: flatiter[NDArray[_ScalarType]]) -> _ScalarType: ...  # 返回迭代器的下一个元素 `_ScalarType`
    def __len__(self) -> int: ...  # 返回迭代器的长度（元素数量）
    @overload
    def __getitem__(
        self: flatiter[NDArray[_ScalarType]],
        key: int | integer[Any] | tuple[int | integer[Any]],
    ) -> _ScalarType: ...
    @overload
    def __getitem__(
        self,
        key: _ArrayLikeInt | slice | ellipsis | tuple[_ArrayLikeInt | slice | ellipsis],
    ) -> _NdArraySubClass: ...
    # TODO: `__setitem__` 使用 `unsafe` 的转换规则，可以接受任何与底层 `np.generic` 构造函数兼容的类型。
    # 这意味着 `value` 实际上必须是 `npt.ArrayLike` 的超类型。
    def __setitem__(
        self,
        key: _ArrayLikeInt | slice | ellipsis | tuple[_ArrayLikeInt | slice | ellipsis],
        value: Any,
    ) -> None: ...
    @overload
    def __array__(self: flatiter[ndarray[Any, _DType]], dtype: None = ..., /) -> ndarray[Any, _DType]: ...
    @overload
    def __array__(self, dtype: _DType, /) -> ndarray[Any, _DType]: ...

# `_OrderKACF`, `_OrderACF`, `_OrderCF` 是类型别名，表示不同的排序顺序组合
_OrderKACF = L[None, "K", "A", "C", "F"]
_OrderACF = L[None, "A", "C", "F"]
_OrderCF = L[None, "C", "F"]

# `_ModeKind`, `_PartitionKind`, `_SortKind`, `_SortSide` 是枚举类型别名，表示不同的模式、分区方式、排序方式、排序侧边
_ModeKind = L["raise", "wrap", "clip"]
_PartitionKind = L["introselect"]
_SortKind = L["quicksort", "mergesort", "heapsort", "stable"]
_SortSide = L["left", "right"]

# `_ArraySelf` 是泛型类型变量，限定为 `_ArrayOrScalarCommon` 的子类
_ArraySelf = TypeVar("_ArraySelf", bound=_ArrayOrScalarCommon)

# `_ArrayOrScalarCommon` 类公共基类，表示数组或标量的共有属性和方法
class _ArrayOrScalarCommon:
    @property
    def T(self: _ArraySelf) -> _ArraySelf: ...  # 返回转置后的 `_ArraySelf` 对象
    @property
    def mT(self: _ArraySelf) -> _ArraySelf: ...  # 返回共轭转置后的 `_ArraySelf` 对象
    @property
    def data(self) -> memoryview: ...  # 返回对象的内存视图
    @property
    def flags(self) -> flagsobj: ...  # 返回对象的标志信息
    @property
    def itemsize(self) -> int: ...  # 返回对象中每个元素的字节数
    @property
    def nbytes(self) -> int: ...  # 返回对象占用的总字节数
    def __bool__(self) -> builtins.bool: ...  # 返回对象的布尔值
    def __bytes__(self) -> bytes: ...  # 返回对象的字节表示
    def __str__(self) -> str: ...  # 返回对象的字符串表示
    def __repr__(self) -> str: ...  # 返回对象的详细字符串表示
    def __copy__(self: _ArraySelf) -> _ArraySelf: ...  # 浅复制对象并返回 `_ArraySelf`
    def __deepcopy__(self: _ArraySelf, memo: None | dict[int, Any], /) -> _ArraySelf: ...
    # TODO: 如何处理 `==` 和 `!=` 的非交换性质？
    # 参考 numpy/numpy#17368
    def __eq__(self, other: Any) -> Any: ...  # 比较对象是否相等
    def __ne__(self, other: Any) -> Any: ...  # 比较对象是否不相等
    def copy(self: _ArraySelf, order: _OrderKACF = ...) -> _ArraySelf: ...  # 复制对象，可指定顺序
    # 定义一个方法 `dump`，接受一个文件路径或字节流作为参数，用于将数据转储到指定文件
    def dump(self, file: str | bytes | os.PathLike[str] | os.PathLike[bytes] | _SupportsWrite[bytes]) -> None: ...

    # 定义一个方法 `dumps`，将数据转换为字节流并返回
    def dumps(self) -> bytes: ...

    # 定义一个方法 `tobytes`，将数组转换为字节流，默认按指定顺序转换
    def tobytes(self, order: _OrderKACF = ...) -> bytes: ...

    # 注意：`tostring()` 已被弃用，因此在此处被排除在外
    # def tostring(self, order=...): ...

    # 定义一个方法 `tofile`，将数据写入指定文件，可以指定分隔符和格式
    def tofile(
        self,
        fid: str | bytes | os.PathLike[str] | os.PathLike[bytes] | _IOProtocol,
        sep: str = ...,
        format: str = ...,
    ) -> None: ...

    # 定义一个方法 `tolist`，将数组转换为列表形式并返回
    def tolist(self) -> Any: ...

    # 定义一个属性 `__array_interface__`，返回数组接口的字典形式
    @property
    def __array_interface__(self) -> dict[str, Any]: ...

    # 定义一个属性 `__array_priority__`，返回数组的优先级（浮点数）
    @property
    def __array_priority__(self) -> float: ...

    # 定义一个属性 `__array_struct__`，返回数组的结构（内置的 PyCapsule）
    @property
    def __array_struct__(self) -> Any: ...  # builtins.PyCapsule

    # 定义一个方法 `__setstate__`，用于设置数组对象的状态
    # 参数 state 包含版本号、形状、数据类型、是否 F 连续以及数据本身
    def __setstate__(self, state: tuple[
        SupportsIndex,  # version
        _ShapeLike,  # Shape
        _DType_co,  # DType
        np.bool,  # F-continuous
        bytes | list[Any],  # Data
    ], /) -> None: ...

    # 当 `keepdims=True` 且数组为 0 维时，返回 `np.bool`
    # `all` 方法的第一重载
    @overload
    def all(
        self,
        axis: None = ...,
        out: None = ...,
        keepdims: L[False] = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> np.bool: ...

    # `all` 方法的第二重载
    @overload
    def all(
        self,
        axis: None | _ShapeLike = ...,
        out: None = ...,
        keepdims: builtins.bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...

    # `all` 方法的第三重载
    @overload
    def all(
        self,
        axis: None | _ShapeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: builtins.bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    # `any` 方法的第一重载
    @overload
    def any(
        self,
        axis: None = ...,
        out: None = ...,
        keepdims: L[False] = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> np.bool: ...

    # `any` 方法的第二重载
    @overload
    def any(
        self,
        axis: None | _ShapeLike = ...,
        out: None = ...,
        keepdims: builtins.bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...

    # `any` 方法的第三重载
    @overload
    def any(
        self,
        axis: None | _ShapeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: builtins.bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    # 返回沿指定轴的最大值的索引，第一重载
    @overload
    def argmax(
        self,
        axis: None = ...,
        out: None = ...,
        *,
        keepdims: L[False] = ...,
    ) -> intp: ...

    # 返回沿指定轴的最大值的索引，第二重载
    @overload
    def argmax(
        self,
        axis: SupportsIndex = ...,
        out: None = ...,
        *,
        keepdims: builtins.bool = ...,
    ) -> Any: ...

    # 返回沿指定轴的最大值的索引，第三重载
    @overload
    def argmax(
        self,
        axis: None | SupportsIndex = ...,
        out: _NdArraySubClass = ...,
        *,
        keepdims: builtins.bool = ...,
    ) -> _NdArraySubClass: ...
    def argmin(
        self,
        axis: None = ...,
        out: None = ...,
        *,
        keepdims: L[False] = ...,
    ) -> intp:
        ...

    @overload
    def argmin(
        self,
        axis: SupportsIndex = ...,
        out: None = ...,
        *,
        keepdims: builtins.bool = ...,
    ) -> Any:
        ...

    @overload
    def argmin(
        self,
        axis: None | SupportsIndex = ...,
        out: _NdArraySubClass = ...,
        *,
        keepdims: builtins.bool = ...,
    ) -> _NdArraySubClass:
        ...

    def argsort(
        self,
        axis: None | SupportsIndex = ...,
        kind: None | _SortKind = ...,
        order: None | str | Sequence[str] = ...,
        *,
        stable: None | bool = ...,
    ) -> NDArray[Any]:
        ...

    @overload
    def choose(
        self,
        choices: ArrayLike,
        out: None = ...,
        mode: _ModeKind = ...,
    ) -> NDArray[Any]:
        ...

    @overload
    def choose(
        self,
        choices: ArrayLike,
        out: _NdArraySubClass = ...,
        mode: _ModeKind = ...,
    ) -> _NdArraySubClass:
        ...

    @overload
    def clip(
        self,
        min: ArrayLike = ...,
        max: None | ArrayLike = ...,
        out: None = ...,
        **kwargs: Any,
    ) -> NDArray[Any]:
        ...

    @overload
    def clip(
        self,
        min: None = ...,
        max: ArrayLike = ...,
        out: None = ...,
        **kwargs: Any,
    ) -> NDArray[Any]:
        ...

    @overload
    def clip(
        self,
        min: ArrayLike = ...,
        max: None | ArrayLike = ...,
        out: _NdArraySubClass = ...,
        **kwargs: Any,
    ) -> _NdArraySubClass:
        ...

    @overload
    def clip(
        self,
        min: None = ...,
        max: ArrayLike = ...,
        out: _NdArraySubClass = ...,
        **kwargs: Any,
    ) -> _NdArraySubClass:
        ...

    @overload
    def compress(
        self,
        a: ArrayLike,
        axis: None | SupportsIndex = ...,
        out: None = ...,
    ) -> NDArray[Any]:
        ...

    @overload
    def compress(
        self,
        a: ArrayLike,
        axis: None | SupportsIndex = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass:
        ...

    def conj(self: _ArraySelf) -> _ArraySelf:
        ...

    def conjugate(self: _ArraySelf) -> _ArraySelf:
        ...

    @overload
    def cumprod(
        self,
        axis: None | SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
    ) -> NDArray[Any]:
        ...

    @overload
    def cumprod(
        self,
        axis: None | SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass:
        ...

    @overload
    def cumsum(
        self,
        axis: None | SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
    ) -> NDArray[Any]:
        ...

    @overload
    def cumsum(
        self,
        axis: None | SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass:
        ...



# Explanation:
每一个函数定义都是对 `self` 对象的方法扩展，用于操作 `NDArray` 类型的数据。
- `argmin`: 返回沿指定轴的最小元素的索引。
- `argsort`: 返回沿指定轴排序后的索引。
- `choose`: 根据索引数组从一组选项中选择元素。
- `clip`: 将数组的值限制在一个范围内。
- `compress`: 根据条件压缩数组。
- `conj` 和 `conjugate`: 分别返回复数数组的共轭。
- `cumprod`: 返回数组元素的累积乘积。
- `cumsum`: 返回数组元素的累积和。
    @overload
    def max(
        self,
        axis: None | _ShapeLike = ...,
        out: None = ...,
        keepdims: builtins.bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def max(
        self,
        axis: None | _ShapeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: builtins.bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...
    
    @overload
    def mean(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        keepdims: builtins.bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def mean(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: builtins.bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...
    
    @overload
    def min(
        self,
        axis: None | _ShapeLike = ...,
        out: None = ...,
        keepdims: builtins.bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def min(
        self,
        axis: None | _ShapeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: builtins.bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...
    
    @overload
    def prod(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        keepdims: builtins.bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def prod(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: builtins.bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...
    
    @overload
    def round(
        self: _ArraySelf,
        decimals: SupportsIndex = ...,
        out: None = ...,
    ) -> _ArraySelf: ...
    @overload
    def round(
        self,
        decimals: SupportsIndex = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...
    
    @overload
    def std(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        ddof: float = ...,
        keepdims: builtins.bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def std(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        ddof: float = ...,
        keepdims: builtins.bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...
    
    
    
    # 按照特定的重载格式定义了一系列方法：max, mean, min, prod, round, std
    # 每个方法具有多个重载，根据参数的不同类型和个数来决定调用哪个版本
    # 每个方法接受不同的参数组合，并返回相应类型的值，可以处理数组的最大值、平均值、最小值、乘积、四舍五入、标准差等运算
    # 定义 sum 方法的函数签名，接受多个参数并返回任意类型的结果
    def sum(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        keepdims: builtins.bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...

    # 使用 @overload 装饰器定义 sum 方法的函数签名重载，返回特定的 NdArraySubClass 类型结果
    @overload
    def sum(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: builtins.bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    # 定义 var 方法的函数签名，接受多个参数并返回任意类型的结果
    @overload
    def var(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        ddof: float = ...,
        keepdims: builtins.bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...

    # 使用 @overload 装饰器定义 var 方法的函数签名重载，返回特定的 NdArraySubClass 类型结果
    @overload
    def var(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        ddof: float = ...,
        keepdims: builtins.bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...
# 定义一个泛型类型变量 `_DType`，其上界为 `dtype[Any]`
_DType = TypeVar("_DType", bound=dtype[Any])

# 定义一个协变型泛型类型变量 `_DType_co`，其上界为 `dtype[Any]`
_DType_co = TypeVar("_DType_co", covariant=True, bound=dtype[Any])

# 定义一个泛型类型变量 `_FlexDType`，其上界为 `dtype[flexible]`
_FlexDType = TypeVar("_FlexDType", bound=dtype[flexible])

# 定义一个泛型类型变量 `_ShapeType`，其上界为 `Any`
_ShapeType = TypeVar("_ShapeType", bound=Any)

# 定义另一个泛型类型变量 `_ShapeType2`，其上界为 `Any`
_ShapeType2 = TypeVar("_ShapeType2", bound=Any)

# 定义一个泛型类型变量 `_NumberType`，其上界为 `number[Any]`
_NumberType = TypeVar("_NumberType", bound=number[Any])

# 根据 Python 版本选择合适的 `Buffer` 类型
if sys.version_info >= (3, 12):
    from collections.abc import Buffer as _SupportsBuffer
else:
    _SupportsBuffer = (
        bytes
        | bytearray
        | memoryview
        | _array.array[Any]
        | mmap.mmap
        | NDArray[Any]
        | generic
    )

# 定义一个泛型类型变量 `_T`
_T = TypeVar("_T")

# 定义一个协变型泛型类型变量 `_T_co`
_T_co = TypeVar("_T_co", covariant=True)

# 定义一个逆变型泛型类型变量 `_T_contra`
_T_contra = TypeVar("_T_contra", contravariant=True)

# 定义一个元组类型 `_2Tuple`，包含两个相同类型的元素
_2Tuple = tuple[_T, _T]

# 定义一个枚举类型 `_CastingKind`，包含几种转换方式
_CastingKind = L["no", "equiv", "safe", "same_kind", "unsafe"]

# 定义几种特定数组类型的别名
_ArrayUInt_co = NDArray[np.bool | unsignedinteger[Any]]
_ArrayInt_co = NDArray[np.bool | integer[Any]]
_ArrayFloat_co = NDArray[np.bool | integer[Any] | floating[Any]]
_ArrayComplex_co = NDArray[np.bool | integer[Any] | floating[Any] | complexfloating[Any, Any]]
_ArrayNumber_co = NDArray[np.bool | number[Any]]
_ArrayTD64_co = NDArray[np.bool | integer[Any] | timedelta64]

# 引入 `dtype` 的别名以避免命名冲突
_dtype = dtype

# 对于 `builtins.PyCapsule`，由于缺乏注解，暂时使用 `Any`
_PyCapsule = Any

# 定义 `_SupportsItem` 协议，要求包含 `item` 方法
class _SupportsItem(Protocol[_T_co]):
    def item(self, args: Any, /) -> _T_co: ...

# 定义 `_SupportsReal` 协议，要求包含 `real` 属性
class _SupportsReal(Protocol[_T_co]):
    @property
    def real(self) -> _T_co: ...

# 定义 `_SupportsImag` 协议，要求包含 `imag` 属性
class _SupportsImag(Protocol[_T_co]):
    @property
    def imag(self) -> _T_co: ...

# 定义 `ndarray` 类，继承自 `_ArrayOrScalarCommon`，具有两个泛型类型参数 `_ShapeType` 和 `_DType_co`
class ndarray(_ArrayOrScalarCommon, Generic[_ShapeType, _DType_co]):
    __hash__: ClassVar[None]

    # `base` 属性，返回空或者 `NDArray[Any]`
    @property
    def base(self) -> None | NDArray[Any]: ...

    # `ndim` 属性，返回数组的维度数
    @property
    def ndim(self) -> int: ...

    # `size` 属性，返回数组的大小
    @property
    def size(self) -> int: ...

    # `real` 属性，设置实部数据类型为 `_ScalarType` 的 `ndarray`
    @property
    def real(
        self: ndarray[_ShapeType, dtype[_SupportsReal[_ScalarType]]],  # type: ignore[type-var]
    ) -> ndarray[_ShapeType, _dtype[_ScalarType]]: ...

    # `real` 属性的设置器方法，设置实部数据类型为 `ArrayLike` 的 `ndarray`
    @real.setter
    def real(self, value: ArrayLike) -> None: ...

    # `imag` 属性，设置虚部数据类型为 `_ScalarType` 的 `ndarray`
    @property
    def imag(
        self: ndarray[_ShapeType, dtype[_SupportsImag[_ScalarType]]],  # type: ignore[type-var]
    ) -> ndarray[_ShapeType, _dtype[_ScalarType]]: ...

    # `imag` 属性的设置器方法，设置虚部数据类型为 `ArrayLike` 的 `ndarray`
    @imag.setter
    def imag(self, value: ArrayLike) -> None: ...

    # `__new__` 方法，创建新的 `ndarray` 对象
    def __new__(
        cls: type[_ArraySelf],
        shape: _ShapeLike,
        dtype: DTypeLike = ...,
        buffer: None | _SupportsBuffer = ...,
        offset: SupportsIndex = ...,
        strides: None | _ShapeLike = ...,
        order: _OrderKACF = ...,
    ) -> _ArraySelf: ...

    # 如果 Python 版本 >= 3.12，则定义 `__buffer__` 方法，返回 `memoryview`
    if sys.version_info >= (3, 12):
        def __buffer__(self, flags: int, /) -> memoryview: ...

    # `__class_getitem__` 方法，返回泛型别名 `GenericAlias`
    def __class_getitem__(self, item: Any) -> GenericAlias: ...

    # 以下省略 `overload` 的处理部分
    # 将对象转换为 ndarray 类型，支持多种参数形式和数据类型
    def __array__(
        self, dtype: None = ..., /, *, copy: None | bool = ...
    ) -> ndarray[Any, _DType_co]: ...

    # 用于支持 ndarray 的通用函数（ufunc）操作
    def __array_ufunc__(
        self,
        ufunc: ufunc,
        method: L["__call__", "reduce", "reduceat", "accumulate", "outer", "at"],
        *inputs: Any,
        **kwargs: Any,
    ) -> Any: ...

    # 用于支持 ndarray 的通用函数（array function）操作
    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Iterable[type],
        args: Iterable[Any],
        kwargs: Mapping[str, Any],
    ) -> Any: ...

    # 注意：在实践中，`obj` 可接受任何对象，但由于 `__array_finalize__`
    # 是一个伪抽象方法，类型已经被缩小，以便为子类提供更多的灵活性
    def __array_finalize__(self, obj: None | NDArray[Any], /) -> None: ...

    # 根据情况包装 ndarray 对象，支持不同的上下文和返回值类型
    def __array_wrap__(
        self,
        array: ndarray[_ShapeType2, _DType],
        context: None | tuple[ufunc, tuple[Any, ...], int] = ...,
        return_scalar: builtins.bool = ...,
        /,
    ) -> ndarray[_ShapeType2, _DType]: ...

    # 支持多种索引方式的 ndarray 的索引操作重载
    @overload
    def __getitem__(self, key: (
        NDArray[integer[Any]]
        | NDArray[np.bool]
        | tuple[NDArray[integer[Any]] | NDArray[np.bool], ...]
    )) -> ndarray[Any, _DType_co]: ...
    @overload
    def __getitem__(self, key: SupportsIndex | tuple[SupportsIndex, ...]) -> Any: ...
    @overload
    def __getitem__(self, key: (
        None
        | slice
        | ellipsis
        | SupportsIndex
        | _ArrayLikeInt_co
        | tuple[None | slice | ellipsis | _ArrayLikeInt_co | SupportsIndex, ...]
    )) -> ndarray[Any, _DType_co]: ...
    @overload
    def __getitem__(self: NDArray[void], key: str) -> NDArray[Any]: ...
    @overload
    def __getitem__(self: NDArray[void], key: list[str]) -> ndarray[_ShapeType, _dtype[void]]: ...

    # 返回一个与底层 `generic` 类型相同的项
    @property
    def ctypes(self) -> _ctypes[int]: ...

    # 返回 ndarray 对象的形状信息
    @property
    def shape(self) -> _Shape: ...

    # 设置 ndarray 对象的形状信息
    @shape.setter
    def shape(self, value: _ShapeLike) -> None: ...

    # 返回 ndarray 对象的步幅信息
    @property
    def strides(self) -> _Shape: ...

    # 设置 ndarray 对象的步幅信息
    @strides.setter
    def strides(self, value: _ShapeLike) -> None: ...

    # 在 ndarray 对象上执行字节顺序转换操作
    def byteswap(self: _ArraySelf, inplace: builtins.bool = ...) -> _ArraySelf: ...

    # 使用指定值填充整个 ndarray 对象
    def fill(self, value: Any) -> None: ...

    # 返回一个扁平的 ndarray 对象迭代器
    @property
    def flat(self: _NdArraySubClass) -> flatiter[_NdArraySubClass]: ...

    # 以底层 `generic` 类型相同的类型返回项
    @overload
    def item(
        self: ndarray[Any, _dtype[_SupportsItem[_T]]],  # type: ignore[type-var]
        *args: SupportsIndex,
    ) -> _T: ...
    @overload
    def item(
        self: ndarray[Any, _dtype[_SupportsItem[_T]]],  # type: ignore[type-var]
        args: tuple[SupportsIndex, ...],
        /,
    ) -> _T: ...

    # 调整 ndarray 对象的大小，支持参考检查选项
    @overload
    def resize(self, new_shape: _ShapeLike, /, *, refcheck: builtins.bool = ...) -> None: ...
    # 定义一个方法 `resize`，用于改变数组的形状，支持多个新形状参数和一个可选的参考检查标志
    def resize(self, *new_shape: SupportsIndex, refcheck: builtins.bool = ...) -> None: ...

    # 定义一个方法 `setflags`，用于设置数组的标志，包括写入标志、对齐标志和用户指定的标志
    def setflags(
        self, write: builtins.bool = ..., align: builtins.bool = ..., uic: builtins.bool = ...
    ) -> None: ...

    # 定义一个方法 `squeeze`，用于压缩数组，去除维度为1的轴，支持指定压缩的轴
    def squeeze(
        self,
        axis: None | SupportsIndex | tuple[SupportsIndex, ...] = ...,
    ) -> ndarray[Any, _DType_co]: ...

    # 定义一个方法 `swapaxes`，用于交换数组的两个轴的位置
    def swapaxes(
        self,
        axis1: SupportsIndex,
        axis2: SupportsIndex,
    ) -> ndarray[Any, _DType_co]: ...

    # 定义一个方法 `transpose`，支持不同的重载方式，用于对数组进行转置操作
    @overload
    def transpose(self: _ArraySelf, axes: None | _ShapeLike, /) -> _ArraySelf: ...
    @overload
    def transpose(self: _ArraySelf, *axes: SupportsIndex) -> _ArraySelf: ...

    # 定义一个方法 `argpartition`，用于对数组进行部分排序操作，并返回索引数组
    def argpartition(
        self,
        kth: _ArrayLikeInt_co,
        axis: None | SupportsIndex = ...,
        kind: _PartitionKind = ...,
        order: None | str | Sequence[str] = ...,
    ) -> NDArray[intp]: ...

    # 定义一个方法 `diagonal`，用于获取数组的对角线元素
    def diagonal(
        self,
        offset: SupportsIndex = ...,
        axis1: SupportsIndex = ...,
        axis2: SupportsIndex = ...,
    ) -> ndarray[Any, _DType_co]: ...

    # 定义一个方法 `dot`，支持不同的重载方式，用于计算数组的点积
    # 1D + 1D 返回标量；其他情况下至少有一个非0维数组返回一个数组
    @overload
    def dot(self, b: _ScalarLike_co, out: None = ...) -> NDArray[Any]: ...
    @overload
    def dot(self, b: ArrayLike, out: None = ...) -> Any: ...  # type: ignore[misc]
    @overload
    def dot(self, b: ArrayLike, out: _NdArraySubClass) -> _NdArraySubClass: ...

    # 定义一个方法 `nonzero`，用于获取数组中非零元素的索引，但对于0维数组和泛型数组已被弃用
    def nonzero(self) -> tuple[NDArray[intp], ...]: ...

    # 定义一个方法 `partition`，用于对数组进行分区排序操作
    def partition(
        self,
        kth: _ArrayLikeInt_co,
        axis: SupportsIndex = ...,
        kind: _PartitionKind = ...,
        order: None | str | Sequence[str] = ...,
    ) -> None: ...

    # 定义一个方法 `put`，用于在数组中放置元素，但对于泛型数组来说是无效的，因为泛型数组是不可变的
    def put(
        self,
        ind: _ArrayLikeInt_co,
        v: ArrayLike,
        mode: _ModeKind = ...,
    ) -> None: ...

    # 定义一个方法 `searchsorted`，支持不同的重载方式，用于在已排序数组中搜索指定值的插入点
    @overload
    def searchsorted(  # type: ignore[misc]
        self,  # >= 1D array
        v: _ScalarLike_co,  # 0D array-like
        side: _SortSide = ...,
        sorter: None | _ArrayLikeInt_co = ...,
    ) -> intp: ...
    @overload
    def searchsorted(
        self,  # >= 1D array
        v: ArrayLike,
        side: _SortSide = ...,
        sorter: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[intp]: ...

    # 定义一个方法 `setfield`，用于设置数组的字段值，根据指定的数据类型和偏移量
    def setfield(
        self,
        val: ArrayLike,
        dtype: DTypeLike,
        offset: SupportsIndex = ...,
    ) -> None: ...

    # 定义一个方法 `sort`，用于对数组进行排序操作
    def sort(
        self,
        axis: SupportsIndex = ...,
        kind: None | _SortKind = ...,
        order: None | str | Sequence[str] = ...,
        *,
        stable: None | bool = ...,
    ) -> None: ...
    def trace(
        self,  # >= 2D array
        offset: SupportsIndex = ...,
        axis1: SupportsIndex = ...,
        axis2: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
    ) -> Any: ...


    @overload
    def trace(
        self,  # >= 2D array
        offset: SupportsIndex = ...,
        axis1: SupportsIndex = ...,
        axis2: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...


    @overload
    def take(  # type: ignore[misc]
        self: NDArray[_ScalarType],
        indices: _IntLike_co,
        axis: None | SupportsIndex = ...,
        out: None = ...,
        mode: _ModeKind = ...,
    ) -> _ScalarType: ...


    @overload
    def take(  # type: ignore[misc]
        self,
        indices: _ArrayLikeInt_co,
        axis: None | SupportsIndex = ...,
        out: None = ...,
        mode: _ModeKind = ...,
    ) -> ndarray[Any, _DType_co]: ...


    @overload
    def take(
        self,
        indices: _ArrayLikeInt_co,
        axis: None | SupportsIndex = ...,
        out: _NdArraySubClass = ...,
        mode: _ModeKind = ...,
    ) -> _NdArraySubClass: ...


    def repeat(
        self,
        repeats: _ArrayLikeInt_co,
        axis: None | SupportsIndex = ...,
    ) -> ndarray[Any, _DType_co]: ...


    def flatten(
        self,
        order: _OrderKACF = ...,
    ) -> ndarray[Any, _DType_co]: ...


    def ravel(
        self,
        order: _OrderKACF = ...,
    ) -> ndarray[Any, _DType_co]: ...


    @overload
    def reshape(
        self,
        shape: _ShapeLike,
        /,
        *,
        order: _OrderACF = ...,
        copy: None | bool = ...,
    ) -> ndarray[Any, _DType_co]: ...


    @overload
    def reshape(
        self,
        *shape: SupportsIndex,
        order: _OrderACF = ...,
        copy: None | bool = ...,
    ) -> ndarray[Any, _DType_co]: ...


    @overload
    def astype(
        self,
        dtype: _DTypeLike[_ScalarType],
        order: _OrderKACF = ...,
        casting: _CastingKind = ...,
        subok: builtins.bool = ...,
        copy: builtins.bool | _CopyMode = ...,
    ) -> NDArray[_ScalarType]: ...


    @overload
    def astype(
        self,
        dtype: DTypeLike,
        order: _OrderKACF = ...,
        casting: _CastingKind = ...,
        subok: builtins.bool = ...,
        copy: builtins.bool | _CopyMode = ...,
    ) -> NDArray[Any]: ...


    @overload
    def view(self: _ArraySelf) -> _ArraySelf: ...


    @overload
    def view(self, type: type[_NdArraySubClass]) -> _NdArraySubClass: ...


    @overload
    def view(self, dtype: _DTypeLike[_ScalarType]) -> NDArray[_ScalarType]: ...


    @overload
    def view(self, dtype: DTypeLike) -> NDArray[Any]: ...


    @overload
    def view(
        self,
        dtype: DTypeLike,
        type: type[_NdArraySubClass],
    ) -> _NdArraySubClass: ...


    @overload
    # 定义方法 getfield，用于获取数组中特定字段的数据
    def getfield(
        self,
        dtype: _DTypeLike[_ScalarType],  # 指定字段数据类型
        offset: SupportsIndex = ...  # 字段的偏移量，默认值为省略号
    ) -> NDArray[_ScalarType]: ...  # 返回值为指定数据类型的 NumPy 数组

    # 重载方法 getfield，支持更泛化的数据类型
    @overload
    def getfield(
        self,
        dtype: DTypeLike,  # 指定字段数据类型
        offset: SupportsIndex = ...  # 字段的偏移量，默认值为省略号
    ) -> NDArray[Any]: ...

    # 将整数类型的数组转换为整数类型
    def __int__(
        self: NDArray[SupportsInt],  # 当前对象是支持整数的 NumPy 数组
    ) -> int: ...

    # 将浮点数类型的数组转换为浮点数类型
    def __float__(
        self: NDArray[SupportsFloat],  # 当前对象是支持浮点数的 NumPy 数组
    ) -> float: ...

    # 将复数类型的数组转换为复数类型
    def __complex__(
        self: NDArray[SupportsComplex],  # 当前对象是支持复数的 NumPy 数组
    ) -> complex: ...

    # 将数组类型的对象转换为整数类型
    def __index__(
        self: NDArray[SupportsIndex],  # 当前对象是支持索引的 NumPy 数组
    ) -> int: ...

    # 返回数组的长度
    def __len__(self) -> int: ...

    # 设置数组中指定键的值
    def __setitem__(self, key, value): ...

    # 迭代数组对象
    def __iter__(self) -> Any: ...

    # 检查数组对象是否包含指定的键
    def __contains__(self, key) -> builtins.bool: ...

    # 最后一个重载用于捕获递归对象，其嵌套层次过深
    # 第一个重载用于捕获字节串（因为它们是 `Sequence[int]` 的子类型）和字符串。
    # 由于字符串是一个递归的字符串序列，否则它将通过最后一个重载。
    @overload
    def __lt__(self: _ArrayNumber_co, other: _ArrayLikeNumber_co) -> NDArray[np.bool]: ...
    @overload
    def __lt__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[np.bool]: ...
    @overload
    def __lt__(self: NDArray[datetime64], other: _ArrayLikeDT64_co) -> NDArray[np.bool]: ...
    @overload
    def __lt__(self: NDArray[object_], other: Any) -> NDArray[np.bool]: ...
    @overload
    def __lt__(self: NDArray[Any], other: _ArrayLikeObject_co) -> NDArray[np.bool]: ...

    @overload
    def __le__(self: _ArrayNumber_co, other: _ArrayLikeNumber_co) -> NDArray[np.bool]: ...
    @overload
    def __le__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[np.bool]: ...
    @overload
    def __le__(self: NDArray[datetime64], other: _ArrayLikeDT64_co) -> NDArray[np.bool]: ...
    @overload
    def __le__(self: NDArray[object_], other: Any) -> NDArray[np.bool]: ...
    @overload
    def __le__(self: NDArray[Any], other: _ArrayLikeObject_co) -> NDArray[np.bool]: ...

    @overload
    def __gt__(self: _ArrayNumber_co, other: _ArrayLikeNumber_co) -> NDArray[np.bool]: ...
    @overload
    def __gt__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[np.bool]: ...
    @overload
    def __gt__(self: NDArray[datetime64], other: _ArrayLikeDT64_co) -> NDArray[np.bool]: ...
    @overload
    def __gt__(self: NDArray[object_], other: Any) -> NDArray[np.bool]: ...
    @overload
    def __gt__(self: NDArray[Any], other: _ArrayLikeObject_co) -> NDArray[np.bool]: ...

    @overload
    def __ge__(self: _ArrayNumber_co, other: _ArrayLikeNumber_co) -> NDArray[np.bool]: ...
    @overload
    def __ge__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[np.bool]: ...
    # 定义特殊方法 __ge__，用于比较操作，支持 datetime64 类型的数组
    def __ge__(self: NDArray[datetime64], other: _ArrayLikeDT64_co) -> NDArray[np.bool]: ...

    # 重载 __ge__ 方法，支持 object_ 类型的数组和任意类型的比较操作
    @overload
    def __ge__(self: NDArray[object_], other: Any) -> NDArray[np.bool]: ...

    # 重载 __ge__ 方法，支持任意类型的数组和 _ArrayLikeObject_co 类型的比较操作
    @overload
    def __ge__(self: NDArray[Any], other: _ArrayLikeObject_co) -> NDArray[np.bool]: ...


    # Unary ops (一元操作)

    # 重载 __abs__ 方法，返回任意类型的数组
    @overload
    def __abs__(self: NDArray[_UnknownType]) -> NDArray[Any]: ...

    # 重载 __abs__ 方法，返回布尔类型的数组
    @overload
    def __abs__(self: NDArray[np.bool]) -> NDArray[np.bool]: ...

    # 重载 __abs__ 方法，返回复数浮点类型数组的绝对值
    @overload
    def __abs__(self: NDArray[complexfloating[_NBit1, _NBit1]]) -> NDArray[floating[_NBit1]]: ...

    # 重载 __abs__ 方法，返回数字类型数组的绝对值
    @overload
    def __abs__(self: NDArray[_NumberType]) -> NDArray[_NumberType]: ...

    # 重载 __abs__ 方法，返回时间间隔类型的数组
    @overload
    def __abs__(self: NDArray[timedelta64]) -> NDArray[timedelta64]: ...

    # 重载 __abs__ 方法，返回任意对象类型的结果
    @overload
    def __abs__(self: NDArray[object_]) -> Any: ...


    # Unary ops (一元操作)

    # 重载 __invert__ 方法，返回任意类型的数组
    @overload
    def __invert__(self: NDArray[_UnknownType]) -> NDArray[Any]: ...

    # 重载 __invert__ 方法，返回布尔类型的数组
    @overload
    def __invert__(self: NDArray[np.bool]) -> NDArray[np.bool]: ...

    # 重载 __invert__ 方法，返回整数类型的数组
    @overload
    def __invert__(self: NDArray[_IntType]) -> NDArray[_IntType]: ...

    # 重载 __invert__ 方法，返回任意对象类型的结果
    @overload
    def __invert__(self: NDArray[object_]) -> Any: ...


    # Unary ops (一元操作)

    # 重载 __pos__ 方法，返回数字类型的数组
    @overload
    def __pos__(self: NDArray[_NumberType]) -> NDArray[_NumberType]: ...

    # 重载 __pos__ 方法，返回时间间隔类型的数组
    @overload
    def __pos__(self: NDArray[timedelta64]) -> NDArray[timedelta64]: ...

    # 重载 __pos__ 方法，返回任意对象类型的结果
    @overload
    def __pos__(self: NDArray[object_]) -> Any: ...


    # Unary ops (一元操作)

    # 重载 __neg__ 方法，返回数字类型的数组
    @overload
    def __neg__(self: NDArray[_NumberType]) -> NDArray[_NumberType]: ...

    # 重载 __neg__ 方法，返回时间间隔类型的数组
    @overload
    def __neg__(self: NDArray[timedelta64]) -> NDArray[timedelta64]: ...

    # 重载 __neg__ 方法，返回任意对象类型的结果
    @overload
    def __neg__(self: NDArray[object_]) -> Any: ...


    # Binary ops (二元操作)

    # 重载 __matmul__ 方法，支持未知类型数组与 _ArrayLikeUnknown 类型的矩阵乘法操作
    @overload
    def __matmul__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...

    # 重载 __matmul__ 方法，支持布尔类型数组与 _ArrayLikeBool_co 类型的矩阵乘法操作
    @overload
    def __matmul__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[np.bool]: ...  # type: ignore[misc]

    # 重载 __matmul__ 方法，支持无符号整数类型数组与 _ArrayLikeUInt_co 类型的矩阵乘法操作
    @overload
    def __matmul__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]

    # 重载 __matmul__ 方法，支持有符号整数类型数组与 _ArrayLikeInt_co 类型的矩阵乘法操作
    @overload
    def __matmul__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]

    # 重载 __matmul__ 方法，支持浮点数类型数组与 _ArrayLikeFloat_co 类型的矩阵乘法操作
    @overload
    def __matmul__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]

    # 重载 __matmul__ 方法，支持复数浮点数类型数组与 _ArrayLikeComplex_co 类型的矩阵乘法操作
    @overload
    def __matmul__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...

    # 重载 __matmul__ 方法，支持数字类型数组与 _ArrayLikeNumber_co 类型的矩阵乘法操作
    @overload
    def __matmul__(self: NDArray[number[Any]], other: _ArrayLikeNumber_co) -> NDArray[number[Any]]: ...

    # 重载 __matmul__ 方法，支持任意对象类型数组与任意对象的矩阵乘法操作
    @overload
    def __matmul__(self: NDArray[object_], other: Any) -> Any: ...

    # 重载 __matmul__ 方法，支持任意类型数组与 _ArrayLikeObject_co 类型的矩阵乘法操作
    @overload
    def __matmul__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...


    # Binary ops (二元操作)

    # 重载 __rmatmul__ 方法，支持未知类型数组与 _ArrayLikeUnknown 类型的右侧矩阵乘法操作
    @overload
    def __rmatmul__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...

    # 重载 __rmatmul__ 方法，支持布尔类型数组与 _ArrayLikeBool_co 类型的右侧矩阵乘法操作
    @overload
    def __rmatmul__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[np.bool]: ...  # type: ignore[misc]
    def __rmatmul__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmatmul__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmatmul__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmatmul__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...
    @overload
    def __rmatmul__(self: NDArray[number[Any]], other: _ArrayLikeNumber_co) -> NDArray[number[Any]]: ...
    @overload
    def __rmatmul__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rmatmul__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...


    # Type hinting and overload for the right-matrix multiplication operator (__rmatmul__)
    def __rmatmul__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    # Overload for unsigned integer arrays (__rmatmul__)
    @overload
    def __rmatmul__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    # Overload for signed integer arrays (__rmatmul__)
    @overload
    def __rmatmul__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    # Overload for floating point arrays (__rmatmul__)
    @overload
    def __rmatmul__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...
    # Overload for complex floating point arrays (__rmatmul__)
    @overload
    def __rmatmul__(self: NDArray[number[Any]], other: _ArrayLikeNumber_co) -> NDArray[number[Any]]: ...
    # Overload for number arrays (__rmatmul__)
    @overload
    def __rmatmul__(self: NDArray[object_], other: Any) -> Any: ...
    # Overload for generic object arrays (__rmatmul__)
    @overload
    def __rmatmul__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    # Overload for object arrays (__rmatmul__)


    @overload
    def __mod__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    @overload
    def __mod__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    @overload
    def __mod__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __mod__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __mod__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __mod__(self: _ArrayTD64_co, other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> NDArray[timedelta64]: ...
    @overload
    def __mod__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __mod__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...


    # Type hinting and overload for the modulo operator (__mod__)
    @overload
    def __mod__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    # Overload for unknown type arrays (__mod__)
    @overload
    def __mod__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    # Overload for boolean arrays (__mod__)
    @overload
    def __mod__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    # Overload for unsigned integer arrays (__mod__)
    @overload
    def __mod__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    # Overload for signed integer arrays (__mod__)
    @overload
    def __mod__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    # Overload for floating point arrays (__mod__)
    @overload
    def __mod__(self: _ArrayTD64_co, other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> NDArray[timedelta64]: ...
    # Overload for timedelta64 arrays (__mod__)
    @overload
    def __mod__(self: NDArray[object_], other: Any) -> Any: ...
    # Overload for object arrays (__mod__)
    @overload
    def __mod__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    # Overload for generic object arrays (__mod__)


    @overload
    def __rmod__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    @overload
    def __rmod__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    @overload
    def __rmod__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmod__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmod__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmod__(self: _ArrayTD64_co, other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> NDArray[timedelta64]: ...
    @overload
    def __rmod__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rmod__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...


    # Type hinting and overload for the right modulo operator (__rmod__)
    @overload
    def __rmod__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    # Overload for unknown type arrays (__rmod__)
    @overload
    def __rmod__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    # Overload for boolean arrays (__rmod__)
    @overload
    def __rmod__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    # Overload for unsigned integer arrays (__rmod__)
    @overload
    def __rmod__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    # Overload for signed integer arrays (__rmod__)
    @overload
    def __rmod__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    # Overload for floating point arrays (__rmod__)
    @overload
    def __rmod__(self: _ArrayTD64_co, other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> NDArray[timedelta64]: ...
    # Overload for timedelta64 arrays (__rmod__)
    def __divmod__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> _2Tuple[NDArray[int8]]: ...  # type: ignore[misc]
    @overload
    def __divmod__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _2Tuple[NDArray[unsignedinteger[Any]]]: ...  # type: ignore[misc]
    @overload
    def __divmod__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _2Tuple[NDArray[signedinteger[Any]]]: ...  # type: ignore[misc]
    @overload
    def __divmod__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _2Tuple[NDArray[floating[Any]]]: ...  # type: ignore[misc]
    @overload
    def __divmod__(self: _ArrayTD64_co, other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> tuple[NDArray[int64], NDArray[timedelta64]]: ...


    # 定义了多个函数重载，用于执行除法和取余操作，对不同类型的数组进行操作
    def __divmod__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> _2Tuple[NDArray[int8]]: ...  # type: ignore[misc]
    # 第一个重载，处理布尔类型数组与其它布尔类型数组的除法和取余操作

    @overload
    def __divmod__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _2Tuple[NDArray[unsignedinteger[Any]]]: ...  # type: ignore[misc]
    # 第二个重载，处理无符号整数类型数组与其它无符号整数类型数组的除法和取余操作

    @overload
    def __divmod__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _2Tuple[NDArray[signedinteger[Any]]]: ...  # type: ignore[misc]
    # 第三个重载，处理有符号整数类型数组与其它有符号整数类型数组的除法和取余操作

    @overload
    def __divmod__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _2Tuple[NDArray[floating[Any]]]: ...  # type: ignore[misc]
    # 第四个重载，处理浮点数类型数组与其它浮点数类型数组的除法和取余操作

    @overload
    def __divmod__(self: _ArrayTD64_co, other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> tuple[NDArray[int64], NDArray[timedelta64]]: ...
    # 第五个重载，处理时间增量类型数组与支持时间增量数组或嵌套时间增量数组的除法和取余操作


    @overload
    def __rdivmod__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> _2Tuple[NDArray[Any]]: ...
    @overload
    def __rdivmod__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> _2Tuple[NDArray[int8]]: ...  # type: ignore[misc]
    @overload
    def __rdivmod__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _2Tuple[NDArray[unsignedinteger[Any]]]: ...  # type: ignore[misc]
    @overload
    def __rdivmod__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _2Tuple[NDArray[signedinteger[Any]]]: ...  # type: ignore[misc]
    @overload
    def __rdivmod__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _2Tuple[NDArray[floating[Any]]]: ...  # type: ignore[misc]
    @overload
    def __rdivmod__(self: _ArrayTD64_co, other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> tuple[NDArray[int64], NDArray[timedelta64]]: ...


    # 定义了多个函数重载，用于执行右侧操作数在前的除法和取余操作，对不同类型的数组进行操作
    @overload
    def __rdivmod__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> _2Tuple[NDArray[Any]]: ...
    # 第一个重载，处理未知类型数组与其它未知类型数组的右侧操作数在前的除法和取余操作

    @overload
    def __rdivmod__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> _2Tuple[NDArray[int8]]: ...  # type: ignore[misc]
    # 第二个重载，处理布尔类型数组与其它布尔类型数组的右侧操作数在前的除法和取余操作

    @overload
    def __rdivmod__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _2Tuple[NDArray[unsignedinteger[Any]]]: ...  # type: ignore[misc]
    # 第三个重载，处理无符号整数类型数组与其它无符号整数类型数组的右侧操作数在前的除法和取余操作

    @overload
    def __rdivmod__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _2Tuple[NDArray[signedinteger[Any]]]: ...  # type: ignore[misc]
    # 第四个重载，处理有符号整数类型数组与其它有符号整数类型数组的右侧操作数在前的除法和取余操作

    @overload
    def __rdivmod__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _2Tuple[NDArray[floating[Any]]]: ...  # type: ignore[misc]
    # 第五个重载，处理浮点数类型数组与其它浮点数类型数组的右侧操作数在前的除法和取余操作

    @overload
    def __rdivmod__(self: _ArrayTD64_co, other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> tuple[NDArray[int64], NDArray[timedelta64]]: ...
    # 第六个重载，处理时间增量类型数组与支持时间增量数组或嵌套时间增量数组的右侧操作数在前的除法和取余操作


    @overload
    def __add__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    @overload
    def __add__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[np.bool]: ...  # type: ignore[misc]
    @overload
    def __add__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __add__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __add__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __add__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __add__(self: NDArray[number[Any]], other: _ArrayLikeNumber_co) -> NDArray[number[Any]]: ...
    @overload
    def __add__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...  # type: ignore[misc]
    @overload
    def __add__(self: _ArrayTD64_co, other: _ArrayLikeDT64_co) -> NDArray[datetime64]: ...
    @overload
    def __add__(self: NDArray[datetime64], other: _ArrayLikeTD64_co) -> NDArray[datetime64]: ...
    @overload
    def __add__(self: NDArray[object_], other: Any) -> Any: ...


    # 定义了多个函数重载，用于执行加法操作，对不
    @overload
    def __add__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    # 定义了一个方法重载：用于NDArray和_ArrayLikeObject_co类型的加法操作，返回类型为Any

    @overload
    def __radd__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    # 定义了一个方法重载：用于NDArray[_UnknownType]和_ArrayLikeUnknown类型的右加法操作，返回类型为NDArray[Any]

    @overload
    def __radd__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[np.bool]: ...
    # 定义了一个方法重载：用于NDArray[np.bool]和_ArrayLikeBool_co类型的右加法操作，返回类型为NDArray[np.bool]
    # 忽略类型检查(misc)

    @overload
    def __radd__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...
    # 定义了一个方法重载：用于_ArrayUInt_co和_ArrayLikeUInt_co类型的右加法操作，返回类型为NDArray[unsignedinteger[Any]]
    # 忽略类型检查(misc)

    @overload
    def __radd__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    # 定义了一个方法重载：用于_ArrayInt_co和_ArrayLikeInt_co类型的右加法操作，返回类型为NDArray[signedinteger[Any]]
    # 忽略类型检查(misc)

    @overload
    def __radd__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...
    # 定义了一个方法重载：用于_ArrayFloat_co和_ArrayLikeFloat_co类型的右加法操作，返回类型为NDArray[floating[Any]]
    # 忽略类型检查(misc)

    @overload
    def __radd__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...
    # 定义了一个方法重载：用于_ArrayComplex_co和_ArrayLikeComplex_co类型的右加法操作，返回类型为NDArray[complexfloating[Any, Any]]
    # 忽略类型检查(misc)

    @overload
    def __radd__(self: NDArray[number[Any]], other: _ArrayLikeNumber_co) -> NDArray[number[Any]]: ...
    # 定义了一个方法重载：用于NDArray[number[Any]]和_ArrayLikeNumber_co类型的右加法操作，返回类型为NDArray[number[Any]]

    @overload
    def __radd__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...
    # 定义了一个方法重载：用于_ArrayTD64_co和_ArrayLikeTD64_co类型的右加法操作，返回类型为NDArray[timedelta64]
    # 忽略类型检查(misc)

    @overload
    def __radd__(self: _ArrayTD64_co, other: _ArrayLikeDT64_co) -> NDArray[datetime64]: ...
    # 定义了一个方法重载：用于_ArrayTD64_co和_ArrayLikeDT64_co类型的右加法操作，返回类型为NDArray[datetime64]

    @overload
    def __radd__(self: NDArray[datetime64], other: _ArrayLikeTD64_co) -> NDArray[datetime64]: ...
    # 定义了一个方法重载：用于NDArray[datetime64]和_ArrayLikeTD64_co类型的右加法操作，返回类型为NDArray[datetime64]

    @overload
    def __radd__(self: NDArray[object_], other: Any) -> Any: ...
    # 定义了一个方法重载：用于NDArray[object_]和任意类型的右加法操作，返回类型为Any

    @overload
    def __radd__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    # 定义了一个方法重载：用于NDArray[Any]和_ArrayLikeObject_co类型的右加法操作，返回类型为Any

    @overload
    def __sub__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    # 定义了一个方法重载：用于NDArray[_UnknownType]和_ArrayLikeUnknown类型的减法操作，返回类型为NDArray[Any]

    @overload
    def __sub__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NoReturn: ...
    # 定义了一个方法重载：用于NDArray[np.bool]和_ArrayLikeBool_co类型的减法操作，返回类型为NoReturn

    @overload
    def __sub__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...
    # 定义了一个方法重载：用于_ArrayUInt_co和_ArrayLikeUInt_co类型的减法操作，返回类型为NDArray[unsignedinteger[Any]]
    # 忽略类型检查(misc)

    @overload
    def __sub__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    # 定义了一个方法重载：用于_ArrayInt_co和_ArrayLikeInt_co类型的减法操作，返回类型为NDArray[signedinteger[Any]]
    # 忽略类型检查(misc)

    @overload
    def __sub__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...
    # 定义了一个方法重载：用于_ArrayFloat_co和_ArrayLikeFloat_co类型的减法操作，返回类型为NDArray[floating[Any]]
    # 忽略类型检查(misc)

    @overload
    def __sub__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...
    # 定义了一个方法重载：用于_ArrayComplex_co和_ArrayLikeComplex_co类型的减法操作，返回类型为NDArray[complexfloating[Any, Any]]
    # 忽略类型检查(misc)

    @overload
    def __sub__(self: NDArray[number[Any]], other: _ArrayLikeNumber_co) -> NDArray[number[Any]]: ...
    # 定义了一个方法重载：用于NDArray[number[Any]]和_ArrayLikeNumber_co类型的减法操作，返回类型为NDArray[number[Any]]

    @overload
    def __sub__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...
    # 定义了一个方法重载：用于_ArrayTD64_co和_ArrayLikeTD64_co类型的减法操作，返回类型为NDArray[timedelta64]
    # 忽略类型检查(misc)

    @overload
    def __sub__(self: NDArray[datetime64], other: _ArrayLikeTD64_co) -> NDArray[datetime64]: ...
    # 定义了一个方法重载：用于NDArray[datetime64]和_ArrayLikeTD64_co类型的减法操作，返回类型为NDArray[datetime64]

    @overload
    def __sub__(self: NDArray[datetime64], other: _ArrayLikeDT64_co) -> NDArray[timedelta64]: ...
    # 定义了一个方法重载：用于NDArray[datetime64]和_ArrayLikeDT64_co类型的减法操作，返回类型为NDArray[timedelta64]

    @overload
    def __sub__(self: NDArray[object_], other: Any) -> Any: ...
    # 定义了一个方法重载：用于NDArray[object_]和任意类型的减法操作，返回类型为Any

    @overload
    def __sub__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    # 定义了一个方法重载：用于NDArray[Any]和_ArrayLikeObject_co类型的减法操作，返回类型为Any
    # 定义了 ndarray 类的 __rsub__ 方法的多个重载，实现右操作数为各种类型的数组时的减法操作
    def __rsub__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    
    # 定义了 ndarray 类的 __rsub__ 方法的重载，实现右操作数为布尔类型数组时的减法操作
    def __rsub__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NoReturn: ...
    
    # 定义了 ndarray 类的 __rsub__ 方法的重载，实现右操作数为无符号整数数组时的减法操作
    def __rsub__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    
    # 定义了 ndarray 类的 __rsub__ 方法的重载，实现右操作数为有符号整数数组时的减法操作
    def __rsub__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    
    # 定义了 ndarray 类的 __rsub__ 方法的重载，实现右操作数为浮点数数组时的减法操作
    def __rsub__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    
    # 定义了 ndarray 类的 __rsub__ 方法的重载，实现右操作数为复数数组时的减法操作
    def __rsub__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    
    # 定义了 ndarray 类的 __rsub__ 方法的重载，实现右操作数为任意数值类型数组时的减法操作
    def __rsub__(self: NDArray[number[Any]], other: _ArrayLikeNumber_co) -> NDArray[number[Any]]: ...
    
    # 定义了 ndarray 类的 __rsub__ 方法的重载，实现右操作数为时间间隔数组时的减法操作
    def __rsub__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...  # type: ignore[misc]
    
    # 定义了 ndarray 类的 __rsub__ 方法的重载，实现右操作数为日期时间数组时的减法操作
    def __rsub__(self: _ArrayTD64_co, other: _ArrayLikeDT64_co) -> NDArray[datetime64]: ...  # type: ignore[misc]
    
    # 定义了 ndarray 类的 __rsub__ 方法的重载，实现右操作数为日期时间数组时的减法操作
    def __rsub__(self: NDArray[datetime64], other: _ArrayLikeDT64_co) -> NDArray[timedelta64]: ...
    
    # 定义了 ndarray 类的 __rsub__ 方法的重载，实现右操作数为对象数组时的减法操作
    def __rsub__(self: NDArray[object_], other: Any) -> Any: ...
    
    # 定义了 ndarray 类的 __rsub__ 方法的重载，实现右操作数为任意类型数组时的减法操作
    def __rsub__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    
    # 定义了 ndarray 类的 __mul__ 方法的多个重载，实现乘法操作
    def __mul__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    
    # 定义了 ndarray 类的 __mul__ 方法的重载，实现右操作数为布尔类型数组时的乘法操作
    def __mul__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[np.bool]: ...  # type: ignore[misc]
    
    # 定义了 ndarray 类的 __mul__ 方法的重载，实现右操作数为无符号整数数组时的乘法操作
    def __mul__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    
    # 定义了 ndarray 类的 __mul__ 方法的重载，实现右操作数为有符号整数数组时的乘法操作
    def __mul__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    
    # 定义了 ndarray 类的 __mul__ 方法的重载，实现右操作数为浮点数数组时的乘法操作
    def __mul__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    
    # 定义了 ndarray 类的 __mul__ 方法的重载，实现右操作数为复数数组时的乘法操作
    def __mul__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    
    # 定义了 ndarray 类的 __mul__ 方法的重载，实现右操作数为任意数值类型数组时的乘法操作
    def __mul__(self: NDArray[number[Any]], other: _ArrayLikeNumber_co) -> NDArray[number[Any]]: ...
    
    # 定义了 ndarray 类的 __mul__ 方法的重载，实现右操作数为时间间隔数组时的乘法操作
    def __mul__(self: _ArrayTD64_co, other: _ArrayLikeFloat_co) -> NDArray[timedelta64]: ...
    
    # 定义了 ndarray 类的 __mul__ 方法的重载，实现右操作数为浮点数与时间间隔数组时的乘法操作
    def __mul__(self: _ArrayFloat_co, other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...
    
    # 定义了 ndarray 类的 __mul__ 方法的重载，实现右操作数为对象数组时的乘法操作
    def __mul__(self: NDArray[object_], other: Any) -> Any: ...
    
    # 定义了 ndarray 类的 __mul__ 方法的重载，实现右操作数为任意类型数组时的乘法操作
    def __mul__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    def __rmul__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: NDArray[number[Any]], other: _ArrayLikeNumber_co) -> NDArray[number[Any]]: ...
    @overload
    def __rmul__(self: _ArrayTD64_co, other: _ArrayLikeFloat_co) -> NDArray[timedelta64]: ...
    @overload
    def __rmul__(self: _ArrayFloat_co, other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...
    @overload
    def __rmul__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rmul__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...



    @overload
    def __floordiv__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    @overload
    def __floordiv__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self: NDArray[timedelta64], other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> NDArray[int64]: ...
    @overload
    def __floordiv__(self: NDArray[timedelta64], other: _ArrayLikeBool_co) -> NoReturn: ...
    @overload
    def __floordiv__(self: NDArray[timedelta64], other: _ArrayLikeFloat_co) -> NDArray[timedelta64]: ...
    @overload
    def __floordiv__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __floordiv__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...



    @overload
    def __rfloordiv__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    @overload
    def __rfloordiv__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self: NDArray[timedelta64], other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> NDArray[int64]: ...
    @overload
    def __rfloordiv__(self: NDArray[timedelta64], other: _ArrayLikeBool_co) -> NoReturn: ...
    @overload
    def __rfloordiv__(self: NDArray[timedelta64], other: _ArrayLikeFloat_co) -> NDArray[timedelta64]: ...
    @overload
    def __rfloordiv__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rfloordiv__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...



# 这部分代码定义了多个方法的重载（overload），用于实现数组的右乘（__rmul__）、整除（__floordiv__）和右整除（__rfloordiv__）运算。
# 每个方法根据不同的输入类型（例如整数、浮点数、复数、时间间隔等）返回相应类型的NumPy数组（NDArray）。
# type: ignore[misc] 注释指示类型检查忽略某些警告，这通常是为了处理特定的类型推断问题或类型系统的限制。
    # 定义特殊方法 __rfloordiv__，用于实现右除法运算符 // 的功能
    def __rfloordiv__(self: NDArray[timedelta64], other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> NDArray[int64]: ...
    
    # __rfloordiv__ 的重载，处理布尔类型的右除法运算
    @overload
    def __rfloordiv__(self: NDArray[np.bool], other: _ArrayLikeTD64_co) -> NoReturn: ...
    
    # __rfloordiv__ 的重载，处理浮点数数组与时间增量数组之间的右除法运算
    @overload
    def __rfloordiv__(self: _ArrayFloat_co, other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...
    
    # __rfloordiv__ 的重载，处理对象数组与任意类型的右除法运算
    @overload
    def __rfloordiv__(self: NDArray[object_], other: Any) -> Any: ...
    
    # __rfloordiv__ 的重载，处理任意类型数组与对象数组之间的右除法运算
    @overload
    def __rfloordiv__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    # 定义特殊方法 __pow__，用于实现乘方运算符 ** 的功能
    def __pow__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    
    # __pow__ 的重载，处理布尔类型与布尔类型数组之间的乘方运算
    @overload
    def __pow__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    
    # __pow__ 的重载，处理无符号整数数组与无符号整数数组之间的乘方运算
    @overload
    def __pow__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    
    # __pow__ 的重载，处理有符号整数数组与有符号整数数组之间的乘方运算
    @overload
    def __pow__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    
    # __pow__ 的重载，处理浮点数数组与浮点数数组之间的乘方运算
    @overload
    def __pow__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    
    # __pow__ 的重载，处理复数数组与复数数组之间的乘方运算
    @overload
    def __pow__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...
    
    # __pow__ 的重载，处理任意数值类型数组与数值类型数组之间的乘方运算
    @overload
    def __pow__(self: NDArray[number[Any]], other: _ArrayLikeNumber_co) -> NDArray[number[Any]]: ...
    
    # __pow__ 的重载，处理对象数组与任意类型的乘方运算
    @overload
    def __pow__(self: NDArray[object_], other: Any) -> Any: ...
    
    # __pow__ 的重载，处理任意类型数组与对象数组之间的乘方运算
    @overload
    def __pow__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    # 定义特殊方法 __rpow__，用于实现右乘方运算符 ** 的功能
    def __rpow__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    
    # __rpow__ 的重载，处理布尔类型与布尔类型数组之间的右乘方运算
    @overload
    def __rpow__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    
    # __rpow__ 的重载，处理无符号整数数组与无符号整数数组之间的右乘方运算
    @overload
    def __rpow__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    
    # __rpow__ 的重载，处理有符号整数数组与有符号整数数组之间的右乘方运算
    @overload
    def __rpow__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    
    # __rpow__ 的重载，处理浮点数数组与浮点数数组之间的右乘方运算
    @overload
    def __rpow__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    
    # __rpow__ 的重载，处理复数数组与复数数组之间的右乘方运算
    @overload
    def __rpow__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...
    
    # __rpow__ 的重载，处理任意数值类型数组与数值类型数组之间的右乘方运算
    @overload
    def __rpow__(self: NDArray[number[Any]], other: _ArrayLikeNumber_co) -> NDArray[number[Any]]: ...
    
    # __rpow__ 的重载，处理对象数组与任意类型的右乘方运算
    @overload
    def __rpow__(self: NDArray[object_], other: Any) -> Any: ...
    
    # __rpow__ 的重载，处理任意类型数组与对象数组之间的右乘方运算
    @overload
    def __rpow__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    # 定义特殊方法 __truediv__，用于实现真除法运算符 / 的功能
    def __truediv__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    
    # __truediv__ 的重载，处理浮点数数组与浮点数数组之间的真除法运算
    @overload
    def __truediv__(self: _ArrayInt_co, other: _ArrayInt_co) -> NDArray[float64]: ...  # type: ignore[misc]
    
    # __truediv__ 的重载，处理整数数组与整数数组之间的真除法运算
    @overload
    def __truediv__(self: _ArrayUInt_co, other: _ArrayUInt_co) -> NDArray[float64]: ...  # type: ignore[misc]
    
    # __truediv__ 的重载，处理对象数组与任意类型的真除法运算
    @overload
    def __truediv__(self: NDArray[object_], other: Any) -> Any: ...
    
    # __truediv__ 的重载，处理任意类型数组与对象数组之间的真除法运算
    @overload
    def __truediv__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    def __truediv__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __truediv__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __truediv__(self: NDArray[number[Any]], other: _ArrayLikeNumber_co) -> NDArray[number[Any]]: ...
    @overload
    def __truediv__(self: NDArray[timedelta64], other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> NDArray[float64]: ...
    @overload
    def __truediv__(self: NDArray[timedelta64], other: _ArrayLikeBool_co) -> NoReturn: ...
    @overload
    def __truediv__(self: NDArray[timedelta64], other: _ArrayLikeFloat_co) -> NDArray[timedelta64]: ...
    @overload
    def __truediv__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __truediv__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...


    @overload
    def __rtruediv__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    @overload
    def __rtruediv__(self: _ArrayInt_co, other: _ArrayInt_co) -> NDArray[float64]: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self: NDArray[number[Any]], other: _ArrayLikeNumber_co) -> NDArray[number[Any]]: ...
    @overload
    def __rtruediv__(self: NDArray[timedelta64], other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> NDArray[float64]: ...
    @overload
    def __rtruediv__(self: NDArray[np.bool], other: _ArrayLikeTD64_co) -> NoReturn: ...
    @overload
    def __rtruediv__(self: _ArrayFloat_co, other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...
    @overload
    def __rtruediv__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rtruediv__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...


    @overload
    def __lshift__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    @overload
    def __lshift__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    @overload
    def __lshift__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __lshift__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __lshift__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __lshift__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...


    @overload
    def __rlshift__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    @overload
    def __rlshift__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    # 定义位左移操作的特殊方法，用于处理布尔类型数组的按位左移运算，返回的结果是 int8 类型的数组

    @overload
    def __rlshift__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    # 定义位左移操作的重载方法，用于处理无符号整数类型数组的按位左移运算，返回的结果是 unsignedinteger 类型的数组

    @overload
    def __rlshift__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    # 定义位左移操作的重载方法，用于处理有符号整数类型数组的按位左移运算，返回的结果是 signedinteger 类型的数组

    @overload
    def __rlshift__(self: NDArray[object_], other: Any) -> Any: ...
    # 定义位左移操作的重载方法，用于处理对象类型数组的按位左移运算，返回的结果可以是任意类型

    @overload
    def __rlshift__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    # 定义位左移操作的重载方法，用于处理任意类型数组的按位左移运算，返回的结果可以是任意类型

    @overload
    def __rshift__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    # 定义位右移操作的重载方法，用于处理未知类型数组的按位右移运算，返回的结果可以是任意类型的数组

    @overload
    def __rshift__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    # 定义位右移操作的重载方法，用于处理布尔类型数组的按位右移运算，返回的结果是 int8 类型的数组

    @overload
    def __rshift__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    # 定义位右移操作的重载方法，用于处理无符号整数类型数组的按位右移运算，返回的结果是 unsignedinteger 类型的数组

    @overload
    def __rshift__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    # 定义位右移操作的重载方法，用于处理有符号整数类型数组的按位右移运算，返回的结果是 signedinteger 类型的数组

    @overload
    def __rshift__(self: NDArray[object_], other: Any) -> Any: ...
    # 定义位右移操作的重载方法，用于处理对象类型数组的按位右移运算，返回的结果可以是任意类型

    @overload
    def __rshift__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    # 定义位右移操作的重载方法，用于处理任意类型数组的按位右移运算，返回的结果可以是任意类型

    @overload
    def __rrshift__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    # 定义反向位右移操作的重载方法，用于处理未知类型数组的反向按位右移运算，返回的结果可以是任意类型的数组

    @overload
    def __rrshift__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    # 定义反向位右移操作的重载方法，用于处理布尔类型数组的反向按位右移运算，返回的结果是 int8 类型的数组

    @overload
    def __rrshift__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    # 定义反向位右移操作的重载方法，用于处理无符号整数类型数组的反向按位右移运算，返回的结果是 unsignedinteger 类型的数组

    @overload
    def __rrshift__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    # 定义反向位右移操作的重载方法，用于处理有符号整数类型数组的反向按位右移运算，返回的结果是 signedinteger 类型的数组

    @overload
    def __rrshift__(self: NDArray[object_], other: Any) -> Any: ...
    # 定义反向位右移操作的重载方法，用于处理对象类型数组的反向按位右移运算，返回的结果可以是任意类型

    @overload
    def __rrshift__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    # 定义反向位右移操作的重载方法，用于处理任意类型数组的反向按位右移运算，返回的结果可以是任意类型

    @overload
    def __and__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    # 定义按位与操作的重载方法，用于处理未知类型数组的按位与运算，返回的结果可以是任意类型的数组

    @overload
    def __and__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[np.bool]: ...  # type: ignore[misc]
    # 定义按位与操作的重载方法，用于处理布尔类型数组的按位与运算，返回的结果是布尔类型的数组

    @overload
    def __and__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    # 定义按位与操作的重载方法，用于处理无符号整数类型数组的按位与运算，返回的结果是 unsignedinteger 类型的数组

    @overload
    def __and__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    # 定义按位与操作的重载方法，用于处理有符号整数类型数组的按位与运算，返回的结果是 signedinteger 类型的数组

    @overload
    def __and__(self: NDArray[object_], other: Any) -> Any: ...
    # 定义按位与操作的重载方法，用于处理对象类型数组的按位与运算，返回的结果可以是任意类型

    @overload
    def __and__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    # 定义按位与操作的重载方法，用于处理任意类型数组的按位与运算，返回的结果可以是任意类型

    @overload
    def __rand__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    # 定义反向按位与操作的重载方法，用于处理未知类型数组的反向按位与运算，返回的结果可以是任意类型的数组

    @overload
    def __rand__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[np.bool]: ...  # type: ignore[misc]
    # 定义反向按位与操作的重载方法，用于处理布尔类型数组的反向按位与运算，返回的结果是布尔类型的数组

    @overload
    def __rand__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    # 定义反向按位与操作的重载方法，用于处理无符号整数类型数组的反向按位与运算，返回的结果是 unsignedinteger 类型的数组
    # 定义特殊方法 `__rand__`，用于实现数组和整数或数组的按位与操作
    def __rand__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    # 重载 `__rand__` 方法，支持数组和任意对象的按位与操作
    @overload
    def __rand__(self: NDArray[object_], other: Any) -> Any: ...
    # 重载 `__rand__` 方法，支持数组和对象数组的按位与操作
    @overload
    def __rand__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    
    # 定义特殊方法 `__xor__`，用于实现数组和数组或整数的按位异或操作
    def __xor__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    # 重载 `__xor__` 方法，支持布尔数组和布尔数组的按位异或操作，忽略类型检查
    @overload
    def __xor__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[np.bool]: ...  # type: ignore[misc]
    # 重载 `__xor__` 方法，支持无符号整数数组和无符号整数数组的按位异或操作，忽略类型检查
    @overload
    def __xor__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    # 重载 `__xor__` 方法，支持有符号整数数组和有符号整数数组的按位异或操作
    @overload
    def __xor__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    # 重载 `__xor__` 方法，支持数组和任意对象的按位异或操作
    @overload
    def __xor__(self: NDArray[object_], other: Any) -> Any: ...
    # 重载 `__xor__` 方法，支持数组和对象数组的按位异或操作
    @overload
    def __xor__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    
    # 定义特殊方法 `__rxor__`，用于实现整数或数组和数组的反向按位异或操作
    def __rxor__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    # 重载 `__rxor__` 方法，支持布尔数组和布尔数组的反向按位异或操作，忽略类型检查
    @overload
    def __rxor__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[np.bool]: ...  # type: ignore[misc]
    # 重载 `__rxor__` 方法，支持无符号整数数组和无符号整数数组的反向按位异或操作，忽略类型检查
    @overload
    def __rxor__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    # 重载 `__rxor__` 方法，支持有符号整数数组和有符号整数数组的反向按位异或操作
    @overload
    def __rxor__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    # 重载 `__rxor__` 方法，支持数组和任意对象的反向按位异或操作
    @overload
    def __rxor__(self: NDArray[object_], other: Any) -> Any: ...
    # 重载 `__rxor__` 方法，支持数组和对象数组的反向按位异或操作
    @overload
    def __rxor__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    
    # 定义特殊方法 `__or__`，用于实现数组和数组或整数的按位或操作
    def __or__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    # 重载 `__or__` 方法，支持布尔数组和布尔数组的按位或操作，忽略类型检查
    @overload
    def __or__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[np.bool]: ...  # type: ignore[misc]
    # 重载 `__or__` 方法，支持无符号整数数组和无符号整数数组的按位或操作，忽略类型检查
    @overload
    def __or__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    # 重载 `__or__` 方法，支持有符号整数数组和有符号整数数组的按位或操作
    @overload
    def __or__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    # 重载 `__or__` 方法，支持数组和任意对象的按位或操作
    @overload
    def __or__(self: NDArray[object_], other: Any) -> Any: ...
    # 重载 `__or__` 方法，支持数组和对象数组的按位或操作
    @overload
    def __or__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    
    # 定义特殊方法 `__ror__`，用于实现整数或数组和数组的反向按位或操作
    def __ror__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    # 重载 `__ror__` 方法，支持布尔数组和布尔数组的反向按位或操作，忽略类型检查
    @overload
    def __ror__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[np.bool]: ...  # type: ignore[misc]
    # 重载 `__ror__` 方法，支持无符号整数数组和无符号整数数组的反向按位或操作，忽略类型检查
    @overload
    def __ror__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    # 重载 `__ror__` 方法，支持有符号整数数组和有符号整数数组的反向按位或操作
    @overload
    def __ror__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    # 重载 `__ror__` 方法，支持数组和任意对象的反向按位或操作
    @overload
    def __ror__(self: NDArray[object_], other: Any) -> Any: ...
    # 重载 `__ror__` 方法，支持数组和对象数组的反向按位或操作
    @overload
    def __ror__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...
    
    # `np.generic` 类型不支持原地操作
    # NOTE: Inplace addition operation overloads for NumPy arrays.
    # Defines the behavior of __iadd__ for various types of NumPy arrays.
    @overload
    def __iadd__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...

    # Defines __iadd__ for boolean NumPy arrays (__iadd__ with boolean arrays).
    @overload
    def __iadd__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[np.bool]: ...

    # Defines __iadd__ for unsigned integer NumPy arrays.
    # Allows adding a signed integer when the right operand is 0-dimensional and >= 0.
    @overload
    def __iadd__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co | _IntLike_co) -> NDArray[unsignedinteger[_NBit1]]: ...

    # Defines __iadd__ for signed integer NumPy arrays.
    @overload
    def __iadd__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...

    # Defines __iadd__ for floating-point NumPy arrays.
    @overload
    def __iadd__(self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co) -> NDArray[floating[_NBit1]]: ...

    # Defines __iadd__ for complex floating-point NumPy arrays.
    @overload
    def __iadd__(self: NDArray[complexfloating[_NBit1, _NBit1]], other: _ArrayLikeComplex_co) -> NDArray[complexfloating[_NBit1, _NBit1]]: ...

    # Defines __iadd__ for NumPy timedelta64 arrays.
    @overload
    def __iadd__(self: NDArray[timedelta64], other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...

    # Defines __iadd__ for NumPy datetime64 arrays.
    @overload
    def __iadd__(self: NDArray[datetime64], other: _ArrayLikeTD64_co) -> NDArray[datetime64]: ...

    # Defines __iadd__ for generic object NumPy arrays.
    @overload
    def __iadd__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    # Defines the behavior of inplace subtraction (__isub__) for various types of NumPy arrays.
    @overload
    def __isub__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...

    # Defines __isub__ for unsigned integer NumPy arrays.
    @overload
    def __isub__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co | _IntLike_co) -> NDArray[unsignedinteger[_NBit1]]: ...

    # Defines __isub__ for signed integer NumPy arrays.
    @overload
    def __isub__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...

    # Defines __isub__ for floating-point NumPy arrays.
    @overload
    def __isub__(self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co) -> NDArray[floating[_NBit1]]: ...

    # Defines __isub__ for complex floating-point NumPy arrays.
    @overload
    def __isub__(self: NDArray[complexfloating[_NBit1, _NBit1]], other: _ArrayLikeComplex_co) -> NDArray[complexfloating[_NBit1, _NBit1]]: ...

    # Defines __isub__ for NumPy timedelta64 arrays.
    @overload
    def __isub__(self: NDArray[timedelta64], other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...

    # Defines __isub__ for NumPy datetime64 arrays.
    @overload
    def __isub__(self: NDArray[datetime64], other: _ArrayLikeTD64_co) -> NDArray[datetime64]: ...

    # Defines __isub__ for generic object NumPy arrays.
    @overload
    def __isub__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    # Defines the behavior of inplace multiplication (__imul__) for various types of NumPy arrays.
    @overload
    def __imul__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...

    # Defines __imul__ for boolean NumPy arrays (__imul__ with boolean arrays).
    @overload
    def __imul__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[np.bool]: ...

    # Defines __imul__ for unsigned integer NumPy arrays.
    @overload
    def __imul__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co | _IntLike_co) -> NDArray[unsignedinteger[_NBit1]]: ...
    # 定义 __imul__ 方法的类型注解和返回类型，接收两个参数：self 和 other
    def __imul__(self: NDArray[complexfloating[_NBit1, _NBit1]], other: _ArrayLikeComplex_co) -> NDArray[complexfloating[_NBit1, _NBit1]]: ...

    # __imul__ 方法的类型注解和返回类型，接收两个参数：self 和 other，针对 timedelta64 类型的数组
    @overload
    def __imul__(self: NDArray[timedelta64], other: _ArrayLikeFloat_co) -> NDArray[timedelta64]: ...

    # __imul__ 方法的类型注解和返回类型，接收两个参数：self 和 other，针对 object_ 类型的数组
    @overload
    def __imul__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    # 定义 __itruediv__ 方法的类型注解和返回类型，接收两个参数：self 和 other
    @overload
    def __itruediv__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...

    # __itruediv__ 方法的类型注解和返回类型，接收两个参数：self 和 other，针对 floating[_NBit1] 类型的数组
    @overload
    def __itruediv__(self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co) -> NDArray[floating[_NBit1]]: ...

    # __itruediv__ 方法的类型注解和返回类型，接收两个参数：self 和 other，针对 complexfloating[_NBit1, _NBit1] 类型的数组
    @overload
    def __itruediv__(self: NDArray[complexfloating[_NBit1, _NBit1]], other: _ArrayLikeComplex_co) -> NDArray[complexfloating[_NBit1, _NBit1]]: ...

    # __itruediv__ 方法的类型注解和返回类型，接收两个参数：self 和 other，针对 timedelta64 类型的数组和 _ArrayLikeBool_co 类型的参数
    @overload
    def __itruediv__(self: NDArray[timedelta64], other: _ArrayLikeBool_co) -> NoReturn: ...

    # __itruediv__ 方法的类型注解和返回类型，接收两个参数：self 和 other，针对 timedelta64 类型的数组和 _ArrayLikeInt_co 类型的参数
    @overload
    def __itruediv__(self: NDArray[timedelta64], other: _ArrayLikeInt_co) -> NDArray[timedelta64]: ...

    # __itruediv__ 方法的类型注解和返回类型，接收两个参数：self 和 other，针对 object_ 类型的数组
    @overload
    def __itruediv__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    # 定义 __ifloordiv__ 方法的类型注解和返回类型，接收两个参数：self 和 other
    @overload
    def __ifloordiv__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...

    # __ifloordiv__ 方法的类型注解和返回类型，接收两个参数：self 和 other，针对 unsignedinteger[_NBit1] 类型的数组和 _ArrayLikeUInt_co | _IntLike_co 类型的参数
    @overload
    def __ifloordiv__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co | _IntLike_co) -> NDArray[unsignedinteger[_NBit1]]: ...

    # __ifloordiv__ 方法的类型注解和返回类型，接收两个参数：self 和 other，针对 signedinteger[_NBit1] 类型的数组和 _ArrayLikeInt_co 类型的参数
    @overload
    def __ifloordiv__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...

    # __ifloordiv__ 方法的类型注解和返回类型，接收两个参数：self 和 other，针对 floating[_NBit1] 类型的数组和 _ArrayLikeFloat_co 类型的参数
    @overload
    def __ifloordiv__(self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co) -> NDArray[floating[_NBit1]]: ...

    # __ifloordiv__ 方法的类型注解和返回类型，接收两个参数：self 和 other，针对 complexfloating[_NBit1, _NBit1] 类型的数组和 _ArrayLikeComplex_co 类型的参数
    @overload
    def __ifloordiv__(self: NDArray[complexfloating[_NBit1, _NBit1]], other: _ArrayLikeComplex_co) -> NDArray[complexfloating[_NBit1, _NBit1]]: ...

    # __ifloordiv__ 方法的类型注解和返回类型，接收两个参数：self 和 other，针对 timedelta64 类型的数组和 _ArrayLikeBool_co 类型的参数
    @overload
    def __ifloordiv__(self: NDArray[timedelta64], other: _ArrayLikeBool_co) -> NoReturn: ...

    # __ifloordiv__ 方法的类型注解和返回类型，接收两个参数：self 和 other，针对 timedelta64 类型的数组和 _ArrayLikeInt_co 类型的参数
    @overload
    def __ifloordiv__(self: NDArray[timedelta64], other: _ArrayLikeInt_co) -> NDArray[timedelta64]: ...

    # __ifloordiv__ 方法的类型注解和返回类型，接收两个参数：self 和 other，针对 object_ 类型的数组
    @overload
    def __ifloordiv__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    # 定义 __ipow__ 方法的类型注解和返回类型，接收两个参数：self 和 other
    @overload
    def __ipow__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...

    # __ipow__ 方法的类型注解和返回类型，接收两个参数：self 和 other，针对 unsignedinteger[_NBit1] 类型的数组和 _ArrayLikeUInt_co | _IntLike_co 类型的参数
    @overload
    def __ipow__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co | _IntLike_co) -> NDArray[unsignedinteger[_NBit1]]: ...

    # __ipow__ 方法的类型注解和返回类型，接收两个参数：self 和 other，针对 signedinteger[_NBit1] 类型的数组和 _ArrayLikeInt_co 类型的参数
    @overload
    def __ipow__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...

    # __ipow__ 方法的类型注解和返回类型，接收两个参数：self 和 other，针对 floating[_NBit1] 类型的数组和 _ArrayLikeFloat_co 类型的参数
    @overload
    def __ipow__(self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co) -> NDArray[floating[_NBit1]]: ...

    # __ipow__ 方法的类型注解和返回类型，接收两个参数：self 和 other，针对 complexfloating[_NBit1, _NBit1] 类型的数组和 _ArrayLikeComplex_co 类型的参数
    @overload
    def __ipow__(self: NDArray[complexfloating[_NBit1, _NBit1]], other: _ArrayLikeComplex_co) -> NDArray[complexfloating[_NBit1, _NBit1]]: ...

    # __ipow__ 方法的类型注解和返回类型，接收两个参数：self 和 other，针对 object_ 类型的数组
    @overload
    def __ipow__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    # 定义 __imod__ 方法的类型注解和返回类型，接收两个参数：self 和 other
    @overload
    def __imod__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    # 定义特殊方法 __imod__，用于就地修改（in-place）NDArray对象的操作，当对象元素为 unsigned integer 时
    def __imod__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co | _IntLike_co) -> NDArray[unsignedinteger[_NBit1]]: ...

    # 定义特殊方法 __imod__ 的重载，处理 signed integer 元素的情况
    @overload
    def __imod__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...

    # 定义特殊方法 __imod__ 的重载，处理 floating point 元素的情况
    @overload
    def __imod__(self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co) -> NDArray[floating[_NBit1]]: ...

    # 定义特殊方法 __imod__ 的重载，处理 timedelta64 元素的情况
    @overload
    def __imod__(self: NDArray[timedelta64], other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> NDArray[timedelta64]: ...

    # 定义特殊方法 __imod__ 的重载，处理 object 元素的情况
    @overload
    def __imod__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    # 定义特殊方法 __ilshift__，用于就地左移（in-place left shift）NDArray对象的操作，处理未知类型元素的情况
    @overload
    def __ilshift__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...

    # 定义特殊方法 __ilshift__ 的重载，处理 unsigned integer 元素的情况
    @overload
    def __ilshift__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co | _IntLike_co) -> NDArray[unsignedinteger[_NBit1]]: ...

    # 定义特殊方法 __ilshift__ 的重载，处理 signed integer 元素的情况
    @overload
    def __ilshift__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...

    # 定义特殊方法 __ilshift__ 的重载，处理 object 元素的情况
    @overload
    def __ilshift__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    # 定义特殊方法 __irshift__，用于就地右移（in-place right shift）NDArray对象的操作，处理未知类型元素的情况
    @overload
    def __irshift__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...

    # 定义特殊方法 __irshift__ 的重载，处理 unsigned integer 元素的情况
    @overload
    def __irshift__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co | _IntLike_co) -> NDArray[unsignedinteger[_NBit1]]: ...

    # 定义特殊方法 __irshift__ 的重载，处理 signed integer 元素的情况
    @overload
    def __irshift__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...

    # 定义特殊方法 __irshift__ 的重载，处理 object 元素的情况
    @overload
    def __irshift__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    # 定义特殊方法 __iand__，用于就地按位与（in-place bitwise AND）NDArray对象的操作，处理未知类型元素的情况
    @overload
    def __iand__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...

    # 定义特殊方法 __iand__ 的重载，处理布尔类型元素的情况
    @overload
    def __iand__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[np.bool]: ...

    # 定义特殊方法 __iand__ 的重载，处理 unsigned integer 元素的情况
    @overload
    def __iand__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co | _IntLike_co) -> NDArray[unsignedinteger[_NBit1]]: ...

    # 定义特殊方法 __iand__ 的重载，处理 signed integer 元素的情况
    @overload
    def __iand__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...

    # 定义特殊方法 __iand__ 的重载，处理 object 元素的情况
    @overload
    def __iand__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    # 定义特殊方法 __ixor__，用于就地按位异或（in-place bitwise XOR）NDArray对象的操作，处理未知类型元素的情况
    @overload
    def __ixor__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...

    # 定义特殊方法 __ixor__ 的重载，处理布尔类型元素的情况
    @overload
    def __ixor__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[np.bool]: ...

    # 定义特殊方法 __ixor__ 的重载，处理 unsigned integer 元素的情况
    @overload
    def __ixor__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co | _IntLike_co) -> NDArray[unsignedinteger[_NBit1]]: ...

    # 定义特殊方法 __ixor__ 的重载，处理 signed integer 元素的情况
    @overload
    def __ixor__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...

    # 定义特殊方法 __ixor__ 的重载，处理 object 元素的情况
    @overload
    def __ixor__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    # 定义特殊方法 __ior__，用于就地按位或（in-place bitwise OR）NDArray对象的操作，处理未知类型元素的情况
    @overload
    def __ior__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    # 定义了 __ior__ 方法的重载，用于原位或操作，操作数是 bool 类型的 NumPy 数组
    def __ior__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[np.bool]: ...
    
    # 定义了 __ior__ 方法的重载，用于原位或操作，操作数是 unsigned integer 类型的 NumPy 数组
    def __ior__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co | _IntLike_co) -> NDArray[unsignedinteger[_NBit1]]: ...
    
    # 定义了 __ior__ 方法的重载，用于原位或操作，操作数是 signed integer 类型的 NumPy 数组
    def __ior__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...
    
    # 定义了 __ior__ 方法的重载，用于原位或操作，操作数是 object 类型的 NumPy 数组
    def __ior__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...
    
    # 定义了 __imatmul__ 方法的重载，用于原位矩阵乘法操作，操作数是任意类型的 NumPy 数组
    def __imatmul__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    
    # 定义了 __imatmul__ 方法的重载，用于原位矩阵乘法操作，操作数是 bool 类型的 NumPy 数组
    def __imatmul__(self: NDArray[np.bool], other: _ArrayLikeBool_co) -> NDArray[np.bool]: ...
    
    # 定义了 __imatmul__ 方法的重载，用于原位矩阵乘法操作，操作数是 unsigned integer 类型的 NumPy 数组
    def __imatmul__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[_NBit1]]: ...
    
    # 定义了 __imatmul__ 方法的重载，用于原位矩阵乘法操作，操作数是 signed integer 类型的 NumPy 数组
    def __imatmul__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...
    
    # 定义了 __imatmul__ 方法的重载，用于原位矩阵乘法操作，操作数是 floating 类型的 NumPy 数组
    def __imatmul__(self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co) -> NDArray[floating[_NBit1]]: ...
    
    # 定义了 __imatmul__ 方法的重载，用于原位矩阵乘法操作，操作数是 complexfloating 类型的 NumPy 数组
    def __imatmul__(self: NDArray[complexfloating[_NBit1, _NBit1]], other: _ArrayLikeComplex_co) -> NDArray[complexfloating[_NBit1, _NBit1]]: ...
    
    # 定义了 __imatmul__ 方法的重载，用于原位矩阵乘法操作，操作数是 object 类型的 NumPy 数组
    def __imatmul__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...
    
    # 定义了 __dlpack__ 方法，返回一个 PyCapsule 对象，用于与 DLPack 兼容
    def __dlpack__(self: NDArray[number[Any]], *, stream: None = ...) -> _PyCapsule: ...
    
    # 定义了 __dlpack_device__ 方法，返回一个元组，表示数组的设备信息
    def __dlpack_device__(self) -> tuple[int, L[0]]: ...
    
    # 定义了 __array_namespace__ 方法，用于返回数组的命名空间
    def __array_namespace__(self, *, api_version: str | None = ...) -> Any: ...
    
    # 定义了 to_device 方法，将数组复制到指定的设备上，并返回结果数组
    def to_device(self, device: L["cpu"], /, *, stream: None | int | Any = ...) -> NDArray[Any]: ...
    
    # 定义了 device 属性，表示数组所在的设备类型为 "cpu"
    @property
    def device(self) -> L["cpu"]: ...
    
    # 定义了 bitwise_count 方法，用于计算数组中位为 1 的个数
    def bitwise_count(
        self,
        out: None | NDArray[Any] = ...,
        *,
        where: _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: builtins.bool = ...,
    ) -> NDArray[Any]: ...
    
    # 定义了 dtype 属性，用于返回数组的数据类型，避免与 np.dtype 发生命名冲突，放在最后定义
    @property
    def dtype(self) -> _DType_co: ...
# NOTE: while `np.generic` is not technically an instance of `ABCMeta`,
# the `@abstractmethod` decorator is herein used to (forcefully) deny
# the creation of `np.generic` instances.
# The `# type: ignore` comments are necessary to silence mypy errors regarding
# the missing `ABCMeta` metaclass.
# See https://github.com/numpy/numpy-stubs/pull/80 for more details.

# 定义一个泛型类型变量 `_ScalarType`，其上界为 `generic`
_ScalarType = TypeVar("_ScalarType", bound=generic)

# 定义两个泛型类型变量 `_NBit1` 和 `_NBit2`，它们的上界为 `NBitBase`
_NBit1 = TypeVar("_NBit1", bound=NBitBase)
_NBit2 = TypeVar("_NBit2", bound=NBitBase)

# 定义一个名为 `generic` 的类，继承自 `_ArrayOrScalarCommon`
class generic(_ArrayOrScalarCommon):
    
    # 声明抽象方法 `__init__`，接受任意参数并无返回值
    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    
    # 定义 `__array__` 方法的两种重载，用于数组类型转换
    # 第一种重载接受 `dtype=None`，返回类型为 `NDArray[_ScalarType]`
    @overload
    def __array__(self: _ScalarType, dtype: None = ..., /) -> NDArray[_ScalarType]: ...
    # 第二种重载接受指定 `dtype`，返回类型为 `ndarray[Any, _DType]`
    @overload
    def __array__(self, dtype: _DType, /) -> ndarray[Any, _DType]: ...
    
    # 声明 `__hash__` 方法，返回值为整数类型
    def __hash__(self) -> int: ...
    
    # 声明 `base` 属性，无返回值类型
    @property
    def base(self) -> None: ...
    
    # 声明 `ndim` 属性，返回值类型为 0
    @property
    def ndim(self) -> L[0]: ...
    
    # 声明 `size` 属性，返回值类型为 1
    @property
    def size(self) -> L[1]: ...
    
    # 声明 `shape` 属性，返回空元组类型
    @property
    def shape(self) -> tuple[()]: ...
    
    # 声明 `strides` 属性，返回空元组类型
    @property
    def strides(self) -> tuple[()]: ...
    
    # 声明 `byteswap` 方法，接受一个布尔类型的参数 `inplace`，默认为 False
    # 返回值类型为 `_ScalarType`
    def byteswap(self: _ScalarType, inplace: L[False] = ...) -> _ScalarType: ...
    
    # 声明 `flat` 属性，返回类型为 `flatiter[NDArray[_ScalarType]]`
    @property
    def flat(self: _ScalarType) -> flatiter[NDArray[_ScalarType]]: ...
    
    # 如果 Python 版本 >= 3.12，则声明 `__buffer__` 方法
    # 接受一个整数类型参数 `flags`，返回类型为 `memoryview`
    if sys.version_info >= (3, 12):
        def __buffer__(self, flags: int, /) -> memoryview: ...
    
    # 声明 `astype` 方法的两种重载
    # 第一种重载接受 `_DTypeLike[_ScalarType]` 类型的 `dtype` 参数
    @overload
    def astype(
        self,
        dtype: _DTypeLike[_ScalarType],
        order: _OrderKACF = ...,
        casting: _CastingKind = ...,
        subok: builtins.bool = ...,
        copy: builtins.bool | _CopyMode = ...,
    ) -> _ScalarType: ...
    # 第二种重载接受 `DTypeLike` 类型的 `dtype` 参数
    @overload
    def astype(
        self,
        dtype: DTypeLike,
        order: _OrderKACF = ...,
        casting: _CastingKind = ...,
        subok: builtins.bool = ...,
        copy: builtins.bool | _CopyMode = ...,
    ) -> Any: ...
    
    # NOTE: `view` 方法将进行从 0D 到标量的类型转换，
    # 因此数组的 `type` 与输出类型无关
    # 声明 `view` 方法的三种重载
    @overload
    def view(
        self: _ScalarType,
        type: type[NDArray[Any]] = ...,
    ) -> _ScalarType: ...
    @overload
    def view(
        self,
        dtype: _DTypeLike[_ScalarType],
        type: type[NDArray[Any]] = ...,
    ) -> _ScalarType: ...
    @overload
    def view(
        self,
        dtype: DTypeLike,
        type: type[NDArray[Any]] = ...,
    ) -> Any: ...
    
    # 声明 `getfield` 方法的两种重载
    # 第一种重载接受 `_DTypeLike[_ScalarType]` 类型的 `dtype` 参数
    @overload
    def getfield(
        self,
        dtype: _DTypeLike[_ScalarType],
        offset: SupportsIndex = ...
    ) -> _ScalarType: ...
    # 第二种重载接受 `DTypeLike` 类型的 `dtype` 参数
    @overload
    def getfield(
        self,
        dtype: DTypeLike,
        offset: SupportsIndex = ...
    ) -> Any: ...
    
    # 声明 `item` 方法，接受参数 `args`，返回值类型为 `Any`
    def item(
        self, args: L[0] | tuple[()] | tuple[L[0]] = ..., /,
    ) -> Any: ...
    
    # 声明 `take` 方法的一种重载，用于忽略类型检查错误 `# type: ignore[misc]`
    @overload
    def take(
        self: _ScalarType,
        indices: _IntLike_co,
        axis: None | SupportsIndex = ...,
        out: None = ...,
        mode: _ModeKind = ...,
    ) -> _ScalarType: ...
    def take(  # type: ignore[misc]
        self: _ScalarType,
        indices: _ArrayLikeInt_co,
        axis: None | SupportsIndex = ...,
        out: None = ...,
        mode: _ModeKind = ...,
    ) -> NDArray[_ScalarType]: ...


    @overload
    def take(
        self,
        indices: _ArrayLikeInt_co,
        axis: None | SupportsIndex = ...,
        out: _NdArraySubClass = ...,
        mode: _ModeKind = ...,
    ) -> _NdArraySubClass: ...


    def repeat(
        self: _ScalarType,
        repeats: _ArrayLikeInt_co,
        axis: None | SupportsIndex = ...,
    ) -> NDArray[_ScalarType]: ...


    def flatten(
        self: _ScalarType,
        order: _OrderKACF = ...,
    ) -> NDArray[_ScalarType]: ...


    def ravel(
        self: _ScalarType,
        order: _OrderKACF = ...,
    ) -> NDArray[_ScalarType]: ...


    @overload
    def reshape(
        self: _ScalarType, shape: _ShapeLike, /, *, order: _OrderACF = ...
    ) -> NDArray[_ScalarType]: ...
    @overload
    def reshape(
        self: _ScalarType, *shape: SupportsIndex, order: _OrderACF = ...
    ) -> NDArray[_ScalarType]: ...


    def bitwise_count(
        self,
        out: None | NDArray[Any] = ...,
        *,
        where: _ArrayLikeBool_co = ...,
        casting: _CastingKind = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: builtins.bool = ...,
    ) -> Any: ...


    def squeeze(
        self: _ScalarType, axis: None | L[0] | tuple[()] = ...
    ) -> _ScalarType: ...


    def transpose(self: _ScalarType, axes: None | tuple[()] = ..., /) -> _ScalarType: ...


    # Keep `dtype` at the bottom to avoid name conflicts with `np.dtype`
    @property
    def dtype(self: _ScalarType) -> _dtype[_ScalarType]: ...
# 定义名为 `number` 的类，继承自 `generic` 和泛型 `_NBit1`
# 忽略类型检查
class number(generic, Generic[_NBit1]):  # type: ignore

    # 定义 `real` 属性方法，返回 `_ArraySelf` 类型
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...

    # 定义 `imag` 属性方法，返回 `_ArraySelf` 类型
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...

    # 定义 `__class_getitem__` 方法，接受任意类型参数 `item`，返回 `GenericAlias` 泛型
    def __class_getitem__(self, item: Any) -> GenericAlias: ...

    # 定义 `__int__` 方法，返回 `int` 类型
    def __int__(self) -> int: ...

    # 定义 `__float__` 方法，返回 `float` 类型
    def __float__(self) -> float: ...

    # 定义 `__complex__` 方法，返回 `complex` 类型
    def __complex__(self) -> complex: ...

    # 定义 `__neg__` 方法，返回 `_ArraySelf` 类型
    def __neg__(self: _ArraySelf) -> _ArraySelf: ...

    # 定义 `__pos__` 方法，返回 `_ArraySelf` 类型
    def __pos__(self: _ArraySelf) -> _ArraySelf: ...

    # 定义 `__abs__` 方法，返回 `_ArraySelf` 类型
    def __abs__(self: _ArraySelf) -> _ArraySelf: ...

    # 确保被标注为 `number` 的对象支持算术操作

    # 定义 `__add__`、`__radd__` 等运算符方法，均返回 `_NumberOp` 类型
    __add__: _NumberOp
    __radd__: _NumberOp
    __sub__: _NumberOp
    __rsub__: _NumberOp
    __mul__: _NumberOp
    __rmul__: _NumberOp
    __floordiv__: _NumberOp
    __rfloordiv__: _NumberOp
    __pow__: _NumberOp
    __rpow__: _NumberOp
    __truediv__: _NumberOp
    __rtruediv__: _NumberOp

    # 定义比较运算符方法 `__lt__`、`__le__` 等，接受 `_NumberLike_co` 和 `_ArrayLikeNumber_co` 类型参数
    __lt__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __le__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __gt__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __ge__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]

# 定义名为 `bool` 的类，继承自 `generic`
class bool(generic):

    # 定义 `__init__` 构造方法，接受 `value` 参数，默认值为 `...`
    def __init__(self, value: object = ..., /) -> None: ...

    # 定义 `item` 方法，返回 `builtins.bool` 类型，接受参数 `args`
    def item(
        self, args: L[0] | tuple[()] | tuple[L[0]] = ..., /,
    ) -> builtins.bool: ...

    # 定义 `tolist` 方法，返回 `builtins.bool` 类型
    def tolist(self) -> builtins.bool: ...

    # 定义 `real` 属性方法，返回 `_ArraySelf` 类型
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...

    # 定义 `imag` 属性方法，返回 `_ArraySelf` 类型
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...

    # 定义 `__int__` 方法，返回 `int` 类型
    def __int__(self) -> int: ...

    # 定义 `__float__` 方法，返回 `float` 类型
    def __float__(self) -> float: ...

    # 定义 `__complex__` 方法，返回 `complex` 类型
    def __complex__(self) -> complex: ...

    # 定义 `__abs__` 方法，返回 `_ArraySelf` 类型
    def __abs__(self: _ArraySelf) -> _ArraySelf: ...

    # 定义 `__invert__` 方法，返回 `np.bool` 类型
    def __invert__(self) -> np.bool: ...

    # 定义位运算符方法 `__lshift__`、`__rlshift__` 等，返回对应的操作类型
    __lshift__: _BoolBitOp[int8]
    __rlshift__: _BoolBitOp[int8]
    __rshift__: _BoolBitOp[int8]
    __rrshift__: _BoolBitOp[int8]
    __and__: _BoolBitOp[np.bool]
    __rand__: _BoolBitOp[np.bool]
    __xor__: _BoolBitOp[np.bool]
    __rxor__: _BoolBitOp[np.bool]
    __or__: _BoolBitOp[np.bool]
    __ror__: _BoolBitOp[np.bool]

    # 定义算术运算符方法 `__add__`、`__radd__` 等，返回对应的操作类型
    __add__: _BoolOp[np.bool]
    __radd__: _BoolOp[np.bool]
    __sub__: _BoolSub
    __rsub__: _BoolSub
    __mul__: _BoolOp[np.bool]
    __rmul__: _BoolOp[np.bool]
    __floordiv__: _BoolOp[int8]
    __rfloordiv__: _BoolOp[int8]
    __pow__: _BoolOp[int8]
    __rpow__: _BoolOp[int8]
    __truediv__: _BoolTrueDiv
    __rtruediv__: _BoolTrueDiv

    # 定义其他运算符方法 `__mod__`、`__divmod__` 等，返回对应的操作类型
    __mod__: _BoolMod
    __rmod__: _BoolMod
    __divmod__: _BoolDivMod
    __rdivmod__: _BoolDivMod

    # 定义比较运算符方法 `__lt__`、`__le__` 等，接受 `_NumberLike_co` 和 `_ArrayLikeNumber_co` 类型参数
    __lt__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __le__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __gt__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __ge__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]

# 定义 `bool_` 为 `bool` 类的别名
bool_ = bool

# 定义名为 `object_` 的类，继承自 `generic`
class object_(generic):

    # 定义 `__init__` 构造方法，接受 `value` 参数，默认值为 `...`
    def __init__(self, value: object = ..., /) -> None: ...

    # 定义 `real` 属性方法，返回 `_ArraySelf` 类型
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...

    # 定义 `imag` 属性方法，返回 `_ArraySelf` 类型
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...
    # 定义特殊方法 __int__，用于将对象转换为整数，返回一个整数
    # 这里的返回类型注解指定返回值为 int 类型
    def __int__(self) -> int: ...

    # 定义特殊方法 __float__，用于将对象转换为浮点数，返回一个浮点数
    # 这里的返回类型注解指定返回值为 float 类型
    def __float__(self) -> float: ...

    # 定义特殊方法 __complex__，用于将对象转换为复数，返回一个复数
    # 这里的返回类型注解指定返回值为 complex 类型
    def __complex__(self) -> complex: ...

    # 如果 Python 解释器版本大于等于 3.12
    if sys.version_info >= (3, 12):
        # 定义特殊方法 __release_buffer__，用于释放内存视图
        # 参数 buffer 是 memoryview 对象，类型注解指定参数为不定位置参数
        # 返回类型注解指定返回值为 None，即没有返回值
        def __release_buffer__(self, buffer: memoryview, /) -> None: ...
# `datetime64` 构造器要求对象具有以下三个属性，因此支持日期时间的鸭子类型
class _DatetimeScalar(Protocol):
    @property
    def day(self) -> int: ...  # 属性：返回日期中的天数
    @property
    def month(self) -> int: ...  # 属性：返回日期中的月份
    @property
    def year(self) -> int: ...  # 属性：返回日期中的年份

# TODO: `item`/`tolist` 方法根据单位返回 `dt.date`、`dt.datetime` 或 `int`
# 类型的值
class datetime64(generic):
    @overload
    def __init__(
        self,
        value: None | datetime64 | _CharLike_co | _DatetimeScalar = ...,
        format: _CharLike_co | tuple[_CharLike_co, _IntLike_co] = ...,
        /,
    ) -> None: ...  # 构造函数：初始化 `datetime64` 对象
    @overload
    def __init__(
        self,
        value: int,
        format: _CharLike_co | tuple[_CharLike_co, _IntLike_co],
        /,
    ) -> None: ...  # 构造函数：初始化 `datetime64` 对象
    def __add__(self, other: _TD64Like_co) -> datetime64: ...  # 方法：日期时间相加
    def __radd__(self, other: _TD64Like_co) -> datetime64: ...  # 方法：反向日期时间相加
    @overload
    def __sub__(self, other: datetime64) -> timedelta64: ...  # 方法：日期时间相减
    @overload
    def __sub__(self, other: _TD64Like_co) -> datetime64: ...  # 方法：日期时间相减
    def __rsub__(self, other: datetime64) -> timedelta64: ...  # 方法：反向日期时间相减
    __lt__: _ComparisonOp[datetime64, _ArrayLikeDT64_co]  # 属性：小于比较操作
    __le__: _ComparisonOp[datetime64, _ArrayLikeDT64_co]  # 属性：小于等于比较操作
    __gt__: _ComparisonOp[datetime64, _ArrayLikeDT64_co]  # 属性：大于比较操作
    __ge__: _ComparisonOp[datetime64, _ArrayLikeDT64_co]  # 属性：大于等于比较操作

_IntValue = SupportsInt | _CharLike_co | SupportsIndex  # 类型：整数支持的值类型
_FloatValue = None | _CharLike_co | SupportsFloat | SupportsIndex  # 类型：浮点数支持的值类型
_ComplexValue = (
    None
    | _CharLike_co
    | SupportsFloat
    | SupportsComplex
    | SupportsIndex
    | complex  # `complex` 不是 `SupportsComplex` 的子类型
)

class integer(number[_NBit1]):  # type: ignore
    @property
    def numerator(self: _ScalarType) -> _ScalarType: ...  # 属性：返回分子
    @property
    def denominator(self) -> L[1]: ...  # 属性：返回分母
    @overload
    def __round__(self, ndigits: None = ...) -> int: ...  # 方法：四舍五入
    @overload
    def __round__(self: _ScalarType, ndigits: SupportsIndex) -> _ScalarType: ...  # 方法：带精度的四舍五入
    # NOTE: `__index__` 实际上是在最底层子类 (`int64`, `uint32` 等) 中定义的
    def item(
        self, args: L[0] | tuple[()] | tuple[L[0]] = ..., /,
    ) -> int: ...  # 方法：获取元素
    def tolist(self) -> int: ...  # 方法：转换为列表
    def is_integer(self) -> L[True]: ...  # 方法：检查是否为整数
    def bit_count(self: _ScalarType) -> int: ...  # 方法：计算比特数
    def __index__(self) -> int: ...  # 方法：返回索引值
    __truediv__: _IntTrueDiv[_NBit1]  # 属性：真除法
    __rtruediv__: _IntTrueDiv[_NBit1]  # 属性：反向真除法
    def __mod__(self, value: _IntLike_co) -> integer[Any]: ...  # 方法：求余
    def __rmod__(self, value: _IntLike_co) -> integer[Any]: ...  # 方法：反向求余
    def __invert__(self: _IntType) -> _IntType: ...  # 方法：按位取反
    # 确保标记为 `integer` 的对象支持位操作
    def __lshift__(self, other: _IntLike_co) -> integer[Any]: ...  # 方法：左移位
    def __rlshift__(self, other: _IntLike_co) -> integer[Any]: ...  # 方法：反向左移位
    def __rshift__(self, other: _IntLike_co) -> integer[Any]: ...  # 方法：右移位
    def __rrshift__(self, other: _IntLike_co) -> integer[Any]: ...  # 方法：反向右移位
    # 定义按位与操作符方法，用于当前对象和另一个 _IntLike_co 类型对象的按位与操作
    def __and__(self, other: _IntLike_co) -> integer[Any]: ...
    
    # 定义反向按位与操作符方法，用于另一个 _IntLike_co 类型对象和当前对象的按位与操作
    def __rand__(self, other: _IntLike_co) -> integer[Any]: ...
    
    # 定义按位或操作符方法，用于当前对象和另一个 _IntLike_co 类型对象的按位或操作
    def __or__(self, other: _IntLike_co) -> integer[Any]: ...
    
    # 定义反向按位或操作符方法，用于另一个 _IntLike_co 类型对象和当前对象的按位或操作
    def __ror__(self, other: _IntLike_co) -> integer[Any]: ...
    
    # 定义按位异或操作符方法，用于当前对象和另一个 _IntLike_co 类型对象的按位异或操作
    def __xor__(self, other: _IntLike_co) -> integer[Any]: ...
    
    # 定义反向按位异或操作符方法，用于另一个 _IntLike_co 类型对象和当前对象的按位异或操作
    def __rxor__(self, other: _IntLike_co) -> integer[Any]: ...
# 定义一个名为 `signedinteger` 的类，继承自 `integer[_NBit1]` 类型
class signedinteger(integer[_NBit1]):
    
    # 初始化方法，接受一个 `_IntValue` 类型的参数 `value`，无返回值
    def __init__(self, value: _IntValue = ..., /) -> None: ...

    # 定义加法运算符重载方法，返回 `_SignedIntOp[_NBit1]` 类型
    __add__: _SignedIntOp[_NBit1]
    
    # 定义右加法运算符重载方法，返回 `_SignedIntOp[_NBit1]` 类型
    __radd__: _SignedIntOp[_NBit1]
    
    # 定义减法运算符重载方法，返回 `_SignedIntOp[_NBit1]` 类型
    __sub__: _SignedIntOp[_NBit1]
    
    # 定义右减法运算符重载方法，返回 `_SignedIntOp[_NBit1]` 类型
    __rsub__: _SignedIntOp[_NBit1]
    
    # 定义乘法运算符重载方法，返回 `_SignedIntOp[_NBit1]` 类型
    __mul__: _SignedIntOp[_NBit1]
    
    # 定义右乘法运算符重载方法，返回 `_SignedIntOp[_NBit1]` 类型
    __rmul__: _SignedIntOp[_NBit1]
    
    # 定义整除运算符重载方法，返回 `_SignedIntOp[_NBit1]` 类型
    __floordiv__: _SignedIntOp[_NBit1]
    
    # 定义右整除运算符重载方法，返回 `_SignedIntOp[_NBit1]` 类型
    __rfloordiv__: _SignedIntOp[_NBit1]
    
    # 定义幂运算符重载方法，返回 `_SignedIntOp[_NBit1]` 类型
    __pow__: _SignedIntOp[_NBit1]
    
    # 定义右幂运算符重载方法，返回 `_SignedIntOp[_NBit1]` 类型
    __rpow__: _SignedIntOp[_NBit1]
    
    # 定义左移位运算符重载方法，返回 `_SignedIntBitOp[_NBit1]` 类型
    __lshift__: _SignedIntBitOp[_NBit1]
    
    # 定义右左移位运算符重载方法，返回 `_SignedIntBitOp[_NBit1]` 类型
    __rlshift__: _SignedIntBitOp[_NBit1]
    
    # 定义右移位运算符重载方法，返回 `_SignedIntBitOp[_NBit1]` 类型
    __rshift__: _SignedIntBitOp[_NBit1]
    
    # 定义右右移位运算符重载方法，返回 `_SignedIntBitOp[_NBit1]` 类型
    __rrshift__: _SignedIntBitOp[_NBit1]
    
    # 定义按位与运算符重载方法，返回 `_SignedIntBitOp[_NBit1]` 类型
    __and__: _SignedIntBitOp[_NBit1]
    
    # 定义右按位与运算符重载方法，返回 `_SignedIntBitOp[_NBit1]` 类型
    __rand__: _SignedIntBitOp[_NBit1]
    
    # 定义按位异或运算符重载方法，返回 `_SignedIntBitOp[_NBit1]` 类型
    __xor__: _SignedIntBitOp[_NBit1]
    
    # 定义右按位异或运算符重载方法，返回 `_SignedIntBitOp[_NBit1]` 类型
    __rxor__: _SignedIntBitOp[_NBit1]
    
    # 定义按位或运算符重载方法，返回 `_SignedIntBitOp[_NBit1]` 类型
    __or__: _SignedIntBitOp[_NBit1]
    
    # 定义右按位或运算符重载方法，返回 `_SignedIntBitOp[_NBit1]` 类型
    __ror__: _SignedIntBitOp[_NBit1]
    
    # 定义取模运算符重载方法，返回 `_SignedIntMod[_NBit1]` 类型
    __mod__: _SignedIntMod[_NBit1]
    
    # 定义右取模运算符重载方法，返回 `_SignedIntMod[_NBit1]` 类型
    __rmod__: _SignedIntMod[_NBit1]
    
    # 定义整除和取模运算符重载方法，返回 `_SignedIntDivMod[_NBit1]` 类型
    __divmod__: _SignedIntDivMod[_NBit1]
    
    # 定义右整除和取模运算符重载方法，返回 `_SignedIntDivMod[_NBit1]` 类型
    __rdivmod__: _SignedIntDivMod[_NBit1]

# 将 `signedinteger` 类型实例化为 `int8`，表示一个8位有符号整数
int8 = signedinteger[_8Bit]

# 将 `signedinteger` 类型实例化为 `int16`，表示一个16位有符号整数
int16 = signedinteger[_16Bit]

# 将 `signedinteger` 类型实例化为 `int32`，表示一个32位有符号整数
int32 = signedinteger[_32Bit]

# 将 `signedinteger` 类型实例化为 `int64`，表示一个64位有符号整数
int64 = signedinteger[_64Bit]

# 将 `signedinteger` 类型实例化为 `byte`，表示一个字节大小的有符号整数
byte = signedinteger[_NBitByte]

# 将 `signedinteger` 类型实例化为 `short`，表示一个短整数大小的有符号整数
short = signedinteger[_NBitShort]

# 将 `signedinteger` 类型实例化为 `intc`，表示一个C语言整数大小的有符号整数
intc = signedinteger[_NBitIntC]

# 将 `signedinteger` 类型实例化为 `intp`，表示一个平台整数大小的有符号整数
intp = signedinteger[_NBitIntP]

# `int_` 是 `intp` 的别名，表示一个平台整数大小的有符号整数
int_ = intp

# 将 `signedinteger` 类型实例化为 `long`，表示一个长整数大小的有符号整数
long = signedinteger[_NBitLong]

# 将 `signedinteger` 类型实例化为 `longlong`，表示一个长长整数大小的有符号整数
longlong = signedinteger[_NBitLongLong]

# TODO: `item`/`tolist` 返回 `dt.timedelta` 或 `int`，具体取决于时间单位
# 根据不同的时间单位，实现了 `item`/`tolist` 方法的转换
class timedelta64(generic):
    
    # 初始化方法，接受多种类型的参数，无返回值
    def __init__(
        self,
        value: None | int | _CharLike_co | dt.timedelta | timedelta64 = ...,
        format: _CharLike_co | tuple[_CharLike_co, _IntLike_co] = ...,
        /,
    ) -> None: ...

    # 返回属性 `numerator`，类型为 `_ScalarType`
    @property
    def numerator(self: _ScalarType) -> _ScalarType: ...
    
    # 返回属性 `denominator`，固定为列表 `[1]`
    @property
    def denominator(self) -> L[1]: ...

    # NOTE: 只有少数单位支持转换为内建标量类型：`Y`, `M`, `ns`, `ps`, `fs`, `as`
    # 只有指定的单位支持转换为内建标量类型
    def __int__(self) -> int: ...
    
    # 转换为浮点数的特殊方法
    def __float__(self) -> float: ...
    
    # 转换为复数的特殊方法
    def __complex__(self) -> complex: ...
    
    # 取负数的特殊方法
    def __neg__(self: _ArraySelf) -> _ArraySelf: ...
    
    # 取正数的特殊方法
    def __pos__(self: _ArraySelf) -> _ArraySelf: ...
    
    # 取绝对值的特殊方法
    def __abs__(self: _ArraySelf) -> _ArraySelf: ...
    
    # 加法运算符重载方法，返回 `timedelta64` 类型
    # 定义特殊方法 __rdivmod__，用于实现 timedelta64 对象与其他对象的反向整除和取模操作，返回一个元组，包含整数和 timedelta64 类型的对象
    def __rdivmod__(self, other: timedelta64) -> tuple[int64, timedelta64]: ...
    
    # 定义特殊方法 __lt__，用于实现 timedelta64 对象与 _ArrayLikeTD64_co 类型对象的小于比较操作
    __lt__: _ComparisonOp[_TD64Like_co, _ArrayLikeTD64_co]
    
    # 定义特殊方法 __le__，用于实现 timedelta64 对象与 _ArrayLikeTD64_co 类型对象的小于等于比较操作
    __le__: _ComparisonOp[_TD64Like_co, _ArrayLikeTD64_co]
    
    # 定义特殊方法 __gt__，用于实现 timedelta64 对象与 _ArrayLikeTD64_co 类型对象的大于比较操作
    __gt__: _ComparisonOp[_TD64Like_co, _ArrayLikeTD64_co]
    
    # 定义特殊方法 __ge__，用于实现 timedelta64 对象与 _ArrayLikeTD64_co 类型对象的大于等于比较操作
    __ge__: _ComparisonOp[_TD64Like_co, _ArrayLikeTD64_co]
class unsignedinteger(integer[_NBit1]):
    # 定义一个无符号整数类，继承自integer[_NBit1]，表示N位1的整数
    # NOTE: `uint64 + signedinteger -> float64`
    # 定义了几种运算符重载方法，支持无符号整数的加减乘除等操作

    def __init__(self, value: _IntValue = ..., /) -> None:
        # 初始化方法，接受一个可选的_IntValue参数，无返回值
        ...

    __add__: _UnsignedIntOp[_NBit1]
    __radd__: _UnsignedIntOp[_NBit1]
    __sub__: _UnsignedIntOp[_NBit1]
    __rsub__: _UnsignedIntOp[_NBit1]
    __mul__: _UnsignedIntOp[_NBit1]
    __rmul__: _UnsignedIntOp[_NBit1]
    __floordiv__: _UnsignedIntOp[_NBit1]
    __rfloordiv__: _UnsignedIntOp[_NBit1]
    __pow__: _UnsignedIntOp[_NBit1]
    __rpow__: _UnsignedIntOp[_NBit1]
    __lshift__: _UnsignedIntBitOp[_NBit1]
    __rlshift__: _UnsignedIntBitOp[_NBit1]
    __rshift__: _UnsignedIntBitOp[_NBit1]
    __rrshift__: _UnsignedIntBitOp[_NBit1]
    __and__: _UnsignedIntBitOp[_NBit1]
    __rand__: _UnsignedIntBitOp[_NBit1]
    __xor__: _UnsignedIntBitOp[_NBit1]
    __rxor__: _UnsignedIntBitOp[_NBit1]
    __or__: _UnsignedIntBitOp[_NBit1]
    __ror__: _UnsignedIntBitOp[_NBit1]
    __mod__: _UnsignedIntMod[_NBit1]
    __rmod__: _UnsignedIntMod[_NBit1]
    __divmod__: _UnsignedIntDivMod[_NBit1]
    __rdivmod__: _UnsignedIntDivMod[_NBit1]
    # 定义了一系列运算符重载方法，用于无符号整数的各种位运算和算术运算

uint8 = unsignedinteger[_8Bit]
uint16 = unsignedinteger[_16Bit]
uint32 = unsignedinteger[_32Bit]
uint64 = unsignedinteger[_64Bit]
# 定义了几种具体位数的无符号整数类型，分别为8位、16位、32位和64位

ubyte = unsignedinteger[_NBitByte]
ushort = unsignedinteger[_NBitShort]
uintc = unsignedinteger[_NBitIntC]
uintp = unsignedinteger[_NBitIntP]
uint = uintp
ulong = unsignedinteger[_NBitLong]
ulonglong = unsignedinteger[_NBitLongLong]
# 定义了更多位数的无符号整数类型，以及为其定义的别名

class inexact(number[_NBit1]):  # type: ignore
    # 定义了一个不精确数类，继承自number[_NBit1]，表示N位1的数字

    def __getnewargs__(self: inexact[_64Bit]) -> tuple[float, ...]:
        # 返回一个包含浮点数的元组作为新参数的方法，返回类型为tuple[float, ...]
        ...

_IntType = TypeVar("_IntType", bound=integer[Any])
_FloatType = TypeVar('_FloatType', bound=floating[Any])

class floating(inexact[_NBit1]):
    # 定义了一个浮点数类，继承自inexact[_NBit1]，表示N位1的浮点数

    def __init__(self, value: _FloatValue = ..., /) -> None:
        # 初始化方法，接受一个可选的_FloatValue参数，无返回值
        ...

    def item(
        self, args: L[0] | tuple[()] | tuple[L[0]] = ...,
        /,
    ) -> float:
        # 返回一个浮点数的方法，接受参数args，并返回类型为float

    def tolist(self) -> float:
        # 将浮点数转换为列表的方法，返回类型为float

    def is_integer(self) -> builtins.bool:
        # 判断浮点数是否为整数的方法，返回类型为bool

    def hex(self: float64) -> str:
        # 返回浮点数的十六进制表示的方法，接受类型为float64，返回类型为str

    @classmethod
    def fromhex(cls: type[float64], string: str, /) -> float64:
        # 从十六进制字符串创建浮点数的类方法，接受类型为float64和str，返回类型为float64

    def as_integer_ratio(self) -> tuple[int, int]:
        # 返回浮点数的整数比例的方法，返回类型为tuple[int, int]

    def __ceil__(self: float64) -> int:
        # 返回浮点数的上限整数的方法，接受类型为float64，返回类型为int

    def __floor__(self: float64) -> int:
        # 返回浮点数的下限整数的方法，接受类型为float64，返回类型为int

    def __trunc__(self: float64) -> int:
        # 返回浮点数的截断整数的方法，接受类型为float64，返回类型为int

    def __getnewargs__(self: float64) -> tuple[float]:
        # 返回一个包含浮点数的元组作为新参数的方法，接受类型为float64，返回类型为tuple[float]

    def __getformat__(self: float64, typestr: L["double", "float"], /) -> str:
        # 返回浮点数格式的方法，接受类型为float64和L["double", "float"]，返回类型为str

    @overload
    def __round__(self, ndigits: None = ...) -> int:
        # 对浮点数进行四舍五入的方法，接受可选的ndigits参数，返回类型为int

    @overload
    def __round__(self, ndigits: SupportsIndex) -> _ScalarType:
        # 对浮点数进行指定小数位数的四舍五入的方法，接受类型为SupportsIndex，返回类型为_ScalarType

    __add__: _FloatOp[_NBit1]
    __radd__: _FloatOp[_NBit1]
    __sub__: _FloatOp[_NBit1]
    __rsub__: _FloatOp[_NBit1]
    __mul__: _FloatOp[_NBit1]
    __rmul__: _FloatOp[_NBit1]
    __truediv__: _FloatOp[_NBit1]
    __rtruediv__: _FloatOp[_NBit1]
    __floordiv__: _FloatOp[_NBit1]
    __rfloordiv__: _FloatOp[_NBit1]
    __pow__: _FloatOp[_NBit1]
    __rpow__: _FloatOp[_NBit1]
    # 定义了一系列运算符重载方法，用于浮点数的各种算术运算
    # 定义魔术方法 __mod__，处理浮点数的取模运算，返回结果为 _FloatMod[_NBit1] 类型
    __mod__: _FloatMod[_NBit1]
    
    # 定义反向魔术方法 __rmod__，处理浮点数的反向取模运算，返回结果为 _FloatMod[_NBit1] 类型
    __rmod__: _FloatMod[_NBit1]
    
    # 定义魔术方法 __divmod__，处理浮点数的除法取模运算，返回结果为 _FloatDivMod[_NBit1] 类型
    __divmod__: _FloatDivMod[_NBit1]
    
    # 定义反向魔术方法 __rdivmod__，处理浮点数的反向除法取模运算，返回结果为 _FloatDivMod[_NBit1] 类型
    __rdivmod__: _FloatDivMod[_NBit1]
float16 = floating[_16Bit]
float32 = floating[_32Bit]
float64 = floating[_64Bit]

half = floating[_NBitHalf]
single = floating[_NBitSingle]
double = floating[_NBitDouble]
longdouble = floating[_NBitLongDouble]

# `complexfloating` 类型有两个类型变量，主要是为了清晰表达 `complex128` 的精度是 `_64Bit`，
# 后者描述了其实部和虚部所用的两个64位浮点数

class complexfloating(inexact[_NBit1], Generic[_NBit1, _NBit2]):
    def __init__(self, value: _ComplexValue = ..., /) -> None: ...
    def item(
        self, args: L[0] | tuple[()] | tuple[L[0]] = ..., /,
    ) -> complex: ...
    def tolist(self) -> complex: ...
    @property
    def real(self) -> floating[_NBit1]: ...  # type: ignore[override]
    @property
    def imag(self) -> floating[_NBit2]: ...  # type: ignore[override]
    def __abs__(self) -> floating[_NBit1]: ...  # type: ignore[override]
    def __getnewargs__(self: complex128) -> tuple[float, float]: ...
    # 注意：已废弃
    # def __round__(self, ndigits=...): ...
    __add__: _ComplexOp[_NBit1]
    __radd__: _ComplexOp[_NBit1]
    __sub__: _ComplexOp[_NBit1]
    __rsub__: _ComplexOp[_NBit1]
    __mul__: _ComplexOp[_NBit1]
    __rmul__: _ComplexOp[_NBit1]
    __truediv__: _ComplexOp[_NBit1]
    __rtruediv__: _ComplexOp[_NBit1]
    __pow__: _ComplexOp[_NBit1]
    __rpow__: _ComplexOp[_NBit1]

complex64 = complexfloating[_32Bit, _32Bit]
complex128 = complexfloating[_64Bit, _64Bit]

csingle = complexfloating[_NBitSingle, _NBitSingle]
cdouble = complexfloating[_NBitDouble, _NBitDouble]
clongdouble = complexfloating[_NBitLongDouble, _NBitLongDouble]

class flexible(generic): ...  # type: ignore

# TODO: `item`/`tolist` 返回 `bytes` 或 `tuple`，取决于其作为不透明字节序列还是结构体的使用方式
class void(flexible):
    @overload
    def __init__(self, value: _IntLike_co | bytes, /, dtype : None = ...) -> None: ...
    @overload
    def __init__(self, value: Any, /, dtype: _DTypeLikeVoid) -> None: ...
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...
    def setfield(
        self, val: ArrayLike, dtype: DTypeLike, offset: int = ...
    ) -> None: ...
    @overload
    def __getitem__(self, key: str | SupportsIndex) -> Any: ...
    @overload
    def __getitem__(self, key: list[str]) -> void: ...
    def __setitem__(
        self,
        key: str | list[str] | SupportsIndex,
        value: ArrayLike,
    ) -> None: ...

class character(flexible):  # type: ignore
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...

# 注意：大多数 `np.bytes_` / `np.str_` 方法返回其内建的 `bytes` / `str` 对应物

class bytes_(character, bytes):
    @overload
    def __init__(self, value: object = ..., /) -> None: ...
    @overload
    # 初始化方法，用于创建对象实例
    def __init__(
        self, value: str, /, encoding: str = ..., errors: str = ...
    ) -> None: ...
    
    # item 方法，用于返回字节数据
    def item(
        self, args: L[0] | tuple[()] | tuple[L[0]] = ..., /,
    ) -> bytes: ...
    
    # tolist 方法，用于返回字节数据
    def tolist(self) -> bytes: ...
# 定义一个名为 `str_` 的类，继承自 `character` 和 `str`
class str_(character, str):
    # 重载方法：初始化函数，接受一个对象值参数，无返回值
    @overload
    def __init__(self, value: object = ..., /) -> None: ...
    # 重载方法：初始化函数，接受一个字节串和编码参数，默认错误处理方式，无返回值
    @overload
    def __init__(
        self, value: bytes, /, encoding: str = ..., errors: str = ...
    ) -> None: ...
    # 方法 `item`，接受一个参数列表或空元组或单一参数的元组，返回字符串
    def item(
        self, args: L[0] | tuple[()] | tuple[L[0]] = ..., /,
    ) -> str: ...
    # 方法 `tolist`，返回字符串
    def tolist(self) -> str: ...

#
# 常量定义
#

# 自然常数 e
e: Final[float]
# 欧拉常数 γ
euler_gamma: Final[float]
# 无穷大
inf: Final[float]
# 非数值
nan: Final[float]
# 圆周率 π
pi: Final[float]

# 小端序标识
little_endian: Final[builtins.bool]
# 真值常量 True
True_: Final[np.bool]
# 假值常量 False
False_: Final[np.bool]

# 新轴常量
newaxis: None

# 更多具体的 nin-/nout- 特定存根参见 `numpy._typing._ufunc`
@final
class ufunc:
    # 属性 `__name__`，返回字符串
    @property
    def __name__(self) -> str: ...
    # 属性 `__doc__`，返回字符串
    @property
    def __doc__(self) -> str: ...
    # 方法 `__call__`，可调用对象，接受任意参数，返回任意类型
    __call__: Callable[..., Any]
    # 属性 `nin`，输入参数数量，返回整数
    @property
    def nin(self) -> int: ...
    # 属性 `nout`，输出参数数量，返回整数
    @property
    def nout(self) -> int: ...
    # 属性 `nargs`，参数数量，返回整数
    @property
    def nargs(self) -> int: ...
    # 属性 `ntypes`，类型数量，返回整数
    @property
    def ntypes(self) -> int: ...
    # 属性 `types`，类型列表，返回字符串列表
    @property
    def types(self) -> list[str]: ...
    # 属性 `identity`，标识性质，返回任意类型
    # 对于 `numpy` 中的 `ufunc`，返回值可以是 `True`、`0`、`None` 等
    @property
    def identity(self) -> Any: ...
    # 属性 `signature`，签名，返回字符串或 `None`
    # 对于 `ufunc` 是 `None`，对于 `gufunc` 是字符串
    @property
    def signature(self) -> None | str: ...
    # 方法 `reduce`、`accumulate`、`reduceat`、`outer`、`at`
    # 这些方法对于多输出的 `ufunc` 不会被定义，具体返回值类型不确定
    reduce: Any
    accumulate: Any
    reduceat: Any
    outer: Any
    at: Any

# 参数定义：`__name__`、`ntypes` 和 `identity`
absolute: _UFunc_Nin1_Nout1[L['absolute'], L[20], None]
add: _UFunc_Nin2_Nout1[L['add'], L[22], L[0]]
arccos: _UFunc_Nin1_Nout1[L['arccos'], L[8], None]
arccosh: _UFunc_Nin1_Nout1[L['arccosh'], L[8], None]
arcsin: _UFunc_Nin1_Nout1[L['arcsin'], L[8], None]
arcsinh: _UFunc_Nin1_Nout1[L['arcsinh'], L[8], None]
arctan2: _UFunc_Nin2_Nout1[L['arctan2'], L[5], None]
arctan: _UFunc_Nin1_Nout1[L['arctan'], L[8], None]
arctanh: _UFunc_Nin1_Nout1[L['arctanh'], L[8], None]
bitwise_and: _UFunc_Nin2_Nout1[L['bitwise_and'], L[12], L[-1]]
bitwise_count: _UFunc_Nin1_Nout1[L['bitwise_count'], L[11], None]
bitwise_not: _UFunc_Nin1_Nout1[L['invert'], L[12], None]
bitwise_or: _UFunc_Nin2_Nout1[L['bitwise_or'], L[12], L[0]]
bitwise_xor: _UFunc_Nin2_Nout1[L['bitwise_xor'], L[12], L[0]]
cbrt: _UFunc_Nin1_Nout1[L['cbrt'], L[5], None]
ceil: _UFunc_Nin1_Nout1[L['ceil'], L[7], None]
conj: _UFunc_Nin1_Nout1[L['conjugate'], L[18], None]
conjugate: _UFunc_Nin1_Nout1[L['conjugate'], L[18], None]
copysign: _UFunc_Nin2_Nout1[L['copysign'], L[4], None]
cos: _UFunc_Nin1_Nout1[L['cos'], L[9], None]
cosh: _UFunc_Nin1_Nout1[L['cosh'], L[8], None]
# 定义一个双曲余弦函数，接受一个输入参数，返回一个输出参数

deg2rad: _UFunc_Nin1_Nout1[L['deg2rad'], L[5], None]
# 将角度转换为弧度的函数，接受一个输入参数，返回一个输出参数

degrees: _UFunc_Nin1_Nout1[L['degrees'], L[5], None]
# 将弧度转换为角度的函数，接受一个输入参数，返回一个输出参数

divide: _UFunc_Nin2_Nout1[L['true_divide'], L[11], None]
# 执行两个数的真实除法运算，接受两个输入参数，返回一个输出参数

divmod: _UFunc_Nin2_Nout2[L['divmod'], L[15], None]
# 计算两个数的商和余数，接受两个输入参数，返回两个输出参数

equal: _UFunc_Nin2_Nout1[L['equal'], L[23], None]
# 比较两个数是否相等，接受两个输入参数，返回一个输出参数

exp2: _UFunc_Nin1_Nout1[L['exp2'], L[8], None]
# 计算2的指数幂，接受一个输入参数，返回一个输出参数

exp: _UFunc_Nin1_Nout1[L['exp'], L[10], None]
# 计算自然指数的幂，接受一个输入参数，返回一个输出参数

expm1: _UFunc_Nin1_Nout1[L['expm1'], L[8], None]
# 计算自然指数的幂减去1，接受一个输入参数，返回一个输出参数

fabs: _UFunc_Nin1_Nout1[L['fabs'], L[5], None]
# 计算浮点数的绝对值，接受一个输入参数，返回一个输出参数

float_power: _UFunc_Nin2_Nout1[L['float_power'], L[4], None]
# 计算一个数的幂次方，接受两个输入参数，返回一个输出参数

floor: _UFunc_Nin1_Nout1[L['floor'], L[7], None]
# 计算浮点数的下限（向下取整），接受一个输入参数，返回一个输出参数

floor_divide: _UFunc_Nin2_Nout1[L['floor_divide'], L[21], None]
# 执行两个数的地板除法运算，接受两个输入参数，返回一个输出参数

fmax: _UFunc_Nin2_Nout1[L['fmax'], L[21], None]
# 返回两个数的最大值，接受两个输入参数，返回一个输出参数

fmin: _UFunc_Nin2_Nout1[L['fmin'], L[21], None]
# 返回两个数的最小值，接受两个输入参数，返回一个输出参数

fmod: _UFunc_Nin2_Nout1[L['fmod'], L[15], None]
# 计算两个数的浮点余数，接受两个输入参数，返回一个输出参数

frexp: _UFunc_Nin1_Nout2[L['frexp'], L[4], None]
# 将一个浮点数分解为尾数和指数，接受一个输入参数，返回两个输出参数

gcd: _UFunc_Nin2_Nout1[L['gcd'], L[11], L[0]]
# 计算两个整数的最大公约数，接受两个输入参数，返回一个输出参数

greater: _UFunc_Nin2_Nout1[L['greater'], L[23], None]
# 比较两个数是否大于，接受两个输入参数，返回一个输出参数

greater_equal: _UFunc_Nin2_Nout1[L['greater_equal'], L[23], None]
# 比较两个数是否大于等于，接受两个输入参数，返回一个输出参数

heaviside: _UFunc_Nin2_Nout1[L['heaviside'], L[4], None]
# 计算海维赛德函数，接受两个输入参数，返回一个输出参数

hypot: _UFunc_Nin2_Nout1[L['hypot'], L[5], L[0]]
# 计算两个数的欧几里得范数，接受两个输入参数，返回一个输出参数

invert: _UFunc_Nin1_Nout1[L['invert'], L[12], None]
# 计算整数的按位取反，接受一个输入参数，返回一个输出参数

isfinite: _UFunc_Nin1_Nout1[L['isfinite'], L[20], None]
# 检查浮点数是否为有限数，接受一个输入参数，返回一个输出参数

isinf: _UFunc_Nin1_Nout1[L['isinf'], L[20], None]
# 检查浮点数是否为无穷大，接受一个输入参数，返回一个输出参数

isnan: _UFunc_Nin1_Nout1[L['isnan'], L[20], None]
# 检查浮点数是否为NaN（非数值），接受一个输入参数，返回一个输出参数

isnat: _UFunc_Nin1_Nout1[L['isnat'], L[2], None]
# 检查日期时间是否为自然日期时间，接受一个输入参数，返回一个输出参数

lcm: _UFunc_Nin2_Nout1[L['lcm'], L[11], None]
# 计算两个整数的最小公倍数，接受两个输入参数，返回一个输出参数

ldexp: _UFunc_Nin2_Nout1[L['ldexp'], L[8], None]
# 计算x乘以2的y次幂，接受两个输入参数，返回一个输出参数

left_shift: _UFunc_Nin2_Nout1[L['left_shift'], L[11], None]
# 执行整数的左移位运算，接受两个输入参数，返回一个输出参数

less: _UFunc_Nin2_Nout1[L['less'], L[23], None]
# 比较两个数是否小于，接受两个输入参数，返回一个输出参数

less_equal: _UFunc_Nin2_Nout1[L['less_equal'], L[23], None]
# 比较两个数是否小于等于，接受两个输入参数，返回一个输出参数

log10: _UFunc_Nin1_Nout1[L['log10'], L[8], None]
# 计算以10为底的对数，接受一个输入参数，返回一个输出参数

log1p: _UFunc_Nin1_Nout1[L['log1p'], L[8], None]
# 计算1加上自然对数的值，接受一个输入参数，返回一个输出参数

log2: _UFunc_Nin1_Nout1[L['log2'], L[8], None]
# 计算以2为底的对数，接受一个输入参数，返回一个输出参数

log: _UFunc_Nin1_Nout1[L['log'], L[10], None]
# 计算自然对数，接受一个输入参数，返回一个输出参数

logaddexp2: _UFunc_Nin2_Nout1[L['logaddexp2'], L[4], float]
# 计算以2为底的指数相加，接受两个输入参数，返回一个输出参数

logaddexp: _UFunc_Nin2_Nout1[L['logaddexp'], L[4], float]
# 计算自然对数的指数相加，接受两个输入参数，返回一个输出参数

logical_and: _UFunc_Nin2_Nout1[L['logical_and'], L[20], L[True]]
# 执行逻辑与运算，接受两个输入参数，返回一个输出参数

logical_not: _UFunc_Nin1_Nout1[L['logical_not'], L[20], None]
# 执行逻辑非运算，接受一个输入参数，返回一个输出参数

logical_or: _UFunc_Nin2_Nout1[L['logical_or'], L[20], L[False]]
# 执行逻辑或运算，接受两个输入参数，返回一个输出参数

logical_xor: _UFunc_Nin2_Nout1[L['logical_xor'], L[19], L[False]]
# 执行
# 计算倒数的通用函数，输入一个数组返回对应位置的倒数值
reciprocal: _UFunc_Nin1_Nout1[L['reciprocal'], L[18], None]
# 计算两个数相除的余数，输入两个数组返回对应位置的余数值
remainder: _UFunc_Nin2_Nout1[L['remainder'], L[16], None]
# 逐位右移操作的通用函数，输入两个数组返回对应位置的右移结果
right_shift: _UFunc_Nin2_Nout1[L['right_shift'], L[11], None]
# 将数组中每个元素四舍五入到最近的整数
rint: _UFunc_Nin1_Nout1[L['rint'], L[10], None]
# 返回每个元素的符号，正数返回 1，负数返回 -1，0 返回 0
sign: _UFunc_Nin1_Nout1[L['sign'], L[19], None]
# 返回每个元素是否为负数的布尔值
signbit: _UFunc_Nin1_Nout1[L['signbit'], L[4], None]
# 计算每个元素的正弦值
sin: _UFunc_Nin1_Nout1[L['sin'], L[9], None]
# 计算每个元素的双曲正弦值
sinh: _UFunc_Nin1_Nout1[L['sinh'], L[8], None]
# 计算每个元素的距离到下一个浮点数的绝对距离
spacing: _UFunc_Nin1_Nout1[L['spacing'], L[4], None]
# 计算每个元素的平方根
sqrt: _UFunc_Nin1_Nout1[L['sqrt'], L[10], None]
# 计算每个元素的平方
square: _UFunc_Nin1_Nout1[L['square'], L[18], None]
# 计算两个数组对应位置元素相减的结果
subtract: _UFunc_Nin2_Nout1[L['subtract'], L[21], None]
# 计算每个元素的正切值
tan: _UFunc_Nin1_Nout1[L['tan'], L[8], None]
# 计算每个元素的双曲正切值
tanh: _UFunc_Nin1_Nout1[L['tanh'], L[8], None]
# 计算两个数组对应位置元素相除的结果
true_divide: _UFunc_Nin2_Nout1[L['true_divide'], L[11], None]
# 计算每个元素的截断整数部分
trunc: _UFunc_Nin1_Nout1[L['trunc'], L[7], None]
# 计算两个向量的点积
vecdot: _GUFunc_Nin2_Nout1[L['vecdot'], L[19], None, L["(n),(n)->()"]]

# 将 abs 定义为 absolute 的别名
abs = absolute
# 将 acos 定义为 arccos 的别名
acos = arccos
# 将 acosh 定义为 arccosh 的别名
acosh = arccosh
# 将 asin 定义为 arcsin 的别名
asin = arcsin
# 将 asinh 定义为 arcsinh 的别名
asinh = arcsinh
# 将 atan 定义为 arctan 的别名
atan = arctan
# 将 atanh 定义为 arctanh 的别名
atanh = arctanh
# 将 atan2 定义为 arctan2 的别名
atan2 = arctan2
# 将 concat 定义为 concatenate 的别名
concat = concatenate
# 将 bitwise_left_shift 定义为 left_shift 的别名
bitwise_left_shift = left_shift
# 将 bitwise_invert 定义为 invert 的别名
bitwise_invert = invert
# 将 bitwise_right_shift 定义为 right_shift 的别名
bitwise_right_shift = right_shift
# 将 permute_dims 定义为 transpose 的别名
permute_dims = transpose
# 将 pow 定义为 power 的别名
pow = power

# 定义枚举 _CopyMode，包含三个成员常量：ALWAYS、IF_NEEDED 和 NEVER
class _CopyMode(enum.Enum):
    ALWAYS: L[True]
    IF_NEEDED: L[False]
    NEVER: L[2]

# 定义泛型 _CallType，表示一种可调用类型的范型
_CallType = TypeVar("_CallType", bound=Callable[..., Any])

# 定义 errstate 上下文管理器类，用于管理特定的错误状态
class errstate:
    def __init__(
        self,
        *,
        call: _ErrFunc | _SupportsWrite[str] = ...,
        all: None | _ErrKind = ...,
        divide: None | _ErrKind = ...,
        over: None | _ErrKind = ...,
        under: None | _ErrKind = ...,
        invalid: None | _ErrKind = ...,
    ) -> None: ...
    # 进入上下文管理器时调用的方法
    def __enter__(self) -> None: ...
    # 退出上下文管理器时调用的方法
    def __exit__(
        self,
        exc_type: None | type[BaseException],
        exc_value: None | BaseException,
        traceback: None | TracebackType,
        /,
    ) -> None: ...
    # 调用 errstate 实例时调用的方法
    def __call__(self, func: _CallType) -> _CallType: ...

# 定义 _no_nep50_warning 上下文管理器函数，用于在上下文中禁止特定的警告
@contextmanager
def _no_nep50_warning() -> Generator[None, None, None]: ...
# 获取当前推广状态的函数
def _get_promotion_state() -> str: ...
# 设置推广状态的函数
def _set_promotion_state(state: str, /) -> None: ...

# 定义泛型类 ndenumerate，用于在数组中枚举元素及其索引
class ndenumerate(Generic[_ScalarType]):
    iter: flatiter[NDArray[_ScalarType]]
    @overload
    def __new__(
        cls, arr: _FiniteNestedSequence[_SupportsArray[dtype[_ScalarType]]],
    ) -> ndenumerate[_ScalarType]: ...
    @overload
    def __new__(cls, arr: str | _NestedSequence[str]) -> ndenumerate[str_]: ...
    @overload
    def __new__(cls, arr: bytes | _NestedSequence[bytes]) -> ndenumerate[bytes_]: ...
    @overload
    def __new__(cls, arr: builtins.bool | _NestedSequence[builtins.bool]) -> ndenumerate[np.bool]: ...
    @overload
    def __new__(cls, arr: int | _NestedSequence[int]) -> ndenumerate[int_]: ...
    @overload
    def __new__(cls, arr: float | _NestedSequence[float]) -> ndenumerate[float64]: ...
    @overload
    def __new__(cls, arr: complex | _NestedSequence[complex]) -> ndenumerate[complex128]: ...
    # 定义一个方法 `__next__`，接受一个类型为 `ndenumerate[_ScalarType]` 的参数 `self`，返回一个由 `_Shape` 和 `_ScalarType` 组成的元组
    def __next__(self: ndenumerate[_ScalarType]) -> tuple[_Shape, _ScalarType]: ...
    
    # 定义一个方法 `__iter__`，接受一个类型为 `_T` 的参数 `self`，返回一个 `_T` 类型的迭代器对象
    def __iter__(self: _T) -> _T: ...
class ndindex:
    # 定义了一个名为 `ndindex` 的类，用于处理多维索引

    @overload
    def __init__(self, shape: tuple[SupportsIndex, ...], /) -> None:
        # 类的初始化方法，支持接受一个元组形式的 shape 参数

    @overload
    def __init__(self, *shape: SupportsIndex) -> None:
        # 类的初始化方法，支持接受多个支持索引的参数作为 shape

    def __iter__(self: _T) -> _T:
        # 定义了一个迭代器方法 `__iter__`，返回类型为 `_T`

    def __next__(self) -> _Shape:
        # 定义了一个下一个迭代元素的方法 `__next__`，返回类型为 `_Shape`

# TODO: The type of each `__next__` and `iters` return-type depends
# on the length and dtype of `args`; we can't describe this behavior yet
# as we lack variadics (PEP 646).

@final
class broadcast:
    # 定义了一个名为 `broadcast` 的最终类，用于广播操作

    def __new__(cls, *args: ArrayLike) -> broadcast:
        # 类的特殊方法 `__new__`，用于创建实例，接受多个 `ArrayLike` 类型的参数

    @property
    def index(self) -> int:
        # 返回当前广播操作的索引

    @property
    def iters(self) -> tuple[flatiter[Any], ...]:
        # 返回一个元组，包含广播操作的迭代器

    @property
    def nd(self) -> int:
        # 返回广播操作的维度数

    @property
    def ndim(self) -> int:
        # 返回广播操作的维度数

    @property
    def numiter(self) -> int:
        # 返回广播操作的迭代器数量

    @property
    def shape(self) -> _Shape:
        # 返回广播操作的形状

    @property
    def size(self) -> int:
        # 返回广播操作的大小

    def __next__(self) -> tuple[Any, ...]:
        # 定义了下一个迭代元素的方法 `__next__`，返回类型为元组 `tuple[Any, ...]`

    def __iter__(self: _T) -> _T:
        # 定义了一个迭代器方法 `__iter__`，返回类型为 `_T`

    def reset(self) -> None:
        # 重置广播操作的状态

@final
class busdaycalendar:
    # 定义了一个名为 `busdaycalendar` 的最终类，处理工作日历

    def __new__(
        cls,
        weekmask: ArrayLike = ...,
        holidays: ArrayLike | dt.date | _NestedSequence[dt.date] = ...,
    ) -> busdaycalendar:
        # 类的特殊方法 `__new__`，用于创建实例，接受参数 `weekmask` 和 `holidays`

    @property
    def weekmask(self) -> NDArray[np.bool]:
        # 返回工作日掩码

    @property
    def holidays(self) -> NDArray[datetime64]:
        # 返回节假日数组

class finfo(Generic[_FloatType]):
    # 泛型类 `finfo`，处理浮点数信息

    dtype: dtype[_FloatType]
    bits: int
    eps: _FloatType
    epsneg: _FloatType
    iexp: int
    machep: int
    max: _FloatType
    maxexp: int
    min: _FloatType
    minexp: int
    negep: int
    nexp: int
    nmant: int
    precision: int
    resolution: _FloatType
    smallest_subnormal: _FloatType

    @property
    def smallest_normal(self) -> _FloatType:
        # 返回最小正规化浮点数

    @property
    def tiny(self) -> _FloatType:
        # 返回一个极小值，表示浮点数的精度

    @overload
    def __new__(
        cls, dtype: inexact[_NBit1] | _DTypeLike[inexact[_NBit1]]
    ) -> finfo[floating[_NBit1]]:
        # 类的特殊方法 `__new__`，支持接受不精确类型的参数

    @overload
    def __new__(
        cls, dtype: complex | float | type[complex] | type[float]
    ) -> finfo[float64]:
        # 类的特殊方法 `__new__`，支持接受复数或浮点数类型的参数

    @overload
    def __new__(
        cls, dtype: str
    ) -> finfo[floating[Any]]:
        # 类的特殊方法 `__new__`，支持接受字符串类型的参数

class iinfo(Generic[_IntType]):
    # 泛型类 `iinfo`，处理整数信息

    dtype: dtype[_IntType]
    kind: str
    bits: int
    key: str

    @property
    def min(self) -> int:
        # 返回最小整数值

    @property
    def max(self) -> int:
        # 返回最大整数值

    @overload
    def __new__(cls, dtype: _IntType | _DTypeLike[_IntType]) -> iinfo[_IntType]:
        # 类的特殊方法 `__new__`，支持接受整数或整数类型的参数

    @overload
    def __new__(cls, dtype: int | type[int]) -> iinfo[int_]:
        # 类的特殊方法 `__new__`，支持接受整数或整数类型的参数

    @overload
    def __new__(cls, dtype: str) -> iinfo[Any]:
        # 类的特殊方法 `__new__`，支持接受字符串类型的参数

_NDIterFlagsKind = L[
    "buffered",
    "c_index",
    "copy_if_overlap",
    "common_dtype",
    "delay_bufalloc",
    "external_loop",
    "f_index",
    "grow_inner", "growinner",
    "multi_index",
    "ranged",
    "refs_ok",
    "reduce_ok",
    "zerosize_ok",
]

_NDIterOpFlagsKind = L[
    "aligned",
    "allocate",
    "arraymask",
    "copy",
    "config",
    "nbo",
    "no_subtype",
    "no_broadcast",
    "overlap_assume_elementwise",
]
    "readonly",      # 字符串 "readonly"
    "readwrite",     # 字符串 "readwrite"
    "updateifcopy",  # 字符串 "updateifcopy"
    "virtual",       # 字符串 "virtual"
    "writeonly",     # 字符串 "writeonly"
    "writemasked"    # 字符串 "writemasked"
# 引入类型提示的相关模块
]

# 使用 @final 装饰器标记类 nditer 为终态，禁止继承
@final
# 定义 nditer 类，用于迭代操作
class nditer:
    # __new__ 方法用于创建新的 nditer 实例，接受多个参数，返回 nditer 对象
    def __new__(
        cls,
        op: ArrayLike | Sequence[ArrayLike],
        flags: None | Sequence[_NDIterFlagsKind] = ...,
        op_flags: None | Sequence[Sequence[_NDIterOpFlagsKind]] = ...,
        op_dtypes: DTypeLike | Sequence[DTypeLike] = ...,
        order: _OrderKACF = ...,
        casting: _CastingKind = ...,
        op_axes: None | Sequence[Sequence[SupportsIndex]] = ...,
        itershape: None | _ShapeLike = ...,
        buffersize: SupportsIndex = ...,
    ) -> nditer: ...
    
    # __enter__ 方法定义，支持上下文管理器进入操作，返回 nditer 对象
    def __enter__(self) -> nditer: ...
    
    # __exit__ 方法定义，支持上下文管理器退出操作，接受异常类型、异常值和回溯信息
    def __exit__(
        self,
        exc_type: None | type[BaseException],
        exc_value: None | BaseException,
        traceback: None | TracebackType,
    ) -> None: ...
    
    # __iter__ 方法定义，返回迭代器自身，支持迭代操作
    def __iter__(self) -> nditer: ...
    
    # __next__ 方法定义，返回迭代器的下一个元素，返回类型为元组
    def __next__(self) -> tuple[NDArray[Any], ...]: ...
    
    # __len__ 方法定义，返回迭代器的长度，即迭代器包含的元素数量
    def __len__(self) -> int: ...
    
    # __copy__ 方法定义，返回当前迭代器的浅拷贝
    def __copy__(self) -> nditer: ...
    
    # __getitem__ 方法重载，接受支持的索引类型，返回对应的数组元素
    @overload
    def __getitem__(self, index: SupportsIndex) -> NDArray[Any]: ...
    
    # __getitem__ 方法重载，接受切片作为索引，返回元组形式的多个数组元素
    @overload
    def __getitem__(self, index: slice) -> tuple[NDArray[Any], ...]: ...
    
    # __setitem__ 方法定义，接受支持的索引类型和数组样式的值，用于设置迭代器的元素值
    def __setitem__(self, index: slice | SupportsIndex, value: ArrayLike) -> None: ...
    
    # close 方法定义，关闭迭代器，释放资源
    def close(self) -> None: ...
    
    # copy 方法定义，返回当前迭代器的副本
    def copy(self) -> nditer: ...
    
    # debug_print 方法定义，用于调试打印迭代器相关信息
    def debug_print(self) -> None: ...
    
    # enable_external_loop 方法定义，启用外部循环模式
    def enable_external_loop(self) -> None: ...
    
    # iternext 方法定义，迭代器指向下一个元素，返回布尔值表示是否成功
    def iternext(self) -> builtins.bool: ...
    
    # remove_axis 方法定义，移除指定轴
    def remove_axis(self, i: SupportsIndex, /) -> None: ...
    
    # remove_multi_index 方法定义，移除多重索引
    def remove_multi_index(self) -> None: ...
    
    # reset 方法定义，重置迭代器状态
    def reset(self) -> None: ...
    
    # dtypes 属性定义，返回迭代器内部数组的数据类型元组
    @property
    def dtypes(self) -> tuple[dtype[Any], ...]: ...
    
    # finished 属性定义，返回布尔值，表示迭代是否已完成
    @property
    def finished(self) -> builtins.bool: ...
    
    # has_delayed_bufalloc 属性定义，返回布尔值，表示是否存在延迟的缓冲分配
    @property
    def has_delayed_bufalloc(self) -> builtins.bool: ...
    
    # has_index 属性定义，返回布尔值，表示迭代器是否有索引
    @property
    def has_index(self) -> builtins.bool: ...
    
    # has_multi_index 属性定义，返回布尔值，表示迭代器是否有多重索引
    @property
    def has_multi_index(self) -> builtins.bool: ...
    
    # index 属性定义，返回整数，表示当前索引值
    @property
    def index(self) -> int: ...
    
    # iterationneedsapi 属性定义，返回布尔值，表示迭代器是否需要 API
    @property
    def iterationneedsapi(self) -> builtins.bool: ...
    
    # iterindex 属性定义，返回整数，表示当前迭代器的索引值
    @property
    def iterindex(self) -> int: ...
    
    # iterrange 属性定义，返回元组，表示迭代器的范围
    @property
    def iterrange(self) -> tuple[int, ...]: ...
    
    # itersize 属性定义，返回整数，表示迭代器的大小
    @property
    def itersize(self) -> int: ...
    
    # itviews 属性定义，返回元组，包含迭代器内部的数组视图
    @property
    def itviews(self) -> tuple[NDArray[Any], ...]: ...
    
    # multi_index 属性定义，返回元组，表示当前的多重索引
    @property
    def multi_index(self) -> tuple[int, ...]: ...
    
    # ndim 属性定义，返回整数，表示迭代器中数组的维度数
    @property
    def ndim(self) -> int: ...
    
    # nop 属性定义，返回整数，表示迭代器的操作数数量
    @property
    def nop(self) -> int: ...
    
    # operands 属性定义，返回元组，表示迭代器的操作数数组
    @property
    def operands(self) -> tuple[NDArray[Any], ...]: ...
    
    # shape 属性定义，返回元组，表示迭代器中数组的形状
    @property
    def shape(self) -> tuple[int, ...]: ...
    
    # value 属性定义，返回元组，表示迭代器当前位置的值
    @property
    def value(self) -> tuple[NDArray[Any], ...]: ...

# _MemMapModeKind 类型别名定义，表示 memmap 类的模式列表
_MemMapModeKind = L[
    "readonly", "r",
    "copyonwrite", "c",
    "readwrite", "r+",
    "write", "w+",
]

# memmap 类定义，继承自 ndarray 类，表示内存映射数组
class memmap(ndarray[_ShapeType, _DType_co]):
    # __array_priority__ 类变量定义，表示数组优先级
    __array_priority__: ClassVar[float]
    # filename 属性定义，表示映射的文件名，可以为字符串或 None
    filename: str | None
    # offset 属性定义，表示在文件中的偏移量
    offset: int
    # mode 属性定义，表示映射模式，为字符串类型
    mode: str
    # 定义 __new__ 方法，用于创建新的 memmap 对象
    def __new__(
        subtype,
        filename: str | bytes | os.PathLike[str] | os.PathLike[bytes] | _MemMapIOProtocol,
        dtype: type[uint8] = ...,
        mode: _MemMapModeKind = ...,
        offset: int = ...,
        shape: None | int | tuple[int, ...] = ...,
        order: _OrderKACF = ...,
    ) -> memmap[Any, dtype[uint8]]: ...

    # 重载 __new__ 方法，支持不同的 dtype 类型参数
    @overload
    def __new__(
        subtype,
        filename: str | bytes | os.PathLike[str] | os.PathLike[bytes] | _MemMapIOProtocol,
        dtype: _DTypeLike[_ScalarType],
        mode: _MemMapModeKind = ...,
        offset: int = ...,
        shape: None | int | tuple[int, ...] = ...,
        order: _OrderKACF = ...,
    ) -> memmap[Any, dtype[_ScalarType]]: ...

    # 重载 __new__ 方法，支持通用的 dtype 参数
    @overload
    def __new__(
        subtype,
        filename: str | bytes | os.PathLike[str] | os.PathLike[bytes] | _MemMapIOProtocol,
        dtype: DTypeLike,
        mode: _MemMapModeKind = ...,
        offset: int = ...,
        shape: None | int | tuple[int, ...] = ...,
        order: _OrderKACF = ...,
    ) -> memmap[Any, dtype[Any]]: ...

    # 定义 __array_finalize__ 方法，用于在数组 finalization 时执行操作
    def __array_finalize__(self, obj: object) -> None: ...

    # 定义 __array_wrap__ 方法，用于在数组 wrap 操作时处理 memmap 对象
    def __array_wrap__(
        self,
        array: memmap[_ShapeType, _DType_co],
        context: None | tuple[ufunc, tuple[Any, ...], int] = ...,
        return_scalar: builtins.bool = ...,
    ) -> Any: ...

    # 定义 flush 方法，用于刷新 memmap 对象的数据
    def flush(self) -> None: ...
# 定义一个类 `vectorize`
class vectorize:
    # 属性：接受任意参数的可调用对象
    pyfunc: Callable[..., Any]
    # 属性：布尔值，指示是否启用缓存
    cache: builtins.bool
    # 属性：签名字符串或者空值
    signature: None | str
    # 属性：输出类型字符串或者空值
    otypes: None | str
    # 属性：排除的整数或字符串集合
    excluded: set[int | str]
    # 属性：文档字符串或者空值
    __doc__: None | str

    # 初始化方法，接受可调用对象、输出类型、文档字符串、排除项、缓存标志、签名字符串作为参数
    def __init__(
        self,
        pyfunc: Callable[..., Any],
        otypes: None | str | Iterable[DTypeLike] = ...,
        doc: None | str = ...,
        excluded: None | Iterable[int | str] = ...,
        cache: builtins.bool = ...,
        signature: None | str = ...,
    ) -> None: ...

    # 调用对象实例时的方法，接受任意位置参数和关键字参数
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

# 定义一个类 `poly1d`
class poly1d:
    # 属性：变量名称字符串
    @property
    def variable(self) -> str: ...
    # 属性：多项式的阶数
    @property
    def order(self) -> int: ...
    # 属性：未知
    @property
    def o(self) -> int: ...
    # 属性：多项式的根，数组
    @property
    def roots(self) -> NDArray[Any]: ...
    # 属性：未知
    @property
    def r(self) -> NDArray[Any]: ...

    # 属性：多项式的系数，数组
    @property
    def coeffs(self) -> NDArray[Any]: ...
    # 设置 `coeffs` 属性的方法，接受数组类型的值
    @coeffs.setter
    def coeffs(self, value: NDArray[Any]) -> None: ...

    # 属性：未知
    @property
    def c(self) -> NDArray[Any]: ...
    # 设置 `c` 属性的方法，接受数组类型的值
    @c.setter
    def c(self, value: NDArray[Any]) -> None: ...

    # 属性：未知
    @property
    def coef(self) -> NDArray[Any]: ...
    # 设置 `coef` 属性的方法，接受数组类型的值
    @coef.setter
    def coef(self, value: NDArray[Any]) -> None: ...

    # 属性：多项式的系数，数组
    @property
    def coefficients(self) -> NDArray[Any]: ...
    # 设置 `coefficients` 属性的方法，接受数组类型的值
    @coefficients.setter
    def coefficients(self, value: NDArray[Any]) -> None: ...

    # 类型忽略的 `__hash__` 属性
    __hash__: ClassVar[None]

    # 转换为数组的方法重载，接受类型、复制标志作为参数
    @overload
    def __array__(self, t: None = ..., copy: None | bool = ...) -> NDArray[Any]: ...
    @overload
    def __array__(self, t: _DType, copy: None | bool = ...) -> ndarray[Any, _DType]: ...

    # 调用对象实例时的方法重载，接受标量或 `poly1d` 对象或类似数组作为参数
    @overload
    def __call__(self, val: _ScalarLike_co) -> Any: ...
    @overload
    def __call__(self, val: poly1d) -> poly1d: ...
    @overload
    def __call__(self, val: ArrayLike) -> NDArray[Any]: ...

    # 初始化方法，接受数组或类似数组、布尔值、变量名称字符串作为参数
    def __init__(
        self,
        c_or_r: ArrayLike,
        r: builtins.bool = ...,
        variable: None | str = ...,
    ) -> None: ...

    # 返回多项式长度的方法
    def __len__(self) -> int: ...

    # 返回多项式的负值的方法
    def __neg__(self) -> poly1d: ...

    # 返回多项式的正值的方法
    def __pos__(self) -> poly1d: ...

    # 返回多项式与另一数组的乘积的方法
    def __mul__(self, other: ArrayLike) -> poly1d: ...

    # 返回另一数组与多项式的乘积的方法
    def __rmul__(self, other: ArrayLike) -> poly1d: ...

    # 返回多项式与另一数组的和的方法
    def __add__(self, other: ArrayLike) -> poly1d: ...

    # 返回另一数组与多项式的和的方法
    def __radd__(self, other: ArrayLike) -> poly1d: ...

    # 返回多项式的指数幂的方法，接受浮点数类型的参数
    def __pow__(self, val: _FloatLike_co) -> poly1d: ...

    # 返回多项式与另一数组的差的方法
    def __sub__(self, other: ArrayLike) -> poly1d: ...

    # 返回另一数组与多项式的差的方法
    def __rsub__(self, other: ArrayLike) -> poly1d: ...

    # 返回多项式与另一数组的除法的方法
    def __div__(self, other: ArrayLike) -> poly1d: ...

    # 返回多项式与另一数组的真除法的方法
    def __truediv__(self, other: ArrayLike) -> poly1d: ...

    # 返回另一数组与多项式的除法的方法
    def __rdiv__(self, other: ArrayLike) -> poly1d: ...

    # 返回另一数组与多项式的真除法的方法
    def __rtruediv__(self, other: ArrayLike) -> poly1d: ...

    # 获取多项式某个索引位置的方法，接受整数作为参数
    def __getitem__(self, val: int) -> Any: ...

    # 设置多项式某个索引位置的方法，接受整数和任意类型的值作为参数
    def __setitem__(self, key: int, val: Any) -> None: ...

    # 返回迭代器的方法，迭代多项式的每个元素
    def __iter__(self) -> Iterator[Any]: ...
    # 定义一个方法 deriv，用于计算多项式的导数，参数 m 表示导数的阶数，默认为任意整数或索引支持的类型，返回一个 poly1d 对象
    def deriv(self, m: SupportsInt | SupportsIndex = ...) -> poly1d: ...
    
    # 定义一个方法 integ，用于计算多项式的不定积分，参数 m 表示积分的阶数，默认为任意整数或索引支持的类型，参数 k 表示积分常数，默认为 None 或复数数组或对象数组支持的类型，返回一个 poly1d 对象
    def integ(
        self,
        m: SupportsInt | SupportsIndex = ...,
        k: None | _ArrayLikeComplex_co | _ArrayLikeObject_co = ...,
    ) -> poly1d: ...
# 定义一个名为 `matrix` 的类，继承自 `ndarray[_ShapeType, _DType_co]`
class matrix(ndarray[_ShapeType, _DType_co]):
    # 设置类变量 `__array_priority__`，用于确定数组操作的优先级
    __array_priority__: ClassVar[float]

    # 定义 `__new__` 方法，用于创建新的 matrix 实例
    def __new__(
        subtype,
        data: ArrayLike,
        dtype: DTypeLike = ...,
        copy: builtins.bool = ...,
    ) -> matrix[Any, Any]: ...

    # 定义 `__array_finalize__` 方法，用于对数组实例进行最终初始化
    def __array_finalize__(self, obj: object) -> None: ...

    # 定义 __getitem__ 方法的重载，支持不同类型的索引和切片操作
    @overload
    def __getitem__(self, key: (
        SupportsIndex
        | _ArrayLikeInt_co
        | tuple[SupportsIndex | _ArrayLikeInt_co, ...]
    )) -> Any: ...
    @overload
    def __getitem__(self, key: (
        None
        | slice
        | ellipsis
        | SupportsIndex
        | _ArrayLikeInt_co
        | tuple[None | slice | ellipsis | _ArrayLikeInt_co | SupportsIndex, ...]
    )) -> matrix[Any, _DType_co]: ...
    @overload
    def __getitem__(self: NDArray[void], key: str) -> matrix[Any, dtype[Any]]: ...
    @overload
    def __getitem__(self: NDArray[void], key: list[str]) -> matrix[_ShapeType, dtype[void]]: ...

    # 定义 `__mul__` 方法，实现矩阵乘法操作
    def __mul__(self, other: ArrayLike) -> matrix[Any, Any]: ...

    # 定义 `__rmul__` 方法，实现反向矩阵乘法操作
    def __rmul__(self, other: ArrayLike) -> matrix[Any, Any]: ...

    # 定义 `__imul__` 方法，实现就地乘法操作
    def __imul__(self, other: ArrayLike) -> matrix[_ShapeType, _DType_co]: ...

    # 定义 `__pow__` 方法，实现幂运算操作
    def __pow__(self, other: ArrayLike) -> matrix[Any, Any]: ...

    # 定义 `__ipow__` 方法，实现就地幂运算操作
    def __ipow__(self, other: ArrayLike) -> matrix[_ShapeType, _DType_co]: ...

    # 定义 `sum` 方法的重载，计算数组元素的和
    @overload
    def sum(self, axis: None = ..., dtype: DTypeLike = ..., out: None = ...) -> Any: ...
    @overload
    def sum(self, axis: _ShapeLike, dtype: DTypeLike = ..., out: None = ...) -> matrix[Any, Any]: ...
    @overload
    def sum(self, axis: None | _ShapeLike = ..., dtype: DTypeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    # 定义 `mean` 方法的重载，计算数组元素的平均值
    @overload
    def mean(self, axis: None = ..., dtype: DTypeLike = ..., out: None = ...) -> Any: ...
    @overload
    def mean(self, axis: _ShapeLike, dtype: DTypeLike = ..., out: None = ...) -> matrix[Any, Any]: ...
    @overload
    def mean(self, axis: None | _ShapeLike = ..., dtype: DTypeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    # 定义 `std` 方法的重载，计算数组元素的标准差
    @overload
    def std(self, axis: None = ..., dtype: DTypeLike = ..., out: None = ..., ddof: float = ...) -> Any: ...
    @overload
    def std(self, axis: _ShapeLike, dtype: DTypeLike = ..., out: None = ..., ddof: float = ...) -> matrix[Any, Any]: ...
    @overload
    def std(self, axis: None | _ShapeLike = ..., dtype: DTypeLike = ..., out: _NdArraySubClass = ..., ddof: float = ...) -> _NdArraySubClass: ...

    # 定义 `var` 方法的重载，计算数组元素的方差
    @overload
    def var(self, axis: None = ..., dtype: DTypeLike = ..., out: None = ..., ddof: float = ...) -> Any: ...
    @overload
    def var(self, axis: _ShapeLike, dtype: DTypeLike = ..., out: None = ..., ddof: float = ...) -> matrix[Any, Any]: ...
    @overload
    def var(self, axis: None | _ShapeLike = ..., dtype: DTypeLike = ..., out: _NdArraySubClass = ..., ddof: float = ...) -> _NdArraySubClass: ...

    # 定义 `prod` 方法的重载，计算数组元素的乘积
    @overload
    def prod(self, axis: None = ..., dtype: DTypeLike = ..., out: None = ...) -> Any: ...
    @overload
    # 定义一个名为 `prod` 的方法，用于计算数组在指定轴向上的乘积，支持多种重载形式
    def prod(self, axis: _ShapeLike, dtype: DTypeLike = ..., out: None = ...) -> matrix[Any, Any]: ...
    # 重载 `prod` 方法，支持返回值为自定义数组类型 `_NdArraySubClass` 的情况

    # 定义一个名为 `any` 的方法，用于检查数组在指定轴向上是否存在非零元素，支持多种重载形式
    @overload
    def any(self, axis: None = ..., out: None = ...) -> np.bool: ...
    @overload
    def any(self, axis: _ShapeLike, out: None = ...) -> matrix[Any, dtype[np.bool]]: ...
    @overload
    def any(self, axis: None | _ShapeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    # 定义一个名为 `all` 的方法，用于检查数组在指定轴向上是否所有元素都非零，支持多种重载形式
    @overload
    def all(self, axis: None = ..., out: None = ...) -> np.bool: ...
    @overload
    def all(self, axis: _ShapeLike, out: None = ...) -> matrix[Any, dtype[np.bool]]: ...
    @overload
    def all(self, axis: None | _ShapeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    # 定义一个名为 `max` 的方法，用于计算数组在指定轴向上的最大值，支持多种重载形式
    @overload
    def max(self: NDArray[_ScalarType], axis: None = ..., out: None = ...) -> _ScalarType: ...
    @overload
    def max(self, axis: _ShapeLike, out: None = ...) -> matrix[Any, _DType_co]: ...
    @overload
    def max(self, axis: None | _ShapeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    # 定义一个名为 `min` 的方法，用于计算数组在指定轴向上的最小值，支持多种重载形式
    @overload
    def min(self: NDArray[_ScalarType], axis: None = ..., out: None = ...) -> _ScalarType: ...
    @overload
    def min(self, axis: _ShapeLike, out: None = ...) -> matrix[Any, _DType_co]: ...
    @overload
    def min(self, axis: None | _ShapeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    # 定义一个名为 `argmax` 的方法，用于找出数组在指定轴向上最大值的索引，支持多种重载形式
    @overload
    def argmax(self: NDArray[_ScalarType], axis: None = ..., out: None = ...) -> intp: ...
    @overload
    def argmax(self, axis: _ShapeLike, out: None = ...) -> matrix[Any, dtype[intp]]: ...
    @overload
    def argmax(self, axis: None | _ShapeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    # 定义一个名为 `argmin` 的方法，用于找出数组在指定轴向上最小值的索引，支持多种重载形式
    @overload
    def argmin(self: NDArray[_ScalarType], axis: None = ..., out: None = ...) -> intp: ...
    @overload
    def argmin(self, axis: _ShapeLike, out: None = ...) -> matrix[Any, dtype[intp]]: ...
    @overload
    def argmin(self, axis: None | _ShapeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    # 定义一个名为 `ptp` 的方法，用于计算数组在指定轴向上的最大值与最小值之差，支持多种重载形式
    @overload
    def ptp(self: NDArray[_ScalarType], axis: None = ..., out: None = ...) -> _ScalarType: ...
    @overload
    def ptp(self, axis: _ShapeLike, out: None = ...) -> matrix[Any, _DType_co]: ...
    @overload
    def ptp(self, axis: None | _ShapeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    # 定义一个名为 `squeeze` 的方法，用于移除数组中的单维度条目，支持多种重载形式
    def squeeze(self, axis: None | _ShapeLike = ...) -> matrix[Any, _DType_co]: ...

    # 定义一个名为 `tolist` 的方法，用于将数组转换为 Python 嵌套列表
    def tolist(self: matrix[Any, dtype[_SupportsItem[_T]]]) -> list[list[_T]]: ...  # type: ignore[typevar]

    # 定义一个名为 `ravel` 的方法，用于将多维数组转换为一维数组
    def ravel(self, order: _OrderKACF = ...) -> matrix[Any, _DType_co]: ...

    # 定义一个名为 `flatten` 的方法，用于将数组按行展开为一维数组
    def flatten(self, order: _OrderKACF = ...) -> matrix[Any, _DType_co]: ...

    # 定义一个名为 `T` 的属性，用于返回数组的转置
    @property
    def T(self) -> matrix[Any, _DType_co]: ...

    # 定义一个名为 `I` 的属性，用于返回数组的单位矩阵或单位数组
    @property
    def I(self) -> matrix[Any, Any]: ...
    # 定义方法 A，返回类型为 ndarray，具体的形状和数据类型由 _ShapeType 和 _DType_co 决定
    def A(self) -> ndarray[_ShapeType, _DType_co]: ...
    
    # 定义属性 A1，返回类型为 ndarray，形状可以是任意的，数据类型由 _DType_co 决定
    @property
    def A1(self) -> ndarray[Any, _DType_co]: ...
    
    # 定义方法 H，返回类型为 matrix，形状可以是任意的，数据类型由 _DType_co 决定
    def H(self) -> matrix[Any, _DType_co]: ...
    
    # 定义方法 getT，返回类型为 matrix，形状可以是任意的，数据类型由 _DType_co 决定
    def getT(self) -> matrix[Any, _DType_co]: ...
    
    # 定义方法 getI，返回类型为 matrix，形状和数据类型都可以是任意的
    def getI(self) -> matrix[Any, Any]: ...
    
    # 定义方法 getA，返回类型为 ndarray，具体的形状和数据类型由 _ShapeType 和 _DType_co 决定
    def getA(self) -> ndarray[_ShapeType, _DType_co]: ...
    
    # 定义方法 getA1，返回类型为 ndarray，形状可以是任意的，数据类型由 _DType_co 决定
    def getA1(self) -> ndarray[Any, _DType_co]: ...
    
    # 定义方法 getH，返回类型为 matrix，形状可以是任意的，数据类型由 _DType_co 决定
    def getH(self) -> matrix[Any, _DType_co]: ...
# 定义一个泛型类型变量 `_CharType`，它可以是 str 或 bytes 类型之一
_CharType = TypeVar("_CharType", str_, bytes_)

# 定义一个泛型类型变量 `_CharDType`，它可以是 dtype[str_] 或 dtype[bytes_] 之一
_CharDType = TypeVar("_CharDType", dtype[str_], dtype[bytes_])

# NOTE: Deprecated
# 这是一个已弃用的类 MachAr，不建议继续使用，可能会在未来的版本中移除
# class MachAr: ...

# 定义一个协议 `_SupportsDLPack`，它支持从某种类型 `_T_contra` 转换为 DLPack，返回一个 PyCapsule 对象
class _SupportsDLPack(Protocol[_T_contra]):
    def __dlpack__(self, *, stream: None | _T_contra = ...) -> _PyCapsule: ...

# 定义一个函数 from_dlpack，接受一个支持 DLPack 协议的对象 obj，返回一个任意类型的 NDArray
def from_dlpack(obj: _SupportsDLPack[None], /) -> NDArray[Any]: ...
```