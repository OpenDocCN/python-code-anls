# `.\numpy\numpy\_core\getlimits.py`

```py
"""Machine limits for Float32 and Float64 and (long double) if available...

"""
# 定义模块中公开的所有变量和函数名
__all__ = ['finfo', 'iinfo']

# 引入警告模块
import warnings

# 从内部模块导入设置模块函数
from .._utils import set_module
# 从内部模块导入机器精度相关的类
from ._machar import MachAr
# 从当前目录下的numeric模块导入array、inf、nan等
from . import numeric
# 从当前目录下的numerictypes模块导入别名ntypes
from . import numerictypes as ntypes
# 从numeric模块导入array、inf、nan等
from .numeric import array, inf, nan
# 从umath模块导入log10、exp2、nextafter、isnan等
from .umath import log10, exp2, nextafter, isnan


# 定义函数_fr0，用于将rank-0的数组修正为rank-1
def _fr0(a):
    """fix rank-0 --> rank-1"""
    if a.ndim == 0:
        a = a.copy()
        a.shape = (1,)
    return a


# 定义函数_fr1，用于将rank大于0的数组修正为rank-0
def _fr1(a):
    """fix rank > 0 --> rank-0"""
    if a.size == 1:
        a = a.copy()
        a.shape = ()
    return a


# 定义类MachArLike，用于模拟MachAr实例
class MachArLike:
    """ Object to simulate MachAr instance """
    
    # 初始化方法，接收多个参数来模拟MachAr实例
    def __init__(self, ftype, *, eps, epsneg, huge, tiny,
                 ibeta, smallest_subnormal=None, **kwargs):
        # 使用_MACHAR_PARAMS字典中的参数来设置实例的params属性
        self.params = _MACHAR_PARAMS[ftype]
        # 设置实例的ftype属性为传入的ftype参数
        self.ftype = ftype
        # 设置实例的title属性为params字典中的title值
        self.title = self.params['title']
        
        # 如果未提供smallest_subnormal参数，则计算最小的subnormal值
        if not smallest_subnormal:
            self._smallest_subnormal = nextafter(
                self.ftype(0), self.ftype(1), dtype=self.ftype)
        else:
            self._smallest_subnormal = smallest_subnormal
        
        # 使用_float_to_float方法将eps参数转换为浮点数，设置为实例的epsilon属性
        self.epsilon = self.eps = self._float_to_float(eps)
        # 使用_float_to_float方法将epsneg参数转换为浮点数，设置为实例的epsneg属性
        self.epsneg = self._float_to_float(epsneg)
        # 使用_float_to_float方法将huge参数转换为浮点数，设置为实例的xmax属性
        self.xmax = self.huge = self._float_to_float(huge)
        # 使用_float_to_float方法将tiny参数转换为浮点数，设置为实例的xmin属性
        self.xmin = self._float_to_float(tiny)
        # 使用_float_to_float方法将tiny参数转换为浮点数，设置为实例的smallest_normal属性
        self.smallest_normal = self.tiny = self._float_to_float(tiny)
        # 使用params字典中的itype属性将ibeta参数转换为整数，设置为实例的ibeta属性
        self.ibeta = self.params['itype'](ibeta)
        
        # 将kwargs中的任何其他参数更新到实例的属性中
        self.__dict__.update(kwargs)
        
        # 计算精度并设置实例的precision属性
        self.precision = int(-log10(self.eps))
        # 使用_float_conv方法计算分辨率并设置实例的resolution属性
        self.resolution = self._float_to_float(
            self._float_conv(10) ** (-self.precision))
        
        # 使用_float_to_str方法将eps、epsneg、xmin、xmax、resolution属性转换为字符串，并设置相应的_str属性
        self._str_eps = self._float_to_str(self.eps)
        self._str_epsneg = self._float_to_str(self.epsneg)
        self._str_xmin = self._float_to_str(self.xmin)
        self._str_xmax = self._float_to_str(self.xmax)
        self._str_resolution = self._float_to_str(self.resolution)
        self._str_smallest_normal = self._float_to_str(self.xmin)

    @property
    # 定义smallest_subnormal属性的getter方法，返回最小subnormal值的浮点数表示
    def smallest_subnormal(self):
        """Return the value for the smallest subnormal.

        Returns
        -------
        smallest_subnormal : float
            value for the smallest subnormal.

        Warns
        -----
        UserWarning
            If the calculated value for the smallest subnormal is zero.
        """
        # 检查计算出的最小subnormal值是否为零，如果是则发出警告
        value = self._smallest_subnormal
        if self.ftype(0) == value:
            warnings.warn(
                'The value of the smallest subnormal for {} type '
                'is zero.'.format(self.ftype), UserWarning, stacklevel=2)

        # 使用_float_to_float方法将计算出的值转换为浮点数并返回
        return self._float_to_float(value)

    @property
    # 定义_str_smallest_subnormal属性的getter方法，返回最小subnormal值的字符串表示
    def _str_smallest_subnormal(self):
        """Return the string representation of the smallest subnormal."""
        # 使用_float_to_str方法将最小subnormal值转换为字符串并返回
        return self._float_to_str(self.smallest_subnormal)
    # 将浮点数转换为浮点数
    def _float_to_float(self, value):
        """Converts float to float.

        Parameters
        ----------
        value : float
            value to be converted.
        """
        # 调用内部方法 _float_conv 进行转换
        return _fr1(self._float_conv(value))

    # 将浮点数转换为某种格式的数组
    def _float_conv(self, value):
        """Converts float to conv.

        Parameters
        ----------
        value : float
            value to be converted.
        """
        # 使用数组将给定的浮点数转换为指定类型(self.ftype)
        return array([value], self.ftype)

    # 将浮点数转换为字符串
    def _float_to_str(self, value):
        """Converts float to str.

        Parameters
        ----------
        value : float
            value to be converted.
        """
        # 从参数中提取格式化字符串，并将浮点数转换为符合格式的字符串
        return self.params['fmt'] % array(_fr0(value)[0], self.ftype)
# 将复数类型映射为对应的浮点数类型
_convert_to_float = {
    ntypes.csingle: ntypes.single,           # 将单精度复数映射为单精度浮点数
    ntypes.complex128: ntypes.float64,       # 将复数128位类型映射为双精度浮点数
    ntypes.clongdouble: ntypes.longdouble    # 将长双精度复数映射为长双精度浮点数
    }

# 创建 MachAr / 类似 MachAr 对象的参数
_title_fmt = 'numpy {} precision floating point number'
_MACHAR_PARAMS = {
    ntypes.double: dict(
        itype = ntypes.int64,                 # 双精度浮点数的整数类型为64位整数
        fmt = '%24.16e',                      # 格式化输出双精度浮点数的格式
        title = _title_fmt.format('double')), # 双精度浮点数的标题
    ntypes.single: dict(
        itype = ntypes.int32,                 # 单精度浮点数的整数类型为32位整数
        fmt = '%15.7e',                       # 格式化输出单精度浮点数的格式
        title = _title_fmt.format('single')), # 单精度浮点数的标题
    ntypes.longdouble: dict(
        itype = ntypes.longlong,              # 长双精度浮点数的整数类型为长长整数
        fmt = '%s',                           # 格式化输出长双精度浮点数的格式
        title = _title_fmt.format('long double')),  # 长双精度浮点数的标题
    ntypes.half: dict(
        itype = ntypes.int16,                 # 半精度浮点数的整数类型为16位整数
        fmt = '%12.5e',                       # 格式化输出半精度浮点数的格式
        title = _title_fmt.format('half'))    # 半精度浮点数的标题
}

# 用于识别浮点数类型的关键字。关键字的生成方式如下：
#    ftype = np.longdouble        # 或 float64、float32 等
#    v = (ftype(-1.0) / ftype(10.0))
#    v.view(v.dtype.newbyteorder('<')).tobytes()
#
# 使用除法来解决某些平台上 strtold 的不足之处。
# 参考：
# https://perl5.git.perl.org/perl.git/blob/3118d7d684b56cbeb702af874f4326683c45f045:/Configure

_KNOWN_TYPES = {}

def _register_type(machar, bytepat):
    _KNOWN_TYPES[bytepat] = machar

_float_ma = {}

def _register_known_types():
    # 已知的 float16 参数
    # 参见 MachAr 类的文档字符串，描述这些参数的含义。
    f16 = ntypes.float16
    float16_ma = MachArLike(f16,
                            machep=-10,
                            negep=-11,
                            minexp=-14,
                            maxexp=16,
                            it=10,
                            iexp=5,
                            ibeta=2,
                            irnd=5,
                            ngrd=0,
                            eps=exp2(f16(-10)),
                            epsneg=exp2(f16(-11)),
                            huge=f16(65504),
                            tiny=f16(2 ** -14))
    _register_type(float16_ma, b'f\xae')
    _float_ma[16] = float16_ma

    # 已知的 float32 参数
    f32 = ntypes.float32
    float32_ma = MachArLike(f32,
                            machep=-23,
                            negep=-24,
                            minexp=-126,
                            maxexp=128,
                            it=23,
                            iexp=8,
                            ibeta=2,
                            irnd=5,
                            ngrd=0,
                            eps=exp2(f32(-23)),
                            epsneg=exp2(f32(-24)),
                            huge=f32((1 - 2 ** -24) * 2**128),
                            tiny=exp2(f32(-126)))
    _register_type(float32_ma, b'\xcd\xcc\xcc\xbd')
    _float_ma[32] = float32_ma

    # 已知的 float64 参数
    f64 = ntypes.float64
    epsneg_f64 = 2.0 ** -53.0
    tiny_f64 = 2.0 ** -1022.0
    # 创建一个 MachArLike 实例，表示 64 位浮点数的机器精度
    float64_ma = MachArLike(f64,
                            machep=-52,
                            negep=-53,
                            minexp=-1022,
                            maxexp=1024,
                            it=52,
                            iexp=11,
                            ibeta=2,
                            irnd=5,
                            ngrd=0,
                            eps=2.0 ** -52.0,
                            epsneg=epsneg_f64,
                            huge=(1.0 - epsneg_f64) / tiny_f64 * f64(4),
                            tiny=tiny_f64)
    # 将该实例注册到类型系统中，用特定的字节序列标识 64 位浮点数
    _register_type(float64_ma, b'\x9a\x99\x99\x99\x99\x99\xb9\xbf')
    # 将该机器精度信息存储到 _float_ma 字典中，键为 64
    _float_ma[64] = float64_ma

    # 已知的 IEEE 754 128 位二进制浮点数参数
    ld = ntypes.longdouble
    epsneg_f128 = exp2(ld(-113))
    tiny_f128 = exp2(ld(-16382))
    # 忽略运行时错误，当这不是 f128 时
    with numeric.errstate(all='ignore'):
        huge_f128 = (ld(1) - epsneg_f128) / tiny_f128 * ld(4)
    # 创建一个 MachArLike 实例，表示 128 位浮点数的机器精度
    float128_ma = MachArLike(ld,
                             machep=-112,
                             negep=-113,
                             minexp=-16382,
                             maxexp=16384,
                             it=112,
                             iexp=15,
                             ibeta=2,
                             irnd=5,
                             ngrd=0,
                             eps=exp2(ld(-112)),
                             epsneg=epsneg_f128,
                             huge=huge_f128,
                             tiny=tiny_f128)
    # 将该实例注册到类型系统中，用特定的字节序列标识 128 位浮点数
    _register_type(float128_ma,
        b'\x9a\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\xfb\xbf')
    # 将该机器精度信息存储到 _float_ma 字典中，键为 128
    _float_ma[128] = float128_ma

    # 已知的 Intel 80 位扩展精度浮点数（float80）参数
    epsneg_f80 = exp2(ld(-64))
    tiny_f80 = exp2(ld(-16382))
    # 忽略运行时错误，当这不是 f80 时
    with numeric.errstate(all='ignore'):
        huge_f80 = (ld(1) - epsneg_f80) / tiny_f80 * ld(4)
    # 创建一个 MachArLike 实例，表示 80 位浮点数的机器精度
    float80_ma = MachArLike(ld,
                            machep=-63,
                            negep=-64,
                            minexp=-16382,
                            maxexp=16384,
                            it=63,
                            iexp=15,
                            ibeta=2,
                            irnd=5,
                            ngrd=0,
                            eps=exp2(ld(-63)),
                            epsneg=epsneg_f80,
                            huge=huge_f80,
                            tiny=tiny_f80)
    # 将该实例注册到类型系统中，用特定的字节序列标识 80 位浮点数
    _register_type(float80_ma, b'\xcd\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xfb\xbf')
    # 将该机器精度信息存储到 _float_ma 字典中，键为 80
    _float_ma[80] = float80_ma

    # 猜测 / 已知的双倍精度浮点数参数；参考：
    # https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format#Double-double_arithmetic
    # 这些数字具有与 float64 相同的指数范围，但扩展了精度
    # 在双重精度浮点数中，表示尾数的位数。
    huge_dd = nextafter(ld(inf), ld(0), dtype=ld)
    # 由于双重精度浮点数中最小正常数难以计算，因此将其设为 NaN。
    smallest_normal_dd = nan
    # 将双重精度浮点数中最小次正常数设为与普通双精度浮点数相同的值。
    smallest_subnormal_dd = ld(nextafter(0., 1.))
    # 创建一个类似于 MachAr 的对象，用于描述双重精度浮点数的机器精度特性。
    float_dd_ma = MachArLike(ld,
                             machep=-105,
                             negep=-106,
                             minexp=-1022,
                             maxexp=1024,
                             it=105,
                             iexp=11,
                             ibeta=2,
                             irnd=5,
                             ngrd=0,
                             eps=exp2(ld(-105)),
                             epsneg=exp2(ld(-106)),
                             huge=huge_dd,
                             tiny=smallest_normal_dd,
                             smallest_subnormal=smallest_subnormal_dd)
    # 将双重精度浮点数的低位和高位顺序注册为一种类型，例如 PPC 64 架构。
    _register_type(float_dd_ma,
        b'\x9a\x99\x99\x99\x99\x99Y<\x9a\x99\x99\x99\x99\x99\xb9\xbf')
    # 将双重精度浮点数的高位和低位顺序注册为一种类型，例如 PPC 64 架构（小端序）。
    _register_type(float_dd_ma,
        b'\x9a\x99\x99\x99\x99\x99\xb9\xbf\x9a\x99\x99\x99\x99\x99Y<')
    # 将双重精度浮点数的机器精度特性对象添加到浮点数类型的字典中，使用 'dd' 作为键。
    _float_ma['dd'] = float_dd_ma
# 定义一个函数 `_get_machar`，用于获取 `MachAr` 实例或类似 `MachAr` 的实例
def _get_machar(ftype):
    # 尝试根据各种已知的浮点类型签名获取浮点类型的参数
    params = _MACHAR_PARAMS.get(ftype)
    # 如果找不到与给定浮点类型匹配的参数，则抛出 ValueError 异常
    if params is None:
        raise ValueError(repr(ftype))
    
    # 检测已知/猜测的浮点类型
    key = (ftype(-1.0) / ftype(10.))
    # 将 key 转换为小字节序，并转换为字节序列
    key = key.view(key.dtype.newbyteorder("<")).tobytes()
    
    # 初始化 ma_like 变量
    ma_like = None
    
    # 如果 ftype 是 ntypes.longdouble 类型
    if ftype == ntypes.longdouble:
        # 可能是 80 位 == 10 字节扩展精度，其中最后几个字节可能是随机垃圾
        # 比较前 10 字节与模式的首字节以避免在随机垃圾上分支
        ma_like = _KNOWN_TYPES.get(key[:10])
    
    # 如果 ma_like 仍然为空
    if ma_like is None:
        # 查看是否已知整个 key
        ma_like = _KNOWN_TYPES.get(key)
    
    # 如果 ma_like 仍然为空并且 key 的长度为 16
    if ma_like is None and len(key) == 16:
        # 机器限制可能是伪装成 np.float128 的 f80，找出所有长度为 16 的键并创建新字典，
        # 但使键只有 10 个字节长，最后几个字节可能是随机垃圾
        _kt = {k[:10]: v for k, v in _KNOWN_TYPES.items() if len(k) == 16}
        ma_like = _kt.get(key[:10])
    
    # 如果 ma_like 不为空，则返回它
    if ma_like is not None:
        return ma_like
    
    # 如果上述检测都失败，则发出警告，并返回通过 _discovered_machar 函数探测到的浮点类型
    warnings.warn(
        f'Signature {key} for {ftype} does not match any known type: '
        'falling back to type probe function.\n'
        'This warnings indicates broken support for the dtype!',
        UserWarning, stacklevel=2)
    return _discovered_machar(ftype)


# 定义一个函数 `_discovered_machar`，用于创建包含浮点类型信息的 `MachAr` 实例
def _discovered_machar(ftype):
    # 获取浮点类型的参数
    params = _MACHAR_PARAMS[ftype]
    # 返回一个 MachAr 实例，使用 lambda 表达式定义各种函数
    return MachAr(
        lambda v: array([v], ftype),  # 将值 v 转换为包含单个元素的数组，并指定类型为 ftype
        lambda v: _fr0(v.astype(params['itype']))[0],  # 对值 v 进行转换并返回其第一个元素
        lambda v: array(_fr0(v)[0], ftype),  # 将值 v 转换为数组并指定类型为 ftype
        lambda v: params['fmt'] % array(_fr0(v)[0], ftype),  # 使用格式化字符串返回转换后的值 v
        params['title']  # 返回浮点类型的标题
    )


# 设置一个类 `finfo`，表示浮点类型的机器限制
@set_module('numpy')
class finfo:
    """
    finfo(dtype)

    浮点类型的机器限制。

    Attributes
    ----------
    bits : int
        类型占用的位数。
    """
    dtype : dtype
        # 返回`finfo`返回信息的数据类型。对于复数输入，返回的数据类型是它的实部和虚部相关联的`float*`数据类型。
    eps : float
        # 返回1.0和大于1.0的下一个最小可表示浮点数之间的差异。例如，在IEEE-754标准的64位二进制浮点数中，`eps = 2**-52`，约为2.22e-16。
    epsneg : float
        # 返回1.0和小于1.0的下一个最小可表示浮点数之间的差异。例如，在IEEE-754标准的64位二进制浮点数中，`epsneg = 2**-53`，约为1.11e-16。
    iexp : int
        # 浮点表示的指数部分中的位数。
    machep : int
        # 返回`eps`的幂。
    max : floating point number of the appropriate type
        # 可表示的最大数。
    maxexp : int
        # 导致溢出的基数(2)的最小正幂。
    min : floating point number of the appropriate type
        # 可表示的最小数，通常为`-max`。
    minexp : int
        # 与尾数中没有前导 0 一致的基数(2)的最小负幂。
    negep : int
        # 返回`epsneg`的幂。
    nexp : int
        # 指数的位数，包括其符号和偏置。
    nmant : int
        # 尾数的位数。
    precision : int
        # 此种类型浮点数精确的大致十进制数字个数。
    resolution : floating point number of the appropriate type
        # 此类型的大致十进制分辨率，即`10**-precision`。
    tiny : float
        # `smallest_normal`的别名，保留了向后兼容性。
    smallest_normal : float
        # 以下IEEE-754标准为首位的最小正浮点数（参见注释）。
    smallest_subnormal : float
        # 以下IEEE-754标准为首位的最小正浮点数。

    Parameters
    ----------
    dtype : float, dtype, or instance
        # 要获取信息的浮点数或复数浮点数数据类型的种类。

    See Also
    --------
    iinfo : 整数数据类型的等价物。
    spacing : 值与最近的相邻数之间的距离
    nextafter : x1朝向x2的下一个浮点数值

    Notes
    -----
    对于NumPy的开发人员：不要在模块级别实例化此对象。
    初始计算这些参数是昂贵的，并且会对导入时间产生负面影响。这些对象是缓存的，所以在函数内部重复调用`finfo()`并不是问题。

    请注意，`smallest_normal`实际上不是最小的正数。
    # 创建一个缓存字典用于存储各种 NumPy 数值类型的 finfo 对象
    _finfo_cache = {}
    # 定义一个特殊方法 __new__，用于创建新的对象实例，接收类 cls 和数据类型 dtype 作为参数
    def __new__(cls, dtype):
        try:
            # 尝试从 _finfo_cache 中获取 dtype 对应的对象（通常情况）
            obj = cls._finfo_cache.get(dtype)  # most common path
            # 如果对象不为 None，直接返回该对象
            if obj is not None:
                return obj
        except TypeError:
            pass

        # 如果 dtype 为 None，发出警告（从 NumPy 1.25 开始弃用）
        if dtype is None:
            # Deprecated in NumPy 1.25, 2023-01-16
            warnings.warn(
                "finfo() dtype cannot be None. This behavior will "
                "raise an error in the future. (Deprecated in NumPy 1.25)",
                DeprecationWarning,
                stacklevel=2
            )

        try:
            # 尝试将 dtype 转换为 numpy 的数据类型
            dtype = numeric.dtype(dtype)
        except TypeError:
            # 如果给定的 dtype 是一个 float 实例，转换为其类型的数据类型
            dtype = numeric.dtype(type(dtype))

        # 重新尝试从 _finfo_cache 中获取 dtype 对应的对象
        obj = cls._finfo_cache.get(dtype)
        # 如果对象不为 None，直接返回该对象
        if obj is not None:
            return obj

        # 将当前 dtype 加入到待处理的数据类型列表中
        dtypes = [dtype]
        # 将 dtype 转换为相应的数据类型
        newdtype = ntypes.obj2sctype(dtype)
        # 如果转换后的数据类型与原始 dtype 不同，加入到待处理数据类型列表中
        if newdtype is not dtype:
            dtypes.append(newdtype)
            dtype = newdtype

        # 如果 dtype 不是 numeric.inexact 的子类，抛出数值错误异常
        if not issubclass(dtype, numeric.inexact):
            raise ValueError("data type %r not inexact" % (dtype))

        # 尝试从 _finfo_cache 中再次获取 dtype 对应的对象
        obj = cls._finfo_cache.get(dtype)
        # 如果对象不为 None，直接返回该对象
        if obj is not None:
            return obj

        # 如果 dtype 不是 numeric.floating 的子类，尝试将其转换为相应的浮点数数据类型
        if not issubclass(dtype, numeric.floating):
            newdtype = _convert_to_float[dtype]
            # 如果转换后的数据类型与原始 dtype 不同，更新数据类型列表
            if newdtype is not dtype:
                # 数据类型已更改，例如从 complex128 更改为 float64
                dtypes.append(newdtype)
                dtype = newdtype

                # 尝试从 _finfo_cache 中获取新的 dtype 对应的对象
                obj = cls._finfo_cache.get(dtype, None)
                # 如果对象不为 None，将原始的 dtypes 添加到缓存中并返回结果
                if obj is not None:
                    for dt in dtypes:
                        cls._finfo_cache[dt] = obj
                    return obj

        # 使用 object.__new__ 创建新的对象实例，并初始化
        obj = object.__new__(cls)._init(dtype)
        # 将所有待处理的 dtypes 添加到缓存中
        for dt in dtypes:
            cls._finfo_cache[dt] = obj
        return obj
    # 初始化方法，设置对象的数据类型和机器参数
    def _init(self, dtype):
        # 将数据类型设置为指定的 dtype
        self.dtype = numeric.dtype(dtype)
        # 获取指定数据类型的机器参数
        machar = _get_machar(dtype)

        # 使用 machar 对象设置对象的属性
        for word in ['precision', 'iexp',
                     'maxexp', 'minexp', 'negep',
                     'machep']:
            setattr(self, word, getattr(machar, word))
        
        # 设置另一组属性，从 machar 对象获取
        for word in ['resolution', 'epsneg', 'smallest_subnormal']:
            setattr(self, word, getattr(machar, word).flat[0])
        
        # 设置一些其他的属性
        self.bits = self.dtype.itemsize * 8  # 计算位数
        self.max = machar.huge.flat[0]  # 最大值
        self.min = -self.max  # 最小值
        self.eps = machar.eps.flat[0]  # 机器精度
        self.nexp = machar.iexp  # 指数位数
        self.nmant = machar.it  # 尾数位数
        self._machar = machar  # 保留 machar 对象的引用
        self._str_tiny = machar._str_xmin.strip()  # 最小正数的字符串表示
        self._str_max = machar._str_xmax.strip()  # 最大数的字符串表示
        self._str_epsneg = machar._str_epsneg.strip()  # 负的机器精度的字符串表示
        self._str_eps = machar._str_eps.strip()  # 机器精度的字符串表示
        self._str_resolution = machar._str_resolution.strip()  # 分辨率的字符串表示
        self._str_smallest_normal = machar._str_smallest_normal.strip()  # 最小正常数的字符串表示
        self._str_smallest_subnormal = machar._str_smallest_subnormal.strip()  # 最小非正常数的字符串表示
        return self  # 返回初始化后的对象

    # 返回对象的字符串表示形式
    def __str__(self):
        # 格式化字符串，显示对象的机器参数
        fmt = (
            'Machine parameters for %(dtype)s\n'
            '---------------------------------------------------------------\n'
            'precision = %(precision)3s   resolution = %(_str_resolution)s\n'
            'machep = %(machep)6s   eps =        %(_str_eps)s\n'
            'negep =  %(negep)6s   epsneg =     %(_str_epsneg)s\n'
            'minexp = %(minexp)6s   tiny =       %(_str_tiny)s\n'
            'maxexp = %(maxexp)6s   max =        %(_str_max)s\n'
            'nexp =   %(nexp)6s   min =        -max\n'
            'smallest_normal = %(_str_smallest_normal)s   '
            'smallest_subnormal = %(_str_smallest_subnormal)s\n'
            '---------------------------------------------------------------\n'
            )
        return fmt % self.__dict__  # 使用对象的字典属性进行格式化并返回字符串

    # 返回对象的官方字符串表示形式
    def __repr__(self):
        c = self.__class__.__name__  # 获取类名
        d = self.__dict__.copy()  # 复制对象的字典属性
        d['klass'] = c  # 添加类名属性到字典中
        return (("%(klass)s(resolution=%(resolution)s, min=-%(_str_max)s,"
                 " max=%(_str_max)s, dtype=%(dtype)s)") % d)  # 返回格式化后的官方字符串表示形式

    # 计算属性，返回最小正常数的值
    @property
    def smallest_normal(self):
        """Return the value for the smallest normal.

        Returns
        -------
        smallest_normal : float
            Value for the smallest normal.

        Warns
        -----
        UserWarning
            If the calculated value for the smallest normal is requested for
            double-double.
        """
        # 检查最小正常数是否对于 double-double 类型是未定义的
        if isnan(self._machar.smallest_normal.flat[0]):
            # 如果是，发出警告
            warnings.warn(
                'The value of smallest normal is undefined for double double',
                UserWarning, stacklevel=2)
        return self._machar.smallest_normal.flat[0]  # 返回最小正常数的值
    def tiny(self):
        """
        返回 tiny 的值，它是 smallest_normal 的别名。

        Returns
        -------
        tiny : float
            最小正常值的值，即 smallest_normal 的别名。

        Warns
        -----
        UserWarning
            如果请求了双倍精度 (double-double) 的最小正常值计算结果。
        """
        return self.smallest_normal
@set_module('numpy')
class iinfo:
    """
    iinfo(type)

    Machine limits for integer types.

    Attributes
    ----------
    bits : int
        The number of bits occupied by the type.
    dtype : dtype
        Returns the dtype for which `iinfo` returns information.
    min : int
        The smallest integer expressible by the type.
    max : int
        The largest integer expressible by the type.

    Parameters
    ----------
    int_type : integer type, dtype, or instance
        The kind of integer data type to get information about.

    See Also
    --------
    finfo : The equivalent for floating point data types.

    Examples
    --------
    With types:

    >>> ii16 = np.iinfo(np.int16)
    >>> ii16.min
    -32768
    >>> ii16.max
    32767
    >>> ii32 = np.iinfo(np.int32)
    >>> ii32.min
    -2147483648
    >>> ii32.max
    2147483647

    With instances:

    >>> ii32 = np.iinfo(np.int32(10))
    >>> ii32.min
    -2147483648
    >>> ii32.max
    2147483647

    """

    _min_vals = {}  # 存储已计算的最小值的缓存字典
    _max_vals = {}  # 存储已计算的最大值的缓存字典

    def __init__(self, int_type):
        try:
            self.dtype = numeric.dtype(int_type)  # 获取输入类型的数据类型
        except TypeError:
            self.dtype = numeric.dtype(type(int_type))  # 获取输入类型的数据类型
        self.kind = self.dtype.kind  # 获取数据类型的种类标识符
        self.bits = self.dtype.itemsize * 8  # 计算数据类型所占比特数
        self.key = "%s%d" % (self.kind, self.bits)  # 生成用于缓存的键
        if self.kind not in 'iu':  # 如果数据类型标识符不是无符号整数或有符号整数
            raise ValueError("Invalid integer data type %r." % (self.kind,))

    @property
    def min(self):
        """Minimum value of given dtype."""
        if self.kind == 'u':  # 如果是无符号整数类型
            return 0  # 返回最小值为0
        else:
            try:
                val = iinfo._min_vals[self.key]  # 尝试从缓存中获取最小值
            except KeyError:
                val = int(-(1 << (self.bits-1)))  # 计算有符号整数类型的最小值
                iinfo._min_vals[self.key] = val  # 将计算结果存入缓存
            return val  # 返回最小值

    @property
    def max(self):
        """Maximum value of given dtype."""
        try:
            val = iinfo._max_vals[self.key]  # 尝试从缓存中获取最大值
        except KeyError:
            if self.kind == 'u':  # 如果是无符号整数类型
                val = int((1 << self.bits) - 1)  # 计算无符号整数类型的最大值
            else:
                val = int((1 << (self.bits-1)) - 1)  # 计算有符号整数类型的最大值
            iinfo._max_vals[self.key] = val  # 将计算结果存入缓存
        return val  # 返回最大值

    def __str__(self):
        """String representation."""
        fmt = (
            'Machine parameters for %(dtype)s\n'
            '---------------------------------------------------------------\n'
            'min = %(min)s\n'
            'max = %(max)s\n'
            '---------------------------------------------------------------\n'
            )
        return fmt % {'dtype': self.dtype, 'min': self.min, 'max': self.max}  # 返回对象的字符串表示形式

    def __repr__(self):
        return "%s(min=%s, max=%s, dtype=%s)" % (self.__class__.__name__,
                                    self.min, self.max, self.dtype)  # 返回对象的详细字符串表示形式
```