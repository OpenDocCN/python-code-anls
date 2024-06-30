# `D:\src\scipysrc\scipy\scipy\special\_support_alternative_backends.py`

```
# 导入标准库和第三方模块
import os
import sys
import functools

# 导入科学计算相关模块
import numpy as np
from scipy._lib._array_api import (
    array_namespace, scipy_namespace_for, is_numpy
)
# 导入本地模块 _ufuncs 中的函数
from . import _ufuncs
# 这些函数在这个文件中被定义，但实际上不需要被导入；导入它们是为了让 IDE 能够识别并不报错
from ._ufuncs import (
    log_ndtr, ndtr, ndtri, erf, erfc, i0, i0e, i1, i1e, gammaln,  # noqa: F401
    gammainc, gammaincc, logit, expit, entr, rel_entr, xlogy,  # noqa: F401
    chdtr, chdtrc, betainc, betaincc, stdtr  # noqa: F401
)

# 获取环境变量中的 SCIPY_ARRAY_API 设置
_SCIPY_ARRAY_API = os.environ.get("SCIPY_ARRAY_API", False)
# 设置数组 API 兼容性前缀
array_api_compat_prefix = "scipy._lib.array_api_compat"

# 根据函数名和数组库 xp，获取特定的数组特殊函数
def get_array_special_func(f_name, xp, n_array_args):
    # 根据数组库确定 scipy 命名空间
    spx = scipy_namespace_for(xp)
    f = None
    # 如果数组库是 NumPy，则尝试从 _ufuncs 中获取对应函数
    if is_numpy(xp):
        f = getattr(_ufuncs, f_name, None)
    # 如果是 SciPy 或者有相应的命名空间，尝试从 spx.special 中获取函数
    elif spx is not None:
        f = getattr(spx.special, f_name, None)

    # 如果成功获取到函数，返回该函数
    if f is not None:
        return f

    # 如果存在通用的数组 API 实现，并且 spx 不为空，尝试使用通用实现
    if f_name in _generic_implementations and spx is not None:
        _f = _generic_implementations[f_name](xp=xp, spx=spx)
        if _f is not None:
            return _f

    # 如果以上方法均未成功，最后尝试从 _ufuncs 中获取函数
    _f = getattr(_ufuncs, f_name, None)
    # 定义一个新的函数，用于处理数组参数和其他参数
    def f(*args, _f=_f, _xp=xp, **kwargs):
        # 将数组参数转换为 NumPy 数组
        array_args = args[:n_array_args]
        other_args = args[n_array_args:]
        array_args = [np.asarray(arg) for arg in array_args]
        # 调用原始函数处理数据
        out = _f(*array_args, *other_args, **kwargs)
        # 将结果转换为 xp 对应的数组类型
        return _xp.asarray(out)

    return f

# 辅助函数，用于获取参数数组的形状和数据类型
def _get_shape_dtype(*args, xp):
    # 广播所有输入数组，使其形状一致
    args = xp.broadcast_arrays(*args)
    shape = args[0].shape
    # 确定输出数据类型
    dtype = xp.result_type(*args)
    # 如果数据类型是整数类型，转换为浮点型
    if xp.isdtype(dtype, 'integral'):
        dtype = xp.float64
        args = [xp.asarray(arg, dtype=dtype) for arg in args]
    return args, shape, dtype

# 定义相对熵函数，针对不同的数组库进行适配
def _rel_entr(xp, spx):
    def __rel_entr(x, y, *, xp=xp):
        # 获取参数数组的形状和数据类型
        args, shape, dtype = _get_shape_dtype(x, y, xp=xp)
        x, y = args
        # 创建与 x 形状相同的结果数组，并初始化为无穷大
        res = xp.full(x.shape, xp.inf, dtype=dtype)
        # 根据 x 和 y 的取值情况计算相对熵
        res[(x == 0) & (y >= 0)] = xp.asarray(0, dtype=dtype)
        i = (x > 0) & (y > 0)
        res[i] = x[i] * (xp.log(x[i]) - xp.log(y[i]))
        return res
    return __rel_entr

# 定义 xlogy 函数，处理数值的乘积与对数
def _xlogy(xp, spx):
    def __xlogy(x, y, *, xp=xp):
        # 忽略除零和无效操作的浮点数错误
        with np.errstate(divide='ignore', invalid='ignore'):
            temp = x * xp.log(y)
        # 当 x 为零时，返回零；否则返回计算结果
        return xp.where(x == 0., xp.asarray(0., dtype=temp.dtype), temp)
    return __xlogy

# 定义 chdtr 函数，与 gammainc 的不同之处在于当 gammainc 未找到时，返回 None
def _chdtr(xp, spx):
    # 尝试从 spx.special 中获取 gammainc 函数
    gammainc = getattr(spx.special, 'gammainc', None)  # noqa: F811
    if gammainc is None and hasattr(xp, 'special'):
        gammainc = getattr(xp.special, 'gammainc', None)
    # 如果 gammainc 未找到，返回 None
    if gammainc is None:
        return None
    # 定义一个内部函数 __chdtr，接受参数 v 和 x
    def __chdtr(v, x):
        # 使用 NumPy 的 where 函数根据条件计算结果 res
        res = xp.where(x >= 0, gammainc(v/2, x/2), 0)
        # 检查并标记出特定的 NaN（Not a Number）情况，将结果中对应位置设为 NaN
        i_nan = ((x == 0) & (v == 0)) | xp.isnan(x) | xp.isnan(v)
        res = xp.where(i_nan, xp.nan, res)
        # 返回计算后的结果 res
        return res
    # 返回内部函数 __chdtr 作为最终结果
    return __chdtr
# 定义一个函数 _chdtrc，用于计算特定的统计函数
def _chdtrc(xp, spx):
    # 从 spx.special 中获取 gammaincc 函数，如果不存在则为 None
    gammaincc = getattr(spx.special, 'gammaincc', None)  # noqa: F811
    # 如果 gammaincc 为 None 并且 xp 具有 special 属性，则尝试从 xp.special 中获取
    if gammaincc is None and hasattr(xp, 'special'):
        gammaincc = getattr(xp.special, 'gammaincc', None)
    # 如果 gammaincc 仍然为 None，则返回 None
    if gammaincc is None:
        return None
    
    # 定义内部函数 __chdtrc，计算 chdtrc 函数的具体实现
    def __chdtrc(v, x):
        # 使用 xp.where 进行条件判断和操作，计算 gammaincc(v/2, x/2) 或者 1
        res = xp.where(x >= 0, gammaincc(v/2, x/2), 1)
        # 检查是否出现 NaN 值或非法的参数，将结果设为 NaN
        i_nan = ((x == 0) & (v == 0)) | xp.isnan(x) | xp.isnan(v) | (v <= 0)
        res = xp.where(i_nan, xp.nan, res)
        return res
    
    # 返回内部函数 __chdtrc
    return __chdtrc


# 定义一个函数 _betaincc，用于计算特定的统计函数
def _betaincc(xp, spx):
    # 从 spx.special 中获取 betainc 函数，如果不存在则为 None
    betainc = getattr(spx.special, 'betainc', None)  # noqa: F811
    # 如果 betainc 为 None 并且 xp 具有 special 属性，则尝试从 xp.special 中获取
    if betainc is None and hasattr(xp, 'special'):
        betainc = getattr(xp.special, 'betainc', None)
    # 如果 betainc 仍然为 None，则返回 None
    if betainc is None:
        return None
    
    # 定义内部函数 __betaincc，计算 betaincc 函数的具体实现
    def __betaincc(a, b, x):
        # 返回 betainc(b, a, 1-x) 的计算结果，不完美的实现
        return betainc(b, a, 1-x)
    
    # 返回内部函数 __betaincc
    return __betaincc


# 定义一个函数 _stdtr，用于计算特定的统计函数
def _stdtr(xp, spx):
    # 从 spx.special 中获取 betainc 函数，如果不存在则为 None
    betainc = getattr(spx.special, 'betainc', None)  # noqa: F811
    # 如果 betainc 为 None 并且 xp 具有 special 属性，则尝试从 xp.special 中获取
    if betainc is None and hasattr(xp, 'special'):
        betainc = getattr(xp.special, 'betainc', None)
    # 如果 betainc 仍然为 None，则返回 None
    if betainc is None:
        return None
    
    # 定义内部函数 __stdtr，计算 stdtr 函数的具体实现
    def __stdtr(df, t):
        # 计算 x = df / (t ** 2 + df)
        x = df / (t ** 2 + df)
        # 计算尾部概率，使用 betainc(df / 2, 0.5, x) / 2
        tail = betainc(df / 2, 0.5, x) / 2
        # 使用 xp.where 进行条件判断和操作，计算尾部概率或者 1 - 尾部概率
        return xp.where(t < 0, tail, 1 - tail)
    
    # 返回内部函数 __stdtr
    return __stdtr


# 定义一个字典 _generic_implementations，包含特定函数名称到对应实现函数的映射
_generic_implementations = {'rel_entr': _rel_entr,
                            'xlogy': _xlogy,
                            'chdtr,': _chdtr,  # 注意这里可能是个笔误，应为 'chdtr'
                            'chdtrc': _chdtrc,
                            'betaincc': _betaincc,
                            'stdtr': _stdtr,
                            }


# 定义一个函数 support_alternative_backends，用于支持不同后端的统计函数计算
def support_alternative_backends(f_name, n_array_args):
    # 从 _ufuncs 模块中获取指定名称的函数对象 func
    func = getattr(_ufuncs, f_name)
    
    # 定义内部函数 wrapped，作为 f_name 函数的包装器
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        # 根据前 n_array_args 个参数创建 array_namespace
        xp = array_namespace(*args[:n_array_args])
        # 获取对应函数的特定实现 f
        f = get_array_special_func(f_name, xp, n_array_args)
        # 调用特定实现函数 f，并返回其结果
        return f(*args, **kwargs)
    
    # 返回包装器函数 wrapped
    return wrapped


# 定义一个字典 array_special_func_map，包含特定函数名称到其参数个数的映射关系
array_special_func_map = {
    'log_ndtr': 1,
    'ndtr': 1,
    'ndtri': 1,
    'erf': 1,
    'erfc': 1,
    'i0': 1,
    'i0e': 1,
    'i1': 1,
    'i1e': 1,
    'gammaln': 1,
    'gammainc': 2,
    'gammaincc': 2,
    'logit': 1,
    'expit': 1,
    'entr': 1,
    'rel_entr': 2,
    'xlogy': 2,
    'chdtr': 2,
    'chdtrc': 2,
    'betainc': 3,
    'betaincc': 3,
    'stdtr': 2,
}

# 遍历 array_special_func_map 中的每个函数名称及其参数个数
for f_name, n_array_args in array_special_func_map.items():
    # 根据条件选择不同的函数实现方式，并赋值给 f
    f = (support_alternative_backends(f_name, n_array_args) if _SCIPY_ARRAY_API
         else getattr(_ufuncs, f_name))
    # 将函数 f 添加到当前模块的全局命名空间中，使用函数名称作为键
    sys.modules[__name__].__dict__[f_name] = f

# 定义一个列表 __all__，包含所有在 array_special_func_map 中的函数名称
__all__ = list(array_special_func_map)
```