# `D:\src\scipysrc\scipy\scipy\fft\_basic_backend.py`

```
# 从 scipy._lib._array_api 导入指定的符号
# array_namespace 用于获取 x 的数组命名空间，is_numpy 用于检查是否为 NumPy 数组
# xp_unsupported_param_msg 提供关于不支持的参数的错误信息，is_complex 用于检查是否为复数
from scipy._lib._array_api import (
    array_namespace, is_numpy, xp_unsupported_param_msg, is_complex
)
# 从当前包中导入 _pocketfft 模块
from . import _pocketfft
# 导入 NumPy 库并命名为 np
import numpy as np


# 函数用于验证 FFT 相关参数，如果有不支持的参数则引发 ValueError
def _validate_fft_args(workers, plan, norm):
    # 如果 workers 参数不为空，则引发 ValueError 异常，指出 workers 参数不被支持
    if workers is not None:
        raise ValueError(xp_unsupported_param_msg("workers"))
    # 如果 plan 参数不为空，则引发 ValueError 异常，指出 plan 参数不被支持
    if plan is not None:
        raise ValueError(xp_unsupported_param_msg("plan"))
    # 如果 norm 参数为空，则将其设为 'backward'
    if norm is None:
        norm = 'backward'
    # 返回更新后的 norm 参数
    return norm


# 函数说明：
# 当 SCIPY_ARRAY_API 未设置或 x 是 NumPy 数组或类似数组时，使用 pocketfft。
# 当设置了 SCIPY_ARRAY_API 时，尝试使用 xp.fft 处理 CuPy 数组、PyTorch 数组和其他支持标准数组 API 的对象。
# 如果 xp.fft 不存在，则尝试转换为 NumPy 数组再使用 pocketfft。
# 函数 _execute_1D 用于执行一维 FFT 操作。
def _execute_1D(func_str, pocketfft_func, x, n, axis, norm, overwrite_x, workers, plan):
    # 获取 x 的数组命名空间
    xp = array_namespace(x)

    # 如果 x 是 NumPy 数组
    if is_numpy(xp):
        # 调用 pocketfft_func 执行 FFT 操作并返回结果
        return pocketfft_func(x, n=n, axis=axis, norm=norm,
                              overwrite_x=overwrite_x, workers=workers, plan=plan)

    # 验证 FFT 相关参数，更新 norm 参数
    norm = _validate_fft_args(workers, plan, norm)
    
    # 如果 xp 对象具有 'fft' 属性
    if hasattr(xp, 'fft'):
        # 获取 xp.fft 中指定名称的函数对象
        xp_func = getattr(xp.fft, func_str)
        # 调用 xp_func 执行 FFT 操作并返回结果
        return xp_func(x, n=n, axis=axis, norm=norm)

    # 将 x 转换为 NumPy 数组
    x = np.asarray(x)
    # 使用 pocketfft_func 执行 FFT 操作并返回结果，将结果转换为 xp 对应的数组类型
    y = pocketfft_func(x, n=n, axis=axis, norm=norm)
    return xp.asarray(y)


# 函数 _execute_nD 用于执行多维 FFT 操作。
def _execute_nD(func_str, pocketfft_func, x, s, axes, norm, overwrite_x, workers, plan):
    # 获取 x 的数组命名空间
    xp = array_namespace(x)
    
    # 如果 x 是 NumPy 数组
    if is_numpy(xp):
        # 调用 pocketfft_func 执行 FFT 操作并返回结果
        return pocketfft_func(x, s=s, axes=axes, norm=norm,
                              overwrite_x=overwrite_x, workers=workers, plan=plan)

    # 验证 FFT 相关参数，更新 norm 参数
    norm = _validate_fft_args(workers, plan, norm)
    
    # 如果 xp 对象具有 'fft' 属性
    if hasattr(xp, 'fft'):
        # 获取 xp.fft 中指定名称的函数对象
        xp_func = getattr(xp.fft, func_str)
        # 调用 xp_func 执行 FFT 操作并返回结果
        return xp_func(x, s=s, axes=axes, norm=norm)

    # 将 x 转换为 NumPy 数组
    x = np.asarray(x)
    # 使用 pocketfft_func 执行 FFT 操作并返回结果，将结果转换为 xp 对应的数组类型
    y = pocketfft_func(x, s=s, axes=axes, norm=norm)
    return xp.asarray(y)


# 函数 fft 执行一维快速傅里叶变换（FFT）操作。
def fft(x, n=None, axis=-1, norm=None,
        overwrite_x=False, workers=None, *, plan=None):
    return _execute_1D('fft', _pocketfft.fft, x, n=n, axis=axis, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


# 函数 ifft 执行一维逆快速傅里叶变换（IFFT）操作。
def ifft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
         plan=None):
    return _execute_1D('ifft', _pocketfft.ifft, x, n=n, axis=axis, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


# 函数 rfft 执行一维实数快速傅里叶变换（RFFT）操作。
def rfft(x, n=None, axis=-1, norm=None,
         overwrite_x=False, workers=None, *, plan=None):
    return _execute_1D('rfft', _pocketfft.rfft, x, n=n, axis=axis, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


# 函数 irfft 执行一维实数逆快速傅里叶变换（IRFFT）操作。
def irfft(x, n=None, axis=-1, norm=None,
          overwrite_x=False, workers=None, *, plan=None):
    return _execute_1D('irfft', _pocketfft.irfft, x, n=n, axis=axis, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


# 函数 hfft 执行一维 Hermite 快速傅里叶变换（HFFT）操作。
def hfft(x, n=None, axis=-1, norm=None,
         overwrite_x=False, workers=None, *, plan=None):
    # 调用执行一维高速傅里叶变换（HFFT）的函数，并返回结果
    return _execute_1D('hfft', _pocketfft.hfft, x, n=n, axis=axis, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)
# 执行逆离散傅立叶变换（IDFT）的功能，使用 _pocketfft 库中的 ihfft 函数
def ihfft(x, n=None, axis=-1, norm=None,
          overwrite_x=False, workers=None, *, plan=None):
    return _execute_1D('ihfft', _pocketfft.ihfft, x, n=n, axis=axis, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


# 执行多维傅立叶变换（NDFT）的功能，使用 _pocketfft 库中的 fftn 函数
def fftn(x, s=None, axes=None, norm=None,
         overwrite_x=False, workers=None, *, plan=None):
    return _execute_nD('fftn', _pocketfft.fftn, x, s=s, axes=axes, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


# 执行多维逆傅立叶变换（INDFT）的功能，使用 _pocketfft 库中的 ifftn 函数
def ifftn(x, s=None, axes=None, norm=None,
          overwrite_x=False, workers=None, *, plan=None):
    return _execute_nD('ifftn', _pocketfft.ifftn, x, s=s, axes=axes, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


# 执行二维傅立叶变换（2D DFT）的功能，调用 fftn 函数进行实现
def fft2(x, s=None, axes=(-2, -1), norm=None,
         overwrite_x=False, workers=None, *, plan=None):
    return fftn(x, s, axes, norm, overwrite_x, workers, plan=plan)


# 执行二维逆傅立叶变换（2D INDFT）的功能，调用 ifftn 函数进行实现
def ifft2(x, s=None, axes=(-2, -1), norm=None,
          overwrite_x=False, workers=None, *, plan=None):
    return ifftn(x, s, axes, norm, overwrite_x, workers, plan=plan)


# 执行实部的多维傅立叶变换（Real FFT），使用 _pocketfft 库中的 rfftn 函数
def rfftn(x, s=None, axes=None, norm=None,
          overwrite_x=False, workers=None, *, plan=None):
    return _execute_nD('rfftn', _pocketfft.rfftn, x, s=s, axes=axes, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


# 执行实部的二维傅立叶变换（2D Real FFT），调用 rfftn 函数进行实现
def rfft2(x, s=None, axes=(-2, -1), norm=None,
          overwrite_x=False, workers=None, *, plan=None):
    return rfftn(x, s, axes, norm, overwrite_x, workers, plan=plan)


# 执行实部的多维逆傅立叶变换（Inverse Real FFT），使用 _pocketfft 库中的 irfftn 函数
def irfftn(x, s=None, axes=None, norm=None,
           overwrite_x=False, workers=None, *, plan=None):
    return _execute_nD('irfftn', _pocketfft.irfftn, x, s=s, axes=axes, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


# 执行实部的二维逆傅立叶变换（2D Inverse Real FFT），调用 irfftn 函数进行实现
def irfft2(x, s=None, axes=(-2, -1), norm=None,
           overwrite_x=False, workers=None, *, plan=None):
    return irfftn(x, s, axes, norm, overwrite_x, workers, plan=plan)


# 根据给定的 norm 值确定变换的方向（正向或反向）
def _swap_direction(norm):
    if norm in (None, 'backward'):
        norm = 'forward'
    elif norm == 'forward':
        norm = 'backward'
    elif norm != 'ortho':
        raise ValueError('Invalid norm value %s; should be "backward", '
                         '"ortho", or "forward".' % norm)
    return norm


# 执行 Hermite 函数的多维傅立叶变换（Hermite Function FFT），根据输入数据类型选择合适的函数进行变换
def hfftn(x, s=None, axes=None, norm=None,
          overwrite_x=False, workers=None, *, plan=None):
    xp = array_namespace(x)
    if is_numpy(xp):  # 如果输入是 numpy 数组，直接调用 _pocketfft 库中的 hfftn 函数
        return _pocketfft.hfftn(x, s, axes, norm, overwrite_x, workers, plan=plan)
    if is_complex(x, xp):  # 如果输入是复数，对其共轭并调用逆实部傅立叶变换 irfftn 函数
        x = xp.conj(x)
    return irfftn(x, s, axes, _swap_direction(norm),
                  overwrite_x, workers, plan=plan)


# 执行 Hermite 函数的二维傅立叶变换（2D Hermite Function FFT），调用 hfftn 函数进行实现
def hfft2(x, s=None, axes=(-2, -1), norm=None,
          overwrite_x=False, workers=None, *, plan=None):
    return hfftn(x, s, axes, norm, overwrite_x, workers, plan=plan)


# 执行 Hermite 函数的多维逆傅立叶变换（Inverse Hermite Function FFT），暂未实现
def ihfftn(x, s=None, axes=None, norm=None,
           overwrite_x=False, workers=None, *, plan=None):
    # 待实现
    # 将输入数组 x 转换为适合当前命名空间的数组类型 xp
    xp = array_namespace(x)
    # 检查 xp 是否为 NumPy 数组
    if is_numpy(xp):
        # 如果是 NumPy 数组，则使用 _pocketfft 库中的 ihfftn 函数进行逆高速 Fourier 变换
        return _pocketfft.ihfftn(x, s, axes, norm, overwrite_x, workers, plan=plan)
    # 如果不是 NumPy 数组，则使用 xp 的共轭进行实数快速 Fourier 变换
    return xp.conj(rfftn(x, s, axes, _swap_direction(norm),
                         overwrite_x, workers, plan=plan))
# 定义一个函数 `ihfft2`，用于执行反变换到域的高速傅里叶变换（IHFFT）的二维版本。
def ihfft2(x, s=None, axes=(-2, -1), norm=None,
           overwrite_x=False, workers=None, *, plan=None):
    # 调用 `ihfftn` 函数，传递参数 `x` 作为输入，执行高速傅里叶反变换到域操作。
    # 其余参数 `s`, `axes`, `norm`, `overwrite_x`, `workers` 被传递给 `ihfftn` 函数。
    return ihfftn(x, s, axes, norm, overwrite_x, workers, plan=plan)
```