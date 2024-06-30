# `D:\src\scipysrc\scipy\scipy\fft\_pocketfft\basic.py`

```
"""
Discrete Fourier Transforms - basic.py
"""
import numpy as np  # 导入 NumPy 库，用于数值计算
import functools  # 导入 functools 库，用于创建偏函数
from . import pypocketfft as pfft  # 导入 pypocketfft 库中的 pfft 模块
from .helper import (_asfarray, _init_nd_shape_and_axes, _datacopied,
                     _fix_shape, _fix_shape_1d, _normalization,
                     _workers)  # 从 helper 模块中导入多个辅助函数

def c2c(forward, x, n=None, axis=-1, norm=None, overwrite_x=False,
        workers=None, *, plan=None):
    """ Return discrete Fourier transform of real or complex sequence. """
    if plan is not None:
        # 如果指定了 plan 参数，抛出 NotImplementedError 异常
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    tmp = _asfarray(x)  # 将输入数据 x 转换为浮点数组
    overwrite_x = overwrite_x or _datacopied(tmp, x)  # 检查是否复制了输入数据 x
    norm = _normalization(norm, forward)  # 根据 forward 参数确定规范化方式
    workers = _workers(workers)  # 根据 workers 参数设置并行工作的线程数

    if n is not None:
        # 如果指定了 n，调整数据的形状以匹配 n
        tmp, copied = _fix_shape_1d(tmp, n, axis)
        overwrite_x = overwrite_x or copied  # 标记是否覆盖了输入数据 x
    elif tmp.shape[axis] < 1:
        # 如果数据点数小于 1，抛出 ValueError 异常
        message = f"invalid number of data points ({tmp.shape[axis]}) specified"
        raise ValueError(message)

    out = (tmp if overwrite_x and tmp.dtype.kind == 'c' else None)

    return pfft.c2c(tmp, (axis,), forward, norm, out, workers)  # 调用 pfft 模块中的 c2c 函数进行变换


fft = functools.partial(c2c, True)  # 创建快速傅立叶变换的偏函数 fft
fft.__name__ = 'fft'  # 设置偏函数的名称为 'fft'
ifft = functools.partial(c2c, False)  # 创建逆快速傅立叶变换的偏函数 ifft
ifft.__name__ = 'ifft'  # 设置偏函数的名称为 'ifft'


def r2c(forward, x, n=None, axis=-1, norm=None, overwrite_x=False,
        workers=None, *, plan=None):
    """
    Discrete Fourier transform of a real sequence.
    """
    if plan is not None:
        # 如果指定了 plan 参数，抛出 NotImplementedError 异常
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    tmp = _asfarray(x)  # 将输入数据 x 转换为浮点数组
    norm = _normalization(norm, forward)  # 根据 forward 参数确定规范化方式
    workers = _workers(workers)  # 根据 workers 参数设置并行工作的线程数

    if not np.isrealobj(tmp):
        # 如果输入数据不是实数对象，抛出 TypeError 异常
        raise TypeError("x must be a real sequence")

    if n is not None:
        # 如果指定了 n，调整数据的形状以匹配 n
        tmp, _ = _fix_shape_1d(tmp, n, axis)
    elif tmp.shape[axis] < 1:
        # 如果数据点数小于 1，抛出 ValueError 异常
        raise ValueError(f"invalid number of data points ({tmp.shape[axis]}) specified")

    # 注意: overwrite_x 参数在这里未被使用
    return pfft.r2c(tmp, (axis,), forward, norm, None, workers)  # 调用 pfft 模块中的 r2c 函数进行变换


rfft = functools.partial(r2c, True)  # 创建实数输入快速傅立叶变换的偏函数 rfft
rfft.__name__ = 'rfft'  # 设置偏函数的名称为 'rfft'
ihfft = functools.partial(r2c, False)  # 创建实数输入逆快速傅立叶变换的偏函数 ihfft
ihfft.__name__ = 'ihfft'  # 设置偏函数的名称为 'ihfft'


def c2r(forward, x, n=None, axis=-1, norm=None, overwrite_x=False,
        workers=None, *, plan=None):
    """
    Return inverse discrete Fourier transform of real sequence x.
    """
    if plan is not None:
        # 如果指定了 plan 参数，抛出 NotImplementedError 异常
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    tmp = _asfarray(x)  # 将输入数据 x 转换为浮点数组
    norm = _normalization(norm, forward)  # 根据 forward 参数确定规范化方式
    workers = _workers(workers)  # 根据 workers 参数设置并行工作的线程数

    # TODO: Optimize for hermitian and real?
    if np.isrealobj(tmp):
        tmp = tmp + 0.j  # 如果输入数据为实数对象，转换为复数对象

    # 最后一个轴利用 Hermitian 对称性
    if n is None:
        n = (tmp.shape[axis] - 1) * 2
        if n < 1:
            # 如果数据点数小于 1，抛出 ValueError 异常
            raise ValueError(f"Invalid number of data points ({n}) specified")
    else:
        # 调用 _fix_shape_1d 函数，将 tmp 调整为一维数组形状
        tmp, _ = _fix_shape_1d(tmp, (n//2) + 1, axis)

    # 注意：overwrite_x 参数未被使用，可以忽略
    # 使用 pfft.c2r 进行复数到实数的转换
    return pfft.c2r(tmp, (axis,), n, forward, norm, None, workers)
hfft = functools.partial(c2r, True)
# 创建一个偏函数 hfft，用于执行 c2r 函数并指定 forward 参数为 True
hfft.__name__ = 'hfft'
# 设置偏函数 hfft 的名称为 'hfft'

irfft = functools.partial(c2r, False)
# 创建一个偏函数 irfft，用于执行 c2r 函数并指定 forward 参数为 False
irfft.__name__ = 'irfft'
# 设置偏函数 irfft 的名称为 'irfft'


def hfft2(x, s=None, axes=(-2,-1), norm=None, overwrite_x=False, workers=None,
          *, plan=None):
    """
    2-D discrete Fourier transform of a Hermitian sequence
    """
    if plan is not None:
        # 如果传入了 plan 参数，则抛出 NotImplementedError 异常
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    return hfftn(x, s, axes, norm, overwrite_x, workers)


def ihfft2(x, s=None, axes=(-2,-1), norm=None, overwrite_x=False, workers=None,
           *, plan=None):
    """
    2-D discrete inverse Fourier transform of a Hermitian sequence
    """
    if plan is not None:
        # 如果传入了 plan 参数，则抛出 NotImplementedError 异常
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    return ihfftn(x, s, axes, norm, overwrite_x, workers)


def c2cn(forward, x, s=None, axes=None, norm=None, overwrite_x=False,
         workers=None, *, plan=None):
    """
    Return multidimensional discrete Fourier transform.
    """
    if plan is not None:
        # 如果传入了 plan 参数，则抛出 NotImplementedError 异常
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    tmp = _asfarray(x)

    shape, axes = _init_nd_shape_and_axes(tmp, s, axes)
    # 初始化输入数组的形状和轴

    overwrite_x = overwrite_x or _datacopied(tmp, x)
    workers = _workers(workers)

    if len(axes) == 0:
        return x

    tmp, copied = _fix_shape(tmp, shape, axes)
    # 根据形状和轴修正输入数组的形状
    overwrite_x = overwrite_x or copied

    norm = _normalization(norm, forward)
    out = (tmp if overwrite_x and tmp.dtype.kind == 'c' else None)
    # 如果 overwrite_x 为 True 并且输入数组的类型为复数，输出结果存储在 tmp 中

    return pfft.c2c(tmp, axes, forward, norm, out, workers)


fftn = functools.partial(c2cn, True)
# 创建一个偏函数 fftn，用于执行 c2cn 函数并指定 forward 参数为 True
fftn.__name__ = 'fftn'
# 设置偏函数 fftn 的名称为 'fftn'

ifftn = functools.partial(c2cn, False)
# 创建一个偏函数 ifftn，用于执行 c2cn 函数并指定 forward 参数为 False
ifftn.__name__ = 'ifftn'


def r2cn(forward, x, s=None, axes=None, norm=None, overwrite_x=False,
         workers=None, *, plan=None):
    """Return multidimensional discrete Fourier transform of real input"""
    if plan is not None:
        # 如果传入了 plan 参数，则抛出 NotImplementedError 异常
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    tmp = _asfarray(x)

    if not np.isrealobj(tmp):
        # 如果输入数组不是实数类型，则抛出 TypeError 异常
        raise TypeError("x must be a real sequence")

    shape, axes = _init_nd_shape_and_axes(tmp, s, axes)
    tmp, _ = _fix_shape(tmp, shape, axes)
    # 根据形状和轴修正输入数组的形状
    norm = _normalization(norm, forward)
    workers = _workers(workers)

    if len(axes) == 0:
        # 如果轴的数量为零，则抛出 ValueError 异常
        raise ValueError("at least 1 axis must be transformed")

    # Note: overwrite_x is not utilized
    # 注意：overwrite_x 参数在此处未被使用
    return pfft.r2c(tmp, axes, forward, norm, None, workers)


rfftn = functools.partial(r2cn, True)
# 创建一个偏函数 rfftn，用于执行 r2cn 函数并指定 forward 参数为 True
rfftn.__name__ = 'rfftn'
# 设置偏函数 rfftn 的名称为 'rfftn'

ihfftn = functools.partial(r2cn, False)
# 创建一个偏函数 ihfftn，用于执行 r2cn 函数并指定 forward 参数为 False
ihfftn.__name__ = 'ihfftn'
# 设置偏函数 ihfftn 的名称为 'ihfftn'


def c2rn(forward, x, s=None, axes=None, norm=None, overwrite_x=False,
         workers=None, *, plan=None):
    """Multidimensional inverse discrete fourier transform with real output"""
    # 如果预先计划不为空，则抛出未实现的错误，因为 scipy.fft 函数暂不支持传递预计算的计划
    if plan is not None:
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    # 将输入 x 转换为浮点数组 tmp
    tmp = _asfarray(x)

    # TODO: Optimize for hermitian and real?
    # 如果 tmp 是实数组对象，则将其转换为包含虚部的复数数组
    if np.isrealobj(tmp):
        tmp = tmp + 0.j

    # 判断是否没有给定输出形状 s
    noshape = s is None
    # 初始化转换的形状和轴
    shape, axes = _init_nd_shape_and_axes(tmp, s, axes)

    # 至少需要一个轴进行变换，否则抛出值错误
    if len(axes) == 0:
        raise ValueError("at least 1 axis must be transformed")

    # 将形状转换为列表以便后续修改
    shape = list(shape)
    # 如果没有给定输出形状，则根据最后一个轴的大小设置输出形状
    if noshape:
        shape[-1] = (x.shape[axes[-1]] - 1) * 2

    # 根据前向或反向变换规范化常数
    norm = _normalization(norm, forward)
    # 获取处理工作的线程数
    workers = _workers(workers)

    # 最后一个轴利用厄米特对称性
    lastsize = shape[-1]
    shape[-1] = (shape[-1] // 2) + 1

    # 调整 tmp 的形状以匹配变换所需的形状
    tmp, _ = tuple(_fix_shape(tmp, shape, axes))

    # 注意: overwrite_x 参数未被使用
    # 返回傅里叶变换的实部结果，根据是否是前向变换确定具体实现
    return pfft.c2r(tmp, axes, lastsize, forward, norm, None, workers)
# 使用 functools.partial 创建一个偏函数 hfftn，调用 c2rn 函数，传入 True 作为第一个参数
hfftn = functools.partial(c2rn, True)
# 将偏函数的 __name__ 属性设置为 'hfftn'
hfftn.__name__ = 'hfftn'

# 使用 functools.partial 创建一个偏函数 irfftn，调用 c2rn 函数，传入 False 作为第一个参数
irfftn = functools.partial(c2rn, False)
# 将偏函数的 __name__ 属性设置为 'irfftn'
irfftn.__name__ = 'irfftn'


def r2r_fftpack(forward, x, n=None, axis=-1, norm=None, overwrite_x=False):
    """FFT of a real sequence, returning fftpack half complex format"""
    # 将 x 转换为浮点数组，存储在 tmp 中
    tmp = _asfarray(x)
    # 根据 overwrite_x 的值或者数据是否复制过来判断是否覆盖原始数据
    overwrite_x = overwrite_x or _datacopied(tmp, x)
    # 根据 forward 和 norm 参数获取正规化后的值
    norm = _normalization(norm, forward)
    # 获取工作线程数
    workers = _workers(None)

    # 如果 tmp 的数据类型为复数，则抛出 TypeError 异常
    if tmp.dtype.kind == 'c':
        raise TypeError('x must be a real sequence')

    # 如果 n 参数不为 None，则根据指定的 n 和 axis 修正 tmp 的形状
    if n is not None:
        tmp, copied = _fix_shape_1d(tmp, n, axis)
        overwrite_x = overwrite_x or copied
    # 如果指定 axis 上的数据点数小于 1，则抛出 ValueError 异常
    elif tmp.shape[axis] < 1:
        raise ValueError(f"invalid number of data points ({tmp.shape[axis]}) specified")

    # 如果 overwrite_x 为 True，则 out 等于 tmp；否则 out 为 None
    out = (tmp if overwrite_x else None)

    # 调用 pfft.r2r_fftpack 函数进行实数序列的 FFT 转换，返回结果
    return pfft.r2r_fftpack(tmp, (axis,), forward, forward, norm, out, workers)


# 使用 functools.partial 创建一个偏函数 rfft_fftpack，调用 r2r_fftpack 函数，传入 True 作为第一个参数
rfft_fftpack = functools.partial(r2r_fftpack, True)
# 将偏函数的 __name__ 属性设置为 'rfft_fftpack'
rfft_fftpack.__name__ = 'rfft_fftpack'

# 使用 functools.partial 创建一个偏函数 irfft_fftpack，调用 r2r_fftpack 函数，传入 False 作为第一个参数
irfft_fftpack = functools.partial(r2r_fftpack, False)
# 将偏函数的 __name__ 属性设置为 'irfft_fftpack'
irfft_fftpack.__name__ = 'irfft_fftpack'
```