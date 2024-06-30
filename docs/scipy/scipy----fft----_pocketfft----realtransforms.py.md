# `D:\src\scipysrc\scipy\scipy\fft\_pocketfft\realtransforms.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from . import pypocketfft as pfft  # 导入 pypocketfft 库中的 pfft 模块
from .helper import (_asfarray, _init_nd_shape_and_axes, _datacopied,
                     _fix_shape, _fix_shape_1d, _normalization, _workers)  # 从 helper 模块导入多个函数
import functools  # 导入 functools 模块，用于函数式编程

def _r2r(forward, transform, x, type=2, n=None, axis=-1, norm=None,
         overwrite_x=False, workers=None, orthogonalize=None):
    """Forward or backward 1-D DCT/DST

    Parameters
    ----------
    forward : bool
        Transform direction (determines type and normalisation)
    transform : {pypocketfft.dct, pypocketfft.dst}
        The transform to perform
    """
    tmp = _asfarray(x)  # 将输入 x 转换为浮点数组
    overwrite_x = overwrite_x or _datacopied(tmp, x)  # 根据条件判断是否覆盖输入 x
    norm = _normalization(norm, forward)  # 根据转换方向确定归一化方式
    workers = _workers(workers)  # 获取工作线程数

    if not forward:
        if type == 2:
            type = 3
        elif type == 3:
            type = 2

    if n is not None:
        tmp, copied = _fix_shape_1d(tmp, n, axis)  # 调整输入 tmp 的形状以匹配指定的维度 n
        overwrite_x = overwrite_x or copied  # 根据情况更新覆盖标志
    elif tmp.shape[axis] < 1:
        raise ValueError(f"invalid number of data points ({tmp.shape[axis]}) specified")  # 如果指定维度小于1，抛出值错误异常

    out = (tmp if overwrite_x else None)  # 如果覆盖标志为真，则输出 tmp，否则为 None

    # 对于复数输入，分别处理实部和虚部
    if np.iscomplexobj(x):
        out = np.empty_like(tmp) if out is None else out  # 如果输出为空，则创建与 tmp 相同形状的空数组
        transform(tmp.real, type, (axis,), norm, out.real, workers)  # 对实部进行变换
        transform(tmp.imag, type, (axis,), norm, out.imag, workers)  # 对虚部进行变换
        return out  # 返回处理后的数组

    return transform(tmp, type, (axis,), norm, out, workers, orthogonalize)  # 否则，直接进行变换


dct = functools.partial(_r2r, True, pfft.dct)  # 创建偏函数 dct，指定为正向 DCT 变换
dct.__name__ = 'dct'  # 设置偏函数的名称为 'dct'
idct = functools.partial(_r2r, False, pfft.dct)  # 创建偏函数 idct，指定为反向 DCT 变换
idct.__name__ = 'idct'  # 设置偏函数的名称为 'idct'

dst = functools.partial(_r2r, True, pfft.dst)  # 创建偏函数 dst，指定为正向 DST 变换
dst.__name__ = 'dst'  # 设置偏函数的名称为 'dst'
idst = functools.partial(_r2r, False, pfft.dst)  # 创建偏函数 idst，指定为反向 DST 变换
idst.__name__ = 'idst'  # 设置偏函数的名称为 'idst'


def _r2rn(forward, transform, x, type=2, s=None, axes=None, norm=None,
          overwrite_x=False, workers=None, orthogonalize=None):
    """Forward or backward nd DCT/DST

    Parameters
    ----------
    forward : bool
        Transform direction (determines type and normalisation)
    transform : {pypocketfft.dct, pypocketfft.dst}
        The transform to perform
    """
    tmp = _asfarray(x)  # 将输入 x 转换为浮点数组

    shape, axes = _init_nd_shape_and_axes(tmp, s, axes)  # 初始化输入数据的形状和轴列表
    overwrite_x = overwrite_x or _datacopied(tmp, x)  # 根据条件判断是否覆盖输入 x

    if len(axes) == 0:
        return x  # 如果轴列表为空，直接返回输入 x

    tmp, copied = _fix_shape(tmp, shape, axes)  # 调整输入 tmp 的形状以匹配指定的形状和轴
    overwrite_x = overwrite_x or copied  # 根据情况更新覆盖标志

    if not forward:
        if type == 2:
            type = 3
        elif type == 3:
            type = 2

    norm = _normalization(norm, forward)  # 根据转换方向确定归一化方式
    workers = _workers(workers)  # 获取工作线程数
    out = (tmp if overwrite_x else None)  # 如果覆盖标志为真，则输出 tmp，否则为 None

    # 对于复数输入，分别处理实部和虚部
    if np.iscomplexobj(x):
        out = np.empty_like(tmp) if out is None else out  # 如果输出为空，则创建与 tmp 相同形状的空数组
        transform(tmp.real, type, axes, norm, out.real, workers)  # 对实部进行变换
        transform(tmp.imag, type, axes, norm, out.imag, workers)  # 对虚部进行变换
        return out  # 返回处理后的数组
    # 调用名为 transform 的函数，传递参数 tmp, type, axes, norm, out, workers, orthogonalize
    return transform(tmp, type, axes, norm, out, workers, orthogonalize)
# 创建一个偏函数 dctn，用于应用 _r2rn 函数，固定参数 True 和 pfft.dct
dctn = functools.partial(_r2rn, True, pfft.dct)
# 设置偏函数 dctn 的名称为 'dctn'
dctn.__name__ = 'dctn'

# 创建一个偏函数 idctn，用于应用 _r2rn 函数，固定参数 False 和 pfft.dct
idctn = functools.partial(_r2rn, False, pfft.dct)
# 设置偏函数 idctn 的名称为 'idctn'
idctn.__name__ = 'idctn'

# 创建一个偏函数 dstn，用于应用 _r2rn 函数，固定参数 True 和 pfft.dst
dstn = functools.partial(_r2rn, True, pfft.dst)
# 设置偏函数 dstn 的名称为 'dstn'
dstn.__name__ = 'dstn'

# 创建一个偏函数 idstn，用于应用 _r2rn 函数，固定参数 False 和 pfft.dst
idstn = functools.partial(_r2rn, False, pfft.dst)
# 设置偏函数 idstn 的名称为 'idstn'
idstn.__name__ = 'idstn'
```