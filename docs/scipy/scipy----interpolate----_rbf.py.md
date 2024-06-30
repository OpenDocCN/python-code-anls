# `D:\src\scipysrc\scipy\scipy\interpolate\_rbf.py`

```
# 导入 numpy 库，通常用于科学计算中的数组操作
import numpy as np

# 导入 scipy 库中的线性代数模块，用于进行线性代数运算
from scipy import linalg
# 导入 scipy 库中的数学函数 xlogy，用于计算 x * log(y) 安全的数值
from scipy.special import xlogy
# 导入 scipy 库中的空间距离计算模块
from scipy.spatial.distance import cdist, pdist, squareform

# 定义此模块可以公开的类名列表，仅包含 Rbf 类
__all__ = ['Rbf']


class Rbf:
    """
    Rbf(*args, **kwargs)

    A class for radial basis function interpolation of functions from
    N-D scattered data to an M-D domain.

    .. legacy:: class

        `Rbf` is legacy code, for new usage please use `RBFInterpolator`
        instead.

    Parameters
    ----------
    *args : arrays
        x, y, z, ..., d, where x, y, z, ... are the coordinates of the nodes
        and d is the array of values at the nodes
    """
    function : str or callable, optional
        # 定义径向基函数，根据半径 r 计算，使用默认的欧几里得距离
        # 默认函数为 'multiquadric'::
        #   'multiquadric': sqrt((r/self.epsilon)**2 + 1)
        #   'inverse': 1.0/sqrt((r/self.epsilon)**2 + 1)
        #   'gaussian': exp(-(r/self.epsilon)**2)
        #   'linear': r
        #   'cubic': r**3
        #   'quintic': r**5
        #   'thin_plate': r**2 * log(r)
        
        # 如果传入的是可调用对象，必须接受两个参数 (self, r)
        # 参数 epsilon 可通过 self.epsilon 访问，同时可以访问传入的其它关键字参数

    epsilon : float, optional
        # 可调整的常量，适用于高斯或多项式基函数，默认值大致为节点之间的平均距离

    smooth : float, optional
        # 平滑度参数，大于零会增加逼近的平滑程度
        # 当 smooth 为 0 时，执行插值操作（默认），此时函数必定通过节点

    norm : str, callable, optional
        # 用于计算两点之间距离的函数，接受位置数组作为输入（如 x, y, z...），输出为距离数组
        # 默认为 'euclidean'，即默认情况下计算 x1 中每个点到 x2 中每个点的距离矩阵
        # 更多选项请参考 `scipy.spatial.distances.cdist` 的文档

    mode : str, optional
        # 插值模式，可选 '1-D'（默认）或 'N-D'
        # 当为 '1-D' 时，数据 `d` 被视为 1-D，并在内部展平处理
        # 当为 'N-D' 时，数据 `d` 被假设为形状为 (n_samples, m) 的数组，其中 m 是目标域的维度

    Attributes
    ----------
    N : int
        # 数据点的数量（由输入数组确定）

    di : ndarray
        # 数据点 `xi` 处的数据值的 1-D 数组

    xi : ndarray
        # 数据坐标的 2-D 数组

    function : str or callable
        # 径向基函数，参见 Parameters 中的描述

    epsilon : float
        # 高斯或多项式基函数使用的参数，参见 Parameters 中的描述

    smooth : float
        # 平滑度参数，参见 Parameters 中的描述

    norm : str or callable
        # 距离函数，参见 Parameters 中的描述

    mode : str
        # 插值模式，参见 Parameters 中的描述

    nodes : ndarray
        # 插值的节点值数组

    A : internal property, do not use
        # 内部属性，不要使用

    See Also
    --------
    RBFInterpolator
        # 参见 RBFInterpolator 相关文档

    Examples
    --------
    # 示例
    >>> import numpy as np
    >>> from scipy.interpolate import Rbf
    >>> rng = np.random.default_rng()
    >>> x, y, z, d = rng.random((4, 50))
    >>> rbfi = Rbf(x, y, z, d)  # 径向基函数插值器实例
    >>> xi = yi = zi = np.linspace(0, 1, 20)
    >>> di = rbfi(xi, yi, zi)   # interpolated values
    >>> di.shape
    (20,)

    """
    # 可用的径向基函数，可以通过字符串选择；
    # 它们都以 _h_ 开头（self._init_function 依赖于此）
    def _h_multiquadric(self, r):
        return np.sqrt((1.0/self.epsilon*r)**2 + 1)

    def _h_inverse_multiquadric(self, r):
        return 1.0/np.sqrt((1.0/self.epsilon*r)**2 + 1)

    def _h_gaussian(self, r):
        return np.exp(-(1.0/self.epsilon*r)**2)

    def _h_linear(self, r):
        return r

    def _h_cubic(self, r):
        return r**3

    def _h_quintic(self, r):
        return r**5

    def _h_thin_plate(self, r):
        return xlogy(r**2, r)

    # 设置 self._function 并对初始 r 进行验证
    def _init_function(self, r):
        # 如果 function 是字符串
        if isinstance(self.function, str):
            self.function = self.function.lower()
            # 映射一些别名到标准函数名
            _mapped = {'inverse': 'inverse_multiquadric',
                       'inverse multiquadric': 'inverse_multiquadric',
                       'thin-plate': 'thin_plate'}
            if self.function in _mapped:
                self.function = _mapped[self.function]

            # 构建函数名
            func_name = "_h_" + self.function
            # 检查是否存在对应的函数
            if hasattr(self, func_name):
                self._function = getattr(self, func_name)
            else:
                # 如果不存在对应函数，列出可用的函数列表并引发错误
                functionlist = [x[3:] for x in dir(self)
                                if x.startswith('_h_')]
                raise ValueError("function must be a callable or one of " +
                                 ", ".join(functionlist))
            # 获取函数对象
            self._function = getattr(self, "_h_"+self.function)
        # 如果 function 是可调用的
        elif callable(self.function):
            allow_one = False
            # 如果具有 func_code 或者 __code__ 属性，认为是可调用的函数
            if hasattr(self.function, 'func_code') or \
               hasattr(self.function, '__code__'):
                val = self.function
                allow_one = True
            elif hasattr(self.function, "__call__"):
                val = self.function.__call__.__func__
            else:
                raise ValueError("Cannot determine number of arguments to "
                                 "function")

            # 获取参数个数
            argcount = val.__code__.co_argcount
            # 根据参数个数选择如何绑定函数
            if allow_one and argcount == 1:
                self._function = self.function
            elif argcount == 2:
                self._function = self.function.__get__(self, Rbf)
            else:
                raise ValueError("Function argument must take 1 or 2 "
                                 "arguments.")

        # 对于给定的 r，计算函数值 a0
        a0 = self._function(r)
        # 如果函数值 a0 的形状与 r 的形状不一致，引发错误
        if a0.shape != r.shape:
            raise ValueError("Callable must take array and return array of "
                             "the same shape")
        # 返回计算得到的函数值 a0
        return a0
    def __init__(self, *args, **kwargs):
        # `args` 可以是多个数组；我们将它们展平并存储为一个二维数组 `xi`，形状为 (n_args-1, array_size)，
        # 再加上一个一维数组 `di` 存储数值。
        # 所有数组必须具有相同数量的元素。
        self.xi = np.asarray([np.asarray(a, dtype=np.float64).flatten()
                              for a in args[:-1]])
        self.N = self.xi.shape[-1]  # 获取数组 `xi` 的最后一个维度长度，即总数据点数

        self.mode = kwargs.pop('mode', '1-D')  # 从 kwargs 中获取 'mode' 参数，默认为 '1-D'

        if self.mode == '1-D':
            self.di = np.asarray(args[-1]).flatten()  # 如果 'mode' 是 '1-D'，则将最后一个参数数组展平并存储到 `di`
            self._target_dim = 1  # 设置目标维度为 1
        elif self.mode == 'N-D':
            self.di = np.asarray(args[-1])  # 如果 'mode' 是 'N-D'，直接存储最后一个参数数组到 `di`
            self._target_dim = self.di.shape[-1]  # 设置目标维度为 `di` 的最后一个维度长度
        else:
            raise ValueError("Mode has to be 1-D or N-D.")  # 如果 'mode' 不是 '1-D' 或 'N-D'，抛出异常

        if not all([x.size == self.di.shape[0] for x in self.xi]):
            raise ValueError("All arrays must be equal length.")  # 检查所有数组是否具有相同长度，否则抛出异常

        self.norm = kwargs.pop('norm', 'euclidean')  # 从 kwargs 中获取 'norm' 参数，默认为 'euclidean'
        self.epsilon = kwargs.pop('epsilon', None)  # 从 kwargs 中获取 'epsilon' 参数，默认为 None
        if self.epsilon is None:
            # 默认 epsilon 是基于一个包围超立方体的 "节点之间的平均距离"
            ximax = np.amax(self.xi, axis=1)  # 计算每行的最大值
            ximin = np.amin(self.xi, axis=1)  # 计算每行的最小值
            edges = ximax - ximin  # 计算每行的边长
            edges = edges[np.nonzero(edges)]  # 去除为零的边长
            self.epsilon = np.power(np.prod(edges)/self.N, 1.0/edges.size)  # 计算 epsilon

        self.smooth = kwargs.pop('smooth', 0.0)  # 从 kwargs 中获取 'smooth' 参数，默认为 0.0
        self.function = kwargs.pop('function', 'multiquadric')  # 从 kwargs 中获取 'function' 参数，默认为 'multiquadric'

        # 将 kwargs 中剩余的任何项附加到 self，供任何可由用户调用的函数使用或用于保存在返回的对象中。
        for item, value in kwargs.items():
            setattr(self, item, value)

        # 计算权重
        if self._target_dim > 1:  # 如果目标维度大于 1
            # 首先因式分解矩阵
            self.nodes = np.zeros((self.N, self._target_dim), dtype=self.di.dtype)
            lu, piv = linalg.lu_factor(self.A)  # 对矩阵进行 LU 分解
            for i in range(self._target_dim):
                self.nodes[:, i] = linalg.lu_solve((lu, piv), self.di[:, i])  # 使用 LU 分解解线性方程组
        else:
            self.nodes = linalg.solve(self.A, self.di)  # 解线性方程组 A * x = di

    @property
    def A(self):
        # 这只是为了向后兼容存在的：self.A 是可用的，至少从技术上来说是公共的。
        r = squareform(pdist(self.xi.T, self.norm))  # 计算 xi 转置后的点对之间的距离
        return self._init_function(r) - np.eye(self.N)*self.smooth  # 返回经过初始化函数处理后的矩阵 A

    def _call_norm(self, x1, x2):
        return cdist(x1.T, x2.T, self.norm)  # 计算两个数组 x1 和 x2 之间的距离，使用给定的范数
    # 定义一个可调用对象，接受任意数量的参数
    def __call__(self, *args):
        # 将所有参数转换为 NumPy 数组
        args = [np.asarray(x) for x in args]
        
        # 检查所有参数的形状是否相等，如果不相等则抛出数值错误异常
        if not all([x.shape == y.shape for x in args for y in args]):
            raise ValueError("Array lengths must be equal")
        
        # 根据目标维度设置输出的形状
        if self._target_dim > 1:
            shp = args[0].shape + (self._target_dim,)
        else:
            shp = args[0].shape
        
        # 将所有参数展平并转换为 NumPy 数组，数据类型为 np.float64
        xa = np.asarray([a.flatten() for a in args], dtype=np.float64)
        
        # 调用对象内部的 _call_norm 方法，计算数据的归一化
        r = self._call_norm(xa, self.xi)
        
        # 对归一化后的数据应用设定的函数，然后与节点权重相乘，最后根据之前设置的形状进行重塑
        return np.dot(self._function(r), self.nodes).reshape(shp)
```