# `numpy-ml\numpy_ml\utils\kernels.py`

```
# 导入 re 模块，用于正则表达式操作
# 导入 ABC 抽象基类和 abstractmethod 装饰器
import re
from abc import ABC, abstractmethod

# 导入 numpy 模块，并重命名为 np
import numpy as np

# 定义一个抽象基类 KernelBase
class KernelBase(ABC):
    # 初始化方法
    def __init__(self):
        super().__init__()
        # 初始化参数字典和超参数字典
        self.parameters = {}
        self.hyperparameters = {}

    # 抽象方法，用于计算核函数
    @abstractmethod
    def _kernel(self, X, Y):
        raise NotImplementedError

    # 重载 __call__ 方法，调用 _kernel 方法
    def __call__(self, X, Y=None):
        """Refer to documentation for the `_kernel` method"""
        return self._kernel(X, Y)

    # 重载 __str__ 方法，返回模型的字符串表示
    def __str__(self):
        P, H = self.parameters, self.hyperparameters
        p_str = ", ".join(["{}={}".format(k, v) for k, v in P.items()])
        return "{}({})".format(H["id"], p_str)

    # 定义 summary 方法，返回模型参数、超参数和 ID 的字典
    def summary(self):
        """Return the dictionary of model parameters, hyperparameters, and ID"""
        return {
            "id": self.hyperparameters["id"],
            "parameters": self.parameters,
            "hyperparameters": self.hyperparameters,
        }
    def set_params(self, summary_dict):
        """
        Set the model parameters and hyperparameters using the settings in
        `summary_dict`.

        Parameters
        ----------
        summary_dict : dict
            A dictionary with keys 'parameters' and 'hyperparameters',
            structured as would be returned by the :meth:`summary` method. If
            a particular (hyper)parameter is not included in this dict, the
            current value will be used.

        Returns
        -------
        new_kernel : :doc:`Kernel <numpy_ml.utils.kernels>` instance
            A kernel with parameters and hyperparameters adjusted to those
            specified in `summary_dict`.
        """
        # 将 self 和 summary_dict 分别赋值给 kr 和 sd
        kr, sd = self, summary_dict

        # 将 `parameters` 和 `hyperparameters` 嵌套字典合并为一个字典
        flatten_keys = ["parameters", "hyperparameters"]
        for k in flatten_keys:
            if k in sd:
                entry = sd[k]
                sd.update(entry)
                del sd[k]

        # 遍历 summary_dict 中的键值对
        for k, v in sd.items():
            # 如果键在 self 的 parameters 中，则更新参数值
            if k in self.parameters:
                kr.parameters[k] = v
            # 如果键在 self 的 hyperparameters 中，则更新超参数值
            if k in self.hyperparameters:
                kr.hyperparameters[k] = v
        # 返回更新后的 kernel 实例
        return kr
# 定义线性核函数类，继承自 KernelBase 类
class LinearKernel(KernelBase):
    # 初始化函数，设置线性核函数的参数 c0
    def __init__(self, c0=0):
        """
        The linear (i.e., dot-product) kernel.

        Notes
        -----
        For input vectors :math:`\mathbf{x}` and :math:`\mathbf{y}`, the linear
        kernel is:

        .. math::

            k(\mathbf{x}, \mathbf{y}) = \mathbf{x}^\\top \mathbf{y} + c_0

        Parameters
        ----------
        c0 : float
            An "inhomogeneity" parameter. When `c0` = 0, the kernel is said to be
            homogenous. Default is 1.
        """
        # 调用父类的初始化函数
        super().__init__()
        # 设置超参数字典
        self.hyperparameters = {"id": "LinearKernel"}
        # 设置参数字典，包括 c0 参数
        self.parameters = {"c0": c0}

    # 计算线性核函数的内部函数，计算输入矩阵 X 和 Y 的点积
    def _kernel(self, X, Y=None):
        """
        Compute the linear kernel (i.e., dot-product) between all pairs of rows in
        `X` and `Y`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            Collection of `N` input vectors
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(M, C)` or None
            Collection of `M` input vectors. If None, assume `Y` = `X`.
            Default is None.

        Returns
        -------
        out : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            Similarity between `X` and `Y`, where index (`i`, `j`) gives
            :math:`k(x_i, y_j)`.
        """
        # 检查输入矩阵 X 和 Y，确保它们符合要求
        X, Y = kernel_checks(X, Y)
        # 返回 X 和 Y 的点积加上参数 c0 的结果
        return X @ Y.T + self.parameters["c0"]


class PolynomialKernel(KernelBase):
    # 初始化多项式核函数对象，设置默认参数值
    def __init__(self, d=3, gamma=None, c0=1):
        """
        The degree-`d` polynomial kernel.

        Notes
        -----
        For input vectors :math:`\mathbf{x}` and :math:`\mathbf{y}`, the polynomial
        kernel is:

        .. math::

            k(\mathbf{x}, \mathbf{y}) = (\gamma \mathbf{x}^\\top \mathbf{y} + c_0)^d

        In contrast to the linear kernel, the polynomial kernel also computes
        similarities *across* dimensions of the **x** and **y** vectors,
        allowing it to account for interactions between features.  As an
        instance of the dot product family of kernels, the polynomial kernel is
        invariant to a rotation of the coordinates about the origin, but *not*
        to translations.

        Parameters
        ----------
        d : int
            Degree of the polynomial kernel. Default is 3.
        gamma : float or None
            A scaling parameter for the dot product between `x` and `y`,
            determining the amount of smoothing/resonlution of the kernel.
            Larger values result in greater smoothing. If None, defaults to 1 /
            `C`.  Sometimes referred to as the kernel bandwidth.  Default is
            None.
        c0 : float
            Parameter trading off the influence of higher-order versus lower-order
            terms in the polynomial. If `c0` = 0, the kernel is said to be
            homogenous. Default is 1.
        """
        # 调用父类的初始化方法
        super().__init__()
        # 设置核函数的超参数字典
        self.hyperparameters = {"id": "PolynomialKernel"}
        # 设置核函数的参数字典
        self.parameters = {"d": d, "c0": c0, "gamma": gamma}
    # 定义一个私有方法，计算输入矩阵 X 和 Y 之间的度为 d 的多项式核
    def _kernel(self, X, Y=None):
        """
        Compute the degree-`d` polynomial kernel between all pairs of rows in `X`
        and `Y`.

        # 参数：
        # X：形状为 `(N, C)` 的 ndarray，包含 `N` 个输入向量
        # Y：形状为 `(M, C)` 的 ndarray 或 None
        #     包含 `M` 个输入向量。如果为 None，则假定 `Y = X`。默认为 None。

        # 返回：
        # out：形状为 `(N, M)` 的 ndarray
        #     `X` 和 `Y` 之间的相似度，其中索引 (`i`, `j`) 给出了 :math:`k(x_i, y_j)`（即核的格拉姆矩阵）。
        """
        # 获取模型参数
        P = self.parameters
        # 检查并处理输入矩阵 X 和 Y
        X, Y = kernel_checks(X, Y)
        # 计算 gamma 值，如果参数中未指定 gamma，则默认为 1 / X.shape[1]
        gamma = 1 / X.shape[1] if P["gamma"] is None else P["gamma"]
        # 计算多项式核的值并返回结果
        return (gamma * (X @ Y.T) + P["c0"]) ** P["d"]
# 定义 RBFKernel 类，继承自 KernelBase 类
class RBFKernel(KernelBase):
    # 初始化方法，接受一个 sigma 参数
    def __init__(self, sigma=None):
        """
        Radial basis function (RBF) / squared exponential kernel.

        Notes
        -----
        For input vectors :math:`\mathbf{x}` and :math:`\mathbf{y}`, the radial
        basis function kernel is:

        .. math::

            k(\mathbf{x}, \mathbf{y}) = \exp \left\{ -0.5
                \left\lVert \\frac{\mathbf{x} -
                    \mathbf{y}}{\sigma} \\right\\rVert_2^2 \\right\}

        The RBF kernel decreases with distance and ranges between zero (in the
        limit) to one (when **x** = **y**). Notably, the implied feature space
        of the kernel has an infinite number of dimensions.

        Parameters
        ----------
        sigma : float or array of shape `(C,)` or None
            A scaling parameter for the vectors **x** and **y**, producing an
            isotropic kernel if a float, or an anisotropic kernel if an array of
            length `C`.  Larger values result in higher resolution / greater
            smoothing. If None, defaults to :math:`\sqrt(C / 2)`. Sometimes
            referred to as the kernel 'bandwidth'. Default is None.
        """
        # 调用父类的初始化方法
        super().__init__()
        # 设置超参数字典
        self.hyperparameters = {"id": "RBFKernel"}
        # 设置参数字典，包含 sigma 参数
        self.parameters = {"sigma": sigma}
    # 计算输入向量 X 和 Y 之间的径向基函数（RBF）核
    def _kernel(self, X, Y=None):
        """
        Computes the radial basis function (RBF) kernel between all pairs of
        rows in `X` and `Y`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            Collection of `N` input vectors, each with dimension `C`.
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(M, C)`
            Collection of `M` input vectors. If None, assume `Y` = `X`. Default
            is None.

        Returns
        -------
        out : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            Similarity between `X` and `Y` where index (i, j) gives :math:`k(x_i, y_j)`.
        """
        # 获取参数字典
        P = self.parameters
        # 检查输入向量 X 和 Y，确保它们是合法的 ndarray
        X, Y = kernel_checks(X, Y)
        # 计算默认的 sigma 值，如果参数字允许的话
        sigma = np.sqrt(X.shape[1] / 2) if P["sigma"] is None else P["sigma"]
        # 计算核矩阵，使用高斯核函数
        return np.exp(-0.5 * pairwise_l2_distances(X / sigma, Y / sigma) ** 2)
class KernelInitializer(object):
    # 初始化学习率调度器的类，可以接受以下类型的输入：
    # (a) `KernelBase` 实例的 __str__ 表示
    # (b) `KernelBase` 实例
    # (c) 参数字典（例如，通过 `KernelBase` 实例的 `summary` 方法生成的）

    # 如果 `param` 为 None，则返回 `LinearKernel`
    def __init__(self, param=None):
        self.param = param

    def __call__(self):
        param = self.param
        if param is None:
            kernel = LinearKernel()
        elif isinstance(param, KernelBase):
            kernel = param
        elif isinstance(param, str):
            kernel = self.init_from_str()
        elif isinstance(param, dict):
            kernel = self.init_from_dict()
        return kernel

    def init_from_str(self):
        # 定义正则表达式，用于解析参数字符串
        r = r"([a-zA-Z0-9]*)=([^,)]*)"
        # 将参数字符串转换为小写
        kr_str = self.param.lower()
        # 解析参数字符串，生成参数字典
        kwargs = dict([(i, eval(j)) for (i, j) in re.findall(r, self.param)])

        if "linear" in kr_str:
            kernel = LinearKernel(**kwargs)
        elif "polynomial" in kr_str:
            kernel = PolynomialKernel(**kwargs)
        elif "rbf" in kr_str:
            kernel = RBFKernel(**kwargs)
        else:
            raise NotImplementedError("{}".format(kr_str))
        return kernel

    def init_from_dict(self):
        S = self.param
        # 获取超参数字典
        sc = S["hyperparameters"] if "hyperparameters" in S else None

        if sc is None:
            raise ValueError("Must have `hyperparameters` key: {}".format(S))

        if sc and sc["id"] == "LinearKernel":
            scheduler = LinearKernel().set_params(S)
        elif sc and sc["id"] == "PolynomialKernel":
            scheduler = PolynomialKernel().set_params(S)
        elif sc and sc["id"] == "RBFKernel":
            scheduler = RBFKernel().set_params(S)
        elif sc:
            raise NotImplementedError("{}".format(sc["id"]))
        return scheduler
# 检查输入数据 X 和 Y 的维度，如果 X 是一维数组，则将其转换为二维数组
X = X.reshape(-1, 1) if X.ndim == 1 else X
# 如果 Y 为 None，则将 Y 设置为 X
Y = X if Y is None else Y
# 检查 Y 的维度，如果 Y 是一维数组，则将其转换为二维数组
Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y

# 断言 X 必须有两个维度
assert X.ndim == 2, "X must have 2 dimensions, but got {}".format(X.ndim)
# 断言 Y 必须有两个维度
assert Y.ndim == 2, "Y must have 2 dimensions, but got {}".format(Y.ndim)
# 断言 X 和 Y 必须有相同的列数
assert X.shape[1] == Y.shape[1], "X and Y must have the same number of columns"
# 返回经过检查和处理后的 X 和 Y
return X, Y


def pairwise_l2_distances(X, Y):
    """
    A fast, vectorized way to compute pairwise l2 distances between rows in `X`
    and `Y`.

    Notes
    -----
    An entry of the pairwise Euclidean distance matrix for two vectors is

    .. math::

        d[i, j]  &=  \sqrt{(x_i - y_i) @ (x_i - y_i)} \\\\
                 &=  \sqrt{sum (x_i - y_j)^2} \\\\
                 &=  \sqrt{sum (x_i)^2 - 2 x_i y_j + (y_j)^2}

    The code below computes the the third line using numpy broadcasting
    fanciness to avoid any for loops.

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
        Collection of `N` input vectors
    Y : :py:class:`ndarray <numpy.ndarray>` of shape `(M, C)`
        Collection of `M` input vectors. If None, assume `Y` = `X`. Default is
        None.

    Returns
    -------
    dists : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
        Pairwise distance matrix. Entry (i, j) contains the `L2` distance between
        :math:`x_i` and :math:`y_j`.
    """
    # 计算 X 和 Y 之间的 L2 距离矩阵
    D = -2 * X @ Y.T + np.sum(Y ** 2, axis=1) + np.sum(X ** 2, axis=1)[:, np.newaxis]
    # 将小于 0 的值裁剪为 0（由于数值精度问题可能导致小于 0 的值）
    D[D < 0] = 0
    # 返回计算得到的 L2 距离矩阵
    return np.sqrt(D)
```