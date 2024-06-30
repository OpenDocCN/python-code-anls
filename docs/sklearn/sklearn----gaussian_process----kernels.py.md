# `D:\src\scipysrc\scikit-learn\sklearn\gaussian_process\kernels.py`

```
"""A set of kernels that can be combined by operators and used in Gaussian processes."""

# Kernels for Gaussian process regression and classification.
#
# The kernels in this module allow kernel-engineering, i.e., they can be
# combined via the "+" and "*" operators or be exponentiated with a scalar
# via "**". These sum and product expressions can also contain scalar values,
# which are automatically converted to a constant kernel.
#
# All kernels allow (analytic) gradient-based hyperparameter optimization.
# The space of hyperparameters can be specified by giving lower und upper
# boundaries for the value of each hyperparameter (the search space is thus
# rectangular). Instead of specifying bounds, hyperparameters can also be
# declared to be "fixed", which causes these hyperparameters to be excluded from
# optimization.


# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# Note: this module is strongly inspired by the kernel module of the george
#       package.

import math
import warnings
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from inspect import signature

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import gamma, kv

from ..base import clone
from ..exceptions import ConvergenceWarning
from ..metrics.pairwise import pairwise_kernels
from ..utils.validation import _num_samples


def _check_length_scale(X, length_scale):
    # Ensure length_scale is a float array and has appropriate dimensions
    length_scale = np.squeeze(length_scale).astype(float)
    if np.ndim(length_scale) > 1:
        raise ValueError("length_scale cannot be of dimension greater than 1")
    if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
        raise ValueError(
            "Anisotropic kernel must have the same number of "
            "dimensions as data (%d!=%d)" % (length_scale.shape[0], X.shape[1])
        )
    return length_scale


class Hyperparameter(
    namedtuple(
        "Hyperparameter", ("name", "value_type", "bounds", "n_elements", "fixed")
    )
):
    """A kernel hyperparameter's specification in form of a namedtuple.

    .. versionadded:: 0.18

    Attributes
    ----------
    name : str
        The name of the hyperparameter. Note that a kernel using a
        hyperparameter with name "x" must have the attributes self.x and
        self.x_bounds

    value_type : str
        The type of the hyperparameter. Currently, only "numeric"
        hyperparameters are supported.

    bounds : pair of floats >= 0 or "fixed"
        The lower and upper bound on the parameter. If n_elements>1, a pair
        of 1d array with n_elements each may be given alternatively. If
        the string "fixed" is passed as bounds, the hyperparameter's value
        cannot be changed.
    """
    # 定义一个命名元组的子类 Hyperparameter，用于表示高斯过程的超参数
    __slots__ = ()

    def __new__(cls, name, value_type, bounds, n_elements=1, fixed=None):
        # 如果 bounds 不是字符串或者不等于 "fixed"，则将其至少转换为二维数组
        if not isinstance(bounds, str) or bounds != "fixed":
            bounds = np.atleast_2d(bounds)
            # 如果 n_elements 大于 1，表示超参数是向量值的参数
            if n_elements > 1:
                # 如果 bounds 的行数为 1，则重复该行以匹配 n_elements 的个数
                if bounds.shape[0] == 1:
                    bounds = np.repeat(bounds, n_elements, 0)
                # 如果 bounds 的行数既不是 1 也不等于 n_elements，则引发 ValueError 异常
                elif bounds.shape[0] != n_elements:
                    raise ValueError(
                        "Bounds on %s should have either 1 or "
                        "%d dimensions. Given are %d"
                        % (name, n_elements, bounds.shape[0])
                    )

        # 如果 fixed 参数为 None，则根据 bounds 是否为字符串 "fixed" 来确定是否为固定参数
        if fixed is None:
            fixed = isinstance(bounds, str) and bounds == "fixed"
        
        # 使用父类的 __new__ 方法创建 Hyperparameter 的实例
        return super(Hyperparameter, cls).__new__(
            cls, name, value_type, bounds, n_elements, fixed
        )

    # 这是一个用于测试的实用工具，用于检查两个超参数是否相等
    # 定义一个特殊方法 __eq__，用于判断两个对象是否相等
    def __eq__(self, other):
        # 返回多个条件的逻辑与结果，用于比较对象的相等性
        return (
            self.name == other.name  # 检查对象的名称是否相等
            and self.value_type == other.value_type  # 检查对象的值类型是否相等
            and np.all(self.bounds == other.bounds)  # 使用 NumPy 检查对象的边界数组是否完全相等
            and self.n_elements == other.n_elements  # 检查对象的元素数量是否相等
            and self.fixed == other.fixed  # 检查对象的固定性是否相等
        )
class Kernel(metaclass=ABCMeta):
    """Base class for all kernels.

    .. versionadded:: 0.18

    Examples
    --------
    >>> from sklearn.gaussian_process.kernels import Kernel, RBF
    >>> import numpy as np
    >>> class CustomKernel(Kernel):
    ...     def __init__(self, length_scale=1.0):
    ...         self.length_scale = length_scale
    ...     def __call__(self, X, Y=None):
    ...         if Y is None:
    ...             Y = X
    ...         return np.inner(X, X if Y is None else Y) ** 2
    ...     def diag(self, X):
    ...         return np.ones(X.shape[0])
    ...     def is_stationary(self):
    ...         return True
    >>> kernel = CustomKernel(length_scale=2.0)
    >>> X = np.array([[1, 2], [3, 4]])
    >>> print(kernel(X))
    [[ 25 121]
     [121 625]]
    """

    def get_params(self, deep=True):
        """Get parameters of this kernel.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        # Initialize an empty dictionary to store parameters
        params = dict()

        # introspect the constructor arguments to find the model parameters
        # to represent
        cls = self.__class__
        # Obtain the original __init__ method of the class
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        # Extract the signature of the __init__ method
        init_sign = signature(init)
        args, varargs = [], []
        # Iterate over the parameters of the constructor
        for parameter in init_sign.parameters.values():
            # Exclude *args and **kwargs and 'self' parameter
            if parameter.kind != parameter.VAR_KEYWORD and parameter.name != "self":
                args.append(parameter.name)
            # Check for positional arguments (*args), which are not allowed
            if parameter.kind == parameter.VAR_POSITIONAL:
                varargs.append(parameter.name)

        # Raise an error if any positional arguments (*args) are present
        if len(varargs) != 0:
            raise RuntimeError(
                "scikit-learn kernels should always "
                "specify their parameters in the signature"
                " of their __init__ (no varargs)."
                " %s doesn't follow this convention." % (cls,)
            )
        
        # Populate the parameters dictionary with attribute values
        for arg in args:
            params[arg] = getattr(self, arg)

        return params
    def set_params(self, **params):
        """设置这个核函数的参数。

        该方法适用于简单的核函数以及嵌套核函数。后者的参数形式为``<component>__<parameter>``，
        这样可以更新嵌套对象的每个组件。

        Parameters
        ----------
        **params : dict
            参数字典，键为参数名，值为参数值。

        Returns
        -------
        self : object
            返回自身对象。
        """
        if not params:
            # 简单的优化，避免调用 inspect 增加速度
            return self
        valid_params = self.get_params(deep=True)
        for key, value in params.items():
            split = key.split("__", 1)
            if len(split) > 1:
                # 处理嵌套对象的情况
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError(
                        "Invalid parameter %s for kernel %s. "
                        "Check the list of available parameters "
                        "with `kernel.get_params().keys()`." % (name, self)
                    )
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # 处理简单对象的情况
                if key not in valid_params:
                    raise ValueError(
                        "Invalid parameter %s for kernel %s. "
                        "Check the list of available parameters "
                        "with `kernel.get_params().keys()`."
                        % (key, self.__class__.__name__)
                    )
                setattr(self, key, value)
        return self

    def clone_with_theta(self, theta):
        """使用给定的超参数 theta 返回 self 的克隆对象。

        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            超参数数组。

        Returns
        -------
        cloned : object
            克隆的对象，其 theta 被设置为给定的值。
        """
        cloned = clone(self)
        cloned.theta = theta
        return cloned

    @property
    def n_dims(self):
        """返回核函数的非固定超参数数量。"""
        return self.theta.shape[0]

    @property
    def hyperparameters(self):
        """返回所有超参数的列表。"""
        r = [
            getattr(self, attr)
            for attr in dir(self)
            if attr.startswith("hyperparameter_")
        ]
        return r

    @property
    def theta(self):
        """
        Returns the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns
        -------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        theta = []
        # 获取当前对象的所有参数
        params = self.get_params()
        # 遍历每个超参数
        for hyperparameter in self.hyperparameters:
            # 如果超参数不是固定的
            if not hyperparameter.fixed:
                # 将其对应的值加入 theta 中
                theta.append(params[hyperparameter.name])
        # 如果 theta 非空，则返回其的对数值
        if len(theta) > 0:
            return np.log(np.hstack(theta))
        else:
            return np.array([])

    @theta.setter
    def theta(self, theta):
        """
        Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        # 获取当前对象的所有参数
        params = self.get_params()
        i = 0
        # 遍历每个超参数
        for hyperparameter in self.hyperparameters:
            # 如果超参数是固定的，则跳过
            if hyperparameter.fixed:
                continue
            # 如果超参数有多个元素
            if hyperparameter.n_elements > 1:
                # 向量值参数
                params[hyperparameter.name] = np.exp(
                    theta[i : i + hyperparameter.n_elements]
                )
                i += hyperparameter.n_elements
            else:
                # 单个值参数
                params[hyperparameter.name] = np.exp(theta[i])
                i += 1

        # 如果 i 不等于 theta 的长度，则抛出异常
        if i != len(theta):
            raise ValueError(
                "theta has not the correct number of entries."
                " Should be %d; given are %d" % (i, len(theta))
            )
        # 更新参数
        self.set_params(**params)

    @property
    def bounds(self):
        """
        Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : ndarray of shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        bounds = [
            hyperparameter.bounds
            for hyperparameter in self.hyperparameters
            if not hyperparameter.fixed
        ]
        # 如果 bounds 非空，则返回其的对数值
        if len(bounds) > 0:
            return np.log(np.vstack(bounds))
        else:
            return np.array([])

    def __add__(self, b):
        """
        Adds two kernel objects or a kernel object and a constant.

        Parameters
        ----------
        b : Kernel or numeric
            Another kernel object or a numeric constant

        Returns
        -------
        Sum(self, b) : Sum
            The sum of the current kernel object with b
        """
        if not isinstance(b, Kernel):
            return Sum(self, ConstantKernel(b))
        return Sum(self, b)

    def __radd__(self, b):
        """
        Adds a kernel object or a constant to the current kernel object (right-side addition).

        Parameters
        ----------
        b : Kernel or numeric
            Another kernel object or a numeric constant

        Returns
        -------
        Sum(ConstantKernel(b), self) : Sum
            The sum of b with the current kernel object
        """
        if not isinstance(b, Kernel):
            return Sum(ConstantKernel(b), self)
        return Sum(b, self)

    def __mul__(self, b):
        """
        Multiplies two kernel objects or a kernel object and a constant.

        Parameters
        ----------
        b : Kernel or numeric
            Another kernel object or a numeric constant

        Returns
        -------
        Product(self, b) : Product
            The product of the current kernel object with b
        """
        if not isinstance(b, Kernel):
            return Product(self, ConstantKernel(b))
        return Product(self, b)
    def __rmul__(self, b):
        # 检查 b 是否为 Kernel 类型的对象，如果不是，则创建一个 ConstantKernel(b) 和 self 的乘积
        if not isinstance(b, Kernel):
            return Product(ConstantKernel(b), self)
        # 如果 b 是 Kernel 类型的对象，则返回 b 和 self 的乘积
        return Product(b, self)

    def __pow__(self, b):
        # 返回一个 Exponentiation 对象，表示 self 的 b 次方
        return Exponentiation(self, b)

    def __eq__(self, b):
        # 检查 self 和 b 是否为同一类型的对象
        if type(self) != type(b):
            return False
        # 获取 self 和 b 的参数字典
        params_a = self.get_params()
        params_b = b.get_params()
        # 遍历并比较两个参数字典的键和值
        for key in set(list(params_a.keys()) + list(params_b.keys())):
            # 如果任何参数值不相等，则返回 False
            if np.any(params_a.get(key, None) != params_b.get(key, None)):
                return False
        # 如果所有参数都相等，则返回 True
        return True

    def __repr__(self):
        # 返回对象的字符串表示，包括类名和对象的参数向量格式化为字符串
        return "{0}({1})".format(
            self.__class__.__name__, ", ".join(map("{0:.3g}".format, self.theta))
        )

    @abstractmethod
    def __call__(self, X, Y=None, eval_gradient=False):
        """Evaluate the kernel."""
        # 抽象方法：计算核函数的值

    @abstractmethod
    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        # 抽象方法：返回核函数在输入 X 上的对角线值

    @abstractmethod
    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        # 抽象方法：返回核函数是否是平稳的

    @property
    def requires_vector_input(self):
        """Returns whether the kernel is defined on fixed-length feature
        vectors or generic objects. Defaults to True for backward
        compatibility."""
        # 返回核函数是否定义在固定长度的特征向量上，或者泛化的对象上，默认为 True 以保持向后兼容性
        return True
    def _check_bounds_params(self):
        """Called after fitting to warn if bounds may have been too tight."""
        # 检查是否有超过边界的参数，用于在拟合后发出警告
        list_close = np.isclose(self.bounds, np.atleast_2d(self.theta).T)
        # 初始化索引
        idx = 0
        # 遍历超参数列表
        for hyp in self.hyperparameters:
            # 如果当前超参数是固定的，则跳过
            if hyp.fixed:
                continue
            # 遍历当前超参数的维度
            for dim in range(hyp.n_elements):
                # 如果参数接近于下界，发出警告
                if list_close[idx, 0]:
                    warnings.warn(
                        "The optimal value found for "
                        "dimension %s of parameter %s is "
                        "close to the specified lower "
                        "bound %s. Decreasing the bound and"
                        " calling fit again may find a "
                        "better value." % (dim, hyp.name, hyp.bounds[dim][0]),
                        ConvergenceWarning,
                    )
                # 如果参数接近于上界，发出警告
                elif list_close[idx, 1]:
                    warnings.warn(
                        "The optimal value found for "
                        "dimension %s of parameter %s is "
                        "close to the specified upper "
                        "bound %s. Increasing the bound and"
                        " calling fit again may find a "
                        "better value." % (dim, hyp.name, hyp.bounds[dim][1]),
                        ConvergenceWarning,
                    )
                # 更新索引
                idx += 1
class NormalizedKernelMixin:
    """Mixin for kernels which are normalized: k(X, X)=1.

    .. versionadded:: 0.18
    """

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return np.ones(X.shape[0])


class StationaryKernelMixin:
    """Mixin for kernels which are stationary: k(X, Y)= f(X-Y).

    .. versionadded:: 0.18
    """

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return True


class GenericKernelMixin:
    """Mixin for kernels which operate on generic objects such as variable-
    length sequences, trees, and graphs.

    .. versionadded:: 0.22
    """

    @property
    def requires_vector_input(self):
        """Whether the kernel works only on fixed-length feature vectors."""
        return False


class CompoundKernel(Kernel):
    """Kernel which is composed of a set of other kernels.

    .. versionadded:: 0.18

    Parameters
    ----------
    kernels : list of Kernels
        The other kernels

    Examples
    --------
    >>> from sklearn.gaussian_process.kernels import WhiteKernel
    >>> from sklearn.gaussian_process.kernels import RBF
    >>> from sklearn.gaussian_process.kernels import CompoundKernel
    >>> kernel = CompoundKernel(
    ...     [WhiteKernel(noise_level=3.0), RBF(length_scale=2.0)])
    >>> print(kernel.bounds)
    [[-11.51292546  11.51292546]
     [-11.51292546  11.51292546]]
    >>> print(kernel.n_dims)
    2
    >>> print(kernel.theta)
    [1.09861229 0.69314718]
    """

    def __init__(self, kernels):
        self.kernels = kernels

    def get_params(self, deep=True):
        """Get parameters of this kernel.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return dict(kernels=self.kernels)

    @property
    def bounds(self):
        """Returns the bounds on the kernel parameters.

        These bounds are determined based on the individual kernels
        that make up this compound kernel.

        Returns
        -------
        bounds : ndarray of shape (n_params, 2)
            Lower and upper bounds on the parameters.
        """
        # Implementing logic to compute bounds based on component kernels
        lower_bounds = np.min([kernel.bounds[:, 0] for kernel in self.kernels], axis=0)
        upper_bounds = np.max([kernel.bounds[:, 1] for kernel in self.kernels], axis=0)
        return np.stack((lower_bounds, upper_bounds), axis=-1)

    @property
    def n_dims(self):
        """Returns the number of dimensions of the kernel.

        This is determined by the sum of dimensions of all component kernels.

        Returns
        -------
        n_dims : int
            Number of dimensions of the kernel.
        """
        return sum([kernel.n_dims for kernel in self.kernels])

    @property
    def theta(self):
        """Returns the parameters of the kernel.

        Returns
        -------
        theta : ndarray of shape (n_params,)
            Parameters of the kernel.
        """
        return np.concatenate([kernel.theta for kernel in self.kernels])
    def theta(self):
        """
        Returns the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns
        -------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        # 返回所有内核的 theta 属性的水平堆叠
        return np.hstack([kernel.theta for kernel in self.kernels])

    @theta.setter
    def theta(self, theta):
        """
        Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : array of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        # 确定每个内核的维度
        k_dims = self.k1.n_dims
        # 为每个内核设置对应的 theta 属性
        for i, kernel in enumerate(self.kernels):
            kernel.theta = theta[i * k_dims : (i + 1) * k_dims]

    @property
    def bounds(self):
        """
        Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : array of shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        # 返回所有内核的 bounds 属性的垂直堆叠
        return np.vstack([kernel.bounds for kernel in self.kernels])
    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Note that this compound kernel returns the results of all simple kernel
        stacked along an additional axis.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object, \
            default=None
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (n_samples_X, n_features) or list of object, \
            default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of the
            kernel hyperparameter is computed.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y, n_kernels)
            Kernel k(X, Y)

        K_gradient : ndarray of shape \
                (n_samples_X, n_samples_X, n_dims, n_kernels), optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        if eval_gradient:
            # Initialize empty lists for kernels and their gradients
            K = []
            K_grad = []
            # Iterate through each kernel in the list of kernels
            for kernel in self.kernels:
                # Compute kernel and its gradient for given X, Y
                K_single, K_grad_single = kernel(X, Y, eval_gradient)
                # Append kernel result to K list
                K.append(K_single)
                # Append kernel gradient result to K_grad list,
                # adding an extra axis using np.newaxis
                K_grad.append(K_grad_single[..., np.newaxis])
            # Stack kernels along the third axis to form the final K tensor
            return np.dstack(K), np.concatenate(K_grad, 3)
        else:
            # Return stacked kernel results for each kernel in self.kernels
            return np.dstack([kernel(X, Y, eval_gradient) for kernel in self.kernels])

    def __eq__(self, b):
        # Check if self and b are of the same type and have the same number of kernels
        if type(self) != type(b) or len(self.kernels) != len(b.kernels):
            return False
        # Check element-wise equality of each kernel in self.kernels and b.kernels
        return np.all(
            [self.kernels[i] == b.kernels[i] for i in range(len(self.kernels))]
        )

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        # Check if all kernels in self.kernels are stationary
        return np.all([kernel.is_stationary() for kernel in self.kernels])

    @property
    def requires_vector_input(self):
        """Returns whether the kernel is defined on discrete structures."""
        # Check if any kernel in self.kernels requires vector input
        return np.any([kernel.requires_vector_input for kernel in self.kernels])

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to `np.diag(self(X))`; however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X, n_kernels)
            Diagonal of kernel k(X, X)
        """
        # Compute the diagonal for each kernel in self.kernels and stack them
        return np.vstack([kernel.diag(X) for kernel in self.kernels]).T
class KernelOperator(Kernel):
    """Base class for all kernel operators.

    .. versionadded:: 0.18
    """

    def __init__(self, k1, k2):
        # 初始化方法，接受两个内核对象作为参数并存储它们
        self.k1 = k1
        self.k2 = k2

    def get_params(self, deep=True):
        """Get parameters of this kernel.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        # 初始化参数字典，包含自身以及两个子内核的参数
        params = dict(k1=self.k1, k2=self.k2)
        if deep:
            # 如果 deep 为 True，则获取子内核的参数并加入到 params 中
            deep_items = self.k1.get_params().items()
            params.update(("k1__" + k, val) for k, val in deep_items)
            deep_items = self.k2.get_params().items()
            params.update(("k2__" + k, val) for k, val in deep_items)

        return params

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter."""
        # 返回包含两个子内核所有超参数的列表
        r = [
            Hyperparameter(
                "k1__" + hyperparameter.name,
                hyperparameter.value_type,
                hyperparameter.bounds,
                hyperparameter.n_elements,
            )
            for hyperparameter in self.k1.hyperparameters
        ]

        for hyperparameter in self.k2.hyperparameters:
            r.append(
                Hyperparameter(
                    "k2__" + hyperparameter.name,
                    hyperparameter.value_type,
                    hyperparameter.bounds,
                    hyperparameter.n_elements,
                )
            )
        return r

    @property
    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns
        -------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        # 返回包含两个子内核超参数的连接数组
        return np.append(self.k1.theta, self.k2.theta)

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        # 根据传入的 theta 数组设置两个子内核的超参数
        k1_dims = self.k1.n_dims
        self.k1.theta = theta[:k1_dims]
        self.k2.theta = theta[k1_dims:]
    # 返回对 theta 的对数变换后的边界

    def bounds(self):
        """Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : ndarray of shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        # 如果 k1 的边界大小为 0，则返回 k2 的边界
        if self.k1.bounds.size == 0:
            return self.k2.bounds
        # 如果 k2 的边界大小为 0，则返回 k1 的边界
        if self.k2.bounds.size == 0:
            return self.k1.bounds
        # 否则，返回 k1 和 k2 边界的垂直堆叠
        return np.vstack((self.k1.bounds, self.k2.bounds))

    def __eq__(self, b):
        # 检查两个对象是否相等
        if type(self) != type(b):
            return False
        # 检查两个内部核是否相等（顺序可能不同）
        return (self.k1 == b.k1 and self.k2 == b.k2) or (
            self.k1 == b.k2 and self.k2 == b.k1
        )

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        # 返回内部核 k1 和 k2 是否都是平稳核
        return self.k1.is_stationary() and self.k2.is_stationary()

    @property
    def requires_vector_input(self):
        """Returns whether the kernel is stationary."""
        # 返回内部核 k1 或 k2 是否需要向量输入
        return self.k1.requires_vector_input or self.k2.requires_vector_input
# 定义一个名为 Sum 的类，继承自 KernelOperator
class Sum(KernelOperator):
    """The `Sum` kernel takes two kernels :math:`k_1` and :math:`k_2`
    and combines them via

    .. math::
        k_{sum}(X, Y) = k_1(X, Y) + k_2(X, Y)

    Note that the `__add__` magic method is overridden, so
    `Sum(RBF(), RBF())` is equivalent to using the + operator
    with `RBF() + RBF()`.


    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    k1 : Kernel
        The first base-kernel of the sum-kernel

    k2 : Kernel
        The second base-kernel of the sum-kernel

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import RBF, Sum, ConstantKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = Sum(ConstantKernel(2), RBF())
    >>> gpr = GaussianProcessRegressor(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    1.0
    >>> kernel
    1.41**2 + RBF(length_scale=1)
    """

    # 定义类的调用方法，返回核函数 k(X, Y) 及其梯度（如果需要）
    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (n_samples_X, n_features) or list of object,\
                default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims),\
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        # 如果需要计算梯度
        if eval_gradient:
            # 计算第一个核函数 k1 的值及其梯度
            K1, K1_gradient = self.k1(X, Y, eval_gradient=True)
            # 计算第二个核函数 k2 的值及其梯度
            K2, K2_gradient = self.k2(X, Y, eval_gradient=True)
            # 返回两个核函数的和以及它们梯度的叠加
            return K1 + K2, np.dstack((K1_gradient, K2_gradient))
        else:
            # 直接返回两个核函数的和
            return self.k1(X, Y) + self.k2(X, Y)
    # 返回核函数 k(X, X) 的对角线元素
    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to `np.diag(self(X))`; however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        # 调用 self.k1 对象的 diag 方法和 self.k2 对象的 diag 方法，将它们的结果相加
        return self.k1.diag(X) + self.k2.diag(X)

    # 返回当前对象的字符串表示，形式为 "{k1} + {k2}"
    def __repr__(self):
        return "{0} + {1}".format(self.k1, self.k2)
class Product(KernelOperator):
    """The `Product` kernel takes two kernels :math:`k_1` and :math:`k_2`
    and combines them via

    .. math::
        k_{prod}(X, Y) = k_1(X, Y) * k_2(X, Y)

    Note that the `__mul__` magic method is overridden, so
    `Product(RBF(), RBF())` is equivalent to using the * operator
    with `RBF() * RBF()`.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    k1 : Kernel
        The first base-kernel of the product-kernel

    k2 : Kernel
        The second base-kernel of the product-kernel


    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import (RBF, Product,
    ...            ConstantKernel)
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = Product(ConstantKernel(2), RBF())
    >>> gpr = GaussianProcessRegressor(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    1.0
    >>> kernel
    1.41**2 * RBF(length_scale=1)
    """

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (n_samples_Y, n_features) or list of object,\
            default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        # If eval_gradient is True, compute the kernels and their gradients
        if eval_gradient:
            # Compute kernel K1 and its gradient
            K1, K1_gradient = self.k1(X, Y, eval_gradient=True)
            # Compute kernel K2 and its gradient
            K2, K2_gradient = self.k2(X, Y, eval_gradient=True)
            # Return the product kernel K and its gradient
            return K1 * K2, np.dstack(
                (K1_gradient * K2[:, :, np.newaxis], K2_gradient * K1[:, :, np.newaxis])
            )
        else:
            # If eval_gradient is False, compute only the product kernel K
            return self.k1(X, Y) * self.k2(X, Y)
    # 返回核函数 k(X, X) 的对角线元素
    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        # 调用两个子核函数的 diag 方法，分别计算对角线元素，然后相乘得到结果
        return self.k1.diag(X) * self.k2.diag(X)

    # 返回描述核组合的字符串表示形式
    def __repr__(self):
        return "{0} * {1}".format(self.k1, self.k2)
class Exponentiation(Kernel):
    """The Exponentiation kernel takes one base kernel and a scalar parameter
    :math:`p` and combines them via

    .. math::
        k_{exp}(X, Y) = k(X, Y) ^p

    Note that the `__pow__` magic method is overridden, so
    `Exponentiation(RBF(), 2)` is equivalent to using the ** operator
    with `RBF() ** 2`.


    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    kernel : Kernel
        The base kernel

    exponent : float
        The exponent for the base kernel


    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import (RationalQuadratic,
    ...            Exponentiation)
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = Exponentiation(RationalQuadratic(), exponent=2)
    >>> gpr = GaussianProcessRegressor(kernel=kernel, alpha=5,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    0.419...
    >>> gpr.predict(X[:1,:], return_std=True)
    (array([635.5...]), array([0.559...]))
    """

    def __init__(self, kernel, exponent):
        # 初始化方法，接受一个基础核函数和一个指数参数
        self.kernel = kernel
        self.exponent = exponent

    def get_params(self, deep=True):
        """Get parameters of this kernel.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = dict(kernel=self.kernel, exponent=self.exponent)
        if deep:
            # 如果深度为True，递归获取内部核函数的参数
            deep_items = self.kernel.get_params().items()
            params.update(("kernel__" + k, val) for k, val in deep_items)
        return params

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter."""
        r = []
        for hyperparameter in self.kernel.hyperparameters:
            # 返回所有超参数的列表
            r.append(
                Hyperparameter(
                    "kernel__" + hyperparameter.name,
                    hyperparameter.value_type,
                    hyperparameter.bounds,
                    hyperparameter.n_elements,
                )
            )
        return r

    @property
    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns
        -------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        return self.kernel.theta
    # 定义 theta 属性的 setter 方法，用于设置（扁平化且对数转换后的）非固定超参数

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        self.kernel.theta = theta

    @property
    def bounds(self):
        """Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : ndarray of shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        return self.kernel.bounds

    def __eq__(self, b):
        """Checks if two instances of the class are equal.

        Parameters
        ----------
        b : object
            Another instance to compare with

        Returns
        -------
        bool
            True if both instances have the same kernel and exponent, False otherwise
        """
        if type(self) != type(b):
            return False
        return self.kernel == b.kernel and self.exponent == b.exponent

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (n_samples_Y, n_features) or list of object, default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X) is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        if eval_gradient:
            K, K_gradient = self.kernel(X, Y, eval_gradient=True)
            K_gradient *= self.exponent * K[:, :, np.newaxis] ** (self.exponent - 1)
            return K ** self.exponent, K_gradient
        else:
            K = self.kernel(X, Y, eval_gradient=False)
            return K ** self.exponent

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return self.kernel.diag(X) ** self.exponent

    def __repr__(self):
        """Return a string representation of the object."""
        return "{0} ** {1}".format(self.kernel, self.exponent)

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return self.kernel.is_stationary()
    # 定义一个属性方法，用于返回核函数是否在离散结构上定义
    @property
    def requires_vector_input(self):
        # 返回核函数对象的requires_vector_input属性，表示是否需要向量输入
        """Returns whether the kernel is defined on discrete structures."""
        return self.kernel.requires_vector_input
# 定义一个常数核类，继承自 StationaryKernelMixin、GenericKernelMixin 和 Kernel 类
class ConstantKernel(StationaryKernelMixin, GenericKernelMixin, Kernel):
    """Constant kernel.

    Can be used as part of a product-kernel where it scales the magnitude of
    the other factor (kernel) or as part of a sum-kernel, where it modifies
    the mean of the Gaussian process.

    .. math::
        k(x_1, x_2) = constant\\_value \\;\\forall\\; x_1, x_2

    Adding a constant kernel is equivalent to adding a constant::

            kernel = RBF() + ConstantKernel(constant_value=2)

    is the same as::

            kernel = RBF() + 2

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    constant_value : float, default=1.0
        The constant value which defines the covariance:
        k(x_1, x_2) = constant_value

    constant_value_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on `constant_value`.
        If set to "fixed", `constant_value` cannot be changed during
        hyperparameter tuning.

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = RBF() + ConstantKernel(constant_value=2)
    >>> gpr = GaussianProcessRegressor(kernel=kernel, alpha=5,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    0.3696...
    >>> gpr.predict(X[:1,:], return_std=True)
    (array([606.1...]), array([0.24...]))

    """

    def __init__(self, constant_value=1.0, constant_value_bounds=(1e-5, 1e5)):
        # 初始化方法，设置常数核的常数值和范围
        self.constant_value = constant_value
        self.constant_value_bounds = constant_value_bounds

    @property
    def hyperparameter_constant_value(self):
        # 返回一个描述常数值超参数的 Hyperparameter 对象
        return Hyperparameter("constant_value", "numeric", self.constant_value_bounds)
    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (n_samples_X, n_features) or list of object, \
            default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
            optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        # 如果 Y 为 None，则将 Y 设为 X，即计算 k(X, X)
        if Y is None:
            Y = X
        # 如果 eval_gradient 为 True，但 Y 不为 None，则抛出错误
        elif eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        # 初始化核矩阵 K，填充为常数值 self.constant_value
        K = np.full(
            (_num_samples(X), _num_samples(Y)),
            self.constant_value,
            dtype=np.array(self.constant_value).dtype,
        )
        # 如果 eval_gradient 为 True
        if eval_gradient:
            # 如果核超参数 self.hyperparameter_constant_value.fixed 未固定
            if not self.hyperparameter_constant_value.fixed:
                # 返回 K 和对于核超参数对数的梯度矩阵
                return (
                    K,
                    np.full(
                        (_num_samples(X), _num_samples(X), 1),
                        self.constant_value,
                        dtype=np.array(self.constant_value).dtype,
                    ),
                )
            else:
                # 返回 K 和空的梯度矩阵（维度为 0）
                return K, np.empty((_num_samples(X), _num_samples(X), 0))
        else:
            # 如果 eval_gradient 为 False，只返回核矩阵 K
            return K

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        # 返回核 k(X, X) 的对角线，使用常数值 self.constant_value 填充
        return np.full(
            _num_samples(X),
            self.constant_value,
            dtype=np.array(self.constant_value).dtype,
        )

    def __repr__(self):
        # 返回核的字符串表示，显示为 "{constant_value}**2" 格式
        return "{0:.3g}**2".format(np.sqrt(self.constant_value))
class WhiteKernel(StationaryKernelMixin, GenericKernelMixin, Kernel):
    """White kernel.

    The main use-case of this kernel is as part of a sum-kernel where it
    explains the noise of the signal as independently and identically
    normally-distributed. The parameter noise_level equals the variance of this
    noise.

    .. math::
        k(x_1, x_2) = noise\\_level \\text{ if } x_i == x_j \\text{ else } 0

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    noise_level : float, default=1.0
        Parameter controlling the noise level (variance)

    noise_level_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'noise_level'.
        If set to "fixed", 'noise_level' cannot be changed during
        hyperparameter tuning.

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = DotProduct() + WhiteKernel(noise_level=0.5)
    >>> gpr = GaussianProcessRegressor(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    0.3680...
    >>> gpr.predict(X[:2,:], return_std=True)
    (array([653.0..., 592.1... ]), array([316.6..., 316.6...]))

    """

    def __init__(self, noise_level=1.0, noise_level_bounds=(1e-5, 1e5)):
        # 初始化 WhiteKernel 类，设定噪声水平和噪声水平范围
        self.noise_level = noise_level
        self.noise_level_bounds = noise_level_bounds

    @property
    def hyperparameter_noise_level(self):
        # 返回噪声水平超参数对象，用于模型超参数的管理和优化
        return Hyperparameter("noise_level", "numeric", self.noise_level_bounds)
    # 定义一个方法，使对象可以像函数一样被调用，返回核函数 k(X, Y) 及其梯度（如果需要）
    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (n_samples_X, n_features) or list of object,\
            default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims),\
            optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        # 如果 Y 不为 None 并且需要计算梯度，抛出异常
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        # 如果 Y 为 None，计算并返回 k(X, X) 的结果
        if Y is None:
            # 构造对角阵 K，使用噪声水平乘以单位矩阵
            K = self.noise_level * np.eye(_num_samples(X))
            # 如果需要计算梯度
            if eval_gradient:
                # 如果噪声水平未固定，返回 K 和 K 的梯度
                if not self.hyperparameter_noise_level.fixed:
                    return (
                        K,
                        self.noise_level * np.eye(_num_samples(X))[:, :, np.newaxis],
                    )
                # 如果噪声水平已固定，返回 K 和空的梯度数组
                else:
                    return K, np.empty((_num_samples(X), _num_samples(X), 0))
            # 如果不需要计算梯度，只返回 K
            else:
                return K
        # 如果 Y 不为 None，返回一个全零数组作为 k(X, Y) 的结果
        else:
            return np.zeros((_num_samples(X), _num_samples(Y)))

    # 返回核函数 k(X, X) 对角线上的值
    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        # 返回一个数组，数组元素为噪声水平，长度与 X 的样本数相同
        return np.full(
            _num_samples(X), self.noise_level, dtype=np.array(self.noise_level).dtype
        )

    # 返回核函数对象的字符串表示，包括噪声水平
    def __repr__(self):
        return "{0}(noise_level={1:.3g})".format(
            self.__class__.__name__, self.noise_level
        )
class RBF(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """Radial basis function kernel (aka squared-exponential kernel).

    The RBF kernel is a stationary kernel. It is also known as the
    "squared exponential" kernel. It is parameterized by a length scale
    parameter :math:`l>0`, which can either be a scalar (isotropic variant
    of the kernel) or a vector with the same number of dimensions as the inputs
    X (anisotropic variant of the kernel). The kernel is given by:

    .. math::
        k(x_i, x_j) = \\exp\\left(- \\frac{d(x_i, x_j)^2}{2l^2} \\right)

    where :math:`l` is the length scale of the kernel and
    :math:`d(\\cdot,\\cdot)` is the Euclidean distance.
    For advice on how to set the length scale parameter, see e.g. [1]_.

    This kernel is infinitely differentiable, which implies that GPs with this
    kernel as covariance function have mean square derivatives of all orders,
    and are thus very smooth.
    See [2]_, Chapter 4, Section 4.2, for further details of the RBF kernel.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    length_scale : float or ndarray of shape (n_features,), default=1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.

    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'length_scale'.
        If set to "fixed", 'length_scale' cannot be changed during
        hyperparameter tuning.

    References
    ----------
    .. [1] `David Duvenaud (2014). "The Kernel Cookbook:
        Advice on Covariance functions".
        <https://www.cs.toronto.edu/~duvenaud/cookbook/>`_

    .. [2] `Carl Edward Rasmussen, Christopher K. I. Williams (2006).
        "Gaussian Processes for Machine Learning". The MIT Press.
        <http://www.gaussianprocess.org/gpml/>`_

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import RBF
    >>> X, y = load_iris(return_X_y=True)
    >>> kernel = 1.0 * RBF(1.0)
    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpc.score(X, y)
    0.9866...
    >>> gpc.predict_proba(X[:2,:])
    array([[0.8354..., 0.03228..., 0.1322...],
           [0.7906..., 0.0652..., 0.1441...]])

    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        # 初始化 RBF 核函数对象
        self.length_scale = length_scale
        # 设置长度尺度参数
        self.length_scale_bounds = length_scale_bounds
        # 设置长度尺度的边界范围

    @property
    def anisotropic(self):
        # 检查长度尺度是否为可迭代对象且长度大于1，确定是否是各向异性的核函数
        return np.iterable(self.length_scale) and len(self.length_scale) > 1
    # 定义方法 hyperparameter_length_scale，根据是否各向异性返回超参数对象
    def hyperparameter_length_scale(self):
        # 如果设置为各向异性，返回具有指定长度范围和长度列表长度的超参数对象
        if self.anisotropic:
            return Hyperparameter(
                "length_scale",
                "numeric",
                self.length_scale_bounds,
                len(self.length_scale),
            )
        # 否则返回具有指定长度范围的超参数对象
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    # 实现 __call__ 方法，计算核函数 k(X, Y) 及其梯度（如果需要）
    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        # 将输入 X 至少转换为二维数组
        X = np.atleast_2d(X)
        # 检查长度尺度，并返回有效长度尺度
        length_scale = _check_length_scale(X, self.length_scale)
        
        # 如果 Y 为 None，计算 X 内部距离矩阵的平方欧氏距离
        if Y is None:
            dists = pdist(X / length_scale, metric="sqeuclidean")
            # 计算核矩阵 K
            K = np.exp(-0.5 * dists)
            # 将上三角矩阵转换为完整方阵
            K = squareform(K)
            # 主对角线元素设为 1
            np.fill_diagonal(K, 1)
        else:
            # 如果需要计算梯度但 Y 不为 None，抛出异常
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            # 计算 X 和 Y 之间的平方欧氏距离
            dists = cdist(X / length_scale, Y / length_scale, metric="sqeuclidean")
            # 计算核矩阵 K
            K = np.exp(-0.5 * dists)

        # 如果需要计算梯度
        if eval_gradient:
            # 如果长度尺度超参数被固定
            if self.hyperparameter_length_scale.fixed:
                # 返回核矩阵 K 和空的梯度数组
                return K, np.empty((X.shape[0], X.shape[0], 0))
            # 如果没有各向异性或长度尺度是标量
            elif not self.anisotropic or length_scale.shape[0] == 1:
                # 计算核矩阵 K 的梯度
                K_gradient = (K * squareform(dists))[:, :, np.newaxis]
                # 返回核矩阵 K 和其梯度
                return K, K_gradient
            # 如果存在各向异性
            elif self.anisotropic:
                # 需要重新计算按维度的成对距离
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (
                    length_scale**2
                )
                K_gradient *= K[..., np.newaxis]
                # 返回核矩阵 K 和其梯度
                return K, K_gradient
        else:
            # 如果不需要计算梯度，直接返回核矩阵 K
            return K
    # 定义对象的字符串表示方法，用于返回对象的描述信息
    def __repr__(self):
        # 如果是各向异性（anisotropic）的情况
        if self.anisotropic:
            # 格式化字符串，返回对象的类名和长度尺度列表
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__,  # 返回对象的类名
                ", ".join(map("{0:.3g}".format, self.length_scale)),  # 将长度尺度列表格式化为字符串
            )
        else:  # 如果是各向同性（isotropic）的情况
            # 格式化字符串，返回对象的类名和第一个长度尺度值
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__,  # 返回对象的类名
                np.ravel(self.length_scale)[0]  # 获取长度尺度数组的第一个值并格式化为字符串
            )
# Matern核函数，继承自RBF核函数
class Matern(RBF):
    """Matern kernel.

    The class of Matern kernels is a generalization of the :class:`RBF`.
    It has an additional parameter :math:`\\nu` which controls the
    smoothness of the resulting function. The smaller :math:`\\nu`,
    the less smooth the approximated function is.
    As :math:`\\nu\\rightarrow\\infty`, the kernel becomes equivalent to
    the :class:`RBF` kernel. When :math:`\\nu = 1/2`, the Matérn kernel
    becomes identical to the absolute exponential kernel.
    Important intermediate values are
    :math:`\\nu=1.5` (once differentiable functions)
    and :math:`\\nu=2.5` (twice differentiable functions).

    The kernel is given by:

    .. math::
         k(x_i, x_j) =  \\frac{1}{\\Gamma(\\nu)2^{\\nu-1}}\\Bigg(
         \\frac{\\sqrt{2\\nu}}{l} d(x_i , x_j )
         \\Bigg)^\\nu K_\\nu\\Bigg(
         \\frac{\\sqrt{2\\nu}}{l} d(x_i , x_j )\\Bigg)

    where :math:`d(\\cdot,\\cdot)` is the Euclidean distance,
    :math:`K_{\\nu}(\\cdot)` is a modified Bessel function and
    :math:`\\Gamma(\\cdot)` is the gamma function.
    See [1]_, Chapter 4, Section 4.2, for details regarding the different
    variants of the Matern kernel.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    length_scale : float or ndarray of shape (n_features,), default=1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.

    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'length_scale'.
        If set to "fixed", 'length_scale' cannot be changed during
        hyperparameter tuning.

    nu : float, default=1.5
        The parameter nu controlling the smoothness of the learned function.
        The smaller nu, the less smooth the approximated function is.
        For nu=inf, the kernel becomes equivalent to the RBF kernel and for
        nu=0.5 to the absolute exponential kernel. Important intermediate
        values are nu=1.5 (once differentiable functions) and nu=2.5
        (twice differentiable functions). Note that values of nu not in
        [0.5, 1.5, 2.5, inf] incur a considerably higher computational cost
        (appr. 10 times higher) since they require to evaluate the modified
        Bessel function. Furthermore, in contrast to l, nu is kept fixed to
        its initial value and not optimized.

    References
    ----------
    .. [1] `Carl Edward Rasmussen, Christopher K. I. Williams (2006).
        "Gaussian Processes for Machine Learning". The MIT Press.
        <http://www.gaussianprocess.org/gpml/>`_

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import Matern
    >>> X, y = load_iris(return_X_y=True)
    # 加载鸢尾花数据集，将特征存储在 X 中，将目标变量存储在 y 中

    >>> kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
    # 定义 Matern 核函数，并设置其参数 length_scale 为 1.0，参数 nu 为 1.5

    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y)
    # 使用定义的核函数创建高斯过程分类器对象 gpc，使用随机种子 0 进行训练，训练数据为 X 和 y

    >>> gpc.score(X, y)
    # 计算高斯过程分类器 gpc 在训练数据 X, y 上的准确率得分

    0.9866...
    # 准确率得分为约 0.9866

    >>> gpc.predict_proba(X[:2,:])
    # 预测前两个样本的类别概率，X[:2,:] 表示取前两个样本的所有特征

    array([[0.8513..., 0.0368..., 0.1117...],
            [0.8086..., 0.0693..., 0.1220...]])
    # 返回一个数组，包含了前两个样本属于各个类别的概率
    ```

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=1.5):
        # 调用父类初始化方法，设置 length_scale 和 length_scale_bounds 参数
        super().__init__(length_scale, length_scale_bounds)
        # 设置对象的 nu 参数
        self.nu = nu

    def __repr__(self):
        if self.anisotropic:
            # 如果核函数是各向异性的，返回格式化后的字符串表示
            return "{0}(length_scale=[{1}], nu={2:.3g})".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.length_scale)),
                self.nu,
            )
        else:
            # 如果核函数是同向异性的，返回格式化后的字符串表示
            return "{0}(length_scale={1:.3g}, nu={2:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0], self.nu
            )
# RationalQuadratic 类定义了一个有理二次核函数，继承自 StationaryKernelMixin、NormalizedKernelMixin 和 Kernel。
# 它可以被看作是具有不同特征长度尺度的 RBF 核函数的缩放混合（无穷和）。参数化包括长度尺度参数 l > 0 和缩放混合参数 α > 0。
# 目前仅支持长度尺度为标量的各向同性变体。
class RationalQuadratic(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """Rational Quadratic kernel.

    The RationalQuadratic kernel can be seen as a scale mixture (an infinite
    sum) of RBF kernels with different characteristic length scales. It is
    parameterized by a length scale parameter :math:`l>0` and a scale
    mixture parameter :math:`\\alpha>0`. Only the isotropic variant
    where length_scale :math:`l` is a scalar is supported at the moment.
    The kernel is given by:

    .. math::
        k(x_i, x_j) = \\left(
        1 + \\frac{d(x_i, x_j)^2 }{ 2\\alpha  l^2}\\right)^{-\\alpha}

    where :math:`\\alpha` is the scale mixture parameter, :math:`l` is
    the length scale of the kernel and :math:`d(\\cdot,\\cdot)` is the
    Euclidean distance.
    For advice on how to set the parameters, see e.g. [1]_.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    length_scale : float > 0, default=1.0
        The length scale of the kernel.

    alpha : float > 0, default=1.0
        Scale mixture parameter

    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'length_scale'.
        If set to "fixed", 'length_scale' cannot be changed during
        hyperparameter tuning.

    alpha_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'alpha'.
        If set to "fixed", 'alpha' cannot be changed during
        hyperparameter tuning.

    References
    ----------
    .. [1] `David Duvenaud (2014). "The Kernel Cookbook:
        Advice on Covariance functions".
        <https://www.cs.toronto.edu/~duvenaud/cookbook/>`_

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import RationalQuadratic
    >>> X, y = load_iris(return_X_y=True)
    >>> kernel = RationalQuadratic(length_scale=1.0, alpha=1.5)
    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpc.score(X, y)
    0.9733...
    >>> gpc.predict_proba(X[:2,:])
    array([[0.8881..., 0.0566..., 0.05518...],
            [0.8678..., 0.0707... , 0.0614...]])
    """

    # 初始化方法，设置有理二次核函数的长度尺度和缩放混合参数，以及它们的上下界
    def __init__(
        self,
        length_scale=1.0,
        alpha=1.0,
        length_scale_bounds=(1e-5, 1e5),
        alpha_bounds=(1e-5, 1e5),
    ):
        self.length_scale = length_scale
        self.alpha = alpha
        self.length_scale_bounds = length_scale_bounds
        self.alpha_bounds = alpha_bounds

    # 返回长度尺度的超参数对象，用于在超参数调优期间使用
    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    # 返回缩放混合参数的超参数对象，用于在超参数调优期间使用
    @property
    def hyperparameter_alpha(self):
        return Hyperparameter("alpha", "numeric", self.alpha_bounds)
    def __call__(self, X, Y=None, eval_gradient=False):
        """
        Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        if len(np.atleast_1d(self.length_scale)) > 1:
            # 检查是否为等距版本的RationalQuadratic核，只支持等距版本
            raise AttributeError(
                "RationalQuadratic kernel only supports isotropic version, "
                "please use a single scalar for length_scale"
            )
        X = np.atleast_2d(X)
        if Y is None:
            # 计算X与自身之间的欧氏距离的平方
            dists = squareform(pdist(X, metric="sqeuclidean"))
            tmp = dists / (2 * self.alpha * self.length_scale**2)
            base = 1 + tmp
            # 计算核矩阵K
            K = base**-self.alpha
            np.fill_diagonal(K, 1)  # 将对角线上的元素设为1
        else:
            if eval_gradient:
                # 当Y不为None时，不能计算梯度
                raise ValueError("Gradient can only be evaluated when Y is None.")
            # 计算X与Y之间的欧氏距离的平方
            dists = cdist(X, Y, metric="sqeuclidean")
            # 计算核矩阵K
            K = (1 + dists / (2 * self.alpha * self.length_scale**2)) ** -self.alpha

        if eval_gradient:
            # 关于length_scale的梯度
            if not self.hyperparameter_length_scale.fixed:
                length_scale_gradient = dists * K / (self.length_scale**2 * base)
                length_scale_gradient = length_scale_gradient[:, :, np.newaxis]
            else:  # length_scale被固定
                length_scale_gradient = np.empty((K.shape[0], K.shape[1], 0))

            # 关于alpha的梯度
            if not self.hyperparameter_alpha.fixed:
                alpha_gradient = K * (
                    -self.alpha * np.log(base)
                    + dists / (2 * self.length_scale**2 * base)
                )
                alpha_gradient = alpha_gradient[:, :, np.newaxis]
            else:  # alpha被固定
                alpha_gradient = np.empty((K.shape[0], K.shape[1], 0))

            return K, np.dstack((alpha_gradient, length_scale_gradient))
        else:
            return K

    def __repr__(self):
        # 返回对象的字符串表示，包括alpha和length_scale的值
        return "{0}(alpha={1:.3g}, length_scale={2:.3g})".format(
            self.__class__.__name__, self.alpha, self.length_scale
        )
class ExpSineSquared(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    r"""Exp-Sine-Squared kernel (aka periodic kernel).

    The ExpSineSquared kernel allows one to model functions which repeat
    themselves exactly. It is parameterized by a length scale
    parameter :math:`l>0` and a periodicity parameter :math:`p>0`.
    Only the isotropic variant where :math:`l` is a scalar is
    supported at the moment. The kernel is given by:

    .. math::
        k(x_i, x_j) = \text{exp}\left(-
        \frac{ 2\sin^2(\pi d(x_i, x_j)/p) }{ l^ 2} \right)

    where :math:`l` is the length scale of the kernel, :math:`p` the
    periodicity of the kernel and :math:`d(\cdot,\cdot)` is the
    Euclidean distance.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------

    length_scale : float > 0, default=1.0
        The length scale of the kernel.

    periodicity : float > 0, default=1.0
        The periodicity of the kernel.

    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'length_scale'.
        If set to "fixed", 'length_scale' cannot be changed during
        hyperparameter tuning.

    periodicity_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'periodicity'.
        If set to "fixed", 'periodicity' cannot be changed during
        hyperparameter tuning.

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import ExpSineSquared
    >>> X, y = make_friedman2(n_samples=50, noise=0, random_state=0)
    >>> kernel = ExpSineSquared(length_scale=1, periodicity=1)
    >>> gpr = GaussianProcessRegressor(kernel=kernel, alpha=5,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    0.0144...
    >>> gpr.predict(X[:2,:], return_std=True)
    (array([425.6..., 457.5...]), array([0.3894..., 0.3467...]))

    """

    def __init__(
        self,
        length_scale=1.0,
        periodicity=1.0,
        length_scale_bounds=(1e-5, 1e5),
        periodicity_bounds=(1e-5, 1e5),
    ):
        """
        Initialize the ExpSineSquared kernel with specified parameters.

        Parameters
        ----------
        length_scale : float, optional
            The length scale of the kernel.
        periodicity : float, optional
            The periodicity of the kernel.
        length_scale_bounds : tuple of floats or str, optional
            Bounds for the length scale hyperparameter.
        periodicity_bounds : tuple of floats or str, optional
            Bounds for the periodicity hyperparameter.
        """
        # Initialize the kernel with given parameters
        self.length_scale = length_scale
        self.periodicity = periodicity
        self.length_scale_bounds = length_scale_bounds
        self.periodicity_bounds = periodicity_bounds

    @property
    def hyperparameter_length_scale(self):
        """Returns the hyperparameter object for length scale."""
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    @property
    def hyperparameter_periodicity(self):
        """Returns the hyperparameter object for periodicity."""
        return Hyperparameter("periodicity", "numeric", self.periodicity_bounds)
    def __call__(self, X, Y=None, eval_gradient=False):
        """
        Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X = np.atleast_2d(X)  # 将输入 X 至少转换为二维数组
        if Y is None:
            dists = squareform(pdist(X, metric="euclidean"))  # 计算 X 中样本之间的欧氏距离
            arg = np.pi * dists / self.periodicity  # 计算用于核函数的参数
            sin_of_arg = np.sin(arg)  # 计算参数的正弦值
            K = np.exp(-2 * (sin_of_arg / self.length_scale) ** 2)  # 计算核函数值
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = cdist(X, Y, metric="euclidean")  # 计算 X 和 Y 之间的欧氏距离
            K = np.exp(
                -2 * (np.sin(np.pi / self.periodicity * dists) / self.length_scale) ** 2
            )  # 计算核函数值

        if eval_gradient:
            cos_of_arg = np.cos(arg)  # 计算参数的余弦值
            # gradient with respect to length_scale
            if not self.hyperparameter_length_scale.fixed:
                length_scale_gradient = 4 / self.length_scale**2 * sin_of_arg**2 * K
                length_scale_gradient = length_scale_gradient[:, :, np.newaxis]
            else:  # length_scale is kept fixed
                length_scale_gradient = np.empty((K.shape[0], K.shape[1], 0))
            # gradient with respect to p
            if not self.hyperparameter_periodicity.fixed:
                periodicity_gradient = (
                    4 * arg / self.length_scale**2 * cos_of_arg * sin_of_arg * K
                )
                periodicity_gradient = periodicity_gradient[:, :, np.newaxis]
            else:  # p is kept fixed
                periodicity_gradient = np.empty((K.shape[0], K.shape[1], 0))

            return K, np.dstack((length_scale_gradient, periodicity_gradient))
        else:
            return K

    def __repr__(self):
        return "{0}(length_scale={1:.3g}, periodicity={2:.3g})".format(
            self.__class__.__name__, self.length_scale, self.periodicity
        )
# 定义一个 DotProduct 类，继承自 Kernel 类
class DotProduct(Kernel):
    r"""Dot-Product kernel.

    The DotProduct kernel is non-stationary and can be obtained from linear
    regression by putting :math:`N(0, 1)` priors on the coefficients
    of :math:`x_d (d = 1, . . . , D)` and a prior of :math:`N(0, \sigma_0^2)`
    on the bias. The DotProduct kernel is invariant to a rotation of
    the coordinates about the origin, but not translations.
    It is parameterized by a parameter sigma_0 :math:`\sigma`
    which controls the inhomogenity of the kernel. For :math:`\sigma_0^2 =0`,
    the kernel is called the homogeneous linear kernel, otherwise
    it is inhomogeneous. The kernel is given by

    .. math::
        k(x_i, x_j) = \sigma_0 ^ 2 + x_i \cdot x_j

    The DotProduct kernel is commonly combined with exponentiation.

    See [1]_, Chapter 4, Section 4.2, for further details regarding the
    DotProduct kernel.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    sigma_0 : float >= 0, default=1.0
        Parameter controlling the inhomogenity of the kernel. If sigma_0=0,
        the kernel is homogeneous.

    sigma_0_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'sigma_0'.
        If set to "fixed", 'sigma_0' cannot be changed during
        hyperparameter tuning.

    References
    ----------
    .. [1] `Carl Edward Rasmussen, Christopher K. I. Williams (2006).
        "Gaussian Processes for Machine Learning". The MIT Press.
        <http://www.gaussianprocess.org/gpml/>`_

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = DotProduct() + WhiteKernel()
    >>> gpr = GaussianProcessRegressor(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    0.3680...
    >>> gpr.predict(X[:2,:], return_std=True)
    (array([653.0..., 592.1...]), array([316.6..., 316.6...]))
    """

    # 定义初始化方法，设置 sigma_0 和 sigma_0_bounds 参数
    def __init__(self, sigma_0=1.0, sigma_0_bounds=(1e-5, 1e5)):
        self.sigma_0 = sigma_0  # 设置 sigma_0 参数
        self.sigma_0_bounds = sigma_0_bounds  # 设置 sigma_0_bounds 参数

    # 定义 hyperparameter_sigma_0 属性方法，返回 sigma_0 的超参数信息
    @property
    def hyperparameter_sigma_0(self):
        return Hyperparameter("sigma_0", "numeric", self.sigma_0_bounds)
    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims),\
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        # 将输入 X 至少视为二维数组
        X = np.atleast_2d(X)
        
        if Y is None:
            # 计算 k(X, X) 并加上初始方差平方 sigma_0^2
            K = np.inner(X, X) + self.sigma_0**2
        else:
            if eval_gradient:
                # 如果 eval_gradient=True，则抛出异常，因为此时不支持计算梯度
                raise ValueError("Gradient can only be evaluated when Y is None.")
            # 计算 k(X, Y) 并加上初始方差平方 sigma_0^2
            K = np.inner(X, Y) + self.sigma_0**2

        if eval_gradient:
            if not self.hyperparameter_sigma_0.fixed:
                # 如果未固定超参数 sigma_0，则计算 k(X, X) 对于超参数的梯度
                K_gradient = np.empty((K.shape[0], K.shape[1], 1))
                K_gradient[..., 0] = 2 * self.sigma_0**2
                return K, K_gradient
            else:
                # 如果超参数 sigma_0 已固定，则返回一个空的梯度数组
                return K, np.empty((X.shape[0], X.shape[0], 0))
        else:
            # 如果不需要计算梯度，则只返回 k(X, Y) 或者 k(X, X)
            return K

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y).

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X).
        """
        # 计算并返回 k(X, X) 的对角线元素
        return np.einsum("ij,ij->i", X, X) + self.sigma_0**2

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        # 始终返回 False，表明该核函数不是平稳的
        return False

    def __repr__(self):
        # 返回核函数对象的字符串表示，包括初始方差 sigma_0 的值
        return "{0}(sigma_0={1:.3g})".format(self.__class__.__name__, self.sigma_0)
# 从 scipy/optimize/optimize.py 中适配用于具有二维输出函数的近似梯度函数
def _approx_fprime(xk, f, epsilon, args=()):
    # 计算函数在 xk 处的函数值 f0
    f0 = f(*((xk,) + args))
    # 初始化一个与 f0 形状相同的全零数组作为梯度 grad
    grad = np.zeros((f0.shape[0], f0.shape[1], len(xk)), float)
    # 初始化单位向量 ei
    ei = np.zeros((len(xk),), float)
    # 遍历 xk 的每个维度
    for k in range(len(xk)):
        # 设置 ei[k] 为 1.0
        ei[k] = 1.0
        # 计算扰动 d
        d = epsilon * ei
        # 计算偏导数 grad[:, :, k]
        grad[:, :, k] = (f(*((xk + d,) + args)) - f0) / d[k]
        # 将 ei[k] 重置为 0.0
        ei[k] = 0.0
    # 返回计算得到的梯度 grad
    return grad


class PairwiseKernel(Kernel):
    """sklearn.metrics.pairwise 中核函数的包装器。

    对 sklearn.metrics.pairwise 中核函数功能的轻量封装。

    注意：eval_gradient 的评估是数值的而非解析的，所有核函数仅支持各向同性距离。
         参数 gamma 被视为超参数，可以进行优化。其他核参数在初始化时直接设置并保持不变。

    .. versionadded:: 0.18

    Parameters
    ----------
    gamma : float, default=1.0
        由 metric 指定的核函数的参数 gamma。应为正数。

    gamma_bounds : 一对 >= 0 的浮点数或 "fixed"，默认为 (1e-5, 1e5)
        'gamma' 的下限和上限。
        如果设置为 "fixed"，在超参数调优期间 'gamma' 不能改变。

    metric : {"linear", "additive_chi2", "chi2", "poly", "polynomial", \
              "rbf", "laplacian", "sigmoid", "cosine"} 或可调用函数，默认为 "linear"
        计算特征数组中实例之间核的度量标准。如果 metric 是字符串，必须是 pairwise.PAIRWISE_KERNEL_FUNCTIONS 中的一个度量标准。
        如果 metric 是 "precomputed"，假定 X 是一个核矩阵。
        或者，如果 metric 是可调用函数，则对 X 中的每对实例（行）调用该函数，并记录结果值。该可调用函数应接受两个来自 X 的数组作为输入，并返回一个指示它们之间距离的值。

    pairwise_kernels_kwargs : dict， 默认为 None
        该字典的所有条目（如果有）作为关键字参数传递给 pairwise 核函数。

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import PairwiseKernel
    >>> X, y = load_iris(return_X_y=True)
    >>> kernel = PairwiseKernel(metric='rbf')
    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpc.score(X, y)
    0.9733...
    >>> gpc.predict_proba(X[:2,:])
    array([[0.8880..., 0.05663..., 0.05532...],
           [0.8676..., 0.07073..., 0.06165...]])
    """

    def __init__(
        self,
        gamma=1.0,
        gamma_bounds=(1e-5, 1e5),
        metric="linear",
        pairwise_kernels_kwargs=None,
    ):
        self.gamma = gamma
        self.gamma_bounds = gamma_bounds
        self.metric = metric
        self.pairwise_kernels_kwargs = pairwise_kernels_kwargs

    @property
    def hyperparameter_gamma(self):
        # 返回一个描述 gamma 超参数的 Hyperparameter 对象，类型为 numeric，取值范围为 gamma_bounds
        return Hyperparameter("gamma", "numeric", self.gamma_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims),\
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        pairwise_kernels_kwargs = self.pairwise_kernels_kwargs
        if self.pairwise_kernels_kwargs is None:
            # 如果没有指定 pairwise_kernels_kwargs，则设为空字典
            pairwise_kernels_kwargs = {}

        X = np.atleast_2d(X)
        # 计算核函数 k(X, Y)
        K = pairwise_kernels(
            X,
            Y,
            metric=self.metric,
            gamma=self.gamma,
            filter_params=True,
            **pairwise_kernels_kwargs,
        )
        if eval_gradient:
            if self.hyperparameter_gamma.fixed:
                # 如果 gamma 超参数是固定的，则返回 K 和一个空的梯度数组
                return K, np.empty((X.shape[0], X.shape[0], 0))
            else:
                # 否则，通过数值方法近似计算梯度
                def f(gamma):  # 辅助函数
                    return pairwise_kernels(
                        X,
                        Y,
                        metric=self.metric,
                        gamma=np.exp(gamma),
                        filter_params=True,
                        **pairwise_kernels_kwargs,
                    )

                # 返回 K 和计算得到的梯度
                return K, _approx_fprime(self.theta, f, 1e-10)
        else:
            # 如果不计算梯度，直接返回 K
            return K
    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        # 使用 np.apply_along_axis 函数计算核函数对 X 中每个样本的对角线值，并展平为一维数组
        return np.apply_along_axis(self, 1, X).ravel()

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        # 返回核函数是否是平稳的，判断标准是核函数的度量方式是否为 "rbf"
        return self.metric in ["rbf"]

    def __repr__(self):
        # 返回核函数对象的字符串表示，包括类名、gamma 参数和度量方式
        return "{0}(gamma={1}, metric={2})".format(
            self.__class__.__name__, self.gamma, self.metric
        )
```