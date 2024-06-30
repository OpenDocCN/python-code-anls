# `D:\src\scipysrc\scipy\benchmarks\benchmarks\go_benchmark_functions\go_benchmark.py`

```
# 导入NumPy库，使用别名np
import numpy as np
# 从NumPy库中导入abs和asarray函数
from numpy import abs, asarray

# 导入safe_import函数，来自上级目录的common模块，禁止检查未使用的导入
from ..common import safe_import  # noqa:F401

# 定义Benchmark类，用于全局优化基准问题
class Benchmark:

    """
    Defines a global optimization benchmark problem.

    This abstract class defines the basic structure of a global
    optimization problem. Subclasses should implement the ``fun`` method
    for a particular optimization problem.

    Attributes
    ----------
    N : int
        The dimensionality of the problem.
    bounds : sequence
        The lower/upper bounds to be used for minimizing the problem.
        This a list of (lower, upper) tuples that contain the lower and upper
        bounds for the problem.  The problem should not be asked for evaluation
        outside these bounds. ``len(bounds) == N``.
    xmin : sequence
        The lower bounds for the problem
    xmax : sequence
        The upper bounds for the problem
    fglob : float
        The global minimum of the evaluated function.
    global_optimum : sequence
        A list of vectors that provide the locations of the global minimum.
        Note that some problems have multiple global minima, not all of which
        may be listed.
    nfev : int
        the number of function evaluations that the object has been asked to
        calculate.
    change_dimensionality : bool
        Whether we can change the benchmark function `x` variable length (i.e.,
        the dimensionality of the problem)
    custom_bounds : sequence
        a list of tuples that contain lower/upper bounds for use in plotting.
    """

    def __init__(self, dimensions):
        """
        Initialises the problem

        Parameters
        ----------

        dimensions : int
            The dimensionality of the problem
        """

        # 设置问题的维度
        self._dimensions = dimensions
        # 初始化函数评估计数
        self.nfev = 0
        # 全局最小值设为NaN
        self.fglob = np.nan
        # 全局最优位置设为None
        self.global_optimum = None
        # 是否允许改变问题的维度
        self.change_dimensionality = False
        # 自定义边界设为None
        self.custom_bounds = None

    def __str__(self):
        # 返回类的字符串表示，包含问题的维度信息
        return f'{self.__class__.__name__} ({self.N} dimensions)'

    def __repr__(self):
        # 返回类的正式表示名称
        return self.__class__.__name__

    def initial_vector(self):
        """
        Random initialisation for the benchmark problem.

        Returns
        -------
        x : sequence
            a vector of length ``N`` that contains random floating point
            numbers that lie between the lower and upper bounds for a given
            parameter.
        """

        # 返回一个长度为N的向量，包含在每个参数给定的下限和上限之间的随机浮点数
        return asarray([np.random.uniform(l, u) for l, u in self.bounds])
    def success(self, x, tol=1.e-5):
        """
        Tests if a candidate solution is at the global minimum.

        Parameters
        ----------
        x : sequence
            The candidate vector for testing if the global minimum has been
            reached. Must have ``len(x) == self.N``
        tol : float
            The maximum allowable difference between the evaluated function
            value and the known global minimum to consider the solution at
            the global minimum.

        Returns
        -------
        bool : True if the candidate vector is at the global minimum, False otherwise
        """
        # Evaluate the objective function at the candidate vector `x`
        val = self.fun(asarray(x))
        
        # Check if the evaluated function value is close enough to the known global minimum
        if abs(val - self.fglob) < tol:
            return True

        # Check if the candidate vector `x` is within the defined bounds
        bounds = np.asarray(self.bounds, dtype=np.float64)
        if np.any(x > bounds[:, 1]):
            return False
        if np.any(x < bounds[:, 0]):
            return False

        # If the evaluated value `val` is lower than the known global minimum `self.fglob`,
        # raise an error indicating an unexpected lower minimum has been found
        if val < self.fglob:
            raise ValueError("Found a lower global minimum", x, val, self.fglob)

        return False

    def fun(self, x):
        """
        Evaluation of the benchmark function.

        Parameters
        ----------
        x : sequence
            The candidate vector for evaluating the benchmark problem. Must
            have ``len(x) == self.N``.

        Returns
        -------
        val : float
              The evaluated benchmark function value at the given vector `x`.
        """
        # Placeholder function for evaluating the benchmark function,
        # to be implemented in derived classes.
        raise NotImplementedError

    def change_dimensions(self, ndim):
        """
        Changes the dimensionality of the benchmark problem if allowed.

        Parameters
        ----------
        ndim : int
               The new dimensionality for the problem.
        """
        # Check if dimensionality change is allowed (`self.change_dimensionality`)
        if self.change_dimensionality:
            # Update the dimensionality `_dimensions` to `ndim`
            self._dimensions = ndim
        else:
            # Raise an error if dimensionality change is not allowed for this problem
            raise ValueError('Dimensionality cannot be changed for this problem')

    @property
    def bounds(self):
        """
        The lower/upper bounds to be used for minimizing the problem.
        
        This returns a list of (lower, upper) tuples that define the lower and upper
        bounds for each dimension of the problem. The problem should not be evaluated
        outside these bounds. `len(bounds) == N`.
        """
        # If dimensionality change is allowed, return a list of lower bounds
        # duplicated `self.N` times
        if self.change_dimensionality:
            return [self._bounds[0]] * self.N
        else:
            # Otherwise, return the original `_bounds` list
            return self._bounds

    @property
    def N(self):
        """
        The dimensionality of the problem.

        Returns
        -------
        N : int
            The dimensionality of the problem
        """
        # Return the dimensionality `_dimensions` of the problem
        return self._dimensions
    # 定义一个方法 `xmin`，用于获取问题的下界

    def xmin(self):
        """
        The lower bounds for the problem

        Returns
        -------
        xmin : sequence
            The lower bounds for the problem
        """
        # 从 `self.bounds` 中获取每个元素的第一个值，作为问题的下界
        return asarray([b[0] for b in self.bounds])

    # 定义一个属性方法 `xmax`，用于获取问题的上界

    @property
    def xmax(self):
        """
        The upper bounds for the problem

        Returns
        -------
        xmax : sequence
            The upper bounds for the problem
        """
        # 从 `self.bounds` 中获取每个元素的第二个值，作为问题的上界
        return asarray([b[1] for b in self.bounds])
```