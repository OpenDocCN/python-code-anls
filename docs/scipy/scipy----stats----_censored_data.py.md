# `D:\src\scipysrc\scipy\scipy\stats\_censored_data.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数值计算


def _validate_1d(a, name, allow_inf=False):
    if np.ndim(a) != 1:  # 检查数组 a 是否是一维的
        raise ValueError(f'`{name}` must be a one-dimensional sequence.')
    if np.isnan(a).any():  # 检查数组 a 是否包含 NaN
        raise ValueError(f'`{name}` must not contain nan.')
    if not allow_inf and np.isinf(a).any():  # 如果不允许无穷大且数组 a 包含无穷大数值
        raise ValueError(f'`{name}` must contain only finite values.')


def _validate_interval(interval):
    interval = np.asarray(interval)  # 将 interval 转换为 NumPy 数组
    if interval.shape == (0,):  # 如果输入的序列长度为 0
        # The input was a sequence with length 0.
        interval = interval.reshape((0, 2))  # 将 interval 重塑为 (0, 2) 的形状
    if interval.ndim != 2 or interval.shape[-1] != 2:  # 检查 interval 是否为二维数组且每行有两个元素
        raise ValueError('`interval` must be a two-dimensional array with '
                         'shape (m, 2), where m is the number of '
                         'interval-censored values, but got shape '
                         f'{interval.shape}')

    if np.isnan(interval).any():  # 检查 interval 是否包含 NaN
        raise ValueError('`interval` must not contain nan.')
    if np.isinf(interval).all(axis=1).any():  # 检查每行的两个值是否同时为无穷大
        raise ValueError('In each row in `interval`, both values must not'
                         ' be infinite.')
    if (interval[:, 0] > interval[:, 1]).any():  # 检查每行的左值是否小于等于右值
        raise ValueError('In each row of `interval`, the left value must not'
                         ' exceed the right value.')

    # 创建不同掩码以筛选出不同类型的数据
    uncensored_mask = interval[:, 0] == interval[:, 1]
    left_mask = np.isinf(interval[:, 0])
    right_mask = np.isinf(interval[:, 1])
    interval_mask = np.isfinite(interval).all(axis=1) & ~uncensored_mask

    # 根据掩码选择不同类型的数据
    uncensored2 = interval[uncensored_mask, 0]
    left2 = interval[left_mask, 1]
    right2 = interval[right_mask, 0]
    interval2 = interval[interval_mask]

    return uncensored2, left2, right2, interval2


def _validate_x_censored(x, censored):
    x = np.asarray(x)  # 将 x 转换为 NumPy 数组
    if x.ndim != 1:  # 检查 x 是否为一维数组
        raise ValueError('`x` must be one-dimensional.')
    censored = np.asarray(censored)  # 将 censored 转换为 NumPy 数组
    if censored.ndim != 1:  # 检查 censored 是否为一维数组
        raise ValueError('`censored` must be one-dimensional.')
    if (~np.isfinite(x)).any():  # 检查 x 是否包含 NaN 或无穷大
        raise ValueError('`x` must not contain nan or inf.')
    if censored.size != x.size:  # 检查 x 和 censored 的长度是否一致
        raise ValueError('`x` and `censored` must have the same length.')
    return x, censored.astype(bool)


class CensoredData:
    """
    Instances of this class represent censored data.

    Instances may be passed to the ``fit`` method of continuous
    univariate SciPy distributions for maximum likelihood estimation.
    The *only* method of the univariate continuous distributions that
    understands `CensoredData` is the ``fit`` method.  An instance of
    `CensoredData` can not be passed to methods such as ``pdf`` and
    ``cdf``.

    An observation is said to be *censored* when the precise value is unknown,
    but it has a known upper and/or lower bound.  The conventional terminology
    is:

    * left-censored: an observation is below a certain value but it is
      unknown by how much.
    """
    # 在统计学中，观测数据可以被分为四种类型：无法观测的、左截断的、右截断的和区间截断的。
    # * 无法观测的：观测值存在，但具体数值未知。
    # * 左截断的：观测值低于某个值，但具体数值未知。
    # * 右截断的：观测值高于某个值，但具体数值未知。
    # * 区间截断的：观测值位于某个区间内，但具体数值未知。
    
    # `CensoredData` 类用于方便地处理这些截断数据类型。
    # 提供了类方法 `left_censored` 和 `right_censored`，可以根据一个一维测量数组和对应的布尔数组（指示哪些测量是截断的）创建一个 `CensoredData` 实例。
    # 类方法 `interval_censored` 接受两个一维数组，表示区间截断的下限和上限。
    
    class CensoredData:
        """
        Parameters
        ----------
        uncensored : array_like, 1D
            无截断的观测值。
        left : array_like, 1D
            左截断的观测值。
        right : array_like, 1D
            右截断的观测值。
        interval : array_like, 2D，形状为 (m, 2)
            区间截断的观测值。每行 `interval[k, :]` 表示第 k 个区间截断观测值的区间。
    
        Notes
        -----
        在输入数组 `interval` 中，区间的下限可以是 `-inf`，上限可以是 `inf`，但至少其中一个必须是有限值。
        当下限是 `-inf` 时，该行表示左截断观测值；当上限是 `inf` 时，该行表示右截断观测值。
        如果一个区间的长度为 0（即 `interval[k, 0] == interval[k, 1]`），则表示该观测值是无截断的。
        因此，可以使用 `uncensored`、`left` 和 `right` 来分别表示无截断、左截断和右截断的观测值。
        但通常更方便使用 `uncensored`、`left` 和 `right` 来表示所有类型的截断和无截断数据。
    
        Examples
        --------
        在最一般的情况下，截断数据集可能包含左截断、右截断、区间截断和无截断的观测值。
        例如，下面创建一个包含五个观测值的数据集。其中两个是无截断的观测值（值为 1 和 1.5），
        一个是左截断观测值（值为 0），一个是右截断观测值（值为 10），还有一个是区间截断在 [2, 3] 内的观测值。
    
        >>> import numpy as np
        >>> from scipy.stats import CensoredData
        >>> data = CensoredData(uncensored=[1, 1.5], left=[0], right=[10],
        ...                     interval=[[2, 3]])
        >>> print(data)
        CensoredData(5 values: 2 not censored, 1 left-censored,
        1 right-censored, 1 interval-censored)
    
        等价地，
    
        >>> data = CensoredData(interval=[[1, 1],
        ...                               [1.5, 1.5],
        ...                               [-np.inf, 0],
        ...                               [10, np.inf],
        ...                               [2, 3]])
        >>> print(data)
        CensoredData(5 values: 2 not censored, 1 left-censored,
        1 right-censored, 1 interval-censored)
    A common case is to have a mix of uncensored observations and censored
    observations that are all right-censored (or all left-censored). For
    example, consider an experiment in which six devices are started at
    various times and left running until they fail.  Assume that time is
    measured in hours, and the experiment is stopped after 30 hours, even
    if all the devices have not failed by that time.  We might end up with
    data such as this::

        Device  Start-time  Fail-time  Time-to-failure
           1         0         13           13
           2         2         24           22
           3         5         22           17
           4         8         23           15
           5        10        ***          >20
           6        12        ***          >18

    Two of the devices had not failed when the experiment was stopped;
    the observations of the time-to-failure for these two devices are
    right-censored.  We can represent this data with

    >>> data = CensoredData(uncensored=[13, 22, 17, 15], right=[20, 18])
    >>> print(data)
    CensoredData(6 values: 4 not censored, 2 right-censored)

    Alternatively, we can use the method `CensoredData.right_censored` to
    create a representation of this data.  The time-to-failure observations
    are put the list ``ttf``.  The ``censored`` list indicates which values
    in ``ttf`` are censored.

    >>> ttf = [13, 22, 17, 15, 20, 18]
    >>> censored = [False, False, False, False, True, True]

    Pass these lists to `CensoredData.right_censored` to create an
    instance of `CensoredData`.

    >>> data = CensoredData.right_censored(ttf, censored)
    >>> print(data)
    CensoredData(6 values: 4 not censored, 2 right-censored)

    If the input data is interval censored and already stored in two
    arrays, one holding the low end of the intervals and another
    holding the high ends, the class method ``interval_censored`` can
    be used to create the `CensoredData` instance.

    This example creates an instance with four interval-censored values.
    The intervals are [10, 11], [0.5, 1], [2, 3], and [12.5, 13.5].

    >>> a = [10, 0.5, 2, 12.5]  # Low ends of the intervals
    >>> b = [11, 1.0, 3, 13.5]  # High ends of the intervals
    >>> data = CensoredData.interval_censored(low=a, high=b)
    >>> print(data)
    CensoredData(4 values: 0 not censored, 4 interval-censored)

    Finally, we create and censor some data from the `weibull_min`
    distribution, and then fit `weibull_min` to that data. We'll assume
    that the location parameter is known to be 0.

    >>> from scipy.stats import weibull_min
    >>> rng = np.random.default_rng()

    Create the random data set.

    >>> x = weibull_min.rvs(2.5, loc=0, scale=30, size=250, random_state=rng)
    >>> x[x > 40] = 40  # Right-censor values greater or equal to 40.

    Create the `CensoredData` instance with the `right_censored` method.
    # CensoredData 类，用于处理被审查的数据，支持不同类型的审查数据
    """

    # 初始化函数，用于创建 CensoredData 对象
    def __init__(self, uncensored=None, *, left=None, right=None,
                 interval=None):
        # 如果未指定未被审查的数据，默认为空列表
        if uncensored is None:
            uncensored = []
        # 如果未指定左审查的数据，默认为空列表
        if left is None:
            left = []
        # 如果未指定右审查的数据，默认为空列表
        if right is None:
            right = []
        # 如果未指定区间审查的数据，默认为一个空的 2D NumPy 数组
        if interval is None:
            interval = np.empty((0, 2))

        # 使用 _validate_1d 函数验证各类型数据的一维性
        _validate_1d(uncensored, 'uncensored')
        _validate_1d(left, 'left')
        _validate_1d(right, 'right')
        # 使用 _validate_interval 函数验证区间审查的数据，并返回验证后的数据
        uncensored2, left2, right2, interval2 = _validate_interval(interval)

        # 将所有类型的审查数据连接起来，构成对象的私有属性
        self._uncensored = np.concatenate((uncensored, uncensored2))
        self._left = np.concatenate((left, left2))
        self._right = np.concatenate((right, right2))
        # 注意：_interval 是一个二维数组，只包含有限值，表示长度非零的区间
        self._interval = interval2

    # 返回对象的详细字符串表示形式，用于调试和输出
    def __repr__(self):
        # 将各类型数据转换成可打印的字符串形式
        uncensored_str = " ".join(np.array_repr(self._uncensored).split())
        left_str = " ".join(np.array_repr(self._left).split())
        right_str = " ".join(np.array_repr(self._right).split())
        interval_str = " ".join(np.array_repr(self._interval).split())
        # 返回 CensoredData 对象的字符串表示，包括未审查、左审查、右审查和区间审查的数据
        return (f"CensoredData(uncensored={uncensored_str}, left={left_str}, "
                f"right={right_str}, interval={interval_str})")

    # 返回对象的简洁字符串表示形式，用于显示和输出
    def __str__(self):
        # 计算各类型数据的长度
        num_nc = len(self._uncensored)
        num_lc = len(self._left)
        num_rc = len(self._right)
        num_ic = len(self._interval)
        n = num_nc + num_lc + num_rc + num_ic
        parts = [f'{num_nc} not censored']
        if num_lc > 0:
            parts.append(f'{num_lc} left-censored')
        if num_rc > 0:
            parts.append(f'{num_rc} right-censored')
        if num_ic > 0:
            parts.append(f'{num_ic} interval-censored')
        # 返回 CensoredData 对象的简洁表示，包括各类型审查数据的数量
        return f'CensoredData({n} values: ' + ', '.join(parts) + ')'

    # 实现减法运算符的重载，用于从各类型的审查数据中减去一个标量
    def __sub__(self, other):
        # 返回新的 CensoredData 对象，其中各类型数据减去给定的标量 other
        return CensoredData(uncensored=self._uncensored - other,
                            left=self._left - other,
                            right=self._right - other,
                            interval=self._interval - other)
    def __truediv__(self, other):
        # 返回一个新的 CensoredData 实例，其各属性值分别为当前实例对应属性值除以给定的 other 参数
        return CensoredData(uncensored=self._uncensored / other,
                            left=self._left / other,
                            right=self._right / other,
                            interval=self._interval / other)

    def __len__(self):
        """
        The number of values (censored and not censored).
        """
        # 返回所有属性中的元素总数，即不被审查的、左右被审查的和区间的元素总和
        return (len(self._uncensored) + len(self._left) + len(self._right)
                + len(self._interval))

    def num_censored(self):
        """
        Number of censored values.
        """
        # 返回被审查的值的数量，包括左边界被审查、右边界被审查和区间被审查的元素总数
        return len(self._left) + len(self._right) + len(self._interval)

    @classmethod
    def right_censored(cls, x, censored):
        """
        Create a `CensoredData` instance of right-censored data.

        Parameters
        ----------
        x : array_like
            `x` is the array of observed data or measurements.
            `x` must be a one-dimensional sequence of finite numbers.
        censored : array_like of bool
            `censored` must be a one-dimensional sequence of boolean
            values.  If ``censored[k]`` is True, the corresponding value
            in `x` is right-censored.  That is, the value ``x[k]``
            is the lower bound of the true (but unknown) value.

        Returns
        -------
        data : `CensoredData`
            An instance of `CensoredData` that represents the
            collection of uncensored and right-censored values.

        Examples
        --------
        >>> from scipy.stats import CensoredData

        Two uncensored values (4 and 10) and two right-censored values
        (24 and 25).

        >>> data = CensoredData.right_censored([4, 10, 24, 25],
        ...                                    [False, False, True, True])
        >>> data
        CensoredData(uncensored=array([ 4., 10.]),
        left=array([], dtype=float64), right=array([24., 25.]),
        interval=array([], shape=(0, 2), dtype=float64))
        >>> print(data)
        CensoredData(4 values: 2 not censored, 2 right-censored)
        """
        # 验证输入的 x 和 censored 参数，确保它们符合要求
        x, censored = _validate_x_censored(x, censored)
        # 返回一个 CensoredData 的类方法，用于创建包含右截尾数据的实例
        return cls(uncensored=x[~censored], right=x[censored])

    @classmethod
    # 定义一个类方法 `left_censored`，用于创建左截尾数据的 `CensoredData` 实例

    def left_censored(cls, x, censored):
        """
        Create a `CensoredData` instance of left-censored data.

        Parameters
        ----------
        x : array_like
            `x` is the array of observed data or measurements.
            `x` must be a one-dimensional sequence of finite numbers.
        censored : array_like of bool
            `censored` must be a one-dimensional sequence of boolean
            values.  If ``censored[k]`` is True, the corresponding value
            in `x` is left-censored.  That is, the value ``x[k]``
            is the upper bound of the true (but unknown) value.

        Returns
        -------
        data : `CensoredData`
            An instance of `CensoredData` that represents the
            collection of uncensored and left-censored values.

        Examples
        --------
        >>> from scipy.stats import CensoredData

        Two uncensored values (0.12 and 0.033) and two left-censored values
        (both 1e-3).

        >>> data = CensoredData.left_censored([0.12, 0.033, 1e-3, 1e-3],
        ...                                   [False, False, True, True])
        >>> data
        CensoredData(uncensored=array([0.12 , 0.033]),
        left=array([0.001, 0.001]), right=array([], dtype=float64),
        interval=array([], shape=(0, 2), dtype=float64))
        >>> print(data)
        CensoredData(4 values: 2 not censored, 2 left-censored)
        """
        
        # 调用 `_validate_x_censored` 函数验证和处理输入的 `x` 和 `censored` 数组
        x, censored = _validate_x_censored(x, censored)
        
        # 返回一个 `CensoredData` 实例，包含未被截尾的 `x[~censored]` 和被左截尾的 `x[censored]`
        return cls(uncensored=x[~censored], left=x[censored])

    @classmethod
    def interval_censored(cls, low, high):
        """
        Create a `CensoredData` instance of interval-censored data.

        This method is useful when all the data is interval-censored, and
        the low and high ends of the intervals are already stored in
        separate one-dimensional arrays.

        Parameters
        ----------
        low : array_like
            The one-dimensional array containing the low ends of the
            intervals.
        high : array_like
            The one-dimensional array containing the high ends of the
            intervals.

        Returns
        -------
        data : `CensoredData`
            An instance of `CensoredData` that represents the
            collection of censored values.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import CensoredData

        ``a`` and ``b`` are the low and high ends of a collection of
        interval-censored values.

        >>> a = [0.5, 2.0, 3.0, 5.5]
        >>> b = [1.0, 2.5, 3.5, 7.0]
        >>> data = CensoredData.interval_censored(low=a, high=b)
        >>> print(data)
        CensoredData(4 values: 0 not censored, 4 interval-censored)
        """
        # Validate that `low` is a valid one-dimensional array
        _validate_1d(low, 'low', allow_inf=True)
        # Validate that `high` is a valid one-dimensional array
        _validate_1d(high, 'high', allow_inf=True)
        # Check if `low` and `high` have the same length
        if len(low) != len(high):
            raise ValueError('`low` and `high` must have the same length.')
        # Combine `low` and `high` arrays into a 2D array `interval`
        interval = np.column_stack((low, high))
        # Validate the interval data and retrieve components
        uncensored, left, right, interval = _validate_interval(interval)
        # Create and return a `CensoredData` instance with validated data
        return cls(uncensored=uncensored, left=left, right=right,
                   interval=interval)

    def _uncensor(self):
        """
        This function is used when a non-censored version of the data
        is needed to create a rough estimate of the parameters of a
        distribution via the method of moments or some similar method.
        The data is "uncensored" by taking the given endpoints as the
        data for the left- or right-censored data, and the mean for the
        interval-censored data.
        """
        # Concatenate arrays `_uncensored`, `_left`, `_right`, and mean of `_interval`
        data = np.concatenate((self._uncensored, self._left, self._right,
                               self._interval.mean(axis=1)))
        # Return the concatenated data array
        return data

    def _supported(self, a, b):
        """
        Return a subset of self containing the values that are in
        (or overlap with) the interval (a, b).
        """
        # Retrieve `_uncensored` data within interval (a, b)
        uncensored = self._uncensored
        uncensored = uncensored[(a < uncensored) & (uncensored < b)]
        # Retrieve `_left` data greater than a
        left = self._left
        left = left[a < left]
        # Retrieve `_right` data less than b
        right = self._right
        right = right[right < b]
        # Retrieve `_interval` rows where the interval overlaps with (a, b)
        interval = self._interval
        interval = interval[(a < interval[:, 1]) & (interval[:, 0] < b)]
        # Return a new `CensoredData` instance with filtered data
        return CensoredData(uncensored, left=left, right=right,
                            interval=interval)
```