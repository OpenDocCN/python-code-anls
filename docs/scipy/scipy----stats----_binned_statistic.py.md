# `D:\src\scipysrc\scipy\scipy\stats\_binned_statistic.py`

```
# 导入内建模块 builtins，提供 Python 内置函数和异常类的访问
import builtins
# 导入警告模块中的 catch_warnings 和 simplefilter 函数
from warnings import catch_warnings, simplefilter
# 导入 NumPy 库，并使用别名 np
import numpy as np
# 从 operator 模块中导入 index 函数
from operator import index
# 导入 collections 模块中的 namedtuple 类
from collections import namedtuple

# 将以下标识符添加到模块的 __all__ 列表中，使它们在使用 from module import * 时可见
__all__ = ['binned_statistic',
           'binned_statistic_2d',
           'binned_statistic_dd']

# 命名元组 BinnedStatisticResult，用于存储分箱统计的结果
BinnedStatisticResult = namedtuple('BinnedStatisticResult',
                                   ('statistic', 'bin_edges', 'binnumber'))


def binned_statistic(x, values, statistic='mean',
                     bins=10, range=None):
    """
    Compute a binned statistic for one or more sets of data.

    This is a generalization of a histogram function.  A histogram divides
    the space into bins, and returns the count of the number of points in
    each bin.  This function allows the computation of the sum, mean, median,
    or other statistic of the values (or set of values) within each bin.

    Parameters
    ----------
    x : (N,) array_like
        A sequence of values to be binned.
    values : (N,) array_like or list of (N,) array_like
        The data on which the statistic will be computed.  This must be
        the same shape as `x`, or a set of sequences - each the same shape as
        `x`.  If `values` is a set of sequences, the statistic will be computed
        on each independently.
    statistic : string or callable, optional
        The statistic to compute (default is 'mean').
        The following statistics are available:

          * 'mean' : compute the mean of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'std' : compute the standard deviation within each bin. This
            is implicitly calculated with ddof=0.
          * 'median' : compute the median of values for points within each
            bin. Empty bins will be represented by NaN.
          * 'count' : compute the count of points within each bin.  This is
            identical to an unweighted histogram.  `values` array is not
            referenced.
          * 'sum' : compute the sum of values for points within each bin.
            This is identical to a weighted histogram.
          * 'min' : compute the minimum of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'max' : compute the maximum of values for point within each bin.
            Empty bins will be represented by NaN.
          * function : a user-defined function which takes a 1D array of
            values, and outputs a single numerical statistic. This function
            will be called on the values in each bin.  Empty bins will be
            represented by function([]), or NaN if this returns an error.
    bins : int or sequence of scalars, optional
        The number of bins to use. If `bins` is an int, it defines the number
        of equal-width bins in the given range (default is 10).
    range : (float, float) or [(float, float)], optional
        The lower and upper range of the bins.  If not provided, range is
        simply ``(values.min(), values.max())``.  Values outside the range are
        ignored.

    Returns
    -------
    statistic : (M,) ndarray, optional
        The values of the selected statistic in each bin.
    bin_edges : (M+1,) ndarray
        Return the bin edges ``(length M+1)``.
    binnumber : (N,) ndarray of ints
        This assigns to each element of `x` an integer that represents the bin
        in which this element falls.

    """
    # 实现对一个或多个数据集进行分箱统计的函数

    # 省略部分代码实现，具体实现可参考实际函数定义
    pass
    # bins参数定义了分箱的方式：
    #   - 如果是一个整数，则表示在给定范围内的等宽分箱数量（默认为10）。
    #   - 如果是一个序列，则表示分箱的边界，包括最右边的边界，允许非均匀的分箱宽度。
    #   - 小于最低分箱边界的x值被分配到编号为0的分箱中，大于最高分箱边界的值被分配到bins[-1]中。
    #   如果指定了分箱边界，则分箱的数量为len(bins)-1。
    bins : int or sequence of scalars, optional

    # range参数定义了分箱的范围：
    #   - 是一个元组(float, float)或者列表[(float, float)]，表示分箱的下限和上限。
    #   - 如果未提供，则范围为(x.min(), x.max())。超出范围的值将被忽略。
    range : (float, float) or [(float, float)], optional

    # 返回值：
    # statistic: 数组
    #   每个分箱中所选统计量的值。
    # bin_edges: 浮点数dtype的数组
    #   返回分箱的边界，长度为(statistic的长度+1)。
    # binnumber: 整数类型的1维ndarray
    #   每个x值属于的分箱的索引（对应于bin_edges）。与values长度相同。
    #   binnumber为i表示相应的值在(bin_edges[i-1], bin_edges[i])之间。
    #
    # 参见：
    # numpy.digitize, numpy.histogram, binned_statistic_2d, binned_statistic_dd
    #
    # 注意：
    # 所有除了最后一个（最右边的）分箱都是半开放的。换句话说，如果bins是[1, 2, 3, 4]，
    # 那么第一个分箱是[1, 2)（包括1，不包括2），第二个是[2, 3)。然而，最后一个分箱是[3, 4]，
    # 包括4。
    #
    # .. versionadded:: 0.11.0
    #
    # 示例：
    # 这里有一些基本的例子：
    #
    # 首先，创建两个在给定样本范围内均匀分布的分箱，并对每个分箱中的相应值求和：
    #
    # 多组值也可以传递进来。统计量将独立地在每个集合上计算：
    #
    # 第二个例子中，我们生成了一些随机数据，代表帆船速度。
    """
    Try to get the length of bins; if it's not possible, set N to 1.
    If N is not equal to 1, convert bins into a list of NumPy arrays.

    Args:
        bins: Input bins for binning statistic.
        range: Optional range for binning statistic.

    Returns:
        BinnedStatisticResult: Result object containing medians, edges, and bin numbers.
    """
    try:
        N = len(bins)
    except TypeError:
        N = 1

    if N != 1:
        bins = [np.asarray(bins, float)]

    # If 'range' is provided and is a tuple with 2 elements, convert it into a list with a single tuple.
    if range is not None:
        if len(range) == 2:
            range = [range]

    # Compute medians, edges, and bin numbers using binned_statistic_dd function.
    medians, edges, binnumbers = binned_statistic_dd(
        [x], values, statistic, bins, range)

    # Return a BinnedStatisticResult object containing computed results.
    return BinnedStatisticResult(medians, edges[0], binnumbers)
# 定义一个命名元组 BinnedStatistic2dResult，用于存储二维分箱统计的结果，包括统计量、x 轴边界、y 轴边界和分箱编号
BinnedStatistic2dResult = namedtuple('BinnedStatistic2dResult',
                                     ('statistic', 'x_edge', 'y_edge',
                                      'binnumber'))


# 定义函数 binned_statistic_2d，计算一个或多个数据集的二维分箱统计
def binned_statistic_2d(x, y, values, statistic='mean',
                        bins=10, range=None, expand_binnumbers=False):
    """
    Compute a bidimensional binned statistic for one or more sets of data.

    This is a generalization of a histogram2d function.  A histogram divides
    the space into bins, and returns the count of the number of points in
    each bin.  This function allows the computation of the sum, mean, median,
    or other statistic of the values (or set of values) within each bin.

    Parameters
    ----------
    x : (N,) array_like
        A sequence of values to be binned along the first dimension.
    y : (N,) array_like
        A sequence of values to be binned along the second dimension.
    values : (N,) array_like or list of (N,) array_like
        The data on which the statistic will be computed.  This must be
        the same shape as `x`, or a list of sequences - each with the same
        shape as `x`.  If `values` is such a list, the statistic will be
        computed on each independently.
    statistic : string or callable, optional
        The statistic to compute (default is 'mean').
        The following statistics are available:

          * 'mean' : compute the mean of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'std' : compute the standard deviation within each bin. This
            is implicitly calculated with ddof=0.
          * 'median' : compute the median of values for points within each
            bin. Empty bins will be represented by NaN.
          * 'count' : compute the count of points within each bin.  This is
            identical to an unweighted histogram.  `values` array is not
            referenced.
          * 'sum' : compute the sum of values for points within each bin.
            This is identical to a weighted histogram.
          * 'min' : compute the minimum of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'max' : compute the maximum of values for point within each bin.
            Empty bins will be represented by NaN.
          * function : a user-defined function which takes a 1D array of
            values, and outputs a single numerical statistic. This function
            will be called on the values in each bin.  Empty bins will be
            represented by function([]), or NaN if this returns an error.
    bins : int or [int, int] or array_like or [array, array], optional
        The bin specification:

          * the number of bins for the two dimensions (nx, ny = bins),
          * the number of bins for each dimension (nx, ny = bins),
          * or the bin edges for the two dimensions (x_edge, y_edge = bins).

    range : (2,2) array_like, optional
        The minimum and maximum values for each dimension (x_min, x_max,
        y_min, y_max). Values outside this range are ignored.
    expand_binnumbers : bool, optional
        'False' (default): the returned binnumber is a shape (N,) array of
        linear indices; 'True': the returned binnumber is 'broadcast' to the
        shape of `values`, with a bin for each dataset element.

    Returns
    -------
    statistic : ndarray
        The values of the selected statistic in each two-dimensional bin.
    x_edge : ndarray
        The bin edges along the x-axis.
    y_edge : ndarray
        The bin edges along the y-axis.
    binnumber : ndarray or list of 2D arrays
        This assigns to each element in `values` the bin number it belongs to.
        The shape of `binnumber` is either `(N,)` or `(N, D)`, where `D` is
        the number of dimensions of `bins`. If `expand_binnumbers` is `True`,
        the shape is `(N, values.ndim)`.
    """
    pass  # 函数体暂时为空，用于后续实现二维分箱统计的具体计算逻辑
    bins : int or [int, int] or array_like or [array, array], optional
        # 定义直方图的箱子（bin）规格：

          * 如果是两个维度上的相同数量的箱子 (nx = ny = bins),
          * 如果是每个维度上不同数量的箱子 (nx, ny = bins),
          * 如果是两个维度上的箱子边界 (x_edge = y_edge = bins),
          * 如果是每个维度上的箱子边界 (x_edge, y_edge = bins).

        如果指定了箱子边界，则箱子的数量将为 (nx = len(x_edge)-1, ny = len(y_edge)-1).

    range : (2,2) array_like, optional
        # 没有在 `bins` 参数中明确指定时，每个维度上箱子的最左和最右边界：
        [[xmin, xmax], [ymin, ymax]]. 所有在此范围之外的值将被视为异常值，并不计入直方图中。

    expand_binnumbers : bool, optional
        # 是否展开返回的 `binnumber`：

        'False'（默认）：返回的 `binnumber` 是一个形状为 (N,) 的数组，其中包含线性化的箱子索引。
        'True'：返回的 `binnumber` 被展开为一个形状为 (2,N) 的 ndarray，每行分别给出对应维度上的箱子编号。
        参见返回的 `binnumber` 值以及 `Examples` 部分。

        .. versionadded:: 0.17.0

    Returns
    -------
    statistic : (nx, ny) ndarray
        # 在每个二维箱子中选定统计量的数值。
    x_edge : (nx + 1) ndarray
        # 第一维度上的箱子边界。
    y_edge : (ny + 1) ndarray
        # 第二维度上的箱子边界。
    binnumber : (N,) array of ints or (2,N) ndarray of ints
        # 将每个 `sample` 元素分配到表示其所属箱子的整数中。表示取决于 `expand_binnumbers` 参数。详见 `Notes`。

    See Also
    --------
    numpy.digitize, numpy.histogram2d, binned_statistic, binned_statistic_dd

    Notes
    -----
    Binedges:
    # 所有但最后一个（最右边的）箱子都是半开放的。换句话说，如果 `bins` 是 `[1, 2, 3, 4]`，那么第一个箱子是 `[1, 2)`（包含1但不包含2），第二个是 `[2, 3)`。然而，最后一个箱子是 `[3, 4]`，包含4。

    `binnumber`:
    # 此返回的参数将每个 `sample` 元素分配到表示其所属箱子的整数中。表示取决于 `expand_binnumbers` 参数。
    如果为 'False'（默认）：返回的 `binnumber` 是一个形状为 (N,) 的数组，其中包含用于将每个 `sample` 元素映射到其对应箱子的线性化索引（使用行优先顺序）。
    请注意，返回的线性化箱子索引用于具有在外部箱子边界上的额外箱子以捕获超出定义箱子边界的值的数组。
    如果为 'True'：返回的 `binnumber` 是一个形状为 (2,N) 的 ndarray，其中每行分别表示每个维度上的箱子放置情况。
    ```
    # 此代码片段为 scipy 的 binned_statistic_2d 函数的文档示例及其实现部分
    
    # 尝试获取 bins 的长度，以确定其维度 N
    try:
        N = len(bins)
    except TypeError:
        N = 1
    
    # 如果 bins 不是长度为 1 或 2 的数组，则将其转换为包含 xedges 和 yedges 的数组
    if N != 1 and N != 2:
        xedges = yedges = np.asarray(bins, float)
        bins = [xedges, yedges]
    
    # 调用 binned_statistic_dd 函数进行二维统计
    medians, edges, binnumbers = binned_statistic_dd(
        [x, y], values, statistic, bins, range,
        expand_binnumbers=expand_binnumbers)
    
    # 返回 BinnedStatistic2dResult 对象，其中包含统计中值、xedges、yedges 和 binnumbers
    return BinnedStatistic2dResult(medians, edges[0], edges[1], binnumbers)
# 定义了一个名为 BinnedStatisticddResult 的命名元组，用于存储多维直方图统计结果
BinnedStatisticddResult = namedtuple('BinnedStatisticddResult',
                                     ('statistic', 'bin_edges',
                                      'binnumber'))


# 定义了一个函数 _bincount，用于根据权重计算统计结果
def _bincount(x, weights):
    # 如果权重是复数对象，则分别对实部和虚部进行统计
    if np.iscomplexobj(weights):
        a = np.bincount(x, np.real(weights))  # 计算实部的统计结果
        b = np.bincount(x, np.imag(weights))  # 计算虚部的统计结果
        z = a + b*1j  # 将实部和虚部的统计结果合并为复数
    else:
        z = np.bincount(x, weights)  # 对权重进行统计
    return z


# 定义了一个函数 binned_statistic_dd，用于计算多维直方图统计
def binned_statistic_dd(sample, values, statistic='mean',
                        bins=10, range=None, expand_binnumbers=False,
                        binned_statistic_result=None):
    """
    Compute a multidimensional binned statistic for a set of data.

    This is a generalization of a histogramdd function.  A histogram divides
    the space into bins, and returns the count of the number of points in
    each bin.  This function allows the computation of the sum, mean, median,
    or other statistic of the values within each bin.

    Parameters
    ----------
    sample : array_like
        Data to histogram passed as a sequence of N arrays of length D, or
        as an (N,D) array.
    values : (N,) array_like or list of (N,) array_like
        The data on which the statistic will be computed.  This must be
        the same shape as `sample`, or a list of sequences - each with the
        same shape as `sample`.  If `values` is such a list, the statistic
        will be computed on each independently.
    statistic : string or callable, optional
        The statistic to compute (default is 'mean').
        The following statistics are available:

          * 'mean' : compute the mean of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'median' : compute the median of values for points within each
            bin. Empty bins will be represented by NaN.
          * 'count' : compute the count of points within each bin.  This is
            identical to an unweighted histogram.  `values` array is not
            referenced.
          * 'sum' : compute the sum of values for points within each bin.
            This is identical to a weighted histogram.
          * 'std' : compute the standard deviation within each bin. This
            is implicitly calculated with ddof=0. If the number of values
            within a given bin is 0 or 1, the computed standard deviation value
            will be 0 for the bin.
          * 'min' : compute the minimum of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'max' : compute the maximum of values for point within each bin.
            Empty bins will be represented by NaN.
          * function : a user-defined function which takes a 1D array of
            values, and outputs a single numerical statistic. This function
            will be called on the values in each bin.  Empty bins will be
            represented by function([]), or NaN if this returns an error.
    bins : int or sequence of scalars or str, optional
        The bin specification:

          * the number of bins for the two dimensions (nx, ny) as integers.
          * the number of bins for each dimension (nx, ny, ...).
          * the bin edges for the two dimensions (x_edge, y_edge) as sequences of scalars.
          * the bin edges for each dimension (x_edge, y_edge, ...) as sequences of sequences.
          * the method to estimate the optimal number of bins ('auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt', 'integer', or callable) - see `numpy.histogram_bin_edges` for more information.

    range : array_like, shape(2, D), optional
        The minimum and maximum ranges for each dimension D = 2

    expand_binnumbers : bool, optional
        'True' to return (N,) array of bin indices.

    binned_statistic_result : namedtuple, optional
        Default 'None'.

    """
    # 函数实现多维直方图统计，可根据参数选择不同的统计方法（均值、中位数等）
    pass  # 占位符，函数体尚未实现
    # bins参数可以是一个序列或正整数，用于指定每个维度的箱体边界。
    # 可以采用以下形式之一：
    #   * 描述每个维度上箱体边界的数组序列。
    #   * 每个维度上的箱体数量（nx, ny, ... = bins）。
    #   * 所有维度上相同数量的箱体（nx = ny = ... = bins）。
    bins : sequence or positive int, optional

    # range参数是一个序列，用于指定如果bins没有明确给出边界时要使用的下限和上限边界。
    # 默认情况下，边界是沿每个维度的最小值和最大值。
    range : sequence, optional

    # expand_binnumbers是一个布尔值参数，默认为False。
    # 'False'：返回的binnumber是一个形状为(N,)的数组，其中存放了每个样本点所属的线性化的箱索引。
    # 'True'：返回的binnumber会被展开为一个形状为(D,N)的ndarray，其中每行给出了对应维度的箱号。
    # 参见返回的binnumber值，以及'binned_statistic_2d'的'Examples'部分。
    expand_binnumbers : bool, optional

    # binned_statistic_result参数是一个binnedStatisticddResult对象的实例，用于重用bin边界和bin编号以及新值和/或不同统计信息的返回结果。
    # 要重用bin编号，expand_binnumbers必须设置为False（默认值）。
    binned_statistic_result : binnedStatisticddResult

    # 返回值
    # -------
    # statistic：ndarray，shape(nx1, nx2, nx3,...)
    #   每个二维箱中所选统计量的值。
    # bin_edges：ndarray的列表
    #   包含每个维度的(nxi + 1)个bin边界的D个数组的列表。
    # binnumber：(N,) int数组或(D,N) int ndarray
    #   将每个样本元素分配给其所属的箱中的整数。展示依赖于expand_binnumbers参数。详见Notes部分。
    # 
    # See Also
    # --------
    # numpy.digitize, numpy.histogramdd, binned_statistic, binned_statistic_2d

    # Notes
    # -----
    # Binedges:
    # 每个维度上除了最后一个（最右边的）箱是半开放的。换句话说，如果bins是[1, 2, 3, 4]，那么第一个箱是[1, 2)（包含1，但不包含2），第二个箱是[2, 3)。然而，最后一个箱是[3, 4]，包含4。
    # 
    # binnumber：
    # 返回的参数将每个样本元素分配到其所属的箱中的整数。表示依赖于expand_binnumbers参数。
    # 如果为'False'（默认值）：返回的binnumber是一个形状为(N,)的数组，其中存放了每个样本点对应的箱的线性化索引（使用行主序排列）。
    # 如果为'True'：返回的binnumber是一个形状为(D,N)的ndarray，其中每行分别表示每个维度的箱的放置情况。在每个维度上，binnumber为'i'表示相应的值位于(bin_edges[D][i-1], bin_edges[D][i])之间。
    # 
    # 
    """
        .. versionadded:: 0.11.0
    
        Examples
        --------
        >>> import numpy as np
        >>> from scipy import stats
        >>> import matplotlib.pyplot as plt
        >>> from mpl_toolkits.mplot3d import Axes3D
    
        Take an array of 600 (x, y) coordinates as an example.
        `binned_statistic_dd` can handle arrays of higher dimension `D`. But a plot
        of dimension `D+1` is required.
    
        >>> mu = np.array([0., 1.])
        >>> sigma = np.array([[1., -0.5],[-0.5, 1.5]])
        >>> multinormal = stats.multivariate_normal(mu, sigma)
        >>> data = multinormal.rvs(size=600, random_state=235412)
        >>> data.shape
        (600, 2)
    
        Create bins and count how many arrays fall in each bin:
    
        >>> N = 60
        >>> x = np.linspace(-3, 3, N)
        >>> y = np.linspace(-3, 4, N)
        >>> ret = stats.binned_statistic_dd(data, np.arange(600), bins=[x, y],
        ...                                 statistic='count')
        >>> bincounts = ret.statistic
    
        Set the volume and the location of bars:
    
        >>> dx = x[1] - x[0]
        >>> dy = y[1] - y[0]
        >>> x, y = np.meshgrid(x[:-1]+dx/2, y[:-1]+dy/2)
        >>> z = 0
    
        >>> bincounts = bincounts.ravel()
        >>> x = x.ravel()
        >>> y = y.ravel()
    
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection='3d')
        >>> with np.errstate(divide='ignore'):   # silence random axes3d warning
        ...     ax.bar3d(x, y, z, dx, dy, bincounts)
    
        Reuse bin numbers and bin edges with new values:
    
        >>> ret2 = stats.binned_statistic_dd(data, -np.arange(600),
        ...                                  binned_statistic_result=ret,
        ...                                  statistic='mean')
        """
    
        # `known_stats` lists valid statistic types for `binned_statistic_dd`
        known_stats = ['mean', 'median', 'count', 'sum', 'std', 'min', 'max']
        # Check if `statistic` is a callable function or a known statistic type
        if not callable(statistic) and statistic not in known_stats:
            raise ValueError(f'invalid statistic {statistic!r}')
    
        try:
            # Attempt to treat `bins` as an index
            bins = index(bins)
        except TypeError:
            # If `bins` is not an integer, it remains unchanged
            # bins is not an integer
            pass
        # If bins was an integer-like object, now it is an actual Python int.
    
        # NOTE: for _bin_edges(), see e.g. gh-11365
        # Check if `bins` is an integer and if `sample` contains non-finite values
        if isinstance(bins, int) and not np.isfinite(sample).all():
            raise ValueError(f'{sample!r} contains non-finite values.')
    
        # `Ndim` is the number of dimensions (e.g., 2 for `binned_statistic_2d`)
        # `Dlen` is the length of elements along each dimension.
        # This code is based on np.histogramdd
        try:
            # Attempt to retrieve shape information from `sample` assuming it's an ND-array
            # `sample` is an ND-array.
            Dlen, Ndim = sample.shape
        except (AttributeError, ValueError):
            # If `sample` is not an ND-array, convert it to one-dimensional arrays
            sample = np.atleast_2d(sample).T
            Dlen, Ndim = sample.shape
    
        # Store initial shape of `values` to preserve it in the output
        values = np.asarray(values)
        input_shape = list(values.shape)
        # Ensure `values` is at least 2D to iterate over rows
        values = np.atleast_2d(values)
        Vdim, Vlen = values.shape
    
        # Ensure `values` match `sample`
    """
    # 检查统计方法不为 'count' 且值的长度与样本维度不匹配时，抛出异常
    if statistic != 'count' and Vlen != Dlen:
        raise AttributeError('The number of `values` elements must match the '
                             'length of each `sample` dimension.')

    try:
        # 获取 bins 的长度 M
        M = len(bins)
        # 检查 bins 的维度与样本 x 的维度是否一致，若不一致则抛出异常
        if M != Ndim:
            raise AttributeError('The dimension of bins must be equal '
                                 'to the dimension of the sample x.')
    except TypeError:
        # 若 bins 不是可迭代对象，则将其复制为长度为 Ndim 的列表
        bins = Ndim * [bins]

    if binned_statistic_result is None:
        # 若未提供 binned_statistic_result，则计算样本的边缘和 bin 编号
        nbin, edges, dedges = _bin_edges(sample, bins, range)
        binnumbers = _bin_numbers(sample, nbin, edges, dedges)
    else:
        # 若提供了 binned_statistic_result，则使用其存储的边缘
        edges = binned_statistic_result.bin_edges
        # 计算每个维度上的 bin 数量，+1 用于处理离群值的 bin
        nbin = np.array([len(edges[i]) + 1 for i in builtins.range(Ndim)])
        # 计算每个维度上的边缘间距
        dedges = [np.diff(edges[i]) for i in builtins.range(Ndim)]
        # 获取样本数据对应的 bin 编号
        binnumbers = binned_statistic_result.binnumber

    # 避免使用双精度浮点数时的溢出问题，将结果类型设置为 values 的结果类型或 np.float64
    result_type = np.result_type(values, np.float64)
    # 初始化结果数组，形状为 [Vdim, nbin.prod()]，使用 result_type 指定数据类型
    result = np.empty([Vdim, nbin.prod()], dtype=result_type)

    if statistic in {'mean', np.mean}:
        # 若统计方法为均值，则将结果数组填充为 NaN
        result.fill(np.nan)
        # 计算每个 bin 内的样本计数
        flatcount = _bincount(binnumbers, None)
        # 获取非零元素的索引
        a = flatcount.nonzero()
        for vv in builtins.range(Vdim):
            # 计算每个 bin 内的样本值之和
            flatsum = _bincount(binnumbers, values[vv])
            # 将均值填充至结果数组中
            result[vv, a] = flatsum[a] / flatcount[a]
    elif statistic in {'std', np.std}:
        # 若统计方法为标准差，则将结果数组填充为 NaN
        result.fill(np.nan)
        # 计算每个 bin 内的样本计数
        flatcount = _bincount(binnumbers, None)
        # 获取非零元素的索引
        a = flatcount.nonzero()
        for vv in builtins.range(Vdim):
            # 计算每个 bin 内的样本值之和
            flatsum = _bincount(binnumbers, values[vv])
            # 计算每个 bin 内样本值与平均值的差值
            delta = values[vv] - flatsum[binnumbers] / flatcount[binnumbers]
            # 计算标准差
            std = np.sqrt(
                _bincount(binnumbers, delta*np.conj(delta))[a] / flatcount[a]
            )
            # 将实部结果填充至结果数组中
            result[vv, a] = std
        result = np.real(result)
    elif statistic == 'count':
        # 若统计方法为计数，则初始化结果数组为 0
        result = np.empty([Vdim, nbin.prod()], dtype=np.float64)
        result.fill(0)
        # 计算每个 bin 内的样本计数
        flatcount = _bincount(binnumbers, None)
        # 创建索引数组 a
        a = np.arange(len(flatcount))
        # 将计数值填充至结果数组中
        result[:, a] = flatcount[np.newaxis, :]
    elif statistic in {'sum', np.sum}:
        # 若统计方法为求和，则将结果数组填充为 0
        result.fill(0)
        for vv in builtins.range(Vdim):
            # 计算每个 bin 内的样本值之和
            flatsum = _bincount(binnumbers, values[vv])
            # 创建索引数组 a
            a = np.arange(len(flatsum))
            # 将求和结果填充至结果数组中
            result[vv, a] = flatsum
    elif statistic in {'median', np.median}:
        # 若统计方法为中位数，则将结果数组填充为 NaN
        result.fill(np.nan)
        for vv in builtins.range(Vdim):
            # 根据 binnumbers 和 values[vv] 对样本数据进行排序
            i = np.lexsort((values[vv], binnumbers))
            # 获取唯一的 bin 编号，以及每个 bin 的起始索引和计数
            _, j, counts = np.unique(binnumbers[i],
                                     return_index=True, return_counts=True)
            # 计算每个 bin 的中位数索引
            mid = j + (counts - 1) / 2
            # 获取中位数的值
            mid_a = values[vv, i][np.floor(mid).astype(int)]
            mid_b = values[vv, i][np.ceil(mid).astype(int)]
            medians = (mid_a + mid_b) / 2
            # 将中位数填充至结果数组中
            result[vv, binnumbers[i][j]] = medians
    # 如果统计方法为 'min' 或 np.min 函数时执行以下代码块
    elif statistic in {'min', np.min}:
        # 用 NaN 填充结果数组
        result.fill(np.nan)
        # 遍历每个维度 Vdim
        for vv in builtins.range(Vdim):
            # 对每个值数组 values[vv] 进行排序，返回排序后的索引 i，逆序排列以便最小值在最后
            i = np.argsort(values[vv])[::-1]  # Reversed so the min is last
            # 将排序后的 values[vv] 数据填入 result 数组中对应的 binnumbers[i] 位置
            result[vv, binnumbers[i]] = values[vv, i]

    # 如果统计方法为 'max' 或 np.max 函数时执行以下代码块
    elif statistic in {'max', np.max}:
        # 用 NaN 填充结果数组
        result.fill(np.nan)
        # 遍历每个维度 Vdim
        for vv in builtins.range(Vdim):
            # 对每个值数组 values[vv] 进行排序，返回排序后的索引 i
            i = np.argsort(values[vv])
            # 将排序后的 values[vv] 数据填入 result 数组中对应的 binnumbers[i] 位置
            result[vv, binnumbers[i]] = values[vv, i]

    # 如果 statistic 是一个可调用对象时执行以下代码块
    elif callable(statistic):
        # 忽略无效操作错误，并忽略运行时警告
        with np.errstate(invalid='ignore'), catch_warnings():
            simplefilter("ignore", RuntimeWarning)
            try:
                # 尝试使用 statistic 函数计算空数组 []
                null = statistic([])
            except Exception:
                # 若计算出现异常，则将 null 设置为 NaN
                null = np.nan
        # 如果 null 是复数对象，则将结果数组 result 转换为复数类型
        if np.iscomplexobj(null):
            result = result.astype(np.complex128)
        # 用 null 填充结果数组
        result.fill(null)
        try:
            # 调用 _calc_binned_statistic 函数计算统计结果
            _calc_binned_statistic(
                Vdim, binnumbers, result, values, statistic
            )
        except ValueError:
            # 若计算出现值错误，则将结果数组 result 转换为复数类型后重新计算
            result = result.astype(np.complex128)
            _calc_binned_statistic(
                Vdim, binnumbers, result, values, statistic
            )

    # 将 result 重新整形为合适的矩阵形式
    result = result.reshape(np.append(Vdim, nbin))

    # 去除每个维度的离群值（去除每个维度的首尾元素）
    core = tuple([slice(None)] + Ndim * [slice(1, -1)])
    result = result[core]

    # 如果需要扩展 binnumbers 并且 Ndim 大于 1 时执行以下代码块
    if expand_binnumbers and Ndim > 1:
        # 将 binnumbers 展开为一个 ndarray，每行为每个维度的 bin
        binnumbers = np.asarray(np.unravel_index(binnumbers, nbin))

    # 如果结果数组的形状与 nbin - 2 不匹配，则抛出运行时错误
    if np.any(result.shape[1:] != nbin - 2):
        raise RuntimeError('Internal Shape Error')

    # 将结果数组 result 重新整形为与输入数据 values 相同的形状
    result = result.reshape(input_shape[:-1] + list(nbin-2))

    # 返回 BinnedStatisticddResult 对象，包含结果数组 result、边界 edges 和 binnumbers
    return BinnedStatisticddResult(result, edges, binnumbers)
def _calc_binned_statistic(Vdim, bin_numbers, result, values, stat_func):
    # 获取唯一的 bin 编号
    unique_bin_numbers = np.unique(bin_numbers)
    # 遍历每个维度的值
    for vv in builtins.range(Vdim):
        # 创建一个字典，将每个 bin 编号映射到对应的数据值列表
        bin_map = _create_binned_data(bin_numbers, unique_bin_numbers,
                                      values, vv)
        # 对于每个唯一的 bin 编号
        for i in unique_bin_numbers:
            # 计算统计函数在当前 bin 中数据的统计值
            stat = stat_func(np.array(bin_map[i]))
            # 如果统计值是复数对象而结果不是，则抛出异常
            if np.iscomplexobj(stat) and not np.iscomplexobj(result):
                raise ValueError("The statistic function returns complex ")
            # 将统计值存储在结果数组中的对应位置
            result[vv, i] = stat


def _create_binned_data(bin_numbers, unique_bin_numbers, values, vv):
    """ Create hashmap of bin ids to values in bins
    key: bin number
    value: list of binned data
    """
    # 创建一个空的字典，用于存储 bin 编号到数据值列表的映射关系
    bin_map = dict()
    # 对于每个唯一的 bin 编号，初始化一个空列表
    for i in unique_bin_numbers:
        bin_map[i] = []
    # 遍历每个数据点
    for i in builtins.range(len(bin_numbers)):
        # 将当前数据点的值添加到对应 bin 编号的列表中
        bin_map[bin_numbers[i]].append(values[vv, i])
    return bin_map


def _bin_edges(sample, bins=None, range=None):
    """ Create edge arrays
    """
    # 获取样本的维度信息
    Dlen, Ndim = sample.shape

    nbin = np.empty(Ndim, int)    # 每个维度中的 bin 数量
    edges = Ndim * [None]         # 每个维度的边界数组（将是一个二维数组）
    dedges = Ndim * [None]        # 边界之间的间距数组（也将是一个二维数组）

    # 为每个维度选择范围
    # 仅在给定 bin 数量时使用
    if range is None:
        smin = np.atleast_1d(np.array(sample.min(axis=0), float))
        smax = np.atleast_1d(np.array(sample.max(axis=0), float))
    else:
        if len(range) != Ndim:
            raise ValueError(
                f"range given for {len(range)} dimensions; {Ndim} required")
        smin = np.empty(Ndim)
        smax = np.empty(Ndim)
        for i in builtins.range(Ndim):
            if range[i][1] < range[i][0]:
                raise ValueError(
                    "In {}range, start must be <= stop".format(
                        f"dimension {i + 1} of " if Ndim > 1 else ""))
            smin[i], smax[i] = range[i]

    # 确保每个 bin 的宽度是有限的
    for i in builtins.range(len(smin)):
        if smin[i] == smax[i]:
            smin[i] = smin[i] - .5
            smax[i] = smax[i] + .5

    # 在 bin 边界中保持样本浮点精度
    edges_dtype = (sample.dtype if np.issubdtype(sample.dtype, np.floating)
                   else float)

    # 创建边界数组
    for i in builtins.range(Ndim):
        if np.isscalar(bins[i]):
            nbin[i] = bins[i] + 2  # +2 用于异常值的 bin
            edges[i] = np.linspace(smin[i], smax[i], nbin[i] - 1,
                                   dtype=edges_dtype)
        else:
            edges[i] = np.asarray(bins[i], edges_dtype)
            nbin[i] = len(edges[i]) + 1  # +1 用于异常值的 bin
        dedges[i] = np.diff(edges[i])

    nbin = np.asarray(nbin)

    return nbin, edges, dedges


def _bin_numbers(sample, nbin, edges, dedges):
    """Compute the bin number each sample falls into, in each dimension
    """
    # 获取样本数据的维度信息
    Dlen, Ndim = sample.shape

    # 对每个维度的样本值使用 `digitize` 函数，将其分配到相应的箱中
    sampBin = [
        np.digitize(sample[:, i], edges[i])
        for i in range(Ndim)
    ]

    # 使用 `digitize` 函数后，位于边界上的值会被放入右侧的箱中。
    # 对于最右侧的箱，我们希望与右边界相等的值被归入最后一个箱中，而不算作离群值。
    for i in range(Ndim):
        # 查找边界值的最小差异
        dedges_min = dedges[i].min()
        # 如果最小差异为0，则抛出数值错误异常
        if dedges_min == 0:
            raise ValueError('The smallest edge difference is numerically 0.')
        # 计算四舍五入的精度
        decimal = int(-np.log10(dedges_min)) + 6
        # 找出位于最右边界上的点
        on_edge = np.where((sample[:, i] >= edges[i][-1]) &
                           (np.around(sample[:, i], decimal) ==
                            np.around(edges[i][-1], decimal)))[0]
        # 将这些点向左移动一个箱
        sampBin[i][on_edge] -= 1

    # 计算样本在压扁后统计矩阵中的索引
    binnumbers = np.ravel_multi_index(sampBin, nbin)

    # 返回计算得到的箱编号数组
    return binnumbers
```