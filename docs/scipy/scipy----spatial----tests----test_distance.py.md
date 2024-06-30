# `D:\src\scipysrc\scipy\scipy\spatial\tests\test_distance.py`

```
# 导入标准库中的 sys 和 os.path 模块
import sys
import os.path

# 从 functools 模块导入 wraps 和 partial 函数
from functools import wraps, partial
# 导入 weakref 模块中的全部内容
import weakref

# 导入 numpy 库，并重命名为 np
import numpy as np
# 导入 warnings 模块中的全部内容
import warnings
# 从 numpy.linalg 模块中导入 norm 函数
from numpy.linalg import norm
# 从 numpy.testing 模块中选择性地导入 verbose, assert_, assert_array_equal,
# assert_equal, assert_almost_equal, assert_allclose, break_cycles, IS_PYPY 函数和常量
from numpy.testing import (verbose, assert_,
                           assert_array_equal, assert_equal,
                           assert_almost_equal, assert_allclose,
                           break_cycles, IS_PYPY)

# 导入 pytest 模块
import pytest

# 从 scipy.spatial.distance 模块中导入 squareform, pdist, cdist, num_obs_y, num_obs_dm,
# is_valid_dm, is_valid_y, _validate_vector, _METRICS_NAMES 函数和常量
import scipy.spatial.distance

from scipy.spatial.distance import (
    squareform, pdist, cdist, num_obs_y, num_obs_dm, is_valid_dm, is_valid_y,
    _validate_vector, _METRICS_NAMES)

# 从 scipy.spatial.distance 模块中选择性地导入 braycurtis, canberra, chebyshev, cityblock,
# correlation, cosine, dice, euclidean, hamming, jaccard, jensenshannon, kulczynski1,
# mahalanobis, minkowski, rogerstanimoto, russellrao, seuclidean, sokalmichener,
# sokalsneath, sqeuclidean, yule 函数和常量
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
                                    correlation, cosine, dice, euclidean,
                                    hamming, jaccard, jensenshannon,
                                    kulczynski1, mahalanobis,
                                    minkowski, rogerstanimoto,
                                    russellrao, seuclidean, sokalmichener,  # noqa: F401
                                    sokalsneath, sqeuclidean, yule)

# 从 scipy._lib._util 模块中导入 np_long 和 np_ulong 常量
from scipy._lib._util import np_long, np_ulong


@pytest.fixture(params=_METRICS_NAMES, scope="session")
def metric(request):
    """
    Fixture for all metrics in scipy.spatial.distance
    """
    # 返回 request 参数中的度量名称作为 fixture
    return request.param
# 列出需要读取的文件名列表
_filenames = [
              "cdist-X1.txt",
              "cdist-X2.txt",
              "iris.txt",
              "pdist-boolean-inp.txt",
              "pdist-chebyshev-ml-iris.txt",
              "pdist-chebyshev-ml.txt",
              "pdist-cityblock-ml-iris.txt",
              "pdist-cityblock-ml.txt",
              "pdist-correlation-ml-iris.txt",
              "pdist-correlation-ml.txt",
              "pdist-cosine-ml-iris.txt",
              "pdist-cosine-ml.txt",
              "pdist-double-inp.txt",
              "pdist-euclidean-ml-iris.txt",
              "pdist-euclidean-ml.txt",
              "pdist-hamming-ml.txt",
              "pdist-jaccard-ml.txt",
              "pdist-jensenshannon-ml-iris.txt",
              "pdist-jensenshannon-ml.txt",
              "pdist-minkowski-3.2-ml-iris.txt",
              "pdist-minkowski-3.2-ml.txt",
              "pdist-minkowski-5.8-ml-iris.txt",
              "pdist-seuclidean-ml-iris.txt",
              "pdist-seuclidean-ml.txt",
              "pdist-spearman-ml.txt",
              "random-bool-data.txt",
              "random-double-data.txt",
              "random-int-data.txt",
              "random-uint-data.txt",
              ]

# 定义一个距离矩阵的二维数组，存储距离数据
_tdist = np.array([[0, 662, 877, 255, 412, 996],
                      [662, 0, 295, 468, 268, 400],
                      [877, 295, 0, 754, 564, 138],
                      [255, 468, 754, 0, 219, 869],
                      [412, 268, 564, 219, 0, 669],
                      [996, 400, 138, 869, 669, 0]], dtype='double')

# 将_tdist转换成紧凑形式的距离矩阵
_ytdist = squareform(_tdist)

# 期望输出数组的哈希映射，用于测试。这些数组来自一组文本文件，在测试之前被读取。
# 每个测试从这个字典中加载输入和输出。
eo = {}


def load_testing_files():
    # 遍历文件名列表
    for fn in _filenames:
        # 根据文件名生成相应的键名
        name = fn.replace(".txt", "").replace("-ml", "")
        # 构建完整的文件路径和文件名
        fqfn = os.path.join(os.path.dirname(__file__), 'data', fn)
        # 打开文件
        fp = open(fqfn)
        # 从文件中加载数据到numpy数组，并存储在eo字典中
        eo[name] = np.loadtxt(fp)
        # 关闭文件
        fp.close()
    
    # 针对特定的数据，将它们转换为布尔型、浮点型等格式
    eo['pdist-boolean-inp'] = np.bool_(eo['pdist-boolean-inp'])
    eo['random-bool-data'] = np.bool_(eo['random-bool-data'])
    eo['random-float32-data'] = np.float32(eo['random-double-data'])
    eo['random-int-data'] = np_long(eo['random-int-data'])
    eo['random-uint-data'] = np_ulong(eo['random-uint-data'])


# 加载测试数据文件到eo字典中
load_testing_files()


def _is_32bit():
    # 判断系统是否为32位
    return np.intp(0).itemsize < 8


def _chk_asarrays(arrays, axis=None):
    # 将输入的数组转换为numpy数组
    arrays = [np.asanyarray(a) for a in arrays]
    if axis is None:
        # 如果axis为None，对数组进行处理使其至少为1维
        arrays = [np.ravel(a) if a.ndim != 1 else a
                  for a in arrays]
        axis = 0
    # 确保所有输入数组至少为1维，并指定轴参数
    arrays = tuple(np.atleast_1d(a) for a in arrays)
    if axis < 0:
        # 如果axis为负数，确保所有数组的维度相同
        if not all(a.ndim == arrays[0].ndim for a in arrays):
            raise ValueError("array ndim must be the same for neg axis")
        axis = range(arrays[0].ndim)[axis]
    return arrays + (axis,)
# 对传入的数组进行检查，将它们转换为数组列表和轴位置
def _chk_weights(arrays, weights=None, axis=None,
                 force_weights=False, simplify_weights=True,
                 pos_only=False, neg_check=False,
                 nan_screen=False, mask_screen=False,
                 ddof=None):
    # 调用辅助函数 _chk_asarrays 对数组进行检查，并获取轴位置
    chked = _chk_asarrays(arrays, axis=axis)
    arrays, axis = chked[:-1], chked[-1]

    # 简化权重数组的标志设置
    simplify_weights = simplify_weights and not force_weights
    # 如果不强制使用权重且存在掩码屏幕，检查是否有任何数组包含掩码
    if not force_weights and mask_screen:
        force_weights = any(np.ma.getmask(a) is not np.ma.nomask for a in arrays)

    # 如果启用了 NaN 屏幕功能
    if nan_screen:
        # 检查每个数组是否包含 NaN 值
        has_nans = [np.isnan(np.sum(a)) for a in arrays]
        if any(has_nans):
            # 如果有数组包含 NaN 值，则启用掩码屏幕和强制权重标志
            mask_screen = True
            force_weights = True
            # 使用 np.ma.masked_invalid 函数将包含 NaN 的数组元素掩码
            arrays = tuple(np.ma.masked_invalid(a) if has_nan else a
                           for a, has_nan in zip(arrays, has_nans))

    # 如果传入了权重数组
    if weights is not None:
        weights = np.asanyarray(weights)
    # 如果未传入权重且需要强制权重
    elif force_weights:
        # 创建一个形状与数组轴长度相同的全为 1 的权重数组
        weights = np.ones(arrays[0].shape[axis])
    else:
        # 如果既未传入权重也不需要强制权重，则直接返回数组和轴
        return arrays + (weights, axis)

    # 如果指定了自由度参数 ddof，则调用 _freq_weights 函数处理权重
    if ddof:
        weights = _freq_weights(weights)

    # 如果启用了掩码屏幕，则调用 _weight_masked 函数处理权重
    if mask_screen:
        weights = _weight_masked(arrays, weights, axis)

    # 检查权重数组的形状是否与每个数组在指定轴上的形状一致
    if not all(weights.shape == (a.shape[axis],) for a in arrays):
        raise ValueError("weights shape must match arrays along axis")
    # 如果启用了负值检查，并且权重数组中有负值，则抛出异常
    if neg_check and (weights < 0).any():
        raise ValueError("weights cannot be negative")

    # 如果仅考虑正值权重，并且存在非正值权重，则从数组中剔除非正值权重
    if pos_only:
        pos_weights = np.nonzero(weights > 0)[0]
        if pos_weights.size < weights.size:
            # 从数组中选择正值权重对应的元素，并更新权重数组
            arrays = tuple(np.take(a, pos_weights, axis=axis) for a in arrays)
            weights = weights[pos_weights]
    # 如果简化权重标志被激活，并且所有权重都为 1，则将权重设置为 None
    if simplify_weights and (weights == 1).all():
        weights = None
    return arrays + (weights, axis)


# 将权重数组转换为整数类型的频率权重数组
def _freq_weights(weights):
    if weights is None:
        return weights
    # 将权重数组转换为整数类型
    int_weights = weights.astype(int)
    if (weights != int_weights).any():
        raise ValueError("frequency (integer count-type) weights required %s" % weights)
    return int_weights


# 处理包含掩码的权重数组，根据掩码将权重设置为 0
def _weight_masked(arrays, weights, axis):
    # 如果轴位置未指定，默认设置为 0
    if axis is None:
        axis = 0
    # 将权重数组转换为 numpy 数组类型
    weights = np.asanyarray(weights)
    # 遍历数组，检查每个数组是否包含掩码，并将相应位置的权重设置为 0
    for a in arrays:
        axis_mask = np.ma.getmask(a)
        if axis_mask is np.ma.nomask:
            continue
        if a.ndim > 1:
            not_axes = tuple(i for i in range(a.ndim) if i != axis)
            axis_mask = axis_mask.any(axis=not_axes)
        weights *= 1 - axis_mask.astype(int)
    return weights


# 随机分割数组，将其转换为 float64 类型以避免 NaN 问题
def _rand_split(arrays, weights, axis, split_per, seed=None):
    # 如果数组元素为整数类型，则将其转换为 float64 类型以避免 NaN 问题
    arrays = [arr.astype(np.float64) if np.issubdtype(arr.dtype, np.integer)
              else arr for arr in arrays]

    # 将权重数组转换为 float64 类型
    weights = np.array(weights, dtype=np.float64)  # modified inplace; need a copy
    # 使用指定的种子创建随机数生成器
    seeded_rand = np.random.RandomState(seed)
    # 定义一个函数，用于从数组 `a` 中取出索引 `ix` 处的元素，根据指定的 `axis` 轴
    def mytake(a, ix, axis):
        # 将输入数组 `a` 转换为任意数组类型，并从中取出索引 `ix` 处的元素
        record = np.asanyarray(np.take(a, ix, axis=axis))
        # 将取出的元素重新整形成新的数组，保持原数组的维度，只有 `axis` 轴的长度为 1
        return record.reshape([a.shape[i] if i != axis else 1
                               for i in range(a.ndim)])

    # 获取数组列表中第一个数组的特定轴向的长度作为观测值数量 `n_obs`
    n_obs = arrays[0].shape[axis]
    # 确保所有数组在指定轴向上的长度都与 `n_obs` 相等，否则抛出异常信息
    assert all(a.shape[axis] == n_obs for a in arrays), \
           "data must be aligned on sample axis"

    # 根据拆分比例 `split_per` 以及观测值数量 `n_obs` 循环执行拆分操作
    for i in range(int(split_per) * n_obs):
        # 使用种子随机数生成器 `seeded_rand` 生成一个小于 `n_obs + i` 的随机整数作为拆分索引
        split_ix = seeded_rand.randint(n_obs + i)
        # 获取拆分索引位置处的权重值，并计算新的权重值 `prev_w` 乘以随机数 `q`
        prev_w = weights[split_ix]
        q = seeded_rand.rand()
        weights[split_ix] = q * prev_w
        # 将 `(1. - q) * prev_w` 添加到权重数组 `weights` 中
        weights = np.append(weights, (1. - q) * prev_w)
        # 对每个数组 `a` 执行在指定轴向 `axis` 处的拆分操作，并将结果添加到数组列表 `arrays` 中
        arrays = [np.append(a, mytake(a, split_ix, axis=axis),
                            axis=axis) for a in arrays]
    
    # 返回拆分后的数组列表 `arrays` 和更新后的权重数组 `weights`
    return arrays, weights
# 定义一个函数用于粗略检查两个输入对象是否相等，支持自定义比较函数和关键字提取函数
def _rough_check(a, b, compare_assert=partial(assert_allclose, atol=1e-5),
                  key=lambda x: x, w=None):
    # 获取经过关键字提取函数处理后的对象 a 和 b
    check_a = key(a)
    check_b = key(b)
    try:
        # 尝试使用严格的相等性检查字符串类型的对象
        if np.array(check_a != check_b).any():  # try strict equality for string types
            # 使用指定的比较断言函数检查 a 和 b 是否相等
            compare_assert(check_a, check_b)
    except AttributeError:  # 处理掩码数组类型的异常
        # 对掩码数组类型，使用指定的比较断言函数检查 a 和 b 是否相等
        compare_assert(check_a, check_b)
    except (TypeError, ValueError):  # 处理嵌套数据结构类型的异常
        # 对于嵌套数据结构类型，递归调用 _rough_check 函数进行检查
        for a_i, b_i in zip(check_a, check_b):
            _rough_check(a_i, b_i, compare_assert=compare_assert)

# 从 test_stats 模块中导入的函数的不同之处：
#  n_args=2, weight_arg='w', default_axis=None
#  ma_safe = False, nan_safe = False
# 定义一个装饰器函数，用于对指定函数进行多种方式的参数组合调用和结果检查
def _weight_checked(fn, n_args=2, default_axis=None, key=lambda x: x, weight_arg='w',
                    squeeze=True, silent=False,
                    ones_test=True, const_test=True, dup_test=True,
                    split_test=True, dud_test=True, ma_safe=False, ma_very_safe=False,
                    nan_safe=False, split_per=1.0, seed=0,
                    compare_assert=partial(assert_allclose, atol=1e-5)):
    """runs fn on its arguments 2 or 3 ways, checks that the results are the same,
       then returns the same thing it would have returned before"""
    # 使用 functools 模块的 wraps 函数，确保装饰器装饰后函数的元信息不丢失
    @wraps(fn)
    def wrapped(*args, **kwargs):
        # 在此执行 fn 函数的调用，通过比较断言函数确保不同方式的调用结果一致
        return fn(*args, **kwargs)

    return wrapped


# 使用 _weight_checked 装饰器创建不同的带参数的函数
wcdist = _weight_checked(cdist, default_axis=1, squeeze=False)
wcdist_no_const = _weight_checked(cdist, default_axis=1,
                                  squeeze=False, const_test=False)
wpdist = _weight_checked(pdist, default_axis=1, squeeze=False, n_args=1)
wpdist_no_const = _weight_checked(pdist, default_axis=1, squeeze=False,
                                  const_test=False, n_args=1)
wrogerstanimoto = _weight_checked(rogerstanimoto)
wmatching = whamming = _weight_checked(hamming, dud_test=False)
wyule = _weight_checked(yule)
wdice = _weight_checked(dice)
wcityblock = _weight_checked(cityblock)
wchebyshev = _weight_checked(chebyshev)
wcosine = _weight_checked(cosine)
wcorrelation = _weight_checked(correlation)
wkulczynski1 = _weight_checked(kulczynski1)
wjaccard = _weight_checked(jaccard)
weuclidean = _weight_checked(euclidean, const_test=False)
wsqeuclidean = _weight_checked(sqeuclidean, const_test=False)
wbraycurtis = _weight_checked(braycurtis)
wcanberra = _weight_checked(canberra, const_test=False)
wsokalsneath = _weight_checked(sokalsneath)
wsokalmichener = _weight_checked(sokalmichener)
wrussellrao = _weight_checked(russellrao)


# 定义一个测试类 TestCdist
class TestCdist:

    # 在每个测试方法运行之前设置测试环境
    def setup_method(self):
        # 随机生成的数据集名称列表
        self.rnd_eo_names = ['random-float32-data', 'random-int-data',
                             'random-uint-data', 'random-double-data',
                             'random-bool-data']
        # 不同数据类型之间的有效类型转换字典
        self.valid_upcasts = {'bool': [np_ulong, np_long, np.float32, np.float64],
                              'uint': [np_long, np.float32, np.float64],
                              'int': [np.float32, np.float64],
                              'float32': [np.float64]}
    # 测试带有额外参数的 cdist 函数的行为，使用不同的度量标准 metric 进行测试
    def test_cdist_extra_args(self, metric):
        # Tests that args and kwargs are correctly handled

        # 创建两个示例的数据集 X1 和 X2
        X1 = [[1., 2., 3.], [1.2, 2.3, 3.4], [2.2, 2.3, 4.4]]
        X2 = [[7., 5., 8.], [7.5, 5.8, 8.4], [5.5, 5.8, 4.4]]
        # 创建一个包含无效参数的 kwargs 字典
        kwargs = {"N0tV4l1D_p4raM": 3.14, "w": np.arange(3)}
        # 创建一个包含多个相同参数的 args 列表
        args = [3.14] * 200

        # 使用 pytest 的 assertRaises 检查不同情况下 cdist 函数的参数是否能正确触发 TypeError 异常
        with pytest.raises(TypeError):
            cdist(X1, X2, metric=metric, **kwargs)
        with pytest.raises(TypeError):
            cdist(X1, X2, metric=eval(metric), **kwargs)
        with pytest.raises(TypeError):
            cdist(X1, X2, metric="test_" + metric, **kwargs)
        with pytest.raises(TypeError):
            cdist(X1, X2, metric=metric, *args)
        with pytest.raises(TypeError):
            cdist(X1, X2, metric=eval(metric), *args)
        with pytest.raises(TypeError):
            cdist(X1, X2, metric="test_" + metric, *args)

    # 测试带有额外参数的 cdist 函数在自定义度量标准下的行为
    def test_cdist_extra_args_custom(self):
        # Tests that args and kwargs are correctly handled
        # also for custom metric

        # 定义一个简单的自定义度量函数 _my_metric
        def _my_metric(x, y, arg, kwarg=1, kwarg2=2):
            return arg + kwarg + kwarg2

        # 创建两个示例的数据集 X1 和 X2
        X1 = [[1., 2., 3.], [1.2, 2.3, 3.4], [2.2, 2.3, 4.4]]
        X2 = [[7., 5., 8.], [7.5, 5.8, 8.4], [5.5, 5.8, 4.4]]
        # 创建一个包含无效参数的 kwargs 字典
        kwargs = {"N0tV4l1D_p4raM": 3.14, "w": np.arange(3)}
        # 创建一个包含多个相同参数的 args 列表
        args = [3.14] * 200

        # 使用 pytest 的 assertRaises 检查不同情况下 cdist 函数使用自定义度量函数 _my_metric 是否能正确触发 TypeError 异常
        with pytest.raises(TypeError):
            cdist(X1, X2, _my_metric)
        with pytest.raises(TypeError):
            cdist(X1, X2, _my_metric, *args)
        with pytest.raises(TypeError):
            cdist(X1, X2, _my_metric, **kwargs)
        with pytest.raises(TypeError):
            cdist(X1, X2, _my_metric, kwarg=2.2, kwarg2=3.3)
        with pytest.raises(TypeError):
            cdist(X1, X2, _my_metric, 1, 2, kwarg=2.2)
        with pytest.raises(TypeError):
            cdist(X1, X2, _my_metric, 1.1, 2.2, 3.3)
        with pytest.raises(TypeError):
            cdist(X1, X2, _my_metric, 1.1, 2.2)
        with pytest.raises(TypeError):
            cdist(X1, X2, _my_metric, 1.1)
        with pytest.raises(TypeError):
            cdist(X1, X2, _my_metric, 1.1, kwarg=2.2, kwarg2=3.3)

        # 检查 cdist 函数在正确的参数下是否能够返回预期的结果，使用 assert_allclose 进行比较
        assert_allclose(cdist(X1, X2, metric=_my_metric, arg=1.1, kwarg2=3.3), 5.4)

    # 测试使用欧几里德距离的随机 Unicode 数据的行为
    def test_cdist_euclidean_random_unicode(self):
        eps = 1e-15
        # 获取预定义的 Unicode 数据集 X1 和 X2
        X1 = eo['cdist-X1']
        X2 = eo['cdist-X2']
        # 使用两种方式计算欧几里德距离，分别存储在 Y1 和 Y2 中
        Y1 = wcdist_no_const(X1, X2, 'euclidean')
        Y2 = wcdist_no_const(X1, X2, 'test_euclidean')
        # 使用 assert_allclose 检查两种方式计算的结果是否在指定的相对容差 eps 内相等
        assert_allclose(Y1, Y2, rtol=eps, verbose=verbose > 2)

    # 使用 pytest.mark.parametrize 定义参数化测试，参数为 p
    def test_cdist_minkowski_random(self, p):
        # 设定一个极小的误差值
        eps = 1e-13
        # 从输入参数字典中获取数据集 X1 和 X2
        X1 = eo['cdist-X1']
        X2 = eo['cdist-X2']
        # 使用自定义函数计算 Minkowski 距离，不包括常数项
        Y1 = wcdist_no_const(X1, X2, 'minkowski', p=p)
        # 使用另一种自定义函数计算 Minkowski 距离，用于测试
        Y2 = wcdist_no_const(X1, X2, 'test_minkowski', p=p)
        # 断言两种方法计算的距离非常接近
        assert_allclose(Y1, Y2, atol=0, rtol=eps, verbose=verbose > 2)

    def test_cdist_cosine_random(self):
        # 设定一个极小的误差值
        eps = 1e-14
        # 从输入参数字典中获取数据集 X1 和 X2
        X1 = eo['cdist-X1']
        X2 = eo['cdist-X2']
        # 使用库函数计算余弦距离
        Y1 = wcdist(X1, X2, 'cosine')

        # Naive implementation
        # 定义一个函数用于计算向量的范数
        def norms(X):
            return np.linalg.norm(X, axis=1).reshape(-1, 1)

        # 使用向量的范数计算余弦距离的替代方法
        Y2 = 1 - np.dot((X1 / norms(X1)), (X2 / norms(X2)).T)

        # 断言两种方法计算的距离非常接近
        assert_allclose(Y1, Y2, rtol=eps, verbose=verbose > 2)

    def test_cdist_mahalanobis(self):
        # 对于一维观测数据
        x1 = np.array([[2], [3]])
        x2 = np.array([[2], [5]])
        # 使用马氏距离计算距离矩阵
        dist = cdist(x1, x2, metric='mahalanobis')
        # 断言计算出的距离矩阵与预期非常接近
        assert_allclose(dist, [[0.0, np.sqrt(4.5)], [np.sqrt(0.5), np.sqrt(2)]])

        # 对于二维观测数据
        x1 = np.array([[0, 0], [-1, 0]])
        x2 = np.array([[0, 2], [1, 0], [0, -2]])
        # 使用马氏距离计算距离矩阵
        dist = cdist(x1, x2, metric='mahalanobis')
        rt2 = np.sqrt(2)
        # 断言计算出的距离矩阵与预期非常接近
        assert_allclose(dist, [[rt2, rt2, rt2], [2, 2 * rt2, 2]])

        # 对于过少观测数据的情况
        with pytest.raises(ValueError):
            # 使用马氏距离计算距离矩阵，会引发异常
            cdist([[0, 1]], [[2, 3]], metric='mahalanobis')

    def test_cdist_custom_notdouble(self):
        # 定义一个空的类
        class myclass:
            pass

        # 自定义的距离度量函数
        def _my_metric(x, y):
            # 如果输入数据类型不是 myclass 类型，则引发异常
            if not isinstance(x[0], myclass) or not isinstance(y[0], myclass):
                raise ValueError("Type has been changed")
            # 返回固定的距离值
            return 1.123

        # 创建一个包含 myclass 对象的数据数组
        data = np.array([[myclass()]], dtype=object)
        # 使用自定义的距离度量函数计算距离矩阵
        cdist_y = cdist(data, data, metric=_my_metric)
        right_y = 1.123
        # 断言计算出的距离与预期值非常接近
        assert_equal(cdist_y, right_y, verbose=verbose > 2)

    def _check_calling_conventions(self, X1, X2, metric, eps=1e-07, **kwargs):
        # 辅助函数，用于测试 cdist 的调用约定
        try:
            # 使用不同的度量方式计算距离矩阵
            y1 = cdist(X1, X2, metric=metric, **kwargs)
            y2 = cdist(X1, X2, metric=eval(metric), **kwargs)
            y3 = cdist(X1, X2, metric="test_" + metric, **kwargs)
        except Exception as e:
            # 如果发生异常，获取异常类并记录异常信息
            e_cls = e.__class__
            if verbose > 2:
                print(e_cls.__name__)
                print(e)
            # 断言在特定异常下，cdist 函数也会引发相同的异常
            with pytest.raises(e_cls):
                cdist(X1, X2, metric=metric, **kwargs)
            with pytest.raises(e_cls):
                cdist(X1, X2, metric=eval(metric), **kwargs)
            with pytest.raises(e_cls):
                cdist(X1, X2, metric="test_" + metric, **kwargs)
        else:
            # 断言使用不同的度量方式计算得到的距离矩阵非常接近
            assert_allclose(y1, y2, rtol=eps, verbose=verbose > 2)
            assert_allclose(y1, y3, rtol=eps, verbose=verbose > 2)
    # 确保指定度量方法的调用方式与使用字符串或scipy函数相同
    # 行为应保持一致（即产生相同的结果或相同的异常）
    # 注意：正确性应在每个度量测试中进行检查。
    def test_cdist_calling_conventions(self, metric):
        # 对每个随机生成的数据集名称进行迭代
        for eo_name in self.rnd_eo_names:
            # 对输入数据进行子采样以加快测试速度
            # 注意：对于mahalanobis度量，样本数需要大于维度数
            X1 = eo[eo_name][::5, ::-2]
            X2 = eo[eo_name][1::5, ::2]
            if verbose > 2:
                print("testing: ", metric, " with: ", eo_name)
            # 如果度量方法属于特定集合，并且数据集名称中不包含'bool'
            if metric in {'dice', 'yule',
                          'rogerstanimoto',
                          'russellrao', 'sokalmichener',
                          'sokalsneath',
                          'kulczynski1'} and 'bool' not in eo_name:
                # Python版本允许非布尔类型数据，例如模糊逻辑
                continue
            # 调用检查方法，验证调用约定
            self._check_calling_conventions(X1, X2, metric)

            # 使用额外参数测试内置度量方法
            if metric == "seuclidean":
                X12 = np.vstack([X1, X2]).astype(np.float64)
                V = np.var(X12, axis=0, ddof=1)
                self._check_calling_conventions(X1, X2, metric, V=V)
            elif metric == "mahalanobis":
                X12 = np.vstack([X1, X2]).astype(np.float64)
                V = np.atleast_2d(np.cov(X12.T))
                VI = np.array(np.linalg.inv(V).T)
                self._check_calling_conventions(X1, X2, metric, VI=VI)

    # 测试结果不受类型提升影响的影响
    def test_cdist_dtype_equivalence(self, metric):
        eps = 1e-07
        # 对每个测试用例进行迭代
        tests = [(eo['random-bool-data'], self.valid_upcasts['bool']),
                 (eo['random-uint-data'], self.valid_upcasts['uint']),
                 (eo['random-int-data'], self.valid_upcasts['int']),
                 (eo['random-float32-data'], self.valid_upcasts['float32'])]
        for test in tests:
            X1 = test[0][::5, ::-2]
            X2 = test[0][1::5, ::2]
            try:
                # 尝试调用cdist函数，并捕获异常
                y1 = cdist(X1, X2, metric=metric)
            except Exception as e:
                e_cls = e.__class__
                if verbose > 2:
                    print(e_cls.__name__)
                    print(e)
                # 对每种新的数据类型进行迭代，期望引发相同的异常
                for new_type in test[1]:
                    X1new = new_type(X1)
                    X2new = new_type(X2)
                    with pytest.raises(e_cls):
                        cdist(X1new, X2new, metric=metric)
            else:
                # 如果没有异常被抛出，验证结果不受数据类型提升的影响
                for new_type in test[1]:
                    y2 = cdist(new_type(X1), new_type(X2), metric=metric)
                    assert_allclose(y1, y2, rtol=eps, verbose=verbose > 2)
    # 测试 cdist 函数的 out 参数是否正常工作
    def test_cdist_out(self, metric):
        # 定义一个极小的数值误差阈值
        eps = 1e-15
        # 从测试数据中获取 X1 和 X2，这是要计算距离的两个数据集
        X1 = eo['cdist-X1']
        X2 = eo['cdist-X2']
        # 获取 X1 和 X2 的行数和列数，用于创建输出数组
        out_r, out_c = X1.shape[0], X2.shape[0]

        kwargs = dict()
        # 如果距离度量是 'minkowski'，设置额外的参数 p 为 1.23
        if metric == 'minkowski':
            kwargs['p'] = 1.23
        # 创建一个空的输出数组 out1，用于存储 cdist 函数的输出
        out1 = np.empty((out_r, out_c), dtype=np.float64)
        # 调用 cdist 函数计算两个数据集之间的距离，存储在 Y1 中
        Y1 = cdist(X1, X2, metric, **kwargs)
        # 使用预先分配的 out1 数组存储 cdist 函数的输出，存储在 Y2 中
        Y2 = cdist(X1, X2, metric, out=out1, **kwargs)

        # 断言 Y1 和 Y2 的值在数值上非常接近，rtol 为误差容忍度
        assert_allclose(Y1, Y2, rtol=eps, verbose=verbose > 2)

        # 断言 Y2 和 out1 是同一个对象，即 cdist 是否正确地使用了 out 参数
        assert_(Y2 is out1)

        # 创建一个形状不匹配的空数组 out2，用于测试错误的形状输入时是否抛出异常
        out2 = np.empty((out_r-1, out_c+1), dtype=np.float64)
        with pytest.raises(ValueError):
            # 测试 cdist 函数在形状不匹配的情况下是否抛出 ValueError 异常
            cdist(X1, X2, metric, out=out2, **kwargs)

        # 创建一个非 C-contiguous（非连续存储）的空数组 out3，用于测试错误的存储顺序输入时是否抛出异常
        out3 = np.empty(
            (2 * out_r, 2 * out_c), dtype=np.float64)[::2, ::2]
        # 创建一个指定存储顺序为 'F'（Fortran）的空数组 out4，同样用于测试错误的存储顺序输入时是否抛出异常
        out4 = np.empty((out_r, out_c), dtype=np.float64, order='F')
        with pytest.raises(ValueError):
            # 测试 cdist 函数在错误的存储顺序输入时是否抛出 ValueError 异常
            cdist(X1, X2, metric, out=out3, **kwargs)
        with pytest.raises(ValueError):
            # 测试 cdist 函数在错误的存储顺序输入时是否抛出 ValueError 异常
            cdist(X1, X2, metric, out=out4, **kwargs)

        # 创建一个错误数据类型的空数组 out5，用于测试错误的数据类型输入时是否抛出异常
        out5 = np.empty((out_r, out_c), dtype=np.int64)
        with pytest.raises(ValueError):
            # 测试 cdist 函数在错误的数据类型输入时是否抛出 ValueError 异常
            cdist(X1, X2, metric, out=out5, **kwargs)

    # 测试 cdist 函数在处理 striding 时是否正确处理 _copy_array_if_base_present 调用
    def test_striding(self, metric):
        # 定义一个极小的数值误差阈值
        eps = 1e-15
        # 从测试数据中获取 X1 和 X2，并使用步长为2的切片获取其中的子集 X1 和 X2
        X1 = eo['cdist-X1'][::2, ::2]
        X2 = eo['cdist-X2'][::2, ::2]
        # 复制 X1 和 X2 的切片，以便后续的比较和确认
        X1_copy = X1.copy()
        X2_copy = X2.copy()

        # 确认 X1 和 X1_copy，X2 和 X2_copy 的值相等
        assert_equal(X1, X1_copy)
        assert_equal(X2, X2_copy)
        # 确认 X1 和 X2 不是 C-contiguous（非连续存储）
        assert_(not X1.flags.c_contiguous)
        assert_(not X2.flags.c_contiguous)
        # 确认 X1_copy 和 X2_copy 是 C-contiguous（连续存储）
        assert_(X1_copy.flags.c_contiguous)
        assert_(X2_copy.flags.c_contiguous)

        kwargs = dict()
        # 如果距离度量是 'minkowski'，设置额外的参数 p 为 1.23
        if metric == 'minkowski':
            kwargs['p'] = 1.23
        # 调用 cdist 函数计算两个数据集之间的距离，存储在 Y1 中
        Y1 = cdist(X1, X2, metric, **kwargs)
        # 调用 cdist 函数计算复制的数据集之间的距离，存储在 Y2 中
        Y2 = cdist(X1_copy, X2_copy, metric, **kwargs)
        # 断言 Y1 和 Y2 的值在数值上非常接近，rtol 为误差容忍度
        assert_allclose(Y1, Y2, rtol=eps, verbose=verbose > 2)

    # 测试 cdist 函数在处理引用计数时是否正确
    def test_cdist_refcount(self, metric):
        # 创建两个随机的 10x10 的数组 x1 和 x2
        x1 = np.random.rand(10, 10)
        x2 = np.random.rand(10, 10)

        kwargs = dict()
        # 如果距离度量是 'minkowski'，设置额外的参数 p 为 1.23
        if metric == 'minkowski':
            kwargs['p'] = 1.23

        # 调用 cdist 函数计算两个数组之间的距离，将结果存储在 out 中
        out = cdist(x1, x2, metric=metric, **kwargs)

        # 检查引用计数是否正确，如果只有弱引用存在，数组应该被释放
        weak_refs = [weakref.ref(v) for v in (x1, x2, out)]
        del x1, x2, out

        if IS_PYPY:
            break_cycles()
        # 断言所有数组的引用都已经被释放
        assert all(weak_ref() is None for weak_ref in weak_refs)
class TestPdist:
    # 定义测试类 TestPdist

    def setup_method(self):
        # 在每个测试方法运行前设置初始化方法
        self.rnd_eo_names = ['random-float32-data', 'random-int-data',
                             'random-uint-data', 'random-double-data',
                             'random-bool-data']
        # 初始化随机数据集的文件名列表

        self.valid_upcasts = {'bool': [np_ulong, np_long, np.float32, np.float64],
                              'uint': [np_long, np.float32, np.float64],
                              'int': [np.float32, np.float64],
                              'float32': [np.float64]}
        # 定义数据类型上转型的有效映射字典

    def test_pdist_extra_args(self, metric):
        # 测试参数和关键字参数的正确处理
        X1 = [[1., 2.], [1.2, 2.3], [2.2, 2.3]]
        # 定义测试用的数据集 X1
        kwargs = {"N0tV4l1D_p4raM": 3.14, "w": np.arange(2)}
        # 定义测试用的关键字参数字典 kwargs
        args = [3.14] * 200
        # 定义测试用的位置参数列表 args

        with pytest.raises(TypeError):
            # 测试期望抛出 TypeError 异常
            pdist(X1, metric=metric, **kwargs)
            # 调用 pdist 函数，使用 metric 参数和关键字参数 kwargs

        with pytest.raises(TypeError):
            # 测试期望抛出 TypeError 异常
            pdist(X1, metric=eval(metric), **kwargs)
            # 调用 pdist 函数，使用评估后的 metric 参数和关键字参数 kwargs

        with pytest.raises(TypeError):
            # 测试期望抛出 TypeError 异常
            pdist(X1, metric="test_" + metric, **kwargs)
            # 调用 pdist 函数，使用测试 metric 参数和关键字参数 kwargs

        with pytest.raises(TypeError):
            # 测试期望抛出 TypeError 异常
            pdist(X1, metric=metric, *args)
            # 调用 pdist 函数，使用 metric 参数和位置参数 args

        with pytest.raises(TypeError):
            # 测试期望抛出 TypeError 异常
            pdist(X1, metric=eval(metric), *args)
            # 调用 pdist 函数，使用评估后的 metric 参数和位置参数 args

        with pytest.raises(TypeError):
            # 测试期望抛出 TypeError 异常
            pdist(X1, metric="test_" + metric, *args)
            # 调用 pdist 函数，使用测试 metric 参数和位置参数 args

    def test_pdist_extra_args_custom(self):
        # 测试参数和关键字参数的正确处理，同时测试自定义度量函数
        def _my_metric(x, y, arg, kwarg=1, kwarg2=2):
            return arg + kwarg + kwarg2
        # 定义一个自定义的度量函数 _my_metric

        X1 = [[1., 2.], [1.2, 2.3], [2.2, 2.3]]
        # 定义测试用的数据集 X1

        kwargs = {"N0tV4l1D_p4raM": 3.14, "w": np.arange(2)}
        # 定义测试用的关键字参数字典 kwargs

        args = [3.14] * 200
        # 定义测试用的位置参数列表 args

        with pytest.raises(TypeError):
            # 测试期望抛出 TypeError 异常
            pdist(X1, _my_metric)
            # 调用 pdist 函数，使用自定义度量函数 _my_metric

        with pytest.raises(TypeError):
            # 测试期望抛出 TypeError 异常
            pdist(X1, _my_metric, *args)
            # 调用 pdist 函数，使用自定义度量函数 _my_metric 和位置参数 args

        with pytest.raises(TypeError):
            # 测试期望抛出 TypeError 异常
            pdist(X1, _my_metric, **kwargs)
            # 调用 pdist 函数，使用自定义度量函数 _my_metric 和关键字参数 kwargs

        with pytest.raises(TypeError):
            # 测试期望抛出 TypeError 异常
            pdist(X1, _my_metric, kwarg=2.2, kwarg2=3.3)
            # 调用 pdist 函数，使用自定义度量函数 _my_metric 和关键字参数 kwarg=2.2, kwarg2=3.3

        with pytest.raises(TypeError):
            # 测试期望抛出 TypeError 异常
            pdist(X1, _my_metric, 1, 2, kwarg=2.2)
            # 调用 pdist 函数，使用自定义度量函数 _my_metric 和位置参数 1, 2，关键字参数 kwarg=2.2

        with pytest.raises(TypeError):
            # 测试期望抛出 TypeError 异常
            pdist(X1, _my_metric, 1.1, 2.2)
            # 调用 pdist 函数，使用自定义度量函数 _my_metric 和位置参数 1.1, 2.2

        with pytest.raises(TypeError):
            # 测试期望抛出 TypeError 异常
            pdist(X1, _my_metric, 1.1, kwarg=2.2, kwarg2=3.3)
            # 调用 pdist 函数，使用自定义度量函数 _my_metric 和位置参数 1.1，关键字参数 kwarg=2.2, kwarg2=3.3

        # 这些应该正常工作
        assert_allclose(pdist(X1, metric=_my_metric,
                              arg=1.1, kwarg2=3.3), 5.4)
        # 使用 assert_allclose 函数验证 pdist 调用结果与期望值的接近程度

    def test_pdist_euclidean_random(self):
        # 测试欧氏距离计算的随机数据集
        eps = 1e-07
        # 设置误差阈值
        X = eo['pdist-double-inp']
        # 从 eo 数据集中获取双精度输入数据 X
        Y_right = eo['pdist-euclidean']
        # 从 eo 数据集中获取欧氏距离的正确结果 Y_right
        Y_test1 = wpdist_no_const(X, 'euclidean')
        # 使用 wpdist_no_const 函数计算数据集 X 的欧氏距离，结果为 Y_test1
        assert_allclose(Y_test1, Y_right, rtol=eps)
        # 使用 assert_allclose 函数验证 Y_test1 与 Y_right 的接近程度，相对误差为 eps
    # 定义测试函数，测试使用随机数据进行欧几里得距离计算
    def test_pdist_euclidean_random_u(self):
        # 设置误差阈值
        eps = 1e-07
        # 获取测试数据 X 和预期结果 Y_right
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-euclidean']
        # 调用函数 wpdist_no_const 计算欧几里得距离
        Y_test1 = wpdist_no_const(X, 'euclidean')
        # 使用 assert_allclose 断言 Y_test1 与 Y_right 在指定的相对误差范围内相等
        assert_allclose(Y_test1, Y_right, rtol=eps)

    # 定义测试函数，测试使用 float32 类型的数据进行欧几里得距离计算
    def test_pdist_euclidean_random_float32(self):
        # 设置误差阈值
        eps = 1e-07
        # 获取测试数据 X 和预期结果 Y_right，将 X 转换为 float32 类型
        X = np.float32(eo['pdist-double-inp'])
        Y_right = eo['pdist-euclidean']
        # 调用函数 wpdist_no_const 计算欧几里得距离
        Y_test1 = wpdist_no_const(X, 'euclidean')
        # 使用 assert_allclose 断言 Y_test1 与 Y_right 在指定的相对误差范围内相等
        assert_allclose(Y_test1, Y_right, rtol=eps)

    # 定义测试函数，测试使用非 C 实现的欧几里得距离计算
    def test_pdist_euclidean_random_nonC(self):
        # 设置误差阈值
        eps = 1e-07
        # 获取测试数据 X 和预期结果 Y_right
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-euclidean']
        # 调用函数 wpdist_no_const 使用测试欧几里得距离的非 C 实现
        Y_test2 = wpdist_no_const(X, 'test_euclidean')
        # 使用 assert_allclose 断言 Y_test2 与 Y_right 在指定的相对误差范围内相等
        assert_allclose(Y_test2, Y_right, rtol=eps)

    # 使用 pytest 的标记标记为慢速测试，测试双精度 Iris 数据集的欧几里得距离计算
    @pytest.mark.slow
    def test_pdist_euclidean_iris_double(self):
        # 设置误差阈值
        eps = 1e-7
        # 获取测试数据 X 和预期结果 Y_right
        X = eo['iris']
        Y_right = eo['pdist-euclidean-iris']
        # 调用函数 wpdist_no_const 计算欧几里得距离
        Y_test1 = wpdist_no_const(X, 'euclidean')
        # 使用 assert_allclose 断言 Y_test1 与 Y_right 在指定的相对误差范围内相等
        assert_allclose(Y_test1, Y_right, rtol=eps)

    # 使用 pytest 的标记标记为慢速测试，测试 float32 类型的 Iris 数据集的欧几里得距离计算
    @pytest.mark.slow
    def test_pdist_euclidean_iris_float32(self):
        # 设置误差阈值
        eps = 1e-5
        # 获取测试数据 X 和预期结果 Y_right，将 X 转换为 float32 类型
        X = np.float32(eo['iris'])
        Y_right = eo['pdist-euclidean-iris']
        # 调用函数 wpdist_no_const 计算欧几里得距离
        Y_test1 = wpdist_no_const(X, 'euclidean')
        # 使用 assert_allclose 断言 Y_test1 与 Y_right 在指定的相对误差范围内相等，输出详细信息
        assert_allclose(Y_test1, Y_right, rtol=eps, verbose=verbose > 2)

    # 使用 pytest 的标记标记为慢速测试，测试非 C 实现的 Iris 数据集的欧几里得距离计算
    @pytest.mark.slow
    def test_pdist_euclidean_iris_nonC(self):
        # 设置误差阈值
        eps = 1e-7
        # 获取测试数据 X 和预期结果 Y_right
        X = eo['iris']
        Y_right = eo['pdist-euclidean-iris']
        # 调用函数 wpdist_no_const 使用测试欧几里得距离的非 C 实现
        Y_test2 = wpdist_no_const(X, 'test_euclidean')
        # 使用 assert_allclose 断言 Y_test2 与 Y_right 在指定的相对误差范围内相等
        assert_allclose(Y_test2, Y_right, rtol=eps)

    # 测试使用随机数据进行标准化欧几里得距离计算
    def test_pdist_seuclidean_random(self):
        # 设置误差阈值
        eps = 1e-7
        # 获取测试数据 X 和预期结果 Y_right
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-seuclidean']
        # 调用函数 pdist 计算标准化欧几里得距离
        Y_test1 = pdist(X, 'seuclidean')
        # 使用 assert_allclose 断言 Y_test1 与 Y_right 在指定的相对误差范围内相等
        assert_allclose(Y_test1, Y_right, rtol=eps)

    # 测试使用 float32 类型的随机数据进行标准化欧几里得距离计算
    def test_pdist_seuclidean_random_float32(self):
        # 设置误差阈值
        eps = 1e-7
        # 获取测试数据 X 和预期结果 Y_right，将 X 转换为 float32 类型
        X = np.float32(eo['pdist-double-inp'])
        Y_right = eo['pdist-seuclidean']
        # 调用函数 pdist 计算标准化欧几里得距离
        Y_test1 = pdist(X, 'seuclidean')
        # 使用 assert_allclose 断言 Y_test1 与 Y_right 在指定的相对误差范围内相等
        assert_allclose(Y_test1, Y_right, rtol=eps)

        # 检查在 V 具有 float32 数据类型时不会引发错误（GitHub issue #11171）
        V = np.var(X, axis=0, ddof=1)
        Y_test2 = pdist(X, 'seuclidean', V=V)
        # 使用 assert_allclose 断言 Y_test2 与 Y_right 在指定的相对误差范围内相等
        assert_allclose(Y_test2, Y_right, rtol=eps)

    # 测试使用非 C 实现的随机数据进行标准化欧几里得距离计算
    def test_pdist_seuclidean_random_nonC(self):
        # 测试 pdist(X, 'test_sqeuclidean')，即非 C 实现的标准化欧几里得距离计算
        eps = 1e-07
        # 获取测试数据 X 和预期结果 Y_right
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-seuclidean']
        # 调用函数 pdist 使用非 C 实现的标准化欧几里得距离
        Y_test2 = pdist(X, 'test_seuclidean')
        # 使用 assert_allclose 断言 Y_test2 与 Y_right 在指定的相对误差范围内相等
        assert_allclose(Y_test2, Y_right, rtol=eps)

    # 测试使用 Iris 数据集进行标准化欧几里得距离计算
    def test_pdist_seuclidean_iris(self):
        # 设置误差阈值
        eps = 1e-7
        # 获取测试数据 X 和预期结果 Y_right
        X = eo['iris']
        Y_right = eo['pdist-seuclidean-iris']
        # 调用函数 pdist 计算标
    # 定义一个测试函数，用于测试 pdist(X, 'seuclidean') 在 Iris 数据集（float32）上的表现
    def test_pdist_seuclidean_iris_float32(self):
        # 设置容差阈值
        eps = 1e-5
        # 从数据字典中获取 Iris 数据集并转换为 float32 类型
        X = np.float32(eo['iris'])
        # 从数据字典中获取预期的结果
        Y_right = eo['pdist-seuclidean-iris']
        # 调用 pdist 函数计算 seuclidean 距离
        Y_test1 = pdist(X, 'seuclidean')
        # 使用 assert_allclose 检查 Y_test1 和 Y_right 是否在容差范围内相似
        assert_allclose(Y_test1, Y_right, rtol=eps)

    # 定义一个测试函数，用于测试 pdist(X, 'test_seuclidean') [非 C 实现] 在 Iris 数据集上的表现
    def test_pdist_seuclidean_iris_nonC(self):
        # 设置容差阈值
        eps = 1e-7
        # 从数据字典中获取 Iris 数据集
        X = eo['iris']
        # 从数据字典中获取预期的结果
        Y_right = eo['pdist-seuclidean-iris']
        # 调用 pdist 函数计算 test_seuclidean 距离
        Y_test2 = pdist(X, 'test_seuclidean')
        # 使用 assert_allclose 检查 Y_test2 和 Y_right 是否在容差范围内相似
        assert_allclose(Y_test2, Y_right, rtol=eps)

    # 定义一个测试函数，用于测试 wpdist(X, 'cosine') 在随机数据集上的表现
    def test_pdist_cosine_random(self):
        # 设置容差阈值
        eps = 1e-7
        # 从数据字典中获取随机双精度输入数据集
        X = eo['pdist-double-inp']
        # 从数据字典中获取预期的结果
        Y_right = eo['pdist-cosine']
        # 调用 wpdist 函数计算 cosine 距离
        Y_test1 = wpdist(X, 'cosine')
        # 使用 assert_allclose 检查 Y_test1 和 Y_right 是否在容差范围内相似
        assert_allclose(Y_test1, Y_right, rtol=eps)

    # 定义一个测试函数，用于测试 wpdist(X, 'cosine') 在随机 float32 数据集上的表现
    def test_pdist_cosine_random_float32(self):
        # 设置容差阈值
        eps = 1e-7
        # 从数据字典中获取随机双精度输入数据集并转换为 float32 类型
        X = np.float32(eo['pdist-double-inp'])
        # 从数据字典中获取预期的结果
        Y_right = eo['pdist-cosine']
        # 调用 wpdist 函数计算 cosine 距离
        Y_test1 = wpdist(X, 'cosine')
        # 使用 assert_allclose 检查 Y_test1 和 Y_right 是否在容差范围内相似，同时设置更详细的 verbose 模式
        assert_allclose(Y_test1, Y_right, rtol=eps, verbose=verbose > 2)

    # 定义一个测试函数，用于测试 wpdist(X, 'test_cosine') [非 C 实现] 在 Iris 数据集上的表现
    def test_pdist_cosine_random_nonC(self):
        # 设置容差阈值
        eps = 1e-7
        # 从数据字典中获取 Iris 数据集
        X = eo['iris']
        # 从数据字典中获取预期的结果
        Y_right = eo['pdist-cosine']
        # 调用 wpdist 函数计算 test_cosine 距离
        Y_test2 = wpdist(X, 'test_cosine')
        # 使用 assert_allclose 检查 Y_test2 和 Y_right 是否在容差范围内相似
        assert_allclose(Y_test2, Y_right, rtol=eps)

    # 定义一个测试函数，用于测试 wpdist(X, 'cosine') 在特定边界条件下的表现
    def test_pdist_cosine_bounds(self):
        # 测试来自 @joernhees 的案例，检查余弦距离是否可能为负值
        x = np.abs(np.random.RandomState(1337).rand(91))
        # 构建一个包含相同行的垂直堆栈
        X = np.vstack([x, x])
        # 使用 assert_ 确保计算的余弦距离不为负
        assert_(wpdist(X, 'cosine')[0] >= 0,
                msg='cosine distance should be non-negative')

    # 定义一个测试函数，用于测试 wpdist_no_const(X, 'cityblock') 在随机数据集上的表现
    def test_pdist_cityblock_random(self):
        # 设置容差阈值
        eps = 1e-7
        # 从数据字典中获取随机双精度输入数据集
        X = eo['pdist-double-inp']
        # 从数据字典中获取预期的结果
        Y_right = eo['pdist-cityblock']
        # 调用 wpdist_no_const 函数计算 cityblock 距离
        Y_test1 = wpdist_no_const(X, 'cityblock')
        # 使用 assert_allclose 检查 Y_test1 和 Y_right 是否在容差范围内相似
        assert_allclose(Y_test1, Y_right, rtol=eps)
    # 定义测试函数，用于测试带有城市街区距离度量的随机浮点数输入
    def test_pdist_cityblock_random_float32(self):
        eps = 1e-7
        # 从测试数据集中获取浮点数输入
        X = np.float32(eo['pdist-double-inp'])
        # 获取预期的城市街区距离度量结果
        Y_right = eo['pdist-cityblock']
        # 调用函数 wpdist_no_const 计算城市街区距离
        Y_test1 = wpdist_no_const(X, 'cityblock')
        # 使用 assert_allclose 断言实际计算结果与预期结果的接近程度
        assert_allclose(Y_test1, Y_right, rtol=eps)

    # 定义测试函数，用于测试带有城市街区距离度量的随机非 C 输入
    def test_pdist_cityblock_random_nonC(self):
        eps = 1e-7
        # 从测试数据集中获取非 C 输入
        X = eo['pdist-double-inp']
        # 获取预期的城市街区距离度量结果
        Y_right = eo['pdist-cityblock']
        # 调用函数 wpdist_no_const 计算城市街区距离
        Y_test2 = wpdist_no_const(X, 'test_cityblock')
        # 使用 assert_allclose 断言实际计算结果与预期结果的接近程度
        assert_allclose(Y_test2, Y_right, rtol=eps)

    # 标记为慢速测试，用于测试带有城市街区距离度量的 Iris 数据集
    @pytest.mark.slow
    def test_pdist_cityblock_iris(self):
        eps = 1e-14
        # 获取 Iris 数据集
        X = eo['iris']
        # 获取预期的城市街区距离度量结果
        Y_right = eo['pdist-cityblock-iris']
        # 调用函数 wpdist_no_const 计算城市街区距离
        Y_test1 = wpdist_no_const(X, 'cityblock')
        # 使用 assert_allclose 断言实际计算结果与预期结果的接近程度
        assert_allclose(Y_test1, Y_right, rtol=eps)

    # 标记为慢速测试，用于测试带有城市街区距离度量的 Iris 数据集（浮点数输入）
    @pytest.mark.slow
    def test_pdist_cityblock_iris_float32(self):
        eps = 1e-5
        # 从测试数据集中获取浮点数输入的 Iris 数据集
        X = np.float32(eo['iris'])
        # 获取预期的城市街区距离度量结果
        Y_right = eo['pdist-cityblock-iris']
        # 调用函数 wpdist_no_const 计算城市街区距离
        Y_test1 = wpdist_no_const(X, 'cityblock')
        # 使用 assert_allclose 断言实际计算结果与预期结果的接近程度，并指定详细输出
        assert_allclose(Y_test1, Y_right, rtol=eps, verbose=verbose > 2)

    # 标记为慢速测试，用于测试带有城市街区距离度量的 Iris 数据集（非 C 实现）
    @pytest.mark.slow
    def test_pdist_cityblock_iris_nonC(self):
        # 测试在 Iris 数据集上使用 pdist(X, 'test_cityblock') 的非 C 实现
        eps = 1e-14
        # 获取 Iris 数据集
        X = eo['iris']
        # 获取预期的城市街区距离度量结果
        Y_right = eo['pdist-cityblock-iris']
        # 调用函数 wpdist_no_const 计算城市街区距离（非 C 实现）
        Y_test2 = wpdist_no_const(X, 'test_cityblock')
        # 使用 assert_allclose 断言实际计算结果与预期结果的接近程度
        assert_allclose(Y_test2, Y_right, rtol=eps)

    # 定义测试函数，用于测试带有相关系数距离度量的随机输入
    def test_pdist_correlation_random(self):
        eps = 1e-7
        # 从测试数据集中获取输入
        X = eo['pdist-double-inp']
        # 获取预期的相关系数距离度量结果
        Y_right = eo['pdist-correlation']
        # 调用函数 wpdist 计算相关系数距离
        Y_test1 = wpdist(X, 'correlation')
        # 使用 assert_allclose 断言实际计算结果与预期结果的接近程度
        assert_allclose(Y_test1, Y_right, rtol=eps)

    # 定义测试函数，用于测试带有相关系数距离度量的随机浮点数输入
    def test_pdist_correlation_random_float32(self):
        eps = 1e-7
        # 从测试数据集中获取浮点数输入
        X = np.float32(eo['pdist-double-inp'])
        # 获取预期的相关系数距离度量结果
        Y_right = eo['pdist-correlation']
        # 调用函数 wpdist 计算相关系数距离
        Y_test1 = wpdist(X, 'correlation')
        # 使用 assert_allclose 断言实际计算结果与预期结果的接近程度
        assert_allclose(Y_test1, Y_right, rtol=eps)

    # 定义测试函数，用于测试带有相关系数距离度量的随机非 C 输入
    def test_pdist_correlation_random_nonC(self):
        eps = 1e-7
        # 从测试数据集中获取非 C 输入
        X = eo['pdist-double-inp']
        # 获取预期的相关系数距离度量结果
        Y_right = eo['pdist-correlation']
        # 调用函数 wpdist 计算相关系数距离（非 C 实现）
        Y_test2 = wpdist(X, 'test_correlation')
        # 使用 assert_allclose 断言实际计算结果与预期结果的接近程度
        assert_allclose(Y_test2, Y_right, rtol=eps)

    # 标记为慢速测试，用于测试带有相关系数距离度量的 Iris 数据集
    @pytest.mark.slow
    def test_pdist_correlation_iris(self):
        eps = 1e-7
        # 获取 Iris 数据集
        X = eo['iris']
        # 获取预期的相关系数距离度量结果
        Y_right = eo['pdist-correlation-iris']
        # 调用函数 wpdist 计算相关系数距离
        Y_test1 = wpdist(X, 'correlation')
        # 使用 assert_allclose 断言实际计算结果与预期结果的接近程度
        assert_allclose(Y_test1, Y_right, rtol=eps)

    # 标记为慢速测试，用于测试带有相关系数距离度量的 Iris 数据集（浮点数输入）
    @pytest.mark.slow
    def test_pdist_correlation_iris_float32(self):
        eps = 1e-7
        # 获取 Iris 数据集
        X = eo['iris']
        # 获取预期的相关系数距离度量结果（转换为浮点数）
        Y_right = np.float32(eo['pdist-correlation-iris'])
        # 调用函数 wpdist 计算相关系数距离
        Y_test1 = wpdist(X, 'correlation')
        # 使用 assert_allclose 断言实际计算结果与预期结果的接近程度，并指定详细输出
        assert_allclose(Y_test1, Y_right, rtol=eps, verbose=verbose > 2)
    def test_pdist_correlation_iris_nonC(self):
        # 检查系统是否支持大于32位的整数，设置浮点数精度阈值
        if sys.maxsize > 2**32:
            eps = 1e-7
        else:
            # 如果系统不支持，跳过测试，并提示查看 gh-16456
            pytest.skip("see gh-16456")
        # 获取测试数据 X 和参考数据 Y_right
        X = eo['iris']
        Y_right = eo['pdist-correlation-iris']
        # 计算 X 的相关性距离，得到测试数据 Y_test2
        Y_test2 = wpdist(X, 'test_correlation')
        # 断言 Y_test2 和 Y_right 的接近程度，使用相对误差 eps
        assert_allclose(Y_test2, Y_right, rtol=eps)

    @pytest.mark.parametrize("p", [0.1, 0.25, 1.0, 2.0, 3.2, np.inf])
    def test_pdist_minkowski_random_p(self, p):
        # 设置浮点数精度阈值
        eps = 1e-13
        # 获取测试数据 X
        X = eo['pdist-double-inp']
        # 计算使用不同 p 值的 Minkowski 距离，得到 Y1 和 Y2
        Y1 = wpdist_no_const(X, 'minkowski', p=p)
        Y2 = wpdist_no_const(X, 'test_minkowski', p=p)
        # 断言 Y1 和 Y2 的接近程度，使用绝对误差 atol 和相对误差 eps
        assert_allclose(Y1, Y2, atol=0, rtol=eps)

    def test_pdist_minkowski_random(self):
        # 设置浮点数精度阈值
        eps = 1e-7
        # 获取测试数据 X 和参考数据 Y_right
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-minkowski-3.2']
        # 计算 Minkowski 距离为 3.2 的测试数据 Y_test1
        Y_test1 = wpdist_no_const(X, 'minkowski', p=3.2)
        # 断言 Y_test1 和 Y_right 的接近程度，使用相对误差 eps
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_minkowski_random_float32(self):
        # 设置浮点数精度阈值
        eps = 1e-7
        # 获取测试数据 X（转换为 float32 类型）和参考数据 Y_right
        X = np.float32(eo['pdist-double-inp'])
        Y_right = eo['pdist-minkowski-3.2']
        # 计算 Minkowski 距离为 3.2 的测试数据 Y_test1
        Y_test1 = wpdist_no_const(X, 'minkowski', p=3.2)
        # 断言 Y_test1 和 Y_right 的接近程度，使用相对误差 eps
        assert_allclose(Y_test1, Y_right, rtol=eps)

    def test_pdist_minkowski_random_nonC(self):
        # 设置浮点数精度阈值
        eps = 1e-7
        # 获取测试数据 X 和参考数据 Y_right
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-minkowski-3.2']
        # 计算使用函数 'test_minkowski' 计算的 Minkowski 距离为 3.2 的测试数据 Y_test2
        Y_test2 = wpdist_no_const(X, 'test_minkowski', p=3.2)
        # 断言 Y_test2 和 Y_right 的接近程度，使用相对误差 eps
        assert_allclose(Y_test2, Y_right, rtol=eps)

    @pytest.mark.slow
    def test_pdist_minkowski_3_2_iris(self):
        # 设置浮点数精度阈值
        eps = 1e-7
        # 获取测试数据 X 和参考数据 Y_right
        X = eo['iris']
        Y_right = eo['pdist-minkowski-3.2-iris']
        # 计算 Iris 数据集上 Minkowski 距离为 3.2 的测试数据 Y_test1
        Y_test1 = wpdist_no_const(X, 'minkowski', p=3.2)
        # 断言 Y_test1 和 Y_right 的接近程度，使用相对误差 eps
        assert_allclose(Y_test1, Y_right, rtol=eps)

    @pytest.mark.slow
    def test_pdist_minkowski_3_2_iris_float32(self):
        # 设置浮点数精度阈值
        eps = 1e-5
        # 获取测试数据 X（转换为 float32 类型）和参考数据 Y_right
        X = np.float32(eo['iris'])
        Y_right = eo['pdist-minkowski-3.2-iris']
        # 计算 Iris 数据集上 Minkowski 距离为 3.2 的测试数据 Y_test1
        Y_test1 = wpdist_no_const(X, 'minkowski', p=3.2)
        # 断言 Y_test1 和 Y_right 的接近程度，使用相对误差 eps
        assert_allclose(Y_test1, Y_right, rtol=eps)

    @pytest.mark.slow
    def test_pdist_minkowski_3_2_iris_nonC(self):
        # 设置浮点数精度阈值
        eps = 1e-7
        # 获取测试数据 X 和参考数据 Y_right
        X = eo['iris']
        Y_right = eo['pdist-minkowski-3.2-iris']
        # 计算使用函数 'test_minkowski' 计算的 Iris 数据集上 Minkowski 距离为 3.2 的测试数据 Y_test2
        Y_test2 = wpdist_no_const(X, 'test_minkowski', p=3.2)
        # 断言 Y_test2 和 Y_right 的接近程度，使用相对误差 eps
        assert_allclose(Y_test2, Y_right, rtol=eps)

    @pytest.mark.slow
    def test_pdist_minkowski_5_8_iris(self):
        # 设置浮点数精度阈值
        eps = 1e-7
        # 获取测试数据 X 和参考数据 Y_right
        X = eo['iris']
        Y_right = eo['pdist-minkowski-5.8-iris']
        # 计算 Iris 数据集上 Minkowski 距离为 5.8 的测试数据 Y_test1
        Y_test1 = wpdist_no_const(X, 'minkowski', p=5.8)
        # 断言 Y_test1 和 Y_right 的接近程度，使用相对误差 eps
        assert_allclose(Y_test1, Y_right, rtol=eps)

    @pytest.mark.slow
    def test_pdist_minkowski_5_8_iris_float32(self):
        # 设置浮点数精度阈值
        eps = 1e-5
        # 获取测试数据 X（转换为 float32 类型）和参考数据 Y_right
        X = np.float32(eo['iris'])
        Y_right = eo['pdist-minkowski-5.8-iris']
        # 计算 Iris 数据集上 Minkowski 距离为 5.8 的测试数据 Y_test1
        Y_test1 = wpdist_no_const(X, 'minkowski', p=5.8)
        # 断言 Y_test1 和 Y_right 的接近程度，使用相对误差 eps，同时输出详细信息
        assert_allclose(Y_test1, Y_right, rtol=eps, verbose=verbose > 2)
    # 测试使用 Minkowski 距离计算非常规化 Iris 数据集的相似性
    def test_pdist_minkowski_5_8_iris_nonC(self):
        # 设定数值稳定性的阈值
        eps = 1e-7
        # 从外部数据对象中获取 Iris 数据集
        X = eo['iris']
        # 获取预先计算好的 Minkowski 距离
        Y_right = eo['pdist-minkowski-5.8-iris']
        # 使用自定义函数计算 Minkowski 距离，并与预期结果比较
        Y_test2 = wpdist_no_const(X, 'test_minkowski', p=5.8)
        assert_allclose(Y_test2, Y_right, rtol=eps)

    # 测试使用 Mahalanobis 距离计算不同维度数据集的相似性
    def test_pdist_mahalanobis(self):
        # 1维观测值
        x = np.array([2.0, 2.0, 3.0, 5.0]).reshape(-1, 1)
        # 计算 Mahalanobis 距离
        dist = pdist(x, metric='mahalanobis')
        assert_allclose(dist, [0.0, np.sqrt(0.5), np.sqrt(4.5),
                               np.sqrt(0.5), np.sqrt(4.5), np.sqrt(2.0)])

        # 2维观测值
        x = np.array([[0, 0], [-1, 0], [0, 2], [1, 0], [0, -2]])
        # 再次计算 Mahalanobis 距离
        dist = pdist(x, metric='mahalanobis')
        rt2 = np.sqrt(2)
        assert_allclose(dist, [rt2, rt2, rt2, rt2, 2, 2 * rt2, 2, 2, 2 * rt2, 2])

        # 观测值数量过少
        with pytest.raises(ValueError):
            wpdist([[0, 1], [2, 3]], metric='mahalanobis')

    # 测试使用 Hamming 距离计算随机数据集的相似性
    def test_pdist_hamming_random(self):
        # 设定数值稳定性的阈值
        eps = 1e-15
        # 从外部数据对象中获取布尔输入数据集
        X = eo['pdist-boolean-inp']
        # 获取预先计算好的 Hamming 距离
        Y_right = eo['pdist-hamming']
        # 使用自定义函数计算 Hamming 距离，并与预期结果比较
        Y_test1 = wpdist(X, 'hamming')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    # 测试使用 Hamming 距离计算随机数据集的相似性（使用 float32 类型）
    def test_pdist_hamming_random_float32(self):
        # 设定数值稳定性的阈值
        eps = 1e-15
        # 从外部数据对象中获取布尔输入数据集，并转换为 float32 类型
        X = np.float32(eo['pdist-boolean-inp'])
        # 获取预先计算好的 Hamming 距离
        Y_right = eo['pdist-hamming']
        # 使用自定义函数计算 Hamming 距离，并与预期结果比较
        Y_test1 = wpdist(X, 'hamming')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    # 测试使用 Hamming 距离计算随机数据集的相似性（非常规化）
    def test_pdist_hamming_random_nonC(self):
        # 设定数值稳定性的阈值
        eps = 1e-15
        # 从外部数据对象中获取布尔输入数据集
        X = eo['pdist-boolean-inp']
        # 获取预先计算好的 Hamming 距离
        Y_right = eo['pdist-hamming']
        # 使用自定义函数计算 Hamming 距离（使用非标准方法），并与预期结果比较
        Y_test2 = wpdist(X, 'test_hamming')
        assert_allclose(Y_test2, Y_right, rtol=eps)

    # 测试使用 Hamming 距离计算随机数据集的相似性（使用 double 类型）
    def test_pdist_dhamming_random(self):
        # 设定数值稳定性的阈值
        eps = 1e-15
        # 从外部数据对象中获取布尔输入数据集，并转换为 float64 类型
        X = np.float64(eo['pdist-boolean-inp'])
        # 获取预先计算好的 Hamming 距离
        Y_right = eo['pdist-hamming']
        # 使用自定义函数计算 Hamming 距离，并与预期结果比较
        Y_test1 = wpdist(X, 'hamming')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    # 测试使用 Hamming 距离计算随机数据集的相似性（使用 double 类型和非标准方法）
    def test_pdist_dhamming_random_float32(self):
        # 设定数值稳定性的阈值
        eps = 1e-15
        # 从外部数据对象中获取布尔输入数据集，并转换为 float32 类型
        X = np.float32(eo['pdist-boolean-inp'])
        # 获取预先计算好的 Hamming 距离
        Y_right = eo['pdist-hamming']
        # 使用自定义函数计算 Hamming 距离，并与预期结果比较
        Y_test1 = wpdist(X, 'hamming')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    # 测试使用 Jaccard 距离计算随机数据集的相似性
    def test_pdist_jaccard_random(self):
        # 设定数值稳定性的阈值
        eps = 1e-8
        # 从外部数据对象中获取布尔输入数据集
        X = eo['pdist-boolean-inp']
        # 获取预先计算好的 Jaccard 距离
        Y_right = eo['pdist-jaccard']
        # 使用自定义函数计算 Jaccard 距离，并与预期结果比较
        Y_test1 = wpdist(X, 'jaccard')
        assert_allclose(Y_test1, Y_right, rtol=eps)

    # 测试使用 Jaccard 距离计算随机数据集的相似性（使用 float32 类型）
    def test_pdist_jaccard_random_float32(self):
        # 设定数值稳定性的阈值
        eps = 1e-8
        # 从外部数据对象中获取布尔输入数据集，并转换为 float32 类型
        X = np.float32(eo['pdist-boolean-inp'])
        # 获取预先计算好的 Jaccard 距离
        Y_right = eo['pdist-jaccard']
        # 使用自定义函数计算 Jaccard 距离，并与预期结果比较
        Y_test1 = wpdist(X, 'jaccard')
        assert_allclose(Y_test1, Y_right, rtol=eps)
    # 测试用例：计算基于布尔输入数据的 Jaccard 距离，并验证与预期结果的接近程度
    def test_pdist_jaccard_random_nonC(self):
        eps = 1e-8
        # 获取布尔输入数据
        X = eo['pdist-boolean-inp']
        # 获取预期的 Jaccard 距离结果
        Y_right = eo['pdist-jaccard']
        # 使用 wpdist 函数计算 Jaccard 距离
        Y_test2 = wpdist(X, 'test_jaccard')
        # 断言计算得到的距离与预期结果在给定的误差范围内接近
        assert_allclose(Y_test2, Y_right, rtol=eps)

    # 测试用例：计算基于浮点64位布尔输入数据的 Jaccard 距离，并验证与预期结果的接近程度
    def test_pdist_djaccard_random(self):
        eps = 1e-8
        # 获取浮点64位布尔输入数据
        X = np.float64(eo['pdist-boolean-inp'])
        # 获取预期的 Jaccard 距离结果
        Y_right = eo['pdist-jaccard']
        # 使用 wpdist 函数计算 Jaccard 距离
        Y_test1 = wpdist(X, 'jaccard')
        # 断言计算得到的距离与预期结果在给定的误差范围内接近
        assert_allclose(Y_test1, Y_right, rtol=eps)

    # 测试用例：计算基于浮点32位布尔输入数据的 Jaccard 距离，并验证与预期结果的接近程度
    def test_pdist_djaccard_random_float32(self):
        eps = 1e-8
        # 获取浮点32位布尔输入数据
        X = np.float32(eo['pdist-boolean-inp'])
        # 获取预期的 Jaccard 距离结果
        Y_right = eo['pdist-jaccard']
        # 使用 wpdist 函数计算 Jaccard 距离
        Y_test1 = wpdist(X, 'jaccard')
        # 断言计算得到的距离与预期结果在给定的误差范围内接近
        assert_allclose(Y_test1, Y_right, rtol=eps)

    # 测试用例：计算全零输入数据的 Jaccard 距离，并验证与预期结果的接近程度
    def test_pdist_djaccard_allzeros(self):
        eps = 1e-15
        # 使用 pdist 函数计算全零输入数据的 Jaccard 距离
        Y = pdist(np.zeros((5, 3)), 'jaccard')
        # 断言计算得到的距离全为零
        assert_allclose(np.zeros(10), Y, rtol=eps)

    # 测试用例：计算基于非C输入的浮点64位布尔数据的 Jaccard 距离，并验证与预期结果的接近程度
    def test_pdist_djaccard_random_nonC(self):
        eps = 1e-8
        # 获取浮点64位布尔输入数据
        X = np.float64(eo['pdist-boolean-inp'])
        # 获取预期的 Jaccard 距离结果
        Y_right = eo['pdist-jaccard']
        # 使用 pdist 函数计算 Jaccard 距离
        Y_test2 = pdist(X, 'test_jaccard')
        # 断言计算得到的距离与预期结果在给定的误差范围内接近
        assert_allclose(Y_test2, Y_right, rtol=eps)

    # 测试用例：计算基于双精度输入数据的 Jensen-Shannon 距离，并验证与预期结果的接近程度
    def test_pdist_jensenshannon_random(self):
        eps = 1e-11
        # 获取双精度输入数据
        X = eo['pdist-double-inp']
        # 获取预期的 Jensen-Shannon 距离结果
        Y_right = eo['pdist-jensenshannon']
        # 使用 pdist 函数计算 Jensen-Shannon 距离
        Y_test1 = pdist(X, 'jensenshannon')
        # 断言计算得到的距离与预期结果在给定的误差范围内接近
        assert_allclose(Y_test1, Y_right, rtol=eps)

    # 测试用例：计算基于浮点32位双精度输入数据的 Jensen-Shannon 距离，并验证与预期结果的接近程度
    def test_pdist_jensenshannon_random_float32(self):
        eps = 1e-8
        # 获取浮点32位双精度输入数据
        X = np.float32(eo['pdist-double-inp'])
        # 获取预期的 Jensen-Shannon 距离结果
        Y_right = eo['pdist-jensenshannon']
        # 使用 pdist 函数计算 Jensen-Shannon 距离
        Y_test1 = pdist(X, 'jensenshannon')
        # 断言计算得到的距离与预期结果在给定的误差范围内接近，输出详细信息
        assert_allclose(Y_test1, Y_right, rtol=eps, verbose=verbose > 2)

    # 测试用例：计算基于非C输入的双精度输入数据的 Jensen-Shannon 距离，并验证与预期结果的接近程度
    def test_pdist_jensenshannon_random_nonC(self):
        eps = 1e-11
        # 获取双精度输入数据
        X = eo['pdist-double-inp']
        # 获取预期的 Jensen-Shannon 距离结果
        Y_right = eo['pdist-jensenshannon']
        # 使用 pdist 函数计算 Jensen-Shannon 距离
        Y_test2 = pdist(X, 'test_jensenshannon')
        # 断言计算得到的距离与预期结果在给定的误差范围内接近
        assert_allclose(Y_test2, Y_right, rtol=eps)

    # 测试用例：计算基于鸢尾花数据的 Jensen-Shannon 距离，并验证与预期结果的接近程度
    def test_pdist_jensenshannon_iris(self):
        if _is_32bit():
            # 在32位 Linux Azure 上的特定情况下测试失败，参见 gh-12810
            eps = 2.5e-10
        else:
            eps = 1e-12

        # 获取鸢尾花数据
        X = eo['iris']
        # 获取预期的鸢尾花数据的 Jensen-Shannon 距离结果
        Y_right = eo['pdist-jensenshannon-iris']
        # 使用 pdist 函数计算 Jensen-Shannon 距离
        Y_test1 = pdist(X, 'jensenshannon')
        # 断言计算得到的距离与预期结果在给定的误差范围内接近
        assert_allclose(Y_test1, Y_right, atol=eps)

    # 测试用例：计算基于浮点32位鸢尾花数据的 Jensen-Shannon 距离，并验证与预期结果的接近程度
    def test_pdist_jensenshannon_iris_float32(self):
        eps = 1e-06
        # 获取浮点32位鸢尾花数据
        X = np.float32(eo['iris'])
        # 获取预期的鸢尾花数据的 Jensen-Shannon 距离结果
        Y_right = eo['pdist-jensenshannon-iris']
        # 使用 pdist 函数计算 Jensen-Shannon 距离
        Y_test1 = pdist(X, 'jensenshannon')
        # 断言计算得到的距离与预期结果在给定的误差范围内接近，输出详细信息
        assert_allclose(Y_test1, Y_right, atol=eps, verbose=verbose > 2)

    # 测试
    def test_pdist_chebyshev_random(self):
        eps = 1e-8  # 定义误差容限
        X = eo['pdist-double-inp']  # 获取测试数据集 X
        Y_right = eo['pdist-chebyshev']  # 获取预期的 Chebyshev 距离结果 Y_right
        Y_test1 = pdist(X, 'chebyshev')  # 计算 X 的 Chebyshev 距离 Y_test1
        assert_allclose(Y_test1, Y_right, rtol=eps)  # 断言 Y_test1 与 Y_right 的近似性

    def test_pdist_chebyshev_random_float32(self):
        eps = 1e-7  # 定义误差容限
        X = np.float32(eo['pdist-double-inp'])  # 将测试数据集 X 转换为 float32 类型
        Y_right = eo['pdist-chebyshev']  # 获取预期的 Chebyshev 距离结果 Y_right
        Y_test1 = pdist(X, 'chebyshev')  # 计算 X 的 Chebyshev 距离 Y_test1
        assert_allclose(Y_test1, Y_right, rtol=eps, verbose=verbose > 2)  # 断言 Y_test1 与 Y_right 的近似性，打印详细信息若 verbose > 2

    def test_pdist_chebyshev_random_nonC(self):
        eps = 1e-8  # 定义误差容限
        X = eo['pdist-double-inp']  # 获取测试数据集 X
        Y_right = eo['pdist-chebyshev']  # 获取预期的 Chebyshev 距离结果 Y_right
        Y_test2 = pdist(X, 'test_chebyshev')  # 使用自定义 Chebyshev 测试计算 X 的距离 Y_test2
        assert_allclose(Y_test2, Y_right, rtol=eps)  # 断言 Y_test2 与 Y_right 的近似性

    def test_pdist_chebyshev_iris(self):
        eps = 1e-14  # 定义误差容限
        X = eo['iris']  # 获取测试数据集 X（鸢尾花数据集）
        Y_right = eo['pdist-chebyshev-iris']  # 获取预期的 Chebyshev 距离结果 Y_right
        Y_test1 = pdist(X, 'chebyshev')  # 计算 X 的 Chebyshev 距离 Y_test1
        assert_allclose(Y_test1, Y_right, rtol=eps)  # 断言 Y_test1 与 Y_right 的近似性

    def test_pdist_chebyshev_iris_float32(self):
        eps = 1e-5  # 定义误差容限
        X = np.float32(eo['iris'])  # 将测试数据集 X（鸢尾花数据集）转换为 float32 类型
        Y_right = eo['pdist-chebyshev-iris']  # 获取预期的 Chebyshev 距离结果 Y_right
        Y_test1 = pdist(X, 'chebyshev')  # 计算 X 的 Chebyshev 距离 Y_test1
        assert_allclose(Y_test1, Y_right, rtol=eps, verbose=verbose > 2)  # 断言 Y_test1 与 Y_right 的近似性，打印详细信息若 verbose > 2

    def test_pdist_chebyshev_iris_nonC(self):
        eps = 1e-14  # 定义误差容限
        X = eo['iris']  # 获取测试数据集 X（鸢尾花数据集）
        Y_right = eo['pdist-chebyshev-iris']  # 获取预期的 Chebyshev 距离结果 Y_right
        Y_test2 = pdist(X, 'test_chebyshev')  # 使用自定义 Chebyshev 测试计算 X 的距离 Y_test2
        assert_allclose(Y_test2, Y_right, rtol=eps)  # 断言 Y_test2 与 Y_right 的近似性

    def test_pdist_matching_mtica1(self):
        # 使用 mtica 示例 #1 进行匹配测试
        m = wmatching(np.array([1, 0, 1, 1, 0]),  # 设置输入向量 m
                      np.array([1, 1, 0, 1, 1]))  # 设置输入向量 n
        m2 = wmatching(np.array([1, 0, 1, 1, 0], dtype=bool),  # 设置输入向量 m2
                       np.array([1, 1, 0, 1, 1], dtype=bool))  # 设置输入向量 n2
        assert_allclose(m, 0.6, rtol=0, atol=1e-10)  # 断言 m 与预期值 0.6 的近似性
        assert_allclose(m2, 0.6, rtol=0, atol=1e-10)  # 断言 m2 与预期值 0.6 的近似性

    def test_pdist_matching_mtica2(self):
        # 使用 mtica 示例 #2 进行匹配测试
        m = wmatching(np.array([1, 0, 1]),  # 设置输入向量 m
                     np.array([1, 1, 0]))  # 设置输入向量 n
        m2 = wmatching(np.array([1, 0, 1], dtype=bool),  # 设置输入向量 m2
                      np.array([1, 1, 0], dtype=bool))  # 设置输入向量 n2
        assert_allclose(m, 2 / 3, rtol=0, atol=1e-10)  # 断言 m 与预期值 2/3 的近似性
        assert_allclose(m2, 2 / 3, rtol=0, atol=1e-10)  # 断言 m2 与预期值 2/3 的近似性

    def test_pdist_jaccard_mtica1(self):
        m = wjaccard(np.array([1, 0, 1, 1, 0]),  # 使用 Jaccard 方法计算输入向量 m 的相似性
                     np.array([1, 1, 0, 1, 1]))  # 使用 Jaccard 方法计算输入向量 n 的相似性
        m2 = wjaccard(np.array([1, 0, 1, 1, 0], dtype=bool),  # 使用 Jaccard 方法计算输入向量 m2 的相似性
                      np.array([1, 1, 0, 1, 1], dtype=bool))  # 使用 Jaccard 方法计算输入向量 n2 的相似性
        assert_allclose(m, 0.6, rtol=0, atol=1e-10)  # 断言 m 与预期值 0.6 的近似性
        assert_allclose(m2, 0.6, rtol=0, atol=1e-10)  # 断言 m2 与预期值 0.6 的近似性

    def test_pdist_jaccard_mtica2(self):
        m = wjaccard(np.array([1, 0, 1]),  # 使用 Jaccard 方法计算输入向量 m 的相似性
                     np.array([1, 1, 0]))  # 使用 Jaccard 方法计算输入向量 n 的相似性
        m2 = wjaccard(np.array([1, 0, 1], dtype=bool),  # 使用 Jaccard 方法计算输入向量 m2 的相似性
                      np.array([1, 1, 0], dtype=bool))  # 使用 Jaccard 方法计算输入向量 n2 的相似性
        assert_allclose(m, 2 / 3, rtol=0, atol=1e-10)  # 断言 m 与预期值 2/3 的近似性
        assert_allclose(m2, 2 / 3, rtol=0, atol=1e-10)  # 断言 m2 与预期值 2/3 的近似性
    # 测试用例，计算两个布尔数组之间的 Yule 距离指数
    def test_pdist_yule_mtica1(self):
        # 创建 Yule 距离指数对象，输入两个数组作为参数
        m = wyule(np.array([1, 0, 1, 1, 0]),
                  np.array([1, 1, 0, 1, 1]))
        # 创建 Yule 距离指数对象，输入两个布尔数组作为参数
        m2 = wyule(np.array([1, 0, 1, 1, 0], dtype=bool),
                   np.array([1, 1, 0, 1, 1], dtype=bool))
        # 如果 verbose 大于 2，打印 m 的值
        if verbose > 2:
            print(m)
        # 断言 m 的值接近于 2，相对误差为 0，绝对误差为 1e-10
        assert_allclose(m, 2, rtol=0, atol=1e-10)
        # 断言 m2 的值接近于 2，相对误差为 0，绝对误差为 1e-10
        assert_allclose(m2, 2, rtol=0, atol=1e-10)

    # 测试用例，计算两个布尔数组之间的 Yule 距离指数
    def test_pdist_yule_mtica2(self):
        # 创建 Yule 距离指数对象，输入两个数组作为参数
        m = wyule(np.array([1, 0, 1]),
                  np.array([1, 1, 0]))
        # 创建 Yule 距离指数对象，输入两个布尔数组作为参数
        m2 = wyule(np.array([1, 0, 1], dtype=bool),
                   np.array([1, 1, 0], dtype=bool))
        # 如果 verbose 大于 2，打印 m 的值
        if verbose > 2:
            print(m)
        # 断言 m 的值接近于 2，相对误差为 0，绝对误差为 1e-10
        assert_allclose(m, 2, rtol=0, atol=1e-10)
        # 断言 m2 的值接近于 2，相对误差为 0，绝对误差为 1e-10
        assert_allclose(m2, 2, rtol=0, atol=1e-10)

    # 测试用例，计算两个布尔数组之间的 Dice 距离指数
    def test_pdist_dice_mtica1(self):
        # 创建 Dice 距离指数对象，输入两个数组作为参数
        m = wdice(np.array([1, 0, 1, 1, 0]),
                  np.array([1, 1, 0, 1, 1]))
        # 创建 Dice 距离指数对象，输入两个布尔数组作为参数
        m2 = wdice(np.array([1, 0, 1, 1, 0], dtype=bool),
                   np.array([1, 1, 0, 1, 1], dtype=bool))
        # 如果 verbose 大于 2，打印 m 的值
        if verbose > 2:
            print(m)
        # 断言 m 的值接近于 3/7，相对误差为 0，绝对误差为 1e-10
        assert_allclose(m, 3 / 7, rtol=0, atol=1e-10)
        # 断言 m2 的值接近于 3/7，相对误差为 0，绝对误差为 1e-10
        assert_allclose(m2, 3 / 7, rtol=0, atol=1e-10)

    # 测试用例，计算两个布尔数组之间的 Dice 距离指数
    def test_pdist_dice_mtica2(self):
        # 创建 Dice 距离指数对象，输入两个数组作为参数
        m = wdice(np.array([1, 0, 1]),
                  np.array([1, 1, 0]))
        # 创建 Dice 距离指数对象，输入两个布尔数组作为参数
        m2 = wdice(np.array([1, 0, 1], dtype=bool),
                   np.array([1, 1, 0], dtype=bool))
        # 如果 verbose 大于 2，打印 m 的值
        if verbose > 2:
            print(m)
        # 断言 m 的值接近于 0.5，相对误差为 0，绝对误差为 1e-10
        assert_allclose(m, 0.5, rtol=0, atol=1e-10)
        # 断言 m2 的值接近于 0.5，相对误差为 0，绝对误差为 1e-10
        assert_allclose(m2, 0.5, rtol=0, atol=1e-10)

    # 测试用例，计算两个布尔数组之间的 Sokal-Sneath 距离指数
    def test_pdist_sokalsneath_mtica1(self):
        # 创建 Sokal-Sneath 距离指数对象，输入两个数组作为参数
        m = sokalsneath(np.array([1, 0, 1, 1, 0]),
                        np.array([1, 1, 0, 1, 1]))
        # 创建 Sokal-Sneath 距离指数对象，输入两个布尔数组作为参数
        m2 = sokalsneath(np.array([1, 0, 1, 1, 0], dtype=bool),
                         np.array([1, 1, 0, 1, 1], dtype=bool))
        # 如果 verbose 大于 2，打印 m 的值
        if verbose > 2:
            print(m)
        # 断言 m 的值接近于 3/4，相对误差为 0，绝对误差为 1e-10
        assert_allclose(m, 3 / 4, rtol=0, atol=1e-10)
        # 断言 m2 的值接近于 3/4，相对误差为 0，绝对误差为 1e-10
        assert_allclose(m2, 3 / 4, rtol=0, atol=1e-10)

    # 测试用例，计算两个布尔数组之间的 Sokal-Sneath 距离指数
    def test_pdist_sokalsneath_mtica2(self):
        # 创建 Sokal-Sneath 距离指数对象，输入两个数组作为参数
        m = wsokalsneath(np.array([1, 0, 1]),
                         np.array([1, 1, 0]))
        # 创建 Sokal-Sneath 距离指数对象，输入两个布尔数组作为参数
        m2 = wsokalsneath(np.array([1, 0, 1], dtype=bool),
                          np.array([1, 1, 0], dtype=bool))
        # 如果 verbose 大于 2，打印 m 的值
        if verbose > 2:
            print(m)
        # 断言 m 的值接近于 4/5，相对误差为 0，绝对误差为 1e-10
        assert_allclose(m, 4 / 5, rtol=0, atol=1e-10)
        # 断言 m2 的值接近于 4/5，相对误差为 0，绝对误差为 1e-10
        assert_allclose(m2, 4 / 5, rtol=0, atol=1e-10)

    # 测试用例，计算两个布尔数组之间的 Rogers-Tanimoto 距离指数
    def test_pdist_rogerstanimoto_mtica1(self):
        # 创建 Rogers-Tanimoto 距离指数对象，输入两个数组作为参数
        m = wrogerstanimoto(np.array([1, 0, 1, 1, 0]),
                            np.array([1, 1, 0, 1, 1]))
        # 创建 Rogers-Tanimoto 距离指数对象，输入两个布尔数组作为参数
        m2 = wrogerstanimoto(np.array([1, 0, 1, 1, 0], dtype=bool),
    # 测试使用 Rogerstanimoto 距离函数计算两个向量之间的相似度
    def test_pdist_rogerstanimoto_mtica2(self):
        # 使用非布尔类型的数组计算 Rogerstanimoto 相似度
        m = wrogerstanimoto(np.array([1, 0, 1]),
                            np.array([1, 1, 0]))
        # 使用布尔类型的数组计算 Rogerstanimoto 相似度
        m2 = wrogerstanimoto(np.array([1, 0, 1], dtype=bool),
                             np.array([1, 1, 0], dtype=bool))
        # 如果 verbose 大于 2，则打印计算结果
        if verbose > 2:
            print(m)
        # 断言计算结果与预期值接近，使用绝对误差和相对误差进行比较
        assert_allclose(m, 4 / 5, rtol=0, atol=1e-10)
        assert_allclose(m2, 4 / 5, rtol=0, atol=1e-10)

    # 测试使用 Russellrao 距离函数计算两个向量之间的相似度
    def test_pdist_russellrao_mtica1(self):
        # 使用非布尔类型的数组计算 Russellrao 相似度
        m = wrussellrao(np.array([1, 0, 1, 1, 0]),
                        np.array([1, 1, 0, 1, 1]))
        # 使用布尔类型的数组计算 Russellrao 相似度
        m2 = wrussellrao(np.array([1, 0, 1, 1, 0], dtype=bool),
                         np.array([1, 1, 0, 1, 1], dtype=bool))
        # 如果 verbose 大于 2，则打印计算结果
        if verbose > 2:
            print(m)
        # 断言计算结果与预期值接近，使用绝对误差和相对误差进行比较
        assert_allclose(m, 3 / 5, rtol=0, atol=1e-10)
        assert_allclose(m2, 3 / 5, rtol=0, atol=1e-10)

    # 测试使用 Russellrao 距离函数计算两个向量之间的相似度
    def test_pdist_russellrao_mtica2(self):
        # 使用非布尔类型的数组计算 Russellrao 相似度
        m = wrussellrao(np.array([1, 0, 1]),
                        np.array([1, 1, 0]))
        # 使用布尔类型的数组计算 Russellrao 相似度
        m2 = wrussellrao(np.array([1, 0, 1], dtype=bool),
                         np.array([1, 1, 0], dtype=bool))
        # 如果 verbose 大于 2，则打印计算结果
        if verbose > 2:
            print(m)
        # 断言计算结果与预期值接近，使用绝对误差和相对误差进行比较
        assert_allclose(m, 2 / 3, rtol=0, atol=1e-10)
        assert_allclose(m2, 2 / 3, rtol=0, atol=1e-10)

    # 标记为慢速测试的测试用例，用于测试使用 Canberra 距离计算的结果
    @pytest.mark.slow
    def test_pdist_canberra_match(self):
        # 从数据集中选择 iris 数据
        D = eo['iris']
        # 如果 verbose 大于 2，则打印数据集的形状和数据类型
        if verbose > 2:
            print(D.shape, D.dtype)
        # 定义非常小的容差值
        eps = 1e-15
        # 计算使用 Canberra 距离的结果 y1
        y1 = wpdist_no_const(D, "canberra")
        # 计算使用测试版 Canberra 距离的结果 y2
        y2 = wpdist_no_const(D, "test_canberra")
        # 断言 y1 和 y2 的计算结果接近，使用绝对误差进行比较
        assert_allclose(y1, y2, rtol=eps, verbose=verbose > 2)

    # 测试解决 GitHub issue #711 中报告的问题，验证 pdist(X, 'canberra') 的结果是否正确
    def test_pdist_canberra_ticket_711(self):
        # 定义非常小的容差值
        eps = 1e-8
        # 使用 pdist 计算 ([3.3], [3.4]) 的 Canberra 距离
        pdist_y = wpdist_no_const(([3.3], [3.4]), "canberra")
        # 正确的计算结果
        right_y = 0.01492537
        # 断言计算结果与预期值接近，使用绝对误差进行比较
        assert_allclose(pdist_y, right_y, atol=eps, verbose=verbose > 2)

    # 测试自定义度量时，数据类型是否保持不变
    def test_pdist_custom_notdouble(self):
        # 定义一个简单的类 myclass
        class myclass:
            pass

        # 定义一个自定义的度量函数 _my_metric
        def _my_metric(x, y):
            # 如果输入的向量不是 myclass 类型，则引发 ValueError
            if not isinstance(x[0], myclass) or not isinstance(y[0], myclass):
                raise ValueError("Type has been changed")
            # 返回固定的相似度值
            return 1.123

        # 创建一个包含 myclass 实例的数组
        data = np.array([[myclass()], [myclass()]], dtype=object)
        # 使用自定义的度量函数 _my_metric 计算数据集的距离
        pdist_y = pdist(data, metric=_my_metric)
        # 正确的计算结果
        right_y = 1.123
        # 断言计算结果与预期值相等
        assert_equal(pdist_y, right_y, verbose=verbose > 2)
    # 定义了一个名为 _check_calling_conventions 的方法，用于测试 pdist 函数的不同调用约定
    def _check_calling_conventions(self, X, metric, eps=1e-07, **kwargs):
        # 尝试使用不同的 metric 参数调用 pdist 函数，并捕获可能的异常
        try:
            # 使用给定的 metric 计算 X 的距离矩阵 y1
            y1 = pdist(X, metric=metric, **kwargs)
            # 使用 eval(metric) 计算 X 的距离矩阵 y2
            y2 = pdist(X, metric=eval(metric), **kwargs)
            # 使用 "test_" + metric 计算 X 的距离矩阵 y3
            y3 = pdist(X, metric="test_" + metric, **kwargs)
        except Exception as e:
            # 如果有异常发生，记录异常类和内容，并根据异常类型使用 pytest 断言
            e_cls = e.__class__
            if verbose > 2:
                print(e_cls.__name__)
                print(e)
            with pytest.raises(e_cls):
                pdist(X, metric=metric, **kwargs)
            with pytest.raises(e_cls):
                pdist(X, metric=eval(metric), **kwargs)
            with pytest.raises(e_cls):
                pdist(X, metric="test_" + metric, **kwargs)
        else:
            # 如果没有异常发生，使用 assert_allclose 断言 y1 与 y2 的结果接近
            assert_allclose(y1, y2, rtol=eps, verbose=verbose > 2)
            # 使用 assert_allclose 断言 y1 与 y3 的结果接近
            assert_allclose(y1, y3, rtol=eps, verbose=verbose > 2)

    # 定义了一个名为 test_pdist_calling_conventions 的方法，用于测试 pdist 函数不同调用约定的行为一致性
    def test_pdist_calling_conventions(self, metric):
        # 确保使用字符串或者 scipy 函数指定 metric 参数会得到相同的行为（即相同的结果或相同的异常）
        # 注意：正确性应在每个 metric 测试中进行检查。
        # 注意：额外的参数应该使用专门的测试进行检查。
        for eo_name in self.rnd_eo_names:
            # 对输入数据进行子采样以加快测试速度
            # 注意：对于 mahalanobis 距离，样本数应该大于维度数
            X = eo[eo_name][::5, ::2]
            if verbose > 2:
                print("testing: ", metric, " with: ", eo_name)
            if metric in {'dice', 'yule', 'matching',
                          'rogerstanimoto', 'russellrao', 'sokalmichener',
                          'sokalsneath',
                          'kulczynski1'} and 'bool' not in eo_name:
                # 对于特定的 metric 和不适用于布尔值的 eo_name，跳过测试
                # Python 版本允许非布尔值，例如模糊逻辑
                continue
            # 调用 _check_calling_conventions 方法来测试当前 metric 的调用约定
            self._check_calling_conventions(X, metric)

            # 使用额外参数测试内置 metric
            if metric == "seuclidean":
                # 计算 X 的每列的方差，并使用 V 作为额外参数进行测试
                V = np.var(X.astype(np.float64), axis=0, ddof=1)
                self._check_calling_conventions(X, metric, V=V)
            elif metric == "mahalanobis":
                # 计算 X 的转置后的协方差矩阵，并计算其逆矩阵 VI
                V = np.atleast_2d(np.cov(X.astype(np.float64).T))
                VI = np.array(np.linalg.inv(V).T)
                # 使用 VI 作为额外参数进行测试
                self._check_calling_conventions(X, metric, VI=VI)
    # 测试函数，用于验证 pdist 函数在不同数据类型上转换时的等效性
    def test_pdist_dtype_equivalence(self, metric):
        # 设置数值精度阈值
        eps = 1e-07
        # 定义一系列测试数据，包括随机布尔型数据、无符号整型数据、有符号整型数据、浮点型数据
        tests = [(eo['random-bool-data'], self.valid_upcasts['bool']),
                 (eo['random-uint-data'], self.valid_upcasts['uint']),
                 (eo['random-int-data'], self.valid_upcasts['int']),
                 (eo['random-float32-data'], self.valid_upcasts['float32'])]
        # 遍历每组测试数据
        for test in tests:
            # 对测试数据进行步长切片
            X1 = test[0][::5, ::2]
            try:
                # 调用 pdist 函数计算距离
                y1 = pdist(X1, metric=metric)
            except Exception as e:
                # 捕获异常并记录异常类型
                e_cls = e.__class__
                # 如果详细输出级别大于2，打印异常类型和具体异常信息
                if verbose > 2:
                    print(e_cls.__name__)
                    print(e)
                # 针对每种新数据类型，进行类型转换并预期引发相同类型的异常
                for new_type in test[1]:
                    X2 = new_type(X1)
                    with pytest.raises(e_cls):
                        pdist(X2, metric=metric)
            else:
                # 如果没有异常发生
                for new_type in test[1]:
                    # 将 X1 转换为新数据类型，并调用 pdist 计算距离
                    y2 = pdist(new_type(X1), metric=metric)
                    # 验证两次计算结果的数值近似性
                    assert_allclose(y1, y2, rtol=eps, verbose=verbose > 2)

    # 测试函数，验证 pdist 函数的 out 参数功能是否正常
    def test_pdist_out(self, metric):
        # 设置数值精度阈值
        eps = 1e-15
        # 从随机浮点型数据中进行步长切片，构造输入数据 X
        X = eo['random-float32-data'][::5, ::2]
        # 计算输出数组 out_size 的大小
        out_size = int((X.shape[0] * (X.shape[0] - 1)) / 2)

        kwargs = dict()
        if metric == 'minkowski':
            kwargs['p'] = 1.23
        # 创建一个空的 float64 类型的数组 out1
        out1 = np.empty(out_size, dtype=np.float64)
        # 调用 pdist 函数计算距离，其中 Y_right 是标准的计算结果
        Y_right = pdist(X, metric, **kwargs)
        # 使用 out1 参数调用 pdist 函数，Y_test1 是计算结果
        Y_test1 = pdist(X, metric, out=out1, **kwargs)

        # 验证计算结果的数值近似性
        assert_allclose(Y_test1, Y_right, rtol=eps)

        # 验证 Y_test1 和 out1 是否是同一个对象
        assert_(Y_test1 is out1)

        # 测试不正确的输出形状
        out2 = np.empty(out_size + 3, dtype=np.float64)
        with pytest.raises(ValueError):
            pdist(X, metric, out=out2, **kwargs)

        # 测试 (C-)连续性输出
        out3 = np.empty(2 * out_size, dtype=np.float64)[::2]
        with pytest.raises(ValueError):
            pdist(X, metric, out=out3, **kwargs)

        # 测试不正确的数据类型
        out5 = np.empty(out_size, dtype=np.int64)
        with pytest.raises(ValueError):
            pdist(X, metric, out=out5, **kwargs)

    # 测试函数，验证 pdist 函数在处理步长时的正确性，包括对 _copy_array_if_base_present 的调用
    def test_striding(self, metric):
        # 设置数值精度阈值
        eps = 1e-15
        # 从随机浮点型数据中进行步长切片，构造输入数据 X
        X = eo['random-float32-data'][::5, ::2]
        # 复制 X，得到 X_copy
        X_copy = X.copy()

        # 确认 X 不是 C 连续的，X_copy 是 C 连续的
        assert_(not X.flags.c_contiguous)
        assert_(X_copy.flags.c_contiguous)

        kwargs = dict()
        if metric == 'minkowski':
            kwargs['p'] = 1.23
        # 调用 pdist 函数计算距离，得到 Y1 和 Y2
        Y1 = pdist(X, metric, **kwargs)
        Y2 = pdist(X_copy, metric, **kwargs)
        # 验证计算结果的数值近似性
        assert_allclose(Y1, Y2, rtol=eps, verbose=verbose > 2)
class TestSomeDistanceFunctions:

    def setup_method(self):
        # 定义1维数组x和y作为测试用例
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 1.0, 5.0])

        # 将测试用例作为元组列表保存在self.cases中
        self.cases = [(x, y)]

    def test_minkowski(self):
        # 遍历self.cases中的每对测试用例(x, y)
        for x, y in self.cases:
            # 计算Minkowski距离，p=1时的距离
            dist1 = minkowski(x, y, p=1)
            assert_almost_equal(dist1, 3.0)
            
            # 计算Minkowski距离，p=1.5时的距离
            dist1p5 = minkowski(x, y, p=1.5)
            assert_almost_equal(dist1p5, (1.0 + 2.0**1.5)**(2. / 3))
            
            # 计算Minkowski距离，p=2时的距离
            dist2 = minkowski(x, y, p=2)
            assert_almost_equal(dist2, 5.0 ** 0.5)
            
            # 计算Minkowski距离，p=0.25时的距离
            dist0p25 = minkowski(x, y, p=0.25)
            assert_almost_equal(dist0p25, (1.0 + 2.0 ** 0.25) ** 4)

        # 检查将输入转换为最小标量类型对结果的影响
        # （问题＃10262）。可以扩展到更多测试输入，使用np.min_scalar_type(np.max(input_matrix))。
        a = np.array([352, 916])
        b = np.array([350, 660])
        assert_equal(minkowski(a, b),
                     minkowski(a.astype('uint16'), b.astype('uint16')))

    def test_euclidean(self):
        # 对于self.cases中的每对测试用例(x, y)
        for x, y in self.cases:
            # 计算欧几里得距离
            dist = weuclidean(x, y)
            assert_almost_equal(dist, np.sqrt(5))

    def test_sqeuclidean(self):
        # 对于self.cases中的每对测试用例(x, y)
        for x, y in self.cases:
            # 计算平方欧几里得距离
            dist = wsqeuclidean(x, y)
            assert_almost_equal(dist, 5.0)

    def test_cosine(self):
        # 对于self.cases中的每对测试用例(x, y)
        for x, y in self.cases:
            # 计算余弦相似度
            dist = wcosine(x, y)
            assert_almost_equal(dist, 1.0 - 18.0 / (np.sqrt(14) * np.sqrt(27)))

    def test_cosine_output_dtype(self):
        # 回归测试gh-19541，检查余弦相似度函数返回类型
        assert isinstance(wcorrelation([1, 1], [1, 1], centered=False), float)
        assert isinstance(wcosine([1, 1], [1, 1]), float)

    def test_correlation(self):
        xm = np.array([-1.0, 0, 1.0])
        ym = np.array([-4.0 / 3, -4.0 / 3, 5.0 - 7.0 / 3])
        # 对于self.cases中的每对测试用例(x, y)
        for x, y in self.cases:
            # 计算相关性
            dist = wcorrelation(x, y)
            assert_almost_equal(dist, 1.0 - np.dot(xm, ym) / (norm(xm) * norm(ym)))

    def test_correlation_positive(self):
        # 回归测试gh-12320，检查相关性函数是否返回非负值
        x = np.array([0., 0., 0., 0., 0., 0., -2., 0., 0., 0., -2., -2., -2.,
                      0., -2., 0., -2., 0., 0., -1., -2., 0., 1., 0., 0., -2.,
                      0., 0., -2., 0., -2., -2., -2., -2., -2., -2., 0.])
        y = np.array([1., 1., 1., 1., 1., 1., -1., 1., 1., 1., -1., -1., -1.,
                      1., -1., 1., -1., 1., 1., 0., -1., 1., 2., 1., 1., -1.,
                      1., 1., -1., 1., -1., -1., -1., -1., -1., -1., 1.])
        dist = correlation(x, y)
        assert 0 <= dist <= 10 * np.finfo(np.float64).eps

    def test_mahalanobis(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 1.0, 5.0])
        vi = np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
        # 对于self.cases中的每对测试用例(x, y)
        for x, y in self.cases:
            # 计算马氏距离
            dist = mahalanobis(x, y, vi)
            assert_almost_equal(dist, np.sqrt(6.0))
class TestSquareForm:
    # 定义要检查的数据类型列表
    checked_dtypes = [np.float64, np.float32, np.int32, np.int8, bool]

    # 测试将矩阵转换为方阵的功能
    def test_squareform_matrix(self):
        for dtype in self.checked_dtypes:
            self.check_squareform_matrix(dtype)

    # 测试将向量转换为方阵的功能
    def test_squareform_vector(self):
        for dtype in self.checked_dtypes:
            self.check_squareform_vector(dtype)

    # 检查将矩阵转换为方阵的具体功能实现
    def check_squareform_matrix(self, dtype):
        # 创建零矩阵 A，并进行转换
        A = np.zeros((0, 0), dtype=dtype)
        rA = squareform(A)
        # 断言转换后的形状和数据类型
        assert_equal(rA.shape, (0,))
        assert_equal(rA.dtype, dtype)

        # 创建单元素矩阵 A，并进行转换
        A = np.zeros((1, 1), dtype=dtype)
        rA = squareform(A)
        # 断言转换后的形状和数据类型
        assert_equal(rA.shape, (0,))
        assert_equal(rA.dtype, dtype)

        # 创建具有实际值的矩阵 A，并进行转换
        A = np.array([[0, 4.2], [4.2, 0]], dtype=dtype)
        rA = squareform(A)
        # 断言转换后的形状、数据类型和值
        assert_equal(rA.shape, (1,))
        assert_equal(rA.dtype, dtype)
        assert_array_equal(rA, np.array([4.2], dtype=dtype))

    # 检查将向量转换为方阵的具体功能实现
    def check_squareform_vector(self, dtype):
        # 创建空向量 v，并进行转换
        v = np.zeros((0,), dtype=dtype)
        rv = squareform(v)
        # 断言转换后的形状、数据类型和值
        assert_equal(rv.shape, (1, 1))
        assert_equal(rv.dtype, dtype)
        assert_array_equal(rv, [[0]])

        # 创建单元素向量 v，并进行转换
        v = np.array([8.3], dtype=dtype)
        rv = squareform(v)
        # 断言转换后的形状、数据类型和值
        assert_equal(rv.shape, (2, 2))
        assert_equal(rv.dtype, dtype)
        assert_array_equal(rv, np.array([[0, 8.3], [8.3, 0]], dtype=dtype))

    # 测试多个矩阵的批量转换功能
    def test_squareform_multi_matrix(self):
        for n in range(2, 5):
            self.check_squareform_multi_matrix(n)

    # 检查多个矩阵的批量转换功能的具体实现
    def check_squareform_multi_matrix(self, n):
        # 创建随机矩阵 X，并计算其无常数项的加权距离
        X = np.random.rand(n, 4)
        Y = wpdist_no_const(X)
        # 断言转换后的形状和性质
        assert_equal(len(Y.shape), 1)
        A = squareform(Y)
        Yr = squareform(A)
        s = A.shape
        k = 0
        # 如果详细模式大于等于 3，则打印相关信息
        if verbose >= 3:
            print(A.shape, Y.shape, Yr.shape)
        # 断言转换后的形状和性质
        assert_equal(len(s), 2)
        assert_equal(len(Yr.shape), 1)
        assert_equal(s[0], s[1])
        # 遍历矩阵 A 的所有元素
        for i in range(0, s[0]):
            for j in range(i + 1, s[1]):
                # 断言每对非对角线元素的一致性
                if i != j:
                    assert_equal(A[i, j], Y[k])
                    k += 1
                else:
                    assert_equal(A[i, j], 0)


class TestNumObsY:

    # 测试在多个矩阵上计算观测数的功能
    def test_num_obs_y_multi_matrix(self):
        for n in range(2, 10):
            X = np.random.rand(n, 4)
            Y = wpdist_no_const(X)
            # 断言观测数计算的正确性
            assert_equal(num_obs_y(Y), n)

    # 在对 1 个观测的压缩距离矩阵进行测试，预期引发异常
    def test_num_obs_y_1(self):
        with pytest.raises(ValueError):
            self.check_y(1)

    # 在对 2 个观测的压缩距离矩阵进行测试
    def test_num_obs_y_2(self):
        assert_(self.check_y(2))

    # 在对 3 个观测的压缩距离矩阵进行测试
    def test_num_obs_y_3(self):
        assert_(self.check_y(3))

    # 在对 4 个观测的压缩距离矩阵进行测试
    def test_num_obs_y_4(self):
        assert_(self.check_y(4))

    # 在对 5 到 10 个观测的压缩距离矩阵进行测试
    def test_num_obs_y_5_10(self):
        for i in range(5, 16):
            self.minit(i)
    # 测试 num_obs_y(y) 在 100 个不正确的紧凑距离矩阵上的表现。
    # 预期会抛出异常。
    def test_num_obs_y_2_100(self):
        a = set()
        # 计算可能的紧凑距离矩阵的大小并添加到集合 a 中
        for n in range(2, 16):
            a.add(n * (n - 1) / 2)
        # 对于从 5 到 104 的每个数值进行迭代
        for i in range(5, 105):
            # 如果 i 不在集合 a 中
            if i not in a:
                # 预期会抛出 ValueError 异常
                with pytest.raises(ValueError):
                    # 调用 self.bad_y 方法处理异常
                    self.bad_y(i)

    # 断言检查给定的 n 是否满足 self.check_y 方法的条件
    def minit(self, n):
        assert_(self.check_y(n))

    # 生成一个大小为 n 的随机数组，并调用 num_obs_y 函数处理
    def bad_y(self, n):
        y = np.random.rand(n)
        return num_obs_y(y)

    # 检查给定的 n 是否满足 num_obs_y(self.make_y(n)) == n 的条件
    def check_y(self, n):
        return num_obs_y(self.make_y(n)) == n

    # 生成一个大小为 n 的随机数组，用于生成紧凑距离矩阵
    def make_y(self, n):
        return np.random.rand((n * (n - 1)) // 2)
class TestNumObsDM:

    def test_num_obs_dm_multi_matrix(self):
        # 对多个不同大小的矩阵进行测试
        for n in range(1, 10):
            # 创建一个随机 n x 4 的矩阵 X
            X = np.random.rand(n, 4)
            # 计算矩阵 X 的加权距离，返回距离矩阵 Y
            Y = wpdist_no_const(X)
            # 将 Y 转换成距离向量 A
            A = squareform(Y)
            # 如果 verbose 大于等于 3，则打印 A 和 Y 的形状信息
            if verbose >= 3:
                print(A.shape, Y.shape)
            # 断言 num_obs_dm(A) 等于 n
            assert_equal(num_obs_dm(A), n)

    def test_num_obs_dm_0(self):
        # 测试 num_obs_dm(D) 在一个 0x0 的距离矩阵上，预期会抛出异常
        assert_(self.check_D(0))

    def test_num_obs_dm_1(self):
        # 测试 num_obs_dm(D) 在一个 1x1 的距离矩阵上
        assert_(self.check_D(1))

    def test_num_obs_dm_2(self):
        # 测试 num_obs_dm(D) 在一个 2x2 的距离矩阵上
        assert_(self.check_D(2))

    def test_num_obs_dm_3(self):
        # 测试 num_obs_dm(D) 在一个 2x2 的距离矩阵上
        assert_(self.check_D(2))

    def test_num_obs_dm_4(self):
        # 测试 num_obs_dm(D) 在一个 4x4 的距离矩阵上
        assert_(self.check_D(4))

    def check_D(self, n):
        # 检查 num_obs_dm(D) 是否等于 n
        return num_obs_dm(self.make_D(n)) == n

    def make_D(self, n):
        # 创建一个随机的 n x n 距离矩阵
        return np.random.rand(n, n)


def is_valid_dm_throw(D):
    # 调用 is_valid_dm 函数，如果出现问题则抛出异常
    return is_valid_dm(D, throw=True)


class TestIsValidDM:

    def test_is_valid_dm_improper_shape_1D_E(self):
        # 测试在 1 维数组形状下，是否会抛出 ValueError 异常
        D = np.zeros((5,), dtype=np.float64)
        with pytest.raises(ValueError):
            is_valid_dm_throw(D)

    def test_is_valid_dm_improper_shape_1D_F(self):
        # 测试在 1 维数组形状下，是否返回 False
        D = np.zeros((5,), dtype=np.float64)
        assert_equal(is_valid_dm(D), False)

    def test_is_valid_dm_improper_shape_3D_E(self):
        # 测试在 3 维数组形状下，是否会抛出 ValueError 异常
        D = np.zeros((3, 3, 3), dtype=np.float64)
        with pytest.raises(ValueError):
            is_valid_dm_throw(D)

    def test_is_valid_dm_improper_shape_3D_F(self):
        # 测试在 3 维数组形状下，是否返回 False
        D = np.zeros((3, 3, 3), dtype=np.float64)
        assert_equal(is_valid_dm(D), False)

    def test_is_valid_dm_nonzero_diagonal_E(self):
        # 测试非零对角线元素是否会抛出 ValueError 异常
        y = np.random.rand(10)
        D = squareform(y)
        for i in range(0, 5):
            D[i, i] = 2.0
        with pytest.raises(ValueError):
            is_valid_dm_throw(D)

    def test_is_valid_dm_nonzero_diagonal_F(self):
        # 测试非零对角线元素是否返回 False
        y = np.random.rand(10)
        D = squareform(y)
        for i in range(0, 5):
            D[i, i] = 2.0
        assert_equal(is_valid_dm(D), False)

    def test_is_valid_dm_asymmetric_E(self):
        # 测试非对称矩阵是否会抛出 ValueError 异常
        y = np.random.rand(10)
        D = squareform(y)
        D[1, 3] = D[3, 1] + 1
        with pytest.raises(ValueError):
            is_valid_dm_throw(D)

    def test_is_valid_dm_asymmetric_F(self):
        # 测试非对称矩阵是否返回 False
        y = np.random.rand(10)
        D = squareform(y)
        D[1, 3] = D[3, 1] + 1
        assert_equal(is_valid_dm(D), False)

    def test_is_valid_dm_correct_1_by_1(self):
        # 测试 1x1 的距离矩阵是否返回 True
        D = np.zeros((1, 1), dtype=np.float64)
        assert_equal(is_valid_dm(D), True)

    def test_is_valid_dm_correct_2_by_2(self):
        # 测试 2x2 的距离矩阵是否返回 True
        y = np.random.rand(1)
        D = squareform(y)
        assert_equal(is_valid_dm(D), True)

    def test_is_valid_dm_correct_3_by_3(self):
        # 测试 3x3 的距离矩阵是否返回 True
        y = np.random.rand(3)
        D = squareform(y)
        assert_equal(is_valid_dm(D), True)
    # 定义一个测试函数，用于测试 is_valid_dm 函数对于 4x4 距离矩阵的正确性
    def test_is_valid_dm_correct_4_by_4(self):
        # 生成一个包含6个随机数的数组
        y = np.random.rand(6)
        # 将数组转换为距离矩阵形式
        D = squareform(y)
        # 断言调用 is_valid_dm 函数后返回 True
        assert_equal(is_valid_dm(D), True)

    # 定义另一个测试函数，用于测试 is_valid_dm 函数对于 5x5 距离矩阵的正确性
    def test_is_valid_dm_correct_5_by_5(self):
        # 生成一个包含10个随机数的数组
        y = np.random.rand(10)
        # 将数组转换为距离矩阵形式
        D = squareform(y)
        # 断言调用 is_valid_dm 函数后返回 True
        assert_equal(is_valid_dm(D), True)
class TestIsValidY:
    # Test class for validating the function is_valid_y.
    
    def test_is_valid_y_improper_shape_2D_E(self):
        # Test for improper 2D shape that expects a ValueError.
        y = np.zeros((3, 3,), dtype=np.float64)
        with pytest.raises(ValueError):
            is_valid_y_throw(y)

    def test_is_valid_y_improper_shape_2D_F(self):
        # Test for improper 2D shape expecting False from is_valid_y.
        y = np.zeros((3, 3,), dtype=np.float64)
        assert_equal(is_valid_y(y), False)

    def test_is_valid_y_improper_shape_3D_E(self):
        # Test for improper 3D shape that expects a ValueError.
        y = np.zeros((3, 3, 3), dtype=np.float64)
        with pytest.raises(ValueError):
            is_valid_y_throw(y)

    def test_is_valid_y_improper_shape_3D_F(self):
        # Test for improper 3D shape expecting False from is_valid_y.
        y = np.zeros((3, 3, 3), dtype=np.float64)
        assert_equal(is_valid_y(y), False)

    def test_is_valid_y_correct_2_by_2(self):
        # Test for correct 2x2 shape using random data.
        y = self.correct_n_by_n(2)
        assert_equal(is_valid_y(y), True)

    def test_is_valid_y_correct_3_by_3(self):
        # Test for correct 3x3 shape using random data.
        y = self.correct_n_by_n(3)
        assert_equal(is_valid_y(y), True)

    def test_is_valid_y_correct_4_by_4(self):
        # Test for correct 4x4 shape using random data.
        y = self.correct_n_by_n(4)
        assert_equal(is_valid_y(y), True)

    def test_is_valid_y_correct_5_by_5(self):
        # Test for correct 5x5 shape using random data.
        y = self.correct_n_by_n(5)
        assert_equal(is_valid_y(y), True)

    def test_is_valid_y_2_100(self):
        # Test various invalid y shapes using random data.
        a = set()
        for n in range(2, 16):
            a.add(n * (n - 1) / 2)
        for i in range(5, 105):
            if i not in a:
                with pytest.raises(ValueError):
                    self.bad_y(i)

    def bad_y(self, n):
        # Generate a random array of size n and validate with is_valid_y_throw.
        y = np.random.rand(n)
        return is_valid_y(y, throw=True)

    def correct_n_by_n(self, n):
        # Generate a random array of size (n * (n - 1)) // 2 for valid y shape.
        y = np.random.rand((n * (n - 1)) // 2)
        return y
    # 使用 assert_almost_equal 函数检查 weuclidean 函数计算的欧几里得距离是否接近平方根的3，精度为小数点后14位。
    assert_almost_equal(weuclidean(x1, x2), np.sqrt(3), decimal=14)
    
    # 检查输入向量是否为(1, N)或(N, 1)，若不是则引发 ValueError 异常，匹配错误信息"Input vector should be 1-D"。
    with pytest.raises(ValueError, match="Input vector should be 1-D"):
        weuclidean(x1[np.newaxis, :], x2[np.newaxis, :])
        # np.sqrt(3)  # 这里原本有一个注释的错误，应删除
    with pytest.raises(ValueError, match="Input vector should be 1-D"):
        wsqeuclidean(x1[np.newaxis, :], x2[np.newaxis, :])
    with pytest.raises(ValueError, match="Input vector should be 1-D"):
        wsqeuclidean(x1[:, np.newaxis], x2[:, np.newaxis])
    
    # 距离度量仅对向量（即1-D数组）定义，对于非1-D数组输入，引发 ValueError 异常。
    x = np.arange(4).reshape(2, 2)
    with pytest.raises(ValueError):
        weuclidean(x, x)
    with pytest.raises(ValueError):
        wsqeuclidean(x, x)
    
    # 使用随机数据进行另一轮检查。
    rs = np.random.RandomState(1234567890)
    x = rs.rand(10)
    y = rs.rand(10)
    # 计算 x 和 y 之间的欧几里得距离 d1 和加权欧几里得距离 d2。
    d1 = weuclidean(x, y)
    d2 = wsqeuclidean(x, y)
    # 断言 d1 的平方近似等于 d2，精度为小数点后14位。
    assert_almost_equal(d1**2, d2, decimal=14)
# 测试不同长度的Hamming距离计算函数
def test_hamming_unequal_length():
    # 用于回归测试gh-4290。
    x = [0, 0, 1]  # 第一个列表
    y = [1, 0, 1, 0]  # 第二个列表，长度不同
    # 检查是否会因为在布尔类型上调用ndarray.mean而导致AttributeError
    with pytest.raises(ValueError):
        whamming(x, y)


# 测试带权重的不同长度的Hamming距离计算函数
def test_hamming_unequal_length_with_w():
    u = [0, 0, 1]  # 第一个列表
    v = [0, 0, 1]  # 第二个列表，长度相同
    w = [1, 0, 1, 0]  # 权重列表，与u和v长度不同
    msg = "'w' should have the same length as 'u' and 'v'."
    # 检查是否会因为权重长度与u、v不同而引发ValueError，并匹配给定消息
    with pytest.raises(ValueError, match=msg):
        whamming(u, v, w)


# 测试字符串数组的Hamming距离计算函数
def test_hamming_string_array():
    # https://github.com/scikit-learn/scikit-learn/issues/4014
    a = np.array(['eggs', 'spam', 'spam', 'eggs', 'spam', 'spam', 'spam',
                  'spam', 'spam', 'spam', 'spam', 'eggs', 'eggs', 'spam',
                  'eggs', 'eggs', 'eggs', 'eggs', 'eggs', 'spam'],
                  dtype='|S4')  # 第一个字符串数组
    b = np.array(['eggs', 'spam', 'spam', 'eggs', 'eggs', 'spam', 'spam',
                  'spam', 'spam', 'eggs', 'spam', 'eggs', 'spam', 'eggs',
                  'spam', 'spam', 'eggs', 'spam', 'spam', 'eggs'],
                  dtype='|S4')  # 第二个字符串数组
    desired = 0.45  # 期望的Hamming距离
    # 检查计算出的Hamming距离是否接近期望值
    assert_allclose(whamming(a, b), desired)


# 测试Minkowski距离函数
def test_minkowski_w():
    # Regression test for gh-8142.
    arr_in = np.array([[83.33333333, 100., 83.33333333, 100., 36.,
                        60., 90., 150., 24., 48.],
                       [83.33333333, 100., 83.33333333, 100., 36.,
                        60., 90., 150., 24., 48.]])
    p0 = pdist(arr_in, metric='minkowski', p=1, w=None)  # 计算Minkowski距离，无权重
    c0 = cdist(arr_in, arr_in, metric='minkowski', p=1, w=None)  # 计算Minkowski距离，无权重
    p1 = pdist(arr_in, metric='minkowski', p=1)  # 使用默认参数计算Minkowski距离
    c1 = cdist(arr_in, arr_in, metric='minkowski', p=1)  # 使用默认参数计算Minkowski距离

    # 检查两种不同参数配置下计算出的距离是否接近
    assert_allclose(p0, p1, rtol=1e-15)
    assert_allclose(c0, c1, rtol=1e-15)


# 测试sqeuclidean距离函数返回类型
def test_sqeuclidean_dtypes():
    # 断言sqeuclidean函数返回正确的数值类型。
    # 整数类型应转换为浮点数以保持稳定性。
    # 浮点类型应与输入保持一致。
    x = [1, 2, 3]  # 第一个列表
    y = [4, 5, 6]  # 第二个列表

    # 检查不同整数类型的输入是否会返回浮点类型的距离
    for dtype in [np.int8, np.int16, np.int32, np.int64]:
        d = wsqeuclidean(np.asarray(x, dtype=dtype), np.asarray(y, dtype=dtype))
        assert_(np.issubdtype(d.dtype, np.floating))

    # 检查不同无符号整数类型的输入是否会返回相同的浮点类型的距离，并且距离的值是否正确
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        umax = np.iinfo(dtype).max
        d1 = wsqeuclidean([0], np.asarray([umax], dtype=dtype))
        d2 = wsqeuclidean(np.asarray([umax], dtype=dtype), [0])

        assert_equal(d1, d2)
        assert_equal(d1, np.float64(umax)**2)

    dtypes = [np.float32, np.float64, np.complex64, np.complex128]
    for dtype in ['float16', 'float128']:
        # 这些类型在较旧的numpy版本中可能不存在；在所有平台上也可能没有float128。
        if hasattr(np, dtype):
            dtypes.append(getattr(np, dtype))

    # 检查不同数据类型的输入是否会返回相同数据类型的距离
    for dtype in dtypes:
        d = wsqeuclidean(np.asarray(x, dtype=dtype), np.asarray(y, dtype=dtype))
        assert_equal(d.dtype, dtype)
    # 创建包含布尔值的列表 p 和 q，用于测试 sokalmichener 函数对布尔值和整数输入的相同性能
    p = [True, True, False]
    q = [True, False, True]
    
    # 将布尔列表转换为整数列表，其中 True 转换为 1，False 转换为 0
    x = [int(b) for b in p]
    y = [int(b) for b in q]
    
    # 使用 sokalmichener 函数计算布尔列表 p 和 q 的相似度距离
    dist1 = sokalmichener(p, q)
    
    # 使用 sokalmichener 函数计算整数列表 x 和 y 的相似度距离
    dist2 = sokalmichener(x, y)
    
    # 断言两个距离值应该完全相同
    assert_equal(dist1, dist2)
def test_sokalmichener_with_weight():
    # 初始化四个变量，表示不同条件下的计数
    ntf = 0 * 1 + 0 * 0.2  # ntf 的计算
    nft = 0 * 1 + 1 * 0.2  # nft 的计算
    ntt = 1 * 1 + 0 * 0.2  # ntt 的计算
    nff = 0 * 1 + 0 * 0.2  # nff 的计算
    # 计算预期的值
    expected = 2 * (nft + ntf) / (ntt + nff + 2 * (nft + ntf))
    # 使用 assert_almost_equal 函数断言期望值与指定值接近
    assert_almost_equal(expected, 0.2857143)
    # 调用 sokalmichener 函数，断言期望值与实际返回值接近
    actual = sokalmichener([1, 0], [1, 1], w=[1, 0.2])
    assert_almost_equal(expected, actual)

    # 定义两个布尔数组 a1 和 a2
    a1 = [False, False, True, True, True, False, False, True, True, True, True,
          True, True, False, True, False, False, False, True, True]
    a2 = [True, True, True, False, False, True, True, True, False, True,
          True, True, True, True, False, False, False, True, True, True]

    # 遍历不同的权重值进行断言
    for w in [0.05, 0.1, 1.0, 20.0]:
        assert_almost_equal(sokalmichener(a2, a1, [w]), 0.6666666666666666)


def test_modifies_input(metric):
    # 测试 cdist 和 pdist 是否修改输入数组
    X1 = np.asarray([[1., 2., 3.],
                     [1.2, 2.3, 3.4],
                     [2.2, 2.3, 4.4],
                     [22.2, 23.3, 44.4]])
    X1_copy = X1.copy()  # 复制 X1 数组
    cdist(X1, X1, metric)  # 调用 cdist 函数
    pdist(X1, metric)  # 调用 pdist 函数
    # 使用 assert_array_equal 函数断言 X1 数组未被修改
    assert_array_equal(X1, X1_copy)


def test_Xdist_deprecated_args(metric):
    # 测试 cdist 和 pdist 中已弃用参数的警告
    X1 = np.asarray([[1., 2., 3.],
                     [1.2, 2.3, 3.4],
                     [2.2, 2.3, 4.4],
                     [22.2, 23.3, 44.4]])

    # 断言在传递已弃用参数时抛出 TypeError 异常
    with pytest.raises(TypeError):
        cdist(X1, X1, metric, 2.)

    with pytest.raises(TypeError):
        pdist(X1, metric, 2.)

    # 遍历不同的参数值进行断言
    for arg in ["p", "V", "VI"]:
        kwargs = {arg: "foo"}

        # 跳过不适用于当前 metric 的参数组合
        if ((arg == "V" and metric == "seuclidean")
                or (arg == "VI" and metric == "mahalanobis")
                or (arg == "p" and metric == "minkowski")):
            continue

        # 断言在传递参数组合时抛出 TypeError 异常
        with pytest.raises(TypeError):
            cdist(X1, X1, metric, **kwargs)

        with pytest.raises(TypeError):
            pdist(X1, metric, **kwargs)


def test_Xdist_non_negative_weights(metric):
    X = eo['random-float32-data'][::5, ::2]
    w = np.ones(X.shape[1])
    w[::5] = -w[::5]

    # 如果 metric 是 ['seuclidean', 'mahalanobis', 'jensenshannon'] 中的一种，跳过测试
    if metric in ['seuclidean', 'mahalanobis', 'jensenshannon']:
        pytest.skip("not applicable")

    # 断言在传递非负权重时抛出 ValueError 异常
    for m in [metric, eval(metric), "test_" + metric]:
        with pytest.raises(ValueError):
            pdist(X, m, w=w)
        with pytest.raises(ValueError):
            cdist(X, X, m, w=w)


def test__validate_vector():
    x = [1, 2, 3]
    y = _validate_vector(x)
    assert_array_equal(y, x)  # 断言返回的向量与输入向量相等

    y = _validate_vector(x, dtype=np.float64)
    assert_array_equal(y, x)  # 断言返回的向量与输入向量相等
    assert_equal(y.dtype, np.float64)  # 断言返回的向量类型为 np.float64

    x = [1]
    y = _validate_vector(x)
    assert_equal(y.ndim, 1)  # 断言返回的向量是一维的
    assert_equal(y, x)  # 断言返回的向量与输入向量相等

    x = 1
    # 断言传递标量时抛出 ValueError 异常
    with pytest.raises(ValueError, match="Input vector should be 1-D"):
        _validate_vector(x)

    x = np.arange(5).reshape(1, -1, 1)
    # 使用 pytest 库中的 pytest.raises 上下文管理器，期望捕获 ValueError 异常，并验证异常消息是否包含指定的字符串 "Input vector should be 1-D"
    with pytest.raises(ValueError, match="Input vector should be 1-D"):
        # 调用 _validate_vector 函数，传入参数 x 进行验证
        _validate_vector(x)
    
    # 将变量 x 赋值为一个二维列表 [[1, 2], [3, 4]]
    x = [[1, 2], [3, 4]]
    # 使用 pytest 库中的 pytest.raises 上下文管理器，期望捕获 ValueError 异常，并验证异常消息是否包含指定的字符串 "Input vector should be 1-D"
    with pytest.raises(ValueError, match="Input vector should be 1-D"):
        # 再次调用 _validate_vector 函数，传入参数 x 进行验证
        _validate_vector(x)
# 测试函数，用于验证 Yule 指标在全部相同情况下的行为
def test_yule_all_same():
    # 创建一个2行6列的布尔类型数组，所有元素均为True
    x = np.ones((2, 6), dtype=bool)
    # 使用 wyule 函数计算相同行向量之间的 Yule 距离
    d = wyule(x[0], x[0])
    # 断言距离应该为0.0
    assert d == 0.0

    # 使用 pdist 函数计算数组 x 中的 Yule 距离，并断言结果应为一个包含单个元素的列表，值为0.0
    d = pdist(x, 'yule')
    assert_equal(d, [0.0])

    # 使用 cdist 函数计算数组 x[:1] 和 x[:1] 之间的 Yule 距离，并断言结果应为一个包含单个元素的二维数组，值为 [[0.0]]
    d = cdist(x[:1], x[:1], 'yule')
    assert_equal(d, [[0.0]])


# 测试函数，验证 Jensen-Shannon 距离的计算
def test_jensenshannon():
    # 断言 Jensen-Shannon 距离函数对给定输入的计算结果，应该接近于预期值
    assert_almost_equal(jensenshannon([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 2.0),
                        1.0)
    assert_almost_equal(jensenshannon([1.0, 0.0], [0.5, 0.5]),
                        0.46450140402245893)
    assert_almost_equal(jensenshannon([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]), 0.0)

    # 断言 Jensen-Shannon 距离函数对矩阵输入的计算结果，应该接近于预期值
    assert_almost_equal(jensenshannon([[1.0, 2.0]], [[0.5, 1.5]], axis=0),
                        [0.0, 0.0])
    assert_almost_equal(jensenshannon([[1.0, 2.0]], [[0.5, 1.5]], axis=1),
                        [0.0649045])
    assert_almost_equal(jensenshannon([[1.0, 2.0]], [[0.5, 1.5]], axis=0,
                                      keepdims=True), [[0.0, 0.0]])
    assert_almost_equal(jensenshannon([[1.0, 2.0]], [[0.5, 1.5]], axis=1,
                                      keepdims=True), [[0.0649045]])

    # 创建两个矩阵 a 和 b，用于验证 Jensen-Shannon 距离在矩阵输入下的计算结果
    a = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])
    b = np.array([[13, 14, 15, 16],
                  [17, 18, 19, 20],
                  [21, 22, 23, 24]])

    # 断言 Jensen-Shannon 距离函数对矩阵输入的计算结果，应该接近于预期值
    assert_almost_equal(jensenshannon(a, b, axis=0),
                        [0.1954288, 0.1447697, 0.1138377, 0.0927636])
    assert_almost_equal(jensenshannon(a, b, axis=1),
                        [0.1402339, 0.0399106, 0.0201815])


# 测试函数，验证 GitHub Issue #17703 的修复情况
def test_gh_17703():
    # 创建两个数组，用于验证 Dice 距离在输入数组下的计算结果
    arr_1 = np.array([1, 0, 0])
    arr_2 = np.array([2, 0, 0])
    expected = dice(arr_1, arr_2)
    # 使用 pdist 函数计算输入数组的 Dice 距离，并断言结果应与预期值接近
    actual = pdist([arr_1, arr_2], metric='dice')
    assert_allclose(actual, expected)
    # 使用 cdist 函数计算输入数组的 Dice 距离，并断言结果应与预期值接近
    actual = cdist(np.atleast_2d(arr_1),
                   np.atleast_2d(arr_2), metric='dice')
    assert_allclose(actual, expected)


# 测试函数，验证在特定度量标准下不可变输入的行为
def test_immutable_input(metric):
    # 如果度量标准为 "jensenshannon", "mahalanobis", "seuclidean" 中的一种，则跳过测试
    if metric in ("jensenshannon", "mahalanobis", "seuclidean"):
        pytest.skip("not applicable")
    # 创建一个包含10个浮点数的数组，并设置为不可写
    x = np.arange(10, dtype=np.float64)
    x.setflags(write=False)
    # 调用 scipy.spatial.distance 模块中的指定度量标准函数，验证不可写输入的行为
    getattr(scipy.spatial.distance, metric)(x, x, w=x)
```