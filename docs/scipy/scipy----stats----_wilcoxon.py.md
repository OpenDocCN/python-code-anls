# `D:\src\scipysrc\scipy\scipy\stats\_wilcoxon.py`

```
import warnings  # 导入警告模块，用于处理警告信息
import numpy as np  # 导入NumPy库，用于数值计算

from scipy import stats  # 导入SciPy库中的统计模块
from ._stats_py import _get_pvalue, _rankdata, _SimpleNormal  # 导入自定义的统计相关函数和类
from . import _morestats  # 导入更多统计函数
from ._axis_nan_policy import _broadcast_arrays  # 导入数组广播相关函数
from ._hypotests import _get_wilcoxon_distr  # 导入威尔科克森秩和检验分布计算函数
from scipy._lib._util import _lazywhere, _get_nan  # 导入延迟计算函数和获取NaN值函数


class WilcoxonDistribution:
    
    def __init__(self, n):
        n = np.asarray(n).astype(int, copy=False)  # 将输入的n转换为NumPy数组，确保其为整数类型
        self.n = n  # 将处理后的n存储在实例变量中
        self._dists = {ni: _get_wilcoxon_distr(ni) for ni in np.unique(n)}  # 使用不同的n值计算对应的威尔科克森秩和检验分布

    def _cdf1(self, k, n):
        pmfs = self._dists[n]  # 获取对应n值的威尔科克森秩和检验分布
        return pmfs[:k + 1].sum()  # 计算累积分布函数中k以下秩和检验的概率和

    def _cdf(self, k, n):
        return np.vectorize(self._cdf1, otypes=[float])(k, n)  # 向量化计算累积分布函数

    def _sf1(self, k, n):
        pmfs = self._dists[n]  # 获取对应n值的威尔科克森秩和检验分布
        return pmfs[k:].sum()  # 计算生存函数中k以上秩和检验的概率和

    def _sf(self, k, n):
        return np.vectorize(self._sf1, otypes=[float])(k, n)  # 向量化计算生存函数

    def mean(self):
        return self.n * (self.n + 1) / 4  # 计算威尔科克森分布的均值

    def _prep(self, k):
        k = np.asarray(k).astype(int, copy=False)  # 将输入的k转换为NumPy数组，确保其为整数类型
        mn = self.mean()  # 计算威尔科克森分布的均值
        out = np.empty(k.shape, dtype=np.float64)  # 创建一个空的浮点数数组，用于存储结果
        return k, mn, out

    def cdf(self, k):
        k, mn, out = self._prep(k)  # 准备计算累积分布函数所需的参数
        return _lazywhere(k <= mn, (k, self.n), self._cdf,
                          f2=lambda k, n: 1 - self._sf(k+1, n))[()]  # 使用延迟计算根据条件选择累积分布函数或生存函数

    def sf(self, k):
        k, mn, out = self._prep(k)  # 准备计算生存函数所需的参数
        return _lazywhere(k <= mn, (k, self.n), self._sf,
                          f2=lambda k, n: 1 - self._cdf(k-1, n))[()]  # 使用延迟计算根据条件选择生存函数或累积分布函数


def _wilcoxon_iv(x, y, zero_method, correction, alternative, method, axis):

    axis = np.asarray(axis)[()]  # 将输入的轴参数转换为NumPy数组，并获取其标量值
    message = "`axis` must be an integer."  # 定义错误信息字符串
    if not np.issubdtype(axis.dtype, np.integer) or axis.ndim != 0:  # 检查轴参数是否为整数类型且维度为0
        raise ValueError(message)  # 若不满足条件，抛出值错误异常

    message = '`axis` must be compatible with the shape(s) of `x` (and `y`)'  # 更新错误信息字符串
    try:
        if y is None:
            x = np.asarray(x)  # 将输入的x转换为NumPy数组
            d = x  # 将x赋值给d
        else:
            x, y = _broadcast_arrays((x, y), axis=axis)  # 广播输入的x和y数组，以适应给定的轴
            d = x - y  # 计算x和y的差值
        d = np.moveaxis(d, axis, -1)  # 将轴移动到最后一个位置
    except np.AxisError as e:
        raise ValueError(message) from e  # 捕获轴错误并抛出值错误异常

    message = "`x` and `y` must have the same length along `axis`."  # 更新错误信息字符串
    if y is not None and x.shape[axis] != y.shape[axis]:  # 检查x和y是否在给定轴上具有相同的长度
        raise ValueError(message)  # 若不满足条件，抛出值错误异常

    message = "`x` (and `y`, if provided) must be an array of real numbers."  # 更新错误信息字符串
    if np.issubdtype(d.dtype, np.integer):  # 检查差值数组是否为整数类型
        d = d.astype(np.float64)  # 将差值数组转换为浮点数类型
    if not np.issubdtype(d.dtype, np.floating):  # 检查差值数组是否为浮点数类型
        raise ValueError(message)  # 若不满足条件，抛出值错误异常

    zero_method = str(zero_method).lower()  # 将零值处理方法转换为小写字符串
    zero_methods = {"wilcox", "pratt", "zsplit"}  # 支持的零值处理方法集合
    message = f"`zero_method` must be one of {zero_methods}."  # 更新错误信息字符串
    if zero_method not in zero_methods:  # 检查零值处理方法是否在支持的集合中
        raise ValueError(message)  # 若不满足条件，抛出值错误异常

    corrections = {True, False}  # 可用的校正方法集合
    message = f"`correction` must be one of {corrections}."  # 更新错误信息字符串
    if correction not in corrections:  # 检查校正方法是否在可用的集合中
        raise ValueError(message)  # 若不满足条件，抛出值错误异常

    alternative = str(alternative).lower()  # 将备择假设转换为小写字符串
    alternatives = {"two-sided", "less", "greater"}  # 支持的备择假设集合
    message = f"`alternative` must be one of {alternatives}."  # 更新错误信息字符串
    if alternative not in alternatives:  # 检查备择假设是否在支持的集合中
        raise ValueError(message)  # 若不满足条件，抛出值错误异常
    # 构建错误信息，指示 `alternative` 必须是给定列表中的一个
    message = f"`alternative` must be one of {alternatives}."
    # 检查 `alternative` 是否在给定的替代列表中，如果不在则抛出值错误异常
    if alternative not in alternatives:
        raise ValueError(message)

    # 检查 `method` 是否为 `stats.PermutationMethod` 的一个实例，如果不是则设定有效的方法列表
    if not isinstance(method, stats.PermutationMethod):
        methods = {"auto", "approx", "exact"}
        # 构建错误信息，指示 `method` 必须是给定列表中的一个，或者是 `stats.PermutationMethod` 的实例
        message = (f"`method` must be one of {methods} or "
                   "an instance of `stats.PermutationMethod`.")
        # 如果 `method` 不在有效的方法列表中则抛出值错误异常
        if method not in methods:
            raise ValueError(message)
    # 根据 `method` 的取值设定 `output_z` 为 True 或 False
    output_z = True if method == 'approx' else False

    # 以下代码段保持不变，用于向后兼容逻辑
    # 统计数组 `d` 中每行中为零的元素数量
    n_zero = np.sum(d == 0, axis=-1)
    # 检查是否存在任何一行中有为零的元素
    has_zeros = np.any(n_zero > 0)
    # 自动确定 `method` 的选择，若 `d` 的最后一个维度小于等于50且没有零元素，则选择 "exact" 方法，否则选择 "approx" 方法
    if method == "auto":
        if d.shape[-1] <= 50 and not has_zeros:
            method = "exact"
        else:
            method = "approx"

    # 统计数组 `d` 中为零的元素总数
    n_zero = np.sum(d == 0)
    # 如果 `d` 中有零元素且 `method` 为 "exact"，则切换至 "approx" 方法，并发出警告
    if n_zero > 0 and method == "exact":
        method = "approx"
        warnings.warn("Exact p-value calculation does not work if there are "
                      "zeros. Switching to normal approximation.",
                      stacklevel=2)

    # 若 `method` 为 "approx"，且 `zero_method` 在 ["wilcox", "pratt"] 中，并且 `d` 全为零并且不为空且为一维数组，则抛出值错误异常
    if (method == "approx" and zero_method in ["wilcox", "pratt"]
            and n_zero == d.size and d.size > 0 and d.ndim == 1):
        raise ValueError("zero_method 'wilcox' and 'pratt' do not "
                         "work if x - y is zero for all elements.")

    # 如果 `d` 的最后一个维度在 (0, 10) 范围内且 `method` 为 "approx"，则发出警告，提示样本量太小无法进行正态近似
    if 0 < d.shape[-1] < 10 and method == "approx":
        warnings.warn("Sample size too small for normal approximation.", stacklevel=2)

    # 返回处理后的结果
    return d, zero_method, correction, alternative, method, axis, output_z
def _wilcoxon_statistic(d, zero_method='wilcox'):
    # 计算所有值为零的索引
    i_zeros = (d == 0)

    if zero_method == 'wilcox':
        # 如果使用 Wilcoxon 方法处理零值，则将零值替换为 NaN，
        # 因为 NaN 在计算中会被忽略
        if not d.flags['WRITEABLE']:
            d = d.copy()
        d[i_zeros] = np.nan

    # 找出所有 NaN 值的索引
    i_nan = np.isnan(d)
    # 计算每行中 NaN 值的数量
    n_nan = np.sum(i_nan, axis=-1)
    # 计算有效值的数量
    count = d.shape[-1] - n_nan

    # 计算排名和并返回 tie 信息
    r, t = _rankdata(abs(d), 'average', return_ties=True)

    # 计算正向排名和负向排名的和
    r_plus = np.sum((d > 0) * r, axis=-1)
    r_minus = np.sum((d < 0) * r, axis=-1)

    if zero_method == "zsplit":
        # 如果使用 "zero-split" 方法处理零值，则将零值的贡献平均分给正向和负向排名
        r_zero_2 = np.sum(i_zeros * r, axis=-1) / 2
        r_plus += r_zero_2
        r_minus += r_zero_2

    # 计算 mn 和 se
    mn = count * (count + 1.) * 0.25
    se = count * (count + 1.) * (2. * count + 1.)

    if zero_method == "pratt":
        # 如果使用 Pratt 方法处理零值，则修改 mn 和 se 的值
        n_zero = i_zeros.sum(axis=-1)
        mn -= n_zero * (n_zero + 1.) * 0.25
        se -= n_zero * (n_zero + 1.) * (2. * n_zero + 1.)

        # 零值对于 tie 修正不应包括在内
        t[i_zeros.any(axis=-1), 0] = 0

    # 计算 tie 修正
    tie_correct = (t**3 - t).sum(axis=-1)
    se -= tie_correct / 2
    se = np.sqrt(se / 24)

    # 计算 z 统计量
    z = (r_plus - mn) / se

    return r_plus, r_minus, se, z, count


def _correction_sign(z, alternative):
    # 根据 alternative 返回 z 统计量的正负符号修正值
    if alternative == 'greater':
        return 1
    elif alternative == 'less':
        return -1
    else:
        return np.sign(z)


def _wilcoxon_nd(x, y=None, zero_method='wilcox', correction=True,
                 alternative='two-sided', method='auto', axis=0):
    # 调用 _wilcoxon_iv 函数，获取 d 等相关变量
    temp = _wilcoxon_iv(x, y, zero_method, correction, alternative, method, axis)
    d, zero_method, correction, alternative, method, axis, output_z = temp

    if d.size == 0:
        # 如果 d 为空，则返回 NaN 的统计值和 p 值
        NaN = _get_nan(d)
        res = _morestats.WilcoxonResult(statistic=NaN, pvalue=NaN)
        if method == 'approx':
            res.zstatistic = NaN
        return res

    # 计算 r_plus, r_minus, se, z 和 count
    r_plus, r_minus, se, z, count = _wilcoxon_statistic(d, zero_method)

    if method == 'approx':
        # 如果 method 是 'approx'，则进行修正
        if correction:
            sign = _correction_sign(z, alternative)
            z -= sign * 0.5 / se
        # 计算 p 值
        p = _get_pvalue(z, _SimpleNormal(), alternative, xp=np)
    # 如果使用 'exact' 方法进行统计检验
    elif method == 'exact':
        # 根据观测数据生成 Wilcoxon 分布
        dist = WilcoxonDistribution(count)
        
        # 如果零假设分布（null distribution）只在没有并列数据或零的情况下是精确的
        # 如果存在并列数据或零，统计量可能是非整数的，但是零假设分布只对整数统计量有定义
        # 因此，我们采取保守的方式：在计算累积分布函数（CDF）之前向上取整非整数统计量，
        # 在计算生存函数（SF）之前向下取整。这样保留了相对于备择假设的对称性和输入参数的顺序。
        # 参见 gh-19872。
        
        if alternative == 'less':
            # 如果备择假设是小于（左尾检验），则使用向上取整后的统计量计算累积分布函数
            p = dist.cdf(np.ceil(r_plus))
        elif alternative == 'greater':
            # 如果备择假设是大于（右尾检验），则使用向下取整后的统计量计算生存函数
            p = dist.sf(np.floor(r_plus))
        else:
            # 如果备择假设是双侧检验，则同时计算向上取整和向下取整后的统计量对应的生存函数和累积分布函数
            p = 2 * np.minimum(dist.sf(np.floor(r_plus)),
                               dist.cdf(np.ceil(r_plus)))
            # 将计算结果夹在 [0, 1] 的范围内
            p = np.clip(p, 0, 1)
    
    else:  # `PermutationMethod` 实例（已验证）
        # 对已验证的排列方法实例执行排列检验，使用 `stats.permutation_test` 函数
        p = stats.permutation_test(
            (d,), lambda d: _wilcoxon_statistic(d, zero_method)[0],
            permutation_type='samples', **method._asdict(),
            alternative=alternative, axis=-1).pvalue
    
    # 为了向后兼容性...
    # 根据备择假设类型确定最终的统计量，如果是双侧检验则选择较小的 r_plus 和 r_minus，否则选择 r_plus
    statistic = np.minimum(r_plus, r_minus) if alternative=='two-sided' else r_plus
    
    # 如果备择假设是双侧检验并且方法是 'approx'，则将 z 的绝对值取反
    z = -np.abs(z) if (alternative == 'two-sided' and method == 'approx') else z
    
    # 构造 Wilcoxon 检验的结果对象
    res = _morestats.WilcoxonResult(statistic=statistic, pvalue=p[()])
    
    # 如果需要输出 z 统计量，则将 z 统计量保存到结果对象中
    if output_z:
        res.zstatistic = z[()]
    
    # 返回最终的结果对象
    return res
```