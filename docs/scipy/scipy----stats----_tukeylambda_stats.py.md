# `D:\src\scipysrc\scipy\scipy\stats\_tukeylambda_stats.py`

```
# 导入必要的库
import numpy as np
from numpy import poly1d
from scipy.special import beta

# 以下代码用于生成 Tukey Lambda 方差函数的 Pade 系数。
#---------------------------------------------------------------------------
# import mpmath as mp
#
# mp.mp.dps = 60
#
# one   = mp.mpf(1)
# two   = mp.mpf(2)
#
# def mpvar(lam):
#     if lam == 0:
#         v = mp.pi**2 / three
#     else:
#         v = (two / lam**2) * (one / (one + two*lam) -
#                               mp.beta(lam + one, lam + one))
#     return v
#
# t = mp.taylor(mpvar, 0, 8)
# p, q = mp.pade(t, 4, 4)
# print("p =", [mp.fp.mpf(c) for c in p])
# print("q =", [mp.fp.mpf(c) for c in q])
#---------------------------------------------------------------------------

# Tukey Lambda 方差函数的 Pade 近似的分子部分系数
_tukeylambda_var_pc = [3.289868133696453, 0.7306125098871127,
                       -0.5370742306855439, 0.17292046290190008,
                       -0.02371146284628187]
# Tukey Lambda 方差函数的 Pade 近似的分母部分系数
_tukeylambda_var_qc = [1.0, 3.683605511659861, 4.184152498888124,
                       1.7660926747377275, 0.2643989311168465]

# 创建 numpy.poly1d 实例，表示 Tukey Lambda 方差函数的 Pade 近似的分子和分母
_tukeylambda_var_p = poly1d(_tukeylambda_var_pc[::-1])
_tukeylambda_var_q = poly1d(_tukeylambda_var_qc[::-1])

def tukeylambda_variance(lam):
    """Tukey Lambda 分布的方差计算函数。

    Parameters
    ----------
    lam : array_like
        需要计算方差的 lambda 值数组。

    Returns
    -------
    v : ndarray
        计算得到的方差。对于 lam < -0.5，方差为 np.nan；对于 lam = 0.5，方差为 np.inf。

    Notes
    -----
    在 lambda=0 附近的区间内，此函数使用 [4,4] Pade 近似来计算方差。
    否则使用标准公式（https://en.wikipedia.org/wiki/Tukey_lambda_distribution）。
    使用 Pade 近似是因为标准公式在 lambda=0 处有可去除的不连续性，且在 lambda=0 附近不产生准确的数值结果。

    """
    # 将输入的 lam 转换为 numpy 数组，并确保其为 float64 类型
    lam = np.asarray(lam)
    shp = lam.shape
    lam = np.atleast_1d(lam).astype(np.float64)

    # 对于绝对值小于阈值的 lam，使用 Pade 近似
    threshold = 0.075

    # 设置条件掩码，实现条件分支的计算
    # lambda < -0.5:  var = nan
    low_mask = lam < -0.5
    # lambda == -0.5: var = inf
    neghalf_mask = lam == -0.5
    # abs(lambda) < threshold:  use Pade approximation
    small_mask = np.abs(lam) < threshold
    # 否则使用标准公式
    reg_mask = ~(low_mask | neghalf_mask | small_mask)

    # 根据掩码将 lam 分成不同的部分
    small = lam[small_mask]
    reg = lam[reg_mask]

    # 对每种情况计算函数值。
    # 创建一个和 lam 相同形状的空数组 v
    v = np.empty_like(lam)
    # 将符合 low_mask 条件的元素设为 NaN
    v[low_mask] = np.nan
    # 将符合 neghalf_mask 条件的元素设为正无穷大
    v[neghalf_mask] = np.inf
    if small.size > 0:
        # 对于符合 small_mask 条件的元素，使用 Pade 近似公式计算 Tukey lambda 分布的方差
        v[small_mask] = _tukeylambda_var_p(small) / _tukeylambda_var_q(small)
    if reg.size > 0:
        # 对于符合 reg_mask 条件的元素，计算 Tukey lambda 分布的方差
        v[reg_mask] = (2.0 / reg**2) * (1.0 / (1.0 + 2 * reg) -
                                        beta(reg + 1, reg + 1))
    # 将 v 的形状重置为 shp
    v.shape = shp
    # 返回数组 v
    return v
# The following code was used to generate the Pade coefficients for the
# Tukey Lambda kurtosis function.  Version 0.17 of mpmath was used.
#---------------------------------------------------------------------------
import mpmath as mp

# Set the desired precision (decimal places) to 60 for mpmath calculations
mp.mp.dps = 60

# Define constants as arbitrary precision floating point numbers
one   = mp.mpf(1)
two   = mp.mpf(2)
three = mp.mpf(3)
four  = mp.mpf(4)

# Define the function mpkurt to compute the kurtosis of Tukey Lambda distribution
def mpkurt(lam):
    if lam == 0:
        # Special case when lambda is 0
        k = mp.mpf(6)/5
    else:
        # Calculate numerator and denominator for kurtosis formula
        numer = (one/(four*lam+one) - four*mp.beta(three*lam+one, lam+one) +
                 three*mp.beta(two*lam+one, two*lam+one))
        denom = two*(one/(two*lam+one) - mp.beta(lam+one,lam+one))**2
        # Compute kurtosis using the derived formula
        k = numer / denom - three
    return k

# There is a known issue with mpmath 0.17 where requesting a degree 9 Taylor polynomial
# with 'method'='quad' actually provides a degree 8 polynomial.
t = mp.taylor(mpkurt, 0, 9, method='quad', radius=0.01)
# Remove very small coefficients from the Taylor series approximation
t = [mp.chop(c, tol=1e-15) for c in t]
# Compute Pade approximants for the Taylor series
p, q = mp.pade(t, 4, 4)
# Print the Pade coefficients for inspection
print("p =", [mp.fp.mpf(c) for c in p])
print("q =", [mp.fp.mpf(c) for c in q])
#---------------------------------------------------------------------------

# Pade coefficients for the Tukey Lambda kurtosis function.
_tukeylambda_kurt_pc = [1.2, -5.853465139719495, -22.653447381131077,
                        0.20601184383406815, 4.59796302262789]
_tukeylambda_kurt_qc = [1.0, 7.171149192233599, 12.96663094361842,
                        0.43075235247853005, -2.789746758009912]

# numpy.poly1d instances for the numerator and denominator of the
# Pade approximation to the Tukey Lambda kurtosis.
_tukeylambda_kurt_p = poly1d(_tukeylambda_kurt_pc[::-1])
_tukeylambda_kurt_q = poly1d(_tukeylambda_kurt_qc[::-1])


def tukeylambda_kurtosis(lam):
    """Kurtosis of the Tukey Lambda distribution.

    Parameters
    ----------
    lam : array_like
        The lambda values at which to compute the variance.

    Returns
    -------
    v : ndarray
        The variance.  For lam < -0.25, the variance is not defined, so
        np.nan is returned.  For lam = 0.25, np.inf is returned.

    """
    # Convert lam to a numpy array of type float64
    lam = np.asarray(lam)
    # Store the original shape of lam
    shp = lam.shape
    # Ensure lam is at least 1-dimensional and of type float64
    lam = np.atleast_1d(lam).astype(np.float64)

    # Define a threshold below which the Pade approximation is used
    threshold = 0.055

    # Define masks for different cases of lam values
    low_mask = lam < -0.25         # lambda < -0.25
    negqrtr_mask = lam == -0.25    # lambda == -0.25
    small_mask = np.abs(lam) < threshold  # lambda near 0
    reg_mask = ~(low_mask | negqrtr_mask | small_mask)  # "regular" case

    # Extract lam values for specific cases
    small = lam[small_mask]
    reg = lam[reg_mask]

    # Initialize array for kurtosis values
    k = np.empty_like(lam)
    # Set kurtosis values based on masks
    k[low_mask] = np.nan        # For lambda < -0.25, kurtosis is NaN
    k[negqrtr_mask] = np.inf    # For lambda == -0.25, kurtosis is Inf
    # 如果 `small` 的大小大于 0，则计算 `small` 的特定函数结果并赋值给 `k` 的相应位置
    if small.size > 0:
        k[small_mask] = _tukeylambda_kurt_p(small) / _tukeylambda_kurt_q(small)
    
    # 如果 `reg` 的大小大于 0，则计算 `reg` 的特定数值 `numer` 和 `denom`，并将结果赋值给 `k` 的相应位置
    if reg.size > 0:
        numer = (1.0 / (4 * reg + 1) - 4 * beta(3 * reg + 1, reg + 1) +
                 3 * beta(2 * reg + 1, 2 * reg + 1))
        denom = 2 * (1.0/(2 * reg + 1) - beta(reg + 1, reg + 1))**2
        k[reg_mask] = numer / denom - 3

    # 返回值是一个 NumPy 数组；重新设置形状以确保如果 `k` 最初是标量，则返回值是一个零维数组。
    k.shape = shp
    return k
```