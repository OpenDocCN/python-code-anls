# `D:\src\scipysrc\scipy\scipy\stats\_qmvnt.py`

```
# Integration of multivariate normal and t distributions.

# Adapted from the MATLAB original implementations by Dr. Alan Genz.

#     http://www.math.wsu.edu/faculty/genz/software/software.html

# Copyright (C) 2013, Alan Genz,  All rights reserved.
# Python implementation is copyright (C) 2022, Robert Kern,  All rights
# reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided the following conditions are met:
#   1. Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#   2. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in
#      the documentation and/or other materials provided with the
#      distribution.
#   3. The contributor name(s) may not be used to endorse or promote
#      products derived from this software without specific prior
#      written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np

from scipy.fft import fft, ifft
from scipy.special import gammaincinv, ndtr, ndtri
from scipy.stats._qmc import primes_from_2_to


# 使用 `ndtr` 函数别名 `phi` 表示正态累积分布函数
phi = ndtr
# 使用 `ndtri` 函数别名 `phinv` 表示正态分布的逆累积分布函数
phinv = ndtri


def _factorize_int(n):
    """Return a sorted list of the unique prime factors of a positive integer.
    """
    # 计算正整数 `n` 的唯一质因数，并按升序排序返回列表
    factors = set()
    for p in primes_from_2_to(int(np.sqrt(n)) + 1):
        while not (n % p):
            factors.add(p)
            n //= p
        if n == 1:
            break
    if n != 1:
        factors.add(n)
    return sorted(factors)


def _primitive_root(p):
    """Compute a primitive root of the prime number `p`.

    Used in the CBC lattice construction.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Primitive_root_modulo_n
    """
    # 计算质数 `p` 的一个原根
    pm = p - 1
    # 获取 `p-1` 的所有唯一质因数
    factors = _factorize_int(pm)
    n = len(factors)
    r = 2
    k = 0
    while k < n:
        d = pm // factors[k]
        # 使用快速幂计算 r^d % p，以确定是否为原根
        rd = pow(int(r), int(d), int(p))
        if rd == 1:
            r += 1
            k = 0
        else:
            k += 1
    return r
# 定义一个函数，用于计算基于 Fast CBC 构造的 QMC 格点生成器。
def _cbc_lattice(n_dim, n_qmc_samples):
    """Compute a QMC lattice generator using a Fast CBC construction.

    Parameters
    ----------
    n_dim : int > 0
        格点的维度数。
    n_qmc_samples : int > 0
        所需的 QMC 样本数。将会向下取最接近的素数以启用 CBC 构造。

    Returns
    -------
    q : float array : shape=(n_dim,)
        格点生成器向量。所有值都在开区间 ``(0, 1)`` 内。
    actual_n_qmc_samples : int
        必须与此格点一起使用的素数 QMC 样本数。

    References
    ----------
    .. [1] Nuyens, D. and Cools, R. "Fast Component-by-Component Construction,
           a Reprise for Different Kernels", In H. Niederreiter and D. Talay,
           editors, Monte-Carlo and Quasi-Monte Carlo Methods 2004,
           Springer-Verlag, 2006, 371-385.
    """
    # 向下取最接近的素数。
    primes = primes_from_2_to(n_qmc_samples + 1)
    n_qmc_samples = primes[-1]

    bt = np.ones(n_dim)
    gm = np.hstack([1.0, 0.8 ** np.arange(n_dim - 1)])
    q = 1
    w = 0
    z = np.arange(1, n_dim + 1)
    m = (n_qmc_samples - 1) // 2
    g = _primitive_root(n_qmc_samples)
    # 更快地计算 perm[j] = pow(g, j, n_qmc_samples)
    # 遗憾的是，我们没有将 pow() 实现为 ufunc 的模块化。
    perm = np.ones(m, dtype=int)
    for j in range(m - 1):
        perm[j + 1] = (g * perm[j]) % n_qmc_samples
    perm = np.minimum(n_qmc_samples - perm, perm)
    pn = perm / n_qmc_samples
    c = pn * pn - pn + 1.0 / 6
    fc = fft(c)
    for s in range(1, n_dim):
        reordered = np.hstack([
            c[:w+1][::-1],
            c[w+1:m][::-1],
        ])
        q = q * (bt[s-1] + gm[s-1] * reordered)
        w = ifft(fc * fft(q)).real.argmin()
        z[s] = perm[w]
    q = z / n_qmc_samples
    return q, n_qmc_samples


# 注意：此函数目前未被任何 SciPy 代码使用或测试。它包含在此文件中，以便为用户设置所需的 CDF 精度参数提供开发支持，
# 但在使用之前必须进行审查和测试。
def _qauto(func, covar, low, high, rng, error=1e-3, limit=10_000, **kwds):
    """Automatically rerun the integration to get the required error bound.

    Parameters
    ----------
    func : callable
        Either :func:`_qmvn` or :func:`_qmvt`.
    covar, low, high : array
        As specified in :func:`_qmvn` and :func:`_qmvt`.
    rng : Generator, optional
        default_rng(), yada, yada
    error : float > 0
        The desired error bound.
    limit : int > 0:
        The rough limit of the number of integration points to consider. The
        integration will stop looping once this limit has been *exceeded*.
    **kwds :
        Other keyword arguments to pass to `func`. When using :func:`_qmvt`, be
        sure to include ``nu=`` as one of these.

    Returns
    -------
    ```
    # 定义返回的变量prob（概率）、est_error（估计误差）、n_samples（实际使用的积分点数）
    prob : float
        # 在边界内估计的概率质量。
    est_error : float
        # 批量估计的标准误差的3倍。
    n_samples : int
        # 实际使用的积分点数。
    """
    # 获取协方差矩阵的维度
    n = len(covar)
    # 初始化实际使用的积分点数为0
    n_samples = 0
    # 如果协方差矩阵维度为1
    if n == 1:
        # 通过累积分布函数计算概率的估计值
        prob = phi(high) - phi(low)
        # 设置一个非常小的估计误差
        est_error = 1e-15
    else:
        # 初始化限制最大迭代次数mi为给定限制和协方差矩阵维度乘以1000的最小值
        mi = min(limit, n * 1000)
        # 初始化概率估计为0
        prob = 0.0
        # 初始化估计误差为1.0
        est_error = 1.0
        # 初始化期望增量ei为0.0
        ei = 0.0
        # 当估计误差大于预设误差并且实际使用的积分点数小于限制时执行循环
        while est_error > error and n_samples < limit:
            # 更新迭代次数的上限
            mi = round(np.sqrt(2) * mi)
            # 调用给定的函数func计算pi（估计的概率）、ei（估计的误差）、ni（积分点数）
            pi, ei, ni = func(mi, covar, low, high, rng=rng, **kwds)
            # 累计实际使用的积分点数
            n_samples += ni
            # 计算权重，以更准确估计概率值
            wt = 1.0 / (1 + (ei / est_error)**2)
            # 更新概率的估计值
            prob += wt * (pi - prob)
            # 更新估计误差
            est_error = np.sqrt(wt) * ei
    # 返回估计的概率、估计的误差和实际使用的积分点数
    return prob, est_error, n_samples
# Note: this function is not currently used or tested by any SciPy code. It is
# included in this file to facilitate the resolution of gh-8367, gh-16142, and
# possibly gh-14286, but must be reviewed and tested before use.
def _qmvn(m, covar, low, high, rng, lattice='cbc', n_batches=10):
    """Multivariate normal integration over box bounds.

    Parameters
    ----------
    m : int > n_batches
        The number of points to sample. This number will be divided into
        `n_batches` batches that apply random offsets of the sampling lattice
        for each batch in order to estimate the error.
    covar : (n, n) float array
        Possibly singular, positive semidefinite symmetric covariance matrix.
    low, high : (n,) float array
        The low and high integration bounds.
    rng : Generator, optional
        default_rng(), yada, yada
        Random number generator used for generating random shifts.
    lattice : 'cbc' or callable
        The type of lattice rule to use to construct the integration points.
    n_batches : int > 0, optional
        The number of QMC batches to apply.

    Returns
    -------
    prob : float
        The estimated probability mass within the bounds.
    est_error : float
        3 times the standard error of the batch estimates.
    """
    # Compute permuted Cholesky factorization and adjust integration bounds
    cho, lo, hi = _permuted_cholesky(covar, low, high)
    n = cho.shape[0]
    ct = cho[0, 0]
    c = phi(lo[0] / ct)
    d = phi(hi[0] / ct)
    ci = c
    dci = d - ci
    prob = 0.0
    error_var = 0.0
    # Generate quasi-Monte Carlo (QMC) lattice points and initialize arrays
    q, n_qmc_samples = _cbc_lattice(n - 1, max(m // n_batches, 1))
    y = np.zeros((n - 1, n_qmc_samples))
    i_samples = np.arange(n_qmc_samples) + 1
    # Perform QMC integration over batches
    for j in range(n_batches):
        c = np.full(n_qmc_samples, ci)
        dc = np.full(n_qmc_samples, dci)
        pv = dc.copy()
        for i in range(1, n):
            # Pseudorandomly-shifted lattice coordinate.
            z = q[i - 1] * i_samples + rng.random()
            # Fast remainder(z, 1.0)
            z -= z.astype(int)
            # Tent periodization transform.
            x = abs(2 * z - 1)
            y[i - 1, :] = phinv(c + x * dc)
            s = cho[i, :i] @ y[:i, :]
            ct = cho[i, i]
            c = phi((lo[i] - s) / ct)
            d = phi((hi[i] - s) / ct)
            dc = d - c
            pv = pv * dc
        # Accumulate the mean and error variances with online formulations.
        d = (pv.mean() - prob) / (j + 1)
        prob += d
        error_var = (j - 1) * error_var / (j + 1) + d * d
    # Error bounds are 3 times the standard error of the estimates.
    est_error = 3 * np.sqrt(error_var)
    n_samples = n_qmc_samples * n_batches
    return prob, est_error, n_samples


# Note: this function is not currently used or tested by any SciPy code. It is
# included in this file to facilitate the resolution of gh-8367, gh-16142, and
# possibly gh-14286, but must be reviewed and tested before use.
def _mvn_qmc_integrand(covar, low, high, use_tent=False):
    """Transform the multivariate normal integration into a QMC integrand over
    # 定义一个函数，返回一个用于 QMC 积分的可调用函数和积分维度
    def integrate_qmc_transform(covar, low, high, use_tent=False):
        # 调用 _permuted_cholesky 函数，返回置换后的 Cholesky 分解结果和更新后的积分上下界
        cho, lo, hi = _permuted_cholesky(covar, low, high)
        # 矩阵 cho 的行数，即积分维度的原始维度 n
        n = cho.shape[0]
        # 积分函数的维度为 n-1
        ndim_integrand = n - 1
        # 选取 cho 矩阵的第一个元素作为 ct
        ct = cho[0, 0]
        # 计算低界的归一化 phi 值
        c = phi(lo[0] / ct)
        # 计算高界的归一化 phi 值
        d = phi(hi[0] / ct)
        # 将 c 的值赋给 ci
        ci = c
        # 计算 dci（高界和低界 phi 值之差）
        dci = d - ci
    
        # 定义积分函数 integrand
        def integrand(*zs):
            # zs 的维度应该等于积分维度 ndim_integrand
            ndim_qmc = len(zs)
            # 获取 QMC 样本数量
            n_qmc_samples = len(np.atleast_1d(zs[0]))
            # 断言 ndim_qmc 等于 ndim_integrand
            assert ndim_qmc == ndim_integrand
            # 初始化一个 ndim_qmc x n_qmc_samples 的零矩阵 y
            y = np.zeros((ndim_qmc, n_qmc_samples))
            # 将 ci 复制为长度为 n_qmc_samples 的数组 c
            c = np.full(n_qmc_samples, ci)
            # 将 dci 复制为长度为 n_qmc_samples 的数组 dc
            dc = np.full(n_qmc_samples, dci)
            # 将 pv 初始化为 dc 的复制
            pv = dc.copy()
    
            # 循环遍历 cho 矩阵的行数
            for i in range(1, n):
                # 如果 use_tent 为真，则使用 Tent periodization 变换
                if use_tent:
                    # Tent periodization 变换
                    x = abs(2 * zs[i-1] - 1)
                else:
                    # 否则直接使用 zs[i-1]
                    x = zs[i-1]
                # 计算 y 的第 i-1 行，即 phi 逆函数应用于 c + x * dc
                y[i - 1, :] = phinv(c + x * dc)
                # 计算矩阵 s，为 cho 矩阵第 i 行前 i 列与 y 的点积
                s = cho[i, :i] @ y[:i, :]
                # 更新 ct 为 cho 矩阵第 i 行第 i 列元素
                ct = cho[i, i]
                # 计算 c 为 phi 函数应用于 (lo[i] - s) / ct
                c = phi((lo[i] - s) / ct)
                # 计算 d 为 phi 函数应用于 (hi[i] - s) / ct
                d = phi((hi[i] - s) / ct)
                # 更新 dc 为 d - c
                dc = d - c
                # 更新 pv 为 pv 乘以 dc
                pv = pv * dc
            
            # 返回最终的 pv 结果
            return pv
    
        # 返回积分函数 integrand 和积分维度 ndim_integrand
        return integrand, ndim_integrand
# 定义一个函数 `_qmvt`，用于多元 t 分布在给定盒子边界上的积分估算

def _qmvt(m, nu, covar, low, high, rng, lattice='cbc', n_batches=10):
    """Multivariate t integration over box bounds.

    Parameters
    ----------
    m : int > n_batches
        要采样的点数。这个数字将被分成 `n_batches` 批次，每个批次应用采样点阵的随机偏移，
        以估计每个批次的误差。
    nu : float >= 0
        多元 t 分布的形状参数。
    covar : (n, n) float array
        可能奇异的、半正定的对称协方差矩阵。
    low, high : (n,) float array
        积分的低和高边界。
    rng : Generator, optional
        default_rng()，生成随机数用的生成器
    lattice : 'cbc' or callable
        构造积分点阵使用的格点规则类型。
    n_batches : int > 0, optional
        要应用的 QMC 批次数量。

    Returns
    -------
    prob : float
        边界内的估计概率质量。
    est_error : float
        批次估计标准误差的3倍。
    n_samples : int
        实际使用的样本数。
    """
    
    # 计算 sn，作为 nu 的平方根或者 1.0 中的较大者
    sn = max(1.0, np.sqrt(nu))
    
    # 将 low 和 high 转换为 np.float64 类型的数组
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    
    # 使用 `_permuted_cholesky` 函数对 covar / sn 范围内的 low 和 high 进行排列并返回 cho, lo, hi
    cho, lo, hi = _permuted_cholesky(covar, low / sn, high / sn)
    
    # 获取 cho 的形状，并初始化 prob 和 error_var
    n = cho.shape[0]
    prob = 0.0
    error_var = 0.0
    
    # 使用 `_cbc_lattice` 函数生成 QMC 点阵 q 和 n_qmc_samples 数量
    q, n_qmc_samples = _cbc_lattice(n, max(m // n_batches, 1))
    
    # 生成从 1 到 n_qmc_samples 的整数数组，并存储在 i_samples 中
    i_samples = np.arange(n_qmc_samples) + 1
    for j in range(n_batches):
        # Initialize array of ones for partial variance calculation
        pv = np.ones(n_qmc_samples)
        # Initialize array for accumulating sums of transformed variables
        s = np.zeros((n, n_qmc_samples))
        for i in range(n):
            # Pseudorandomly-shifted lattice coordinate.
            z = q[i] * i_samples + rng.random()
            # Fast remainder(z, 1.0)
            z -= z.astype(int)
            # Tent periodization transform.
            x = abs(2 * z - 1)
            # FIXME: Lift the i==0 case out of the loop to make the logic
            # easier to follow.
            if i == 0:
                # We'll use one of the QR variates to pull out the
                # t-distribution scaling.
                if nu > 0:
                    # Calculate scaling factor using inverse incomplete gamma function
                    r = np.sqrt(2 * gammaincinv(nu / 2, x))
                else:
                    # If nu <= 0, set scaling factor to ones
                    r = np.ones_like(x)
            else:
                # Apply the inverse phi transform to the shifted variable
                y = phinv(c + x * dc)  # noqa: F821
                # Accumulate transformed variables weighted by Cholesky factors
                with np.errstate(invalid='ignore'):
                    s[i:, :] += cho[i:, i - 1][:, np.newaxis] * y
            # Extract transformed variables for the current coordinate
            si = s[i, :]

            # Initialize arrays for conditions and probabilities
            c = np.ones(n_qmc_samples)
            d = np.ones(n_qmc_samples)
            # Apply transformations considering error states
            with np.errstate(invalid='ignore'):
                lois = lo[i] * r - si
                hiis = hi[i] * r - si
            # Apply thresholds for conditionals based on predefined values
            c[lois < -9] = 0.0
            d[hiis < -9] = 0.0
            lo_mask = abs(lois) < 9
            hi_mask = abs(hiis) < 9
            # Apply normal distribution to the conditions
            c[lo_mask] = phi(lois[lo_mask])
            d[hi_mask] = phi(hiis[hi_mask])

            # Calculate delta C
            dc = d - c
            # Assign the value
            pv *= dc

        # Accumulate the mean and error variances with online formulations.
        d = (pv.mean() - prob) / (j + 1)
        prob += d
        error_var = (j - 1) * error_var / (j + 1) + d * d
    # Error bounds are 3 times the standard error of the estimates.
    est_error = 3 * np.sqrt(error_var)
    n_samples = n_qmc_samples * n_batches
    return prob, est_error, n_samples
# 定义一个函数，计算一个经过缩放和置换的 Cholesky 分解因子，并且包含积分边界。

# 用于输出的复制品。
cho = np.array(covar, dtype=np.float64)
new_lo = np.array(low, dtype=np.float64)
new_hi = np.array(high, dtype=np.float64)

# 获取矩阵的维度。
n = cho.shape[0]

# 检查矩阵是否为方阵。
if cho.shape != (n, n):
    raise ValueError("expected a square symmetric array")

# 检查积分边界是否与协方差矩阵的维度一致。
if new_lo.shape != (n,) or new_hi.shape != (n,):
    raise ValueError(
        "expected integration boundaries the same dimensions "
        "as the covariance matrix"
    )

# 通过协方差矩阵对角线元素的平方根进行缩放。
dc = np.sqrt(np.maximum(np.diag(cho), 0.0))

# 避免除以 0。
dc[dc == 0.0] = 1.0

# 根据对角线元素的平方根对积分下界和上界进行缩放。
new_lo /= dc
new_hi /= dc

# 根据对角线元素的平方根对 Cholesky 分解因子进行缩放。
cho /= dc
cho /= dc[:, np.newaxis]

# 初始化一个长度为 n 的零向量 y 和 sqrt(2*pi) 的平方根 sqtp。
y = np.zeros(n)
sqtp = np.sqrt(2 * np.pi)
    # 对于每一个 k，执行以下操作
    for k in range(n):
        # 计算当前的容忍度
        epk = (k + 1) * tol
        # 初始化最小元素的索引为 k
        im = k
        # 初始化 ck、dem、s、lo_m、hi_m
        ck = 0.0
        dem = 1.0
        s = 0.0
        lo_m = 0.0
        hi_m = 0.0
        # 对于每一个 i 从 k 到 n 进行迭代
        for i in range(k, n):
            # 如果 cho[i, i] 大于容忍度
            if cho[i, i] > tol:
                # 计算 ci 作为 cho[i, i] 的平方根
                ci = np.sqrt(cho[i, i])
                # 如果 i 大于 0，则计算 cho[i, :k] 与 y[:k] 的点积
                if i > 0:
                    s = cho[i, :k] @ y[:k]
                # 计算 lo_i 和 hi_i
                lo_i = (new_lo[i] - s) / ci
                hi_i = (new_hi[i] - s) / ci
                # 计算 de 作为 phi(hi_i) 减去 phi(lo_i)
                de = phi(hi_i) - phi(lo_i)
                # 如果 de 小于等于 dem
                if de <= dem:
                    # 更新 ck、dem、lo_m、hi_m 和 im
                    ck = ci
                    dem = de
                    lo_m = lo_i
                    hi_m = hi_i
                    im = i
        # 如果找到的最小元素的索引 im 大于 k，则进行交换
        if im > k:
            # 交换 im 和 k
            cho[im, im] = cho[k, k]
            _swap_slices(cho, np.s_[im, :k], np.s_[k, :k])
            _swap_slices(cho, np.s_[im + 1:, im], np.s_[im + 1:, k])
            _swap_slices(cho, np.s_[k + 1:im, k], np.s_[im, k + 1:im])
            _swap_slices(new_lo, k, im)
            _swap_slices(new_hi, k, im)
        # 如果 ck 大于 epk
        if ck > epk:
            # 更新 cho[k, k] 和 cho[k, k+1:]
            cho[k, k] = ck
            cho[k, k + 1:] = 0.0
            # 对于每一个 i 从 k+1 到 n 进行迭代
            for i in range(k + 1, n):
                # 更新 cho[i, k] 和 cho[i, k+1:i+1]
                cho[i, k] /= ck
                cho[i, k + 1:i + 1] -= cho[i, k] * cho[k + 1:i + 1, k]
            # 如果 dem 的绝对值大于容忍度
            if abs(dem) > tol:
                # 计算 y[k]，使用正态分布的累积分布函数 phi
                y[k] = ((np.exp(-lo_m * lo_m / 2) - np.exp(-hi_m * hi_m / 2)) /
                        (sqtp * dem))
            else:
                # 否则，使用 lo_m 和 hi_m 的平均值作为 y[k]
                y[k] = (lo_m + hi_m) / 2
                # 如果 lo_m 小于 -10，则使用 hi_m 作为 y[k]
                if lo_m < -10:
                    y[k] = hi_m
                # 如果 hi_m 大于 10，则使用 lo_m 作为 y[k]
                elif hi_m > 10:
                    y[k] = lo_m
            # 更新 cho[k, :k+1]、new_lo[k] 和 new_hi[k]
            cho[k, :k + 1] /= ck
            new_lo[k] /= ck
            new_hi[k] /= ck
        else:
            # 否则，更新 cho[k:, k] 为 0.0，且使用 new_lo[k] 和 new_hi[k] 的平均值作为 y[k]
            cho[k:, k] = 0.0
            y[k] = (new_lo[k] + new_hi[k]) / 2
    # 返回 cho 矩阵、new_lo 和 new_hi 数组
    return cho, new_lo, new_hi
# 定义一个函数，用于交换列表 x 中指定切片 slc1 和 slc2 的内容
def _swap_slices(x, slc1, slc2):
    # 复制 slc1 切片的内容到临时变量 t 中
    t = x[slc1].copy()
    # 将 slc2 切片的内容复制到 slc1 切片位置
    x[slc1] = x[slc2].copy()
    # 将临时变量 t 的内容复制到 slc2 切片位置，实现 slc1 和 slc2 切片内容的交换
    x[slc2] = t
```