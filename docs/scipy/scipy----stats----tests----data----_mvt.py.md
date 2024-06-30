# `D:\src\scipysrc\scipy\scipy\stats\tests\data\_mvt.py`

```
import math
import numpy as np
from scipy import special
from scipy.stats._qmc import primes_from_2_to

def _primes(n):
    # 定义以便进行翻译和源代码的比较
    # 在 Matlab 中，primes(10.5) -> 前四个质数，primes(11.5) -> 前五个
    return primes_from_2_to(math.ceil(n))

def _gaminv(a, b):
    # 定义以便进行翻译和源代码的比较
    # Matlab 的 `gaminv` 类似于 `special.gammaincinv`，但参数顺序相反
    return special.gammaincinv(b, a)

def _qsimvtv(m, nu, sigma, a, b, rng):
    """使用随机 QMC 估算多元 t 分布的累积分布函数

    Parameters
    ----------
    m : int
        点的数量
    nu : float
        自由度
    sigma : ndarray
        二维正半定义协方差矩阵
    a : ndarray
        下积分限
    b : ndarray
        上积分限
    rng : Generator
        伪随机数生成器

    Returns
    -------
    p : float
        估算的累积分布函数值
    e : float
        绝对误差估计

    """
    # _qsimvtv 是 Matlab 函数 qsimvtv 的 Python 翻译，分号也一并保留。
    #
    #   该函数使用的算法参见论文
    #      "Comparison of Methods for the Numerical Computation of
    #       Multivariate t Probabilities", in
    #      J. of Computational and Graphical Stat., 11(2002), pp. 950-971, by
    #          Alan Genz and Frank Bretz
    #
    #   数值积分的主要参考文献包括
    #    "On a Number-Theoretical Integration Method"
    #    H. Niederreiter, Aequationes Mathematicae, 8(1972), pp. 304-11.
    #    和
    #    "Randomization of Number Theoretic Methods for Multiple Integration"
    #     R. Cranley & T.N.L. Patterson, SIAM J Numer Anal, 13(1976), pp. 904-14.
    #
    #   Alan Genz 是该函数及以下 Matlab 函数的作者。
    #          Alan Genz, WSU Math, PO Box 643113, Pullman, WA 99164-3113
    #          Email : alangenz@wsu.edu
    #
    # 版权所有 © 2013, Alan Genz
    #
    # 源代码及其衍生的源码，在符合以下条件的情况下，允许以源码和二进制形式重新发布和使用：
    #   1. 源代码的再分发必须保留上述版权声明、此条件列表和以下免责声明。
    #   2. 以二进制形式再分发时，必须在文档和/或其他提供的材料中重现上述版权声明、此条件列表和以下免责声明。
    #   3. 未经特定书面许可，不能使用贡献者的名称来认可或推广由此软件衍生的产品。
    # 本软件由版权持有人和贡献者“按原样”提供，不提供任何明示或暗示的担保，
    # 包括但不限于对适销性和特定用途的适用性的暗示担保。
    #
    # 根据上述注释，_qsimvtv 函数执行多元 t 分布的累积分布函数的估算。
    pass
    # 初始化
    sn = max(1, math.sqrt(nu)); ch, az, bz = _chlrps(sigma, a/sn, b/sn)
    n = len(sigma); N = 10; P = math.ceil(m/N); on = np.ones(P); p = 0; e = 0
    ps = np.sqrt(_primes(5*n*math.log(n+4)/4)); q = ps[:, np.newaxis]  # Richtmyer gens.

    # 针对ns个样本的随机化循环
    c = None; dc = None
    for S in range(N):
        vp = on.copy(); s = np.zeros((n, P))
        for i in range(n):
            x = np.abs(2*np.mod(q[i]*np.arange(1, P+1) + rng.random(), 1)-1)  # 周期化变换
            if i == 0:
                r = on
                if nu > 0:
                    r = np.sqrt(2*_gaminv(x, nu/2))
            else:
                y = _Phinv(c + x*dc)
                s[i:] += ch[i:, i-1:i] * y
            si = s[i, :]; c = on.copy(); ai = az[i]*r - si; d = on.copy(); bi = bz[i]*r - si
            c[ai <= -9] = 0; tl = abs(ai) < 9; c[tl] = _Phi(ai[tl])
            d[bi <= -9] = 0; tl = abs(bi) < 9; d[tl] = _Phi(bi[tl])
            dc = d - c; vp = vp * dc
        d = (np.mean(vp) - p)/(S + 1); p = p + d; e = (S - 1)*e/(S + 1) + d**2
    e = math.sqrt(e)  # 误差估计为N个样本的3倍标准误差。
    return p, e
# Standard statistical normal distribution functions

# 定义标准正态分布的累积分布函数 Phi
def _Phi(z):
    return special.ndtr(z)

# 定义标准正态分布的逆累积分布函数 Phi^{-1}
def _Phinv(p):
    return special.ndtri(p)

# 计算经过置换和缩放的下三角 Cholesky 分解因子 c，处理可能是奇异的 R，同时对积分限制向量 a 和 b 进行置换和缩放
def _chlrps(R, a, b):
    """
    Computes permuted and scaled lower Cholesky factor c for R which may be
    singular, also permuting and scaling integration limit vectors a and b.
    """
    ep = 1e-10  # 奇异性容差值
    eps = np.finfo(R.dtype).eps

    n = len(R); c = R.copy(); ap = a.copy(); bp = b.copy(); d = np.sqrt(np.maximum(np.diag(c), 0))
    
    # 对每个主对角元素进行归一化处理
    for i in range(n):
        if d[i] > 0:
            c[:, i] /= d[i]
            c[i, :] /= d[i]
            ap[i] /= d[i]
            bp[i] /= d[i]
    
    y = np.zeros((n, 1)); sqtp = math.sqrt(2*math.pi)

    # 开始 Cholesky 分解和置换过程
    for k in range(n):
        im = k; ckk = 0; dem = 1; s = 0
        
        # 在当前列 k 中找出主元素及其位置
        for i in range(k, n):
            if c[i, i] > eps:
                cii = math.sqrt(max(c[i, i], 0))
                if i > 0:
                    s = c[i, :k] @ y[:k]  # 矩阵乘法运算
                ai = (ap[i]-s)/cii
                bi = (bp[i]-s)/cii
                de = _Phi(bi) - _Phi(ai)
                
                # 更新最小 dem 和相关变量
                if de <= dem:
                    ckk = cii
                    dem = de
                    am = ai
                    bm = bi
                    im = i
        
        # 若找到更好的主元素位置，则进行置换操作
        if im > k:
            ap[[im, k]] = ap[[k, im]]
            bp[[im, k]] = bp[[k, im]]
            c[im, im] = c[k, k]
            t = c[im, :k].copy()
            c[im, :k] = c[k, :k]
            c[k, :k] = t
            t = c[im+1:, im].copy()
            c[im+1:, im] = c[im+1:, k]
            c[im+1:, k] = t
            t = c[k+1:im, k].copy()
            c[k+1:im, k] = c[im, k+1:im].T
            c[im, k+1:im] = t.T
        
        # 若主元素大于奇异性容差值，则进行 Cholesky 因子处理
        if ckk > ep*(k+1):
            c[k, k] = ckk
            c[k, k+1:] = 0
            
            # 更新下一列的系数
            for i in range(k+1, n):
                c[i, k] = c[i, k] / ckk
                c[i, k+1:i+1] = c[i, k+1:i+1] - c[i, k] * c[k+1:i+1, k].T
            
            # 计算 y[k] 的值
            if abs(dem) > ep:
                y[k] = (np.exp(-am**2/2) - np.exp(-bm**2/2)) / (sqtp * dem)
            else:
                y[k] = (am + bm) / 2
                if am < -10:
                    y[k] = bm
                elif bm > 10:
                    y[k] = am
            
            c[k, :k+1] /= ckk
            ap[k] /= ckk
            bp[k] /= ckk
        
        # 若主元素小于等于奇异性容差值，则将该列置为零
        else:
            c[k:, k] = 0
            y[k] = (ap[k] + bp[k]) / 2
        
        pass
    
    # 返回结果 c, ap, bp
    return c, ap, bp
```