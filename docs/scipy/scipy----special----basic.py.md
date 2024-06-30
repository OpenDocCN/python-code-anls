# `D:\src\scipysrc\scipy\scipy\special\basic.py`

```
# 导入 _sub_module_deprecation 函数，用于处理子模块过时的警告
from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个列表，包含特定子模块中的所有函数名称
__all__ = [  # noqa: F822
    'ai_zeros',         # Airy 函数 Ai 的零点
    'assoc_laguerre',   # 关联的Laguerre多项式
    'bei_zeros',        # 第一类贝塞尔函数的零点
    'beip_zeros',       # 第一类贝塞尔函数的导数的零点
    'ber_zeros',        # 第二类贝塞尔函数的零点
    'bernoulli',        # 伯努利数
    'berp_zeros',       # 第二类贝塞尔函数的导数的零点
    'bi_zeros',         # Airy 函数 Bi 的零点
    'clpmn',            # Clebsch-Gordan系数
    'comb',             # 计算组合数
    'digamma',          # Digamma 函数
    'diric',            # Dirichlet 冲激函数
    'erf_zeros',        # 误差函数的零点
    'euler',            # 欧拉常数
    'factorial',        # 阶乘函数
    'factorial2',       # 双阶乘函数
    'factorialk',       # 广义阶乘函数
    'fresnel_zeros',    # Fresnel S 和 C 函数的零点
    'fresnelc_zeros',   # Fresnel C 函数的零点
    'fresnels_zeros',   # Fresnel S 函数的零点
    'gamma',            # Gamma 函数
    'h1vp',             # Hankel 函数 H1 的导数
    'h2vp',             # Hankel 函数 H2 的导数
    'hankel1',          # 第一类 Hankel 函数
    'hankel2',          # 第二类 Hankel 函数
    'iv',               # Modified Bessel 函数 I
    'ivp',              # Modified Bessel 函数 I 的导数
    'jn_zeros',         # 贝塞尔函数 J 的零点
    'jnjnp_zeros',      # 贝塞尔函数 J 和 J' 的零点
    'jnp_zeros',        # 贝塞尔函数 J' 的零点
    'jnyn_zeros',       # 贝塞尔函数 J 和 Y 的零点
    'jv',               # 贝塞尔函数 J
    'jvp',              # 贝塞尔函数 J 的导数
    'kei_zeros',        # 凯尔函数 Kei 的零点
    'keip_zeros',       # 凯尔函数 Kei 的导数的零点
    'kelvin_zeros',     # 开尔文函数的零点
    'ker_zeros',        # Modified Bessel 函数 K 的零点
    'kerp_zeros',       # Modified Bessel 函数 K 的导数的零点
    'kv',               # Modified Bessel 函数 K
    'kvp',              # Modified Bessel 函数 K 的导数
    'lmbda',            # Lambda 函数
    'lpmn',             # 联合的 Legendre 函数
    'lpn',              # Legendre 函数 P
    'lqmn',             # 联合的 Laguerre 函数
    'lqn',              # Laguerre 函数 Q
    'mathieu_a',        # Mathieu 函数参数 a 的特定值
    'mathieu_b',        # Mathieu 函数参数 b 的特定值
    'mathieu_even_coef',# Mathieu 函数的偶数系数
    'mathieu_odd_coef', # Mathieu 函数的奇数系数
    'obl_cv_seq',       # 奥布尔函数的相干序列
    'pbdn_seq',         # 球谐函数 P 的正定序列
    'pbdv_seq',         # 球谐函数 P 的导数的正定序列
    'pbvv_seq',         # 球谐函数 P 和 Q 的交叉序列
    'perm',             # 计算排列数
    'polygamma',        # Polygamma 函数
    'pro_cv_seq',       # 斯蒂尔切斯积分的相干序列
    'psi',              # Digamma 函数
    'riccati_jn',       # Riccati-Bessel 函数 J_n 的解
    'riccati_yn',       # Riccati-Bessel 函数 Y_n 的解
    'sinc',             # 同步函数
    'y0_zeros',         # 贝塞尔函数 Y0 的零点
    'y1_zeros',         # 贝塞尔函数 Y1 的零点
    'y1p_zeros',        # 贝塞尔函数 Y1 的导数的零点
    'yn_zeros',         # 贝塞尔函数 Y 的零点
    'ynp_zeros',        # 贝塞尔函数 Y 的导数的零点
    'yv',               # 贝塞尔函数 Y
    'yvp',              # 贝塞尔函数 Y 的导数
    'zeta'              # Riemann zeta 函数
]

# 定义 __dir__() 函数，返回预定义的 __all__ 列表
def __dir__():
    return __all__

# 定义 __getattr__(name) 函数，处理对不存在的属性的访问
def __getattr__(name):
    return _sub_module_deprecation(sub_package="special", module="basic",
                                   private_modules=["_basic", "_ufuncs"], all=__all__,
                                   attribute=name)
```