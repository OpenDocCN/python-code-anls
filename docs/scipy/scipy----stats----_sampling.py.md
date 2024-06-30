# `D:\src\scipysrc\scipy\scipy\stats\_sampling.py`

```
# 导入所需的数学库和模块
import math
import numbers
import numpy as np
from scipy import stats
from scipy import special as sc
# 从 _qmc 模块导入指定函数和类别
from ._qmc import (check_random_state as check_random_state_qmc,
                   Halton, QMCEngine)
# 从 _unuran.unuran_wrapper 模块导入 NumericalInversePolynomial 类
from ._unuran.unuran_wrapper import NumericalInversePolynomial
# 从 scipy._lib._util 模块导入 check_random_state 函数

__all__ = ['FastGeneratorInversion', 'RatioUniforms']

# 定义 PDF 函数和其他辅助函数以创建生成器

def argus_pdf(x, chi):
    # 根据 Baumgarten/Hoermann 的方法生成 ARGUS 随机变量
    # 对于 chi > 5，使用 ARGUS 分布与 Gamma(1.5) 的关系
    if chi <= 5:
        y = 1 - x * x
        return x * math.sqrt(y) * math.exp(-0.5 * chi**2 * y)
    return math.sqrt(x) * math.exp(-x)

def argus_gamma_trf(x, chi):
    if chi <= 5:
        return x
    return np.sqrt(1.0 - 2 * x / chi**2)

def argus_gamma_inv_trf(x, chi):
    if chi <= 5:
        return x
    return 0.5 * chi**2 * (1 - x**2)

def betaprime_pdf(x, a, b):
    if x > 0:
        logf = (a - 1) * math.log(x) - (a + b) * math.log1p(x) - sc.betaln(a, b)
        return math.exp(logf)
    else:
        # 单独计算 x == 0 时的概率密度函数，避免运行时警告
        if a > 1:
            return 0
        elif a < 1:
            return np.inf
        else:
            return 1 / sc.beta(a, b)

def beta_valid_params(a, b):
    return (min(a, b) >= 0.1) and (max(a, b) <= 700)

def gamma_pdf(x, a):
    if x > 0:
        return math.exp(-math.lgamma(a) + (a - 1.0) * math.log(x) - x)
    else:
        return 0 if a >= 1 else np.inf

def invgamma_pdf(x, a):
    if x > 0:
        return math.exp(-(a + 1.0) * math.log(x) - math.lgamma(a) - 1 / x)
    else:
        return 0 if a >= 1 else np.inf

def burr_pdf(x, cc, dd):
    # 注意：我们使用 np.exp 而不是 math.exp，否则在设置中可能会发生溢出错误，
    # 例如参数为 1.89128135, 0.30195177，参见 test_burr_overflow 测试
    if x > 0:
        lx = math.log(x)
        return np.exp(-(cc + 1) * lx - (dd + 1) * math.log1p(np.exp(-cc * lx)))
    else:
        return 0

def burr12_pdf(x, cc, dd):
    if x > 0:
        lx = math.log(x)
        logterm = math.log1p(math.exp(cc * lx))
        return math.exp((cc - 1) * lx - (dd + 1) * logterm + math.log(cc * dd))
    else:
        return 0

def chi_pdf(x, a):
    if x > 0:
        return math.exp(
            (a - 1) * math.log(x)
            - 0.5 * (x * x)
            - (a / 2 - 1) * math.log(2)
            - math.lgamma(0.5 * a)
        )
    else:
        return 0 if a >= 1 else np.inf

def chi2_pdf(x, df):
    if x > 0:
        return math.exp(
            (df / 2 - 1) * math.log(x)
            - 0.5 * x
            - (df / 2) * math.log(2)
            - math.lgamma(0.5 * df)
        )
    else:
        return 0 if df >= 1 else np.inf

def alpha_pdf(x, a):
    if x > 0:
        return math.exp(-2.0 * math.log(x) - 0.5 * (a - 1.0 / x) ** 2)
    return 0.0

def bradford_pdf(x, c):
    # 这里还需要继续完成函数的编写和注释
    # 如果 x 的值在区间 [0, 1] 内
    if 0 <= x <= 1:
        # 返回计算结果，这里是一个分式的形式，计算 1.0 / (1.0 + c * x)
        return 1.0 / (1.0 + c * x)
    # 如果 x 的值不在区间 [0, 1] 内，则返回 0.0
    return 0.0
# 定义一个概率密度函数，描述水晶球分布的概率密度函数
def crystalball_pdf(x, b, m):
    # 如果 x 大于 -b，返回水晶球分布的概率密度函数值
    if x > -b:
        return math.exp(-0.5 * x * x)
    # 否则返回另一种形式的水晶球分布的概率密度函数值
    return math.exp(m * math.log(m / b) - 0.5 * b * b - m * math.log(m / b - b - x))


# 定义一个概率密度函数，描述最小韦伯分布的概率密度函数
def weibull_min_pdf(x, c):
    # 如果 x 大于 0，返回最小韦伯分布的概率密度函数值
    if x > 0:
        return c * math.exp((c - 1) * math.log(x) - x**c)
    # 否则返回 0.0
    return 0.0


# 定义一个概率密度函数，描述最大韦伯分布的概率密度函数
def weibull_max_pdf(x, c):
    # 如果 x 小于 0，返回最大韦伯分布的概率密度函数值
    if x < 0:
        return c * math.exp((c - 1) * math.log(-x) - ((-x) ** c))
    # 否则返回 0.0
    return 0.0


# 定义一个概率密度函数，描述反韦伯分布的概率密度函数
def invweibull_pdf(x, c):
    # 如果 x 大于 0，返回反韦伯分布的概率密度函数值
    if x > 0:
        return c * math.exp(-(c + 1) * math.log(x) - x ** (-c))
    # 否则返回 0.0
    return 0.0


# 定义一个概率密度函数，描述 Wald 分布的概率密度函数
def wald_pdf(x):
    # 如果 x 大于 0，返回 Wald 分布的概率密度函数值
    if x > 0:
        return math.exp(-((x - 1) ** 2) / (2 * x)) / math.sqrt(x**3)
    # 否则返回 0.0
    return 0.0


# 定义一个概率密度函数，描述广义逆高斯分布的众数
def geninvgauss_mode(p, b):
    # 如果 p 大于 1，返回广义逆高斯分布的众数
    if p > 1:
        return (math.sqrt((1 - p) ** 2 + b**2) - (1 - p)) / b
    # 否则返回另一种形式的广义逆高斯分布的众数
    return b / (math.sqrt((1 - p) ** 2 + b**2) + (1 - p))


# 定义一个概率密度函数，描述广义逆高斯分布的概率密度函数
def geninvgauss_pdf(x, p, b):
    # 计算广义逆高斯分布的众数
    m = geninvgauss_mode(p, b)
    # 计算广义逆高斯分布的概率密度函数的对数值
    lfm = (p - 1) * math.log(m) - 0.5 * b * (m + 1 / m)
    # 如果 x 大于 0，返回广义逆高斯分布的概率密度函数值
    if x > 0:
        return math.exp((p - 1) * math.log(x) - 0.5 * b * (x + 1 / x) - lfm)
    # 否则返回 0.0
    return 0.0


# 定义一个概率密度函数，描述逆高斯分布的众数
def invgauss_mode(mu):
    # 返回逆高斯分布的众数
    return 1.0 / (math.sqrt(1.5 * 1.5 + 1 / (mu * mu)) + 1.5)


# 定义一个概率密度函数，描述逆高斯分布的概率密度函数
def invgauss_pdf(x, mu):
    # 计算逆高斯分布的众数
    m = invgauss_mode(mu)
    # 计算逆高斯分布的概率密度函数的对数值
    lfm = -1.5 * math.log(m) - (m - mu) ** 2 / (2 * m * mu**2)
    # 如果 x 大于 0，返回逆高斯分布的概率密度函数值
    if x > 0:
        return math.exp(-1.5 * math.log(x) - (x - mu) ** 2 / (2 * x * mu**2) - lfm)
    # 否则返回 0.0
    return 0.0


# 定义一个概率密度函数，描述幂律分布的概率密度函数
def powerlaw_pdf(x, a):
    # 如果 x 大于 0，返回幂律分布的概率密度函数值
    if x > 0:
        return x ** (a - 1)
    # 否则返回 0.0
    return 0.0


# 定义一个字典，为给定的分布（键），另一个字典（值）指定 NumericalInversePolynomial（PINV）的参数
# 其中，后者字典的键包括：
# - pdf: 分布的概率密度函数（可调用）。概率密度函数的签名为 float -> float
#   （即，该函数不需要矢量化）。如果可能，应优先使用 math 模块的 log 或 exp 函数，
#   而不是 numpy 中的函数，因为这样设置 PINV 将更快。
# - check_pinv_params: 可调用函数 f，如果形状参数（args）推荐用于 PINV，则返回 true
#   （即，u-误差不超过默认容差）
# - center: 如果中心不依赖于 args，则为标量；否则为可调用函数，根据形状参数返回中心
# - rvs_transform: 可调用函数，用于将根据概率密度函数分布的随机变量变换为目标分布
# - rvs_transform_inv: rvs_transform 的逆函数（在转换的 ppf 中需要）
# - mirror_uniform: 布尔值或可调用函数，根据形状参数返回 true 或 false。
#   如果为 True，则将 ppf 应用于 1-u 而不是 u，以生成随机变量（其中 u 是均匀分布的随机变量）。
#   虽然 u 和 1-u 都是均匀分布，但计算 u-误差时可能需要使用 1-u 正确。
#   这仅与 argus 分布相关。
# 定义一个配置字典 PINV_CONFIG，包含不同概率分布的参数和函数
PINV_CONFIG = {
    # alpha 分布的配置
    "alpha": {
        "pdf": alpha_pdf,  # 概率密度函数
        "check_pinv_params": lambda a: 1.0e-11 <= a < 2.1e5,  # 参数检查函数
        "center": lambda a: 0.25 * (math.sqrt(a * a + 8.0) - a),  # 中心计算函数
    },
    # anglit 分布的配置
    "anglit": {
        "pdf": lambda x: math.cos(2 * x) + 1.0e-13,  # 概率密度函数，包含修正项
        # f(upper border) is very close to 0，修正因素
        "center": 0,  # 中心值
    },
    # argus 分布的配置
    "argus": {
        "pdf": argus_pdf,  # 概率密度函数
        "center": lambda chi: 0.7 if chi <= 5 else 0.5,  # 中心计算函数
        "check_pinv_params": lambda chi: 1e-20 < chi < 901,  # 参数检查函数
        "rvs_transform": argus_gamma_trf,  # 随机变换函数
        "rvs_transform_inv": argus_gamma_inv_trf,  # 随机变换的逆函数
        "mirror_uniform": lambda chi: chi > 5,  # 条件函数
    },
    # beta 分布的配置
    "beta": {
        "pdf": betaprime_pdf,  # 概率密度函数
        "center": lambda a, b: max(0.1, (a - 1) / (b + 1)),  # 中心计算函数
        "check_pinv_params": beta_valid_params,  # 参数检查函数
        "rvs_transform": lambda x, *args: x / (1 + x),  # 随机变换函数
        "rvs_transform_inv": lambda x, *args: x / (1 - x) if x < 1 else np.inf,  # 随机变换的逆函数
    },
    # betaprime 分布的配置
    "betaprime": {
        "pdf": betaprime_pdf,  # 概率密度函数
        "center": lambda a, b: max(0.1, (a - 1) / (b + 1)),  # 中心计算函数
        "check_pinv_params": beta_valid_params,  # 参数检查函数
    },
    # bradford 分布的配置
    "bradford": {
        "pdf": bradford_pdf,  # 概率密度函数
        "check_pinv_params": lambda a: 1.0e-6 <= a <= 1e9,  # 参数检查函数
        "center": 0.5,  # 中心值
    },
    # burr 分布的配置
    "burr": {
        "pdf": burr_pdf,  # 概率密度函数
        "center": lambda a, b: (2 ** (1 / b) - 1) ** (-1 / a),  # 中心计算函数
        "check_pinv_params": lambda a, b: (min(a, b) >= 0.3) and (max(a, b) <= 50),  # 参数检查函数
    },
    # burr12 分布的配置
    "burr12": {
        "pdf": burr12_pdf,  # 概率密度函数
        "center": lambda a, b: (2 ** (1 / b) - 1) ** (1 / a),  # 中心计算函数
        "check_pinv_params": lambda a, b: (min(a, b) >= 0.2) and (max(a, b) <= 50),  # 参数检查函数
    },
    # cauchy 分布的配置
    "cauchy": {
        "pdf": lambda x: 1 / (1 + (x * x)),  # 概率密度函数
        "center": 0,  # 中心值
    },
    # chi 分布的配置
    "chi": {
        "pdf": chi_pdf,  # 概率密度函数
        "check_pinv_params": lambda df: 0.05 <= df <= 1.0e6,  # 参数检查函数
        "center": lambda a: math.sqrt(a),  # 中心计算函数
    },
    # chi2 分布的配置
    "chi2": {
        "pdf": chi2_pdf,  # 概率密度函数
        "check_pinv_params": lambda df: 0.07 <= df <= 1e6,  # 参数检查函数
        "center": lambda a: a,  # 中心计算函数
    },
    # cosine 分布的配置
    "cosine": {
        "pdf": lambda x: 1 + math.cos(x),  # 概率密度函数
        "center": 0,  # 中心值
    },
    # crystalball 分布的配置
    "crystalball": {
        "pdf": crystalball_pdf,  # 概率密度函数
        "check_pinv_params": lambda b, m: (0.01 <= b <= 5.5) and (1.1 <= m <= 75.1),  # 参数检查函数
        "center": 0.0,  # 中心值
    },
    # expon 分布的配置
    "expon": {
        "pdf": lambda x: math.exp(-x),  # 概率密度函数
        "center": 1.0,  # 中心值
    },
    # gamma 分布的配置
    "gamma": {
        "pdf": gamma_pdf,  # 概率密度函数
        "check_pinv_params": lambda a: 0.04 <= a <= 1e6,  # 参数检查函数
        "center": lambda a: a,  # 中心计算函数
    },
    # gennorm 分布的配置
    "gennorm": {
        "pdf": lambda x, b: math.exp(-abs(x) ** b),  # 概率密度函数
        "check_pinv_params": lambda b: 0.081 <= b <= 45.0,  # 参数检查函数
        "center": 0.0,  # 中心值
    },
    # geninvgauss 分布的配置
    "geninvgauss": {
        "pdf": geninvgauss_pdf,  # 概率密度函数
        "check_pinv_params": lambda p, b: (abs(p) <= 1200.0) and (1.0e-10 <= b <= 1200.0),  # 参数检查函数
        "center": geninvgauss_mode,  # 中心值
    },
}
    "gumbel_l": {
        # 定义逆Gumbel（左）分布的概率密度函数
        "pdf": lambda x: math.exp(x - math.exp(x)),
        # 分布的中心值
        "center": -0.6,
    },
    "gumbel_r": {
        # 定义逆Gumbel（右）分布的概率密度函数
        "pdf": lambda x: math.exp(-x - math.exp(-x)),
        # 分布的中心值
        "center": 0.6,
    },
    "hypsecant": {
        # 定义双曲正切分布的概率密度函数
        "pdf": lambda x: 1.0 / (math.exp(x) + math.exp(-x)),
        # 分布的中心值
        "center": 0.0,
    },
    "invgamma": {
        # 使用给定的函数计算逆Gamma分布的概率密度函数
        "pdf": invgamma_pdf,
        # 检查参数是否满足逆Gamma分布的条件
        "check_pinv_params": lambda a: 0.04 <= a <= 1e6,
        # 分布的中心值函数
        "center": lambda a: 1 / a,
    },
    "invgauss": {
        # 使用给定的函数计算逆高斯分布的概率密度函数
        "pdf": invgauss_pdf,
        # 检查参数是否满足逆高斯分布的条件
        "check_pinv_params": lambda mu: 1.0e-10 <= mu <= 1.0e9,
        # 分布的众数（最高点）
        "center": invgauss_mode,
    },
    "invweibull": {
        # 使用给定的函数计算逆威布尔分布的概率密度函数
        "pdf": invweibull_pdf,
        # 检查参数是否满足逆威布尔分布的条件
        "check_pinv_params": lambda a: 0.12 <= a <= 512,
        # 分布的中心值
        "center": 1.0,
    },
    "laplace": {
        # 定义拉普拉斯分布的概率密度函数
        "pdf": lambda x: math.exp(-abs(x)),
        # 分布的中心值
        "center": 0.0,
    },
    "logistic": {
        # 定义逻辑斯蒂分布的概率密度函数
        "pdf": lambda x: math.exp(-x) / (1 + math.exp(-x)) ** 2,
        # 分布的中心值
        "center": 0.0,
    },
    "maxwell": {
        # 定义麦克斯韦分布的概率密度函数
        "pdf": lambda x: x * x * math.exp(-0.5 * x * x),
        # 分布的中心值
        "center": 1.41421,
    },
    "moyal": {
        # 定义Moyal分布的概率密度函数
        "pdf": lambda x: math.exp(-(x + math.exp(-x)) / 2),
        # 分布的中心值
        "center": 1.2,
    },
    "norm": {
        # 定义正态分布的概率密度函数
        "pdf": lambda x: math.exp(-x * x / 2),
        # 分布的中心值
        "center": 0.0,
    },
    "pareto": {
        # 定义帕累托分布的概率密度函数，带参数b
        "pdf": lambda x, b: x ** -(b + 1),
        # 计算分布的中心值函数，根据参数b的值决定返回值
        "center": lambda b: b / (b - 1) if b > 2 else 1.5,
        # 检查参数是否满足帕累托分布的条件
        "check_pinv_params": lambda b: 0.08 <= b <= 400000,
    },
    "powerlaw": {
        # 使用给定的函数计算幂律分布的概率密度函数
        "pdf": powerlaw_pdf,
        # 分布的中心值
        "center": 1.0,
        # 检查参数是否满足幂律分布的条件
        "check_pinv_params": lambda a: 0.06 <= a <= 1.0e5,
    },
    "t": {
        # 定义t分布的概率密度函数，带参数df
        "pdf": lambda x, df: (1 + x * x / df) ** (-0.5 * (df + 1)),
        # 检查参数是否满足t分布的条件
        "check_pinv_params": lambda a: 0.07 <= a <= 1e6,
        # 分布的中心值
        "center": 0.0,
    },
    "rayleigh": {
        # 定义Rayleigh分布的概率密度函数
        "pdf": lambda x: x * math.exp(-0.5 * (x * x)),
        # 分布的中心值
        "center": 1.0,
    },
    "semicircular": {
        # 定义半圆分布的概率密度函数
        "pdf": lambda x: math.sqrt(1.0 - (x * x)),
        # 分布的中心值
        "center": 0,
    },
    "wald": {
        # 使用给定的函数计算Wald分布的概率密度函数
        "pdf": wald_pdf,
        # 分布的中心值
        "center": 1.0,
    },
    "weibull_max": {
        # 使用给定的函数计算最大威布尔分布的概率密度函数
        "pdf": weibull_max_pdf,
        # 检查参数是否满足最大威布尔分布的条件
        "check_pinv_params": lambda a: 0.25 <= a <= 512,
        # 分布的中心值
        "center": -1.0,
    },
    "weibull_min": {
        # 使用给定的函数计算最小威布尔分布的概率密度函数
        "pdf": weibull_min_pdf,
        # 检查参数是否满足最小威布尔分布的条件
        "check_pinv_params": lambda a: 0.25 <= a <= 512,
        # 分布的中心值
        "center": 1.0,
    },
}

# 对输入的 `qmc_engine` 和 `d` 进行验证
def _validate_qmc_input(qmc_engine, d, seed):
    # 如果 `qmc_engine` 是 QMCEngine 的实例
    if isinstance(qmc_engine, QMCEngine):
        # 如果 `d` 不为 None 且与 `qmc_engine` 的维度不一致，则引发错误
        if d is not None and qmc_engine.d != d:
            message = "`d` must be consistent with dimension of `qmc_engine`."
            raise ValueError(message)
        # 如果 `d` 为 None，则使用 `qmc_engine` 的维度
        d = qmc_engine.d if d is None else d
    # 如果 `qmc_engine` 是 None
    elif qmc_engine is None:
        # 如果 `d` 为 None，则设定 `d` 为 1；否则使用给定的 `d`
        d = 1 if d is None else d
        # 使用 Halton 序列作为 `qmc_engine`，使用给定的 `seed`
        qmc_engine = Halton(d, seed=seed)
    else:
        # 如果 `qmc_engine` 不是 QMCEngine 的实例，引发 ValueError 错误
        message = (
            "`qmc_engine` must be an instance of "
            "`scipy.stats.qmc.QMCEngine` or `None`."
        )
        raise ValueError(message)

    return qmc_engine, d


class CustomDistPINV:
    def __init__(self, pdf, args):
        # 初始化函数，使用给定的概率密度函数 `pdf` 和参数 `args`
        self._pdf = lambda x: pdf(x, *args)

    def pdf(self, x):
        # 返回概率密度函数 `pdf` 在 `x` 处的值
        return self._pdf(x)


class FastGeneratorInversion:
    """
    通过数值反演 `scipy.stats` 中大类连续分布的累积分布函数进行快速抽样。

    Parameters
    ----------
    dist : rv_frozen object
        `scipy.stats` 中的冻结分布对象。支持的分布列表可参见注释。用于创建分布的形状参数 `loc` 和 `scale` 必须是标量。
        例如，对于形状参数为 `p` 的 Gamma 分布，`p` 必须是浮点数；对于形状参数为 (a, b) 的 beta 分布，a 和 b 都必须是浮点数。
    domain : tuple of floats, optional
        如果希望从截断/条件分布中抽样，需要指定域。
        默认为 None。在这种情况下，随机变量不被截断，并且域从分布的支持中推断出来。
    ignore_shape_range : boolean, optional
        如果为 False，则形状参数超出有效范围的值将引发 ValueError 以确保数值精度高（见注释）。
        如果为 True，则接受任何分布的有效形状参数值。这对于测试很有用。
        默认为 False。
    random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional
        一个 NumPy 随机数生成器或用于生成均匀随机数流的基础 NumPy 随机数生成器的种子。
        如果 `random_state` 为 None，则使用 `self.random_state`。
        如果 `random_state` 是一个整数，则使用 ``np.random.default_rng(random_state)``。
        如果 `random_state` 已经是 ``Generator`` 或 ``RandomState`` 实例，则使用该实例。

    Attributes
    ----------
    loc : float
        位置参数。
    ```
    random_state : {`numpy.random.Generator`, `numpy.random.RandomState`}
        随机数生成器的状态参数，可以是 `numpy.random.Generator` 或 `numpy.random.RandomState`

    scale : float
        缩放参数，用于调整分布的尺度。

    Methods
    -------
    cdf
        累积分布函数（CDF），用于计算随机变量的累积概率分布。
    evaluate_error
        评估误差的方法，用于检查数值精度。
    ppf
        百分点函数（PPF），是CDF的逆函数，用于生成指定累积概率下的变量值。
    qrvs
        使用准随机数生成的随机变量生成方法。
    rvs
        随机变量生成方法，根据指定分布生成随机样本。
    support
        分布支持范围的方法。

    Notes
    -----
    该类用于创建连续分布的对象，由参数 `dist` 指定。方法 `rvs` 使用从 `scipy.stats.sampling` 创建的生成器，
    在对象实例化时创建。此外，还添加了方法 `qrvs` 和 `ppf`。
    `qrvs` 基于 `scipy.stats.qmc` 中的准随机数生成样本。
    `ppf` 基于文献 [1]_ 中的数值反转多项式方法（NumericalInversePolynomial），
    用于生成随机变量。

    支持的分布（`distname`）包括：
    ``alpha``, ``anglit``, ``argus``, ``beta``, ``betaprime``, ``bradford``,
    ``burr``, ``burr12``, ``cauchy``, ``chi``, ``chi2``, ``cosine``,
    ``crystalball``, ``expon``, ``gamma``, ``gennorm``, ``geninvgauss``,
    ``gumbel_l``, ``gumbel_r``, ``hypsecant``, ``invgamma``, ``invgauss``,
    ``invweibull``, ``laplace``, ``logistic``, ``maxwell``, ``moyal``,
    ``norm``, ``pareto``, ``powerlaw``, ``t``, ``rayleigh``, ``semicircular``,
    ``wald``, ``weibull_max``, ``weibull_min``.

    `rvs` 方法依赖于数值反转的准确性。如果使用极端形状参数，则数值反转可能无法工作。
    然而，对于所有实现的分布，已经测试了可接受的形状参数，并且如果用户提供超出允许范围的值，将会引发错误。
    所有有效参数的 u-误差不应超过 1e-10。请注意，即使在实例化对象时参数在有效范围内，也可能会引发警告。
    可以使用 `evaluate_error` 方法检查数值精度。

    所有实现的分布也是 `scipy.stats` 的一部分，由 `FastGeneratorInversion` 创建的对象依赖于 `rv_frozen` 的 `ppf`、`cdf` 和 `pdf` 方法。
    使用此类的主要优点可以总结如下：一旦在设置步骤中创建了用于采样随机变量的生成器，
    使用 `ppf` 进行采样和评估 PPF 是非常快速的，性能基本与分布无关。
    因此，如果需要大量随机变量，可以实现显著的加速。重要的是要知道，这种快速采样是通过反转累积分布函数（CDF）实现的。
    因此，一个均匀随机变量被转换为一个非均匀变量，这对于多种模拟方法是一个优势，例如当使用常见随机变量的方差减少方法或
    对偶变量时 ([2]_)。
    # Import necessary modules from scipy.stats for handling quasi-random number generation
    >>> import numpy as np
    >>> from scipy import stats
    >>> from scipy.stats.sampling import FastGeneratorInversion

    # Define a frozen gamma distribution with shape parameter 1.5 for random variate generation
    >>> gamma_frozen = stats.gamma(1.5)
    
    # Create a FastGeneratorInversion object based on the gamma distribution
    >>> gamma_dist = FastGeneratorInversion(gamma_frozen)
    
    # Generate 1000 random variates using the defined distribution
    >>> r = gamma_dist.rvs(size=1000)

    # Calculate the mean of the generated random variates, expected to be approximately 1.5
    >>> r.mean()
    1.52423591130436  # may vary

    # Generate 1000 quasi-random variates using the inverse transform method
    >>> r = gamma_dist.qrvs(size=1000)
    
    # Calculate the mean of the quasi-random variates
    >>> r.mean()
    1.4996639255942914  # may vary

    # Compare the percent point function (PPF) values between the original gamma distribution and the inverted one
    >>> q = [0.001, 0.2, 0.5, 0.8, 0.999]
    >>> np.max(np.abs(gamma_frozen.ppf(q) - gamma_dist.ppf(q)))
    4.313394796895409e-08

    # Evaluate the approximation error (u-error) of the numerical inversion method
    >>> gamma_dist.evaluate_error()
    (7.446320551265581e-11, nan)  # may vary

    # Modify the location and scale parameters of the distribution without creating a new generator
    >>> gamma_dist.loc = 2
    >>> gamma_dist.scale = 3
    
    # Generate 1000 random variates with the modified distribution parameters
    >>> r = gamma_dist.rvs(size=1000)

    # Calculate the mean of the modified random variates, expected to be approximately 6.5
    >>> r.mean()
    6.399549295242894  # may vary

    # Illustrate truncation of a normal distribution to the interval (3, 4)
    >>> trunc_norm = FastGeneratorInversion(stats.norm(), domain=(3, 4))
    
    # Generate 1000 truncated random variates from the normal distribution
    >>> r = trunc_norm.rvs(size=1000)
    
    # Check if all generated values lie within the specified interval
    >>> 3 < r.min() < r.max() < 4
    True

    # Calculate the mean of the truncated normal variates
    >>> r.mean()
    3.250433367078603  # may vary

    # Calculate the expected value using the conditional expectation function of the normal distribution
    >>> stats.norm.expect(lb=3, ub=4, conditional=True)
    3.260454285589997
    In this particular, case, `scipy.stats.truncnorm` could also be used to
    generate truncated normal random variates.

    """

    # 初始化函数，接受参数 dist（分布），domain（定义域），ignore_shape_range（是否忽略形状范围），random_state（随机状态）
    def __init__(
        self,
        dist,
        *,
        domain=None,
        ignore_shape_range=False,
        random_state=None,
    ):
        # 设置随机状态属性
        @property
        def random_state(self):
            return self._random_state

        # 设置随机状态的 setter 方法，用于验证并设置随机状态
        @random_state.setter
        def random_state(self, random_state):
            self._random_state = check_random_state_qmc(random_state)

        # 获取 loc 属性的值
        @property
        def loc(self):
            return self._frozendist.kwds.get("loc", 0)

        # 设置 loc 属性的 setter 方法，用于验证并设置 loc
        @loc.setter
        def loc(self, loc):
            if not np.isscalar(loc):
                raise ValueError("loc must be scalar.")
            self._frozendist.kwds["loc"] = loc
            # 更新依赖于 loc 和 scale 的调整后的定义域
            self._set_domain_adj()

        # 获取 scale 属性的值
        @property
        def scale(self):
            return self._frozendist.kwds.get("scale", 0)

        # 设置 scale 属性的 setter 方法，用于验证并设置 scale
        @scale.setter
        def scale(self, scale):
            if not np.isscalar(scale):
                raise ValueError("scale must be scalar.")
            self._frozendist.kwds["scale"] = scale
            # 更新依赖于 loc 和 scale 的调整后的定义域
            self._set_domain_adj()

        # 根据 loc 和 scale 调整定义域的内部方法
        def _set_domain_adj(self):
            """ Adjust the domain based on loc and scale. """
            loc = self.loc
            scale = self.scale
            lb = self._domain[0] * scale + loc
            ub = self._domain[1] * scale + loc
            self._domain_adj = (lb, ub)

        # 处理配置信息的内部方法，接受分布名称和参数 args
        def _process_config(self, distname, args):
            # 获取指定分布名称的配置信息
            cfg = PINV_CONFIG[distname]

            # 如果配置中包含检查逆参数的方法，并且不忽略形状范围
            if "check_pinv_params" in cfg:
                if not self._ignore_shape_range:
                    # 如果参数不符合配置要求，则抛出 ValueError 异常
                    if not cfg["check_pinv_params"](*args):
                        msg = ("No generator is defined for the shape parameters "
                               f"{args}. Use ignore_shape_range to proceed "
                               "with the selected values.")
                        raise ValueError(msg)

            # 如果配置中包含 center 属性
            if "center" in cfg.keys():
                # 如果 center 不是标量，则计算它并赋值给 self._center
                if not np.isscalar(cfg["center"]):
                    self._center = cfg["center"](*args)
                else:
                    self._center = cfg["center"]
            else:
                self._center = None

            # 获取配置中的随机变量转换和其逆转换方法
            self._rvs_transform = cfg.get("rvs_transform", None)
            self._rvs_transform_inv = cfg.get("rvs_transform_inv", None)

            # 获取镜像均匀分布的配置信息
            _mirror_uniform = cfg.get("mirror_uniform", None)
            if _mirror_uniform is None:
                self._mirror_uniform = False
            else:
                self._mirror_uniform = _mirror_uniform(*args)

            # 返回根据配置信息生成的自定义分布 PINV 对象
            return CustomDistPINV(cfg["pdf"], args)
    def rvs(self, size=None):
        """
        Sample from the distribution by inversion.

        Parameters
        ----------
        size : int or tuple, optional
            The shape of samples. Default is ``None`` in which case a scalar
            sample is returned.

        Returns
        -------
        rvs : array_like
            A NumPy array of random variates.

        Notes
        -----
        Random variates are generated by numerical inversion of the CDF, i.e.,
        `ppf` computed by `NumericalInversePolynomial` when the class
        is instantiated. Note that the
        default ``rvs`` method of the rv_continuous class is
        overwritten. Hence, a different stream of random numbers is generated
        even if the same seed is used.
        """
        # note: we cannot use self._rng.rvs directly in case
        # self._mirror_uniform is true
        # 根据设定的 size 参数，生成均匀分布的随机数 u
        u = self.random_state.uniform(size=size)
        # 如果设定了镜像均匀分布标志，则对 u 进行镜像操作
        if self._mirror_uniform:
            u = 1 - u
        # 根据 u 值通过反分布函数 PPF 计算随机变量 r
        r = self._rng.ppf(u)
        # 如果定义了变换函数，则对 r 进行变换
        if self._rvs_transform is not None:
            r = self._rvs_transform(r, *self._frozendist.args)
        # 返回按指定参数 loc 和 scale 转换后的随机变量 r
        return self.loc + self.scale * r

    def ppf(self, q):
        """
        Very fast PPF (inverse CDF) of the distribution which
        is a very close approximation of the exact PPF values.

        Parameters
        ----------
        q : array_like
            Array with probabilities.

        Returns
        -------
        ppf : array_like
            Quantiles corresponding to the values in `q`.

        Notes
        -----
        The evaluation of the PPF is very fast but it may have a large
        relative error in the far tails. The numerical precision of the PPF
        is controlled by the q-error, that is,
        ``max |q - CDF(PPF(q))|`` where the max is taken over points in
        the interval [0,1], see `evaluate_error`.

        Note that this PPF is designed to generate random samples.
        """
        # 将 q 转换为 NumPy 数组以便处理
        q = np.asarray(q)
        # 如果设定了镜像均匀分布标志，则对 1 - q 进行反分布函数 PPF 计算
        if self._mirror_uniform:
            x = self._rng.ppf(1 - q)
        else:
            # 否则，对 q 进行反分布函数 PPF 计算
            x = self._rng.ppf(q)
        # 如果定义了变换函数，则对 x 进行变换
        if self._rvs_transform is not None:
            x = self._rvs_transform(x, *self._frozendist.args)
        # 返回按指定参数 loc 和 scale 转换后的 x 值
        return self.scale * x + self.loc
    def support(self):
        """Distribution的支持范围。

        Returns
        -------
        a, b : float
            分布的支持范围的端点。

        Notes
        -----

        注意，分布的支持范围取决于 `loc`、`scale` 和 `domain`。

        Examples
        --------

        >>> from scipy import stats
        >>> from scipy.stats.sampling import FastGeneratorInversion

        定义一个截断正态分布：

        >>> d_norm = FastGeneratorInversion(stats.norm(), domain=(0, 1))
        >>> d_norm.support()
        (0, 1)

        移动分布的位置：

        >>> d_norm.loc = 2.5
        >>> d_norm.support()
        (2.5, 3.5)

        """
        return self._domain_adj

    def _cdf(self, x):
        """累积分布函数（CDF）

        Parameters
        ----------
        x : array_like
            待评估的值

        Returns
        -------
        y : ndarray
            在 x 处评估的 CDF

        """
        y = self._frozendist.cdf(x)
        if self._p_domain == 1.0:
            return y
        return np.clip((y - self._p_lower) / self._p_domain, 0, 1)

    def _ppf(self, q):
        """百分位点函数（CDF 的反函数）

        Parameters
        ----------
        q : array_like
            下尾概率

        Returns
        -------
        x : array_like
            对应于下尾概率 q 的分位数。

        """
        if self._p_domain == 1.0:
            return self._frozendist.ppf(q)
        x = self._frozendist.ppf(self._p_domain * np.array(q) + self._p_lower)
        return np.clip(x, self._domain_adj[0], self._domain_adj[1])
class RatioUniforms:
    """
    Generate random samples from a probability density function using the
    ratio-of-uniforms method.

    Parameters
    ----------
    pdf : callable
        A function with signature `pdf(x)` that is proportional to the
        probability density function of the distribution.
    umax : float
        The upper bound of the bounding rectangle in the u-direction.
    vmin : float
        The lower bound of the bounding rectangle in the v-direction.
    vmax : float
        The upper bound of the bounding rectangle in the v-direction.
    c : float, optional.
        Shift parameter of ratio-of-uniforms method, see Notes. Default is 0.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Methods
    -------
    rvs
        Generates random samples using the ratio-of-uniforms method.

    Notes
    -----
    Given a univariate probability density function `pdf` and a constant `c`,
    define the set ``A = {(u, v) : 0 < u <= sqrt(pdf(v/u + c))}``.
    If ``(U, V)`` is a random vector uniformly distributed over ``A``,
    then ``V/U + c`` follows a distribution according to `pdf`.

    The above result (see [1]_, [2]_) can be used to sample random variables
    using only the PDF, i.e. no inversion of the CDF is required. Typical
    choices of `c` are zero or the mode of `pdf`. The set ``A`` is a subset of
    the rectangle ``R = [0, umax] x [vmin, vmax]`` where

    - ``umax = sup sqrt(pdf(x))``
    - ``vmin = inf (x - c) sqrt(pdf(x))``
    - ``vmax = sup (x - c) sqrt(pdf(x))``

    In particular, these values are finite if `pdf` is bounded and
    ``x**2 * pdf(x)`` is bounded (i.e. subquadratic tails).
    One can generate ``(U, V)`` uniformly on ``R`` and return
    ``V/U + c`` if ``(U, V)`` are also in ``A`` which can be directly
    verified.

    The algorithm is not changed if one replaces `pdf` by k * `pdf` for any
    constant k > 0. Thus, it is often convenient to work with a function
    that is proportional to the probability density function by dropping
    unnecessary normalization factors.

    Intuitively, the method works well if ``A`` fills up most of the
    enclosing rectangle such that the probability is high that ``(U, V)``
    lies in ``A`` whenever it lies in ``R`` as the number of required
    iterations becomes too large otherwise. To be more precise, note that
    the expected number of iterations to draw ``(U, V)`` uniformly
    distributed on ``R`` such that ``(U, V)`` is also in ``A`` is given by
    the ratio ``area(R) / area(A) = 2 * umax * (vmax - vmin) / area(pdf)``,
    where `area(pdf)` is the integral of `pdf` (which is equal to one if the
    PDF is properly normalized).
    """
    
    def __init__(self, pdf, umax, vmin, vmax, c=0, random_state=None):
        """
        Initialize the RatioUniforms object.

        Parameters
        ----------
        pdf : callable
            Probability density function proportional to `pdf(x)`.
        umax : float
            Upper bound of the bounding rectangle in the u-direction.
        vmin : float
            Lower bound of the bounding rectangle in the v-direction.
        vmax : float
            Upper bound of the bounding rectangle in the v-direction.
        c : float, optional
            Shift parameter for the ratio-of-uniforms method.
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional
            Random state for reproducibility.
        """
        self.pdf = pdf
        self.umax = umax
        self.vmin = vmin
        self.vmax = vmax
        self.c = c
        self.random_state = random_state
    
    def rvs(self, size=None):
        """
        Generate random samples using the ratio-of-uniforms method.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape. If None, a single sample is returned.

        Returns
        -------
        ndarray or scalar
            Random samples from the distribution.
        """
        # Implementation of the ratio-of-uniforms method to generate samples
        # from the given probability density function `pdf`.
        pass
    """
    Initialize a random variate generator for a distribution proportional to
    the given probability density function (pdf), using the Ratio of Uniforms
    method.

    Parameters
    ----------
    pdf : callable
        The probability density function of the distribution.
    umax : float
        Upper bound of the uniform distribution used in the ratio method.
    vmin : float
        Minimum value for the support of the distribution.
    vmax : float
        Maximum value for the support of the distribution.
    c : float, optional
        Location parameter (default is 0).
    random_state : int, RandomState instance or None, optional
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Raises
    ------
    ValueError
        If `vmin` is not less than `vmax`.
        If `umax` is non-positive.

    Notes
    -----
    This initializer sets up the parameters needed for generating random
    variates from a distribution proportional to `pdf` using the Ratio of
    Uniforms method. It checks that the bounding rectangle [vmin, vmax] is
    correctly specified to ensure proper sampling from the given `pdf`.

    References
    ----------
    .. [1] L. Devroye, "Non-Uniform Random Variate Generation",
       Springer-Verlag, 1986.

    .. [2] W. Hoermann and J. Leydold, "Generating generalized inverse Gaussian
       random variates", Statistics and Computing, 24(4), p. 547--557, 2014.

    .. [3] A.J. Kinderman and J.F. Monahan, "Computer Generation of Random
       Variables Using the Ratio of Uniform Deviates",
       ACM Transactions on Mathematical Software, 3(3), p. 257--260, 1977.
    """
    
    def __init__(self, pdf, *, umax, vmin, vmax, c=0, random_state=None):
        # Check if vmin is less than vmax, ensuring a valid bounding rectangle
        if vmin >= vmax:
            raise ValueError("vmin must be smaller than vmax.")

        # Check if umax is positive, as it determines the upper bound for the uniform distribution
        if umax <= 0:
            raise ValueError("umax must be positive.")
        
        # Initialize instance variables based on input parameters
        self._pdf = pdf
        self._umax = umax
        self._vmin = vmin
        self._vmax = vmax
        self._c = c
        # Set up the random number generator using the provided random_state
        self._rng = check_random_state(random_state)
    def rvs(self, size=1):
        """Sampling of random variates
        
        Parameters
        ----------
        size : int or tuple of ints, optional
            Number of random variates to be generated (default is 1).
        
        Returns
        -------
        rvs : ndarray
            The random variates distributed according to the probability
            distribution defined by the pdf.
        
        """
        # Convert size to at least 1-dimensional tuple
        size1d = tuple(np.atleast_1d(size))
        # Calculate the total number of random variates needed
        N = np.prod(size1d)  # number of rvs needed, reshape upon return
        
        # start sampling using ratio of uniforms method
        x = np.zeros(N)
        simulated, i = 0, 1
        
        # loop until N rvs have been generated: expected runtime is finite.
        # to avoid infinite loop, raise exception if not a single rv has been
        # generated after 50000 tries. even if the expected number of iterations
        # is 1000, the probability of this event is (1-1/1000)**50000
        # which is of order 10e-22
        while simulated < N:
            k = N - simulated
            # simulate uniform rvs on [0, umax] and [vmin, vmax]
            u1 = self._umax * self._rng.uniform(size=k)
            v1 = self._rng.uniform(self._vmin, self._vmax, size=k)
            # apply rejection method
            rvs = v1 / u1 + self._c
            accept = (u1**2 <= self._pdf(rvs))
            num_accept = np.sum(accept)
            if num_accept > 0:
                # Store accepted random variates in x
                x[simulated:(simulated + num_accept)] = rvs[accept]
                simulated += num_accept
            
            # Check if no random variate has been generated and raise exception if
            # criteria are met
            if (simulated == 0) and (i*N >= 50000):
                msg = (
                    f"Not a single random variate could be generated in {i*N} "
                    "attempts. The ratio of uniforms method does not appear "
                    "to work for the provided parameters. Please check the "
                    "pdf and the bounds."
                )
                raise RuntimeError(msg)
            i += 1
        
        # Reshape the array of random variates to the original size requested
        return np.reshape(x, size1d)
```