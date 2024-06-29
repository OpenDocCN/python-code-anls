# `.\numpy\numpy\random\__init__.py`

```
"""
========================
Random Number Generation
========================

Use ``default_rng()`` to create a `Generator` and call its methods.

=============== =========================================================
Generator
--------------- ---------------------------------------------------------
Generator       实现所有随机数分布的类
default_rng     ``Generator`` 的默认构造函数
=============== =========================================================

============================================= ===
BitGenerator Streams that work with Generator
--------------------------------------------- ---
MT19937
PCG64
PCG64DXSM
Philox
SFC64
============================================= ===

============================================= ===
Getting entropy to initialize a BitGenerator
--------------------------------------------- ---
SeedSequence
============================================= ===


Legacy
------

For backwards compatibility with previous versions of numpy before 1.17, the
various aliases to the global `RandomState` methods are left alone and do not
use the new `Generator` API.

==================== =========================================================
Utility functions
-------------------- ---------------------------------------------------------
random               ``[0, 1)`` 区间上均匀分布的浮点数
bytes                均匀分布的随机字节
permutation          随机排列序列 / 生成随机序列
shuffle              原地随机排列序列
choice               从一维数组中随机抽样
==================== =========================================================

==================== =========================================================
Compatibility
functions - removed
in the new API
-------------------- ---------------------------------------------------------
rand                 均匀分布的随机值
randn                正态分布的随机值
ranf                 均匀分布的浮点数
random_integers      给定范围内均匀分布的整数（已弃用，请使用 ``integers(..., closed=True)``）
random_sample        `random_sample` 的别名
randint              给定范围内均匀分布的整数
seed                 种子以初始化旧版本的随机数生成器
==================== =========================================================

==================== =========================================================
Univariate
distributions
-------------------- ---------------------------------------------------------
beta                 ``[0, 1]`` 区间上的 Beta 分布
binomial             二项分布
chisquare            :math:`\\chi^2` 分布
exponential          指数分布
f                    F 分布（Fisher-Snedecor 分布）
gamma                Gamma 分布
# geometric            几何分布。
# gumbel               甘贝尔分布。
# hypergeometric       超几何分布。
# laplace              拉普拉斯分布。
# logistic             逻辑斯蒂分布。
# lognormal            对数正态分布。
# logseries            对数级数分布。
# negative_binomial    负二项分布。
# noncentral_chisquare 非中心卡方分布。
# noncentral_f         非中心 F 分布。
# normal               正态分布。
# pareto               帕累托分布。
# poisson              泊松分布。
# power                幂分布。
# rayleigh             瑞利分布。
# triangular           三角分布。
# uniform              均匀分布。
# vonmises             冯·米塞斯分布（圆形分布）。
# wald                 瓦尔德分布（反高斯分布）。
# weibull              威布尔分布。
# zipf                 费希尔分布（用于排名数据的分布）。
==================== =========================================================

==================== ==========================================================
Multivariate
distributions
-------------------- ----------------------------------------------------------
# dirichlet            贝塔分布的多元化泛化。
# multinomial          二项分布的多元化泛化。
# multivariate_normal  正态分布的多元化泛化。
==================== ==========================================================

==================== =========================================================
Standard
distributions
-------------------- ---------------------------------------------------------
# standard_cauchy      标准柯西-洛伦兹分布。
# standard_exponential 标准指数分布。
# standard_gamma       标准伽马分布。
# standard_normal      标准正态分布。
# standard_t           标准学生 t 分布。
==================== =========================================================

==================== =========================================================
Internal functions
-------------------- ---------------------------------------------------------
# get_state            获取生成器内部状态的元组表示。
# set_state            设置生成器的状态。
==================== =========================================================
    'randn',                    # 从标准正态分布中抽取随机样本
    'random',                   # 生成一个0到1之间的随机浮点数
    'random_integers',          # 生成一个指定范围内的随机整数
    'random_sample',            # 生成一个指定范围内的随机浮点数
    'ranf',                     # 生成一个0到1之间的随机浮点数
    'rayleigh',                 # 从瑞利分布中抽取随机样本
    'sample',                   # 从指定的序列中随机抽取指定长度的样本
    'seed',                     # 初始化随机数生成器的种子
    'set_state',                # 设置随机数生成器的状态
    'shuffle',                  # 将序列中的元素随机排序
    'standard_cauchy',          # 从标准柯西分布中抽取随机样本
    'standard_exponential',     # 从标准指数分布中抽取随机样本
    'standard_gamma',           # 从标准伽马分布中抽取随机样本
    'standard_normal',          # 从标准正态分布中抽取随机样本
    'standard_t',               # 从标准学生 t 分布中抽取随机样本
    'triangular',               # 从三角分布中抽取随机样本
    'uniform',                  # 从均匀分布中抽取随机样本
    'vonmises',                 # 从冯·米塞斯分布中抽取随机样本
    'wald',                     # 从瓦尔德分布中抽取随机样本
    'weibull',                  # 从威布尔分布中抽取随机样本
    'zipf',                     # 从齐普夫分布中抽取随机样本
# 将以下模块导入用于模块冻结分析（例如 PyInstaller）
from . import _pickle
from . import _common
from . import _bounded_integers

# 导入 Generator 类和 default_rng 函数
from ._generator import Generator, default_rng
# 导入 SeedSequence 类和 BitGenerator 类
from .bit_generator import SeedSequence, BitGenerator
# 导入 MT19937 类
from ._mt19937 import MT19937
# 导入 PCG64 和 PCG64DXSM 类
from ._pcg64 import PCG64, PCG64DXSM
# 导入 Philox 类
from ._philox import Philox
# 导入 SFC64 类
from ._sfc64 import SFC64
# 导入 mtrand 中的所有内容
from .mtrand import *

# 将以下名称添加到 __all__ 列表中，以便在模块被 import * 时可以访问
__all__ += ['Generator', 'RandomState', 'SeedSequence', 'MT19937',
            'Philox', 'PCG64', 'PCG64DXSM', 'SFC64', 'default_rng',
            'BitGenerator']

def __RandomState_ctor():
    """Return a RandomState instance.

    This function exists solely to assist (un)pickling.

    Note that the state of the RandomState returned here is irrelevant, as this
    function's entire purpose is to return a newly allocated RandomState whose
    state pickle can set.  Consequently the RandomState returned by this function
    is a freshly allocated copy with a seed=0.

    See https://github.com/numpy/numpy/issues/4763 for a detailed discussion

    """
    # 返回一个具有种子为 0 的新分配 RandomState 实例
    return RandomState(seed=0)

# 导入 PytestTester 类并将其命名为 test，用于单元测试
from numpy._pytesttester import PytestTester
test = PytestTester(__name__)
# 删除 PytestTester，使其不再在模块中可用
del PytestTester
```