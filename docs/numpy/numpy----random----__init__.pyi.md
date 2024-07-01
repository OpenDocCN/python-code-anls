# `.\numpy\numpy\random\__init__.pyi`

```py
# 从 numpy._pytesttester 模块导入 PytestTester 类
from numpy._pytesttester import PytestTester

# 从 numpy.random._generator 模块导入 Generator 类作为 Generator
# 从 numpy.random._generator 模块导入 default_rng 函数作为 default_rng
from numpy.random._generator import Generator as Generator
from numpy.random._generator import default_rng as default_rng

# 从 numpy.random._mt19937 模块导入 MT19937 类作为 MT19937
from numpy.random._mt19937 import MT19937 as MT19937

# 从 numpy.random._pcg64 模块导入 PCG64 类作为 PCG64
# 从 numpy.random._pcg64 模块导入 PCG64DXSM 类作为 PCG64DXSM
from numpy.random._pcg64 import (
    PCG64 as PCG64,
    PCG64DXSM as PCG64DXSM,
)

# 从 numpy.random._philox 模块导入 Philox 类作为 Philox
from numpy.random._philox import Philox as Philox

# 从 numpy.random._sfc64 模块导入 SFC64 类作为 SFC64
from numpy.random._sfc64 import SFC64 as SFC64

# 从 numpy.random.bit_generator 模块导入 BitGenerator 类作为 BitGenerator
from numpy.random.bit_generator import BitGenerator as BitGenerator

# 从 numpy.random.bit_generator 模块导入 SeedSequence 类作为 SeedSequence
from numpy.random.bit_generator import SeedSequence as SeedSequence

# 从 numpy.random.mtrand 模块导入以下内容：
# RandomState 类作为 RandomState
# 各种随机分布函数（例如 beta, binomial, bytes 等）
# 其他的功能函数（例如 get_bit_generator, get_state, set_bit_generator 等）
from numpy.random.mtrand import (
    RandomState as RandomState,
    beta as beta,
    binomial as binomial,
    bytes as bytes,
    chisquare as chisquare,
    choice as choice,
    dirichlet as dirichlet,
    exponential as exponential,
    f as f,
    gamma as gamma,
    geometric as geometric,
    get_bit_generator as get_bit_generator,
    get_state as get_state,
    gumbel as gumbel,
    hypergeometric as hypergeometric,
    laplace as laplace,
    logistic as logistic,
    lognormal as lognormal,
    logseries as logseries,
    multinomial as multinomial,
    multivariate_normal as multivariate_normal,
    negative_binomial as negative_binomial,
    noncentral_chisquare as noncentral_chisquare,
    noncentral_f as noncentral_f,
    normal as normal,
    pareto as pareto,
    permutation as permutation,
    poisson as poisson,
    power as power,
    rand as rand,
    randint as randint,
    randn as randn,
    random as random,
    random_integers as random_integers,
    random_sample as random_sample,
    ranf as ranf,
    rayleigh as rayleigh,
    sample as sample,
    seed as seed,
    set_bit_generator as set_bit_generator,
    set_state as set_state,
    shuffle as shuffle,
    standard_cauchy as standard_cauchy,
    standard_exponential as standard_exponential,
    standard_gamma as standard_gamma,
    standard_normal as standard_normal,
    standard_t as standard_t,
    triangular as triangular,
    uniform as uniform,
    vonmises as vonmises,
    wald as wald,
    weibull as weibull,
    zipf as zipf,
)

# 定义 __all__ 列表，包含所有需要导出的符号名称字符串
__all__: list[str]

# 定义 test 变量，类型为 PytestTester 类的一个实例
test: PytestTester
```