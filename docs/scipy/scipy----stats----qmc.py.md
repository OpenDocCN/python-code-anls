# `D:\src\scipysrc\scipy\scipy\stats\qmc.py`

```
r"""
====================================================
Quasi-Monte Carlo submodule (:mod:`scipy.stats.qmc`)
====================================================

.. currentmodule:: scipy.stats.qmc

This module provides Quasi-Monte Carlo generators and associated helper
functions.


Quasi-Monte Carlo
=================

Engines
-------

.. autosummary::
   :toctree: generated/

   QMCEngine
   Sobol
   Halton
   LatinHypercube
   PoissonDisk
   MultinomialQMC
   MultivariateNormalQMC

Helpers
-------

.. autosummary::
   :toctree: generated/

   discrepancy
   geometric_discrepancy
   update_discrepancy
   scale


Introduction to Quasi-Monte Carlo
=================================

Quasi-Monte Carlo (QMC) methods [1]_, [2]_, [3]_ provide an
:math:`n \times d` array of numbers in :math:`[0,1]`. They can be used in
place of :math:`n` points from the :math:`U[0,1]^{d}` distribution. Compared to
random points, QMC points are designed to have fewer gaps and clumps. This is
quantified by discrepancy measures [4]_. From the Koksma-Hlawka
inequality [5]_ we know that low discrepancy reduces a bound on
integration error. Averaging a function :math:`f` over :math:`n` QMC points
can achieve an integration error close to :math:`O(n^{-1})` for well
behaved functions [2]_.

Most QMC constructions are designed for special values of :math:`n`
such as powers of 2 or large primes. Changing the sample
size by even one can degrade their performance, even their
rate of convergence [6]_. For instance :math:`n=100` points may give less
accuracy than :math:`n=64` if the method was designed for :math:`n=2^m`.

Some QMC constructions are extensible in :math:`n`: we can find
another special sample size :math:`n' > n` and often an infinite
sequence of increasing special sample sizes. Some QMC
constructions are extensible in :math:`d`: we can increase the dimension,
possibly to some upper bound, and typically without requiring
special values of :math:`d`. Some QMC methods are extensible in
both :math:`n` and :math:`d`.

QMC points are deterministic. That makes it hard to estimate the accuracy of
integrals estimated by averages over QMC points. Randomized QMC (RQMC) [7]_
points are constructed so that each point is individually :math:`U[0,1]^{d}`
while collectively the :math:`n` points retain their low discrepancy.
One can make :math:`R` independent replications of RQMC points to
see how stable a computation is. From :math:`R` independent values,
a t-test (or bootstrap t-test [8]_) then gives approximate confidence
intervals on the mean value. Some RQMC methods produce a
root mean squared error that is actually :math:`o(1/n)` and smaller than
the rate seen in unrandomized QMC. An intuitive explanation is
that the error is a sum of many small ones and random errors
cancel in a way that deterministic ones do not. RQMC also
has advantages on integrands that are singular or, for other
reasons, fail to be Riemann integrable.

"""

# This is a documentation string (docstring) in reStructuredText format.
# It provides an overview of the Quasi-Monte Carlo submodule in scipy.stats.qmc.
# It includes module information, descriptions of engines and helpers,
# an introduction to Quasi-Monte Carlo methods, and their theoretical foundations.
# RQMC 不能克服Bahkvalov的维数诅咒（参见[9]_）。对于任何随机或确定性方法，
# 在高维度下存在最坏情况的函数，导致其性能不佳。例如，QMC 的最坏情况函数
# 可能在所有n个点处为0，但在其他地方非常大。在高维度下，最坏情况分析变得非常悲观。
# 当(R)QMC在非最坏情况的函数上使用时，可以显著改善MC的性能。
# 例如，当积分函数可以很好地近似为每次输入变量的少量函数之和时，
# (R)QMC可能特别有效 [10]_, [11]_。这种性质通常是对这些函数的惊人发现之一。

# 此外，为了比IID MC有所改进，(R)QMC要求积分函数具有一定的平滑性，
# 大致等同于每个方向上的混合一阶导数，
# :math:`\partial^d f/\partial x_1 \cdots \partial x_d` 必须是可积的。
# 例如，一个在超球内为1且在其外为0的函数在任何维度下都具有Hardy和Krause意义上的无穷变化。

# 搅乱网（Scrambled nets）是一种具有一些有价值的鲁棒性质的RQMC [12]_。
# 如果积分函数是平方可积的，它们提供的方差 :math:`var_{SNET} = o(1/n)`。
# 对于每个平方可积积分函数，存在一个有限的上界 :math:`var_{SNET} / var_{MC}`。
# 当 :math:`p > 1` 时，搅乱网在 :math:`L^p` 中满足大数定律。
# 在某些特殊情况下存在中心极限定理 [13]_。对于足够平滑的积分函数，
# 它们可以实现接近 :math:`O(n^{-3})` 的RMSE。有关这些性质的参考请见 [12]_。

# QMC的主要方法包括格规则（lattice rules）[14]_、数字网和序列 [2]_, [15]_。
# 这些理论在多项式格规则 [16]_ 中得到结合，后者可以生成数字网。
# 格规则需要寻找良好的构造形式。对于数字网，通常使用默认构造方法。

# 最广泛使用的QMC方法是Sobol'序列 [17]_。它们是数字网，可以在 :math:`n` 和 :math:`d` 上扩展，
# 可以被搅乱。它们的特殊样本大小是2的幂。另一个流行的方法是Halton序列 [18]_。
# 其构造类似于数字网。较早的维度具有比后续维度更好的均匀分布性质。
# 它们没有特殊的样本大小。人们认为它们不如Sobol'序列精确。
# 它们可以被搅乱。Faure的网 [19]_ 也被广泛使用。所有维度均具有相同的优秀性质，
# 但特殊的样本大小随维度 :math:`d` 快速增长。它们可以被搅乱。
# Niederreiter和Xing的网 [20]_ 具有最佳的渐近性质，但在实证表现上不佳 [21]_。

# 高阶数字网通过点的数字交错过程形成。
# 它们可以实现更高的...
# Levels of asymptotic accuracy given higher smoothness conditions on :math:`f`
# and they can be scrambled [22]_. There is little or no empirical work
# showing the improved rate to be attained.
# 
# Using QMC is like using the entire period of a small random
# number generator. The constructions are similar and so
# therefore are the computational costs [23]_.
# 
# (R)QMC is sometimes improved by passing the points through
# a baker's transformation (tent function) prior to using them.
# That function has the form :math:`1-2|x-1/2|`. As :math:`x` goes from 0 to
# 1, this function goes from 0 to 1 and then back. It is very
# useful to produce a periodic function for lattice rules [14]_,
# and sometimes it improves the convergence rate [24]_.
# 
# It is not straightforward to apply QMC methods to Markov
# chain Monte Carlo (MCMC).  We can think of MCMC as using
# :math:`n=1` point in :math:`[0,1]^{d}` for very large :math:`d`, with
# ergodic results corresponding to :math:`d \to \infty`. One proposal is
# in [25]_ and under strong conditions an improved rate of convergence
# has been shown [26]_.
# 
# Returning to Sobol' points: there are many versions depending
# on what are called direction numbers. Those are the result of
# searches and are tabulated. A very widely used set of direction
# numbers come from [27]_. It is extensible in dimension up to
# :math:`d=21201`.
# 
# References
# ----------
# .. [1] Owen, Art B. "Monte Carlo Book: the Quasi-Monte Carlo parts." 2019.
# .. [2] Niederreiter, Harald. "Random number generation and quasi-Monte Carlo
#    methods." Society for Industrial and Applied Mathematics, 1992.
# .. [3] Dick, Josef, Frances Y. Kuo, and Ian H. Sloan. "High-dimensional
#    integration: the quasi-Monte Carlo way." Acta Numerica no. 22: 133, 2013.
# .. [4] Aho, A. V., C. Aistleitner, T. Anderson, K. Appel, V. Arnol'd, N.
#    Aronszajn, D. Asotsky et al. "W. Chen et al.(eds.), "A Panorama of
#    Discrepancy Theory", Sringer International Publishing,
#    Switzerland: 679, 2014.
# .. [5] Hickernell, Fred J. "Koksma-Hlawka Inequality." Wiley StatsRef:
#    Statistics Reference Online, 2014.
# .. [6] Owen, Art B. "On dropping the first Sobol' point." :arxiv:`2008.08051`,
#    2020.
# .. [7] L'Ecuyer, Pierre, and Christiane Lemieux. "Recent advances in randomized
#    quasi-Monte Carlo methods." In Modeling uncertainty, pp. 419-474. Springer,
#    New York, NY, 2002.
# .. [8] DiCiccio, Thomas J., and Bradley Efron. "Bootstrap confidence
#    intervals." Statistical science: 189-212, 1996.
# .. [9] Dimov, Ivan T. "Monte Carlo methods for applied scientists." World
#    Scientific, 2008.
# .. [10] Caflisch, Russel E., William J. Morokoff, and Art B. Owen. "Valuation
#    of mortgage backed securities using Brownian bridges to reduce effective
#    dimension." Journal of Computational Finance: no. 1 27-46, 1997.
# .. [11] Sloan, Ian H., and Henryk Wozniakowski. "When are quasi-Monte Carlo
#    algorithms efficient for high dimensional integrals?." Journal of Complexity
#    14, no. 1 (1998): 1-33.
# 导入 _qmc 模块中所有的公共符号，这些符号将被当前模块使用
from ._qmc import *  # noqa: F403

# 导入 _qmc 模块中定义的 __all__ 列表，用于显示哪些符号是公共的
from ._qmc import __all__  # noqa: F401
```