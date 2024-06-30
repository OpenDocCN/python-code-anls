# `D:\src\scipysrc\sympy\sympy\stats\stochastic_process.py`

```
from sympy.core.basic import Basic
from sympy.stats.joint_rv import ProductPSpace
from sympy.stats.rv import ProductDomain, _symbol_converter, Distribution

class StochasticPSpace(ProductPSpace):
    """
    Represents probability space of stochastic processes
    and their random variables. Contains mechanics to do
    computations for queries of stochastic processes.
    """

    def __new__(cls, sym, process, distribution=None):
        # 转换符号为内部统一格式
        sym = _symbol_converter(sym)
        # 导入必要的随机过程类
        from sympy.stats.stochastic_process_types import StochasticProcess
        # 检查传入的随机过程是否符合要求
        if not isinstance(process, StochasticProcess):
            raise TypeError("`process` must be an instance of StochasticProcess.")
        # 如果未指定分布，则使用默认分布
        if distribution is None:
            distribution = Distribution()
        # 调用父类的构造函数
        return Basic.__new__(cls, sym, process, distribution)

    @property
    def process(self):
        """
        返回关联的随机过程。
        """
        return self.args[1]

    @property
    def domain(self):
        """
        返回随机过程的定义域。
        """
        return ProductDomain(self.process.index_set,
                             self.process.state_space)

    @property
    def symbol(self):
        """
        返回表示该概率空间的符号。
        """
        return self.args[0]

    @property
    def distribution(self):
        """
        返回该概率空间的分布。
        """
        return self.args[2]

    def probability(self, condition, given_condition=None, evaluate=True, **kwargs):
        """
        将处理查询的任务转移到特定的随机过程，因为每个过程都有处理这类查询的独特逻辑。
        """
        return self.process.probability(condition, given_condition, evaluate, **kwargs)

    def compute_expectation(self, expr, condition=None, evaluate=True, **kwargs):
        """
        将处理查询的任务转移到特定的随机过程，因为每个过程都有处理这类查询的独特逻辑。
        """
        return self.process.expectation(expr, condition, evaluate, **kwargs)
```