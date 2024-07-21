# `.\pytorch\torch\distributions\distribution.py`

```py
# mypy: allow-untyped-defs
# 引入警告模块，用于处理警告信息
import warnings
# 引入类型提示相关模块
from typing import Any, Dict, Optional, Tuple
# 引入已弃用的类型提示
from typing_extensions import deprecated

# 引入 PyTorch 库
import torch
# 从 PyTorch 分布模块中导入约束模块
from torch.distributions import constraints
# 从 PyTorch 分布模块中导入延迟属性工具函数
from torch.distributions.utils import lazy_property
# 引入 Torch 的类型定义
from torch.types import _size

# 声明该模块对外暴露的类名列表
__all__ = ["Distribution"]


class Distribution:
    r"""
    Distribution is the abstract base class for probability distributions.
    """

    # 标记是否支持 rsample 方法，默认为 False
    has_rsample = False
    # 标记是否支持 enumerate_support 方法，默认为 False
    has_enumerate_support = False
    # 是否验证参数的标志，取决于 Python 是否处于调试模式
    _validate_args = __debug__

    @staticmethod
    def set_default_validate_args(value: bool) -> None:
        """
        Sets whether validation is enabled or disabled.

        The default behavior mimics Python's ``assert`` statement: validation
        is on by default, but is disabled if Python is run in optimized mode
        (via ``python -O``). Validation may be expensive, so you may want to
        disable it once a model is working.

        Args:
            value (bool): Whether to enable validation.
        """
        # 如果传入的 value 不是 True 或 False，则抛出 ValueError 异常
        if value not in [True, False]:
            raise ValueError
        # 更新全局验证参数标志位
        Distribution._validate_args = value

    def __init__(
        self,
        batch_shape: torch.Size = torch.Size(),
        event_shape: torch.Size = torch.Size(),
        validate_args: Optional[bool] = None,
    ):
        # 初始化批次形状和事件形状
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        # 如果提供了验证参数，更新验证参数标志位
        if validate_args is not None:
            self._validate_args = validate_args
        # 如果启用验证参数，则进行参数验证
        if self._validate_args:
            try:
                arg_constraints = self.arg_constraints
            except NotImplementedError:
                arg_constraints = {}
                # 报出警告，提示未定义参数约束信息
                warnings.warn(
                    f"{self.__class__} does not define `arg_constraints`. "
                    + "Please set `arg_constraints = {}` or initialize the distribution "
                    + "with `validate_args=False` to turn off validation."
                )
            # 遍历参数约束字典，检查参数是否满足约束条件
            for param, constraint in arg_constraints.items():
                if constraints.is_dependent(constraint):
                    continue  # 跳过无法检查的约束条件
                if param not in self.__dict__ and isinstance(
                    getattr(type(self), param), lazy_property
                ):
                    continue  # 跳过延迟构造的参数
                value = getattr(self, param)
                valid = constraint.check(value)
                # 如果参数值不满足约束条件，则抛出 ValueError 异常
                if not valid.all():
                    raise ValueError(
                        f"Expected parameter {param} "
                        f"({type(value).__name__} of shape {tuple(value.shape)}) "
                        f"of distribution {repr(self)} "
                        f"to satisfy the constraint {repr(constraint)}, "
                        f"but found invalid values:\n{value}"
                    )
        # 调用父类的初始化方法
        super().__init__()
    def expand(self, batch_shape: torch.Size, _instance=None):
        """
        Returns a new distribution instance (or populates an existing instance
        provided by a derived class) with batch dimensions expanded to
        `batch_shape`. This method calls :class:`~torch.Tensor.expand` on
        the distribution's parameters. As such, this does not allocate new
        memory for the expanded distribution instance. Additionally,
        this does not repeat any args checking or parameter broadcasting in
        `__init__.py`, when an instance is first created.

        Args:
            batch_shape (torch.Size): the desired expanded size.
            _instance: new instance provided by subclasses that
                need to override `.expand`.

        Returns:
            New distribution instance with batch dimensions expanded to
            `batch_size`.
        """
        raise NotImplementedError
        # 抛出未实现错误，表示此方法在基类中未被实现，需要在子类中覆盖实现

    @property
    def batch_shape(self) -> torch.Size:
        """
        Returns the shape over which parameters are batched.
        """
        return self._batch_shape
        # 返回参数批处理的形状

    @property
    def event_shape(self) -> torch.Size:
        """
        Returns the shape of a single sample (without batching).
        """
        return self._event_shape
        # 返回单个样本（不考虑批处理）的形状

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        """
        Returns a dictionary from argument names to
        :class:`~torch.distributions.constraints.Constraint` objects that
        should be satisfied by each argument of this distribution. Args that
        are not tensors need not appear in this dict.
        """
        raise NotImplementedError
        # 抛出未实现错误，表示此方法在基类中未被实现，需要在子类中覆盖实现

    @property
    def support(self) -> Optional[Any]:
        """
        Returns a :class:`~torch.distributions.constraints.Constraint` object
        representing this distribution's support.
        """
        raise NotImplementedError
        # 抛出未实现错误，表示此方法在基类中未被实现，需要在子类中覆盖实现

    @property
    def mean(self) -> torch.Tensor:
        """
        Returns the mean of the distribution.
        """
        raise NotImplementedError
        # 抛出未实现错误，表示此方法在基类中未被实现，需要在子类中覆盖实现

    @property
    def mode(self) -> torch.Tensor:
        """
        Returns the mode of the distribution.
        """
        raise NotImplementedError(f"{self.__class__} does not implement mode")
        # 抛出未实现错误，并指示特定类没有实现模式计算

    @property
    def variance(self) -> torch.Tensor:
        """
        Returns the variance of the distribution.
        """
        raise NotImplementedError
        # 抛出未实现错误，表示此方法在基类中未被实现，需要在子类中覆盖实现

    @property
    def stddev(self) -> torch.Tensor:
        """
        Returns the standard deviation of the distribution.
        """
        return self.variance.sqrt()
        # 返回分布的标准差，通过方差的平方根计算得出

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched.
        """
        with torch.no_grad():
            return self.rsample(sample_shape)
        # 使用`rsample`方法生成样本，通过`torch.no_grad()`上下文管理器禁用梯度计算
    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.
        """
        # 抛出未实现错误，提示需要在子类中实现具体的采样方法
        raise NotImplementedError

    @deprecated(
        "`sample_n(n)` will be deprecated. Use `sample((n,))` instead.",
        category=FutureWarning,
    )
    def sample_n(self, n: int) -> torch.Tensor:
        """
        Generates n samples or n batches of samples if the distribution
        parameters are batched.
        """
        # 使用警告标记函数即将被弃用，并建议使用更一致的函数 `sample((n,))`
        return self.sample(torch.Size((n,)))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Returns the log of the probability density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
                Tensor representing values at which to evaluate the log
                probability density/mass function.
        """
        # 抛出未实现错误，提示需要在子类中实现具体的对数概率密度/质量函数
        raise NotImplementedError

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        """
        Returns the cumulative density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
                Tensor representing values at which to evaluate the cumulative
                density/mass function.
        """
        # 抛出未实现错误，提示需要在子类中实现具体的累积分布函数
        raise NotImplementedError

    def icdf(self, value: torch.Tensor) -> torch.Tensor:
        """
        Returns the inverse cumulative density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
                Tensor representing values at which to evaluate the inverse
                cumulative density/mass function.
        """
        # 抛出未实现错误，提示需要在子类中实现具体的反向累积分布函数
        raise NotImplementedError

    def enumerate_support(self, expand: bool = True) -> torch.Tensor:
        """
        Returns tensor containing all values supported by a discrete
        distribution. The result will enumerate over dimension 0, so the shape
        of the result will be `(cardinality,) + batch_shape + event_shape`
        (where `event_shape = ()` for univariate distributions).

        Note that this enumerates over all batched tensors in lock-step
        `[[0, 0], [1, 1], ...]`. With `expand=False`, enumeration happens
        along dim 0, but with the remaining batch dimensions being
        singleton dimensions, `[[0], [1], ..`.

        To iterate over the full Cartesian product use
        `itertools.product(m.enumerate_support())`.

        Args:
            expand (bool): whether to expand the support over the
                batch dims to match the distribution's `batch_shape`.

        Returns:
            Tensor iterating over dimension 0.
        """
        # 抛出未实现错误，提示需要在子类中实现具体的支持枚举函数
        raise NotImplementedError

    def entropy(self) -> torch.Tensor:
        """
        Returns entropy of distribution, batched over batch_shape.

        Returns:
            Tensor of shape batch_shape.
        """
        # 抛出未实现错误，提示需要在子类中实现具体的熵计算函数
        raise NotImplementedError

    def perplexity(self) -> torch.Tensor:
        """
        Returns perplexity of distribution, batched over batch_shape.

        Returns:
            Tensor of shape batch_shape.
        """
        # 返回分布的困惑度，即指数化熵值
        return torch.exp(self.entropy())
    def _extended_shape(self, sample_shape: _size = torch.Size()) -> Tuple[int, ...]:
        """
        返回分布返回的样本大小，给定 `sample_shape`。注意，分布实例的批处理和事件形状在构造时是固定的。
        如果 `sample_shape` 是空的，则返回的形状被提升为 (1,)。

        Args:
            sample_shape (torch.Size): 要绘制样本的大小。
        """
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        # 返回扩展后的样本形状，结合分布实例的批处理和事件形状
        return torch.Size(sample_shape + self._batch_shape + self._event_shape)

    def _validate_sample(self, value: torch.Tensor) -> None:
        """
        对分布方法（如 `log_prob`、`cdf` 和 `icdf`）的参数进行验证。要通过这些方法评分的值的最右边维度必须与分布的批处理和事件形状匹配。

        Args:
            value (Tensor): 要计算其对数概率的张量。

        Raises:
            ValueError: 当 `value` 的最右边维度与分布的批处理和事件形状不匹配时。
        """
        if not isinstance(value, torch.Tensor):
            raise ValueError("The value argument to log_prob must be a Tensor")

        event_dim_start = len(value.size()) - len(self._event_shape)
        if value.size()[event_dim_start:] != self._event_shape:
            raise ValueError(
                f"The right-most size of value must match event_shape: {value.size()} vs {self._event_shape}."
            )

        actual_shape = value.size()
        expected_shape = self._batch_shape + self._event_shape
        for i, j in zip(reversed(actual_shape), reversed(expected_shape)):
            if i != 1 and j != 1 and i != j:
                raise ValueError(
                    f"Value is not broadcastable with batch_shape+event_shape: {actual_shape} vs {expected_shape}."
                )
        try:
            support = self.support
        except NotImplementedError:
            warnings.warn(
                f"{self.__class__} does not define `support` to enable "
                + "sample validation. Please initialize the distribution with "
                + "`validate_args=False` to turn off validation."
            )
            return
        assert support is not None
        # 检查值是否在支持范围内
        valid = support.check(value)
        if not valid.all():
            raise ValueError(
                "Expected value argument "
                f"({type(value).__name__} of shape {tuple(value.shape)}) "
                f"to be within the support ({repr(support)}) "
                f"of the distribution {repr(self)}, "
                f"but found invalid values:\n{value}"
            )
    # 返回一个实例化的对象的字符串表示形式
    def _get_checked_instance(self, cls, _instance=None):
        # 如果没有提供实例，并且当前类的初始化方法与给定类的初始化方法不同，则抛出未实现错误
        if _instance is None and type(self).__init__ != cls.__init__:
            raise NotImplementedError(
                f"Subclass {self.__class__.__name__} of {cls.__name__} that defines a custom __init__ method "
                "must also define a custom .expand() method."
            )
        # 如果没有提供实例，返回一个当前类类型的新实例；否则返回提供的实例
        return self.__new__(type(self)) if _instance is None else _instance

    # 返回对象的字符串表示形式，用于调试和显示
    def __repr__(self) -> str:
        # 获取参数名称列表，这些参数在参数约束字典中，并且存在于对象的实例变量中
        param_names = [k for k, _ in self.arg_constraints.items() if k in self.__dict__]
        # 构建参数字符串，显示每个参数的名称及其值或大小
        args_string = ", ".join(
            [
                f"{p}: {self.__dict__[p] if self.__dict__[p].numel() == 1 else self.__dict__[p].size()}"
                for p in param_names
            ]
        )
        # 返回类名及其参数的字符串表示形式
        return self.__class__.__name__ + "(" + args_string + ")"
```