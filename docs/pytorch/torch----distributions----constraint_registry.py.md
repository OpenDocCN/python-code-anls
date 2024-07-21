# `.\pytorch\torch\distributions\constraint_registry.py`

```
# 设置类型检查选项，允许未标记类型的函数定义
# 通常用于 PyTorch 提供的全局 ConstraintRegistry 对象，将 Constraint 对象链接到 Transform 对象上

r"""
PyTorch 提供了两个全局的 ConstraintRegistry 对象，将 Constraint 对象与 Transform 对象关联起来。
这些对象都接受约束条件并返回变换，但它们在双射性方面有不同的保证。

1. ``biject_to(constraint)`` 从 ``constraints.real`` 查找一个双射的 Transform，将其映射到给定的 constraint。
   返回的 Transform 保证具有 ``.bijective = True``，并应实现 ``.log_abs_det_jacobian()`` 方法。
   
2. ``transform_to(constraint)`` 从 ``constraints.real`` 查找一个不一定是双射的 Transform，将其映射到给定的 constraint。
   返回的 Transform 不保证实现 ``.log_abs_det_jacobian()`` 方法。

``transform_to()`` 注册表对于在概率分布的约束参数上执行无约束优化非常有用，
这些约束由每个分布的 ``.arg_constraints`` 字典指示。这些变换通常过度参数化空间以避免旋转；
因此，它们更适合于像 Adam 这样的逐坐标优化算法：

    loc = torch.zeros(100, requires_grad=True)
    unconstrained = torch.zeros(100, requires_grad=True)
    scale = transform_to(Normal.arg_constraints['scale'])(unconstrained)
    loss = -Normal(loc, scale).log_prob(data).sum()

``biject_to()`` 注册表对于 Hamiltonian Monte Carlo 非常有用，
在这种算法中，来自具有约束 ``.support`` 的概率分布的样本在无约束空间中传播，并且通常算法是旋转不变的：

    dist = Exponential(rate)
    unconstrained = torch.zeros(100, requires_grad=True)
    sample = biject_to(dist.support)(unconstrained)
    potential_energy = -dist.log_prob(sample).sum()

.. note::

    ``transform_to`` 和 ``biject_to`` 的一个区别示例是 ``constraints.simplex``：
    ``transform_to(constraints.simplex)`` 返回一个 SoftmaxTransform，
    它简单地对其输入进行指数化和归一化；这是一种廉价且大多数情况下适合于 SVI 等算法的逐坐标操作。
    相反，``biject_to(constraints.simplex)`` 返回一个 StickBreakingTransform，
    它将其输入映射到一个少一维空间；这是一种更昂贵、数值稳定性较差的变换，但适用于像 HMC 这样的算法。

``biject_to`` 和 ``transform_to`` 对象可以通过它们的 ``.register()`` 方法进行扩展，
可以作为单例约束上的函数使用：

    transform_to.register(my_constraint, my_transform)

或者作为参数化约束上的装饰器使用：

    @transform_to.register(MyConstraintClass)
    # 定义一个名为 my_factory 的函数，用于根据给定的约束条件创建并返回一个 MyTransform 对象
    def my_factory(constraint):
        # 断言确保传入的 constraint 参数是 MyConstraintClass 的一个实例
        assert isinstance(constraint, MyConstraintClass)
        # 根据 constraint 的参数创建一个 MyTransform 对象，并返回
        return MyTransform(constraint.param1, constraint.param2)
# 引入需要的模块：numbers，constraints和transforms
import numbers
from torch.distributions import constraints, transforms

# 定义模块内公开的接口列表
__all__ = [
    "ConstraintRegistry",
    "biject_to",
    "transform_to",
]

# 定义约束注册器类
class ConstraintRegistry:
    """
    Registry to link constraints to transforms.
    """

    def __init__(self):
        # 初始化约束注册字典
        self._registry = {}
        super().__init__()  # 调用父类初始化方法

    def register(self, constraint, factory=None):
        """
        Registers a :class:`~torch.distributions.constraints.Constraint`
        subclass in this registry. Usage::

            @my_registry.register(MyConstraintClass)
            def construct_transform(constraint):
                assert isinstance(constraint, MyConstraint)
                return MyTransform(constraint.arg_constraints)

        Args:
            constraint (subclass of :class:`~torch.distributions.constraints.Constraint`):
                A subclass of :class:`~torch.distributions.constraints.Constraint`, or
                a singleton object of the desired class.
            factory (Callable): A callable that inputs a constraint object and returns
                a  :class:`~torch.distributions.transforms.Transform` object.
        """
        # 支持用作装饰器
        if factory is None:
            return lambda factory: self.register(constraint, factory)

        # 支持对单例实例调用
        if isinstance(constraint, constraints.Constraint):
            constraint = type(constraint)

        # 检查约束是否为约束类的子类或实例
        if not isinstance(constraint, type) or not issubclass(
            constraint, constraints.Constraint
        ):
            raise TypeError(
                f"Expected constraint to be either a Constraint subclass or instance, but got {constraint}"
            )

        # 将约束类及其对应的工厂函数注册到注册字典中
        self._registry[constraint] = factory
        return factory

    def __call__(self, constraint):
        """
        Looks up a transform to constrained space, given a constraint object.
        Usage::

            constraint = Normal.arg_constraints['scale']
            scale = transform_to(constraint)(torch.zeros(1))  # constrained
            u = transform_to(constraint).inv(scale)           # unconstrained

        Args:
            constraint (:class:`~torch.distributions.constraints.Constraint`):
                A constraint object.

        Returns:
            A :class:`~torch.distributions.transforms.Transform` object.

        Raises:
            `NotImplementedError` if no transform has been registered.
        """
        # 根据约束类查找对应的工厂函数
        try:
            factory = self._registry[type(constraint)]
        except KeyError:
            # 如果未找到注册的转换函数，抛出未实现错误
            raise NotImplementedError(
                f"Cannot transform {type(constraint).__name__} constraints"
            ) from None
        return factory(constraint)

# 创建两个全局实例对象，作为约束注册器和转换注册器
biject_to = ConstraintRegistry()
transform_to = ConstraintRegistry()
################################################################################
# Registration Table
################################################################################

# 注册函数，将实数约束映射到恒等变换
@biject_to.register(constraints.real)
@transform_to.register(constraints.real)
def _transform_to_real(constraint):
    return transforms.identity_transform

# 注册函数，将独立约束映射到独立变换
@biject_to.register(constraints.independent)
def _biject_to_independent(constraint):
    base_transform = biject_to(constraint.base_constraint)
    return transforms.IndependentTransform(
        base_transform, constraint.reinterpreted_batch_ndims
    )

# 注册函数，将独立约束映射到独立变换
@transform_to.register(constraints.independent)
def _transform_to_independent(constraint):
    base_transform = transform_to(constraint.base_constraint)
    return transforms.IndependentTransform(
        base_transform, constraint.reinterpreted_batch_ndims
    )

# 注册函数，将正数或非负数约束映射到指数变换
@biject_to.register(constraints.positive)
@biject_to.register(constraints.nonnegative)
@transform_to.register(constraints.positive)
@transform_to.register(constraints.nonnegative)
def _transform_to_positive(constraint):
    return transforms.ExpTransform()

# 注册函数，将大于或大于等于约束映射到复合变换（指数变换和仿射变换）
@biject_to.register(constraints.greater_than)
@biject_to.register(constraints.greater_than_eq)
@transform_to.register(constraints.greater_than)
@transform_to.register(constraints.greater_than_eq)
def _transform_to_greater_than(constraint):
    return transforms.ComposeTransform(
        [
            transforms.ExpTransform(),
            transforms.AffineTransform(constraint.lower_bound, 1),
        ]
    )

# 注册函数，将小于约束映射到复合变换（指数变换和仿射变换）
@biject_to.register(constraints.less_than)
@transform_to.register(constraints.less_than)
def _transform_to_less_than(constraint):
    return transforms.ComposeTransform(
        [
            transforms.ExpTransform(),
            transforms.AffineTransform(constraint.upper_bound, -1),
        ]
    )

# 注册函数，将区间约束映射到复合变换（sigmoid变换和仿射变换）
@biject_to.register(constraints.interval)
@biject_to.register(constraints.half_open_interval)
@transform_to.register(constraints.interval)
@transform_to.register(constraints.half_open_interval)
def _transform_to_interval(constraint):
    # 处理单位区间的特殊情况
    lower_is_0 = (
        isinstance(constraint.lower_bound, numbers.Number)
        and constraint.lower_bound == 0
    )
    upper_is_1 = (
        isinstance(constraint.upper_bound, numbers.Number)
        and constraint.upper_bound == 1
    )
    if lower_is_0 and upper_is_1:
        return transforms.SigmoidTransform()

    loc = constraint.lower_bound
    scale = constraint.upper_bound - constraint.lower_bound
    return transforms.ComposeTransform(
        [transforms.SigmoidTransform(), transforms.AffineTransform(loc, scale)]
    )

# 注册函数，将单纯形约束映射到stick-breaking变换
@biject_to.register(constraints.simplex)
def _biject_to_simplex(constraint):
    return transforms.StickBreakingTransform()

# 注册函数，将单纯形约束映射到softmax变换
@transform_to.register(constraints.simplex)
def _transform_to_simplex(constraint):
    return transforms.SoftmaxTransform()

# TODO define a bijection for LowerCholeskyTransform
# 注册一个转换函数，将给定的 lower_cholesky 约束转换为 LowerCholeskyTransform 对象
@transform_to.register(constraints.lower_cholesky)
def _transform_to_lower_cholesky(constraint):
    return transforms.LowerCholeskyTransform()


# 注册一个转换函数，将给定的 positive_definite 约束或 positive_semidefinite 约束转换为 PositiveDefiniteTransform 对象
@transform_to.register(constraints.positive_definite)
@transform_to.register(constraints.positive_semidefinite)
def _transform_to_positive_definite(constraint):
    return transforms.PositiveDefiniteTransform()


# 注册一个转换函数，将给定的 corr_cholesky 约束转换为 CorrCholeskyTransform 对象
# 同时它也作为 biject_to.register(constraints.corr_cholesky) 的注册函数
@biject_to.register(constraints.corr_cholesky)
@transform_to.register(constraints.corr_cholesky)
def _transform_to_corr_cholesky(constraint):
    return transforms.CorrCholeskyTransform()


# 注册一个转换函数，将给定的 cat 约束转换为 CatTransform 对象
@biject_to.register(constraints.cat)
def _biject_to_cat(constraint):
    return transforms.CatTransform(
        [biject_to(c) for c in constraint.cseq], constraint.dim, constraint.lengths
    )


# 注册一个转换函数，将给定的 cat 约束转换为 CatTransform 对象
@transform_to.register(constraints.cat)
def _transform_to_cat(constraint):
    return transforms.CatTransform(
        [transform_to(c) for c in constraint.cseq], constraint.dim, constraint.lengths
    )


# 注册一个转换函数，将给定的 stack 约束转换为 StackTransform 对象
@biject_to.register(constraints.stack)
def _biject_to_stack(constraint):
    return transforms.StackTransform(
        [biject_to(c) for c in constraint.cseq], constraint.dim
    )


# 注册一个转换函数，将给定的 stack 约束转换为 StackTransform 对象
@transform_to.register(constraints.stack)
def _transform_to_stack(constraint):
    return transforms.StackTransform(
        [transform_to(c) for c in constraint.cseq], constraint.dim
    )
```