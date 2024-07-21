# `.\pytorch\torch\distributions\transforms.py`

```
# mypy: allow-untyped-defs
# 引入必要的模块和库
import functools  # 提供高阶函数操作
import math  # 提供数学函数
import numbers  # 处理数值类型的抽象基类
import operator  # 提供Python内置操作符的函数实现
import weakref  # 提供弱引用对象，帮助实现自动内存管理
from typing import List  # 引入类型提示支持列表类型

import torch  # PyTorch深度学习框架
import torch.nn.functional as F  # PyTorch中的函数操作模块
from torch.distributions import constraints  # PyTorch分布库中的约束模块
from torch.distributions.utils import (
    _sum_rightmost,  # 辅助函数，对张量的最右边维度求和
    broadcast_all,  # 辅助函数，广播操作，使输入张量的维度相同
    lazy_property,  # 装饰器，用于延迟属性的计算
    tril_matrix_to_vec,  # 辅助函数，将下三角矩阵转换为向量
    vec_to_tril_matrix,  # 辅助函数，将向量转换为下三角矩阵
)
from torch.nn.functional import pad, softplus  # PyTorch中的填充函数和softplus函数

__all__ = [
    "AbsTransform",  # 绝对值变换
    "AffineTransform",  # 仿射变换
    "CatTransform",  # 拼接变换
    "ComposeTransform",  # 组合变换
    "CorrCholeskyTransform",  # 相关性Cholesky变换
    "CumulativeDistributionTransform",  # 累积分布变换
    "ExpTransform",  # 指数变换
    "IndependentTransform",  # 独立变换
    "LowerCholeskyTransform",  # 下三角Cholesky变换
    "PositiveDefiniteTransform",  # 正定变换
    "PowerTransform",  # 幂变换
    "ReshapeTransform",  # 重塑变换
    "SigmoidTransform",  # sigmoid变换
    "SoftplusTransform",  # softplus变换
    "TanhTransform",  # tanh变换
    "SoftmaxTransform",  # softmax变换
    "StackTransform",  # 堆叠变换
    "StickBreakingTransform",  # 粘性分数变换
    "Transform",  # 变换基类
    "identity_transform",  # 单位变换
]


class Transform:
    """
    Abstract class for invertable transformations with computable log
    det jacobians. They are primarily used in
    :class:`torch.distributions.TransformedDistribution`.

    Caching is useful for transforms whose inverses are either expensive or
    numerically unstable. Note that care must be taken with memoized values
    since the autograd graph may be reversed. For example while the following
    works with or without caching::

        y = t(x)
        t.log_abs_det_jacobian(x, y).backward()  # x will receive gradients.

    However the following will error when caching due to dependency reversal::

        y = t(x)
        z = t.inv(y)
        grad(z.sum(), [y])  # error because z is x

    Derived classes should implement one or both of :meth:`_call` or
    :meth:`_inverse`. Derived classes that set `bijective=True` should also
    implement :meth:`log_abs_det_jacobian`.

    Args:
        cache_size (int): Size of cache. If zero, no caching is done. If one,
            the latest single value is cached. Only 0 and 1 are supported.

    Attributes:
        domain (:class:`~torch.distributions.constraints.Constraint`):
            The constraint representing valid inputs to this transform.
        codomain (:class:`~torch.distributions.constraints.Constraint`):
            The constraint representing valid outputs to this transform
            which are inputs to the inverse transform.
        bijective (bool): Whether this transform is bijective. A transform
            ``t`` is bijective iff ``t.inv(t(x)) == x`` and
            ``t(t.inv(y)) == y`` for every ``x`` in the domain and ``y`` in
            the codomain. Transforms that are not bijective should at least
            maintain the weaker pseudoinverse properties
            ``t(t.inv(t(x)) == t(x)`` and ``t.inv(t(t.inv(y))) == t.inv(y)``.
        sign (int or Tensor): For bijective univariate transforms, this
            should be +1 or -1 depending on whether transform is monotone
            increasing or decreasing.
    """
    # 设定默认值为 False，表示该变换不是双射
    bijective = False
    # 声明域和值域，均为约束对象 constraints.Constraint
    domain: constraints.Constraint
    codomain: constraints.Constraint

    # 初始化函数，可以指定缓存大小，默认为 0
    def __init__(self, cache_size=0):
        # 设置缓存大小
        self._cache_size = cache_size
        # 初始化反向变换为 None
        self._inv = None
        # 根据缓存大小进行不同的初始化操作
        if cache_size == 0:
            pass  # 默认行为
        elif cache_size == 1:
            self._cached_x_y = None, None  # 初始化缓存元组
        else:
            raise ValueError("cache_size must be 0 or 1")  # 抛出异常，缓存大小只能是 0 或 1
        super().__init__()  # 调用父类的初始化函数

    # 返回对象的序列化状态
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_inv"] = None  # 将反向变换状态置为 None
        return state

    # 返回事件维度属性
    @property
    def event_dim(self):
        # 如果域和值域的事件维度相同，则返回该维度
        if self.domain.event_dim == self.codomain.event_dim:
            return self.domain.event_dim
        # 否则抛出异常
        raise ValueError("Please use either .domain.event_dim or .codomain.event_dim")

    # 返回变换的逆变换对象
    @property
    def inv(self):
        """
        返回该变换的逆变换 :class:`Transform`。
        应该满足 `t.inv.inv is t`。
        """
        inv = None
        # 如果已经存在逆变换对象，则获取其引用
        if self._inv is not None:
            inv = self._inv()
        # 如果没有现成的逆变换对象，则创建一个新的逆变换对象，并将其弱引用保存在 _inv 属性中
        if inv is None:
            inv = _InverseTransform(self)
            self._inv = weakref.ref(inv)
        return inv

    # 返回雅可比行列式的符号，如果适用的话
    @property
    def sign(self):
        """
        返回雅可比行列式的符号，如果适用的话。
        通常仅对双射变换有意义。
        """
        raise NotImplementedError  # 抛出未实现异常

    # 设置缓存大小，并返回带有新缓存大小的对象
    def with_cache(self, cache_size=1):
        # 如果请求的缓存大小与当前相同，则返回原对象
        if self._cache_size == cache_size:
            return self
        # 如果调用的是 Transform 的构造函数，则返回一个新对象
        if type(self).__init__ is Transform.__init__:
            return type(self)(cache_size=cache_size)
        # 否则抛出未实现异常
        raise NotImplementedError(f"{type(self)}.with_cache is not implemented")

    # 比较函数，判断两个对象是否相等
    def __eq__(self, other):
        return self is other

    # 不等函数，用于 Python2 兼容性
    def __ne__(self, other):
        # 调用 __eq__ 函数的结果取反
        return not self.__eq__(other)

    # 对象调用函数，计算变换后的结果 y
    def __call__(self, x):
        """
        计算变换 `x => y`。
        """
        # 如果缓存大小为 0，则直接调用 _call 方法计算结果
        if self._cache_size == 0:
            return self._call(x)
        # 否则使用缓存，检查是否已经计算过给定 x 的结果
        x_old, y_old = self._cached_x_y
        if x is x_old:
            return y_old
        y = self._call(x)
        self._cached_x_y = x, y  # 更新缓存
        return y

    # 对象的逆调用函数，计算逆变换结果 x
    def _inv_call(self, y):
        """
        计算逆变换 `y => x`。
        """
        # 如果缓存大小为 0，则直接调用 _inverse 方法计算结果
        if self._cache_size == 0:
            return self._inverse(y)
        # 否则使用缓存，检查是否已经计算过给定 y 的结果
        x_old, y_old = self._cached_x_y
        if y is y_old:
            return x_old
        x = self._inverse(y)
        self._cached_x_y = x, y  # 更新缓存
        return x

    # 抽象方法，用于计算正向变换的具体实现
    def _call(self, x):
        """
        计算正向变换的抽象方法。
        """
        raise NotImplementedError

    # 抽象方法，用于计算逆向变换的具体实现
    def _inverse(self, y):
        """
        计算逆向变换的抽象方法。
        """
        raise NotImplementedError

    # 计算给定输入和输出的对数行列式的绝对值 `log |dy/dx|`
    def log_abs_det_jacobian(self, x, y):
        """
        计算给定输入和输出的对数行列式的绝对值 `log |dy/dx|`
        """
        raise NotImplementedError
    # 返回表示对象类名的字符串形式，类似于 "<类名>()"
    def __repr__(self):
        return self.__class__.__name__ + "()"

    # 接收输入形状参数，推断前向计算的输出形状，通常保持输入形状不变
    def forward_shape(self, shape):
        """
        根据输入形状推断前向计算的输出形状。
        默认保持形状不变。
        """
        return shape

    # 接收输出形状参数，推断逆向计算的输入形状，通常保持输出形状不变
    def inverse_shape(self, shape):
        """
        根据输出形状推断逆向计算的输入形状。
        默认保持形状不变。
        """
        return shape
    """
    组合多个变换以形成一个链式变换。
    被组合的变换负责缓存。

    Args:
        parts (list of :class:`Transform`): 要组合的变换列表。
        cache_size (int): 缓存的大小。如果为零，则不进行缓存。如果为一，
            则缓存最新的单个值。只支持0和1。

    """

    def __init__(self, parts: List[Transform], cache_size=0):
        # 如果缓存大小不为零，则对每个部分应用缓存
        if cache_size:
            parts = [part.with_cache(cache_size) for part in parts]
        # 调用父类的初始化方法
        super().__init__(cache_size=cache_size)
        # 将处理后的部分保存在实例变量中
        self.parts = parts

    def __eq__(self, other):
        # 检查是否与另一个ComposeTransform对象相等
        if not isinstance(other, ComposeTransform):
            return False
        # 比较两个对象的部分是否相同
        return self.parts == other.parts

    @constraints.dependent_property(is_discrete=False)
    def domain(self):
        # 如果没有部件，返回实数约束
        if not self.parts:
            return constraints.real
        # 初始域为第一个部件的域
        domain = self.parts[0].domain
        # 调整事件维度为所有部件中的最大值
        event_dim = self.parts[-1].codomain.event_dim
        # 反向遍历部件
        for part in reversed(self.parts):
            # 更新事件维度
            event_dim += part.domain.event_dim - part.codomain.event_dim
            # 取当前事件维度与部件域中的最大值
            event_dim = max(event_dim, part.domain.event_dim)
        # 确保事件维度大于等于域的事件维度
        assert event_dim >= domain.event_dim
        # 如果事件维度大于域的事件维度，使用独立约束增加域的维度
        if event_dim > domain.event_dim:
            domain = constraints.independent(domain, event_dim - domain.event_dim)
        return domain

    @constraints.dependent_property(is_discrete=False)
    def codomain(self):
        # 如果没有部件，返回实数约束
        if not self.parts:
            return constraints.real
        # 初始共域为最后一个部件的共域
        codomain = self.parts[-1].codomain
        # 调整事件维度为所有部件中的最大值
        event_dim = self.parts[0].domain.event_dim
        # 遍历所有部件
        for part in self.parts:
            # 更新事件维度
            event_dim += part.codomain.event_dim - part.domain.event_dim
            # 取当前事件维度与部件共域中的最大值
            event_dim = max(event_dim, part.codomain.event_dim)
        # 确保事件维度大于等于共域的事件维度
        assert event_dim >= codomain.event_dim
        # 如果事件维度大于共域的事件维度，使用独立约束增加共域的维度
        if event_dim > codomain.event_dim:
            codomain = constraints.independent(codomain, event_dim - codomain.event_dim)
        return codomain

    @lazy_property
    def bijective(self):
        # 检查所有部件是否双射
        return all(p.bijective for p in self.parts)

    @lazy_property
    def sign(self):
        # 计算所有部件的符号乘积
        sign = 1
        for p in self.parts:
            sign = sign * p.sign
        return sign

    @property
    def inv(self):
        # 获取逆变换，如果已缓存，则直接返回；否则创建新的组合逆变换
        inv = None
        if self._inv is not None:
            inv = self._inv()
        if inv is None:
            inv = ComposeTransform([p.inv for p in reversed(self.parts)])
            self._inv = weakref.ref(inv)
            inv._inv = weakref.ref(self)
        return inv

    def with_cache(self, cache_size=1):
        # 如果缓存大小与当前大小相同，则直接返回当前对象；否则返回新的组合变换对象
        if self._cache_size == cache_size:
            return self
        return ComposeTransform(self.parts, cache_size=cache_size)

    def __call__(self, x):
        # 对输入 x 应用所有部件的变换
        for part in self.parts:
            x = part(x)
        return x

    def log_abs_det_jacobian(self, x, y):
        # 如果没有部件，返回与输入 x 形状相同的零张量
        if not self.parts:
            return torch.zeros_like(x)

        # 计算中间值，如果部件[:-1]都已缓存，这步计算是免费的
        xs = [x]
        for part in self.parts[:-1]:
            xs.append(part(xs[-1]))
        xs.append(y)

        terms = []
        event_dim = self.domain.event_dim
        # 遍历部件、中间值 x 和 y
        for part, x, y in zip(self.parts, xs[:-1], xs[1:]):
            # 计算每个部件的对数绝对行列式雅可比
            terms.append(
                _sum_rightmost(
                    part.log_abs_det_jacobian(x, y), event_dim - part.domain.event_dim
                )
            )
            # 更新事件维度
            event_dim += part.codomain.event_dim - part.domain.event_dim
        # 返回所有部件的雅可比行列式的和
        return functools.reduce(operator.add, terms)

    def forward_shape(self, shape):
        # 对输入形状应用所有部件的形状前向变换
        for part in self.parts:
            shape = part.forward_shape(shape)
        return shape
    # 定义一个方法，用于计算给定形状的逆转形状
    def inverse_shape(self, shape):
        # 反向遍历存储在对象中的各部分，并递归计算逆转形状
        for part in reversed(self.parts):
            shape = part.inverse_shape(shape)
        # 返回最终的逆转形状
        return shape

    # 重写对象的字符串表示方法，返回对象的详细信息的字符串形式
    def __repr__(self):
        # 初始化字符串，表示对象类名和换行缩进
        fmt_string = self.__class__.__name__ + "(\n    "
        # 使用列表推导式生成包含每个部分字符串表示的列表，并用逗号和换行符连接
        fmt_string += ",\n    ".join([p.__repr__() for p in self.parts])
        # 结尾处添加换行和括号，表示对象信息结束
        fmt_string += "\n)"
        # 返回格式化后的字符串表示形式
        return fmt_string
# 定义一个身份变换，不对输入进行任何改变
identity_transform = ComposeTransform([])

# 定义一个独立变换类，包装另一个变换以将右侧的特定维度视为依赖的一部分
class IndependentTransform(Transform):
    """
    包装另一个变换以将 ``reinterpreted_batch_ndims`` 个额外的右侧维度视为依赖。
    这对于前向或反向变换没有影响，但会在 :meth:`log_abs_det_jacobian` 中对右侧的
    ``reinterpreted_batch_ndims`` 个维度进行求和。

    Args:
        base_transform (:class:`Transform`): 基础变换。
        reinterpreted_batch_ndims (int): 要视为依赖的额外右侧维度的数量。
    """

    def __init__(self, base_transform, reinterpreted_batch_ndims, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.base_transform = base_transform.with_cache(cache_size)
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return IndependentTransform(
            self.base_transform, self.reinterpreted_batch_ndims, cache_size=cache_size
        )

    @constraints.dependent_property(is_discrete=False)
    def domain(self):
        return constraints.independent(
            self.base_transform.domain, self.reinterpreted_batch_ndims
        )

    @constraints.dependent_property(is_discrete=False)
    def codomain(self):
        return constraints.independent(
            self.base_transform.codomain, self.reinterpreted_batch_ndims
        )

    @property
    def bijective(self):
        return self.base_transform.bijective

    @property
    def sign(self):
        return self.base_transform.sign

    def _call(self, x):
        if x.dim() < self.domain.event_dim:
            raise ValueError("Too few dimensions on input")
        return self.base_transform(x)

    def _inverse(self, y):
        if y.dim() < self.codomain.event_dim:
            raise ValueError("Too few dimensions on input")
        return self.base_transform.inv(y)

    def log_abs_det_jacobian(self, x, y):
        result = self.base_transform.log_abs_det_jacobian(x, y)
        result = _sum_rightmost(result, self.reinterpreted_batch_ndims)
        return result

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.base_transform)}, {self.reinterpreted_batch_ndims})"

    def forward_shape(self, shape):
        return self.base_transform.forward_shape(shape)

    def inverse_shape(self, shape):
        return self.base_transform.inverse_shape(shape)


class ReshapeTransform(Transform):
    """
    单位雅可比变换，用于重塑张量的右侧部分。

    注意，``in_shape`` 和 ``out_shape`` 必须具有相同数量的元素，就像 :meth:`torch.Tensor.reshape` 一样。

    Arguments:
        in_shape (torch.Size): 输入事件形状。
        out_shape (torch.Size): 输出事件形状。
    """

    bijective = True
    # 初始化函数，设置输入形状、输出形状，并可选地设置缓存大小
    def __init__(self, in_shape, out_shape, cache_size=0):
        # 将输入形状转换为 torch 的 Size 类型
        self.in_shape = torch.Size(in_shape)
        # 将输出形状转换为 torch 的 Size 类型
        self.out_shape = torch.Size(out_shape)
        # 检查输入形状和输出形状的元素数量是否相同，若不同则抛出 ValueError 异常
        if self.in_shape.numel() != self.out_shape.numel():
            raise ValueError("in_shape, out_shape have different numbers of elements")
        # 调用父类的构造函数，设置缓存大小
        super().__init__(cache_size=cache_size)

    # 返回输入变量的定义域约束，这里约束为实数类型
    @constraints.dependent_property
    def domain(self):
        return constraints.independent(constraints.real, len(self.in_shape))

    # 返回输出变量的值域约束，这里约束为实数类型
    @constraints.dependent_property
    def codomain(self):
        return constraints.independent(constraints.real, len(self.out_shape))

    # 返回一个新的 ReshapeTransform 对象，指定不同的缓存大小
    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return ReshapeTransform(self.in_shape, self.out_shape, cache_size=cache_size)

    # 实现正向变换，将输入 x 转换为输出，保持批次维度不变
    def _call(self, x):
        batch_shape = x.shape[: x.dim() - len(self.in_shape)]
        return x.reshape(batch_shape + self.out_shape)

    # 实现逆向变换，将输出 y 转换为输入，保持批次维度不变
    def _inverse(self, y):
        batch_shape = y.shape[: y.dim() - len(self.out_shape)]
        return y.reshape(batch_shape + self.in_shape)

    # 计算正向变换时的对数绝对值行列式的 Jacobian 行列式，返回一个全零张量
    def log_abs_det_jacobian(self, x, y):
        batch_shape = x.shape[: x.dim() - len(self.in_shape)]
        return x.new_zeros(batch_shape)

    # 根据输入的形状计算正向变换后的形状，并检查输入形状是否匹配当前输入形状
    def forward_shape(self, shape):
        # 若输入形状的维度少于当前输入形状的维度，抛出 ValueError 异常
        if len(shape) < len(self.in_shape):
            raise ValueError("Too few dimensions on input")
        # 计算截断位置，即保留的维度数
        cut = len(shape) - len(self.in_shape)
        # 检查截取后的形状是否与当前输入形状匹配，若不匹配则抛出 ValueError 异常
        if shape[cut:] != self.in_shape:
            raise ValueError(
                f"Shape mismatch: expected {shape[cut:]} but got {self.in_shape}"
            )
        # 返回正向变换后的形状，即保留截断位置之前的维度加上当前输出形状
        return shape[:cut] + self.out_shape

    # 根据输入的形状计算逆向变换后的形状，并检查输入形状是否匹配当前输出形状
    def inverse_shape(self, shape):
        # 若输入形状的维度少于当前输出形状的维度，抛出 ValueError 异常
        if len(shape) < len(self.out_shape):
            raise ValueError("Too few dimensions on input")
        # 计算截断位置，即保留的维度数
        cut = len(shape) - len(self.out_shape)
        # 检查截取后的形状是否与当前输出形状匹配，若不匹配则抛出 ValueError 异常
        if shape[cut:] != self.out_shape:
            raise ValueError(
                f"Shape mismatch: expected {shape[cut:]} but got {self.out_shape}"
            )
        # 返回逆向变换后的形状，即保留截断位置之前的维度加上当前输入形状
        return shape[:cut] + self.in_shape
class ExpTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \exp(x)`.
    """
    # 定义定义域为实数
    domain = constraints.real
    # 定义值域为正数
    codomain = constraints.positive
    # 表示是双射映射
    bijective = True
    # 映射是正数
    sign = +1

    # 判断是否与另一个ExpTransform对象相等
    def __eq__(self, other):
        return isinstance(other, ExpTransform)

    # 实现映射函数，计算指数函数
    def _call(self, x):
        return x.exp()

    # 实现反函数，计算对数函数
    def _inverse(self, y):
        return y.log()

    # 计算对数绝对值行列式雅可比行列式
    def log_abs_det_jacobian(self, x, y):
        return x


class PowerTransform(Transform):
    r"""
    Transform via the mapping :math:`y = x^{\text{exponent}}`.
    """
    # 定义定义域为正数
    domain = constraints.positive
    # 定义值域为正数
    codomain = constraints.positive
    # 表示是双射映射
    bijective = True

    # 初始化函数，设置指数和缓存大小
    def __init__(self, exponent, cache_size=0):
        super().__init__(cache_size=cache_size)
        (self.exponent,) = broadcast_all(exponent)

    # 根据缓存大小返回相应的对象
    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return PowerTransform(self.exponent, cache_size=cache_size)

    # 计算指数的符号
    @lazy_property
    def sign(self):
        return self.exponent.sign()

    # 判断是否与另一个PowerTransform对象相等
    def __eq__(self, other):
        if not isinstance(other, PowerTransform):
            return False
        return self.exponent.eq(other.exponent).all().item()

    # 实现映射函数，计算幂函数
    def _call(self, x):
        return x.pow(self.exponent)

    # 实现反函数，计算幂的倒数
    def _inverse(self, y):
        return y.pow(1 / self.exponent)

    # 计算对数绝对值行列式雅可比行列式
    def log_abs_det_jacobian(self, x, y):
        return (self.exponent * y / x).abs().log()

    # 计算正向变换的形状
    def forward_shape(self, shape):
        return torch.broadcast_shapes(shape, getattr(self.exponent, "shape", ()))

    # 计算反向变换的形状
    def inverse_shape(self, shape):
        return torch.broadcast_shapes(shape, getattr(self.exponent, "shape", ()))


def _clipped_sigmoid(x):
    # 获取浮点数的信息
    finfo = torch.finfo(x.dtype)
    # 对sigmoid函数的输出进行截断
    return torch.clamp(torch.sigmoid(x), min=finfo.tiny, max=1.0 - finfo.eps)


class SigmoidTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \frac{1}{1 + \exp(-x)}` and :math:`x = \text{logit}(y)`.
    """
    # 定义定义域为实数
    domain = constraints.real
    # 定义值域为单位区间
    codomain = constraints.unit_interval
    # 表示是双射映射
    bijective = True
    # 映射是正数
    sign = +1

    # 判断是否与另一个SigmoidTransform对象相等
    def __eq__(self, other):
        return isinstance(other, SigmoidTransform)

    # 实现映射函数，调用_clipped_sigmoid函数
    def _call(self, x):
        return _clipped_sigmoid(x)

    # 实现反函数，计算logit函数的反函数
    def _inverse(self, y):
        finfo = torch.finfo(y.dtype)
        # 对y进行截断
        y = y.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
        return y.log() - (-y).log1p()

    # 计算对数绝对值行列式雅可比行列式
    def log_abs_det_jacobian(self, x, y):
        return -F.softplus(-x) - F.softplus(x)


class SoftplusTransform(Transform):
    r"""
    Transform via the mapping :math:`\text{Softplus}(x) = \log(1 + \exp(x))`.
    The implementation reverts to the linear function when :math:`x > 20`.
    """
    # 定义定义域为实数
    domain = constraints.real
    # 定义值域为正数
    codomain = constraints.positive
    # 表示是双射映射
    bijective = True
    # 映射是正数
    sign = +1

    # 判断是否与另一个SoftplusTransform对象相等
    def __eq__(self, other):
        return isinstance(other, SoftplusTransform)

    # 实现映射函数，调用softplus函数
    def _call(self, x):
        return softplus(x)

    # 实现反函数，计算softplus函数的反函数
    def _inverse(self, y):
        return (-y).expm1().neg().log() + y
    # 定义一个方法用于计算对数绝对行列式雅可比的负值
    def log_abs_det_jacobian(self, x, y):
        # 使用 softplus 函数计算 -x 的值，然后再取其相反数
        return -softplus(-x)
class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.

    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.

    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.

    """
    # 定义了一个基于双曲正切函数的变换类
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __eq__(self, other):
        # 检查两个变换是否相等
        return isinstance(other, TanhTransform)

    def _call(self, x):
        # 实现变换的正向映射，返回输入的双曲正切值
        return x.tanh()

    def _inverse(self, y):
        # 实现变换的逆向映射，返回输入的反双曲正切值
        # 在这里不会将值夹在边界上，因为这可能降低某些算法的性能
        # 建议使用 `cache_size=1`
        return torch.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # 计算对数绝对值行列式的雅可比对数
        # 使用了一个更稳定的公式，详细信息见链接：
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2.0 * (math.log(2.0) - x - softplus(-2.0 * x))


class AbsTransform(Transform):
    r"""
    Transform via the mapping :math:`y = |x|`.
    """
    # 定义了一个取绝对值的变换类
    domain = constraints.real
    codomain = constraints.positive

    def __eq__(self, other):
        # 检查两个变换是否相等
        return isinstance(other, AbsTransform)

    def _call(self, x):
        # 实现变换的正向映射，返回输入的绝对值
        return x.abs()

    def _inverse(self, y):
        # 实现变换的逆向映射，返回输入值本身
        return y


class AffineTransform(Transform):
    r"""
    Transform via the pointwise affine mapping :math:`y = \text{loc} + \text{scale} \times x`.

    Args:
        loc (Tensor or float): Location parameter.
        scale (Tensor or float): Scale parameter.
        event_dim (int): Optional size of `event_shape`. This should be zero
            for univariate random variables, 1 for distributions over vectors,
            2 for distributions over matrices, etc.
    """
    # 定义了一个仿射变换类，通过点式仿射映射变换
    bijective = True

    def __init__(self, loc, scale, event_dim=0, cache_size=0):
        # 初始化函数，设置仿射变换的参数
        super().__init__(cache_size=cache_size)
        self.loc = loc
        self.scale = scale
        self._event_dim = event_dim

    @property
    def event_dim(self):
        # 返回事件维度的属性
        return self._event_dim

    @constraints.dependent_property(is_discrete=False)
    def domain(self):
        # 返回定义域的属性，根据事件维度的不同返回不同的约束条件
        if self.event_dim == 0:
            return constraints.real
        return constraints.independent(constraints.real, self.event_dim)

    @constraints.dependent_property(is_discrete=False)
    def codomain(self):
        # 返回值域的属性，根据事件维度的不同返回不同的约束条件
        if self.event_dim == 0:
            return constraints.real
        return constraints.independent(constraints.real, self.event_dim)

    def with_cache(self, cache_size=1):
        # 返回一个带有指定缓存大小的新的仿射变换对象
        if self._cache_size == cache_size:
            return self
        return AffineTransform(
            self.loc, self.scale, self.event_dim, cache_size=cache_size
        )
    # 比较两个 AffineTransform 对象是否相等
    def __eq__(self, other):
        # 如果 other 不是 AffineTransform 类型，则返回 False
        if not isinstance(other, AffineTransform):
            return False

        # 检查 self.loc 和 other.loc 是否都是数字类型
        if isinstance(self.loc, numbers.Number) and isinstance(
            other.loc, numbers.Number
        ):
            # 如果是数字类型，比较它们的值
            if self.loc != other.loc:
                return False
        else:
            # 如果不是数字类型，比较它们的元素是否全部相等
            if not (self.loc == other.loc).all().item():
                return False

        # 检查 self.scale 和 other.scale 是否都是数字类型
        if isinstance(self.scale, numbers.Number) and isinstance(
            other.scale, numbers.Number
        ):
            # 如果是数字类型，比较它们的值
            if self.scale != other.scale:
                return False
        else:
            # 如果不是数字类型，比较它们的元素是否全部相等
            if not (self.scale == other.scale).all().item():
                return False

        # 如果所有条件都满足，则返回 True，表示相等
        return True

    # 计算 AffineTransform 对象的符号
    @property
    def sign(self):
        # 如果 self.scale 是实数类型，返回其符号
        if isinstance(self.scale, numbers.Real):
            return 1 if float(self.scale) > 0 else -1 if float(self.scale) < 0 else 0
        # 否则调用 self.scale 对象的 sign 方法返回符号
        return self.scale.sign()

    # 计算仿射变换后的结果
    def _call(self, x):
        return self.loc + self.scale * x

    # 计算仿射变换的逆操作
    def _inverse(self, y):
        return (y - self.loc) / self.scale

    # 计算对数绝对值行列式的雅可比对数
    def log_abs_det_jacobian(self, x, y):
        shape = x.shape
        scale = self.scale
        # 如果 scale 是实数类型，结果是对 x 元素的对数绝对值 scale 的填充
        if isinstance(scale, numbers.Real):
            result = torch.full_like(x, math.log(abs(scale)))
        else:
            # 否则结果是 scale 的绝对值的对数
            result = torch.abs(scale).log()
        # 如果有 event_dim 属性，对结果进行形状调整并求和
        if self.event_dim:
            result_size = result.size()[: -self.event_dim] + (-1,)
            result = result.view(result_size).sum(-1)
            shape = shape[: -self.event_dim]
        # 扩展结果的形状以匹配 x 的形状
        return result.expand(shape)

    # 计算前向操作的输出形状
    def forward_shape(self, shape):
        return torch.broadcast_shapes(
            shape, getattr(self.loc, "shape", ()), getattr(self.scale, "shape", ())
        )

    # 计算逆操作的输出形状
    def inverse_shape(self, shape):
        return torch.broadcast_shapes(
            shape, getattr(self.loc, "shape", ()), getattr(self.scale, "shape", ())
        )
class CorrCholeskyTransform(Transform):
    r"""
    将长度为 :math:`D*(D-1)/2` 的无约束实数向量 :math:`x` 转换为 D 维相关矩阵的 Cholesky 因子。
    这个 Cholesky 因子是一个具有正对角线和每行单位欧几里得范数的下三角矩阵。
    转换过程如下：

        1. 首先将 x 转换为行顺序的下三角矩阵。
        2. 对于下三角部分的每一行 :math:`X_i`，使用 :class:`StickBreakingTransform` 类的带符号版本将 :math:`X_i` 转换为
           一个具有单位欧几里得长度的向量，具体步骤如下：
           - 缩放到区间 :math:`(-1, 1)`：:math:`r_i = \tanh(X_i)`.
           - 转换到无符号区间：:math:`z_i = r_i^2`.
           - 应用 :math:`s_i = StickBreakingTransform(z_i)`.
           - 转换回带符号区间：:math:`y_i = sign(r_i) * \sqrt{s_i}`.

    """
    domain = constraints.real_vector
    codomain = constraints.corr_cholesky
    bijective = True

    def _call(self, x):
        # 对输入向量 x 应用双曲正切函数
        x = torch.tanh(x)
        eps = torch.finfo(x.dtype).eps
        # 将 x 的值限制在 (-1 + eps, 1 - eps) 的范围内
        x = x.clamp(min=-1 + eps, max=1 - eps)
        # 将 x 转换为下三角矩阵 r
        r = vec_to_tril_matrix(x, diag=-1)
        # 对平方值应用 stick-breaking 转换
        z = r**2
        z1m_cumprod_sqrt = (1 - z).sqrt().cumprod(-1)
        # 对角元素必须为 1
        r = r + torch.eye(r.shape[-1], dtype=r.dtype, device=r.device)
        # 计算最终结果 y
        y = r * pad(z1m_cumprod_sqrt[..., :-1], [1, 0], value=1)
        return y

    def _inverse(self, y):
        # 反向 stick-breaking 转换
        # 参考：https://mc-stan.org/docs/2_18/reference-manual/cholesky-factors-of-correlation-matrices-1.html
        y_cumsum = 1 - torch.cumsum(y * y, dim=-1)
        y_cumsum_shifted = pad(y_cumsum[..., :-1], [1, 0], value=1)
        # 将 y 转换为向量形式
        y_vec = tril_matrix_to_vec(y, diag=-1)
        y_cumsum_vec = tril_matrix_to_vec(y_cumsum_shifted, diag=-1)
        # 计算逆 tanh 函数
        t = y_vec / (y_cumsum_vec).sqrt()
        x = (t.log1p() - t.neg().log1p()) / 2
        return x
    def log_abs_det_jacobian(self, x, y, intermediates=None):
        # 计算 log(abs(det(Jacobian)))，其中 x 和 y 的维度不同，Jacobian 行列式不是严格定义的。
        # 返回 x 的 log_abs_det_jacobian 和 y 的展平下三角部分的 log_abs_det_jacobian。

        # 计算 1 - y*y 沿着最后一个维度的累积和
        y1m_cumsum = 1 - (y * y).cumsum(dim=-1)
        # 将下三角矩阵 y1m_cumsum 转换为向量，对角线为 -2
        y1m_cumsum_tril = tril_matrix_to_vec(y1m_cumsum, diag=-2)
        # 计算 stick_breaking_logdet，即 y1m_cumsum_tril 的 0.5 倍的对数和
        stick_breaking_logdet = 0.5 * (y1m_cumsum_tril).log().sum(-1)
        # 计算 tanh_logdet，即 x + softplus(-2 * x) - log(2.0) 的 -2 倍的和
        tanh_logdet = -2 * (x + softplus(-2 * x) - math.log(2.0)).sum(dim=-1)
        # 返回 log_abs_det_jacobian 的总和
        return stick_breaking_logdet + tanh_logdet

    def forward_shape(self, shape):
        # 将形状从 (..., N) 重塑为 (..., D, D)。

        if len(shape) < 1:
            raise ValueError("输入维度太少")
        N = shape[-1]
        D = round((0.25 + 2 * N) ** 0.5 + 0.5)
        if D * (D - 1) // 2 != N:
            raise ValueError("输入不是展平的下三角数")
        # 返回重塑后的形状
        return shape[:-1] + (D, D)

    def inverse_shape(self, shape):
        # 将形状从 (..., D, D) 重塑为 (..., N)。

        if len(shape) < 2:
            raise ValueError("输入维度太少")
        if shape[-2] != shape[-1]:
            raise ValueError("输入不是方阵")
        D = shape[-1]
        N = D * (D - 1) // 2
        # 返回重塑后的形状
        return shape[:-2] + (N,)
class SoftmaxTransform(Transform):
    r"""
    Transform from unconstrained space to the simplex via :math:`y = \exp(x)` then
    normalizing.

    This is not bijective and cannot be used for HMC. However this acts mostly
    coordinate-wise (except for the final normalization), and thus is
    appropriate for coordinate-wise optimization algorithms.
    """
    # 定义定义域为实数向量和值域为单纯形的约束
    domain = constraints.real_vector
    codomain = constraints.simplex

    # 判断是否相等的方法
    def __eq__(self, other):
        return isinstance(other, SoftmaxTransform)

    # 转换函数的实现，将输入的对数概率转换为概率分布
    def _call(self, x):
        logprobs = x
        # 计算指数，减去最大值后再进行指数化，得到概率分布
        probs = (logprobs - logprobs.max(-1, True)[0]).exp()
        # 进行归一化操作
        return probs / probs.sum(-1, True)

    # 逆转换函数的实现，将概率分布转换为对数概率
    def _inverse(self, y):
        probs = y
        return probs.log()

    # 前向变换的形状处理函数，确保输入的维度不少于1
    def forward_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape

    # 逆向变换的形状处理函数，确保输入的维度不少于1
    def inverse_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape


class StickBreakingTransform(Transform):
    """
    Transform from unconstrained space to the simplex of one additional
    dimension via a stick-breaking process.

    This transform arises as an iterated sigmoid transform in a stick-breaking
    construction of the `Dirichlet` distribution: the first logit is
    transformed via sigmoid to the first probability and the probability of
    everything else, and then the process recurses.

    This is bijective and appropriate for use in HMC; however it mixes
    coordinates together and is less appropriate for optimization.
    """

    # 定义定义域为实数向量和值域为单纯形的约束，并且是双射的
    domain = constraints.real_vector
    codomain = constraints.simplex
    bijective = True

    # 判断是否相等的方法
    def __eq__(self, other):
        return isinstance(other, StickBreakingTransform)

    # 转换函数的实现，通过分段破棍过程将输入的向量转换为单纯形上的向量
    def _call(self, x):
        offset = x.shape[-1] + 1 - x.new_ones(x.shape[-1]).cumsum(-1)
        z = _clipped_sigmoid(x - offset.log())
        z_cumprod = (1 - z).cumprod(-1)
        y = pad(z, [0, 1], value=1) * pad(z_cumprod, [1, 0], value=1)
        return y

    # 逆转换函数的实现，将单纯形上的向量转换为对应的实数向量
    def _inverse(self, y):
        y_crop = y[..., :-1]
        offset = y.shape[-1] - y.new_ones(y_crop.shape[-1]).cumsum(-1)
        sf = 1 - y_crop.cumsum(-1)
        # 确保 sf 是正数，避免出现 y[-1] ~ 0 或者 y[:-1].sum() ~ 1 的情况
        sf = torch.clamp(sf, min=torch.finfo(y.dtype).tiny)
        x = y_crop.log() - sf.log() + offset.log()
        return x

    # 计算正向变换的对数行列式雅可比行列式的函数
    def log_abs_det_jacobian(self, x, y):
        offset = x.shape[-1] + 1 - x.new_ones(x.shape[-1]).cumsum(-1)
        x = x - offset.log()
        # 使用恒等式 1 - sigmoid(x) = exp(-x) * sigmoid(x)
        detJ = (-x + F.logsigmoid(x) + y[..., :-1].log()).sum(-1)
        return detJ

    # 前向变换的形状处理函数，确保输入的维度不少于1
    def forward_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape[:-1] + (shape[-1] + 1,)
    # 定义一个方法 inverse_shape，接受一个参数 shape
    def inverse_shape(self, shape):
        # 如果 shape 的长度小于 1，抛出 ValueError 异常，提示输入维度过少
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        # 返回一个新的元组，这个元组由 shape 切片取出除最后一个元素外的所有元素，再加上最后一个元素减一构成
        return shape[:-1] + (shape[-1] - 1,)
class LowerCholeskyTransform(Transform):
    """
    Transform from unconstrained matrices to lower-triangular matrices with
    nonnegative diagonal entries.

    This is useful for parameterizing positive definite matrices in terms of
    their Cholesky factorization.
    """

    # 定义定义域为两个独立的实数约束
    domain = constraints.independent(constraints.real, 2)
    # 定义值域为下三角约束
    codomain = constraints.lower_cholesky

    def __eq__(self, other):
        # 判断是否与另一个 LowerCholeskyTransform 对象相等
        return isinstance(other, LowerCholeskyTransform)

    def _call(self, x):
        # 将输入矩阵 x 转换为下三角矩阵，并确保对角线元素为非负值
        return x.tril(-1) + x.diagonal(dim1=-2, dim2=-1).exp().diag_embed()

    def _inverse(self, y):
        # 从下三角矩阵 y 还原为原始矩阵 x
        return y.tril(-1) + y.diagonal(dim1=-2, dim2=-1).log().diag_embed()


class PositiveDefiniteTransform(Transform):
    """
    Transform from unconstrained matrices to positive-definite matrices.
    """

    # 定义定义域为两个独立的实数约束
    domain = constraints.independent(constraints.real, 2)
    # 定义值域为正定矩阵约束
    codomain = constraints.positive_definite  # type: ignore[assignment]

    def __eq__(self, other):
        # 判断是否与另一个 PositiveDefiniteTransform 对象相等
        return isinstance(other, PositiveDefiniteTransform)

    def _call(self, x):
        # 使用 LowerCholeskyTransform 将输入矩阵 x 转换为下三角矩阵，并求其乘积 x @ x^T
        x = LowerCholeskyTransform()(x)
        return x @ x.mT

    def _inverse(self, y):
        # 使用 Torch 函数计算正定矩阵 y 的 Cholesky 分解，并使用 LowerCholeskyTransform 进行逆变换
        y = torch.linalg.cholesky(y)
        return LowerCholeskyTransform().inv(y)


class CatTransform(Transform):
    """
    Transform functor that applies a sequence of transforms `tseq`
    component-wise to each submatrix at `dim`, of length `lengths[dim]`,
    in a way compatible with :func:`torch.cat`.

    Example::

       x0 = torch.cat([torch.range(1, 10), torch.range(1, 10)], dim=0)
       x = torch.cat([x0, x0], dim=0)
       t0 = CatTransform([ExpTransform(), identity_transform], dim=0, lengths=[10, 10])
       t = CatTransform([t0, t0], dim=0, lengths=[20, 20])
       y = t(x)
    """

    transforms: List[Transform]

    def __init__(self, tseq, dim=0, lengths=None, cache_size=0):
        assert all(isinstance(t, Transform) for t in tseq)
        if cache_size:
            tseq = [t.with_cache(cache_size) for t in tseq]
        super().__init__(cache_size=cache_size)
        # 初始化 CatTransform 对象时，传入的转换序列和长度信息，并进行缓存设置
        self.transforms = list(tseq)
        if lengths is None:
            lengths = [1] * len(self.transforms)
        self.lengths = list(lengths)
        assert len(self.lengths) == len(self.transforms)
        self.dim = dim

    @lazy_property
    def event_dim(self):
        # 返回转换序列中最大的事件维度
        return max(t.event_dim for t in self.transforms)

    @lazy_property
    def length(self):
        # 返回所有子矩阵的总长度
        return sum(self.lengths)

    def with_cache(self, cache_size=1):
        # 返回一个带有新缓存大小的 CatTransform 对象，如果缓存大小相同则返回自身
        if self._cache_size == cache_size:
            return self
        return CatTransform(self.transforms, self.dim, self.lengths, cache_size)
    # 定义一个方法 `_call`，用于将输入张量 x 沿指定维度进行分块变换
    def _call(self, x):
        # 断言维度索引在有效范围内
        assert -x.dim() <= self.dim < x.dim()
        # 断言输入张量 x 在指定维度上的大小与变换对象的长度一致
        assert x.size(self.dim) == self.length
        # 初始化空列表用于存储分块变换后的结果
        yslices = []
        start = 0
        # 遍历变换对象列表和对应的长度
        for trans, length in zip(self.transforms, self.lengths):
            # 在指定维度上获取 x 的切片
            xslice = x.narrow(self.dim, start, length)
            # 对切片进行变换，并将结果添加到 yslices 中
            yslices.append(trans(xslice))
            start = start + length  # 避免使用 += 以兼容 JIT
        # 将所有分块变换后的结果在指定维度上连接起来，并返回结果张量
        return torch.cat(yslices, dim=self.dim)

    # 定义一个方法 `_inverse`，用于将输出张量 y 沿指定维度进行逆变换
    def _inverse(self, y):
        # 断言维度索引在有效范围内
        assert -y.dim() <= self.dim < y.dim()
        # 断言输出张量 y 在指定维度上的大小与变换对象的长度一致
        assert y.size(self.dim) == self.length
        # 初始化空列表用于存储逆变换后的结果
        xslices = []
        start = 0
        # 遍历变换对象列表和对应的长度
        for trans, length in zip(self.transforms, self.lengths):
            # 在指定维度上获取 y 的切片
            yslice = y.narrow(self.dim, start, length)
            # 对切片进行逆变换，并将结果添加到 xslices 中
            xslices.append(trans.inv(yslice))
            start = start + length  # 避免使用 += 以兼容 JIT
        # 将所有逆变换后的结果在指定维度上连接起来，并返回结果张量
        return torch.cat(xslices, dim=self.dim)

    # 定义一个方法 `log_abs_det_jacobian`，计算输入 x 到输出 y 的对数绝对行列式雅可比的和或连接
    def log_abs_det_jacobian(self, x, y):
        # 断言输入张量 x 的维度索引在有效范围内
        assert -x.dim() <= self.dim < x.dim()
        # 断言输入张量 x 在指定维度上的大小与变换对象的长度一致
        assert x.size(self.dim) == self.length
        # 断言输出张量 y 的维度索引在有效范围内
        assert -y.dim() <= self.dim < y.dim()
        # 断言输出张量 y 在指定维度上的大小与变换对象的长度一致
        assert y.size(self.dim) == self.length
        # 初始化空列表用于存储每个分块变换的对数绝对行列式雅可比
        logdetjacs = []
        start = 0
        # 遍历变换对象列表和对应的长度
        for trans, length in zip(self.transforms, self.lengths):
            # 在指定维度上获取 x 和 y 的切片
            xslice = x.narrow(self.dim, start, length)
            yslice = y.narrow(self.dim, start, length)
            # 计算当前变换的对数绝对行列式雅可比
            logdetjac = trans.log_abs_det_jacobian(xslice, yslice)
            # 如果当前变换的事件维度小于自身的事件维度，则进行求和操作以匹配维度
            if trans.event_dim < self.event_dim:
                logdetjac = _sum_rightmost(logdetjac, self.event_dim - trans.event_dim)
            # 将当前变换的对数绝对行列式雅可比添加到列表中
            logdetjacs.append(logdetjac)
            start = start + length  # 避免使用 += 以兼容 JIT
        # 决定是连接还是求和操作的维度
        dim = self.dim
        if dim >= 0:
            dim = dim - x.dim()
        dim = dim + self.event_dim
        if dim < 0:
            return torch.cat(logdetjacs, dim=dim)
        else:
            return sum(logdetjacs)

    # 定义一个属性 `bijective`，判断所有变换对象是否都是双射的
    @property
    def bijective(self):
        return all(t.bijective for t in self.transforms)

    # 定义一个依赖属性 `domain`，返回所有变换对象的定义域的连接
    @constraints.dependent_property
    def domain(self):
        return constraints.cat(
            [t.domain for t in self.transforms], self.dim, self.lengths
        )

    # 定义一个依赖属性 `codomain`，返回所有变换对象的值域的连接
    @constraints.dependent_property
    def codomain(self):
        return constraints.cat(
            [t.codomain for t in self.transforms], self.dim, self.lengths
        )
class StackTransform(Transform):
    """
    Transform functor that applies a sequence of transforms `tseq`
    component-wise to each submatrix at `dim`
    in a way compatible with :func:`torch.stack`.

    Example::

       x = torch.stack([torch.range(1, 10), torch.range(1, 10)], dim=1)
       t = StackTransform([ExpTransform(), identity_transform], dim=1)
       y = t(x)
    """

    transforms: List[Transform]

    def __init__(self, tseq, dim=0, cache_size=0):
        assert all(isinstance(t, Transform) for t in tseq)
        if cache_size:
            tseq = [t.with_cache(cache_size) for t in tseq]
        super().__init__(cache_size=cache_size)
        self.transforms = list(tseq)  # 初始化时将输入的变换序列转为列表并赋给transforms属性
        self.dim = dim  # 将输入的dim赋给对象的dim属性

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return StackTransform(self.transforms, self.dim, cache_size)  # 如果缓存大小和当前缓存大小相同，则返回当前对象；否则返回一个新的StackTransform对象

    def _slice(self, z):
        return [z.select(self.dim, i) for i in range(z.size(self.dim))]  # 对输入的张量z按维度dim进行切片操作并返回切片列表

    def _call(self, x):
        assert -x.dim() <= self.dim < x.dim()
        assert x.size(self.dim) == len(self.transforms)
        yslices = []
        for xslice, trans in zip(self._slice(x), self.transforms):
            yslices.append(trans(xslice))  # 对每个切片应用相应的变换并将结果加入到yslices列表中
        return torch.stack(yslices, dim=self.dim)  # 将所有变换后的切片再次堆叠成张量并返回

    def _inverse(self, y):
        assert -y.dim() <= self.dim < y.dim()
        assert y.size(self.dim) == len(self.transforms)
        xslices = []
        for yslice, trans in zip(self._slice(y), self.transforms):
            xslices.append(trans.inv(yslice))  # 对每个切片应用逆变换并将结果加入到xslices列表中
        return torch.stack(xslices, dim=self.dim)  # 将所有逆变换后的切片再次堆叠成张量并返回

    def log_abs_det_jacobian(self, x, y):
        assert -x.dim() <= self.dim < x.dim()
        assert x.size(self.dim) == len(self.transforms)
        assert -y.dim() <= self.dim < y.dim()
        assert y.size(self.dim) == len(self.transforms)
        logdetjacs = []
        yslices = self._slice(y)
        xslices = self._slice(x)
        for xslice, yslice, trans in zip(xslices, yslices, self.transforms):
            logdetjacs.append(trans.log_abs_det_jacobian(xslice, yslice))  # 计算每个变换的对数绝对值行列式的Jacobian，并将结果加入到logdetjacs列表中
        return torch.stack(logdetjacs, dim=self.dim)  # 将所有Jacobian结果堆叠成张量并返回

    @property
    def bijective(self):
        return all(t.bijective for t in self.transforms)  # 返回所有变换是否都是双射的布尔值

    @constraints.dependent_property
    def domain(self):
        return constraints.stack([t.domain for t in self.transforms], self.dim)  # 返回所有变换的定义域堆叠后的约束

    @constraints.dependent_property
    def codomain(self):
        return constraints.stack([t.codomain for t in self.transforms], self.dim)  # 返回所有变换的值域堆叠后的约束


class CumulativeDistributionTransform(Transform):
    """
    Transform via the cumulative distribution function of a probability distribution.

    Args:
        distribution (Distribution): Distribution whose cumulative distribution function to use for
            the transformation.
    """
    bijective = True
    codomain = constraints.unit_interval
    sign = +1


# 设置变量bijective为True，表明这是一个双射变换
bijective = True
# 设置变量codomain为constraints.unit_interval，表示定义域为单位间隔[0, 1]
codomain = constraints.unit_interval
# 设置变量sign为+1，表示这是一个正向变换
sign = +1



    def __init__(self, distribution, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.distribution = distribution


# 初始化方法，接受分布和缓存大小作为参数
def __init__(self, distribution, cache_size=0):
    # 调用父类的初始化方法，设置缓存大小
    super().__init__(cache_size=cache_size)
    # 将传入的分布对象保存到实例变量中
    self.distribution = distribution



    @property
    def domain(self):
        return self.distribution.support


# 定义一个属性方法domain，返回分布对象的支持集
@property
def domain(self):
    return self.distribution.support



    def _call(self, x):
        return self.distribution.cdf(x)


# 定义一个私有方法_call，接受参数x，并返回分布对象的累积分布函数在x处的取值
def _call(self, x):
    return self.distribution.cdf(x)



    def _inverse(self, y):
        return self.distribution.icdf(y)


# 定义一个私有方法_inverse，接受参数y，并返回分布对象的逆累积分布函数在y处的取值
def _inverse(self, y):
    return self.distribution.icdf(y)



    def log_abs_det_jacobian(self, x, y):
        return self.distribution.log_prob(x)


# 定义方法log_abs_det_jacobian，接受参数x和y，返回分布对象在x处的对数绝对行列式的雅可比对数
def log_abs_det_jacobian(self, x, y):
    return self.distribution.log_prob(x)



    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return CumulativeDistributionTransform(self.distribution, cache_size=cache_size)


# 定义方法with_cache，接受参数cache_size，返回一个带有指定缓存大小的新CumulativeDistributionTransform对象
def with_cache(self, cache_size=1):
    # 如果当前对象的缓存大小与参数指定的大小相同，则直接返回当前对象
    if self._cache_size == cache_size:
        return self
    # 否则，返回一个新的CumulativeDistributionTransform对象，带有指定的缓存大小
    return CumulativeDistributionTransform(self.distribution, cache_size=cache_size)
```