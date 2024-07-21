# `.\pytorch\test\distributions\test_transforms.py`

```
# 导入所需模块和函数库，包括IO操作、数值判断、测试框架和PyTorch相关模块
import io
from numbers import Number
import pytest
import torch
from torch.autograd import grad
from torch.autograd.functional import jacobian
from torch.distributions import (
    constraints,
    Dirichlet,
    Independent,
    Normal,
    TransformedDistribution,
)
from torch.distributions.transforms import (
    _InverseTransform,
    AbsTransform,
    AffineTransform,
    ComposeTransform,
    CorrCholeskyTransform,
    CumulativeDistributionTransform,
    ExpTransform,
    identity_transform,
    IndependentTransform,
    LowerCholeskyTransform,
    PositiveDefiniteTransform,
    PowerTransform,
    ReshapeTransform,
    SigmoidTransform,
    SoftmaxTransform,
    SoftplusTransform,
    StickBreakingTransform,
    TanhTransform,
    Transform,
)
from torch.distributions.utils import tril_matrix_to_vec, vec_to_tril_matrix
from torch.testing._internal.common_utils import run_tests

# 定义一个函数get_transforms，接受一个参数cache_size
def get_transforms(cache_size):
    # 创建变换对象列表，用于数据变换
    transforms = [
        # 绝对值变换，设置缓存大小
        AbsTransform(cache_size=cache_size),
        # 指数变换，设置缓存大小
        ExpTransform(cache_size=cache_size),
        # 幂次方变换，指数为2，设置缓存大小
        PowerTransform(exponent=2, cache_size=cache_size),
        # 幂次方变换，指数为-2，设置缓存大小
        PowerTransform(exponent=-2, cache_size=cache_size),
        # 幂次方变换，指数为服从正态分布的5.0，设置缓存大小
        PowerTransform(exponent=torch.tensor(5.0).normal_(), cache_size=cache_size),
        # 幂次方变换，指数为服从正态分布的5.0，设置缓存大小
        PowerTransform(exponent=torch.tensor(5.0).normal_(), cache_size=cache_size),
        # Sigmoid变换，设置缓存大小
        SigmoidTransform(cache_size=cache_size),
        # Tanh变换，设置缓存大小
        TanhTransform(cache_size=cache_size),
        # 仿射变换，从0到1，设置缓存大小
        AffineTransform(0, 1, cache_size=cache_size),
        # 仿射变换，从1到-2，设置缓存大小
        AffineTransform(1, -2, cache_size=cache_size),
        # 仿射变换，从服从正态分布的5个元素的张量到服从正态分布的5个元素的张量，设置缓存大小
        AffineTransform(torch.randn(5), torch.randn(5), cache_size=cache_size),
        # 仿射变换，从服从正态分布的4x5张量到服从正态分布的4x5张量，设置缓存大小
        AffineTransform(torch.randn(4, 5), torch.randn(4, 5), cache_size=cache_size),
        # Softmax变换，设置缓存大小
        SoftmaxTransform(cache_size=cache_size),
        # Softplus变换，设置缓存大小
        SoftplusTransform(cache_size=cache_size),
        # Stick-breaking变换，设置缓存大小
        StickBreakingTransform(cache_size=cache_size),
        # 下三角Cholesky分解变换，设置缓存大小
        LowerCholeskyTransform(cache_size=cache_size),
        # 相关Cholesky分解变换，设置缓存大小
        CorrCholeskyTransform(cache_size=cache_size),
        # 正定变换，设置缓存大小
        PositiveDefiniteTransform(cache_size=cache_size),
        # 复合变换，包含一个从服从正态分布的4x5张量到服从正态分布的4x5张量的仿射变换
        ComposeTransform(
            [
                AffineTransform(
                    torch.randn(4, 5), torch.randn(4, 5), cache_size=cache_size
                ),
            ]
        ),
        # 复合变换，包含一个从服从正态分布的4x5张量到服从正态分布的4x5张量的仿射变换和指数变换，设置缓存大小
        ComposeTransform(
            [
                AffineTransform(
                    torch.randn(4, 5), torch.randn(4, 5), cache_size=cache_size
                ),
                ExpTransform(cache_size=cache_size),
            ]
        ),
        # 复合变换，包含从0到1的仿射变换、从服从正态分布的4x5张量到服从正态分布的4x5张量的仿射变换、从1到-2的仿射变换、从服从正态分布的4x5张量到服从正态分布的4x5张量的仿射变换
        ComposeTransform(
            [
                AffineTransform(0, 1, cache_size=cache_size),
                AffineTransform(
                    torch.randn(4, 5), torch.randn(4, 5), cache_size=cache_size
                ),
                AffineTransform(1, -2, cache_size=cache_size),
                AffineTransform(
                    torch.randn(4, 5), torch.randn(4, 5), cache_size=cache_size
                ),
            ]
        ),
        # 重塑变换，从(4, 5)到(2, 5, 2)
        ReshapeTransform((4, 5), (2, 5, 2)),
        # 独立变换，包含一个从服从正态分布的5个元素的张量到服从正态分布的5个元素的张量的仿射变换，轴为1
        IndependentTransform(
            AffineTransform(torch.randn(5), torch.randn(5), cache_size=cache_size), 1
        ),
        # 累积分布变换，基于均值0标准差1的正态分布
        CumulativeDistributionTransform(Normal(0, 1)),
    ]
    
    # 将每个变换对象的逆变换添加到列表中
    transforms += [t.inv for t in transforms]
    
    # 返回变换对象列表
    return transforms
# 对变换进行重塑以便进行雅可比测试
def reshape_transform(transform, shape):
    # 如果是仿射变换
    if isinstance(transform, AffineTransform):
        # 如果变换的 loc 是数字，则直接返回该变换
        if isinstance(transform.loc, Number):
            return transform
        # 尝试根据 shape 扩展 loc 和 scale 创建新的 AffineTransform
        try:
            return AffineTransform(
                transform.loc.expand(shape),
                transform.scale.expand(shape),
                cache_size=transform._cache_size,
            )
        # 如果扩展失败，则尝试根据 shape 重塑 loc 和 scale 创建新的 AffineTransform
        except RuntimeError:
            return AffineTransform(
                transform.loc.reshape(shape),
                transform.scale.reshape(shape),
                cache_size=transform._cache_size,
            )
    # 如果是组合变换
    if isinstance(transform, ComposeTransform):
        # 对组合变换中的每个部分进行重塑
        reshaped_parts = []
        for p in transform.parts:
            reshaped_parts.append(reshape_transform(p, shape))
        return ComposeTransform(reshaped_parts, cache_size=transform._cache_size)
    # 如果逆变换是仿射变换，则递归重塑逆变换
    if isinstance(transform.inv, AffineTransform):
        return reshape_transform(transform.inv, shape).inv
    # 如果逆变换是组合变换，则递归重塑逆变换
    if isinstance(transform.inv, ComposeTransform):
        return reshape_transform(transform.inv, shape).inv
    # 其他情况下直接返回原始变换
    return transform


# 生成 pytest 的标识符
def transform_id(x):
    assert isinstance(x, Transform)
    # 根据变换类型生成相应的标识符，如果是逆变换，则标识符包含 "Inv()" 表示
    name = (
        f"Inv({type(x._inv).__name__})"
        if isinstance(x, _InverseTransform)
        else f"{type(x).__name__}"
    )
    return f"{name}(cache_size={x._cache_size})"


# 生成数据用于测试
def generate_data(transform):
    torch.manual_seed(1)
    # 处理 IndependentTransform，找到其基本变换
    while isinstance(transform, IndependentTransform):
        transform = transform.base_transform
    # 如果是 ReshapeTransform，则返回符合其输入形状的随机数据
    if isinstance(transform, ReshapeTransform):
        return torch.randn(transform.in_shape)
    # 如果逆变换是 ReshapeTransform，则返回符合其输出形状的随机数据
    if isinstance(transform.inv, ReshapeTransform):
        return torch.randn(transform.inv.out_shape)
    # 处理 domain 和 codomain 约束，生成相应的测试数据
    domain = transform.domain
    while (
        isinstance(domain, constraints.independent)
        and domain is not constraints.real_vector
    ):
        domain = domain.base_constraint
    codomain = transform.codomain
    x = torch.empty(4, 5)
    positive_definite_constraints = [
        constraints.lower_cholesky,
        constraints.positive_definite,
    ]
    # 根据 domain 和 codomain 约束生成相应的数据
    if domain in positive_definite_constraints:
        x = torch.randn(6, 6)
        x = x.tril(-1) + x.diag().exp().diag_embed()
        if domain is constraints.positive_definite:
            return x @ x.T
        return x
    elif codomain in positive_definite_constraints:
        return torch.randn(6, 6)
    elif domain is constraints.real:
        return x.normal_()
    elif domain is constraints.real_vector:
        # 对于 corr_cholesky，向量的最后一维大小必须是 (dim * dim) // 2
        x = torch.empty(3, 6)
        x = x.normal_()
        return x
    elif domain is constraints.positive:
        return x.normal_().exp()
    elif domain is constraints.unit_interval:
        return x.uniform_()
    # 如果 domain 是 constraints.interval 类型
    elif isinstance(domain, constraints.interval):
        # 生成一个均匀分布的张量 x
        x = x.uniform_()
        # 对 x 进行线性变换，将其映射到 domain 的范围内
        x = x.mul_(domain.upper_bound - domain.lower_bound).add_(domain.lower_bound)
        # 返回映射后的张量 x
        return x
    
    # 如果 domain 是 constraints.simplex 类型
    elif domain is constraints.simplex:
        # 生成一个正态分布的张量 x，并对其进行指数运算
        x = x.normal_().exp()
        # 对 x 进行归一化，使其成为一个简单形式的张量（simplex）
        x /= x.sum(-1, True)
        # 返回归一化后的张量 x
        return x
    
    # 如果 domain 是 constraints.corr_cholesky 类型
    elif domain is constraints.corr_cholesky:
        # 创建一个形状为 (4, 5, 5) 的空张量 x
        x = torch.empty(4, 5, 5)
        # 对 x 进行正态分布填充，并将其下三角部分提取出来
        x = x.normal_().tril()
        # 对 x 进行归一化，使其每行向量的范数为 1
        x /= x.norm(dim=-1, keepdim=True)
        # 将 x 的对角线元素取绝对值
        x.diagonal(dim1=-1).copy_(x.diagonal(dim1=-1).abs())
        # 返回处理后的张量 x
        return x
    
    # 如果 domain 不是支持的任何类型，则抛出 ValueError 异常
    raise ValueError(f"Unsupported domain: {domain}")
# 激活状态下的转换列表，包括一个缓存大小为1的转换
TRANSFORMS_CACHE_ACTIVE = get_transforms(cache_size=1)

# 非激活状态下的转换列表，缓存大小为0
TRANSFORMS_CACHE_INACTIVE = get_transforms(cache_size=0)

# 所有转换的集合，包括激活和非激活状态的转换，以及一个身份转换
ALL_TRANSFORMS = (
    TRANSFORMS_CACHE_ACTIVE + TRANSFORMS_CACHE_INACTIVE + [identity_transform]
)


@pytest.mark.parametrize("transform", ALL_TRANSFORMS, ids=transform_id)
# 测试逆变换的逆运算是否等于原变换
def test_inv_inv(transform, ids=transform_id):
    assert transform.inv.inv is transform


@pytest.mark.parametrize("x", TRANSFORMS_CACHE_INACTIVE, ids=transform_id)
@pytest.mark.parametrize("y", TRANSFORMS_CACHE_INACTIVE, ids=transform_id)
# 测试转换的相等性，以及身份转换的自反性
def test_equality(x, y):
    if x is y:
        assert x == y
    else:
        assert x != y
    assert identity_transform == identity_transform.inv


@pytest.mark.parametrize("transform", ALL_TRANSFORMS, ids=transform_id)
# 测试带缓存的转换
def test_with_cache(transform):
    if transform._cache_size == 0:
        transform = transform.with_cache(1)
    assert transform._cache_size == 1
    x = generate_data(transform).requires_grad_()
    try:
        y = transform(x)
    except NotImplementedError:
        pytest.skip("Not implemented.")
    y2 = transform(x)
    assert y2 is y


@pytest.mark.parametrize("transform", ALL_TRANSFORMS, ids=transform_id)
@pytest.mark.parametrize("test_cached", [True, False])
# 测试转换的正向和逆向操作
def test_forward_inverse(transform, test_cached):
    x = generate_data(transform).requires_grad_()
    assert transform.domain.check(x).all()  # 验证输入数据是否有效
    try:
        y = transform(x)
    except NotImplementedError:
        pytest.skip("Not implemented.")
    assert y.shape == transform.forward_shape(x.shape)
    if test_cached:
        x2 = transform.inv(y)  # 至少应该通过缓存实现
    else:
        try:
            x2 = transform.inv(y.clone())  # 绕过缓存
        except NotImplementedError:
            pytest.skip("Not implemented.")
    assert x2.shape == transform.inverse_shape(y.shape)
    y2 = transform(x2)
    if transform.bijective:
        # 验证函数的逆操作
        assert torch.allclose(x2, x, atol=1e-4, equal_nan=True), "\n".join(
            [
                f"{transform} t.inv(t(-)) error",
                f"x = {x}",
                f"y = t(x) = {y}",
                f"x2 = t.inv(y) = {x2}",
            ]
        )
    else:
        # 验证较弱的伪逆操作
        assert torch.allclose(y2, y, atol=1e-4, equal_nan=True), "\n".join(
            [
                f"{transform} t(t.inv(t(-))) error",
                f"x = {x}",
                f"y = t(x) = {y}",
                f"x2 = t.inv(y) = {x2}",
                f"y2 = t(x2) = {y2}",
            ]
        )


# 测试组合转换的形状
def test_compose_transform_shapes():
    transform0 = ExpTransform()
    transform1 = SoftmaxTransform()
    transform2 = LowerCholeskyTransform()

    assert transform0.event_dim == 0
    assert transform1.event_dim == 1
    assert transform2.event_dim == 2
    assert ComposeTransform([transform0, transform1]).event_dim == 1
    # 断言：使用 ComposeTransform 类结合 transform0 和 transform2 创建的组合转换对象的事件维度应为 2
    assert ComposeTransform([transform0, transform2]).event_dim == 2
    
    # 断言：使用 ComposeTransform 类结合 transform1 和 transform2 创建的组合转换对象的事件维度应为 2
    assert ComposeTransform([transform1, transform2]).event_dim == 2
# 创建三个不同的变换对象，用于概率分布的转换操作
transform0 = ExpTransform()
transform1 = SoftmaxTransform()
transform2 = LowerCholeskyTransform()

# 创建三种不同的基础分布对象，分别是正态分布和狄利克雷分布的实例
base_dist0 = Normal(torch.zeros(4, 4), torch.ones(4, 4))
base_dist1 = Dirichlet(torch.ones(4, 4))
base_dist2 = Normal(torch.zeros(3, 4, 4), torch.ones(3, 4, 4))

# 使用 pytest 的 parametrize 装饰器来定义多组参数化测试用例
@pytest.mark.parametrize(
    ("batch_shape", "event_shape", "dist"),
    [
        # 第一组测试用例：基础分布是 base_dist0，没有变换
        ((4, 4), (), base_dist0),
        # 第二组测试用例：基础分布是 base_dist1，没有变换
        ((4,), (4,), base_dist1),
        # 后续的测试用例均为 TransformedDistribution，使用不同的变换方式
        ((4, 4), (), TransformedDistribution(base_dist0, [transform0])),
        ((4,), (4,), TransformedDistribution(base_dist0, [transform1])),
        ((4,), (4,), TransformedDistribution(base_dist0, [transform0, transform1])),
        ((), (4, 4), TransformedDistribution(base_dist0, [transform0, transform2])),
        ((4,), (4,), TransformedDistribution(base_dist0, [transform1, transform0])),
        ((), (4, 4), TransformedDistribution(base_dist0, [transform1, transform2])),
        ((), (4, 4), TransformedDistribution(base_dist0, [transform2, transform0])),
        ((), (4, 4), TransformedDistribution(base_dist0, [transform2, transform1])),
        ((4,), (4,), TransformedDistribution(base_dist1, [transform0])),
        ((4,), (4,), TransformedDistribution(base_dist1, [transform1])),
        ((), (4, 4), TransformedDistribution(base_dist1, [transform2])),
        ((4,), (4,), TransformedDistribution(base_dist1, [transform0, transform1])),
        ((), (4, 4), TransformedDistribution(base_dist1, [transform0, transform2])),
        ((4,), (4,), TransformedDistribution(base_dist1, [transform1, transform0])),
        ((), (4, 4), TransformedDistribution(base_dist1, [transform1, transform2])),
        ((), (4, 4), TransformedDistribution(base_dist1, [transform2, transform0])),
        ((), (4, 4), TransformedDistribution(base_dist1, [transform2, transform1])),
        # 最后一组测试用例：基础分布是 base_dist2，使用变换 transform2
        ((3, 4, 4), (), base_dist2),
        ((3,), (4, 4), TransformedDistribution(base_dist2, [transform2])),
        ((3,), (4, 4), TransformedDistribution(base_dist2, [transform0, transform2])),
        ((3,), (4, 4), TransformedDistribution(base_dist2, [transform1, transform2])),
        ((3,), (4, 4), TransformedDistribution(base_dist2, [transform2, transform0])),
        ((3,), (4, 4), TransformedDistribution(base_dist2, [transform2, transform1])),
    ],
)
# 定义测试函数 test_transformed_distribution_shapes，检验 TransformedDistribution 的形状和样本生成
def test_transformed_distribution_shapes(batch_shape, event_shape, dist):
    assert dist.batch_shape == batch_shape
    assert dist.event_shape == event_shape
    x = dist.rsample()
    try:
        dist.log_prob(x)  # 检验 log_prob 方法不会导致崩溃
    except NotImplementedError:
        pytest.skip("Not implemented.")


# 使用 pytest 的 parametrize 装饰器定义测试函数 test_jit_fwd，针对不同的变换执行前向传播测试
@pytest.mark.parametrize("transform", TRANSFORMS_CACHE_INACTIVE, ids=transform_id)
def test_jit_fwd(transform):
    # 生成随机数据，并要求梯度计算
    x = generate_data(transform).requires_grad_()

    # 定义函数 f，对输入 x 执行变换 transform
    def f(x):
        return transform(x)

    try:
        traced_f = torch.jit.trace(f, (x,))
    except NotImplementedError:
        pytest.skip("Not implemented.")

    # 检验不同输入下的前向传播结果
    x = generate_data(transform).requires_grad_()
    # 使用断言检查两个张量 f(x) 和 traced_f(x) 是否在给定的绝对容差范围内全部相等，同时考虑 NaN 值相等
    assert torch.allclose(f(x), traced_f(x), atol=1e-5, equal_nan=True)
@pytest.mark.parametrize("transform", TRANSFORMS_CACHE_INACTIVE, ids=transform_id)
def test_jit_inv(transform):
    # 生成数据 y，并设置其需要梯度计算
    y = generate_data(transform.inv).requires_grad_()

    def f(y):
        # 对 y 应用逆变换
        return transform.inv(y)

    try:
        # 尝试使用 Torch JIT 对函数 f 进行跟踪编译
        traced_f = torch.jit.trace(f, (y,))
    except NotImplementedError:
        # 如果不支持 JIT 编译，则跳过测试
        pytest.skip("Not implemented.")

    # 检查不同输入下的输出是否接近
    y = generate_data(transform.inv).requires_grad_()
    assert torch.allclose(f(y), traced_f(y), atol=1e-5, equal_nan=True)


@pytest.mark.parametrize("transform", TRANSFORMS_CACHE_INACTIVE, ids=transform_id)
def test_jit_jacobian(transform):
    # 生成数据 x，并设置其需要梯度计算
    x = generate_data(transform).requires_grad_()

    def f(x):
        # 对 x 应用变换，计算其对数绝对行列式的 Jacobian 矩阵
        y = transform(x)
        return transform.log_abs_det_jacobian(x, y)

    try:
        # 尝试使用 Torch JIT 对函数 f 进行跟踪编译
        traced_f = torch.jit.trace(f, (x,))
    except NotImplementedError:
        # 如果不支持 JIT 编译，则跳过测试
        pytest.skip("Not implemented.")

    # 检查不同输入下的输出是否接近
    x = generate_data(transform).requires_grad_()
    assert torch.allclose(f(x), traced_f(x), atol=1e-5, equal_nan=True)


@pytest.mark.parametrize("transform", ALL_TRANSFORMS, ids=transform_id)
def test_jacobian(transform):
    # 生成数据 x
    x = generate_data(transform)
    try:
        # 对 x 应用变换，并计算变换的对数绝对行列式的 Jacobian 矩阵
        y = transform(x)
        actual = transform.log_abs_det_jacobian(x, y)
    except NotImplementedError:
        # 如果不支持当前变换，则跳过测试
        pytest.skip("Not implemented.")
    
    # 测试形状是否符合预期
    target_shape = x.shape[: x.dim() - transform.domain.event_dim]
    assert actual.shape == target_shape

    # 如果需要，扩展形状
    transform = reshape_transform(transform, x.shape)
    ndims = len(x.shape)
    event_dim = ndims - transform.domain.event_dim
    x_ = x.view((-1,) + x.shape[event_dim:])
    n = x_.shape[0]

    # 重新调整变换，以压缩批次维到单个批次维
    transform = reshape_transform(transform, x_.shape)

    # 1. 单位雅可比矩阵的变换
    if isinstance(transform, ReshapeTransform) or isinstance(
        transform.inv, ReshapeTransform
    ):
        # 预期的结果为零向量
        expected = x.new_zeros(x.shape[x.dim() - transform.domain.event_dim])

    # 2. 雅可比矩阵中无对角元素的变换
    elif transform.domain.event_dim == 0:
        jac = jacobian(transform, x_)
        # 断言对角线上的元素为零
        assert torch.allclose(jac, jac.diagonal().diag_embed())
        expected = jac.diagonal().abs().log().reshape(x.shape)

    # 3. 雅可比矩阵中有非零对角元素的变换
    # （这一部分的代码未完整给出，需要根据具体情况进行补充）
    else:
        if isinstance(transform, CorrCholeskyTransform):
            # 如果变换是 CorrCholeskyTransform 类型，则计算其雅可比矩阵
            jac = jacobian(lambda x: tril_matrix_to_vec(transform(x), diag=-1), x_)
        elif isinstance(transform.inv, CorrCholeskyTransform):
            # 如果变换的逆是 CorrCholeskyTransform 类型，则计算其雅可比矩阵
            jac = jacobian(
                lambda x: transform(vec_to_tril_matrix(x, diag=-1)),
                tril_matrix_to_vec(x_, diag=-1),
            )
        elif isinstance(transform, StickBreakingTransform):
            # 如果变换是 StickBreakingTransform 类型，则计算其雅可比矩阵
            jac = jacobian(lambda x: transform(x)[..., :-1], x_)
        else:
            # 否则，直接计算变换的雅可比矩阵
            jac = jacobian(transform, x_)

        # 注意：jacobian 的形状为 (batch_dims, y_event_dims, batch_dims, x_event_dims)
        # 但是每个批次是独立的，所以可以将其转换为形状为 (batch_dims, event_dims, event_dims) 的批次方阵，
        # 然后计算其行列式。
        gather_idx_shape = list(jac.shape)
        gather_idx_shape[-2] = 1
        # 创建用于 gather 操作的索引
        gather_idxs = (
            torch.arange(n)
            .reshape((n,) + (1,) * (len(jac.shape) - 1))
            .expand(gather_idx_shape)
        )
        # 执行 gather 操作并去除多余的维度
        jac = jac.gather(-2, gather_idxs).squeeze(-2)
        out_ndims = jac.shape[-2]
        # 去除多余的零值维度（针对逆 stick-breaking 变换）
        jac = jac[..., :out_ndims]
        # 计算雅可比矩阵的行列式的对数绝对值作为期望值
        expected = torch.slogdet(jac).logabsdet

    # 断言实际值与期望值在指定的容差内相等
    assert torch.allclose(actual, expected, atol=1e-5)
# 使用 pytest 模块的 parametrize 装饰器，定义了多个参数化测试用例，每个测试用例都会生成一个单独的测试实例
@pytest.mark.parametrize(
    "event_dims", [(0,), (1,), (2, 3), (0, 1, 2), (1, 2, 0), (2, 0, 1)], ids=str
)
# 定义测试函数 test_compose_affine，用于测试 AffineTransform 和 ComposeTransform 的功能
def test_compose_affine(event_dims):
    # 根据给定的 event_dims 列表创建 AffineTransform 实例列表
    transforms = [
        AffineTransform(torch.zeros((1,) * e), 1, event_dim=e) for e in event_dims
    ]
    # 将 AffineTransform 实例列表传递给 ComposeTransform 构造函数，创建组合变换对象
    transform = ComposeTransform(transforms)
    # 断言组合变换对象的 codomain（目标域）的事件维度与 event_dims 中的最大值相等
    assert transform.codomain.event_dim == max(event_dims)
    # 断言组合变换对象的 domain（源域）的事件维度与 event_dims 中的最大值相等
    assert transform.domain.event_dim == max(event_dims)

    # 创建一个标准正态分布对象 base_dist
    base_dist = Normal(0, 1)
    # 如果 transform 的 domain 的事件维度大于 0，则将 base_dist 扩展到与之匹配的维度
    if transform.domain.event_dim:
        base_dist = base_dist.expand((1,) * transform.domain.event_dim)
    # 使用 TransformedDistribution 类创建一个新的分布对象 dist，应用 transform 中的变换
    dist = TransformedDistribution(base_dist, transform.parts)
    # 断言 dist 的支持域的事件维度与 event_dims 中的最大值相等
    assert dist.support.event_dim == max(event_dims)

    # 创建一个五维全一向量的 Dirichlet 分布对象 base_dist
    base_dist = Dirichlet(torch.ones(5))
    # 如果 transform 的 domain 的事件维度大于 1，则将 base_dist 扩展到相应维度
    if transform.domain.event_dim > 1:
        base_dist = base_dist.expand((1,) * (transform.domain.event_dim - 1))
    # 使用 TransformedDistribution 类再次创建一个新的分布对象 dist，应用 transforms 中的变换
    dist = TransformedDistribution(base_dist, transforms)
    # 断言 dist 的支持域的事件维度为 1 或者 event_dims 中的最大值
    assert dist.support.event_dim == max(1, *event_dims)


# 使用 pytest 模块的 parametrize 装饰器，定义了多个参数化测试用例，每个测试用例都会生成一个单独的测试实例
@pytest.mark.parametrize("batch_shape", [(), (6,), (5, 4)], ids=str)
# 定义测试函数 test_compose_reshape，用于测试 ReshapeTransform 和 ComposeTransform 的功能
def test_compose_reshape(batch_shape):
    # 根据不同的 batch_shape 创建 ReshapeTransform 实例列表 transforms
    transforms = [
        ReshapeTransform((), ()),
        ReshapeTransform((2,), (1, 2)),
        ReshapeTransform((3, 1, 2), (6,)),
        ReshapeTransform((6,), (2, 3)),
    ]
    # 将 ReshapeTransform 实例列表传递给 ComposeTransform 构造函数，创建组合变换对象
    transform = ComposeTransform(transforms)
    # 断言组合变换对象的 codomain（目标域）的事件维度为 2
    assert transform.codomain.event_dim == 2
    # 断言组合变换对象的 domain（源域）的事件维度为 2
    assert transform.domain.event_dim == 2
    # 创建一个形状为 batch_shape + (3, 2) 的随机数据张量 data
    data = torch.randn(batch_shape + (3, 2))
    # 断言对数据张量应用 transform 后的形状为 batch_shape + (2, 3)
    assert transform(data).shape == batch_shape + (2, 3)

    # 使用 TransformedDistribution 类创建一个新的分布对象 dist，基于 Normal 分布和 transforms 中的变换
    dist = TransformedDistribution(Normal(data, 1), transforms)
    # 断言 dist 的 batch_shape 与给定的 batch_shape 相等
    assert dist.batch_shape == batch_shape
    # 断言 dist 的 event_shape 为 (2, 3)
    assert dist.event_shape == (2, 3)
    # 断言 dist 的支持域的事件维度为 2
    assert dist.support.event_dim == 2


# 使用 pytest 模块的 parametrize 装饰器，定义多个参数化测试用例，每个测试用例都会生成一个单独的测试实例
@pytest.mark.parametrize("sample_shape", [(), (7,)], ids=str)
@pytest.mark.parametrize("transform_dim", [0, 1, 2])
@pytest.mark.parametrize("base_batch_dim", [0, 1, 2])
@pytest.mark.parametrize("base_event_dim", [0, 1, 2])
@pytest.mark.parametrize("num_transforms", [0, 1, 2, 3])
# 定义测试函数 test_transformed_distribution，用于测试 TransformedDistribution 的功能
def test_transformed_distribution(
    base_batch_dim, base_event_dim, transform_dim, num_transforms, sample_shape
):
    # 定义一个形状为 torch.Size([2, 3, 4, 5]) 的张量 shape
    shape = torch.Size([2, 3, 4, 5])
    # 创建一个标准正态分布对象 base_dist，根据 base_batch_dim 和 base_event_dim 扩展其形状
    base_dist = Normal(0, 1)
    base_dist = base_dist.expand(shape[4 - base_batch_dim - base_event_dim :])
    # 如果 base_event_dim 不为 0，则将 base_dist 包装成 Independent 分布
    if base_event_dim:
        base_dist = Independent(base_dist, base_event_dim)
    # 根据不同的 num_transforms 创建 AffineTransform 和 ReshapeTransform 实例列表 transforms
    transforms = [
        AffineTransform(torch.zeros(shape[4 - transform_dim :]), 1),
        ReshapeTransform((4, 5), (20,)),
        ReshapeTransform((3, 20), (6, 10)),
    ]
    transforms = transforms[:num_transforms]
    # 将 transforms 实例列表传递给 ComposeTransform 构造函数，创建组合变换对象 transform
    transform = ComposeTransform(transforms)

    # 在初始化 TransformedDistribution 时，验证 base_batch_dim 和 base_event_dim 是否符合要求
    if base_batch_dim + base_event_dim < transform.domain.event_dim:
        # 如果不符合要求，预期会引发 ValueError 异常
        with pytest.raises(ValueError):
            TransformedDistribution(base_dist, transforms)
        return
    # 创建一个新的 TransformedDistribution 分布对象 d，应用 transforms 中的变换
    d = TransformedDistribution(base_dist, transforms)

    # 检查从分布对象 d 中采样是否扩展得足够充分
    x = d.sample(sample_shape)
    # 断言采样结果的形状应为 sample_shape + d.batch_shape + d.event_shape
    assert x.shape == sample_shape + d.batch_shape + d.event_shape
    # 计算 x 中唯一元素的数量
    num_unique = len(set(x.reshape(-1).tolist()))
    # 断言确保唯一元素数量不少于 x 元素总数的 90%
    assert num_unique >= 0.9 * x.numel()

    # 检查在完整样本上的 log_prob 的形状。
    log_prob = d.log_prob(x)
    # 断言确保 log_prob 的形状与 sample_shape 和 d.batch_shape 相匹配
    assert log_prob.shape == sample_shape + d.batch_shape

    # 检查在部分样本上的 log_prob 的形状。
    y = x
    # 当 y 的维度大于 d.event_shape 的长度时，取其第一个元素
    while y.dim() > len(d.event_shape):
        y = y[0]
    log_prob = d.log_prob(y)
    # 断言确保 log_prob 的形状与 d.batch_shape 相匹配
    assert log_prob.shape == d.batch_shape
# 定义一个测试函数，用于测试保存、加载和转换操作的正确性
def test_save_load_transform():
    # 创建一个以正态分布为基础的变换分布对象，添加仿射变换作为转换器
    dist = TransformedDistribution(Normal(0, 1), [AffineTransform(2, 3)])
    # 生成一个张量，包含从0到1的10个均匀间隔数值
    x = torch.linspace(0, 1, 10)
    # 计算变换分布对象在给定输入x下的对数概率密度
    log_prob = dist.log_prob(x)
    # 创建一个字节流对象
    stream = io.BytesIO()
    # 将变换分布对象dist保存到字节流中
    torch.save(dist, stream)
    # 将流的位置设置回起始位置
    stream.seek(0)
    # 从字节流中加载保存的对象到other
    other = torch.load(stream)
    # 断言两个变换分布对象在相同输入x下的对数概率密度近似相等
    assert torch.allclose(log_prob, other.log_prob(x))


# 使用参数化测试装饰器，遍历所有的变换器，并为每个变换器指定一个ID
@pytest.mark.parametrize("transform", ALL_TRANSFORMS, ids=transform_id)
def test_transform_sign(transform: Transform):
    try:
        # 尝试获取当前变换器的符号属性
        sign = transform.sign
    except NotImplementedError:
        # 如果获取符号属性抛出未实现错误，则跳过当前测试
        pytest.skip("Not implemented.")

    # 生成一个由当前变换器生成数据的张量，并要求其梯度信息
    x = generate_data(transform).requires_grad_()
    # 对变换器应用到数据张量x上，然后计算结果的和
    y = transform(x).sum()
    # 计算y关于x的梯度
    (derivatives,) = grad(y, [x])
    # 断言所有计算得到的梯度乘以变换器的符号都大于零
    assert torch.less(torch.as_tensor(0.0), derivatives * sign).all()


# 如果当前脚本作为主程序运行，则执行测试函数
if __name__ == "__main__":
    run_tests()
```