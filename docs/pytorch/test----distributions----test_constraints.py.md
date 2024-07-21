# `.\pytorch\test\distributions\test_constraints.py`

```
# Owner(s): ["module: distributions"]

# 导入 pytest 模块，用于测试
import pytest

# 导入 torch 模块
import torch

# 从 torch.distributions 模块中导入 biject_to, constraints, transform_to
from torch.distributions import biject_to, constraints, transform_to

# 从 torch.testing._internal.common_cuda 导入 TEST_CUDA
from torch.testing._internal.common_cuda import TEST_CUDA

# 从 torch.testing._internal.common_utils 导入 run_tests
from torch.testing._internal.common_utils import run_tests

# EXAMPLES 列表包含了多个元组，每个元组包含一个约束函数、一个布尔值和一个数值列表或嵌套列表
EXAMPLES = [
    (constraints.symmetric, False, [[2.0, 0], [2.0, 2]]),
    (constraints.positive_semidefinite, False, [[2.0, 0], [2.0, 2]]),
    (constraints.positive_definite, False, [[2.0, 0], [2.0, 2]]),
    (constraints.symmetric, True, [[3.0, -5], [-5.0, 3]]),
    (constraints.positive_semidefinite, False, [[3.0, -5], [-5.0, 3]]),
    (constraints.positive_definite, False, [[3.0, -5], [-5.0, 3]]),
    (constraints.symmetric, True, [[1.0, 2], [2.0, 4]]),
    (constraints.positive_semidefinite, True, [[1.0, 2], [2.0, 4]]),
    (constraints.positive_definite, False, [[1.0, 2], [2.0, 4]]),
    (constraints.symmetric, True, [[[1.0, -2], [-2.0, 1]], [[2.0, 3], [3.0, 2]]]),
    (
        constraints.positive_semidefinite,
        False,
        [[[1.0, -2], [-2.0, 1]], [[2.0, 3], [3.0, 2]]],
    ),
    (
        constraints.positive_definite,
        False,
        [[[1.0, -2], [-2.0, 1]], [[2.0, 3], [3.0, 2]]],
    ),
    (constraints.symmetric, True, [[[1.0, -2], [-2.0, 4]], [[1.0, -1], [-1.0, 1]]]),
    (
        constraints.positive_semidefinite,
        True,
        [[[1.0, -2], [-2.0, 4]], [[1.0, -1], [-1.0, 1]]],
    ),
    (
        constraints.positive_definite,
        False,
        [[[1.0, -2], [-2.0, 4]], [[1.0, -1], [-1.0, 1]]],
    ),
    (constraints.symmetric, True, [[[4.0, 2], [2.0, 4]], [[3.0, -1], [-1.0, 3]]]),
    (
        constraints.positive_semidefinite,
        True,
        [[[4.0, 2], [2.0, 4]], [[3.0, -1], [-1.0, 3]]],
    ),
    (
        constraints.positive_definite,
        True,
        [[[4.0, 2], [2.0, 4]], [[3.0, -1], [-1.0, 3]]],
    ),
]

# CONSTRAINTS 列表包含了多个元组，每个元组包含一个约束函数和一个或多个参数
CONSTRAINTS = [
    (constraints.real,),
    (constraints.real_vector,),
    (constraints.positive,),
    (constraints.greater_than, [-10.0, -2, 0, 2, 10]),
    (constraints.greater_than, 0),
    (constraints.greater_than, 2),
    (constraints.greater_than, -2),
    (constraints.greater_than_eq, 0),
    (constraints.greater_than_eq, 2),
    (constraints.greater_than_eq, -2),
    (constraints.less_than, [-10.0, -2, 0, 2, 10]),
    (constraints.less_than, 0),
    (constraints.less_than, 2),
    (constraints.less_than, -2),
    (constraints.unit_interval,),
    (constraints.interval, [-4.0, -2, 0, 2, 4], [-3.0, 3, 1, 5, 5]),
    (constraints.interval, -2, -1),
    (constraints.interval, 1, 2),
    (constraints.half_open_interval, [-4.0, -2, 0, 2, 4], [-3.0, 3, 1, 5, 5]),
    (constraints.half_open_interval, -2, -1),
    (constraints.half_open_interval, 1, 2),
    (constraints.simplex,),
    (constraints.corr_cholesky,),
    (constraints.lower_cholesky,),
    (constraints.positive_definite,),
]

# build_constraint 函数接受一个约束函数和参数，返回约束函数或者使用参数构建的约束函数
def build_constraint(constraint_fn, args, is_cuda=False):
    # 如果参数为空，直接返回约束函数本身
    if not args:
        return constraint_fn
    # 根据当前是否启用 CUDA 决定使用不同类型的张量，如果启用 CUDA 则使用 cuda.DoubleTensor，否则使用 DoubleTensor
    t = torch.cuda.DoubleTensor if is_cuda else torch.DoubleTensor
    # 调用约束函数 constraint_fn，并根据参数的类型决定是否将其转换为张量类型
    return constraint_fn(*(t(x) if isinstance(x, list) else x for x in args))
# 使用 pytest 的参数化功能，为每个测试用例组合不同的约束函数、期望结果和输入值
@pytest.mark.parametrize(("constraint_fn", "result", "value"), EXAMPLES)
# 使用 pytest 的参数化功能，为每个测试用例组合是否使用 CUDA 的选项
@pytest.mark.parametrize(
    "is_cuda",
    [
        False,
        # 如果 CUDA 可用，标记此测试跳过，理由是 CUDA 未找到
        pytest.param(
            True, marks=pytest.mark.skipif(not TEST_CUDA, reason="CUDA not found.")
        ),
    ],
)
def test_constraint(constraint_fn, result, value, is_cuda):
    # 根据是否使用 CUDA 选择不同类型的 Tensor
    t = torch.cuda.DoubleTensor if is_cuda else torch.DoubleTensor
    # 断言约束函数对给定值的检查结果是否与期望一致
    assert constraint_fn.check(t(value)).all() == result


# 使用 pytest 的参数化功能，为每个测试用例组合不同的约束函数和参数
@pytest.mark.parametrize(
    ("constraint_fn", "args"), [(c[0], c[1:]) for c in CONSTRAINTS]
)
# 使用 pytest 的参数化功能，为每个测试用例组合是否使用 CUDA 的选项
@pytest.mark.parametrize(
    "is_cuda",
    [
        False,
        # 如果 CUDA 可用，标记此测试跳过，理由是 CUDA 未找到
        pytest.param(
            True, marks=pytest.mark.skipif(not TEST_CUDA, reason="CUDA not found.")
        ),
    ],
)
def test_biject_to(constraint_fn, args, is_cuda):
    # 根据约束函数和参数构建约束对象
    constraint = build_constraint(constraint_fn, args, is_cuda=is_cuda)
    try:
        # 尝试根据约束构建双射函数
        t = biject_to(constraint)
    except NotImplementedError:
        # 如果双射函数未实现，则跳过此测试
        pytest.skip("`biject_to` not implemented.")
    # 断言双射函数是否满射
    assert t.bijective, f"biject_to({constraint}) is not bijective"
    # 根据约束类型选择不同大小的输入张量 x
    if constraint_fn is constraints.corr_cholesky:
        x = torch.randn(6, 6, dtype=torch.double)  # 对角线元素数为 6
    else:
        x = torch.randn(5, 5, dtype=torch.double)  # 对角线元素数为 5
    # 如果使用 CUDA，将张量 x 移动到 GPU
    if is_cuda:
        x = x.cuda()
    # 对 x 进行双射操作得到结果 y
    y = t(x)
    # 断言约束函数对结果 y 的检查是否全部通过
    assert constraint.check(y).all(), "\n".join(
        [
            f"Failed to biject_to({constraint})",
            f"x = {x}",
            f"biject_to(...)(x) = {y}",
        ]
    )
    # 根据双射函数计算其逆操作得到 x2
    x2 = t.inv(y)
    # 断言 x 与 x2 在数值上的接近程度
    assert torch.allclose(x, x2), f"Error in biject_to({constraint}) inverse"

    # 计算双射函数的对数绝对行列式雅可比矩阵
    j = t.log_abs_det_jacobian(x, y)
    # 断言雅可比矩阵的形状是否与 x 的形状匹配
    assert j.shape == x.shape[: x.dim() - t.domain.event_dim]


# 使用 pytest 的参数化功能，为每个测试用例组合不同的约束函数和参数
@pytest.mark.parametrize(
    ("constraint_fn", "args"), [(c[0], c[1:]) for c in CONSTRAINTS]
)
# 使用 pytest 的参数化功能，为每个测试用例组合是否使用 CUDA 的选项
@pytest.mark.parametrize(
    "is_cuda",
    [
        False,
        # 如果 CUDA 可用，标记此测试跳过，理由是 CUDA 未找到
        pytest.param(
            True, marks=pytest.mark.skipif(not TEST_CUDA, reason="CUDA not found.")
        ),
    ],
)
def test_transform_to(constraint_fn, args, is_cuda):
    # 根据约束函数和参数构建约束对象
    constraint = build_constraint(constraint_fn, args, is_cuda=is_cuda)
    # 根据约束构建转换函数
    t = transform_to(constraint)
    # 根据约束类型选择不同大小的输入张量 x
    if constraint_fn is constraints.corr_cholesky:
        x = torch.randn(6, 6, dtype=torch.double)  # 对角线元素数为 6
    else:
        x = torch.randn(5, 5, dtype=torch.double)  # 对角线元素数为 5
    # 如果使用 CUDA，将张量 x 移动到 GPU
    if is_cuda:
        x = x.cuda()
    # 对 x 进行约束变换得到结果 y
    y = t(x)
    # 断言约束函数对结果 y 的检查是否全部通过
    assert constraint.check(y).all(), f"Failed to transform_to({constraint})"
    # 根据转换函数计算其逆操作得到 x2
    x2 = t.inv(y)
    # 使用转换函数对 x2 进行再次变换得到 y2
    y2 = t(x2)
    # 断言 y 与 y2 在数值上的接近程度
    assert torch.allclose(y, y2), f"Error in transform_to({constraint}) pseudoinverse"


# 如果此脚本作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```