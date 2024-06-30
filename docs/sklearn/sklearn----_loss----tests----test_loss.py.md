# `D:\src\scipysrc\scikit-learn\sklearn\_loss\tests\test_loss.py`

```
import pickle  # 导入 pickle 模块，用于序列化和反序列化 Python 对象

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试
from numpy.testing import assert_allclose, assert_array_equal  # 从 NumPy.testing 模块导入断言函数
from pytest import approx  # 从 pytest 模块导入 approx 断言函数
from scipy.optimize import (  # 导入 SciPy 中的优化模块和函数
    LinearConstraint,
    minimize,
    minimize_scalar,
    newton,
)
from scipy.special import logsumexp  # 从 SciPy.special 模块导入 logsumexp 函数

from sklearn._loss.link import IdentityLink, _inclusive_low_high  # 导入 sklearn 内部模块
from sklearn._loss.loss import (  # 导入 sklearn 中的损失函数类
    _LOSSES,
    AbsoluteError,
    BaseLoss,
    HalfBinomialLoss,
    HalfGammaLoss,
    HalfMultinomialLoss,
    HalfPoissonLoss,
    HalfSquaredError,
    HalfTweedieLoss,
    HalfTweedieLossIdentity,
    HuberLoss,
    PinballLoss,
)
from sklearn.utils import assert_all_finite  # 从 sklearn.utils 模块导入断言函数
from sklearn.utils._testing import create_memmap_backed_data, skip_if_32bit  # 导入 sklearn 内部测试相关函数
from sklearn.utils.fixes import _IS_WASM  # 导入 sklearn 内部修复相关变量

ALL_LOSSES = list(_LOSSES.values())  # 将 _LOSSES 字典中的值转换为列表并赋值给 ALL_LOSSES

LOSS_INSTANCES = [loss() for loss in ALL_LOSSES]
# 根据 ALL_LOSSES 中的每个损失函数类实例化一个对象，组成列表 LOSS_INSTANCES

# 默认已包含 HalfTweedieLoss(power=1.5)
LOSS_INSTANCES += [
    PinballLoss(quantile=0.25),  # 添加分位数损失 PinballLoss(quantile=0.25) 实例
    HuberLoss(quantile=0.75),    # 添加 HuberLoss(quantile=0.75) 实例
    HalfTweedieLoss(power=-1.5), # 添加 HalfTweedieLoss(power=-1.5) 实例
    HalfTweedieLoss(power=0),    # 添加 HalfTweedieLoss(power=0) 实例
    HalfTweedieLoss(power=1),    # 添加 HalfTweedieLoss(power=1) 实例
    HalfTweedieLoss(power=2),    # 添加 HalfTweedieLoss(power=2) 实例
    HalfTweedieLoss(power=3.0),  # 添加 HalfTweedieLoss(power=3.0) 实例
    HalfTweedieLossIdentity(power=0),   # 添加 HalfTweedieLossIdentity(power=0) 实例
    HalfTweedieLossIdentity(power=1),   # 添加 HalfTweedieLossIdentity(power=1) 实例
    HalfTweedieLossIdentity(power=2),   # 添加 HalfTweedieLossIdentity(power=2) 实例
    HalfTweedieLossIdentity(power=3.0), # 添加 HalfTweedieLossIdentity(power=3.0) 实例
]


def loss_instance_name(param):
    """根据参数返回损失函数实例的名称，包括参数信息。"""
    if isinstance(param, BaseLoss):
        loss = param
        name = loss.__class__.__name__  # 获取损失函数类的名称
        if isinstance(loss, PinballLoss):
            name += f"(quantile={loss.closs.quantile})"  # 对于 PinballLoss 添加其分位数信息
        elif isinstance(loss, HuberLoss):
            name += f"(quantile={loss.quantile}"  # 对于 HuberLoss 添加其分位数信息
        elif hasattr(loss, "closs") and hasattr(loss.closs, "power"):
            name += f"(power={loss.closs.power})"  # 对于其他损失函数添加其特定参数信息
        return name
    else:
        return str(param)  # 如果参数不是损失函数实例，则返回其字符串表示


def random_y_true_raw_prediction(
    loss, n_samples, y_bound=(-100, 100), raw_bound=(-5, 5), seed=42
):
    """随机生成在有效范围内的 y_true 和 raw_prediction。"""
    rng = np.random.RandomState(seed)  # 创建随机数生成器对象
    if loss.is_multiclass:
        raw_prediction = np.empty((n_samples, loss.n_classes))  # 创建空的原始预测数组
        raw_prediction.flat[:] = rng.uniform(
            low=raw_bound[0],
            high=raw_bound[1],
            size=n_samples * loss.n_classes,
        )  # 使用均匀分布填充原始预测数组
        y_true = np.arange(n_samples).astype(float) % loss.n_classes  # 生成有效的 y_true

        )
        y_true = np.arange(n_samples).astype(float) % loss.n_classes  # 生成有效的 y_true，确保在浮点数和类数之间循环
    else:
        # 如果链接函数是IdentityLink，我们必须尊重y_pred的区间：
        if isinstance(loss.link, IdentityLink):
            # 获取y_pred的包含区间的上下界
            low, high = _inclusive_low_high(loss.interval_y_pred)
            # 确保上下界符合raw_bound的限制
            low = np.amax([low, raw_bound[0]])
            high = np.amin([high, raw_bound[1]])
            raw_bound = (low, high)
        # 在指定的raw_bound范围内生成原始预测值
        raw_prediction = rng.uniform(
            low=raw_bound[0], high=raw_bound[1], size=n_samples
        )
        # 生成一个在有效范围内的y_true
        # 获取y_true的包含区间的上下界
        low, high = _inclusive_low_high(loss.interval_y_true)
        # 确保上下界符合y_bound的限制
        low = max(low, y_bound[0])
        high = min(high, y_bound[1])
        # 在指定的y_bound范围内生成y_true
        y_true = rng.uniform(low, high, size=n_samples)
        # 在特殊边界设置一些值
        # 如果y_true的下界是0且包含在区间内，将每隔三个样本设置为0
        if loss.interval_y_true.low == 0 and loss.interval_y_true.low_inclusive:
            y_true[:: (n_samples // 3)] = 0
        # 如果y_true的上界是1且包含在区间内，将每隔三个样本的第二个设置为1
        if loss.interval_y_true.high == 1 and loss.interval_y_true.high_inclusive:
            y_true[1 :: (n_samples // 3)] = 1

    # 返回生成的y_true和raw_prediction
    return y_true, raw_prediction
# 定义一个函数用于计算数值导数（一阶导数）的辅助函数
def numerical_derivative(func, x, eps):
    """Helper function for numerical (first) derivatives."""
    # 对于数值导数，请参考以下链接：
    # https://en.wikipedia.org/wiki/Numerical_differentiation
    # https://en.wikipedia.org/wiki/Finite_difference_coefficient
    # 我们使用精度为4的中心有限差分法

    # 创建一个与 x 相同形状的数组，每个元素值均为 eps
    h = np.full_like(x, fill_value=eps)
    # 计算 func(x - 2*h)、func(x - h)、func(x + h)、func(x + 2*h)
    f_minus_2h = func(x - 2 * h)
    f_minus_1h = func(x - h)
    f_plus_1h = func(x + h)
    f_plus_2h = func(x + 2 * h)
    # 使用中心有限差分公式计算数值导数
    return (-f_plus_2h + 8 * f_plus_1h - 8 * f_minus_1h + f_minus_2h) / (12.0 * eps)


@pytest.mark.parametrize("loss", LOSS_INSTANCES, ids=loss_instance_name)
def test_loss_boundary(loss):
    """Test interval ranges of y_true and y_pred in losses."""
    # 确保在 loss.is_multiclass 为真时，n_classes 设置为默认值 3
    if loss.is_multiclass:
        n_classes = 3  # 默认值
        # 创建长度为 3*n_classes 的一维数组，用于测试 y_true 的范围
        y_true = np.tile(np.linspace(0, n_classes - 1, num=n_classes), 3)
    else:
        # 获取 y_true 的上下界
        low, high = _inclusive_low_high(loss.interval_y_true)
        # 创建长度为 10 的数组，用于测试 y_true 的范围
        y_true = np.linspace(low, high, num=10)

    # 如果 loss.interval_y_true.low_inclusive 为真，则将 low 加入 y_true
    if loss.interval_y_true.low_inclusive:
        y_true = np.r_[y_true, loss.interval_y_true.low]
    # 如果 loss.interval_y_true.high_inclusive 为真，则将 high 加入 y_true
    if loss.interval_y_true.high_inclusive:
        y_true = np.r_[y_true, loss.interval_y_true.high]

    # 断言 loss.in_y_true_range(y_true) 为真
    assert loss.in_y_true_range(y_true)

    # 获取 y_pred 的上下界
    n = y_true.shape[0]
    low, high = _inclusive_low_high(loss.interval_y_pred)
    # 根据 loss.is_multiclass 的值选择不同的创建 y_pred 的方式
    if loss.is_multiclass:
        y_pred = np.empty((n, n_classes))
        y_pred[:, 0] = np.linspace(low, high, num=n)
        y_pred[:, 1] = 0.5 * (1 - y_pred[:, 0])
        y_pred[:, 2] = 0.5 * (1 - y_pred[:, 0])
    else:
        y_pred = np.linspace(low, high, num=n)

    # 断言 loss.in_y_pred_range(y_pred) 为真
    assert loss.in_y_pred_range(y_pred)

    # 计算损失值不应该失败
    raw_prediction = loss.link.link(y_pred)
    loss.loss(y_true=y_true, raw_prediction=raw_prediction)


# 用于测试有效值范围的装置。
Y_COMMON_PARAMS = [
    # (loss, [y success], [y fail])
    (HalfSquaredError(), [-100, 0, 0.1, 100], [-np.inf, np.inf]),
    (AbsoluteError(), [-100, 0, 0.1, 100], [-np.inf, np.inf]),
    (PinballLoss(), [-100, 0, 0.1, 100], [-np.inf, np.inf]),
    (HuberLoss(), [-100, 0, 0.1, 100], [-np.inf, np.inf]),
    (HalfPoissonLoss(), [0.1, 100], [-np.inf, -3, -0.1, np.inf]),
    (HalfGammaLoss(), [0.1, 100], [-np.inf, -3, -0.1, 0, np.inf]),
    (HalfTweedieLoss(power=-3), [0.1, 100], [-np.inf, np.inf]),
    (HalfTweedieLoss(power=0), [0.1, 100], [-np.inf, np.inf]),
    (HalfTweedieLoss(power=1.5), [0.1, 100], [-np.inf, -3, -0.1, np.inf]),
    (HalfTweedieLoss(power=2), [0.1, 100], [-np.inf, -3, -0.1, 0, np.inf]),
    (HalfTweedieLoss(power=3), [0.1, 100], [-np.inf, -3, -0.1, 0, np.inf]),
    (HalfTweedieLossIdentity(power=-3), [0.1, 100], [-np.inf, np.inf]),
    (HalfTweedieLossIdentity(power=0), [-3, -0.1, 0, 0.1, 100], [-np.inf, np.inf]),
    (HalfTweedieLossIdentity(power=1.5), [0.1, 100], [-np.inf, -3, -0.1, np.inf]),
    # 创建一个包含不同损失函数及其参数范围的元组列表
    (HalfTweedieLossIdentity(power=2), [0.1, 100], [-np.inf, -3, -0.1, 0, np.inf]),
    # 包含 Tweedie 损失函数（指数为2），参数范围为[0.1, 100]和[-∞, -3, -0.1, 0, +∞]

    (HalfTweedieLossIdentity(power=3), [0.1, 100], [-np.inf, -3, -0.1, 0, np.inf]),
    # 包含 Tweedie 损失函数（指数为3），参数范围为[0.1, 100]和[-∞, -3, -0.1, 0, +∞]

    (HalfBinomialLoss(), [0.1, 0.5, 0.9], [-np.inf, -1, 2, np.inf]),
    # 包含二项式损失函数，参数范围为[0.1, 0.5, 0.9]和[-∞, -1, 2, +∞]

    (HalfMultinomialLoss(), [], [-np.inf, -1, 1.1, np.inf]),
    # 包含多项式损失函数，参数范围为空列表[]和[-∞, -1, 1.1, +∞]
# y_pred and y_true do not always have the same domain (valid value range).
# Hence, we define extra sets of parameters for each of them.
Y_TRUE_PARAMS = [  # type: ignore
    # (loss, [y success], [y fail])
    (HalfPoissonLoss(), [0], []),  # HalfPoissonLoss with y_true=0 considered a success
    (HuberLoss(), [0], []),  # HuberLoss with y_true=0 considered a success
    (HalfTweedieLoss(power=-3), [-100, -0.1, 0], []),  # HalfTweedieLoss with specific y_true values
    (HalfTweedieLoss(power=0), [-100, 0], []),  # HalfTweedieLoss with specific y_true values
    (HalfTweedieLoss(power=1.5), [0], []),  # HalfTweedieLoss with y_true=0 considered a success
    (HalfTweedieLossIdentity(power=-3), [-100, -0.1, 0], []),  # HalfTweedieLossIdentity with specific y_true values
    (HalfTweedieLossIdentity(power=0), [-100, 0], []),  # HalfTweedieLossIdentity with specific y_true values
    (HalfTweedieLossIdentity(power=1.5), [0], []),  # HalfTweedieLossIdentity with y_true=0 considered a success
    (HalfBinomialLoss(), [0, 1], []),  # HalfBinomialLoss with y_true=0 or 1 considered a success
    (HalfMultinomialLoss(), [0.0, 1.0, 2], []),  # HalfMultinomialLoss with specific y_true values
]
Y_PRED_PARAMS = [
    # (loss, [y success], [y fail])
    (HalfPoissonLoss(), [], [0]),  # HalfPoissonLoss with y_pred=0 considered a failure
    (HalfTweedieLoss(power=-3), [], [-3, -0.1, 0]),  # HalfTweedieLoss with specific y_pred values
    (HalfTweedieLoss(power=0), [], [-3, -0.1, 0]),  # HalfTweedieLoss with specific y_pred values
    (HalfTweedieLoss(power=1.5), [], [0]),  # HalfTweedieLoss with y_pred=0 considered a failure
    (HalfTweedieLossIdentity(power=-3), [], [-3, -0.1, 0]),  # HalfTweedieLossIdentity with specific y_pred values
    (HalfTweedieLossIdentity(power=0), [-3, -0.1, 0], []),  # HalfTweedieLossIdentity with specific y_pred values
    (HalfTweedieLossIdentity(power=1.5), [], [0]),  # HalfTweedieLossIdentity with y_pred=0 considered a failure
    (HalfBinomialLoss(), [], [0, 1]),  # HalfBinomialLoss with y_pred=0 or 1 considered a failure
    (HalfMultinomialLoss(), [0.1, 0.5], [0, 1]),  # HalfMultinomialLoss with specific y_pred values
]


@pytest.mark.parametrize(
    "loss, y_true_success, y_true_fail", Y_COMMON_PARAMS + Y_TRUE_PARAMS
)
def test_loss_boundary_y_true(loss, y_true_success, y_true_fail):
    """Test boundaries of y_true for loss functions."""
    for y in y_true_success:
        assert loss.in_y_true_range(np.array([y]))  # Check if y_true is within the valid range for the loss
    for y in y_true_fail:
        assert not loss.in_y_true_range(np.array([y]))  # Check if y_true is outside the valid range for the loss


@pytest.mark.parametrize(
    "loss, y_pred_success, y_pred_fail", Y_COMMON_PARAMS + Y_PRED_PARAMS  # type: ignore
)
def test_loss_boundary_y_pred(loss, y_pred_success, y_pred_fail):
    """Test boundaries of y_pred for loss functions."""
    for y in y_pred_success:
        assert loss.in_y_pred_range(np.array([y]))  # Check if y_pred is within the valid range for the loss
    for y in y_pred_fail:
        assert not loss.in_y_pred_range(np.array([y]))  # Check if y_pred is outside the valid range for the loss


@pytest.mark.parametrize(
    "loss, y_true, raw_prediction, loss_true, gradient_true, hessian_true",
    Y_COMMON_PARAMS + Y_TRUE_PARAMS + Y_PRED_PARAMS,  # Combine all sets of parameters
    ids=loss_instance_name,  # Use loss_instance_name as ids for the tests
)
def test_loss_on_specific_values(
    loss, y_true, raw_prediction, loss_true, gradient_true, hessian_true
):
    """Test losses, gradients and hessians at specific values."""
    loss1 = loss(y_true=np.array([y_true]), raw_prediction=np.array([raw_prediction]))
    grad1 = loss.gradient(
        y_true=np.array([y_true]), raw_prediction=np.array([raw_prediction])
    )
    loss2, grad2 = loss.loss_gradient(
        y_true=np.array([y_true]), raw_prediction=np.array([raw_prediction])
    )
    grad3, hess = loss.gradient_hessian(
        y_true=np.array([y_true]), raw_prediction=np.array([raw_prediction])
    )

    assert loss1 == approx(loss_true, rel=1e-15, abs=1e-15)  # Check loss value approximation
    assert loss2 == approx(loss_true, rel=1e-15, abs=1e-15)  # Check loss value approximation
    # 如果给定的 gradient_true 不为 None，则进行以下断言检查
    if gradient_true is not None:
        # 检查 grad1 是否近似等于 gradient_true，相对误差和绝对误差均为 1e-15
        assert grad1 == approx(gradient_true, rel=1e-15, abs=1e-15)
        # 检查 grad2 是否近似等于 gradient_true，相对误差和绝对误差均为 1e-15
        assert grad2 == approx(gradient_true, rel=1e-15, abs=1e-15)
        # 检查 grad3 是否近似等于 gradient_true，相对误差和绝对误差均为 1e-15
        assert grad3 == approx(gradient_true, rel=1e-15, abs=1e-15)

    # 如果给定的 hessian_true 不为 None，则进行以下断言检查
    if hessian_true is not None:
        # 检查 hess 是否近似等于 hessian_true，相对误差和绝对误差均为 1e-15
        assert hess == approx(hessian_true, rel=1e-15, abs=1e-15)
@pytest.mark.parametrize("loss", ALL_LOSSES)
@pytest.mark.parametrize("readonly_memmap", [False, True])
@pytest.mark.parametrize("dtype_in", [np.float32, np.float64])
@pytest.mark.parametrize("dtype_out", [np.float32, np.float64])
@pytest.mark.parametrize("sample_weight", [None, 1])
@pytest.mark.parametrize("out1", [None, 1])
@pytest.mark.parametrize("out2", [None, 1])
@pytest.mark.parametrize("n_threads", [1, 2])
def test_loss_dtype(
    loss, readonly_memmap, dtype_in, dtype_out, sample_weight, out1, out2, n_threads
):
    """Test acceptance of dtypes, readonly and writeable arrays in loss functions.

    Check that loss accepts if all input arrays are either all float32 or all
    float64, and all output arrays are either all float32 or all float64.

    Also check that input arrays can be readonly, e.g. memory mapped.
    """
    if _IS_WASM and readonly_memmap:  # pragma: nocover
        # 如果在 WASM 环境下且只读内存映射，标记为预期失败，原因是内存映射支持不完整
        pytest.xfail(reason="memmap not fully supported")

    # 创建指定类型的损失对象实例
    loss = loss()

    # 生成一个在有效范围内的 y_true 和 raw_prediction
    n_samples = 5
    y_true, raw_prediction = random_y_true_raw_prediction(
        loss=loss,
        n_samples=n_samples,
        y_bound=(-100, 100),
        raw_bound=(-10, 10),
        seed=42,
    )

    # 将生成的数组转换为指定的输入数据类型
    y_true = y_true.astype(dtype_in)
    raw_prediction = raw_prediction.astype(dtype_in)

    # 如果存在样本权重，则创建相应类型的数组
    if sample_weight is not None:
        sample_weight = np.array([2.0] * n_samples, dtype=dtype_in)

    # 如果 out1 不为空，则创建与 y_true 相同形状的空数组
    if out1 is not None:
        out1 = np.empty_like(y_true, dtype=dtype_out)

    # 如果 out2 不为空，则创建与 raw_prediction 相同形状的空数组
    if out2 is not None:
        out2 = np.empty_like(raw_prediction, dtype=dtype_out)

    # 如果是只读内存映射，则将相应数组转换为内存映射数组
    if readonly_memmap:
        y_true = create_memmap_backed_data(y_true)
        raw_prediction = create_memmap_backed_data(raw_prediction)
        if sample_weight is not None:
            sample_weight = create_memmap_backed_data(sample_weight)

    # 调用损失函数对象的 loss 方法，计算损失值
    loss.loss(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        loss_out=out1,
        n_threads=n_threads,
    )

    # 调用损失函数对象的 gradient 方法，计算梯度
    loss.gradient(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        gradient_out=out2,
        n_threads=n_threads,
    )

    # 调用损失函数对象的 loss_gradient 方法，同时计算损失值和梯度
    loss.loss_gradient(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        loss_out=out1,
        gradient_out=out2,
        n_threads=n_threads,
    )

    # 如果 out1 不为空且损失函数是多类别的，则创建与 raw_prediction 相同形状的空数组
    if out1 is not None and loss.is_multiclass:
        out1 = np.empty_like(raw_prediction, dtype=dtype_out)

    # 调用损失函数对象的 gradient_hessian 方法，同时计算梯度和海森矩阵
    loss.gradient_hessian(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        gradient_out=out1,
        hessian_out=out2,
        n_threads=n_threads,
    )

    # 调用损失函数对象，仅计算损失值，不返回梯度
    loss(y_true=y_true, raw_prediction=raw_prediction, sample_weight=sample_weight)

    # 调用损失函数对象的 fit_intercept_only 方法，计算拟合截距项
    loss.fit_intercept_only(y_true=y_true, sample_weight=sample_weight)

    # 调用损失函数对象的 constant_to_optimal_zero 方法，将常数项优化为零
    loss.constant_to_optimal_zero(y_true=y_true, sample_weight=sample_weight)
    # 检查 `loss` 对象是否具有 `predict_proba` 方法，如果有则调用该方法
    if hasattr(loss, "predict_proba"):
        loss.predict_proba(raw_prediction=raw_prediction)
    
    # 检查 `loss` 对象是否具有 `gradient_proba` 方法，如果有则调用该方法
    if hasattr(loss, "gradient_proba"):
        # 调用 `gradient_proba` 方法，传入以下参数：
        # - y_true: 真实的标签值
        # - raw_prediction: 原始预测值
        # - sample_weight: 样本权重
        # - gradient_out: 梯度输出的位置
        # - proba_out: 概率输出的位置
        # - n_threads: 线程数
        loss.gradient_proba(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            gradient_out=out1,
            proba_out=out2,
            n_threads=n_threads,
        )
# 使用pytest的parametrize装饰器，为测试函数添加参数化测试，loss为LOSS_INSTANCES中的每个实例，ids为loss_instance_name
@pytest.mark.parametrize("loss", LOSS_INSTANCES, ids=loss_instance_name)
# 使用pytest的parametrize装饰器，为测试函数添加参数化测试，sample_weight分别为None和"range"
@pytest.mark.parametrize("sample_weight", [None, "range"])
# 定义测试函数，测试Python和Cython函数返回相同结果
def test_loss_same_as_C_functions(loss, sample_weight):
    """Test that Python and Cython functions return same results."""
    # 生成随机的y_true和raw_prediction作为测试数据
    y_true, raw_prediction = random_y_true_raw_prediction(
        loss=loss,
        n_samples=20,
        y_bound=(-100, 100),
        raw_bound=(-10, 10),
        seed=42,
    )
    # 如果sample_weight为"range"，则创建一个np.linspace的权重数组
    if sample_weight == "range":
        sample_weight = np.linspace(1, y_true.shape[0], num=y_true.shape[0])

    # 初始化空的数组，用于存储损失值和梯度值
    out_l1 = np.empty_like(y_true)
    out_l2 = np.empty_like(y_true)
    out_g1 = np.empty_like(raw_prediction)
    out_g2 = np.empty_like(raw_prediction)
    out_h1 = np.empty_like(raw_prediction)
    out_h2 = np.empty_like(raw_prediction)

    # 计算损失值，将结果存储在out_l1和out_l2中
    loss.loss(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        loss_out=out_l1,
    )
    loss.closs.loss(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        loss_out=out_l2,
    )

    # 断言两种计算方法得到的损失值应该相近
    assert_allclose(out_l1, out_l2)

    # 计算梯度值，将结果存储在out_g1和out_g2中
    loss.gradient(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        gradient_out=out_g1,
    )
    loss.closs.gradient(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        gradient_out=out_g2,
    )

    # 断言两种计算方法得到的梯度值应该相近
    assert_allclose(out_g1, out_g2)

    # 计算损失值和梯度值，将结果存储在out_l1和out_g1中
    loss.closs.loss_gradient(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        loss_out=out_l1,
        gradient_out=out_g1,
    )
    loss.closs.loss_gradient(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        loss_out=out_l2,
        gradient_out=out_g2,
    )

    # 断言两种计算方法得到的损失值和梯度值应该相近
    assert_allclose(out_l1, out_l2)
    assert_allclose(out_g1, out_g2)

    # 计算梯度和Hessian矩阵，将结果存储在out_g1和out_h1中
    loss.gradient_hessian(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        gradient_out=out_g1,
        hessian_out=out_h1,
    )
    loss.closs.gradient_hessian(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        gradient_out=out_g2,
        hessian_out=out_h2,
    )

    # 断言两种计算方法得到的梯度值和Hessian矩阵应该相近
    assert_allclose(out_g1, out_g2)
    assert_allclose(out_h1, out_h2)


# 使用pytest的parametrize装饰器，为测试函数添加参数化测试，loss为LOSS_INSTANCES中的每个实例，ids为loss_instance_name
@pytest.mark.parametrize("loss", LOSS_INSTANCES, ids=loss_instance_name)
# 使用pytest的parametrize装饰器，为测试函数添加参数化测试，sample_weight分别为None和"range"
@pytest.mark.parametrize("sample_weight", [None, "range"])
# 测试函数，测试损失函数的梯度是否相同
def test_loss_gradients_are_the_same(loss, sample_weight, global_random_seed):
    """Test that loss and gradient are the same across different functions.

    Also test that output arguments contain correct results.
    """
    # 生成随机的y_true和raw_prediction作为测试数据
    y_true, raw_prediction = random_y_true_raw_prediction(
        loss=loss,
        n_samples=20,
        y_bound=(-100, 100),
        raw_bound=(-10, 10),
        seed=global_random_seed,
    )
    # 如果样本权重是字符串"range"，则生成一个均匀分布的样本权重数组
    if sample_weight == "range":
        sample_weight = np.linspace(1, y_true.shape[0], num=y_true.shape[0])

    # 初始化用于存储损失函数计算结果的数组，其形状与y_true相同
    out_l1 = np.empty_like(y_true)
    out_l2 = np.empty_like(y_true)
    # 初始化用于存储梯度计算结果的数组，其形状与raw_prediction相同
    out_g1 = np.empty_like(raw_prediction)
    out_g2 = np.empty_like(raw_prediction)
    out_g3 = np.empty_like(raw_prediction)
    out_h3 = np.empty_like(raw_prediction)

    # 计算第一个损失函数及其输出到out_l1
    l1 = loss.loss(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        loss_out=out_l1,
    )
    # 计算第一个梯度及其输出到out_g1
    g1 = loss.gradient(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        gradient_out=out_g1,
    )
    # 计算第二个损失函数及其梯度，并分别输出到out_l2和out_g2
    l2, g2 = loss.loss_gradient(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        loss_out=out_l2,
        gradient_out=out_g2,
    )
    # 计算第三个梯度及其海森矩阵，并分别输出到out_g3和out_h3
    g3, h3 = loss.gradient_hessian(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        gradient_out=out_g3,
        hessian_out=out_h3,
    )

    # 断言两个损失函数的计算结果应该接近
    assert_allclose(l1, l2)
    # 断言第一个损失函数与其输出结果out_l1相等
    assert_array_equal(l1, out_l1)
    # 断言l1与out_l1共享内存
    assert np.shares_memory(l1, out_l1)
    # 断言第二个损失函数与其输出结果out_l2相等
    assert_array_equal(l2, out_l2)
    # 断言l2与out_l2共享内存
    assert np.shares_memory(l2, out_l2)
    # 断言第一个梯度与其输出结果out_g1相等
    assert_allclose(g1, out_g1)
    # 断言第一个梯度与第二个梯度应该接近
    assert_allclose(g1, g2)
    # 断言第一个梯度与第三个梯度应该接近
    assert_allclose(g1, g3)
    # 断言第一个梯度与其输出结果out_g1相等
    assert_array_equal(g1, out_g1)
    # 断言g1与out_g1共享内存
    assert np.shares_memory(g1, out_g1)
    # 断言第二个梯度与其输出结果out_g2相等
    assert_array_equal(g2, out_g2)
    # 断言g2与out_g2共享内存
    assert np.shares_memory(g2, out_g2)
    # 断言第三个梯度与其输出结果out_g3相等
    assert_array_equal(g3, out_g3)
    # 断言g3与out_g3共享内存
    assert np.shares_memory(g3, out_g3)

    # 如果损失函数对象具有gradient_proba方法，则执行以下断言
    if hasattr(loss, "gradient_proba"):
        # 断言损失函数是多分类问题，仅用于HalfMultinomialLoss
        assert loss.is_multiclass
        # 初始化用于存储梯度和概率计算结果的数组
        out_g4 = np.empty_like(raw_prediction)
        out_proba = np.empty_like(raw_prediction)
        # 计算梯度和概率，并分别输出到out_g4和out_proba
        g4, proba = loss.gradient_proba(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            gradient_out=out_g4,
            proba_out=out_proba,
        )
        # 断言第一个梯度与其输出结果out_g4相等
        assert_allclose(g1, out_g4)
        # 断言第一个梯度与g4应该接近
        assert_allclose(g1, g4)
        # 断言概率与其输出结果out_proba相等
        assert_allclose(proba, out_proba)
        # 断言每个样本的概率之和为1
        assert_allclose(np.sum(proba, axis=1), 1, rtol=1e-11)
# 使用 pytest.mark.parametrize 装饰器为 test_sample_weight_multiplies 函数添加参数化测试
# 使用 LOSS_INSTANCES 列表中的值作为 loss 参数的测试数据，使用 loss_instance_name 列表中的值作为测试 ID
@pytest.mark.parametrize("loss", LOSS_INSTANCES, ids=loss_instance_name)
# 为 test_sample_weight_multiplies 函数再次添加参数化测试
# 使用 "ones" 和 "random" 作为 sample_weight 参数的测试数据
def test_sample_weight_multiplies(loss, sample_weight, global_random_seed):
    """Test sample weights in loss, gradients and hessians.

    Make sure that passing sample weights to loss, gradient and hessian
    computation methods is equivalent to multiplying by the weights.
    """
    # 生成随机的 y_true 和 raw_prediction
    n_samples = 100
    y_true, raw_prediction = random_y_true_raw_prediction(
        loss=loss,
        n_samples=n_samples,
        y_bound=(-100, 100),
        raw_bound=(-5, 5),
        seed=global_random_seed,
    )

    # 根据 sample_weight 参数的不同取值，创建不同的权重数组
    if sample_weight == "ones":
        sample_weight = np.ones(shape=n_samples, dtype=np.float64)
    else:
        rng = np.random.RandomState(global_random_seed)
        sample_weight = rng.normal(size=n_samples).astype(np.float64)

    # 断言：使用权重计算的损失函数结果应当等于权重乘以未加权损失函数结果
    assert_allclose(
        loss.loss(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
        ),
        sample_weight
        * loss.loss(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=None,
        ),
    )

    # 计算未加权和加权的损失函数和梯度
    losses, gradient = loss.loss_gradient(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=None,
    )
    losses_sw, gradient_sw = loss.loss_gradient(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
    )
    # 断言：加权损失应当等于未加权损失乘以权重
    assert_allclose(losses * sample_weight, losses_sw)
    # 如果不是多分类问题，断言：加权梯度应当等于未加权梯度乘以权重
    if not loss.is_multiclass:
        assert_allclose(gradient * sample_weight, gradient_sw)
    else:
        assert_allclose(gradient * sample_weight[:, None], gradient_sw)

    # 计算未加权和加权的梯度和黑塞矩阵
    gradient, hessian = loss.gradient_hessian(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=None,
    )
    gradient_sw, hessian_sw = loss.gradient_hessian(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
    )
    # 断言：如果不是多分类问题，加权梯度和黑塞矩阵应当等于未加权值乘以权重
    # 如果是多分类问题，加权梯度和黑塞矩阵应当等于未加权值乘以权重的转置
    if not loss.is_multiclass:
        assert_allclose(gradient * sample_weight, gradient_sw)
        assert_allclose(hessian * sample_weight, hessian_sw)
    else:
        assert_allclose(gradient * sample_weight[:, None], gradient_sw)
        assert_allclose(hessian * sample_weight[:, None], hessian_sw)


# 使用 pytest.mark.parametrize 装饰器为 test_graceful_squeezing 函数添加参数化测试
# 使用 LOSS_INSTANCES 列表中的值作为 loss 参数的测试数据，使用 loss_instance_name 列表中的值作为测试 ID
@pytest.mark.parametrize("loss", LOSS_INSTANCES, ids=loss_instance_name)
def test_graceful_squeezing(loss):
    """Test that reshaped raw_prediction gives same results."""
    # 生成随机的 y_true 和 raw_prediction
    y_true, raw_prediction = random_y_true_raw_prediction(
        loss=loss,
        n_samples=20,
        y_bound=(-100, 100),
        raw_bound=(-10, 10),
        seed=42,
    )
    # 如果 raw_prediction 是一维数组，将其转换为二维数组（列向量形式）
    if raw_prediction.ndim == 1:
        raw_prediction_2d = raw_prediction[:, None]
        # 断言两次调用 loss 函数对同样的 y_true 和不同形式的 raw_prediction 计算的损失值应该接近
        assert_allclose(
            loss.loss(y_true=y_true, raw_prediction=raw_prediction_2d),
            loss.loss(y_true=y_true, raw_prediction=raw_prediction),
        )
        # 断言两次调用 loss_gradient 函数对同样的 y_true 和不同形式的 raw_prediction 计算的梯度应该接近
        assert_allclose(
            loss.loss_gradient(y_true=y_true, raw_prediction=raw_prediction_2d),
            loss.loss_gradient(y_true=y_true, raw_prediction=raw_prediction),
        )
        # 断言两次调用 gradient 函数对同样的 y_true 和不同形式的 raw_prediction 计算的梯度应该接近
        assert_allclose(
            loss.gradient(y_true=y_true, raw_prediction=raw_prediction_2d),
            loss.gradient(y_true=y_true, raw_prediction=raw_prediction),
        )
        # 断言两次调用 gradient_hessian 函数对同样的 y_true 和不同形式的 raw_prediction 计算的梯度和 Hessian 矩阵应该接近
        assert_allclose(
            loss.gradient_hessian(y_true=y_true, raw_prediction=raw_prediction_2d),
            loss.gradient_hessian(y_true=y_true, raw_prediction=raw_prediction),
        )
@pytest.mark.parametrize("loss", LOSS_INSTANCES, ids=loss_instance_name)
# 使用 pytest.mark.parametrize 装饰器，为 loss 参数传递 LOSS_INSTANCES 列表中的实例，使用 loss_instance_name 作为测试用例的标识符
@pytest.mark.parametrize("sample_weight", [None, "range"])
# 使用 pytest.mark.parametrize 装饰器，为 sample_weight 参数传递 None 和 "range" 两个值
def test_loss_of_perfect_prediction(loss, sample_weight):
    """Test value of perfect predictions.

    Loss of y_pred = y_true plus constant_to_optimal_zero should sums up to
    zero.
    """
    if not loss.is_multiclass:
        # Use small values such that exp(value) is not nan.
        raw_prediction = np.array([-10, -0.1, 0, 0.1, 3, 10])
        # If link is identity, we must respect the interval of y_pred:
        if isinstance(loss.link, IdentityLink):
            eps = 1e-10
            low = loss.interval_y_pred.low
            if not loss.interval_y_pred.low_inclusive:
                low = low + eps
            high = loss.interval_y_pred.high
            if not loss.interval_y_pred.high_inclusive:
                high = high - eps
            raw_prediction = np.clip(raw_prediction, low, high)
        y_true = loss.link.inverse(raw_prediction)
    else:
        # HalfMultinomialLoss
        y_true = np.arange(loss.n_classes).astype(float)
        # raw_prediction with entries -exp(10), but +exp(10) on the diagonal
        # this is close enough to np.inf which would produce nan
        raw_prediction = np.full(
            shape=(loss.n_classes, loss.n_classes),
            fill_value=-np.exp(10),
            dtype=float,
        )
        raw_prediction.flat[:: loss.n_classes + 1] = np.exp(10)

    if sample_weight == "range":
        sample_weight = np.linspace(1, y_true.shape[0], num=y_true.shape[0])

    loss_value = loss.loss(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
    )
    constant_term = loss.constant_to_optimal_zero(
        y_true=y_true, sample_weight=sample_weight
    )
    # Comparing loss_value + constant_term to zero would result in large
    # round-off errors.
    assert_allclose(loss_value, -constant_term, atol=1e-14, rtol=1e-15)


@pytest.mark.parametrize("loss", LOSS_INSTANCES, ids=loss_instance_name)
# 使用 pytest.mark.parametrize 装饰器，为 loss 参数传递 LOSS_INSTANCES 列表中的实例，使用 loss_instance_name 作为测试用例的标识符
@pytest.mark.parametrize("sample_weight", [None, "range"])
# 使用 pytest.mark.parametrize 装饰器，为 sample_weight 参数传递 None 和 "range" 两个值
def test_gradients_hessians_numerically(loss, sample_weight, global_random_seed):
    """Test gradients and hessians with numerical derivatives.

    Gradient should equal the numerical derivatives of the loss function.
    Hessians should equal the numerical derivatives of gradients.
    """
    n_samples = 20
    y_true, raw_prediction = random_y_true_raw_prediction(
        loss=loss,
        n_samples=n_samples,
        y_bound=(-100, 100),
        raw_bound=(-5, 5),
        seed=global_random_seed,
    )

    if sample_weight == "range":
        sample_weight = np.linspace(1, y_true.shape[0], num=y_true.shape[0])

    g, h = loss.gradient_hessian(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
    )

    assert g.shape == raw_prediction.shape
    assert h.shape == raw_prediction.shape
    # 如果损失函数不是多分类的情况下
    if not loss.is_multiclass:
    
        # 定义损失函数，接受参数 x，返回损失值
        def loss_func(x):
            return loss.loss(
                y_true=y_true,
                raw_prediction=x,
                sample_weight=sample_weight,
            )
        
        # 计算数值导数，使用 numerical_derivative 函数，计算结果存储在 g_numeric 中
        g_numeric = numerical_derivative(loss_func, raw_prediction, eps=1e-6)
        # 断言 g 与 g_numeric 很接近，使用相对和绝对误差作为容忍度
        assert_allclose(g, g_numeric, rtol=5e-6, atol=1e-10)
        
        # 定义梯度函数，接受参数 x，返回梯度值
        def grad_func(x):
            return loss.gradient(
                y_true=y_true,
                raw_prediction=x,
                sample_weight=sample_weight,
            )
        
        # 计算数值导数，使用 numerical_derivative 函数，计算结果存储在 h_numeric 中
        h_numeric = numerical_derivative(grad_func, raw_prediction, eps=1e-6)
        
        # 如果损失函数使用近似 Hessian 矩阵
        if loss.approx_hessian:
            # TODO: 如果损失函数使用近似 Hessian 矩阵，可以添加对应的测试
            pass
        else:
            # 断言 h 与 h_numeric 很接近，使用相对和绝对误差作为容忍度
            assert_allclose(h, h_numeric, rtol=5e-6, atol=1e-10)
    
    # 如果损失函数是多分类的情况下
    else:
        # 对每一个类别 k 进行循环
        for k in range(loss.n_classes):
            
            # 定义损失函数，接受参数 x，返回损失值，只改变类别 k 的预测值
            def loss_func(x):
                raw = raw_prediction.copy()
                raw[:, k] = x
                return loss.loss(
                    y_true=y_true,
                    raw_prediction=raw,
                    sample_weight=sample_weight,
                )
            
            # 计算数值导数，使用 numerical_derivative 函数，计算结果存储在 g_numeric 中
            g_numeric = numerical_derivative(loss_func, raw_prediction[:, k], eps=1e-5)
            # 断言 g[:, k] 与 g_numeric 很接近，使用相对和绝对误差作为容忍度
            assert_allclose(g[:, k], g_numeric, rtol=5e-6, atol=1e-10)
            
            # 定义梯度函数，接受参数 x，返回梯度值，只改变类别 k 的预测值
            def grad_func(x):
                raw = raw_prediction.copy()
                raw[:, k] = x
                return loss.gradient(
                    y_true=y_true,
                    raw_prediction=raw,
                    sample_weight=sample_weight,
                )[:, k]
            
            # 计算数值导数，使用 numerical_derivative 函数，计算结果存储在 h_numeric 中
            h_numeric = numerical_derivative(grad_func, raw_prediction[:, k], eps=1e-6)
            
            # 如果损失函数使用近似 Hessian 矩阵
            if loss.approx_hessian:
                # TODO: 如果损失函数使用近似 Hessian 矩阵，可以添加对应的测试
                pass
            else:
                # 断言 h[:, k] 与 h_numeric 很接近，使用相对和绝对误差作为容忍度
                assert_allclose(h[:, k], h_numeric, rtol=5e-6, atol=1e-10)
@pytest.mark.parametrize(
    "loss, x0, y_true",
    [
        ("squared_error", -2.0, 42),
        ("squared_error", 117.0, 1.05),
        ("squared_error", 0.0, 0.0),
        # The argmin of binomial_loss for y_true=0 and y_true=1 is resp.
        # -inf and +inf due to logit, cf. "complete separation". Therefore, we
        # use 0 < y_true < 1.
        ("binomial_loss", 0.3, 0.1),
        ("binomial_loss", -12, 0.2),
        ("binomial_loss", 30, 0.9),
        ("poisson_loss", 12.0, 1.0),
        ("poisson_loss", 0.0, 2.0),
        ("poisson_loss", -22.0, 10.0),
    ],
)
@skip_if_32bit
def test_derivatives(loss, x0, y_true):
    """Test that gradients are zero at the minimum of the loss.

    We check this on a single value/sample using Halley's method with the
    first and second order derivatives computed by the Loss instance.
    Note that methods of Loss instances operate on arrays while the newton
    root finder expects a scalar or a one-element array for this purpose.
    """
    # 根据传入的损失函数类型创建对应的损失实例
    loss = _LOSSES[loss](sample_weight=None)
    # 将 y_true 和 x0 转换为 float64 类型的数组
    y_true = np.array([y_true], dtype=np.float64)
    x0 = np.array([x0], dtype=np.float64)

    def func(x: np.ndarray) -> np.ndarray:
        """Compute loss plus constant term.

        The constant term is such that the minimum function value is zero,
        which is required by the Newton method.
        """
        # 计算损失函数加上常数项，使得函数的最小值为零，符合牛顿法的要求
        return loss.loss(
            y_true=y_true, raw_prediction=x
        ) + loss.constant_to_optimal_zero(y_true=y_true)

    def fprime(x: np.ndarray) -> np.ndarray:
        # 计算损失函数关于预测值 x 的梯度
        return loss.gradient(y_true=y_true, raw_prediction=x)

    def fprime2(x: np.ndarray) -> np.ndarray:
        # 计算损失函数关于预测值 x 的梯度的海森矩阵中的第二部分
        return loss.gradient_hessian(y_true=y_true, raw_prediction=x)[1]

    # 使用牛顿法寻找函数的最小值点
    optimum = newton(
        func,
        x0=x0,
        fprime=fprime,
        fprime2=fprime2,
        maxiter=100,
        tol=5e-8,
    )

    # 需要展平数组，因为 assert_allclose 要求维度匹配
    y_true = y_true.ravel()
    optimum = optimum.ravel()
    # 断言最小值点对应的反链接函数值接近真实值 y_true
    assert_allclose(loss.link.inverse(optimum), y_true)
    # 断言函数在最小值点处的值接近零
    assert_allclose(func(optimum), 0, atol=1e-14)
    # 断言在最小值点处的梯度接近零
    assert_allclose(loss.gradient(y_true=y_true, raw_prediction=optimum), 0, atol=5e-7)


@pytest.mark.parametrize("loss", LOSS_INSTANCES, ids=loss_instance_name)
@pytest.mark.parametrize("sample_weight", [None, "range"])
def test_loss_intercept_only(loss, sample_weight):
    """Test that fit_intercept_only returns the argmin of the loss.

    Also test that the gradient is zero at the minimum.
    """
    n_samples = 50
    if not loss.is_multiclass:
        # 如果不是多分类问题，则生成一个线性分布的 y_true
        y_true = loss.link.inverse(np.linspace(-4, 4, num=n_samples))
    else:
        # 如果是多分类问题，则生成一个分类标签
        y_true = np.arange(n_samples).astype(np.float64) % loss.n_classes
        y_true[::5] = 0  # 将部分类别 0 的标签设置得更高，模拟类别分离情况

    if sample_weight == "range":
        # 如果 sample_weight 是 'range'，则生成一个线性分布的权重数组
        sample_weight = np.linspace(0.1, 2, num=n_samples)

    # 调用 fit_intercept_only 方法来找到损失函数的最小值点
    a = loss.fit_intercept_only(y_true=y_true, sample_weight=sample_weight)

    # find minimum by optimization
    # 定义一个函数 `fun`，接受一个参数 `x`
    def fun(x):
        # 如果损失函数不是多类别的
        if not loss.is_multiclass:
            # 用值 `x` 填充长度为 `n_samples` 的数组作为原始预测值
            raw_prediction = np.full(shape=(n_samples), fill_value=x)
        else:
            # 将 `x` 广播到形状为 `(n_samples, loss.n_classes)` 的连续数组中作为原始预测值
            raw_prediction = np.ascontiguousarray(
                np.broadcast_to(x, shape=(n_samples, loss.n_classes))
            )
        # 调用损失函数 `loss` 计算损失值，传入真实标签 `y_true`、原始预测值 `raw_prediction` 和样本权重 `sample_weight`
        return loss(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
        )

    # 如果损失函数不是多类别的
    if not loss.is_multiclass:
        # 使用 minimize_scalar 函数最小化 `fun` 函数，设置容差为 `1e-7`，最大迭代次数为 `100`
        opt = minimize_scalar(fun, tol=1e-7, options={"maxiter": 100})
        # 计算损失函数在 `y_true` 和全为 `a` 的数组上的梯度
        grad = loss.gradient(
            y_true=y_true,
            raw_prediction=np.full_like(y_true, a),
            sample_weight=sample_weight,
        )
        # 断言 `a` 的形状是标量
        assert a.shape == tuple()  # scalar
        # 断言 `a` 的数据类型与 `y_true` 的数据类型相同
        assert a.dtype == y_true.dtype
        # 断言 `a` 的所有元素都是有限的
        assert_all_finite(a)
        # 断言 `a` 等于最小化结果 `opt.x`，允许的相对误差为 `1e-7`
        a == approx(opt.x, rel=1e-7)
        # 断言梯度 `grad` 的总和接近 `0`，允许的绝对误差为 `1e-12`
        grad.sum() == approx(0, abs=1e-12)
    else:
        # 如果损失函数是多类别的
        # 最小化 `fun` 函数，初始值为全零数组 `(loss.n_classes,)`，设置容差为 `1e-13`，最大迭代次数为 `100`，使用 SLSQP 方法
        # 添加约束条件 `sum(raw_prediction) = 0`，通过 LinearConstraint 定义
        opt = minimize(
            fun,
            np.zeros((loss.n_classes)),
            tol=1e-13,
            options={"maxiter": 100},
            method="SLSQP",
            constraints=LinearConstraint(np.ones((1, loss.n_classes)), 0, 0),
        )
        # 计算损失函数在 `y_true` 和重复 `a` 到 `(n_samples, 1)` 形状的数组上的梯度
        grad = loss.gradient(
            y_true=y_true,
            raw_prediction=np.tile(a, (n_samples, 1)),
            sample_weight=sample_weight,
        )
        # 断言 `a` 的数据类型与 `y_true` 的数据类型相同
        assert a.dtype == y_true.dtype
        # 断言 `a` 的所有元素都是有限的
        assert_all_finite(a)
        # 断言 `a` 的所有元素与最小化结果 `opt.x` 接近，允许的相对误差为 `5e-6`，绝对误差为 `1e-12`
        assert_allclose(a, opt.x, rtol=5e-6, atol=1e-12)
        # 断言梯度 `grad` 沿着第一个轴的总和接近 `0`，允许的绝对误差为 `1e-12`
        assert_allclose(grad.sum(axis=0), 0, atol=1e-12)
@pytest.mark.parametrize(
    "loss, func, random_dist",
    [  # 使用 pytest 的 parametrize 装饰器，定义多组参数化测试数据
        (HalfSquaredError(), np.mean, "normal"),  # 使用 HalfSquaredError 类的实例，使用 np.mean 函数，"normal" 分布
        (AbsoluteError(), np.median, "normal"),  # 使用 AbsoluteError 类的实例，使用 np.median 函数，"normal" 分布
        (PinballLoss(quantile=0.25), lambda x: np.percentile(x, q=25), "normal"),  # 使用 PinballLoss 类的实例，使用 lambda 表达式调用 np.percentile 函数，"normal" 分布
        (HalfPoissonLoss(), np.mean, "poisson"),  # 使用 HalfPoissonLoss 类的实例，使用 np.mean 函数，"poisson" 分布
        (HalfGammaLoss(), np.mean, "exponential"),  # 使用 HalfGammaLoss 类的实例，使用 np.mean 函数，"exponential" 分布
        (HalfTweedieLoss(), np.mean, "exponential"),  # 使用 HalfTweedieLoss 类的实例，使用 np.mean 函数，"exponential" 分布
        (HalfBinomialLoss(), np.mean, "binomial"),  # 使用 HalfBinomialLoss 类的实例，使用 np.mean 函数，"binomial" 分布
    ],
)
def test_specific_fit_intercept_only(loss, func, random_dist, global_random_seed):
    """Test that fit_intercept_only returns the correct functional.

    We test the functional for specific, meaningful distributions, e.g.
    squared error estimates the expectation of a probability distribution.
    """
    rng = np.random.RandomState(global_random_seed)  # 创建具有特定种子的随机数生成器
    if random_dist == "binomial":
        y_train = rng.binomial(1, 0.5, size=100)  # 如果是二项分布，生成二项分布的随机数据
    else:
        y_train = getattr(rng, random_dist)(size=100)  # 否则，根据 random_dist 生成相应分布的随机数据
    baseline_prediction = loss.fit_intercept_only(y_true=y_train)  # 调用 fit_intercept_only 方法计算基线预测
    # 确保基线预测的函数值符合预期，例如 func 对应的平均值或中位数
    assert_all_finite(baseline_prediction)  # 检查基线预测结果是否为有限值
    assert baseline_prediction == approx(loss.link.link(func(y_train)))  # 检查基线预测是否等于 func(y_train) 对应的函数值
    assert loss.link.inverse(baseline_prediction) == approx(func(y_train))  # 检查基线预测的反函数值是否等于 func(y_train)
    if isinstance(loss, IdentityLink):
        assert_allclose(loss.link.inverse(baseline_prediction), baseline_prediction)  # 如果 loss 是 IdentityLink 类的实例，检查反函数值是否与基线预测值几乎相等

    # 测试边界处的基线预测
    if loss.interval_y_true.low_inclusive:
        y_train.fill(loss.interval_y_true.low)  # 将 y_train 填充为低边界值
        baseline_prediction = loss.fit_intercept_only(y_true=y_train)  # 计算边界处的基线预测
        assert_all_finite(baseline_prediction)  # 检查边界处的基线预测结果是否为有限值
    if loss.interval_y_true.high_inclusive:
        y_train.fill(loss.interval_y_true.high)  # 将 y_train 填充为高边界值
        baseline_prediction = loss.fit_intercept_only(y_true=y_train)  # 计算边界处的基线预测
        assert_all_finite(baseline_prediction)  # 检查边界处的基线预测结果是否为有限值


def test_multinomial_loss_fit_intercept_only():
    """Test that fit_intercept_only returns the mean functional for CCE."""
    rng = np.random.RandomState(0)  # 创建具有种子 0 的随机数生成器
    n_classes = 4  # 类别数为 4
    loss = HalfMultinomialLoss(n_classes=n_classes)  # 使用 HalfMultinomialLoss 类的实例
    # 与 test_specific_fit_intercept_only 相同的逻辑。这里的反函数 = softmax 函数，函数 = log 函数 - 对称项。
    y_train = rng.randint(0, n_classes + 1, size=100).astype(np.float64)  # 生成随机分类标签数据
    baseline_prediction = loss.fit_intercept_only(y_true=y_train)  # 调用 fit_intercept_only 方法计算基线预测
    assert baseline_prediction.shape == (n_classes,)  # 检查基线预测结果的形状是否正确
    p = np.zeros(n_classes, dtype=y_train.dtype)
    for k in range(n_classes):
        p[k] = (y_train == k).mean()
    assert_allclose(baseline_prediction, np.log(p) - np.mean(np.log(p)))  # 检查基线预测结果是否与预期的函数值几乎相等
    assert_allclose(baseline_prediction[None, :], loss.link.link(p[None, :]))  # 检查基线预测结果的链接函数值是否与预期的几乎相等

    for y_train in (np.zeros(shape=10), np.ones(shape=10)):
        y_train = y_train.astype(np.float64)  # 将 y_train 转换为浮点数类型
        baseline_prediction = loss.fit_intercept_only(y_true=y_train)  # 调用 fit_intercept_only 方法计算基线预测
        assert baseline_prediction.dtype == y_train.dtype  # 检查基线预测结果的数据类型是否正确
        assert_all_finite(baseline_prediction)  # 检查基线预测结果是否为有限值
# 测试二项损失和多项式损失的函数
def test_binomial_and_multinomial_loss(global_random_seed):
    """Test that multinomial loss with n_classes = 2 is the same as binomial loss."""
    # 设置全局随机种子
    rng = np.random.RandomState(global_random_seed)
    # 设定样本数量
    n_samples = 20
    # 创建二项损失对象
    binom = HalfBinomialLoss()
    # 创建多项式损失对象，设定类别数为2
    multinom = HalfMultinomialLoss(n_classes=2)
    # 生成随机的训练标签
    y_train = rng.randint(0, 2, size=n_samples).astype(np.float64)
    # 生成随机的原始预测值
    raw_prediction = rng.normal(size=n_samples)
    # 生成随机的多项式损失的原始预测值矩阵
    raw_multinom = np.empty((n_samples, 2))
    raw_multinom[:, 0] = -0.5 * raw_prediction
    raw_multinom[:, 1] = 0.5 * raw_prediction
    # 断言二项损失和多项式损失在给定相同输入下的损失值近似相等
    assert_allclose(
        binom.loss(y_true=y_train, raw_prediction=raw_prediction),
        multinom.loss(y_true=y_train, raw_prediction=raw_multinom),
    )


@pytest.mark.parametrize("y_true", (np.array([0.0, 0, 0]), np.array([1.0, 1, 1])))
@pytest.mark.parametrize("y_pred", (np.array([-5.0, -5, -5]), np.array([3.0, 3, 3])))
def test_binomial_vs_alternative_formulation(y_true, y_pred, global_dtype):
    """Test that both formulations of the binomial deviance agree.

    Often, the binomial deviance or log loss is written in terms of a variable
    z in {-1, +1}, but we use y in {0, 1}, hence z = 2 * y - 1.
    ESL II Eq. (10.18):

        -loglike(z, f) = log(1 + exp(-2 * z * f))

    Note:
        - ESL 2*f = raw_prediction, hence the factor 2 of ESL disappears.
        - Deviance = -2*loglike + .., but HalfBinomialLoss is half of the
          deviance, hence the factor of 2 cancels in the comparison.
    """

    def alt_loss(y, raw_pred):
        # 根据 ESL 的另一种形式计算损失
        z = 2 * y - 1
        return np.mean(np.log(1 + np.exp(-z * raw_pred)))

    def alt_gradient(y, raw_pred):
        # 根据 ESL 的另一种形式计算梯度
        z = 2 * y - 1
        return -z / (1 + np.exp(z * raw_pred))

    # 创建二项损失对象
    bin_loss = HalfBinomialLoss()

    # 将输入的y_true和y_pred转换为全局数据类型
    y_true = y_true.astype(global_dtype)
    y_pred = y_pred.astype(global_dtype)
    datum = (y_true, y_pred)

    # 断言二项损失和基于 ESL 另一种形式计算的损失在相同输入下近似相等
    assert bin_loss(*datum) == approx(alt_loss(*datum))
    # 断言二项损失和基于 ESL 另一种形式计算的梯度在相同输入下近似相等
    assert_allclose(bin_loss.gradient(*datum), alt_gradient(*datum))


@pytest.mark.parametrize("loss", LOSS_INSTANCES, ids=loss_instance_name)
def test_predict_proba(loss, global_random_seed):
    """Test that predict_proba and gradient_proba work as expected."""
    # 设定样本数量
    n_samples = 20
    # 生成随机的真实标签和原始预测值
    y_true, raw_prediction = random_y_true_raw_prediction(
        loss=loss,
        n_samples=n_samples,
        y_bound=(-100, 100),
        raw_bound=(-5, 5),
        seed=global_random_seed,
    )

    # 如果损失对象具有 predict_proba 方法
    if hasattr(loss, "predict_proba"):
        # 测试预测概率的输出形状和概率和为1的近似性
        proba = loss.predict_proba(raw_prediction)
        assert proba.shape == (n_samples, loss.n_classes)
        assert np.sum(proba, axis=1) == approx(1, rel=1e-11)
    # 检查损失函数对象是否具有 gradient_proba 属性
    if hasattr(loss, "gradient_proba"):
        # 遍历四组 (grad, proba) 组合，初始化为 (None, None), (None, raw_prediction), (raw_prediction, None), (raw_prediction, raw_prediction)
        for grad, proba in (
            (None, None),
            (None, np.empty_like(raw_prediction)),
            (np.empty_like(raw_prediction), None),
            (np.empty_like(raw_prediction), np.empty_like(raw_prediction)),
        ):
            # 调用损失函数对象的 gradient_proba 方法，获取梯度和概率
            grad, proba = loss.gradient_proba(
                y_true=y_true,
                raw_prediction=raw_prediction,
                sample_weight=None,
                gradient_out=grad,
                proba_out=proba,
            )
            # 断言概率数组的形状为 (样本数, 类别数)
            assert proba.shape == (n_samples, loss.n_classes)
            # 断言每行概率的和约等于 1，相对误差为 1e-11
            assert np.sum(proba, axis=1) == approx(1, rel=1e-11)
            # 断言梯度数组与损失函数对象的 gradient 方法计算的结果近似相等
            assert_allclose(
                grad,
                loss.gradient(
                    y_true=y_true,
                    raw_prediction=raw_prediction,
                    sample_weight=None,
                    gradient_out=None,
                ),
            )
# 使用 pytest 的 mark.parametrize 装饰器定义了多个参数化测试用例，对应的参数分别是 ALL_LOSSES
# 所有损失函数列表中的损失函数，以及 sample_weight 可选的 None 和 "range" 两种情况，
# dtype 可选的 np.float32 和 np.float64，以及 order 可选的 "C" 和 "F" 两种存储顺序。
@pytest.mark.parametrize("loss", ALL_LOSSES)
@pytest.mark.parametrize("sample_weight", [None, "range"])
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("order", ("C", "F"))
def test_init_gradient_and_hessians(loss, sample_weight, dtype, order):
    """Test that init_gradient_and_hessian works as expected.

    passing sample_weight to a loss correctly influences the constant_hessian
    attribute, and consequently the shape of the hessian array.
    """
    # 设置测试样本数为 5
    n_samples = 5
    # 如果 sample_weight 为 "range"，则将其设为长度为 n_samples 的全一数组
    if sample_weight == "range":
        sample_weight = np.ones(n_samples)
    # 创建指定损失函数实例，传入 sample_weight 参数
    loss = loss(sample_weight=sample_weight)
    # 调用损失函数的 init_gradient_and_hessian 方法，获取梯度和海森矩阵
    gradient, hessian = loss.init_gradient_and_hessian(
        n_samples=n_samples,
        dtype=dtype,
        order=order,
    )
    # 根据损失函数的 constant_hessian 属性判断
    if loss.constant_hessian:
        # 如果 constant_hessian 为 True，验证梯度和海森矩阵的形状
        assert gradient.shape == (n_samples,)
        assert hessian.shape == (1,)
    elif loss.is_multiclass:
        # 如果损失函数是多分类的，验证梯度和海森矩阵的形状
        assert gradient.shape == (n_samples, loss.n_classes)
        assert hessian.shape == (n_samples, loss.n_classes)
    else:
        # 否则验证海森矩阵的形状
        assert hessian.shape == (n_samples,)
        assert hessian.shape == (n_samples,)

    # 验证梯度和海森矩阵的数据类型是否为指定的 dtype
    assert gradient.dtype == dtype
    assert hessian.dtype == dtype

    # 根据 order 参数验证梯度和海森矩阵的存储顺序是否符合预期
    if order == "C":
        assert gradient.flags.c_contiguous
        assert hessian.flags.c_contiguous
    else:
        assert gradient.flags.f_contiguous
        assert hessian.flags.f_contiguous


@pytest.mark.parametrize("loss", ALL_LOSSES)
@pytest.mark.parametrize(
    "params, err_msg",
    [
        (
            {"dtype": np.int64},
            f"Valid options for 'dtype' are .* Got dtype={np.int64} instead.",
        ),
    ],
)
def test_init_gradient_and_hessian_raises(loss, params, err_msg):
    """Test that init_gradient_and_hessian raises errors for invalid input."""
    # 创建指定损失函数实例
    loss = loss()
    # 使用 pytest 的 raises 断言检查损失函数在参数化输入 params 下是否抛出指定类型和错误消息的异常
    with pytest.raises((ValueError, TypeError), match=err_msg):
        gradient, hessian = loss.init_gradient_and_hessian(n_samples=5, **params)


@pytest.mark.parametrize(
    "loss, params, err_type, err_msg",
    [
        (
            PinballLoss,
            {"quantile": None},
            TypeError,
            "quantile must be an instance of float, not NoneType.",
        ),
        (
            PinballLoss,
            {"quantile": 0},
            ValueError,
            "quantile == 0, must be > 0.",
        ),
        (PinballLoss, {"quantile": 1.1}, ValueError, "quantile == 1.1, must be < 1."),
        (
            HuberLoss,
            {"quantile": None},
            TypeError,
            "quantile must be an instance of float, not NoneType.",
        ),
        (
            HuberLoss,
            {"quantile": 0},
            ValueError,
            "quantile == 0, must be > 0.",
        ),
        (HuberLoss, {"quantile": 1.1}, ValueError, "quantile == 1.1, must be < 1."),
    ],
)
def test_loss_init_parameter_validation(loss, params, err_type, err_msg):
    """Test that loss raises errors for invalid input."""
    # 使用 pytest 模块中的 `raises` 上下文管理器来捕获特定类型的异常，并验证异常消息是否匹配指定的模式
    with pytest.raises(err_type, match=err_msg):
        # 调用 loss 函数，并传入 params 字典中的参数
        loss(**params)
# 使用 pytest.mark.parametrize 装饰器，参数化测试函数，以便多次运行测试并对不同参数进行测试
@pytest.mark.parametrize("loss", LOSS_INSTANCES, ids=loss_instance_name)
# 定义测试函数，测试损失函数能否被序列化（pickled）
def test_loss_pickle(loss):
    """Test that losses can be pickled."""
    # 生成随机的 y_true 和 raw_prediction
    n_samples = 20
    y_true, raw_prediction = random_y_true_raw_prediction(
        loss=loss,
        n_samples=n_samples,
        y_bound=(-100, 100),
        raw_bound=(-5, 5),
        seed=42,
    )
    # 序列化损失函数
    pickled_loss = pickle.dumps(loss)
    # 反序列化损失函数
    unpickled_loss = pickle.loads(pickled_loss)
    # 断言序列化前后损失函数的行为一致
    assert loss(y_true=y_true, raw_prediction=raw_prediction) == approx(
        unpickled_loss(y_true=y_true, raw_prediction=raw_prediction)
    )


# 使用 pytest.mark.parametrize 装饰器，参数化测试函数，测试 Tweedie 损失函数中的不同参数
@pytest.mark.parametrize("p", [-1.5, 0, 1, 1.5, 2, 3])
# 定义测试函数，测试 Tweedie 损失函数在不同参数下的一致性
def test_tweedie_log_identity_consistency(p):
    """Test for identical losses when only the link function is different."""
    # 创建 Tweedie 损失函数对象，分别使用 log 和 identity 连接函数
    half_tweedie_log = HalfTweedieLoss(power=p)
    half_tweedie_identity = HalfTweedieLossIdentity(power=p)
    # 生成随机的 y_true 和 raw_prediction
    n_samples = 10
    y_true, raw_prediction = random_y_true_raw_prediction(
        loss=half_tweedie_log, n_samples=n_samples, seed=42
    )
    # 计算预测值 y_pred
    y_pred = half_tweedie_log.link.inverse(raw_prediction)  # exp(raw_prediction)

    # 比较损失值，忽略某些常数项
    loss_log = half_tweedie_log.loss(
        y_true=y_true, raw_prediction=raw_prediction
    ) + half_tweedie_log.constant_to_optimal_zero(y_true)
    loss_identity = half_tweedie_identity.loss(
        y_true=y_true, raw_prediction=y_pred
    ) + half_tweedie_identity.constant_to_optimal_zero(y_true)
    # 注意 HalfTweedieLoss 和 HalfTweedieLossIdentity 忽略的常数项不同，
    # constant_to_optimal_zero 方法通过添加这些项使得两个损失函数给出相同的值
    assert_allclose(loss_log, loss_identity)

    # 对于梯度和 Hessian 矩阵，常数项并不重要。但是需要考虑链式法则，
    # 当 x=raw_prediction 时，
    #     gradient_log(x) = d/dx loss_log(x)
    #                     = d/dx loss_identity(exp(x))
    #                     = exp(x) * gradient_identity(exp(x))
    # 同样地，
    #     hessian_log(x) = exp(x) * gradient_identity(exp(x))
    #                    + exp(x)**2 * hessian_identity(x)
    gradient_log, hessian_log = half_tweedie_log.gradient_hessian(
        y_true=y_true, raw_prediction=raw_prediction
    )
    gradient_identity, hessian_identity = half_tweedie_identity.gradient_hessian(
        y_true=y_true, raw_prediction=y_pred
    )
    assert_allclose(gradient_log, y_pred * gradient_identity)
    assert_allclose(
        hessian_log, y_pred * gradient_identity + y_pred**2 * hessian_identity
    )
```