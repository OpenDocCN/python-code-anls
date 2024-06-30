# `D:\src\scipysrc\scikit-learn\sklearn\neural_network\tests\test_stochastic_optimizers.py`

```
# 导入必要的库函数
import numpy as np

# 导入需要测试的优化器类和辅助函数
from sklearn.neural_network._stochastic_optimizers import (
    AdamOptimizer,
    BaseOptimizer,
    SGDOptimizer,
)
from sklearn.utils._testing import assert_array_equal

# 定义不同形状的参数列表，用于测试
shapes = [(4, 6), (6, 8), (7, 8, 9)]

# 测试基础优化器类的函数
def test_base_optimizer():
    # 遍历不同的学习率
    for lr in [10**i for i in range(-3, 4)]:
        # 创建基础优化器对象
        optimizer = BaseOptimizer(lr)
        # 断言优化器的停止触发条件，空字符串和假布尔值
        assert optimizer.trigger_stopping("", False)

# 测试没有动量的随机梯度下降优化器函数
def test_sgd_optimizer_no_momentum():
    # 初始化参数列表为全零数组
    params = [np.zeros(shape) for shape in shapes]
    # 使用固定种子生成随机数
    rng = np.random.RandomState(0)

    # 遍历不同的学习率
    for lr in [10**i for i in range(-3, 4)]:
        # 创建随机梯度下降优化器对象
        optimizer = SGDOptimizer(params, lr, momentum=0, nesterov=False)
        # 生成随机梯度
        grads = [rng.random_sample(shape) for shape in shapes]
        # 计算预期参数更新值
        expected = [param - lr * grad for param, grad in zip(params, grads)]
        # 更新参数
        optimizer.update_params(params, grads)

        # 断言参数更新后的值与预期相等
        for exp, param in zip(expected, params):
            assert_array_equal(exp, param)

# 测试有动量的随机梯度下降优化器函数
def test_sgd_optimizer_momentum():
    # 初始化参数列表为全零数组
    params = [np.zeros(shape) for shape in shapes]
    lr = 0.1  # 学习率
    rng = np.random.RandomState(0)

    # 遍历不同的动量值
    for momentum in np.arange(0.5, 0.9, 0.1):
        # 创建有动量的随机梯度下降优化器对象
        optimizer = SGDOptimizer(params, lr, momentum=momentum, nesterov=False)
        # 随机初始化速度
        velocities = [rng.random_sample(shape) for shape in shapes]
        optimizer.velocities = velocities
        # 生成随机梯度
        grads = [rng.random_sample(shape) for shape in shapes]
        # 计算参数更新值
        updates = [
            momentum * velocity - lr * grad for velocity, grad in zip(velocities, grads)
        ]
        expected = [param + update for param, update in zip(params, updates)]
        # 更新参数
        optimizer.update_params(params, grads)

        # 断言参数更新后的值与预期相等
        for exp, param in zip(expected, params):
            assert_array_equal(exp, param)

# 测试触发停止条件的随机梯度下降优化器函数
def test_sgd_optimizer_trigger_stopping():
    # 初始化参数列表为全零数组
    params = [np.zeros(shape) for shape in shapes]
    lr = 2e-6  # 学习率
    optimizer = SGDOptimizer(params, lr, lr_schedule="adaptive")
    # 断言不触发停止条件
    assert not optimizer.trigger_stopping("", False)
    # 断言学习率的更新值
    assert lr / 5 == optimizer.learning_rate
    # 断言触发停止条件
    assert optimizer.trigger_stopping("", False)

# 测试有 Nesterov 动量的随机梯度下降优化器函数
def test_sgd_optimizer_nesterovs_momentum():
    # 初始化参数列表为全零数组
    params = [np.zeros(shape) for shape in shapes]
    lr = 0.1  # 学习率
    rng = np.random.RandomState(0)

    # 遍历不同的动量值
    for momentum in np.arange(0.5, 0.9, 0.1):
        # 创建有 Nesterov 动量的随机梯度下降优化器对象
        optimizer = SGDOptimizer(params, lr, momentum=momentum, nesterov=True)
        # 随机初始化速度
        velocities = [rng.random_sample(shape) for shape in shapes]
        optimizer.velocities = velocities
        # 生成随机梯度
        grads = [rng.random_sample(shape) for shape in shapes]
        # 计算参数更新值
        updates = [
            momentum * velocity - lr * grad for velocity, grad in zip(velocities, grads)
        ]
        updates = [
            momentum * update - lr * grad for update, grad in zip(updates, grads)
        ]
        expected = [param + update for param, update in zip(params, updates)]
        # 更新参数
        optimizer.update_params(params, grads)

        # 断言参数更新后的值与预期相等
        for exp, param in zip(expected, params):
            assert_array_equal(exp, param)

# 测试 Adam 优化器函数
def test_adam_optimizer():
    # 初始化参数列表为全零数组
    params = [np.zeros(shape) for shape in shapes]
    # 设置初始学习率
    lr = 0.001
    # 防止除零错误的小常数
    epsilon = 1e-8
    # 初始化随机数生成器
    rng = np.random.RandomState(0)

    # 循环遍历不同的 beta_1 值
    for beta_1 in np.arange(0.9, 1.0, 0.05):
        # 循环遍历不同的 beta_2 值
        for beta_2 in np.arange(0.995, 1.0, 0.001):
            # 使用 Adam 优化器初始化
            optimizer = AdamOptimizer(params, lr, beta_1, beta_2, epsilon)
            # 使用随机数生成器生成与参数形状相同的随机数作为初始 ms
            ms = [rng.random_sample(shape) for shape in shapes]
            # 使用随机数生成器生成与参数形状相同的随机数作为初始 vs
            vs = [rng.random_sample(shape) for shape in shapes]
            # 设置初始时间步 t
            t = 10
            # 将 optimizer 的 ms, vs, t 设置为当前的 ms, vs, t-1
            optimizer.ms = ms
            optimizer.vs = vs
            optimizer.t = t - 1
            # 使用随机数生成器生成与参数形状相同的随机数作为梯度值 grads
            grads = [rng.random_sample(shape) for shape in shapes]

            # 更新 ms：m = beta_1 * m + (1 - beta_1) * grad
            ms = [beta_1 * m + (1 - beta_1) * grad for m, grad in zip(ms, grads)]
            # 更新 vs：v = beta_2 * v + (1 - beta_2) * (grad**2)
            vs = [beta_2 * v + (1 - beta_2) * (grad**2) for v, grad in zip(vs, grads)]
            # 计算当前时间步的学习率
            learning_rate = lr * np.sqrt(1 - beta_2**t) / (1 - beta_1**t)
            # 计算参数更新的量
            updates = [
                -learning_rate * m / (np.sqrt(v) + epsilon) for m, v in zip(ms, vs)
            ]
            # 计算预期的参数更新后的值
            expected = [param + update for param, update in zip(params, updates)]

            # 使用 optimizer 更新参数 params
            optimizer.update_params(params, grads)
            # 断言更新后的参数与预期值相等
            for exp, param in zip(expected, params):
                assert_array_equal(exp, param)
```