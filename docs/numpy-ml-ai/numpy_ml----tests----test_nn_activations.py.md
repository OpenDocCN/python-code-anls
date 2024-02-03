# `numpy-ml\numpy_ml\tests\test_nn_activations.py`

```
# 禁用 flake8 检查
# 导入时间模块
import time
# 导入 numpy 模块并重命名为 np
import numpy as np

# 从 numpy.testing 模块中导入 assert_almost_equal 函数
from numpy.testing import assert_almost_equal
# 从 scipy.special 模块中导入 expit 函数
from scipy.special import expit

# 导入 torch 模块
import torch
# 从 torch.nn.functional 模块中导入 F
import torch.nn.functional as F

# 从 numpy_ml.utils.testing 模块中导入 random_stochastic_matrix 和 random_tensor 函数
from numpy_ml.utils.testing import random_stochastic_matrix, random_tensor

# 定义一个函数，用于生成 torch 梯度
def torch_gradient_generator(fn, **kwargs):
    # 定义内部函数 get_grad，用于计算梯度
    def get_grad(z):
        # 将 numpy 数组 z 转换为 torch 变量，并设置 requires_grad 为 True
        z1 = torch.autograd.Variable(torch.from_numpy(z), requires_grad=True)
        # 调用传入的函数 fn 计算 z1 的值，并对结果求和
        z2 = fn(z1, **kwargs).sum()
        # 对 z2 进行反向传播
        z2.backward()
        # 获取 z1 的梯度，并转换为 numpy 数组返回
        grad = z1.grad.numpy()
        return grad

    return get_grad


#######################################################################
#                           Debug Formatter                           #
#######################################################################

# 定义一个函数，用于格式化错误信息
def err_fmt(params, golds, ix, warn_str=""):
    mine, label = params[ix]
    err_msg = "-" * 25 + " DEBUG " + "-" * 25 + "\n"
    prev_mine, prev_label = params[max(ix - 1, 0)]
    err_msg += "Mine (prev) [{}]:\n{}\n\nTheirs (prev) [{}]:\n{}".format(
        prev_label, prev_mine, prev_label, golds[prev_label]
    )
    err_msg += "\n\nMine [{}]:\n{}\n\nTheirs [{}]:\n{}".format(
        label, mine, label, golds[label]
    )
    err_msg += warn_str
    err_msg += "\n" + "-" * 23 + " END DEBUG " + "-" * 23
    return err_msg


#######################################################################
#                            Test Suite                               #
#######################################################################
#
#
#  def test_activations(N=50):
#      print("Testing Sigmoid activation")
#      time.sleep(1)
#      test_sigmoid_activation(N)
#      test_sigmoid_grad(N)
#
#      #  print("Testing Softmax activation")
#      #  time.sleep(1)
#      #  test_softmax_activation(N)
#      #  test_softmax_grad(N)
#
#      print("Testing Tanh activation")
#      time.sleep(1)
#      test_tanh_grad(N)
#
#      print("Testing ReLU activation")
#      time.sleep(1)
#      test_relu_activation(N)
#      test_relu_grad(N)
#
#      print("Testing ELU activation")
#      time.sleep(1)
#      test_elu_activation(N)
#      test_elu_grad(N)
#
#      print("Testing SELU activation")
#      time.sleep(1)
#      test_selu_activation(N)
#      test_selu_grad(N)
#
#      print("Testing LeakyRelu activation")
#      time.sleep(1)
#      test_leakyrelu_activation(N)
#      test_leakyrelu_grad(N)
#
#      print("Testing SoftPlus activation")
#      time.sleep(1)
#      test_softplus_activation(N)
#      test_softplus_grad(N)
#

#######################################################################
#                          Activations                                #
#######################################################################


# 测试 Sigmoid 激活函数
def test_sigmoid_activation(N=50):
    # 导入 Sigmoid 激活函数
    from numpy_ml.neural_nets.activations import Sigmoid

    # 如果 N 为 None，则设为无穷大
    N = np.inf if N is None else N

    # 创建 Sigmoid 激活函数对象
    mine = Sigmoid()
    # 创建 gold 函数对象，用于比较
    gold = expit

    i = 0
    while i < N:
        # 生成随机维度
        n_dims = np.random.randint(1, 100)
        # 生成随机张量
        z = random_tensor((1, n_dims))
        # 断言 Sigmoid 函数计算结果与 gold 函数计算结果几乎相等
        assert_almost_equal(mine.fn(z), gold(z))
        print("PASSED")
        i += 1


# 测试 SoftPlus 激活函数
def test_softplus_activation(N=50):
    # 导入 SoftPlus 激活函数
    from numpy_ml.neural_nets.activations import SoftPlus

    # 如果 N 为 None，则设为无穷大
    N = np.inf if N is None else N

    # 创建 SoftPlus 激活函数对象
    mine = SoftPlus()
    # 创建 gold 函数对象，用于比较
    gold = lambda z: F.softplus(torch.FloatTensor(z)).numpy()

    i = 0
    while i < N:
        # 生成随机维度
        n_dims = np.random.randint(1, 100)
        # 生成随机随机矩阵
        z = random_stochastic_matrix(1, n_dims)
        # 断言 SoftPlus 函数计算结果与 gold 函数计算结果几乎相等
        assert_almost_equal(mine.fn(z), gold(z))
        print("PASSED")
        i += 1


# 测试 ELU 激活函数
def test_elu_activation(N=50):
    # 导入 ELU 激活函数
    from numpy_ml.neural_nets.activations import ELU

    # 如果 N 为 None，则设为无穷大
    N = np.inf if N is None else N

    i = 0
    while i < N:
        # 生成随机维度
        n_dims = np.random.randint(1, 10)
        # 生成随机张量
        z = random_tensor((1, n_dims))

        # 生成随机 alpha 值
        alpha = np.random.uniform(0, 10)

        # 创建 ELU 激活函数对象
        mine = ELU(alpha)
        # 创建 gold 函数对象，用于比较
        gold = lambda z, a: F.elu(torch.from_numpy(z), alpha).numpy()

        # 断言 ELU 函数计算结果与 gold 函数计算结果几乎相等
        assert_almost_equal(mine.fn(z), gold(z, alpha))
        print("PASSED")
        i += 1
# 测试 ReLU 激活函数的功能
def test_relu_activation(N=50):
    # 导入 ReLU 激活函数
    from numpy_ml.neural_nets.activations import ReLU

    # 如果 N 为 None，则将 N 设置为无穷大
    N = np.inf if N is None else N

    # 创建自定义的 ReLU 激活函数对象
    mine = ReLU()
    # 创建 PyTorch 中的 ReLU 激活函数对象
    gold = lambda z: F.relu(torch.FloatTensor(z)).numpy()

    # 初始化计数器 i
    i = 0
    # 循环执行 N 次
    while i < N:
        # 随机生成维度在 1 到 100 之间的随机数
        n_dims = np.random.randint(1, 100)
        # 生成一个随机的矩阵 z
        z = random_stochastic_matrix(1, n_dims)
        # 断言自定义的 ReLU 激活函数和 PyTorch 的 ReLU 激活函数的输出几乎相等
        assert_almost_equal(mine.fn(z), gold(z))
        # 打印 "PASSED"
        print("PASSED")
        # 更新计数器
        i += 1


# 测试 SELU 激活函数的功能
def test_selu_activation(N=50):
    # 导入 SELU 激活函数
    from numpy_ml.neural_nets.activations import SELU

    # 如果 N 为 None，则将 N 设置为无穷大
    N = np.inf if N is None else N

    # 创建自定义的 SELU 激活函数对象
    mine = SELU()
    # 创建 PyTorch 中的 SELU 激活函数对象
    gold = lambda z: F.selu(torch.FloatTensor(z)).numpy()

    # 初始化计数器 i
    i = 0
    # 循环执行 N 次
    while i < N:
        # 随机生成维度在 1 到 100 之间的随机数
        n_dims = np.random.randint(1, 100)
        # 生成一个随机的矩阵 z
        z = random_stochastic_matrix(1, n_dims)
        # 断言自定义的 SELU 激活函数和 PyTorch 的 SELU 激活函数的输出几乎相等
        assert_almost_equal(mine.fn(z), gold(z))
        # 打印 "PASSED"
        print("PASSED")
        # 更新计数器
        i += 1


# 测试 LeakyReLU 激活函数的功能
def test_leakyrelu_activation(N=50):
    # 导入 LeakyReLU 激活函数
    from numpy_ml.neural_nets.activations import LeakyReLU

    # 如果 N 为 None，则将 N 设置为无穷大
    N = np.inf if N is None else N

    # 初始化计数器 i
    i = 0
    # 循环执行 N 次
    while i < N:
        # 随机生成维度在 1 到 100 之间的随机数
        n_dims = np.random.randint(1, 100)
        # 生成一个随机的矩阵 z
        z = random_stochastic_matrix(1, n_dims)
        # 随机生成一个在 0 到 10 之间的 alpha 值
        alpha = np.random.uniform(0, 10)

        # 创建自定义的 LeakyReLU 激活函数对象
        mine = LeakyReLU(alpha=alpha)
        # 创建 PyTorch 中的 LeakyReLU 激活函数对象
        gold = lambda z: F.leaky_relu(torch.FloatTensor(z), alpha).numpy()
        # 断言自定义的 LeakyReLU 激活函数和 PyTorch 的 LeakyReLU 激活函数的输出几乎相等
        assert_almost_equal(mine.fn(z), gold(z))

        # 打印 "PASSED"
        print("PASSED")
        # 更新计数器
        i += 1


# 测试 GELU 激活函数的功能
def test_gelu_activation(N=50):
    # 导入 GELU 激活函数
    from numpy_ml.neural_nets.activations import GELU

    # 如果 N 为 None，则将 N 设置为无穷大
    N = np.inf if N is None else N

    # 初始化计数器 i
    i = 0
    # 循环执行 N 次
    while i < N:
        # 随机生成维度在 1 到 100 之间的随机数
        n_dims = np.random.randint(1, 100)
        # 生成一个随机的矩阵 z
        z = random_stochastic_matrix(1, n_dims)
        # 随机选择是否使用近似计算
        approx = np.random.choice([True, False])

        # 创建自定义的 GELU 激活函数对象
        mine = GELU(approximate=False)
        mine_approx = GELU(approximate=True)
        # 创建 PyTorch 中的 GELU 激活函数对象
        gold = lambda z: F.gelu(torch.FloatTensor(z)).numpy()
        # 断言自定义的 GELU 激活函数和 PyTorch 的 GELU 激活函数的输出在相对误差范围内接近
        np.testing.assert_allclose(mine.fn(z), gold(z), rtol=1e-3)
        # 断言自定义的 GELU 激活函数和近似计算的 GELU 激活函数的输出几乎相等
        assert_almost_equal(mine.fn(z), mine_approx.fn(z))

        # 打印 "PASSED"
        print("PASSED")
        # 更新计数器
        i += 1


#######################################################################
#                      Activation Gradients                           #
#######################################################################

# 测试 Sigmoid 激活函数的梯度
def test_sigmoid_grad(N=50):
    # 导入 Sigmoid 激活函数
    from numpy_ml.neural_nets.activations import Sigmoid

    # 如果 N 为 None，则设为无穷大
    N = np.inf if N is None else N

    # 创建 Sigmoid 激活函数对象
    mine = Sigmoid()
    # 创建 PyTorch 中 Sigmoid 激活函数的梯度函数
    gold = torch_gradient_generator(torch.sigmoid)

    i = 0
    while i < N:
        # 生成随机的样本数和维度
        n_ex = np.random.randint(1, 100)
        n_dims = np.random.randint(1, 100)
        z = random_tensor((n_ex, n_dims))
        # 断言 mine 的梯度与 gold 的梯度几乎相等
        assert_almost_equal(mine.grad(z), gold(z))
        print("PASSED")
        i += 1

# 测试 ELU 激活函数的梯度
def test_elu_grad(N=50):
    # 导入 ELU 激活函数
    from numpy_ml.neural_nets.activations import ELU

    # 如果 N 为 None，则设为无穷大
    N = np.inf if N is None else N

    i = 0
    while i < N:
        # 生成随机的样本数、维度和 alpha 值
        n_ex = np.random.randint(1, 10)
        n_dims = np.random.randint(1, 10)
        alpha = np.random.uniform(0, 10)
        z = random_tensor((n_ex, n_dims))

        # 创建 ELU 激活函数对象和 PyTorch 中 ELU 激活函数的梯度函数
        mine = ELU(alpha)
        gold = torch_gradient_generator(F.elu, alpha=alpha)
        # 断言 mine 的梯度与 gold 的梯度几乎相等
        assert_almost_equal(mine.grad(z), gold(z), decimal=6)
        print("PASSED")
        i += 1

# 测试 Tanh 激活函数的梯度
def test_tanh_grad(N=50):
    # 导入 Tanh 激活函数
    from numpy_ml.neural_nets.activations import Tanh

    # 如果 N 为 None，则设为无穷大
    N = np.inf if N is None else N

    # 创建 Tanh 激活函数对象
    mine = Tanh()
    # 创建 PyTorch 中 Tanh 激活函数的梯度函数
    gold = torch_gradient_generator(torch.tanh)

    i = 0
    while i < N:
        # 生成随机的样本数和维度
        n_ex = np.random.randint(1, 100)
        n_dims = np.random.randint(1, 100)
        z = random_tensor((n_ex, n_dims))
        # 断言 mine 的梯度与 gold 的梯度几乎相等
        assert_almost_equal(mine.grad(z), gold(z))
        print("PASSED")
        i += 1

# 测试 ReLU 激活函数的梯度
def test_relu_grad(N=50):
    # 导入 ReLU 激活函数
    from numpy_ml.neural_nets.activations import ReLU

    # 如果 N 为 None，则设为无穷大
    N = np.inf if N is None else N

    # 创建 ReLU 激活函数对象
    mine = ReLU()
    # 创建 PyTorch 中 ReLU 激活函数的梯度函数
    gold = torch_gradient_generator(F.relu)

    i = 0
    while i < N:
        # 生成随机的样本数和维度
        n_ex = np.random.randint(1, 100)
        n_dims = np.random.randint(1, 100)
        z = random_tensor((n_ex, n_dims))
        # 断言 mine 的梯度与 gold 的梯度几乎相等
        assert_almost_equal(mine.grad(z), gold(z))
        print("PASSED")
        i += 1

# 测试 GELU 激活函数的梯度
def test_gelu_grad(N=50):
    # 从 numpy_ml.neural_nets.activations 模块中导入 GELU 激活函数
    from numpy_ml.neural_nets.activations import GELU
    
    # 如果 N 为 None，则将 N 设置为正无穷
    N = np.inf if N is None else N
    
    # 创建一个不使用近似的 GELU 激活函数对象
    mine = GELU(approximate=False)
    # 创建一个使用近似的 GELU 激活函数对象
    mine_approx = GELU(approximate=True)
    # 创建一个使用 PyTorch 的 GELU 梯度生成器对象
    gold = torch_gradient_generator(F.gelu)
    
    # 初始化计数器 i
    i = 0
    # 当 i 小于 N 时执行循环
    while i < N:
        # 生成随机的样本数和维度数
        n_ex = np.random.randint(1, 100)
        n_dims = np.random.randint(1, 100)
        # 生成随机的张量 z
        z = random_tensor((n_ex, n_dims))
        # 断言 mine 激活函数的梯度与 gold 激活函数的梯度在小数点后三位上几乎相等
        assert_almost_equal(mine.grad(z), gold(z), decimal=3)
        # 断言 mine 激活函数的梯度与 mine_approx 激活函数的梯度在小数点后三位上几乎相等
        assert_almost_equal(mine.grad(z), mine_approx.grad(z))
        # 打印 "PASSED"
        print("PASSED")
        # 计数器 i 自增
        i += 1
# 测试 SELU 激活函数的梯度计算
def test_selu_grad(N=50):
    # 导入 SELU 激活函数
    from numpy_ml.neural_nets.activations import SELU

    # 如果 N 为 None，则将 N 设置为无穷大
    N = np.inf if N is None else N

    # 创建 SELU 激活函数对象
    mine = SELU()
    # 创建 PyTorch 中 SELU 激活函数的梯度计算函数
    gold = torch_gradient_generator(F.selu)

    # 初始化计数器
    i = 0
    # 循环进行 N 次测试
    while i < N:
        # 随机生成样本数和维度
        n_ex = np.random.randint(1, 100)
        n_dims = np.random.randint(1, 100)
        # 生成随机张量
        z = random_tensor((n_ex, n_dims))
        # 断言 SELU 激活函数的梯度与 PyTorch 中的梯度计算函数结果相近
        assert_almost_equal(mine.grad(z), gold(z), decimal=6)
        # 打印测试通过信息
        print("PASSED")
        i += 1


# 测试 LeakyReLU 激活函数的梯度计算
def test_leakyrelu_grad(N=50):
    # 导入 LeakyReLU 激活函数
    from numpy_ml.neural_nets.activations import LeakyReLU

    # 如果 N 为 None，则将 N 设置为无穷大
    N = np.inf if N is None else N

    # 初始化计数器
    i = 0
    # 循环进行 N 次测试
    while i < N:
        # 随机生成样本数、维度和 alpha 参数
        n_ex = np.random.randint(1, 10)
        n_dims = np.random.randint(1, 10)
        alpha = np.random.uniform(0, 10)
        # 生成随机张量
        z = random_tensor((n_ex, n_dims))

        # 创建 LeakyReLU 激活函数对象
        mine = LeakyReLU(alpha)
        # 创建 PyTorch 中 LeakyReLU 激活函数的梯度计算函数
        gold = torch_gradient_generator(F.leaky_relu, negative_slope=alpha)
        # 断言 LeakyReLU 激活函数的梯度与 PyTorch 中的梯度计算函数结果相近
        assert_almost_equal(mine.grad(z), gold(z), decimal=6)
        # 打印测试通过信息
        print("PASSED")
        i += 1


# 测试 SoftPlus 激活函数的梯度计算
def test_softplus_grad(N=50):
    # 导入 SoftPlus 激活函数
    from numpy_ml.neural_nets.activations import SoftPlus

    # 如果 N 为 None，则将 N 设置为无穷大
    N = np.inf if N is None else N

    # 创建 SoftPlus 激活函数对象
    mine = SoftPlus()
    # 创建 PyTorch 中 SoftPlus 激活函数的梯度计算函数
    gold = torch_gradient_generator(F.softplus)

    # 初始化计数器
    i = 0
    # 循环进行 N 次测试
    while i < N:
        # 随机生成样本数、维度，并标准化生成的随机张量
        n_ex = np.random.randint(1, 100)
        n_dims = np.random.randint(1, 100)
        z = random_tensor((n_ex, n_dims), standardize=True)
        # 断言 SoftPlus 激活函数的梯度与 PyTorch 中的梯度计算函数结果相近
        assert_almost_equal(mine.grad(z), gold(z))
        # 打印测试通过信息
        print("PASSED")
        i += 1


# 如果作为主程序运行，则执行激活函数测试
if __name__ == "__main__":
    test_activations(N=50)
```