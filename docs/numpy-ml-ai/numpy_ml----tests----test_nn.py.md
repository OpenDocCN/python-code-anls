# `numpy-ml\numpy_ml\tests\test_nn.py`

```
# 禁用 flake8 检查
# 导入时间模块
import time
# 从 copy 模块中导入 deepcopy 函数
from copy import deepcopy

# 导入 numpy 模块，并将其命名为 np
import numpy as np
# 从 numpy.testing 模块中导入 assert_almost_equal 函数
from numpy.testing import assert_almost_equal

# 导入 sklearn.metrics 模块中的 log_loss 和 mean_squared_error 函数
from sklearn.metrics import log_loss, mean_squared_error

# 导入 scipy.special 模块中的 expit 函数，用于测试 sigmoid 函数
from scipy.special import expit

# 导入 torch 模块
import torch
# 从 torch.nn 模块中导入 nn 和 F
import torch.nn as nn
import torch.nn.functional as F

# 从 numpy_ml.neural_nets.utils 模块中导入一系列函数
from numpy_ml.neural_nets.utils import (
    calc_pad_dims_2D,
    conv2D_naive,
    conv2D,
    pad2D,
    pad1D,
)
# 从 numpy_ml.utils.testing 模块中导入一系列函数
from numpy_ml.utils.testing import (
    random_one_hot_matrix,
    random_stochastic_matrix,
    random_tensor,
)

# 从当前目录下的 nn_torch_models 模块中导入一系列类和函数
from .nn_torch_models import (
    TFNCELoss,
    WGAN_GP_tf,
    torch_xe_grad,
    torch_mse_grad,
    TorchVAELoss,
    TorchFCLayer,
    TorchRNNCell,
    TorchLSTMCell,
    TorchAddLayer,
    TorchWGANGPLoss,
    TorchConv1DLayer,
    TorchConv2DLayer,
    TorchPool2DLayer,
    TorchWavenetModule,
    TorchMultiplyLayer,
    TorchDeconv2DLayer,
    TorchLayerNormLayer,
    TorchBatchNormLayer,
    TorchEmbeddingLayer,
    TorchLinearActivation,
    TorchSDPAttentionLayer,
    TorchBidirectionalLSTM,
    torch_gradient_generator,
    TorchSkipConnectionConv,
    TorchSkipConnectionIdentity,
    TorchMultiHeadedAttentionModule,
)

#######################################################################
#                           Debug Formatter                           #
#######################################################################

# 定义一个函数，用于格式化错误信息
def err_fmt(params, golds, ix, warn_str=""):
    # 获取当前参数和标签
    mine, label = params[ix]
    # 构建错误信息字符串
    err_msg = "-" * 25 + " DEBUG " + "-" * 25 + "\n"
    # 获取前一个参数和标签
    prev_mine, prev_label = params[max(ix - 1, 0)]
    # 添加前一个参数和标签的信息到错误信息字符串中
    err_msg += "Mine (prev) [{}]:\n{}\n\nTheirs (prev) [{}]:\n{}".format(
        prev_label, prev_mine, prev_label, golds[prev_label]
    )
    # 添加当前参数和标签的信息到错误信息字符串中
    err_msg += "\n\nMine [{}]:\n{}\n\nTheirs [{}]:\n{}".format(
        label, mine, label, golds[label]
    )
    # 添加警告信息到错误信息字符串中
    err_msg += warn_str
    err_msg += "\n" + "-" * 23 + " END DEBUG " + "-" * 23
    # 返回错误信息字符串
    return err_msg

#######################################################################
#                         Loss Functions                              #
#######################################################################

# 测试均方误差损失函数
def test_squared_error(N=15):
    # 从 numpy_ml.neural_nets.losses 模块导入 SquaredError 类
    from numpy_ml.neural_nets.losses import SquaredError

    # 设置随机种子
    np.random.seed(12345)

    # 如果 N 为 None，则将 N 设置为无穷大
    N = np.inf if N is None else N

    # 创建 SquaredError 实例
    mine = SquaredError()
    # 创建参考的均方误差损失函数
    gold = (
        lambda y, y_pred: mean_squared_error(y, y_pred)
        * y_pred.shape[0]
        * y_pred.shape[1]
        * 0.5
    )

    # 确保当两个数组相等时得到 0
    n_dims = np.random.randint(2, 100)
    n_examples = np.random.randint(1, 1000)
    y = y_pred = random_tensor((n_examples, n_dims))
    assert_almost_equal(mine.loss(y, y_pred), gold(y, y_pred))
    print("PASSED")

    i = 1
    while i < N:
        n_dims = np.random.randint(2, 100)
        n_examples = np.random.randint(1, 1000)
        y = random_tensor((n_examples, n_dims))
        y_pred = random_tensor((n_examples, n_dims))
        assert_almost_equal(mine.loss(y, y_pred), gold(y, y_pred), decimal=5)
        print("PASSED")
        i += 1

# 测试交叉熵损失函数
def test_cross_entropy(N=15):
    # 从 numpy_ml.neural_nets.losses 模块导入 CrossEntropy 类
    from numpy_ml.neural_nets.losses import CrossEntropy

    # 设置随机种子
    np.random.seed(12345)

    # 如果 N 为 None，则将 N 设置为无穷大
    N = np.inf if N is None else N

    # 创建 CrossEntropy 实例
    mine = CrossEntropy()
    # 创建参考的对数损失函数
    gold = log_loss

    # 确保当两个数组相等时得到 0
    n_classes = np.random.randint(2, 100)
    n_examples = np.random.randint(1, 1000)
    y = y_pred = random_one_hot_matrix(n_examples, n_classes)
    assert_almost_equal(mine.loss(y, y_pred), gold(y, y_pred))
    print("PASSED")

    # 在随机输入上进行测试
    i = 1
    while i < N:
        n_classes = np.random.randint(2, 100)
        n_examples = np.random.randint(1, 1000)
        y = random_one_hot_matrix(n_examples, n_classes)
        y_pred = random_stochastic_matrix(n_examples, n_classes)

        assert_almost_equal(mine.loss(y, y_pred), gold(y, y_pred, normalize=False))
        print("PASSED")
        i += 1

# 测试变分自编码器损失函数
def test_VAE_loss(N=15):
    # 从numpy_ml.neural_nets.losses模块中导入VAELoss类
    from numpy_ml.neural_nets.losses import VAELoss

    # 设置随机种子为12345
    np.random.seed(12345)

    # 如果N为None，则将N设置为无穷大，否则保持不变
    N = np.inf if N is None else N
    # 计算浮点数的最小值
    eps = np.finfo(float).eps

    # 初始化循环变量i为1
    i = 1
    # 当i小于N时执行循环
    while i < N:
        # 生成1到10之间的随机整数作为样本数
        n_ex = np.random.randint(1, 10)
        # 生成2到10之间的随机整数作为特征维度
        t_dim = np.random.randint(2, 10)
        # 生成服从标准化的随机张量作为t_mean
        t_mean = random_tensor([n_ex, t_dim], standardize=True)
        # 生成服从标准化的随机张量，取对数绝对值后加上eps作为t_log_var
        t_log_var = np.log(np.abs(random_tensor([n_ex, t_dim], standardize=True) + eps))
        # 生成2到40之间的随机整数作为图像列数和行数
        im_cols, im_rows = np.random.randint(2, 40), np.random.randint(2, 40)
        # 生成n_ex行im_rows*im_cols列的随机矩阵作为X和X_recon
        X = np.random.rand(n_ex, im_rows * im_cols)
        X_recon = np.random.rand(n_ex, im_rows * im_cols)

        # 创建VAELoss对象
        mine = VAELoss()
        # 计算VAE损失
        mine_loss = mine(X, X_recon, t_mean, t_log_var)
        # 计算损失函数关于输入的梯度
        dX_recon, dLogVar, dMean = mine.grad(X, X_recon, t_mean, t_log_var)
        # 从TorchVAELoss对象中提取梯度
        golds = TorchVAELoss().extract_grads(X, X_recon, t_mean, t_log_var)

        # 将损失和梯度存储在params列表中
        params = [
            (mine_loss, "loss"),
            (dX_recon, "dX_recon"),
            (dLogVar, "dt_log_var"),
            (dMean, "dt_mean"),
        ]
        # 打印当前试验的信息
        print("\nTrial {}".format(i))
        # 遍历params列表，进行梯度检验
        for ix, (mine, label) in enumerate(params):
            # 使用np.testing.assert_allclose函数检查梯度是否接近期望值
            np.testing.assert_allclose(
                mine,
                golds[label],
                err_msg=err_fmt(params, golds, ix),
                rtol=0.1,
                atol=1e-2,
            )
            # 打印通过梯度检验的信息
            print("\tPASSED {}".format(label))
        # 更新循环变量i
        i += 1
def test_WGAN_GP_loss(N=5):
    # 导入 WGAN_GPLoss 类
    from numpy_ml.neural_nets.losses import WGAN_GPLoss

    # 设置随机种子
    np.random.seed(12345)

    # 如果 N 为 None，则设置为无穷大
    N = np.inf if N is None else N

    # 初始化循环计数器 i
    i = 1
    # 循环执行直到 i 达到 N
    while i < N:
        # 生成 lambda_ 值
        lambda_ = np.random.randint(0, 10)
        # 生成样本数 n_ex
        n_ex = np.random.randint(1, 10)
        # 生成特征数 n_feats
        n_feats = np.random.randint(2, 10)
        # 生成真实样本 Y_real
        Y_real = random_tensor([n_ex], standardize=True)
        # 生成虚假样本 Y_fake
        Y_fake = random_tensor([n_ex], standardize=True)
        # 生成梯度插值 gradInterp
        gradInterp = random_tensor([n_ex, n_feats], standardize=True)

        # 创建 WGAN_GPLoss 实例
        mine = WGAN_GPLoss(lambda_=lambda_)
        # 计算 C_loss
        C_loss = mine(Y_fake, "C", Y_real, gradInterp)
        # 计算 G_loss
        G_loss = mine(Y_fake, "G")

        # 计算 C_loss 的梯度
        C_dY_fake, dY_real, dGradInterp = mine.grad(Y_fake, "C", Y_real, gradInterp)
        # 计算 G_loss 的梯度
        G_dY_fake = mine.grad(Y_fake, "G")

        # 提取 TorchWGANGPLoss 类的梯度
        golds = TorchWGANGPLoss(lambda_).extract_grads(Y_real, Y_fake, gradInterp)
        # 如果梯度中存在 NaN 值，则跳过当前循环
        if np.isnan(golds["C_dGradInterp"]).any():
            continue

        # 设置参数列表
        params = [
            (Y_real, "Y_real"),
            (Y_fake, "Y_fake"),
            (gradInterp, "gradInterp"),
            (C_loss, "C_loss"),
            (G_loss, "G_loss"),
            (-dY_real, "C_dY_real"),
            (-C_dY_fake, "C_dY_fake"),
            (dGradInterp, "C_dGradInterp"),
            (G_dY_fake, "G_dY_fake"),
        ]

        # 打印当前试验的信息
        print("\nTrial {}".format(i))
        # 遍历参数列表，进行断言比较
        for ix, (mine, label) in enumerate(params):
            np.testing.assert_allclose(
                mine,
                golds[label],
                err_msg=err_fmt(params, golds, ix),
                rtol=0.1,
                atol=1e-2,
            )
            print("\tPASSED {}".format(label))
        # 更新循环计数器 i
        i += 1


def test_NCELoss(N=1):
    # 导入 NCELoss 类和 DiscreteSampler 类
    from numpy_ml.neural_nets.losses import NCELoss
    from numpy_ml.utils.data_structures import DiscreteSampler

    # 设置随机种子
    np.random.seed(12345)

    # 如果 N 为 None，则设置为无穷大
    N = np.inf if N is None else N

    # 初始化循环计数器 i
    i = 1
#######################################################################
#                       Loss Function Gradients                       #
# 导入所需的模块和函数
def test_squared_error_grad(N=15):
    # 导入SquaredError损失函数和Tanh激活函数
    from numpy_ml.neural_nets.losses import SquaredError
    from numpy_ml.neural_nets.activations import Tanh

    # 设置随机种子
    np.random.seed(12345)

    # 如果N为None，则将N设置为无穷大
    N = np.inf if N is None else N

    # 创建SquaredError对象和torch_mse_grad对象
    mine = SquaredError()
    gold = torch_mse_grad
    act = Tanh()

    # 初始化循环计数器i
    i = 1
    # 当i小于N时执行循环
    while i < N:
        # 随机生成维度和样本数
        n_dims = np.random.randint(2, 100)
        n_examples = np.random.randint(1, 1000)
        y = random_tensor((n_examples, n_dims))

        # 生成随机输入
        z = random_tensor((n_examples, n_dims))
        y_pred = act.fn(z)

        # 断言SquaredError的梯度计算结果与torch_mse_grad的结果相近
        assert_almost_equal(
            mine.grad(y, y_pred, z, act), 0.5 * gold(y, z, torch.tanh), decimal=4
        )
        print("PASSED")
        i += 1


# 定义测试交叉熵梯度的函数
def test_cross_entropy_grad(N=15):
    # 导入CrossEntropy损失函数和Softmax层
    from numpy_ml.neural_nets.losses import CrossEntropy
    from numpy_ml.neural_nets.layers import Softmax

    # 设置随机种子
    np.random.seed(12345)

    # 如果N为None，则将N设置为无穷大
    N = np.inf if N is None else N

    # 创建CrossEntropy对象和torch_xe_grad对象
    mine = CrossEntropy()
    gold = torch_xe_grad
    sm = Softmax()

    # 初始化循环计数器i
    i = 1
    # 当i小于N时执行循环
    while i < N:
        # 随机生成类别数和样本数
        n_classes = np.random.randint(2, 100)
        n_examples = np.random.randint(1, 1000)

        y = random_one_hot_matrix(n_examples, n_classes)

        # cross_entropy_gradient返回相对于z的梯度（而不是softmax(z)）
        z = random_tensor((n_examples, n_classes))
        y_pred = sm.forward(z)

        # 断言CrossEntropy的梯度计算结果与torch_xe_grad的结果相近
        assert_almost_equal(mine.grad(y, y_pred), gold(y, z), decimal=5)
        print("PASSED")
        i += 1


#######################################################################
#                          Activations                                #
#######################################################################


# 定义测试Sigmoid激活函数的函数
def test_sigmoid_activation(N=15):
    # 导入Sigmoid激活函数
    from numpy_ml.neural_nets.activations import Sigmoid

    # 设置随机种子
    np.random.seed(12345)

    # 如果N为None，则将N设置为无穷大
    N = np.inf if N is None else N

    # 创建Sigmoid对象和expit函数
    mine = Sigmoid()
    gold = expit

    # 初始化循环计数器i
    i = 0
    # 当 i 小于 N 时执行循环
    while i < N:
        # 生成一个随机整数，表示张量的维度数量
        n_dims = np.random.randint(1, 100)
        # 生成一个随机张量，形状为 (1, n_dims)
        z = random_tensor((1, n_dims))
        # 断言自定义函数 mine.fn(z) 的输出与标准函数 gold(z) 的输出几乎相等
        assert_almost_equal(mine.fn(z), gold(z))
        # 打印 "PASSED" 表示测试通过
        print("PASSED")
        # i 自增
        i += 1
# 测试 ELU 激活函数的功能
def test_elu_activation(N=15):
    # 导入 ELU 激活函数
    from numpy_ml.neural_nets.activations import ELU

    # 设置随机种子
    np.random.seed(12345)

    # 如果 N 为 None，则将 N 设置为无穷大
    N = np.inf if N is None else N

    # 初始化计数器 i
    i = 0
    # 循环 N 次
    while i < N:
        # 生成随机维度
        n_dims = np.random.randint(1, 10)
        # 生成随机张量
        z = random_tensor((1, n_dims))

        # 生成随机 alpha 值
        alpha = np.random.uniform(0, 10)

        # 创建 ELU 激活函数对象
        mine = ELU(alpha)
        # 创建 PyTorch 中的 ELU 函数
        gold = lambda z, a: F.elu(torch.from_numpy(z), alpha).numpy()

        # 断言 ELU 函数的输出与 PyTorch 中的 ELU 函数的输出几乎相等
        assert_almost_equal(mine.fn(z), gold(z, alpha))
        # 打印 "PASSED"
        print("PASSED")
        # 更新计数器
        i += 1


# 测试 Softmax 激活函数的功能
def test_softmax_activation(N=15):
    # 导入 Softmax 层
    from numpy_ml.neural_nets.layers import Softmax

    # 设置随机种子
    np.random.seed(12345)

    # 如果 N 为 None，则将 N 设置为无穷大
    N = np.inf if N is None else N

    # 创建 Softmax 层对象
    mine = Softmax()
    # 创建 PyTorch 中的 Softmax 函数
    gold = lambda z: F.softmax(torch.FloatTensor(z), dim=1).numpy()

    # 初始化计数器 i
    i = 0
    # 循环 N 次
    while i < N:
        # 生成随机维度
        n_dims = np.random.randint(1, 100)
        # 生成随机概率矩阵
        z = random_stochastic_matrix(1, n_dims)
        # 断言 Softmax 函数的输出与 PyTorch 中的 Softmax 函数的输出几乎相等
        assert_almost_equal(mine.forward(z), gold(z))
        # 打印 "PASSED"
        print("PASSED")
        # 更新计数器
        i += 1


# 测试 ReLU 激活函数的功能
def test_relu_activation(N=15):
    # 导入 ReLU 激活函数
    from numpy_ml.neural_nets.activations import ReLU

    # 设置随机种子
    np.random.seed(12345)

    # 如果 N 为 None，则将 N 设置为无穷大
    N = np.inf if N is None else N

    # 创建 ReLU 激活函数对象
    mine = ReLU()
    # 创建 PyTorch 中的 ReLU 函数
    gold = lambda z: F.relu(torch.FloatTensor(z)).numpy()

    # 初始化计数器 i
    i = 0
    # 循环 N 次
    while i < N:
        # 生成随机维度
        n_dims = np.random.randint(1, 100)
        # 生成随机概率矩阵
        z = random_stochastic_matrix(1, n_dims)
        # 断言 ReLU 函数的输出与 PyTorch 中的 ReLU 函数的输出几乎相等
        assert_almost_equal(mine.fn(z), gold(z))
        # 打印 "PASSED"
        print("PASSED")
        # 更新计数器
        i += 1


# 测试 SoftPlus 激活函数的功能
def test_softplus_activation(N=15):
    # 导入 SoftPlus 激活函数
    from numpy_ml.neural_nets.activations import SoftPlus

    # 设置随机种子
    np.random.seed(12345)

    # 如果 N 为 None，则将 N 设置为无穷大
    N = np.inf if N is None else N

    # 创建 SoftPlus 激活函数对象
    mine = SoftPlus()
    # 创建 PyTorch 中的 SoftPlus 函数
    gold = lambda z: F.softplus(torch.FloatTensor(z)).numpy()

    # 初始化计数器 i
    i = 0
    # 循环 N 次
    while i < N:
        # 生成随机维度
        n_dims = np.random.randint(1, 100)
        # 生成随机概率矩阵
        z = random_stochastic_matrix(1, n_dims)
        # 断言 SoftPlus 函数的输出与 PyTorch 中的 SoftPlus 函数的输出几乎相等
        assert_almost_equal(mine.fn(z), gold(z))
        # 打印 "PASSED"
        print("PASSED")
        # 更新计数器
        i += 1


#######################################################################
#                      Activation Gradients                           #
# 导入所需的库和模块
def test_sigmoid_grad(N=15):
    # 从 numpy_ml.neural_nets.activations 模块中导入 Sigmoid 类
    from numpy_ml.neural_nets.activations import Sigmoid

    # 设置随机种子
    np.random.seed(12345)

    # 如果 N 为 None，则将 N 设置为无穷大
    N = np.inf if N is None else N

    # 创建 Sigmoid 实例 mine 和 torch 中的 sigmoid 梯度函数实例 gold
    mine = Sigmoid()
    gold = torch_gradient_generator(torch.sigmoid)

    # 初始化计数器 i
    i = 0
    # 循环执行 N 次
    while i < N:
        # 生成随机的样本数和维度
        n_ex = np.random.randint(1, 100)
        n_dims = np.random.randint(1, 100)
        # 生成随机的张量 z
        z = random_tensor((n_ex, n_dims))
        # 断言 Sigmoid 实例的梯度和 torch 中的 sigmoid 梯度函数的结果几乎相等
        assert_almost_equal(mine.grad(z), gold(z))
        # 打印 "PASSED"
        print("PASSED")
        # 更新计数器
        i += 1


# 类似上面的注释，以下函数 test_elu_grad, test_tanh_grad, test_relu_grad 的注释内容相同，只是激活函数不同
# 测试 Softmax 层的梯度计算
def test_softmax_grad(N=15):
    # 导入 Softmax 层和部分函数
    from numpy_ml.neural_nets.layers import Softmax
    from functools import partial

    # 设置随机种子
    np.random.seed(12345)

    # 如果 N 为 None，则设置为无穷大
    N = np.inf if N is None else N
    # 创建 Softmax 函数的偏函数
    p_soft = partial(F.softmax, dim=1)
    # 生成 Torch 梯度函数
    gold = torch_gradient_generator(p_soft)

    # 初始化计数器 i
    i = 0
    # 循环直到达到 N 次
    while i < N:
        # 创建 Softmax 层实例
        mine = Softmax()
        # 随机生成样本数和维度
        n_ex = np.random.randint(1, 3)
        n_dims = np.random.randint(1, 50)
        # 生成随机张量
        z = random_tensor((n_ex, n_dims), standardize=True)
        # 前向传播
        out = mine.forward(z)

        # 断言梯度计算结果准确性
        assert_almost_equal(
            gold(z),
            mine.backward(np.ones_like(out)),
            err_msg="Theirs:\n{}\n\nMine:\n{}\n".format(
                gold(z), mine.backward(np.ones_like(out))
            ),
            decimal=3,
        )
        # 打印测试通过信息
        print("PASSED")
        i += 1


# 测试 Softplus 层的梯度计算
def test_softplus_grad(N=15):
    # 导入 Softplus 层
    from numpy_ml.neural_nets.activations import SoftPlus

    # 设置随机种子
    np.random.seed(12345)

    # 如果 N 为 None，则设置为无穷大
    N = np.inf if N is None else N

    # 创建 Softplus 层实例
    mine = SoftPlus()
    # 生成 Torch 梯度函数
    gold = torch_gradient_generator(F.softplus)

    # 初始化计数器 i
    i = 0
    # 循环直到达到 N 次
    while i < N:
        # 随机生成样本数和维度
        n_ex = np.random.randint(1, 100)
        n_dims = np.random.randint(1, 100)
        # 生成随机张量
        z = random_tensor((n_ex, n_dims), standardize=True)
        # 断言梯度计算结果准确性
        assert_almost_equal(mine.grad(z), gold(z))
        # 打印测试通过信息
        print("PASSED")
        i += 1


#######################################################################
#                          Layers                                     #
#######################################################################


# 测试全连接层
def test_FullyConnected(N=15):
    # 导入全连接层和激活函数
    from numpy_ml.neural_nets.layers import FullyConnected
    from numpy_ml.neural_nets.activations import Tanh, ReLU, Sigmoid, Affine

    # 设置随机种子
    np.random.seed(12345)

    # 如果 N 为 None，则设置为无穷大
    N = np.inf if N is None else N

    # 定义激活函数列表
    acts = [
        (Tanh(), nn.Tanh(), "Tanh"),
        (Sigmoid(), nn.Sigmoid(), "Sigmoid"),
        (ReLU(), nn.ReLU(), "ReLU"),
        (Affine(), TorchLinearActivation(), "Affine"),
    ]

    # 初始化计数器 i
    i = 1
    # 当 i 小于 N + 1 时执行循环
    while i < N + 1:
        # 生成随机整数，作为外部神经元、内部神经元和输出神经元的数量
        n_ex = np.random.randint(1, 100)
        n_in = np.random.randint(1, 100)
        n_out = np.random.randint(1, 100)
        # 生成随机张量 X，形状为 (n_ex, n_in)，并进行标准化处理
        X = random_tensor((n_ex, n_in), standardize=True)

        # 随机选择一个激活函数
        act_fn, torch_fn, act_fn_name = acts[np.random.randint(0, len(acts))]

        # 初始化全连接层 L1，设置输出神经元数量和激活函数
        L1 = FullyConnected(n_out=n_out, act_fn=act_fn)

        # 前向传播
        y_pred = L1.forward(X)

        # 反向传播
        dLdy = np.ones_like(y_pred)
        dLdX = L1.backward(dLdy)

        # 获取标准梯度
        gold_mod = TorchFCLayer(n_in, n_out, torch_fn, L1.parameters)
        golds = gold_mod.extract_grads(X)

        # 定义参数列表，包括输入 X、预测值 y、权重 W、偏置 b、损失对预测值的梯度 dLdy、权重梯度 dLdW、偏置梯度 dLdB、输入梯度 dLdX
        params = [
            (L1.X[0], "X"),
            (y_pred, "y"),
            (L1.parameters["W"].T, "W"),
            (L1.parameters["b"], "b"),
            (dLdy, "dLdy"),
            (L1.gradients["W"].T, "dLdW"),
            (L1.gradients["b"], "dLdB"),
            (dLdX, "dLdX"),
        ]

        # 打印当前试验的信息和激活函数名称
        print("\nTrial {}\nact_fn={}".format(i, act_fn_name))
        # 遍历参数列表，逐个比较计算得到的梯度和标准梯度是否接近
        for ix, (mine, label) in enumerate(params):
            assert_almost_equal(
                mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=3
            )
            print("\tPASSED {}".format(label))
        # 更新试验次数
        i += 1
# 测试 Embedding 层的功能，包括前向传播和反向传播
def test_Embedding(N=15):
    # 从 numpy_ml.neural_nets.layers 导入 Embedding 模块
    from numpy_ml.neural_nets.layers import Embedding

    # 设置随机种子
    np.random.seed(12345)

    # 如果 N 为 None，则将 N 设置为无穷大
    N = np.inf if N is None else N

    # 初始化计数器 i
    i = 1
    # 当 i 小于 N + 1 时循环
    while i < N + 1:
        # 随机生成词汇表大小
        vocab_size = np.random.randint(1, 2000)
        # 随机生成示例数
        n_ex = np.random.randint(1, 100)
        # 随机生成输入维度
        n_in = np.random.randint(1, 100)
        # 随机生成嵌入维度
        emb_dim = np.random.randint(1, 100)

        # 随机生成输入数据 X
        X = np.random.randint(0, vocab_size, (n_ex, n_in))

        # 初始化 Embedding 层
        L1 = Embedding(n_out=emb_dim, vocab_size=vocab_size)

        # 前向传播
        y_pred = L1.forward(X)

        # 反向传播
        dLdy = np.ones_like(y_pred)
        L1.backward(dLdy)

        # 获取标准梯度
        gold_mod = TorchEmbeddingLayer(vocab_size, emb_dim, L1.parameters)
        golds = gold_mod.extract_grads(X)

        # 定义参数列表
        params = [
            (L1.X[0], "X"),
            (y_pred, "y"),
            (L1.parameters["W"], "W"),
            (dLdy, "dLdy"),
            (L1.gradients["W"], "dLdW"),
        ]

        # 打印测试结果
        print("\nTrial {}".format(i))
        for ix, (mine, label) in enumerate(params):
            assert_almost_equal(
                mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=3
            )
            print("\tPASSED {}".format(label))
        i += 1


# 测试 BatchNorm1D 层的功能
def test_BatchNorm1D(N=15):
    # 从 numpy_ml.neural_nets.layers 导入 BatchNorm1D 模块
    from numpy_ml.neural_nets.layers import BatchNorm1D

    # 设置随机种子
    np.random.seed(12345)

    # 如果 N 为 None，则将 N 设置为无穷大
    N = np.inf if N is None else N

    # 重新设置随机种子
    np.random.seed(12345)

    # 初始化计数器 i
    i = 1
    # 当 i 小于 N+1 时执行循环
    while i < N + 1:
        # 生成一个随机整数，范围在[2, 1000)
        n_ex = np.random.randint(2, 1000)
        # 生成一个随机整数，范围在[1, 1000)
        n_in = np.random.randint(1, 1000)
        # 生成一个随机的张量，形状为(n_ex, n_in)，并进行标准化处理
        X = random_tensor((n_ex, n_in), standardize=True)

        # 初始化 BatchNorm1D 层
        L1 = BatchNorm1D()

        # 前向传播
        y_pred = L1.forward(X)

        # 反向传播
        dLdy = np.ones_like(y_pred)
        dLdX = L1.backward(dLdy)

        # 获取标准梯度
        gold_mod = TorchBatchNormLayer(
            n_in, L1.parameters, "1D", epsilon=L1.epsilon, momentum=L1.momentum
        )
        golds = gold_mod.extract_grads(X)

        # 定义参数列表
        params = [
            (L1.X[0], "X"),
            (y_pred, "y"),
            (L1.parameters["scaler"].T, "scaler"),
            (L1.parameters["intercept"], "intercept"),
            (L1.parameters["running_mean"], "running_mean"),
            #  (L1.parameters["running_var"], "running_var"),
            (L1.gradients["scaler"], "dLdScaler"),
            (L1.gradients["intercept"], "dLdIntercept"),
            (dLdX, "dLdX"),
        ]

        # 打印当前试验的信息
        print("Trial {}".format(i))
        # 遍历参数列表，逐个进行梯度检验
        for ix, (mine, label) in enumerate(params):
            assert_almost_equal(
                mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=1
            )
            print("\tPASSED {}".format(label))
        # 更新试验次数
        i += 1
# 定义一个测试函数，用于测试 LayerNorm1D 层
def test_LayerNorm1D(N=15):
    # 从 numpy_ml.neural_nets.layers 模块导入 LayerNorm1D 类
    from numpy_ml.neural_nets.layers import LayerNorm1D

    # 如果 N 为 None，则将其设置为无穷大
    N = np.inf if N is None else N

    # 设置随机种子
    np.random.seed(12345)

    # 初始化循环计数器 i
    i = 1
    # 当 i 小于 N + 1 时循环执行以下代码块
    while i < N + 1:
        # 生成随机的样本数和输入特征数
        n_ex = np.random.randint(2, 1000)
        n_in = np.random.randint(1, 1000)
        # 生成随机的输入数据 X
        X = random_tensor((n_ex, n_in), standardize=True)

        # 初始化 LayerNorm1D 层
        L1 = LayerNorm1D()

        # 前向传播
        y_pred = L1.forward(X)

        # 反向传播
        dLdy = np.ones_like(y_pred)
        dLdX = L1.backward(dLdy)

        # 获取标准梯度
        gold_mod = TorchLayerNormLayer(n_in, L1.parameters, "1D", epsilon=L1.epsilon)
        golds = gold_mod.extract_grads(X)

        # 定义参数列表
        params = [
            (L1.X[0], "X"),
            (y_pred, "y"),
            (L1.parameters["scaler"].T, "scaler"),
            (L1.parameters["intercept"], "intercept"),
            (L1.gradients["scaler"], "dLdScaler"),
            (L1.gradients["intercept"], "dLdIntercept"),
            (dLdX, "dLdX"),
        ]

        # 打印当前试验的编号
        print("Trial {}".format(i))
        # 遍历参数列表，比较计算得到的梯度和标准梯度是否接近
        for ix, (mine, label) in enumerate(params):
            assert_almost_equal(
                mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=3
            )
            print("\tPASSED {}".format(label))
        # 更新循环计数器
        i += 1


# 定义一个测试函数，用于测试 LayerNorm2D 层
def test_LayerNorm2D(N=15):
    # 从 numpy_ml.neural_nets.layers 模块导入 LayerNorm2D 类
    from numpy_ml.neural_nets.layers import LayerNorm2D

    # 如果 N 为 None，则将其设置为无穷大
    N = np.inf if N is None else N

    # 设置随机种子
    np.random.seed(12345)

    # 初始化循环计数器 i
    i = 1
    # 当 i 小于 N + 1 时循环执行以下代码块
    while i < N + 1:
        # 生成一个随机整数，范围在 [2, 10)
        n_ex = np.random.randint(2, 10)
        # 生成一个随机整数，范围在 [1, 10)
        in_rows = np.random.randint(1, 10)
        # 生成一个随机整数，范围在 [1, 10)
        in_cols = np.random.randint(1, 10)
        # 生成一个随机整数，范围在 [1, 3)
        n_in = np.random.randint(1, 3)

        # 初始化 LayerNorm2D 层
        X = random_tensor((n_ex, in_rows, in_cols, n_in), standardize=True)
        L1 = LayerNorm2D()

        # 前向传播
        y_pred = L1.forward(X)

        # 标准损失函数
        dLdy = np.ones_like(X)
        dLdX = L1.backward(dLdy)

        # 获取标准梯度
        gold_mod = TorchLayerNormLayer(
            [n_in, in_rows, in_cols], L1.parameters, mode="2D", epsilon=L1.epsilon
        )
        golds = gold_mod.extract_grads(X, Y_true=None)

        # 定义参数列表
        params = [
            (L1.X[0], "X"),
            (L1.hyperparameters["epsilon"], "epsilon"),
            (L1.parameters["scaler"], "scaler"),
            (L1.parameters["intercept"], "intercept"),
            (y_pred, "y"),
            (L1.gradients["scaler"], "dLdScaler"),
            (L1.gradients["intercept"], "dLdIntercept"),
            (dLdX, "dLdX"),
        ]

        # 打印当前试验的信息
        print("Trial {}".format(i))
        # 遍历参数列表，逐个进行断言比较
        for ix, (mine, label) in enumerate(params):
            assert_almost_equal(
                mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=3
            )

            # 打印通过断言的参数信息
            print("\tPASSED {}".format(label))

        # 更新循环变量 i
        i += 1
# 定义一个测试 MultiplyLayer 的函数，可以指定测试次数 N，默认为 15
def test_MultiplyLayer(N=15):
    # 导入所需的模块和类
    from numpy_ml.neural_nets.layers import Multiply
    from numpy_ml.neural_nets.activations import Tanh, ReLU, Sigmoid, Affine

    # 如果 N 为 None，则将 N 设置为无穷大
    N = np.inf if N is None else N

    # 设置随机种子
    np.random.seed(12345)

    # 定义激活函数列表
    acts = [
        (Tanh(), nn.Tanh(), "Tanh"),
        (Sigmoid(), nn.Sigmoid(), "Sigmoid"),
        (ReLU(), nn.ReLU(), "ReLU"),
        (Affine(), TorchLinearActivation(), "Affine"),
    ]

    # 初始化计数器 i
    i = 1
    # 循环进行 N 次测试
    while i < N + 1:
        # 初始化输入数据列表 Xs
        Xs = []
        # 生成随机数，作为样本数和输入维度
        n_ex = np.random.randint(1, 100)
        n_in = np.random.randint(1, 100)
        n_entries = np.random.randint(2, 5)
        # 生成 n_entries 个随机张量，加入 Xs 列表
        for _ in range(n_entries):
            Xs.append(random_tensor((n_ex, n_in), standardize=True))

        # 随机选择一个激活函数
        act_fn, torch_fn, act_fn_name = acts[np.random.randint(0, len(acts))]

        # 初始化 Multiply 层
        L1 = Multiply(act_fn)

        # 前向传播
        y_pred = L1.forward(Xs)

        # 反向传播
        dLdy = np.ones_like(y_pred)
        dLdXs = L1.backward(dLdy)

        # 获取标准梯度
        gold_mod = TorchMultiplyLayer(torch_fn)
        golds = gold_mod.extract_grads(Xs)

        # 构建参数列表
        params = [(Xs, "Xs"), (y_pred, "Y")]
        params.extend(
            [(dldxi, "dLdX{}".format(i + 1)) for i, dldxi in enumerate(dLdXs)]
        )

        # 打印测试结果
        print("\nTrial {}".format(i))
        print("n_ex={}, n_in={}".format(n_ex, n_in))
        print("n_entries={}, act_fn={}".format(n_entries, str(act_fn)))
        for ix, (mine, label) in enumerate(params):
            # 断言近似相等
            assert_almost_equal(
                mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=1
            )
            print("\tPASSED {}".format(label))
        i += 1


# 定义一个测试 AddLayer 的函数，可以指定测试次数 N，默认为 15
def test_AddLayer(N=15):
    # 导入所需的模块和类
    from numpy_ml.neural_nets.layers import Add
    from numpy_ml.neural_nets.activations import Tanh, ReLU, Sigmoid, Affine

    # 如果 N 为 None，则将 N 设置为无穷大
    N = np.inf if N is None else N

    # 设置随机种子
    np.random.seed(12345)
    # 定义不同激活函数的元组列表
    acts = [
        (Tanh(), nn.Tanh(), "Tanh"),
        (Sigmoid(), nn.Sigmoid(), "Sigmoid"),
        (ReLU(), nn.ReLU(), "ReLU"),
        (Affine(), TorchLinearActivation(), "Affine"),
    ]

    # 初始化循环计数器
    i = 1
    # 循环执行 N 次
    while i < N + 1:
        # 初始化输入数据列表
        Xs = []
        # 生成随机的样本数和输入维度
        n_ex = np.random.randint(1, 100)
        n_in = np.random.randint(1, 100)
        # 生成随机的输入数据条目数
        n_entries = np.random.randint(2, 5)
        # 生成随机的输入数据并添加到 Xs 列表中
        for _ in range(n_entries):
            Xs.append(random_tensor((n_ex, n_in), standardize=True))

        # 随机选择激活函数
        act_fn, torch_fn, act_fn_name = acts[np.random.randint(0, len(acts))]

        # 初始化 Add 层
        L1 = Add(act_fn)

        # 前向传播
        y_pred = L1.forward(Xs)

        # 反向传播
        dLdy = np.ones_like(y_pred)
        dLdXs = L1.backward(dLdy)

        # 获取标准梯度
        gold_mod = TorchAddLayer(torch_fn)
        golds = gold_mod.extract_grads(Xs)

        # 构建参数列表
        params = [(Xs, "Xs"), (y_pred, "Y")]
        params.extend(
            [(dldxi, "dLdX{}".format(i + 1)) for i, dldxi in enumerate(dLdXs)]
        )

        # 打印当前试验信息
        print("\nTrial {}".format(i))
        print("n_ex={}, n_in={}".format(n_ex, n_in))
        print("n_entries={}, act_fn={}".format(n_entries, str(act_fn)))
        # 遍历参数列表，进行梯度检验
        for ix, (mine, label) in enumerate(params):
            assert_almost_equal(
                mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=1
            )
            print("\tPASSED {}".format(label))
        # 更新循环计数器
        i += 1
# 定义测试 BatchNorm2D 的函数，参数 N 为测试次数，默认为 15
def test_BatchNorm2D(N=15):
    # 导入 BatchNorm2D 模块
    from numpy_ml.neural_nets.layers import BatchNorm2D

    # 如果 N 为 None，则将其设为无穷大
    N = np.inf if N is None else N

    # 设置随机种子
    np.random.seed(12345)

    # 初始化循环计数器 i
    i = 1
    # 循环执行 N 次测试
    while i < N + 1:
        # 生成随机的样本数、输入行数、输入列数和输入通道数
        n_ex = np.random.randint(2, 10)
        in_rows = np.random.randint(1, 10)
        in_cols = np.random.randint(1, 10)
        n_in = np.random.randint(1, 3)

        # 初始化 BatchNorm2D 层
        X = random_tensor((n_ex, in_rows, in_cols, n_in), standardize=True)
        L1 = BatchNorm2D()

        # 前向传播
        y_pred = L1.forward(X)

        # 标准损失函数
        dLdy = np.ones_like(X)
        dLdX = L1.backward(dLdy)

        # 获取标准梯度
        gold_mod = TorchBatchNormLayer(
            n_in, L1.parameters, mode="2D", epsilon=L1.epsilon, momentum=L1.momentum
        )
        golds = gold_mod.extract_grads(X, Y_true=None)

        # 定义参数列表
        params = [
            (L1.X[0], "X"),
            (L1.hyperparameters["momentum"], "momentum"),
            (L1.hyperparameters["epsilon"], "epsilon"),
            (L1.parameters["scaler"].T, "scaler"),
            (L1.parameters["intercept"], "intercept"),
            (L1.parameters["running_mean"], "running_mean"),
            #  (L1.parameters["running_var"], "running_var"),
            (y_pred, "y"),
            (L1.gradients["scaler"], "dLdScaler"),
            (L1.gradients["intercept"], "dLdIntercept"),
            (dLdX, "dLdX"),
        ]

        # 打印当前测试的序号
        print("Trial {}".format(i))
        # 遍历参数列表，逐个进行断言比较
        for ix, (mine, label) in enumerate(params):
            assert_almost_equal(
                mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=3
            )

            print("\tPASSED {}".format(label))

        # 更新循环计数器
        i += 1


# 定义测试 RNNCell 的函数，参数 N 为测试次数，默认为 15
def test_RNNCell(N=15):
    # 导入 RNNCell 模块
    from numpy_ml.neural_nets.layers import RNNCell

    # 如果 N 为 None，则将其设为无穷大
    N = np.inf if N is None else N

    # 设置随机种子
    np.random.seed(12345)

    # 初始化循环计数器 i
    i = 1
    # 循环执行 N 次
    while i < N + 1:
        # 生成随机数，表示外部输入、内部状态、输出、时间步数
        n_ex = np.random.randint(1, 10)
        n_in = np.random.randint(1, 10)
        n_out = np.random.randint(1, 10)
        n_t = np.random.randint(1, 10)
        # 生成随机张量 X
        X = random_tensor((n_ex, n_in, n_t), standardize=True)

        # 初始化 RNN 层
        L1 = RNNCell(n_out=n_out)

        # 前向传播
        y_preds = []
        for t in range(n_t):
            y_pred = L1.forward(X[:, :, t])
            y_preds += [y_pred]

        # 反向传播
        dLdX = []
        dLdAt = np.ones_like(y_preds[t])
        for t in reversed(range(n_t)):
            dLdXt = L1.backward(dLdAt)
            dLdX.insert(0, dLdXt)
        dLdX = np.dstack(dLdX)

        # 获取标准梯度
        gold_mod = TorchRNNCell(n_in, n_out, L1.parameters)
        golds = gold_mod.extract_grads(X)

        # 定义参数列表
        params = [
            (X, "X"),
            (np.array(y_preds), "y"),
            (L1.parameters["ba"].T, "ba"),
            (L1.parameters["bx"].T, "bx"),
            (L1.parameters["Wax"].T, "Wax"),
            (L1.parameters["Waa"].T, "Waa"),
            (L1.gradients["ba"].T, "dLdBa"),
            (L1.gradients["bx"].T, "dLdBx"),
            (L1.gradients["Wax"].T, "dLdWax"),
            (L1.gradients["Waa"].T, "dLdWaa"),
            (dLdX, "dLdX"),
        ]

        # 打印当前试验次数
        print("Trial {}".format(i))
        # 遍历参数列表，进行梯度检验
        for ix, (mine, label) in enumerate(params):
            np.testing.assert_allclose(
                mine,
                golds[label],
                err_msg=err_fmt(params, golds, ix),
                atol=1e-3,
                rtol=1e-3,
            )
            print("\tPASSED {}".format(label))
        i += 1
# 定义一个测试函数，用于测试 Conv2D 层的功能
def test_Conv2D(N=15):
    # 从相应的模块中导入需要的类
    from numpy_ml.neural_nets.layers import Conv2D
    from numpy_ml.neural_nets.activations import Tanh, ReLU, Sigmoid, Affine

    # 如果 N 为 None，则将其设置为无穷大
    N = np.inf if N is None else N

    # 设置随机种子
    np.random.seed(12345)

    # 定义激活函数列表
    acts = [
        (Tanh(), nn.Tanh(), "Tanh"),
        (Sigmoid(), nn.Sigmoid(), "Sigmoid"),
        (ReLU(), nn.ReLU(), "ReLU"),
        (Affine(), TorchLinearActivation(), "Affine"),
    ]

    # 初始化计数器 i
    i = 1

# 定义一个测试函数，用于测试 DPAttention 层的功能
def test_DPAttention(N=15):
    # 从相应的模块中导入 DotProductAttention 类
    from numpy_ml.neural_nets.layers import DotProductAttention

    # 如果 N 为 None，则将其设置为无穷大
    N = np.inf if N is None else N

    # 设置随机种子
    np.random.seed(12345)

    # 初始化计数器 i
    i = 1
    # 当 i 小于 N+1 时循环
    while i < N + 1:
        # 生成随机数
        n_ex = np.random.randint(1, 10)
        d_k = np.random.randint(1, 100)
        d_v = np.random.randint(1, 100)

        # 生成随机张量 Q, K, V
        Q = random_tensor((n_ex, d_k), standardize=True)
        K = random_tensor((n_ex, d_k), standardize=True)
        V = random_tensor((n_ex, d_v), standardize=True)

        # 初始化 DotProductAttention 层
        mine = DotProductAttention(scale=True, dropout_p=0)

        # 前向传播
        y_pred = mine.forward(Q, K, V)

        # 反向传播
        dLdy = np.ones_like(y_pred)
        dLdQ, dLdK, dLdV = mine.backward(dLdy)

        # 获取标准梯度
        gold_mod = TorchSDPAttentionLayer()
        golds = gold_mod.extract_grads(Q, K, V)

        # 定义参数列表
        params = [
            (mine.X[0][0], "Q"),
            (mine.X[0][1], "K"),
            (mine.X[0][2], "V"),
            (y_pred, "Y"),
            (dLdV, "dLdV"),
            (dLdK, "dLdK"),
            (dLdQ, "dLdQ"),
        ]

        # 打印测试结果
        print("\nTrial {}".format(i))
        print("n_ex={} d_k={} d_v={}".format(n_ex, d_k, d_v))
        for ix, (mine, label) in enumerate(params):
            assert_almost_equal(
                mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=4
            )
            print("\tPASSED {}".format(label))
        i += 1

# 定义一个测试函数，用于测试 Conv1D 层的功能
def test_Conv1D(N=15):
    # 从相应的模块中导入 Conv1D 类
    from numpy_ml.neural_nets.layers import Conv1D
    # 从指定路径导入 Tanh、ReLU、Sigmoid 和 Affine 激活函数以及 Affine 层
    from numpy_ml.neural_nets.activations import Tanh, ReLU, Sigmoid, Affine
    
    # 如果 N 为 None，则将 N 设置为正无穷
    N = np.inf if N is None else N
    
    # 设置随机种子为 12345
    np.random.seed(12345)
    
    # 定义激活函数列表，每个元素包含自定义的激活函数对象、PyTorch 中对应的激活函数对象和激活函数名称
    acts = [
        (Tanh(), nn.Tanh(), "Tanh"),
        (Sigmoid(), nn.Sigmoid(), "Sigmoid"),
        (ReLU(), nn.ReLU(), "ReLU"),
        (Affine(), TorchLinearActivation(), "Affine"),
    ]
    
    # 初始化计数器 i 为 1
    i = 1
    # 循环执行 N 次
    while i < N + 1:
        # 生成随机整数，表示例外数量
        n_ex = np.random.randint(1, 10)
        # 生成随机整数，表示输入序列长度
        l_in = np.random.randint(1, 10)
        # 生成随机整数，表示输入输出通道数
        n_in, n_out = np.random.randint(1, 3), np.random.randint(1, 3)
        # 生成随机整数，表示卷积核宽度
        f_width = min(l_in, np.random.randint(1, 5))
        # 生成随机整数，表示填充和步长
        p, s = np.random.randint(0, 5), np.random.randint(1, 3)
        # 生成随机整数，表示膨胀率
        d = np.random.randint(0, 5)

        # 计算卷积核的参数数量
        fc = f_width * (d + 1) - d
        # 计算输出序列长度
        l_out = int(1 + (l_in + 2 * p - fc) / s)

        # 如果输出序列长度小于等于0，则跳过本次循环
        if l_out <= 0:
            continue

        # 生成随机张量作为输入数据
        X = random_tensor((n_ex, l_in, n_in), standardize=True)

        # 随机选择激活函数
        act_fn, torch_fn, act_fn_name = acts[np.random.randint(0, len(acts))]

        # 初始化一维卷积层
        L1 = Conv1D(
            out_ch=n_out,
            kernel_width=f_width,
            act_fn=act_fn,
            pad=p,
            stride=s,
            dilation=d,
        )

        # 前向传播
        y_pred = L1.forward(X)

        # 反向传播
        dLdy = np.ones_like(y_pred)
        dLdX = L1.backward(dLdy)

        # 获取标准梯度
        gold_mod = TorchConv1DLayer(
            n_in, n_out, torch_fn, L1.parameters, L1.hyperparameters
        )
        golds = gold_mod.extract_grads(X)

        # 定义参数列表
        params = [
            (L1.X[0], "X"),
            (y_pred, "y"),
            (L1.parameters["W"], "W"),
            (L1.parameters["b"], "b"),
            (L1.gradients["W"], "dLdW"),
            (L1.gradients["b"], "dLdB"),
            (dLdX, "dLdX"),
        ]

        # 打印当前试验信息
        print("\nTrial {}".format(i))
        print("pad={}, stride={}, f_width={}, n_ex={}".format(p, s, f_width, n_ex))
        print("l_in={}, n_in={}".format(l_in, n_in))
        print("l_out={}, n_out={}".format(l_out, n_out))
        print("dilation={}".format(d))
        # 遍历参数列表，检查梯度是否正确
        for ix, (mine, label) in enumerate(params):
            assert_almost_equal(
                mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=4
            )
            print("\tPASSED {}".format(label))
        i += 1
# 定义用于测试 Deconv2D 层的函数，N 默认为 15
def test_Deconv2D(N=15):
    # 导入必要的模块和类
    from numpy_ml.neural_nets.layers import Deconv2D
    from numpy_ml.neural_nets.activations import Tanh, ReLU, Sigmoid, Affine

    # 如果 N 为 None，则将 N 设置为无穷大
    N = np.inf if N is None else N

    # 设置随机种子
    np.random.seed(12345)

    # 定义激活函数列表，每个元素包含激活函数对象、对应的 Torch 模块、激活函数名称
    acts = [
        (Tanh(), nn.Tanh(), "Tanh"),
        (Sigmoid(), nn.Sigmoid(), "Sigmoid"),
        (ReLU(), nn.ReLU(), "ReLU"),
        (Affine(), TorchLinearActivation(), "Affine"),
    ]

    # 初始化计数器
    i = 1

# 定义用于测试 Pool2D 层的函数，N 默认为 15
def test_Pool2D(N=15):
    # 导入必要的模块和类
    from numpy_ml.neural_nets.layers import Pool2D

    # 如果 N 为 None，则将 N 设置为无穷大
    N = np.inf if N is None else N

    # 设置随机种子
    np.random.seed(12345)

    # 初始化计数器
    i = 1
    # 循环执行直到 i 大于 N
    while i < N + 1:
        # 生成随机整数，表示样本数
        n_ex = np.random.randint(1, 10)
        # 生成随机整数，表示输入数据的行数
        in_rows = np.random.randint(1, 10)
        # 生成随机整数，表示输入数据的列数
        in_cols = np.random.randint(1, 10)
        # 生成随机整数，表示输入数据的通道数
        n_in = np.random.randint(1, 3)
        # 生成随机的过滤器形状
        f_shape = (
            min(in_rows, np.random.randint(1, 5)),
            min(in_cols, np.random.randint(1, 5)),
        )
        # 生成随机的填充值和步长
        p, s = np.random.randint(0, max(1, min(f_shape) // 2)), np.random.randint(1, 3)
        # 设置池化层的模式为"average"
        mode = "average"
        # 计算输出数据的行数和列数
        out_rows = int(1 + (in_rows + 2 * p - f_shape[0]) / s)
        out_cols = int(1 + (in_cols + 2 * p - f_shape[1]) / s)

        # 生成随机输入数据
        X = random_tensor((n_ex, in_rows, in_cols, n_in), standardize=True)
        print("\nmode: {}".format(mode))
        print("pad={}, stride={}, f_shape={}, n_ex={}".format(p, s, f_shape, n_ex))
        print("in_rows={}, in_cols={}, n_in={}".format(in_rows, in_cols, n_in))
        print("out_rows={}, out_cols={}, n_out={}".format(out_rows, out_cols, n_in))

        # 初始化 Pool2D 层
        L1 = Pool2D(kernel_shape=f_shape, pad=p, stride=s, mode=mode)

        # 前向传播
        y_pred = L1.forward(X)

        # 反向传播
        dLdy = np.ones_like(y_pred)
        dLdX = L1.backward(dLdy)

        # 获取标准梯度
        gold_mod = TorchPool2DLayer(n_in, L1.hyperparameters)
        golds = gold_mod.extract_grads(X)

        # 检查梯度是否正确
        params = [(L1.X[0], "X"), (y_pred, "y"), (dLdX, "dLdX")]
        for ix, (mine, label) in enumerate(params):
            assert_almost_equal(
                mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=4
            )
            print("\tPASSED {}".format(label))
        i += 1
# 定义一个测试 LSTMCell 的函数，N 默认为 15
def test_LSTMCell(N=15):
    # 导入 LSTMCell 模块
    from numpy_ml.neural_nets.layers import LSTMCell

    # 如果 N 为 None，则将 N 设置为无穷大
    N = np.inf if N is None else N

    # 设置随机种子为 12345
    np.random.seed(12345)

    # 初始化变量 i 为 1
    i = 1

# 手动计算 vanilla RNN 参数的梯度
def grad_check_RNN(model, loss_func, param_name, n_t, X, epsilon=1e-7):
    """
    Manual gradient calc for vanilla RNN parameters
    """
    # 如果参数名为 "Ba" 或 "Bx"，则将参数名转换为小写
    if param_name in ["Ba", "Bx"]:
        param_name = param_name.lower()
    # 如果参数名为 "X" 或 "y"，则返回 None
    elif param_name in ["X", "y"]:
        return None

    # 复制原始参数，并初始化梯度为与原始参数相同形状的零矩阵
    param_orig = model.parameters[param_name]
    model.flush_gradients()
    grads = np.zeros_like(param_orig)

    # 遍历参数的每个元素
    for flat_ix, val in enumerate(param_orig.flat):
        param = deepcopy(param_orig)
        md_ix = np.unravel_index(flat_ix, param.shape)

        # 正向计算
        y_preds_plus = []
        param[md_ix] = val + epsilon
        model.parameters[param_name] = param
        for t in range(n_t):
            y_pred_plus = model.forward(X[:, :, t])
            y_preds_plus += [y_pred_plus]
        loss_plus = loss_func(y_preds_plus)
        model.flush_gradients()

        # 反向计算
        y_preds_minus = []
        param[md_ix] = val - epsilon
        model.parameters[param_name] = param
        for t in range(n_t):
            y_pred_minus = model.forward(X[:, :, t])
            y_preds_minus += [y_pred_minus]
        loss_minus = loss_func(y_preds_minus)
        model.flush_gradients()

        # 计算梯度
        grad = (loss_plus - loss_minus) / (2 * epsilon)
        grads[md_ix] = grad
    return grads.T

# 定义一个测试 MultiHeadedAttentionModule 的函数，N 默认为 15
def test_MultiHeadedAttentionModule(N=15):
    # 导入 MultiHeadedAttentionModule 模块
    from numpy_ml.neural_nets.modules import MultiHeadedAttentionModule

    # 如果 N 为 None，则将 N 设置为无穷大
    N = np.inf if N is None else N
    # 设置随机种子为 12345
    np.random.seed(12345)

    # 初始化变量 i 为 1
    i = 1

# 定义一个测试 SkipConnectionIdentityModule 的函数，N 默认为 15
def test_SkipConnectionIdentityModule(N=15):
    # 导入 SkipConnectionIdentityModule 模块
    from numpy_ml.neural_nets.modules import SkipConnectionIdentityModule
    # 从指定路径导入 Tanh、ReLU、Sigmoid 和 Affine 激活函数以及 Affine 层
    from numpy_ml.neural_nets.activations import Tanh, ReLU, Sigmoid, Affine
    
    # 如果 N 为 None，则将 N 设置为正无穷
    N = np.inf if N is None else N
    
    # 设置随机种子为 12345
    np.random.seed(12345)
    
    # 定义激活函数列表，每个元素包含自定义的激活函数对象、PyTorch 中对应的激活函数对象和激活函数名称
    acts = [
        (Tanh(), nn.Tanh(), "Tanh"),
        (Sigmoid(), nn.Sigmoid(), "Sigmoid"),
        (ReLU(), nn.ReLU(), "ReLU"),
        (Affine(), TorchLinearActivation(), "Affine"),
    ]
    
    # 初始化计数器 i 为 1
    i = 1
# 测试 SkipConnectionConvModule 模块
def test_SkipConnectionConvModule(N=15):
    # 导入需要的模块和激活函数
    from numpy_ml.neural_nets.modules import SkipConnectionConvModule
    from numpy_ml.neural_nets.activations import Tanh, ReLU, Sigmoid, Affine

    # 如果 N 为 None，则设置为无穷大
    N = np.inf if N is None else N

    # 设置随机种子
    np.random.seed(12345)

    # 定义激活函数列表
    acts = [
        (Tanh(), nn.Tanh(), "Tanh"),
        (Sigmoid(), nn.Sigmoid(), "Sigmoid"),
        (ReLU(), nn.ReLU(), "ReLU"),
        (Affine(), TorchLinearActivation(), "Affine"),
    ]

    # 初始化计数器
    i = 1

# 测试 BidirectionalLSTM 模块
def test_BidirectionalLSTM(N=15):
    # 导入需要的模块
    from numpy_ml.neural_nets.modules import BidirectionalLSTM

    # 如果 N 为 None，则设置为无穷大
    N = np.inf if N is None else N

    # 设置随机种子
    np.random.seed(12345)

    # 初始化计数器
    i = 1

# 测试 WaveNetModule 模块
def test_WaveNetModule(N=10):
    # 导入需要的模块
    from numpy_ml.neural_nets.modules import WavenetResidualModule

    # 如果 N 为 None，则设置为无穷大
    N = np.inf if N is None else N

    # 设置随机种子
    np.random.seed(12345)

    # 初始化计数器
    i = 1

#######################################################################
#                                Utils                                #
#######################################################################

# 测试 pad1D 函数
def test_pad1D(N=15):
    # 导入需要的模块
    from numpy_ml.neural_nets.layers import Conv1D
    from .nn_torch_models import TorchCausalConv1d, torchify

    # 设置随机种子
    np.random.seed(12345)

    # 如果 N 为 None，则设置为无穷大
    N = np.inf if N is None else N

    # 初始化计数器
    i = 1

# 测试 conv 函数
def test_conv(N=15):
    # 设置随机种子
    np.random.seed(12345)
    # 如果 N 为 None，则设置为无穷大
    N = np.inf if N is None else N
    # 初始化计数器
    i = 0
    # 当 i 小于 N 时，执行以下循环
    while i < N:
        # 生成一个介于 2 到 15 之间的随机整数，作为例子数量
        n_ex = np.random.randint(2, 15)
        # 生成一个介于 2 到 15 之间的随机整数，作为输入数据的行数
        in_rows = np.random.randint(2, 15)
        # 生成一个介于 2 到 15 之间的随机整数，作为输入数据的列数
        in_cols = np.random.randint(2, 15)
        # 生成一个介于 2 到 15 之间的随机整数，作为输入数据的通道数
        in_ch = np.random.randint(2, 15)
        # 生成一个介于 2 到 15 之间的随机整数，作为输出数据的通道数
        out_ch = np.random.randint(2, 15)
        # 生成一个随机的形状元组，元组中的元素为输入数据的行数和列数
        f_shape = (
            min(in_rows, np.random.randint(2, 10)),
            min(in_cols, np.random.randint(2, 10)),
        )
        # 生成一个介于 1 到 3 之间的随机整数，作为卷积步长
        s = np.random.randint(1, 3)
        # 生成一个介于 0 到 5 之间的随机整数，作为填充大小
        p = np.random.randint(0, 5)

        # 生成一个随机的输入数据张量
        X = np.random.rand(n_ex, in_rows, in_cols, in_ch)
        # 对输入数据进行二维填充
        X_pad, p = pad2D(X, p)
        # 生成一个随机的权重张量
        W = np.random.randn(f_shape[0], f_shape[1], in_ch, out_ch)

        # 使用朴素的方法进行二维卷积操作，得到期望的输出
        gold = conv2D_naive(X, W, s, p)
        # 使用优化的方法进行二维卷积操作，得到实际的输出
        mine = conv2D(X, W, s, p)

        # 检查实际输出和期望输出是否几乎相等
        np.testing.assert_almost_equal(mine, gold)
        # 打印“PASSED”表示测试通过
        print("PASSED")
        # 更新循环变量 i
        i += 1
# 模型部分

# 定义训练 Variational Autoencoder (VAE) 模型的函数
def fit_VAE():
    # 导入所需的库和模块
    # 用于测试
    import tensorflow.keras.datasets.mnist as mnist
    from numpy_ml.neural_nets.models.vae import BernoulliVAE

    # 设置随机种子
    np.random.seed(12345)

    # 加载 MNIST 数据集
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # 将像素强度缩放到 [0, 1] 范围内
    X_train = np.expand_dims(X_train.astype("float32") / 255.0, 3)
    X_test = np.expand_dims(X_test.astype("float32") / 255.0, 3)

    # 只使用前 128 * 1 个样本作为一个 batch
    X_train = X_train[: 128 * 1]

    # 创建 BernoulliVAE 实例
    BV = BernoulliVAE()
    # 训练 VAE 模型
    BV.fit(X_train, n_epochs=1, verbose=False)


# 定义测试 Wasserstein GAN with Gradient Penalty (WGAN-GP) 模型的函数
def test_WGAN_GP(N=1):
    # 导入 WGAN-GP 模型
    from numpy_ml.neural_nets.models.wgan_gp import WGAN_GP

    # 设置随机种子
    np.random.seed(12345)

    # 生成一个随机种子
    ss = np.random.randint(0, 1000)
    np.random.seed(ss)

    # 如果 N 为 None，则设置为无穷大
    N = np.inf if N is None else N

    # 初始化 i 为 1
    i = 1
```