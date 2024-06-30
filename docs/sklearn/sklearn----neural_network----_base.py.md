# `D:\src\scipysrc\scikit-learn\sklearn\neural_network\_base.py`

```
# 引入 NumPy 库，简化数学运算和数组操作
import numpy as np
# 从 SciPy 库中导入 logistic sigmoid 函数，用于计算 logistic 函数
from scipy.special import expit as logistic_sigmoid
# 从 SciPy 库中导入 xlogy 函数，未使用到
from scipy.special import xlogy

# 定义一个函数，什么也不做，直接返回输入的数组
def inplace_identity(X):
    """Simply leave the input array unchanged.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Data, where `n_samples` is the number of samples
        and `n_features` is the number of features.
    """
    # Nothing to do

# 定义一个函数，计算 logistic 函数，并用计算结果替换输入的数组
def inplace_logistic(X):
    """Compute the logistic function inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    """
    logistic_sigmoid(X, out=X)

# 定义一个函数，计算双曲正切函数，并用计算结果替换输入的数组
def inplace_tanh(X):
    """Compute the hyperbolic tan function inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    """
    np.tanh(X, out=X)

# 定义一个函数，计算 ReLU 函数，并用计算结果替换输入的数组
def inplace_relu(X):
    """Compute the rectified linear unit function inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    """
    np.maximum(X, 0, out=X)

# 定义一个函数，计算 softmax 函数，并用计算结果替换输入的数组
def inplace_softmax(X):
    """Compute the K-way softmax function inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    """
    # 将每个样本的值减去其最大值，防止数值溢出
    tmp = X - X.max(axis=1)[:, np.newaxis]
    # 计算指数
    np.exp(tmp, out=X)
    # 将结果归一化为概率分布
    X /= X.sum(axis=1)[:, np.newaxis]

# 定义一个字典，将激活函数的名称映射到对应的函数对象
ACTIVATIONS = {
    "identity": inplace_identity,
    "tanh": inplace_tanh,
    "logistic": inplace_logistic,
    "relu": inplace_relu,
    "softmax": inplace_softmax,
}

# 定义一个函数，应用 identity 函数的导数，什么也不做
def inplace_identity_derivative(Z, delta):
    """Apply the derivative of the identity function: do nothing.

    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the identity activation function during
        the forward pass.

    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    # Nothing to do

# 定义一个函数，应用 logistic 函数的导数
def inplace_logistic_derivative(Z, delta):
    """Apply the derivative of the logistic sigmoid function.

    It exploits the fact that the derivative is a simple function of the output
    value from logistic function.

    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the logistic activation function during
        the forward pass.

    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    # 计算 logistic 函数的导数，将其应用到误差信号上
    delta *= Z
    delta *= 1 - Z

# 定义一个函数，应用 tanh 函数的导数
def inplace_tanh_derivative(Z, delta):
    """Apply the derivative of the hyperbolic tanh function.

    It exploits the fact that the derivative is a simple function of the output
    value from hyperbolic tangent.

    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the tanh activation function during
        the forward pass.

    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    # 计算 tanh 函数的导数，将其应用到误差信号上
    delta *= (1 - Z ** 2)
    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        前向传播过程中从双曲正切激活函数输出的数据，形状为 (n_samples, n_features)

    delta : {array-like}, shape (n_samples, n_features)
         要就地修改的反向传播误差信号。
    """
    # 更新反向传播误差信号，根据双曲正切激活函数的导数形式进行调整
    delta *= 1 - Z**2
# 定义一个函数，用于在反向传播时应用ReLU函数的导数。

# 参数Z：
#   - 形状为 (n_samples, n_features) 的数组或稀疏矩阵，包含ReLU激活函数前向传播输出的数据。

# 参数delta：
#   - 形状为 (n_samples, n_features) 的数组，表示需要原地修改的反向传播误差信号。
def inplace_relu_derivative(Z, delta):
    # 将Z中等于0的元素对应位置的delta值置为0，即ReLU函数的导数在Z为0时为0。
    delta[Z == 0] = 0


# 定义一个字典，将激活函数名称映射到对应的导数函数。
DERIVATIVES = {
    "identity": inplace_identity_derivative,
    "tanh": inplace_tanh_derivative,
    "logistic": inplace_logistic_derivative,
    "relu": inplace_relu_derivative,
}


# 定义一个函数，计算回归任务中的均方误差损失。
# 参数y_true：
#   - 数组或标签指示矩阵，包含真实的目标值。
# 参数y_pred：
#   - 数组或标签指示矩阵，包含回归估计器预测的值。
# 返回值：
#   - float，表示样本被正确预测的程度。
def squared_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean() / 2


# 定义一个函数，计算分类任务中的对数损失。
# 参数y_true：
#   - 数组或标签指示矩阵，包含正确的类标签。
# 参数y_prob：
#   - 形状为 (n_samples, n_classes) 的浮点数数组，包含分类器预测的概率。
# 返回值：
#   - float，表示样本被正确预测的程度。
def log_loss(y_true, y_prob):
    # 获取y_prob的数据类型的机器极小值
    eps = np.finfo(y_prob.dtype).eps
    # 将y_prob限制在eps和1-eps之间的范围内，避免数值稳定性问题
    y_prob = np.clip(y_prob, eps, 1 - eps)

    # 如果y_prob的第二维度为1，则将其转换为二分类问题的格式
    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)

    # 如果y_true的第二维度为1，则将其转换为二分类问题的格式
    if y_true.shape[1] == 1:
        y_true = np.append(1 - y_true, y_true, axis=1)

    # 计算对数损失
    return -xlogy(y_true, y_prob).sum() / y_prob.shape[0]


# 定义一个函数，计算二分类任务中的对数损失。
# 参数y_true：
#   - 数组或标签指示矩阵，包含正确的类标签。
# 参数y_prob：
#   - 形状为 (n_samples, 1) 的浮点数数组，包含分类器预测的概率。
# 返回值：
#   - float，表示样本被正确预测的程度。
def binary_log_loss(y_true, y_prob):
    # 获取y_prob的数据类型的机器极小值
    eps = np.finfo(y_prob.dtype).eps
    # 将y_prob限制在eps和1-eps之间的范围内，避免数值稳定性问题
    y_prob = np.clip(y_prob, eps, 1 - eps)
    # 计算二分类问题的对数损失
    return (
        -(xlogy(y_true, y_prob).sum() + xlogy(1 - y_true, 1 - y_prob).sum())
        / y_prob.shape[0]
    )


# 定义一个字典，将损失函数名称映射到对应的损失计算函数。
LOSS_FUNCTIONS = {
    "squared_error": squared_loss,
    "log_loss": log_loss,
    "binary_log_loss": binary_log_loss,
}
```