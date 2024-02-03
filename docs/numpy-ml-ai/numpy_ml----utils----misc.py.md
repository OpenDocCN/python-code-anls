# `numpy-ml\numpy_ml\utils\misc.py`

```py
# 导入 numpy 库
import numpy as np

# 重新定义 logsumexp 函数，计算对数概率的和
def logsumexp(log_probs, axis=None):
    # 计算输入数组中的最大值
    _max = np.max(log_probs)
    # 对数概率减去最大值，避免数值溢出
    ds = log_probs - _max
    # 计算指数和
    exp_sum = np.exp(ds).sum(axis=axis)
    # 返回对数概率的和
    return _max + np.log(exp_sum)

# 计算对数高斯概率密度函数 log N(x_i | mu, sigma)
def log_gaussian_pdf(x_i, mu, sigma):
    # 获取均值向量的长度
    n = len(mu)
    # 计算常数项 a
    a = n * np.log(2 * np.pi)
    # 计算矩阵的行列式的自然对数
    _, b = np.linalg.slogdet(sigma)

    # 计算 y = sigma^(-1) * (x_i - mu)
    y = np.linalg.solve(sigma, x_i - mu)
    # 计算 c = (x_i - mu)^T * y
    c = np.dot(x_i - mu, y)
    # 返回对数高斯概率密度函数的值
    return -0.5 * (a + b + c)
```