# `.\transformers\activations_tf.py`

```
import math  # 导入 math 模块

import tensorflow as tf  # 导入 TensorFlow 模块
from packaging import version  # 导入 version 函数

def _gelu(x):
    """
    高斯误差线性单元。这是在 Google Bert 仓库最初创建时的 gelu 激活函数的原始实现。
    有关信息：OpenAI GPT 的 gelu 稍有不同（并且产生稍微不同的结果）:
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    也请参阅 https://arxiv.org/abs/1606.08415
    """
    x = tf.convert_to_tensor(x)  # 将输入转换为 TensorFlow 张量
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.cast(tf.sqrt(2.0), x.dtype)))  # 计算高斯误差函数值

    return x * cdf  # 返回带有 gelu 激活的输入

def _gelu_new(x):
    """
    高斯误差线性单元。这是 GELU 的平滑版本。
    原始论文：https://arxiv.org/abs/1606.0841

    Args:
        x: 要执行激活的浮点张量

    Returns:
        应用了 GELU 激活的 `x`。
    """
    x = tf.convert_to_tensor(x)  # 将输入转换为 TensorFlow 张量
    pi = tf.cast(math.pi, x.dtype)  # 将 pi 转换为张量的数据类型
    coeff = tf.cast(0.044715, x.dtype)  # 将常数系数转换为张量的数据类型
    cdf = 0.5 * (1.0 + tf.tanh(tf.sqrt(2.0 / pi) * (x + coeff * tf.pow(x, 3))))  # 计算 GELU 激活函数

    return x * cdf  # 返回带有 GELU 激活的输入

def mish(x):
    x = tf.convert_to_tensor(x)  # 将输入转换为 TensorFlow 张量

    return x * tf.tanh(tf.math.softplus(x))  # 返回带有 mish 激活的输入

def gelu_fast(x):
    x = tf.convert_to_tensor(x)  # 将输入转换为 TensorFlow 张量
    coeff1 = tf.cast(0.044715, x.dtype)  # 将常数系数转换为张量的数据类型
    coeff2 = tf.cast(0.7978845608, x.dtype)  # 将常数系数转换为张量的数据类型

    return 0.5 * x * (1.0 + tf.tanh(x * coeff2 * (1.0 + coeff1 * x * x)))  # 返回带有快速 gelu 激活的输入

def quick_gelu(x):
    x = tf.convert_to_tensor(x)  # 将输入转换为 TensorFlow 张量
    coeff = tf.cast(1.702, x.dtype)  # 将常数系数转换为张量的数据类型
    
    return x * tf.math.sigmoid(coeff * x)  # 返回带有快速 gelu 激活的输入

def gelu_10(x):
    """
    将可能的 GeLU 输出范围剪裁在 [-10, 10] 之间。这对于量化目的特别有用，因为它允许在 GeLU 谱中映射 2 个负值。
    关于此技巧的更多信息，请参阅 https://arxiv.org/abs/2004.09602

    高斯误差线性单元。这是在 Google Bert 仓库最初创建时的 gelu 激活函数的原始实现。
    有关信息：OpenAI GPT 的 gelu 稍有不同（并且产生稍微不同的结果）:
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    也请参阅 https://arxiv.org/abs/1606.08415
    :param x: 输入
    :return: 
    """
    return tf.clip_by_value(_gelu(x), -10, 10)  # 返回剪裁后的 gelu 输入

def glu(x, axis=-1):
    """
    门控线性单元。实现如原始论文中所定义的，参见 https://arxiv.org/abs/1612.08083

    """
    # 将输入的 `x` 沿着指定维度（`axis`）分成两半，分别为 A 和 B，返回 A * sigmoid(B) 的结果
    # 参数:
    #   `x`: 要执行激活的浮点张量
    #   `axis`: 沿着哪个维度将 `x` 分成两半
    # 返回:
    #   应用 GLU 激活后的 `x`（在维度 `axis` 上尺寸减半）
    """
    # 使用 TensorFlow 的 split 函数将输入张量 `x` 沿着指定维度分成两半，分别赋值给 a 和 b
    a, b = tf.split(x, 2, axis=axis)
    # 返回 a 乘以 b 经过 sigmoid 函数处理后的结果
    return a * tf.math.sigmoid(b)
# 检查 TensorFlow 版本是否大于等于 2.4
if version.parse(tf.version.VERSION) >= version.parse("2.4"):

    # 如果 TensorFlow 版本大于等于 2.4，定义一个近似的 GeLU 函数
    def approximate_gelu_wrap(x):
        # 使用 TensorFlow 提供的 GeLU 激活函数，使用近似方法
        return tf.keras.activations.gelu(x, approximate=True)

    # 将原始的 GeLU 函数赋值给 gelu
    gelu = tf.keras.activations.gelu
    # 将近似的 GeLU 函数赋值给 gelu_new
    gelu_new = approximate_gelu_wrap
else:
    # 如果 TensorFlow 版本小于 2.4，使用原始的 GeLU 实现
    gelu = _gelu
    # 使用原始的 GeLU 实现
    gelu_new = _gelu_new

# 激活函数名称到激活函数对象的映射
ACT2FN = {
    "gelu": gelu,
    "gelu_10": gelu_10,
    "gelu_fast": gelu_fast,
    "gelu_new": gelu_new,
    "glu": glu,
    "mish": mish,
    "quick_gelu": quick_gelu,
    "relu": tf.keras.activations.relu,
    "sigmoid": tf.keras.activations.sigmoid,
    "silu": tf.keras.activations.swish,
    "swish": tf.keras.activations.swish,
    "tanh": tf.keras.activations.tanh,
}

# 根据激活函数字符串获取对应的 TensorFlow 激活函数对象
def get_tf_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        # 如果激活函数字符串不在映射中，则抛出 KeyError 异常
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")
```