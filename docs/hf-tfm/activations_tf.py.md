# `.\activations_tf.py`

```
# 引入数学库
import math

# 引入 TensorFlow 库
import tensorflow as tf

# 引入版本解析工具
from packaging.version import parse

# 尝试引入 tf_keras 库，如果失败则引入 keras 库
try:
    import tf_keras as keras
except (ModuleNotFoundError, ImportError):
    import keras

    # 检查 keras 版本是否大于 2，如果是则抛出异常
    if parse(keras.__version__).major > 2:
        raise ValueError(
            "Your currently installed version of Keras is Keras 3, but this is not yet supported in "
            "Transformers. Please install the backwards-compatible tf-keras package with "
            "`pip install tf-keras`."
        )


def _gelu(x):
    """
    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created. For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) Also see
    https://arxiv.org/abs/1606.08415
    """
    # 将输入转换为 TensorFlow 张量
    x = tf.convert_to_tensor(x)
    # 计算高斯误差线性单元的输出
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.cast(tf.sqrt(2.0), x.dtype)))

    return x * cdf


def _gelu_new(x):
    """
    Gaussian Error Linear Unit. This is a smoother version of the GELU. Original paper: https://arxiv.org/abs/1606.0841

    Args:
        x: float Tensor to perform activation

    Returns:
        `x` with the GELU activation applied.
    """
    # 将输入转换为 TensorFlow 张量
    x = tf.convert_to_tensor(x)
    # 定义 pi 和系数
    pi = tf.cast(math.pi, x.dtype)
    coeff = tf.cast(0.044715, x.dtype)
    # 计算平滑的 GELU 输出
    cdf = 0.5 * (1.0 + tf.tanh(tf.sqrt(2.0 / pi) * (x + coeff * tf.pow(x, 3))))

    return x * cdf


def mish(x):
    # 将输入转换为 TensorFlow 张量
    x = tf.convert_to_tensor(x)
    # 计算 Mish 激活函数的输出
    return x * tf.tanh(tf.math.softplus(x))


def gelu_fast(x):
    # 将输入转换为 TensorFlow 张量
    x = tf.convert_to_tensor(x)
    # 定义系数
    coeff1 = tf.cast(0.044715, x.dtype)
    coeff2 = tf.cast(0.7978845608, x.dtype)
    # 计算快速 GELU 的输出
    return 0.5 * x * (1.0 + tf.tanh(x * coeff2 * (1.0 + coeff1 * x * x)))


def quick_gelu(x):
    # 将输入转换为 TensorFlow 张量
    x = tf.convert_to_tensor(x)
    # 定义系数
    coeff = tf.cast(1.702, x.dtype)
    # 计算快速 GELU 的输出
    return x * tf.math.sigmoid(coeff * x)


def gelu_10(x):
    """
    Clip the range of possible GeLU outputs between [-10, 10]. This is especially useful for quantization purpose, as
    it allows mapping 2 negatives values in the GeLU spectrum. For more information on this trick, please refer to
    https://arxiv.org/abs/2004.09602

    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    """
    # 截断 GeLU 输出的范围在 [-10, 10] 之间
    # 这对于量化目的非常有用，因为它允许在 GeLU 光谱中映射 2 个负值。有关此技巧的更多信息，请参阅链接
    """
    对输入的张量 x 应用改进的 GELU（Gaussian Error Linear Unit）激活函数，并进行值裁剪。

    GELU 函数的数学表达式是：
    0.5 * x * (1 + tanh(math.sqrt(2 / pi) * (x + 0.044715 * x^3)))

    这里使用了一个 TensorFlow 的内置函数 _gelu 来实现 GELU 激活函数。

    参数 x: 输入的张量
    返回值: 应用 GELU 激活函数后的张量，裁剪在 [-10, 10] 的范围内
    """
    return tf.clip_by_value(_gelu(x), -10, 10)
def glu(x, axis=-1):
    """
    Gated Linear Unit. Implementation as defined in the original paper (see https://arxiv.org/abs/1612.08083), where
    the input `x` is split in two halves across a dimension (`axis`), A and B, returning A * sigmoid(B).

    Args:
        `x`: float Tensor to perform activation
        `axis`: dimension across which `x` be split in half

    Returns:
        `x` with the GLU activation applied (with its size halved across the dimension `axis`).
    """
    # 将输入 `x` 沿指定轴 `axis` 分成两半，命名为 A 和 B
    a, b = tf.split(x, 2, axis=axis)
    # 返回 A * sigmoid(B) 的结果，即 GLU 激活函数的计算结果
    return a * tf.math.sigmoid(b)


if parse(tf.version.VERSION) >= parse("2.4"):

    def approximate_gelu_wrap(x):
        # 使用 Keras 中的 approximate gelu 函数来计算 gelu 激活
        return keras.activations.gelu(x, approximate=True)

    # 设置 gelu 和 gelu_new 激活函数
    gelu = keras.activations.gelu
    gelu_new = approximate_gelu_wrap
else:
    # 如果 TensorFlow 版本低于 2.4，则使用自定义的 _gelu 和 _gelu_new 函数
    gelu = _gelu
    gelu_new = _gelu_new


# 定义激活函数名称到对应函数的映射字典
ACT2FN = {
    "gelu": gelu,
    "gelu_10": gelu_10,
    "gelu_fast": gelu_fast,
    "gelu_new": gelu_new,
    "glu": glu,
    "mish": mish,
    "quick_gelu": quick_gelu,
    "relu": keras.activations.relu,
    "sigmoid": keras.activations.sigmoid,
    "silu": keras.activations.swish,
    "swish": keras.activations.swish,
    "tanh": keras.activations.tanh,
}


def get_tf_activation(activation_string):
    # 根据激活函数名称 `activation_string` 在 ACT2FN 字典中查找对应的函数
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        # 如果未找到对应的函数，则抛出 KeyError 异常
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")
```