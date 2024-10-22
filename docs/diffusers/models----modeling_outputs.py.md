# `.\diffusers\models\modeling_outputs.py`

```py
# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass

# 从上级目录的 utils 模块导入 BaseOutput 类
from ..utils import BaseOutput


# 定义 AutoencoderKLOutput 类，继承自 BaseOutput
@dataclass
class AutoencoderKLOutput(BaseOutput):
    """
    AutoencoderKL 编码方法的输出。

    参数：
        latent_dist (`DiagonalGaussianDistribution`):
            编码器的输出，以 `DiagonalGaussianDistribution` 的均值和对数方差表示。
            `DiagonalGaussianDistribution` 允许从分布中采样潜在变量。
    """

    # 定义 latent_dist 属性，类型为 DiagonalGaussianDistribution
    latent_dist: "DiagonalGaussianDistribution"  # noqa: F821


# 定义 Transformer2DModelOutput 类，继承自 BaseOutput
@dataclass
class Transformer2DModelOutput(BaseOutput):
    """
    [`Transformer2DModel`] 的输出。

    参数：
        sample (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)` 或 `(batch size, num_vector_embeds - 1, num_latent_pixels)` 如果 [`Transformer2DModel`] 是离散的):
            基于 `encoder_hidden_states` 输入的隐藏状态输出。如果是离散的，则返回无噪声潜在像素的概率分布。
    """

    # 定义 sample 属性，类型为 torch.Tensor
    sample: "torch.Tensor"  # noqa: F821
```