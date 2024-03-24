# `.\lucidrains\x-transformers\x_transformers\__init__.py`

```
# 从 x_transformers.x_transformers 模块中导入以下类
from x_transformers.x_transformers import (
    XTransformer,  # XTransformer 类，用于定义 Transformer 模型
    Encoder,  # Encoder 类，用于定义编码器
    Decoder,  # Decoder 类，用于定义解码器
    PrefixDecoder,  # PrefixDecoder 类，用于定义前缀解码器
    CrossAttender,  # CrossAttender 类，用于定义交叉注意力机制
    Attention,  # Attention 类，用于定义注意力机制
    TransformerWrapper,  # TransformerWrapper 类，用于包装 Transformer 模型
    ViTransformerWrapper  # ViTransformerWrapper 类，用于包装 Vision Transformer 模型
)

# 从 x_transformers.autoregressive_wrapper 模块中导入 AutoregressiveWrapper 类
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

# 从 x_transformers.nonautoregressive_wrapper 模块中导入 NonAutoregressiveWrapper 类
from x_transformers.nonautoregressive_wrapper import NonAutoregressiveWrapper

# 从 x_transformers.continuous 模块中导入以下类
from x_transformers.continuous import (
    ContinuousTransformerWrapper,  # ContinuousTransformerWrapper 类，用于包装连续 Transformer 模型
    ContinuousAutoregressiveWrapper  # ContinuousAutoregressiveWrapper 类，用于包装连续自回归模型
)

# 从 x_transformers.xval 模块中导入以下类
from x_transformers.xval import (
    XValTransformerWrapper,  # XValTransformerWrapper 类，用于包装交叉验证 Transformer 模型
    XValAutoregressiveWrapper  # XValAutoregressiveWrapper 类，用于包装交叉验证自回归模型
)

# 从 x_transformers.xl_autoregressive_wrapper 模块中导入 XLAutoregressiveWrapper 类
from x_transformers.xl_autoregressive_wrapper import XLAutoregressiveWrapper

# 从 x_transformers.dpo 模块中导入 DPO 类
from x_transformers.dpo import (
    DPO  # DPO 类，用于定义 Discrete-Continuous-Optimization 模型
)
```