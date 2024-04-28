# `.\transformers\generation_tf_utils.py`

```
# 导入警告模块
import warnings

# 导入 TFGenerationMixin 类，该类是 Transformers 库中的一个混合类，用于生成文本数据
from .generation import TFGenerationMixin

# 定义 TFGenerationMixin 类，继承自 TFGenerationMixin 类
class TFGenerationMixin(TFGenerationMixin):
    # 在导入时发出警告，提醒用户导入的方式已过时，将在 Transformers v5 中移除
    warnings.warn(
        "Importing `TFGenerationMixin` from `src/transformers/generation_tf_utils.py` is deprecated and will "
        "be removed in Transformers v5. Import as `from transformers import TFGenerationMixin` instead.",
        FutureWarning,
    )
```