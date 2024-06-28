# `.\generation_utils.py`

```
# 导入警告模块
import warnings

# 从generation模块中导入GenerationMixin类
from .generation import GenerationMixin

# 定义GenerationMixin类，继承自GenerationMixin类
class GenerationMixin(GenerationMixin):
    # 在导入时发出警告，提示正在从旧路径导入GenerationMixin，该功能将在未来版本中移除
    warnings.warn(
        "Importing `GenerationMixin` from `src/transformers/generation_utils.py` is deprecated and will "
        "be removed in Transformers v4.40. Import as `from transformers import GenerationMixin` instead.",
        FutureWarning,
    )
```