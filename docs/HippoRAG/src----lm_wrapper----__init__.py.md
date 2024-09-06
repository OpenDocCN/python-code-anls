# `.\HippoRAG\src\lm_wrapper\__init__.py`

```py
# 导入 NumPy 库
import numpy as np


# 定义一个包装类，封装嵌入模型
class EmbeddingModelWrapper:
    # 定义一个方法，用于对文本进行编码
    def encode_text(self, text, instruction: str, norm: bool, return_cpu: bool, return_numpy: bool) -> np.ndarray:
        # 抛出未实现错误，表示此方法需在子类中实现
        raise NotImplementedError
```