# `.\graphrag\graphrag\index\graph\extractors\claims\__init__.py`

```py
# 版权声明，指明此部分代码版权归 Microsoft Corporation 所有，使用 MIT 许可证授权
# 从当前目录中导入 claim_extractor 和 CLAIM_EXTRACTION_PROMPT
from .claim_extractor import ClaimExtractor
from .prompts import CLAIM_EXTRACTION_PROMPT

# __all__ 列表，用于指定模块中哪些对象会被 `from module import *` 导入
__all__ = ["CLAIM_EXTRACTION_PROMPT", "ClaimExtractor"]
```