# `.\graphrag\graphrag\index\graph\extractors\summarize\__init__.py`

```py
# 版权声明，说明此代码文件的版权归属于 2024 年的 Microsoft 公司，并遵循 MIT 许可证
# Licensed under the MIT License

# 当前文件是索引引擎单分图包的根目录

# 导入从 description_summary_extractor 模块中的指定内容：
# - SummarizationResult：用于描述摘要结果的类
# - SummarizeExtractor：用于执行摘要提取的类
from .description_summary_extractor import (
    SummarizationResult,
    SummarizeExtractor,
)

# 从 prompts 模块导入 SUMMARIZE_PROMPT，用作摘要提示的常量
from .prompts import SUMMARIZE_PROMPT

# __all__ 是一个特殊的 Python 变量，定义了在使用 `from module import *` 时导出的符号名列表
# 这里定义了当前模块中可导出的公共接口，包括：
# - "SUMMARIZE_PROMPT": 摘要提示常量
# - "SummarizationResult": 摘要结果类
# - "SummarizeExtractor": 摘要提取类
__all__ = ["SUMMARIZE_PROMPT", "SummarizationResult", "SummarizeExtractor"]
```