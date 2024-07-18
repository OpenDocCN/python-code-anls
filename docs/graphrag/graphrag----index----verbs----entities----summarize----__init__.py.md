# `.\graphrag\graphrag\index\verbs\entities\summarize\__init__.py`

```py
# 版权声明和许可证信息，声明代码版权归 Microsoft Corporation 所有，使用 MIT 许可证进行授权
# Licensed under the MIT License
# 这是一个包的根目录，用于处理解析实体相关的功能。

# 导入描述摘要相关的模块和函数
from .description_summarize import SummarizeStrategyType, summarize_descriptions

# 设置模块中可以被导出的符号列表，这里包括了 SummarizeStrategyType 类和 summarize_descriptions 函数
__all__ = ["SummarizeStrategyType", "summarize_descriptions"]
```