# `.\DB-GPT-src\dbgpt\rag\evaluation\__init__.py`

```py
"""Module for evaluation of RAG."""

# 导入评估相关的检索器模块和类，忽略 F401 错误
from .retriever import (
    RetrieverEvaluationMetric,  # 导入检索器的评估指标类
    RetrieverEvaluator,          # 导入检索器的评估器类
    RetrieverSimilarityMetric,   # 导入检索器的相似度指标类
)

# 声明导出的全部变量列表，包括评估器和相关的指标类
__ALL__ = [
    "RetrieverEvaluator",          # 导出检索器评估器类
    "RetrieverSimilarityMetric",   # 导出检索器相似度指标类
    "RetrieverEvaluationMetric",   # 导出检索器评估指标类
]
```