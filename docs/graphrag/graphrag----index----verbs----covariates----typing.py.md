# `.\graphrag\graphrag\index\verbs\covariates\typing.py`

```py
# 2024年版权所有 Microsoft Corporation.
# 根据 MIT 许可证授权

"""包含 'Covariate' 和 'CovariateExtractionResult' 模型的模块。"""

# 从 collections.abc 导入 Awaitable、Callable 和 Iterable 类型
from collections.abc import Awaitable, Callable, Iterable
# 导入 dataclass 装饰器，用于创建数据类
from dataclasses import dataclass
# 导入 Any 类型，用于支持任意类型的数据
from typing import Any

# 从 datashaper 模块导入 VerbCallbacks 类
from datashaper import VerbCallbacks

# 从 graphrag.index.cache 模块导入 PipelineCache 类
from graphrag.index.cache import PipelineCache


@dataclass
class Covariate:
    """协变量类的定义。"""

    # 协变量的类型
    covariate_type: str | None = None
    # 主体 ID
    subject_id: str | None = None
    # 主体类型
    subject_type: str | None = None
    # 对象 ID
    object_id: str | None = None
    # 对象类型
    object_type: str | None = None
    # 类型
    type: str | None = None
    # 状态
    status: str | None = None
    # 开始日期
    start_date: str | None = None
    # 结束日期
    end_date: str | None = None
    # 描述
    description: str | None = None
    # 源文本
    source_text: list[str] | None = None
    # 文档 ID
    doc_id: str | None = None
    # 记录 ID
    record_id: int | None = None
    # ID
    id: str | None = None


@dataclass
class CovariateExtractionResult:
    """协变量提取结果类的定义。"""

    # 协变量数据列表
    covariate_data: list[Covariate]


# 定义协变量提取策略类型别名
CovariateExtractStrategy = Callable[
    [
        Iterable[str],       # 可迭代对象，用于提取协变量
        list[str],           # 协变量 ID 列表
        dict[str, str],      # 其他上下文信息
        VerbCallbacks,       # 动词回调对象
        PipelineCache,       # 管道缓存对象
        dict[str, Any],      # 其他参数字典
    ],
    Awaitable[CovariateExtractionResult],  # 异步返回协变量提取结果
]
```