# `.\DB-GPT-src\dbgpt\util\pagination_utils.py`

```py
from typing import Generic, List, TypeVar
# 导入必要的类型声明模块

from dbgpt._private.pydantic import BaseModel, ConfigDict, Field
# 导入基础模型类 BaseModel、配置字典类 ConfigDict 以及字段描述器 Field

T = TypeVar("T")
# 定义一个泛型类型变量 T

class PaginationResult(BaseModel, Generic[T]):
    """Pagination result"""
    # 定义分页结果类 PaginationResult，继承自基础模型类 BaseModel，并支持泛型类型 T

    model_config = ConfigDict(arbitrary_types_allowed=True)
    # 定义模型配置，允许任意类型的配置字典

    items: List[T] = Field(..., description="The items in the current page")
    # 分页结果中当前页的项目列表，类型为 T 的列表，使用 Field 描述器定义字段，必填，描述为当前页的项目列表

    total_count: int = Field(..., description="Total number of items")
    # 总项目数，整数类型，必填，描述为总项目数

    total_pages: int = Field(..., description="total number of pages")
    # 总页数，整数类型，必填，描述为总页数

    page: int = Field(..., description="Current page number")
    # 当前页码，整数类型，必填，描述为当前页码数

    page_size: int = Field(..., description="Number of items per page")
    # 每页项目数，整数类型，必填，描述为每页项目数
```