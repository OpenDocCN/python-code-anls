# `.\DB-GPT-src\dbgpt\serve\agent\model.py`

```py
from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

from dbgpt._private.pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")

# 分页过滤器模型，支持泛型
class PagenationFilter(BaseModel, Generic[T]):
    page_index: int = 1  # 默认页索引为1
    page_size: int = 20  # 默认页大小为20
    filter: T = None     # 过滤条件，默认为None

# 分页结果模型，支持泛型
class PagenationResult(BaseModel, Generic[T]):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # 配置模型允许任意类型
    page_index: int = 1        # 当前页索引，默认为1
    page_size: int = 20        # 每页大小，默认为20
    total_page: int = 0        # 总页数，默认为0
    total_row_count: int = 0   # 总行数，默认为0
    datas: List[T] = []        # 数据列表，默认为空列表

    def to_dic(self):
        # 将数据列表中每个元素的字典表示添加到结果字典中
        data_dicts = []
        for item in self.datas:
            data_dicts.append(item.__dict__)
        return {
            "page_index": self.page_index,
            "page_size": self.page_size,
            "total_page": self.total_page,
            "total_row_count": self.total_row_count,
            "datas": data_dicts,
        }

# 插件中心过滤器模型
@dataclass
class PluginHubFilter(BaseModel):
    name: Optional[str] = None             # 插件名称，可选
    description: Optional[str] = None      # 插件描述，可选
    author: Optional[str] = None           # 插件作者，可选
    email: Optional[str] = None            # 插件作者邮箱，可选
    type: Optional[str] = None             # 插件类型，可选
    version: Optional[str] = None          # 插件版本，可选
    storage_channel: Optional[str] = None  # 存储通道，可选
    storage_url: Optional[str] = None      # 存储 URL，可选

# 我的插件过滤器模型
@dataclass
class MyPluginFilter(BaseModel):
    tenant: str       # 租户，必需
    user_code: str    # 用户代码，必需
    user_name: str    # 用户名，必需
    name: str         # 插件名称，必需
    file_name: str    # 文件名，必需
    type: str         # 类型，必需
    version: str      # 版本，必需
    # 用户名，可选字段，用于存储插件的用户名称
    user_name: Optional[str] = Field(None, description="My Plugin user name")
    
    # 系统代码，可选字段，用于存储插件的系统代码
    sys_code: Optional[str] = Field(None, description="My Plugin sys code")
    
    # 名称，必填字段，用于存储插件的名称
    name: str = Field(..., description="My Plugin name")
    
    # 文件名，必填字段，用于存储插件的文件名
    file_name: str = Field(..., description="My Plugin file name")
    
    # 类型，可选字段，用于存储插件的类型
    type: Optional[str] = Field(None, description="My Plugin type")
    
    # 版本，可选字段，用于存储插件的版本号
    version: Optional[str] = Field(None, description="My Plugin version")
    
    # 使用次数，可选字段，用于存储插件的使用次数统计
    use_count: Optional[int] = Field(None, description="My Plugin use count")
    
    # 成功次数，可选字段，用于存储插件的成功执行次数统计
    succ_count: Optional[int] = Field(None, description="My Plugin succ count")
    
    # 创建时间，可选字段，用于存储插件的安装时间（GMT 时间）
    gmt_created: Optional[str] = Field(None, description="My Plugin install time")
```