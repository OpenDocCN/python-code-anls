# `.\DB-GPT-src\dbgpt\app\knowledge\request\request.py`

```py
from enum import Enum  # 导入枚举类型模块
from typing import List, Optional  # 导入类型提示模块中的列表和可选类型

from dbgpt._private.pydantic import BaseModel, ConfigDict  # 导入自定义模块中的基础模型和配置字典


class KnowledgeQueryRequest(BaseModel):
    """知识查询请求模型"""

    query: str  # 查询字符串
    top_k: int  # 返回前K个文档


class KnowledgeSpaceRequest(BaseModel):
    """知识空间请求模型"""

    id: int = None  # 知识空间的唯一标识
    name: str = None  # 知识空间的名称
    vector_type: str = None  # 向量类型
    domain_type: str = "normal"  # 域类型，默认为普通
    desc: str = None  # 描述信息
    owner: str = None  # 所有者


class BusinessFieldType(Enum):
    """业务字段类型枚举"""

    NORMAL = "Normal"  # 普通类型


class KnowledgeDocumentRequest(BaseModel):
    """知识文档请求模型"""

    doc_name: str = None  # 文档路径
    doc_type: str = None  # 文档类型
    content: str = None  # 文档内容
    source: str = None  # 文档来源


class DocumentQueryRequest(BaseModel):
    """文档查询请求模型"""

    doc_name: str = None  # 文档路径
    doc_ids: Optional[List] = None  # 文档ID列表
    doc_type: str = None  # 文档类型
    status: str = None  # 文档状态
    page: int = 1  # 页码，默认为第1页
    page_size: int = 20  # 每页大小，默认为20条


class GraphVisRequest(BaseModel):
    """图形可视化请求模型"""

    limit: int = 100  # 限制返回结果的数量，默认为100条


class DocumentSyncRequest(BaseModel):
    """文档同步请求模型"""

    model_config = ConfigDict(protected_namespaces=())  # 模型配置信息

    doc_ids: List  # 文档ID列表

    model_name: Optional[str] = None  # 模型名称

    pre_separator: Optional[str] = None  # 预分隔符，用于文档分隔前的预处理分隔符，不包含在向量化的文本中

    separators: Optional[List[str]] = None  # 自定义分隔符列表

    chunk_size: Optional[int] = None  # 自定义块大小

    chunk_overlap: Optional[int] = None  # 自定义块重叠


class ChunkQueryRequest(BaseModel):
    """块查询请求模型"""

    id: int = None  # 块的唯一标识
    document_id: int = None  # 文档ID
    doc_name: str = None  # 文档路径
    doc_type: str = None  # 文档类型
    page: int = 1  # 页码，默认为第1页
    page_size: int = 20  # 每页大小，默认为20条


class SpaceArgumentRequest(BaseModel):
    """空间参数请求模型"""

    argument: str  # 参数名称


class DocumentSummaryRequest(BaseModel):
    """文档摘要请求模型"""

    model_config = ConfigDict(protected_namespaces=())  # 模型配置信息

    doc_id: int  # 文档ID
    model_name: str  # 模型名称
    conv_uid: str  # 会话唯一标识


class EntityExtractRequest(BaseModel):
    """实体提取请求模型"""

    model_config = ConfigDict(protected_namespaces=())  # 模型配置信息

    text: str  # 文本内容
    model_name: str  # 模型名称
```