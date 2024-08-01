# `.\DB-GPT-src\dbgpt\app\knowledge\request\response.py`

```py
from typing import List, Optional  # 引入类型提示模块

from dbgpt._private.pydantic import BaseModel, Field  # 引入 Pydantic 模块中的 BaseModel 和 Field
from dbgpt.serve.rag.api.schemas import DocumentChunkVO, DocumentVO  # 从指定路径导入 DocumentChunkVO 和 DocumentVO 类


class ChunkQueryResponse(BaseModel):
    """文档分块查询响应模型"""

    data: List[DocumentChunkVO] = Field(..., description="文档分块列表")  # data 字段，包含 DocumentChunkVO 对象的列表
    """文档总结"""
    summary: Optional[str] = Field(None, description="文档总结")  # 可选的文档总结信息
    """总数"""
    total: Optional[int] = Field(None, description="总数")  # 可选的总数信息
    """页码"""
    page: Optional[int] = Field(None, description="当前页码")  # 可选的当前页码信息


class DocumentQueryResponse(BaseModel):
    """文档查询响应模型"""

    data: List[DocumentVO] = Field(..., description="文档列表")  # data 字段，包含 DocumentVO 对象的列表
    """总数"""
    total: Optional[int] = Field(None, description="总数")  # 可选的总数信息
    """页码"""
    page: Optional[int] = Field(None, description="当前页码")  # 可选的当前页码信息


class SpaceQueryResponse(BaseModel):
    """空间查询响应模型"""

    data: List[SpaceVO] = Field(..., description="空间列表")  # data 字段，包含 SpaceVO 对象的列表
    """ID"""
    id: int = None  # ID，整数类型
    """名称"""
    name: str = None  # 名称，字符串类型
    """向量类型"""
    vector_type: str = None  # 向量类型，字符串类型
    """领域类型"""
    domain_type: str = None  # 领域类型，字符串类型
    """描述"""
    desc: str = None  # 描述，字符串类型
    """上下文"""
    context: str = None  # 上下文，字符串类型
    """所有者"""
    owner: str = None  # 所有者，字符串类型
    """创建时间"""
    gmt_created: str = None  # 创建时间，字符串类型
    """修改时间"""
    gmt_modified: str = None  # 修改时间，字符串类型
    """文档计数"""
    docs: int = None  # 文档计数，整数类型


class KnowledgeQueryResponse(BaseModel):
    """知识查询响应模型"""

    """来源"""
    source: str  # 知识引用的来源，字符串类型
    """分数"""
    score: float = 0.0  # 知识向量查询的相似度分数，浮点数类型，默认值为 0.0
    """文本信息"""
    text: str  # 原始文本信息，字符串类型
```