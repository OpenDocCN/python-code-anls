# `.\DB-GPT-src\dbgpt\serve\rag\api\schemas.py`

```py
from typing import List, Optional  # 导入需要的类型提示

from fastapi import File, UploadFile  # 导入 FastAPI 的文件上传相关功能

from dbgpt._private.pydantic import BaseModel, ConfigDict, Field  # 导入 Pydantic 相关模块
from dbgpt.rag.chunk_manager import ChunkParameters  # 导入 ChunkParameters 类

from ..config import SERVE_APP_NAME_HUMP  # 导入 SERVE_APP_NAME_HUMP 变量


class SpaceServeRequest(BaseModel):
    """表示空间服务请求的数据模型"""

    id: Optional[int] = Field(None, description="空间的唯一标识")
    name: str = Field(None, description="空间的名称")
    vector_type: str = Field(None, description="向量类型")
    domain_type: str = Field(None, description="领域类型")
    desc: Optional[str] = Field(None, description="描述信息")
    owner: Optional[str] = Field(None, description="所有者")
    context: Optional[str] = Field(None, description="相关上下文信息")
    gmt_created: Optional[str] = Field(None, description="创建时间")
    gmt_modified: Optional[str] = Field(None, description="修改时间")


class DocumentServeRequest(BaseModel):
    """表示文档服务请求的数据模型"""

    id: Optional[int] = Field(None, description="文档的唯一标识")
    doc_name: Optional[str] = Field(None, description="文档名称")
    doc_type: Optional[str] = Field(None, description="文档类型")
    content: Optional[str] = Field(None, description="文档内容")
    doc_file: UploadFile = File(..., description="上传的文档文件")
    doc_source: Optional[str] = Field(None, description="文档来源")
    space_id: Optional[str] = Field(None, description="所属空间的唯一标识")


class DocumentServeResponse(BaseModel):
    """表示文档服务响应的数据模型"""

    id: Optional[int] = Field(None, description="文档的唯一标识")
    doc_name: Optional[str] = Field(None, description="文档名称")
    doc_type: Optional[str] = Field(None, description="文档类型")
    content: Optional[str] = Field(None, description="文档内容")
    vector_ids: Optional[str] = Field(None, description="向量标识")
    doc_source: Optional[str] = Field(None, description="文档来源")
    space: Optional[str] = Field(None, description="所属空间的名称")


class KnowledgeSyncRequest(BaseModel):
    """表示知识同步请求的数据模型"""

    doc_id: Optional[int] = Field(None, description="文档的唯一标识")
    space_id: Optional[str] = Field(None, description="空间的唯一标识")
    model_name: Optional[str] = Field(None, description="模型名称")
    chunk_parameters: Optional[ChunkParameters] = Field(
        None, description="分块参数"
    )  # 使用 ChunkParameters 类描述的分块参数
class SpaceServeResponse(BaseModel):
    """Flow response model"""

    # 配置模型参数，标题为 SERVE_APP_NAME_HUMP 所指定的服务响应标题
    model_config = ConfigDict(title=f"ServeResponse for {SERVE_APP_NAME_HUMP}")

    """name: knowledge space name"""

    """vector_type: vector type"""

    # 空间ID，可选整数类型字段，用于标识空间的唯一ID
    id: Optional[int] = Field(None, description="The space id")

    # 空间名称，可选字符串类型字段，表示空间的名称
    name: Optional[str] = Field(None, description="The space name")

    """vector_type: vector type"""

    # 矢量类型，可选字符串类型字段，表示空间使用的矢量类型
    vector_type: Optional[str] = Field(None, description="The vector type")

    """desc: description"""

    # 描述，可选字符串类型字段，描述空间的详细信息
    desc: Optional[str] = Field(None, description="The description")

    """context: argument context"""

    # 上下文，可选字符串类型字段，表示空间的使用上下文
    context: Optional[str] = Field(None, description="The context")

    """owner: owner"""

    # 所有者，可选字符串类型字段，表示空间的所有者
    owner: Optional[str] = Field(None, description="The owner")

    """sys code"""

    # 系统代码，可选字符串类型字段，表示空间的系统代码
    sys_code: Optional[str] = Field(None, description="The sys code")

    """domain type"""

    # 领域类型，可选字符串类型字段，表示空间所属的领域类型
    domain_type: Optional[str] = Field(None, description="domain_type")

    # TODO define your own fields here


class DocumentChunkVO(BaseModel):
    # 文档分块值对象

    # 文档分块ID，整数类型字段，用于唯一标识文档分块
    id: int = Field(..., description="document chunk id")

    # 文档ID，整数类型字段，表示文档的唯一标识
    document_id: int = Field(..., description="document id")

    # 文档名称，字符串类型字段，表示文档的名称
    doc_name: str = Field(..., description="document name")

    # 文档类型，字符串类型字段，表示文档的类型
    doc_type: str = Field(..., description="document type")

    # 文档内容，字符串类型字段，表示文档的具体内容
    content: str = Field(..., description="document content")

    # 元信息，字符串类型字段，表示文档的元信息
    meta_info: str = Field(..., description="document meta info")

    # 创建时间，字符串类型字段，表示文档的创建时间
    gmt_created: str = Field(..., description="document create time")

    # 修改时间，字符串类型字段，表示文档的最后修改时间
    gmt_modified: str = Field(..., description="document modify time")


class DocumentVO(BaseModel):
    """Document Entity."""

    # 文档实体

    # 文档ID，整数类型字段，用于唯一标识文档
    id: int = Field(..., description="document id")

    # 文档名称，字符串类型字段，表示文档的名称
    doc_name: str = Field(..., description="document name")

    # 文档类型，字符串类型字段，表示文档的类型
    doc_type: str = Field(..., description="document type")

    # 空间，字符串类型字段，表示文档所属的空间名称
    space: str = Field(..., description="document space name")

    # 分块大小，整数类型字段，表示文档分块的大小
    chunk_size: int = Field(..., description="document chunk size")

    # 状态，字符串类型字段，表示文档的状态
    status: str = Field(..., description="document status")

    # 最后同步时间，字符串类型字段，表示文档的最后同步时间
    last_sync: str = Field(..., description="document last sync time")

    # 内容，字符串类型字段，表示文档的具体内容
    content: str = Field(..., description="document content")

    # 结果，可选字符串类型字段，表示文档的处理结果
    result: Optional[str] = Field(None, description="document result")

    # 矢量ID，可选字符串类型字段，表示文档关联的矢量ID
    vector_ids: Optional[str] = Field(None, description="document vector ids")

    # 摘要，可选字符串类型字段，表示文档的摘要信息
    summary: Optional[str] = Field(None, description="document summary")

    # 创建时间，字符串类型字段，表示文档的创建时间
    gmt_created: str = Field(..., description="document create time")

    # 修改时间，字符串类型字段，表示文档的最后修改时间
    gmt_modified: str = Field(..., description="document modify time")


class KnowledgeDomainType(BaseModel):
    """Knowledge domain type"""

    # 知识领域类型

    # 名称，字符串类型字段，表示领域类型的名称
    name: str = Field(..., description="The domain type name")

    # 描述，字符串类型字段，表示领域类型的描述信息
    desc: str = Field(..., description="The domain type description")


class KnowledgeStorageType(BaseModel):
    """Knowledge storage type"""

    # 知识存储类型

    # 名称，字符串类型字段，表示存储类型的名称
    name: str = Field(..., description="The storage type name")

    # 描述，字符串类型字段，表示存储类型的描述信息
    desc: str = Field(..., description="The storage type description")

    # 领域类型列表，包含 KnowledgeDomainType 对象的列表字段，表示存储类型支持的领域类型
    domain_types: List[KnowledgeDomainType] = Field(..., description="The domain types")


class KnowledgeConfigResponse(BaseModel):
    # 知识配置响应模型
    """Knowledge config response"""

    # 声明一个名为storage的变量，其类型为List[KnowledgeStorageType]，用于存储知识配置响应中的存储类型信息
    storage: List[KnowledgeStorageType] = Field(..., description="The storage types")
```