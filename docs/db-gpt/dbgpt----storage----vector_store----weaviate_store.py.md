# `.\DB-GPT-src\dbgpt\storage\vector_store\weaviate_store.py`

```py
"""Weaviate vector store."""  # 导入 Weaviate 向量存储相关模块

import logging  # 导入日志模块
import os  # 导入操作系统功能模块
from typing import List, Optional  # 导入类型提示模块

from dbgpt._private.pydantic import ConfigDict, Field  # 导入配置字典和字段定义
from dbgpt.core import Chunk  # 从 dbgpt 核心模块导入 Chunk 类
from dbgpt.core.awel.flow import Parameter, ResourceCategory, register_resource  # 从 dbgpt 核心 awel 流程模块导入参数、资源分类和注册资源函数
from dbgpt.util.i18n_utils import _  # 导入国际化处理函数 _

from .base import _COMMON_PARAMETERS, VectorStoreBase, VectorStoreConfig  # 从当前包中导入基础参数、向量存储基类和向量存储配置类
from .filters import MetadataFilters  # 从当前包中导入元数据过滤器类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


@register_resource(
    _("Weaviate Vector Store"),  # 注册资源名称为 "Weaviate Vector Store"
    "weaviate_vector_store",  # 注册资源 ID 为 "weaviate_vector_store"
    category=ResourceCategory.VECTOR_STORE,  # 注册资源类别为向量存储
    description=_("Weaviate vector store."),  # 资源描述为 "Weaviate vector store."
    parameters=[
        *_COMMON_PARAMETERS,  # 包含通用参数列表
        Parameter.build_from(
            _("Weaviate URL"),  # 参数名称为 "Weaviate URL"
            "weaviate_url",  # 参数 ID 为 "weaviate_url"
            str,  # 参数类型为字符串
            description=_("weaviate url address, if not set, will use the default url."),  # 参数描述
            optional=True,  # 参数可选
            default=None,  # 默认值为 None
        ),
        Parameter.build_from(
            _("Persist Path"),  # 参数名称为 "Persist Path"
            "persist_path",  # 参数 ID 为 "persist_path"
            str,  # 参数类型为字符串
            description=_("the persist path of vector store."),  # 参数描述为向量存储的持久化路径
            optional=True,  # 参数可选
            default=None,  # 默认值为 None
        ),
    ],
)
class WeaviateVectorConfig(VectorStoreConfig):
    """Weaviate vector store config."""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # 定义允许任意类型的配置字典

    weaviate_url: str = Field(
        default=os.getenv("WEAVIATE_URL", None),  # 默认使用环境变量中的 WEAVIATE_URL，如果未设置则为 None
        description="weaviate url address, if not set, will use the default url.",  # 参数描述
    )
    persist_path: str = Field(
        default=os.getenv("WEAVIATE_PERSIST_PATH", None),  # 默认使用环境变量中的 WEAVIATE_PERSIST_PATH，如果未设置则为 None
        description="weaviate persist path.",  # 参数描述为 Weaviate 的持久化路径
    )


class WeaviateStore(VectorStoreBase):
    """Weaviate database."""

    def __init__(self, vector_store_config: WeaviateVectorConfig) -> None:
        """Initialize with Weaviate client."""
        try:
            import weaviate  # 尝试导入 weaviate 包
        except ImportError:
            raise ValueError(
                "Could not import weaviate python package. "  # 抛出导入错误信息
                "Please install it with `pip install weaviate-client`."
            )
        super().__init__()  # 调用父类初始化方法
        self.weaviate_url = vector_store_config.weaviate_url  # 设置 Weaviate 的 URL 地址
        self.embedding = vector_store_config.embedding_fn  # 设置嵌入函数
        self.vector_name = vector_store_config.name  # 设置向量名称
        self.persist_dir = os.path.join(  # 构建持久化目录路径
            vector_store_config.persist_path, vector_store_config.name + ".vectordb"
        )

        self.vector_store_client = weaviate.Client(self.weaviate_url)  # 创建 Weaviate 客户端对象

    def similar_search(
        self, text: str, topk: int, filters: Optional[MetadataFilters] = None
        # 定义用于执行相似搜索的方法，包括文本、返回结果数量、可选的元数据过滤器参数
    ) -> List[Chunk]:
        """Perform similar search in Weaviate."""
        # 记录日志，指示正在进行 Weaviate 的相似搜索
        logger.info("Weaviate similar search")
        
        # 在旧版本（v1.14 之前）中，使用 "certainty" 而不是 "distance"
        # nearText = {
        #     "concepts": [text],
        #     "distance": 0.75,
        # }
        
        # 调用嵌入式对象来获取查询的向量
        # vector = self.embedding.embed_query(text)
        
        # 使用向量存储客户端执行查询，并限制返回结果数量为 topk
        response = (
            self.vector_store_client.query.get(
                self.vector_name, ["metadata", "page_content"]
            )
            # .with_near_vector({"vector": vector})
            .with_limit(topk).do()
        )
        
        # 从响应中获取数据
        res = response["data"]["Get"][list(response["data"]["Get"].keys())[0]]
        
        # 初始化空列表用于存储 Chunk 对象
        docs = []
        
        # 遍历查询结果，将每个结果包装成 Chunk 对象并添加到 docs 列表中
        for r in res:
            docs.append(
                Chunk(
                    content=r["page_content"],
                    metadata={"metadata": r["metadata"]},
                )
            )
        
        # 返回包含 Chunk 对象的列表
        return docs

    def vector_name_exists(self) -> bool:
        """Whether the vector name exists in Weaviate.

        Returns:
            bool: True if the vector name exists, False otherwise.
        """
        try:
            # 检查向量名称是否存在于 Weaviate 中的模式中
            if self.vector_store_client.schema.get(self.vector_name):
                return True
            return False
        except Exception as e:
            # 记录错误日志，并返回 False 表示向量名称可能不存在或发生异常
            logger.error(f"vector_name_exists error, {str(e)}")
            return False
    def _default_schema(self) -> None:
        """Create default schema in Weaviate.

        Create the schema for Weaviate with a Document class containing metadata and
        text properties.
        """
        # 定义默认的 Weaviate 模式
        schema = {
            "classes": [
                {
                    "class": self.vector_name,
                    "description": "A document with metadata and text",
                    # "moduleConfig": {
                    #     "text2vec-transformers": {
                    #         "poolingStrategy": "masked_mean",
                    #         "vectorizeClassName": False,
                    #     }
                    # },
                    "properties": [
                        {
                            "dataType": ["text"],
                            # "moduleConfig": {
                            #     "text2vec-transformers": {
                            #         "skip": False,
                            #         "vectorizePropertyName": False,
                            #     }
                            # },
                            "description": "Metadata of the document",
                            "name": "metadata",
                        },
                        {
                            "dataType": ["text"],
                            # "moduleConfig": {
                            #     "text2vec-transformers": {
                            #         "skip": False,
                            #         "vectorizePropertyName": False,
                            #     }
                            # },
                            "description": "Text content of the document",
                            "name": "page_content",
                        },
                    ],
                    # "vectorizer": "text2vec-transformers",
                }
            ]
        }

        # 在 Weaviate 中创建定义好的模式
        self.vector_store_client.schema.create(schema)

    def load_document(self, chunks: List[Chunk]) -> List[str]:
        """Load document to Weaviate."""
        # 记录加载文档到 Weaviate
        logger.info("Weaviate load document")
        # 从文档块中提取文本内容和元数据
        texts = [doc.content for doc in chunks]
        metadatas = [doc.metadata for doc in chunks]

        # Import data
        # 使用批处理方式导入数据
        with self.vector_store_client.batch as batch:
            batch.batch_size = 100

            # 批量导入所有文档
            for i in range(len(texts)):
                properties = {
                    "metadata": metadatas[i]["source"],
                    "content": texts[i],
                }

                # 将数据对象添加到批处理中
                self.vector_store_client.batch.add_data_object(
                    data_object=properties, class_name=self.vector_name
                )
            # 刷新批处理队列
            self.vector_store_client.batch.flush()
        # TODO: return ids
        # 返回空列表，表示没有返回文档 ID
        return []
```