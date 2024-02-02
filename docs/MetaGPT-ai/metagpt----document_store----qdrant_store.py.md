# `MetaGPT\metagpt\document_store\qdrant_store.py`

```py

# 导入必要的模块和类
from dataclasses import dataclass
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, PointStruct, VectorParams
from metagpt.document_store.base_store import BaseStore

# 定义一个数据类，用于存储 Qdrant 连接的相关信息
@dataclass
class QdrantConnection:
    """
    Args:
        url: qdrant url
        host: qdrant host
        port: qdrant port
        memory: qdrant service use memory mode
        api_key: qdrant cloud api_key
    """
    url: str = None
    host: str = None
    port: int = None
    memory: bool = False
    api_key: str = None

# 定义 Qdrant 存储类，继承自 BaseStore
class QdrantStore(BaseStore):
    # 初始化方法，根据连接信息创建 QdrantClient 对象
    def __init__(self, connect: QdrantConnection):
        if connect.memory:
            self.client = QdrantClient(":memory:")
        elif connect.url:
            self.client = QdrantClient(url=connect.url, api_key=connect.api_key)
        elif connect.host and connect.port:
            self.client = QdrantClient(host=connect.host, port=connect.port, api_key=connect.api_key)
        else:
            raise Exception("please check QdrantConnection.")

    # 创建一个集合
    def create_collection(
        self,
        collection_name: str,
        vectors_config: VectorParams,
        force_recreate=False,
        **kwargs,
    ):
        """
        create a collection
        Args:
            collection_name: collection name
            vectors_config: VectorParams object,detail in https://github.com/qdrant/qdrant-client
            force_recreate: default is False, if True, will delete exists collection,then create it
            **kwargs:

        Returns:

        """

    # 检查集合是否存在
    def has_collection(self, collection_name: str):
        try:
            self.client.get_collection(collection_name)
            return True
        except:  # noqa: E722
            return False

    # 删除集合
    def delete_collection(self, collection_name: str, timeout=60):
        res = self.client.delete_collection(collection_name, timeout=timeout)
        if not res:
            raise Exception(f"Delete collection {collection_name} failed.")

    # 向集合中添加数据
    def add(self, collection_name: str, points: List[PointStruct]):
        """
        add some vector data to qdrant
        Args:
            collection_name: collection name
            points: list of PointStruct object, about PointStruct detail in https://github.com/qdrant/qdrant-client

        Returns: NoneX

        """

    # 在集合中进行向量搜索
    def search(
        self,
        collection_name: str,
        query: List[float],
        query_filter: Filter = None,
        k=10,
        return_vector=False,
    ):
        """
        vector search
        Args:
            collection_name: qdrant collection name
            query: input vector
            query_filter: Filter object, detail in https://github.com/qdrant/qdrant-client
            k: return the most similar k pieces of data
            return_vector: whether return vector

        Returns: list of dict

        """

    # 写入方法，暂时不做任何操作
    def write(self, *args, **kwargs):
        pass

```