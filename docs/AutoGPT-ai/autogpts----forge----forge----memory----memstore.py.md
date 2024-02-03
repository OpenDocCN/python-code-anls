# `.\AutoGPT\autogpts\forge\forge\memory\memstore.py`

```py
import abc
import hashlib

import chromadb
from chromadb.config import Settings


class MemStore(abc.ABC):
    """
    An abstract class that represents a Memory Store
    """

    @abc.abstractmethod
    def __init__(self, store_path: str):
        """
        Initialize the MemStore with a given store path.

        Args:
            store_path (str): The path to the store.
        """
        pass

    @abc.abstractmethod
    def add_task_memory(self, task_id: str, document: str, metadatas: dict) -> None:
        """
        Add a document to the current tasks MemStore.
        This function calls the base version with the task_id as the collection_name.

        Args:
            task_id (str): The ID of the task.
            document (str): The document to be added.
            metadatas (dict): The metadata of the document.
        """
        self.add(collection_name=task_id, document=document, metadatas=metadatas)

    @abc.abstractmethod
    def query_task_memory(
        self,
        task_id: str,
        query: str,
        filters: dict = None,
        document_search: dict = None,
    ) -> dict:
        """
        Query the current tasks MemStore.
        This function calls the base version with the task_id as the collection_name.

        Args:
            task_id (str): The ID of the task.
            query (str): The query string.
            filters (dict, optional): The filters to be applied. Defaults to None.
            document_search (dict, optional): The search string. Defaults to None.

        Returns:
            dict: The query results.
        """
        return self.query(
            collection_name=task_id,
            query=query,
            filters=filters,
            document_search=document_search,
        )

    @abc.abstractmethod
    def get_task_memory(
        self, task_id: str, doc_ids: list = None, filters: dict = None
    ):
        """
        Get the documents from the current tasks MemStore.
        This function calls the base version with the task_id as the collection_name.

        Args:
            task_id (str): The ID of the task.
            doc_ids (list, optional): The list of document IDs to retrieve. Defaults to None.
            filters (dict, optional): The filters to be applied. Defaults to None.
        """
        return self.get(
            collection_name=task_id,
            doc_ids=doc_ids,
            filters=filters,
        )
    ) -> dict:
        """
        从当前任务的 MemStore 中获取文档。
        此函数使用任务 ID 作为集合名称调用基本版本。

        Args:
            task_id (str): 任务的 ID。
            doc_ids (list, optional): 要检索的文档的 ID。默认为 None。
            filters (dict, optional): 要应用的过滤器。默认为 None。

        Returns:
            dict: 检索到的文档。
        """
        return self.get(collection_name=task_id, doc_ids=doc_ids, filters=filters)

    @abc.abstractmethod
    def update_task_memory(
        self, task_id: str, doc_ids: list, documents: list, metadatas: list
    ):
        """
        更新当前任务的 MemStore 中的文档。
        此函数使用任务 ID 作为集合名称调用基本版本。

        Args:
            task_id (str): 任务的 ID。
            doc_ids (list): 要更新的文档的 ID。
            documents (list): 更新后的文档。
            metadatas (list): 更新后的元数据。
        """
        self.update(
            collection_name=task_id,
            doc_ids=doc_ids,
            documents=documents,
            metadatas=metadatas,
        )

    @abc.abstractmethod
    def delete_task_memory(self, task_id: str, doc_id: str):
        """
        从当前任务的 MemStore 中删除文档。
        此函数使用任务 ID 作为集合名称调用基本版本。

        Args:
            task_id (str): 任务的 ID。
            doc_id (str): 要删除的文档的 ID。
        """
        self.delete(collection_name=task_id, doc_id=doc_id)

    @abc.abstractmethod
    # 定义一个方法，用于向当前集合的 MemStore 中添加文档
    def add(self, collection_name: str, document: str, metadatas: dict) -> None:
        """
        Add a document to the current collection's MemStore.

        Args:
            collection_name (str): The name of the collection.
            document (str): The document to be added.
            metadatas (dict): The metadata of the document.
        """
        pass

    # 定义一个抽象方法，用于查询集合中的文档
    @abc.abstractmethod
    def query(
        self,
        collection_name: str,
        query: str,
        filters: dict = None,
        document_search: dict = None,
    ) -> dict:
        pass

    # 定义一个抽象方法，用于获取集合中的文档
    @abc.abstractmethod
    def get(
        self, collection_name: str, doc_ids: list = None, filters: dict = None
    ) -> dict:
        pass

    # 定义一个抽象方法，用于更新集合中的文档
    @abc.abstractmethod
    def update(
        self, collection_name: str, doc_ids: list, documents: list, metadatas: list
    ):
        pass

    # 定义一个抽象方法，用于删除集合中的文档
    @abc.abstractmethod
    def delete(self, collection_name: str, doc_id: str):
        pass
```