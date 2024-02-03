# `.\AutoGPT\autogpts\forge\forge\memory\chroma_memstore.py`

```py
import hashlib
# 导入 hashlib 模块，用于生成哈希值

import chromadb
# 导入 chromadb 模块

from chromadb.config import Settings
# 从 chromadb.config 模块中导入 Settings 类

from .memstore import MemStore
# 从当前目录下的 memstore 模块中导入 MemStore 类

class ChromaMemStore:
    """
    A class used to represent a Memory Store
    """
    # 用于表示内存存储的类

    def __init__(self, store_path: str):
        """
        Initialize the MemStore with a given store path.

        Args:
            store_path (str): The path to the store.
        """
        # 初始化 MemStore 类，传入存储路径参数

        self.client = chromadb.PersistentClient(
            path=store_path, settings=Settings(anonymized_telemetry=False)
        )
        # 创建 PersistentClient 对象，传入存储路径和设置参数

    def add(self, task_id: str, document: str, metadatas: dict) -> None:
        """
        Add a document to the MemStore.

        Args:
            task_id (str): The ID of the task.
            document (str): The document to be added.
            metadatas (dict): The metadata of the document.
        """
        # 向 MemStore 中添加文档

        doc_id = hashlib.sha256(document.encode()).hexdigest()[:20]
        # 使用 SHA256 算法生成文档的哈希值

        collection = self.client.get_or_create_collection(task_id)
        # 获取或创建指定任务 ID 的集合

        collection.add(documents=[document], metadatas=[metadatas], ids=[doc_id])
        # 向集合中添加文档、元数据和文档 ID

    def query(
        self,
        task_id: str,
        query: str,
        filters: dict = None,
        document_search: dict = None,
    ) -> dict:
        """
        Query the MemStore.

        Args:
            task_id (str): The ID of the task.
            query (str): The query string.
            filters (dict, optional): The filters to be applied. Defaults to None.
            search_string (str, optional): The search string. Defaults to None.

        Returns:
            dict: The query results.
        """
        # 查询 MemStore 中的数据

        collection = self.client.get_or_create_collection(task_id)
        # 获取或创建指定任务 ID 的集合

        kwargs = {
            "query_texts": [query],
            "n_results": 10,
        }
        # 定义查询参数字典，包括查询文本和结果数量

        if filters:
            kwargs["where"] = filters
        # 如果有过滤条件，则添加到查询参数中

        if document_search:
            kwargs["where_document"] = document_search
        # 如果有文档搜索条件，则添加到查询参数中

        return collection.query(**kwargs)
        # 调用集合的查询方法，传入查询参数并返回结果字典
    def get(self, task_id: str, doc_ids: list = None, filters: dict = None) -> dict:
        """
        Get documents from the MemStore.

        Args:
            task_id (str): The ID of the task.
            doc_ids (list, optional): The IDs of the documents to be retrieved. Defaults to None.
            filters (dict, optional): The filters to be applied. Defaults to None.

        Returns:
            dict: The retrieved documents.
        """
        # 获取或创建与任务相关的集合
        collection = self.client.get_or_create_collection(task_id)
        kwargs = {}
        # 如果有指定文档ID，则添加到参数中
        if doc_ids:
            kwargs["ids"] = doc_ids
        # 如果有过滤条件，则添加到参数中
        if filters:
            kwargs["where"] = filters
        # 调用集合对象的get方法获取文档
        return collection.get(**kwargs)

    def update(self, task_id: str, doc_ids: list, documents: list, metadatas: list):
        """
        Update documents in the MemStore.

        Args:
            task_id (str): The ID of the task.
            doc_ids (list): The IDs of the documents to be updated.
            documents (list): The updated documents.
            metadatas (list): The updated metadata.
        """
        # 获取或创建与任务相关的集合
        collection = self.client.get_or_create_collection(task_id)
        # 调用集合对象的update方法更新文档
        collection.update(ids=doc_ids, documents=documents, metadatas=metadatas)

    def delete(self, task_id: str, doc_id: str):
        """
        Delete a document from the MemStore.

        Args:
            task_id (str): The ID of the task.
            doc_id (str): The ID of the document to be deleted.
        """
        # 获取或创建与任务相关的集合
        collection = self.client.get_or_create_collection(task_id)
        # 调用集合对象的delete方法删除文档
        collection.delete(ids=[doc_id])
if __name__ == "__main__":
    # 如果当前脚本被直接执行，则执行以下代码块

    print("#############################################")
    # 打印分隔线

    # 初始化 MemStore
    mem = ChromaMemStore(".agent_mem_store")

    # 测试添加函数
    task_id = "test_task"
    document = "This is a another new test document."
    metadatas = {"metadata": "test_metadata"}
    mem.add(task_id, document, metadatas)

    task_id = "test_task"
    document = "The quick brown fox jumps over the lazy dog."
    metadatas = {"metadata": "test_metadata"}
    mem.add(task_id, document, metadatas)

    task_id = "test_task"
    document = "AI is a new technology that will change the world."
    metadatas = {"timestamp": 1623936000}
    mem.add(task_id, document, metadatas)

    doc_id = hashlib.sha256(document.encode()).hexdigest()[:20]
    # 计算文档的 SHA256 哈希值

    # 测试查询函数
    query = "test"
    filters = {"metadata": {"$eq": "test"}}
    search_string = {"$contains": "test"}
    doc_ids = [doc_id]
    documents = ["This is an updated test document."]
    updated_metadatas = {"metadata": "updated_test_metadata"}

    print("Query:")
    print(mem.query(task_id, query))

    # 测试获取函数
    print("Get:")
    print(mem.get(task_id))

    # 测试更新函数
    print("Update:")
    print(mem.update(task_id, doc_ids, documents, updated_metadatas))

    print("Delete:")
    # 测试删除函数
    print(mem.delete(task_id, doc_ids[0]))
```