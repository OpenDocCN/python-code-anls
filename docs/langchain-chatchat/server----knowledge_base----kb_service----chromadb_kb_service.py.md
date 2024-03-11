# `.\Langchain-Chatchat\server\knowledge_base\kb_service\chromadb_kb_service.py`

```
# 导入必要的模块和类型提示
import uuid
from typing import Any, Dict, List, Tuple

import chromadb
from chromadb.api.types import (GetResult, QueryResult)
from langchain.docstore.document import Document

# 导入配置文件中的阈值
from configs import SCORE_THRESHOLD
# 导入基础服务类和适配器
from server.knowledge_base.kb_service.base import (EmbeddingsFunAdapter, KBService, SupportedVSType)
# 导入工具函数和路径获取函数
from server.knowledge_base.utils import KnowledgeFile, get_kb_path, get_vs_path

# 将获取的结果转换为文档列表
def _get_result_to_documents(get_result: GetResult) -> List[Document]:
    # 如果没有文档，则返回空列表
    if not get_result['documents']:
        return []

    # 如果没有元数据，则使用空字典填充
    _metadatas = get_result['metadatas'] if get_result['metadatas'] else [{}] * len(get_result['documents'])

    document_list = []
    # 遍历文档内容和元数据，创建文档对象并添加到列表中
    for page_content, metadata in zip(get_result['documents'], _metadatas):
        document_list.append(Document(**{'page_content': page_content, 'metadata': metadata}))

    return document_list

# 将结果转换为文档和分数的元组列表
def _results_to_docs_and_scores(results: Any) -> List[Tuple[Document, float]]:
    """
    from langchain_community.vectorstores.chroma import Chroma
    """
    return [
        # TODO: Chroma can do batch querying,
        # 将结果中的文档内容、元数据和距离组成元组，创建文档对象并添加到列表中
        (Document(page_content=result[0], metadata=result[1] or {}), result[2])
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]

# Chroma 知识库服务类，继承自基础服务类
class ChromaKBService(KBService):
    vs_path: str
    kb_path: str

    client = None
    collection = None

    # 返回支持的向量存储类型
    def vs_type(self) -> str:
        return SupportedVSType.CHROMADB

    # 获取向量存储路径
    def get_vs_path(self) -> str:
        return get_vs_path(self.kb_name, self.embed_model)

    # 获取知识库路径
    def get_kb_path(self) -> str:
        return get_kb_path(self.kb_name)

    # 初始化方法，设置知识库路径、向量存储路径，创建客户端和集合
    def do_init(self) -> None:
        self.kb_path = self.get_kb_path()
        self.vs_path = self.get_vs_path()
        self.client = chromadb.PersistentClient(path=self.vs_path)
        self.collection = self.client.get_or_create_collection(self.kb_name)
    def do_create_kb(self) -> None:
        # 在 ChromaDB 中，创建知识库等同于创建一个集合
        self.collection = self.client.get_or_create_collection(self.kb_name)

    def do_drop_kb(self):
        # 删除知识库等同于在 ChromaDB 中删除一个集合
        try:
            self.client.delete_collection(self.kb_name)
        except ValueError as e:
            if not str(e) == f"Collection {self.kb_name} does not exist.":
                raise e

    def do_search(self, query: str, top_k: int, score_threshold: float = SCORE_THRESHOLD) -> List[
        Tuple[Document, float]]:
        # 创建 EmbeddingsFunAdapter 对象，用于嵌入查询
        embed_func = EmbeddingsFunAdapter(self.embed_model)
        # 嵌入查询
        embeddings = embed_func.embed_query(query)
        # 在集合中查询，返回查询结果
        query_result: QueryResult = self.collection.query(query_embeddings=embeddings, n_results=top_k)
        return _results_to_docs_and_scores(query_result)

    def do_add_doc(self, docs: List[Document], **kwargs) -> List[Dict]:
        doc_infos = []
        # 将文档转换为嵌入数据
        data = self._docs_to_embeddings(docs)
        # 为每个文档生成唯一 ID
        ids = [str(uuid.uuid1()) for _ in range(len(data["texts"])]
        # 将文档添加到集合中
        for _id, text, embedding, metadata in zip(ids, data["texts"], data["embeddings"], data["metadatas"]):
            self.collection.add(ids=_id, embeddings=embedding, metadatas=metadata, documents=text)
            doc_infos.append({"id": _id, "metadata": metadata})
        return doc_infos

    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        # 根据 ID 获取文档
        get_result: GetResult = self.collection.get(ids=ids)
        return _get_result_to_documents(get_result)

    def del_doc_by_ids(self, ids: List[str]) -> bool:
        # 根据 ID 删除文档
        self.collection.delete(ids=ids)
        return True

    def do_clear_vs(self):
        # 清空向量存储可能等同于删除并重新创建集合
        self.do_drop_kb()

    def do_delete_doc(self, kb_file: KnowledgeFile, **kwargs):
        # 根据文件路径删除文档
        return self.collection.delete(where={"source": kb_file.filepath})
```