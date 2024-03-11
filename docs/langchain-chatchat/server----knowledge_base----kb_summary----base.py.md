# `.\Langchain-Chatchat\server\knowledge_base\kb_summary\base.py`

```
# 导入所需的模块和类型提示
from typing import List

# 从 configs 模块中导入配置信息
from configs import (
    EMBEDDING_MODEL,
    KB_ROOT_PATH)

# 从 abc 模块中导入抽象基类
from abc import ABC, abstractmethod

# 从 kb_cache 模块中导入 Faiss 缓存相关的类和对象
from server.knowledge_base.kb_cache.faiss_cache import kb_faiss_pool, ThreadSafeFaiss

# 导入 os 和 shutil 模块
import os
import shutil

# 从 knowledge_metadata_repository 模块中导入与知识库元数据相关的函数
from server.db.repository.knowledge_metadata_repository import add_summary_to_db, delete_summary_from_db

# 从 document 模块中导入 Document 类
from langchain.docstore.document import Document

# 定义一个抽象基类 KBSummaryService
class KBSummaryService(ABC):
    kb_name: str
    embed_model: str
    vs_path: str
    kb_path: str

    # 初始化方法，接受知识库名称和嵌入模型名称作为参数
    def __init__(self,
                 knowledge_base_name: str,
                 embed_model: str = EMBEDDING_MODEL
                 ):
        # 初始化实例属性
        self.kb_name = knowledge_base_name
        self.embed_model = embed_model

        # 获取知识库路径和向量存储路径
        self.kb_path = self.get_kb_path()
        self.vs_path = self.get_vs_path()

        # 如果向量存储路径不存在，则创建
        if not os.path.exists(self.vs_path):
            os.makedirs(self.vs_path)

    # 获取向量存储路径
    def get_vs_path(self):
        return os.path.join(self.get_kb_path(), "summary_vector_store")

    # 获取知识库路径
    def get_kb_path(self):
        return os.path.join(KB_ROOT_PATH, self.kb_name)

    # 加载向量存储对象
    def load_vector_store(self) -> ThreadSafeFaiss:
        return kb_faiss_pool.load_vector_store(kb_name=self.kb_name,
                                               vector_name="summary_vector_store",
                                               embed_model=self.embed_model,
                                               create=True)
    # 向向量存储中添加摘要文档的摘要信息
    def add_kb_summary(self, summary_combine_docs: List[Document]):
        # 获取向量存储对象并进入上下文管理器
        with self.load_vector_store().acquire() as vs:
            # 向向量存储中添加文档，并获取文档的ID列表
            ids = vs.add_documents(documents=summary_combine_docs)
            # 将向量存储对象保存到本地
            vs.save_local(self.vs_path)

        # 根据文档ID和摘要文档信息创建摘要信息列表
        summary_infos = [{"summary_context": doc.page_content,
                          "summary_id": id,
                          "doc_ids": doc.metadata.get('doc_ids'),
                          "metadata": doc.metadata} for id, doc in zip(ids, summary_combine_docs)]
        # 将摘要信息添加到数据库中
        status = add_summary_to_db(kb_name=self.kb_name, summary_infos=summary_infos)
        return status

    # 创建知识库chunk summary
    def create_kb_summary(self):
        """
        创建知识库chunk summary
        :return:
        """

        # 如果向量存储路径不存在，则创建
        if not os.path.exists(self.vs_path):
            os.makedirs(self.vs_path)

    # 删除知识库chunk summary
    def drop_kb_summary(self):
        """
        删除知识库chunk summary
        :param kb_name:
        :return:
        """
        # 使用原子操作上下文管理器
        with kb_faiss_pool.atomic:
            # 从向量存储池中移除知识库
            kb_faiss_pool.pop(self.kb_name)
            # 递归删除向量存储路径
            shutil.rmtree(self.vs_path)
        # 从数据库中删除摘要信息
        delete_summary_from_db(kb_name=self.kb_name)
```