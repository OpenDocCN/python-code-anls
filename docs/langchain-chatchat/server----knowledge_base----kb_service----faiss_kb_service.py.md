# `.\Langchain-Chatchat\server\knowledge_base\kb_service\faiss_kb_service.py`

```
# 导入必要的模块
import os
import shutil

# 从configs模块中导入SCORE_THRESHOLD常量
from configs import SCORE_THRESHOLD
# 从server.knowledge_base.kb_service.base模块中导入KBService类、SupportedVSType枚举、EmbeddingsFunAdapter类
from server.knowledge_base.kb_service.base import KBService, SupportedVSType, EmbeddingsFunAdapter
# 从server.knowledge_base.kb_cache.faiss_cache模块中导入kb_faiss_pool、ThreadSafeFaiss类
from server.knowledge_base.kb_cache.faiss_cache import kb_faiss_pool, ThreadSafeFaiss
# 从server.knowledge_base.utils模块中导入KnowledgeFile类、get_kb_path函数、get_vs_path函数
from server.knowledge_base.utils import KnowledgeFile, get_kb_path, get_vs_path
# 从server.utils模块中导入torch_gc函数
from server.utils import torch_gc
# 从langchain.docstore.document模块中导入Document类
from langchain.docstore.document import Document
# 从typing模块中导入List、Dict、Optional、Tuple类
from typing import List, Dict, Optional, Tuple

# 定义FaissKBService类，继承自KBService类
class FaissKBService(KBService):
    # 定义vs_path属性
    vs_path: str
    # 定义kb_path属性
    kb_path: str
    # 定义vector_name属性，默认值为None

    # 定义vs_type方法，返回SupportedVSType.FAISS
    def vs_type(self) -> str:
        return SupportedVSType.FAISS

    # 定义get_vs_path方法，返回由kb_name和vector_name组成的路径
    def get_vs_path(self):
        return get_vs_path(self.kb_name, self.vector_name)

    # 定义get_kb_path方法，返回由kb_name组成的路径
    def get_kb_path(self):
        return get_kb_path(self.kb_name)

    # 定义load_vector_store方法，返回加载的ThreadSafeFaiss对象
    def load_vector_store(self) -> ThreadSafeFaiss:
        return kb_faiss_pool.load_vector_store(kb_name=self.kb_name,
                                               vector_name=self.vector_name,
                                               embed_model=self.embed_model)

    # 定义save_vector_store方法，保存加载的ThreadSafeFaiss对象到vs_path
    def save_vector_store(self):
        self.load_vector_store().save(self.vs_path)

    # 定义get_doc_by_ids方法，根据ids获取Document对象列表
    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        with self.load_vector_store().acquire() as vs:
            return [vs.docstore._dict.get(id) for id in ids]

    # 定义del_doc_by_ids方法，根据ids删除Document对象
    def del_doc_by_ids(self, ids: List[str]) -> bool:
        with self.load_vector_store().acquire() as vs:
            vs.delete(ids)

    # 定义do_init方法，初始化vector_name、kb_path和vs_path属性
    def do_init(self):
        self.vector_name = self.vector_name or self.embed_model
        self.kb_path = self.get_kb_path()
        self.vs_path = self.get_vs_path()

    # 定义do_create_kb方法，如果vs_path不存在则创建，加载vector_store
    def do_create_kb(self):
        if not os.path.exists(self.vs_path):
            os.makedirs(self.vs_path)
        self.load_vector_store()

    # 定义do_drop_kb方法，清空vector_store，尝试删除kb_path
    def do_drop_kb(self):
        self.clear_vs()
        try:
            shutil.rmtree(self.kb_path)
        except Exception:
            ...
    # 执行搜索操作，根据查询内容返回与之相似的文档及其得分
    def do_search(self,
                  query: str,
                  top_k: int,
                  score_threshold: float = SCORE_THRESHOLD,
                  ) -> List[Tuple[Document, float]]:
        # 创建嵌入适配器对象，用于获取查询的嵌入向量
        embed_func = EmbeddingsFunAdapter(self.embed_model)
        # 获取查询的嵌入向量
        embeddings = embed_func.embed_query(query)
        # 获取向量库对象并进行相似性搜索，返回文档及其得分
        with self.load_vector_store().acquire() as vs:
            docs = vs.similarity_search_with_score_by_vector(embeddings, k=top_k, score_threshold=score_threshold)
        return docs

    # 执行添加文档操作
    def do_add_doc(self,
                   docs: List[Document],
                   **kwargs,
                   ) -> List[Dict]:
        # 将文档转换为嵌入向量数据
        data = self._docs_to_embeddings(docs) # 将向量化单独出来可以减少向量库的锁定时间

        # 获取向量库对象并添加嵌入向量数据
        with self.load_vector_store().acquire() as vs:
            ids = vs.add_embeddings(text_embeddings=zip(data["texts"], data["embeddings"]),
                                    metadatas=data["metadatas"],
                                    ids=kwargs.get("ids"))
            # 如果不需要刷新向量库缓存，则保存本地向量库
            if not kwargs.get("not_refresh_vs_cache"):
                vs.save_local(self.vs_path)
        # 返回添加文档的信息
        doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
        # 执行内存回收
        torch_gc()
        return doc_infos

    # 执行删除文档操作
    def do_delete_doc(self,
                      kb_file: KnowledgeFile,
                      **kwargs):
        # 获取向量库对象并根据文件名删除文档
        with self.load_vector_store().acquire() as vs:
            ids = [k for k, v in vs.docstore._dict.items() if v.metadata.get("source").lower() == kb_file.filename.lower()]
            if len(ids) > 0:
                vs.delete(ids)
            # 如果不需要刷新向量库缓存，则保存本地向量库
            if not kwargs.get("not_refresh_vs_cache"):
                vs.save_local(self.vs_path)
        return ids

    # 执行清空向量库操作
    def do_clear_vs():
        # 使用原子操作清空向量库池中的内容
        with kb_faiss_pool.atomic:
            kb_faiss_pool.pop((self.kb_name, self.vector_name))
        try:
            # 尝试删除向量库路径下的所有文件
            shutil.rmtree(self.vs_path)
        except Exception:
            # 如果删除失败，则忽略异常
            ...
        # 创建向量库路径，如果路径已存在则不做任何操作
        os.makedirs(self.vs_path, exist_ok=True)
    # 检查数据库中是否存在指定文件名的文档，如果存在则返回"in_db"
    def exist_doc(self, file_name: str):
        # 调用父类方法检查数据库中是否存在指定文件名的文档
        if super().exist_doc(file_name):
            return "in_db"

        # 构建内容路径
        content_path = os.path.join(self.kb_path, "content")
        # 检查内容路径下是否存在指定文件名的文件，如果存在则返回"in_folder"
        if os.path.isfile(os.path.join(content_path, file_name)):
            return "in_folder"
        else:
            # 如果文件既不在数据库中也不在内容路径下，则返回False
            return False
# 如果当前脚本作为主程序执行
if __name__ == '__main__':
    # 创建一个 FaissKBService 对象，传入参数 "test"
    faissService = FaissKBService("test")
    # 向 FaissKBService 对象添加一个 KnowledgeFile 对象，文件名为 "README.md"，内容为 "test"
    faissService.add_doc(KnowledgeFile("README.md", "test"))
    # 从 FaissKBService 对象中删除一个 KnowledgeFile 对象，文件名为 "README.md"，内容为 "test"
    faissService.delete_doc(KnowledgeFile("README.md", "test"))
    # 执行 FaissKBService 对象的 do_drop_kb 方法，删除所有文档
    faissService.do_drop_kb()
    # 打印 FaissKBService 对象执行 search_docs 方法后返回的结果，搜索关键词为 "如何启动api服务"
    print(faissService.search_docs("如何启动api服务"))
```