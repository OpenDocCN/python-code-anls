# `.\Langchain-Chatchat\server\knowledge_base\kb_service\milvus_kb_service.py`

```py
# 导入必要的模块
from typing import List, Dict, Optional
from langchain.schema import Document
from langchain.vectorstores.milvus import Milvus
import os
from configs import kbs_config
from server.db.repository import list_file_num_docs_id_by_kb_name_and_file_name
from server.knowledge_base.kb_service.base import KBService, SupportedVSType, EmbeddingsFunAdapter, score_threshold_process
from server.knowledge_base.utils import KnowledgeFile

# 定义 MilvusKBService 类，继承自 KBService 类
class MilvusKBService(KBService):
    # 定义 MilvusKBService 类的属性 milvus
    milvus: Milvus

    # 静态方法，根据 milvus_name 获取 Collection 对象
    @staticmethod
    def get_collection(milvus_name):
        from pymilvus import Collection
        return Collection(milvus_name)

    # 根据 ids 获取文档内容
    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        result = []
        if self.milvus.col:
            # ids = [int(id) for id in ids]  # for milvus if needed #pr 2725
            # 查询数据列表
            data_list = self.milvus.col.query(expr=f'pk in {[int(_id) for _id in ids]}', output_fields=["*"])
            for data in data_list:
                text = data.pop("text")
                result.append(Document(page_content=text, metadata=data))
        return result

    # 根据 ids 删除文档
    def del_doc_by_ids(self, ids: List[str]) -> bool:
        self.milvus.col.delete(expr=f'pk in {ids}')

    # 静态方法，根据 milvus_name 和 content 进行搜索
    @staticmethod
    def search(milvus_name, content, limit=3):
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        c = MilvusKBService.get_collection(milvus_name)
        return c.search(content, "embeddings", search_params, limit=limit, output_fields=["content"])

    # 创建知识库
    def do_create_kb(self):
        pass

    # 返回支持的向量存储类型
    def vs_type(self) -> str:
        return SupportedVSType.MILVUS
    # 加载 Milvus 实例，使用指定的嵌入函数、集合名称、连接参数和索引参数
    def _load_milvus(self):
        self.milvus = Milvus(embedding_function=EmbeddingsFunAdapter(self.embed_model),
                             collection_name=self.kb_name,
                             connection_args=kbs_config.get("milvus"),
                             index_params=kbs_config.get("milvus_kwargs")["index_params"],
                             search_params=kbs_config.get("milvus_kwargs")["search_params"]
                             )

    # 初始化操作，调用加载 Milvus 实例的方法
    def do_init(self):
        self._load_milvus()

    # 删除知识库操作，释放集合并删除集合
    def do_drop_kb(self):
        if self.milvus.col:
            self.milvus.col.release()
            self.milvus.col.drop()

    # 搜索操作，加载 Milvus 实例，获取查询的嵌入向量，使用向量进行相似度搜索，返回结果
    def do_search(self, query: str, top_k: int, score_threshold: float):
        self._load_milvus()
        embed_func = EmbeddingsFunAdapter(self.embed_model)
        embeddings = embed_func.embed_query(query)
        docs = self.milvus.similarity_search_with_score_by_vector(embeddings, top_k)
        return score_threshold_process(score_threshold, top_k, docs)

    # 添加文档操作，对文档进行处理后添加到 Milvus 实例中，并返回添加的文档信息
    def do_add_doc(self, docs: List[Document], **kwargs) -> List[Dict]:
        for doc in docs:
            for k, v in doc.metadata.items():
                doc.metadata[k] = str(v)
            for field in self.milvus.fields:
                doc.metadata.setdefault(field, "")
            doc.metadata.pop(self.milvus._text_field, None)
            doc.metadata.pop(self.milvus._vector_field, None)

        ids = self.milvus.add_documents(docs)
        doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
        return doc_infos
    # 删除指定知识文件对应的文档
    def do_delete_doc(self, kb_file: KnowledgeFile, **kwargs):
        # 获取知识库名称和文件名对应的文档 ID 列表
        id_list = list_file_num_docs_id_by_kb_name_and_file_name(kb_file.kb_name, kb_file.filename)
        # 如果 Milvus 集合存在
        if self.milvus.col:
            # 根据文档 ID 列表删除文档
            self.milvus.col.delete(expr=f'pk in {id_list}')

        # 修复 Issue 2846，适用于 Windows
        # 如果 Milvus 集合存在
        # file_path = kb_file.filepath.replace("\\", "\\\\")  # 将文件路径中的反斜杠替换为双反斜杠
        # file_name = os.path.basename(file_path)  # 获取文件名
        # 根据文件名查询文档 ID 列表
        # id_list = [item.get("pk") for item in self.milvus.col.query(expr=f'source == "{file_name}"', output_fields=["pk"])]
        # 根据文档 ID 列表删除文档
        # self.milvus.col.delete(expr=f'pk in {id_list}')

    # 清空向量搜索
    def do_clear_vs(self):
        # 如果 Milvus 集合存在
        if self.milvus.col:
            # 执行删除知识库和初始化操作
            self.do_drop_kb()
            self.do_init()
# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 导入建表所需的模块
    from server.db.base import Base, engine

    # 创建数据库中所有表
    Base.metadata.create_all(bind=engine)
    
    # 创建一个名为"test"的MilvusKBService对象
    milvusService = MilvusKBService("test")
    
    # 添加一个名为"README.md"内容为"test"的KnowledgeFile对象到MilvusKBService对象中
    # milvusService.add_doc(KnowledgeFile("README.md", "test"))

    # 打印根据给定id列表获取的文档
    print(milvusService.get_doc_by_ids(["444022434274215486"]))
    
    # 删除一个名为"README.md"内容为"test"的KnowledgeFile对象
    # milvusService.delete_doc(KnowledgeFile("README.md", "test"))
    
    # 删除整个知识库
    # milvusService.do_drop_kb()
    
    # 打印根据给定关键词搜索到的文档
    # print(milvusService.search_docs("如何启动api服务"))
```