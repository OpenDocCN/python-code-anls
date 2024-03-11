# `.\Langchain-Chatchat\server\knowledge_base\kb_service\zilliz_kb_service.py`

```
# 导入所需的模块和类
from typing import List, Dict, Optional
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores import Zilliz
from configs import kbs_config
from server.knowledge_base.kb_service.base import KBService, SupportedVSType, EmbeddingsFunAdapter, \
    score_threshold_process
from server.knowledge_base.utils import KnowledgeFile

# 定义 ZillizKBService 类，继承自 KBService 类
class ZillizKBService(KBService):
    zilliz: Zilliz

    # 静态方法，根据 zilliz_name 获取 Collection 对象
    @staticmethod
    def get_collection(zilliz_name):
        from pymilvus import Collection
        return Collection(zilliz_name)

    # 根据 ids 获取文档信息并返回文档列表
    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        result = []
        if self.zilliz.col:
            # ids = [int(id) for id in ids]  # for zilliz if needed #pr 2725
            # 查询符合条件的数据
            data_list = self.zilliz.col.query(expr=f'pk in {ids}', output_fields=["*"])
            for data in data_list:
                text = data.pop("text")
                result.append(Document(page_content=text, metadata=data))
        return result

    # 根据 ids 删除文档并返回是否删除成功
    def del_doc_by_ids(self, ids: List[str]) -> bool:
        self.zilliz.col.delete(expr=f'pk in {ids}')

    # 静态方法，根据 zilliz_name 和 content 进行搜索并返回结果
    @staticmethod
    def search(zilliz_name, content, limit=3):
        search_params = {
            "metric_type": "IP",
            "params": {},
        }
        c = ZillizKBService.get_collection(zilliz_name)
        return c.search(content, "embeddings", search_params, limit=limit, output_fields=["content"])

    # 创建知识库的方法，暂时为空
    def do_create_kb(self):
        pass

    # 返回支持的向量存储类型
    def vs_type(self) -> str:
        return SupportedVSType.ZILLIZ

    # 加载 Zilliz 配置信息
    def _load_zilliz(self):
        zilliz_args = kbs_config.get("zilliz")
        self.zilliz = Zilliz(embedding_function=EmbeddingsFunAdapter(self.embed_model),
                             collection_name=self.kb_name, connection_args=zilliz_args)

    # 初始化方法，加载 Zilliz 配置信息
    def do_init(self):
        self._load_zilliz()

    # 删除知识库的方法，释放 Collection 对象
    def do_drop_kb(self):
        if self.zilliz.col:
            self.zilliz.col.release()
            self.zilliz.col.drop()
    # 执行搜索操作，根据查询字符串、返回结果数量和得分阈值进行搜索
    def do_search(self, query: str, top_k: int, score_threshold: float):
        # 加载 Zilliz 实例
        self._load_zilliz()
        # 创建嵌入函数适配器，用于获取查询的嵌入向量
        embed_func = EmbeddingsFunAdapter(self.embed_model)
        # 获取查询的嵌入向量
        embeddings = embed_func.embed_query(query)
        # 使用嵌入向量进行相似度搜索，返回结果文档列表
        docs = self.zilliz.similarity_search_with_score_by_vector(embeddings, top_k)
        # 处理得分阈值，返回符合条件的文档列表
        return score_threshold_process(score_threshold, top_k, docs)

    # 添加文档到 Zilliz 索引中
    def do_add_doc(self, docs: List[Document], **kwargs) -> List[Dict]:
        # 遍历待添加的文档列表
        for doc in docs:
            # 将文档元数据中的值转换为字符串类型
            for k, v in doc.metadata.items():
                doc.metadata[k] = str(v)
            # 设置文档元数据中缺失的字段值为空字符串
            for field in self.zilliz.fields:
                doc.metadata.setdefault(field, "")
            # 移除文档元数据中的文本字段和向量字段
            doc.metadata.pop(self.zilliz._text_field, None)
            doc.metadata.pop(self.zilliz._vector_field, None)

        # 将文档添加到 Zilliz 索引中，返回添加的文档 ID 列表
        ids = self.zilliz.add_documents(docs)
        # 构建包含文档 ID 和元数据的字典列表
        doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
        return doc_infos

    # 从 Zilliz 索引中删除文档
    def do_delete_doc(self, kb_file: KnowledgeFile, **kwargs):
        # 检查 Zilliz 实例是否存在
        if self.zilliz.col:
            # 将文件路径中的反斜杠替换为双反斜杠
            filepath = kb_file.filepath.replace('\\', '\\\\')
            # 查询符合条件的文档 ID 列表
            delete_list = [item.get("pk") for item in
                           self.zilliz.col.query(expr=f'source == "{filepath}"', output_fields=["pk"])]
            # 根据文档 ID 列表删除文档
            self.zilliz.col.delete(expr=f'pk in {delete_list}')

    # 清空 Zilliz 索引中的所有文档和字段
    def do_clear_vs(self):
        # 检查 Zilliz 实例是否存在
        if self.zilliz.col:
            # 执行删除索引操作
            self.do_drop_kb()
            # 执行初始化操作
            self.do_init()
# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 从 server.db.base 模块中导入 Base 类和 engine 对象
    from server.db.base import Base, engine

    # 使用 Base 对象的 metadata 属性创建数据库中所有表
    Base.metadata.create_all(bind=engine)
    # 创建一个 ZillizKBService 对象，传入参数 "test"
    zillizService = ZillizKBService("test")
```