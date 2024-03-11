# `.\Langchain-Chatchat\server\knowledge_base\kb_service\pg_kb_service.py`

```py
# 导入所需的模块
import json
from typing import List, Dict, Optional

# 导入自定义模块
from langchain.schema import Document
from langchain.vectorstores.pgvector import PGVector, DistanceStrategy
from sqlalchemy import text

# 导入配置文件
from configs import kbs_config

# 导入相关模块
from server.knowledge_base.kb_service.base import SupportedVSType, KBService, EmbeddingsFunAdapter, \
    score_threshold_process
from server.knowledge_base.utils import KnowledgeFile
import shutil
import sqlalchemy
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session

# 定义 PGKBService 类，继承自 KBService 类
class PGKBService(KBService):
    # 创建数据库引擎
    engine: Engine = sqlalchemy.create_engine(kbs_config.get("pg").get("connection_uri"), pool_size=10)

    # 加载 PG 向量
    def _load_pg_vector(self):
        self.pg_vector = PGVector(embedding_function=EmbeddingsFunAdapter(self.embed_model),
                                  collection_name=self.kb_name,
                                  distance_strategy=DistanceStrategy.EUCLIDEAN,
                                  connection=PGKBService.engine,
                                  connection_string=kbs_config.get("pg").get("connection_uri"))

    # 根据文档 ID 获取文档内容
    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        with Session(PGKBService.engine) as session:
            stmt = text("SELECT document, cmetadata FROM langchain_pg_embedding WHERE collection_id in :ids")
            results = [Document(page_content=row[0], metadata=row[1]) for row in
                       session.execute(stmt, {'ids': ids}).fetchall()]
            return results

    # 根据文档 ID 删除文档
    def del_doc_by_ids(self, ids: List[str]) -> bool:
        return super().del_doc_by_ids(ids)

    # 初始化操作
    def do_init(self):
        self._load_pg_vector()

    # 创建知识库
    def do_create_kb(self):
        pass

    # 返回向量存储类型
    def vs_type(self) -> str:
        return SupportedVSType.PG
    # 删除知识库中的数据
    def do_drop_kb(self):
        # 使用数据库会话
        with Session(PGKBService.engine) as session:
            # 执行 SQL 语句，删除 langchain_pg_embedding 表中关联到 langchain_pg_collection 表中的记录
            session.execute(text(f'''
                    -- 删除 langchain_pg_embedding 表中关联到 langchain_pg_collection 表中 的记录
                    DELETE FROM langchain_pg_embedding
                    WHERE collection_id IN (
                      SELECT uuid FROM langchain_pg_collection WHERE name = '{self.kb_name}'
                    );
                    -- 删除 langchain_pg_collection 表中 记录
                    DELETE FROM langchain_pg_collection WHERE name = '{self.kb_name}';
            '''))
            # 提交事务
            session.commit()
            # 删除知识库路径下的文件
            shutil.rmtree(self.kb_path)

    # 执行搜索操作
    def do_search(self, query: str, top_k: int, score_threshold: float):
        # 创建 EmbeddingsFunAdapter 对象
        embed_func = EmbeddingsFunAdapter(self.embed_model)
        # 获取查询的嵌入向量
        embeddings = embed_func.embed_query(query)
        # 使用嵌入向量进行相似性搜索
        docs = self.pg_vector.similarity_search_with_score_by_vector(embeddings, top_k)
        # 处理得分阈值
        return score_threshold_process(score_threshold, top_k, docs)

    # 添加文档到知识库
    def do_add_doc(self, docs: List[Document], **kwargs) -> List[Dict]:
        # 添加文档到向量索引
        ids = self.pg_vector.add_documents(docs)
        # 构建文档信息列表
        doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
        return doc_infos

    # 删除知识库中的文档
    def do_delete_doc(self, kb_file: KnowledgeFile, **kwargs):
        # 使用数据库会话
        with Session(PGKBService.engine) as session:
            # 将文件路径中的反斜杠转义
            filepath = kb_file.filepath.replace('\\', '\\\\')
            # 执行 SQL 语句，删除符合条件的记录
            session.execute(
                text(
                    ''' DELETE FROM langchain_pg_embedding WHERE cmetadata::jsonb @> '{"source": "filepath"}'::jsonb;'''.replace(
                        "filepath", filepath)))
            # 提交事务
            session.commit()

    # 清空向量索引
    def do_clear_vs(self):
        # 删除向量索引中的所有文档
        self.pg_vector.delete_collection()
        # 重新创建向量索引
        self.pg_vector.create_collection()
# 如果当前脚本作为主程序执行
if __name__ == '__main__':
    # 从 server.db.base 模块中导入 Base 类和 engine 对象
    from server.db.base import Base, engine

    # 创建数据库中所有表格
    # Base.metadata.create_all(bind=engine)
    
    # 创建一个 PGKBService 对象，传入参数 "test"
    pGKBService = PGKBService("test")
    
    # 创建知识库
    # pGKBService.create_kb()
    
    # 向知识库中添加文档，传入参数为 KnowledgeFile("README.md", "test")
    # pGKBService.add_doc(KnowledgeFile("README.md", "test"))
    
    # 从知识库中删除文档，传入参数为 KnowledgeFile("README.md", "test")
    # pGKBService.delete_doc(KnowledgeFile("README.md", "test"))
    
    # 删除整个知识库
    # pGKBService.drop_kb()
    
    # 通过文档 ID 获取文档信息，传入参数为列表 ["f1e51390-3029-4a19-90dc-7118aaa25772"]
    print(pGKBService.get_doc_by_ids(["f1e51390-3029-4a19-90dc-7118aaa25772"]))
    
    # 搜索文档，传入参数为搜索关键词 "如何启动api服务"
    # print(pGKBService.search_docs("如何启动api服务"))
```