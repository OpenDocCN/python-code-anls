# `.\Langchain-Chatchat\server\knowledge_base\kb_service\es_kb_service.py`

```
# 导入所需的模块
from typing import List
import os
import shutil
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from configs import KB_ROOT_PATH, EMBEDDING_MODEL, EMBEDDING_DEVICE, CACHED_VS_NUM
from server.knowledge_base.kb_service.base import KBService, SupportedVSType
from server.knowledge_base.utils import KnowledgeFile
from server.utils import load_local_embeddings
from elasticsearch import Elasticsearch, BadRequestError
from configs import logger
from configs import kbs_config

# 定义一个类 ESKBService，继承自 KBService 类
class ESKBService(KBService):

    # 静态方法，根据知识库名称获取知识库路径
    @staticmethod
    def get_kb_path(knowledge_base_name: str):
        return os.path.join(KB_ROOT_PATH, knowledge_base_name)

    # 静态方法，根据知识库名称获取向量存储路径
    @staticmethod
    def get_vs_path(knowledge_base_name: str):
        return os.path.join(ESKBService.get_kb_path(knowledge_base_name), "vector_store")

    # 创建知识库的方法
    def do_create_kb(self):
        # 如果文档路径存在
        if os.path.exists(self.doc_path):
            # 如果向量存储路径不存在，则创建
            if not os.path.exists(os.path.join(self.kb_path, "vector_store")):
                os.makedirs(os.path.join(self.kb_path, "vector_store"))
            else:
                logger.warning("directory `vector_store` already exists.")

    # 返回向量存储类型为 ES（Elasticsearch）
    def vs_type(self) -> str:
        return SupportedVSType.ES
    # 加载文档到Elasticsearch中
    def _load_es(self, docs, embed_model):
        # 尝试连接并写入文档
        try:
            # 如果有用户名和密码，则使用用户名和密码连接Elasticsearch并写入文档
            if self.user != "" and self.password != "":
                self.db = ElasticsearchStore.from_documents(
                        documents=docs,
                        embedding=embed_model,
                        es_url= f"http://{self.IP}:{self.PORT}",
                        index_name=self.index_name,
                        distance_strategy="COSINE",
                        query_field="context",
                        vector_query_field="dense_vector",
                        verify_certs=False,
                        es_user=self.user,
                        es_password=self.password
                    )
            # 如果没有用户名和密码，则直接连接Elasticsearch并写入文档
            else:
                self.db = ElasticsearchStore.from_documents(
                        documents=docs,
                        embedding=embed_model,
                        es_url= f"http://{self.IP}:{self.PORT}",
                        index_name=self.index_name,
                        distance_strategy="COSINE",
                        query_field="context",
                        vector_query_field="dense_vector",
                        verify_certs=False)
        except ConnectionError as ce:
            # 捕获连接错误并打印错误信息
            print(ce)
            print("连接到 Elasticsearch 失败！")
            logger.error("连接到 Elasticsearch 失败！")
        except Exception as e:
            # 捕获其他异常并记录错误信息
            logger.error(f"Error 发生 : {e}")
            print(e)

    # 执行文本相似性检索
    def do_search(self, query:str, top_k: int, score_threshold: float):
        # 使用初始化的Elasticsearch实例进行相似性检索，并返回结果文档
        docs = self.db_init.similarity_search_with_score(query=query,
                                         k=top_k)
        return docs
    # 根据给定的文档 ID 列表获取文档对象列表
    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        # 初始化结果列表
        results = []
        # 遍历文档 ID 列表
        for doc_id in ids:
            try:
                # 从 Elasticsearch 中获取文档信息
                response = self.es_client_python.get(index=self.index_name, id=doc_id)
                # 获取文档的内容和元数据
                source = response["_source"]
                # 假设文档包含 "context" 和 "metadata" 字段
                text = source.get("context", "")
                metadata = source.get("metadata", {})
                # 将文档内容和元数据添加到结果列表中
                results.append(Document(page_content=text, metadata=metadata))
            except Exception as e:
                # 捕获异常并记录错误日志
                logger.error(f"Error retrieving document from Elasticsearch! {e}")
        # 返回结果列表
        return results

    # 根据给定的文档 ID 列表删除文档
    def del_doc_by_ids(self, ids: List[str]) -> bool:
        # 遍历文档 ID 列表
        for doc_id in ids:
            try:
                # 从 Elasticsearch 中删除指定文档
                self.es_client_python.delete(index=self.index_name,
                                            id=doc_id,
                                            refresh=True)
            except Exception as e:
                # 捕获异常并记录错误日志
                logger.error(f"ES Docs Delete Error! {e}")
    # 删除文档的方法，接受一个知识库文件和其他关键字参数
    def do_delete_doc(self, kb_file, **kwargs):
        # 检查是否存在指定的索引
        if self.es_client_python.indices.exists(index=self.index_name):
            # 构建查询条件，根据文件路径删除文档
            query = {
                "query": {
                    "term": {
                        "metadata.source.keyword": kb_file.filepath
                    }
                }
            }
            # 执行查询，设置返回结果的数量为50
            search_results = self.es_client_python.search(body=query, size=50)
            # 获取需要删除的文档的 ID 列表
            delete_list = [hit["_id"] for hit in search_results['hits']['hits']]
            # 如果没有需要删除的文档，则返回 None
            if len(delete_list) == 0:
                return None
            else:
                # 遍历需要删除的文档 ID 列表
                for doc_id in delete_list:
                    try:
                        # 删除指定文档
                        self.es_client_python.delete(index=self.index_name,
                                                     id=doc_id,
                                                     refresh=True)
                    except Exception as e:
                        # 捕获异常并记录错误日志
                        logger.error(f"ES Docs Delete Error! {e}")

            # 删除文档后，可以执行其他操作，比如删除数据库中的记录或刷新索引
            # self.db_init.delete(ids=delete_list)
            #self.es_client_python.indices.refresh(index=self.index_name)
    # 向知识库添加文件
    def do_add_doc(self, docs: List[Document], **kwargs):
        '''向知识库添加文件'''
        # 打印输入的docs参数长度
        print(f"server.knowledge_base.kb_service.es_kb_service.do_add_doc 输入的docs参数长度为:{len(docs)}")
        print("*"*100)
        # 调用_load_es方法加载数据到ES中
        self._load_es(docs=docs, embed_model=self.embeddings_model)
        # 获取 id 和 source , 格式：[{"id": str, "metadata": dict}, ...]
        print("写入数据成功.")
        print("*"*100)

        # 检查ES中是否存在指定的索引
        if self.es_client_python.indices.exists(index=self.index_name):
            file_path = docs[0].metadata.get("source")
            # 构建查询条件
            query = {
                "query": {
                    "term": {
                        "metadata.source.keyword": file_path
                    },
                    "term": {
                        "_index": self.index_name
                    }
                }
            }
            # 注意设置size，默认返回10个。
            # 执行查询
            search_results = self.es_client_python.search(body=query, size=50)
            # 如果查询结果为空，则抛出异常
            if len(search_results["hits"]["hits"]) == 0:
                raise ValueError("召回元素个数为0")
        # 从查询结果中提取信息并返回
        info_docs = [{"id":hit["_id"], "metadata": hit["_source"]["metadata"]} for hit in search_results["hits"]["hits"]]
        return info_docs


    # 从知识库删除全部向量
    def do_clear_vs(self):
        """从知识库删除全部向量"""
        # 检查ES中是否存在指定的索引，如果存在则删除
        if self.es_client_python.indices.exists(index=self.kb_name):
            self.es_client_python.indices.delete(index=self.kb_name)


    # 删除知识库
    def do_drop_kb(self):
        """删除知识库"""
        # 检查知识库路径是否存在，如果存在则删除
        if os.path.exists(self.kb_path):
            shutil.rmtree(self.kb_path)
# 如果当前脚本作为主程序执行
if __name__ == '__main__':
    # 创建一个名为"test"的ESKBService对象
    esKBService = ESKBService("test")
    # 调用清空知识库的方法（已被注释掉）
    #esKBService.clear_vs()
    # 调用创建知识库的方法
    #esKBService.create_kb()
    # 向知识库中添加一个名为"README.md"的文档
    esKBService.add_doc(KnowledgeFile(filename="README.md", knowledge_base_name="test"))
    # 打印搜索结果，搜索关键词为"如何启动api服务"
    print(esKBService.search_docs("如何启动api服务"))
```