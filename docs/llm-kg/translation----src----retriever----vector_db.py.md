# `.\translation\src\retriever\vector_db.py`

```
import os  # 导入操作系统相关功能模块
from pathlib import Path  # 导入路径操作相关模块

from dotenv import load_dotenv  # 导入加载环境变量的模块
from langchain.embeddings import HuggingFaceEmbeddings  # 导入使用 HuggingFace 模型的嵌入模块
from langchain.text_splitter import CharacterTextSplitter  # 导入文本拆分模块
from langchain.vectorstores.pgvector import PGVector  # 导入与 PostgreSQL 向量存储相关模块

from base.config import Config  # 导入配置类

load_dotenv(f"{Path().parent.absolute()}/env/connection.env")  # 加载环境变量文件

class VectorDatabase(Config):
    """PGVector database"""

    def __init__(self, encoder: HuggingFaceEmbeddings) -> None:
        """
        初始化函数
        Args:
            encoder (HuggingFaceEmbeddings): 用于将文档转换为嵌入表示的编码器
        """
        super().__init__()  # 调用父类的初始化函数
        self.text_splitter = CharacterTextSplitter(
            chunk_size=self.config["retriever"]["passage"]["chunk_size"],  # 初始化文本拆分器，使用配置中的块大小
            chunk_overlap=self.config["retriever"]["passage"]["chunk_overlap"],  # 使用配置中的重叠大小
        )
        self.encoder = encoder  # 设置编码器
        self.conn_str = PGVector.connection_string_from_db_params(
            driver=os.getenv("DRIVER"),  # 从环境变量获取数据库驱动程序
            host=os.getenv("HOST"),  # 从环境变量获取数据库主机
            port=os.getenv("PORT"),  # 从环境变量获取数据库端口
            database=os.getenv("DATABASE"),  # 从环境变量获取数据库名称
            user=os.getenv("USERNAME"),  # 从环境变量获取数据库用户名
            password=os.getenv("PASSWORD"),  # 从环境变量获取数据库密码
        )

    def create_passages_from_documents(self, documents: list) -> list:
        """
        将文档拆分为特定长度的段落
        Args:
            documents (list): 文档列表
        Returns:
            list: 段落列表
        """
        return self.text_splitter.split_documents(documents)  # 使用文本拆分器拆分文档

    def store_passages_db(self, passages: list, id: str) -> None:
        """
        将段落以嵌入格式存储到向量数据库中
        Args:
            passages (list): 段落列表
            id (str): 用例标识符
        """
        PGVector.from_documents(
            embedding=self.encoder,  # 使用给定的编码器进行嵌入
            documents=passages,  # 要存储的段落列表
            collection_name=id,  # 集合名称
            connection_string=self.conn_str,  # 连接字符串
            pre_delete_collection=True,  # 在存储之前删除集合
        )

    def retrieve_most_similar_document(self, question: str, k: int, id: str) -> str:
        """
        检索特定问题的最相似文档
        Args:
            question (str): 用户问题
            k (int): 要查询的文档数量
            id (str): 用例标识符
        Returns:
            str: 最相似的文档
        """
        self.db = PGVector(
            collection_name=id,  # 集合名称
            connection_string=self.conn_str,  # 连接字符串
            embedding_function=self.encoder,  # 嵌入函数
        )
        docs = self.db.similarity_search(question, k=k)  # 使用相似性搜索检索文档
        docs = [d.page_content for d in docs]  # 提取每个文档的页面内容

        return docs  # 返回最相似文档列表
```