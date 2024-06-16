# `.\rag\simple_demo\retriever\vector_db.py`

```
# 从指定路径导入 Config 类
from base.config import Config
# 从 langchain.embeddings 模块导入 HuggingFaceEmbeddings 类
from langchain.embeddings import HuggingFaceEmbeddings
# 从 langchain.text_splitter 模块导入 CharacterTextSplitter 类
from langchain.text_splitter import CharacterTextSplitter
# 从 langchain.vectorstores 模块导入 FAISS 类
from langchain.vectorstores import FAISS

# 创建 VectorDatabase 类，继承自 Config 类
class VectorDatabase(Config):
    """FAISS database"""

    # 初始化方法
    def __init__(self) -> None:
        # 调用父类 Config 的初始化方法
        super().__init__()
        # 设置检索器为 FAISS
        self.retriever = FAISS
        # 使用配置文件中指定的参数初始化文本分割器 CharacterTextSplitter
        self.text_splitter = CharacterTextSplitter(
            chunk_size=self.config["retriever"]["passage"]["chunk_size"],  # 设置分块大小
            chunk_overlap=self.config["retriever"]["passage"]["chunk_overlap"],  # 设置分块重叠
        )

    # 方法：将文档分割成特定长度的段落
    def create_passages_from_documents(self, documents: list) -> list:
        """
        Splits the documents into passages of a certain length
        Args:
            documents (list): list of documents
        Returns:
            list: list of passages
        """
        return self.text_splitter.split_documents(documents)

    # 方法：将段落存储到向量数据库中，使用指定的编码器进行转换
    def store_passages_db(self, passages: list, encoder: HuggingFaceEmbeddings) -> None:
        """
        Store passages in vector database in embedding format
        Args:
            passages (list): list of passages
            encoder (HuggingFaceEmbeddings): encoder to convert passages into embeddings
        """
        # 使用 FAISS 检索器将段落转换为向量格式存储到数据库中
        self.db = self.retriever.from_documents(passages, encoder)

    # 方法：检索与给定问题最相似的文档
    def retrieve_most_similar_document(self, question: str, k: int) -> str:
        """
        Retrieves the most similar document for a certain question
        Args:
            question (str): user question
            k (int): number of documents to query
        Returns:
            str: most similar document
        """
        # 使用数据库中的相似性搜索方法查找与问题最相似的文档
        docs = self.db.similarity_search(question, k=k)
        # 提取搜索结果中的页面内容并组成列表
        docs = [d.page_content for d in docs]
        # 将结果列表转换为以换行符分隔的字符串形式返回
        return '\n'.join(docs)
```