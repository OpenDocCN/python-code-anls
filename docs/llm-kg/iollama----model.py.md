# `.\iollama\model.py`

```
import chromadb  # 导入 chromadb 模块，用于与 Chroma 数据库交互
import logging  # 导入 logging 模块，用于记录日志信息
import sys  # 导入 sys 模块，提供对 Python 运行时环境的访问

from llama_index.llms.ollama import Ollama  # 从 llama_index.llms.ollama 模块导入 Ollama 类
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # 从 llama_index.embeddings.huggingface 模块导入 HuggingFaceEmbedding 类
from llama_index.core import (Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate)  # 从 llama_index.core 模块导入 Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate 类/函数
from llama_index.core import StorageContext  # 从 llama_index.core 模块导入 StorageContext 类
from llama_index.vector_stores.chroma import ChromaVectorStore  # 从 llama_index.vector_stores.chroma 模块导入 ChromaVectorStore 类

import logging  # 再次导入 logging 模块，确保日志配置一致
import sys  # 再次导入 sys 模块，确保系统相关功能一致

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 配置日志输出到标准输出流，设置日志级别为 INFO，日志格式为时间、级别、消息格式化输出

global query_engine  # 声明全局变量 query_engine
query_engine = None  # 初始化 query_engine 为 None


def init_llm():
    llm = Ollama(model="llama2", request_timeout=300.0)  # 创建 Ollama 对象，使用模型 "llama2"，设置请求超时时间为 300 秒
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")  # 创建 HuggingFaceEmbedding 对象，使用模型名称 "BAAI/bge-small-en-v1.5"

    Settings.llm = llm  # 将创建的 llm 对象设置为 Settings 类的 llm 属性
    Settings.embed_model = embed_model  # 将创建的 embed_model 对象设置为 Settings 类的 embed_model 属性


def init_index(embed_model):
    reader = SimpleDirectoryReader(input_dir="./docs", recursive=True)  # 创建 SimpleDirectoryReader 对象，从指定目录 "./docs" 递归加载数据
    documents = reader.load_data()  # 加载目录中的文档数据

    logging.info("index creating with `%d` documents", len(documents))  # 记录日志信息，显示索引创建过程中加载的文档数量

    chroma_client = chromadb.EphemeralClient()  # 创建 Chroma 数据库的临时客户端对象
    chroma_collection = chroma_client.create_collection("iollama")  # 在 Chroma 数据库中创建集合 "iollama"

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)  # 创建 ChromaVectorStore 对象，使用指定的 chroma_collection
    storage_context = StorageContext.from_defaults(vector_store=vector_store)  # 创建默认的存储上下文对象，指定 vector_store

    # use this to set custom chunk size and splitting
    # https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)
    # 使用文档数据、存储上下文和嵌入模型创建 VectorStoreIndex 对象

    return index  # 返回创建的索引对象


def init_query_engine(index):
    global query_engine  # 使用全局变量 query_engine

    # custome prompt template
    template = (
        "Imagine you are an advanced AI expert in cyber security laws, with access to all current and relevant legal documents, "
        "case studies, and expert analyses. Your goal is to provide insightful, accurate, and concise answers to questions in this domain.\n\n"
        "Here is some context related to the query:\n"
        "-----------------------------------------\n"
        "{context_str}\n"
        "-----------------------------------------\n"
        "Considering the above information, please respond to the following inquiry with detailed references to applicable laws, "
        "precedents, or principles where appropriate:\n\n"
        "Question: {query_str}\n\n"
        "Answer succinctly, starting with the phrase 'According to cyber security law,' and ensure your response is understandable to someone without a legal background."
    )
    qa_template = PromptTemplate(template)  # 使用指定的模板创建 PromptTemplate 对象

    # build query engine with custom template
    # text_qa_template specifies custom template
    # similarity_top_k configure the retriever to return the top 3 most similar documents,
    # the default value of similarity_top_k is 2
    query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=3)
    # 使用指定的文本问答模板和相似度参数构建查询引擎

    return query_engine  # 返回构建的查询引擎对象


def chat(input_question, user):
    global query_engine  # 使用全局变量 query_engine

    response = query_engine.query(input_question)  # 使用查询引擎对输入的问题进行查询
    # 记录信息到日志，表示从LLM接收到响应，日志信息包含响应内容
    logging.info("got response from llm - %s", response)
    
    # 返回LLM响应对象的响应数据
    return response.response
# 定义一个命令行交互函数，用于与用户进行对话
def chat_cmd():
    # 声明全局变量，以便在函数内部访问 query_engine
    global query_engine

    # 无限循环，直到用户输入 'exit' 退出
    while True:
        # 获取用户输入的问题
        input_question = input("Enter your question (or 'exit' to quit): ")
        
        # 如果用户输入 'exit'，则跳出循环，结束程序
        if input_question.lower() == 'exit':
            break

        # 使用 query_engine 处理用户输入的问题，获取回复
        response = query_engine.query(input_question)
        
        # 记录日志，显示从 llm 获取的回复信息
        logging.info("got response from llm - %s", response)


# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 初始化 llm 模型
    init_llm()
    # 使用指定的嵌入模型参数初始化索引
    index = init_index(Settings.embed_model)
    # 初始化查询引擎，使用索引
    init_query_engine(index)
    # 进入命令行交互模式
    chat_cmd()
```