# `.\MinerU\projects\llama_index_rag\query.py`

```
# 导入操作系统模块
import os

# 导入点击库用于命令行接口
import click
# 导入向量存储查询类型
from llama_index.core.vector_stores.types import VectorStoreQuery
# 导入嵌入相关类
from llama_index.embeddings.dashscope import (DashScopeEmbedding,
                                              DashScopeTextEmbeddingModels,
                                              DashScopeTextEmbeddingType)
# 导入 Elasticsearch 的向量存储类
from llama_index.vector_stores.elasticsearch import (AsyncDenseVectorStrategy,
                                                     ElasticsearchStore)
# 导入 Qwen 7B 模型相关类
from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# 初始化 Elasticsearch 向量存储实例
es_vector_store = ElasticsearchStore(
    index_name='rag_index',  # 设置索引名称
    es_url=os.getenv('ES_URL', 'http://127.0.0.1:9200'),  # 获取 Elasticsearch URL
    es_user=os.getenv('ES_USER', 'elastic'),  # 获取 Elasticsearch 用户名
    es_password=os.getenv('ES_PASSWORD', 'llama_index'),  # 获取 Elasticsearch 密码
    retrieval_strategy=AsyncDenseVectorStrategy(),  # 设置检索策略
)


# 定义文本嵌入函数
def embed_text(text):
    # 创建文本嵌入器实例
    embedder = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,  # 指定嵌入模型
        text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,  # 指定文本类型
    )
    # 返回文本的嵌入表示
    return embedder.get_text_embedding(text)


# 定义搜索函数
def search(vector_store: ElasticsearchStore, query: str):
    # 生成查询向量
    query_vec = VectorStoreQuery(query_embedding=embed_text(query))
    # 在向量存储中执行查询
    result = vector_store.query(query_vec)
    # 返回查询结果中所有节点的文本
    return '\n'.join([node.text for node in result.nodes])


# 定义命令行接口命令
@click.command()
@click.option(
    '-q',  # 指定命令行选项
    '--question',
    'question',  # 选项的名称
    required=True,  # 该选项为必需
    help='ask what you want to know!',  # 选项的帮助信息
)
def cli(question):
    # 从预训练模型加载分词器
    tokenizer = AutoTokenizer.from_pretrained('qwen/Qwen-7B-Chat',
                                              revision='v1.0.5',
                                              trust_remote_code=True)
    # 从预训练模型加载生成模型
    model = AutoModelForCausalLM.from_pretrained('qwen/Qwen-7B-Chat',
                                                 revision='v1.0.5',
                                                 device_map='auto',  # 自动设备映射
                                                 trust_remote_code=True,
                                                 fp32=True).eval()  # 设置模型为评估模式
    # 从预训练模型加载生成配置
    model.generation_config = GenerationConfig.from_pretrained(
        'Qwen/Qwen-7B-Chat', revision='v1.0.5', trust_remote_code=True)

    # 定义回答问题的提示模板
    def answer_question(question, context, model):
        # 根据上下文构建提示
        if context == '':
            prompt = question  # 如果没有上下文，直接使用问题
        else:
            prompt = f'''请基于```内的内容回答问题。"
            ```
            {context}  # 上下文内容
            ```
            我的问题是：{question}。  # 追加问题
            '''
        history = None  # 初始化对话历史
        print(prompt)  # 输出提示内容
        # 使用模型生成回答
        response, history = model.chat(tokenizer, prompt, history=None)
        # 返回生成的回答
        return response

    # 获取答案
    answer = answer_question(question, search(es_vector_store, question), model)
    # 输出问题和答案
    print(f'question: {question}\n'
          f'answer: {answer}')


# 运行命令行接口
"""

python query.py -q 'how about the rights of men'
"""

# 如果是主程序则调用命令行接口
if __name__ == '__main__':
    cli()
```