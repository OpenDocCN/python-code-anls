# `.\MinerU\projects\llama_index_rag\data_ingestion.py`

```
# 导入操作系统模块
import os

# 导入点击库以处理命令行选项
import click
# 从 llama_index.core.schema 导入 TextNode 类
from llama_index.core.schema import TextNode
# 从 llama_index.embeddings.dashscope 导入多个类
from llama_index.embeddings.dashscope import (DashScopeEmbedding,
                                              DashScopeTextEmbeddingModels,
                                              DashScopeTextEmbeddingType)
# 从 llama_index.vector_stores.elasticsearch 导入 ElasticsearchStore
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

# 创建 Elasticsearch 存储实例，指定索引名和连接信息
es_vec_store = ElasticsearchStore(
    index_name='rag_index',
    es_url=os.getenv('ES_URL', 'http://127.0.0.1:9200'),  # 获取环境变量中 ES_URL 的值
    es_user=os.getenv('ES_USER', 'elastic'),  # 获取环境变量中 ES_USER 的值
    es_password=os.getenv('ES_PASSWORD', 'llama_index'),  # 获取环境变量中 ES_PASSWORD 的值
)

# 创建嵌入
# text_type=`document` 用于构建索引
def embed_node(node):
    # 创建 DashScopeEmbedding 实例，指定模型和文本类型
    embedder = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
        text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
    )

    # 获取节点文本的嵌入向量
    result_embeddings = embedder.get_text_embedding(node.text)
    # 将嵌入向量赋值给节点
    node.embedding = result_embeddings
    # 返回更新后的节点
    return node

# 定义命令行接口
@click.command()
@click.option(
    '-p',
    '--path',
    'path',
    type=click.Path(exists=True),  # 确保路径存在
    required=True,  # 该选项是必需的
    help='local pdf filepath or directory',  # 帮助信息
)
def cli(path):
    # 指定输出目录
    output_dir = '/tmp/magic_pdf/integrations/rag/'
    # 创建输出目录，如果已存在则不报错
    os.makedirs(output_dir, exist_ok=True)
    # 创建 DataReader 实例，用于读取 PDF 文档
    documents = DataReader(path, 'ocr', output_dir)

    # 初始化节点列表
    nodes = []

    # 遍历文档计数
    for idx in range(documents.get_documents_count()):
        # 获取当前文档的结果
        doc = documents.get_document_result(idx)
        # 如果文档解析失败，跳过该文档
        if doc is None:  # something wrong happens when parse pdf !
            continue

        # 遍历文档中的每一页
        for page in iter(
                doc):  # iterate documents from initial page to last page !
            # 遍历当前页中的每一个元素
            for element in iter(page):  # iterate the element from all page !
                # 如果元素没有文本，跳过
                if element.text is None:
                    continue
                # 将文本节点嵌入并添加到节点列表中
                nodes.append(
                    embed_node(
                        TextNode(text=element.text,
                                 metadata={'purpose': 'demo'})))  # 创建带有元数据的 TextNode
    # 将所有节点添加到 Elasticsearch 存储中
    es_vec_store.add(nodes)

# 如果该文件是主程序，则执行命令行接口
if __name__ == '__main__':
    cli()
```