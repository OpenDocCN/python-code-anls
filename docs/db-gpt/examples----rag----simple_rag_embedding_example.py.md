# `.\DB-GPT-src\examples\rag\simple_rag_embedding_example.py`

```py
"""AWEL: Simple rag embedding operator example.

    Examples:
    pre-requirements:
        python examples/awel/simple_rag_embedding_example.py
        ..code-block:: shell
            curl --location --request POST 'http://127.0.0.1:5555/api/v1/awel/trigger/examples/rag/embedding' \
            --header 'Content-Type: application/json' \
            --data-raw '{
              "url": "https://docs.dbgpt.site/docs/latest/awel/"
            }'
"""

import os  # 导入操作系统模块
from typing import Dict, List  # 导入类型提示相关的模块

from dbgpt._private.config import Config  # 导入私有配置模块中的 Config 类
from dbgpt._private.pydantic import BaseModel, Field  # 导入 Pydantic 相关模块
from dbgpt.configs.model_config import EMBEDDING_MODEL_CONFIG, MODEL_PATH, PILOT_PATH  # 导入模型配置相关路径
from dbgpt.core.awel import DAG, HttpTrigger, MapOperator  # 导入 AWEL 核心模块中的 DAG、HttpTrigger 和 MapOperator 类
from dbgpt.rag.embedding import DefaultEmbeddingFactory  # 导入默认嵌入工厂类
from dbgpt.rag.knowledge import KnowledgeType  # 导入知识类型模块
from dbgpt.rag.operators import EmbeddingAssemblerOperator, KnowledgeOperator  # 导入嵌入组装器操作和知识操作器
from dbgpt.storage.vector_store.chroma_store import ChromaStore, ChromaVectorConfig  # 导入向量存储相关模块

CFG = Config()  # 创建 Config 对象的实例


def _create_vector_connector():
    """Create vector connector."""
    config = ChromaVectorConfig(
        persist_path=PILOT_PATH,  # 设置持久化路径
        name="embedding_rag_test",  # 设置名称
        embedding_fn=DefaultEmbeddingFactory(
            default_model_name=os.path.join(MODEL_PATH, "text2vec-large-chinese"),  # 设置默认模型名称
        ).create(),  # 创建默认嵌入工厂实例
    )

    return ChromaStore(config)  # 返回基于配置的 ChromaStore 实例


class TriggerReqBody(BaseModel):
    url: str = Field(..., description="url")  # 定义请求体模型，包含 url 字段


class RequestHandleOperator(MapOperator[TriggerReqBody, Dict]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def map(self, input_value: TriggerReqBody) -> Dict:
        params = {
            "url": input_value.url,  # 提取请求体中的 URL 字段
        }
        print(f"Receive input value: {input_value}")  # 打印接收到的输入值
        return params  # 返回参数字典


class ResultOperator(MapOperator):
    """The Result Operator."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def map(self, chunks: List) -> str:
        result = f"embedding success, there are {len(chunks)} chunks."  # 构建成功信息字符串
        print(result)  # 打印成功信息
        return result  # 返回成功信息字符串


with DAG("simple_sdk_rag_embedding_example") as dag:  # 创建名为 "simple_sdk_rag_embedding_example" 的 DAG 对象
    trigger = HttpTrigger(
        "/examples/rag/embedding", methods="POST", request_body=TriggerReqBody  # 创建 HTTP 触发器对象，指定路径、方法和请求体类型
    )
    request_handle_task = RequestHandleOperator()  # 创建请求处理操作器实例
    knowledge_operator = KnowledgeOperator(knowledge_type=KnowledgeType.URL.name)  # 创建知识操作器实例，指定知识类型为 URL
    vector_store = _create_vector_connector()  # 创建向量存储连接器实例
    url_parser_operator = MapOperator(map_function=lambda x: x["url"])  # 创建 URL 解析操作器实例
    embedding_operator = EmbeddingAssemblerOperator(
        index_store=vector_store,  # 创建嵌入组装器操作器实例，指定索引存储为之前创建的向量存储实例
    )
    output_task = ResultOperator()  # 创建结果操作器实例
    (
        trigger
        >> request_handle_task
        >> url_parser_operator
        >> knowledge_operator
        >> embedding_operator
        >> output_task
    )  # 定义 DAG 的任务流程

if __name__ == "__main__":
    # 检查 DAG（有向无环图）的第一个叶节点是否处于开发模式
    if dag.leaf_nodes[0].dev_mode:
        # 如果处于开发模式，可以在本地运行 DAG 以进行调试。
        from dbgpt.core.awel import setup_dev_environment
        # 调用设置开发环境的函数，传入 DAG 列表和端口号为5555
        setup_dev_environment([dag], port=5555)
    else:
        # 如果不处于开发模式，则不执行任何操作。
        pass
```