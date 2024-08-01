# `.\DB-GPT-src\examples\rag\simple_rag_retriever_example.py`

```py
# 导入必要的库和模块
import os  # 导入操作系统模块
from typing import Dict, List  # 导入类型提示相关模块

from dbgpt._private.config import Config  # 导入配置相关模块
from dbgpt._private.pydantic import BaseModel, Field  # 导入数据模型相关模块
from dbgpt.configs.model_config import MODEL_PATH, PILOT_PATH  # 导入模型路径配置
from dbgpt.core import Chunk  # 导入核心模块中的Chunk类
from dbgpt.core.awel import DAG, HttpTrigger, JoinOperator, MapOperator  # 导入AWEL相关的核心类和运算符
from dbgpt.model.proxy import OpenAILLMClient  # 导入OpenAI语言模型客户端
from dbgpt.rag.embedding import DefaultEmbeddingFactory  # 导入默认的嵌入工厂
from dbgpt.rag.operators import (
    EmbeddingRetrieverOperator,  # 导入嵌入检索运算符
    QueryRewriteOperator,  # 导入查询重写运算符
    RerankOperator,  # 导入重新排名运算符
)
from dbgpt.storage.vector_store.chroma_store import ChromaStore, ChromaVectorConfig  # 导入向量存储相关模块

CFG = Config()  # 创建配置对象CFG

class TriggerReqBody(BaseModel):
    query: str = Field(..., description="User query")  # 定义请求体数据模型，包含用户查询信息

class RequestHandleOperator(MapOperator[TriggerReqBody, Dict]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # 初始化请求处理运算符

    async def map(self, input_value: TriggerReqBody) -> Dict:
        params = {
            "query": input_value.query,
        }  # 映射处理函数，将输入的查询信息映射为参数字典
        print(f"Receive input value: {input_value}")  # 打印接收到的输入值
        return params  # 返回参数字典

def _context_join_fn(context_dict: Dict, chunks: List[Chunk]) -> Dict:
    """context Join function for JoinOperator.

    Args:
        context_dict (Dict): context dict  # 上下文字典参数
        chunks (List[Chunk]): chunks  # 数据块列表参数
    Returns:
        Dict: context dict  # 返回更新后的上下文字典
    """
    context_dict["context"] = "\n".join([chunk.content for chunk in chunks])  # 将数据块列表的内容合并为上下文文本
    return context_dict  # 返回更新后的上下文字典

def _create_vector_connector():
    """Create vector connector."""
    config = ChromaVectorConfig(
        persist_path=PILOT_PATH,  # 持久化路径为PILOT_PATH
        name="embedding_rag_test",  # 向量连接器名称为embedding_rag_test
        embedding_fn=DefaultEmbeddingFactory(
            default_model_name=os.path.join(MODEL_PATH, "text2vec-large-chinese"),
        ).create(),  # 使用默认嵌入工厂创建嵌入函数
    )

    return ChromaStore(config)  # 返回基于配置创建的ChromaStore对象

with DAG("simple_sdk_rag_retriever_example") as dag:
    vector_store = _create_vector_connector()  # 创建向量存储对象
    trigger = HttpTrigger(
        "/examples/rag/retrieve", methods="POST", request_body=TriggerReqBody
    )  # 创建HTTP触发器对象，用于处理POST请求
    request_handle_task = RequestHandleOperator()  # 创建请求处理运算符实例
    query_parser = MapOperator(map_function=lambda x: x["query"])  # 创建映射运算符，提取查询信息
    context_join_operator = JoinOperator(combine_function=_context_join_fn)  # 创建连接运算符，使用上下文连接函数
    # 创建一个 QueryRewriteOperator 实例，使用 OpenAILLMClient 作为其参数
    rewrite_operator = QueryRewriteOperator(llm_client=OpenAILLMClient())
    
    # 创建一个 EmbeddingRetrieverOperator 实例，设置 top_k 参数为 3，使用 vector_store 作为索引存储
    retriever_context_operator = EmbeddingRetrieverOperator(
        top_k=3,
        index_store=vector_store,
    )
    
    # 创建另一个 EmbeddingRetrieverOperator 实例，设置 top_k 参数为 3，同样使用 vector_store 作为索引存储
    retriever_operator = EmbeddingRetrieverOperator(
        top_k=3,
        index_store=vector_store,
    )
    
    # 创建一个 RerankOperator 实例
    rerank_operator = RerankOperator()
    
    # 创建一个 MapOperator 实例，使用 lambda 函数将输出转换为字典格式
    model_parse_task = MapOperator(lambda out: out.to_dict())
    
    # 设定数据流的顺序：trigger 触发 request_handle_task，然后传递给 context_join_operator
    trigger >> request_handle_task >> context_join_operator
    
    # 设定另一条数据流的顺序：
    # trigger 触发 request_handle_task，然后经过 query_parser 处理，
    # 再传递给 retriever_context_operator，然后经过 context_join_operator
    (
        trigger
        >> request_handle_task
        >> query_parser
        >> retriever_context_operator
        >> context_join_operator
    )
    
    # 设定最后一条数据流的顺序：
    # context_join_operator 的输出传递给 rewrite_operator，
    # 然后经过 retriever_operator 处理，最终传递给 rerank_operator
    context_join_operator >> rewrite_operator >> retriever_operator >> rerank_operator
# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 检查 DAG（有向无环图）的叶子节点列表中的第一个节点是否处于开发模式
    if dag.leaf_nodes[0].dev_mode:
        # 如果处于开发模式，可以在本地运行 DAG 以进行调试
        from dbgpt.core.awel import setup_dev_environment
        
        # 设置调试环境，传入 DAG 对象列表和调试端口号
        setup_dev_environment([dag], port=5555)
    else:
        # 如果不处于开发模式，则不执行任何操作，跳过
        pass
```