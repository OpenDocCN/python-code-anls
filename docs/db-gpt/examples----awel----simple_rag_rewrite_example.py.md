# `.\DB-GPT-src\examples\awel\simple_rag_rewrite_example.py`

```py
"""AWEL: Simple rag rewrite example

    pre-requirements:
        1. install openai python sdk
        ```
            pip install openai
        ```py
        2. set openai key and base
        ```
            export OPENAI_API_KEY={your_openai_key}
            export OPENAI_API_BASE={your_openai_base}
        ```py
        or
        ```
            import os
            os.environ["OPENAI_API_KEY"] = {your_openai_key}
            os.environ["OPENAI_API_BASE"] = {your_openai_base}
        ```py
        python examples/awel/simple_rag_rewrite_example.py
    Example:

    .. code-block:: shell

        DBGPT_SERVER="http://127.0.0.1:5555"
        curl -X POST $DBGPT_SERVER/api/v1/awel/trigger/examples/rag/rewrite \
        -H "Content-Type: application/json" -d '{
            "query": "compare curry and james",
            "context":"steve curry and lebron james are nba all-stars"
        }'
"""
# 导入需要的模块和类
from typing import Dict
from dbgpt._private.pydantic import BaseModel, Field
from dbgpt.core.awel import DAG, HttpTrigger, MapOperator
from dbgpt.model.proxy import OpenAILLMClient
from dbgpt.rag.operators import QueryRewriteOperator

# 定义请求体数据模型，继承自BaseModel
class TriggerReqBody(BaseModel):
    query: str = Field(..., description="User query")  # 用户查询内容
    context: str = Field(..., description="context")  # 查询的上下文环境

# 定义请求处理操作符，继承自MapOperator，并实现map方法
class RequestHandleOperator(MapOperator[TriggerReqBody, Dict]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def map(self, input_value: TriggerReqBody) -> Dict:
        params = {
            "query": input_value.query,    # 提取用户查询
            "context": input_value.context,  # 提取查询上下文
        }
        print(f"Receive input value: {input_value}")  # 打印接收到的输入值
        return params  # 返回处理后的参数字典

# 创建DAG图，命名为dbgpt_awel_simple_rag_rewrite_example
with DAG("dbgpt_awel_simple_rag_rewrite_example") as dag:
    # 定义HTTP触发器，路径为/examples/rag/rewrite，接收POST方法，请求体为TriggerReqBody
    trigger = HttpTrigger(
        "/examples/rag/rewrite", methods="POST", request_body=TriggerReqBody
    )
    # 定义请求处理任务
    request_handle_task = RequestHandleOperator()
    # 创建查询重写操作符，使用OpenAILLMClient作为语言模型客户端，生成两个查询
    rewrite_task = QueryRewriteOperator(llm_client=OpenAILLMClient(), nums=2)
    # 设置任务之间的依赖关系
    trigger >> request_handle_task >> rewrite_task

# 如果当前脚本为主程序
if __name__ == "__main__":
    # 如果DAG图的第一个叶子节点处于开发模式
    if dag.leaf_nodes[0].dev_mode:
        # 开发模式下，设置本地调试环境
        from dbgpt.core.awel import setup_dev_environment
        setup_dev_environment([dag], port=5555)  # 设置调试环境端口为5555
    else:
        pass  # 否则不进行任何操作
```