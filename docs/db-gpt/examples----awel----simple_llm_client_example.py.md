# `.\DB-GPT-src\examples\awel\simple_llm_client_example.py`

```py
# 导入必要的模块和库
import logging  # 导入日志模块
from typing import Any, Dict, List, Optional, Union  # 导入类型提示相关模块

from dbgpt._private.pydantic import BaseModel, Field  # 导入数据模型相关模块
from dbgpt.core import LLMClient  # 导入LLM客户端核心模块
from dbgpt.core.awel import DAG, BranchJoinOperator, HttpTrigger, MapOperator  # 导入AWEL相关核心模块
from dbgpt.core.operators import LLMBranchOperator, RequestBuilderOperator  # 导入操作符相关模块
from dbgpt.model.operators import (
    LLMOperator,
    MixinLLMOperator,
    OpenAIStreamingOutputOperator,
    StreamingLLMOperator,
)  # 导入模型操作符相关模块

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class TriggerReqBody(BaseModel):
    messages: Union[str, List[Dict[str, str]]] = Field(
        ..., description="User input messages"  # 用户输入消息，可以是字符串或包含键值对的列表
    )
    model: str = Field(..., description="Model name")  # 模型名称，必须是字符串类型
    stream: Optional[bool] = Field(default=False, description="Whether return stream")  # 是否返回流式数据，可选布尔类型，默认为False


class MyModelToolOperator(
    MixinLLMOperator, MapOperator[TriggerReqBody, Dict[str, Any]]
):
    def __init__(self, llm_client: Optional[LLMClient] = None, **kwargs):
        super().__init__(llm_client)  # 调用父类MixinLLMOperator的初始化方法
        MapOperator.__init__(self, llm_client, **kwargs)  # 调用父类MapOperator的初始化方法

    async def map(self, input_value: TriggerReqBody) -> Dict[str, Any]:
        # 使用LLM客户端计算输入消息的标记数
        prompt_tokens = await self.llm_client.count_token(
            input_value.model, input_value.messages
        )
        # 获取可用的模型列表
        available_models = await self.llm_client.models()
        # 返回包含标记数和可用模型的字典
        return {
            "prompt_tokens": prompt_tokens,
            "available_models": available_models,
        }


with DAG("dbgpt_awel_simple_llm_client_generate") as client_generate_dag:
    # 创建HTTP触发器，监听指定路径的POST请求，并根据请求是否流式返回进行设置
    trigger = HttpTrigger(
        "/examples/simple_client/chat/completions",
        methods="POST",
        request_body=TriggerReqBody,
        streaming_predict_func=lambda req: req.stream,
    )
    # 创建请求构建操作符
    request_handle_task = RequestBuilderOperator()
    # 创建一个名为 llm_task 的 LLN 操作器实例
    llm_task = LLMOperator(task_name="llm_task")
    
    # 创建一个名为 streaming_llm_task 的 StreamingLLM 操作器实例
    streaming_llm_task = StreamingLLMOperator(task_name="streaming_llm_task")
    
    # 创建一个名为 branch_task 的 LLMBranch 操作器实例，其中包含两个分支任务名：streaming_llm_task 和 llm_task
    branch_task = LLMBranchOperator(
        stream_task_name="streaming_llm_task", no_stream_task_name="llm_task"
    )
    
    # 创建一个名为 model_parse_task 的 MapOperator 实例，用于将输出转换为字典格式
    model_parse_task = MapOperator(lambda out: out.to_dict())
    
    # 创建一个名为 openai_format_stream_task 的 OpenAIStreamingOutputOperator 实例
    openai_format_stream_task = OpenAIStreamingOutputOperator()
    
    # 创建一个名为 result_join_task 的 BranchJoinOperator 实例，用于汇总分支任务的结果
    result_join_task = BranchJoinOperator()
    
    # 设置任务执行流程，依次触发 trigger 任务，然后经过 request_handle_task 到达 branch_task
    trigger >> request_handle_task >> branch_task
    
    # branch_task 分为两条分支：
    # 第一条分支经过 llm_task、model_parse_task，最终汇聚到 result_join_task
    # 第二条分支经过 streaming_llm_task、openai_format_stream_task，最终汇聚到 result_join_task
    branch_task >> llm_task >> model_parse_task >> result_join_task
    branch_task >> streaming_llm_task >> openai_format_stream_task >> result_join_task
with DAG("dbgpt_awel_simple_llm_client_count_token") as client_count_token_dag:
    # 创建一个名为 dbgpt_awel_simple_llm_client_count_token 的 Airflow DAG 对象
    # 该 DAG 用于处理来自 HTTP 请求的触发，并执行相应的任务流程

    # 创建一个 HTTP 触发器对象，用于接收 POST 请求并触发 DAG 的运行
    trigger = HttpTrigger(
        "/examples/simple_client/count_token",
        methods="POST",
        request_body=TriggerReqBody,
    )
    
    # 创建一个自定义操作符对象，用于执行模型任务
    model_task = MyModelToolOperator()
    
    # 将 HTTP 触发器与模型任务操作符连接起来，构建 DAG 的任务依赖关系
    trigger >> model_task


if __name__ == "__main__":
    if client_generate_dag.leaf_nodes[0].dev_mode:
        # 如果是开发模式，可以在本地运行 DAG 进行调试
        from dbgpt.core.awel import setup_dev_environment
        
        # 将当前 DAG 和 client_count_token_dag 添加到调试环境设置中
        dags = [client_generate_dag, client_count_token_dag]
        
        # 设置开发环境，指定调试端口为 5555
        setup_dev_environment(dags, port=5555)
    else:
        # 如果不是开发模式，说明处于生产模式
        # DB-GPT 将在启动后自动加载并执行当前文件
        pass
```