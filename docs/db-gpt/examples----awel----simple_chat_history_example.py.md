# `.\DB-GPT-src\examples\awel\simple_chat_history_example.py`

```py
"""
AWEL: Simple chat with history example

    DB-GPT will automatically load and execute the current file after startup.

    Examples:

        Call with non-streaming response.
        .. code-block:: shell

            DBGPT_SERVER="http://127.0.0.1:5555"
            MODEL="gpt-3.5-turbo"
            # Fist round
            curl -X POST $DBGPT_SERVER/api/v1/awel/trigger/examples/simple_history/multi_round/chat/completions \
            -H "Content-Type: application/json" -d '{
                "model": "'"$MODEL"'",
                "context": {
                    "conv_uid": "uuid_conv_1234"
                },
                "messages": "Who is elon musk?"
            }'

            # Second round
            curl -X POST $DBGPT_SERVER/api/v1/awel/trigger/examples/simple_history/multi_round/chat/completions \
            -H "Content-Type: application/json" -d '{
                "model": "'"$MODEL"'",
                "context": {
                    "conv_uid": "uuid_conv_1234"
                },
                "messages": "Is he rich?"
            }'

        Call with streaming response.
        .. code-block:: shell

            curl -X POST $DBGPT_SERVER/api/v1/awel/trigger/examples/simple_history/multi_round/chat/completions \
            -H "Content-Type: application/json" -d '{
                "model": "'"$MODEL"'",
                "context": {
                    "conv_uid": "uuid_conv_stream_1234"
                },
                "stream": true,
                "messages": "Who is elon musk?"
            }'

            # Second round
            curl -X POST $DBGPT_SERVER/api/v1/awel/trigger/examples/simple_history/multi_round/chat/completions \
            -H "Content-Type: application/json" -d '{
                "model": "'"$MODEL"'",
                "context": {
                    "conv_uid": "uuid_conv_stream_1234"
                },
                "stream": true,
                "messages": "Is he rich?"
            }'


"""

import logging
from typing import Dict, List, Optional, Union

from dbgpt._private.pydantic import BaseModel, Field  # 导入 Pydantic 的 BaseModel 和 Field 类
from dbgpt.core import (  # 导入多个模块和类
    ChatPromptTemplate,
    HumanPromptTemplate,
    InMemoryStorage,
    MessagesPlaceholder,
    ModelMessage,
    ModelRequest,
    ModelRequestContext,
    SystemPromptTemplate,
)
from dbgpt.core.awel import DAG, BranchJoinOperator, HttpTrigger, MapOperator  # 导入 AWEL 相关类
from dbgpt.core.operators import (  # 导入操作符相关类
    ChatComposerInput,
    ChatHistoryPromptComposerOperator,
    LLMBranchOperator,
)
from dbgpt.model.operators import (  # 导入模型操作符相关类
    LLMOperator,
    OpenAIStreamingOutputOperator,
    StreamingLLMOperator,
)

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

class ReqContext(BaseModel):
    user_name: Optional[str] = Field(
        None, description="The user name of the model request."
    )  # 定义 ReqContext 类，包含可选的用户名字段

    sys_code: Optional[str] = Field(
        None, description="The system code of the model request."
    )  # 定义系统代码字段
    conv_uid: Optional[str] = Field(
        None, description="The conversation uid of the model request."
    )



# 定义一个可选的字符串类型变量 conv_uid，用于存储模型请求的对话 UID。
class TriggerReqBody(BaseModel):
    # 定义触发器请求体模型，继承自 BaseModel
    messages: Union[str, List[Dict[str, str]]] = Field(
        ..., description="User input messages"
    )
    model: str = Field(..., description="Model name")
    stream: Optional[bool] = Field(default=False, description="Whether return stream")
    context: Optional[ReqContext] = Field(
        default=None, description="The context of the model request."
    )


async def build_model_request(
    messages: List[ModelMessage], req_body: TriggerReqBody
) -> ModelRequest:
    # 异步函数，用于构建模型请求
    return ModelRequest.build_request(
        model=req_body.model,
        messages=messages,
        context=req_body.context,
        stream=req_body.stream,
    )


with DAG("dbgpt_awel_simple_chat_history") as multi_round_dag:
    # 创建 DAG 对象，命名为 dbgpt_awel_simple_chat_history
    # 接收 HTTP 请求并触发 DAG 运行
    trigger = HttpTrigger(
        "/examples/simple_history/multi_round/chat/completions",
        methods="POST",
        request_body=TriggerReqBody,
        streaming_predict_func=lambda req: req.stream,
    )
    # 定义聊天提示模板
    prompt = ChatPromptTemplate(
        messages=[
            SystemPromptTemplate.from_template("You are a helpful chatbot."),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanPromptTemplate.from_template("{user_input}"),
        ]
    )

    composer_operator = ChatHistoryPromptComposerOperator(
        # 聊天历史提示合成操作符
        prompt_template=prompt,
        keep_end_rounds=5,
        storage=InMemoryStorage(),
        message_storage=InMemoryStorage(),
    )

    # 使用 BaseLLMOperator 生成响应
    llm_task = LLMOperator(task_name="llm_task")
    streaming_llm_task = StreamingLLMOperator(task_name="streaming_llm_task")
    branch_task = LLMBranchOperator(
        stream_task_name="streaming_llm_task", no_stream_task_name="llm_task"
    )
    model_parse_task = MapOperator(lambda out: out.to_dict())
    openai_format_stream_task = OpenAIStreamingOutputOperator()
    result_join_task = BranchJoinOperator()

    req_handle_task = MapOperator(
        # 处理请求任务，将请求映射为 ChatComposerInput 对象
        lambda req: ChatComposerInput(
            context=ModelRequestContext(
                conv_uid=req.context.conv_uid, stream=req.stream
            ),
            prompt_dict={"user_input": req.messages},
            model_dict={
                "model": req.model,
                "context": req.context,
                "stream": req.stream,
            },
        )
    )

    trigger >> req_handle_task >> composer_operator >> branch_task

    # 非流式响应分支
    branch_task >> llm_task >> model_parse_task >> result_join_task
    # 流式响应分支
    branch_task >> streaming_llm_task >> openai_format_stream_task >> result_join_task

if __name__ == "__main__":
    if multi_round_dag.leaf_nodes[0].dev_mode:
        # 开发模式，可用于本地调试运行 DAG
        from dbgpt.core.awel import setup_dev_environment

        setup_dev_environment([multi_round_dag], port=5555)
    else:
        # 如果不是开发模式，则执行以下代码块
        # 生产模式下，DB-GPT 在启动后会自动加载并执行当前文件
        pass
```