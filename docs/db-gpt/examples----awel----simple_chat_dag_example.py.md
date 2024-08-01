# `.\DB-GPT-src\examples\awel\simple_chat_dag_example.py`

```py
# 导入必要的模块和类
from dbgpt._private.pydantic import BaseModel, Field  # 导入基础模型和字段类
from dbgpt.core import ModelMessage, ModelRequest  # 导入模型消息和请求类
from dbgpt.core.awel import DAG, HttpTrigger, MapOperator  # 导入DAG、Http触发器和映射操作符类
from dbgpt.model.operators import LLMOperator  # 导入LLM操作符类


class TriggerReqBody(BaseModel):
    model: str = Field(..., description="Model name")  # 定义模型名称字段
    user_input: str = Field(..., description="User input")  # 定义用户输入字段


class RequestHandleOperator(MapOperator[TriggerReqBody, ModelRequest]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def map(self, input_value: TriggerReqBody) -> ModelRequest:
        messages = [ModelMessage.build_human_message(input_value.user_input)]  # 创建用户消息列表
        print(f"Receive input value: {input_value}")  # 打印接收到的输入值
        return ModelRequest.build_request(input_value.model, messages)  # 构建模型请求对象


# 创建名为'dbgpt_awel_simple_dag_example'的DAG对象，设置标签为'example'
with DAG("dbgpt_awel_simple_dag_example", tags={"label": "example"}) as dag:
    # 创建HTTP触发器，监听'/examples/simple_chat'路径的POST请求，使用TriggerReqBody作为请求体
    trigger = HttpTrigger(
        "/examples/simple_chat", methods="POST", request_body=TriggerReqBody
    )
    # 创建请求处理操作符实例
    request_handle_task = RequestHandleOperator()
    # 创建LLM操作符实例，任务名为'llm_task'
    llm_task = LLMOperator(task_name="llm_task")
    # 创建映射操作符实例，将输出转换为字典
    model_parse_task = MapOperator(lambda out: out.to_dict())

    # 配置DAG任务依赖关系：触发器触发请求处理任务，然后依次执行LLM任务和模型解析任务
    trigger >> request_handle_task >> llm_task >> model_parse_task


if __name__ == "__main__":
    if dag.leaf_nodes[0].dev_mode:
        # 如果处于开发模式，则设置开发环境并运行DAG进行调试
        from dbgpt.core.awel import setup_dev_environment

        setup_dev_environment([dag], port=5555)
    else:
        # 如果处于生产模式，则DB-GPT将在启动后自动加载并执行当前文件
        pass
```