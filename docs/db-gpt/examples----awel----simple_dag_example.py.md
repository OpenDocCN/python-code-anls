# `.\DB-GPT-src\examples\awel\simple_dag_example.py`

```py
"""AWEL: Simple dag example

    DB-GPT will automatically load and execute the current file after startup.

    Example:

    .. code-block:: shell

        DBGPT_SERVER="http://127.0.0.1:5555"
        curl -X GET $DBGPT_SERVER/api/v1/awel/trigger/examples/hello\?name\=zhangsan

"""
# 导入必要的模块和类
from dbgpt._private.pydantic import BaseModel, Field  # 导入 Pydantic 的 BaseModel 和 Field
from dbgpt.core.awel import DAG, HttpTrigger, MapOperator  # 导入 AWEL 框架的 DAG, HttpTrigger 和 MapOperator 类


class TriggerReqBody(BaseModel):
    name: str = Field(..., description="User name")  # 定义 TriggerReqBody 类，包含一个字符串类型的 name 字段
    age: int = Field(18, description="User age")  # 包含一个整数类型的 age 字段，默认值为 18


class RequestHandleOperator(MapOperator[TriggerReqBody, str]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # 定义 RequestHandleOperator 类，继承自 MapOperator，用于处理输入数据

    async def map(self, input_value: TriggerReqBody) -> str:
        print(f"Receive input value: {input_value}")  # 异步方法 map，接收 TriggerReqBody 类型的输入，返回字符串
        return f"Hello, {input_value.name}, your age is {input_value.age}"


with DAG("simple_dag_example") as dag:  # 创建名为 simple_dag_example 的 DAG 对象
    trigger = HttpTrigger("/examples/hello", request_body=TriggerReqBody)  # 创建 HTTP 触发器对象 trigger
    map_node = RequestHandleOperator()  # 创建处理节点对象 map_node
    trigger >> map_node  # 将 trigger 设置为 map_node 的输入

if __name__ == "__main__":  # 如果当前脚本作为主程序执行
    if dag.leaf_nodes[0].dev_mode:  # 如果 DAG 的第一个叶节点处于开发模式
        from dbgpt.core.awel import setup_dev_environment  # 导入设置开发环境的函数

        setup_dev_environment([dag])  # 设置开发环境，传入 DAG 对象列表
    else:
        pass  # 否则不执行任何操作
```