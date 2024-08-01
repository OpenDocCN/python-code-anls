# `.\DB-GPT-src\examples\sdk\simple_sdk_llm_example.py`

```py
# 导入 asyncio 库，用于异步编程
import asyncio

# 从 dbgpt.core 中导入必要的类和函数
from dbgpt.core import BaseOutputParser
from dbgpt.core.awel import DAG
from dbgpt.core.operators import (
    BaseLLMOperator,
    PromptBuilderOperator,
    RequestBuilderOperator,
)
# 从 dbgpt.model.proxy 中导入 OpenAILLMClient 类
from dbgpt.model.proxy import OpenAILLMClient

# 创建一个名为 simple_sdk_llm_example_dag 的 Directed Acyclic Graph (DAG) 实例
with DAG("simple_sdk_llm_example_dag") as dag:
    # 创建一个 PromptBuilderOperator 任务，用于生成 SQL 查询语句的提示
    prompt_task = PromptBuilderOperator(
        "Write a SQL of {dialect} to query all data of {table_name}."
    )
    # 创建一个 RequestBuilderOperator 任务，指定模型为 gpt-3.5-turbo，用于处理模型请求
    model_pre_handle_task = RequestBuilderOperator(model="gpt-3.5-turbo")
    # 创建一个 BaseLLMOperator 任务，使用 OpenAILLMClient 类作为参数，执行语言模型操作
    llm_task = BaseLLMOperator(OpenAILLMClient())
    # 创建一个 BaseOutputParser 任务，用于解析任务的输出结果
    out_parse_task = BaseOutputParser()
    
    # 配置任务之间的依赖关系，依次执行生成提示、处理模型请求、执行语言模型和解析输出
    prompt_task >> model_pre_handle_task >> llm_task >> out_parse_task

# 当该脚本作为主程序运行时
if __name__ == "__main__":
    # 使用 asyncio.run 运行 out_parse_task 的 call 方法，传入调用数据 {"dialect": "mysql", "table_name": "user"}
    output = asyncio.run(
        out_parse_task.call(call_data={"dialect": "mysql", "table_name": "user"})
    )
    # 打印输出结果
    print(f"output: \n\n{output}")
```