# `.\DB-GPT-src\examples\agents\custom_tool_agent_example.py`

```py
# 引入 asyncio 异步编程库，用于支持异步操作
import asyncio
# 引入 logging 日志库，用于记录程序运行时的信息
import logging
# 引入 os 模块，提供对操作系统功能的访问
import os
# 引入 sys 模块，提供对解释器的访问
import sys

# 从 typing_extensions 中引入 Annotated 和 Doc 类型注解
from typing_extensions import Annotated, Doc

# 从 dbgpt.agent 包中导入 AgentContext, AgentMemory, LLMConfig, UserProxyAgent 类
from dbgpt.agent import AgentContext, AgentMemory, LLMConfig, UserProxyAgent
# 从 dbgpt.agent.expand.tool_assistant_agent 模块中导入 ToolAssistantAgent 类
from dbgpt.agent.expand.tool_assistant_agent import ToolAssistantAgent
# 从 dbgpt.agent.resource 模块中导入 ToolPack, tool 装饰器
from dbgpt.agent.resource import ToolPack, tool

# 配置日志输出到标准输出流，设置日志级别为 INFO，定义日志格式
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


@tool
# 定义一个简单的计算器工具函数，接收两个整数和一个操作符，返回计算结果
def simple_calculator(first_number: int, second_number: int, operator: str) -> float:
    """Simple calculator tool. Just support +, -, *, /."""
    # 将输入的数字字符串转换为整数
    if isinstance(first_number, str):
        first_number = int(first_number)
    if isinstance(second_number, str):
        second_number = int(second_number)
    # 根据操作符进行相应的计算
    if operator == "+":
        return first_number + second_number
    elif operator == "-":
        return first_number - second_number
    elif operator == "*":
        return first_number * second_number
    elif operator == "/":
        return first_number / second_number
    else:
        raise ValueError(f"Invalid operator: {operator}")


@tool
# 定义一个统计目录中文件数量的工具函数，接收一个目录路径，返回该目录中文件的数量
def count_directory_files(path: Annotated[str, Doc("The directory path")]) -> int:
    """Count the number of files in a directory."""
    # 如果路径不是一个有效的目录，则抛出 ValueError 异常
    if not os.path.isdir(path):
        raise ValueError(f"Invalid directory path: {path}")
    # 返回目录中文件的数量
    return len(os.listdir(path))


async def main():
    # 从 dbgpt.model.proxy 模块中导入 OpenAILLMClient 类
    from dbgpt.model.proxy import OpenAILLMClient

    # 创建 OpenAILLMClient 实例，指定模型别名为 "gpt-3.5-turbo"
    llm_client = OpenAILLMClient(model_alias="gpt-3.5-turbo")
    # 创建 AgentContext 实例，指定会话 ID 为 "test456"
    context: AgentContext = AgentContext(conv_id="test456")

    # 创建 AgentMemory 实例
    agent_memory = AgentMemory()

    # 创建 ToolPack 实例，包含两个工具函数：simple_calculator 和 count_directory_files
    tools = ToolPack([simple_calculator, count_directory_files])

    # 创建 UserProxyAgent 实例，并绑定 agent_memory 和 context
    user_proxy = await UserProxyAgent().bind(agent_memory).bind(context).build()

    # 创建 ToolAssistantAgent 实例，并绑定 context, llm_client, agent_memory 和 tools
    tool_engineer = (
        await ToolAssistantAgent()
        .bind(context)
        .bind(LLMConfig(llm_client=llm_client))
        .bind(agent_memory)
        .bind(tools)
        .build()
    )

    # 启动用户代理和工具工程师之间的聊天，请求计算 10 和 99 的乘积
    await user_proxy.initiate_chat(
        recipient=tool_engineer,
        reviewer=user_proxy,
        message="Calculate the product of 10 and 99",
    )

    # 启动用户代理和工具工程师之间的聊天，请求统计 /tmp 目录中文件的数量
    await user_proxy.initiate_chat(
        recipient=tool_engineer,
        reviewer=user_proxy,
        message="Count the number of files in /tmp",
    )

    # 打印 agent_memory 中与会话 "test456" 相关的聊天完成信息
    print(await agent_memory.gpts_memory.one_chat_completions("test456"))


if __name__ == "__main__":
    # 运行异步主函数 main()
    asyncio.run(main())
```