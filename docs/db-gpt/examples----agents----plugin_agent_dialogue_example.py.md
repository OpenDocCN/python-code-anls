# `.\DB-GPT-src\examples\agents\plugin_agent_dialogue_example.py`

```py
# 引入 asyncio 模块，用于异步编程
import asyncio
# 引入 os 模块，用于处理操作系统相关的功能
import os

# 从 dbgpt.agent 中引入 AgentContext, AgentMemory, LLMConfig, UserProxyAgent 类
from dbgpt.agent import AgentContext, AgentMemory, LLMConfig, UserProxyAgent
# 从 dbgpt.agent.expand.tool_assistant_agent 中引入 ToolAssistantAgent 类
from dbgpt.agent.expand.tool_assistant_agent import ToolAssistantAgent
# 从 dbgpt.agent.resource 中引入 AutoGPTPluginToolPack 类
from dbgpt.agent.resource import AutoGPTPluginToolPack

# 获取当前工作目录
current_dir = os.getcwd()
# 获取当前工作目录的父目录
parent_dir = os.path.dirname(current_dir)
# 拼接测试插件目录的路径
test_plugin_dir = os.path.join(parent_dir, "test_files/plugins")


# 异步函数 main，程序的入口点
async def main():
    # 从 dbgpt.model.proxy 中引入 OpenAILLMClient 类
    from dbgpt.model.proxy import OpenAILLMClient

    # 创建 OpenAILLMClient 对象，指定模型别名为 "gpt-3.5-turbo"
    llm_client = OpenAILLMClient(model_alias="gpt-3.5-turbo")
    # 创建 AgentContext 对象，设置会话 ID 为 "test456"
    context: AgentContext = AgentContext(conv_id="test456")

    # 创建 AgentMemory 对象，用于代理的记忆功能
    agent_memory = AgentMemory()

    # 创建 AutoGPTPluginToolPack 对象，加载测试插件目录中的工具包
    tools = AutoGPTPluginToolPack(test_plugin_dir)

    # 创建 UserProxyAgent 对象，并绑定代理记忆和上下文，构建用户代理
    user_proxy = await UserProxyAgent().bind(agent_memory).bind(context).build()

    # 创建 ToolAssistantAgent 对象，并绑定上下文、LLM 配置、代理记忆和工具包，构建工具助理
    tool_engineer = (
        await ToolAssistantAgent()
        .bind(context)
        .bind(LLMConfig(llm_client=llm_client))
        .bind(agent_memory)
        .bind(tools)
        .build()
    )

    # 用户代理发起对话，将消息发送给工具工程师，同时设置回顾者为用户代理
    await user_proxy.initiate_chat(
        recipient=tool_engineer,
        reviewer=user_proxy,
        message="查询今天成都的天气",
    )

    # 打印代理记忆中的对话完成信息
    print(await agent_memory.gpts_memory.one_chat_completions("test456"))


# 如果当前脚本作为主程序运行，则执行 main 函数
if __name__ == "__main__":
    asyncio.run(main())
```