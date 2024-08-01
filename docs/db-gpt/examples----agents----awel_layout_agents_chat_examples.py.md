# `.\DB-GPT-src\examples\agents\awel_layout_agents_chat_examples.py`

```py
# 异步模块导入 asyncio，用于支持异步操作
import asyncio

# 从 dbgpt.agent 模块导入多个类，包括 AgentContext, AgentMemory, LLMConfig 等
from dbgpt.agent import (
    AgentContext,
    AgentMemory,
    LLMConfig,
    UserProxyAgent,
    WrappedAWELLayoutManager,
)

# 从 dbgpt.agent.expand.resources.search_tool 导入 baidu_search 函数
from dbgpt.agent.expand.resources.search_tool import baidu_search

# 从 dbgpt.agent.expand.summary_assistant_agent 导入 SummaryAssistantAgent 类
from dbgpt.agent.expand.summary_assistant_agent import SummaryAssistantAgent

# 从 dbgpt.agent.expand.tool_assistant_agent 导入 ToolAssistantAgent 类
from dbgpt.agent.expand.tool_assistant_agent import ToolAssistantAgent

# 从 dbgpt.agent.resource 导入 ToolPack 类
from dbgpt.agent.resource import ToolPack

# 从 dbgpt.util.tracer 导入 initialize_tracer 函数
from dbgpt.util.tracer import initialize_tracer

# 调用 initialize_tracer 函数，初始化追踪器，将追踪信息写入指定路径的文件中
initialize_tracer("/tmp/agent_trace.jsonl", create_system_app=True)

# 异步函数定义，程序的入口点
async def main():
    # 从 dbgpt.model.proxy 导入 OpenAILLMClient 类
    from dbgpt.model.proxy import OpenAILLMClient

    # 创建 OpenAILLMClient 实例，指定模型别名为 "gpt-3.5-turbo"
    llm_client = OpenAILLMClient(model_alias="gpt-3.5-turbo")

    # 创建 AgentContext 实例，设置会话 ID 和 GPTS 应用名称为 "信息析助手"
    context: AgentContext = AgentContext(conv_id="test456", gpts_app_name="信息析助手")

    # 创建 AgentMemory 实例，用于存储代理的记忆信息
    agent_memory = AgentMemory()

    # 创建 ToolPack 实例，包含一个名为 baidu_search 的工具
    tools = ToolPack([baidu_search])

    # 创建 ToolAssistantAgent 实例，配置代理上下文、LLM 配置、代理记忆和工具，构建工具工程师代理
    tool_engineer = (
        await ToolAssistantAgent()
        .bind(context)
        .bind(LLMConfig(llm_client=llm_client))
        .bind(agent_memory)
        .bind(tools)
        .build()
    )

    # 创建 SummaryAssistantAgent 实例，配置代理上下文、代理记忆、LLM 配置，构建摘要助手代理
    summarizer = (
        await SummaryAssistantAgent()
        .bind(context)
        .bind(agent_memory)
        .bind(LLMConfig(llm_client=llm_client))
        .build()
    )

    # 创建 WrappedAWELLayoutManager 实例，配置代理上下文、代理记忆、LLM 配置，构建布局管理器代理
    manager = (
        await WrappedAWELLayoutManager()
        .bind(context)
        .bind(agent_memory)
        .bind(LLMConfig(llm_client=llm_client))
        .build()
    )

    # 向布局管理器代理添加工具工程师代理和摘要助手代理
    manager.hire([tool_engineer, summarizer])

    # 创建 UserProxyAgent 实例，配置代理上下文和代理记忆，构建用户代理
    user_proxy = await UserProxyAgent().bind(context).bind(agent_memory).build()

    # 初始化聊天过程，发送消息给布局管理器代理
    await user_proxy.initiate_chat(
        recipient=manager,
        reviewer=user_proxy,
        message="查询北京今天天气",
        # message="查询今天的最新热点财经新闻",
        # message="Find papers on gpt-4 in the past three weeks on arxiv, and organize their titles, authors, and links into a markdown table",
        # message="find papers on LLM applications from arxiv in the last month, create a markdown table of different domains.",
    )

    # 打印代理记忆中与会话 "test456" 相关的聊天完成信息
    print(await agent_memory.gpts_memory.one_chat_completions("test456"))


# 程序入口，当脚本直接运行时执行 main 函数
if __name__ == "__main__":
    # 运行异步函数 main()
    asyncio.run(main())
```