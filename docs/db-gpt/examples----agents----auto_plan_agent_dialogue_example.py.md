# `.\DB-GPT-src\examples\agents\auto_plan_agent_dialogue_example.py`

```py
"""Agents: auto plan agents example?

Examples:
 
Execute the following command in the terminal:
Set env params.
.. code-block:: shell

    export OPENAI_API_KEY=sk-xx
    export OPENAI_API_BASE=https://xx:80/v1

run example.
..code-block:: shell
    python examples/agents/auto_plan_agent_dialogue_example.py 
"""

# 导入 asyncio 库，用于异步编程
import asyncio

# 从 dbgpt.agent 模块导入所需的类
from dbgpt.agent import (
    AgentContext,
    AgentMemory,
    AutoPlanChatManager,
    LLMConfig,
    UserProxyAgent,
)
# 从 dbgpt.agent.expand.code_assistant_agent 中导入 CodeAssistantAgent 类
from dbgpt.agent.expand.code_assistant_agent import CodeAssistantAgent
# 从 dbgpt.util.tracer 中导入初始化跟踪器的函数
from dbgpt.util.tracer import initialize_tracer

# 初始化跟踪器，指定输出文件路径并创建系统应用的日志
initialize_tracer(
    "/tmp/agent_auto_plan_agent_dialogue_example_trace.jsonl", create_system_app=True
)


async def main():
    # 从 dbgpt.model.proxy 中导入 OpenAILLMClient 类
    from dbgpt.model.proxy import OpenAILLMClient

    # 创建 AgentMemory 对象
    agent_memory = AgentMemory()

    # 实例化 OpenAILLMClient，指定模型别名为 "gpt-4o"
    llm_client = OpenAILLMClient(model_alias="gpt-4o")
    # 创建 AgentContext 对象，设定对话 ID、应用名称和最大新 token 数量
    context: AgentContext = AgentContext(
        conv_id="test456", gpts_app_name="代码分析助手", max_new_tokens=2048
    )

    # 实例化 CodeAssistantAgent，并绑定上下文、LLMConfig 和 agent_memory，构建 agent
    coder = (
        await CodeAssistantAgent()
        .bind(context)
        .bind(LLMConfig(llm_client=llm_client))
        .bind(agent_memory)
        .build()
    )

    # 实例化 AutoPlanChatManager，并绑定上下文、agent_memory 和 LLMConfig，构建 manager
    manager = (
        await AutoPlanChatManager()
        .bind(context)
        .bind(agent_memory)
        .bind(LLMConfig(llm_client=llm_client))
        .build()
    )
    # 将 coder 添加到 manager 的聘用列表中
    manager.hire([coder])

    # 实例化 UserProxyAgent，并绑定上下文和 agent_memory，构建 user_proxy
    user_proxy = await UserProxyAgent().bind(context).bind(agent_memory).build()

    # 初始化对话，发送消息给 manager，并由 user_proxy 作为 reviewer
    await user_proxy.initiate_chat(
        recipient=manager,
        reviewer=user_proxy,
        message="Obtain simple information about issues in the repository 'eosphoros-ai/DB-GPT' in the past three days and analyze the data. Create a Markdown table grouped by day and status.",
        # message="Find papers on gpt-4 in the past three weeks on arxiv, and organize their titles, authors, and links into a markdown table",
        # message="find papers on LLM applications from arxiv in the last month, create a markdown table of different domains.",
    )

    # 打印对话内存中的结果
    print(await agent_memory.gpts_memory.one_chat_completions("test456"))


if __name__ == "__main__":
    ## dbgpt-vis message infos
    # 运行主函数 main()，使用 asyncio.run() 确保异步运行环境
    asyncio.run(main())
```