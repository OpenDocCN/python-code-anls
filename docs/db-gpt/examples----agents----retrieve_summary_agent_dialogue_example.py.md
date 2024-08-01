# `.\DB-GPT-src\examples\agents\retrieve_summary_agent_dialogue_example.py`

```py
"""Agents: single agents about CodeAssistantAgent?

    Examples:
     
        Execute the following command in the terminal:
        Set env params.
        .. code-block:: shell

            export OPENAI_API_KEY=sk-xx
            export OPENAI_API_BASE=https://xx:80/v1

        run example.
        ..code-block:: shell
            python examples/agents/retrieve_summary_agent_dialogue_example.py
"""

# 异步编程模块导入
import asyncio
# 操作系统相关功能模块导入
import os

# 调试代理相关模块导入
from dbgpt.agent import AgentContext, AgentMemory, LLMConfig, UserProxyAgent
# 调试代理中扩展的摘要检索助理代理导入
from dbgpt.agent.expand.retrieve_summary_assistant_agent import (
    RetrieveSummaryAssistantAgent,
)
# 模型配置的根路径导入
from dbgpt.configs.model_config import ROOT_PATH


async def summary_example_with_success():
    # 从模型代理中导入OpenAI语言模型客户端
    from dbgpt.model.proxy import OpenAILLMClient

    # 创建OpenAI语言模型客户端实例
    llm_client = OpenAILLMClient(model_alias="gpt-3.5-turbo-16k")
    # 创建代理上下文实例
    context: AgentContext = AgentContext(conv_id="retrieve_summarize")
    # 创建代理记忆实例
    agent_memory = AgentMemory()
    # 创建摘要检索助理代理实例
    summarizer = (
        await RetrieveSummaryAssistantAgent()
        .bind(context)  # 绑定上下文
        .bind(LLMConfig(llm_client=llm_client))  # 绑定语言模型客户端配置
        .bind(agent_memory)  # 绑定代理记忆
        .build()  # 构建代理
    )

    # 创建用户代理实例
    user_proxy = UserProxyAgent(memory=agent_memory, agent_context=context)

    # 指定文件路径和URL列表
    paths_urls = [
        os.path.join(ROOT_PATH, "examples/agents/example_files/Nuclear_power.pdf"),
        os.path.join(ROOT_PATH, "examples/agents/example_files/Taylor_Swift.pdf"),
        "https://en.wikipedia.org/wiki/Modern_Family",
        "https://en.wikipedia.org/wiki/Chernobyl_disaster",
    ]

    # 初始化对话，用户请求摘要核能优势
    await user_proxy.initiate_chat(
        recipient=summarizer,  # 摘要代理接收者
        reviewer=user_proxy,  # 评论者为用户代理自身
        message=f"I want to summarize advantages of Nuclear Power. You can refer the "
        f"following file paths and URLs: {paths_urls}",  # 消息内容包含文件路径和URL
    )

    # 打印代理记忆中与摘要相关的消息信息
    print(await agent_memory.gpts_memory.one_chat_completions("retrieve_summarize"))


if __name__ == "__main__":
    # 运行摘要示例函数
    asyncio.run(summary_example_with_success())
    # 打印摘要代理成功结果结束信息
    print(
        "\033[92m=======================The Summary Assistant with Successful Results Ended==================\n\n\033[91m"
    )
```