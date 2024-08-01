# `.\DB-GPT-src\examples\agents\single_summary_agent_dialogue_example.py`

```py
# 引入 asyncio 库，支持异步编程
import asyncio

# 从 dbgpt.agent 模块中引入必要的类和函数
from dbgpt.agent import AgentContext, AgentMemory, LLMConfig, UserProxyAgent
# 从 dbgpt.agent.expand.summary_assistant_agent 模块中引入 SummaryAssistantAgent 类
from dbgpt.agent.expand.summary_assistant_agent import SummaryAssistantAgent


# 异步函数：演示成功的摘要示例
async def summary_example_with_success():
    # 从 dbgpt.model.proxy 模块中引入 OpenAILLMClient 类
    from dbgpt.model.proxy import OpenAILLMClient

    # 创建 OpenAILLMClient 对象，指定模型别名为 "gpt-3.5-turbo"
    llm_client = OpenAILLMClient(model_alias="gpt-3.5-turbo")
    # 创建 AgentContext 对象，设置对话 ID 为 "summarize"
    context: AgentContext = AgentContext(conv_id="summarize")

    # 创建 AgentMemory 对象
    agent_memory = AgentMemory()
    # 创建 SummaryAssistantAgent 实例，并绑定上下文、LLMConfig 和 agent_memory，然后构建对象
    summarizer = (
        await SummaryAssistantAgent()
        .bind(context)
        .bind(LLMConfig(llm_client=llm_client))
        .bind(agent_memory)
        .build()
    )

    # 创建 UserProxyAgent 实例，绑定 agent_memory 和 context，然后构建对象

    # dbgpt-vis message infos 打印 agent_memory 中与 "summarize" 对话相关的 GPT-3.5 模型的单条消息完成
    print(await agent_memory.gpts_memory.one_chat_completions("summarize"))


# 异步函数：演示失败的摘要示例
async def summary_example_with_faliure():
    # 从 dbgpt.model.proxy 模块中引入 OpenAILLMClient 类
    from dbgpt.model.proxy import OpenAILLMClient

    # 创建 OpenAILLMClient 对象，指定模型别名为 "gpt-3.5-turbo"
    llm_client = OpenAILLMClient(model_alias="gpt-3.5-turbo")
    # 创建 AgentContext 对象，设置对话 ID 为 "summarize"
    context: AgentContext = AgentContext(conv_id="summarize")

    # 创建 AgentMemory 对象
    agent_memory = AgentMemory()
    # 创建 SummaryAssistantAgent 实例，并绑定上下文、LLMConfig 和 agent_memory，然后构建对象
    summarizer = (
        await SummaryAssistantAgent()
        .bind(context)
        .bind(LLMConfig(llm_client=llm_client))
        .bind(agent_memory)
        .build()
    )

    # 创建 UserProxyAgent 实例，绑定 agent_memory 和 context，然后构建对象

    # 测试失败示例
    # 使用 user_proxy 对象初始化一个聊天会话，向 summarizer 发送消息并指定 reviewer 为 user_proxy 自身
    await user_proxy.initiate_chat(
        recipient=summarizer,
        reviewer=user_proxy,
        message="""I want to summarize advantages of Nuclear Power according to the following content.
    
            Taylor Swift is an American singer-songwriter and actress who is one of the most prominent and successful figures in the music industry. She was born on December 13, 1989, in Reading, Pennsylvania, USA. Taylor Swift gained widespread recognition for her narrative songwriting style, which often draws from her personal experiences and relationships.
    
            Swift's career began in country music, and her self-titled debut album was released in 2006. She quickly became a sensation in the country music scene with hits like "Tim McGraw" and "Teardrops on My Guitar." However, it was her transition to pop music with albums like "Fearless," "Speak Now," and "Red" that catapulted her to international superstardom.
    
            Throughout her career, Taylor Swift has won numerous awards, including multiple Grammy Awards. Her albums consistently top charts, and her songs resonate with a wide audience due to their relatable lyrics and catchy melodies. Some of her most famous songs include "Love Story," "Blank Space," "Shake It Off," "Bad Blood," and "Lover."
    
            Beyond music, Taylor Swift has ventured into acting with roles in movies like "Valentine's Day" and "The Giver." She is also known for her philanthropic efforts and her willingness to use her platform to advocate for various causes.
    
            Taylor Swift is not only a successful artist but also an influential cultural icon known for her evolving musical style, storytelling abilities, and her impact on the entertainment industry.
            """,
    )
    
    # 打印 agent_memory.gpts_memory 中关于 "summarize" 的单次聊天完成结果
    print(await agent_memory.gpts_memory.one_chat_completions("summarize"))
if __name__ == "__main__":
    # 检查当前模块是否作为主程序运行
    print(
        "\033[92m=======================Start The Summary Assistant with Successful Results==================\033[0m"
    )
    # 打印成功结果的起始信息

    # 运行异步函数 summary_example_with_success，并等待其完成
    asyncio.run(summary_example_with_success())

    # 打印成功结果的结束信息
    print(
        "\033[92m=======================The Summary Assistant with Successful Results Ended==================\n\n\033[91m"
    )

    # 打印失败结果的起始信息
    print(
        "\033[91m=======================Start The Summary Assistant with Fail Results==================\033[91m"
    )

    # 运行异步函数 summary_example_with_failure，并等待其完成
    asyncio.run(summary_example_with_failure())

    # 打印失败结果的结束信息
    print(
        "\033[91m=======================The Summary Assistant with Fail Results Ended==================\033[91m"
    )
```