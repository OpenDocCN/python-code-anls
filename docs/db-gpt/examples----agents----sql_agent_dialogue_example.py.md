# `.\DB-GPT-src\examples\agents\sql_agent_dialogue_example.py`

```py
# 引入异步IO库 asyncio
import asyncio
# 引入操作系统相关功能的库 os

# 从 dbgpt.agent 中引入 AgentContext, AgentMemory, LLMConfig, UserProxyAgent 类
from dbgpt.agent import AgentContext, AgentMemory, LLMConfig, UserProxyAgent
# 从 dbgpt.agent.expand.data_scientist_agent 中引入 DataScientistAgent 类
from dbgpt.agent.expand.data_scientist_agent import DataScientistAgent
# 从 dbgpt.agent.resource 中引入 SQLiteDBResource 类
from dbgpt.agent.resource import SQLiteDBResource
# 从 dbgpt.util.tracer 中引入 initialize_tracer 函数
from dbgpt.util.tracer import initialize_tracer

# 获取当前工作目录路径
current_dir = os.getcwd()
# 获取当前工作目录的父目录路径
parent_dir = os.path.dirname(current_dir)
# 拼接测试插件目录路径
test_plugin_dir = os.path.join(parent_dir, "test_files")

# 初始化追踪器，指定追踪日志文件路径为 /tmp/agent_trace.jsonl，并创建系统应用
initialize_tracer("/tmp/agent_trace.jsonl", create_system_app=True)


# 异步函数 main，用于执行主程序逻辑
async def main():
    # 从 dbgpt.model.proxy 中引入 OpenAILLMClient 类
    from dbgpt.model.proxy import OpenAILLMClient

    # 创建 OpenAILLMClient 对象，指定模型别名为 "gpt-3.5-turbo"
    llm_client = OpenAILLMClient(model_alias="gpt-3.5-turbo")
    # 创建 AgentContext 对象，指定会话 ID 为 "test456"
    context: AgentContext = AgentContext(conv_id="test456")
    # 创建 AgentMemory 对象
    agent_memory = AgentMemory()

    # 创建 SQLiteDBResource 对象，指定资源名称为 "SQLite Database"，数据库文件路径为 "{test_plugin_dir}/dbgpt.db"
    sqlite_resource = SQLiteDBResource("SQLite Database", f"{test_plugin_dir}/dbgpt.db")

    # 创建 UserProxyAgent 对象，并绑定 AgentMemory 和 AgentContext，然后构建代理
    user_proxy = await UserProxyAgent().bind(agent_memory).bind(context).build()

    # 创建 DataScientistAgent 对象 sql_boy，并绑定 AgentContext、LLMConfig、SQLiteDBResource 和 AgentMemory，然后构建代理
    sql_boy = (
        await DataScientistAgent()
        .bind(context)
        .bind(LLMConfig(llm_client=llm_client))
        .bind(sqlite_resource)
        .bind(agent_memory)
        .build()
    )

    # 使用 user_proxy 代理对象发起对 sql_boy 代理对象的聊天，发送消息 "当前库有那些表"
    await user_proxy.initiate_chat(
        recipient=sql_boy,
        reviewer=user_proxy,
        message="当前库有那些表",
    )

    # 打印 agent_memory 的 GPTs 记忆中 conv_id 为 "test456" 的对话完成情况
    print(await agent_memory.gpts_memory.one_chat_completions("test456"))


# 如果当前脚本作为主程序运行，则执行异步函数 main()
if __name__ == "__main__":
    asyncio.run(main())
```