# `.\Langchain-Chatchat\server\chat\agent_chat.py`

```
# 导入所需的模块
import json
import asyncio

# 从 fastapi 框架中导入 Body 类
from fastapi import Body
# 从 sse_starlette 库中导入 EventSourceResponse 类
from sse_starlette.sse import EventSourceResponse
# 从 configs 模块中导入 LLM_MODELS, TEMPERATURE, HISTORY_LEN, Agent_MODEL 变量
from configs import LLM_MODELS, TEMPERATURE, HISTORY_LEN, Agent_MODEL

# 从 langchain.chains 模块中导入 LLMChain 类
from langchain.chains import LLMChain
# 从 langchain.memory 模块中导入 ConversationBufferWindowMemory 类
from langchain.memory import ConversationBufferWindowMemory
# 从 langchain.agents 模块中导入 LLMSingleActionAgent, AgentExecutor 类
from langchain.agents import LLMSingleActionAgent, AgentExecutor
# 从 typing 模块中导入 AsyncIterable, Optional, List 类型
from typing import AsyncIterable, Optional, List

# 从 server.utils 模块中导入 wrap_done, get_ChatOpenAI, get_prompt_template 函数
from server.utils import wrap_done, get_ChatOpenAI, get_prompt_template
# 从 server.knowledge_base.kb_service.base 模块中导入 get_kb_details 函数
from server.knowledge_base.kb_service.base import get_kb_details
# 从 server.agent.custom_agent.ChatGLM3Agent 模块中导入 initialize_glm3_agent 函数
from server.agent.custom_agent.ChatGLM3Agent import initialize_glm3_agent
# 从 server.agent.tools_select 模块中导入 tools, tool_names 变量
from server.agent.tools_select import tools, tool_names
# 从 server.agent.callbacks 模块中导入 CustomAsyncIteratorCallbackHandler, Status 类
from server.agent.callbacks import CustomAsyncIteratorCallbackHandler, Status
# 从 server.chat.utils 模块中导入 History 类
from server.chat.utils import History
# 从 server.agent 模块中导入 model_container 变量
from server.agent import model_container
# 从 server.agent.custom_template 模块中导入 CustomOutputParser, CustomPromptTemplate 类
from server.agent.custom_template import CustomOutputParser, CustomPromptTemplate
# 定义一个异步函数 agent_chat，接收多个参数
async def agent_chat(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
                     history: List[History] = Body([],
                                                   description="历史对话",
                                                   examples=[[
                                                       {"role": "user", "content": "请使用知识库工具查询今天北京天气"},
                                                       {"role": "assistant",
                                                        "content": "使用天气查询工具查询到今天北京多云，10-14摄氏度，东北风2级，易感冒"}]]
                                                   ),
                     stream: bool = Body(False, description="流式输出"),
                     model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                     temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                     max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
                     prompt_name: str = Body("default",
                                             description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                     ):
    # 将历史对话数据转换为 History 对象列表
    history = [History.from_data(h) for h in history]

    # 定义一个内部异步函数 agent_chat_iterator，接收多个参数
    async def agent_chat_iterator(
            query: str,
            history: Optional[List[History]],
            model_name: str = LLM_MODELS[0],
            prompt_name: str = prompt_name,
    # 返回一个 EventSourceResponse 对象，调用 agent_chat_iterator 函数
    return EventSourceResponse(agent_chat_iterator(query=query,
                                                   history=history,
                                                   model_name=model_name,
                                                   prompt_name=prompt_name),
                               )
```