# `.\Langchain-Chatchat\server\chat\knowledge_base_chat.py`

```
# 导入必要的模块和类
from fastapi import Body, Request
from sse_starlette.sse import EventSourceResponse
from fastapi.concurrency import run_in_threadpool
from configs import (LLM_MODELS, 
                     VECTOR_SEARCH_TOP_K, 
                     SCORE_THRESHOLD, 
                     TEMPERATURE,
                     USE_RERANKER,
                     RERANKER_MODEL,
                     RERANKER_MAX_LENGTH,
                     MODEL_PATH)
from server.utils import wrap_done, get_ChatOpenAI
from server.utils import BaseResponse, get_prompt_template
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBServiceFactory
import json
from urllib.parse import urlencode
from server.knowledge_base.kb_doc_api import search_docs
from server.reranker.reranker import LangchainReranker
from server.utils import embedding_device
# 异步函数，用于基于知识库进行对话
async def knowledge_base_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                              # 知识库名称参数
                              knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                              # 匹配向量数参数
                              top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                              # 知识库匹配相关度阈值参数
                              score_threshold: float = Body(
                                  SCORE_THRESHOLD,
                                  description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                                  ge=0,
                                  le=2
                              ),
                              # 历史对话参数
                              history: List[History] = Body(
                                  [],
                                  description="历史对话",
                                  examples=[[
                                      {"role": "user",
                                       "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                      {"role": "assistant",
                                       "content": "虎头虎脑"}]]
                              ),
                              # 流式输出参数
                              stream: bool = Body(False, description="流式输出"),
                              # LLM 模型名称参数
                              model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                              # LLM 采样温度参数
                              temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                              # 限制LLM生成Token数量参数
                              max_tokens: Optional[int] = Body(
                                  None,
                                  description="限制LLM生成Token数量，默认None代表模型最大值"
                              ),
                              # 使用的prompt模板名称参数
                              prompt_name: str = Body(
                                  "default",
                                  description="使用的prompt模板名称(在configs/prompt_config.py中配置)"
                              ),
                              request: Request = None,
                              ):
    # 根据知识库名称获取知识库服务
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    # 如果知识库为空，则返回未找到知识库的响应
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    # 将历史记录列表中的数据转换为 History 对象列表
    history = [History.from_data(h) for h in history]

    # 定义一个异步函数，用于生成知识库对话的迭代器
    async def knowledge_base_chat_iterator(
            query: str,
            top_k: int,
            history: Optional[List[History]],
            model_name: str = model_name,
            prompt_name: str = prompt_name,
    
    # 返回一个事件源响应，调用知识库对话迭代器函数
    return EventSourceResponse(knowledge_base_chat_iterator(query, top_k, history,model_name,prompt_name))
```