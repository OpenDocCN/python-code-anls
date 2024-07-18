# `.\graphrag\graphrag\query\structured_search\global_search\search.py`

```py
# 引入 asyncio 模块，支持异步编程
import asyncio
# 引入 json 模块，处理 JSON 数据
import json
# 引入 logging 模块，用于记录日志
import logging
# 引入 time 模块，提供时间相关函数
import time
# 引入 dataclass 装饰器，用于定义数据类
from dataclasses import dataclass
# 引入 Any 类型提示，表示可以是任何类型
from typing import Any

# 引入 pandas 库，用于处理数据
import pandas as pd
# 引入 tiktoken 模块，可能用于编码
import tiktoken

# 从 graphrag.index.utils.json 模块中引入 clean_up_json 函数
from graphrag.index.utils.json import clean_up_json
# 从 graphrag.query.context_builder.builders 模块中引入 GlobalContextBuilder 类
from graphrag.query.context_builder.builders import GlobalContextBuilder
# 从 graphrag.query.context_builder.conversation_history 模块中引入 ConversationHistory 类
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
)
# 从 graphrag.query.llm.base 模块中引入 BaseLLM 类
from graphrag.query.llm.base import BaseLLM
# 从 graphrag.query.llm.text_utils 模块中引入 num_tokens 函数
from graphrag.query.llm.text_utils import num_tokens
# 从 graphrag.query.structured_search.base 模块中引入 BaseSearch 类和 SearchResult 类
from graphrag.query.structured_search.base import BaseSearch, SearchResult
# 从 graphrag.query.structured_search.global_search.callbacks 模块中引入 GlobalSearchLLMCallback 类
from graphrag.query.structured_search.global_search.callbacks import (
    GlobalSearchLLMCallback,
)
# 从 graphrag.query.structured_search.global_search.map_system_prompt 模块中引入 MAP_SYSTEM_PROMPT 常量
from graphrag.query.structured_search.global_search.map_system_prompt import (
    MAP_SYSTEM_PROMPT,
)
# 从 graphrag.query.structured_search.global_search.reduce_system_prompt 模块中引入多个常量和字符串
from graphrag.query.structured_search.global_search.reduce_system_prompt import (
    GENERAL_KNOWLEDGE_INSTRUCTION,
    NO_DATA_ANSWER,
    REDUCE_SYSTEM_PROMPT,
)

# 默认的 MAP LLM 参数字典
DEFAULT_MAP_LLM_PARAMS = {
    "max_tokens": 1000,
    "temperature": 0.0,
}

# 默认的 REDUCE LLM 参数字典
DEFAULT_REDUCE_LLM_PARAMS = {
    "max_tokens": 2000,
    "temperature": 0.0,
}

# 日志记录器对象
log = logging.getLogger(__name__)


@dataclass
# 继承自 SearchResult 类的 GlobalSearchResult 类，表示全局搜索的结果
class GlobalSearchResult(SearchResult):
    """A GlobalSearch result."""

    # map_responses 属性，表示映射系统的响应列表
    map_responses: list[SearchResult]
    # reduce_context_data 属性，表示缩减上下文的数据，可以是字符串、DataFrame 列表或字典
    reduce_context_data: str | list[pd.DataFrame] | dict[str, pd.DataFrame]
    # reduce_context_text 属性，表示缩减上下文的文本，可以是字符串、字符串列表或字典
    reduce_context_text: str | list[str] | dict[str, str]


# GlobalSearch 类，继承自 BaseSearch 类，表示全局搜索的实现
class GlobalSearch(BaseSearch):
    """Search orchestration for global search mode."""

    # 初始化方法
    def __init__(
        self,
        llm: BaseLLM,  # 参数 llm，表示用于处理语言模型的对象
        context_builder: GlobalContextBuilder,  # 参数 context_builder，表示全局上下文构建器对象
        token_encoder: tiktoken.Encoding | None = None,  # 参数 token_encoder，表示编码器对象或 None
        map_system_prompt: str = MAP_SYSTEM_PROMPT,  # 参数 map_system_prompt，表示映射系统提示字符串
        reduce_system_prompt: str = REDUCE_SYSTEM_PROMPT,  # 参数 reduce_system_prompt，表示缩减系统提示字符串
        response_type: str = "multiple paragraphs",  # 参数 response_type，表示响应类型，默认为多段落
        allow_general_knowledge: bool = False,  # 参数 allow_general_knowledge，是否允许通用知识
        general_knowledge_inclusion_prompt: str = GENERAL_KNOWLEDGE_INSTRUCTION,  # 参数 general_knowledge_inclusion_prompt，通用知识包含提示
        json_mode: bool = True,  # 参数 json_mode，是否启用 JSON 模式，默认为 True
        callbacks: list[GlobalSearchLLMCallback] | None = None,  # 参数 callbacks，全局搜索回调函数列表或 None
        max_data_tokens: int = 8000,  # 参数 max_data_tokens，最大数据令牌数，默认为 8000
        map_llm_params: dict[str, Any] = DEFAULT_MAP_LLM_PARAMS,  # 参数 map_llm_params，映射系统的参数字典，默认为 DEFAULT_MAP_LLM_PARAMS
        reduce_llm_params: dict[str, Any] = DEFAULT_REDUCE_LLM_PARAMS,  # 参数 reduce_llm_params，缩减系统的参数字典，默认为 DEFAULT_REDUCE_LLM_PARAMS
        context_builder_params: dict[str, Any] | None = None,  # 参数 context_builder_params，全局上下文构建器的参数字典或 None
        concurrent_coroutines: int = 32,  # 参数 concurrent_coroutines，并发协程数，默认为 32
    # 定义构造函数，继承父类__init__方法，并传入所需参数
    ):
        # 调用父类构造函数，初始化实例变量
        super().__init__(
            llm=llm,
            context_builder=context_builder,
            token_encoder=token_encoder,
            context_builder_params=context_builder_params,
        )
        # 设置映射系统提示
        self.map_system_prompt = map_system_prompt
        # 设置减少系统提示
        self.reduce_system_prompt = reduce_system_prompt
        # 设置响应类型
        self.response_type = response_type
        # 设置是否允许使用通用知识
        self.allow_general_knowledge = allow_general_knowledge
        # 设置通用知识包含提示
        self.general_knowledge_inclusion_prompt = general_knowledge_inclusion_prompt
        # 设置回调函数
        self.callbacks = callbacks
        # 设置最大数据令牌数
        self.max_data_tokens = max_data_tokens

        # 设置映射LLM参数
        self.map_llm_params = map_llm_params
        # 设置减少LLM参数
        self.reduce_llm_params = reduce_llm_params
        # 如果采用json模式，则设置响应格式为JSON对象
        if json_mode:
            self.map_llm_params["response_format"] = {"type": "json_object"}
        else:
            # 如果非json模式，则删除响应格式键
            self.map_llm_params.pop("response_format", None)

        # 初始化信号量，用于限制并发协程
        self.semaphore = asyncio.Semaphore(concurrent_coroutines)

    # 定义异步搜索函数，接收查询内容、对话历史和其他参数
    async def asearch(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs: Any,
    ) -> GlobalSearchResult:
        """
        Perform a global search.

        Global search mode includes two steps:

        - Step 1: Run parallel LLM calls on communities' short summaries to generate answer for each batch
        - Step 2: Combine the answers from step 1 to generate the final answer
        """
        # Step 1: Generate answers for each batch of community short summaries
        start_time = time.time()  # 记录开始时间
        # 调用上下文构建器构建上下文
        context_chunks, context_records = self.context_builder.build_context(
            conversation_history=conversation_history, **self.context_builder_params
        )

        # 如果有回调函数，则依次调用回调函数的开始响应方法
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_map_response_start(context_chunks)  # type: ignore
        
        # 并行调用_map_response_single_batch处理每个上下文数据块
        map_responses = await asyncio.gather(*[
            self._map_response_single_batch(
                context_data=data, query=query, **self.map_llm_params
            )
            for data in context_chunks
        ])
        
        # 如果有回调函数，则依次调用回调函数的结束响应方法
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_map_response_end(map_responses)
        
        # 计算所有map_responses中的LLM调用总数和提示令牌总数
        map_llm_calls = sum(response.llm_calls for response in map_responses)
        map_prompt_tokens = sum(response.prompt_tokens for response in map_responses)

        # Step 2: Combine the intermediate answers from step 1 to generate the final answer
        # 调用_reduce_response方法，将中间结果合并为最终结果
        reduce_response = await self._reduce_response(
            map_responses=map_responses,
            query=query,
            **self.reduce_llm_params,
        )

        # 返回一个GlobalSearchResult对象，包含最终的搜索结果和相关统计信息
        return GlobalSearchResult(
            response=reduce_response.response,
            context_data=context_records,
            context_text=context_chunks,
            map_responses=map_responses,
            reduce_context_data=reduce_response.context_data,
            reduce_context_text=reduce_response.context_text,
            completion_time=time.time() - start_time,  # 计算搜索完成时间
            llm_calls=map_llm_calls + reduce_response.llm_calls,  # 合并总的LLM调用数
            prompt_tokens=map_prompt_tokens + reduce_response.prompt_tokens,  # 合并总的提示令牌数
        )

    def search(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs: Any,
    ) -> GlobalSearchResult:
        """Perform a global search synchronously."""
        # 使用asyncio.run方法同步执行asearch方法，返回全局搜索结果
        return asyncio.run(self.asearch(query, conversation_history))

    async def _map_response_single_batch(
        self,
        context_data: str,
        query: str,
        **llm_kwargs,
    ) -> SearchResult:
        """Generate answer for a single chunk of community reports."""
        # 记录函数开始时间
        start_time = time.time()
        # 初始化搜索提示为空字符串
        search_prompt = ""
        try:
            # 根据上下文数据格式化搜索提示
            search_prompt = self.map_system_prompt.format(context_data=context_data)
            # 构建搜索消息列表，包括系统角色和用户角色的内容
            search_messages = [
                {"role": "system", "content": search_prompt},
                {"role": "user", "content": query},
            ]
            # 使用信号量进行异步访问
            async with self.semaphore:
                # 调用语言模型生成器，获取搜索响应
                search_response = await self.llm.agenerate(
                    messages=search_messages, streaming=False, **llm_kwargs
                )
                # 记录日志，输出搜索响应
                log.info("Map response: %s", search_response)
            try:
                # 解析搜索响应的 JSON 数据
                processed_response = self.parse_search_response(search_response)
            except ValueError:
                # 如果解析失败，清理 JSON 数据并重试解析
                search_response = clean_up_json(search_response)
                try:
                    # 再次尝试解析搜索响应的 JSON 数据
                    processed_response = self.parse_search_response(search_response)
                except ValueError:
                    # 如果再次失败，记录异常日志，并返回空列表
                    log.exception("Error parsing search response json")
                    processed_response = []
            
            # 返回搜索结果对象，包括解析后的响应数据、上下文数据、文本数据、完成时间、LLM 调用次数和提示令牌数
            return SearchResult(
                response=processed_response,
                context_data=context_data,
                context_text=context_data,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
            )

        except Exception:
            # 捕获并记录异常信息，并返回默认的搜索结果对象
            log.exception("Exception in _map_response_single_batch")
            return SearchResult(
                response=[{"answer": "", "score": 0}],
                context_data=context_data,
                context_text=context_data,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
            )

    def parse_search_response(self, search_response: str) -> list[dict[str, Any]]:
        """Parse the search response json and return a list of key points.

        Parameters
        ----------
        search_response: str
            The search response json string

        Returns
        -------
        list[dict[str, Any]]
            A list of key points, each key point is a dictionary with "answer" and "score" keys
        """
        # 解析搜索响应 JSON 字符串，提取每个元素的描述和分数信息，返回列表
        parsed_elements = json.loads(search_response)["points"]
        return [
            {
                "answer": element["description"],
                "score": int(element["score"]),
            }
            for element in parsed_elements
        ]

    async def _reduce_response(
        self,
        map_responses: list[SearchResult],
        query: str,
        **llm_kwargs,
```