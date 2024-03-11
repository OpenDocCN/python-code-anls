# `.\Langchain-Chatchat\server\chat\completion.py`

```
# 从 fastapi 模块中导入 Body 类，用于定义请求体参数
from fastapi import Body
# 从 sse_starlette.sse 模块中导入 EventSourceResponse 类，用于服务器端发送事件流响应
from sse_starlette.sse import EventSourceResponse
# 从 configs 模块中导入 LLM_MODELS 和 TEMPERATURE 常量
from configs import LLM_MODELS, TEMPERATURE
# 从 server.utils 模块中导入 wrap_done 和 get_OpenAI 函数
from server.utils import wrap_done, get_OpenAI
# 从 langchain.chains 模块中导入 LLMChain 类
from langchain.chains import LLMChain
# 从 langchain.callbacks 模块中导入 AsyncIteratorCallbackHandler 类
from langchain.callbacks import AsyncIteratorCallbackHandler
# 从 typing 模块中导入 AsyncIterable 和 Optional 类型
from typing import AsyncIterable, Optional
# 导入 asyncio 模块，用于异步编程
import asyncio
# 从 langchain.prompts 模块中导入 PromptTemplate 类
from langchain.prompts import PromptTemplate

# 从 server.utils 模块中导入 get_prompt_template 函数
from server.utils import get_prompt_template

# 定义异步函数 completion，接收多个请求体参数
async def completion(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
                     stream: bool = Body(False, description="流式输出"),
                     echo: bool = Body(False, description="除了输出之外，还回显输入"),
                     model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                     temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                     max_tokens: Optional[int] = Body(1024, description="限制LLM生成Token数量，默认None代表模型最大值"),
                     # top_p: float = Body(TOP_P, description="LLM 核采样。勿与temperature同时设置", gt=0.0, lt=1.0),
                     prompt_name: str = Body("default",
                                             description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                     ):

    # 提示：因ApiModelWorker 默认是按chat处理的，会对params["prompt"] 解析为messages，因此ApiModelWorker 使用时需要有相应处理
    # 定义一个异步迭代器函数，用于生成完成的文本
    async def completion_iterator(query: str,
                                  model_name: str = LLM_MODELS[0],
                                  prompt_name: str = prompt_name,
                                  echo: bool = echo,
                                  ) -> AsyncIterable[str]:
        # 使用 nonlocal 关键字声明 max_tokens 变量为非局部变量
        nonlocal max_tokens
        # 如果 max_tokens 是整数且小于等于0，则将其设为 None
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        # 获取 OpenAI 模型对象
        model = get_OpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
            echo=echo
        )

        # 获取用于生成提示的模板
        prompt_template = get_prompt_template("completion", prompt_name)
        prompt = PromptTemplate.from_template(prompt_template)
        # 创建 LLMChain 对象
        chain = LLMChain(prompt=prompt, llm=model)

        # 开始一个在后台运行的任务
        task = asyncio.create_task(wrap_done(
            chain.acall({"input": query}),
            callback.done),
        )

        # 如果 stream 为真，则使用服务器发送事件来流式传输响应
        if stream:
            async for token in callback.aiter():
                yield token
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield answer

        # 等待任务完成
        await task

    # 返回一个 EventSourceResponse 对象，用于处理 completion_iterator 生成的异步迭代器
    return EventSourceResponse(completion_iterator(query=query,
                                                 model_name=model_name,
                                                 prompt_name=prompt_name),
                             )
```