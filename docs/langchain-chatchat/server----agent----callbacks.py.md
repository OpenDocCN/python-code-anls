# `.\Langchain-Chatchat\server\agent\callbacks.py`

```py
# 导入必要的模块和类
from __future__ import annotations
from uuid import UUID
from langchain.callbacks import AsyncIteratorCallbackHandler
import json
import asyncio
from typing import Any, Dict, List, Optional

from langchain.schema import AgentFinish, AgentAction
from langchain.schema.output import LLMResult

# 定义一个将字典对象转换为字符串的函数
def dumps(obj: Dict) -> str:
    return json.dumps(obj, ensure_ascii=False)

# 定义一个状态类，包含不同状态的常量值
class Status:
    start: int = 1
    running: int = 2
    complete: int = 3
    agent_action: int = 4
    agent_finish: int = 5
    error: int = 6
    tool_finish: int = 7

# 自定义异步迭代器回调处理类，继承自AsyncIteratorCallbackHandler
class CustomAsyncIteratorCallbackHandler(AsyncIteratorCallbackHandler):
    def __init__(self):
        super().__init__()
        # 初始化队列和事件
        self.queue = asyncio.Queue()
        self.done = asyncio.Event()
        self.cur_tool = {}
        self.out = True

    # 异步处理工具启动事件
    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, *, run_id: UUID,
                            parent_run_id: UUID | None = None, tags: List[str] | None = None,
                            metadata: Dict[str, Any] | None = None, **kwargs: Any) -> None:

        # 对于截断不能自理的大模型，我来帮他截断
        stop_words = ["Observation:", "Thought","\"","（", "\n","\t"]
        for stop_word in stop_words:
            index = input_str.find(stop_word)
            if index != -1:
                input_str = input_str[:index]
                break

        # 设置当前工具的信息
        self.cur_tool = {
            "tool_name": serialized["name"],
            "input_str": input_str,
            "output_str": "",
            "status": Status.agent_action,
            "run_id": run_id.hex,
            "llm_token": "",
            "final_answer": "",
            "error": "",
        }
        # 将当前工具信息转换为字符串并放入队列中
        self.queue.put_nowait(dumps(self.cur_tool))
    # 异步处理工具结束事件，更新工具状态为完成，替换输出中的"Answer:"字符串，将更新后的工具信息加入队列
    async def on_tool_end(self, output: str, *, run_id: UUID, parent_run_id: UUID | None = None,
                          tags: List[str] | None = None, **kwargs: Any) -> None:
        self.out = True ## 重置输出
        self.cur_tool.update(
            status=Status.tool_finish,
            output_str=output.replace("Answer:", ""),
        )
        self.queue.put_nowait(dumps(self.cur_tool))

    # 异步处理工具错误事件，更新工具状态为错误，记录错误信息，将更新后的工具信息加入队列
    async def on_tool_error(self, error: Exception | KeyboardInterrupt, *, run_id: UUID,
                            parent_run_id: UUID | None = None, tags: List[str] | None = None, **kwargs: Any) -> None:
        self.cur_tool.update(
            status=Status.error,
            error=str(error),
        )
        self.queue.put_nowait(dumps(self.cur_tool))

    # async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
    #     if "Action" in token: ## 减少重复输出
    #         before_action = token.split("Action")[0]
    #         self.cur_tool.update(
    #             status=Status.running,
    #             llm_token=before_action + "\n",
    #         )
    #         self.queue.put_nowait(dumps(self.cur_tool))
    #
    #         self.out = False
    #
    #     if token and self.out:
    #         self.cur_tool.update(
    #                 status=Status.running,
    #                 llm_token=token,
    #         )
    #         self.queue.put_nowait(dumps(self.cur_tool))
    # 当接收到新的 LLAMA 令牌时触发的异步函数，处理特殊令牌并更新当前工具状态
    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        # 定义特殊令牌列表
        special_tokens = ["Action", "<|observation|>"]
        # 遍历特殊令牌列表
        for stoken in special_tokens:
            # 如果特殊令牌在接收到的令牌中
            if stoken in token:
                # 根据特殊令牌拆分令牌，获取特殊令牌之前的部分
                before_action = token.split(stoken)[0]
                # 更新当前工具状态，设置 LLAMA 令牌为特殊令牌之前的部分
                self.cur_tool.update(
                    status=Status.running,
                    llm_token=before_action + "\n",
                )
                # 将更新后的工具状态加入队列
                self.queue.put_nowait(dumps(self.cur_tool))
                # 设置输出标志为 False，表示已处理特殊令牌
                self.out = False
                # 跳出循环
                break

        # 如果令牌不为空且输出标志为 True
        if token and self.out:
            # 更新当前工具状态，设置 LLAMA 令牌为接收到的令牌
            self.cur_tool.update(
                status=Status.running,
                llm_token=token,
            )
            # 将更新后的工具状态加入队列
            self.queue.put_nowait(dumps(self.cur_tool))

    # 当 LLAMA 开始时触发的异步函数，更新当前工具状态为开始状态
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        # 更新当前工具状态为开始状态，LLAMA 令牌为空
        self.cur_tool.update(
            status=Status.start,
            llm_token="",
        )
        # 将更新后的工具状态加入队列
        self.queue.put_nowait(dumps(self.cur_tool))

    # 当聊天模型开始时触发的异步函数，更新当前工具状态为开始状态
    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # 更新当前工具状态为开始状态，LLAMA 令牌为空
        self.cur_tool.update(
            status=Status.start,
            llm_token="",
        )
        # 将更新后的工具状态加入队列
        self.queue.put_nowait(dumps(self.cur_tool))

    # 当 LLAMA 结束时触发的异步函数，更新当前工具状态为完成状态
    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        # 更新当前工具状态为完成状态，LLAMA 令牌为换行符
        self.cur_tool.update(
            status=Status.complete,
            llm_token="\n",
        )
        # 将更新后的工具状态加入队列
        self.queue.put_nowait(dumps(self.cur_tool))

    # 当 LLAMA 出现错误时触发的异步函数，更新当前工具状态为错误状态
    async def on_llm_error(self, error: Exception | KeyboardInterrupt, **kwargs: Any) -> None:
        # 更新当前工具状态为错误状态，设置错误信息为异常信息字符串
        self.cur_tool.update(
            status=Status.error,
            error=str(error),
        )
        # 将更新后的工具状态加入队列
        self.queue.put_nowait(dumps(self.cur_tool))
    # 定义一个异步方法，处理代理完成事件
    async def on_agent_finish(
            self, finish: AgentFinish, *, run_id: UUID, parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> None:
        # 更新当前工具的状态为代理完成，并设置最终答案为输出值
        self.cur_tool.update(
            status=Status.agent_finish,
            final_answer=finish.return_values["output"],
        )
        # 将更新后的工具对象加入队列中
        self.queue.put_nowait(dumps(self.cur_tool))
        # 重置当前工具对象为空字典
        self.cur_tool = {}
```