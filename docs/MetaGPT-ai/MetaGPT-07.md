# MetaGPT源码解析 7

# `metagpt/prompts/__init__.py`

这段代码是一个Python脚本，使用了PEP 8编码规范。

它是一个Python解释器脚本，当运行时，会读取并按照规范执行该脚本中定义的所有函数和变量。

具体来说，该脚本实现了以下功能：

1. 定义了一个名为"__init__.py"的文件。

2. 在文件中定义了一个名为"aqua_api"的函数，该函数可以接收一个字符串参数，并返回该字符串的对象。函数的实现使用了Python标准库中的字符串类型和字符串方法，例如"format"方法。

3. 在文件中定义了一个名为"qu+"的函数，该函数没有具体的实现，只是使用了Python 2.6中定义的"__add__"函数名称。因此，无论程序如何，该函数都会在运行时调用，但实际上不会做任何有意义的工作。

4. 在文件中定义了一个名为"version"的函数，该函数使用了Python标准库中的"print"函数，打印出了当前Python版本号。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/30 09:51
@Author  : alexanderwu
@File    : __init__.py
"""

```

# `metagpt/provider/anthropic_api.py`

该代码是一个Python脚本，使用了Anthropic库来与Claude2模型进行交互。该脚本实现了两个函数，分别是`aask`函数用于从用户接收输入，并从Claude2模型中获取回答；`ask`函数用于向Claude2模型发送用户输入，并获取模型的回答。

具体来说，该脚本使用`metagpt.config`包中的`CONFIG`类来设置Claude2模型的API密钥。然后定义了一个`Claude2`类，该类包含两个函数，分别用于向模型发送用户输入并获取回答以及从模型中获取回答。

在`aask`函数中，首先创建一个`client`实例，然后使用`client.completions.create`方法来向Claude2模型发送用户输入并获取回答。其中，用户输入使用了`anthropic.HUMAN_PROMPT`和`anthropic.AI_PROMPT`格式，表示让模型从人类和人工智能两种方式中选择回答。同时，使用了`max_tokens_to_sample`参数来限制从模型中获取的最大文本长度，为1000个单词。

在`ask`函数中，创建一个与上面定义的`client`相同的实例，然后使用`client.completions.create`方法来向模型发送用户输入并获取回答。其中，用户输入同样使用了`anthropic.HUMAN_PROMPT`和`anthropic.AI_PROMPT`格式，表示让模型从人类和人工智能两种方式中选择回答。同时，该函数还使用了`asyncio`库中的`run`函数，以便在Python 3.9+中运行该脚本。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/7/21 11:15
@Author  : Leo Xiao
@File    : anthropic_api.py
"""

import anthropic
from anthropic import Anthropic

from metagpt.config import CONFIG


class Claude2:
    def ask(self, prompt):
        client = Anthropic(api_key=CONFIG.claude_api_key)

        res = client.completions.create(
            model="claude-2",
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
            max_tokens_to_sample=1000,
        )
        return res.completion

    async def aask(self, prompt):
        client = Anthropic(api_key=CONFIG.claude_api_key)

        res = client.completions.create(
            model="claude-2",
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
            max_tokens_to_sample=1000,
        )
        return res.completion
    
```

# `metagpt/provider/base_chatbot.py`

这段代码定义了一个名为 `BaseChatbot` 的抽象类，该类继承自 `ABC` 类（定义了 abstract方法的容器类）和 `dataclass` 类（定义了 dataclass 类型的容器类）。`BaseChatbot` 的抽象方法包括 `ask`、`ask_batch` 和 `ask_code`，它们分别用于询问 GPT 不同的功能。

具体来说，`BaseChatbot` 中的 `mode` 属性指定了一个询问模式，可以是 `API`、`为模型` 或 `为服务`。如果 `mode` 是 `API`，那么 `BaseChatbot` 将向 GPT 发送一个 HTTP/POST 请求，并提供请求 body 中包含的所有参数。如果 `mode` 是 `为模型` 或 `为服务`，那么 `BaseChatbot` 将使用 GPT 的 API 或服务来询问用户问题。

对于每个抽象方法，都有一个 `__abstractmethod__` 标记，用于通知程序在运行时该方法不会实现任何方法。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/5 23:00
@Author  : alexanderwu
@File    : base_chatbot.py
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BaseChatbot(ABC):
    """Abstract GPT class"""
    mode: str = "API"

    @abstractmethod
    def ask(self, msg: str) -> str:
        """Ask GPT a question and get an answer"""

    @abstractmethod
    def ask_batch(self, msgs: list) -> str:
        """Ask GPT multiple questions and get a series of answers"""

    @abstractmethod
    def ask_code(self, msgs: list) -> str:
        """Ask GPT multiple questions and get a piece of code"""
        
```

# `metagpt/provider/base_gpt_api.py`

This appears to be a class that uses GPT (Generative Pre-trained Transformer) AI to provide chatbots and other virtual assistants with responses to messages. The class has several methods for completing tasks, including completion, which accepts a list of messages and returns a response. The completion method can be either synchronous or asynchronous, and it is required to provide a standard OpenAI completion interface. The class also provides methods for getting the first text of choice and messages to prompt, which are required to provide the bot with a starting point for the conversation.


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/5 23:04
@Author  : alexanderwu
@File    : base_gpt_api.py
"""
from abc import abstractmethod
from typing import Optional

from metagpt.logs import logger
from metagpt.provider.base_chatbot import BaseChatbot


class BaseGPTAPI(BaseChatbot):
    """GPT API abstract class, requiring all inheritors to provide a series of standard capabilities"""
    system_prompt = 'You are a helpful assistant.'

    def _user_msg(self, msg: str) -> dict[str, str]:
        return {"role": "user", "content": msg}

    def _assistant_msg(self, msg: str) -> dict[str, str]:
        return {"role": "assistant", "content": msg}

    def _system_msg(self, msg: str) -> dict[str, str]:
        return {"role": "system", "content": msg}

    def _system_msgs(self, msgs: list[str]) -> list[dict[str, str]]:
        return [self._system_msg(msg) for msg in msgs]

    def _default_system_msg(self):
        return self._system_msg(self.system_prompt)

    def ask(self, msg: str) -> str:
        message = [self._default_system_msg(), self._user_msg(msg)]
        rsp = self.completion(message)
        return self.get_choice_text(rsp)

    async def aask(self, msg: str, system_msgs: Optional[list[str]] = None) -> str:
        if system_msgs:
            message = self._system_msgs(system_msgs) + [self._user_msg(msg)]
        else:
            message = [self._default_system_msg(), self._user_msg(msg)]
        rsp = await self.acompletion_text(message, stream=True)
        logger.debug(message)
        # logger.debug(rsp)
        return rsp

    def _extract_assistant_rsp(self, context):
        return "\n".join([i["content"] for i in context if i["role"] == "assistant"])

    def ask_batch(self, msgs: list) -> str:
        context = []
        for msg in msgs:
            umsg = self._user_msg(msg)
            context.append(umsg)
            rsp = self.completion(context)
            rsp_text = self.get_choice_text(rsp)
            context.append(self._assistant_msg(rsp_text))
        return self._extract_assistant_rsp(context)

    async def aask_batch(self, msgs: list) -> str:
        """Sequential questioning"""
        context = []
        for msg in msgs:
            umsg = self._user_msg(msg)
            context.append(umsg)
            rsp_text = await self.acompletion_text(context)
            context.append(self._assistant_msg(rsp_text))
        return self._extract_assistant_rsp(context)

    def ask_code(self, msgs: list[str]) -> str:
        """FIXME: No code segment filtering has been done here, and all results are actually displayed"""
        rsp_text = self.ask_batch(msgs)
        return rsp_text

    async def aask_code(self, msgs: list[str]) -> str:
        """FIXME: No code segment filtering has been done here, and all results are actually displayed"""
        rsp_text = await self.aask_batch(msgs)
        return rsp_text

    @abstractmethod
    def completion(self, messages: list[dict]):
        """All GPTAPIs are required to provide the standard OpenAI completion interface
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello, show me python hello world code"},
            # {"role": "assistant", "content": ...}, # If there is an answer in the history, also include it
        ]
        """

    @abstractmethod
    async def acompletion(self, messages: list[dict]):
        """Asynchronous version of completion
        All GPTAPIs are required to provide the standard OpenAI completion interface
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello, show me python hello world code"},
            # {"role": "assistant", "content": ...}, # If there is an answer in the history, also include it
        ]
        """

    @abstractmethod
    async def acompletion_text(self, messages: list[dict], stream=False) -> str:
        """Asynchronous version of completion. Return str. Support stream-print"""

    def get_choice_text(self, rsp: dict) -> str:
        """Required to provide the first text of choice"""
        return rsp.get("choices")[0]["message"]["content"]

    def messages_to_prompt(self, messages: list[dict]):
        """[{"role": "user", "content": msg}] to user: <msg> etc."""
        return '\n'.join([f"{i['role']}: {i['content']}" for i in messages])

    def messages_to_dict(self, messages):
        """objects to [{"role": "user", "content": msg}] etc."""
        return [i.to_dict() for i in messages]
    
```

# `metagpt/provider/openai_api.py`

这段代码定义了一个函数 `handle_error`，它接收一个 `Tuple` 类型的参数 `error`，并执行一系列步骤来处理这个错误。这里 `handle_error` 函数的作用是：

1. 导入 `asyncio`、`time` 和 `typing.NamedTuple`；
2. 导入 `openai` 和 `openai.error`；
3. 通过 `openai.py` 安装 `openai` API，并创建一个 `Client` 实例；
4. 创建一个名为 `error` 的命名元组，用于存储错误信息；
5. 使用 `after_log`、`retry`、`retry_if_exception_type`、`stop_after_attempt` 和 `wait_fixed` 函数对 `error` 进行处理，分别是：
   - `after_log` 记录错误信息，但不输出；
   - `retry` 尝试在遇到 `APIConnectionError` 时重新尝试执行，最多尝试 5 次；
   - `retry_if_exception_type` 在遇到 `APIConnectionError` 时，异步执行 `handle_error` 函数，并将错误信息作为参数传入；
   - `stop_after_attempt` 设置在最多尝试次数达到后，停止执行；
   - `wait_fixed` 用于固定一个固定的延迟时间，例如 1 秒。

这段代码的作用是处理在 `handle_error` 函数中定义的错误，包括记录错误信息、异步执行 `handle_error` 函数、使用 `openai` API 等。


```py
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/5 23:08
@Author  : alexanderwu
@File    : openai.py
"""
import asyncio
import time
from typing import NamedTuple, Union

import openai
from openai.error import APIConnectionError
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

```

这段代码定义了一个名为RateLimiter的类，用于对API进行速率限制以提高GPT模型的性能。具体来说，这个类实现了以下几个方法：

1. `__init__`：初始化函数，根据需要设置每秒的请求速率（RPM），以及等待时间和响应时间的关系。
2. `split_batches`：将传入的批量大小进行分割，以便每个GPT模型实例能够处理请求。
3. `wait_if_needed`：等待信号量，并在请求数量未达到预设限制时等待一段时间。这个方法会根据传入的请求数量对等待时间进行微调。

使用这个RateLimiter的典型用法如下：

```pypython
rlim = RateLimiter(1.2)

# 这里使用 RateLimiter 对一个 GPT 模型实例进行请求限制
@example
def main(model):
   for i in range(10):
       response = model.generate_response(
           "Hey",
           "The quick brown fox jumps over the lazy dog.",
           "There once was a big dry sky",
           "But now the earth is defying the upward merge.",
           "With a little luck, and a lot of assistance, ["
           "You'll be登山發的交易 Quid future",
           "I'll be SCAulating you, just like a也需要水的鱼儿一样"
       )
       print(len(response))
```

在上面的示例中，我们创建了一个名为`RateLimiter`的实例，并将其与一个GPT模型实例一起使用。`__init__`方法设置请求速率为1.2，在这个速率下，模型将等待大约1.2秒。

我们使用`generate_response`方法生成一个响应，如果在这个请求中包含的消息数量不超过预设的速率（在这个例子中是10），那么它将被返回。否则，RateLimiter将等待一段时间，然后重试发送这个请求。


```py
from metagpt.config import CONFIG
from metagpt.logs import logger
from metagpt.provider.base_gpt_api import BaseGPTAPI
from metagpt.utils.singleton import Singleton
from metagpt.utils.token_counter import (
    TOKEN_COSTS,
    count_message_tokens,
    count_string_tokens,
    get_max_completion_tokens,
)


class RateLimiter:
    """Rate control class, each call goes through wait_if_needed, sleep if rate control is needed"""

    def __init__(self, rpm):
        self.last_call_time = 0
        # Here 1.1 is used because even if the calls are made strictly according to time,
        # they will still be QOS'd; consider switching to simple error retry later
        self.interval = 1.1 * 60 / rpm
        self.rpm = rpm

    def split_batches(self, batch):
        return [batch[i : i + self.rpm] for i in range(0, len(batch), self.rpm)]

    async def wait_if_needed(self, num_requests):
        current_time = time.time()
        elapsed_time = current_time - self.last_call_time

        if elapsed_time < self.interval * num_requests:
            remaining_time = self.interval * num_requests - elapsed_time
            logger.info(f"sleep {remaining_time}")
            await asyncio.sleep(remaining_time)

        self.last_call_time = time.time()


```

This class appears to be a simple token-based budgeting system for an AI assistant. It has a number of methods for updating the total cost, prompt tokens, and completion tokens, as well as getting the current running cost and budget. The running cost is calculated by multiplying the number of prompt and completion tokens by the cost of the token in question, dividing that number by 1000, and adding it to the running cost. The class also has a method for getting the total number of prompt and completion tokens.


```py
class Costs(NamedTuple):
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost: float
    total_budget: float


class CostManager(metaclass=Singleton):
    """计算使用接口的开销"""

    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0

    def update_cost(self, prompt_tokens, completion_tokens, model):
        """
        Update the total cost, prompt tokens, and completion tokens.

        Args:
        prompt_tokens (int): The number of tokens used in the prompt.
        completion_tokens (int): The number of tokens used in the completion.
        model (str): The model used for the API call.
        """
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        cost = (
            prompt_tokens * TOKEN_COSTS[model]["prompt"] + completion_tokens * TOKEN_COSTS[model]["completion"]
        ) / 1000
        self.total_cost += cost
        logger.info(
            f"Total running cost: ${self.total_cost:.3f} | Max budget: ${CONFIG.max_budget:.3f} | "
            f"Current cost: ${cost:.3f}, prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}"
        )
        CONFIG.total_cost = self.total_cost

    def get_total_prompt_tokens(self):
        """
        Get the total number of prompt tokens.

        Returns:
        int: The total number of prompt tokens.
        """
        return self.total_prompt_tokens

    def get_total_completion_tokens(self):
        """
        Get the total number of completion tokens.

        Returns:
        int: The total number of completion tokens.
        """
        return self.total_completion_tokens


    def get_total_cost(self):
        """
        Get the total cost of API calls.

        Returns:
        float: The total cost of API calls.
        """
        return self.total_cost


    def get_costs(self) -> Costs:
        """Get all costs"""
        return Costs(self.total_prompt_tokens, self.total_completion_tokens, self.total_cost, self.total_budget)


```

This is a class that appears to implement a simple natural language processing (NLP) system that can moderate content based on a set of predefined policies. The system has a `llm` backend for managing the rules and policies, and a frontend for generating prompts and receiving results.

The `llm` backend is responsible for enforcing the policies by generating an appropriate response for any given input. The `frontend` frontend handles generating prompts for the user to enter, and receives the results from the `llm` backend.

The system also has a moderation mechanism, which allows users to flag potentially offensive or inappropriate content for review.

Note that this is just a basic example and does not implement a complete and fully functional policy system. It is intended to demonstrate how a simple NLP system might structure and some of its core functionalities.


```py
def log_and_reraise(retry_state):
    logger.error(f"Retry attempts exhausted. Last exception: {retry_state.outcome.exception()}")
    logger.warning(
        """
Recommend going to https://deepwisdom.feishu.cn/wiki/MsGnwQBjiif9c3koSJNcYaoSnu4#part-XdatdVlhEojeAfxaaEZcMV3ZniQ
See FAQ 5.8
"""
    )
    raise retry_state.outcome.exception()


class OpenAIGPTAPI(BaseGPTAPI, RateLimiter):
    """
    Check https://platform.openai.com/examples for examples
    """

    def __init__(self):
        self.__init_openai(CONFIG)
        self.llm = openai
        self.model = CONFIG.openai_api_model
        self.auto_max_tokens = False
        self._cost_manager = CostManager()
        RateLimiter.__init__(self, rpm=self.rpm)

    def __init_openai(self, config):
        openai.api_key = config.openai_api_key
        if config.openai_api_base:
            openai.api_base = config.openai_api_base
        if config.openai_api_type:
            openai.api_type = config.openai_api_type
            openai.api_version = config.openai_api_version
        self.rpm = int(config.get("RPM", 10))

    async def _achat_completion_stream(self, messages: list[dict]) -> str:
        response = await openai.ChatCompletion.acreate(**self._cons_kwargs(messages), stream=True)

        # create variables to collect the stream of chunks
        collected_chunks = []
        collected_messages = []
        # iterate through the stream of events
        async for chunk in response:
            collected_chunks.append(chunk)  # save the event response
            choices = chunk["choices"]
            if len(choices) > 0:
                chunk_message = chunk["choices"][0].get("delta", {})  # extract the message
                collected_messages.append(chunk_message)  # save the message
                if "content" in chunk_message:
                    print(chunk_message["content"], end="")
        print()

        full_reply_content = "".join([m.get("content", "") for m in collected_messages])
        usage = self._calc_usage(messages, full_reply_content)
        self._update_costs(usage)
        return full_reply_content

    def _cons_kwargs(self, messages: list[dict]) -> dict:
        kwargs = {
            "messages": messages,
            "max_tokens": self.get_max_tokens(messages),
            "n": 1,
            "stop": None,
            "temperature": 0.3,
            "timeout": 3,
        }
        if CONFIG.openai_api_type == "azure":
            if CONFIG.deployment_name and CONFIG.deployment_id:
                raise ValueError("You can only use one of the `deployment_id` or `deployment_name` model")
            elif not CONFIG.deployment_name and not CONFIG.deployment_id:
                raise ValueError("You must specify `DEPLOYMENT_NAME` or `DEPLOYMENT_ID` parameter")
            kwargs_mode = (
                {"engine": CONFIG.deployment_name}
                if CONFIG.deployment_name
                else {"deployment_id": CONFIG.deployment_id}
            )
        else:
            kwargs_mode = {"model": self.model}
        kwargs.update(kwargs_mode)
        return kwargs

    async def _achat_completion(self, messages: list[dict]) -> dict:
        rsp = await self.llm.ChatCompletion.acreate(**self._cons_kwargs(messages))
        self._update_costs(rsp.get("usage"))
        return rsp

    def _chat_completion(self, messages: list[dict]) -> dict:
        rsp = self.llm.ChatCompletion.create(**self._cons_kwargs(messages))
        self._update_costs(rsp)
        return rsp

    def completion(self, messages: list[dict]) -> dict:
        # if isinstance(messages[0], Message):
        #     messages = self.messages_to_dict(messages)
        return self._chat_completion(messages)

    async def acompletion(self, messages: list[dict]) -> dict:
        # if isinstance(messages[0], Message):
        #     messages = self.messages_to_dict(messages)
        return await self._achat_completion(messages)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        after=after_log(logger, logger.level("WARNING").name),
        retry=retry_if_exception_type(APIConnectionError),
        retry_error_callback=log_and_reraise,
    )
    async def acompletion_text(self, messages: list[dict], stream=False) -> str:
        """when streaming, print each token in place."""
        if stream:
            return await self._achat_completion_stream(messages)
        rsp = await self._achat_completion(messages)
        return self.get_choice_text(rsp)

    def _calc_usage(self, messages: list[dict], rsp: str) -> dict:
        usage = {}
        if CONFIG.calc_usage:
            try:
                prompt_tokens = count_message_tokens(messages, self.model)
                completion_tokens = count_string_tokens(rsp, self.model)
                usage["prompt_tokens"] = prompt_tokens
                usage["completion_tokens"] = completion_tokens
                return usage
            except Exception as e:
                logger.error("usage calculation failed!", e)
        else:
            return usage

    async def acompletion_batch(self, batch: list[list[dict]]) -> list[dict]:
        """Return full JSON"""
        split_batches = self.split_batches(batch)
        all_results = []

        for small_batch in split_batches:
            logger.info(small_batch)
            await self.wait_if_needed(len(small_batch))

            future = [self.acompletion(prompt) for prompt in small_batch]
            results = await asyncio.gather(*future)
            logger.info(results)
            all_results.extend(results)

        return all_results

    async def acompletion_batch_text(self, batch: list[list[dict]]) -> list[str]:
        """Only return plain text"""
        raw_results = await self.acompletion_batch(batch)
        results = []
        for idx, raw_result in enumerate(raw_results, start=1):
            result = self.get_choice_text(raw_result)
            results.append(result)
            logger.info(f"Result of task {idx}: {result}")
        return results

    def _update_costs(self, usage: dict):
        if CONFIG.calc_usage:
            try:
                prompt_tokens = int(usage["prompt_tokens"])
                completion_tokens = int(usage["completion_tokens"])
                self._cost_manager.update_cost(prompt_tokens, completion_tokens, self.model)
            except Exception as e:
                logger.error("updating costs failed!", e)

    def get_costs(self) -> Costs:
        return self._cost_manager.get_costs()

    def get_max_tokens(self, messages: list[dict]):
        if not self.auto_max_tokens:
            return CONFIG.max_tokens_rsp
        return get_max_completion_tokens(messages, self.model, CONFIG.max_tokens_rsp)

    def moderation(self, content: Union[str, list[str]]):
        try:
            if not content:
                logger.error("content cannot be empty!")
            else:
                rsp = self._moderation(content=content)
                return rsp
        except Exception as e:
            logger.error(f"moderating failed:{e}")

    def _moderation(self, content: Union[str, list[str]]):
        rsp = self.llm.Moderation.create(input=content)
        return rsp

    async def amoderation(self, content: Union[str, list[str]]):
        try:
            if not content:
                logger.error("content cannot be empty!")
            else:
                rsp = await self._amoderation(content=content)
                return rsp
        except Exception as e:
            logger.error(f"moderating failed:{e}")

    async def _amoderation(self, content: Union[str, list[str]]):
        rsp = await self.llm.Moderation.acreate(input=content)
        return rsp

```

# `metagpt/provider/spark_api.py`

该代码是一个Python脚本，使用了环境变量来加载名为"anthropic_api.py"的第三方包。

作用：该脚本主要用于安装一个名为"anthropic_api"的第三方API，该API在多个操作系统上进行安装，并在多个Python版本上进行支持。它通过引入多个模块来完成这个任务，包括：

1. 使用了操作系统命令"ls"来查找名为"anthropic_api.py"的文件，并将其下载到本地。
2. 通过引入"_thread"模块，使得该脚本能够在多线程环境中运行。
3. 通过引入"base64"、"datetime"和"hashlib"模块，实现了将文件内容作为参数传递给"hashlib.sha256"哈希算法，以获取文件的哈希值。
4. 通过引入"hmac"模块，实现了创建一个消息认证码（MAC），并将其与文件内容进行哈希，以确保文件完整。
5. 通过引入"json"和"ssl"模块，实现了从API返回的数据中解析JSON和SSL数据。
6. 通过创建一个名为"anthropic_api.py"的文件，保存了所有需要安装的第三方包，并设置了一个变量"message_id"，用于在安装成功后返回一个消息ID。

由于该脚本在安装第三方包时需要下载和运行多个操作系统命令，因此它需要一定的系统权限才能正常工作。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/7/21 11:15
@Author  : Leo Xiao
@File    : anthropic_api.py
"""
import _thread as thread
import base64
import datetime
import hashlib
import hmac
import json
import ssl
from time import mktime
```

这段代码是一个基于Spark算法的API，其目的是提供给用户提供一个类比自然语言处理更加流畅的交互。以下是代码的主要部分功能解释：

1. 从typing模块中导入Optional类型，以便从HTTP请求中提取参数。
2. 从urllib.parse模块中导入urlencode函数，以便将URL编码的参数连接到请求的URL中。
3. 从urllib.parse模块中导入urlparse函数，以便解析请求URL的元素。
4. 从wsgiref.handlers模块中导入format_date_time函数，以便格式化日期和时间。
5. 从websocket模块中导入SparkAPI类，作为Spark算法的代表。
6. 在SparkAPI类中，重写了基类GPTAPI的方法，以实现Spark算法的接口。
7. 实现了以下方法：
	* ask：接收一个自然语言文本作为参数，并返回一个自然语言文本作为回答。
	* aask：接收一个自然语言文本和系统消息，并返回一个自然语言文本作为回答。
	* get_choice_text：从系统消息中提取系统消息的选择项，并返回其中的text内容。
	* acompletion_text：接收一组系统消息，并返回给定系统消息的运行结果。
	* completion：接收一组系统消息，并返回给定系统消息的运行结果。
	* acompletion：接收一组系统消息，并返回给定系统消息的运行结果。
	* format_date_time：将日期和时间格式化并返回。
8. 在SparkAPI类构造函数中，设置了一个日志输出，警告Spark算法不支持异步运行，并提醒用户在使用acompletion时不能并行访问。


```py
from typing import Optional
from urllib.parse import urlencode
from urllib.parse import urlparse
from wsgiref.handlers import format_date_time

import websocket  # 使用websocket_client

from metagpt.config import CONFIG
from metagpt.logs import logger
from metagpt.provider.base_gpt_api import BaseGPTAPI


class SparkAPI(BaseGPTAPI):

    def __init__(self):
        logger.warning('当前方法无法支持异步运行。当你使用acompletion时，并不能并行访问。')

    def ask(self, msg: str) -> str:
        message = [self._default_system_msg(), self._user_msg(msg)]
        rsp = self.completion(message)
        return rsp

    async def aask(self, msg: str, system_msgs: Optional[list[str]] = None) -> str:
        if system_msgs:
            message = self._system_msgs(system_msgs) + [self._user_msg(msg)]
        else:
            message = [self._default_system_msg(), self._user_msg(msg)]
        rsp = await self.acompletion(message)
        logger.debug(message)
        return rsp

    def get_choice_text(self, rsp: dict) -> str:
        return rsp["payload"]["choices"]["text"][-1]["content"]

    async def acompletion_text(self, messages: list[dict], stream=False) -> str:
        # 不支持
        logger.error('该功能禁用。')
        w = GetMessageFromWeb(messages)
        return w.run()

    async def acompletion(self, messages: list[dict]):
        # 不支持异步
        w = GetMessageFromWeb(messages)
        return w.run()

    def completion(self, messages: list[dict]):
        w = GetMessageFromWeb(messages)
        return w.run()


```

注意：官方建议，temperature和top_k修改一个即可
```py
{
   "闲聊机器人参数": {
       "闲聊": {
           "filters": ["盼"]
       },
       "不可以说话": ["新世界的衣服"],
       "小趣味的中文": ["我是一个人工智能助理"]
   },
   "message_api": {
       "合法消息": ["你是个机器人吗", "我是人工智能助手"]
   },
   "WsParam": {
       "api_appid": "YOUR_APP_ID",
       "api_key": "YOUR_APP_SECRET",
       "api_secret": "YOUR_APP_SECRET",
       "url": "wss://api.example.com/abcde"
   },
   "spark_appid": "YOUR_APP_ID",
   "spark_api_key": "YOUR_APP_SECRET",
   "spark_api_secret": "YOUR_APP_SECRET",
   "spark_url": "wss://api.example.com/abcde"
},
```
以上代码是对于接受一个新人用户的中文问题，并根据问题给出回答的一个`MessageBot`类。


```py
class GetMessageFromWeb:
    class WsParam:
        """
        该类适合讯飞星火大部分接口的调用。
        输入 app_id, api_key, api_secret, spark_url以初始化，
        create_url方法返回接口url
        """

        # 初始化
        def __init__(self, app_id, api_key, api_secret, spark_url, message=None):
            self.app_id = app_id
            self.api_key = api_key
            self.api_secret = api_secret
            self.host = urlparse(spark_url).netloc
            self.path = urlparse(spark_url).path
            self.spark_url = spark_url
            self.message = message

        # 生成url
        def create_url(self):
            # 生成RFC1123格式的时间戳
            now = datetime.datetime.now()
            date = format_date_time(mktime(now.timetuple()))

            # 拼接字符串
            signature_origin = "host: " + self.host + "\n"
            signature_origin += "date: " + date + "\n"
            signature_origin += "GET " + self.path + " HTTP/1.1"

            # 进行hmac-sha256进行加密
            signature_sha = hmac.new(self.api_secret.encode('utf-8'), signature_origin.encode('utf-8'),
                                     digestmod=hashlib.sha256).digest()

            signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

            authorization_origin = f'api_key="{self.api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

            authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

            # 将请求的鉴权参数组合为字典
            v = {
                "authorization": authorization,
                "date": date,
                "host": self.host
            }
            # 拼接鉴权参数，生成url
            url = self.spark_url + '?' + urlencode(v)
            # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
            return url

    def __init__(self, text):
        self.text = text
        self.ret = ''
        self.spark_appid = CONFIG.spark_appid
        self.spark_api_secret = CONFIG.spark_api_secret
        self.spark_api_key = CONFIG.spark_api_key
        self.domain = CONFIG.domain
        self.spark_url = CONFIG.spark_url

    def on_message(self, ws, message):
        data = json.loads(message)
        code = data['header']['code']

        if code != 0:
            ws.close()  # 请求错误，则关闭socket
            logger.critical(f'回答获取失败，响应信息反序列化之后为： {data}')
            return
        else:
            choices = data["payload"]["choices"]
            seq = choices["seq"]  # 服务端是流式返回，seq为返回的数据序号
            status = choices["status"]  # 服务端是流式返回，status用于判断信息是否传送完毕
            content = choices["text"][0]["content"]  # 本次接收到的回答文本
            self.ret += content
            if status == 2:
                ws.close()

    # 收到websocket错误的处理
    def on_error(self, ws, error):
        # on_message方法处理接收到的信息，出现任何错误，都会调用这个方法
        logger.critical(f'通讯连接出错，【错误提示: {error}】')

    # 收到websocket关闭的处理
    def on_close(self, ws, one, two):
        pass

    # 处理请求数据
    def gen_params(self):

        data = {
            "header": {
                "app_id": self.spark_appid,
                "uid": "1234"
            },
            "parameter": {
                "chat": {
                    # domain为必传参数
                    "domain": self.domain,

                    # 以下为可微调，非必传参数
                    # 注意：官方建议，temperature和top_k修改一个即可
                    "max_tokens": 2048,  # 默认2048，模型回答的tokens的最大长度，即允许它输出文本的最长字数
                    "temperature": 0.5,  # 取值为[0,1],默认为0.5。取值越高随机性越强、发散性越高，即相同的问题得到的不同答案的可能性越高
                    "top_k": 4,  # 取值为[1，6],默认为4。从k个候选中随机选择一个（非等概率）
                }
            },
            "payload": {
                "message": {
                    "text": self.text
                }
            }
        }
        return data

    def send(self, ws, *args):
        data = json.dumps(self.gen_params())
        ws.send(data)

    # 收到websocket连接建立的处理
    def on_open(self, ws):
        thread.start_new_thread(self.send, (ws,))

    # 处理收到的 websocket消息，出现任何错误，调用on_error方法
    def run(self):
        return self._run(self.text)

    def _run(self, text_list):

        ws_param = self.WsParam(
            self.spark_appid,
            self.spark_api_key,
            self.spark_api_secret,
            self.spark_url,
            text_list)
        ws_url = ws_param.create_url()

        websocket.enableTrace(False)  # 默认禁用 WebSocket 的跟踪功能
        ws = websocket.WebSocketApp(ws_url, on_message=self.on_message, on_error=self.on_error, on_close=self.on_close,
                                    on_open=self.on_open)
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        return self.ret

```

# `metagpt/provider/__init__.py`

这段代码是一个Python脚本，它定义了一个名为`__main__`的函数。在这个函数中，它导入了`metagpt.provider.openai_api`模块，该模块可能是用于调用或依赖OpenAIGPTAPI的第三方库。然后，它创建了一个`OpenAIGPTAPI`实例，并将其命名为`OpenAIGPTAPI`。

由于该脚本没有具体的函数或类定义，因此它无法执行实际的代码。相反，它定义了一个可对外部函数或类进行引用的`__all__`列表，这意味着`OpenAIGPTAPI`实例可以被外部函数或类所引用。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/5 22:59
@Author  : alexanderwu
@File    : __init__.py
"""

from metagpt.provider.openai_api import OpenAIGPTAPI


__all__ = ["OpenAIGPTAPI"]

```

# `metagpt/roles/architect.py`

该代码定义了一个名为Architect的类，表示软件开发过程中的建筑师角色。该类包含以下属性和方法：

- name: 建筑师的姓名。
- profile: 建筑师角色的文档，默认为Architect。
- goal: 建筑师的主要目标或职责。
- constraints: 建筑师应该遵守的约束或指导方针。

在__init__方法中，建筑师根据传入的参数初始化这些属性。然后，使用超级类方法__init__，确保符合Python环境和Metagpt行动的规范。

此外，还初始化了一个actions特定于建筑师角色的WriteDesign方法，并注册了一个WritePRD事件，以便建筑师可以监视该事件的行动。最后，还定义了一个should方法，用于检查约束条件是否满足。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:43
@Author  : alexanderwu
@File    : architect.py
"""

from metagpt.actions import WritePRD
from metagpt.actions.design_api import WriteDesign
from metagpt.roles import Role


class Architect(Role):
    """
    Represents an Architect role in a software development process.

    Attributes:
        name (str): Name of the architect.
        profile (str): Role profile, default is 'Architect'.
        goal (str): Primary goal or responsibility of the architect.
        constraints (str): Constraints or guidelines for the architect.
    """

    def __init__(
        self,
        name: str = "Bob",
        profile: str = "Architect",
        goal: str = "Design a concise, usable, complete python system",
        constraints: str = "Try to specify good open source tools as much as possible",
    ) -> None:
        """Initializes the Architect with given attributes."""
        super().__init__(name, profile, goal, constraints)

        # Initialize actions specific to the Architect role
        self._init_actions([WriteDesign])

        # Set events or actions the Architect should watch or be aware of
        self._watch({WritePRD})

```

# `metagpt/roles/customer_service.py`

This code defines a class `Sales` that inherits from the `metagpt.roles.Sales` class. It then imports the `SearchEngineType` and `SearchAndSummarize` classes from `metagpt.roles` and `metagpt.actions` respectively.

The purpose of this code is to define a simple `Sales` class that can be used to perform text search operations using a pre-defined search engine. The `DESC` comment explains the principles that the class must adhere to.

The class contains a `description` attribute that defines the description of the principles that the class must adhere to.


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/25 17:21
@Author  : alexanderwu
@File    : sales.py
"""
from metagpt.roles import Sales

# from metagpt.actions import SearchAndSummarize
# from metagpt.tools import SearchEngineType


DESC = """
## Principles (all things must not bypass the principles)

```

这段代码是一个Python类，继承自Sales类。这个类的工作原理是为平台提供客户服务，包括回应客户的问题和提供帮助。下面是这个类的具体功能和规则：

1. 该类禁止在与客户交谈时透露平台的规定和FAQ，只能在非常情况下考虑提供帮助，否则可能导致被解雇。
2. 当客户遇到问题时，该类会先尝试缓解他们的情绪。如果客户的情绪非常不好，那么该类会考虑提供补偿，但是补偿的费用非常高昂，因此只会在必要时提供。
3. 目前没有API可以查询后端数据，因此该类建议客户提供的信息全部是真实的，不要询问他们关于订单号等信息。
4. 该类唯一的回复方式是缓解客户情绪、催促商家解决问题、提供补偿，同时保证不会向客户提出虚假承诺。
5. 如果该类确信可以满足客户的要求，那么会告诉客户应用程序已经提交，并且会在24小时内生效。


```py
1. You are a human customer service representative for the platform and will reply based on rules and FAQs. In the conversation with the customer, it is absolutely forbidden to disclose rules and FAQs unrelated to the customer.
2. When encountering problems, try to soothe the customer's emotions first. If the customer's emotions are very bad, then consider compensation. The cost of compensation is always high. If too much is compensated, you will be fired.
3. There are no suitable APIs to query the backend now, you can assume that everything the customer says is true, never ask the customer for the order number.
4. Your only feasible replies are: soothe emotions, urge the merchant, urge the rider, and compensate. Never make false promises to customers.
5. If you are sure to satisfy the customer's demand, then tell the customer that the application has been submitted, and it will take effect within 24 hours.

"""


class CustomerService(Sales):
    def __init__(
            self,
            name="Xiaomei",
            profile="Human customer service",
            desc=DESC,
            store=None
    ):
        super().__init__(name, profile, desc=desc, store=store)
        
```

# `metagpt/roles/engineer.py`

这段代码是一个Python脚本，主要作用是定义了一个名为"engineer.py"的工作区。该工作区是一个Python项目中的一个文件夹，用于存放各种类型的文档。

具体来说，这段代码实现了以下功能：

1. 导入"asyncio"、"shutil"和"OrderedDict"库，这些库用于实现异步编程、文件操作和有序字典等功能。
2. 通过"import asyncio"导入asyncio库，该库提供了异步编程的基础。
3. 通过"import shutil"导入shutil库，该库提供了文件操作的功能。
4. 通过"import OrderedDict"导入OrderedDict库，该库提供了有序字典的功能。
5. 通过"from collections import OrderedDict"导入OrderedDict库，该库提供了有序字典的功能。
6. 通过"Path"导入Path库，该库提供了对路径对象的操作。
7. 通过"import metagpt.actions"导入metagpt.actions库，该库提供了写代码的动作。
8. 通过"import metagpt.const"导入metagpt.const库，该库提供了常量。
9. 通过"import metagpt.logs"导入metagpt.logs库，该库提供了日志的记录功能。
10. 通过"@Time"定义了一个名为"2023/5/11 14:43"的时间戳，用于记录代码的创建时间。
11. 通过"@Author"定义了一个名为"alexanderwu"的用户名，用于标识代码的作者。
12. 通过"@File"定义了一个名为"engineer.py"的工作区，用于存放各种类型的文档。

总之，这段代码定义了一个名为"engineer.py"的工作区，用于存放各种类型的文档。该工作区内部可能还会包含其他Python库和文件的引用。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:43
@Author  : alexanderwu
@File    : engineer.py
"""
import asyncio
import shutil
from collections import OrderedDict
from pathlib import Path

from metagpt.actions import WriteCode, WriteCodeReview, WriteDesign, WriteTasks
from metagpt.const import WORKSPACE_ROOT
from metagpt.logs import logger
```

这段代码是一个名为 `gather_ordered_k` 的函数，其作用是聚集给定协程 `coros` 中 `k` 个不同的协程，并返回它们的并集。

具体实现过程如下：

1. 定义了两个导入的函数 `Role` 和 `Message`，用于从 `metagpt.roles` 和 `metagpt.schema` 包中导入需要的函数和类。
2. 定义了一个名为 `CodeParser` 的函数，从 `metagpt.utils.common` 包中导入一个名为 `FILENAME_CODE_SEP` 的常量，以及导入一个名为 `MSG_SEP` 的常量。
3. 定义了一个名为 `gather_ordered_k` 的函数，它使用 `asyncio` 库中的 `Queue` 和 `Wait` 函数，以及一些特殊字符串 `FILENAME_CODE_SEP` 和 `MSG_SEP`。
4. 在函数内部，首先定义了一个 `OrderedDict` 类型的 `tasks` 变量，用于记录每个协程的任务编号。然后定义了一个包含 `k` 个不同协程的列表 `results` 变量，用于存储每个协程的并集结果。接着定义了一个 `asyncio.Queue` 类型的 `done_queue` 变量，用于存储已经完成的任务，并使用 `asyncio.wait` 函数等待队列中的任务完成。
5. 使用一个循环遍历给定的协程 `coros`。对于每个协程，首先检查自己所在的任务数是否大于 `k`，如果是，就执行以下操作：
  a. 从 `tasks` 字典中弹出编号最小的任务，并将其添加到 `done_queue` 中。
  b. 使用 `asyncio.create_task` 函数创建一个新任务，并将该任务作为协程加入 `tasks` 字典中。
6. 如果 `done_queue` 不为空，就执行以下操作：
  a. 获取队列中的所有任务并按索引顺序循环遍历。
  b. 对于每个任务，提取其结果并将其添加到 `results` 列表中。
7. 最后，函数返回 `results` 列表，其中包含了所有协程的并集结果。


```py
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.utils.common import CodeParser
from metagpt.utils.special_tokens import FILENAME_CODE_SEP, MSG_SEP


async def gather_ordered_k(coros, k) -> list:
    tasks = OrderedDict()
    results = [None] * len(coros)
    done_queue = asyncio.Queue()

    for i, coro in enumerate(coros):
        if len(tasks) >= k:
            done, _ = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                index = tasks.pop(task)
                await done_queue.put((index, task.result()))
        task = asyncio.create_task(coro)
        tasks[task] = i

    if tasks:
        done, _ = await asyncio.wait(tasks.keys())
        for task in done:
            index = tasks[task]
            await done_queue.put((index, task.result()))

    while not done_queue.empty():
        index, result = await done_queue.get()
        results[index] = result

    return results


```

Based on the provided code, it seems like the code you want to execute is "WriteDesign, WriteTasks, WriteCode". These are all actions related to the "Architect" role, which suggests that the primary focus of this task is to define and write the design for a given project.

Therefore, the implementation should focus on writing code that corresponds to this role. This includes writing code for the "ProjectManager" role, as well as any actions related to the "Architect" role.

It is also important to note that the code you provided includes a "todo" field, which suggests that there may be other tasks or code related to this task that are not included in the provided code. It would be helpful to review the remaining code and any additional documentation to fully understand the capabilities and requirements of this task.


```py
class Engineer(Role):
    """
    Represents an Engineer role responsible for writing and possibly reviewing code.

    Attributes:
        name (str): Name of the engineer.
        profile (str): Role profile, default is 'Engineer'.
        goal (str): Goal of the engineer.
        constraints (str): Constraints for the engineer.
        n_borg (int): Number of borgs.
        use_code_review (bool): Whether to use code review.
        todos (list): List of tasks.
    """

    def __init__(
        self,
        name: str = "Alex",
        profile: str = "Engineer",
        goal: str = "Write elegant, readable, extensible, efficient code",
        constraints: str = "The code should conform to standards like PEP8 and be modular and maintainable",
        n_borg: int = 1,
        use_code_review: bool = False,
    ) -> None:
        """Initializes the Engineer role with given attributes."""
        super().__init__(name, profile, goal, constraints)
        self._init_actions([WriteCode])
        self.use_code_review = use_code_review
        if self.use_code_review:
            self._init_actions([WriteCode, WriteCodeReview])
        self._watch([WriteTasks])
        self.todos = []
        self.n_borg = n_borg

    @classmethod
    def parse_tasks(self, task_msg: Message) -> list[str]:
        if task_msg.instruct_content:
            return task_msg.instruct_content.dict().get("Task list")
        return CodeParser.parse_file_list(block="Task list", text=task_msg.content)

    @classmethod
    def parse_code(self, code_text: str) -> str:
        return CodeParser.parse_code(block="", text=code_text)

    @classmethod
    def parse_workspace(cls, system_design_msg: Message) -> str:
        if system_design_msg.instruct_content:
            return system_design_msg.instruct_content.dict().get("Python package name").strip().strip("'").strip('"')
        return CodeParser.parse_str(block="Python package name", text=system_design_msg.content)

    def get_workspace(self) -> Path:
        msg = self._rc.memory.get_by_action(WriteDesign)[-1]
        if not msg:
            return WORKSPACE_ROOT / "src"
        workspace = self.parse_workspace(msg)
        # Codes are written in workspace/{package_name}/{package_name}
        return WORKSPACE_ROOT / workspace / workspace

    def recreate_workspace(self):
        workspace = self.get_workspace()
        try:
            shutil.rmtree(workspace)
        except FileNotFoundError:
            pass  # The folder does not exist, but we don't care
        workspace.mkdir(parents=True, exist_ok=True)

    def write_file(self, filename: str, code: str):
        workspace = self.get_workspace()
        filename = filename.replace('"', "").replace("\n", "")
        file = workspace / filename
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(code)
        return file

    def recv(self, message: Message) -> None:
        self._rc.memory.add(message)
        if message in self._rc.important_memory:
            self.todos = self.parse_tasks(message)

    async def _act_mp(self) -> Message:
        # self.recreate_workspace()
        todo_coros = []
        for todo in self.todos:
            todo_coro = WriteCode().run(
                context=self._rc.memory.get_by_actions([WriteTasks, WriteDesign]), filename=todo
            )
            todo_coros.append(todo_coro)

        rsps = await gather_ordered_k(todo_coros, self.n_borg)
        for todo, code_rsp in zip(self.todos, rsps):
            _ = self.parse_code(code_rsp)
            logger.info(todo)
            logger.info(code_rsp)
            # self.write_file(todo, code)
            msg = Message(content=code_rsp, role=self.profile, cause_by=type(self._rc.todo))
            self._rc.memory.add(msg)
            del self.todos[0]

        logger.info(f"Done {self.get_workspace()} generating.")
        msg = Message(content="all done.", role=self.profile, cause_by=type(self._rc.todo))
        return msg

    async def _act_sp(self) -> Message:
        code_msg_all = []  # gather all code info, will pass to qa_engineer for tests later
        for todo in self.todos:
            code = await WriteCode().run(context=self._rc.history, filename=todo)
            # logger.info(todo)
            # logger.info(code_rsp)
            # code = self.parse_code(code_rsp)
            file_path = self.write_file(todo, code)
            msg = Message(content=code, role=self.profile, cause_by=type(self._rc.todo))
            self._rc.memory.add(msg)

            code_msg = todo + FILENAME_CODE_SEP + str(file_path)
            code_msg_all.append(code_msg)

        logger.info(f"Done {self.get_workspace()} generating.")
        msg = Message(
            content=MSG_SEP.join(code_msg_all), role=self.profile, cause_by=type(self._rc.todo), send_to="QaEngineer"
        )
        return msg

    async def _act_sp_precision(self) -> Message:
        code_msg_all = []  # gather all code info, will pass to qa_engineer for tests later
        for todo in self.todos:
            """
            # Select essential information from the historical data to reduce the length of the prompt (summarized from human experience):
            1. All from Architect
            2. All from ProjectManager
            3. Do we need other codes (currently needed)?
            TODO: The goal is not to need it. After clear task decomposition, based on the design idea, you should be able to write a single file without needing other codes. If you can't, it means you need a clearer definition. This is the key to writing longer code.
            """
            context = []
            msg = self._rc.memory.get_by_actions([WriteDesign, WriteTasks, WriteCode])
            for m in msg:
                context.append(m.content)
            context_str = "\n".join(context)
            # Write code
            code = await WriteCode().run(context=context_str, filename=todo)
            # Code review
            if self.use_code_review:
                try:
                    rewrite_code = await WriteCodeReview().run(context=context_str, code=code, filename=todo)
                    code = rewrite_code
                except Exception as e:
                    logger.error("code review failed!", e)
                    pass
            file_path = self.write_file(todo, code)
            msg = Message(content=code, role=self.profile, cause_by=WriteCode)
            self._rc.memory.add(msg)

            code_msg = todo + FILENAME_CODE_SEP + str(file_path)
            code_msg_all.append(code_msg)

        logger.info(f"Done {self.get_workspace()} generating.")
        msg = Message(
            content=MSG_SEP.join(code_msg_all), role=self.profile, cause_by=type(self._rc.todo), send_to="QaEngineer"
        )
        return msg

    async def _act(self) -> Message:
        """Determines the mode of action based on whether code review is used."""
        if self.use_code_review:
            return await self._act_sp_precision()
        return await self._act_sp()

```

# `metagpt/roles/invoice_ocr_assistant.py`

这段代码是一个Python脚本，用于实现发票OCR自动识别功能。具体来说，它实现了以下功能：

1. 导入pandas库以便于读取和写入数据。
2. 导入metagpt库，包括actions、prompts、roles和schema模块，以便于使用其中的actions、prompts、roles和schema库。
3. 使用INVOICE_OCR_SUCCESS作为metagptprompts库中的prompts参数，实现了回复人类输入的invoice_ocr_success的结果。
4. 创建一个名为invoice_ocr_assistant的角色，并继承自Role类。
5. 在__init__方法中，初始化一个InvoiceOCR实例，用于执行OCR识别操作。
6. 在invoice_ocr_assistant类中，定义了一个ReplyQuestion类，用于在识别结果返回给人类用户时，提出相关问题以获取更多信息。
7. 在main方法中，使用cdr（clientdataandrole）方法将INVOICE_OCR_SUCCESS角色与invoice_ocr_assistant类一起使用，并在识别结果为INVOICE_OCR_SUCCESS时，调用GenerateTable和ReplyQuestion类中的相应方法。


```py
#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

"""
@Time    : 2023/9/21 14:10:05
@Author  : Stitch-z
@File    : invoice_ocr_assistant.py
"""

import pandas as pd

from metagpt.actions.invoice_ocr import InvoiceOCR, GenerateTable, ReplyQuestion
from metagpt.prompts.invoice_ocr import INVOICE_OCR_SUCCESS
from metagpt.roles import Role
from metagpt.schema import Message


```

This is a class definition for an `InvoiceOCR` object that uses the `助理思考` and `助理回答` methods for interacting with the助理. It also has a `生成表格` method for preparing answers for users to print out.

The `__init__` method is used for initializing the object with a `self._rc` object, which is supposed to store the information about the robot. The `self._rc.memory.add` method is used to add the object to the robot's memory.

The `_act` method is used for taking actions based on the role. It tries to execute the action by calling the appropriate method, such as `todo.run` for processing the invoice file. If the action is `GenerateTable`, it will prompt the user to enter the invoice file path, and then it will convert the OCR results to a Markdown table.

The `_think` method is used for waiting for the robot to complete its action. It returns a `Message` object containing the information about the action.

The `_react` method is used for waiting for the robot to complete its action and returning a final message. It calls the `_act` method in a loop until the `todo` attribute is no longer `None`.

The `__repr__` method is used for returning a human-readable representation of the object.


```py
class InvoiceOCRAssistant(Role):
    """Invoice OCR assistant, support OCR text recognition of invoice PDF, png, jpg, and zip files,
    generate a table for the payee, city, total amount, and invoicing date of the invoice,
    and ask questions for a single file based on the OCR recognition results of the invoice.

    Args:
        name: The name of the role.
        profile: The role profile description.
        goal: The goal of the role.
        constraints: Constraints or requirements for the role.
        language: The language in which the invoice table will be generated.
    """

    def __init__(
        self,
        name: str = "Stitch",
        profile: str = "Invoice OCR Assistant",
        goal: str = "OCR identifies invoice files and generates invoice main information table",
        constraints: str = "",
        language: str = "ch",
    ):
        super().__init__(name, profile, goal, constraints)
        self._init_actions([InvoiceOCR])
        self.language = language
        self.filename = ""
        self.origin_query = ""
        self.orc_data = None

    async def _think(self) -> None:
        """Determine the next action to be taken by the role."""
        if self._rc.todo is None:
            self._set_state(0)
            return

        if self._rc.state + 1 < len(self._states):
            self._set_state(self._rc.state + 1)
        else:
            self._rc.todo = None

    async def _act(self) -> Message:
        """Perform an action as determined by the role.

        Returns:
            A message containing the result of the action.
        """
        msg = self._rc.memory.get(k=1)[0]
        todo = self._rc.todo
        if isinstance(todo, InvoiceOCR):
            self.origin_query = msg.content
            file_path = msg.instruct_content.get("file_path")
            self.filename = file_path.name
            if not file_path:
                raise Exception("Invoice file not uploaded")

            resp = await todo.run(file_path)
            if len(resp) == 1:
                # Single file support for questioning based on OCR recognition results
                self._init_actions([GenerateTable, ReplyQuestion])
                self.orc_data = resp[0]
            else:
                self._init_actions([GenerateTable])

            self._rc.todo = None
            content = INVOICE_OCR_SUCCESS
        elif isinstance(todo, GenerateTable):
            ocr_results = msg.instruct_content
            resp = await todo.run(ocr_results, self.filename)

            # Convert list to Markdown format string
            df = pd.DataFrame(resp)
            markdown_table = df.to_markdown(index=False)
            content = f"{markdown_table}\n\n\n"
        else:
            resp = await todo.run(self.origin_query, self.orc_data)
            content = resp

        msg = Message(content=content, instruct_content=resp)
        self._rc.memory.add(msg)
        return msg

    async def _react(self) -> Message:
        """Execute the invoice ocr assistant's think and actions.

        Returns:
            A message containing the final result of the assistant's actions.
        """
        while True:
            await self._think()
            if self._rc.todo is None:
                break
            msg = await self._act()
        return msg


```

# `metagpt/roles/product_manager.py`

该代码定义了一个名为ProductManager的类，代表一个负责产品开发和管理的产品经理角色。ProductManager类包含产品经理名称、角色配置文件、目标和约束等属性，并实现了metagpt.actions.BossRequirement和metagpt.roles.Role接口，用于管理产品经理角色和其相关动作和角色。

具体来说，ProductManager类在__init__方法中，初始化了产品经理的名称、角色配置文件、目标和约束，然后调用父类metagpt.roles.Role的__init__方法，实现产品经理角色的初始化。同时，还实现了metagpt.actions.BossRequirement和metagpt.roles.Role接口的相应方法，用于管理和执行产品经理角色对应的动作和角色。

由于ProductManager类中的actions和role都是使用metagpt.roles.Role接口实现的，因此它们都具有与角色相关的属性和方法。而针对actions,ProductManager类中的实现主要体现在[[Python废墟]]中，用于创建一个产品经理角色的行动，具体方法包括submit_action、submit_event和schedule_action等。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:43
@Author  : alexanderwu
@File    : product_manager.py
"""
from metagpt.actions import BossRequirement, WritePRD
from metagpt.roles import Role


class ProductManager(Role):
    """
    Represents a Product Manager role responsible for product development and management.

    Attributes:
        name (str): Name of the product manager.
        profile (str): Role profile, default is 'Product Manager'.
        goal (str): Goal of the product manager.
        constraints (str): Constraints or limitations for the product manager.
    """

    def __init__(
        self,
        name: str = "Alice",
        profile: str = "Product Manager",
        goal: str = "Efficiently create a successful product",
        constraints: str = "",
    ) -> None:
        """
        Initializes the ProductManager role with given attributes.

        Args:
            name (str): Name of the product manager.
            profile (str): Role profile.
            goal (str): Goal of the product manager.
            constraints (str): Constraints or limitations for the product manager.
        """
        super().__init__(name, profile, goal, constraints)
        self._init_actions([WritePRD])
        self._watch([BossRequirement])

```