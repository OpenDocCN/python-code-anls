# AutoGPT源码解析 12

# `autogpts/autogpt/autogpt/logs/handlers.py`

这段代码是一个异步编程框架，从未来可以获取更多信息，通过这个框架可以访问到将来定义的类型。这是一个魔法数字 79，它会在需要时创建一个自定义的模块。


```py
from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import TYPE_CHECKING

from autogpt.logs.utils import remove_color_codes
from autogpt.speech import TextToSpeechProvider

if TYPE_CHECKING:
    from autogpt.speech import TTSConfig


```

这段代码定义了一个名为 `TypingConsoleHandler` 的类，它继承自 `logging.StreamHandler` 类，用于将输出流到 console。

该 `TypingConsoleHandler` 类的实现了一个 `emit` 方法，该方法接受一个 `logging.LogRecord` 对象作为参数，并将其写入 console 的流中。

在 `emit` 方法的内部，首先定义了 `TypingConsoleHandler` 中 `MIN_WPS` 和 `MAX_WPS` 两个概念，分别表示最小和最大的 WPS(每秒单词数)。

然后，定义了一个 `self.format` 方法，用于将传入的 `logging.LogRecord` 对象格式化并输出。

接着，使用正则表达式和 `re.findall` 方法来遍历 `words` 列表中的每个单词，并将它们输出到流中。

在遍历 `words` 列表的过程中，使用 `min_typing_interval` 和 `max_typing_interval` 变量来控制每次输出单词的时间间隔。这些变量是根据 `MIN_WPS` 和 `MAX_WPS` 计算出来的，分别代表了每秒输出单词的最小时间和最大时间。

最后，如果出现异常情况(比如 `self.format` 方法抛出了 `Exception` 对象)，则调用 `self.handleError` 方法来处理异常。


```py
class TypingConsoleHandler(logging.StreamHandler):
    """Output stream to console using simulated typing"""

    # Typing speed settings in WPS (Words Per Second)
    MIN_WPS = 25
    MAX_WPS = 100

    def emit(self, record: logging.LogRecord) -> None:
        min_typing_interval = 1 / TypingConsoleHandler.MAX_WPS
        max_typing_interval = 1 / TypingConsoleHandler.MIN_WPS

        msg = self.format(record)
        try:
            # Split without discarding whitespace
            words = re.findall(r"\S+\s*", msg)

            for i, word in enumerate(words):
                self.stream.write(word)
                self.flush()
                if i >= len(words) - 1:
                    self.stream.write(self.terminator)
                    self.flush()
                    break

                interval = random.uniform(min_typing_interval, max_typing_interval)
                # type faster after each word
                min_typing_interval = min_typing_interval * 0.95
                max_typing_interval = max_typing_interval * 0.95
                time.sleep(interval)
        except Exception:
            self.handleError(record)


```

这段代码定义了一个名为 TTSHandler 的类，继承自名为 logging.Handler 的类。这个类的功能是输出到配置好的 TTS 引擎，如果没有配置好则输出到标准输出（通常是终端）。

在类的初始化函数中，首先调用父类的初始化函数，然后设置本类的实例为当前实例，本类实例也设置好了 TTSProvider 和 config 参数。

在 format 函数中，根据 TTSProvider 的配置，提取出 record 对象中的标题（如果没有设置，则默认为空）并将其与 record 对象的消息内容连接起来，去除消息中的颜色代码，最后输出消息。

在 emit 函数中，首先检查 TTSProvider 是否开启了说话模式，如果没有开启，则不做任何操作。否则，使用 TTSProvider 的 say 方法将消息输出到 TTS 引擎。


```py
class TTSHandler(logging.Handler):
    """Output messages to the configured TTS engine (if any)"""

    def __init__(self, config: TTSConfig):
        super().__init__()
        self.config = config
        self.tts_provider = TextToSpeechProvider(config)

    def format(self, record: logging.LogRecord) -> str:
        if getattr(record, "title", ""):
            msg = f"{getattr(record, 'title')} {record.msg}"
        else:
            msg = f"{record.msg}"

        return remove_color_codes(msg)

    def emit(self, record: logging.LogRecord) -> None:
        if not self.config.speak_mode:
            return

        message = self.format(record)
        self.tts_provider.say(message)


```

这段代码定义了一个名为 `JsonFileHandler` 的类，继承自 `logging.FileHandler` 类，用于处理 JSON 格式的日志记录。

在该类的 `format` 方法中，将传入的 `record` 对象获取到的消息(即日志记录中的 `getMessage` 方法返回的值)解析为 JSON 格式，并将其存储在 `record.json_data` 属性中。然后，使用 `json.dumps` 方法将 `record.json_data` 对象转换为字符串，并使用 `ensure_ascii=False` 和 `indent=4` 参数对 JSON 数据进行格式化。最后，将格式化后的字符串写入到文件中，并返回。

在 `emit` 方法中，使用 `with` 语句打开一个文件，用于写入 `record` 对象的消息。然后，使用 `self.format` 方法将 `record` 对象的消息格式化后写入到文件中。由于 `self.format` 方法已经对消息进行了格式化，因此直接将 `self.format(record)` 返回即可，而不需要再次调用 `format` 方法。


```py
class JsonFileHandler(logging.FileHandler):
    def format(self, record: logging.LogRecord) -> str:
        record.json_data = json.loads(record.getMessage())
        return json.dumps(getattr(record, "json_data"), ensure_ascii=False, indent=4)

    def emit(self, record: logging.LogRecord) -> None:
        with open(self.baseFilename, "w", encoding="utf-8") as f:
            f.write(self.format(record))

```

# `autogpts/autogpt/autogpt/logs/helpers.py`

这段代码是一个名为`user_friendly_output`的函数，它使用了`logging`库来自动处理日志输出。这个函数的作用是输出一条用户友好的消息，包括在聊天应用中发送给用户的消息。

具体来说，这个函数接受四个参数：

1. `message`：要输出的消息
2. `level`：设置输出的日志级别，可以传递给`logging.getLogger`函数来控制级别的深度
3. `title`：在输出的消息中添加一个标题，可以传递一个字符串或者一个占位符`{}`，用于在输出的消息中引用这个标题
4. `title_color`：设置标题颜色，可以是任何颜色名称或者一个占位符，例如`"red"`或者`{color.RED}`
5. `preserve_message_color`：如果设置了这个参数，那么在输出消息的时候要保留颜色的状态，否则将消息和标题的彩色部分全部忽略

如果`_chat_plugins`的值为`True`，那么这个函数会尝试加载聊天应用程序的插件，并将插件的结果输出到插件的日志中。


```py
import logging
from typing import Any, Optional

from colorama import Fore

from .config import SPEECH_OUTPUT_LOGGER, USER_FRIENDLY_OUTPUT_LOGGER, _chat_plugins


def user_friendly_output(
    message: str,
    level: int = logging.INFO,
    title: str = "",
    title_color: str = "",
    preserve_message_color: bool = False,
) -> None:
    """Outputs a message to the user in a user-friendly way.

    This function outputs on up to two channels:
    1. The console, in typewriter style
    2. Text To Speech, if configured
    """
    logger = logging.getLogger(USER_FRIENDLY_OUTPUT_LOGGER)

    if _chat_plugins:
        for plugin in _chat_plugins:
            plugin.report(f"{title}: {message}")

    logger.log(
        level,
        message,
        extra={
            "title": title,
            "title_color": title_color,
            "preserve_color": preserve_message_color,
        },
    )


```

这是一个Python函数，作用是打印一个给定属性的值，并在日志中通知。

具体来说，这个函数接受两个参数：

- `title`：打印标题，如果没有指定，则使用默认的"No Title"标题。
- `value`：要打印的值。
- `title_color`：打印标题的颜色，如果没有指定，则使用默认的"No Color"颜色。
- `value_color`：打印值的颜色，如果没有指定，则使用默认的"No Color"颜色。

函数内部首先获取logger实例，然后使用`logger.info`方法打印给定属性的值，并添加以下额外的元数据：

- `title`：标题，已经去除了冒号。
- `extra`：额外的元数据，包含`title`和`color`，如果没有指定，则使用默认的标题和颜色。

接下来，函数内部使用`logger.info`方法的`extra`参数来添加这些额外的元数据，并输出一条带有这些元数据的警告消息。


```py
def print_attribute(
    title: str, value: Any, title_color: str = Fore.GREEN, value_color: str = ""
) -> None:
    logger = logging.getLogger()
    logger.info(
        str(value),
        extra={
            "title": f"{title.rstrip(':')}:",
            "title_color": title_color,
            "color": value_color,
        },
    )


def request_user_double_check(additionalText: Optional[str] = None) -> None:
    if not additionalText:
        additionalText = (
            "Please ensure you've setup and configured everything"
            " correctly. Read https://github.com/Significant-Gravitas/AutoGPT/autogpts/autogpt#readme to "
            "double check. You can also create a github issue or join the discord"
            " and ask there!"
        )

    user_friendly_output(
        additionalText,
        level=logging.WARN,
        title="DOUBLE CHECK CONFIGURATION",
        preserve_message_color=True,
    )


```

这段代码定义了一个名为 speak 的函数，该函数接受一个字符串参数 message 和一个整数参数 level，然后使用logging.getLogger() 方法来获取一个名为 SPEECH_OUTPUT_LOGGER 的 logger 实例，并将 level 和 message 作为参数传递给该 instance 的 log() 方法。

具体来说，这段代码的作用是向名为 SPEECH_OUTPUT_LOGGER 的 logger 实例发送一个指定级别的消息，并将其记录到日志中。其中，level 参数表示要设置的日志级别，可以使用 int(logging.INFO) 来获取当前日志级别的整数类型。如果 level 不确定或者为负数，则会将其默认为 logging.INFO。而 message 参数则是要传递给 logger 的消息，它的类型必须是 str 类型。


```py
def speak(message: str, level: int = logging.INFO) -> None:
    logging.getLogger(SPEECH_OUTPUT_LOGGER).log(level, message)

```

# `autogpts/autogpt/autogpt/logs/log_cycle.py`

这段代码的作用是读取并处理一个机器人领域的日志数据，用于训练和评估机器人的性能。具体来说，它将读取以下文件：

1. log.json：存储了机器人在每次运行时产生的所有日志记录，包括时间戳、日志类型、日志数据等信息。
2. current_context.json：存储了机器人当前所处的上下文信息，包括任务、状态、执行动作等信息。
3. next_action.json：存储了机器人从当前状态预测出的下一个动作，用于控制机器人的执行流程。
4. prompt_summary.json：存储了机器人从当前状态得到的摘要信息，用于在机器人启动时加载已知世界。
5. supervisor_feedback.txt：存储了机器人从上级获取的反馈信息，用于评估机器人的表现。
6. prompt_supervisor_feedback.json：存储了机器人向上级发送的反馈信息，用于更新上级的评估结果。
7. user_input.txt：存储了用户输入的指令或反馈信息，用于更新机器人的策略。

通过读取这些文件，机器人可以获取当前状态、执行动作等信息，从而能够更好地完成任务。此外，这段代码还引入了一个常见的日志处理库——Python的json库，以及pathlib库，用于处理文件和目录。


```py
import json
import os
from pathlib import Path
from typing import Any, Dict, Union

from .config import LOG_DIR

DEFAULT_PREFIX = "agent"
CURRENT_CONTEXT_FILE_NAME = "current_context.json"
NEXT_ACTION_FILE_NAME = "next_action.json"
PROMPT_SUMMARY_FILE_NAME = "prompt_summary.json"
SUMMARY_FILE_NAME = "summary.txt"
SUPERVISOR_FEEDBACK_FILE_NAME = "supervisor_feedback.txt"
PROMPT_SUPERVISOR_FEEDBACK_FILE_NAME = "prompt_supervisor_feedback.json"
USER_INPUT_FILE_NAME = "user_input.txt"


```

This is a simple library that provides methods for logging cycle data for an AI system. The library has two main functions: `create_nested_directory` and `create_inner_directory`. These functions are used to create a nested directory and an inner directory, respectively, where the log data for each cycle will be saved.

Additionally, there is a `log_cycle` function, which takes in the AI name, created at time, cycle count, and data to be logged. This function creates a nested directory with the name of the log file, based on the current cycle count, and writes the log data to the file. The log data is written in JSON format, and the file is saved with the name based on the current cycle count.

To use this library, an instance of the `CycleLogger` class should be created and the `create_nested_directory` and `create_inner_directory` functions should be called to create the directory structure. The AI name and log file name should also be set.


```py
class LogCycleHandler:
    """
    A class for logging cycle data.
    """

    def __init__(self):
        self.log_count_within_cycle = 0

    def create_outer_directory(self, ai_name: str, created_at: str) -> Path:
        if os.environ.get("OVERWRITE_DEBUG") == "1":
            outer_folder_name = "auto_gpt"
        else:
            ai_name_short = self.get_agent_short_name(ai_name)
            outer_folder_name = f"{created_at}_{ai_name_short}"

        outer_folder_path = LOG_DIR / "DEBUG" / outer_folder_name
        if not outer_folder_path.exists():
            outer_folder_path.mkdir(parents=True)

        return outer_folder_path

    def get_agent_short_name(self, ai_name: str) -> str:
        return ai_name[:15].rstrip() if ai_name else DEFAULT_PREFIX

    def create_inner_directory(self, outer_folder_path: Path, cycle_count: int) -> Path:
        nested_folder_name = str(cycle_count).zfill(3)
        nested_folder_path = outer_folder_path / nested_folder_name
        if not nested_folder_path.exists():
            nested_folder_path.mkdir()

        return nested_folder_path

    def create_nested_directory(
        self, ai_name: str, created_at: str, cycle_count: int
    ) -> Path:
        outer_folder_path = self.create_outer_directory(ai_name, created_at)
        nested_folder_path = self.create_inner_directory(outer_folder_path, cycle_count)

        return nested_folder_path

    def log_cycle(
        self,
        ai_name: str,
        created_at: str,
        cycle_count: int,
        data: Union[Dict[str, Any], Any],
        file_name: str,
    ) -> None:
        """
        Log cycle data to a JSON file.

        Args:
            data (Any): The data to be logged.
            file_name (str): The name of the file to save the logged data.
        """
        cycle_log_dir = self.create_nested_directory(ai_name, created_at, cycle_count)

        json_data = json.dumps(data, ensure_ascii=False, indent=4)
        log_file_path = cycle_log_dir / f"{self.log_count_within_cycle}_{file_name}"

        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(json_data + "\n")

        self.log_count_within_cycle += 1

```

# `autogpts/autogpt/autogpt/logs/utils.py`

这段代码使用了Python标准库中的正则表达式函数`re`，来实现从字符串中删除颜色代码。具体来说，代码的作用是定义了一个名为`remove_color_codes`的函数，它接受一个字符串参数`s`，并返回一个修改后的字符串。

在函数内部，使用正则表达式来查找给定字符串中的所有颜色代码，包括`\x1B`代码，`[@-Z\\-_]`，`[[0-?]*[ -/]*[@-~]`，以及`~`。然后使用`re.sub`函数将这些代码替换为空字符串，最终返回修改后的字符串。

总之，这段代码的作用是移除给定字符串中的所有颜色代码，使其不再影响结果输出。


```py
import re


def remove_color_codes(s: str) -> str:
    return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", s)

```

# `autogpts/autogpt/autogpt/logs/__init__.py`

这段代码定义了一系列辅助函数和类，用于将日志循环过程中的信息记录到文件中，以便于用户查看。具体来说，这段代码的作用是：

1. 从 `.helpers` 包中导入 `user_friendly_output` 函数，用于将输出信息以用户友好的方式呈现；
2. 从 `.log_cycle` 包中导入若干函数和类，包括 `CURRENT_CONTEXT_FILE_NAME`、`NEXT_ACTION_FILE_NAME`、`PROMPT_SUMMARY_FILE_NAME`、`PROMPT_SUPERVISOR_FEEDBACK_FILE_NAME`、`SUMMARY_FILE_NAME` 和 `SUPERVISOR_FEEDBACK_FILE_NAME`，它们用于将日志循环过程中的信息记录到相应的文件中；
3. 从 `.log_cycle` 包中导入 `LogCycleHandler` 类，用于监听文件记录的事件并输出信息；
4. 在 `LogCycleHandler` 的 `__init__` 方法中，将需要记录到文件的日志循环信息添加到 `SUMMARY_FILE_NAME` 文件中；
5. 创建一个 `user_input_file_name`，未在代码中使用，但可以猜测是一个文件用于输入用户信息。


```py
from .helpers import user_friendly_output
from .log_cycle import (
    CURRENT_CONTEXT_FILE_NAME,
    NEXT_ACTION_FILE_NAME,
    PROMPT_SUMMARY_FILE_NAME,
    PROMPT_SUPERVISOR_FEEDBACK_FILE_NAME,
    SUMMARY_FILE_NAME,
    SUPERVISOR_FEEDBACK_FILE_NAME,
    USER_INPUT_FILE_NAME,
    LogCycleHandler,
)

```

# `autogpts/autogpt/autogpt/memory/vector/memory_item.py`

这段代码是一个自定义的 Python 类，旨在实现一个带有标签的文本分类模型。这个模型使用了多个第三方库：json、logging、typing、ftfy、numpy、pydantic和autogpt.模型实现了从命令行输入文本到输出分类标签的数据预处理、文本处理和模型加载等功能。

具体来说，这个模型包括以下组件：

1. 从 json 文件中读取配置文件，其中包含了模型需要的一些参数和选项。

2. 从 stdin(通常是终端)读取输入文本，并将其保存到一个 NumPy 数组中。

3. 对输入文本进行预处理，包括分词、去除停用词、移除数字等操作。

4. 使用 pydantic 库实现了模型的 Pydantic 类，这样我们就可以使用 pydantic 的类型注来定义模型的输入和输出。

5. 加载预训练的语言模型，并将其保存到 model_providers 变量中。

6. 实现了一些文本处理的函数，包括：

   - chunk_content：对输入文本进行分段处理，每个分段的长度为 128。
   - split_text：把分段后的文本进行词向量拼接，形成一个新的 NumPy 数组。
   - summarize_text：对输入文本进行总结，只保留最重要的前几个词。

7. 实现了从命令行输入文本到输出分类标签的数据预处理。

8. 调用了一些外部库的功能：

   - 引入了 json 库，以便读取和写入 json 文件。
   - 引入了 logging 库，以便记录模型的训练和推理过程中的信息。
   - 引入了 ftfy 库，以便进行预处理文本。
   - 引入了 numpy 库，以便进行数学计算。
   - 引入了 pydantic 库，提供了模型输入和输出的 Pydantic 类型注。
   - 引入了 autogpt.model_providers.ChatMessage 类，以便在模型训练和推理过程中获取 ChatGPT 模型的实例。

9. 最终，我们定义了一个 ChatMessage 类，实现了从命令行输入文本到输出分类标签的数据预处理。


```py
from __future__ import annotations

import json
import logging
from typing import Literal

import ftfy
import numpy as np
from pydantic import BaseModel

from autogpt.config import Config
from autogpt.core.resource.model_providers import ChatMessage
from autogpt.processing.text import chunk_content, split_text, summarize_text

from .utils import Embedding, get_embedding

```

I am an AI assistant, and I have been trained to answer questions about various topics.

To summarize, I can provide information about the following topics:

{self.llm_provider.count_tokens(self.raw_content, Config().embedding_model).暑期班 2022}
""".strip()

   def memory_file(content: str, path: str):
       return MemoryItem.from_text(content, "code_file", {"location": path})

   def from_ai_action(ai_message: ChatMessage, result_message: ChatMessage):
       return MemoryItem.from_text(
           text=result_message.content,
           source_type="agent_history",
           how_to_summarize="if possible, also make clear the link between the command in the assistant's response and the command result. Do not mention the human feedback if there is none",
       )

   def from_webpage(
       content: str, url: str, config: Config, question: str | None = None
   ):
       return MemoryItem.from_text(
           text=content,
           source_type="webpage",
           config=config,
           metadata={"location": url},
           question_for_summary=question,
       )

   def question_answer(self, question: str):
       return f"{self.bot_name} 是你的AI助手，很高兴能够回答你的问题。""
















































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































```py
logger = logging.getLogger(__name__)

MemoryDocType = Literal["webpage", "text_file", "code_file", "agent_history"]


# FIXME: implement validators instead of allowing arbitrary types
class MemoryItem(BaseModel, arbitrary_types_allowed=True):
    """Memory object containing raw content as well as embeddings"""

    raw_content: str
    summary: str
    chunks: list[str]
    chunk_summaries: list[str]
    e_summary: Embedding
    e_chunks: list[Embedding]
    metadata: dict

    def relevance_for(self, query: str, e_query: Embedding | None = None):
        return MemoryItemRelevance.of(self, query, e_query)

    @staticmethod
    def from_text(
        text: str,
        source_type: MemoryDocType,
        config: Config,
        metadata: dict = {},
        how_to_summarize: str | None = None,
        question_for_summary: str | None = None,
    ):
        logger.debug(f"Memorizing text:\n{'-'*32}\n{text}\n{'-'*32}\n")

        # Fix encoding, e.g. removing unicode surrogates (see issue #778)
        text = ftfy.fix_text(text)

        # FIXME: needs ModelProvider
        chunks = [
            chunk
            for chunk, _ in (
                split_text(text, config.embedding_model, config)
                if source_type != "code_file"
                else chunk_content(text, config.embedding_model)
            )
        ]
        logger.debug("Chunks: " + str(chunks))

        chunk_summaries = [
            summary
            for summary, _ in [
                summarize_text(
                    text_chunk,
                    config,
                    instruction=how_to_summarize,
                    question=question_for_summary,
                )
                for text_chunk in chunks
            ]
        ]
        logger.debug("Chunk summaries: " + str(chunk_summaries))

        e_chunks = get_embedding(chunks, config)

        summary = (
            chunk_summaries[0]
            if len(chunks) == 1
            else summarize_text(
                "\n\n".join(chunk_summaries),
                config,
                instruction=how_to_summarize,
                question=question_for_summary,
            )[0]
        )
        logger.debug("Total summary: " + summary)

        # TODO: investigate search performance of weighted average vs summary
        # e_average = np.average(e_chunks, axis=0, weights=[len(c) for c in chunks])
        e_summary = get_embedding(summary, config)

        metadata["source_type"] = source_type

        return MemoryItem(
            raw_content=text,
            summary=summary,
            chunks=chunks,
            chunk_summaries=chunk_summaries,
            e_summary=e_summary,
            e_chunks=e_chunks,
            metadata=metadata,
        )

    @staticmethod
    def from_text_file(content: str, path: str, config: Config):
        return MemoryItem.from_text(content, "text_file", config, {"location": path})

    @staticmethod
    def from_code_file(content: str, path: str):
        # TODO: implement tailored code memories
        return MemoryItem.from_text(content, "code_file", {"location": path})

    @staticmethod
    def from_ai_action(ai_message: ChatMessage, result_message: ChatMessage):
        # The result_message contains either user feedback
        # or the result of the command specified in ai_message

        if ai_message.role != "assistant":
            raise ValueError(f"Invalid role on 'ai_message': {ai_message.role}")

        result = (
            result_message.content
            if result_message.content.startswith("Command")
            else "None"
        )
        user_input = (
            result_message.content
            if result_message.content.startswith("Human feedback")
            else "None"
        )
        memory_content = (
            f"Assistant Reply: {ai_message.content}"
            "\n\n"
            f"Result: {result}"
            "\n\n"
            f"Human Feedback: {user_input}"
        )

        return MemoryItem.from_text(
            text=memory_content,
            source_type="agent_history",
            how_to_summarize="if possible, also make clear the link between the command in the assistant's response and the command result. Do not mention the human feedback if there is none",
        )

    @staticmethod
    def from_webpage(
        content: str, url: str, config: Config, question: str | None = None
    ):
        return MemoryItem.from_text(
            text=content,
            source_type="webpage",
            config=config,
            metadata={"location": url},
            question_for_summary=question,
        )

    def dump(self, calculate_length=False) -> str:
        if calculate_length:
            token_length = self.llm_provider.count_tokens(
                self.raw_content, Config().embedding_model
            )
        return f"""
```

这段代码是一个类MemoryItem的定义，表示一个二进制数据块，其中包含一些元数据(比如文本长度、每个数据段的批次大小、元数据json数据)、摘要信息、数据内容。

在这个类中，有两个方法：__eq__和__get__。

__eq__是用来检查两个MemoryItem是否相等，根据比较两个内存块的原始内容(e_content和raw_content)、批次大小(chunks)以及元数据(summary和metadata)，然后返回两个块是相等的。

__get__方法用于获取一个MemoryItem中的e_content、raw_content、chunks、summary、metadata中的哪一个值。它根据需要将这个值转换成相应的类型(比如，如果摘要信息是一个列表，那么需要将其转换成一个python内置的类型——因为元数据是json字符串，需要将其转换成python内置的json模块中的Dictionary类型的值)。


```py
=============== MemoryItem ===============
Size: {f'{token_length} tokens in ' if calculate_length else ''}{len(self.e_chunks)} chunks
Metadata: {json.dumps(self.metadata, indent=2)}
---------------- SUMMARY -----------------
{self.summary}
------------------ RAW -------------------
{self.raw_content}
==========================================
"""

    def __eq__(self, other: MemoryItem):
        return (
            self.raw_content == other.raw_content
            and self.chunks == other.chunks
            and self.chunk_summaries == other.chunk_summaries
            # Embeddings can either be list[float] or np.ndarray[float32],
            # and for comparison they must be of the same type
            and np.array_equal(
                self.e_summary
                if isinstance(self.e_summary, np.ndarray)
                else np.array(self.e_summary, dtype=np.float32),
                other.e_summary
                if isinstance(other.e_summary, np.ndarray)
                else np.array(other.e_summary, dtype=np.float32),
            )
            and np.array_equal(
                self.e_chunks
                if isinstance(self.e_chunks[0], np.ndarray)
                else [np.array(c, dtype=np.float32) for c in self.e_chunks],
                other.e_chunks
                if isinstance(other.e_chunks[0], np.ndarray)
                else [np.array(c, dtype=np.float32) for c in other.e_chunks],
            )
        )


```

This is a class called "MemoryChunkRelevanceCalculator" which is used to calculate the relevance score of a given memory item. It takes a memory item, a pre-trained embedding to compare it to, and optionally, a query to evaluate the relevance of.

The class has methods to calculate the aggregate relevance score, the most relevant chunk, and a relevance score for each chunk.

The method "calculate\_scores" compares the given memory item to all embeddings of the pre-trained embedding and optionally, the query. It returns the maximum relevance score, the relevance score of the memory summary, and a list of relevance scores for each chunk.

The method "score" returns the aggregate relevance score of the memory item for the given query.

The method "most\_relevant\_chunk" returns the most relevant chunk of the memory item and its score for the given query.


```py
class MemoryItemRelevance(BaseModel):
    """
    Class that encapsulates memory relevance search functionality and data.
    Instances contain a MemoryItem and its relevance scores for a given query.
    """

    memory_item: MemoryItem
    for_query: str
    summary_relevance_score: float
    chunk_relevance_scores: list[float]

    @staticmethod
    def of(
        memory_item: MemoryItem, for_query: str, e_query: Embedding | None = None
    ) -> MemoryItemRelevance:
        e_query = e_query if e_query is not None else get_embedding(for_query)
        _, srs, crs = MemoryItemRelevance.calculate_scores(memory_item, e_query)
        return MemoryItemRelevance(
            for_query=for_query,
            memory_item=memory_item,
            summary_relevance_score=srs,
            chunk_relevance_scores=crs,
        )

    @staticmethod
    def calculate_scores(
        memory: MemoryItem, compare_to: Embedding
    ) -> tuple[float, float, list[float]]:
        """
        Calculates similarity between given embedding and all embeddings of the memory

        Returns:
            float: the aggregate (max) relevance score of the memory
            float: the relevance score of the memory summary
            list: the relevance scores of the memory chunks
        """
        summary_relevance_score = np.dot(memory.e_summary, compare_to)
        chunk_relevance_scores = np.dot(memory.e_chunks, compare_to).tolist()
        logger.debug(f"Relevance of summary: {summary_relevance_score}")
        logger.debug(f"Relevance of chunks: {chunk_relevance_scores}")

        relevance_scores = [summary_relevance_score, *chunk_relevance_scores]
        logger.debug(f"Relevance scores: {relevance_scores}")
        return max(relevance_scores), summary_relevance_score, chunk_relevance_scores

    @property
    def score(self) -> float:
        """The aggregate relevance score of the memory item for the given query"""
        return max([self.summary_relevance_score, *self.chunk_relevance_scores])

    @property
    def most_relevant_chunk(self) -> tuple[str, float]:
        """The most relevant chunk of the memory item + its score for the given query"""
        i_relmax = np.argmax(self.chunk_relevance_scores)
        return self.memory_item.chunks[i_relmax], self.chunk_relevance_scores[i_relmax]

    def __str__(self):
        return (
            f"{self.memory_item.summary} ({self.summary_relevance_score}) "
            f"{self.chunk_relevance_scores}"
        )

```