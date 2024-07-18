# `.\graphrag\graphrag\index\verbs\text\translate\strategies\openai.py`

```py
# 导入必要的模块和库
import logging  # 导入日志记录模块
import traceback  # 导入异常追踪模块
from typing import Any  # 导入类型提示模块

from datashaper import VerbCallbacks  # 导入数据整形模块中的动词回调函数

import graphrag.config.defaults as defs  # 导入配置文件中的默认设置
from graphrag.config.enums import LLMType  # 从配置文件中导入LLM类型枚举
from graphrag.index.cache import PipelineCache  # 导入管道缓存模块
from graphrag.index.llm import load_llm  # 导入加载LLM模块
from graphrag.index.text_splitting import TokenTextSplitter  # 导入文本分割模块
from graphrag.llm import CompletionLLM  # 导入LLM完成模块

from .defaults import TRANSLATION_PROMPT as DEFAULT_TRANSLATION_PROMPT  # 从当前目录的默认设置中导入翻译提示
from .typing import TextTranslationResult  # 从当前目录的类型定义文件中导入文本翻译结果类型

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


async def run(
    input: str | list[str],  # 输入参数，可以是字符串或字符串列表
    args: dict[str, Any],  # 其他参数作为字典，键是字符串，值可以是任意类型
    callbacks: VerbCallbacks,  # 回调函数集合，基于动词的回调函数
    pipeline_cache: PipelineCache,  # 管道缓存对象
) -> TextTranslationResult:
    """Run the Claim extraction chain."""
    llm_config = args.get("llm", {"type": LLMType.StaticResponse})  # 获取LLM配置，如果未指定，默认为静态响应类型
    llm_type = llm_config.get("type", LLMType.StaticResponse)  # 获取LLM类型，如果未指定，默认为静态响应类型
    llm = load_llm(
        "text_translation",  # 加载LLM的名称
        llm_type,  # 使用的LLM类型
        callbacks,  # 回调函数集合
        pipeline_cache,  # 管道缓存对象
        llm_config,  # LLM配置参数
        chat_only=True,  # 仅限聊天模式
    )
    language = args.get("language", "English")  # 获取语言参数，默认为英语
    prompt = args.get("prompt")  # 获取提示文本参数
    chunk_size = args.get("chunk_size", defs.CHUNK_SIZE)  # 获取块大小参数，默认从配置文件中获取
    chunk_overlap = args.get("chunk_overlap", defs.CHUNK_OVERLAP)  # 获取块重叠参数，默认从配置文件中获取

    input = [input] if isinstance(input, str) else input  # 如果输入是字符串，则转换为字符串列表
    return TextTranslationResult(
        translations=[
            await _translate_text(
                text, language, prompt, llm, chunk_size, chunk_overlap, callbacks
            )
            for text in input  # 对输入列表中的每个文本进行翻译
        ]
    )


async def _translate_text(
    text: str,  # 需要翻译的文本
    language: str,  # 目标语言
    prompt: str | None,  # 翻译提示文本，可选
    llm: CompletionLLM,  # LLM完成对象
    chunk_size: int,  # 块大小
    chunk_overlap: int,  # 块重叠
    callbacks: VerbCallbacks,  # 回调函数集合
) -> str:
    """Translate a single piece of text."""
    splitter = TokenTextSplitter(  # 创建文本分割器对象
        chunk_size=chunk_size,  # 设置块大小
        chunk_overlap=chunk_overlap,  # 设置块重叠
    )

    out = ""  # 初始化输出文本
    chunks = splitter.split_text(text)  # 使用文本分割器将文本分割为块
    for chunk in chunks:  # 遍历每个分割后的文本块
        try:
            result = await llm(
                chunk,  # 当前处理的文本块
                history=[  # 翻译历史记录，包括系统角色和翻译提示文本
                    {
                        "role": "system",  # 角色为系统
                        "content": (prompt or DEFAULT_TRANSLATION_PROMPT),  # 使用提示文本或默认翻译提示文本
                    }
                ],
                variables={"language": language},  # 设置语言变量
            )
            out += result.output or ""  # 将翻译结果追加到输出文本中
        except Exception as e:
            log.exception("error translating text")  # 记录异常日志
            callbacks.error("Error translating text", e, traceback.format_exc())  # 调用回调函数记录翻译异常信息
            out += ""  # 在出现异常时追加空文本到输出

    return out  # 返回完整的翻译结果文本
```