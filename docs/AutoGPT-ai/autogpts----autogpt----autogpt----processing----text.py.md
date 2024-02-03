# `.\AutoGPT\autogpts\autogpt\autogpt\processing\text.py`

```py
# 导入所需的模块和库
import json
import logging
import math
from typing import Iterator, Optional, TypeVar

import spacy

from autogpt.config import Config
from autogpt.core.prompting import ChatPrompt
from autogpt.core.resource.model_providers import (
    ChatMessage,
    ChatModelProvider,
    ModelTokenizer,
)

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

T = TypeVar("T")

# 将输入序列按照指定长度进行分批处理，最后一批可能长度不足
def batch(
    sequence: list[T], max_batch_length: int, overlap: int = 0
) -> Iterator[list[T]]:
    """
    Batch data from iterable into slices of length N. The last batch may be shorter.

    Example: `batched('ABCDEFGHIJ', 3)` --> `ABC DEF GHI J`
    """
    # 如果指定的批次长度小于1，则抛出数值错误
    if max_batch_length < 1:
        raise ValueError("n must be at least one")
    # 根据指定的批次长度和重叠值对序列进行分批处理
    for i in range(0, len(sequence), max_batch_length - overlap):
        yield sequence[i : i + max_batch_length]

# 将内容分块，使每个块的 token 长度大致相等
def chunk_content(
    content: str,
    max_chunk_length: int,
    tokenizer: ModelTokenizer,
    with_overlap: bool = True,
) -> Iterator[tuple[str, int]]:
    """Split content into chunks of approximately equal token length."""

    MAX_OVERLAP = 200  # limit overlap to save tokens

    # 对内容进行 token 化
    tokenized_text = tokenizer.encode(content)
    total_length = len(tokenized_text)
    n_chunks = math.ceil(total_length / max_chunk_length)

    # 计算每个块的长度
    chunk_length = math.ceil(total_length / n_chunks)
    # 计算重叠值，限制在最大重叠值以内
    overlap = min(max_chunk_length - chunk_length, MAX_OVERLAP) if with_overlap else 0

    # 根据指定的块长度和重叠值对 token 进行分块处理
    for token_batch in batch(tokenized_text, chunk_length + overlap, overlap):
        yield tokenizer.decode(token_batch), len(token_batch)

# 对文本进行摘要生成
async def summarize_text(
    text: str,
    llm_provider: ChatModelProvider,
    config: Config,
    question: Optional[str] = None,
    instruction: Optional[str] = None,
) -> tuple[str, list[tuple[str, str]]]:
    # 如果有问题需要回答
    if question:
        # 如果有指示，抛出数值错误
        if instruction:
            raise ValueError(
                "Parameters 'question' and 'instructions' cannot both be set"
            )

        # 根据问题生成指示
        instruction = (
            f'From the text, answer the question: "{question}". '
            "If the answer is not in the text, indicate this clearly "
            "and concisely state why the text is not suitable to answer the question."
        )
    # 如果没有问题需要回答，并且没有指示
    elif not instruction:
        # 生成默认指示
        instruction = (
            "Summarize or describe the text clearly and concisely, "
            "whichever seems more appropriate."
        )

    # 调用异步处理文本的函数，传入文本、指示、llm_provider和config参数
    return await _process_text(  # type: ignore
        text=text,
        instruction=instruction,
        llm_provider=llm_provider,
        config=config,
    )
# 异步函数，用于从给定文本中提取信息
async def extract_information(
    source_text: str,  # 源文本
    topics_of_interest: list[str],  # 感兴趣的主题列表
    llm_provider: ChatModelProvider,  # 聊天模型提供者
    config: Config,  # 配置信息
) -> list[str]:  # 返回一个字符串列表
    # 格式化感兴趣主题列表
    fmt_topics_list = "\n".join(f"* {topic}." for topic in topics_of_interest)
    # 提示信息
    instruction = (
        "Extract relevant pieces of information about the following topics:\n"
        f"{fmt_topics_list}\n"
        "Reword pieces of information if needed to make them self-explanatory. "
        "Be concise.\n\n"
        "Respond with an `Array<string>` in JSON format AND NOTHING ELSE. "
        'If the text contains no relevant information, return "[]".'
    )
    # 调用 _process_text 函数处理文本
    return await _process_text(  # type: ignore
        text=source_text,
        instruction=instruction,
        output_type=list[str],
        llm_provider=llm_provider,
        config=config,
    )


# 处理文本的内部函数
async def _process_text(
    text: str,  # 待处理的文本
    instruction: str,  # 处理指令
    llm_provider: ChatModelProvider,  # 聊天模型提供者
    config: Config,  # 配置信息
    output_type: type[str | list[str]] = str,  # 输出类型，默认为字符串
) -> tuple[str, list[tuple[str, str]]] | list[str]:  # 返回元组或字符串列表
    """Process text using the OpenAI API for summarization or information extraction

    Params:
        text (str): The text to process.
        instruction (str): Additional instruction for processing.
        llm_provider: LLM provider to use.
        config (Config): The global application config.
        output_type: `str` for summaries or `list[str]` for piece-wise info extraction.

    Returns:
        For summarization: tuple[str, None | list[(summary, chunk)]]
        For piece-wise information extraction: list[str]
    """
    # 如果文本为空，则抛出异常
    if not text.strip():
        raise ValueError("No content")

    # 获取快速LLM模型
    model = config.fast_llm

    # 计算文本长度
    text_tlength = llm_provider.count_tokens(text, model)
    logger.debug(f"Text length: {text_tlength} tokens")

    # 设置最大结果令牌数和最大块长度
    max_result_tokens = 500
    max_chunk_length = llm_provider.get_token_limit(model) - max_result_tokens - 50
    logger.debug(f"Max chunk length: {max_chunk_length} tokens")
    # 如果文本长度小于最大分块长度
    if text_tlength < max_chunk_length:
        # 创建一个聊天提示对象，包括系统消息和用户消息
        prompt = ChatPrompt(
            messages=[
                ChatMessage.system(
                    "The user is going to give you a text enclosed in triple quotes. "
                    f"{instruction}"
                ),
                ChatMessage.user(f'"""{text}"""'),
            ]
        )

        # 记录调试信息
        logger.debug(f"PROCESSING:\n{prompt}")

        # 调用 llm_provider 的 create_chat_completion 方法，生成聊天完成结果
        response = await llm_provider.create_chat_completion(
            model_prompt=prompt.messages,
            model_name=model,
            temperature=0.5,
            max_tokens=max_result_tokens,
            completion_parser=lambda s: (
                json.loads(s.content) if output_type is not str else None
            ),
        )

        # 如果输出类型为列表
        if output_type == list[str]:
            # 记录原始 LLM 响应内容
            logger.debug(f"Raw LLM response: {repr(response.response.content)}")
            # 格式化结果为项目符号列表
            fmt_result_bullet_list = "\n".join(f"* {r}" for r in response.parsed_result)
            logger.debug(
                f"\n{'-'*11} EXTRACTION RESULT {'-'*12}\n"
                f"{fmt_result_bullet_list}\n"
                f"{'-'*42}\n"
            )
            # 返回解析结果列表
            return response.parsed_result
        else:
            # 获取摘要内容
            summary = response.response.content
            logger.debug(f"\n{'-'*16} SUMMARY {'-'*17}\n{summary}\n{'-'*42}\n")
            # 返回摘要内容和原始文本的元组
            return summary.strip(), [(summary, text)]
    # 如果输出类型不是列表，则将文本拆分成多个块，并对每个块进行处理
    else:
        # 将文本拆分成块，根据配置、最大块长度和模型获取的分词器
        chunks = list(
            split_text(
                text,
                config=config,
                max_chunk_length=max_chunk_length,
                tokenizer=llm_provider.get_tokenizer(model),
            )
        )

        processed_results = []
        # 遍历每个块并处理
        for i, (chunk, _) in enumerate(chunks):
            logger.info(f"Processing chunk {i + 1} / {len(chunks)}")
            # 处理文本块
            chunk_result = await _process_text(
                text=chunk,
                instruction=instruction,
                output_type=output_type,
                llm_provider=llm_provider,
                config=config,
            )
            # 将处理结果添加到列表中
            processed_results.extend(
                chunk_result if output_type == list[str] else [chunk_result]
            )

        # 如果输出类型是列表，则返回处理结果列表
        if output_type == list[str]:
            return processed_results
        else:
            # 否则，将所有处理结果合并成一个摘要
            summary, _ = await _process_text(
                "\n\n".join([result[0] for result in processed_results]),
                instruction=(
                    "The text consists of multiple partial summaries. "
                    "Combine these partial summaries into one."
                ),
                llm_provider=llm_provider,
                config=config,
            )
            # 返回合并后的摘要和每个块对应的处理结果
            return summary.strip(), [
                (processed_results[i], chunks[i][0]) for i in range(0, len(chunks))
            ]
def split_text(
    text: str,
    config: Config,
    max_chunk_length: int,
    tokenizer: ModelTokenizer,
    with_overlap: bool = True,
) -> Iterator[tuple[str, int]]:
    """
    Split text into chunks of sentences, with each chunk not exceeding the max length.

    Args:
        text (str): The text to split.
        config (Config): Config object containing the Spacy model setting.
        max_chunk_length (int, optional): The maximum length of a chunk.
        tokenizer (ModelTokenizer): Tokenizer to use for determining chunk length.
        with_overlap (bool, optional): Whether to allow overlap between chunks.

    Yields:
        str: The next chunk of text

    Raises:
        ValueError: when a sentence is longer than the maximum length
    """
    # 计算文本的长度
    text_length = len(tokenizer.encode(text))

    # 如果文本长度小于最大分块长度，则直接返回整个文本
    if text_length < max_chunk_length:
        yield text, text_length
        return

    # 计算需要分成的块数
    n_chunks = math.ceil(text_length / max_chunk_length)
    # 计算每个块的目标长度
    target_chunk_length = math.ceil(text_length / n_chunks)

    # 加载 Spacy 语言模型
    nlp: spacy.language.Language = spacy.load(config.browse_spacy_language_model)
    # 添加句子分割器管道
    nlp.add_pipe("sentencizer")
    # 对文本进行处理，生成句子列表
    doc = nlp(text)
    sentences = [sentence.text.strip() for sentence in doc.sents]

    # 初始化当前块和长度
    current_chunk: list[str] = []
    current_chunk_length = 0
    last_sentence = None
    last_sentence_length = 0

    i = 0
    # 如果当前块不为空，则生成当前块的文本和长度
    if current_chunk:
        yield " ".join(current_chunk), current_chunk_length
```