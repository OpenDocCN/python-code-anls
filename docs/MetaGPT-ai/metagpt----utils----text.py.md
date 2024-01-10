# `MetaGPT\metagpt\utils\text.py`

```

# 导入必要的模块
from typing import Generator, Sequence
from metagpt.utils.token_counter import TOKEN_MAX, count_string_tokens

# 减少消息长度以适应最大标记大小
def reduce_message_length(
    msgs: Generator[str, None, None],  # 生成器，产生逐渐变短的有效提示字符串
    model_name: str,  # 使用的编码名称
    system_text: str,  # 系统提示
    reserved: int = 0,  # 保留的标记数量
) -> str:  # 返回值为字符串
    """Reduce the length of concatenated message segments to fit within the maximum token size.

    Args:
        msgs: A generator of strings representing progressively shorter valid prompts.
        model_name: The name of the encoding to use. (e.g., "gpt-3.5-turbo")
        system_text: The system prompts.
        reserved: The number of reserved tokens.

    Returns:
        The concatenated message segments reduced to fit within the maximum token size.

    Raises:
        RuntimeError: If it fails to reduce the concatenated message length.
    """

# 生成提示块
def generate_prompt_chunk(
    text: str,  # 要分割的文本
    prompt_template: str,  # 提示的模板，包含一个`{}`占位符
    model_name: str,  # 使用的编码名称
    system_text: str,  # 系统提示
    reserved: int = 0,  # 保留的标记数量
) -> Generator[str, None, None]:  # 返回值为生成器
    """Split the text into chunks of a maximum token size.

    Args:
        text: The text to split.
        prompt_template: The template for the prompt, containing a single `{}` placeholder. For example, "### Reference\n{}".
        model_name: The name of the encoding to use. (e.g., "gpt-3.5-turbo")
        system_text: The system prompts.
        reserved: The number of reserved tokens.

    Yields:
        The chunk of text.
    """

# 将段落拆分为多个部分
def split_paragraph(paragraph: str, sep: str = ".,", count: int = 2) -> list[str]:
    """Split a paragraph into multiple parts.

    Args:
        paragraph: The paragraph to split.
        sep: The separator character.
        count: The number of parts to split the paragraph into.

    Returns:
        A list of split parts of the paragraph.
    """

# 解码具有Unicode转义序列的文本
def decode_unicode_escape(text: str) -> str:
    """Decode a text with unicode escape sequences.

    Args:
        text: The text to decode.

    Returns:
        The decoded text.
    """

# 将列表按数量拆分
def _split_by_count(lst: Sequence, count: int):
    avg = len(lst) // count
    remainder = len(lst) % count
    start = 0
    for i in range(count):
        end = start + avg + (1 if i < remainder else 0)
        yield lst[start:end]
        start = end

# 使用结束符拆分文本
def _split_text_with_ends(text: str, sep: str = "."):
    parts = []
    for i in text:
        parts.append(i)
        if i == sep:
            yield "".join(parts)
            parts = []
    if parts:
        yield "".join(parts)

```