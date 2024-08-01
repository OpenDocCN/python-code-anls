# `.\DB-GPT-src\dbgpt\util\prompt_util.py`

```py
"""
General prompt helper that can help deal with LLM context window token limitations.

At its core, it calculates available context size by starting with the context window
size of an LLM and reserve token space for the prompt template, and the output.

It provides utility for "repacking" text chunks (retrieved from index) to maximally
make use of the available context window (and thereby reducing the number of LLM calls
needed), or truncating them so that they fit in a single LLM call.
"""

import logging  # 引入日志模块
from string import Formatter  # 从标准库中引入字符串格式化工具 Formatter
from typing import Callable, List, Optional, Sequence, Set  # 引入类型提示

from dbgpt._private.llm_metadata import LLMMetadata  # 导入私有模块 dbgpt._private.llm_metadata 中的 LLMMetadata 类
from dbgpt._private.pydantic import BaseModel, Field, PrivateAttr, model_validator  # 导入私有模块 dbgpt._private.pydantic 中的 BaseModel, Field, PrivateAttr, model_validator
from dbgpt.core.interface.prompt import get_template_vars  # 导入 dbgpt.core.interface.prompt 中的 get_template_vars 函数
from dbgpt.rag.text_splitter.token_splitter import TokenTextSplitter  # 从 dbgpt.rag.text_splitter.token_splitter 导入 TokenTextSplitter 类
from dbgpt.util.global_helper import globals_helper  # 导入 dbgpt.util.global_helper 中的 globals_helper 函数

DEFAULT_PADDING = 5  # 默认填充值为 5
DEFAULT_CHUNK_OVERLAP_RATIO = 0.1  # 默认块重叠比率为 0.1

DEFAULT_CONTEXT_WINDOW = 3000  # tokens，默认上下文窗口大小为 3000 个标记
DEFAULT_NUM_OUTPUTS = 256  # tokens，默认输出的标记数为 256 个标记

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

class PromptHelper(BaseModel):
    """
    Prompt helper.

    General prompt helper that can help deal with LLM context window token limitations.

    At its core, it calculates available context size by starting with the context
    window size of an LLM and reserve token space for the prompt template, and the
    output.

    It provides utility for "repacking" text chunks (retrieved from index) to maximally
    make use of the available context window (and thereby reducing the number of LLM
    calls needed), or truncating them so that they fit in a single LLM call.

    Args:
        context_window (int):                   Context window for the LLM.
        num_output (int):                       Number of outputs for the LLM.
        chunk_overlap_ratio (float):            Chunk overlap as a ratio of chunk size
        chunk_size_limit (Optional[int]):         Maximum chunk size to use.
        tokenizer (Optional[Callable[[str], List]]): Tokenizer to use.
        separator (str):                        Separator for text splitter

    """

    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum context size that will get sent to the LLM.",
    )  # LLM 的最大上下文大小，默认为 DEFAULT_CONTEXT_WINDOW

    num_output: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The amount of token-space to leave in input for generation.",
    )  # 生成时在输入中保留的标记空间量，默认为 DEFAULT_NUM_OUTPUTS

    chunk_overlap_ratio: float = Field(
        default=DEFAULT_CHUNK_OVERLAP_RATIO,
        description="The percentage token amount that each chunk should overlap.",
    )  # 每个块应重叠的标记比例，默认为 DEFAULT_CHUNK_OVERLAP_RATIO

    chunk_size_limit: Optional[int] = Field(
        None, description="The maximum size of a chunk."
    )  # 块的最大大小，默认为 None

    separator: str = Field(
        default=" ", description="The separator when chunking tokens."
    )  # 在分块标记时使用的分隔符，默认为单空格

    _tokenizer: Optional[Callable[[str], List]] = PrivateAttr()  # 私有属性，用于存储标记化函数
    def __init__(
        self,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        num_output: int = DEFAULT_NUM_OUTPUTS,
        chunk_overlap_ratio: float = DEFAULT_CHUNK_OVERLAP_RATIO,
        chunk_size_limit: Optional[int] = None,
        tokenizer: Optional[Callable[[str], List]] = None,
        separator: str = " ",
        **kwargs,
    ) -> None:
        """Init params."""
        # 检查 chunk_overlap_ratio 是否在有效范围内，不在则引发异常
        if chunk_overlap_ratio > 1.0 or chunk_overlap_ratio < 0.0:
            raise ValueError("chunk_overlap_ratio must be a float between 0. and 1.")

        # 调用父类的初始化方法，传递参数以设置对象的初始状态
        super().__init__(
            context_window=context_window,
            num_output=num_output,
            chunk_overlap_ratio=chunk_overlap_ratio,
            chunk_size_limit=chunk_size_limit,
            separator=separator,
            **kwargs,
        )
        # 设置对象的 _tokenizer 属性，如果未提供，则使用全局的默认 tokenizer
        # TODO: make configurable
        self._tokenizer = tokenizer or globals_helper.tokenizer

    def token_count(self, prompt_template: str) -> int:
        """Get token count of prompt template."""
        # 获取空提示文本
        empty_prompt_txt = get_empty_prompt_txt(prompt_template)
        # 使用对象的 tokenizer 对空提示文本进行分词，并返回分词后的数量
        return len(self._tokenizer(empty_prompt_txt))

    @classmethod
    def from_llm_metadata(
        cls,
        llm_metadata: LLMMetadata,
        chunk_overlap_ratio: float = DEFAULT_CHUNK_OVERLAP_RATIO,
        chunk_size_limit: Optional[int] = None,
        tokenizer: Optional[Callable[[str], List]] = None,
        separator: str = " ",
    ) -> "PromptHelper":
        """Create from llm predictor.

        This will autofill values like context_window and num_output.

        """
        # 从 LLMMetadata 对象中获取 context_window 的值
        context_window = llm_metadata.context_window
        # 如果 llm_metadata 中的 num_output 为 -1，则使用默认的 num_output
        if llm_metadata.num_output == -1:
            num_output = DEFAULT_NUM_OUTPUTS
        else:
            num_output = llm_metadata.num_output

        # 使用给定的参数创建并返回 PromptHelper 对象
        return cls(
            context_window=context_window,
            num_output=num_output,
            chunk_overlap_ratio=chunk_overlap_ratio,
            chunk_size_limit=chunk_size_limit,
            tokenizer=tokenizer,
            separator=separator,
        )

    @classmethod
    def class_name(cls) -> str:
        # 返回类的名称字符串 "PromptHelper"
        return "PromptHelper"
    def _get_available_context_size(self, template: str) -> int:
        """Get available context size.

        This is calculated as:
            available context window = total context window
                - input (partially filled prompt)
                - output (room reserved for response)

        Notes:
        - Available context size is further clamped to be non-negative.
        """
        # 获取空白的提示文本
        empty_prompt_txt = get_empty_prompt_txt(template)
        # 计算空白提示文本的标记数
        num_empty_prompt_tokens = len(self._tokenizer(empty_prompt_txt))
        # 计算可用的上下文窗口大小（标记数）
        context_size_tokens = (
            self.context_window - num_empty_prompt_tokens - self.num_output
        )
        if context_size_tokens < 0:
            # 如果计算出的可用上下文大小为负数，则引发错误
            raise ValueError(
                f"Calculated available context size {context_size_tokens} was"
                " not non-negative."
            )
        return context_size_tokens

    def _get_available_chunk_size(
        self, prompt_template: str, num_chunks: int = 1, padding: int = 5
    ) -> int:
        """Get available chunk size.

        This is calculated as:
            available chunk size = available context window  // number_chunks
                - padding

        Notes:
        - By default, we use padding of 5 (to save space for formatting needs).
        - Available chunk size is further clamped to chunk_size_limit if specified.
        """
        # 获取可用的上下文窗口大小
        available_context_size = self._get_available_context_size(prompt_template)
        # 计算可用块大小
        result = available_context_size // num_chunks - padding
        if self.chunk_size_limit is not None:
            # 如果指定了块大小限制，则取结果和限制值中的较小者
            result = min(result, self.chunk_size_limit)
        return result

    def get_text_splitter_given_prompt(
        self,
        prompt_template: str,
        num_chunks: int = 1,
        padding: int = DEFAULT_PADDING,
    ) -> TokenTextSplitter:
        """Get text splitter configured to maximally pack available context window,
        taking into account of given prompt, and desired number of chunks.
        """
        # 获取可用的块大小
        chunk_size = self._get_available_chunk_size(
            prompt_template, num_chunks, padding=padding
        )
        if chunk_size <= 0:
            # 如果块大小不为正数，则引发错误
            raise ValueError(f"Chunk size {chunk_size} is not positive.")
        # 计算块重叠量
        chunk_overlap = int(self.chunk_overlap_ratio * chunk_size)
        # 返回配置好的 TokenTextSplitter 实例
        return TokenTextSplitter(
            separator=self.separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=self._tokenizer,
        )

    def repack(
        self,
        prompt_template: str,
        text_chunks: Sequence[str],
        padding: int = DEFAULT_PADDING,
    ) -> Tuple[TokenTextSplitter, List[str]]:
        """Repack text chunks into token text splitter with given prompt.

        Args:
        - prompt_template: Template for the prompt.
        - text_chunks: List of text chunks to be repacked.
        - padding: Padding to be applied in chunk size calculation.

        Returns:
        - TokenTextSplitter instance.
        - Repacked text chunks as a list.
        """
    ) -> List[str]:
        """
        Repack text chunks to fit available context window.

        This function consolidates text chunks into larger chunks that fit the context window
        defined by the prompt template.

        Args:
            prompt_template (str): The template of the prompt used for text generation.
            padding (Optional[int]): Padding to apply when splitting text.

        Returns:
            List[str]: A list of strings representing the repacked text chunks.
        """

        # 获取基于给定提示模板的文本分割器实例
        text_splitter = self.get_text_splitter_given_prompt(
            prompt_template, padding=padding
        )

        # 将所有非空的文本片段连接成一个字符串，每个片段去除首尾空白字符后再连接
        combined_str = "\n\n".join([c.strip() for c in text_chunks if c.strip()])

        # 使用文本分割器对组合后的字符串进行分割，以适应上下文窗口
        return text_splitter.split_text(combined_str)
# 获取空白的提示文本。
# 替换提示的部分内容为空字符串，对已部分格式化的变量跳过。这用于计算初始的令牌。

def get_empty_prompt_txt(template: str) -> str:
    # 从模板中获取变量名列表，用于填充空字符串的字典。
    template_vars = get_template_vars(template)
    
    # 初始化部分填充的关键字参数字典为一个空字典。
    partial_kargs = {}
    
    # 创建一个包含未在部分关键字参数中的模板变量的空关键字参数字典。
    empty_kwargs = {v: "" for v in template_vars if v not in partial_kargs}
    
    # 将部分填充的关键字参数和空关键字参数合并成一个完整的关键字参数字典。
    all_kwargs = {**partial_kargs, **empty_kwargs}
    
    # 使用所有关键字参数格式化模板，生成最终的提示文本。
    prompt = template.format(**all_kwargs)
    
    # 返回生成的提示文本。
    return prompt
```