# `.\graphrag\graphrag\index\graph\extractors\graph\graph_extractor.py`

```py
# 导入必要的模块和库
import logging  # 导入日志模块
import numbers  # 导入数字模块
import re  # 导入正则表达式模块
import traceback  # 导入异常跟踪模块
from collections.abc import Mapping  # 导入抽象基类中的 Mapping 类
from dataclasses import dataclass  # 导入 dataclass 装饰器
from typing import Any  # 导入 Any 类型

import networkx as nx  # 导入 NetworkX 库，用于操作图结构
import tiktoken  # 导入 tiktoken 模块

# 导入本地模块和类
import graphrag.config.defaults as defs  # 导入 graphrag 库中的默认配置
from graphrag.index.typing import ErrorHandlerFn  # 导入错误处理函数类型
from graphrag.index.utils import clean_str  # 导入清理字符串函数
from graphrag.llm import CompletionLLM  # 导入完成模型类

from .prompts import CONTINUE_PROMPT, GRAPH_EXTRACTION_PROMPT, LOOP_PROMPT  # 导入特定的提示常量

# 定义默认常量
DEFAULT_TUPLE_DELIMITER = "<|>"  # 定义元组分隔符的默认值
DEFAULT_RECORD_DELIMITER = "##"  # 定义记录分隔符的默认值
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"  # 定义完成标志的默认值
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]  # 定义实体类型列表的默认值

@dataclass
class GraphExtractionResult:
    """Unipartite graph extraction result class definition."""

    output: nx.Graph  # 输出属性，存储 NetworkX 图对象
    source_docs: dict[Any, Any]  # 源文档属性，存储任意类型的键值对字典

class GraphExtractor:
    """Unipartite graph extractor class definition."""

    _llm: CompletionLLM  # 私有属性，存储完成模型对象
    _join_descriptions: bool  # 存储描述连接的布尔值
    _tuple_delimiter_key: str  # 存储元组分隔符键的字符串
    _record_delimiter_key: str  # 存储记录分隔符键的字符串
    _entity_types_key: str  # 存储实体类型键的字符串
    _input_text_key: str  # 存储输入文本键的字符串
    _completion_delimiter_key: str  # 存储完成标志键的字符串
    _entity_name_key: str  # 存储实体名称键的字符串
    _input_descriptions_key: str  # 存储输入描述键的字符串
    _extraction_prompt: str  # 存储提取提示的字符串
    _summarization_prompt: str  # 存储总结提示的字符串
    _loop_args: dict[str, Any]  # 存储循环参数的字符串到任意类型值的字典
    _max_gleanings: int  # 存储最大获取数的整数
    _on_error: ErrorHandlerFn  # 存储错误处理函数对象

    def __init__(
        self,
        llm_invoker: CompletionLLM,  # 完成模型调用器参数
        tuple_delimiter_key: str | None = None,  # 元组分隔符键参数，默认为空
        record_delimiter_key: str | None = None,  # 记录分隔符键参数，默认为空
        input_text_key: str | None = None,  # 输入文本键参数，默认为空
        entity_types_key: str | None = None,  # 实体类型键参数，默认为空
        completion_delimiter_key: str | None = None,  # 完成标志键参数，默认为空
        prompt: str | None = None,  # 提示参数，默认为空
        join_descriptions=True,  # 描述连接布尔参数，默认为 True
        encoding_model: str | None = None,  # 编码模型参数，默认为空
        max_gleanings: int | None = None,  # 最大获取数参数，默认为空
        on_error: ErrorHandlerFn | None = None,  # 错误处理函数参数，默认为空
    ):
        """Init method definition."""
        # TODO: streamline construction
        # 初始化方法定义

        # 设置语言模型调用器
        self._llm = llm_invoker
        # 是否连接描述信息
        self._join_descriptions = join_descriptions
        # 输入文本键名，默认为"input_text"
        self._input_text_key = input_text_key or "input_text"
        # 元组分隔键名，默认为"tuple_delimiter"
        self._tuple_delimiter_key = tuple_delimiter_key or "tuple_delimiter"
        # 记录分隔键名，默认为"record_delimiter"
        self._record_delimiter_key = record_delimiter_key or "record_delimiter"
        # 完成分隔键名，默认为"completion_delimiter"
        self._completion_delimiter_key = (
            completion_delimiter_key or "completion_delimiter"
        )
        # 实体类型键名，默认为"entity_types"
        self._entity_types_key = entity_types_key or "entity_types"
        # 提取提示语，默认为 GRAPH_EXTRACTION_PROMPT
        self._extraction_prompt = prompt or GRAPH_EXTRACTION_PROMPT
        # 最大获取数，默认从 defs.ENTITY_EXTRACTION_MAX_GLEANINGS 获取
        self._max_gleanings = (
            max_gleanings
            if max_gleanings is not None
            else defs.ENTITY_EXTRACTION_MAX_GLEANINGS
        )
        # 错误处理方法，默认为无操作 lambda 函数
        self._on_error = on_error or (lambda _e, _s, _d: None)

        # 构造循环参数
        # 获取编码模型，默认为"cl100k_base"的编码
        encoding = tiktoken.get_encoding(encoding_model or "cl100k_base")
        # 编码"YES"字符串
        yes = encoding.encode("YES")
        # 编码"NO"字符串
        no = encoding.encode("NO")
        # 设置循环参数字典
        self._loop_args = {"logit_bias": {yes[0]: 100, no[0]: 100}, "max_tokens": 1}
    ) -> GraphExtractionResult:
        """Call method definition."""
        # 如果未提供提示变量，则初始化为空字典
        if prompt_variables is None:
            prompt_variables = {}
        # 初始化空字典，用于存储所有记录的索引和文本
        all_records: dict[int, str] = {}
        # 初始化空字典，用于存储文档索引和原始文本
        source_doc_map: dict[int, str] = {}

        # 将默认值填充到提示变量中
        prompt_variables = {
            **prompt_variables,
            # 如果未定义元组分隔键，则使用默认值
            self._tuple_delimiter_key: prompt_variables.get(self._tuple_delimiter_key)
            or DEFAULT_TUPLE_DELIMITER,
            # 如果未定义记录分隔键，则使用默认值
            self._record_delimiter_key: prompt_variables.get(self._record_delimiter_key)
            or DEFAULT_RECORD_DELIMITER,
            # 如果未定义完成分隔键，则使用默认值
            self._completion_delimiter_key: prompt_variables.get(
                self._completion_delimiter_key
            )
            or DEFAULT_COMPLETION_DELIMITER,
            # 如果未定义实体类型键，则使用默认值
            self._entity_types_key: ",".join(
                prompt_variables[self._entity_types_key] or DEFAULT_ENTITY_TYPES
            ),
        }

        # 遍历给定的文本列表
        for doc_index, text in enumerate(texts):
            try:
                # 调用实体提取方法
                result = await self._process_document(text, prompt_variables)
                # 将文档索引和原始文本存入映射中
                source_doc_map[doc_index] = text
                # 将文档索引和提取结果存入记录中
                all_records[doc_index] = result
            except Exception as e:
                # 记录异常信息
                logging.exception("error extracting graph")
                # 处理提取过程中的错误
                self._on_error(
                    e,
                    traceback.format_exc(),
                    {
                        "doc_index": doc_index,
                        "text": text,
                    },
                )

        # 处理所有记录的结果，生成最终输出
        output = await self._process_results(
            all_records,
            # 获取元组分隔符的值，若未定义则使用默认值
            prompt_variables.get(self._tuple_delimiter_key, DEFAULT_TUPLE_DELIMITER),
            # 获取记录分隔符的值，若未定义则使用默认值
            prompt_variables.get(self._record_delimiter_key, DEFAULT_RECORD_DELIMITER),
        )

        # 返回图形提取的结果对象，包括输出和原始文档映射
        return GraphExtractionResult(
            output=output,
            source_docs=source_doc_map,
        )

    async def _process_document(
        self, text: str, prompt_variables: dict[str, str]
    # 定义异步方法 _process_results，接受以下参数和返回值类型
    async def _process_results(
        self,
        # 结果字典，键为整数，值为字符串
        results: dict[int, str],
        # 元组分隔符字符串
        tuple_delimiter: str,
        # 记录分隔符字符串
        record_delimiter: str,
    ) -> str:
        # 使用异步方法 _llm 处理提取提示，将变量传递给提示变量，并包含输入文本
        response = await self._llm(
            self._extraction_prompt,
            variables={
                **prompt_variables,
                self._input_text_key: text,
            },
        )
        # 初始化结果字符串，如果响应的输出为空，则赋空字符串
        results = response.output or ""

        # 重复处理以确保最大化实体数量
        for i in range(self._max_gleanings):
            # 使用异步方法 _llm 处理继续提示，包括命名、历史记录
            glean_response = await self._llm(
                CONTINUE_PROMPT,
                name=f"extract-continuation-{i}",
                history=response.history or [],
            )
            # 将 glean_response 的输出追加到结果字符串中，如果输出为空则追加空字符串
            results += glean_response.output or ""

            # 如果这是最后一个 glean，不需要更新继续标志
            if i >= self._max_gleanings - 1:
                break

            # 使用异步方法 _llm 处理循环检查提示，包括命名、历史记录和模型参数
            continuation = await self._llm(
                LOOP_PROMPT,
                name=f"extract-loopcheck-{i}",
                history=glean_response.history or [],
                model_parameters=self._loop_args,
            )
            # 如果 continuation 的输出不是 "YES"，则终止循环
            if continuation.output != "YES":
                break

        # 返回处理后的结果字符串
        return results
# 解包描述信息
def _unpack_descriptions(data: Mapping) -> list[str]:
    # 从传入的数据中获取键为"description"的值
    value = data.get("description", None)
    # 如果值为None，则返回空列表；否则按换行符分割值，并返回列表
    return [] if value is None else value.split("\n")


# 解包来源ID信息
def _unpack_source_ids(data: Mapping) -> list[str]:
    # 从传入的数据中获取键为"source_id"的值
    value = data.get("source_id", None)
    # 如果值为None，则返回空列表；否则按逗号和空格分割值，并返回列表
    return [] if value is None else value.split(", ")
```