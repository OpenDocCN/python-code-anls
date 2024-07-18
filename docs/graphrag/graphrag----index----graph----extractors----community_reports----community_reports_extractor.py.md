# `.\graphrag\graphrag\index\graph\extractors\community_reports\community_reports_extractor.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'CommunityReportsResult' and 'CommunityReportsExtractor' models."""

# 导入日志记录模块
import logging
# 导入异常回溯模块，用于记录异常信息
import traceback
# 导入用于定义数据类的模块
from dataclasses import dataclass
# 导入用于类型提示的模块
from typing import Any

# 导入自定义模块中的函数和类
from graphrag.index.typing import ErrorHandlerFn
from graphrag.index.utils import dict_has_keys_with_types
from graphrag.llm import CompletionLLM

# 导入本地的 prompt 常量
from .prompts import COMMUNITY_REPORT_PROMPT

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)


@dataclass
class CommunityReportsResult:
    """Community reports result class definition."""

    # 输出文本信息
    output: str
    # 结构化输出信息，为字典形式
    structured_output: dict


class CommunityReportsExtractor:
    """Community reports extractor class definition."""

    # 定义私有成员变量，下面几行用于初始化这些变量
    _llm: CompletionLLM  # LLM 完成类实例
    _input_text_key: str  # 输入文本关键字
    _extraction_prompt: str  # 提取信息的 prompt
    _output_formatter_prompt: str  # 输出格式化 prompt
    _on_error: ErrorHandlerFn  # 错误处理函数
    _max_report_length: int  # 最大报告长度

    def __init__(
        self,
        llm_invoker: CompletionLLM,
        input_text_key: str | None = None,
        extraction_prompt: str | None = None,
        on_error: ErrorHandlerFn | None = None,
        max_report_length: int | None = None,
    ):
        """Init method definition."""
        # 初始化成员变量
        self._llm = llm_invoker
        # 如果没有指定输入文本关键字，则默认为 "input_text"
        self._input_text_key = input_text_key or "input_text"
        # 如果没有指定提取信息的 prompt，则使用默认的 COMMUNITY_REPORT_PROMPT
        self._extraction_prompt = extraction_prompt or COMMUNITY_REPORT_PROMPT
        # 如果没有指定错误处理函数，则使用一个空的默认函数
        self._on_error = on_error or (lambda _e, _s, _d: None)
        # 如果没有指定最大报告长度，则默认为 1500
        self._max_report_length = max_report_length or 1500

    async def __call__(self, inputs: dict[str, Any]):
        """Call method definition."""
        # 初始化 output 为 None
        output = None
        try:
            # 调用 LLM 实例，生成社区报告
            response = (
                await self._llm(
                    self._extraction_prompt,
                    json=True,
                    name="create_community_report",
                    variables={self._input_text_key: inputs[self._input_text_key]},
                    # 检查响应是否符合预期的结构
                    is_response_valid=lambda x: dict_has_keys_with_types(
                        x,
                        [
                            ("title", str),
                            ("summary", str),
                            ("findings", list),
                            ("rating", float),
                            ("rating_explanation", str),
                        ],
                    ),
                    model_parameters={"max_tokens": self._max_report_length},
                )
                # 如果 LLM 调用失败，则返回一个空字典
                or {}
            )
            # 提取响应的 JSON 数据，如果无法提取则返回一个空字典
            output = response.json or {}
        except Exception as e:
            # 记录异常信息到日志中
            log.exception("error generating community report")
            # 调用自定义的错误处理函数
            self._on_error(e, traceback.format_exc(), None)
            # 将 output 设置为空字典
            output = {}

        # 获取文本格式的输出结果
        text_output = self._get_text_output(output)
        # 返回 CommunityReportsResult 对象，包含结构化输出和文本输出
        return CommunityReportsResult(
            structured_output=output,
            output=text_output,
        )
    # 定义一个方法来生成文本输出，接受一个解析后的输出字典作为参数，返回一个字符串
    def _get_text_output(self, parsed_output: dict) -> str:
        # 从解析后的输出字典中获取报告标题，如果不存在则默认为 "Report"
        title = parsed_output.get("title", "Report")
        # 从解析后的输出字典中获取报告摘要，如果不存在则为空字符串
        summary = parsed_output.get("summary", "")
        # 从解析后的输出字典中获取发现的列表，如果不存在则为空列表
        findings = parsed_output.get("findings", [])

        # 定义一个内部函数，用于获取每个发现的摘要
        def finding_summary(finding: dict):
            # 如果发现是字符串类型，则直接返回该字符串
            if isinstance(finding, str):
                return finding
            # 否则返回发现字典中的摘要字段
            return finding.get("summary")

        # 定义一个内部函数，用于获取每个发现的详细解释
        def finding_explanation(finding: dict):
            # 如果发现是字符串类型，则返回空字符串
            if isinstance(finding, str):
                return ""
            # 否则返回发现字典中的解释字段
            return finding.get("explanation")

        # 使用列表推导式生成报告的各个章节，每个章节由发现的摘要和详细解释构成
        report_sections = "\n\n".join(
            f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
        )
        
        # 返回最终的报告字符串，包括标题、摘要和所有章节内容
        return f"# {title}\n\n{summary}\n\n{report_sections}"
```