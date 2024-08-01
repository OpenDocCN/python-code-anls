# `.\DB-GPT-src\dbgpt\core\interface\output_parser.py`

```py
"""The output parser is used to parse the output of an LLM call.

TODO: Make this more general and clear.
"""

from __future__ import annotations

import json  # 导入处理 JSON 格式的模块
import logging  # 导入日志记录模块
from abc import ABC  # 导入抽象基类 ABC
from dataclasses import asdict  # 导入将数据类实例转换为字典的函数
from typing import Any, TypeVar, Union  # 导入类型注解需要的模块

from dbgpt.core import ModelOutput  # 导入模型输出相关的模块
from dbgpt.core.awel import MapOperator  # 导入映射操作符模块
from dbgpt.core.awel.flow import IOField, OperatorCategory, OperatorType, ViewMetadata  # 导入流字段、操作符类别、操作符类型和视图元数据相关的模块
from dbgpt.util.i18n_utils import _  # 导入国际化翻译函数

T = TypeVar("T")  # 声明一个泛型类型变量 T
ResponseTye = Union[str, bytes, ModelOutput]  # 声明一个响应类型，可以是字符串、字节串或模型输出对象

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class BaseOutputParser(MapOperator[ModelOutput, Any], ABC):
    """Class to parse the output of an LLM call.

    Output parsers help structure language model responses.
    """

    metadata = ViewMetadata(
        label=_("Base Output Operator"),
        name="base_output_operator",
        operator_type=OperatorType.TRANSFORM_STREAM,
        category=OperatorCategory.OUTPUT_PARSER,
        description=_("The base LLM out parse."),
        parameters=[],  # 参数为空列表
        inputs=[
            IOField.build_from(
                _("Model Output"),
                "model_output",
                ModelOutput,
                is_list=True,
                description=_("The model output of upstream."),
            )
        ],  # 输入为模型输出的列表
        outputs=[
            IOField.build_from(
                _("Model Output"),
                "model_output",
                str,
                is_list=True,
                description=_("The model output after parsing."),
            )
        ],  # 输出为经解析后的模型输出的列表
    )

    def __init__(self, is_stream_out: bool = True, **kwargs):
        """Create a new output parser."""
        super().__init__(**kwargs)  # 调用父类构造函数初始化
        self.is_stream_out = is_stream_out  # 设置是否流式输出的标志
        self.data_schema = None  # 初始化数据模式为 None

    def update(self, data_schema):
        """Update the data schema.

        TODO: Remove this method.
        """
        self.data_schema = data_schema  # 更新数据模式的方法，用于后续移除

    def __post_process_code(self, code):
        sep = "\n```"  # 分隔符设为换行和三个反引号
        if sep in code:  # 如果分隔符在代码中
            blocks = code.split(sep)  # 按分隔符分割代码块
            if len(blocks) % 2 == 1:  # 如果分块数为奇数
                for i in range(1, len(blocks), 2):
                    blocks[i] = blocks[i].replace("\\_", "_")  # 替换转义后的下划线
            code = sep.join(blocks)  # 连接处理后的代码块
        return code  # 返回处理后的代码
    def parse_model_stream_resp_ex(self, chunk: ResponseTye, skip_echo_len):
        """Parse the output of an LLM call.

        Args:
            chunk (ResponseTye): The output of an LLM call.
            skip_echo_len (int): The length of the prompt to skip.
        """
        # 解析模型响应数据
        data = _parse_model_response(chunk)

        # TODO: 多模型输出处理程序，重新为多模型编写此部分，使用适配器模式。
        # 暂时未实现多模型处理，仍需重构。

        # 从数据中提取模型上下文信息
        model_context = data.get("model_context")
        has_echo = False

        # 检查模型上下文中是否包含提示回显长度信息
        if model_context and "prompt_echo_len_char" in model_context:
            # 获取提示回显长度
            prompt_echo_len_char = int(model_context.get("prompt_echo_len_char", -1))
            # 检查是否存在回显
            has_echo = bool(model_context.get("echo", False))
            # 如果提示回显长度有效，则更新跳过长度
            if prompt_echo_len_char != -1:
                skip_echo_len = prompt_echo_len_char

        # 如果返回的数据没有错误码
        if data.get("error_code", 0) == 0:
            # 如果存在回显，则从文本中跳过指定长度的回显部分
            if has_echo:
                # TODO 根据模型上下文判断
                output = data["text"][skip_echo_len:].strip()
            else:
                output = data["text"].strip()

            # 对输出进行后处理
            output = self.__post_process_code(output)
            return output
        else:
            # 如果有错误码，则将错误信息与文本合并返回
            output = data["text"] + f" (error_code: {data['error_code']})"
            return output

    def parse_model_nostream_resp(self, response: ResponseTye, sep: str):
        """Parse the output of an LLM call."""
        # 解析模型响应数据
        resp_obj_ex = _parse_model_response(response)

        # 如果返回的响应是字符串，则转换为 JSON 对象
        if isinstance(resp_obj_ex, str):
            resp_obj_ex = json.loads(resp_obj_ex)

        # 如果模型返回的错误码为0
        if resp_obj_ex["error_code"] == 0:
            # 获取所有的文本内容
            all_text = resp_obj_ex["text"]

            # 解析返回的文本以获取 AI 的回复部分
            tmp_resp = all_text.split(sep)
            last_index = -1

            # 遍历临时响应列表，查找包含 "assistant:" 的最后一个条目
            for i in range(len(tmp_resp)):
                if tmp_resp[i].find("assistant:") != -1:
                    last_index = i

            # 提取 AI 的响应部分并进行字符串替换
            ai_response = tmp_resp[last_index]
            ai_response = ai_response.replace("assistant:", "")
            ai_response = ai_response.replace("Assistant:", "")
            ai_response = ai_response.replace("ASSISTANT:", "")
            ai_response = ai_response.replace("\\_", "_")
            ai_response = ai_response.replace("\\*", "*")
            ai_response = ai_response.replace("\t", "")

            # 打印非流式 AI 响应的内容
            print("un_stream ai response:", ai_response)
            return ai_response
        else:
            # 如果模型返回错误码不为0，则抛出值错误异常
            raise ValueError(
                f"Model server error! code={resp_obj_ex['error_code']}, error msg is "
                f"{resp_obj_ex['text']}"
            )
    # 检测并修复 JSON 字符串中不合法的结尾，返回修复后的字符串
    def _illegal_json_ends(self, s):
        # 复制输入的 JSON 字符串
        temp_json = s
        # 定义不合法的 JSON 结尾列表
        illegal_json_ends_1 = [", }", ",}"]
        illegal_json_ends_2 = ", ]", ",]"
        
        # 替换第一类不合法结尾为合法结尾
        for illegal_json_end in illegal_json_ends_1:
            temp_json = temp_json.replace(illegal_json_end, " }")
        
        # 替换第二类不合法结尾为合法结尾
        for illegal_json_end in illegal_json_ends_2:
            temp_json = temp_json.replace(illegal_json_end, " ]")
        
        # 返回修复后的 JSON 字符串
        return temp_json

    # 从字符串中提取 JSON 数据，返回提取的 JSON 字符串
    def _extract_json(self, s):
        try:
            # 首先尝试获取简单模式的 JSON 数据，然后获取数组模式的 JSON 数据，并比较长度
            temp_json_simple = self._json_interception(s)
            temp_json_array = self._json_interception(s, True)
            if len(temp_json_simple) > len(temp_json_array):
                temp_json = temp_json_simple
            else:
                temp_json = temp_json_array

            # 如果未能成功获取有效的 JSON 数据，再次尝试获取简单模式的 JSON 数据
            if not temp_json:
                temp_json = self._json_interception(s)

            # 修复可能存在的不合法 JSON 结尾
            temp_json = self._illegal_json_ends(temp_json)
            # 返回提取和修复后的 JSON 字符串
            return temp_json
        except Exception:
            # 捕获异常并抛出带有错误消息的 ValueError 异常
            raise ValueError("Failed to find a valid json in LLM response！" + temp_json)

    # 从字符串中截取 JSON 数据片段，根据 is_json_array 参数决定返回数组或对象形式的 JSON 字符串
    def _json_interception(self, s, is_json_array: bool = False):
        try:
            # 如果需要获取数组形式的 JSON 数据
            if is_json_array:
                # 寻找字符串中的第一个 "[" 符号
                i = s.find("[")
                if i < 0:
                    return ""
                count = 1
                # 从 "[" 开始，逐字符检查，直到找到与之匹配的 "]" 符号
                for j, c in enumerate(s[i + 1 :], start=i + 1):
                    if c == "]":
                        count -= 1
                    elif c == "[":
                        count += 1
                    # 当计数为零时，表示找到完整的 JSON 数组字符串
                    if count == 0:
                        break
                # 断言计数为零，即找到完整的 JSON 数组字符串
                assert count == 0
                # 返回截取到的 JSON 数组字符串
                return s[i : j + 1]
            else:
                # 如果需要获取对象形式的 JSON 数据
                # 寻找字符串中的第一个 "{" 符号
                i = s.find("{")
                if i < 0:
                    return ""
                count = 1
                # 从 "{" 开始，逐字符检查，直到找到与之匹配的 "}" 符号
                for j, c in enumerate(s[i + 1 :], start=i + 1):
                    if c == "}":
                        count -= 1
                    elif c == "{":
                        count += 1
                    # 当计数为零时，表示找到完整的 JSON 对象字符串
                    if count == 0:
                        break
                # 断言计数为零，即找到完整的 JSON 对象字符串
                assert count == 0
                # 返回截取到的 JSON 对象字符串
                return s[i : j + 1]
        except Exception:
            # 捕获异常并返回空字符串，表示未能成功截取有效的 JSON 数据
            return ""
    def parse_prompt_response(self, model_out_text) -> Any:
        """Parse model out text to prompt define response.

        Args:
            model_out_text: The output of an LLM call.

        Returns:
            Any: The parsed output of an LLM call.
        """
        # 去除字符串末尾的空白字符
        cleaned_output = model_out_text.rstrip()
        
        # 如果字符串包含 "```py"，则根据它分割字符串并取后半部分
        if "```json" in cleaned_output:
            _, cleaned_output = cleaned_output.split("```py")
        
        # 如果字符串以 "```json" 开头，则去除开头的 "```py"
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[len("```py") :]
        
        # 如果字符串以 "```" 开头，则去除开头的 "```py"
        if cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[len("```py") :]
        
        # 如果字符串以 "```" 结尾，则去除末尾的 "```py"
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[: -len("```py")]
        
        # 去除字符串两端的空白字符
        cleaned_output = cleaned_output.strip()
        
        # 如果清理后的字符串不以 '{' 开头或 '}' 结尾，则记录日志并尝试提取 JSON 数据
        if not cleaned_output.startswith("{") or not cleaned_output.endswith("}"):
            logger.info("illegal json processing:\n" + cleaned_output)
            cleaned_output = self._extract_json(cleaned_output)
        
        # 如果清理后的字符串为空或长度小于等于 0，则返回原始输入
        if not cleaned_output or len(cleaned_output) <= 0:
            return model_out_text
        
        # 清理字符串中的特殊字符
        cleaned_output = (
            cleaned_output.strip()
            .replace("\\n", " ")
            .replace("\n", " ")
            .replace("\\", " ")
            .replace("\\_", "_")
        )
        
        # 进一步处理不合法的 JSON 结尾
        cleaned_output = self._illegal_json_ends(cleaned_output)
        
        # 返回清理后的输出
        return cleaned_output

    def parse_view_response(
        self, ai_text, data, parse_prompt_response: Any = None
    ) -> str:
        """Parse the AI response info to user view.

        Args:
            ai_text (str): The output of an LLM call.
            data (dict): The data has been handled by some scene.
            parse_prompt_response (Any): The prompt response has been parsed.

        Returns:
            str: The parsed output of an LLM call.

        """
        # 直接返回 AI 的文本输出
        return ai_text

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""
        # 抛出未实现的错误，子类需要实现该方法
        raise NotImplementedError

    async def map(self, input_value: ModelOutput) -> Any:
        """Parse the output of an LLM call.

        Args:
            input_value (ModelOutput): The output of an LLM call.

        Returns:
            Any: The parsed output of an LLM call.
        """
        # 如果当前的 DAG 上下文是流式调用，则调用流式处理方法
        if self.current_dag_context.streaming_call:
            return self.parse_model_stream_resp_ex(input_value, 0)
        else:
            # 否则调用非流式处理方法
            return self.parse_model_nostream_resp(input_value, "###")
# 定义一个函数用于解析模型的响应数据，根据不同的响应类型返回相应的对象或数据
def _parse_model_response(response: ResponseTye):
    # 如果响应为空，则返回空字符串
    if response is None:
        resp_obj_ex = ""
    # 如果响应是 ModelOutput 类型的对象，则将其转换为字典形式
    elif isinstance(response, ModelOutput):
        resp_obj_ex = asdict(response)
    # 如果响应是字符串类型，则将其解析为 JSON 格式数据
    elif isinstance(response, str):
        resp_obj_ex = json.loads(response)
    # 如果响应是字节流类型，则先检查是否包含空字符（null terminator），并去除后再解析为 JSON 格式数据
    elif isinstance(response, bytes):
        if b"\0" in response:
            response = response.replace(b"\0", b"")
        resp_obj_ex = json.loads(response.decode())
    # 如果响应类型不支持，则抛出 ValueError 异常
    else:
        raise ValueError(f"Unsupported response type {type(response)}")
    # 返回处理后的响应对象或数据
    return resp_obj_ex


class SQLOutputParser(BaseOutputParser):
    """Parse the SQL output of an LLM call."""

    def __init__(self, is_stream_out: bool = False, **kwargs):
        """Create a new SQL output parser."""
        super().__init__(is_stream_out=is_stream_out, **kwargs)

    def parse_model_nostream_resp(self, response: ResponseTye, sep: str):
        """Parse the output of an LLM call."""
        # 调用父类方法解析模型的非流式输出响应
        model_out_text = super().parse_model_nostream_resp(response, sep)
        # 清理解析后的字符串，并将其解析为 JSON 格式数据，严格模式
        clean_str = super().parse_prompt_response(model_out_text)
        return json.loads(clean_str, strict=True)
```