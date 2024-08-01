# `.\DB-GPT-src\dbgpt\agent\util\api_call.py`

```py
"""Module for managing commands and command plugins."""

import json  # 导入处理 JSON 格式数据的模块
import logging  # 导入日志记录模块
import xml.etree.ElementTree as ET  # 导入用于处理 XML 的模块，并使用 ET 别名
from datetime import datetime  # 从 datetime 模块中导入 datetime 类
from typing import Any, Dict, List, Optional, Union  # 导入用于类型注解的模块

from dbgpt._private.pydantic import BaseModel  # 从 dbgpt._private.pydantic 模块导入 BaseModel 类
from dbgpt.agent.core.schema import Status  # 从 dbgpt.agent.core.schema 模块导入 Status 枚举
from dbgpt.util.json_utils import serialize  # 从 dbgpt.util.json_utils 模块导入 serialize 函数
from dbgpt.util.string_utils import extract_content, extract_content_open_ending  # 从 dbgpt.util.string_utils 模块导入两个函数

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class PluginStatus(BaseModel):
    """A class representing the status of a plugin."""

    name: str  # 插件的名称，字符串类型
    location: List[int]  # 插件的位置列表，元素为整数
    args: dict  # 插件的参数，字典类型
    status: Union[Status, str] = Status.TODO.value  # 插件的状态，可以是 Status 枚举类型或字符串，默认为 Status.TODO.value
    logo_url: Optional[str] = None  # 插件的 logo URL，可选字符串，默认为 None
    api_result: Optional[str] = None  # API 调用的结果，可选字符串，默认为 None
    err_msg: Optional[str] = None  # 错误消息，可选字符串，默认为 None
    start_time: float = datetime.now().timestamp() * 1000  # 插件启动时间，浮点数，以毫秒为单位
    end_time: Optional[str] = None  # 插件结束时间，可选字符串，默认为 None

    df: Any = None  # 插件数据，任意类型，默认为 None


class ApiCall:
    """A class representing an API call."""

    agent_prefix = "<api-call>"  # API 调用的前缀字符串
    agent_end = "</api-call>"  # API 调用的结束字符串
    name_prefix = "<name>"  # 名称的前缀字符串
    name_end = "</name>"  # 名称的结束字符串

    def __init__(
        self,
        plugin_generator: Any = None,  # 插件生成器，任意类型，默认为 None
        display_registry: Any = None,  # 显示注册表，任意类型，默认为 None
        backend_rendering: bool = False,  # 后端渲染标志，布尔类型，默认为 False
    ):
        """Create a new ApiCall object."""
        self.plugin_status_map: Dict[str, PluginStatus] = {}  # 插件状态映射字典，键为字符串，值为 PluginStatus 对象

        self.plugin_generator = plugin_generator  # 设置插件生成器
        self.display_registry = display_registry  # 设置显示注册表
        self.start_time = datetime.now().timestamp() * 1000  # 设置 API 调用的启动时间，浮点数，以毫秒为单位
        self.backend_rendering: bool = backend_rendering  # 设置后端渲染标志

    def _is_need_wait_plugin_call(self, api_call_context):
        """Check if waiting is needed for a plugin call."""
        start_agent_count = api_call_context.count(self.agent_prefix)  # 计算 API 调用上下文中 API 调用前缀的数量

        if start_agent_count > 0:  # 如果存在 API 调用前缀
            return True  # 返回 True
        else:
            # 检查末尾的新字符
            check_len = len(self.agent_prefix)  # 获取 API 调用前缀的长度
            last_text = api_call_context[-check_len:]  # 获取 API 调用上下文中末尾的长度为 check_len 的文本
            for i in range(check_len):  # 遍历长度
                text_tmp = last_text[-i:]  # 获取末尾的部分文本
                prefix_tmp = self.agent_prefix[:i]  # 获取部分前缀
                if text_tmp == prefix_tmp:  # 如果末尾文本等于部分前缀
                    return True  # 返回 True
                else:
                    i += 1  # 增加索引
        return False  # 返回 False

    def check_last_plugin_call_ready(self, all_context):
        """Check if the last plugin call is ready."""
        start_agent_count = all_context.count(self.agent_prefix)  # 计算所有上下文中 API 调用前缀的数量
        end_agent_count = all_context.count(self.agent_end)  # 计算所有上下文中 API 调用结束的数量

        if start_agent_count > 0 and start_agent_count == end_agent_count:  # 如果存在 API 调用前缀且数量与结束标记相同
            return True  # 返回 True
        return False  # 返回 False
    # 处理错误的 Markdown 标签
    def _deal_error_md_tags(self, all_context, api_context, include_end: bool = True):
        # 定义可能包含错误信息的 Markdown 标签列表
        error_md_tags = [
            "```",
            "```py",
            "```xml",
            "```py",
            "```markdown",
            "```py",
        ]
        # 根据 include_end 参数确定 Markdown 标签的结尾内容
        if not include_end:
            md_tag_end = ""
        else:
            md_tag_end = "```"
        # 遍历所有的 Markdown 标签
        for tag in error_md_tags:
            # 替换掉错误信息的 Markdown 标签，保留 api_context 的内容
            all_context = all_context.replace(
                tag + api_context + md_tag_end, api_context
            )
            # 替换带换行的错误信息的 Markdown 标签，保留 api_context 的内容
            all_context = all_context.replace(
                tag + "\n" + api_context + "\n" + md_tag_end, api_context
            )
            # 替换带空格的错误信息的 Markdown 标签，保留 api_context 的内容
            all_context = all_context.replace(
                tag + " " + api_context + " " + md_tag_end, api_context
            )
            # 替换掉错误信息的 Markdown 标签，保留 api_context 的内容
            all_context = all_context.replace(tag + api_context, api_context)
        # 返回处理后的所有内容
        return all_context

    # 处理 API 视图的内容
    def api_view_context(self, all_context: str, display_mode: bool = False):
        """返回视图内容。"""
        # 提取内容的 API 调用映射
        call_context_map = extract_content_open_ending(
            all_context, self.agent_prefix, self.agent_end, True
        )
        # 遍历 API 调用映射
        for api_index, api_context in call_context_map.items():
            # 获取 API 状态
            api_status = self.plugin_status_map.get(api_context)
            # 如果 API 状态不为空
            if api_status is not None:
                # 如果处于显示模式
                if display_mode:
                    # 处理错误的 Markdown 标签
                    all_context = self._deal_error_md_tags(all_context, api_context)
                    # 如果 API 状态为失败
                    if Status.FAILED.value == api_status.status:
                        # 获取错误消息
                        err_msg = api_status.err_msg
                        # 替换掉 api_context，显示错误消息和视图
                        all_context = all_context.replace(
                            api_context,
                            f'\n<span style="color:red">Error:</span>{err_msg}\n'
                            + self.to_view_antv_vis(api_status),
                        )
                    else:
                        # 替换掉 api_context，显示视图
                        all_context = all_context.replace(
                            api_context, self.to_view_antv_vis(api_status)
                        )

                else:
                    # 处理错误的 Markdown 标签，不包含结束标记
                    all_context = self._deal_error_md_tags(
                        all_context, api_context, False
                    )
                    # 替换掉 api_context，显示文本视图
                    all_context = all_context.replace(
                        api_context, self.to_view_text(api_status)
                    )

            else:
                # 如果 API 状态为空，显示等待消息
                now_time = datetime.now().timestamp() * 1000
                cost = (now_time - self.start_time) / 1000
                cost_str = "{:.2f}".format(cost)
                # 处理错误的 Markdown 标签
                all_context = self._deal_error_md_tags(all_context, api_context)

                # 替换掉 api_context，显示等待消息和耗时
                all_context = all_context.replace(
                    api_context,
                    f'\n<span style="color:green">Waiting...{cost_str}S</span>\n',
                )

        # 返回处理后的所有内容
        return all_context
    def update_from_context(self, all_context):
        """Modify the plugin status map based on the context."""
        # 从上下文中提取内容，形成 API 上下文映射字典
        api_context_map: Dict[int, str] = extract_content(
            all_context, self.agent_prefix, self.agent_end, True
        )
        # 遍历 API 上下文映射字典的每个元素
        for api_index, api_context in api_context_map.items():
            # 移除 API 上下文中的换行符
            api_context = api_context.replace("\\n", "").replace("\n", "")
            # 解析 API 调用元素为 XML 元素
            api_call_element = ET.fromstring(api_context)
            # 获取 API 名称
            api_name = api_call_element.find("name").text
            # 如果 API 名称包含方括号，则移除方括号
            if api_name.find("[") >= 0 or api_name.find("]") >= 0:
                api_name = api_name.replace("[", "").replace("]", "")
            # 初始化 API 参数字典
            api_args = {}
            # 获取 API 参数的 XML 元素
            args_elements = api_call_element.find("args")
            # 遍历 API 参数的每个子元素
            for child_element in args_elements.iter():
                api_args[child_element.tag] = child_element.text

            # 获取 API 状态对象
            api_status = self.plugin_status_map.get(api_context)
            # 如果 API 状态对象不存在，则创建新的 PluginStatus 对象
            if api_status is None:
                api_status = PluginStatus(
                    name=api_name, location=[api_index], args=api_args
                )
                self.plugin_status_map[api_context] = api_status
            else:
                # 否则，更新 API 状态对象的位置信息
                api_status.location.append(api_index)

    def _to_view_param_str(self, api_status):
        """Convert PluginStatus object to JSON string."""
        # 初始化参数字典
        param = {}
        # 如果 API 状态对象有名称，则添加到参数字典中
        if api_status.name:
            param["name"] = api_status.name
        # 添加 API 状态到参数字典中
        param["status"] = api_status.status
        # 如果 API 状态对象有 logo URL，则添加到参数字典中
        if api_status.logo_url:
            param["logo"] = api_status.logo_url

        # 如果 API 状态对象有错误信息，则添加到参数字典中
        if api_status.err_msg:
            param["err_msg"] = api_status.err_msg

        # 如果 API 状态对象有 API 结果，则添加到参数字典中
        if api_status.api_result:
            param["result"] = api_status.api_result

        # 将参数字典转换为 JSON 格式的字符串
        return json.dumps(param, default=serialize, ensure_ascii=False)

    def to_view_text(self, api_status: PluginStatus):
        """Return the view content as XML."""
        # 创建 dbgpt-view XML 元素
        api_call_element = ET.Element("dbgpt-view")
        # 设置元素的文本内容为 API 状态对象的参数字符串
        api_call_element.text = self._to_view_param_str(api_status)
        # 将 XML 元素转换为 UTF-8 编码的字符串并返回
        result = ET.tostring(api_call_element, encoding="utf-8")
        return result.decode("utf-8")

    def to_view_antv_vis(self, api_status: PluginStatus):
        """Return the visualization content."""
        # 如果启用后端渲染
        if self.backend_rendering:
            # 将 DataFrame 转换为 HTML 表格字符串
            html_table = api_status.df.to_html(
                index=False, escape=False, sparsify=False
            )
            # 去除 HTML 表格字符串中的空白字符和换行符
            table_str = "".join(html_table.split())
            table_str = table_str.replace("\n", " ")
            # 获取 API 参数中的 SQL 语句
            sql = api_status.args["sql"]
            # 构建包含 SQL 语句和 HTML 表格的 HTML 内容
            html = (
                f' \n<div><b>[SQL]{sql}</b></div><div class="w-full overflow-auto">'
                f"{table_str}</div>\n "
            )
            return html
        else:
            # 否则，创建 chart-view XML 元素
            api_call_element = ET.Element("chart-view")
            # 设置元素的 content 属性为 API 状态对象的可视化参数字符串
            api_call_element.attrib["content"] = self._to_antv_vis_param(api_status)
            # 设置元素的文本内容为空字符串
            api_call_element.text = "\n"
            # 将 XML 元素转换为 UTF-8 编码的字符串并返回
            result = ET.tostring(api_call_element, encoding="utf-8")
            return result.decode("utf-8")
    def _to_antv_vis_param(self, api_status: PluginStatus):
        # 初始化空参数字典
        param = {}
        # 如果 API 状态对象有名称，将名称添加到参数字典中
        if api_status.name:
            param["type"] = api_status.name
        # 如果 API 状态对象有参数，将 SQL 参数添加到参数字典中
        if api_status.args:
            param["sql"] = api_status.args["sql"]

        # 初始化数据为任意类型的空列表
        data: Any = []
        # 如果 API 状态对象有 API 结果，将 API 结果赋值给 data 变量
        if api_status.api_result:
            data = api_status.api_result
        # 将 data 变量保存到参数字典中的 "data" 键
        param["data"] = data
        # 返回 JSON 格式的参数字典字符串
        return json.dumps(param, ensure_ascii=False)

    def run_display_sql(self, llm_text, sql_run_func):
        """Run the API calls for displaying SQL data."""
        # 如果需要等待插件调用，并且上次插件调用已经准备好
        if self._is_need_wait_plugin_call(
            llm_text
        ) and self.check_last_plugin_call_ready(llm_text):
            # 更新上下文信息
            self.update_from_context(llm_text)
            # 遍历插件状态映射中的每一项
            for key, value in self.plugin_status_map.items():
                # 如果状态为待处理
                if value.status == Status.TODO.value:
                    # 将状态设置为运行中
                    value.status = Status.RUNNING.value
                    # 输出日志信息，显示执行的 SQL 显示操作
                    logger.info(f"sql display execution:{value.name},{value.args}")
                    try:
                        # 获取 SQL 参数
                        sql = value.args["sql"]
                        # 如果存在 SQL 参数
                        if sql:
                            # 设置参数字典，包含数据框作为 "df" 键的值
                            param = {
                                "df": sql_run_func(sql),
                            }
                            # 设置 API 结果，根据插件名称调用显示注册表中的函数
                            if self.display_registry.is_valid_command(value.name):
                                value.api_result = self.display_registry.call(
                                    value.name, **param
                                )
                            else:
                                value.api_result = self.display_registry.call(
                                    "response_table", **param
                                )

                        # 设置状态为完成
                        value.status = Status.COMPLETE.value
                    except Exception as e:
                        # 设置状态为失败，并记录错误消息
                        value.status = Status.FAILED.value
                        value.err_msg = str(e)
                    # 设置结束时间为当前时间戳的毫秒值
                    value.end_time = datetime.now().timestamp() * 1000
        # 返回 API 视图上下文信息
        return self.api_view_context(llm_text, True)
    def display_sql_llmvis(self, llm_text, sql_run_func):
        """Render charts using the Antv standard protocol.

        Args:
            llm_text: LLM response text  # 接收LLM响应文本作为参数
            sql_run_func: sql run function  # 接收SQL运行函数作为参数

        Returns:
           ChartView protocol text  # 返回图表视图的协议文本
        """
        try:
            if self._is_need_wait_plugin_call(
                llm_text
            ) and self.check_last_plugin_call_ready(llm_text):
                # 等待API调用完成
                self.update_from_context(llm_text)  # 更新上下文信息
                for key, value in self.plugin_status_map.items():
                    if value.status == Status.TODO.value:
                        value.status = Status.RUNNING.value  # 将状态设置为运行中
                        logger.info(f"SQL execution:{value.name},{value.args}")  # 记录SQL执行日志
                        try:
                            sql = value.args["sql"]
                            if sql is not None and len(sql) > 0:
                                data_df = sql_run_func(sql)  # 执行SQL并获取数据DataFrame
                                value.df = data_df  # 将DataFrame存储在插件状态对象中
                                value.api_result = json.loads(
                                    data_df.to_json(
                                        orient="records",
                                        date_format="iso",
                                        date_unit="s",
                                    )
                                )  # 将DataFrame转换为JSON格式存储在插件状态对象中
                                value.status = Status.COMPLETE.value  # 设置状态为完成
                            else:
                                value.status = Status.FAILED.value  # 如果SQL为空，则设置状态为失败
                                value.err_msg = "No executable sql！"  # 设置错误消息

                        except Exception as e:
                            logger.error(f"data prepare exception！{str(e)}")  # 记录数据准备异常
                            value.status = Status.FAILED.value  # 设置状态为失败
                            value.err_msg = str(e)  # 存储异常信息到错误消息字段
                        value.end_time = datetime.now().timestamp() * 1000  # 设置结束时间戳

        except Exception as e:
            logger.error("Api parsing exception", e)  # 记录API解析异常
            raise ValueError("Api parsing exception," + str(e))  # 抛出异常信息

        return self.api_view_context(llm_text, True)  # 返回API视图上下文
    def display_only_sql_vis(self, chart: dict, sql_2_df_func):
        """Display the chart using the vis standard protocol."""
        # 初始化错误消息为None
        err_msg = None
        # 从图表字典中获取SQL查询语句
        sql = chart.get("sql", None)
        try:
            # 准备一个空的参数字典
            param = {}
            # 使用给定的函数将SQL查询转换为DataFrame
            df = sql_2_df_func(sql)
            # 如果SQL为空或长度为0，则直接返回None
            if not sql or len(sql) <= 0:
                return None

            # 设置参数字典的各个字段
            param["sql"] = sql
            param["type"] = chart.get("display_type", "response_table")
            param["title"] = chart.get("title", "")
            param["describe"] = chart.get("thought", "")

            # 将DataFrame转换为JSON格式，并设置到参数字典中的"data"字段
            param["data"] = json.loads(
                df.to_json(orient="records", date_format="iso", date_unit="s")
            )
            # 将参数字典转换为JSON字符串，并进行定制的序列化
            view_json_str = json.dumps(param, default=serialize, ensure_ascii=False)
        except Exception as e:
            # 捕获任何异常，并记录错误信息到日志
            logger.error("parse_view_response error!" + str(e))
            # 准备一个错误信息参数字典，用于错误处理
            err_param = {"sql": f"{sql}", "type": "response_table", "data": []}
            # 记录错误消息字符串
            err_msg = str(e)
            # 将错误信息参数字典转换为JSON字符串，并进行定制的序列化
            view_json_str = json.dumps(err_param, default=serialize, ensure_ascii=False)

        # 构造最终返回的结果字符串，格式化为vis-chart格式
        result = f"```py-chart\n{view_json_str}\n```"
        # 如果有错误消息，则返回带有错误提示的结果字符串
        if err_msg:
            return f"""<span style=\"color:red\">ERROR!</span>{err_msg} \n {result}"""
        else:
            # 否则，返回正常的结果字符串
            return result
    ):
        """Display the dashboard using the vis standard protocol."""
        # 初始化错误消息和视图 JSON 字符串
        err_msg = None
        view_json_str = None

        # 初始化图表项列表
        chart_items = []
        try:
            # 如果图表数据为空，返回错误消息
            if not charts or len(charts) <= 0:
                return "Have no chart data!"
            # 遍历每个图表
            for chart in charts:
                param = {}
                sql = chart.get("sql", "")
                param["sql"] = sql
                param["type"] = chart.get("display_type", "response_table")
                param["title"] = chart.get("title", "")
                param["describe"] = chart.get("thought", "")
                try:
                    # 调用 sql_2_df_func 函数执行 SQL 查询并将结果转换为 JSON
                    df = sql_2_df_func(sql)
                    param["data"] = json.loads(
                        df.to_json(orient="records", date_format="iso", date_unit="s")
                    )
                except Exception as e:
                    # 若出现异常，设置空数据和错误消息
                    param["data"] = []
                    param["err_msg"] = str(e)
                # 将当前图表参数添加到图表项列表中
                chart_items.append(param)

            # 构建仪表盘参数字典
            dashboard_param = {
                "data": chart_items,
                "chart_count": len(chart_items),
                "title": title,
                "display_strategy": "default",
                "style": "default",
            }
            # 将仪表盘参数字典转换为 JSON 字符串
            view_json_str = json.dumps(
                dashboard_param, default=serialize, ensure_ascii=False
            )

        except Exception as e:
            # 若整体操作出现异常，记录错误日志并返回异常信息
            logger.error("parse_view_response error!" + str(e))
            return f"```py\nReport rendering exception！{str(e)}\n```"

        # 将视图 JSON 字符串包装为特定格式的结果字符串
        result = f"```py-dashboard\n{view_json_str}\n```"
        # 如果存在错误消息，返回带有错误提示的结果字符串
        if err_msg:
            return (
                f"""\\n <span style=\"color:red\">ERROR!</span>{err_msg} \n {result}"""
            )
        else:
            # 否则，直接返回结果字符串
            return result
```