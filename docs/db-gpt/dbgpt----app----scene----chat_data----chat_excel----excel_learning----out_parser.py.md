# `.\DB-GPT-src\dbgpt\app\scene\chat_data\chat_excel\excel_learning\out_parser.py`

```py
import json  # 导入处理 JSON 数据的模块
import logging  # 导入日志记录模块
from typing import List, NamedTuple  # 导入类型提示模块

from dbgpt.core.interface.output_parser import BaseOutputParser  # 导入基础输出解析器


class ExcelResponse(NamedTuple):
    desciption: str
    clounms: List
    plans: List


logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class LearningExcelOutputParser(BaseOutputParser):
    def __init__(self, is_stream_out: bool, **kwargs):
        super().__init__(is_stream_out=is_stream_out, **kwargs)
        self.is_downgraded = False  # 初始化下降级状态为 False

    def parse_prompt_response(self, model_out_text):
        try:
            clean_str = super().parse_prompt_response(model_out_text)  # 调用父类方法清理模型输出文本
            logger.info(f"parse_prompt_response:{model_out_text},{model_out_text}")  # 记录日志，记录解析的模型输出文本
            response = json.loads(clean_str)  # 解析清理后的 JSON 字符串
            for key in sorted(response):
                if key.strip() == "DataAnalysis":
                    desciption = response[key]  # 提取数据分析描述
                if key.strip() == "ColumnAnalysis":
                    clounms = response[key]  # 提取列分析数据
                if key.strip() == "AnalysisProgram":
                    plans = response[key]  # 提取分析计划数据
            return ExcelResponse(desciption=desciption, clounms=clounms, plans=plans)  # 返回 ExcelResponse 对象
        except Exception as e:
            logger.error(f"parse_prompt_response Faild!{str(e)}")  # 记录解析失败的错误日志
            clounms = []
            for name in self.data_schema:
                clounms.append({name: "-"})  # 创建空的列分析数据列表
            return ExcelResponse(desciption=model_out_text, clounms=clounms, plans=None)  # 返回错误情况下的 ExcelResponse 对象

    def __build_colunms_html(self, clounms_data):
        html_colunms = f"### **Data Structure**\n"  # 构建列结构部分的 HTML 头部
        column_index = 0
        for item in clounms_data:
            column_index += 1
            keys = item.keys()
            for key in keys:
                html_colunms = (
                    html_colunms + f"- **{column_index}.[{key}]**   _{item[key]}_\n"
                )  # 构建每列的 HTML 格式字符串
        return html_colunms  # 返回构建好的列结构 HTML

    def __build_plans_html(self, plans_data):
        html_plans = f"### **Analysis plans**\n"  # 构建分析计划部分的 HTML 头部
        index = 0
        if plans_data:
            for item in plans_data:
                index += 1
                html_plans = html_plans + f"{item} \n"  # 将每个分析计划添加到 HTML 中
        return html_plans  # 返回构建好的分析计划 HTML

    def parse_view_response(self, speak, data, prompt_response) -> str:
        if data and not isinstance(data, str):
            ### tool out data to table view
            html_title = f"### **Data Summary**\n{data.desciption} "  # 构建数据总结部分的 HTML 标题
            html_colunms = self.__build_colunms_html(data.clounms)  # 构建列结构的 HTML
            html_plans = self.__build_plans_html(data.plans)  # 构建分析计划的 HTML

            html = f"""{html_title}\n{html_colunms}\n{html_plans}"""  # 拼接整体 HTML
            return html  # 返回构建好的 HTML
        else:
            return speak  # 如果数据为空或为字符串，返回原始文本
```