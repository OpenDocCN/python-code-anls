# `.\DB-GPT-src\dbgpt\app\scene\chat_dashboard\out_parser.py`

```py
import json
import logging
from typing import List, NamedTuple

from dbgpt.app.scene import ChatScene  # 导入ChatScene模块，用于场景处理
from dbgpt.core.interface.output_parser import BaseOutputParser  # 导入BaseOutputParser接口类


class ChartItem(NamedTuple):
    sql: str        # SQL查询语句
    title: str      # 标题
    thoughts: str   # 思考内容
    showcase: str   # 展示内容


logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class ChatDashboardOutputParser(BaseOutputParser):
    def __init__(self, is_stream_out: bool, **kwargs):
        super().__init__(is_stream_out=is_stream_out, **kwargs)  # 调用父类BaseOutputParser的初始化方法

    def parse_prompt_response(self, model_out_text):
        clean_str = super().parse_prompt_response(model_out_text)  # 调用父类的parse_prompt_response方法清洗模型输出的文本
        print("clean prompt response:", clean_str)  # 打印清洗后的模型输出文本
        response = json.loads(clean_str)  # 将清洗后的文本解析为JSON格式
        chart_items: List[ChartItem] = []  # 初始化一个空列表，用于存放ChartItem对象
        if not isinstance(response, list):
            response = [response]  # 如果response不是列表，则将其转为包含response的列表
        for item in response:
            chart_items.append(
                ChartItem(
                    item["sql"].replace("\\", " "),  # 为ChartItem对象赋值SQL查询语句，替换反斜杠为空格
                    item["title"],                  # 为ChartItem对象赋值标题
                    item["thoughts"],               # 为ChartItem对象赋值思考内容
                    item["showcase"],               # 为ChartItem对象赋值展示内容
                )
            )
        return chart_items  # 返回包含所有ChartItem对象的列表作为解析后的结果

    def parse_view_response(self, speak, data, prompt_response) -> str:
        return json.dumps(data.prepare_dict())  # 将data对象准备好的字典格式化为JSON字符串返回

    @property
    def _type(self) -> str:
        return ChatScene.ChatDashboard.value  # 返回ChatScene模块中ChatDashboard的值作为_type属性
```