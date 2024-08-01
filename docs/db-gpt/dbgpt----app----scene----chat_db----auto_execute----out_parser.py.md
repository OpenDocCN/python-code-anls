# `.\DB-GPT-src\dbgpt\app\scene\chat_db\auto_execute\out_parser.py`

```py
import json  # 导入处理 JSON 数据的模块
import logging  # 导入日志记录模块
import xml.etree.ElementTree as ET  # 导入处理 XML 数据的模块，重命名为ET
from typing import Dict, NamedTuple  # 导入类型提示相关模块

import sqlparse  # 导入 SQL 解析模块

from dbgpt._private.config import Config  # 导入配置模块中的Config类
from dbgpt.core.interface.output_parser import BaseOutputParser  # 导入输出解析基类
from dbgpt.util.json_utils import serialize  # 导入 JSON 序列化工具函数

from ...exceptions import AppActionException  # 导入应用异常模块

CFG = Config()  # 创建一个Config类的实例对象

class SqlAction(NamedTuple):
    sql: str
    thoughts: Dict
    display: str

    def to_dict(self) -> Dict[str, Dict]:
        return {
            "sql": self.sql,
            "thoughts": self.thoughts,
            "display": self.display,
        }


logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class DbChatOutputParser(BaseOutputParser):
    def __init__(self, is_stream_out: bool, **kwargs):
        super().__init__(is_stream_out=is_stream_out, **kwargs)  # 调用父类BaseOutputParser的构造函数

    def is_sql_statement(self, statement):
        parsed = sqlparse.parse(statement)  # 解析输入的SQL语句
        if not parsed:  # 如果没有解析结果
            return False
        for stmt in parsed:  # 遍历解析结果的语句列表
            if stmt.get_type() != "UNKNOWN":  # 如果解析出的语句类型不是UNKNOWN
                return True  # 判定为SQL语句
        return False  # 否则不是SQL语句

    def parse_prompt_response(self, model_out_text):
        clean_str = super().parse_prompt_response(model_out_text)  # 调用父类方法处理输出文本
        logger.info(f"clean prompt response: {clean_str}")  # 记录处理后的输出文本信息到日志
        # 兼容社区纯SQL输出模型
        if self.is_sql_statement(clean_str):  # 如果输出文本是SQL语句
            return SqlAction(clean_str, "", "")  # 创建一个SqlAction实例对象
        else:  # 否则
            try:
                response = json.loads(clean_str, strict=False)  # 尝试解析为JSON格式
                sql = ""
                thoughts = {}
                display = ""
                for key in sorted(response):  # 遍历排序后的JSON键
                    if key.strip() == "sql":  # 如果键为"sql"
                        sql = response[key]  # 获取对应的值
                    if key.strip() == "thoughts":  # 如果键为"thoughts"
                        thoughts = response[key]  # 获取对应的值
                    if key.strip() == "display_type":  # 如果键为"display_type"
                        display = response[key]  # 获取对应的值
                return SqlAction(sql, thoughts, display)  # 创建一个SqlAction实例对象
            except Exception as e:  # 捕获异常
                logger.error(f"json load failed:{clean_str}")  # 记录错误信息到日志
                return SqlAction("", clean_str, "")  # 返回一个空的SqlAction实例对象
    def parse_view_response(self, speak, data, prompt_response) -> str:
        param = {}  # 初始化空字典，用于存储视图参数
        api_call_element = ET.Element("chart-view")  # 创建XML元素对象，命名为"chart-view"
        err_msg = None  # 初始化错误消息变量为None
        success = False  # 初始化成功标志为False，用于捕获操作成功或失败的状态
        try:
            if not prompt_response.sql or len(prompt_response.sql) <= 0:
                # 如果prompt_response中的sql为空或长度为0，抛出自定义异常
                raise AppActionException("Can not find sql in response", speak)

            df = data(prompt_response.sql)  # 使用prompt_response中的sql查询数据，并赋值给df
            param["type"] = prompt_response.display  # 将prompt_response的display字段赋值给param字典的"type"键
            param["sql"] = prompt_response.sql  # 将prompt_response的sql字段赋值给param字典的"sql"键
            param["data"] = json.loads(
                df.to_json(orient="records", date_format="iso", date_unit="s")
            )  # 将DataFrame df转换为JSON格式并存储在param字典的"data"键中
            view_json_str = json.dumps(param, default=serialize, ensure_ascii=False)  # 将param字典转换为JSON字符串
            success = True  # 操作成功标志置为True
        except Exception as e:
            logger.error("parse_view_response error!" + str(e))  # 记录错误日志
            err_param = {
                "sql": f"{prompt_response.sql}",  # 将引发异常的sql语句存储在err_param字典中的"sql"键
                "type": "response_table",  # 设置err_param字典中的"type"键为"response_table"
                "data": [],  # 设置err_param字典中的"data"键为空列表
            }
            # err_param["err_msg"] = str(e)  # 注释掉的代码，本行用于将异常消息存储在err_param字典中的"err_msg"键
            err_msg = str(e)  # 将异常消息转换为字符串并存储在err_msg变量中
            view_json_str = json.dumps(err_param, default=serialize, ensure_ascii=False)  # 将err_param字典转换为JSON字符串

        # api_call_element.text = view_json_str  # 设置api_call_element的文本内容为view_json_str（已注释掉）
        api_call_element.set("content", view_json_str)  # 设置api_call_element的"content"属性为view_json_str
        result = ET.tostring(api_call_element, encoding="utf-8")  # 将api_call_element转换为UTF-8编码的字符串

        if not success:
            view_content = (
                f'{speak} \\n <span style="color:red">ERROR!</span>'  # 构建包含错误信息的视图内容字符串
                f"{err_msg} \n {result.decode('utf-8')}"  # 将错误消息、XML结果转换为UTF-8并添加到视图内容字符串中
            )
            raise AppActionException("Generate view content failed", view_content)  # 抛出生成视图内容失败的异常
        else:
            return speak + "\n" + result.decode("utf-8")  # 返回包含speak和UTF-8编码结果的字符串
```