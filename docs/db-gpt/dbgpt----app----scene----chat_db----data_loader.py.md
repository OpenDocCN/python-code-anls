# `.\DB-GPT-src\dbgpt\app\scene\chat_db\data_loader.py`

```py
import json  # 导入json模块，用于处理JSON数据
import logging  # 导入logging模块，用于日志记录
import xml.etree.ElementTree as ET  # 导入ElementTree模块，用于处理XML数据

from dbgpt.util.json_utils import serialize  # 从dbgpt.util.json_utils模块中导入serialize函数


class DbDataLoader:
    def get_table_view_by_conn(self, data, speak, sql: str = None):
        param = {}  # 初始化空字典param，用于存储参数信息
        api_call_element = ET.Element("chart-view")  # 创建名为"chart-view"的XML元素对象
        err_msg = None  # 初始化错误消息变量err_msg为None

        try:
            param["type"] = "response_table"  # 设置param字典的"type"键为"response_table"
            param["sql"] = sql  # 将传入的SQL语句赋值给param字典的"sql"键
            # 将data数据转换为JSON格式，并赋值给param字典的"data"键
            param["data"] = json.loads(
                data.to_json(orient="records", date_format="iso", date_unit="s")
            )
            # 将param字典转换为JSON字符串，其中使用自定义的serialize函数处理对象序列化，不使用ASCII编码
            view_json_str = json.dumps(param, default=serialize, ensure_ascii=False)
        except Exception as e:
            logging.error("parse_view_response error!" + str(e))  # 记录异常日志信息
            # 构建错误时的参数字典err_param
            err_param = {
                "sql": f"{sql}",
                "type": "response_table",
                "err_msg": str(e),
                "data": []
            }
            err_msg = str(e)  # 设置错误消息为异常的字符串表示
            # 将err_param字典转换为JSON字符串，使用自定义的serialize函数处理对象序列化，不使用ASCII编码
            view_json_str = json.dumps(err_param, default=serialize, ensure_ascii=False)

        # 设置api_call_element元素的"text"属性为view_json_str
        # api_call_element.text = view_json_str
        api_call_element.set("content", view_json_str)  # 设置api_call_element元素的"content"属性为view_json_str
        result = ET.tostring(api_call_element, encoding="utf-8")  # 将api_call_element转换为字节串，使用UTF-8编码

        if err_msg:
            # 如果有错误消息，返回包含错误消息和结果的字符串
            return f"""{speak} \\n <span style=\"color:red\">ERROR!</span>{err_msg} \n {result.decode("utf-8")}"""
        else:
            # 如果没有错误消息，返回结果字符串
            return speak + "\n" + result.decode("utf-8")
```