# `.\DB-GPT-src\dbgpt\app\scene\chat_data\chat_excel\excel_learning\chat.py`

```py
import json  # 导入处理 JSON 格式数据的模块
from typing import Any, Dict  # 引入类型提示模块中的 Any 和 Dict 类型

from dbgpt.app.scene import BaseChat, ChatScene  # 导入基础聊天类和聊天场景枚举
from dbgpt.core.interface.message import AIMessage, ViewMessage  # 导入消息接口相关类
from dbgpt.util.executor_utils import blocking_func_to_async  # 导入异步执行工具函数
from dbgpt.util.json_utils import EnhancedJSONEncoder  # 导入增强型 JSON 编码器
from dbgpt.util.tracer import trace  # 导入追踪器函数


class ExcelLearning(BaseChat):
    chat_scene: str = ChatScene.ExcelLearning.value()

    def __init__(
        self,
        chat_session_id,
        user_input,
        parent_mode: Any = None,
        select_param: str = None,
        excel_reader: Any = None,
        model_name: str = None,
    ):
        chat_mode = ChatScene.ExcelLearning
        """ 初始化 ExcelLearning 类，设定初始参数 """
        self.excel_file_path = select_param  # 设定 Excel 文件路径参数
        self.excel_reader = excel_reader  # 设定 Excel 读取器
        chat_param = {
            "chat_mode": chat_mode,
            "chat_session_id": chat_session_id,
            "current_user_input": user_input,
            "select_param": select_param,
            "model_name": model_name,
        }
        super().__init__(chat_param=chat_param)  # 调用父类初始化方法
        if parent_mode:
            self.current_message.chat_mode = parent_mode.value()

    @trace()  # 使用追踪器装饰异步方法
    async def generate_input_values(self) -> Dict:
        # colunms, datas = self.excel_reader.get_sample_data()
        colunms, datas = await blocking_func_to_async(
            self._executor, self.excel_reader.get_sample_data
        )  # 异步调用 Excel 读取器获取示例数据
        self.prompt_template.output_parser.update(colunms)  # 更新输出解析模板
        datas.insert(0, colunms)  # 在数据列表中插入列名行

        input_values = {
            "data_example": json.dumps(datas, cls=EnhancedJSONEncoder),  # 将数据转换为 JSON 字符串
            "file_name": self.excel_reader.excel_file_name,  # 获取 Excel 文件名
        }
        return input_values  # 返回生成的输入数值

    def message_adjust(self):
        ### 调整消息中的学习结果
        # TODO: 无法在多轮对话中工作
        view_message = ""
        for message in self.current_message.messages:
            if message.type == ViewMessage.type:
                view_message = message.content  # 获取视图消息内容

        for message in self.current_message.messages:
            if message.type == AIMessage.type:
                message.content = view_message  # 将 AI 消息内容设定为视图消息内容
```