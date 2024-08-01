# `.\DB-GPT-src\dbgpt\app\scene\chat_normal\out_parser.py`

```py
import logging  # 导入 logging 模块，用于记录日志信息

from dbgpt.core.interface.output_parser import BaseOutputParser, T  # 导入自定义模块和类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class NormalChatOutputParser(BaseOutputParser):
    def parse_prompt_response(self, model_out_text) -> T:
        return model_out_text  # 解析模型输出的文本并返回

    def get_format_instructions(self) -> str:
        pass  # 获取格式化指令的方法，目前未实现具体逻辑，故使用 pass 占位符
```