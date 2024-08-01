# `.\DB-GPT-src\dbgpt\app\scene\chat_knowledge\v1\out_parser.py`

```py
# 导入日志模块
import logging

# 导入基础输出解析器和类型变量 T
from dbgpt.core.interface.output_parser import BaseOutputParser, T

# 获取当前模块的日志记录器对象
logger = logging.getLogger(__name__)


# 定义一个普通聊天输出解析器类，继承自 BaseOutputParser
class NormalChatOutputParser(BaseOutputParser):
    
    # 方法：解析提示响应
    def parse_prompt_response(self, model_out_text) -> T:
        # 直接返回模型输出的文本
        return model_out_text

    # 方法：获取格式化指令说明
    def get_format_instructions(self) -> str:
        # 未实现具体的格式化指令，故返回空
        pass
```