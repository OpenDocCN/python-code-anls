# `.\DB-GPT-src\dbgpt\app\scene\chat_db\professional_qa\out_parser.py`

```py
# 导入需要的模块或类，这里从dbgpt.core.interface.output_parser中导入BaseOutputParser和T
from dbgpt.core.interface.output_parser import BaseOutputParser, T

# 定义一个名为NormalChatOutputParser的类，继承自BaseOutputParser类
class NormalChatOutputParser(BaseOutputParser):

    # 定义一个方法parse_prompt_response，接收一个model_out_text参数，返回类型为T
    def parse_prompt_response(self, model_out_text) -> T:
        # 直接返回传入的model_out_text参数，不进行任何处理
        return model_out_text

    # 定义一个方法get_format_instructions，返回类型为str
    def get_format_instructions(self) -> str:
        # pass表示该方法当前没有实现任何功能，保留了方法的框架但不执行具体操作
        pass
```