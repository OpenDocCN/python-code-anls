# `.\DB-GPT-src\dbgpt\model\llm\conversation.py`

```py
"""
Fork from fastchat: https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py

Conversation prompt templates.


This code file will be deprecated in the future. 
We have integrated fastchat. For details, see: dbgpt/model/model_adapter.py
"""

# 导入必要的模块
import dataclasses  # 用于支持数据类
from enum import IntEnum, auto  # 用于定义枚举类型和自动编号
from typing import Callable, Dict, List  # 用于类型提示


class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()  # 自动编号，添加单冒号分隔
    ADD_COLON_TWO = auto()  # 自动编号，添加双冒号分隔
    ADD_COLON_SPACE_SINGLE = auto()  # 自动编号，添加单冒号和空格分隔
    NO_COLON_SINGLE = auto()  # 自动编号，无冒号单分隔
    NO_COLON_TWO = auto()  # 自动编号，无冒号双分隔
    ADD_NEW_LINE_SINGLE = auto()  # 自动编号，添加换行单分隔
    LLAMA2 = auto()  # 自动编号，LLAMA2样式
    CHATGLM = auto()  # 自动编号，CHATGLM样式
    CHATML = auto()  # 自动编号，CHATML样式
    CHATINTERN = auto()  # 自动编号，CHATINTERN样式
    DOLLY = auto()  # 自动编号，DOLLY样式
    RWKV = auto()  # 自动编号，RWKV样式
    PHOENIX = auto()  # 自动编号，PHOENIX样式
    ROBIN = auto()  # 自动编号，ROBIN样式


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # 对话模板的名称
    name: str
    # 系统提示信息
    system: str
    # 两个角色
    roles: List[str]
    # 所有消息列表，每个元素是一个角色和消息的列表
    messages: List[List[str]]
    # 少数示例的数量
    offset: int
    # 分隔符样式
    sep_style: SeparatorStyle
    sep: str  # 分隔符
    sep2: str = None  # 可选的第二个分隔符
    # 停止标志（默认为EOS标记）
    stop_str: str = None  # 停止生成的字符串
    # 如果遇到此列表中的任何令牌，则停止生成
    stop_token_ids: List[int] = None  # 停止生成的令牌ID列表

    # 格式化系统消息的函数
    system_formatter: Callable = None

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def update_system_message(self, system_message: str):
        """Update system message"""
        if self.system_formatter:
            self.system = self.system_formatter(system_message)  # 使用系统消息格式化器更新系统消息
        else:
            self.system = system_message  # 直接更新系统消息为给定的系统消息

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])  # 如果是偶数索引，添加消息和空值
            else:
                ret[-1][-1] = msg  # 如果是奇数索引，更新前一个元素的第二个值为消息
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{"role": "system", "content": self.system}]  # 初始化返回结果，包含系统消息

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})  # 如果是偶数索引，添加用户消息
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})  # 如果是奇数索引且消息不为空，添加助手消息
        return ret
    # 创建并返回一个新的Conversation对象，复制当前对象的属性
    def copy(self):
        return Conversation(
            # 复制当前对象的名称属性
            name=self.name,
            # 复制当前对象的系统属性
            system=self.system,
            # 复制当前对象的角色列表属性
            roles=self.roles,
            # 复制当前对象的消息列表属性，使用列表推导式复制每个消息对[x, y]
            messages=[[x, y] for x, y in self.messages],
            # 复制当前对象的偏移量属性
            offset=self.offset,
            # 复制当前对象的分隔样式属性
            sep_style=self.sep_style,
            # 复制当前对象的分隔符属性
            sep=self.sep,
            # 复制当前对象的第二个分隔符属性
            sep2=self.sep2,
            # 复制当前对象的停止字符串属性
            stop_str=self.stop_str,
            # 复制当前对象的停止标记ID属性
            stop_token_ids=self.stop_token_ids,
            # 复制当前对象的系统格式化器属性
            system_formatter=self.system_formatter,
        )

    # 返回一个包含Conversation对象属性的字典表示
    def dict(self):
        return {
            # 设置字典中的模板名称为当前对象的名称属性
            "template_name": self.name,
            # 设置字典中的系统属性为当前对象的系统属性
            "system": self.system,
            # 设置字典中的角色属性为当前对象的角色列表属性
            "roles": self.roles,
            # 设置字典中的消息属性为当前对象的消息列表属性
            "messages": self.messages,
            # 设置字典中的偏移量属性为当前对象的偏移量属性
            "offset": self.offset,
        }
# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}

# Register a new conversation template.
def register_conv_template(template: Conversation, override: bool = False):
    if not override:
        # Ensure the template name is unique in the registry.
        assert (
            template.name not in conv_templates
        ), f"{template.name} has been registered."

    # Add the template to the global registry.
    conv_templates[template.name] = template


# Get a conversation template by name.
def get_conv_template(name: str) -> Conversation:
    return conv_templates[name].copy()


# A template similar to the "one_shot" template but without an example.
register_conv_template(
    Conversation(
        name="zero_shot",
        system="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# Vicuna v1.1 template
register_conv_template(
    Conversation(
        name="vicuna_v1.1",
        system="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

# llama2 template
# Reference: https://github.com/facebookresearch/llama/blob/cfc3fc8c1968d390eb830e65c63865e980873a06/llama/generation.py#L212
register_conv_template(
    Conversation(
        name="llama-2",
        system="<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n",
        roles=("[INST]", "[/INST]"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s><s>",
        stop_token_ids=[2],
        system_formatter=lambda msg: f"<s>[INST] <<SYS>>\n{msg}\n<</SYS>>\n\n",
    )
)

# codellama template
# Reference: https://github.com/facebookresearch/llama/blob/cfc3fc8c1968d390eb830e65c63865e980873a06/llama/generation.py#L212
# Reference2: https://github.com/eosphoros-ai/DB-GPT-Hub/blob/main/README.zh.md
register_conv_template(
    # 创建一个 Conversation 对象，用于表示对话
    Conversation(
        # 对话的名称为 "codellama"
        name="codellama",
        # 系统消息，指示 SQL 终端在示例数据库前执行任务
        system="<s>[INST] <<SYS>>\nI want you to act as a SQL terminal in front of an example database, you need only to return the sql command to me.Below is an instruction that describes a task, Write a response that appropriately completes the request."
        "If you don't know the answer to the request, please don't share false information.\n<</SYS>>\n\n",
        # 角色标记，指示任务的起始和结束
        roles=("[INST]", "[/INST]"),
        # 消息列表为空
        messages=(),
        # 偏移量为0，可能用于标记位置信息
        offset=0,
        # 分隔符样式为 SeparatorStyle.LLAMA2
        sep_style=SeparatorStyle.LLAMA2,
        # 主分隔符为空格
        sep=" ",
        # 次要分隔符为 "</s><s>"
        sep2=" </s><s>",
        # 停止令牌的 ID 列表，可能用于结束对话
        stop_token_ids=[2],
        # 系统格式化函数，用于生成系统消息的格式
        system_formatter=lambda msg: f"<s>[INST] <<SYS>>\n{msg}\n<</SYS>>\n\n",
    )
# 注册一个名为 "alpaca" 的对话模板
register_conv_template(
    # 创建一个名为 "alpaca" 的对话模板对象
    Conversation(
        name="alpaca",
        # 系统字段包含关于任务的说明指令
        system="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        # 角色定义为指令和响应者
        roles=("### Instruction", "### Response
```