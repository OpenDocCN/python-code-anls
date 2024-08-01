# `.\DB-GPT-src\dbgpt\app\scene\chat_normal\prompt.py`

```py
from dbgpt._private.config import Config  # 导入Config类，用于获取配置信息
from dbgpt.app.scene import AppScenePromptTemplateAdapter, ChatScene  # 导入场景相关的类和适配器
from dbgpt.app.scene.chat_normal.out_parser import NormalChatOutputParser  # 导入普通聊天输出解析器
from dbgpt.core import (  # 导入核心模块的多个类
    ChatPromptTemplate,
    HumanPromptTemplate,
    MessagesPlaceholder,
    SystemPromptTemplate,
)

PROMPT_SCENE_DEFINE_EN = "You are a helpful AI assistant."  # 英文场景定义字符串常量
PROMPT_SCENE_DEFINE_ZH = "你是一个有用的 AI 助手。"  # 中文场景定义字符串常量

CFG = Config()  # 创建Config实例，用于获取系统配置信息

# 根据配置的语言选择不同的场景定义字符串常量
PROMPT_SCENE_DEFINE = (
    PROMPT_SCENE_DEFINE_ZH if CFG.LANGUAGE == "zh" else PROMPT_SCENE_DEFINE_EN
)

PROMPT_NEED_STREAM_OUT = True  # 布尔常量，指示是否需要流输出

# 创建聊天提示模板对象，包括系统场景模板、消息占位符和人类输入模板
prompt = ChatPromptTemplate(
    messages=[
        SystemPromptTemplate.from_template(PROMPT_SCENE_DEFINE),  # 使用场景定义字符串创建系统场景模板
        MessagesPlaceholder(variable_name="chat_history"),  # 创建消息占位符，用于聊天历史记录
        HumanPromptTemplate.from_template("{input}"),  # 使用输入模板创建人类输入模板
    ]
)

# 创建应用场景提示模板适配器对象，包括聊天提示模板、聊天场景、流输出、输出解析器和历史消息需要标识
prompt_adapter = AppScenePromptTemplateAdapter(
    prompt=prompt,
    template_scene=ChatScene.ChatNormal.value(),  # 设置聊天场景为普通聊天场景
    stream_out=PROMPT_NEED_STREAM_OUT,  # 设置是否需要流输出
    output_parser=NormalChatOutputParser(is_stream_out=PROMPT_NEED_STREAM_OUT),  # 创建输出解析器
    need_historical_messages=True,  # 指示是否需要历史消息
)

# 将聊天提示模板适配器注册到配置的提示模板注册表中，设为默认
CFG.prompt_template_registry.register(
    prompt_adapter, language=CFG.LANGUAGE, is_default=True
)
```