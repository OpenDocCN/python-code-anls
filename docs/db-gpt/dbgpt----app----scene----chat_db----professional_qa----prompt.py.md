# `.\DB-GPT-src\dbgpt\app\scene\chat_db\professional_qa\prompt.py`

```py
# 从 dbgpt._private.config 模块中导入 Config 类
from dbgpt._private.config import Config
# 从 dbgpt.app.scene 模块中导入 AppScenePromptTemplateAdapter 和 ChatScene 类
from dbgpt.app.scene import AppScenePromptTemplateAdapter, ChatScene
# 从 dbgpt.app.scene.chat_db.professional_qa.out_parser 模块中导入 NormalChatOutputParser 类
from dbgpt.app.scene.chat_db.professional_qa.out_parser import NormalChatOutputParser
# 从 dbgpt.core 模块中导入 ChatPromptTemplate, HumanPromptTemplate, MessagesPlaceholder, SystemPromptTemplate 类
from dbgpt.core import (
    ChatPromptTemplate,
    HumanPromptTemplate,
    MessagesPlaceholder,
    SystemPromptTemplate,
)

# 创建 Config 类的实例 CFG
CFG = Config()

# 英文默认模板字符串，提供专业答案及处理不足信息的回答方式
_DEFAULT_TEMPLATE_EN = """
Provide professional answers to requests and questions. If you can't get an answer from what you've provided, say: "Insufficient information in the knowledge base is available to answer this question." Feel free to fudge information.
Use the following tables generate sql if have any table info:
{table_info}

user question:
{input}
think step by step.
"""

# 中文默认模板字符串，提供专业答案及处理不足信息的回答方式
_DEFAULT_TEMPLATE_ZH = """
根据要求和问题，提供专业的答案。如果无法从提供的内容中获取答案，请说：“知识库中提供的信息不足以回答此问题。” 禁止随意捏造信息。

使用以下表结构信息: 
{table_info}

问题:
{input}
一步步思考。
"""

# 根据 CFG 的语言设置选择合适的默认模板字符串
_DEFAULT_TEMPLATE = (
    _DEFAULT_TEMPLATE_EN if CFG.LANGUAGE == "en" else _DEFAULT_TEMPLATE_ZH
)

# 布尔值，表示是否需要流输出
PROMPT_NEED_STREAM_OUT = True

# 创建 ChatPromptTemplate 实例 prompt，包含系统模板、消息占位符和人类输入模板
prompt = ChatPromptTemplate(
    messages=[
        SystemPromptTemplate.from_template(_DEFAULT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanPromptTemplate.from_template("{input}"),
    ]
)

# 创建 AppScenePromptTemplateAdapter 实例 prompt_adapter，用于场景适配
prompt_adapter = AppScenePromptTemplateAdapter(
    prompt=prompt,
    template_scene=ChatScene.ChatWithDbQA.value(),
    stream_out=PROMPT_NEED_STREAM_OUT,
    output_parser=NormalChatOutputParser(is_stream_out=PROMPT_NEED_STREAM_OUT),
    need_historical_messages=True,
)

# 将 prompt_adapter 注册到 CFG 的 prompt_template_registry 中作为默认模板
CFG.prompt_template_registry.register(
    prompt_adapter, language=CFG.LANGUAGE, is_default=True
)
```