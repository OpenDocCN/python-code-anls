# `.\DB-GPT-src\dbgpt\app\scene\chat_knowledge\v1\prompt_chatglm.py`

```py
# 从 dbgpt._private.config 模块中导入 Config 类
from dbgpt._private.config import Config
# 从 dbgpt.app.scene 模块中导入 AppScenePromptTemplateAdapter 和 ChatScene 类
from dbgpt.app.scene import AppScenePromptTemplateAdapter, ChatScene
# 从 dbgpt.app.scene.chat_normal.out_parser 模块中导入 NormalChatOutputParser 类
from dbgpt.app.scene.chat_normal.out_parser import NormalChatOutputParser
# 从 dbgpt.core 模块中导入 ChatPromptTemplate, HumanPromptTemplate, MessagesPlaceholder, SystemPromptTemplate 类
from dbgpt.core import (
    ChatPromptTemplate,
    HumanPromptTemplate,
    MessagesPlaceholder,
    SystemPromptTemplate,
)

# 创建 Config 类的实例，赋值给 CFG 变量
CFG = Config()

# 定义用于提示场景的说明文本，该文本描述了用户和 AI 之间的对话场景
PROMPT_SCENE_DEFINE = """A chat between a curious user and an artificial intelligence assistant, who very familiar with database related knowledge. 
    The assistant gives helpful, detailed, professional and polite answers to the user's questions. """

# 定义中文的默认提示模板字符串
_DEFAULT_TEMPLATE_ZH = """ 基于以下已知的信息, 专业、简要的回答用户的问题,
            如果无法从提供的内容中获取答案, 请说: "知识库中提供的内容不足以回答此问题" 禁止胡乱编造。 
            已知内容: 
            {context}
            问题:
            {question}
"""
# 定义英文的默认提示模板字符串
_DEFAULT_TEMPLATE_EN = """ Based on the known information below, provide users with professional and concise answers to their questions. If the answer cannot be obtained from the provided content, please say: "The information provided in the knowledge base is not sufficient to answer this question." It is forbidden to make up information randomly. 
            known information: 
            {context}
            question:
            {question}
"""

# 根据 CFG 的语言设置，选择合适的默认提示模板字符串
_DEFAULT_TEMPLATE = (
    _DEFAULT_TEMPLATE_EN if CFG.LANGUAGE == "en" else _DEFAULT_TEMPLATE_ZH
)

# 定义是否需要流式输出的标志
PROMPT_NEED_STREAM_OUT = True

# 创建 ChatPromptTemplate 实例，包含系统提示模板、消息占位符和用户提示模板
prompt = ChatPromptTemplate(
    messages=[
        SystemPromptTemplate.from_template(_DEFAULT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanPromptTemplate.from_template("{question}"),
    ]
)

# 创建 AppScenePromptTemplateAdapter 实例，用于适配提示模板到特定场景
prompt_adapter = AppScenePromptTemplateAdapter(
    prompt=prompt,
    template_scene=ChatScene.ChatKnowledge.value(),
    stream_out=True,
    output_parser=NormalChatOutputParser(is_stream_out=PROMPT_NEED_STREAM_OUT),
    need_historical_messages=False,
)

# 将创建的提示适配器注册到 CFG 的提示模板注册表中，指定语言和模型名称
CFG.prompt_template_registry.register(
    prompt_adapter,
    language=CFG.LANGUAGE,
    is_default=False,
    model_names=["chatglm-6b-int4", "chatglm-6b", "chatglm2-6b", "chatglm2-6b-int4"],
)
```