# `.\DB-GPT-src\dbgpt\app\scene\chat_knowledge\refine_summary\prompt.py`

```py
# 从dbgpt._private.config模块中导入Config类
from dbgpt._private.config import Config
# 从dbgpt.app.scene模块中导入AppScenePromptTemplateAdapter和ChatScene类
from dbgpt.app.scene import AppScenePromptTemplateAdapter, ChatScene
# 从dbgpt.app.scene.chat_knowledge.refine_summary.out_parser模块中导入ExtractRefineSummaryParser类
from dbgpt.app.scene.chat_knowledge.refine_summary.out_parser import ExtractRefineSummaryParser
# 从dbgpt.core模块中导入ChatPromptTemplate和HumanPromptTemplate类
from dbgpt.core import ChatPromptTemplate, HumanPromptTemplate

# 创建Config类的实例，用于获取配置信息
CFG = Config()

# 定义PROMPT_SCENE_DEFINE变量，描述了用户和AI助理之间的对话场景
PROMPT_SCENE_DEFINE = """A chat between a curious user and an artificial intelligence assistant, who very familiar with database related knowledge. 
The assistant gives helpful, detailed, professional and polite answers to the user's questions."""

# 定义_DEFAULT_TEMPLATE_ZH变量，包含了中文情况下的默认提示模板字符串
_DEFAULT_TEMPLATE_ZH = (
    """我们已经提供了一个到某一点的现有总结:{existing_answer}\n 请根据你之前推理的内容进行最终的总结,总结回答的时候最好按照1.2.3.进行."""
)

# 定义_DEFAULT_TEMPLATE_EN变量，包含了英文情况下的默认提示模板字符串
_DEFAULT_TEMPLATE_EN = """
We have provided an existing summary up to a certain point: {existing_answer}\nWe have the opportunity to refine the existing summary (only if needed) with some more context below. 
\nBased on the previous reasoning, please summarize the final conclusion in accordance with points 1.2.and 3.
"""

# 根据CFG.LANGUAGE的值确定使用哪种语言的默认提示模板
_DEFAULT_TEMPLATE = (
    _DEFAULT_TEMPLATE_EN if CFG.LANGUAGE == "en" else _DEFAULT_TEMPLATE_ZH
)

# 初始化PROMPT_RESPONSE为空字符串
PROMPT_RESPONSE = """"""

# 将默认模板和响应组合成HumanPromptTemplate的实例，用于构建用户的提示消息
prompt = ChatPromptTemplate(
    messages=[
        # 通过模板创建HumanPromptTemplate的实例，用PROMPT_SCENE_DEFINE作为提示内容
        # SystemPromptTemplate.from_template(PROMPT_SCENE_DEFINE),
        HumanPromptTemplate.from_template(_DEFAULT_TEMPLATE + PROMPT_RESPONSE),
    ]
)

# 创建AppScenePromptTemplateAdapter的实例，用于适配特定的场景和模板
prompt_adapter = AppScenePromptTemplateAdapter(
    prompt=prompt,  # 使用上面创建的prompt作为提示消息的模板
    template_scene=ChatScene.ExtractRefineSummary.value(),  # 设定场景为ChatScene中的ExtractRefineSummary
    stream_out=PROMPT_NEED_NEED_STREAM_OUT,  # 指定是否需要输出流
    output_parser=ExtractRefineSummaryParser(is_stream_out=PROMPT_NEED_NEED_STREAM_OUT),  # 指定输出解析器
    need_historical_messages=False,  # 指定是否需要历史消息
)

# 将prompt_adapter注册到prompt_template_registry中，并标记为默认模板
CFG.prompt_template_registry.register(prompt_adapter, is_default=True)
```