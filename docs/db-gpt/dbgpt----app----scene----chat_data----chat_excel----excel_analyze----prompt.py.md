# `.\DB-GPT-src\dbgpt\app\scene\chat_data\chat_excel\excel_analyze\prompt.py`

```py
# 从dbgpt._private.config模块中导入Config类
from dbgpt._private.config import Config
# 从dbgpt.app.scene模块中导入AppScenePromptTemplateAdapter和ChatScene类
from dbgpt.app.scene import AppScenePromptTemplateAdapter, ChatScene
# 从dbgpt.app.scene.chat_data.chat_excel.excel_analyze.out_parser模块中导入ChatExcelOutputParser类
from dbgpt.app.scene.chat_data.chat_excel.excel_analyze.out_parser import (
    ChatExcelOutputParser,
)
# 从dbgpt.core模块中导入ChatPromptTemplate, HumanPromptTemplate, MessagesPlaceholder, SystemPromptTemplate类
from dbgpt.core import (
    ChatPromptTemplate,
    HumanPromptTemplate,
    MessagesPlaceholder,
    SystemPromptTemplate,
)

# 创建Config类的实例对象CFG
CFG = Config()

# 如果CFG的LANGUAGE属性为'en'，则选择英文版的场景定义字符串
_PROMPT_SCENE_DEFINE_EN = "You are a data analysis expert. "
# 否则选择中文版的场景定义字符串
_PROMPT_SCENE_DEFINE_ZH = """你是一个数据分析专家！"""

# 英文版的默认模板字符串，用于数据分析场景的回答模板
_DEFAULT_TEMPLATE_EN = """
Please use the data structure column analysis information generated in the above historical dialogue to answer the user's questions through duckdb sql data analysis under the following constraints..

Constraint:
    1.Please fully understand the user's problem and use duckdb sql for analysis. The analysis content is returned in the output format required below. Please output the sql in the corresponding sql parameter.
    2.Please choose the best one from the display methods given below for data rendering, and put the type name into the name parameter value that returns the required format. If you cannot find the most suitable one, use 'Table' as the display method. , the available data display methods are as follows: {display_type}
    3.The table name that needs to be used in SQL is: {table_name}. Please check the sql you generated and do not use column names that are not in the data structure.
    4.Give priority to answering using data analysis. If the user's question does not involve data analysis, you can answer according to your understanding.
    5.The sql part of the output content is converted to: <api-call><name>[data display mode]</name><args><sql>[correct duckdb data analysis sql]</sql></args></api - call> For this format, please refer to the return format requirements.
    
Please think step by step and give your answer, and make sure your answer is formatted as follows:
    thoughts summary to say to user.<api-call><name>[Data display method]</name><args><sql>[Correct duckdb data analysis sql]</sql></args></api-call>
    
User Questions:
    {user_input}
"""

# 中文版的默认模板字符串，用于数据分析场景的回答模板
_DEFAULT_TEMPLATE_ZH = """
请使用历史对话中的数据结构信息，在满足下面约束条件下通过duckdb sql数据分析回答用户的问题。
约束条件:
    1.请充分理解用户的问题，使用duckdb sql的方式进行分析， 分析内容按下面要求的输出格式返回，sql请输出在对应的sql参数中
    2.请从如下给出的展示方式种选择最优的一种用以进行数据渲染，将类型名称放入返回要求格式的name参数值，如果找不到最合适的则使用'Table'作为展示方式，可用数据展示方式如下: {display_type}
    3.SQL中需要使用的表名是: {table_name},请检查你生成的sql，不要使用没在数据结构中的列名
    4.优先使用数据分析的方式回答，如果用户问题不涉及数据分析内容，你可以按你的理解进行回答
    5.输出内容中sql部分转换为：<api-call><name>[数据显示方式]</name><args><sql>[正确的duckdb数据分析sql]</sql></args></api- call> 这样的格式，参考返回格式要求
    
请一步一步思考，给出回答，并确保你的回答内容格式如下:
    对用户说的想法摘要.<api-call><name>[数据展示方式]</name><args><sql>[正确的duckdb数据分析sql]</sql></args></api-call>

用户问题：{user_input}
"""

# 根据CFG的LANGUAGE属性选择相应的默认模板字符串
_DEFAULT_TEMPLATE = (
    _DEFAULT_TEMPLATE_EN if CFG.LANGUAGE == "en" else _DEFAULT_TEMPLATE_ZH
)

# 根据CFG的LANGUAGE属性选择相应的场景定义字符串
_PROMPT_SCENE_DEFINE = (
    _PROMPT_SCENE_DEFINE_EN if CFG.LANGUAGE == "en" else _PROMPT_SCENE_DEFINE_ZH
)

# 设置一个标志，指示是否需要将输出流输出
PROMPT_NEED_STREAM_OUT = True
# 设定温度参数，用于控制语言模型生成文本的随机性
PROMPT_TEMPERATURE = 0.3

# 创建聊天提示模板对象，用于配置生成聊天内容的模板
prompt = ChatPromptTemplate(
    messages=[
        SystemPromptTemplate.from_template(_PROMPT_SCENE_DEFINE + _DEFAULT_TEMPLATE),  # 从系统定义和默认模板生成场景提示
        MessagesPlaceholder(variable_name="chat_history"),  # 插入聊天历史记录占位符
        HumanPromptTemplate.from_template("{user_input}"),  # 插入用户输入占位符
    ]
)

# 创建应用场景提示模板适配器，用于将聊天提示模板应用到具体的聊天场景中
prompt_adapter = AppScenePromptTemplateAdapter(
    prompt=prompt,  # 使用上面定义的聊天提示模板
    template_scene=ChatScene.ChatExcel.value(),  # 指定聊天场景为Excel聊天
    stream_out=PROMPT_NEED_STREAM_OUT,  # 设定是否需要流输出
    output_parser=ChatExcelOutputParser(is_stream_out=PROMPT_NEED_STREAM_OUT),  # 配置输出解析器，支持流输出
    need_historical_messages=True,  # 指定需要历史消息
    temperature=PROMPT_TEMPERATURE,  # 使用上面定义的温度参数控制生成文本的随机性
)

# 将提示适配器注册到配置的默认提示模板注册表中
CFG.prompt_template_registry.register(prompt_adapter, is_default=True)
```