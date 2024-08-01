# `.\DB-GPT-src\dbgpt\app\scene\chat_db\auto_execute\prompt.py`

```py
# 导入json模块，用于处理JSON数据
import json

# 从dbgpt._private.config模块中导入Config类
from dbgpt._private.config import Config

# 从dbgpt.app.scene模块中导入AppScenePromptTemplateAdapter和ChatScene类
from dbgpt.app.scene import AppScenePromptTemplateAdapter, ChatScene

# 从dbgpt.app.scene.chat_db.auto_execute.out_parser模块中导入DbChatOutputParser类
from dbgpt.app.scene.chat_db.auto_execute.out_parser import DbChatOutputParser

# 从dbgpt.core模块中导入ChatPromptTemplate, HumanPromptTemplate, MessagesPlaceholder, SystemPromptTemplate类
from dbgpt.core import (
    ChatPromptTemplate,
    HumanPromptTemplate,
    MessagesPlaceholder,
    SystemPromptTemplate,
)

# 创建Config类的实例对象，用于获取配置信息
CFG = Config()

# 如果CFG.LANGUAGE等于"en"，则定义_PROMPT_SCENE_DEFINE_EN为英文场景定义，否则定义为中文场景定义
_PROMPT_SCENE_DEFINE_EN = "You are a database expert. "
_PROMPT_SCENE_DEFINE_ZH = "你是一个数据库专家. "

# 如果CFG.LANGUAGE等于"en"，则使用默认的英文模板_DEFAULT_TEMPLATE_EN，否则使用中文模板_DEFAULT_TEMPLATE_ZH
_DEFAULT_TEMPLATE_EN = """
Please answer the user's question based on the database selected by the user and some of the available table structure definitions of the database.
Database name:
     {db_name}
Table structure definition:
     {table_info}

Constraint:
    1.Please understand the user's intention based on the user's question, and use the given table structure definition to create a grammatically correct {dialect} sql. If sql is not required, answer the user's question directly.. 
    2.Always limit the query to a maximum of {top_k} results unless the user specifies in the question the specific number of rows of data he wishes to obtain.
    3.You can only use the tables provided in the table structure information to generate sql. If you cannot generate sql based on the provided table structure, please say: "The table structure information provided is not enough to generate sql queries." It is prohibited to fabricate information at will.
    4.Please be careful not to mistake the relationship between tables and columns when generating SQL.
    5.Please check the correctness of the SQL and ensure that the query performance is optimized under correct conditions.
    6.Please choose the best one from the display methods given below for data rendering, and put the type name into the name parameter value that returns the required format. If you cannot find the most suitable one, use 'Table' as the display method. , the available data display methods are as follows: {display_type}
    
User Question:
    {user_input}
Please think step by step and respond according to the following JSON format:
    {response}
Ensure the response is correct json and can be parsed by Python json.loads.

"""

_DEFAULT_TEMPLATE_ZH = """
请根据用户选择的数据库和该库的部分可用表结构定义来回答用户问题.
数据库名:
    {db_name}
表结构定义:
    {table_info}

约束:
    1. 请根据用户问题理解用户意图，使用给出表结构定义创建一个语法正确的 {dialect} sql，如果不需要sql，则直接回答用户问题。
    2. 除非用户在问题中指定了他希望获得的具体数据行数，否则始终将查询限制为最多 {top_k} 个结果。
    3. 只能使用表结构信息中提供的表来生成 sql，如果无法根据提供的表结构中生成 sql ，请说：“提供的表结构信息不足以生成 sql 查询。” 禁止随意捏造信息。
    4. 请注意生成SQL时不要弄错表和列的关系
    5. 请检查SQL的正确性，并保证正确的情况下优化查询性能
    6.请从如下给出的展示方式种选择最优的一种用以进行数据渲染，将类型名称放入返回要求格式的name参数值种，如果找不到最合适的则使用'Table'作为展示方式，可用数据展示方式如下: {display_type}
用户问题:
    {user_input}
请一步步思考并按照以下JSON格式回复：
      {response}
确保返回正确的json并且可以被Python json.loads方法解析.

"""

# 根据CFG.LANGUAGE的值选择合适的模板语言，存储到_DEFAULT_TEMPLATE中
_DEFAULT_TEMPLATE = (
    _DEFAULT_TEMPLATE_EN if CFG.LANGUAGE == "en" else _DEFAULT_TEMPLATE_ZH
)

# 根据CFG.LANGUAGE的值选择合适的场景定义语言，存储到PROMPT_SCENE_DEFINE中
PROMPT_SCENE_DEFINE = (
    _PROMPT_SCENE_DEFINE_EN if CFG.LANGUAGE == "en" else _PROMPT_SCENE_DEFINE_ZH
)
RESPONSE_FORMAT_SIMPLE = {
    "thoughts": "thoughts summary to say to user",
    "sql": "SQL Query to run",
    "display_type": "Data display method",
}
# 简单响应格式定义，包含了向用户展示的想法摘要、要运行的 SQL 查询以及数据展示方式


PROMPT_NEED_STREAM_OUT = False
# 指示是否需要输出流到响应，此处为 False，表示不需要流输出


# Temperature is a configuration hyperparameter that controls the randomness of language model output.
# A high temperature produces more unpredictable and creative results, while a low temperature produces more common and conservative output.
# For example, if you adjust the temperature to 0.5, the model will usually generate text that is more predictable and less creative than if you set the temperature to 1.0.
PROMPT_TEMPERATURE = 0.5
# 温度是一个配置超参数，控制语言模型输出的随机性。较高的温度会产生更不可预测和创造性的结果，而较低的温度会产生更常见和保守的输出。例如，将温度设置为 0.5 时，模型通常会生成比将温度设置为 1.0 时更可预测且不太创造性的文本。


# 创建对话提示的模板
prompt = ChatPromptTemplate(
    messages=[
        SystemPromptTemplate.from_template(
            _DEFAULT_TEMPLATE,
            response_format=json.dumps(
                RESPONSE_FORMAT_SIMPLE, ensure_ascii=False, indent=4
            ),
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanPromptTemplate.from_template("{user_input}"),
    ]
)

# 创建应用场景对话提示的适配器
prompt_adapter = AppScenePromptTemplateAdapter(
    prompt=prompt,
    template_scene=ChatScene.ChatWithDbExecute.value(),
    stream_out=PROMPT_NEED_STREAM_OUT,
    output_parser=DbChatOutputParser(is_stream_out=PROMPT_NEED_STREAM_OUT),
    temperature=PROMPT_TEMPERATURE,
    need_historical_messages=False,
)
# 使用预定义的对话模板和场景，配置对话适配器，并设置相关参数


CFG.prompt_template_registry.register(prompt_adapter, is_default=True)
# 将配置好的对话适配器注册到对话模板注册表中，并设置为默认适配器
```