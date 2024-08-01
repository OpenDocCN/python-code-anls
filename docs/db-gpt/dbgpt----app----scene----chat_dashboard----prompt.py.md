# `.\DB-GPT-src\dbgpt\app\scene\chat_dashboard\prompt.py`

```py
# 导入json模块，用于处理JSON格式数据
import json

# 导入Config类，该类来自dbgpt._private.config模块
from dbgpt._private.config import Config

# 导入AppScenePromptTemplateAdapter和ChatScene类，用于处理场景模板和聊天场景
from dbgpt.app.scene import AppScenePromptTemplateAdapter, ChatScene

# 导入ChatDashboardOutputParser类，用于解析聊天仪表盘的输出
from dbgpt.app.scene.chat_dashboard.out_parser import ChatDashboardOutputParser

# 导入ChatPromptTemplate、HumanPromptTemplate和SystemPromptTemplate类，用于定义不同类型的提示模板
from dbgpt.core import ChatPromptTemplate, HumanPromptTemplate, SystemPromptTemplate

# 创建一个Config对象实例
CFG = Config()

# 定义用于场景提示的常量字符串
PROMPT_SCENE_DEFINE = "You are a data analysis expert, please provide a professional data analysis solution"

# 定义默认的模板字符串，用于生成聊天提示
_DEFAULT_TEMPLATE = """
According to the following table structure definition:
{table_info}
Provide professional data analysis to support users' goals:
{input}

Provide at least 4 and at most 8 dimensions of analysis according to user goals.
The output data of the analysis cannot exceed 4 columns, and do not use columns such as pay_status in the SQL where condition for data filtering.
According to the characteristics of the analyzed data, choose the most suitable one from the charts provided below for data display, chart type:
{supported_chat_type}

Pay attention to the length of the output content of the analysis result, do not exceed 4000 tokens

Give the correct {dialect} analysis SQL
1.Do not use unprovided values such as 'paid'
2.All queried values must have aliases, such as select count(*) as count from table
3.If the table structure definition uses the keywords of {dialect} as field names, you need to use escape characters, such as select `count` from table
4.Carefully check the correctness of the SQL, the SQL must be correct, display method and summary of brief analysis thinking, and respond in the following json format:
{response}
The important thing is: Please make sure to only return the json string, do not add any other content (for direct processing by the program), and the json can be parsed by Python json.loads
5. Please use the same language as the "user"
"""

# 定义响应格式的列表，包含数据分析的相关信息
RESPONSE_FORMAT = [
    {
        "thoughts": "Current thinking and value of data analysis",
        "showcase": "What type of charts to show",
        "sql": "data analysis SQL",
        "title": "Data Analysis Title",
    }
]

# 定义是否需要流输出的标志，默认为False
PROMPT_NEED_STREAM_OUT = False

# 创建ChatPromptTemplate对象，用于生成聊天提示模板
prompt = ChatPromptTemplate(
    messages=[
        SystemPromptTemplate.from_template(
            PROMPT_SCENE_DEFINE + _DEFAULT_TEMPLATE,  # 拼接场景定义和默认模板字符串
            response_format=json.dumps(RESPONSE_FORMAT, indent=4),  # 将响应格式转换为JSON字符串
        ),
        HumanPromptTemplate.from_template("{input}"),  # 从输入模板生成人类提示模板
    ]
)

# 创建AppScenePromptTemplateAdapter适配器对象，用于适配场景提示模板
prompt_adapter = AppScenePromptTemplateAdapter(
    prompt=prompt,  # 设置聊天提示模板
    template_scene=ChatScene.ChatDashboard.value(),  # 设置模板场景为聊天仪表盘
    stream_out=PROMPT_NEED_STREAM_OUT,  # 设置是否需要流输出
    output_parser=ChatDashboardOutputParser(is_stream_out=PROMPT_NEED_STREAM_OUT),  # 设置输出解析器
    need_historical_messages=False,  # 设置是否需要历史消息
)

# 将创建的提示适配器注册到配置的提示模板注册表中，并设置为默认
CFG.prompt_template_registry.register(prompt_adapter, is_default=True)
```