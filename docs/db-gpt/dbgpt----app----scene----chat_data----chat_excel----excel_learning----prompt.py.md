# `.\DB-GPT-src\dbgpt\app\scene\chat_data\chat_excel\excel_learning\prompt.py`

```py
# 导入json模块，用于处理JSON数据格式
import json

# 从dbgpt._private.config模块导入Config类，用于配置管理
from dbgpt._private.config import Config

# 从dbgpt.app.scene模块导入AppScenePromptTemplateAdapter和ChatScene类，用于场景处理
from dbgpt.app.scene import AppScenePromptTemplateAdapter, ChatScene

# 从dbgpt.app.scene.chat_data.chat_excel.excel_learning.out_parser模块导入LearningExcelOutputParser类，用于Excel输出解析
from dbgpt.app.scene.chat_data.chat_excel.excel_learning.out_parser import (
    LearningExcelOutputParser,
)

# 从dbgpt.core模块导入各种模板和占位符类
from dbgpt.core import (
    ChatPromptTemplate,
    HumanPromptTemplate,
    MessagesPlaceholder,
    SystemPromptTemplate,
)

# 从dbgpt.core.interface.prompt模块导入PromptTemplate接口类，用于定义提示模板
from dbgpt.core.interface.prompt import PromptTemplate

# 创建Config类的实例对象，用于读取配置信息
CFG = Config()

# 如果CFG.LANGUAGE为英文，则使用英文场景定义
_PROMPT_SCENE_DEFINE_EN = "You are a data analysis expert. "

# 如果CFG.LANGUAGE为中文，则使用中文场景定义
_PROMPT_SCENE_DEFINE_ZH = "你是一个数据分析专家. "

# 英文默认模板字符串，用于提示用户对数据进行分析
_DEFAULT_TEMPLATE_EN = """
The following is part of the data of the user file {file_name}. Please learn to understand the structure and content of the data and output the parsing results as required:
    {data_example}
Explain the meaning and function of each column, and give a simple and clear explanation of the technical terms， If it is a Date column, please summarize the Date format like: yyyy-MM-dd HH:MM:ss.
Use the column name as the attribute name and the analysis explanation as the attribute value to form a json array and output it in the ColumnAnalysis attribute that returns the json content.
Please do not modify or translate the column names, make sure they are consistent with the given data column names.
Provide some useful analysis ideas to users from different dimensions for data.

Please think step by step and give your answer. Make sure to answer only in JSON format，the format is as follows:
    {response}
"""

# 中文默认模板字符串，用于提示用户对数据进行分析
_DEFAULT_TEMPLATE_ZH = """
下面是用户文件{file_name}的一部分数据，请学习理解该数据的结构和内容，按要求输出解析结果:
    {data_example}
分析各列数据的含义和作用，并对专业术语进行简单明了的解释, 如果是时间类型请给出时间格式类似:yyyy-MM-dd HH:MM:ss.
将列名作为属性名，分析解释作为属性值,组成json数组，并输出在返回json内容的ColumnAnalysis属性中.
请不要修改或者翻译列名，确保和给出数据列名一致.
针对数据从不同维度提供一些有用的分析思路给用户。

请一步一步思考,确保只以JSON格式回答，具体格式如下：
    {response}
"""

# 根据配置语言选择对应的默认模板字符串
_DEFAULT_TEMPLATE = (
    _DEFAULT_TEMPLATE_EN if CFG.LANGUAGE == "en" else _DEFAULT_TEMPLATE_ZH
)

# 根据配置语言选择对应的场景定义字符串
PROMPT_SCENE_DEFINE = (
    _PROMPT_SCENE_DEFINE_EN if CFG.LANGUAGE == "en" else _PROMPT_SCENE_DEFINE_ZH
)

# 如果CFG.LANGUAGE为英文，使用英文的响应格式
_RESPONSE_FORMAT_SIMPLE_EN = {
    "DataAnalysis": "Data content analysis summary",
    "ColumnAnalysis": [
        {
            "column name": "Introduction to Column 1 and explanation of professional terms (please try to be as simple and clear as possible)"
        }
    ],
    "AnalysisProgram": ["1. Analysis plan ", "2. Analysis plan "],
}

# 如果CFG.LANGUAGE为中文，使用中文的响应格式
_RESPONSE_FORMAT_SIMPLE_ZH = {
    "DataAnalysis": "数据内容分析总结",
    "ColumnAnalysis": [{"column name": "字段1介绍，专业术语解释(请尽量简单明了)"}],
    "AnalysisProgram": ["1.分析方案1", "2.分析方案2"],
}

# 根据配置语言选择对应的响应格式
RESPONSE_FORMAT_SIMPLE = (
    _RESPONSE_FORMAT_SIMPLE_EN if CFG.LANGUAGE == "en" else _RESPONSE_FORMAT_SIMPLE_ZH
)

# 是否需要流式输出的标志，这里设置为False
PROMPT_NEED_STREAM_OUT = False

# 温度（Temperature）是控制语言模型输出随机性的配置超参数。
# 高温度会产生更不可预测和创造性的结果，而低温度则产生更常见和保守的输出。
# 设定生成文本的温度，影响文本生成的创造性和可预测性，0.8 是一个中等的温度设置
PROMPT_TEMPERATURE = 0.8

# 定义对话的模板，包括系统模板和人类输入模板
prompt = ChatPromptTemplate(
    messages=[
        # 从系统模板生成对话场景定义，结合默认模板
        SystemPromptTemplate.from_template(
            PROMPT_SCENE_DEFINE + _DEFAULT_TEMPLATE,
            # 将响应格式设定为简单的 JSON 格式，并进行格式化以便易读
            response_format=json.dumps(
                RESPONSE_FORMAT_SIMPLE, ensure_ascii=False, indent=4
            ),
        ),
        # 从人类输入模板生成对话模板，使用变量 "{file_name}"
        HumanPromptTemplate.from_template("{file_name}"),
    ]
)

# 创建应用场景的对话模板适配器，用于处理 Excel 学习场景
prompt_adapter = AppScenePromptTemplateAdapter(
    prompt=prompt,
    # 指定对话模板的场景为 Excel 学习
    template_scene=ChatScene.ExcelLearning.value(),
    # 指定是否需要输出流
    stream_out=PROMPT_NEED_STREAM_OUT,
    # 输出解析器为 LearningExcelOutputParser 类，用于处理输出流
    output_parser=LearningExcelOutputParser(is_stream_out=PROMPT_NEED_STREAM_OUT),
    # 指定是否需要历史消息记录
    need_historical_messages=False,
    # 设定生成文本的温度，使用之前设定的 PROMPT_TEMPERATURE
    temperature=PROMPT_TEMPERATURE,
)

# 将上述创建的对话模板适配器注册到对话模板注册表中，并标记为默认模板
CFG.prompt_template_registry.register(prompt_adapter, is_default=True)
```