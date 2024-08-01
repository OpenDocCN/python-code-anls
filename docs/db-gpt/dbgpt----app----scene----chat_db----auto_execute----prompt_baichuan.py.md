# `.\DB-GPT-src\dbgpt\app\scene\chat_db\auto_execute\prompt_baichuan.py`

```py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 导入 json 模块，用于处理 JSON 数据
import json

# 导入 Config 类，用于获取配置信息
from dbgpt._private.config import Config

# 导入 ChatScene 类，用于定义场景
from dbgpt.app.scene import ChatScene

# 导入 DbChatOutputParser 类，用于处理数据库聊天输出的解析
from dbgpt.app.scene.chat_db.auto_execute.out_parser import DbChatOutputParser

# 导入 PromptTemplate 类，用于创建聊天模板
from dbgpt.core.interface.prompt import PromptTemplate

# 创建 Config 类的实例
CFG = Config()

# 初始化场景定义为 None
PROMPT_SCENE_DEFINE = None

# 默认模板字符串，用于生成 SQL 查询语句的说明模板
_DEFAULT_TEMPLATE = """
你是一个 SQL 专家，给你一个用户的问题，你会生成一条对应的 {dialect} 语法的 SQL 语句。

如果用户没有在问题中指定 sql 返回多少条数据，那么你生成的 sql 最多返回 {top_k} 条数据。 
你应该尽可能少地使用表。

已知表结构信息如下：
{table_info}

注意：
1. 只能使用表结构信息中提供的表来生成 sql，如果无法根据提供的表结构中生成 sql 查询，请说：“提供的表结构信息不足以生成 sql 查询。” 禁止随意捏造信息。
2. 不要查询不存在的列，注意哪一列位于哪张表中。
3. 使用 json 格式回答，确保你的回答是必须是正确的 json 格式，并且能被 python 语言的 `json.loads` 库解析, 格式如下：
{response}
"""

# 简单的回复格式，定义了数据库聊天输出的格式
RESPONSE_FORMAT_SIMPLE = {
    "thoughts": "对用户说的想法摘要",
    "sql": "生成的将被执行的 SQL",
}

# 确定是否需要流式输出
PROMPT_NEED_STREAM_OUT = False

# 设置温度参数，控制语言模型输出的随机性
PROMPT_TEMPERATURE = 0.5

# 创建 PromptTemplate 实例，定义数据库聊天模板
prompt = PromptTemplate(
    template_scene=ChatScene.ChatWithDbExecute.value(),  # 指定场景为 ChatWithDbExecute
    input_variables=["input", "table_info", "dialect", "top_k", "response"],  # 输入变量列表
    response_format=json.dumps(RESPONSE_FORMAT_SIMPLE, ensure_ascii=False, indent=4),  # 回复格式化为 JSON
    template_is_strict=False,  # 模板是否严格
    template_define=PROMPT_SCENE_DEFINE,  # 模板定义
    template=_DEFAULT_TEMPLATE,  # 使用的模板字符串
    stream_out=PROMPT_NEED_STREAM_OUT,  # 是否流式输出
    output_parser=DbChatOutputParser(is_stream_out=PROMPT_NEED_STREAM_OUT),  # 输出解析器
    temperature=PROMPT_TEMPERATURE,  # 温度参数
)

# 将 prompt 注册到配置的 prompt 模板注册表中
CFG.prompt_template_registry.register(
    prompt,
    language=CFG.LANGUAGE,  # 使用的语言
    is_default=False,  # 是否默认模板
    model_names=["baichuan-13b", "baichuan-7b"],  # 支持的模型名称列表
)
```