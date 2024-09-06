# `.\HippoRAG\src\openie_extraction_instructions.py`

```py
# 从 langchain_core.messages 模块导入 AIMessage、SystemMessage 和 HumanMessage
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
# 从 langchain_core.prompts 模块导入 HumanMessagePromptTemplate 和 ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate

## General Prompts

# 定义一个包含 Radio City 相关信息的文本段落
one_shot_passage = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""

# 定义一个包含 Radio City 相关命名实体的 JSON 字符串
one_shot_passage_entities = """{"named_entities":
    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}
"""

# 定义一个包含 Radio City 相关三元组的 JSON 字符串
one_shot_passage_triples = """{"triples": [
            ["Radio City", "located in", "India"],
            ["Radio City", "is", "private FM radio station"],
            ["Radio City", "started on", "3 July 2001"],
            ["Radio City", "plays songs in", "Hindi"],
            ["Radio City", "plays songs in", "English"]
            ["Radio City", "forayed into", "New Media"],
            ["Radio City", "launched", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "launched in", "May 2008"],
            ["PlanetRadiocity.com", "is", "music portal"],
            ["PlanetRadiocity.com", "offers", "news"],
            ["PlanetRadiocity.com", "offers", "videos"],
            ["PlanetRadiocity.com", "offers", "songs"]
    ]
}
"""

## NER Prompts

# 定义用于命名实体识别的指令
ner_instruction = """Your task is to extract named entities from the given paragraph. 
Respond with a JSON list of entities.
"""

# 定义包含待识别段落的格式化输入
ner_input_one_shot = """Paragraph:

{}

""".format(one_shot_passage)

# 定义命名实体识别的期望输出
ner_output_one_shot = one_shot_passage_entities

# 定义用户输入的格式化模板
ner_user_input = "Paragraph:```\n{user_input}\n```py"
# 使用定义的消息创建聊天提示模板
ner_prompts = ChatPromptTemplate.from_messages([SystemMessage(ner_instruction),
                                                HumanMessage(ner_input_one_shot),
                                                AIMessage(ner_output_one_shot),
                                                HumanMessagePromptTemplate.from_template(ner_user_input)])

## Post NER OpenIE Prompts

# 定义用于从段落和命名实体列表构建 RDF 图的指令
openie_post_ner_instruction = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists. 
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. 

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.

"""

# 定义用于构建 RDF 图的输入模板
openie_post_ner_frame = """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:

{passage}


{named_entity_json}
"""

# 通过替换模板中的占位符生成一段示例输入
openie_post_ner_input_one_shot = openie_post_ner_frame.replace("{passage}", one_shot_passage).replace("{named_entity_json}", one_shot_passage_entities)
# 将 `one_shot_passage_triples` 赋值给 `openie_post_ner_output_one_shot`
openie_post_ner_output_one_shot = one_shot_passage_triples

# 创建一个 `ChatPromptTemplate` 对象，并用一系列消息来初始化
# 第一个消息是系统消息，内容是 `openie_post_ner_instruction`
# 第二个消息是人类消息，内容是 `openie_post_ner_input_one_shot`
# 第三个消息是 AI 消息，内容是 `openie_post_ner_output_one_shot`
# 第四个消息是人类消息模板，模板内容是 `openie_post_ner_frame`
openie_post_ner_prompts = ChatPromptTemplate.from_messages([SystemMessage(openie_post_ner_instruction),
                                                            HumanMessage(openie_post_ner_input_one_shot),
                                                            AIMessage(openie_post_ner_output_one_shot),
                                                            HumanMessagePromptTemplate.from_template(openie_post_ner_frame)])
```