# `.\Langchain-Chatchat\chains\llmchain_with_history.py`

```py
# 从 server.utils 模块中导入 get_ChatOpenAI 函数
# 从 configs.model_config 模块中导入 LLM_MODELS 和 TEMPERATURE 常量
# 从 langchain.chains 模块中导入 LLMChain 类
# 从 langchain.prompts.chat 模块中导入 ChatPromptTemplate 和 HumanMessagePromptTemplate 类
from server.utils import get_ChatOpenAI
from configs.model_config import LLM_MODELS, TEMPERATURE
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

# 使用 get_ChatOpenAI 函数获取 ChatOpenAI 模型
# 使用 LLM_MODELS 列表中的第一个模型和 TEMPERATURE 常量
model = get_ChatOpenAI(model_name=LLM_MODELS[0], temperature=TEMPERATURE)

# 设置人类输入的提示
human_prompt = "{input}"
# 从人类输入的提示创建 HumanMessagePromptTemplate 实例
human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

# 设置对话提示模板，包括人类和 AI 的消息
chat_prompt = ChatPromptTemplate.from_messages(
    [("human", "我们来玩成语接龙，我先来，生龙活虎"),
     ("ai", "虎头虎脑"),
     ("human", "{input}")])

# 创建 LLMChain 实例，使用 chat_prompt 作为提示，model 作为 LLM 模型，verbose 设置为 True
chain = LLMChain(prompt=chat_prompt, llm=model, verbose=True)
# 打印根据输入 "恼羞成怒" 生成的对话结果
print(chain({"input": "恼羞成怒"}))
```