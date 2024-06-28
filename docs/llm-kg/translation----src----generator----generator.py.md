# `.\translation\src\generator\generator.py`

```
# 从langchain.prompts模块导入PromptTemplate类
# 从langchain.chains模块导入LLMChain类
# 从langchain.llms模块导入LlamaCpp类
# 从base.config模块导入Config类
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp

from base.config import Config

# 定义Generator类，继承自Config类
class Generator(Config):
    """Generator, aka LLM, to provide an answer based on some question and context"""

    def __init__(self) -> None:
        super().__init__()
        # 初始化模板字符串，用于生成问题和上下文的填充格式
        self.template = """
            Use the following pieces of context to answer the question at the end. 
            {context}
            Question: {question}
            Answer:
        """
        # 使用LlamaCpp类从本地文件加载LLM模型
        self.llm = LlamaCpp(
            model_path=f"{self.parent_path}/{self.config['generator']['llm_path']}",  # 模型路径
            n_ctx=self.config["generator"]["context_length"],  # 上下文长度
            temperature=self.config["generator"]["temperature"],  # 温度参数
            encoding='latin-1',  # 编码格式
        )
        # 创建PromptTemplate对象，用于填充模板字符串
        self.prompt = PromptTemplate(
            template=self.template,  # 使用的模板字符串
            input_variables=["context", "question"]  # 模板中要填充的变量
        )

    def get_answer(self, context: str, question: str) -> str:
        """
        Get the answer from llm based on context and user's question
        Args:
            context (str): most similar document retrieved
            question (str): user's question
        Returns:
            str: llm answer
        """
        # 创建LLMChain对象，用于生成基于问题和上下文的答案
        query_llm = LLMChain(
            llm=self.llm,  # 使用的LLM模型
            prompt=self.prompt,  # 使用的填充模板
            llm_kwargs={"max_tokens": self.config["generator"]["max_tokens"]},  # LLMChain的其他参数
        )

        return query_llm.run({"context": context, "question": question})  # 运行LLMChain生成答案
```