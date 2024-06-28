# `.\rag\simple_demo\generator\generator.py`

```
# 导入必要的模块和类
from base.config import Config  # 从基础配置中导入 Config 类
from langchain.prompts import PromptTemplate  # 导入提示模板类
from langchain.chains import LLMChain  # 导入 LLNChain 类
from langchain.llms import LlamaCpp  # 导入 LlamaCpp 类


class Generator(Config):
    """生成器，即LLM，根据问题和上下文提供答案"""

    def __init__(self, template) -> None:
        super().__init__()  # 调用父类 Config 的初始化方法
        # 从本地文件加载 Llama 模型
        self.llm = LlamaCpp(
            model_path=f"{self.parent_path}/{self.config['generator']['llm_path']}",  # 设置模型路径
            n_ctx=self.config["generator"]["context_length"],  # 设置上下文长度
            temperature=self.config["generator"]["temperature"],  # 设置温度参数
            encoding='latin-1',  # 设置编码方式
        )
        # 创建提示模板
        self.prompt = PromptTemplate(
            template=template,  # 使用给定的模板
            input_variables=["context", "question"]  # 设置输入变量
        )

    def get_answer(self, context: str, question: str) -> str:
        """
        根据上下文和用户的问题从 llm 中获取答案
        Args:
            context (str): 最相似的文档
            question (str): 用户的问题
        Returns:
            str: llm 的答案
        """

        query_llm = LLMChain(
            llm=self.llm,  # 使用设定好的 LlamaCpp 对象
            prompt=self.prompt,  # 使用设定好的 PromptTemplate 对象
            llm_kwargs={"max_tokens": self.config["generator"]["max_tokens"]},  # 设置 llm 参数
        )

        return query_llm.run({"context": context, "question": question})  # 运行查询并返回结果
```