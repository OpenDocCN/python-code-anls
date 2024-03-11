# `.\Langchain-Chatchat\server\agent\tools\search_knowledgebase_complex.py`

```
# 导入必要的模块和库
from __future__ import annotations
import json
import re
import warnings
from typing import Dict
from langchain.callbacks.manager import AsyncCallbackManagerForChainRun, CallbackManagerForChainRun
from langchain.chains.llm import LLMChain
from langchain.pydantic_v1 import Extra, root_validator
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from typing import List, Any, Optional
from langchain.prompts import PromptTemplate
from server.chat.knowledge_base_chat import knowledge_base_chat
from configs import VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, MAX_TOKENS
import asyncio
from server.agent import model_container
from pydantic import BaseModel, Field

# 异步函数，用于搜索知识库并返回结果
async def search_knowledge_base_iter(database: str, query: str) -> str:
    # 调用知识库聊天函数，传入查询参数和相关设置
    response = await knowledge_base_chat(query=query,
                                         knowledge_base_name=database,
                                         model_name=model_container.MODEL.model_name,
                                         temperature=0.01,
                                         history=[],
                                         top_k=VECTOR_SEARCH_TOP_K,
                                         max_tokens=MAX_TOKENS,
                                         prompt_name="default",
                                         score_threshold=SCORE_THRESHOLD,
                                         stream=False)

    # 初始化内容字符串
    contents = ""
    # 异步迭代响应体中的数据
    async for data in response.body_iterator:  # 这里的data是一个json字符串
        # 将 JSON 字符串解析为字典
        data = json.loads(data)
        # 将答案添加到内容字符串
        contents += data["answer"]
        # 获取文档信息
        docs = data["docs"]
    # 返回内容字符串
    return contents

# 异步函数，用于搜索多个知识库并返回结果列表
async def search_knowledge_multiple(queries) -> List[str]:
    # queries 应该是一个包含多个 (database, query) 元组的列表
    # 创建任务列表，每个任务对应一个查询
    tasks = [search_knowledge_base_iter(database, query) for database, query in queries]
    # 并发执行所有任务，获取结果列表
    results = await asyncio.gather(*tasks)
    # 初始化合并结果列表
    combined_results = []
    # 遍历 queries 和 results 的元素，分别为 (database, _) 和 result
    for (database, _), result in zip(queries, results):
        # 根据数据库名称和查询结果生成消息字符串
        message = f"\n查询到 {database} 知识库的相关信息:\n{result}"
        # 将消息字符串添加到 combined_results 列表中
        combined_results.append(message)

    # 返回包含所有消息字符串的列表
    return combined_results
def search_knowledge(queries) -> str:
    # 运行异步函数search_knowledge_multiple来搜索知识库
    responses = asyncio.run(search_knowledge_multiple(queries))
    # 初始化内容字符串
    contents = ""
    # 遍历每个查询结果，将其添加到内容字符串中
    for response in responses:
        contents += response + "\n\n"
    # 返回整合后的查询结果
    return contents


_PROMPT_TEMPLATE = """
用户会提出一个需要你查询知识库的问题，你应该对问题进行理解和拆解，并在知识库中查询相关的内容。

对于每个知识库，你输出的内容应该是一个一行的字符串，这行字符串包含知识库名称和查询内容，中间用逗号隔开，不要有多余的文字和符号。你可以同时查询多个知识库，下面这个例子就是同时查询两个知识库的内容。

例子:

robotic,机器人男女比例是多少
bigdata,大数据的就业情况如何 


这些数据库是你能访问的，冒号之前是他们的名字，冒号之后是他们的功能，你应该参考他们的功能来帮助你思考


{database_names}

你的回答格式应该按照下面的内容，请注意```text 等标记都必须输出，这是我用来提取答案的标记。
不要输出中文的逗号，不要输出引号。

Question: ${{用户的问题}}


${{知识库名称,查询问题,不要带有任何除了,之外的符号,比如不要输出中文的逗号，不要输出引号}}


数据库查询的结果

现在，我们开始作答
问题: {question}
"""

PROMPT = PromptTemplate(
    input_variables=["question", "database_names"],
    template=_PROMPT_TEMPLATE,
)


class LLMKnowledgeChain(LLMChain):
    llm_chain: LLMChain
    llm: Optional[BaseLanguageModel] = None
    """[Deprecated] LLM wrapper to use."""
    prompt: BasePromptTemplate = PROMPT
    """[Deprecated] Prompt to use to translate to python if necessary."""
    database_names: Dict[str, str] = None
    input_key: str = "question"  #: :meta private:
    output_key: str = "answer"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def raise_deprecation(cls, values: Dict) -> Dict:
        # 如果values中包含"llm"，则发出警告
        if "llm" in values:
            warnings.warn(
                "Directly instantiating an LLMKnowledgeChain with an llm is deprecated. "
                "Please instantiate with llm_chain argument or using the from_llm "
                "class method."
            )
            # 如果values中没有"llm_chain"且"llm"不为None，则使用llm创建LLMChain对象
            if "llm_chain" not in values and values["llm"] is not None:
                prompt = values.get("prompt", PROMPT)
                values["llm_chain"] = LLMChain(llm=values["llm"], prompt=prompt)
        return values

    @property
    # 返回输入键列表
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    # 返回输出键列表
    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]

    # 评估表达式并返回结果
    def _evaluate_expression(self, queries) -> str:
        try:
            # 搜索知识库中的信息
            output = search_knowledge(queries)
        except Exception as e:
            # 处理异常情况
            output = "输入的信息有误或不存在知识库,错误信息如下:\n"
            return output + str(e)
        return output

    # 处理LLM结果
    def _process_llm_result(
            self,
            llm_output: str,
            run_manager: CallbackManagerForChainRun
    ) -> Dict[str, str]:

        # 在运行管理器上显示LLM输出文本，颜色为绿色，根据verbose参数决定是否显示详细信息
        run_manager.on_text(llm_output, color="green", verbose=self.verbose)

        # 去除LLM输出文本两端的空白字符
        llm_output = llm_output.strip()
        # 使用正则表达式搜索匹配包含"text"的文本块
        text_match = re.search(r"```text(.*)", llm_output, re.DOTALL)
        if text_match:
            # 获取匹配到的文本块内容，并去除首尾空白字符
            expression = text_match.group(1).strip()
            # 清理文本中的特殊字符，如引号和反引号
            cleaned_input_str = (expression.replace("\"", "").replace("“", "").
                                 replace("”", "").replace("```", "").strip())
            # 将清理后的文本按行分割
            lines = cleaned_input_str.split("\n")
            # 尝试使用逗号分割每行文本，形成（数据库，查询）元组的列表
            try:
                queries = [(line.split(",")[0].strip(), line.split(",")[1].strip()) for line in lines]
            except:
                # 如果逗号分割失败，则使用中文逗号分割
                queries = [(line.split("，")[0].strip(), line.split("，")[1].strip()) for line in lines]
            # 在运行管理器上显示知识库查询内容，根据verbose参数决定是否显示详细信息
            run_manager.on_text("知识库查询询内容:\n\n" + str(queries) + " \n\n", color="blue", verbose=self.verbose)
            # 对查询进行评估
            output = self._evaluate_expression(queries)
            run_manager.on_text("\nAnswer: ", verbose=self.verbose)
            run_manager.on_text(output, color="yellow", verbose=self.verbose)
            answer = "Answer: " + output
        elif llm_output.startswith("Answer:"):
            answer = llm_output
        elif "Answer:" in llm_output:
            # 从LLM输出中提取答案
            answer = llm_output.split("Answer:")[-1]
        else:
            # 返回格式不正确的提示信息
            return {self.output_key: f"输入的格式不对:\n {llm_output}"}
        return {self.output_key: answer}

    async def _aprocess_llm_result(
            self,
            llm_output: str,
            run_manager: AsyncCallbackManagerForChainRun,
    ) -> Dict[str, str]:
        # 在运行管理器上显示文本输出，颜色为绿色，根据 verbose 参数决定是否显示详细信息
        await run_manager.on_text(llm_output, color="green", verbose=self.verbose)
        # 去除字符串两端的空格
        llm_output = llm_output.strip()
        # 使用正则表达式匹配包含"```text"的字符串
        text_match = re.search(r"```text(.*)", llm_output, re.DOTALL)
        if text_match:
            # 获取匹配到的文本内容
            expression = text_match.group(1).strip()
            # 清理输入字符串，去除特殊字符和空格
            cleaned_input_str = (
                expression.replace("\"", "").replace("“", "").replace("”", "").replace("```", "").strip())
            # 将清理后的字符串按行分割
            lines = cleaned_input_str.split("\n")
            try:
                # 尝试将每行按逗号分割成查询对
                queries = [(line.split(",")[0].strip(), line.split(",")[1].strip()) for line in lines]
            except:
                # 如果逗号无法正确分割，则使用中文逗号分割
                queries = [(line.split("，")[0].strip(), line.split("，")[1].strip()) for line in lines]
            # 在运行管理器上显示查询内容，颜色为蓝色，根据 verbose 参数决定是否显示详细信息
            await run_manager.on_text("知识库查询询内容:\n\n" + str(queries) + " \n\n", color="blue",
                                      verbose=self.verbose)
            # 对查询进行评估
            output = self._evaluate_expression(queries)
            # 在运行管理器上显示答案标签
            await run_manager.on_text("\nAnswer: ", verbose=self.verbose)
            # 在运行管理器上显示输出内容，颜色为黄色，根据 verbose 参数决定是否显示详细信息
            await run_manager.on_text(output, color="yellow", verbose=self.verbose)
            answer = "Answer: " + output
        elif llm_output.startswith("Answer:"):
            answer = llm_output
        elif "Answer:" in llm_output:
            # 从 llm_output 中提取答案内容
            answer = "Answer: " + llm_output.split("Answer:")[-1]
        else:
            # 如果无法识别 LLM 的输出格式，则引发 ValueError 异常
            raise ValueError(f"unknown format from LLM: {llm_output}")
        # 返回包含答案的字典
        return {self.output_key: answer}

    def _call(
            self,
            inputs: Dict[str, str],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    # 定义一个方法，用于运行链式运行管理器或者获取一个空的运行管理器
    def run_chain(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # 如果没有传入运行管理器，则使用空的运行管理器
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        # 在运行管理器上记录输入文本
        _run_manager.on_text(inputs[self.input_key])
        # 设置数据库名称为模型容器中的数据库
        self.database_names = model_container.DATABASE
        # 格式化数据库名称和值，用逗号分隔
        data_formatted_str = ',\n'.join([f' "{k}":"{v}"' for k, v in self.database_names.items()])
        # 使用llm_chain预测结果
        llm_output = self.llm_chain.predict(
            database_names=data_formatted_str,
            question=inputs[self.input_key],
            stop=["```output"],
            callbacks=_run_manager.get_child(),
        )
        # 处理llm预测结果并返回
        return self._process_llm_result(llm_output, _run_manager)

    # 定义一个异步方法，用于运行链式运行管理器或者获取一个空的异步运行管理器
    async def _acall(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # 如果没有传入运行管理器，则使用空的异步运行管理器
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        # 在异步运行管理器上记录输入文本
        await _run_manager.on_text(inputs[self.input_key])
        # 设置数据库名称为模型容器中的数据库
        self.database_names = model_container.DATABASE
        # 格式化数据库名称和值，用逗号分隔
        data_formatted_str = ',\n'.join([f' "{k}":"{v}"' for k, v in self.database_names.items()])
        # 使用llm_chain异步预测结果
        llm_output = await self.llm_chain.apredict(
            database_names=data_formatted_str,
            question=inputs[self.input_key],
            stop=["```output"],
            callbacks=_run_manager.get_child(),
        )
        # 处理llm异步预测结果并返回
        return await self._aprocess_llm_result(llm_output, inputs[self.input_key], _run_manager)

    # 定义一个属性，返回链式运行的类型
    @property
    def _chain_type(self) -> str:
        return "llm_knowledge_chain"

    # 从llm创建一个LLMKnowledgeChain实例
    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate = PROMPT,
        **kwargs: Any,
    ) -> LLMKnowledgeChain:
        # 创建一个LLMChain实例
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        # 返回一个LLMKnowledgeChain实例
        return cls(llm_chain=llm_chain, **kwargs)
# 定义一个函数，用于在知识库中搜索复杂查询
def search_knowledgebase_complex(query: str):
    # 从模型容器中获取模型
    model = model_container.MODEL
    # 从LLMKnowledgeChain类中创建LLMKnowledgeChain对象，设置为详细模式，使用给定的提示
    llm_knowledge = LLMKnowledgeChain.from_llm(model, verbose=True, prompt=PROMPT)
    # 运行查询并返回结果
    ans = llm_knowledge.run(query)
    return ans

# 定义一个数据模型类，用于表示知识搜索的输入
class KnowledgeSearchInput(BaseModel):
    location: str = Field(description="The query to be searched")

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 对给定查询运行搜索函数，并打印结果
    result = search_knowledgebase_complex("机器人和大数据在代码教学上有什么区别")
    print(result)

# 这是一个正常的切割
#     queries = [
#         ("bigdata", "大数据专业的男女比例"),
#         ("robotic", "机器人专业的优势")
#     ]
#     result = search_knowledge(queries)
#     print(result)
```