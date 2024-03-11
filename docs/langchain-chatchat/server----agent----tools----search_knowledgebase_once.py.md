# `.\Langchain-Chatchat\server\agent\tools\search_knowledgebase_once.py`

```py
# 导入必要的模块
from __future__ import annotations
import re
import warnings
from typing import Dict

# 导入回调管理器相关模块
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
# 导入LLMChain类
from langchain.chains.llm import LLMChain
# 导入pydantic_v1模块中的Extra和root_validator
from langchain.pydantic_v1 import Extra, root_validator
# 导入BasePromptTemplate类
from langchain.schema import BasePromptTemplate
# 导入BaseLanguageModel类
from langchain.schema.language_model import BaseLanguageModel
# 导入List, Any, Optional类型
from typing import List, Any, Optional
# 导入PromptTemplate类
from langchain.prompts import PromptTemplate
# 导入sys和os模块
import sys
import os
# 导入json模块
import json

# 将父目录的路径添加到sys.path中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 导入knowledge_base_chat函数
from server.chat.knowledge_base_chat import knowledge_base_chat
# 导入VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, MAX_TOKENS常量
from configs import VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, MAX_TOKENS

# 导入asyncio模块
import asyncio
# 导入model_container模块
from server.agent import model_container
# 导入BaseModel, Field类
from pydantic import BaseModel, Field

# 异步函数，用于搜索知识库
async def search_knowledge_base_iter(database: str, query: str):
    # 调用knowledge_base_chat函数进行查询
    response = await knowledge_base_chat(query=query,
                                         knowledge_base_name=database,
                                         model_name=model_container.MODEL.model_name,
                                         temperature=0.01,
                                         history=[],
                                         top_k=VECTOR_SEARCH_TOP_K,
                                         max_tokens=MAX_TOKENS,
                                         prompt_name="knowledge_base_chat",
                                         score_threshold=SCORE_THRESHOLD,
                                         stream=False)

    contents = ""
    # 异步迭代response.body_iterator，data是一个json字符串
    async for data in response.body_iterator:
        # 将data解析为json格式
        data = json.loads(data)
        # 将答案添加到contents中
        contents += data["answer"]
        # 获取docs
        docs = data["docs"]
    return contents

# 定义_PROMPT_TEMPLATE字符串模板
_PROMPT_TEMPLATE = """
用户会提出一个需要你查询知识库的问题，你应该按照我提供的思想进行思考
Question: ${{用户的问题}}
这些数据库是你能访问的，冒号之前是他们的名字，冒号之后是他们的功能：

{database_names}

你的回答格式应该按照下面的内容，请注意，格式内的```text 等标记都必须输出，这是我用来提取答案的标记。

${{知识库的名称}}




"""
# 定义数据库查询的结果
class LLMKnowledgeChain(LLMChain):
    # 初始化LLMChain对象
    llm_chain: LLMChain
    # 不推荐使用的LLM对象
    llm: Optional[BaseLanguageModel] = None
    # 不推荐使用的Prompt模板
    prompt: BasePromptTemplate = PROMPT
    # 数据库名称映射字典
    database_names: Dict[str, str] = model_container.DATABASE
    # 输入键
    input_key: str = "question"  #: :meta private:
    # 输出键
    output_key: str = "answer"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""
        # 禁止额外字段
        extra = Extra.forbid
        # 允许任意类型
        arbitrary_types_allowed = True

    # 在验证之前引发弃用警告
    @root_validator(pre=True)
    def raise_deprecation(cls, values: Dict) -> Dict:
        # 如果存在llm对象
        if "llm" in values:
            # 发出警告
            warnings.warn(
                "Directly instantiating an LLMKnowledgeChain with an llm is deprecated. "
                "Please instantiate with llm_chain argument or using the from_llm "
                "class method."
            )
            # 如果不存在llm_chain对象且llm对象不为空
            if "llm_chain" not in values and values["llm"] is not None:
                # 获取prompt对象
                prompt = values.get("prompt", PROMPT)
                # 使用llm对象实例化LLMChain对象
                values["llm_chain"] = LLMChain(llm=values["llm"], prompt=prompt)
        return values

    # 返回输入键列表
    @property
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

    # 评估表达式
    def _evaluate_expression(self, dataset, query) -> str:
        try:
            # 运行知识库搜索
            output = asyncio.run(search_knowledge_base_iter(dataset, query))
        except Exception as e:
            # 处理异常情况
            output = "输入的信息有误或不存在知识库"
            return output
        return output
    # 处理LLM结果的私有方法，返回一个包含结果的字典
    def _process_llm_result(
            self,
            llm_output: str,
            llm_input: str,
            run_manager: CallbackManagerForChainRun
    ) -> Dict[str, str]:

        # 在运行管理器上显示LLM输出文本，使用绿色字体
        run_manager.on_text(llm_output, color="green", verbose=self.verbose)

        # 去除LLM输出文本两端的空白字符
        llm_output = llm_output.strip()
        
        # 使用正则表达式匹配文本中的特定格式，提取数据库内容
        text_match = re.search(r"^```py(.*?)```", llm_output, re.DOTALL)
        if text_match:
            database = text_match.group(1).strip()
            # 调用私有方法，评估数据库表达式
            output = self._evaluate_expression(database, llm_input)
            run_manager.on_text("\nAnswer: ", verbose=self.verbose)
            run_manager.on_text(output, color="yellow", verbose=self.verbose)
            answer = "Answer: " + output
        elif llm_output.startswith("Answer:"):
            answer = llm_output
        elif "Answer:" in llm_output:
            # 如果LLM输出中包含"Answer:"，提取出最后一个"Answer:"后的内容
            answer = "Answer: " + llm_output.split("Answer:")[-1]
        else:
            # 如果LLM输出不符合预期格式，返回错误信息
            return {self.output_key: f"输入的格式不对: {llm_output}"}
        
        # 返回包含结果的字典
        return {self.output_key: answer}

    # 异步处理LLM结果的私有方法
    async def _aprocess_llm_result(
            self,
            llm_output: str,
            run_manager: AsyncCallbackManagerForChainRun,
    ) -> Dict[str, str]:
        # 在运行管理器上显示文本输出，使用绿色字体，根据详细程度输出信息
        await run_manager.on_text(llm_output, color="green", verbose=self.verbose)
        # 去除文本输出两侧的空格
        llm_output = llm_output.strip()
        # 使用正则表达式匹配文本中的特定格式，获取匹配的文本
        text_match = re.search(r"^```py(.*?)```", llm_output, re.DOTALL)
        if text_match:
            # 获取匹配的表达式
            expression = text_match.group(1)
            # 评估表达式并获取输出结果
            output = self._evaluate_expression(expression)
            # 在运行管理器上显示"Answer: "文本输出，根据详细程度输出信息
            await run_manager.on_text("\nAnswer: ", verbose=self.verbose)
            # 在运行管理器上显示输出结果，使用黄色字体，根据详细程度输出信息
            await run_manager.on_text(output, color="yellow", verbose=self.verbose)
            # 构建包含"Answer: "的答案字符串
            answer = "Answer: " + output
        elif llm_output.startswith("Answer:"):
            # 如果文本以"Answer:"开头，则答案为文本本身
            answer = llm_output
        elif "Answer:" in llm_output:
            # 如果文本中包含"Answer:"，则获取最后一个"Answer:"后的内容作为答案
            answer = "Answer: " + llm_output.split("Answer:")[-1]
        else:
            # 如果文本格式未知，则引发值错误
            raise ValueError(f"unknown format from LLM: {llm_output}")
        # 返回包含答案的字典
        return {self.output_key: answer}

    def _call(
            self,
            inputs: Dict[str, str],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # 如果未提供运行管理器，则创建一个空的运行管理器
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        # 在运行管理器上显示输入的文本
        _run_manager.on_text(inputs[self.input_key])
        # 格式化数据字符串，将数据库名称和值组成键值对
        data_formatted_str = ',\n'.join([f' "{k}":"{v}"' for k, v in self.database_names.items()])
        # 使用LLM链预测结果，传入数据库名称和格式化后的数据字符串，问题文本，停止标志和回调函数
        llm_output = self.llm_chain.predict(
            database_names=data_formatted_str,
            question=inputs[self.input_key],
            stop=["```py"],
            callbacks=_run_manager.get_child(),
        )
        # 处理LLM的结果并返回
        return self._process_llm_result(llm_output, inputs[self.input_key], _run_manager)

    async def _acall(
            self,
            inputs: Dict[str, str],
            run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # 如果未提供运行管理器，则使用空操作的回调管理器
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        # 在文本上运行回调管理器
        await _run_manager.on_text(inputs[self.input_key])
        # 格式化数据为字符串，用逗号分隔
        data_formatted_str = ',\n'.join([f' "{k}":"{v}"' for k, v in self.database_names.items()])
        # 使用LLM链进行预测
        llm_output = await self.llm_chain.apredict(
            database_names=data_formatted_str,
            question=inputs[self.input_key],
            stop=["```output"],
            callbacks=_run_manager.get_child(),
        )
        # 处理LLM结果并返回
        return await self._aprocess_llm_result(llm_output, inputs[self.input_key], _run_manager)

    @property
    def _chain_type(self) -> str:
        # 返回链的类型为"llm_knowledge_chain"
        return "llm_knowledge_chain"

    @classmethod
    def from_llm(
            cls,
            llm: BaseLanguageModel,
            prompt: BasePromptTemplate = PROMPT,
            **kwargs: Any,
    ) -> LLMKnowledgeChain:
        # 创建LLM链
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        # 返回LLM知识链实例
        return cls(llm_chain=llm_chain, **kwargs)
# 定义一个函数，用于一次性搜索知识库
def search_knowledgebase_once(query: str):
    # 从模型容器中获取模型
    model = model_container.MODEL
    # 创建一个LLMKnowledgeChain对象，从LLM模型中获取知识，打印详细信息，使用给定的提示
    llm_knowledge = LLMKnowledgeChain.from_llm(model, verbose=True, prompt=PROMPT)
    # 运行查询并获取结果
    ans = llm_knowledge.run(query)
    # 返回结果
    return ans

# 定义一个输入类，用于接收查询的位置信息
class KnowledgeSearchInput(BaseModel):
    location: str = Field(description="The query to be searched")

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 调用search_knowledgebase_once函数，传入查询字符串"大数据的男女比例"，并将结果保存在result中
    result = search_knowledgebase_once("大数据的男女比例")
    # 打印结果
    print(result)
```