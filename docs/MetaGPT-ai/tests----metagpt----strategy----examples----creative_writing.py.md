# `MetaGPT\tests\metagpt\strategy\examples\creative_writing.py`

```py

# -*- coding: utf-8 -*-
# @Date    : 12/25/2023 1:06 PM
# @Author  : stellahong (stellahong@fuzhi.ai)
# @Desc    :

# 导入所需的模块
import re
from typing import Dict

# 导入自定义模块
from metagpt.strategy.tot import TreeofThought
from metagpt.strategy.tot_schema import (
    BaseEvaluator,
    BaseParser,
    Strategy,
    ThoughtSolverConfig,
)
from tests.metagpt.strategy.prompt_templates.creative_writing import (
    cot_prompt,
    vote_prompt,
)

# 定义一个类 TextGenParser，继承自 BaseParser
class TextGenParser(BaseParser):
    # 定义类属性 propose_prompt 和 value_prompt
    propose_prompt: str = cot_prompt
    value_prompt: str = vote_prompt

    # 定义 __call__ 方法，接受一个字符串参数并返回一个字符串
    def __call__(self, input_text: str) -> str:
        return input_text

    # 定义 propose 方法，接受 current_state 和 kwargs 参数，并返回一个字符串
    def propose(self, current_state: str, **kwargs) -> str:
        return self.propose_prompt.format(input=current_state, **kwargs)

    # 定义 value 方法，接受 input 和 kwargs 参数，并返回一个字符串
    def value(self, input: str = "", **kwargs) -> str:
        # 获取 node_id 参数的值，如果不存在则默认为 "0"
        id = kwargs.get("node_id", "0")
        return self.value_prompt + f"Choice {id}:\n{input}\n"

# 定义一个类 TextGenEvaluator，继承自 BaseEvaluator
class TextGenEvaluator(BaseEvaluator):
    # 定义类属性 value_map，是一个字符串到浮点数的字典
    value_map: Dict[str, float] = {"impossible": 0.001, "likely": 1, "sure": 20}  # TODO: ad hoc
    # 定义类属性 status_map，是一个值到键的字典，通过 value_map 生成
    status_map: Dict = {val: key for key, val in value_map.items()}

    # 定义 __call__ 方法，接受 evaluation 和 kwargs 参数，并返回一个浮点数
    def __call__(self, evaluation: str, **kwargs) -> float:
        try:
            value = 0
            # 获取 node_id 参数的值，如果不存在则默认为 "0"
            node_id = kwargs.get("node_id", "0")
            # 定义正则表达式模式
            pattern = r".*best choice is .*(\d+).*"
            # 使用正则表达式匹配 evaluation
            match = re.match(pattern, evaluation, re.DOTALL)

            if match:
                # 获取匹配到的数字
                vote = int(match.groups()[0])
                print(vote)
                # 如果 vote 等于 node_id，则将 value 设为 1
                if vote == int(node_id):
                    value = 1
        except:
            value = 0
        return value

    # 定义 status_verify 方法，接受 value 参数，并返回一个布尔值
    def status_verify(self, value):
        status = False
        # 如果 value 在 status_map 中
        if value in self.status_map:
            status_value = self.status_map[value]
            # 如果 status_value 不是 "impossible"，则将 status 设为 True
            if status_value != "impossible":
                status = True
        return status

# 定义一个测试函数 test_creative_writing
def test_creative_writing():
    # 导入 asyncio 模块
    import asyncio

    # 初始化 initial_prompt
    initial_prompt = """It isn't difficult to do a handstand if you just stand on your hands. It caught him off guard that space smelled of seared steak. When she didn’t like a guy who was trying to pick her up, she started using sign language. Each person who knows you has a different perception of who you are."""

    # 创建 TextGenParser 实例
    parser = TextGenParser()
    # 创建 TextGenEvaluator 实例
    evaluator = TextGenEvaluator()

    # 创建 ThoughtSolverConfig 实例
    config = ThoughtSolverConfig(max_step=2, n_generate_sample=1, n_select_sample=1, parser=parser, evaluator=evaluator)

    # 创建 TreeofThought 实例
    tot_base = TreeofThought(strategy=Strategy.BFS, config=config)
    # 运行 tot_base 的 solve 方法
    asyncio.run(tot_base.solve(init_prompt=initial_prompt))

```