# `MetaGPT\tests\metagpt\strategy\examples\game24.py`

```py

# -*- coding: utf-8 -*-  # 设置文件编码格式为 UTF-8
# @Date    : 12/25/2023 1:36 AM  # 代码编写日期
# @Author  : stellahong (stellahong@fuzhi.ai)  # 作者信息
# @Desc    :  # 代码描述

import re  # 导入正则表达式模块
from typing import Dict  # 导入类型提示模块中的字典类型

from metagpt.strategy.tot import TreeofThought  # 从 metagpt.strategy.tot 模块中导入 TreeofThought 类
from metagpt.strategy.tot_schema import (  # 从 metagpt.strategy.tot_schema 模块中导入 BaseEvaluator、BaseParser、Strategy、ThoughtSolverConfig 类
    BaseEvaluator,
    BaseParser,
    Strategy,
    ThoughtSolverConfig,
)
from tests.metagpt.strategy.prompt_templates.game24 import propose_prompt, value_prompt  # 从 tests.metagpt.strategy.prompt_templates.game24 模块中导入 propose_prompt 和 value_prompt

# 定义 Game24Parser 类，继承自 BaseParser 类
class Game24Parser(BaseParser):
    propose_prompt: str = propose_prompt  # 定义 propose_prompt 属性并赋值为导入的 propose_prompt
    value_prompt: str = value_prompt  # 定义 value_prompt 属性并赋值为导入的 value_prompt

    # 定义 __call__ 方法，接收 input_text 参数并返回字符串
    def __call__(self, input_text: str) -> str:
        last_line = input_text.strip().split("\n")[-1]  # 获取输入文本的最后一行
        return last_line.split("left: ")[-1].split(")")[0]  # 返回处理后的字符串

    # 定义 propose 方法，接收 current_state 和 kwargs 参数并返回字符串
    def propose(self, current_state: str, **kwargs) -> str:
        return self.propose_prompt.format(input=current_state, **kwargs)  # 格式化返回提示字符串

    # 定义 value 方法，接收 input 和 kwargs 参数并返回字符串
    def value(self, input: str = "", **kwargs) -> str:
        node_result = self(input)  # 调用自身方法处理输入
        return self.value_prompt.format(input=node_result)  # 格式化返回值提示字符串

# 定义 Game24Evaluator 类，继承自 BaseEvaluator 类
class Game24Evaluator(BaseEvaluator):
    value_map: Dict[str, float] = {"impossible": 0.001, "likely": 1, "sure": 20}  # 定义 value_map 属性为字符串到浮点数的字典
    status_map: Dict = {val: key for key, val in value_map.items()}  # 定义 status_map 属性为 value_map 的键值对颠倒的字典

    # 定义 __call__ 方法，接收 evaluation 和 kwargs 参数并返回浮点数
    def __call__(self, evaluation: str, **kwargs) -> float:
        try:
            matches = re.findall(r"\b(impossible|sure|likely)\b", evaluation)  # 使用正则表达式匹配字符串中的单词
            value = self.value_map[matches[0]]  # 获取匹配结果对应的值
        except:
            value = 0.001  # 处理异常情况
        return value  # 返回值

    # 定义 status_verify 方法，接收 value 参数并返回布尔值
    def status_verify(self, value):
        status = False  # 初始化状态为 False
        if value in self.status_map:  # 如果值在 status_map 中
            status_value = self.status_map[value]  # 获取对应的状态值
            if status_value != "impossible":  # 如果状态值不是 "impossible"
                status = True  # 更新状态为 True
        return status  # 返回状态值

# 定义测试函数 test_game24
def test_game24():
    import asyncio  # 导入异步 I/O 模块

    initial_prompt = """4 5 6 10"""  # 初始化提示字符串
    parser = Game24Parser()  # 创建 Game24Parser 实例
    evaluator = Game24Evaluator()  # 创建 Game24Evaluator 实例

    config = ThoughtSolverConfig(n_generate_sample=5, parser=parser, evaluator=evaluator)  # 创建 ThoughtSolverConfig 实例

    tot = TreeofThought(strategy=Strategy.BFS, config=config)  # 创建 TreeofThought 实例
    asyncio.run(tot.solve(init_prompt=initial_prompt))  # 运行异步任务

```