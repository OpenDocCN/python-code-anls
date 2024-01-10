# `MetaGPT\metagpt\strategy\tot_schema.py`

```

# -*- coding: utf-8 -*- 
# 设置文件编码为 UTF-8

# 导入需要的模块
from enum import Enum
from pydantic import BaseModel, Field
from metagpt.strategy.base import BaseEvaluator, BaseParser

# 定义枚举类型 MethodSelect，包含两个选项：SAMPLE 和 GREEDY
class MethodSelect(Enum):
    SAMPLE = "sample"
    GREEDY = "greedy"

# 定义枚举类型 Strategy，包含三个选项：BFS、DFS 和 MCTS
class Strategy(Enum):
    BFS = "BFS"
    DFS = "DFS"
    MCTS = "MCTS"

# 定义 ThoughtSolverConfig 类，继承自 BaseModel
class ThoughtSolverConfig(BaseModel):
    # 定义属性 max_steps，默认值为 3
    max_steps: int = 3
    # 定义属性 method_select，默认值为 MethodSelect.GREEDY，可选值为 "sample" 或 "greedy"
    method_select: str = MethodSelect.GREEDY  
    # 定义属性 n_generate_sample，默认值为 5
    n_generate_sample: int = 5  # per node
    # 定义属性 n_select_sample，默认值为 3
    n_select_sample: int = 3  # per path
    # 定义属性 n_solution_sample，默认值为 5，仅适用于 DFS
    n_solution_sample: int = 5  # only for dfs
    # 定义属性 parser，默认工厂值为 BaseParser
    parser: BaseParser = Field(default_factory=BaseParser)
    # 定义属性 evaluator，默认工厂值为 BaseEvaluator
    evaluator: BaseEvaluator = Field(default_factory=BaseEvaluator)

```