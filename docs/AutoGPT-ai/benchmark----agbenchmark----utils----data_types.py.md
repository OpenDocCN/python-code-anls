# `.\AutoGPT\benchmark\agbenchmark\utils\data_types.py`

```py
# 导入所需的模块和类型
from enum import Enum
from typing import Literal
from pydantic import BaseModel

# 定义困难级别的枚举类
class DifficultyLevel(Enum):
    interface = "interface"
    basic = "basic"
    novice = "novice"
    intermediate = "intermediate"
    advanced = "advanced"
    expert = "expert"
    human = "human"

# 将枚举类映射到困难级别的数字
DIFFICULTY_MAP = {
    DifficultyLevel.interface: 1,
    DifficultyLevel.basic: 2,
    DifficultyLevel.novice: 3,
    DifficultyLevel.intermediate: 4,
    DifficultyLevel.advanced: 5,
    DifficultyLevel.expert: 6,
    DifficultyLevel.human: 7,
}

# 将困难级别的字符串值映射到对应的数字
STRING_DIFFICULTY_MAP = {e.value: DIFFICULTY_MAP[e] for e in DifficultyLevel}

# 定义类别的枚举类
class Category(str, Enum):
    DATA = "data"
    GENERALIST = "general"
    CODING = "coding"
    SCRAPE_SYNTHESIZE = "scrape_synthesize"
    WEB = "web"
    GAIA_1 = "GAIA_1"
    GAIA_2 = "GAIA_2"
    GAIA_3 = "GAIA_3"

# 定义评估结果的数据模型
class EvalResult(BaseModel):
    result: str
    result_source: Literal["step_output"] | str
    score: float
    passed: bool
```