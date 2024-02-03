# `.\AutoGPT\autogpts\autogpt\autogpt\core\planning\prompt_strategies\__init__.py`

```py
# 从当前目录中导入 initial_plan 模块中的 InitialPlan 和 InitialPlanConfiguration 类
from .initial_plan import InitialPlan, InitialPlanConfiguration
# 从当前目录中导入 name_and_goals 模块中的 NameAndGoals 和 NameAndGoalsConfiguration 类
from .name_and_goals import NameAndGoals, NameAndGoalsConfiguration
# 从当前目录中导入 next_ability 模块中的 NextAbility 和 NextAbilityConfiguration 类

from .next_ability import NextAbility, NextAbilityConfiguration

# 定义一个列表，包含需要导出的类名
__all__ = [
    "InitialPlan",
    "InitialPlanConfiguration",
    "NameAndGoals",
    "NameAndGoalsConfiguration",
    "NextAbility",
    "NextAbilityConfiguration",
]
```