# `.\DB-GPT-src\dbgpt\serve\agent\team\base.py`

```py
from enum import Enum

# 导入枚举类型模块 `Enum`


class TeamMode(Enum):

# 定义枚举类型 `TeamMode`


    AUTO_PLAN = "auto_plan"

# 枚举成员 `AUTO_PLAN`，对应的值是字符串 `"auto_plan"`


    AWEL_LAYOUT = "awel_layout"

# 枚举成员 `AWEL_LAYOUT`，对应的值是字符串 `"awel_layout"`


    SINGLE_AGENT = "singe_agent"

# 枚举成员 `SINGLE_AGENT`，对应的值是字符串 `"singe_agent"`
```