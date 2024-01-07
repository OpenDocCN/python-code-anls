# `KubiScan\engine\priority.py`

```

# 导入枚举类型模块
from enum import Enum

# 定义优先级枚举类型
class Priority(Enum):
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    NONE = 0

# 根据优先级名称获取对应的优先级枚举值
def get_priority_by_name(priority):
    return {
        Priority.CRITICAL.name: Priority.CRITICAL,
        Priority.HIGH.name: Priority.HIGH,
        Priority.MEDIUM.name: Priority.MEDIUM,
        Priority.LOW.name: Priority.LOW,
        Priority.NONE.name: Priority.NONE,
    }[priority]

```