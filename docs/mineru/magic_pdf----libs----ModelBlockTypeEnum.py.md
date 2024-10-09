# `.\MinerU\magic_pdf\libs\ModelBlockTypeEnum.py`

```
# 导入 Enum 模块，用于创建枚举类
from enum import Enum

# 定义一个枚举类 ModelBlockTypeEnum，用于表示模型块的类型
class ModelBlockTypeEnum(Enum):
    # 定义标题类型，值为 0
    TITLE = 0
    # 定义普通文本类型，值为 1
    PLAIN_TEXT = 1
    # 定义弃用类型，值为 2
    ABANDON = 2
    # 定义孤立公式类型，值为 8
    ISOLATE_FORMULA = 8
    # 定义嵌入类型，值为 13
    EMBEDDING = 13
    # 定义孤立类型，值为 14
    ISOLATED = 14
```