# `.\chatglm4-finetune\composite_demo\src\tools\interface.py`

```py
# 从数据类模块导入 dataclass 装饰器
from dataclasses import dataclass
# 从类型提示模块导入 Any 类型
from typing import Any

# 定义一个数据类 ToolObservation，自动生成初始化方法等
@dataclass
class ToolObservation:
    # 定义 content_type 属性，表示内容的类型
    content_type: str
    # 定义 text 属性，表示文本内容
    text: str
    # 定义 image_url 属性，表示图像的 URL，可以为 None
    image_url: str | None = None
    # 定义 role_metadata 属性，表示角色的元数据，可以为 None
    role_metadata: str | None = None
    # 定义 metadata 属性，表示其他元数据，可以是任何类型
    metadata: Any = None
```