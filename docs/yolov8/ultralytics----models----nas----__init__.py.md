# `.\yolov8\ultralytics\models\nas\__init__.py`

```py
# 导入自定义模块中的 NAS 模型类
from .model import NAS

# 导入自定义模块中的 NASPredictor 类
from .predict import NASPredictor

# 导入自定义模块中的 NASValidator 类
from .val import NASValidator

# 设置 __all__ 变量，指定在使用 from module import * 时导入的符号列表
__all__ = "NASPredictor", "NASValidator", "NAS"
```