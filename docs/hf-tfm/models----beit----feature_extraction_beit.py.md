# `.\transformers\models\beit\feature_extraction_beit.py`

```
# 设置文件编码为 UTF-8
# 版权声明和许可信息，表明代码的版权和使用许可
# 导入警告模块
# 从工具包中导入日志模块
from ...utils import logging

# 获取名为 '__main__' 的模块的日志记录器
logger = logging.get_logger(__name__)

# 创建一个名为 BeitFeatureExtractor 的类，它继承自 BeitImageProcessor 类
class BeitFeatureExtractor(BeitImageProcessor):
    # 类的初始化方法，接受任意位置参数和关键字参数
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，提示 BeitFeatureExtractor 类已被弃用，将在 Transformers 的第 5 版中删除，建议使用 BeitImageProcessor 类代替
        warnings.warn(
            "The class BeitFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use BeitImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类的初始化方法，并传递所有的位置参数和关键字参数
        super().__init__(*args, **kwargs)
```