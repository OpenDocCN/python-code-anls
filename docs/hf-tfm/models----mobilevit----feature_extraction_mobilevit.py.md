# `.\transformers\models\mobilevit\feature_extraction_mobilevit.py`

```
# 设置文件编码为 UTF-8
# 版权声明和许可证信息
# 导入警告模块
# 导入日志模块
# 导入 MobileViTImageProcessor 类

# 获取日志记录器
logger = logging.get_logger(__name__)

# MobileViTFeatureExtractor 类，继承自 MobileViTImageProcessor 类
class MobileViTFeatureExtractor(MobileViTImageProcessor):
    # 初始化函数
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告信息，MobileViTFeatureExtractor 类即将被弃用，并将在 Transformers 的版本 5 中被移除
        warnings.warn(
            "The class MobileViTFeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use MobileViTImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类的初始化函数
        super().__init__(*args, **kwargs)
```