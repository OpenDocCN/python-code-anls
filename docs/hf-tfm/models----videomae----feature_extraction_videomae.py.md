# `.\transformers\models\videomae\feature_extraction_videomae.py`

```py
# 设置编码格式为 UTF-8

# 版权声明

# 导入警告模块

# 从 utils 模块中导入日志记录器

# 从 image_processing_videomae 模块中导入 VideoMAEImageProcessor 类

# 从 logging 模块中获取日志记录器对象
logger = logging.get_logger(__name__)

# 定义 VideoMAEFeatureExtractor 类，继承自 VideoMAEImageProcessor 类
class VideoMAEFeatureExtractor(VideoMAEImageProcessor):
    # 定义初始化方法
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，表示 VideoMAEFeatureExtractor 类已被废弃，并将在 Transformers 版本 5 中移除
        # 建议使用 VideoMAEImageProcessor 代替
        warnings.warn(
            "The class VideoMAEFeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use VideoMAEImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类 VideoMAEImageProcessor 的初始化方法
        super().__init__(*args, **kwargs)
```