# `.\models\dpt\feature_extraction_dpt.py`

```
# 设定编码格式为 utf-8
# 版权声明等信息
# 特征提取类为 DPT 的特征提取器
# 导入警告模块
# 导入 logging 工具类
# 从图像处理的 DPT 模块中导入 DPTImageProcessor 类

# 获取记录器
logger = logging.get_logger(__name__)

# 定义 DPTFeatureExtractor 类，继承自 DPTImageProcessor 类
class DPTFeatureExtractor(DPTImageProcessor):
    # 初始化方法，接受任意数量的参数和关键字参数
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，DPTFeatureExtractor 类已被弃用，将在 Transformers 的第五个版本中被移除，建议使用 DPTImageProcessor
        warnings.warn(
            "The class DPTFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use DPTImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
```