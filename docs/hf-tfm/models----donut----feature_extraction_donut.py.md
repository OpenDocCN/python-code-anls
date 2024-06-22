# `.\models\donut\feature_extraction_donut.py`

```py
# 设置编码格式为 UTF-8
# 版权声明
# 根据 Apache 许可证版本 2.0 保护权利
# 除非符合许可证要求或经许可证书上书面同意，否则不得使用此文件
# 您可以在以下网址获得许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则依据"AS IS"基础分发软件，无论是明示的还是暗示的，都没有任何形式的担保或条件
# 请参阅许可证以了解特定语言指定权限和限制
# 用于 Donut 的特征提取器类
# 引入警告模块
# 引入日志模块
# 从图像处理模块 image_processing_donut 中引入 DonutImageProcessor 类
# 获取当前模块的日志记录器
# 定义 DonutFeatureExtractor 类，继承自 DonutImageProcessor 类
def __init__(self, *args, **kwargs) -> None:
    # 发出警告信息，告知 DonutFeatureExtractor 类已弃用，并将在 Transformers 的第 5 版中移除
    # 建议使用 DonutImageProcessor 类代替
    warnings.warn(
        "The class DonutFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
        " use DonutImageProcessor instead.",
        FutureWarning,
    )
    # 调用父类的构造函数
    super().__init__(*args, **kwargs)
```