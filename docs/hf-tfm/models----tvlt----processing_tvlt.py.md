# `.\transformers\models\tvlt\processing_tvlt.py`

```py
# 设置编码格式为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本许可
# 获取许可证的副本链接
# 在适用法律要求或书面同意的情况下，以"原样"的方式分发软件，不包括任何明示或暗示的担保或条件
# 请参阅许可证以了解具体语言和限制

"""
TVLT 的处理器类。
"""

# 从 processing_utils 模块导入 ProcessorMixin
from ...processing_utils import ProcessorMixin

# TVLT 处理器类，继承自 ProcessorMixin
class TvltProcessor(ProcessorMixin):
    r"""
    构建 TVLT 处理器，将 TVLT 图像处理器和 TVLT 特征提取器封装成单个处理器。

    [`TvltProcessor`] 提供了 [`TvltImageProcessor`] 和 [`TvltFeatureExtractor`] 的所有功能。查看 [`~TvltProcessor.__call__`] 的文档字符串以获取更多信息。

    参数:
        image_processor (`TvltImageProcessor`):
            [`TvltImageProcessor`] 的实例。图像处理器是必需的输入。
        feature_extractor (`TvltFeatureExtractor`):
            [`TvltFeatureExtractor`] 的实例。特征提取器是必需的输入。
    """

    # 类属性
    attributes = ["image_processor", "feature_extractor"]
    image_processor_class = "TvltImageProcessor"
    feature_extractor_class = "TvltFeatureExtractor"

    # 构造函数
    def __init__(self, image_processor, feature_extractor):
        super().__init__(image_processor=image_processor, feature_extractor=feature_extractor)

        # 初始化属性
        self.image_processor = image_processor
        self.feature_extractor = feature_extractor

    # __call__ 方法
    def __call__(
        self,
        images=None,
        audio=None,
        images_mixed=None,
        sampling_rate=None,
        mask_audio=False,
        mask_pixel=False,
        *args,
        **kwargs,
    ):
        """
        Forwards the `images` argument to TvltImageProcessor's [`~TvltImageProcessor.preprocess`] and the `audio`
        argument to TvltFeatureExtractor's [`~TvltFeatureExtractor.__call__`]. Please refer to the docstring of the
        above two methods for more information.
        """

        # 如果没有指定 images 和 audio 输入，则抛出数值错误
        if images is None and audio is None:
            raise ValueError("You need to specify either an `images` or `audio` input to process.")

        # 初始化 images_mixed_dict
        images_mixed_dict = None
        # 如果指定了 images，则调用 image_processor 方法处理
        if images is not None:
            images_dict = self.image_processor(images, mask_pixel=mask_pixel, *args, **kwargs)
        # 如果指定了 images_mixed，则调用 image_processor 方法处理，指定 is_mixed=True
        if images_mixed is not None:
            images_mixed_dict = self.image_processor(images_mixed, is_mixed=True, *args, **kwargs)
        # 如果指定了 audio，则调用 feature_extractor 方法处理
        if audio is not None:
            audio_dict = self.feature_extractor(
                audio, *args, sampling_rate=sampling_rate, mask_audio=mask_audio, **kwargs
            )

        # 初始化 output_dict
        output_dict = {}
        # 如果指定了 audio，则更新 output_dict
        if audio is not None:
            output_dict.update(audio_dict)
        # 如果指定了 images，则更新 output_dict
        if images is not None:
            output_dict.update(images_dict)
        # 如果 images_mixed_dict 不为 None，则更新 output_dict
        if images_mixed_dict is not None:
            output_dict.update(images_mixed_dict)
        # 返回结果字典
        return output_dict

    # 获取 model_input_names 属性
    @property
    def model_input_names(self):
        # 获取 image_processor 和 feature_extractor 的 model_input_names 属性，合并并去重
        image_processor_input_names = self.image_processor.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(image_processor_input_names + feature_extractor_input_names))
```