# `.\models\tvlt\processing_tvlt.py`

```
# coding=utf-8
# 版权所有 2023 年 HuggingFace Inc. 团队。
#
# 根据 Apache 许可证 2.0 版本授权；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 不提供任何形式的担保或条件。详见许可证。
"""
TVLT 的处理器类。
"""

from ...processing_utils import ProcessorMixin

class TvltProcessor(ProcessorMixin):
    r"""
    构建一个 TVLT 处理器，将 TVLT 图像处理器和 TVLT 特征提取器包装成一个单一的处理器。

    [`TvltProcessor`] 提供了 [`TvltImageProcessor`] 和 [`TvltFeatureExtractor`] 的所有功能。查看
    [`~TvltProcessor.__call__`] 的文档字符串以获取更多信息。

    Args:
        image_processor (`TvltImageProcessor`):
            [`TvltImageProcessor`] 的实例。图像处理器是必需的输入。
        feature_extractor (`TvltFeatureExtractor`):
            [`TvltFeatureExtractor`] 的实例。特征提取器是必需的输入。
    """

    attributes = ["image_processor", "feature_extractor"]
    image_processor_class = "TvltImageProcessor"
    feature_extractor_class = "TvltFeatureExtractor"

    def __init__(self, image_processor, feature_extractor):
        super().__init__(image_processor=image_processor, feature_extractor=feature_extractor)

        self.image_processor = image_processor  # 初始化图像处理器
        self.feature_extractor = feature_extractor  # 初始化特征提取器

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

        # 检查输入参数 `images` 和 `audio` 是否都为 None
        if images is None and audio is None:
            # 如果都为 None，则抛出数值错误异常
            raise ValueError("You need to specify either an `images` or `audio` input to process.")

        images_mixed_dict = None
        # 如果 `images` 参数不为 None
        if images is not None:
            # 调用 self.image_processor 对象的 preprocess 方法处理 images 参数
            images_dict = self.image_processor(images, mask_pixel=mask_pixel, *args, **kwargs)
        
        # 如果 `images_mixed` 参数不为 None
        if images_mixed is not None:
            # 调用 self.image_processor 对象的 preprocess 方法处理 images_mixed 参数，设置 is_mixed 为 True
            images_mixed_dict = self.image_processor(images_mixed, is_mixed=True, *args, **kwargs)
        
        # 如果 `audio` 参数不为 None
        if audio is not None:
            # 调用 self.feature_extractor 对象的 __call__ 方法处理 audio 参数
            audio_dict = self.feature_extractor(
                audio, *args, sampling_rate=sampling_rate, mask_audio=mask_audio, **kwargs
            )

        # 初始化空的输出字典
        output_dict = {}
        # 如果 `audio` 参数不为 None，则将 audio_dict 中的内容更新到 output_dict 中
        if audio is not None:
            output_dict.update(audio_dict)
        # 如果 `images` 参数不为 None，则将 images_dict 中的内容更新到 output_dict 中
        if images is not None:
            output_dict.update(images_dict)
        # 如果 `images_mixed_dict` 不为 None，则将 images_mixed_dict 中的内容更新到 output_dict 中
        if images_mixed_dict is not None:
            output_dict.update(images_mixed_dict)
        
        # 返回最终的输出字典
        return output_dict

    @property
    def model_input_names(self):
        # 获取 self.image_processor 对象的 model_input_names 属性
        image_processor_input_names = self.image_processor.model_input_names
        # 获取 self.feature_extractor 对象的 model_input_names 属性
        feature_extractor_input_names = self.feature_extractor.model_input_names
        # 将两个列表合并成一个，并去除重复元素，作为最终的模型输入名称列表
        return list(dict.fromkeys(image_processor_input_names + feature_extractor_input_names))
```