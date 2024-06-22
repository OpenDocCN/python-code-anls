# `.\transformers\models\tvp\processing_tvp.py`

```py
# coding=utf-8
# 版权声明，版权归 Intel AIA 团队和 HuggingFace 公司所有
# 根据 Apache 许可证 2.0 版进行许可
# 除非符合许可证，否则不得使用此文件
# 您可以在以下网址获得许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则依法分发的软件
# 以"原样"分发，不附带任何形式的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取特定语言管理权限和限制
"""
# 导入类 ProcessorMixin，BatchEncoding 和处理工具模块
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding

# 定义类 TvpProcessor，继承 ProcessorMixin 类
class TvpProcessor(ProcessorMixin):
    r"""
    构建一个 TVP 处理器，将 TVP 图像处理器和 Bert 分词器包装成单个处理器

    [`TvpProcessor`] 提供了 [`TvpImageProcessor`] 和 [`BertTokenizerFast`] 的所有功能。参见
    [`~TvpProcessor.__call__`] 和 [`~TvpProcessor.decode`] 以获取更多信息。

    Args:
        image_processor ([`TvpImageProcessor`], *optional*):
            图像处理器是必需的输入。
        tokenizer ([`BertTokenizerFast`], *optional*):
            分词器是必需的输入。
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "TvpImageProcessor"
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")

    # 初始化方法
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)

    # 批量解码方法，将所有参数转发给 BertTokenizerFast 的 [`~PreTrainedTokenizer.batch_decode`]
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # 解码方法，将所有参数转发给 BertTokenizerFast 的 [`~PreTrainedTokenizer.decode`]
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)
    def post_process_video_grounding(self, logits, video_durations):
        """
        Compute the time of the video.

        Args:
            logits (`torch.Tensor`):
                The logits output of TvpForVideoGrounding.
            video_durations (`float`):
                The video's duration.

        Returns:
            start (`float`):
                The start time of the video.
            end (`float`):
                The end time of the video.
        """
        # 根据模型输出的logits和视频持续时间计算视频的开始时间和结束时间
        start, end = (
            round(logits.tolist()[0][0] * video_durations, 1),  # 计算视频的开始时间
            round(logits.tolist()[0][1] * video_durations, 1),  # 计算视频的结束时间
        )

        return start, end

    @property
    # Copied from transformers.models.blip.processing_blip.BlipProcessor.model_input_names
    def model_input_names(self):
        # 获取tokenizer和image_processor的模型输入名称并合并去重后返回
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
```