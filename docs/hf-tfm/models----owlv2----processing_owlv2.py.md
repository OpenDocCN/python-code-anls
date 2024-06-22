# `.\transformers\models\owlv2\processing_owlv2.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，指明版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本使用本文件，除非符合许可证的要求，否则不能使用
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按“原样”分发软件
# 软件分发时不附带任何形式的担保或条件，无论是明示的还是暗示的
# 有关许可证的详细信息，请参阅许可证

"""
OWLv2 的图像/文本处理器类
"""

# 导入 List 类型
from typing import List

# 导入 numpy 库，并将其重命名为 np
import numpy as np

# 导入处理工具
from ...processing_utils import ProcessorMixin
# 导入 BatchEncoding 类
from ...tokenization_utils_base import BatchEncoding
# 导入检查 Flax 是否可用的函数
from ...utils import is_flax_available
# 导入检查 TensorFlow 是否可用的函数
from ...utils import is_tf_available
# 导入检查 PyTorch 是否可用的函数
from ...utils import is_torch_available

# 定义 OWLv2 处理器类，继承自 ProcessorMixin
class Owlv2Processor(ProcessorMixin):
    r"""
    构造一个 OWLv2 处理器，将 [`Owlv2ImageProcessor`] 和 [`CLIPTokenizer`] 或 [`CLIPTokenizerFast`] 封装成一个处理器，
    继承了图像处理器和分词器的功能。查看 [`~OwlViTProcessor.__call__`] 和 [`~OwlViTProcessor.decode`] 获取更多信息。

    Args:
        image_processor ([`Owlv2ImageProcessor`]):
            图像处理器，是必需的输入。
        tokenizer ([`CLIPTokenizer`, `CLIPTokenizerFast`]):
            分词器，是必需的输入。
    """

    # 类属性，包含图像处理器和分词器
    attributes = ["image_processor", "tokenizer"]
    # 图像处理器类名
    image_processor_class = "Owlv2ImageProcessor"
    # 分词器类名
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")

    # 初始化方法，接受图像处理器和分词器作为参数
    def __init__(self, image_processor, tokenizer, **kwargs):
        # 调用父类 ProcessorMixin 的初始化方法
        super().__init__(image_processor, tokenizer)

    # 从 transformers.models.owlvit.processing_owlvit.OwlViTProcessor.__call__ 复制的方法，将 OWLViT 替换为 OWLv2
    # 从 transformers.models.owlvit.processing_owlvit.OwlViTProcessor.post_process_object_detection 复制的方法，将 OWLViT 替换为 OWLv2
    def post_process_object_detection(self, *args, **kwargs):
        """
        此方法将所有参数转发给 [`OwlViTImageProcessor.post_process_object_detection`]。请参考该方法的文档字符串获取更多信息。
        """
        return self.image_processor.post_process_object_detection(*args, **kwargs)

    # 从 transformers.models.owlvit.processing_owlvit.OwlViTProcessor.post_process_image_guided_detection 复制的方法，将 OWLViT 替换为 OWLv2
    def post_process_image_guided_detection(self, *args, **kwargs):
        """
        此方法将所有参数转发给 [`OwlViTImageProcessor.post_process_one_shot_object_detection`]。
        请参考该方法的文档字符串获取更多信息。
        """
        return self.image_processor.post_process_image_guided_detection(*args, **kwargs)

    # 从 transformers.models.owlvit.processing_owlvit.OwlViTProcessor.batch_decode 复制的方法
``` 
    # 此方法将所有参数转发给 CLIPTokenizerFast 的 batch_decode 方法，可参考该方法的文档获得更多信息
    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)
    
    # 此方法将所有参数转发给 CLIPTokenizerFast 的 decode 方法，可参考该方法的文档获得更多信息   
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
```