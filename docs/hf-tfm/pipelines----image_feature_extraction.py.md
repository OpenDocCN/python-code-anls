# `.\pipelines\image_feature_extraction.py`

```py
# 导入类型提示字典
from typing import Dict

# 从当前包的utils模块中导入函数和变量
from ..utils import add_end_docstrings, is_vision_available

# 从当前包的base模块中导入指定类和函数
from .base import GenericTensor, Pipeline, build_pipeline_init_args

# 如果视觉功能可用，则从当前包的image_utils模块中导入load_image函数
if is_vision_available():
    from ..image_utils import load_image

# 使用装饰器添加终端文档字符串和初始化参数
@add_end_docstrings(
    build_pipeline_init_args(has_image_processor=True),  # 使用构建管道初始化参数装饰函数
    """
        image_processor_kwargs (`dict`, *optional*):
                Additional dictionary of keyword arguments passed along to the image processor e.g.
                {"size": {"height": 100, "width": 100}}
        pool (`bool`, *optional*, defaults to `False`):
            Whether or not to return the pooled output. If `False`, the model will return the raw hidden states.
    """,
)
# 图像特征提取管道类，继承自Pipeline基类
class ImageFeatureExtractionPipeline(Pipeline):
    """
    Image feature extraction pipeline uses no model head. This pipeline extracts the hidden states from the base
    transformer, which can be used as features in downstream tasks.

    Example:

    ```
    >>> from transformers import pipeline

    >>> extractor = pipeline(model="google/vit-base-patch16-224", task="image-feature-extraction")
    >>> result = extractor("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png", return_tensors=True)
    >>> result.shape  # This is a tensor of shape [1, sequence_lenth, hidden_dimension] representing the input image.
    torch.Size([1, 197, 768])
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This image feature extraction pipeline can currently be loaded from [`pipeline`] using the task identifier:
    `"image-feature-extraction"`.

    All vision models may be used for this pipeline. See a list of all models, including community-contributed models on
    [huggingface.co/models](https://huggingface.co/models).
    """

    # 参数清理函数，处理图像处理器参数和后处理参数
    def _sanitize_parameters(self, image_processor_kwargs=None, return_tensors=None, pool=None, **kwargs):
        # 如果没有指定图像处理器参数，设置为空字典
        preprocess_params = {} if image_processor_kwargs is None else image_processor_kwargs

        # 设置后处理参数为空字典
        postprocess_params = {}

        # 如果指定了池化参数，将其添加到后处理参数中
        if pool is not None:
            postprocess_params["pool"] = pool
        # 如果指定了返回张量参数，将其添加到后处理参数中
        if return_tensors is not None:
            postprocess_params["return_tensors"] = return_tensors

        # 如果kwargs中包含超时参数，将其添加到预处理参数中
        if "timeout" in kwargs:
            preprocess_params["timeout"] = kwargs["timeout"]

        # 返回预处理参数、空字典和后处理参数
        return preprocess_params, {}, postprocess_params

    # 预处理函数，加载图像并使用图像处理器处理，返回模型输入
    def preprocess(self, image, timeout=None, **image_processor_kwargs) -> Dict[str, GenericTensor]:
        # 加载图像数据
        image = load_image(image, timeout=timeout)
        # 使用图像处理器处理图像并返回模型输入
        model_inputs = self.image_processor(image, return_tensors=self.framework, **image_processor_kwargs)
        return model_inputs

    # 前向传播函数，使用模型进行推理并返回模型输出
    def _forward(self, model_inputs):
        # 使用模型进行推理并返回输出
        model_outputs = self.model(**model_inputs)
        return model_outputs
    # 定义一个方法用于后处理模型输出
    def postprocess(self, model_outputs, pool=None, return_tensors=False):
        # 如果 pool 参数为真，则检查是否有 "pooler_output" 在模型输出中
        pool = pool if pool is not None else False

        if pool:
            # 如果使用了池化选项且模型输出中没有 "pooler_output"，则抛出数值错误
            if "pooler_output" not in model_outputs:
                raise ValueError(
                    "No pooled output was returned. Make sure the model has a `pooler` layer when using the `pool` option."
                )
            # 将模型输出中的 "pooler_output" 赋值给 outputs
            outputs = model_outputs["pooler_output"]
        else:
            # 如果未使用池化选项，则将模型输出中的第一个张量（logits 或者 last_hidden_state）赋值给 outputs
            # [0] is the first available tensor, logits or last_hidden_state.
            outputs = model_outputs[0]

        # 如果设置了 return_tensors，则直接返回 outputs
        if return_tensors:
            return outputs
        # 根据指定的深度学习框架返回 outputs 的转换结果
        if self.framework == "pt":
            return outputs.tolist()  # 返回 PyTorch 张量的转换为列表
        elif self.framework == "tf":
            return outputs.numpy().tolist()  # 返回 TensorFlow 张量的转换为列表

    # 定义一个方法使对象可被调用，用于提取输入的特征
    def __call__(self, *args, **kwargs):
        """
        Extract the features of the input(s).

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images, which must then be passed as a string.
                Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL
                images.
            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is used and
                the call may block forever.
        Return:
            A nested list of `float`: The features computed by the model.
        """
        # 调用父类的 __call__ 方法，并传递所有的位置参数和关键字参数
        return super().__call__(*args, **kwargs)
```