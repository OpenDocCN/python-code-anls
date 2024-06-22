# `.\transformers\pipelines\image_to_text.py`

```py
# 从 typing 模块导入 List 和 Union 类型，用于类型提示
from typing import List, Union

# 从 ..utils 模块导入各种函数和变量，包括 add_end_docstrings、is_tf_available、is_torch_available、is_vision_available、logging、requires_backends
from ..utils import (
    add_end_docstrings,
    is_tf_available,
    is_torch_available,
    is_vision_available,
    logging,
    requires_backends,
)

# 从 .base 模块导入 PIPELINE_INIT_ARGS 和 Pipeline 类
from .base import PIPELINE_INIT_ARGS, Pipeline

# 如果 is_vision_available() 返回 True，说明视觉处理模块可用
if is_vision_available():
    # 从 PIL 模块导入 Image 类
    from PIL import Image
    # 从 ..image_utils 模块导入 load_image 函数
    from ..image_utils import load_image

# 如果 is_tf_available() 返回 True，说明 TensorFlow 可用
if is_tf_available():
    # 从 ..models.auto.modeling_tf_auto 模块导入 TF_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES 变量
    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES

# 如果 is_torch_available() 返回 True，说明 PyTorch 可用
if is_torch_available():
    # 从 ..models.auto.modeling_auto 模块导入 MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES 变量
    import torch
    from ..models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES

# 从 logging 模块获取 logger 对象
logger = logging.get_logger(__name__)

# 使用 @add_end_docstrings 装饰器添加文档字符串，内容为 PIPELINE_INIT_ARGS 变量的值
@add_end_docstrings(PIPELINE_INIT_ARGS)
# 定义 ImageToTextPipeline 类，继承自 Pipeline 类
class ImageToTextPipeline(Pipeline):
    """
    Image To Text pipeline using a `AutoModelForVision2Seq`. This pipeline predicts a caption for a given image.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> captioner = pipeline(model="ydshieh/vit-gpt2-coco-en")
    >>> captioner("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    [{'generated_text': 'two birds are standing next to each other '}]
    ```py

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This image to text pipeline can currently be loaded from pipeline() using the following task identifier:
    "image-to-text".

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?pipeline_tag=image-to-text).
    """

    # 定义初始化方法
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
        # 检查是否需要 vision 后端
        requires_backends(self, "vision")
        # 检查模型类型
        self.check_model_type(
            TF_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES if self.framework == "tf" else MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
        )

    # 定义 _sanitize_parameters 方法，用于清理参数
    def _sanitize_parameters(self, max_new_tokens=None, generate_kwargs=None, prompt=None, timeout=None):
        # 初始化 forward_kwargs 和 preprocess_params 字典
        forward_kwargs = {}
        preprocess_params = {}

        # 如果 prompt 参数不为空，则将其添加到 preprocess_params 中
        if prompt is not None:
            preprocess_params["prompt"] = prompt
        # 如果 timeout 参数不为空，则将其添加到 preprocess_params 中
        if timeout is not None:
            preprocess_params["timeout"] = timeout

        # 如果 generate_kwargs 参数不为空，则将其添加到 forward_kwargs 中
        if generate_kwargs is not None:
            forward_kwargs["generate_kwargs"] = generate_kwargs
        # 如果 max_new_tokens 参数不为空，则将其添加到 forward_kwargs 中
        if max_new_tokens is not None:
            # 如果 forward_kwargs 中没有 generate_kwargs，则初始化 generate_kwargs
            if "generate_kwargs" not in forward_kwargs:
                forward_kwargs["generate_kwargs"] = {}
            # 如果 generate_kwargs 中已经定义了 max_new_tokens，则引发 ValueError
            if "max_new_tokens" in forward_kwargs["generate_kwargs"]:
                raise ValueError(
                    "'max_new_tokens' is defined twice, once in 'generate_kwargs' and once as a direct parameter,"
                    " please use only one"
                )
            # 否则，将 max_new_tokens 添加到 generate_kwargs 中
            forward_kwargs["generate_kwargs"]["max_new_tokens"] = max_new_tokens
        # 返回 preprocess_params 和 forward_kwargs 字典
        return preprocess_params, forward_kwargs, {}
    # 重写 __call__ 方法，用来给传入的图像分配标签
    def __call__(self, images: Union[str, List[str], "Image.Image", List["Image.Image"]], **kwargs):
        """
        # 给传入的图像（单张或批量）分配标签
        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                # 图像可以是三种类型：

                - 包含指向图像的 HTTP(s) 链接的字符串
                - 包含指向本地图像的本地路径的字符串
                - 直接加载在 PIL 中的图像

            max_new_tokens (`int`, *optional*):
                # 生成的最大标记数量，默认使用 `generate` 默认值即可。

            generate_kwargs (`Dict`, *optional*):
                # 传递所有这些参数直接发送到 `generate`，允许完全控制此函数。
            timeout (`float`, *optional*, defaults to None):
                # 从网页获取图像的最大等待时间（秒）。如果为 None，则不设置超时，调用可能会一直阻塞。

        Return:
            # 一个列表或一组列表的 `dict`：每个结果都作为包含以下键的字典返回：

            - **generated_text** (`str`) -- 生成的文本。
        """
        # 调用父类的 __call__ 方法，传入图像和其他关键字参数
        return super().__call__(images, **kwargs)
    # 对图像进行预处理，可传入图像、提示文本、超时参数
    def preprocess(self, image, prompt=None, timeout=None):
        # 调用load_image函数加载图像，设置超时参数
        image = load_image(image, timeout=timeout)
        
        # 如果有提示文本
        if prompt is not None:
            # 如果提示文本不是字符串类型，抛出数值错误
            if not isinstance(prompt, str):
                raise ValueError(
                    f"Received an invalid text input, got - {type(prompt)} - but expected a single string. "
                    "Note also that one single text can be provided for conditional image to text generation."
                )
            
            # 获取模型类型
            model_type = self.model.config.model_type

            # 如果模型类型是 "git"
            if model_type == "git":
                # 使用image_processor处理图像，返回张量对象
                model_inputs = self.image_processor(images=image, return_tensors=self.framework)
                # 通过tokenizer将提示文本编码成输入张量id列表
                input_ids = self.tokenizer(text=prompt, add_special_tokens=False).input_ids
                # 添加开头特殊标记的张量id
                input_ids = [self.tokenizer.cls_token_id] + input_ids
                # 转换成张量对象并增加维度
                input_ids = torch.tensor(input_ids).unsqueeze(0)
                model_inputs.update({"input_ids": input_ids})
            
            # 如果模型类型是 "pix2struct"
            elif model_type == "pix2struct":
                # 使用image_processor处理图像，同时传入头部文本参数，返回张量对象
                model_inputs = self.image_processor(images=image, header_text=prompt, return_tensors=self.framework)
            
            # 如果模型类型不是 "vision-encoder-decoder"
            elif model_type != "vision-encoder-decoder":
                # vision-encoder-decoder 不支持条件生成
                # 使用image_processor处理图像，返回张量对象
                model_inputs = self.image_processor(images=image, return_tensors=self.framework)
                # 通过tokenizer将prompt编码成输入张量对象
                text_inputs = self.tokenizer(prompt, return_tensors=self.framework)
                # 更新model_inputs
                model_inputs.update(text_inputs)
            
            # 其他情况
            else:
                # 抛出数值错误，模型类型不支持条件文本生成
                raise ValueError(f"Model type {model_type} does not support conditional text generation")
        
        # 如果没有提示文本
        else:
            # 使用image_processor处��图像，返回张量对象
            model_inputs = self.image_processor(images=image, return_tensors=self.framework)
        
        # 如果模型类型是 "git"，且没有提示文本
        if self.model.config.model_type == "git" and prompt is None:
            # 将input_ids设置为None
            model_inputs["input_ids"] = None
        
        # 返回处理好的模型输入
        return model_inputs
    # 用于模型前向传播，处理模型输入数据和生成参数
    def _forward(self, model_inputs, generate_kwargs=None):
        # 当模型输入中包含"input_ids"且为列表且所有元素均为None时，将其置为None，避免导致前向传播失败
        if (
            "input_ids" in model_inputs
            and isinstance(model_inputs["input_ids"], list)
            and all(x is None for x in model_inputs["input_ids"])
        ):
            model_inputs["input_ids"] = None

        if generate_kwargs is None:
            generate_kwargs = {}
        # FIXME: 在这里需要弹出model_inputs中的内容，因为`generation.py`和`generation.tf_utils.py`中对输入的解析方式有所不同。
        # 在TensorFlow版本中，如果我们不使用`input_ids`，`generate`会引发错误，而PyTorch版本中则将其匹配为`self.model.main_input_name`或`self.model.encoder.main_input_name`。
        inputs = model_inputs.pop(self.model.main_input_name)
        # 通过模型生成方法生成模型输出
        model_outputs = self.model.generate(inputs, **model_inputs, **generate_kwargs)
        return model_outputs

    # 后处理生成的模型输出，生成记录列表
    def postprocess(self, model_outputs):
        records = []
        for output_ids in model_outputs:
            # 将模型生成的token ids解码为文本，去掉特殊标记后添加到记录中
            record = {
                "generated_text": self.tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                )
            }
            records.append(record)
        return records
```