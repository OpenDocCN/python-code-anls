# `.\pipelines\image_to_text.py`

```
# 设置文件编码为utf-8
# 版权声明
#     2024年 HuggingFace Inc. 团队保留所有权利。
# 
# 根据 Apache 许可证 2.0 版（“许可证”）获得许可。
# 您只能在遵守许可证的情况下使用本文件。
# 您可以获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非法律另有规定或书面同意，软件在
# 分发时是按“按现状”分发的，
# 没有任何明示或暗示的保证或条件
# 请参阅特定语言的许可证
# 限制条件
# 
# 加载所需的模块和库
from typing import List, Union

from ..utils import (
    add_end_docstrings,
    is_tf_available,
    is_torch_available,
    is_vision_available,
    logging,
    requires_backends,
)
# 导入 Pipeline 类，构建 Pipeline 初始化参数
from .base import Pipeline, build_pipeline_init_args

# 如果可用视觉处理模块
# 则导入 PIL 图像库和加载图像的函数
if is_vision_available():
    from PIL import Image
    from ..image_utils import load_image

# 如果可用 TensorFlow
# 则导入 TF 模型与视觉到序列映射的名称
if is_tf_available():
    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES

# 如果可用 PyTorch
# 则导入 Torch 模块和模型到序列映射的名称
if is_torch_available():
    import torch
    from ..models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES

# 获取日志记录器
logger = logging.get_logger(__name__)

# 添加文档字符串
@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True, has_image_processor=True))
# 定义一个 ImageToTextPipeline 类，继承自 Pipeline 类
class ImageToTextPipeline(Pipeline):
    """
    图像到文本的 Pipeline，使用 AutoModelForVision2Seq 模型。该 Pipeline 预测给定图像的标题。

    示例：

    ```python
    >>> from transformers import pipeline

    >>> captioner = pipeline(model="ydshieh/vit-gpt2-coco-en")
    >>> captioner("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    [{'generated_text': 'two birds are standing next to each other '}]
    ```

    在[pipeline 教程](../pipeline_tutorial)中了解有关使用 Pipeline 的基础知识

    当前的图像到文本 Pipeline 可以使用下面的任务标识符加载：
    "image-to-text"。

    查看 [huggingface.co/models](https://huggingface.co/models?pipeline_tag=image-to-text)
    上可用模型的列表。
    """

    def __init__(self, *args, **kwargs):
        # 调用父类的构造函数
        super().__init__(*args, **kwargs)
        # 检查后端，确保 vision 模块可用
        requires_backends(self, "vision")
        # 检查模型类型并确定使用 TF 或者 Torch
        self.check_model_type(
            TF_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES if self.framework == "tf" else MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
        )
    # 对输入参数进行清理和预处理，返回两个字典：预处理参数和转发参数
    def _sanitize_parameters(self, max_new_tokens=None, generate_kwargs=None, prompt=None, timeout=None):
        # 初始化两个空字典，用于存储预处理参数和转发参数
        forward_params = {}
        preprocess_params = {}

        # 如果有传入提示文本，则添加到预处理参数字典中
        if prompt is not None:
            preprocess_params["prompt"] = prompt
        # 如果有传入超时时间，则添加到预处理参数字典中
        if timeout is not None:
            preprocess_params["timeout"] = timeout

        # 如果有传入最大新生成标记数，则添加到转发参数字典中
        if max_new_tokens is not None:
            forward_params["max_new_tokens"] = max_new_tokens
        # 如果有传入生成参数字典，并且生成参数字典中没有重复定义最大新生成标记数，则将其合并到转发参数字典中
        if generate_kwargs is not None:
            if max_new_tokens is not None and "max_new_tokens" in generate_kwargs:
                # 如果同时在参数和生成参数中定义了最大新生成标记数，则抛出数值错误异常
                raise ValueError(
                    "`max_new_tokens` is defined both as an argument and inside `generate_kwargs` argument, please use"
                    " only 1 version"
                )
            forward_params.update(generate_kwargs)

        # 返回预处理参数字典、转发参数字典和一个空字典作为补充
        return preprocess_params, forward_params, {}

    # 调用函数，将标签分配给传入的图像或图像列表
    def __call__(self, images: Union[str, List[str], "Image.Image", List["Image.Image"]], **kwargs):
        """
        为传入的图像或图像列表分配标签。

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                该管道处理三种类型的图像：

                - 包含指向图像的HTTP(s)链接的字符串
                - 包含指向图像的本地路径的字符串
                - 直接加载到PIL中的图像

                该管道可以接受单个图像或批量图像。

            max_new_tokens (`int`, *optional*):
                要生成的最大标记数。默认情况下，将使用`generate`的默认值。

            generate_kwargs (`Dict`, *optional*):
                将这些参数直接传递给`generate`，允许对此函数进行完全控制。
                
            timeout (`float`, *optional*, 默认为None):
                从网络获取图像的最大等待时间（以秒为单位）。如果为None，则不设置超时，调用可能会永久阻塞。

        Return:
            `list` 或 `list` 的 `list`，每个结果作为一个字典返回，包含以下键：

            - **generated_text** (`str`) -- 生成的文本。
        """
        # 调用父类的`__call__`方法，传递图像和任意其他关键字参数
        return super().__call__(images, **kwargs)
    # 对输入的图像进行预处理，加载图像并返回处理后的图像对象
    def preprocess(self, image, prompt=None, timeout=None):
        image = load_image(image, timeout=timeout)

        # 如果有文本提示，则检查其是否为字符串类型，否则抛出数值错误异常
        if prompt is not None:
            if not isinstance(prompt, str):
                raise ValueError(
                    f"Received an invalid text input, got - {type(prompt)} - but expected a single string. "
                    "Note also that one single text can be provided for conditional image to text generation."
                )

            # 获取模型的类型
            model_type = self.model.config.model_type

            # 根据模型类型进行不同的处理
            if model_type == "git":
                # 使用图像处理器处理图像，返回张量数据
                model_inputs = self.image_processor(images=image, return_tensors=self.framework)
                # 使用分词器处理文本提示，生成输入的token IDs
                input_ids = self.tokenizer(text=prompt, add_special_tokens=False).input_ids
                # 将CLS token ID添加到输入token序列的开头
                input_ids = [self.tokenizer.cls_token_id] + input_ids
                input_ids = torch.tensor(input_ids).unsqueeze(0)
                model_inputs.update({"input_ids": input_ids})

            elif model_type == "pix2struct":
                # 使用图像处理器处理图像，并将文本提示作为头部文本处理
                model_inputs = self.image_processor(images=image, header_text=prompt, return_tensors=self.framework)

            elif model_type != "vision-encoder-decoder":
                # 对于不支持条件生成的模型类型，使用图像处理器处理图像，同时使用分词器处理文本输入
                model_inputs = self.image_processor(images=image, return_tensors=self.framework)
                text_inputs = self.tokenizer(prompt, return_tensors=self.framework)
                model_inputs.update(text_inputs)

            else:
                # 如果模型类型不在预期的范围内，则抛出数值错误异常
                raise ValueError(f"Model type {model_type} does not support conditional text generation")

        else:
            # 如果没有文本提示，则仅使用图像处理器处理图像
            model_inputs = self.image_processor(images=image, return_tensors=self.framework)

        # 如果模型类型为"git"并且没有文本提示，则将输入的token IDs设为None
        if self.model.config.model_type == "git" and prompt is None:
            model_inputs["input_ids"] = None

        # 返回预处理后的模型输入
        return model_inputs

    # 执行模型推理过程的内部方法，根据模型输入调用生成方法生成模型输出
    def _forward(self, model_inputs, **generate_kwargs):
        # 对于批处理中的空输入token IDs列表，将其设为None，避免导致生成失败
        if (
            "input_ids" in model_inputs
            and isinstance(model_inputs["input_ids"], list)
            and all(x is None for x in model_inputs["input_ids"])
        ):
            model_inputs["input_ids"] = None

        # FIXME: 由于`generation.py`和`generation.tf_utils.py`中输入解析的差异，需要在此处弹出特定的输入键
        # 在TensorFlow版本中，如果不使用`input_ids`，则会导致生成方法抛出错误，而在PyTorch版本中，会使用
        # `self.model.main_input_name`或`self.model.encoder.main_input_name`作为输入名称来匹配。
        inputs = model_inputs.pop(self.model.main_input_name)
        # 调用模型的生成方法，生成模型输出
        model_outputs = self.model.generate(inputs, **model_inputs, **generate_kwargs)
        # 返回模型输出
        return model_outputs
    # 定义一个方法用于后处理模型输出的结果
    def postprocess(self, model_outputs):
        # 初始化一个空列表用于存储处理后的记录
        records = []
        # 遍历模型输出中的每个输出 ID 列表
        for output_ids in model_outputs:
            # 解码当前输出 ID 列表，跳过特殊标记，生成文本
            record = {
                "generated_text": self.tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                )
            }
            # 将生成的记录添加到记录列表中
            records.append(record)
        # 返回所有处理后的记录列表
        return records
```