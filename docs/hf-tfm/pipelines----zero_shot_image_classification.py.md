# `.\pipelines\zero_shot_image_classification.py`

```py
# 导入必要的模块和函数
from collections import UserDict  # 导入UserDict用于创建自定义字典
from typing import List, Union  # 导入List和Union用于类型提示

# 从上级目录的utils模块导入各种函数和类
from ..utils import (
    add_end_docstrings,  # 导入函数add_end_docstrings，用于添加文档字符串
    is_tf_available,  # 导入函数is_tf_available，检查是否可以使用TensorFlow
    is_torch_available,  # 导入函数is_torch_available，检查是否可以使用PyTorch
    is_vision_available,  # 导入函数is_vision_available，检查是否可以使用视觉处理功能
    logging,  # 导入logging模块，用于日志记录
    requires_backends,  # 导入requires_backends函数，用于检查后端依赖
)

# 从当前目录的base模块导入Pipeline类和build_pipeline_init_args函数
from .base import Pipeline, build_pipeline_init_args

# 如果可以使用视觉处理功能
if is_vision_available():
    # 从PIL库中导入Image模块，用于处理图像
    from PIL import Image
    # 从image_utils模块导入load_image函数，用于加载图像数据

# 如果可以使用PyTorch
if is_torch_available():
    # 导入torch库，用于深度学习任务
    import torch
    # 从models.auto模块导入模型映射名称字典

# 如果可以使用TensorFlow
if is_tf_available():
    # 从models.auto模块导入TensorFlow相关的模型映射名称字典
    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES
    # 从tf_utils模块导入稳定的softmax函数，用于概率计算

# 获取当前模块的日志记录器对象
logger = logging.get_logger(__name__)

# 使用装饰器add_end_docstrings为ZeroShotImageClassificationPipeline类添加文档字符串
@add_end_docstrings(build_pipeline_init_args(has_image_processor=True))
class ZeroShotImageClassificationPipeline(Pipeline):
    """
    Zero shot image classification pipeline using `CLIPModel`. This pipeline predicts the class of an image when you
    provide an image and a set of `candidate_labels`.

    Example:

    ```
    >>> from transformers import pipeline

    >>> classifier = pipeline(model="google/siglip-so400m-patch14-384")
    >>> classifier(
    ...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
    ...     candidate_labels=["animals", "humans", "landscape"],
    ... )
    [{'score': 0.965, 'label': 'animals'}, {'score': 0.03, 'label': 'humans'}, {'score': 0.005, 'label': 'landscape'}]

    >>> classifier(
    ...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
    ...     candidate_labels=["black and white", "photorealist", "painting"],
    ... )
    [{'score': 0.996, 'label': 'black and white'}, {'score': 0.003, 'label': 'photorealist'}, {'score': 0.0, 'label': 'painting'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This image classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"zero-shot-image-classification"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=zero-shot-image-classification).
    """

    # 初始化函数，继承自Pipeline类
    def __init__(self, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 检查当前实例是否满足视觉后端的依赖
        requires_backends(self, "vision")
        
        # 根据当前框架选择适当的模型映射名称字典，用于后续任务
        self.check_model_type(
            TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES
            if self.framework == "tf"
            else MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES
        )
    def __call__(self, images: Union[str, List[str], "Image", List["Image"]], **kwargs):
        """
        将标签分配给作为输入传递的图像。

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                处理三种类型的图像：

                - 包含指向图像的 http 链接的字符串
                - 包含指向本地图像路径的字符串
                - 直接加载到 PIL 中的图像

            candidate_labels (`List[str]`):
                此图像的候选标签列表

            hypothesis_template (`str`, *可选*, 默认为 `"This is a photo of {}"`):
                与 *candidate_labels* 结合使用的句子，通过将占位符替换为 candidate_labels 尝试图像分类。
                然后使用 logits_per_image 估算可能性。

            timeout (`float`, *可选*, 默认为 None):
                从网络获取图像的最长等待时间（以秒为单位）。如果为 None，则不设置超时，调用可能会永远阻塞。

        Return:
            包含结果的字典列表，每个提议的标签一个字典。字典包含以下键：

            - **label** (`str`) -- 模型识别的标签之一。它是建议的 `candidate_label` 之一。
            - **score** (`float`) -- 模型为该标签分配的分数（介于0和1之间）。
        """
        return super().__call__(images, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        if "candidate_labels" in kwargs:
            preprocess_params["candidate_labels"] = kwargs["candidate_labels"]
        if "timeout" in kwargs:
            preprocess_params["timeout"] = kwargs["timeout"]
        if "hypothesis_template" in kwargs:
            preprocess_params["hypothesis_template"] = kwargs["hypothesis_template"]

        return preprocess_params, {}, {}

    def preprocess(self, image, candidate_labels=None, hypothesis_template="This is a photo of {}.", timeout=None):
        """
        预处理图像及其相关参数。

        Args:
            image: 图像数据
            candidate_labels (`List[str]`, optional): 图像的候选标签
            hypothesis_template (`str`, optional, defaults to `"This is a photo of {}."`):
                用于替换占位符生成假设句子的模板
            timeout (`float`, optional): 从网络获取图像的最长等待时间（以秒为单位）

        Returns:
            inputs: 包含预处理后数据的字典
        """
        image = load_image(image, timeout=timeout)  # 加载图像数据
        inputs = self.image_processor(images=[image], return_tensors=self.framework)  # 处理图像数据
        inputs["candidate_labels"] = candidate_labels  # 设置候选标签
        sequences = [hypothesis_template.format(x) for x in candidate_labels]  # 根据模板生成假设句子序列
        padding = "max_length" if self.model.config.model_type == "siglip" else True  # 根据模型类型设置填充方式
        text_inputs = self.tokenizer(sequences, return_tensors=self.framework, padding=padding)  # 对假设句子序列进行tokenize
        inputs["text_inputs"] = [text_inputs]  # 设置文本输入
        return inputs
    # 定义一个方法用于模型推断，接收模型输入
    def _forward(self, model_inputs):
        # 弹出输入中的候选标签
        candidate_labels = model_inputs.pop("candidate_labels")
        # 弹出输入中的文本数据
        text_inputs = model_inputs.pop("text_inputs")
        
        # 如果文本输入的第一个元素是 UserDict 类型的对象
        if isinstance(text_inputs[0], UserDict):
            # 将文本输入重新赋值为第一个元素（UserDict对象）
            text_inputs = text_inputs[0]
        else:
            # 如果不是 UserDict 对象，则为批处理情况，取第一个元素的第一个元素
            # （这里假设 text_inputs 是一个二重嵌套列表，第一个元素是批处理的列表）
            text_inputs = text_inputs[0][0]

        # 使用模型进行推断，传入文本输入和模型输入
        outputs = self.model(**text_inputs, **model_inputs)

        # 构建模型输出字典，包括候选标签和模型的 logits
        model_outputs = {
            "candidate_labels": candidate_labels,
            "logits": outputs.logits_per_image,
        }
        return model_outputs

    # 定义一个方法用于后处理模型输出
    def postprocess(self, model_outputs):
        # 弹出模型输出中的候选标签
        candidate_labels = model_outputs.pop("candidate_labels")
        # 取出 logits，并在第一个维度上进行压缩，即去除维度为1的维度
        logits = model_outputs["logits"][0]

        # 根据不同的框架和模型类型进行处理概率
        if self.framework == "pt" and self.model.config.model_type == "siglip":
            # 对 logits 应用 sigmoid 函数，并在最后一个维度上进行压缩
            probs = torch.sigmoid(logits).squeeze(-1)
            # 将概率转换为列表
            scores = probs.tolist()
            # 如果 scores 不是列表，则转换为列表
            if not isinstance(scores, list):
                scores = [scores]
        elif self.framework == "pt":
            # 对 logits 应用 softmax 函数，并在最后一个维度上进行压缩
            probs = logits.softmax(dim=-1).squeeze(-1)
            # 将概率转换为列表
            scores = probs.tolist()
            # 如果 scores 不是列表，则转换为列表
            if not isinstance(scores, list):
                scores = [scores]
        elif self.framework == "tf":
            # 对 logits 应用稳定的 softmax 函数，并在最后一个维度上进行处理
            probs = stable_softmax(logits, axis=-1)
            # 将概率转换为 numpy 数组，再转换为列表
            scores = probs.numpy().tolist()
        else:
            # 如果框架不支持，则引发异常
            raise ValueError(f"Unsupported framework: {self.framework}")

        # 将概率分数与候选标签组成字典列表，并按分数降序排列
        result = [
            {"score": score, "label": candidate_label}
            for score, candidate_label in sorted(zip(scores, candidate_labels), key=lambda x: -x[0])
        ]
        return result
```