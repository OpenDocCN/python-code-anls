# `.\transformers\pipelines\zero_shot_image_classification.py`

```
# 从 collections 模块导入 UserDict 类
from collections import UserDict
# 从 typing 模块导入 List 和 Union 类型
from typing import List, Union

# 从 ..utils 模块导入各种函数和对象
from ..utils import (
    add_end_docstrings,  # 导入函数 add_end_docstrings
    is_tf_available,  # 导入函数 is_tf_available
    is_torch_available,  # 导入函数 is_torch_available
    is_vision_available,  # 导入函数 is_vision_available
    logging,  # 导入 logging 对象
    requires_backends,  # 导入函数 requires_backends
)
# 从 .base 模块导入 PIPELINE_INIT_ARGS 和 Pipeline 类
from .base import PIPELINE_INIT_ARGS, Pipeline


# 如果 vision 可用
if is_vision_available():
    # 从 PIL 模块导入 Image 类
    from PIL import Image
    # 从 ..image_utils 模块导入 load_image 函数
    from ..image_utils import load_image

# 如果 torch 可用
if is_torch_available():
    # 导入 torch 模块
    import torch
    # 从 ..models.auto.modeling_auto 模块导入 MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES 对象
    from ..models.auto.modeling_auto import MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES

# 如果 tensorflow 可用
if is_tf_available():
    # 从 ..models.auto.modeling_tf_auto 模块导入 TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES 对象
    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES
    # 从 ..tf_utils 模块导入 stable_softmax 函数
    from ..tf_utils import stable_softmax

# 获取 logger 对象
logger = logging.get_logger(__name__)


# 使用装饰器 @add_end_docstrings(PIPELINE_INIT_ARGS) 为 ZeroShotImageClassificationPipeline 类添加文档字符串
@add_end_docstrings(PIPELINE_INIT_ARGS)
# 定义 ZeroShotImageClassificationPipeline 类，继承自 Pipeline 类
class ZeroShotImageClassificationPipeline(Pipeline):
    """
    Zero shot image classification pipeline using `CLIPModel`. This pipeline predicts the class of an image when you
    provide an image and a set of `candidate_labels`.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> classifier = pipeline(model="openai/clip-vit-large-patch14")
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

    # 定义初始化方法
    def __init__(self, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 检查是否需要 vision 后端支持
        requires_backends(self, "vision")
        # 检查模型类型，如果使用 TensorFlow 则检查 TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES，否则检查 MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES
        self.check_model_type(
            TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES
            if self.framework == "tf"
            else MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES
        )
    def __call__(self, images: Union[str, List[str], "Image", List["Image"]], **kwargs):
        """
        通过传入的图片对其进行标记。
    
        参数:
            images (`str`, `List[str]`, `PIL.Image` 或 `List[PIL.Image]`):
                该流水线处理三种类型的图片:
    
                - 包含指向图像的 HTTP 链接的字符串
                - 包含指向本地图像的本地路径的字符串
                - 直接在 PIL 中加载的图像
    
            candidate_labels (`List[str]`):
                该图片的候选标签
    
            hypothesis_template (`str`, *可选的*, 默认值为 `"This is a photo of {}"`):
                与 *candidate_labels* 一同使用的句子，通过替换占位符(candidate_labels)尝试对图片进行分类。
                然后使用 logits_per_image 来估计可能性。
    
            timeout (`float`, *可选的*, 默认值为 None):
                从 Web 中获取图片的最大等待时间（秒）。如果为 None，则不设置超时，调用可能会一直阻塞。
    
        返回:
            包含结果的字典列表，每个提议标签一个字典。字典包含以下键:
    
            - **label** (`str`) -- 模型识别的标签。它是建议的 `candidate_label` 之一。
            - **score** (`float`) -- 该标签被模型分配的得分（取值范围为 0-1）。
        """
        return super().__call__(images, **kwargs)
    
    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        如果 "candidate_labels" 存在于 kwargs 中:
            将 kwargs["candidate_labels"] 存储到 preprocess_params["candidate_labels"] 中
        如果 "timeout" 存在于 kwargs 中:
            将 kwargs["timeout"] 存储到 preprocess_params["timeout"] 中
        如果 "hypothesis_template" 存在于 kwargs 中:
            将 kwargs["hypothesis_template"] 存储到 preprocess_params["hypothesis_template"] 中
    
        返回 preprocess_params, {}, {}
    
    def preprocess(self, image, candidate_labels=None, hypothesis_template="This is a photo of {}.", timeout=None):
        将图像加载为 image，并设置超时时间 timeout
        image = load_image(image, timeout=timeout)
        根据框架的类型，将 image 转换为 tensor
        inputs  = self.image_processor(images=[image], return_tensors=self.framework)
        将 candidate_labels 存储到 inputs 中的 "candidate_labels" 键中
        inputs["candidate_labels"] = candidate_labels
        根据 candidate_labels，使用 hypothesis_template 格式化为句子 sequences
        sequences = [hypothesis_template.format(x) for x in candidate_labels]
        根据模型的类型，确定填充方式
        如果模型的配置中的 model_type 为 "siglip"，则填充为 "max_length"，否则为 True
        padding = "max_length" if self.model.config.model_type == "siglip" else True
        将 sequences 转换为 tensor
        text_inputs = self.tokenizer(sequences, return_tensors=self.framework, padding=padding)
        在 inputs 中存储 text_inputs
        inputs["text_inputs"] = [text_inputs]
        返回 inputs
    # 将候选标签从输入中取出
    candidate_labels = model_inputs.pop("candidate_labels")
    # 将文本输入从输入中取出
    text_inputs = model_inputs.pop("text_inputs")
    # 如果文本输入是 UserDict 类型的实例，就将其赋值给 text_inputs
    if isinstance(text_inputs[0], UserDict):
        text_inputs = text_inputs[0]
    else:
        # 批处理情况
        # 将 text_inputs 解包成单个文本输入
        text_inputs = text_inputs[0][0]

        # 使用模型进行预测，得到输出
        outputs = self.model(**text_inputs, **model_inputs)

        # 组装模型输出结果
        model_outputs = {
            "candidate_labels": candidate_labels,  # 候选标签
            "logits": outputs.logits_per_image,  # 逻辑回归结果
        }
        # 返回模型输出结果
        return model_outputs

    # 后处理模型输出结果
    def postprocess(self, model_outputs):
        # 从模型输出结果中取出候选标签
        candidate_labels = model_outputs.pop("candidate_labels")
        # 从模型输出结果中取出逻辑回归结果
        logits = model_outputs["logits"][0]
        
        # 根据框架和模型类型对逻辑回归结果进行处理
        if self.framework == "pt" and self.model.config.model_type == "siglip":
            # 对逻辑回归结果进行sigmoid处理，并压缩维度
            probs = torch.sigmoid(logits).squeeze(-1)
            # 将概率转换为列表
            scores = probs.tolist()
            # 如果 scores 不是列表类型，将其转换为列表
            if not isinstance(scores, list):
                scores = [scores]
        elif self.framework == "pt":
            # 对逻辑回归结果进行softmax处理，并压缩维度
            probs = logits.softmax(dim=-1).squeeze(-1)
            # 将概率转换为列表
            scores = probs.tolist()
            # 如果 scores 不是列表类型，将其转换为列表
            if not isinstance(scores, list):
                scores = [scores]
        elif self.framework == "tf":
            # 对逻辑回归结果进行softmax处理，使用 stable_softmax 函数，将结果转换为numpy数组再转换为列表
            probs = stable_softmax(logits, axis=-1)
            scores = probs.numpy().tolist()
        else:
            # 如果框架不支持，抛出异常
            raise ValueError(f"Unsupported framework: {self.framework}")

        # 根据得分和候选标签排序，并构造结果列表
        result = [
            {"score": score, "label": candidate_label}
            for score, candidate_label in sorted(zip(scores, candidate_labels), key=lambda x: -x[0])
        ]
        # 返回结果
        return result
```