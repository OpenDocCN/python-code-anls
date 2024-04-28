# `.\transformers\pipelines\image_classification.py`

```py
# 从 typing 模块中导入 List 和 Union 类型
from typing import List, Union

# 从 numpy 模块中导入 np 别名
import numpy as np

# 从 ..utils 模块中导入一些函数和类
from ..utils import (
    ExplicitEnum,
    add_end_docstrings,
    is_tf_available,
    is_torch_available,
    is_vision_available,
    logging,
    requires_backends,
)
# 从 .base 模块中导入 PIPELINE_INIT_ARGS 和 Pipeline 类
from .base import PIPELINE_INIT_ARGS, Pipeline

# 如果 is_vision_available() 返回 True 则执行以下语句
if is_vision_available():
    # 从 PIL 模块中导入 Image 类
    from PIL import Image

    # 从 ..image_utils 模块中导入 load_image 函数
    from ..image_utils import load_image

# 如果 is_tf_available() 返回 True 则执行以下语句
if is_tf_available():
    # 从 ..models.auto.modeling_tf_auto 模块中导入 TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES 类
    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES

# 如果 is_torch_available() 返回 True 则执行以下语句
if is_torch_available():
    # 从 ..models.auto.modeling_auto 模块中导入 MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES 类
    from ..models.auto.modeling_auto import MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES

# 创建一个名为 logger 的日志记录器
logger = logging.get_logger(__name__)

# 定义一个名为 sigmoid 的函数，接受一个参数 _outputs
def sigmoid(_outputs):
    # 返回 1.0 除以 (1.0 加上 _outputs 的负数指数)
    return 1.0 / (1.0 + np.exp(-_outputs))

# 定义一个名为 softmax 的函数，接受一个参数 _outputs
def softmax(_outputs):
    # 计算 _outputs 沿着最后一个维度的最大值，保持其维度
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    # 计算 _outputs 减去 maxes 的指数
    shifted_exp = np.exp(_outputs - maxes)
    # 返回 shifted_exp 除以 shifted_exp 沿着最后一个维度的和，保持其维度
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

# 定义一个名为 ClassificationFunction 的类继承自 ExplicitEnum 类
class ClassificationFunction(ExplicitEnum):
    # 定义枚举值 SIGMOID 为 "sigmoid"
    SIGMOID = "sigmoid"
    # 定义枚举值 SOFTMAX 为 "softmax"
    SOFTMAX = "softmax"
    # 定义枚举值 NONE 为 "none"
    NONE = "none"

# 使用装饰器 add_end_docstrings 对 ImageClassificationPipeline 进行注释
@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        function_to_apply (`str`, *optional*, defaults to `"default"`):
            The function to apply to the model outputs in order to retrieve the scores. Accepts four different values:

            - `"default"`: if the model has a single label, will apply the sigmoid function on the output. If the model
              has several labels, will apply the softmax function on the output.
            - `"sigmoid"`: Applies the sigmoid function on the output.
            - `"softmax"`: Applies the softmax function on the output.
            - `"none"`: Does not apply any function on the output.
    """,
)
# 定义一个名为 ImageClassificationPipeline 的类继承自 Pipeline 类
class ImageClassificationPipeline(Pipeline):
    """
    Image classification pipeline using any `AutoModelForImageClassification`. This pipeline predicts the class of an
    image.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> classifier = pipeline(model="microsoft/beit-base-patch16-224-pt22k-ft22k")
    >>> classifier("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    [{'score': 0.442, 'label': 'macaw'}, {'score': 0.088, 'label': 'popinjay'}, {'score': 0.075, 'label': 'parrot'}, {'score': 0.073, 'label': 'parodist, lampooner'}, {'score': 0.046, 'label': 'poll, poll_parrot'}]
    ```py

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This image classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"image-classification"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=image-classification).
    """
    # 定义一个默认值为ClassificationFunction.NONE的function_to_apply变量
    function_to_apply: ClassificationFunction = ClassificationFunction.NONE
    
    # 初始化函数，调用父类的初始化方法，并检查是否需要vision后端支持
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        requires_backends(self, "vision")
        # 检查模型类型，根据框架选择相应的模型名称映射
        self.check_model_type(
            TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES
            if self.framework == "tf"
            else MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES
        )
    
    # 清理参数方法，处理top_k, function_to_apply, timeout参数
    def _sanitize_parameters(self, top_k=None, function_to_apply=None, timeout=None):
        # 预处理参数字典
        preprocess_params = {}
        # 如果传入了timeout参数，添加到预处理参数字典中
        if timeout is not None:
            preprocess_params["timeout"] = timeout
        # 后处理参数字典
        postprocess_params = {}
        # 如果传入了top_k参数，添加到后处理参数字典中
        if top_k is not None:
            postprocess_params["top_k"] = top_k
        # 如果function_to_apply是字符串，转换为ClassificationFunction枚举类型
        if isinstance(function_to_apply, str):
            function_to_apply = ClassificationFunction(function_to_apply.lower())
        # 如果function_to_apply不为None，添加到后处理参数字典中
        if function_to_apply is not None:
            postprocess_params["function_to_apply"] = function_to_apply
        # 返回预处理参数字典，空字典，后处理参数字典
        return preprocess_params, {}, postprocess_params
    # 定义一个方法，用来为输入的图像分配标签
    def __call__(self, images: Union[str, List[str], "Image.Image", List["Image.Image"]], **kwargs):
        """
        为输入的图像分配标签。

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                表示三种类型的图像：

                - 包含指向图像的 http 链接的字符串
                - 包含指向图像的本地路径的字符串
                - 直接在 PIL 中加载的图像

                该流程接受单个图像或一批图像，必须以字符串形式传递。
                批量图像必须都是相同格式的：全部为 http 链接、全部为本地路径或全部为 PIL 图像。
            function_to_apply (`str`, *optional*, defaults to `"default"`):
                用于从模型输出中检索分数的函数。有四个不同的值可接受：

                如果未指定此参数，那么它将根据标签数量应用以下函数：

                - 如果模型有单个标签，则对输出应用 sigmoid 函数。
                - 如果模型有多个标签，则对输出应用 softmax 函数。

                可能的值有：

                - `"sigmoid"`: 对输出应用 sigmoid 函数。
                - `"softmax"`: 对输出应用 softmax 函数。
                - `"none"`: 不对输出应用任何函数。
            top_k (`int`, *optional*, defaults to 5):
                将由流程返回的前 k 个标签的数量。如果提供的数字高于模型配置中标签的数量，则默认为标签数量。
            timeout (`float`, *optional*, defaults to None):
                等待从网络获取图像的最长时间，单位为秒。如果为 None，则不设置超时，调用可能会一直阻塞。

        Return:
            包含结果的字典或字典列表。如果输入是单张图像，则返回一个字典；如果输入是多张图像，则返回与图像对应的字典列表。

            字典包含以下键：

            - **label** (`str`) -- 模型识别出的标签。
            - **score** (`int`) -- 模型为该标签分配的分数。
        """
        return super().__call__(images, **kwargs)

    # 预处理图像，加载图像并将其转换为模型的输入格式
    def preprocess(self, image, timeout=None):
        image = load_image(image, timeout=timeout)
        model_inputs = self.image_processor(images=image, return_tensors=self.framework)
        return model_inputs

    # 前向传播，将模型输入传递给模型并获取输出
    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        return model_outputs
    # 后处理函数，用于处理模型输出
    def postprocess(self, model_outputs, function_to_apply=None, top_k=5):
        # 如果未指定应用的函数，则根据模型配置确定
        if function_to_apply is None:
            # 如果问题类型是多标签分类或者标签数量为1，则应用 sigmoid 函数
            if self.model.config.problem_type == "multi_label_classification" or self.model.config.num_labels == 1:
                function_to_apply = ClassificationFunction.SIGMOID
            # 如果问题类型是单标签分类或者标签数量大于1，则应用 softmax 函数
            elif self.model.config.problem_type == "single_label_classification" or self.model.config.num_labels > 1:
                function_to_apply = ClassificationFunction.SOFTMAX
            # 如果模型配置中有指定应用的函数但未指定具体函数，则使用配置中的函数
            elif hasattr(self.model.config, "function_to_apply") and function_to_apply is None:
                function_to_apply = self.model.config.function_to_apply
            # 否则，不应用任何函数
            else:
                function_to_apply = ClassificationFunction.NONE

        # 如果 top_k 大于标签数量，则将其设为标签数量
        if top_k > self.model.config.num_labels:
            top_k = self.model.config.num_labels

        # 获取模型输出中的 logits，并转换为 NumPy 数组
        outputs = model_outputs["logits"][0]
        outputs = outputs.numpy()

        # 根据指定的函数对输出进行处理
        if function_to_apply == ClassificationFunction.SIGMOID:
            scores = sigmoid(outputs)
        elif function_to_apply == ClassificationFunction.SOFTMAX:
            scores = softmax(outputs)
        elif function_to_apply == ClassificationFunction.NONE:
            scores = outputs
        else:
            # 如果指定的函数不在预定义的函数中，则引发 ValueError 异常
            raise ValueError(f"Unrecognized `function_to_apply` argument: {function_to_apply}")

        # 构建标签和得分的字典列表
        dict_scores = [
            {"label": self.model.config.id2label[i], "score": score.item()} for i, score in enumerate(scores)
        ]
        # 根据得分降序排序
        dict_scores.sort(key=lambda x: x["score"], reverse=True)
        # 如果指定了 top_k，则保留前 top_k 个结果
        if top_k is not None:
            dict_scores = dict_scores[:top_k]

        # 返回结果字典列表
        return dict_scores
```