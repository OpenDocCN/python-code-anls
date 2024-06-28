# `.\pipelines\image_classification.py`

```py
# 引入必要的类型和模块
from typing import List, Union
import numpy as np

# 引入一些自定义工具函数和类
from ..utils import (
    ExplicitEnum,
    add_end_docstrings,
    is_tf_available,
    is_torch_available,
    is_vision_available,
    logging,
    requires_backends,
)

# 导入基础的管道类和初始化函数
from .base import Pipeline, build_pipeline_init_args

# 如果当前环境支持视觉处理，则导入图像处理相关的模块
if is_vision_available():
    from PIL import Image
    from ..image_utils import load_image

# 如果当前环境支持 TensorFlow，则导入相关的模块
if is_tf_available():
    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES

# 如果当前环境支持 PyTorch，则导入相关的模块
if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 定义 sigmoid 函数，用于计算 sigmoid 激活函数的输出
def sigmoid(_outputs):
    return 1.0 / (1.0 + np.exp(-_outputs))

# 定义 softmax 函数，用于计算 softmax 激活函数的输出
def softmax(_outputs):
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

# 定义分类函数的枚举类，包括 sigmoid、softmax 和 none 三种选择
class ClassificationFunction(ExplicitEnum):
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    NONE = "none"

# 添加文档注释，描述了初始化图像分类管道的参数和功能
@add_end_docstrings(
    build_pipeline_init_args(has_image_processor=True),
    r"""
        function_to_apply (`str`, *optional*, defaults to `"default"`):
            The function to apply to the model outputs in order to retrieve the scores. Accepts four different values:

            - `"default"`: if the model has a single label, will apply the sigmoid function on the output. If the model
              has several labels, will apply the softmax function on the output.
            - `"sigmoid"`: Applies the sigmoid function on the output.
            - `"softmax"`: Applies the softmax function on the output.
            - `"none"`: Does not apply any function on the output.""",
)
# 定义图像分类管道类，继承自 Pipeline 类
class ImageClassificationPipeline(Pipeline):
    """
    Image classification pipeline using any `AutoModelForImageClassification`. This pipeline predicts the class of an
    image.

    Example:

    ```
    >>> from transformers import pipeline

    >>> classifier = pipeline(model="microsoft/beit-base-patch16-224-pt22k-ft22k")
    >>> classifier("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    [{'score': 0.442, 'label': 'macaw'}, {'score': 0.088, 'label': 'popinjay'}, {'score': 0.075, 'label': 'parrot'}, {'score': 0.073, 'label': 'parodist, lampooner'}, {'score': 0.046, 'label': 'poll, poll_parrot'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This image classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"image-classification"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=image-classification).
    """
    function_to_apply: ClassificationFunction = ClassificationFunction.NONE

# 初始化一个属性 `function_to_apply`，默认为 `ClassificationFunction.NONE`，表示没有指定分类函数。


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        requires_backends(self, "vision")
        self.check_model_type(
            TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES
            if self.framework == "tf"
            else MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES
        )

# 类的初始化方法，接受任意位置和关键字参数。调用父类的初始化方法，并确保当前对象需要 "vision" 后端支持。根据当前框架（"tf" 或其他），检查模型类型是否符合预期。


    def _sanitize_parameters(self, top_k=None, function_to_apply=None, timeout=None):
        preprocess_params = {}
        if timeout is not None:
            preprocess_params["timeout"] = timeout
        postprocess_params = {}
        if top_k is not None:
            postprocess_params["top_k"] = top_k
        if isinstance(function_to_apply, str):
            function_to_apply = ClassificationFunction(function_to_apply.lower())
        if function_to_apply is not None:
            postprocess_params["function_to_apply"] = function_to_apply
        return preprocess_params, {}, postprocess_params

# 方法 `_sanitize_parameters`，用于处理和清理参数。根据传入的参数，构建预处理参数和后处理参数字典。如果传入了 `timeout` 参数，则将其加入 `preprocess_params` 字典。如果传入了 `top_k` 参数，则将其加入 `postprocess_params` 字典。如果 `function_to_apply` 是字符串类型，则将其转换为 `ClassificationFunction` 枚举类型。最后，如果 `function_to_apply` 不为 None，则将其加入 `postprocess_params` 字典。最终返回三个空字典构成的元组 `(preprocess_params, {}, postprocess_params)`。
    # 继承父类的 __call__ 方法，用于给传入的图像（或图像列表）分配标签
    def __call__(self, images: Union[str, List[str], "Image.Image", List["Image.Image"]], **kwargs):
        """
        Assign labels to the image(s) passed as inputs.

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images, which must then be passed as a string.
                Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL
                images.
            function_to_apply (`str`, *optional*, defaults to `"default"`):
                The function to apply to the model outputs in order to retrieve the scores. Accepts four different
                values:

                If this argument is not specified, then it will apply the following functions according to the number
                of labels:

                - If the model has a single label, will apply the sigmoid function on the output.
                - If the model has several labels, will apply the softmax function on the output.

                Possible values are:

                - `"sigmoid"`: Applies the sigmoid function on the output.
                - `"softmax"`: Applies the softmax function on the output.
                - `"none"`: Does not apply any function on the output.
            top_k (`int`, *optional*, defaults to 5):
                The number of top labels that will be returned by the pipeline. If the provided number is higher than
                the number of labels available in the model configuration, it will default to the number of labels.
            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
                the call may block forever.

        Return:
            A dictionary or a list of dictionaries containing result. If the input is a single image, will return a
            dictionary, if the input is a list of several images, will return a list of dictionaries corresponding to
            the images.

            The dictionaries contain the following keys:

            - **label** (`str`) -- The label identified by the model.
            - **score** (`int`) -- The score attributed by the model for that label.
        """
        # 调用父类的 __call__ 方法，将图像（或图像列表）和其他关键字参数传递给父类
        return super().__call__(images, **kwargs)

    # 图像预处理方法，加载图像并进行预处理
    def preprocess(self, image, timeout=None):
        # 调用 load_image 方法加载图像，设置超时时间
        image = load_image(image, timeout=timeout)
        # 使用 image_processor 对象处理图像，将处理结果返回作为模型输入
        model_inputs = self.image_processor(images=image, return_tensors=self.framework)
        return model_inputs

    # 模型前向传播方法，接收模型输入并返回模型输出
    def _forward(self, model_inputs):
        # 使用模型对象处理模型输入，得到模型输出
        model_outputs = self.model(**model_inputs)
        return model_outputs
    # 对模型输出进行后处理，根据给定的函数应用规则或默认选择
    def postprocess(self, model_outputs, function_to_apply=None, top_k=5):
        # 如果未提供特定的函数应用规则，则根据模型配置选择默认规则
        if function_to_apply is None:
            if self.model.config.problem_type == "multi_label_classification" or self.model.config.num_labels == 1:
                function_to_apply = ClassificationFunction.SIGMOID  # 使用 sigmoid 函数进行分类
            elif self.model.config.problem_type == "single_label_classification" or self.model.config.num_labels > 1:
                function_to_apply = ClassificationFunction.SOFTMAX  # 使用 softmax 函数进行分类
            elif hasattr(self.model.config, "function_to_apply") and function_to_apply is None:
                function_to_apply = self.model.config.function_to_apply  # 使用配置中指定的函数
            else:
                function_to_apply = ClassificationFunction.NONE  # 不应用特定的函数

        # 如果 top_k 超过可用标签数量，则将其限制为标签数量
        if top_k > self.model.config.num_labels:
            top_k = self.model.config.num_labels

        # 从模型输出中提取 logits 并转换为 NumPy 数组
        outputs = model_outputs["logits"][0]
        outputs = outputs.numpy()

        # 根据选择的函数应用规则对 logits 进行处理
        if function_to_apply == ClassificationFunction.SIGMOID:
            scores = sigmoid(outputs)  # 应用 sigmoid 函数
        elif function_to_apply == ClassificationFunction.SOFTMAX:
            scores = softmax(outputs)  # 应用 softmax 函数
        elif function_to_apply == ClassificationFunction.NONE:
            scores = outputs  # 不应用额外的函数，直接使用 logits
        else:
            raise ValueError(f"Unrecognized `function_to_apply` argument: {function_to_apply}")  # 抛出异常，指示无法识别的函数应用规则

        # 将得分与标签对应，并按得分降序排序
        dict_scores = [
            {"label": self.model.config.id2label[i], "score": score.item()} for i, score in enumerate(scores)
        ]
        dict_scores.sort(key=lambda x: x["score"], reverse=True)

        # 如果指定了 top_k，则仅保留前 top_k 个条目
        if top_k is not None:
            dict_scores = dict_scores[:top_k]

        # 返回包含标签及其得分的字典列表
        return dict_scores
```