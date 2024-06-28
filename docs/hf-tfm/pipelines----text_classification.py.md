# `.\pipelines\text_classification.py`

```
import inspect  # 导入inspect模块，用于获取对象信息
import warnings  # 导入warnings模块，用于处理警告信息
from typing import Dict  # 从typing模块导入Dict类型提示

import numpy as np  # 导入NumPy库，用于数值计算

from ..utils import ExplicitEnum, add_end_docstrings, is_tf_available, is_torch_available  # 导入自定义模块和函数
from .base import GenericTensor, Pipeline, build_pipeline_init_args  # 从本地模块导入指定类和函数

if is_tf_available():  # 如果TensorFlow可用，则导入相关模型映射名称
    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES

if is_torch_available():  # 如果PyTorch可用，则导入相关模型映射名称
    from ..models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES


def sigmoid(_outputs):  # 定义sigmoid函数，接受一个参数_outputs
    return 1.0 / (1.0 + np.exp(-_outputs))  # 返回sigmoid函数的计算结果


def softmax(_outputs):  # 定义softmax函数，接受一个参数_outputs
    maxes = np.max(_outputs, axis=-1, keepdims=True)  # 计算_outputs在最后一个轴上的最大值，并保持维度
    shifted_exp = np.exp(_outputs - maxes)  # 计算_outputs减去最大值后的指数值
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)  # 返回softmax归一化后的结果


class ClassificationFunction(ExplicitEnum):  # 定义一个枚举类ClassificationFunction
    SIGMOID = "sigmoid"  # 枚举项：sigmoid
    SOFTMAX = "softmax"  # 枚举项：softmax
    NONE = "none"  # 枚举项：none


@add_end_docstrings(  # 使用add_end_docstrings装饰器，添加文档字符串
    build_pipeline_init_args(has_tokenizer=True),  # 调用build_pipeline_init_args函数生成初始化参数文档
    r"""
        return_all_scores (`bool`, *optional*, defaults to `False`):
            Whether to return all prediction scores or just the one of the predicted class.
        function_to_apply (`str`, *optional*, defaults to `"default"`):
            The function to apply to the model outputs in order to retrieve the scores. Accepts four different values:

            - `"default"`: if the model has a single label, will apply the sigmoid function on the output. If the model
              has several labels, will apply the softmax function on the output.
            - `"sigmoid"`: Applies the sigmoid function on the output.
            - `"softmax"`: Applies the softmax function on the output.
            - `"none"`: Does not apply any function on the output.""",
)
class TextClassificationPipeline(Pipeline):  # 定义TextClassificationPipeline类，继承自Pipeline类
    """
    Text classification pipeline using any `ModelForSequenceClassification`. See the [sequence classification
    examples](../task_summary#sequence-classification) for more information.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> classifier = pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    >>> classifier("This movie is disgustingly good !")
    [{'label': 'POSITIVE', 'score': 1.0}]

    >>> classifier("Director tried too much.")
    [{'label': 'NEGATIVE', 'score': 0.996}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This text classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"sentiment-analysis"` (for classifying sequences according to positive or negative sentiments).

    If multiple classification labels are available (`model.config.num_labels >= 2`), the pipeline will run a softmax
    over the results. If there is a single label, the pipeline will run a sigmoid over the result.

    The models that this pipeline can use are models that have been fine-tuned on a sequence classification task. See
    """
    """
    the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=text-classification).
    """

    return_all_scores = False  # 初始化一个布尔变量，表示是否返回所有分数，默认为 False
    function_to_apply = ClassificationFunction.NONE  # 初始化一个枚举变量，表示应用的分类函数，默认为 NONE

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.check_model_type(
            TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
            if self.framework == "tf"
            else MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
        )
        # 初始化函数，调用父类的初始化方法，并根据框架类型检查模型类型

    def _sanitize_parameters(self, return_all_scores=None, function_to_apply=None, top_k="", **tokenizer_kwargs):
        # 使用 "" 作为默认参数是因为我们将在用户代码中使用 `top_k=None` 来表示"没有 top_k"
        preprocess_params = tokenizer_kwargs  # 将除了预处理参数外的其他参数赋值给 preprocess_params

        postprocess_params = {}  # 初始化后处理参数的字典
        if hasattr(self.model.config, "return_all_scores") and return_all_scores is None:
            return_all_scores = self.model.config.return_all_scores
            # 如果模型配置有 `return_all_scores` 属性且用户没有提供 return_all_scores 参数，则使用模型配置的值

        if isinstance(top_k, int) or top_k is None:
            postprocess_params["top_k"] = top_k  # 设置后处理参数中的 top_k
            postprocess_params["_legacy"] = False  # 设置后处理参数中的 _legacy 属性为 False
        elif return_all_scores is not None:
            warnings.warn(
                "`return_all_scores` is now deprecated,  if want a similar functionality use `top_k=None` instead of"
                " `return_all_scores=True` or `top_k=1` instead of `return_all_scores=False`.",
                UserWarning,
            )
            if return_all_scores:
                postprocess_params["top_k"] = None  # 如果 return_all_scores 为 True，则将 top_k 设置为 None
            else:
                postprocess_params["top_k"] = 1  # 如果 return_all_scores 为 False，则将 top_k 设置为 1

        if isinstance(function_to_apply, str):
            function_to_apply = ClassificationFunction[function_to_apply.upper()]
            # 如果 function_to_apply 是字符串，则将其转换为大写并尝试从 ClassificationFunction 枚举中获取对应的值

        if function_to_apply is not None:
            postprocess_params["function_to_apply"] = function_to_apply
            # 如果 function_to_apply 不为 None，则将其添加到后处理参数中的 function_to_apply 中

        return preprocess_params, {}, postprocess_params
        # 返回预处理参数、空字典和后处理参数
    def __call__(self, inputs, **kwargs):
        """
        Classify the text(s) given as inputs.

        Args:
            inputs (`str` or `List[str]` or `Dict[str]`, or `List[Dict[str]]`):
                One or several texts to classify. In order to use text pairs for your classification, you can send a
                dictionary containing `{"text", "text_pair"}` keys, or a list of those.
            top_k (`int`, *optional*, defaults to `1`):
                How many results to return.
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

        Return:
            A list or a list of list of `dict`: Each result comes as list of dictionaries with the following keys:

            - **label** (`str`) -- The label predicted.
            - **score** (`float`) -- The corresponding probability.

            If `top_k` is used, one such dictionary is returned per label.
        """
        # Ensure inputs are treated as a tuple, even if initially a single string
        inputs = (inputs,)
        # Call the superclass's __call__ method to perform the classification
        result = super().__call__(*inputs, **kwargs)
        # TODO try and retrieve it in a nicer way from _sanitize_parameters.
        # Check if 'top_k' is not in kwargs to determine legacy behavior
        _legacy = "top_k" not in kwargs
        # If inputs are a single string and _legacy is True, return result as a list
        if isinstance(inputs[0], str) and _legacy:
            # This pipeline is odd, and returns a list when a single item is processed
            return [result]
        else:
            # Otherwise, return the result as it is
            return result
    # 预处理方法，将输入转换为模型所需的张量字典
    def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, GenericTensor]:
        # 确定返回的张量类型由框架决定
        return_tensors = self.framework
        # 如果输入是字典类型，则使用标记器处理并返回结果
        if isinstance(inputs, dict):
            return self.tokenizer(**inputs, return_tensors=return_tensors, **tokenizer_kwargs)
        # 如果输入是列表且符合特定条件（用于处理文本对），则继续使用旧有的路径兼容处理
        elif isinstance(inputs, list) and len(inputs) == 1 and isinstance(inputs[0], list) and len(inputs[0]) == 2:
            return self.tokenizer(
                text=inputs[0][0], text_pair=inputs[0][1], return_tensors=return_tensors, **tokenizer_kwargs
            )
        # 如果输入是列表但不符合上述条件，则抛出数值错误，提示不支持的输入方式
        elif isinstance(inputs, list):
            raise ValueError(
                "The pipeline received invalid inputs, if you are trying to send text pairs, you can try to send a"
                ' dictionary `{"text": "My text", "text_pair": "My pair"}` in order to send a text pair.'
            )
        # 对于其他类型的输入，使用标记器处理并返回结果
        return self.tokenizer(inputs, return_tensors=return_tensors, **tokenizer_kwargs)

    # 内部方法，根据模型输入调用模型的前向传播方法
    def _forward(self, model_inputs):
        # 对于 `XXXForSequenceClassification` 类型的模型，即使支持 `use_cache=True`，也不应该使用
        model_forward = self.model.forward if self.framework == "pt" else self.model.call
        # 检查模型的前向传播方法签名，如果支持 `use_cache` 参数，则设为 False
        if "use_cache" in inspect.signature(model_forward).parameters.keys():
            model_inputs["use_cache"] = False
        # 调用模型的前向传播方法并返回结果
        return self.model(**model_inputs)
    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, _legacy=True):
        # `_legacy` 用于确定我们是在运行裸管道和向后兼容模式，还是在运行带有 `pipeline(..., top_k=1)` 的更自然结果包含的列表。
        # 在 `set_parameters` 之前的默认值
        
        # 如果未指定应用的函数，则根据模型配置确定默认的应用函数
        if function_to_apply is None:
            if self.model.config.problem_type == "multi_label_classification" or self.model.config.num_labels == 1:
                function_to_apply = ClassificationFunction.SIGMOID
            elif self.model.config.problem_type == "single_label_classification" or self.model.config.num_labels > 1:
                function_to_apply = ClassificationFunction.SOFTMAX
            elif hasattr(self.model.config, "function_to_apply") and function_to_apply is None:
                function_to_apply = self.model.config.function_to_apply
            else:
                function_to_apply = ClassificationFunction.NONE

        # 获取模型输出的 logits，并将其转换为 numpy 数组
        outputs = model_outputs["logits"][0]
        outputs = outputs.numpy()

        # 根据指定的函数应用对输出进行转换
        if function_to_apply == ClassificationFunction.SIGMOID:
            scores = sigmoid(outputs)
        elif function_to_apply == ClassificationFunction.SOFTMAX:
            scores = softmax(outputs)
        elif function_to_apply == ClassificationFunction.NONE:
            scores = outputs
        else:
            raise ValueError(f"Unrecognized `function_to_apply` argument: {function_to_apply}")

        # 如果 `top_k` 为 1 并且 `_legacy` 为 True，则返回最高分的标签和分数
        if top_k == 1 and _legacy:
            return {"label": self.model.config.id2label[scores.argmax().item()], "score": scores.max().item()}

        # 否则，构建包含所有标签及其分数的字典列表
        dict_scores = [
            {"label": self.model.config.id2label[i], "score": score.item()} for i, score in enumerate(scores)
        ]
        
        # 如果不是 `_legacy` 模式，则根据分数降序排序字典列表，并根据 `top_k` 进行截断
        if not _legacy:
            dict_scores.sort(key=lambda x: x["score"], reverse=True)
            if top_k is not None:
                dict_scores = dict_scores[:top_k]
        
        # 返回最终的标签及其分数的字典列表
        return dict_scores
```