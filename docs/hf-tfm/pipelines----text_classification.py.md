# `.\transformers\pipelines\text_classification.py`

```py
import inspect  # 导入inspect模块，用于检查和获取对象信息
import warnings  # 导入warnings模块，用于处理警告信息
from typing import Dict  # 从typing模块导入Dict类型提示

import numpy as np  # 导入numpy库，并将其命名为np

from ..utils import ExplicitEnum, add_end_docstrings, is_tf_available, is_torch_available  # 从相对路径上级目录导入模块和函数
from .base import PIPELINE_INIT_ARGS, GenericTensor, Pipeline  # 从当前目录的base模块导入常量和类

if is_tf_available():  # 如果TensorFlow可用
    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES  # 从相对路径上级目录导入TensorFlow的相关模型映射名称

if is_torch_available():  # 如果PyTorch可用
    from ..models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES  # 从相对路径上级目录导入PyTorch的相关模型映射名称

# 定义sigmoid函数，接受_outputs作为输入参数，返回1.0 / (1.0 + exp(-_outputs))
def sigmoid(_outputs):
    return 1.0 / (1.0 + np.exp(-_outputs))

# 定义softmax函数，接受_outputs作为输入参数，返回经过softmax函数计算后的结果
def softmax(_outputs):
    # 计算每个维度上的最大值
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    # 计算每个维度上的指数值
    shifted_exp = np.exp(_outputs - maxes)
    # 对每个维度上的指数值进行归一化处理
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

# 定义一个显示枚举类，用于表示分类函数的类型
class ClassificationFunction(ExplicitEnum):
    SIGMOID = "sigmoid"  # sigmoid函数
    SOFTMAX = "softmax"  # softmax函数
    NONE = "none"  # 不应用任何函数

# 添加文档字符串注释，并继承Pipeline类
@add_end_docstrings(
    PIPELINE_INIT_ARGS,  # 继承Pipeline类的初始化参数文档字符串
    r"""
        return_all_scores (`bool`, *optional*, defaults to `False`):
            Whether to return all prediction scores or just the one of the predicted class.
        function_to_apply (`str`, *optional*, defaults to `"default"`):
            The function to apply to the model outputs in order to retrieve the scores. Accepts four different values:

            - `"default"`: if the model has a single label, will apply the sigmoid function on the output. If the model
              has several labels, will apply the softmax function on the output.
            - `"sigmoid"`: Applies the sigmoid function on the output.
            - `"softmax"`: Applies the softmax function on the output.
            - `"none"`: Does not apply any function on the output.
    """,
)
# 定义文本分类管道类，继承自Pipeline类
class TextClassificationPipeline(Pipeline):
    """
    Text classification pipeline using any `ModelForSequenceClassification`. See the [sequence classification
    examples](../task_summary#sequence-classification) for more information.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> classifier = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")
    >>> classifier("This movie is disgustingly good !")
    [{'label': 'POSITIVE', 'score': 1.0}]

    >>> classifier("Director tried too much.")
    [{'label': 'NEGATIVE', 'score': 0.996}]
    ```py

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This text classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"sentiment-analysis"` (for classifying sequences according to positive or negative sentiments).

    If multiple classification labels are available (`model.config.num_labels >= 2`), the pipeline will run a softmax
    over the results. If there is a single label, the pipeline will run a sigmoid over the result.

    The models that this pipeline can use are models that have been fine-tuned on a sequence classification task. See
    the up-to-date list of available models on

"""
    # 这个类用于加载和使用预训练的文本分类模型。
    class TextClassificationPipeline(Pipeline):
        """
        Text classification pipeline using any model trained on a sequence classification task.
        Available models can be found on 
        [huggingface.co/models](https://huggingface.co/models?filter=text-classification).
        """
    
        # 返回所有分类得分的标记是否开启
        return_all_scores = False
        # 要应用的分类函数（无/软最大值/硬最大值）
        function_to_apply = ClassificationFunction.NONE
    
        def __init__(self, **kwargs):
            # 调用父类的初始化方法
            super().__init__(**kwargs)
    
            # 检查模型类型是否匹配文本分类任务
            self.check_model_type(
                TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
                if self.framework == "tf"
                else MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
            )
    
        def _sanitize_parameters(self, return_all_scores=None, function_to_apply=None, top_k="", **tokenizer_kwargs):
            # 使用 "" 作为 top_k 的默认参数，因为我们将在用户代码中使用 top_k=None 来表示"不使用 top_k"
            preprocess_params = tokenizer_kwargs
    
            postprocess_params = {}
            # 如果模型配置支持 return_all_scores，且用户没有设置，则使用模型配置中的默认值
            if hasattr(self.model.config, "return_all_scores") and return_all_scores is None:
                return_all_scores = self.model.config.return_all_scores
    
            # 处理 top_k 参数
            if isinstance(top_k, int) or top_k is None:
                postprocess_params["top_k"] = top_k
                postprocess_params["_legacy"] = False
            elif return_all_scores is not None:
                # 如果用户同时设置了 return_all_scores 和 top_k，显示警告并根据 return_all_scores 设置 top_k
                warnings.warn(
                    "`return_all_scores` is now deprecated,  if want a similar functionality use `top_k=None` instead of"
                    " `return_all_scores=True` or `top_k=1` instead of `return_all_scores=False`.",
                    UserWarning,
                )
                if return_all_scores:
                    postprocess_params["top_k"] = None
                else:
                    postprocess_params["top_k"] = 1
    
            # 处理 function_to_apply 参数
            if isinstance(function_to_apply, str):
                function_to_apply = ClassificationFunction[function_to_apply.upper()]
    
            if function_to_apply is not None:
                postprocess_params["function_to_apply"] = function_to_apply
            return preprocess_params, {}, postprocess_params
    # 使用 __call__ 方法实现对文本进行分类
    def __call__(self, *args, **kwargs):
        """
        Classify the text(s) given as inputs.

        Args:
            args (`str` or `List[str]` or `Dict[str]`, or `List[Dict[str]]`):
                用作分类的一个或多个文本。为了在分类中使用文本对，可以发送包含`{"text", "text_pair"}`键的字典，或者包含这些字典的列表。
            top_k (`int`, *optional*, defaults to `1`):
                要返回的结果数量。
            function_to_apply (`str`, *optional*, defaults to `"default"`):
                用于检索分数的模型输出的函数。接受四个不同的值：

                如果未指定此参数，则将根据标签的数量应用以下函数：

                - 如果模型只有一个标签，则在输出上应用 Sigmoid 函数。
                - 如果模型有多个标签，则在输出上应用 Softmax 函数。

                可能的值有:

                - `"sigmoid"`: 在输出上应用 Sigmoid 函数。
                - `"softmax"`: 在输出上应用 Softmax 函数。
                - `"none"`: 不在输出上应用任何函数。

        Return:
            A list or a list of list of `dict`: 每个结果都作为具有以下键的字典列表:

            - **label** (`str`) -- 预测的标签。
            - **score** (`float`) -- 相应的概率。

            如果使用 `top_k`，则会根据每个标签返回一个这样的字典。
        """
        # 调用父类的 __call__ 方法
        result = super().__call__(*args, **kwargs)
        # 尝试以更好的方式从 _sanitize_parameters 中检索它
        _legacy = "top_k" not in kwargs
        if isinstance(args[0], str) and _legacy:
            # 当仅运行单个项目时，这个管道很奇怪，返回一个列表
            return [result]
        else:
            return result
    # 预处理输入数据，将其转换为模型所需格式的张量
    def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, GenericTensor]:
        # 确定返回张量类型
        return_tensors = self.framework
        # 如果输入是字典类型
        if isinstance(inputs, dict):
            # 使用 tokenizer 处理输入字典，并返回张量化的结果
            return self.tokenizer(**inputs, return_tensors=return_tensors, **tokenizer_kwargs)
        # 如果输入是列表类型，且列表中只有一个元素，且该元素是二维列表，保持兼容性
        elif isinstance(inputs, list) and len(inputs) == 1 and isinstance(inputs[0], list) and len(inputs[0]) == 2:
            # 以兼容性方式使用 tokenizer 处理文本对，返回张量化的结果
            return self.tokenizer(
                text=inputs[0][0], text_pair=inputs[0][1], return_tensors=return_tensors, **tokenizer_kwargs
            )
        # 如果输入是列表类型
        elif isinstance(inputs, list):
            # 抛出数值错误，因为尝试传递文本对是无效的用法
            raise ValueError(
                "The pipeline received invalid inputs, if you are trying to send text pairs, you can try to send a"
                ' dictionary `{"text": "My text", "text_pair": "My pair"}` in order to send a text pair.'
            )
        # 使用 tokenizer 处理输入，并返回张量化的结果
        return self.tokenizer(inputs, return_tensors=return_tensors, **tokenizer_kwargs)

    # 执行前向传播
    def _forward(self, model_inputs):
        # `XXXForSequenceClassification` 模型不应使用 `use_cache=True`，即使支持也不应使用
        model_forward = self.model.forward if self.framework == "pt" else self.model.call
        # 如果模型前向传播函数支持使用缓存参数
        if "use_cache" in inspect.signature(model_forward).parameters.keys():
            # 禁用缓存
            model_inputs["use_cache"] = False
        # 调用模型的前向传播函数，并返回结果
        return self.model(**model_inputs)
    # 对模型输出进行后处理，根据指定参数应用函数，返回前 k 个结果
    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, _legacy=True):
        # `_legacy` 用于确定我们是否在运行裸管道并处于向后兼容模式，或者如果运行带有 `pipeline(..., top_k=1)` 的管道，我们正在运行包含列表的更自然结果。
        # `set_parameters` 之前的默认值
        if function_to_apply is None:
            # 如果模型问题类型为多标签分类或者标签数量为1，则应用 sigmoid 函数
            if self.model.config.problem_type == "multi_label_classification" or self.model.config.num_labels == 1:
                function_to_apply = ClassificationFunction.SIGMOID
            # 如果模型问题类型为单标签分类或者标签数量大于1，则应用 softmax 函数
            elif self.model.config.problem_type == "single_label_classification" or self.model.config.num_labels > 1:
                function_to_apply = ClassificationFunction.SOFTMAX
            # 如果模型配置中存在 `function_to_apply` 属性并且 function_to_apply 为 None，则使用模型配置中的 function_to_apply
            elif hasattr(self.model.config, "function_to_apply") and function_to_apply is None:
                function_to_apply = self.model.config.function_to_apply
            # 否则，不应用任何函数
            else:
                function_to_apply = ClassificationFunction.NONE

        # 从模型输出中获取 logits，并将其转换为 NumPy 数组
        outputs = model_outputs["logits"][0]
        outputs = outputs.numpy()

        # 根据指定的函数应用对输出进行处理
        if function_to_apply == ClassificationFunction.SIGMOID:
            scores = sigmoid(outputs)
        elif function_to_apply == ClassificationFunction.SOFTMAX:
            scores = softmax(outputs)
        elif function_to_apply == ClassificationFunction.NONE:
            scores = outputs
        else:
            raise ValueError(f"Unrecognized `function_to_apply` argument: {function_to_apply}")

        # 如果 top_k 为 1 且为 _legacy 模式，返回预测结果和置信度
        if top_k == 1 and _legacy:
            return {"label": self.model.config.id2label[scores.argmax().item()], "score": scores.max().item()}

        # 将预测结果和置信度以字典形式存储
        dict_scores = [
            {"label": self.model.config.id2label[i], "score": score.item()} for i, score in enumerate(scores)
        ]

        # 如果非 _legacy 模式，则根据置信度降序排序结果，并只保留前 k 个结果
        if not _legacy:
            dict_scores.sort(key=lambda x: x["score"], reverse=True)
            if top_k is not None:
                dict_scores = dict_scores[:top_k]
        
        return dict_scores
```