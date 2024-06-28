# `.\pipelines\feature_extraction.py`

```py
# 引入类型提示字典 Dict
from typing import Dict

# 从当前模块的 utils 中导入 add_end_docstrings 函数
from ..utils import add_end_docstrings
# 从当前包的 base 模块中导入 GenericTensor, Pipeline, build_pipeline_init_args 函数
from .base import GenericTensor, Pipeline, build_pipeline_init_args

# 使用 add_end_docstrings 装饰器，添加结尾文档字符串
@add_end_docstrings(
    # 调用 build_pipeline_init_args 函数生成初始化参数
    build_pipeline_init_args(has_tokenizer=True, supports_binary_output=False),
    r"""
        tokenize_kwargs (`dict`, *optional*):
                Additional dictionary of keyword arguments passed along to the tokenizer.
        return_tensors (`bool`, *optional*):
            If `True`, returns a tensor according to the specified framework, otherwise returns a list.""",
)
# 定义 FeatureExtractionPipeline 类，继承自 Pipeline 类
class FeatureExtractionPipeline(Pipeline):
    """
    Feature extraction pipeline uses no model head. This pipeline extracts the hidden states from the base
    transformer, which can be used as features in downstream tasks.

    Example:

    ```
    >>> from transformers import pipeline

    >>> extractor = pipeline(model="google-bert/bert-base-uncased", task="feature-extraction")
    >>> result = extractor("This is a simple test.", return_tensors=True)
    >>> result.shape  # This is a tensor of shape [1, sequence_lenth, hidden_dimension] representing the input string.
    torch.Size([1, 8, 768])
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This feature extraction pipeline can currently be loaded from [`pipeline`] using the task identifier:
    `"feature-extraction"`.

    All models may be used for this pipeline. See a list of all models, including community-contributed models on
    [huggingface.co/models](https://huggingface.co/models).
    """

    # 定义 _sanitize_parameters 方法，用于预处理参数
    def _sanitize_parameters(self, truncation=None, tokenize_kwargs=None, return_tensors=None, **kwargs):
        # 如果 tokenize_kwargs 为 None，则初始化为空字典
        if tokenize_kwargs is None:
            tokenize_kwargs = {}

        # 如果 truncation 参数不为 None
        if truncation is not None:
            # 如果 tokenize_kwargs 中已经包含 'truncation' 键
            if "truncation" in tokenize_kwargs:
                # 抛出 ValueError 异常，指示 truncation 参数重复定义
                raise ValueError(
                    "truncation parameter defined twice (given as keyword argument as well as in tokenize_kwargs)"
                )
            # 否则将 truncation 参数添加到 tokenize_kwargs 中
            tokenize_kwargs["truncation"] = truncation

        # 将 tokenize_kwargs 赋值给 preprocess_params
        preprocess_params = tokenize_kwargs

        # 初始化 postprocess_params 为空字典
        postprocess_params = {}
        # 如果 return_tensors 不为 None
        if return_tensors is not None:
            # 将 return_tensors 参数添加到 postprocess_params 中
            postprocess_params["return_tensors"] = return_tensors

        # 返回预处理参数、空字典和后处理参数
        return preprocess_params, {}, postprocess_params

    # 定义 preprocess 方法，用于数据预处理
    def preprocess(self, inputs, **tokenize_kwargs) -> Dict[str, GenericTensor]:
        # 使用 self.tokenizer 对输入进行标记化，根据 self.framework 返回张量
        model_inputs = self.tokenizer(inputs, return_tensors=self.framework, **tokenize_kwargs)
        # 返回模型输入数据字典
        return model_inputs

    # 定义 _forward 方法，用于模型前向传播
    def _forward(self, model_inputs):
        # 使用 self.model 对模型输入进行前向传播，得到模型输出
        model_outputs = self.model(**model_inputs)
        # 返回模型输出
        return model_outputs

    # 定义 postprocess 方法，用于数据后处理
    def postprocess(self, model_outputs, return_tensors=False):
        # 如果 return_tensors 为 True，则返回第一个可用的张量，即 logits 或 last_hidden_state
        if return_tensors:
            return model_outputs[0]
        # 如果 self.framework 为 'pt'，则将张量转换为列表返回
        if self.framework == "pt":
            return model_outputs[0].tolist()
        # 如果 self.framework 为 'tf'，则将张量转换为 NumPy 数组再转换为列表返回
        elif self.framework == "tf":
            return model_outputs[0].numpy().tolist()
    # 定义 `__call__` 方法，该方法允许对象实例像函数一样被调用
    def __call__(self, *args, **kwargs):
        """
        提取输入文本的特征。

        Args:
            args (`str` or `List[str]`): 一个或多个文本（或文本列表），用于提取特征。

        Return:
            A nested list of `float`: 模型计算得到的特征。
        """
        # 调用父类的 `__call__` 方法，执行特征提取操作
        return super().__call__(*args, **kwargs)
```