# `.\transformers\pipelines\feature_extraction.py`

```py
# 导入 Dict 类型
from typing import Dict
# 从base模块导入GenericTensor和Pipeline类
from .base import GenericTensor, Pipeline

# 不能在这里使用@add_end_docstrings(PIPELINE_INIT_ARGS)，因为这里不支持`binary_output`
class FeatureExtractionPipeline(Pipeline):
    """
    特征提取管道，不使用模型头。此管道从基础transformer中提取隐藏状态，可以用作下游任务中的特征。

    示例：

    ```python
    >>> from transformers import pipeline

    >>> extractor = pipeline(model="bert-base-uncased", task="feature-extraction")
    >>> result = extractor("This is a simple test.", return_tensors=True)
    >>> result.shape  # 这是一个形状为[1, sequence_lenth, hidden_dimension]的张量，表示输入字符串。
    torch.Size([1, 8, 768])
    ```py

    了解有关在[pipeline tutorial](../pipeline_tutorial)中使用管道的基础知识

    目前可以使用此特征提取管道通过任务标识符从[`pipeline`]加载。

    所有模型都可以用于此管道。在[huggingface.co/models](https://huggingface.co/models)上查看所有模型，包括社区贡献的模型。

    参数：
        model ([`PreTrainedModel`]或[`TFPreTrainedModel`]）：用于管道进行预测的模型。这需要是继承自[`PreTrainedModel`]的PyTorch模型和继承自[`TFPreTrainedModel`]的TensorFlow模型。
        tokenizer ([`PreTrainedTokenizer`]）：用于为模型编码数据的分词器。该对象继承自[`PreTrainedTokenizer`]。
        modelcard（`str`或[`ModelCard`]，*可选*）：针对此管道的模型卡。
        framework（`str`，*可选*）：要使用的框架，可以是`"pt"`表示PyTorch，也可以是`"tf"`表示TensorFlow。指定的框架必须已安装。

            如果未指定框架，则将默认使用当前已安装的框架。如果未指定框架，并且两个框架都已安装，则将默认使用`model`的框架，或者如果未提供模型，则将默认使用PyTorch。
        return_tensors（`bool`，*可选*）：如果为`True`，则根据指定的框架返回张量，否则返回列表。
        task（`str`，默认为`""`）：管道的任务标识符。
        args_parser（[`~pipelines.ArgumentHandler`]，*可选*）：负责解析提供的管道参数的对象的引用。
        device（`int`，*可选*，默认为-1）：用于CPU/GPU支持的设备顺序。将其设置为-1将利用CPU，设置为正数将在相关的CUDA设备ID上运行模型。
        tokenize_kwargs（`dict`，*可选*）：传递给分词器的附加关键字参数的字典。
    """
    # 清理参数，将截断、分词参数和返回张量参数从输入参数中分离出来
    def _sanitize_parameters(self, truncation=None, tokenize_kwargs=None, return_tensors=None, **kwargs):
        # 如果未提供分词参数，则初始化为空字典
        if tokenize_kwargs is None:
            tokenize_kwargs = {}

        # 如果存在截断参数，则检查是否已经在分词参数中定义，如果是则引发 ValueError
        if truncation is not None:
            if "truncation" in tokenize_kwargs:
                raise ValueError(
                    "truncation parameter defined twice (given as keyword argument as well as in tokenize_kwargs)"
                )
            # 将截断参数添加到分词参数中
            tokenize_kwargs["truncation"] = truncation

        # 将预处理参数设置为分词参数
        preprocess_params = tokenize_kwargs

        # 初始化后处理参数为空字典
        postprocess_params = {}
        # 如果指定了返回张量参数，则在后处理参数中设置返回张量
        if return_tensors is not None:
            postprocess_params["return_tensors"] = return_tensors

        # 返回预处理参数、空字典和后处理参数
        return preprocess_params, {}, postprocess_params

    # 对输入进行预处理，返回模型输入
    def preprocess(self, inputs, **tokenize_kwargs) -> Dict[str, GenericTensor]:
        # 使用分词器对输入进行分词，并根据框架类型返回张量
        model_inputs = self.tokenizer(inputs, return_tensors=self.framework, **tokenize_kwargs)
        # 返回模型输入
        return model_inputs

    # 模型前向传播
    def _forward(self, model_inputs):
        # 调用模型进行前向传播，获取模型输出
        model_outputs = self.model(**model_inputs)
        # 返回模型输出
        return model_outputs

    # 对模型输出进行后处理
    def postprocess(self, model_outputs, return_tensors=False):
        # 如果 return_tensors 为 True，则返回模型输出的第一个张量
        if return_tensors:
            return model_outputs[0]
        # 如果框架类型为 PyTorch，则将张量转换为列表
        if self.framework == "pt":
            return model_outputs[0].tolist()
        # 如果框架类型为 TensorFlow，则将张量转换为 NumPy 数组，再转换为列表
        elif self.framework == "tf":
            return model_outputs[0].numpy().tolist()

    # 对象可调用，用于提取输入的特征
    def __call__(self, *args, **kwargs):
        """
        Extract the features of the input(s).

        Args:
            args (`str` or `List[str]`): One or several texts (or one list of texts) to get the features of.

        Return:
            A nested list of `float`: The features computed by the model.
        """
        # 调用父类的 __call__ 方法，返回输入的特征
        return super().__call__(*args, **kwargs)
```