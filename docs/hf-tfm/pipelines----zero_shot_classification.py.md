# `.\pipelines\zero_shot_classification.py`

```py
import inspect  # 导入 inspect 模块，用于获取对象的信息
from typing import List, Union  # 引入类型提示中的 List 和 Union 类型

import numpy as np  # 导入 NumPy 库，用于数值计算

from ..tokenization_utils import TruncationStrategy  # 导入相对路径下的 tokenization_utils 模块中的 TruncationStrategy 类
from ..utils import add_end_docstrings, logging  # 导入相对路径下的 utils 模块中的 add_end_docstrings 和 logging

from .base import ArgumentHandler, ChunkPipeline, build_pipeline_init_args  # 从当前目录下的 base 模块中导入 ArgumentHandler、ChunkPipeline 和 build_pipeline_init_args 类

logger = logging.get_logger(__name__)  # 获取当前模块的 logger 对象


class ZeroShotClassificationArgumentHandler(ArgumentHandler):
    """
    Handles arguments for zero-shot for text classification by turning each possible label into an NLI
    premise/hypothesis pair.
    """

    def _parse_labels(self, labels):
        # 将 labels 转换为列表形式，如果 labels 是字符串则按逗号分隔并去除空白项
        if isinstance(labels, str):
            labels = [label.strip() for label in labels.split(",") if label.strip()]
        return labels

    def __call__(self, sequences, labels, hypothesis_template):
        # 处理调用，验证输入的 sequences 和 labels，确保至少有一个 label 和一个 sequence
        if len(labels) == 0 or len(sequences) == 0:
            raise ValueError("You must include at least one label and at least one sequence.")
        if hypothesis_template.format(labels[0]) == hypothesis_template:
            raise ValueError(
                (
                    'The provided hypothesis_template "{}" was not able to be formatted with the target labels. '
                    "Make sure the passed template includes formatting syntax such as {{}} where the label should go."
                ).format(hypothesis_template)
            )

        if isinstance(sequences, str):
            sequences = [sequences]

        sequence_pairs = []
        for sequence in sequences:
            # 生成每个 sequence 和对应 label 格式化后的假设/前提对
            sequence_pairs.extend([[sequence, hypothesis_template.format(label)] for label in labels])

        return sequence_pairs, sequences


@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True))
class ZeroShotClassificationPipeline(ChunkPipeline):
    """
    NLI-based zero-shot classification pipeline using a `ModelForSequenceClassification` trained on NLI (natural
    language inference) tasks. Equivalent of `text-classification` pipelines, but these models don't require a
    hardcoded number of potential classes, they can be chosen at runtime. It usually means it's slower but it is
    **much** more flexible.

    Any combination of sequences and labels can be passed and each combination will be posed as a premise/hypothesis
    pair and passed to the pretrained model. Then, the logit for *entailment* is taken as the logit for the candidate
    label being valid. Any NLI model can be used, but the id of the *entailment* label must be included in the model
    config's :attr:*~transformers.PretrainedConfig.label2id*.

    Example:

    ```
    >>> from transformers import pipeline

    >>> oracle = pipeline(model="facebook/bart-large-mnli")
    >>> oracle(
    ...     "I have a problem with my iphone that needs to be resolved asap!!",
    ...     candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
    ... )

    """

    pass  # 这是一个基于 NLI 的零样本分类管道，继承自 ChunkPipeline 类，但未实现额外功能
    {'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['urgent', 'phone', 'computer', 'not urgent', 'tablet'], 'scores': [0.504, 0.479, 0.013, 0.003, 0.002]}

    >>> oracle(
    ...     "I have a problem with my iphone that needs to be resolved asap!!",
    ...     candidate_labels=["english", "german"],
    ... )
    {'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['english', 'german'], 'scores': [0.814, 0.186]}


# 示例输入和输出，展示了使用 oracle 函数进行 zero-shot 分类的过程。
# 输入包含一个文本序列和候选标签列表，输出包含预测的标签和对应的置信度分数。

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This NLI pipeline can currently be loaded from [`pipeline`] using the following task identifier:
`"zero-shot-classification"`.

The models that this pipeline can use are models that have been fine-tuned on an NLI task. See the up-to-date list
of available models on [huggingface.co/models](https://huggingface.co/models?search=nli).


# 提供了关于如何使用流水线的基础知识的链接和一些相关信息。
# 说明了当前的 NLI 流水线可以通过指定的任务标识符 `"zero-shot-classification"` 来加载。

    def __init__(self, args_parser=ZeroShotClassificationArgumentHandler(), *args, **kwargs):
        self._args_parser = args_parser
        super().__init__(*args, **kwargs)
        if self.entailment_id == -1:
            logger.warning(
                "Failed to determine 'entailment' label id from the label2id mapping in the model config. Setting to "
                "-1. Define a descriptive label2id mapping in the model config to ensure correct outputs."
            )


# 初始化方法，接受参数解析器并调用父类的初始化方法。
# 如果 entailment_id 为 -1，记录警告信息，指示无法从模型配置中的 label2id 映射中确定 'entailment' 标签的 id。
# 建议在模型配置中定义一个描述性的 label2id 映射以确保正确的输出。


    @property
    def entailment_id(self):
        for label, ind in self.model.config.label2id.items():
            if label.lower().startswith("entail"):
                return ind
        return -1


# 属性方法，用于获取模型配置中与 'entail' 开头的标签对应的 id。
# 如果找不到符合条件的标签，则返回 -1。


    def _parse_and_tokenize(
        self, sequence_pairs, padding=True, add_special_tokens=True, truncation=TruncationStrategy.ONLY_FIRST, **kwargs


# 私有方法，用于解析和标记化序列对。
# 接受序列对、是否填充、是否添加特殊标记、截断策略等参数。
    ):
        """
        Parse arguments and tokenize only_first so that hypothesis (label) is not truncated
        """
        # 将返回的张量设置为框架的默认值
        return_tensors = self.framework
        # 如果当前分词器不支持填充操作
        if self.tokenizer.pad_token is None:
            # 为不支持填充的分词器设置 `pad_token` 为 `eos_token`
            logger.error(
                "Tokenizer was not supporting padding necessary for zero-shot, attempting to use "
                " `pad_token=eos_token`"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            # 使用分词器对序列对进行分词处理，包括特殊标记、返回张量、填充和截断设置
            inputs = self.tokenizer(
                sequence_pairs,
                add_special_tokens=add_special_tokens,
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
            )
        except Exception as e:
            # 如果出现异常且异常信息包含 "too short"
            if "too short" in str(e):
                # 分词器可能会报告我们想要截断到一个甚至不会被输入达到的值。
                # 在这种情况下，我们不希望进行截断。
                # 看起来没有更好的方法来捕获这个异常。
                
                # 以 `DO_NOT_TRUNCATE` 策略再次尝试使用分词器对序列对进行处理
                inputs = self.tokenizer(
                    sequence_pairs,
                    add_special_tokens=add_special_tokens,
                    return_tensors=return_tensors,
                    padding=padding,
                    truncation=TruncationStrategy.DO_NOT_TRUNCATE,
                )
            else:
                # 如果异常不符合上述条件，则抛出该异常
                raise e

        # 返回分词后的输入
        return inputs

    def _sanitize_parameters(self, **kwargs):
        # 如果 `multi_class` 参数不为 None，则将其重命名为 `multi_label`
        if kwargs.get("multi_class", None) is not None:
            kwargs["multi_label"] = kwargs["multi_class"]
            logger.warning(
                "The `multi_class` argument has been deprecated and renamed to `multi_label`. "
                "`multi_class` will be removed in a future version of Transformers."
            )
        preprocess_params = {}
        # 如果参数中包含 "candidate_labels"，则解析标签并存储到预处理参数中
        if "candidate_labels" in kwargs:
            preprocess_params["candidate_labels"] = self._args_parser._parse_labels(kwargs["candidate_labels"])
        # 如果参数中包含 "hypothesis_template"，则存储到预处理参数中
        if "hypothesis_template" in kwargs:
            preprocess_params["hypothesis_template"] = kwargs["hypothesis_template"]

        postprocess_params = {}
        # 如果参数中包含 "multi_label"，则存储到后处理参数中
        if "multi_label" in kwargs:
            postprocess_params["multi_label"] = kwargs["multi_label"]
        # 返回预处理参数、空字典和后处理参数
        return preprocess_params, {}, postprocess_params

    def __call__(
        self,
        sequences: Union[str, List[str]],
        *args,
        **kwargs,
    ):
        """
        Classify the sequence(s) given as inputs using a zero-shot classification pipeline.

        Args:
            sequences (`str` or `List[str]`):
                The sequence(s) to classify, will be truncated if the model input is too large.
            candidate_labels (`str` or `List[str]`):
                The set of possible class labels to classify each sequence into. Can be a single label, a string of
                comma-separated labels, or a list of labels.
            hypothesis_template (`str`, *optional*, defaults to `"This example is {}."`):
                The template used to turn each label into an NLI-style hypothesis. This template must include a {} or
                similar syntax for the candidate label to be inserted into the template. For example, the default
                template is `"This example is {}."` With the candidate label `"sports"`, this would be fed into the
                model like `"<cls> sequence to classify <sep> This example is sports . <sep>"`. The default template
                works well in many cases, but it may be worthwhile to experiment with different templates depending on
                the task setting.
            multi_label (`bool`, *optional*, defaults to `False`):
                Whether or not multiple candidate labels can be true. If `False`, the scores are normalized such that
                the sum of the label likelihoods for each sequence is 1. If `True`, the labels are considered
                independent and probabilities are normalized for each candidate by doing a softmax of the entailment
                score vs. the contradiction score.

        Return:
            A `dict` or a list of `dict`: Each result comes as a dictionary with the following keys:

            - **sequence** (`str`) -- The sequence for which this is the output.
            - **labels** (`List[str]`) -- The labels sorted by order of likelihood.
            - **scores** (`List[float]`) -- The probabilities for each of the labels.
        """
        if len(args) == 0:
            # If no positional arguments (`args`) are provided, do nothing
            pass
        elif len(args) == 1 and "candidate_labels" not in kwargs:
            # If exactly one positional argument is provided and `candidate_labels` is not already in keyword arguments (`kwargs`), set it to the provided argument
            kwargs["candidate_labels"] = args[0]
        else:
            # Raise a ValueError if extra arguments (`args`) are present or if `candidate_labels` is already specified in `kwargs`
            raise ValueError(f"Unable to understand extra arguments {args}")

        # Call the superclass's `__call__` method with `sequences` and other keyword arguments (`kwargs`)
        return super().__call__(sequences, **kwargs)

    def preprocess(self, inputs, candidate_labels=None, hypothesis_template="This example is {}."):
        # Parse input arguments and prepare for processing
        sequence_pairs, sequences = self._args_parser(inputs, candidate_labels, hypothesis_template)

        # Iterate over candidate labels and corresponding sequence pairs
        for i, (candidate_label, sequence_pair) in enumerate(zip(candidate_labels, sequence_pairs)):
            # Parse and tokenize the sequence pair for model input
            model_input = self._parse_and_tokenize([sequence_pair])

            # Yield a dictionary containing processed information
            yield {
                "candidate_label": candidate_label,
                "sequence": sequences[0],  # Assuming sequences contains only one element
                "is_last": i == len(candidate_labels) - 1,  # Flag indicating if it's the last iteration
                **model_input,  # Include parsed and tokenized model input
            }
    # 定义一个方法 `_forward`，用于执行模型的前向推理过程
    def _forward(self, inputs):
        # 从输入中获取候选标签
        candidate_label = inputs["candidate_label"]
        # 从输入中获取序列数据
        sequence = inputs["sequence"]
        # 创建一个字典，包含模型所需的输入数据，使用 tokenizer 支持的模型输入名称作为键
        model_inputs = {k: inputs[k] for k in self.tokenizer.model_input_names}
        # 根据框架选择适当的模型前向推理函数
        model_forward = self.model.forward if self.framework == "pt" else self.model.call
        # 如果模型前向推理函数支持 `use_cache` 参数，则设为 False
        if "use_cache" in inspect.signature(model_forward).parameters.keys():
            model_inputs["use_cache"] = False
        # 执行模型推理，并获取输出
        outputs = self.model(**model_inputs)

        # 构建模型输出字典，包括候选标签、序列和是否最后一个输入的标志，以及模型的其他输出
        model_outputs = {
            "candidate_label": candidate_label,
            "sequence": sequence,
            "is_last": inputs["is_last"],
            **outputs,
        }
        # 返回模型的输出
        return model_outputs

    # 定义一个后处理方法 `postprocess`，用于处理模型的输出结果
    def postprocess(self, model_outputs, multi_label=False):
        # 从模型输出中提取候选标签列表和序列列表
        candidate_labels = [outputs["candidate_label"] for outputs in model_outputs]
        sequences = [outputs["sequence"] for outputs in model_outputs]
        # 提取模型输出中的 logits，并拼接成一个 numpy 数组
        logits = np.concatenate([output["logits"].numpy() for output in model_outputs])
        # 获取 logits 的维度信息
        N = logits.shape[0]
        n = len(candidate_labels)
        # 计算序列的数量
        num_sequences = N // n
        # 将 logits 重塑成三维数组
        reshaped_outputs = logits.reshape((num_sequences, n, -1))

        if multi_label or len(candidate_labels) == 1:
            # 对每个标签独立进行 entailment vs. contradiction 的 softmax 处理
            entailment_id = self.entailment_id
            contradiction_id = -1 if entailment_id == 0 else 0
            entail_contr_logits = reshaped_outputs[..., [contradiction_id, entailment_id]]
            scores = np.exp(entail_contr_logits) / np.exp(entail_contr_logits).sum(-1, keepdims=True)
            # 提取 entailment 的概率分数
            scores = scores[..., 1]
        else:
            # 对所有候选标签的 "entailment" logits 进行 softmax 处理
            entail_logits = reshaped_outputs[..., self.entailment_id]
            scores = np.exp(entail_logits) / np.exp(entail_logits).sum(-1, keepdims=True)

        # 获取 top K 概率最高的标签索引
        top_inds = list(reversed(scores[0].argsort()))
        # 返回后处理的结果，包括序列、按概率排序的标签和对应的分数列表
        return {
            "sequence": sequences[0],
            "labels": [candidate_labels[i] for i in top_inds],
            "scores": scores[0, top_inds].tolist(),
        }
```