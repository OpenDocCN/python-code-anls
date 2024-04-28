# `.\transformers\pipelines\zero_shot_classification.py`

```
# 导入模块
import inspect
from typing import List, Union

import numpy as np

# 从相对路径导入模块
from ..tokenization_utils import TruncationStrategy
from ..utils import add_end_docstrings, logging
from .base import PIPELINE_INIT_ARGS, ArgumentHandler, ChunkPipeline

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义 ZeroShotClassificationArgumentHandler 类，用于处理零样本文本分类的参数
class ZeroShotClassificationArgumentHandler(ArgumentHandler):
    """
    Handles arguments for zero-shot for text classification by turning each possible label into an NLI
    premise/hypothesis pair.
    """

    # 解析标签
    def _parse_labels(self, labels):
        if isinstance(labels, str):
            labels = [label.strip() for label in labels.split(",") if label.strip()]
        return labels

    # 调用方法
    def __call__(self, sequences, labels, hypothesis_template):
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
            sequence_pairs.extend([[sequence, hypothesis_template.format(label)] for label in labels])

        return sequence_pairs, sequences

# 添加文档结尾
@add_end_docstrings(PIPELINE_INIT_ARGS)
# 定义 ZeroShotClassificationPipeline 类，使用 ModelForSequenceClassification 在 NLI 任务上训练
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

    ```python
    >>> from transformers import pipeline

    >>> oracle = pipeline(model="facebook/bart-large-mnli")
    >>> oracle(
    ...     "I have a problem with my iphone that needs to be resolved asap!!",
    ...     candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
    ... )
    {'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['urgent', 'phone', 'computer', 'not urgent', 'tablet'], 'scores': [0.504, 0.479, 0.013, 0.003, 0.002]}
    ```
    # 调用 oracle 方法，传入文本序列和候选标签，在零样本分类任务中进行文本分类
    oracle(
        "I have a problem with my iphone that needs to be resolved asap!!",
        candidate_labels=["english", "german"],
    )
    # 返回预测结果字典，包括文本序列、标签和得分
    {'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['english', 'german'], 'scores': [0.814, 0.186]}

    # 在 pipeline 教程中了解如何使用管道的基础知识
    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    # 当前可以从 [`pipeline`] 使用以下任务标识符加载 NLI 管道："zero-shot-classification"
    This NLI pipeline can currently be loaded from [`pipeline`] using the following task identifier: "zero-shot-classification".

    # 此管道可使用已在 NLI 任务上进行过微调的模型。在 [huggingface.co/models](https://huggingface.co/models?search=nli) 上查看可用模型的最新列表
    The models that this pipeline can use are models that have been fine-tuned on an NLI task. See the up-to-date list of available models on [huggingface.co/models](https://huggingface.co/models?search=nli).
    """

    # 初始化 ZeroShotClassificationPipeline 类
    def __init__(self, args_parser=ZeroShotClassificationArgumentHandler(), *args, **kwargs):
        # 设置参数解析器
        self._args_parser = args_parser
        # 调用父类构造方法
        super().__init__(*args, **kwargs)
        # 如果 entailment_id 为 -1，则记录警告信息
        if self.entailment_id == -1:
            logger.warning(
                "Failed to determine 'entailment' label id from the label2id mapping in the model config. Setting to "
                "-1. Define a descriptive label2id mapping in the model config to ensure correct outputs."
            )

    # 获取 entailment 标签的 id
    @property
    def entailment_id(self):
        # 遍历模型配置的 label2id 映射，寻找以 entail 开头的标签
        for label, ind in self.model.config.label2id.items():
            if label.lower().startswith("entail"):
                return ind
        # 找不到则返回 -1
        return -1

    # 解析和标记文本序列对
    def _parse_and_tokenize(
        self, sequence_pairs, padding=True, add_special_tokens=True, truncation=TruncationStrategy.ONLY_FIRST, **kwargs
    ):
        """
        Parse arguments and tokenize only_first so that hypothesis (label) is not truncated
        """
        # 设置返回结果为指定的框架
        return_tensors = self.framework
        # 如果分词器不支持填充，则覆盖为使用 eos_token 作为填充符号
        if self.tokenizer.pad_token is None:
            logger.error(
                "Tokenizer was not supporting padding necessary for zero-shot, attempting to use "
                " `pad_token=eos_token`"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # 尝试使用分词器对输入进行分词，根据参数配置进行加工
        try:
            inputs = self.tokenizer(
                sequence_pairs,
                add_special_tokens=add_special_tokens,
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
            )
        # 捕获异常，处理特定情况下的分词输入错误
        except Exception as e:
            if "too short" in str(e):
                # 分词器可能会报告要截断到一个甚至没有达到的值，这种情况下不要截断而直接使用原始输入
                inputs = self.tokenizer(
                    sequence_pairs,
                    add_special_tokens=add_special_tokens,
                    return_tensors=return_tensors,
                    padding=padding,
                    truncation=TruncationStrategy.DO_NOT_TRUNCATE,
                )
            else:
                raise e

        return inputs

    # 对传入的参数进行清理和处理
    def _sanitize_parameters(self, **kwargs):
        # 如果存在 multi_class 参数，将其重命名为 multi_label，同时发出警告提示
        if kwargs.get("multi_class", None) is not None:
            kwargs["multi_label"] = kwargs["multi_class"]
            logger.warning(
                "The `multi_class` argument has been deprecated and renamed to `multi_label`. "
                "`multi_class` will be removed in a future version of Transformers."
            )
        preprocess_params = {}
        # 如果参数中存在 candidate_labels，则将其处理为候选标签
        if "candidate_labels" in kwargs:
            preprocess_params["candidate_labels"] = self._args_parser._parse_labels(kwargs["candidate_labels"])
        # 如果参数中存在 hypothesis_template，则将其作为预设假设处理
        if "hypothesis_template" in kwargs:
            preprocess_params["hypothesis_template"] = kwargs["hypothesis_template"]

        postprocess_params = {}
        # 如果参数中存在 multi_label，则作为后处理参数中的 multi_label 处理
        if "multi_label" in kwargs:
            postprocess_params["multi_label"] = kwargs["multi_label"]
        # 返回参数处理结果
        return preprocess_params, {}, postprocess_params

    # 调用函数，对传入的序列进行处理
    def __call__(
        self,
        sequences: Union[str, List[str]],
        *args,
        **kwargs,
        ):
        """
        Classify the sequence(s) given as inputs. See the [`ZeroShotClassificationPipeline`] documentation for more
        information.

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
        # Check if any additional arguments are provided
        if len(args) == 0:
            # If no additional arguments, do nothing
            pass
        elif len(args) == 1 and "candidate_labels" not in kwargs:
            # If only one additional argument is provided and "candidate_labels" not in kwargs, set it as candidate_labels
            kwargs["candidate_labels"] = args[0]
        else:
            # Raise an error if unable to understand extra arguments
            raise ValueError(f"Unable to understand extra arguments {args}")

        # Call the parent class method with provided sequences and keyword arguments
        return super().__call__(sequences, **kwargs)

    def preprocess(self, inputs, candidate_labels=None, hypothesis_template="This example is {}."):
        # Parse the inputs to get sequence pairs and sequences to classify
        sequence_pairs, sequences = self._args_parser(inputs, candidate_labels, hypothesis_template)

        # Iterate through candidate labels and corresponding sequence pairs
        for i, (candidate_label, sequence_pair) in enumerate(zip(candidate_labels, sequence_pairs)):
            # Parse and tokenize the sequence pair
            model_input = self._parse_and_tokenize([sequence_pair])

            # Yield a dictionary containing candidate_label, sequence, is_last, and model input
            yield {
                "candidate_label": candidate_label,
                "sequence": sequences[0],
                "is_last": i == len(candidate_labels) - 1,
                **model_input,
            }
    # 定义一个内部函数用于前向传播计算，接收输入数据
    def _forward(self, inputs):
        # 获取输入中的候选标签和序列数据
        candidate_label = inputs["candidate_label"]
        sequence = inputs["sequence"]
        # 根据模型tokenizer的输入名称，构建模型输入数据字典
        model_inputs = {k: inputs[k] for k in self.tokenizer.model_input_names}
        # 如果是PyTorch框架，则使用model.forward方法，否则使用model.call方法
        model_forward = self.model.forward if self.framework == "pt" else self.model.call
        # 如果模型的前向传播方法支持使用缓存，设置model_inputs中的"use_cache"为False
        if "use_cache" in inspect.signature(model_forward).parameters.keys():
            model_inputs["use_cache"] = False
        # 调用模型进行前向计算
        outputs = self.model(**model_inputs)

        # 整理模型输出，包括候选标签、序列、是否为最后一个输入等信息
        model_outputs = {
            "candidate_label": candidate_label,
            "sequence": sequence,
            "is_last": inputs["is_last"],
            **outputs,  # 包含了模型的其他输出信息
        }
        # 返回模型输出
        return model_outputs

    # 后处理函数，对模型输出进行处理
    def postprocess(self, model_outputs, multi_label=False):
        # 获取各个模型输出中的候选标签和序列数据
        candidate_labels = [outputs["candidate_label"] for outputs in model_outputs]
        sequences = [outputs["sequence"] for outputs in model_outputs]
        # 将各个模型输出中的logits拼接成一个数组
        logits = np.concatenate([output["logits"].numpy() for output in model_outputs])
        N = logits.shape[0]  # 计算logits数组的行数
        n = len(candidate_labels)  # 获取候选标签的数量
        num_sequences = N // n  # 计算序列的数量
        reshaped_outputs = logits.reshape((num_sequences, n, -1))  # 重新塑形logits数组

        # 根据是否为多标签任务或者候选标签数量等情况，进行不同的处理
        if multi_label or len(candidate_labels) == 1:
            # 对每个标签独立进行entailment vs. contradiction维度上的softmax操作
            entailment_id = self.entailment_id
            contradiction_id = -1 if entailment_id == 0 else 0
            entail_contr_logits = reshaped_outputs[..., [contradiction_id, entailment_id]]
            scores = np.exp(entail_contr_logits) / np.exp(entail_contr_logits).sum(-1, keepdims=True)
            scores = scores[..., 1]
        else:
            # 对所有候选标签上的"entailment" logits进行softmax操作
            entail_logits = reshaped_outputs[..., self.entailment_id]
            scores = np.exp(entail_logits) / np.exp(entail_logits).sum(-1, keepdims=True)

        # 对scores进行排序，获取得分最高的标签索引
        top_inds = list(reversed(scores[0].argsort()))
        # 返回处理后的结果，包括序列、标签和对应得分
        return {
            "sequence": sequences[0],
            "labels": [candidate_labels[i] for i in top_inds],
            "scores": scores[0, top_inds].tolist(),
        }
```