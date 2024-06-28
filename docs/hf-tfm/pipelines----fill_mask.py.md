# `.\pipelines\fill_mask.py`

```
from typing import Dict  # 导入 Dict 类型提示，用于声明字典类型变量

import numpy as np  # 导入 NumPy 库，用于数值计算

from ..utils import add_end_docstrings, is_tf_available, is_torch_available, logging  # 导入自定义模块和函数

from .base import GenericTensor, Pipeline, PipelineException, build_pipeline_init_args  # 导入本地模块和类


if is_tf_available():  # 检查是否导入了 TensorFlow
    import tensorflow as tf  # 导入 TensorFlow 库

    from ..tf_utils import stable_softmax  # 导入自定义 TensorFlow 工具函数


if is_torch_available():  # 检查是否导入了 PyTorch
    import torch  # 导入 PyTorch 库


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


@add_end_docstrings(  # 使用装饰器为类添加文档字符串
    build_pipeline_init_args(has_tokenizer=True),  # 调用 build_pipeline_init_args 函数生成初始化参数说明
    r"""
        top_k (`int`, defaults to 5):
            The number of predictions to return.
        targets (`str` or `List[str]`, *optional*):
            When passed, the model will limit the scores to the passed targets instead of looking up in the whole
            vocab. If the provided targets are not in the model vocab, they will be tokenized and the first resulting
            token will be used (with a warning, and that might be slower).
        tokenizer_kwargs (`dict`, *optional*):
            Additional dictionary of keyword arguments passed along to the tokenizer."""  # 对类的文档字符串进行详细注释
)
class FillMaskPipeline(Pipeline):  # 定义 FillMaskPipeline 类，继承自 Pipeline 类
    """
    Masked language modeling prediction pipeline using any `ModelWithLMHead`. See the [masked language modeling
    examples](../task_summary#masked-language-modeling) for more information.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> fill_masker = pipeline(model="google-bert/bert-base-uncased")
    >>> fill_masker("This is a simple [MASK].")
    [{'score': 0.042, 'token': 3291, 'token_str': 'problem', 'sequence': 'this is a simple problem.'}, {'score': 0.031, 'token': 3160, 'token_str': 'question', 'sequence': 'this is a simple question.'}, {'score': 0.03, 'token': 8522, 'token_str': 'equation', 'sequence': 'this is a simple equation.'}, {'score': 0.027, 'token': 2028, 'token_str': 'one', 'sequence': 'this is a simple one.'}, {'score': 0.024, 'token': 3627, 'token_str': 'rule', 'sequence': 'this is a simple rule.'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This mask filling pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"fill-mask"`.

    The models that this pipeline can use are models that have been trained with a masked language modeling objective,
    which includes the bi-directional models in the library. See the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=fill-mask).

    <Tip>

    This pipeline only works for inputs with exactly one token masked. Experimental: We added support for multiple
    masks. The returned values are raw model output, and correspond to disjoint probabilities where one might expect
    joint probabilities (See [discussion](https://github.com/huggingface/transformers/pull/10222)).

    </Tip>

    <Tip>

    This pipeline now supports tokenizer_kwargs. For example try:

    ```python
    >>> from transformers import pipeline

    >>> fill_masker = pipeline(model="google-bert/bert-base-uncased", tokenizer_kwargs={"do_lower_case": False})
    >>> fill_masker("This is a simple [MASK].")
    ```

    This will make the tokenizer to treat "This" and "this" as distinct words.

    </Tip>
    """
    # 类的主体部分包含了关于使用遮罩语言建模的预测流水线的详细说明和示例，以及相关的提示和链接。
    pass  # 类体中没有代码，所以使用 pass 语句进行占位
    >>> from transformers import pipeline
导入transformers库中的pipeline模块，用于创建基于预训练模型的NLP处理管道。

    >>> fill_masker = pipeline(model="google-bert/bert-base-uncased")
创建一个新的填充掩码（fill-mask）管道，使用Google BERT模型的基本未大写模型。

    >>> tokenizer_kwargs = {"truncation": True}
定义一个字典tokenizer_kwargs，其中包含了一个键值对，用于设置tokenizer的参数，这里指定了截断为True。

    >>> fill_masker(
    ...     "This is a simple [MASK]. " + "...with a large amount of repeated text appended. " * 100,
    ...     tokenizer_kwargs=tokenizer_kwargs,
    ... )
调用填充掩码（fill-mask）管道的函数，传入一个带有填充掩码标记的文本以及tokenizer的额外参数。这里的文本是一个简单的句子，加上了大量重复的文本内容。

    """

    def get_masked_index(self, input_ids: GenericTensor) -> np.ndarray:
        if self.framework == "tf":
            masked_index = tf.where(input_ids == self.tokenizer.mask_token_id).numpy()
        elif self.framework == "pt":
            masked_index = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=False)
        else:
            raise ValueError("Unsupported framework")
        return masked_index
定义一个方法get_masked_index，用于获取输入张量中掩码标记的索引。根据self.framework属性，如果是"tf"则使用TensorFlow库的方法找到掩码标记索引并转换为NumPy数组，如果是"pt"则使用PyTorch库的方法返回掩码标记索引，否则抛出异常。

    def _ensure_exactly_one_mask_token(self, input_ids: GenericTensor) -> np.ndarray:
        masked_index = self.get_masked_index(input_ids)
        numel = np.prod(masked_index.shape)
        if numel < 1:
            raise PipelineException(
                "fill-mask",
                self.model.base_model_prefix,
                f"No mask_token ({self.tokenizer.mask_token}) found on the input",
            )
定义一个方法_ensure_exactly_one_mask_token，确保输入张量中只有一个掩码标记。首先调用get_masked_index方法获取掩码标记的索引，然后计算索引数组的元素数量。如果数量小于1，则抛出PipelineException异常，提示输入中未找到掩码标记。

    def ensure_exactly_one_mask_token(self, model_inputs: GenericTensor):
        if isinstance(model_inputs, list):
            for model_input in model_inputs:
                self._ensure_exactly_one_mask_token(model_input["input_ids"][0])
        else:
            for input_ids in model_inputs["input_ids"]:
                self._ensure_exactly_one_mask_token(input_ids)
定义一个公共方法ensure_exactly_one_mask_token，用于确保模型输入中每个示例只有一个掩码标记。根据输入类型（列表或单个输入），对每个模型输入调用_ensure_exactly_one_mask_token方法。

    def preprocess(
        self, inputs, return_tensors=None, tokenizer_kwargs=None, **preprocess_parameters
    ) -> Dict[str, GenericTensor]:
        if return_tensors is None:
            return_tensors = self.framework
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}

        model_inputs = self.tokenizer(inputs, return_tensors=return_tensors, **tokenizer_kwargs)
        self.ensure_exactly_one_mask_token(model_inputs)
        return model_inputs
定义一个预处理方法preprocess，用于将原始输入处理成适合模型的输入格式。根据参数设置，调用tokenizer将输入转换成张量表示，并调用ensure_exactly_one_mask_token方法确保每个模型输入只有一个掩码标记。

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        model_outputs["input_ids"] = model_inputs["input_ids"]
        return model_outputs
定义一个方法_forward，执行模型的前向传播。调用模型将输入传递给模型并返回输出，同时保留输入中的input_ids信息。
    # 定义一个方法用于后处理模型输出，接受模型输出、top_k 参数和目标标识符作为输入
    def postprocess(self, model_outputs, top_k=5, target_ids=None):
        # 如果存在目标标识符并且目标标识符的数量少于 top_k，则将 top_k 设置为目标标识符的数量
        if target_ids is not None and target_ids.shape[0] < top_k:
            top_k = target_ids.shape[0]
        # 获取模型输出中的输入标识符
        input_ids = model_outputs["input_ids"][0]
        # 获取模型输出中的预测 logits
        outputs = model_outputs["logits"]

        # 如果使用 TensorFlow 框架
        if self.framework == "tf":
            # 找到输入标识符中等于 tokenizer 的 mask_token_id 的位置索引
            masked_index = tf.where(input_ids == self.tokenizer.mask_token_id).numpy()[:, 0]

            # 将 logits 转换为 numpy 数组
            outputs = outputs.numpy()

            # 提取特定位置的 logits
            logits = outputs[0, masked_index, :]
            # 对 logits 进行稳定的 softmax 操作
            probs = stable_softmax(logits, axis=-1)
            # 如果存在目标标识符，则根据目标标识符从 probs 中抽取对应的概率
            if target_ids is not None:
                probs = tf.gather_nd(tf.squeeze(probs, 0), target_ids.reshape(-1, 1))
                probs = tf.expand_dims(probs, 0)

            # 获取概率最高的 top_k 个值和对应的索引
            topk = tf.math.top_k(probs, k=top_k)
            values, predictions = topk.values.numpy(), topk.indices.numpy()
        else:
            # 如果使用的是 PyTorch 框架，找到输入标识符中等于 tokenizer 的 mask_token_id 的位置索引
            masked_index = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=False).squeeze(-1)
            # Fill mask pipeline supports only one ${mask_token} per sample

            # 提取特定位置的 logits
            logits = outputs[0, masked_index, :]
            # 对 logits 进行 softmax 操作
            probs = logits.softmax(dim=-1)
            # 如果存在目标标识符，则根据目标标识符从 probs 中抽取对应的概率
            if target_ids is not None:
                probs = probs[..., target_ids]

            # 获取概率最高的 top_k 个值和对应的索引
            values, predictions = probs.topk(top_k)

        # 初始化结果列表
        result = []
        # 检查是否只有单个 mask
        single_mask = values.shape[0] == 1
        # 遍历概率值和对应的预测值
        for i, (_values, _predictions) in enumerate(zip(values.tolist(), predictions.tolist())):
            row = []
            for v, p in zip(_values, _predictions):
                # 创建输入标识符的副本，因为后续会修改此数组
                tokens = input_ids.numpy().copy()
                # 如果存在目标标识符，则将 p 替换为目标标识符中的对应值
                if target_ids is not None:
                    p = target_ids[p].tolist()

                # 将输入标识符中的 mask 位置替换为 p
                tokens[masked_index[i]] = p
                # 过滤掉填充标记
                tokens = tokens[np.where(tokens != self.tokenizer.pad_token_id)]
                # 使用 tokenizer 解码 tokens 生成序列，根据 single_mask 决定是否跳过特殊标记
                sequence = self.tokenizer.decode(tokens, skip_special_tokens=single_mask)
                # 创建建议字典，包含分数、标记、标记字符串和序列
                proposition = {"score": v, "token": p, "token_str": self.tokenizer.decode([p]), "sequence": sequence}
                row.append(proposition)
            result.append(row)
        # 如果只有单个 mask，则返回结果列表的第一个元素
        if single_mask:
            return result[0]
        # 否则返回完整的结果列表
        return result
    # 获取目标标记的对应 ID 列表
    def get_target_ids(self, targets, top_k=None):
        # 如果目标是字符串，则转换为列表形式
        if isinstance(targets, str):
            targets = [targets]
        try:
            # 获取当前 tokenizer 的词汇表
            vocab = self.tokenizer.get_vocab()
        except Exception:
            # 若获取失败则设置空词汇表
            vocab = {}
        # 初始化目标 ID 列表
        target_ids = []
        # 遍历每个目标标记
        for target in targets:
            # 获取目标标记在词汇表中的 ID，若不存在则为 None
            id_ = vocab.get(target, None)
            # 如果 ID 不存在
            if id_ is None:
                # 使用 tokenizer 处理目标标记，获取其对应的 input_ids
                input_ids = self.tokenizer(
                    target,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    max_length=1,
                    truncation=True,
                )["input_ids"]
                # 如果 input_ids 长度为 0，表示标记在模型词汇表中不存在
                if len(input_ids) == 0:
                    # 发出警告，指出指定的目标标记在模型词汇表中不存在
                    logger.warning(
                        f"The specified target token `{target}` does not exist in the model vocabulary. "
                        "We cannot replace it with anything meaningful, ignoring it"
                    )
                    # 继续下一个目标标记的处理
                    continue
                # 将第一个 input_id 作为替代标记的 ID
                id_ = input_ids[0]
                # 发出警告，指出替代了不存在的目标标记，并提示替代的标记
                logger.warning(
                    f"The specified target token `{target}` does not exist in the model vocabulary. "
                    f"Replacing with `{self.tokenizer.convert_ids_to_tokens(id_)}`."
                )
            # 将获取到的目标标记 ID 添加到列表中
            target_ids.append(id_)
        # 去重目标 ID 列表
        target_ids = list(set(target_ids))
        # 如果目标 ID 列表为空，则抛出数值错误异常
        if len(target_ids) == 0:
            raise ValueError("At least one target must be provided when passed.")
        # 转换目标 ID 列表为 NumPy 数组格式
        target_ids = np.array(target_ids)
        # 返回目标 ID 数组
        return target_ids

    # 清理参数函数，返回预处理、后处理参数及空字典
    def _sanitize_parameters(self, top_k=None, targets=None, tokenizer_kwargs=None):
        preprocess_params = {}

        # 如果存在 tokenizer_kwargs 参数，则添加到预处理参数中
        if tokenizer_kwargs is not None:
            preprocess_params["tokenizer_kwargs"] = tokenizer_kwargs

        postprocess_params = {}

        # 如果存在 targets 参数，则获取目标标记的 ID 列表
        if targets is not None:
            target_ids = self.get_target_ids(targets, top_k)
            postprocess_params["target_ids"] = target_ids

        # 如果存在 top_k 参数，则添加到后处理参数中
        if top_k is not None:
            postprocess_params["top_k"] = top_k

        # 如果 tokenizer 的 mask_token_id 为 None，则抛出管道异常
        if self.tokenizer.mask_token_id is None:
            raise PipelineException(
                "fill-mask", self.model.base_model_prefix, "The tokenizer does not define a `mask_token`."
            )

        # 返回预处理参数、空字典和后处理参数
        return preprocess_params, {}, postprocess_params
    # 覆盖父类的 __call__ 方法，用于填充输入文本中的掩码标记。

    outputs = super().__call__(inputs, **kwargs)
    # 调用父类的 __call__ 方法，传入输入参数 inputs 和其他关键字参数 kwargs，并获取输出结果

    if isinstance(inputs, list) and len(inputs) == 1:
        # 检查 inputs 是否为列表且长度为1
        return outputs[0]
        # 如果是单个文本输入，则直接返回第一个输出结果
    return outputs
    # 否则返回所有输出结果
```