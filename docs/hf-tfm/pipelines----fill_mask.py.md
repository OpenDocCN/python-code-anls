# `.\transformers\pipelines\fill_mask.py`

```
# 从 typing 模块中导入 Dict 类型提示
from typing import Dict
# 从 numpy 模块中导入 np 模块
import numpy as np
# 从 ..utils 模块中导入 add_end_docstrings, is_tf_available, is_torch_available, logging 函数
from ..utils import add_end_docstrings, is_tf_available, is_torch_available, logging
# 从 .base 模块中导入 PIPELINE_INIT_ARGS, GenericTensor, Pipeline, PipelineException 函数
from .base import PIPELINE_INIT_ARGS, GenericTensor, Pipeline, PipelineException

# 如果当前环境支持 TensorFlow
if is_tf_available():
    # 导入 tensorflow 模块
    import tensorflow as tf
    # 从 ..tf_utils 模块中导入 stable_softmax 函数

# 如果当前环境支持 PyTorch
if is_torch_available():
    # 导入 torch 模块
    import torch
# 获取 logger 对象
logger = logging.get_logger(__name__)

# 使用 add_end_docstrings 函数添加多段注释
@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        top_k (`int`, defaults to 5):
            The number of predictions to return.
        targets (`str` or `List[str]`, *optional*):
            When passed, the model will limit the scores to the passed targets instead of looking up in the whole
            vocab. If the provided targets are not in the model vocab, they will be tokenized and the first resulting
            token will be used (with a warning, and that might be slower).

    """,
)
# 定义 FillMaskPipeline 类，继承自 Pipeline 类
class FillMaskPipeline(Pipeline):
    """
    Masked language modeling prediction pipeline using any `ModelWithLMHead`. See the [masked language modeling
    examples](../task_summary#masked-language-modeling) for more information.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> fill_masker = pipeline(model="bert-base-uncased")
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

    >>> fill_masker = pipeline(model="bert-base-uncased")
    >>> tokenizer_kwargs = {"truncation": True}
    >>> fill_masker(
    def get_masked_index(self, input_ids: GenericTensor) -> np.ndarray:
        # 判断当前框架（tf或pt），如果是tf，则使用numpy()方法将Tensorflow张量转换为NumPy数组
        if self.framework == "tf":
            masked_index = tf.where(input_ids == self.tokenizer.mask_token_id).numpy()
        # 判断当前框架（tf或pt），如果是pt，则使用torch.nonzero()方法返回张量中非零元素的索引
        elif self.framework == "pt":
            masked_index = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=False)
        # 如果框架不是tf或pt，则抛出ValueError
        else:
            raise ValueError("Unsupported framework")
        # 返回具有掩码标记的索引数组
        return masked_index

    def _ensure_exactly_one_mask_token(self, input_ids: GenericTensor) -> np.ndarray:
        # 调用get_masked_index()方法获取掩码标记的索引数组
        masked_index = self.get_masked_index(input_ids)
        # 获取数组形状的乘积，判断是否至少有一个掩码标记
        numel = np.prod(masked_index.shape)
        # 如果没有掩码标记，则抛出PipelineException异常
        if numel < 1:
            raise PipelineException(
                "fill-mask",
                self.model.base_model_prefix,
                f"No mask_token ({self.tokenizer.mask_token}) found on the input",
            )

    def ensure_exactly_one_mask_token(self, model_inputs: GenericTensor):
        # 如果model_inputs是列表类型，则遍历列表中每个模型输入
        if isinstance(model_inputs, list):
            for model_input in model_inputs:
                # 调用_ensure_exactly_one_mask_token()方法检查每个模型输入是否至少有一个掩码标记
                self._ensure_exactly_one_mask_token(model_input["input_ids"][0])
        # 如果model_inputs不是列表类型，则直接调用_ensure_exactly_one_mask_token()方法检查模型输入是否至少有一个掩码标记
        else:
            for input_ids in model_inputs["input_ids"]:
                self._ensure_exactly_one_mask_token(input_ids)

    def preprocess(
        self, inputs, return_tensors=None, tokenizer_kwargs=None, **preprocess_parameters
    ) -> Dict[str, GenericTensor]:
        # 如果return_tensors参数未提供，则使用默认的框架类型作为return_tensors
        if return_tensors is None:
            return_tensors = self.framework
        # 如果tokenizer_kwargs参数未提供，则使用空字典
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}

        # 使用tokenizer将输入预处理为模型可以接受的输入张量
        model_inputs = self.tokenizer(inputs, return_tensors=return_tensors, **tokenizer_kwargs)
        # 确保模型输入中至少有一个掩��标记
        self.ensure_exactly_one_mask_token(model_inputs)
        # 返回预处理后的模型输入
        return model_inputs

    def _forward(self, model_inputs):
        # 对模型进行正向传播，获取模型输出
        model_outputs = self.model(**model_inputs)
        # 将模型输入中的input_ids添加到模型输出中
        model_outputs["input_ids"] = model_inputs["input_ids"]
        # 返回模型输出
        return model_outputs
    # 对模型的输出进行后处理，获取前top_k个预测结果
    def postprocess(self, model_outputs, top_k=5, target_ids=None):
        # 如果指定了目标id并且目标id的数量小于top_k，则将top_k设为目标id的数量
        if target_ids is not None and target_ids.shape[0] < top_k:
            top_k = target_ids.shape[0]
        # 获取输入的序列id
        input_ids = model_outputs["input_ids"][0]
        # 获取模型的输出
        outputs = model_outputs["logits"]

        if self.framework == "tf":
            # 使用TensorFlow判断输入序列中的mask位置，并转换为numpy数组
            masked_index = tf.where(input_ids == self.tokenizer.mask_token_id).numpy()[:, 0]

            # 将模型输出转换为numpy数组
            outputs = outputs.numpy()

            # 获取[mask_token]的logits和对应的概率分布
            logits = outputs[0, masked_index, :]
            probs = stable_softmax(logits, axis=-1)   # 使用稳定的softmax函数对logits进行归一化操作
            if target_ids is not None:
                # 如果指定了目标id，则只保留目标id对应的概率值
                probs = tf.gather_nd(tf.squeeze(probs, 0), target_ids.reshape(-1, 1))
                probs = tf.expand_dims(probs, 0)

            # 获取前top_k个概率值和对应的预测
            topk = tf.math.top_k(probs, k=top_k)
            values, predictions = topk.values.numpy(), topk.indices.numpy()
        else:
            # 使用PyTorch判断输入序列中的mask位置，去除张量中的冗余维度
            masked_index = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=False).squeeze(-1)
            # 填充mask在预测结果中的位置，该管道仅支持每个样本中的一个[mask_token]

            # 获取[mask_token]的logits和对应的概率分布
            logits = outputs[0, masked_index, :]
            probs = logits.softmax(dim=-1)   # 对logits进行softmax操作得到概率分布
            if target_ids is not None:
                # 如果指定了目标id，则只保留目标id对应的概率值
                probs = probs[..., target_ids]

            # 获取前top_k个概率值和对应的预测
            values, predictions = probs.topk(top_k)

        # 存储结果
        result = []
        # 判断是否为单个[mask_token]
        single_mask = values.shape[0] == 1
        for i, (_values, _predictions) in enumerate(zip(values.tolist(), predictions.tolist())):
            row = []
            for v, p in zip(_values, _predictions):
                # 复制输入序列，用于替换[mask_token]
                tokens = input_ids.numpy().copy()
                if target_ids is not None:
                    # 对指定的目标id进行编码转换
                    p = target_ids[p].tolist()

                tokens[masked_index[i]] = p
                # 过滤掉填充的标记
                tokens = tokens[np.where(tokens != self.tokenizer.pad_token_id)]
                # 对输出的序列进行解码得到文本
                sequence = self.tokenizer.decode(tokens, skip_special_tokens=single_mask)
                proposition = {"score": v, "token": p, "token_str": self.tokenizer.decode([p]), "sequence": sequence}
                row.append(proposition)
            result.append(row)
        if single_mask:
            return result[0]
        return result
    # 获取目标词的索引
    def get_target_ids(self, targets, top_k=None):
        # 如果目标词是字符串，则转换成列表
        if isinstance(targets, str):
            targets = [targets]
        try:
            # 获取模型tokenizer的词汇表
            vocab = self.tokenizer.get_vocab()
        except Exception:
            # 若出现异常则将词汇表设为空
            vocab = {}
        # 初始化目标词的索引列表
        target_ids = []
        # 遍历目标词列表
        for target in targets:
            # 获取目标词在词汇表中的索引，若不存在则执行下面代码
            id_ = vocab.get(target, None)
            if id_ is None:
                # 使用tokenizer对目标词进行编码，以获取其索引
                input_ids = self.tokenizer(
                    target,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    max_length=1,
                    truncation=True,
                )["input_ids"]
                # 若编码后的结果为空，则记录警告信息，并继续下一个目标词
                if len(input_ids) == 0:
                    logger.warning(
                        f"The specified target token `{target}` does not exist in the model vocabulary. "
                        "We cannot replace it with anything meaningful, ignoring it"
                    )
                    continue
                id_ = input_ids[0]
                # 记录警告信息，提示用户替换目标词以提高性能
                logger.warning(
                    f"The specified target token `{target}` does not exist in the model vocabulary. "
                    f"Replacing with `{self.tokenizer.convert_ids_to_tokens(id_)}`."
                )
            # 将目标词的索引添加到列表中
            target_ids.append(id_)
        # 去除重复的目标词索引
        target_ids = list(set(target_ids))
        # 如果目标词索引为空，则抛出数值错误异常
        if len(target_ids) == 0:
            raise ValueError("At least one target must be provided when passed.")
        # 将目标词索引数组转换成NumPy数组
        target_ids = np.array(target_ids)
        # 返回目标词索引数组
        return target_ids

    # 对参数进行处理
    def _sanitize_parameters(self, top_k=None, targets=None, tokenizer_kwargs=None):
        preprocess_params = {}

        # 若存在tokenizer_kwargs，则添加到预处理参数中
        if tokenizer_kwargs is not None:
            preprocess_params["tokenizer_kwargs"] = tokenizer_kwargs

        postprocess_params = {}

        # 若存在目标词，则获取其索引并添加到后处理参数中
        if targets is not None:
            target_ids = self.get_target_ids(targets, top_k)
            postprocess_params["target_ids"] = target_ids

        # 若存在top_k，则添加到后处理参数中
        if top_k is not None:
            postprocess_params["top_k"] = top_k

        # 若模型tokenizer不包含mask_token_id，则抛出异常
        if self.tokenizer.mask_token_id is None:
            raise PipelineException(
                "fill-mask", self.model.base_model_prefix, "The tokenizer does not define a `mask_token`."
            )
        # 返回预处理参数、空字典和后处理参数
        return preprocess_params, {}, postprocess_params
    # 定义一个方法，用于填充输入文本中的被屏蔽标记
    def __call__(self, inputs, *args, **kwargs):
        """
        Fill the masked token in the text(s) given as inputs.

        Args:
            args (`str` or `List[str]`):
                One or several texts (or one list of prompts) with masked tokens.
            targets (`str` or `List[str]`, *optional*):
                When passed, the model will limit the scores to the passed targets instead of looking up in the whole
                vocab. If the provided targets are not in the model vocab, they will be tokenized and the first
                resulting token will be used (with a warning, and that might be slower).
            top_k (`int`, *optional*):
                When passed, overrides the number of predictions to return.

        Return:
            A list or a list of list of `dict`: Each result comes as list of dictionaries with the following keys:

            - **sequence** (`str`) -- The corresponding input with the mask token prediction.
            - **score** (`float`) -- The corresponding probability.
            - **token** (`int`) -- The predicted token id (to replace the masked one).
            - **token_str** (`str`) -- The predicted token (to replace the masked one).
        """
        # 调用父类方法，在输入上执行模型
        outputs = super().__call__(inputs, **kwargs)
        # 当输入为列表且长度为1时，返回结果的第一个元素
        if isinstance(inputs, list) and len(inputs) == 1:
            return outputs[0]
        # 否则返回全部结果
        return outputs
```