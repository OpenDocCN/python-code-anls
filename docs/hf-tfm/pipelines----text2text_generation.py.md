# `.\transformers\pipelines\text2text_generation.py`

```
# 导入必要的库
import enum  # 用于定义枚举类型
import warnings  # 用于处理警告信息

# 导入相关的模块和函数
from ..tokenization_utils import TruncationStrategy  # 导入截断策略
from ..utils import add_end_docstrings, is_tf_available, is_torch_available, logging  # 导入函数和工具
from .base import PIPELINE_INIT_ARGS, Pipeline  # 从基类中导入必要的参数和类


# 如果可以使用 TensorFlow，则导入 TensorFlow 库
if is_tf_available():
    import tensorflow as tf  # 导入 TensorFlow 库

    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES  # 导入 TF 下的序列到序列模型映射

# 如果可以使用 PyTorch，则导入 PyTorch 库
if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES  # 导入 PyTorch 下的序列到序列模型映射

# 获取日志记录器对象
logger = logging.get_logger(__name__)


# 枚举类型，用于指示返回类型
class ReturnType(enum.Enum):
    TENSORS = 0  # 返回张量类型
    TEXT = 1  # 返回文本类型


# 使用装饰器添加文档字符串，包括初始化参数
@add_end_docstrings(PIPELINE_INIT_ARGS)
# 文本到文本生成管道类，继承自 Pipeline 基类
class Text2TextGenerationPipeline(Pipeline):
    """
    Pipeline for text to text generation using seq2seq models.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> generator = pipeline(model="mrm8488/t5-base-finetuned-question-generation-ap")
    >>> generator(
    ...     "answer: Manuel context: Manuel has created RuPERTa-base with the support of HF-Transformers and Google"
    ... )
    [{'generated_text': 'question: Who created the RuPERTa-base?'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial). You can pass text
    generation parameters to this pipeline to control stopping criteria, decoding strategy, and more. Learn more about
    text generation parameters in [Text generation strategies](../generation_strategies) and [Text
    generation](text_generation).

    This Text2TextGenerationPipeline pipeline can currently be loaded from [`pipeline`] using the following task
    identifier: `"text2text-generation"`.

    The models that this pipeline can use are models that have been fine-tuned on a translation task. See the
    up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=text2text-generation). For a list of available
    parameters, see the [following
    documentation](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.generation.GenerationMixin.generate)

    Usage:

    ```python
    text2text_generator = pipeline("text2text-generation")
    text2text_generator("question: What is 42 ? context: 42 is the answer to life, the universe and everything")
    ```"""

    # 用于表示返回结果中生成文本的键名
    return_name = "generated"

    # 初始化方法
    def __init__(self, *args, **kwargs):
        # 调用基类的初始化方法
        super().__init__(*args, **kwargs)

        # 检查模型类型是否正确
        self.check_model_type(
            TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
            if self.framework == "tf"
            else MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
        )

    # 用于清理参数的私有方法
    def _sanitize_parameters(
        self,
        return_tensors=None,
        return_text=None,
        return_type=None,
        clean_up_tokenization_spaces=None,
        truncation=None,
        stop_sequence=None,
        **generate_kwargs,
        ):
        # 初始化预处理参数字典
        preprocess_params = {}
        # 如果截断参数不为空，则将其加入预处理参数中
        if truncation is not None:
            preprocess_params["truncation"] = truncation

        # 将生成参数直接赋值给前向参数
        forward_params = generate_kwargs

        # 初始化后处理参数字典
        postprocess_params = {}
        # 如果返回张量不为空且返回类型为空，则根据返回张量的值确定返回类型是张量还是文本
        if return_tensors is not None and return_type is None:
            return_type = ReturnType.TENSORS if return_tensors else ReturnType.TEXT
        # 如果返回类型不为空，则将其加入后处理参数中
        if return_type is not None:
            postprocess_params["return_type"] = return_type

        # 如果去除标记化空格参数不为空，则将其加入后处理参数中
        if clean_up_tokenization_spaces is not None:
            postprocess_params["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces

        # 如果停止序列不为空，则将停止序列编码成ID，并将第一个ID作为生成参数字典中的EOS标记ID
        if stop_sequence is not None:
            stop_sequence_ids = self.tokenizer.encode(stop_sequence, add_special_tokens=False)
            # 如果停止序列ID数量大于1，则发出警告
            if len(stop_sequence_ids) > 1:
                warnings.warn(
                    "Stopping on a multiple token sequence is not yet supported on transformers. The first token of"
                    " the stop sequence will be used as the stop sequence string in the interim."
                )
            generate_kwargs["eos_token_id"] = stop_sequence_ids[0]

        # 返回预处理参数、前向参数和后处理参数
        return preprocess_params, forward_params, postprocess_params

    # 检查输入长度是否符合模型要求
    def check_inputs(self, input_length: int, min_length: int, max_length: int):
        """
        Checks whether there might be something wrong with given input with regard to the model.
        """
        return True

    # 解析并标记化输入参数
    def _parse_and_tokenize(self, *args, truncation):
        # 获取模型的前缀
        prefix = self.model.config.prefix if self.model.config.prefix is not None else ""
        # 如果输入参数是列表，则处理每个元素，并添加前缀；如果分词器没有填充标记ID，则报错
        if isinstance(args[0], list):
            if self.tokenizer.pad_token_id is None:
                raise ValueError("Please make sure that the tokenizer has a pad_token_id when using a batch input")
            args = ([prefix + arg for arg in args[0]],)
            padding = True

        # 如果输入参数是字符串，则添加前缀，并不进行填充
        elif isinstance(args[0], str):
            args = (prefix + args[0],)
            padding = False
        else:
            # 如果输入参数格式错误，则报错
            raise ValueError(
                f" `args[0]`: {args[0]} have the wrong format. The should be either of type `str` or type `list`"
            )
        # 使用分词器处理输入参数，并返回设置了填充、截断和返回张量类型的输入
        inputs = self.tokenizer(*args, padding=padding, truncation=truncation, return_tensors=self.framework)
        # 使用分词器生成的“token_type_ids”是无效的生成参数，删除它
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        return inputs
    # 定义一个特殊方法，用于对输入文本进行编码并生成输出文本
    def __call__(self, *args, **kwargs):
        """
        Generate the output text(s) using text(s) given as inputs.

        Args:
            args (`str` or `List[str]`):
                Input text for the encoder.
            return_tensors (`bool`, *optional*, defaults to `False`):
                Whether or not to include the tensors of predictions (as token indices) in the outputs.
            return_text (`bool`, *optional*, defaults to `True`):
                Whether or not to include the decoded texts in the outputs.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the potential extra spaces in the text output.
            truncation (`TruncationStrategy`, *optional*, defaults to `TruncationStrategy.DO_NOT_TRUNCATE`):
                The truncation strategy for the tokenization within the pipeline. `TruncationStrategy.DO_NOT_TRUNCATE`
                (default) will never truncate, but it is sometimes desirable to truncate the input to fit the model's
                max_length instead of throwing an error down the line.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework [here](./model#generative-models)).

        Return:
            A list or a list of list of `dict`: Each result comes as a dictionary with the following keys:

            - **generated_text** (`str`, present when `return_text=True`) -- The generated text.
            - **generated_token_ids** (`torch.Tensor` or `tf.Tensor`, present when `return_tensors=True`) -- The token
              ids of the generated text.
        """

        # 调用父类方法对输入进行处理
        result = super().__call__(*args, **kwargs)
        
        # 检查是否需要返回结果列表中每个字典的第一个元素
        if (
            isinstance(args[0], list)
            and all(isinstance(el, str) for el in args[0])
            and all(len(res) == 1 for res in result)
        ):
            return [res[0] for res in result]
        
        # 返回处理后的结果
        return result

    # 预处理输入数据，包括解析和标记化
    def preprocess(self, inputs, truncation=TruncationStrategy.DO_NOT_TRUNCATE, **kwargs):
        inputs = self._parse_and_tokenize(inputs, truncation=truncation, **kwargs)
        return inputs
    # 定义前向传播方法
    def _forward(self, model_inputs, **generate_kwargs):
        # 获取输入数据的批量大小和序列长度
        if self.framework == "pt":
            in_b, input_length = model_inputs["input_ids"].shape
        elif self.framework == "tf":
            in_b, input_length = tf.shape(model_inputs["input_ids"]).numpy()
    
        # 检查输入长度是否符合要求
        self.check_inputs(
            input_length,
            generate_kwargs.get("min_length", self.model.config.min_length),
            generate_kwargs.get("max_length", self.model.config.max_length),
        )
    
        # 调用模型生成输出
        output_ids = self.model.generate(**model_inputs, **generate_kwargs)
    
        # 整理输出数据的形状
        out_b = output_ids.shape[0]
        if self.framework == "pt":
            output_ids = output_ids.reshape(in_b, out_b // in_b, *output_ids.shape[1:])
        elif self.framework == "tf":
            output_ids = tf.reshape(output_ids, (in_b, out_b // in_b, *output_ids.shape[1:]))
    
        # 返回生成的输出
        return {"output_ids": output_ids}
    
    # 定义后处理方法
    def postprocess(self, model_outputs, return_type=ReturnType.TEXT, clean_up_tokenization_spaces=False):
        records = []
        # 遍历每个输出序列
        for output_ids in model_outputs["output_ids"][0]:
            # 根据返回类型构建结果字典
            if return_type == ReturnType.TENSORS:
                record = {f"{self.return_name}_token_ids": output_ids}
            elif return_type == ReturnType.TEXT:
                record = {
                    f"{self.return_name}_text": self.tokenizer.decode(
                        output_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    )
                }
            records.append(record)
        # 返回结果列表
        return records
# 将SummarizationPipeline类应用于PIPELINE_INIT_ARGS参数
@add_end_docstrings(PIPELINE_INIT_ARGS)
class SummarizationPipeline(Text2TextGenerationPipeline):
    """
    Summarize news articles and other documents.

    This summarizing pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"summarization"`.

    The models that this pipeline can use are models that have been fine-tuned on a summarization task, which is
    currently, '*bart-large-cnn*', '*t5-small*', '*t5-base*', '*t5-large*', '*t5-3b*', '*t5-11b*'. See the up-to-date
    list of available models on [huggingface.co/models](https://huggingface.co/models?filter=summarization). For a list
    of available parameters, see the [following
    documentation](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.generation.GenerationMixin.generate)

    Usage:

    ```python
    # use bart in pytorch
    summarizer = pipeline("summarization")
    summarizer("An apple a day, keeps the doctor away", min_length=5, max_length=20)

    # use t5 in tf
    summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
    summarizer("An apple a day, keeps the doctor away", min_length=5, max_length=20)
    ```"""

    # Used in the return key of the pipeline.
    return_name = "summary"

    # 调用SummarizationPipeline类实例时执行的方法
    def __call__(self, *args, **kwargs):
        # Summarize the text(s) given as inputs.

        # 参数说明:
        # documents (*str* or `List[str]`):
        #     One or several articles (or one list of articles) to summarize.
        # return_text (`bool`, *optional*, defaults to `True`):
        #     Whether or not to include the decoded texts in the outputs
        # return_tensors (`bool`, *optional*, defaults to `False`):
        #     Whether or not to include the tensors of predictions (as token indices) in the outputs.
        # clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
        #     Whether or not to clean up the potential extra spaces in the text output.
        # generate_kwargs:
        #     Additional keyword arguments to pass along to the generate method of the model (see the generate method
        #     corresponding to your framework [here](./model#generative-models)).

        # Return:
        # A list or a list of list of `dict`: Each result comes as a dictionary with the following keys:
        # - **summary_text** (`str`, present when `return_text=True`) -- The summary of the corresponding input.
        # - **summary_token_ids** (`torch.Tensor` or `tf.Tensor`, present when `return_tensors=True`) -- The token
        #   ids of the summary.
        return super().__call__(*args, **kwargs)
    # 检查输入长度是否满足模型的要求
    def check_inputs(self, input_length: int, min_length: int, max_length: int) -> bool:
        """
        Checks whether there might be something wrong with given input with regard to the model.
        """
        # 如果最大长度小于最小长度，记录警告信息
        if max_length < min_length:
            logger.warning(f"Your min_length={min_length} must be inferior than your max_length={max_length}.")

        # 如果输入长度小于最大长度，记录警告信息
        if input_length < max_length:
            logger.warning(
                f"Your max_length is set to {max_length}, but your input_length is only {input_length}. Since this is "
                "a summarization task, where outputs shorter than the input are typically wanted, you might "
                f"consider decreasing max_length manually, e.g. summarizer('...', max_length={input_length//2})"
            )
@add_end_docstrings(PIPELINE_INIT_ARGS)
class TranslationPipeline(Text2TextGenerationPipeline):
    """
    Translates from one language to another.

    This translation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"translation_xx_to_yy"`.

    The models that this pipeline can use are models that have been fine-tuned on a translation task. See the
    up-to-date list of available models on [huggingface.co/models](https://huggingface.co/models?filter=translation).
    For a list of available parameters, see the [following
    documentation](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.generation.GenerationMixin.generate)

    Usage:

    ```python
    en_fr_translator = pipeline("translation_en_to_fr")
    en_fr_translator("How old are you?")
    ```"""

    # 用于 pipeline 返回键的名称
    return_name = "translation"

    # 检查输入的长度是否符合要求
    def check_inputs(self, input_length: int, min_length: int, max_length: int):
        if input_length > 0.9 * max_length:
            # 若输入长度大于最大长度的 90%，发出警告
            logger.warning(
                f"Your input_length: {input_length} is bigger than 0.9 * max_length: {max_length}. You might consider "
                "increasing your max_length manually, e.g. translator('...', max_length=400)"
            )
        # 返回 True 表示输入合法
        return True

    # 预处理函数，用于准备输入数据
    def preprocess(self, *args, truncation=TruncationStrategy.DO_NOT_TRUNCATE, src_lang=None, tgt_lang=None):
        # 检查 tokenizer 是否实现了 "_build_translation_inputs" 方法
        if getattr(self.tokenizer, "_build_translation_inputs", None):
            # 若实现了，调用 "_build_translation_inputs" 方法准备输入数据
            return self.tokenizer._build_translation_inputs(
                *args, return_tensors=self.framework, truncation=truncation, src_lang=src_lang, tgt_lang=tgt_lang
            )
        else:
            # 若未实现，调用基类的 "_parse_and_tokenize" 方法进行默认处理
            return super()._parse_and_tokenize(*args, truncation=truncation)

    # 清理参数，确保参数格式正确
    def _sanitize_parameters(self, src_lang=None, tgt_lang=None, **kwargs):
        # 调用基类的 "_sanitize_parameters" 方法清理参数
        preprocess_params, forward_params, postprocess_params = super()._sanitize_parameters(**kwargs)
        # 若指定了源语言，则更新预处理参数中的源语言
        if src_lang is not None:
            preprocess_params["src_lang"] = src_lang
        # 若指定了目标语言，则更新预处理参数中的目标语言
        if tgt_lang is not None:
            preprocess_params["tgt_lang"] = tgt_lang
        # 若未指定源语言和目标语言，则尝试从任务名称中解析
        if src_lang is None and tgt_lang is None:
            # 向后兼容，优先使用直接的参数
            task = kwargs.get("task", self.task)
            items = task.split("_")
            if task and len(items) == 4:
                # translation, XX, to YY
                preprocess_params["src_lang"] = items[1]
                preprocess_params["tgt_lang"] = items[3]
        # 返回清理后的参数
        return preprocess_params, forward_params, postprocess_params
    # 定义 __call__ 方法，用于调用翻译模型进行文本翻译
    def __call__(self, *args, **kwargs):
        # 翻译给定的文本
        r"""
        Translate the text(s) given as inputs.

        Args:
            args (`str` or `List[str]`):
                Texts to be translated.
            return_tensors (`bool`, *optional*, defaults to `False`):
                Whether or not to include the tensors of predictions (as token indices) in the outputs.
            return_text (`bool`, *optional*, defaults to `True`):
                Whether or not to include the decoded texts in the outputs.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the potential extra spaces in the text output.
            src_lang (`str`, *optional*):
                The language of the input. Might be required for multilingual models. Will not have any effect for
                single pair translation models
            tgt_lang (`str`, *optional*):
                The language of the desired output. Might be required for multilingual models. Will not have any effect
                for single pair translation models
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework [here](./model#generative-models)).

        Return:
            A list or a list of list of `dict`: Each result comes as a dictionary with the following keys:

            - **translation_text** (`str`, present when `return_text=True`) -- The translation.
            - **translation_token_ids** (`torch.Tensor` or `tf.Tensor`, present when `return_tensors=True`) -- The
              token ids of the translation.
        """
        # 调用父类的 __call__ 方法，传入参数并返回结果
        return super().__call__(*args, **kwargs)
```