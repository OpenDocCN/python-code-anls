# `.\pipelines\text2text_generation.py`

```py
import enum  # 导入 Python 标准库中的 enum 模块，用于定义枚举类型
import warnings  # 导入 Python 标准库中的 warnings 模块，用于警告处理

from ..tokenization_utils import TruncationStrategy  # 从上级目录的 tokenization_utils 模块导入 TruncationStrategy 类
from ..utils import add_end_docstrings, is_tf_available, is_torch_available, logging  # 从上级目录的 utils 模块导入函数和变量
from .base import Pipeline, build_pipeline_init_args  # 从当前目录的 base 模块导入 Pipeline 类和函数 build_pipeline_init_args

if is_tf_available():  # 如果 TensorFlow 可用
    import tensorflow as tf  # 导入 TensorFlow 库

    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES  # 从上级目录的 models 中导入 TensorFlow 自动化建模模块中的常量

if is_torch_available():  # 如果 PyTorch 可用
    from ..models.auto.modeling_auto import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES  # 从上级目录的 models 中导入 PyTorch 自动化建模模块中的常量

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

class ReturnType(enum.Enum):  # 定义 ReturnType 枚举类
    TENSORS = 0  # 枚举成员：TENSORS 对应值为 0
    TEXT = 1  # 枚举成员：TEXT 对应值为 1

@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True))  # 装饰器，用于为类添加文档字符串的结束部分
class Text2TextGenerationPipeline(Pipeline):  # 文本到文本生成管道类，继承自 Pipeline 类
    """
    Pipeline for text to text generation using seq2seq models.

    Example:

    ```
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

    ```
    text2text_generator = pipeline("text2text-generation")
    text2text_generator("question: What is 42 ? context: 42 is the answer to life, the universe and everything")
    ```
    """

    # Used in the return key of the pipeline.
    return_name = "generated"  # 定义管道返回结果的键名为 "generated"

    def __init__(self, *args, **kwargs):  # 初始化方法
        super().__init__(*args, **kwargs)  # 调用父类的初始化方法

        self.check_model_type(
            TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES  # 如果使用 TensorFlow 框架，检查模型类型是否在 TensorFlow 映射名称中
            if self.framework == "tf"  # 如果当前框架为 TensorFlow
            else MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES  # 否则检查模型类型是否在 PyTorch 映射名称中
        )

    def _sanitize_parameters(  # 定义私有方法 _sanitize_parameters，用于处理生成器参数
        self,
        return_tensors=None,  # 返回张量数据
        return_text=None,  # 返回文本数据
        return_type=None,  # 返回类型
        clean_up_tokenization_spaces=None,  # 清理分词空格
        truncation=None,  # 截断
        stop_sequence=None,  # 停止序列
        **generate_kwargs,  # 其余生成参数
    ):
        preprocess_params = {}
        if truncation is not None:
            preprocess_params["truncation"] = truncation
        # 将生成参数设置为 generate_kwargs
        forward_params = generate_kwargs

        postprocess_params = {}
        # 如果 return_tensors 不为 None 且 return_type 为 None，则根据 return_tensors 的值确定 return_type
        if return_tensors is not None and return_type is None:
            return_type = ReturnType.TENSORS if return_tensors else ReturnType.TEXT
        # 如果 return_type 不为 None，则设置 postprocess_params 中的 return_type
        if return_type is not None:
            postprocess_params["return_type"] = return_type

        # 如果 clean_up_tokenization_spaces 不为 None，则设置 postprocess_params 中的 clean_up_tokenization_spaces
        if clean_up_tokenization_spaces is not None:
            postprocess_params["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces

        # 如果 stop_sequence 不为 None，则根据 stop_sequence 编码并设置生成参数中的 eos_token_id
        if stop_sequence is not None:
            stop_sequence_ids = self.tokenizer.encode(stop_sequence, add_special_tokens=False)
            if len(stop_sequence_ids) > 1:
                # 如果 stop_sequence_ids 包含多个 token，则发出警告
                warnings.warn(
                    "Stopping on a multiple token sequence is not yet supported on transformers. The first token of"
                    " the stop sequence will be used as the stop sequence string in the interim."
                )
            generate_kwargs["eos_token_id"] = stop_sequence_ids[0]

        # 返回预处理参数、生成参数和后处理参数
        return preprocess_params, forward_params, postprocess_params

    def check_inputs(self, input_length: int, min_length: int, max_length: int):
        """
        Checks whether there might be something wrong with given input with regard to the model.
        """
        return True

    def _parse_and_tokenize(self, *args, truncation):
        # 获取模型配置中的前缀，如果不存在则为空字符串
        prefix = self.model.config.prefix if self.model.config.prefix is not None else ""
        # 如果 args[0] 是列表，则检查 tokenizer 是否有 pad_token_id，没有则抛出 ValueError
        if isinstance(args[0], list):
            if self.tokenizer.pad_token_id is None:
                raise ValueError("Please make sure that the tokenizer has a pad_token_id when using a batch input")
            # 将列表中的每个元素加上前缀 prefix，并设置 padding 为 True
            args = ([prefix + arg for arg in args[0]],)
            padding = True

        # 如果 args[0] 是字符串，则将其加上前缀 prefix，并设置 padding 为 False
        elif isinstance(args[0], str):
            args = (prefix + args[0],)
            padding = False
        else:
            # 如果 args[0] 不是字符串也不是列表，则抛出 ValueError
            raise ValueError(
                f" `args[0]`: {args[0]} have the wrong format. The should be either of type `str` or type `list`"
            )
        # 使用 tokenizer 对参数进行解析和标记化，根据参数设置 padding、truncation、return_tensors 和 framework
        inputs = self.tokenizer(*args, padding=padding, truncation=truncation, return_tensors=self.framework)
        # 如果 inputs 中存在 "token_type_ids" 键，则将其删除（这是 tokenizers 生成的无效生成参数）
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        return inputs
    def __call__(self, *args, **kwargs):
        r"""
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

        # 调用父类的方法处理输入参数并生成结果
        result = super().__call__(*args, **kwargs)
        
        # 如果输入参数是一个列表且所有元素都是字符串，并且每个结果都是长度为1的列表，则返回结果中的第一个元素
        if (
            isinstance(args[0], list)
            and all(isinstance(el, str) for el in args[0])
            and all(len(res) == 1 for res in result)
        ):
            return [res[0] for res in result]
        
        # 否则直接返回生成的结果
        return result

    def preprocess(self, inputs, truncation=TruncationStrategy.DO_NOT_TRUNCATE, **kwargs):
        # 解析和标记化输入文本，并根据指定的截断策略进行处理
        inputs = self._parse_and_tokenize(inputs, truncation=truncation, **kwargs)
        
        # 返回预处理后的输入
        return inputs
    # 定义一个方法 `_forward`，用于模型推理的前向过程，接受模型输入和生成参数
    def _forward(self, model_inputs, **generate_kwargs):
        # 根据选择的深度学习框架确定输入张量的形状
        if self.framework == "pt":
            in_b, input_length = model_inputs["input_ids"].shape
        elif self.framework == "tf":
            # 使用 TensorFlow 的 API 获取张量的形状，并转换为 NumPy 数组
            in_b, input_length = tf.shape(model_inputs["input_ids"]).numpy()

        # 检查输入的长度是否在指定的最小和最大长度范围内
        self.check_inputs(
            input_length,
            generate_kwargs.get("min_length", self.model.config.min_length),
            generate_kwargs.get("max_length", self.model.config.max_length),
        )
        # 调用模型的生成方法，生成输出序列的标识符
        output_ids = self.model.generate(**model_inputs, **generate_kwargs)
        # 计算输出张量的第一维度大小
        out_b = output_ids.shape[0]
        # 根据选择的深度学习框架进行输出张量的形状重塑
        if self.framework == "pt":
            output_ids = output_ids.reshape(in_b, out_b // in_b, *output_ids.shape[1:])
        elif self.framework == "tf":
            output_ids = tf.reshape(output_ids, (in_b, out_b // in_b, *output_ids.shape[1:]))
        # 返回包含输出标识符的字典
        return {"output_ids": output_ids}

    # 定义一个方法 `postprocess`，用于处理模型输出，根据返回类型进行后处理
    def postprocess(self, model_outputs, return_type=ReturnType.TEXT, clean_up_tokenization_spaces=False):
        # 初始化记录列表，用于存储处理后的结果
        records = []
        # 遍历模型输出中的每个输出标识符序列
        for output_ids in model_outputs["output_ids"][0]:
            # 根据返回类型不同，创建不同形式的记录对象
            if return_type == ReturnType.TENSORS:
                record = {f"{self.return_name}_token_ids": output_ids}
            elif return_type == ReturnType.TEXT:
                # 使用分词器解码标识符序列，生成文本，并根据参数进行清理处理
                record = {
                    f"{self.return_name}_text": self.tokenizer.decode(
                        output_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    )
                }
            # 将记录对象添加到记录列表中
            records.append(record)
        # 返回处理后的记录列表
        return records
# 使用装饰器 `@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True))` 对类进行装饰，添加初始化参数和文档字符串。
class SummarizationPipeline(Text2TextGenerationPipeline):
    """
    Summarize news articles and other documents.

    This summarizing pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"summarization"`.

    The models that this pipeline can use are models that have been fine-tuned on a summarization task, which is
    currently, '*bart-large-cnn*', '*google-t5/t5-small*', '*google-t5/t5-base*', '*google-t5/t5-large*', '*google-t5/t5-3b*', '*google-t5/t5-11b*'. See the up-to-date
    list of available models on [huggingface.co/models](https://huggingface.co/models?filter=summarization). For a list
    of available parameters, see the [following
    documentation](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.generation.GenerationMixin.generate)

    Usage:

    ```
    # use bart in pytorch
    summarizer = pipeline("summarization")
    summarizer("An apple a day, keeps the doctor away", min_length=5, max_length=20)

    # use t5 in tf
    summarizer = pipeline("summarization", model="google-t5/t5-base", tokenizer="google-t5/t5-base", framework="tf")
    summarizer("An apple a day, keeps the doctor away", min_length=5, max_length=20)
    ```
    """

    # 定义用于标识返回摘要的键名
    return_name = "summary"

    def __call__(self, *args, **kwargs):
        r"""
        Summarize the text(s) given as inputs.

        Args:
            documents (*str* or `List[str]`):
                One or several articles (or one list of articles) to summarize.
            return_text (`bool`, *optional*, defaults to `True`):
                Whether or not to include the decoded texts in the outputs
            return_tensors (`bool`, *optional*, defaults to `False`):
                Whether or not to include the tensors of predictions (as token indices) in the outputs.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the potential extra spaces in the text output.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework [here](./model#generative-models)).

        Return:
            A list or a list of list of `dict`: Each result comes as a dictionary with the following keys:

            - **summary_text** (`str`, present when `return_text=True`) -- The summary of the corresponding input.
            - **summary_token_ids** (`torch.Tensor` or `tf.Tensor`, present when `return_tensors=True`) -- The token
              ids of the summary.
        """
        # 调用父类的 `__call__` 方法，传递所有参数和关键字参数
        return super().__call__(*args, **kwargs)
    def check_inputs(self, input_length: int, min_length: int, max_length: int) -> bool:
        """
        Checks whether there might be something wrong with given input with regard to the model.
        """
        # 检查最大长度是否小于最小长度，如果是则记录警告日志
        if max_length < min_length:
            logger.warning(f"Your min_length={min_length} must be inferior than your max_length={max_length}.")

        # 检查输入长度是否小于最大长度，如果是则记录警告日志
        if input_length < max_length:
            logger.warning(
                f"Your max_length is set to {max_length}, but your input_length is only {input_length}. Since this is "
                "a summarization task, where outputs shorter than the input are typically wanted, you might "
                f"consider decreasing max_length manually, e.g. summarizer('...', max_length={input_length//2})"
            )
# 使用装饰器为类添加文档字符串，并调用函数`build_pipeline_init_args`作为参数
@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True))
# 定义一个翻译管道类，继承自`Text2TextGenerationPipeline`
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

    ```
    en_fr_translator = pipeline("translation_en_to_fr")
    en_fr_translator("How old are you?")
    ```
    """

    # 定义一个类变量，表示此管道的返回键
    return_name = "translation"

    # 检查输入长度是否符合要求的方法
    def check_inputs(self, input_length: int, min_length: int, max_length: int):
        # 如果输入长度超过最大长度的90%，发出警告信息
        if input_length > 0.9 * max_length:
            logger.warning(
                f"Your input_length: {input_length} is bigger than 0.9 * max_length: {max_length}. You might consider "
                "increasing your max_length manually, e.g. translator('...', max_length=400)"
            )
        # 返回True表示检查通过
        return True

    # 数据预处理方法，根据条件使用不同的处理方式
    def preprocess(self, *args, truncation=TruncationStrategy.DO_NOT_TRUNCATE, src_lang=None, tgt_lang=None):
        # 如果存在`_build_translation_inputs`方法，则调用该方法构建翻译输入
        if getattr(self.tokenizer, "_build_translation_inputs", None):
            return self.tokenizer._build_translation_inputs(
                *args, return_tensors=self.framework, truncation=truncation, src_lang=src_lang, tgt_lang=tgt_lang
            )
        else:
            # 否则调用父类的`_parse_and_tokenize`方法进行分析和标记化处理
            return super()._parse_and_tokenize(*args, truncation=truncation)

    # 参数清理方法，用于处理源语言和目标语言的参数，并返回处理后的参数字典
    def _sanitize_parameters(self, src_lang=None, tgt_lang=None, **kwargs):
        # 调用父类的方法，获取预处理、前向和后处理的参数字典
        preprocess_params, forward_params, postprocess_params = super()._sanitize_parameters(**kwargs)
        # 如果指定了源语言，则将其添加到预处理参数中
        if src_lang is not None:
            preprocess_params["src_lang"] = src_lang
        # 如果指定了目标语言，则将其添加到预处理参数中
        if tgt_lang is not None:
            preprocess_params["tgt_lang"] = tgt_lang
        # 如果既未指定源语言也未指定目标语言，则尝试从任务标识中解析出语言信息
        if src_lang is None and tgt_lang is None:
            # 向后兼容性，优先使用直接参数
            task = kwargs.get("task", self.task)
            items = task.split("_")
            if task and len(items) == 4:
                # translation, XX, to YY 格式
                preprocess_params["src_lang"] = items[1]
                preprocess_params["tgt_lang"] = items[3]
        # 返回清理后的参数字典
        return preprocess_params, forward_params, postprocess_params
    # 重写 `__call__` 方法，使其能够将输入的文本进行翻译
    def __call__(self, *args, **kwargs):
        # 翻译输入的文本
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
        # 调用父类的 `__call__` 方法，并传入参数
        return super().__call__(*args, **kwargs)
```