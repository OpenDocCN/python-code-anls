# `.\transformers\pipelines\text_generation.py`

```py
# 导入 enum、warnings 模块
import enum
import warnings
# 从 ..utils 模块导入 add_end_docstrings、is_tf_available 和 is_torch_available 函数
from ..utils import add_end_docstrings, is_tf_available, is_torch_available
# 从 .base 模块导入 PIPELINE_INIT_ARGS、Pipeline 类
from .base import PIPELINE_INIT_ARGS, Pipeline

# 如果有 PyTorch 被导入，从 ..models.auto.modeling_auto 模块导入 MODEL_FOR_CAUSAL_LM_MAPPING_NAMES 参数
if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
# 如果有 TensorFlow 被导入，导入 tensorflow 模块
if is_tf_available():
    import tensorflow as tf
    # 从 ..models.auto.modeling_tf_auto 模块导入 TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES 参数

    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

# 定义 ReturnType 枚举类型
class ReturnType(enum.Enum):
    TENSORS = 0
    NEW_TEXT = 1
    FULL_TEXT = 2

# 给 TextGenerationPipeline 类增加文档字符串
@add_end_docstrings(PIPELINE_INIT_ARGS)
class TextGenerationPipeline(Pipeline):
    """
    Language generation pipeline using any `ModelWithLMHead`. This pipeline predicts the words that will follow a
    specified text prompt.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> generator = pipeline(model="gpt2")
    >>> generator("I can't believe you did such a ", do_sample=False)
    [{'generated_text': "I can't believe you did such a icky thing to me. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I"}]

    >>> # These parameters will return suggestions, and only the newly created text making it easier for prompting suggestions.
    >>> outputs = generator("My tart needs some", num_return_sequences=4, return_full_text=False)
    ```py

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial). You can pass text
    generation parameters to this pipeline to control stopping criteria, decoding strategy, and more. Learn more about
    text generation parameters in [Text generation strategies](../generation_strategies) and [Text
    generation](text_generation).

    This language generation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"text-generation"`.

    The models that this pipeline can use are models that have been trained with an autoregressive language modeling
    objective, which includes the uni-directional models in the library (e.g. gpt2). See the list of available models
    on [huggingface.co/models](https://huggingface.co/models?filter=text-generation).
    """

    # Prefix text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
    # in https://github.com/rusiaaman/XLNet-gen#methodology
    # and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e

    XL_PREFIX = """
    In 1991, the remains of Russian Tsar Nicholas II and his family (except for Alexei and Maria) are discovered. The
    voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the remainder of the story. 1883 Western
    Siberia, a young Grigori Rasputin is asked by his father and a group of men to perform magic. Rasputin has a vision
    and denounces one of the men as a horse thief. Although his father initially slaps him for making such an
    accusation, Rasputin watches as the man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
    """
    定义一个名为 "Rasputin" 的类，表示拉斯普京这个人物
    """

    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
        # 检查模型类型是否符合要求
        self.check_model_type(
            TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES if self.framework == "tf" else MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        )
        if "prefix" not in self._preprocess_params:
            # 检查是否存在特定的前缀
            # 这里包含了逻辑复杂的默认值设定
            # 同时定义了 preprocess_kwargs 和 generate_kwargs
            # 所以无法将它们放在各自的方法中
            prefix = None
            if self.model.config.prefix is not None:
                prefix = self.model.config.prefix
            if prefix is None and self.model.__class__.__name__ in [
                "XLNetLMHeadModel",
                "TransfoXLLMHeadModel",
                "TFXLNetLMHeadModel",
                "TFTransfoXLLMHeadModel",
            ]:
                # 对于 XLNet 和 TransformerXL 模型，添加前缀以提供更多状态给模型
                prefix = self.XL_PREFIX
            if prefix is not None:
                # 重新计算与前缀相关的一些 generate_kwargs
                preprocess_params, forward_params, _ = self._sanitize_parameters(prefix=prefix, **self._forward_params)
                self._preprocess_params = {**self._preprocess_params, **preprocess_params}
                self._forward_params = {**self._forward_params, **forward_params}

    def _sanitize_parameters(
        self,
        return_full_text=None,
        return_tensors=None,
        return_text=None,
        return_type=None,
        clean_up_tokenization_spaces=None,
        prefix=None,
        handle_long_generation=None,
        stop_sequence=None,
        add_special_tokens=False,
        truncation=None,
        padding=False,
        max_length=None,
        **generate_kwargs,
    ):
    """
    内部方法，用于清理和规范化参数
    """
        ):
        # 准备预处理参数字典，包括是否添加特殊标记、截断、填充、最大长度等
        preprocess_params = {
            "add_special_tokens": add_special_tokens,
            "truncation": truncation,
            "padding": padding,
            "max_length": max_length,
        }
        # 如果设置了最大长度，则更新生成参数中的最大长度
        if max_length is not None:
            generate_kwargs["max_length"] = max_length

        # 如果设置了前缀，则在预处理参数中添加前缀
        if prefix is not None:
            preprocess_params["prefix"] = prefix
        # 如果存在前缀，则对前缀进行分词处理
        if prefix:
            prefix_inputs = self.tokenizer(
                prefix, padding=False, add_special_tokens=add_special_tokens, return_tensors=self.framework
            )
            # 更新生成参数中的前缀长度
            generate_kwargs["prefix_length"] = prefix_inputs["input_ids"].shape[-1]

        # 如果设置了长序列处理方式，则检查是否为"hole"，否则抛出 ValueError 异常
        if handle_long_generation is not None:
            if handle_long_generation not in {"hole"}:
                raise ValueError(
                    f"{handle_long_generation} is not a valid value for `handle_long_generation` parameter expected"
                    " [None, 'hole']"
                )
            preprocess_params["handle_long_generation"] = handle_long_generation

        # 更新预处理参数和生成参数
        preprocess_params.update(generate_kwargs)
        forward_params = generate_kwargs

        # 准备后处理参数字典
        postprocess_params = {}
        # 处理返回类型相关的参数设置
        if return_full_text is not None and return_type is None:
            if return_text is not None:
                raise ValueError("`return_text` is mutually exclusive with `return_full_text`")
            if return_tensors is not None:
                raise ValueError("`return_full_text` is mutually exclusive with `return_tensors`")
            return_type = ReturnType.FULL_TEXT if return_full_text else ReturnType.NEW_TEXT
        if return_tensors is not None and return_type is None:
            if return_text is not None:
                raise ValueError("`return_text` is mutually exclusive with `return_tensors`")
            return_type = ReturnType.TENSORS
        if return_type is not None:
            postprocess_params["return_type"] = return_type
        # 处理清理标记化空格的设置
        if clean_up_tokenization_spaces is not None:
            postprocess_params["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces

        # 如果设置了停止序列，则编码停止序列并更新生成参数中的 EOS 标记 ID
        if stop_sequence is not None:
            stop_sequence_ids = self.tokenizer.encode(stop_sequence, add_special_tokens=False)
            if len(stop_sequence_ids) > 1:
                # 如果停止序列长度大于1，发出警告提示暂时不支持停止在多个标记序列上
                warnings.warn(
                    "Stopping on a multiple token sequence is not yet supported on transformers. The first token of"
                    " the stop sequence will be used as the stop sequence string in the interim."
                )
            generate_kwargs["eos_token_id"] = stop_sequence_ids[0]

        # 返回预处理参数、前向传递参数、后处理参数
        return preprocess_params, forward_params, postprocess_params

    # 重写 _parse_and_tokenize 方法以允许不寻常的语言建模分词器参数
    def _parse_and_tokenize(self, *args, **kwargs):
        """
        解析参数并进行分词处理
        """
        # 解析参数
        if self.model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
            # 如果模型类别是"TransfoXLLMHeadModel"，更新关键字参数，添加一个标志以在标点符号前加空格
            kwargs.update({"add_space_before_punct_symbol": True})
    
        # 调用父类方法，将参数传递给父类的_parse_and_tokenize方法处理
        return super()._parse_and_tokenize(*args, **kwargs)
    
    def __call__(self, text_inputs, **kwargs):
        """
        完成给定输入的提示。
    
        Args:
            args (`str` or `List[str]`):
                一个或多个提示（或一个提示列表）以供完成。
            return_tensors (`bool`, *optional*, defaults to `False`):
                是否返回预测结果的张量（作为标记索引）。如果设置为`True`，则不返回解码的文本。
            return_text (`bool`, *optional*, defaults to `True`):
                是否返回解码后的文本。
            return_full_text (`bool`, *optional*, defaults to `True`):
                如果设置为`False`，则仅返回添加的文本，否则返回完整的文本。仅在`return_text`设置为True时有意义。
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                是否清除文本输出中的额外空格。
            prefix (`str`, *optional*):
                添加到提示的前缀。
            handle_long_generation (`str`, *optional*):
                默认情况下，此管道不处理长生成（超出模型最大长度的生成）。针对此问题提供常见策略，具体取决于您的用例。
                - `None`：默认策略，没有特别的处理
                - `"hole"`：截断输入左侧，并留下足够宽的空白以进行生成（可能会截断大部分提示，当生成超出模型容量时不适用）
    
            generate_kwargs:
                传递给模型的generate方法的其他关键字参数（请参阅您的框架中相应的generate方法）。
    
        Return:
            一个字典列表或字典的列表：返回以下之一的字典（不能同时返回`generated_text`和`generated_token_ids`）：
            - **generated_text** (`str`，当`return_text=True`时存在) -- 生成的文本。
            - **generated_token_ids** (`torch.Tensor`或`tf.Tensor`，当`return_tensors=True`时存在) -- 生成文本的标记id。
        """
        # 调用父类方法，将文本输入和关键字参数传递给父类的__call__方法处理
        return super().__call__(text_inputs, **kwargs)
    # 对输入的文本进行预处理，返回模型输入所需的格式
    def preprocess(
        self,
        prompt_text,  # 输入的提示文本
        prefix="",  # 前缀，默认为空
        handle_long_generation=None,  # 处理长文本生成的方法，默认为空
        add_special_tokens=False,  # 是否添加特殊标记，默认为False
        truncation=None,  # 截断方式，默认为空
        padding=False,  # 是否进行填充，默认为False
        max_length=None,  # 最大长度，默认为空
        **generate_kwargs,  # 其它生成参数
    ):
        # 使用分词器对输入的前缀和提示文本进行编码并返回张量
        inputs = self.tokenizer(
            prefix + prompt_text,
            return_tensors=self.framework,  # 返回的格式为框架的张量
            truncation=truncation,  # 截断方式
            padding=padding,  # 是否填充
            max_length=max_length,  # 最大长度
            add_special_tokens=add_special_tokens,  # 是否添加特殊标记
        )
        inputs["prompt_text"] = prompt_text  # 将输入的提示文本添加到输入中

        # 处理长文本生成的方法为"hole"
        if handle_long_generation == "hole":
            # 获取当前输入的标记数
            cur_len = inputs["input_ids"].shape[-1]
            # 获取生成参数中的最大新标记数，如果不存在则使用最大长度减去当前长度
            if "max_new_tokens" in generate_kwargs:
                new_tokens = generate_kwargs["max_new_tokens"]
            else:
                new_tokens = generate_kwargs.get("max_length", self.model.config.max_length) - cur_len
                # 如果计算出的新标记数小于0，则抛出数值错误
                if new_tokens < 0:
                    raise ValueError("We cannot infer how many new tokens are expected")
            # 如果当前长度加上新标记数超过了分词器的最大长度
            if cur_len + new_tokens > self.tokenizer.model_max_length:
                # 计算需要保持的长度
                keep_length = self.tokenizer.model_max_length - new_tokens
                # 如果需要保持的长度小于等于0，则抛出数值错误
                if keep_length <= 0:
                    raise ValueError(
                        "We cannot use `hole` to handle this generation the number of desired tokens exceeds the"
                        " models max length"
                    )
                # 裁剪输入的标记和注意力掩码
                inputs["input_ids"] = inputs["input_ids"][:, -keep_length:]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, -keep_length:]

        return inputs  # 返回处理后的模型输入
    # 定义一个名为 _forward 的方法，接收 model_inputs 和 generate_kwargs 作为输入参数
    def _forward(self, model_inputs, **generate_kwargs):
        # 从 model_inputs 中获取 input_ids
        input_ids = model_inputs["input_ids"]
        # 从 model_inputs 中获取 attention_mask，若不存在则赋值为 None
        attention_mask = model_inputs.get("attention_mask", None)
        # 如果 input_ids 的列数为 0，则将 input_ids 和 attention_mask 赋值为 None，并设置 in_b 为 1
        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            # 否则设置 in_b 为 input_ids 的行数
            in_b = input_ids.shape[0]
        # 从 model_inputs 中弹出 prompt_text，并赋值给 prompt_text 变量
        prompt_text = model_inputs.pop("prompt_text")

        # 如果存在前缀（prefix），可能需要调整生成的长度（length）。在不永久修改 generate_kwargs 的情况下执行这些操作，
        # 因为一些参数可能来自 pipeline 的初始化。
        prefix_length = generate_kwargs.pop("prefix_length", 0)
        if prefix_length > 0:
            # 检测 generate_kwargs 中是否存在 "max_new_tokens" 或 "generation_config" 中的 max_new_tokens，若不存在则对 max_length 进行调整
            has_max_new_tokens = "max_new_tokens" in generate_kwargs or (
                "generation_config" in generate_kwargs
                and generate_kwargs["generation_config"].max_new_tokens is not None
            )
            if not has_max_new_tokens:
                # 调整 max_length
                generate_kwargs["max_length"] = generate_kwargs.get("max_length") or self.model.config.max_length
                generate_kwargs["max_length"] += prefix_length
            # 检测 generate_kwargs 中是否存在 "min_new_tokens" 或 "generation_config" 中的 min_new_tokens，若不存在则对 min_length 进行调整
            has_min_new_tokens = "min_new_tokens" in generate_kwargs or (
                "generation_config" in generate_kwargs
                and generate_kwargs["generation_config"].min_new_tokens is not None
            )
            if not has_min_new_tokens and "min_length" in generate_kwargs:
                generate_kwargs["min_length"] += prefix_length

        # 通过 model.generate 方法生成序列，传入 input_ids、attention_mask 和其他 generate_kwargs
        generated_sequence = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
        # 获取生成序列的行数，赋值给 out_b
        out_b = generated_sequence.shape[0]
        # 根据框架类型进行不同的处理
        if self.framework == "pt":
            # 若框架为 "pt"，则对生成的序列进行形状重塑，使其符合输出格式
            generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])
        elif self.framework == "tf":
            # 若框架为 "tf"，则使用 TensorFlow 中的 reshape 方法对生成的序列进行形状重塑，使其符合输出格式
            generated_sequence = tf.reshape(generated_sequence, (in_b, out_b // in_b, *generated_sequence.shape[1:]))
        # 返回包含生成序列、input_ids 和 prompt_text 的字典
        return {"generated_sequence": generated_sequence, "input_ids": input_ids, "prompt_text": prompt_text}
    # 用于后处理生成的文本序列的方法
    def postprocess(self, model_outputs, return_type=ReturnType.FULL_TEXT, clean_up_tokenization_spaces=True):
        # 从模型输出中获取生成的序列
        generated_sequence = model_outputs["generated_sequence"][0]
        # 从模型输出中获取输入 ID
        input_ids = model_outputs["input_ids"]
        # 从模型输出中获取提示文本
        prompt_text = model_outputs["prompt_text"]
        # 将生成的序列转换为 Python 列表
        generated_sequence = generated_sequence.numpy().tolist()
        
        # 存储结果的列表
        records = []
        
        # 遍历生成的序列
        for sequence in generated_sequence:
            # 根据 return_type 确定要返回的内容
            if return_type == ReturnType.TENSORS:
                # 如果是 ReturnType.TENSORS，返回生成的 token ID 列表
                record = {"generated_token_ids": sequence}
            elif return_type in {ReturnType.NEW_TEXT, ReturnType.FULL_TEXT}:
                # 如果是 ReturnType.NEW_TEXT 或 ReturnType.FULL_TEXT，解码生成的文本
                text = self.tokenizer.decode(
                    sequence,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )
                
                # 如果使用了 XLNet 或 Transfo-XL 模型，需要移除提示文本的 PADDING
                if input_ids is None:
                    prompt_length = 0
                else:
                    prompt_length = len(
                        self.tokenizer.decode(
                            input_ids[0],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                        )
                    )
                
                # 提取生成的文本，并根据 return_type 决定是否附加提示文本
                all_text = text[prompt_length:]
                if return_type == ReturnType.FULL_TEXT:
                    all_text = prompt_text + all_text
                
                record = {"generated_text": all_text}
            
            # 将结果添加到记录列表
            records.append(record)
        
        # 返回结果列表
        return records
```