# `.\pipelines\text_generation.py`

```py
import enum  # 导入枚举类型的模块
import warnings  # 导入警告模块
from typing import Dict  # 导入字典类型的类型提示

from ..utils import add_end_docstrings, is_tf_available, is_torch_available  # 导入自定义工具函数和判断TensorFlow、PyTorch是否可用的函数
from .base import Pipeline, build_pipeline_init_args  # 导入基础类Pipeline和构建初始化参数的函数


if is_torch_available():  # 如果PyTorch可用
    from ..models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES  # 导入PyTorch自动模型命名映射

if is_tf_available():  # 如果TensorFlow可用
    import tensorflow as tf  # 导入TensorFlow模块

    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES  # 导入TensorFlow自动模型命名映射


class ReturnType(enum.Enum):  # 定义返回类型枚举类
    TENSORS = 0  # 返回张量类型
    NEW_TEXT = 1  # 返回新文本类型
    FULL_TEXT = 2  # 返回完整文本类型


class Chat:  # 聊天类定义
    """This class is intended to just be used internally in this pipeline and not exposed to users. We convert chats
    to this format because the rest of the pipeline code tends to assume that lists of messages are
    actually a batch of samples rather than messages in the same conversation."""

    def __init__(self, messages: Dict):  # 初始化方法，接收消息字典作为参数
        for message in messages:  # 遍历消息字典中的每个消息
            if not ("role" in message and "content" in message):  # 检查消息中是否包含必要的'role'和'content'键
                raise ValueError("When passing chat dicts as input, each dict must have a 'role' and 'content' key.")
        self.messages = messages  # 将消息字典赋值给实例变量self.messages


@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True))  # 添加尾部文档字符串，调用build_pipeline_init_args函数，声明带有分词器的初始化参数
class TextGenerationPipeline(Pipeline):  # 文本生成管道类，继承自基础类Pipeline
    """
    Language generation pipeline using any `ModelWithLMHead`. This pipeline predicts the words that will follow a
    specified text prompt. It can also accept one or more chats. Each chat takes the form of a list of dicts,
    where each dict contains "role" and "content" keys.

    Example:

    ```
    >>> from transformers import pipeline

    >>> generator = pipeline(model="openai-community/gpt2")
    >>> generator("I can't believe you did such a ", do_sample=False)
    [{'generated_text': "I can't believe you did such a icky thing to me. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I"}]

    >>> # These parameters will return suggestions, and only the newly created text making it easier for prompting suggestions.
    >>> outputs = generator("My tart needs some", num_return_sequences=4, return_full_text=False)
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial). You can pass text
    generation parameters to this pipeline to control stopping criteria, decoding strategy, and more. Learn more about
    text generation parameters in [Text generation strategies](../generation_strategies) and [Text
    generation](text_generation).

    This language generation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"text-generation"`.

    The models that this pipeline can use are models that have been trained with an autoregressive language modeling
    objective, which includes the uni-directional models in the library (e.g. openai-community/gpt2). See the list of available models
    on [huggingface.co/models](https://huggingface.co/models?filter=text-generation).
    """
    # 定义一个用于 XLNet 和 TransformerXL 模型的前缀文本，以帮助处理短提示
    XL_PREFIX = """
    In 1991, the remains of Russian Tsar Nicholas II and his family (except for Alexei and Maria) are discovered. The
    voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the remainder of the story. 1883 Western
    Siberia, a young Grigori Rasputin is asked by his father and a group of men to perform magic. Rasputin has a vision
    and denounces one of the men as a horse thief. Although his father initially slaps him for making such an
    accusation, Rasputin watches as the man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
    the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous, with people, even a bishop,
    begging for his blessing. <eod> </s> <eos>
    """

    # 初始化方法，继承自父类构造函数，并检查模型类型
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 根据框架类型选择相应的映射名称列表，然后检查模型类型
        self.check_model_type(
            TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES if self.framework == "tf" else MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        )
        # 如果预处理参数中不存在 "prefix"
        if "prefix" not in self._preprocess_params:
            # 设置默认值为 None
            prefix = None
            # 如果模型配置中的 prefix 不为 None，则将其赋给 prefix
            if self.model.config.prefix is not None:
                prefix = self.model.config.prefix
            # 如果 prefix 仍为 None，并且模型类名在指定列表中
            if prefix is None and self.model.__class__.__name__ in [
                "XLNetLMHeadModel",
                "TransfoXLLMHeadModel",
                "TFXLNetLMHeadModel",
                "TFTransfoXLLMHeadModel",
            ]:
                # 对于 XLNet 和 TransformerXL 模型，使用预先定义的 XL_PREFIX 作为 prefix
                prefix = self.XL_PREFIX
            # 如果最终确定了 prefix 的值
            if prefix is not None:
                # 重新计算与 prefix 相关的一些生成参数
                preprocess_params, forward_params, _ = self._sanitize_parameters(prefix=prefix, **self._forward_params)
                # 更新预处理参数和前向参数
                self._preprocess_params = {**self._preprocess_params, **preprocess_params}
                self._forward_params = {**self._forward_params, **forward_params}

    # 根据指定参数进行参数清理和更新的内部方法
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
            # 定义预处理参数字典，包括特殊标记添加、截断、填充和最大长度等设置
            preprocess_params = {
                "add_special_tokens": add_special_tokens,
                "truncation": truncation,
                "padding": padding,
                "max_length": max_length,
            }
            # 如果设置了最大长度，将其添加到生成参数中
            if max_length is not None:
                generate_kwargs["max_length"] = max_length

            # 如果设置了前缀，将其加入预处理参数中
            if prefix is not None:
                preprocess_params["prefix"] = prefix
            # 如果前缀不为空，则通过分词器处理前缀输入并设置前缀长度
            if prefix:
                prefix_inputs = self.tokenizer(
                    prefix, padding=False, add_special_tokens=add_special_tokens, return_tensors=self.framework
                )
                generate_kwargs["prefix_length"] = prefix_inputs["input_ids"].shape[-1]

            # 如果设置了处理长生成文本的选项，验证选项的有效性
            if handle_long_generation is not None:
                if handle_long_generation not in {"hole"}:
                    raise ValueError(
                        f"{handle_long_generation} is not a valid value for `handle_long_generation` parameter expected"
                        " [None, 'hole']"
                    )
                preprocess_params["handle_long_generation"] = handle_long_generation

            # 将生成参数更新到预处理参数中
            preprocess_params.update(generate_kwargs)
            forward_params = generate_kwargs

            # 定义后处理参数字典
            postprocess_params = {}
            # 如果设置了返回全文和返回类型为空，则根据返回全文和返回文本的互斥关系设置返回类型
            if return_full_text is not None and return_type is None:
                if return_text is not None:
                    raise ValueError("`return_text` is mutually exclusive with `return_full_text`")
                if return_tensors is not None:
                    raise ValueError("`return_full_text` is mutually exclusive with `return_tensors`")
                return_type = ReturnType.FULL_TEXT if return_full_text else ReturnType.NEW_TEXT
            # 如果设置了返回张量并且返回类型为空，则设置返回类型为张量
            if return_tensors is not None and return_type is None:
                if return_text is not None:
                    raise ValueError("`return_text` is mutually exclusive with `return_tensors`")
                return_type = ReturnType.TENSORS
            # 如果设置了返回类型，则加入后处理参数中
            if return_type is not None:
                postprocess_params["return_type"] = return_type
            # 如果设置了清理分词空格选项，则加入后处理参数中
            if clean_up_tokenization_spaces is not None:
                postprocess_params["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces

            # 如果设置了停止序列，则编码停止序列并设置生成参数中的结束标记 ID
            if stop_sequence is not None:
                stop_sequence_ids = self.tokenizer.encode(stop_sequence, add_special_tokens=False)
                if len(stop_sequence_ids) > 1:
                    warnings.warn(
                        "Stopping on a multiple token sequence is not yet supported on transformers. The first token of"
                        " the stop sequence will be used as the stop sequence string in the interim."
                    )
                generate_kwargs["eos_token_id"] = stop_sequence_ids[0]

            # 返回预处理参数、前向参数和后处理参数
            return preprocess_params, forward_params, postprocess_params

        # 重写 _parse_and_tokenize 方法以允许非常规的语言建模分词器参数
    def _parse_and_tokenize(self, *args, **kwargs):
        """
        Parse arguments and tokenize
        """
        # 解析参数
        # 如果模型的类名在特定列表中，则更新kwargs以添加一个标志
        if self.model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
            kwargs.update({"add_space_before_punct_symbol": True})

        # 调用父类方法，将解析后的参数和标记化处理
        return super()._parse_and_tokenize(*args, **kwargs)

    def preprocess(
        self,
        prompt_text,
        prefix="",
        handle_long_generation=None,
        add_special_tokens=False,
        truncation=None,
        padding=False,
        max_length=None,
        **generate_kwargs,
    ):
        # 如果prompt_text是Chat类型的对象，则应用特定的tokenizer方法
        if isinstance(prompt_text, Chat):
            inputs = self.tokenizer.apply_chat_template(
                prompt_text.messages,
                truncation=truncation,
                padding=padding,
                max_length=max_length,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors=self.framework,
            )
        else:
            # 否则，使用tokenizer对prompt_text进行标记化处理
            inputs = self.tokenizer(
                prefix + prompt_text,
                truncation=truncation,
                padding=padding,
                max_length=max_length,
                add_special_tokens=add_special_tokens,
                return_tensors=self.framework,
            )
        
        # 将原始的prompt_text存储在inputs中
        inputs["prompt_text"] = prompt_text

        # 处理长生成情况下的特殊处理
        if handle_long_generation == "hole":
            cur_len = inputs["input_ids"].shape[-1]
            # 根据generate_kwargs获取最大新增token数或长度
            if "max_new_tokens" in generate_kwargs:
                new_tokens = generate_kwargs["max_new_tokens"]
            else:
                new_tokens = generate_kwargs.get("max_length", self.model.config.max_length) - cur_len
                if new_tokens < 0:
                    raise ValueError("We cannot infer how many new tokens are expected")
            
            # 如果当前长度加上新token数超过了tokenizer的最大长度限制
            if cur_len + new_tokens > self.tokenizer.model_max_length:
                keep_length = self.tokenizer.model_max_length - new_tokens
                if keep_length <= 0:
                    raise ValueError(
                        "We cannot use `hole` to handle this generation the number of desired tokens exceeds the"
                        " models max length"
                    )

                # 裁剪input_ids和attention_mask以保持长度在tokenizer的最大长度内
                inputs["input_ids"] = inputs["input_ids"][:, -keep_length:]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, -keep_length:]

        # 返回处理后的inputs
        return inputs
    # 定义一个方法 `_forward`，用于执行模型的前向推理
    def _forward(self, model_inputs, **generate_kwargs):
        # 从模型输入中获取输入的 token IDs
        input_ids = model_inputs["input_ids"]
        # 获取注意力掩码，如果不存在则设为 None
        attention_mask = model_inputs.get("attention_mask", None)
        
        # 允许空的提示文本
        # 如果输入的 token IDs 的第二维度为 0，则将 input_ids 和 attention_mask 设为 None，并设置 in_b 为 1
        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            # 否则，in_b 等于输入 token IDs 的第一维度的大小
            in_b = input_ids.shape[0]
        
        # 从模型输入中弹出提示文本
        prompt_text = model_inputs.pop("prompt_text")

        # 如果有前缀，则可能需要调整生成长度。
        # 在不永久修改 generate_kwargs 的情况下进行调整，因为一些参数可能来自管道的初始化。
        # 弹出并获取前缀长度
        prefix_length = generate_kwargs.pop("prefix_length", 0)
        if prefix_length > 0:
            # 检查是否存在 max_new_tokens 参数或者在 generate_kwargs 中的 generation_config 中存在 max_new_tokens 参数
            has_max_new_tokens = "max_new_tokens" in generate_kwargs or (
                "generation_config" in generate_kwargs
                and generate_kwargs["generation_config"].max_new_tokens is not None
            )
            # 如果没有 max_new_tokens 参数，则将 max_length 设置为 generate_kwargs 中的 max_length 或者模型配置中的 max_length，并增加 prefix_length
            if not has_max_new_tokens:
                generate_kwargs["max_length"] = generate_kwargs.get("max_length") or self.model.config.max_length
                generate_kwargs["max_length"] += prefix_length
            
            # 检查是否存在 min_new_tokens 参数或者在 generate_kwargs 中的 generation_config 中存在 min_new_tokens 参数
            has_min_new_tokens = "min_new_tokens" in generate_kwargs or (
                "generation_config" in generate_kwargs
                and generate_kwargs["generation_config"].min_new_tokens is not None
            )
            # 如果没有 min_new_tokens 参数，并且存在 min_length 参数，则将 min_length 增加 prefix_length
            if not has_min_new_tokens and "min_length" in generate_kwargs:
                generate_kwargs["min_length"] += prefix_length

        # 使用模型生成方法生成序列，传入 input_ids 和 attention_mask，以及其他生成参数
        generated_sequence = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
        # 获取生成序列的第一维度的大小
        out_b = generated_sequence.shape[0]
        
        # 根据框架类型进行形状调整
        if self.framework == "pt":  # 如果框架是 PyTorch
            generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])
        elif self.framework == "tf":  # 如果框架是 TensorFlow
            generated_sequence = tf.reshape(generated_sequence, (in_b, out_b // in_b, *generated_sequence.shape[1:]))
        
        # 返回包含生成序列、输入的 token IDs 和提示文本的字典
        return {"generated_sequence": generated_sequence, "input_ids": input_ids, "prompt_text": prompt_text}
    def postprocess(self, model_outputs, return_type=ReturnType.FULL_TEXT, clean_up_tokenization_spaces=True):
        # 获取生成的文本序列
        generated_sequence = model_outputs["generated_sequence"][0]
        # 获取输入的 token IDs
        input_ids = model_outputs["input_ids"]
        # 获取提示文本
        prompt_text = model_outputs["prompt_text"]
        # 将生成的序列转换为 numpy 数组，再转换为 Python 列表
        generated_sequence = generated_sequence.numpy().tolist()
        # 初始化记录列表
        records = []
        # 遍历生成的序列
        for sequence in generated_sequence:
            if return_type == ReturnType.TENSORS:
                # 如果返回类型是 TENSORS，则记录生成的 token IDs
                record = {"generated_token_ids": sequence}
            elif return_type in {ReturnType.NEW_TEXT, ReturnType.FULL_TEXT}:
                # 解码生成的文本
                text = self.tokenizer.decode(
                    sequence,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )

                # 如果 input_ids 为空，则使用的是 XLNet 或 Transfo-XL 模型，需要移除 PADDING prompt
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

                # 移除提示长度对应的文本
                all_text = text[prompt_length:]
                # 如果返回类型是 FULL_TEXT
                if return_type == ReturnType.FULL_TEXT:
                    # 如果提示文本是字符串，则将其与生成的文本合并
                    if isinstance(prompt_text, str):
                        all_text = prompt_text + all_text
                    # 如果提示文本是 Chat 类型，则将其消息与生成的文本合并
                    elif isinstance(prompt_text, Chat):
                        all_text = prompt_text.messages + [{"role": "assistant", "content": all_text}]

                # 创建记录包含生成的文本
                record = {"generated_text": all_text}
            # 将记录加入记录列表
            records.append(record)

        # 返回所有记录
        return records
```