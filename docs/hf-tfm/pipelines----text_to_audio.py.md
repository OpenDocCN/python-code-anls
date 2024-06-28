# `.\pipelines\text_to_audio.py`

```py
# 导入必要的模块和函数
from typing import List, Union
from ..utils import is_torch_available
from .base import Pipeline

# 如果 torch 可用，导入特定模型的映射和 SpeechT5HifiGan
if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING
    from ..models.speecht5.modeling_speecht5 import SpeechT5HifiGan

# 默认的声码器模型标识符
DEFAULT_VOCODER_ID = "microsoft/speecht5_hifigan"

class TextToAudioPipeline(Pipeline):
    """
    文本到音频生成管道，使用任意 `AutoModelForTextToWaveform` 或 `AutoModelForTextToSpectrogram` 模型。
    该管道从输入文本生成音频文件，并可选地接收其他条件输入。

    Example:

    ```
    >>> from transformers import pipeline

    >>> pipe = pipeline(model="suno/bark-small")
    >>> output = pipe("Hey it's HuggingFace on the phone!")

    >>> audio = output["audio"]
    >>> sampling_rate = output["sampling_rate"]
    ```

    了解如何使用管道的基础知识，参见[pipeline tutorial](../pipeline_tutorial)

    <Tip>

    可以通过使用 [`TextToAudioPipeline.__call__.forward_params`] 或 [`TextToAudioPipeline.__call__.generate_kwargs`] 来指定传递给模型的参数。

    Example:

    ```
    >>> from transformers import pipeline

    >>> music_generator = pipeline(task="text-to-audio", model="facebook/musicgen-small", framework="pt")

    >>> # 通过使用较高的温度添加随机性来增强音乐生成，并设置最大音乐长度
    >>> generate_kwargs = {
    ...     "do_sample": True,
    ...     "temperature": 0.7,
    ...     "max_new_tokens": 35,
    ... }

    >>> outputs = music_generator("Techno music with high melodic riffs", generate_kwargs=generate_kwargs)
    ```

    </Tip>

    目前可以通过 [`pipeline`] 加载此管道，使用以下任务标识符："text-to-speech" 或 "text-to-audio"。

    查看 [huggingface.co/models](https://huggingface.co/models?filter=text-to-speech) 上可用模型列表。
    """
    # 初始化函数，接受任意数量的位置参数和关键字参数
    def __init__(self, *args, vocoder=None, sampling_rate=None, **kwargs):
        # 调用父类的初始化函数
        super().__init__(*args, **kwargs)

        # 如果使用 TensorFlow 框架，则抛出数值错误异常，因为 TextToAudioPipeline 只能在 PyTorch 中使用
        if self.framework == "tf":
            raise ValueError("The TextToAudioPipeline is only available in PyTorch.")

        # 初始化属性 vocoder 为 None
        self.vocoder = None
        # 如果模型的类在 MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING 的值之中
        if self.model.__class__ in MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING.values():
            # 根据 DEFAULT_VOCODER_ID 加载 SpeechT5HifiGan 模型，并放置在模型所在的设备上
            self.vocoder = (
                SpeechT5HifiGan.from_pretrained(DEFAULT_VOCODER_ID).to(self.model.device)
                if vocoder is None  # 如果未提供 vocoder 参数，则使用默认的 vocoder
                else vocoder  # 否则使用传入的 vocoder
            )

        # 初始化属性 sampling_rate 为传入的 sampling_rate 参数
        self.sampling_rate = sampling_rate
        # 如果 vocoder 不为 None，则设置 sampling_rate 为 vocoder 配置的 sampling_rate
        if self.vocoder is not None:
            self.sampling_rate = self.vocoder.config.sampling_rate

        # 如果 sampling_rate 仍为 None，则从模型的配置和生成配置中获取 sampling_rate
        if self.sampling_rate is None:
            # 获取模型的配置
            config = self.model.config
            # 获取模型的生成配置（如果存在）
            gen_config = self.model.__dict__.get("generation_config", None)
            # 如果生成配置存在，则更新模型配置
            if gen_config is not None:
                config.update(gen_config.to_dict())

            # 尝试从配置中的多个可能的属性名中获取 sampling_rate
            for sampling_rate_name in ["sample_rate", "sampling_rate"]:
                sampling_rate = getattr(config, sampling_rate_name, None)
                # 如果成功获取到 sampling_rate，则将其赋值给 self.sampling_rate，并结束循环
                if sampling_rate is not None:
                    self.sampling_rate = sampling_rate

    # 预处理函数，接受文本输入和其他关键字参数
    def preprocess(self, text, **kwargs):
        # 如果 text 是字符串，则转换为单元素列表
        if isinstance(text, str):
            text = [text]

        # 如果模型的类型是 "bark"
        if self.model.config.model_type == "bark":
            # 创建一个新的关键字参数字典 new_kwargs
            new_kwargs = {
                # 设置 max_length 为模型生成配置中的 max_input_semantic_length，最大长度默认为 256
                "max_length": self.model.generation_config.semantic_config.get("max_input_semantic_length", 256),
                "add_special_tokens": False,  # 不添加特殊标记
                "return_attention_mask": True,  # 返回注意力掩码
                "return_token_type_ids": False,  # 不返回 token 类型 IDs
                "padding": "max_length",  # 使用 "max_length" 进行填充
            }

            # 优先使用传入的 kwargs 更新 new_kwargs
            new_kwargs.update(kwargs)

            # 将 kwargs 指向 new_kwargs
            kwargs = new_kwargs

        # 使用 tokenizer 对文本进行处理，返回 PyTorch 张量表示的输出
        output = self.tokenizer(text, **kwargs, return_tensors="pt")

        return output
    # 定义私有方法 `_forward`，用于执行模型的前向推断过程
    def _forward(self, model_inputs, **kwargs):
        # 需要确保一些关键字参数处于正确的设备上
        kwargs = self._ensure_tensor_on_device(kwargs, device=self.device)
        # 获取前向推断所需的参数
        forward_params = kwargs["forward_params"]
        # 获取生成过程的关键字参数
        generate_kwargs = kwargs["generate_kwargs"]

        # 如果模型支持生成操作
        if self.model.can_generate():
            # 确保生成过程的关键字参数处于正确的设备上
            generate_kwargs = self._ensure_tensor_on_device(generate_kwargs, device=self.device)

            # 生成过程的参数优先级高于前向推断的参数
            forward_params.update(generate_kwargs)

            # 调用模型的生成方法
            output = self.model.generate(**model_inputs, **forward_params)
        else:
            # 如果不支持生成操作，则使用前向推断的参数调用模型
            if len(generate_kwargs):
                # 抛出数值错误，提醒用户使用前向推断模型时应使用 forward_params 而不是 generate_kwargs
                raise ValueError(
                    f"""You're using the `TextToAudioPipeline` with a forward-only model, but `generate_kwargs` is non empty.
                                 For forward-only TTA models, please use `forward_params` instead of of
                                 `generate_kwargs`. For reference, here are the `generate_kwargs` used here:
                                 {generate_kwargs.keys()}"""
                )
            # 使用前向推断的参数调用模型，并取第一个输出
            output = self.model(**model_inputs, **forward_params)[0]

        # 如果存在声码器，将输出转换为波形
        if self.vocoder is not None:
            # 将输出作为频谱图输入声码器，得到波形作为最终输出
            output = self.vocoder(output)

        # 返回最终输出
        return output

    # 重载 `__call__` 方法，允许对象被调用，用于从文本生成语音/音频
    def __call__(self, text_inputs: Union[str, List[str]], **forward_params):
        """
        从输入文本生成语音/音频。详细信息请参阅 [`TextToAudioPipeline`] 文档。

        Args:
            text_inputs (`str` or `List[str]`):
                要生成的文本或文本列表。
            forward_params (`dict`, *可选*):
                传递给模型生成/前向方法的参数。`forward_params` 总是传递给底层模型。

        Return:
            `dict` 或 `list` of `dict`: 返回的字典包含两个键值对:

            - **audio** (`np.ndarray` of shape `(nb_channels, audio_length)`) -- 生成的音频波形。
            - **sampling_rate** (`int`) -- 生成的音频波形的采样率。
        """
        return super().__call__(text_inputs, **forward_params)

    # 定义私有方法 `_sanitize_parameters`，用于清理和规范化输入参数
    def _sanitize_parameters(
        self,
        preprocess_params=None,
        forward_params=None,
        generate_kwargs=None,
    ):
    ):
        # 定义一个包含参数的字典，包括前向参数和生成参数
        params = {
            "forward_params": forward_params if forward_params else {},  # 如果前向参数存在则使用，否则使用空字典
            "generate_kwargs": generate_kwargs if generate_kwargs else {},  # 如果生成参数存在则使用，否则使用空字典
        }

        # 如果预处理参数为None，则将其设为空字典
        if preprocess_params is None:
            preprocess_params = {}
        postprocess_params = {}  # 初始化后处理参数为空字典

        # 返回预处理参数、params字典和后处理参数
        return preprocess_params, params, postprocess_params

    def postprocess(self, waveform):
        # 定义一个空的输出字典
        output_dict = {}

        # 将音波数据转换为CPU上的浮点数数组，并存入输出字典中的"audio"键
        output_dict["audio"] = waveform.cpu().float().numpy()
        output_dict["sampling_rate"] = self.sampling_rate  # 将采样率存入输出字典的"sampling_rate"键

        # 返回填充了音频数据和采样率的输出字典
        return output_dict
```