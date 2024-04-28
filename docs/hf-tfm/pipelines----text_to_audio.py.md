# `.\transformers\pipelines\text_to_audio.py`

```py
# 从 typing 模块中导入 List 和 Union 类型
from typing import List, Union

# 从 ..utils 模块中导入 is_torch_available 函数
from ..utils import is_torch_available
# 从 .base 模块中导入 Pipeline 类
from .base import Pipeline

# 如果 torch 可用
if is_torch_available():
    # 从 ..models.auto.modeling_auto 模块中导入 MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING 常量
    from ..models.auto.modeling_auto import MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING
    # 从 ..models.speecht5.modeling_speecht5 模块中导入 SpeechT5HifiGan 类
    from ..models.speecht5.modeling_speecht5 import SpeechT5HifiGan

# 设置默认的 VOCODER_ID 为 "microsoft/speecht5_hifigan"
DEFAULT_VOCODER_ID = "microsoft/speecht5_hifigan"


# 定义 TextToAudioPipeline 类，继承自 Pipeline 类
class TextToAudioPipeline(Pipeline):
    """
    Text-to-audio generation pipeline using any `AutoModelForTextToWaveform` or `AutoModelForTextToSpectrogram`. This
    pipeline generates an audio file from an input text and optional other conditional inputs.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> pipe = pipeline(model="suno/bark-small")
    >>> output = pipe("Hey it's HuggingFace on the phone!")

    >>> audio = output["audio"]
    >>> sampling_rate = output["sampling_rate"]
    ```py

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    <Tip>

    You can specify parameters passed to the model by using [`TextToAudioPipeline.__call__.forward_params`] or
    [`TextToAudioPipeline.__call__.generate_kwargs`].

    Example:

    ```python
    >>> from transformers import pipeline

    >>> music_generator = pipeline(task="text-to-audio", model="facebook/musicgen-small", framework="pt")

    >>> # diversify the music generation by adding randomness with a high temperature and set a maximum music length
    >>> generate_kwargs = {
    ...     "do_sample": True,
    ...     "temperature": 0.7,
    ...     "max_new_tokens": 35,
    ... }

    >>> outputs = music_generator("Techno music with high melodic riffs", generate_kwargs=generate_kwargs)
    ```py

    </Tip>

    This pipeline can currently be loaded from [`pipeline`] using the following task identifiers: `"text-to-speech"` or
    `"text-to-audio"`.

    See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=text-to-speech).
    """
    # 初始化方法，接受一系列位置参数和关键字参数；vocoder用于语音合成，sampling_rate用于采样率
    def __init__(self, *args, vocoder=None, sampling_rate=None, **kwargs):
        # 调用父类的初始化方法，传入位置参数和关键字参数
        super().__init__(*args, **kwargs)

        # 如果框架是 "tf"，则抛出数值错误
        if self.framework == "tf":
            raise ValueError("The TextToAudioPipeline is only available in PyTorch.")

        # 初始化vocoder
        self.vocoder = None
        # 如果模型的类在MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING的值中
        if self.model.__class__ in MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING.values():
            # 根据给定的DEFAULT_VOCODER_ID加载SpeechT5HifiGan模型并移到模型所在设备
            self.vocoder = (
                SpeechT5HifiGan.from_pretrained(DEFAULT_VOCODER_ID).to(self.model.device)
                if vocoder is None
                else vocoder
            )

        # 初始化采样率
        self.sampling_rate = sampling_rate
        # 如果vocoder不为空
        if self.vocoder is not None:
            # 将采样率设置为vocoder的配置采样率
            self.sampling_rate = self.vocoder.config.sampling_rate

        # 如果采样率为空
        if self.sampling_rate is None:
            # 从模型配置和生成配置中获取采样率
            config = self.model.config
            gen_config = self.model.__dict__.get("generation_config", None)
            if gen_config is not None:
                config.update(gen_config.to_dict())

            for sampling_rate_name in ["sample_rate", "sampling_rate"]:
                sampling_rate = getattr(config, sampling_rate_name, None)
                if sampling_rate is not None:
                    self.sampling_rate = sampling_rate

    # 预处理方法，接受一个字符串文本和一系列关键字参数
    def preprocess(self, text, **kwargs):
        # 如果文本是字符串，转换成列表
        if isinstance(text, str):
            text = [text]

        # 如果模型的类型是 "bark"
        if self.model.config.model_type == "bark":
            # 使用BarkProcessor调用bark分词器，并使用指定关键字参数
            new_kwargs = {
                "max_length": self.model.generation_config.semantic_config.get("max_input_semantic_length", 256),
                "add_special_tokens": False,
                "return_attention_mask": True,
                "return_token_type_ids": False,
                "padding": "max_length",
            }

            # 优先使用传入的关键字参数
            new_kwargs.update(kwargs)

            kwargs = new_kwargs

        # 使用tokenizer对文本进行预处理，返回结果
        output = self.tokenizer(text, **kwargs, return_tensors="pt")

        return output
    def _forward(self, model_inputs, **kwargs):
        # we expect some kwargs to be additional tensors which need to be on the right device
        kwargs = self._ensure_tensor_on_device(kwargs, device=self.device)
        forward_params = kwargs["forward_params"]
        generate_kwargs = kwargs["generate_kwargs"]

        if self.model.can_generate():
            # we expect some kwargs to be additional tensors which need to be on the right device
            generate_kwargs = self._ensure_tensor_on_device(generate_kwargs, device=self.device)

            # generate_kwargs get priority over forward_params
            forward_params.update(generate_kwargs)

            output = self.model.generate(**model_inputs, **forward_params)
        else:
            if len(generate_kwargs):
                # raise an error if using `TextToAudioPipeline` with a forward-only model and `generate_kwargs` is non-empty
                raise ValueError(
                    f"""You're using the `TextToAudioPipeline` with a forward-only model, but `generate_kwargs` is non empty.
                                 For forward-only TTA models, please use `forward_params` instead of of
                                 `generate_kwargs`. For reference, here are the `generate_kwargs` used here:
                                 {generate_kwargs.keys()}"""
                )
            # generate audio using the model and forward_params
            output = self.model(**model_inputs, **forward_params)[0]

        if self.vocoder is not None:
            # if a vocoder is available, convert the output spectrogram into a waveform
            output = self.vocoder(output)

        return output

    def __call__(self, text_inputs: Union[str, List[str]], **forward_params):
        """
        Generates speech/audio from the inputs. See the [`TextToAudioPipeline`] documentation for more information.

        Args:
            text_inputs (`str` or `List[str]`):
                The text(s) to generate.
            forward_params (`dict`, *optional*):
                Parameters passed to the model generation/forward method. `forward_params` are always passed to the
                underlying model.
            generate_kwargs (`dict`, *optional*):
                The dictionary of ad-hoc parametrization of `generate_config` to be used for the generation call. For a
                complete overview of generate, check the [following
                guide](https://huggingface.co/docs/transformers/en/main_classes/text_generation). `generate_kwargs` are
                only passed to the underlying model if the latter is a generative model.

        Return:
            A `dict` or a list of `dict`: The dictionaries have two keys:

            - **audio** (`np.ndarray` of shape `(nb_channels, audio_length)`) -- The generated audio waveform.
            - **sampling_rate** (`int`) -- The sampling rate of the generated audio waveform.
        """
        return super().__call__(text_inputs, **forward_params)

    def _sanitize_parameters(
        self,
        preprocess_params=None,
        forward_params=None,
        generate_kwargs=None,
    # 定义一个函数，接收参数 forward_params, generate_kwargs
    def __call__(self, waveform, forward_params=None, generate_kwargs=None, preprocess_params=None):
        # 定义一个字典，包含 forward_params 或者空字典，以及 generate_kwargs 或者空字典
        params = {
            "forward_params": forward_params if forward_params else {},
            "generate_kwargs": generate_kwargs if generate_kwargs else {},
        }

        # 如果 preprocess_params 为 None，则置为一个空字典
        if preprocess_params is None:
            preprocess_params = {}
        # 定义一个空的字典 postprocess_params
        postprocess_params = {}

        # 返回 preprocess_params, params, postprocess_params
        return preprocess_params, params, postprocess_params

    # 定义一个函数，接收参数 waveform
    def postprocess(self, waveform):
        # 定义一个空的字典 output_dict
        output_dict = {}

        # 将音频数据转换为 CPU 上的浮点数，并转换为 numpy 数组，存入 output_dict
        output_dict["audio"] = waveform.cpu().float().numpy()
        # 将采样率存入 output_dict
        output_dict["sampling_rate"] = self.sampling_rate

        # 返回 output_dict
        return output_dict
```