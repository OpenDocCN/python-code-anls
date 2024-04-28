# `.\transformers\pipelines\audio_classification.py`

```
# 引入模块subprocess、Union
import subprocess
from typing import Union

# 引入模块numpy、requests
import numpy as np
import requests

# 引入模块..utils中的add_end_docstrings、is_torch_available、is_torchaudio_available、logging
from ..utils import add_end_docstrings, is_torch_available, is_torchaudio_available, logging

# 引入..models.auto.modeling_auto中的MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES
if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES

# 引入模块logging
logger = logging.get_logger(__name__)

# 定义了函数ffmpeg_read，函数参数为bpayload: bytes和sampling_rate: int，返回值为np.array
def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.array:
    """
    Helper function to read an audio file through ffmpeg.
    """
    # 设定音频采样率
    ar = f"{sampling_rate}"
    # 设定音频通道数，此处为单声道
    ac = "1"
    # 设定音频转换格式，此处为f32le
    format_for_conversion = "f32le"
    # 设定ffmpeg所需的命令
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        "pipe:0",
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]

    # 尝试通过subprocess模块的Popen方法创建并返回一个Process类的对象
    try:
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    # 如果运行subprocess模块的Popen方法出错，则抛出异常信息
    except FileNotFoundError:
        raise ValueError("ffmpeg was not found but is required to load audio files from filename")
    # 获得输出流
    output_stream = ffmpeg_process.communicate(bpayload)
    out_bytes = output_stream[0]
    # 根据输出流创建Numpy数组
    audio = np.frombuffer(out_bytes, np.float32)
    # 如果音频内容为空，则抛出异常信息
    if audio.shape[0] == 0:
        raise ValueError("Malformed soundfile")
    # 返回Numpy数组
    return audio

# 定义了AudioClassificationPipeline类，继承自Pipeline类，并添加了相应的参数
@add_end_docstrings(PIPELINE_INIT_ARGS)
class AudioClassificationPipeline(Pipeline):
    """
    Audio classification pipeline using any `AutoModelForAudioClassification`. This pipeline predicts the class of a
    raw waveform or an audio file. In case of an audio file, ffmpeg should be installed to support multiple audio
    formats.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> classifier = pipeline(model="superb/wav2vec2-base-superb-ks")
    >>> classifier("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
    [{'score': 0.997, 'label': '_unknown_'}, {'score': 0.002, 'label': 'left'}, {'score': 0.0, 'label': 'yes'}, {'score': 0.0, 'label': 'down'}, {'score': 0.0, 'label': 'stop'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)


    This pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"audio-classification"`.

    See the list of available models on
    ```
    """
        [huggingface.co/models](https://huggingface.co/models?filter=audio-classification).
        """
    
        # 初始化方法，设置默认的 top_k 参数为 5
        def __init__(self, *args, **kwargs):
            kwargs["top_k"] = 5
            # 调用父类的初始化方法
            super().__init__(*args, **kwargs)
    
            # 检查是否为 PyTorch 框架
            if self.framework != "pt":
                raise ValueError(f"The {self.__class__} is only available in PyTorch.")
    
            # 检查模型类型
            self.check_model_type(MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES)
    
        # 对输入进行分类，输入可以是数组、字节、字符串或字典
        def __call__(
            self,
            inputs: Union[np.ndarray, bytes, str],
            **kwargs,
        ):
            """
            Classify the sequence(s) given as inputs. See the [`AutomaticSpeechRecognitionPipeline`] documentation for more
            information.
    
            Args:
                inputs (`np.ndarray` or `bytes` or `str` or `dict`):
                    The inputs is either :
                        - `str` that is the filename of the audio file, the file will be read at the correct sampling rate
                          to get the waveform using *ffmpeg*. This requires *ffmpeg* to be installed on the system.
                        - `bytes` it is supposed to be the content of an audio file and is interpreted by *ffmpeg* in the
                          same way.
                        - (`np.ndarray` of shape (n, ) of type `np.float32` or `np.float64`)
                            Raw audio at the correct sampling rate (no further check will be done)
                        - `dict` form can be used to pass raw audio sampled at arbitrary `sampling_rate` and let this
                          pipeline do the resampling. The dict must be either be in the format `{"sampling_rate": int,
                          "raw": np.array}`, or `{"sampling_rate": int, "array": np.array}`, where the key `"raw"` or
                          `"array"` is used to denote the raw audio waveform.
                top_k (`int`, *optional*, defaults to None):
                    The number of top labels that will be returned by the pipeline. If the provided number is `None` or
                    higher than the number of labels available in the model configuration, it will default to the number of
                    labels.
    
            Return:
                A list of `dict` with the following keys:
    
                - **label** (`str`) -- The label predicted.
                - **score** (`float`) -- The corresponding probability.
            """
            # 调用父类的分类方法
            return super().__call__(inputs, **kwargs)
    
        # 对参数进行清理，当前管道不需要参数
        def _sanitize_parameters(self, top_k=None, **kwargs):
            # No parameters on this pipeline right now
            postprocess_params = {}
            if top_k is not None:
                if top_k > self.model.config.num_labels:
                    top_k = self.model.config.num_labels
                postprocess_params["top_k"] = top_k
            return {}, {}, postprocess_params
    # 预处理输入数据
    def preprocess(self, inputs):
        # 如果输入数据是字符串
        if isinstance(inputs, str):
            # 如果是以 "http://" 或 "https://" 开头
            if inputs.startswith("http://") or inputs.startswith("https://"):
                # 需要验证真实的协议，否则无法使用本地文件
                # 如 http_huggingface_co.png
                inputs = requests.get(inputs).content
            else:
                # 以二进制形式打开输入文件
                with open(inputs, "rb") as f:
                    inputs = f.read()

        # 如果输入数据是字节类型
        if isinstance(inputs, bytes):
            # 使用 ffmpeg_read 函数读取输入数据，并指定采样率
            inputs = ffmpeg_read(inputs, self.feature_extractor.sampling_rate)

        # 如果输入数据是字典
        if isinstance(inputs, dict):
            # 接受"array"键，该键在“datasets”中有定义，以实现更好的集成
            if not ("sampling_rate" in inputs and ("raw" in inputs or "array" in inputs)):
                raise ValueError(
                    "传递字典给 AudioClassificationPipeline 时，字典必须包含一个包含表示音频的 numpy 数组的 "
                    '"raw" 键和一个包含与该数组关联的采样率的 "sampling_rate" 键'
                )

            _inputs = inputs.pop("raw", None)
            if _inputs is None:
                # 从`datasets`中删除不会被使用的路径
                inputs.pop("path", None)
                _inputs = inputs.pop("array", None)
            in_sampling_rate = inputs.pop("sampling_rate")
            inputs = _inputs
            # 如果输入采样率与特征提取器的采样率不同
            if in_sampling_rate != self.feature_extractor.sampling_rate:
                import torch

                if is_torchaudio_available():
                    from torchaudio import functional as F
                else:
                    raise ImportError(
                        "torchaudio is required to resample audio samples in AudioClassificationPipeline. "
                        "The torchaudio package can be installed through: `pip install torchaudio`."
                    )

                # 使用 torch 重新采样音频样本，并转换为 numpy 数组
                inputs = F.resample(
                    torch.from_numpy(inputs), in_sampling_rate, self.feature_extractor.sampling_rate
                ).numpy()

        # 如果输入数据不是 numpy 数组
        if not isinstance(inputs, np.ndarray):
            raise ValueError("我们期望输入为 numpy 数组")
        # 如果输入数据维度不是 1
        if len(inputs.shape) != 1:
            raise ValueError("我们期望为 AudioClassificationPipeline 提供单通道音频输入")

        # 使用 feature_extractor 处理输入数据，返回 PyTorch 张量
        processed = self.feature_extractor(
            inputs, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt"
        )
        return processed

    # 私有方法，前向传播
    def _forward(self, model_inputs):
        # 调用模型进行前向传播
        model_outputs = self.model(**model_inputs)
        return model_outputs
    def postprocess(self, model_outputs, top_k=5):
        # 获取模型输出的logits，并进行softmax操作得到概率值
        probs = model_outputs.logits[0].softmax(-1)
        # 获取top-k的概率值和对应的id
        scores, ids = probs.topk(top_k)

        # 将scores和ids转换为列表形式
        scores = scores.tolist()
        ids = ids.tolist()

        # 根据id获取对应的标签，将score和label组成字典列表
        labels = [{"score": score, "label": self.model.config.id2label[_id]} for score, _id in zip(scores, ids)]

        # 返回标签列表
        return labels
```