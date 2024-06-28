# `.\pipelines\audio_classification.py`

```py
# 导入子进程管理模块
import subprocess
# 导入 Union 用于类型提示
from typing import Union

# 导入 numpy 和 requests 库
import numpy as np
import requests

# 从相对路径导入工具函数和判断 Torch 是否可用的函数
from ..utils import add_end_docstrings, is_torch_available, is_torchaudio_available, logging
# 从当前目录的 base.py 文件中导入 Pipeline 类和初始化管道参数的函数
from .base import Pipeline, build_pipeline_init_args

# 如果 Torch 可用，则从相对路径导入模型映射名称
if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES

# 获取日志记录器
logger = logging.get_logger(__name__)


def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.array:
    """
    使用 ffmpeg 读取音频文件的辅助函数。
    """
    # 将采样率转换为字符串
    ar = f"{sampling_rate}"
    # 设置音频通道数为 1
    ac = "1"
    # 设置转换格式为 f32le
    format_for_conversion = "f32le"
    # 构建 ffmpeg 命令
    ffmpeg_command = [
        "ffmpeg",
        "-i",        # 输入文件（从标准输入流读取）
        "pipe:0",    # 使用标准输入流作为输入
        "-ac",       # 设置音频通道数
        ac,
        "-ar",       # 设置音频采样率
        ar,
        "-f",        # 设置输出格式
        format_for_conversion,
        "-hide_banner",  # 隐藏 ffmpeg 的 banner
        "-loglevel",     # 设置日志级别为 quiet（不输出冗长的日志信息）
        "quiet",
        "pipe:1",    # 使用标准输出流输出结果
    ]

    try:
        # 启动 ffmpeg 子进程，将 bpayload 作为输入，获取标准输出流
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    except FileNotFoundError:
        # 如果找不到 ffmpeg，则抛出 ValueError 异常
        raise ValueError("ffmpeg was not found but is required to load audio files from filename")

    # 与 ffmpeg 进程交互，获取输出流数据
    output_stream = ffmpeg_process.communicate(bpayload)
    out_bytes = output_stream[0]

    # 将输出流数据解析为 np.float32 类型的数组
    audio = np.frombuffer(out_bytes, np.float32)
    # 如果音频数组长度为 0，则抛出 ValueError 异常
    if audio.shape[0] == 0:
        raise ValueError("Malformed soundfile")

    # 返回解析后的音频数据数组
    return audio


# 使用装饰器添加文档字符串，并标记具有特征提取器的管道
@add_end_docstrings(build_pipeline_init_args(has_feature_extractor=True))
class AudioClassificationPipeline(Pipeline):
    """
    使用任意 `AutoModelForAudioClassification` 进行音频分类的管道。此管道预测原始波形或音频文件的类别。
    对于音频文件，需要安装 ffmpeg 支持多种音频格式的解析。

    示例：

    ```
    >>> from transformers import pipeline

    >>> classifier = pipeline(model="superb/wav2vec2-base-superb-ks")
    >>> classifier("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
    [{'score': 0.997, 'label': '_unknown_'}, {'score': 0.002, 'label': 'left'}, {'score': 0.0, 'label': 'yes'}, {'score': 0.0, 'label': 'down'}, {'score': 0.0, 'label': 'stop'}]
    ```

    了解如何在 [管道教程](../pipeline_tutorial) 中使用管道的基础知识。

    此管道可以通过 [`pipeline`] 使用以下任务标识符加载：
    `"audio-classification"`.

    """
    """
    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=audio-classification).
    """
    
    # 初始化方法，接受任意位置参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 设置默认的 top_k 参数为 5，可能会被 model.config 覆盖
        kwargs["top_k"] = 5
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
        
        # 如果模型不是基于 PyTorch 的，抛出 ValueError 异常
        if self.framework != "pt":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")
        
        # 检查模型类型是否属于音频分类映射名称
        self.check_model_type(MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES)

    # 调用实例时的方法，用于分类给定的输入序列
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
        # 调用父类的 __call__ 方法，执行实际的分类操作
        return super().__call__(inputs, **kwargs)

    # 清理参数的私有方法，目前此管道不接受参数
    def _sanitize_parameters(self, top_k=None, **kwargs):
        # 初始化后处理参数字典
        postprocess_params = {}
        # 如果指定了 top_k 参数
        if top_k is not None:
            # 如果 top_k 大于模型配置中的标签数，将其设为模型配置中的标签数
            if top_k > self.model.config.num_labels:
                top_k = self.model.config.num_labels
            # 将 top_k 参数添加到后处理参数字典中
            postprocess_params["top_k"] = top_k
        
        # 返回空字典和两个空元组作为参数
        return {}, {}, postprocess_params
    # 定义预处理方法，处理输入数据以准备用于模型输入
    def preprocess(self, inputs):
        # 如果输入是字符串
        if isinstance(inputs, str):
            # 如果输入以 "http://" 或 "https://" 开头，表明是网络地址
            if inputs.startswith("http://") or inputs.startswith("https://"):
                # 发起网络请求获取内容并返回其二进制数据
                inputs = requests.get(inputs).content
            else:
                # 否则假设输入是本地文件路径，以二进制方式读取文件内容
                with open(inputs, "rb") as f:
                    inputs = f.read()

        # 如果输入是字节流
        if isinstance(inputs, bytes):
            # 使用自定义的 ffmpeg_read 函数读取音频数据，指定采样率
            inputs = ffmpeg_read(inputs, self.feature_extractor.sampling_rate)

        # 如果输入是字典
        if isinstance(inputs, dict):
            # 检查字典是否包含所需的键 "sampling_rate" 和 "raw" 或 "array"
            if not ("sampling_rate" in inputs and ("raw" in inputs or "array" in inputs)):
                # 如果不符合要求，抛出数值错误异常
                raise ValueError(
                    "When passing a dictionary to AudioClassificationPipeline, the dict needs to contain a "
                    '"raw" key containing the numpy array representing the audio and a "sampling_rate" key, '
                    "containing the sampling_rate associated with that array"
                )

            # 从字典中取出 "raw" 数据（如果存在），否则取出 "array" 数据
            _inputs = inputs.pop("raw", None)
            if _inputs is None:
                # 如果没有 "raw" 数据，则移除字典中的 "path" 和 "array" 键，再取出 "array" 数据
                inputs.pop("path", None)
                _inputs = inputs.pop("array", None)
            # 取出输入音频的采样率
            in_sampling_rate = inputs.pop("sampling_rate")
            # 将输入设为取出的数据
            inputs = _inputs
            # 如果输入的采样率不等于特征提取器的采样率
            if in_sampling_rate != self.feature_extractor.sampling_rate:
                import torch

                # 检查是否可以使用 torchaudio 库
                if is_torchaudio_available():
                    from torchaudio import functional as F
                else:
                    # 如果没有 torchaudio 库，抛出导入错误异常
                    raise ImportError(
                        "torchaudio is required to resample audio samples in AudioClassificationPipeline. "
                        "The torchaudio package can be installed through: `pip install torchaudio`."
                    )

                # 使用 torch 和 torchaudio 进行音频重采样，并将结果转换为 numpy 数组
                inputs = F.resample(
                    torch.from_numpy(inputs), in_sampling_rate, self.feature_extractor.sampling_rate
                ).numpy()

        # 如果输入不是 numpy 数组，则抛出数值错误异常
        if not isinstance(inputs, np.ndarray):
            raise ValueError("We expect a numpy ndarray as input")
        # 如果输入的维度不是 1，说明不是单声道音频，抛出数值错误异常
        if len(inputs.shape) != 1:
            raise ValueError("We expect a single channel audio input for AudioClassificationPipeline")

        # 使用特征提取器提取输入的特征，返回处理后的结果
        processed = self.feature_extractor(
            inputs, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt"
        )
        return processed

    # 定义私有方法 _forward，执行模型的前向传播
    def _forward(self, model_inputs):
        # 调用模型进行前向传播，返回模型输出
        model_outputs = self.model(**model_inputs)
        return model_outputs
    # 定义一个方法用于后处理模型输出，生成排名前top_k的标签和对应的概率
    def postprocess(self, model_outputs, top_k=5):
        # 从模型输出中获取概率分布并进行 softmax 归一化
        probs = model_outputs.logits[0].softmax(-1)
        # 获取概率最高的top_k个值和它们的索引
        scores, ids = probs.topk(top_k)

        # 将张量转换为 Python 列表
        scores = scores.tolist()
        ids = ids.tolist()

        # 根据模型配置中的 id2label 映射，生成标签和对应的概率列表
        labels = [{"score": score, "label": self.model.config.id2label[_id]} for score, _id in zip(scores, ids)]

        # 返回标签和对应概率的列表
        return labels
```