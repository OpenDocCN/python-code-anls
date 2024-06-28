# `.\pipelines\zero_shot_audio_classification.py`

```
# 导入所需模块和库
from collections import UserDict  # 导入 UserDict 类，用于自定义字典类型
from typing import Union  # 导入 Union 用于支持多种类型的注解

import numpy as np  # 导入 numpy 库，用于数值计算
import requests  # 导入 requests 库，用于发送 HTTP 请求

# 从相对路径导入工具函数和模块
from ..utils import (
    add_end_docstrings,  # 导入函数 add_end_docstrings，用于添加文档字符串
    logging,  # 导入 logging 模块，用于记录日志
)
# 从本地模块中导入音频分类相关函数
from .audio_classification import ffmpeg_read  # 导入音频处理函数 ffmpeg_read
from .base import Pipeline, build_pipeline_init_args  # 从基础模块导入 Pipeline 类和构建初始化参数函数 build_pipeline_init_args

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)


@add_end_docstrings(build_pipeline_init_args(has_feature_extractor=True, has_tokenizer=True))
# 使用装饰器 add_end_docstrings，为类添加结尾文档字符串，并指定初始化参数的特性
class ZeroShotAudioClassificationPipeline(Pipeline):
    """
    Zero shot audio classification pipeline using `ClapModel`. This pipeline predicts the class of an audio when you
    provide an audio and a set of `candidate_labels`.

    Example:
    ```python
    >>> from transformers import pipeline
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("ashraq/esc50")
    >>> audio = next(iter(dataset["train"]["audio"]))["array"]
    >>> classifier = pipeline(task="zero-shot-audio-classification", model="laion/clap-htsat-unfused")
    >>> classifier(audio, candidate_labels=["Sound of a dog", "Sound of vaccum cleaner"])
    [{'score': 0.9996, 'label': 'Sound of a dog'}, {'score': 0.0004, 'label': 'Sound of vaccum cleaner'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial) This audio
    classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"zero-shot-audio-classification"`. See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=zero-shot-audio-classification).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # 调用父类的初始化方法

        if self.framework != "pt":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")
        # 检查框架是否为 PyTorch，如果不是则抛出 ValueError 异常
        # 暂时没有特定的 FOR_XXX 可用
    # 继承父类的 __call__ 方法，用于给传入的音频分配标签

    def __call__(self, audios: Union[np.ndarray, bytes, str], **kwargs):
        """
        Assign labels to the audio(s) passed as inputs.

        Args:
            audios (`str`, `List[str]`, `np.array` or `List[np.array]`):
                The pipeline handles three types of inputs:
                - A string containing a http link pointing to an audio
                - A string containing a local path to an audio
                - An audio loaded in numpy
            candidate_labels (`List[str]`):
                The candidate labels for this audio
            hypothesis_template (`str`, *optional*, defaults to `"This is a sound of {}"`):
                The sentence used in conjunction with *candidate_labels* to attempt the audio classification by
                replacing the placeholder with the candidate_labels. Then likelihood is estimated by using
                logits_per_audio
        Return:
            A list of dictionaries containing result, one dictionary per proposed label. The dictionaries contain the
            following keys:
            - **label** (`str`) -- The label identified by the model. It is one of the suggested `candidate_label`.
            - **score** (`float`) -- The score attributed by the model for that label (between 0 and 1).
        """
        # 调用父类的 __call__ 方法，处理传入的音频数据和其他参数
        return super().__call__(audios, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        # 初始化预处理参数字典
        preprocess_params = {}
        
        # 如果参数中包含 candidate_labels，将其添加到预处理参数中
        if "candidate_labels" in kwargs:
            preprocess_params["candidate_labels"] = kwargs["candidate_labels"]
        
        # 如果参数中包含 hypothesis_template，将其添加到预处理参数中
        if "hypothesis_template" in kwargs:
            preprocess_params["hypothesis_template"] = kwargs["hypothesis_template"]

        # 返回预处理参数字典和空字典（用于其他参数）
        return preprocess_params, {}, {}
    # 对音频进行预处理，将音频转换为字节流或从 URL 下载音频内容
    def preprocess(self, audio, candidate_labels=None, hypothesis_template="This is a sound of {}."):

        # 如果音频参数是字符串类型且以 "http://" 或 "https://" 开头，则下载远程音频内容
        if isinstance(audio, str):
            if audio.startswith("http://") or audio.startswith("https://"):
                # 实际需要检查协议是否存在，否则无法使用像 http_huggingface_co.png 这样的本地文件
                audio = requests.get(audio).content
            else:
                # 否则假定为本地文件路径，以二进制形式读取音频内容
                with open(audio, "rb") as f:
                    audio = f.read()

        # 如果音频是字节流，则使用特征提取器的采样率将其解码为 numpy 数组
        if isinstance(audio, bytes):
            audio = ffmpeg_read(audio, self.feature_extractor.sampling_rate)

        # 检查音频是否为 numpy 数组
        if not isinstance(audio, np.ndarray):
            raise ValueError("We expect a numpy ndarray as input")
        
        # 检查音频是否为单通道音频
        if len(audio.shape) != 1:
            raise ValueError("We expect a single channel audio input for ZeroShotAudioClassificationPipeline")

        # 使用特征提取器提取音频特征，转换为 PyTorch 张量输入
        inputs = self.feature_extractor(
            [audio], sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt"
        )
        
        # 将候选标签添加到输入中
        inputs["candidate_labels"] = candidate_labels
        
        # 根据模板生成假设序列
        sequences = [hypothesis_template.format(x) for x in candidate_labels]
        
        # 使用分词器处理文本输入，返回适当的张量类型
        text_inputs = self.tokenizer(sequences, return_tensors=self.framework, padding=True)
        
        # 将文本输入添加到输入字典中
        inputs["text_inputs"] = [text_inputs]
        
        # 返回处理后的输入字典
        return inputs

    # 私有方法：模型前向推断
    def _forward(self, model_inputs):
        # 弹出候选标签
        candidate_labels = model_inputs.pop("candidate_labels")
        
        # 弹出文本输入
        text_inputs = model_inputs.pop("text_inputs")
        
        # 如果文本输入是 UserDict 类型，则获取其第一个元素
        if isinstance(text_inputs[0], UserDict):
            text_inputs = text_inputs[0]
        else:
            # 否则为批处理情况，获取其第一个元素的第一个元素
            text_inputs = text_inputs[0][0]
        
        # 使用模型进行推断，传入文本输入和其他模型输入
        outputs = self.model(**text_inputs, **model_inputs)
        
        # 构建模型输出字典
        model_outputs = {
            "candidate_labels": candidate_labels,
            "logits": outputs.logits_per_audio,
        }
        
        # 返回模型输出
        return model_outputs

    # 后处理方法：处理模型输出，生成最终结果
    def postprocess(self, model_outputs):
        # 弹出候选标签
        candidate_labels = model_outputs.pop("candidate_labels")
        
        # 获取 logits
        logits = model_outputs["logits"][0]

        # 如果使用 PyTorch 框架，则对 logits 进行 softmax 处理，得到概率分数
        if self.framework == "pt":
            probs = logits.softmax(dim=0)
            scores = probs.tolist()
        else:
            # 不支持的框架类型
            raise ValueError("`tf` framework not supported.")

        # 根据分数排序候选标签，生成结果列表
        result = [
            {"score": score, "label": candidate_label}
            for score, candidate_label in sorted(zip(scores, candidate_labels), key=lambda x: -x[0])
        ]
        
        # 返回最终结果
        return result
```