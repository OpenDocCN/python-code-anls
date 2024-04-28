# `.\transformers\pipelines\zero_shot_audio_classification.py`

```
# 导入所需模块和库
from collections import UserDict  # 导入UserDict类，用于创建自定义字典类型
from typing import Union  # 导入Union类型，用于指定多种可能的类型

import numpy as np  # 导入numpy库，并简化其命名为np，用于处理数组数据
import requests  # 导入requests库，用于发送HTTP请求

from ..utils import (  # 从上层目录的utils模块中导入指定函数和变量
    add_end_docstrings,  # 导入添加末尾文档字符串的函数
    logging,  # 导入日志记录模块
)
from .audio_classification import ffmpeg_read  # 从当前目录下的audio_classification模块中导入ffmpeg_read函数
from .base import PIPELINE_INIT_ARGS, Pipeline  # 从当前目录下的base模块中导入指定变量和类

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 使用装饰器添加末尾文档字符串
@add_end_docstrings(PIPELINE_INIT_ARGS)
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

    # 初始化方法
    def __init__(self, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 检查当前使用的框架是否为PyTorch，若不是则抛出异常
        if self.framework != "pt":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")
        # 暂无特定的FOR_XXX可用
    # 用于对输入的音频进行标签分配
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
        return super().__call__(audios, **kwargs)

    # 用于清理和过滤参数
    def _sanitize_parameters(self, **kwargs):
        # 初始化预处理参数
        preprocess_params = {}
        # 如果传入的参数中包含 "candidate_labels"，则将其添加到预处理参数中
        if "candidate_labels" in kwargs:
            preprocess_params["candidate_labels"] = kwargs["candidate_labels"]
        # 如果传入的参数中包含 "hypothesis_template"，则将其添加到预处理参数中
        if "hypothesis_template" in kwargs:
            preprocess_params["hypothesis_template"] = kwargs["hypothesis_template"]

        # 返回预处理参数
        return preprocess_params, {}, {}
    def preprocess(self, audio, candidate_labels=None, hypothesis_template="This is a sound of {}."):
        # 预处理音频数据，将音频数据转换为适合模型输入的格式
        if isinstance(audio, str):
            if audio.startswith("http://") or audio.startswith("https://"):
                # 如果音频是一个 URL，则通过 requests 模块获取音频内容
                # 需要检查实际的协议，否则无法使用本地文件，如 http_huggingface_co.png
                audio = requests.get(audio).content
            else:
                # 如果音频是一个文件路径，则读取文件内容
                with open(audio, "rb") as f:
                    audio = f.read()

        if isinstance(audio, bytes):
            # 如果音频是字节类型，则使用 ffmpeg_read 函数将其转换为 ndarray 类型
            audio = ffmpeg_read(audio, self.feature_extractor.sampling_rate)

        if not isinstance(audio, np.ndarray):
            raise ValueError("We expect a numpy ndarray as input")
        if len(audio.shape) != 1:
            raise ValueError("We expect a single channel audio input for ZeroShotAudioClassificationPipeline")

        # 使用特征提取器处理音频数据，返回模型输入所需的格式
        inputs = self.feature_extractor(
            [audio], sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt"
        )
        inputs["candidate_labels"] = candidate_labels
        # 根据候选标签模板生成假设序列
        sequences = [hypothesis_template.format(x) for x in candidate_labels]
        # 使用分词器处理假设序列，返回模型输入所需的格式
        text_inputs = self.tokenizer(sequences, return_tensors=self.framework, padding=True)
        inputs["text_inputs"] = [text_inputs]
        return inputs

    def _forward(self, model_inputs):
        # 执行前向传播操作，获取模型输出
        candidate_labels = model_inputs.pop("candidate_labels")
        text_inputs = model_inputs.pop("text_inputs")
        if isinstance(text_inputs[0], UserDict):
            text_inputs = text_inputs[0]
        else:
            # 批处理情况下
            text_inputs = text_inputs[0][0]

        outputs = self.model(**text_inputs, **model_inputs)

        # 整理模型输出，包括候选标签和 logits
        model_outputs = {
            "candidate_labels": candidate_labels,
            "logits": outputs.logits_per_audio,
        }
        return model_outputs

    def postprocess(self, model_outputs):
        # 后处理模型输出，将 logits 转换为概率，排序后返回结果
        candidate_labels = model_outputs.pop("candidate_labels")
        logits = model_outputs["logits"][0]

        if self.framework == "pt":
            probs = logits.softmax(dim=0)
            scores = probs.tolist()
        else:
            raise ValueError("`tf` framework not supported.")

        # 整理结果，包括分数和标签，根据分数排序
        result = [
            {"score": score, "label": candidate_label}
            for score, candidate_label in sorted(zip(scores, candidate_labels), key=lambda x: -x[0])
        ]
        return result
```