# `.\pipelines\video_classification.py`

```
# 导入所需模块和类
from io import BytesIO  # 导入 BytesIO 类
from typing import List, Union  # 导入 List 和 Union 类型

import requests  # 导入 requests 模块

from ..utils import add_end_docstrings, is_decord_available, is_torch_available, logging, requires_backends  # 导入自定义模块和函数
from .base import Pipeline, build_pipeline_init_args  # 从 base 模块中导入 Pipeline 和 build_pipeline_init_args 函数

# 如果 decord 可用，则导入相应模块
if is_decord_available():
    import numpy as np  # 导入 numpy 模块
    from decord import VideoReader  # 从 decord 模块中导入 VideoReader 类

# 如果 torch 可用，则从自动模型中导入 MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES 变量
if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES  # 从自动模型中导入 MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES 变量

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


# 添加文档字符串的装饰器
@add_end_docstrings(build_pipeline_init_args(has_image_processor=True))
class VideoClassificationPipeline(Pipeline):
    """
    Video classification pipeline using any `AutoModelForVideoClassification`. This pipeline predicts the class of a
    video.

    This video classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"video-classification"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=video-classification).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 调用父类的初始化方法
        requires_backends(self, "decord")  # 确保 decord 可用
        self.check_model_type(MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES)  # 检查模型类型

    def _sanitize_parameters(self, top_k=None, num_frames=None, frame_sampling_rate=None):
        # 对参数进行预处理
        preprocess_params = {}
        if frame_sampling_rate is not None:
            preprocess_params["frame_sampling_rate"] = frame_sampling_rate
        if num_frames is not None:
            preprocess_params["num_frames"] = num_frames

        # 对参数进行后处理
        postprocess_params = {}
        if top_k is not None:
            postprocess_params["top_k"] = top_k
        return preprocess_params, {}, postprocess_params  # 返回处理后的参数
    def __call__(self, videos: Union[str, List[str]], **kwargs):
        """
        将标签分配给作为输入传递的视频。

        Args:
            videos (`str`, `List[str]`):
                管道处理三种类型的视频：

                - 包含指向视频的 HTTP 链接的字符串
                - 包含指向视频的本地路径的字符串

                管道接受单个视频或视频批处理，必须作为字符串传递。
                批处理中的所有视频必须具有相同的格式：全部是 HTTP 链接或全部是本地路径。
            top_k (`int`, *可选*, 默认为 5):
                管道将返回的顶部标签数。如果提供的数字高于模型配置中可用的标签数，将默认为标签数。
            num_frames (`int`, *可选*, 默认为 `self.model.config.num_frames`):
                从视频中抽样的帧数。如果未提供，则默认为模型配置中指定的帧数。
            frame_sampling_rate (`int`, *可选*, 默认为 1):
                用于从视频中选择帧的采样率。如果未提供，则默认为 1，即每帧都将被使用。

        Return:
            包含结果的字典或字典列表。如果输入为单个视频，则返回一个字典；如果输入为多个视频，则返回相应的字典列表。

            字典包含以下键：

            - **label** (`str`) -- 模型识别的标签。
            - **score** (`int`) -- 模型为该标签分配的分数。
        """
        return super().__call__(videos, **kwargs)

    def preprocess(self, video, num_frames=None, frame_sampling_rate=1):
        """
        预处理视频以用于模型输入。

        Args:
            video (`str` or `BytesIO`):
                视频的路径或 BytesIO 对象。
            num_frames (`int`, *可选*):
                从视频中抽样的帧数。如果未提供，则默认为 self.model.config.num_frames。
            frame_sampling_rate (`int`, *可选*, 默认为 1):
                用于从视频中选择帧的采样率。

        Returns:
            模型输入的字典表示形式。

        Raises:
            ValueError: 如果视频格式不支持或无法识别。
        """
        if num_frames is None:
            num_frames = self.model.config.num_frames

        if video.startswith("http://") or video.startswith("https://"):
            # 如果视频是一个 HTTP/HTTPS 链接，则从网络获取视频内容
            video = BytesIO(requests.get(video).content)

        # 创建视频阅读器对象
        videoreader = VideoReader(video)
        videoreader.seek(0)  # 将视频的读取位置设置为起始位置

        start_idx = 0
        end_idx = num_frames * frame_sampling_rate - 1
        # 生成需要抽样的帧索引
        indices = np.linspace(start_idx, end_idx, num=num_frames, dtype=np.int64)

        # 从视频中获取指定索引的帧
        video = videoreader.get_batch(indices).asnumpy()
        video = list(video)

        # 使用图像处理器将视频帧处理为模型输入
        model_inputs = self.image_processor(video, return_tensors=self.framework)
        return model_inputs

    def _forward(self, model_inputs):
        """
        将模型输入传递给模型并获取模型输出。

        Args:
            model_inputs:
                模型的输入数据字典。

        Returns:
            模型的输出结果。
        """
        model_outputs = self.model(**model_inputs)
        return model_outputs
    # 对模型输出进行后处理，返回top_k个预测结果
    def postprocess(self, model_outputs, top_k=5):
        # 如果top_k大于模型配置的标签数，则将top_k设为模型配置的标签数
        if top_k > self.model.config.num_labels:
            top_k = self.model.config.num_labels

        # 根据选择的框架进行后处理
        if self.framework == "pt":
            # 对PyTorch框架的模型输出进行softmax处理，获取概率分布
            probs = model_outputs.logits.softmax(-1)[0]
            # 获取top_k个最高概率对应的分数和标签ID
            scores, ids = probs.topk(top_k)
        else:
            # 如果框架不是pt（PyTorch），则抛出错误
            raise ValueError(f"Unsupported framework: {self.framework}")

        # 将分数和标签ID转换为列表形式
        scores = scores.tolist()
        ids = ids.tolist()
        
        # 返回包含分数和对应标签的列表字典
        return [{"score": score, "label": self.model.config.id2label[_id]} for score, _id in zip(scores, ids)]
```