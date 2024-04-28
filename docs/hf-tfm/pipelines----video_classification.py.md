# `.\transformers\pipelines\video_classification.py`

```py
# 导入所需模块
from io import BytesIO
from typing import List, Union
import requests

# 从自定义模块中导入指定函数和变量
from ..utils import add_end_docstrings, is_decord_available, is_torch_available, logging, requires_backends
from .base import PIPELINE_INIT_ARGS, Pipeline

# 如果 decord 模块可用，则导入 numpy 和 VideoReader 类
if is_decord_available():
    import numpy as np
    from decord import VideoReader

# 如果 torch 模块可用，则导入模型映射名称
if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 添加结束文档注释，并创建视频分类流水线类
@add_end_docstrings(PIPELINE_INIT_ARGS)
class VideoClassificationPipeline(Pipeline):
    """
    Video classification pipeline using any `AutoModelForVideoClassification`. This pipeline predicts the class of a video.

    This video classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"video-classification"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=video-classification).
    """

    # 初始化视频分类流水线对象
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
        # 要求后端支持 decord 模块
        requires_backends(self, "decord")
        # 检查模型类型是否在视频分类模型映射名称中
        self.check_model_type(MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES)

    # 清理参数，确保输入参数合法
    def _sanitize_parameters(self, top_k=None, num_frames=None, frame_sampling_rate=None):
        preprocess_params = {}
        if frame_sampling_rate is not None:
            preprocess_params["frame_sampling_rate"] = frame_sampling_rate
        if num_frames is not None:
            preprocess_params["num_frames"] = num_frames

        postprocess_params = {}
        if top_k is not None:
            postprocess_params["top_k"] = top_k
        return preprocess_params, {}, postprocess_params
    # 为视频（或视频列表）添加标签
    def __call__(self, videos: Union[str, List[str]], **kwargs):
        """
        Assign labels to the video(s) passed as inputs.

        Args:
            videos (`str`, `List[str]`):
                The pipeline handles three types of videos:

                - A string containing a http link pointing to a video
                - A string containing a local path to a video

                The pipeline accepts either a single video or a batch of videos, which must then be passed as a string.
                Videos in a batch must all be in the same format: all as http links or all as local paths.
            top_k (`int`, *optional*, defaults to 5):
                The number of top labels that will be returned by the pipeline. If the provided number is higher than
                the number of labels available in the model configuration, it will default to the number of labels.
            num_frames (`int`, *optional*, defaults to `self.model.config.num_frames`):
                The number of frames sampled from the video to run the classification on. If not provided, will default
                to the number of frames specified in the model configuration.
            frame_sampling_rate (`int`, *optional*, defaults to 1):
                The sampling rate used to select frames from the video. If not provided, will default to 1, i.e. every
                frame will be used.

        Return:
            A dictionary or a list of dictionaries containing result. If the input is a single video, will return a
            dictionary, if the input is a list of several videos, will return a list of dictionaries corresponding to
            the videos.

            The dictionaries contain the following keys:

            - **label** (`str`) -- The label identified by the model.
            - **score** (`int`) -- The score attributed by the model for that label.
        """
        # 调用父类方法并传递参数
        return super().__call__(videos, **kwargs)

    # 预处理视频，返回模型输入
    def preprocess(self, video, num_frames=None, frame_sampling_rate=1):
        # 如果未指定帧数，则使用模型配置的帧数
        if num_frames is None:
            num_frames = self.model.config.num_frames

        # 如果视频是http链接，则将其转换为字节流
        if video.startswith("http://") or video.startswith("https://"):
            video = BytesIO(requests.get(video).content)

        videoreader = VideoReader(video)
        videoreader.seek(0)

        # 计算帧索引
        start_idx = 0
        end_idx = num_frames * frame_sampling_rate - 1
        indices = np.linspace(start_idx, end_idx, num=num_frames, dtype=np.int64)

        # 从视频中获取批量帧数据
        video = videoreader.get_batch(indices).asnumpy()
        video = list(video)

        # 处理视频数据，并返回模型输入
        model_inputs = self.image_processor(video, return_tensors=self.framework)
        return model_inputs

    # 模型正向传播
    def _forward(self, model_inputs):
        # 运行模型，返回模型输出
        model_outputs = self.model(**model_inputs)
        return model_outputs
        # 对模型输出进行后处理，返回top_k概率最高的标签和对应的分数
        if top_k > self.model.config.num_labels:  # 如果top_k大于模型配置的标签数量，则将top_k设为标签数量
            top_k = self.model.config.num_labels

        if self.framework == "pt":  # 如果使用的是PyTorch框架
            # 对模型输出的logits进行softmax处理，得到概率
            probs = model_outputs.logits.softmax(-1)[0]
            # 取top_k概率最高的分数和对应的标签id
            scores, ids = probs.topk(top_k)
        else:  # 如果使用的框架不是PyTorch
            raise ValueError(f"Unsupported framework: {self.framework}")  # 抛出不支持的框架数值错误

        scores = scores.tolist()  # 将分数转换为列表
        ids = ids.tolist()  # 将标签id转换为列表
        # 返回包含top_k概率最高标签和对应分数的字典列表
        return [{"score": score, "label": self.model.config.id2label[_id]} for score, _id in zip(scores, ids)]
```