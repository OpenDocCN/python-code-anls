# `.\transformers\pipelines\zero_shot_object_detection.py`

```
# 从 typing 模块中导入 Any, Dict, List, Union 类型
# 从 ..utils 模块中导入 add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends 函数
# 从 .base 模块中导入 PIPELINE_INIT_ARGS, ChunkPipeline 类
from typing import Any, Dict, List, Union
from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends
from .base import PIPELINE_INIT_ARGS, ChunkPipeline

# 如果 is_vision_available() 返回 True，则导入 PIL 模块中的 Image 类和 ..image_utils 模块中的 load_image 函数
if is_vision_available():
    from PIL import Image
    from ..image_utils import load_image

# 如果 is_torch_available() 返回 True，则导入 torch 模块
# 从 transformers.modeling_outputs 模块中导入 BaseModelOutput 类
# 从 ..models.auto.modeling_auto 模块中导入 MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES 变量
if is_torch_available():
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    from ..models.auto.modeling_auto import MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES

# 从 logging 模块中导入 get_logger 函数
logger = logging.get_logger(__name__)

# 使用 @add_end_docstrings(PIPELINE_INIT_ARGS) 装饰器为 ZeroShotObjectDetectionPipeline 类添加结尾的文档说明
class ZeroShotObjectDetectionPipeline(ChunkPipeline):
    """
    Zero shot object detection pipeline using `OwlViTForObjectDetection`. This pipeline predicts bounding boxes of
    objects when you provide an image and a set of `candidate_labels`.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> detector = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection")
    >>> detector(
    ...     "http://images.cocodataset.org/val2017/000000039769.jpg",
    ...     candidate_labels=["cat", "couch"],
    ... )
    [{'score': 0.287, 'label': 'cat', 'box': {'xmin': 324, 'ymin': 20, 'xmax': 640, 'ymax': 373}}, {'score': 0.254, 'label': 'cat', 'box': {'xmin': 1, 'ymin': 55, 'xmax': 315, 'ymax': 472}}, {'score': 0.121, 'label': 'couch', 'box': {'xmin': 4, 'ymin': 0, 'xmax': 642, 'ymax': 476}}]

    >>> detector(
    ...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
    ...     candidate_labels=["head", "bird"],
    ... )
    [{'score': 0.119, 'label': 'bird', 'box': {'xmin': 71, 'ymin': 170, 'xmax': 410, 'ymax': 508}}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This object detection pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"zero-shot-object-detection"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=zero-shot-object-detection).
    """

    # 初始化方法
    def __init__(self, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 如果当前 framework 不是 "tf"，则抛出 ValueError 异常
        if self.framework == "tf":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")
        # 要求当前环境满足 vision 后端的需求
        requires_backends(self, "vision")
        # 检查模型类型是否在 MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES 中
        self.check_model_type(MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES)

    # __call__ 方法用于执行对象实例，接受 image、candidate_labels 和 kwargs 三个参数
    def __call__(
        self,
        image: Union[str, "Image.Image", List[Dict[str, Any]]],
        candidate_labels: Union[str, List[str]] = None,
        **kwargs,
    # 对输入参数进行清理处理，获取预处理参数和后处理参数
    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        # 如果参数中包含"timeout"，将其添加到预处理参数中
        if "timeout" in kwargs:
            preprocess_params["timeout"] = kwargs["timeout"]
        postprocess_params = {}
        # 如果参数中包含"threshold"，将其添加到后处理参数中
        if "threshold" in kwargs:
            postprocess_params["threshold"] = kwargs["threshold"]
        # 如果参数中包含"top_k"，将其添加到后处理参数中
        if "top_k" in kwargs:
            postprocess_params["top_k"] = kwargs["top_k"]
        # 返回预处理参数、空字典和后处理参数
        return preprocess_params, {}, postprocess_params

    # 对输入数据进行预处理
    def preprocess(self, inputs, timeout=None):
        # 加载图像数据，设置超时时间
        image = load_image(inputs["image"], timeout=timeout)
        candidate_labels = inputs["candidate_labels"]
        # 如果候选标签是字符串，以逗号分割成列表
        if isinstance(candidate_labels, str):
            candidate_labels = candidate_labels.split(",")

        # 设置目标大小为图像的高度和宽度
        target_size = torch.tensor([[image.height, image.width]], dtype=torch.int32)
        # 遍历候选标签，生成文本输入和图像特征，使用生成器函数返回结果
        for i, candidate_label in enumerate(candidate_labels):
            text_inputs = self.tokenizer(candidate_label, return_tensors=self.framework)
            image_features = self.image_processor(image, return_tensors=self.framework)
            yield {
                "is_last": i == len(candidate_labels) - 1,
                "target_size": target_size,
                "candidate_label": candidate_label,
                **text_inputs,
                **image_features,
            }

    # 模型正向传播函数
    def _forward(self, model_inputs):
        # 弹出模型输入中的目标大小、候选标签和是否是最后一个标签
        target_size = model_inputs.pop("target_size")
        candidate_label = model_inputs.pop("candidate_label")
        is_last = model_inputs.pop("is_last")

        # 模型进行正向传播操作
        outputs = self.model(**model_inputs)

        # 构建模型输出结果字典
        model_outputs = {"target_size": target_size, "candidate_label": candidate_label, "is_last": is_last, **outputs}
        return model_outputs

    # 对模型输出进行后处理
    def postprocess(self, model_outputs, threshold=0.1, top_k=None):
        results = []
        # 遍历模型输出结果
        for model_output in model_outputs:
            label = model_output["candidate_label"]
            # 将模型输出结果转换成BaseModelOutput对象
            model_output = BaseModelOutput(model_output)
            # 对目标检测模型输出结果进行后处理
            outputs = self.image_processor.post_process_object_detection(
                outputs=model_output, threshold=threshold, target_sizes=model_output["target_size"]
            )[0]

            # 遍历输出结果中的得分，坐标框信息，构建结果字典并添加到结果列表中
            for index in outputs["scores"].nonzero():
                score = outputs["scores"][index].item()
                box = self._get_bounding_box(outputs["boxes"][index][0])

                result = {"score": score, "label": label, "box": box}
                results.append(result)

        # 根据得分进行降序排序
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        # 如果设定了top_k值，只取前top_k个结果
        if top_k:
            results = results[:top_k]

        return results
    # 定义获取边界框的函数，参数为带类型提示的张量，返回类型为包含边界框坐标的字典
    def _get_bounding_box(self, box: "torch.Tensor") -> Dict[str, int]:
        """
        Turns list [xmin, xmax, ymin, ymax] into dict { "xmin": xmin, ... }

        Args:
            box (`torch.Tensor`): Tensor containing the coordinates in corners format.

        Returns:
            bbox (`Dict[str, int]`): Dict containing the coordinates in corners format.
        """
        # 如果框架不是 PyTorch，则抛出数值错误
        if self.framework != "pt":
            raise ValueError("The ZeroShotObjectDetectionPipeline is only available in PyTorch.")
        # 将张量转换为整数列表，分别表示左上角和右下角坐标
        xmin, ymin, xmax, ymax = box.int().tolist()
        # 创建包含边界框坐标的字典
        bbox = {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        }
        # 返回边界框字典
        return bbox
```