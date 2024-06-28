# `.\pipelines\zero_shot_object_detection.py`

```py
from typing import Any, Dict, List, Union  # 导入需要的类型提示模块

from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends  # 导入自定义工具函数和模块
from .base import ChunkPipeline, build_pipeline_init_args  # 导入基础类和初始化函数构建器


if is_vision_available():  # 如果视觉处理模块可用
    from PIL import Image  # 导入PIL图像处理库中的Image模块
    from ..image_utils import load_image  # 从自定义图像处理工具中导入加载图像的函数

if is_torch_available():  # 如果PyTorch可用
    import torch  # 导入PyTorch模块
    from transformers.modeling_outputs import BaseModelOutput  # 导入模型输出基类
    from ..models.auto.modeling_auto import MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES  # 导入零样本对象检测模型映射名称

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


@add_end_docstrings(build_pipeline_init_args(has_image_processor=True))  # 添加文档字符串的装饰器，指定初始化参数为具有图像处理器
class ZeroShotObjectDetectionPipeline(ChunkPipeline):  # 定义零样本对象检测流水线，继承自ChunkPipeline基类
    """
    Zero shot object detection pipeline using `OwlViTForObjectDetection`. This pipeline predicts bounding boxes of
    objects when you provide an image and a set of `candidate_labels`.

    Example:

    ```
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

    def __init__(self, **kwargs):  # 定义初始化方法，接受任意关键字参数
        super().__init__(**kwargs)  # 调用父类的初始化方法，传递所有接收到的关键字参数

        if self.framework == "tf":  # 如果当前框架是TensorFlow
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")  # 抛出错误，表明该类只在PyTorch中可用

        requires_backends(self, "vision")  # 确保必要的后端模块可用，这里要求视觉处理模块可用
        self.check_model_type(MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES)  # 检查当前模型类型是否符合零样本对象检测模型的映射名称

    def __call__(  # 定义对象实例可调用的方法
        self,
        image: Union[str, "Image.Image", List[Dict[str, Any]]],  # 图像参数可以是字符串、PIL图像对象或包含字典的列表
        candidate_labels: Union[str, List[str]] = None,  # 候选标签可以是字符串或字符串列表，默认为None
        **kwargs,  # 允许接收额外的关键字参数
   `
# 定义一个方法用于清理参数
def _sanitize_parameters(self, **kwargs):
    # 创建一个空的预处理参数字典
    preprocess_params = {}
    # 如果参数中包含超时(timeout)，将其加入预处理参数中
    if "timeout" in kwargs:
        preprocess_params["timeout"] = kwargs["timeout"]
    
    # 创建一个空的后处理参数字典
    postprocess_params = {}
    # 如果参数中包含阈值(threshold)，将其加入后处理参数中
    if "threshold" in kwargs:
        postprocess_params["threshold"] = kwargs["threshold"]
    # 如果参数中包含前 k 个(top_k)，将其加入后处理参数中
    if "top_k" in kwargs:
        postprocess_params["top_k"] = kwargs["top_k"]
    
    # 返回预处理参数字典、空字典和后处理参数字典
    return preprocess_params, {}, postprocess_params

# 定义一个预处理方法
def preprocess(self, inputs, timeout=None):
    # 加载图像，并设定超时时间
    image = load_image(inputs["image"], timeout=timeout)
    # 获取候选标签
    candidate_labels = inputs["candidate_labels"]
    # 如果候选标签是字符串，则按逗号分隔
    if isinstance(candidate_labels, str):
        candidate_labels = candidate_labels.split(",")

    # 创建目标尺寸张量
    target_size = torch.tensor([[image.height, image.width]], dtype=torch.int32)
    
    # 遍历候选标签
    for i, candidate_label in enumerate(candidate_labels):
        # 使用分词器处理候选标签，返回张量
        text_inputs = self.tokenizer(candidate_label, return_tensors=self.framework)
        # 使用图像处理器处理图像，返回张量
        image_features = self.image_processor(image, return_tensors=self.framework)
        
        # 生成字典，包括是否最后一个、目标尺寸、候选标签及其它特征
        yield {
            "is_last": i == len(candidate_labels) - 1,
            "target_size": target_size,
            "candidate_label": candidate_label,
            **text_inputs,
            **image_features,
        }

# 定义一个前向方法
def _forward(self, model_inputs):
    # 弹出目标尺寸、候选标签和是否最后一个标志
    target_size = model_inputs.pop("target_size")
    candidate_label = model_inputs.pop("candidate_label")
    is_last = model_inputs.pop("is_last")

    # 使用模型处理输入，返回输出
    outputs = self.model(**model_inputs)

    # 创建模型输出字典，包括目标尺寸、候选标签、是否最后一个及其它输出
    model_outputs = {"target_size": target_size, "candidate_label": candidate_label, "is_last": is_last, **outputs}
    return model_outputs

# 定义一个后处理方法
def postprocess(self, model_outputs, threshold=0.1, top_k=None):
    # 存储结果列表
    results = []
    
    # 遍历模型输出
    for model_output in model_outputs:
        # 获取候选标签
        label = model_output["candidate_label"]
        # 将模型输出封装成基本模型输出对象
        model_output = BaseModelOutput(model_output)
        
        # 使用图像处理器后处理目标检测结果，返回输出
        outputs = self.image_processor.post_process_object_detection(
            outputs=model_output, threshold=threshold, target_sizes=model_output["target_size"]
        )[0]

        # 遍历输出的分eshold, target_sizes=model_output["target_size"]
            )[0]

            # 遍历输出结果中的得分，生成包含得分、标签和边界框的结果字典，并添加到结果列表中
            for index in outputs["scores"].nonzero():
                score = outputs["scores"][index].item()
                box = self._get_bounding_box(outputs["boxes"][index][0])

                result = {"score": score, "label": label, "box": box}
                results.append(result)

        # 按得分倒序排列结果列表
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        # 如果指定了 top_k 参数，则返回前 top_k 个结果
        if top_k:
            results = results[:top_k]

        return results
    # 定义一个方法 `_get_bounding_box`，用于将列表 [xmin, xmax, ymin, ymax] 转换为包含这些坐标的字典
    def _get_bounding_box(self, box: "torch.Tensor") -> Dict[str, int]:
        """
        Turns list [xmin, xmax, ymin, ymax] into dict { "xmin": xmin, ... }

        Args:
            box (`torch.Tensor`): Tensor containing the coordinates in corners format.

        Returns:
            bbox (`Dict[str, int]`): Dict containing the coordinates in corners format.
        """
        # 检查当前所用的深度学习框架是否为 PyTorch，若不是则抛出 ValueError 异常
        if self.framework != "pt":
            raise ValueError("The ZeroShotObjectDetectionPipeline is only available in PyTorch.")
        # 将输入的 box 张量转换为整数列表，并将其转换为 Python 中的标准列表形式
        xmin, ymin, xmax, ymax = box.int().tolist()
        # 创建包含坐标的字典 bbox，键为坐标名，值为对应的坐标值
        bbox = {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        }
        # 返回坐标字典 bbox
        return bbox
```