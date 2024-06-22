# `.\transformers\pipelines\object_detection.py`

```py
# 从typing模块导入Any, Dict, List, Union类
from typing import Any, Dict, List, Union

# 从..utils包中导入add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends方法
from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends
# 从.base模块中导入PIPELINE_INIT_ARGS, Pipeline类
from .base import PIPELINE_INIT_ARGS, Pipeline

# 如果torch可用，则导入torch模块，并从..models.auto.modeling_auto模块导入MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
if is_torch_available():
    import torch
    from ..models.auto.modeling_auto import (
        MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
    )

# 从logging模块导入logger对象
logger = logging.get_logger(__name__)

# 定义Prediction为包含任意类型的字典，Predictions为Prediction的列表
Prediction = Dict[str, Any]
Predictions = List[Prediction]

# 使用装饰器向ObjectDetectionPipeline类添加文档字符串，文档字符串内容为PIPELINE_INIT_ARGS
@add_end_docstrings(PIPELINE_INIT_ARGS)
# 定义ObjectDetectionPipeline类，继承自Pipeline基类
class ObjectDetectionPipeline(Pipeline):
    """
    Object detection pipeline using any `AutoModelForObjectDetection`. This pipeline predicts bounding boxes of objects
    and their classes.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> detector = pipeline(model="facebook/detr-resnet-50")
    >>> detector("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    [{'score': 0.997, 'label': 'bird', 'box': {'xmin': 69, 'ymin': 171, 'xmax': 396, 'ymax': 507}}, {'score': 0.999, 'label': 'bird', 'box': {'xmin': 398, 'ymin': 105, 'xmax': 767, 'ymax': 507}}]

    >>> # x, y  are expressed relative to the top left hand corner.
    ```py

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This object detection pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"object-detection"`.

    See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=object-detection).
    """

    # 定义初始化方法，接收任意多的位置参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用父类Pipeline的初始化方法
        super().__init__(*args, **kwargs)

        # 如果框架是"tf"，则抛出带有错误信息的异常提示
        if self.framework == "tf":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

        # 确保环境中有"vision"后端
        requires_backends(self, "vision")
        # 复制MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES到mapping，并更新为MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
        mapping = MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES.copy()
        mapping.update(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES)
        # 检查模型类型是否匹配规定的映射
        self.check_model_type(mapping)

    # 定义_sanitize_parameters方法，处理参数
    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        # 如果关键字参数中有"timeout"，则加入预处理参数中
        if "timeout" in kwargs:
            preprocess_params["timeout"] = kwargs["timeout"]
        postprocess_kwargs = {}
        # 如果关键字参数中有"threshold"，则加入后处理参数中
        if "threshold" in kwargs:
            postprocess_kwargs["threshold"] = kwargs["threshold"]
        # 返回预处理参数、空字典、后处理参数
        return preprocess_params, {}, postprocess_kwargs
    # 定义一个特殊方法，该方法接收任意数量的位置参数和关键字参数，返回值可以是 Predictions 或 List[Prediction] 类型
    def __call__(self, *args, **kwargs) -> Union[Predictions, List[Prediction]]:
        """
        检测输入的图像中的对象（边界框和类别）。

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                pipeline 处理三种类型的图像：

                - 包含指向图像的 HTTP(S) 链接的字符串
                - 包含图像的本地路径的字符串
                - 直接在 PIL 中加载的图像

                pipeline 接受单个图像或一批图像。批处理中的图像必须都是相同格式：全都是 HTTP(S) 链接，全都是本地路径，或者全都是 PIL 图像。
            threshold (`float`, *optional*, defaults to 0.9):
                进行预测所需的概率阈值。
            timeout (`float`, *optional*, defaults to None):
                从网页获取图像的最大等待时间（秒）。如果为 None，没有设置超时时间，调用可能会一直阻塞。

        Return:
            包含结果的字典列表或字典列表的列表。如果输入是单个图像，将返回一个字典列表；如果输入是多个图像，将返回一个与每个图像对应的字典列表的列表。

            字典包含以下键：

            - **label** (`str`) -- 模型识别的类别标签。
            - **score** (`float`) -- 模型为该标签打分。
            - **box** (`List[Dict[str, int]]`) -- 在图像原始大小中检测到的对象的边界框。
        """

        return super().__call__(*args, **kwargs)

    # 对图像进行预处理
    def preprocess(self, image, timeout=None):
        # 加载图像，获取图像对象
        image = load_image(image, timeout=timeout)
        # 获取目标尺寸
        target_size = torch.IntTensor([[image.height, image.width]])
        # 处理图像输入，转换成 PyTorch 格式
        inputs = self.image_processor(images=[image], return_tensors="pt")
        # 如果存在分词器，将文本和边界框传给分词器，返回 PyTorch 格式
        if self.tokenizer is not None:
            inputs = self.tokenizer(text=inputs["words"], boxes=inputs["boxes"], return_tensors="pt")
        inputs["target_size"] = target_size
        return inputs

    # 执行模型的前向传播
    def _forward(self, model_inputs):
        # 弹出目标尺寸，执行模型的前向传播
        target_size = model_inputs.pop("target_size")
        outputs = self.model(**model_inputs)
        # 包装模型输出，添加目标尺寸
        model_outputs = outputs.__class__({"target_size": target_size, **outputs})
        # 如果存在分词器，添加模型输入中的边界框
        if self.tokenizer is not None:
            model_outputs["bbox"] = model_inputs["bbox"]
        return model_outputs
    def postprocess(self, model_outputs, threshold=0.9):
        # 获取目标尺寸
        target_size = model_outputs["target_size"]
        # 如果存在分词器
        if self.tokenizer is not None:
            # 这是 LayoutLMForTokenClassification 的一个变体。
            # OCR 获取了框并且模型对单词进行了分类。
            height, width = target_size[0].tolist()

            # 将边界框反归一化
            def unnormalize(bbox):
                return self._get_bounding_box(
                    torch.Tensor(
                        [
                            (width * bbox[0] / 1000),
                            (height * bbox[1] / 1000),
                            (width * bbox[2] / 1000),
                            (height * bbox[3] / 1000),
                        ]
                    )
                )

            # 计算分数和类别
            scores, classes = model_outputs["logits"].squeeze(0).softmax(dim=-1).max(dim=-1)
            # 将类别转换为标签
            labels = [self.model.config.id2label[prediction] for prediction in classes.tolist()]
            # 反归一化边界框
            boxes = [unnormalize(bbox) for bbox in model_outputs["bbox"].squeeze(0)]
            # 设置键和注释
            keys = ["score", "label", "box"]
            # 创建注释列表，筛选出分数高于阈值的项
            annotation = [dict(zip(keys, vals)) for vals in zip(scores.tolist(), labels, boxes) if vals[0] > threshold]
        else:
            # 这是一个常规的 ForObjectDetectionModel
            # 对目标检测模型的原始注释进行后处理
            raw_annotations = self.image_processor.post_process_object_detection(model_outputs, threshold, target_size)
            raw_annotation = raw_annotations[0]
            scores = raw_annotation["scores"]
            labels = raw_annotation["labels"]
            boxes = raw_annotation["boxes"]

            # 将得分、标签和边界框转换为列表形式
            raw_annotation["scores"] = scores.tolist()
            raw_annotation["labels"] = [self.model.config.id2label[label.item()] for label in labels]
            raw_annotation["boxes"] = [self._get_bounding_box(box) for box in boxes]

            # 将原始注释转换为特定格式的注释列表
            keys = ["score", "label", "box"]
            annotation = [
                dict(zip(keys, vals))
                for vals in zip(raw_annotation["scores"], raw_annotation["labels"], raw_annotation["boxes"])
            ]

        # 返回注释列表
        return annotation

    def _get_bounding_box(self, box: "torch.Tensor") -> Dict[str, int]:
        """
        将列表 [xmin, xmax, ymin, ymax] 转换为字典 { "xmin": xmin, ... }

        Args:
            box (`torch.Tensor`): 包含边界框坐标的张量，格式为 [xmin, xmax, ymin, ymax]。

        Returns:
            bbox (`Dict[str, int]`): 包含边界框坐标的字典，格式为 {"xmin": xmin, ...}。
        """
        # 如果框架不是 PyTorch，抛出错误
        if self.framework != "pt":
            raise ValueError("The ObjectDetectionPipeline is only available in PyTorch.")
        # 将边界框转换为字典格式
        xmin, ymin, xmax, ymax = box.int().tolist()
        bbox = {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        }
        return bbox
```