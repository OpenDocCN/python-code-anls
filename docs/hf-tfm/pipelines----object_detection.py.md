# `.\pipelines\object_detection.py`

```py
# 从 typing 模块中导入 Any, Dict, List, Union 类型
from typing import Any, Dict, List, Union

# 从 ..utils 模块中导入必要的函数和类
from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends
# 从 .base 模块中导入 Pipeline 类和 build_pipeline_init_args 函数
from .base import Pipeline, build_pipeline_init_args

# 如果 vision 可用，从 ..image_utils 模块中导入 load_image 函数
if is_vision_available():
    from ..image_utils import load_image

# 如果 torch 可用，导入 torch 模块和必要的类
if is_torch_available():
    import torch
    from ..models.auto.modeling_auto import (
        MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
    )

# 从 logging 模块中获取当前模块的 logger
logger = logging.get_logger(__name__)

# 定义用于预测结果的类型别名
Prediction = Dict[str, Any]
Predictions = List[Prediction]

# 使用装饰器为 ObjectDetectionPipeline 类添加文档字符串
@add_end_docstrings(build_pipeline_init_args(has_image_processor=True))
class ObjectDetectionPipeline(Pipeline):
    """
    Object detection pipeline using any `AutoModelForObjectDetection`. This pipeline predicts bounding boxes of objects
    and their classes.

    Example:

    ```
    >>> from transformers import pipeline

    >>> detector = pipeline(model="facebook/detr-resnet-50")
    >>> detector("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    [{'score': 0.997, 'label': 'bird', 'box': {'xmin': 69, 'ymin': 171, 'xmax': 396, 'ymax': 507}}, {'score': 0.999, 'label': 'bird', 'box': {'xmin': 398, 'ymin': 105, 'xmax': 767, 'ymax': 507}}]

    >>> # x, y  are expressed relative to the top left hand corner.
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This object detection pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"object-detection"`.

    See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=object-detection).
    """

    # 初始化方法，继承自 Pipeline 类的初始化方法
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 如果使用的框架是 "tf"，抛出 ValueError 异常
        if self.framework == "tf":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

        # 确保依赖的后端库已加载，这里要求加载 "vision"
        requires_backends(self, "vision")

        # 复制对象检测模型映射，并更新为包含对象分类映射的名称
        mapping = MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES.copy()
        mapping.update(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES)
        
        # 检查模型类型是否符合预期
        self.check_model_type(mapping)

    # 私有方法，用于处理和清理参数
    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        # 如果参数中包含 "timeout"，则将其添加到预处理参数中
        if "timeout" in kwargs:
            preprocess_params["timeout"] = kwargs["timeout"]
        postprocess_kwargs = {}
        # 如果参数中包含 "threshold"，则将其添加到后处理参数中
        if "threshold" in kwargs:
            postprocess_kwargs["threshold"] = kwargs["threshold"]
        return preprocess_params, {}, postprocess_kwargs
    # 调用对象实例时执行的方法，用于检测输入图像中的对象（边界框和类别）

    def __call__(self, *args, **kwargs) -> Union[Predictions, List[Prediction]]:
        """
        Detect objects (bounding boxes & classes) in the image(s) passed as inputs.

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing an HTTP(S) link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images. Images in a batch must all be in the
                same format: all as HTTP(S) links, all as local paths, or all as PIL images.
            threshold (`float`, *optional*, defaults to 0.9):
                The probability necessary to make a prediction.
            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
                the call may block forever.

        Return:
            A list of dictionaries or a list of list of dictionaries containing the result. If the input is a single
            image, will return a list of dictionaries, if the input is a list of several images, will return a list of
            list of dictionaries corresponding to each image.

            The dictionaries contain the following keys:

            - **label** (`str`) -- The class label identified by the model.
            - **score** (`float`) -- The score attributed by the model for that label.
            - **box** (`List[Dict[str, int]]`) -- The bounding box of detected object in image's original size.
        """

        return super().__call__(*args, **kwargs)

    # 对输入图像进行预处理，返回模型所需的输入格式
    def preprocess(self, image, timeout=None):
        # 载入图像，根据需要设置超时时间
        image = load_image(image, timeout=timeout)
        # 获取图像的高度和宽度并组成张量
        target_size = torch.IntTensor([[image.height, image.width]])
        # 使用图像处理器处理图像，返回PyTorch格式的输入
        inputs = self.image_processor(images=[image], return_tensors="pt")
        # 如果存在分词器，则使用分词器对文本和边界框进行处理，并返回PyTorch格式的输入
        if self.tokenizer is not None:
            inputs = self.tokenizer(text=inputs["words"], boxes=inputs["boxes"], return_tensors="pt")
        # 将图像的目标尺寸添加到输入中
        inputs["target_size"] = target_size
        return inputs

    # 模型的内部前向传播方法，处理模型输入并返回模型输出
    def _forward(self, model_inputs):
        # 弹出目标尺寸以避免传递给模型
        target_size = model_inputs.pop("target_size")
        # 使用模型进行前向传播，获取输出
        outputs = self.model(**model_inputs)
        # 构建模型的输出对象，并将目标尺寸添加到输出中
        model_outputs = outputs.__class__({"target_size": target_size, **outputs})
        # 如果存在分词器，则将边界框信息添加到模型输出中
        if self.tokenizer is not None:
            model_outputs["bbox"] = model_inputs["bbox"]
        return model_outputs
    def postprocess(self, model_outputs, threshold=0.9):
        # 获取模型输出中的目标尺寸
        target_size = model_outputs["target_size"]
        if self.tokenizer is not None:
            # 这是 LayoutLMForTokenClassification 的变种。
            # OCR 获取了文本框，模型对单词进行了分类。
            # 从目标尺寸中获取高度和宽度
            height, width = target_size[0].tolist()

            def unnormalize(bbox):
                # 将归一化的边界框坐标转换为原始坐标
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

            # 计算模型输出中的得分和类别
            scores, classes = model_outputs["logits"].squeeze(0).softmax(dim=-1).max(dim=-1)
            # 根据预测的类别获取类别标签
            labels = [self.model.config.id2label[prediction] for prediction in classes.tolist()]
            # 将模型输出的边界框进行反归一化处理
            boxes = [unnormalize(bbox) for bbox in model_outputs["bbox"].squeeze(0)]
            keys = ["score", "label", "box"]
            # 创建注释列表，包含得分、标签和边界框
            annotation = [dict(zip(keys, vals)) for vals in zip(scores.tolist(), labels, boxes) if vals[0] > threshold]
        else:
            # 这是一个常规的 ForObjectDetectionModel
            # 对象检测后处理，获取原始注释信息
            raw_annotations = self.image_processor.post_process_object_detection(model_outputs, threshold, target_size)
            raw_annotation = raw_annotations[0]
            # 获取原始注释中的分数、标签和边界框
            scores = raw_annotation["scores"]
            labels = raw_annotation["labels"]
            boxes = raw_annotation["boxes"]

            # 将分数、标签和边界框转换为列表形式
            raw_annotation["scores"] = scores.tolist()
            raw_annotation["labels"] = [self.model.config.id2label[label.item()] for label in labels]
            raw_annotation["boxes"] = [self._get_bounding_box(box) for box in boxes]

            # 构建注释列表，包含得分、标签和边界框
            keys = ["score", "label", "box"]
            annotation = [
                dict(zip(keys, vals))
                for vals in zip(raw_annotation["scores"], raw_annotation["labels"], raw_annotation["boxes"])
            ]

        return annotation

    def _get_bounding_box(self, box: "torch.Tensor") -> Dict[str, int]:
        """
        将列表 [xmin, xmax, ymin, ymax] 转换为字典 { "xmin": xmin, ... }

        Args:
            box (`torch.Tensor`): 包含角落格式坐标的张量。

        Returns:
            bbox (`Dict[str, int]`): 包含角落格式坐标的字典。
        """
        if self.framework != "pt":
            # 如果框架不是 PyTorch，则抛出数值错误
            raise ValueError("The ObjectDetectionPipeline is only available in PyTorch.")
        # 将边界框张量转换为整数列表，并命名为边界框
        xmin, ymin, xmax, ymax = box.int().tolist()
        bbox = {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        }
        return bbox
```