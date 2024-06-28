# `.\pipelines\image_segmentation.py`

```py
# 从 typing 模块导入 Any、Dict、List、Union 类型
from typing import Any, Dict, List, Union

# 导入 numpy 库并用 np 别名引用
import numpy as np

# 从当前目录的 ..utils 模块中导入指定的函数和类
from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends

# 从当前目录的 .base 模块中导入 Pipeline 类和 build_pipeline_init_args 函数
from .base import Pipeline, build_pipeline_init_args

# 如果 torch 可用，则执行以下导入
if is_vision_available():
    # 从 PIL 库中导入 Image 类
    from PIL import Image
    # 从 ..image_utils 模块中导入 load_image 函数
    from ..image_utils import load_image

# 如果 torch 可用，则执行以下导入
if is_torch_available():
    # 从 ..models.auto.modeling_auto 模块中导入以下命名
    from ..models.auto.modeling_auto import (
        MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES,
        MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING_NAMES,
        MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,
        MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES,
    )

# 从 logging 模块中获取 logger 对象
logger = logging.get_logger(__name__)

# 定义类型别名 Prediction 为字典类型
Prediction = Dict[str, Any]
# 定义类型别名 Predictions 为列表，其中每个元素是 Prediction 类型
Predictions = List[Prediction]

# 使用装饰器 add_end_docstrings，为 ImageSegmentationPipeline 类添加文档字符串
@add_end_docstrings(build_pipeline_init_args(has_image_processor=True))
# 继承 Pipeline 类，实现图像分割管道
class ImageSegmentationPipeline(Pipeline):
    """
    Image segmentation pipeline using any `AutoModelForXXXSegmentation`. This pipeline predicts masks of objects and
    their classes.

    Example:

    ```
    >>> from transformers import pipeline

    >>> segmenter = pipeline(model="facebook/detr-resnet-50-panoptic")
    >>> segments = segmenter("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    >>> len(segments)
    2

    >>> segments[0]["label"]
    'bird'

    >>> segments[1]["label"]
    'bird'

    >>> type(segments[0]["mask"])  # This is a black and white mask showing where is the bird on the original image.
    <class 'PIL.Image.Image'>

    >>> segments[0]["mask"].size
    (768, 512)
    ```

    This image segmentation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"image-segmentation"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=image-segmentation).
    """

    # 构造函数初始化，接受任意位置和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用父类 Pipeline 的构造函数
        super().__init__(*args, **kwargs)

        # 如果使用的是 TensorFlow 框架，则抛出 ValueError 异常
        if self.framework == "tf":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

        # 检查是否需要 vision 后端支持
        requires_backends(self, "vision")

        # 复制并更新模型映射字典
        mapping = MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES.copy()
        mapping.update(MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES)
        mapping.update(MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING_NAMES)
        mapping.update(MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES)

        # 检查当前模型的类型
        self.check_model_type(mapping)
    # 定义一个方法用于清理和规范化输入参数
    def _sanitize_parameters(self, **kwargs):
        # 初始化预处理和后处理参数字典
        preprocess_kwargs = {}
        postprocess_kwargs = {}

        # 如果参数中包含"subtask"，则在预处理和后处理参数中设置"subtask"
        if "subtask" in kwargs:
            postprocess_kwargs["subtask"] = kwargs["subtask"]
            preprocess_kwargs["subtask"] = kwargs["subtask"]

        # 如果参数中包含"threshold"，则在后处理参数中设置"threshold"
        if "threshold" in kwargs:
            postprocess_kwargs["threshold"] = kwargs["threshold"]

        # 如果参数中包含"mask_threshold"，则在后处理参数中设置"mask_threshold"
        if "mask_threshold" in kwargs:
            postprocess_kwargs["mask_threshold"] = kwargs["mask_threshold"]

        # 如果参数中包含"overlap_mask_area_threshold"，则在后处理参数中设置"overlap_mask_area_threshold"
        if "overlap_mask_area_threshold" in kwargs:
            postprocess_kwargs["overlap_mask_area_threshold"] = kwargs["overlap_mask_area_threshold"]

        # 如果参数中包含"timeout"，则在预处理参数中设置"timeout"
        if "timeout" in kwargs:
            preprocess_kwargs["timeout"] = kwargs["timeout"]

        # 返回预处理参数、空字典和后处理参数的元组
        return preprocess_kwargs, {}, postprocess_kwargs
    # 定义一个特殊方法 __call__，允许对象被像函数一样调用，用于执行图像分割（检测掩码和类别）。
    def __call__(self, images, **kwargs) -> Union[Predictions, List[Prediction]]:
        """
        执行图像分割（检测掩码和类别）在作为输入的图像中。

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                处理三种类型的图像：

                - 包含 HTTP(S) 链接指向图像的字符串
                - 包含指向本地图像路径的字符串
                - 直接加载的 PIL 图像

                管道接受单个图像或批量图像。批量图像必须统一格式：全部是 HTTP(S) 链接，全部是本地路径，或全部是 PIL 图像。
            subtask (`str`, *optional*):
                要执行的分割任务，根据模型能力选择 [`semantic`, `instance` 和 `panoptic`]。如果未设置，管道将按照以下顺序尝试解析：
                  `panoptic`, `instance`, `semantic`.
            threshold (`float`, *optional*, defaults to 0.9):
                过滤预测掩码的概率阈值。
            mask_threshold (`float`, *optional*, defaults to 0.5):
                在将预测掩码转换为二进制值时使用的阈值。
            overlap_mask_area_threshold (`float`, *optional*, defaults to 0.5):
                掩码重叠阈值，用于消除小的断开分段。
            timeout (`float`, *optional*, defaults to None):
                从网络获取图像的最大等待时间（秒）。如果为 None，则不设置超时，调用可能会永远阻塞。

        Return:
            包含结果的字典或字典列表。如果输入是单个图像，则返回字典列表；如果输入是多个图像，则返回与每个图像对应的字典列表。

            字典包含每个检测到对象的掩码、标签和得分（适用时），包含以下键：

            - **label** (`str`) -- 模型识别的类别标签。
            - **mask** (`PIL.Image`) -- 检测到对象的二进制掩码，作为原始图像的 PIL 图像，形状为 (width, height)。如果未找到对象，则返回填充零的掩码。
            - **score** (*optional* `float`) -- 当模型能够估计标签和掩码描述的 "对象" 的置信度时，可选地包含。
        """
        # 调用父类的 __call__ 方法，执行实际的图像处理操作
        return super().__call__(images, **kwargs)
    # 对图像进行预处理，加载图像并根据需要设置超时时间
    image = load_image(image, timeout=timeout)
    
    # 设置目标尺寸为图像的高度和宽度的元组列表
    target_size = [(image.height, image.width)]
    
    # 如果模型配置是 OneFormerConfig 类型，则根据子任务设置输入参数
    if self.model.config.__class__.__name__ == "OneFormerConfig":
        if subtask is None:
            kwargs = {}  # 如果没有子任务，初始化空的关键字参数字典
        else:
            kwargs = {"task_inputs": [subtask]}  # 如果有子任务，设置任务输入参数
        # 使用图像处理器处理图像，返回 PyTorch 张量作为输入
        inputs = self.image_processor(images=[image], return_tensors="pt", **kwargs)
        
        # 使用分词器处理任务输入，将任务输入转换为输入 ID 的张量，填充到最大长度
        inputs["task_inputs"] = self.tokenizer(
            inputs["task_inputs"],
            padding="max_length",
            max_length=self.model.config.task_seq_len,
            return_tensors=self.framework,
        )["input_ids"]
    else:
        # 如果模型配置不是 OneFormerConfig 类型，则仅使用图像处理器处理图像
        inputs = self.image_processor(images=[image], return_tensors="pt")
    
    # 将目标尺寸添加到输入字典中
    inputs["target_size"] = target_size
    
    # 返回预处理后的输入字典
    return inputs

def _forward(self, model_inputs):
    # 弹出输入字典中的目标尺寸，并保存到 target_size 变量中
    target_size = model_inputs.pop("target_size")
    
    # 使用模型处理输入字典，获取模型的输出
    model_outputs = self.model(**model_inputs)
    
    # 将目标尺寸添加到模型输出中
    model_outputs["target_size"] = target_size
    
    # 返回处理后的模型输出字典
    return model_outputs

def postprocess(
    self, model_outputs, subtask=None, threshold=0.9, mask_threshold=0.5, overlap_mask_area_threshold=0.5
):
    # 此方法用于后处理模型的输出，可以根据需要进行进一步的处理和过滤
        ):
        # 初始化一个空的函数对象
        fn = None
        # 如果子任务是'panoptic'或者None，并且self.image_processor具有'post_process_panoptic_segmentation'属性
        if subtask in {"panoptic", None} and hasattr(self.image_processor, "post_process_panoptic_segmentation"):
            # 将函数指向self.image_processor中的'post_process_panoptic_segmentation'函数
            fn = self.image_processor.post_process_panoptic_segmentation
        # 或者如果子任务是'instance'或者None，并且self.image_processor具有'post_process_instance_segmentation'属性
        elif subtask in {"instance", None} and hasattr(self.image_processor, "post_process_instance_segmentation"):
            # 将函数指向self.image_processor中的'post_process_instance_segmentation'函数
            fn = self.image_processor.post_process_instance_segmentation

        # 如果fn不为None，则执行以下代码块
        if fn is not None:
            # 调用fn函数，处理模型输出，根据指定阈值进行后处理
            outputs = fn(
                model_outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                overlap_mask_area_threshold=overlap_mask_area_threshold,
                target_sizes=model_outputs["target_size"],
            )[0]

            # 初始化一个空的注释列表
            annotation = []
            # 获取输出中的分割结果
            segmentation = outputs["segmentation"]

            # 遍历每个分割信息
            for segment in outputs["segments_info"]:
                # 根据分割ID生成对应的掩码
                mask = (segmentation == segment["id"]) * 255
                mask = Image.fromarray(mask.numpy().astype(np.uint8), mode="L")
                # 获取分割标签和得分
                label = self.model.config.id2label[segment["label_id"]]
                score = segment["score"]
                # 将标签、得分和掩码添加到注释列表中
                annotation.append({"score": score, "label": label, "mask": mask})

        # 如果fn为None，并且子任务是'semantic'或者None，并且self.image_processor具有'post_process_semantic_segmentation'属性
        elif subtask in {"semantic", None} and hasattr(self.image_processor, "post_process_semantic_segmentation"):
            # 调用self.image_processor中的'post_process_semantic_segmentation'函数，处理语义分割的模型输出
            outputs = self.image_processor.post_process_semantic_segmentation(
                model_outputs, target_sizes=model_outputs["target_size"]
            )[0]

            # 初始化一个空的注释列表
            annotation = []
            # 将输出转换为numpy数组形式的分割结果
            segmentation = outputs.numpy()
            # 获取分割结果中的所有标签
            labels = np.unique(segmentation)

            # 遍历每个标签
            for label in labels:
                # 根据标签生成对应的掩码
                mask = (segmentation == label) * 255
                mask = Image.fromarray(mask.astype(np.uint8), mode="L")
                # 获取标签名称
                label = self.model.config.id2label[label]
                # 将标签和掩码添加到注释列表中，得分设为None
                annotation.append({"score": None, "label": label, "mask": mask})
        else:
            # 如果不满足任何处理条件，则抛出异常
            raise ValueError(f"Subtask {subtask} is not supported for model {type(self.model)}")
        # 返回最终的注释列表
        return annotation
```