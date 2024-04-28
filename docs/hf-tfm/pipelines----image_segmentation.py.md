# `.\transformers\pipelines\image_segmentation.py`

```
# 导入必要的库
from typing import Any, Dict, List, Union
import numpy as np
from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends
from .base import PIPELINE_INIT_ARGS, Pipeline

# 如果 vision 库可用，导入 PIL 库和 load_image 函数
if is_vision_available():
    from PIL import Image
    from ..image_utils import load_image

# 如果 torch 库可用，导入相关模型
if is_torch_available():
    from ..models.auto.modeling_auto import (
        MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES,
        MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING_NAMES,
        MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,
        MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES,
    )

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义类型别名
Prediction = Dict[str, Any]
Predictions = List[Prediction]

# 添加结束文档注释
@add_end_docstrings(PIPELINE_INIT_ARGS)
# 定义图像分割 Pipeline 类
class ImageSegmentationPipeline(Pipeline):
    """
    Image segmentation pipeline using any `AutoModelForXXXSegmentation`. This pipeline predicts masks of objects and
    their classes.

    Example:

    # 示例代码

    This image segmentation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"image-segmentation"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=image-segmentation).
    """

    # 初始化函数
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 如果框架是 tf，则抛出异常
        if self.framework == "tf":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

        # 确保后端可用
        requires_backends(self, "vision")
        
        # 更新模型类型映射
        mapping = MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES.copy()
        mapping.update(MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES)
        mapping.update(MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING_NAMES)
        mapping.update(MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES)
        
        # 检查模型类型
        self.check_model_type(mapping)
    # 对参数进行清理和预处理，返回预处理、后处理参数的字典
    def _sanitize_parameters(self, **kwargs):
        # 预处理参数字典
        preprocess_kwargs = {}
        # 后处理参数字典
        postprocess_kwargs = {}
        # 如果参数中包含 "subtask"
        if "subtask" in kwargs:
            # 将 "subtask" 添加到后处理参数中
            postprocess_kwargs["subtask"] = kwargs["subtask"]
            # 将 "subtask" 添加到预处理参数中
            preprocess_kwargs["subtask"] = kwargs["subtask"]
        # 如果参数中包含 "threshold"
        if "threshold" in kwargs:
            # 将 "threshold" 添加到后处理参数中
            postprocess_kwargs["threshold"] = kwargs["threshold"]
        # 如果参数中包含 "mask_threshold"
        if "mask_threshold" in kwargs:
            # 将 "mask_threshold" 添加到后处理参数中
            postprocess_kwargs["mask_threshold"] = kwargs["mask_threshold"]
        # 如果参数中包含 "overlap_mask_area_threshold"
        if "overlap_mask_area_threshold" in kwargs:
            # 将 "overlap_mask_area_threshold" 添加到后处理参数中
            postprocess_kwargs["overlap_mask_area_threshold"] = kwargs["overlap_mask_area_threshold"]
        # 如果参数中包含 "timeout"
        if "timeout" in kwargs:
            # 将 "timeout" 添加到预处理参数中
            preprocess_kwargs["timeout"] = kwargs["timeout"]

        # 返回预处理参数字典、空字典和后处理参数字典
        return preprocess_kwargs, {}, postprocess_kwargs
    # 重写父类的 __call__ 方法，用于执行图像分割操作，检测出掩码和类别
    def __call__(self, images, **kwargs) -> Union[Predictions, List[Prediction]]:
        """
        执行图像分割（检测掩码和类别）操作，传入的图像作为输入。

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                管道处理三种类型的图像：

                - 包含指向图像的 HTTP(S) 链接的字符串
                - 包含指向图像的本地路径的字符串
                - 直接加载在 PIL 中的图像

                管道接受单个图像或一批图像。批处理中的图像必须是相同的格式：全部为 HTTP(S) 链接、全部为本地路径或全部为 PIL 图像。
            subtask (`str`, *optional*):
                要执行的分割任务，根据模型的能力选择 [`semantic`, `instance` and `panoptic`]。如果未设置，则管道将尝试按顺序解析：
                  `panoptic`, `instance`, `semantic`.
            threshold (`float`, *optional*, defaults to 0.9):
                用于过滤预测掩码的概率阈值。
            mask_threshold (`float`, *optional*, defaults to 0.5):
                在将预测掩码转换为二进制值时使用的阈值。
            overlap_mask_area_threshold (`float`, *optional*, defaults to 0.5):
                用于消除小的、断开的分段的掩码重叠阈值。
            timeout (`float`, *optional*, defaults to None):
                从网络获取图像的最长等待时间（以秒为单位）。如果为 None，则不设置超时，调用可能会永远阻塞。

        Return:
            包含结果的字典或字典列表。如果输入是单个图像，将返回一个字典列表；如果输入是多个图像，将返回一个字典列表的列表，每个列表对应一个图像。

            字典包含每个检测到对象的掩码、标签和分数（如果适用），包含以下键：

            - **label** (`str`) -- 模型识别的类别标签。
            - **mask** (`PIL.Image`) -- 检测到对象的二进制掩码，作为原始图像的 PIL 图像（宽度、高度）。如果未找到对象，则返回一个填充了零的掩码。
            - **score** (*optional* `float`) -- 可选项，当模型能够估计标签和掩码描述的 "对象" 的置信度时。
        """
        return super().__call__(images, **kwargs)
    # 对输入图像进行预处理，返回模型所需的输入数据
    def preprocess(self, image, subtask=None, timeout=None):
        # 载入图像数据
        image = load_image(image, timeout=timeout)
        # 初始化目标尺寸列表
        target_size = [(image.height, image.width)]
        # 判断模型配置是否为 OneFormerConfig 类
        if self.model.config.__class__.__name__ == "OneFormerConfig":
            # 如果子任务为空，则设置 kwargs 为空字典
            if subtask is None:
                kwargs = {}
            else:
                # 否则，设置 kwargs 包含子任务
                kwargs = {"task_inputs": [subtask]}
            # 对图像进行处理，返回模型所需的输入数据
            inputs = self.image_processor(images=[image], return_tensors="pt", **kwargs)
            # 对任务输入进行分词处理
            inputs["task_inputs"] = self.tokenizer(
                inputs["task_inputs"],
                padding="max_length",
                max_length=self.model.config.task_seq_len,
                return_tensors=self.framework,
            )["input_ids"]
        else:
            # 对图像进行处理，返回模型所需的输入数据
            inputs = self.image_processor(images=[image], return_tensors="pt")
        # 设置输入数据的目标尺寸
        inputs["target_size"] = target_size
        return inputs

    # 处理模型推理过程的方法
    def _forward(self, model_inputs):
        # 弹出目标尺寸信息
        target_size = model_inputs.pop("target_size")
        # 模型推理过程，获取输出结果
        model_outputs = self.model(**model_inputs)
        # 将目标尺寸信息添加到输出结果中
        model_outputs["target_size"] = target_size
        return model_outputs

    # 对模型输出结果进行后处理
    def postprocess(
        self, model_outputs, subtask=None, threshold=0.9, mask_threshold=0.5, overlap_mask_area_threshold=0.5
        ):
            # 初始化文件名为 None
            fn = None
            # 如果子任务为 "panoptic" 或者为 None，并且 self.image_processor 具有方法 "post_process_panoptic_segmentation" 
            if subtask in {"panoptic", None} and hasattr(self.image_processor, "post_process_panoptic_segmentation"):
                # 将方法 "post_process_panoptic_segmentation" 赋值给 fn
                fn = self.image_processor.post_process_panoptic_segmentation
            # 如果子任务为 "instance" 或者为 None，并且 self.image_processor 具有方法 "post_process_instance_segmentation" 
            elif subtask in {"instance", None} and hasattr(self.image_processor, "post_process_instance_segmentation"):
                # 将方法 "post_process_instance_segmentation" 赋值给 fn
                fn = self.image_processor.post_process_instance_segmentation

            # 如果 fn 不为 None
            if fn is not None:
                # 调用 fn 方法，处理模型输出和阈值参数，获取输出结果
                outputs = fn(
                    model_outputs,
                    threshold=threshold,
                    mask_threshold=mask_threshold,
                    overlap_mask_area_threshold=overlap_mask_area_threshold,
                    target_sizes=model_outputs["target_size"],
                )[0]

                # 初始化注释列表
                annotation = []
                # 获取输出结果中的分割
                segmentation = outputs["segmentation"]

                # 遍历输出结果中的各个 segments_info
                for segment in outputs["segments_info"]:
                    # 生成 mask
                    mask = (segmentation == segment["id"]) * 255
                    mask = Image.fromarray(mask.numpy().astype(np.uint8), mode="L")
                    # 获取标签、分数，添加到注释列表中
                    label = self.model.config.id2label[segment["label_id"]]
                    score = segment["score"]
                    annotation.append({"score": score, "label": label, "mask": mask})

            # 如果子任务为 "semantic" 或者为 None，并且 self.image_processor 具有方法 "post_process_semantic_segmentation" 
            elif subtask in {"semantic", None} and hasattr(self.image_processor, "post_process_semantic_segmentation"):
                # 调用 "post_process_semantic_segmentation" 方法，处理模型输出，获取输出结果
                outputs = self.image_processor.post_process_semantic_segmentation(
                    model_outputs, target_sizes=model_outputs["target_size"]
                )[0]

                # 初始化注释列表
                annotation = []
                # 获取输出结果中的分割
                segmentation = outputs.numpy()
                # 获取分割结果中的标签
                labels = np.unique(segmentation)

                # 遍历分割结果中的每个标签
                for label in labels:
                    # 生成 mask
                    mask = (segmentation == label) * 255
                    mask = Image.fromarray(mask.astype(np.uint8), mode="L")
                    # 获取标签，添加到注释列表中
                    label = self.model.config.id2label[label]
                    annotation.append({"score": None, "label": label, "mask": mask})
            # 如果不满足以上条件
            else:
                # 抛出异常，说明不支持该子任务
                raise ValueError(f"Subtask {subtask} is not supported for model {type(self.model)}")
            # 返回注释列表
            return annotation
```