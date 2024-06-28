# `.\pipelines\mask_generation.py`

```py
from collections import defaultdict  # 导入 defaultdict 类，用于创建默认值为列表的字典
from typing import Optional  # 导入 Optional 类型，表示某些参数可选

from ..image_utils import load_image  # 导入 load_image 函数，用于加载图像
from ..utils import (  # 导入多个工具函数和类
    add_end_docstrings,  # 添加文档结尾的装饰器
    is_torch_available,  # 检查是否可用 Torch 库
    logging,  # 记录日志相关功能
    requires_backends,  # 检查所需后端
)
from .base import ChunkPipeline, build_pipeline_init_args  # 导入 ChunkPipeline 类和初始化参数构建函数

if is_torch_available():  # 如果 Torch 库可用
    import torch  # 导入 Torch 库

    from ..models.auto.modeling_auto import MODEL_FOR_MASK_GENERATION_MAPPING_NAMES  # 导入自动掩模生成模型映射名称

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


@add_end_docstrings(  # 使用装饰器添加文档结尾
    build_pipeline_init_args(has_image_processor=True),  # 使用构建管道初始化参数函数，指定有图像处理器
    r"""
        points_per_batch (*optional*, int, default to 64):
            设置模型同时运行的点数。数字越高可能速度更快但使用更多 GPU 内存。
        output_bboxes_mask (`bool`, *optional*, default to `False`):
            是否输出边界框预测。
        output_rle_masks (`bool`, *optional*, default to `False`):
            是否以 RLE 格式输出掩码。""",
)
class MaskGenerationPipeline(ChunkPipeline):  # 定义掩模生成管道类，继承自 ChunkPipeline
    """
    自动为图像生成掩模，使用 `SamForMaskGeneration` 模型。该管道预测给定图像的二进制掩模。它是一个 `ChunkPipeline`，
    因为可以将小批量中的点分开处理，以避免内存不足问题。使用 `points_per_batch` 参数控制同时处理的点数，默认为 `64`。

    该管道工作分为三个步骤：
        1. `preprocess`: 生成一个均匀分布的 1024 个点网格，以及边界框和点标签。
            更多关于如何创建点和边界框的细节，请查看 `_generate_crop_boxes` 函数。同时使用 `image_processor` 预处理图像。
            该函数生成一个 `points_per_batch` 的小批量。

        2. `forward`: 将 `preprocess` 的输出馈送到模型。仅计算图像嵌入一次。
            调用 `self.model.get_image_embeddings`，确保不计算梯度，并且张量和模型在同一设备上。

        3. `postprocess`: 自动掩模生成的最重要部分发生在这里。包括三个步骤：
            - image_processor.postprocess_masks（在每个小批量循环中运行）：处理原始输出掩模，根据图像大小调整它们的大小，并将其转换为二进制掩模。
            - image_processor.filter_masks（在每个小批量循环中运行）：使用 `pred_iou_thresh` 和 `stability_scores`，以及基于非最大抑制的各种过滤器，去除不良掩模。
            - image_processor.postprocess_masks_for_amg：对掩模应用 NSM，仅保留相关掩模。

    示例：

    ```
    >>> from transformers import pipeline
    ```
    >>> generator = pipeline(model="facebook/sam-vit-base", task="mask-generation")
    # 使用预定义的pipeline函数创建一个生成器，用于执行模型推断任务，指定模型和任务类型为“mask-generation”。
    
    >>> outputs = generator(
    ...     "http://images.cocodataset.org/val2017/000000039769.jpg",
    ... )
    # 使用生成器执行推断任务，输入为指定的图像URL。该步骤将返回推断结果。
    
    >>> outputs = generator(
    ...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png", points_per_batch=128
    ... )
    # 再次使用生成器执行推断任务，输入为另一个图像URL，并设置额外的参数points_per_batch为128。该步骤将返回推断结果。
    
    """
    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)
    
    This segmentation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"mask-generation"`.
    
    See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=mask-generation).
    """
    # 提供了有关使用管道的基础信息，并指出此分割管道可使用task标识符“mask-generation”从[`pipeline`]加载。
    
    class YourClassName:
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            requires_backends(self, "vision")
            requires_backends(self, "torch")
    
            if self.framework != "pt":
                raise ValueError(f"The {self.__class__} is only available in PyTorch.")
    
            self.check_model_type(MODEL_FOR_MASK_GENERATION_MAPPING_NAMES)
        # 初始化类的构造函数，进行基本的设置和检查，确保所需的后端库和框架为PyTorch。
    
        def _sanitize_parameters(self, **kwargs):
            preprocess_kwargs = {}
            postprocess_kwargs = {}
            forward_params = {}
    
            # 预处理参数
            if "points_per_batch" in kwargs:
                preprocess_kwargs["points_per_batch"] = kwargs["points_per_batch"]
            if "points_per_crop" in kwargs:
                preprocess_kwargs["points_per_crop"] = kwargs["points_per_crop"]
            if "crops_n_layers" in kwargs:
                preprocess_kwargs["crops_n_layers"] = kwargs["crops_n_layers"]
            if "crop_overlap_ratio" in kwargs:
                preprocess_kwargs["crop_overlap_ratio"] = kwargs["crop_overlap_ratio"]
            if "crop_n_points_downscale_factor" in kwargs:
                preprocess_kwargs["crop_n_points_downscale_factor"] = kwargs["crop_n_points_downscale_factor"]
            if "timeout" in kwargs:
                preprocess_kwargs["timeout"] = kwargs["timeout"]
    
            # 后处理参数
            if "pred_iou_thresh" in kwargs:
                forward_params["pred_iou_thresh"] = kwargs["pred_iou_thresh"]
            if "stability_score_offset" in kwargs:
                forward_params["stability_score_offset"] = kwargs["stability_score_offset"]
            if "mask_threshold" in kwargs:
                forward_params["mask_threshold"] = kwargs["mask_threshold"]
            if "stability_score_thresh" in kwargs:
                forward_params["stability_score_thresh"] = kwargs["stability_score_thresh"]
            if "crops_nms_thresh" in kwargs:
                postprocess_kwargs["crops_nms_thresh"] = kwargs["crops_nms_thresh"]
            if "output_rle_mask" in kwargs:
                postprocess_kwargs["output_rle_mask"] = kwargs["output_rle_mask"]
            if "output_bboxes_mask" in kwargs:
                postprocess_kwargs["output_bboxes_mask"] = kwargs["output_bboxes_mask"]
    
            return preprocess_kwargs, forward_params, postprocess_kwargs
        # 对传入的参数进行清理和预处理，将预处理、前向和后处理的参数分别整理到三个字典中。
    def __call__(self, image, *args, num_workers=None, batch_size=None, **kwargs):
        """
        通过调用实例对象，生成二进制分割掩码

        Args:
            inputs (`np.ndarray` or `bytes` or `str` or `dict`):
                图像或图像列表。
            mask_threshold (`float`, *optional*, defaults to 0.0):
                将预测的掩码转换为二进制值时使用的阈值。
            pred_iou_thresh (`float`, *optional*, defaults to 0.88):
                应用于模型预测掩码质量的过滤阈值，取值范围为 `[0,1]`。
            stability_score_thresh (`float`, *optional*, defaults to 0.95):
                应用于模型掩码预测稳定性的过滤阈值，取值范围为 `[0,1]`。
            stability_score_offset (`int`, *optional*, defaults to 1):
                在计算稳定性分数时，用于偏移截断的量。
            crops_nms_thresh (`float`, *optional*, defaults to 0.7):
                由非极大值抑制使用的框 IoU 截断，用于过滤重复的掩码。
            crops_n_layers (`int`, *optional*, defaults to 0):
                如果 `crops_n_layers>0`，则将再次对图像的裁剪运行掩码预测。设置运行的层数，每层有 2**i_layer 个图像裁剪。
            crop_overlap_ratio (`float`, *optional*, defaults to `512 / 1500`):
                设置裁剪重叠的程度。在第一层裁剪中，裁剪将以图像长度的这一分数重叠。随后的层级通过更多的裁剪减少此重叠。
            crop_n_points_downscale_factor (`int`, *optional*, defaults to `1`):
                在第 n 层采样的每边点数按 crop_n_points_downscale_factor**n 缩小。
            timeout (`float`, *optional*, defaults to None):
                从网页获取图像的最大等待时间（秒）。如果为 None，则不设置超时，调用可能会一直阻塞。

        Return:
            `Dict`: 包含以下键的字典：
                - **mask** (`PIL.Image`) -- 检测到对象的二进制掩码，作为原始图像 `(width, height)` 的 PIL 图像。如果未检测到对象，则返回一个填充零的掩码。
                - **score** (*optional* `float`) -- 可选，当模型能够估计标签和掩码描述的 "对象" 的置信度时。

        """
        return super().__call__(image, *args, num_workers=num_workers, batch_size=batch_size, **kwargs)
    def preprocess(
        self,
        image,
        points_per_batch=64,  # 每批处理的点数，默认为64
        crops_n_layers: int = 0,  # 裁剪层数，默认为0
        crop_overlap_ratio: float = 512 / 1500,  # 裁剪重叠比例，默认为512/1500
        points_per_crop: Optional[int] = 32,  # 每个裁剪的点数，默认为32
        crop_n_points_downscale_factor: Optional[int] = 1,  # 裁剪点数缩放因子，默认为1
        timeout: Optional[float] = None,  # 超时时间，默认为None
    ):
        image = load_image(image, timeout=timeout)  # 调用load_image函数加载图像，可以设置超时时间
        target_size = self.image_processor.size["longest_edge"]  # 获取图像处理器中最长边的尺寸作为目标尺寸
        crop_boxes, grid_points, cropped_images, input_labels = self.image_processor.generate_crop_boxes(
            image, target_size, crops_n_layers, crop_overlap_ratio, points_per_crop, crop_n_points_downscale_factor
        )  # 使用图像处理器生成裁剪框、网格点、裁剪后的图像和输入标签

        model_inputs = self.image_processor(images=cropped_images, return_tensors="pt")  # 使用图像处理器处理裁剪后的图像，返回PyTorch张量格式的模型输入

        with self.device_placement():  # 使用设备分配上下文管理器
            if self.framework == "pt":  # 如果框架是PyTorch
                inference_context = self.get_inference_context()  # 获取推断上下文
                with inference_context():  # 使用推断上下文管理器
                    model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)  # 确保模型输入张量位于指定设备上
                    image_embeddings = self.model.get_image_embeddings(model_inputs.pop("pixel_values"))  # 获取图像嵌入向量
                    model_inputs["image_embeddings"] = image_embeddings  # 将图像嵌入向量添加到模型输入中

        n_points = grid_points.shape[1]  # 获取网格点的数量
        points_per_batch = points_per_batch if points_per_batch is not None else n_points  # 如果指定了每批处理的点数则使用，否则使用网格点的数量

        if points_per_batch <= 0:  # 如果每批处理的点数小于等于0
            raise ValueError(
                "Cannot have points_per_batch<=0. Must be >=1 to returned batched outputs. "
                "To return all points at once, set points_per_batch to None"
            )  # 抛出数值错误异常，要求每批处理的点数必须大于等于1，或者设置为None以一次返回所有点

        for i in range(0, n_points, points_per_batch):  # 遍历网格点，每次处理points_per_batch个点
            batched_points = grid_points[:, i : i + points_per_batch, :, :]  # 分批次获取网格点
            labels = input_labels[:, i : i + points_per_batch]  # 获取对应的输入标签
            is_last = i == n_points - points_per_batch  # 判断是否是最后一批

            yield {
                "input_points": batched_points,  # 返回批次的输入点
                "input_labels": labels,  # 返回对应的输入标签
                "input_boxes": crop_boxes,  # 返回裁剪框
                "is_last": is_last,  # 返回是否是最后一批
                **model_inputs,  # 返回模型输入的其它内容
            }

    def _forward(
        self,
        model_inputs,
        pred_iou_thresh=0.88,  # 预测IOU阈值，默认为0.88
        stability_score_thresh=0.95,  # 稳定性分数阈值，默认为0.95
        mask_threshold=0,  # 掩码阈值，默认为0
        stability_score_offset=1,  # 稳定性分数偏移量，默认为1
    ):
        # 从模型输入中弹出"input_boxes"，并保存在input_boxes变量中
        input_boxes = model_inputs.pop("input_boxes")
        # 从模型输入中弹出"is_last"，并保存在is_last变量中
        is_last = model_inputs.pop("is_last")
        # 从模型输入中弹出"original_sizes"，并将其转换为列表保存在original_sizes变量中
        original_sizes = model_inputs.pop("original_sizes").tolist()
        # 从模型输入中弹出"reshaped_input_sizes"，并将其转换为列表保存在reshaped_input_sizes变量中
        reshaped_input_sizes = model_inputs.pop("reshaped_input_sizes").tolist()

        # 使用模型进行推理，将模型输入传递给模型并获取模型输出
        model_outputs = self.model(**model_inputs)

        # 在这里进行后处理，以避免复制所有掩码的CPU GPU
        # 从模型输出中获取"pred_masks"，即低分辨率掩码
        low_resolution_masks = model_outputs["pred_masks"]
        # 调用图像处理器的方法对掩码进行后处理，得到更高分辨率的掩码
        masks = self.image_processor.post_process_masks(
            low_resolution_masks, original_sizes, reshaped_input_sizes, mask_threshold, binarize=False
        )
        # 从模型输出中获取"iou_scores"，即IoU分数
        iou_scores = model_outputs["iou_scores"]
        # 使用图像处理器的方法对掩码进行筛选，得到最终的掩码、IoU分数和边界框
        masks, iou_scores, boxes = self.image_processor.filter_masks(
            masks[0],
            iou_scores[0],
            original_sizes[0],
            input_boxes[0],
            pred_iou_thresh,
            stability_score_thresh,
            mask_threshold,
            stability_score_offset,
        )
        # 返回处理后的结果，包括掩码、is_last标志、边界框和IoU分数
        return {
            "masks": masks,
            "is_last": is_last,
            "boxes": boxes,
            "iou_scores": iou_scores,
        }

    # 定义后处理方法，用于整合多个模型输出并生成最终的掩码和分数
    def postprocess(
        self,
        model_outputs,
        output_rle_mask=False,
        output_bboxes_mask=False,
        crops_nms_thresh=0.7,
    ):
        # 存储所有模型输出的IoU分数、掩码和边界框
        all_scores = []
        all_masks = []
        all_boxes = []
        for model_output in model_outputs:
            # 弹出模型输出中的"IoU_scores"并添加到all_scores列表中
            all_scores.append(model_output.pop("iou_scores"))
            # 扩展模型输出中的"masks"并添加到all_masks列表中
            all_masks.extend(model_output.pop("masks"))
            # 弹出模型输出中的"boxes"并添加到all_boxes列表中
            all_boxes.append(model_output.pop("boxes"))

        # 使用PyTorch的方法连接所有IoU分数和边界框
        all_scores = torch.cat(all_scores)
        all_boxes = torch.cat(all_boxes)
        # 调用图像处理器的方法进行掩码生成的后处理，得到输出掩码、IoU分数、RLE掩码和边界框
        output_masks, iou_scores, rle_mask, bounding_boxes = self.image_processor.post_process_for_mask_generation(
            all_masks, all_scores, all_boxes, crops_nms_thresh
        )

        # 创建默认字典，用于存储额外的输出结果
        extra = defaultdict(list)
        for output in model_outputs:
            for k, v in output.items():
                extra[k].append(v)

        # 创建可选项字典，根据需要添加RLE掩码或边界框
        optional = {}
        if output_rle_mask:
            optional["rle_mask"] = rle_mask

        if output_bboxes_mask:
            optional["bounding_boxes"] = bounding_boxes

        # 返回最终处理结果，包括输出掩码、IoU分数以及额外的输出结果
        return {"masks": output_masks, "scores": iou_scores, **optional, **extra}
```