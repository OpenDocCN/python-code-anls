# `.\transformers\pipelines\mask_generation.py`

```py
# 从 collections 模块中导入 defaultdict 类
from collections import defaultdict
# 从 typing 模块中导入 Optional 类型提示
from typing import Optional

# 从上一级目录中的 image_utils 模块导入 load_image 函数
from ..image_utils import load_image
# 从上一级目录中的 utils 模块导入 add_end_docstrings, is_torch_available, logging, requires_backends 函数
from ..utils import (
    add_end_docstrings,
    is_torch_available,
    logging,
    requires_backends,
)
# 从 .base 模块中导入 PIPELINE_INIT_ARGS, ChunkPipeline 类
from .base import PIPELINE_INIT_ARGS, ChunkPipeline

# 检查是否安装了 PyTorch
if is_torch_available():
    # 导入 torch 模块
    import torch

    # 从 ..models.auto.modeling_auto 模块中导入 MODEL_FOR_MASK_GENERATION_MAPPING_NAMES 常量

# 获取 logger 实例
logger = logging.get_logger(__name__)

# 使用装饰器添加文档字符串描述，继承 ChunkPipeline
@add_end_docstrings(PIPELINE_INIT_ARGS)
class MaskGenerationPipeline(ChunkPipeline):
    """
    Automatic mask generation for images using `SamForMaskGeneration`. This pipeline predicts binary masks for an
    image, given an image. It is a `ChunkPipeline` because you can seperate the points in a mini-batch in order to
    avoid OOM issues. Use the `points_per_batch` argument to control the number of points that will be processed at the
    same time. Default is `64`.

    The pipeline works in 3 steps:
        1. `preprocess`: A grid of 1024 points evenly separated is generated along with bounding boxes and point
           labels.
            For more details on how the points and bounding boxes are created, check the `_generate_crop_boxes`
            function. The image is also preprocessed using the `image_processor`. This function `yields` a minibatch of
            `points_per_batch`.

        2. `forward`: feeds the outputs of `preprocess` to the model. The image embedding is computed only once.
            Calls both `self.model.get_image_embeddings` and makes sure that the gradients are not computed, and the
            tensors and models are on the same device.

        3. `postprocess`: The most important part of the automatic mask generation happens here. Three steps
            are induced:
                - image_processor.postprocess_masks (run on each minibatch loop): takes in the raw output masks,
                  resizes them according
                to the image size, and transforms there to binary masks.
                - image_processor.filter_masks (on each minibatch loop): uses both `pred_iou_thresh` and
                  `stability_scores`. Also
                applies a variety of filters based on non maximum suppression to remove bad masks.
                - image_processor.postprocess_masks_for_amg applies the NSM on the mask to only keep relevant ones.
    ```
    # 该函数是一个 Segmentation 管道类的初始化函数
    def __init__(self, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 检查是否支持 vision 和 torch 后端
        requires_backends(self, "vision")
        requires_backends(self, "torch")
        # 如果不是 PyTorch 框架，则抛出错误
        if self.framework != "pt":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")
        # 检查模型类型是否符合 mask-generation 任务
        self.check_model_type(MODEL_FOR_MASK_GENERATION_MAPPING_NAMES)
    # 对传入的参数进行过滤和预处理，分成预处理参数、后处理参数和前向传递参数
    def _sanitize_parameters(self, **kwargs):
        # 定义预处理参数字典和后处理参数字典
        preprocess_kwargs = {}
        postprocess_kwargs = {}
        forward_params = {}
        # 预处理参数
        # 如果传入参数中有"points_per_batch"，则将其存入preprocess_kwargs中
        if "points_per_batch" in kwargs:
            preprocess_kwargs["points_per_batch"] = kwargs["points_per_batch"]
        # 如果传入参数中有"points_per_crop"，则将其存入preprocess_kwargs中
        if "points_per_crop" in kwargs:
            preprocess_kwargs["points_per_crop"] = kwargs["points_per_crop"]
        # 如果传入参数中有"crops_n_layers"，则将其存入preprocess_kwargs中
        if "crops_n_layers" in kwargs:
            preprocess_kwargs["crops_n_layers"] = kwargs["crops_n_layers"]
        # 如果传入参数中有"crop_overlap_ratio"，则将其存入preprocess_kwargs中
        if "crop_overlap_ratio" in kwargs:
            preprocess_kwargs["crop_overlap_ratio"] = kwargs["crop_overlap_ratio"]
        # 如果传入参数中有"crop_n_points_downscale_factor"，则将其存入preprocess_kwargs中
        if "crop_n_points_downscale_factor" in kwargs:
            preprocess_kwargs["crop_n_points_downscale_factor"] = kwargs["crop_n_points_downscale_factor"]
        # 如果传入参数中有"timeout"，则将其存入preprocess_kwargs中
        if "timeout" in kwargs:
            preprocess_kwargs["timeout"] = kwargs["timeout"]
        # 后处理参数
        # 如果传入参数中有"pred_iou_thresh"，则将其存入forward_params中
        if "pred_iou_thresh" in kwargs:
            forward_params["pred_iou_thresh"] = kwargs["pred_iou_thresh"]
        # 如果传入参数中有"stability_score_offset"，则将其存入forward_params中
        if "stability_score_offset" in kwargs:
            forward_params["stability_score_offset"] = kwargs["stability_score_offset"]
        # 如果传入参数中有"mask_threshold"，则将其存入forward_params中
        if "mask_threshold" in kwargs:
            forward_params["mask_threshold"] = kwargs["mask_threshold"]
        # 如果传入参数中有"stability_score_thresh"，则将其存入forward_params中
        if "stability_score_thresh" in kwargs:
            forward_params["stability_score_thresh"] = kwargs["stability_score_thresh"]
        # 如果传入参数中有"crops_nms_thresh"，则将其存入postprocess_kwargs中
        if "crops_nms_thresh" in kwargs:
            postprocess_kwargs["crops_nms_thresh"] = kwargs["crops_nms_thresh"]
        # 如果传入参数中有"output_rle_mask"，则将其存入postprocess_kwargs中
        if "output_rle_mask" in kwargs:
            postprocess_kwargs["output_rle_mask"] = kwargs["output_rle_mask"]
        # 如果传入参数中有"output_bboxes_mask"，则将其存入postprocess_kwargs中
        if "output_bboxes_mask" in kwargs:
            postprocess_kwargs["output_bboxes_mask"] = kwargs["output_bboxes_mask"]
        # 返回处理后的预处理参数、前向传递参数和后处理参数
        return preprocess_kwargs, forward_params, postprocess_kwargs
    # 定义 __call__ 方法，用于生成二进制分割掩码
    def __call__(self, image, *args, num_workers=None, batch_size=None, **kwargs):
        """
        生成二进制分割掩码

        Args:
            inputs (`np.ndarray` or `bytes` or `str` or `dict`):
                图像或图像列表。
            mask_threshold (`float`, *optional*, defaults to 0.0):
                将预测的掩码转换为二进制值时使用的阈值。
            pred_iou_thresh (`float`, *optional*, defaults to 0.88):
                应用于模型预测的掩码质量的过滤阈值，范围为 `[0,1]`。
            stability_score_thresh (`float`, *optional*, defaults to 0.95):
                在模型的掩码预测二值化截断值变化下，使用掩码稳定性的过滤阈值，范围为 `[0,1]`。
            stability_score_offset (`int`, *optional*, defaults to 1):
                计算稳定性分数时截断值的偏移量。
            crops_nms_thresh (`float`, *optional*, defaults to 0.7):
                用于非最大抑制的盒子 IoU 截断，用于过滤重复的掩码。
            crops_n_layers (`int`, *optional*, defaults to 0):
                如果 `crops_n_layers>0`，则将再次在图像的剪裁上运行掩码预测。 设置要运行的层数，其中每个层都有 2**i_layer 个图像剪裁。
            crop_overlap_ratio (`float`, *optional*, defaults to `512 / 1500`):
                设置剪裁之间的重叠程度。在第一层剪裁中，剪裁将以图像长度的该分数重叠。具有更多剪裁的后续层会缩小此重叠。
            crop_n_points_downscale_factor (`int`, *optional*, defaults to `1`):
                在第 n 层采样的每边点的数量按 crop_n_points_downscale_factor**n 缩小。
            timeout (`float`, *optional*, defaults to None):
                从网络获取图像的最长等待时间（以秒为单位）。如果为 None，则不设置超时，并且调用可能会永远阻塞。

        Return:
            `Dict`: 一个包含以下键的字典：
                - **mask** (`PIL.Image`) -- 作为原始图像的 PIL 图像 `(width, height)` 形状的检测到对象的二进制掩码。如果未找到对象，则返回填充了零的掩码。
                - **score** (*optional* `float`) -- 可选地，当模型能够估计标签和掩码描述的 "对象" 的置信度时。
        """
        # 调用父类的 __call__ 方法，并传递参数
        return super().__call__(image, *args, num_workers=num_workers, batch_size=batch_size, **kwargs)
    # 预处理图像并生成用于模型推理的数据
    def preprocess(
        self,
        image,
        points_per_batch=64,
        crops_n_layers: int = 0,
        crop_overlap_ratio: float = 512 / 1500,
        points_per_crop: Optional[int] = 32,
        crop_n_points_downscale_factor: Optional[int] = 1,
        timeout: Optional[float] = None,
    ):
        # 加载输入图像
        image = load_image(image, timeout=timeout)
        # 获取目标图像尺寸
        target_size = self.image_processor.size["longest_edge"]
        # 生成裁剪框、网格点、裁剪后图像和输入标签
        crop_boxes, grid_points, cropped_images, input_labels = self.image_processor.generate_crop_boxes(
            image, target_size, crops_n_layers, crop_overlap_ratio, points_per_crop, crop_n_points_downscale_factor
        )
        # 处理输入图像并返回张量格式
        model_inputs = self.image_processor(images=cropped_images, return_tensors="pt")
        
        # 在设备上执行推理
        with self.device_placement():
            if self.framework == "pt":
                inference_context = self.get_inference_context()
                with inference_context():
                    model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
                    image_embeddings = self.model.get_image_embeddings(model_inputs.pop("pixel_values"))
                    model_inputs["image_embeddings"] = image_embeddings
    
        # 计算网格点的数量
        n_points = grid_points.shape[1]
        # 如果 points_per_batch 为 None，则返回所有点
        points_per_batch = points_per_batch if points_per_batch is not None else n_points
    
        # 如果 points_per_batch 小于等于 0，则抛出错误
        if points_per_batch <= 0:
            raise ValueError(
                "Cannot have points_per_batch<=0. Must be >=1 to returned batched outputs. "
                "To return all points at once, set points_per_batch to None"
            )
    
        # 分批生成输出
        for i in range(0, n_points, points_per_batch):
            batched_points = grid_points[:, i : i + points_per_batch, :, :]
            labels = input_labels[:, i : i + points_per_batch]
            is_last = i == n_points - points_per_batch
            yield {
                "input_points": batched_points,
                "input_labels": labels,
                "input_boxes": crop_boxes,
                "is_last": is_last,
                **model_inputs,
            }
    
    # 前向传播函数
    def _forward(
        self,
        model_inputs,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        mask_threshold=0,
        stability_score_offset=1,
    ):
    ):
        # 从模型输入中弹出输入框信息
        input_boxes = model_inputs.pop("input_boxes")
        # 从模型输入中弹出是否为最后一个批次信息
        is_last = model_inputs.pop("is_last")
        # 从模型输入中弹出原始尺寸信息，并转换为列表
        original_sizes = model_inputs.pop("original_sizes").tolist()
        # 从模型输入中弹出调整后输入尺寸信息，并转换为列表
        reshaped_input_sizes = model_inputs.pop("reshaped_input_sizes").tolist()

        # 使用模型对模型的输入进行预测
        model_outputs = self.model(**model_inputs)

        # 预处理过程在此处进行，以避免将所有掩模的CPU GPU复制
        low_resolution_masks = model_outputs["pred_masks"]
        # 对低分辨率掩模进行后处理，得到掩模，同时应用指定的阈值并进行二值化
        masks = self.image_processor.post_process_masks(
            low_resolution_masks, original_sizes, reshaped_input_sizes, mask_threshold, binarize=False
        )
        # 获取模型输出的IoU分数
        iou_scores = model_outputs["iou_scores"]
        # 筛选掩模，IoU分数和边界框，应用指定的阈值
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
        # 返回结果字典
        return {
            "masks": masks,
            "is_last": is_last,
            "boxes": boxes,
            "iou_scores": iou_scores,
        }

    # 后处理方法
    def postprocess(
        self,
        model_outputs,
        output_rle_mask=False,
        output_bboxes_mask=False,
        crops_nms_thresh=0.7,
    ):
        # 初始化存储所有分数、所有掩模和所有边界框的列表
        all_scores = []
        all_masks = []
        all_boxes = []
        # 遍历模型输出
        for model_output in model_outputs:
            # 弹出IoU分数，并添加到分数列表中
            all_scores.append(model_output.pop("iou_scores"))
            # 扩展掩模列表
            all_masks.extend(model_output.pop("masks"))
            # 弹出边界框信息，并添加到边界框列表中
            all_boxes.append(model_output.pop("boxes"))

        # 对所有IoU分数和边界框进行连接
        all_scores = torch.cat(all_scores)
        all_boxes = torch.cat(all_boxes)
        # 对所有掩模、所有分数和所有边界框进行后处理，生成掩模，IoU分数，RLE掩模和边界框
        output_masks, iou_scores, rle_mask, bounding_boxes = self.image_processor.post_process_for_mask_generation(
            all_masks, all_scores, all_boxes, crops_nms_thresh
        )

        # 初始化额外信息字典
        extra = defaultdict(list)
        # 遍历模型输出中的每个输出
        for output in model_outputs:
            # 遍历每个输出项，添加到额外信息字典中
            for k, v in output.items():
                extra[k].append(v)

        # 初始化可选项字典
        optional = {}
        # 如果需要输出RLE掩模，则将其添加到可选项字典中
        if output_rle_mask:
            optional["rle_mask"] = rle_mask

        # 如果需要输出边界框掩模，则将其添加到可选项字典中
        if output_bboxes_mask:
            optional["bounding_boxes"] = bounding_boxes

        # 返回结果字典，包括掩模，分数，可选项和额外信息
        return {"masks": output_masks, "scores": iou_scores, **optional, **extra}
```