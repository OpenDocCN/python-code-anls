# `.\segment-anything\segment_anything\utils\onnx.py`

```
# 导入 torch 库中的相关模块
import torch
# 导入 torch 中的神经网络模块
import torch.nn as nn
# 导入 torch 中的函数模块
from torch.nn import functional as F

# 导入 Tuple 类型
from typing import Tuple

# 从当前目录下的 modeling 模块中导入 Sam 类
from ..modeling import Sam
# 从当前目录下的 amg 模块中导入 calculate_stability_score 函数

# 定义 SamOnnxModel 类，继承自 nn.Module
class SamOnnxModel(nn.Module):
    """
    This model should not be called directly, but is used in ONNX export.
    It combines the prompt encoder, mask decoder, and mask postprocessing of Sam,
    with some functions modified to enable model tracing. Also supports extra
    options controlling what information. See the ONNX export script for details.
    """

    # 初始化函数，接受参数 model, return_single_mask, use_stability_score, return_extra_metrics
    def __init__(
        self,
        model: Sam,
        return_single_mask: bool,
        use_stability_score: bool = False,
        return_extra_metrics: bool = False,
    ) -> None:
        # 调用父类的初始化函数
        super().__init__()
        # 将 model 的 mask_decoder 赋值给当前对象的 mask_decoder
        self.mask_decoder = model.mask_decoder
        # 将 model 赋值给当前对象的 model
        self.model = model
        # 将 model 的 image_encoder 的 img_size 赋值给当前对象的 img_size
        self.img_size = model.image_encoder.img_size
        # 将 return_single_mask 赋值给当前对象的 return_single_mask
        self.return_single_mask = return_single_mask
        # 将 use_stability_score 赋值给当前对象的 use_stability_score
        self.use_stability_score = use_stability_score
        # 设置 stability_score_offset 为 1.0
        self.stability_score_offset = 1.0
        # 将 return_extra_metrics 赋值给当前对象的 return_extra_metrics
        self.return_extra_metrics = return_extra_metrics

    # 静态方法，用于调整输入图像大小
    @staticmethod
    def resize_longest_image_size(
        input_image_size: torch.Tensor, longest_side: int
    ) -> torch.Tensor:
        # 将 input_image_size 转换为 float32 类型
        input_image_size = input_image_size.to(torch.float32)
        # 计算缩放比例
        scale = longest_side / torch.max(input_image_size)
        # 计算调整后的图像大小
        transformed_size = scale * input_image_size
        # 对调整后的大小进行四舍五入取整
        transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
        # 返回调整后的图像大小
        return transformed_size
    # 将点的坐标加上0.5，用于将坐标值从0-1映射到0.5-1.5范围
    point_coords = point_coords + 0.5
    # 将点的坐标值除以图像大小，用于将坐标值映射到0-1范围
    point_coords = point_coords / self.img_size
    # 使用位置编码器对点的坐标进行编码
    point_embedding = self.model.prompt_encoder.pe_layer._pe_encoding(point_coords)
    # 将点的标签扩展为与点编码相同的形状
    point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

    # 根据点的标签，对点的编码进行处理
    point_embedding = point_embedding * (point_labels != -1)
    point_embedding = point_embedding + self.model.prompt_encoder.not_a_point_embed.weight * (
        point_labels == -1
    )

    # 遍历点的数量，根据点的标签对点的编码进行处理
    for i in range(self.model.prompt_encoder.num_point_embeddings):
        point_embedding = point_embedding + self.model.prompt_encoder.point_embeddings[
            i
        ].weight * (point_labels == i)

    # 返回处理后的点的编码
    return point_embedding

    # 根据输入的掩码和是否有掩码输入，对掩码进行处理
    mask_embedding = has_mask_input * self.model.prompt_encoder.mask_downscaling(input_mask)
    mask_embedding = mask_embedding + (
        1 - has_mask_input
    ) * self.model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
    # 返回处理后的掩码编码
    return mask_embedding

    # 对掩码进行后处理，调整大小并裁剪到原始图像大小
    masks = F.interpolate(
        masks,
        size=(self.img_size, self.img_size),
        mode="bilinear",
        align_corners=False,
    )

    prepadded_size = self.resize_longest_image_size(orig_im_size, self.img_size).to(torch.int64)
    masks = masks[..., : prepadded_size[0], : prepadded_size[1]]  # type: ignore

    orig_im_size = orig_im_size.to(torch.int64)
    h, w = orig_im_size[0], orig_im_size[1]
    masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
    # 返回处理后的掩码
    return masks

    # 选择掩码，根据IOU预测和点的数量
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 确定是否应该从点的数量返回多次点击掩码
        # 重新加权用于避免控制流
        score_reweight = torch.tensor(
            [[1000] + [0] * (self.model.mask_decoder.num_mask_tokens - 1)]
        ).to(iou_preds.device)
        # 计算得分，考虑点的数量
        score = iou_preds + (num_points - 2.5) * score_reweight
        # 找到最高得分的索引
        best_idx = torch.argmax(score, dim=1)
        # 选择最佳掩码
        masks = masks[torch.arange(masks.shape[0]), best_idx, :, :].unsqueeze(1)
        # 选择最佳 IOU 预测
        iou_preds = iou_preds[torch.arange(masks.shape[0]), best_idx].unsqueeze(1)

        return masks, iou_preds

    @torch.no_grad()
    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
        orig_im_size: torch.Tensor,
    # 使用输入的点坐标和标签进行稀疏嵌入
    sparse_embedding = self._embed_points(point_coords, point_labels)
    # 使用输入的掩码和是否有掩码输入进行密集嵌入
    dense_embedding = self._embed_masks(mask_input, has_mask_input)

    # 使用模型的 mask_decoder 预测掩码和分数
    masks, scores = self.model.mask_decoder.predict_masks(
        image_embeddings=image_embeddings,
        image_pe=self.model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embedding,
        dense_prompt_embeddings=dense_embedding,
    )

    # 如果需要使用稳定性分数
    if self.use_stability_score:
        # 计算稳定性分数
        scores = calculate_stability_score(
            masks, self.model.mask_threshold, self.stability_score_offset
        )

    # 如果需要返回单个掩码
    if self.return_single_mask:
        # 选择单个掩码
        masks, scores = self.select_masks(masks, scores, point_coords.shape[1])

    # 对掩码进行后处理，将其放大到原始图像大小
    upscaled_masks = self.mask_postprocessing(masks, orig_im_size)

    # 如果需要返回额外的指标
    if self.return_extra_metrics:
        # 计算放大后的掩码的稳定性分数
        stability_scores = calculate_stability_score(
            upscaled_masks, self.model.mask_threshold, self.stability_score_offset
        )
        # 计算掩码的面积
        areas = (upscaled_masks > self.model.mask_threshold).sum(-1).sum(-1)
        # 返回放大后的掩码、分数、稳定性分数、面积和原始掩码
        return upscaled_masks, scores, stability_scores, areas, masks

    # 返回放大后的掩码、分数和原始掩码
    return upscaled_masks, scores, masks
```