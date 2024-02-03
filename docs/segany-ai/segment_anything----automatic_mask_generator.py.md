# `.\segment-anything\segment_anything\automatic_mask_generator.py`

```
# 导入所需的库和模块
import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

# 导入类型提示相关的模块
from typing import Any, Dict, List, Optional, Tuple

# 导入自定义模块和类
from .modeling import Sam
from .predictor import SamPredictor
from .utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)

# 定义自动掩模生成器类
class SamAutomaticMaskGenerator:
    # 初始化方法，接受多个参数
    def __init__(
        self,
        model: Sam,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
    # 禁用梯度计算
    @torch.no_grad()
    # 生成掩码数据，根据输入的图像生成裁剪框和裁剪层索引
    def _generate_masks(self, image: np.ndarray) -> MaskData:
        # 获取原始图像的尺寸
        orig_size = image.shape[:2]
        # 生成裁剪框和裁剪层索引
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # 遍历图像裁剪
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            # 处理裁剪后的图像数据
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)

        # 移除裁剪之间的重复掩码
        if len(crop_boxes) > 1:
            # 优先选择较小裁剪框的掩码
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            # 使用非极大值抑制（NMS）保留掩码
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # 类别
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        # 将数据转换为 numpy 格式
        data.to_numpy()
        return data

    # 处理裁剪后的图像数据
    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        # 定义函数，接受一个裁剪框和图像，返回 MaskData 对象
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        # 根据裁剪框裁剪图像
        cropped_im = image[y0:y1, x0:x1, :]
        # 获取裁剪后图像的尺寸
        cropped_im_size = cropped_im.shape[:2]
        # 设置预测器的图像为裁剪后的图像
        self.predictor.set_image(cropped_im)

        # Get points for this crop
        # 计算裁剪框内的关键点坐标
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        # 为裁剪后的图像生成掩码
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            # 处理每个批次的数据
            batch_data = self._process_batch(points, cropped_im_size, crop_box, orig_size)
            data.cat(batch_data)
            del batch_data
        self.predictor.reset_image()

        # Remove duplicates within this crop.
        # 在裁剪框内去除重复的掩码
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        # 将数据返回到原始图像框架中
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        return data

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        # 获取原始图像的高度和宽度
        orig_h, orig_w = orig_size

        # 在该批次上运行模型
        # 将点坐标转换为模型需要的格式
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        # 使用模型进行预测
        masks, iou_preds, _ = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=True,
            return_logits=True,
        )

        # 序列化预测结果并存储在 MaskData 中
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks

        # 根据预测的 IoU 进行筛选
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        # 计算稳定性分数
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor.model.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        # 阈值化 masks 并计算 boxes
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # 过滤与裁剪边界接触的 boxes
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # 压缩为 RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data

    @staticmethod
    # 对小区域进行后处理，包括移除小的孤立区域和空洞，然后重新运行框NMS以移除任何新的重复项
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        # 如果mask_data中的rles为空，则直接返回mask_data
        if len(mask_data["rles"]) == 0:
            return mask_data

        # 过滤小的孤立区域和空洞
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            # 将rle编码的mask转换为mask
            mask = rle_to_mask(rle)

            # 移除小的孤立区域和空洞
            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # 给需要后处理的mask打分为0，给不需要后处理的mask打分为1，以便NMS优先选择不需要后处理的mask
            scores.append(float(unchanged))

        # 重新计算框并移除任何新的重复项
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # 仅对发生变化的mask重新计算RLEs
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # 直接更新结果
        mask_data.filter(keep_by_nms)

        return mask_data
```