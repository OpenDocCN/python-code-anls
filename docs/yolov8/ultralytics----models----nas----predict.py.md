# `.\yolov8\ultralytics\models\nas\predict.py`

```py
# 导入 PyTorch 库
import torch

# 从 Ultralytics 引擎中导入基础预测器、结果和操作工具
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class NASPredictor(BasePredictor):
    """
    Ultralytics YOLO NAS 预测器，用于目标检测。

    这个类扩展了 Ultralytics 引擎中的 `BasePredictor`，负责对 YOLO NAS 模型生成的原始预测进行后处理。
    它应用了非极大值抑制和缩放边界框以适应原始图像尺寸等操作。

    Attributes:
        args (Namespace): 包含各种后处理配置的命名空间。

    Example:
        ```python
        from ultralytics import NAS

        model = NAS('yolo_nas_s')
        predictor = model.predictor
        # 假设 raw_preds, img, orig_imgs 可用
        results = predictor.postprocess(raw_preds, img, orig_imgs)
        ```py

    Note:
        通常情况下，不会直接实例化这个类，而是在 `NAS` 类的内部使用。

    """

    def postprocess(self, preds_in, img, orig_imgs):
        """后处理预测结果并返回 Results 对象的列表。"""

        # 将预测结果转换为 xywh 格式的边界框
        boxes = ops.xyxy2xywh(preds_in[0][0])
        # 将边界框和类别分数连接起来，并进行维度变换
        preds = torch.cat((boxes, preds_in[0][1]), -1).permute(0, 2, 1)

        # 应用非极大值抑制处理预测结果
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        # 如果输入图像不是列表而是 torch.Tensor，则转换为 numpy 数组的批量
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        # 初始化结果列表
        results = []
        # 遍历每个预测结果、原始图像和图像路径，生成 Results 对象并添加到 results 列表中
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            # 缩放边界框以适应原始图像尺寸
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        # 返回最终的 results 列表
        return results
```