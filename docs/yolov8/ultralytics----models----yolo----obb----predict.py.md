# `.\yolov8\ultralytics\models\yolo\obb\predict.py`

```py
# Ultralytics YOLO , AGPL-3.0 license

# 导入 PyTorch 库
import torch

# 导入 Ultralytics 相关模块和函数
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class OBBPredictor(DetectionPredictor):
    """
    一个扩展了 DetectionPredictor 类的类，用于基于定向边界框（OBB）模型进行预测。

    示例:
        ```py
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.obb import OBBPredictor

        args = dict(model='yolov8n-obb.pt', source=ASSETS)
        predictor = OBBPredictor(overrides=args)
        predictor.predict_cli()
        ```py
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """初始化 OBBPredictor 类，可选择模型和数据配置的覆盖设置。"""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "obb"

    def postprocess(self, preds, img, orig_imgs):
        """后处理预测结果并返回 Results 对象的列表。"""
        # 执行非最大抑制以筛选预测框
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
            rotated=True,
        )

        # 如果输入的原始图像不是列表而是一个 torch.Tensor
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        # 遍历每个预测结果、原始图像和图像路径
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            # 规范化旋转框坐标并进行缩放调整
            rboxes = ops.regularize_rboxes(torch.cat([pred[:, :4], pred[:, -1:]], dim=-1))
            rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
            # 创建包含 OBB 信息的 tensor：xywh, r, conf, cls
            obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)
            # 将处理后的结果添加到 results 列表中
            results.append(Results(orig_img, path=img_path, names=self.model.names, obb=obb))
        return results
```