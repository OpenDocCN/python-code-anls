# `.\yolov8\ultralytics\models\yolo\pose\predict.py`

```py
# 导入所需模块和类
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops

class PosePredictor(DetectionPredictor):
    """
    一个扩展自DetectionPredictor类的类，用于基于姿势模型的预测。

    示例:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.pose import PosePredictor

        args = dict(model='yolov8n-pose.pt', source=ASSETS)
        predictor = PosePredictor(overrides=args)
        predictor.predict_cli()
        ```py
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """初始化PosePredictor类，设置任务为'pose'，并记录使用'mps'设备时的警告信息。"""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "pose"
        # 如果设备是字符串类型且为'mps'，记录警告信息
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def postprocess(self, preds, img, orig_imgs):
        """对给定的输入图像或图像列表进行后处理，返回检测结果。"""
        # 执行非最大抑制操作，获取最终的预测结果
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
            nc=len(self.model.names),
        )

        # 如果输入的原始图像不是列表而是torch.Tensor
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        # 存储处理后的结果
        results = []
        # 对每个预测结果进行处理
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            # 缩放边界框坐标到原始图像尺寸
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
            # 如果存在关键点预测，也对关键点坐标进行缩放
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
            # 构建Results对象并添加到结果列表中
            results.append(
                Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], keypoints=pred_kpts)
            )
        # 返回最终的结果列表
        return results
```