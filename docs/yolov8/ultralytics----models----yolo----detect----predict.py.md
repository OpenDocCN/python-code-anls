# `.\yolov8\ultralytics\models\yolo\detect\predict.py`

```py
# 导入需要的模块和类
from ultralytics.engine.predictor import BasePredictor  # 从 Ultralytics 引擎中导入 BasePredictor 类
from ultralytics.engine.results import Results  # 从 Ultralytics 引擎中导入 Results 类
from ultralytics.utils import ops  # 从 Ultralytics 工具中导入 ops 模块


class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model='yolov8n.pt', source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```py
    """

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        # 进行非最大抑制处理，返回预测结果 preds
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # 如果输入的原始图像不是列表而是 torch.Tensor
            # 将 torch.Tensor 转换为 numpy 数组
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            # 缩放预测框的坐标，使其适应原始图像的尺寸
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            # 创建一个 Results 对象并添加到 results 列表中
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        # 返回处理后的 results 列表
        return results
```