# `.\yolov8\ultralytics\models\yolo\segment\predict.py`

```py
# 导入必要的模块和类
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops

class SegmentationPredictor(DetectionPredictor):
    """
    一个扩展了DetectionPredictor类的类，用于基于分割模型进行预测。

    示例:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.segment import SegmentationPredictor

        args = dict(model='yolov8n-seg.pt', source=ASSETS)
        predictor = SegmentationPredictor(overrides=args)
        predictor.predict_cli()
        ```py
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        初始化SegmentationPredictor对象，使用提供的配置、覆盖和回调函数。
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"  # 设置预测任务为分割任务

    def postprocess(self, preds, img, orig_imgs):
        """
        对每个输入批次中的图像应用非最大抑制，并处理检测结果。
        """
        # 对预测结果应用非最大抑制
        p = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
        )

        # 如果输入图像不是一个列表，而是一个torch.Tensor，则转换为numpy数组
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []  # 初始化结果列表
        proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]  # 确定使用的协议格式
        for i, (pred, orig_img, img_path) in enumerate(zip(p, orig_imgs, self.batch[0])):
            if not len(pred):  # 如果预测结果为空，保存空框
                masks = None
            elif self.args.retina_masks:  # 如果需要返回掩膜
                # 缩放框，并处理原始图像生成掩膜
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                # 处理掩膜，生成掩膜，并缩放框
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            # 将处理后的结果添加到结果列表中
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        return results  # 返回处理后的结果列表
```