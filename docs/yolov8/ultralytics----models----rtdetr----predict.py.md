# `.\yolov8\ultralytics\models\rtdetr\predict.py`

```py
# 导入PyTorch库
import torch

# 导入图像处理库中的LetterBox类
from ultralytics.data.augment import LetterBox
# 导入预测器基类BasePredictor
from ultralytics.engine.predictor import BasePredictor
# 导入结果处理类Results
from ultralytics.engine.results import Results
# 导入工具操作库中的ops模块
from ultralytics.utils import ops

# 定义RT-DETR预测器类，继承自BasePredictor
class RTDETRPredictor(BasePredictor):
    """
    RT-DETR（Real-Time Detection Transformer）预测器，扩展自BasePredictor类，用于使用Baidu的RT-DETR模型进行预测。

    该类利用Vision Transformers实现实时目标检测，并保持高准确性。支持高效的混合编码和IoU感知的查询选择。

    示例：
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.rtdetr import RTDETRPredictor

        args = dict(model='rtdetr-l.pt', source=ASSETS)
        predictor = RTDETRPredictor(overrides=args)
        predictor.predict_cli()
        ```py

    属性：
        imgsz (int): 推断时的图像尺寸（必须是方形且填充为比例尺寸）。
        args (dict): 预测器的参数覆盖。

    """

    # 定义后处理方法，用于从模型的原始预测生成边界框和置信度分数
    def postprocess(self, preds, img, orig_imgs):
        """
        后处理方法，从模型的原始预测生成边界框和置信度分数。

        该方法基于置信度和类别（如果在self.args中指定）筛选检测结果。

        Args:
            preds (list): 模型的预测结果列表。
            img (torch.Tensor): 处理过的输入图像。
            orig_imgs (list or torch.Tensor): 原始未处理的图像。

        Returns:
            (list[Results]): 包含后处理边界框、置信度分数和类别标签的Results对象列表。
        """
        if not isinstance(preds, (list, tuple)):  # 对于PyTorch推断，预测结果是列表，但对于导出推断，预测结果是列表中的第一个张量
            preds = [preds, None]

        nd = preds[0].shape[-1]
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)

        if not isinstance(orig_imgs, list):  # 输入图像是torch.Tensor而不是列表
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for bbox, score, orig_img, img_path in zip(bboxes, scores, orig_imgs, self.batch[0]):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            max_score, cls = score.max(-1, keepdim=True)  # (300, 1)
            idx = max_score.squeeze(-1) > self.args.conf  # (300, )
            if self.args.classes is not None:
                idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx
            pred = torch.cat([bbox, max_score, cls], dim=-1)[idx]  # 进行过滤
            oh, ow = orig_img.shape[:2]
            pred[..., [0, 2]] *= ow
            pred[..., [1, 3]] *= oh
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
    # 定义一个方法用于在模型推理之前对输入图像进行预处理。
    # 输入图像将进行letterboxing以确保正方形纵横比并进行scale-fill处理。
    # 图像的大小必须是640x640，并且要进行scale-fill处理。

    def pre_transform(self, im):
        """
        Pre-transforms the input images before feeding them into the model for inference. The input images are
        letterboxed to ensure a square aspect ratio and scale-filled. The size must be square(640) and scaleFilled.

        Args:
            im (list[np.ndarray] |torch.Tensor): Input images of shape (N,3,h,w) for tensor, [(h,w,3) x N] for list.

        Returns:
            (list): List of pre-transformed images ready for model inference.
        """
        # 创建一个LetterBox对象，用于进行图像的letterboxing操作，保持图像尺寸为指定的self.imgsz大小，自动缩放填充。
        letterbox = LetterBox(self.imgsz, auto=False, scaleFill=True)
        # 对输入的每张图像进行预处理，应用上述定义的LetterBox对象进行处理。
        return [letterbox(image=x) for x in im]
```