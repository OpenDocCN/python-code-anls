# `.\yolov8\ultralytics\models\fastsam\predict.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license
import torch
from PIL import Image

from ultralytics.models.yolo.segment import SegmentationPredictor  # 导入分割预测器类
from ultralytics.utils import DEFAULT_CFG, checks  # 导入默认配置和检查工具
from ultralytics.utils.metrics import box_iou  # 导入 IoU 计算工具
from ultralytics.utils.ops import scale_masks  # 导入 mask 缩放操作

from .utils import adjust_bboxes_to_image_border  # 导入边界框调整函数


class FastSAMPredictor(SegmentationPredictor):
    """
    FastSAMPredictor is specialized for fast SAM (Segment Anything Model) segmentation prediction tasks in Ultralytics
    YOLO framework.

    This class extends the SegmentationPredictor, customizing the prediction pipeline specifically for fast SAM. It
    adjusts post-processing steps to incorporate mask prediction and non-max suppression while optimizing for single-
    class segmentation.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        # 调用父类构造函数，初始化 FastSAMPredictor 对象
        super().__init__(cfg, overrides, _callbacks)
        # 初始化提示信息为空字典
        self.prompts = {}

    def postprocess(self, preds, img, orig_imgs):
        """Applies box postprocess for FastSAM predictions."""
        # 从提示信息中取出边界框、点、标签和文本信息
        bboxes = self.prompts.pop("bboxes", None)
        points = self.prompts.pop("points", None)
        labels = self.prompts.pop("labels", None)
        texts = self.prompts.pop("texts", None)
        # 调用父类的 postprocess 方法进行预测结果后处理
        results = super().postprocess(preds, img, orig_imgs)
        # 遍历每个结果
        for result in results:
            # 创建一个包含整个图像边界的框
            full_box = torch.tensor(
                [0, 0, result.orig_shape[1], result.orig_shape[0]], device=preds[0].device, dtype=torch.float32
            )
            # 调整结果中的边界框，使其适应图像边界
            boxes = adjust_bboxes_to_image_border(result.boxes.xyxy, result.orig_shape)
            # 找到与整个图像边界框 IoU 大于 0.9 的边界框索引
            idx = torch.nonzero(box_iou(full_box[None], boxes) > 0.9).flatten()
            # 如果找到匹配的边界框索引，则将这些边界框设置为整个图像边界框
            if idx.numel() != 0:
                result.boxes.xyxy[idx] = full_box

        # 返回处理后的结果，并将原始提示信息传递给下一个函数
        return self.prompt(results, bboxes=bboxes, points=points, labels=labels, texts=texts)
    def _clip_inference(self, images, texts):
        """
        CLIP Inference process.

        Args:
            images (List[PIL.Image]): A list of source images and each of them should be PIL.Image type with RGB channel order.
            texts (List[str]): A list of prompt texts and each of them should be string object.

        Returns:
            (torch.Tensor): The similarity between given images and texts.
        """
        try:
            import clip  # 尝试导入 CLIP 库
        except ImportError:
            checks.check_requirements("git+https://github.com/ultralytics/CLIP.git")  # 如果导入失败，则检查并安装所需的依赖
            import clip  # 再次尝试导入 CLIP 库

        if (not hasattr(self, "clip_model")) or (not hasattr(self, "clip_preprocess")):
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            # 如果对象实例中没有 clip_model 或 clip_preprocess 属性，则加载 CLIP 模型和预处理器

        images = torch.stack([self.clip_preprocess(image).to(self.device) for image in images])
        # 将输入的图像列表转换为 torch 张量，并使用 clip_preprocess 进行预处理，并移到设备上

        tokenized_text = clip.tokenize(texts).to(self.device)
        # 对输入的文本列表进行标记化，并移到设备上

        image_features = self.clip_model.encode_image(images)
        # 使用 CLIP 模型对图像进行编码，得到图像特征

        text_features = self.clip_model.encode_text(tokenized_text)
        # 使用 CLIP 模型对文本进行编码，得到文本特征

        image_features /= image_features.norm(dim=-1, keepdim=True)  # 对图像特征进行归一化处理
        text_features /= text_features.norm(dim=-1, keepdim=True)  # 对文本特征进行归一化处理

        return (image_features * text_features[:, None]).sum(-1)  # 计算图像和文本之间的相似性
        # 返回图像和文本之间的相似性得分，形状为 (M, N)

    def set_prompts(self, prompts):
        """Set prompts in advance."""
        self.prompts = prompts
        # 设置预设提示信息，存储在对象实例的 prompts 属性中
```