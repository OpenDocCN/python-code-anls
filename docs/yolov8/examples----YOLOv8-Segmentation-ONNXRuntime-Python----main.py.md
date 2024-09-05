# `.\yolov8\examples\YOLOv8-Segmentation-ONNXRuntime-Python\main.py`

```py
# Ultralytics YOLO , AGPL-3.0 license

import argparse  # 导入命令行参数解析模块

import cv2  # 导入 OpenCV 库
import numpy as np  # 导入 NumPy 库
import onnxruntime as ort  # 导入 ONNX Runtime 库

from ultralytics.utils import ASSETS, yaml_load  # 从 ultralytics.utils 中导入 ASSETS 和 yaml_load 函数
from ultralytics.utils.checks import check_yaml  # 从 ultralytics.utils.checks 中导入 check_yaml 函数
from ultralytics.utils.plotting import Colors  # 从 ultralytics.utils.plotting 中导入 Colors 类


class YOLOv8Seg:
    """YOLOv8 segmentation model."""

    def __init__(self, onnx_model):
        """
        Initialization.

        Args:
            onnx_model (str): Path to the ONNX model.
        """

        # Build Ort session
        self.session = ort.InferenceSession(
            onnx_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"],
        )

        # Numpy dtype: support both FP32 and FP16 onnx model
        self.ndtype = np.half if self.session.get_inputs()[0].type == "tensor(float16)" else np.single

        # Get model width and height(YOLOv8-seg only has one input)
        self.model_height, self.model_width = [x.shape for x in self.session.get_inputs()][0][-2:]

        # Load COCO class names
        self.classes = yaml_load(check_yaml("coco8.yaml"))["names"]

        # Create color palette
        self.color_palette = Colors()

    def __call__(self, im0, conf_threshold=0.4, iou_threshold=0.45, nm=32):
        """
        The whole pipeline: pre-process -> inference -> post-process.

        Args:
            im0 (Numpy.ndarray): original input image.
            conf_threshold (float): confidence threshold for filtering predictions.
            iou_threshold (float): iou threshold for NMS.
            nm (int): the number of masks.

        Returns:
            boxes (List): list of bounding boxes.
            segments (List): list of segments.
            masks (np.ndarray): [N, H, W], output masks.
        """

        # Pre-process
        im, ratio, (pad_w, pad_h) = self.preprocess(im0)  # 调用 preprocess 方法进行图像预处理

        # Ort inference
        preds = self.session.run(None, {self.session.get_inputs()[0].name: im})  # 使用 ONNX Runtime 进行推理

        # Post-process
        boxes, segments, masks = self.postprocess(
            preds,
            im0=im0,
            ratio=ratio,
            pad_w=pad_w,
            pad_h=pad_h,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            nm=nm,
        )  # 调用 postprocess 方法进行后处理
        return boxes, segments, masks
    def preprocess(self, img):
        """
        Pre-processes the input image.

        Args:
            img (Numpy.ndarray): image about to be processed.

        Returns:
            img_process (Numpy.ndarray): image preprocessed for inference.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
        """

        # 使用 letterbox() 函数调整输入图像的大小并填充
        shape = img.shape[:2]  # 原始图像的形状
        new_shape = (self.model_height, self.model_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r  # 计算宽高比例
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 计算调整后的尺寸
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # 计算填充的宽度和高度
        if shape[::-1] != new_unpad:  # 如果尺寸不一致，则进行 resize 操作
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        # 使用指定颜色进行边界填充
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # 图像转换流程：HWC 转换为 CHW -> BGR 转换为 RGB -> 归一化处理 -> 连续化处理 -> 添加额外维度（可选）
        img = np.ascontiguousarray(np.einsum("HWC->CHW", img)[::-1], dtype=self.ndtype) / 255.0
        img_process = img[None] if len(img.shape) == 3 else img  # 添加额外维度以适应网络输入
        return img_process, ratio, (pad_w, pad_h)
    def postprocess(self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32):
        """
        Post-process the prediction.

        Args:
            preds (Numpy.ndarray): predictions come from ort.session.run().
            im0 (Numpy.ndarray): [h, w, c] original input image.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
            conf_threshold (float): conf threshold.
            iou_threshold (float): iou threshold.
            nm (int): the number of masks.

        Returns:
            boxes (List): list of bounding boxes.
            segments (List): list of segments.
            masks (np.ndarray): [N, H, W], output masks.
        """
        x, protos = preds[0], preds[1]  # Two outputs: predictions and protos

        # Transpose the first output: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
        x = np.einsum("bcn->bnc", x)

        # Predictions filtering by conf-threshold
        x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]

        # Create a new matrix which merge these(box, score, cls, nm) into one
        # For more details about `numpy.c_()`: https://numpy.org/doc/1.26/reference/generated/numpy.c_.html
        x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]

        # NMS filtering
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]

        # Decode and return
        if len(x) > 0:
            # Bounding boxes format change: cxcywh -> xyxy
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            # Bounding boxes boundary clamp
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])

            # Process masks
            masks = self.process_mask(protos[0], x[:, 6:], x[:, :4], im0.shape)

            # Masks -> Segments(contours)
            segments = self.masks2segments(masks)
            return x[..., :6], segments, masks  # boxes, segments, masks
        else:
            return [], [], []


注释：

        x, protos = preds[0], preds[1]  # 从预测结果中分离出两个输出：预测和原型
        x = np.einsum("bcn->bnc", x)  # 转置第一个输出：(Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
        x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]  # 根据置信度阈值过滤预测结果
        x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]  # 将(box, score, cls, nm)合并成一个新矩阵
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]  # 使用非极大值抑制筛选边界框
        if len(x) > 0:
            x[..., [0, 1]] -= x[..., [2, 3]] / 2  # 边界框格式从cxcywh转换为xyxy
            x[..., [2, 3]] += x[..., [0, 1]]  # 调整边界框坐标
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]  # 将边界框从模型尺寸转换为原始图像尺寸
            x[..., :4] /= min(ratio)  # 根据图像缩放比例重新缩放边界框
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])  # 对边界框的x坐标进行边界限制
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])  # 对边界框的y坐标进行边界限制
            masks = self.process_mask(protos[0], x[:, 6:], x[:, :4], im0.shape)  # 处理生成掩码
            segments = self.masks2segments(masks)  # 将掩码转换为轮廓
            return x[..., :6], segments, masks  # 返回边界框、轮廓和掩码
        else:
            return [], [], []  # 如果没有检测到边界框，返回空列表
    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        """
        Takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher quality
        but is slower. (Borrowed from https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L618)

        Args:
            protos (numpy.ndarray): [mask_dim, mask_h, mask_w].
                用于生成掩码的原型张量，其形状为 [掩码维度, 掩码高度, 掩码宽度]。
            masks_in (numpy.ndarray): [n, mask_dim], n is number of masks after nms.
                经过非极大值抑制后的掩码张量，形状为 [n, 掩码维度]。
            bboxes (numpy.ndarray): bboxes re-scaled to original image shape.
                重新缩放到原始图像形状的边界框坐标张量，形状为 [n, 4]。
            im0_shape (tuple): the size of the input image (h,w,c).
                输入图像的尺寸，以元组形式表示 (高度, 宽度, 通道数)。

        Returns:
            (numpy.ndarray): The upsampled masks.
                返回经处理的掩码，形状为 [n, mask_h, mask_w]。
        """
        c, mh, mw = protos.shape
        # 使用原型张量和掩码输入执行矩阵乘法，得到掩码的高质量版本
        masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN
        masks = np.ascontiguousarray(masks)
        # 将掩码从 P3 形状重新缩放到原始输入图像形状
        masks = self.scale_mask(masks, im0_shape)
        # 对掩码进行转置，从 HWN 形状转换为 NHW 形状
        masks = np.einsum("HWN -> NHW", masks)
        # 根据边界框裁剪掩码
        masks = self.crop_mask(masks, bboxes)
        # 将掩码中大于0.5的部分设置为True，小于等于0.5的部分设置为False
        return np.greater(masks, 0.5)
    def scale_mask(masks, im0_shape, ratio_pad=None):
        """
        Takes a mask, and resizes it to the original image size. (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L305)

        Args:
            masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
            im0_shape (tuple): the original image shape.
            ratio_pad (tuple): the ratio of the padding to the original image.

        Returns:
            masks (np.ndarray): The masks that are being returned.
        """
        # 获取当前 masks 的形状，取前两个维度（高度和宽度）
        im1_shape = masks.shape[:2]
        
        # 如果 ratio_pad 为 None，则根据 im0_shape 计算比例
        if ratio_pad is None:
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # 计算缩放比例 gain = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # 计算填充量 pad = (width_padding, height_padding)
        else:
            pad = ratio_pad[1]  # 否则直接取 ratio_pad 的第二个元素作为 pad

        # 计算 mask 的 top-left 和 bottom-right 边界
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # 计算顶部和左侧边界
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))  # 计算底部和右侧边界
        
        # 如果 masks 的形状维度小于 2，则抛出 ValueError 异常
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        
        # 根据计算得到的边界裁剪 masks
        masks = masks[top:bottom, left:right]
        
        # 使用 OpenCV 进行图像缩放至原始图像大小
        masks = cv2.resize(
            masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR
        )  # 使用线性插值进行缩放，也可以考虑使用 INTER_CUBIC

        # 如果 masks 的形状维度为 2，则添加一个维度，使其变为三维
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        
        # 返回调整大小后的 masks
        return masks
    def draw_and_visualize(self, im, bboxes, segments, vis=False, save=True):
        """
        Draw and visualize results.

        Args:
            im (np.ndarray): original image, shape [h, w, c].
            bboxes (numpy.ndarray): [n, 4], n is number of bboxes.
            segments (List): list of segment masks.
            vis (bool): imshow using OpenCV.
            save (bool): save image annotated.

        Returns:
            None
        """

        # 复制原始图像作为绘图画布
        im_canvas = im.copy()

        # 遍历边界框和分割掩码
        for (*box, conf, cls_), segment in zip(bboxes, segments):
            # 绘制多边形边界，并填充分割掩码
            cv2.polylines(im, np.int32([segment]), True, (255, 255, 255), 2)  # 白色边界线
            cv2.fillPoly(im_canvas, np.int32([segment]), self.color_palette(int(cls_), bgr=True))

            # 绘制边界框矩形
            cv2.rectangle(
                im,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                self.color_palette(int(cls_), bgr=True),
                1,
                cv2.LINE_AA,
            )

            # 添加边界框标签
            cv2.putText(
                im,
                f"{self.classes[cls_]}: {conf:.3f}",
                (int(box[0]), int(box[1] - 9)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.color_palette(int(cls_), bgr=True),
                2,
                cv2.LINE_AA,
            )

        # 混合原始图像和绘图结果
        im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)

        # 显示图像
        if vis:
            cv2.imshow("demo", im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # 保存图像
        if save:
            cv2.imwrite("demo.jpg", im)
# 如果当前脚本被作为主程序执行，则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器，用于处理命令行参数
    parser = argparse.ArgumentParser()
    # 添加必需的参数：模型文件的路径
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    # 添加可选参数：输入图像的路径，默认为 ASSETS 目录下的 bus.jpg
    parser.add_argument("--source", type=str, default=str(ASSETS / "bus.jpg"), help="Path to input image")
    # 添加可选参数：置信度阈值，默认为 0.25
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    # 添加可选参数：IoU 阈值，默认为 0.45
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    # 解析命令行参数
    args = parser.parse_args()

    # 构建模型实例，使用指定的 ONNX 模型路径
    model = YOLOv8Seg(args.model)

    # 使用 OpenCV 读取指定路径的图像文件
    img = cv2.imread(args.source)

    # 进行推理
    boxes, segments, _ = model(img, conf_threshold=args.conf, iou_threshold=args.iou)

    # 绘制边界框和多边形
    if len(boxes) > 0:
        # 在图像上绘制边界框和多边形，并根据需要保存或显示
        model.draw_and_visualize(img, boxes, segments, vis=False, save=True)
```