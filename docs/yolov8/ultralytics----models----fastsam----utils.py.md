# `.\yolov8\ultralytics\models\fastsam\utils.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

# 调整边界框使其在一定阈值内粘合到图像边界

def adjust_bboxes_to_image_border(boxes, image_shape, threshold=20):
    """
    Adjust bounding boxes to stick to image border if they are within a certain threshold.

    Args:
        boxes (torch.Tensor): (n, 4) 边界框坐标
        image_shape (tuple): (height, width) 图像高度和宽度
        threshold (int): pixel threshold 像素阈值

    Returns:
        adjusted_boxes (torch.Tensor): adjusted bounding boxes 调整后的边界框
    """

    # 图像尺寸
    h, w = image_shape

    # 调整边界框
    boxes[boxes[:, 0] < threshold, 0] = 0  # x1 左上角 x 坐标
    boxes[boxes[:, 1] < threshold, 1] = 0  # y1 左上角 y 坐标
    boxes[boxes[:, 2] > w - threshold, 2] = w  # x2 右下角 x 坐标
    boxes[boxes[:, 3] > h - threshold, 3] = h  # y2 右下角 y 坐标
    return boxes
```