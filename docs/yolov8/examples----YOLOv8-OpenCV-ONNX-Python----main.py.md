# `.\yolov8\examples\YOLOv8-OpenCV-ONNX-Python\main.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

# 导入必要的库
import argparse  # 用于解析命令行参数

import cv2.dnn  # OpenCV的深度学习模块
import numpy as np  # 用于处理图像数据的库

from ultralytics.utils import ASSETS, yaml_load  # 导入自定义工具函数和数据
from ultralytics.utils.checks import check_yaml  # 导入检查 YAML 文件的函数

# 从 coco8.yaml 文件中加载类别名称列表
CLASSES = yaml_load(check_yaml("coco8.yaml"))["names"]

# 随机生成用于每个类别的颜色
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    在输入图像上绘制边界框，基于提供的参数。

    Args:
        img (numpy.ndarray): 要绘制边界框的输入图像。
        class_id (int): 检测到对象的类别ID。
        confidence (float): 检测到对象的置信度分数。
        x (int): 边界框左上角的X坐标。
        y (int): 边界框左上角的Y坐标。
        x_plus_w (int): 边界框右下角的X坐标。
        y_plus_h (int): 边界框右下角的Y坐标。
    """
    # 根据类别ID获取类别名称和置信度，构建标签
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]  # 根据类别ID获取颜色
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)  # 在图像上绘制矩形边界框
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # 在边界框上方绘制类别标签


def main(onnx_model, input_image):
    """
    主函数，加载ONNX模型，执行推理，绘制边界框，并显示输出图像。

    Args:
        onnx_model (str): ONNX模型的路径。
        input_image (str): 输入图像的路径。

    Returns:
        list: 包含检测信息的字典列表，如类别ID、类别名称、置信度等。
    """
    # 加载ONNX模型
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)

    # 读取输入图像
    original_image: np.ndarray = cv2.imread(input_image)
    [height, width, _] = original_image.shape  # 获取原始图像的尺寸

    # 准备一个正方形图像进行推理
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image  # 将原始图像复制到正方形图像中

    # 计算缩放因子
    scale = length / 640

    # 对图像进行预处理并为模型准备blob
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)

    # 执行推理
    outputs = model.forward()

    # 准备输出数组
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    # 遍历输出以收集边界框、置信度分数和类别ID
    # 遍历检测到的每个目标框
    for i in range(rows):
        # 获取当前目标框的类别置信度分数
        classes_scores = outputs[0][i][4:]
        # 使用 cv2.minMaxLoc 函数找到最大置信度及其位置
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        # 如果最大置信度大于等于0.25，则处理该目标框
        if maxScore >= 0.25:
            # 计算目标框的左上角坐标及宽高
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            # 将目标框的坐标信息添加到列表中
            boxes.append(box)
            # 将最大置信度添加到分数列表中
            scores.append(maxScore)
            # 将最大置信度对应的类别索引添加到类别ID列表中
            class_ids.append(maxClassIndex)

    # 应用非极大值抑制（NMS）来剔除重叠的边界框
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    # 初始化检测结果列表
    detections = []

    # 遍历NMS后剩余的边界框结果，生成检测到的物体信息
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        # 获取对应索引的边界框信息
        box = boxes[index]
        # 构建单个检测结果的字典
        detection = {
            "class_id": class_ids[index],
            "class_name": CLASSES[class_ids[index]],  # 获取类别名称
            "confidence": scores[index],  # 获取置信度分数
            "box": box,  # 获取边界框坐标
            "scale": scale,  # 获取缩放比例
        }
        # 将当前检测结果添加到检测结果列表中
        detections.append(detection)
        # 在原始图像上绘制边界框及标签
        draw_bounding_box(
            original_image,
            class_ids[index],
            scores[index],
            round(box[0] * scale),  # 缩放后的左上角x坐标
            round(box[1] * scale),  # 缩放后的左上角y坐标
            round((box[0] + box[2]) * scale),  # 缩放后的右下角x坐标
            round((box[1] + box[3]) * scale),  # 缩放后的右下角y坐标
        )

    # 显示带有边界框的图像
    cv2.imshow("image", original_image)
    cv2.waitKey(0)  # 等待用户按键操作
    cv2.destroyAllWindows()  # 关闭所有图像窗口

    # 返回所有检测到的物体信息列表
    return detections
# 如果该脚本作为主程序运行，则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加一个名为 --model 的命令行参数，指定默认值为 'yolov8n.onnx'，用于输入 ONNX 模型
    parser.add_argument("--model", default="yolov8n.onnx", help="Input your ONNX model.")
    # 添加一个名为 --img 的命令行参数，指定默认值为 ASSETS 目录下的 'bus.jpg' 文件路径，用于输入图像
    parser.add_argument("--img", default=str(ASSETS / "bus.jpg"), help="Path to input image.")
    # 解析命令行参数，并将其存储在 args 对象中
    args = parser.parse_args()
    # 调用 main 函数，传入解析后的模型和图像路径作为参数
    main(args.model, args.img)
```