# `.\yolov8\examples\YOLOv8-ONNXRuntime\main.py`

```py
# 导入需要的库和模块
import argparse  # 用于解析命令行参数

import cv2  # OpenCV库，用于图像处理
import numpy as np  # NumPy库，用于数值计算
import onnxruntime as ort  # ONNX Runtime，用于加载和运行ONNX模型
import torch  # PyTorch库，用于深度学习模型

# 导入自定义的工具函数和模块
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml


class YOLOv8:
    """YOLOv8对象检测模型类，处理推理和可视化。"""

    def __init__(self, onnx_model, input_image, confidence_thres, iou_thres):
        """
        初始化YOLOv8类的实例。

        Args:
            onnx_model: ONNX模型的路径。
            input_image: 输入图像的路径。
            confidence_thres: 过滤检测的置信度阈值。
            iou_thres: 非最大抑制的IoU（交并比）阈值。
        """
        self.onnx_model = onnx_model  # 设置ONNX模型的路径
        self.input_image = input_image  # 设置输入图像的路径
        self.confidence_thres = confidence_thres  # 设置置信度阈值
        self.iou_thres = iou_thres  # 设置IoU阈值

        # 从COCO数据集的yaml文件中加载类别名称
        self.classes = yaml_load(check_yaml("coco8.yaml"))["names"]

        # 为每个类别生成颜色调色板
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, img, box, score, class_id):
        """
        根据检测到的对象在输入图像上绘制边界框和标签。

        Args:
            img: 要绘制检测结果的输入图像。
            box: 检测到的边界框。
            score: 对应的检测置信度。
            class_id: 检测对象的类别ID。

        Returns:
            None
        """

        # 提取边界框的坐标
        x1, y1, w, h = box

        # 获取该类别ID对应的颜色
        color = self.color_palette[class_id]

        # 在图像上绘制边界框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # 创建包含类别名称和置信度的标签文本
        label = f"{self.classes[class_id]}: {score:.2f}"

        # 计算标签文本的尺寸
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # 计算标签文本的位置
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # 在标签文本背景上绘制填充的矩形
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # 在图像上绘制标签文本
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # 预处理输入图像以便进行推断

    # 使用 OpenCV 读取输入图像
    self.img = cv2.imread(self.input_image)

    # 获取输入图像的高度和宽度
    self.img_height, self.img_width = self.img.shape[:2]

    # 将图像的颜色空间从BGR转换为RGB
    img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

    # 调整图像大小以匹配输入形状
    img = cv2.resize(img, (self.input_width, self.input_height))

    # 将图像数据标准化，通过除以255.0将像素值缩放到[0, 1]范围内
    image_data = np.array(img) / 255.0

    # 转置图像数组，使通道维度成为第一维度
    image_data = np.transpose(image_data, (2, 0, 1))  # 通道维度优先

    # 扩展图像数据的维度以匹配预期的输入形状
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

    # 返回预处理后的图像数据
    return image_data
    # 对模型输出进行转置和挤压，以匹配预期的形状
    outputs = np.transpose(np.squeeze(output[0]))

    # 获取输出数组的行数
    rows = outputs.shape[0]

    # 存储检测结果的列表：边界框、置信度分数和类别 ID
    boxes = []
    scores = []
    class_ids = []

    # 计算边界框坐标的缩放因子
    x_factor = self.img_width / self.input_width
    y_factor = self.img_height / self.input_height

    # 遍历输出数组中的每一行
    for i in range(rows):
        # 从当前行提取类别分数
        classes_scores = outputs[i][4:]

        # 找出类别分数中的最大值
        max_score = np.amax(classes_scores)

        # 如果最大分数超过置信度阈值
        if max_score >= self.confidence_thres:
            # 获取具有最高分数的类别 ID
            class_id = np.argmax(classes_scores)

            # 从当前行提取边界框坐标
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

            # 计算边界框的缩放后坐标
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            # 将类别 ID、分数和边界框坐标添加到相应的列表中
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

    # 应用非极大值抑制以过滤重叠的边界框
    indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

    # 遍历非极大值抑制后选择的索引
    for i in indices:
        # 获取与索引对应的边界框、分数和类别 ID
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]

        # 在输入图像上绘制检测结果
        self.draw_detections(input_image, box, score, class_id)

    # 返回修改后的输入图像
    return input_image
    def main(self):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        # 创建一个 ONNX 模型推理会话，并指定执行提供者为 CUDAExecutionProvider 和 CPUExecutionProvider
        session = ort.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        # 获取模型的输入信息
        model_inputs = session.get_inputs()

        # 存储输入的形状以便后续使用
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        # 对图像数据进行预处理
        img_data = self.preprocess()

        # 使用预处理后的图像数据进行推理
        outputs = session.run(None, {model_inputs[0].name: img_data})

        # 对输出进行后处理以获取输出图像
        return self.postprocess(self.img, outputs)  # 返回输出图像
if __name__ == "__main__":
    # 创建一个参数解析器来处理命令行参数
    parser = argparse.ArgumentParser()
    # 添加命令行参数：模型文件的路径，默认为'yolov8n.onnx'
    parser.add_argument("--model", type=str, default="yolov8n.onnx", help="Input your ONNX model.")
    # 添加命令行参数：输入图片的路径，默认为'ASSETS/bus.jpg'
    parser.add_argument("--img", type=str, default=str(ASSETS / "bus.jpg"), help="Path to input image.")
    # 添加命令行参数：置信度阈值，默认为0.5
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    # 添加命令行参数：NMS IoU（非极大值抑制的IoU阈值），默认为0.5
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    # 解析命令行参数并存储到args对象中
    args = parser.parse_args()

    # 检查系统环境要求，并选择适当的后端（CPU或GPU）
    check_requirements("onnxruntime-gpu" if torch.cuda.is_available() else "onnxruntime")

    # 使用指定的参数创建YOLOv8类的实例
    detection = YOLOv8(args.model, args.img, args.conf_thres, args.iou_thres)

    # 执行目标检测并获取输出图像
    output_image = detection.main()

    # 在一个窗口中显示输出图像
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", output_image)

    # 等待按键输入以退出
    cv2.waitKey(0)
```