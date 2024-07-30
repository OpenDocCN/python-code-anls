# `.\yolov8\examples\YOLOv8-OpenCV-int8-tflite-Python\main.py`

```py
# 导入 argparse 模块，用于处理命令行参数
import argparse

# 导入 OpenCV 库，用于图像处理操作
import cv2

# 导入 NumPy 库，用于数组和矩阵运算
import numpy as np

# 导入 TensorFlow Lite 解释器，用于在移动和嵌入式设备上运行 TensorFlow Lite 模型
from tflite_runtime import interpreter as tflite

# 导入 Ultralytics 自定义模块中的 ASSETS 和 yaml_load 函数
from ultralytics.utils import ASSETS, yaml_load

# 导入 Ultralytics 自定义模块中的 check_yaml 函数，用于检查和处理 YAML 文件
from ultralytics.utils.checks import check_yaml

# 声明全局变量，用于指定训练模型期望的图像宽度和高度
img_width = 640
img_height = 640

class LetterBox:
    """Resizes and reshapes images while maintaining aspect ratio by adding padding, suitable for YOLO models."""

    def __init__(
        self, new_shape=(img_width, img_height), auto=False, scaleFill=False, scaleup=True, center=True, stride=32
    ):
        """
        初始化 LetterBox 对象，配置图像缩放和重塑参数，以保持图像长宽比，并添加填充。

        参数:
        - new_shape: 新图像的目标尺寸 (宽度, 高度)
        - auto: 是否自动处理
        - scaleFill: 是否填充以适应目标尺寸
        - scaleup: 是否放大图像
        - center: 图像放置位置是否居中
        - stride: 网格步长
        """
        self.new_shape = new_shape  # 新图像的目标尺寸
        self.auto = auto  # 是否自动处理
        self.scaleFill = scaleFill  # 是否填充以适应目标尺寸
        self.scaleup = scaleup  # 是否放大图像
        self.stride = stride  # 网格步长
        self.center = center  # 图像放置位置是否居中，默认为居中
    def __call__(self, labels=None, image=None):
        """Return updated labels and image with added border."""
        
        # 如果 labels 为 None，则初始化为空字典
        if labels is None:
            labels = {}
        
        # 如果 image 为 None，则从 labels 中获取 "img" 键对应的图像数据
        img = labels.get("img") if image is None else image
        
        # 获取图像的高度和宽度
        shape = img.shape[:2]  # current shape [height, width]
        
        # 获取要调整的新形状，如果新形状是整数，则转换为元组 (new_shape, new_shape)
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        # 计算缩放比例 (新 / 旧)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # 如果不允许放大 (self.scaleup 为 False)，则限制 r 不超过 1.0，只能缩小图像以获得更好的 mAP 值
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)
        
        # 计算填充量
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
        # 如果设置了 self.auto，则使用最小矩形的方式进行填充
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        
        # 如果设置了 self.scaleFill，则进行拉伸填充
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
        
        # 如果设置了 self.center，则将填充量均分到两侧
        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2
        
        # 如果当前图像的尺寸与新的非填充尺寸不同，则进行缩放
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # 计算上下左右的填充量
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        
        # 使用 cv2.copyMakeBorder() 函数给图像添加边框
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border
        
        # 如果 labels 中存在 "ratio_pad" 键，则更新其值为填充的比例和填充的位置
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation
        
        # 如果 labels 不为空，则调用 _update_labels() 方法更新标签，并返回更新后的 labels
        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels."""
        
        # 将实例的边界框转换为 (x1, y1, x2, y2) 格式
        labels["instances"].convert_bbox(format="xyxy")
        
        # 对边界框进行反归一化处理，以原始图像的高度和宽度为参数
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        
        # 根据比例对边界框进行缩放
        labels["instances"].scale(*ratio)
        
        # 添加填充到边界框
        labels["instances"].add_padding(padw, padh)
        
        # 返回更新后的 labels
        return labels
class Yolov8TFLite:
    """Class for performing object detection using YOLOv8 model converted to TensorFlow Lite format."""

    def __init__(self, tflite_model, input_image, confidence_thres, iou_thres):
        """
        Initializes an instance of the Yolov8TFLite class.

        Args:
            tflite_model: Path to the TFLite model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """

        self.tflite_model = tflite_model  # 保存 TFLite 模型的路径
        self.input_image = input_image    # 保存输入图像的路径
        self.confidence_thres = confidence_thres  # 设定置信度阈值，用于筛选检测结果
        self.iou_thres = iou_thres        # 设定 IoU 阈值，用于非最大抑制

        # 从 COCO 数据集加载类别名称列表
        self.classes = yaml_load(check_yaml("coco8.yaml"))["names"]

        # 为类别生成一个颜色调色板
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # 提取边界框的坐标
        x1, y1, w, h = box

        # 获取类别对应的颜色
        color = self.color_palette[class_id]

        # 在图像上绘制边界框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # 创建包含类别名称和置信度得分的标签文本
        label = f"{self.classes[class_id]}: {score:.2f}"

        # 计算标签文本的尺寸
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # 计算标签文本的位置
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # 在标签文本背景上绘制填充矩形
        cv2.rectangle(
            img,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)),
            color,
            cv2.FILLED,
        )

        # 在图像上绘制标签文本
        cv2.putText(img, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    def preprocess(self):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """

        # Read the input image using OpenCV
        self.img = cv2.imread(self.input_image)

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Initialize a LetterBox object for resizing the image
        letterbox = LetterBox(new_shape=[self.img_width, self.img_height], auto=False, stride=32)
        
        # Resize the input image using the LetterBox object
        image = letterbox(image=self.img)
        image = [image]  # Convert image to list
        image = np.stack(image)  # Stack images along a new axis
        image = image[..., ::-1].transpose((0, 3, 1, 2))  # Rearrange dimensions for model compatibility
        img = np.ascontiguousarray(image)  # Return a contiguous array in memory

        # Convert image data to float32 and normalize
        image = img.astype(np.float32)
        return image / 255  # Return normalized image data for inference

    def postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        boxes = []
        scores = []
        class_ids = []

        # Process each prediction in the model's output
        for pred in output:
            pred = np.transpose(pred)
            for box in pred:
                x, y, w, h = box[:4]
                x1 = x - w / 2
                y1 = y - h / 2
                boxes.append([x1, y1, w, h])  # Store box coordinates
                idx = np.argmax(box[4:])
                scores.append(box[idx + 4])  # Store score
                class_ids.append(idx)  # Store class ID

        # Perform non-maximum suppression to filter out overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # Draw bounding boxes on the input image based on selected indices
        for i in indices:
            box = boxes[i]
            gain = min(self.img_width / self.img_width, self.img_height / self.img_height)
            pad = (
                round((self.img_width - self.img_width * gain) / 2 - 0.1),
                round((self.img_height - self.img_height * gain) / 2 - 0.1),
            )
            box[0] = (box[0] - pad[0]) / gain
            box[1] = (box[1] - pad[1]) / gain
            box[2] = box[2] / gain
            box[3] = box[3] / gain
            score = scores[i]
            class_id = class_ids[i]
            
            if score > 0.25:
                print(box, score, class_id)
                # Draw detections on the input image
                self.draw_detections(input_image, box, score, class_id)

        return input_image
    def main(self):
        """
        Performs inference using a TFLite model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """

        # 创建一个用于 TFLite 模型的解释器
        interpreter = tflite.Interpreter(model_path=self.tflite_model)
        self.model = interpreter  # 将解释器保存到对象属性中
        interpreter.allocate_tensors()  # 分配张量空间

        # 获取模型的输入和输出详情
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # 存储输入形状以备后用
        input_shape = input_details[0]["shape"]
        self.input_width = input_shape[1]  # 保存输入宽度
        self.input_height = input_shape[2]  # 保存输入高度

        # 预处理图像数据
        img_data = self.preprocess()
        img_data = img_data  # 这行代码没有实际作用，保留原样

        # 转置图像数据的通道顺序，以符合模型的要求
        img_data = img_data.transpose((0, 2, 3, 1))

        # 获取输入张量的量化参数，并将图像数据转换为 int8 类型
        scale, zero_point = input_details[0]["quantization"]
        img_data_int8 = (img_data / scale + zero_point).astype(np.int8)
        interpreter.set_tensor(input_details[0]["index"], img_data_int8)

        # 执行推断
        interpreter.invoke()

        # 从解释器获取输出张量
        output = interpreter.get_tensor(output_details[0]["index"])

        # 对输出张量进行反量化操作，以得到原始数值
        scale, zero_point = output_details[0]["quantization"]
        output = (output.astype(np.float32) - zero_point) * scale

        # 将检测框坐标映射回原始图像尺寸
        output[:, [0, 2]] *= img_width
        output[:, [1, 3]] *= img_height

        print(output)  # 打印输出结果（用于调试）

        # 对输出进行后处理，生成带有检测结果的输出图像
        return self.postprocess(self.img, output)
if __name__ == "__main__":
    # 创建参数解析器来处理命令行参数
    parser = argparse.ArgumentParser()
    # 添加命令行参数：模型文件路径，默认为"yolov8n_full_integer_quant.tflite"
    parser.add_argument(
        "--model", type=str, default="yolov8n_full_integer_quant.tflite", help="Input your TFLite model."
    )
    # 添加命令行参数：输入图片路径，默认为 ASSETS 目录下的"bus.jpg"
    parser.add_argument("--img", type=str, default=str(ASSETS / "bus.jpg"), help="Path to input image.")
    # 添加命令行参数：置信度阈值，默认为0.5
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    # 添加命令行参数：NMS IoU 阈值，默认为0.5
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    # 解析命令行参数并存储到 args 中
    args = parser.parse_args()

    # 使用指定的参数实例化 Yolov8TFLite 类
    detection = Yolov8TFLite(args.model, args.img, args.conf_thres, args.iou_thres)

    # 执行物体检测并获取输出图像
    output_image = detection.main()

    # 在窗口中显示输出图像
    cv2.imshow("Output", output_image)

    # 等待按键按下以退出
    cv2.waitKey(0)
```