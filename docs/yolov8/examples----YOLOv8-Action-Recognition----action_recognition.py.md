# `.\yolov8\examples\YOLOv8-Action-Recognition\action_recognition.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

import argparse  # 导入命令行参数解析模块
import time  # 导入时间模块
from collections import defaultdict  # 导入默认字典模块
from typing import List, Optional, Tuple  # 导入类型提示相关模块
from urllib.parse import urlparse  # 导入 URL 解析模块

import cv2  # 导入 OpenCV 图像处理库
import numpy as np  # 导入 NumPy 数学计算库
import torch  # 导入 PyTorch 深度学习库
from transformers import AutoModel, AutoProcessor  # 导入 Hugging Face Transformers 模块

from ultralytics import YOLO  # 导入 Ultralytics YOLO 目标检测模块
from ultralytics.data.loaders import get_best_youtube_url  # 导入获取最佳 YouTube URL 的函数
from ultralytics.utils.plotting import Annotator  # 导入图像标注工具类
from ultralytics.utils.torch_utils import select_device  # 导入选择设备的工具函数

class TorchVisionVideoClassifier:
    """Classifies videos using pretrained TorchVision models; see https://pytorch.org/vision/stable/."""

    from torchvision.models.video import (
        MViT_V1_B_Weights,
        MViT_V2_S_Weights,
        R3D_18_Weights,
        S3D_Weights,
        Swin3D_B_Weights,
        Swin3D_T_Weights,
        mvit_v1_b,
        mvit_v2_s,
        r3d_18,
        s3d,
        swin3d_b,
        swin3d_t,
    )

    model_name_to_model_and_weights = {
        "s3d": (s3d, S3D_Weights.DEFAULT),
        "r3d_18": (r3d_18, R3D_18_Weights.DEFAULT),
        "swin3d_t": (swin3d_t, Swin3D_T_Weights.DEFAULT),
        "swin3d_b": (swin3d_b, Swin3D_B_Weights.DEFAULT),
        "mvit_v1_b": (mvit_v1_b, MViT_V1_B_Weights.DEFAULT),
        "mvit_v2_s": (mvit_v2_s, MViT_V2_S_Weights.DEFAULT),
    }

    def __init__(self, model_name: str, device: str or torch.device = ""):
        """
        Initialize the VideoClassifier with the specified model name and device.

        Args:
            model_name (str): The name of the model to use.
            device (str or torch.device, optional): The device to run the model on. Defaults to "".

        Raises:
            ValueError: If an invalid model name is provided.
        """
        if model_name not in self.model_name_to_model_and_weights:
            raise ValueError(f"Invalid model name '{model_name}'. Available models: {self.available_model_names()}")
        model, self.weights = self.model_name_to_model_and_weights[model_name]
        self.device = select_device(device)  # 选择设备（GPU 或 CPU）
        self.model = model(weights=self.weights).to(self.device).eval()  # 初始化模型并将其移动到指定设备

    @staticmethod
    def available_model_names() -> List[str]:
        """
        Get the list of available model names.

        Returns:
            list: List of available model names.
        """
        return list(TorchVisionVideoClassifier.model_name_to_model_and_weights.keys())
    # 对视频分类任务中的一组裁剪图像进行预处理
    def preprocess_crops_for_video_cls(self, crops: List[np.ndarray], input_size: list = None) -> torch.Tensor:
        """
        Preprocess a list of crops for video classification.

        Args:
            crops (List[np.ndarray]): List of crops to preprocess. Each crop should have dimensions (H, W, C)
            input_size (tuple, optional): The target input size for the model. Defaults to (224, 224).

        Returns:
            torch.Tensor: Preprocessed crops as a tensor with dimensions (1, T, C, H, W).
        """
        # 如果未提供输入大小，则默认为 (224, 224)
        if input_size is None:
            input_size = [224, 224]
        # 导入 torchvision.transforms.v2 模块，并创建变换序列 transform
        from torchvision.transforms import v2

        transform = v2.Compose(
            [
                # 将图像数据类型转换为 float32，并进行尺度缩放
                v2.ToDtype(torch.float32, scale=True),
                # 调整图像大小到指定的 input_size，使用抗锯齿方法
                v2.Resize(input_size, antialias=True),
                # 根据预先定义的均值和标准差进行图像归一化
                v2.Normalize(mean=self.weights.transforms().mean, std=self.weights.transforms().std),
            ]
        )

        # 对每个裁剪图像应用 transform 变换，转换为张量并重新排列维度
        processed_crops = [transform(torch.from_numpy(crop).permute(2, 0, 1)) for crop in crops]
        # 将处理后的裁剪图像堆叠成一个张量，添加批次维度，重新排列维度以适应模型输入格式，并将结果移动到指定设备上
        return torch.stack(processed_crops).unsqueeze(0).permute(0, 2, 1, 3, 4).to(self.device)

    # 调用对象作为函数时执行的方法，用于在给定序列上进行推断
    def __call__(self, sequences: torch.Tensor):
        """
        Perform inference on the given sequences.

        Args:
            sequences (torch.Tensor): The input sequences for the model. The expected input dimensions are
                                      (B, T, C, H, W) for batched video frames or (T, C, H, W) for single video frames.

        Returns:
            torch.Tensor: The model's output.
        """
        # 进入推断模式，确保不进行梯度计算
        with torch.inference_mode():
            # 调用模型进行推断，返回模型的输出结果
            return self.model(sequences)

    # 对模型的输出进行后处理，得到预测的类别标签和置信度
    def postprocess(self, outputs: torch.Tensor) -> Tuple[List[str], List[float]]:
        """
        Postprocess the model's batch output.

        Args:
            outputs (torch.Tensor): The model's output.

        Returns:
            List[str]: The predicted labels.
            List[float]: The predicted confidences.
        """
        # 初始化预测标签列表和置信度列表
        pred_labels = []
        pred_confs = []
        # 遍历模型输出的每个样本
        for output in outputs:
            # 找到输出张量中最高置信度的类别索引
            pred_class = output.argmax(0).item()
            # 根据索引从预先定义的类别字典中获取预测标签
            pred_label = self.weights.meta["categories"][pred_class]
            # 将预测标签添加到列表中
            pred_labels.append(pred_label)
            # 计算并获取该类别的置信度值
            pred_conf = output.softmax(0)[pred_class].item()
            # 将置信度值添加到列表中
            pred_confs.append(pred_conf)

        # 返回预测标签列表和置信度列表作为元组
        return pred_labels, pred_confs
# 定义一个视频分类器类，使用 Hugging Face 模型进行零样本分类，适用于多种设备
class HuggingFaceVideoClassifier:
    """Zero-shot video classifier using Hugging Face models for various devices."""

    def __init__(
        self,
        labels: List[str],
        model_name: str = "microsoft/xclip-base-patch16-zero-shot",
        device: str or torch.device = "",
        fp16: bool = False,
    ):
        """
        Initialize the HuggingFaceVideoClassifier with the specified model name.

        Args:
            labels (List[str]): List of labels for zero-shot classification.
            model_name (str): The name of the model to use. Defaults to "microsoft/xclip-base-patch16-zero-shot".
            device (str or torch.device, optional): The device to run the model on. Defaults to "".
            fp16 (bool, optional): Whether to use FP16 for inference. Defaults to False.
        """
        # 设置是否使用 FP16 进行推断
        self.fp16 = fp16
        # 存储分类器的标签列表
        self.labels = labels
        # 选择设备并将其分配给 self.device
        self.device = select_device(device)
        # 从预训练模型名称加载处理器
        self.processor = AutoProcessor.from_pretrained(model_name)
        # 加载预训练模型并将其移至所选设备
        model = AutoModel.from_pretrained(model_name).to(self.device)
        # 如果使用 FP16，则将模型转换为 FP16 格式
        if fp16:
            model = model.half()
        # 将模型设置为评估模式
        self.model = model.eval()

    def preprocess_crops_for_video_cls(self, crops: List[np.ndarray], input_size: list = None) -> torch.Tensor:
        """
        Preprocess a list of crops for video classification.

        Args:
            crops (List[np.ndarray]): List of crops to preprocess. Each crop should have dimensions (H, W, C)
            input_size (tuple, optional): The target input size for the model. Defaults to (224, 224).

        Returns:
            torch.Tensor: Preprocessed crops as a tensor (1, T, C, H, W).
        """
        # 如果未提供输入尺寸，则默认为 (224, 224)
        if input_size is None:
            input_size = [224, 224]
        # 导入 torchvision 中的 transforms 模块
        from torchvision import transforms

        # 定义图像预处理管道
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.float() / 255.0),  # 将像素值缩放到 [0, 1]
                transforms.Resize(input_size),  # 调整图像大小至指定尺寸
                transforms.Normalize(
                    mean=self.processor.image_processor.image_mean,  # 根据处理器定义的均值进行归一化
                    std=self.processor.image_processor.image_std  # 根据处理器定义的标准差进行归一化
                ),
            ]
        )

        # 对输入的每个 crop 进行预处理
        processed_crops = [transform(torch.from_numpy(crop).permute(2, 0, 1)) for crop in crops]  # (T, C, H, W)
        # 将预处理后的 crop 堆叠成一个张量，并在最前面增加一个维度表示批处理
        output = torch.stack(processed_crops).unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        # 如果使用 FP16，则将输出张量转换为 FP16 格式
        if self.fp16:
            output = output.half()
        return output
    # 定义一个方法，使对象可以像函数一样被调用，执行推断操作
    def __call__(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Perform inference on the given sequences.

        Args:
            sequences (torch.Tensor): The input sequences for the model. Batched video frames with shape (B, T, H, W, C).

        Returns:
            torch.Tensor: The model's output.
        """

        # 使用处理器（processor）处理标签，返回包含输入ids的PyTorch张量，填充数据为True
        input_ids = self.processor(text=self.labels, return_tensors="pt", padding=True)["input_ids"].to(self.device)

        # 构建输入字典，包含像素值（sequences）和输入ids（input_ids）
        inputs = {"pixel_values": sequences, "input_ids": input_ids}

        # 进入推断模式
        with torch.inference_mode():
            # 使用模型进行推断，传入inputs字典作为参数
            outputs = self.model(**inputs)

        # 返回模型输出中的logits_per_video
        return outputs.logits_per_video

    # 定义一个方法，用于后处理模型的批量输出
    def postprocess(self, outputs: torch.Tensor) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Postprocess the model's batch output.

        Args:
            outputs (torch.Tensor): The model's output.

        Returns:
            List[List[str]]: The predicted top3 labels.
            List[List[float]]: The predicted top3 confidences.
        """
        # 初始化预测标签和置信度列表
        pred_labels = []
        pred_confs = []

        # 使用torch.no_grad()上下文管理器，关闭梯度计算
        with torch.no_grad():
            # 假设outputs已经是logits张量
            logits_per_video = outputs

            # 对logits进行softmax操作，将其转换为概率
            probs = logits_per_video.softmax(dim=-1)

        # 遍历每个视频的概率分布
        for prob in probs:
            # 获取概率最高的两个索引
            top2_indices = prob.topk(2).indices.tolist()

            # 根据索引获取对应的标签和置信度，并转换为列表形式
            top2_labels = [self.labels[idx] for idx in top2_indices]
            top2_confs = prob[top2_indices].tolist()

            # 将预测的top2标签和置信度添加到对应的列表中
            pred_labels.append(top2_labels)
            pred_confs.append(top2_confs)

        # 返回预测的top3标签列表和置信度列表
        return pred_labels, pred_confs
# 初始化裁剪并填充函数，用于从视频帧中裁剪指定区域并添加边距，返回尺寸为 224x224 的裁剪图像
def crop_and_pad(frame, box, margin_percent):
    """Crop box with margin and take square crop from frame."""
    # 解析框框的坐标
    x1, y1, x2, y2 = map(int, box)
    # 计算框框的宽度和高度
    w, h = x2 - x1, y2 - y1

    # 添加边距
    margin_x, margin_y = int(w * margin_percent / 100), int(h * margin_percent / 100)
    # 调整框框的位置，确保不超出图像边界
    x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
    x2, y2 = min(frame.shape[1], x2 + margin_x), min(frame.shape[0], y2 + margin_y)

    # 从图像中心获取正方形裁剪
    size = max(y2 - y1, x2 - x1)
    center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
    half_size = size // 2
    # 裁剪出正方形区域
    square_crop = frame[
        max(0, center_y - half_size) : min(frame.shape[0], center_y + half_size),
        max(0, center_x - half_size) : min(frame.shape[1], center_x + half_size),
    ]

    # 将裁剪的图像大小调整为 224x224 像素
    return cv2.resize(square_crop, (224, 224), interpolation=cv2.INTER_LINEAR)


def run(
    weights: str = "yolov8n.pt",
    device: str = "",
    source: str = "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    output_path: Optional[str] = None,
    crop_margin_percentage: int = 10,
    num_video_sequence_samples: int = 8,
    skip_frame: int = 2,
    video_cls_overlap_ratio: float = 0.25,
    fp16: bool = False,
    video_classifier_model: str = "microsoft/xclip-base-patch32",
    labels: List[str] = None,
) -> None:
    """
    Run action recognition on a video source using YOLO for object detection and a video classifier.

    Args:
        weights (str): Path to the YOLO model weights. Defaults to "yolov8n.pt".
        device (str): Device to run the model on. Use 'cuda' for NVIDIA GPU, 'mps' for Apple Silicon, or 'cpu'. Defaults to auto-detection.
        source (str): Path to mp4 video file or YouTube URL. Defaults to a sample YouTube video.
        output_path (Optional[str], optional): Path to save the output video. Defaults to None.
        crop_margin_percentage (int, optional): Percentage of margin to add around detected objects. Defaults to 10.
        num_video_sequence_samples (int, optional): Number of video frames to use for classification. Defaults to 8.
        skip_frame (int, optional): Number of frames to skip between detections. Defaults to 4.
        video_cls_overlap_ratio (float, optional): Overlap ratio between video sequences. Defaults to 0.25.
        fp16 (bool, optional): Whether to use half-precision floating point. Defaults to False.
        video_classifier_model (str, optional): Name or path of the video classifier model. Defaults to "microsoft/xclip-base-patch32".
        labels (List[str], optional): List of labels for zero-shot classification. Defaults to predefined list.

    Returns:
        None
    """
    # 如果标签列表为空，使用预定义的动作标签
    if labels is None:
        labels = [
            "walking",
            "running",
            "brushing teeth",
            "looking into phone",
            "weight lifting",
            "cooking",
            "sitting",
        ]
    
    # 初始化模型和设备
    device = select_device(device)  # 选择运行的设备
    yolo_model = YOLO(weights).to(device)  # 加载并移动 YOLO 模型到指定设备
    # 如果视频分类模型在 TorchVisionVideoClassifier 可用模型列表中
    if video_classifier_model in TorchVisionVideoClassifier.available_model_names():
        # 打印警告信息，指出 'fp16' 不支持 TorchVisionVideoClassifier，将其设置为 False
        print("'fp16' is not supported for TorchVisionVideoClassifier. Setting fp16 to False.")
        # 打印警告信息，指出 'labels' 在 TorchVisionVideoClassifier 中不使用，忽略提供的标签并使用 Kinetics-400 标签
        print(
            "'labels' is not used for TorchVisionVideoClassifier. Ignoring the provided labels and using Kinetics-400 labels."
        )
        # 使用 TorchVisionVideoClassifier 初始化视频分类器对象，设备为给定设备
        video_classifier = TorchVisionVideoClassifier(video_classifier_model, device=device)
    else:
        # 使用 HuggingFaceVideoClassifier 初始化视频分类器对象
        video_classifier = HuggingFaceVideoClassifier(
            labels, model_name=video_classifier_model, device=device, fp16=fp16
        )
    
    # 初始化视频捕获对象
    # 如果源地址以 "http" 开头且主机名是 YouTube 相关的地址，则获取最佳的 YouTube 视频地址
    if source.startswith("http") and urlparse(source).hostname in {"www.youtube.com", "youtube.com", "youtu.be"}:
        source = get_best_youtube_url(source)
    # 否则，如果源地址不是以 ".mp4" 结尾，则抛出值错误异常
    elif not source.endswith(".mp4"):
        raise ValueError("Invalid source. Supported sources are YouTube URLs and MP4 files.")
    # 使用 OpenCV 打开视频捕获对象
    cap = cv2.VideoCapture(source)
    
    # 获取视频的属性信息：帧宽度、帧高度、帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 如果指定了输出路径，则初始化视频写入对象
    if output_path is not None:
        # 使用 mp4v 编解码器创建视频写入对象
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # 初始化跟踪历史字典和帧计数器
    track_history = defaultdict(list)
    frame_counter = 0
    
    # 初始化需要推断的跟踪 ID、需要推断的裁剪图像、预测标签和置信度列表
    track_ids_to_infer = []
    crops_to_infer = []
    pred_labels = []
    pred_confs = []
    
    # 释放视频捕获对象
    cap.release()
    
    # 如果指定了输出路径，则释放视频写入对象
    if output_path is not None:
        out.release()
    
    # 关闭所有 OpenCV 窗口
    cv2.destroyAllWindows()
# 解析命令行参数的函数
def parse_opt():
    """Parse command line arguments."""
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加一个参数选项：权重文件的路径，默认为"yolov8n.pt"
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="ultralytics detector model path")
    # 添加一个参数选项：设备类型，默认为空字符串，支持 cuda 设备（如 '0' 或 '0,1,2,3'）、cpu 或 mps，空字符串表示自动检测
    parser.add_argument("--device", default="", help='cuda device, i.e. 0 or 0,1,2,3 or cpu/mps, "" for auto-detection')
    # 添加一个参数选项：视频文件路径或 YouTube URL，默认为 Rick Astley 的视频链接
    parser.add_argument(
        "--source",
        type=str,
        default="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        help="video file path or youtube URL",
    )
    # 添加一个参数选项：输出视频文件路径，默认为"output_video.mp4"
    parser.add_argument("--output-path", type=str, default="output_video.mp4", help="output video file path")
    # 添加一个参数选项：检测到的对象周围添加的裁剪边距百分比，默认为10%
    parser.add_argument(
        "--crop-margin-percentage", type=int, default=10, help="percentage of margin to add around detected objects"
    )
    # 添加一个参数选项：用于分类的视频帧样本数量，默认为8帧
    parser.add_argument(
        "--num-video-sequence-samples", type=int, default=8, help="number of video frames to use for classification"
    )
    # 添加一个参数选项：在检测之间跳过的帧数，默认为2帧
    parser.add_argument("--skip-frame", type=int, default=2, help="number of frames to skip between detections")
    # 添加一个参数选项：视频序列之间的重叠比率，默认为0.25
    parser.add_argument(
        "--video-cls-overlap-ratio", type=float, default=0.25, help="overlap ratio between video sequences"
    )
    # 添加一个参数选项：是否使用 FP16 进行推断，默认为 False
    parser.add_argument("--fp16", action="store_true", help="use FP16 for inference")
    # 添加一个参数选项：视频分类器模型的名称，默认为"microsoft/xclip-base-patch32"
    parser.add_argument(
        "--video-classifier-model", type=str, default="microsoft/xclip-base-patch32", help="video classifier model name"
    )
    # 添加一个参数选项：用于零样本视频分类的标签列表，默认为["dancing", "singing a song"]
    parser.add_argument(
        "--labels",
        nargs="+",
        type=str,
        default=["dancing", "singing a song"],
        help="labels for zero-shot video classification",
    )
    # 解析命令行参数并返回结果
    return parser.parse_args()


# 主函数，运行时接受一个参数 opt
def main(opt):
    """Main function."""
    # 将 opt 解包后作为关键字参数传递给 run 函数
    run(**vars(opt))


# 如果当前脚本作为主程序运行，则执行以下代码
if __name__ == "__main__":
    # 解析命令行参数并赋值给 opt
    opt = parse_opt()
    # 调用主函数，传入解析后的参数 opt
    main(opt)
```