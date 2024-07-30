# `.\yolov8\examples\YOLOv8-SAHI-Inference-Video\yolov8_sahi.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

import argparse             # 导入命令行参数解析模块
from pathlib import Path    # 导入处理路径的模块

import cv2                  # 导入OpenCV图像处理库
from sahi import AutoDetectionModel  # 导入SAHI自动检测模型
from sahi.predict import get_sliced_prediction  # 导入预测函数
from sahi.utils.yolov8 import download_yolov8s_model  # 导入YOLOv8模型下载函数

from ultralytics.utils.files import increment_path  # 导入路径增加函数


def run(weights="yolov8n.pt", source="test.mp4", view_img=False, save_img=False, exist_ok=False):
    """
    Run object detection on a video using YOLOv8 and SAHI.

    Args:
        weights (str): Model weights path.
        source (str): Video file path.
        view_img (bool): Show results.
        save_img (bool): Save results.
        exist_ok (bool): Overwrite existing files.
    """

    # Check source path
    if not Path(source).exists():  # 检查视频文件路径是否存在，若不存在则抛出文件未找到异常
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    yolov8_model_path = f"models/{weights}"  # 设置YOLOv8模型的路径
    download_yolov8s_model(yolov8_model_path)  # 下载YOLOv8模型到指定路径
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=yolov8_model_path, confidence_threshold=0.3, device="cpu"
    )  # 使用SAHI加载预训练的YOLOv8模型，设定置信度阈值和使用CPU设备

    # Video setup
    videocapture = cv2.VideoCapture(source)  # 使用OpenCV打开视频文件
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))  # 获取视频帧宽度和高度
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")  # 获取视频帧率和视频编解码器格式

    # Output setup
    save_dir = increment_path(Path("ultralytics_results_with_sahi") / "exp", exist_ok)  # 使用增加路径函数创建结果保存目录
    save_dir.mkdir(parents=True, exist_ok=True)  # 创建保存目录，若不存在则递归创建
    video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width, frame_height))  # 设置视频写入对象，指定输出路径、帧率和尺寸
    # 循环直到视频捕获对象不再打开
    while videocapture.isOpened():
        # 从视频捕获对象中读取一帧图像
        success, frame = videocapture.read()
        # 如果读取失败，则跳出循环
        if not success:
            break

        # 使用模型对图像进行分块预测
        results = get_sliced_prediction(
            frame, detection_model, slice_height=512, slice_width=512, overlap_height_ratio=0.2, overlap_width_ratio=0.2
        )
        object_prediction_list = results.object_prediction_list

        # 初始化用于存储边界框和类别的列表
        boxes_list = []
        clss_list = []

        # 遍历检测到的对象预测列表
        for ind, _ in enumerate(object_prediction_list):
            # 获取边界框的坐标信息
            boxes = (
                object_prediction_list[ind].bbox.minx,
                object_prediction_list[ind].bbox.miny,
                object_prediction_list[ind].bbox.maxx,
                object_prediction_list[ind].bbox.maxy,
            )
            # 获取对象类别名称
            clss = object_prediction_list[ind].category.name
            # 将边界框坐标和类别名称添加到相应的列表中
            boxes_list.append(boxes)
            clss_list.append(clss)

        # 遍历边界框列表和类别列表，并在图像上绘制边界框和标签
        for box, cls in zip(boxes_list, clss_list):
            x1, y1, x2, y2 = box
            # 绘制边界框
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (56, 56, 255), 2)
            label = str(cls)
            t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
            # 绘制标签背景
            cv2.rectangle(
                frame, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1) + 3), (56, 56, 255), -1
            )
            # 绘制标签文本
            cv2.putText(
                frame, label, (int(x1), int(y1) - 2), 0, 0.6, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA
            )

        # 如果需要显示图像，则显示当前帧图像
        if view_img:
            cv2.imshow(Path(source).stem, frame)
        
        # 如果需要保存图像，则将当前帧图像写入视频文件
        if save_img:
            video_writer.write(frame)

        # 检测用户是否按下 'q' 键，如果是则跳出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 释放视频写入对象和视频捕获对象
    video_writer.release()
    videocapture.release()
    # 关闭所有的 OpenCV 窗口
    cv2.destroyAllWindows()
# 解析命令行参数的函数
def parse_opt():
    """Parse command line arguments."""
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加参数选项：--weights，类型为字符串，默认值为"yolov8n.pt"，用于指定初始权重路径
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="initial weights path")
    # 添加参数选项：--source，类型为字符串，必须指定，用于指定视频文件路径
    parser.add_argument("--source", type=str, required=True, help="video file path")
    # 添加参数选项：--view-img，若存在则设置为 True，用于显示结果
    parser.add_argument("--view-img", action="store_true", help="show results")
    # 添加参数选项：--save-img，若存在则设置为 True，用于保存结果
    parser.add_argument("--save-img", action="store_true", help="save results")
    # 添加参数选项：--exist-ok，若存在则设置为 True，用于指示项目/名称已存在时不递增
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    # 解析命令行参数并返回
    return parser.parse_args()


# 主函数入口
def main(opt):
    """Main function."""
    # 调用 run 函数，传入 opt 参数的所有变量作为关键字参数
    run(**vars(opt))


# 当作为脚本直接执行时的入口
if __name__ == "__main__":
    # 解析命令行参数并存储在 opt 变量中
    opt = parse_opt()
    # 调用主函数，传入解析后的命令行参数 opt
    main(opt)
```