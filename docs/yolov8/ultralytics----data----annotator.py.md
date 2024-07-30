# `.\yolov8\ultralytics\data\annotator.py`

```py
# 导入必要的模块和类
from pathlib import Path
from ultralytics import SAM, YOLO

# 定义自动标注函数，用于使用 YOLO 目标检测模型和 SAM 分割模型自动标注图像
def auto_annotate(data, det_model="yolov8x.pt", sam_model="sam_b.pt", device="", output_dir=None):
    """
    Automatically annotates images using a YOLO object detection model and a SAM segmentation model.

    This function processes images in a specified directory, detects objects using a YOLO model, and then generates
    segmentation masks using a SAM model. The resulting annotations are saved as text files.

    Args:
        data (str): Path to a folder containing images to be annotated.
        det_model (str): Path or name of the pre-trained YOLO detection model.
        sam_model (str): Path or name of the pre-trained SAM segmentation model.
        device (str): Device to run the models on (e.g., 'cpu', 'cuda', '0').
        output_dir (str | None): Directory to save the annotated results. If None, a default directory is created.

    Examples:
        >>> from ultralytics.data.annotator import auto_annotate
        >>> auto_annotate(data='ultralytics/assets', det_model='yolov8n.pt', sam_model='mobile_sam.pt')

    Notes:
        - The function creates a new directory for output if not specified.
        - Annotation results are saved as text files with the same names as the input images.
        - Each line in the output text file represents a detected object with its class ID and segmentation points.
    """
    # 创建 YOLO 检测模型对象
    det_model = YOLO(det_model)
    # 创建 SAM 分割模型对象
    sam_model = SAM(sam_model)

    # 将输入的数据路径转换为 pathlib.Path 对象
    data = Path(data)
    # 如果未指定输出目录，则创建一个默认的输出目录，命名规则为原始数据文件夹名 + '_auto_annotate_labels'
    if not output_dir:
        output_dir = data.parent / f"{data.stem}_auto_annotate_labels"
    # 确保输出目录存在，如果不存在则创建
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # 使用 YOLO 模型处理数据，返回检测结果
    det_results = det_model(data, stream=True, device=device)

    # 遍历检测结果
    for result in det_results:
        # 获取检测到的类别 ID 列表
        class_ids = result.boxes.cls.int().tolist()  # noqa
        # 如果有检测结果
        if len(class_ids):
            # 获取检测到的边界框信息
            boxes = result.boxes.xyxy  # Boxes object for bbox outputs
            # 使用 SAM 模型生成分割结果，不输出详细信息，不保存结果，使用指定设备
            sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False, device=device)
            # 获取分割结果中的分割掩码
            segments = sam_results[0].masks.xyn  # noqa

            # 针对每个检测到的对象，将结果写入对应的文本文件
            with open(f"{Path(output_dir) / Path(result.path).stem}.txt", "w") as f:
                for i in range(len(segments)):
                    s = segments[i]
                    # 如果分割结果为空，则跳过
                    if len(s) == 0:
                        continue
                    # 将分割掩码转换为字符串形式，并写入文件
                    segment = map(str, segments[i].reshape(-1).tolist())
                    f.write(f"{class_ids[i]} " + " ".join(segment) + "\n")
```