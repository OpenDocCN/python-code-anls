# `.\yolov8\ultralytics\data\converter.py`

```py
# 导入必要的库和模块
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# 导入 Ultralytics 自定义的日志记录和进度条显示工具
from ultralytics.utils import LOGGER, TQDM
# 导入 Ultralytics 自定义的文件处理工具中的路径增量函数
from ultralytics.utils.files import increment_path

# 将 COCO 91 类别映射到 COCO 80 类别的函数
def coco91_to_coco80_class():
    """
    Converts 91-index COCO class IDs to 80-index COCO class IDs.

    Returns:
        (list): A list of 91 class IDs where the index represents the 80-index class ID and the value is the
            corresponding 91-index class ID.
    """
    return [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        None,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        None,
        24,
        25,
        None,
        None,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        None,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        None,
        60,
        None,
        None,
        61,
        None,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        None,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        None,
    ]


# 将 COCO 80 类别映射到 COCO 91 类别的函数
def coco80_to_coco91_class():
    """
    Converts 80-index (val2014) to 91-index (paper).
    For details see https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/.

    Example:
        ```python
        import numpy as np

        a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
        b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
        x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
        x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
        ```py
    """
    # 返回一个包含指定整数的列表
    return [
        1,    # 第一个整数
        2,    # 第二个整数
        3,    # 第三个整数
        4,    # 第四个整数
        5,    # 第五个整数
        6,    # 第六个整数
        7,    # 第七个整数
        8,    # 第八个整数
        9,    # 第九个整数
        10,   # 第十个整数
        11,   # 第十一个整数
        13,   # 第十二个整数（注意此处应为第十三个整数，实际上有一个数字被跳过）
        14,   # 第十四个整数
        15,   # 第十五个整数
        16,   # 第十六个整数
        17,   # 第十七个整数
        18,   # 第十八个整数
        19,   # 第十九个整数
        20,   # 第二十个整数
        21,   # 第二十一个整数
        22,   # 第二十二个整数
        23,   # 第二十三个整数
        24,   # 第二十四个整数
        25,   # 第二十五个整数
        27,   # 第二十六个整数
        28,   # 第二十七个整数
        31,   # 第二十八个整数
        32,   # 第二十九个整数
        33,   # 第三十个整数
        34,   # 第三十一个整数
        35,   # 第三十二个整数
        36,   # 第三十三个整数
        37,   # 第三十四个整数
        38,   # 第三十五个整数
        39,   # 第三十六个整数
        40,   # 第三十七个整数
        41,   # 第三十八个整数
        42,   # 第三十九个整数
        43,   # 第四十个整数
        44,   # 第四十一个整数
        46,   # 第四十二个整数
        47,   # 第四十三个整数
        48,   # 第四十四个整数
        49,   # 第四十五个整数
        50,   # 第四十六个整数
        51,   # 第四十七个整数
        52,   # 第四十八个整数
        53,   # 第四十九个整数
        54,   # 第五十个整数
        55,   # 第五十一个整数
        56,   # 第五十二个整数
        57,   # 第五十三个整数
        58,   # 第五十四个整数
        59,   # 第五十五个整数
        60,   # 第五十六个整数
        61,   # 第五十七个整数
        62,   # 第五十八个整数
        63,   # 第五十九个整数
        64,   # 第六十个整数
        65,   # 第六十一个整数
        67,   # 第六十二个整数
        70,   # 第六十三个整数
        72,   # 第六十四个整数
        73,   # 第六十五个整数
        74,   # 第六十六个整数
        75,   # 第六十七个整数
        76,   # 第六十八个整数
        77,   # 第六十九个整数
        78,   # 第七十个整数
        79,   # 第七十一个整数
        80,   # 第七十二个整数
        81,   # 第七十三个整数
        82,   # 第七十四个整数
        84,   # 第七十五个整数
        85,   # 第七十六个整数
        86,   # 第七十七个整数
        87,   # 第七十八个整数
        88,   # 第七十九个整数
        89,   # 第八十个整数
        90,   # 第八十一个整数
    ]
def convert_coco(
    labels_dir="../coco/annotations/",
    save_dir="coco_converted/",
    use_segments=False,
    use_keypoints=False,
    cls91to80=True,
    lvis=False,
):
    """
    Converts COCO dataset annotations to a YOLO annotation format suitable for training YOLO models.

    Args:
        labels_dir (str, optional): Path to directory containing COCO dataset annotation files.
        save_dir (str, optional): Path to directory to save results to.
        use_segments (bool, optional): Whether to include segmentation masks in the output.
        use_keypoints (bool, optional): Whether to include keypoint annotations in the output.
        cls91to80 (bool, optional): Whether to map 91 COCO class IDs to the corresponding 80 COCO class IDs.
        lvis (bool, optional): Whether to convert data in lvis dataset way.

    Example:
        ```python
        from ultralytics.data.converter import convert_coco

        convert_coco('../datasets/coco/annotations/', use_segments=True, use_keypoints=False, cls91to80=True)
        convert_coco('../datasets/lvis/annotations/', use_segments=True, use_keypoints=False, cls91to80=False, lvis=True)
        ```py

    Output:
        Generates output files in the specified output directory.
    """

    # Create dataset directory
    save_dir = increment_path(save_dir)  # 如果保存目录已存在，则增加路径编号
    for p in save_dir / "labels", save_dir / "images":
        p.mkdir(parents=True, exist_ok=True)  # 创建目录

    # Convert classes
    coco80 = coco91_to_coco80_class()  # 转换 COCO 数据集的 91 类别到 80 类别

    # Import json
    LOGGER.info(f"{'LVIS' if lvis else 'COCO'} data converted successfully.\nResults saved to {save_dir.resolve()}")

def convert_dota_to_yolo_obb(dota_root_path: str):
    """
    Converts DOTA dataset annotations to YOLO OBB (Oriented Bounding Box) format.

    The function processes images in the 'train' and 'val' folders of the DOTA dataset. For each image, it reads the
    associated label from the original labels directory and writes new labels in YOLO OBB format to a new directory.

    Args:
        dota_root_path (str): The root directory path of the DOTA dataset.

    Example:
        ```python
        from ultralytics.data.converter import convert_dota_to_yolo_obb

        convert_dota_to_yolo_obb('path/to/DOTA')
        ```py

    Notes:
        The directory structure assumed for the DOTA dataset:

            - DOTA
                ├─ images
                │   ├─ train
                │   └─ val
                └─ labels
                    ├─ train_original
                    └─ val_original

        After execution, the function will organize the labels into:

            - DOTA
                └─ labels
                    ├─ train
                    └─ val
    """
    dota_root_path = Path(dota_root_path)

    # Class names to indices mapping
    # 定义一个类别映射字典，将字符串类别映射到整数编码
    class_mapping = {
        "plane": 0,
        "ship": 1,
        "storage-tank": 2,
        "baseball-diamond": 3,
        "tennis-court": 4,
        "basketball-court": 5,
        "ground-track-field": 6,
        "harbor": 7,
        "bridge": 8,
        "large-vehicle": 9,
        "small-vehicle": 10,
        "helicopter": 11,
        "roundabout": 12,
        "soccer-ball-field": 13,
        "swimming-pool": 14,
        "container-crane": 15,
        "airport": 16,
        "helipad": 17,
    }
    
    def convert_label(image_name, image_width, image_height, orig_label_dir, save_dir):
        """将单个图片的DOTA标注转换为YOLO OBB格式，并保存到指定目录。"""
        # 构建原始标签文件路径和保存路径
        orig_label_path = orig_label_dir / f"{image_name}.txt"
        save_path = save_dir / f"{image_name}.txt"
    
        # 使用原始标签文件进行读取，保存转换后的标签
        with orig_label_path.open("r") as f, save_path.open("w") as g:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                # 提取类别名称并映射到整数编码
                class_name = parts[8]
                class_idx = class_mapping[class_name]
                # 提取坐标信息并进行归一化
                coords = [float(p) for p in parts[:8]]
                normalized_coords = [
                    coords[i] / image_width if i % 2 == 0 else coords[i] / image_height for i in range(8)
                ]
                # 格式化坐标信息，保留小数点后六位
                formatted_coords = ["{:.6g}".format(coord) for coord in normalized_coords]
                # 写入转换后的标签信息到文件中
                g.write(f"{class_idx} {' '.join(formatted_coords)}\n")
    
    # 对训练集和验证集两个阶段进行循环处理
    for phase in ["train", "val"]:
        # 构建图片路径、原始标签路径和保存标签的路径
        image_dir = dota_root_path / "images" / phase
        orig_label_dir = dota_root_path / "labels" / f"{phase}_original"
        save_dir = dota_root_path / "labels" / phase
    
        # 如果保存标签的目录不存在，则创建
        save_dir.mkdir(parents=True, exist_ok=True)
    
        # 获取当前阶段图片的路径列表，并对每张图片进行处理
        image_paths = list(image_dir.iterdir())
        for image_path in TQDM(image_paths, desc=f"Processing {phase} images"):
            # 如果图片不是PNG格式则跳过
            if image_path.suffix != ".png":
                continue
            # 获取图片名称（不含扩展名）、读取图片并获取其高度和宽度
            image_name_without_ext = image_path.stem
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            # 调用函数将标签进行转换并保存到指定目录
            convert_label(image_name_without_ext, w, h, orig_label_dir, save_dir)
# 将 YOLO 格式的边界框数据转换为分割数据或方向边界框（OBB）数据
# 生成分割数据时可能使用 SAM 自动标注器
def yolo_bbox2segment(im_dir, save_dir=None, sam_model="sam_b.pt"):
    # 读取 SAM 模型的路径
    """
    Args:
        im_dir (str): 图像文件夹的路径，包含待处理的图像
        save_dir (str, optional): 结果保存的文件夹路径，默认为 None
        sam_model (str, optional): SAM 自动标注器的模型文件名，默认为 "sam_b.pt"

    Returns:
        s (List[np.ndarray]): 连接后的分割数据列表，每个元素为 NumPy 数组
    """
    Args:
        im_dir (str | Path): 要转换的图像目录的路径。
        save_dir (str | Path): 生成标签的保存路径，如果为None，则保存到与im_dir同级的`labels-segment`目录中。默认为None。
        sam_model (str): 用于中间分割数据的分割模型；可选参数。

    Notes:
        数据集假设的输入目录结构：

            - im_dir
                ├─ 001.jpg
                ├─ ..
                └─ NNN.jpg
            - labels
                ├─ 001.txt
                ├─ ..
                └─ NNN.txt
    """
    from tqdm import tqdm  # 导入进度条库tqdm

    from ultralytics import SAM  # 导入分割模型SAM
    from ultralytics.data import YOLODataset  # 导入YOLO数据集
    from ultralytics.utils import LOGGER  # 导入日志记录器
    from ultralytics.utils.ops import xywh2xyxy  # 导入辅助操作函数xywh2xyxy

    # NOTE: add placeholder to pass class index check
    dataset = YOLODataset(im_dir, data=dict(names=list(range(1000))))  # 创建YOLO数据集对象，传入图像目录和类名列表
    if len(dataset.labels[0]["segments"]) > 0:  # 如果存在分割数据
        LOGGER.info("Segmentation labels detected, no need to generate new ones!")  # 记录日志，表示检测到分割标签，无需生成新标签
        return  # 返回

    LOGGER.info("Detection labels detected, generating segment labels by SAM model!")  # 记录日志，表示检测到检测标签，将使用SAM模型生成分割标签
    sam_model = SAM(sam_model)  # 创建SAM模型对象
    for label in tqdm(dataset.labels, total=len(dataset.labels), desc="Generating segment labels"):  # 使用进度条遍历数据集标签
        h, w = label["shape"]  # 获取标签图像的高度和宽度
        boxes = label["bboxes"]  # 获取标签中的边界框信息
        if len(boxes) == 0:  # 如果边界框数量为0，则跳过空标签
            continue
        boxes[:, [0, 2]] *= w  # 将边界框的x坐标缩放到图像宽度上
        boxes[:, [1, 3]] *= h  # 将边界框的y坐标缩放到图像高度上
        im = cv2.imread(label["im_file"])  # 读取标签对应的图像
        sam_results = sam_model(im, bboxes=xywh2xyxy(boxes), verbose=False, save=False)  # 使用SAM模型进行分割，获取分割结果
        label["segments"] = sam_results[0].masks.xyn  # 将分割结果存储在标签数据中的segments字段

    save_dir = Path(save_dir) if save_dir else Path(im_dir).parent / "labels-segment"  # 确定保存目录路径
    save_dir.mkdir(parents=True, exist_ok=True)  # 创建保存目录，如果不存在则创建

    for label in dataset.labels:  # 遍历数据集中的每个标签
        texts = []  # 存储要写入文件的文本列表
        lb_name = Path(label["im_file"]).with_suffix(".txt").name  # 获取标签文件的名称
        txt_file = save_dir / lb_name  # 确定要保存的文本文件路径
        cls = label["cls"]  # 获取标签的类别信息
        for i, s in enumerate(label["segments"]):  # 遍历每个分割标签
            line = (int(cls[i]), *s.reshape(-1))  # 构造要写入文件的一行文本内容
            texts.append(("%g " * len(line)).rstrip() % line)  # 将文本内容格式化并添加到文本列表中
        if texts:  # 如果存在文本内容
            with open(txt_file, "a") as f:  # 打开文件，追加写入模式
                f.writelines(text + "\n" for text in texts)  # 将文本列表中的内容逐行写入文件
    LOGGER.info(f"Generated segment labels saved in {save_dir}")  # 记录日志，表示生成的分割标签已保存在指定目录中
```