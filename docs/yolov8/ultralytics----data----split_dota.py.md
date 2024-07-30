# `.\yolov8\ultralytics\data\split_dota.py`

```py
# 导入必要的库和模块
import itertools  # 导入 itertools 库，用于迭代操作
from glob import glob  # 从 glob 模块中导入 glob 函数，用于文件路径的匹配
from math import ceil  # 导入 math 模块中的 ceil 函数，用于向上取整
from pathlib import Path  # 导入 pathlib 模块中的 Path 类，用于处理路径操作

import cv2  # 导入 OpenCV 库
import numpy as np  # 导入 NumPy 库
from PIL import Image  # 从 PIL 库中导入 Image 模块
from tqdm import tqdm  # 导入 tqdm 库，用于显示进度条

from ultralytics.data.utils import exif_size, img2label_paths  # 导入 ultralytics.data.utils 中的函数
from ultralytics.utils.checks import check_requirements  # 从 ultralytics.utils.checks 导入 check_requirements 函数

# 检查并确保安装了 shapely 库
check_requirements("shapely")
from shapely.geometry import Polygon  # 导入 shapely 库中的 Polygon 类


def bbox_iof(polygon1, bbox2, eps=1e-6):
    """
    Calculate iofs between bbox1 and bbox2.

    Args:
        polygon1 (np.ndarray): Polygon coordinates, (n, 8).
        bbox2 (np.ndarray): Bounding boxes, (n ,4).
    """
    polygon1 = polygon1.reshape(-1, 4, 2)  # 将 polygon1 重新组织成 (n, 4, 2) 的数组
    lt_point = np.min(polygon1, axis=-2)  # 计算 polygon1 中每个多边形的左上角点
    rb_point = np.max(polygon1, axis=-2)  # 计算 polygon1 中每个多边形的右下角点
    bbox1 = np.concatenate([lt_point, rb_point], axis=-1)  # 将左上角和右下角点合并为 bbox1

    lt = np.maximum(bbox1[:, None, :2], bbox2[..., :2])  # 计算左上角点的最大值
    rb = np.minimum(bbox1[:, None, 2:], bbox2[..., 2:])  # 计算右下角点的最小值
    wh = np.clip(rb - lt, 0, np.inf)  # 计算宽度和高度，并将其限制在非负范围内
    h_overlaps = wh[..., 0] * wh[..., 1]  # 计算高度上的重叠区域面积

    left, top, right, bottom = (bbox2[..., i] for i in range(4))  # 提取 bbox2 的左上右下边界坐标
    polygon2 = np.stack([left, top, right, top, right, bottom, left, bottom], axis=-1).reshape(-1, 4, 2)  # 重新组织 polygon2

    sg_polys1 = [Polygon(p) for p in polygon1]  # 创建 polygon1 的多边形对象列表
    sg_polys2 = [Polygon(p) for p in polygon2]  # 创建 polygon2 的多边形对象列表
    overlaps = np.zeros(h_overlaps.shape)  # 创建全零数组用于存储重叠面积
    for p in zip(*np.nonzero(h_overlaps)):
        overlaps[p] = sg_polys1[p[0]].intersection(sg_polys2[p[-1]]).area  # 计算多边形的交集面积
    unions = np.array([p.area for p in sg_polys1], dtype=np.float32)  # 计算多边形的联合面积
    unions = unions[..., None]

    unions = np.clip(unions, eps, np.inf)  # 将 unions 数组限制在 eps 到无穷大的范围内
    outputs = overlaps / unions  # 计算 IOF（Intersection over Full）
    if outputs.ndim == 1:
        outputs = outputs[..., None]
    return outputs  # 返回 IOF 数组


def load_yolo_dota(data_root, split="train"):
    """
    Load DOTA dataset.

    Args:
        data_root (str): Data root.
        split (str): The split data set, could be train or val.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    assert split in {"train", "val"}, f"Split must be 'train' or 'val', not {split}."
    im_dir = Path(data_root) / "images" / split  # 图像目录路径
    assert im_dir.exists(), f"Can't find {im_dir}, please check your data root."
    im_files = glob(str(Path(data_root) / "images" / split / "*"))  # 获取图像文件列表
    lb_files = img2label_paths(im_files)  # 根据图像文件获取标签文件列表
    annos = []
    for im_file, lb_file in zip(im_files, lb_files):
        w, h = exif_size(Image.open(im_file))  # 获取图像的宽度和高度
        with open(lb_file) as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]  # 读取标签文件并处理成列表
            lb = np.array(lb, dtype=np.float32)  # 转换为 NumPy 数组
        annos.append(dict(ori_size=(h, w), label=lb, filepath=im_file))  # 将图像信息和标签信息添加到注释列表中
    return annos  # 返回注释列表


def get_windows(im_size, crop_sizes=(1024,), gaps=(200,), im_rate_thr=0.6, eps=0.01):
    """
    Get the coordinates of windows.
    """
    Args:
        im_size (tuple): Original image size, (h, w).
        crop_sizes (List(int)): Crop size of windows.
        gaps (List(int)): Gap between crops.
        im_rate_thr (float): Threshold of windows areas divided by image areas.
        eps (float): Epsilon value for math operations.
    """
    # 解包图像尺寸
    h, w = im_size
    # 初始化空列表用于存储窗口坐标
    windows = []
    # 遍历crop_sizes和gaps列表，分别为crop_size和gap赋值，生成窗口坐标
    for crop_size, gap in zip(crop_sizes, gaps):
        # 断言crop_size大于gap，否则抛出异常
        assert crop_size > gap, f"invalid crop_size gap pair [{crop_size} {gap}]"
        # 计算步长
        step = crop_size - gap

        # 计算在宽度方向上的窗口数量及其起始位置
        xn = 1 if w <= crop_size else ceil((w - crop_size) / step + 1)
        xs = [step * i for i in range(xn)]
        # 调整最后一个窗口的位置，确保不超出图像边界
        if len(xs) > 1 and xs[-1] + crop_size > w:
            xs[-1] = w - crop_size

        # 计算在高度方向上的窗口数量及其起始位置
        yn = 1 if h <= crop_size else ceil((h - crop_size) / step + 1)
        ys = [step * i for i in range(yn)]
        # 调整最后一个窗口的位置，确保不超出图像边界
        if len(ys) > 1 and ys[-1] + crop_size > h:
            ys[-1] = h - crop_size

        # 使用itertools生成所有可能的窗口坐标，并转换为numpy数组
        start = np.array(list(itertools.product(xs, ys)), dtype=np.int64)
        stop = start + crop_size
        # 将起始和结束坐标连接起来形成完整的窗口坐标
        windows.append(np.concatenate([start, stop], axis=1))
    
    # 将所有窗口坐标连接成一个numpy数组
    windows = np.concatenate(windows, axis=0)

    # 复制窗口坐标，用于进行边界裁剪
    im_in_wins = windows.copy()
    # 对窗口坐标的x坐标进行裁剪，确保不超出图像宽度边界
    im_in_wins[:, 0::2] = np.clip(im_in_wins[:, 0::2], 0, w)
    # 对窗口坐标的y坐标进行裁剪，确保不超出图像高度边界
    im_in_wins[:, 1::2] = np.clip(im_in_wins[:, 1::2], 0, h)
    
    # 计算每个窗口在原始图像中的面积
    im_areas = (im_in_wins[:, 2] - im_in_wins[:, 0]) * (im_in_wins[:, 3] - im_in_wins[:, 1])
    # 计算每个窗口的面积
    win_areas = (windows[:, 2] - windows[:, 0]) * (windows[:, 3] - windows[:, 1])
    # 计算每个窗口的面积比率
    im_rates = im_areas / win_areas
    
    # 如果所有窗口的面积比率都小于等于阈值im_rate_thr，则选择最大比率的窗口
    if not (im_rates > im_rate_thr).any():
        max_rate = im_rates.max()
        im_rates[abs(im_rates - max_rate) < eps] = 1
    
    # 返回符合条件的窗口坐标
    return windows[im_rates > im_rate_thr]
# 将给定窗口中的对象分别提取出来。
def get_window_obj(anno, windows, iof_thr=0.7):
    """Get objects for each window."""
    # 获取原始图像的高度和宽度
    h, w = anno["ori_size"]
    # 获取标签数据
    label = anno["label"]
    # 如果标签非空，则对标签中的坐标进行宽度和高度的缩放
    if len(label):
        label[:, 1::2] *= w
        label[:, 2::2] *= h
        # 计算每个窗口与标签框之间的重叠度
        iofs = bbox_iof(label[:, 1:], windows)
        # 根据重叠度阈值筛选出符合条件的标签框，组成列表
        return [(label[iofs[:, i] >= iof_thr]) for i in range(len(windows))]  # window_anns
    else:
        # 如果标签为空，则返回空的数组
        return [np.zeros((0, 9), dtype=np.float32) for _ in range(len(windows))]  # window_anns


# 裁剪图像并保存新的标签
def crop_and_save(anno, windows, window_objs, im_dir, lb_dir):
    """
    Crop images and save new labels.

    Args:
        anno (dict): Annotation dict, including `filepath`, `label`, `ori_size` as its keys.
        windows (list): A list of windows coordinates.
        window_objs (list): A list of labels inside each window.
        im_dir (str): The output directory path of images.
        lb_dir (str): The output directory path of labels.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    # 读取原始图像
    im = cv2.imread(anno["filepath"])
    # 获取图像文件名的基本部分
    name = Path(anno["filepath"]).stem
    # 遍历每个窗口并进行图像裁剪和保存
    for i, window in enumerate(windows):
        # 解析窗口的起始和结束坐标
        x_start, y_start, x_stop, y_stop = window.tolist()
        # 生成新的文件名，包含窗口大小和起始坐标信息
        new_name = f"{name}__{x_stop - x_start}__{x_start}___{y_start}"
        # 根据窗口坐标裁剪图像
        patch_im = im[y_start:y_stop, x_start:x_stop]
        # 获取裁剪后图像的高度和宽度
        ph, pw = patch_im.shape[:2]

        # 将裁剪后的图像保存为 JPEG 文件
        cv2.imwrite(str(Path(im_dir) / f"{new_name}.jpg"), patch_im)
        # 获取当前窗口对应的标签
        label = window_objs[i]
        # 如果标签为空，则跳过当前窗口
        if len(label) == 0:
            continue
        # 调整标签的坐标，使其相对于裁剪后的图像
        label[:, 1::2] -= x_start
        label[:, 2::2] -= y_start
        label[:, 1::2] /= pw
        label[:, 2::2] /= ph

        # 将调整后的标签保存到文本文件中
        with open(Path(lb_dir) / f"{new_name}.txt", "w") as f:
            for lb in label:
                # 格式化标签的坐标信息，并写入文件
                formatted_coords = ["{:.6g}".format(coord) for coord in lb[1:]]
                f.write(f"{int(lb[0])} {' '.join(formatted_coords)}\n")


# 分割图像和标签
def split_images_and_labels(data_root, save_dir, split="train", crop_sizes=(1024,), gaps=(200,)):
    """
    Split both images and labels.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - split
                - labels
                    - split
        and the output directory structure is:
            - save_dir
                - images
                    - split
                - labels
                    - split
    """
    # 构建输出图像和标签的目录结构
    im_dir = Path(save_dir) / "images" / split
    im_dir.mkdir(parents=True, exist_ok=True)
    lb_dir = Path(save_dir) / "labels" / split
    lb_dir.mkdir(parents=True, exist_ok=True)

    # 加载 YOLO 格式的 DOTA 数据集的注释信息
    annos = load_yolo_dota(data_root, split=split)
    # 使用 tqdm 迭代处理每个标注对象 anno
    for anno in tqdm(annos, total=len(annos), desc=split):
        # 根据原始大小和给定的裁剪尺寸和间隔，获取裁剪窗口列表
        windows = get_windows(anno["ori_size"], crop_sizes, gaps)
        # 根据标注信息和裁剪窗口列表，获取窗口对象列表
        window_objs = get_window_obj(anno, windows)
        # 对原始图像进行裁剪并保存裁剪结果，保存到指定的图像和标签目录
        crop_and_save(anno, windows, window_objs, str(im_dir), str(lb_dir))
# 定义一个函数，用于将 DOTA 数据集的训练和验证集进行分割
def split_trainval(data_root, save_dir, crop_size=1024, gap=200, rates=(1.0,)):
    """
    Split train and val set of DOTA.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
        and the output directory structure is:
            - save_dir
                - images
                    - train
                    - val
                - labels
                    - train
                    - val

    Parameters:
        data_root (str): 数据集的根目录路径
        save_dir (str): 分割后数据集的保存路径
        crop_size (int): 裁剪尺寸，默认为 1024
        gap (int): 裁剪间隙，默认为 200
        rates (tuple): 裁剪尺寸和间隙的比例因子，默认为 (1.0,)

    Returns:
        None
    """
    # 初始化裁剪尺寸列表和间隙列表
    crop_sizes, gaps = [], []
    # 根据比例因子计算实际裁剪尺寸和间隙
    for r in rates:
        crop_sizes.append(int(crop_size / r))
        gaps.append(int(gap / r))
    # 分别处理训练集和验证集
    for split in ["train", "val"]:
        # 调用函数处理每个数据集的图片和标签
        split_images_and_labels(data_root, save_dir, split, crop_sizes, gaps)


# 定义一个函数，用于将 DOTA 数据集的测试集进行分割
def split_test(data_root, save_dir, crop_size=1024, gap=200, rates=(1.0,)):
    """
    Split test set of DOTA, labels are not included within this set.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - test
        and the output directory structure is:
            - save_dir
                - images
                    - test

    Parameters:
        data_root (str): 数据集的根目录路径
        save_dir (str): 分割后数据集的保存路径
        crop_size (int): 裁剪尺寸，默认为 1024
        gap (int): 裁剪间隙，默认为 200
        rates (tuple): 裁剪尺寸和间隙的比例因子，默认为 (1.0,)

    Returns:
        None
    """
    # 初始化裁剪尺寸列表和间隙列表
    crop_sizes, gaps = [], []
    # 根据比例因子计算实际裁剪尺寸和间隙
    for r in rates:
        crop_sizes.append(int(crop_size / r))
        gaps.append(int(gap / r))
    # 确定保存测试集图片的路径并创建目录
    save_dir = Path(save_dir) / "images" / "test"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 获取测试集图片所在目录并检查是否存在
    im_dir = Path(data_root) / "images" / "test"
    assert im_dir.exists(), f"Can't find {im_dir}, please check your data root."
    # 获取所有测试集图片文件列表
    im_files = glob(str(im_dir / "*"))
    # 遍历测试集图片文件
    for im_file in tqdm(im_files, total=len(im_files), desc="test"):
        # 获取图片的原始尺寸
        w, h = exif_size(Image.open(im_file))
        # 根据裁剪尺寸和间隙获取窗口列表
        windows = get_windows((h, w), crop_sizes=crop_sizes, gaps=gaps)
        # 读取图片文件
        im = cv2.imread(im_file)
        # 获取图片文件名（不包含扩展名）
        name = Path(im_file).stem
        # 遍历每个窗口并处理
        for window in windows:
            x_start, y_start, x_stop, y_stop = window.tolist()
            # 构造新的文件名，包含窗口尺寸和起始位置信息
            new_name = f"{name}__{x_stop - x_start}__{x_start}___{y_start}"
            # 裁剪图像并保存
            patch_im = im[y_start:y_stop, x_start:x_stop]
            cv2.imwrite(str(save_dir / f"{new_name}.jpg"), patch_im)


if __name__ == "__main__":
    # 调用函数进行训练集和验证集的分割
    split_trainval(data_root="DOTAv2", save_dir="DOTAv2-split")
    # 调用函数进行测试集的分割
    split_test(data_root="DOTAv2", save_dir="DOTAv2-split")
```