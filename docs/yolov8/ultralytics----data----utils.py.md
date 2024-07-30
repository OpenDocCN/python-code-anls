# `.\yolov8\ultralytics\data\utils.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import hashlib
import json
import os
import random
import subprocess
import time
import zipfile
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import is_tarfile

import cv2
import numpy as np
from PIL import Image, ImageOps

# 导入自定义模块和函数
from ultralytics.nn.autobackend import check_class_names
from ultralytics.utils import (
    DATASETS_DIR,
    LOGGER,
    NUM_THREADS,
    ROOT,
    SETTINGS_YAML,
    TQDM,
    clean_url,
    colorstr,
    emojis,
    is_dir_writeable,
    yaml_load,
    yaml_save,
)
# 导入数据校验函数和下载函数
from ultralytics.utils.checks import check_file, check_font, is_ascii
from ultralytics.utils.downloads import download, safe_download, unzip_file
# 导入操作函数
from ultralytics.utils.ops import segments2boxes

# 设置帮助链接
HELP_URL = "See https://docs.ultralytics.com/datasets for dataset formatting guidance."
# 定义支持的图片格式和视频格式
IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp"}  # image suffixes
VID_FORMATS = {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"}  # video suffixes
# 确定是否启用内存固定标记，根据环境变量PIN_MEMORY的值
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders
# 格式帮助信息
FORMATS_HELP_MSG = f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"


def img2label_paths(img_paths):
    """Define label paths as a function of image paths."""
    # 定义图片路径和标签路径的转换关系
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


def get_hash(paths):
    """Returns a single hash value of a list of paths (files or dirs)."""
    # 计算路径列表中文件或目录的总大小
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    # 使用SHA-256算法计算路径列表的哈希值
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img: Image.Image):
    """Returns exif-corrected PIL size."""
    s = img.size  # (width, height)
    if img.format == "JPEG":  # only support JPEG images
        # 尝试获取图像的EXIF信息，并根据EXIF信息修正图像尺寸
        with contextlib.suppress(Exception):
            exif = img.getexif()
            if exif:
                rotation = exif.get(274, None)  # the EXIF key for the orientation tag is 274
                if rotation in {6, 8}:  # rotation 270 or 90
                    s = s[1], s[0]
    return s


def verify_image(args):
    """Verify one image."""
    (im_file, cls), prefix = args
    # 初始化计数器和消息字符串
    nf, nc, msg = 0, 0, ""
    try:
        # 尝试打开图像文件
        im = Image.open(im_file)
        # 使用PIL库验证图像文件
        im.verify()  # PIL verify
        # 获取图像的尺寸信息
        shape = exif_size(im)  # image size
        # 调整尺寸信息的顺序为宽度在前，高度在后
        shape = (shape[1], shape[0])  # hw
        # 断言图像的宽度和高度大于9像素
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        # 断言图像的格式在允许的图像格式列表中
        assert im.format.lower() in IMG_FORMATS, f"Invalid image format {im.format}. {FORMATS_HELP_MSG}"
        # 如果图像格式是JPEG，则进一步检查是否损坏
        if im.format.lower() in {"jpg", "jpeg"}:
            # 使用二进制模式打开文件，定位到文件末尾的倒数第二个字节
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                # 检查文件末尾两个字节是否为JPEG文件的结束标记
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    # 修复并保存损坏的JPEG文件
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    # 生成警告信息
                    msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"
        # 如果没有异常发生，设置nf为1
        nf = 1
    except Exception as e:
        # 捕获异常，并设置nc为1，生成警告信息
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
    # 返回结果元组
    return (im_file, cls), nf, nc, msg
# 验证单个图像-标签对的有效性
def verify_image_label(args):
    # 解包参数：图像文件路径、标签文件路径、前缀、关键点、类别数、关键点数、维度数
    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim = args
    # 初始化计数器和消息变量
    # nm: 缺失的数量
    # nf: 发现的数量
    # ne: 空的数量
    # nc: 损坏的数量
    # msg: 信息字符串
    # segments: 段
    # keypoints: 关键点
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None

    # 捕获任何异常并记录为损坏的图像/标签对
    except Exception as e:
        # 标记为损坏的数量增加
        nc = 1
        # 设置消息内容，标记文件和具体的错误信息
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        # 返回空的计数和消息，其余变量为 None
        return [None, None, None, None, None, nm, nf, ne, nc, msg]


def polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):
    """
    将多边形列表转换为指定图像尺寸的二进制掩码。

    Args:
        imgsz (tuple): 图像的大小，格式为 (height, width)。
        polygons (list[np.ndarray]): 多边形列表。每个多边形是一个形状为 [N, M] 的数组，
                                     其中 N 是多边形的数量，M 是点的数量，满足 M % 2 = 0。
        color (int, optional): 在掩码中填充多边形的颜色值。默认为 1。
        downsample_ratio (int, optional): 缩小掩码的因子。默认为 1。

    Returns:
        (np.ndarray): 指定图像尺寸的二进制掩码，填充了多边形。
    """
    # 创建一个全零数组作为掩码
    mask = np.zeros(imgsz, dtype=np.uint8)
    # 将多边形列表转换为 numpy 数组，类型为 int32
    polygons = np.asarray(polygons, dtype=np.int32)
    # 重新整形多边形数组以便填充多边形的顶点
    polygons = polygons.reshape((polygons.shape[0], -1, 2))
    # 使用指定的颜色值填充多边形到掩码中
    cv2.fillPoly(mask, polygons, color=color)
    # 计算缩小后的掩码尺寸
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    # 返回缩小后的掩码，保持与原掩码相同的填充方法
    return cv2.resize(mask, (nw, nh))


def polygons2masks(imgsz, polygons, color, downsample_ratio=1):
    """
    将多边形列表转换为指定图像尺寸的一组二进制掩码。

    Args:
        imgsz (tuple): 图像的大小，格式为 (height, width)。
        polygons (list[np.ndarray]): 多边形列表。每个多边形是一个形状为 [N, M] 的数组，
                                     其中 N 是多边形的数量，M 是点的数量，满足 M % 2 = 0。
        color (int): 在掩码中填充多边形的颜色值。
        downsample_ratio (int, optional): 缩小每个掩码的因子。默认为 1。

    Returns:
        (np.ndarray): 指定图像尺寸的一组二进制掩码，填充了多边形。
    """
    # 对多边形列表中的每个多边形，调用 polygon2mask 函数生成掩码数组，并返回为 numpy 数组
    return np.array([polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio) for x in polygons])


def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    """
    返回一个 (640, 640) 的重叠掩码。

    Args:
        imgsz (tuple): 图像的大小，格式为 (height, width)。
        segments (list): 段列表。
        downsample_ratio (int, optional): 缩小掩码的因子。默认为 1。

    Returns:
        np.ndarray: 指定图像尺寸的重叠掩码。
    """
    # 创建一个全零数组作为掩码，尺寸根据缩小因子调整
    masks = np.zeros(
        (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
        dtype=np.int32 if len(segments) > 255 else np.uint8,
    )
    # 初始化一个区域列表
    areas = []
    # 初始化一个段列表
    ms = []
    # 对于每个分割段落进行迭代
    for si in range(len(segments)):
        # 根据分割段落创建一个二进制掩码
        mask = polygon2mask(imgsz, [segments[si].reshape(-1)], downsample_ratio=downsample_ratio, color=1)
        # 将生成的掩码添加到掩码列表中
        ms.append(mask)
        # 计算掩码的像素总数，并将其添加到面积列表中
        areas.append(mask.sum())
    
    # 将面积列表转换为 NumPy 数组
    areas = np.asarray(areas)
    # 按照面积大小降序排列索引
    index = np.argsort(-areas)
    # 根据排序后的索引重新排列掩码列表
    ms = np.array(ms)[index]
    
    # 对每个分割段落再次进行迭代
    for i in range(len(segments)):
        # 将重新排序的掩码乘以当前索引加一，生成最终的分割掩码
        mask = ms[i] * (i + 1)
        # 将生成的分割掩码加到总掩码中
        masks = masks + mask
        # 对总掩码进行截断，确保像素值在指定范围内
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    
    # 返回最终生成的总掩码和排序后的索引
    return masks, index
def find_dataset_yaml(path: Path) -> Path:
    """
    Find and return the YAML file associated with a Detect, Segment or Pose dataset.

    This function searches for a YAML file at the root level of the provided directory first, and if not found, it
    performs a recursive search. It prefers YAML files that have the same stem as the provided path. An AssertionError
    is raised if no YAML file is found or if multiple YAML files are found.

    Args:
        path (Path): The directory path to search for the YAML file.

    Returns:
        (Path): The path of the found YAML file.
    """
    # Attempt to find YAML files at the root level first, otherwise perform a recursive search
    files = list(path.glob("*.yaml")) or list(path.rglob("*.yaml"))  # try root level first and then recursive
    
    # Ensure at least one YAML file is found; otherwise, raise an AssertionError
    assert files, f"No YAML file found in '{path.resolve()}'"
    
    # If multiple YAML files are found, filter to prefer those with the same stem as the provided path
    if len(files) > 1:
        files = [f for f in files if f.stem == path.stem]  # prefer *.yaml files that match
    
    # Ensure exactly one YAML file is found; otherwise, raise an AssertionError with details
    assert len(files) == 1, f"Expected 1 YAML file in '{path.resolve()}', but found {len(files)}.\n{files}"
    
    # Return the path of the found YAML file
    return files[0]


def check_det_dataset(dataset, autodownload=True):
    """
    Download, verify, and/or unzip a dataset if not found locally.

    This function checks the availability of a specified dataset, and if not found, it has the option to download and
    unzip the dataset. It then reads and parses the accompanying YAML data, ensuring key requirements are met and also
    resolves paths related to the dataset.

    Args:
        dataset (str): Path to the dataset or dataset descriptor (like a YAML file).
        autodownload (bool, optional): Whether to automatically download the dataset if not found. Defaults to True.

    Returns:
        (dict): Parsed dataset information and paths.
    """

    # Check if the dataset file exists locally and get its path
    file = check_file(dataset)

    # If the dataset file is a ZIP or TAR archive, download and unzip it if necessary
    extract_dir = ""
    if zipfile.is_zipfile(file) or is_tarfile(file):
        new_dir = safe_download(file, dir=DATASETS_DIR, unzip=True, delete=False)
        # Find and return the YAML file within the extracted directory
        file = find_dataset_yaml(DATASETS_DIR / new_dir)
        extract_dir, autodownload = file.parent, False

    # Load YAML data from the specified file, appending the filename to the loaded data
    data = yaml_load(file, append_filename=True)  # dictionary

    # Perform checks on the loaded YAML data
    for k in "train", "val":
        if k not in data:
            if k != "val" or "validation" not in data:
                # Raise a SyntaxError if required keys 'train' and 'val' (or 'validation') are missing
                raise SyntaxError(
                    emojis(f"{dataset} '{k}:' key missing ❌.\n'train' and 'val' are required in all data YAMLs.")
                )
            # Log a warning and rename 'validation' key to 'val' if necessary
            LOGGER.info("WARNING ⚠️ renaming data YAML 'validation' key to 'val' to match YOLO format.")
            data["val"] = data.pop("validation")  # replace 'validation' key with 'val' key

    # Ensure 'names' or 'nc' keys are present in the data; otherwise, raise a SyntaxError
    if "names" not in data and "nc" not in data:
        raise SyntaxError(emojis(f"{dataset} key missing ❌.\n either 'names' or 'nc' are required in all data YAMLs."))

    # Ensure the lengths of 'names' and 'nc' match if both are present
    if "names" in data and "nc" in data and len(data["names"]) != data["nc"]:
        raise SyntaxError(emojis(f"{dataset} 'names' length {len(data['names'])} and 'nc: {data['nc']}' must match."))
    # 如果数据字典中不存在键 "names"，则创建一个名为 "names" 的列表，包含以"class_{i}"命名的元素，其中i从0到data["nc"]-1
    # 如果数据字典中已经存在 "names" 键，则将 "nc" 设置为 "names" 列表的长度
    if "names" not in data:
        data["names"] = [f"class_{i}" for i in range(data["nc"])]
    else:
        data["nc"] = len(data["names"])

    # 调用函数 check_class_names()，检查并修正 "names" 列表中的每个元素
    data["names"] = check_class_names(data["names"])

    # 解析和设置路径信息
    # path 变量根据 extract_dir、data["path"] 或者 data["yaml_file"] 的父路径创建，表示数据集的根路径
    path = Path(extract_dir or data.get("path") or Path(data.get("yaml_file", "")).parent)  # dataset root
    if not path.is_absolute():
        path = (DATASETS_DIR / path).resolve()  # 如果路径不是绝对路径，则基于 DATASETS_DIR 设置绝对路径

    # 设置 data["path"] 为解析后的路径
    data["path"] = path  # download scripts

    # 对于 "train", "val", "test", "minival" 中的每个键，如果数据字典中存在该键，则将其路径设置为绝对路径
    for k in "train", "val", "test", "minival":
        if data.get(k):  # 如果该键存在
            if isinstance(data[k], str):
                # 如果路径是字符串类型，则基于 path 设置绝对路径
                x = (path / data[k]).resolve()
                # 如果路径不存在且以 "../" 开头，则修正路径
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                # 如果路径是列表，则对列表中每个路径基于 path 设置绝对路径
                data[k] = [str((path / x).resolve()) for x in data[k]]

    # 解析 YAML 文件
    val, s = (data.get(x) for x in ("val", "download"))
    if val:
        # 如果存在 val，将其解析为绝对路径列表
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        # 如果存在某个路径不存在，则抛出 FileNotFoundError
        if not all(x.exists() for x in val):
            name = clean_url(dataset)  # 去除 URL 认证信息后的数据集名称
            # 构建错误信息字符串
            m = f"\nDataset '{name}' images not found ⚠️, missing path '{[x for x in val if not x.exists()][0]}'"
            if s and autodownload:
                LOGGER.warning(m)
            else:
                m += f"\nNote dataset download directory is '{DATASETS_DIR}'. You can update this in '{SETTINGS_YAML}'"
                raise FileNotFoundError(m)
            t = time.time()
            r = None  # 表示成功
            # 如果 s 是以 "http" 开头且以 ".zip" 结尾，则执行安全下载
            if s.startswith("http") and s.endswith(".zip"):  # URL
                safe_download(url=s, dir=DATASETS_DIR, delete=True)
            elif s.startswith("bash "):  # 如果 s 是以 "bash " 开头，则运行 bash 脚本
                LOGGER.info(f"Running {s} ...")
                r = os.system(s)
            else:  # 否则，执行 Python 脚本
                exec(s, {"yaml": data})
            dt = f"({round(time.time() - t, 1)}s)"
            # 根据执行结果设置日志消息
            s = f"success ✅ {dt}, saved to {colorstr('bold', DATASETS_DIR)}" if r in {0, None} else f"failure {dt} ❌"
            LOGGER.info(f"Dataset download {s}\n")

    # 检查并下载字体文件，根据 "names" 是否只包含 ASCII 字符选择不同的字体文件进行下载
    check_font("Arial.ttf" if is_ascii(data["names"]) else "Arial.Unicode.ttf")  # download fonts

    return data  # 返回更新后的数据字典
    # 检查分类数据集，如Imagenet。

    # 如果 `dataset` 以 "http:/" 或 "https:/" 开头，尝试从网络下载数据集并保存到本地。
    # 如果 `dataset` 是以 ".zip", ".tar", 或 ".gz" 结尾的文件路径，检查文件的有效性后，下载并解压数据集到指定目录。

    # 将 `dataset` 转换为 `Path` 对象，并解析其绝对路径。
    dataset = Path(dataset)
    data_dir = (dataset if dataset.is_dir() else (DATASETS_DIR / dataset)).resolve()

    # 如果指定路径的数据集不存在，尝试从网络下载。
    if not data_dir.is_dir():
        # 如果 `dataset` 是 "imagenet"，执行特定的数据集下载脚本。
        # 否则，从 GitHub 发布的资源中下载指定的数据集压缩文件。
        LOGGER.warning(f"\nDataset not found ⚠️, missing path {data_dir}, attempting download...")
        t = time.time()
        if str(dataset) == "imagenet":
            subprocess.run(f"bash {ROOT / 'data/scripts/get_imagenet.sh'}", shell=True, check=True)
        else:
            url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{dataset}.zip"
            download(url, dir=data_dir.parent)
        s = f"Dataset download success ✅ ({time.time() - t:.1f}s), saved to {colorstr('bold', data_dir)}\n"
        LOGGER.info(s)

    # 训练集的路径
    train_set = data_dir / "train"

    # 验证集的路径，优先选择 "val" 目录，其次选择 "validation" 目录，如果都不存在则为 None。
    val_set = (
        data_dir / "val"
        if (data_dir / "val").exists()
        else data_dir / "validation"
        if (data_dir / "validation").exists()
        else None
    )  # data/test or data/val

    # 测试集的路径，优先选择 "test" 目录，如果不存在则为 None。
    test_set = data_dir / "test" if (data_dir / "test").exists() else None  # data/val or data/test

    # 如果 `split` 参数为 "val"，但验证集路径 `val_set` 不存在时，发出警告并使用测试集路径代替。
    if split == "val" and not val_set:
        LOGGER.warning("WARNING ⚠️ Dataset 'split=val' not found, using 'split=test' instead.")
    
    # 如果 `split` 参数为 "test"，但测试集路径 `test_set` 不存在时，发出警告并使用验证集路径代替。
    elif split == "test" and not test_set:
        LOGGER.warning("WARNING ⚠️ Dataset 'split=test' not found, using 'split=val' instead.")

    # 计算数据集中的类别数目，通过统计 `train` 目录下的子目录数量来得到。
    nc = len([x for x in (data_dir / "train").glob("*") if x.is_dir()])  # number of classes

    # 获取训练集中的类别名称列表，并按字母顺序排序后构建成字典，键为类别索引。
    names = [x.name for x in (data_dir / "train").iterdir() if x.is_dir()]  # class names list
    names = dict(enumerate(sorted(names)))

    # 打印结果到控制台
    # 遍历包含训练集、验证集和测试集的字典，每次迭代获取键值对（k为键，v为对应的数据集）
    for k, v in {"train": train_set, "val": val_set, "test": test_set}.items():
        # 使用f-string生成带颜色的前缀字符串，指示当前数据集的名称和状态
        prefix = f'{colorstr(f"{k}:")} {v}...'
        # 如果当前数据集为空（None），记录信息到日志
        if v is None:
            LOGGER.info(prefix)
        else:
            # 获取当前数据集中所有符合图像格式的文件路径列表
            files = [path for path in v.rglob("*.*") if path.suffix[1:].lower() in IMG_FORMATS]
            # 计算当前数据集中的文件数目（nf）和不重复父目录数（nd）
            nf = len(files)  # 文件数目
            nd = len({file.parent for file in files})  # 不重复父目录数
            # 如果当前数据集中没有找到图像文件
            if nf == 0:
                # 如果是训练集，抛出文件未找到的错误并记录
                if k == "train":
                    raise FileNotFoundError(emojis(f"{dataset} '{k}:' no training images found ❌ "))
                else:
                    # 否则记录警告信息，指示没有找到图像文件
                    LOGGER.warning(f"{prefix} found {nf} images in {nd} classes: WARNING ⚠️ no images found")
            # 如果当前数据集中的类别数目与期望的类别数目不匹配
            elif nd != nc:
                # 记录警告信息，指示类别数目不匹配的错误
                LOGGER.warning(f"{prefix} found {nf} images in {nd} classes: ERROR ❌️ requires {nc} classes, not {nd}")
            else:
                # 记录信息，指示成功找到图像文件并且类别数目匹配
                LOGGER.info(f"{prefix} found {nf} images in {nd} classes ✅ ")

    # 返回包含训练集、验证集、测试集、类别数和类别名称的字典
    return {"train": train_set, "val": val_set, "test": test_set, "nc": nc, "names": names}
    """
    A class for generating HUB dataset JSON and `-hub` dataset directory.

    Args:
        path (str): Path to data.yaml or data.zip (with data.yaml inside data.zip). Default is 'coco8.yaml'.
        task (str): Dataset task. Options are 'detect', 'segment', 'pose', 'classify'. Default is 'detect'.
        autodownload (bool): Attempt to download dataset if not found locally. Default is False.

    Example:
        Download *.zip files from https://github.com/ultralytics/hub/tree/main/example_datasets
            i.e. https://github.com/ultralytics/hub/raw/main/example_datasets/coco8.zip for coco8.zip.
        ```py
        from ultralytics.data.utils import HUBDatasetStats

        stats = HUBDatasetStats('path/to/coco8.zip', task='detect')  # detect dataset
        stats = HUBDatasetStats('path/to/coco8-seg.zip', task='segment')  # segment dataset
        stats = HUBDatasetStats('path/to/coco8-pose.zip', task='pose')  # pose dataset
        stats = HUBDatasetStats('path/to/dota8.zip', task='obb')  # OBB dataset
        stats = HUBDatasetStats('path/to/imagenet10.zip', task='classify')  # classification dataset

        stats.get_json(save=True)
        stats.process_images()
        ```
    """

    def __init__(self, path="coco8.yaml", task="detect", autodownload=False):
        """Initialize class."""
        # Resolve the given path to its absolute form
        path = Path(path).resolve()
        # Log information message about starting dataset checks
        LOGGER.info(f"Starting HUB dataset checks for {path}....")

        # Initialize class attributes based on arguments
        self.task = task  # detect, segment, pose, classify

        # Depending on the task type, perform different operations
        if self.task == "classify":
            # Unzip the file and check the classification dataset
            unzip_dir = unzip_file(path)
            data = check_cls_dataset(unzip_dir)
            data["path"] = unzip_dir
        else:  # detect, segment, pose
            # Unzip the file, extract data directory and yaml path
            _, data_dir, yaml_path = self._unzip(Path(path))
            try:
                # Load YAML with checks
                data = yaml_load(yaml_path)
                # Strip path since YAML should be in dataset root for all HUB datasets
                data["path"] = ""
                yaml_save(yaml_path, data)
                # Perform dataset checks for detection dataset
                data = check_det_dataset(yaml_path, autodownload)  # dict
                # Set YAML path to data directory (relative) or parent (absolute)
                data["path"] = data_dir
            except Exception as e:
                # Raise an exception with a specific error message
                raise Exception("error/HUB/dataset_stats/init") from e

        # Set attributes for dataset directory and related paths
        self.hub_dir = Path(f'{data["path"]}-hub')
        self.im_dir = self.hub_dir / "images"
        # Create a statistics dictionary based on loaded data
        self.stats = {"nc": len(data["names"]), "names": list(data["names"].values())}
        self.data = data
    # 解压缩指定路径的 ZIP 文件，并返回解压后的目录路径和数据集 YAML 文件路径
    def _unzip(path):
        """Unzip data.zip."""
        # 如果路径不是以 ".zip" 结尾，则认为是数据文件而非压缩文件，直接返回 False 表示未解压，以及原始路径
        if not str(path).endswith(".zip"):  # path is data.yaml
            return False, None, path
        # 调用 unzip_file 函数解压指定路径的 ZIP 文件到其父目录
        unzip_dir = unzip_file(path, path=path.parent)
        # 断言解压后的目录存在，否则输出错误信息，提示预期的解压路径
        assert unzip_dir.is_dir(), (
            f"Error unzipping {path}, {unzip_dir} not found. " f"path/to/abc.zip MUST unzip to path/to/abc/"
        )
        # 返回 True 表示成功解压，解压后的目录路径字符串，以及在解压目录中找到的数据集 YAML 文件路径
        return True, str(unzip_dir), find_dataset_yaml(unzip_dir)  # zipped, data_dir, yaml_path

    # 保存压缩后的图像用于 HUB 预览
    def _hub_ops(self, f):
        """Saves a compressed image for HUB previews."""
        # 调用 compress_one_image 函数，将指定文件 f 压缩保存到 self.im_dir 目录下，使用文件名作为保存的文件名
        compress_one_image(f, self.im_dir / Path(f).name)  # save to dataset-hub

    # 处理图像，为 Ultralytics HUB 压缩图像
    def process_images(self):
        """Compress images for Ultralytics HUB."""
        from ultralytics.data import YOLODataset  # ClassificationDataset

        # 创建目录 self.im_dir，如果不存在则创建，用于保存压缩后的图像文件
        self.im_dir.mkdir(parents=True, exist_ok=True)  # makes dataset-hub/images/
        
        # 遍历 "train", "val", "test" 三个数据集分割
        for split in "train", "val", "test":
            # 如果 self.data 中不存在当前分割的数据集，则跳过继续下一个分割
            if self.data.get(split) is None:
                continue
            # 创建 YOLODataset 对象，指定图像路径为 self.data[split]，数据为 self.data
            dataset = YOLODataset(img_path=self.data[split], data=self.data)
            # 使用线程池 ThreadPool，并发处理图像压缩操作
            with ThreadPool(NUM_THREADS) as pool:
                # 使用 TQDM 显示进度条，遍历数据集中的图像文件，对每个图像文件调用 _hub_ops 方法进行压缩保存操作
                for _ in TQDM(pool.imap(self._hub_ops, dataset.im_files), total=len(dataset), desc=f"{split} images"):
                    pass
        # 输出日志信息，指示所有图像保存到 self.im_dir 目录下完成
        LOGGER.info(f"Done. All images saved to {self.im_dir}")
        # 返回保存压缩图像的目录路径
        return self.im_dir
# 自动将数据集分割成训练集/验证集/测试集，并将结果保存到autosplit_*.txt文件中
def autosplit(path=DATASETS_DIR / "coco8/images", weights=(0.9, 0.1, 0.0), annotated_only=False):
    """
    Automatically split a dataset into train/val/test splits and save the resulting splits into autosplit_*.txt files.

    Args:
        path (Path, optional): Path to images directory. Defaults to DATASETS_DIR / 'coco8/images'.
        weights (list | tuple, optional): Train, validation, and test split fractions. Defaults to (0.9, 0.1, 0.0).
        annotated_only (bool, optional): If True, only images with an associated txt file are used. Defaults to False.

    Example:
        ```py
        from ultralytics.data.utils import autosplit

        autosplit()
        ```
    """

    path = Path(path)  # 图像目录的路径
    # 筛选出所有符合图片格式的文件，以列表形式存储在files中
    files = sorted(x for x in path.rglob("*.*") if x.suffix[1:].lower() in IMG_FORMATS)  # 只保留图片文件
    n = len(files)  # 文件总数
    random.seed(0)  # 设置随机种子以便复现结果
    # 根据权重随机分配每个图片到训练集、验证集或测试集，k=n表示生成n个随机数
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # 将每个图片分配到相应的集合中

    # 定义三个txt文件名，分别用于存储训练集、验证集、测试集的文件列表
    txt = ["autosplit_train.txt", "autosplit_val.txt", "autosplit_test.txt"]
    # 如果文件已存在，则先删除
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()  # 删除已存在的文件

    # 输出信息，指示正在对图像进行自动分割处理，并显示是否只使用有标签的图像文件
    LOGGER.info(f"Autosplitting images from {path}" + ", using *.txt labeled images only" * annotated_only)
    # 使用 tqdm 迭代处理索引和文件列表 zip(indices, files)，总数为 n，同时显示进度条
    for i, img in TQDM(zip(indices, files), total=n):
        # 如果 annotated_only 为 False 或者对应图片的标签文件存在，则执行下面的操作
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # 检查标签文件是否存在
            # 以追加模式打开路径 path.parent / txt[i] 对应的文件，并写入当前图片路径
            with open(path.parent / txt[i], "a") as f:
                # 将当前图片相对于 path.parent 的路径作为 POSIX 路径添加到文本文件中，并换行
                f.write(f"./{img.relative_to(path.parent).as_posix()}" + "\n")
def load_dataset_cache_file(path):
    """Load an Ultralytics *.cache dictionary from path."""
    import gc  # 导入垃圾回收模块

    gc.disable()  # 禁用垃圾回收，以减少反序列化时间 https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True).item()  # 加载字典对象
    gc.enable()  # 启用垃圾回收
    return cache  # 返回加载的缓存数据


def save_dataset_cache_file(prefix, path, x, version):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    x["version"] = version  # 添加缓存版本信息
    if is_dir_writeable(path.parent):  # 检查父目录是否可写
        if path.exists():
            path.unlink()  # 如果文件已存在，则删除 *.cache 文件
        np.save(str(path), x)  # 将缓存保存到文件中以便下次使用
        path.with_suffix(".cache.npy").rename(path)  # 移除 .npy 后缀
        LOGGER.info(f"{prefix}New cache created: {path}")  # 记录日志，显示创建了新的缓存文件
    else:
        LOGGER.warning(f"{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.")
        # 记录警告日志，显示缓存目录不可写，未保存缓存信息
```