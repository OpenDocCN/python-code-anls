# `.\PaddleOCR\ppocr\utils\utility.py`

```py
# 导入所需的库
import logging
import os
import imghdr
import cv2
import random
import numpy as np
import paddle
import importlib.util
import sys
import subprocess

# 定义一个函数，用于递归打印字典内容，并根据键的关系进行缩进
def print_dict(d, logger, delimiter=0):
    """
    Recursively visualize a dict and
    indenting acrrording by the relationship of keys.
    """
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            logger.info("{}{} : ".format(delimiter * " ", str(k)))
            print_dict(v, logger, delimiter + 4)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            logger.info("{}{} : ".format(delimiter * " ", str(k)))
            for value in v:
                print_dict(value, logger, delimiter + 4)
        else:
            logger.info("{}{} : {}".format(delimiter * " ", k, v))

# 根据模式返回需要检查的全局参数列表
def get_check_global_params(mode):
    check_params = ['use_gpu', 'max_text_length', 'image_shape', \
                    'image_shape', 'character_type', 'loss_type']
    if mode == "train_eval":
        check_params = check_params + [ \
            'train_batch_size_per_card', 'test_batch_size_per_card']
    elif mode == "test":
        check_params = check_params + ['test_batch_size_per_card']
    return check_params

# 检查文件是否为图片文件
def _check_image_file(path):
    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'pdf'}
    return any([path.lower().endswith(e) for e in img_end])
# 获取图片文件列表
def get_image_file_list(img_file):
    # 初始化图片列表
    imgs_lists = []
    # 如果图片文件为空或不存在，则抛出异常
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    # 如果是文件且是图片文件，则将其添加到图片列表中
    if os.path.isfile(img_file) and _check_image_file(img_file):
        imgs_lists.append(img_file)
    # 如果是目录，则遍历目录下的文件
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            # 如果是文件且是图片文件，则将其添加到图片列表中
            if os.path.isfile(file_path) and _check_image_file(file_path):
                imgs_lists.append(file_path)
    # 如果图片列表为空，则抛出异常
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    # 对图片列表进行排序
    imgs_lists = sorted(imgs_lists)
    # 返回图片列表
    return imgs_lists

# 将图片二值化
def binarize_img(img):
    # 如果图片是三通道彩色图片
    if len(img.shape) == 3 and img.shape[2] == 3:
        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 使用 OpenCV 进行阈值二值化
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 将二值化后的灰度图像转换为三通道彩色图片
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return img

# 将 alpha 通道转换为颜色
def alpha_to_color(img, alpha_color=(255, 255, 255)):
    # 如果图片是四通道图片
    if len(img.shape) == 3 and img.shape[2] == 4:
        # 分离通道
        B, G, R, A = cv2.split(img)
        alpha = A / 255

        # 根据 alpha 通道将颜色转换
        R = (alpha_color[0] * (1 - alpha) + R * alpha).astype(np.uint8)
        G = (alpha_color[1] * (1 - alpha) + G * alpha).astype(np.uint8)
        B = (alpha_color[2] * (1 - alpha) + B * alpha).astype(np.uint8)

        # 合并通道
        img = cv2.merge((B, G, R))
    return img

# 检查并读取图片路径
def check_and_read(img_path):
    # 检查文件路径的后缀是否为gif
    if os.path.basename(img_path)[-3:].lower() == 'gif':
        # 读取gif文件
        gif = cv2.VideoCapture(img_path)
        # 读取gif的第一帧
        ret, frame = gif.read()
        # 如果读取失败，记录日志并返回None
        if not ret:
            logger = logging.getLogger('ppocr')
            logger.info("Cannot read {}. This gif image maybe corrupted.")
            return None, False
        # 如果帧的维度为2或者最后一个维度为1，将灰度图像转换为RGB
        if len(frame.shape) == 2 or frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        # 将图像值进行翻转
        imgvalue = frame[:, :, ::-1]
        # 返回图像值和True和False标志
        return imgvalue, True, False
    # 如果文件路径的后缀为pdf
    elif os.path.basename(img_path)[-3:].lower() == 'pdf':
        # 导入fitz和Image模块
        import fitz
        from PIL import Image
        # 创建空列表imgs
        imgs = []
        # 打开pdf文件
        with fitz.open(img_path) as pdf:
            # 遍历pdf的每一页
            for pg in range(0, pdf.page_count):
                page = pdf[pg]
                mat = fitz.Matrix(2, 2)
                pm = page.get_pixmap(matrix=mat, alpha=False)

                # 如果宽度或高度大于2000像素，则不放大图像
                if pm.width > 2000 or pm.height > 2000:
                    pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

                # 将图像数据转换为RGB格式
                img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                imgs.append(img)
            # 返回图像列表和False和True标志
            return imgs, False, True
    # 如果不是gif或pdf文件，返回None和False和False标志
    return None, False, False
# 从文件中加载 VQA 的 BIO 标签映射关系
def load_vqa_bio_label_maps(label_map_path):
    # 打开文件，读取所有行
    with open(label_map_path, "r", encoding='utf-8') as fin:
        lines = fin.readlines()
    # 去除每行两端的空白字符
    old_lines = [line.strip() for line in lines]
    # 初始化一个包含 "O" 的列表
    lines = ["O"]
    # 遍历旧列表，将不是 "OTHER", "OTHERS", "IGNORE" 的行添加到新列表中
    for line in old_lines:
        if line.upper() in ["OTHER", "OTHERS", "IGNORE"]:
            continue
        lines.append(line)
    # 初始化一个包含 "O" 的标签列表
    labels = ["O"]
    # 为每个非 "O" 的标签添加 "B-" 和 "I-" 前缀
    for line in lines[1:]:
        labels.append("B-" + line)
        labels.append("I-" + line)
    # 创建标签到 ID 的映射字典
    label2id_map = {label.upper(): idx for idx, label in enumerate(labels)}
    # 创建 ID 到标签的映射字典
    id2label_map = {idx: label.upper() for idx, label in enumerate(labels)}
    # 返回标签到 ID 和 ID 到标签的映射字典
    return label2id_map, id2label_map


# 设置随机种子
def set_seed(seed=1024):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


# 检查模块是否安装，如果未安装则尝试自动安装
def check_install(module_name, install_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f'Warnning! The {module_name} module is NOT installed')
        print(
            f'Try install {module_name} module automatically. You can also try to install manually by pip install {install_name}.'
        )
        python = sys.executable
        try:
            subprocess.check_call(
                [python, '-m', 'pip', 'install', install_name],
                stdout=subprocess.DEVNULL)
            print(f'The {module_name} module is now installed')
        except subprocess.CalledProcessError as exc:
            raise Exception(
                f"Install {module_name} failed, please install manually")
    else:
        print(f"{module_name} has been installed.")


# 定义一个用于计算平均值的类
class AverageMeter:
    def __init__(self):
        self.reset()

    # 重置计数器
    def reset(self):
        """reset"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    # 更新计数器
    def update(self, val, n=1):
        """update"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
```