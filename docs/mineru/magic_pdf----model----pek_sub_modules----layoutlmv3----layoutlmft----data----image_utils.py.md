# `.\MinerU\magic_pdf\model\pek_sub_modules\layoutlmv3\layoutlmft\data\image_utils.py`

```
# 导入 torchvision 库中的功能模块
import torchvision.transforms.functional as F
# 导入警告模块
import warnings
# 导入数学模块
import math
# 导入随机数模块
import random
# 导入 NumPy 库
import numpy as np
# 从 PIL 导入图像处理模块
from PIL import Image
# 导入 PyTorch 库
import torch

# 从 detectron2 导入读取图像的工具
from detectron2.data.detection_utils import read_image
# 从 detectron2 导入变换工具
from detectron2.data.transforms import ResizeTransform, TransformList

# 归一化边界框，缩放到 1000x1000 的范围
def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),  # 归一化 x1
        int(1000 * bbox[1] / size[1]),  # 归一化 y1
        int(1000 * bbox[2] / size[0]),  # 归一化 x2
        int(1000 * bbox[3] / size[1]),  # 归一化 y2
    ]

# 加载图像并应用变换
def load_image(image_path):
    image = read_image(image_path, format="BGR")  # 读取图像，格式为 BGR
    h = image.shape[0]  # 获取图像高度
    w = image.shape[1]  # 获取图像宽度
    img_trans = TransformList([ResizeTransform(h=h, w=w, new_h=224, new_w=224)])  # 创建变换列表
    image = torch.tensor(img_trans.apply_image(image).copy()).permute(2, 0, 1)  # 应用变换并转换为张量
    return image, (w, h)  # 返回图像和原始尺寸

# 裁剪图像并处理边界框
def crop(image, i, j, h, w, boxes=None):
    cropped_image = F.crop(image, i, j, h, w)  # 裁剪图像

    if boxes is not None:  # 如果有边界框
        # 处理裁剪后的边界框，确保不会超出裁剪图像范围
        max_size = torch.as_tensor([w, h], dtype=torch.float32)  # 创建最大尺寸张量
        cropped_boxes = torch.as_tensor(boxes) - torch.as_tensor([j, i, j, i])  # 调整边界框坐标
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)  # 确保不超出最大尺寸
        cropped_boxes = cropped_boxes.clamp(min=0)  # 限制最小值为 0
        boxes = cropped_boxes.reshape(-1, 4)  # 重新调整边界框形状

    return cropped_image, boxes  # 返回裁剪后的图像和边界框

# 调整图像大小并处理边界框
def resize(image, size, interpolation, boxes=None):
    # 此处不需要调整边界框，因为最终会调整到 1000x1000 尺寸
    rescaled_image = F.resize(image, size, interpolation)  # 调整图像大小

    if boxes is None:  # 如果没有边界框
        return rescaled_image, None  # 返回调整后的图像和 None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))  # 计算尺寸比
    ratio_width, ratio_height = ratios  # 获取宽高比

    # boxes = boxes.copy()  # 复制边界框（已注释）

    scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])  # 缩放边界框

    return rescaled_image, scaled_boxes  # 返回调整后的图像和缩放后的边界框

# 限制数字在给定的范围内
def clamp(num, min_value, max_value):
    return max(min(num, max_value), min_value)  # 返回限制后的值

# 获取边界框并进行处理
def get_bb(bb, page_size):
    bbs = [float(j) for j in bb]  # 转换边界框为浮点数
    xs, ys = [], []  # 存储 x 和 y 坐标
    for i, b in enumerate(bbs):  # 遍历边界框坐标
        if i % 2 == 0:  # 如果是 x 坐标
            xs.append(b)  # 添加到 x 列表
        else:  # 如果是 y 坐标
            ys.append(b)  # 添加到 y 列表
    (width, height) = page_size  # 获取页面尺寸
    return_bb = [
        clamp(min(xs), 0, width - 1),  # 限制 x1
        clamp(min(ys), 0, height - 1),  # 限制 y1
        clamp(max(xs), 0, width - 1),  # 限制 x2
        clamp(max(ys), 0, height - 1),  # 限制 y2
    ]
    return_bb = [
            int(1000 * return_bb[0] / width),  # 归一化 x1
            int(1000 * return_bb[1] / height),  # 归一化 y1
            int(1000 * return_bb[2] / width),  # 归一化 x2
            int(1000 * return_bb[3] / height),  # 归一化 y2
        ]
    return return_bb  # 返回归一化后的边界框

# 定义一个将图像转换为 NumPy 数组的类
class ToNumpy:
    # 定义一个可调用的方法，接收 PIL 图像作为参数
        def __call__(self, pil_img):
            # 将 PIL 图像转换为 NumPy 数组，数据类型为无符号 8 位整数
            np_img = np.array(pil_img, dtype=np.uint8)
            # 检查数组的维度，如果维度小于 3，则在最后一维增加一个维度
            if np_img.ndim < 3:
                np_img = np.expand_dims(np_img, axis=-1)
            # 将图像的维度从 HWC（高度、宽度、通道）转换为 CHW（通道、高度、宽度）
            np_img = np.rollaxis(np_img, 2)  # HWC to CHW
            # 返回转换后的 NumPy 数组
            return np_img
# 定义一个将图像转换为张量的类
class ToTensor:

    # 初始化方法，设置数据类型，默认为 torch.float32
    def __init__(self, dtype=torch.float32):
        # 存储数据类型
        self.dtype = dtype

    # 调用方法，将 PIL 图像转换为张量
    def __call__(self, pil_img):
        # 将 PIL 图像转换为 uint8 类型的 NumPy 数组
        np_img = np.array(pil_img, dtype=np.uint8)
        # 如果数组的维度小于 3，增加一个维度以表示通道
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        # 将图像从 HWC 格式转换为 CHW 格式
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        # 将 NumPy 数组转换为张量并转换为指定的数据类型
        return torch.from_numpy(np_img).to(dtype=self.dtype)


# 定义 PIL 图像插值方法与其字符串表示的映射
_pil_interpolation_to_str = {
    F.InterpolationMode.NEAREST: 'F.InterpolationMode.NEAREST',  # 最近邻插值
    F.InterpolationMode.BILINEAR: 'F.InterpolationMode.BILINEAR',  # 双线性插值
    F.InterpolationMode.BICUBIC: 'F.InterpolationMode.BICUBIC',  # 三次插值
    F.InterpolationMode.LANCZOS: 'F.InterpolationMode.LANCZOS',  # Lanczos 插值
    F.InterpolationMode.HAMMING: 'F.InterpolationMode.HAMMING',  # Hamming 插值
    F.InterpolationMode.BOX: 'F.InterpolationMode.BOX',  # 箱型插值
}


# 根据给定的插值方法返回相应的 PIL 插值模式
def _pil_interp(method):
    # 如果方法是 'bicubic'，返回三次插值模式
    if method == 'bicubic':
        return F.InterpolationMode.BICUBIC
    # 如果方法是 'lanczos'，返回 Lanczos 插值模式
    elif method == 'lanczos':
        return F.InterpolationMode.LANCZOS
    # 如果方法是 'hamming'，返回 Hamming 插值模式
    elif method == 'hamming':
        return F.InterpolationMode.HAMMING
    else:
        # 默认返回双线性插值模式
        return F.InterpolationMode.BILINEAR


# 定义一个组合多个变换的类
class Compose:
    """组合多个变换。此变换不支持 torchscript。
    请查看下面的说明。

    Args:
        transforms (list of ``Transform`` objects): 组合的变换列表。

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),  # 中心裁剪为 10x10
        >>>     transforms.PILToTensor(),  # 将 PIL 图像转换为张量
        >>>     transforms.ConvertImageDtype(torch.float),  # 转换图像数据类型为 float
        >>> ])

    .. note::
        为了脚本化变换，请使用 ``torch.nn.Sequential``，如下面所示。

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),  # 中心裁剪为 10x10
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # 标准化图像
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)  # 脚本化变换

        确保仅使用可脚本化的变换，即与 ``torch.Tensor`` 兼容，不需要
        `lambda` 函数或 ``PIL.Image``。
    """

    # 初始化方法，接收变换列表
    def __init__(self, transforms):
        # 存储变换列表
        self.transforms = transforms

    # 调用方法，依次应用每个变换
    def __call__(self, img, augmentation=False, box=None):
        # 遍历每个变换并应用于图像
        for t in self.transforms:
            img = t(img, augmentation, box)
        # 返回经过所有变换处理的图像
        return img


# 定义一个随机裁剪并插值的类
class RandomResizedCropAndInterpolationWithTwoPic:
    """将给定的 PIL 图像裁剪为随机大小和纵横比，并进行随机插值。
    从原始大小随机裁剪出一个随机大小（默认范围：0.08 到 1.0），
    以及原始纵横比的随机范围（默认范围：3/4 到 4/3）。
    该裁剪最终调整为给定大小。
    这在训练 Inception 网络时非常流行。
    Args:
        size: 每条边的预期输出大小
        scale: 原始裁剪大小的范围
        ratio: 原始裁剪纵横比的范围
        interpolation: 默认：PIL.Image.BILINEAR
    """
    # 初始化方法，设置图像处理的基本参数
        def __init__(self, size, second_size=None, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                     interpolation='bilinear', second_interpolation='lanczos'):
            # 如果 size 是元组，则直接赋值
            if isinstance(size, tuple):
                self.size = size
            # 否则，将其转换为正方形尺寸元组
            else:
                self.size = (size, size)
            # 如果提供了 second_size
            if second_size is not None:
                # 如果 second_size 是元组，则直接赋值
                if isinstance(second_size, tuple):
                    self.second_size = second_size
                # 否则，将其转换为正方形尺寸元组
                else:
                    self.second_size = (second_size, second_size)
            # 如果没有提供 second_size，则设置为 None
            else:
                self.second_size = None
            # 检查 scale 和 ratio 的范围是否合法
            if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
                warnings.warn("range should be of kind (min, max)")
    
            # 根据指定的插值方法进行初始化
            self.interpolation = _pil_interp(interpolation)
            # 初始化第二个插值方法
            self.second_interpolation = _pil_interp(second_interpolation)
            # 赋值缩放范围
            self.scale = scale
            # 赋值宽高比范围
            self.ratio = ratio
    
        @staticmethod
        # 获取随机尺寸裁剪的参数
        def get_params(img, scale, ratio):
            """Get parameters for ``crop`` for a random sized crop.
            Args:
                img (PIL Image): Image to be cropped.
                scale (tuple): range of size of the origin size cropped
                ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
            Returns:
                tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                    sized crop.
            """
            # 计算图像的总面积
            area = img.size[0] * img.size[1]
    
            # 尝试 10 次来获取有效的裁剪参数
            for attempt in range(10):
                # 在 scale 范围内随机选择目标面积
                target_area = random.uniform(*scale) * area
                # 计算宽高比的对数范围
                log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
                # 在对数范围内随机选择宽高比
                aspect_ratio = math.exp(random.uniform(*log_ratio))
    
                # 计算对应的宽和高
                w = int(round(math.sqrt(target_area * aspect_ratio)))
                h = int(round(math.sqrt(target_area / aspect_ratio)))
    
                # 如果计算得到的宽和高都在图像范围内
                if w <= img.size[0] and h <= img.size[1]:
                    # 随机选择裁剪区域的起始点
                    i = random.randint(0, img.size[1] - h)
                    j = random.randint(0, img.size[0] - w)
                    # 返回裁剪参数
                    return i, j, h, w
    
            # 如果未能找到合适的裁剪参数，则使用中心裁剪作为备用
            in_ratio = img.size[0] / img.size[1]
            # 根据输入图像的宽高比和最小宽高比计算宽高
            if in_ratio < min(ratio):
                w = img.size[0]
                h = int(round(w / min(ratio)))
            elif in_ratio > max(ratio):
                h = img.size[1]
                w = int(round(h * max(ratio)))
            else:  # 如果宽高比符合要求，保持原始大小
                w = img.size[0]
                h = img.size[1]
            # 计算中心裁剪的起始点
            i = (img.size[1] - h) // 2
            j = (img.size[0] - w) // 2
            # 返回裁剪参数
            return i, j, h, w
    # 定义一个可调用的类方法，接受图像和相关参数
        def __call__(self, img, augmentation=False, box=None):
            """
            参数:
                img (PIL Image): 需要裁剪和调整大小的图像。
            返回:
                PIL Image: 随机裁剪和调整大小后的图像。
            """
            # 如果启用数据增强
            if augmentation:
                # 获取裁剪参数：起始点i, j和高度h, 宽度w
                i, j, h, w = self.get_params(img, self.scale, self.ratio)
                # 根据裁剪参数对图像进行裁剪
                img = F.crop(img, i, j, h, w)
                # img, box = crop(img, i, j, h, w, box)  # 注释掉的裁剪盒子代码
            # 将图像调整到指定大小
            img = F.resize(img, self.size, self.interpolation)
            # 如果存在第二个大小，则调整第二个图像的大小，否则为None
            second_img = F.resize(img, self.second_size, self.second_interpolation) \
                if self.second_size is not None else None
            # 返回调整后的图像和第二个图像
            return img, second_img
    
        # 定义表示类的字符串表示方法
        def __repr__(self):
            # 如果插值方法是元组或列表，生成插值字符串
            if isinstance(self.interpolation, (tuple, list)):
                interpolate_str = ' '.join([_pil_interpolation_to_str[x] for x in self.interpolation])
            else:
                # 否则直接获取插值字符串
                interpolate_str = _pil_interpolation_to_str[self.interpolation]
            # 创建格式化字符串，包含类名和主要参数
            format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
            # 添加缩放参数
            format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
            # 添加比例参数
            format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
            # 添加插值参数
            format_string += ', interpolation={0}'.format(interpolate_str)
            # 如果存在第二个大小，添加其信息
            if self.second_size is not None:
                format_string += ', second_size={0}'.format(self.second_size)
                format_string += ', second_interpolation={0}'.format(_pil_interpolation_to_str[self.second_interpolation])
            # 结束格式化字符串并返回
            format_string += ')'
            return format_string
# 定义一个加载图像的函数，参数为图像文件路径
def pil_loader(path: str) -> Image.Image:
    # 以二进制模式打开指定路径的文件，避免资源警告
    with open(path, 'rb') as f:
        # 使用打开的文件对象加载图像
        img = Image.open(f)
        # 将图像转换为 RGB 格式并返回
        return img.convert('RGB')
```