# `.\MinerU\magic_pdf\model\pek_sub_modules\layoutlmv3\visualizer.py`

```
# 版权声明，说明此代码归 Facebook, Inc. 及其关联公司所有
# 导入颜色处理模块
import colorsys
# 导入日志模块
import logging
# 导入数学模块
import math
# 导入 NumPy 库
import numpy as np
# 从 enum 模块导入 Enum 和 unique，用于创建枚举类
from enum import Enum, unique
# 导入 OpenCV 库
import cv2
# 导入 Matplotlib 库及其相关模块
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
# 导入 COCO 格式的掩膜处理工具
import pycocotools.mask as mask_util
# 导入 PyTorch 库
import torch
# 导入 Matplotlib 的后端以支持图形绘制
from matplotlib.backends.backend_agg import FigureCanvasAgg
# 导入图像处理库
from PIL import Image

# 从 detectron2 数据集模块导入元数据目录
from detectron2.data import MetadataCatalog
# 从 detectron2 结构模块导入多种结构类型
from detectron2.structures import BitMasks, Boxes, BoxMode, Keypoints, PolygonMasks, RotatedBoxes
# 从 detectron2 文件输入输出模块导入路径管理工具
from detectron2.utils.file_io import PathManager

# 从 detectron2 颜色映射模块导入随机颜色生成工具
from detectron2.utils.colormap import random_color

# 导入 Python 调试工具
import pdb

# 创建一个日志记录器
logger = logging.getLogger(__name__)

# 公开模块中的类和变量
__all__ = ["ColorMode", "VisImage", "Visualizer"]

# 定义小物体面积阈值
_SMALL_OBJECT_AREA_THRESH = 1000
# 定义大掩膜面积阈值
_LARGE_MASK_AREA_THRESH = 120000
# 定义接近白色的颜色
_OFF_WHITE = (1.0, 1.0, 240.0 / 255)
# 定义黑色
_BLACK = (0, 0, 0)
# 定义红色
_RED = (1.0, 0, 0)

# 定义关键点阈值
_KEYPOINT_THRESHOLD = 0.05

# CLASS_NAMES = ["footnote", "footer", "header"]

@unique
class ColorMode(Enum):
    """
    不同颜色模式的枚举，用于实例可视化。
    """

    IMAGE = 0
    """
    为每个实例选择随机颜色，并以低不透明度叠加分割。
    """
    SEGMENTATION = 1
    """
    同一类别的实例具有相似颜色
    （来自 metadata.thing_colors），并以高不透明度叠加。
    这提高了对分割质量的关注。
    """
    IMAGE_BW = 2
    """
    与 IMAGE 相同，但将没有掩膜的区域转换为灰度。
    仅适用于绘制每个实例的掩膜预测。
    """


class GenericMask:
    """
    属性：
        polygons (list[ndarray]): 此掩膜的多边形列表。
            每个 ndarray 的格式为 [x, y, x, y, ...]
        mask (ndarray): 二进制掩膜
    """

    def __init__(self, mask_or_polygons, height, width):
        # 初始化掩膜和多边形属性
        self._mask = self._polygons = self._has_holes = None
        # 设置掩膜的高度和宽度
        self.height = height
        self.width = width

        # 将输入赋值给变量 m
        m = mask_or_polygons
        if isinstance(m, dict):
            # 处理 RLE（游程长度编码）
            assert "counts" in m and "size" in m
            if isinstance(m["counts"], list):  # 处理未压缩的 RLE
                h, w = m["size"]
                assert h == height and w == width
                # 将 RLE 转换为多边形对象
                m = mask_util.frPyObjects(m, h, w)
            # 解码 RLE 得到掩膜
            self._mask = mask_util.decode(m)[:, :]
            return

        if isinstance(m, list):  # 处理多边形列表
            # 将每个多边形转换为 ndarray 并重塑
            self._polygons = [np.asarray(x).reshape(-1) for x in m]
            return

        if isinstance(m, np.ndarray):  # 假设为二进制掩膜
            assert m.shape[1] != 2, m.shape
            assert m.shape == (
                height,
                width,
            ), f"mask shape: {m.shape}, target dims: {height}, {width}"
            # 将掩膜转换为无符号8位整数
            self._mask = m.astype("uint8")
            return

        # 如果输入类型不匹配，抛出错误
        raise ValueError("GenericMask cannot handle object {} of type '{}'".format(m, type(m)))

    @property
    # 定义 mask 方法，用于生成或返回当前对象的遮罩
        def mask(self):
            # 如果遮罩尚未初始化，则调用 polygons_to_mask 方法生成遮罩
            if self._mask is None:
                self._mask = self.polygons_to_mask(self._polygons)
            # 返回当前对象的遮罩
            return self._mask
    
        # 定义 polygons 属性，获取多边形数据
        @property
        def polygons(self):
            # 如果多边形尚未初始化，则根据遮罩生成多边形数据
            if self._polygons is None:
                self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
            # 返回当前对象的多边形数据
            return self._polygons
    
        # 定义 has_holes 属性，检查多边形是否有孔
        @property
        def has_holes(self):
            # 如果孔的状态尚未初始化
            if self._has_holes is None:
                # 如果遮罩存在，则根据遮罩生成多边形和孔的状态
                if self._mask is not None:
                    self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
                else:
                    # 如果原始格式是多边形，则默认为没有孔
                    self._has_holes = False  
            # 返回孔的状态
            return self._has_holes
    
        # 定义 mask_to_polygons 方法，将遮罩转换为多边形
        def mask_to_polygons(self, mask):
            # 使用 cv2.RETR_CCOMP 标志检索所有轮廓并组织成两级层次结构
            # 外部轮廓放在层次结构的第一层，内部轮廓放在第二层
            # cv2.CHAIN_APPROX_NONE 标志获取轮廓的多边形顶点
            mask = np.ascontiguousarray(mask)  # 将遮罩转换为连续数组格式以支持某些 cv2 版本
            # 查找轮廓并返回结果，结果的最后一个元素是层次结构
            res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            hierarchy = res[-1]
            # 如果层次结构为空，返回空多边形和没有孔的状态
            if hierarchy is None:  
                return [], False
            # 检查是否存在孔，通过层次结构判断
            has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
            res = res[-2]  # 获取轮廓结果
            res = [x.flatten() for x in res]  # 将轮廓扁平化
            # 将坐标转换为实值坐标空间
            res = [x + 0.5 for x in res if len(x) >= 6]
            # 返回多边形和孔的状态
            return res, has_holes
    
        # 定义 polygons_to_mask 方法，将多边形转换为遮罩
        def polygons_to_mask(self, polygons):
            # 将多边形转换为 RLE（游程编码）
            rle = mask_util.frPyObjects(polygons, self.height, self.width)
            # 合并 RLE 数据
            rle = mask_util.merge(rle)
            # 解码 RLE 数据，返回遮罩
            return mask_util.decode(rle)[:, :]
    
        # 定义 area 方法，计算遮罩的面积
        def area(self):
            # 返回遮罩中所有像素的总和，即面积
            return self.mask.sum()
    
        # 定义 bbox 方法，获取多边形的边界框
        def bbox(self):
            # 将多边形转换为 RLE 格式
            p = mask_util.frPyObjects(self.polygons, self.height, self.width)
            # 合并 RLE 数据
            p = mask_util.merge(p)
            # 将 RLE 转换为边界框
            bbox = mask_util.toBbox(p)
            # 计算边界框的实际坐标
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            # 返回边界框
            return bbox
# 定义一个类，用于统一不同的全景注释/预测格式
class _PanopticPrediction:
    """
    统一不同的全景注释/预测格式
    """

    # 初始化方法，接受全景分割、分段信息和可选元数据
    def __init__(self, panoptic_seg, segments_info, metadata=None):
        # 如果分段信息为 None，确保元数据不为 None
        if segments_info is None:
            assert metadata is not None
            # 如果 "segments_info" 为 None，假设 "panoptic_img" 是一个
            # H*W int32 图像，存储 panoptic_id，格式为
            # category_id * label_divisor + instance_id。我们保留 -1 作为
            # VOID 标签。
            label_divisor = metadata.label_divisor
            segments_info = []
            # 遍历 panoptic_seg 中唯一的 panoptic_label
            for panoptic_label in np.unique(panoptic_seg.numpy()):
                if panoptic_label == -1:
                    # VOID 区域，跳过
                    continue
                # 计算预测的类别
                pred_class = panoptic_label // label_divisor
                # 判断是否为物体
                isthing = pred_class in metadata.thing_dataset_id_to_contiguous_id.values()
                # 添加分段信息到列表
                segments_info.append(
                    {
                        "id": int(panoptic_label),          # 分段 ID
                        "category_id": int(pred_class),     # 类别 ID
                        "isthing": bool(isthing),           # 是否为物体
                    }
                )
        # 删除元数据以释放内存
        del metadata

        # 保存全景分割
        self._seg = panoptic_seg

        # 将分段信息转换为字典，seg id -> seg info
        self._sinfo = {s["id"]: s for s in segments_info}  
        # 获取唯一的分段 ID 和对应的区域面积
        segment_ids, areas = torch.unique(panoptic_seg, sorted=True, return_counts=True)
        areas = areas.numpy()  # 转换为 NumPy 数组
        # 按面积降序排列分段 ID
        sorted_idxs = np.argsort(-areas)
        self._seg_ids, self._seg_areas = segment_ids[sorted_idxs], areas[sorted_idxs]
        # 将分段 ID 转换为列表
        self._seg_ids = self._seg_ids.tolist()
        # 为每个分段 ID 赋予面积
        for sid, area in zip(self._seg_ids, self._seg_areas):
            if sid in self._sinfo:
                self._sinfo[sid]["area"] = float(area)  # 添加面积信息

    # 返回一个掩码，表示所有有预测的像素
    def non_empty_mask(self):
        """
        返回:
            (H, W) 数组，表示所有有预测的像素的掩码
        """
        empty_ids = []  # 初始化空 ID 列表
        # 检查每个分段 ID 是否在分段信息中
        for id in self._seg_ids:
            if id not in self._sinfo:
                empty_ids.append(id)  # 添加到空 ID 列表
        # 如果没有空 ID，返回全零的掩码
        if len(empty_ids) == 0:
            return np.zeros(self._seg.shape, dtype=np.uint8)
        # 确保只有一个空 ID，不支持多个
        assert (
            len(empty_ids) == 1
        ), ">1 ids corresponds to no labels. This is currently not supported"
        # 返回掩码，表示所有非空像素
        return (self._seg != empty_ids[0]).numpy().astype(np.bool)

    # 生成语义掩码
    def semantic_masks(self):
        for sid in self._seg_ids:
            sinfo = self._sinfo.get(sid)  # 获取分段信息
            if sinfo is None or sinfo["isthing"]:
                # 一些像素（例如 PanopticFPN 中的 ID 0）没有实例或语义预测，跳过
                continue
            # 生成语义掩码和相应的信息
            yield (self._seg == sid).numpy().astype(np.bool), sinfo

    # 生成实例掩码
    def instance_masks(self):
        for sid in self._seg_ids:
            sinfo = self._sinfo.get(sid)  # 获取分段信息
            if sinfo is None or not sinfo["isthing"]:
                continue  # 如果不是物体，跳过
            mask = (self._seg == sid).numpy().astype(np.bool)  # 创建实例掩码
            if mask.sum() > 0:  # 如果掩码中有像素
                yield mask, sinfo  # 生成掩码和相应的信息
# 创建文本标签，根据输入的类别、分数和类名生成标签
def _create_text_labels(classes, scores, class_names, is_crowd=None):
    """
    参数:
        classes (list[int] or None): 类别索引列表
        scores (list[float] or None): 分数列表
        class_names (list[str] or None): 类别名称列表
        is_crowd (list[bool] or None): 表示是否为人群的布尔列表

    返回:
        list[str] or None: 生成的标签列表或 None
    """
    # 初始化标签为 None
    labels = None
    # 如果类别列表不为 None
    if classes is not None:
        # 如果类名列表不为 None 且不为空
        if class_names is not None and len(class_names) > 0:
            # 根据类别索引生成对应的类名标签
            labels = [class_names[i] for i in classes]
        else:
            # 如果没有类名，则直接使用类别索引转换为字符串
            labels = [str(i) for i in classes]
            
    # 如果分数列表不为 None
    if scores is not None:
        # 如果标签为 None，直接生成分数标签
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            # 将标签和分数结合，生成“标签 分数”的格式
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    # 如果标签和人群信息都不为 None
    if labels is not None and is_crowd is not None:
        # 在标签中添加人群标识
        labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
    # 返回生成的标签
    return labels


# 可视化图像类
class VisImage:
    # 初始化方法
    def __init__(self, img, scale=1.0):
        """
        参数:
            img (ndarray): 一个形状为 (H, W, 3) 的 RGB 图像，值范围 [0, 255]。
            scale (float): 输入图像的缩放因子
        """
        # 保存输入图像和缩放因子
        self.img = img
        self.scale = scale
        # 获取图像的宽度和高度
        self.width, self.height = img.shape[1], img.shape[0]
        # 设置图像的绘图结构
        self._setup_figure(img)

    # 设置图像绘制的图形
    def _setup_figure(self, img):
        """
        参数:
            同 :meth:`__init__()`.

        返回:
            fig (matplotlib.pyplot.figure): 所有图像绘图元素的顶层容器。
            ax (matplotlib.pyplot.Axes): 包含图形元素并设置坐标系统。
        """
        # 创建一个没有边框的图形
        fig = mplfigure.Figure(frameon=False)
        # 获取图形的 dpi
        self.dpi = fig.get_dpi()
        # 设置图形大小，添加 1e-2 避免精度损失
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        # 创建图形的画布
        self.canvas = FigureCanvasAgg(fig)
        # ax 用于添加坐标轴
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        # 关闭坐标轴
        ax.axis("off")
        # 保存图形和坐标轴对象
        self.fig = fig
        self.ax = ax
        # 重置图像
        self.reset_image(img)

    # 重置图像方法
    def reset_image(self, img):
        """
        参数:
            img: 与 __init__ 中相同
        """
        # 将图像数据类型转换为 uint8
        img = img.astype("uint8")
        # 在坐标轴上显示图像
        self.ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

    # 保存图像方法
    def save(self, filepath):
        """
        参数:
            filepath (str): 包含绝对路径及文件名的字符串，指定保存可视化图像的位置。
        """
        # 保存图形到指定文件路径
        self.fig.savefig(filepath)
    # 定义一个获取图像的方法
    def get_image(self):
        """
        返回值:
            ndarray:
                返回形状为 (H, W, 3) (RGB) 的可视化图像，数据类型为 uint8。
                图像的形状根据输入图像和给定的 `scale` 参数进行缩放。
        """
        # 获取当前画布对象
        canvas = self.canvas
        # 将画布内容打印到缓冲区，并获取其宽度和高度
        s, (width, height) = canvas.print_to_buffer()
        # buf = io.BytesIO()  # 对于 cairo 后端可用
        # canvas.print_rgba(buf)  # 将 RGBA 图像打印到缓冲区
        # width, height = self.width, self.height  # 获取图像的宽度和高度
        # s = buf.getvalue()  # 从缓冲区获取字节内容
    
        # 从缓冲区字节转换为 uint8 类型的 NumPy 数组
        buffer = np.frombuffer(s, dtype="uint8")
    
        # 将缓冲区数组重塑为指定高度和宽度，并包含 RGBA 通道
        img_rgba = buffer.reshape(height, width, 4)
        # 将 RGBA 数组分割为 RGB 和 Alpha 通道
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        # 返回 RGB 数组，并转换为 uint8 类型
        return rgb.astype("uint8")
# 定义可视化类，用于在图像上绘制检测/分割数据
class Visualizer:
    """
    Visualizer that draws data about detection/segmentation on images.

    It contains methods like `draw_{text,box,circle,line,binary_mask,polygon}`
    that draw primitive objects to images, as well as high-level wrappers like
    `draw_{instance_predictions,sem_seg,panoptic_seg_predictions,dataset_dict}`
    that draw composite data in some pre-defined style.

    Note that the exact visualization style for the high-level wrappers are subject to change.
    Style such as color, opacity, label contents, visibility of labels, or even the visibility
    of objects themselves (e.g. when the object is too small) may change according
    to different heuristics, as long as the results still look visually reasonable.

    To obtain a consistent style, you can implement custom drawing functions with the
    abovementioned primitive methods instead. If you need more customized visualization
    styles, you can process the data yourself following their format documented in
    tutorials (:doc:`/tutorials/models`, :doc:`/tutorials/datasets`). This class does not
    intend to satisfy everyone's preference on drawing styles.

    This visualizer focuses on high rendering quality rather than performance. It is not
    designed to be used for real-time applications.
    """

    # TODO implement a fast, rasterized version using OpenCV
    # TODO: 实现一个使用 OpenCV 的快速光栅化版本

    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE):
        """
        Args:
            img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
            metadata (Metadata): dataset metadata (e.g. class names and colors)
            instance_mode (ColorMode): defines one of the pre-defined style for drawing
                instances on an image.
        """
        # 将输入的 RGB 图像转换为 NumPy 数组，限制在 0 到 255 之间，并转换为无符号整型
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        # 如果没有提供元数据，则使用默认的元数据
        if metadata is None:
            metadata = MetadataCatalog.get("__nonexist__")
        # 保存元数据
        self.metadata = metadata
        # 使用给定的缩放因子创建可视化图像对象
        self.output = VisImage(self.img, scale=scale)
        # 设置设备为 CPU
        self.cpu_device = torch.device("cpu")

        # 文本大小太小是无用的，因此限制为最小 9
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 90, 10 // scale
        )
        # 设置实例模式
        self._instance_mode = instance_mode
        # 设置关键点阈值
        self.keypoint_threshold = _KEYPOINT_THRESHOLD
    # 定义一个方法，用于在图像上绘制实例级别的预测结果
    def draw_instance_predictions(self, predictions):
        """
        在图像上绘制实例级别的预测结果。

        参数:
            predictions (Instances): 实例检测/分割模型的输出。
                以下字段将用于绘制：
                "pred_boxes", "pred_classes", "scores", "pred_masks"（或 "pred_masks_rle"）。

        返回:
            output (VisImage): 带有可视化结果的图像对象。
        """
        # 如果预测结果中有 "pred_boxes"，则获取边界框；否则为 None
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        # 如果预测结果中有 "scores"，则获取分数；否则为 None
        scores = predictions.scores if predictions.has("scores") else None
        # 如果预测结果中有 "pred_classes"，则将类转换为列表；否则为 None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        # 创建文本标签，用于显示类别和分数
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        # 如果预测结果中有 "pred_keypoints"，则获取关键点；否则为 None
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        # 如果预测结果中有 "pred_masks"
        if predictions.has("pred_masks"):
            # 将预测的掩码转换为 NumPy 数组
            masks = np.asarray(predictions.pred_masks)
            # 将每个掩码转换为 GenericMask 对象
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            # 如果没有掩码，则设置为 None
            masks = None

        # 如果实例模式为 SEGMENTATION 且存在类的颜色信息
        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            # 为每个类生成抖动的颜色
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            # 设置透明度
            alpha = 0.8
        else:
            # 否则颜色设置为 None，透明度为 0.5
            colors = None
            alpha = 0.5

        # 如果实例模式为 IMAGE_BW
        if self._instance_mode == ColorMode.IMAGE_BW:
            # 重置图像为灰度图像，基于掩码的存在与否
            self.output.reset_image(
                self._create_grayscale_image(
                    (predictions.pred_masks.any(dim=0) > 0).numpy()
                    if predictions.has("pred_masks")
                    else None
                )
            )
            # 设置透明度
            alpha = 0.3

        # 将掩码、边界框、标签、关键点和颜色等信息叠加到图像上
        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        # 返回包含可视化结果的图像对象
        return self.output
    # 定义绘制语义分割预测/标签的函数
    def draw_sem_seg(self, sem_seg, area_threshold=None, alpha=0.8):
        """
        绘制语义分割预测/标签。

        参数：
            sem_seg (Tensor 或 ndarray): 形状为 (H, W) 的分割图。
                每个值是像素的整数标签。
            area_threshold (int): 面积小于 `area_threshold` 的区域不绘制。
            alpha (float): 值越大，分割越不透明。

        返回：
            output (VisImage): 带有可视化的图像对象。
        """
        # 检查输入是否为 PyTorch 张量，如果是，则转换为 NumPy 数组
        if isinstance(sem_seg, torch.Tensor):
            sem_seg = sem_seg.numpy()
        # 获取唯一标签及其对应的面积
        labels, areas = np.unique(sem_seg, return_counts=True)
        # 按照面积降序排序索引
        sorted_idxs = np.argsort(-areas).tolist()
        # 根据排序后的索引重新排列标签
        labels = labels[sorted_idxs]
        # 遍历所有有效标签（小于类的数量）
        for label in filter(lambda l: l < len(self.metadata.stuff_classes), labels):
            try:
                # 尝试获取当前标签的颜色，并进行归一化处理
                mask_color = [x / 255 for x in self.metadata.stuff_colors[label]]
            except (AttributeError, IndexError):
                # 如果获取颜色失败，则将颜色设为 None
                mask_color = None

            # 创建当前标签的二进制掩码
            binary_mask = (sem_seg == label).astype(np.uint8)
            # 获取当前标签对应的类名
            text = self.metadata.stuff_classes[label]
            # 调用绘制二进制掩码的函数，传入相应参数
            self.draw_binary_mask(
                binary_mask,
                color=mask_color,
                edge_color=_OFF_WHITE,
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )
        # 返回带有可视化的输出图像
        return self.output
    # 定义一个绘制全景分割结果的函数，接受全景分割图、段信息、面积阈值和透明度
    def draw_panoptic_seg(self, panoptic_seg, segments_info, area_threshold=None, alpha=0.7):
        """
        绘制全景预测注释或结果。

        参数：
            panoptic_seg (Tensor): 形状为 (height, width)，每个值为对应段的 id。
            segments_info (list[dict] or None): 描述 `panoptic_seg` 中的每个段。
                如果是 ``list[dict]``，每个字典包含 "id" 和 "category_id" 键。
                如果为 None，则通过
                ``pixel // metadata.label_divisor`` 计算每个像素的类别 id。
            area_threshold (int): 面积小于 `area_threshold` 的段不被绘制。

        返回：
            output (VisImage): 包含可视化结果的图像对象。
        """
        # 创建一个全景预测对象，使用给定的分割图和段信息
        pred = _PanopticPrediction(panoptic_seg, segments_info, self.metadata)

        # 如果实例模式为黑白图像模式，重置输出图像为灰度图像
        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(self._create_grayscale_image(pred.non_empty_mask()))

        # 首先绘制所有语义段的掩码，即“物体”
        for mask, sinfo in pred.semantic_masks():
            # 获取当前段的类别索引
            category_idx = sinfo["category_id"]
            try:
                # 从元数据中获取段的颜色并归一化
                mask_color = [x / 255 for x in self.metadata.stuff_colors[category_idx]]
            except AttributeError:
                # 如果没有颜色属性，则设置为 None
                mask_color = None

            # 获取当前段的类别文本
            text = self.metadata.stuff_classes[category_idx]
            # 绘制二进制掩码，传入颜色、边缘颜色、文本、透明度和面积阈值
            self.draw_binary_mask(
                mask,
                color=mask_color,
                edge_color=_OFF_WHITE,
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )

        # 然后绘制所有实例的掩码
        all_instances = list(pred.instance_masks())
        # 如果没有实例，直接返回输出
        if len(all_instances) == 0:
            return self.output
        # 解压实例掩码和段信息
        masks, sinfo = list(zip(*all_instances))
        # 提取每个实例的类别 id
        category_ids = [x["category_id"] for x in sinfo]

        try:
            # 提取每个实例的得分
            scores = [x["score"] for x in sinfo]
        except KeyError:
            # 如果没有得分，设置为 None
            scores = None
        # 创建文本标签，包含类别 id、得分和是否为人群的标记
        labels = _create_text_labels(
            category_ids, scores, self.metadata.thing_classes, [x.get("iscrowd", 0) for x in sinfo]
        )

        try:
            # 从元数据中获取实例的颜色，并进行随机抖动处理
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in category_ids
            ]
        except AttributeError:
            # 如果没有颜色属性，则设置为 None
            colors = None
        # 将实例掩码叠加到输出图像上
        self.overlay_instances(masks=masks, labels=labels, assigned_colors=colors, alpha=alpha)

        # 返回可视化结果
        return self.output

    # 为了向后兼容，将绘制全景分割的函数赋值给另一个名称
    draw_panoptic_seg_predictions = draw_panoptic_seg  # backward compatibility
    # 定义一个方法用于绘制 Detectron2 数据集格式的注释/分割
        def draw_dataset_dict(self, dic):
            """
            绘制 Detectron2 数据集格式中的注释/分割。
    
            参数：
                dic (dict): 单幅图像的注释/分割数据，采用 Detectron2 数据集格式。
    
            返回：
                output (VisImage): 带有可视化的图像对象。
            """
            # 从字典中获取注释，如果不存在则为 None
            annos = dic.get("annotations", None)
            # 如果注释存在
            if annos:
                # 如果第一条注释包含分割信息，则提取所有分割掩码
                if "segmentation" in annos[0]:
                    masks = [x["segmentation"] for x in annos]
                else:
                    masks = None
                # 如果第一条注释包含关键点信息，则提取所有关键点并重塑数组形状
                if "keypoints" in annos[0]:
                    keypts = [x["keypoints"] for x in annos]
                    keypts = np.array(keypts).reshape(len(annos), -1, 3)
                else:
                    keypts = None
    
                # 转换每个注释的边界框格式
                boxes = [
                    BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS)
                    if len(x["bbox"]) == 4
                    else x["bbox"]
                    for x in annos
                ]
    
                # 初始化颜色为 None
                colors = None
                # 提取每个注释的类别 ID
                category_ids = [x["category_id"] for x in annos]
                # 如果实例模式为分割并且有颜色元数据，则生成抖动后的颜色列表
                if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
                    colors = [
                        self._jitter([x / 255 for x in self.metadata.thing_colors[c]])
                        for c in category_ids
                    ]
                # 获取类别名称
                names = self.metadata.get("thing_classes", None)
                # 创建文本标签，包含类别 ID、分数、类别名称和是否为拥挤
                labels = _create_text_labels(
                    category_ids,
                    scores=None,
                    class_names=names,
                    is_crowd=[x.get("iscrowd", 0) for x in annos],
                )
                # 在图像上叠加实例信息
                self.overlay_instances(
                    labels=labels, boxes=boxes, masks=masks, keypoints=keypts, assigned_colors=colors
                )
    
            # 从字典中获取语义分割信息，如果不存在且有对应文件名，则读取图像
            sem_seg = dic.get("sem_seg", None)
            if sem_seg is None and "sem_seg_file_name" in dic:
                with PathManager.open(dic["sem_seg_file_name"], "rb") as f:
                    sem_seg = Image.open(f)  # 打开语义分割图像
                    sem_seg = np.asarray(sem_seg, dtype="uint8")  # 转换为数组格式
            # 如果语义分割信息存在，则绘制
            if sem_seg is not None:
                self.draw_sem_seg(sem_seg, area_threshold=0, alpha=0.5)
    
            # 从字典中获取全景分割信息，如果不存在且有对应文件名，则读取图像
            pan_seg = dic.get("pan_seg", None)
            if pan_seg is None and "pan_seg_file_name" in dic:
                with PathManager.open(dic["pan_seg_file_name"], "rb") as f:
                    pan_seg = Image.open(f)  # 打开全景分割图像
                    pan_seg = np.asarray(pan_seg)  # 转换为数组格式
                    from panopticapi.utils import rgb2id  # 导入颜色到 ID 的转换函数
    
                    pan_seg = rgb2id(pan_seg)  # 将 RGB 格式转换为 ID 格式
            # 如果全景分割信息存在，则绘制全景分割
            if pan_seg is not None:
                segments_info = dic["segments_info"]  # 获取分段信息
                pan_seg = torch.tensor(pan_seg)  # 将全景分割转换为张量格式
                self.draw_panoptic_seg(pan_seg, segments_info, area_threshold=0, alpha=0.5)  # 绘制全景分割
            # 返回输出对象
            return self.output
    
    # 定义方法用于叠加实例信息到图像
        def overlay_instances(
            self,
            *,
            boxes=None,
            labels=None,
            masks=None,
            keypoints=None,
            assigned_colors=None,
            alpha=0.5,
    # 定义一个函数，用于叠加旋转实例
    def overlay_rotated_instances(self, boxes=None, labels=None, assigned_colors=None):
        """
        参数:
            boxes (ndarray): Nx5的numpy数组，格式为
                (x_center, y_center, width, height, angle_degrees)，表示单张图像中的N个对象。
            labels (list[str]): 每个实例要显示的文本。
            assigned_colors (list[matplotlib.colors]): 每个掩码或框对应的颜色列表。请参阅'matplotlib.colors'获取可接受颜色格式的完整列表。
    
        返回:
            output (VisImage): 带有可视化效果的图像对象。
        """
        # 计算实例数量
        num_instances = len(boxes)
    
        # 如果未指定颜色，生成随机颜色列表
        if assigned_colors is None:
            assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
        # 如果没有实例，直接返回输出
        if num_instances == 0:
            return self.output
    
        # 根据面积从大到小显示，减少遮挡
        if boxes is not None:
            areas = boxes[:, 2] * boxes[:, 3]
    
        # 按面积降序排序索引
        sorted_idxs = np.argsort(-areas).tolist()
        # 重新排序重叠实例，按降序排列
        boxes = boxes[sorted_idxs]
        labels = [labels[k] for k in sorted_idxs] if labels is not None else None
        colors = [assigned_colors[idx] for idx in sorted_idxs]
    
        # 遍历所有实例并绘制旋转框和标签
        for i in range(num_instances):
            self.draw_rotated_box_with_label(
                boxes[i], edge_color=colors[i], label=labels[i] if labels is not None else None
            )
    
        # 返回可视化效果的图像
        return self.output
    # 定义一个方法来绘制关键点并连接合适的关键点
        def draw_and_connect_keypoints(self, keypoints):
            """
            绘制实例的关键点，并根据关键点连接规则绘制线条，线条颜色根据颜色启发式确定。
    
            参数：
                keypoints (Tensor): 形状为 (K, 3) 的张量，其中 K 是关键点数量，
                    最后一维对应 (x, y, 概率)。
    
            返回：
                output (VisImage): 带有可视化的图像对象。
            """
            # 创建一个字典来存储可见的关键点
            visible = {}
            # 从元数据中获取关键点名称
            keypoint_names = self.metadata.get("keypoint_names")
            # 遍历每个关键点及其索引
            for idx, keypoint in enumerate(keypoints):
                # 解包关键点的 x, y 坐标和概率
                x, y, prob = keypoint
                # 如果概率大于阈值，则绘制关键点
                if prob > self.keypoint_threshold:
                    self.draw_circle((x, y), color=_RED)
                    # 如果有关键点名称，则记录该关键点的名称和坐标
                    if keypoint_names:
                        keypoint_name = keypoint_names[idx]
                        visible[keypoint_name] = (x, y)
    
            # 如果有关键点连接规则，则进行连接绘制
            if self.metadata.get("keypoint_connection_rules"):
                # 遍历所有连接规则
                for kp0, kp1, color in self.metadata.keypoint_connection_rules:
                    # 如果两个关键点都可见，则绘制连接线
                    if kp0 in visible and kp1 in visible:
                        x0, y0 = visible[kp0]
                        x1, y1 = visible[kp1]
                        color = tuple(x / 255.0 for x in color)
                        self.draw_line([x0, x1], [y0, y1], color=color)
    
            # 从鼻子到中肩和中肩到中臀的连接线绘制
            # 注意：该策略特定于人类关键点
            try:
                # 获取左肩和右肩的坐标
                ls_x, ls_y = visible["left_shoulder"]
                rs_x, rs_y = visible["right_shoulder"]
                # 计算中肩的坐标
                mid_shoulder_x, mid_shoulder_y = (ls_x + rs_x) / 2, (ls_y + rs_y) / 2
            except KeyError:
                # 如果没有找到肩部关键点，则跳过
                pass
            else:
                # 从鼻子到中肩绘制连接线
                nose_x, nose_y = visible.get("nose", (None, None))
                if nose_x is not None:
                    self.draw_line([nose_x, mid_shoulder_x], [nose_y, mid_shoulder_y], color=_RED)
    
                try:
                    # 从中肩到中臀的连接线绘制
                    lh_x, lh_y = visible["left_hip"]
                    rh_x, rh_y = visible["right_hip"]
                except KeyError:
                    # 如果没有找到臀部关键点，则跳过
                    pass
                else:
                    # 计算中臀的坐标
                    mid_hip_x, mid_hip_y = (lh_x + rh_x) / 2, (lh_y + rh_y) / 2
                    self.draw_line([mid_hip_x, mid_shoulder_x], [mid_hip_y, mid_shoulder_y], color=_RED)
            # 返回绘制结果
            return self.output
    
        """
        原始绘制函数：
        """
    
        # 定义一个方法来绘制文本
        def draw_text(
            self,
            text,
            position,
            *,
            font_size=None,
            color="g",
            horizontal_alignment="center",
            rotation=0,
    ):
        """
        Args:
            text (str): class label  # 文本标签，表示要绘制的内容
            position (tuple): a tuple of the x and y coordinates to place text on image.  # 文本在图像上的位置坐标
            font_size (int, optional): font of the text. If not provided, a font size  # 文本字体大小，如果未提供，将根据图像宽度计算并使用
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list  # 文本颜色，参考 `matplotlib.colors` 获取可接受的格式列表
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`  # 水平对齐方式，参考 `matplotlib.text.Text` 文档
            rotation: rotation angle in degrees CCW  # 文本旋转角度，以逆时针方向的度数表示

        Returns:
            output (VisImage): image object with text drawn.  # 返回绘制文本后的图像对象
        """
        if not font_size:  # 如果未提供字体大小
            font_size = self._default_font_size  # 使用默认字体大小

        # since the text background is dark, we don't want the text to be dark  # 由于文本背景较暗，因此不希望文本颜色也较暗
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)  # 将颜色转换为 RGB，并确保最小值为 0.2
        color[np.argmax(color)] = max(0.8, np.max(color))  # 将最大颜色分量设置为不小于 0.8 的值

        x, y = position  # 解包文本位置坐标
        self.output.ax.text(  # 在输出图像的轴上绘制文本
            x,
            y,
            text,  # 要绘制的文本内容
            size=font_size * self.output.scale,  # 文本大小，考虑图像缩放
            family="sans-serif",  # 字体族设置为无衬线字体
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},  # 设置文本的边框属性
            verticalalignment="top",  # 文本垂直对齐方式设置为顶部对齐
            horizontalalignment=horizontal_alignment,  # 文本水平对齐方式
            color=color,  # 文本颜色
            zorder=10,  # 图层顺序，确保文本在其他元素之上
            rotation=rotation,  # 文本旋转角度
        )
        return self.output  # 返回绘制后的图像对象

    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):  # 定义绘制框的函数
        """
        Args:
            box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0  # 坐标元组，包含框的左上角和右下角坐标
                are the coordinates of the image's top left corner. x1 and y1 are the
                coordinates of the image's bottom right corner.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.  # 透明度参数，值越小越透明
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`  # 边框颜色，参考 `matplotlib.colors` 获取可接受的格式
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.  # 边框样式的字符串

        Returns:
            output (VisImage): image object with box drawn.  # 返回绘制框后的图像对象
        """
        x0, y0, x1, y1 = box_coord  # 解包框的四个坐标
        width = x1 - x0  # 计算框的宽度
        height = y1 - y0  # 计算框的高度

        linewidth = max(self._default_font_size / 4, 1)  # 计算边框线宽，确保不小于 1

        self.output.ax.add_patch(  # 在输出图像的轴上添加矩形补丁
            mpl.patches.Rectangle(  # 创建一个矩形补丁
                (x0, y0),  # 矩形左下角坐标
                width,  # 矩形的宽度
                height,  # 矩形的高度
                fill=False,  # 不填充矩形
                edgecolor=edge_color,  # 设置矩形的边框颜色
                linewidth=linewidth * self.output.scale,  # 根据图像缩放设置边框线宽
                alpha=alpha,  # 设置边框透明度
                linestyle=line_style,  # 设置边框样式
            )
        )
        return self.output  # 返回绘制后的图像对象

    def draw_rotated_box_with_label(  # 定义绘制带标签的旋转框的函数
        self, rotated_box, alpha=0.5, edge_color="g", line_style="-", label=None  # 参数包括旋转框、透明度、边框颜色、边框样式和标签
    ):
        """
        绘制一个旋转的框，并在左上角添加标签。

        参数:
            rotated_box (tuple): 包含 (cnt_x, cnt_y, w, h, angle) 的元组，
                其中 cnt_x 和 cnt_y 是框的中心坐标。
                w 和 h 是框的宽度和高度。angle 表示框相对于 0 度的逆时针旋转角度。
            alpha (float): 混合效率。较小的值会导致更透明的遮罩。
            edge_color: 框的轮廓颜色。有关可接受格式的完整列表，请参见 `matplotlib.colors`。
            line_style (string): 用于创建框轮廓的字符串。
            label (string): 旋转框的标签。设置为 None 时不会渲染。

        返回:
            output (VisImage): 绘制框的图像对象。
        """
        # 解包旋转框的参数
        cnt_x, cnt_y, w, h, angle = rotated_box
        # 计算框的面积
        area = w * h
        # 当框较小时使用更细的线条
        linewidth = self._default_font_size / (
            6 if area < _SMALL_OBJECT_AREA_THRESH * self.output.scale else 3
        )

        # 将角度转换为弧度
        theta = angle * math.pi / 180.0
        # 计算旋转角度的余弦和正弦值
        c = math.cos(theta)
        s = math.sin(theta)
        # 定义未旋转的矩形的四个顶点
        rect = [(-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2)]
        # 计算旋转后的矩形的顶点坐标
        rotated_rect = [(s * yy + c * xx + cnt_x, c * yy - s * xx + cnt_y) for (xx, yy) in rect]
        # 绘制旋转矩形的四条边
        for k in range(4):
            j = (k + 1) % 4
            self.draw_line(
                [rotated_rect[k][0], rotated_rect[j][0]],
                [rotated_rect[k][1], rotated_rect[j][1]],
                color=edge_color,
                linestyle="--" if k == 1 else line_style,
                linewidth=linewidth,
            )

        # 如果标签不为 None，则绘制标签
        if label is not None:
            text_pos = rotated_rect[1]  # 左上角位置

            # 计算高度比例
            height_ratio = h / np.sqrt(self.output.height * self.output.width)
            # 调整标签颜色的亮度
            label_color = self._change_color_brightness(edge_color, brightness_factor=0.7)
            # 计算字体大小
            font_size = (
                np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5 * self._default_font_size
            )
            # 绘制标签文本
            self.draw_text(label, text_pos, color=label_color, font_size=font_size, rotation=angle)

        # 返回绘制的图像对象
        return self.output
    # 绘制圆形的方法，接受圆心坐标、颜色和半径作为参数
    def draw_circle(self, circle_coord, color, radius=3):
        """
        参数:
            circle_coord (list(int) or tuple(int)): 包含圆心的 x 和 y 坐标。
            color: 多边形的颜色。请参阅 `matplotlib.colors` 获取接受的格式完整列表。
            radius (int): 圆的半径。

        返回:
            output (VisImage): 绘制了圆形的图像对象。
        """
        # 解包圆心坐标
        x, y = circle_coord
        # 在图像的轴上添加一个圆形补丁，设置圆心、半径、填充状态和颜色
        self.output.ax.add_patch(
            mpl.patches.Circle(circle_coord, radius=radius, fill=True, color=color)
        )
        # 返回绘制圆形后的图像对象
        return self.output

    # 绘制线条的方法，接受 x 数据、y 数据、颜色、线条样式和线条宽度作为参数
    def draw_line(self, x_data, y_data, color, linestyle="-", linewidth=None):
        """
        参数:
            x_data (list[int]): 包含所有绘制点的 x 值的列表。
                列表长度应与 y_data 的长度匹配。
            y_data (list[int]): 包含所有绘制点的 y 值的列表。
                列表长度应与 x_data 的长度匹配。
            color: 线条的颜色。请参阅 `matplotlib.colors` 获取接受的格式完整列表。
            linestyle: 线条的样式。请参阅 `matplotlib.lines.Line2D`
                获取接受的格式完整列表。
            linewidth (float or None): 线条的宽度。当为 None 时，
                将计算并使用默认值。

        返回:
            output (VisImage): 绘制了线条的图像对象。
        """
        # 如果没有指定线宽，则使用默认字体大小的三分之一作为线宽
        if linewidth is None:
            linewidth = self._default_font_size / 3
        # 确保线宽不小于1
        linewidth = max(linewidth, 1)
        # 在图像的轴上添加线条，设置 x 数据、y 数据、线宽、颜色和线条样式
        self.output.ax.add_line(
            mpl.lines.Line2D(
                x_data,
                y_data,
                linewidth=linewidth * self.output.scale,
                color=color,
                linestyle=linestyle,
            )
        )
        # 返回绘制线条后的图像对象
        return self.output

    # 绘制二进制掩码的方法，接受掩码、颜色和其他可选参数
    def draw_binary_mask(
        self, binary_mask, color=None, *, edge_color=None, text=None, alpha=0.5, area_threshold=0
    # 定义绘制多边形的函数，接受顶点、颜色、边缘颜色和透明度参数
    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
        """
        参数：
            segment: 形状为 Nx2 的 numpy 数组，包含多边形的所有点。
            color: 多边形的颜色。请参考 `matplotlib.colors` 获取完整的
                格式列表。
            edge_color: 多边形边缘的颜色。请参考 `matplotlib.colors` 获取
                完整的格式列表。如果未提供，则使用多边形颜色的较暗色调。
            alpha (float): 混合效率。较小值导致更透明的遮罩。
    
        返回：
            output (VisImage): 绘制有多边形的图像对象。
        """
        # 如果未提供边缘颜色
        if edge_color is None:
            # 将边缘颜色设为比多边形颜色更深
            if alpha > 0.8:
                edge_color = self._change_color_brightness(color, brightness_factor=-0.7)
            else:
                edge_color = color
        # 将边缘颜色转换为 RGB 格式并添加 alpha 通道
        edge_color = mplc.to_rgb(edge_color) + (1,)
    
        # 创建多边形对象，设置填充和边缘颜色
        polygon = mpl.patches.Polygon(
            segment,
            fill=True,
            facecolor=mplc.to_rgb(color) + (alpha,),
            edgecolor=edge_color,
            linewidth=max(self._default_font_size // 15 * self.output.scale, 1),
        )
        # 将多边形添加到输出图像的坐标轴中
        self.output.ax.add_patch(polygon)
        # 返回包含多边形的输出图像对象
        return self.output
    
        """
        内部方法：
        """
    
        # 定义随机扰动颜色的函数，生成略微不同的颜色
        def _jitter(self, color):
            """
            随机修改给定颜色，以产生与给定颜色稍有不同的颜色。
    
            参数：
                color (tuple[double]): 包含所选颜色的 RGB 值的三元组，
                    值在 [0.0, 1.0] 范围内。
    
            返回：
                jittered_color (tuple[double]): 包含经过扰动后的颜色的 RGB 值的三元组，
                    值在 [0.0, 1.0] 范围内。
            """
            # 将颜色转换为 RGB 格式
            color = mplc.to_rgb(color)
            # 生成随机向量
            vec = np.random.rand(3)
            # 更好地在另一个颜色空间中处理
            vec = vec / np.linalg.norm(vec) * 0.5
            # 将生成的颜色与原始颜色相加并限制范围
            res = np.clip(vec + color, 0, 1)
            # 返回扰动后的颜色
            return tuple(res)
    
        # 定义创建灰度图像的函数，接受可选的遮罩参数
        def _create_grayscale_image(self, mask=None):
            """
            创建原始图像的灰度版本。
            如果给定遮罩区域，则保留其中的颜色。
            """
            # 计算图像的灰度值
            img_bw = self.img.astype("f4").mean(axis=2)
            # 将灰度值堆叠成三通道图像
            img_bw = np.stack([img_bw] * 3, axis=2)
            # 如果提供了遮罩，则在遮罩区域保留原始图像的颜色
            if mask is not None:
                img_bw[mask] = self.img[mask]
            # 返回灰度图像
            return img_bw
    # 定义一个私有方法，用于根据亮度因子调整颜色亮度
    def _change_color_brightness(self, color, brightness_factor):
        """
        根据亮度因子调整颜色的亮度，即生成饱和度比原始颜色更低或更高的颜色。
        
        参数:
            color: 多边形的颜色。详细格式可参考 `matplotlib.colors`。
            brightness_factor (float): 取值范围为[-1.0, 1.0]。0表示无变化，[-1.0, 0)范围内的值会使颜色变暗，(0, 1.0]范围内的值会使颜色变亮。
        
        返回:
            modified_color (tuple[double]): 包含修改后颜色的RGB值的元组。每个值的范围为[0.0, 1.0]。
        """
        # 确保亮度因子在有效范围内
        assert brightness_factor >= -1.0 and brightness_factor <= 1.0
        # 将颜色转换为RGB格式
        color = mplc.to_rgb(color)
        # 将RGB颜色转换为HLS格式
        polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        # 根据亮度因子计算修改后的亮度
        modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
        # 限制亮度的下界
        modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
        # 限制亮度的上界
        modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
        # 将修改后的HLS颜色转换回RGB格式
        modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
        # 返回修改后的颜色
        return modified_color
    
    # 定义一个私有方法，将不同格式的框转换为NxB数组，B为4或5
    def _convert_boxes(self, boxes):
        """
        将不同格式的框转换为NxB数组，其中B = 4或5表示框的维度。
        """
        # 如果输入是Boxes或RotatedBoxes类型，返回其张量的numpy数组
        if isinstance(boxes, Boxes) or isinstance(boxes, RotatedBoxes):
            return boxes.tensor.detach().numpy()
        else:
            # 否则将输入转换为numpy数组
            return np.asarray(boxes)
    
    # 定义一个私有方法，将不同格式的掩膜或多边形转换为掩膜和多边形的元组
    def _convert_masks(self, masks_or_polygons):
        """
        将不同格式的掩膜或多边形转换为掩膜和多边形的元组。
    
        返回:
            list[GenericMask]:
        """
        
        # 将输入赋值给m
        m = masks_or_polygons
        # 如果输入是PolygonMasks类型，获取其多边形
        if isinstance(m, PolygonMasks):
            m = m.polygons
        # 如果输入是BitMasks类型，获取其张量的numpy数组
        if isinstance(m, BitMasks):
            m = m.tensor.numpy()
        # 如果输入是torch张量，转换为numpy数组
        if isinstance(m, torch.Tensor):
            m = m.numpy()
        # 初始化返回列表
        ret = []
        # 遍历每个元素
        for x in m:
            # 如果元素是GenericMask类型，添加到返回列表
            if isinstance(x, GenericMask):
                ret.append(x)
            else:
                # 否则将其转换为GenericMask并添加到返回列表
                ret.append(GenericMask(x, self.output.height, self.output.width))
        # 返回生成的列表
        return ret
    
    # 定义一个私有方法，将关键点转换为numpy数组
    def _convert_keypoints(self, keypoints):
        # 如果输入是Keypoints类型，获取其张量
        if isinstance(keypoints, Keypoints):
            keypoints = keypoints.tensor
        # 将关键点转换为numpy数组
        keypoints = np.asarray(keypoints)
        # 返回转换后的关键点
        return keypoints
    
    # 定义一个公共方法，获取输出
    def get_output(self):
        """
        返回:
            output (VisImage): 包含添加到图像的可视化结果的图像输出。
        """
        # 返回存储的输出
        return self.output
```