# `.\YOLO-World\yolo_world\datasets\transformers\mm_mix_img_transforms.py`

```py
# 导入必要的库和模块
import collections
import copy
from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.structures.bbox import autocast_box_type
from mmengine.dataset import BaseDataset
from mmengine.dataset.base_dataset import Compose
from numpy import random
from mmyolo.registry import TRANSFORMS

# 定义一个抽象基类，用于多模态多图像混合变换
class BaseMultiModalMixImageTransform(BaseTransform, metaclass=ABCMeta):
    """A Base Transform of Multimodal multiple images mixed.

    Suitable for training on multiple images mixed data augmentation like
    mosaic and mixup.

    Cached mosaic transform will random select images from the cache
    and combine them into one output image if use_cached is True.

    Args:
        pre_transform(Sequence[str]): Sequence of transform object or
            config dict to be composed. Defaults to None.
        prob(float): The transformation probability. Defaults to 1.0.
        use_cached (bool): Whether to use cache. Defaults to False.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 40.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        max_refetch (int): The maximum number of retry iterations for getting
            valid results from the pipeline. If the number of iterations is
            greater than `max_refetch`, but results is still None, then the
            iteration is terminated and raise the error. Defaults to 15.
    """
    # 初始化函数，设置各种参数
    def __init__(self,
                 pre_transform: Optional[Sequence[str]] = None,  # 预处理转换序列的可选参数
                 prob: float = 1.0,  # 概率参数，默认为1.0
                 use_cached: bool = False,  # 是否使用缓存的布尔值，默认为False
                 max_cached_images: int = 40,  # 最大缓存图像数量，默认为40
                 random_pop: bool = True,  # 是否随机弹出的布尔值，默认为True
                 max_refetch: int = 15):  # 最大重新获取次数，默认为15
    
        # 设置最大重新获取次数
        self.max_refetch = max_refetch
        # 设置概率参数
        self.prob = prob
    
        # 设置是否使用缓存的布尔值
        self.use_cached = use_cached
        # 设置最大缓存图像数量
        self.max_cached_images = max_cached_images
        # 设置是否随机弹出的布尔值
        self.random_pop = random_pop
        # 初始化结果缓存列表
        self.results_cache = []
    
        # 如果预处理转换序列为None，则将预处理转换设置为None，否则使用Compose函数创建预处理转换
        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)
    
    @abstractmethod
    def get_indexes(self, dataset: Union[BaseDataset,
                                         list]) -> Union[list, int]:
        """Call function to collect indexes.
    
        Args:
            dataset (:obj:`Dataset` or list): The dataset or cached list.
    
        Returns:
            list or int: indexes.
        """
        pass
    
    @abstractmethod
    def mix_img_transform(self, results: dict) -> dict:
        """Mixed image data transformation.
    
        Args:
            results (dict): Result dict.
    
        Returns:
            results (dict): Updated result dict.
        """
        pass
    # 更新标签文本内容
    def _update_label_text(self, results: dict) -> dict:
        """Update label text."""
        # 如果结果中没有文本信息，则直接返回结果
        if 'texts' not in results:
            return results

        # 将所有文本信息合并并去重
        mix_texts = sum(
            [results['texts']] +
            [x['texts'] for x in results['mix_results']], [])
        mix_texts = list({tuple(x) for x in mix_texts})
        # 创建文本到索引的映射
        text2id = {text: i for i, text in enumerate(mix_texts)}

        # 更新结果中的标签文本
        for res in [results] + results['mix_results']:
            for i, label in enumerate(res['gt_bboxes_labels']):
                text = res['texts'][label]
                updated_id = text2id[tuple(text)]
                res['gt_bboxes_labels'][i] = updated_id
            res['texts'] = mix_texts
        # 返回更新后的结果
        return results

    # 装饰器，用于自动转换框类型
    @autocast_box_type()
# 注册多模态马赛克数据增强类到TRANSFORMS中
@TRANSFORMS.register_module()
class MultiModalMosaic(BaseMultiModalMixImageTransform):
    """Mosaic augmentation.

    给定4个图像，马赛克变换将它们合并成一个输出图像。输出图像由每个子图像的部分组成。

    .. code:: text

                        马赛克变换
                           center_x
                +------------------------------+
                |       pad        |           |
                |      +-----------+    pad    |
                |      |           |           |
                |      |  image1   +-----------+
                |      |           |           |
                |      |           |   image2  |
     center_y   |----+-+-----------+-----------+
                |    |   cropped   |           |
                |pad |   image3    |   image4  |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     马赛克变换步骤如下：

         1. 选择4个图像的交叉点作为马赛克中心
         2. 根据索引获取左上角图像，并从自定义数据集中随机采样另外3个图像
         3. 如果图像大于马赛克块，则将子图像裁剪

    必需键：

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (可选)
    - gt_bboxes_labels (np.int64) (可选)
    - gt_ignore_flags (bool) (可选)
    - mix_results (List[dict])

    修改后的键：

    - img
    - img_shape
    - gt_bboxes (可选)
    - gt_bboxes_labels (可选)
    - gt_ignore_flags (可选)
    Args:
        img_scale (Sequence[int]): Image size after mosaic pipeline of single
            image. The shape order should be (width, height).
            Defaults to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Defaults to (0.5, 1.5).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        pad_val (int): Pad value. Defaults to 114.
        pre_transform(Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
        use_cached (bool): Whether to use cache. Defaults to False.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 40.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        max_refetch (int): The maximum number of retry iterations for getting
            valid results from the pipeline. If the number of iterations is
            greater than `max_refetch`, but results is still None, then the
            iteration is terminated and raise the error. Defaults to 15.
    """
    # 初始化函数，设置数据增强的参数
    def __init__(self,
                 img_scale: Tuple[int, int] = (640, 640),  # 设置图像缩放的大小，默认为(640, 640)
                 center_ratio_range: Tuple[float, float] = (0.5, 1.5),  # 设置中心比例范围，默认为(0.5, 1.5)
                 bbox_clip_border: bool = True,  # 是否裁剪边界框，默认为True
                 pad_val: float = 114.0,  # 设置填充值，默认为114.0
                 pre_transform: Sequence[dict] = None,  # 预处理变换序列，默认为None
                 prob: float = 1.0,  # 数据增强的概率，默认为1.0
                 use_cached: bool = False,  # 是否使用缓存，默认为False
                 max_cached_images: int = 40,  # 最大缓存图像数量，默认为40
                 random_pop: bool = True,  # 是否随机弹出，默认为True
                 max_refetch: int = 15):  # 最大重新获取次数，默认为15
        assert isinstance(img_scale, tuple)  # 断言img_scale是元组类型
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \  # 断言概率在[0,1]范围内
                                 f'got {prob}.'
        if use_cached:
            assert max_cached_images >= 4, 'The length of cache must >= 4, ' \  # 断言缓存长度大于等于4
                                           f'but got {max_cached_images}.'
    
        # 调用父类的初始化函数
        super().__init__(
            pre_transform=pre_transform,
            prob=prob,
            use_cached=use_cached,
            max_cached_images=max_cached_images,
            random_pop=random_pop,
            max_refetch=max_refetch)
    
        # 设置参数值
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.bbox_clip_border = bbox_clip_border
        self.pad_val = pad_val
    
    # 获取数据集的索引
    def get_indexes(self, dataset: Union[BaseDataset, list]) -> list:
        """Call function to collect indexes.
    
        Args:
            dataset (:obj:`Dataset` or list): The dataset or cached list.
    
        Returns:
            list: indexes.
        """
        # 随机生成3个索引
        indexes = [random.randint(0, len(dataset)) for _ in range(3)]
        return indexes
    
    # 返回对象的字符串表示形式
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '  # 添加图像缩放参数
        repr_str += f'center_ratio_range={self.center_ratio_range}, '  # 添加中心比例范围参数
        repr_str += f'pad_val={self.pad_val}, '  # 添加填充值参数
        repr_str += f'prob={self.prob})'  # 添加概率参数
        return repr_str
# 注册 MultiModalMosaic9 类到 TRANSFORMS 模块
@TRANSFORMS.register_module()
class MultiModalMosaic9(BaseMultiModalMixImageTransform):
    """Mosaic9 augmentation.

    给定9个图像，mosaic 变换将它们合并成一个输出图像。输出图像由每个子图像的部分组成。

    .. code:: text

                +-------------------------------+------------+
                | pad           |      pad      |            |
                |    +----------+               |            |
                |    |          +---------------+  top_right |
                |    |          |      top      |   image2   |
                |    | top_left |     image1    |            |
                |    |  image8  o--------+------+--------+---+
                |    |          |        |               |   |
                +----+----------+        |     right     |pad|
                |               | center |     image3    |   |
                |     left      | image0 +---------------+---|
                |    image7     |        |               |   |
            +---+-----------+---+--------+               |   |
            |   |  cropped  |            |  bottom_right |pad|
            |   |bottom_left|            |    image4     |   |
            |   |  image6   |   bottom   |               |   |
            +---|-----------+   image5   +---------------+---|
                |    pad    |            |        pad        |
                +-----------+------------+-------------------+

     Mosaic 变换步骤如下：

         1. 根据索引获取中心图像，并从自定义数据集中随机采样另外8个图像。
         2. 在 Mosaic 后随机偏移图像

    需要的键：

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (可选)
    - gt_bboxes_labels (np.int64) (可选)
    - gt_ignore_flags (bool) (可选)
    - mix_results (List[dict])

    修改的键：

    - img
    - img_shape
    - gt_bboxes (可选)
    # gt_bboxes_labels (可选)：真实边界框标签，用于指定对象的类别
    # gt_ignore_flags (可选)：真实边界框忽略标志，用于指定是否忽略某些对象
    
    Args:
        img_scale (Sequence[int]): 单个图像经过马赛克管道后的图像大小。形状顺序应为（宽度，高度）。
            默认为（640，640）。
        bbox_clip_border (bool, optional): 是否裁剪超出图像边界的对象。在某些数据集中，如MOT17，允许gt边界框越过图像边界。
            因此，在这些情况下，我们不需要裁剪gt边界框。默认为True。
        pad_val (int): 填充值。默认为114。
        pre_transform(Sequence[dict]): 要组合的转换对象或配置字典序列。
        prob (float): 应用此转换的概率。默认为1.0。
        use_cached (bool): 是否使用缓存。默认为False。
        max_cached_images (int): 缓存的最大长度。缓存越大，此转换的随机性越强。一般来说，为每个图像提供5个缓存足以保证随机性。默认为50。
        random_pop (bool): 当缓存已满时是否随机弹出一个结果。如果设置为False，则使用FIFO弹出方法。默认为True。
        max_refetch (int): 从管道获取有效结果的最大重试次数。如果迭代次数大于`max_refetch`，但结果仍为None，则终止迭代并引发错误。默认为15。
    # 初始化函数，设置默认参数和属性
    def __init__(self,
                 img_scale: Tuple[int, int] = (640, 640),  # 设置图像缩放尺寸，默认为(640, 640)
                 bbox_clip_border: bool = True,  # 是否裁剪边界框，默认为True
                 pad_val: Union[float, int] = 114.0,  # 设置填充值，默认为114.0
                 pre_transform: Sequence[dict] = None,  # 预处理变换序列，默认为None
                 prob: float = 1.0,  # 概率值，默认为1.0
                 use_cached: bool = False,  # 是否使用缓存，默认为False
                 max_cached_images: int = 50,  # 最大缓存图像数量，默认为50
                 random_pop: bool = True,  # 是否随机弹出，默认为True
                 max_refetch: int = 15):  # 最大重新获取次数，默认为15
        assert isinstance(img_scale, tuple)  # 断言img_scale为元组类型
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \  # 断言概率值在[0,1]范围内
                                 f'got {prob}.'
        if use_cached:
            assert max_cached_images >= 9, 'The length of cache must >= 9, ' \  # 如果使用缓存，断言最大缓存图像数量大于等于9
                                           f'but got {max_cached_images}.'
    
        super().__init__(  # 调用父类的初始化函数
            pre_transform=pre_transform,
            prob=prob,
            use_cached=use_cached,
            max_cached_images=max_cached_images,
            random_pop=random_pop,
            max_refetch=max_refetch)
    
        self.img_scale = img_scale  # 设置img_scale属性
        self.bbox_clip_border = bbox_clip_border  # 设置bbox_clip_border属性
        self.pad_val = pad_val  # 设置pad_val属性
    
        # 中间变量
        self._current_img_shape = [0, 0]  # 当前图像形状
        self._center_img_shape = [0, 0]  # 中心图像形状
        self._previous_img_shape = [0, 0]  # 上一个图像形状
    
    # 获取索引函数，返回一个包含8个随机索引的列表
    def get_indexes(self, dataset: Union[BaseDataset, list]) -> list:
        """Call function to collect indexes.
    
        Args:
            dataset (:obj:`Dataset` or list): The dataset or cached list.
    
        Returns:
            list: indexes.
        """
        indexes = [random.randint(0, len(dataset)) for _ in range(8)]  # 生成8个随机索引
        return indexes
    
    # 返回对象的字符串表示形式
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__  # 获取类名
        repr_str += f'(img_scale={self.img_scale}, '  # 添加img_scale属性
        repr_str += f'pad_val={self.pad_val}, '  # 添加pad_val属性
        repr_str += f'prob={self.prob})'  # 添加prob属性
        return repr_str  # 返回字符串表示形式
# 注册 YOLOv5MultiModalMixUp 类到 TRANSFORMS 模块中
@TRANSFORMS.register_module()
class YOLOv5MultiModalMixUp(BaseMultiModalMixImageTransform):
    """MixUp data augmentation for YOLOv5.

    .. code:: text

    The mixup transform steps are as follows:

        1. Another random image is picked by dataset.
        2. Randomly obtain the fusion ratio from the beta distribution,
            then fuse the target
        of the original image and mixup image through this ratio.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - mix_results (List[dict])


    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)


    Args:
        alpha (float): parameter of beta distribution to get mixup ratio.
            Defaults to 32.
        beta (float):  parameter of beta distribution to get mixup ratio.
            Defaults to 32.
        pre_transform (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
        use_cached (bool): Whether to use cache. Defaults to False.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 20.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        max_refetch (int): The maximum number of iterations. If the number of
            iterations is greater than `max_refetch`, but gt_bbox is still
            empty, then the iteration is terminated. Defaults to 15.
    """
    # 初始化函数，设置默认参数值
    def __init__(self,
                 alpha: float = 32.0,
                 beta: float = 32.0,
                 pre_transform: Sequence[dict] = None,
                 prob: float = 1.0,
                 use_cached: bool = False,
                 max_cached_images: int = 20,
                 random_pop: bool = True,
                 max_refetch: int = 15):
        # 如果使用缓存，确保缓存长度大于等于2
        if use_cached:
            assert max_cached_images >= 2, 'The length of cache must >= 2, ' \
                                           f'but got {max_cached_images}.'
        # 调用父类的初始化函数
        super().__init__(
            pre_transform=pre_transform,
            prob=prob,
            use_cached=use_cached,
            max_cached_images=max_cached_images,
            random_pop=random_pop,
            max_refetch=max_refetch)
        # 设置 alpha 和 beta 参数
        self.alpha = alpha
        self.beta = beta

    # 获取索引函数，返回随机索引
    def get_indexes(self, dataset: Union[BaseDataset, list]) -> int:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`Dataset` or list): The dataset or cached list.

        Returns:
            int: indexes.
        """
        # 返回一个随机索引，范围为 [0, 数据集长度)
        return random.randint(0, len(dataset))
    def mix_img_transform(self, results: dict) -> dict:
        """YOLOv5 MixUp transform function.

        Args:
            results (dict): Result dict

        Returns:
            results (dict): Updated result dict.
        """
        # 确保结果字典中包含'mix_results'键
        assert 'mix_results' in results

        # 从'mix_results'中获取第一个结果字典
        retrieve_results = results['mix_results'][0]
        # 获取原始图像和混合图像
        retrieve_img = retrieve_results['img']
        ori_img = results['img']
        # 确保原始图像和混合图像的形状相同
        assert ori_img.shape == retrieve_img.shape

        # 从 beta 分布中随机获取融合比例，大约为0.5
        ratio = np.random.beta(self.alpha, self.beta)
        mixup_img = (ori_img * ratio + retrieve_img * (1 - ratio))

        # 获取混合图像的 ground truth 边界框、标签和忽略标志
        retrieve_gt_bboxes = retrieve_results['gt_bboxes']
        retrieve_gt_bboxes_labels = retrieve_results['gt_bboxes_labels']
        retrieve_gt_ignore_flags = retrieve_results['gt_ignore_flags']

        # 合并原始图像和混合图像的 ground truth 边界框、标签和忽略标志
        mixup_gt_bboxes = retrieve_gt_bboxes.cat(
            (results['gt_bboxes'], retrieve_gt_bboxes), dim=0)
        mixup_gt_bboxes_labels = np.concatenate(
            (results['gt_bboxes_labels'], retrieve_gt_bboxes_labels), axis=0)
        mixup_gt_ignore_flags = np.concatenate(
            (results['gt_ignore_flags'], retrieve_gt_ignore_flags), axis=0)
        
        # 如果结果字典中包含'gt_masks'键
        if 'gt_masks' in results:
            # 确保'retrieve_results'中也包含'gt_masks'键
            assert 'gt_masks' in retrieve_results
            # 合并原始图像和混合图像的 ground truth masks
            mixup_gt_masks = results['gt_masks'].cat(
                [results['gt_masks'], retrieve_results['gt_masks']])
            results['gt_masks'] = mixup_gt_masks

        # 更新结果字典中的图像、图像形状、ground truth 边界框、标签和忽略标志
        results['img'] = mixup_img.astype(np.uint8)
        results['img_shape'] = mixup_img.shape
        results['gt_bboxes'] = mixup_gt_bboxes
        results['gt_bboxes_labels'] = mixup_gt_bboxes_labels
        results['gt_ignore_flags'] = mixup_gt_ignore_flags

        return results
# 注册 YOLOXMultiModalMixUp 类到 TRANSFORMS 模块中
@TRANSFORMS.register_module()
class YOLOXMultiModalMixUp(BaseMultiModalMixImageTransform):
    """MixUp data augmentation for YOLOX.

    .. code:: text

                         mixup transform
                +---------------+--------------+
                | mixup image   |              |
                |      +--------|--------+     |
                |      |        |        |     |
                +---------------+        |     |
                |      |                 |     |
                |      |      image      |     |
                |      |                 |     |
                |      |                 |     |
                |      +-----------------+     |
                |             pad              |
                +------------------------------+

    The mixup transform steps are as follows:

        1. Another random image is picked by dataset and embedded in
           the top left patch(after padding and resizing)
        2. The target of mixup transform is the weighted average of mixup
           image and origin image.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - mix_results (List[dict])


    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)
    Args:
        img_scale (Sequence[int]): Image output size after mixup pipeline.
            The shape order should be (width, height). Defaults to (640, 640).
        ratio_range (Sequence[float]): Scale ratio of mixup image.
            Defaults to (0.5, 1.5).
        flip_ratio (float): Horizontal flip ratio of mixup image.
            Defaults to 0.5.
        pad_val (int): Pad value. Defaults to 114.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        pre_transform(Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
        use_cached (bool): Whether to use cache. Defaults to False.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 20.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        max_refetch (int): The maximum number of iterations. If the number of
            iterations is greater than `max_refetch`, but gt_bbox is still
            empty, then the iteration is terminated. Defaults to 15.
    """
    # 初始化函数，设置默认参数值
    def __init__(self,
                 img_scale: Tuple[int, int] = (640, 640),
                 ratio_range: Tuple[float, float] = (0.5, 1.5),
                 flip_ratio: float = 0.5,
                 pad_val: float = 114.0,
                 bbox_clip_border: bool = True,
                 pre_transform: Sequence[dict] = None,
                 prob: float = 1.0,
                 use_cached: bool = False,
                 max_cached_images: int = 20,
                 random_pop: bool = True,
                 max_refetch: int = 15):
        # 断言img_scale是元组类型
        assert isinstance(img_scale, tuple)
        # 如果使用缓存，确保最大缓存图片数量大于等于2
        if use_cached:
            assert max_cached_images >= 2, 'The length of cache must >= 2, ' \
                                           f'but got {max_cached_images}.'
        # 调用父类的初始化函数
        super().__init__(
            pre_transform=pre_transform,
            prob=prob,
            use_cached=use_cached,
            max_cached_images=max_cached_images,
            random_pop=random_pop,
            max_refetch=max_refetch)
        # 设置各个参数的值
        self.img_scale = img_scale
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.bbox_clip_border = bbox_clip_border

    # 获取索引的函数
    def get_indexes(self, dataset: Union[BaseDataset, list]) -> int:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`Dataset` or list): The dataset or cached list.

        Returns:
            int: indexes.
        """
        # 返回一个随机索引
        return random.randint(0, len(dataset))

    # 返回对象的字符串表示形式
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'flip_ratio={self.flip_ratio}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'max_refetch={self.max_refetch}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str
```