# `.\yolov8\ultralytics\data\augment.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

# 导入必要的库
import math
import random
from copy import deepcopy
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

# 导入自定义模块和函数
from ultralytics.data.utils import polygons2masks, polygons2masks_overlap
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.checks import check_version
from ultralytics.utils.instance import Instances
from ultralytics.utils.metrics import bbox_ioa
from ultralytics.utils.ops import segment2box, xyxyxyxy2xywhr
from ultralytics.utils.torch_utils import TORCHVISION_0_10, TORCHVISION_0_11, TORCHVISION_0_13

# 设置默认的均值、标准差和裁剪比例
DEFAULT_MEAN = (0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0)
DEFAULT_CROP_FRACTION = 1.0

# 图像转换基类
class BaseTransform:
    """
    Base class for image transformations in the Ultralytics library.

    This class serves as a foundation for implementing various image processing operations, designed to be
    compatible with both classification and semantic segmentation tasks.

    Methods:
        apply_image: Applies image transformations to labels.
        apply_instances: Applies transformations to object instances in labels.
        apply_semantic: Applies semantic segmentation to an image.
        __call__: Applies all label transformations to an image, instances, and semantic masks.

    Examples:
        >>> transform = BaseTransform()
        >>> labels = {'image': np.array(...), 'instances': [...], 'semantic': np.array(...)}
        >>> transformed_labels = transform(labels)
    """

    def __init__(self) -> None:
        """
        Initializes the BaseTransform object.

        This constructor sets up the base transformation object, which can be extended for specific image
        processing tasks. It is designed to be compatible with both classification and semantic segmentation.

        Examples:
            >>> transform = BaseTransform()
        """
        # 初始化方法，此处为空，用于子类扩展
        pass

    def apply_image(self, labels):
        """
        Applies image transformations to labels.

        This method is intended to be overridden by subclasses to implement specific image transformation
        logic. In its base form, it returns the input labels unchanged.

        Args:
            labels (Any): The input labels to be transformed. The exact type and structure of labels may
                vary depending on the specific implementation.

        Returns:
            (Any): The transformed labels. In the base implementation, this is identical to the input.

        Examples:
            >>> transform = BaseTransform()
            >>> original_labels = [1, 2, 3]
            >>> transformed_labels = transform.apply_image(original_labels)
            >>> print(transformed_labels)
            [1, 2, 3]
        """
        # 图像转换方法，基类中不进行任何转换，直接返回输入的标签
        pass
    # 对标签中的对象实例应用变换操作
    def apply_instances(self, labels):
        """
        Applies transformations to object instances in labels.

        This method is responsible for applying various transformations to object instances within the given
        labels. It is designed to be overridden by subclasses to implement specific instance transformation
        logic.

        Args:
            labels (Dict): A dictionary containing label information, including object instances.

        Returns:
            (Dict): The modified labels dictionary with transformed object instances.

        Examples:
            >>> transform = BaseTransform()
            >>> labels = {'instances': Instances(xyxy=torch.rand(5, 4), cls=torch.randint(0, 80, (5,)))}
            >>> transformed_labels = transform.apply_instances(labels)
        """
        pass

    # 对图像应用语义分割变换
    def apply_semantic(self, labels):
        """
        Applies semantic segmentation transformations to an image.

        This method is intended to be overridden by subclasses to implement specific semantic segmentation
        transformations. In its base form, it does not perform any operations.

        Args:
            labels (Any): The input labels or semantic segmentation mask to be transformed.

        Returns:
            (Any): The transformed semantic segmentation mask or labels.

        Examples:
            >>> transform = BaseTransform()
            >>> semantic_mask = np.zeros((100, 100), dtype=np.uint8)
            >>> transformed_mask = transform.apply_semantic(semantic_mask)
        """
        pass

    # 调用所有的标签变换操作，包括图像、实例和语义分割
    def __call__(self, labels):
        """
        Applies all label transformations to an image, instances, and semantic masks.

        This method orchestrates the application of various transformations defined in the BaseTransform class
        to the input labels. It sequentially calls the apply_image and apply_instances methods to process the
        image and object instances, respectively.

        Args:
            labels (Dict): A dictionary containing image data and annotations. Expected keys include 'img' for
                the image data, and 'instances' for object instances.

        Returns:
            (Dict): The input labels dictionary with transformed image and instances.

        Examples:
            >>> transform = BaseTransform()
            >>> labels = {'img': np.random.rand(640, 640, 3), 'instances': []}
            >>> transformed_labels = transform(labels)
        """
        self.apply_image(labels)  # 调用应用图像变换的方法
        self.apply_instances(labels)  # 调用应用实例变换的方法
        self.apply_semantic(labels)  # 调用应用语义分割变换的方法
class Compose:
    """
    A class for composing multiple image transformations.

    Attributes:
        transforms (List[Callable]): A list of transformation functions to be applied sequentially.

    Methods:
        __call__: Applies a series of transformations to input data.
        append: Appends a new transform to the existing list of transforms.
        insert: Inserts a new transform at a specified index in the list of transforms.
        __getitem__: Retrieves a specific transform or a set of transforms using indexing.
        __setitem__: Sets a specific transform or a set of transforms using indexing.
        tolist: Converts the list of transforms to a standard Python list.

    Examples:
        >>> transforms = [RandomFlip(), RandomPerspective(30)]
        >>> compose = Compose(transforms)
        >>> transformed_data = compose(data)
        >>> compose.append(CenterCrop((224, 224)))
        >>> compose.insert(0, RandomFlip())
    """

    def __init__(self, transforms):
        """
        Initializes the Compose object with a list of transforms.

        Args:
            transforms (List[Callable]): A list of callable transform objects to be applied sequentially.

        Examples:
            >>> from ultralytics.data.augment import Compose, RandomHSV, RandomFlip
            >>> transforms = [RandomHSV(), RandomFlip()]
            >>> compose = Compose(transforms)
        """
        # 将传入的 transforms 参数转换为列表，如果不是列表则转为包含该参数的列表
        self.transforms = transforms if isinstance(transforms, list) else [transforms]

    def __call__(self, data):
        """
        Applies a series of transformations to input data. This method sequentially applies each transformation in the
        Compose object's list of transforms to the input data.

        Args:
            data (Any): The input data to be transformed. This can be of any type, depending on the
                transformations in the list.

        Returns:
            (Any): The transformed data after applying all transformations in sequence.

        Examples:
            >>> transforms = [Transform1(), Transform2(), Transform3()]
            >>> compose = Compose(transforms)
            >>> transformed_data = compose(input_data)
        """
        # 依次对输入的数据应用所有的变换
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform):
        """
        Appends a new transform to the existing list of transforms.

        Args:
            transform (BaseTransform): The transformation to be added to the composition.

        Examples:
            >>> compose = Compose([RandomFlip(), RandomPerspective()])
            >>> compose.append(RandomHSV())
        """
        # 将新的变换 transform 追加到变换列表 transforms 的末尾
        self.transforms.append(transform)
    def insert(self, index, transform):
        """
        Inserts a new transform at a specified index in the existing list of transforms.

        Args:
            index (int): The index at which to insert the new transform.
            transform (BaseTransform): The transform object to be inserted.

        Examples:
            >>> compose = Compose([Transform1(), Transform2()])
            >>> compose.insert(1, Transform3())
            >>> len(compose.transforms)
            3
        """
        self.transforms.insert(index, transform)
        # 在指定的索引位置插入新的变换对象

    def __getitem__(self, index: Union[list, int]) -> "Compose":
        """
        Retrieves a specific transform or a set of transforms using indexing.

        Args:
            index (int | List[int]): Index or list of indices of the transforms to retrieve.

        Returns:
            (Compose): A new Compose object containing the selected transform(s).

        Raises:
            AssertionError: If the index is not of type int or list.

        Examples:
            >>> transforms = [RandomFlip(), RandomPerspective(10), RandomHSV(0.5, 0.5, 0.5)]
            >>> compose = Compose(transforms)
            >>> single_transform = compose[1]  # Returns a Compose object with only RandomPerspective
            >>> multiple_transforms = compose[0:2]  # Returns a Compose object with RandomFlip and RandomPerspective
        """
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"
        index = [index] if isinstance(index, int) else index
        # 如果索引是整数，则转换为列表以便处理
        return Compose([self.transforms[i] for i in index])
        # 返回包含选定变换对象的新 Compose 对象

    def __setitem__(self, index: Union[list, int], value: Union[list, int]) -> None:
        """
        Sets one or more transforms in the composition using indexing.

        Args:
            index (int | List[int]): Index or list of indices to set transforms at.
            value (Any | List[Any]): Transform or list of transforms to set at the specified index(es).

        Raises:
            AssertionError: If index type is invalid, value type doesn't match index type, or index is out of range.

        Examples:
            >>> compose = Compose([Transform1(), Transform2(), Transform3()])
            >>> compose[1] = NewTransform()  # Replace second transform
            >>> compose[0:2] = [NewTransform1(), NewTransform2()]  # Replace first two transforms
        """
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"
        if isinstance(index, list):
            assert isinstance(
                value, list
            ), f"The indices should be the same type as values, but got {type(index)} and {type(value)}"
        # 如果索引是列表，则值也必须是列表
        if isinstance(index, int):
            index, value = [index], [value]
        # 如果索引是整数，则将其转换为列表以便处理
        for i, v in zip(index, value):
            assert i < len(self.transforms), f"list index {i} out of range {len(self.transforms)}."
            # 确保索引不超出变换列表的范围
            self.transforms[i] = v
        # 在指定的索引位置设置一个或多个变换对象
    # 返回当前 Compose 对象中的 transforms 列表，这个方法将 transforms 转换为标准的 Python 列表格式
    def tolist(self):
        """
        Converts the list of transforms to a standard Python list.

        Returns:
            (List): A list containing all the transform objects in the Compose instance.

        Examples:
            >>> transforms = [RandomFlip(), RandomPerspective(10), CenterCrop()]
            >>> compose = Compose(transforms)
            >>> transform_list = compose.tolist()
            >>> print(len(transform_list))
            3
        """
        return self.transforms

    # 返回 Compose 对象的字符串表示形式，包括其包含的 transforms 列表
    def __repr__(self):
        """
        Returns a string representation of the Compose object.

        Returns:
            (str): A string representation of the Compose object, including the list of transforms.

        Examples:
            >>> transforms = [RandomFlip(), RandomPerspective(degrees=10, translate=0.1, scale=0.1)]
            >>> compose = Compose(transforms)
            >>> print(compose)
            Compose([
                RandomFlip(),
                RandomPerspective(degrees=10, translate=0.1, scale=0.1)
            ])
        """
        return f"{self.__class__.__name__}({', '.join([f'{t}' for t in self.transforms])})"
# 定义一个基类 BaseMixTransform，用于混合变换（如 MixUp 和 Mosaic）的基础操作。

class BaseMixTransform:
    """
    Base class for mix transformations like MixUp and Mosaic.

    This class provides a foundation for implementing mix transformations on datasets. It handles the
    probability-based application of transforms and manages the mixing of multiple images and labels.

    Attributes:
        dataset (Any): The dataset object containing images and labels.
        pre_transform (Callable | None): Optional transform to apply before mixing.
        p (float): Probability of applying the mix transformation.

    Methods:
        __call__: Applies the mix transformation to the input labels.
        _mix_transform: Abstract method to be implemented by subclasses for specific mix operations.
        get_indexes: Abstract method to get indexes of images to be mixed.
        _update_label_text: Updates label text for mixed images.

    Examples:
        >>> class CustomMixTransform(BaseMixTransform):
        ...     def _mix_transform(self, labels):
        ...         # Implement custom mix logic here
        ...         return labels
        ...     def get_indexes(self):
        ...         return [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
        >>> dataset = YourDataset()
        >>> transform = CustomMixTransform(dataset, p=0.5)
        >>> mixed_labels = transform(original_labels)
    """

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        """
        Initializes the BaseMixTransform object for mix transformations like MixUp and Mosaic.

        This class serves as a base for implementing mix transformations in image processing pipelines.

        Args:
            dataset (Any): The dataset object containing images and labels for mixing.
            pre_transform (Callable | None): Optional transform to apply before mixing.
            p (float): Probability of applying the mix transformation. Should be in the range [0.0, 1.0].

        Examples:
            >>> dataset = YOLODataset("path/to/data")
            >>> pre_transform = Compose([RandomFlip(), RandomPerspective()])
            >>> mix_transform = BaseMixTransform(dataset, pre_transform, p=0.5)
        """
        # 初始化方法，设置实例的属性
        self.dataset = dataset  # 数据集对象，包含图像和标签
        self.pre_transform = pre_transform  # 可选的混合前转换函数
        self.p = p  # 应用混合变换的概率
    # 定义一个方法，使对象可调用，对标签数据进行预处理变换和混合/马赛克变换

    def __call__(self, labels):
        """
        Applies pre-processing transforms and mixup/mosaic transforms to labels data.

        This method determines whether to apply the mix transform based on a probability factor. If applied, it
        selects additional images, applies pre-transforms if specified, and then performs the mix transform.

        Args:
            labels (Dict): A dictionary containing label data for an image.

        Returns:
            (Dict): The transformed labels dictionary, which may include mixed data from other images.

        Examples:
            >>> transform = BaseMixTransform(dataset, pre_transform=None, p=0.5)
            >>> result = transform({"image": img, "bboxes": boxes, "cls": classes})
        """

        # 根据概率因子决定是否应用混合变换
        if random.uniform(0, 1) > self.p:
            return labels

        # 获取一个或三个其他图像的索引
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        # 获取将用于Mosaic或MixUp的图像信息
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]

        # 如果有预处理函数，则对混合图像应用预处理
        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        
        # 将混合的标签数据存入原始标签字典中
        labels["mix_labels"] = mix_labels

        # 更新类别和文本信息
        labels = self._update_label_text(labels)
        
        # 进行Mosaic或MixUp变换
        labels = self._mix_transform(labels)
        
        # 移除标签字典中的混合数据键
        labels.pop("mix_labels", None)
        
        return labels

    def _mix_transform(self, labels):
        """
        Applies MixUp or Mosaic augmentation to the label dictionary.

        This method should be implemented by subclasses to perform specific mix transformations like MixUp or
        Mosaic. It modifies the input label dictionary in-place with the augmented data.

        Args:
            labels (Dict): A dictionary containing image and label data. Expected to have a 'mix_labels' key
                with a list of additional image and label data for mixing.

        Returns:
            (Dict): The modified labels dictionary with augmented data after applying the mix transform.

        Examples:
            >>> transform = BaseMixTransform(dataset)
            >>> labels = {'image': img, 'bboxes': boxes, 'mix_labels': [{'image': img2, 'bboxes': boxes2}]}
            >>> augmented_labels = transform._mix_transform(labels)
        """

        # 抽象方法，由子类实现特定的混合变换，如MixUp或Mosaic
        raise NotImplementedError

    def get_indexes(self):
        """
        Gets a list of shuffled indexes for mosaic augmentation.

        Returns:
            (List[int]): A list of shuffled indexes from the dataset.

        Examples:
            >>> transform = BaseMixTransform(dataset)
            >>> indexes = transform.get_indexes()
            >>> print(indexes)  # [3, 18, 7, 2]
        """

        # 获取用于Mosaic增强的打乱索引列表
        raise NotImplementedError
    # 更新标签文本和类别ID，处理图像增强中的混合标签。
    #
    # 此方法处理输入标签字典的 'texts' 和 'cls' 字段以及任何混合标签，
    # 创建统一的文本标签集并相应更新类别ID。
    #
    # Args:
    #     labels (Dict): 包含标签信息的字典，包括 'texts' 和 'cls' 字段，
    #                    可选的还有一个 'mix_labels' 字段，包含额外的标签字典。
    #
    # Returns:
    #     (Dict): 更新后的标签字典，包含统一的文本标签和更新后的类别ID。
    #
    # Examples:
    #     >>> labels = {
    #     ...     'texts': [['cat'], ['dog']],
    #     ...     'cls': torch.tensor([[0], [1]]),
    #     ...     'mix_labels': [{
    #     ...         'texts': [['bird'], ['fish']],
    #     ...         'cls': torch.tensor([[0], [1]])
    #     ...     }]
    #     ... }
    #     >>> updated_labels = self._update_label_text(labels)
    #     >>> print(updated_labels['texts'])
    #     [['cat'], ['dog'], ['bird'], ['fish']]
    #     >>> print(updated_labels['cls'])
    #     tensor([[0],
    #             [1]])
    #     >>> print(updated_labels['mix_labels'][0]['cls'])
    #     tensor([[2],
    #             [3]])
    #
    def _update_label_text(self, labels):
        if "texts" not in labels:
            return labels

        # 收集所有标签文本，包括主标签和混合标签
        mix_texts = sum([labels["texts"]] + [x["texts"] for x in labels["mix_labels"]], [])
        # 去重并转换为列表形式
        mix_texts = list({tuple(x) for x in mix_texts})
        # 创建文本到ID的映射
        text2id = {text: i for i, text in enumerate(mix_texts)}

        # 更新所有标签的类别ID和文本
        for label in [labels] + labels["mix_labels"]:
            for i, cls in enumerate(label["cls"].squeeze(-1).tolist()):
                text = label["texts"][int(cls)]
                label["cls"][i] = text2id[tuple(text)]
            label["texts"] = mix_texts

        return labels
# 继承自BaseMixTransform的Mosaic类，用于图像数据集的马赛克增强。

class Mosaic(BaseMixTransform):
    """
    Mosaic augmentation for image datasets.

    This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image.
    The augmentation is applied to a dataset with a given probability.

    Attributes:
        dataset: The dataset on which the mosaic augmentation is applied.
        imgsz (int): Image size (height and width) after mosaic pipeline of a single image.
        p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.
        n (int): The grid size, either 4 (for 2x2) or 9 (for 3x3).
        border (Tuple[int, int]): Border size for width and height.

    Methods:
        get_indexes: Returns a list of random indexes from the dataset.
        _mix_transform: Applies mixup transformation to the input image and labels.
        _mosaic3: Creates a 1x3 image mosaic.
        _mosaic4: Creates a 2x2 image mosaic.
        _mosaic9: Creates a 3x3 image mosaic.
        _update_labels: Updates labels with padding.
        _cat_labels: Concatenates labels and clips mosaic border instances.

    Examples:
        >>> from ultralytics.data.augment import Mosaic
        >>> dataset = YourDataset(...)  # Your image dataset
        >>> mosaic_aug = Mosaic(dataset, imgsz=640, p=0.5, n=4)
        >>> augmented_labels = mosaic_aug(original_labels)
    """

    def __init__(self, dataset, imgsz=640, p=1.0, n=4):
        """
        Initializes the Mosaic augmentation object.

        This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image.
        The augmentation is applied to a dataset with a given probability.

        Args:
            dataset (Any): The dataset on which the mosaic augmentation is applied.
            imgsz (int): Image size (height and width) after mosaic pipeline of a single image.
            p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.
            n (int): The grid size, either 4 (for 2x2) or 9 (for 3x3).

        Examples:
            >>> from ultralytics.data.augment import Mosaic
            >>> dataset = YourDataset(...)
            >>> mosaic_aug = Mosaic(dataset, imgsz=640, p=0.5, n=4)
        """
        # 检查概率值是否在合理范围内
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."
        # 检查网格大小是否为4或9
        assert n in {4, 9}, "grid must be equal to 4 or 9."
        # 调用父类BaseMixTransform的构造函数初始化
        super().__init__(dataset=dataset, p=p)
        # 设置数据集
        self.dataset = dataset
        # 设置图像大小
        self.imgsz = imgsz
        # 设置边界大小，为了创建马赛克图像
        self.border = (-imgsz // 2, -imgsz // 2)  # width, height
        # 设置网格大小
        self.n = n
    def get_indexes(self, buffer=True):
        """
        Returns a list of random indexes from the dataset for mosaic augmentation.

        This method selects random image indexes either from a buffer or from the entire dataset, depending on
        the 'buffer' parameter. It is used to choose images for creating mosaic augmentations.

        Args:
            buffer (bool): If True, selects images from the dataset buffer. If False, selects from the entire
                dataset.

        Returns:
            (List[int]): A list of random image indexes. The length of the list is n-1, where n is the number
                of images used in the mosaic (either 3 or 8, depending on whether n is 4 or 9).

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
            >>> indexes = mosaic.get_indexes()
            >>> print(len(indexes))  # Output: 3
        """
        if buffer:  # 从数据集缓冲区中选择图像
            return random.choices(list(self.dataset.buffer), k=self.n - 1)
        else:  # 从整个数据集中随机选择图像
            return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    def _mix_transform(self, labels):
        """
        Applies mosaic augmentation to the input image and labels.

        This method combines multiple images (3, 4, or 9) into a single mosaic image based on the 'n' attribute.
        It ensures that rectangular annotations are not present and that there are other images available for
        mosaic augmentation.

        Args:
            labels (Dict): A dictionary containing image data and annotations. Expected keys include:
                - 'rect_shape': Should be None as rect and mosaic are mutually exclusive.
                - 'mix_labels': A list of dictionaries containing data for other images to be used in the mosaic.

        Returns:
            (Dict): A dictionary containing the mosaic-augmented image and updated annotations.

        Raises:
            AssertionError: If 'rect_shape' is not None or if 'mix_labels' is empty.

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
            >>> augmented_data = mosaic._mix_transform(labels)
        """
        assert labels.get("rect_shape", None) is None, "rect and mosaic are mutually exclusive."
        assert len(labels.get("mix_labels", [])), "There are no other images for mosaic augment."
        return (
            self._mosaic3(labels) if self.n == 3 else self._mosaic4(labels) if self.n == 4 else self._mosaic9(labels)
        )  # 根据 'n' 属性选择合适的方法来生成马赛克图像
    def _mosaic3(self, labels):
        """
        Creates a 1x3 image mosaic by combining three images.

        This method arranges three images in a horizontal layout, with the main image in the center and two
        additional images on either side. It's part of the Mosaic augmentation technique used in object detection.

        Args:
            labels (Dict): A dictionary containing image and label information for the main (center) image.
                Must include 'img' key with the image array, and 'mix_labels' key with a list of two
                dictionaries containing information for the side images.

        Returns:
            (Dict): A dictionary with the mosaic image and updated labels. Keys include:
                - 'img' (np.ndarray): The mosaic image array with shape (H, W, C).
                - Other keys from the input labels, updated to reflect the new image dimensions.

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=3)
            >>> labels = {'img': np.random.rand(480, 640, 3), 'mix_labels': [{'img': np.random.rand(480, 640, 3)} for _ in range(2)]}
            >>> result = mosaic._mosaic3(labels)
            >>> print(result['img'].shape)
            (640, 640, 3)
        """
        mosaic_labels = []
        s = self.imgsz
        for i in range(3):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # Load image
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # Place img in img3
            if i == 0:  # center
                img3 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 3 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 2:  # left
                c = s - w, s + h0 - h, s, s + h0

            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            img3[y1:y2, x1:x2] = img[y1 - padh :, x1 - padw :]  # img3[ymin:ymax, xmin:xmax]
            # hp, wp = h, w  # height, width previous for next iteration

            # Labels assuming imgsz*2 mosaic size
            labels_patch = self._update_labels(labels_patch, padw + self.border[0], padh + self.border[1])
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)

        final_labels["img"] = img3[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
        return final_labels
    def _mosaic4(self, labels):
        """
        Creates a 2x2 image mosaic from four input images.

        This method combines four images into a single mosaic image by placing them in a 2x2 grid. It also
        updates the corresponding labels for each image in the mosaic.

        Args:
            labels (Dict): A dictionary containing image data and labels for the base image (index 0) and three
                additional images (indices 1-3) in the 'mix_labels' key.

        Returns:
            (Dict): A dictionary containing the mosaic image and updated labels. The 'img' key contains the mosaic
                image as a numpy array, and other keys contain the combined and adjusted labels for all four images.

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
            >>> labels = {"img": np.random.rand(480, 640, 3), "mix_labels": [
            ...     {"img": np.random.rand(480, 640, 3)} for _ in range(3)
            ... ]}
            >>> result = mosaic._mosaic4(labels)
            >>> assert result["img"].shape == (1280, 1280, 3)
        """
        mosaic_labels = []  # 初始化一个空列表，用于存储拼接后的标签
        s = self.imgsz  # 获取图像块的尺寸
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)  # 随机生成拼接中心点的坐标

        for i in range(4):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # 如果是第一个图像块，则使用整体标签；否则使用混合标签中对应的图像标签

            # 加载图像
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")  # 弹出并获取调整后的图像形状

            # 放置图像到mosaic图像块中
            if i == 0:  # 左上角
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # 创建一个基础图像块
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # 大图坐标范围
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # 小图坐标范围
            elif i == 1:  # 右上角
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # 左下角
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # 右下角
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # 将小图像放置到大图像块对应位置
            padw = x1a - x1b  # 计算水平填充量
            padh = y1a - y1b  # 计算垂直填充量

            labels_patch = self._update_labels(labels_patch, padw, padh)  # 更新标签坐标
            mosaic_labels.append(labels_patch)  # 将更新后的标签添加到mosaic_labels列表中

        final_labels = self._cat_labels(mosaic_labels)  # 合并所有的标签信息
        final_labels["img"] = img4  # 存储最终生成的mosaic图像到标签中
        return final_labels  # 返回包含mosaic图像和更新标签的字典
    # 更新标签坐标，增加填充值

    """
    Updates label coordinates with padding values.

    This method adjusts the bounding box coordinates of object instances in the labels by adding padding
    values. It also denormalizes the coordinates if they were previously normalized.

    Args:
        labels (Dict): A dictionary containing image and instance information.
        padw (int): Padding width to be added to the x-coordinates.
        padh (int): Padding height to be added to the y-coordinates.

    Returns:
        (Dict): Updated labels dictionary with adjusted instance coordinates.

    Examples:
        >>> labels = {"img": np.zeros((100, 100, 3)), "instances": Instances(...)}
        >>> padw, padh = 50, 50
        >>> updated_labels = Mosaic._update_labels(labels, padw, padh)
    """
    # 获取图像的高度和宽度
    nh, nw = labels["img"].shape[:2]
    # 将实例的边界框格式转换为 (x1, y1, x2, y2) 格式
    labels["instances"].convert_bbox(format="xyxy")
    # 如果坐标是归一化的，则反归一化坐标
    labels["instances"].denormalize(nw, nh)
    # 给标签添加填充值到边界框坐标中
    labels["instances"].add_padding(padw, padh)
    # 返回更新后的标签字典
    return labels
    def _cat_labels(self, mosaic_labels):
        """
        Concatenates and processes labels for mosaic augmentation.

        This method combines labels from multiple images used in mosaic augmentation, clips instances to the
        mosaic border, and removes zero-area boxes.

        Args:
            mosaic_labels (List[Dict]): A list of label dictionaries for each image in the mosaic.

        Returns:
            (Dict): A dictionary containing concatenated and processed labels for the mosaic image, including:
                - im_file (str): File path of the first image in the mosaic.
                - ori_shape (Tuple[int, int]): Original shape of the first image.
                - resized_shape (Tuple[int, int]): Shape of the mosaic image (imgsz * 2, imgsz * 2).
                - cls (np.ndarray): Concatenated class labels.
                - instances (Instances): Concatenated instance annotations.
                - mosaic_border (Tuple[int, int]): Mosaic border size.
                - texts (List[str], optional): Text labels if present in the original labels.

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640)
            >>> mosaic_labels = [{"cls": np.array([0, 1]), "instances": Instances(...)} for _ in range(4)]
            >>> result = mosaic._cat_labels(mosaic_labels)
            >>> print(result.keys())
            dict_keys(['im_file', 'ori_shape', 'resized_shape', 'cls', 'instances', 'mosaic_border'])
        """
        # 如果 mosaic_labels 为空列表，则直接返回空字典
        if len(mosaic_labels) == 0:
            return {}
        
        # 初始化空列表用于存储类别标签和实例注释
        cls = []
        instances = []
        
        # 计算 mosaic 图像的大小，并进行迭代处理每个标签字典
        imgsz = self.imgsz * 2  # mosaic imgsz
        for labels in mosaic_labels:
            cls.append(labels["cls"])  # 提取并存储类别标签
            instances.append(labels["instances"])  # 提取并存储实例注释
        
        # 构建最终的标签字典
        final_labels = {
            "im_file": mosaic_labels[0]["im_file"],  # 使用第一个图像的文件路径作为主图像的路径
            "ori_shape": mosaic_labels[0]["ori_shape"],  # 使用第一个图像的原始形状
            "resized_shape": (imgsz, imgsz),  # 设置 mosaic 图像的调整后形状为 (imgsz * 2, imgsz * 2)
            "cls": np.concatenate(cls, 0),  # 沿着第一个轴连接所有类别标签数组
            "instances": Instances.concatenate(instances, axis=0),  # 沿着 axis=0 连接所有实例注释
            "mosaic_border": self.border,  # 使用预设的 mosaic 边界大小
        }
        
        # 将 instances 对象中的标注信息裁剪到指定的图像大小
        final_labels["instances"].clip(imgsz, imgsz)
        
        # 移除标注信息中的零面积框，并获取有效的索引
        good = final_labels["instances"].remove_zero_area_boxes()
        
        # 根据有效索引筛选出有效的类别标签
        final_labels["cls"] = final_labels["cls"][good]
        
        # 如果原始标签中包含文本信息，则添加到最终标签字典中
        if "texts" in mosaic_labels[0]:
            final_labels["texts"] = mosaic_labels[0]["texts"]
        
        # 返回处理后的最终标签字典
        return final_labels
# MixUp 类用于实现 MixUp 数据增强技术，适用于图像数据集。
class MixUp(BaseMixTransform):
    """
    Applies MixUp augmentation to image datasets.

    This class implements the MixUp augmentation technique as described in the paper "mixup: Beyond Empirical Risk
    Minimization" (https://arxiv.org/abs/1710.09412). MixUp combines two images and their labels using a random weight.

    Attributes:
        dataset (Any): The dataset to which MixUp augmentation will be applied.
        pre_transform (Callable | None): Optional transform to apply before MixUp.
        p (float): Probability of applying MixUp augmentation.

    Methods:
        get_indexes: Returns a random index from the dataset.
        _mix_transform: Applies MixUp augmentation to the input labels.

    Examples:
        >>> from ultralytics.data.augment import MixUp
        >>> dataset = YourDataset(...)  # Your image dataset
        >>> mixup = MixUp(dataset, p=0.5)
        >>> augmented_labels = mixup(original_labels)
    """

    # 初始化 MixUp 对象
    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        """
        Initializes the MixUp augmentation object.

        MixUp is an image augmentation technique that combines two images by taking a weighted sum of their pixel
        values and labels. This implementation is designed for use with the Ultralytics YOLO framework.

        Args:
            dataset (Any): The dataset to which MixUp augmentation will be applied.
            pre_transform (Callable | None): Optional transform to apply to images before MixUp.
            p (float): Probability of applying MixUp augmentation to an image. Must be in the range [0, 1].

        Examples:
            >>> from ultralytics.data.dataset import YOLODataset
            >>> dataset = YOLODataset('path/to/data.yaml')
            >>> mixup = MixUp(dataset, pre_transform=None, p=0.5)
        """
        # 调用父类构造函数初始化对象
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)

    # 获取数据集中的随机索引
    def get_indexes(self):
        """
        Get a random index from the dataset.

        This method returns a single random index from the dataset, which is used to select an image for MixUp
        augmentation.

        Returns:
            (int): A random integer index within the range of the dataset length.

        Examples:
            >>> mixup = MixUp(dataset)
            >>> index = mixup.get_indexes()
            >>> print(index)
            42
        """
        # 返回一个介于 0 到数据集长度减 1 之间的随机整数索引
        return random.randint(0, len(self.dataset) - 1)
    # 定义一个方法 `_mix_transform`，用于执行 MixUp 数据增强操作
    def _mix_transform(self, labels):
        """
        Applies MixUp augmentation to the input labels.

        This method implements the MixUp augmentation technique as described in the paper
        "mixup: Beyond Empirical Risk Minimization" (https://arxiv.org/abs/1710.09412).

        Args:
            labels (Dict): A dictionary containing the original image and label information.

        Returns:
            (Dict): A dictionary containing the mixed-up image and combined label information.

        Examples:
            >>> mixer = MixUp(dataset)
            >>> mixed_labels = mixer._mix_transform(labels)
        """

        # 生成一个 Beta 分布的随机数，作为 MixUp 的比率，其中 alpha=32.0, beta=32.0
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        
        # 获取混合标签中的第一个标签数据
        labels2 = labels["mix_labels"][0]
        
        # 执行 MixUp 操作，将原始图像和标签与第二个标签数据按照比率 r 混合
        labels["img"] = (labels["img"] * r + labels2["img"] * (1 - r)).astype(np.uint8)
        
        # 合并两个实例集合
        labels["instances"] = Instances.concatenate([labels["instances"], labels2["instances"]], axis=0)
        
        # 合并两个类别标签数组
        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"]], 0)
        
        # 返回混合后的标签字典
        return labels
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border
        self.pre_transform = pre_transform


        """
        初始化函数，设定随机透视和仿射变换的参数。

        Parameters:
            degrees (float): 最大旋转角度的绝对值范围。
            translate (float): 最大平移量，作为图像尺寸的比例。
            scale (float): 缩放因子范围，例如，scale=0.1 表示 0.9 到 1.1 的范围。
            shear (float): 最大剪切角度。
            perspective (float): 透视失真因子。
            border (Tuple[int, int]): 马赛克边界大小，格式为 (x, y)。
            pre_transform (Callable | None): 可选的预变换，应用于随机透视之前。
        """
        """
        初始化 RandomPerspective 对象，并设置变换参数。

        此类实现了图像及其对应边界框、段和关键点的随机透视和仿射变换。变换包括旋转、平移、缩放和剪切。

        Args:
            degrees (float): 随机旋转的角度范围。
            translate (float): 随机平移的总宽度和高度的分数。
            scale (float): 缩放因子的区间，例如，缩放因子为 0.5 允许在 50% 到 150% 之间调整大小。
            shear (float): 剪切强度（角度）。
            perspective (float): 透视失真因子。
            border (Tuple[int, int]): 指定镶嵌边界的元组（上/下，左/右）。
            pre_transform (Callable | None): 应用于图像的函数/变换，在开始随机变换之前。

        Examples:
            >>> transform = RandomPerspective(degrees=10.0, translate=0.1, scale=0.5, shear=5.0)
            >>> result = transform(labels)  # 对标签应用随机透视
        """

        # 将参数分配给对象的属性
        self.degrees = degrees  # 保存旋转角度范围
        self.translate = translate  # 保存平移比例
        self.scale = scale  # 保存缩放因子
        self.shear = shear  # 保存剪切强度
        self.perspective = perspective  # 保存透视失真因子
        self.border = border  # 保存镶嵌边界
        self.pre_transform = pre_transform  # 保存预变换函数或 None
    # 定义一个方法，对输入的图像执行一系列以图像中心为中心点的仿射变换

    def affine_transform(self, img, border):
        """
        Applies a sequence of affine transformations centered around the image center.

        This function performs a series of geometric transformations on the input image, including
        translation, perspective change, rotation, scaling, and shearing. The transformations are
        applied in a specific order to maintain consistency.

        Args:
            img (np.ndarray): Input image to be transformed.
            border (Tuple[int, int]): Border dimensions for the transformed image.

        Returns:
            (Tuple[np.ndarray, np.ndarray, float]): A tuple containing:
                - np.ndarray: Transformed image.
                - np.ndarray: 3x3 transformation matrix.
                - float: Scale factor applied during the transformation.

        Examples:
            >>> import numpy as np
            >>> img = np.random.rand(100, 100, 3)
            >>> border = (10, 10)
            >>> transformed_img, matrix, scale = affine_transform(img, border)
        """

        # 创建一个单位矩阵 C，用于图像中心化
        C = np.eye(3, dtype=np.float32)

        # 设置平移参数，将图像中心移到原点
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # 创建一个单位矩阵 P，用于透视变换
        P = np.eye(3, dtype=np.float32)
        # 设置透视变换的参数
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # 创建一个单位矩阵 R，用于旋转和缩放
        R = np.eye(3, dtype=np.float32)
        # 设置旋转角度和缩放参数
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # 创建一个单位矩阵 S，用于剪切变换
        S = np.eye(3, dtype=np.float32)
        # 设置剪切变换的参数
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # 创建一个单位矩阵 T，用于平移变换
        T = np.eye(3, dtype=np.float32)
        # 设置平移变换的参数
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]  # y translation (pixels)

        # 组合所有的变换矩阵，构成最终的仿射变换矩阵 M
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT

        # 应用仿射变换到图像上
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=(114, 114, 114))

        # 返回经过变换后的图像、变换矩阵和缩放因子
        return img, M, s
    def apply_bboxes(self, bboxes, M):
        """
        Apply affine transformation to bounding boxes.

        This function applies an affine transformation to a set of bounding boxes using the provided
        transformation matrix.

        Args:
            bboxes (torch.Tensor): Bounding boxes in xyxy format with shape (N, 4), where N is the number
                of bounding boxes.
            M (torch.Tensor): Affine transformation matrix with shape (3, 3).

        Returns:
            (torch.Tensor): Transformed bounding boxes in xyxy format with shape (N, 4).

        Examples:
            >>> bboxes = torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]])
            >>> M = torch.eye(3)
            >>> transformed_bboxes = apply_bboxes(bboxes, M)
        """
        n = len(bboxes)  # 获取 bounding boxes 的数量

        if n == 0:  # 如果没有 bounding boxes，则直接返回空的 bboxes
            return bboxes

        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)  # 创建一个全为 1 的数组，用于存储坐标点和偏移项
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # 将 bboxes 转换成点坐标格式 x1y1, x2y2, x1y2, x2y1

        xy = xy @ M.T  # 应用仿射变换

        if self.perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # 如果是透视变换，进行透视缩放
        else:
            xy = xy[:, :2].reshape(n, 8)  # 否则，保持仿射变换

        # 创建新的 bounding boxes
        x = xy[:, [0, 2, 4, 6]]  # 提取 x 坐标
        y = xy[:, [1, 3, 5, 7]]  # 提取 y 坐标

        return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T
    def apply_segments(self, segments, M):
        """
        Apply affine transformations to segments and generate new bounding boxes.

        This function applies affine transformations to input segments and generates new bounding boxes based on
        the transformed segments. It clips the transformed segments to fit within the new bounding boxes.

        Args:
            segments (np.ndarray): Input segments with shape (N, M, 2), where N is the number of segments and M is the
                number of points in each segment.
            M (np.ndarray): Affine transformation matrix with shape (3, 3).

        Returns:
            (Tuple[np.ndarray, np.ndarray]): A tuple containing:
                - New bounding boxes with shape (N, 4) in xyxy format.
                - Transformed and clipped segments with shape (N, M, 2).

        Examples:
            >>> segments = np.random.rand(10, 500, 2)  # 10 segments with 500 points each
            >>> M = np.eye(3)  # Identity transformation matrix
            >>> new_bboxes, new_segments = apply_segments(segments, M)
        """
        # 获取输入段落的数量 (N) 和每个段落中点的数量 (M)
        n, num = segments.shape[:2]
        # 如果段落数量为0，则返回空列表和原始段落
        if n == 0:
            return [], segments
        
        # 创建一个形状为 (n * num, 3) 的全为1的数组，数据类型与segments相同
        xy = np.ones((n * num, 3), dtype=segments.dtype)
        # 将输入段落重新塑造为形状为 (-1, 2) 的数组，并赋给segments
        segments = segments.reshape(-1, 2)
        # 将segments的前两列赋值给xy的前两列
        xy[:, :2] = segments
        # 对xy数组应用仿射变换矩阵M的转置
        xy = xy @ M.T  # transform
        # 将xy数组的前两列除以第三列，得到新的坐标
        xy = xy[:, :2] / xy[:, 2:3]
        # 将xy重新塑造为形状为 (n, -1, 2) 的数组，并赋给segments
        segments = xy.reshape(n, -1, 2)
        # 对每个段落中的坐标应用函数segment2box，生成新的边界框数组bboxes
        bboxes = np.stack([segment2box(xy, self.size[0], self.size[1]) for xy in segments], 0)
        # 将segments的第一列限制在bboxes的左上角和右下角之间
        segments[..., 0] = segments[..., 0].clip(bboxes[:, 0:1], bboxes[:, 2:3])
        # 将segments的第二列限制在bboxes的左上角和右下角之间
        segments[..., 1] = segments[..., 1].clip(bboxes[:, 1:2], bboxes[:, 3:4])
        # 返回生成的边界框数组和被剪裁后的segments数组
        return bboxes, segments
    def apply_keypoints(self, keypoints, M):
        """
        Applies affine transformation to keypoints.

        This method transforms the input keypoints using the provided affine transformation matrix. It handles
        perspective rescaling if necessary and updates the visibility of keypoints that fall outside the image
        boundaries after transformation.

        Args:
            keypoints (np.ndarray): Array of keypoints with shape (N, 17, 3), where N is the number of instances,
                17 is the number of keypoints per instance, and 3 represents (x, y, visibility).
            M (np.ndarray): 3x3 affine transformation matrix.

        Returns:
            (np.ndarray): Transformed keypoints array with the same shape as input (N, 17, 3).

        Examples:
            >>> random_perspective = RandomPerspective()
            >>> keypoints = np.random.rand(5, 17, 3)  # 5 instances, 17 keypoints each
            >>> M = np.eye(3)  # Identity transformation
            >>> transformed_keypoints = random_perspective.apply_keypoints(keypoints, M)
        """
        # 获取 keypoints 数组的维度信息
        n, nkpt = keypoints.shape[:2]
        # 如果 keypoints 数组为空，则直接返回
        if n == 0:
            return keypoints
        # 创建一个 (n * nkpt, 3) 的数组，用于存放扩展后的 keypoints 坐标
        xy = np.ones((n * nkpt, 3), dtype=keypoints.dtype)
        # 提取可见性信息并进行形状调整，以便后续处理
        visible = keypoints[..., 2].reshape(n * nkpt, 1)
        # 将 keypoints 的坐标部分（x, y）进行扁平化并填入 xy 数组的前两列
        xy[:, :2] = keypoints[..., :2].reshape(n * nkpt, 2)
        # 应用仿射变换矩阵 M 对 keypoints 坐标进行变换
        xy = xy @ M.T  # transform
        # 根据变换后的坐标，进行透视缩放或仿射变换
        xy = xy[:, :2] / xy[:, 2:3]  # perspective rescale or affine
        # 检查变换后的坐标是否超出图像边界，更新可见性信息
        out_mask = (xy[:, 0] < 0) | (xy[:, 1] < 0) | (xy[:, 0] > self.size[0]) | (xy[:, 1] > self.size[1])
        visible[out_mask] = 0
        # 将更新后的坐标和可见性信息拼接成 keypoints 数组，恢复原始形状并返回
        return np.concatenate([xy, visible], axis=-1).reshape(n, nkpt, 3)
    def box_candidates(self, box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
        """
        Compute candidate boxes for further processing based on size and aspect ratio criteria.

        This method compares boxes before and after augmentation to determine if they meet specified
        thresholds for width, height, aspect ratio, and area. It's used to filter out boxes that have
        been overly distorted or reduced by the augmentation process.

        Args:
            box1 (numpy.ndarray): Original boxes before augmentation, shape (4, N) where n is the
                number of boxes. Format is [x1, y1, x2, y2] in absolute coordinates.
            box2 (numpy.ndarray): Augmented boxes after transformation, shape (4, N). Format is
                [x1, y1, x2, y2] in absolute coordinates.
            wh_thr (float): Width and height threshold in pixels. Boxes smaller than this in either
                dimension are rejected.
            ar_thr (float): Aspect ratio threshold. Boxes with an aspect ratio greater than this
                value are rejected.
            area_thr (float): Area ratio threshold. Boxes with an area ratio (new/old) less than
                this value are rejected.
            eps (float): Small epsilon value to prevent division by zero.

        Returns:
            (numpy.ndarray): Boolean array of shape (n,) indicating which boxes are candidates.
                True values correspond to boxes that meet all criteria.

        Examples:
            >>> random_perspective = RandomPerspective()
            >>> box1 = np.array([[0, 0, 100, 100], [0, 0, 50, 50]]).T
            >>> box2 = np.array([[10, 10, 90, 90], [5, 5, 45, 45]]).T
            >>> candidates = random_perspective.box_candidates(box1, box2)
            >>> print(candidates)
            [True True]
        """
        # 计算原始框和增强后框的宽度和高度
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        
        # 计算增强后框的宽高比
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        
        # 返回布尔数组，指示哪些框符合所有的筛选条件
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates
# 定义一个类 RandomHSV，用于随机调整图像的 Hue（色调）、Saturation（饱和度）、Value（亮度）（HSV）通道。

class RandomHSV:
    """
    Randomly adjusts the Hue, Saturation, and Value (HSV) channels of an image.

    This class applies random HSV augmentation to images within predefined limits set by hgain, sgain, and vgain.

    Attributes:
        hgain (float): Maximum variation for hue. Range is typically [0, 1].
        sgain (float): Maximum variation for saturation. Range is typically [0, 1].
        vgain (float): Maximum variation for value. Range is typically [0, 1].

    Methods:
        __call__: Applies random HSV augmentation to an image.

    Examples:
        >>> import numpy as np
        >>> from ultralytics.data.augment import RandomHSV
        >>> augmenter = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
        >>> image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> labels = {"img": image}
        >>> augmented_labels = augmenter(labels)
        >>> augmented_image = augmented_labels["img"]
    """

    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5) -> None:
        """
        Initializes the RandomHSV object for random HSV (Hue, Saturation, Value) augmentation.

        This class applies random adjustments to the HSV channels of an image within specified limits.

        Args:
            hgain (float): Maximum variation for hue. Should be in the range [0, 1].
            sgain (float): Maximum variation for saturation. Should be in the range [0, 1].
            vgain (float): Maximum variation for value. Should be in the range [0, 1].

        Examples:
            >>> hsv_aug = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
            >>> augmented_image = hsv_aug(image)
        """
        # 设置对象的属性 hgain、sgain、vgain 分别表示色调、饱和度、亮度的最大变化范围，通常为 [0, 1] 之间
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
    def __call__(self, labels):
        """
        对给定图像执行随机的HSV增强。

        该方法通过随机调整输入图像的色调（Hue）、饱和度（Saturation）和亮度（Value）通道来修改图像。
        调整范围由初始化时设置的hgain、sgain和vgain参数决定。

        Args:
            labels (Dict): 包含图像数据和元数据的字典。必须包含一个键为'img'的项，其值为numpy数组表示的图像。

        Returns:
            (None): 函数直接在原地修改输入的'labels'字典，更新其中的'img'键为经过HSV增强后的图像。

        Examples:
            >>> hsv_augmenter = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
            >>> labels = {'img': np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)}
            >>> hsv_augmenter(labels)
            >>> augmented_img = labels['img']
        """
        img = labels["img"]  # 获取输入字典中的图像数据

        if self.hgain or self.sgain or self.vgain:
            # 生成随机增益
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains

            # 将图像转换为HSV通道
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8

            # 创建Hue、Saturation、Value通道的查找表
            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            # 应用查找表到HSV图像
            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))

            # 将HSV图像转换回BGR格式，并直接更新原始图像数据
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

        return labels  # 返回经过处理的labels字典
class RandomFlip:
    """
    Applies a random horizontal or vertical flip to an image with a given probability.

    This class performs random image flipping and updates corresponding instance annotations such as
    bounding boxes and keypoints.

    Attributes:
        p (float): Probability of applying the flip. Must be between 0 and 1.
        direction (str): Direction of flip, either 'horizontal' or 'vertical'.
        flip_idx (array-like): Index mapping for flipping keypoints, if applicable.

    Methods:
        __call__: Applies the random flip transformation to an image and its annotations.

    Examples:
        >>> transform = RandomFlip(p=0.5, direction='horizontal')
        >>> result = transform({"img": image, "instances": instances})
        >>> flipped_image = result["img"]
        >>> flipped_instances = result["instances"]
    """

    def __init__(self, p=0.5, direction="horizontal", flip_idx=None) -> None:
        """
        Initializes the RandomFlip class with probability and direction.

        This class applies a random horizontal or vertical flip to an image with a given probability.
        It also updates any instances (bounding boxes, keypoints, etc.) accordingly.

        Args:
            p (float): The probability of applying the flip. Must be between 0 and 1.
            direction (str): The direction to apply the flip. Must be 'horizontal' or 'vertical'.
            flip_idx (List[int] | None): Index mapping for flipping keypoints, if any.

        Raises:
            AssertionError: If direction is not 'horizontal' or 'vertical', or if p is not between 0 and 1.

        Examples:
            >>> flip = RandomFlip(p=0.5, direction='horizontal')
            >>> flip = RandomFlip(p=0.7, direction='vertical', flip_idx=[1, 0, 3, 2, 5, 4])
        """
        # 检查传入的方向参数是否为 'horizontal' 或 'vertical'
        assert direction in {"horizontal", "vertical"}, f"Support direction `horizontal` or `vertical`, got {direction}"
        # 检查传入的概率参数是否在 0 到 1 之间
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."

        # 将参数赋值给实例的属性
        self.p = p
        self.direction = direction
        self.flip_idx = flip_idx
    def __call__(self, labels):
        """
        Applies random flip to an image and updates any instances like bounding boxes or keypoints accordingly.

        This method randomly flips the input image either horizontally or vertically based on the initialized
        probability and direction. It also updates the corresponding instances (bounding boxes, keypoints) to
        match the flipped image.

        Args:
            labels (Dict): A dictionary containing the following keys:
                'img' (numpy.ndarray): The image to be flipped.
                'instances' (ultralytics.utils.instance.Instances): An object containing bounding boxes and
                    optionally keypoints.

        Returns:
            (Dict): The same dictionary with the flipped image and updated instances:
                'img' (numpy.ndarray): The flipped image.
                'instances' (ultralytics.utils.instance.Instances): Updated instances matching the flipped image.

        Examples:
            >>> labels = {'img': np.random.rand(640, 640, 3), 'instances': Instances(...)}
            >>> random_flip = RandomFlip(p=0.5, direction='horizontal')
            >>> flipped_labels = random_flip(labels)
        """
        # 获取输入字典中的图像数据和实例对象
        img = labels["img"]
        instances = labels.pop("instances")
        # 将边界框转换为格式 "xywh"
        instances.convert_bbox(format="xywh")
        h, w = img.shape[:2]
        # 如果实例对象是归一化的，则设置高度和宽度为1
        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w

        # 根据指定的概率和方向进行图像翻转
        # 垂直翻转
        if self.direction == "vertical" and random.random() < self.p:
            img = np.flipud(img)
            instances.flipud(h)
        # 水平翻转
        if self.direction == "horizontal" and random.random() < self.p:
            img = np.fliplr(img)
            instances.fliplr(w)
            # 对于关键点的处理
            if self.flip_idx is not None and instances.keypoints is not None:
                # 重新排序关键点数组，以匹配水平翻转后的图像
                instances.keypoints = np.ascontiguousarray(instances.keypoints[:, self.flip_idx, :])
        
        # 更新字典中的图像和实例对象，并返回更新后的字典
        labels["img"] = np.ascontiguousarray(img)
        labels["instances"] = instances
        return labels
    """
    Resize image and padding for detection, instance segmentation, pose.

    This class resizes and pads images to a specified shape while preserving aspect ratio. It also updates
    corresponding labels and bounding boxes.

    Attributes:
        new_shape (tuple): Target shape (height, width) for resizing.
        auto (bool): Whether to use minimum rectangle.
        scaleFill (bool): Whether to stretch the image to new_shape.
        scaleup (bool): Whether to allow scaling up. If False, only scale down.
        stride (int): Stride for rounding padding.
        center (bool): Whether to center the image or align to top-left.

    Methods:
        __call__: Resize and pad image, update labels and bounding boxes.

    Examples:
        >>> transform = LetterBox(new_shape=(640, 640))
        >>> result = transform(labels)
        >>> resized_img = result['img']
        >>> updated_instances = result['instances']
    """

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        """
        Initialize LetterBox object for resizing and padding images.

        This class is designed to resize and pad images for object detection, instance segmentation, and pose estimation
        tasks. It supports various resizing modes including auto-sizing, scale-fill, and letterboxing.

        Args:
            new_shape (Tuple[int, int]): Target size (height, width) for the resized image.
            auto (bool): If True, use minimum rectangle to resize. If False, use new_shape directly.
            scaleFill (bool): If True, stretch the image to new_shape without padding.
            scaleup (bool): If True, allow scaling up. If False, only scale down.
            center (bool): If True, center the placed image. If False, place image in top-left corner.
            stride (int): Stride of the model (e.g., 32 for YOLOv5).

        Attributes:
            new_shape (Tuple[int, int]): Target size for the resized image.
            auto (bool): Flag for using minimum rectangle resizing.
            scaleFill (bool): Flag for stretching image without padding.
            scaleup (bool): Flag for allowing upscaling.
            stride (int): Stride value for ensuring image size is divisible by stride.

        Examples:
            >>> letterbox = LetterBox(new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32)
            >>> resized_img = letterbox(original_img)
        """
        self.new_shape = new_shape  # Initialize the target shape for resizing
        self.auto = auto  # Initialize whether to use minimum rectangle resizing
        self.scaleFill = scaleFill  # Initialize whether to stretch image to new_shape without padding
        self.scaleup = scaleup  # Initialize whether to allow upscaling
        self.stride = stride  # Initialize the stride value for padding rounding
        self.center = center  # Initialize whether to center the image or place it in the top-left corner
    # 更新标签以反映应用于图像的letterboxing后的变化

    # 将实例的边界框坐标格式转换为(x1, y1, x2, y2)
    labels["instances"].convert_bbox(format="xyxy")

    # 将实例的边界框坐标反归一化，使用图像的原始宽度和高度
    labels["instances"].denormalize(*labels["img"].shape[:2][::-1])

    # 缩放实例的边界框坐标，按照给定的比例因子
    labels["instances"].scale(*ratio)

    # 向实例的边界框添加指定的宽度和高度的填充
    labels["instances"].add_padding(padw, padh)

    # 返回更新后的标签字典，其中包含修改后的实例坐标
    return labels
# 定义一个名为 CopyPaste 的类，实现了文中描述的 Copy-Paste 数据增强方法
class CopyPaste:
    """
    Implements Copy-Paste augmentation as described in https://arxiv.org/abs/2012.07177.

    This class applies Copy-Paste augmentation on images and their corresponding instances.

    Attributes:
        p (float): Probability of applying the Copy-Paste augmentation. Must be between 0 and 1.

    Methods:
        __call__: Applies Copy-Paste augmentation to given image and instances.

    Examples:
        >>> copypaste = CopyPaste(p=0.5)
        >>> augmented_labels = copypaste(labels)
        >>> augmented_image = augmented_labels['img']
    """

    def __init__(self, p=0.5) -> None:
        """
        Initializes the CopyPaste augmentation object.

        This class implements the Copy-Paste augmentation as described in the paper "Simple Copy-Paste is a Strong Data
        Augmentation Method for Instance Segmentation" (https://arxiv.org/abs/2012.07177). It applies the Copy-Paste
        augmentation on images and their corresponding instances with a given probability.

        Args:
            p (float): The probability of applying the Copy-Paste augmentation. Must be between 0 and 1.

        Attributes:
            p (float): Stores the probability of applying the augmentation.

        Examples:
            >>> augment = CopyPaste(p=0.7)
            >>> augmented_data = augment(original_data)
        """
        # 初始化函数，设置实例的数据增强概率
        self.p = p
    def __call__(self, labels):
        """
        Applies Copy-Paste augmentation to an image and its instances.

        Args:
            labels (Dict): A dictionary containing:
                - 'img' (np.ndarray): The image to augment.
                - 'cls' (np.ndarray): Class labels for the instances.
                - 'instances' (ultralytics.engine.results.Instances): Object containing bounding boxes, segments, etc.

        Returns:
            (Dict): Dictionary with augmented image and updated instances under 'img', 'cls', and 'instances' keys.

        Examples:
            >>> labels = {'img': np.random.rand(640, 640, 3), 'cls': np.array([0, 1, 2]), 'instances': Instances(...)}
            >>> augmenter = CopyPaste(p=0.5)
            >>> augmented_labels = augmenter(labels)
        """
        # Extract the image array from the labels dictionary
        im = labels["img"]
        # Extract the class labels array from the labels dictionary
        cls = labels["cls"]
        # Get the height and width of the image
        h, w = im.shape[:2]
        # Extract and remove 'instances' object from labels dictionary
        instances = labels.pop("instances")
        # Convert bounding boxes format to 'xyxy'
        instances.convert_bbox(format="xyxy")
        # Denormalize bounding boxes using image width (w) and height (h)
        instances.denormalize(w, h)

        # Check if augmentation probability is set and if there are segments in instances
        if self.p and len(instances.segments):
            # Number of instances
            n = len(instances)
            # Get image width
            _, w, _ = im.shape  # height, width, channels
            # Create a new image array filled with zeros of the same shape
            im_new = np.zeros(im.shape, np.uint8)

            # Create a deep copy of instances for flipping
            ins_flip = deepcopy(instances)
            # Flip instances horizontally
            ins_flip.fliplr(w)

            # Calculate intersection over area (ioa) for bounding boxes
            ioa = bbox_ioa(ins_flip.bboxes, instances.bboxes)  # intersection over area, (N, M)
            # Select indexes where intersection over area is less than 0.30
            indexes = np.nonzero((ioa < 0.30).all(1))[0]  # (N, )
            n = len(indexes)
            
            # Select random subset of indexes for augmentation
            for j in random.sample(list(indexes), k=round(self.p * n)):
                # Append the class label for the selected instance
                cls = np.concatenate((cls, cls[[j]]), axis=0)
                # Concatenate the selected instances with flipped instances
                instances = Instances.concatenate((instances, ins_flip[[j]]), axis=0)
                # Draw filled contours on the new image based on segments
                cv2.drawContours(im_new, instances.segments[[j]].astype(np.int32), -1, (1, 1, 1), cv2.FILLED)

            # Flip the original image left-right for augmentation
            result = cv2.flip(im, 1)  # augment segments (flip left-right)
            # Convert im_new to boolean array
            i = cv2.flip(im_new, 1).astype(bool)
            # Copy augmented segments from result to original image
            im[i] = result[i]

        # Update labels dictionary with augmented image, class labels, and instances
        labels["img"] = im
        labels["cls"] = cls
        labels["instances"] = instances
        # Return augmented labels dictionary
        return labels
# 定义 Albumentations 类，用于图像增强的 Albumentations 变换

class Albumentations:
    """
    Albumentations transformations for image augmentation.

    This class applies various image transformations using the Albumentations library. It includes operations such as
    Blur, Median Blur, conversion to grayscale, Contrast Limited Adaptive Histogram Equalization (CLAHE), random changes
    in brightness and contrast, RandomGamma, and image quality reduction through compression.

    Attributes:
        p (float): Probability of applying the transformations.
            变换应用的概率
        transform (albumentations.Compose): Composed Albumentations transforms.
            组合的 Albumentations 变换对象
        contains_spatial (bool): Indicates if the transforms include spatial operations.
            表示变换是否包含空间操作

    Methods:
        __call__: Applies the Albumentations transformations to the input labels.
            将 Albumentations 变换应用于输入的标签数据

    Examples:
        >>> transform = Albumentations(p=0.5)
        >>> augmented_labels = transform(labels)

    Notes:
        - The Albumentations package must be installed to use this class.
            要使用此类，必须安装 Albumentations 包
        - If the package is not installed or an error occurs during initialization, the transform will be set to None.
            如果未安装该包或在初始化期间出现错误，则 transform 将设置为 None
        - Spatial transforms are handled differently and require special processing for bounding boxes.
            空间变换的处理方式不同，并且需要对边界框进行特殊处理
    """
    # 定义一个特殊方法，使对象可以像函数一样调用，对输入的标签应用 Albumentations 转换
    def __call__(self, labels):
        """
        Applies Albumentations transformations to input labels.

        This method applies a series of image augmentations using the Albumentations library. It can perform both
        spatial and non-spatial transformations on the input image and its corresponding labels.

        Args:
            labels (Dict): A dictionary containing image data and annotations. Expected keys are:
                - 'img': numpy.ndarray representing the image
                - 'cls': numpy.ndarray of class labels
                - 'instances': object containing bounding boxes and other instance information

        Returns:
            (Dict): The input dictionary with augmented image and updated annotations.

        Examples:
            >>> transform = Albumentations(p=0.5)
            >>> labels = {
            ...     "img": np.random.rand(640, 640, 3),
            ...     "cls": np.array([0, 1]),
            ...     "instances": Instances(bboxes=np.array([[0, 0, 1, 1], [0.5, 0.5, 0.8, 0.8]]))
            ... }
            >>> augmented = transform(labels)
            >>> assert augmented["img"].shape == (640, 640, 3)

        Notes:
            - The method applies transformations with probability self.p.
            - Spatial transforms update bounding boxes, while non-spatial transforms only modify the image.
            - Requires the Albumentations library to be installed.
        """
        # 如果未设置变换或者以概率 self.p 不执行变换，则直接返回原始标签
        if self.transform is None or random.random() > self.p:
            return labels

        # 如果包含空间变换
        if self.contains_spatial:
            cls = labels["cls"]
            if len(cls):
                im = labels["img"]
                # 将实例的边界框转换为 xywh 格式并进行归一化
                labels["instances"].convert_bbox("xywh")
                labels["instances"].normalize(*im.shape[:2][::-1])
                bboxes = labels["instances"].bboxes
                # TODO: add supports of segments and keypoints
                # 对图像及其边界框进行变换
                new = self.transform(image=im, bboxes=bboxes, class_labels=cls)  # transformed
                # 如果新图像中存在边界框，则更新标签
                if len(new["class_labels"]) > 0:
                    labels["img"] = new["image"]
                    labels["cls"] = np.array(new["class_labels"])
                    bboxes = np.array(new["bboxes"], dtype=np.float32)
                # 更新实例对象的边界框信息
                labels["instances"].update(bboxes=bboxes)
        else:
            # 对图像进行非空间变换
            labels["img"] = self.transform(image=labels["img"])["image"]  # transformed

        return labels
# 图像注释格式化类，用于目标检测、实例分割和姿态估计任务中的图像和实例注释标准化
class Format:
    """
    A class for formatting image annotations for object detection, instance segmentation, and pose estimation tasks.

    This class standardizes image and instance annotations to be used by the `collate_fn` in PyTorch DataLoader.

    Attributes:
        bbox_format (str): Format for bounding boxes. Options are 'xywh' or 'xyxy'.
        normalize (bool): Whether to normalize bounding boxes.
        return_mask (bool): Whether to return instance masks for segmentation.
        return_keypoint (bool): Whether to return keypoints for pose estimation.
        return_obb (bool): Whether to return oriented bounding boxes.
        mask_ratio (int): Downsample ratio for masks.
        mask_overlap (bool): Whether to overlap masks.
        batch_idx (bool): Whether to keep batch indexes.
        bgr (float): The probability to return BGR images.

    Methods:
        __call__: Formats labels dictionary with image, classes, bounding boxes, and optionally masks and keypoints.
        _format_img: Converts image from Numpy array to PyTorch tensor.
        _format_segments: Converts polygon points to bitmap masks.

    Examples:
        >>> formatter = Format(bbox_format='xywh', normalize=True, return_mask=True)
        >>> formatted_labels = formatter(labels)
        >>> img = formatted_labels['img']
        >>> bboxes = formatted_labels['bboxes']
        >>> masks = formatted_labels['masks']
    """

    # 初始化方法，设置图像注释格式化的各种属性
    def __init__(
        self,
        bbox_format="xywh",       # 边界框格式，可选值为 'xywh' 或 'xyxy'
        normalize=True,           # 是否归一化边界框
        return_mask=False,        # 是否返回用于分割的实例掩码
        return_keypoint=False,    # 是否返回用于姿态估计的关键点
        return_obb=False,         # 是否返回方向边界框
        mask_ratio=4,             # 掩码的下采样比率
        mask_overlap=True,        # 掩码是否重叠
        batch_idx=True,           # 是否保留批次索引
        bgr=0.0,                  # 返回BGR图像的概率
    ):
        self.bbox_format = bbox_format
        self.normalize = normalize
        self.return_mask = return_mask
        self.return_keypoint = return_keypoint
        self.return_obb = return_obb
        self.mask_ratio = mask_ratio
        self.mask_overlap = mask_overlap
        self.batch_idx = batch_idx
        self.bgr = bgr
    ):
        """
        Initializes the Format class with given parameters for image and instance annotation formatting.

        This class standardizes image and instance annotations for object detection, instance segmentation, and pose
        estimation tasks, preparing them for use in PyTorch DataLoader's `collate_fn`.

        Args:
            bbox_format (str): Format for bounding boxes. Options are 'xywh', 'xyxy', etc.
            normalize (bool): Whether to normalize bounding boxes to [0,1].
            return_mask (bool): If True, returns instance masks for segmentation tasks.
            return_keypoint (bool): If True, returns keypoints for pose estimation tasks.
            return_obb (bool): If True, returns oriented bounding boxes.
            mask_ratio (int): Downsample ratio for masks.
            mask_overlap (bool): If True, allows mask overlap.
            batch_idx (bool): If True, keeps batch indexes.
            bgr (float): Probability of returning BGR images instead of RGB.

        Attributes:
            bbox_format (str): Format for bounding boxes.
            normalize (bool): Whether bounding boxes are normalized.
            return_mask (bool): Whether to return instance masks.
            return_keypoint (bool): Whether to return keypoints.
            return_obb (bool): Whether to return oriented bounding boxes.
            mask_ratio (int): Downsample ratio for masks.
            mask_overlap (bool): Whether masks can overlap.
            batch_idx (bool): Whether to keep batch indexes.
            bgr (float): The probability to return BGR images.

        Examples:
            >>> format = Format(bbox_format='xyxy', return_mask=True, return_keypoint=False)
            >>> print(format.bbox_format)
            xyxy
        """
        # Initialize the Format class with provided parameters
        self.bbox_format = bbox_format  # Assign the bounding box format
        self.normalize = normalize  # Set whether to normalize bounding boxes
        self.return_mask = return_mask  # Set whether to return instance masks
        self.return_keypoint = return_keypoint  # Set whether to return keypoints
        self.return_obb = return_obb  # Set whether to return oriented bounding boxes
        self.mask_ratio = mask_ratio  # Assign the mask downsample ratio
        self.mask_overlap = mask_overlap  # Set whether masks can overlap
        self.batch_idx = batch_idx  # Set whether to keep batch indexes
        self.bgr = bgr  # Assign the probability of returning BGR images
    def _format_img(self, img):
        """
        Formats an image for YOLO from a Numpy array to a PyTorch tensor.

        This function performs the following operations:
        1. Ensures the image has 3 dimensions (adds a channel dimension if needed).
        2. Transposes the image from HWC to CHW format.
        3. Optionally flips the color channels from RGB to BGR.
        4. Converts the image to a contiguous array.
        5. Converts the Numpy array to a PyTorch tensor.

        Args:
            img (np.ndarray): Input image as a Numpy array with shape (H, W, C) or (H, W).

        Returns:
            (torch.Tensor): Formatted image as a PyTorch tensor with shape (C, H, W).

        Examples:
            >>> import numpy as np
            >>> img = np.random.rand(100, 100, 3)
            >>> formatted_img = self._format_img(img)
            >>> print(formatted_img.shape)
            torch.Size([3, 100, 100])
        """
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)  # 如果图像维度小于3，则在最后一个维度上扩展一个维度
        img = img.transpose(2, 0, 1)  # 将图像从HWC格式转换为CHW格式
        # 根据self.bgr的随机值决定是否将RGB颜色通道翻转为BGR
        img = np.ascontiguousarray(img[::-1] if random.uniform(0, 1) > self.bgr else img)
        img = torch.from_numpy(img)  # 将Numpy数组转换为PyTorch张量
        return img

    def _format_segments(self, instances, cls, w, h):
        """
        Converts polygon segments to bitmap masks.

        Args:
            instances (Instances): Object containing segment information.
            cls (numpy.ndarray): Class labels for each instance.
            w (int): Width of the image.
            h (int): Height of the image.

        Returns:
            (tuple): Tuple containing:
                masks (numpy.ndarray): Bitmap masks with shape (N, H, W) or (1, H, W) if mask_overlap is True.
                instances (Instances): Updated instances object with sorted segments if mask_overlap is True.
                cls (numpy.ndarray): Updated class labels, sorted if mask_overlap is True.

        Notes:
            - If self.mask_overlap is True, masks are overlapped and sorted by area.
            - If self.mask_overlap is False, each mask is represented separately.
            - Masks are downsampled according to self.mask_ratio.
        """
        segments = instances.segments
        if self.mask_overlap:
            # 将多边形段转换为位图掩码，并根据面积重叠和排序
            masks, sorted_idx = polygons2masks_overlap((h, w), segments, downsample_ratio=self.mask_ratio)
            masks = masks[None]  # 将位图掩码形状从(640, 640)改为(1, 640, 640)
            instances = instances[sorted_idx]  # 更新实例对象以包含按面积排序的段
            cls = cls[sorted_idx]  # 更新类标签以包含按面积排序的段
        else:
            # 将多边形段转换为位图掩码，每个掩码分开表示
            masks = polygons2masks((h, w), segments, color=1, downsample_ratio=self.mask_ratio)

        return masks, instances, cls
        self,
        prompt_format: str = "{}",
        neg_samples: Tuple[int, int] = (80, 80),
        max_samples: int = 80,
        padding: bool = False,
        padding_value: str = "",

# 初始化方法，用于创建一个RandomLoadText对象，设置各种属性和默认参数


    ):
        self.prompt_format = prompt_format
        # 设置文本提示格式的字符串，用于生成文本样本
        self.neg_samples = neg_samples
        # 设置负样本（不在图像中的文本）随机采样的范围
        self.max_samples = max_samples
        # 设置每个图像中不同文本样本的最大数量
        self.padding = padding
        # 设置是否对文本进行填充，以达到最大样本数量
        self.padding_value = padding_value
        # 设置填充文本的内容，当padding为True时使用
    ) -> None:
        """
        Initializes the RandomLoadText class for randomly sampling positive and negative texts.

        This class is designed to randomly sample positive texts and negative texts, and update the class
        indices accordingly to the number of samples. It can be used for text-based object detection tasks.

        Args:
            prompt_format (str): Format string for the prompt. Default is '{}'. The format string should
                contain a single pair of curly braces {} where the text will be inserted.
            neg_samples (Tuple[int, int]): A range to randomly sample negative texts. The first integer
                specifies the minimum number of negative samples, and the second integer specifies the
                maximum. Default is (80, 80).
            max_samples (int): The maximum number of different text samples in one image. Default is 80.
            padding (bool): Whether to pad texts to max_samples. If True, the number of texts will always
                be equal to max_samples. Default is False.
            padding_value (str): The padding text to use when padding is True. Default is an empty string.

        Attributes:
            prompt_format (str): The format string for the prompt.
            neg_samples (Tuple[int, int]): The range for sampling negative texts.
            max_samples (int): The maximum number of text samples.
            padding (bool): Whether padding is enabled.
            padding_value (str): The value used for padding.

        Examples:
            >>> random_load_text = RandomLoadText(prompt_format="Object: {}", neg_samples=(50, 100), max_samples=120)
            >>> random_load_text.prompt_format
            'Object: {}'
            >>> random_load_text.neg_samples
            (50, 100)
            >>> random_load_text.max_samples
            120
        """
        # 设置实例变量 prompt_format 为传入的 prompt_format 参数
        self.prompt_format = prompt_format
        # 设置实例变量 neg_samples 为传入的 neg_samples 参数
        self.neg_samples = neg_samples
        # 设置实例变量 max_samples 为传入的 max_samples 参数
        self.max_samples = max_samples
        # 设置实例变量 padding 为传入的 padding 参数
        self.padding = padding
        # 设置实例变量 padding_value 为传入的 padding_value 参数
        self.padding_value = padding_value
    def __call__(self, labels: dict) -> dict:
        """
        随机抽样正负文本，并更新类别索引。

        该方法根据图像中现有的类别标签，随机抽取正样本文本，并从剩余类别中随机选择负样本文本。
        然后更新类别索引以匹配新抽样的文本顺序。

        Args:
            labels (Dict): 包含图像标签和元数据的字典。必须包含 'texts' 和 'cls' 键。

        Returns:
            (Dict): 更新后的标签字典，包含新的 'cls' 和 'texts' 条目。

        Examples:
            >>> loader = RandomLoadText(prompt_format="A photo of {}", neg_samples=(5, 10), max_samples=20)
            >>> labels = {"cls": np.array([[0], [1], [2]]), "texts": [["dog"], ["cat"], ["bird"]]}
            >>> updated_labels = loader(labels)
        """
        assert "texts" in labels, "No texts found in labels."
        # 获取类别对应的文本
        class_texts = labels["texts"]
        # 确定类别的数量
        num_classes = len(class_texts)
        # 转换类别数组为 numpy 数组
        cls = np.asarray(labels.pop("cls"), dtype=int)
        # 获取所有正样本的类别标签
        pos_labels = np.unique(cls).tolist()

        # 如果正样本数超过最大样本数限制，则随机选择一部分正样本
        if len(pos_labels) > self.max_samples:
            pos_labels = random.sample(pos_labels, k=self.max_samples)

        # 确定负样本数目，限制在剩余类别数和设定范围内
        neg_samples = min(min(num_classes, self.max_samples) - len(pos_labels), random.randint(*self.neg_samples))
        # 获取所有负样本的类别标签
        neg_labels = [i for i in range(num_classes) if i not in pos_labels]
        neg_labels = random.sample(neg_labels, k=neg_samples)

        # 合并正负样本标签，并随机打乱顺序
        sampled_labels = pos_labels + neg_labels
        random.shuffle(sampled_labels)

        # 创建类别到新索引的映射
        label2ids = {label: i for i, label in enumerate(sampled_labels)}
        # 创建一个布尔数组，标记有效的实例索引
        valid_idx = np.zeros(len(labels["instances"]), dtype=bool)
        new_cls = []
        for i, label in enumerate(cls.squeeze(-1).tolist()):
            if label not in label2ids:
                continue
            valid_idx[i] = True
            new_cls.append([label2ids[label]])
        # 根据有效索引筛选实例
        labels["instances"] = labels["instances"][valid_idx]
        labels["cls"] = np.array(new_cls)

        # 当存在多个提示时，随机选择一个提示文本
        texts = []
        for label in sampled_labels:
            prompts = class_texts[label]
            assert len(prompts) > 0
            prompt = self.prompt_format.format(prompts[random.randrange(len(prompts))])
            texts.append(prompt)

        # 如果需要填充文本，添加填充值
        if self.padding:
            valid_labels = len(pos_labels) + len(neg_labels)
            num_padding = self.max_samples - valid_labels
            if num_padding > 0:
                texts += [self.padding_value] * num_padding

        # 更新标签字典中的文本数据
        labels["texts"] = texts
        return labels
# 为YOLOv8训练准备一系列图像转换操作

def v8_transforms(dataset, imgsz, hyp, stretch=False):
    """
    Applies a series of image transformations for YOLOv8 training.

    This function creates a composition of image augmentation techniques to prepare images for YOLOv8 training.
    It includes operations such as mosaic, copy-paste, random perspective, mixup, and various color adjustments.

    Args:
        dataset (Dataset): The dataset object containing image data and annotations.
        imgsz (int): The target image size for resizing.
        hyp (Dict): A dictionary of hyperparameters controlling various aspects of the transformations.
        stretch (bool): If True, applies stretching to the image. If False, uses LetterBox resizing.

    Returns:
        (Compose): A composition of image transformations to be applied to the dataset.

    Examples:
        >>> from ultralytics.data.dataset import YOLODataset
        >>> dataset = YOLODataset(img_path='path/to/images', imgsz=640)
        >>> hyp = {'mosaic': 1.0, 'copy_paste': 0.5, 'degrees': 10.0, 'translate': 0.2, 'scale': 0.9}
        >>> transforms = v8_transforms(dataset, imgsz=640, hyp=hyp)
        >>> augmented_data = transforms(dataset[0])
    """
    
    # 创建预处理转换操作组合
    pre_transform = Compose(
        [
            # 创建Mosaic操作实例，用于图像合并
            Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic),
            # 创建CopyPaste操作实例，用于图像复制粘贴
            CopyPaste(p=hyp.copy_paste),
            # 创建RandomPerspective操作实例，用于随机透视变换
            RandomPerspective(
                degrees=hyp.degrees,
                translate=hyp.translate,
                scale=hyp.scale,
                shear=hyp.shear,
                perspective=hyp.perspective,
                # 根据stretch参数决定是否使用LetterBox进行预处理
                pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
            ),
        ]
    )
    
    # 获取关键点翻转索引以进行关键点增强
    flip_idx = dataset.data.get("flip_idx", [])  # for keypoints augmentation
    
    # 如果数据集使用关键点
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get("kpt_shape", None)
        
        # 如果flip_idx为空且hyp.fliplr大于0.0，发出警告并设置fliplr为0.0
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
        
        # 如果flip_idx不为空且其长度不等于kpt_shape的第一个维度，抛出数值错误
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")

    # 返回最终的图像转换操作组合
    return Compose(
        [
            pre_transform,
            # 创建MixUp操作实例，用于混合图像
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            # 创建Albumentations操作实例，用于图像增强
            Albumentations(p=1.0),
            # 创建RandomHSV操作实例，用于随机HSV调整
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            # 创建垂直方向随机翻转操作实例
            RandomFlip(direction="vertical", p=hyp.flipud),
            # 创建水平方向随机翻转操作实例，根据flip_idx进行关键点翻转
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )  # transforms
    """
    This function generates a sequence of torchvision transforms suitable for preprocessing images
    for classification models during evaluation or inference. The transforms include resizing,
    center cropping, conversion to tensor, and normalization.
    
    Args:
        size (int | tuple): The target size for the transformed image. If an int, it defines the shortest edge. If a
            tuple, it defines (height, width).
        mean (tuple): Mean values for each RGB channel used in normalization.
        std (tuple): Standard deviation values for each RGB channel used in normalization.
        interpolation (str): Interpolation method of either 'NEAREST', 'BILINEAR' or 'BICUBIC'.
        crop_fraction (float): Fraction of the image to be cropped.
    
    Returns:
        (torchvision.transforms.Compose): A composition of torchvision transforms.
    
    Examples:
        >>> transforms = classify_transforms(size=224)
        >>> img = Image.open('path/to/image.jpg')
        >>> transformed_img = transforms(img)
    """
    import torchvision.transforms as T  # scope for faster 'import ultralytics'
    
    if isinstance(size, (tuple, list)):
        assert len(size) == 2, f"'size' tuples must be length 2, not length {len(size)}"
        scale_size = tuple(math.floor(x / crop_fraction) for x in size)
    else:
        scale_size = math.floor(size / crop_fraction)
        scale_size = (scale_size, scale_size)
    
    # Aspect ratio is preserved, crops center within image, no borders are added, image is lost
    if scale_size[0] == scale_size[1]:
        # Simple case, use torchvision built-in Resize with the shortest edge mode (scalar size arg)
        tfl = [T.Resize(scale_size[0], interpolation=getattr(T.InterpolationMode, interpolation))]
    else:
        # Resize the shortest edge to matching target dim for non-square target
        tfl = [T.Resize(scale_size)]
    tfl.extend(
        [
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]
    )
    return T.Compose(tfl)
# Classification training augmentations --------------------------------------------------------------------------------
def classify_augmentations(
    size=224,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.0,
    auto_augment=None,
    hsv_h=0.015,  # image HSV-Hue augmentation (fraction)
    hsv_s=0.4,    # image HSV-Saturation augmentation (fraction)
    hsv_v=0.4,    # image HSV-Value augmentation (fraction)
    force_color_jitter=False,
    erasing=0.0,
    interpolation="BILINEAR",
):
    """
    Creates a composition of image augmentation transforms for classification tasks.

    This function generates a set of image transformations suitable for training classification models. It includes
    options for resizing, flipping, color jittering, auto augmentation, and random erasing.

    Args:
        size (int): Target size for the image after transformations.
        mean (tuple): Mean values for normalization, one per channel.
        std (tuple): Standard deviation values for normalization, one per channel.
        scale (tuple | None): Range of size of the origin size cropped.
        ratio (tuple | None): Range of aspect ratio of the origin aspect ratio cropped.
        hflip (float): Probability of horizontal flip.
        vflip (float): Probability of vertical flip.
        auto_augment (str | None): Auto augmentation policy. Can be 'randaugment', 'augmix', 'autoaugment' or None.
        hsv_h (float): Image HSV-Hue augmentation factor.
        hsv_s (float): Image HSV-Saturation augmentation factor.
        hsv_v (float): Image HSV-Value augmentation factor.
        force_color_jitter (bool): Whether to apply color jitter even if auto augment is enabled.
        erasing (float): Probability of random erasing.
        interpolation (str): Interpolation method of either 'NEAREST', 'BILINEAR' or 'BICUBIC'.

    Returns:
        (torchvision.transforms.Compose): A composition of image augmentation transforms.

    Examples:
        >>> transforms = classify_augmentations(size=224, auto_augment='randaugment')
        >>> augmented_image = transforms(original_image)
    """

    # Transforms to apply if Albumentations not installed
    import torchvision.transforms as T  # scope for faster 'import ultralytics'

    # Check if size is an integer, raise TypeError if not
    if not isinstance(size, int):
        raise TypeError(f"classify_transforms() size {size} must be integer, not (list, tuple)")

    # Set default values for scale and ratio if not provided
    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range

    # Get the interpolation mode based on the provided string
    interpolation = getattr(T.InterpolationMode, interpolation)

    # Primary list of transformations, starting with RandomResizedCrop
    primary_tfl = [T.RandomResizedCrop(size, scale=scale, ratio=ratio, interpolation=interpolation)]

    # Add horizontal flip transformation if probability is greater than 0
    if hflip > 0.0:
        primary_tfl.append(T.RandomHorizontalFlip(p=hflip))

    # Add vertical flip transformation if probability is greater than 0
    if vflip > 0.0:
        primary_tfl.append(T.RandomVerticalFlip(p=vflip))

    # Initialize secondary transformations list
    secondary_tfl = []

    # Flag to disable color jitter if auto augment is enabled
    disable_color_jitter = False
    # 如果开启了自动增强选项
    if auto_augment:
        # 断言自动增强参数是字符串类型，如果不是则抛出异常
        assert isinstance(auto_augment, str), f"Provided argument should be string, but got type {type(auto_augment)}"
        
        # 如果强制关闭颜色抖动，则禁用颜色抖动
        # 这允许在不破坏旧的超参数配置的情况下覆盖默认设置
        disable_color_jitter = not force_color_jitter

        # 如果自动增强策略是 "randaugment"
        if auto_augment == "randaugment":
            # 如果使用的是 torchvision >= 0.11.0，则添加 RandAugment 转换器
            if TORCHVISION_0_11:
                secondary_tfl.append(T.RandAugment(interpolation=interpolation))
            else:
                # 如果 torchvision 版本不足以支持 "randaugment"，发出警告并禁用它
                LOGGER.warning('"auto_augment=randaugment" requires torchvision >= 0.11.0. Disabling it.')

        # 如果自动增强策略是 "augmix"
        elif auto_augment == "augmix":
            # 如果使用的是 torchvision >= 0.13.0，则添加 AugMix 转换器
            if TORCHVISION_0_13:
                secondary_tfl.append(T.AugMix(interpolation=interpolation))
            else:
                # 如果 torchvision 版本不足以支持 "augmix"，发出警告并禁用它
                LOGGER.warning('"auto_augment=augmix" requires torchvision >= 0.13.0. Disabling it.')

        # 如果自动增强策略是 "autoaugment"
        elif auto_augment == "autoaugment":
            # 如果使用的是 torchvision >= 0.10.0，则添加 AutoAugment 转换器
            if TORCHVISION_0_10:
                secondary_tfl.append(T.AutoAugment(interpolation=interpolation))
            else:
                # 如果 torchvision 版本不足以支持 "autoaugment"，发出警告并禁用它
                LOGGER.warning('"auto_augment=autoaugment" requires torchvision >= 0.10.0. Disabling it.')

        # 如果自动增强策略既不是 "randaugment" 也不是 "augmix" 也不是 "autoaugment"，抛出值错误异常
        else:
            raise ValueError(
                f'Invalid auto_augment policy: {auto_augment}. Should be one of "randaugment", '
                f'"augmix", "autoaugment" or None'
            )

    # 如果未禁用颜色抖动
    if not disable_color_jitter:
        # 添加颜色抖动转换器，使用提供的 HSV 值
        secondary_tfl.append(T.ColorJitter(brightness=hsv_v, contrast=hsv_v, saturation=hsv_s, hue=hsv_h))

    # 最终的转换器列表，包括将图像转换为张量、归一化和随机擦除
    final_tfl = [
        T.ToTensor(),
        T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        T.RandomErasing(p=erasing, inplace=True),
    ]

    # 返回所有转换器的组合
    return T.Compose(primary_tfl + secondary_tfl + final_tfl)
# NOTE: keep this class for backward compatibility
class ClassifyLetterBox:
    """
    A class for resizing and padding images for classification tasks.

    This class is designed to be part of a transformation pipeline, e.g., T.Compose([LetterBox(size), ToTensor()]).
    It resizes and pads images to a specified size while maintaining the original aspect ratio.

    Attributes:
        h (int): Target height of the image.
        w (int): Target width of the image.
        auto (bool): If True, automatically calculates the short side using stride.
        stride (int): The stride value, used when 'auto' is True.

    Methods:
        __call__: Applies the letterbox transformation to an input image.

    Examples:
        >>> transform = ClassifyLetterBox(size=(640, 640), auto=False, stride=32)
        >>> img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        >>> result = transform(img)
        >>> print(result.shape)
        (640, 640, 3)
    """

    def __init__(self, size=(640, 640), auto=False, stride=32):
        """
        Initializes the ClassifyLetterBox object for image preprocessing.

        This class is designed to be part of a transformation pipeline for image classification tasks. It resizes and
        pads images to a specified size while maintaining the original aspect ratio.

        Args:
            size (int | Tuple[int, int]): Target size for the letterboxed image. If an int, a square image of
                (size, size) is created. If a tuple, it should be (height, width).
            auto (bool): If True, automatically calculates the short side based on stride. Default is False.
            stride (int): The stride value, used when 'auto' is True. Default is 32.

        Attributes:
            h (int): Target height of the letterboxed image.
            w (int): Target width of the letterboxed image.
            auto (bool): Flag indicating whether to automatically calculate short side.
            stride (int): Stride value for automatic short side calculation.

        Examples:
            >>> transform = ClassifyLetterBox(size=224)
            >>> img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> result = transform(img)
            >>> print(result.shape)
            (224, 224, 3)
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size  # 设置目标图像的高度和宽度
        self.auto = auto  # 是否自动计算短边以及使用步幅值
        self.stride = stride  # 自动计算短边时使用的步幅值
    def __call__(self, im):
        """
        Resizes and pads an image using the letterbox method.

        This method resizes the input image to fit within the specified dimensions while maintaining its aspect ratio,
        then pads the resized image to match the target size.

        Args:
            im (numpy.ndarray): Input image as a numpy array with shape (H, W, C).

        Returns:
            (numpy.ndarray): Resized and padded image as a numpy array with shape (hs, ws, 3), where hs and ws are
                the target height and width respectively.

        Examples:
            >>> letterbox = ClassifyLetterBox(size=(640, 640))
            >>> image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            >>> resized_image = letterbox(image)
            >>> print(resized_image.shape)
            (640, 640, 3)
        """
        # Extract height and width of the input image
        imh, imw = im.shape[:2]
        
        # Calculate the resizing ratio based on the smaller dimension
        r = min(self.h / imh, self.w / imw)  # ratio of new/old dimensions
        
        # Compute the new dimensions after resizing
        h, w = round(imh * r), round(imw * r)  # resized image dimensions

        # Calculate padding dimensions based on stride or fixed size
        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else (self.h, self.w)
        
        # Calculate top and left padding offsets to center the image
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)

        # Create a new image with padding, initialized with a gray value of 114
        im_out = np.full((hs, ws, 3), 114, dtype=im.dtype)
        
        # Resize the original image and place it within the padded image
        im_out[top : top + h, left : left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Return the resized and padded image
        return im_out
# NOTE: keep this class for backward compatibility
class CenterCrop:
    """
    Applies center cropping to images for classification tasks.

    This class performs center cropping on input images, resizing them to a specified size while maintaining the aspect
    ratio. It is designed to be part of a transformation pipeline, e.g., T.Compose([CenterCrop(size), ToTensor()]).

    Attributes:
        h (int): Target height of the cropped image.
        w (int): Target width of the cropped image.

    Methods:
        __call__: Applies the center crop transformation to an input image.

    Examples:
        >>> transform = CenterCrop(640)
        >>> image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        >>> cropped_image = transform(image)
        >>> print(cropped_image.shape)
        (640, 640, 3)
    """

    def __init__(self, size=640):
        """
        Initializes the CenterCrop object for image preprocessing.

        This class is designed to be part of a transformation pipeline, e.g., T.Compose([CenterCrop(size), ToTensor()]).
        It performs a center crop on input images to a specified size.

        Args:
            size (int | Tuple[int, int]): The desired output size of the crop. If size is an int, a square crop
                (size, size) is made. If size is a sequence like (h, w), it is used as the output size.

        Returns:
            (None): This method initializes the object and does not return anything.

        Examples:
            >>> transform = CenterCrop(224)
            >>> img = np.random.rand(300, 300, 3)
            >>> cropped_img = transform(img)
            >>> print(cropped_img.shape)
            (224, 224, 3)
        """
        # 调用父类初始化方法
        super().__init__()
        # 如果指定的 size 是整数，则将高度和宽度设为相同的值
        self.h, self.w = (size, size) if isinstance(size, int) else size
    def __call__(self, im):
        """
        Applies center cropping to an input image.

        This method resizes and crops the center of the image using a letterbox method. It maintains the aspect
        ratio of the original image while fitting it into the specified dimensions.

        Args:
            im (numpy.ndarray | PIL.Image.Image): The input image as a numpy array of shape (H, W, C) or a
                PIL Image object.

        Returns:
            (numpy.ndarray): The center-cropped and resized image as a numpy array of shape (self.h, self.w, C).

        Examples:
            >>> transform = CenterCrop(size=224)
            >>> image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            >>> cropped_image = transform(image)
            >>> assert cropped_image.shape == (224, 224, 3)
        """
        # 如果输入的是 PIL Image 对象，则转换为 numpy 数组
        if isinstance(im, Image.Image):
            im = np.asarray(im)
        # 获取输入图像的高度和宽度
        imh, imw = im.shape[:2]
        # 计算较小的一维作为裁剪的边长
        m = min(imh, imw)
        # 计算裁剪后图像的左上角坐标
        top, left = (imh - m) // 2, (imw - m) // 2
        # 使用线性插值对裁剪后的图像进行缩放，得到指定尺寸的输出图像
        return cv2.resize(im[top:top + m, left:left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)
# NOTE: keep this class for backward compatibility
class ToTensor:
    """
    Converts an image from a numpy array to a PyTorch tensor.

    This class is designed to be part of a transformation pipeline, e.g., T.Compose([LetterBox(size), ToTensor()]).

    Attributes:
        half (bool): If True, converts the image to half precision (float16).

    Methods:
        __call__: Applies the tensor conversion to an input image.

    Examples:
        >>> transform = ToTensor(half=True)
        >>> img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        >>> tensor_img = transform(img)
        >>> print(tensor_img.shape, tensor_img.dtype)
        torch.Size([3, 640, 640]) torch.float16

    Notes:
        The input image is expected to be in BGR format with shape (H, W, C).
        The output tensor will be in RGB format with shape (C, H, W), normalized to [0, 1].
    """

    def __init__(self, half=False):
        """
        Initializes the ToTensor object for converting images to PyTorch tensors.

        This class is designed to be used as part of a transformation pipeline for image preprocessing in the
        Ultralytics YOLO framework. It converts numpy arrays or PIL Images to PyTorch tensors, with an option
        for half-precision (float16) conversion.

        Args:
            half (bool): If True, converts the tensor to half precision (float16). Default is False.

        Examples:
            >>> transform = ToTensor(half=True)
            >>> img = np.random.rand(640, 640, 3)
            >>> tensor_img = transform(img)
            >>> print(tensor_img.dtype)
            torch.float16
        """
        super().__init__()  # 调用父类初始化方法
        self.half = half  # 初始化对象属性，指示是否使用半精度

    def __call__(self, im):
        """
        Transforms an image from a numpy array to a PyTorch tensor.

        This method converts the input image from a numpy array to a PyTorch tensor, applying optional
        half-precision conversion and normalization. The image is transposed from HWC to CHW format and
        the color channels are reversed from BGR to RGB.

        Args:
            im (numpy.ndarray): Input image as a numpy array with shape (H, W, C) in BGR order.

        Returns:
            (torch.Tensor): The transformed image as a PyTorch tensor in float32 or float16, normalized
                to [0, 1] with shape (C, H, W) in RGB order.

        Examples:
            >>> transform = ToTensor(half=True)
            >>> img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            >>> tensor_img = transform(img)
            >>> print(tensor_img.shape, tensor_img.dtype)
            torch.Size([3, 640, 640]) torch.float16
        """
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # 将numpy数组转换为torch张量
        im = im.half() if self.half else im.float()  # 根据初始化时的half属性选择是否转换为半精度浮点数
        im /= 255.0  # 将像素值从0-255范围归一化到0.0-1.0范围
        return im
```