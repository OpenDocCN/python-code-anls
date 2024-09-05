# `.\yolov8\ultralytics\engine\results.py`

```py
# Ultralytics YOLO , AGPL-3.0 license
"""
Ultralytics Results, Boxes and Masks classes for handling inference results.

Usage: See https://docs.ultralytics.com/modes/predict/
"""

from copy import deepcopy                # 导入深拷贝函数deepcopy
from functools import lru_cache          # 导入LRU缓存函数lru_cache
from pathlib import Path                 # 导入处理路径的Path模块

import numpy as np                      # 导入NumPy库
import torch                            # 导入PyTorch库

from ultralytics.data.augment import LetterBox   # 导入augment模块中的LetterBox类
from ultralytics.utils import LOGGER, SimpleClass, ops   # 导入utils模块中的LOGGER、SimpleClass和ops
from ultralytics.utils.plotting import Annotator, colors, save_one_box   # 导入plotting模块中的Annotator、colors和save_one_box函数
from ultralytics.utils.torch_utils import smart_inference_mode   # 导入torch_utils模块中的smart_inference_mode函数


class BaseTensor(SimpleClass):
    """
    Base tensor class with additional methods for easy manipulation and device handling.

    Attributes:
        data (torch.Tensor | np.ndarray): Prediction data such as bounding boxes, masks, or keypoints.
        orig_shape (Tuple[int, int]): Original shape of the image, typically in the format (height, width).

    Methods:
        cpu: Return a copy of the tensor stored in CPU memory.
        numpy: Returns a copy of the tensor as a numpy array.
        cuda: Moves the tensor to GPU memory, returning a new instance if necessary.
        to: Return a copy of the tensor with the specified device and dtype.

    Examples:
        >>> import torch
        >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> orig_shape = (720, 1280)
        >>> base_tensor = BaseTensor(data, orig_shape)
        >>> cpu_tensor = base_tensor.cpu()
        >>> numpy_array = base_tensor.numpy()
        >>> gpu_tensor = base_tensor.cuda()
    """

    def __init__(self, data, orig_shape) -> None:
        """
        Initialize BaseTensor with prediction data and the original shape of the image.

        Args:
            data (torch.Tensor | np.ndarray): Prediction data such as bounding boxes, masks, or keypoints.
            orig_shape (Tuple[int, int]): Original shape of the image in (height, width) format.

        Examples:
            >>> import torch
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> orig_shape = (720, 1280)
            >>> base_tensor = BaseTensor(data, orig_shape)
        """
        assert isinstance(data, (torch.Tensor, np.ndarray)), "data must be torch.Tensor or np.ndarray"
        self.data = data                # 设置BaseTensor类的data属性为传入的data
        self.orig_shape = orig_shape    # 设置BaseTensor类的orig_shape属性为传入的orig_shape

    @property
    def shape(self):
        """
        Returns the shape of the underlying data tensor.

        Returns:
            (Tuple[int, ...]): The shape of the data tensor.

        Examples:
            >>> data = torch.rand(100, 4)
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> print(base_tensor.shape)
            (100, 4)
        """
        return self.data.shape           # 返回BaseTensor类的data属性的形状
    def cpu(self):
        """
        Returns a copy of the tensor stored in CPU memory.

        Returns:
            (BaseTensor): A new BaseTensor object with the data tensor moved to CPU memory.

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]]).cuda()
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> cpu_tensor = base_tensor.cpu()
            >>> isinstance(cpu_tensor, BaseTensor)
            True
            >>> cpu_tensor.data.device
            device(type='cpu')
        """
        # 如果数据已经是 numpy 数组，则直接返回 self
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)

    def numpy(self):
        """
        Returns a copy of the tensor as a numpy array.

        Returns:
            (np.ndarray): A numpy array containing the same data as the original tensor.

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> orig_shape = (720, 1280)
            >>> base_tensor = BaseTensor(data, orig_shape)
            >>> numpy_array = base_tensor.numpy()
            >>> print(type(numpy_array))
            <class 'numpy.ndarray'>
        """
        # 如果数据已经是 numpy 数组，则直接返回 self
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def cuda(self):
        """
        Moves the tensor to GPU memory.

        Returns:
            (BaseTensor): A new BaseTensor instance with the data moved to GPU memory if it's not already a
                numpy array, otherwise returns self.

        Examples:
            >>> import torch
            >>> from ultralytics.engine.results import BaseTensor
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> gpu_tensor = base_tensor.cuda()
            >>> print(gpu_tensor.data.device)
            cuda:0
        """
        # 将数据转换为 tensor，并移动到 GPU，然后创建新的 BaseTensor 实例返回
        return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape)

    def to(self, *args, **kwargs):
        """
        Return a copy of the tensor with the specified device and dtype.

        Args:
            *args (Any): Variable length argument list to be passed to torch.Tensor.to().
            **kwargs (Any): Arbitrary keyword arguments to be passed to torch.Tensor.to().

        Returns:
            (BaseTensor): A new BaseTensor instance with the data moved to the specified device and/or dtype.

        Examples:
            >>> base_tensor = BaseTensor(torch.randn(3, 4), orig_shape=(480, 640))
            >>> cuda_tensor = base_tensor.to('cuda')
            >>> float16_tensor = base_tensor.to(dtype=torch.float16)
        """
        # 将数据转换为 tensor，并按照参数指定的设备和数据类型进行转换，然后创建新的 BaseTensor 实例返回
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)
    # 重写 len(results)，返回基础数据张量的长度。
    def __len__(self):  # override len(results)
        """
        Returns the length of the underlying data tensor.

        Returns:
            (int): The number of elements in the first dimension of the data tensor.

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> len(base_tensor)
            2
        """
        # 返回数据张量第一维的元素个数
        return len(self.data)

    # 获取指定索引的数据，返回一个包含指定索引数据的新的 BaseTensor 实例
    def __getitem__(self, idx):
        """
        Returns a new BaseTensor instance containing the specified indexed elements of the data tensor.

        Args:
            idx (int | List[int] | torch.Tensor): Index or indices to select from the data tensor.

        Returns:
            (BaseTensor): A new BaseTensor instance containing the indexed data.

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> result = base_tensor[0]  # Select the first row
            >>> print(result.data)
            tensor([1, 2, 3])
        """
        # 返回包含指定索引数据的新 BaseTensor 实例
        return self.__class__(self.data[idx], self.orig_shape)
# Results 类用于存储和操作推理结果，继承自 SimpleClass 类
class Results(SimpleClass):
    """
    A class for storing and manipulating inference results.

    This class encapsulates the functionality for handling detection, segmentation, pose estimation,
    and classification results from YOLO models.

    Attributes:
        orig_img (numpy.ndarray): Original image as a numpy array. 原始图像的 numpy 数组表示
        orig_shape (Tuple[int, int]): Original image shape in (height, width) format. 原始图像的高度和宽度
        boxes (Boxes | None): Object containing detection bounding boxes. 检测框的对象，可能为 None
        masks (Masks | None): Object containing detection masks. 检测到的掩模的对象，可能为 None
        probs (Probs | None): Object containing class probabilities for classification tasks. 分类任务的类别概率的对象，可能为 None
        keypoints (Keypoints | None): Object containing detected keypoints for each object. 每个对象检测到的关键点的对象，可能为 None
        obb (OBB | None): Object containing oriented bounding boxes. 方向边界框的对象，可能为 None
        speed (Dict[str, float | None]): Dictionary of preprocess, inference, and postprocess speeds.
            预处理、推理和后处理速度的字典，包含字符串键和浮点数或 None 值
        names (Dict[int, str]): Dictionary mapping class IDs to class names. 将类别 ID 映射到类别名称的字典
        path (str): Path to the image file. 图像文件的路径
        _keys (Tuple[str, ...]): Tuple of attribute names for internal use. 内部使用的属性名称元组

    Methods:
        update: Updates object attributes with new detection results. 使用新的检测结果更新对象属性
        cpu: Returns a copy of the Results object with all tensors on CPU memory. 返回所有张量在 CPU 内存上的 Results 对象副本
        numpy: Returns a copy of the Results object with all tensors as numpy arrays. 返回所有张量作为 numpy 数组的 Results 对象副本
        cuda: Returns a copy of the Results object with all tensors on GPU memory. 返回所有张量在 GPU 内存上的 Results 对象副本
        to: Returns a copy of the Results object with tensors on a specified device and dtype. 返回指定设备和数据类型上的张量的 Results 对象副本
        new: Returns a new Results object with the same image, path, and names. 返回具有相同图像、路径和名称的新 Results 对象
        plot: Plots detection results on an input image, returning an annotated image. 在输入图像上绘制检测结果，返回带注释的图像
        show: Shows annotated results on screen. 在屏幕上显示带注释的结果
        save: Saves annotated results to file. 将带注释的结果保存到文件
        verbose: Returns a log string for each task, detailing detections and classifications. 返回每个任务的日志字符串，详细描述检测和分类
        save_txt: Saves detection results to a text file. 将检测结果保存到文本文件
        save_crop: Saves cropped detection images. 保存裁剪后的检测图像
        tojson: Converts detection results to JSON format. 将检测结果转换为 JSON 格式

    Examples:
        >>> results = model("path/to/image.jpg")
        >>> for result in results:
        ...     print(result.boxes)  # Print detection boxes 打印检测框
        ...     result.show()  # Display the annotated image 显示带注释的图像
        ...     result.save(filename='result.jpg')  # Save annotated image 保存带注释的图像
    """

    def __init__(
        self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None, obb=None, speed=None
        """
        Initialize the Results class for storing and manipulating inference results.

        Args:
            orig_img (numpy.ndarray): The original image as a numpy array.
            path (str): The path to the image file.
            names (Dict): A dictionary of class names.
            boxes (torch.Tensor | None): A 2D tensor of bounding box coordinates for each detection.
            masks (torch.Tensor | None): A 3D tensor of detection masks, where each mask is a binary image.
            probs (torch.Tensor | None): A 1D tensor of probabilities of each class for classification task.
            keypoints (torch.Tensor | None): A 2D tensor of keypoint coordinates for each detection.
            obb (torch.Tensor | None): A 2D tensor of oriented bounding box coordinates for each detection.
            speed (Dict | None): A dictionary containing preprocess, inference, and postprocess speeds (ms/image).

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> result = results[0]  # Get the first result
            >>> boxes = result.boxes  # Get the boxes for the first result
            >>> masks = result.masks  # Get the masks for the first result

        Notes:
            For the default pose model, keypoint indices for human body pose estimation are:
            0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
            5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow
            9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip
            13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle
        """
        # 存储原始图像的 numpy 数组
        self.orig_img = orig_img
        # 存储原始图像的形状（高度和宽度）
        self.orig_shape = orig_img.shape[:2]
        # 如果提供了边界框数据，则用边界框数据初始化 Boxes 类，否则为 None
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None  # native size boxes
        # 如果提供了掩码数据，则用掩码数据初始化 Masks 类，否则为 None
        self.masks = Masks(masks, self.orig_shape) if masks is not None else None  # native size or imgsz masks
        # 如果提供了概率数据，则用概率数据初始化 Probs 类，否则为 None
        self.probs = Probs(probs) if probs is not None else None
        # 如果提供了关键点数据，则用关键点数据初始化 Keypoints 类，否则为 None
        self.keypoints = Keypoints(keypoints, self.orig_shape) if keypoints is not None else None
        # 如果提供了方向边界框数据，则用方向边界框数据初始化 OBB 类，否则为 None
        self.obb = OBB(obb, self.orig_shape) if obb is not None else None
        # 如果提供了速度数据，则使用提供的速度数据，否则初始化为空字典
        self.speed = speed if speed is not None else {"preprocess": None, "inference": None, "postprocess": None}
        # 存储类别名称的字典
        self.names = names
        # 存储图像文件的路径
        self.path = path
        # 初始化保存目录为空
        self.save_dir = None
        # 存储需要公开的属性名称的元组
        self._keys = "boxes", "masks", "probs", "keypoints", "obb"
    def __getitem__(self, idx):
        """
        Return a Results object for a specific index of inference results.

        Args:
            idx (int | slice): Index or slice to retrieve from the Results object.

        Returns:
            (Results): A new Results object containing the specified subset of inference results.

        Examples:
            >>> results = model('path/to/image.jpg')  # Perform inference
            >>> single_result = results[0]  # Get the first result
            >>> subset_results = results[1:4]  # Get a slice of results
        """
        # 调用内部方法 _apply，用于处理索引操作
        return self._apply("__getitem__", idx)

    def __len__(self):
        """
        Return the number of detections in the Results object.

        Returns:
            (int): The number of detections, determined by the length of the first non-empty attribute
                (boxes, masks, probs, keypoints, or obb).

        Examples:
            >>> results = Results(orig_img, path, names, boxes=torch.rand(5, 4))
            >>> len(results)
            5
        """
        # 遍历 self._keys 中的属性，找到第一个非空属性并返回其长度
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                return len(v)

    def update(self, boxes=None, masks=None, probs=None, obb=None):
        """
        Updates the Results object with new detection data.

        This method allows updating the boxes, masks, probabilities, and oriented bounding boxes (OBB) of the
        Results object. It ensures that boxes are clipped to the original image shape.

        Args:
            boxes (torch.Tensor | None): A tensor of shape (N, 6) containing bounding box coordinates and
                confidence scores. The format is (x1, y1, x2, y2, conf, class).
            masks (torch.Tensor | None): A tensor of shape (N, H, W) containing segmentation masks.
            probs (torch.Tensor | None): A tensor of shape (num_classes,) containing class probabilities.
            obb (torch.Tensor | None): A tensor of shape (N, 5) containing oriented bounding box coordinates.

        Examples:
            >>> results = model('image.jpg')
            >>> new_boxes = torch.tensor([[100, 100, 200, 200, 0.9, 0]])
            >>> results[0].update(boxes=new_boxes)
        """
        # 如果参数不为 None，则更新相应的属性值
        if boxes is not None:
            # 更新 boxes 属性，确保边界框被剪裁到原始图像形状内
            self.boxes = Boxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)
        if masks is not None:
            # 更新 masks 属性
            self.masks = Masks(masks, self.orig_shape)
        if probs is not None:
            # 更新 probs 属性
            self.probs = probs
        if obb is not None:
            # 更新 obb 属性
            self.obb = OBB(obb, self.orig_shape)
    def _apply(self, fn, *args, **kwargs):
        """
        Applies a function to all non-empty attributes and returns a new Results object with modified attributes.

        This method is internally called by methods like .to(), .cuda(), .cpu(), etc.

        Args:
            fn (str): The name of the function to apply.
            *args (Any): Variable length argument list to pass to the function.
            **kwargs (Any): Arbitrary keyword arguments to pass to the function.

        Returns:
            (Results): A new Results object with attributes modified by the applied function.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result_cuda = result.cuda()
            ...     result_cpu = result.cpu()
        """
        # 创建一个新的 Results 对象，用于存储应用函数后的结果
        r = self.new()
        # 遍历当前对象的所有属性名
        for k in self._keys:
            # 获取当前属性的值
            v = getattr(self, k)
            # 如果属性值不为 None，则对其调用指定的函数，并将结果设置到新的 Results 对象中对应的属性
            if v is not None:
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
        # 返回经过函数应用后的新 Results 对象
        return r

    def cpu(self):
        """
        Returns a copy of the Results object with all its tensors moved to CPU memory.

        This method creates a new Results object with all tensor attributes (boxes, masks, probs, keypoints, obb)
        transferred to CPU memory. It's useful for moving data from GPU to CPU for further processing or saving.

        Returns:
            (Results): A new Results object with all tensor attributes on CPU memory.

        Examples:
            >>> results = model('path/to/image.jpg')  # Perform inference
            >>> cpu_result = results[0].cpu()  # Move the first result to CPU
            >>> print(cpu_result.boxes.device)  # Output: cpu
        """
        # 调用 _apply 方法，将所有张量移到 CPU 上
        return self._apply("cpu")

    def numpy(self):
        """
        Converts all tensors in the Results object to numpy arrays.

        Returns:
            (Results): A new Results object with all tensors converted to numpy arrays.

        Examples:
            >>> results = model('path/to/image.jpg')
            >>> numpy_result = results[0].numpy()
            >>> type(numpy_result.boxes.data)
            <class 'numpy.ndarray'>

        Notes:
            This method creates a new Results object, leaving the original unchanged. It's useful for
            interoperability with numpy-based libraries or when CPU-based operations are required.
        """
        # 调用 _apply 方法，将所有张量转换为 numpy 数组
        return self._apply("numpy")

    def cuda(self):
        """
        Moves all tensors in the Results object to GPU memory.

        Returns:
            (Results): A new Results object with all tensors moved to CUDA device.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> cuda_results = results[0].cuda()  # Move first result to GPU
            >>> for result in results:
            ...     result_cuda = result.cuda()  # Move each result to GPU
        """
        # 调用 _apply 方法，将所有张量移到 GPU 上
        return self._apply("cuda")
    def to(self, *args, **kwargs):
        """
        Moves all tensors in the Results object to the specified device and dtype.

        Args:
            *args (Any): Variable length argument list to be passed to torch.Tensor.to().
            **kwargs (Any): Arbitrary keyword arguments to be passed to torch.Tensor.to().

        Returns:
            (Results): A new Results object with all tensors moved to the specified device and dtype.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> result_cuda = results[0].to("cuda")  # Move first result to GPU
            >>> result_cpu = results[0].to("cpu")  # Move first result to CPU
            >>> result_half = results[0].to(dtype=torch.float16)  # Convert first result to half precision
        """
        # 调用私有方法 _apply()，将指定参数传递给它
        return self._apply("to", *args, **kwargs)

    def new(self):
        """
        Creates a new Results object with the same image, path, names, and speed attributes.

        Returns:
            (Results): A new Results object with copied attributes from the original instance.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> new_result = results[0].new()
        """
        # 使用当前实例的属性创建一个新的 Results 对象并返回
        return Results(orig_img=self.orig_img, path=self.path, names=self.names, speed=self.speed)

    def plot(
        self,
        conf=True,
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        pil=False,
        img=None,
        im_gpu=None,
        kpt_radius=5,
        kpt_line=True,
        labels=True,
        boxes=True,
        masks=True,
        probs=True,
        show=False,
        save=False,
        filename=None,
    ):
        """
        Plotting method for visualizing inference results on an image.

        Args:
            conf (bool): Whether to display confidence scores.
            line_width (int or None): Width of lines to draw (default: None).
            font_size (int or None): Size of the font for labels (default: None).
            font (str): Font type for labels (default: "Arial.ttf").
            pil (bool): Whether to use PIL for plotting (default: False).
            img (PIL.Image.Image or None): Optional PIL image to plot results on.
            im_gpu (torch.Tensor or None): Optional GPU tensor image to plot results on.
            kpt_radius (int): Radius of keypoints markers (default: 5).
            kpt_line (bool): Whether to draw lines between keypoints (default: True).
            labels (bool): Whether to display class labels (default: True).
            boxes (bool): Whether to display bounding boxes (default: True).
            masks (bool): Whether to display masks (default: True).
            probs (bool): Whether to display probabilities/confidence scores (default: True).
            show (bool): Whether to display the plot (default: False).
            save (bool): Whether to save the plot to a file (default: False).
            filename (str or None): Optional filename to save the plot as.

        Returns:
            None

        Notes:
            This method allows visualization of object detection results on an image. It supports various options
            such as displaying bounding boxes, masks, keypoints, labels, and confidence scores.

        Examples:
            >>> results = model('path/to/image.jpg')
            >>> results[0].plot(show=True)  # Plot and display the first result
        """

    def show(self, *args, **kwargs):
        """
        Display the image with annotated inference results.

        This method plots the detection results on the original image and displays it. It's a convenient way to
        visualize the model's predictions directly.

        Args:
            *args (Any): Variable length argument list to be passed to the `plot()` method.
            **kwargs (Any): Arbitrary keyword arguments to be passed to the `plot()` method.

        Returns:
            None

        Examples:
            >>> results = model('path/to/image.jpg')
            >>> results[0].show()  # Display the first result
            >>> for result in results:
            ...     result.show()  # Display all results
        """
        # 调用当前对象的 plot 方法，并传递参数 show=True 及其他参数
        self.plot(show=True, *args, **kwargs)
    def save(self, filename=None, *args, **kwargs):
        """
        Saves annotated inference results image to file.

        This method plots the detection results on the original image and saves the annotated image to a file. It
        utilizes the `plot` method to generate the annotated image and then saves it to the specified filename.

        Args:
            filename (str | Path | None): The filename to save the annotated image. If None, a default filename
                is generated based on the original image path.
            *args (Any): Variable length argument list to be passed to the `plot` method.
            **kwargs (Any): Arbitrary keyword arguments to be passed to the `plot` method.

        Examples:
            >>> results = model('path/to/image.jpg')
            >>> for result in results:
            ...     result.save('annotated_image.jpg')
            >>> # Or with custom plot arguments
            >>> for result in results:
            ...     result.save('annotated_image.jpg', conf=False, line_width=2)
        """
        # 如果没有指定文件名，基于原始图像路径生成默认文件名
        if not filename:
            filename = f"results_{Path(self.path).name}"
        # 调用 plot 方法生成带有标注的图像，并保存到指定的文件名
        self.plot(save=True, filename=filename, *args, **kwargs)
        # 返回保存的文件名
        return filename

    def verbose(self):
        """
        Returns a log string for each task in the results, detailing detection and classification outcomes.

        This method generates a human-readable string summarizing the detection and classification results. It includes
        the number of detections for each class and the top probabilities for classification tasks.

        Returns:
            (str): A formatted string containing a summary of the results. For detection tasks, it includes the
                number of detections per class. For classification tasks, it includes the top 5 class probabilities.

        Examples:
            >>> results = model('path/to/image.jpg')
            >>> for result in results:
            ...     print(result.verbose())
            2 persons, 1 car, 3 traffic lights,
            dog 0.92, cat 0.78, horse 0.64,

        Notes:
            - If there are no detections, the method returns "(no detections), " for detection tasks.
            - For classification tasks, it returns the top 5 class probabilities and their corresponding class names.
            - The returned string is comma-separated and ends with a comma and a space.
        """
        # 初始化日志字符串
        log_string = ""
        # 获取概率和边界框信息
        probs = self.probs
        boxes = self.boxes
        # 如果没有检测到结果，根据情况返回空字符串或者“(no detections), ”
        if len(self) == 0:
            return log_string if probs is None else f"{log_string}(no detections), "
        # 如果有概率信息，将前五个概率最高的类别及其概率加入日志字符串
        if probs is not None:
            log_string += f"{', '.join(f'{self.names[j]} {probs.data[j]:.2f}' for j in probs.top5)}, "
        # 如果有边界框信息，计算每个类别的检测数量，并加入日志字符串
        if boxes:
            for c in boxes.cls.unique():
                n = (boxes.cls == c).sum()  # detections per class
                log_string += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
        # 返回生成的日志字符串
        return log_string
    # 将检测结果保存到文本文件中
    def save_txt(self, txt_file, save_conf=False):
        """
        Save detection results to a text file.

        Args:
            txt_file (str | Path): Path to the output text file.
            save_conf (bool): Whether to include confidence scores in the output.

        Returns:
            (str): Path to the saved text file.

        Examples:
            >>> from ultralytics import YOLO
            >>> model = YOLO('yolov8n.pt')
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result.save_txt("output.txt")

        Notes:
            - The file will contain one line per detection or classification with the following structure:
              - For detections: `class confidence x_center y_center width height`
              - For classifications: `confidence class_name`
              - For masks and keypoints, the specific formats will vary accordingly.
            - The function will create the output directory if it does not exist.
            - If save_conf is False, the confidence scores will be excluded from the output.
            - Existing contents of the file will not be overwritten; new results will be appended.
        """
        # 检查是否使用旋转边界框
        is_obb = self.obb is not None
        # 获取边界框、掩模、概率、关键点的数据
        boxes = self.obb if is_obb else self.boxes
        masks = self.masks
        probs = self.probs
        kpts = self.keypoints
        # 保存文本结果的列表
        texts = []
        # 如果有概率信息
        if probs is not None:
            # 对每个类别按置信度保存
            [texts.append(f"{probs.data[j]:.2f} {self.names[j]}") for j in probs.top5]
        # 如果有边界框信息
        elif boxes:
            # 对每个边界框进行处理
            for j, d in enumerate(boxes):
                # 获取类别、置信度、ID
                c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
                # 构造文本行
                line = (c, *(d.xyxyxyxyn.view(-1) if is_obb else d.xywhn.view(-1)))
                # 如果有掩模信息
                if masks:
                    # 处理掩模数据
                    seg = masks[j].xyn[0].copy().reshape(-1)  # reversed mask.xyn, (n,2) to (n*2)
                    line = (c, *seg)
                # 如果有关键点信息
                if kpts is not None:
                    kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
                    line += (*kpt.reshape(-1).tolist(),)
                # 根据是否保存置信度，添加到文本行末尾
                line += (conf,) * save_conf + (() if id is None else (id,))
                # 格式化文本行并添加到文本列表
                texts.append(("%g " * len(line)).rstrip() % line)

        # 如果有文本结果
        if texts:
            # 确保输出目录存在
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
            # 将文本行写入文件
            with open(txt_file, "a") as f:
                f.writelines(text + "\n" for text in texts)
    # 将检测到的物体裁剪图像保存到指定目录

    # 如果 self.probs 不为 None，输出警告信息并返回，因为不支持分类任务
    if self.probs is not None:
        LOGGER.warning("WARNING ⚠️ Classify task do not support `save_crop`.")
        return

    # 如果 self.obb 不为 None，输出警告信息并返回，因为不支持有向边界框任务
    if self.obb is not None:
        LOGGER.warning("WARNING ⚠️ OBB task do not support `save_crop`.")
        return

    # 遍历每个检测到的框
    for d in self.boxes:
        # 调用 save_one_box 函数保存单个框的裁剪图像
        # 使用 d.xyxy 作为边界框坐标
        # 使用 self.orig_img 的副本进行裁剪，避免修改原始图像
        # 生成的文件路径为 save_dir/class_name/file_name.jpg
        save_one_box(
            d.xyxy,
            self.orig_img.copy(),
            file=Path(save_dir) / self.names[int(d.cls)] / f"{Path(file_name)}.jpg",
            BGR=True,
        )
    def tojson(self, normalize=False, decimals=5):
        """
        Converts detection results to JSON format.

        This method serializes the detection results into a JSON-compatible format. It includes information
        about detected objects such as bounding boxes, class names, confidence scores, and optionally
        segmentation masks and keypoints.

        Args:
            normalize (bool): Whether to normalize the bounding box coordinates by the image dimensions.
                If True, coordinates will be returned as float values between 0 and 1. Defaults to False.
            decimals (int): Number of decimal places to round the output values to. Defaults to 5.

        Returns:
            (str): A JSON string containing the serialized detection results.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> json_result = results[0].tojson()
            >>> print(json_result)

        Notes:
            - For classification tasks, the JSON will contain class probabilities instead of bounding boxes.
            - For object detection tasks, the JSON will include bounding box coordinates, class names, and
              confidence scores.
            - If available, segmentation masks and keypoints will also be included in the JSON output.
            - The method uses the `summary` method internally to generate the data structure before
              converting it to JSON.
        """
        import json  # 导入 JSON 模块

        # 调用对象的 `summary` 方法生成数据结构，并转换成 JSON 格式字符串，缩进为 2
        return json.dumps(self.summary(normalize=normalize, decimals=decimals), indent=2)
# 定义一个 Boxes 类，继承自 BaseTensor 类，用于管理和操作检测框。

"""
A class for managing and manipulating detection boxes.

This class provides functionality for handling detection boxes, including their coordinates, confidence scores,
class labels, and optional tracking IDs. It supports various box formats and offers methods for easy manipulation
and conversion between different coordinate systems.

Attributes:
    data (torch.Tensor | numpy.ndarray): The raw tensor containing detection boxes and associated data.
    orig_shape (Tuple[int, int]): The original image dimensions (height, width).
    is_track (bool): Indicates whether tracking IDs are included in the box data.
    xyxy (torch.Tensor | numpy.ndarray): Boxes in [x1, y1, x2, y2] format.
    conf (torch.Tensor | numpy.ndarray): Confidence scores for each box.
    cls (torch.Tensor | numpy.ndarray): Class labels for each box.
    id (torch.Tensor | numpy.ndarray): Tracking IDs for each box (if available).
    xywh (torch.Tensor | numpy.ndarray): Boxes in [x, y, width, height] format.
    xyxyn (torch.Tensor | numpy.ndarray): Normalized [x1, y1, x2, y2] boxes relative to orig_shape.
    xywhn (torch.Tensor | numpy.ndarray): Normalized [x, y, width, height] boxes relative to orig_shape.

Methods:
    cpu(): Returns a copy of the object with all tensors on CPU memory.
    numpy(): Returns a copy of the object with all tensors as numpy arrays.
    cuda(): Returns a copy of the object with all tensors on GPU memory.
    to(*args, **kwargs): Returns a copy of the object with tensors on specified device and dtype.

Examples:
    >>> import torch
    >>> boxes_data = torch.tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]])
    >>> orig_shape = (480, 640)  # height, width
    >>> boxes = Boxes(boxes_data, orig_shape)
    >>> print(boxes.xyxy)
    >>> print(boxes.conf)
    >>> print(boxes.cls)
    >>> print(boxes.xywhn)
"""
    def __init__(self, boxes, orig_shape) -> None:
        """
        Initialize the Boxes class with detection box data and the original image shape.

        This class manages detection boxes, providing easy access and manipulation of box coordinates,
        confidence scores, class identifiers, and optional tracking IDs. It supports multiple formats
        for box coordinates, including both absolute and normalized forms.

        Args:
            boxes (torch.Tensor | np.ndarray): A tensor or numpy array with detection boxes of shape
                (num_boxes, 6) or (num_boxes, 7). Columns should contain
                [x1, y1, x2, y2, confidence, class, (optional) track_id].
            orig_shape (Tuple[int, int]): The original image shape as (height, width). Used for normalization.

        Attributes:
            data (torch.Tensor): The raw tensor containing detection boxes and their associated data.
            orig_shape (Tuple[int, int]): The original image size, used for normalization.
            is_track (bool): Indicates whether tracking IDs are included in the box data.

        Examples:
            >>> import torch
            >>> boxes = torch.tensor([[100, 50, 150, 100, 0.9, 0]])
            >>> orig_shape = (480, 640)
            >>> detection_boxes = Boxes(boxes, orig_shape)
            >>> print(detection_boxes.xyxy)
            tensor([[100.,  50., 150., 100.]])
        """
        # 如果输入的 boxes 是一维的，则将其转换为二维的
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        # 检查 boxes 的最后一个维度是否为 6 或 7，分别对应 xyxy, track_id, conf, cls 的不同格式
        n = boxes.shape[-1]
        assert n in {6, 7}, f"expected 6 or 7 values but got {n}"  # xyxy, track_id, conf, cls
        # 调用父类的初始化方法，将 boxes 和 orig_shape 传入
        super().__init__(boxes, orig_shape)
        # 设置是否包含 track_id 的标志位
        self.is_track = n == 7
        # 设置原始图像的形状信息
        self.orig_shape = orig_shape

    @property
    def xyxy(self):
        """
        Returns bounding boxes in [x1, y1, x2, y2] format.

        Returns:
            (torch.Tensor | numpy.ndarray): A tensor or numpy array of shape (n, 4) containing bounding box
                coordinates in [x1, y1, x2, y2] format, where n is the number of boxes.

        Examples:
            >>> results = model('image.jpg')
            >>> boxes = results[0].boxes
            >>> xyxy = boxes.xyxy
            >>> print(xyxy)
        """
        # 返回数据中的前四列，即 [x1, y1, x2, y2] 形式的边界框坐标
        return self.data[:, :4]

    @property
    def conf(self):
        """
        Returns the confidence scores for each detection box.

        Returns:
            (torch.Tensor | numpy.ndarray): A 1D tensor or array containing confidence scores for each detection,
                with shape (N,) where N is the number of detections.

        Examples:
            >>> boxes = Boxes(torch.tensor([[10, 20, 30, 40, 0.9, 0]]), orig_shape=(100, 100))
            >>> conf_scores = boxes.conf
            >>> print(conf_scores)
            tensor([0.9000])
        """
        # 返回数据中倒数第二列，即检测框的置信度分数
        return self.data[:, -2]

    @property


这段代码是一个 Python 类的初始化方法和其属性方法的定义。初始化方法用于设置检测框数据和原始图像形状，而属性方法分别用于获取边界框坐标和置信度分数。
    def cls(self):
        """
        Returns the class ID tensor representing category predictions for each bounding box.

        Returns:
            (torch.Tensor | numpy.ndarray): A tensor or numpy array containing the class IDs for each detection box.
                The shape is (N,), where N is the number of boxes.

        Examples:
            >>> results = model('image.jpg')
            >>> boxes = results[0].boxes
            >>> class_ids = boxes.cls
            >>> print(class_ids)  # tensor([0., 2., 1.])
        """
        # 返回每个检测框的类别ID张量，这些ID表示每个框所属的类别
        return self.data[:, -1]

    @property
    def id(self):
        """
        Returns the tracking IDs for each detection box if available.

        Returns:
            (torch.Tensor | None): A tensor containing tracking IDs for each box if tracking is enabled,
                otherwise None. Shape is (N,) where N is the number of boxes.

        Examples:
            >>> results = model.track('path/to/video.mp4')
            >>> for result in results:
            ...     boxes = result.boxes
            ...     if boxes.is_track:
            ...         track_ids = boxes.id
            ...         print(f"Tracking IDs: {track_ids}")
            ...     else:
            ...         print("Tracking is not enabled for these boxes.")

        Notes:
            - This property is only available when tracking is enabled (i.e., when `is_track` is True).
            - The tracking IDs are typically used to associate detections across multiple frames in video analysis.
        """
        # 如果启用了跟踪（即 `is_track` 为True），返回每个检测框的跟踪ID张量；否则返回None
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)  # maxsize 1 should suffice
    def xywh(self):
        """
        Convert bounding boxes from [x1, y1, x2, y2] format to [x, y, width, height] format.

        Returns:
            (torch.Tensor | numpy.ndarray): Boxes in [x, y, width, height] format, where x, y are the coordinates of
                the top-left corner of the bounding box, width, height are the dimensions of the bounding box and the
                shape of the returned tensor is (N, 4), where N is the number of boxes.

        Examples:
            >>> boxes = Boxes(torch.tensor([[100, 50, 150, 100], [200, 150, 300, 250]]), orig_shape=(480, 640))
            >>> xywh = boxes.xywh
            >>> print(xywh)
            tensor([[100.0000,  50.0000,  50.0000,  50.0000],
                    [200.0000, 150.0000, 100.0000, 100.0000]])
        """
        # 将边界框从[x1, y1, x2, y2]格式转换为[x, y, width, height]格式
        return ops.xyxy2xywh(self.xyxy)

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self):
        """
        Returns normalized bounding box coordinates relative to the original image size.

        This property calculates and returns the bounding box coordinates in [x1, y1, x2, y2] format,
        normalized to the range [0, 1] based on the original image dimensions.

        Returns:
            (torch.Tensor | numpy.ndarray): Normalized bounding box coordinates with shape (N, 4), where N is
                the number of boxes. Each row contains [x1, y1, x2, y2] values normalized to [0, 1].

        Examples:
            >>> boxes = Boxes(torch.tensor([[100, 50, 300, 400, 0.9, 0]]), orig_shape=(480, 640))
            >>> normalized = boxes.xyxyn
            >>> print(normalized)
            tensor([[0.1562, 0.1042, 0.4688, 0.8333]])
        """
        # Clone the bounding box tensor if it's a torch.Tensor; otherwise, create a numpy copy
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        # Normalize x1 and x2 coordinates by dividing with the width of the original image
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        # Normalize y1 and y2 coordinates by dividing with the height of the original image
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        # Return the normalized bounding box coordinates
        return xyxy

    @property
    @lru_cache(maxsize=2)
    def xywhn(self):
        """
        Returns normalized bounding boxes in [x, y, width, height] format.

        This property calculates and returns the normalized bounding box coordinates in the format
        [x_center, y_center, width, height], where all values are relative to the original image dimensions.

        Returns:
            (torch.Tensor | numpy.ndarray): Normalized bounding boxes with shape (N, 4), where N is the
                number of boxes. Each row contains [x_center, y_center, width, height] values normalized
                to [0, 1] based on the original image dimensions.

        Examples:
            >>> boxes = Boxes(torch.tensor([[100, 50, 150, 100, 0.9, 0]]), orig_shape=(480, 640))
            >>> normalized = boxes.xywhn
            >>> print(normalized)
            tensor([[0.1953, 0.1562, 0.0781, 0.1042]])
        """
        # Convert bounding box coordinates from [x1, y1, x2, y2] to [x_center, y_center, width, height]
        xywh = ops.xyxy2xywh(self.xyxy)
        # Normalize x_center and width by dividing with the width of the original image
        xywh[..., [0, 2]] /= self.orig_shape[1]
        # Normalize y_center and height by dividing with the height of the original image
        xywh[..., [1, 3]] /= self.orig_shape[0]
        # Return the normalized bounding boxes in [x, y, width, height] format
        return xywh
    @property
    @lru_cache(maxsize=1)
    def xy(self) -> List[np.ndarray]:
        """
        Property method that caches and returns a list of pixel coordinates of segmentation masks.

        Returns:
            List[np.ndarray]: A list where each element is a numpy array representing pixel coordinates
                              of a segmentation mask.
        """
        return [self.data[i].nonzero()[:, :2] for i in range(self.data.shape[0])]
    # 定义一个方法 xyn，用于返回分割掩模的归一化 xy 坐标

    def xyn(self):
        """
        Returns normalized xy-coordinates of the segmentation masks.

        This property calculates and caches the normalized xy-coordinates of the segmentation masks. The coordinates
        are normalized relative to the original image shape.

        Returns:
            (List[numpy.ndarray]): A list of numpy arrays, where each array contains the normalized xy-coordinates
                of a single segmentation mask. Each array has shape (N, 2), where N is the number of points in the
                mask contour.

        Examples:
            >>> results = model('image.jpg')
            >>> masks = results[0].masks
            >>> normalized_coords = masks.xyn
            >>> print(normalized_coords[0])  # Normalized coordinates of the first mask
        """
        # 使用 ops.masks2segments 方法将分割数据转换为分割段的列表，并对每个段的坐标进行归一化处理
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=True)
            for x in ops.masks2segments(self.data)
        ]

    @property
    @lru_cache(maxsize=1)
    # 使用 property 装饰器定义一个只读属性 xy，返回每个掩模张量中每个分割的像素坐标 [x, y]
    def xy(self):
        """
        Returns the [x, y] pixel coordinates for each segment in the mask tensor.

        This property calculates and returns a list of pixel coordinates for each segmentation mask in the
        Masks object. The coordinates are scaled to match the original image dimensions.

        Returns:
            (List[numpy.ndarray]): A list of numpy arrays, where each array contains the [x, y] pixel
                coordinates for a single segmentation mask. Each array has shape (N, 2), where N is the
                number of points in the segment.

        Examples:
            >>> results = model('image.jpg')
            >>> masks = results[0].masks
            >>> xy_coords = masks.xy
            >>> print(len(xy_coords))  # Number of masks
            >>> print(xy_coords[0].shape)  # Shape of first mask's coordinates
        """
        # 使用 ops.masks2segments 方法将分割数据转换为分割段的列表，并根据原始图像尺寸对每个段的坐标进行缩放处理
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=False)
            for x in ops.masks2segments(self.data)
        ]
# 定义 Keypoints 类，继承自 BaseTensor
class Keypoints(BaseTensor):
    """
    A class for storing and manipulating detection keypoints.

    This class encapsulates functionality for handling keypoint data, including coordinate manipulation,
    normalization, and confidence values.

    Attributes:
        data (torch.Tensor): The raw tensor containing keypoint data.
        orig_shape (Tuple[int, int]): The original image dimensions (height, width).
        has_visible (bool): Indicates whether visibility information is available for keypoints.
        xy (torch.Tensor): Keypoint coordinates in [x, y] format.
        xyn (torch.Tensor): Normalized keypoint coordinates in [x, y] format, relative to orig_shape.
        conf (torch.Tensor): Confidence values for each keypoint, if available.

    Methods:
        cpu(): Returns a copy of the keypoints tensor on CPU memory.
        numpy(): Returns a copy of the keypoints tensor as a numpy array.
        cuda(): Returns a copy of the keypoints tensor on GPU memory.
        to(*args, **kwargs): Returns a copy of the keypoints tensor with specified device and dtype.

    Examples:
        >>> import torch
        >>> from ultralytics.engine.results import Keypoints
        >>> keypoints_data = torch.rand(1, 17, 3)  # 1 detection, 17 keypoints, (x, y, conf)
        >>> orig_shape = (480, 640)  # Original image shape (height, width)
        >>> keypoints = Keypoints(keypoints_data, orig_shape)
        >>> print(keypoints.xy.shape)  # Access xy coordinates
        >>> print(keypoints.conf)  # Access confidence values
        >>> keypoints_cpu = keypoints.cpu()  # Move keypoints to CPU
    """

    # 使用 smart_inference_mode 装饰器，用于避免在处理 keypoints 时出现小于 confidence 阈值的情况
    @smart_inference_mode()
    def __init__(self, keypoints, orig_shape) -> None:
        """
        Initializes the Keypoints object with detection keypoints and original image dimensions.

        This method processes the input keypoints tensor, handling both 2D and 3D formats. For 3D tensors
        (x, y, confidence), it masks out low-confidence keypoints by setting their coordinates to zero.

        Args:
            keypoints (torch.Tensor): A tensor containing keypoint data. Shape can be either:
                - (num_objects, num_keypoints, 2) for x, y coordinates only
                - (num_objects, num_keypoints, 3) for x, y coordinates and confidence scores
            orig_shape (Tuple[int, int]): The original image dimensions (height, width).

        Examples:
            >>> kpts = torch.rand(1, 17, 3)  # 1 object, 17 keypoints (COCO format), x,y,conf
            >>> orig_shape = (720, 1280)  # Original image height, width
            >>> keypoints = Keypoints(kpts, orig_shape)
        """
        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]  # 将2维关键点数据转换为3维（添加一个维度）
        if keypoints.shape[2] == 3:  # x, y, conf
            mask = keypoints[..., 2] < 0.5  # 创建掩码，标记低置信度的关键点
            keypoints[..., :2][mask] = 0  # 将低置信度关键点的坐标设置为0
        super().__init__(keypoints, orig_shape)  # 调用父类的初始化方法
        self.has_visible = self.data.shape[-1] == 3  # 检查是否包含可见关键点信息

    @property
    @lru_cache(maxsize=1)
    def xy(self):
        """
        Returns x, y coordinates of keypoints.

        Returns:
            (torch.Tensor): A tensor containing the x, y coordinates of keypoints with shape (N, K, 2), where N is
                the number of detections and K is the number of keypoints per detection.

        Examples:
            >>> results = model('image.jpg')
            >>> keypoints = results[0].keypoints
            >>> xy = keypoints.xy
            >>> print(xy.shape)  # (N, K, 2)
            >>> print(xy[0])  # x, y coordinates of keypoints for first detection

        Notes:
            - The returned coordinates are in pixel units relative to the original image dimensions.
            - If keypoints were initialized with confidence values, only keypoints with confidence >= 0.5 are returned.
            - This property uses LRU caching to improve performance on repeated access.
        """
        return self.data[..., :2]  # 返回关键点的 x, y 坐标信息

    @property
    @lru_cache(maxsize=1)


这些注释解释了初始化方法和两个属性方法的功能及其作用，确保代码的每一部分都得到了清晰的解释。
    def xyn(self):
        """
        Returns normalized coordinates (x, y) of keypoints relative to the original image size.

        Returns:
            (torch.Tensor | numpy.ndarray): A tensor or array of shape (N, K, 2) containing normalized keypoint
                coordinates, where N is the number of instances, K is the number of keypoints, and the last
                dimension contains [x, y] values in the range [0, 1].

        Examples:
            >>> keypoints = Keypoints(torch.rand(1, 17, 2), orig_shape=(480, 640))
            >>> normalized_kpts = keypoints.xyn
            >>> print(normalized_kpts.shape)
            torch.Size([1, 17, 2])
        """
        # Clone the keypoint coordinates if they are a torch.Tensor; otherwise, create a numpy copy
        xy = self.xy.clone() if isinstance(self.xy, torch.Tensor) else np.copy(self.xy)
        # Normalize x coordinates by dividing by the width of the original image
        xy[..., 0] /= self.orig_shape[1]
        # Normalize y coordinates by dividing by the height of the original image
        xy[..., 1] /= self.orig_shape[0]
        # Return the normalized keypoint coordinates
        return xy

    @property
    @lru_cache(maxsize=1)
    def conf(self):
        """
        Returns confidence values for each keypoint.

        Returns:
            (torch.Tensor | None): A tensor containing confidence scores for each keypoint if available,
                otherwise None. Shape is (num_detections, num_keypoints) for batched data or (num_keypoints,)
                for single detection.

        Examples:
            >>> keypoints = Keypoints(torch.rand(1, 17, 3), orig_shape=(640, 640))  # 1 detection, 17 keypoints
            >>> conf = keypoints.conf
            >>> print(conf.shape)  # torch.Size([1, 17])
        """
        # Return confidence scores if keypoints have visibility information; otherwise return None
        return self.data[..., 2] if self.has_visible else None
    @property
    @lru_cache(maxsize=1)
    def top1(self):
        """
        Return the index of the class with the highest probability.

        This property computes and returns the index of the class with the highest probability
        from the stored classification probabilities.

        Returns:
            int: Index of the class with the highest probability.
        """
    def top1(self):
        """
        Returns the index of the class with the highest probability.

        Returns:
            (int): Index of the class with the highest probability.

        Examples:
            >>> probs = Probs(torch.tensor([0.1, 0.3, 0.6]))
            >>> probs.top1
            2
        """
        # 返回数据张量中最大值的索引，即最高概率对应的类别索引
        return int(self.data.argmax())

    @property
    @lru_cache(maxsize=1)
    def top5(self):
        """
        Returns the indices of the top 5 class probabilities.

        Returns:
            (List[int]): A list containing the indices of the top 5 class probabilities, sorted in descending order.

        Examples:
            >>> probs = Probs(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))
            >>> print(probs.top5)
            [4, 3, 2, 1, 0]
        """
        # 返回数据张量中前五个最大值的索引列表，按降序排列
        return (-self.data).argsort(0)[:5].tolist()  # this way works with both torch and numpy.

    @property
    @lru_cache(maxsize=1)
    def top1conf(self):
        """
        Returns the confidence score of the highest probability class.

        This property retrieves the confidence score (probability) of the class with the highest predicted probability
        from the classification results.

        Returns:
            (torch.Tensor | numpy.ndarray): A tensor containing the confidence score of the top 1 class.

        Examples:
            >>> results = model('image.jpg')  # classify an image
            >>> probs = results[0].probs  # get classification probabilities
            >>> top1_confidence = probs.top1conf  # get confidence of top 1 class
            >>> print(f"Top 1 class confidence: {top1_confidence.item():.4f}")
        """
        # 返回数据张量中最高概率类别的置信度分数
        return self.data[self.top1]

    @property
    @lru_cache(maxsize=1)
    def top5conf(self):
        """
        Returns confidence scores for the top 5 classification predictions.

        This property retrieves the confidence scores corresponding to the top 5 class probabilities
        predicted by the model. It provides a quick way to access the most likely class predictions
        along with their associated confidence levels.

        Returns:
            (torch.Tensor | numpy.ndarray): A tensor or array containing the confidence scores for the
                top 5 predicted classes, sorted in descending order of probability.

        Examples:
            >>> results = model('image.jpg')
            >>> probs = results[0].probs
            >>> top5_conf = probs.top5conf
            >>> print(top5_conf)  # Prints confidence scores for top 5 classes
        """
        # 返回数据张量中前五个最大概率类别的置信度分数
        return self.data[self.top5]
# 定义一个名为 OBB 的类，继承自 BaseTensor
class OBB(BaseTensor):
    """
    A class for storing and manipulating Oriented Bounding Boxes (OBB).

    This class provides functionality to handle oriented bounding boxes, including conversion between
    different formats, normalization, and access to various properties of the boxes.

    Attributes:
        data (torch.Tensor): The raw OBB tensor containing box coordinates and associated data.
        orig_shape (tuple): Original image size as (height, width).
        is_track (bool): Indicates whether tracking IDs are included in the box data.
        xywhr (torch.Tensor | numpy.ndarray): Boxes in [x_center, y_center, width, height, rotation] format.
        conf (torch.Tensor | numpy.ndarray): Confidence scores for each box.
        cls (torch.Tensor | numpy.ndarray): Class labels for each box.
        id (torch.Tensor | numpy.ndarray): Tracking IDs for each box, if available.
        xyxyxyxy (torch.Tensor | numpy.ndarray): Boxes in 8-point [x1, y1, x2, y2, x3, y3, x4, y4] format.
        xyxyxyxyn (torch.Tensor | numpy.ndarray): Normalized 8-point coordinates relative to orig_shape.
        xyxy (torch.Tensor | numpy.ndarray): Axis-aligned bounding boxes in [x1, y1, x2, y2] format.

    Methods:
        cpu(): Returns a copy of the OBB object with all tensors on CPU memory.
        numpy(): Returns a copy of the OBB object with all tensors as numpy arrays.
        cuda(): Returns a copy of the OBB object with all tensors on GPU memory.
        to(*args, **kwargs): Returns a copy of the OBB object with tensors on specified device and dtype.

    Examples:
        >>> boxes = torch.tensor([[100, 50, 150, 100, 30, 0.9, 0]])  # xywhr, conf, cls
        >>> obb = OBB(boxes, orig_shape=(480, 640))
        >>> print(obb.xyxyxyxy)
        >>> print(obb.conf)
        >>> print(obb.cls)
    """
    def __init__(self, boxes, orig_shape) -> None:
        """
        Initialize an OBB (Oriented Bounding Box) instance with oriented bounding box data and original image shape.

        This class stores and manipulates Oriented Bounding Boxes (OBB) for object detection tasks. It provides
        various properties and methods to access and transform the OBB data.

        Args:
            boxes (torch.Tensor | numpy.ndarray): A tensor or numpy array containing the detection boxes,
                with shape (num_boxes, 7) or (num_boxes, 8). The last two columns contain confidence and class values.
                If present, the third last column contains track IDs, and the fifth column contains rotation.
            orig_shape (Tuple[int, int]): Original image size, in the format (height, width).

        Attributes:
            data (torch.Tensor | numpy.ndarray): The raw OBB tensor.
            orig_shape (Tuple[int, int]): The original image shape.
            is_track (bool): Whether the boxes include tracking IDs.

        Raises:
            AssertionError: If the number of values per box is not 7 or 8.

        Examples:
            >>> import torch
            >>> boxes = torch.rand(3, 7)  # 3 boxes with 7 values each
            >>> orig_shape = (640, 480)
            >>> obb = OBB(boxes, orig_shape)
            >>> print(obb.xywhr)  # Access the boxes in xywhr format
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]  # 将一维的boxes转换为二维，以确保正确的形状
        n = boxes.shape[-1]  # 获取最后一个维度的大小，即每个box的值个数
        assert n in {7, 8}, f"expected 7 or 8 values but got {n}"  # 断言检查每个box的值个数是否为7或8
        super().__init__(boxes, orig_shape)  # 调用父类的初始化方法，传入boxes和orig_shape
        self.is_track = n == 8  # 判断是否包含track IDs，如果值个数为8，则包含
        self.orig_shape = orig_shape  # 存储原始图像的形状信息

    @property
    def xywhr(self):
        """
        Returns boxes in [x_center, y_center, width, height, rotation] format.

        Returns:
            (torch.Tensor | numpy.ndarray): A tensor or numpy array containing the oriented bounding boxes with format
                [x_center, y_center, width, height, rotation]. The shape is (N, 5) where N is the number of boxes.

        Examples:
            >>> results = model('image.jpg')
            >>> obb = results[0].obb
            >>> xywhr = obb.xywhr
            >>> print(xywhr.shape)
            torch.Size([3, 5])
        """
        return self.data[:, :5]  # 返回包含 [x_center, y_center, width, height, rotation] 的部分数据
    def conf(self):
        """
        Returns the confidence scores for Oriented Bounding Boxes (OBBs).

        This property retrieves the confidence values associated with each OBB detection. The confidence score
        represents the model's certainty in the detection.

        Returns:
            (torch.Tensor | numpy.ndarray): A tensor or numpy array of shape (N,) containing confidence scores
                for N detections, where each score is in the range [0, 1].

        Examples:
            >>> results = model('image.jpg')
            >>> obb_result = results[0].obb
            >>> confidence_scores = obb_result.conf
            >>> print(confidence_scores)
        """
        # 返回包含所有检测结果置信度的数据列
        return self.data[:, -2]

    @property
    def cls(self):
        """
        Returns the class values of the oriented bounding boxes.

        Returns:
            (torch.Tensor | numpy.ndarray): A tensor or numpy array containing the class values for each oriented
                bounding box. The shape is (N,), where N is the number of boxes.

        Examples:
            >>> results = model('image.jpg')
            >>> result = results[0]
            >>> obb = result.obb
            >>> class_values = obb.cls
            >>> print(class_values)
        """
        # 返回包含所有检测结果类别的数据列
        return self.data[:, -1]

    @property
    def id(self):
        """
        Returns the tracking IDs of the oriented bounding boxes (if available).

        Returns:
            (torch.Tensor | numpy.ndarray | None): A tensor or numpy array containing the tracking IDs for each
                oriented bounding box. Returns None if tracking IDs are not available.

        Examples:
            >>> results = model('image.jpg', tracker=True)  # Run inference with tracking
            >>> for result in results:
            ...     if result.obb is not None:
            ...         track_ids = result.obb.id
            ...         if track_ids is not None:
            ...             print(f"Tracking IDs: {track_ids}")
        """
        # 如果追踪信息可用，则返回包含所有检测结果追踪 ID 的数据列；否则返回 None
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)
    def xyxyxyxy(self):
        """
        Converts OBB format to 8-point (xyxyxyxy) coordinate format for rotated bounding boxes.

        Returns:
            (torch.Tensor | numpy.ndarray): Rotated bounding boxes in xyxyxyxy format with shape (N, 4, 2), where N is
                the number of boxes. Each box is represented by 4 points (x, y), starting from the top-left corner and
                moving clockwise.

        Examples:
            >>> obb = OBB(torch.tensor([[100, 100, 50, 30, 0.5, 0.9, 0]]), orig_shape=(640, 640))
            >>> xyxyxyxy = obb.xyxyxyxy
            >>> print(xyxyxyxy.shape)
            torch.Size([1, 4, 2])
        """
        # 将 OBB 格式转换为 xyxyxyxy 格式的旋转边界框坐标
        return ops.xywhr2xyxyxyxy(self.xywhr)

    @property
    @lru_cache(maxsize=2)
    def xyxyxyxyn(self):
        """
        Converts rotated bounding boxes to normalized xyxyxyxy format.

        Returns:
            (torch.Tensor | numpy.ndarray): Normalized rotated bounding boxes in xyxyxyxy format with shape (N, 4, 2),
                where N is the number of boxes. Each box is represented by 4 points (x, y), normalized relative to
                the original image dimensions.

        Examples:
            >>> obb = OBB(torch.rand(10, 7), orig_shape=(640, 480))  # 10 random OBBs
            >>> normalized_boxes = obb.xyxyxyxyn
            >>> print(normalized_boxes.shape)
            torch.Size([10, 4, 2])
        """
        # 创建副本以确保不修改原始数据
        xyxyxyxyn = self.xyxyxyxy.clone() if isinstance(self.xyxyxyxy, torch.Tensor) else np.copy(self.xyxyxyxy)
        # 归一化 x 坐标
        xyxyxyxyn[..., 0] /= self.orig_shape[1]
        # 归一化 y 坐标
        xyxyxyxyn[..., 1] /= self.orig_shape[0]
        # 返回归一化后的坐标
        return xyxyxyxyn

    @property
    @lru_cache(maxsize=2)
    def xyxy(self):
        """
        Converts oriented bounding boxes (OBB) to axis-aligned bounding boxes in xyxy format.

        This property calculates the minimal enclosing rectangle for each oriented bounding box and returns it in
        xyxy format (x1, y1, x2, y2). This is useful for operations that require axis-aligned bounding boxes, such
        as IoU calculation with non-rotated boxes.

        Returns:
            (torch.Tensor | numpy.ndarray): Axis-aligned bounding boxes in xyxy format with shape (N, 4), where N
                is the number of boxes. Each row contains [x1, y1, x2, y2] coordinates.

        Examples:
            >>> import torch
            >>> from ultralytics import YOLO
            >>> model = YOLO('yolov8n-obb.pt')
            >>> results = model('path/to/image.jpg')
            >>> for result in results:
            ...     obb = result.obb
            ...     if obb is not None:
            ...         xyxy_boxes = obb.xyxy
            ...         print(xyxy_boxes.shape)  # (N, 4)

        Notes:
            - This method approximates the OBB by its minimal enclosing rectangle.
            - The returned format is compatible with standard object detection metrics and visualization tools.
            - The property uses caching to improve performance for repeated access.
        """
        # 提取 x 和 y 坐标
        x = self.xyxyxyxy[..., 0]
        y = self.xyxyxyxy[..., 1]
        # 根据最小值和最大值创建包围框
        return (
            torch.stack([x.amin(1), y.amin(1), x.amax(1), y.amax(1)], -1)
            if isinstance(x, torch.Tensor)
            else np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], -1)
        )
```