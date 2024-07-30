# `.\yolov8\ultralytics\data\loaders.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

import glob  # 导入glob模块，用于获取文件路径列表
import math  # 导入math模块，提供数学计算函数
import os  # 导入os模块，用于与操作系统进行交互
import time  # 导入time模块，提供时间相关函数
from dataclasses import dataclass  # 导入dataclass类，用于创建数据类
from pathlib import Path  # 导入Path类，用于处理路径
from threading import Thread  # 导入Thread类，用于实现多线程操作
from urllib.parse import urlparse  # 导入urlparse函数，用于解析URL

import cv2  # 导入cv2模块，OpenCV库
import numpy as np  # 导入numpy库，用于数值计算
import requests  # 导入requests模块，用于HTTP请求
import torch  # 导入torch模块，PyTorch深度学习库
from PIL import Image  # 导入Image类，Python图像处理库PIL的一部分

from ultralytics.data.utils import FORMATS_HELP_MSG, IMG_FORMATS, VID_FORMATS  # 导入自定义模块的特定内容
from ultralytics.utils import IS_COLAB, IS_KAGGLE, LOGGER, ops  # 导入自定义模块的特定内容
from ultralytics.utils.checks import check_requirements  # 导入自定义模块的特定函数


@dataclass
class SourceTypes:
    """Class to represent various types of input sources for predictions."""
    
    stream: bool = False  # 是否为流类型输入，默认为False
    screenshot: bool = False  # 是否为截图类型输入，默认为False
    from_img: bool = False  # 是否为图像文件类型输入，默认为False
    tensor: bool = False  # 是否为张量类型输入，默认为False


class LoadStreams:
    """
    Stream Loader for various types of video streams, Supports RTSP, RTMP, HTTP, and TCP streams.

    Attributes:
        sources (str): The source input paths or URLs for the video streams.
        vid_stride (int): Video frame-rate stride, defaults to 1.
        buffer (bool): Whether to buffer input streams, defaults to False.
        running (bool): Flag to indicate if the streaming thread is running.
        mode (str): Set to 'stream' indicating real-time capture.
        imgs (list): List of image frames for each stream.
        fps (list): List of FPS for each stream.
        frames (list): List of total frames for each stream.
        threads (list): List of threads for each stream.
        shape (list): List of shapes for each stream.
        caps (list): List of cv2.VideoCapture objects for each stream.
        bs (int): Batch size for processing.

    Methods:
        __init__: Initialize the stream loader.
        update: Read stream frames in daemon thread.
        close: Close stream loader and release resources.
        __iter__: Returns an iterator object for the class.
        __next__: Returns source paths, transformed, and original images for processing.
        __len__: Return the length of the sources object.

    Example:
         ```py
         yolo predict source='rtsp://example.com/media.mp4'
         ```
    """
    def __init__(self, sources="file.streams", vid_stride=1, buffer=False):
        """Initialize instance variables and check for consistent input stream shapes."""
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.buffer = buffer  # buffer input streams
        self.running = True  # running flag for Thread
        self.mode = "stream"
        self.vid_stride = vid_stride  # video frame-rate stride

        # Read sources from file or use directly if already a list
        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)  # Number of sources
        self.bs = n  # Set batch size to number of sources
        self.fps = [0] * n  # Initialize frames per second list for each source
        self.frames = [0] * n  # Initialize frame count list for each source
        self.threads = [None] * n  # Initialize threads list for each source
        self.caps = [None] * n  # Initialize video capture objects list for each source
        self.imgs = [[] for _ in range(n)]  # Initialize empty list to store images for each source
        self.shape = [[] for _ in range(n)]  # Initialize empty list to store image shapes for each source
        self.sources = [ops.clean_str(x) for x in sources]  # Clean and store source names for later use

        for i, s in enumerate(sources):  # Loop through each source with index i and source s
            # Start thread to read frames from video stream
            st = f"{i + 1}/{n}: {s}... "

            # Check if source is a YouTube video and convert URL if necessary
            if urlparse(s).hostname in {"www.youtube.com", "youtube.com", "youtu.be"}:
                s = get_best_youtube_url(s)

            # Evaluate string if numeric (e.g., '0' for local webcam)
            s = eval(s) if s.isnumeric() else s

            # Raise error if trying to use webcam in Colab or Kaggle environments
            if s == 0 and (IS_COLAB or IS_KAGGLE):
                raise NotImplementedError(
                    "'source=0' webcam not supported in Colab and Kaggle notebooks. "
                    "Try running 'source=0' in a local environment."
                )

            # Initialize video capture object for the current source
            self.caps[i] = cv2.VideoCapture(s)

            # Raise error if video capture object fails to open
            if not self.caps[i].isOpened():
                raise ConnectionError(f"{st}Failed to open {s}")

            # Retrieve and store video properties: width, height, frames per second
            w = int(self.caps[i].get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.caps[i].get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.caps[i].get(cv2.CAP_PROP_FPS)
            
            # Calculate total frames; handle cases where frame count might be 0 or NaN
            self.frames[i] = max(int(self.caps[i].get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float("inf")

            # Calculate frames per second, ensuring a minimum of 30 FPS
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30

            # Read the first frame to ensure successful connection
            success, im = self.caps[i].read()
            if not success or im is None:
                raise ConnectionError(f"{st}Failed to read images from {s}")

            # Store the first frame and its shape
            self.imgs[i].append(im)
            self.shape[i] = im.shape

            # Start a thread to continuously update frames for the current source
            self.threads[i] = Thread(target=self.update, args=([i, self.caps[i], s]), daemon=True)
            LOGGER.info(f"{st}Success ✅ ({self.frames[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)")

            # Start the thread for reading frames
            self.threads[i].start()

        LOGGER.info("")  # Print a newline for logging clarity
    def update(self, i, cap, stream):
        """
        Read stream `i` frames in daemon thread.
        """
        n, f = 0, self.frames[i]  # 初始化帧号和帧数组
        while self.running and cap.isOpened() and n < (f - 1):
            if len(self.imgs[i]) < 30:  # 保持不超过30帧的图像缓冲
                n += 1
                cap.grab()  # 捕获视频帧，不直接读取，而是先抓取再检索
                if n % self.vid_stride == 0:  # 每 vid_stride 帧执行一次
                    success, im = cap.retrieve()  # 检索已抓取的视频帧
                    if not success:
                        im = np.zeros(self.shape[i], dtype=np.uint8)  # 如果检索失败，创建全零图像
                        LOGGER.warning("WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.")
                        cap.open(stream)  # 如果信号丢失，重新打开流
                    if self.buffer:
                        self.imgs[i].append(im)  # 将图像帧添加到缓冲区
                    else:
                        self.imgs[i] = [im]  # 替换当前缓冲区的图像帧
            else:
                time.sleep(0.01)  # 等待直到缓冲区为空

    def close(self):
        """
        Close stream loader and release resources.
        """
        self.running = False  # 停止线程的标志
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)  # 等待线程结束，设置超时时间
        for cap in self.caps:  # 遍历存储的 VideoCapture 对象
            try:
                cap.release()  # 释放视频捕获对象
            except Exception as e:
                LOGGER.warning(f"WARNING ⚠️ Could not release VideoCapture object: {e}")  # 捕获异常并记录警告信息
        cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口

    def __iter__(self):
        """
        Iterates through YOLO image feed and re-opens unresponsive streams.
        """
        self.count = -1  # 初始化计数器
        return self

    def __next__(self):
        """
        Returns source paths, transformed and original images for processing.
        """
        self.count += 1  # 计数器自增

        images = []
        for i, x in enumerate(self.imgs):
            # 等待直到每个缓冲区中有帧可用
            while not x:
                if not self.threads[i].is_alive() or cv2.waitKey(1) == ord("q"):  # 检查线程状态或用户是否按下 'q'
                    self.close()  # 关闭对象
                    raise StopIteration  # 抛出停止迭代异常
                time.sleep(1 / min(self.fps))  # 等待时间间隔，最小 FPS
                x = self.imgs[i]  # 更新缓冲区状态
                if not x:
                    LOGGER.warning(f"WARNING ⚠️ Waiting for stream {i}")  # 记录警告信息

            # 从 imgs 缓冲区中获取并移除第一帧图像
            if self.buffer:
                images.append(x.pop(0))
            # 获取最后一帧图像，并清空缓冲区的其余图像帧
            else:
                images.append(x.pop(-1) if x else np.zeros(self.shape[i], dtype=np.uint8))
                x.clear()

        return self.sources, images, [""] * self.bs  # 返回源路径、转换后的图像和原始图像列表

    def __len__(self):
        """
        Return the length of the sources object.
        """
        return self.bs  # 返回源对象的长度，即 batch size
class LoadScreenshots:
    """
    YOLOv8 screenshot dataloader.

    This class manages the loading of screenshot images for processing with YOLOv8.
    Suitable for use with `yolo predict source=screen`.

    Attributes:
        source (str): The source input indicating which screen to capture.
        screen (int): The screen number to capture.
        left (int): The left coordinate for screen capture area.
        top (int): The top coordinate for screen capture area.
        width (int): The width of the screen capture area.
        height (int): The height of the screen capture area.
        mode (str): Set to 'stream' indicating real-time capture.
        frame (int): Counter for captured frames.
        sct (mss.mss): Screen capture object from `mss` library.
        bs (int): Batch size, set to 1.
        monitor (dict): Monitor configuration details.

    Methods:
        __iter__: Returns an iterator object.
        __next__: Captures the next screenshot and returns it.
    """

    def __init__(self, source):
        """Source = [screen_number left top width height] (pixels)."""
        # 检查并确保mss库已经安装
        check_requirements("mss")
        # 导入mss库
        import mss  # noqa

        # 解析source参数，根据参数设置截图的屏幕区域
        source, *params = source.split()
        self.screen, left, top, width, height = 0, None, None, None, None  # default to full screen 0
        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)
        
        # 设置截图模式为实时流
        self.mode = "stream"
        # 初始化帧计数器
        self.frame = 0
        # 创建mss对象用于屏幕截图
        self.sct = mss.mss()
        # 设置批处理大小为1
        self.bs = 1
        # 设置帧率为30帧每秒
        self.fps = 30

        # 解析monitor参数，根据屏幕和截图区域设置监视器配置
        monitor = self.sct.monitors[self.screen]
        self.top = monitor["top"] if top is None else (monitor["top"] + top)
        self.left = monitor["left"] if left is None else (monitor["left"] + left)
        self.width = width or monitor["width"]
        self.height = height or monitor["height"]
        self.monitor = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    def __iter__(self):
        """Returns an iterator of the object."""
        return self

    def __next__(self):
        """mss screen capture: get raw pixels from the screen as np array."""
        # 使用mss对象获取屏幕截图，并将像素转换为numpy数组
        im0 = np.asarray(self.sct.grab(self.monitor))[:, :, :3]  # BGRA to BGR
        s = f"screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: "

        # 增加帧计数
        self.frame += 1
        # 返回截图相关信息
        return [str(self.screen)], [im0], [s]  # screen, img, string
    # 定义一个数据加载器类，用于加载图像和视频文件
    class Dataloader:
        """
        Attributes:
            files (list): List of image and video file paths.
            nf (int): Total number of files (images and videos).
            video_flag (list): Flags indicating whether a file is a video (True) or an image (False).
            mode (str): Current mode, 'image' or 'video'.
            vid_stride (int): Stride for video frame-rate, defaults to 1.
            bs (int): Batch size, set to 1 for this class.
            cap (cv2.VideoCapture): Video capture object for OpenCV.
            frame (int): Frame counter for video.
            frames (int): Total number of frames in the video.
            count (int): Counter for iteration, initialized at 0 during `__iter__()`.

        Methods:
            _new_video(path): Create a new cv2.VideoCapture object for a given video path.
        """

        def __init__(self, path, batch=1, vid_stride=1):
            """Initialize the Dataloader and raise FileNotFoundError if file not found."""
            parent = None
            if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
                parent = Path(path).parent
                path = Path(path).read_text().splitlines()  # list of sources
            files = []
            for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
                a = str(Path(p).absolute())  # do not use .resolve() https://github.com/ultralytics/ultralytics/issues/2912
                if "*" in a:
                    files.extend(sorted(glob.glob(a, recursive=True)))  # glob
                elif os.path.isdir(a):
                    files.extend(sorted(glob.glob(os.path.join(a, "*.*"))))  # dir
                elif os.path.isfile(a):
                    files.append(a)  # files (absolute or relative to CWD)
                elif parent and (parent / p).is_file():
                    files.append(str((parent / p).absolute()))  # files (relative to *.txt file parent)
                else:
                    raise FileNotFoundError(f"{p} does not exist")

            # Define files as images or videos
            images, videos = [], []
            for f in files:
                suffix = f.split(".")[-1].lower()  # Get file extension without the dot and lowercase
                if suffix in IMG_FORMATS:
                    images.append(f)
                elif suffix in VID_FORMATS:
                    videos.append(f)
            ni, nv = len(images), len(videos)

            self.files = images + videos
            self.nf = ni + nv  # number of files
            self.ni = ni  # number of images
            self.video_flag = [False] * ni + [True] * nv
            self.mode = "image"
            self.vid_stride = vid_stride  # video frame-rate stride
            self.bs = batch
            if any(videos):
                self._new_video(videos[0])  # new video
            else:
                self.cap = None
            if self.nf == 0:
                raise FileNotFoundError(f"No images or videos found in {p}. {FORMATS_HELP_MSG}")

        def __iter__(self):
            """Returns an iterator object for VideoStream or ImageFolder."""
            self.count = 0
            return self
    def __next__(self):
        """Returns the next batch of images or video frames along with their paths and metadata."""
        paths, imgs, info = [], [], []  # 初始化空列表，用于存储路径、图像/视频帧和元数据信息
        while len(imgs) < self.bs:  # 当图像/视频帧列表长度小于批次大小时执行循环
            if self.count >= self.nf:  # 如果计数器超过文件总数，则表示文件列表结束
                if imgs:
                    return paths, imgs, info  # 返回最后一个不完整的批次
                else:
                    raise StopIteration  # 否则抛出迭代结束异常

            path = self.files[self.count]  # 获取当前文件路径
            if self.video_flag[self.count]:  # 检查当前文件是否为视频
                self.mode = "video"  # 设置模式为视频
                if not self.cap or not self.cap.isOpened():  # 如果视频捕获对象不存在或未打开
                    self._new_video(path)  # 创建新的视频捕获对象

                for _ in range(self.vid_stride):  # 循环抓取视频帧
                    success = self.cap.grab()
                    if not success:
                        break  # 如果抓取失败，则退出循环

                if success:  # 如果抓取成功
                    success, im0 = self.cap.retrieve()  # 检索抓取的视频帧
                    if success:
                        self.frame += 1  # 帧数加一
                        paths.append(path)  # 添加路径到列表
                        imgs.append(im0)  # 添加图像帧到列表
                        info.append(f"video {self.count + 1}/{self.nf} (frame {self.frame}/{self.frames}) {path}: ")  # 添加视频信息到列表
                        if self.frame == self.frames:  # 如果达到视频帧数的最大值
                            self.count += 1  # 计数器加一
                            self.cap.release()  # 释放视频捕获对象
                else:
                    # 如果当前视频结束或打开失败，移动到下一个文件
                    self.count += 1
                    if self.cap:
                        self.cap.release()  # 释放视频捕获对象
                    if self.count < self.nf:
                        self._new_video(self.files[self.count])  # 创建新的视频捕获对象
            else:
                self.mode = "image"  # 设置模式为图像
                im0 = cv2.imread(path)  # 读取图像（BGR格式）
                if im0 is None:
                    LOGGER.warning(f"WARNING ⚠️ Image Read Error {path}")  # 如果图像读取失败，记录警告信息
                else:
                    paths.append(path)  # 添加路径到列表
                    imgs.append(im0)  # 添加图像到列表
                    info.append(f"image {self.count + 1}/{self.nf} {path}: ")  # 添加图像信息到列表
                self.count += 1  # 计数器加一，移动到下一个文件
                if self.count >= self.ni:  # 如果计数器超过图像总数
                    break  # 跳出循环，结束图像列表的读取

        return paths, imgs, info  # 返回路径、图像/视频帧和元数据信息列表

    def _new_video(self, path):
        """Creates a new video capture object for the given path."""
        self.frame = 0  # 初始化帧数
        self.cap = cv2.VideoCapture(path)  # 创建新的视频捕获对象
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))  # 获取视频帧率
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to open video {path}")  # 如果视频打开失败，抛出文件未找到异常
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)  # 计算视频帧数

    def __len__(self):
        """Returns the number of batches in the object."""
        return math.ceil(self.nf / self.bs)  # 返回对象中批次的数量，向上取整
    """
    Load images from PIL and Numpy arrays for batch processing.

    This class is designed to manage loading and pre-processing of image data from both PIL and Numpy formats.
    It performs basic validation and format conversion to ensure that the images are in the required format for
    downstream processing.

    Attributes:
        paths (list): List of image paths or autogenerated filenames.
        im0 (list): List of images stored as Numpy arrays.
        mode (str): Type of data being processed, defaults to 'image'.
        bs (int): Batch size, equivalent to the length of `im0`.

    Methods:
        _single_check(im): Validate and format a single image to a Numpy array.
    """

    def __init__(self, im0):
        """Initialize PIL and Numpy Dataloader."""
        if not isinstance(im0, list):
            im0 = [im0]
        # Generate filenames or use existing ones from input images
        self.paths = [getattr(im, "filename", f"image{i}.jpg") for i, im in enumerate(im0)]
        # Validate and convert each image in `im0` to Numpy arrays
        self.im0 = [self._single_check(im) for im in im0]
        # Set the processing mode to 'image'
        self.mode = "image"
        # Set the batch size to the number of images in `im0`
        self.bs = len(self.im0)

    @staticmethod
    def _single_check(im):
        """Validate and format an image to numpy array."""
        # Ensure `im` is either a PIL.Image or np.ndarray
        assert isinstance(im, (Image.Image, np.ndarray)), f"Expected PIL/np.ndarray image type, but got {type(im)}"
        if isinstance(im, Image.Image):
            # Convert PIL.Image to RGB mode if not already
            if im.mode != "RGB":
                im = im.convert("RGB")
            # Convert PIL.Image to Numpy array and reverse channels
            im = np.asarray(im)[:, :, ::-1]
            im = np.ascontiguousarray(im)  # Make sure the array is contiguous
        return im

    def __len__(self):
        """Returns the length of the 'im0' attribute."""
        return len(self.im0)

    def __next__(self):
        """Returns batch paths, images, processed images, None, ''."""
        if self.count == 1:  # loop only once as it's batch inference
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, [""] * self.bs

    def __iter__(self):
        """Enables iteration for class LoadPilAndNumpy."""
        self.count = 0
        return self


class LoadTensor:
    """
    Load images from torch.Tensor data.

    This class manages the loading and pre-processing of image data from PyTorch tensors for further processing.

    Attributes:
        im0 (torch.Tensor): The input tensor containing the image(s).
        bs (int): Batch size, inferred from the shape of `im0`.
        mode (str): Current mode, set to 'image'.
        paths (list): List of image paths or filenames.
        count (int): Counter for iteration, initialized at 0 during `__iter__()`.

    Methods:
        _single_check(im, stride): Validate and possibly modify the input tensor.
    """

    def __init__(self, im0) -> None:
        """Initialize Tensor Dataloader."""
        # Validate and store the input tensor `im0`
        self.im0 = self._single_check(im0)
        # Infer batch size from the first dimension of the tensor
        self.bs = self.im0.shape[0]
        # Set the processing mode to 'image'
        self.mode = "image"
        # Generate filenames or use existing ones from input tensors
        self.paths = [getattr(im, "filename", f"image{i}.jpg") for i, im in enumerate(im0)]

    @staticmethod
    # 验证并将图像格式化为 torch.Tensor
    def _single_check(im, stride=32):
        """Validate and format an image to torch.Tensor."""
        # 构建警告信息，确保输入的 torch.Tensor 应为 BCHW 格式，即 shape(1, 3, 640, 640)，且能被指定的步长 stride 整除。如果不兼容则抛出错误。
        s = (
            f"WARNING ⚠️ torch.Tensor inputs should be BCHW i.e. shape(1, 3, 640, 640) "
            f"divisible by stride {stride}. Input shape{tuple(im.shape)} is incompatible."
        )
        # 检查输入图像的维度是否为4维，如果不是，则尝试在第0维度上增加一个维度。
        if len(im.shape) != 4:
            if len(im.shape) != 3:
                raise ValueError(s)
            # 记录警告日志，表示输入图像维度不符合要求
            LOGGER.warning(s)
            im = im.unsqueeze(0)
        # 检查图像的高度和宽度是否能被指定的步长整除，如果不能则抛出错误。
        if im.shape[2] % stride or im.shape[3] % stride:
            raise ValueError(s)
        # 如果图像中的最大值超过了1.0加上 torch.float32 类型的误差允许值，记录警告日志，并将输入图像转换为 float 类型后归一化到0.0-1.0范围内。
        if im.max() > 1.0 + torch.finfo(im.dtype).eps:  # torch.float32 eps is 1.2e-07
            LOGGER.warning(
                f"WARNING ⚠️ torch.Tensor inputs should be normalized 0.0-1.0 but max value is {im.max()}. "
                f"Dividing input by 255."
            )
            im = im.float() / 255.0

        return im

    # 返回一个迭代器对象
    def __iter__(self):
        """Returns an iterator object."""
        self.count = 0
        return self

    # 返回迭代器的下一个项目
    def __next__(self):
        """Return next item in the iterator."""
        # 如果计数器达到1，抛出 StopIteration 异常
        if self.count == 1:
            raise StopIteration
        # 增加计数器的值，并返回路径、im0 和空列表组成的元组
        self.count += 1
        return self.paths, self.im0, [""] * self.bs

    # 返回批处理大小
    def __len__(self):
        """Returns the batch size."""
        return self.bs
def autocast_list(source):
    """
    Merges a list of source of different types into a list of numpy arrays or PIL images.

    Args:
        source (list): A list containing elements of various types like filenames, URIs, PIL Images, or numpy arrays.

    Returns:
        list: A list containing PIL Images or numpy arrays converted from the input sources.

    Raises:
        TypeError: If the input element is not of a supported type.

    """
    files = []
    for im in source:
        if isinstance(im, (str, Path)):  # filename or uri
            # Open the image from URL if it starts with "http", otherwise directly open as file
            files.append(Image.open(requests.get(im, stream=True).raw if str(im).startswith("http") else im))
        elif isinstance(im, (Image.Image, np.ndarray)):  # PIL or np Image
            files.append(im)
        else:
            raise TypeError(
                f"type {type(im).__name__} is not a supported Ultralytics prediction source type. \n"
                f"See https://docs.ultralytics.com/modes/predict for supported source types."
            )

    return files


def get_best_youtube_url(url, method="pytube"):
    """
    Retrieves the URL of the best quality MP4 video stream from a given YouTube video.

    Args:
        url (str): The URL of the YouTube video.
        method (str): The method to use for extracting video info. Default is "pytube". Other options are "pafy" and
            "yt-dlp".

    Returns:
        str: The URL of the best quality MP4 video stream, or None if no suitable stream is found.

    """
    if method == "pytube":
        # Ensure compatibility with pytubefix library version
        check_requirements("pytubefix>=6.5.2")
        from pytubefix import YouTube

        # Fetch video streams filtered by MP4 format and only video streams
        streams = YouTube(url).streams.filter(file_extension="mp4", only_video=True)
        # Sort streams by resolution in descending order
        streams = sorted(streams, key=lambda s: s.resolution, reverse=True)
        for stream in streams:
            # Check if stream resolution is at least 1080p
            if stream.resolution and int(stream.resolution[:-1]) >= 1080:
                return stream.url

    elif method == "pafy":
        # Ensure necessary libraries are installed and import pafy
        check_requirements(("pafy", "youtube_dl==2020.12.2"))
        import pafy  # noqa

        # Fetch the best available MP4 video stream URL
        return pafy.new(url).getbestvideo(preftype="mp4").url
    # 如果下载方法为 "yt-dlp"，则执行以下代码块
    elif method == "yt-dlp":
        # 检查是否满足使用 yt-dlp 的要求
        check_requirements("yt-dlp")
        # 导入 yt_dlp 模块
        import yt_dlp

        # 使用 yt-dlp.YoutubeDL 创建一个实例 ydl，并设置参数 {"quiet": True}
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            # 调用 extract_info 方法从指定的 url 提取视频信息，但不下载视频
            info_dict = ydl.extract_info(url, download=False)

        # 遍历视频格式信息列表（反向遍历，因为最佳格式通常在最后）
        for f in reversed(info_dict.get("formats", [])):
            # 检查当前格式是否满足条件：视频编解码器存在、无音频、扩展名为 mp4、至少 1920x1080 大小
            good_size = (f.get("width") or 0) >= 1920 or (f.get("height") or 0) >= 1080
            if good_size and f["vcodec"] != "none" and f["acodec"] == "none" and f["ext"] == "mp4":
                # 如果符合条件，返回该格式的视频 URL
                return f.get("url")
# 定义常量 LOADERS，包含四个不同的加载器类
LOADERS = (LoadStreams, LoadPilAndNumpy, LoadImagesAndVideos, LoadScreenshots)
```