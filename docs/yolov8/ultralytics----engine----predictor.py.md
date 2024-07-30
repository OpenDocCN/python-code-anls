# `.\yolov8\ultralytics\engine\predictor.py`

```py
# 导入必要的库和模块
import platform  # 导入平台信息模块
import re  # 导入正则表达式模块
import threading  # 导入线程模块
from pathlib import Path  # 导入路径操作模块

import cv2  # 导入OpenCV库
import numpy as np  # 导入NumPy库
import torch  # 导入PyTorch库

# 从Ultralytics库中导入各种函数和类
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox, classify_transforms
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops
from ultralytics.utils.checks import check_imgsz, check_imshow
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import select_device, smart_inference_mode

# 定义用于流警告的多行字符串
STREAM_WARNING = """
WARNING ⚠️ inference results will accumulate in RAM unless `stream=True` is passed, causing potential out-of-memory
errors for large sources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict/ for help.

Example:
    results = model(source=..., stream=True)  # generator of Results objects
"""
    for r in results:
        # 从结果列表中依次取出每个结果对象 r

        boxes = r.boxes  # 获取结果对象 r 中的 boxes 属性，用于包围框的输出
        masks = r.masks  # 获取结果对象 r 中的 masks 属性，用于分割掩模的输出
        probs = r.probs  # 获取结果对象 r 中的 probs 属性，用于分类输出的类别概率
    """
    BasePredictor.

    A base class for creating predictors.

    Attributes:
        args (SimpleNamespace): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        done_warmup (bool): Whether the predictor has finished setup.
        model (nn.Module): Model used for prediction.
        data (dict): Data configuration.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
        vid_writer (dict): Dictionary of {save_path: video_writer, ...} writer for saving video output.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        # 获取配置并初始化预测器参数
        self.args = get_cfg(cfg, overrides)
        # 获取保存结果的目录路径
        self.save_dir = get_save_dir(self.args)
        # 如果配置中未指定 conf 参数，则设为默认值 0.25
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        # 标记预热过程未完成
        self.done_warmup = False
        # 如果设置了 args.show 为 True，则检查是否支持显示
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # 可用于完成设置后使用的变量初始化
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_writer = {}  # dict of {save_path: video_writer, ...}
        self.plotted_img = None
        self.source_type = None
        self.seen = 0
        self.windows = []
        self.batch = None
        self.results = None
        self.transforms = None
        # 获取默认回调函数，如果未提供则使用默认值
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.txt_path = None
        # 初始化用于自动线程安全推理的锁
        self._lock = threading.Lock()
        callbacks.add_integration_callbacks(self)

    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        # 如果输入不是 Tensor，则进行预处理转换
        if not_tensor:
            # 对图像进行预处理转换
            im = np.stack(self.pre_transform(im))
            # 将图像由 BGR 转换为 RGB，将格式由 BHWC 转换为 BCHW，(n, 3, h, w)
            im = im[..., ::-1].transpose((0, 3, 1, 2))
            # 转换为连续的内存布局
            im = np.ascontiguousarray(im)
            # 转换为 PyTorch 的 Tensor 格式
            im = torch.from_numpy(im)

        # 将图像移动到指定的计算设备上
        im = im.to(self.device)
        # 如果模型使用 fp16，则将输入转换为半精度浮点数
        im = im.half() if self.model.fp16 else im.float()
        # 如果输入不是 Tensor，则将像素值范围从 0-255 转换为 0.0-1.0
        if not_tensor:
            im /= 255
        return im
    def inference(self, im, *args, **kwargs):
        """Runs inference on a given image using the specified model and arguments."""
        # 根据参数确定是否需要可视化输出，并根据条件创建保存目录路径
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
            if self.args.visualize and (not self.source_type.tensor)
            else False
        )
        # 调用模型进行推理，传递参数 augment, visualize, embed 以及其他可变参数
        return self.model(im, augment=self.args.augment, visualize=visualize, embed=self.args.embed, *args, **kwargs)

    def pre_transform(self, im):
        """
        Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Returns:
            (list): A list of transformed images.
        """
        # 检查输入图像是否具有相同的形状
        same_shapes = len({x.shape for x in im}) == 1
        # 创建 LetterBox 对象进行图像预处理，保证图像尺寸与模型期望输入一致
        letterbox = LetterBox(self.imgsz, auto=same_shapes and self.model.pt, stride=self.model.stride)
        # 返回经过预处理的图像列表
        return [letterbox(image=x) for x in im]

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions for an image and returns them."""
        # 目前只是简单返回预测结果，后续可以在此处添加更复杂的后处理逻辑
        return preds

    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
        """Performs inference on an image or stream."""
        # 根据传入的参数确定是否进行流式推理
        self.stream = stream
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)
        else:
            # 对非流式推理的结果进行汇总，返回一个结果列表
            return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Result into one

    def predict_cli(self, source=None, model=None):
        """
        Method used for Command Line Interface (CLI) prediction.

        This function is designed to run predictions using the CLI. It sets up the source and model, then processes
        the inputs in a streaming manner. This method ensures that no outputs accumulate in memory by consuming the
        generator without storing results.

        Note:
            Do not modify this function or remove the generator. The generator ensures that no outputs are
            accumulated in memory, which is critical for preventing memory issues during long-running predictions.
        """
        # 获取流式推理生成器并逐个消费其结果，确保在长时间运行的预测过程中不会出现内存积累问题
        gen = self.stream_inference(source, model)
        for _ in gen:  # sourcery skip: remove-empty-nested-block, noqa
            pass
    def setup_source(self, source):
        """
        Sets up source and inference mode.
        """
        # 检查并获取图片尺寸
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        
        # 根据任务类型设置数据变换（如果是分类任务）
        self.transforms = (
            getattr(
                self.model.model,
                "transforms",
                classify_transforms(self.imgsz[0], crop_fraction=self.args.crop_fraction),
            )
            if self.args.task == "classify"
            else None
        )
        
        # 载入推理数据集
        self.dataset = load_inference_source(
            source=source,
            batch=self.args.batch,
            vid_stride=self.args.vid_stride,
            buffer=self.args.stream_buffer,
        )
        
        # 设置数据源类型
        self.source_type = self.dataset.source_type
        
        # 如果不是流式处理，并且数据源类型表明是流式或截图，或者数据集长度超过1000（很多图片），或者数据集中包含视频标志
        if not getattr(self, "stream", True) and (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000  # many images
            or any(getattr(self.dataset, "video_flag", [False]))
        ):  # videos
            LOGGER.warning(STREAM_WARNING)  # 发出流式处理警告
        
        # 初始化视频写入器
        self.vid_writer = {}

    @smart_inference_mode()
    def setup_model(self, model, verbose=True):
        """
        Initialize YOLO model with given parameters and set it to evaluation mode.
        """
        # 使用给定参数初始化 YOLO 模型，并设置为评估模式
        self.model = AutoBackend(
            weights=model or self.args.model,
            device=select_device(self.args.device, verbose=verbose),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
            batch=self.args.batch,
            fuse=True,
            verbose=verbose,
        )

        # 更新设备信息
        self.device = self.model.device  # update device
        self.args.half = self.model.fp16  # update half
        
        # 设置模型为评估模式
        self.model.eval()
    def write_results(self, i, p, im, s):
        """Write inference results to a file or directory."""
        string = ""  # 用于存储输出字符串

        # 如果图像是三维的，扩展成四维（针对批处理维度）
        if len(im.shape) == 3:
            im = im[None]

        # 如果数据源是流、图像或张量，则添加序号和帧数信息
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:
            string += f"{i}: "  # 输出结果序号
            frame = self.dataset.count
        else:
            # 从字符串 s[i] 中提取帧数信息
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None  # 如果未确定帧数，则默认为0

        # 设置保存结果的文本路径
        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))

        # 构建输出字符串，包括图像尺寸和推理结果的详细信息和速度
        string += "%gx%g " % im.shape[2:]
        result = self.results[i]
        result.save_dir = self.save_dir.__str__()  # 在其他位置可能会用到保存目录的字符串表示
        string += f"{result.verbose()}{result.speed['inference']:.1f}ms"

        # 如果需要保存或展示结果图像
        if self.args.save or self.args.show:
            self.plotted_img = result.plot(
                line_width=self.args.line_width,
                boxes=self.args.show_boxes,
                conf=self.args.show_conf,
                labels=self.args.show_labels,
                im_gpu=None if self.args.retina_masks else im[i],
            )

        # 如果需要保存为文本文件
        if self.args.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)

        # 如果需要保存裁剪的结果
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)

        # 如果需要展示结果
        if self.args.show:
            self.show(str(p))

        # 如果需要保存预测的图像
        if self.args.save:
            self.save_predicted_images(str(self.save_dir / p.name), frame)

        # 返回生成的字符串
        return string
    def save_predicted_images(self, save_path="", frame=0):
        """Save video predictions as mp4 at specified path."""
        # 获取要保存的图像
        im = self.plotted_img

        # 保存视频和流
        if self.dataset.mode in {"stream", "video"}:
            # 根据数据集模式确定帧率
            fps = self.dataset.fps if self.dataset.mode == "video" else 30
            # 创建保存帧图像的路径
            frames_path = f'{save_path.split(".", 1)[0]}_frames/'
            # 如果路径不存在于视频写入对象中，创建新视频文件
            if save_path not in self.vid_writer:  # new video
                # 如果需要保存帧图像，创建保存路径
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                # 根据操作系统选择文件后缀和编解码器
                suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                # 创建视频写入对象
                self.vid_writer[save_path] = cv2.VideoWriter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc),
                    fps=fps,  # 需要整数值，浮点数在 MP4 编解码器中会出错
                    frameSize=(im.shape[1], im.shape[0]),  # (width, height)
                )

            # 将图像写入视频文件
            self.vid_writer[save_path].write(im)
            # 如果需要保存帧图像，写入帧图像
            if self.args.save_frames:
                cv2.imwrite(f"{frames_path}{frame}.jpg", im)

        # 保存单张图像
        else:
            cv2.imwrite(save_path, im)

    def show(self, p=""):
        """Display an image in a window using OpenCV imshow()."""
        # 获取要显示的图像
        im = self.plotted_img
        # 在 Linux 系统下，如果窗口名不在已有列表中，创建新窗口
        if platform.system() == "Linux" and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(p, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # 允许窗口调整大小 (Linux)
            cv2.resizeWindow(p, im.shape[1], im.shape[0])  # (width, height)
        # 显示图像
        cv2.imshow(p, im)
        # 等待按键响应，时间取决于数据集模式是图像还是视频
        cv2.waitKey(300 if self.dataset.mode == "image" else 1)  # 1 毫秒

    def run_callbacks(self, event: str):
        """Runs all registered callbacks for a specific event."""
        # 遍历特定事件的所有注册回调函数，并依次执行
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func):
        """Add callback."""
        # 向特定事件的回调函数列表中添加新的回调函数
        self.callbacks[event].append(func)
```