# `.\diffusers\pipelines\marigold\marigold_image_processing.py`

```py
# 从类型提示模块导入所需类型
from typing import List, Optional, Tuple, Union

# 导入 NumPy 库作为 np
import numpy as np
# 导入 PIL 库
import PIL
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的功能模块
import torch.nn.functional as F
# 从 PIL 导入 Image 类
from PIL import Image

# 从上级模块导入 ConfigMixin 类
from ... import ConfigMixin
# 从配置工具模块导入注册配置的装饰器
from ...configuration_utils import register_to_config
# 从图像处理模块导入管道图像输入类
from ...image_processor import PipelineImageInput
# 从工具模块导入配置名称和日志工具
from ...utils import CONFIG_NAME, logging
# 从导入工具模块导入判断是否可用的 Matplotlib
from ...utils.import_utils import is_matplotlib_available

# 创建一个日志记录器，使用当前模块的名称
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义 MarigoldImageProcessor 类，继承自 ConfigMixin
class MarigoldImageProcessor(ConfigMixin):
    # 设置配置名称为常量 CONFIG_NAME
    config_name = CONFIG_NAME

    # 注册初始化方法到配置中
    @register_to_config
    def __init__(
        self,
        vae_scale_factor: int = 8,  # 变分自编码器的缩放因子，默认为 8
        do_normalize: bool = True,   # 是否进行归一化，默认为 True
        do_range_check: bool = True,  # 是否进行范围检查，默认为 True
    ):
        super().__init__()  # 调用父类的初始化方法

    # 定义静态方法，扩展张量或数组
    @staticmethod
    def expand_tensor_or_array(images: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        扩展张量或数组到指定数量的图像。
        """
        if isinstance(images, np.ndarray):  # 检查是否为 NumPy 数组
            if images.ndim == 2:  # 如果是二维数组 [H,W] -> [1,H,W,1]
                images = images[None, ..., None]
            if images.ndim == 3:  # 如果是三维数组 [H,W,C] -> [1,H,W,C]
                images = images[None]
        elif isinstance(images, torch.Tensor):  # 检查是否为 PyTorch 张量
            if images.ndim == 2:  # 如果是二维张量 [H,W] -> [1,1,H,W]
                images = images[None, None]
            elif images.ndim == 3:  # 如果是三维张量 [1,H,W] -> [1,1,H,W]
                images = images[None]
        else:  # 如果类型不匹配，抛出错误
            raise ValueError(f"Unexpected input type: {type(images)}")
        return images  # 返回处理后的图像

    # 定义静态方法，将 PyTorch 张量转换为 NumPy 图像
    @staticmethod
    def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
        """
        将 PyTorch 张量转换为 NumPy 图像。
        """
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()  # 转移到 CPU，调整维度并转换为 NumPy 数组
        return images  # 返回转换后的图像

    # 定义静态方法，将 NumPy 图像转换为 PyTorch 张量
    @staticmethod
    def numpy_to_pt(images: np.ndarray) -> torch.Tensor:
        """
        将 NumPy 图像转换为 PyTorch 张量。
        """
        if np.issubdtype(images.dtype, np.integer) and not np.issubdtype(images.dtype, np.unsignedinteger):
            # 如果图像数据类型是有符号整数，抛出错误
            raise ValueError(f"Input image dtype={images.dtype} cannot be a signed integer.")
        if np.issubdtype(images.dtype, np.complexfloating):
            # 如果图像数据类型是复数，抛出错误
            raise ValueError(f"Input image dtype={images.dtype} cannot be complex.")
        if np.issubdtype(images.dtype, bool):
            # 如果图像数据类型是布尔型，抛出错误
            raise ValueError(f"Input image dtype={images.dtype} cannot be boolean.")

        images = torch.from_numpy(images.transpose(0, 3, 1, 2))  # 将 NumPy 数组转换为 PyTorch 张量，并调整维度
        return images  # 返回转换后的张量

    # 定义静态方法，用于带抗锯齿的图像调整大小
    @staticmethod
    def resize_antialias(
        image: torch.Tensor, size: Tuple[int, int], mode: str, is_aa: Optional[bool] = None  # 接受图像、目标大小、模式和抗锯齿参数
    ) -> torch.Tensor:  # 定义一个返回类型为 torch.Tensor 的函数
        # 检查输入是否为 tensor，如果不是，抛出 ValueError 异常
        if not torch.is_tensor(image):
            raise ValueError(f"Invalid input type={type(image)}.")
        # 检查输入的 dtype 是否为浮点型，如果不是，抛出 ValueError 异常
        if not torch.is_floating_point(image):
            raise ValueError(f"Invalid input dtype={image.dtype}.")
        # 检查输入的维度是否为 4，如果不是，抛出 ValueError 异常
        if image.dim() != 4:
            raise ValueError(f"Invalid input dimensions; shape={image.shape}.")

        # 判断是否需要抗锯齿处理，并检查模式是否为双线性或双三次插值
        antialias = is_aa and mode in ("bilinear", "bicubic")
        # 对输入图像进行插值处理，调整到指定大小
        image = F.interpolate(image, size, mode=mode, antialias=antialias)

        # 返回调整大小后的图像
        return image

    @staticmethod  # 表示这是一个静态方法
    def resize_to_max_edge(image: torch.Tensor, max_edge_sz: int, mode: str) -> torch.Tensor:  # 定义一个返回类型为 torch.Tensor 的函数
        # 检查输入是否为 tensor，如果不是，抛出 ValueError 异常
        if not torch.is_tensor(image):
            raise ValueError(f"Invalid input type={type(image)}.")
        # 检查输入的 dtype 是否为浮点型，如果不是，抛出 ValueError 异常
        if not torch.is_floating_point(image):
            raise ValueError(f"Invalid input dtype={image.dtype}.")
        # 检查输入的维度是否为 4，如果不是，抛出 ValueError 异常
        if image.dim() != 4:
            raise ValueError(f"Invalid input dimensions; shape={image.shape}.")

        # 获取图像的高度和宽度
        h, w = image.shape[-2:]
        # 计算原始图像的最大边长度
        max_orig = max(h, w)
        # 计算新的高度和宽度，保持最大边不超过 max_edge_sz
        new_h = h * max_edge_sz // max_orig
        new_w = w * max_edge_sz // max_orig

        # 检查新的高度和宽度是否为 0，如果是，抛出 ValueError 异常
        if new_h == 0 or new_w == 0:
            raise ValueError(f"Extreme aspect ratio of the input image: [{w} x {h}]")

        # 调用静态方法进行抗锯齿处理并调整图像大小
        image = MarigoldImageProcessor.resize_antialias(image, (new_h, new_w), mode, is_aa=True)

        # 返回调整后的图像
        return image

    @staticmethod  # 表示这是一个静态方法
    def pad_image(image: torch.Tensor, align: int) -> Tuple[torch.Tensor, Tuple[int, int]]:  # 定义一个返回类型为元组的函数
        # 检查输入是否为 tensor，如果不是，抛出 ValueError 异常
        if not torch.is_tensor(image):
            raise ValueError(f"Invalid input type={type(image)}.")
        # 检查输入的 dtype 是否为浮点型，如果不是，抛出 ValueError 异常
        if not torch.is_floating_point(image):
            raise ValueError(f"Invalid input dtype={image.dtype}.")
        # 检查输入的维度是否为 4，如果不是，抛出 ValueError 异常
        if image.dim() != 4:
            raise ValueError(f"Invalid input dimensions; shape={image.shape}.")

        # 获取图像的高度和宽度
        h, w = image.shape[-2:]
        # 计算需要的填充量以满足对齐要求
        ph, pw = -h % align, -w % align

        # 使用重复模式对图像进行填充
        image = F.pad(image, (0, pw, 0, ph), mode="replicate")

        # 返回填充后的图像和填充量
        return image, (ph, pw)

    @staticmethod  # 表示这是一个静态方法
    def unpad_image(image: torch.Tensor, padding: Tuple[int, int]) -> torch.Tensor:  # 定义一个返回类型为 torch.Tensor 的函数
        # 检查输入是否为 tensor，如果不是，抛出 ValueError 异常
        if not torch.is_tensor(image):
            raise ValueError(f"Invalid input type={type(image)}.")
        # 检查输入的 dtype 是否为浮点型，如果不是，抛出 ValueError 异常
        if not torch.is_floating_point(image):
            raise ValueError(f"Invalid input dtype={image.dtype}.")
        # 检查输入的维度是否为 4，如果不是，抛出 ValueError 异常
        if image.dim() != 4:
            raise ValueError(f"Invalid input dimensions; shape={image.shape}.")

        # 从填充元组中提取高度和宽度的填充量
        ph, pw = padding
        # 如果填充量为 0，设置为 None，否则取负填充量
        uh = None if ph == 0 else -ph
        uw = None if pw == 0 else -pw

        # 根据计算的新的高度和宽度裁剪图像
        image = image[:, :, :uh, :uw]

        # 返回裁剪后的图像
        return image

    @staticmethod  # 表示这是一个静态方法
    def load_image_canonical(  # 定义一个接受多种类型的输入图像的函数
        image: Union[torch.Tensor, np.ndarray, Image.Image],  # 接受 tensor、numpy 数组或 PIL 图像
        device: torch.device = torch.device("cpu"),  # 设置默认设备为 CPU
        dtype: torch.dtype = torch.float32,  # 设置默认数据类型为 float32
    # 返回类型为元组，包括一个张量和一个整数
    ) -> Tuple[torch.Tensor, int]:
        # 检查输入是否为 PIL 图像类型
        if isinstance(image, Image.Image):
            # 将 PIL 图像转换为 NumPy 数组
            image = np.array(image)
    
        # 初始化图像数据类型的最大值
        image_dtype_max = None
        # 检查输入是否为 NumPy 数组或 PyTorch 张量
        if isinstance(image, (np.ndarray, torch.Tensor)):
            # 扩展张量或数组的维度
            image = MarigoldImageProcessor.expand_tensor_or_array(image)
            # 确保图像的维度是 2、3 或 4
            if image.ndim != 4:
                raise ValueError("Input image is not 2-, 3-, or 4-dimensional.")
        # 检查输入是否为 NumPy 数组
        if isinstance(image, np.ndarray):
            # 检查图像数据类型是否为有符号整数
            if np.issubdtype(image.dtype, np.integer) and not np.issubdtype(image.dtype, np.unsignedinteger):
                raise ValueError(f"Input image dtype={image.dtype} cannot be a signed integer.")
            # 检查图像数据类型是否为复数
            if np.issubdtype(image.dtype, np.complexfloating):
                raise ValueError(f"Input image dtype={image.dtype} cannot be complex.")
            # 检查图像数据类型是否为布尔值
            if np.issubdtype(image.dtype, bool):
                raise ValueError(f"Input image dtype={image.dtype} cannot be boolean.")
            # 检查图像数据类型是否为无符号整数
            if np.issubdtype(image.dtype, np.unsignedinteger):
                # 获取无符号整数的最大值
                image_dtype_max = np.iinfo(image.dtype).max
                # 转换数据类型为浮点数
                image = image.astype(np.float32)  # 因为 torch 不支持无符号数据类型超过 torch.uint8
            # 将 NumPy 数组转换为 PyTorch 张量
            image = MarigoldImageProcessor.numpy_to_pt(image)
    
        # 检查是否为张量并且不是浮点类型且没有最大值
        if torch.is_tensor(image) and not torch.is_floating_point(image) and image_dtype_max is None:
            # 检查图像数据类型是否为 uint8
            if image.dtype != torch.uint8:
                raise ValueError(f"Image dtype={image.dtype} is not supported.")
            # 设置最大值为 255
            image_dtype_max = 255
    
        # 确保输入是张量
        if not torch.is_tensor(image):
            raise ValueError(f"Input type unsupported: {type(image)}.")
    
        # 如果图像的通道数为 1，则重复通道以形成 RGB 图像
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)  # [N,1,H,W] -> [N,3,H,W]
        # 确保图像是 1 通道或 3 通道
        if image.shape[1] != 3:
            raise ValueError(f"Input image is not 1- or 3-channel: {image.shape}.")
    
        # 将图像移动到指定设备，并转换为指定数据类型
        image = image.to(device=device, dtype=dtype)
    
        # 如果存在数据类型最大值，则将图像数据归一化
        if image_dtype_max is not None:
            image = image / image_dtype_max
    
        # 返回处理后的图像
        return image
    
    # 静态方法，检查图像的值范围
    @staticmethod
    def check_image_values_range(image: torch.Tensor) -> None:
        # 确保输入是张量
        if not torch.is_tensor(image):
            raise ValueError(f"Invalid input type={type(image)}.")
        # 确保输入是浮点类型
        if not torch.is_floating_point(image):
            raise ValueError(f"Invalid input dtype={image.dtype}.")
        # 检查图像数据是否在 [0,1] 范围内
        if image.min().item() < 0.0 or image.max().item() > 1.0:
            raise ValueError("Input image data is partially outside of the [0,1] range.")
    
    # 预处理方法
    def preprocess(
        self,
        # 输入图像，可以是多种格式
        image: PipelineImageInput,
        # 可选的处理分辨率
        processing_resolution: Optional[int] = None,
        # 输入的重采样方法
        resample_method_input: str = "bilinear",
        # 指定设备（CPU 或 GPU）
        device: torch.device = torch.device("cpu"),
        # 指定数据类型，默认为浮点数
        dtype: torch.dtype = torch.float32,
    ):
        # 检查输入的图像是否为列表类型
        if isinstance(image, list):
            # 初始化图像变量
            images = None
            # 遍历图像列表，获取每个图像的索引和内容
            for i, img in enumerate(image):
                # 加载图像并标准化为指定的格式，返回形状为[N,3,H,W]
                img = self.load_image_canonical(img, device, dtype)  # [N,3,H,W]
                # 如果还没有图像，直接赋值
                if images is None:
                    images = img
                else:
                    # 检查当前图像的维度是否与已有图像兼容
                    if images.shape[2:] != img.shape[2:]:
                        # 如果不兼容，抛出错误并给出详细信息
                        raise ValueError(
                            f"Input image[{i}] has incompatible dimensions {img.shape[2:]} with the previous images "
                            f"{images.shape[2:]}"
                        )
                    # 将当前图像与已有图像在第一维拼接
                    images = torch.cat((images, img), dim=0)
            # 将最终图像集赋值回原变量
            image = images
            # 删除临时图像变量以释放内存
            del images
        else:
            # 加载单个图像并标准化为指定的格式，返回形状为[N,3,H,W]
            image = self.load_image_canonical(image, device, dtype)  # [N,3,H,W]

        # 获取图像的原始分辨率
        original_resolution = image.shape[2:]

        # 如果配置要求进行值范围检查，则执行检查
        if self.config.do_range_check:
            self.check_image_values_range(image)

        # 如果配置要求进行归一化处理，则进行操作
        if self.config.do_normalize:
            image = image * 2.0 - 1.0

        # 如果处理分辨率被指定且大于0，则调整图像大小
        if processing_resolution is not None and processing_resolution > 0:
            # 调整图像到最大边长，返回形状为[N,3,PH,PW]
            image = self.resize_to_max_edge(image, processing_resolution, resample_method_input)  # [N,3,PH,PW]

        # 对图像进行填充，返回填充后的图像和填充信息，形状为[N,3,PPH,PPW]
        image, padding = self.pad_image(image, self.config.vae_scale_factor)  # [N,3,PPH,PPW]

        # 返回处理后的图像、填充信息和原始分辨率
        return image, padding, original_resolution

    # 定义静态方法 colormap，用于图像上色
    @staticmethod
    def colormap(
        # 输入图像，支持多种类型
        image: Union[np.ndarray, torch.Tensor],
        # 颜色映射名称，默认为 "Spectral"
        cmap: str = "Spectral",
        # 是否返回字节类型的图像
        bytes: bool = False,
        # 强制使用的特定方法，默认为 None
        _force_method: Optional[str] = None,
    # 定义静态方法 visualize_depth，用于可视化深度信息
    @staticmethod
    def visualize_depth(
        # 输入深度图像，支持多种类型
        depth: Union[
            PIL.Image.Image,
            np.ndarray,
            torch.Tensor,
            List[PIL.Image.Image],
            List[np.ndarray],
            List[torch.Tensor],
        ],
        # 深度值的最小阈值，默认为0.0
        val_min: float = 0.0,
        # 深度值的最大阈值，默认为1.0
        val_max: float = 1.0,
        # 颜色映射名称，默认为 "Spectral"
        color_map: str = "Spectral",
    # 返回深度图像的可视化结果，可以是单个图像或图像列表
    ) -> Union[PIL.Image.Image, List[PIL.Image.Image]]:
        """
        可视化深度图，例如 `MarigoldDepthPipeline` 的预测结果。
    
        参数:
            depth (`Union[PIL.Image.Image, np.ndarray, torch.Tensor, List[PIL.Image.Image], List[np.ndarray],
                List[torch.Tensor]]`): 深度图。
            val_min (`float`, *可选*, 默认值为 `0.0`): 可视化深度范围的最小值。
            val_max (`float`, *可选*, 默认值为 `1.0`): 可视化深度范围的最大值。
            color_map (`str`, *可选*, 默认值为 `"Spectral"`): 用于将单通道深度预测转换为彩色表示的颜色映射。
    
        返回: `PIL.Image.Image` 或 `List[PIL.Image.Image]`，包含深度图可视化结果。
        """
        # 检查最大值是否小于等于最小值，若是则抛出错误
        if val_max <= val_min:
            raise ValueError(f"Invalid values range: [{val_min}, {val_max}].")
    
        # 定义用于可视化单个深度图的函数
        def visualize_depth_one(img, idx=None):
            # 为图像前缀生成字符串，包含索引（如果存在）
            prefix = "Depth" + (f"[{idx}]" if idx else "")
            # 检查输入图像是否为 PIL 图像
            if isinstance(img, PIL.Image.Image):
                # 验证图像模式是否为 "I;16"
                if img.mode != "I;16":
                    raise ValueError(f"{prefix}: invalid PIL mode={img.mode}.")
                # 将 PIL 图像转换为 numpy 数组并归一化
                img = np.array(img).astype(np.float32) / (2**16 - 1)
            # 检查输入图像是否为 numpy 数组或 PyTorch 张量
            if isinstance(img, np.ndarray) or torch.is_tensor(img):
                # 确保输入图像是二维的
                if img.ndim != 2:
                    raise ValueError(f"{prefix}: unexpected shape={img.shape}.")
                # 若为 numpy 数组，则转换为 PyTorch 张量
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img)
                # 确保图像是浮点类型
                if not torch.is_floating_point(img):
                    raise ValueError(f"{prefix}: unexected dtype={img.dtype}.")
            else:
                # 如果输入类型不匹配，则抛出错误
                raise ValueError(f"{prefix}: unexpected type={type(img)}.")
            # 如果最小值或最大值不为默认值，则进行归一化处理
            if val_min != 0.0 or val_max != 1.0:
                img = (img - val_min) / (val_max - val_min)
            # 使用颜色映射处理深度图像并转换为 RGB 格式
            img = MarigoldImageProcessor.colormap(img, cmap=color_map, bytes=True)  # [H,W,3]
            # 将数组转换回 PIL 图像
            img = PIL.Image.fromarray(img.cpu().numpy())
            return img
    
        # 检查输入深度是否为 None 或列表中的元素为 None
        if depth is None or isinstance(depth, list) and any(o is None for o in depth):
            raise ValueError("Input depth is `None`")
        # 如果输入深度为 numpy 数组或 PyTorch 张量
        if isinstance(depth, (np.ndarray, torch.Tensor)):
            # 扩展张量或数组以匹配预期形状
            depth = MarigoldImageProcessor.expand_tensor_or_array(depth)
            # 若为 numpy 数组，则转换为 PyTorch 张量，形状调整为 [N,1,H,W]
            if isinstance(depth, np.ndarray):
                depth = MarigoldImageProcessor.numpy_to_pt(depth)  # [N,H,W,1] -> [N,1,H,W]
            # 验证深度图形状是否符合预期
            if not (depth.ndim == 4 and depth.shape[1] == 1):  # [N,1,H,W]
                raise ValueError(f"Unexpected input shape={depth.shape}, expecting [N,1,H,W].")
            # 返回每个图像的可视化结果列表
            return [visualize_depth_one(img[0], idx) for idx, img in enumerate(depth)]
        # 如果输入深度为列表，则对每个图像进行可视化
        elif isinstance(depth, list):
            return [visualize_depth_one(img, idx) for idx, img in enumerate(depth)]
        else:
            # 如果输入类型不匹配，则抛出错误
            raise ValueError(f"Unexpected input type: {type(depth)}")
    
    # 定义静态方法标识
        @staticmethod
    # 导出深度图为16位PNG格式
    def export_depth_to_16bit_png(
            # 深度数据，支持多种输入格式
            depth: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
            # 深度值的最小范围
            val_min: float = 0.0,
            # 深度值的最大范围
            val_max: float = 1.0,
        ) -> Union[PIL.Image.Image, List[PIL.Image.Image]]:
            # 导出单张深度图为16位PNG格式的内部函数
            def export_depth_to_16bit_png_one(img, idx=None):
                # 生成深度图的前缀，用于错误信息
                prefix = "Depth" + (f"[{idx}]" if idx else "")
                # 检查输入是否为有效类型
                if not isinstance(img, np.ndarray) and not torch.is_tensor(img):
                    raise ValueError(f"{prefix}: unexpected type={type(img)}.")
                # 检查输入的维度是否为2D
                if img.ndim != 2:
                    raise ValueError(f"{prefix}: unexpected shape={img.shape}.")
                # 将PyTorch张量转换为NumPy数组
                if torch.is_tensor(img):
                    img = img.cpu().numpy()
                # 检查数据类型是否为浮点数
                if not np.issubdtype(img.dtype, np.floating):
                    raise ValueError(f"{prefix}: unexected dtype={img.dtype}.")
                # 根据给定范围标准化深度图
                if val_min != 0.0 or val_max != 1.0:
                    img = (img - val_min) / (val_max - val_min)
                # 将深度图值转换为16位整数
                img = (img * (2**16 - 1)).astype(np.uint16)
                # 将数组转换为16位PNG格式的图像
                img = PIL.Image.fromarray(img, mode="I;16")
                # 返回生成的图像
                return img
    
            # 检查输入深度数据是否为None或包含None
            if depth is None or isinstance(depth, list) and any(o is None for o in depth):
                raise ValueError("Input depth is `None`")
            # 如果输入为NumPy数组或PyTorch张量
            if isinstance(depth, (np.ndarray, torch.Tensor)):
                # 扩展张量或数组的维度
                depth = MarigoldImageProcessor.expand_tensor_or_array(depth)
                # 如果输入是NumPy数组，转换为PyTorch张量
                if isinstance(depth, np.ndarray):
                    depth = MarigoldImageProcessor.numpy_to_pt(depth)  # [N,H,W,1] -> [N,1,H,W]
                # 检查扩展后的深度图形状
                if not (depth.ndim == 4 and depth.shape[1] == 1):
                    raise ValueError(f"Unexpected input shape={depth.shape}, expecting [N,1,H,W].")
                # 返回每张深度图的16位PNG图像
                return [export_depth_to_16bit_png_one(img[0], idx) for idx, img in enumerate(depth)]
            # 如果输入是列表
            elif isinstance(depth, list):
                # 返回每张深度图的16位PNG图像
                return [export_depth_to_16bit_png_one(img, idx) for idx, img in enumerate(depth)]
            else:
                # 抛出不支持的输入类型错误
                raise ValueError(f"Unexpected input type: {type(depth)}")
    
        # 可视化法线的静态方法
        @staticmethod
        def visualize_normals(
            # 法线数据，支持多种输入格式
            normals: Union[
                np.ndarray,
                torch.Tensor,
                List[np.ndarray],
                List[torch.Tensor],
            ],
            # 是否沿X轴翻转
            flip_x: bool = False,
            # 是否沿Y轴翻转
            flip_y: bool = False,
            # 是否沿Z轴翻转
            flip_z: bool = False,
    # 返回类型为 PIL.Image.Image 或 List[PIL.Image.Image]，用于可视化表面法线
    ) -> Union[PIL.Image.Image, List[PIL.Image.Image]]:
        """
        可视化表面法线，例如 `MarigoldNormalsPipeline` 的预测结果。
    
        参数：
            normals (`Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]`):
                表面法线数据。
            flip_x (`bool`, *可选*, 默认值为 `False`): 翻转法线参考系的 X 轴。
                      默认方向为右。
            flip_y (`bool`, *可选*, 默认值为 `False`): 翻转法线参考系的 Y 轴。
                      默认方向为上。
            flip_z (`bool`, *可选*, 默认值为 `False`): 翻转法线参考系的 Z 轴。
                      默认方向为面向观察者。
    
        返回值: `PIL.Image.Image` 或 `List[PIL.Image.Image]`，包含表面法线的可视化图像。
        """
        # 初始化翻转向量为 None
        flip_vec = None
        # 如果任一翻转标志为真，则创建翻转向量
        if any((flip_x, flip_y, flip_z)):
            flip_vec = torch.tensor(
                [
                    (-1) ** flip_x,  # 根据 flip_x 计算 X 轴的翻转因子
                    (-1) ** flip_y,  # 根据 flip_y 计算 Y 轴的翻转因子
                    (-1) ** flip_z,  # 根据 flip_z 计算 Z 轴的翻转因子
                ],
                dtype=torch.float32,  # 数据类型为浮点数
            )
    
        # 定义一个用于可视化单个法线图像的函数
        def visualize_normals_one(img, idx=None):
            img = img.permute(1, 2, 0)  # 改变图像维度顺序为 (H, W, C)
            if flip_vec is not None:
                img *= flip_vec.to(img.device)  # 应用翻转向量
            img = (img + 1.0) * 0.5  # 将图像数据归一化到 [0, 1]
            img = (img * 255).to(dtype=torch.uint8, device="cpu").numpy()  # 转换为 uint8 格式并转为 numpy 数组
            img = PIL.Image.fromarray(img)  # 将 numpy 数组转换为 PIL 图像
            return img  # 返回处理后的图像
    
        # 检查输入法线是否为 None 或含有 None 的列表
        if normals is None or isinstance(normals, list) and any(o is None for o in normals):
            raise ValueError("Input normals is `None`")  # 抛出异常
    
        # 如果法线数据为 numpy 数组或 torch 张量
        if isinstance(normals, (np.ndarray, torch.Tensor)):
            normals = MarigoldImageProcessor.expand_tensor_or_array(normals)  # 扩展法线数据
            if isinstance(normals, np.ndarray):
                normals = MarigoldImageProcessor.numpy_to_pt(normals)  # 转换 numpy 数组为 PyTorch 张量，形状为 [N,3,H,W]
            # 检查法线数据的维度和形状
            if not (normals.ndim == 4 and normals.shape[1] == 3):
                raise ValueError(f"Unexpected input shape={normals.shape}, expecting [N,3,H,W].")  # 抛出异常
            # 可视化每个法线图像并返回图像列表
            return [visualize_normals_one(img, idx) for idx, img in enumerate(normals)]
        # 如果法线数据为列表
        elif isinstance(normals, list):
            # 可视化每个法线图像并返回图像列表
            return [visualize_normals_one(img, idx) for idx, img in enumerate(normals)]
        else:
            raise ValueError(f"Unexpected input type: {type(normals)}")  # 抛出异常，处理未知类型
    
    # 定义静态方法可视化不确定性
    @staticmethod
    def visualize_uncertainty(
        uncertainty: Union[
            np.ndarray,
            torch.Tensor,
            List[np.ndarray],
            List[torch.Tensor],
        ],
        saturation_percentile=95,  # 定义饱和度百分位参数，默认为95%
    ) -> Union[PIL.Image.Image, List[PIL.Image.Image]]:
        # 指定函数返回类型为单个 PIL.Image.Image 或者 PIL.Image.Image 的列表
        """
        # 文档字符串，说明函数的功能，参数及返回值
        Visualizes dense uncertainties, such as produced by `MarigoldDepthPipeline` or `MarigoldNormalsPipeline`.

        Args:
            # 参数说明，uncertainty 可以是不同类型的数组
            uncertainty (`Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]`):
                Uncertainty maps.
            # 参数说明，饱和度百分位数，默认为 95
            saturation_percentile (`int`, *optional*, defaults to `95`):
                Specifies the percentile uncertainty value visualized with maximum intensity.

        Returns: # 返回值说明
            `PIL.Image.Image` or `List[PIL.Image.Image]` with uncertainty visualization.
        """

        # 定义内部函数，用于可视化单张不确定性图
        def visualize_uncertainty_one(img, idx=None):
            # 构建图像前缀，包含索引（如果提供）
            prefix = "Uncertainty" + (f"[{idx}]" if idx else "")
            # 检查图像最小值是否小于 0，若是则抛出异常
            if img.min() < 0:
                raise ValueError(f"{prefix}: unexected data range, min={img.min()}.")
            # 将图像张量降维并转换为 NumPy 数组
            img = img.squeeze(0).cpu().numpy()
            # 计算图像的饱和度值，基于给定的百分位数
            saturation_value = np.percentile(img, saturation_percentile)
            # 将图像值归一化并限制在 0 到 255 之间
            img = np.clip(img * 255 / saturation_value, 0, 255)
            # 将图像数据类型转换为无符号整型（uint8）
            img = img.astype(np.uint8)
            # 从 NumPy 数组创建 PIL 图像对象
            img = PIL.Image.fromarray(img)
            # 返回处理后的图像
            return img

        # 检查不确定性输入是否为 None 或者是包含 None 的列表
        if uncertainty is None or isinstance(uncertainty, list) and any(o is None for o in uncertainty):
            # 抛出异常，输入不确定性为 None
            raise ValueError("Input uncertainty is `None`")
        # 如果不确定性是 NumPy 数组或 PyTorch 张量
        if isinstance(uncertainty, (np.ndarray, torch.Tensor)):
            # 扩展张量或数组以适应处理
            uncertainty = MarigoldImageProcessor.expand_tensor_or_array(uncertainty)
            # 如果不确定性为 NumPy 数组，将其转换为 PyTorch 张量
            if isinstance(uncertainty, np.ndarray):
                uncertainty = MarigoldImageProcessor.numpy_to_pt(uncertainty)  # [N,1,H,W]
            # 检查不确定性数组的维度和形状是否符合预期
            if not (uncertainty.ndim == 4 and uncertainty.shape[1] == 1):
                # 抛出异常，形状不符合预期
                raise ValueError(f"Unexpected input shape={uncertainty.shape}, expecting [N,1,H,W].")
            # 返回每个图像的可视化结果，生成图像列表
            return [visualize_uncertainty_one(img, idx) for idx, img in enumerate(uncertainty)]
        # 如果不确定性是一个列表
        elif isinstance(uncertainty, list):
            # 返回每个图像的可视化结果，生成图像列表
            return [visualize_uncertainty_one(img, idx) for idx, img in enumerate(uncertainty)]
        else:
            # 抛出异常，输入类型不符合预期
            raise ValueError(f"Unexpected input type: {type(uncertainty)}")
```