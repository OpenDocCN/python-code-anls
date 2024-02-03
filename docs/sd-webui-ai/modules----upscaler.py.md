# `stable-diffusion-webui\modules\upscaler.py`

```
# 导入必要的模块
import os
from abc import abstractmethod

import PIL
from PIL import Image

# 导入自定义模块
import modules.shared
from modules import modelloader, shared

# 根据 PIL 版本选择不同的重采样方法
LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
NEAREST = (Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST)

# 定义一个名为 Upscaler 的类
class Upscaler:
    name = None
    model_path = None
    model_name = None
    model_url = None
    enable = True
    filter = None
    model = None
    user_path = None
    scalers: []
    tile = True

    # 初始化方法
    def __init__(self, create_dirs=False):
        # 初始化一些属性
        self.mod_pad_h = None
        self.tile_size = modules.shared.opts.ESRGAN_tile
        self.tile_pad = modules.shared.opts.ESRGAN_tile_overlap
        self.device = modules.shared.device
        self.img = None
        self.output = None
        self.scale = 1
        self.half = not modules.shared.cmd_opts.no_half
        self.pre_pad = 0
        self.mod_scale = None
        self.model_download_path = None

        # 如果模型路径为空且存在名称，则设置模型路径为共享模型路径下的名称文件夹
        if self.model_path is None and self.name:
            self.model_path = os.path.join(shared.models_path, self.name)
        # 如果模型路径存在且需要创建目录，则创建模型路径
        if self.model_path and create_dirs:
            os.makedirs(self.model_path, exist_ok=True)

        # 尝试导入 cv2 模块，如果成功则设置 can_tile 为 True
        try:
            import cv2  # noqa: F401
            self.can_tile = True
        except Exception:
            pass

    # 定义一个抽象方法，用于执行放大操作
    @abstractmethod
    def do_upscale(self, img: PIL.Image, selected_model: str):
        return img
    # 对图像进行放大处理
    def upscale(self, img: PIL.Image, scale, selected_model: str = None):
        # 设置放大倍数
        self.scale = scale
        # 计算放大后的目标宽度和高度
        dest_w = int((img.width * scale) // 8 * 8)
        dest_h = int((img.height * scale) // 8 * 8)

        # 循环3次
        for _ in range(3):
            # 如果图像宽度大于等于目标宽度且高度大于等于目标高度，则跳出循环
            if img.width >= dest_w and img.height >= dest_h:
                break

            # 保存当前图像的形状
            shape = (img.width, img.height)

            # 调用do_upscale方法对图像进行放大处理
            img = self.do_upscale(img, selected_model)

            # 如果放大后的图像形状与之前相同，则跳出循环
            if shape == (img.width, img.height):
                break

        # 如果放大后的图像宽度或高度不等于目标宽度或高度，则进行resize操作
        if img.width != dest_w or img.height != dest_h:
            img = img.resize((int(dest_w), int(dest_h)), resample=LANCZOS)

        # 返回处理后的图像
        return img

    @abstractmethod
    # 加载模型的抽象方法
    def load_model(self, path: str):
        pass

    # 查找模型文件的方法
    def find_models(self, ext_filter=None) -> list:
        # 调用modelloader模块的load_models方法查找模型文件
        return modelloader.load_models(model_path=self.model_path, model_url=self.model_url, command_path=self.user_path, ext_filter=ext_filter)

    # 更新状态信息的方法
    def update_status(self, prompt):
        # 打印提示信息
        print(f"\nextras: {prompt}", file=shared.progress_print_out)
# 定义一个类 UpscalerData，用于存储图像放大器的相关数据
class UpscalerData:
    # 初始化类属性
    name = None
    data_path = None
    scale: int = 4
    scaler: Upscaler = None
    model: None

    # 初始化方法，接受名称、路径、放大器、放大倍数和模型作为参数
    def __init__(self, name: str, path: str, upscaler: Upscaler = None, scale: int = 4, model=None):
        # 设置实例属性
        self.name = name
        self.data_path = path
        self.local_data_path = path
        self.scaler = upscaler
        self.scale = scale
        self.model = model


# 定义一个类 UpscalerNone，继承自 Upscaler 类
class UpscalerNone(Upscaler):
    # 初始化类属性
    name = "None"
    scalers = []

    # 加载模型的方法，不执行任何操作
    def load_model(self, path):
        pass

    # 执行图像放大的方法，直接返回原始图像
    def do_upscale(self, img, selected_model=None):
        return img

    # 初始化方法，接受目录名作为参数
    def __init__(self, dirname=None):
        # 调用父类的初始化方法
        super().__init__(False)
        # 设置类属性
        self.scalers = [UpscalerData("None", None, self)]


# 定义一个类 UpscalerLanczos，继承自 Upscaler 类
class UpscalerLanczos(Upscaler):
    scalers = []

    # 执行图像放大的方法，使用 Lanczos 插值算法对图像进行放大
    def do_upscale(self, img, selected_model=None):
        return img.resize((int(img.width * self.scale), int(img.height * self.scale)), resample=LANCZOS)

    # 加载模型的方法，不执行任何操作
    def load_model(self, _):
        pass

    # 初始化方法，接受目录名作为参数
    def __init__(self, dirname=None):
        # 调用父类的初始化方法
        super().__init__(False)
        # 设置类属性
        self.name = "Lanczos"
        self.scalers = [UpscalerData("Lanczos", None, self)]


# 定义一个类 UpscalerNearest，继承自 Upscaler 类
class UpscalerNearest(Upscaler):
    scalers = []

    # 执行图像放大的方法，使用最近邻插值算法对图像进行放大
    def do_upscale(self, img, selected_model=None):
        return img.resize((int(img.width * self.scale), int(img.height * self.scale)), resample=NEAREST)

    # 加载模型的方法，不执行任何操作
    def load_model(self, _):
        pass

    # 初始化方法，接受目录名作为参数
    def __init__(self, dirname=None):
        # 调用父类的初始化方法
        super().__init__(False)
        # 设置类属性
        self.name = "Nearest"
        self.scalers = [UpscalerData("Nearest", None, self)]
```