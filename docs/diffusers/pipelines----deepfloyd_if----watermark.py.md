# `.\diffusers\pipelines\deepfloyd_if\watermark.py`

```py
# 从 typing 模块导入 List 类型，用于类型注释
from typing import List

# 导入 PIL.Image 库以处理图像
import PIL.Image
# 导入 torch 库用于张量操作
import torch
# 从 PIL 导入 Image 类以创建和处理图像
from PIL import Image

# 从配置工具模块导入 ConfigMixin 类
from ...configuration_utils import ConfigMixin
# 从模型工具模块导入 ModelMixin 类
from ...models.modeling_utils import ModelMixin
# 从工具模块导入 PIL_INTERPOLATION 以获取插值方法
from ...utils import PIL_INTERPOLATION


# 定义 IFWatermarker 类，继承自 ModelMixin 和 ConfigMixin
class IFWatermarker(ModelMixin, ConfigMixin):
    # 初始化方法
    def __init__(self):
        # 调用父类初始化方法
        super().__init__()

        # 注册一个形状为 (62, 62, 4) 的零张量作为水印图像
        self.register_buffer("watermark_image", torch.zeros((62, 62, 4)))
        # 初始化水印图像的 PIL 表示为 None
        self.watermark_image_as_pil = None

    # 定义应用水印的方法，接受图像列表和可选的样本大小
    def apply_watermark(self, images: List[PIL.Image.Image], sample_size=None):
        # 从 GitHub 复制的代码

        # 获取第一张图像的高度
        h = images[0].height
        # 获取第一张图像的宽度
        w = images[0].width

        # 如果未指定样本大小，则使用图像高度
        sample_size = sample_size or h

        # 计算宽高比系数
        coef = min(h / sample_size, w / sample_size)
        # 根据系数计算图像的新高度和宽度
        img_h, img_w = (int(h / coef), int(w / coef)) if coef < 1 else (h, w)

        # 定义 S1 和 S2，用于计算 K
        S1, S2 = 1024**2, img_w * img_h
        # 计算 K 值
        K = (S2 / S1) ** 0.5
        # 计算水印大小及其在图像中的位置
        wm_size, wm_x, wm_y = int(K * 62), img_w - int(14 * K), img_h - int(14 * K)

        # 如果水印图像尚未创建
        if self.watermark_image_as_pil is None:
            # 将水印张量转换为 uint8 类型并转移到 CPU，转换为 NumPy 数组
            watermark_image = self.watermark_image.to(torch.uint8).cpu().numpy()
            # 将 NumPy 数组转换为 RGBA 模式的 PIL 图像
            watermark_image = Image.fromarray(watermark_image, mode="RGBA")
            # 将 PIL 图像保存到实例变量
            self.watermark_image_as_pil = watermark_image

        # 调整水印图像大小
        wm_img = self.watermark_image_as_pil.resize(
            (wm_size, wm_size), PIL_INTERPOLATION["bicubic"], reducing_gap=None
        )

        # 遍历输入图像列表
        for pil_img in images:
            # 将水印图像粘贴到每张图像上，使用水印的 alpha 通道作为掩码
            pil_img.paste(wm_img, box=(wm_x - wm_size, wm_y - wm_size, wm_x, wm_y), mask=wm_img.split()[-1])

        # 返回添加水印后的图像列表
        return images
```