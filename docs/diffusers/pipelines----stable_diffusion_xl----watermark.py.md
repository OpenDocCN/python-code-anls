# `.\diffusers\pipelines\stable_diffusion_xl\watermark.py`

```py
# 导入 numpy 库，用于数值计算
import numpy as np
# 导入 torch 库，用于深度学习操作
import torch

# 从上级目录导入 is_invisible_watermark_available 函数，用于检查水印功能是否可用
from ...utils import is_invisible_watermark_available

# 如果水印功能可用，则导入 WatermarkEncoder 类
if is_invisible_watermark_available():
    from imwatermark import WatermarkEncoder

# 定义水印信息的二进制表示
WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
# 将水印信息转换为二进制字符串，并将每个比特转换为 0/1 的整数列表
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]

# 创建一个用于水印的类
class StableDiffusionXLWatermarker:
    # 初始化方法
    def __init__(self):
        # 设置水印比特
        self.watermark = WATERMARK_BITS
        # 创建水印编码器实例
        self.encoder = WatermarkEncoder()

        # 使用比特设置水印
        self.encoder.set_watermark("bits", self.watermark)

    # 应用水印的方法，接收图像张量作为输入
    def apply_watermark(self, images: torch.Tensor):
        # 如果图像尺寸小于 256，则不能编码，直接返回原图像
        if images.shape[-1] < 256:
            return images

        # 将图像标准化到 0-255，并调整维度顺序
        images = (255 * (images / 2 + 0.5)).cpu().permute(0, 2, 3, 1).float().numpy()

        # 将 RGB 图像转换为 BGR，以符合水印编码器的通道顺序
        images = images[:, :, :, ::-1]

        # 添加水印并将 BGR 图像转换回 RGB
        images = [self.encoder.encode(image, "dwtDct")[:, :, ::-1] for image in images]

        # 将列表转换为 numpy 数组
        images = np.array(images)

        # 将 numpy 数组转换回 torch 张量，并调整维度顺序
        images = torch.from_numpy(images).permute(0, 3, 1, 2)

        # 将图像数值重新标准化到 [-1, 1] 范围
        images = torch.clamp(2 * (images / 255 - 0.5), min=-1.0, max=1.0)
        # 返回处理后的图像张量
        return images
```