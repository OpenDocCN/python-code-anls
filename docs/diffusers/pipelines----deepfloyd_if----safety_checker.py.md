# `.\diffusers\pipelines\deepfloyd_if\safety_checker.py`

```py
# 导入必要的库和模块
import numpy as np  # 导入 NumPy 库，用于数值计算
import torch  # 导入 PyTorch 库，用于深度学习
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
from transformers import CLIPConfig, CLIPVisionModelWithProjection, PreTrainedModel  # 导入 CLIP 配置和模型

from ...utils import logging  # 从父级模块导入日志工具


# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)


# 定义一个安全检查器类，继承自预训练模型
class IFSafetyChecker(PreTrainedModel):
    # 指定配置类为 CLIPConfig
    config_class = CLIPConfig

    # 指定不进行拆分的模块
    _no_split_modules = ["CLIPEncoderLayer"]

    # 初始化方法，接受一个配置对象
    def __init__(self, config: CLIPConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建视觉模型，使用配置中的视觉部分
        self.vision_model = CLIPVisionModelWithProjection(config.vision_config)

        # 定义线性层用于 NSFW 检测
        self.p_head = nn.Linear(config.vision_config.projection_dim, 1)
        # 定义线性层用于水印检测
        self.w_head = nn.Linear(config.vision_config.projection_dim, 1)

    # 无梯度计算的前向传播方法
    @torch.no_grad()
    def forward(self, clip_input, images, p_threshold=0.5, w_threshold=0.5):
        # 获取图像的嵌入向量
        image_embeds = self.vision_model(clip_input)[0]

        # 检测 NSFW 内容
        nsfw_detected = self.p_head(image_embeds)
        # 将输出展平为一维
        nsfw_detected = nsfw_detected.flatten()
        # 根据阈值判断是否检测到 NSFW 内容
        nsfw_detected = nsfw_detected > p_threshold
        # 转换为列表格式
        nsfw_detected = nsfw_detected.tolist()

        # 如果检测到 NSFW 内容，记录警告日志
        if any(nsfw_detected):
            logger.warning(
                "Potential NSFW content was detected in one or more images. A black image will be returned instead."
                " Try again with a different prompt and/or seed."
            )

        # 遍历每个图像，处理检测到 NSFW 内容的图像
        for idx, nsfw_detected_ in enumerate(nsfw_detected):
            if nsfw_detected_:
                # 将检测到的 NSFW 图像替换为全黑图像
                images[idx] = np.zeros(images[idx].shape)

        # 检测水印内容
        watermark_detected = self.w_head(image_embeds)
        # 将输出展平为一维
        watermark_detected = watermark_detected.flatten()
        # 根据阈值判断是否检测到水印内容
        watermark_detected = watermark_detected > w_threshold
        # 转换为列表格式
        watermark_detected = watermark_detected.tolist()

        # 如果检测到水印内容，记录警告日志
        if any(watermark_detected):
            logger.warning(
                "Potential watermarked content was detected in one or more images. A black image will be returned instead."
                " Try again with a different prompt and/or seed."
            )

        # 遍历每个图像，处理检测到水印的图像
        for idx, watermark_detected_ in enumerate(watermark_detected):
            if watermark_detected_:
                # 将检测到水印的图像替换为全黑图像
                images[idx] = np.zeros(images[idx].shape)

        # 返回处理后的图像和检测结果
        return images, nsfw_detected, watermark_detected
```