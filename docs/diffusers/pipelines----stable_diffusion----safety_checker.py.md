# `.\diffusers\pipelines\stable_diffusion\safety_checker.py`

```py
# 版权声明，表明该代码的版权所有者及其权利
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 在 Apache 2.0 许可证下授权（"许可证"）；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，
# 否则根据许可证分发的软件是按“原样”基础提供的，
# 不提供任何明示或暗示的担保或条件。
# 有关许可证所 governing 权限和限制的详细信息，请参见许可证。

# 导入 NumPy 库，用于数值计算
import numpy as np
# 导入 PyTorch 库
import torch
# 导入 PyTorch 神经网络模块
import torch.nn as nn
# 从 transformers 库导入 CLIP 配置、视觉模型和预训练模型
from transformers import CLIPConfig, CLIPVisionModel, PreTrainedModel

# 从上级目录导入 logging 工具
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义计算余弦距离的函数
def cosine_distance(image_embeds, text_embeds):
    # 对图像嵌入进行归一化处理
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    # 对文本嵌入进行归一化处理
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    # 返回归一化的图像嵌入和文本嵌入的矩阵乘法结果
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())

# 定义 StableDiffusionSafetyChecker 类，继承自 PreTrainedModel
class StableDiffusionSafetyChecker(PreTrainedModel):
    # 设置配置类为 CLIPConfig
    config_class = CLIPConfig
    # 指定主要输入名称为 "clip_input"
    main_input_name = "clip_input"

    # 指定不需要拆分的模块列表
    _no_split_modules = ["CLIPEncoderLayer"]

    # 初始化方法，接收 CLIPConfig 配置
    def __init__(self, config: CLIPConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化视觉模型，使用配置中的视觉部分
        self.vision_model = CLIPVisionModel(config.vision_config)
        # 创建线性层进行视觉投影，输入维度为 hidden_size，输出维度为 projection_dim
        self.visual_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        # 创建概念嵌入的可学习参数，维度为 (17, projection_dim)，并设置为不需要梯度更新
        self.concept_embeds = nn.Parameter(torch.ones(17, config.projection_dim), requires_grad=False)
        # 创建特殊关心的嵌入参数，维度为 (3, projection_dim)，并设置为不需要梯度更新
        self.special_care_embeds = nn.Parameter(torch.ones(3, config.projection_dim), requires_grad=False)

        # 创建概念嵌入权重参数，维度为 (17)，并设置为不需要梯度更新
        self.concept_embeds_weights = nn.Parameter(torch.ones(17), requires_grad=False)
        # 创建特殊关心嵌入权重参数，维度为 (3)，并设置为不需要梯度更新
        self.special_care_embeds_weights = nn.Parameter(torch.ones(3), requires_grad=False)

    # 该装饰器指示后续方法不需要计算梯度
    @torch.no_grad()
    # 前向传播方法，接收 CLIP 输入和图像
    def forward(self, clip_input, images):
        # 使用视觉模型处理 CLIP 输入，获取池化输出
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        # 将池化输出通过视觉投影层，生成图像嵌入
        image_embeds = self.visual_projection(pooled_output)

        # 始终将结果转换为 float32，避免显著的开销，且与 bfloat16 兼容
        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds).cpu().float().numpy()
        # 计算图像嵌入与概念嵌入之间的余弦距离
        cos_dist = cosine_distance(image_embeds, self.concept_embeds).cpu().float().numpy()

        # 初始化结果列表
        result = []
        # 获取批次大小
        batch_size = image_embeds.shape[0]
        # 遍历每一张图像
        for i in range(batch_size):
            # 初始化结果字典，包含特殊分数、特殊关心和概念分数
            result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []}

            # 增加此值以创建更强的 `nfsw` 过滤器
            # 代价是增加过滤良性图像的可能性
            adjustment = 0.0

            # 遍历每个特殊概念的余弦距离
            for concept_idx in range(len(special_cos_dist[0])):
                # 获取当前概念的余弦距离
                concept_cos = special_cos_dist[i][concept_idx]
                # 获取当前概念的阈值
                concept_threshold = self.special_care_embeds_weights[concept_idx].item()
                # 计算并存储特殊分数
                result_img["special_scores"][concept_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                # 如果特殊分数大于0，添加到特殊关心列表
                if result_img["special_scores"][concept_idx] > 0:
                    result_img["special_care"].append({concept_idx, result_img["special_scores"][concept_idx]})
                    # 增加调整值
                    adjustment = 0.01

            # 遍历每个概念的余弦距离
            for concept_idx in range(len(cos_dist[0])):
                # 获取当前概念的余弦距离
                concept_cos = cos_dist[i][concept_idx]
                # 获取当前概念的阈值
                concept_threshold = self.concept_embeds_weights[concept_idx].item()
                # 计算并存储概念分数
                result_img["concept_scores"][concept_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                # 如果概念分数大于0，添加到不良概念列表
                if result_img["concept_scores"][concept_idx] > 0:
                    result_img["bad_concepts"].append(concept_idx)

            # 将当前图像的结果添加到结果列表
            result.append(result_img)

        # 检查是否存在任何不适宜内容的概念
        has_nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in result]

        # 遍历每个结果，处理不适宜内容的图像
        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
            if has_nsfw_concept:
                # 如果 images 是张量，将该图像替换为黑图
                if torch.is_tensor(images) or torch.is_tensor(images[0]):
                    images[idx] = torch.zeros_like(images[idx])  # black image
                # 如果 images 是 NumPy 数组，将该图像替换为黑图
                else:
                    images[idx] = np.zeros(images[idx].shape)  # black image

        # 如果检测到任何不适宜内容，记录警告信息
        if any(has_nsfw_concepts):
            logger.warning(
                "Potential NSFW content was detected in one or more images. A black image will be returned instead."
                " Try again with a different prompt and/or seed."
            )

        # 返回处理后的图像和不适宜内容的概念标识
        return images, has_nsfw_concepts

    # 在不计算梯度的情况下进行操作
    @torch.no_grad()
    # 定义一个处理输入的前向传播方法，接收 CLIP 输入和图像张量
    def forward_onnx(self, clip_input: torch.Tensor, images: torch.Tensor):
        # 通过视觉模型处理 CLIP 输入，获取池化输出
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        # 将池化输出通过视觉投影生成图像嵌入
        image_embeds = self.visual_projection(pooled_output)
    
        # 计算图像嵌入与特殊关心嵌入之间的余弦距离
        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds)
        # 计算图像嵌入与概念嵌入之间的余弦距离
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)
    
        # 增加此值以创建更强的 NSFW 过滤器
        # 代价是增加过滤良性图像的可能性
        adjustment = 0.0
    
        # 计算特殊分数，考虑余弦距离、特殊关心嵌入权重和调整值
        special_scores = special_cos_dist - self.special_care_embeds_weights + adjustment
        # 对特殊分数进行四舍五入（注释掉的代码）
        # special_scores = special_scores.round(decimals=3)
        # 检查特殊分数是否大于零，返回布尔值表示是否关注
        special_care = torch.any(special_scores > 0, dim=1)
        # 如果特殊关心成立，调整值增加
        special_adjustment = special_care * 0.01
        # 扩展调整值以匹配余弦距离的形状
        special_adjustment = special_adjustment.unsqueeze(1).expand(-1, cos_dist.shape[1])
    
        # 计算概念分数，考虑余弦距离、概念嵌入权重和特殊调整
        concept_scores = (cos_dist - self.concept_embeds_weights) + special_adjustment
        # 对概念分数进行四舍五入（注释掉的代码）
        # concept_scores = concept_scores.round(decimals=3)
        # 检查概念分数是否大于零，返回布尔值表示是否有 NSFW 概念
        has_nsfw_concepts = torch.any(concept_scores > 0, dim=1)
    
        # 将具有 NSFW 概念的图像设置为黑色图像
        images[has_nsfw_concepts] = 0.0  # black image
    
        # 返回处理后的图像和 NSFW 概念的布尔值
        return images, has_nsfw_concepts
```