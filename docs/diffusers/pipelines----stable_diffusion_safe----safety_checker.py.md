# `.\diffusers\pipelines\stable_diffusion_safe\safety_checker.py`

```py
# 版权声明，表明此文件归 HuggingFace 团队所有
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache License 2.0 版许可使用此文件
# Licensed under the Apache License, Version 2.0 (the "License");
# 仅在遵守许可的情况下使用此文件
# you may not use this file except in compliance with the License.
# 可以在以下网址获取许可证的副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非根据适用法律或书面协议另有规定，否则根据许可证分发的软件是“按原样”提供的
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何明示或暗示的担保或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 请参阅许可证以了解特定的权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入 PyTorch 库
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 从 transformers 库导入 CLIP 配置和模型
from transformers import CLIPConfig, CLIPVisionModel, PreTrainedModel

# 从上级模块导入日志工具
from ...utils import logging

# 创建一个记录器，用于记录日志信息
logger = logging.get_logger(__name__)

# 定义计算余弦距离的函数
def cosine_distance(image_embeds, text_embeds):
    # 对图像嵌入进行归一化处理
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    # 对文本嵌入进行归一化处理
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    # 返回图像嵌入与文本嵌入的余弦相似度矩阵
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())

# 定义安全的稳定扩散安全检查器类，继承自预训练模型
class SafeStableDiffusionSafetyChecker(PreTrainedModel):
    # 设置配置类为 CLIPConfig
    config_class = CLIPConfig

    # 定义不进行模型分割的模块列表
    _no_split_modules = ["CLIPEncoderLayer"]

    # 初始化方法，接收 CLIPConfig 配置
    def __init__(self, config: CLIPConfig):
        # 调用父类构造函数
        super().__init__(config)

        # 创建 CLIP 视觉模型实例
        self.vision_model = CLIPVisionModel(config.vision_config)
        # 定义线性层，用于视觉投影
        self.visual_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        # 定义概念嵌入参数，不需要梯度更新
        self.concept_embeds = nn.Parameter(torch.ones(17, config.projection_dim), requires_grad=False)
        # 定义特殊关注嵌入参数，不需要梯度更新
        self.special_care_embeds = nn.Parameter(torch.ones(3, config.projection_dim), requires_grad=False)

        # 定义概念嵌入权重参数，不需要梯度更新
        self.concept_embeds_weights = nn.Parameter(torch.ones(17), requires_grad=False)
        # 定义特殊关注嵌入权重参数，不需要梯度更新
        self.special_care_embeds_weights = nn.Parameter(torch.ones(3), requires_grad=False)

    # 使用装饰器，表示该方法在计算图中不需要梯度
    @torch.no_grad()
    # 定义前向传播方法，接受 CLIP 输入和图像
        def forward(self, clip_input, images):
            # 使用视觉模型处理 CLIP 输入，并提取池化输出
            pooled_output = self.vision_model(clip_input)[1]  # pooled_output
            # 将池化输出进行视觉投影，生成图像嵌入
            image_embeds = self.visual_projection(pooled_output)
    
            # 计算图像嵌入与特殊关心嵌入之间的余弦距离，并转换为 NumPy 数组
            special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds).cpu().float().numpy()
            # 计算图像嵌入与概念嵌入之间的余弦距离，并转换为 NumPy 数组
            cos_dist = cosine_distance(image_embeds, self.concept_embeds).cpu().float().numpy()
    
            # 初始化结果列表
            result = []
            # 获取当前批次的大小
            batch_size = image_embeds.shape[0]
            # 遍历每个图像
            for i in range(batch_size):
                # 初始化结果字典，存储分数和关心概念
                result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []}
    
                # 调整值，增加此值将增强 `nfsw` 过滤器的强度
                adjustment = 0.0
    
                # 遍历特殊关心嵌入的余弦距离
                for concept_idx in range(len(special_cos_dist[0])):
                    concept_cos = special_cos_dist[i][concept_idx]  # 获取特定概念的余弦距离
                    concept_threshold = self.special_care_embeds_weights[concept_idx].item()  # 获取对应阈值
                    # 计算特殊分数并进行四舍五入
                    result_img["special_scores"][concept_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                    # 如果分数大于零，记录特殊关心概念及其分数
                    if result_img["special_scores"][concept_idx] > 0:
                        result_img["special_care"].append({concept_idx, result_img["special_scores"][concept_idx]})
                        adjustment = 0.01  # 调整值增加，防止重复
    
                # 遍历概念嵌入的余弦距离
                for concept_idx in range(len(cos_dist[0])):
                    concept_cos = cos_dist[i][concept_idx]  # 获取特定概念的余弦距离
                    concept_threshold = self.concept_embeds_weights[concept_idx].item()  # 获取对应阈值
                    # 计算概念分数并进行四舍五入
                    result_img["concept_scores"][concept_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                    # 如果分数大于零，记录不良概念
                    if result_img["concept_scores"][concept_idx] > 0:
                        result_img["bad_concepts"].append(concept_idx)
    
                # 将当前图像结果添加到总结果中
                result.append(result_img)
    
            # 检查是否存在不良概念
            has_nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in result]
    
            # 返回图像和不良概念的标记
            return images, has_nsfw_concepts
    
        # 禁用梯度计算，减少内存消耗
        @torch.no_grad()
    # 定义前向传播方法，接受输入张量和图像张量
    def forward_onnx(self, clip_input: torch.Tensor, images: torch.Tensor):
        # 通过视觉模型处理输入，获取池化输出
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        # 将池化输出通过视觉投影生成图像嵌入
        image_embeds = self.visual_projection(pooled_output)
    
        # 计算图像嵌入与特殊关注嵌入之间的余弦距离
        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds)
        # 计算图像嵌入与概念嵌入之间的余弦距离
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)
    
        # 增加该值以创建更强的“nsfw”过滤器
        # 但可能增加误过滤无害图像的概率
        adjustment = 0.0
    
        # 计算特殊评分，调整后与特殊关注嵌入权重比较
        special_scores = special_cos_dist - self.special_care_embeds_weights + adjustment
        # 取整到小数点后三位（已注释）
        # special_scores = special_scores.round(decimals=3)
        # 判断特殊评分是否大于0，生成布尔值张量
        special_care = torch.any(special_scores > 0, dim=1)
        # 为特殊关注创建调整值
        special_adjustment = special_care * 0.01
        # 扩展调整值以匹配余弦距离的形状
        special_adjustment = special_adjustment.unsqueeze(1).expand(-1, cos_dist.shape[1])
    
        # 计算概念评分，加入特殊调整
        concept_scores = (cos_dist - self.concept_embeds_weights) + special_adjustment
        # 取整到小数点后三位（已注释）
        # concept_scores = concept_scores.round(decimals=3)
        # 判断概念评分是否大于0，生成布尔值张量
        has_nsfw_concepts = torch.any(concept_scores > 0, dim=1)
    
        # 返回图像和是否存在nsfw概念的布尔值
        return images, has_nsfw_concepts
```