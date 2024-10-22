# `.\diffusers\pipelines\unclip\text_proj.py`

```py
# 版权声明，标明版权归 Kakao Brain 和 HuggingFace Team 所有
# 
# 根据 Apache 许可证第 2.0 版（"许可证"）授权；
# 除非遵循该许可证，否则您不得使用此文件。
# 您可以在以下网址获得许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面协议另有规定，依据该许可证分发的软件是以“原样”基础分发的，
# 不提供任何明示或暗示的担保或条件。
# 有关特定权限和限制，请参见许可证。

# 导入 PyTorch 库
import torch
# 从 PyTorch 导入神经网络模块
from torch import nn

# 从配置工具模块导入 ConfigMixin 和 register_to_config
from ...configuration_utils import ConfigMixin, register_to_config
# 从模型模块导入 ModelMixin
from ...models import ModelMixin


class UnCLIPTextProjModel(ModelMixin, ConfigMixin):
    """
    CLIP 嵌入的工具类。用于将图像和文本嵌入组合成解码器可用的格式。

    更多详细信息，请参见原始论文： https://arxiv.org/abs/2204.06125 第 2.1 节
    """

    @register_to_config
    # 初始化函数，设置模型的参数
    def __init__(
        self,
        *,
        clip_extra_context_tokens: int = 4,  # 额外上下文令牌的数量，默认为 4
        clip_embeddings_dim: int = 768,       # CLIP 嵌入的维度，默认为 768
        time_embed_dim: int,                   # 时间嵌入的维度
        cross_attention_dim,                   # 交叉注意力的维度
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 学习的无分类器自由引导嵌入参数，初始化为零
        self.learned_classifier_free_guidance_embeddings = nn.Parameter(torch.zeros(clip_embeddings_dim))

        # 为额外的 CLIP 时间嵌入设置线性变换
        self.embedding_proj = nn.Linear(clip_embeddings_dim, time_embed_dim)
        # 将 CLIP 图像嵌入转换为时间嵌入的线性变换
        self.clip_image_embeddings_project_to_time_embeddings = nn.Linear(clip_embeddings_dim, time_embed_dim)

        # 为编码器的隐藏状态参数
        self.clip_extra_context_tokens = clip_extra_context_tokens  # 保存额外上下文令牌的数量
        # 将 CLIP 嵌入映射到交叉注意力维度的线性变换
        self.clip_extra_context_tokens_proj = nn.Linear(
            clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim
        )
        # 将 CLIP 嵌入映射到编码器隐藏状态的线性变换
        self.encoder_hidden_states_proj = nn.Linear(clip_embeddings_dim, cross_attention_dim)
        # 对编码器隐藏状态进行层归一化
        self.text_encoder_hidden_states_norm = nn.LayerNorm(cross_attention_dim)
    # 定义前向传播方法，接受图像嵌入、提示嵌入、文本编码器隐藏状态和分类器自由引导标志
        def forward(self, *, image_embeddings, prompt_embeds, text_encoder_hidden_states, do_classifier_free_guidance):
            # 如果启用了分类器自由引导
            if do_classifier_free_guidance:
                # 获取图像嵌入的批次大小
                image_embeddings_batch_size = image_embeddings.shape[0]
                # 扩展学习到的分类器自由引导嵌入，以匹配批次大小
                classifier_free_guidance_embeddings = self.learned_classifier_free_guidance_embeddings.unsqueeze(0)
                classifier_free_guidance_embeddings = classifier_free_guidance_embeddings.expand(
                    image_embeddings_batch_size, -1
                )
                # 将分类器自由引导嵌入与图像嵌入拼接
                image_embeddings = torch.cat([classifier_free_guidance_embeddings, image_embeddings], dim=0)
    
            # 确保图像嵌入和提示嵌入的批次大小相等
            assert image_embeddings.shape[0] == prompt_embeds.shape[0]
    
            # 获取批次大小
            batch_size = prompt_embeds.shape[0]
    
            # 修改架构，通过投影并添加 CLIP 嵌入到现有时间步嵌入
            time_projected_prompt_embeds = self.embedding_proj(prompt_embeds)
            time_projected_image_embeddings = self.clip_image_embeddings_project_to_time_embeddings(image_embeddings)
            # 计算添加的 CLIP 时间嵌入
            additive_clip_time_embeddings = time_projected_image_embeddings + time_projected_prompt_embeds
    
            # 投影 CLIP 嵌入到四个额外的上下文标记，并与 GLIDE 文本编码器的输出序列拼接
            clip_extra_context_tokens = self.clip_extra_context_tokens_proj(image_embeddings)
            clip_extra_context_tokens = clip_extra_context_tokens.reshape(batch_size, -1, self.clip_extra_context_tokens)
            clip_extra_context_tokens = clip_extra_context_tokens.permute(0, 2, 1)
    
            # 对文本编码器隐藏状态进行投影和归一化
            text_encoder_hidden_states = self.encoder_hidden_states_proj(text_encoder_hidden_states)
            text_encoder_hidden_states = self.text_encoder_hidden_states_norm(text_encoder_hidden_states)
            # 将额外的上下文标记与文本编码器隐藏状态拼接
            text_encoder_hidden_states = torch.cat([clip_extra_context_tokens, text_encoder_hidden_states], dim=1)
    
            # 返回文本编码器隐藏状态和添加的 CLIP 时间嵌入
            return text_encoder_hidden_states, additive_clip_time_embeddings
```