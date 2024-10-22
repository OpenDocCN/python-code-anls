# `.\cogview3-finetune\sat\sgm\modules\diffusionmodules\dit.py`

```
# 从 omegaconf 导入 DictConfig 类，用于配置管理
from omegaconf import DictConfig
# 从 functools 导入 partial 函数，用于偏函数应用
from functools import partial
# 从 einops 导入 rearrange 函数，用于重排张量
from einops import rearrange
# 导入 numpy 库，用于数值计算
import numpy as np

# 导入 PyTorch 库
import torch
# 从 torch 导入 nn 模块，包含神经网络构建相关的工具
from torch import nn
# 导入 PyTorch 的分布式训练模块
import torch.distributed

# 从 sat.model.base_model 导入 BaseModel 类，作为模型基类
from sat.model.base_model import BaseModel
# 从 sat.model.mixins 导入 BaseMixin 类，用于混入模型功能
from sat.model.mixins import BaseMixin
# 从 sat.ops.layernorm 导入 LayerNorm 类，用于层归一化
from sat.ops.layernorm import LayerNorm
# 从 sat.transformer_defaults 导入默认的 hooks 和注意力函数
from sat.transformer_defaults import HOOKS_DEFAULT, attention_fn_default
# 从 sat.mpu.utils 导入用于张量分割的工具
from sat.mpu.utils import split_tensor_along_last_dim
# 从 sgm.util 导入一些工具函数
from sgm.util import (
    disabled_train,          # 禁用训练的装饰器
    instantiate_from_config, # 从配置实例化对象
)

# 从 sgm.modules.diffusionmodules.openaimodel 导入时间步类
from sgm.modules.diffusionmodules.openaimodel import Timestep
# 从 sgm.modules.diffusionmodules.util 导入卷积、线性层和时间步嵌入等工具
from sgm.modules.diffusionmodules.util import (
    conv_nd,                # 多维卷积函数
    linear,                 # 线性变换函数
    timestep_embedding,     # 时间步嵌入函数
)

# 定义调制函数，接受输入张量、偏移量和缩放因子
def modulate(x, shift, scale):
    # 根据缩放因子和偏移量对输入张量进行调制
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# 定义解patch化函数，将补丁张量恢复为图像张量
def unpatchify(x, channels, patch_size, height, width):
    # 使用 rearrange 函数将补丁张量重排为图像张量
    x = rearrange(
        x,
        "b (h w) (c p1 p2) -> b c (h p1) (w p2)",  # 指定输入和输出的张量维度
        h=height // patch_size,  # 计算行数
        w=width // patch_size,    # 计算列数
        p1=patch_size,            # 每个补丁的高度
        p2=patch_size,            # 每个补丁的宽度
    )
    # 返回解patch化后的图像张量
    return x

# 定义图像补丁嵌入混入类，继承自 BaseMixin
class ImagePatchEmbeddingMixin(BaseMixin):
    # 初始化函数，设置输入通道、隐藏层大小、补丁大小等属性
    def __init__(
            self,
            in_channels,
            hidden_size,
            patch_size,
            text_hidden_size=None,
            do_rearrange=True,
    ):
        # 调用父类初始化
        super().__init__()
        # 设置输入通道数
        self.in_channels = in_channels
        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置补丁大小
        self.patch_size = patch_size
        # 设置文本隐藏层大小（如果有的话）
        self.text_hidden_size = text_hidden_size
        # 设置是否重排张量的标志
        self.do_rearrange = do_rearrange

        # 初始化线性层，将补丁的通道数映射到隐藏层大小
        self.proj = nn.Linear(in_channels * patch_size ** 2, hidden_size)
        # 如果提供了文本隐藏层大小，则初始化相应的线性层
        if text_hidden_size is not None:
            self.text_proj = nn.Linear(text_hidden_size, hidden_size)

    # 定义词嵌入前向传播函数，接受输入ID、图像和编码器输出
    def word_embedding_forward(self, input_ids, images, encoder_outputs, **kwargs):
        # images: B x C x H x W，表示批量图像的形状
        # 如果需要重排图像张量
        if self.do_rearrange:
            # 使用 rearrange 函数将图像重排为补丁格式
            patches_images = rearrange(
                images, "b c (h p1) (w p2) -> b (h w) (c p1 p2)",  # 指定输入和输出的张量维度
                p1=self.patch_size,  # 每个补丁的高度
                p2=self.patch_size,  # 每个补丁的宽度
            )
        else:
            # 否则直接使用原始图像张量
            patches_images = images
        # 通过线性层对补丁图像进行映射
        emb = self.proj(patches_images)

        # 如果有文本隐藏层大小
        if self.text_hidden_size is not None:
            # 对编码器输出进行线性映射
            text_emb = self.text_proj(encoder_outputs)
            # 将文本嵌入与图像嵌入在维度1上进行连接
            emb = torch.cat([text_emb, emb], dim=1)

        # 返回最终的嵌入结果
        return emb

    # 定义重新初始化函数
    def reinit(self, parent_model=None):
        # 获取线性层的权重
        w = self.proj.weight.data
        # 使用 Xavier 均匀分布初始化权重
        nn.init.xavier_uniform_(self.proj.weight)
        # 将偏置初始化为零
        nn.init.constant_(self.proj.bias, 0)
        # 删除 transformer 的词嵌入
        del self.transformer.word_embeddings

# 定义获取 2D 正弦余弦位置嵌入的函数
def get_2d_sincos_pos_embed(embed_dim, grid_height, grid_width, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # 创建一个表示网格高度的数组
    grid_h = np.arange(grid_height, dtype=np.float32)
    # 创建一个表示网格宽度的数组
    grid_w = np.arange(grid_width, dtype=np.float32)
    # 生成网格坐标，宽度优先
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    # 将网格数组堆叠到一个新的维度
    grid = np.stack(grid, axis=0)

    # 将网格重塑为 [2, 1, grid_height, grid_width] 形状
    grid = grid.reshape([2, 1, grid_height, grid_width])
    # 从网格生成二维正弦余弦位置嵌入
        pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        # 如果需要分类标记且额外的标记数大于0，则进行处理
        if cls_token and extra_tokens > 0:
            # 在位置嵌入前添加额外的零嵌入，形成新的位置嵌入
            pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
        # 返回最终的位置嵌入
        return pos_embed
# 从网格生成二维正弦余弦位置嵌入
def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    # 确保嵌入维度是偶数
    assert embed_dim % 2 == 0

    # 使用一半的维度编码网格高度
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    # 使用一半的维度编码网格宽度
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    # 将高度和宽度的嵌入合并
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    # 返回最终的嵌入
    return emb


# 从网格生成一维正弦余弦位置嵌入
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: 每个位置的输出维度
    pos: 需要编码的位置列表：大小 (M,)
    out: (M, D)
    """
    # 确保嵌入维度是偶数
    assert embed_dim % 2 == 0
    # 生成 omega 数组用于位置编码
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    # 计算频率，得到 (D/2,)
    omega = 1.0 / 10000 ** omega  # (D/2,)

    # 将位置调整为一维
    pos = pos.reshape(-1)  # (M,)
    # 计算位置与频率的外积
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    # 计算正弦和余弦嵌入
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    # 将正弦和余弦嵌入合并
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    # 返回最终的嵌入
    return emb


# 位置嵌入混合类
class PositionEmbeddingMixin(BaseMixin):
    def __init__(
            self,
            max_height,
            max_width,
            hidden_size,
            text_length=0,
            block_size=16,
            **kwargs,
    ):
        # 初始化父类
        super().__init__()
        # 设置最大高度
        self.max_height = max_height
        # 设置最大宽度
        self.max_width = max_width
        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置文本长度
        self.text_length = text_length
        # 设置块大小
        self.block_size = block_size
        # 初始化图像位置嵌入参数
        self.image_pos_embedding = nn.Parameter(
            torch.zeros(self.max_height, self.max_width, hidden_size), requires_grad=False
        )

    # 前向传播计算位置嵌入
    def position_embedding_forward(self, position_ids, target_size, **kwargs):
        ret = []
        # 遍历目标大小
        for h, w in target_size:
            # 将高度和宽度除以块大小
            h, w = h // self.block_size, w // self.block_size
            # 获取图像位置嵌入并重塑
            image_pos_embed = self.image_pos_embedding[:h, :w].reshape(h * w, -1)
            # 连接文本嵌入与图像嵌入
            pos_embed = torch.cat(
                [
                    torch.zeros(
                        (self.text_length, self.hidden_size),
                        dtype=image_pos_embed.dtype,
                        device=image_pos_embed.device,
                    ),
                    image_pos_embed,
                ],
                dim=0,
            )
            # 添加到结果列表中
            ret.append(pos_embed[None, ...])
        # 合并所有位置嵌入
        return torch.cat(ret, dim=0)

    # 重新初始化位置嵌入
    def reinit(self, parent_model=None):
        # 删除当前位置嵌入
        del self.transformer.position_embeddings
        # 获取新的二维正弦余弦位置嵌入
        pos_embed = get_2d_sincos_pos_embed(self.image_pos_embedding.shape[-1], self.max_height, self.max_width)
        # 重塑位置嵌入为二维形状
        pos_embed = pos_embed.reshape(self.max_height, self.max_width, -1)

        # 复制新的位置嵌入数据
        self.image_pos_embedding.data.copy_(torch.from_numpy(pos_embed).float())


# 最终层混合类
class FinalLayerMixin(BaseMixin):
    def __init__(
            self,
            hidden_size,
            time_embed_dim,
            patch_size,
            block_size,
            out_channels,
            elementwise_affine=False,
            eps=1e-6,
            do_unpatchify=True,
    ):
        # 调用父类构造函数进行初始化
        super().__init__()
        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置每个补丁的大小
        self.patch_size = patch_size
        # 设置块的大小
        self.block_size = block_size
        # 设置输出通道数
        self.out_channels = out_channels
        # 确定是否进行去补丁处理
        self.do_unpatchify = do_unpatchify

        # 初始化最终的层归一化，带有可学习的参数
        self.norm_final = nn.LayerNorm(
            hidden_size,
            elementwise_affine=elementwise_affine,
            eps=eps,
        )
        # 创建一个包含SiLU激活和线性层的序列
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * hidden_size),
        )
        # 初始化线性层以将隐藏状态映射到输出通道
        self.linear = nn.Linear(hidden_size, out_channels * patch_size ** 2)

    def final_forward(self, logits, emb, text_length, target_size=None, **kwargs):
        # 截取logits以获取文本长度后的部分
        x = logits[:, text_length:]
        # 使用adaln模块对嵌入进行变换并拆分为偏移和缩放
        shift, scale = self.adaln(emb).chunk(2, dim=1)
        # 对x进行归一化并应用偏移和缩放
        x = modulate(self.norm_final(x), shift, scale)
        # 通过线性层获得最终输出
        x = self.linear(x)

        # 如果需要去补丁处理
        if self.do_unpatchify:
            # 从目标大小中提取高度和宽度
            target_height, target_width = target_size[0]
            # 断言目标大小必须能被块大小整除
            assert (
                    target_height % self.block_size == 0 and target_width % self.block_size == 0
            ), "target size must be divisible by block size"
            # 计算输出高度和宽度
            out_height, out_width = (
                target_height // self.block_size * self.patch_size,
                target_width // self.block_size * self.patch_size,
            )
            # 进行去补丁处理，恢复原图
            x = unpatchify(
                x, channels=self.out_channels, patch_size=self.patch_size, height=out_height, width=out_width
            )
        # 返回最终输出
        return x

    def reinit(self, parent_model=None):
        # 使用Xavier均匀分布初始化线性层权重
        nn.init.xavier_uniform_(self.linear.weight)
        # 将线性层偏置初始化为0
        nn.init.constant_(self.linear.bias, 0)
# 定义一个混合类 AdalnAttentionMixin，继承自 BaseMixin
class AdalnAttentionMixin(BaseMixin):
    # 初始化函数，接受多个参数以设置模型的各项属性
    def __init__(
            self,
            hidden_size,  # 隐藏层大小
            num_layers,  # 层数
            time_embed_dim,  # 时间嵌入维度
            qk_ln=True,  # 是否使用查询和键的层归一化
            hidden_size_head=None,  # 头部的隐藏层大小
            elementwise_affine=False,  # 是否使用逐元素仿射变换
            eps=1e-6,  # 用于层归一化的平滑项
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 创建一个包含多个顺序模块的模块列表，每个模块由 SiLU 激活函数和线性层组成
        self.adaln_modules = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 12 * hidden_size)) for _ in range(num_layers)]
        )

        # 记录是否使用查询和键的层归一化
        self.qk_ln = qk_ln
        # 如果使用层归一化，则为查询和键分别创建模块列表
        if qk_ln:
            # 创建用于查询的层归一化模块列表
            self.query_layernorms = nn.ModuleList(
                [
                    LayerNorm(hidden_size_head, elementwise_affine=elementwise_affine, eps=eps)
                    for _ in range(num_layers)  # 为每一层创建一个层归一化模块
                ]
            )
            # 创建用于键的层归一化模块列表
            self.key_layernorms = nn.ModuleList(
                [
                    LayerNorm(hidden_size_head, elementwise_affine=elementwise_affine, eps=eps)
                    for _ in range(num_layers)  # 为每一层创建一个层归一化模块
                ]
            )

    # 定义前向传播的方法，接收隐藏状态、掩码、文本长度等参数
    def layer_forward(
            self,
            hidden_states,  # 当前层的隐藏状态
            mask,  # 掩码，用于忽略特定位置
            text_length,  # 文本的长度
            layer_id,  # 当前层的索引
            emb,  # 嵌入表示
            *args,  # 可变参数
            **kwargs,  # 关键字参数
    # 定义一个方法的结尾，接受必要的参数
        ):
            # 获取指定层的 Transformer 层
            layer = self.transformer.layers[layer_id]
            # 获取与该层对应的自适应层归一化模块
            adaln_module = self.adaln_modules[layer_id]
    
            # 从自适应层归一化模块中处理输入并将结果分块为 12 个部分
            (
                shift_msa_img,
                scale_msa_img,
                gate_msa_img,
                shift_mlp_img,
                scale_mlp_img,
                gate_mlp_img,
                shift_msa_txt,
                scale_msa_txt,
                gate_msa_txt,
                shift_mlp_txt,
                scale_mlp_txt,
                gate_mlp_txt,
            ) = adaln_module(emb).chunk(12, dim=1)
            # 扩展门控张量的维度，以便后续处理
            gate_msa_img, gate_mlp_img, gate_msa_txt, gate_mlp_txt = (
                gate_msa_img.unsqueeze(1),
                gate_mlp_img.unsqueeze(1),
                gate_msa_txt.unsqueeze(1),
                gate_mlp_txt.unsqueeze(1),
            )
    
            # 对输入的隐藏状态进行层归一化处理
            attention_input = layer.input_layernorm(hidden_states)
    
            # 对文本输入进行调制，应用相应的偏移和缩放
            text_attention_input = modulate(attention_input[:, :text_length], shift_msa_txt, scale_msa_txt)
            # 对图像输入进行调制，应用相应的偏移和缩放
            image_attention_input = modulate(attention_input[:, text_length:], shift_msa_img, scale_msa_img)
            # 将文本和图像的注意力输入合并
            attention_input = torch.cat((text_attention_input, image_attention_input), dim=1)
    
            # 计算注意力输出，应用遮罩和层 ID
            attention_output = layer.attention(attention_input, mask, layer_id=layer_id, **kwargs)
            # 如果层归一化顺序是 "sandwich"，则应用第三次层归一化
            if self.transformer.layernorm_order == "sandwich":
                attention_output = layer.third_layernorm(attention_output)
    
            # 将隐藏状态分为文本和图像部分
            text_hidden_states, image_hidden_states = hidden_states[:, :text_length], hidden_states[:, text_length:]
            # 将注意力输出分为文本和图像部分
            text_attention_output, image_attention_output = (
                attention_output[:, :text_length],
                attention_output[:, text_length:],
            )
            # 更新文本隐藏状态，加上文本注意力输出的加权
            text_hidden_states = text_hidden_states + gate_msa_txt * text_attention_output
            # 更新图像隐藏状态，加上图像注意力输出的加权
            image_hidden_states = image_hidden_states + gate_msa_img * image_attention_output
            # 合并更新后的隐藏状态
            hidden_states = torch.cat((text_hidden_states, image_hidden_states), dim=1)
    
            # 对合并后的隐藏状态进行后注意力层归一化
            mlp_input = layer.post_attention_layernorm(hidden_states)
    
            # 对文本输入进行调制，应用相应的偏移和缩放
            text_mlp_input = modulate(mlp_input[:, :text_length], shift_mlp_txt, scale_mlp_txt)
            # 对图像输入进行调制，应用相应的偏移和缩放
            image_mlp_input = modulate(mlp_input[:, text_length:], shift_mlp_img, scale_mlp_img)
            # 将文本和图像的 MLP 输入合并
            mlp_input = torch.cat((text_mlp_input, image_mlp_input), dim=1)
    
            # 计算 MLP 输出，应用层 ID
            mlp_output = layer.mlp(mlp_input, layer_id=layer_id, **kwargs)
            # 如果层归一化顺序是 "sandwich"，则应用第四次层归一化
            if self.transformer.layernorm_order == "sandwich":
                mlp_output = layer.fourth_layernorm(mlp_output)
    
            # 将隐藏状态分为文本和图像部分
            text_hidden_states, image_hidden_states = hidden_states[:, :text_length], hidden_states[:, text_length:]
            # 将 MLP 输出分为文本和图像部分
            text_mlp_output, image_mlp_output = mlp_output[:, :text_length], mlp_output[:, text_length:]
            # 更新文本隐藏状态，加上文本 MLP 输出的加权
            text_hidden_states = text_hidden_states + gate_mlp_txt * text_mlp_output
            # 更新图像隐藏状态，加上图像 MLP 输出的加权
            image_hidden_states = image_hidden_states + gate_mlp_img * image_mlp_output
            # 合并更新后的隐藏状态
            hidden_states = torch.cat((text_hidden_states, image_hidden_states), dim=1)
    
            # 返回最终的隐藏状态
            return hidden_states
    # 定义注意力前向传播函数，接收隐藏状态、掩码、层ID及其他参数
    def attention_forward(self, hidden_states, mask, layer_id, **kwargs):
        # 获取指定层的注意力模块
        attention = self.transformer.layers[layer_id].attention

        # 默认的注意力计算函数
        attention_fn = attention_fn_default
        # 如果注意力模块有自定义的注意力函数，则使用它
        if "attention_fn" in attention.hooks:
            attention_fn = attention.hooks["attention_fn"]

        # 通过隐藏状态计算查询、键、值
        qkv = attention.query_key_value(hidden_states)
        # 将查询、键、值沿最后一个维度分离
        mixed_query_layer, mixed_key_layer, mixed_value_layer = split_tensor_along_last_dim(qkv, 3)

        # 根据训练状态选择是否应用 dropout
        dropout_fn = attention.attention_dropout if self.training else None

        # 转置查询、键、值以便于后续计算
        query_layer = attention._transpose_for_scores(mixed_query_layer)
        key_layer = attention._transpose_for_scores(mixed_key_layer)
        value_layer = attention._transpose_for_scores(mixed_value_layer)

        # 如果使用层归一化，应用于查询和键
        if self.qk_ln:
            query_layernorm = self.query_layernorms[layer_id]
            key_layernorm = self.key_layernorms[layer_id]
            query_layer = query_layernorm(query_layer)
            key_layer = key_layernorm(key_layer)

        # 计算上下文层，使用指定的注意力函数
        context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kwargs)
        # 调整上下文层的维度顺序
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # 创建新的上下文层形状，适配后续计算
        new_context_layer_shape = context_layer.size()[:-2] + (attention.hidden_size_per_partition,)
        # 重新调整上下文层的形状
        context_layer = context_layer.view(*new_context_layer_shape)

        # 通过全连接层计算输出
        output = attention.dense(context_layer)
        # 如果处于训练状态，应用输出的 dropout
        if self.training:
            output = attention.output_dropout(output)

        # 返回最终输出
        return output

    # 定义多层感知机的前向传播函数
    def mlp_forward(self, hidden_states, layer_id, **kwargs):
        # 获取指定层的多层感知机模块
        mlp = self.transformer.layers[layer_id].mlp

        # 通过全连接层将隐藏状态映射到更高维度
        intermediate_parallel = mlp.dense_h_to_4h(hidden_states)
        # 应用激活函数
        intermediate_parallel = mlp.activation_func(intermediate_parallel)
        # 将高维结果映射回原维度
        output = mlp.dense_4h_to_h(intermediate_parallel)

        # 如果处于训练状态，应用 dropout
        if self.training:
            output = mlp.dropout(output)

        # 返回最终输出
        return output

    # 定义重新初始化函数，接受可选的父模型参数
    def reinit(self, parent_model=None):
        # 遍历自适应层模块
        for layer in self.adaln_modules:
            # 将最后一层的权重初始化为 0
            nn.init.constant_(layer[-1].weight, 0)
            # 将最后一层的偏置初始化为 0
            nn.init.constant_(layer[-1].bias, 0)
# 定义一个字符串到数据类型的映射
str_to_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


# 定义一个扩散变换器类，继承自基础模型
class DiffusionTransformer(BaseModel):
    # 初始化方法，接受多个参数以配置模型
    def __init__(
            self,
            in_channels,  # 输入通道数
            out_channels,  # 输出通道数
            hidden_size,  # 隐藏层大小
            patch_size,  # 图像块大小
            num_layers,  # 层数
            num_attention_heads,  # 注意力头数
            text_length,  # 文本长度
            time_embed_dim=None,  # 时间嵌入维度，默认为 None
            num_classes=None,  # 类别数量，默认为 None
            adm_in_channels=None,  # 自适应输入通道，默认为 None
            modules={},  # 额外模块，默认为空字典
            dtype="fp32",  # 数据类型，默认为 fp32
            layernorm_order="pre",  # 层归一化顺序，默认为 "pre"
            elementwise_affine=False,  # 是否启用逐元素仿射，默认为 False
            parallel_output=True,  # 是否并行输出，默认为 True
            block_size=16,  # 块大小，默认为 16
            **kwargs,  # 其他关键字参数
    ):
        # 初始化基类
        super().__init__(**kwargs)

    # 前向传播方法，定义模型的前向计算
    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        # 将输入张量 x 转换为指定的数据类型
        x = x.to(self.dtype)
        # 获取时间步的嵌入表示
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False, dtype=self.dtype)
        # 对时间嵌入进行处理
        emb = self.time_embed(t_emb)

        # 确保 y 和 x 的批次大小一致
        assert y.shape[0] == x.shape[0]
        # 将标签嵌入与时间嵌入相加
        emb = emb + self.label_emb(y)
        # 创建输入 ID、位置 ID 和注意力掩码，均初始化为 1
        input_ids = position_ids = attention_mask = torch.ones((1, 1)).to(x.dtype)

        # 调用基类的前向方法，传入多个参数以计算输出
        output = super().forward(
            images=x,  # 输入图像
            emb=emb,  # 嵌入表示
            encoder_outputs=context,  # 编码器输出
            text_length=self.text_length,  # 文本长度
            input_ids=input_ids,  # 输入 ID
            position_ids=position_ids,  # 位置 ID
            attention_mask=attention_mask,  # 注意力掩码
            **kwargs,  # 其他关键字参数
        )[0]  # 获取输出的第一个元素

        # 返回模型的输出
        return output
```