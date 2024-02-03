# `stable-diffusion-webui\modules\codeformer\codeformer_arch.py`

```py
# 导入必要的库和模块
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional

# 导入自定义的模块和类
from modules.codeformer.vqgan_arch import VQAutoEncoder, ResBlock
from basicsr.utils.registry import ARCH_REGISTRY

# 定义函数，用于计算特征的均值和标准差，用于自适应实例归一化
def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.

    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    # 获取特征的大小
    size = feat.size()
    # 确保输入特征是4D张量
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    # 计算特征的方差和标准差
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std

# 定义自适应实例归一化函数
def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.

    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.

    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    # 获取特征的大小
    size = content_feat.size()
    # 计算样式特征和内容特征的均值和标准差
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    # 对内容特征进行归一化
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

# 定义位置编码的类
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    # 初始化函数，设置位置编码器的参数
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        # 调用父类的初始化函数
        super().__init__()
        # 设置位置编码器的参数
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        # 如果传入了 scale 参数但未设置 normalize 为 True，则抛出数值错误
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        # 如果未传入 scale 参数，则设置默认值为 2π
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    # 前向传播函数，计算位置编码
    def forward(self, x, mask=None):
        # 如果未传入 mask 参数，则创建全零的 mask 张量
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        # 计算非 mask 部分的位置编码
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        # 如果需要归一化，则进行归一化处理
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # 生成位置编码的维度参数
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # 计算位置编码的 x 和 y 坐标
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # 对位置编码进行正弦和余弦变换，并展平
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        # 合并 x 和 y 的位置编码，并进行维度置换
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
def _get_activation_fn(activation):
    """根据激活函数名称返回对应的激活函数"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        # 实现前馈神经网络模型 - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        # 自注意力机制
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        # 前馈神经网络
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt

class Fuse_sft_block(nn.Module):
    # 初始化函数，接受输入通道数和输出通道数作为参数
    def __init__(self, in_ch, out_ch):
        # 调用父类的初始化函数
        super().__init__()
        # 创建编码器的残差块，输入通道数为两倍的输入通道数，输出通道数为指定的输出通道数
        self.encode_enc = ResBlock(2*in_ch, out_ch)

        # 创建尺度变换层，包括两个卷积层和激活函数
        self.scale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        # 创建平移变换层，包括两个卷积层和激活函数
        self.shift = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

    # 前向传播函数，接受编码器特征、解码器特征和权重作为参数
    def forward(self, enc_feat, dec_feat, w=1):
        # 将编码器特征和解码器特征拼接后输入编码器残差块
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        # 经过尺度变换层得到尺度变换结果
        scale = self.scale(enc_feat)
        # 经过平移变换层得到平移变换结果
        shift = self.shift(enc_feat)
        # 计算残差项，包括尺度变换结果和平移变换结果的加权和
        residual = w * (dec_feat * scale + shift)
        # 最终输出为解码器特征和残差项的和
        out = dec_feat + residual
        # 返回输出结果
        return out
# 注册 CodeFormer 类到 ARCH_REGISTRY 中
@ARCH_REGISTRY.register()
class CodeFormer(VQAutoEncoder):
    # 初始化 CodeFormer 类
    def __init__(self, dim_embd=512, n_head=8, n_layers=9,
                codebook_size=1024, latent_size=256,
                connect_list=('32', '64', '128', '256'),
                fix_modules=('quantize', 'generator')):
        # 调用父类 VQAutoEncoder 的初始化方法
        super(CodeFormer, self).__init__(512, 64, [1, 2, 2, 4, 4, 8], 'nearest',2, [16], codebook_size)

        # 如果指定了需要固定的模块，则将其参数设置为不需要梯度更新
        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False

        # 设置类的属性
        self.connect_list = connect_list
        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd*2

        # 初始化位置编码和特征编码
        self.position_emb = nn.Parameter(torch.zeros(latent_size, self.dim_embd))
        self.feat_emb = nn.Linear(256, self.dim_embd)

        # 初始化 transformer 层
        self.ft_layers = nn.Sequential(*[TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0)
                                    for _ in range(self.n_layers)])

        # 初始化 logits_predict 头部
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False))

        # 定义通道数
        self.channels = {
            '16': 512,
            '32': 256,
            '64': 256,
            '128': 128,
            '256': 128,
            '512': 64,
        }

        # 定义 fuse_encoder_block 和 fuse_generator_block
        # fuse_encoder_block: 在第二个残差块之后（大于16通道），在等于16通道时在注意力层之前
        self.fuse_encoder_block = {'512':2, '256':5, '128':8, '64':11, '32':14, '16':18}
        # fuse_generator_block: 在第一个残差块之后（大于16通道），在等于16通道时在注意力层之前
        self.fuse_generator_block = {'16':6, '32': 9, '64':12, '128':15, '256':18, '512':21}

        # 初始化 fuse_convs_dict
        self.fuse_convs_dict = nn.ModuleDict()
        for f_size in self.connect_list:
            in_ch = self.channels[f_size]
            self.fuse_convs_dict[f_size] = Fuse_sft_block(in_ch, in_ch)
    # 初始化神经网络模块的权重
    def _init_weights(self, module):
        # 如果模块是线性层或者嵌入层
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # 从正态分布中初始化权重数据
            module.weight.data.normal_(mean=0.0, std=0.02)
            # 如果是线性层并且有偏置项，则将偏置项初始化为0
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为0
            module.bias.data.zero_()
            # 将权重初始化为1
            module.weight.data.fill_(1.0)
```