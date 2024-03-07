# `.\YOLO-World\yolo_world\models\layers\yolo_bricks.py`

```py
# 版权声明，版权归腾讯公司所有
from typing import List  # 导入 List 类型

import torch  # 导入 torch 库
import torch.nn as nn  # 导入 torch.nn 模块
from torch import Tensor  # 导入 Tensor 类型
import torch.nn.functional as F  # 导入 torch.nn.functional 模块
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, Linear  # 导入 mmcv.cnn 模块中的 ConvModule、DepthwiseSeparableConvModule、Linear 类
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig  # 导入 mmdet.utils 模块中的 ConfigType、OptConfigType、OptMultiConfig 类
from mmengine.model import BaseModule  # 导入 mmengine.model 模块中的 BaseModule 类
from mmyolo.registry import MODELS  # 导入 mmyolo.registry 模块中的 MODELS 注册器
from mmyolo.models.layers import CSPLayerWithTwoConv  # 导入 mmyolo.models.layers 模块中的 CSPLayerWithTwoConv 类

@MODELS.register_module()  # 使用 MODELS 注册器注册该类
class MaxSigmoidAttnBlock(BaseModule):  # 定义 MaxSigmoidAttnBlock 类，继承自 BaseModule
    """Max Sigmoid attention block."""  # 类的简要描述
    # 初始化函数，定义了模型的各种参数
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 guide_channels: int,
                 embed_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 num_heads: int = 1,
                 use_depthwise: bool = False,
                 with_scale: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 init_cfg: OptMultiConfig = None) -> None:
        # 调用父类的初始化函数
        super().__init__(init_cfg=init_cfg)
        # 根据是否使用深度可分离卷积选择不同的卷积模块
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # 检查输出通道数和嵌入通道数是否能被头数整除
        assert (out_channels % num_heads == 0 and
                embed_channels % num_heads == 0), \
            'out_channels and embed_channels should be divisible by num_heads.'
        # 设置头数和每个头的通道数
        self.num_heads = num_heads
        self.head_channels = out_channels // num_heads

        # 如果嵌入通道数不等于输入通道数，则定义一个卷积模块用于嵌入
        self.embed_conv = ConvModule(
            in_channels,
            embed_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None) if embed_channels != in_channels else None
        # 定义一个全连接层用于引导通道到嵌入通道的映射
        self.guide_fc = Linear(guide_channels, embed_channels)
        # 定义一个偏置参数
        self.bias = nn.Parameter(torch.zeros(num_heads))
        # 如果设置了缩放参数，则定义一个缩放参数
        if with_scale:
            self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        else:
            self.scale = 1.0

        # 定义一个卷积模块用于将输入通道映射到输出通道
        self.project_conv = conv(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=1,
                                 padding=padding,
                                 conv_cfg=conv_cfg,
                                 norm_cfg=norm_cfg,
                                 act_cfg=None)
    # 定义一个前向传播函数，接受输入张量 x 和引导张量 guide，返回处理后的张量
    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        # 获取输入张量 x 的形状信息
        B, _, H, W = x.shape

        # 使用引导张量 guide 经过全连接层处理
        guide = self.guide_fc(guide)
        # 重新调整 guide 的形状，将其分成多个头部
        guide = guide.reshape(B, -1, self.num_heads, self.head_channels)
        # 如果存在嵌入卷积层，对输入张量 x 进行卷积操作，否则直接使用 x
        embed = self.embed_conv(x) if self.embed_conv is not None else x
        # 调整嵌入结果的形状，分成多个头部
        embed = embed.reshape(B, self.num_heads, self.head_channels, H, W)

        # 使用 einsum 函数计算注意力权重
        attn_weight = torch.einsum('bmchw,bnmc->bmhwn', embed, guide)
        # 在最后一个维度上取最大值
        attn_weight = attn_weight.max(dim=-1)[0]
        # 归一化注意力权重
        attn_weight = attn_weight / (self.head_channels**0.5)
        # 添加偏置项
        attn_weight = attn_weight + self.bias[None, :, None, None]
        # 对注意力权重进行 sigmoid 激活函数处理，并乘以缩放因子
        attn_weight = attn_weight.sigmoid() * self.scale

        # 对输入张量进行投影卷积
        x = self.project_conv(x)
        # 调整投影结果的形状，分成多个头部
        x = x.reshape(B, self.num_heads, -1, H, W)
        # 将投影结果与注意力权重相乘
        x = x * attn_weight.unsqueeze(2)
        # 调整结果的形状
        x = x.reshape(B, -1, H, W)
        # 返回处理后的张量
        return x
# 使用 @MODELS.register_module() 装饰器注册 MaxSigmoidCSPLayerWithTwoConv 类
@MODELS.register_module()
# 定义 MaxSigmoidCSPLayerWithTwoConv 类，继承自 CSPLayerWithTwoConv 类
class MaxSigmoidCSPLayerWithTwoConv(CSPLayerWithTwoConv):
    """Sigmoid-attention based CSP layer with two convolution layers."""
    # 类的说明文档，描述该类是基于 Sigmoid-attention 的 CSP 层，包含两个卷积层
    # 初始化函数，定义了网络结构的各种参数
    def __init__(
            self,
            in_channels: int,  # 输入通道数
            out_channels: int,  # 输出通道数
            guide_channels: int,  # 引导通道数
            embed_channels: int,  # 嵌入通道数
            num_heads: int = 1,  # 多头注意力机制的头数，默认为1
            expand_ratio: float = 0.5,  # 扩展比例，默认为0.5
            num_blocks: int = 1,  # 块的数量，默认为1
            with_scale: bool = False,  # 是否使用缩放，默认为False
            add_identity: bool = True,  # 是否添加身份连接，默认为True
            conv_cfg: OptConfigType = None,  # 卷积配置，默认为None
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),  # 归一化配置，默认为BatchNorm
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),  # 激活函数配置，默认为SiLU
            init_cfg: OptMultiConfig = None) -> None:  # 初始化配置，默认为None，返回None
        # 调用父类的初始化函数，传入各种参数
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         expand_ratio=expand_ratio,
                         num_blocks=num_blocks,
                         add_identity=add_identity,
                         conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)

        # 定义最终的卷积层，输入通道数为(3 + num_blocks) * self.mid_channels，输出通道数为out_channels
        self.final_conv = ConvModule((3 + num_blocks) * self.mid_channels,
                                     out_channels,
                                     1,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)

        # 定义注意力块，输入通道数为self.mid_channels，输出通道数为self.mid_channels
        self.attn_block = MaxSigmoidAttnBlock(self.mid_channels,
                                              self.mid_channels,
                                              guide_channels=guide_channels,
                                              embed_channels=embed_channels,
                                              num_heads=num_heads,
                                              with_scale=with_scale,
                                              conv_cfg=conv_cfg,
                                              norm_cfg=norm_cfg)
    # 定义一个前向传播函数，接受输入张量 x 和引导张量 guide，返回处理后的张量
    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        # 使用主要卷积层处理输入张量 x
        x_main = self.main_conv(x)
        # 将处理后的张量按照通道数分割成两部分
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        # 对每个分割后的部分依次应用不同的块
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        # 将最后一个处理后的部分与引导张量 guide 一起传入注意力块
        x_main.append(self.attn_block(x_main[-1], guide))
        # 将所有处理后的部分拼接在一起，然后通过最终卷积层处理得到最终输出
        return self.final_conv(torch.cat(x_main, 1))
# 注册 ImagePoolingAttentionModule 类到 MODELS 模块
@MODELS.register_module()
class ImagePoolingAttentionModule(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(self,
                 image_channels: List[int],
                 text_channels: int,
                 embed_channels: int,
                 with_scale: bool = False,
                 num_feats: int = 3,
                 num_heads: int = 8,
                 pool_size: int = 3):
        # 调用父类的初始化函数
        super().__init__()

        # 初始化各个属性
        self.text_channels = text_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.num_feats = num_feats
        self.head_channels = embed_channels // num_heads
        self.pool_size = pool_size

        # 根据 with_scale 参数决定是否添加可学习的缩放参数
        if with_scale:
            self.scale = nn.Parameter(torch.tensor([0.]), requires_grad=True)
        else:
            self.scale = 1.0
        # 创建投影层，将输入的图像通道数映射到嵌入通道数
        self.projections = nn.ModuleList([
            ConvModule(in_channels, embed_channels, 1, act_cfg=None)
            for in_channels in image_channels
        ])
        # 创建查询、键、值和投影层
        self.query = nn.Sequential(nn.LayerNorm(text_channels),
                                   Linear(text_channels, embed_channels))
        self.key = nn.Sequential(nn.LayerNorm(embed_channels),
                                 Linear(embed_channels, embed_channels))
        self.value = nn.Sequential(nn.LayerNorm(embed_channels),
                                   Linear(embed_channels, embed_channels))
        self.proj = Linear(embed_channels, text_channels)

        # 创建图像池化层，用于对图像特征进行池化
        self.image_pools = nn.ModuleList([
            nn.AdaptiveMaxPool2d((pool_size, pool_size))
            for _ in range(num_feats)
        ])
    # 前向传播函数，接收文本特征和图像特征作为输入
    def forward(self, text_features, image_features):
        # 获取 batch size
        B = image_features[0].shape[0]
        # 断言图像特征列表长度等于预定义的特征数量
        assert len(image_features) == self.num_feats
        # 计算每个图像特征的像素块数量
        num_patches = self.pool_size**2
        # 对每个图像特征进行投影和池化操作，然后将结果拼接在一起
        mlvl_image_features = [
            pool(proj(x)).view(B, -1, num_patches)
            for (x, proj, pool
                 ) in zip(image_features, self.projections, self.image_pools)
        ]
        # 将拼接后的图像特征进行维度转置
        mlvl_image_features = torch.cat(mlvl_image_features,
                                        dim=-1).transpose(1, 2)
        # 对文本特征进行查询操作
        q = self.query(text_features)
        # 对图像特征进行键值对操作
        k = self.key(mlvl_image_features)
        v = self.value(mlvl_image_features)

        # 将查询、键、值进行维度重塑
        q = q.reshape(B, -1, self.num_heads, self.head_channels)
        k = k.reshape(B, -1, self.num_heads, self.head_channels)
        v = v.reshape(B, -1, self.num_heads, self.head_channels)

        # 计算注意力权重
        attn_weight = torch.einsum('bnmc,bkmc->bmnk', q, k)
        # 缩放注意力权重
        attn_weight = attn_weight / (self.head_channels**0.5)
        # 对注意力权重进行 softmax 操作
        attn_weight = F.softmax(attn_weight, dim=-1)

        # 根据注意力权重计算加权值
        x = torch.einsum('bmnk,bkmc->bnmc', attn_weight, v)
        # 将加权值进行投影操作
        x = self.proj(x.reshape(B, -1, self.embed_channels))
        # 返回最终结果，加上文本特征并乘以缩放因子
        return x * self.scale + text_features
# 注册模块为VanillaSigmoidBlock，表示使用Sigmoid激活函数的注意力块
@MODELS.register_module()
class VanillaSigmoidBlock(BaseModule):
    """Sigmoid attention block."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 guide_channels: int,
                 embed_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 num_heads: int = 1,
                 use_depthwise: bool = False,
                 with_scale: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        # 根据是否使用深度可分离卷积选择不同的卷积模块
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # 确保输出通道数和嵌入通道数能够被头数整除
        assert (out_channels % num_heads == 0 and
                embed_channels % num_heads == 0), \
            'out_channels and embed_channels should be divisible by num_heads.'
        self.num_heads = num_heads
        self.head_channels = out_channels // num_heads

        # 定义投影卷积层
        self.project_conv = conv(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=1,
                                 padding=padding,
                                 conv_cfg=conv_cfg,
                                 norm_cfg=norm_cfg,
                                 act_cfg=None)

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        # 进行投影卷积
        x = self.project_conv(x)
        # 使用Sigmoid激活函数进行注意力加权
        x = x * x.sigmoid()
        return x


# 注册模块为EfficientCSPLayerWithTwoConv，表示使用两个卷积层的CSP层，基于Sigmoid注意力机制
@MODELS.register_module()
class EfficientCSPLayerWithTwoConv(CSPLayerWithTwoConv):
    """Sigmoid-attention based CSP layer with two convolution layers."""
    # 初始化函数，定义了一个自定义的神经网络模块
    def __init__(
            self,
            in_channels: int,  # 输入通道数
            out_channels: int,  # 输出通道数
            guide_channels: int,  # 引导通道数
            embed_channels: int,  # 嵌入通道数
            num_heads: int = 1,  # 多头注意力机制的头数，默认为1
            expand_ratio: float = 0.5,  # 扩展比例，默认为0.5
            num_blocks: int = 1,  # 块的数量，默认为1
            with_scale: bool = False,  # 是否使用缩放，默认为False
            add_identity: bool = True,  # 是否添加身份映射，默认为True
            conv_cfg: OptConfigType = None,  # 卷积配置，默认为None
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),  # 归一化配置，默认为BatchNorm
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),  # 激活函数配置，默认为SiLU
            init_cfg: OptMultiConfig = None) -> None:  # 初始化配置，默认为None
        # 调用父类的初始化函数
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         expand_ratio=expand_ratio,
                         num_blocks=num_blocks,
                         add_identity=add_identity,
                         conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)

        # 定义最终的卷积层
        self.final_conv = ConvModule((3 + num_blocks) * self.mid_channels,
                                     out_channels,
                                     1,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)

        # 定义注意力块
        self.attn_block = VanillaSigmoidBlock(self.mid_channels,
                                              self.mid_channels,
                                              guide_channels=guide_channels,
                                              embed_channels=embed_channels,
                                              num_heads=num_heads,
                                              with_scale=with_scale,
                                              conv_cfg=conv_cfg,
                                              norm_cfg=norm_cfg)
    # 定义一个前向传播函数，接受输入张量 x 和引导张量 guide，返回处理后的张量
    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        # 使用主要卷积层处理输入张量 x
        x_main = self.main_conv(x)
        # 将处理后的张量按照通道数分割成两部分
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        # 对每个块中的处理函数对最后一个处理后的张量进行处理，并将结果添加到 x_main 中
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        # 将最后一个处理后的张量和引导张量传入注意力块中进行处理，并将结果添加到 x_main 中
        x_main.append(self.attn_block(x_main[-1], guide))
        # 将所有处理后的张量拼接在一起，并传入最终卷积层进行处理，返回结果
        return self.final_conv(torch.cat(x_main, 1))
```