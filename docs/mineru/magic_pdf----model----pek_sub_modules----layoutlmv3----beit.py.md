# `.\MinerU\magic_pdf\model\pek_sub_modules\layoutlmv3\beit.py`

```
# Vision Transformer (ViT) 的 PyTorch 实现
""" Vision Transformer (ViT) in PyTorch

# 该实现参考了论文《一张图片值 16 x 16 个词：大规模图像识别的 Transformers》
A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

# 官方 JAX 代码链接
The official jax code is released and available at https://github.com/google-research/vision_transformer

# 状态和待办事项
Status/TODO:
* 模型更新以兼容官方实现，添加参数以支持旧 PyTorch 权重的向后兼容。
* 权重从官方 JAX 实现移植，支持 384x384 基本和小型模型，16x16 和 32x32 的图像块。
* 在 ImageNet-1k 上对我自定义的 '小型' 图像块模型进行监督训练，获得 77.9，'基本' 模型 79.4 的 Top-1 准确率。
* 希望未来有时间和 GPU 对 OpenImages 进行 SSL 或无监督预训练，并在 ImageNet 上进行微调。

# 感谢致辞
Acknowledgments:
* 感谢论文作者发布代码和权重！
* 我根据 Phil Wang 的实现修复了我的类 token 实现，链接见 https://github.com/lucidrains/vit-pytorch ... 可以查看一些 einops/einsum 的有趣内容
* 简单的 transformer 风格灵感来自 Andrej Karpathy 的 https://github.com/karpathy/minGPT
* Bert 参考代码与 Huggingface Transformers 和 Tensorflow Bert 进行比较

# 版权信息
Hacked together by / Copyright 2020 Ross Wightman
"""
# 导入警告模块，用于处理警告信息
import warnings
# 导入数学模块，用于数学计算
import math
# 导入 PyTorch 主模块
import torch
# 导入部分函数，用于函数式编程
from functools import partial
# 导入 PyTorch 神经网络模块
import torch.nn as nn
# 导入 PyTorch 功能模块，提供各种激活函数
import torch.nn.functional as F
# 导入 PyTorch 检查点功能模块
import torch.utils.checkpoint as checkpoint
# 从 timm 库导入常用层和工具函数
from timm.models.layers import drop_path, to_2tuple, trunc_normal_


# 配置函数，用于设置模型的超参数和输入信息
def _cfg(url='', **kwargs):
    return {
        # 设置模型权重下载链接
        'url': url,
        # 设置类别数、输入尺寸和其他参数
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


# 定义 DropPath 类，应用于残差块的随机深度
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    # 初始化函数，设置丢弃概率
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    # 前向传播函数
    def forward(self, x):
        # 应用随机深度的丢弃操作
        return drop_path(x, self.drop_prob, self.training)

    # 额外的字符串表示函数
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


# 定义 Mlp 类，构建多层感知机结构
class Mlp(nn.Module):
    # 初始化函数，设置输入、隐藏和输出特征，激活层和丢弃率
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # 确定输出特征数，若未指定则使用输入特征数
        out_features = out_features or in_features
        # 确定隐藏特征数，若未指定则使用输入特征数
        hidden_features = hidden_features or in_features
        # 定义第一层全连接层
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 定义激活层
        self.act = act_layer()
        # 定义第二层全连接层
        self.fc2 = nn.Linear(hidden_features, out_features)
        # 定义丢弃层
        self.drop = nn.Dropout(drop)

    # 前向传播函数
    def forward(self, x):
        # 通过第一层全连接
        x = self.fc1(x)
        # 通过激活函数
        x = self.act(x)
        # x = self.drop(x)
        # 保持原始 BERT 实现
        x = self.fc2(x)
        # 应用丢弃层
        x = self.drop(x)
        # 返回最终结果
        return x


# 定义 Attention 类，构建注意力机制
class Attention(nn.Module):
    # 初始化类的构造函数，设置多种参数，包括维度、头数等
        def __init__(
                self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                proj_drop=0., window_size=None, attn_head_dim=None):
            # 调用父类的构造函数
            super().__init__()
            # 设置注意力头的数量
            self.num_heads = num_heads
            # 计算每个头的维度
            head_dim = dim // num_heads
            # 如果指定了注意力头维度，则使用指定值
            if attn_head_dim is not None:
                head_dim = attn_head_dim
            # 计算所有头的总维度
            all_head_dim = head_dim * self.num_heads
            # 设置缩放因子，默认是head_dim的负0.5次幂
            self.scale = qk_scale or head_dim ** -0.5
    
            # 定义一个线性层，用于生成查询、键、值
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
            # 如果使用偏置，则初始化查询和值的偏置参数
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
                self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
            else:
                # 否则偏置设为None
                self.q_bias = None
                self.v_bias = None
    
            # 如果窗口大小被定义
            if window_size:
                # 设置窗口大小
                self.window_size = window_size
                # 计算相对距离的数量
                self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
                # 初始化相对位置偏置表
                self.relative_position_bias_table = nn.Parameter(
                    torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
                # 获取每个token的相对位置索引
    
                # 创建窗口内每个token的坐标
                coords_h = torch.arange(window_size[0])
                coords_w = torch.arange(window_size[1])
                # 使用网格生成坐标
                coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
                # 展平坐标
                coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
                # 计算相对坐标
                relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
                # 调整维度顺序以便后续处理
                relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
                # 调整坐标以从0开始
                relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
                relative_coords[:, :, 1] += window_size[1] - 1
                # 计算相对位置索引
                relative_coords[:, :, 0] *= 2 * window_size[1] - 1
                relative_position_index = \
                    torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
                # 设置相对位置索引的值
                relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
                relative_position_index[0, 0:] = self.num_relative_distance - 3
                relative_position_index[0:, 0] = self.num_relative_distance - 2
                relative_position_index[0, 0] = self.num_relative_distance - 1
    
                # 注册相对位置索引为缓冲区
                self.register_buffer("relative_position_index", relative_position_index)
    
                # 初始化相对位置偏置表的正态分布（注释掉的代码）
            else:
                # 如果没有窗口大小，则相关属性设为None
                self.window_size = None
                self.relative_position_bias_table = None
                self.relative_position_index = None
    
            # 定义注意力丢弃层
            self.attn_drop = nn.Dropout(attn_drop)
            # 定义输出的线性层
            self.proj = nn.Linear(all_head_dim, dim)
            # 定义投影的丢弃层
            self.proj_drop = nn.Dropout(proj_drop)
# 定义一个名为 Block 的类，继承自 nn.Module
class Block(nn.Module):

    # 初始化方法，定义 Block 的属性和参数
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        # 调用父类初始化方法
        super().__init__()
        # 创建归一化层，输入维度为 dim
        self.norm1 = norm_layer(dim)
        # 创建注意力层，设置相关参数
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        # 根据给定的 drop_path 值选择相应的 DropPath 或 Identity 层
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 创建第二个归一化层
        self.norm2 = norm_layer(dim)
        # 根据 mlp_ratio 计算 MLP 隐藏层的维度
        mlp_hidden_dim = int(dim * mlp_ratio)
        # 创建 MLP 层，设置相关参数
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 如果 init_values 不为 None，则创建可训练的参数 gamma_1 和 gamma_2
        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            # 否则将 gamma_1 和 gamma_2 设置为 None
            self.gamma_1, self.gamma_2 = None, None

    # 前向传播方法
    def forward(self, x, rel_pos_bias=None, training_window_size=None):
        # 如果 gamma_1 为 None，则执行普通的前向计算
        if self.gamma_1 is None:
            x = x + self.drop_path(
                self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, training_window_size=training_window_size))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            # 否则，使用 gamma_1 和 gamma_2 加权计算
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias,
                                                            training_window_size=training_window_size))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        # 返回处理后的输出
        return x


# 定义一个名为 PatchEmbed 的类，继承自 nn.Module
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    # 初始化方法，定义 PatchEmbed 的属性和参数
    def __init__(self, img_size=[224, 224], patch_size=16, in_chans=3, embed_dim=768):
        # 调用父类初始化方法
        super().__init__()
        # 将 img_size 转换为二元组格式
        img_size = to_2tuple(img_size)
        # 将 patch_size 转换为二元组格式
        patch_size = to_2tuple(patch_size)
        # 计算总的 patch 数量
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        # 计算每个维度的 patch 形状
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # 保存宽度方向的 patch 数量
        self.num_patches_w = self.patch_shape[0]
        # 保存高度方向的 patch 数量
        self.num_patches_h = self.patch_shape[1]
        # 保存原始图像尺寸
        self.img_size = img_size
        # 保存 patch 尺寸
        self.patch_size = patch_size
        # 保存总的 patch 数量
        self.num_patches = num_patches

        # 创建卷积层，将输入通道映射到嵌入维度
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    # 前向传播方法，接收输入 x 和可选的位置嵌入
    def forward(self, x, position_embedding=None, **kwargs):
        # FIXME 提醒检查放宽尺寸约束
        # 确保输入图像的高和宽与模型预期的尺寸匹配
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # 对输入 x 应用投影变换
        x = self.proj(x)
        # 获取变换后图像的高和宽
        Hp, Wp = x.shape[2], x.shape[3]
    
        # 如果提供了位置嵌入，则进行处理
        if position_embedding is not None:
            # 将位置嵌入调整为对应的尺寸
            position_embedding = position_embedding.view(1, self.patch_shape[0], self.patch_shape[1], -1).permute(0, 3,
                                                                                                                      1, 2)
            # 使用双三次插值调整位置嵌入的大小
            position_embedding = F.interpolate(position_embedding, size=(Hp, Wp), mode='bicubic')
            # 将位置嵌入加到输入 x 上
            x = x + position_embedding
    
        # 将 x 展平为二维，并转置维度
        x = x.flatten(2).transpose(1, 2)
        # 返回处理后的 x 以及其高和宽
        return x, (Hp, Wp)
# 定义混合嵌入类，继承自 nn.Module
class HybridEmbed(nn.Module):
    """ CNN 特征图嵌入
    从 CNN 中提取特征图，展平并投影到嵌入维度。
    """

    # 初始化函数，定义网络结构和参数
    def __init__(self, backbone, img_size=[224, 224], feature_size=None, in_chans=3, embed_dim=768):
        # 调用父类构造函数
        super().__init__()
        # 确保 backbone 是 nn.Module 的实例
        assert isinstance(backbone, nn.Module)
        # 将图像尺寸转换为二元组
        img_size = to_2tuple(img_size)
        # 保存图像尺寸
        self.img_size = img_size
        # 保存 backbone 网络
        self.backbone = backbone
        # 如果没有提供特征尺寸
        if feature_size is None:
            with torch.no_grad():
                # 使用零张量来推断输出特征图的尺寸
                # FIXME 这种方法虽然不够优雅，但能最可靠地确定输出特征图的精确维度
                training = backbone.training
                # 如果 backbone 正在训练，切换到评估模式
                if training:
                    backbone.eval()
                # 通过 backbone 网络计算输出特征图
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                # 获取特征图的尺寸
                feature_size = o.shape[-2:]
                # 获取特征图的通道数
                feature_dim = o.shape[1]
                # 恢复 backbone 到训练模式
                backbone.train(training)
        else:
            # 如果提供了特征尺寸，将其转换为二元组
            feature_size = to_2tuple(feature_size)
            # 从 backbone 获取特征通道数
            feature_dim = self.backbone.feature_info.channels()[-1]
        # 计算补丁的数量
        self.num_patches = feature_size[0] * feature_size[1]
        # 定义线性投影层，将特征维度投影到嵌入维度
        self.proj = nn.Linear(feature_dim, embed_dim)

    # 前向传播函数
    def forward(self, x):
        # 通过 backbone 获取特征图
        x = self.backbone(x)[-1]
        # 展平特征图并转置维度
        x = x.flatten(2).transpose(1, 2)
        # 通过线性投影层进行投影
        x = self.proj(x)
        # 返回投影后的结果
        return x


# 定义相对位置偏差类，继承自 nn.Module
class RelativePositionBias(nn.Module):
    # 初始化方法，接收窗口大小和头数作为参数
    def __init__(self, window_size, num_heads):
        # 调用父类的初始化方法
        super().__init__()
        # 保存窗口大小
        self.window_size = window_size
        # 保存头的数量
        self.num_heads = num_heads
        # 计算相对距离的数量
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        # 创建一个可学习的参数，用于存储相对位置偏置，初始化为零
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # 获取窗口内每个 token 的成对相对位置索引
        coords_h = torch.arange(window_size[0])  # 创建高度坐标
        coords_w = torch.arange(window_size[1])  # 创建宽度坐标
        # 生成网格坐标，并堆叠在一起
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # 将坐标展平，得到每个坐标的线性表示
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # 计算相对坐标，得到每对 token 之间的相对位置
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        # 调整维度顺序，方便后续计算
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # 将相对坐标的第一个维度偏移，以便从 0 开始
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        # 计算相对位置索引
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        # 初始化相对位置索引张量
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        # 计算相对位置索引，填充在相应位置
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # 设置左上角的相对位置索引
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        # 将相对位置索引注册到模型缓冲区中
        self.register_buffer("relative_position_index", relative_position_index)

        # 使用截断正态分布初始化相对位置偏置表，标准差为 0.02
        # trunc_normal_(self.relative_position_bias_table, std=.02)
# 定义一个名为 BEiT 的类，继承自 nn.Module，用于实现视觉变换器
class BEiT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    # 定义一个方法 fix_init_weight，用于修正初始化权重
    def fix_init_weight(self):
        # 定义内部函数 rescale，调整参数的尺度
        def rescale(param, layer_id):
            # 根据层ID调整参数的值，防止梯度消失或爆炸
            param.div_(math.sqrt(2.0 * layer_id))

        # 遍历网络中的每一层及其索引
        for layer_id, layer in enumerate(self.blocks):
            # 调整注意力层的投影权重
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            # 调整全连接层的第二层权重
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    # 定义初始化权重的方法
    def _init_weights(self, m):
        # 检查参数是否为线性层
        if isinstance(m, nn.Linear):
            # 用截断正态分布初始化权重，标准差为0.02
            trunc_normal_(m.weight, std=.02)
            # 如果是线性层且有偏置，则将偏置初始化为0
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # 检查参数是否为层归一化
        elif isinstance(m, nn.LayerNorm):
            # 将偏置初始化为0
            nn.init.constant_(m.bias, 0)
            # 将权重初始化为1.0
            nn.init.constant_(m.weight, 1.0)

    '''
    # 定义初始化权重的函数 init_weights
    def init_weights(self):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        # 获取根日志记录器
        logger = get_root_logger()

        # 如果位置嵌入存在，则用截断正态分布初始化
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        # 用截断正态分布初始化类标记
        trunc_normal_(self.cls_token, std=.02)
        # 应用初始化权重的方法
        self.apply(self._init_weights)
        # 修正初始化权重
        self.fix_init_weight()

        # 如果初始化配置为 None，发出警告
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
        else:
            # 检查初始化配置中是否包含 'checkpoint'
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            # 日志记录将从指定的检查点加载
            logger.info(f"Will load ckpt from {self.init_cfg['checkpoint']}")
            # 从检查点加载权重
            load_checkpoint(self,
                            filename=self.init_cfg['checkpoint'],
                            strict=False,
                            logger=logger,
                            beit_spec_expand_rel_pos = self.use_rel_pos_bias,
                            )
    '''

    # 定义一个方法 get_num_layers，返回网络中层的数量
    def get_num_layers(self):
        return len(self.blocks)

    # 使用 @torch.jit.ignore 装饰器，表明该方法在 JIT 编译时应被忽略
    @torch.jit.ignore
    # 定义一个不需要权重衰减的方法
    def no_weight_decay(self):
        # 返回不需要权重衰减的参数集
        return {'pos_embed', 'cls_token'}
    # 定义前向特征提取的方法，输入为 x
    def forward_features(self, x):
        # 获取输入张量的形状，B: 批量大小, C: 通道数, H: 高度, W: 宽度
        B, C, H, W = x.shape
        # 通过 patch 嵌入处理输入 x，得到嵌入后的张量和补丁的高度与宽度
        x, (Hp, Wp) = self.patch_embed(x, self.pos_embed[:, 1:, :] if self.pos_embed is not None else None)
        # Hp, Wp 是补丁的高和宽
        # 获取批量大小和序列长度
        batch_size, seq_len, _ = x.size()

        # 扩展 cls_token，以适应批量大小
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # 从 Phil Wang 的实现中借用 cls_tokens，感谢
        # 如果存在位置嵌入，则将其加到 cls_tokens 上
        if self.pos_embed is not None:
            cls_tokens = cls_tokens + self.pos_embed[:, :1, :]
        # 将 cls_tokens 与 x 在维度 1 上拼接
        x = torch.cat((cls_tokens, x), dim=1)
        # 对 x 进行位置丢弃操作
        x = self.pos_drop(x)

        # 初始化特征列表
        features = []
        # 将 Hp 和 Wp 转换为张量
        training_window_size = torch.tensor([Hp, Wp])

        # 计算相对位置偏置，如果存在相对位置偏置则计算
        rel_pos_bias = self.rel_pos_bias(training_window_size) if self.rel_pos_bias is not None else None

        # 遍历每个块
        for i, blk in enumerate(self.blocks):
            # 如果使用检查点机制，则进行检查点计算
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, rel_pos_bias, training_window_size)
            else:
                # 否则直接进行块的前向计算
                x = blk(x, rel_pos_bias=rel_pos_bias, training_window_size=training_window_size)
            # 如果当前索引在输出索引中，则处理特征
            if i in self.out_indices:
                # 从 x 中提取特征，重新排列和调整形状
                xp = x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                # 将处理后的特征添加到特征列表中
                features.append(xp.contiguous())

        # 定义操作列表
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        # 遍历每个特征并应用相应的操作
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        # 初始化输出特征字典
        feat_out = {}

        # 将特征按名称存入输出字典
        for name, value in zip(self.out_features, features):
            feat_out[name] = value

        # 返回输出特征字典
        return feat_out

    # 定义前向计算的方法，输入为 x
    def forward(self, x):
        # 调用 forward_features 方法进行特征提取
        x = self.forward_features(x)
        # 返回提取的特征
        return x
# 定义一个函数，用于创建一个 BEiT 模型，默认不使用预训练权重
def beit_base_patch16(pretrained=False, **kwargs):
    # 创建一个 BEiT 模型实例，指定各个超参数
    model = BEiT(
        patch_size=16,         # 设置补丁大小为16
        embed_dim=768,        # 设置嵌入维度为768
        depth=12,             # 设置网络深度为12层
        num_heads=12,         # 设置多头注意力机制的头数为12
        mlp_ratio=4,          # 设置 MLP 的比例为4
        qkv_bias=True,        # 启用 QKV 的偏置
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  # 使用带小 epsilon 的 LayerNorm
        init_values=None,     # 初始化值设为 None
        **kwargs)             # 允许传递其他参数
    # 设置模型的默认配置
    model.default_cfg = _cfg()
    # 返回创建的模型
    return model

# 定义一个函数，用于创建一个大型的 BEiT 模型，默认不使用预训练权重
def beit_large_patch16(pretrained=False, **kwargs):
    # 创建一个 BEiT 模型实例，指定各个超参数
    model = BEiT(
        patch_size=16,         # 设置补丁大小为16
        embed_dim=1024,       # 设置嵌入维度为1024
        depth=24,             # 设置网络深度为24层
        num_heads=16,         # 设置多头注意力机制的头数为16
        mlp_ratio=4,          # 设置 MLP 的比例为4
        qkv_bias=True,        # 启用 QKV 的偏置
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  # 使用带小 epsilon 的 LayerNorm
        init_values=None,     # 初始化值设为 None
        **kwargs)             # 允许传递其他参数
    # 设置模型的默认配置
    model.default_cfg = _cfg()
    # 返回创建的模型
    return model

# 定义一个函数，用于创建一个基础的 DIT 模型，默认不使用预训练权重
def dit_base_patch16(pretrained=False, **kwargs):
    # 创建一个 BEiT 模型实例，指定各个超参数
    model = BEiT(
        patch_size=16,         # 设置补丁大小为16
        embed_dim=768,        # 设置嵌入维度为768
        depth=12,             # 设置网络深度为12层
        num_heads=12,         # 设置多头注意力机制的头数为12
        mlp_ratio=4,          # 设置 MLP 的比例为4
        qkv_bias=True,        # 启用 QKV 的偏置
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  # 使用带小 epsilon 的 LayerNorm
        init_values=0.1,      # 初始化值设为0.1
        **kwargs)             # 允许传递其他参数
    # 设置模型的默认配置
    model.default_cfg = _cfg()
    # 返回创建的模型
    return model

# 定义一个函数，用于创建一个大型的 DIT 模型，默认不使用预训练权重
def dit_large_patch16(pretrained=False, **kwargs):
    # 创建一个 BEiT 模型实例，指定各个超参数
    model = BEiT(
        patch_size=16,         # 设置补丁大小为16
        embed_dim=1024,       # 设置嵌入维度为1024
        depth=24,             # 设置网络深度为24层
        num_heads=16,         # 设置多头注意力机制的头数为16
        mlp_ratio=4,          # 设置 MLP 的比例为4
        qkv_bias=True,        # 启用 QKV 的偏置
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  # 使用带小 epsilon 的 LayerNorm
        init_values=1e-5,     # 初始化值设为1e-5
        **kwargs)             # 允许传递其他参数
    # 设置模型的默认配置
    model.default_cfg = _cfg()
    # 返回创建的模型
    return model

# 当脚本作为主程序运行时执行以下代码
if __name__ == '__main__':
    # 创建一个 BEiT 模型实例，启用检查点和共享相对位置偏置
    model = BEiT(use_checkpoint=True, use_shared_rel_pos_bias=True)
    # 将模型移动到 GPU 的第一个设备
    model = model.to("cuda:0")
    # 创建一个随机输入张量，大小为 (2, 3, 512, 762)，并移动到 GPU
    input1 = torch.rand(2, 3, 512, 762).to("cuda:0")
    # 创建另一个随机输入张量，大小为 (2, 3, 800, 1200)，并移动到 GPU
    input2 = torch.rand(2, 3, 800, 1200).to("cuda:0")
    # 创建第三个随机输入张量，大小为 (2, 3, 720, 1000)，并移动到 GPU
    input3 = torch.rand(2, 3, 720, 1000).to("cuda:0")
    # 将第一个输入张量传入模型，获取输出
    output1 = model(input1)
    # 将第二个输入张量传入模型，获取输出
    output2 = model(input2)
    # 将第三个输入张量传入模型，获取输出
    output3 = model(input3)
    # 打印完成消息
    print("all done")
```