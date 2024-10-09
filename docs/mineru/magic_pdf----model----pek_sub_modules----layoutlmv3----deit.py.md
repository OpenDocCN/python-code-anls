# `.\MinerU\magic_pdf\model\pek_sub_modules\layoutlmv3\deit.py`

```
"""
主要复制自 DINO 和 timm 库：
https://github.com/facebookresearch/dino
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
# 导入警告模块，处理警告信息
import warnings

# 导入数学模块
import math
# 导入 PyTorch 库
import torch
# 导入神经网络模块
import torch.nn as nn
# 导入检查点模块，支持内存优化
import torch.utils.checkpoint as checkpoint
# 从 timm 库导入特定的模型层功能
from timm.models.layers import trunc_normal_, drop_path, to_2tuple
# 导入部分功能，便于使用
from functools import partial

# 定义一个配置函数，返回包含模型配置的字典
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

# 定义 DropPath 类，执行随机深度的模块
class DropPath(nn.Module):
    """每个样本的 DropPath（随机深度），应用于残差块的主路径。
    """

    # 初始化 DropPath 模块
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        # 存储丢弃概率
        self.drop_prob = drop_prob

    # 定义前向传播方法
    def forward(self, x):
        # 应用 drop_path 函数，返回处理后的输出
        return drop_path(x, self.drop_prob, self.training)

    # 定义额外的字符串表示
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


# 定义 Mlp 类，构建多层感知机
class Mlp(nn.Module):
    # 初始化 Mlp 模块
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # 如果未提供输出特征，则设置为输入特征
        out_features = out_features or in_features
        # 如果未提供隐藏特征，则设置为输入特征
        hidden_features = hidden_features or in_features
        # 定义第一层线性变换
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 定义激活层
        self.act = act_layer()
        # 定义第二层线性变换
        self.fc2 = nn.Linear(hidden_features, out_features)
        # 定义丢弃层
        self.drop = nn.Dropout(drop)

    # 定义前向传播方法
    def forward(self, x):
        # 通过第一层线性变换
        x = self.fc1(x)
        # 应用激活函数
        x = self.act(x)
        # 应用丢弃
        x = self.drop(x)
        # 通过第二层线性变换
        x = self.fc2(x)
        # 再次应用丢弃
        x = self.drop(x)
        # 返回最终输出
        return x


# 定义 Attention 类，实现自注意力机制
class Attention(nn.Module):
    # 初始化 Attention 模块
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        # 存储头数
        self.num_heads = num_heads
        # 计算每个头的维度
        head_dim = dim // num_heads
        # 如果未提供缩放因子，则使用默认计算方式
        self.scale = qk_scale or head_dim ** -0.5

        # 定义 QKV 线性变换
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 定义注意力丢弃层
        self.attn_drop = nn.Dropout(attn_drop)
        # 定义输出线性变换
        self.proj = nn.Linear(dim, dim)
        # 定义投影丢弃层
        self.proj_drop = nn.Dropout(proj_drop)

    # 定义前向传播方法
    def forward(self, x):
        # 获取输入的批次大小、序列长度和特征维度
        B, N, C = x.shape
        # 计算 Q、K、V，重塑形状并进行转置
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                      C // self.num_heads).permute(2, 0, 3, 1, 4)

        # 计算注意力分数并应用缩放
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # 计算 softmax 得到注意力权重
        attn = attn.softmax(dim=-1)
        # 应用注意力丢弃
        attn = self.attn_drop(attn)

        # 计算加权和并重塑形状
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 通过输出线性变换
        x = self.proj(x)
        # 应用投影丢弃
        x = self.proj_drop(x)
        # 返回最终输出
        return x


# 定义 Block 类，作为神经网络的基本构建块
class Block(nn.Module):
    # 初始化函数，设置类的基本参数
        def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                     drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
            # 调用父类初始化方法
            super().__init__()
            # 使用归一化层对输入维度进行归一化处理
            self.norm1 = norm_layer(dim)
            # 创建注意力机制实例，设置相关参数
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            # 设定随机深度的丢弃路径，如果 drop_path 大于 0，则使用 DropPath，否则使用身份映射
            self.drop_path = DropPath(
                drop_path) if drop_path > 0. else nn.Identity()
            # 第二次归一化层
            self.norm2 = norm_layer(dim)
            # 计算 MLP 隐藏层的维度
            mlp_hidden_dim = int(dim * mlp_ratio)
            # 创建 MLP 实例，设置输入特征和隐藏特征
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                           act_layer=act_layer, drop=drop)
    
        # 前向传播函数
        def forward(self, x):
            # 将输入通过归一化层、注意力机制和丢弃路径，进行第一步处理
            x = x + self.drop_path(self.attn(self.norm1(x)))
            # 将处理后的结果通过第二次归一化层和 MLP 进行处理
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            # 返回最终的输出结果
            return x
class PatchEmbed(nn.Module):
    """ 图像到补丁嵌入的类
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        # 初始化 PatchEmbed 类，设置图像大小、补丁大小、输入通道和嵌入维度
        super().__init__()
        # 将图像大小转换为二元组格式
        img_size = to_2tuple(img_size)
        # 将补丁大小转换为二元组格式
        patch_size = to_2tuple(patch_size)

        # 计算窗口大小，即每个维度上的补丁数量
        self.window_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        # 解包窗口大小为宽和高的补丁数量
        self.num_patches_w, self.num_patches_h = self.window_size

        # 计算总补丁数量
        self.num_patches = self.window_size[0] * self.window_size[1]
        # 保存图像大小
        self.img_size = img_size
        # 保存补丁大小
        self.patch_size = patch_size

        # 定义卷积层，将输入通道转换为嵌入维度，使用补丁大小作为卷积核和步幅
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # 将输入通过卷积层进行投影
        x = self.proj(x)
        # 返回投影后的结果
        return x


class HybridEmbed(nn.Module):
    """ CNN 特征图嵌入类
    从 CNN 提取特征图，展平并投影到嵌入维度。
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        # 初始化 HybridEmbed 类，设置主干网络、图像大小、特征大小、输入通道和嵌入维度
        super().__init__()
        # 确保主干网络是 nn.Module 的实例
        assert isinstance(backbone, nn.Module)
        # 将图像大小转换为二元组格式
        img_size = to_2tuple(img_size)
        # 保存图像大小
        self.img_size = img_size
        # 保存主干网络
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME 这种方式有些 hacky，但在确定输出特征图的精确维度时是最可靠的方法
                # 特征元数据具有可靠的通道和步幅信息，但使用步幅计算特征维度需要关于每个阶段填充的信息
                training = backbone.training
                if training:
                    backbone.eval()  # 将主干网络设置为评估模式
                # 通过主干网络传递一个零张量以获取输出特征图
                o = self.backbone(torch.zeros(
                    1, in_chans, img_size[0], img_size[1]))[-1]
                # 获取输出特征图的形状作为特征大小
                feature_size = o.shape[-2:]
                # 获取特征图的通道数作为特征维度
                feature_dim = o.shape[1]
                backbone.train(training)  # 恢复主干网络的训练模式
        else:
            # 如果提供特征大小，将其转换为二元组格式
            feature_size = to_2tuple(feature_size)
            # 从主干网络获取特征维度
            feature_dim = self.backbone.feature_info.channels()[-1]
        # 计算总补丁数量
        self.num_patches = feature_size[0] * feature_size[1]
        # 定义线性层，将特征维度映射到嵌入维度
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        # 通过主干网络获取最后的特征图
        x = self.backbone(x)[-1]
        # 将特征图展平，并转置维度以适应后续处理
        x = x.flatten(2).transpose(1, 2)
        # 将展平的特征图通过线性层进行投影
        x = self.proj(x)
        # 返回投影后的结果
        return x


class ViT(nn.Module):
    """ 支持补丁或混合 CNN 输入阶段的视觉变换器类
    """

    def fix_init_weight(self):
        # 定义重新缩放权重的内部函数
        def rescale(param, layer_id):
            # 将参数除以层ID的平方根以进行缩放
            param.div_(math.sqrt(2.0 * layer_id))

        # 遍历每个层以重新缩放其权重
        for layer_id, layer in enumerate(self.blocks):
            # 对注意力投影权重进行缩放
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            # 对多层感知器的第二层权重进行缩放
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)
    # 初始化权重的私有方法
    def _init_weights(self, m):
        # 如果 m 是线性层
        if isinstance(m, nn.Linear):
            # 用截断正态分布初始化权重，标准差为 0.02
            trunc_normal_(m.weight, std=.02)
            # 如果 m 是线性层且有偏置项
            if isinstance(m, nn.Linear) and m.bias is not None:
                # 将偏置初始化为 0
                nn.init.constant_(m.bias, 0)
        # 如果 m 是层归一化层
        elif isinstance(m, nn.LayerNorm):
            # 将偏置初始化为 0
            nn.init.constant_(m.bias, 0)
            # 将权重初始化为 1.0
            nn.init.constant_(m.weight, 1.0)

    '''
    # 初始化权重的方法
    def init_weights(self):
        # 获取根日志记录器
        logger = get_root_logger()

        # 用截断正态分布初始化位置嵌入，标准差为 0.02
        trunc_normal_(self.pos_embed, std=.02)
        # 用截断正态分布初始化类标记，标准差为 0.02
        trunc_normal_(self.cls_token, std=.02)
        # 应用初始化权重的方法到所有模块
        self.apply(self._init_weights)

        # 如果没有初始化配置
        if self.init_cfg is None:
            # 记录警告信息，表示没有预训练权重，从头开始训练
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
        else:
            # 断言初始化配置中包含检查点信息
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            # 记录加载检查点的信息
            logger.info(f"Will load ckpt from {self.init_cfg['checkpoint']}")
            # 从指定的检查点加载权重，严格模式为 False，记录日志
            load_checkpoint(self, filename=self.init_cfg['checkpoint'], strict=False, logger=logger)
    '''

    # 获取模型层数的方法
    def get_num_layers(self):
        # 返回块的数量
        return len(self.blocks)

    # 不参与权重衰减的方法，忽略 JIT 编译
    @torch.jit.ignore
    def no_weight_decay(self):
        # 返回不需要权重衰减的参数名称
        return {'pos_embed', 'cls_token'}

    # 过滤状态字典的方法，将手动补丁嵌入权重转换为卷积
    def _conv_filter(self, state_dict, patch_size=16):
        """ convert patch embedding weight from manual patchify + linear proj to conv"""
        # 创建一个空字典以存储输出
        out_dict = {}
        # 遍历状态字典的每个键值对
        for k, v in state_dict.items():
            # 如果键包含 'patch_embed.proj.weight'
            if 'patch_embed.proj.weight' in k:
                # 将权重重塑为 (batch_size, 3, patch_size, patch_size)
                v = v.reshape((v.shape[0], 3, patch_size, patch_size))
            # 将处理后的权重加入输出字典
            out_dict[k] = v
        # 返回转换后的权重字典
        return out_dict

    # 将一维数据转换为二维的函数
    def to_2D(self, x):
        # 获取输入的形状
        n, hw, c = x.shape
        # 计算高度和宽度
        h = w = int(math.sqrt(hw))
        # 转置并重塑为 (batch_size, channels, height, width)
        x = x.transpose(1, 2).reshape(n, c, h, w)
        # 返回转换后的二维数据
        return x

    # 将二维数据转换为一维的函数
    def to_1D(self, x):
        # 获取输入的形状
        n, c, h, w = x.shape
        # 重塑为 (batch_size, channels, -1) 并转置
        x = x.reshape(n, c, -1).transpose(1, 2)
        # 返回转换后的数据
        return x
    # 插值位置编码，根据输入张量 x 的大小和宽高 w, h 生成位置编码
    def interpolate_pos_encoding(self, x, w, h):
        # 计算有效的补丁数量
        npatch = x.shape[1] - self.num_extra_tokens
        # 计算位置编码中的有效补丁数量
        N = self.pos_embed.shape[1] - self.num_extra_tokens
        # 如果补丁数量相等且宽高相同，直接返回已有位置编码
        if npatch == N and w == h:
            return self.pos_embed

        # 获取类或距离的位置信息
        class_ORdist_pos_embed = self.pos_embed[:, 0:self.num_extra_tokens]

        # 获取补丁的位置信息
        patch_pos_embed = self.pos_embed[:, self.num_extra_tokens:]

        # 获取输入特征的维度
        dim = x.shape[-1]
        # 计算宽度和高度的补丁数量
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[1]
        # 为避免插值中的浮点数误差，给宽高加一个小数
        # 参考讨论: https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        # 使用双线性插值调整补丁的位置编码
        patch_pos_embed = nn.functional.interpolate(
            # 先调整形状以适应插值函数
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            # 设置缩放因子
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            # 使用双线性插值模式
            mode='bicubic',
        )
        # 断言插值后的宽高与预期一致
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        # 调整补丁位置编码的维度
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        # 将类或距离位置编码与补丁位置编码连接在一起
        return torch.cat((class_ORdist_pos_embed, patch_pos_embed), dim=1)

    # 准备输入令牌，包括补丁嵌入和掩码处理
    def prepare_tokens(self, x, mask=None):
        # 获取输入的批量大小、通道数、宽和高
        B, nc, w, h = x.shape
        # 对输入进行补丁线性嵌入
        x = self.patch_embed(x)

        # 进行掩码图像建模（如果有掩码）
        if mask is not None:
            x = self.mask_model(x, mask)
        # 将张量展平并转置
        x = x.flatten(2).transpose(1, 2)

        # 添加 [CLS] 令牌到补丁令牌中
        all_tokens = [self.cls_token.expand(B, -1, -1)]

        # 如果有额外的两个令牌，则添加距离令牌
        if self.num_extra_tokens == 2:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            all_tokens.append(dist_tokens)
        # 添加补丁令牌
        all_tokens.append(x)

        # 将所有令牌在维度 1 上连接
        x = torch.cat(all_tokens, dim=1)

        # 为每个令牌添加位置编码
        x = x + self.interpolate_pos_encoding(x, w, h)

        # 通过位置丢弃层返回处理后的令牌
        return self.pos_drop(x)

    # 向前传播特征
    def forward_features(self, x):
        # 打印输入张量的形状（调试信息）
        # print(f"==========shape of x is {x.shape}==========")
        # 获取批量大小、通道数、高和宽
        B, _, H, W = x.shape
        # 计算补丁的高和宽
        Hp, Wp = H // self.patch_size, W // self.patch_size
        # 准备输入令牌
        x = self.prepare_tokens(x)

        # 初始化特征列表
        features = []
        # 遍历每个块
        for i, blk in enumerate(self.blocks):
            # 如果使用检查点，则进行检查点操作
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                # 直接通过块处理输入
                x = blk(x)
            # 如果索引在输出索引中，则处理特征
            if i in self.out_indices:
                # 调整特征的维度并存储
                xp = x[:, self.num_extra_tokens:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                features.append(xp.contiguous())

        # 应用特征金字塔网络层
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            # 在每个特征上应用对应的操作
            features[i] = ops[i](features[i])

        # 初始化输出特征字典
        feat_out = {}

        # 将特征名称与特征值对应存入字典
        for name, value in zip(self.out_features, features):
            feat_out[name] = value

        # 返回特征输出
        return feat_out

    # 前向传播方法
    def forward(self, x):
        # 调用前向特征传播方法处理输入
        x = self.forward_features(x)
        # 返回最终输出
        return x
# 定义一个函数，创建 Deit 模型，支持加载预训练权重
def deit_base_patch16(pretrained=False, **kwargs):
    # 初始化 ViT 模型，设定不同的超参数
    model = ViT(
        # 每个 patch 的大小为 16
        patch_size=16,
        # 不使用 dropout
        drop_rate=0.,
        # 嵌入维度为 768
        embed_dim=768,
        # 模型深度为 12
        depth=12,
        # 头的数量为 12
        num_heads=12,
        # 输出类的数量为 1000
        num_classes=1000,
        # MLP 的比率为 4
        mlp_ratio=4.,
        # 使用 QKV 偏置
        qkv_bias=True,
        # 启用检查点功能以节省内存
        use_checkpoint=True,
        # 额外 token 的数量为 2
        num_extra_tokens=2,
        **kwargs)  # 允许其他参数通过
    # 设置模型的默认配置
    model.default_cfg = _cfg()
    # 返回创建的模型
    return model

# 定义一个函数，创建 MAE 模型，支持加载预训练权重
def mae_base_patch16(pretrained=False, **kwargs):
    # 初始化 ViT 模型，设定不同的超参数
    model = ViT(
        # 每个 patch 的大小为 16
        patch_size=16,
        # 不使用 dropout
        drop_rate=0.,
        # 嵌入维度为 768
        embed_dim=768,
        # 模型深度为 12
        depth=12,
        # 头的数量为 12
        num_heads=12,
        # 输出类的数量为 1000
        num_classes=1000,
        # MLP 的比率为 4
        mlp_ratio=4.,
        # 使用 QKV 偏置
        qkv_bias=True,
        # 启用检查点功能以节省内存
        use_checkpoint=True,
        # 额外 token 的数量为 1
        num_extra_tokens=1,
        **kwargs)  # 允许其他参数通过
    # 设置模型的默认配置
    model.default_cfg = _cfg()
    # 返回创建的模型
    return model
```