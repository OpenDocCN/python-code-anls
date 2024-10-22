# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\vqvae\movq_modules.py`

```py
# pytorch_diffusion + derived encoder decoder
# 导入数学库
import math
# 导入 PyTorch 库
import torch
# 导入 PyTorch 神经网络模块
import torch.nn as nn
# 导入 NumPy 库
import numpy as np


def get_timestep_embedding(timesteps, embedding_dim):
    """
    该函数实现了去噪扩散概率模型中的时间步嵌入构建。
    来自 Fairseq。
    构建正弦嵌入。
    该实现与 tensor2tensor 中的实现匹配，但与“Attention Is All You Need”第 3.5 节中的描述略有不同。
    """
    # 确保 timesteps 是一维张量
    assert len(timesteps.shape) == 1

    # 计算嵌入维度的一半
    half_dim = embedding_dim // 2
    # 计算嵌入的指数缩放因子
    emb = math.log(10000) / (half_dim - 1)
    # 创建半维度的指数衰减张量
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    # 将嵌入移动到与 timesteps 相同的设备上
    emb = emb.to(device=timesteps.device)
    # 计算时间步嵌入，使用广播机制
    emb = timesteps.float()[:, None] * emb[None, :]
    # 将正弦和余弦嵌入连接在一起
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    # 如果嵌入维度是奇数，则进行零填充
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    # 返回生成的嵌入
    return emb


def nonlinearity(x):
    # 定义非线性激活函数：swish
    return x * torch.sigmoid(x)


class SpatialNorm(nn.Module):
    # 定义空间归一化模块
    def __init__(
        self,
        f_channels,
        zq_channels,
        norm_layer=nn.GroupNorm,
        freeze_norm_layer=False,
        add_conv=False,
        **norm_layer_params,
    ):
        # 初始化父类
        super().__init__()
        # 创建归一化层
        self.norm_layer = norm_layer(num_channels=f_channels, **norm_layer_params)
        # 如果需要冻结归一化层的参数
        if freeze_norm_layer:
            # 将归一化层的所有参数设置为不需要梯度
            for p in self.norm_layer.parameters:
                p.requires_grad = False
        # 是否添加卷积层
        self.add_conv = add_conv
        # 如果添加卷积层，定义卷积层
        if self.add_conv:
            self.conv = nn.Conv2d(zq_channels, zq_channels, kernel_size=3, stride=1, padding=1)
        # 定义用于处理 zq 的卷积层
        self.conv_y = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        # 定义另一个用于处理 zq 的卷积层
        self.conv_b = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, f, zq):
        # 获取 f 的空间尺寸
        f_size = f.shape[-2:]
        # 将 zq 进行上采样以匹配 f 的尺寸
        zq = torch.nn.functional.interpolate(zq, size=f_size, mode="nearest")
        # 如果需要添加卷积层，则对 zq 进行卷积处理
        if self.add_conv:
            zq = self.conv(zq)
        # 对 f 应用归一化层
        norm_f = self.norm_layer(f)
        # 计算新的 f，结合归一化后的 f 和 zq 的卷积结果
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        # 返回新的 f
        return new_f


def Normalize(in_channels, zq_ch, add_conv):
    # 创建并返回一个 SpatialNorm 实例
    return SpatialNorm(
        in_channels,
        zq_ch,
        norm_layer=nn.GroupNorm,
        freeze_norm_layer=False,
        add_conv=add_conv,
        num_groups=32,
        eps=1e-6,
        affine=True,
    )


class Upsample(nn.Module):
    # 定义上采样模块
    def __init__(self, in_channels, with_conv):
        # 初始化父类
        super().__init__()
        # 是否使用卷积层
        self.with_conv = with_conv
        # 如果使用卷积层，定义卷积层
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 将输入 x 上采样，放大两倍
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        # 如果使用卷积层，对上采样后的 x 进行卷积处理
        if self.with_conv:
            x = self.conv(x)
        # 返回处理后的 x
        return x


class Downsample(nn.Module):
    # 该类的定义未完成
    # 初始化方法，设置输入通道和是否使用卷积
        def __init__(self, in_channels, with_conv):
            # 调用父类初始化方法
            super().__init__()
            # 存储是否使用卷积的标志
            self.with_conv = with_conv
            # 如果选择使用卷积
            if self.with_conv:
                # 创建一个 2D 卷积层，卷积核大小为 3，步幅为 2，填充为 0
                # 注意：在 torch 卷积中没有非对称填充，必须手动处理
                self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
    
        # 前向传播方法，接收输入张量 x
        def forward(self, x):
            # 如果选择使用卷积
            if self.with_conv:
                # 定义填充参数，添加零填充以匹配卷积输入要求
                pad = (0, 1, 0, 1)
                # 对输入 x 进行常量填充
                x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
                # 通过卷积层处理输入 x
                x = self.conv(x)
            else:
                # 如果不使用卷积，执行平均池化操作，降低维度
                x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
            # 返回处理后的输出 x
            return x
# 定义一个残差块类，继承自 nn.Module
class ResnetBlock(nn.Module):
    # 初始化方法，接受多个参数配置残差块
    def __init__(
        self,
        *,
        in_channels,  # 输入通道数
        out_channels=None,  # 输出通道数，默认为 None
        conv_shortcut=False,  # 是否使用卷积短接
        dropout,  # dropout 的比率
        temb_channels=512,  # 时间嵌入的通道数，默认值为 512
        zq_ch=None,  # zq 的通道数，默认为 None
        add_conv=False,  # 是否添加卷积
    ):
        # 调用父类初始化方法
        super().__init__()
        self.in_channels = in_channels  # 设置输入通道数
        # 确定输出通道数，如果未提供，则等于输入通道数
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels  # 设置输出通道数
        self.use_conv_shortcut = conv_shortcut  # 设置是否使用卷积短接

        # 初始化归一化层
        self.norm1 = Normalize(in_channels, zq_ch, add_conv=add_conv)
        # 初始化第一个卷积层
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 如果时间嵌入通道数大于 0，初始化时间嵌入投影层
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        # 初始化第二个归一化层
        self.norm2 = Normalize(out_channels, zq_ch, add_conv=add_conv)
        # 初始化 dropout 层
        self.dropout = torch.nn.Dropout(dropout)
        # 初始化第二个卷积层
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 如果输入通道数与输出通道数不同，设置短接卷积
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                # 使用卷积短接
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                # 使用 1x1 卷积短接
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    # 前向传播方法，接受输入 x、时间嵌入 temb 和 zq
    def forward(self, x, temb, zq):
        h = x  # 将输入赋值给 h
        h = self.norm1(h, zq)  # 对 h 进行第一次归一化
        h = nonlinearity(h)  # 应用非线性激活函数
        h = self.conv1(h)  # 通过第一个卷积层

        # 如果时间嵌入存在，则进行相应处理
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]  # 加入时间嵌入投影

        h = self.norm2(h, zq)  # 对 h 进行第二次归一化
        h = nonlinearity(h)  # 应用非线性激活函数
        h = self.dropout(h)  # 应用 dropout
        h = self.conv2(h)  # 通过第二个卷积层

        # 如果输入和输出通道数不同，进行短接处理
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)  # 使用卷积短接
            else:
                x = self.nin_shortcut(x)  # 使用 1x1 卷积短接

        # 返回输入和 h 的和
        return x + h  # 残差连接


# 定义一个注意力块类，继承自 nn.Module
class AttnBlock(nn.Module):
    # 初始化方法，接受输入通道数和可选参数
    def __init__(self, in_channels, zq_ch=None, add_conv=False):
        # 调用父类初始化方法
        super().__init__()
        self.in_channels = in_channels  # 设置输入通道数

        # 初始化归一化层
        self.norm = Normalize(in_channels, zq_ch, add_conv=add_conv)
        # 初始化查询卷积层
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 初始化键卷积层
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 初始化值卷积层
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 初始化输出卷积层
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    # 定义前向传播函数，接受输入 x 和 zq
        def forward(self, x, zq):
            # 将输入 x 赋值给 h_
            h_ = x
            # 对 h_ 进行归一化处理，依据 zq
            h_ = self.norm(h_, zq)
            # 通过 q 层对 h_ 进行变换，得到查询 q
            q = self.q(h_)
            # 通过 k 层对 h_ 进行变换，得到键 k
            k = self.k(h_)
            # 通过 v 层对 h_ 进行变换，得到值 v
            v = self.v(h_)
    
            # 计算注意力
            # 获取 q 的形状，b 是批大小，c 是通道数，h 和 w 是高和宽
            b, c, h, w = q.shape
            # 将 q 重塑为 (b, c, h*w) 的形状
            q = q.reshape(b, c, h * w)
            # 变换 q 的维度顺序为 (b, hw, c)
            q = q.permute(0, 2, 1)  # b,hw,c
            # 将 k 重塑为 (b, c, h*w) 的形状
            k = k.reshape(b, c, h * w)  # b,c,hw
            # 计算 q 和 k 的乘积，得到注意力权重 w_
            w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
            # 对权重进行缩放
            w_ = w_ * (int(c) ** (-0.5))
            # 对权重应用 softmax 函数，进行归一化
            w_ = torch.nn.functional.softmax(w_, dim=2)
    
            # 对值进行注意力计算
            # 将 v 重塑为 (b, c, h*w) 的形状
            v = v.reshape(b, c, h * w)
            # 变换 w_ 的维度顺序为 (b, hw, hw)
            w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
            # 计算 v 和 w_ 的乘积，得到最终的 h_
            h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
            # 将 h_ 重塑为 (b, c, h, w) 的形状
            h_ = h_.reshape(b, c, h, w)
    
            # 对 h_ 进行投影处理
            h_ = self.proj_out(h_)
    
            # 返回输入 x 和 h_ 的和
            return x + h_
# 定义一个名为 MOVQDecoder 的类，继承自 nn.Module
class MOVQDecoder(nn.Module):
    # 初始化方法，用于创建 MOVQDecoder 的实例
    def __init__(
        # 接受关键字参数 ch，表示输入通道数
        *,
        ch,
        # 接受关键字参数 out_ch，表示输出通道数
        out_ch,
        # 接受关键字参数 ch_mult，表示通道倍增的因子，默认为 (1, 2, 4, 8)
        ch_mult=(1, 2, 4, 8),
        # 接受关键字参数 num_res_blocks，表示残差块的数量
        num_res_blocks,
        # 接受关键字参数 attn_resolutions，表示注意力分辨率
        attn_resolutions,
        # 接受关键字参数 dropout，表示 dropout 的比例，默认为 0.0
        dropout=0.0,
        # 接受关键字参数 resamp_with_conv，表示是否使用卷积进行重采样，默认为 True
        resamp_with_conv=True,
        # 接受关键字参数 in_channels，表示输入数据的通道数
        in_channels,
        # 接受关键字参数 resolution，表示输入数据的分辨率
        resolution,
        # 接受关键字参数 z_channels，表示潜在空间的通道数
        z_channels,
        # 接受关键字参数 give_pre_end，表示是否提供前置结束标志，默认为 False
        give_pre_end=False,
        # 接受关键字参数 zq_ch，表示潜在空间的量化通道数，默认为 None
        zq_ch=None,
        # 接受关键字参数 add_conv，表示是否添加额外的卷积层，默认为 False
        add_conv=False,
        # 接受其他未指定的关键字参数，使用 **ignorekwargs 收集
        **ignorekwargs,
    # 定义构造函数的结束部分
        ):
            # 调用父类构造函数
            super().__init__()
            # 存储输入参数 ch
            self.ch = ch
            # 初始化 temb_ch 为 0
            self.temb_ch = 0
            # 计算分辨率的数量
            self.num_resolutions = len(ch_mult)
            # 存储残差块的数量
            self.num_res_blocks = num_res_blocks
            # 存储分辨率
            self.resolution = resolution
            # 存储输入通道数
            self.in_channels = in_channels
            # 存储是否给出预处理结束标志
            self.give_pre_end = give_pre_end
    
            # 计算输入通道数乘法，块输入和当前分辨率
            in_ch_mult = (1,) + tuple(ch_mult)
            # 计算当前块的输入通道数
            block_in = ch * ch_mult[self.num_resolutions - 1]
            # 计算当前分辨率
            curr_res = resolution // 2 ** (self.num_resolutions - 1)
            # 定义 z_shape，表示 z 的形状
            self.z_shape = (1, z_channels, curr_res, curr_res)
            # 打印 z 的形状信息
            print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))
    
            # 定义从 z 到块输入的卷积层
            self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
    
            # 创建中间模块
            self.mid = nn.Module()
            # 定义中间块 1
            self.mid.block_1 = ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                temb_channels=self.temb_ch,
                dropout=dropout,
                zq_ch=zq_ch,
                add_conv=add_conv,
            )
            # 定义中间注意力块 1
            self.mid.attn_1 = AttnBlock(block_in, zq_ch, add_conv=add_conv)
            # 定义中间块 2
            self.mid.block_2 = ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                temb_channels=self.temb_ch,
                dropout=dropout,
                zq_ch=zq_ch,
                add_conv=add_conv,
            )
    
            # 初始化上采样模块列表
            self.up = nn.ModuleList()
            # 遍历分辨率，从高到低
            for i_level in reversed(range(self.num_resolutions)):
                # 初始化块和注意力模块列表
                block = nn.ModuleList()
                attn = nn.ModuleList()
                # 计算当前输出块的通道数
                block_out = ch * ch_mult[i_level]
                # 遍历每个残差块
                for i_block in range(self.num_res_blocks + 1):
                    # 添加残差块到块列表
                    block.append(
                        ResnetBlock(
                            in_channels=block_in,
                            out_channels=block_out,
                            temb_channels=self.temb_ch,
                            dropout=dropout,
                            zq_ch=zq_ch,
                            add_conv=add_conv,
                        )
                    )
                    # 更新块输入通道数
                    block_in = block_out
                    # 如果当前分辨率需要注意力模块，添加注意力模块
                    if curr_res in attn_resolutions:
                        attn.append(AttnBlock(block_in, zq_ch, add_conv=add_conv))
                # 创建上采样模块
                up = nn.Module()
                # 存储块和注意力模块
                up.block = block
                up.attn = attn
                # 如果不是最低分辨率，添加上采样层
                if i_level != 0:
                    up.upsample = Upsample(block_in, resamp_with_conv)
                    # 更新当前分辨率
                    curr_res = curr_res * 2
                # 将上采样模块插入到列表前面，确保顺序一致
                self.up.insert(0, up)  # prepend to get consistent order
    
            # 创建输出的归一化层
            self.norm_out = Normalize(block_in, zq_ch, add_conv=add_conv)
            # 定义输出的卷积层
            self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)
    # 前向传播方法，接受输入 z 和条件 zq
    def forward(self, z, zq):
        # 断言 z 的形状与预期形状相符（已注释掉）
        # assert z.shape[1:] == self.z_shape[1:]
        # 保存输入 z 的形状
        self.last_z_shape = z.shape
    
        # 时间步嵌入初始化
        temb = None
    
        # 将 z 输入到卷积层
        h = self.conv_in(z)
    
        # 中间处理层
        h = self.mid.block_1(h, temb, zq)
        h = self.mid.attn_1(h, zq)
        h = self.mid.block_2(h, temb, zq)
    
        # 上采样过程
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, zq)
                # 如果当前层有注意力模块，则应用注意力模块
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
            # 如果不是最后一层，则进行上采样
            if i_level != 0:
                h = self.up[i_level].upsample(h)
    
        # 结束处理，条件性返回结果
        if self.give_pre_end:
            return h
    
        # 输出层归一化处理
        h = self.norm_out(h, zq)
        # 应用非线性激活函数
        h = nonlinearity(h)
        # 最后卷积层处理
        h = self.conv_out(h)
        return h
    
    # 带特征输出的前向传播方法，接受输入 z 和条件 zq
    def forward_with_features_output(self, z, zq):
        # 断言 z 的形状与预期形状相符（已注释掉）
        # assert z.shape[1:] == self.z_shape[1:]
        # 保存输入 z 的形状
        self.last_z_shape = z.shape
    
        # 时间步嵌入初始化
        temb = None
        output_features = {}
    
        # 将 z 输入到卷积层
        h = self.conv_in(z)
        # 保存卷积层输出特征
        output_features["conv_in"] = h
    
        # 中间处理层
        h = self.mid.block_1(h, temb, zq)
        # 保存中间块 1 的输出特征
        output_features["mid_block_1"] = h
        h = self.mid.attn_1(h, zq)
        # 保存中间注意力 1 的输出特征
        output_features["mid_attn_1"] = h
        h = self.mid.block_2(h, temb, zq)
        # 保存中间块 2 的输出特征
        output_features["mid_block_2"] = h
    
        # 上采样过程
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, zq)
                # 保存每个上采样块的输出特征
                output_features[f"up_{i_level}_block_{i_block}"] = h
                # 如果当前层有注意力模块，则应用注意力模块并保存特征
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
                    output_features[f"up_{i_level}_attn_{i_block}"] = h
            # 如果不是最后一层，则进行上采样并保存特征
            if i_level != 0:
                h = self.up[i_level].upsample(h)
                output_features[f"up_{i_level}_upsample"] = h
    
        # 结束处理，条件性返回结果
        if self.give_pre_end:
            return h
    
        # 输出层归一化处理
        h = self.norm_out(h, zq)
        # 保存归一化后的特征
        output_features["norm_out"] = h
        # 应用非线性激活函数并保存特征
        h = nonlinearity(h)
        output_features["nonlinearity"] = h
        # 最后卷积层处理并保存特征
        h = self.conv_out(h)
        output_features["conv_out"] = h
    
        return h, output_features
```