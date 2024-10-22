# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\vqvae\vqvae_blocks.py`

```py
# pytorch_diffusion + derived encoder decoder
import math  # 导入数学库，用于数学计算
import torch  # 导入 PyTorch 库，用于深度学习
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import numpy as np  # 导入 NumPy 库，用于数组处理

def get_timestep_embedding(timesteps, embedding_dim):
    """
    该函数实现了 Denoising Diffusion Probabilistic Models 中的嵌入构建
    从 Fairseq。
    构建正弦波嵌入。
    该实现与 tensor2tensor 中的实现匹配，但与 "Attention Is All You Need" 的
    第 3.5 节中的描述略有不同。
    """
    # 确保 timesteps 是一维数组
    assert len(timesteps.shape) == 1

    # 计算嵌入维度的一半
    half_dim = embedding_dim // 2
    # 计算用于正弦和余弦的频率
    emb = math.log(10000) / (half_dim - 1)
    # 生成正弦和余弦嵌入所需的频率向量
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    # 将频率向量移动到与 timesteps 相同的设备
    emb = emb.to(device=timesteps.device)
    # 根据时间步生成最终的嵌入
    emb = timesteps.float()[:, None] * emb[None, :]
    # 将正弦和余弦嵌入合并
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    # 如果嵌入维度是奇数，进行零填充
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    # 返回生成的嵌入
    return emb

def nonlinearity(x):
    # 定义 Swish 非线性激活函数
    return x * torch.sigmoid(x)  # 返回 Swish 激活值

def Normalize(in_channels):
    # 定义归一化层，使用 GroupNorm
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        # 初始化 Upsample 类
        super().__init__()  # 调用父类构造函数
        self.with_conv = with_conv  # 根据参数决定是否使用卷积层
        if self.with_conv:
            # 如果使用卷积，则定义卷积层
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 定义前向传播方法
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")  # 进行上采样
        if self.with_conv:
            x = self.conv(x)  # 如果需要，经过卷积层处理
        return x  # 返回处理后的张量

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        # 初始化 Downsample 类
        super().__init__()  # 调用父类构造函数
        self.with_conv = with_conv  # 根据参数决定是否使用卷积层
        if self.with_conv:
            # 定义卷积层，步幅为 2，填充为 0
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        # 定义前向传播方法
        if self.with_conv:
            pad = (0, 1, 0, 1)  # 定义填充参数
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)  # 对输入张量进行填充
            x = self.conv(x)  # 经过卷积层处理
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)  # 进行平均池化
        return x  # 返回处理后的张量

class ResnetBlock(nn.Module):
    # 初始化方法，设置输入输出通道、卷积快捷连接、丢弃率和时间嵌入通道数
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        # 调用父类的初始化方法
        super().__init__()
        # 设置输入通道数
        self.in_channels = in_channels
        # 如果未提供输出通道数，则设置为输入通道数
        out_channels = in_channels if out_channels is None else out_channels
        # 设置输出通道数
        self.out_channels = out_channels
        # 设置是否使用卷积快捷连接的标志
        self.use_conv_shortcut = conv_shortcut

        # 初始化归一化层，处理输入通道
        self.norm1 = Normalize(in_channels)
        # 初始化第一层卷积，输入输出通道数相应，使用3x3卷积核
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 如果时间嵌入通道数大于0，则初始化时间嵌入投影层
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        # 初始化第二层归一化，处理输出通道
        self.norm2 = Normalize(out_channels)
        # 初始化丢弃层，设置丢弃率
        self.dropout = torch.nn.Dropout(dropout)
        # 初始化第二层卷积，输入输出通道数相应，使用3x3卷积核
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 如果输入通道数与输出通道数不相同
        if self.in_channels != self.out_channels:
            # 根据是否使用卷积快捷连接，初始化相应的卷积层
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    # 前向传播方法，处理输入数据和时间嵌入
    def forward(self, x, temb):
        # 将输入赋值给中间变量
        h = x
        # 对输入进行归一化处理
        h = self.norm1(h)
        # 应用非线性激活函数
        h = nonlinearity(h)
        # 经过第一层卷积
        h = self.conv1(h)

        # 如果时间嵌入不为 None，则将其添加到中间变量中
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        # 对中间变量进行第二次归一化
        h = self.norm2(h)
        # 应用非线性激活函数
        h = nonlinearity(h)
        # 应用丢弃层
        h = self.dropout(h)
        # 经过第二层卷积
        h = self.conv2(h)

        # 如果输入通道数与输出通道数不相同
        if self.in_channels != self.out_channels:
            # 根据是否使用卷积快捷连接处理输入
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        # 返回输入和中间变量的和
        return x + h
# 定义注意力模块类，继承自 nn.Module
class AttnBlock(nn.Module):
    # 初始化方法，接受输入通道数
    def __init__(self, in_channels):
        # 调用父类的初始化方法
        super().__init__()
        # 保存输入通道数
        self.in_channels = in_channels

        # 初始化归一化层
        self.norm = Normalize(in_channels)
        # 初始化用于生成查询（q）的卷积层
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 初始化用于生成键（k）的卷积层
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 初始化用于生成值（v）的卷积层
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 初始化输出投影的卷积层
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    # 前向传播方法
    def forward(self, x):
        # 将输入赋值给 h_
        h_ = x
        # 对输入进行归一化处理
        h_ = self.norm(h_)
        # 通过查询卷积层得到查询向量 q
        q = self.q(h_)
        # 通过键卷积层得到键向量 k
        k = self.k(h_)
        # 通过值卷积层得到值向量 v
        v = self.v(h_)

        # 计算注意力
        # 获取查询向量的形状参数
        b, c, h, w = q.shape
        # 将 q 进行重塑，改变形状为 (b, c, h*w)
        q = q.reshape(b, c, h * w)
        # 变换 q 的维度顺序，变为 (b, hw, c)
        q = q.permute(0, 2, 1)  # b,hw,c
        # 将 k 进行重塑，改变形状为 (b, c, hw)
        k = k.reshape(b, c, h * w)  # b,c,hw

        # # 原始版本，在 fp16 中会出现 nan
        # w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        # w_ = w_ * (int(c)**(-0.5))
        # # 实现 c**-0.5 在 q 上
        # 将查询向量 q 乘以 c 的倒数平方根
        q = q * (int(c) ** (-0.5))
        # 计算权重 w_，使用批量矩阵相乘
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]

        # 对权重 w_ 进行 softmax 归一化
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # 根据值向量进行加权
        # 将值向量 v 进行重塑，改变形状为 (b, c, h*w)
        v = v.reshape(b, c, h * w)
        # 变换 w_ 的维度顺序，变为 (b, hw, hw) 
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # 通过批量矩阵相乘计算加权后的结果 h_
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        # 将 h_ 进行重塑，改变形状回到 (b, c, h, w)
        h_ = h_.reshape(b, c, h, w)

        # 通过输出投影层生成最终结果
        h_ = self.proj_out(h_)

        # 返回输入和经过注意力模块处理后的结果相加
        return x + h_


# 定义编码器类，继承自 nn.Module
class Encoder(nn.Module):
    # 初始化方法，接收多个参数
    def __init__(
        self,
        *,
        ch,  # 通道数
        out_ch,  # 输出通道数
        ch_mult=(1, 2, 4, 8),  # 通道数的倍率
        num_res_blocks,  # 残差块的数量
        attn_resolutions,  # 注意力计算的分辨率
        dropout=0.0,  # dropout 概率
        resamp_with_conv=True,  # 是否使用卷积进行重采样
        in_channels,  # 输入通道数
        resolution,  # 输入分辨率
        z_channels,  # z 维度通道数
        double_z=True,  # 是否使用双 z
        **ignore_kwargs,  # 其他忽略的关键字参数
    # 定义类的初始化方法
        ):
            # 调用父类的初始化方法
            super().__init__()
            # 初始化通道数
            self.ch = ch
            # 初始化时间嵌入通道数
            self.temb_ch = 0
            # 获取分辨率数量
            self.num_resolutions = len(ch_mult)
            # 获取残差块数量
            self.num_res_blocks = num_res_blocks
            # 设置分辨率
            self.resolution = resolution
            # 输入通道数
            self.in_channels = in_channels
    
            # downsampling
            # 初始化卷积层，输入通道为in_channels，输出通道为self.ch
            self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
    
            # 当前分辨率
            curr_res = resolution
            # 设置输入通道的倍数
            in_ch_mult = (1,) + tuple(ch_mult)
            # 初始化一个模块列表来存储下采样模块
            self.down = nn.ModuleList()
            # 遍历每个分辨率级别
            for i_level in range(self.num_resolutions):
                # 初始化块和注意力模块的模块列表
                block = nn.ModuleList()
                attn = nn.ModuleList()
                # 计算当前块的输入和输出通道数
                block_in = ch * in_ch_mult[i_level]
                block_out = ch * ch_mult[i_level]
                # 遍历每个残差块
                for i_block in range(self.num_res_blocks):
                    # 添加残差块到块列表
                    block.append(
                        ResnetBlock(
                            in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout
                        )
                    )
                    # 更新输入通道数为输出通道数
                    block_in = block_out
                    # 如果当前分辨率在注意力分辨率中，添加注意力块
                    if curr_res in attn_resolutions:
                        attn.append(AttnBlock(block_in))
                # 创建下采样模块
                down = nn.Module()
                down.block = block
                down.attn = attn
                # 如果不是最后一个分辨率级别，添加下采样层
                if i_level != self.num_resolutions - 1:
                    down.downsample = Downsample(block_in, resamp_with_conv)
                    # 将当前分辨率减半
                    curr_res = curr_res // 2
                # 将下采样模块添加到列表中
                self.down.append(down)
    
            # middle
            # 创建中间模块
            self.mid = nn.Module()
            # 添加第一个残差块
            self.mid.block_1 = ResnetBlock(
                in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
            )
            # 添加中间注意力块
            self.mid.attn_1 = AttnBlock(block_in)
            # 添加第二个残差块
            self.mid.block_2 = ResnetBlock(
                in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
            )
    
            # end
            # 规范化输出
            self.norm_out = Normalize(block_in)
            # 初始化输出卷积层，输出通道为2 * z_channels或z_channels
            self.conv_out = torch.nn.Conv2d(
                block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1
            )
    
        # 定义前向传播方法
        def forward(self, x):
            # 确保输入张量的宽和高等于设定的分辨率，若不匹配则抛出异常
            # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)
    
            # timestep embedding
            # 初始化时间嵌入
            temb = None
    
            # downsampling
            # 通过输入数据进行卷积操作
            hs = [self.conv_in(x)]
            # 遍历每个分辨率级别
            for i_level in range(self.num_resolutions):
                # 遍历每个残差块
                for i_block in range(self.num_res_blocks):
                    # 通过残差块处理前一层的输出
                    h = self.down[i_level].block[i_block](hs[-1], temb)
                    # 如果存在注意力模块，则进行注意力处理
                    if len(self.down[i_level].attn) > 0:
                        h = self.down[i_level].attn[i_block](h)
                    # 将当前层输出添加到列表中
                    hs.append(h)
                # 如果不是最后一个分辨率级别，则进行下采样
                if i_level != self.num_resolutions - 1:
                    hs.append(self.down[i_level].downsample(hs[-1]))
    
            # middle
            # 获取最后一层的输出
            h = hs[-1]
            # 通过中间块进行处理
            h = self.mid.block_1(h, temb)
            # 进行注意力处理
            h = self.mid.attn_1(h)
            # 通过第二个中间块进行处理
            h = self.mid.block_2(h, temb)
    
            # end
            # 规范化处理
            h = self.norm_out(h)
            # 应用非线性激活函数
            h = nonlinearity(h)
            # 通过输出卷积层得到最终输出
            h = self.conv_out(h)
            # 返回处理后的输出
            return h
    # 定义一个前向传播函数，输出特征
    def forward_with_features_output(self, x):
        # 断言输入张量的高和宽等于预设的分辨率
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # 时间步嵌入初始化为空
        temb = None
        # 用于存储各层输出特征的字典
        output_features = {}

        # 下采样阶段，首先通过输入卷积层处理输入 x
        hs = [self.conv_in(x)]
        # 将输入卷积的输出保存到输出特征字典中
        output_features["conv_in"] = hs[-1]
        # 遍历每个分辨率层级
        for i_level in range(self.num_resolutions):
            # 遍历每个残差块
            for i_block in range(self.num_res_blocks):
                # 通过当前层级的当前块进行处理
                h = self.down[i_level].block[i_block](hs[-1], temb)
                # 将当前块的输出保存到输出特征字典中
                output_features["down{}_block{}".format(i_level, i_block)] = h
                # 如果当前层级有注意力机制，应用注意力机制
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                    # 将注意力机制的输出保存到输出特征字典中
                    output_features["down{}_attn{}".format(i_level, i_block)] = h
                # 将当前块的输出加入历史输出列表
                hs.append(h)
            # 如果不是最后一个分辨率层级，进行下采样
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
                # 将下采样的输出保存到输出特征字典中
                output_features["down{}_downsample".format(i_level)] = hs[-1]

        # 中间层处理
        h = hs[-1]
        # 通过中间块1进行处理
        h = self.mid.block_1(h, temb)
        # 将中间块1的输出保存到输出特征字典中
        output_features["mid_block_1"] = h
        # 应用中间层的注意力机制
        h = self.mid.attn_1(h)
        # 将中间层注意力机制的输出保存到输出特征字典中
        output_features["mid_attn_1"] = h
        # 通过中间块2进行处理
        h = self.mid.block_2(h, temb)
        # 将中间块2的输出保存到输出特征字典中
        output_features["mid_block_2"] = h

        # 结束处理阶段
        h = self.norm_out(h)  # 进行归一化处理
        output_features["norm_out"] = h  # 保存归一化输出
        h = nonlinearity(h)  # 应用非线性激活函数
        output_features["nonlinearity"] = h  # 保存非线性输出
        h = self.conv_out(h)  # 通过输出卷积层处理
        output_features["conv_out"] = h  # 保存输出卷积的结果

        # 返回最终输出和特征字典
        return h, output_features
# 定义一个名为Decoder的类，继承自nn.Module
class Decoder(nn.Module):
    # 初始化函数，接收一系列参数
    def __init__(
        self,
        *,
        ch,  # 输入通道数
        out_ch,  # 输出通道数
        ch_mult=(1, 2, 4, 8),  # 通道数的倍数
        num_res_blocks,  # 残差块的数量
        attn_resolutions,  # 注意力机制的分辨率
        dropout=0.0,  # dropout的比例
        resamp_with_conv=True,  # 是否使用卷积进行重采样
        in_channels,  # 输入的通道数
        resolution,  # 分辨率
        z_channels,  # z的通道数
        give_pre_end=False,  # 是否给出预处理结果
        **ignorekwargs,  # 忽略的关键字参数
    ):
        super().__init__()
        # 将参数赋值给类的属性
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # 计算最低分辨率下的in_ch_mult、block_in和curr_res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        # 打印z的形状
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # 将z转换为block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # 中间部分
        self.mid = nn.Module()
        # 中间部分的第一个残差块
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        # 中间部分的注意力机制
        self.mid.attn_1 = AttnBlock(block_in)
        # 中间部分的第二个残差块
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )

        # 上采样部分
        self.up = nn.ModuleList()
        # 从最高分辨率开始逆序遍历
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            # 根据残差块的数量创建残差块和注意力机制
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout
                    )
                )
                block_in = block_out
                # 如果当前分辨率在attn_resolutions中，则添加注意力机制
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            # 如果不是最低分辨率，则添加上采样层
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            # 将up插入到up列表的开头，以保持一致的顺序
            self.up.insert(0, up)

        # 结束部分
        # 归一化层
        self.norm_out = Normalize(block_in)
        # 输出卷积层
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)
    # 定义前向传播函数，接收输入 z
    def forward(self, z):
        # 检查 z 的形状是否与期望的 z_shape 相匹配（被注释掉的断言）
        # assert z.shape[1:] == self.z_shape[1:]
        # 保存当前输入 z 的形状
        self.last_z_shape = z.shape

        # 初始化时间步嵌入为 None
        temb = None

        # 将输入 z 通过第一层卷积层进行处理，得到 block_in
        h = self.conv_in(z)

        # 中间处理阶段
        # 将 h 传入第一块中间模块，结合时间步嵌入 temb
        h = self.mid.block_1(h, temb)
        # 经过第一个注意力层
        h = self.mid.attn_1(h)
        # 将 h 传入第二块中间模块，结合时间步嵌入 temb
        h = self.mid.block_2(h, temb)

        # 上采样阶段
        # 从高到低的分辨率进行迭代
        for i_level in reversed(range(self.num_resolutions)):
            # 在每个分辨率下遍历所有的块
            for i_block in range(self.num_res_blocks + 1):
                # 将 h 传入当前上采样块进行处理，结合时间步嵌入 temb
                h = self.up[i_level].block[i_block](h, temb)
                # 如果当前块有注意力层，进行注意力处理
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            # 如果不是最后一层，进行上采样
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # 结束阶段
        # 如果给定了预结束标志，直接返回 h
        if self.give_pre_end:
            return h

        # 对 h 进行归一化处理
        h = self.norm_out(h)
        # 应用非线性激活函数
        h = nonlinearity(h)
        # 通过输出卷积层得到最终结果
        h = self.conv_out(h)
        # 返回处理后的结果 h
        return h
```