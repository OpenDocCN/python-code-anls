# `.\comic-translate\modules\inpainting\ffc.py`

```py
# 导入 PyTorch 库
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义 FFCSE_block 类，继承自 nn.Module
class FFCSE_block(nn.Module):

    # 初始化方法，接收 channels 和 ratio_g 两个参数
    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        # 计算输入到全局和局部的通道数
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16  # 定义 r 为 16

        # 定义自适应平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 定义第一个 1x1 卷积层
        self.conv1 = nn.Conv2d(channels, channels // r,
                               kernel_size=1, bias=True)
        # 定义 ReLU 激活函数
        self.relu1 = nn.ReLU(inplace=True)
        
        # 如果 in_cl 不为 0，则定义局部注意力机制的卷积层
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)
        
        # 如果 in_cg 不为 0，则定义全局注意力机制的卷积层
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)
        
        # 定义 Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播方法，接收输入 x
    def forward(self, x):
        # 如果 x 的类型是 tuple，则分别取出 id_l 和 id_g
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        # 如果 id_g 的类型是 int，则将 id_l 作为输入 x
        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        # 进行自适应平均池化
        x = self.avgpool(x)
        # 使用第一个卷积层和 ReLU 激活函数处理
        x = self.relu1(self.conv1(x))

        # 计算局部和全局的注意力机制输出
        x_l = 0 if self.conv_a2l is None else id_l * \
            self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * \
            self.sigmoid(self.conv_a2g(x))
        
        # 返回局部和全局注意力机制的输出
        return x_l, x_g


# 定义 FourierUnit 类，继承自 nn.Module
class FourierUnit(nn.Module):

    # 初始化方法，接收多个参数
    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # 调用父类的初始化方法
        super(FourierUnit, self).__init__()
        # 设置 groups 参数
        self.groups = groups

        # 定义卷积层
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        # 定义批归一化层
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        # 定义 ReLU 激活函数
        self.relu = torch.nn.ReLU(inplace=True)

        # 是否使用 squeeze and excitation block
        self.use_se = use_se
        # 如果使用 squeeze and excitation
        # if use_se:
        #     if se_kwargs is None:
        #         se_kwargs = {}
        #     self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        # 空间尺度因子和模式
        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        # 是否使用谱位置编码
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        # FFT 规范化方式
        self.fft_norm = fft_norm
    # 定义一个前向传播方法，接受输入张量 x
    def forward(self, x):
        # 获取输入张量 x 的批量大小
        batch = x.shape[0]

        # 如果设置了空间缩放因子，保存原始尺寸并按比例插值调整输入张量 x
        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        # 记录调整后的 x 的尺寸
        r_size = x.size()
        
        # 确定 FFT 的维度，针对三维 FFT 或二维 FFT 的不同维度排序
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)

        # 如果输入张量 x 的数据类型是 torch.float16 或 torch.bfloat16，则转换为 torch.float32
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.type(torch.float32)

        # 对输入张量 x 进行实数部分的快速傅里叶变换
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        
        # 将实部和虚部合并成一个复数张量
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        
        # 调整复数张量的维度顺序为 (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        
        # 将张量展平，保持批量维度不变
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        # 如果设置了频谱位置编码，添加垂直和水平坐标到 ffted 的前面
        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        # 如果使用了 SE（Squeeze-and-Excitation）模块，对 ffted 进行处理
        if self.use_se:
            ffted = self.se(ffted)

        # 对 ffted 应用卷积层
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        
        # 经过批量归一化和 ReLU 激活函数
        ffted = self.relu(self.bn(ffted))

        # 将 ffted 的维度顺序调整为 (batch, c, t, h, w/2+1, 2)
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()

        # 如果 ffted 的数据类型是 torch.float16 或 torch.bfloat16，则转换为 torch.float32
        if ffted.dtype in (torch.float16, torch.bfloat16):
            ffted = ffted.type(torch.float32)
        
        # 将复数张量 ffted 拆分为实部和虚部，返回复数张量
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        # 根据需要执行逆快速傅里叶变换，恢复原始尺寸的输出
        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        # 如果设置了空间缩放因子，按原始尺寸插值调整输出
        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        # 返回处理后的输出张量
        return output
class SpectralTransform(nn.Module):
    # 定义一个名为 SpectralTransform 的 PyTorch 模块
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # 初始化函数，设置模块的各种属性和层
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        # 是否启用 LFU 模块
        self.enable_lfu = enable_lfu
        # 根据 stride 设置下采样层，如果 stride 为 2，则使用平均池化层进行下采样，否则使用恒等映射
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        # 第一个卷积层，包括 1x1 的卷积和批归一化层，激活函数为 ReLU
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        # FourierUnit 对象，用于进行频域变换
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        # 如果启用 LFU，则创建另一个 FourierUnit 对象
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        # 第二个卷积层，1x1 的卷积层
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        # 前向传播函数
        # 对输入进行下采样
        x = self.downsample(x)
        # 第一层卷积
        x = self.conv1(x)
        # 使用 FourierUnit 进行频域变换
        output = self.fu(x)

        # 如果启用 LFU
        if self.enable_lfu:
            # 对 x 进行形状变换，划分为多个小块进行 LFU 操作
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            # LFU 操作
            xs = self.lfu(xs)
            # 将结果扩展回原始形状
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        # 最终输出，经过第二个卷积层
        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):
    # 定义一个名为 FFC 的 PyTorch 模块，暂未实现内容
    # 定义 FFC 类，继承自 nn.Module
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        # 调用父类的初始化方法
        super(FFC, self).__init__()

        # 断言确保步长为 1 或 2
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        # 保存步长
        self.stride = stride

        # 计算输入通道的全局和局部部分
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        # 计算输出通道的全局和局部部分
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        # 保存全局和局部通道比例
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        # 保存全局输入通道数
        self.global_in_num = in_cg

        # 根据条件选择使用 nn.Identity 或者 nn.Conv2d 模块作为局部到局部卷积
        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        # 根据条件选择使用 nn.Identity 或者 nn.Conv2d 模块作为局部到全局卷积
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        # 根据条件选择使用 nn.Identity 或者 nn.Conv2d 模块作为全局到局部卷积
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        # 根据条件选择使用 nn.Identity 或者 SpectralTransform 模块作为全局到全局卷积
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        # 保存是否启用门控模块的标志
        self.gated = gated
        # 根据条件选择使用 nn.Identity 或者 nn.Conv2d 模块作为门控模块
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    # 前向传播方法
    def forward(self, x):
        # 如果输入是元组，则分别提取局部和全局输入
        x_l, x_g = x if type(x) is tuple else (x, 0)
        # 初始化局部和全局输出为零张量
        out_xl, out_xg = 0, 0

        # 如果启用门控模块
        if self.gated:
            # 将局部和全局输入拼接在一起
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            # 计算门控信号
            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            # 若未启用门控模块，则门控信号为全 1
            g2l_gate, l2g_gate = 1, 1

        # 如果全局输出比例不为 1，则计算局部到局部卷积结果并加上全局到局部卷积结果
        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        # 如果全局输出比例不为 0，则计算局部到全局卷积结果并加上全局到全局卷积结果
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        # 返回局部和全局输出结果
        return out_xl, out_xg
class FFC_BN_ACT(nn.Module):
    # FFC_BN_ACT 类定义，继承自 nn.Module
    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        # 初始化方法，设置模块属性
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        # 如果 ratio_gout 为 1，则使用 nn.Identity 作为 norm_layer，否则使用 norm_layer
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        # 如果 ratio_gout 为 0，则使用 nn.Identity 作为 norm_layer，否则使用 norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        # 计算全局通道数量
        global_channels = int(out_channels * ratio_gout)
        # 初始化局部批归一化层和全局批归一化层
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        # 如果 ratio_gout 为 1，则使用 nn.Identity 作为 activation_layer，否则使用 activation_layer
        lact = nn.Identity if ratio_gout == 1 else activation_layer
        # 如果 ratio_gout 为 0，则使用 nn.Identity 作为 activation_layer，否则使用 activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        # 初始化局部激活层和全局激活层
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        # 前向传播函数
        # 使用 FFC 模块处理输入 x，得到局部特征 x_l 和全局特征 x_g
        x_l, x_g = self.ffc(x)
        # 对局部特征 x_l 应用局部激活和局部批归一化
        x_l = self.act_l(self.bn_l(x_l))
        # 对全局特征 x_g 应用全局激活和全局批归一化
        x_g = self.act_g(self.bn_g(x_g))
        # 返回处理后的局部特征 x_l 和全局特征 x_g
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    # FFCResnetBlock 类定义，继承自 nn.Module
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        # 初始化方法，设置模块属性
        # 第一个卷积层模块，使用 FFC_BN_ACT 模块，输入输出维度为 dim
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        # 第二个卷积层模块，使用 FFC_BN_ACT 模块，输入输出维度为 dim
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        # 如果存在 spatial_transform_kwargs 参数，则使用 LearnableSpatialTransformWrapper 封装卷积层
        # self.conv1 和 self.conv2
        # self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
        # self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x):
        # 前向传播函数
        if self.inline:
            # 如果 inline 为 True，分离输入 x 中的局部特征 x_l 和全局特征 x_g
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            # 如果 inline 不为 True，假设输入 x 为元组，分离局部特征 x_l 和全局特征 x_g
            x_l, x_g = x if type(x) is tuple else (x, 0)

        # 保存输入的局部特征和全局特征作为 id_l 和 id_g
        id_l, id_g = x_l, x_g

        # 对第一个卷积层模块进行前向传播，得到新的局部特征 x_l 和全局特征 x_g
        x_l, x_g = self.conv1((x_l, x_g))
        # 对第二个卷积层模块进行前向传播，得到新的局部特征 x_l 和全局特征 x_g
        x_l, x_g = self.conv2((x_l, x_g))

        # 将原始的局部特征和全局特征与处理后的特征相加
        x_l, x_g = id_l + x_l, id_g + x_g
        # 将处理后的局部特征和全局特征作为输出
        out = x_l, x_g
        # 如果 inline 为 True，将输出 x_l 和 x_g 拼接在一起
        if self.inline:
            out = torch.cat(out, dim=1)
        # 返回最终的输出
        return out
```