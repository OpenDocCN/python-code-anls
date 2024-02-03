# `stable-diffusion-webui\modules\esrgan_model_arch.py`

```py
# 导入所需的库
from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义 RRDBNet 生成器类
class RRDBNet(nn.Module):
    # 初始化函数，定义了超分辨率网络的结构和参数
    def __init__(self, in_nc, out_nc, nf, nb, nr=3, gc=32, upscale=4, norm_type=None,
            act_type='leakyrelu', mode='CNA', upsample_mode='upconv', convtype='Conv2D',
            finalact=None, gaussian_noise=False, plus=False):
        # 调用父类的初始化函数
        super(RRDBNet, self).__init__()
        # 计算上采样倍数
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        # 初始化变量用于记录是否支持 Residual in Residual Dense Block (RRDB) 的缩放
        self.resrgan_scale = 0
        # 判断输入通道数是否能被 16 整除
        if in_nc % 16 == 0:
            self.resrgan_scale = 1
        # 判断输入通道数是否不等于 4 且能被 4 整除
        elif in_nc != 4 and in_nc % 4 == 0:
            self.resrgan_scale = 2

        # 创建特征提取卷积层
        fea_conv = conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None, convtype=convtype)
        # 创建多个 RRDB 模块
        rb_blocks = [RRDB(nf, nr, kernel_size=3, gc=32, stride=1, bias=1, pad_type='zero',
            norm_type=norm_type, act_type=act_type, mode='CNA', convtype=convtype,
            gaussian_noise=gaussian_noise, plus=plus) for _ in range(nb)]
        # 创建低分辨率卷积层
        LR_conv = conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode, convtype=convtype)

        # 根据上采样模式选择上采样块
        if upsample_mode == 'upconv':
            upsample_block = upconv_block
        elif upsample_mode == 'pixelshuffle':
            upsample_block = pixelshuffle_block
        else:
            raise NotImplementedError(f'upsample mode [{upsample_mode}] is not found')
        # 根据上采样倍数创建上采样器
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type, convtype=convtype)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type, convtype=convtype) for _ in range(n_upscale)]
        # 创建高分辨率卷积层
        HR_conv0 = conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type, convtype=convtype)
        # 创建输出卷积层
        HR_conv1 = conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None, convtype=convtype)

        # 根据最终激活函数类型创建激活函数
        outact = act(finalact) if finalact else None

        # 构建整个网络模型
        self.model = sequential(fea_conv, ShortcutBlock(sequential(*rb_blocks, LR_conv)),
            *upsampler, HR_conv0, HR_conv1, outact)
    # 定义一个前向传播函数，接受输入 x 和输出 outm（默认为 None）
    def forward(self, x, outm=None):
        # 如果超分辨率比例为1，则使用像素解缩放函数对输入 x 进行处理
        if self.resrgan_scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        # 如果超分辨率比例为2，则使用像素解缩放函数对输入 x 进行处理
        elif self.resrgan_scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        # 如果超分辨率比例不是1或2，则直接使用输入 x
        else:
            feat = x

        # 将处理后的特征传递给模型进行处理，并返回结果
        return self.model(feat)
class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    """

    def __init__(self, nf, nr=3, kernel_size=3, gc=32, stride=1, bias=1, pad_type='zero',
            norm_type=None, act_type='leakyrelu', mode='CNA', convtype='Conv2D',
            spectral_norm=False, gaussian_noise=False, plus=False):
        # 初始化 Residual in Residual Dense Block 类
        super(RRDB, self).__init__()
        # 这是为了向后兼容现有模型
        if nr == 3:
            # 如果 nr 等于 3，则创建三个 ResidualDenseBlock_5C 实例
            self.RDB1 = ResidualDenseBlock_5C(nf, kernel_size, gc, stride, bias, pad_type,
                    norm_type, act_type, mode, convtype, spectral_norm=spectral_norm,
                    gaussian_noise=gaussian_noise, plus=plus)
            self.RDB2 = ResidualDenseBlock_5C(nf, kernel_size, gc, stride, bias, pad_type,
                    norm_type, act_type, mode, convtype, spectral_norm=spectral_norm,
                    gaussian_noise=gaussian_noise, plus=plus)
            self.RDB3 = ResidualDenseBlock_5C(nf, kernel_size, gc, stride, bias, pad_type,
                    norm_type, act_type, mode, convtype, spectral_norm=spectral_norm,
                    gaussian_noise=gaussian_noise, plus=plus)
        else:
            # 如果 nr 不等于 3，则创建 nr 个 ResidualDenseBlock_5C 实例
            RDB_list = [ResidualDenseBlock_5C(nf, kernel_size, gc, stride, bias, pad_type,
                                              norm_type, act_type, mode, convtype, spectral_norm=spectral_norm,
                                              gaussian_noise=gaussian_noise, plus=plus) for _ in range(nr)]
            self.RDBs = nn.Sequential(*RDB_list)

    def forward(self, x):
        # 如果存在 self.RDB1，则依次对 x 进行 RDB1、RDB2、RDB3 的处理
        if hasattr(self, 'RDB1'):
            out = self.RDB1(x)
            out = self.RDB2(out)
            out = self.RDB3(out)
        else:
            # 否则对 x 进行 RDBs 的处理
            out = self.RDBs(x)
        return out * 0.2 + x


class ResidualDenseBlock_5C(nn.Module):
    """
    Residual Dense Block
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    Modified options that can be used:
        - "Partial Convolution based Padding" arXiv:1811.11718
        - "Spectral normalization" arXiv:1802.05957
        - "ICASSP 2020 - ESRGAN+ : Further Improving ESRGAN" N. C.
            {Rakotonirina} and A. {Rasoanaivo}
    """

    # 定义 ResidualDenseBlock_5C 类，继承自父类
    def __init__(self, nf=64, kernel_size=3, gc=32, stride=1, bias=1, pad_type='zero',
            norm_type=None, act_type='leakyrelu', mode='CNA', convtype='Conv2D',
            spectral_norm=False, gaussian_noise=False, plus=False):
        super(ResidualDenseBlock_5C, self).__init__()

        # 如果 gaussian_noise 为 True，则创建 GaussianNoise 对象，否则为 None
        self.noise = GaussianNoise() if gaussian_noise else None
        # 如果 plus 为 True，则创建 conv1x1 对象，否则为 None
        self.conv1x1 = conv1x1(nf, gc) if plus else None

        # 创建第一个卷积块
        self.conv1 = conv_block(nf, gc, kernel_size, stride, bias=bias, pad_type=pad_type,
            norm_type=norm_type, act_type=act_type, mode=mode, convtype=convtype,
            spectral_norm=spectral_norm)
        # 创建第二个卷积块
        self.conv2 = conv_block(nf+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type,
            norm_type=norm_type, act_type=act_type, mode=mode, convtype=convtype,
            spectral_norm=spectral_norm)
        # 创建第三个卷积块
        self.conv3 = conv_block(nf+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type,
            norm_type=norm_type, act_type=act_type, mode=mode, convtype=convtype,
            spectral_norm=spectral_norm)
        # 创建第四个卷积块
        self.conv4 = conv_block(nf+3*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type,
            norm_type=norm_type, act_type=act_type, mode=mode, convtype=convtype,
            spectral_norm=spectral_norm)
        # 根据 mode 的不同选择最后一个卷积块的激活函数
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        # 创建最后一个卷积块
        self.conv5 = conv_block(nf+4*gc, nf, 3, stride, bias=bias, pad_type=pad_type,
            norm_type=norm_type, act_type=last_act, mode=mode, convtype=convtype,
            spectral_norm=spectral_norm)
    # 前向传播函数，接受输入 x
    def forward(self, x):
        # 使用第一个卷积层对输入 x 进行卷积操作，得到 x1
        x1 = self.conv1(x)
        # 将输入 x 和 x1 连接后，使用第二个卷积层对结果进行卷积操作，得到 x2
        x2 = self.conv2(torch.cat((x, x1), 1))
        # 如果存在 1x1 卷积层，将其应用于输入 x，并将结果与 x2 相加
        if self.conv1x1:
            x2 = x2 + self.conv1x1(x)
        # 将输入 x、x1 和 x2 连接后，使用第三个卷积层对结果进行卷积操作，得到 x3
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        # 将输入 x、x1、x2 和 x3 连接后，使用第四个卷积层对结果进行卷积操作，得到 x4
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        # 如果存在 1x1 卷积层，将其应用于 x2，并将结果与 x4 相加
        if self.conv1x1:
            x4 = x4 + x2
        # 将输入 x、x1、x2、x3 和 x4 连接后，使用第五个卷积层对结果进行卷积操作，得到 x5
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # 如果存在噪声层，对 x5 进行处理后返回结果
        if self.noise:
            return self.noise(x5.mul(0.2) + x)
        else:
            # 否则，对 x5 进行线性变换后返回结果
            return x5 * 0.2 + x
####################
# ESRGANplus
####################

# 定义一个添加高斯噪声的模块
class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=False):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, dtype=torch.float)

    # 前向传播函数，用于添加高斯噪声
    def forward(self, x):
        # 如果处于训练状态且噪声标准差不为0
        if self.training and self.sigma != 0:
            self.noise = self.noise.to(x.device)
            # 计算噪声的缩放系数
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            # 生成与输入相同大小的噪声并乘以缩放系数
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            # 将噪声添加到输入中
            x = x + sampled_noise
        return x

# 定义一个1x1卷积层
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


####################
# SRVGGNetCompact
####################

# 定义一个紧凑的VGG风格网络结构，用于超分辨率
# 该类从 https://github.com/xinntao/Real-ESRGAN 复制而来
class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution.
    This class is copied from https://github.com/xinntao/Real-ESRGAN
    """
    # 定义一个名为 SRVGGNetCompact 的类，继承自 nn.Module 类
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        # 调用父类的构造函数
        super(SRVGGNetCompact, self).__init__()
        # 初始化类的属性
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type
    
        # 创建一个空的 nn.ModuleList 用于存储网络的层
        self.body = nn.ModuleList()
        # 添加第一个卷积层
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # 添加第一个激活函数
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)
    
        # 构建网络主体结构
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            # 添加激活函数
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)
    
        # 添加最后一个卷积层
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # 上采样
        self.upsampler = nn.PixelShuffle(upscale)
    
    # 定义前向传播函数
    def forward(self, x):
        out = x
        # 遍历网络的每一层
        for i in range(0, len(self.body)):
            out = self.body[i](out)
    
        # 上采样
        out = self.upsampler(out)
        # 添加最近邻插值上采样的图像，使网络学习残差
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return out
####################
# Upsampler
####################

# 定义一个上采样类，用于上采样给定的多通道1D（时间）、2D（空间）或3D（体积）数据。
# 输入数据假设为`minibatch x channels x [optional depth] x [optional height] x width`的形式。
class Upsample(nn.Module):
    
    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=None):
        super(Upsample, self).__init__()
        # 如果scale_factor是元组，则将其转换为浮点数元组
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.size = size
        self.align_corners = align_corners

    def forward(self, x):
        # 使用nn.functional.interpolate函数进行插值操作
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

    def extra_repr(self):
        # 根据scale_factor或size返回额外的信息
        if self.scale_factor is not None:
            info = f'scale_factor={self.scale_factor}'
        else:
            info = f'size={self.size}'
        info += f', mode={self.mode}'
        return info


# 像素解缩
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.
    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.
    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    # 对x_view进行维度置换和重塑操作
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


# 像素洗牌块
def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True,
                        pad_type='zero', norm_type=None, act_type='relu', convtype='Conv2D'):
    """
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    """
    """
    # 创建卷积块，输入通道数为in_nc，输出通道数为out_nc * (upscale_factor ** 2)，使用指定的卷积核大小、步长、偏置等参数
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias,
                        pad_type=pad_type, norm_type=None, act_type=None, convtype=convtype)
    # 创建像素重排层，上采样因子为upscale_factor
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    # 如果指定了归一化类型，创建归一化层n，否则为None
    n = norm(norm_type, out_nc) if norm_type else None
    # 如果指定了激活函数类型，创建激活函数层a，否则为None
    a = act(act_type) if act_type else None
    # 返回包含卷积块、像素重排层、归一化层和激活函数层的序列模型
    return sequential(conv, pixel_shuffle, n, a)
# 定义上采样卷积块函数，用于上采样操作
def upconv_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True,
                pad_type='zero', norm_type=None, act_type='relu', mode='nearest', convtype='Conv2D'):
    """ Upconv layer """
    # 如果是 Conv3D 类型，则将 upscale_factor 转换为三元组
    upscale_factor = (1, upscale_factor, upscale_factor) if convtype == 'Conv3D' else upscale_factor
    # 创建上采样层
    upsample = Upsample(scale_factor=upscale_factor, mode=mode)
    # 创建卷积块
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias,
                        pad_type=pad_type, norm_type=norm_type, act_type=act_type, convtype=convtype)
    # 返回上采样和卷积块的序列
    return sequential(upsample, conv)

####################
# Basic blocks
####################

# 创建多个相同类型的块并堆叠在一起的函数
def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        basic_block (nn.module): nn.module class for basic block. (block)
        num_basic_block (int): number of blocks. (n_layers)
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

# 激活函数辅助函数
def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1, beta=1.0):
    """ activation helper """
    act_type = act_type.lower()
    # 根据激活函数类型创建对应的激活层
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type in ('leakyrelu', 'lrelu'):
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == 'tanh':  # [-1, 1] range output
        layer = nn.Tanh()
    elif act_type == 'sigmoid':  # [0, 1] range output
        layer = nn.Sigmoid()
    else:
        raise NotImplementedError(f'activation layer [{act_type}] is not found')
    return layer

# 定义一个恒等映射的模块
class Identity(nn.Module):
    def __init__(self, *kwargs):
        super(Identity, self).__init__()

    def forward(self, x, *kwargs):
        return x

# 返回一个规范化层
def norm(norm_type, nc):
    """ Return a normalization layer """
    # 将规范化类型转换为小写
    norm_type = norm_type.lower()
    # 如果规范化类型为'batch'，则创建批量归一化层
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    # 如果规范化类型为'instance'，则创建实例归一化层
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    # 如果规范化类型为'none'，则定义一个恒等函数作为规范化层
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    # 如果规范化类型不在上述三种情况中，则抛出未实现错误
    else:
        raise NotImplementedError(f'normalization layer [{norm_type}] is not found')
    # 返回相应的规范化层
    return layer
# 定义一个填充层辅助函数
def pad(pad_type, padding):
    """ padding layer helper """
    # 将填充类型转换为小写
    pad_type = pad_type.lower()
    # 如果填充大小为0，则返回None
    if padding == 0:
        return None
    # 根据填充类型选择不同的填充层
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    elif pad_type == 'zero':
        layer = nn.ZeroPad2d(padding)
    else:
        raise NotImplementedError(f'padding layer [{pad_type}] is not implemented')
    return layer


# 计算有效填充大小
def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


# 定义一个ShortcutBlock类，将子模块的输出与输入相加
class ShortcutBlock(nn.Module):
    """ Elementwise sum the output of a submodule to its input """
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        return 'Identity + \n|' + self.sub.__repr__().replace('\n', '\n|')


# 展开Sequential，去除nn.Sequential的包装
def sequential(*args):
    """ Flatten Sequential. It unwraps nn.Sequential. """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# 创建卷积块，包含填充、归一化、激活函数
def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA', convtype='Conv2D',
               spectral_norm=False):
    """ Conv layer with padding, normalization, activation """
    assert mode in ['CNA', 'NAC', 'CNAC'], f'Wrong conv mode [{mode}]'
    # 根据卷积核大小和膨胀率获取有效的填充大小
    padding = get_valid_padding(kernel_size, dilation)
    # 如果需要填充且填充类型不是'zero'，则进行填充，否则设为None
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    # 如果填充类型是'zero'，则将padding设为0
    padding = padding if pad_type == 'zero' else 0

    # 根据不同的卷积类型创建不同的卷积层
    if convtype=='PartialConv2D':
        # 导入PartialConv2d模块
        from torchvision.ops import PartialConv2d  # this is definitely not going to work, but PartialConv2d doesn't work anyway and this shuts up static analyzer
        # 创建PartialConv2d卷积层
        c = PartialConv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
               dilation=dilation, bias=bias, groups=groups)
    elif convtype=='DeformConv2D':
        # 导入DeformConv2d模块
        from torchvision.ops import DeformConv2d  # not tested
        # 创建DeformConv2d卷积层
        c = DeformConv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
               dilation=dilation, bias=bias, groups=groups)
    elif convtype=='Conv3D':
        # 创建3D卷积层
        c = nn.Conv3d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                dilation=dilation, bias=bias, groups=groups)
    else:
        # 创建2D卷积层
        c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                dilation=dilation, bias=bias, groups=groups)

    # 如果需要使用谱范数归一化，则对卷积层进行谱范数归一化
    if spectral_norm:
        c = nn.utils.spectral_norm(c)

    # 根据激活函数类型创建激活函数层
    a = act(act_type) if act_type else None
    # 如果模式中包含'CNA'，则创建规范化层并返回序列化的网络模块
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    # 如果模式是'NAC'，则根据条件创建规范化层和激活函数层，并返回序列化的网络模块
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)
```