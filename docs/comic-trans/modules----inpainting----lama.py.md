# `.\comic-translate\modules\inpainting\lama.py`

```py
# 导入必要的库
import os
import cv2
import numpy as np
import torch

# 导入 PyTorch 中的神经网络模块和张量类型
import torch.nn as nn
from torch import Tensor

# 导入自定义的模块 FFC_BN_ACT
from .ffc import FFC_BN_ACT

# 导入自定义工具函数和类
from ..utils.inpainting import (
    norm_img,
    get_cache_path_by_url,
    load_jit_model,
)
from .base import InpaintModel
from .schema import Config

# 设置 LAMA_MODEL_URL 和 LAMA_MODEL_MD5 的默认值
LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://huggingface.co/dreMaz/AnimeMangaInpainting/resolve/main/lama_large_512px.ckpt",
)
LAMA_MODEL_MD5 = os.environ.get("LAMA_MODEL_MD5", "c09472d8ff584452a2c4529af520fe0b")

# 定义 InpaintModel 的子类 LaMa，用于图像修复任务
class LaMa(InpaintModel):
    name = "lama"  # 模型名称为 "lama"
    pad_mod = 8  # 填充模数为 8

    # 初始化模型函数，加载 LAMA 模型
    def init_model(self, device, **kwargs):
        self.model = load_lama_model(model_path='models/inpainting/lama_large_512px.ckpt', device=device, large_arch=True)
        # self.model = load_jit_model(LAMA_MODEL_URL, device, LAMA_MODEL_MD5).eval()

    # 检查模型是否已下载
    @staticmethod
    def is_downloaded() -> bool:
        return os.path.exists(get_cache_path_by_url(LAMA_MODEL_URL))

    # 前向传播函数，输入图像和掩码，输出修复后的图像
    def forward(self, image, mask, config: Config):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W]
        return: BGR IMAGE
        """
        # 对输入图像和掩码进行归一化
        image = norm_img(image)
        mask = norm_img(mask)

        # 将掩码转换为二进制掩码
        mask = (mask > 0) * 1
        # 将图像和掩码转换为 PyTorch 张量并移动到指定设备上
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        # 使用模型进行图像修复
        inpainted_image = self.model(image, mask)

        # 处理修复后的图像，转换为 BGR 格式并返回
        cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        return cur_res
    
# 用于加载 LAMA 模型而不使用 JIT 编译
def get_activation(kind='tanh'):
    if kind == 'tanh':
        return nn.Tanh()
    if kind == 'sigmoid':
        return nn.Sigmoid()
    if kind is False:
        return nn.Identity()
    raise ValueError(f'Unknown activation kind {kind}')

# 定义 FFCResnetBlock 类，继承自 nn.Module
class FFCResnetBlock(nn.Module):
    # 初始化函数，用于创建一个卷积神经网络模型
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 inline=False, **conv_kwargs):
        super().__init__()
        # 创建第一个卷积层对象，包括特征图维度、卷积核大小、填充类型、扩张率等参数
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        # 创建第二个卷积层对象，参数与第一个卷积层类似
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        # 是否使用内联模式的标志
        self.inline = inline

    # 前向传播函数，定义了数据从输入到输出的流程
    def forward(self, x):
        # 如果使用内联模式
        if self.inline:
            # 将输入张量按列切片，分为局部特征和全局特征
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            # 如果不使用内联模式，将输入张量分为局部特征和全局特征
            x_l, x_g = x if type(x) is tuple else (x, 0)

        # 保存初始的局部特征和全局特征
        id_l, id_g = x_l, x_g

        # 第一次卷积操作，处理局部特征和全局特征
        x_l, x_g = self.conv1((x_l, x_g))

        # 第二次卷积操作，处理更新后的局部特征和全局特征
        x_l, x_g = self.conv2((x_l, x_g))

        # 将初始的局部特征和全局特征与更新后的局部特征和全局特征相加
        x_l, x_g = id_l + x_l, id_g + x_g

        # 将最终的局部特征和全局特征作为输出
        out = x_l, x_g

        # 如果使用内联模式，则将局部特征和全局特征在列维度上连接起来
        if self.inline:
            out = torch.cat(out, dim=1)

        # 返回最终的输出结果
        return out
class ConcatTupleLayer(nn.Module):
    # 定义一个继承自 nn.Module 的类 ConcatTupleLayer，用于处理元组输入的连接层

    def forward(self, x):
        # 定义前向传播函数，接收输入 x

        assert isinstance(x, tuple)
        # 断言 x 是一个元组

        x_l, x_g = x
        # 将元组 x 拆解为两部分 x_l 和 x_g

        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        # 断言 x_l 或者 x_g 是张量（Tensor）

        if not torch.is_tensor(x_g):
            # 如果 x_g 不是张量，返回 x_l
            return x_l
        
        # 如果 x_g 是张量，则将 x_l 和 x_g 沿着第一个维度连接起来，并返回结果
        return torch.cat(x, dim=1)


class FFCResNetGenerator(nn.Module):
    # 定义一个继承自 nn.Module 的类 FFCResNetGenerator，用于实现一个生成器模型


这些注释将代码中每个语句的作用进行了解释，符合给定的注意事项和示例格式要求。
    # 初始化函数，设置模型的各种参数和层次结构
    def __init__(self, input_nc=4, output_nc=3, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={}, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        # 确保 n_blocks 大于等于 0
        assert (n_blocks >= 0)
        # 调用父类的初始化函数
        super().__init__()

        # 初始化模型的层次列表，包括一个 ReflectionPad2d 层和一个初始的 FFC_BN_ACT 层
        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        # 创建下采样部分的卷积层
        for i in range(n_downsampling):
            mult = 2 ** i
            # 如果是最后一层下采样，使用特定的 conv_kwargs 设置
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            # 添加一个 FFC_BN_ACT 层作为下采样层
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        # 计算最终下采样后的特征数
        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        # 添加 ResNet 块
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            model += [cur_resblock]

        # 添加一个 ConcatTupleLayer 层
        model += [ConcatTupleLayer()]

        ### upsample
        # 创建上采样部分的卷积层
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            # 添加一个转置卷积层、规范化层和激活函数作为上采样层
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        # 如果指定了 out_ffc，则添加一个额外的 FFCResnetBlock 层
        if out_ffc:
            model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        # 最后添加一个 ReflectionPad2d 层和一个输出层
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        # 如果指定了 add_out_act，则添加一个输出激活函数层
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        
        # 将所有层次整合成一个序列化的神经网络模型
        self.model = nn.Sequential(*model)
    # 定义前向传播方法，接受输入的图像、遮罩、相对位置和方向信息，返回张量
    def forward(self, img, mask, rel_pos=None, direct=None) -> Tensor:
        # 将图像和其遮罩拼接在一起，形成输入模型的张量
        masked_img = torch.cat([img * (1 - mask), mask], dim=1)
        # 如果没有相对位置信息，直接使用模型进行处理并返回结果
        if rel_pos is None:
            return self.model(masked_img)
        else:
            # 否则，通过模型的前两层处理得到局部特征 x_l 和全局特征 x_g
            x_l, x_g = self.model[:2](masked_img)
            # 将局部特征 x_l 转换为 float32 类型
            x_l = x_l.to(torch.float32)
            # 添加相对位置信息到局部特征 x_l
            x_l += rel_pos
            # 添加方向信息到局部特征 x_l
            x_l += direct
            # 将更新后的局部特征 x_l 和全局特征 x_g 传递给模型的后续层处理，并返回结果
            return self.model[2:]((x_l, x_g))
# 设置模型参数是否需要梯度计算
def set_requires_grad(module, value):
    # 遍历模型的所有参数
    for param in module.parameters():
        param.requires_grad = value


class MaskedSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        # 创建正弦和余弦位置编码
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # 设置为不需要梯度以避免 pytorch-1.8+ 的错误
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        # 分别填充正弦和余弦位置编码到权重矩阵
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids):
        """`input_ids` is expected to be [bsz x seqlen]."""
        return super().forward(input_ids)


class MultiLabelEmbedding(nn.Module):
    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_positions, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化权重参数为正态分布
        nn.init.normal_(self.weight)

    def forward(self, input_ids):
        # 计算多标签嵌入的输出
        # input_ids:[B,HW,4](onehot)
        out = torch.matmul(input_ids, self.weight)  # [B,HW,dim]
        return out


class LamaFourier:
    def __init__(self, large_arch: bool = False) -> None:
        # 根据是否使用大型架构确定块数
        n_blocks = 9
        if large_arch:
            n_blocks = 18
        
        # 初始化生成器网络
        self.generator = FFCResNetGenerator(4, 3, add_out_act='sigmoid', 
                            n_blocks = n_blocks,
                            init_conv_kwargs={
                            'ratio_gin': 0,
                            'ratio_gout': 0,
                            'enable_lfu': False
                        }, downsample_conv_kwargs={
                            'ratio_gin': 0,
                            'ratio_gout': 0,
                            'enable_lfu': False
                        }, resnet_conv_kwargs={
                            'ratio_gin': 0.75,
                            'ratio_gout': 0.75,
                            'enable_lfu': False
                        }, 
                    )
        
        self.inpaint_only = False

    def to(self, device):
        # 将生成器网络移动到指定设备上
        self.generator.to(device)

    def eval(self):
        # 设置为只进行修复输入
        self.inpaint_only = True
        # 将生成器网络设置为评估模式
        self.generator.eval()
        return self
    # 定义一个方法，用于处理输入的图像和掩码，生成预测的图像
    def __call__(self, img: Tensor, mask: Tensor, rel_pos=None, direct=None):

        # 将相对位置和方向置为 None
        rel_pos, direct = None, None
        # 使用生成器模型生成预测的图像
        predicted_img = self.generator(img, mask, rel_pos, direct)

        # 如果仅需要修复缺失部分，则返回修复后的图像
        if self.inpaint_only:
            return predicted_img * mask + (1 - mask) * img

        # 否则返回预测的图像及其它相关信息的字典
        return {
                'predicted_img': predicted_img
            }

    # 定义一个方法，用于加载掩码的位置编码
    def load_masked_position_encoding(self, mask):
        # 将掩码转换为 uint8 类型并扩展到 255 值
        mask = (mask * 255).astype(np.uint8)
        # 创建一个全为 1 的 3x3 浮点数数组
        ones_filter = np.ones((3, 3), dtype=np.float32)
        # 定义四个方向的滤波器
        d_filter1 = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32)
        d_filter2 = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=np.float32)
        d_filter3 = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.float32)
        d_filter4 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float32)
        # 设置字符串大小和位置数量
        str_size = 256
        pos_num = 128

        # 复制原始掩码并将其归一化
        ori_mask = mask.copy()
        ori_h, ori_w = ori_mask.shape[0:2]
        ori_mask = ori_mask / 255
        # 将掩码调整大小到指定的字符串大小
        mask = cv2.resize(mask, (str_size, str_size), interpolation=cv2.INTER_AREA)
        mask[mask > 0] = 255
        h, w = mask.shape[0:2]
        mask3 = mask.copy()
        mask3 = 1. - (mask3 / 255.0)
        # 创建位置和方向的空数组
        pos = np.zeros((h, w), dtype=np.int32)
        direct = np.zeros((h, w, 4), dtype=np.int32)
        i = 0

        # 如果掩码的最大值大于 0，则执行以下循环
        if mask3.max() > 0:
            # 否则会导致无限循环
        
            while np.sum(1 - mask3) > 0:
                i += 1
                # 应用全为 1 的滤波器，得到当前掩码的结果
                mask3_ = cv2.filter2D(mask3, -1, ones_filter)
                mask3_[mask3_ > 0] = 1
                sub_mask = mask3_ - mask3
                # 将新生成的掩码结果应用到位置数组中
                pos[sub_mask == 1] = i

                m = cv2.filter2D(mask3, -1, d_filter1)
                m[m > 0] = 1
                m = m - mask3
                direct[m == 1, 0] = 1

                m = cv2.filter2D(mask3, -1, d_filter2)
                m[m > 0] = 1
                m = m - mask3
                direct[m == 1, 1] = 1

                m = cv2.filter2D(mask3, -1, d_filter3)
                m[m > 0] = 1
                m = m - mask3
                direct[m == 1, 2] = 1

                m = cv2.filter2D(mask3, -1, d_filter4)
                m[m > 0] = 1
                m = m - mask3
                direct[m == 1, 3] = 1

                mask3 = mask3_

        # 复制位置数组作为绝对位置数组
        abs_pos = pos.copy()
        # 计算相对位置数组，将其标准化到 0~1 的范围，可能大于 1
        rel_pos = pos / (str_size / 2)  # to 0~1 maybe larger than 1
        # 将相对位置数组映射到指定范围内的整数值
        rel_pos = (rel_pos * pos_num).astype(np.int32)
        rel_pos = np.clip(rel_pos, 0, pos_num - 1)

        # 如果原始宽高与当前宽高不同，则调整位置和方向数组的大小
        if ori_w != w or ori_h != h:
            rel_pos = cv2.resize(rel_pos, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
            rel_pos[ori_mask == 0] = 0
            direct = cv2.resize(direct, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
            direct[ori_mask == 0, :] = 0

        # 返回相对位置数组、绝对位置数组和方向数组
        return rel_pos, abs_pos, direct
# 加载 LAMA 模型的函数，根据给定的模型路径、设备和是否使用大型架构进行加载
def load_lama_model(model_path, device, large_arch: bool = False) -> LamaFourier:
    # 创建一个 LamaFourier 模型实例，根据 large_arch 参数确定是否使用大型架构
    model = LamaFourier(large_arch=large_arch)
    # 使用 torch.load 加载模型的状态字典，指定 'cpu' 作为设备位置
    sd = torch.load(model_path, map_location='cpu')
    # 加载模型的生成器的状态字典
    model.generator.load_state_dict(sd['gen_state_dict'])
    # 将模型设置为评估模式，并将其移动到指定的设备上
    model.eval().to(device)
    # 返回加载好的模型实例
    return model
```