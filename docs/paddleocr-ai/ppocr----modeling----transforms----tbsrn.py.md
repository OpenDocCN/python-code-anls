# `.\PaddleOCR\ppocr\modeling\transforms\tbsrn.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件按"原样"分发，
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 代码参考自：
# https://github.com/FudanVI/FudanOCR/blob/main/scene-text-telescope/model/tbsrn.py

import math
import warnings
import numpy as np
import paddle
from paddle import nn
import string

# 忽略警告信息
warnings.filterwarnings("ignore")

# 导入自定义模块
from .tps_spatial_transformer import TPSSpatialTransformer
from .stn import STN as STNHead
from .tsrn import GruBlock, mish, UpsampleBLock
from ppocr.modeling.heads.sr_rensnet_transformer import Transformer, LayerNorm, \
    PositionwiseFeedForward, MultiHeadedAttention

# 定义一个二维位置编码函数
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: 模型的维度
    :param height: 位置的高度
    :param width: 位置的宽度
    :return: d_model*height*width 的位置矩阵
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    # 创建一个全零矩阵作为位置编码矩阵
    pe = paddle.zeros([d_model, height, width])
    # 将模型维度减半
    d_model = int(d_model / 2)
    # 计算位置编码的分母
    div_term = paddle.exp(
        paddle.arange(0., d_model, 2, dtype='int64') * -(math.log(10000.0) / d_model))
    # 生成宽度方向的位置编码
    pos_w = paddle.arange(0., width, dtype='float32').unsqueeze(1)
    # 生成高度方向的位置编码
    pos_h = paddle.arange(0., height, dtype='float32').unsqueeze(1)
    # 将位置编码矩阵中偶数行的数值更新为正弦值
    pe[0:d_model:2, :, :] = paddle.sin(pos_w * div_term).transpose(
        [1, 0]).unsqueeze(1).tile([1, height, 1])
    # 将位置编码矩阵中奇数行的数值更新为余弦值
    pe[1:d_model:2, :, :] = paddle.cos(pos_w * div_term).transpose(
        [1, 0]).unsqueeze(1).tile([1, height, 1])
    # 将位置编码矩阵中偶数列的数值更新为正弦值
    pe[d_model::2, :, :] = paddle.sin(pos_h * div_term).transpose(
        [1, 0]).unsqueeze(2).tile([1, 1, width])
    # 将位置编码矩阵中奇数列的数值更新为余弦值
    pe[d_model + 1::2, :, :] = paddle.cos(pos_h * div_term).transpose(
        [1, 0]).unsqueeze(2).tile([1, 1, width])
    
    # 返回更新后的位置编码矩阵
    return pe
# 定义一个名为 FeatureEnhancer 的类，继承自 nn.Layer
class FeatureEnhancer(nn.Layer):
    # 初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        super(FeatureEnhancer, self).__init__()

        # 创建一个 MultiHeadedAttention 对象，设置参数 h=4, d_model=128, dropout=0.1
        self.multihead = MultiHeadedAttention(h=4, d_model=128, dropout=0.1)
        # 创建一个 LayerNorm 对象，设置特征数为 128
        self.mul_layernorm1 = LayerNorm(features=128)

        # 创建一个 PositionwiseFeedForward 对象，输入输出特征数均为 128
        self.pff = PositionwiseFeedForward(128, 128)
        # 创建一个 LayerNorm 对象，设置特征数为 128
        self.mul_layernorm3 = LayerNorm(features=128)

        # 创建一个线性层，输入特征数为 128，输出特征数为 64
        self.linear = nn.Linear(128, 64)

    # 前向传播方法
    def forward(self, conv_feature):
        '''
        text : (batch, seq_len, embedding_size)
        global_info: (batch, embedding_size, 1, 1)
        conv_feature: (batch, channel, H, W)
        '''
        # 获取 batch 大小
        batch = paddle.shape(conv_feature)[0]
        # 生成二维位置编码
        position2d = positionalencoding2d(64, 16, 64).cast('float32').unsqueeze(0).reshape([1, 64, 1024])
        # 复制位置编码以匹配 batch 大小
        position2d = position2d.tile([batch, 1, 1])
        # 在通道维度上拼接 conv_feature 和位置编码
        conv_feature = paddle.concat([conv_feature, position2d], 1)  # batch, 128(64+64), 32, 128
        # 调整维度顺序
        result = conv_feature.transpose([0, 2, 1])
        origin_result = result
        # 进行多头注意力计算并进行残差连接和 LayerNorm
        result = self.mul_layernorm1(origin_result + self.multihead(result, result, result, mask=None)[0])
        origin_result = result
        # 进行位置前馈网络计算并进行残差连接和 LayerNorm
        result = self.mul_layernorm3(origin_result + self.pff(result))
        # 线性变换
        result = self.linear(result)
        # 调整维度顺序
        return result.transpose([0, 2, 1])


# 定义一个名为 str_filt 的函数，用于过滤字符串中的字符
def str_filt(str_, voc_type):
    # 定义不同类型字符集合
    alpha_dict = {
        'digit': string.digits,
        'lower': string.digits + string.ascii_lowercase,
        'upper': string.digits + string.ascii_letters,
        'all': string.digits + string.ascii_letters + string.punctuation
    }
    # 如果 voc_type 为 'lower'，将字符串转换为小写
    if voc_type == 'lower':
        str_ = str_.lower()
    # 遍历字符串中的字符，如果不在指定字符集合中，则将其替换为空字符
    for char in str_:
        if char not in alpha_dict[voc_type]:
            str_ = str_.replace(char, '')
    # 将字符串转换为小写并返回
    str_ = str_.lower()
    return str_


# 定义一个名为 TBSRN 的类，继承自 nn.Layer
class TBSRN(nn.Layer):
    # 定义一个标签编码器方法，接受一个标签列表作为输入
    def label_encoder(self, label):
        # 获取标签列表的批量大小
        batch = len(label)

        # 计算每个标签的长度，并将其转换为张量
        length = [len(i) for i in label]
        length_tensor = paddle.to_tensor(length, dtype='int64')

        # 获取标签中最长的长度
        max_length = max(length)
        
        # 创建一个全零数组作为输入张量，形状为(batch, max_length)
        input_tensor = np.zeros((batch, max_length))
        for i in range(batch):
            for j in range(length[i] - 1):
                # 使用英语字典将标签编码为整数并填充到输入张量中
                input_tensor[i][j + 1] = self.english_dict[label[i][j]]

        # 将标签列表中的所有字符编码为整数并存储在text_gt中
        text_gt = []
        for i in label:
            for j in i:
                text_gt.append(self.english_dict[j])
        text_gt = paddle.to_tensor(text_gt, dtype='int64')

        # 将输入张量和text_gt转换为PaddlePaddle张量并返回
        input_tensor = paddle.to_tensor(input_tensor, dtype='int64')
        return length_tensor, input_tensor, text_gt
    # 定义前向传播函数，接受输入 x
    def forward(self, x):
        # 初始化输出字典
        output = {}
        # 如果处于推断模式
        if self.infer_mode:
            # 将输入 x 存入输出字典中的 "lr_img" 键
            output["lr_img"] = x
            # y 等于 x
            y = x
        else:
            # 将输入 x 的第一个元素存入输出字典中的 "lr_img" 键
            output["lr_img"] = x[0]
            # 将输入 x 的第二个元素存入输出字典中的 "hr_img" 键
            output["hr_img"] = x[1]
            # y 等于 x 的第一个元素
            y = x[0]
        # 如果启用空间变换网络并且处于训练模式
        if self.stn and self.training:
            # 获取空间变换网络的控制点
            _, ctrl_points_x = self.stn_head(y)
            # 对输入 y 进行空间变换
            y, _ = self.tps(y, ctrl_points_x)
        # 创建字典，存储每个块的输出
        block = {'1': self.block1(y)}
        # 循环遍历每个残差块
        for i in range(self.srb_nums + 1):
            # 将每个残差块的输出存入字典中
            block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        # 计算最后一个残差块的输出
        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            ((block['1'] + block[str(self.srb_nums + 2)]))

        # 对最终的超分辨率图像进行 tanh 激活
        sr_img = paddle.tanh(block[str(self.srb_nums + 3)])
        # 将超分辨率图像存入输出字典中的 "sr_img" 键
        output["sr_img"] = sr_img

        # 如果处于训练模式
        if self.training:
            # 获取输入 x 的第二个元素作为高分辨率图像
            hr_img = x[1]

            # 添加变换器
            # 对输入 x 的第三个元素进行处理，生成标签
            label = [str_filt(i, 'lower') + '-' for i in x[2]]
            # 将标签编码为张量
            length_tensor, input_tensor, text_gt = self.label_encoder(label)
            # 使用变换器处理高分辨率图像
            hr_pred, word_attention_map_gt, hr_correct_list = self.transformer(
                hr_img, length_tensor, input_tensor)
            # 使用变换器处理超分辨率图像
            sr_pred, word_attention_map_pred, sr_correct_list = self.transformer(
                sr_img, length_tensor, input_tensor)
            # 将高分辨率图像及相关信息存入输出字典中
            output["hr_img"] = hr_img
            output["hr_pred"] = hr_pred
            output["text_gt"] = text_gt
            output["word_attention_map_gt"] = word_attention_map_gt
            output["sr_pred"] = sr_pred
            output["word_attention_map_pred"] = word_attention_map_pred

        # 返回输出字典
        return output
class RecurrentResidualBlock(nn.Layer):
    # 定义一个循环残差块的类，继承自 nn.Layer
    def __init__(self, channels):
        # 初始化函数，接受通道数作为参数
        super(RecurrentResidualBlock, self).__init__()
        # 调用父类的初始化函数
        self.conv1 = nn.Conv2D(channels, channels, kernel_size=3, padding=1)
        # 创建一个卷积层，输入和输出通道数均为channels，卷积核大小为3，padding为1
        self.bn1 = nn.BatchNorm2D(channels)
        # 创建一个二维批归一化层，通道数为channels
        self.gru1 = GruBlock(channels, channels)
        # 创建一个 GruBlock，输入和输出通道数均为channels
        self.prelu = mish()
        # 创建一个激活函数 mish
        self.conv2 = nn.Conv2D(channels, channels, kernel_size=3, padding=1)
        # 创建第二个卷积层，输入和输出通道数均为channels，卷积核大小为3，padding为1
        self.bn2 = nn.BatchNorm2D(channels)
        # 创建第二个二维批归一化层，通道数为channels
        self.gru2 = GruBlock(channels, channels)
        # 创建第二个 GruBlock，输入和输出通道数均为channels
        self.feature_enhancer = FeatureEnhancer()
        # 创建一个特征增强器

        for p in self.parameters():
            # 遍历模型的参数
            if p.dim() > 1:
                # 如果参数的维度大于1
                paddle.nn.initializer.XavierUniform(p)
                # 使用 XavierUniform 初始化该参数

    def forward(self, x):
        # 定义前向传播函数，接受输入x
        residual = self.conv1(x)
        # 使用第一个卷积层处理输入x，得到残差
        residual = self.bn1(residual)
        # 对残差进行批归一化
        residual = self.prelu(residual)
        # 使用激活函数处理残差
        residual = self.conv2(residual)
        # 使用第二个卷积层处理残差
        residual = self.bn2(residual)
        # 对残差进行批归一化

        size = paddle.shape(residual)
        # 获取残差的形状信息
        residual = residual.reshape([size[0], size[1], -1])
        # 将残差重塑为 [size[0], size[1], -1] 的形状
        residual = self.feature_enhancer(residual)
        # 使用特征增强器处理残差
        residual = residual.reshape([size[0], size[1], size[2], size[3])
        # 将处理后的残差重塑为原始形状
        return x + residual
        # 返回输入x与处理后的残差的和作为输出
```