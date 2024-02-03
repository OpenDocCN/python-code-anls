# `.\PaddleOCR\ppocr\modeling\backbones\kie_unet_sdmgr.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权;
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“原样”分发，不附带任何担保或条件，
# 无论是明示的还是暗示的。请查看许可证以获取详细信息
# 以及许可证下的特定语言管理权限和限制。

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn
import numpy as np
import cv2

# 定义模块的导出列表
__all__ = ["Kie_backbone"]

# 定义编码器类
class Encoder(nn.Layer):
    def __init__(self, num_channels, num_filters):
        super(Encoder, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2D(
            num_channels,
            num_filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        self.bn1 = nn.BatchNorm(num_filters, act='relu')

        # 第二个卷积层
        self.conv2 = nn.Conv2D(
            num_filters,
            num_filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        self.bn2 = nn.BatchNorm(num_filters, act='relu')

        # 最大池化层
        self.pool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

    # 前向传播函数
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x_pooled = self.pool(x)
        return x, x_pooled

# 定义解码器类
class Decoder(nn.Layer):
    # 初始化 Decoder 类，传入通道数和滤波器数量作为参数
    def __init__(self, num_channels, num_filters):
        # 调用父类 Decoder 的初始化方法
        super(Decoder, self).__init__()

        # 创建第一个卷积层，设置输入通道数、输出滤波器数量、卷积核大小、步长、填充和是否有偏置
        self.conv1 = nn.Conv2D(
            num_channels,
            num_filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        # 创建第一个批归一化层，设置输入通道数和激活函数为 relu
        self.bn1 = nn.BatchNorm(num_filters, act='relu')

        # 创建第二个卷积层，设置输入通道数、输出滤波器数量、卷积核大小、步长、填充和是否有偏置
        self.conv2 = nn.Conv2D(
            num_filters,
            num_filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        # 创建第二个批归一化层，设置输入通道数和激活函数为 relu
        self.bn2 = nn.BatchNorm(num_filters, act='relu')

        # 创建第三个卷积层，设置输入通道数、输出滤波器数量、卷积核大小、步长、填充和是否有偏置
        self.conv0 = nn.Conv2D(
            num_channels,
            num_filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=False)
        # 创建第三个批归一化层，设置输入通道数和激活函数为 relu
        self.bn0 = nn.BatchNorm(num_filters, act='relu')

    # 定义前向传播方法，接收上一层输入和当前层输入作为参数
    def forward(self, inputs_prev, inputs):
        # 对当前层输入进行第三个卷积操作
        x = self.conv0(inputs)
        # 对卷积结果进行第三个批归一化操作
        x = self.bn0(x)
        # 对结果进行双线性插值上采样，缩放因子为2，插值模式为双线性，不对齐角点
        x = paddle.nn.functional.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=False)
        # 将上一层输入和上采样结果在通道维度上拼接
        x = paddle.concat([inputs_prev, x], axis=1)
        # 对拼接结果进行第一个卷积操作
        x = self.conv1(x)
        # 对卷积结果进行第一个批归一化操作
        x = self.bn1(x)
        # 对结果进行第二个卷积操作
        x = self.conv2(x)
        # 对卷积结果进行第二个批归一化操作
        x = self.bn2(x)
        # 返回最终结果
        return x
class UNet(nn.Layer):
    # 定义 UNet 类，继承自 nn.Layer
    def __init__(self):
        # 初始化函数
        super(UNet, self).__init__()
        # 创建 Encoder 实例，输入通道数为 3，输出通道数为 16
        self.down1 = Encoder(num_channels=3, num_filters=16)
        # 创建 Encoder 实例，输入通道数为 16，输出通道数为 32
        self.down2 = Encoder(num_channels=16, num_filters=32)
        # 创建 Encoder 实例，输入通道数为 32，输出通道数为 64
        self.down3 = Encoder(num_channels=32, num_filters=64)
        # 创建 Encoder 实例，输入通道数为 64，输出通道数为 128
        self.down4 = Encoder(num_channels=64, num_filters=128)
        # 创建 Encoder 实例，输入通道数为 128，输出通道数为 256
        self.down5 = Encoder(num_channels=128, num_filters=256)

        # 创建 Decoder 实例，输入通道数为 32，输出通道数为 16
        self.up1 = Decoder(32, 16)
        # 创建 Decoder 实例，输入通道数为 64，输出通道数为 32
        self.up2 = Decoder(64, 32)
        # 创建 Decoder 实例，输入通道数为 128，输出通道数为 64
        self.up3 = Decoder(128, 64)
        # 创建 Decoder 实例，输入通道数为 256，输出通道数为 128
        self.up4 = Decoder(256, 128)
        # 输出通道数设为 16
        self.out_channels = 16

    def forward(self, inputs):
        # UNet 前向传播函数
        x1, _ = self.down1(inputs)
        _, x2 = self.down2(x1)
        _, x3 = self.down3(x2)
        _, x4 = self.down4(x3)
        _, x5 = self.down5(x4)

        x = self.up4(x4, x5)
        x = self.up3(x3, x)
        x = self.up2(x2, x)
        x = self.up1(x1, x)
        return x


class Kie_backbone(nn.Layer):
    # 定义 Kie_backbone 类，继承自 nn.Layer
    def __init__(self, in_channels, **kwargs):
        # 初始化函数
        super(Kie_backbone, self).__init__()
        # 输出通道数设为 16
        self.out_channels = 16
        # 创建 UNet 实例
        self.img_feat = UNet()
        # 创建最大池化层，核大小为 7
        self.maxpool = nn.MaxPool2D(kernel_size=7)

    def bbox2roi(self, bbox_list):
        # 将边界框列表转换为感兴趣区域列表
        rois_list = []
        rois_num = []
        for img_id, bboxes in enumerate(bbox_list):
            rois_num.append(bboxes.shape[0])
            rois_list.append(bboxes)
        rois = paddle.concat(rois_list, 0)
        rois_num = paddle.to_tensor(rois_num, dtype='int32')
        return rois, rois_num
    # 对输入数据进行预处理，包括将数据转换为 numpy 数组，提取相关信息，调整图像大小等操作
    def pre_process(self, img, relations, texts, gt_bboxes, tag, img_size):
        img, relations, texts, gt_bboxes, tag, img_size = img.numpy(
        ), relations.numpy(), texts.numpy(), gt_bboxes.numpy(), tag.numpy(
        ).tolist(), img_size.numpy()
        temp_relations, temp_texts, temp_gt_bboxes = [], [], []
        # 获取图像的高度和宽度
        h, w = int(np.max(img_size[:, 0])), int(np.max(img_size[:, 1]))
        # 裁剪图像到指定大小
        img = paddle.to_tensor(img[:, :, :h, :w])
        batch = len(tag)
        # 遍历每个样本
        for i in range(batch):
            num, recoder_len = tag[i][0], tag[i][1]
            # 将关系数据转换为 paddle 张量
            temp_relations.append(
                paddle.to_tensor(
                    relations[i, :num, :num, :], dtype='float32'))
            # 将文本数据转换为 paddle 张量
            temp_texts.append(
                paddle.to_tensor(
                    texts[i, :num, :recoder_len], dtype='float32'))
            # 将 ground truth 边界框数据转换为 paddle 张量
            temp_gt_bboxes.append(
                paddle.to_tensor(
                    gt_bboxes[i, :num, ...], dtype='float32'))
        # 返回处理后的图像和数据
        return img, temp_relations, temp_texts, temp_gt_bboxes

    # 前向传播函数
    def forward(self, inputs):
        # 获取输入数据
        img = inputs[0]
        relations, texts, gt_bboxes, tag, img_size = inputs[1], inputs[
            2], inputs[3], inputs[5], inputs[-1]
        # 对输入数据进行预处理
        img, relations, texts, gt_bboxes = self.pre_process(
            img, relations, texts, gt_bboxes, tag, img_size)
        # 提取图像特征
        x = self.img_feat(img)
        # 将 ground truth 边界框转换为 ROI
        boxes, rois_num = self.bbox2roi(gt_bboxes)
        # 使用 ROI Align 提取特征
        feats = paddle.vision.ops.roi_align(
            x, boxes, spatial_scale=1.0, output_size=7, boxes_num=rois_num)
        # 最大池化操作
        feats = self.maxpool(feats).squeeze(-1).squeeze(-1)
        # 返回结果
        return [relations, texts, feats]
```