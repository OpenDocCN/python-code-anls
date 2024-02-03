# `.\PaddleOCR\ppocr\modeling\backbones\table_master_resnet.py`

```py
# 版权声明，告知代码版权归属于 PaddlePaddle 作者
#
# 根据 Apache 许可证 2.0 版本授权使用此文件
# 只有在遵守许可证的情况下才能使用此文件
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是基于“按原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的具体语言
"""
# 引用的代码来源链接
# https://github.com/JiaquanYe/TableMASTER-mmocr/blob/master/mmocr/models/textrecog/backbones/table_resnet_extra.py

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# 定义 BasicBlock 类，用于构建 ResNet 的基本块
class BasicBlock(nn.Layer):
    # 扩展系数为 1，用于计算输出通道数
    expansion = 1
    # 定义 BasicBlock 类，继承自 nn.Module 类
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 gcb_config=None):
        # 调用父类的构造函数
        super(BasicBlock, self).__init__()
        # 创建第一个卷积层，输入通道数为 inplanes，输出通道数为 planes
        self.conv1 = nn.Conv2D(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias_attr=False)
        # 创建第一个批归一化层，输入通道数为 planes
        self.bn1 = nn.BatchNorm2D(planes, momentum=0.9)
        # 创建 ReLU 激活函数层
        self.relu = nn.ReLU()
        # 创建第二个卷积层，输入通道数为 planes，输出通道数为 planes
        self.conv2 = nn.Conv2D(
            planes, planes, kernel_size=3, stride=1, padding=1, bias_attr=False)
        # 创建第二个批归一化层，输入通道数为 planes
        self.bn2 = nn.BatchNorm2D(planes, momentum=0.9)
        # 保存 downsample、stride 和 gcb_config 参数
        self.downsample = downsample
        self.stride = stride
        self.gcb_config = gcb_config
    
        # 如果存在 gcb_config 参数
        if self.gcb_config is not None:
            # 获取 gcb_config 参数中的 ratio、headers、att_scale 和 fusion_type
            gcb_ratio = gcb_config['ratio']
            gcb_headers = gcb_config['headers']
            att_scale = gcb_config['att_scale']
            fusion_type = gcb_config['fusion_type']
            # 创建 MultiAspectGCAttention 上下文注意力模块
            self.context_block = MultiAspectGCAttention(
                inplanes=planes,
                ratio=gcb_ratio,
                headers=gcb_headers,
                att_scale=att_scale,
                fusion_type=fusion_type)
    
    # 定义前向传播函数
    def forward(self, x):
        # 保存输入 x 作为残差连接
        residual = x
    
        # 第一个卷积层、批归一化层和 ReLU 激活函数层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
    
        # 第二个卷积层和批归一化层
        out = self.conv2(out)
        out = self.bn2(out)
    
        # 如果存在 gcb_config 参数，应用上下文注意力模块
        if self.gcb_config is not None:
            out = self.context_block(out)
    
        # 如果存在 downsample 参数，对输入 x 进行下采样
        if self.downsample is not None:
            residual = self.downsample(x)
    
        # 将残差连接和当前输出相加，并经过 ReLU 激活函数
        out += residual
        out = self.relu(out)
    
        # 返回输出结果
        return out
# 获取指定层的 GCB 配置信息，如果配置为空或指定层不存在配置，则返回 None
def get_gcb_config(gcb_config, layer):
    if gcb_config is None or not gcb_config['layers'][layer]:
        return None
    else:
        return gcb_config

# 定义一个 TableResNetExtra 类，用于创建 ResNet 模型的额外层
class TableResNetExtra(nn.Layer):
    # 创建 ResNet 模型的一个层
    def _make_layer(self, block, planes, blocks, stride=1, gcb_config=None):
        downsample = None
        # 如果步长不为 1 或输入通道数不等于输出通道数，则创建下采样层
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion), )

        layers = []
        # 添加当前层
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                gcb_config=gcb_config))
        self.inplanes = planes * block.expansion
        # 添加额外的 blocks 层
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # 前向传播函数
    def forward(self, x):
        f = []
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.maxpool1(x)
        x = self.layer1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        f.append(x)

        x = self.maxpool2(x)
        x = self.layer2(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        f.append(x)

        x = self.maxpool3(x)

        x = self.layer3(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.layer4(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        f.append(x)
        return f

# 定义一个 MultiAspectGCAttention 类
class MultiAspectGCAttention(nn.Layer):
    # 对输入张量进行空间池化操作
    def spatial_pool(self, x):
        # 获取输入张量的形状信息：batch大小、通道数、高度、宽度
        batch, channel, height, width = x.shape
        # 如果池化类型为'att'，则执行以下操作
        if self.pooling_type == 'att':
            # 重塑输入张量的形状：[N*headers, C', H , W]，其中 C = headers * C'
            x = x.reshape([
                batch * self.headers, self.single_header_inplanes, height, width
            ])
            # 保存重塑后的输入张量
            input_x = x

            # 重塑输入张量的形状：[N*headers, C', H * W]，其中 C = headers * C'
            input_x = input_x.reshape([
                batch * self.headers, self.single_header_inplanes,
                height * width
            ])

            # 在第1维度上增加一个维度：[N*headers, 1, C', H * W]
            input_x = input_x.unsqueeze(1)
            # 使用卷积操作生成上下文掩码：[N*headers, 1, H, W]
            context_mask = self.conv_mask(x)
            # 重塑上下文掩码的形状：[N*headers, 1, H * W]
            context_mask = context_mask.reshape(
                [batch * self.headers, 1, height * width])

            # 如果启用了缩放方差且headers大于1，则对上下文掩码进行缩放
            if self.att_scale and self.headers > 1:
                context_mask = context_mask / paddle.sqrt(
                    self.single_header_inplanes)

            # 对上下文掩码进行softmax操作：[N*headers, 1, H * W]
            context_mask = self.softmax(context_mask)

            # 在最后一个维度上增加一个维度：[N*headers, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # 计算上下文信息：[N*headers, 1, C', 1] = [N*headers, 1, C', H * W] * [N*headers, 1, H * W, 1]
            context = paddle.matmul(input_x, context_mask)

            # 重塑上下文信息的形状：[N, headers * C', 1, 1]
            context = context.reshape(
                [batch, self.headers * self.single_header_inplanes, 1, 1])
        else:
            # 使用平均池化操作生成上下文信息：[N, C, 1, 1]
            context = self.avg_pool(x)

        # 返回上下文信息
        return context
    # 定义前向传播函数，接受输入 x
    def forward(self, x):
        # 对输入 x 进行空间池化操作，得到上下文信息，形状为 [N, C, 1, 1]
        context = self.spatial_pool(x)

        # 将输出初始化为输入 x
        out = x

        # 根据融合类型进行不同的操作
        if self.fusion_type == 'channel_mul':
            # 对上下文信息进行通道乘法操作，形状为 [N, C, 1, 1]
            channel_mul_term = F.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        elif self.fusion_type == 'channel_add':
            # 对上下文信息进行通道加法操作，形状为 [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        else:
            # 对上下文信息进行通道拼接操作，形状为 [N, C, 1, 1]
            channel_concat_term = self.channel_concat_conv(context)

            # 使用拼接
            # 获取通道拼接后的特征维度 C1
            _, C1, _, _ = channel_concat_term.shape
            # 获取输入 x 的维度信息 N, C2, H, W
            N, C2, H, W = out.shape

            # 在通道维度上拼接输入 x 和通道拼接后的特征
            out = paddle.concat(
                [out, channel_concat_term.expand([-1, -1, H, W])], axis=1)
            # 经过卷积操作
            out = self.cat_conv(out)
            # 进行 Layer Norm 操作
            out = F.layer_norm(out, [self.inplanes, H, W])
            # 使用 ReLU 激活函数
            out = F.relu(out)

        # 返回处理后的输出
        return out
```