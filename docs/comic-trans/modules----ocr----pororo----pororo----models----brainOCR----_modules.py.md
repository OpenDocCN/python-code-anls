# `.\comic-translate\modules\ocr\pororo\pororo\models\brainOCR\_modules.py`

```py
# 导入命名元组模块
from collections import namedtuple

# 导入必要的库
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import models

# 定义所有的 VGG 模型名称
all = [
'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
'vgg19_bn', 'vgg19',
]

# 定义预训练模型的下载链接
model_urls = {
'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

# 根据CUDA是否可用选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化网络层的权重
def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)  # 使用Xavier初始化卷积层的权重
            if m.bias is not None:
                m.bias.data.zero_()  # 如果存在偏置，则初始化为零
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)  # 批归一化层的权重初始化为1
            m.bias.data.zero_()  # 批归一化层的偏置初始化为零
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)  # 线性层的权重使用正态分布初始化
            m.bias.data.zero_()  # 线性层的偏置初始化为零

# 定义 Vgg16BN 类，继承自 torch.nn.Module
class Vgg16BN(torch.nn.Module):
    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super(Vgg16BN, self).__init__()
        
        # 修改预训练模型的 URL，使用 HTTP 而不是 HTTPS
        model_urls["vgg16_bn"] = model_urls["vgg16_bn"].replace(
            "https://", "http://")
        
        # 加载预训练的 VGG16-BN 模型的特征提取部分
        vgg_pretrained_features = models.vgg16_bn(
            pretrained=pretrained).features
        
        # 初始化网络的不同层（slice1到slice4）为空的顺序容器
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        # 将预训练模型的卷积层按照指定的范围加入到不同的层中
        for x in range(12):  # conv2_2，添加第1段特征提取器中的卷积层
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):  # conv3_3，添加第2段特征提取器中的卷积层
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):  # conv4_3，添加第3段特征提取器中的卷积层
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):  # conv5_3，添加第4段特征提取器中的卷积层
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        # 定义第5段特征提取器，包括最大池化层和两个卷积层（fc6和fc7）
        self.slice5 = torch.nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1),
        )

        # 如果未使用预训练模型，则对各段特征提取器进行权重初始化
        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())

        # 对第5段特征提取器进行权重初始化，因为没有预训练模型可用于 fc6 和 fc7
        init_weights(
            self.slice5.modules())

        # 如果需要冻结（不更新）第1段特征提取器的参数（仅限第一个卷积层）
        if freeze:
            for param in self.slice1.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)  # 经过第1段特征提取器
        h_relu2_2 = h
        h = self.slice2(h)  # 经过第2段特征提取器
        h_relu3_2 = h
        h = self.slice3(h)  # 经过第3段特征提取器
        h_relu4_3 = h
        h = self.slice4(h)  # 经过第4段特征提取器
        h_relu5_3 = h
        h = self.slice5(h)  # 经过第5段特征提取器（包括 fc6 和 fc7）
        h_fc7 = h
        # 返回经过 VGG16-BN 模型各段特征提取器后的输出
        vgg_outputs = namedtuple(
            "VggOutputs", ["fc7", "relu5_3", "relu4_3", "relu3_2", "relu2_2"])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out
class VGGFeatureExtractor(nn.Module):
    """ FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf) """

    def __init__(self, n_input_channels: int = 1, n_output_channels: int = 512):
        super(VGGFeatureExtractor, self).__init__()

        self.output_channel = [
            int(n_output_channels / 8),
            int(n_output_channels / 4),
            int(n_output_channels / 2),
            n_output_channels,
        ]  # [64, 128, 256, 512]
        
        # 定义卷积神经网络的层次结构
        self.ConvNet = nn.Sequential(
            nn.Conv2d(n_input_channels, self.output_channel[0], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64x16x50
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 128x8x25
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1),
            nn.ReLU(True),  # 256x8x25
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 256x4x25
            nn.Conv2d(
                self.output_channel[2],
                self.output_channel[3],
                3,
                1,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),  # 512x4x25
            nn.Conv2d(
                self.output_channel[3],
                self.output_channel[3],
                3,
                1,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 512x2x25
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0),
            nn.ReLU(True),
        )  # 512x1x24

    def forward(self, x):
        return self.ConvNet(x)


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(BidirectionalLSTM, self).__init__()
        
        # 初始化双向 LSTM 层
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        
        # 线性层将 LSTM 输出映射到最终输出尺寸
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        x : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(
            x
        )  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class ResNetFeatureExtractor(nn.Module):
    """
    FeatureExtractor of FAN
    (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf)

    """
    # 定义 ResNetFeatureExtractor 类，继承自父类 nn.Module
    def __init__(self, n_input_channels: int = 1, n_output_channels: int = 512):
        # 调用父类构造函数初始化
        super(ResNetFeatureExtractor, self).__init__()
        # 创建 ResNet 模型，设置输入通道数和输出通道数，并使用 BasicBlock 作为基本构建块
        # [1, 2, 5, 3] 表示每个阶段的残差块数量，对应 ResNet 的结构
        self.ConvNet = ResNet(
            n_input_channels,
            n_output_channels,
            BasicBlock,
            [1, 2, 5, 3],
        )
    
    # 定义前向传播函数 forward，接收输入 inputs
    def forward(self, inputs):
        # 将输入 inputs 传递给 self.ConvNet 进行前向计算，并返回计算结果
        return self.ConvNet(inputs)
class BasicBlock(nn.Module):
    expansion = 1  # 定义模块的扩展因子，默认为1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample=None):
        super(BasicBlock, self).__init__()
        # 第一个3x3卷积层
        self.conv1 = self._conv3x3(inplanes, planes)
        # Batch Normalization层
        self.bn1 = nn.BatchNorm2d(planes)
        # 第二个3x3卷积层
        self.conv2 = self._conv3x3(planes, planes)
        # 另一个Batch Normalization层
        self.bn2 = nn.BatchNorm2d(planes)
        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        "3x3 convolution with padding"
        # 返回一个3x3卷积层，不带偏置项
        return nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

    def forward(self, x):
        residual = x

        # 第一层卷积、BN、ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二层卷积、BN
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果有下采样，对输入进行下采样
        if self.downsample is not None:
            residual = self.downsample(x)

        # 残差连接并应用ReLU
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        n_input_channels: int,
        n_output_channels: int,
        block,
        layers,
    ):
        super(ResNet, self).__init__()
        self.inplanes = 64
        # 第一层卷积、BN、ReLU
        self.conv0_1 = nn.Conv2d(n_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn0_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 池化层
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet主体结构的四个层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
        self.fc = nn.Linear(512 * block.expansion, n_output_channels)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 如果需要下采样，构建下采样模块
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # 第一个块
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        # 后续块
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 前向传播过程
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class TpsSpatialTransformerNetwork(nn.Module):
    """ Rectification Network of RARE, namely TPS based STN """
    def __init__(self, F, I_size, I_r_size, I_channel_num: int = 1):
        """Based on RARE TPS
        input:
            batch_I: Batch Input Image [batch_size x I_channel_num x I_height x I_width]
            I_size : (height, width) of the input image I
            I_r_size : (height, width) of the rectified image I_r
            I_channel_num : the number of channels of the input image I
        output:
            batch_I_r: rectified image [batch_size x I_channel_num x I_r_height x I_r_width]
        """
        # 调用父类的初始化方法
        super(TpsSpatialTransformerNetwork, self).__init__()
        # 设置输入参数
        self.F = F  # 参数 F，可能是变换参数
        self.I_size = I_size  # 输入图像 I 的尺寸 (height, width)
        self.I_r_size = I_r_size  # 矫正后图像 I_r 的尺寸 (I_r_height, I_r_width)
        self.I_channel_num = I_channel_num  # 输入图像 I 的通道数

        # 初始化定位网络，传入变换参数 F 和输入图像通道数 I_channel_num
        self.LocalizationNetwork = LocalizationNetwork(
            self.F,
            self.I_channel_num,
        )
        # 初始化网格生成器，传入变换参数 F 和矫正图像尺寸 I_r_size
        self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    def forward(self, batch_I):
        # 使用定位网络计算 batch_I 的控制点 batch_C_prime，形状为 batch_size x K x 2
        batch_C_prime = self.LocalizationNetwork(batch_I)

        # 使用网格生成器生成基于 batch_C_prime 的网格点 build_P_prime，形状为 batch_size x n x 2
        build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)

        # 将 build_P_prime 重新形状为 [batch_size, I_r_height, I_r_width, 2]
        build_P_prime_reshape = build_P_prime.reshape(
            [build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2])

        # 使用 F.grid_sample 对输入图像 batch_I 进行空间变换，填充模式为 "border"
        batch_I_r = F.grid_sample(
            batch_I,
            build_P_prime_reshape,
            padding_mode="border",
        )

        # 返回矫正后的图像 batch_I_r
        return batch_I_r
class LocalizationNetwork(nn.Module):
    """ Localization Network of RARE, which predicts C' (K x 2) from I (I_width x I_height) """

    def __init__(self, F, I_channel_num: int):
        super(LocalizationNetwork, self).__init__()
        self.F = F
        self.I_channel_num = I_channel_num
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.I_channel_num,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),  # 用于从输入图像中提取特征的卷积层
            nn.BatchNorm2d(64),  # 批量归一化层，加速收敛并减少过拟合
            nn.ReLU(True),  # ReLU 激活函数，增强网络的非线性特性
            nn.MaxPool2d(2, 2),  # 最大池化层，减小特征图的空间尺寸
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),  # 第二个卷积层
            nn.BatchNorm2d(128),  # 批量归一化层
            nn.ReLU(True),  # ReLU 激活函数
            nn.MaxPool2d(2, 2),  # 最大池化层
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),  # 第三个卷积层
            nn.BatchNorm2d(256),  # 批量归一化层
            nn.ReLU(True),  # ReLU 激活函数
            nn.MaxPool2d(2, 2),  # 最大池化层
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),  # 第四个卷积层
            nn.BatchNorm2d(512),  # 批量归一化层
            nn.ReLU(True),  # ReLU 激活函数
            nn.AdaptiveAvgPool2d(1),  # 自适应平均池化层，将特征图池化成大小为 1x1
        )

        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256),  # 第一个全连接层
                                              nn.ReLU(True))  # ReLU 激活函数
        self.localization_fc2 = nn.Linear(256, self.F * 2)  # 第二个全连接层，输出 F 个控制点的坐标（每个点两个维度）

        # Init fc2 in LocalizationNetwork
        self.localization_fc2.weight.data.fill_(0)  # 初始化第二个全连接层的权重为 0

        # see RARE paper Fig. 6 (a)
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))  # 在 x 轴上均匀分布 F/2 个控制点
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))  # 在上半部分 y 轴上均匀分布 F/2 个控制点
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))  # 在下半部分 y 轴上均匀分布 F/2 个控制点
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)  # 上半部分控制点的坐标
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)  # 下半部分控制点的坐标
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)  # 初始化第二个全连接层的偏置
        self.localization_fc2.bias.data = (
            torch.from_numpy(initial_bias).float().view(-1))  # 设置第二个全连接层的偏置

    def forward(self, batch_I):
        """
        :param batch_I : Batch Input Image [batch_size x I_channel_num x I_height x I_width]
        :return: batch_C_prime : Predicted coordinates of fiducial points for input batch [batch_size x F x 2]
        """
        batch_size = batch_I.size(0)  # 批量大小
        features = self.conv(batch_I).view(batch_size, -1)  # 提取特征并将其展平
        batch_C_prime = self.localization_fc2(
            self.localization_fc1(features)).view(batch_size, self.F, 2)  # 计算预测的控制点坐标
        return batch_C_prime


class GridGenerator(nn.Module):
    """ Grid Generator of RARE, which produces P_prime by multipling T with P """
    def __init__(self, F, I_r_size):
        """ Generate P_hat and inv_delta_C for later """
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        self.C = self._build_C(self.F)  # F x 2
        self.P = self._build_P(self.I_r_width, self.I_r_height)

        # for multi-gpu, you need register buffer
        self.register_buffer(
            "inv_delta_C",
            torch.tensor(self._build_inv_delta_C(
                self.F,
                self.C,
            )).float(),
        )  # F+3 x F+3
        self.register_buffer(
            "P_hat",
            torch.tensor(self._build_P_hat(
                self.F,
                self.C,
                self.P,
            )).float(),
        )  # n x F+3

    def _build_C(self, F):
        """ Return coordinates of fiducial points in I_r; C """
        # Generate equally spaced control points along x-axis
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F / 2))  # Top control points set to -1
        ctrl_pts_y_bottom = np.ones(int(F / 2))   # Bottom control points set to 1
        # Combine top and bottom control points into a single array
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)  # Concatenate top and bottom arrays
        return C  # F x 2

    def _build_inv_delta_C(self, F, C):
        """ Return inv_delta_C which is needed to calculate T """
        hat_C = np.zeros((F, F), dtype=float)  # Initialize hat_C matrix
        # Calculate pairwise distances between fiducial points
        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])  # Euclidean distance between points i and j
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)  # Set diagonal elements to 1
        hat_C = (hat_C**2) * np.log(hat_C)  # Apply the logarithmic transformation
        # Construct the delta_C matrix by concatenating different components
        delta_C = np.concatenate(
            [
                np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),  # F x F+3
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),  # 2 x F+3
                np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1),  # 1 x F+3
            ],
            axis=0,
        )
        inv_delta_C = np.linalg.inv(delta_C)  # Compute the inverse of delta_C
        return inv_delta_C  # F+3 x F+3

    def _build_P(self, I_r_width, I_r_height):
        """ Return grid points P in I_r """
        # Generate a grid of points across the specified width and height
        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height
        # Create a meshgrid of x and y coordinates
        P = np.stack(
            np.meshgrid(I_r_grid_x, I_r_grid_y),
            axis=2)
        return P.reshape([-1, 2])  # Reshape into n x 2 array
    def _build_P_hat(self, F, C, P):
        n = P.shape[0]  # 获取P矩阵的行数，即数据点的数量，n (= self.I_r_width x self.I_r_height)
        P_tile = np.tile(np.expand_dims(P, axis=1),
                         (1, F, 1))  # 将P矩阵扩展为与C矩阵相同维度的矩阵，形状为 n x F x 2
        C_tile = np.expand_dims(C, axis=0)  # 将C矩阵扩展为与P_tile矩阵相同维度，形状为 1 x F x 2
        P_diff = P_tile - C_tile  # 计算P_tile和C_tile之间的差异，形状为 n x F x 2
        rbf_norm = np.linalg.norm(
            P_diff,
            ord=2,
            axis=2,
            keepdims=False,
        )  # 计算P_diff矩阵每行的2范数，形状为 n x F
        rbf = np.multiply(
            np.square(rbf_norm),
            np.log(rbf_norm + self.eps),
        )  # 计算RBF核函数的结果，形状为 n x F
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)  # 将计算结果与偏置项拼接，形状为 n x F+3
        return P_hat  # 返回生成的P_hat矩阵，形状为 n x F+3

    def build_P_prime(self, batch_C_prime):
        """ 根据输入的batch_C_prime生成网格 [batch_size x F x 2] """
        batch_size = batch_C_prime.size(0)  # 获取batch的大小
        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)  # 将self.inv_delta_C矩阵复制为batch_size份，形状为 batch_size x F x 2
        batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)  # 将self.P_hat矩阵复制为batch_size份，形状为 batch_size x n x F+3
        batch_C_prime_with_zeros = torch.cat(
            (batch_C_prime, torch.zeros(batch_size, 3, 2).float().to(device)),
            dim=1)  # 在batch_C_prime后面添加全零张量，形状变为 batch_size x F+3 x 2
        batch_T = torch.bmm(
            batch_inv_delta_C,
            batch_C_prime_with_zeros,
        )  # 执行batch_inv_delta_C与batch_C_prime_with_zeros的批矩阵乘法，形状为 batch_size x F+3 x 2
        batch_P_prime = torch.bmm(batch_P_hat, batch_T)  # 执行batch_P_hat与batch_T的批矩阵乘法，形状为 batch_size x n x 2
        return batch_P_prime  # 返回生成的batch_P_prime网格，形状为 batch_size x n x 2
```