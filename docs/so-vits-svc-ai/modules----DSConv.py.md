# `so-vits-svc\modules\DSConv.py`

```py
import torch.nn as nn  # 导入 PyTorch 中的神经网络模块
from torch.nn.utils import remove_weight_norm, weight_norm  # 从 PyTorch 中的神经网络工具模块导入去除权重归一化和权重归一化函数


class Depthwise_Separable_Conv1D(nn.Module):  # 定义深度可分离一维卷积神经网络模块
    def __init__(  # 初始化函数
        self,
        in_channels,  # 输入通道数
        out_channels,  # 输出通道数
        kernel_size,  # 卷积核大小
        stride = 1,  # 步长，默认为1
        padding = 0,  # 填充，默认为0
        dilation = 1,  # 膨胀，默认为1
        bias = True,  # 是否使用偏置，默认为True
        padding_mode = 'zeros',  # 填充模式，默认为'zeros'
        device=None,  # 设备，默认为None
        dtype=None  # 数据类型，默认为None
    ):
      super().__init__()  # 调用父类的初始化函数
      self.depth_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels,stride = stride,padding=padding,dilation=dilation,bias=bias,padding_mode=padding_mode,device=device,dtype=dtype)  # 创建深度卷积层
      self.point_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias, device=device,dtype=dtype)  # 创建逐点卷积层
    
    def forward(self, input):  # 前向传播函数
      return self.point_conv(self.depth_conv(input))  # 返回逐点卷积层对深度卷积层的输出

    def weight_norm(self):  # 权重归一化函数
      self.depth_conv = weight_norm(self.depth_conv, name = 'weight')  # 对深度卷积层进行权重归一化
      self.point_conv = weight_norm(self.point_conv, name = 'weight')  # 对逐点卷积层进行权重归一化

    def remove_weight_norm(self):  # 去除权重归一化函数
      self.depth_conv = remove_weight_norm(self.depth_conv, name = 'weight')  # 去除深度卷积层的权重归一化
      self.point_conv = remove_weight_norm(self.point_conv, name = 'weight')  # 去除逐点卷积层的权重归一化

class Depthwise_Separable_TransposeConv1D(nn.Module):  # 定义深度可分离一维转置卷积神经网络模块
    def __init__(  # 初始化函数
        self,
        in_channels,  # 输入通道数
        out_channels,  # 输出通道数
        kernel_size,  # 卷积核大小
        stride = 1,  # 步长，默认为1
        padding = 0,  # 填充，默认为0
        output_padding = 0,  # 输出填充，默认为0
        bias = True,  # 是否使用偏置，默认为True
        dilation = 1,  # 膨胀，默认为1
        padding_mode = 'zeros',  # 填充模式，默认为'zeros'
        device=None,  # 设备，默认为None
        dtype=None  # 数据类型，默认为None
    # 定义一个继承自 nn.Module 的类
    ):
      # 调用父类的构造函数
      super().__init__()
      # 定义深度卷积层，使用转置卷积进行操作
      self.depth_conv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels,stride = stride,output_padding=output_padding,padding=padding,dilation=dilation,bias=bias,padding_mode=padding_mode,device=device,dtype=dtype)
      # 定义点卷积层
      self.point_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias, device=device,dtype=dtype)
    
    # 定义前向传播函数
    def forward(self, input):
      # 返回点卷积层作用于深度卷积层的结果
      return self.point_conv(self.depth_conv(input))

    # 定义权重归一化函数
    def weight_norm(self):
      # 对深度卷积层和点卷积层进行权重归一化处理
      self.depth_conv = weight_norm(self.depth_conv, name = 'weight')
      self.point_conv = weight_norm(self.point_conv, name = 'weight')

    # 定义移除权重归一化函数
    def remove_weight_norm(self):
      # 移除深度卷积层和点卷积层的权重归一化处理
      remove_weight_norm(self.depth_conv, name = 'weight')
      remove_weight_norm(self.point_conv, name = 'weight')
# 对模块进行权重归一化处理，可以指定权重名称和维度
def weight_norm_modules(module, name = 'weight', dim = 0):
    # 如果模块是 Depthwise_Separable_Conv1D 或 Depthwise_Separable_TransposeConv1D 类型，则进行权重归一化处理
    if isinstance(module,Depthwise_Separable_Conv1D) or isinstance(module,Depthwise_Separable_TransposeConv1D):
      module.weight_norm()
      # 返回处理后的模块
      return module
    else:
      # 对普通模块进行权重归一化处理
      return weight_norm(module,name,dim)

# 移除模块的权重归一化处理，可以指定权重名称
def remove_weight_norm_modules(module, name = 'weight'):
    # 如果模块是 Depthwise_Separable_Conv1D 或 Depthwise_Separable_TransposeConv1D 类型，则移除权重归一化处理
    if isinstance(module,Depthwise_Separable_Conv1D) or isinstance(module,Depthwise_Separable_TransposeConv1D):
      module.remove_weight_norm()
    else:
      # 移除普通模块的权重归一化处理
      remove_weight_norm(module,name)
```