# `so-vits-svc\modules\modules.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 torch.nn 模块中导入 functional 模块并重命名为 F
from torch.nn import functional as F

# 导入自定义模块 attentions 和 commons
import modules.attentions as attentions
import modules.commons as commons
# 从 modules.commons 模块中导入 get_padding 和 init_weights 函数
from modules.commons import get_padding, init_weights
# 从 modules.DSConv 模块中导入 Depthwise_Separable_Conv1D 类、remove_weight_norm_modules 函数和 weight_norm_modules 函数

# 设置 LRELU_SLOPE 变量的值为 0.1
LRELU_SLOPE = 0.1

# 将 nn.Conv1d 赋值给 Conv1dModel 变量
Conv1dModel = nn.Conv1d

# 定义函数 set_Conv1dModel，根据参数 use_depthwise_conv 决定将 Depthwise_Separable_Conv1D 或 nn.Conv1d 赋值给 Conv1dModel 变量
def set_Conv1dModel(use_depthwise_conv):
    global Conv1dModel
    Conv1dModel = Depthwise_Separable_Conv1D if use_depthwise_conv else nn.Conv1d

# 定义 LayerNorm 类，继承自 nn.Module
class LayerNorm(nn.Module):
  # 初始化函数，接受 channels 和 eps 两个参数
  def __init__(self, channels, eps=1e-5):
    super().__init__()
    # 初始化 channels 和 eps 属性
    self.channels = channels
    self.eps = eps
    # 初始化 gamma 和 beta 为可学习参数
    self.gamma = nn.Parameter(torch.ones(channels))
    self.beta = nn.Parameter(torch.zeros(channels))

  # 前向传播函数，接受输入 x
  def forward(self, x):
    # 将输入 x 进行维度转换
    x = x.transpose(1, -1)
    # 对 x 进行 Layer Normalization
    x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
    # 再次进行维度转换后返回结果
    return x.transpose(1, -1)

# 定义 ConvReluNorm 类，继承自 nn.Module
class ConvReluNorm(nn.Module):
  # 初始化函数，接受 in_channels、hidden_channels、out_channels、kernel_size、n_layers、p_dropout 六个参数
  def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
    super().__init__()
    # 初始化各个属性
    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.n_layers = n_layers
    self.p_dropout = p_dropout
    # 断言 n_layers 大于 1
    assert n_layers > 1, "Number of layers should be larger than 0."

    # 初始化 nn.ModuleList 类型的 conv_layers 和 norm_layers
    self.conv_layers = nn.ModuleList()
    self.norm_layers = nn.ModuleList()
    # 添加第一个卷积层和 LayerNorm 层
    self.conv_layers.append(Conv1dModel(in_channels, hidden_channels, kernel_size, padding=kernel_size//2))
    self.norm_layers.append(LayerNorm(hidden_channels))
    # 添加 ReLU 和 Dropout 层
    self.relu_drop = nn.Sequential(
        nn.ReLU(),
        nn.Dropout(p_dropout))
    # 循环添加 n_layers-1 个卷积层和 LayerNorm 层
    for _ in range(n_layers-1):
      self.conv_layers.append(Conv1dModel(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2))
      self.norm_layers.append(LayerNorm(hidden_channels))
    # 添加一个 1x1 卷积层作为投影层
    self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
    # 将投影层的权重初始化为 0
    self.proj.weight.data.zero_()
    # 将神经网络模型中的偏置数据全部置零
    self.proj.bias.data.zero_()

  # 前向传播函数，接受输入数据 x 和掩码 x_mask
  def forward(self, x, x_mask):
    # 保存原始输入数据
    x_org = x
    # 循环执行神经网络的卷积层和归一化层
    for i in range(self.n_layers):
      # 对输入数据执行卷积操作，并乘以掩码
      x = self.conv_layers[i](x * x_mask)
      # 对卷积后的数据执行归一化操作
      x = self.norm_layers[i](x)
      # 对归一化后的数据执行激活函数和随机失活操作
      x = self.relu_drop(x)
    # 将原始输入数据与经过神经网络处理后的数据相加，并通过投影层处理
    x = x_org + self.proj(x)
    # 返回处理后的数据乘以掩码
    return x * x_mask
class WN(torch.nn.Module):
  # 定义一个名为WN的神经网络模型类，继承自torch.nn.Module

  def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
    # 初始化函数，接受隐藏层通道数、卷积核大小、膨胀率、层数、输入通道数和dropout概率作为参数

    super(WN, self).__init__()
    # 调用父类的初始化函数

    assert(kernel_size % 2 == 1)
    # 断言，确保卷积核大小为奇数

    self.hidden_channels =hidden_channels
    # 设置隐藏层通道数

    self.kernel_size = kernel_size,
    # 设置卷积核大小

    self.dilation_rate = dilation_rate
    # 设置膨胀率

    self.n_layers = n_layers
    # 设置层数

    self.gin_channels = gin_channels
    # 设置输入通道数

    self.p_dropout = p_dropout
    # 设置dropout概率

    self.in_layers = torch.nn.ModuleList()
    # 初始化输入层列表

    self.res_skip_layers = torch.nn.ModuleList()
    # 初始化残差跳跃连接层列表

    self.drop = nn.Dropout(p_dropout)
    # 初始化dropout层，使用给定的dropout概率

    if gin_channels != 0:
      # 如果输入通道数不为0
      cond_layer = torch.nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)
      # 创建一个1维卷积层，输入通道数为输入通道数，输出通道数为2*隐藏层通道数*层数，卷积核大小为1
      self.cond_layer = weight_norm_modules(cond_layer, name='weight')
      # 对创建的卷积层进行权重归一化处理

    for i in range(n_layers):
      # 循环n_layers次
      dilation = dilation_rate ** i
      # 计算当前层的膨胀率
      padding = int((kernel_size * dilation - dilation) / 2)
      # 计算当前层的填充大小
      in_layer = Conv1dModel(hidden_channels, 2*hidden_channels, kernel_size,
                                 dilation=dilation, padding=padding)
      # 创建一个1维卷积模型，输入通道数为隐藏层通道数，输出通道数为2*隐藏层通道数，卷积核大小为给定的kernel_size，膨胀率为当前层的膨胀率，填充大小为当前层的填充大小
      in_layer = weight_norm_modules(in_layer, name='weight')
      # 对创建的卷积层进行权重归一化处理
      self.in_layers.append(in_layer)
      # 将创建的卷积层添加到输入层列表中

      # last one is not necessary
      if i < n_layers - 1:
        # 如果不是最后一层
        res_skip_channels = 2 * hidden_channels
        # 设置残差跳跃连接层的输出通道数为2倍的隐藏层通道数
      else:
        res_skip_channels = hidden_channels
        # 否则，设置残差跳跃连接层的输出通道数为隐藏层通道数

      res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
      # 创建一个1维卷积层，输入通道数为隐藏层通道数，输出通道数为残差跳跃连接层的输出通道数，卷积核大小为1
      res_skip_layer = weight_norm_modules(res_skip_layer, name='weight')
      # 对创建的卷积层进行权重归一化处理
      self.res_skip_layers.append(res_skip_layer)
      # 将创建的卷积层添加到残差跳跃连接层列表中

  def forward(self, x, x_mask, g=None, **kwargs):
    # 前向传播函数，接受输入x、输入掩码x_mask、条件输入g和其他关键字参数

    output = torch.zeros_like(x)
    # 创建一个与输入x相同大小的全零张量

    n_channels_tensor = torch.IntTensor([self.hidden_channels])
    # 创建一个包含隐藏层通道数的张量

    if g is not None:
      # 如果条件输入g不为空
      g = self.cond_layer(g)
      # 对条件输入g进行卷积处理
    # 遍历神经网络的层数
    for i in range(self.n_layers):
      # 将输入数据传入第i层的输入层
      x_in = self.in_layers[i](x)
      # 如果条件数据不为空
      if g is not None:
        # 计算条件偏移量
        cond_offset = i * 2 * self.hidden_channels
        # 从条件数据中提取当前层的条件信息
        g_l = g[:,cond_offset:cond_offset+2*self.hidden_channels,:]
      else:
        # 如果条件数据为空，则创建与输入数据相同形状的全零张量
        g_l = torch.zeros_like(x_in)

      # 对输入数据和条件数据进行融合操作
      acts = commons.fused_add_tanh_sigmoid_multiply(
          x_in,
          g_l,
          n_channels_tensor)
      # 对融合后的数据进行丢弃操作
      acts = self.drop(acts)

      # 通过残差连接和跳跃连接处理融合后的数据
      res_skip_acts = self.res_skip_layers[i](acts)
      # 如果不是最后一层
      if i < self.n_layers - 1:
        # 提取残差连接部分的数据
        res_acts = res_skip_acts[:,:self.hidden_channels,:]
        # 更新输入数据，加上残差连接部分的数据，并乘以掩码
        x = (x + res_acts) * x_mask
        # 更新输出数据，加上跳跃连接部分的数据
        output = output + res_skip_acts[:,self.hidden_channels:,:]
      else:
        # 更新输出数据，加上残差连接部分的数据
        output = output + res_skip_acts
    # 返回最终输出数据，并乘以掩码
    return output * x_mask

  # 移除权重归一化
  def remove_weight_norm(self):
    # 如果条件通道数不为0，则移除条件层的权重归一化
    if self.gin_channels != 0:
      remove_weight_norm_modules(self.cond_layer)
    # 移除所有输入层的权重归一化
    for l in self.in_layers:
      remove_weight_norm_modules(l)
    # 移除所有残差跳跃连接层的权重归一化
    for l in self.res_skip_layers:
      remove_weight_norm_modules(l)
# 定义一个名为 ResBlock1 的神经网络模块类，继承自 torch.nn.Module
class ResBlock1(torch.nn.Module):
    # 初始化函数，接受 channels（通道数）、kernel_size（卷积核大小）、dilation（空洞卷积率）三个参数
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        # 调用父类的初始化函数
        super(ResBlock1, self).__init__()
        # 定义第一组卷积层，使用 nn.ModuleList 包装多个卷积层
        self.convs1 = nn.ModuleList([
            # 使用 weight_norm_modules 包装 Conv1dModel，并传入相应参数
            weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            # 使用 weight_norm_modules 包装 Conv1dModel，并传入相应参数
            weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            # 使用 weight_norm_modules 包装 Conv1dModel，并传入相应参数
            weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2]))
        ])
        # 对第一组卷积层应用初始化权重的函数
        self.convs1.apply(init_weights)

        # 定义第二组卷积层，使用 nn.ModuleList 包装多个卷积层
        self.convs2 = nn.ModuleList([
            # 使用 weight_norm_modules 包装 Conv1dModel，并传入相应参数
            weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            # 使用 weight_norm_modules 包装 Conv1dModel，并传入相应参数
            weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            # 使用 weight_norm_modules 包装 Conv1dModel，并传入相应参数
            weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        # 对第二组卷积层应用初始化权重的函数
        self.convs2.apply(init_weights)

    # 前向传播函数，接受输入 x 和 x_mask 两个参数
    def forward(self, x, x_mask=None):
        # 遍历第一组和第二组卷积层，分别为 c1 和 c2
        for c1, c2 in zip(self.convs1, self.convs2):
            # 对输入 x 应用 LeakyReLU 激活函数
            xt = F.leaky_relu(x, LRELU_SLOPE)
            # 如果 x_mask 不为空，则将 xt 与 x_mask 相乘
            if x_mask is not None:
                xt = xt * x_mask
            # 将 xt 输入到第一组卷积层 c1 中
            xt = c1(xt)
            # 对输出 xt 应用 LeakyReLU 激活函数
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            # 如果 x_mask 不为空，则将 xt 与 x_mask 相乘
            if x_mask is not None:
                xt = xt * x_mask
            # 将 xt 输入到第二组卷积层 c2 中
            xt = c2(xt)
            # 将 xt 与输入 x 相加，得到输出 x
            x = xt + x
        # 如果 x_mask 不为空，则将 x 与 x_mask 相乘
        if x_mask is not None:
            x = x * x_mask
        # 返回输出 x
        return x
    # 定义一个方法，用于移除权重归一化
    def remove_weight_norm(self):
        # 遍历self.convs1中的每个元素，对每个元素调用remove_weight_norm_modules方法
        for l in self.convs1:
            remove_weight_norm_modules(l)
        # 遍历self.convs2中的每个元素，对每个元素调用remove_weight_norm_modules方法
        for l in self.convs2:
            remove_weight_norm_modules(l)
# 定义一个包含两个卷积层的残差块模型
class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        # 使用 nn.ModuleList 定义包含两个卷积层的列表
        self.convs = nn.ModuleList([
            # 使用权重归一化的 Conv1dModel 模型作为第一个卷积层
            weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            # 使用权重归一化的 Conv1dModel 模型作为第二个卷积层
            weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        # 对两个卷积层应用初始化权重的函数
        self.convs.apply(init_weights)

    # 定义前向传播函数
    def forward(self, x, x_mask=None):
        # 遍历两个卷积层
        for c in self.convs:
            # 使用 LeakyReLU 激活函数处理输入数据
            xt = F.leaky_relu(x, LRELU_SLOPE)
            # 如果输入数据有 mask，则将其应用到数据上
            if x_mask is not None:
                xt = xt * x_mask
            # 将数据传入当前卷积层
            xt = c(xt)
            # 将卷积层的输出与输入数据相加，得到残差连接的结果
            x = xt + x
        # 如果输入数据有 mask，则将其应用到最终的输出数据上
        if x_mask is not None:
            x = x * x_mask
        # 返回最终的输出数据
        return x

    # 移除权重归一化
    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm_modules(l)


# 定义一个对数操作的模型
class Log(nn.Module):
  def forward(self, x, x_mask, reverse=False, **kwargs):
    # 如果不是反向操作，则对输入数据取对数并应用 mask
    if not reverse:
      y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
      # 计算对数的行列和
      logdet = torch.sum(-y, [1, 2])
      return y, logdet
    # 如果是反向操作，则对输入数据取指数并应用 mask
    else:
      x = torch.exp(x) * x_mask
      return x
    

# 定义一个翻转操作的模型
class Flip(nn.Module):
  def forward(self, x, *args, reverse=False, **kwargs):
    # 对输入数据在维度1上进行翻转
    x = torch.flip(x, [1])
    # 如果不是反向操作，则返回翻转后的数据和零的对数行列和
    if not reverse:
      logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
      return x, logdet
    # 如果是反向操作，则直接返回翻转后的数据
    else:
      return x


# 定义一个逐元素仿射变换操作的模型
class ElementwiseAffine(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.channels = channels
    # 定义可学习的偏移参数
    self.m = nn.Parameter(torch.zeros(channels,1))
    # 定义可学习的对数尺度参数
    self.logs = nn.Parameter(torch.zeros(channels,1))

  def forward(self, x, x_mask, reverse=False, **kwargs):
    # 如果不是反向操作，则对输入数据进行仿射变换并应用 mask
    if not reverse:
      y = self.m + torch.exp(self.logs) * x
      y = y * x_mask
      # 计算对数尺度的行列和
      logdet = torch.sum(self.logs * x_mask, [1,2])
      return y, logdet
    else:
      # 如果条件不满足，则执行以下操作
      # 计算 x 减去 self.m 的结果，并乘以 torch.exp(-self.logs) 和 x_mask
      x = (x - self.m) * torch.exp(-self.logs) * x_mask
      # 返回计算结果
      return x
class ResidualCouplingLayer(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      p_dropout=0,
      gin_channels=0,
      mean_only=False,
      wn_sharing_parameter=None
      ):
    # 确保通道数可以被2整除
    assert channels % 2 == 0, "channels should be divisible by 2"
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.half_channels = channels // 2
    self.mean_only = mean_only

    # 创建1维卷积层，用于预处理输入的前半部分
    self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
    # 创建WaveNet层
    self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels) if wn_sharing_parameter is None else wn_sharing_parameter
    # 创建1维卷积层，用于处理WaveNet层的输出
    self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
    self.post.weight.data.zero_()
    self.post.bias.data.zero_()

  def forward(self, x, x_mask, g=None, reverse=False):
    # 将输入张量按通道数的一半分割成两部分
    x0, x1 = torch.split(x, [self.half_channels]*2, 1)
    # 对前半部分进行预处理
    h = self.pre(x0) * x_mask
    # 通过WaveNet层进行编码
    h = self.enc(h, x_mask, g=g)
    # 对编码后的结果进行处理
    stats = self.post(h) * x_mask
    if not self.mean_only:
      # 如果不是仅计算均值，则将结果分割成均值和标准差
      m, logs = torch.split(stats, [self.half_channels]*2, 1)
    else:
      m = stats
      logs = torch.zeros_like(m)

    if not reverse:
      # 如果不是反向操作，则根据均值和标准差对后半部分进行变换
      x1 = m + x1 * torch.exp(logs) * x_mask
      x = torch.cat([x0, x1], 1)
      logdet = torch.sum(logs, [1,2])
      return x, logdet
    else:
      # 如果是反向操作，则根据均值和标准差对后半部分进行逆变换
      x1 = (x1 - m) * torch.exp(-logs) * x_mask
      x = torch.cat([x0, x1], 1)
      return x

class TransformerCouplingLayer(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      n_layers,
      n_heads,
      p_dropout=0,
      filter_channels=0,
      mean_only=False,
      wn_sharing_parameter=None,
      gin_channels = 0
      ):
    # 确保通道数可以被2整除
    assert channels % 2 == 0, "channels should be divisible by 2"
    super().__init__()
    # 设置神经网络的通道数
    self.channels = channels
    # 设置隐藏层的通道数
    self.hidden_channels = hidden_channels
    # 设置卷积核大小
    self.kernel_size = kernel_size
    # 设置网络层数
    self.n_layers = n_layers
    # 计算通道数的一半
    self.half_channels = channels // 2
    # 设置是否仅计算均值
    self.mean_only = mean_only

    # 创建一个一维卷积层，输入通道数为 half_channels，输出通道数为 hidden_channels，卷积核大小为 1
    self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
    # 创建一个 FFT 注意力层
    self.enc = attentions.FFT(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, isflow = True, gin_channels = gin_channels) if wn_sharing_parameter is None else wn_sharing_parameter
    # 创建一个一维卷积层，输入通道数为 hidden_channels，输出通道数为 half_channels * (2 - mean_only)，卷积核大小为 1
    self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
    # 将 post 层的权重和偏置初始化为零
    self.post.weight.data.zero_()
    self.post.bias.data.zero_()

  def forward(self, x, x_mask, g=None, reverse=False):
    # 将输入 x 按照 half_channels 进行分割
    x0, x1 = torch.split(x, [self.half_channels]*2, 1)
    # 对 x0 进行预处理
    h = self.pre(x0) * x_mask
    # 使用 enc 层进行编码
    h = self.enc(h, x_mask, g=g)
    # 对编码结果进行后处理
    stats = self.post(h) * x_mask
    # 如果不仅计算均值
    if not self.mean_only:
      # 将 stats 按照 half_channels 进行分割，得到均值 m 和对数 logs
      m, logs = torch.split(stats, [self.half_channels]*2, 1)
    else:
      # 如果仅计算均值，则 logs 初始化为零
      m = stats
      logs = torch.zeros_like(m)

    # 如果不是反向传播
    if not reverse:
      # 根据均值和 logs 对输入 x1 进行变换
      x1 = m + x1 * torch.exp(logs) * x_mask
      # 将处理后的 x0 和 x1 进行拼接
      x = torch.cat([x0, x1], 1)
      # 计算对数行列式
      logdet = torch.sum(logs, [1,2])
      return x, logdet
    else:
      # 如果是反向传播，则根据均值和 logs 对输入 x1 进行反向变换
      x1 = (x1 - m) * torch.exp(-logs) * x_mask
      # 将处理后的 x0 和 x1 进行拼接
      x = torch.cat([x0, x1], 1)
      return x
```