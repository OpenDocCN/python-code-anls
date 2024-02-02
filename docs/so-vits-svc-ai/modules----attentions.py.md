# `so-vits-svc\modules\attentions.py`

```py
# 导入数学库
import math

# 导入 PyTorch 库
import torch
from torch import nn
from torch.nn import functional as F

# 导入自定义模块
import modules.commons as commons
from modules.DSConv import weight_norm_modules
from modules.modules import LayerNorm

# 定义 FFT 类，继承自 nn.Module
class FFT(nn.Module):
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers=1, kernel_size=1, p_dropout=0.,
               proximal_bias=False, proximal_init=True, isflow = False, **kwargs):
    super().__init__()
    # 初始化各种参数
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.proximal_bias = proximal_bias
    self.proximal_init = proximal_init
    # 如果是流模式
    if isflow:
      # 创建条件层
      cond_layer = torch.nn.Conv1d(kwargs["gin_channels"], 2*hidden_channels*n_layers, 1)
      self.cond_pre = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, 1)
      self.cond_layer = weight_norm_modules(cond_layer, name='weight')
      self.gin_channels = kwargs["gin_channels"]
    # 创建丢弃层
    self.drop = nn.Dropout(p_dropout)
    # 创建自注意力层列表、规范化层列表和前馈神经网络层列表
    self.self_attn_layers = nn.ModuleList()
    self.norm_layers_0 = nn.ModuleList()
    self.ffn_layers = nn.ModuleList()
    self.norm_layers_1 = nn.ModuleList()
    # 循环创建多层自注意力层、规范化层和前馈神经网络层
    for i in range(self.n_layers):
      self.self_attn_layers.append(
        MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias,
                           proximal_init=proximal_init))
      self.norm_layers_0.append(LayerNorm(hidden_channels))
      self.ffn_layers.append(
        FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout, causal=True))
      self.norm_layers_1.append(LayerNorm(hidden_channels))

  # 前向传播函数
  def forward(self, x, x_mask, g = None):
    """
    x: decoder input
    h: encoder output
    """
    # 如果存在条件输入 g
    if g is not None:
      # 对 g 进行条件层处理
      g = self.cond_layer(g)

    # 创建自注意力掩码
    self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
    # 将 x 与 x_mask 逐元素相乘
    x = x * x_mask
    # 遍历神经网络的每一层
    for i in range(self.n_layers):
      # 如果条件输入 g 不为空
      if g is not None:
        # 对输入 x 进行条件预处理
        x = self.cond_pre(x)
        # 计算条件偏移量
        cond_offset = i * 2 * self.hidden_channels
        # 从条件输入 g 中提取当前层的条件信息
        g_l = g[:,cond_offset:cond_offset+2*self.hidden_channels,:]
        # 使用融合的操作对 x 进行变换
        x = commons.fused_add_tanh_sigmoid_multiply(
          x,
          g_l,
          torch.IntTensor([self.hidden_channels]))
      # 对输入 x 进行自注意力机制计算
      y = self.self_attn_layers[i](x, x, self_attn_mask)
      # 对计算结果 y 进行丢弃部分操作
      y = self.drop(y)
      # 对输入 x 进行残差连接和层归一化
      x = self.norm_layers_0[i](x + y)

      # 对输入 x 进行前馈神经网络计算
      y = self.ffn_layers[i](x, x_mask)
      # 对计算结果 y 进行丢弃部分操作
      y = self.drop(y)
      # 对输入 x 进行残差连接和层归一化
      x = self.norm_layers_1[i](x + y)
    # 将 x 与 x_mask 逐元素相乘
    x = x * x_mask
    # 返回最终结果 x
    return x
class Encoder(nn.Module):
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., window_size=4, **kwargs):
    super().__init__()
    self.hidden_channels = hidden_channels  # 隐藏层通道数
    self.filter_channels = filter_channels  # 过滤器通道数
    self.n_heads = n_heads  # 多头注意力机制的头数
    self.n_layers = n_layers  # 编码器层数
    self.kernel_size = kernel_size  # 卷积核大小
    self.p_dropout = p_dropout  # 丢弃概率
    self.window_size = window_size  # 窗口大小

    self.drop = nn.Dropout(p_dropout)  # 定义丢弃层
    self.attn_layers = nn.ModuleList()  # 定义多头注意力层列表
    self.norm_layers_1 = nn.ModuleList()  # 定义编码器层归一化列表
    self.ffn_layers = nn.ModuleList()  # 定义前馈神经网络层列表
    self.norm_layers_2 = nn.ModuleList()  # 定义解码器层归一化列表
    for i in range(self.n_layers):
      self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, window_size=window_size))  # 添加多头注意力层
      self.norm_layers_1.append(LayerNorm(hidden_channels))  # 添加编码器层归一化
      self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))  # 添加前馈神经网络层
      self.norm_layers_2.append(LayerNorm(hidden_channels))  # 添加解码器层归一化

  def forward(self, x, x_mask):
    attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)  # 生成注意力掩码
    x = x * x_mask  # 对输入进行掩码处理
    for i in range(self.n_layers):
      y = self.attn_layers[i](x, x, attn_mask)  # 多头注意力层的前向传播
      y = self.drop(y)  # 使用丢弃层
      x = self.norm_layers_1[i](x + y)  # 编码器层归一化
      y = self.ffn_layers[i](x, x_mask)  # 前馈神经网络层的前向传播
      y = self.drop(y)  # 使用丢弃层
      x = self.norm_layers_2[i](x + y)  # 解码器层归一化
    x = x * x_mask  # 最终输出结果与掩码相乘
    return x


class Decoder(nn.Module):
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., proximal_bias=False, proximal_init=True, **kwargs):
    super().__init__()
    self.hidden_channels = hidden_channels  # 隐藏层通道数
    self.filter_channels = filter_channels  # 过滤器通道数
    self.n_heads = n_heads  # 多头注意力机制的头数
    self.n_layers = n_layers  # 解码器层数
    self.kernel_size = kernel_size  # 卷积核大小
    self.p_dropout = p_dropout  # 丢弃概率
    self.proximal_bias = proximal_bias  # 是否使用邻近偏置
    self.proximal_init = proximal_init  # 是否使用邻近初始化

    self.drop = nn.Dropout(p_dropout)  # 定义丢弃层
    self.self_attn_layers = nn.ModuleList()  # 定义自注意力层列表
    # 初始化多个规范化层的模块列表
    self.norm_layers_0 = nn.ModuleList()
    self.encdec_attn_layers = nn.ModuleList()
    self.norm_layers_1 = nn.ModuleList()
    self.ffn_layers = nn.ModuleList()
    self.norm_layers_2 = nn.ModuleList()
    # 循环创建多个层，并添加到对应的模块列表中
    for i in range(self.n_layers):
      # 创建多头注意力层，并添加到自注意力层模块列表中
      self.self_attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias, proximal_init=proximal_init))
      # 创建规范化层，并添加到规范化层模块列表0中
      self.norm_layers_0.append(LayerNorm(hidden_channels))
      # 创建多头注意力层，并添加到编码器-解码器注意力层模块列表中
      self.encdec_attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout))
      # 创建规范化层，并添加到规范化层模块列表1中
      self.norm_layers_1.append(LayerNorm(hidden_channels))
      # 创建前馈神经网络层，并添加到前馈神经网络层模块列表中
      self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout, causal=True))
      # 创建规范化层，并添加到规范化层模块列表2中
      self.norm_layers_2.append(LayerNorm(hidden_channels))
    
    # 定义前向传播函数
    def forward(self, x, x_mask, h, h_mask):
      """
      x: decoder input
      h: encoder output
      """
      # 生成自注意力掩码
      self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
      # 生成编码器-解码器注意力掩码
      encdec_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
      # 对输入进行掩码
      x = x * x_mask
      # 循环进行多层的自注意力、编码器-解码器注意力和前馈神经网络操作
      for i in range(self.n_layers):
        # 自注意力操作
        y = self.self_attn_layers[i](x, x, self_attn_mask)
        y = self.drop(y)
        x = self.norm_layers_0[i](x + y)
    
        # 编码器-解码器注意力操作
        y = self.encdec_attn_layers[i](x, h, encdec_attn_mask)
        y = self.drop(y)
        x = self.norm_layers_1[i](x + y)
        
        # 前馈神经网络操作
        y = self.ffn_layers[i](x, x_mask)
        y = self.drop(y)
        x = self.norm_layers_2[i](x + y)
      # 再次对结果进行掩码
      x = x * x_mask
      # 返回结果
      return x
class MultiHeadAttention(nn.Module):
  # 初始化多头注意力机制模块
  def __init__(self, channels, out_channels, n_heads, p_dropout=0., window_size=None, heads_share=True, block_length=None, proximal_bias=False, proximal_init=False):
    super().__init__()
    assert channels % n_heads == 0

    self.channels = channels
    self.out_channels = out_channels
    self.n_heads = n_heads
    self.p_dropout = p_dropout
    self.window_size = window_size
    self.heads_share = heads_share
    self.block_length = block_length
    self.proximal_bias = proximal_bias
    self.proximal_init = proximal_init
    self.attn = None

    self.k_channels = channels // n_heads
    # 定义卷积层，用于计算查询、键和值
    self.conv_q = nn.Conv1d(channels, channels, 1)
    self.conv_k = nn.Conv1d(channels, channels, 1)
    self.conv_v = nn.Conv1d(channels, channels, 1)
    self.conv_o = nn.Conv1d(channels, out_channels, 1)
    self.drop = nn.Dropout(p_dropout)

    if window_size is not None:
      n_heads_rel = 1 if heads_share else n_heads
      rel_stddev = self.k_channels**-0.5
      # 初始化相对位置编码的参数
      self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
      self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)

    # 初始化卷积层的权重
    nn.init.xavier_uniform_(self.conv_q.weight)
    nn.init.xavier_uniform_(self.conv_k.weight)
    nn.init.xavier_uniform_(self.conv_v.weight)
    if proximal_init:
      # 如果启用了近邻初始化，则将键的权重和偏置初始化为查询的权重和偏置
      with torch.no_grad():
        self.conv_k.weight.copy_(self.conv_q.weight)
        self.conv_k.bias.copy_(self.conv_q.bias)
      
  def forward(self, x, c, attn_mask=None):
    # 计算查询、键和值
    q = self.conv_q(x)
    k = self.conv_k(c)
    v = self.conv_v(c)
    
    # 进行注意力计算
    x, self.attn = self.attention(q, k, v, mask=attn_mask)

    # 将计算结果进行卷积得到输出
    x = self.conv_o(x)
    return x

  def attention(self, query, key, value, mask=None):
    # 重塑张量形状，以便进行多头注意力计算
    # 将形状从 [b, d, t] 转换为 [b, n_h, t, d_k]
    b, d, t_s, t_t = (*key.size(), query.size(2))
    query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
    # 将 key 转换为指定形状，并进行维度转置
    key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
    # 将 value 转换为指定形状，并进行维度转置
    value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

    # 计算注意力分数
    scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
    # 如果存在窗口大小限制
    if self.window_size is not None:
      # 确保输入序列长度相等
      assert t_s == t_t, "Relative attention is only available for self-attention."
      # 获取相对位置编码
      key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
      # 计算相对位置编码的注意力分数
      rel_logits = self._matmul_with_relative_keys(query /math.sqrt(self.k_channels), key_relative_embeddings)
      # 将相对位置编码的注意力分数转换为绝对位置编码的注意力分数
      scores_local = self._relative_position_to_absolute_position(rel_logits)
      scores = scores + scores_local
    # 如果存在近邻偏置
    if self.proximal_bias:
      # 确保输入序列长度相等
      assert t_s == t_t, "Proximal bias is only available for self-attention."
      # 添加近邻偏置
      scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
    # 如果存在掩码
    if mask is not None:
      # 使用掩码填充注意力分数
      scores = scores.masked_fill(mask == 0, -1e4)
      # 如果存在块长度限制
      if self.block_length is not None:
        # 确保输入序列长度相等
        assert t_s == t_t, "Local attention is only available for self-attention."
        # 创建块掩码
        block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
        scores = scores.masked_fill(block_mask == 0, -1e4)
    # 计算注意力权重
    p_attn = F.softmax(scores, dim=-1) # [b, n_h, t_t, t_s]
    # 对注意力权重进行 dropout
    p_attn = self.drop(p_attn)
    # 计算输出值
    output = torch.matmul(p_attn, value)
    # 如果存在窗口大小限制
    if self.window_size is not None:
      # 将绝对位置编码的注意力权重转换为相对位置编码的注意力权重
      relative_weights = self._absolute_position_to_relative_position(p_attn)
      # 获取相对位置编码的值
      value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
      # 将相对位置编码的值与相对位置编码的注意力权重相乘并加到输出上
      output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
    # 转置输出并重新排列维度
    output = output.transpose(2, 3).contiguous().view(b, d, t_t) # [b, n_h, t_t, d_k] -> [b, d, t_t]
    # 返回输出和注意力权重
    return output, p_attn

  # 计算相对位置编码的值
  def _matmul_with_relative_values(self, x, y):
    """
    x: [b, h, l, m]
    y: [h or 1, m, d]
    ret: [b, h, l, d]
    """
    # 矩阵相乘
    ret = torch.matmul(x, y.unsqueeze(0))
  # 返回变量 ret
  return ret

  # 使用相对键进行矩阵相乘
  def _matmul_with_relative_keys(self, x, y):
    """
    x: [b, h, l, d]
    y: [h or 1, m, d]
    ret: [b, h, l, m]
    """
    # 使用 torch.matmul 进行矩阵相乘
    ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
    return ret

  # 获取相对位置嵌入
  def _get_relative_embeddings(self, relative_embeddings, length):
    # 计算 pad_length
    2 * self.window_size + 1
    pad_length = max(length - (self.window_size + 1), 0)
    # 计算 slice_start_position
    slice_start_position = max((self.window_size + 1) - length, 0)
    # 计算 slice_end_position
    slice_end_position = slice_start_position + 2 * length - 1
    # 根据条件进行填充或者不填充
    if pad_length > 0:
      padded_relative_embeddings = F.pad(
          relative_embeddings,
          commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
    else:
      padded_relative_embeddings = relative_embeddings
    # 获取使用的相对嵌入
    used_relative_embeddings = padded_relative_embeddings[:,slice_start_position:slice_end_position]
    return used_relative_embeddings

  # 将相对位置转换为绝对位置
  def _relative_position_to_absolute_position(self, x):
    """
    x: [b, h, l, 2*l-1]
    ret: [b, h, l, l]
    """
    # 获取 x 的大小
    batch, heads, length, _ = x.size()
    # 在列上进行填充
    x = F.pad(x, commons.convert_pad_shape([[0,0],[0,0],[0,0],[0,1]]))

    # 拼接额外的元素，使其形状为 (len+1, 2*len-1)
    x_flat = x.view([batch, heads, length * 2 * length])
    x_flat = F.pad(x_flat, commons.convert_pad_shape([[0,0],[0,0],[0,length-1]]))

    # 重塑并切片出填充的元素
    x_final = x_flat.view([batch, heads, length+1, 2*length-1])[:, :, :length, length-1:]
    return x_final

  # 将绝对位置转换为相对位置
  def _absolute_position_to_relative_position(self, x):
    """
    x: [b, h, l, l]
    ret: [b, h, l, 2*l-1]
    """
    # 获取 x 的大小
    batch, heads, length, _ = x.size()
    # 沿着列进行填充
    x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length-1]]))
    x_flat = x.view([batch, heads, length**2 + length*(length -1)])
    # 在 x_flat 的开头添加 0，这样在 reshape 后会使元素产生偏移
    x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
    # 将 x_flat 转换成指定形状的张量
    x_final = x_flat.view([batch, heads, length, 2*length])[:,:,:,1:]
    # 返回处理后的张量
    return x_final

  def _attention_bias_proximal(self, length):
    """为自注意力提供偏置，以鼓励关注相邻位置。
    Args:
      length: 一个整数标量。
    Returns:
      一个形状为 [1, 1, length, length] 的张量
    """
    # 创建一个长度为 length 的浮点数张量
    r = torch.arange(length, dtype=torch.float32)
    # 计算位置之间的差异
    diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
    # 对差异取绝对值并取对数，然后添加维度
    return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)
# 定义一个名为 FFN 的神经网络模块
class FFN(nn.Module):
  # 初始化函数，设置输入通道数、输出通道数、滤波器通道数、卷积核大小、丢弃率、激活函数和是否是因果卷积
  def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0., activation=None, causal=False):
    super().__init__()
    # 设置各个参数
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.activation = activation
    self.causal = causal

    # 根据是否是因果卷积选择不同的填充方式
    if causal:
      self.padding = self._causal_padding
    else:
      self.padding = self._same_padding

    # 创建两个一维卷积层和一个丢弃层
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
    self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
    self.drop = nn.Dropout(p_dropout)

  # 前向传播函数，接受输入和输入的掩码
  def forward(self, x, x_mask):
    # 对输入进行卷积和填充
    x = self.conv_1(self.padding(x * x_mask))
    # 根据激活函数类型进行激活
    if self.activation == "gelu":
      x = x * torch.sigmoid(1.702 * x)
    else:
      x = torch.relu(x)
    # 对结果进行丢弃
    x = self.drop(x)
    # 再次进行卷积和填充
    x = self.conv_2(self.padding(x * x_mask))
    # 返回结果乘以输入的掩码
    return x * x_mask
  
  # 定义因果填充函数
  def _causal_padding(self, x):
    # 如果卷积核大小为1，则不进行填充
    if self.kernel_size == 1:
      return x
    # 计算因果填充的左右填充数
    pad_l = self.kernel_size - 1
    pad_r = 0
    padding = [[0, 0], [0, 0], [pad_l, pad_r]]
    # 对输入进行填充
    x = F.pad(x, commons.convert_pad_shape(padding))
    return x

  # 定义相同填充函数
  def _same_padding(self, x):
    # 如果卷积核大小为1，则不进行填充
    if self.kernel_size == 1:
      return x
    # 计算相同填充的左右填充数
    pad_l = (self.kernel_size - 1) // 2
    pad_r = self.kernel_size // 2
    padding = [[0, 0], [0, 0], [pad_l, pad_r]]
    # 对输入进行填充
    x = F.pad(x, commons.convert_pad_shape(padding))
    return x
```