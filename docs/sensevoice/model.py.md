# `.\SenseVoiceSmall-src\model.py`

```
# 导入时间模块
import time
# 导入 PyTorch 框架
import torch
# 从 PyTorch 导入神经网络模块
from torch import nn
# 导入 PyTorch 的功能性操作模块
import torch.nn.functional as F
# 导入类型注解模块
from typing import Iterable, Optional

# 从 funasr.register 模块导入 tables
from funasr.register import tables
# 从 funasr.models.ctc.ctc 模块导入 CTC 类
from funasr.models.ctc.ctc import CTC
# 从 funasr.utils.datadir_writer 导入 DatadirWriter 类
from funasr.utils.datadir_writer import DatadirWriter
# 从 funasr.models.paraformer.search 导入 Hypothesis 类
from funasr.models.paraformer.search import Hypothesis
# 从 funasr.train_utils.device_funcs 导入 force_gatherable 函数
from funasr.train_utils.device_funcs import force_gatherable
# 从 funasr.losses.label_smoothing_loss 导入 LabelSmoothingLoss 类
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
# 从 funasr.metrics.compute_acc 导入 compute_accuracy 和 th_accuracy 函数
from funasr.metrics.compute_acc import compute_accuracy, th_accuracy
# 从 funasr.utils.load_utils 导入 load_audio_text_image_video 和 extract_fbank 函数
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank


# 定义一个正弦位置编码器类，继承自 torch.nn.Module
class SinusoidalPositionEncoder(torch.nn.Module):
    """位置编码器，用于生成正弦和余弦位置编码"""

    # 初始化函数，接收模型维度和丢弃率作为参数
    def __int__(self, d_model=80, dropout_rate=0.1):
        # 该函数目前未实现
        pass

    # 编码函数，接收位置张量、深度和数据类型
    def encode(
        self, positions: torch.Tensor = None, depth: int = None, dtype: torch.dtype = torch.float32
    ):
        # 获取批次大小
        batch_size = positions.size(0)
        # 将位置张量转换为指定的数据类型
        positions = positions.type(dtype)
        # 获取设备信息
        device = positions.device
        # 计算对数时间尺度增量
        log_timescale_increment = torch.log(torch.tensor([10000], dtype=dtype, device=device)) / (
            depth / 2 - 1
        )
        # 计算逆时间尺度
        inv_timescales = torch.exp(
            torch.arange(depth / 2, device=device).type(dtype) * (-log_timescale_increment)
        )
        # 重塑逆时间尺度的形状
        inv_timescales = torch.reshape(inv_timescales, [batch_size, -1])
        # 计算缩放后的时间
        scaled_time = torch.reshape(positions, [1, -1, 1]) * torch.reshape(
            inv_timescales, [1, 1, -1]
        )
        # 计算正弦和余弦编码
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        # 返回编码并转换为指定的数据类型
        return encoding.type(dtype)

    # 前向传播函数，接收输入张量
    def forward(self, x):
        # 获取批次大小、时间步长和输入维度
        batch_size, timesteps, input_dim = x.size()
        # 生成位置张量
        positions = torch.arange(1, timesteps + 1, device=x.device)[None, :]
        # 计算位置编码
        position_encoding = self.encode(positions, input_dim, x.dtype).to(x.device)

        # 返回输入张量和位置编码的和
        return x + position_encoding


# 定义一个位置逐层前馈网络类，继承自 torch.nn.Module
class PositionwiseFeedForward(torch.nn.Module):
    """逐位置前馈层。

    参数：
        idim (int): 输入维度。
        hidden_units (int): 隐藏单元的数量。
        dropout_rate (float): 丢弃率。

    """

    # 初始化函数，接收输入维度、隐藏单元数量、丢弃率和激活函数
    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """构造一个 PositionwiseFeedForward 对象。"""
        # 调用父类构造函数
        super(PositionwiseFeedForward, self).__init__()
        # 创建第一个线性层
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        # 创建第二个线性层
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        # 创建丢弃层
        self.dropout = torch.nn.Dropout(dropout_rate)
        # 设置激活函数
        self.activation = activation

    # 前向传播函数，接收输入张量
    def forward(self, x):
        """前向传播函数。"""
        # 依次通过两个线性层和激活函数，并应用丢弃层
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


# 定义一个多头注意力层类，继承自 nn.Module
class MultiHeadedAttentionSANM(nn.Module):
    """多头注意力层。

    参数：
        n_head (int): 头的数量。
        n_feat (int): 特征的数量。
        dropout_rate (float): 丢弃率。

    """
    # 初始化多头注意力对象的构造函数
    def __init__(
        self,  # 构造函数的第一个参数，表示类实例
        n_head,  # 注意力头的数量
        in_feat,  # 输入特征的维度
        n_feat,  # 输出特征的维度
        dropout_rate,  # dropout的比率
        kernel_size,  # 卷积核的大小
        sanm_shfit=0,  # SANM的偏移量，默认为0
        lora_list=None,  # LoRA参数列表，默认为None
        lora_rank=8,  # LoRA的秩，默认为8
        lora_alpha=16,  # LoRA的α值，默认为16
        lora_dropout=0.1,  # LoRA的dropout比率，默认为0.1
    ):
        """构造一个MultiHeadedAttention对象。"""
        super().__init__()  # 调用父类的构造函数
        assert n_feat % n_head == 0  # 确保输出特征能被注意力头数整除
        # 我们假设 d_v 始终等于 d_k
        self.d_k = n_feat // n_head  # 每个头的特征维度
        self.h = n_head  # 保存头的数量
        # self.linear_q = nn.Linear(n_feat, n_feat)  # 初始化查询的线性变换层
        # self.linear_k = nn.Linear(n_feat, n_feat)  # 初始化键的线性变换层
        # self.linear_v = nn.Linear(n_feat, n_feat)  # 初始化值的线性变换层

        self.linear_out = nn.Linear(n_feat, n_feat)  # 输出的线性变换层
        self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)  # 查询、键和值的线性变换层
        self.attn = None  # 初始化注意力机制的占位符
        self.dropout = nn.Dropout(p=dropout_rate)  # 初始化dropout层

        # 创建一个一维卷积块
        self.fsmn_block = nn.Conv1d(
            n_feat,  # 输入通道数
            n_feat,  # 输出通道数
            kernel_size,  # 卷积核大小
            stride=1,  # 步幅为1
            padding=0,  # 不使用填充
            groups=n_feat,  # 每个输入通道对应一个输出通道
            bias=False  # 不使用偏置
        )
        # 计算填充的大小
        left_padding = (kernel_size - 1) // 2  # 左侧填充大小
        if sanm_shfit > 0:  # 如果偏移量大于0
            left_padding = left_padding + sanm_shfit  # 增加左侧填充
        right_padding = kernel_size - 1 - left_padding  # 右侧填充大小
        # 定义一个常数填充函数
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)  # 用0.0填充

    # 前向传播函数，处理FSMN层
    def forward_fsmn(self, inputs, mask, mask_shfit_chunk=None):
        b, t, d = inputs.size()  # 获取输入的batch大小、时间步和特征维度
        if mask is not None:  # 如果存在mask
            mask = torch.reshape(mask, (b, -1, 1))  # 调整mask的形状
            if mask_shfit_chunk is not None:  # 如果存在mask偏移块
                mask = mask * mask_shfit_chunk  # 应用偏移块
            inputs = inputs * mask  # 将输入与mask相乘

        x = inputs.transpose(1, 2)  # 交换维度，将时间步和特征维度调换
        x = self.pad_fn(x)  # 应用填充函数
        x = self.fsmn_block(x)  # 通过FSMN块进行卷积操作
        x = x.transpose(1, 2)  # 再次交换维度，恢复原始维度顺序
        x += inputs  # 将原输入与卷积输出相加
        x = self.dropout(x)  # 应用dropout
        if mask is not None:  # 如果存在mask
            x = x * mask  # 将输出与mask相乘
        return x  # 返回结果

    # 前向传播函数，处理查询、键和值
    def forward_qkv(self, x):
        """变换查询、键和值。

        Args:
            query (torch.Tensor): 查询张量 (#batch, time1, size)。
            key (torch.Tensor): 键张量 (#batch, time2, size)。
            value (torch.Tensor): 值张量 (#batch, time2, size)。

        Returns:
            torch.Tensor: 变换后的查询张量 (#batch, n_head, time1, d_k)。
            torch.Tensor: 变换后的键张量 (#batch, n_head, time2, d_k)。
            torch.Tensor: 变换后的值张量 (#batch, n_head, time2, d_k)。
        """
        b, t, d = x.size()  # 获取输入的batch大小、时间步和特征维度
        q_k_v = self.linear_q_k_v(x)  # 通过线性层转换输入
        q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)  # 将q_k_v拆分为q、k和v
        q_h = torch.reshape(q, (b, t, self.h, self.d_k)).transpose(1, 2)  # 重塑并交换维度
        k_h = torch.reshape(k, (b, t, self.h, self.d_k)).transpose(1, 2)  # 重塑并交换维度
        v_h = torch.reshape(v, (b, t, self.h, self.d_k)).transpose(1, 2)  # 重塑并交换维度

        return q_h, k_h, v_h, v  # 返回查询、键和值的变换结果
    # 定义前向注意力计算函数，生成注意力上下文向量
    def forward_attention(self, value, scores, mask, mask_att_chunk_encoder=None):
        """计算注意力上下文向量。

        参数：
            value (torch.Tensor): 转换后的值（#batch, n_head, time2, d_k）。
            scores (torch.Tensor): 注意力分数（#batch, n_head, time1, time2）。
            mask (torch.Tensor): 掩码（#batch, 1, time2）或（#batch, time1, time2）。

        返回：
            torch.Tensor: 加权后的转换值（#batch, time1, d_model），
                受注意力分数（#batch, time1, time2）影响。

        """
        # 获取输入张量的批次大小
        n_batch = value.size(0)
        # 如果存在掩码
        if mask is not None:
            # 如果存在额外的编码器掩码，则对当前掩码进行处理
            if mask_att_chunk_encoder is not None:
                mask = mask * mask_att_chunk_encoder

            # 扩展掩码维度并检查值是否为0（表示被遮蔽）
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)

            # 设置最小值为负无穷，用于掩蔽
            min_value = -float(
                "inf"
            )  # float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            # 在分数中应用掩码，掩蔽掉特定位置
            scores = scores.masked_fill(mask, min_value)
            # 对分数进行softmax计算，得到注意力权重，并对掩蔽位置设置为0.0
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            # 如果没有掩码，仅对分数进行softmax计算
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        # 应用dropout以增加模型的鲁棒性
        p_attn = self.dropout(attn)
        # 计算注意力加权的值
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        # 重新排列和调整维度以适配后续层
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        # 通过线性层输出最终结果
        return self.linear_out(x)  # (batch, time1, d_model)

    # 定义前向传播函数，计算缩放点积注意力
    def forward(self, x, mask, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        """计算缩放点积注意力。

        参数：
            query (torch.Tensor): 查询张量（#batch, time1, size）。
            key (torch.Tensor): 键张量（#batch, time2, size）。
            value (torch.Tensor): 值张量（#batch, time2, size）。
            mask (torch.Tensor): 掩码张量（#batch, 1, time2）或
                （#batch, time1, time2）。

        返回：
            torch.Tensor: 输出张量（#batch, time1, d_model）。

        """
        # 通过前向传播函数计算查询、键、值的表示
        q_h, k_h, v_h, v = self.forward_qkv(x)
        # 计算 FSMN 内存
        fsmn_memory = self.forward_fsmn(v, mask, mask_shfit_chunk)
        # 缩放查询以增强稳定性
        q_h = q_h * self.d_k ** (-0.5)
        # 计算查询与键的点积以获得分数
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        # 通过前向注意力函数计算注意力输出
        att_outs = self.forward_attention(v_h, scores, mask, mask_att_chunk_encoder)
        # 返回注意力输出与 FSMN 内存的和
        return att_outs + fsmn_memory
    # 定义前向处理函数，计算缩放点积注意力
    def forward_chunk(self, x, cache=None, chunk_size=None, look_back=0):
        """计算缩放点积注意力。
    
        参数：
            query (torch.Tensor): 查询张量（#batch, time1, size）。
            key (torch.Tensor): 键张量（#batch, time2, size）。
            value (torch.Tensor): 值张量（#batch, time2, size）。
            mask (torch.Tensor): 掩码张量（#batch, 1, time2）或
                (#batch, time1, time2)。
    
        返回：
            torch.Tensor: 输出张量（#batch, time1, d_model）。
    
        """
        # 从输入 x 中前向计算查询、键和值的表示
        q_h, k_h, v_h, v = self.forward_qkv(x)
        # 检查是否有 chunk_size 和 look_back 参数
        if chunk_size is not None and look_back > 0 or look_back == -1:
            # 如果缓存存在
            if cache is not None:
                # 从键和值中去除超出 chunk_size 的部分
                k_h_stride = k_h[:, :, : -(chunk_size[2]), :]
                v_h_stride = v_h[:, :, : -(chunk_size[2]), :]
                # 将缓存的键和值与当前的键和值连接
                k_h = torch.cat((cache["k"], k_h), dim=2)
                v_h = torch.cat((cache["v"], v_h), dim=2)
    
                # 更新缓存的键和值
                cache["k"] = torch.cat((cache["k"], k_h_stride), dim=2)
                cache["v"] = torch.cat((cache["v"], v_h_stride), dim=2)
                # 如果 look_back 不是 -1，裁剪缓存中的键和值
                if look_back != -1:
                    cache["k"] = cache["k"][:, :, -(look_back * chunk_size[1]) :, :]
                    cache["v"] = cache["v"][:, :, -(look_back * chunk_size[1]) :, :]
            else:
                # 如果缓存不存在，创建一个新的缓存
                cache_tmp = {
                    "k": k_h[:, :, : -(chunk_size[2]), :],
                    "v": v_h[:, :, : -(chunk_size[2]), :],
                }
                cache = cache_tmp
        # 通过前向 FSMN 计算内存
        fsmn_memory = self.forward_fsmn(v, None)
        # 缩放查询表示
        q_h = q_h * self.d_k ** (-0.5)
        # 计算注意力分数
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        # 通过前向注意力机制计算输出
        att_outs = self.forward_attention(v_h, scores, None)
        # 返回注意力输出与 FSMN 内存的和，以及缓存
        return att_outs + fsmn_memory, cache
# 自定义一个层归一化类，继承自 nn.LayerNorm
class LayerNorm(nn.LayerNorm):
    # 初始化方法，接收任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)

    # 前向传播方法，接收输入数据
    def forward(self, input):
        # 使用 PyTorch 的层归一化功能，进行归一化处理
        output = F.layer_norm(
            # 将输入数据转换为浮点类型
            input.float(),
            # 归一化的形状
            self.normalized_shape,
            # 如果存在权重，则转换为浮点类型，否则为 None
            self.weight.float() if self.weight is not None else None,
            # 如果存在偏置，则转换为浮点类型，否则为 None
            self.bias.float() if self.bias is not None else None,
            # 设定的 epsilon 值
            self.eps,
        )
        # 返回与输入相同类型的输出
        return output.type_as(input)


# 定义一个序列掩码函数，接收长度、最大长度、数据类型和设备
def sequence_mask(lengths, maxlen=None, dtype=torch.float32, device=None):
    # 如果最大长度未指定，则取长度的最大值
    if maxlen is None:
        maxlen = lengths.max()
    # 创建一个行向量，从 0 到 maxlen
    row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
    # 在最后一维扩展 lengths，形成矩阵
    matrix = torch.unsqueeze(lengths, dim=-1)
    # 生成掩码矩阵，比较 row_vector 和 matrix
    mask = row_vector < matrix
    # 将掩码从计算图中分离
    mask = mask.detach()

    # 如果指定了设备，则将掩码转换为指定的类型并移动到设备上，否则仅转换类型
    return mask.type(dtype).to(device) if device is not None else mask.type(dtype)


# 定义一个编码器层类，继承自 nn.Module
class EncoderLayerSANM(nn.Module):
    # 初始化方法，接收多个参数
    def __init__(
        self,
        in_size,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """构造 EncoderLayer 对象。"""
        # 调用父类的初始化方法
        super(EncoderLayerSANM, self).__init__()
        # 保存自注意力机制对象
        self.self_attn = self_attn
        # 保存前馈神经网络对象
        self.feed_forward = feed_forward
        # 创建第一层归一化对象
        self.norm1 = LayerNorm(in_size)
        # 创建第二层归一化对象
        self.norm2 = LayerNorm(size)
        # 创建丢弃层对象，设定丢弃率
        self.dropout = nn.Dropout(dropout_rate)
        # 保存输入尺寸
        self.in_size = in_size
        # 保存层大小
        self.size = size
        # 保存是否在前面进行归一化的标志
        self.normalize_before = normalize_before
        # 保存是否在后面进行拼接的标志
        self.concat_after = concat_after
        # 如果选择在后面拼接，创建线性层
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        # 保存随机深度率
        self.stochastic_depth_rate = stochastic_depth_rate
        # 保存丢弃率
        self.dropout_rate = dropout_rate
    # 定义前向传播函数，计算编码特征
    def forward(self, x, mask, cache=None, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        """计算编码特征。

        参数：
            x_input (torch.Tensor): 输入张量 (#batch, time, size)。
            mask (torch.Tensor): 输入的掩码张量 (#batch, time)。
            cache (torch.Tensor): 输入的缓存张量 (#batch, time - 1, size)。

        返回：
            torch.Tensor: 输出张量 (#batch, time, size)。
            torch.Tensor: 掩码张量 (#batch, time)。

        """
        skip_layer = False  # 初始化跳过层的标志为假
        # 在随机深度下，残差连接 `x + f(x)` 变为
        # `x <- x + 1 / (1 - p) * f(x)` 在训练时间。
        stoch_layer_coeff = 1.0  # 随机层系数初始为1.0
        if self.training and self.stochastic_depth_rate > 0:  # 如果处于训练状态并且随机深度率大于0
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate  # 根据随机数判断是否跳过层
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)  # 计算随机层系数

        if skip_layer:  # 如果决定跳过层
            if cache is not None:  # 如果缓存不为空
                x = torch.cat([cache, x], dim=1)  # 将缓存和输入张量沿时间维度拼接
            return x, mask  # 返回输入和掩码

        residual = x  # 保存输入作为残差
        if self.normalize_before:  # 如果在此之前进行归一化
            x = self.norm1(x)  # 对输入进行归一化

        if self.concat_after:  # 如果在之后拼接
            x_concat = torch.cat(  # 拼接输入和自注意力的输出
                (
                    x,  # 输入
                    self.self_attn(  # 自注意力模块
                        x,  # 输入
                        mask,  # 掩码
                        mask_shfit_chunk=mask_shfit_chunk,  # 移位掩码
                        mask_att_chunk_encoder=mask_att_chunk_encoder,  # 编码器的注意力掩码
                    ),
                ),
                dim=-1,  # 沿最后一个维度拼接
            )
            if self.in_size == self.size:  # 如果输入和输出大小相同
                x = residual + stoch_layer_coeff * self.concat_linear(x_concat)  # 进行残差连接
            else:  # 如果大小不同
                x = stoch_layer_coeff * self.concat_linear(x_concat)  # 直接输出拼接结果
        else:  # 如果不进行拼接
            if self.in_size == self.size:  # 如果输入和输出大小相同
                x = residual + stoch_layer_coeff * self.dropout(  # 进行残差连接并丢弃一些激活值
                    self.self_attn(  # 自注意力模块
                        x,  # 输入
                        mask,  # 掩码
                        mask_shfit_chunk=mask_shfit_chunk,  # 移位掩码
                        mask_att_chunk_encoder=mask_att_chunk_encoder,  # 编码器的注意力掩码
                    )
                )
            else:  # 如果大小不同
                x = stoch_layer_coeff * self.dropout(  # 仅丢弃激活值
                    self.self_attn(  # 自注意力模块
                        x,  # 输入
                        mask,  # 掩码
                        mask_shfit_chunk=mask_shfit_chunk,  # 移位掩码
                        mask_att_chunk_encoder=mask_att_chunk_encoder,  # 编码器的注意力掩码
                    )
                )
        if not self.normalize_before:  # 如果在此之前不进行归一化
            x = self.norm1(x)  # 对输出进行归一化

        residual = x  # 保存当前输出作为新的残差
        if self.normalize_before:  # 如果在此之前进行归一化
            x = self.norm2(x)  # 对输出进行归一化
        x = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x))  # 进行残差连接并经过前馈网络
        if not self.normalize_before:  # 如果在此之前不进行归一化
            x = self.norm2(x)  # 对输出进行归一化

        return x, mask, cache, mask_shfit_chunk, mask_att_chunk_encoder  # 返回输出、掩码和缓存等
    # 定义一个方法，计算编码特征
    def forward_chunk(self, x, cache=None, chunk_size=None, look_back=0):
        """计算编码特征。

        参数：
            x_input (torch.Tensor): 输入张量 (#batch, time, size)。
            mask (torch.Tensor): 输入的掩码张量 (#batch, time)。
            cache (torch.Tensor): 输入的缓存张量 (#batch, time - 1, size)。

        返回：
            torch.Tensor: 输出张量 (#batch, time, size)。
            torch.Tensor: 掩码张量 (#batch, time)。

        """

        # 将输入张量赋值给残差变量
        residual = x
        # 如果设置了在计算前进行归一化
        if self.normalize_before:
            # 对输入张量进行归一化处理
            x = self.norm1(x)

        # 检查输入维度与输出维度是否一致
        if self.in_size == self.size:
            # 调用自注意力机制的前向计算，并获取注意力和缓存
            attn, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)
            # 将残差与注意力输出相加
            x = residual + attn
        else:
            # 调用自注意力机制的前向计算，并获取输出和缓存
            x, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)

        # 如果没有在计算前进行归一化
        if not self.normalize_before:
            # 对输出张量进行归一化处理
            x = self.norm1(x)

        # 更新残差变量为当前输出
        residual = x
        # 如果设置了在计算前进行归一化
        if self.normalize_before:
            # 对输出张量进行归一化处理
            x = self.norm2(x)
        # 将残差与前馈网络的输出相加
        x = residual + self.feed_forward(x)
        # 如果没有在计算前进行归一化
        if not self.normalize_before:
            # 对输出张量进行归一化处理
            x = self.norm2(x)

        # 返回输出张量和缓存
        return x, cache
# 注册一个名为 "encoder_classes" 的表，并定义类 "SenseVoiceEncoderSmall"
@tables.register("encoder_classes", "SenseVoiceEncoderSmall")
# 定义一个神经网络模块类，继承自 nn.Module
class SenseVoiceEncoderSmall(nn.Module):
    """
    作者: 阿里巴巴集团 DAMO 学院的语音实验室
    SCAMA: 用于在线端到端语音识别的流式块感知多头注意力机制
    https://arxiv.org/abs/2006.01713
    """

    # 初始化方法，设置模块的参数
    def __init__(
        self,
        # 输入特征的大小
        input_size: int,
        # 输出特征的大小，默认为 256
        output_size: int = 256,
        # 注意力头的数量，默认为 4
        attention_heads: int = 4,
        # 线性单元的数量，默认为 2048
        linear_units: int = 2048,
        # 模块的块数量，默认为 6
        num_blocks: int = 6,
        # 转置块的数量，默认为 0
        tp_blocks: int = 0,
        # dropout 比率，默认为 0.1
        dropout_rate: float = 0.1,
        # 位置编码的 dropout 比率，默认为 0.1
        positional_dropout_rate: float = 0.1,
        # 注意力机制的 dropout 比率，默认为 0.0
        attention_dropout_rate: float = 0.0,
        # 随机深度比率，默认为 0.0
        stochastic_depth_rate: float = 0.0,
        # 输入层类型，默认为 "conv2d"
        input_layer: Optional[str] = "conv2d",
        # 位置编码类，默认为 SinusoidalPositionEncoder
        pos_enc_class=SinusoidalPositionEncoder,
        # 是否在前进行归一化，默认为 True
        normalize_before: bool = True,
        # 是否在后进行拼接，默认为 False
        concat_after: bool = False,
        # 位置逐层的类型，默认为 "linear"
        positionwise_layer_type: str = "linear",
        # 位置逐层卷积的内核大小，默认为 1
        positionwise_conv_kernel_size: int = 1,
        # 填充索引，默认为 -1
        padding_idx: int = -1,
        # 卷积的内核大小，默认为 11
        kernel_size: int = 11,
        # SANM 的移位，默认为 0
        sanm_shfit: int = 0,
        # 自注意力层的类型，默认为 "sanm"
        selfattention_layer_type: str = "sanm",
        # 其他关键字参数
        **kwargs,
    ):
        # 调用父类构造函数初始化
        super().__init__()
        # 保存输出大小的参数
        self._output_size = output_size

        # 初始化正弦位置编码器
        self.embed = SinusoidalPositionEncoder()

        # 记录是否在前向传播前进行归一化
        self.normalize_before = normalize_before

        # 定义位置前馈层类
        positionwise_layer = PositionwiseFeedForward
        # 定义位置前馈层的参数
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
        )

        # 定义多头自注意力层类
        encoder_selfattn_layer = MultiHeadedAttentionSANM
        # 定义自注意力层的参数
        encoder_selfattn_layer_args0 = (
            attention_heads,
            input_size,
            output_size,
            attention_dropout_rate,
            kernel_size,
            sanm_shfit,
        )
        # 定义另一组自注意力层的参数
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            output_size,
            attention_dropout_rate,
            kernel_size,
            sanm_shfit,
        )

        # 初始化第一个编码器层的模块列表
        self.encoders0 = nn.ModuleList(
            [
                # 创建 EncoderLayerSANM 实例，传入各参数
                EncoderLayerSANM(
                    input_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args0),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                # 生成 1 个编码器层
                for i in range(1)
            ]
        )
        # 初始化后续编码器层的模块列表
        self.encoders = nn.ModuleList(
            [
                # 创建 EncoderLayerSANM 实例，传入各参数
                EncoderLayerSANM(
                    output_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                # 生成 num_blocks - 1 个编码器层
                for i in range(num_blocks - 1)
            ]
        )

        # 初始化 tp_blocks 数量的 TP 编码器层模块列表
        self.tp_encoders = nn.ModuleList(
            [
                # 创建 EncoderLayerSANM 实例，传入各参数
                EncoderLayerSANM(
                    output_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                # 生成 tp_blocks 个编码器层
                for i in range(tp_blocks)
            ]
        )

        # 初始化输出的归一化层
        self.after_norm = LayerNorm(output_size)

        # 初始化 TP 的归一化层
        self.tp_norm = LayerNorm(output_size)

    # 定义输出大小的 getter 方法
    def output_size(self) -> int:
        # 返回保存的输出大小
        return self._output_size

    # 定义前向传播方法
    def forward(
        # 输入张量和输入长度的张量
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ):
        """在张量中嵌入位置。"""
        # 生成序列掩码，使用 ilens 的设备信息，并添加一个新的维度
        masks = sequence_mask(ilens, device=ilens.device)[:, None, :]

        # 将 xs_pad 的值乘以输出大小的平方根
        xs_pad *= self.output_size() ** 0.5

        # 对 xs_pad 进行嵌入操作
        xs_pad = self.embed(xs_pad)

        # 前向传播到 encoder1
        for layer_idx, encoder_layer in enumerate(self.encoders0):
            # 通过当前编码层处理 xs_pad 和 masks，获取输出
            encoder_outs = encoder_layer(xs_pad, masks)
            # 更新 xs_pad 和 masks 为编码层的输出
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        # 前向传播到 encoder
        for layer_idx, encoder_layer in enumerate(self.encoders):
            # 通过当前编码层处理 xs_pad 和 masks，获取输出
            encoder_outs = encoder_layer(xs_pad, masks)
            # 更新 xs_pad 和 masks 为编码层的输出
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        # 对 xs_pad 进行后归一化处理
        xs_pad = self.after_norm(xs_pad)

        # 前向传播到 encoder2
        # 计算有效长度，即去掉掩码维度后的总和，并转换为整数
        olens = masks.squeeze(1).sum(1).int()

        # 前向传播到 tp_encoders
        for layer_idx, encoder_layer in enumerate(self.tp_encoders):
            # 通过当前编码层处理 xs_pad 和 masks，获取输出
            encoder_outs = encoder_layer(xs_pad, masks)
            # 更新 xs_pad 和 masks 为编码层的输出
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        # 对 xs_pad 进行 tp 归一化处理
        xs_pad = self.tp_norm(xs_pad)
        # 返回处理后的 xs_pad 和有效长度 olens
        return xs_pad, olens
@tables.register("model_classes", "SenseVoiceSmall")  # 将类注册到表格中，标识为 "model_classes" 和 "SenseVoiceSmall"
class SenseVoiceSmall(nn.Module):  # 定义一个名为 SenseVoiceSmall 的类，继承自 nn.Module
    """CTC-attention hybrid Encoder-Decoder model"""  # 类的文档字符串，描述模型类型

    def __init__(  # 初始化方法，定义类的构造函数
        self,  # 类的实例
        specaug: str = None,  # 语音增强类型，默认为 None
        specaug_conf: dict = None,  # 语音增强配置，默认为 None
        normalize: str = None,  # 归一化类型，默认为 None
        normalize_conf: dict = None,  # 归一化配置，默认为 None
        encoder: str = None,  # 编码器类型，默认为 None
        encoder_conf: dict = None,  # 编码器配置，默认为 None
        ctc_conf: dict = None,  # CTC 配置，默认为 None
        input_size: int = 80,  # 输入特征大小，默认为 80
        vocab_size: int = -1,  # 词汇表大小，默认为 -1
        ignore_id: int = -1,  # 忽略的 ID，默认为 -1
        blank_id: int = 0,  # 空白符 ID，默认为 0
        sos: int = 1,  # 起始符 ID，默认为 1
        eos: int = 2,  # 结束符 ID，默认为 2
        length_normalized_loss: bool = False,  # 是否使用长度归一化损失，默认为 False
        **kwargs,  # 其他可选参数
    ):

        super().__init__()  # 调用父类的初始化方法

        if specaug is not None:  # 如果指定了语音增强
            specaug_class = tables.specaug_classes.get(specaug)  # 从表格中获取对应的语音增强类
            specaug = specaug_class(**specaug_conf)  # 实例化语音增强对象，传入配置
        if normalize is not None:  # 如果指定了归一化
            normalize_class = tables.normalize_classes.get(normalize)  # 从表格中获取对应的归一化类
            normalize = normalize_class(**normalize_conf)  # 实例化归一化对象，传入配置
        encoder_class = tables.encoder_classes.get(encoder)  # 从表格中获取对应的编码器类
        encoder = encoder_class(input_size=input_size, **encoder_conf)  # 实例化编码器对象，传入输入大小和配置
        encoder_output_size = encoder.output_size()  # 获取编码器的输出大小

        if ctc_conf is None:  # 如果 CTC 配置为 None
            ctc_conf = {}  # 初始化为空字典
        ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size, **ctc_conf)  # 创建 CTC 对象

        self.blank_id = blank_id  # 保存空白符 ID
        self.sos = sos if sos is not None else vocab_size - 1  # 设置起始符 ID，若未指定则为词汇表大小 - 1
        self.eos = eos if eos is not None else vocab_size - 1  # 设置结束符 ID，若未指定则为词汇表大小 - 1
        self.vocab_size = vocab_size  # 保存词汇表大小
        self.ignore_id = ignore_id  # 保存忽略 ID
        self.specaug = specaug  # 保存语音增强对象
        self.normalize = normalize  # 保存归一化对象
        self.encoder = encoder  # 保存编码器对象
        self.error_calculator = None  # 初始化错误计算器为 None

        self.ctc = ctc  # 保存 CTC 对象

        self.length_normalized_loss = length_normalized_loss  # 保存是否使用长度归一化损失的标志
        self.encoder_output_size = encoder_output_size  # 保存编码器输出大小

        # 定义语言 ID 字典，映射语言名称到 ID
        self.lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
        # 定义语言 ID 整数字典，映射整数到语言 ID
        self.lid_int_dict = {24884: 3, 24885: 4, 24888: 7, 24892: 11, 24896: 12, 24992: 13}
        # 定义文本规范化字典，映射规范化类型到 ID
        self.textnorm_dict = {"withitn": 14, "woitn": 15}
        # 定义文本规范化整数字典，映射整数到规范化 ID
        self.textnorm_int_dict = {25016: 14, 25017: 15}
        # 创建嵌入层，维度为语言 ID 数量加上文本规范化数量加上 7，输入大小为 input_size
        self.embed = torch.nn.Embedding(7 + len(self.lid_dict) + len(self.textnorm_dict), input_size)
        # 定义情感字典，映射情感类型到 ID
        self.emo_dict = {"unk": 25009, "happy": 25001, "sad": 25002, "angry": 25003, "neutral": 25004}
        
        # 创建标签平滑损失对象，传入词汇表大小、忽略 ID 和其他配置
        self.criterion_att = LabelSmoothingLoss(
            size=self.vocab_size,  # 词汇表大小
            padding_idx=self.ignore_id,  # 填充的 ID
            smoothing=kwargs.get("lsm_weight", 0.0),  # 平滑因子
            normalize_length=self.length_normalized_loss,  # 是否使用长度归一化
        )
    
    @staticmethod
    def from_pretrained(model:str=None, **kwargs):  # 静态方法，从预训练模型加载
        from funasr import AutoModel  # 从 funasr 导入 AutoModel
        model, kwargs = AutoModel.build_model(model=model, trust_remote_code=True, **kwargs)  # 构建模型并信任远程代码
        
        return model, kwargs  # 返回构建的模型和参数

    def forward(  # 前向传播方法
        self,  # 类的实例
        speech: torch.Tensor,  # 输入的语音张量
        speech_lengths: torch.Tensor,  # 语音长度的张量
        text: torch.Tensor,  # 输入的文本张量
        text_lengths: torch.Tensor,  # 文本长度的张量
        **kwargs,  # 其他可选参数
    ):
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        # import pdb;  # 导入调试模块
        # pdb.set_trace()  # 设置断点以便调试
        if len(text_lengths.size()) > 1:  # 检查文本长度是否为多维
            text_lengths = text_lengths[:, 0]  # 选择第一维的长度值
        if len(speech_lengths.size()) > 1:  # 检查语音长度是否为多维
            speech_lengths = speech_lengths[:, 0]  # 选择第一维的长度值

        batch_size = speech.shape[0]  # 获取批次大小

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, text)  # 编码语音，输出编码结果和长度

        loss_ctc, cer_ctc = None, None  # 初始化CTC损失和字符错误率
        loss_rich, acc_rich = None, None  # 初始化丰富损失和准确率
        stats = dict()  # 创建一个字典用于存储统计信息

        loss_ctc, cer_ctc = self._calc_ctc_loss(  # 计算CTC损失
            encoder_out[:, 4:, :], encoder_out_lens - 4, text[:, 4:], text_lengths - 4  # 处理编码输出，忽略前4个时间步
        )

        loss_rich, acc_rich = self._calc_rich_ce_loss(  # 计算丰富交叉熵损失
            encoder_out[:, :4, :], text[:, :4]  # 仅使用前4个时间步的编码输出
        )

        loss = loss_ctc + loss_rich  # 计算总损失
        # Collect total loss stats
        stats["loss_ctc"] = torch.clone(loss_ctc.detach()) if loss_ctc is not None else None  # 记录CTC损失
        stats["loss_rich"] = torch.clone(loss_rich.detach()) if loss_rich is not None else None  # 记录丰富损失
        stats["loss"] = torch.clone(loss.detach()) if loss is not None else None  # 记录总损失
        stats["acc_rich"] = acc_rich  # 记录丰富准确率

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:  # 如果启用了长度归一化损失
            batch_size = int((text_lengths + 1).sum())  # 计算批次大小
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)  # 强制收集损失、统计信息和权重
        return loss, stats, weight  # 返回损失、统计信息和权重

    def encode(  # 定义编码函数
        self,
        speech: torch.Tensor,  # 语音输入张量
        speech_lengths: torch.Tensor,  # 语音长度张量
        text: torch.Tensor,  # 文本输入张量
        **kwargs,  # 额外参数
    ):
        """前端 + 编码器。注意此方法由 asr_inference.py 使用
        Args:
                speech: (批次, 长度, ...)
                speech_lengths: (批次, )
                ind: int
        """

        # 数据增强
        if self.specaug is not None and self.training:
            # 如果启用数据增强且当前处于训练模式，则对语音和语音长度进行增强处理
            speech, speech_lengths = self.specaug(speech, speech_lengths)

        # 特征归一化：例如全局 CMVN，话语 CMVN
        if self.normalize is not None:
            # 如果启用归一化，则对语音和语音长度进行归一化处理
            speech, speech_lengths = self.normalize(speech, speech_lengths)

        # 根据条件从语言ID字典中获取语言ID，并生成对应的长整型张量
        lids = torch.LongTensor([[self.lid_int_dict[int(lid)] if torch.rand(1) > 0.2 and int(lid) in self.lid_int_dict else 0 ] for lid in text[:, 0]]).to(speech.device)
        # 将语言ID转换为嵌入向量
        language_query = self.embed(lids)
        
        # 从文本中获取样式ID，并生成对应的长整型张量
        styles = torch.LongTensor([[self.textnorm_int_dict[int(style)]] for style in text[:, 3]]).to(speech.device)
        # 将样式ID转换为嵌入向量
        style_query = self.embed(styles)
        # 将样式查询和语音数据在维度1上进行拼接
        speech = torch.cat((style_query, speech), dim=1)
        # 更新语音长度，加1以考虑样式查询的长度
        speech_lengths += 1

        # 生成事件情感查询的嵌入向量，并重复以匹配批次大小
        event_emo_query = self.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(speech.size(0), 1, 1)
        # 将语言查询和事件情感查询在维度1上进行拼接
        input_query = torch.cat((language_query, event_emo_query), dim=1)
        # 将输入查询和语音数据在维度1上进行拼接
        speech = torch.cat((input_query, speech), dim=1)
        # 更新语音长度，加3以考虑输入查询的长度
        speech_lengths += 3

        # 将处理后的语音和长度传入编码器进行编码
        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)

        # 返回编码器的输出和长度
        return encoder_out, encoder_out_lens

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # 计算 CTC 损失
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # 使用 CTC 计算字符错误率（CER）
        cer_ctc = None
        # 如果不在训练模式且错误计算器已定义
        if not self.training and self.error_calculator is not None:
            # 获取模型预测的标签
            ys_hat = self.ctc.argmax(encoder_out).data
            # 计算 CER
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        # 返回 CTC 损失和 CER
        return loss_ctc, cer_ctc

    def _calc_rich_ce_loss(
        self,
        encoder_out: torch.Tensor,
        ys_pad: torch.Tensor,
    ):
        # 通过 CTC 层计算解码器输出
        decoder_out = self.ctc.ctc_lo(encoder_out)
        # 计算注意力损失
        loss_rich = self.criterion_att(decoder_out, ys_pad.contiguous())
        # 计算准确率
        acc_rich = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_pad.contiguous(),
            ignore_label=self.ignore_id,
        )

        # 返回注意力损失和准确率
        return loss_rich, acc_rich


    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = ["wav_file_tmp_name"],
        tokenizer=None,
        frontend=None,
        **kwargs,
    def export(self, **kwargs):
        from export_meta import export_rebuild_model

        # 检查是否提供了最大序列长度，如果没有则设置为512
        if "max_seq_len" not in kwargs:
            kwargs["max_seq_len"] = 512
        # 调用重建模型的导出函数
        models = export_rebuild_model(model=self, **kwargs)
        # 返回导出的模型
        return models
```