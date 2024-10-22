# `.\chatglm4v-9b\visual.py`

```py
# 导入必要的库
import torch  # 导入 PyTorch 库
from torch import nn  # 从 PyTorch 导入神经网络模块
from argparse import Namespace  # 导入命名空间用于解析命令行参数
import torch.nn.functional as F  # 导入 PyTorch 的函数性模块
from transformers.activations import ACT2FN  # 从 Transformers 导入激活函数
import math  # 导入数学库
from torch.nn import LayerNorm  # 从 PyTorch 导入层归一化模块


# 定义标准注意力机制函数
def standard_attention(query_layer, key_layer, value_layer, scaling_attention_score=True):
    # 如果需要缩放注意力得分，执行缩放
    if scaling_attention_score:
        query_layer = query_layer / math.sqrt(query_layer.shape[-1])  # 对查询层进行缩放
    # 计算注意力得分，通过矩阵乘法得到
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # 计算查询和键的点积

    # 应用 softmax 函数计算注意力概率
    attention_probs = F.softmax(attention_scores, dim=-1)  # 对注意力得分进行归一化处理

    # 通过注意力概率与值层计算上下文层
    context_layer = torch.matmul(attention_probs, value_layer)  # 计算加权后的值
    return context_layer  # 返回上下文层


# 定义默认注意力函数
def attention_fn_default(query_layer, key_layer, value_layer, scaling_attention_score=True):
    # 检查 PyTorch 版本并进行处理
    if int(torch.__version__.split('.')[0]) >= 2 and scaling_attention_score:
        # Pytorch 2.0 的注意力机制在注意力掩码为 float 时消耗大量内存，并在注意力掩码为 None 时有 NaN bug。
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer, key_layer, value_layer,  # 执行缩放点积注意力
            attn_mask=None,  # 不使用注意力掩码
            dropout_p=0.,  # 不使用 dropout
            is_causal=False  # 非因果注意力
        )
        return attn_output  # 返回注意力输出
    else:
        # 使用标准注意力函数
        return standard_attention(
            query_layer, key_layer, value_layer, scaling_attention_score=scaling_attention_score  # 调用标准注意力
        )


# 定义 PatchEmbedding 类
class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()  # 初始化父类
        # 创建卷积层，用于图像切片嵌入
        self.proj = nn.Conv2d(config.in_channels, config.hidden_size, kernel_size=config.patch_size,
                              stride=config.patch_size)  # 定义卷积层
        # 定义 CLS token 嵌入
        self.cls_embedding = nn.Parameter(torch.zeros(1, config.hidden_size))  # 初始化 CLS token
        # 定义位置嵌入
        self.position_embedding = nn.Embedding(config.num_positions, config.hidden_size)  # 初始化位置嵌入

    # 定义前向传播方法
    def forward(self, images: "tensor(B, C, H, W)") -> "tensor(B, L, D)":
        # 对输入图像进行卷积
        x = self.proj(images)  # 应用卷积层
        x = x.flatten(2).transpose(1, 2)  # 将输出展平并转置维度
        cls_token = self.cls_embedding.expand(x.shape[0], -1, -1)  # 扩展 CLS token
        x = torch.cat((cls_token, x), dim=1)  # 将 CLS token 与其他嵌入连接
        x += self.position_embedding.weight.unsqueeze(0)  # 添加位置嵌入
        return x  # 返回最终的嵌入


# 定义 Attention 类
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()  # 初始化父类
        self.num_heads = config.num_heads  # 设置头数
        head_dim = config.hidden_size // config.num_heads  # 计算每个头的维度
        self.scale = head_dim ** -0.5  # 计算缩放因子
        self.query_key_value = nn.Linear(config.hidden_size, config.hidden_size * 3)  # 定义 QKV 线性变换
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 定义输出线性层
        self.output_dropout = torch.nn.Dropout(config.dropout_prob)  # 定义 dropout 层

    # 定义前向传播方法
    def forward(self, x: "tensor(B, L, D)") -> "tensor(B, L, D)":
        B, L, _ = x.shape  # 获取批量大小和序列长度
        qkv = self.query_key_value(x)  # 计算 QKV
        qkv = qkv.reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)  # 重新形状并排列维度
        q, k, v = qkv[0], qkv[1], qkv[2]  # 拆分 QKV

        out = attention_fn_default(  # 调用注意力函数
            q, k, v  # 传入 Q、K 和 V
        )
        output = self.dense(out.transpose(1, 2).reshape(B, L, -1))  # 应用输出线性层
        output = self.output_dropout(output)  # 应用 dropout
        return output  # 返回最终输出
    # 定义注意力机制函数，接受查询（q）、键（k）和值（v）作为输入
        def attention(self, q, k, v):
            # 计算注意力权重，将查询与键的转置相乘，并进行缩放
            attn_weights = torch.matmul(q * self.scale, k.transpose(-2, -1))
            # 对注意力权重进行归一化处理，使用 softmax 函数
            attn_weights = attn_weights.softmax(dim=-1)
            # 将归一化后的注意力权重与值相乘以获得输出
            output = torch.matmul(attn_weights, v)
            # 返回最终的输出
            return output
# 定义一个多层感知机（MLP）类，继承自 nn.Module
class MLP(nn.Module):
    # 初始化方法，接受配置参数
    def __init__(self, config):
        # 调用父类构造方法
        super().__init__()
        # 保存配置参数
        self.config = config
        # 根据配置选择激活函数
        self.activation_fn = ACT2FN[config.hidden_act]
        # 定义第一层全连接层，输入维度为 hidden_size，输出维度为 intermediate_size
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # 定义第二层全连接层，输入维度为 intermediate_size，输出维度为 hidden_size
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    # 前向传播方法，接受输入张量 x
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通过第一层全连接层
        x = self.fc1(x)
        # 应用激活函数
        x = self.activation_fn(x)
        # 通过第二层全连接层
        x = self.fc2(x)
        # 返回输出张量
        return x


# 定义一个变压器层（TransformerLayer）类，继承自 nn.Module
class TransformerLayer(nn.Module):
    # 初始化方法，接受配置参数
    def __init__(self, config):
        # 调用父类构造方法
        super().__init__()
        # 定义输入层归一化层，维度为 hidden_size
        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 定义注意力机制层
        self.attention = Attention(config)
        # 定义多层感知机
        self.mlp = MLP(config)
        # 定义后注意力层归一化层
        self.post_attention_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播方法，接受隐藏状态
    def forward(self, hidden_states):
        # 将隐藏状态赋值给注意力输入
        attention_input = hidden_states
        # 经过注意力层和输入归一化层
        attention_output = self.input_layernorm(self.attention(attention_input))
        # 将注意力输出加回原输入（残差连接）
        hidden_states = attention_input + attention_output
        # 将当前状态赋值给 MLP 输入
        mlp_input = hidden_states

        # https://github.com/THUDM/GLM-4/issues/350
        # 经过后注意力层归一化和 MLP，保持设备一致
        mlp_output = self.post_attention_layernorm(self.mlp(mlp_input)).to(mlp_input.device)
        # 将 MLP 输出加回输入（残差连接）
        output = mlp_input + mlp_output
        # 返回最终输出
        return output


# 定义变压器模型（Transformer）类，继承自 nn.Module
class Transformer(nn.Module):
    # 初始化方法，接受配置参数
    def __init__(self, config):
        # 调用父类构造方法
        super().__init__()
        # 创建包含多个变压器层的模块列表，根据配置中的层数
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_hidden_layers)])

    # 前向传播方法，接受隐藏状态
    def forward(self, hidden_states):
        # 遍历每一层变压器，更新隐藏状态
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        # 返回最终的隐藏状态
        return hidden_states


# 定义 GLU 模型类，继承自 nn.Module
class GLU(nn.Module):
    # 初始化方法，接受配置和输入特征数量
    def __init__(self, config, in_features):
        # 调用父类构造方法
        super().__init__()
        # 定义线性投影层，无偏置
        self.linear_proj = nn.Linear(in_features, config.hidden_size, bias=False)
        # 定义第一层归一化层
        self.norm1 = nn.LayerNorm(config.hidden_size)
        # 定义第一激活函数为 GELU
        self.act1 = nn.GELU()
        # 定义第二激活函数为 SiLU
        self.act2 = nn.functional.silu
        # 定义从 hidden_size 到 ffn_hidden_size 的线性层，无偏置
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        # 定义门控线性层，从 hidden_size 到 ffn_hidden_size，无偏置
        self.gate_proj = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        # 定义从 ffn_hidden_size 回到 hidden_size 的线性层，无偏置
        self.dense_4h_to_h = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=False)

    # 前向传播方法，接受输入 x
    def forward(self, x):
        # 通过线性投影层
        x = self.linear_proj(x)
        # 归一化后应用第一激活函数
        x = self.act1(self.norm1(x))
        # 计算门控乘积并通过 dense 层
        x = self.act2(self.gate_proj(x)) * self.dense_h_to_4h(x)
        # 通过最后的线性层
        x = self.dense_4h_to_h(x)
        # 返回输出 x
        return x


# 定义 EVA2CLIP 模型类，继承自 nn.Module
class EVA2CLIPModel(nn.Module):
    # 初始化方法，接收配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将视觉配置转换为命名空间对象，方便访问其属性
        vision_config = Namespace(**config.vision_config)
        # 创建补丁嵌入层，使用视觉配置
        self.patch_embedding = PatchEmbedding(vision_config)
        # 创建变换器，使用视觉配置
        self.transformer = Transformer(vision_config)
        # 创建线性投影层，输入特征为隐藏层大小
        self.linear_proj = GLU(config, in_features=config.hidden_size)
        # 创建卷积层，输入通道为隐藏层大小，输出通道为配置的隐藏层大小
        self.conv = nn.Conv2d(in_channels=vision_config.hidden_size, out_channels=config.hidden_size, kernel_size=2,
                              stride=2)
        # 创建一个可学习的参数，初始化为零，表示开始位置
        self.boi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # 创建一个可学习的参数，初始化为零，表示结束位置
        self.eoi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # 获取缩放因子，来自视觉配置
        self.scaling_factor = vision_config.scaling_factor
    
    # 前向传播方法，处理输入图像并返回结果
    def forward(self, images: "tensor(B, C, H, W)") -> "tensor(B, L, D)":
        # 将输入图像经过补丁嵌入层处理
        x = self.patch_embedding(images)
        # 将嵌入结果传入变换器
        x = self.transformer(x)
        # 去掉第一个标记，通常是用于序列处理
        x = x[:, 1:]
    
        # 获取当前张量的批量、序列长度和隐藏层维度
        b, s, h = x.shape
        # 计算网格大小，通常用于将序列重塑为二维形式
        grid_size = int(s ** 0.5)
        # 重塑张量形状，调整为适合卷积操作的格式
        x = x.view(b, grid_size, grid_size, h).permute(0, 3, 1, 2)
        # 将重塑后的张量经过卷积层处理
        x = self.conv(x)
    
        # 展平张量的特征维度，并转置以适配后续操作
        x = x.flatten(2).transpose(1, 2)
        # 将张量经过线性投影层处理
        x = self.linear_proj(x)
    
        # 扩展开始位置参数到当前批量大小，保持维度一致
        boi = self.boi.expand(x.shape[0], -1, -1).to(x.device)
        # 扩展结束位置参数到当前批量大小，保持维度一致
        eoi = self.eoi.expand(x.shape[0], -1, -1).to(x.device)
        # 将开始标记、处理后的张量和结束标记在序列维度拼接
        x = torch.cat((boi, x, eoi), dim=1)
        # 对结果进行缩放
        x = x / self.scaling_factor
        # 返回最终处理后的张量
        return x
```