# `so-vits-svc\vencoder\hubert\hubert_model_onnx.py`

```py
# 导入必要的库
import copy
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as t_func
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

# 定义 Hubert 类，继承自 nn.Module
class Hubert(nn.Module):
    # 初始化方法，接受 num_label_embeddings 和 mask 两个参数
    def __init__(self, num_label_embeddings: int = 100, mask: bool = True):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化 mask 属性
        self._mask = mask
        # 创建 FeatureExtractor 实例
        self.feature_extractor = FeatureExtractor()
        # 创建 FeatureProjection 实例
        self.feature_projection = FeatureProjection()
        # 创建 PositionalConvEmbedding 实例
        self.positional_embedding = PositionalConvEmbedding()
        # 创建 LayerNorm 实例
        self.norm = nn.LayerNorm(768)
        # 创建 Dropout 实例
        self.dropout = nn.Dropout(0.1)
        # 创建 TransformerEncoder 实例
        self.encoder = TransformerEncoder(
            nn.TransformerEncoderLayer(
                768, 12, 3072, activation="gelu", batch_first=True
            ),
            12,
        )
        # 创建 Linear 实例
        self.proj = nn.Linear(768, 256)

        # 创建可学习的参数 masked_spec_embed
        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(768).uniform_())
        # 创建嵌入层，用于标签的嵌入
        self.label_embedding = nn.Embedding(num_label_embeddings, 256)

    # 定义 mask 方法，接受输入张量 x，返回处理后的张量和 mask
    def mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = None
        # 如果处于训练状态且 mask 为 True
        if self.training and self._mask:
            # 计算 mask
            mask = _compute_mask((x.size(0), x.size(1)), 0.8, 10, x.device, 2)
            # 将 x 中对应 mask 为 True 的位置替换为 masked_spec_embed
            x[mask] = self.masked_spec_embed.to(x.dtype)
        # 返回处理后的张量和 mask
        return x, mask

    # 定义 encode 方法，接受输入张量 x 和可选参数 layer，返回处理后的张量和 mask
    def encode(
            self, x: torch.Tensor, layer: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 提取特征
        x = self.feature_extractor(x)
        # 特征投影
        x = self.feature_projection(x.transpose(1, 2))
        # 进行 mask 处理
        x, mask = self.mask(x)
        # 加上位置编码
        x = x + self.positional_embedding(x)
        # 进行归一化和 dropout
        x = self.dropout(self.norm(x))
        # 编码器编码
        x = self.encoder(x, output_layer=layer)
        # 返回处理后的张量和 mask
        return x, mask

    # 定义 logits 方法，接受输入张量 x，返回 logits
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        # 计算 logits
        logits = torch.cosine_similarity(
            x.unsqueeze(2),
            self.label_embedding.weight.unsqueeze(0).unsqueeze(0),
            dim=-1,
        )
        # 对 logits 进行缩放
        return logits / 0.1


# 定义 HubertSoft 类，继承自 Hubert 类
class HubertSoft(Hubert):
    # 初始化函数，调用父类的初始化方法
    def __init__(self):
        super().__init__()
    
    # 对输入的音频数据进行处理，填充数据，然后进行编码和投影操作
    def units(self, wav: torch.Tensor) -> torch.Tensor:
        # 对音频数据进行填充，使其长度为400
        wav = t_func.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        # 对填充后的音频数据进行编码
        x, _ = self.encode(wav)
        # 对编码后的数据进行投影
        return self.proj(x)
    
    # 前向传播函数，调用处理音频数据的方法
    def forward(self, x):
        return self.units(x)
# 定义特征提取器类
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义卷积层，输入通道为1，输出通道为512，卷积核大小为10，步长为5，无偏置
        self.conv0 = nn.Conv1d(1, 512, 10, 5, bias=False)
        # 定义组归一化层，组数为512，输入通道为512
        self.norm0 = nn.GroupNorm(512, 512)
        # 定义多个卷积层，输入通道和输出通道均为512，不同的卷积核大小和步长
        self.conv1 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv2 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv3 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv4 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv5 = nn.Conv1d(512, 512, 2, 2, bias=False)
        self.conv6 = nn.Conv1d(512, 512, 2, 2, bias=False)

    # 前向传播函数
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 对输入数据进行卷积、组归一化和GELU激活函数处理
        x = t_func.gelu(self.norm0(self.conv0(x)))
        x = t_func.gelu(self.conv1(x))
        x = t_func.gelu(self.conv2(x))
        x = t_func.gelu(self.conv3(x))
        x = t_func.gelu(self.conv4(x))
        x = t_func.gelu(self.conv5(x))
        x = t_func.gelu(self.conv6(x))
        return x


# 定义特征投影类
class FeatureProjection(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义层归一化层，输入通道为512
        self.norm = nn.LayerNorm(512)
        # 定义线性投影层，输入维度为512，输出维度为768
        self.projection = nn.Linear(512, 768)
        # 定义丢弃层，丢弃概率为0.1
        self.dropout = nn.Dropout(0.1)

    # 前向传播函数
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 对输入数据进行层归一化、线性投影和丢弃处理
        x = self.norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


# 定义位置卷积嵌入类
class PositionalConvEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一维卷积层，输入通道和输出通道均为768，卷积核大小为128，填充为卷积核大小的一半，分组数为16
        self.conv = nn.Conv1d(
            768,
            768,
            kernel_size=128,
            padding=128 // 2,
            groups=16,
        )
        # 对卷积层进行权重归一化处理
        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)

    # 前向传播函数
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 对输入数据进行卷积、GELU激活函数处理和维度转换
        x = self.conv(x.transpose(1, 2))
        x = t_func.gelu(x[:, :, :-1])
        return x.transpose(1, 2)


# 定义Transformer编码器类
class TransformerEncoder(nn.Module):
    def __init__(
            self, encoder_layer: nn.TransformerEncoderLayer, num_layers: int
    # 定义 TransformerEncoder 类，继承自 nn.Module 类
    def __init__(self, encoder_layer: nn.Module, num_layers: int) -> None:
        # 调用父类的初始化方法
        super(TransformerEncoder, self).__init__()
        # 使用深拷贝创建包含多个编码器层的列表
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        # 记录编码器层的数量
        self.num_layers = num_layers

    # 定义前向传播方法
    def forward(
            self,
            src: torch.Tensor,
            mask: torch.Tensor = None,
            src_key_padding_mask: torch.Tensor = None,
            output_layer: Optional[int] = None,
    ) -> torch.Tensor:
        # 将输入的源张量赋值给输出
        output = src
        # 遍历编码器层列表，执行多个编码器层的前向传播
        for layer in self.layers[:output_layer]:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )
        # 返回最终输出
        return output
# 计算掩码
def _compute_mask(
        shape: Tuple[int, int],  # 输入形状，包括批大小和序列长度
        mask_prob: float,  # 掩码概率
        mask_length: int,  # 掩码长度
        device: torch.device,  # 设备
        min_masks: int = 0,  # 最小掩码数
) -> torch.Tensor:  # 返回掩码张量
    batch_size, sequence_length = shape  # 获取批大小和序列长度

    if mask_length < 1:  # 如果掩码长度小于1，抛出数值错误
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:  # 如果掩码长度大于序列长度，抛出数值错误
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`"
        )

    # 计算批中掩码跨度的数量
    num_masked_spans = int(mask_prob * sequence_length / mask_length + random.random())
    num_masked_spans = max(num_masked_spans, min_masks)

    # 确保掩码索引数量小于等于序列长度
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length

    # 创建用于填充的SpecAugment掩码
    mask = torch.zeros((batch_size, sequence_length), device=device, dtype=torch.bool)

    # 从均匀分布中进行采样，确保偏移样本小于序列长度
    uniform_dist = torch.ones(
        (batch_size, sequence_length - (mask_length - 1)), device=device
    )

    # 获取随机掩码索引
    mask_indices = torch.multinomial(uniform_dist, num_masked_spans)

    # 将掩码索引扩展为掩码跨度
    mask_indices = (
        mask_indices.unsqueeze(dim=-1)
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    offsets = (
        torch.arange(mask_length, device=device)[None, None, :]
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    mask_idxs = mask_indices + offsets

    # 将索引散布到掩码中
    mask = mask.scatter(1, mask_idxs, True)

    return mask


def hubert_soft(
        path: str,  # 路径参数
) -> HubertSoft:  # 返回HubertSoft对象
    # 从 "A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion" 中导入 HuBERT-Soft 模型
    # 参数：
    #   path (str): 预训练模型的路径
    hubert = HubertSoft()
    # 加载预训练模型的参数
    checkpoint = torch.load(path)
    # 如果参数中包含以 "module." 开头的键，去除该前缀
    consume_prefix_in_state_dict_if_present(checkpoint, "module.")
    # 将模型的参数加载到 HuBERT-Soft 模型中
    hubert.load_state_dict(checkpoint)
    # 设置模型为评估模式
    hubert.eval()
    # 返回加载好的 HuBERT-Soft 模型
    return hubert
```