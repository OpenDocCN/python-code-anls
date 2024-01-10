# `so-vits-svc\vencoder\hubert\hubert_model.py`

```
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
        super().__init__()
        # 初始化 mask 属性
        self._mask = mask
        # 创建 FeatureExtractor 实例
        self.feature_extractor = FeatureExtractor()
        # 创建 FeatureProjection 实例
        self.feature_projection = FeatureProjection()
        # 创建 PositionalConvEmbedding 实例
        self.positional_embedding = PositionalConvEmbedding()
        # 创建 LayerNorm 层
        self.norm = nn.LayerNorm(768)
        # 创建 Dropout 层
        self.dropout = nn.Dropout(0.1)
        # 创建 TransformerEncoder 实例
        self.encoder = TransformerEncoder(
            # 创建 nn.TransformerEncoderLayer 实例
            nn.TransformerEncoderLayer(
                768, 12, 3072, activation="gelu", batch_first=True
            ),
            12,
        )
        # 创建 Linear 层
        self.proj = nn.Linear(768, 256)
        # 创建可学习的参数，用于掩码
        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(768).uniform_())
        # 创建嵌入层，用于标签嵌入
        self.label_embedding = nn.Embedding(num_label_embeddings, 256)

    # 定义掩码方法，接受输入张量 x，返回元组 (处理后的张量, 掩码张量)
    def mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = None
        # 如果处于训练状态且 mask 为 True
        if self.training and self._mask:
            # 计算掩码
            mask = _compute_mask((x.size(0), x.size(1)), 0.8, 10, x.device, 2)
            # 将 x 中对应掩码位置的值替换为 masked_spec_embed 的值
            x[mask] = self.masked_spec_embed.to(x.dtype)
        # 返回处理后的张量和掩码张量
        return x, mask

    # 定义编码方法，接受输入张量 x 和可选参数 layer，返回元组 (编码后的张量, 掩码张量)
    def encode(
            self, x: torch.Tensor, layer: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 使用 feature_extractor 处理输入张量
        x = self.feature_extractor(x)
        # 使用 feature_projection 处理转置后的张量
        x = self.feature_projection(x.transpose(1, 2))
        # 对张量进行掩码处理
        x, mask = self.mask(x)
        # 加上位置嵌入
        x = x + self.positional_embedding(x)
        # 对张量进行 LayerNorm 和 Dropout 处理
        x = self.dropout(self.norm(x))
        # 使用 encoder 进行编码
        x = self.encoder(x, output_layer=layer)
        # 返回编码后的张量和掩码张量
        return x, mask

    # 定义逻辑方法，接受输入张量 x，返回逻辑张量
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        # 计算余弦相似度
        logits = torch.cosine_similarity(
            x.unsqueeze(2),
            self.label_embedding.weight.unsqueeze(0).unsqueeze(0),
            dim=-1,
        )
        # 对结果进行缩放
        return logits / 0.1
    # 定义一个前向传播函数，接受一个张量 x 作为输入，并返回两个张量
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 调用 encode 方法对输入进行编码，得到编码后的张量 x 和掩码 mask
        x, mask = self.encode(x)
        # 将编码后的张量 x 通过 proj 层进行投影
        x = self.proj(x)
        # 将投影后的张量 x 通过 logits 层得到预测结果 logits
        logits = self.logits(x)
        # 返回预测结果 logits 和掩码 mask
        return logits, mask
# 定义一个类 HubertSoft，继承自 Hubert 类
class HubertSoft(Hubert):
    # 初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()

    # 定义一个装饰器，将 units 方法设置为推断模式
    @torch.inference_mode()
    # 定义 units 方法，接收一个 torch.Tensor 类型的参数 wav，返回一个 torch.Tensor 类型的结果
    def units(self, wav: torch.Tensor) -> torch.Tensor:
        # 对输入的 wav 进行填充操作
        wav = t_func.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        # 调用 encode 方法对 wav 进行编码
        x, _ = self.encode(wav)
        # 返回编码后的结果经过 proj 方法处理的结果
        return self.proj(x)


# 定义一个类 FeatureExtractor，继承自 nn.Module 类
class FeatureExtractor(nn.Module):
    # 初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()
        # 定义一系列卷积层和归一化层
        self.conv0 = nn.Conv1d(1, 512, 10, 5, bias=False)
        self.norm0 = nn.GroupNorm(512, 512)
        self.conv1 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv2 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv3 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv4 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv5 = nn.Conv1d(512, 512, 2, 2, bias=False)
        self.conv6 = nn.Conv1d(512, 512, 2, 2, bias=False)

    # 前向传播方法
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 对输入的 x 进行一系列的卷积和激活函数处理
        x = t_func.gelu(self.norm0(self.conv0(x)))
        x = t_func.gelu(self.conv1(x))
        x = t_func.gelu(self.conv2(x))
        x = t_func.gelu(self.conv3(x))
        x = t_func.gelu(self.conv4(x))
        x = t_func.gelu(self.conv5(x))
        x = t_func.gelu(self.conv6(x))
        # 返回处理后的结果
        return x


# 定义一个类 FeatureProjection，继承自 nn.Module 类
class FeatureProjection(nn.Module):
    # 初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()
        # 定义归一化层、线性变换层和 dropout 层
        self.norm = nn.LayerNorm(512)
        self.projection = nn.Linear(512, 768)
        self.dropout = nn.Dropout(0.1)

    # 前向传播方法
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 对输入的 x 进行一系列的处理
        x = self.norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        # 返回处理后的结果
        return x


# 定义一个类 PositionalConvEmbedding，继承自 nn.Module 类
class PositionalConvEmbedding(nn.Module):
    # 初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()
        # 定义一个卷积层，设置了一些参数
        self.conv = nn.Conv1d(
            768,
            768,
            kernel_size=128,
            padding=128 // 2,
            groups=16,
        )
        # 对卷积层进行权重归一化处理
        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
    # 定义一个前向传播函数，接受一个张量 x 作为输入，并返回一个张量
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 对输入张量进行维度转置，将第1和第2维度进行交换
        x = self.conv(x.transpose(1, 2))
        # 对转置后的张量进行 gelu 激活函数处理，去掉最后一维的数据
        x = t_func.gelu(x[:, :, :-1])
        # 再次对张量进行维度转置，将第1和第2维度进行交换
        return x.transpose(1, 2)
class TransformerEncoder(nn.Module):
    # 初始化函数，接受编码器层和层数，创建TransformerEncoder对象
    def __init__(
            self, encoder_layer: nn.TransformerEncoderLayer, num_layers: int
    ) -> None:
        super(TransformerEncoder, self).__init__()
        # 使用深拷贝创建指定数量的编码器层，并存储在ModuleList中
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    # 前向传播函数，接受输入张量src和可选的mask、src_key_padding_mask、output_layer参数，返回输出张量
    def forward(
            self,
            src: torch.Tensor,
            mask: torch.Tensor = None,
            src_key_padding_mask: torch.Tensor = None,
            output_layer: Optional[int] = None,
    ) -> torch.Tensor:
        # 将输入张量赋值给输出张量
        output = src
        # 遍历编码器层列表，对输入张量进行编码
        for layer in self.layers[:output_layer]:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )
        # 返回编码后的输出张量
        return output


def _compute_mask(
        shape: Tuple[int, int],
        mask_prob: float,
        mask_length: int,
        device: torch.device,
        min_masks: int = 0,
) -> torch.Tensor:
    # 获取输入张量的形状信息
    batch_size, sequence_length = shape

    # 如果mask_length小于1，则抛出数值错误
    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    # 如果mask_length大于sequence_length，则抛出数值错误
    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`"
        )

    # 计算在批次中的被遮罩的跨度数量
    num_masked_spans = int(mask_prob * sequence_length / mask_length + random.random())
    num_masked_spans = max(num_masked_spans, min_masks)

    # 确保被遮罩的跨度数量不超过序列长度
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length

    # 创建一个全零的布尔张量，用于存储遮罩
    mask = torch.zeros((batch_size, sequence_length), device=device, dtype=torch.bool)

    # 均匀分布采样，确保偏移样本小于序列长度
    # 创建一个形状为(batch_size, sequence_length - (mask_length - 1))的张量，其中所有元素都为1，表示均匀分布
    uniform_dist = torch.ones(
        (batch_size, sequence_length - (mask_length - 1)), device=device
    )
    
    # 从均匀分布中随机抽样，得到要进行遮盖的索引
    mask_indices = torch.multinomial(uniform_dist, num_masked_spans)
    
    # 将遮盖的索引扩展成遮盖的区间
    mask_indices = (
        mask_indices.unsqueeze(dim=-1)
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    
    # 创建一个偏移量张量，用于将遮盖的索引扩展成遮盖的区间
    offsets = (
        torch.arange(mask_length, device=device)[None, None, :]
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    
    # 将遮盖的索引和偏移量相加，得到最终的遮盖索引
    mask_idxs = mask_indices + offsets
    
    # 将遮盖索引应用到mask张量上，将对应位置的值设置为True
    mask = mask.scatter(1, mask_idxs, True)
    
    # 返回最终的mask张量
    return mask
# 定义一个函数，用于加载预训练的 HuBERT-Soft 模型
def hubert_soft(
        path: str,
) -> HubertSoft:
    r"""HuBERT-Soft from `"A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"`.
    # 参数说明：path (str) - 预训练模型的路径
    hubert = HubertSoft()  # 创建一个 HuBERT-Soft 模型实例
    checkpoint = torch.load(path)  # 从指定路径加载模型的检查点
    consume_prefix_in_state_dict_if_present(checkpoint, "module.")  # 如果存在指定前缀，则在状态字典中消耗前缀
    hubert.load_state_dict(checkpoint)  # 加载模型的状态字典
    hubert.eval()  # 设置模型为评估模式
    return hubert  # 返回加载并设置好的 HuBERT-Soft 模型实例
```