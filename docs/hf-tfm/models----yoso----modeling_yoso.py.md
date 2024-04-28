# `.\transformers\models\yoso\modeling_yoso.py`

```py
# 导入所需的库和模块
import math
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入激活函数和模型输出
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_yoso import YosoConfig

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义一些常量，如 PyTorch YOSO 预训练模型的路径和配置
_CHECKPOINT_FOR_DOC = "uw-madison/yoso-4096"
_CONFIG_FOR_DOC = "YosoConfig"
YOSO_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "uw-madison/yoso-4096",
    # See all YOSO models at https://huggingface.co/models?filter=yoso
]

# 尝试加载 CUDA 内核
def load_cuda_kernels():
    global lsh_cumulation
    try:
        from torch.utils.cpp_extension import load

        # 定义一个函数，用于拼接文件路径
        def append_root(files):
            src_folder = Path(__file__).resolve().parent.parent.parent / "kernels" / "yoso"
            return [src_folder / file for file in files]

        # 加载 C++ 和 CUDA 扩展
        src_files = append_root(
            ["fast_lsh_cumulation_torch.cpp", "fast_lsh_cumulation.cu", "fast_lsh_cumulation_cuda.cu"]
        )
        load("fast_lsh_cumulation", src_files, verbose=True)
        import fast_lsh_cumulation as lsh_cumulation
        return True
    except Exception:
        lsh_cumulation = None
        return False

# 确保输入张量是连续的
def to_contiguous(input_tensors):
    if isinstance(input_tensors, list):
        out = []
        for tensor in input_tensors:
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            out.append(tensor)
        return out
    else:
        if not input_tensors.is_contiguous():
            input_tensors = input_tensors.contiguous()
        return input_tensors

# 对输入张量进行归一化处理
def normalize(input_tensors):
    pass
    # 如果输入的张量是列表
    if isinstance(input_tensors, list):
        # 创建一个空列表
        out = []
        # 遍历输入的张量列表
        for tensor in input_tensors:
            # 对每个张量进行 L2 范数归一化
            out.append(nn.functional.normalize(tensor, p=2, dim=-1))
        # 返回归一化后的张量列表
        return out
    # 如果输入的是单个张量
    else:
        # 对输入的张量进行 L2 范数归一化
        return nn.functional.normalize(input_tensors, p=2, dim=-1)
# 计算哈希编码
def hashing(query, key, num_hash, hash_len):
    # 检查查询数据是否为3维
    if len(query.size()) != 3:
        raise ValueError("Query has incorrect size.")
    # 检查关键数据是否为3维
    if len(key.size()) != 3:
        raise ValueError("Key has incorrect size.")

    # 生成随机投影矩阵
    rmat = torch.randn(query.size(0), query.size(2), num_hash * hash_len, device=query.device)
    # 计算2的幂次方，用于哈希函数
    raise_pow = 2 ** torch.arange(hash_len, device=query.device)

    # 对查询数据进行投影并重新构形
    query_projection = torch.matmul(query, rmat).reshape(query.size(0), query.size(1), num_hash, hash_len)
    # 对关键数据进行投影并重新构形
    key_projection = torch.matmul(key, rmat).reshape(key.size(0), key.size(1), num_hash, hash_len)
    # 将投影后的数据转换为二进制
    query_binary = (query_projection > 0).int()
    key_binary = (key_projection > 0).int()
    # 计算哈希值
    query_hash = torch.sum(query_binary * raise_pow, dim=-1)
    query_hash = torch.sum(key_binary * raise_pow, dim=-1)

    return query_hash.int(), query_hash.int()


# 自定义的自动微分函数类
class YosoCumulation(torch.autograd.Function):
    @staticmethod
    # 前向传播函数
    def forward(ctx, query_mask, key_mask, query, key, value, config):
        # 获取哈希编码的位数
        hash_code_len = config["hash_code_len"]
        # 计算期望值
        expectation = (1 - torch.acos(torch.matmul(query, key.transpose(-1, -2))) / math.pi) ** hash_code_len
        # 期望值乘以掩码，并进行累积求和
        expectation = expectation * query_mask[:, :, None] * key_mask[:, None, :]
        cumulation_value = torch.matmul(expectation, value)

        # 保存缓存数据，包括掩码、期望值和输入数据
        ctx.save_for_backward(query_mask, key_mask, expectation, query, key, value)
        ctx.config = config

        return cumulation_value

    @staticmethod
    # 反向传播函数
    def backward(ctx, grad):
        grad = to_contiguous(grad)

        query_mask, key_mask, expectation, query, key, value = ctx.saved_tensors
        config = ctx.config

        hash_code_len = config["hash_code_len"]

        # 计算加权期望值，并更新梯度
        weighted_exp = torch.matmul(grad, value.transpose(-1, -2)) * expectation
        grad_query = torch.matmul(weighted_exp, (hash_code_len / 2) * key)
        grad_key = torch.matmul(weighted_exp.transpose(-1, -2), (hash_code_len / 2) * query)
        grad_value = torch.matmul(expectation.transpose(-1, -2), grad)

        return None, None, grad_query, grad_key, grad_value, None

# YosoLSHCumulation 类
class YosoLSHCumulation(torch.autograd.Function):
    @staticmethod
    # 略
    # 此方法用于执行前向传播操作，实现注意力机制的计算
    def forward(ctx, query_mask, key_mask, query, key, value, config):
        # 检查输入张量的维度是否一致，如果不一致则引发异常
        if query_mask.size(0) != key_mask.size(0):
            raise ValueError("Query mask and Key mask differ in sizes in dimension 0")
        if query_mask.size(0) != query.size(0):
            raise ValueError("Query mask and Query differ in sizes in dimension 0")
        if query_mask.size(0) != key.size(0):
            raise ValueError("Query mask and Key differ in sizes in dimension 0")
        if query_mask.size(0) != value.size(0):
            raise ValueError("Query mask and Value mask differ in sizes in dimension 0")
        if key.size(1) != value.size(1):
            raise ValueError("Key and Value differ in sizes in dimension 1")
        if query.size(2) != key.size(2):
            raise ValueError("Query and Key differ in sizes in dimension 2")

        # 将输入张量转换为连续内存布局，以确保后续计算的正确性
        query_mask, key_mask, query, key, value = to_contiguous([query_mask, key_mask, query, key, value])

        # 检查是否在 CUDA 上执行，以确定是否使用 GPU 加速
        use_cuda = query_mask.is_cuda
        # 设置哈希表的参数
        num_hash = config["num_hash"]
        hash_code_len = config["hash_code_len"]
        hashtable_capacity = int(2**hash_code_len)

        # 根据配置决定是否使用快速哈希函数
        if config["use_fast_hash"]:
            # 使用快速哈希函数计算查询和键的哈希码
            query_hash_code, key_hash_code = lsh_cumulation.fast_hash(
                query_mask, query, key_mask, key, num_hash, hash_code_len, use_cuda, 1
            )
        else:
            # 使用标准哈希函数计算查询和键的哈希码
            query_hash_code, key_hash_code = hashing(query, key, num_hash, hash_code_len)

        # 执行局部散列积累操作
        cumulation_value = lsh_cumulation.lsh_cumulation(
            query_mask, query_hash_code, key_mask, key_hash_code, value, hashtable_capacity, use_cuda, 1
        )

        # 保存计算中需要的张量以备反向传播使用
        ctx.save_for_backward(query_mask, key_mask, query_hash_code, key_hash_code, query, key, value)
        # 保存配置信息
        ctx.config = config

        # 返回局部散列积累结果
        return cumulation_value

    # 静态方法定义结束
    @staticmethod
    # 定义反向传播函数，接收上下文和梯度作为参数
    def backward(ctx, grad):
        # 将梯度转换为连续的形式
        grad = to_contiguous(grad)

        # 从上下文中提取保存的张量和配置信息
        query_mask, key_mask, query_hash_code, key_hash_code, query, key, value = ctx.saved_tensors
        config = ctx.config

        # 检查梯度是否在 CUDA 上
        use_cuda = grad.is_cuda
        # 获取哈希码长度和哈希表容量
        hash_code_len = config["hash_code_len"]
        hashtable_capacity = int(2**hash_code_len)

        # 如果配置要求使用 LSH 反向传播
        if config["lsh_backward"]:
            # 使用 LSH 累积函数计算梯度值
            grad_value = lsh_cumulation.lsh_cumulation(
                key_mask, key_hash_code, query_mask, query_hash_code, grad, hashtable_capacity, use_cuda, 1
            )
            # 使用加权 LSH 累积函数计算查询梯度
            grad_query = lsh_cumulation.lsh_weighted_cumulation(
                query_mask,
                query_hash_code,
                grad,
                key_mask,
                key_hash_code,
                value,
                (hash_code_len / 2) * key,
                hashtable_capacity,
                use_cuda,
                4,
            )
            # 使用加权 LSH 累积函数计算键梯度
            grad_key = lsh_cumulation.lsh_weighted_cumulation(
                key_mask,
                key_hash_code,
                value,
                query_mask,
                query_hash_code,
                grad,
                (hash_code_len / 2) * query,
                hashtable_capacity,
                use_cuda,
                4,
            )
        else:
            # 计算期望值
            expectation = (1 - torch.acos(torch.matmul(query, key.transpose(-1, -2))) / math.pi) ** hash_code_len
            # 乘以查询和键的掩码以及期望值
            expectation = expectation * query_mask[:, :, None] * key_mask[:, None, :]
            # 计算加权期望值
            weighted_exp = torch.matmul(grad, value.transpose(-1, -2)) * expectation
            # 计算查询梯度
            grad_query = torch.matmul(weighted_exp, (hash_code_len / 2) * key)
            # 计算键梯度
            grad_key = torch.matmul(weighted_exp.transpose(-1, -2), (hash_code_len / 2) * query)
            # 计算值梯度
            grad_value = torch.matmul(expectation.transpose(-1, -2), grad)

        # 返回梯度，其中置 None 表示没有梯度
        return None, None, grad_query, grad_key, grad_value, None
# 从transformers.models.nystromformer.modeling_nystromformer.NystromformerEmbeddings中复制代码
class YosoEmbeddings(nn.Module):
    """从单词、位置和标记类型嵌入中构建嵌入。"""

    def __init__(self, config):
        super().__init__()
        # 创建一个单词嵌入层，将单词ID映射为隐藏尺寸的向量，并指定填充ID
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建一个位置嵌入层，将位置ID映射为隐藏尺寸的向量
        self.position_embeddings = nn.Embedding(config.max_position_embeddings + 2, config.hidden_size)
        # 创建一个标记类型嵌入层，将标记类型ID映射为隐藏尺寸的向量
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm不使用蛇形命名，以保持与TensorFlow模型变量名称一致，并能够加载任何TensorFlow检查点文件
        # 创建一个LayerNorm层，用于归一化嵌入向量
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个丢弃层，用于在训练过程中随机丢弃部分嵌入向量，以降低过拟合风险
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids是一个内存连续的张量，当序列化时会导出(1, len position emb)
        # 创建position_ids张量，将长度为config.max_position_embeddings的一维向量扩展为(1, config.max_position_embeddings)，并加2
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)) + 2, persistent=False
        )
        # 获取配置中的position_embedding_type属性，默认为"absolute"
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 创建一个token_type_ids张量，与position_ids大小相同，dtype为torch.long，设备与position_ids相同，并初始化为全0
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
            persistent=False,
        )
    # 定义模型的前向传播函数，接受输入的token IDs，token类型 IDs，位置 IDs和嵌入矩阵作为参数
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        # 如果传入了token IDs，则获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        # 否则，获取嵌入矩阵的形状
        else:
            input_shape = inputs_embeds.size()[:-1]

        # 获取输入序列的长度
        seq_length = input_shape[1]

        # 如果位置 IDs 为 None，则从预设的位置 IDs 中选择与序列长度相匹配的部分
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果 token 类型 IDs 为 None
        if token_type_ids is None:
            # 检查是否实例中有 token 类型 IDs 属性
            if hasattr(self, "token_type_ids"):
                # 从预注册缓冲区中获取 token 类型 IDs，并根据序列长度进行截取
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                # 如果不存在 token 类型 IDs 属性，则创建全零的 token 类型 IDs 张量
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果输入嵌入矩阵为 None，则使用模型的词嵌入层获取嵌入矩阵
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 使用 token 类型嵌入层获取 token 类型嵌入矩阵
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将词嵌入矩阵与 token 类型嵌入矩阵相加
        embeddings = inputs_embeds + token_type_embeddings
        # 如果位置嵌入类型为"absolute"，则添加位置嵌入矩阵到 embeddings 中
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        # 对 embeddings 进行 LayerNorm 归一化
        embeddings = self.LayerNorm(embeddings)
        # 对 embeddings 进行 dropout 处理
        embeddings = self.dropout(embeddings)
        # 返回处理后的 embeddings
        return embeddings
# 定义 YosoSelfAttention 类，继承自 nn.Module
class YosoSelfAttention(nn.Module):
    # 初始化函数，接受配置对象和位置嵌入类型作为参数
    def __init__(self, config, position_embedding_type=None):
        # 调用父类的初始化函数
        super().__init__()
        # 如果隐藏大小不能被注意力头的数量整除，且配置对象没有嵌入大小属性，则引发 ValueError
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 设置注意力头的数量
        self.num_attention_heads = config.num_attention_heads
        # 计算每个注意力头的大小
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 计算所有注意力头的总大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 设置位置嵌入类型
        self.position_embedding_type = (
            position_embedding_type if position_embedding_type is not None else config.position_embedding_type
        )

        # 设置是否使用期望值
        self.use_expectation = config.use_expectation
        # 设置哈希码长度
        self.hash_code_len = config.hash_code_len
        # 设置是否使用卷积
        self.use_conv = config.conv_window is not None
        # 设置是否使用快速哈希
        self.use_fast_hash = config.use_fast_hash
        # 设置哈希函数的数量
        self.num_hash = config.num_hash
        # 设置是否后向 LSH
        self.lsh_backward = config.lsh_backward

        # 设置 LSH 配置
        self.lsh_config = {
            "hash_code_len": self.hash_code_len,
            "use_fast_hash": self.use_fast_hash,
            "num_hash": self.num_hash,
            "lsh_backward": self.lsh_backward,
        }

        # 如果使用卷积，则初始化卷积层
        if config.conv_window is not None:
            self.conv = nn.Conv2d(
                in_channels=config.num_attention_heads,
                out_channels=config.num_attention_heads,
                kernel_size=(config.conv_window, 1),
                padding=(config.conv_window // 2, 0),
                bias=False,
                groups=config.num_attention_heads,
            )

    # 将输入层转换为分数矩阵的函数
    def transpose_for_scores(self, layer):
        # 计算新的层形状
        new_layer_shape = layer.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 重塑层
        layer = layer.view(*new_layer_shape)
        # 转置维度
        return layer.permute(0, 2, 1, 3)

# 从 transformers.models.bert.modeling_bert.BertSelfOutput 复制的 YosoSelfOutput 类
class YosoSelfOutput(nn.Module):
    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化线性层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化 LayerNorm 层
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # dropout
        hidden_states = self.dropout(hidden_states)
        # LayerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回结果
        return hidden_states

# 定义 YosoAttention 类
class YosoAttention(nn.Module):
    # 初始化 YosoLayer 类
    def __init__(self, config, position_embedding_type=None):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个 YosoSelfAttention 对象
        self.self = YosoSelfAttention(config, position_embedding_type=position_embedding_type)
        # 创建一个 YosoSelfOutput 对象
        self.output = YosoSelfOutput(config)
        # 初始化一个空的被修剪头的集合
        self.pruned_heads = set()
    
    # 修剪注意力头
    def prune_heads(self, heads):
        # 如果没有头需要修剪，直接返回
        if len(heads) == 0:
            return
        # 找出可被修剪的头及其对应的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )
    
        # 修剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
    
        # 更新超参数并存储被修剪的头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)
    
    # 前向传播
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # 调用 self.self 的前向传播方法
        self_outputs = self.self(hidden_states, attention_mask, output_attentions)
        # 调用 self.output 的前向传播方法
        attention_output = self.output(self_outputs[0], hidden_states)
        # 构建输出元组，包括 attention_output 和其他可能的输出
        outputs = (attention_output,) + self_outputs[1:]
        return outputs
# 从transformers.models.bert.modeling_bert.BertIntermediate中复制过来的类
class YosoIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将输入的特征向量维度从config.hidden_size变为config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果config.hidden_act是字符串类型，则将对应的函数从预定义的ACT2FN字典中取出
        # 否则，将config.hidden_act作为激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层对输入的隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 使用中间激活函数对线性变换的结果进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertOutput中复制过来的类
class YosoOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将输入的特征向量维度从config.intermediate_size变为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个LayerNorm层，用于对输出进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个dropout层，用于随机丢弃一部分神经元的输出
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层对输入的隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 使用dropout层对线性变换的结果进行随机丢弃一部分神经元的输出
        hidden_states = self.dropout(hidden_states)
        # 将处理后的隐藏状态与输入的原始特征向量进行残差连接
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态
        return hidden_states


class YosoLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置用于前向传播过程中的分块操作的超参数
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # 创建一个SelfAttention层
        self.attention = YosoAttention(config)
        # 是否添加跨层Attention
        self.add_cross_attention = config.add_cross_attention
        # 创建一个Intermediate层
        self.intermediate = YosoIntermediate(config)
        # 创建一个Output层
        self.output = YosoOutput(config)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # 对输入的隐藏状态进行SelfAttention操作
        self_attention_outputs = self.attention(hidden_states, attention_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 对SelfAttention操作得到的隐藏状态进行分块处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # 返回处理后的输出
        return outputs

    def feed_forward_chunk(self, attention_output):
        # 对分块后的SelfAttention输出进行Intermediate层的前向传播操作
        intermediate_output = self.intermediate(attention_output)
        # 对经过Intermediate层的输出进行Output层的前向传播操作
        layer_output = self.output(intermediate_output, attention_output)
        # 返回处理后的输出
        return layer_output


class YosoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建包含多个YosoLayer层的列表
        self.layer = nn.ModuleList([YosoLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否使用梯度检查点（Gradient Checkpointing）技术
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    # 如果输出隐藏状态为真，则初始化一个空元组，否则为 None
        all_hidden_states = () if output_hidden_states else None
    # 如果输出自注意力为真，则初始化一个空元组，否则为 None
        all_self_attentions = () if output_attentions else None

    # 遍历模型的每一层
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态为真，则将当前隐藏状态添加到所有隐藏状态中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果启用梯度检查点并且处于训练状态，则使用梯度检查点函数
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            # 否则，直接调用层模块处理隐藏状态和注意力掩码并输出结果
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, output_attentions)

            # 更新隐藏状态为当前层输出的隐藏状态
            hidden_states = layer_outputs[0]
            # 如果输出自注意力为真，则将当前层的自注意力添加到所有自注意力中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果输出隐藏状态为真，则将最终隐藏状态添加到所有隐藏状态中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，则返回非空元素的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则返回带有交叉注意力的基本模型输出
        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
# 从transformers.models.bert.modeling_bert.BertPredictionHeadTransform复制而来，定义了Yoso模型的预测头转换层
class YosoPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 用线性变换定义了一个全连接层，将隐藏状态的大小转换为相同大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 如果隐藏激活函数是字符串类型，则使用预定义的激活函数字典中对应的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 使用LayerNorm对隐藏状态进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 将线性变换后的隐藏状态通过激活函数进行非线性变换
        hidden_states = self.transform_act_fn(hidden_states)
        # 将非线性变换后的隐藏状态进行LayerNorm归一化
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertLMPredictionHead复制而来，定义了Yoso模型的语言模型预测头
class YosoLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用YosoPredictionHeadTransform对隐藏状态进行转换
        self.transform = YosoPredictionHeadTransform(config)

        # 输出权重与输入嵌入相同，但是每个标记有一个仅输出的偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 针对`resize_token_embeddings`，需要这两个变量之间的链接，以便正确调整偏置
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 使用YosoPredictionHeadTransform对隐藏状态进行转换
        hidden_states = self.transform(hidden_states)
        # 将转换后的隐藏状态进行线性变换
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertOnlyMLMHead复制而来，定义了Yoso模型的仅MLM头部
class YosoOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义YosoLMPredictionHead作为MLM头部的预测部分
        self.predictions = YosoLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # 使用MLM头部的预测部分进行前向传播
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# 定义YosoPreTrainedModel，是处理权重初始化以及下载和加载预训练模型的抽象类
class YosoPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # YosoPreTrainedModel使用的配置类
    config_class = YosoConfig
    # YosoPreTrainedModel的基础模型前缀
    base_model_prefix = "yoso"
    # YosoPreTrainedModel是否支持梯度检查点
    supports_gradient_checkpointing = True
```  
    # 初始化神经网络模块的权重
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布随机初始化权重
            # 此处与 TF 版本稍有不同，TF 使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置，则将偏置初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布随机初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在填充索引，则将填充索引对应的权重初始化为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置初始化为0
            module.bias.data.zero_()
            # 将权重初始化为1
            module.weight.data.fill_(1.0)
# YOSO_START_DOCSTRING 变量存储了模型的文档字符串，用于说明模型的基本信息和参数
YOSO_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`YosoConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
# YOSO_INPUTS_DOCSTRING 变量目前为空，需要添加相应的注释信息
YOSO_INPUTS_DOCSTRING = r"""
        Args:
            input_ids (`torch.LongTensor` of shape `({0})`):
                # 输入序列标记在词汇表中的索引。
                # 可以使用 [`AutoTokenizer`] 去获取这些索引。参见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`] 了解详情。
                # [什么是输入 IDs?](../glossary#input-ids)
            attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
                # 避免对填充标记进行注意力计算的掩码。掩码值在 `[0, 1]` 中选择：
                # - 1 表示**未被掩码**的标记，
                # - 0 表示**被掩码**的标记。
                
                # [什么是注意力掩码？](../glossary#attention-mask)
            token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                # 指示输入的第一部分和第二部分的段标记索引。索引在 `[0, 1]` 中选择：
                # - 0 对应于 *句子 A* 的标记，
                # - 1 对应于 *句子 B* 的标记。

                # [什么是令牌类型 IDs？](../glossary#token-type-ids)
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                # 输入序列标记在位置嵌入中的位置索引。在范围 `[0, config.max_position_embeddings - 1]` 中选择。

                # [什么是位置 IDs？](../glossary#position-ids)
            head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                # 空白化自注意力模块的选定头部的掩码。掩码值在 `[0, 1]` 中选择：
                # - 1 表示**未被掩码**的头部，
                # - 0 表示**被掩码**的头部。
            inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
                # 可选的，可以选择直接传递嵌入表示，而不是传递 `input_ids`。如果您想更好地控制如何将 *input_ids* 索引转换成关联向量，这是很有用的，而不是使用模型的内部嵌入查找矩阵。
            output_attentions (`bool`, *optional*):
                # 是否返回所有注意力层的注意力张量。有关详细信息，请参见返回的张量下的 `attentions`。
            output_hidden_states (`bool`, *optional*):
                # 是否返回所有层的隐藏状态。有关详细信息，请参见返回的张量下的 `hidden_states`。
            return_dict (`bool`, *optional*):
                # 是否返回一个 [`~utils.ModelOutput`] 而不是一个普通的元组。
# YOSO 模型类，继承自 YosoPreTrainedModel
@add_start_docstrings(
    "The bare YOSO Model transformer outputting raw hidden-states without any specific head on top.",
    YOSO_START_DOCSTRING,
)
class YosoModel(YosoPreTrainedModel):
    # 初始化方法，接受配置参数，并调用父类的初始化方法
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # 初始化 YosoEmbeddings 和 YosoEncoder
        self.embeddings = YosoEmbeddings(config)
        self.encoder = YosoEncoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝模型的头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 前向传播方法
    @add_start_docstrings_to_model_forward(YOSO_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



# YOSOForMaskedLM 模型类，继承自 YosoPreTrainedModel
@add_start_docstrings("""YOSO Model with a `language modeling` head on top.""", YOSO_START_DOCSTRING)
class YosoForMaskedLM(YosoPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    # 初始化方法，接受配置参数，并调用父类的初始化方法
    def __init__(self, config):
        super().__init__(config)

        # 初始化 YosoModel 和 YosoOnlyMLMHead
        self.yoso = YosoModel(config)
        self.cls = YosoOnlyMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输��嵌入
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播方法
    @add_start_docstrings_to_model_forward(YOSO_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        定义一个前向传播函数，用于模型的推理和训练。

        参数:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                输入序列的token ids。

            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                输入序列的注意力掩码，指示哪些位置需要被关注。

            token_type_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                输入序列的标记类型，用于区分两个句子。

            position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                输入序列的位置 ids，用于序列位置信息的学习。

            head_mask (`torch.Tensor` of shape `(num_attention_heads,) or (num_hidden_layers, num_attention_heads)`, *optional*):
                头部屏蔽，用于对注意力头进行控制。

            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embedding_size)`, *optional*):
                输入的嵌入特征表示，可用于替代 input_ids。

            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                用于计算遮蔽语言模型的损失的标签。索引应该在 `[-100, 0, ..., config.vocab_size]` 范围内。
                索引设为 `-100` 的token将会被忽略(遮蔽)，只会计算标签在 `[0, ..., config.vocab_size]` 范围内的token的损失。

            output_attentions (`bool`, *optional*):
                是否返回注意力权重。

            output_hidden_states (`bool`, *optional*):
                是否返回所有隐藏层的状态。

            return_dict (`bool`, *optional*):
                是否返回一个字典作为输出。

        返回值:
            Union[Tuple, MaskedLMOutput]: 返回类型为 Tuple 或 MaskedLMOutput 的输出。
        """

        # 如果 return_dict 为空，则将 self.config.use_return_dict 赋值给 return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用输入参数调用 yoso() 函数进行模型推理或训练
        outputs = self.yoso(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取 yoso() 函数的第一项输出
        sequence_output = outputs[0]
        # 使用 cls 函数对序列输出进行分类得到预测分数
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        # 如果labels不为空，则计算遮蔽语言模型的损失
        if labels is not None:
            # 使用交叉熵损失函数
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            # 重塑预测分数和标签的形状，并计算损失
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果 return_dict 为 False，则将预测分数和其它输出项作为元组返回
        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 如果 return_dict 为 True，则将输出以 MaskedLMOutput 类型返回
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
class YosoClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        # 初始化函数，定义一个用于句子级分类任务的头部模块
        super().__init__()
        # 继承父类的初始化方法
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个全连接层，输入和输出维度都是config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 定义一个dropout层，以config.hidden_dropout_prob的概率进行dropout
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        # 定义一个全连接层，输入维度是config.hidden_size，输出维度是config.num_labels

        self.config = config
        # 将config保存为模块属性

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # 获取特征中第一个token的输出，即CLS token
        x = self.dropout(x)
        # 对输入进行dropout处理
        x = self.dense(x)
        # 经过全连接层
        x = ACT2FN[self.config.hidden_act](x)
        # 将全连接层的输出通过激活函数处理
        x = self.dropout(x)
        # 再次对输出进行dropout处理
        x = self.out_proj(x)
        # 输出到num_labels的全连接层
        return x


@add_start_docstrings(
    """YOSO Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks.""",
    YOSO_START_DOCSTRING,
)
# 添加描述信息到模型前向传播函数
class YosoForSequenceClassification(YosoPreTrainedModel):
    def __init__(self, config):
        # 初始化函数
        super().__init__(config)
        # 调用父类的初始化方法
        self.num_labels = config.num_labels
        # 获取标签数量
        self.yoso = YosoModel(config)
        # 初始化YosoModel
        self.classifier = YosoClassificationHead(config)
        # 初始化分类器模块

        # Initialize weights and apply final processing
        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(YOSO_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 添加样例和检查点信息到模型前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 设置返回字典（return_dict）是否为None，为None则使用config的use_return_dict属性
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用模型的yoso方法，传入参数
        outputs = self.yoso(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中获取序列输出
        sequence_output = outputs[0]
        # 将序列输出送入分类器获取logits
        logits = self.classifier(sequence_output)

        loss = None
        # 如果labels不为None
        if labels is not None:
            # 如果配置里没有指定问题类型
            if self.config.problem_type is None:
                # 如果标签数量为1，配置问题类型为回归
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                # 如果标签数量大于1且标签数据类型为long或int，配置问题类型为单标签分类
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                # 否则配置问题类型为多标签分类
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算损失
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        # 如果不需要返回字典
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        # 返回带类型的SequenceClassifierOutput
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义一个多选分类模型，使用 YOSO 模型作为基础，并在其顶部添加了一个多选分类头部（一个线性层和一个 softmax 函数）
# 继承自 YosoPreTrainedModel 类
class YosoForMultipleChoice(YosoPreTrainedModel):

    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化 YOSO 模型
        self.yoso = YosoModel(config)
        # 添加一个线性层，作为分类器前的处理层
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        # 添加一个线性层，作为最终分类器
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并进行最终处理
        self.post_init()

    # 前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 该函数用于处理多选题分类任务的输入和输出
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        # 根据输入的 return_dict 配置决定是否使用返回字典的方式
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 计算输入的选择数量
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        
        # 将输入数据展平为二维
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        
        # 通过 yoso 模型计算输出
        outputs = self.yoso(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 从模型输出中提取隐藏状态和池化输出
        hidden_state = outputs[0]  # (bs * num_choices, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs * num_choices, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs * num_choices, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs * num_choices, dim)
        logits = self.classifier(pooled_output)
        
        # 将逻辑输出调整为适合多选择任务的形状
        reshaped_logits = logits.view(-1, num_choices)
        
        # 如果提供了标签,计算损失函数
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        
        # 根据 return_dict 配置返回结果
        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
    
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 引入 add_start_docstrings 函数，用于为 YOSO 模型添加文档字符串
# 在 YOSO 模型上添加了一个标记分类头部（隐藏状态输出的线性层），例如用于命名实体识别（NER）任务
# 引入 YOSO_START_DOCSTRING，用于在文档字符串中描述 YOSO 模型
class YosoForTokenClassification(YosoPreTrainedModel):
    # 初始化函数，接收配置参数
    def __init__(self, config):
        # 调用父类 YosoPreTrainedModel 的初始化函数
        super().__init__(config)
        # 保存标签的数量
        self.num_labels = config.num_labels

        # 创建 YosoModel 对象
        self.yoso = YosoModel(config)
        # 创建丢弃层对象
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建分类器对象
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 引入 add_start_docstrings_to_model_forward 和 add_code_sample_docstrings 函数，用于为 forward 方法添加文档字符串
    # forward 方法用于模型的前向传播
    # 接收多个输入参数，包括输入的标记 ID、注意力遮罩、标记类型 ID 等
    # 返回一个 TokenClassifierOutput 类型的输出
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义函数，接受输入并返回输出
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 如果 return_dict 不为空，则使用其值，否则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 对输入进行序列标注的预测
        outputs = self.yoso(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取预测结果的序列输出
        sequence_output = outputs[0]

        # 对序列输出进行 dropout 处理
        sequence_output = self.dropout(sequence_output)
        # 将序列输出经过分类器得到 logits
        logits = self.classifier(sequence_output)

        # 初始化损失值
        loss = None
        # 如果存在标签
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 只保留损失的有效部分
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果 return_dict 为 False
        if not return_dict:
            output = (logits,) + outputs[1:]
            # 如果存在损失值，则返回损失值和输出结果的元组，否则只返回输出结果
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 对象，包括损失值、logits、隐藏状态和注意力值
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 添加模型文档字符串，描述该模型用于抽取式问答任务的 YOSO 模型，具有一个用于 span 分类的头部（在隐藏状态输出之上的线性层，
# 用于计算“span 开始对数”和“span 结束对数”）。
@add_start_docstrings(
    """YOSO Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).""",
    YOSO_START_DOCSTRING,
)
# 定义 YosoForQuestionAnswering 类，继承自 YosoPreTrainedModel 类
class YosoForQuestionAnswering(YosoPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 设置分类数为 2
        config.num_labels = 2
        # 记录分类数
        self.num_labels = config.num_labels

        # 实例化 YosoModel，并记录到 self.yoso 属性中
        self.yoso = YosoModel(config)
        # 创建一个线性层，将隐藏状态的输出映射到分类数目，用于输出 span 开始和结束的对数
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义前向传播方法
    @add_start_docstrings_to_model_forward(YOSO_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        # 如果return_dict为None，则使用config中的use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用yoso模型处理给定的输入
        outputs = self.yoso(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中获取序列输出
        sequence_output = outputs[0]

        # 在序列输出上应用qa_outputs以获得logits
        logits = self.qa_outputs(sequence_output)

        # 将logits切分为起始和结束logits
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        # 如果start_positions和end_positions都不为空
        if start_positions is not None and end_positions is not None:
            # 如果处于多GPU环境中，添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 有时起始/结束位置超出模型输入范围，忽略这些位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 使用CrossEntropyLoss计算起始和结束loss
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            # 如果return_dict为False，返回输出
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回QuestionAnsweringModelOutput对象，包括loss、start_logits、end_logits、hidden_states和attentions
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```