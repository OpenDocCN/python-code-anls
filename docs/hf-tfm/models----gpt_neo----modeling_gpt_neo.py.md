# `.\models\gpt_neo\modeling_gpt_neo.py`

```
# 指定编码格式为 UTF-8

# 导入必要的库
import os
from typing import Optional, Tuple, Union

# 导入 PyTorch 库
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入相关模块和函数
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_torch_fx_available,
    logging,
)

# 如果支持 Flash Attention 2.0，则导入相关模块和函数
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# 通过 Torch FX 使 _prepare_4d_causal_attention_mask 成为 FX 图中的叶节点
if is_torch_fx_available():
    # 如果 Torch 版本低于 1.13，导入 torch.fx 模块
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    # 使用 torch.fx.wrap 封装 _prepare_4d_causal_attention_mask 函数
    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置
_CONFIG_FOR_DOC = "GPTNeoConfig"

# GPT-Neo 预训练模型的存档列表
GPT_NEO_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "EleutherAI/gpt-neo-1.3B",
    # 查看所有 GPT-Neo 模型：https://huggingface.co/models?filter=gpt_neo
]

# 用于文档的检查点
_CHECKPOINT_FOR_DOC = "EleutherAI/gpt-neo-1.3B"

# 从 transformers.models.llama.modeling_llama._get_unpad_data 复制的函数
def _get_unpad_data(attention_mask):
    # 计算每个样本中非填充部分的长度
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 找到非填充部分的索引
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 获取批次中最大的序列长度
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # 计算累积的序列长度
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
# 加载 TensorFlow 模型的权重到 PyTorch 模型中
def load_tf_weights_in_gpt_neo(model, config, gpt_neo_checkpoint_path):
    try:
        # 尝试导入所需的模块
        import re
        import tensorflow as tf
    # 如果导入错误，输出错误信息并抛出异常
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    # 获取 TensorFlow 检查点的绝对路径
    tf_path = os.path.abspath(gpt_neo_checkpoint_path)
    # 输出日志，显示正在转换的 TensorFlow 检查点路径
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # 从 TensorFlow 模型中加载权重
    init_vars = tf.train.list_variables(tf_path)
    # 初始化名称列表和数组列表
    names = []
    arrays = []
    # 遍历 TensorFlow 模型中的变量和形状
    for name, shape in init_vars:
        # 排除一些特殊变量名
        if "global_step" not in name and "adam" not in name:
            # 加载变量数据
            array = tf.train.load_variable(tf_path, name)
            # 将加载的数组转换为 float32 类型的 numpy 数组
            array = tf.dtypes.cast(array.squeeze(), tf.float32).numpy()
            # 对变量名进行一些替换，以便与 PyTorch 模型的命名一致
            name = name.replace("attn/q", "attn/attention/q_proj/w")
            name = name.replace("attn/k", "attn/attention/k_proj/w")
            name = name.replace("attn/v", "attn/attention/v_proj/w")
            name = name.replace("attn/o", "attn/attention/out_proj/w")
            name = name.replace("norm_1", "ln_1")
            name = name.replace("norm_2", "ln_2")
            name = name.replace("attn/compute_output_bias/o_b", "attn/attention/out_proj/b")
            name = name.replace("conv1d_main/c_fc/kernel", "c_fc/w")
            name = name.replace("conv1d_main/c_fc/bias", "c_fc/b")
            name = name.replace("conv1d_main/c_proj/kernel", "c_proj/w")
            name = name.replace("conv1d_main/c_proj/bias", "c_proj/b")
            # 将处理后的名称添加到名称列表中
            names.append(name)
            # 将处理后的数组添加到数组列表中
            arrays.append(array)
``` 
    for name, array in zip(names, arrays):
        # 从元组 names 和 arrays 中依次取出 name 和 array
        name = name[5:]  # 跳过 "gpt2/"，截取 name 变量的子串
        name = name.split("/")  # 使用 "/" 将 name 变量的字符串分割成列表
        pointer = model.transformer  # 初始化指针变量
        
        # 遍历 name 列表中的每个字符
        for m_name in name:
            # 使用正则表达式检查 m_name 是否匹配字母+数字的格式
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                # 将 m_name 按照数字分割成列表
                scope_names = re.split(r"(\d+)", m_name)
            else:
                # 否则直接作为列表中的一个元素
                scope_names = [m_name]
            
            # 根据不同的条件设置 pointer 指针移动的位置
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            
            # 如果 scope_names 的长度大于等于 2
            if len(scope_names) >= 2:
                num = int(scope_names[1])  # 取出数字部分
                pointer = pointer[num]  # 更新指针指向的位置
        
        # 如果 name 的倒数第一个字符是 "w"，并且倒数第二个字符是 ["out_proj", "k_proj", "q_proj", "v_proj", "c_proj", "c_fc"] 中的一个
        if name[-1] == "w" and name[-2] in ["out_proj", "k_proj", "q_proj", "v_proj", "c_proj", "c_fc"]:
            array = array.transpose()  # 对 array 进行转置操作
        
        # 如果 name 为 ["wte"]
        if name == ["wte"]:
            # 如果词汇表有填充，然后修剪掉填充的嵌入
            array = array[: config.vocab_size]  # 对 array 进行切片操作
        
        # 如果 pointer 和 array 的形状不相等
        if pointer.shape != array.shape:
            raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched {name}")  # 抛出 ValueError 异常
        
        print(f"Initialize PyTorch weight {name}")  # 打印初始化 PyTorch 权重的信息
        pointer.data = torch.from_numpy(array)  # 将 array 转换成 PyTorch 的 Tensor 类型保存到 pointer.data 中

    # 使用词嵌入初始化最终的线性层
    embs = model.transformer.wte.weight  # 获取词嵌入
    lin = nn.Linear(embs.size()[1], embs.size()[0], bias=False)  # 初始化一个线性层
    lin.weight = embs  # 设置线性层的权重
    model.set_output_embeddings(lin)  # 将线性层设置为模型的输出嵌入
    return model  # 返回模型
class GPTNeoSelfAttention(nn.Module):
    # 定义 GPTNeoSelfAttention 类，继承自 nn.Module
    def __init__(self, config, attention_type):
        # 初始化函数，接受 config 和 attention_type 两个参数
        super().__init__()
        # 调用父类的初始化函数
        self.config = config
        # 保存传入的配置参数

        max_positions = config.max_position_embeddings
        # 获取最大位置嵌入的值
        bias = torch.tril(torch.ones((max_positions, max_positions), dtype=bool)).view(
            1, 1, max_positions, max_positions
        )
        # 生成一个下三角的张量作为偏置

        # local causal self attention is a sliding window where each token can only attend to the previous
        # window_size tokens. This is implemented by updating the causal mask such that for each token
        # all other tokens are masked except the previous window_size tokens.
        # 如果 attention_type 为 "local"，则需要更新偏置，使得每个令牌只能关注前 window_size 个令牌
        if attention_type == "local":
            bias = torch.bitwise_xor(bias, torch.tril(bias, -config.window_size))
        # 更新偏置

        self.register_buffer("bias", bias, persistent=False)
        # 注册偏置为模型的缓冲区，不会被保存到模型的状态字典中
        self.register_buffer("masked_bias", torch.tensor(-1e9), persistent=False)
        # 注册掩码偏置为模型的缓冲区，不会被保存到模型的状态字典中

        self.attn_dropout = nn.Dropout(float(config.attention_dropout))
        # 定义注意力分布的 dropout
        self.resid_dropout = nn.Dropout(float(config.resid_dropout))
        # 定义残差连接的 dropout
        self.is_causal = True
        # 指示是否是因果注意力机制

        self.embed_dim = config.hidden_size
        # 获取隐藏层的维度
        self.num_heads = config.num_heads
        # 获取注意力头的数量
        self.head_dim = self.embed_dim // self.num_heads
        # 计算每个注意力头的维度
        if self.head_dim * self.num_heads != self.embed_dim:
            # 如果 embed_dim 不能整除 num_heads，则抛出异常
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        # 定义将输入投影到键空间的线性变换
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        # 定义将输入投影到值空间的线性变换
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        # 定义将输入投影到查询空间���线性变换
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        # 定义输出投影层，包括权重和偏置
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # 将查询、键、值转换为 float32 类型，以避免溢出问题
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        # 计算注意力权重
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        # 获取查询和键的长度
        query_length, key_length = query.size(-2), key.size(-2)
        # 创建自回归遮罩，将多余部分填充为负无穷
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        # 创建遮罩值的张量，保持与注意力权重相同的数据类型和设备
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        # 将自回归遮罩应用于注意力权重
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # 应用外部注意力遮罩
            attn_weights = attn_weights + attention_mask

        # 对注意力权重进行 softmax 归一化
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # 如果需要，对注意力权重应用头部遮罩
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        # 将隐藏状态映射到查询、键、值空间
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # 将查询、键、值分割为多个头
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # 如果存在过去的层信息，则将当前层信息与过去的层信息连接起来
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        # 如果使用缓存，则更新当前层信息
        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # 计算注意力输出和注意力权重
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # 合并多个头的注意力输出
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        # 构建输出元组
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)
# 定义了一个名为 GPTNeoFlashAttention2 的类，继承自 GPTNeoSelfAttention 类，用于处理 GPTNeo 模型中的闪存注意力机制。该模块中的权重保持不变。主要的改动在于前向传播过程中需要正确调用闪存注意力的公共 API，并在输入中处理填充标记（padding tokens）的情况。

class GPTNeoFlashAttention2(GPTNeoSelfAttention):

    """
    GPTNeo 闪存注意力模块。该模块继承自 `GPTNeoSelfAttention`，因为模块的权重保持不变。唯一需要更改的是在前向传播过程中，需要正确调用闪存注意力的公共 API，并处理输入中可能包含的填充标记。
    """

    # 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__ 复制过来的初始化方法
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: 一旦 RoCm 上的 Flash Attention 升级到 2.1 版本，应该将此部分移除。
        # flash_attn
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]  # 获取批量大小
            # 对输入进行去填充（unpad）操作
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length  # 调用_upad_input方法去填充输入
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens  # 获取填充后的序列长度
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens  # 获取每个批次中最大的填充后的序列长度

            # 使用flash_attn_varlen_func计算注意力分数
            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,  # 设置softmax的缩放比例
                causal=causal,  # 是否为因果注意力
            )

            # 对填充后的注意力分数进行填充
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)  # 调用pad_input方法进行填充
        else:
            # 若序列中没有填充标记，则使用flash_attn_func计算注意力分数
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal  # 传入参数计算注意力分数
            )

        return attn_output  # 返回注意力分数

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input
    # 对输入进行处理，根据注意力掩码去除填充部分
    # 获取注意力掩码的未填充数据的索引、当前序列长度和批处理中的最大序列长度
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # 根据未填充数据的索引将 key_layer 重塑，并重新索引
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        # 根据未填充数据的索引将 value_layer 重塑，并重新索引
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        # 如果查询长度等于键值对序列长度，则根据未填充数据的索引将 query_layer 重塑，并重新索引
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        # 如果查询长度为 1，则处理特殊情况
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        # 如果查询长度不等于 1 且不等于键值对序列长度，则进行未填充数据处理
        else:
            # 对注意力掩码进行处理，截取最后的查询长度部分
            attention_mask = attention_mask[:, -query_length:]
            # 对 query_layer 进行处理，返回未填充数据的索引、当前序列长度和批处理中的最大序列长度
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
# 定义一个字典，将字符串映射到相应的自注意力类
GPT_NEO_ATTENTION_CLASSES = {
    "eager": GPTNeoSelfAttention,  # 如果字符串为"eager"，则映射到GPTNeoSelfAttention类
    "flash_attention_2": GPTNeoFlashAttention2,  # 如果字符串为"flash_attention_2"，则映射到GPTNeoFlashAttention2类
}

# 定义GPTNeoAttention类，继承自nn.Module
class GPTNeoAttention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.layer_id = layer_id  # 层的编号
        self.attention_layers = config.attention_layers  # 注意力层列表
        self.attention_type = self.attention_layers[layer_id]  # 获取指定层的注意力类型

        # 根据注意力类型选择对应的自注意力实现类
        if self.attention_type in ["global", "local"]:
            self.attention = GPT_NEO_ATTENTION_CLASSES[config._attn_implementation](config, self.attention_type)
        else:
            # 如果注意力类型不是"global"或"local"，则抛出未实现的错误
            raise NotImplementedError(
                "Only attn layer types 'global' and 'local' exist, but got `config.attention_layers`: "
                f"{config.attention_layers}. Select attn layer types from ['global', 'local'] only."
            )

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        # 调用选择的自注意力类的forward方法，进行自注意力计算
        return self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )


# 定义GPTNeoMLP类，用于多层感知机结构
class GPTNeoMLP(nn.Module):
    def __init__(self, intermediate_size, config):  # 在MLP中，intermediate_size等于4倍的隐藏层大小
        super().__init__()
        embed_dim = config.hidden_size  # 嵌入维度等于隐藏层大小
        # 第一个线性层，输入维度为隐藏层大小，输出维度为中间层大小
        self.c_fc = nn.Linear(embed_dim, intermediate_size)
        # 第二个线性层，输入维度为中间层大小，输出维度为隐藏层大小
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        # 激活函数，根据配置选择激活函数类型
        self.act = ACT2FN[config.activation_function]
        # dropout层，使用给定的丢弃率
        self.dropout = nn.Dropout(float(config.resid_dropout))

    def forward(self, hidden_states):
        # 在第一个线性层后应用激活函数
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        # 在第二个线性层后应用dropout
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# 定义GPTNeoBlock类，代表GPT-Neo的一个块
class GPTNeoBlock(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        hidden_size = config.hidden_size  # 隐藏层大小
        # Layer normalization层，对隐藏状态进行规范化
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # 注意力层，使用GPTNeoAttention类
        self.attn = GPTNeoAttention(config, layer_id)
        # Layer normalization层，对注意力输出进行规范化
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # 多层感知机结构，使用GPTNeoMLP类
        self.mlp = GPTNeoMLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        ):
        # 保存当前隐藏状态作为残差
        residual = hidden_states
        # 对当前隐藏状态进行 LayerNormalization
        hidden_states = self.ln_1(hidden_states)
        # 使用注意力机制处理当前隐藏状态
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 获取注意力机制的输出
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        # 获取除了注意力机制输出外的其他输出
        outputs = attn_outputs[1:]
        # 残差连接
        hidden_states = attn_output + residual

        # 保存当前隐藏状态作为残差
        residual = hidden_states
        # 对当前隐藏状态进行 LayerNormalization
        hidden_states = self.ln_2(hidden_states)
        # 使用多层感知机处理当前隐藏状态
        feed_forward_hidden_states = self.mlp(hidden_states)
        # 残差连接
        hidden_states = residual + feed_forward_hidden_states

        # 如果使用缓存，则将隐藏状态添加到输出中
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            # 否则只保留隐藏状态和其他输出
            outputs = (hidden_states,) + outputs[1:]

        # 返回输出结果，包括隐藏状态、present、(attentions, cross_attentions)
        return outputs  # hidden_states, present, (attentions, cross_attentions)
class GPTNeoPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为 GPTNeoConfig
    config_class = GPTNeoConfig
    # 加载 TensorFlow 权重的函数为 load_tf_weights_in_gpt_neo
    load_tf_weights = load_tf_weights_in_gpt_neo
    # 基础模型前缀为 "transformer"
    base_model_prefix = "transformer"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不需要拆分的模块为 "GPTNeoBlock"
    _no_split_modules = ["GPTNeoBlock"]
    # 跳过设备放置的键为 "past_key_values"
    _skip_keys_device_placement = "past_key_values"
    # 支持 flash attention 2
    _supports_flash_attn_2 = True

    def __init__(self, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear,)):
            # 对线性层的权重进行初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 对嵌入层的权重进行初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 对 LayerNorm 层的权重进行初始化
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


GPT_NEO_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GPTNeoConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GPT_NEO_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "The bare GPT Neo Model transformer outputting raw hidden-states without any specific head on top.",
    GPT_NEO_START_DOCSTRING,
)
class GPTNeoModel(GPTNeoPreTrainedModel):
    # 初始化函数，接受配置参数并调用父类的初始化函数
    def __init__(self, config):
        super().__init__(config)

        # 设置嵌入维度为隐藏大小
        self.embed_dim = config.hidden_size
        # 创建词嵌入层，词汇表大小为配置中的vocab_size，嵌入维度为self.embed_dim
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        # 创建位置嵌入层，最大位置编码为配置中的max_position_embeddings，嵌入维度为self.embed_dim
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        # 创建Dropout层，概率为配置中的embed_dropout
        self.drop = nn.Dropout(float(config.embed_dropout))
        # 创建多层GPTNeoBlock模块列表，每层的配置为config，层数为config.num_layers
        self.h = nn.ModuleList([GPTNeoBlock(config, layer_id=i) for i in range(config.num_layers)])
        # 根据配置中的_attn_implementation判断是否使用flash_attention_2
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        # 创建LayerNorm层，输入维度为self.embed_dim，epsilon为配置中的layer_norm_epsilon
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # 初始化梯度检查点为False
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.wte

    # 设置输入嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    # 前向传播函数，接受多个输入参数，具体含义可查看GPT_NEO_INPUTS_DOCSTRING
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 定义 GPT Neo 模型，带有一个语言建模头部（线性层，权重与输入嵌入层绑定）
@add_start_docstrings(
    """
    The GPT Neo Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT_NEO_START_DOCSTRING,
)
class GPTNeoForCausalLM(GPTNeoPreTrainedModel):
    # 定义权重绑定的键值
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        # 初始化函数
        super().__init__(config)
        # 创建 GPT Neo 模型
        self.transformer = GPTNeoModel(config)
        # 创建语言建模头部
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_output_embeddings(self):
        # 获取输出嵌入
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置输出嵌入
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # 为生成准备输入
        token_type_ids = kwargs.get("token_type_ids", None)
        # 忽略过去键值覆盖的标记
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递最后一个输入 ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认为旧行为：仅保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # 为批量生成动态创建位置 ID
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # 如果传递了 `inputs_embeds`，我们只想在第一个生成步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )

        return model_inputs

    @add_start_docstrings_to_model_forward(GPT_NEO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义一个前向传播函数，接受多个输入参数，都是可选的 torch.Tensor 类型
    def forward(
        # 输入的 token IDs
        input_ids: Optional[torch.Tensor] = None,
        # 用于存储过去的 key-values 的元组
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        # 注意力掩码
        attention_mask: Optional[torch.Tensor] = None,
        # token 类型 IDs
        token_type_ids: Optional[torch.Tensor] = None,
        # 位置 IDs
        position_ids: Optional[torch.Tensor] = None,
        # 头部掩码
        head_mask: Optional[torch.Tensor] = None,
        # 输入的嵌入向量
        inputs_embeds: Optional[torch.Tensor] = None,
        # 标签
        labels: Optional[torch.Tensor] = None,
        # 是否使用缓存
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典形式的结果
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # 确定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用transformer处理输入数据
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # 通过lm_head获取语言模型的logits
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # 将标签移动到正确的设备以启用模型并行处理
            labels = labels.to(lm_logits.device)
            # 以fp32计算损失，以匹配mesh-tf版本
            lm_logits = lm_logits.to(torch.float32)

            # 移动logits以使得tokens < n 预测 n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 展平tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        # 返回一个元组，其中每个元素也是一个元组，用于重新排序`past_key_values`缓存，以匹配每个生成步骤的正确beam_idx
        return tuple(
            # 对于每个层的过去状态，根据beam_idx重新排序，转移到相同设备上
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
# 定义一个带有顶部序列分类头（线性层）的GPTNeo模型转换器
# GPTNeoForSequenceClassification使用最后一个令牌进行分类，与其他因果模型（例如GPT-1）一样
# 如果在配置中定义了pad_token_id，则找到每行中不是填充令牌的最后一个令牌。如果未定义pad_token_id，则简单地取每行批次中的最后一个值
# 当传递inputs_embeds而不是input_ids时，无法猜测填充令牌，因此执行相同操作（取每行批次中的最后一个值）
class GPTNeoForSequenceClassification(GPTNeoPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPTNeoModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    )

# GPT Neo模型，顶部带有令牌分类头（隐藏状态输出顶部的线性层），例如用于命名实体识别（NER）任务
class GPTNeoForTokenClassification(GPTNeoPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = GPTNeoModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    )
    # 定义一个前向传播函数，接受多个输入参数并返回预测结果
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token ID 序列
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # 用于存储过去的键值对
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码，指定哪些位置需要被注意
        token_type_ids: Optional[torch.LongTensor] = None,  # token 类型 ID
        position_ids: Optional[torch.LongTensor] = None,  # 位置 ID
        head_mask: Optional[torch.FloatTensor] = None,  # 多头注意力机制的头部掩码
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量
        labels: Optional[torch.LongTensor] = None,  # 用于计算损失的标签
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式的结果
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 为 None，则使用配置中的 use_return_dict 值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 Transformer 处理输入数据
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取 Transformer 输出的隐藏状态
        hidden_states = transformer_outputs[0]
        # 对隐藏状态进行 dropout 处理
        hidden_states = self.dropout(hidden_states)
        # 使用分类器对隐藏状态进行分类
        logits = self.classifier(hidden_states)

        # 初始化损失值为 None
        loss = None
        # 如果存在标签，则计算损��
        if labels is not None:
            labels = labels.to(logits.device)  # 将标签移动到 logits 所在的设备上
            loss_fct = CrossEntropyLoss()  # 使用交叉熵损失函数
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # 计算损失

        # 如果不返回字典形式的结果
        if not return_dict:
            output = (logits,) + transformer_outputs[2:]  # 输出结果包括 logits 和其他 Transformer 输出
            return ((loss,) + output) if loss is not None else output  # 如果存在损失，则返回损失和输出结果，否则只返回输出结果

        # 返回 TokenClassifierOutput 对象，包括损失、logits、隐藏状态和注意力权重
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
# 定义一个 GPT-Neo 模型，用于提取式问答任务，例如 SQuAD，包含一个用于计算“span start logits”和“span end logits”的线性层
class GPTNeoForQuestionAnswering(GPTNeoPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 获取标签数量
        self.num_labels = config.num_labels
        # 创建 GPTNeo 模型
        self.transformer = GPTNeoModel(config)
        # 创建一个线性层用于输出问题回答
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用transformer模型处理输入数据
        outputs = self.transformer(
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

        # 获取模型输出的序列输出
        sequence_output = outputs[0]

        # 使用qa_outputs对序列输出进行分类
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果在多GPU上，添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 有时开始/结束位置超出模型输入范围，忽略这些位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回问题回答模型的输出
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```