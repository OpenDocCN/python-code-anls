# `.\models\gpt_neo\modeling_gpt_neo.py`

```
# 设置编码为 UTF-8，确保源文件可以正确解析非 ASCII 字符
# 版权声明和许可信息，遵循 Apache License 2.0
# 此模块定义了 PyTorch 中的 GPT Neo 模型

# 引入操作系统相关的功能
import os
# 引入类型提示模块中的类型
from typing import Optional, Tuple, Union

# 引入 PyTorch 相关模块
import torch
# 引入 PyTorch 中的函数库和功能
import torch.nn.functional as F
import torch.utils.checkpoint
# 引入 PyTorch 中的损失函数
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 引入自定义的激活函数映射表
from ...activations import ACT2FN
# 引入处理注意力掩码相关的工具函数
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
# 引入模型输出相关的类定义
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
# 引入模型工具函数
from ...modeling_utils import PreTrainedModel
# 引入 PyTorch 工具函数
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
# 引入通用工具函数
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_torch_fx_available,
    logging,
)

# 如果支持 Flash Attention 2，引入相应的函数
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# 如果支持 Torch FX，包装 _prepare_4d_causal_attention_mask 函数，使其成为 FX 图中的一个叶节点
if is_torch_fx_available():
    # 如果 Torch 版本低于 1.13，则引入 torch.fx 模块
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置信息
_CONFIG_FOR_DOC = "GPTNeoConfig"

# 预训练模型存档列表
GPT_NEO_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "EleutherAI/gpt-neo-1.3B",
    # 更多 GPT Neo 模型信息可以在 https://huggingface.co/models?filter=gpt_neo 查看
]

# 用于文档的检查点信息
_CHECKPOINT_FOR_DOC = "EleutherAI/gpt-neo-1.3B"

# 从 transformers.models.llama.modeling_llama._get_unpad_data 复制的函数
def _get_unpad_data(attention_mask):
    # 计算每个样本的序列长度之和
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 找出注意力掩码中非零元素的索引
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 计算批次中的最大序列长度
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # 计算累积的序列长度
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    # 定义一个名为`data_transform`的函数，接受参数`data`
    def data_transform(data):
        # 将`data`参数以逗号为分隔符进行切分，并返回切分后的结果列表
        return data.split(',')
    # 导入必要的模块：re 用于正则表达式，tf 用于 TensorFlow 操作
    try:
        import re
        import tensorflow as tf
    except ImportError:
        # 如果导入失败，记录错误并抛出异常
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    
    # 获取 TensorFlow 模型的绝对路径
    tf_path = os.path.abspath(gpt_neo_checkpoint_path)
    # 输出日志，指示正在从 TensorFlow checkpoint 转换
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    
    # 加载 TF 模型的变量列表
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    
    # 遍历每个变量名和形状
    for name, shape in init_vars:
        # 排除特定的变量名，如全局步数和优化器参数
        if "global_step" not in name and "adam" not in name:
            # 加载 TF 模型中的变量值
            array = tf.train.load_variable(tf_path, name)
            # 将数组转换为 numpy 数组，并确保数据类型为 float32
            array = tf.dtypes.cast(array.squeeze(), tf.float32).numpy()
            
            # 替换特定名称以适应 PyTorch 模型结构
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

            # 将处理后的名称添加到列表中
            names.append(name)
            # 将处理后的数组添加到列表中
            arrays.append(array)
    for name, array in zip(names, arrays):
        name = name[5:]  # 跳过前缀 "gpt2/"
        name = name.split("/")  # 将文件路径分割为列表

        pointer = model.transformer  # 初始化指针为模型的transformer部分

        # 遍历分割后的文件路径名列表
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)  # 如果匹配字母+数字的模式，则分割名字
            else:
                scope_names = [m_name]

            # 根据scope_names的第一个元素选择指针的属性
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")  # 如果是"w"或"g"，选择weight属性
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")  # 如果是"b"，选择bias属性
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])  # 如果是"wpe"或"wte"，选择对应的属性
                pointer = getattr(pointer, "weight")  # 再选择weight属性
            else:
                pointer = getattr(pointer, scope_names[0])  # 否则选择scope_names的第一个元素作为属性名

            # 如果scope_names的长度大于等于2，选择指针中的第num个元素
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]

        # 如果文件路径的最后一个元素是"w"并且倒数第二个元素在指定的列表中，则对array进行转置操作
        if name[-1] == "w" and name[-2] in ["out_proj", "k_proj", "q_proj", "v_proj", "c_proj", "c_fc"]:
            array = array.transpose()

        # 如果文件路径名为["wte"]，则根据配置截断array的长度
        if name == ["wte"]:
            array = array[: config.vocab_size]

        # 检查pointer和array的形状是否匹配，如果不匹配则抛出异常
        if pointer.shape != array.shape:
            raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched {name}")

        # 打印初始化的PyTorch权重信息
        print(f"Initialize PyTorch weight {name}")

        # 将array转换为torch张量，并赋值给pointer的data属性
        pointer.data = torch.from_numpy(array)

    # 初始化最终的线性层，使用word embeddings
    embs = model.transformer.wte.weight  # 获取模型transformer部分的word embeddings
    lin = nn.Linear(embs.size()[1], embs.size()[0], bias=False)  # 创建一个线性层，输入和输出大小根据embs的形状确定，不带偏置
    lin.weight = embs  # 将embs作为线性层的权重
    model.set_output_embeddings(lin)  # 设置模型的输出embeddings为lin
    return model  # 返回更新后的模型
# 定义一个名为 GPTNeoSelfAttention 的类，继承自 nn.Module
class GPTNeoSelfAttention(nn.Module):
    # 初始化函数，接受 config 和 attention_type 两个参数
    def __init__(self, config, attention_type):
        super().__init__()
        self.config = config

        # 从 config 中获取最大位置嵌入数，并创建一个布尔类型的下三角矩阵作为初始的偏置
        max_positions = config.max_position_embeddings
        bias = torch.tril(torch.ones((max_positions, max_positions), dtype=bool)).view(
            1, 1, max_positions, max_positions
        )

        # 如果 attention_type 是 "local"，则更新偏置，使每个标记只能关注前 window_size 个标记
        if attention_type == "local":
            bias = torch.bitwise_xor(bias, torch.tril(bias, -config.window_size))

        # 将偏置作为缓冲区注册到模型中，不会被视为模型参数
        self.register_buffer("bias", bias, persistent=False)
        # 注册一个固定的掩码偏置，用于在 self-attention 中屏蔽无效位置
        self.register_buffer("masked_bias", torch.tensor(-1e9), persistent=False)

        # 定义注意力(dropout)和残差(dropout)的层
        self.attn_dropout = nn.Dropout(float(config.attention_dropout))
        self.resid_dropout = nn.Dropout(float(config.resid_dropout))
        self.is_causal = True

        # 获取隐藏层大小和注意力头数，并计算每个头的维度
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        # 检查 embed_dim 是否能被 num_heads 整除
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: "
                f"{self.num_heads})."
            )

        # 定义线性变换层，用于生成查询(q_proj)、键(k_proj)、值(v_proj)和输出(out_proj)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    # 将输入张量分割为多头注意力张量
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    # 将多头注意力张量合并为原始张量
    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Keep the attention weights computation in fp32 to avoid overflow issues
        # 将注意力权重的计算保持在fp32中，以避免溢出问题
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        # Compute attention weights using matrix multiplication
        # 使用矩阵乘法计算注意力权重
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        # Determine the dimensions for causal masking
        # 确定因果遮罩的维度
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        # Set the mask value to the minimum value of the data type of attn_weights
        # 将掩码值设置为attn_weights数据类型的最小值
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)

        # Apply causal mask to attention weights
        # 将因果遮罩应用于注意力权重
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the provided attention mask
            # 应用提供的注意力掩码
            attn_weights = attn_weights + attention_mask

        # Apply softmax to get normalized attention weights
        # 应用softmax函数以获取归一化的注意力权重
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Apply dropout to attention weights
        # 对注意力权重应用dropout
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            # Apply head mask if provided
            # 如果提供了头部掩码，则应用头部掩码
            attn_weights = attn_weights * head_mask

        # Compute attention output by weighted sum with value
        # 通过与value的加权求和计算注意力输出
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
        # Project hidden states to query, key, and value tensors
        # 将隐藏状态投影到query、key和value张量上
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Split heads for multi-head attention
        # 对多头注意力进行头部分离
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            # Concatenate past key and value with current key and value
            # 将过去的key和value与当前的key和value拼接起来
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            # Create present tuple if caching is enabled
            # 如果启用缓存，创建present元组
            present = (key, value)
        else:
            present = None

        # Compute attention output and attention weights using _attn function
        # 使用_attn函数计算注意力输出和注意力权重
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # Merge heads back together
        # 将头部重新合并在一起
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        # Project attention output to get final output
        # 投影注意力输出以获得最终输出
        attn_output = self.out_proj(attn_output)

        # Apply residual dropout
        # 应用残差dropout
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)

        if output_attentions:
            # Include attention weights in outputs if requested
            # 如果需要，将注意力权重包含在输出中
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)
class GPTNeoFlashAttention2(GPTNeoSelfAttention):
    """
    GPTNeo flash attention module. This module inherits from `GPTNeoSelfAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignment, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        # Forward pass method for the attention module

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        # Forward pass method for the flash attention mechanism
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
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        # Determine if causal masking is required based on `_flash_attn_uses_top_left_mask` and `query_length`
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # Temporary check for specific conditions in Flash Attention for RoCm
            causal = self.is_causal and query_length != 1

        # Apply unpadding if the input contains padding tokens
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            # Call _upad_input to unpad the input based on attention_mask and query_length
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            # Extract sequence lengths from cu_seq_lens
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            # Extract maximum sequence lengths from max_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            # Perform Flash Attention with variable-length sequences
            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            # Pad the attention output based on the unpadding indices
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            # If no attention_mask is provided, apply regular Flash Attention
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        # Return the computed attention output
        return attn_output
    ```
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 获取未填充数据的索引、当前序列长度和批次中的最大序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        
        # 获取 key_layer 的形状信息
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape
        
        # 根据未填充的索引重新排列 key_layer 和 value_layer 的数据
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        
        # 根据查询长度调整 query_layer
        if query_length == kv_seq_len:
            # 如果查询长度等于 key/value 序列长度，则直接使用索引 k 对 query_layer 进行重新排列
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            # 如果查询长度为 1，则处理为每个批次生成一个长度为 1 的序列
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 这里有一个 memcpy，这样做非常不好。
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # 否则，根据查询长度和注意力掩码对 query_layer 进行未填充处理
            # 注意：-query_length 切片假设左填充。
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回调整后的 query_layer、key_layer、value_layer，以及相关的索引和长度信息
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
# 定义一个字典，将字符串映射到不同的注意力类
GPT_NEO_ATTENTION_CLASSES = {
    "eager": GPTNeoSelfAttention,
    "flash_attention_2": GPTNeoFlashAttention2,
}

# GPTNeoAttention 类，用于处理注意力机制
class GPTNeoAttention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.attention_layers = config.attention_layers  # 从配置中获取注意力层列表
        self.attention_type = self.attention_layers[layer_id]  # 获取当前层的注意力类型

        # 根据不同的注意力类型选择不同的注意力实现类
        if self.attention_type in ["global", "local"]:
            self.attention = GPT_NEO_ATTENTION_CLASSES[config._attn_implementation](config, self.attention_type)
        else:
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
        # 调用注意力机制的前向传播
        return self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )


# GPTNeoMLP 类，多层感知机部分的实现
class GPTNeoMLP(nn.Module):
    def __init__(self, intermediate_size, config):  # 在 MLP 中，intermediate_size=4 * hidden_size
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Linear(embed_dim, intermediate_size)  # 全连接层，将输入维度转换为中间层维度
        self.c_proj = nn.Linear(intermediate_size, embed_dim)  # 全连接层，将中间层维度转换为输出维度
        self.act = ACT2FN[config.activation_function]  # 激活函数，根据配置选择相应的激活函数
        self.dropout = nn.Dropout(float(config.resid_dropout))  # Dropout 层，用于防止过拟合

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)  # 全连接层变换
        hidden_states = self.act(hidden_states)  # 激活函数变换
        hidden_states = self.c_proj(hidden_states)  # 再次全连接层变换
        hidden_states = self.dropout(hidden_states)  # Dropout 处理
        return hidden_states


# GPTNeoBlock 类，GPT-Neo 模型的一个块
class GPTNeoBlock(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.intermediate_size if config.intermediate_size is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)  # LayerNorm 层，第一层
        self.attn = GPTNeoAttention(config, layer_id)  # 注意力层
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)  # LayerNorm 层，第二层
        self.mlp = GPTNeoMLP(inner_dim, config)  # MLP 层，多层感知机

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        # 块的前向传播，依次经过 LayerNorm、注意力、LayerNorm、MLP
        ):
            # 保存输入的 hidden_states 到 residual 变量中，以便后续使用残差连接
            residual = hidden_states
            # 对当前的 hidden_states 进行 Layer Normalization 处理
            hidden_states = self.ln_1(hidden_states)
            # 调用注意力机制模块进行注意力计算
            attn_outputs = self.attn(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            # 获取注意力计算的输出结果
            attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
            # 获取额外的输出结果
            outputs = attn_outputs[1:]
            # 残差连接，将注意力计算的输出和之前保存的 residual 相加
            hidden_states = attn_output + residual

            # 保存当前的 hidden_states 到 residual 变量中，用于后续的残差连接
            residual = hidden_states
            # 对当前的 hidden_states 进行第二层 Layer Normalization 处理
            hidden_states = self.ln_2(hidden_states)
            # 使用 MLP 模块进行前馈网络计算
            feed_forward_hidden_states = self.mlp(hidden_states)
            # 残差连接，将第一次保存的 residual 和前馈网络计算结果相加
            hidden_states = residual + feed_forward_hidden_states

            # 如果 use_cache 为 True，则将 hidden_states 添加到输出中
            if use_cache:
                outputs = (hidden_states,) + outputs
            else:
                # 否则，只将 hidden_states 和第一个额外输出添加到输出中
                outputs = (hidden_states,) + outputs[1:]

            # 返回输出结果，通常包括 hidden_states, present, (attentions, cross_attentions)
            return outputs
class GPTNeoPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 GPTNeoConfig 作为配置类
    config_class = GPTNeoConfig
    # 使用 load_tf_weights_in_gpt_neo 函数来加载 TensorFlow 权重
    load_tf_weights = load_tf_weights_in_gpt_neo
    # 基础模型前缀为 "transformer"
    base_model_prefix = "transformer"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不需要拆分的模块列表
    _no_split_modules = ["GPTNeoBlock"]
    # 跳过的设备放置键名
    _skip_keys_device_placement = "past_key_values"
    # 支持闪电注意力的第二个版本
    _supports_flash_attn_2 = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear,)):
            # 对于线性层，使用正态分布初始化权重，标准差为配置中的 initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果有偏置项，则初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 对于嵌入层，使用正态分布初始化权重，标准差为配置中的 initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果有 padding_idx，则将对应位置的权重初始化为零
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 对于 LayerNorm 层，初始化偏置为零，权重为全一
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
# GPTNeoModel 继承自 GPTNeoPreTrainedModel，并添加了文档字符串
class GPTNeoModel(GPTNeoPreTrainedModel):
    # 初始化方法，接收配置参数并调用父类初始化方法
    def __init__(self, config):
        super().__init__(config)

        # 设置嵌入维度为隐藏大小
        self.embed_dim = config.hidden_size
        # 创建词嵌入层，形状为（词汇表大小，嵌入维度）
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        # 创建位置嵌入层，形状为（最大位置编码数，嵌入维度）
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        # 创建dropout层，使用配置中的嵌入dropout率
        self.drop = nn.Dropout(float(config.embed_dropout))
        # 创建多层GPTNeoBlock模块的列表，每层使用相同的配置，编号从0到num_layers-1
        self.h = nn.ModuleList([GPTNeoBlock(config, layer_id=i) for i in range(config.num_layers)])
        # 根据配置中的注意力机制实现，决定是否使用特定的flash_attention_2实现
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        # 创建LayerNorm层，对嵌入维度进行归一化，使用配置中的epsilon参数
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # 梯度检查点初始化为False，表示不启用梯度检查点技术
        self.gradient_checkpointing = False
        # 执行后续初始化操作，包括权重初始化和最终处理
        self.post_init()

    # 返回词嵌入层wte
    def get_input_embeddings(self):
        return self.wte

    # 设置新的输入嵌入到词嵌入层wte
    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    # 前向传播函数，接收多种输入参数，详细见函数注释文档
    @add_start_docstrings_to_model_forward(GPT_NEO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
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
"""
The GPT Neo Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).
"""
# 基于 GPT Neo 模型的语言建模头部的转换器模型
@add_start_docstrings(
    """
    The GPT Neo Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT_NEO_START_DOCSTRING,
)
class GPTNeoForCausalLM(GPTNeoPreTrainedModel):
    # 定义需要绑定权重的键值对
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        # 调用父类构造函数，初始化模型
        super().__init__(config)
        # 初始化 GPT Neo 模型的主体部分
        self.transformer = GPTNeoModel(config)
        # 初始化语言建模头部的线性层，用于预测下一个词的概率分布
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_output_embeddings(self):
        # 返回语言建模头部的输出嵌入
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置新的输出嵌入到语言建模头部
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # 获取 token_type_ids 参数，如果不存在则设为 None
        token_type_ids = kwargs.get("token_type_ids", None)
        
        # 如果存在 past_key_values，则根据其覆盖情况调整 input_ids
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法可能只传递最后一个输入 ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认行为：仅保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # 为批量生成创建动态的 position_ids
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # 如果传递了 inputs_embeds，我们只在第一个生成步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 更新模型输入的字典
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

    # 添加文档字符串到模型的前向方法
    @add_start_docstrings_to_model_forward(GPT_NEO_INPUTS_DOCSTRING)
    # 添加代码示例的文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 此方法用于执行模型的前向传播，接受多个可选参数用于控制传播过程和返回的内容

    self,
        # "self" 是 Python 中类方法的隐式第一个参数，表示当前实例对象

    input_ids: Optional[torch.Tensor] = None,
        # 输入的 token IDs，是一个可选的 PyTorch 张量，默认为 None

    past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        # 先前计算的键值对，作为元组的形式，包含可选的 PyTorch 浮点数张量，默认为 None

    attention_mask: Optional[torch.Tensor] = None,
        # 注意力掩码，用于控制模型关注哪些 token，可选的 PyTorch 张量，默认为 None

    token_type_ids: Optional[torch.Tensor] = None,
        # token 类型 IDs，用于区分不同类型的 token，可选的 PyTorch 张量，默认为 None

    position_ids: Optional[torch.Tensor] = None,
        # token 的位置 IDs，用于指示 token 的位置信息，可选的 PyTorch 张量，默认为 None

    head_mask: Optional[torch.Tensor] = None,
        # 头部掩码，用于屏蔽特定的注意力头部，可选的 PyTorch 张量，默认为 None

    inputs_embeds: Optional[torch.Tensor] = None,
        # 输入的嵌入表示，用于直接提供 token 的嵌入表示而不是通过输入的 token IDs 进行计算，可选的 PyTorch 张量，默认为 None

    labels: Optional[torch.Tensor] = None,
        # 模型的标签，用于计算损失或评估模型输出的正确性，可选的 PyTorch 张量，默认为 None

    use_cache: Optional[bool] = None,
        # 是否使用缓存以加速推断，可选的布尔值，默认为 None

    output_attentions: Optional[bool] = None,
        # 是否返回注意力权重，可选的布尔值，默认为 None

    output_hidden_states: Optional[bool] = None,
        # 是否返回所有隐藏状态，可选的布尔值，默认为 None

    return_dict: Optional[bool] = None,
        # 是否以字典形式返回输出，可选的布尔值，默认为 None
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 Transformer 处理输入数据，获取隐藏状态和其它附加输出
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

        # 通过 lm_head 计算语言模型的 logits
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # 将 labels 移动到正确的设备以启用模型并行计算
            labels = labels.to(lm_logits.device)
            # 为了与 mesh-tf 版本匹配，在 fp32 中计算损失
            lm_logits = lm_logits.to(torch.float32)

            # 将 logits 和 labels 向左移动一个位置，以便 <n 预测 n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 展平 tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        # 如果不使用 return_dict，输出 loss 和 transformer_outputs 的其它部分
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 使用 CausalLMOutputWithPast 返回 loss、logits 和 transformer_outputs 的其它部分
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
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """

        # 返回一个元组的元组，其中每个内部元组包含经过重新排序后的 `past_key_values` 缓存
        return tuple(
            # 对于每个 `layer_past` 中的 `past_state`，按照 `beam_idx` 的顺序重新选择并返回
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
@add_start_docstrings(
    """
    The GPTNeo Model transformer with a sequence classification head on top (linear layer).

    [`GPTNeoForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    GPT_NEO_START_DOCSTRING,
)
class GPTNeoForSequenceClassification(GPTNeoPreTrainedModel):
    """
    GPTNeo model for sequence classification tasks.

    Inherits from `GPTNeoPreTrainedModel` and adds a linear classification layer for sequence classification.
    """

    def __init__(self, config):
        """
        Initializes the GPTNeoForSequenceClassification model.

        Args:
            config (:class:`~transformers.GPTNeoConfig`):
                The configuration object that defines the model architecture.

        Attributes:
            num_labels (int):
                Number of labels for sequence classification.
            transformer (:class:`~transformers.GPTNeoModel`):
                The GPTNeoModel transformer instance.
            score (:class:`~torch.nn.Linear`):
                Linear layer for computing scores for each label.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPTNeoModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(GPT_NEO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
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
    ):
        """
        Performs forward pass of the GPTNeoForSequenceClassification model.

        Args:
            input_ids (torch.Tensor, optional):
                The input token IDs. Shape [batch_size, sequence_length].
            past_key_values (Tuple[torch.FloatTensor], optional):
                Tuple of length 1 containing the cached key and value tensors from previous attention layers.
            attention_mask (torch.Tensor, optional):
                Mask to avoid performing attention on padding tokens. Shape [batch_size, sequence_length].
            token_type_ids (torch.Tensor, optional):
                Segment token indices to differentiate sequences in batch. Shape [batch_size, sequence_length].
            position_ids (torch.Tensor, optional):
                Indices of positions of each input token in the sequence. Shape [batch_size, sequence_length].
            head_mask (torch.Tensor, optional):
                Mask to nullify selected heads of the attention modules. Shape [num_heads] or [num_layers, num_heads].
            inputs_embeds (torch.Tensor, optional):
                Optionally provided embeddings instead of input_ids. Shape [batch_size, sequence_length, hidden_size].
            labels (torch.Tensor, optional):
                Labels for computing the sequence classification loss. Shape [batch_size].
            use_cache (bool, optional):
                Whether to use cache for the attention mechanism.
            output_attentions (bool, optional):
                Whether to output the attentions tensors.
            output_hidden_states (bool, optional):
                Whether to output the hidden states tensors.
            return_dict (bool, optional):
                Whether to return a :class:`~transformers.file_utils.SequenceClassifierOutputWithPast`.

        Returns:
            :class:`~transformers.file_utils.SequenceClassifierOutputWithPast`:
                Sequence classifier output consisting of loss, logits, past key values, attentions, and hidden states.
        """
        # Implementation of forward pass is handled by the parent class.
        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


@add_start_docstrings(
    """
    GPT Neo model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    GPT_NEO_START_DOCSTRING,
)
class GPTNeoForTokenClassification(GPTNeoPreTrainedModel):
    """
    GPTNeo model for token classification tasks.

    Inherits from `GPTNeoPreTrainedModel` and adds a linear classification layer for token classification.
    """

    def __init__(self, config):
        """
        Initializes the GPTNeoForTokenClassification model.

        Args:
            config (:class:`~transformers.GPTNeoConfig`):
                The configuration object that defines the model architecture.

        Attributes:
            num_labels (int):
                Number of labels for token classification.
            transformer (:class:`~transformers.GPTNeoModel`):
                The GPTNeoModel transformer instance.
            dropout (:class:`~torch.nn.Dropout`):
                Dropout layer for regularization.
            classifier (:class:`~torch.nn.Linear`):
                Linear layer for computing scores for each token label.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPTNeoModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(GPT_NEO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint="EleutherAI/gpt-neo-125m",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_loss=0.25,
    )
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
    ):
        """
        Performs forward pass of the GPTNeoForTokenClassification model.

        Args:
            input_ids (torch.Tensor, optional):
                The input token IDs. Shape [batch_size, sequence_length].
            past_key_values (Tuple[torch.FloatTensor], optional):
                Tuple of length 1 containing the cached key and value tensors from previous attention layers.
            attention_mask (torch.Tensor, optional):
                Mask to avoid performing attention on padding tokens. Shape [batch_size, sequence_length].
            token_type_ids (torch.Tensor, optional):
                Segment token indices to differentiate sequences in batch. Shape [batch_size, sequence_length].
            position_ids (torch.Tensor, optional):
                Indices of positions of each input token in the sequence. Shape [batch_size, sequence_length].
            head_mask (torch.Tensor, optional):
                Mask to nullify selected heads of the attention modules. Shape [num_heads] or [num_layers, num_heads].
            inputs_embeds (torch.Tensor, optional):
                Optionally provided embeddings instead of input_ids. Shape [batch_size, sequence_length, hidden_size].
            labels (torch.Tensor, optional):
                Labels for computing the token classification loss. Shape [batch_size, sequence_length].
            use_cache (bool, optional):
                Whether to use cache for the attention mechanism.
            output_attentions (bool, optional):
                Whether to output the attentions tensors.
            output_hidden_states (bool, optional):
                Whether to output the hidden states tensors.
            return_dict (bool, optional):
                Whether to return a :class:`~transformers.file_utils.TokenClassifierOutput`.

        Returns:
            :class:`~transformers.file_utils.TokenClassifierOutput`:
                Token classifier output consisting of loss, logits, attentions, and hidden states.
        """
        # Implementation of forward pass is handled by the parent class.
        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 为 None，则使用模型配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给 transformer 模型进行处理
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

        # 从 transformer 输出中获取隐藏状态，并应用 dropout
        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)

        # 将 dropout 后的隐藏状态输入分类器，得到 logits
        logits = self.classifier(hidden_states)

        # 初始化损失为 None
        loss = None

        # 如果提供了标签，则计算损失
        if labels is not None:
            # 将标签移动到与 logits 相同的设备上
            labels = labels.to(logits.device)
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 计算交叉熵损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不需要返回字典格式的输出，则返回一个元组
        if not return_dict:
            output = (logits,) + transformer_outputs[2:]  # 这里只返回 logits 和可能的其他输出
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典格式的输出，则使用 TokenClassifierOutput 封装结果
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
# 定义 GPT-Neo 用于问答任务的模型类，包括一个在隐藏状态之上的跨度分类头部用于像 SQuAD 这样的任务
@add_start_docstrings(
    """
    The GPT-Neo Model transformer with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    GPT_NEO_START_DOCSTRING,
)
class GPTNeoForQuestionAnswering(GPTNeoPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # 初始化模型的标签数量
        self.transformer = GPTNeoModel(config)  # 初始化 GPT-Neo 模型
        self.qa_outputs = nn.Linear(config.hidden_size, 2)  # 初始化线性层，用于生成跨度开始和结束的 logit
        
        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(GPT_NEO_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        real_checkpoint=_CHECKPOINT_FOR_DOC,
    )
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
        # 判断是否需要返回字典形式的输出，若不需要则使用配置中的设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给Transformer模型进行处理
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

        # 获取Transformer模型的输出中的序列输出
        sequence_output = outputs[0]

        # 将序列输出传递给QA输出层，得到起始和结束logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()  # 去掉多余的维度并保证连续存储
        end_logits = end_logits.squeeze(-1).contiguous()      # 去掉多余的维度并保证连续存储

        total_loss = None
        # 如果提供了起始和结束位置，计算损失
        if start_positions is not None and end_positions is not None:
            # 如果在多GPU环境下，增加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 忽略超出模型输入的起始/结束位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数，并计算起始和结束位置的损失
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # 如果不需要以字典形式返回结果，则以元组形式返回结果
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 如果需要以字典形式返回结果，则创建QuestionAnsweringModelOutput对象
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```