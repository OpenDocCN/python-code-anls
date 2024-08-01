# `.\DB-GPT-src\dbgpt\model\llm\monkey_patch.py`

```py
#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import math  # 导入数学库，用于数学计算
from typing import Optional, Tuple  # 导入类型提示相关模块

import torch  # 导入PyTorch库
import transformers  # 导入transformers库
from torch import nn  # 从PyTorch库中导入神经网络模块


def rotate_half(x):
    """对输入张量的后一半维度进行旋转。"""
    x1 = x[..., : x.shape[-1] // 2].clone()  # 拷贝张量的前半部分
    x2 = x[..., x.shape[-1] // 2 :].clone()  # 拷贝张量的后半部分
    return torch.cat((-x2, x1), dim=-1)  # 将旋转后的两部分张量拼接在一起


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """应用旋转位置编码到查询和键上。"""
    gather_indices = position_ids[:, None, :, None]  # 构造用于索引的位置张量
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])  # 扩展索引张量的维度
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)  # 根据索引获取对应的余弦值
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)  # 根据索引获取对应的正弦值
    q_embed = (q * cos) + (rotate_half(q) * sin)  # 应用旋转位置编码到查询上
    k_embed = (k * cos) + (rotate_half(k) * sin)  # 应用旋转位置编码到键上
    return q_embed, k_embed  # 返回应用旋转位置编码后的查询和键


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """模型的前向传播函数，用于执行自注意力机制。"""
    bsz, q_len, _ = hidden_states.size()  # 获取隐藏状态张量的维度信息

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )  # 查询状态张量的变换和重塑操作

    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )  # 键状态张量的变换和重塑操作

    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )  # 值状态张量的变换和重塑操作

    kv_seq_len = key_states.shape[-2]  # 获取键状态张量的序列长度信息
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]  # 如果存在过去的键值对，则更新序列长度信息

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)  # 获取旋转位置编码的余弦和正弦值

    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )  # 应用旋转位置编码到查询和键上

    if past_key_value is not None:
        # 如果存在过去的键值对，则将当前键和值拼接起来
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None  # 更新过去的键值对

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )  # 计算注意力权重

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )  # 检查注意力权重的维度是否符合预期
    # 如果注意力掩码不为None，则进行以下处理
    if attention_mask is not None:
        # 检查注意力掩码的尺寸是否符合预期
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            # 如果不符合预期，抛出数值错误异常并显示详细信息
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        
        # 将注意力权重和注意力掩码相加
        attn_weights = attn_weights + attention_mask
        
        # 将注意力权重限制到一个极小的值上限
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
        )

    # 将注意力权重转换为float32类型，并且调整其设备类型以匹配query_states的类型
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    
    # 计算注意力输出，使用注意力权重与value_states的乘积
    attn_output = torch.matmul(attn_weights, value_states)

    # 检查注意力输出的尺寸是否符合预期
    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        # 如果不符合预期，抛出数值错误异常并显示详细信息
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    # 调整注意力输出的维度顺序
    attn_output = attn_output.transpose(1, 2)
    # 重新整形注意力输出的尺寸
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    # 通过o_proj层处理注意力输出
    attn_output = self.o_proj(attn_output)

    # 如果不需要输出注意力权重，则将attn_weights置为None
    if not output_attentions:
        attn_weights = None

    # 返回注意力输出、注意力权重以及过去的键-值对
    return attn_output, attn_weights, past_key_value
# 替换 LLAMA 模型中的注意力机制前向传播函数，以避免 MPS 后端中的错误，该操作不使用就地操作。
def replace_llama_attn_with_non_inplace_operations():
    """避免在 MPS 后端中使用就地操作导致的错误，替换 LLAMA 模型中的注意力机制前向传播函数。"""
    # 将 LLAMA 模型中的 LlamaAttention 类的 forward 方法替换为当前定义的 forward 函数
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
```