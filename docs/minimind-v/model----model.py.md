# `.\minimind-v\model\model.py`

```py
# 导入数学库
import math
# 导入用于处理结构化二进制数据的库
import struct
# 导入用于检查对象签名和源代码的库
import inspect
# 导入时间处理库
import time

# 从本地模块中导入 LMConfig 类
from .LMConfig import LMConfig
# 导入类型提示所需的类型
from typing import Any, Optional, Tuple
# 导入 numpy 库
import numpy as np
# 导入 torch 库
import torch
# 导入 PyTorch 中的常用神经网络功能模块
import torch.nn.functional as F
# 从 PyTorch 导入神经网络模块
from torch import nn
# 从 transformers 库导入预训练模型
from transformers import PreTrainedModel
# 从 transformers 库导入用于处理因果语言建模的输出格式
from transformers.modeling_outputs import CausalLMOutputWithPast


# 定义 RMSNorm 类，继承自 PyTorch 的 nn.Module
class RMSNorm(torch.nn.Module):
    # 初始化函数，接收两个参数：维度 dim 和一个小常数 eps 用于避免除零错误
    def __init__(self, dim: int, eps: float):
        super().__init__()  # 调用父类构造函数
        self.eps = eps  # 将 eps 存储为类的属性
        self.weight = nn.Parameter(torch.ones(dim))  # 创建一个可训练的权重参数，初始值为 1

    # 定义标准化函数
    def _norm(self, x):
        # 计算输入张量 x 的 L2 范数，进行归一化处理
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    # 定义前向传播函数
    def forward(self, x):
        # 对输入 x 进行标准化并转换为与输入 x 相同的数据类型
        output = self._norm(x.float()).type_as(x)
        # 返回标准化后的输出乘以权重
        return output * self.weight


# 预计算旋转位置编码
def precompute_pos_cis(dim: int, end: int, theta: float = 10000.0):
    # 计算位置编码的频率，基于 theta 和 dim 的关系
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 创建一个从 0 到 end 的时间步张量
    t = torch.arange(end, device=freqs.device)  # type: ignore
    # 计算频率的外积，得到每个时间步的频率值
    freqs = torch.outer(t, freqs).float()  # type: ignore
    # 将频率应用到极坐标形式，得到复数位置编码
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    # 返回预计算的复数位置编码
    return pos_cis


# 应用旋转位置编码
def apply_rotary_emb(xq, xk, pos_cis):
    # 定义统一位置编码形状的函数
    def unite_shape(pos_cis, x):
        ndim = x.ndim  # 获取 x 的维度
        assert 0 <= 1 < ndim  # 确保 x 的维度至少为 2
        assert pos_cis.shape == (x.shape[1], x.shape[-1])  # 检查位置编码形状与 x 匹配
        # 计算 pos_cis 应该扩展的形状
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)  # 将 pos_cis 变形为合适的形状

    # 将查询张量 xq 视为复数，并重新排列为最后一维包含实部和虚部
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    # 将键张量 xk 视为复数，并重新排列为最后一维包含实部和虚部
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # 调用 unite_shape 函数，调整位置编码的形状
    pos_cis = unite_shape(pos_cis, xq_)
    # 将查询张量与位置编码相乘，并恢复为实数张量，最后展平
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    # 将键张量与位置编码相乘，并恢复为实数张量，最后展平
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    # 返回与原始输入类型一致的查询和键输出
    return xq_out.type_as(xq), xk_out.type_as(xk)


# 定义重复 KV 张量的函数
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    # 获取输入张量 x 的形状信息
    bs, slen, n_kv_heads, head_dim = x.shape
    # 如果重复次数为 1，直接返回原始张量
    if n_rep == 1:
        return x
    # 否则，将张量的维度扩展到重复次数，并重塑形状
    return (
        x[:, :, :, None, :]  # 在第四维度插入新维度
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)  # 扩展张量
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)  # 重塑为新的形状
    )


# 定义 Attention 类，继承自 PyTorch 的 nn.Module
class Attention(nn.Module):
    # 初始化函数，接受一个LMConfig类型的参数args
    def __init__(self, args: LMConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 设置self.n_kv_heads为args.n_heads，如果args.n_kv_heads为None则使用args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # 断言确保args.n_heads能够整除self.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        # 设置self.n_local_heads为args.n_heads
        self.n_local_heads = args.n_heads
        # 设置self.n_local_kv_heads为self.n_kv_heads
        self.n_local_kv_heads = self.n_kv_heads
        # 计算self.n_rep为self.n_local_heads除以self.n_local_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 设置self.head_dim为args.dim除以args.n_heads
        self.head_dim = args.dim // args.n_heads
        # 初始化线性层wq，输入维度为args.dim，输出维度为args.n_heads * self.head_dim，无偏置
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        # 初始化线性层wk，输入维度为args.dim，输出维度为self.n_kv_heads * self.head_dim，无偏置
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # 初始化线性层wv，输入维度为args.dim，输出维度为self.n_kv_heads * self.head_dim，无偏置
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # 初始化线性层wo，输入维度为args.n_heads * self.head_dim，输出维度为args.dim，无偏置
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        # 初始化self.k_cache和self.v_cache为None
        self.k_cache, self.v_cache = None, None
        # 初始化attn_dropout为Dropout层，丢弃概率为args.dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        # 初始化resid_dropout为Dropout层，丢弃概率为args.dropout
        self.resid_dropout = nn.Dropout(args.dropout)
        # 设置self.dropout为args.dropout
        self.dropout = args.dropout
        # 检查是否支持Flash Attention，并且args.flash_attn为True
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

        # 打印警告信息，提示使用慢速的注意力机制，Flash Attention需要PyTorch版本大于等于2.0
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        # 创建一个填充值为负无穷的mask张量，形状为(1, 1, args.max_seq_len, args.max_seq_len)
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        # 生成上三角矩阵mask，对角线以上的元素为负无穷
        mask = torch.triu(mask, diagonal=1)
        # 将mask注册为模型的缓冲区，不会被优化器更新
        self.register_buffer("mask", mask, persistent=False)
    # 定义前向传播函数，接受输入张量x，位置编码pos_cis，和是否使用kv_cache的标志
    def forward(self, x: torch.Tensor, pos_cis: torch.Tensor, kv_cache=False):
        # 获取输入x的batch size（bsz）、序列长度（seqlen）和通道数（不使用的通道数用'_'占位）
        bsz, seqlen, _ = x.shape
    
        # 通过权重矩阵wq、wk、wv将输入x映射为查询（xq）、键（xk）、值（xv）
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    
        # 调整xq、xk、xv的形状，使其适配多头注意力的结构
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
    
        # 应用旋转编码（apply_rotary_emb），对查询和键进行位置编码
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)
    
        # 如果启用kv_cache，并且处于评估模式下（self.eval()），且序列长度为1，且缓存的k和v不为None
        if kv_cache and self.eval():
            # 如果当前输入序列长度为1，且kv_cache有效，合并缓存中的k和v与当前计算的k和v
            if seqlen == 1 and all(cache is not None for cache in (self.k_cache, self.v_cache)):
                # 将之前缓存的k（self.k_cache）和当前计算的k（xk）拼接在一起
                xk = torch.cat((self.k_cache, xk), dim=1)
                # 将之前缓存的v（self.v_cache）和当前计算的v（xv）拼接在一起
                xv = torch.cat((self.v_cache, xv), dim=1)
            # 更新缓存k和v
            self.k_cache, self.v_cache = xk, xv
    
        # 重复kv（键和值）以适配多头注意力的需要
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
    
        # 转置查询、键和值，使得注意力机制符合要求
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
    
        # 如果启用flash attention并且序列长度不为1
        if self.flash and seqlen != 1:
            # 使用flash attention计算输出
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None,
                                                                      dropout_p=self.dropout if self.training else 0.0,
                                                                      is_causal=True)
        else:
            # 否则，使用标准的点积注意力计算
            # 计算查询与键的点积，并除以头的维度平方根以保持稳定
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            # 将注意力分数与掩码加和，避免某些位置的注意力过大
            scores = scores + self.mask[:, :, :seqlen, :seqlen]  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            # 使用softmax对分数进行归一化，得到概率分布
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            # 应用dropout防止过拟合
            scores = self.attn_dropout(scores)
            # 计算加权求和得到输出
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)
    
        # 将输出的形状调整为(batch_size, seq_len, head_dim * n_heads)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    
        # 通过线性变换wo映射输出
        output = self.wo(output)
        # 应用残差连接后的dropout
        output = self.resid_dropout(output)
        # 返回最终输出
        return output
class FeedForward(nn.Module):
    # 定义前馈神经网络模型
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # 如果隐藏层维度未指定，则设置为输入维度的4倍
        if hidden_dim is None:
            hidden_dim = 4 * dim
            # 将隐藏层维度调整为输入维度的2/3
            hidden_dim = int(2 * hidden_dim / 3)
            # 将隐藏层维度调整为最接近的 multiple_of 的倍数
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        # 定义线性层 w1
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义线性层 w2
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义线性层 w3
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义 dropout 层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 前向传播函数
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MoEGate(nn.Module):
    # 定义 Mixture of Experts 门控模型
    def __init__(self, config: LMConfig):
        super().__init__()
        # 初始化模型配置
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.dim
        # 定义权重参数
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        # 重置参数初始化
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 使用 kaiming_uniform 初始化权重参数
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    # 前向传播函数，计算模型输出的预测值及辅助损失
    def forward(self, hidden_states):
        # 获取输入张量的批大小、序列长度和隐藏层维度
        bsz, seq_len, h = hidden_states.shape
    
        # 将 hidden_states 重新调整形状，将其展平为 (bsz * seq_len, h)
        hidden_states = hidden_states.view(-1, h)
        
        # 通过线性变换计算 logits，不使用偏置项
        logits = F.linear(hidden_states, self.weight, None)
        
        # 如果 scoring_func 是 softmax，计算 logits 的 softmax 分数
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            # 如果 scoring_func 不是 softmax，抛出异常
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
    
        # 获取 scores 中每行的 top-k 权重和对应的索引
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
    
        # 如果 top_k > 1 且需要对 top-k 概率进行归一化，计算归一化后的权重
        if self.top_k > 1 and self.norm_topk_prob:
            # 计算每行 topk_weight 的归一化分母，避免除零错误
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            # 归一化 topk_weight
            topk_weight = topk_weight / denominator
    
        # 如果处于训练模式并且 alpha 大于 0，计算辅助损失
        if self.training and self.alpha > 0.0:
            # 用于辅助损失的 scores 和 topk 索引
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            
            # 如果启用 seq_aux，计算基于序列的辅助损失
            if self.seq_aux:
                # 将 scores_for_aux 重新调整为 (bsz, seq_len, num_classes)
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                # 初始化辅助损失的交叉熵张量
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                # 将 topk 索引对应的位置累加 1，用于计算交叉熵
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                # 计算辅助损失并乘以 alpha
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # 如果未启用 seq_aux，计算不基于序列的辅助损失
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                # 计算每个专家的平均交叉熵
                ce = mask_ce.float().mean(0)
                # 计算 Pi 和 fi，用于计算损失
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                # 计算辅助损失并乘以 alpha
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            # 如果不在训练模式，或者 alpha 为 0，不计算辅助损失
            aux_loss = None
        
        # 返回 top-k 索引、top-k 权重以及辅助损失
        return topk_idx, topk_weight, aux_loss
# 定义一个 MOEFeedForward 类，继承自 nn.Module，表示一个基于 Mixture of Experts (MoE) 的前馈神经网络层
class MOEFeedForward(nn.Module):
    # 初始化方法，传入配置参数 config，设置模型的各个超参数
    def __init__(self, config: LMConfig):
        # 调用父类 nn.Module 的初始化方法
        super().__init__()
        # 保存配置对象，供后续使用
        self.config = config
        # 初始化一个专家列表，每个专家是一个 FeedForward 网络，数量由 config.n_routed_experts 决定
        self.experts = nn.ModuleList([
            FeedForward(
                dim=config.dim,  # 输入的维度
                hidden_dim=config.hidden_dim,  # 隐藏层的维度
                multiple_of=config.multiple_of,  # 隐藏层维度的倍数
                dropout=config.dropout,  # dropout 概率
            )
            for _ in range(config.n_routed_experts)  # 创建 config.n_routed_experts 个专家
        ])

        # 初始化 MoEGate 对象，负责专家选择的门控机制
        self.gate = MoEGate(config)
        # 如果配置中有共享专家的数量，则创建一个共享的 FeedForward 网络
        if config.n_shared_experts is not None:
            self.shared_experts = FeedForward(
                dim=config.dim,
                hidden_dim=config.hidden_dim,
                multiple_of=config.multiple_of,
                dropout=config.dropout,
            )

    # 定义前向传播过程
    def forward(self, x):
        # 保留输入 x，用于后续加上共享专家输出时做残差连接
        identity = x
        # 获取输入的原始形状，以便最后恢复形状
        orig_shape = x.shape
        # 获取批次大小、序列长度和特征维度
        bsz, seq_len, _ = x.shape

        # 使用门控机制选择专家，并获得专家的索引、权重以及辅助损失
        topk_idx, topk_weight, aux_loss = self.gate(x)

        # 将输入展平为二维张量，以便于后续操作
        x = x.view(-1, x.shape[-1])
        # 将 topk_idx 展平
        flat_topk_idx = topk_idx.view(-1)

        # 如果是训练模式
        if self.training:
            # 在训练模式下，重复输入数据 num_experts_per_tok 次，进行专家选择
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            # 创建一个空的张量 y，用于存放每个专家的输出
            y = torch.empty_like(x, dtype=torch.float16)
            # 遍历所有专家，依据 topk_idx 将输入分配给相应专家进行计算
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i])
            # 将专家输出按 topk_weight 加权求和，并恢复原始的形状
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理模式下，只选择最优专家进行计算
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        # 如果有共享专家，则将共享专家的输出加到最终结果中
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)

        # 返回最终输出
        return y

    # 定义一个无梯度的推理方法，用于推理阶段的计算
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # 初始化一个与输入相同形状的缓存，用于存放专家的输出
        expert_cache = torch.zeros_like(x)
        # 获取专家索引的排序，表示每个 token 对应的专家
        idxs = flat_expert_indices.argsort()
        # 计算每个专家处理的 token 数量，并进行累加
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 根据专家索引获取每个 token 的专家分配
        token_idxs = idxs // self.config.num_experts_per_tok
        # 遍历每个专家
        for i, end_idx in enumerate(tokens_per_expert):
            # 获取当前专家处理的 token 范围的起始和结束索引
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            # 如果该专家没有处理任何 token，则跳过
            if start_idx == end_idx:
                continue
            # 获取当前专家对象
            expert = self.experts[i]
            # 获取当前专家需要处理的 token 索引
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 获取当前专家对应的 token 输入
            expert_tokens = x[exp_token_idx]
            # 计算专家的输出
            expert_out = expert(expert_tokens)
            # 将专家输出按照对应的权重进行加权
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 使用 scatter_add_ 操作，将加权后的输出结果加到缓存中
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        # 返回所有专家的加权输出结果
        return expert_cache
    # 初始化该层，设置其参数
    def __init__(self, layer_id: int, args: LMConfig):
        # 调用父类构造函数
        super().__init__()
        # 设置头的数量
        self.n_heads = args.n_heads
        # 设置每个头的维度
        self.dim = args.dim
        # 每个头的维度由总维度除以头数得到
        self.head_dim = args.dim // args.n_heads
        # 初始化注意力机制对象
        self.attention = Attention(args)
    
        # 设置层的 ID
        self.layer_id = layer_id
        # 初始化注意力层的归一化对象（使用 RMSNorm）
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 初始化前馈网络的归一化对象（使用 RMSNorm）
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    
        # 如果使用 MoE（Mixture of Experts），则初始化 MoE 前馈网络
        if args.use_moe:
            self.feed_forward = MOEFeedForward(args)
        else:
            # 否则初始化常规的前馈网络
            self.feed_forward = FeedForward(
                dim=args.dim,
                hidden_dim=args.hidden_dim,
                multiple_of=args.multiple_of,
                dropout=args.dropout,
            )
    
    # 前向传播函数，计算输入的输出
    def forward(self, x, pos_cis, kv_cache=False):
        # 计算注意力的输出，并进行归一化和相加
        h = x + self.attention(self.attention_norm(x), pos_cis, kv_cache)
        # 计算前馈网络的输出，并进行归一化和相加
        out = h + self.feed_forward(self.ffn_norm(h))
        # 返回最终的输出
        return out
# 定义 VisionProj 类，继承自 nn.Module，用于图像特征投影
class VisionProj(nn.Module):
    # 初始化函数，设置视觉输出维度、语言模型维度和图像 ID 列表
    def __init__(self, vision_out_dim=768, lm_dim=512, image_ids=[1, 2, 3, 4]):
        super().__init__()  # 调用父类初始化方法
        self.vision_out_dim = vision_out_dim  # 设置视觉输出维度
        self.lm_dim = lm_dim  # 设置语言模型维度
        self.image_ids = image_ids  # 设置图像 ID 列表
        # 定义视觉特征投影层，将视觉特征从 vision_out_dim 维度转换为 lm_dim 维度
        self.vision_proj = nn.Sequential(
            nn.Linear(self.vision_out_dim, self.lm_dim),  # 使用全连接层进行维度转换
        )

    # 前向传播函数，接收图像编码器的输出并进行投影
    def forward(self, image_encoders):
        # 将图像编码器的输出传入 vision_proj 层，得到投影后的特征
        vision_proj = self.vision_proj(image_encoders)
        return vision_proj  # 返回投影后的视觉特征


# 定义 Transformer 类，继承自 PreTrainedModel
class Transformer(PreTrainedModel):
    # 指定模型配置类
    config_class = LMConfig
    # 初始化时用于存储最后的损失值
    last_loss: Optional[torch.Tensor]

    # 初始化函数，接收一个 LMConfig 类型的配置对象
    def __init__(self, params: LMConfig = None):
        super().__init__(params)  # 调用父类初始化方法
        if not params:
            params = LMConfig()  # 如果没有提供配置对象，则使用默认配置
        self.params = params  # 设置模型参数
        self.vocab_size = params.vocab_size  # 获取词汇表大小
        self.n_layers = params.n_layers  # 获取 Transformer 层数
        # 图像的特殊占位符，对应每张图切分成 M 个 token，和 get_img_process 中的数量对应
        self.image_ids = params.image_ids

        # 定义词嵌入层，将输入的 token 索引映射为对应的词向量
        self.tok_embeddings = nn.Embedding(self.vocab_size, params.dim)
        # 定义 Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(params.dropout)
        # 定义 Transformer 的各层模块
        self.layers = torch.nn.ModuleList()
        # 为每一层 Transformer 添加一个 TransformerBlock 模块
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        # 定义归一化层
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        # 定义输出层，将最后的隐藏状态映射到词汇表大小
        self.output = nn.Linear(params.dim, self.vocab_size, bias=False)
        # 将输出层的权重与词嵌入层的权重共享
        self.tok_embeddings.weight = self.output.weight
        # 预计算位置编码矩阵
        pos_cis = precompute_pos_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        # 注册位置编码矩阵为模型的 buffer，不会参与梯度更新
        self.register_buffer("pos_cis", pos_cis, persistent=False)

        # 初始化权重
        self.apply(self._init_weights)

        # 初始化特定参数的权重
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                # 对于特定层的权重进行正态分布初始化
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_layers))

        # 初始化损失值为 None
        self.last_loss = None
        # 定义输出结构，用于存储 CausalLM 的输出
        self.OUT = CausalLMOutputWithPast()
        # 定义不需要拆分的模块列表
        self._no_split_modules = [name for name, _ in self.named_modules()]

        # 定义 VisionProj 对象，用于图像特征的投影
        self.vision_proj = VisionProj(768, params.dim, self.image_ids)

    # 权重初始化函数
    def _init_weights(self, module):
        # 对于全连接层，进行正态分布初始化
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # 如果有偏置项，将偏置项初始化为 0
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # 对于嵌入层，进行正态分布初始化
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # VLM
    # 定义一个方法，计算并返回图像相关的视觉投影
        def count_vision_proj(self, tokens, h, image_encoders=None, seqlen=200):
            # 定义一个内部函数，查找token中<image>片段的索引，为了后续替换操作做准备
            def find_indices(tokens, image_ids):
                # 将给定的image_ids转换为Tensor并放置在tokens的设备上
                image_ids_tensor = torch.tensor(image_ids).to(tokens.device)
                # 获取image_ids的长度
                len_image_ids = len(image_ids)
                # 如果image_ids的长度大于tokens的长度，则无法匹配，直接返回None
                if len_image_ids > tokens.size(1):
                    return None
    
                # 使用unfold方法在tokens上创建一个视图，模拟滑动窗口，方便逐个窗口检查匹配情况
                tokens_view = tokens.unfold(1, len_image_ids, 1)  # 在第二维度创建滑动窗口
                # 检查每个滑动窗口是否与image_ids_tensor相等
                matches = (tokens_view == image_ids_tensor).all(dim=2)  # 对窗口中的每一行进行比较
    
                # 创建一个字典，用于存储每个batch中匹配的索引范围
                indices = {}
                for batch_idx in range(tokens.size(0)):
                    # 获取匹配的索引位置
                    match_indices = matches[batch_idx].nonzero(as_tuple=True)[0]  # 获取非零（匹配）索引
                    if match_indices.numel() > 0:  # 如果有匹配的索引
                        # 将匹配的起始索引和结束索引组成一个元组并存入字典
                        indices[batch_idx] = [(idx.item(), idx.item() + len_image_ids - 1) for idx in match_indices]
                # 如果找到了匹配的索引，则返回，否则返回None
                return indices if indices else None
    
            # 调用find_indices函数，得到图像在tokens中对应的索引
            image_indices = find_indices(tokens, self.image_ids)  # 字典形式存储索引
    
            # 如果传入了图像编码器
            if image_encoders is not None:
                # 使用图像编码器生成视觉投影特征
                vision_proj = self.vision_proj(image_encoders)
                # 如果vision_proj是3维张量，则通过unsqueeze增加一个维度
                vision_proj = vision_proj.unsqueeze(1) if len(vision_proj.shape) == 3 else vision_proj
                # 如果找到了图像在tokens中的索引
                if image_indices is not None:
                    # 创建一个空列表用于存储拼接后的结果
                    new_h = []
                    for i in range(h.size(0)):
                        # i为当前batch的索引
                        img_idx = 0
                        # 如果当前batch有图像索引
                        if i in image_indices:  # 直接从字典中获取
                            for start_idx, end_idx in image_indices[i]:
                                # 将视觉特征插入到原始token序列的相应位置
                                before = h[i][:start_idx, :]
                                after = h[i][end_idx + 1:, :]
                                # 将前后部分与视觉特征拼接，并限制最大长度为seqlen
                                h[i] = torch.cat((before, vision_proj[i][img_idx], after), dim=0)[:seqlen]
                                # 更新图像索引
                                img_idx += 1
                        # 将当前batch的结果添加到new_h列表
                        new_h.append(h[i])
                    # 将拼接后的所有batch堆叠成一个新的Tensor
                    new_h = torch.stack(new_h, dim=0)  # torch.Size([32, 511, 512])
                    # 返回处理后的结果
                    return new_h
    
            # 如果没有图像编码器或没有匹配的图像索引，返回原始输入
            return h
    # 定义前向传播函数，接受输入 tokens 和 targets，以及其他可选的参数
    def forward(self, tokens: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None,
                    kv_cache=False, image_encoders=None, **keyargs):
        # 初始化 current_idx 为 0
        current_idx = 0
    
        # 如果 keyargs 中有 'input_ids'，将其赋值给 tokens
        if 'input_ids' in keyargs:
            tokens = keyargs['input_ids']
        
        # 如果 keyargs 中有 'attention_mask'，将其赋值给 targets
        if 'attention_mask' in keyargs:
            targets = keyargs['attention_mask']
        
        # 如果 keyargs 中有 'current_idx'，将其转为整数并赋值给 current_idx
        if 'current_idx' in keyargs:
            current_idx = int(keyargs['current_idx'])
    
        # 获取 tokens 的 batch_size (_bsz) 和序列长度 (seqlen)
        _bsz, seqlen = tokens.shape
        
        # 计算语言模型嵌入的 token 表示
        h = self.tok_embeddings(tokens)
        
        # 计算视觉模型的 token 表示，并进行融合
        h = self.count_vision_proj(tokens=tokens, h=h, image_encoders=image_encoders, seqlen=seqlen)
        
        # 对嵌入结果进行 dropout 操作，防止过拟合
        h = self.dropout(h)
    
        # 获取当前序列位置的位置信息
        pos_cis = self.pos_cis[current_idx:current_idx + seqlen]
        
        # 遍历每一层，逐层处理输入数据
        for idx, layer in enumerate(self.layers):
            h = layer(h, pos_cis, kv_cache)
    
        # 对最终结果进行归一化处理
        h = self.norm(h)
    
        # 如果 targets 不为 None，计算损失并返回 logits
        if targets is not None:
            logits = self.output(h)
            # 使用交叉熵损失函数，计算预测值与目标值之间的损失
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0,
                                             reduction='none')
        else:
            # 如果没有目标值，获取最后一个时间步的预测 logits
            logits = self.output(h[:, [-1], :])
            self.last_loss = None
    
        # 将 logits 和 last_loss 存入 OUT 字典
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('loss', self.last_loss)
        
        # 返回 OUT 字典
        return self.OUT
    
    # 定义生成文本的函数，使用给定的输入索引生成最大长度的序列
    @torch.inference_mode()
    def generate(self, idx, eos, max_new_tokens, temperature=0.7, top_k=8, stream=True, rp=1., kv_cache=True,
                     image_encoders=None):
        # rp: 重复惩罚系数
        index = idx.shape[1]
        
        # 初始化推理标志
        init_inference = True
    
        # 循环直到生成所需长度的序列
        while idx.shape[1] < max_new_tokens - 1:
            # 如果是第一次推理或者不使用 kv_cache，则进行完整推理
            if init_inference or not kv_cache:
                inference_res, init_inference = self(idx, kv_cache=kv_cache, image_encoders=image_encoders), False
            else:
                # 如果有缓存，则只传递最后一个 token 进行推理
                inference_res = self(idx[:, -1:], kv_cache=kv_cache, current_idx=idx.shape[1] - 1)
    
            # 获取推理结果的 logits
            logits = inference_res.logits
            # 只保留最后一个时间步的 logits
            logits = logits[:, -1, :]
    
            # 对每个已生成的 token 进行重复惩罚处理
            for token in set(idx.tolist()[0]):
                logits[:, token] /= rp
    
            # 如果 temperature 为 0，则选择最大概率的 token
            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # 否则，根据温度调整 logits
                logits = logits / temperature
                # 如果 top_k 不为 None，则对 logits 进行截断
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
    
                # 计算概率分布
                probs = F.softmax(logits, dim=-1)
                # 从概率分布中采样下一个 token
                idx_next = torch.multinomial(probs, num_samples=1, generator=None)
    
            # 如果生成的 token 是 eos，结束生成过程
            if idx_next == eos:
                break
    
            # 将新生成的 token 拼接到当前序列中
            idx = torch.cat((idx, idx_next), dim=1)
            
            # 如果启用了流式生成，则逐步返回生成的部分序列
            if stream:
                yield idx[:, index:]
    
        # 如果不使用流式生成，则返回完整的生成序列
        if not stream:
            yield idx[:, index:]
    
    # 结束生成函数的定义
    @torch.inference_mode()
    # 定义一个方法用于评估答案
    def eval_answer(self, idx):
        # 如果输入的序列长度超过最大序列长度，则截取最后的部分作为输入
        idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
        # 调用模型进行推理
        inference_res = self(idx_cond)
        # 获取推理结果中的logits
        logits = inference_res.logits
        # 只取最后一个时间步的logits
        logits = logits[:, -1, :]
        # 返回logits作为评估结果
        return logits
```