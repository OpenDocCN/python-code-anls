# `.\models\flaubert\modeling_flaubert.py`

```
# 设置文件编码为UTF-8
# 版权声明
# 授权许可，通过许可使用该文件
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 上获取许可副本
# 根据适用法律或书面同意，分发的软件是基于"AS IS"的基础上分发的，
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 有关限制和准许语言以及语言的详细信息，请参阅许可
""" PyTorch Flaubert model, based on XLM."""

# 导入所需的库和模块
import itertools
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
# 导入Hugging Face的相关模块
from ...activations import gelu
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel, SequenceSummary, SQuADHead
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_flaubert import FlaubertConfig
# 获取日志对象
logger = logging.get_logger(__name__)

# 文档示例
_CHECKPOINT_FOR_DOC = "flaubert/flaubert_base_cased"
_CONFIG_FOR_DOC = "FlaubertConfig"

# Flaubert预训练模型的存档列表
FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "flaubert/flaubert_small_cased",
    "flaubert/flaubert_base_uncased",
    "flaubert/flaubert_base_cased",
    "flaubert/flaubert_large_cased",
    # 查看所有Flaubert模型，请访问 https://huggingface.co/models?filter=flaubert
]

# 从transformers.models.xlm.modeling_xlm.create_sinusoidal_embeddings复制过来
# 创建正弦嵌入
def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False

# 从transformers.models.xlm.modeling_xlm.get_masks复制过来
# 获取遮罩
def get_masks(slen, lengths, causal, padding_mask=None):
    """
    生成隐藏状态遮罩，并可选地生成注意力遮罩。
    """
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    if padding_mask is not None:
        mask = padding_mask
    else:
        assert lengths.max().item() <= slen
        mask = alen < lengths[:, None]

    # 注意力遮罩与普通遮罩相同，或是三角形下三角形的注意力（因果关系）
    bs = lengths.size(0)
    # 如果是因果关系，则创建一个注意力遮罩，将长度限制为alen[None, None, :]，并将其重复到匹配bs, slen, 1的形状
    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    # 如果不是因果关系，则使用给定的注意力遮罩
    else:
        attn_mask = mask

    # 对注意力遮罩进行健壮性检查，确保尺寸匹配
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    # 返回结果注意力遮罩
    return mask, attn_mask
# 定义一个多头注意力机制的类，该类是从transformers.models.xlm.modeling_xlm.MultiHeadAttention中复制而来
class MultiHeadAttention(nn.Module):
    # 静态变量，用于分配每个实例的唯一ID
    NEW_ID = itertools.count()

    # 初始化方法，设置注意力头数、维度和配置
    def __init__(self, n_heads, dim, config):
        super().__init__()
        # 分配实例的唯一ID
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        # 设置维度
        self.dim = dim
        # 设置注意力头数
        self.n_heads = n_heads
        # 设置注意力的dropout概率
        self.dropout = config.attention_dropout
        # 确保维度能够整除注意力头数
        assert self.dim % self.n_heads == 0

        # 定义线性变换层，用于计算Q、K、V以及输出
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        # 用于存储被剪枝的注意力头的集合
        self.pruned_heads = set()

    # 方法：剪枝指定的注意力头
    def prune_heads(self, heads):
        # 计算每个注意力头的大小
        attention_head_size = self.dim // self.n_heads
        # 如果没有需要剪枝的头，则直接返回
        if len(heads) == 0:
            return
        # 找到需要剪枝的头以及对应的索引
        heads, index = find_pruneable_heads_and_indices(heads, self.n_heads, attention_head_size, self.pruned_heads)
        # 剪枝线性层
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        # 更新超参数
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)
    def forward(self, input, mask, kv=None, cache=None, head_mask=None, output_attentions=False):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)  # 输入是 (bs, qlen, dim)，bs为batch size，qlen为query长度，dim为维度
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)  # Mask 是 (bs, klen) (非因果) 或 (bs, klen, klen)
        bs, qlen, dim = input.size()  # 获取输入的维度信息
        if kv is None:  # 如果kv为空
            klen = qlen if cache is None else cache["slen"] + qlen  # klen为qlen如果cache为空，否则为cache["slen"] + qlen
        else:
            klen = kv.size(1)  # 否则klen为kv的第一维度大小
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'  # 断言维度相符
        n_heads = self.n_heads  # 获取头数
        dim_per_head = self.dim // n_heads  # 每个头的维度
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)  # 根据mask的维度构造mask_reshape

        def shape(x):  # 定义形状函数
            """projection"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)  # 对x进行形状变换

        def unshape(x):  # 定义反向形状函数
            """compute context"""
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)  # 对x进行反向形状变换

        q = shape(self.q_lin(input))  # (bs, n_heads, qlen, dim_per_head)  # 对输入进行线性变换并reshape成q
        if kv is None:  # 如果kv为空
            k = shape(self.k_lin(input))  # (bs, n_heads, qlen, dim_per_head)  # 对输入进行线性变换并reshape成k
            v = shape(self.v_lin(input))  # (bs, n_heads, qlen, dim_per_head)  # 对输入进行线性变换并reshape成v
        elif cache is None or self.layer_id not in cache:  # 否则如果cache为空或者layer_id不在cache中
            k = v = kv  # 则k和v为kv
            k = shape(self.k_lin(k))  # (bs, n_heads, qlen, dim_per_head)  # 对k进行线性变换并reshape成k
            v = shape(self.v_lin(v))  # (bs, n_heads, qlen, dim_per_head)  # 对v进行线性变换并reshape成v

        if cache is not None:  # 如果cache不为空
            if self.layer_id in cache:  # 如果layer_id在cache中
                if kv is None:  # 如果kv为空
                    k_, v_ = cache[self.layer_id]  # 获取缓存中的k和v
                    k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)  # 拼接k
                    v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)  # 拼接v
                else:
                    k, v = cache[self.layer_id]  # 否则k和v为缓存中的k和v
            cache[self.layer_id] = (k, v)  # 更新缓存

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, qlen, dim_per_head)  # 对q进行缩放
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, qlen, klen)  # 计算注意力分数
        mask = (mask == 0).view(mask_reshape).expand_as(scores)  # (bs, n_heads, qlen, klen)  # 根据mask构造mask张量
        scores.masked_fill_(mask, torch.finfo(scores.dtype).min)  # (bs, n_heads, qlen, klen)  # 根据mask对scores进行填充

        weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)  # (bs, n_heads, qlen, klen)  # 计算注意力权重
        weights = nn.functional.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)  # 对权重进行dropout

        # Mask heads if we want to  # 如果需要，对头进行mask
        if head_mask is not None:  # 如果head_mask不为空
            weights = weights * head_mask  # 对权重进行mask

        context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)  # 计算context
        context = unshape(context)  # (bs, qlen, dim)  # 对context进行反向形状变换

        outputs = (self.out_lin(context),)  # 构造输出
        if output_attentions:  # 如果需要输出注意力权重
            outputs = outputs + (weights,)  # 则将注意力权重加入输出
        return outputs  # 返回输出
# 定义一个名为 TransformerFFN 的类，继承自 nn.Module
class TransformerFFN(nn.Module):
    # 初始化函数，设置输入维度、隐藏层维度、输出维度和配置参数
    def __init__(self, in_dim, dim_hidden, out_dim, config):
        super().__init__()
        # 设置 dropout 概率
        self.dropout = config.dropout
        # 定义线性变换层1，输入维度为 in_dim，输出维度为 dim_hidden
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        # 定义线性变换层2，输入维度为 dim_hidden，输出维度为 out_dim
        self.lin2 = nn.Linear(dim_hidden, out_dim)
        # 设置激活函数为 GELU 或 ReLU
        self.act = gelu if config.gelu_activation else nn.functional.relu
        # 设定前馈网络的分块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置序列长度维度
        self.seq_len_dim = 1

    # 前向传播函数，对输入进行分块并应用前馈神经网络
    def forward(self, input):
        return apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, input)

    # 前馈神经网络分块函数，包括线性变换、激活函数、线性变换和 dropout 操作
    def ff_chunk(self, input):
        x = self.lin1(input)
        x = self.act(x)
        x = self.lin2(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x


# Flaubert 模型的起始文档字符串
FLAUBERT_START_DOCSTRING = r"""
...

FLAUBERT_START_DOCSTRING = r"""
...

# Flaubert 输入参数文档字符串
FLAUBERT_INPUTS_DOCSTRING = r"""
...

# 定义一个名为 FlaubertPredLayer 的类，继承自 nn.Module
@add_start_docstrings(
    "The bare Flaubert Model transformer outputting raw hidden-states without any specific head on top.",
    FLAUBERT_START_DOCSTRING,
)
class FlaubertPredLayer(nn.Module):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """

    # 初始化函数，设置预测层的配置参数
    def __init__(self, config):
        super().__init__()
        # 是否使用 adaptive softmax
        self.asm = config.asm
        # 词汇量大小
        self.n_words = config.n_words
        # 用于填充的索引
        self.pad_index = config.pad_index
        dim = config.emb_dim

        # 根据是否使用 adaptive softmax，选择线性变换层或者自适应 softmax 损失函数
        if config.asm is False:
            self.proj = nn.Linear(dim, config.n_words, bias=True)
        else:
            self.proj = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=dim,
                n_classes=config.n_words,
                cutoffs=config.asm_cutoffs,
                div_value=config.asm_div_value,
                head_bias=True,  # 默认为 False
            )
    # 计算损失和（可选）分数
    def forward(self, x, y=None):
        # 初始化输出元组
        outputs = ()
        # 如果不使用自注意力机制
        if self.asm is False:
            # 计算分数
            scores = self.proj(x)
            # 将分数添加到输出元组中
            outputs = (scores,) + outputs
            # 如果有目标值
            if y is not None:
                # 计算交叉熵损失并添加到输出元组中
                loss = nn.functional.cross_entropy(scores.view(-1, self.n_words), y.view(-1), reduction="mean")
                outputs = (loss,) + outputs
        # 如果使用自注意力机制
        else:
            # 计算得分的对数概率
            scores = self.proj.log_prob(x)
            # 将得分添加到输出元组中
            outputs = (scores,) + outputs
            # 如果有目标值
            if y is not None:
                # 计算损失并添加到输出元组中
                _, loss = self.proj(x, y)
                outputs = (loss,) + outputs

        # 返回结果元组
        return outputs
# 定义一个名为 FlaubertPreTrainedModel 的类，用于处理权重初始化以及下载和加载预训练模型
class FlaubertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 FlaubertConfig 作为配置类
    config_class = FlaubertConfig
    # 不使用 load_tf_weights
    load_tf_weights = None
    # 基础模型前缀为 "transformer"
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    @property
    def dummy_inputs(self):
        # 创建一个包含示例输入的张量
        inputs_list = torch.tensor([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
        attns_list = torch.tensor([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
        # 如果使用语言嵌入并且语言数量大于1，则生成 langs_list
        if self.config.use_lang_emb and self.config.n_langs > 1:
            langs_list = torch.tensor([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
        else:
            langs_list = None
        return {"input_ids": inputs_list, "attention_mask": attns_list, "langs": langs_list}

    def _init_weights(self, module):
        """Initialize the weights."""
        # 初始化 Embedding 层的权重和偏置
        if isinstance(module, nn.Embedding):
            if self.config is not None and self.config.embed_init_std is not None:
                nn.init.normal_(module.weight, mean=0, std=self.config.embed_init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 初始化 Linear 层的权重和偏置
        if isinstance(module, nn.Linear):
            if self.config is not None and self.config.init_std is not None:
                nn.init.normal_(module.weight, mean=0, std=self.config.init_std)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        # 初始化 LayerNorm 层的权重和偏置
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class FlaubertModel(FlaubertPreTrainedModel):
    # 从 XLMModel 类中复制 get_input_embeddings 方法
    def get_input_embeddings(self):
        return self.embeddings

    # 从 XLMModel 类中复制 set_input_embeddings 方法
    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    # 从 XLMModel 类中复制 _prune_heads 方法
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.attentions[layer].prune_heads(heads)

    # 添加文档字符串注释
    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义一个前向传播函数，接收多个输入参数
    def forward(
        # 输入的 token id，数据类型为长整型张量，可选
        input_ids: Optional[torch.LongTensor] = None,
        # 注意力掩码，数据类型为浮点数张量，可选
        attention_mask: Optional[torch.FloatTensor] = None,
        # 语言 id，数据类型为张量，可选
        langs: Optional[torch.Tensor] = None,
        # 标记类型 id，数据类型为长整型张量，可选
        token_type_ids: Optional[torch.LongTensor] = None,
        # 位置 id，数据类型为长整型张量，可选
        position_ids: Optional[torch.LongTensor] = None,
        # 长度，数据类型为长整型张量，可选
        lengths: Optional[torch.LongTensor] = None,
        # 缓存，数据类型为字典，可选
        cache: Optional[Dict[str, torch.FloatTensor]] = None,
        # 头部掩码，数据类型为浮点数张量，可选
        head_mask: Optional[torch.FloatTensor] = None,
        # 输入嵌入，数据类型为浮点数张量，可选
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 输出注意力信息，数据类型为布尔值，可选
        output_attentions: Optional[bool] = None,
        # 输出隐藏状态，数据类型为布尔值，可选
        output_hidden_states: Optional[bool] = None,
        # 返回字典类型结果，数据类型为布尔值，可选
        return_dict: Optional[bool] = None,
# 导入函数装饰器 add_start_docstrings
# 创建 Flaubert Model transformer，带有语言建模头部（线性层，其权重与输入嵌入层相连）
"""
Flaubert Model transformer，带有在顶部的语言建模头部（线性层，其权重与输入嵌入层相连）。
"""
@add_start_docstrings(FLAUBERT_START_DOCSTRING)
class FlaubertWithLMHeadModel(FlaubertPreTrainedModel):
    # 定义权重共享的键值列表
    _tied_weights_keys = ["pred_layer.proj.weight"]

    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 创建 FlaubertModel 实例
        self.transformer = FlaubertModel(config)
        # 创建 FlaubertPredLayer 实例
        self.pred_layer = FlaubertPredLayer(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入层
    def get_output_embeddings(self):
        return self.pred_layer.proj

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.pred_layer.proj = new_embeddings

    # 为生成准备输入数据
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        mask_token_id = self.config.mask_token_id
        lang_id = self.config.lang_id

        # 计算有效 batch 大小
        effective_batch_size = input_ids.shape[0]
        # 创建 mask token id 的张量
        mask_token = torch.full((effective_batch_size, 1), mask_token_id, dtype=torch.long, device=input_ids.device)
        # 拼接输入和 mask token 张量
        input_ids = torch.cat([input_ids, mask_token], dim=1)
        # 如果 lang id 不为空，则设置 langs 张量
        if lang_id is not None:
            langs = torch.full_like(input_ids, lang_id)
        else:
            langs = None
        return {"input_ids": input_ids, "langs": langs}

    # 前向传播函数
    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC, mask="<special1>")
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # 确定是否返回字典格式的结果，如果未指定则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给Transformer模型以获取Transformer的输出
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从Transformer的输出中获取模型的主要输出
        output = transformer_outputs[0]
        # 将主要输出传递给预测层以获取最终输出
        outputs = self.pred_layer(output, labels)  # (loss, logits) or (logits,) depending on if labels are provided.

        # 如果不返回字典格式的结果，则将Transformer的额外输出添加到最终输出中并返回
        if not return_dict:
            return outputs + transformer_outputs[1:]

        # 返回字典格式的结果，包括损失、logits、隐藏状态和注意力权重
        return MaskedLMOutput(
            loss=outputs[0] if labels is not None else None,
            logits=outputs[0] if labels is None else outputs[1],
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
# 为 Flaubert 模型添加序列分类/回归头部，该头部是基于汇总输出的线性层，用于例如 GLUE 任务
@add_start_docstrings(
    """
    Flaubert Model with a sequence classification/regression head on top (a linear layer on top of the pooled output)
    e.g. for GLUE tasks.
    """,
    FLAUBERT_START_DOCSTRING,
)
# 使用 FLAUBERT_START_DOCSTRING 初始化文档字符串，创建 FlaubertForSequenceClassification 类
class FlaubertForSequenceClassification(FlaubertPreTrainedModel):
    # 初始化函数，接受配置参数 config
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 设置类别数量为配置中的 num_labels
        self.num_labels = config.num_labels
        # 保存配置对象
        self.config = config

        # 初始化 FlaubertModel 对象
        self.transformer = FlaubertModel(config)
        # 初始化 SequenceSummary 对象
        self.sequence_summary = SequenceSummary(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 判断是否需要返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 进行transformer的前向推断
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        output = transformer_outputs[0]  # 获取transformer输出的第一个元素
        logits = self.sequence_summary(output)  # 使用sequence_summary对output进行总结

        loss = None  # 初始化loss为None
        if labels is not None:  # 如果labels不为空
            if self.config.problem_type is None:  # 如果配置的问题类型是None
                if self.num_labels == 1:  # 如果标签数为1
                    self.config.problem_type = "regression"  # 设置问题类型为回归
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):  # 如果标签数大于1且类型是long或者int
                    self.config.problem_type = "single_label_classification"  # 设置问题类型为单标签分类
                else:
                    self.config.problem_type = "multi_label_classification"  # 设置问题类型为多标签分类

            if self.config.problem_type == "regression":  # 如果问题类型是回归
                loss_fct = MSELoss()  # 使用均方误差损失
                if self.num_labels == 1:  # 如果标签数为1
                    loss = loss_fct(logits.squeeze(), labels.squeeze())  # 计算损失
                else:
                    loss = loss_fct(logits, labels)  # 计算损失
            elif self.config.problem_type == "single_label_classification":  # 如果问题类型是单标签分类
                loss_fct = CrossEntropyLoss()  # 使用交叉熵损失
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # 计算损失
            elif self.config.problem_type == "multi_label_classification":  # 如果问题类型是多标签分类
                loss_fct = BCEWithLogitsLoss()  # 使用带logits的二元交叉熵损失
                loss = loss_fct(logits, labels)  # 计算损失

        if not return_dict:  # 如果不需要返回字典
            output = (logits,) + transformer_outputs[1:]  # 将logits和transformer的其他输出组合在一起
            return ((loss,) + output) if loss is not None else output  # 如果有损失则返回损失和输出，否则只返回输出

        return SequenceClassifierOutput(  # 返回序列分类器的输出
            loss=loss,  # 损失
            logits=logits,  # logits
            hidden_states=transformer_outputs.hidden_states,  # 隐层状态
            attentions=transformer_outputs.attentions,  # 注意力
        )
# 使用Flaubert模型进行令牌分类，模型的顶部有一个线性层（隐藏状态输出的一层线性层），用于命名实体识别（NER）任务。
@add_start_docstrings(
    """
    Flaubert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    FLAUBERT_START_DOCSTRING,
)
# 从transformers.models.xlm.modeling_xlm.XLMForTokenClassification复制代码，将XLM_INPUTS->FLAUBERT_INPUTS,XLM->Flaubert
class FlaubertForTokenClassification(FlaubertPreTrainedModel):
    # 定义初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置标签类别数
        self.num_labels = config.num_labels

        # 创建Flaubert模型
        self.transformer = FlaubertModel(config)
        # 添加一个dropout层
        self.dropout = nn.Dropout(config.dropout)
        # 添加一个线性层，用于分类
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 在前向传播方法上添加注释
    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,


注释：
- 使用Flaubert模型进行令牌分类，模型的顶部有一个线性层，用于命名实体识别（NER）任务。
- 定义FlaubertForTokenClassification类，继承自FlaubertPreTrainedModel类。
- 在初始化方法中，设置标签类别数，创建Flaubert模型，并添加dropout层和线性层用于分类。
- 在前向传播方法中，对各个参数进行注释。
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 检查是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给transformer模型进行处理
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从transformer输出中提取序列输出
        sequence_output = outputs[0]

        # 应用丢弃层以减少过拟合风险
        sequence_output = self.dropout(sequence_output)
        # 应用分类器以获得最终的逻辑输出
        logits = self.classifier(sequence_output)

        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果未指定返回字典，则返回元组
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果指定了返回字典，则返回TokenClassifierOutput对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
#
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions` are provided):
            Classification loss as the sum of start token, end token (and is_impossible if provided) classification
            losses.
        start_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the top config.start_n_top start token possibilities (beam-search).
        start_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Indices for the top config.start_n_top start token possibilities (beam-search).
        end_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the top `config.start_n_top * config.end_n_top` end token possibilities
            (beam-search).
        end_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Indices for the top `config.start_n_top * config.end_n_top` end token possibilities (beam-search).
        cls_logits (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the `is_impossible` label of the answers.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    
    # 损失，如果提供了`start_positions`和`end_positions`则返回，表示开始和结束位置的分类损失
    loss: Optional[torch.FloatTensor] = None
    # 如果`start_positions`或`end_positions`未提供，则返回前top `config.start_n_top`个开始标记的对数概率（beam-search）
    start_top_log_probs: Optional[torch.FloatTensor] = None
    # 如果`start_positions`或`end_positions`未提供，则返回前top `config.start_n_top`个开始标记的索引（beam-search）
    start_top_index: Optional[torch.LongTensor] = None
    # 如果`start_positions`或`end_positions`未提供，则返回前top `config.start_n_top * config.end_n_top`个结束标记的对数概率（beam-search）
    end_top_log_probs: Optional[torch.FloatTensor] = None
    # 如果`start_positions`或`end_positions`未提供，则返回前top `config.start_n_top * config.end_n_top`个结束标记的索引（beam-search）
    end_top_index: Optional[torch.LongTensor] = None
    # 如果`start_positions`或`end_positions`未提供，则返回答案`is_impossible`标签的对数概率
    cls_logits: Optional[torch.FloatTensor] = None
    # 当传递了`output_hidden_states=True`或`config.output_hidden_states=True`时返回，每一层的输出和初始嵌入输出的元组
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 当传递了`output_attentions=True`或`config.output_attentions=True`时返回，每一层的注意力权重的元组
    attentions: Optional[Tuple[torch.FloatTensor]] = None
```  
# 从 transformer.models.xlm.modeling_xlm.XLMForQuestionAnswering 复制过来，进行了修改以适应 Flaubert 输入和 Flaubert 模型
class FlaubertForQuestionAnswering(FlaubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 Flaubert 模型
        self.transformer = FlaubertModel(config)
        # 初始化用于问答的输出层
        self.qa_outputs = SQuADHead(config)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=FlaubertForQuestionAnsweringOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        is_impossible: Optional[torch.Tensor] = None,
        cls_index: Optional[torch.Tensor] = None,
        p_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 加入模型前的文档字符串，定义了模型的输入格式和语义
@add_start_docstrings(
    """
    基于 Flaubert 模型的多项选择分类头部（在汇总输出之上的线性层和 softmax），例如用于 RocStories/SWAG 任务。
    """,
    FLAUBERT_START_DOCSTRING,
)
# 从 transformer.models.xlm.modeling_xlm.XLMForMultipleChoice 复制过来，进行了修改以适应 Flaubert 输入和 Flaubert 模型
class FlaubertForMultipleChoice(FlaubertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化 Flaubert 模型
        self.transformer = FlaubertModel(config)
        # 初始化序列汇总层
        self.sequence_summary = SequenceSummary(config)
        # 初始化 logits 投影层
        self.logits_proj = nn.Linear(config.num_labels, 1)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(
        FLAUBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义一个 forward 方法，用于模型推断阶段的前向传播
    def forward(
        # 输入序列的 token IDs，默认为 None
        input_ids: Optional[torch.Tensor] = None,
        # 注意力遮罩，默认为 None
        attention_mask: Optional[torch.Tensor] = None,
        # 语言 ID，默认为 None
        langs: Optional[torch.Tensor] = None,
        # token 类型 IDs，默认为 None
        token_type_ids: Optional[torch.Tensor] = None,
        # 位置 IDs，默认为 None
        position_ids: Optional[torch.Tensor] = None,
        # 序列长度，默认为 None
        lengths: Optional[torch.Tensor] = None,
        # 缓存信息，默认为 None
        cache: Optional[Dict[str, torch.Tensor]] = None,
        # 头部遮罩，默认为 None
        head_mask: Optional[torch.Tensor] = None,
        # 输入嵌入，默认为 None
        inputs_embeds: Optional[torch.Tensor] = None,
        # 标签，默认为 None
        labels: Optional[torch.Tensor] = None,
        # 是否输出注意力权重，默认为 None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，默认为 None
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典，默认为 None
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`:
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 使用给定的 return_dict 值，如果为 None 则使用配置中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取输入 input_ids 的第二维度的大小，即选项的数量
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 根据需要重构输入
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        langs = langs.view(-1, langs.size(-1)) if langs is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 如果 lengths 不为 None，给出警告并将其设为 None
        if lengths is not None:
            logger.warning(
                "The `lengths` parameter cannot be used with the Flaubert multiple choice models. Please use the "
                "attention mask instead."
            )
            lengths = None

        # 传递输入数据到 transformer 模型内进行处理
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取 transformer 的输出
        output = transformer_outputs[0]
        # 对输出进行序列摘要
        logits = self.sequence_summary(output)
        # 对 logits 进行变换
        logits = self.logits_proj(logits)
        reshaped_logits = logits.view(-1, num_choices)

        # 初始化 loss 为 None
        loss = None
        # 如果 labels 不为 None，计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果 return_dict 为 False，返回输出；否则返回 MultipleChoiceModelOutput 对象
        if not return_dict:
            output = (reshaped_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
```