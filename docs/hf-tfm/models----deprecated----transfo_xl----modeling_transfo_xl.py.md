# `.\models\deprecated\transfo_xl\modeling_transfo_xl.py`

```py
# 声明 Python 源文件的编码格式为 UTF-8
# 版权声明和许可信息，指明代码的版权和使用许可
# 包含 Apache License 2.0 许可信息，告知使用者可以按此许可使用代码

"""
 PyTorch Transformer XL model. Adapted from https://github.com/kimiyoung/transformer-xl. In particular
 https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py
"""
# 引入警告模块，用于可能的警告信息
import warnings
# 引入 dataclasses 模块中的 dataclass 装饰器
from dataclasses import dataclass
# 引入 typing 模块中的 List, Optional, Tuple, Union 类型定义
from typing import List, Optional, Tuple, Union

# 引入 PyTorch 库
import torch
# 引入 torch.nn 模块中的 nn 模块
from torch import nn
# 引入损失函数 BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 从 ....modeling_utils 模块导入 PreTrainedModel 类
from ....modeling_utils import PreTrainedModel
# 从 ....utils 模块中导入 ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging 函数
from ....utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
# 从 .configuration_transfo_xl 模块中导入 TransfoXLConfig 类
from .configuration_transfo_xl import TransfoXLConfig
# 从 .modeling_transfo_xl_utilities 模块中导入 ProjectedAdaptiveLogSoftmax 类
from .modeling_transfo_xl_utilities import ProjectedAdaptiveLogSoftmax

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 预设用于文档的模型检查点
_CHECKPOINT_FOR_DOC = "transfo-xl/transfo-xl-wt103"
# 预设用于文档的配置信息
_CONFIG_FOR_DOC = "TransfoXLConfig"

# Transformer XL 的预训练模型存档列表
TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "transfo-xl/transfo-xl-wt103",
    # 可以在 https://huggingface.co/models?filter=transfo-xl 查看所有 Transformer XL 模型
]


def build_tf_to_pytorch_map(model, config):
    """
    A map of modules from TF to PyTorch. This time I use a map to keep the PyTorch model as identical to the original
    PyTorch model as possible.
    """
    # 创建一个从 TF 到 PyTorch 的模块映射字典，目的是尽可能地保持 PyTorch 模型与原始模型的一致性
    tf_to_pt_map = {}
    # 检查模型是否具有"transformer"属性
    if hasattr(model, "transformer"):
        # 如果模型是 TransfoXLLMHeadModel 类型，需要加载 Adaptive Softmax 部分
        tf_to_pt_map.update(
            {
                "transformer/adaptive_softmax/cutoff_0/cluster_W": model.crit.cluster_weight,
                "transformer/adaptive_softmax/cutoff_0/cluster_b": model.crit.cluster_bias,
            }
        )
        # 遍历 Adaptive Softmax 的各层
        for i, (out_l, proj_l, tie_proj) in enumerate(
            zip(model.crit.out_layers, model.crit.out_projs, config.tie_projs)
        ):
            layer_str = f"transformer/adaptive_softmax/cutoff_{i}/"
            if config.tie_word_embeddings:
                # 更新映射，将偏置项加入映射表中
                tf_to_pt_map.update({layer_str + "b": out_l.bias})
            else:
                # 如果未实现绑定词嵌入，则抛出未实现错误
                raise NotImplementedError
                # 下面这行代码可能并未在 TF 代码中实现
                tf_to_pt_map.update({layer_str + "lookup_table": out_l.weight, layer_str + "b": out_l.bias})
            if not tie_proj:
                # 如果不绑定投影层，将投影层添加到映射表中
                tf_to_pt_map.update({layer_str + "proj": proj_l})
        # 加载其余的 Transformer 部分
        model = model.transformer

    # Embeddings（嵌入层）
    for i, (embed_l, proj_l) in enumerate(zip(model.word_emb.emb_layers, model.word_emb.emb_projs)):
        layer_str = f"transformer/adaptive_embed/cutoff_{i}/"
        # 更新映射，将嵌入层的权重和投影矩阵加入映射表中
        tf_to_pt_map.update({layer_str + "lookup_table": embed_l.weight, layer_str + "proj_W": proj_l})

    # Transformer blocks（Transformer 块）
    for i, b in enumerate(model.layers):
        layer_str = f"transformer/layer_{i}/"
        # 更新映射，将注意力层和前馈神经网络层的权重、偏置加入映射表中
        tf_to_pt_map.update(
            {
                layer_str + "rel_attn/LayerNorm/gamma": b.dec_attn.layer_norm.weight,
                layer_str + "rel_attn/LayerNorm/beta": b.dec_attn.layer_norm.bias,
                layer_str + "rel_attn/o/kernel": b.dec_attn.o_net.weight,
                layer_str + "rel_attn/qkv/kernel": b.dec_attn.qkv_net.weight,
                layer_str + "rel_attn/r/kernel": b.dec_attn.r_net.weight,
                layer_str + "ff/LayerNorm/gamma": b.pos_ff.layer_norm.weight,
                layer_str + "ff/LayerNorm/beta": b.pos_ff.layer_norm.bias,
                layer_str + "ff/layer_1/kernel": b.pos_ff.CoreNet[0].weight,
                layer_str + "ff/layer_1/bias": b.pos_ff.CoreNet[0].bias,
                layer_str + "ff/layer_2/kernel": b.pos_ff.CoreNet[3].weight,
                layer_str + "ff/layer_2/bias": b.pos_ff.CoreNet[3].bias,
            }
        )

    # Relative positioning biases（相对位置偏置）
    if config.untie_r:
        # 如果配置为解绑相对位置偏置，则逐个添加到列表中
        r_r_list = []
        r_w_list = []
        for b in model.layers:
            r_r_list.append(b.dec_attn.r_r_bias)
            r_w_list.append(b.dec_attn.r_w_bias)
    else:
        # 否则将整体的相对位置偏置添加到列表中
        r_r_list = [model.r_r_bias]
        r_w_list = [model.r_w_bias]
    # 更新映射，将相对位置偏置加入映射表中
    tf_to_pt_map.update({"transformer/r_r_bias": r_r_list, "transformer/r_w_bias": r_w_list})
    # 返回映射表
    return tf_to_pt_map
# 在给定的 PyTorch 模型中加载 TensorFlow 的权重
def load_tf_weights_in_transfo_xl(model, config, tf_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import numpy as np  # 导入 NumPy 库，用于处理数组数据
        import tensorflow as tf  # 导入 TensorFlow 库，用于加载 TensorFlow 模型权重
    except ImportError:
        logger.error(
            "Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    # 构建 TensorFlow 到 PyTorch 权重加载映射
    tf_to_pt_map = build_tf_to_pytorch_map(model, config)

    # 从 TensorFlow 模型加载权重
    init_vars = tf.train.list_variables(tf_path)
    tf_weights = {}
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        tf_weights[name] = array

    # 将 TensorFlow 权重映射到 PyTorch 模型中
    for name, pointer in tf_to_pt_map.items():
        assert name in tf_weights  # 断言确保映射中的权重在 TensorFlow 加载的权重中存在
        array = tf_weights[name]
        
        # 如果权重名中包含 "kernel" 或 "proj"，需要对数组进行转置
        if "kernel" in name or "proj" in name:
            array = np.transpose(array)
        
        # 如果权重名中包含 "r_r_bias" 或 "r_w_bias"，并且指针长度大于 1，则需要拆分 TensorFlow 的权重
        if ("r_r_bias" in name or "r_w_bias" in name) and len(pointer) > 1:
            assert len(pointer) == array.shape[0]  # 断言确保指针长度与数组第一维度长度相等
            for i, p_i in enumerate(pointer):
                arr_i = array[i, ...]
                try:
                    assert p_i.shape == arr_i.shape  # 断言确保指针形状与数组形状相等
                except AssertionError as e:
                    e.args += (p_i.shape, arr_i.shape)
                    raise
                logger.info(f"Initialize PyTorch weight {name} for layer {i}")
                p_i.data = torch.from_numpy(arr_i)
        else:
            try:
                assert pointer.shape == array.shape  # 断言确保指针形状与数组形状相等
            except AssertionError as e:
                e.args += (pointer.shape, array.shape)
                raise
            logger.info(f"Initialize PyTorch weight {name}")
            pointer.data = torch.from_numpy(array)
        
        # 从 TensorFlow 权重字典中移除已处理的权重项
        tf_weights.pop(name, None)
        tf_weights.pop(name + "/Adam", None)
        tf_weights.pop(name + "/Adam_1", None)

    # 输出未能复制到 PyTorch 模型的权重名称列表
    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}")
    return model


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super().__init__()

        self.demb = demb

        # 计算位置编码的频率逆数，注册为模型的缓冲区
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)
    # 定义一个方法 `forward`，用于生成位置编码
    def forward(self, pos_seq, bsz=None):
        # 根据位置序列和预定义的频率向量生成正弦波和余弦波输入
        sinusoid_inp = torch.outer(pos_seq, self.inv_freq)
        # 将正弦波和余弦波拼接在一起形成位置编码张量
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        # 如果指定了批大小 `bsz`，则将位置编码张量扩展成相应形状后返回
        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            # 否则返回未扩展的位置编码张量
            return pos_emb[:, None, :]
# 定义一个相对位置可学习的多头注意力模块
class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        dropatt=0,
        pre_lnorm=False,
        r_r_bias=None,
        r_w_bias=None,
        layer_norm_epsilon=1e-5,
    ):
        super().__init__()

        self.n_head = n_head  # 注意力头的数量
        self.d_model = d_model  # 模型的维度
        self.d_head = d_head  # 每个注意力头的维度
        self.dropout = dropout  # dropout 概率

        # qkv_net 是用来计算查询、键、值的线性层
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)  # dropout 模块
        self.dropatt = nn.Dropout(dropatt)  # attention dropout 模块
        # o_net 是用来计算输出的线性层
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        # layer_norm 是层归一化层
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)

        self.scale = 1 / (d_head**0.5)  # 缩放因子

        self.pre_lnorm = pre_lnorm  # 是否在层归一化之前应用

        if r_r_bias is None or r_w_bias is None:  # 如果没有提供偏置，创建新的参数
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        else:
            self.r_r_bias = r_r_bias  # 相对位置注意力的查询偏置
            self.r_w_bias = r_w_bias  # 相对位置注意力的键值偏置

        # r_net 是用来计算相对位置编码的线性层
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def _rel_shift(self, x):
        # 对输入张量进行相对位置偏移
        zero_pad_shape = (x.size(0), 1) + x.size()[2:]
        zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded_shape = (x.size(1) + 1, x.size(0)) + x.size()[2:]
        x_padded = x_padded.view(*x_padded_shape)

        x = x_padded[1:].view_as(x)

        return x
    # 初始化方法，用于创建一个新的RelPartialLearnableMultiHeadAttn对象和一个PositionwiseFF对象
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, layer_norm_epsilon=1e-5, **kwargs):
        # 调用父类的初始化方法
        super().__init__()

        # 创建一个RelPartialLearnableMultiHeadAttn对象，用于处理解码器的注意力部分
        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, layer_norm_epsilon=layer_norm_epsilon, **kwargs
        )
        
        # 创建一个PositionwiseFF对象，用于处理解码器的前向传播部分
        self.pos_ff = PositionwiseFF(
            d_model, d_inner, dropout, pre_lnorm=kwargs.get("pre_lnorm"), layer_norm_epsilon=layer_norm_epsilon
        )

    # 前向传播方法，接受解码器输入并进行处理
    def forward(self, dec_inp, r, dec_attn_mask=None, mems=None, head_mask=None, output_attentions=False):
        # 使用RelPartialLearnableMultiHeadAttn对象处理解码器输入，得到注意力输出
        attn_outputs = self.dec_attn(
            dec_inp,
            r,
            attn_mask=dec_attn_mask,
            mems=mems,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        
        # 使用PositionwiseFF对象处理RelPartialLearnableMultiHeadAttn的输出，得到前向传播的输出
        ff_output = self.pos_ff(attn_outputs[0])

        # 将前向传播的输出和注意力机制的输出合并成一个列表作为最终的输出
        outputs = [ff_output] + attn_outputs[1:]

        # 返回最终的输出
        return outputs
# 定义一个自适应嵌入层的神经网络模块
class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, sample_softmax=False):
        super().__init__()

        self.n_token = n_token  # 嵌入层的词汇表大小
        self.d_embed = d_embed  # 嵌入向量的维度

        self.cutoffs = cutoffs + [n_token]  # 嵌入层的截断点列表，加上词汇表大小
        self.div_val = div_val  # 分割值，用于确定每个截断的嵌入维度变化
        self.d_proj = d_proj  # 投影后的维度

        self.emb_scale = d_proj**0.5  # 嵌入向量缩放因子

        self.cutoff_ends = [0] + self.cutoffs  # 嵌入层的截断点列表，包括起始点0

        self.emb_layers = nn.ModuleList()  # 嵌入层模块列表
        self.emb_projs = nn.ParameterList()  # 嵌入投影参数列表
        if div_val == 1:
            self.emb_layers.append(nn.Embedding(n_token, d_embed, sparse=sample_softmax > 0))
            # 如果没有分割，直接创建一个标准的嵌入层
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_embed)))
                # 如果投影维度与嵌入维度不同，添加一个投影参数
        else:
            # 如果有分割，根据每个截断区间创建对应的嵌入层和投影参数
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val**i)
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)  # 使用第一个嵌入层对输入进行嵌入
            if self.d_proj != self.d_embed:
                embed = nn.functional.linear(embed, self.emb_projs[0])
                # 如果投影维度不等于嵌入维度，对嵌入向量进行线性投影
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], dtype=param.dtype, device=param.device)
            # 创建一个与输入扁平化后大小相同的零张量

            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = nn.functional.linear(emb_i, self.emb_projs[i])
                # 对于每个区间，对相应的输入进行嵌入和投影操作，并将结果复制回原始位置

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed_shape = inp.size() + (self.d_proj,)
            embed = emb_flat.view(embed_shape)
            # 将扁平化的嵌入向量形状转换回原始形状

        embed.mul_(self.emb_scale)  # 缩放嵌入向量

        return embed


class TransfoXLPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TransfoXLConfig  # 配置类的引用
    load_tf_weights = load_tf_weights_in_transfo_xl  # 加载 TensorFlow 权重的方法引用
    base_model_prefix = "transformer"  # 基础模型前缀

    def _init_weight(self, weight):
        if self.config.init == "uniform":
            nn.init.uniform_(weight, -self.config.init_range, self.config.init_range)
            # 如果初始化配置为均匀分布，使用均匀分布初始化权重
        elif self.config.init == "normal":
            nn.init.normal_(weight, 0.0, self.config.init_std)
            # 如果初始化配置为正态分布，使用正态分布初始化权重

    def _init_bias(self, bias):
        nn.init.constant_(bias, 0.0)
        # 使用常数初始化偏置
    # 初始化神经网络模型的权重
    def _init_weights(self, m):
        # 获取当前层的类名
        classname = m.__class__.__name__
        # 如果是线性层（Linear）
        if classname.find("Linear") != -1:
            # 初始化权重
            if hasattr(m, "weight") and m.weight is not None:
                self._init_weight(m.weight)
            # 初始化偏置
            if hasattr(m, "bias") and m.bias is not None:
                self._init_bias(m.bias)
        # 如果是自适应嵌入层（AdaptiveEmbedding）
        elif classname.find("AdaptiveEmbedding") != -1:
            # 初始化嵌入投影层的权重
            if hasattr(m, "emb_projs"):
                for i in range(len(m.emb_projs)):
                    if m.emb_projs[i] is not None:
                        nn.init.normal_(m.emb_projs[i], 0.0, self.config.proj_init_std)
        # 如果是普通嵌入层（Embedding）
        elif classname.find("Embedding") != -1:
            # 初始化权重
            if hasattr(m, "weight"):
                self._init_weight(m.weight)
        # 如果是投影自适应对数softmax层（ProjectedAdaptiveLogSoftmax）
        elif classname.find("ProjectedAdaptiveLogSoftmax") != -1:
            # 初始化聚类权重
            if hasattr(m, "cluster_weight") and m.cluster_weight is not None:
                self._init_weight(m.cluster_weight)
            # 初始化聚类偏置
            if hasattr(m, "cluster_bias") and m.cluster_bias is not None:
                self._init_bias(m.cluster_bias)
            # 初始化输出投影层的权重
            if hasattr(m, "out_projs"):
                for i in range(len(m.out_projs)):
                    if m.out_projs[i] is not None:
                        nn.init.normal_(m.out_projs[i], 0.0, self.config.proj_init_std)
        # 如果是层归一化层（LayerNorm）
        elif classname.find("LayerNorm") != -1:
            # 初始化权重，均值为1.0，标准差为self.config.init_std
            if hasattr(m, "weight"):
                nn.init.normal_(m.weight, 1.0, self.config.init_std)
            # 初始化偏置
            if hasattr(m, "bias") and m.bias is not None:
                self._init_bias(m.bias)
        # 对于其他情况
        else:
            # 初始化特定属性的权重
            if hasattr(m, "r_emb"):
                self._init_weight(m.r_emb)
            if hasattr(m, "r_w_bias"):
                self._init_weight(m.r_w_bias)
            if hasattr(m, "r_r_bias"):
                self._init_weight(m.r_r_bias)
            if hasattr(m, "r_bias"):
                self._init_bias(m.r_bias)
    # 调整模型输入的 token embeddings 矩阵大小，如果 new_num_tokens 不等于 config.vocab_size，则进行调整。
    # 调整后需注意是否需要重新绑定权重 embeddings，如果模型类有 tie_weights() 方法的话。
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, layer: Optional[int] = -1):
        """
        Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size. Take care of tying
        weights embeddings afterwards if the model class has a *tie_weights()* method.
    
        Arguments:
            new_num_tokens: (*optional*) int:
                New number of tokens in the embedding matrix. Increasing the size will add newly initialized vectors at
                the end. Reducing the size will remove vectors from the end. If not provided or None: does nothing and
                just returns a pointer to the input tokens `torch.nn.Embeddings` Module of the model.
            layer: (*optional*) int:
                Layer of the *AdaptiveEmbedding* where the resizing should be done. Per default the last layer will be
                resized. Be aware that when resizing other than the last layer, you have to ensure that the new
                token(s) in the tokenizer are at the corresponding position.
    
        Return: `torch.nn.Embeddings` Pointer to the input tokens Embeddings Module of the model
        """
        # 获取基础模型（如果需要的话）
        base_model = getattr(self, self.base_model_prefix, self)
    
        # 如果 new_num_tokens 为 None，则返回当前输入 token embeddings 的指针
        if new_num_tokens is None:
            return self.get_input_embeddings()
    
        # 获取新的 token 数量和层索引
        new_num_tokens_layer, layer = self._get_new_num_tokens_layer(new_num_tokens, layer)
        # 断言新的 embedding 层大小大于 0
        assert new_num_tokens_layer > 0, "The size of the new embedding layer cannot be 0 or less"
        # 调整 token embeddings 并获取模型 embeds
        model_embeds = base_model._resize_token_embeddings(new_num_tokens_layer, layer)
    
        # 更新基础模型和当前模型的配置
        self.config.vocab_size = new_num_tokens
        base_model.vocab_size = new_num_tokens
        base_model.n_token = new_num_tokens
    
        # 获取新的 embedding 形状列表
        new_embedding_shapes = self._get_embedding_shapes()
        # 调整截断点（如果有）
        self._resize_cutoffs(new_num_tokens, new_num_tokens_layer, new_embedding_shapes, layer)
    
        # 如果需要的话重新绑定权重
        self.tie_weights()
    
        # 返回模型 embeds
        return model_embeds
    
    def _get_new_num_tokens_layer(self, new_num_tokens, layer):
        # 获取输入 embeddings
        embeddings = self.get_input_embeddings()
        # 如果 layer 为 -1，则设置为最后一层的索引
        if layer == -1:
            layer = len(embeddings.emb_layers) - 1
        # 断言 layer 在有效范围内
        assert 0 <= layer <= len(embeddings.emb_layers) - 1
    
        # 计算新的 embedding 层的 token 数量
        new_num_tokens_layer = (
            new_num_tokens
            - sum([emb.weight.shape[0] for emb in embeddings.emb_layers[:layer]])
            - sum([emb.weight.shape[0] for emb in embeddings.emb_layers[layer + 1:]])
        )
        return new_num_tokens_layer, layer
    
    def _get_embedding_shapes(self):
        # 获取输入 embeddings
        embeddings = self.get_input_embeddings()
        # 返回每个 embedding 层的 token 数量列表
        return [emb.weight.shape[0] for emb in embeddings.emb_layers]
    # 调整模型中的词嵌入大小，用于处理新的词汇量
    def _resize_token_embeddings(self, new_num_tokens, layer=-1):
        # 获取当前的输入词嵌入层
        embeddings = self.get_input_embeddings()
        # 如果新的词汇量为None，直接返回当前的词嵌入层
        if new_num_tokens is None:
            return embeddings
        # 获取调整后的新词嵌入层
        new_embeddings_layer = self._get_resized_embeddings(embeddings.emb_layers[layer], new_num_tokens)
        # 将新的词嵌入层放回原来的位置
        embeddings.emb_layers[layer] = new_embeddings_layer

        # 更新模型的输入词嵌入
        self.set_input_embeddings(embeddings)

        # 返回更新后的输入词嵌入
        return self.get_input_embeddings()

    # 调整截断点列表，以匹配新的词汇量和词嵌入尺寸
    def _resize_cutoffs(self, new_num_tokens, new_emb_size, new_embedding_shapes, layer):
        # 获取当前的输入词嵌入层
        embeddings = self.get_input_embeddings()

        # 根据新的词嵌入形状更新截断点列表
        for i in range(layer, len(embeddings.cutoffs)):
            embeddings.cutoffs[i] = sum(new_embedding_shapes[: i + 1])

        # 更新截断结束点列表
        embeddings.cutoff_ends = [0] + embeddings.cutoffs
        # 更新词汇量
        embeddings.n_token = new_num_tokens

        # 更新配置中的截断点列表
        self.config.cutoffs = embeddings.cutoffs[:-1]

        # 返回更新后的截断点列表
        return embeddings.cutoffs
# 使用 `dataclass` 装饰器定义一个数据类，表示 TransfoXL 模型的输出结果，继承自 `ModelOutput` 类。
@dataclass
class TransfoXLModelOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems`
            input) to speed up sequential decoding. The token ids which have their past given to this model should not
            be passed as input ids as they have already been computed.
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

    # 定义类的属性，表示模型输出中的最后一个隐藏层的状态
    last_hidden_state: torch.FloatTensor
    # 包含预先计算的隐藏状态列表，用于在顺序解码中加速处理
    mems: List[torch.FloatTensor] = None
    # 可选的元组，包含模型每一层的隐藏状态，返回条件是在 `output_hidden_states=True` 时或 `config.output_hidden_states=True` 时
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 可选的元组，包含每一层的注意力权重，返回条件是在 `output_attentions=True` 时或 `config.output_attentions=True` 时
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 使用 `dataclass` 装饰器定义一个数据类，表示带有历史信息的 TransfoXL 序列分类器的输出结果，继承自 `ModelOutput` 类。
@dataclass
class TransfoXLSequenceClassifierOutputWithPast(ModelOutput):
    """
    Base class for outputs of sentence classification models.
    """
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类（或回归，如果config.num_labels==1）的损失值。
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            分类（或回归，如果config.num_labels==1）的分数（SoftMax之前）。
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            包含预先计算的隐藏状态（注意力模块中的键和值）。可以用来加速顺序解码。
            给定给模型的过去记忆的令牌ID不应作为输入ID传递，因为它们已经被计算过。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `torch.FloatTensor`的元组（一个用于嵌入层的输出 + 每层的输出），形状为`(batch_size, sequence_length, hidden_size)`。
            模型每层的隐藏状态加上初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            `torch.FloatTensor`的元组（每层一个）形状为`(batch_size, num_heads, sequence_length, sequence_length)`。
            注意力softmax后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None  # 分类或回归损失（如果提供了标签，则返回）
    logits: torch.FloatTensor = None  # 分类或回归的分数（SoftMax之前）
    mems: List[torch.FloatTensor] = None  # 预先计算的隐藏状态列表，用于加速顺序解码
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 模型每层的隐藏状态和初始嵌入输出的元组
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # 每层注意力权重的元组，用于自注意力加权平均值计算
@dataclass
class TransfoXLLMHeadModelOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        losses (`torch.FloatTensor` of shape *(batch_size, sequence_length-1)*, *optional*, returned when `labels` is provided):
            Language modeling losses (not reduced).
        prediction_scores (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token after SoftMax).
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems`
            input) to speed up sequential decoding. The token ids which have their past given to this model should not
            be passed as input ids as they have already been computed.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        loss (`torch.FloatTensor` of shape `()`, *optional*, returned when `labels` is provided)
            Reduced language modeling loss.
    """

    losses: Optional[torch.FloatTensor] = None  # 语言模型损失，未经归约
    prediction_scores: torch.FloatTensor = None  # 语言建模头部的预测分数，经过SoftMax后每个词汇标记的分数
    mems: List[torch.FloatTensor] = None  # 预先计算的隐藏状态列表，用于加速顺序解码
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 模型各层输出的隐藏状态，包括初始嵌入输出
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # 注意力权重，经过注意力SoftMax后的结果，用于计算自注意力头的加权平均值
    loss: Optional[torch.FloatTensor] = None  # 归约的语言建模损失

    @property
    def logits(self):
        # 预测分数是自适应SoftMax的输出，参见 `modeling_transfo_xl_utilities` 文件。
        # 由于自适应SoftMax返回log softmax值，因此 `self.prediction_scores` 严格来说不是 `logits`，但其行为方式与logits相同。
        return self.prediction_scores


TRANSFO_XL_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
"""
    # 这个模型也是 PyTorch 中的一个子类 [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)。
    # 可以像普通的 PyTorch 模块一样使用，并且在使用和行为上可以参考 PyTorch 的文档。

    # 参数:
    #     config ([`TransfoXLConfig`]): 包含模型所有参数的配置类。
    #         使用配置文件初始化模型时不会加载模型的权重，只会加载配置信息。
    #         若要加载模型的权重，请查看 [`~PreTrainedModel.from_pretrained`] 方法。
"""
TRANSFO_XL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model (see
            `mems` output below). Can be used to speed up sequential decoding. The token ids which have their mems
            given to this model should not be passed as `input_ids` as they have already been computed.
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    TRANSFO_XL_START_DOCSTRING,
)
class TransfoXLModel(TransfoXLPreTrainedModel):
    """
    TransfoXLModel class inherits from TransfoXLPreTrainedModel and represents the main model for TransfoXL.

    This class provides the core Transformer-XL model for processing sequences, without any task-specific head.

    Args:
        config (:class:`~transformers.TransfoXLConfig`):
            The model configuration class that defines the model architecture and its parameters.

    Inherits from:
        :class:`~transformers.TransfoXLPreTrainedModel`
    """
    # 初始化方法，用于初始化Transformer-XL模型的各个参数和层
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 设置词汇表大小
        self.n_token = config.vocab_size

        # 设置词嵌入的维度和模型的维度
        self.d_embed = config.d_embed
        self.d_model = config.d_model

        # 设置注意力头的数量和每个头的维度
        self.n_head = config.n_head
        self.d_head = config.d_head

        # 创建自适应词嵌入层，根据词汇表大小、词嵌入维度、模型维度、截断参数和分割值创建
        self.word_emb = AdaptiveEmbedding(
            config.vocab_size, config.d_embed, config.d_model, config.cutoffs, div_val=config.div_val
        )

        # 添加Dropout层，使用指定的丢弃率
        self.drop = nn.Dropout(config.dropout)

        # 设置层数和记忆长度
        self.n_layer = config.n_layer
        self.mem_len = config.mem_len

        # 设置注意力类型
        self.attn_type = config.attn_type

        # 如果不是解耦的注意力机制，则创建r_w_bias和r_r_bias作为可学习参数
        if not config.untie_r:
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))

        # 创建多层Transformer-XL解码层
        self.layers = nn.ModuleList()
        if config.attn_type == 0:  # 默认的注意力类型
            # 根据层数循环创建RelPartialLearnableDecoderLayer，并添加到self.layers中
            for i in range(config.n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        config.n_head,
                        config.d_model,
                        config.d_head,
                        config.d_inner,
                        config.dropout,
                        dropatt=config.dropatt,
                        pre_lnorm=config.pre_lnorm,
                        r_w_bias=None if config.untie_r else self.r_w_bias,
                        r_r_bias=None if config.untie_r else self.r_r_bias,
                        layer_norm_epsilon=config.layer_norm_epsilon,
                    )
                )
        else:
            # 如果不是默认的注意力类型，抛出未实现错误
            raise NotImplementedError  # 移除这些以避免维护死代码

        # 设置同等长度和限制长度
        self.same_length = config.same_length
        self.clamp_len = config.clamp_len

        # 根据注意力类型设置位置编码
        if self.attn_type == 0:  # 默认注意力类型
            self.pos_emb = PositionalEmbedding(self.d_model)
        else:
            # 如果不是默认注意力类型，抛出未实现错误
            raise NotImplementedError  # 移除这些以避免维护死代码

        # 初始化权重并进行最终处理
        self.post_init()

    # 返回词嵌入层
    def get_input_embeddings(self):
        return self.word_emb

    # 设置新的输入词嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.word_emb = new_embeddings

    # 向后兼容性方法
    def backward_compatible(self):
        self.sample_softmax = -1

    # 重置记忆长度
    def reset_memory_length(self, mem_len):
        self.mem_len = mem_len

    # 剪枝注意力头
    def _prune_heads(self, heads):
        logger.info("Head pruning is not implemented for Transformer-XL model")
        pass  # 留空函数体
    def init_mems(self, bsz):
        # 如果预定义的记忆长度大于零
        if self.mem_len > 0:
            # 创建一个空列表 mems 来存储记忆张量
            mems = []
            # 获取模型的第一个参数（通常是用来确定数据类型和设备）
            param = next(self.parameters())
            # 对每一层进行循环，创建一个全零张量作为记忆，并加入 mems 列表
            for i in range(self.n_layer):
                empty = torch.zeros(self.mem_len, bsz, self.config.d_model, dtype=param.dtype, device=param.device)
                mems.append(empty)

            # 返回记忆张量列表
            return mems
        else:
            # 如果预定义的记忆长度不大于零，则返回 None
            return None

    def _update_mems(self, hids, mems, mlen, qlen):
        # 如果 mems 为 None，则直接返回 None
        if mems is None:
            return None

        # 断言 hids 和 mems 的长度必须相等，否则抛出异常
        assert len(hids) == len(mems), "len(hids) != len(mems)"

        # 计算可以缓存到 mems 中的步数总和为 mlen + max(0, qlen)
        with torch.no_grad():
            # 创建一个新的 mems 列表
            new_mems = []
            # 计算起始索引和结束索引
            end_idx = mlen + max(0, qlen)
            beg_idx = max(0, end_idx - self.mem_len)
            # 对每一层的隐藏状态进行循环处理
            for i in range(len(hids)):
                # 将旧的记忆和当前隐藏状态拼接起来，然后截取指定范围的片段，并且分离计算图
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        # 返回更新后的 mems 列表
        return new_mems

    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TransfoXLModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        mems: Optional[List[torch.FloatTensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
"""
The Transformer-XL Model with a language modeling head on top (adaptive softmax with weights tied to the adaptive
input embeddings)
"""
@add_start_docstrings(
    """
    The Transformer-XL Model with a language modeling head on top (adaptive softmax with weights tied to the adaptive
    input embeddings)
    """,
    TRANSFO_XL_START_DOCSTRING,
)
class TransfoXLLMHeadModel(TransfoXLPreTrainedModel):
    _tied_weights_keys = [r"crit\.out_projs\.\d+", r"crit\.out_layers\.\d+\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = TransfoXLModel(config)
        self.sample_softmax = config.sample_softmax
        self.trainer_compatible = getattr(config, "trainer_compatible", False)

        if not self.trainer_compatible:
            warnings.warn(
                "The output of TransfoXL will be updated in v5 to support a single loss as first argument. In order "
                "to use that updated output, please specify `trainer_compatible=True` as your configuration"
                " attribute.",
                DeprecationWarning,
            )

        assert self.sample_softmax <= 0, (
            "Sampling from the softmax is not implemented yet. Please look at issue: #3310:"
            " https://github.com/huggingface/transformers/issues/3310"
        )

        self.crit = ProjectedAdaptiveLogSoftmax(
            config.vocab_size, config.d_embed, config.d_model, config.cutoffs, div_val=config.div_val
        )

        # Initialize weights and apply final processing
        self.post_init()

    def tie_weights(self):
        """
        Run this to be sure output and input (adaptive) softmax weights are tied
        """

        if self.config.tie_word_embeddings:
            for i in range(len(self.crit.out_layers)):
                # Tie or clone weights between output and input (adaptive) softmax layers
                self._tie_or_clone_weights(self.crit.out_layers[i], self.transformer.word_emb.emb_layers[i])
        if self.config.tie_projs:
            for i, tie_proj in enumerate(self.config.tie_projs):
                if tie_proj and self.config.div_val == 1 and self.config.d_model != self.config.d_embed:
                    if self.config.torchscript:
                        # Clone weights if using torchscript
                        self.crit.out_projs[i] = nn.Parameter(self.transformer.word_emb.emb_projs[0].clone())
                    else:
                        # Assign weights directly
                        self.crit.out_projs[i] = self.transformer.word_emb.emb_projs[0]
                elif tie_proj and self.config.div_val != 1:
                    if self.config.torchscript:
                        # Clone weights if using torchscript
                        self.crit.out_projs[i] = nn.Parameter(self.transformer.word_emb.emb_projs[i].clone())
                    else:
                        # Assign weights directly
                        self.crit.out_projs[i] = self.transformer.word_emb.emb_projs[i]

    def reset_memory_length(self, mem_len):
        # Reset the memory length for the transformer model
        self.transformer.reset_memory_length(mem_len)

    def init_mems(self, bsz):
        # Initialize memories for the transformer model with batch size bsz
        return self.transformer.init_mems(bsz)

    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TransfoXLLMHeadModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的前向传播方法，接受以下参数：

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的词索引序列，类型为可选的长整型张量
        mems: Optional[List[torch.FloatTensor]] = None,  # 可选的记忆列表，每个元素是浮点数张量
        head_mask: Optional[torch.FloatTensor] = None,  # 可选的注意力头掩码，浮点数张量
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 可选的输入嵌入表示，浮点数张量
        labels: Optional[torch.LongTensor] = None,  # 可选的标签，长整型张量
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，布尔值
    ) -> Union[Tuple, TransfoXLLMHeadModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # Determine whether to use the return dictionary from the function argument or the model's configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Determine batch size (`bsz`) and target sequence length (`tgt_len`) based on input_ids or inputs_embeds
        if input_ids is not None:
            bsz, tgt_len = input_ids.size(0), input_ids.size(1)
        elif inputs_embeds is not None:
            bsz, tgt_len = inputs_embeds.size(0), inputs_embeds.size(1)
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Pass inputs to the transformer model and retrieve outputs
        transformer_outputs = self.transformer(
            input_ids,
            mems=mems,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract the last hidden states from transformer_outputs
        last_hidden = transformer_outputs[0]
        # Predicted hidden states for the target length of the sequence
        pred_hid = last_hidden[:, -tgt_len:]

        # Adjust labels to prevent NaN loss during backward pass
        if labels is not None:
            # Check if all labels are -100 (masked), then modify to prevent NaN during loss computation
            miss_valid_label = labels[0, 1:].sum() == (labels.size(1) - 1) * -100
            if miss_valid_label:
                # Set a valid label (e.g., EOS token) to prevent NaN in loss calculation
                labels[0, 1] = self.config.eos_token_id

        # Compute softmax output based on predicted hidden states and labels
        softmax_output = self.crit(pred_hid, labels)
        # Reshape softmax output into a tensor of shape (batch_size, tgt_len, vocab_size) if labels are None
        prediction_scores = softmax_output.view(bsz, tgt_len, -1) if labels is None else ()

        # Compute losses if labels are provided
        if labels is not None:
            losses = softmax_output.view(bsz, tgt_len - 1)
            # Exclude padding tokens (-100) from loss calculation
            loss = losses[losses != 0].mean()
        else:
            losses, loss = None, None

        # Return outputs based on return_dict and trainer_compatible settings
        if not return_dict:
            if self.trainer_compatible:
                output = (prediction_scores, losses) if losses is not None else (prediction_scores,)
                output += transformer_outputs[1:]
                return ((loss,) + output) if loss is not None else output
            else:
                output = (prediction_scores, *transformer_outputs[1:])
                output = ((losses,) + output) if losses is not None else output
                return (output + (loss,)) if loss is not None else output

        # Return TransfoXLLMHeadModelOutput containing loss, prediction_scores, and other transformer outputs
        return TransfoXLLMHeadModelOutput(
            loss=loss,
            prediction_scores=prediction_scores,
            losses=losses,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    # 获取模型的输出嵌入
    def get_output_embeddings(self):
        """Double-check if you are using adaptive softmax."""
        # 如果使用了自适应softmax，返回输出层对象
        if self.sample_softmax > 0:
            return self.out_layer
        else:
            # 否则返回临界评估器的最后一层输出
            return self.crit.out_layers[-1]

    # 为生成准备输入
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **model_kwargs):
        inputs = {}

        # 如果过去的键值已在模型参数中定义，则使用它以加快解码速度
        if past_key_values:
            inputs["mems"] = past_key_values
            # 取最后一个输入ID，并扩展维度以匹配模型预期的形状
            inputs["input_ids"] = input_ids[:, -1].unsqueeze(-1)
        else:
            inputs["input_ids"] = input_ids

        return inputs

    # 调整截止点大小
    def _resize_cutoffs(self, new_num_tokens, new_emb_size, new_embedding_shapes, layer):
        # 调用父类方法以获取新的截止点
        new_cutoffs = super()._resize_cutoffs(new_num_tokens, new_emb_size, new_embedding_shapes, layer)

        # 更新临界评估器的截止点和token数
        self.crit.cutoffs = new_cutoffs
        self.crit.cutoff_ends = [0] + new_cutoffs
        self.crit.n_token = new_num_tokens

    # 重新排序缓存
    @staticmethod
    def _reorder_cache(mems: List[torch.Tensor], beam_idx: torch.Tensor) -> List[torch.Tensor]:
        """
        This function is used to re-order the `mems` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `mems` with the correct beam_idx at every
        generation step.
        """
        # 对于每一层的过去缓存，根据beam_idx重新排序以匹配生成步骤
        return [layer_past.index_select(1, beam_idx.to(layer_past.device)) for layer_past in mems]
@add_start_docstrings(
    """
    The Transformer-XL Model transformer with a sequence classification head on top (linear layer).

    [`TransfoXLForSequenceClassification`] uses the last token in order to do the classification, as other causal
    models (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    TRANSFO_XL_START_DOCSTRING,
)
class TransfoXLForSequenceClassification(TransfoXLPreTrainedModel):
    """
    Transformer-XL模型的序列分类器，顶部带有线性层。

    [`TransfoXLForSequenceClassification`] 使用最后一个token进行分类，类似于其他因果模型（例如GPT-1）。

    由于它在最后一个token上进行分类，因此需要知道最后一个token的位置。如果配置中定义了`pad_token_id`，则在每一行中找到不是填充token的最后一个token。如果未定义`pad_token_id`，则简单地取批次中每一行的最后一个值。当传递`inputs_embeds`而不是`input_ids`时，无法猜测填充token，因此执行相同操作（取批次中每一行的最后一个值）。
    """
    
    def __init__(self, config):
        """
        初始化方法，设置模型配置。

        Args:
            config (:class:`~transformers.TransfoXLConfig`):
                模型的配置对象，包含了模型的各种参数和超参数。
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = TransfoXLModel(config)
        self.score = nn.Linear(config.d_embed, self.num_labels, bias=False)
        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TransfoXLSequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        mems: Optional[List[torch.FloatTensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        前向传播方法。

        Args:
            input_ids (:obj:`torch.LongTensor`, 可选):
                输入token的ID张量。
            mems (:obj:`List[torch.FloatTensor]`, 可选):
                Transformer-XL的记忆（memory）部分，用于长序列训练。
            head_mask (:obj:`torch.FloatTensor`, 可选):
                头部的掩码张量，用于控制层的注意力权重。
            inputs_embeds (:obj:`torch.FloatTensor`, 可选):
                输入的嵌入张量，替代input_ids。
            labels (:obj:`torch.LongTensor`, 可选):
                分类任务的标签张量。
            output_attentions (:obj:`bool`, 可选):
                是否输出注意力权重。
            output_hidden_states (:obj:`bool`, 可选):
                是否输出隐藏状态。
            return_dict (:obj:`bool`, 可选):
                是否返回字典格式的输出。

        Returns:
            :class:`~transformers.modeling_transfo_xl.TransfoXLSequenceClassifierOutputWithPast`:
                输出对象，包含分类器的结果和可能的附加信息。
        """
        pass  # 实际前向传播逻辑在这里未给出，但这是一个定义前向传播的占位符方法
```