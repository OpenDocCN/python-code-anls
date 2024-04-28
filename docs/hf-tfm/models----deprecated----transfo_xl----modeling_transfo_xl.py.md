# `.\models\deprecated\transfo_xl\modeling_transfo_xl.py`

```py
# 设置文件编码为utf-8
# 2018年谷歌AI、谷歌Brain和卡耐基梅隆大学的作者以及HuggingFace Inc.团队的版权声明
# 版权所有2018年，美国NVIDIA公司保留所有权利
#
# 根据Apache许可，版本2.0（“许可证”）进行许可；
# 除非符合许可证的规定，否则您不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，由许可证分发的软件是基于“原样”基础分发的
# 没有任何保证或条件，无论是明示的还是默示的。
# 请参阅许可证，了解特定语言下的权限和限制
"""
PyTorch Transformer XL模型。改编自https://github.com/kimiyoung/transformer-xl。特别是
https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py
"""
# 导入警告模块
import warnings
# 导入dataclasses模块中的dataclass装饰器
from dataclasses import dataclass
# 导入List、Optional、Tuple和Union类型提示
from typing import List, Optional, Tuple, Union
# 导入torch模块
import torch
# 从torch中导入nn模块
from torch import nn
# 从torch.nn中导入BCEWithLogitsLoss、CrossEntropyLoss和MSELoss
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
# 从....modeling_utils中导入PreTrainedModel
from ....modeling_utils import PreTrainedModel
# 从....utils中导入ModelOutput、add_code_sample_docstrings、add_start_docstrings、add_start_docstrings_to_model_forward和logging
from ....utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
# 从.configuration_transfo_xl中导入TransfoXLConfig
from .configuration_transfo_xl import TransfoXLConfig
# 从.modeling_transfo_xl_utilities中导入ProjectedAdaptiveLogSoftmax
from .modeling_transfo_xl_utilities import ProjectedAdaptiveLogSoftmax
# 从.logging模块获取日志器并命名为logger
logger = logging.get_logger(__name__)

# 用于文档的检查点
_CHECKPOINT_FOR_DOC = "transfo-xl-wt103"
# 用于文档的配置
_CONFIG_FOR_DOC = "TransfoXLConfig"

# Transformer XL预训练模型的存档列表
TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "transfo-xl-wt103",
    # 参见https://huggingface.co/models?filter=transfo-xl中的所有Transformer XL模型
]


# 建立从TensorFlow到PyTorch的映射
def build_tf_to_pytorch_map(model, config):
    """
    从TF到PyTorch的模块映射。这次我使用映射保持PyTorch模型尽可能与原始的PyTorch模型相同
    """
    # 创建一个空的TF到PyTorch的映射字典
    tf_to_pt_map = {}
    # 检查模型是否有"transformer"属性，如果有，则需要加载Adaptive Softmax
    if hasattr(model, "transformer"):
        # 更新字典映射，将TF模型中的一些参数映射到PyTorch模型中
        tf_to_pt_map.update(
            {
                "transformer/adaptive_softmax/cutoff_0/cluster_W": model.crit.cluster_weight,
                "transformer/adaptive_softmax/cutoff_0/cluster_b": model.crit.cluster_bias,
            }
        )
        # 遍历Adaptive Softmax中的输出层、投影层和是否共享参数
        for i, (out_l, proj_l, tie_proj) in enumerate(
            zip(model.crit.out_layers, model.crit.out_projs, config.tie_projs)
        ):
            layer_str = f"transformer/adaptive_softmax/cutoff_{i}/"
            # 如果要共享词嵌入权重，则更新映射
            if config.tie_word_embeddings:
                tf_to_pt_map.update({layer_str + "b": out_l.bias})
            else:
                # 抛出未实现的错误
                raise NotImplementedError
                # 我认为在TF代码中这个并没有实现
                # tf_to_pt_map.update({layer_str + "lookup_table": out_l.weight, layer_str + "b": out_l.bias})
            # 如果不共享投影层，则更新映射
            if not tie_proj:
                tf_to_pt_map.update({layer_str + "proj": proj_l})
        # 现在加载transformer的其余部分
        model = model.transformer

    # 处理词嵌入
    for i, (embed_l, proj_l) in enumerate(zip(model.word_emb.emb_layers, model.word_emb.emb_projs)):
        layer_str = f"transformer/adaptive_embed/cutoff_{i}/"
        tf_to_pt_map.update({layer_str + "lookup_table": embed_l.weight, layer_str + "proj_W": proj_l})

    # 处理transformer块
    for i, b in enumerate(model.layers):
        layer_str = f"transformer/layer_{i}/"
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

    # 处理相���位置偏置
    if config.untie_r:
        r_r_list = []
        r_w_list = []
        for b in model.layers:
            r_r_list.append(b.dec_attn.r_r_bias)
            r_w_list.append(b.dec_attn.r_w_bias)
    else:
        r_r_list = [model.r_r_bias]
        r_w_list = [model.r_w_bias]
    tf_to_pt_map.update({"transformer/r_r_bias": r_r_list, "transformer/r_w_bias": r_w_list})
    return tf_to_pt_map
def load_tf_weights_in_transfo_xl(model, config, tf_path):
    """Load tf checkpoints in a pytorch model"""
    # 导入必要的库
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        # 如果导入失败，输出错误信息并抛出异常
        logger.error(
            "Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    # 建立 TF 到 PyTorch 权重加载的映射关系
    tf_to_pt_map = build_tf_to_pytorch_map(model, config)

    # 从 TF 模型加载权重
    init_vars = tf.train.list_variables(tf_path)
    tf_weights = {}
    for name, shape in init_vars:
        # 输出日志信息，加载 TF 权重
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        tf_weights[name] = array

    for name, pointer in tf_to_pt_map.items():
        assert name in tf_weights
        array = tf_weights[name]
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model

        # 如果权重名称中包含"kernel"或"proj"，进行转置操作
        if "kernel" in name or "proj" in name:
            array = np.transpose(array)
        # 如果名称中包含"r_r_bias"或"r_w_bias"，并且指针列表长度大于1，进行处理
        if ("r_r_bias" in name or "r_w_bias" in name) and len(pointer) > 1:
            # 在此处我们将分割 TF 权重
            assert len(pointer) == array.shape[0]
            for i, p_i in enumerate(pointer):
                arr_i = array[i, ...]
                try:
                    assert p_i.shape == arr_i.shape
                except AssertionError as e:
                    e.args += (p_i.shape, arr_i.shape)
                    raise
                logger.info(f"Initialize PyTorch weight {name} for layer {i}")
                p_i.data = torch.from_numpy(arr_i)
        else:
            try:
                assert (
                    pointer.shape == array.shape
                ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
            except AssertionError as e:
                e.args += (pointer.shape, array.shape)
                raise
            logger.info(f"Initialize PyTorch weight {name}")
            pointer.data = torch.from_numpy(array)
        # 删除已匹配的 TF 权重
        tf_weights.pop(name, None)
        tf_weights.pop(name + "/Adam", None)
        tf_weights.pop(name + "/Adam_1", None)

    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}")
    return model


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super().__init__()

        self.demb = demb

        # 计算位置嵌入的频率
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        # 将频率作为缓冲区注册到模型中
        self.register_buffer("inv_freq", inv_freq)
    # 前向传播方法，接收位置序列和 batch size 作为输入参数
    def forward(self, pos_seq, bsz=None):
        # 根据位置序列和逆频率生成正弦输入
        sinusoid_inp = torch.outer(pos_seq, self.inv_freq)
        # 将正弦和余弦值连接起来形成位置编码
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        # 如果给定了 batch size
        if bsz is not None:
            # 将位置编码扩展为与 batch size 对应的大小，并在第二个维度增加一个维度
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        # 如果未给定 batch size
        else:
            # 在第二个维度增加一个维度
            return pos_emb[:, None, :]
# 定义 PositionwiseFF 类，用于实现位置感知的前馈神经网络
class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False, layer_norm_epsilon=1e-5):
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        # 定义前馈神经网络的核心部分
        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout)
        )

        # 添加 LayerNorm 层
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)

        self.pre_lnorm = pre_lnorm

    # 前向传播函数
    def forward(self, inp):
        if self.pre_lnorm:
            # 如果先做 LayerNorm，再进行前馈神经网络的计算
            core_out = self.CoreNet(self.layer_norm(inp))

            # 加上残差连接
            output = core_out + inp
        else:
            # 如果先进行前馈神经网络的计算
            core_out = self.CoreNet(inp)

            # 加上残差连接后再进行 LayerNorm 处理
            output = self.layer_norm(inp + core_out)

        return output


# 定义 RelPartialLearnableMultiHeadAttn 类，用于实现可学习部分相关性的多头注意力机制
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

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        # 定义 QKV 矩阵的线性变换
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        # 定义 Dropout 层
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)

        # 定义输出转换线性层
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        # 添加 LayerNorm 层
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)

        self.scale = 1 / (d_head**0.5)

        self.pre_lnorm = pre_lnorm

        if r_r_bias is None or r_w_bias is None:  # Biases are not shared
            # 如果 r_r_bias 和 r_w_bias 未提供，则使用 nn.Parameter 创建可学习参数
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        else:
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias

        # 定义与相对��置编码相关的线性层
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    # 定义相对偏移函数，用于将矩阵向上对角平移
    def _rel_shift(self, x):
        zero_pad_shape = (x.size(0), 1) + x.size()[2:]
        zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded_shape = (x.size(1) + 1, x.size(0)) + x.size()[2:]
        x_padded = x_padded.view(*x_padded_shape)

        x = x_padded[1:].view_as(x)

        return x

# 定义 RelPartialLearnableDecoderLayer 类
class RelPartialLearnableDecoderLayer(nn.Module):
    # 初始化函数，定义了Transformer Decoder层的结构和参数
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, layer_norm_epsilon=1e-5, **kwargs):
        # 调用父类的初始化方法
        super().__init__()
        
        # 创建相对位置感知的多头注意力层对象
        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, layer_norm_epsilon=layer_norm_epsilon, **kwargs
        )
        
        # 创建位置前馈神经网络层对象
        self.pos_ff = PositionwiseFF(
            d_model, d_inner, dropout, pre_lnorm=kwargs.get("pre_lnorm"), layer_norm_epsilon=layer_norm_epsilon
        )

    # 前向传播函数，定义了Decoder层的前向传播过程
    def forward(self, dec_inp, r, dec_attn_mask=None, mems=None, head_mask=None, output_attentions=False):
        # 使用相对位置感知的多头注意力层处理Decoder输入和记忆，返回处理结果
        attn_outputs = self.dec_attn(
            dec_inp,
            r,
            attn_mask=dec_attn_mask,
            mems=mems,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        
        # 使用位置前馈神经网络层处理注意力层的输出，得到最终输出
        ff_output = self.pos_ff(attn_outputs[0])

        # 将位置前馈神经网络层的输出和注意力层的各项注意力权重组成列表返回
        outputs = [ff_output] + attn_outputs[1:]

        return outputs
class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, sample_softmax=False):
        # 初始化函数，接受标记个数、嵌入维度、投影维度、截断点、除数，默认值是否采样 Softmax
        super().__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        # 将截断点与标记总数组成截断列表
        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        # 嵌入缩放因子
        self.emb_scale = d_proj**0.5

        # 截断列表起止点
        self.cutoff_ends = [0] + self.cutoffs

        # 嵌入层和投影层
        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            # 如果除数为 1，添加一个嵌入层
            self.emb_layers.append(nn.Embedding(n_token, d_embed, sparse=sample_softmax > 0))
            # 如果投影维度不等于嵌入维度，添加一个投影层参数
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_embed)))
        else:
            # 如果除数不为 1，对每个截断点之间的部分添加嵌入层和投影层参数
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val**i)
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            # 如果除数为 1，直接使用第一个嵌入层
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                # 如果投影维度不等于嵌入维度，进行投影
                embed = nn.functional.linear(embed, self.emb_projs[0])
        else:
            # 如果除数不为 1
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                # 创建掩码，标记对应每个截断点的位置
                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                # 根据掩码选取对应位置的标记，并减去当前截断点位置
                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                # 使用对应的嵌入层和投影层���行嵌入和投影
                emb_i = self.emb_layers[i](inp_i)
                emb_i = nn.functional.linear(emb_i, self.emb_projs[i])

                # 将结果复制到对应位置
                emb_flat.index_copy_(0, indices_i, emb_i)

            embed_shape = inp.size() + (self.d_proj,)
            # 将结果变形成与输入相同的形状
            embed = emb_flat.view(embed_shape)

        # 缩放嵌入结果
        embed.mul_(self.emb_scale)

        return embed


class TransfoXLPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TransfoXLConfig
    load_tf_weights = load_tf_weights_in_transfo_xl
    base_model_prefix = "transformer"

    def _init_weight(self, weight):
        # 根据配置初始化权重
        if self.config.init == "uniform":
            nn.init.uniform_(weight, -self.config.init_range, self.config.init_range)
        elif self.config.init == "normal":
            nn.init.normal_(weight, 0.0, self.config.init_std)

    def _init_bias(self, bias):
        # 初始化偏置为 0
        nn.init.constant_(bias, 0.0)
    def _init_weights(self, m):
        # 初始化权重
        classname = m.__class__.__name__
        查找类名中是否包含"Linear"字符串
        if classname.find("Linear") != -1:
            检测是否具有权重属性并且权重不为空
            if hasattr(m, "weight") and m.weight is not None:
                # 对权重进行初始化
                self._init_weight(m.weight)
            检测是否具有偏置属性并且偏置不为空
            if hasattr(m, "bias") and m.bias is not None:
                # 对偏置进行初始化
                self._init_bias(m.bias)
        查找类名中是否包含"AdaptiveEmbedding"字符串
        elif classname.find("AdaptiveEmbedding") != -1:
            查找是否具有emb_projs属性
            if hasattr(m, "emb_projs"):
                遍历emb_projs属性，对每个不为空的元素进行正态分布初始化
                for i in range(len(m.emb_projs)):
                    if m.emb_projs[i] is not None:
                        nn.init.normal_(m.emb_projs[i], 0.0, self.config.proj_init_std)
        查找类名中是否包含"Embedding"字符串
        elif classname.find("Embedding") != -1:
            检测是否具有权重属性
            if hasattr(m, "weight"):
                # 对权重进行初始化
                self._init_weight(m.weight)
        查找类名中是否包含"ProjectedAdaptiveLogSoftmax"字符串
        elif classname.find("ProjectedAdaptiveLogSoftmax") != -1:
            检测是否具有cluster_weight属性并且cluster_weight不为空
            if hasattr(m, "cluster_weight") and m.cluster_weight is not None:
                # 对cluster_weight进行初始化
                self._init_weight(m.cluster_weight)
            检测是否具有cluster_bias属性并且cluster_bias不为空
            if hasattr(m, "cluster_bias") and m.cluster_bias is not None:
                # 对cluster_bias进行初始化
                self._init_bias(m.cluster_bias)
            检测是否具有out_projs属性
            if hasattr(m, "out_projs"):
                遍历out_projs属性，对每个不为空的元素进行正态分布初始化
                for i in range(len(m.out_projs)):
                    if m.out_projs[i] is not None:
                        nn.init.normal_(m.out_projs[i], 0.0, self.config.proj_init_std)
        查找类名中是否包含"LayerNorm"字符串
        elif classname.find("LayerNorm") != -1:
            检测是否具有权重属性
            if hasattr(m, "weight"):
                # 对权重进行初始化，均值为1.0，标准差为self.config.init_std
                nn.init.normal_(m.weight, 1.0, self.config.init_std)
            检测是否具有偏置属性并且偏置不为空
            if hasattr(m, "bias") and m.bias is not None:
                # 对偏置进行初始化
                self._init_bias(m.bias)
        else:
            检测是否具有r_emb属性
            if hasattr(m, "r_emb"):
                # 对r_emb进行初��化
                self._init_weight(m.r_emb)
            检测是否具有r_w_bias属性
            if hasattr(m, "r_w_bias"):
                # 对r_w_bias进行初始化
                self._init_weight(m.r_w_bias)
            检测是否具有r_r_bias属性
            if hasattr(m, "r_r_bias"):
                # 对r_r_bias进行初始化
                self._init_weight(m.r_r_bias)
            检测是否具有r_bias属性
            if hasattr(m, "r_bias"):
                # 对r_bias进行初始化
                self._init_bias(m.r_bias)
    # 调整模型的输入标记嵌入矩阵的大小，如果 new_num_tokens != config.vocab_size，则调整权重嵌入以确保它们绑定在一起
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
        # 如果需要，获取基本模型
        base_model = getattr(self, self.base_model_prefix, self)  # get the base model if needed

        # 如果 new_num_tokens 为空，返回输入标记的嵌入层
        if new_num_tokens is None:
            return self.get_input_embeddings()

        # 获取新的标记数量和所在层
        new_num_tokens_layer, layer = self._get_new_num_tokens_layer(new_num_tokens, layer)
        # 确保新的嵌入层大小大于0
        assert new_num_tokens_layer > 0, "The size of the new embedding layer cannot be 0 or less"
        # 调整模型的标记嵌入，返回调整后的模型嵌入
        model_embeds = base_model._resize_token_embeddings(new_num_tokens_layer, layer)

        # 更新基本模型和当前模型配置
        self.config.vocab_size = new_num_tokens
        base_model.vocab_size = new_num_tokens
        base_model.n_token = new_num_tokens

        # 获取新的嵌入形状
        new_embedding_shapes = self._get_embedding_shapes()
        # 调整新的截止点
        self._resize_cutoffs(new_num_tokens, new_num_tokens_layer, new_embedding_shapes, layer)

        # 如果需要，重新绑定权重
        self.tie_weights()

        return model_embeds

    # 获取新的标记数量和所在层
    def _get_new_num_tokens_layer(self, new_num_tokens, layer):
        embeddings = self.get_input_embeddings()
        if layer == -1:
            layer = len(embeddings.emb_layers) - 1
        assert 0 <= layer <= len(embeddings.emb_layers) - 1

        new_num_tokens_layer = (
            new_num_tokens
            - sum([emb.weight.shape[0] for emb in embeddings.emb_layers[:layer]])
            - sum([emb.weight.shape[0] for emb in embeddings.emb_layers[layer + 1 :]])
        )
        return new_num_tokens_layer, layer

    # 获取嵌入形状
    def _get_embedding_shapes(self):
        embeddings = self.get_input_embeddings()
        return [emb.weight.shape[0] for emb in embeddings.emb_layers]
    # 调整模型中某一层的词嵌入大小
    def _resize_token_embeddings(self, new_num_tokens, layer=-1):
        # 获取模型输入的词嵌入
        embeddings = self.get_input_embeddings()
        # 如果新的词数量为None，则返回当前词嵌入
        if new_num_tokens is None:
            return embeddings
        # 调整词嵌入层的大小
        new_embeddings_layer = self._get_resized_embeddings(embeddings.emb_layers[layer], new_num_tokens)
        embeddings.emb_layers[layer] = new_embeddings_layer
        # 更新模型输入的词嵌入
        self.set_input_embeddings(embeddings)
        # 返回调整后的词嵌入
        return self.get_input_embeddings()
    
    # 调整模型中某一层的截断点
    def _resize_cutoffs(self, new_num_tokens, new_emb_size, new_embedding_shapes, layer):
        # 获取模型输入的词嵌入
        embeddings = self.get_input_embeddings()
        # 遍历指定层的截断点，更新值
        for i in range(layer, len(embeddings.cutoffs)):
            embeddings.cutoffs[i] = sum(new_embedding_shapes[: i + 1])
        # 更新截断点结束值和词数量
        embeddings.cutoff_ends = [0] + embeddings.cutoffs
        embeddings.n_token = new_num_tokens
        # 更新模型配置中的截断点
        self.config.cutoffs = embeddings.cutoffs[:-1]
        # 返回更新后的截断点
        return embeddings.cutoffs
# 使用 dataclass 装饰器定义 TransfoXLModelOutput 类，该类是模型输出的基类，可能包含过去的键/值（以加速顺序解码）
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

    # 定义类属性
    last_hidden_state: torch.FloatTensor
    mems: List[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

# 使用 dataclass 装饰器定义 TransfoXLSequenceClassifierOutputWithPast 类，该类是句子分类模型输出的基类
class TransfoXLSequenceClassifierOutputWithPast(ModelOutput):
    """
    Base class for outputs of sentence classification models.
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
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

    # 定义 loss 变量，类型为 torch.FloatTensor，可选的返回值，当提供了 `labels` 参数时返回
    loss: Optional[torch.FloatTensor] = None
    # 定义 logits 变量，类型为 torch.FloatTensor，用于存储分类（或回归，如果 config.num_labels==1）得分（SoftMax 之前）
    logits: torch.FloatTensor = None
    # 定义 mems 变量，类型为 List[torch.FloatTensor]，长度为 config.n_layers，用于存储预先计算的隐藏状态（注意力块中的键和值）。可以用于加速顺序解码。
    mems: List[torch.FloatTensor] = None
    # 定义 hidden_states 变量，类型为 Tuple[torch.FloatTensor]，可选的返回值，当传递 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回。存储模型在每层输出的隐藏状态，以及初始嵌入输出。
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义 attentions 变量，类型为 Tuple[torch.FloatTensor]，可选的返回值，当传递 `output_attentions=True` 或 `config.output_attentions=True` 时返回。存储注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# TransfoXLLMHeadModelOutput类继承自ModelOutput类，表示模型的输出
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

    # 定义类的属性，表示模型输出的各个部分
    losses: Optional[torch.FloatTensor] = None
    prediction_scores: torch.FloatTensor = None
    mems: List[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss: Optional[torch.FloatTensor] = None

    # logits是预测分数，即语言模型头部的输出（对每个词汇标记进行SoftMax处理后的分数）
    @property
    def logits(self):
        # prediction_scores是自适应SoftMax的输出，详见`modeling_transfo_xl_utilities`文件。
        # 由于自适应SoftMax返回的是对数SoftMax值，所以严格来说`self.prediction_scores`不完全是`logits`，
        # 但其行为与`logits`一致。
        return self.prediction_scores


# TRANSFO_XL_START_DOCSTRING是一个多行字符串，用于记录TransfoXL模型的基本信息
TRANSFO_XL_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    # 这个模型也是 PyTorch 的 torch.nn.Module 子类。
    # 可以像使用普通的 PyTorch 模块一样使用它，并参考 PyTorch 文档了解有关一般用法和行为的所有内容。
    
    # 参数：
    #     config ([`TransfoXLConfig`]): 包含模型所有参数的模型配置类。
    #         使用配置文件初始化不会加载与模型关联的权重，只加载配置。
    #         查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
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
# 创建 TransfoXLModel 类，继承自 TransfoXLPreTrainedModel 类
class TransfoXLModel(TransfoXLPreTrainedModel):
    # 初始化函数，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 从配置中获取词汇表大小
        self.n_token = config.vocab_size

        # 从配置中获取嵌入维度
        self.d_embed = config.d_embed
        # 从配置中获取模型维度
        self.d_model = config.d_model
        # 从配置中获取注意力头数
        self.n_head = config.n_head
        # 从配置中获取头部维度
        self.d_head = config.d_head

        # 创建自适应嵌入层，用于处理不同频率的词
        self.word_emb = AdaptiveEmbedding(
            config.vocab_size, config.d_embed, config.d_model, config.cutoffs, div_val=config.div_val
        )

        # 添加一个Dropout层
        self.drop = nn.Dropout(config.dropout)

        # 从配置中获取层数
        self.n_layer = config.n_layer
        # 从配置中获取记忆长度
        self.mem_len = config.mem_len
        # 从配置中获取注意力类型
        self.attn_type = config.attn_type

        # 如果不是解耦的注意力机制
        if not config.untie_r:
            # 初始化相对位置编码参数
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))

        # 初始化一个ModuleList，用于存放Transformer层
        self.layers = nn.ModuleList()
        # 如果注意力类型是默认类型（0）
        if config.attn_type == 0:  # the default attention
            # 根据层数，添加相应数量的Transformer层
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
        else:  # learnable embeddings and absolute embeddings are not used in our pretrained checkpoints
            # 如果不是默认类型，则抛出未实现错误
            raise NotImplementedError  # Removed them to avoid maintaining dead code

        # 从配置中获取是否使用相同长度
        self.same_length = config.same_length
        # 从配置中获取固定长度
        self.clamp_len = config.clamp_len

        # 如果是默认类型的注意力
        if self.attn_type == 0:  # default attention
            # 初始化位置编码
            self.pos_emb = PositionalEmbedding(self.d_model)
        else:  # learnable embeddings and absolute embeddings
            # 如果不是默认类型的注意力，则抛出未实现错误
            raise NotImplementedError  # Removed these to avoid maintaining dead code - They are not used in our pretrained checkpoint

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.word_emb

    # 设置输入嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.word_emb = new_embeddings

    # 向后兼容函数
    def backward_compatible(self):
        self.sample_softmax = -1

    # 重置记忆长度
    def reset_memory_length(self, mem_len):
        self.mem_len = mem_len

    # 剪枝头部
    def _prune_heads(self, heads):
        logger.info("Head pruning is not implemented for Transformer-XL model")
        # 未实现头部剪枝，保留函数结构
        pass
    # 初始化记忆（memory）张量列表，用于存储 Transformer-XL 模型的历史隐藏状态
    def init_mems(self, bsz):
        # 如果记忆长度大于 0
        if self.mem_len > 0:
            # 初始化一个空列表 mems 用于存储记忆张量
            mems = []
            # 获取模型参数的一个示例，用于指定数据类型和设备
            param = next(self.parameters())
            # 遍历每一层的隐藏状态
            for i in range(self.n_layer):
                # 创建一个形状为 (mem_len, bsz, d_model) 的全零张量，表示一个空的记忆张量
                empty = torch.zeros(self.mem_len, bsz, self.config.d_model, dtype=param.dtype, device=param.device)
                # 将空的记忆张量添加到记忆列表 mems 中
                mems.append(empty)

            return mems  # 返回初始化后的记忆列表
        else:
            return None  # 如果记忆长度为 0，则返回 None

    # 更新记忆（memory）张量
    def _update_mems(self, hids, mems, mlen, qlen):
        # 如果 mems 为 None，则不需要处理
        if mems is None:
            return None

        # mems 不为 None
        assert len(hids) == len(mems), "len(hids) != len(mems)"  # 断言 hids 和 mems 的长度相等

        # 可以将 `mlen + qlen` 步的隐藏状态缓存到 mems 中
        with torch.no_grad():
            new_mems = []  # 存储更新后的记忆张量
            # 计算需要更新的步数范围的起始和结束索引
            end_idx = mlen + max(0, qlen)
            beg_idx = max(0, end_idx - self.mem_len)
            # 遍历每一层的隐藏状态和对应的记忆张量
            for i in range(len(hids)):
                # 将当前隐藏状态和记忆张量在第一维进行拼接，得到一个新的张量
                cat = torch.cat([mems[i], hids[i]], dim=0)
                # 在新的张量中截取指定范围的片段，并且将梯度信息分离（detach）
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems  # 返回更新后的记忆张量列表

    # Transformer-XL 模型的前向传播函数
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
# 用指定的文档字符串初始化 TransfoXLLMHeadModel 类，该类在顶部带有一个语言建模头部（自适应 softmax 权重绑定到自适应输入嵌入）
@add_start_docstrings(
    """
    The Transformer-XL Model with a language modeling head on top (adaptive softmax with weights tied to the adaptive
    input embeddings)
    """,
    TRANSFO_XL_START_DOCSTRING,
)
class TransfoXLLMHeadModel(TransfoXLPreTrainedModel):
    # 定义用于绑定权重的键列表
    _tied_weights_keys = [r"crit\.out_projs\.\d+", r"crit\.out_layers\.\d+\.weight"]

    # 初始化 TransfoXLLMHeadModel 类
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 Transformer-XL 模型实例
        self.transformer = TransfoXLModel(config)
        # 获取 sample_softmax 参数
        self.sample_softmax = config.sample_softmax
        # 获取 trainer_compatible 参数
        self.trainer_compatible = getattr(config, "trainer_compatible", False)

        # 如果不兼容训练器模式，发出警告
        if not self.trainer_compatible:
            warnings.warn(
                "The output of TransfoXL will be updated in v5 to support a single loss as first argument. In order "
                "to use that updated output, please specify `trainer_compatible=True` as your configuration"
                " attribute.",
                DeprecationWarning,
            )

        # 断言 sample_softmax 参数小于等于 0，因为 softmax 抽样尚未实现
        assert self.sample_softmax <= 0, (
            "Sampling from the softmax is not implemented yet. Please look at issue: #3310:"
            " https://github.com/huggingface/transformers/issues/3310"
        )

        # 创建 ProjectedAdaptiveLogSoftmax 类实例
        self.crit = ProjectedAdaptiveLogSoftmax(
            config.vocab_size, config.d_embed, config.d_model, config.cutoffs, div_val=config.div_val
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 用于确保输出和输入（自适应）softmax 权重被绑定的方法
    def tie_weights(self):
        """
        Run this to be sure output and input (adaptive) softmax weights are tied
        """

        # 如果配置中设置了绑定单词嵌入权重
        if self.config.tie_word_embeddings:
            # 遍历 crit.out_layers，将其权重与 transformer.word_emb.emb_layers 的相应层进行绑定或克隆
            for i in range(len(self.crit.out_layers)):
                self._tie_or_clone_weights(self.crit.out_layers[i], self.transformer.word_emb.emb_layers[i])
        
        # 如果配置中设置了绑定投影
        if self.config.tie_projs:
            # 遍历 tie_projs 列表
            for i, tie_proj in enumerate(self.config.tie_projs):
                # 如果 tie_proj 为真，并且 div_val 等于 1，且 d_model 不等于 d_embed
                if tie_proj and self.config.div_val == 1 and self.config.d_model != self.config.d_embed:
                    # 克隆 transformer.word_emb.emb_projs[0] 的参数，并将其赋值给 crit.out_projs[i]
                    if self.config.torchscript:
                        self.crit.out_projs[i] = nn.Parameter(self.transformer.word_emb.emb_projs[0].clone())
                    else:
                        self.crit.out_projs[i] = self.transformer.word_emb.emb_projs[0]
                
                # 如果 tie_proj 为真，并且 div_val 不等于 1
                elif tie_proj and self.config.div_val != 1:
                    # 克隆 transformer.word_emb.emb_projs[i] 的参数，并将其赋值给 crit.out_projs[i]
                    if self.config.torchscript:
                        self.crit.out_projs[i] = nn.Parameter(self.transformer.word_emb.emb_projs[i].clone())
                    else:
                        self.crit.out_projs[i] = self.transformer.word_emb.emb_projs[i]

    # 重置记忆长度
    def reset_memory_length(self, mem_len):
        self.transformer.reset_memory_length(mem_len)

    # 初始化记忆
    def init_mems(self, bsz):
        return self.transformer.init_mems(bsz)

    # 通过模型前向传播添加开始文档字符串
    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TransfoXLLMHeadModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token IDs，类型为 LongTensor，可选参数，默认为 None
        mems: Optional[List[torch.FloatTensor]] = None,  # 用于存储 Transformer 模型历史记忆的列表，类型为 FloatTensor 的列表，可选参数，默认为 None
        head_mask: Optional[torch.FloatTensor] = None,  # 用于指定 Transformer 模型中每个注意力头的掩码，类型为 FloatTensor，可选参数，默认为 None
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量，类型为 FloatTensor，可选参数，默认为 None
        labels: Optional[torch.LongTensor] = None,  # 预测的标签，类型为 LongTensor，可选参数，默认为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，类型为布尔值，可选参数，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，类型为布尔值，可选参数，默认为 None
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，类型为布尔值，可选参数，默认为 None
    ) -> Union[Tuple, TransfoXLLMHeadModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # 如果 return_dict 参数不为 None，则使用其值；否则使用配置中的 use_return_dict 参数值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果传入 input_ids，则确定 batch size 和目标长度
        if input_ids is not None:
            bsz, tgt_len = input_ids.size(0), input_ids.size(1)
        # 如果传入 inputs_embeds，则确定 batch size 和目标长度
        elif inputs_embeds is not None:
            bsz, tgt_len = inputs_embeds.size(0), inputs_embeds.size(1)
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 使用 Transformer 处理输入数据
        transformer_outputs = self.transformer(
            input_ids,
            mems=mems,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取最后一层隐藏状态，并取出预测部分
        last_hidden = transformer_outputs[0]
        pred_hid = last_hidden[:, -tgt_len:]

        # 如果传入 labels，则执行以下操作
        if labels is not None:
            # 避免所有标签都是 -100，导致反向传播时出错
            miss_valid_label = labels[0, 1:].sum() == (labels.size(1) - 1) * -100
            if miss_valid_label:
                # 设置一个 <EOS> 标记，以防止损失为 NaN
                labels[0, 1] = self.config.eos_token_id

        # 使用交叉熵计算预测与标签之间的损失
        softmax_output = self.crit(pred_hid, labels)
        # 将预测的分数重塑成三维张量形状
        prediction_scores = softmax_output.view(bsz, tgt_len, -1) if labels is None else ()

        # 如果传入 labels，则计算损失
        if labels is not None:
            losses = softmax_output.view(bsz, tgt_len - 1)
            # 避免将填充（-100）标记纳入损失值
            loss = losses[losses != 0].mean()
        else:
            losses, loss = None, None

        # 如果不返回字典，则根据是否兼容 PyTorch Lightning Trainer 返回相应格式的输出
        if not return_dict:
            if self.trainer_compatible:
                output = (prediction_scores, losses) if losses is not None else (prediction_scores,)
                output += transformer_outputs[1:]
                return ((loss,) + output) if loss is not None else output
            else:
                output = (prediction_scores, *transformer_outputs[1:])
                output = ((losses,) + output) if losses is not None else output
                return (output + (loss,)) if loss is not None else output

        # 返回 TransfoXLLMHeadModelOutput 类型的结果
        return TransfoXLLMHeadModelOutput(
            loss=loss,
            prediction_scores=prediction_scores,
            losses=losses,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    # 返回输出嵌入
    def get_output_embeddings(self):
        """Double-check if you are using adaptive softmax."""
        # 如果使用自适应 softmax，则返回输出层
        if self.sample_softmax > 0:
            return self.out_layer
        else:
            # 否则返回临界值输出层的最后一层
            return self.crit.out_layers[-1]

    # 为生成准备输入
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **model_kwargs):
        inputs = {}

        # 如果在模型kwargs中定义了过去的值，则使用它以加快解码速度
        if past_key_values:
            inputs["mems"] = past_key_values
            inputs["input_ids"] = input_ids[:, -1].unsqueeze(-1)
        else:
            inputs["input_ids"] = input_ids

        return inputs

    # 调整截止值
    def _resize_cutoffs(self, new_num_tokens, new_emb_size, new_embedding_shapes, layer):
        new_cutoffs = super()._resize_cutoffs(new_num_tokens, new_emb_size, new_embedding_shapes, layer)

        self.crit.cutoffs = new_cutoffs
        self.crit.cutoff_ends = [0] + new_cutoffs
        self.crit.n_token = new_num_tokens

    @staticmethod
    def _reorder_cache(mems: List[torch.Tensor], beam_idx: torch.Tensor) -> List[torch.Tensor]:
        """
        This function is used to re-order the `mems` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `mems` with the correct beam_idx at every
        generation step.
        """
        # 用于重排 `mems` 缓存，如果调用了 [`~PreTrainedModel.beam_search`] 或 [`~PreTrainedModel.beam_sample`]。这是为了确保在每个生成步骤中 `mems` 与正确的 beam_idx 匹配。
        return [layer_past.index_select(1, beam_idx.to(layer_past.device)) for layer_past in mems]
# 使用 add_start_docstrings 装饰器添加了模型的文档字符串说明
# Transformer-XL 模型带有顶部的序列分类头（线性层）

# 该类使用最后一个词作为分类的依据，与其他因果关系模型（如 GPT-1）一样。

# 由于它在最后一个词上进行分类，它需要知道最后一个词的位置。如果在配置中定义了 `pad_token_id`，则查找每行中不是填充标记的最后一个标记。
# 如果没有定义 `pad_token_id`，则简单地取每个批次行中的最后一个值。由于在传递 `inputs_embeds` 而不是 `input_ids` 时无法猜测填充标记，
# 它执行相同的操作（取每个批次行中的最后一个值）。

# 添加到 TransfoXLModel 的序列分类头层
class TransfoXLForSequenceClassification(TransfoXLPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 获取标签数量
        self.num_labels = config.num_labels
        # 创建 TransfoXLModel 模型
        self.transformer = TransfoXLModel(config)
        # 创建线性层
        self.score = nn.Linear(config.d_embed, self.num_labels, bias=False)
        # 初始化权重并应用最终处理
        self.post_init()

    # 使用 add_start_docstrings_to_model_forward 和 add_code_sample_docstrings 装饰器添加了模型前向传播函数的文档字符串说明
    # FORWARD 输入参数的文档字符串
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
```