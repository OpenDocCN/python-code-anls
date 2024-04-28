# `.\transformers\models\xlnet\modeling_xlnet.py`

```
# 设定文件编码为 UTF-8
# 版权声明
"""
 PyTorch XLNet 模型。
"""
# 导入警告模块
import warnings
# 导入数据类
from dataclasses import dataclass
# 导入类型提示相关模块
from typing import List, Optional, Tuple, Union

# 导入 PyTorch 模块
import torch
# 导入 PyTorch 中的神经网络模块
from torch import nn
# 导入 PyTorch 中的损失函数模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入激活函数相关模块
from ...activations import ACT2FN
# 导入模型工具函数模块
from ...modeling_utils import PoolerAnswerClass, PoolerEndLogits, PoolerStartLogits, PreTrainedModel, SequenceSummary
# 导入 PyTorch 工具函数模块
from ...pytorch_utils import apply_chunking_to_forward
# 导入工具函数模块
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入 XLNet 相关配置模块
from .configuration_xlnet import XLNetConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 文档中所用的检查点名称
_CHECKPOINT_FOR_DOC = "xlnet-base-cased"
# 文档中所用的配置名称
_CONFIG_FOR_DOC = "XLNetConfig"

# XLNet 预训练模型归档列表
XLNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "xlnet-base-cased",
    "xlnet-large-cased",
    # 查看所有 XLNet 模型：https://huggingface.co/models?filter=xlnet
]


# 构建从 TensorFlow 到 PyTorch 的映射函数
def build_tf_xlnet_to_pytorch_map(model, config, tf_weights=None):
    """
    A map of modules from TF to PyTorch. I use a map to keep the PyTorch model as identical to the original PyTorch
    model as possible.
    """

    # TF 到 PT 的映射字典
    tf_to_pt_map = {}

    # 如果模型包含 transformer 属性
    if hasattr(model, "transformer"):
        # 如果模型包含 lm_loss 属性
        if hasattr(model, "lm_loss"):
            # 将 lm_loss 的偏置加载进映射字典
            tf_to_pt_map["model/lm_loss/bias"] = model.lm_loss.bias
        # 如果模型包含 sequence_summary 属性且其权重在 TensorFlow 权重中
        if hasattr(model, "sequence_summary") and "model/sequnece_summary/summary/kernel" in tf_weights:
            # 将 sequence_summary 的权重和偏置加载进映射字典
            tf_to_pt_map["model/sequnece_summary/summary/kernel"] = model.sequence_summary.summary.weight
            tf_to_pt_map["model/sequnece_summary/summary/bias"] = model.sequence_summary.summary.bias
        # 如果模型包含 logits_proj 属性且在 TensorFlow 权重中存在与 fine-tuning 任务相关的权重
        if (
            hasattr(model, "logits_proj")
            and config.finetuning_task is not None
            and f"model/regression_{config.finetuning_task}/logit/kernel" in tf_weights
        ):
            # 将与 fine-tuning 任务相关的权重和偏置加载进映射字典
            tf_to_pt_map[f"model/regression_{config.finetuning_task}/logit/kernel"] = model.logits_proj.weight
            tf_to_pt_map[f"model/regression_{config.finetuning_task}/logit/bias"] = model.logits_proj.bias

        # 加载剩余的 transformer 部分
        model = model.transformer
    # Embeddings and output
    # 更新 TensorFlow 到 PyTorch 的映射，包括词嵌入表和掩码嵌入
    tf_to_pt_map.update(
        {
            "model/transformer/word_embedding/lookup_table": model.word_embedding.weight,
            "model/transformer/mask_emb/mask_emb": model.mask_emb,
        }
    )

    # Transformer blocks
    # 遍历 Transformer 模型的每个层，更新映射
    for i, b in enumerate(model.layer):
        # 每个层的字符串表示
        layer_str = f"model/transformer/layer_{i}/"
        tf_to_pt_map.update(
            {
                layer_str + "rel_attn/LayerNorm/gamma": b.rel_attn.layer_norm.weight,
                layer_str + "rel_attn/LayerNorm/beta": b.rel_attn.layer_norm.bias,
                layer_str + "rel_attn/o/kernel": b.rel_attn.o,
                layer_str + "rel_attn/q/kernel": b.rel_attn.q,
                layer_str + "rel_attn/k/kernel": b.rel_attn.k,
                layer_str + "rel_attn/r/kernel": b.rel_attn.r,
                layer_str + "rel_attn/v/kernel": b.rel_attn.v,
                layer_str + "ff/LayerNorm/gamma": b.ff.layer_norm.weight,
                layer_str + "ff/LayerNorm/beta": b.ff.layer_norm.bias,
                layer_str + "ff/layer_1/kernel": b.ff.layer_1.weight,
                layer_str + "ff/layer_1/bias": b.ff.layer_1.bias,
                layer_str + "ff/layer_2/kernel": b.ff.layer_2.weight,
                layer_str + "ff/layer_2/bias": b.ff.layer_2.bias,
            }
        )

    # Relative positioning biases
    # 如果解除相对位置偏差
    if config.untie_r:
        # 分别存储相对位置偏差
        r_r_list = []
        r_w_list = []
        r_s_list = []
        seg_embed_list = []
        # 遍历每个层
        for b in model.layer:
            r_r_list.append(b.rel_attn.r_r_bias)
            r_w_list.append(b.rel_attn.r_w_bias)
            r_s_list.append(b.rel_attn.r_s_bias)
            seg_embed_list.append(b.rel_attn.seg_embed)
    else:
        # 否则使用模型的相对位置偏差
        r_r_list = [model.r_r_bias]
        r_w_list = [model.r_w_bias]
        r_s_list = [model.r_s_bias]
        seg_embed_list = [model.seg_embed]
    # 更新映射，包括相对位置偏差和分段嵌入
    tf_to_pt_map.update(
        {
            "model/transformer/r_r_bias": r_r_list,
            "model/transformer/r_w_bias": r_w_list,
            "model/transformer/r_s_bias": r_s_list,
            "model/transformer/seg_embed": seg_embed_list,
        }
    )
    # 返回 TensorFlow 到 PyTorch 的映射
    return tf_to_pt_map
# 将 TensorFlow 模型的权重加载到 PyTorch 模型中
def load_tf_weights_in_xlnet(model, config, tf_path):
    # 尝试导入所需的库
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        # 如果导入失败，记录错误信息并抛出 ImportError
        logger.error(
            "Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    # 从 TensorFlow 模型加载权重
    init_vars = tf.train.list_variables(tf_path)
    # 初始化一个空字典来存储 TensorFlow 权重
    tf_weights = {}
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        # 使用 TensorFlow 函数加载变量值
        array = tf.train.load_variable(tf_path, name)
        # 将加载的变量值存储到字典中
        tf_weights[name] = array

    # 构建从 TensorFlow 到 PyTorch 权重的映射
    tf_to_pt_map = build_tf_xlnet_to_pytorch_map(model, config, tf_weights)

    # 遍历映射中的项目
    for name, pointer in tf_to_pt_map.items():
        logger.info(f"Importing {name}")
        # 如果名称不在 TensorFlow 权重中，则跳过
        if name not in tf_weights:
            logger.info(f"{name} not in tf pre-trained weights, skipping")
            continue
        # 获取 TensorFlow 权重数组
        array = tf_weights[name]
        # 如果权重名称包含特定字符串，则需要转置数组
        if "kernel" in name and ("ff" in name or "summary" in name or "logit" in name):
            logger.info("Transposing")
            array = np.transpose(array)
        # 如果指针是列表，则需要分割 TensorFlow 权重
        if isinstance(pointer, list):
            # 检查列表长度与数组的第一个维度长度是否匹配
            assert (
                len(pointer) == array.shape[0]
            ), f"Pointer length {len(pointer)} and array length {array.shape[0]} mismatched"
            # 遍历指针列表
            for i, p_i in enumerate(pointer):
                # 获取当前层的权重数组
                arr_i = array[i, ...]
                # 检查指针形状与数组形状是否匹配
                try:
                    assert (
                        p_i.shape == arr_i.shape
                    ), f"Pointer shape {p_i.shape} and array shape {arr_i.shape} mismatched"
                except AssertionError as e:
                    e.args += (p_i.shape, arr_i.shape)
                    raise
                logger.info(f"Initialize PyTorch weight {name} for layer {i}")
                # 将数组转换为 PyTorch 张量，并赋值给指针
                p_i.data = torch.from_numpy(arr_i)
        else:
            # 检查指针形状与数组形状是否匹配
            try:
                assert (
                    pointer.shape == array.shape
                ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
            except AssertionError as e:
                e.args += (pointer.shape, array.shape)
                raise
            logger.info(f"Initialize PyTorch weight {name}")
            # 将数组转换为 PyTorch 张量，并赋值给指针
            pointer.data = torch.from_numpy(array)
        # 从 TensorFlow 权重字典中移除当前处理的键
        tf_weights.pop(name, None)
        tf_weights.pop(name + "/Adam", None)
        tf_weights.pop(name + "/Adam_1", None)

    # 记录未复制到 PyTorch 模型的权重
    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}")
    # 返回加载了 TensorFlow 权重的 PyTorch 模型
    return model


class XLNetRelativeAttention(nn.Module):
```  
    # 定义初始化函数，接收一个配置对象
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()

        # 如果隐藏维度除以注意力头数量不是整数倍，抛出ValueError异常
        if config.d_model % config.n_head != 0:
            raise ValueError(
                f"The hidden size ({config.d_model}) is not a multiple of the number of attention "
                f"heads ({config.n_head}"
            )

        # 将配置参数赋值给对象属性
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.scale = 1 / (config.d_head**0.5)

        # 定义可学习的参数
        self.q = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.k = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.v = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.o = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.r = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))

        self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.r_s_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.seg_embed = nn.Parameter(torch.FloatTensor(2, self.n_head, self.d_head))

        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    # 定义剪枝注意力头的方法
    def prune_heads(self, heads):
        raise NotImplementedError

    # 定义相对偏移函数，用于形成相对注意力分数
    @staticmethod
    def rel_shift(x, klen=-1):
        """perform relative shift to form the relative attention score."""
        # 获取输入张量的形状
        x_size = x.shape

        # 将张量维度进行变换
        x = x.reshape(x_size[1], x_size[0], x_size[2], x_size[3])
        x = x[1:, ...]
        x = x.reshape(x_size[0], x_size[1] - 1, x_size[2], x_size[3])
        # x = x[:, 0:klen, :, :]
        x = torch.index_select(x, 1, torch.arange(klen, device=x.device, dtype=torch.long))

        # 返回变换后的张量
        return x

    # 定义带细分掩码的相对偏移函数
    @staticmethod
    def rel_shift_bnij(x, klen=-1):
        # 获取输入张量的形状
        x_size = x.shape

        # 将张量维度进行变换
        x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        x = x[:, :, 1:, :]
        x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        # Note: the tensor-slice form was faster in my testing than torch.index_select
        #       However, tracing doesn't like the nature of the slice, and if klen changes
        #       during the run then it'll fail, whereas index_select will be fine.
        x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        # x = x[:, :, :, :klen]

        # 返回变换后的张量
        return x

    # 定义相对注意力的核心函数
    def rel_attn_core(
        self,
        q_head,
        k_head_h,
        v_head_h,
        k_head_r,
        seg_mat=None,
        attn_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        """Core relative positional attention operations."""

        # content based attention score
        ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)

        # position based attention score
        bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        bd = self.rel_shift_bnij(bd, klen=ac.shape[3])

        # segment based attention score
        if seg_mat is None:
            ef = 0
        else:
            ef = torch.einsum("ibnd,snd->ibns", q_head + self.r_s_bias, self.seg_embed)
            ef = torch.einsum("ijbs,ibns->bnij", seg_mat, ef)

        # merge attention scores and perform masking
        attn_score = (ac + bd + ef) * self.scale
        if attn_mask is not None:
            # apply masking to attention scores
            if attn_mask.dtype == torch.float16:
                attn_score = attn_score - 65500 * torch.einsum("ijbn->bnij", attn_mask)
            else:
                attn_score = attn_score - 1e30 * torch.einsum("ijbn->bnij", attn_mask)

        # calculate attention probability using softmax
        attn_prob = nn.functional.softmax(attn_score, dim=3)
        attn_prob = self.dropout(attn_prob)

        # Mask heads if needed
        if head_mask is not None:
            attn_prob = attn_prob * torch.einsum("ijbn->bnij", head_mask)

        # attention output
        attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)

        if output_attentions:
            return attn_vec, torch.einsum("bnij->ijbn", attn_prob)

        return attn_vec

    def post_attention(self, h, attn_vec, residual=True):
        """Post-attention processing."""
        # post-attention projection (back to `d_model`)
        attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)

        attn_out = self.dropout(attn_out)
        if residual:
            attn_out = attn_out + h
        output = self.layer_norm(attn_out)

        return output

    def forward(
        self,
        h,
        g,
        attn_mask_h,
        attn_mask_g,
        r,
        seg_mat,
        mems=None,
        target_mapping=None,
        head_mask=None,
        output_attentions=False,
# 定义一个名为XLNetFeedForward的类，继承自nn.Module
class XLNetFeedForward(nn.Module):
    # 类的初始化方法，接受config作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 使用config中指定的参数创建一个LayerNorm层
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        # 使用config中指定的参数创建一个线性变换层
        self.layer_1 = nn.Linear(config.d_model, config.d_inner)
        # 使用config中指定的参数创建另一个线性变换层
        self.layer_2 = nn.Linear(config.d_inner, config.d_model)
        # 使用config中指定的参数创建一个丢弃层
        self.dropout = nn.Dropout(config.dropout)
        # 如果config中的ff_activation是str类型，则将其对应的函数从配置中选择，否则直接使用config中的函数
        if isinstance(config.ff_activation, str):
            self.activation_function = ACT2FN[config.ff_activation]
        else:
            self.activation_function = config.ff_activation

    # 定义前向传播方法，接受输入变量inp
    def forward(self, inp):
        # 将inp赋值给output变量
        output = inp
        # 对output执行layer_1操作
        output = self.layer_1(output)
        # 对output执行激活函数操作
        output = self.activation_function(output)
        # 对output执行丢弃操作
        output = self.dropout(output)
        # 对output执行layer_2操作
        output = self.layer_2(output)
        # 对output执行丢弃操作
        output = self.dropout(output)
        # 对output执行layer_norm操作，并加上原始输入inp
        output = self.layer_norm(output + inp)
        # 返回output
        return output


# 定义一个名为XLNetLayer的类，继承自nn.Module
class XLNetLayer(nn.Module):
    # 类的初始化方法，接受config作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个相对注意力层对象
        self.rel_attn = XLNetRelativeAttention(config)
        # 创建一个前馈网络对象
        self.ff = XLNetFeedForward(config)
        # 创建一个丢弃层对象
        self.dropout = nn.Dropout(config.dropout)
        # 将config中的chunk_size_feed_forward赋值给self.chunk_size_feed_forward
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 将1赋值给self.seq_len_dim
        self.seq_len_dim = 1

    # 定义前向传播方法，接受多个变量作为输入
    def forward(
        self,
        output_h,
        output_g,
        attn_mask_h,
        attn_mask_g,
        r,
        seg_mat,
        mems=None,
        target_mapping=None,
        head_mask=None,
        output_attentions=False,
    ):
        # 执行相对注意力层的前向传播
        outputs = self.rel_attn(
            output_h,
            output_g,
            attn_mask_h,
            attn_mask_g,
            r,
            seg_mat,
            mems=mems,
            target_mapping=target_mapping,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        # 获取输出的前两个值
        output_h, output_g = outputs[:2]
        # 如果output_g不为None
        if output_g is not None:
            # 对output_g执行apply_chunking_to_forward操作
            output_g = apply_chunking_to_forward(
                self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, output_g
            )
        # 对output_h执行apply_chunking_to_forward操作
        output_h = apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, output_h)
        # 将输出再次赋值给outputs
        outputs = (output_h, output_g) + outputs[2:]  # Add again attentions if there are there
        return outputs

    # 定义一个名为ff_chunk的方法，接受output_x作为输入
    def ff_chunk(self, output_x):
        # 对output_x执行前馈网络操作
        output_x = self.ff(output_x)
        # 返回output_x
        return output_x


# 定义一个名为XLNetPreTrainedModel的类，继承自PreTrainedModel
class XLNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 将XLNetConfig赋值给config_class属性
    config_class = XLNetConfig
    # 将load_tf_weights_in_xlnet赋值给load_tf_weights属性
    load_tf_weights = load_tf_weights_in_xlnet
    # 将transformer赋值给base_model_prefix属性
    base_model_prefix = "transformer"
    # 初始化权重
    def _init_weights(self, module):
        """Initialize the weights."""
        # 如果是全连接层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在padding_idx，将其对应的权重初始化为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是LayerNorm层
        elif isinstance(module, nn.LayerNorm):
            # 偏置项初始化为0
            module.bias.data.zero_()
            # 权重初始化为1
            module.weight.data.fill_(1.0)
        # 如果是XLNetRelativeAttention层
        elif isinstance(module, XLNetRelativeAttention):
            # 对多个参数使用正态分布进行初始化
            for param in [
                module.q,
                module.k,
                module.v,
                module.o,
                module.r,
                module.r_r_bias,
                module.r_s_bias,
                module.r_w_bias,
                module.seg_embed,
            ]:
                param.data.normal_(mean=0.0, std=self.config.initializer_range)
        # 如果是XLNetModel层
        elif isinstance(module, XLNetModel):
            # mask_emb参数使用正态分布进行初始化
            module.mask_emb.data.normal_(mean=0.0, std=self.config.initializer_range)
# 定义一个数据类 XLNetModelOutput，作为 XLNetModel 的输出类型
@dataclass
class XLNetModelOutput(ModelOutput):
    """
    XLNetModel 的输出类型。

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_predict, hidden_size)`):
            模型最后一层的隐藏状态序列。

            `num_predict` 对应于 `target_mapping.shape[1]`。如果 `target_mapping` 为 `None`，则 `num_predict`
            对应于 `sequence_length`。
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            包含预先计算的隐藏状态的列表。可用于加速顺序解码。将其过去给该模型的标记 ID 不应作为 `input_ids`
            传递，因为它们已经计算过了。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `torch.FloatTensor` 元组（一个用于嵌入的输出 + 一个用于每个层的输出）的形状为 `(batch_size, sequence_length, hidden_size)`。

            模型在每一层输出的隐藏状态加上初始嵌入的输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            `torch.FloatTensor` 元组（每一层一个）的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力 softmax 后的注意力权重，用于计算自注意力头的加权平均值。
    """

    # 模型的最后一个隐藏状态
    last_hidden_state: torch.FloatTensor
    # 包含预先计算隐藏状态的列表，用于加速顺序解码
    mems: Optional[List[torch.FloatTensor]] = None
    # 模型每一层的隐藏状态
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 模型的注意力权重
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 定义一个数据类 XLNetLMHeadModelOutput，作为 XLNetLMHeadModel 的输出类型
@dataclass
class XLNetLMHeadModelOutput(ModelOutput):
    """
    XLNetLMHeadModel 的输出类型。
    """
    # 定义函数参数，损失值（当提供`labels`时返回），语言模型头的预测分数，预先计算的隐藏状态，隐藏状态的输出，以及注意力权重
    Args:
        loss (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided)
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, num_predict, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

            `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
            corresponds to `sequence_length`.
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
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

    # 定义变量并初始化为None，用于存储损失值、预测分数、预先计算的隐藏状态、隐藏状态的输出以及注意力权重
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mems: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个数据类，用于存储 XLNet 模型对序列分类任务的输出结果
@dataclass
class XLNetForSequenceClassificationOutput(ModelOutput):
    """
    Output type of [`XLNetForSequenceClassification`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `label` is provided):
            分类（或回归，如果 config.num_labels==1）损失值。
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            分类（或回归，如果 config.num_labels==1）得分（未经 SoftMax 处理）。
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            包含预先计算的隐藏状态。可以用来加速顺序解码。将模型的过去输入给含有这些记忆的 token ids 时应该不应该作为 `input_ids` 传递，因为它们已经计算过了。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            一个元组的 `torch.FloatTensor`（包含嵌入层输出和每一层输出），shape 为 `(batch_size, sequence_length, hidden_size)`。

            每个层的模型输出加上初始嵌入层的输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            一个元组的 `torch.FloatTensor`（每个层一个）的 shape 为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mems: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 定义一个数据类，用于存储 XLNet 模型对标记分类任务的输出结果
@dataclass
class XLNetForTokenClassificationOutput(ModelOutput):
    """
    Output type of [`XLNetForTokenClassificationOutput`].
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.  # 分类损失
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).  # 分类得分（SoftMax之前的）
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.  # 包含预先计算的隐藏状态。可以用来加速顺序解码
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.  # 模型在每一层输出的隐藏状态，以及初始嵌入输出
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.  # 注意力softmax之后的注意力权重，用于计算自注意力头中的加权平均
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mems: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义了一个数据类，用于存储 XLNetForMultipleChoice 模型的输出
@dataclass
class XLNetForMultipleChoiceOutput(ModelOutput):
    """
    XLNetForMultipleChoice 模型的输出类型

    Args:
        loss (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided):
            分类损失。
        logits (`torch.FloatTensor` of shape `(batch_size, num_choices)`):
            *num_choices* 是输入张量的第二维。(参见上面的 *input_ids*)。

            分类分数（SoftMax 之前）。
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            包含预先计算的隐藏状态。可以用于加速顺序解码。已给此模型的过去的标记 ID 不应该作为 `input_ids` 传递，因为它们已经被计算过。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `torch.FloatTensor` 元组（一个用于嵌入的输出 + 一个用于每个层的输出）的形状为 `(batch_size, sequence_length, hidden_size)`。

            每层模型在每层的输出的隐藏状态加上初始嵌入的输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            `torch.FloatTensor` 元组（每个层一个）的形状为 `(batch_size, num_heads, sequence_length,sequence_length)`。

            用于计算自注意力头中加权平均值的注意力软最大化后的注意力权重。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mems: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 定义了一个数据类，用于存储 XLNetForQuestionAnsweringSimple 模型的输出
@dataclass
class XLNetForQuestionAnsweringSimpleOutput(ModelOutput):
    """
    Output type of [`XLNetForQuestionAnsweringSimple`].
    ```
    # `loss`是一个可选的`torch.FloatTensor`对象，形状为`(1,)`，当提供`labels`时返回
    # 总的跨度抽取损失是起始位置和结束位置的交叉熵损失的总和
    loss: Optional[torch.FloatTensor] = None

    # `start_logits`是一个`torch.FloatTensor`对象，形状为`(batch_size, sequence_length,)`
    # 跨度的起始得分（在SoftMax之前）
    start_logits: torch.FloatTensor = None

    # `end_logits`是一个`torch.FloatTensor`对象，形状为`(batch_size, sequence_length,)`
    # 跨度的结束得分（在SoftMax之前）
    end_logits: torch.FloatTensor = None

    # `mems`是一个`List[torch.FloatTensor]`对象，长度为`config.n_layers`
    # 包含预先计算的隐藏状态。可以用于加速序列解码。将已经计算的`input_ids`传递给模型时，不应将其传递为`input_ids`
    mems: Optional[List[torch.FloatTensor]] = None

    # `hidden_states`是包含模型每一层输出的一个`tuple(torch.FloatTensor)`对象
    # 形状为`(batch_size, sequence_length, hidden_size)`
    #
    # 模型每一层输出的隐藏状态加上初始嵌入输出
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

    # `attentions`是一个tuple(torch.FloatTensor)`对象，当传递`output_attentions=True`或`config.output_attentions=True`时返回
    # 包含每一层的注意力权重，形状为`(batch_size, num_heads, sequence_length, sequence_length)`
    #
    # 在自注意头中使用注意力softmax之后的注意力权重，用于计算加权平均值
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个输出类型类，用于XLNetForQuestionAnswering模型的输出
@dataclass
class XLNetForQuestionAnsweringOutput(ModelOutput):
    """
    Output type of [`XLNetForQuestionAnswering`].

    Args:
        # 分类损失，如果提供了`start_positions`和`end_positions`则返回
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions` are provided):
            Classification loss as the sum of start token, end token (and is_impossible if provided) classification
            losses.
        # 开始位置的前config.start_n_top个概率对数值，如果未提供`start_positions`或`end_positions`则返回
        start_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the top config.start_n_top start token possibilities (beam-search).
        # 开始位置的前config.start_n_top个索引，如果未提供`start_positions`或`end_positions`则返回
        start_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Indices for the top config.start_n_top start token possibilities (beam-search).
        # 结束位置的前`config.start_n_top * config.end_n_top`个概率对数值，如果未提供`start_positions`或`end_positions`则返回
        end_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the top `config.start_n_top * config.end_n_top` end token possibilities
            (beam-search).
        # 结束位置的前`config.start_n_top * config.end_n_top`个索引，如果未提供`start_positions`或`end_positions`则返回
        end_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Indices for the top `config.start_n_top * config.end_n_top` end token possibilities (beam-search).
        # 答案`is_impossible`标签的对数概率，如果未提供`start_positions`或`end_positions`则返回
        cls_logits (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the `is_impossible` label of the answers.
        # 包含预先计算的隐藏状态的列表，长度为`config.n_layers`，用于加速顺序解码
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
        # 隐藏层的元组，当传递`output_hidden_states=True`或`config.output_hidden_states=True`时返回
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        # 注意力的元组，当传递`output_attentions=True`或`config.output_attentions=True`时返回
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    # 表示模型的输出结果可能包含以下几个部分:
    # loss: 模型训练过程中计算得到的损失值
    # start_top_log_probs: 模型预测每个位置作为答案起始位置的概率分数
    # start_top_index: 模型预测每个位置作为答案起始位置的索引
    # end_top_log_probs: 模型预测每个位置作为答案结束位置的概率分数
    # end_top_index: 模型预测每个位置作为答案结束位置的索引
    # cls_logits: 模型预测整个输入序列的分类结果
    # mems: 模型的内部状态，可用于语言模型的连续生成
    # hidden_states: 模型各层的隐藏状态
    # attentions: 模型各层的注意力权重
    
    loss: Optional[torch.FloatTensor] = None
    start_top_log_probs: Optional[torch.FloatTensor] = None
    start_top_index: Optional[torch.LongTensor] = None
    end_top_log_probs: Optional[torch.FloatTensor] = None
    end_top_index: Optional[torch.LongTensor] = None
    cls_logits: Optional[torch.FloatTensor] = None
    mems: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义 XLNetModel 类，继承自 XLNetPreTrainedModel
class XLNetModel(XLNetPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化各种模型参数
        self.mem_len = config.mem_len
        self.reuse_len = config.reuse_len
        self.d_model = config.d_model
        self.same_length = config.same_length
        self.attn_type = config.attn_type
        self.bi_data = config.bi_data
        self.clamp_len = config.clamp_len
        self.n_layer = config.n_layer

        # 初始化词嵌入层
        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
        # 初始化 mask_emb
        self.mask_emb = nn.Parameter(torch.FloatTensor(1, 1, config.d_model))
        # 初始化每层的 XLNetLayer，并放入 nn.ModuleList 中
        self.layer = nn.ModuleList([XLNetLayer(config) for _ in range(config.n_layer)])
        # 初始化 dropout 层
        self.dropout = nn.Dropout(config.dropout)

        # 初始化权重和应用最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.word_embedding

    # 设置输入嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.word_embedding = new_embeddings

    # 剪枝操作，将指定的 heads 剪枝
    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError
    # 创建因果注意力掩码。浮点掩码，其中1.0表示已屏蔽，0.0表示未屏蔽。
    def create_mask(self, qlen, mlen):
        mask = torch.ones((qlen, qlen + mlen), device=self.device)  # 创建形状为(qlen, qlen + mlen)的全1张量
        if self.same_length:  # 如果same_length为True
            mask_lo = mask[:, :qlen].tril(-1)  # 生成下三角形矩阵
            mask.triu_(mlen + 1)  # 对mask进行上三角掩码操作
            mask[:, :qlen] += mask_lo  # 将下三角形矩阵加到上三角掩码上
        else:  # 如果same_length为False
            mask.triu_(mlen + 1)  # 对mask进行上三角掩码操作

        return mask  # 返回掩码张量

    # 缓存隐藏状态到记忆中
    def cache_mem(self, curr_out, prev_mem):
        if self.reuse_len is not None and self.reuse_len > 0:  # 如果reuse_len不为空且大于0
            curr_out = curr_out[: self.reuse_len]  # 仅保留前reuse_len个隐藏状态

        if self.mem_len is None or self.mem_len == 0:  # 如果mem_len为空或为0
            cutoff = 0  # 截断索引为0
        else:  # 否则
            cutoff = -self.mem_len  # 截断索引为-mem_len
        if prev_mem is None:  # 如果prev_mem为空
            new_mem = curr_out[cutoff:]  # 将curr_out从cutoff索引截断
        else:  # 否则
            new_mem = torch.cat([prev_mem, curr_out], dim=0)[cutoff:]  # 在维度0上拼接prev_mem和curr_out，并截断索引为cutoff

        return new_mem.detach()  # 返回分离的新的记忆张量

    @staticmethod
    # 生成位置嵌入
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)  # 执行张量einsum乘法
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)  # 沿指定维度拼接正弦和余弦
        pos_emb = pos_emb[:, None, :]  # 增加一个维度

        if bsz is not None:  # 如果bsz不为空
            pos_emb = pos_emb.expand(-1, bsz, -1)  # 在指定维度上扩展张量

        return pos_emb  # 返回位置嵌入张量
    # 生成相对位置编码
    def relative_positional_encoding(self, qlen, klen, bsz=None):
        # 创建频率序列，范围为从0到d_model，步长为2，数据类型为浮点数
        freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.float)
        # 计算逆频率，使用10000作为底数，freq_seq除以d_model作为指数
        inv_freq = 1 / torch.pow(10000, (freq_seq / self.d_model))

        # 根据注意力类型设置beg和end的值
        if self.attn_type == "bi":
            # 双向注意力，beg和end的范围从klen到-qlen
            beg, end = klen, -qlen
        elif self.attn_type == "uni":
            # 单向注意力，beg和end的范围从klen到-1
            beg, end = klen, -1
        else:
            # 如果attn_type不匹配，抛出错误
            raise ValueError(f"Unknown `attn_type` {self.attn_type}.")

        # 如果是双向数据
        if self.bi_data:
            # 生成前向和后向位置序列
            fwd_pos_seq = torch.arange(beg, end, -1.0, dtype=torch.float)
            bwd_pos_seq = torch.arange(-beg, -end, 1.0, dtype=torch.float)

            # 如果clamp_len大于0，则对位置序列进行截断
            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
                bwd_pos_seq = bwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)

            # 如果给定批量大小bsz，则分别计算前向和后向位置编码
            if bsz is not None:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz // 2)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bsz // 2)
            else:
                # 否则，计算前向和后向位置编码
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)

            # 将前向和后向位置编码拼接在一起
            pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb], dim=1)
        else:
            # 如果不是双向数据，计算前向位置序列
            fwd_pos_seq = torch.arange(beg, end, -1.0)
            # 如果clamp_len大于0，则对位置序列进行截断
            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
            # 计算位置编码
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        # 返回位置编码
        return pos_emb

    # 添加文档字符串和代码示例说明
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=XLNetModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # forward方法定义
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mems: Optional[torch.Tensor] = None,
        perm_mask: Optional[torch.Tensor] = None,
        target_mapping: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_mems: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # 在移除废弃警告后删除
# 在 XLNet 模型顶部具有语言建模头的模型
class XLNetLMHeadModel(XLNetPreTrainedModel):
    # 权重绑定的关键字
    _tied_weights_keys = ["lm_loss.weight"]

    # 初始化函数
    def __init__(self, config):
        super().__init__(config)
        self.attn_type = config.attn_type
        self.same_length = config.same_length

        # 初始化 transformer 和 lm_loss
        self.transformer = XLNetModel(config)
        self.lm_loss = nn.Linear(config.d_model, config.vocab_size, bias=True)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_loss

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_loss = new_embeddings

    # 为生成准备输入
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, use_mems=None, **kwargs):
        # 在末尾添加虚拟令牌（对此不进行注意力）
        
        effective_batch_size = input_ids.shape[0]
        dummy_token = torch.zeros((effective_batch_size, 1), dtype=torch.long, device=input_ids.device)

        # 在每个传递中，计算新令牌和最后两个生成的令牌的注意力值，其余从“过去”缓存中重新加载.
        # 纯自回归模型将具有 offset=1; offset=2 看起来计算略有改善。
        offset = 2

        if past_key_values:
            input_ids = torch.cat([input_ids[:, -offset:], dummy_token], dim=1)
        else:
            input_ids = torch.cat([input_ids, dummy_token], dim=1)

        # 构建置换掩码，使先前的标记不看到最后一个标记
        sequence_length = input_ids.shape[1]
        perm_mask = torch.zeros(
            (effective_batch_size, sequence_length, sequence_length), dtype=torch.float, device=input_ids.device
        )
        perm_mask[:, :, -1] = 1.0

        # 我们只预测最后一个令牌
        target_mapping = torch.zeros(
            (effective_batch_size, 1, sequence_length), dtype=torch.float, device=input_ids.device
        )
        target_mapping[:, 0, -1] = 1.0
        
        # 输入数据
        inputs = {
            "input_ids": input_ids,
            "perm_mask": perm_mask,
            "target_mapping": target_mapping,
            "use_mems": use_mems,
        }

        # 如果模型参数中定义了过去，则使用它进行更快的解码
        if past_key_values:
            inputs["mems"] = tuple(layer_past[:-offset, :, :] for layer_past in past_key_values)

        return inputs
    # 前向传播函数，用于模型的前向推断
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的token ID序列，默认为None
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，默认为None
        mems: Optional[torch.Tensor] = None,  # 存储历史隐藏状态的记忆，默认为None
        perm_mask: Optional[torch.Tensor] = None,  # 排列掩码，默认为None
        target_mapping: Optional[torch.Tensor] = None,  # 目标映射，默认为None
        token_type_ids: Optional[torch.Tensor] = None,  # token类型ID，默认为None
        input_mask: Optional[torch.Tensor] = None,  # 输入掩码，默认为None
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码，默认为None
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入，默认为None
        labels: Optional[torch.Tensor] = None,  # 标签，默认为None
        use_mems: Optional[bool] = None,  # 是否使用记忆，默认为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典，默认为None
        **kwargs,  # 删除`use_cache`在XLNetModel中使用时的参数，使用kwargs来接收这些参数
    @staticmethod
    def _reorder_cache(mems: List[torch.Tensor], beam_idx: torch.Tensor) -> List[torch.Tensor]:
        """
        用于在[`~PreTrainedModel.beam_search`]或[`~PreTrainedModel.beam_sample`]调用时重新排列`mems`缓存。
        这是为了在每个生成步骤中将`mems`与正确的beam_idx匹配。
        """
        # 返回重新排序的`mems`缓存列表，确保与给定的beam_idx匹配
        return [layer_past.index_select(1, beam_idx.to(layer_past.device)) for layer_past in mems]
# 使用 XLNet 模型进行序列分类/回归的模型
class XLNetForSequenceClassification(XLNetPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置标签数量
        self.num_labels = config.num_labels
        self.config = config

        # 创建 XLNet 模型
        self.transformer = XLNetModel(config)
        # 创建序列摘要
        self.sequence_summary = SequenceSummary(config)
        # 创建线性层，用于分类
        self.logits_proj = nn.Linear(config.d_model, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的 token ID
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        mems: Optional[torch.Tensor] = None,  # 记忆
        perm_mask: Optional[torch.Tensor] = None,  # 排列掩码
        target_mapping: Optional[torch.Tensor] = None,  # 目标映射
        token_type_ids: Optional[torch.Tensor] = None,  # token 类型 ID
        input_mask: Optional[torch.Tensor] = None,  # 输入掩码
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入
        labels: Optional[torch.Tensor] = None,  # 标签
        use_mems: Optional[bool] = None,  # 是否使用记忆
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典
        **kwargs,  # 当 `use_cache` 在 XLNetModel 中被移除时删除
    ) -> Union[Tuple, XLNetForSequenceClassificationOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 判断是否需要返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给transformer，并获取transformer的输出
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        output = transformer_outputs[0]

        # 使用sequence_summary函数对输出进行处理
        output = self.sequence_summary(output)
        logits = self.logits_proj(output)

        loss = None
        if labels is not None:
            # 根据config中的问题类型确定损失函数
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 计算损失
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

        # 如果不需要返回字典，则返回元组
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回包含损失、预测结果等信息的对象
        return XLNetForSequenceClassificationOutput(
            loss=loss,
            logits=logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
# 定义一个带有标记分类头部的 XLNet 模型（在隐藏状态输出的顶部是线性层），例如用于命名实体识别（NER）任务
@add_start_docstrings(
    """
    XLNet Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    XLNET_START_DOCSTRING,
)
class XLNetForTokenClassification(XLNetPreTrainedModel):
    # 初始化函数，接受配置作为参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 将配置中的标签数赋值给模型对象的属性
        self.num_labels = config.num_labels

        # 创建 XLNet 模型对象
        self.transformer = XLNetModel(config)
        # 创建线性层，输入维度为隐藏状态的大小，输出维度为标签数
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，接受多个输入参数和可选的关键字参数
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=XLNetForTokenClassificationOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 可选的输入标记 ID
        attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力遮罩
        mems: Optional[torch.Tensor] = None,  # 可选的记忆
        perm_mask: Optional[torch.Tensor] = None,  # 可选的排列遮罩
        target_mapping: Optional[torch.Tensor] = None,  # 可选的目标映射
        token_type_ids: Optional[torch.Tensor] = None,  # 可选的标记类型 ID
        input_mask: Optional[torch.Tensor] = None,  # 可选的输入遮罩
        head_mask: Optional[torch.Tensor] = None,  # 可选的头部遮罩
        inputs_embeds: Optional[torch.Tensor] = None,  # 可选的嵌入输入
        labels: Optional[torch.Tensor] = None,  # 可选的标签
        use_mems: Optional[bool] = None,  # 可选的使用记忆
        output_attentions: Optional[bool] = None,  # 可选的输出注意力
        output_hidden_states: Optional[bool] = None,  # 可选的输出隐藏状态
        return_dict: Optional[bool] = None,  # 可选的返回字典
        **kwargs,  # 在 `use_cache` 在 XLNetModel 中被移除时删除
    # 定义一个函数，接受一个 PyTorch 张量作为输入，并返回一个元组或 XLNetForTokenClassificationOutput 对象
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        use_mems=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None,
    ) -> Union[Tuple, XLNetForTokenClassificationOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where *num_choices* is the size of the second dimension of the input tensors. (see *input_ids* above)
        """
        # 如果 return_dict 为 None，则使用配置中的 use_return_dict 属性
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 调用 transformer 模块，传入各种输入参数，获取输出
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        # 获取序列输出
        sequence_output = outputs[0]
    
        # 将序列输出传入分类器，得到分类logits
        logits = self.classifier(sequence_output)
    
        # 如果提供了标签，则计算分类损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
        # 如果不使用返回字典，则返回一个元组
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
    
        # 否则返回 XLNetForTokenClassificationOutput 对象
        return XLNetForTokenClassificationOutput(
            loss=loss,
            logits=logits,
            mems=outputs.mems,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 这个类是 XLNetForMultipleChoice 的实现，它继承自 XLNetPreTrainedModel
@add_start_docstrings(
    """
    XLNet Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RACE/SWAG tasks.
    """,
    XLNET_START_DOCSTRING,
)
class XLNetForMultipleChoice(XLNetPreTrainedModel):
    def __init__(self, config):
        # 调用父类的构造方法
        super().__init__(config)

        # 创建 XLNetModel 对象
        self.transformer = XLNetModel(config)
        # 创建序列摘要层
        self.sequence_summary = SequenceSummary(config)
        # 创建一个线性层用于计算logits
        self.logits_proj = nn.Linear(config.d_model, 1)

        # 执行初始化权重的操作
        self.post_init()

    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=XLNetForMultipleChoiceOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mems: Optional[torch.Tensor] = None,
        perm_mask: Optional[torch.Tensor] = None,
        target_mapping: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_mems: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # delete when `use_cache` is removed in XLNetModel
    ) -> Union[Tuple, XLNetForMultipleChoiceOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 如果 return_dict 不为 None，则使用其值，否则使用配置中的 use_return_dict 值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果 input_ids 不为 None，则获取其第二维度的大小作为 num_choices，否则获取 inputs_embeds 的第二维度大小
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 如果 input_ids 不为 None，则将其展平为二维张量，否则为 None
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 如果 token_type_ids 不为 None，则将其展平为二维张量，否则为 None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # 如果 attention_mask 不为 None，则将其展平为二维张量，否则为 None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 如果 input_mask 不为 None，则将其展平为二维张量，否则为 None
        flat_input_mask = input_mask.view(-1, input_mask.size(-1)) if input_mask is not None else None
        # 如果 inputs_embeds 不为 None，则将其展平为三维张量，否则为 None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 使用 XLNet 的 Transformer 进行前向传播
        transformer_outputs = self.transformer(
            flat_input_ids,
            token_type_ids=flat_token_type_ids,
            input_mask=flat_input_mask,
            attention_mask=flat_attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,  # 接收任意额外的关键字参数
        )

        # 获取 Transformer 输出中的第一个元素
        output = transformer_outputs[0]

        # 对 Transformer 输出进行序列摘要
        output = self.sequence_summary(output)
        # 对序列摘要后的输出进行 logits 投影
        logits = self.logits_proj(output)
        # 将 logits 展平为二维张量
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        # 如果 labels 不为 None，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels.view(-1))

        # 如果不使用 return_dict，则返回一个元组
        if not return_dict:
            output = (reshaped_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果使用 return_dict，则返回 XLNetForMultipleChoiceOutput 对象
        return XLNetForMultipleChoiceOutput(
            loss=loss,
            logits=reshaped_logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
@add_start_docstrings(
    """
    XLNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    XLNET_START_DOCSTRING,
)
# 定义 XLNet 问答模型，用于提取性问答任务，如 SQuAD，具有在隐藏状态输出之上的跨距分类头（用于计算“跨距开始 logits”和“跨距结束 logits”）
class XLNetForQuestionAnsweringSimple(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = XLNetModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=XLNetForQuestionAnsweringSimpleOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义前向传播函数，接收多个输入参数并返回预测结果
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mems: Optional[torch.Tensor] = None,
        perm_mask: Optional[torch.Tensor] = None,
        target_mapping: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        use_mems: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # delete when `use_cache` is removed in XLNetModel
@add_start_docstrings(
    """
    XLNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    XLNET_START_DOCSTRING,
)
# 定义 XLNet 问答模型，用于提取性问答任务，如 SQuAD，具有在隐藏状态输出之上的跨距分类头（用于计算“跨距开始 logits”和“跨距结束 logits”）
class XLNetForQuestionAnswering(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top

        self.transformer = XLNetModel(config)
        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.answer_class = PoolerAnswerClass(config)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=XLNetForQuestionAnsweringOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法，用于执行模型的前向传播
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的token IDs序列，可选的张量
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩，可选的张量
        mems: Optional[torch.Tensor] = None,  # 用于缓存历史隐藏状态的记忆，可选的张量
        perm_mask: Optional[torch.Tensor] = None,  # 排列遮罩，可选的张量
        target_mapping: Optional[torch.Tensor] = None,  # 目标映射，可选的张量
        token_type_ids: Optional[torch.Tensor] = None,  # token类型IDs，可选的张量
        input_mask: Optional[torch.Tensor] = None,  # 输入遮罩，可选的张量
        head_mask: Optional[torch.Tensor] = None,  # 头部遮罩，可选的张量
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入，可选的张量
        start_positions: Optional[torch.Tensor] = None,  # 起始位置，可选的张量
        end_positions: Optional[torch.Tensor] = None,  # 结束位置，可选的张量
        is_impossible: Optional[torch.Tensor] = None,  # 是否不可能的标志，可选的张量
        cls_index: Optional[torch.Tensor] = None,  # 类别标志的索引，可选的张量
        p_mask: Optional[torch.Tensor] = None,  # 掩码，可选的张量
        use_mems: Optional[bool] = None,  # 是否使用记忆，可选的布尔值
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选的布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典格式结果，可选的布尔值
        **kwargs,  # 当`use_cache`在XLNetModel中被移除时，需要删除
```