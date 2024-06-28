# `.\models\xlnet\modeling_xlnet.py`

```py
# 创建一个映射字典，用于从 TensorFlow 到 PyTorch 的模块映射，以尽可能保持 PyTorch 模型与原始 TensorFlow 模型的一致性
def build_tf_xlnet_to_pytorch_map(model, config, tf_weights=None):
    # 初始化一个空的 TensorFlow 到 PyTorch 的映射字典
    tf_to_pt_map = {}
    # 检查模型是否有 "transformer" 属性
    if hasattr(model, "transformer"):
        # 如果模型有 "lm_loss" 属性，加载 lm_loss 的偏置项
        if hasattr(model, "lm_loss"):
            tf_to_pt_map["model/lm_loss/bias"] = model.lm_loss.bias
        
        # 如果模型有 "sequence_summary" 属性，并且在 tf_weights 中包含指定的路径
        if hasattr(model, "sequence_summary") and "model/sequnece_summary/summary/kernel" in tf_weights:
            # 加载 sequence summary 的权重和偏置项
            tf_to_pt_map["model/sequnece_summary/summary/kernel"] = model.sequence_summary.summary.weight
            tf_to_pt_map["model/sequnece_summary/summary/bias"] = model.sequence_summary.summary.bias
        
        # 如果模型有 "logits_proj" 属性，同时满足 finetuning_task 不为 None，并且在 tf_weights 中包含指定的路径
        if (
            hasattr(model, "logits_proj")
            and config.finetuning_task is not None
            and f"model/regression_{config.finetuning_task}/logit/kernel" in tf_weights
        ):
            # 加载 logits_proj 的权重和偏置项
            tf_to_pt_map[f"model/regression_{config.finetuning_task}/logit/kernel"] = model.logits_proj.weight
            tf_to_pt_map[f"model/regression_{config.finetuning_task}/logit/bias"] = model.logits_proj.bias
        
        # 将模型切换至 transformer 属性
        model = model.transformer

    # 加载嵌入和输出层的映射关系
    tf_to_pt_map.update(
        {
            "model/transformer/word_embedding/lookup_table": model.word_embedding.weight,
            "model/transformer/mask_emb/mask_emb": model.mask_emb,
        }
    )

    # 加载 transformer 模块的各个层
    for i, b in enumerate(model.layer):
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

    # 如果 config.untie_r 为真，则加载相对位置偏置列表
    if config.untie_r:
        r_r_list = []
        r_w_list = []
        r_s_list = []
        seg_embed_list = []
        for b in model.layer:
            r_r_list.append(b.rel_attn.r_r_bias)
            r_w_list.append(b.rel_attn.r_w_bias)
            r_s_list.append(b.rel_attn.r_s_bias)
            seg_embed_list.append(b.rel_attn.seg_embed)
    else:
        # 否则加载单一的相对位置偏置项
        r_r_list = [model.r_r_bias]
        r_w_list = [model.r_w_bias]
        r_s_list = [model.r_s_bias]
        seg_embed_list = [model.seg_embed]
    # 更新 tf_to_pt_map 字典，将四个键值对添加或更新到字典中
    tf_to_pt_map.update(
        {
            "model/transformer/r_r_bias": r_r_list,  # 键为 "model/transformer/r_r_bias"，值为 r_r_list
            "model/transformer/r_w_bias": r_w_list,  # 键为 "model/transformer/r_w_bias"，值为 r_w_list
            "model/transformer/r_s_bias": r_s_list,  # 键为 "model/transformer/r_s_bias"，值为 r_s_list
            "model/transformer/seg_embed": seg_embed_list,  # 键为 "model/transformer/seg_embed"，值为 seg_embed_list
        }
    )
    # 返回更新后的 tf_to_pt_map 字典
    return tf_to_pt_map
# 在 PyTorch 模型中加载 TensorFlow 的检查点权重
def load_tf_weights_in_xlnet(model, config, tf_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import numpy as np  # 导入 NumPy 库，用于数组操作
        import tensorflow as tf  # 导入 TensorFlow 库，用于加载 TensorFlow 模型权重
    except ImportError:
        logger.error(
            "Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    # 从 TensorFlow 模型中加载初始变量列表
    init_vars = tf.train.list_variables(tf_path)
    tf_weights = {}

    # 遍历每个变量名和形状，加载 TensorFlow 模型权重
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        tf_weights[name] = array

    # 构建 TensorFlow 到 PyTorch 权重加载映射
    tf_to_pt_map = build_tf_xlnet_to_pytorch_map(model, config, tf_weights)

    # 遍历映射表中的每个变量名和指针
    for name, pointer in tf_to_pt_map.items():
        logger.info(f"Importing {name}")

        # 如果变量名不在 TensorFlow 权重中，则跳过
        if name not in tf_weights:
            logger.info(f"{name} not in tf pre-trained weights, skipping")
            continue

        array = tf_weights[name]

        # 对于特定的变量名模式，需要进行数组转置操作
        if "kernel" in name and ("ff" in name or "summary" in name or "logit" in name):
            logger.info("Transposing")
            array = np.transpose(array)

        # 如果指针是列表，则需要分割 TensorFlow 的权重数组
        if isinstance(pointer, list):
            assert (
                len(pointer) == array.shape[0]
            ), f"Pointer length {len(pointer)} and array length {array.shape[0]} mismatched"
            for i, p_i in enumerate(pointer):
                arr_i = array[i, ...]
                try:
                    assert (
                        p_i.shape == arr_i.shape
                    ), f"Pointer shape {p_i.shape} and array shape {arr_i.shape} mismatched"
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

        # 从 TensorFlow 权重字典中移除处理过的变量名及其相关项
        tf_weights.pop(name, None)
        tf_weights.pop(name + "/Adam", None)
        tf_weights.pop(name + "/Adam_1", None)

    # 输出未复制到 PyTorch 模型的权重变量名列表
    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}")

    # 返回加载完权重后的 PyTorch 模型
    return model
    def __init__(self, config):
        super().__init__()  # 调用父类的初始化方法

        if config.d_model % config.n_head != 0:
            raise ValueError(
                f"The hidden size ({config.d_model}) is not a multiple of the number of attention "
                f"heads ({config.n_head}"  # 抛出异常，如果隐藏大小不是注意力头数的倍数
            )

        self.n_head = config.n_head  # 设置注意力头的数量
        self.d_head = config.d_head  # 设置每个头的隐藏大小
        self.d_model = config.d_model  # 设置模型的隐藏大小
        self.scale = 1 / (config.d_head**0.5)  # 缩放因子，用于缩放注意力分数

        # 下面的四个参数用于存储注意力机制中的查询、键、值和输出矩阵
        self.q = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.k = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.v = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.o = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.r = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))

        # 下面四个参数用于存储注意力中的偏置项
        self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.r_s_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.seg_embed = nn.Parameter(torch.FloatTensor(2, self.n_head, self.d_head))

        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)  # Layer normalization层
        self.dropout = nn.Dropout(config.dropout)  # Dropout层

    def prune_heads(self, heads):
        raise NotImplementedError  # 剪枝注意力头的方法，抛出未实现异常

    @staticmethod
    def rel_shift(x, klen=-1):
        """perform relative shift to form the relative attention score."""
        x_size = x.shape  # 获取输入张量的形状

        x = x.reshape(x_size[1], x_size[0], x_size[2], x_size[3])  # 重塑张量维度顺序
        x = x[1:, ...]  # 去掉第一行，实现相对位移
        x = x.reshape(x_size[0], x_size[1] - 1, x_size[2], x_size[3])  # 重新整形张量维度
        # x = x[:, 0:klen, :, :]
        x = torch.index_select(x, 1, torch.arange(klen, device=x.device, dtype=torch.long))  # 通过索引选择相对位移后的部分

        return x

    @staticmethod
    def rel_shift_bnij(x, klen=-1):
        x_size = x.shape  # 获取输入张量的形状

        x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])  # 重塑张量维度顺序
        x = x[:, :, 1:, :]  # 去掉第一列，实现相对位移
        x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)  # 重新整形张量维度
        # Note: the tensor-slice form was faster in my testing than torch.index_select
        #       However, tracing doesn't like the nature of the slice, and if klen changes
        #       during the run then it'll fail, whereas index_select will be fine.
        x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))  # 通过索引选择相对位移后的部分
        # x = x[:, :, :, :klen]

        return x

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
            # Apply attention mask based on its dtype
            if attn_mask.dtype == torch.float16:
                attn_score = attn_score - 65500 * torch.einsum("ijbn->bnij", attn_mask)
            else:
                attn_score = attn_score - 1e30 * torch.einsum("ijbn->bnij", attn_mask)

        # attention probability
        attn_prob = nn.functional.softmax(attn_score, dim=3)
        attn_prob = self.dropout(attn_prob)

        # Mask heads if specified
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
class XLNetFeedForward(nn.Module):
    # 定义一个 XLNet 模型的前馈层
    def __init__(self, config):
        super().__init__()
        # 初始化层归一化模块，使用给定的 d_model 和 layer_norm_eps 参数
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        # 初始化第一个线性层，输入维度为 d_model，输出维度为 d_inner
        self.layer_1 = nn.Linear(config.d_model, config.d_inner)
        # 初始化第二个线性层，输入维度为 d_inner，输出维度为 d_model
        self.layer_2 = nn.Linear(config.d_inner, config.d_model)
        # 初始化 Dropout 模块，使用给定的 dropout 参数
        self.dropout = nn.Dropout(config.dropout)
        # 如果 ff_activation 是字符串，则根据 ACT2FN 字典选择对应的激活函数，否则直接使用给定的激活函数
        if isinstance(config.ff_activation, str):
            self.activation_function = ACT2FN[config.ff_activation]
        else:
            self.activation_function = config.ff_activation

    # 前向传播函数
    def forward(self, inp):
        # 将输入赋给 output
        output = inp
        # 经过第一个线性层
        output = self.layer_1(output)
        # 应用激活函数
        output = self.activation_function(output)
        # 应用 Dropout
        output = self.dropout(output)
        # 经过第二个线性层
        output = self.layer_2(output)
        # 再次应用 Dropout
        output = self.dropout(output)
        # 对 output 和输入 inp 进行残差连接，并进行层归一化
        output = self.layer_norm(output + inp)
        # 返回输出
        return output


class XLNetLayer(nn.Module):
    # 定义一个 XLNet 模型的层
    def __init__(self, config):
        super().__init__()
        # 初始化相对注意力层
        self.rel_attn = XLNetRelativeAttention(config)
        # 初始化前馈层
        self.ff = XLNetFeedForward(config)
        # 初始化 Dropout 模块
        self.dropout = nn.Dropout(config.dropout)
        # 设置前馈层的分块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置序列长度维度
        self.seq_len_dim = 1

    # 前向传播函数
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
        # 使用相对注意力层计算输出
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
        # 获取相对注意力层的输出中的前两个部分
        output_h, output_g = outputs[:2]

        # 如果 output_g 不为 None，则对其应用分块前馈
        if output_g is not None:
            output_g = apply_chunking_to_forward(
                self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, output_g
            )
        # 对 output_h 应用分块前馈
        output_h = apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, output_h)

        # 将相对注意力层的输出重新组合并添加可能的注意力部分
        outputs = (output_h, output_g) + outputs[2:]  # 如果有的话再次添加注意力
        # 返回所有输出
        return outputs

    # 定义前馈层的分块函数
    def ff_chunk(self, output_x):
        # 对输入 output_x 应用前馈层
        output_x = self.ff(output_x)
        # 返回处理后的输出
        return output_x


class XLNetPreTrainedModel(PreTrainedModel):
    """
    处理权重初始化和下载预训练模型的抽象类。
    """

    # 设置 XLNetConfig 作为配置类
    config_class = XLNetConfig
    # 加载 TensorFlow 权重的函数
    load_tf_weights = load_tf_weights_in_xlnet
    # 设置基础模型前缀为 "transformer"
    base_model_prefix = "transformer"
    def _init_weights(self, module):
        """Initialize the weights."""
        # 如果 module 是 nn.Linear 类型
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为 0，标准差为 self.config.initializer_range
            # 这里与 TensorFlow 版本稍有不同，TensorFlow 使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        
        # 如果 module 是 nn.Embedding 类型
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为 0，标准差为 self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有 padding_idx，则将其对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        
        # 如果 module 是 nn.LayerNorm 类型
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为 1
            module.weight.data.fill_(1.0)
        
        # 如果 module 是 XLNetRelativeAttention 类型
        elif isinstance(module, XLNetRelativeAttention):
            # 对以下参数使用正态分布进行初始化，均值为 0，标准差为 self.config.initializer_range
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
        
        # 如果 module 是 XLNetModel 类型
        elif isinstance(module, XLNetModel):
            # 初始化 mask_emb 参数，使用正态分布，均值为 0，标准差为 self.config.initializer_range
            module.mask_emb.data.normal_(mean=0.0, std=self.config.initializer_range)
@dataclass
class XLNetModelOutput(ModelOutput):
    """
    Output type of [`XLNetModel`].

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_predict, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.

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

    last_hidden_state: torch.FloatTensor
    mems: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class XLNetLMHeadModelOutput(ModelOutput):
    """
    Output type of [`XLNetLMHeadModel`].

    This class serves as the output specification for the XLNet language model head.
    """
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

    # Optional attributes representing outputs from the language model
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mems: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
# 定义一个数据类，表示 XLNet 模型用于序列分类任务的输出
@dataclass
class XLNetForSequenceClassificationOutput(ModelOutput):
    """
    Output type of [`XLNetForSequenceClassification`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `label` is provided):
            分类（或者当 `config.num_labels==1` 时的回归）损失值。
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            分类（或者当 `config.num_labels==1` 时的回归）得分（softmax 之前的分数）。
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            包含预先计算的隐藏状态列表，可用于加速序列解码。
            传给模型的 token id 不应该作为 `input_ids` 再次传入，因为它们已经被计算过。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            包含模型每一层的隐藏状态的元组，包括初始的嵌入输出。
            形状为 `(batch_size, sequence_length, hidden_size)`。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            包含每一层的注意力权重的元组。
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
    """

    # 表示损失值，当提供 `label` 时返回
    loss: Optional[torch.FloatTensor] = None
    # 表示分类（或回归）得分，形状为 `(batch_size, config.num_labels)`
    logits: torch.FloatTensor = None
    # 包含预先计算的隐藏状态列表，长度为 `config.n_layers`
    mems: Optional[List[torch.FloatTensor]] = None
    # 包含每一层的隐藏状态的元组，包括初始的嵌入输出
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 包含每一层的注意力权重的元组
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# 定义一个数据类，表示 XLNet 模型用于标记分类任务的输出
@dataclass
class XLNetForTokenClassificationOutput(ModelOutput):
    """
    Output type of [`XLNetForTokenClassificationOutput`].
    """
    # 定义函数的参数和返回类型注解，这些参数是用于处理分类任务的神经网络模型的输出和中间状态
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            分类损失值。可选的，当提供了 `labels` 参数时返回。
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            分类得分（SoftMax 之前的分数）。
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            包含预先计算的隐藏状态的列表。可用于加速顺序解码。已经计算过其过去的标记 ID 不应作为 `input_ids` 传递给模型。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型每一层输出的隐藏状态的元组，包括初始嵌入输出。

            模型每一层输出的隐藏状态和初始嵌入输出的 torch.FloatTensor，形状为 `(batch_size, sequence_length, hidden_size)`。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            注意力权重的元组，用于计算自注意力头中的加权平均值。

            注意力 softmax 后的注意力权重，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
    """

    # 初始化函数的返回值，所有值都是可选的
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mems: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
@dataclass
class XLNetForMultipleChoiceOutput(ModelOutput):
    """
    Output type of [`XLNetForMultipleChoice`].

    Args:
        loss (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided):
            Classification loss. Represents the loss value associated with the classification task.
        logits (`torch.FloatTensor` of shape `(batch_size, num_choices)`):
            Classification scores (before SoftMax). These scores represent the model's raw output for each choice in a multiple-choice scenario.
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used to speed up sequential decoding by providing already computed hidden states from previous decoding steps.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple containing hidden-states of the model at the output of each layer plus the initial embedding outputs. This helps in accessing intermediate hidden states if needed.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple containing attention weights after the attention softmax. These weights are used to compute the weighted average in the self-attention heads.

            Each element in the tuple corresponds to attention weights from different layers of the model.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mems: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class XLNetForQuestionAnsweringSimpleOutput(ModelOutput):
    """
    Output type of [`XLNetForQuestionAnsweringSimple`].
    
    This class represents the output structure specifically tailored for the task of question answering using XLNet.
    It inherits from `ModelOutput`, providing a standardized way to represent model outputs in this context.
    """
    # 定义函数的参数和返回值的注释文档字符串
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            总的跨度提取损失，是开始位置和结束位置的交叉熵之和。
        start_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length,)`):
            跨度开始位置的分数（SoftMax 之前）。
        end_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length,)`):
            跨度结束位置的分数（SoftMax 之前）。
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            包含预计算隐藏状态的列表。可以用于加速顺序解码。
            这些模型已经计算过的 token id 不应作为 `input_ids` 传递，因为它们已经计算过。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型在每一层输出的隐藏状态的元组。包括初始嵌入输出。
            形状为 `(batch_size, sequence_length, hidden_size)`。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            每一层注意力权重的元组。
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            
    """
    
    # 定义函数内部的变量类型和默认值
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    mems: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
# 定义一个数据类，用于存储 XLNet 问答模型的输出结果，继承自 ModelOutput 类
@dataclass
class XLNetForQuestionAnsweringOutput(ModelOutput):
    
    """
    [`XLNetForQuestionAnswering`] 的输出类型。

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, 如果 `start_positions` 和 `end_positions` 都提供则返回):
            分类损失，作为起始标记、结束标记分类损失的总和（如果提供了 `is_impossible` 则也包括在内）。
        start_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top)`, *optional*, 如果未提供 `start_positions` 或 `end_positions` 则返回):
            针对顶部 config.start_n_top 起始标记可能性（波束搜索）的对数概率。
        start_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top)`, *optional*, 如果未提供 `start_positions` 或 `end_positions` 则返回):
            针对顶部 config.start_n_top 起始标记可能性（波束搜索）的索引。
        end_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, 如果未提供 `start_positions` 或 `end_positions` 则返回):
            针对顶部 `config.start_n_top * config.end_n_top` 结束标记可能性（波束搜索）的对数概率。
        end_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, 如果未提供 `start_positions` 或 `end_positions` 则返回):
            针对顶部 `config.start_n_top * config.end_n_top` 结束标记可能性（波束搜索）的索引。
        cls_logits (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, 如果未提供 `start_positions` 或 `end_positions` 则返回):
            关于答案的 `is_impossible` 标签的对数概率。
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            包含预先计算的隐藏状态。可用于加速顺序解码。已经计算过其过去的令牌 id 不应作为 `input_ids` 传递，因为它们已经计算过。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, 当 `output_hidden_states=True` 时返回或当 `config.output_hidden_states=True` 时返回):
            模型每层输出的隐藏状态的元组（一个用于嵌入的输出 + 每一层的输出），形状为 `(batch_size, sequence_length, hidden_size)`。

            每层输出的隐藏状态加上初始嵌入的输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, 当 `output_attentions=True` 时返回或当 `config.output_attentions=True` 时返回):
            注意力权重的元组（每一层一个），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """
    # 定义可选的损失张量
    loss: Optional[torch.FloatTensor] = None
    # 定义可选的起始位置的对数概率张量
    start_top_log_probs: Optional[torch.FloatTensor] = None
    # 定义可选的起始位置的索引张量
    start_top_index: Optional[torch.LongTensor] = None
    # 定义可选的结束位置的对数概率张量
    end_top_log_probs: Optional[torch.FloatTensor] = None
    # 定义可选的结束位置的索引张量
    end_top_index: Optional[torch.LongTensor] = None
    # 定义可选的分类器输出张量
    cls_logits: Optional[torch.FloatTensor] = None
    # 定义可选的记忆列表，每个元素为张量
    mems: Optional[List[torch.FloatTensor]] = None
    # 定义可选的隐藏状态元组，包含多个张量
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 定义可选的注意力张量元组，包含多个张量
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
# XLNet 模型的文档字符串，描述了模型的继承关系和常见用法
XLNET_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`XLNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# XLNet 模型的输入文档字符串，当前为空
XLNET_INPUTS_DOCSTRING = r"""
"""

# 用于添加文档字符串的装饰器函数，说明了 XLNetModel 类的基本信息和继承关系
@add_start_docstrings(
    "The bare XLNet Model transformer outputting raw hidden-states without any specific head on top.",
    XLNET_START_DOCSTRING,
)
# XLNetModel 类的定义，继承自 XLNetPreTrainedModel 类
class XLNetModel(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化模型参数
        self.mem_len = config.mem_len            # 记忆长度
        self.reuse_len = config.reuse_len        # 复用长度
        self.d_model = config.d_model            # 模型维度
        self.same_length = config.same_length    # 是否相同长度
        self.attn_type = config.attn_type        # 注意力类型
        self.bi_data = config.bi_data            # 是否双向数据
        self.clamp_len = config.clamp_len        # 限制长度
        self.n_layer = config.n_layer            # 层数

        # 初始化词嵌入层、掩码嵌入层、层列表和 dropout 层
        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)  # 词嵌入层
        self.mask_emb = nn.Parameter(torch.FloatTensor(1, 1, config.d_model))  # 掩码嵌入层
        self.layer = nn.ModuleList([XLNetLayer(config) for _ in range(config.n_layer)])  # XLNetLayer 层列表
        self.dropout = nn.Dropout(config.dropout)  # dropout 层

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入词嵌入层
    def get_input_embeddings(self):
        return self.word_embedding

    # 设置输入词嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.word_embedding = new_embeddings

    # 剪枝头部的方法，当前未实现
    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError
    def create_mask(self, qlen, mlen):
        """
        Creates causal attention mask. Float mask where 1.0 indicates masked, 0.0 indicates not-masked.

        Args:
            qlen: Sequence length
            mlen: Mask length

        ::

                  same_length=False: same_length=True: <mlen > < qlen > <mlen > < qlen >
               ^ [0 0 0 0 0 1 1 1 1] [0 0 0 0 0 1 1 1 1]
                 [0 0 0 0 0 0 1 1 1] [1 0 0 0 0 0 1 1 1]
            qlen [0 0 0 0 0 0 0 1 1] [1 1 0 0 0 0 0 1 1]
                 [0 0 0 0 0 0 0 0 1] [1 1 1 0 0 0 0 0 1]
               v [0 0 0 0 0 0 0 0 0] [1 1 1 1 0 0 0 0 0]

        """
        # 创建一个形状为 (qlen, qlen + mlen) 的全为 1 的张量，表示初始的掩码
        mask = torch.ones((qlen, qlen + mlen), device=self.device)
        
        # 如果 same_length 为 True，则需要特殊处理掩码
        if self.same_length:
            # 提取出下三角矩阵，并且将掩码的上三角部分置为 0
            mask_lo = mask[:, :qlen].tril(-1)
            mask.triu_(mlen + 1)
            mask[:, :qlen] += mask_lo
        else:
            # 如果 same_length 不为 True，则只将掩码的上三角部分置为 0
            mask.triu_(mlen + 1)

        return mask

    def cache_mem(self, curr_out, prev_mem):
        # 将当前输出缓存为新的内存状态。

        # 如果 reuse_len 被定义且大于 0，则截取当前输出的前部分
        if self.reuse_len is not None and self.reuse_len > 0:
            curr_out = curr_out[: self.reuse_len]

        # 如果 mem_len 没有被定义或者为 0，则根据情况设置截断点
        if self.mem_len is None or self.mem_len == 0:
            # 如果 use_mems 激活但未定义 mem_len，则模型在推断时的行为类似于 GPT-2，
            # 返回所有过去和当前的隐藏状态。
            cutoff = 0
        else:
            # 如果 use_mems 激活且定义了 mem_len，则模型返回最后的 mem_len 个隐藏状态，
            # 这是训练和生成长文本时的首选设置。
            cutoff = -self.mem_len
        
        # 如果 prev_mem 为 None，则直接使用当前输出的部分作为新的内存
        if prev_mem is None:
            new_mem = curr_out[cutoff:]
        else:
            # 否则将当前输出与之前的内存连接起来，并根据截断点进行截断
            new_mem = torch.cat([prev_mem, curr_out], dim=0)[cutoff:]

        return new_mem.detach()

    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        # 根据位置序列和频率逆序列生成位置编码矩阵

        # 计算正弦和余弦函数的输入
        sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
        
        # 将正弦和余弦函数的结果连接起来形成位置编码
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb[:, None, :]

        # 如果给定了 batch size，则将位置编码矩阵进行扩展
        if bsz is not None:
            pos_emb = pos_emb.expand(-1, bsz, -1)

        return pos_emb
    # 创建相对位置编码。
    freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.int64).float()
    # 计算频率因子，用于相对位置编码
    inv_freq = 1 / torch.pow(10000, (freq_seq / self.d_model))

    if self.attn_type == "bi":
        # 如果是双向注意力机制，设置起始和结束位置
        beg, end = klen, -qlen
    elif self.attn_type == "uni":
        # 如果是单向注意力机制，设置起始和结束位置
        beg, end = klen, -1
    else:
        # 抛出异常，未知的注意力类型
        raise ValueError(f"Unknown `attn_type` {self.attn_type}.")

    if self.bi_data:
        # 创建前向和后向位置序列
        fwd_pos_seq = torch.arange(beg, end, -1.0, dtype=torch.int64).float()
        bwd_pos_seq = torch.arange(-beg, -end, 1.0, dtype=torch.int64).float()

        if self.clamp_len > 0:
            # 对位置序列进行截断
            fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
            bwd_pos_seq = bwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)

        if bsz is not None:
            # 创建前向和后向位置的位置嵌入
            fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz // 2)
            bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bsz // 2)
        else:
            # 创建前向和后向位置的位置嵌入（无批次大小限制）
            fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
            bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)

        # 将前向和后向的位置嵌入拼接在一起
        pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb], dim=1)
    else:
        # 创建前向位置序列
        fwd_pos_seq = torch.arange(beg, end, -1.0, dtype=torch.int64).float()
        if self.clamp_len > 0:
            # 对位置序列进行截断
            fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
        # 创建前向位置的位置嵌入
        pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

    # 返回位置嵌入张量
    return pos_emb
@add_start_docstrings(
    """
    XLNet Model with a language modeling head on top (linear layer with weights tied to the input embeddings).
    """,
    XLNET_START_DOCSTRING,
)
class XLNetLMHeadModel(XLNetPreTrainedModel):
    _tied_weights_keys = ["lm_loss.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.attn_type = config.attn_type  # 从配置中获取注意力类型
        self.same_length = config.same_length  # 从配置中获取是否使用相同长度

        self.transformer = XLNetModel(config)  # 初始化XLNet模型
        self.lm_loss = nn.Linear(config.d_model, config.vocab_size, bias=True)  # 初始化语言建模头部线性层

        # 初始化权重并应用最终处理
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_loss  # 返回语言建模头部的线性层

    def set_output_embeddings(self, new_embeddings):
        self.lm_loss = new_embeddings  # 设置新的输出嵌入层

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, use_mems=None, **kwargs):
        # 在输入的末尾添加一个虚拟标记（该标记不会被注意力机制关注）

        effective_batch_size = input_ids.shape[0]  # 计算有效的批量大小
        dummy_token = torch.zeros((effective_batch_size, 1), dtype=torch.long, device=input_ids.device)  # 创建一个全零的虚拟标记张量

        # 在每次传递中，计算新标记以及最后两个生成标记的注意力值，其余从过去的缓存中重新加载。
        # 纯自回归模型应该有 offset = 1; 使用 offset = 2 似乎计算稍微更好。
        offset = 2

        if past_key_values:
            input_ids = torch.cat([input_ids[:, -offset:], dummy_token], dim=1)  # 如果过去的键值存在，则在末尾添加虚拟标记
        else:
            input_ids = torch.cat([input_ids, dummy_token], dim=1)  # 否则在末尾添加虚拟标记

        # 构建排列掩码，使得之前的标记不会看到最后一个标记
        sequence_length = input_ids.shape[1]  # 获取序列长度
        perm_mask = torch.zeros(
            (effective_batch_size, sequence_length, sequence_length), dtype=torch.float, device=input_ids.device
        )  # 创建全零的排列掩码张量
        perm_mask[:, :, -1] = 1.0  # 最后一个位置设为1.0，表示不允许之前的标记看到最后一个标记

        # 我们只预测最后一个标记
        target_mapping = torch.zeros(
            (effective_batch_size, 1, sequence_length), dtype=torch.float, device=input_ids.device
        )  # 创建全零的目标映射张量
        target_mapping[:, 0, -1] = 1.0  # 最后一个位置设为1.0，表示只预测最后一个标记

        inputs = {
            "input_ids": input_ids,
            "perm_mask": perm_mask,
            "target_mapping": target_mapping,
            "use_mems": use_mems,
        }  # 构建输入字典

        # 如果模型kwargs中定义了过去的键值，则用它进行更快的解码
        if past_key_values:
            inputs["mems"] = tuple(layer_past[:-offset, :, :] for layer_past in past_key_values)  # 为更快的解码使用过去的记忆

        return inputs

    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=XLNetLMHeadModelOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个实例方法 `forward`，用于模型的前向传播
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的token IDs，可选的张量类型
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，可选的张量类型
        mems: Optional[torch.Tensor] = None,  # 记忆缓存，可选的张量类型
        perm_mask: Optional[torch.Tensor] = None,  # 排列掩码，可选的张量类型
        target_mapping: Optional[torch.Tensor] = None,  # 目标映射，可选的张量类型
        token_type_ids: Optional[torch.Tensor] = None,  # token 类型 IDs，可选的张量类型
        input_mask: Optional[torch.Tensor] = None,  # 输入掩码，可选的张量类型
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码，可选的张量类型
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入表示，可选的张量类型
        labels: Optional[torch.Tensor] = None,  # 标签，可选的张量类型
        use_mems: Optional[bool] = None,  # 是否使用记忆缓存，可选的布尔类型
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选的布尔类型
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选的布尔类型
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，可选的布尔类型
        **kwargs,  # 删除 `use_cache` 在 `XLNetModel` 移除时使用
    ):
        pass  # 此处只是定义方法的占位符，实际内容未提供，暂无具体实现

    @staticmethod
    def _reorder_cache(mems: List[torch.Tensor], beam_idx: torch.Tensor) -> List[torch.Tensor]:
        """
        重新排列 `mems` 缓存，如果调用了 `~PreTrainedModel.beam_search` 或 `~PreTrainedModel.beam_sample`，
        这是为了确保在每个生成步骤中，`mems` 与正确的 `beam_idx` 匹配。
        """
        # 使用 `beam_idx` 将每个层级的过去状态重新排序到对应的设备上
        return [layer_past.index_select(1, beam_idx.to(layer_past.device)) for layer_past in mems]
# 定义 XLNet 序列分类/回归模型，其顶部包含一个用于 GLUE 任务的线性层（放在汇总输出之上）
@add_start_docstrings(
    """
    XLNet Model with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g.
    for GLUE tasks.
    """,
    XLNET_START_DOCSTRING,
)
class XLNetForSequenceClassification(XLNetPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 记录类别数量
        self.num_labels = config.num_labels
        # 记录配置信息
        self.config = config

        # 初始化 XLNet 模型
        self.transformer = XLNetModel(config)
        # 序列汇总层
        self.sequence_summary = SequenceSummary(config)
        # 输出层，用于分类任务，将模型输出映射到标签数量
        self.logits_proj = nn.Linear(config.d_model, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=XLNetForSequenceClassificationOutput,
        config_class=_CONFIG_FOR_DOC,
    )
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
        labels: Optional[torch.Tensor] = None,
        use_mems: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # 在 XLNetModel 移除 `use_cache` 时应删除此参数
    ):
    ) -> Union[Tuple, XLNetForSequenceClassificationOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 根据传入参数决定是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用transformer模型处理输入数据，获取transformer的输出结果
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
        # 获取transformer的输出结果中的第一个tensor作为模型输出
        output = transformer_outputs[0]

        # 对模型输出进行序列摘要处理
        output = self.sequence_summary(output)
        # 将摘要后的结果投影到logits空间
        logits = self.logits_proj(output)

        # 初始化损失值为None
        loss = None
        # 如果有传入标签数据
        if labels is not None:
            # 如果配置中未定义问题类型，则根据标签类型和数量设置问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择相应的损失函数
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 对于单标签问题，计算均方误差损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 对于多标签问题，同样计算均方误差损失
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 对于单标签分类问题，使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 对于多标签分类问题，使用带logits的二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不要求返回字典形式的输出，则将损失值和其它输出合并返回
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回经过XLNet模型处理后的输出结果，包括损失、logits、mems、隐藏状态和注意力
        return XLNetForSequenceClassificationOutput(
            loss=loss,
            logits=logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
# 使用 XLNet 模型，在其上面添加一个用于标记分类（例如命名实体识别）的头部，这个头部是隐藏状态输出之上的一个线性层。
@add_start_docstrings(
    """
    XLNet Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    XLNET_START_DOCSTRING,
)
# 定义 XLNetForTokenClassification 类，继承自 XLNetPreTrainedModel
class XLNetForTokenClassification(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 设置标签数量
        self.num_labels = config.num_labels

        # 初始化 XLNet 模型
        self.transformer = XLNetModel(config)
        # 定义分类器，使用线性层将隐藏状态输出转换成指定数量的标签
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，处理输入并返回结果
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=XLNetForTokenClassificationOutput,
        config_class=_CONFIG_FOR_DOC,
    )
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
        labels: Optional[torch.Tensor] = None,
        use_mems: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # 在 XLNetModel 中删除 `use_cache` 时使用
    ) -> Union[Tuple, XLNetForTokenClassificationOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where *num_choices* is the size of the second dimension of the input tensors. (see *input_ids* above)
        """
        # 确定是否使用返回字典，如果未指定则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入参数传递给Transformer模型进行处理
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

        # 从Transformer模型的输出中获取序列输出
        sequence_output = outputs[0]

        # 将序列输出传递给分类器，生成logits
        logits = self.classifier(sequence_output)

        # 初始化损失为None
        loss = None
        # 如果提供了标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不需要返回字典，则返回包含logits和其他输出的元组
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典，则返回XLNetForTokenClassificationOutput对象
        return XLNetForTokenClassificationOutput(
            loss=loss,
            logits=logits,
            mems=outputs.mems,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 带有多选分类头部的 XLNet 模型类定义，用于例如 RACE/SWAG 任务
@add_start_docstrings(
    """
    XLNet Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RACE/SWAG tasks.
    """,
    XLNET_START_DOCSTRING,
)
class XLNetForMultipleChoice(XLNetPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        
        # 初始化 XLNet 模型部分
        self.transformer = XLNetModel(config)
        
        # 序列摘要，用于汇总序列输出
        self.sequence_summary = SequenceSummary(config)
        
        # 线性层，将模型输出映射到一个标量，用于多选分类
        self.logits_proj = nn.Linear(config.d_model, 1)

        # 初始化权重并进行最终处理
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
        **kwargs,  # 当 XLNetModel 中的 `use_cache` 参数移除时删除这部分
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 根据 `return_dict` 参数确定是否返回一个字典对象，如果未指定则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 计算输入的选项数，即第二维的大小，如果输入中不为空则使用 `input_ids.shape[1]`
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 将输入张量展平，以便在批处理维度上进行处理，如果输入不为空，则进行相应的操作
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_input_mask = input_mask.view(-1, input_mask.size(-1)) if input_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用 Transformer 模型进行前向传播，传入相应的参数
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
            **kwargs,
        )

        # 获取 Transformer 输出中的第一个张量，通常是模型输出的主要部分
        output = transformer_outputs[0]

        # 对模型输出进行序列摘要，通常是对序列长度进行汇总或降维
        output = self.sequence_summary(output)

        # 将摘要后的序列输出投影到对应的分类空间，以获得 logits
        logits = self.logits_proj(output)

        # 将 logits 重新整形为二维张量，以便进行多选分类的计算
        reshaped_logits = logits.view(-1, num_choices)

        # 初始化损失为 None
        loss = None

        # 如果提供了 labels，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels.view(-1))

        # 如果不要求返回字典，则重新组织输出的元组
        if not return_dict:
            output = (reshaped_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 构造并返回 XLNetForMultipleChoiceOutput 对象，包含损失、logits 和其他 Transformer 输出
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
class XLNetForQuestionAnsweringSimple(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 初始化 XLNet 模型
        self.transformer = XLNetModel(config)
        # 线性层，用于输出 span 起始和结束的 logits
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=XLNetForQuestionAnsweringSimpleOutput,
        config_class=_CONFIG_FOR_DOC,
    )
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
    ):
        """
        Forward pass for XLNetForQuestionAnsweringSimple.
        """
        # 略，这里是具体的前向传播逻辑，根据输入计算输出
    # 定义一个方法 `forward`，用于模型的前向传播
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的 token IDs，可选的 PyTorch 张量
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，可选的 PyTorch 张量
        mems: Optional[torch.Tensor] = None,  # 记忆（历史隐藏状态），可选的 PyTorch 张量
        perm_mask: Optional[torch.Tensor] = None,  # 排列掩码，可选的 PyTorch 张量
        target_mapping: Optional[torch.Tensor] = None,  # 目标映射，可选的 PyTorch 张量
        token_type_ids: Optional[torch.Tensor] = None,  # token 类型 IDs，可选的 PyTorch 张量
        input_mask: Optional[torch.Tensor] = None,  # 输入掩码，可选的 PyTorch 张量
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码，可选的 PyTorch 张量
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入表示，可选的 PyTorch 张量
        start_positions: Optional[torch.Tensor] = None,  # 起始位置，可选的 PyTorch 张量
        end_positions: Optional[torch.Tensor] = None,  # 结束位置，可选的 PyTorch 张量
        is_impossible: Optional[torch.Tensor] = None,  # 是否不可能的标记，可选的 PyTorch 张量
        cls_index: Optional[torch.Tensor] = None,  # 分类索引，可选的 PyTorch 张量
        p_mask: Optional[torch.Tensor] = None,  # 部分掩码，可选的 PyTorch 张量
        use_mems: Optional[bool] = None,  # 是否使用记忆的标志，可选的布尔值
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选的布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，可选的布尔值
        **kwargs,  # 当 `use_cache` 在 XLNetModel 中被移除时删除这部分内容
```