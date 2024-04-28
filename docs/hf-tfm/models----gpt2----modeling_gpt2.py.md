# `.\models\gpt2\modeling_gpt2.py`

```
# 设置编码格式为 UTF-8
# 版权声明，声明对代码版权拥有权
# 版权声明，声明对代码版权拥有权
# 版权声明，授权条款为 Apache License Version 2.0
# 获取并查看 Apache License 版本 2.0 条款
# 分发方式是“AS IS”，没有任何形式的保证或条件
# 查看具体语言所适用的条件和限制
# 引入需要的库和模块
# 引入需要的库和模块
# 引入需要的库和模块
# 引入需要的库和模块
# 引入需要的库和模块
# 引入需要的库和模块
# 引入需要的库和模块
# 引入需要的库和模块
# 引入需要的库和模块
# 引入需要的库和模块
# 引入需要的库和模块
# 引入需要的库和模块
# 引入需要的库和模块
# 引入需要的库和模块
# 引入需要的库和模块
# 引入需要的库和模块
# 引入需要的库和模块
# 引入需要的库和模块
# 引入需要的库和模块
# 引入需要的库和模块
# 设置日志记录器
# 定义文档中的检查点名称
# 定义文档中的配置名称
# GPT-2 预训练模型存档列表
# 函数：从 TensorFlow 加载权重到 PyTorch 模型中
def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    # 尝试导入模块
    try:
        import re
        import tensorflow as tf
    except ImportError:
        # 如果导入失败，打印错误信息并抛出异常
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    # 获取 TensorFlow 检查点文件路径
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # 从 TF 模型加载权重
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        # 加载 TF 模型的变量
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        # 将加载的权重压缩处理后加入数组中
        arrays.append(array.squeeze())
    # 遍历names和arrays两个列表中的元素，分别为name和array
    for name, array in zip(names, arrays):
        # 去除name中前6个字符"model/"，即跳过"model/"
        name = name[6:]
        # 按"/"分割name，得到一个列表
        name = name.split("/")
        # 初始化指针为model
        pointer = model
        # 遍历name列表中的元素m_name
        for m_name in name:
            # 使用正则表达式匹配是否是由字母后跟数字组成的字符串
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                # 如果是，则以数字分割字符串，得到scope_names列表
                scope_names = re.split(r"(\d+)", m_name)
            else:
                # 否则将m_name作为列表中唯一元素
                scope_names = [m_name]
            # 根据scope_names的第一个元素确定指针的移动
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            # 如果scope_names列表的长度超过1
            if len(scope_names) >= 2:
                # 将第二个元素解析为整数
                num = int(scope_names[1])
                # 指针指向索引为num的元素
                pointer = pointer[num]
        # 尝试判断pointer和array的形状是否相同
        try:
            if pointer.shape != array.shape:
                # 如果不相同则抛出ValueError并输出相关信息
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except ValueError as e:
            # 将指针和数组形状加入ValueError的参数中，并重新抛出异常
            e.args += (pointer.shape, array.shape)
            raise
        # 输出日志信息，初始化PyTorch权重为name
        logger.info(f"Initialize PyTorch weight {name}")
        # 使用array数组数据作为指针的数据
        pointer.data = torch.from_numpy(array)
    # 返回model对象
    return model
# 定义 GPT2Attention 类，继承自 nn.Module
class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()

        # 获取配置中的最大位置嵌入
        max_positions = config.max_position_embeddings
        # 注册缓冲区，包含一个下三角矩阵作为偏置
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        # 注册缓冲区，包含一个负值作为掩码偏置
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        # 获取隐藏大小和注意力头数
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        # 获取配置中的注意力权重缩放参数和交叉注意力标志
        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # 层级注意力缩放、重新排序和升级
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        # 如果是交叉注意力，创建 c_attn 和 q_attn
        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        # 创建 c_proj
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        # 创建注意力和残差的 Dropout 层
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # 存储被剪枝的头部
        self.pruned_heads = set()

    # 剪枝头部
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 寻找可剪枝的头部和索引
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # 剪枝 Conv1D 层
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # 更新超参数
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)
    # 计算注意力权重，使用查询(query)、键(key)和值(value)
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    # 如果设置了缩放注意力权重标志，则对注意力权重进行缩放
    if self.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

    # 如果设置了按层索引逆向缩放注意力权重标志，则按层索引逆向缩放注意力权重
    if self.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(self.layer_idx + 1)

    # 如果不是跨层注意力，则实施因果掩码
    if not self.is_cross_attention:
        # 计算查询长度和键长度
        query_length, key_length = query.size(-2), key.size(-2)
        # 生成因果掩码
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        # 设定掩码值为最小浮点数，需要转换为张量，以避免类型错误和设备不匹配错误
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        # 对注意力权重进行掩码操作
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

    # 如果存在注意力掩码，则应用注意力掩码
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # 对注意力权重进行 softmax 操作
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # 将注意力权重的数据类型降级（如果有必要），回到值的数据类型（如果使用混合精度）-- 否则不进行任何操作
    attn_weights = attn_weights.type(value.dtype)
    # 对注意力权重应用注意力丢弃(dropout)
    attn_weights = self.attn_dropout(attn_weights)

    # 如果需要，对注意力权重进行头部掩码
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    # 计算注意力输出
    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights
```  
    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # 使用 `torch.baddbmm` 进行矩阵乘法计算（使用 alpha 参数进行缩放，效率更高 -- 参考自 Megatron-LM）
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # 为 `baddbmm` 预先分配 attn_weights
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # 计算缩放因子
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (关闭 autocast) 并重新排序 (通过缩放 K 为 1 / sqrt(dk))
        with autocast(enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # 若为非跨 attention 层，则实现原因掩码
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # 需要将其转换为张量，否则会出现错误: `RuntimeError: expected scalar type float but found double`.
            # 需要在相同设备上，否则会出现错误: `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # 应用注意力掩码
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # 如果必要，将其降级回 V 的数据类型（如果处于混合精度模式）-- 否则不执行任何操作
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # 如果需要，掩盖头部
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        将 hidden_size 维度分割为 attn_head_size 和 num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        # 交换张量的维度顺序，将注意力头维度和注意力头数量维度合并到隐藏状态维度
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # 计算新的形状，将注意力头数量和注意力头大小相乘得到新的隐藏状态维度
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        # 重新调整张量的形状
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        # 如果存在编码器的隐藏状态
        if encoder_hidden_states is not None:
            # 如果类被用作交叉注意力，则需要定义权重 `q_attn`
            if not hasattr(self, "q_attn"):
                # 抛出错误提示缺少 `q_attn` 权重
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            # 计算查询向量
            query = self.q_attn(hidden_states)
            # 分割编码器的键值向量
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            # 使用编码器的注意力掩码
            attention_mask = encoder_attention_mask
        else:
            # 计算自注意力机制的查询、键、值向量
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        # 将查询向量分割成多个注意力头
        query = self._split_heads(query, self.num_heads, self.head_dim)
        # 将键向量分割成多个注意力头
        key = self._split_heads(key, self.num_heads, self.head_dim)
        # 将值向量分割成多个注意力头
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # 如果存在过去的层，则将过去的键值向量与当前的键值向量拼接在一起
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        # 如果使用缓存，则保存当前的键值向量
        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # 如果重新排序和提升注意力激活被启用
        if self.reorder_and_upcast_attn:
            # 执行重新排序和提升注意力激活
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            # 执行注意力机制
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # 合并注意力头到隐藏状态维度
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        # 线性变换
        attn_output = self.c_proj(attn_output)
        # 残差连接和 Dropout
        attn_output = self.resid_dropout(attn_output)

        # 输出结果包括注意力输出和可能的缓存
        outputs = (attn_output, present)
        # 如果需要输出注意力权重，则将注意力权重添加到输出结果中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回结果
        return outputs  # a, present, (attentions)
# 定义 GPT2MLP 类，继承自 nn.Module
class GPT2MLP(nn.Module):
    # 初始化函数，接收 intermediate_size 和 config 两个参数
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        # 创建 Conv1D 对象，输出维度为 intermediate_size，输入维度为 embed_dim（hidden_size）
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        # 创建 Conv1D 对象，输出维度为 embed_dim（hidden_size），输入维度为 intermediate_size
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        # 根据 config.activation_function 选择激活函数
        self.act = ACT2FN[config.activation_function]
        # 创建 nn.Dropout 对象，丢弃概率为 config.resid_pdrop
        self.dropout = nn.Dropout(config.resid_pdrop)

    # 前向传播函数，接收 hidden_states 参数，返回 torch.FloatTensor 类型的结果
    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        # 将输入的 hidden_states 经过 self.c_fc、self.act、self.c_proj、self.dropout 依次处理
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # 返回处理后的结果
        return hidden_states


# 定义 GPT2Block 类，继承自 nn.Module
class GPT2Block(nn.Module):
    # 初始化函数，接收 config 和 layer_idx 两个参数
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        # 创建 nn.LayerNorm 对象，输入维度为 hidden_size，eps 为 config.layer_norm_epsilon
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # 创建 GPT2Attention 对象
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        # 创建 nn.LayerNorm 对象，输入维度为 hidden_size，eps 为 config.layer_norm_epsilon
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        # 如果 config.add_cross_attention 为 True，则创建 GPT2Attention 对象
        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)
            # 创建 nn.LayerNorm 对象，输入维度为 hidden_size，eps 为 config.layer_norm_epsilon
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        # 创建 GPT2MLP 对象，输入维度为 inner_dim，config 为初始化参数
        self.mlp = GPT2MLP(inner_dim, config)

    # 前向传播函数，接收多个参数，返回 torch.FloatTensor 类型的结果
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    # 定义transformer层的前向传播函数
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        # 保存残差连接用的原始隐藏状态
        residual = hidden_states
        # Layer Normalization 正则化隐藏状态
        hidden_states = self.ln_1(hidden_states)
        # 调用注意力机制层来处理隐藏状态
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 获取注意力机制处理结果
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        # 保存输出结果
        outputs = attn_outputs[1:]
        # 实现残差连接
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # 对交叉注意力进行处理
            if not hasattr(self, "crossattention"):
                # 如果没有交叉注意力层，抛出错误
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            # 保存残差连接用的原始隐藏状态
            residual = hidden_states
            # Layer Normalization 正则化隐藏状态
            hidden_states = self.ln_cross_attn(hidden_states)
            # 调用交叉注意力层来处理隐藏状态
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            # 获取交叉注意力处理结果
            attn_output = cross_attn_outputs[0]
            # 实现残差连接
            hidden_states = residual + attn_output
            # 保存输出结果，包括交叉注意力权重
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        # 保存残差连接用的原始隐藏状态
        residual = hidden_states
        # Layer Normalization 正则化隐藏状态
        hidden_states = self.ln_2(hidden_states)
        # MLP 网络处理隐藏状态
        feed_forward_hidden_states = self.mlp(hidden_states)
        # 实现残差连接
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            # 如果使用缓存，添加当前隐藏状态到���出结果
            outputs = (hidden_states,) + outputs
        else:
            # 如果不使用缓存，添加当前隐藏状态到输出结果中排除第一个元素
            outputs = (hidden_states,) + outputs[1:]

        # 返回输出结果，包括隐藏状态，现在时刻缓存，注意力权重和交叉注意力权重
        return outputs  # hidden_states, present, (attentions, cross_attentions)
class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类
    config_class = GPT2Config
    # 加载 TensorFlow 权重的方法
    load_tf_weights = load_tf_weights_in_gpt2
    # 基础模型的前缀
    base_model_prefix = "transformer"
    # 是否可并行化
    is_parallelizable = True
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True
    # 不需要拆分的模块列表
    _no_split_modules = ["GPT2Block"]
    # 跳过键的设备放置
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # 对线性层和一维卷积层的权重进行初始化
            # 与 TF 版本略有不同，TF 版本使用截断正态分布进行初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果存在偏置项，则将其初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 对嵌入层的权重进行初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果存在填充索引，则将其对应的权重初始化为零
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 对层归一化层的权重进行初始化
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # 根据 OpenAI GPT-2 论文方案重新初始化选定的权重：
        #   > 一种修改过的初始化方法，考虑了模型深度的残差路径上的累积。初始化时通过因子 1/√N 缩放残差层的权重，其中 N 是残差层的数量。
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # 参考（Megatron-LM）：https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # 特殊的缩放初始化 --> 每个 Transformer 块有 2 个层归一化
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))


@dataclass
class GPT2DoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.
    """
    # loss 参数，类型为 torch.FloatTensor，形状为 (1,)，可选的，当提供了 labels 参数时返回
    Language modeling loss.
    loss: Optional[torch.FloatTensor] = None

    # mc_loss 参数，类型为 torch.FloatTensor，形状为 (1,)，可选的，当提供了 mc_labels 参数时返回
    Multiple choice classification loss.
    mc_loss: Optional[torch.FloatTensor] = None

    # logits 参数，类型为 torch.FloatTensor，形状为 (batch_size, num_choices, sequence_length, config.vocab_size)
    Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    logits: torch.FloatTensor = None

    # mc_logits 参数，类型为 torch.FloatTensor，形状为 (batch_size, num_choices)
    Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
    mc_logits: torch.FloatTensor = None

    # past_key_values 参数，类型为 Tuple[Tuple[torch.FloatTensor]]，可选的，当 use_cache=True 被传入或者 config.use_cache=True 时返回
    Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

    # hidden_states 参数，类型为 tuple(torch.FloatTensor)，可选的，当 output_hidden_states=True 被传入或者 config.output_hidden_states=True 时返回
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

    # attentions 参数，类型为 tuple(torch.FloatTensor)，可选的，当 output_attentions=True 被传入或者 config.output_attentions=True 时返回
    GPT2Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    attentions: Optional[Tuple[torch.FloatTensor]] = None
GPT2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GPT2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GPT2_INPUTS_DOCSTRING = r"""
"""
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.

    Args:
        device_map (`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the gpt2 models have the
            following number of attention modules:

                - gpt2: 12
                - gpt2-medium: 24
                - gpt2-large: 36
                - gpt2-xl: 48

    Example:

    ```python
    # Here is an example of a device map on a machine with 4 GPUs using gpt2-xl, which has a total of 48 attention modules:
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8],
        1: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        2: [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
        3: [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
    }
    model.parallelize(device_map)
    ```
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example:

    ```python
    # On a 4 GPU machine with gpt2-large:
    model = GPT2LMHeadModel.from_pretrained("gpt2-large")
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7],
        1: [8, 9, 10, 11, 12, 13, 14, 15],
        2: [16, 17, 18, 19, 20, 21, 22, 23],
        3: [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    }
    model.parallelize(device_map)  # Splits the model across several devices
    model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```
"""

@add_start_docstrings(
    # GPT2 模型的基本变换器，输出未经任何特定头部处理的原始隐藏状态。
    # GPT2 模型的开始文档字符串
# 定义 GPT2Model 类，继承自 GPT2PreTrainedModel 类
class GPT2Model(GPT2PreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__(config)

        # 设置嵌入维度为 config.hidden_size
        self.embed_dim = config.hidden_size

        # 创建词嵌入层
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        # 创建位置嵌入层
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        # 创建丢弃层
        self.drop = nn.Dropout(config.embd_pdrop)
        # 创建多层 Transformer block
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        # 创建 LayerNorm 层，使用 embed_dim 作为参数
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # 模型并行
        self.model_parallel = False
        self.device_map = None
        # 是否启用梯度检查点
        self.gradient_checkpointing = False

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # 检查设备映射的有效性
        warnings.warn(
            "`GPT2Model.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your"
            " model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'h.0': 0, 'h.1': 1,"
            " ...}",
            FutureWarning,
        )
        # 如果 device_map 为 None，则使用默认的平衡映射
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        # 检查设备映射的有效性
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        # 设置第一个设备为 "cpu" 或者设备映射的最小键
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        # 设置最后一个设备为设备映射的最大键
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # 将词嵌入层和位置嵌入层移到第一个设备
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        # 将每个 block 移到相应的设备
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # 将 ln_f 移到最后一个设备
        self.ln_f = self.ln_f.to(self.last_device)

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        # 关闭模型并行
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        # 将词嵌入层和位置嵌入层移到 CPU
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        # 将每个 block 移到 CPU
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        # 将 ln_f 移到 CPU
        self.ln_f = self.ln_f.to("cpu")
        # 释放 CUDA 缓存
        torch.cuda.empty_cache()

    # 获取输入词嵌入
    def get_input_embeddings(self):
        return self.wte

    # 设置输入词嵌入
    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings
    # 对模型的头部进行修剪，heads_to_prune: {层号: 需要修剪的头部列表}
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        # 遍历需要修剪的层和头部，对模型的注意力头部进行修剪
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    # 前向传播函数，根据输入参数调用模型进行前向推断
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的token ID张量
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # 过去的键值对元组
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力遮罩张量
        token_type_ids: Optional[torch.LongTensor] = None,  # token类型ID张量
        position_ids: Optional[torch.LongTensor] = None,  # 位置ID张量
        head_mask: Optional[torch.FloatTensor] = None,  # 头部遮罩张量
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入张量
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器隐藏状态张量
        encoder_attention_mask: Optional[torch.FloatTensor] = None,  # 编码器注意力遮罩张量
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典型结果
# 使用add_start_docstrings函数添加模型的文档字符串，包括GPT2模型转换器及其顶部的语言建模头部的描述
class GPT2LMHeadModel(GPT2PreTrainedModel):
    # 定义权重绑定的键列表
    _tied_weights_keys = ["lm_head.weight"]

    # 初始化函数，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 创建GPT2模型实例
        self.transformer = GPT2Model(config)
        # 创建线性层实例，用于语言建模
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用add_start_docstrings函数添加并行处理相关的文档字符串
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    # 定义并行处理函数，接受设备映射参数
    def parallelize(self, device_map=None):
        # 引发未来警告，因为parallelize函数将在v5中删除
        warnings.warn(
            "`GPT2LMHeadModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should load"
            " your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'transformer.h.0':"
            " 0, 'transformer.h.1': 1, ...}",
            FutureWarning,
        )
        # 如果device_map为None，则使用获取设备映射函数获取设备映射
        # 否则使用传入的设备映射
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        # 断言设备映射的正确性
        assert_device_map(self.device_map, len(self.transformer.h))
        # 调用transformer的并行化方法
        self.transformer.parallelize(self.device_map)
        # 将lm_head移动到transformer的首选设备
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        # 设置模型并行标志为True
        self.model_parallel = True

    # 使用add_start_docstrings函数添加取消并行处理相关的文档字符串
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    # 定义取消并行处理函数
    def deparallelize(self):
        # 引发未来警告，因为deparallelize函数将在v5中删除
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        # 调用transformer的取消并行化方法
        self.transformer.deparallelize()
        # 将transformer和lm_head移动到CPU
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        # 设置模型并行标志为False
        self.model_parallel = False
        # 释放CUDA缓存
        torch.cuda.empty_cache()

    # 获取输出嵌入函数
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入函数，接受新的嵌入参数
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    # 为生成准备输入数据
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # 获取token_type_ids参数，如果不存在则为None
        token_type_ids = kwargs.get("token_type_ids", None)
        # 如果存在past_key_values
        if past_key_values:
            # 获取past_key_values的长度
            past_length = past_key_values[0][0].shape[2]
            # 如果输入的input_ids长度大于past_length
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认保留最后一个ID
                remove_prefix_length = input_ids.shape[1] - 1
            # 移除已经处理过的token
            input_ids = input_ids[:, remove_prefix_length:]
            # 如果token_type_ids不为空，只保留相关部分
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]
        # 获取attention_mask和position_ids参数
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        # 如果存在attention_mask且position_ids为空
        if attention_mask is not None and position_ids is None:
            # 实时生成position_ids用于批量生成
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            # 如果存在past_key_values，则只保留相关部分
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None
        # 如果输入了inputs_embeds，并且不存在past_key_values
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 更新model_inputs中的参数
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        # 返回model_inputs
        return model_inputs

    # 追加模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    # 追加代码示例的文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 此函数定义了一个神经网络前向传播的方法，接受各种输入参数
    # input_ids: 用于编码输入文本的 token ID，类型为 LongTensor，可选
    # past_key_values: 用于存储前一次注意力机制的 key 和 value，类型为 Tuple[Tuple[torch.Tensor]]，可选
    # attention_mask: 用于掩盖无意义序列的注意力掩码，类型为 FloatTensor，可选
    # token_type_ids: 用于区分不同句子的 token 类型 ID，类型为 LongTensor，可选
    # position_ids: 用于指定每个 token 的位置 ID，类型为 LongTensor，可选
    # head_mask: 用于屏蔽某些注意力头的头掩码，类型为 FloatTensor，可选
    # inputs_embeds: 用于提供自定义的输入嵌入，类型为 FloatTensor，可选
    # encoder_hidden_states: 编码器的隐藏状态，类型为 torch.Tensor，可选
    # encoder_attention_mask: 编码器的注意力掩码，类型为 FloatTensor，可选
    # labels: 用于训练的标签，类型为 LongTensor，可选
    # use_cache: 是否使用缓存，类型为 bool，可选
    # output_attentions: 是否输出注意力权重，类型为 bool，可选
    # output_hidden_states: 是否输出隐藏状态，类型为 bool，可选
    # return_dict: 是否返回结果字典，类型为 bool，可选
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # 确定是否返回字典类型的结果，若未指定则使用模型配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用transformer模型进行前向传播
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取transformer模型的输出隐藏状态
        hidden_states = transformer_outputs[0]

        # 设置模型并行时的设备
        if self.model_parallel:
            # 将当前GPU设备设置为transformer模型所在设备的第一个设备
            torch.cuda.set_device(self.transformer.first_device)
            # 将隐藏状态移动到LM头的设备上
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        # 计算LM头的logits
        lm_logits = self.lm_head(hidden_states)

        # 计算损失
        loss = None
        if labels is not None:
            # 将标签移动到正确的设备以启用模型并行计算
            labels = labels.to(lm_logits.device)
            # 将logits向左移动一个位置以匹配标签的位置
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 将标签展平为1维向量
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            # 如果不返回字典类型的结果，则将结果组装为元组
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回带有交叉注意力的CausalLMOutputWithCrossAttentions对象
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
        ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        # 返回类型为元组中嵌套元组的形式
        return tuple(
            # 对past_key_values中的每个元素进行处理
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
@add_start_docstrings(
    """
    The GPT2 Model transformer with a language modeling and a multiple-choice classification head on top e.g. for
    RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the
    input embeddings, the classification head takes as input the input of a specified classification token index in the
    input sequence).
    """,
    GPT2_START_DOCSTRING,
)
class GPT2DoubleHeadsModel(GPT2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        # 初始化 GPT2DoubleHeadsModel 类
        self.transformer = GPT2Model(config)
        # 定义语言建模头
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 定义多项选择头
        self.multiple_choice_head = SequenceSummary(config)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`GPT2DoubleHeadsModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should"
            " load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your"
            " own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'transformer.h.0': 0, 'transformer.h.1': 1, ...}",
            FutureWarning,
        )
        # 并行化处理
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        # 并行化处理
        self.transformer.parallelize(self.device_map)
        # 将语言建模头和多项选择头移到第一个设备上
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.multiple_choice_head = self.multiple_choice_head.to(self.transformer.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        # 反并行化处理
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.multiple_choice_head = self.multiple_choice_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    # 准备用于生成的输入参数
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # 从 kwargs 中获取 token 类型 ID，如果没有提供则默认为 None
        token_type_ids = kwargs.get("token_type_ids", None)
        # 如果存在之前的 key-value 键值对
        if past_key_values:
            # 获取过去键值对中的序列长度
            past_length = past_key_values[0][0].shape[2]

            # 如果当前输入 ID 的维度大于过去的长度，则需要移除前缀
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认行为：保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 剪切输入 ID，以适应已经处理的过去长度
            input_ids = input_ids[:, remove_prefix_length:]
            # 如果 token 类型 ID 不为空，则同样裁剪
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        # 从 kwargs 中获取注意力掩码
        attention_mask = kwargs.get("attention_mask", None)
        # 从 kwargs 中获取位置 ID
        position_ids = kwargs.get("position_ids", None)

        # 如果有注意力掩码但没有位置 ID，根据注意力掩码计算位置 ID
        if attention_mask is not None and position_ids is None:
            # 通过累积和生成位置 ID
            position_ids = attention_mask.long().cumsum(-1) - 1
            # 使用掩码调整位置 ID，无效位置设为 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            # 如果存在过去的键值对，需要对位置 ID 进行裁剪
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            # 如果没有指定位置 ID，设置为 None
            position_ids = None

        # 返回包含所有相关参数的字典
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    # 使用文档字符串装饰器为模型的 forward 函数添加文档字符串
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    # 使用文档字符串装饰器替换返回类型的文档说明
    @replace_return_docstrings(output_type=GPT2DoubleHeadsModelOutput, config_class=_CONFIG_FOR_DOC)
    # 定义前向传播函数，包含各种可能的输入参数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        mc_token_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mc_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    # 静态方法，用于在使用 beam search 时重新排列缓存的键值对
    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    # 定义一个函数，用于重新排序 `past_key_values` 缓存，如果调用了 [`~PreTrainedModel.beam_search`] 或 [`~PreTrainedModel.beam_sample`]。
    # 这是为了在每个生成步骤中将 `past_key_values` 与正确的 beam_idx 匹配。
    def re_order_cache(self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        # 返回一个元组，其中包含重新排序后的每层的缓存
        return tuple(
            # 对于每一层的缓存，使用 beam_idx 在第一个维度上进行索引选择，并将结果移动到与 past_state 相同的设备上
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            # 对于所有层的缓存，遍历 past_key_values
            for layer_past in past_key_values
        )
# 使用 add_start_docstrings 函数添加类的说明文档
@add_start_docstrings(
    """
    The GPT2 Model transformer with a sequence classification head on top (linear layer).

    [`GPT2ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    GPT2_START_DOCSTRING,
)
# 定义 GPT2ForSequenceClassification 类，继承自 GPT2PreTrainedModel
class GPT2ForSequenceClassification(GPT2PreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # 获取标签数量
        self.transformer = GPT2Model(config)  # GPT2 模型
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)  # 线性层

        # Model parallel
        self.model_parallel = False  # 模型并行策略
        self.device_map = None  # 设备映射

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)  # 添加前向传播的说明文档
    @add_code_sample_docstrings(  # 添加代码示例的说明文档
        checkpoint="microsoft/DialogRPT-updown",
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播函数
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
@add_start_docstrings(  # 使用 add_start_docstrings 函数添加类的说明文档
    """
    GPT2 Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    GPT2_START_DOCSTRING,
)
# 定义 GPT2ForTokenClassification 类，继承自 GPT2PreTrainedModel
class GPT2ForTokenClassification(GPT2PreTrainedModel):
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__(config)
        # 设置分类标签的数量
        self.num_labels = config.num_labels

        # 使用给定的配置初始化 GPT2 模型
        self.transformer = GPT2Model(config)
        # 检查是否存在分类器的丢弃率配置，若不存在则使用隐藏层丢弃率配置，默认为0.1
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        # 添加一个丢弃层，用于模型训练时的丢弃处理
        self.dropout = nn.Dropout(classifier_dropout)
        # 添加一个线性分类器，用于分类任务的输出
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 模型并行化配置
        self.model_parallel = False
        self.device_map = None

        # 初始化权重并应用最终处理
        self.post_init()

    # 正向传播方法，执行模型的前向计算
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    # fmt: off
    @add_code_sample_docstrings(
        checkpoint="brad1141/gpt2-finetuned-comp2",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_loss=0.25,
        expected_output=[
            "Lead",
            "Lead",
            "Lead",
            "Position",
            "Lead",
            "Lead",
            "Lead",
            "Lead",
            "Lead",
            "Lead",
            "Lead",
            "Lead",
        ],
    )
    # fmt: on
    # 正向传播方法，接受各种输入参数进行前向计算
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
    # 定义返回类型：可以是元组或 TokenClassifierOutput
        ) -> Union[Tuple, TokenClassifierOutput]:
            r"""
            # 参数注释，说明 `labels` 的用途和期望的形状
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                # 描述 `labels` 参数的含义和使用
                # 对于分类/回归损失计算的标签。索引应在 `[0, ..., config.num_labels - 1]` 范围内。
                # 如果 `config.num_labels == 1` 则计算回归损失（均方误差损失），
                # 如果 `config.num_labels > 1` 则计算分类损失（交叉熵损失）。
            """
            # 根据输入的 `return_dict` 或配置决定是否返回字典格式
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
            # 调用 transformer 模型，传入所有必要参数，并根据配置选择性传递一些可选参数
            transformer_outputs = self.transformer(
                input_ids,  # 输入的 token IDs
                past_key_values=past_key_values,  # 用于模型并行化或长序列处理的先前 key 和 value
                attention_mask=attention_mask,  # 用于遮蔽不必要的部分
                token_type_ids=token_type_ids,  # 用于区分输入的不同部分
                position_ids=position_ids,  # 位置编码 ID
                head_mask=head_mask,  # 用于遮蔽 transformer 头
                inputs_embeds=inputs_embeds,  # 直接输入嵌入层
                use_cache=use_cache,  # 是否使用缓存
                output_attentions=output_attentions,  # 是否输出注意力权重
                output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
                return_dict=return_dict,  # 是否返回字典格式
            )
    
            # 获取 transformer 输出的隐藏状态
            hidden_states = transformer_outputs[0]
            # 对隐藏状态应用 dropout，防止过拟合
            hidden_states = self.dropout(hidden_states)
            # 对隐藏状态进行分类，生成 logits
            logits = self.classifier(hidden_states)
    
            # 初始化损失为 None
            loss = None
            if labels is not None:
                # 将标签转换到与 logits 相同的设备上
                labels = labels.to(logits.device)
                # 创建交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                # 计算损失：将 logits 和 labels 展平，以计算交叉熵损失
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
            # 如果不返回字典，则返回元组
            if not return_dict:
                # 输出结果，包括 logits 和 transformer 的附加输出
                output = (logits,) + transformer_outputs[2:]
                # 如果存在损失，则将其与输出一起返回
                return ((loss,) + output) if loss is not None else output
    
            # 如果返回字典格式，则返回 TokenClassifierOutput 对象
            return TokenClassifierOutput(
                loss=loss,  # 返回的损失值
                logits=logits,  # 返回的 logits
                hidden_states=transformer_outputs.hidden_states,  # 返回隐藏状态
                attentions=transformer_outputs.attentions,  # 返回注意力权重
            )
# 用于在 GPT-2 模型的基础上添加一个用于提取式问答任务（如 SQuAD）的分类头部，以计算“起始位置对数”和“终止位置对数”
@add_start_docstrings(
    """
    The GPT-2 Model transformer with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    GPT2_START_DOCSTRING,
)
class GPT2ForQuestionAnswering(GPT2PreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 为分类问题数量设置标签
        self.num_labels = config.num_labels
        # 创建 GPT-2 模型对象
        self.transformer = GPT2Model(config)
        # 创建一个线性层，用于计算起始位置和终止位置的对数
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Model parallel
        # 模型并行处理设置为False
        self.model_parallel = False
        # 设备映射为空
        self.device_map = None

        # 初始化权重并应用最终处理
        self.post_init()

    # 实现模型的前向传播
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        # 确定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给transformer模型进行处理
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

        # 获取transformer模型的输出
        sequence_output = outputs[0]

        # 将输出传递给QA输出层得到logits
        logits = self.qa_outputs(sequence_output)
        # 将logits拆分为起始位置和结束位置的预测值
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        # 如果存在起始和结束位置的标签，则计算损失
        if start_positions is not None and end_positions is not None:
            # 如果处于多GPU环境，增加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            # 有时起始/结束位置超出模型输入，忽略这些位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            # 计算总损失
            total_loss = (start_loss + end_loss) / 2

        # 如果不使用返回字典，则返回输出元组
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 使用返回字典封装输出
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```