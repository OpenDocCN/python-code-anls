# `.\lucidrains\routing-transformer\routing_transformer\encoder_decoder.py`

```
# 导入 re 模块，用于正则表达式操作
# 导入 isfunction 函数，用于检查对象是否为函数
# 导入 torch 模块
# 从 torch 模块中导入 nn 模块
# 从 routing_transformer.routing_transformer 模块中导入 RoutingTransformerLM 类和 update_kmeans_on_backwards 函数
# 从 routing_transformer.autoregressive_wrapper 模块中导入 AutoregressiveWrapper 类

# 定义编码器前缀
ENC_PREFIX = 'enc_'
# 定义解码器前缀
DEC_PREFIX = 'dec_'

# 定义默认函数，如果 x 为 None，则返回 d，如果 d 是函数，则调用函数返回结果
def default(x, d):
    if x is None:
        return d if not isfunction(d) else d()
    return x

# 根据条件 cond 对字典 d 进行分组，返回两个字典
def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

# 判断字符串是否以指定前缀开头
def string_begins_with(prefix, str):
    return bool(re.match(f'^{prefix}', str))

# 根据前缀对字典 d 进行分组
def group_by_key_prefix(prefix, d):
    return group_dict_by_key(lambda x: string_begins_with(prefix, x), d)

# 根据前缀对字典 d 进行分组，并移除前缀
def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: string_begins_with(prefix, x), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items()))
    return kwargs_without_prefix, kwargs

# 提取编码器和解码器的关键字参数
def extract_enc_dec_kwargs(kwargs):
    enc_kwargs, kwargs = group_by_key_prefix_and_remove_prefix(ENC_PREFIX, kwargs)
    dec_kwargs, kwargs = group_by_key_prefix_and_remove_prefix(DEC_PREFIX, kwargs)
    return enc_kwargs, dec_kwargs, kwargs

# 提取并设置编码器和解码器的关键字参数
def extract_and_set_enc_dec_kwargs(kwargs):
    enc_kwargs, dec_kwargs, kwargs = extract_enc_dec_kwargs(kwargs)
    if 'input_mask' in enc_kwargs:
        dec_kwargs.setdefault('context_mask', enc_kwargs['input_mask'])
    return enc_kwargs, dec_kwargs, kwargs

# 定义 RoutingTransformerEncDec 类，继承自 nn.Module
class RoutingTransformerEncDec(nn.Module):
    # 初始化方法
    def __init__(self, dim, ignore_index = None, pad_value = 0, **kwargs):
        super().__init__()
        ignore_index = default(ignore_index, pad_value)
        enc_kwargs, dec_kwargs, _ = extract_enc_dec_kwargs(kwargs)
        
        # 断言编码器关键字参数中不包含 'return_embedding'，否则抛出异常
        assert 'return_embedding' not in enc_kwargs, 'you cannot manually set the return embeddings flag for the encoder'
        # 断言解码器和编码器关键字参数中均不包含 'dim'，否则抛出异常
        assert 'dim' not in dec_kwargs and 'dim' not in enc_kwargs, 'you must set the dim for both encoder and decoder'

        # 设置编码器和解码器的维度
        enc_kwargs['dim'] = dec_kwargs['dim'] = dim
        enc_kwargs['return_embeddings'] = True
        dec_kwargs['causal'] = True
        dec_kwargs['receives_context'] = True
        enc_kwargs['_register_kmeans_update'] = dec_kwargs['_register_kmeans_update'] = False

        # 设置默认的窗口大小
        enc_kwargs.setdefault('window_size', 256)
        dec_kwargs.setdefault('window_size', 256)

        # 创建编码器和解码器对象
        enc = RoutingTransformerLM(**enc_kwargs)
        dec = RoutingTransformerLM(**dec_kwargs)

        self.enc = enc
        self.dec = AutoregressiveWrapper(dec, ignore_index = ignore_index, pad_value = pad_value)

        # 如果解码器可逆，则用户必须手动调用编码器辅助损失的反向传播
        # 应该在此处设置一个 bug 赏金
        self.dec_reversible = dec_kwargs.pop('reversible', False)

        # 显示警告消息
        if self.dec_reversible:
            print('Warning! Due to an issue with reversible nets and encoder auxiliary losses, you must explicitly call backwards on the encoder auxiliary loss, which is supplied as the second element of the returned tuple on forward')

        self._handle = None
        self.register_kmeans_update()

    # 取消 K-means 更新
    def cancel_kmeans_update(self):
        if self._handle is None:
            return
        self._handle.remove()
        self._handle = None

    # 注册 K-means 更新
    def register_kmeans_update(self):
        self.cancel_kmeans_update()
        return update_kmeans_on_backwards(self)

    # 使用 torch.no_grad() 修饰的方法
    # 生成序列，根据输入序列和起始输出序列生成目标序列
    def generate(self, seq_in, seq_out_start, max_seq_len = None, **kwargs):
        # 如果未指定最大序列长度，则使用解码器的最大序列长度
        max_seq_len = default(max_seq_len, self.dec.max_seq_len)
        # 提取并设置编码器和解码器的关键字参数
        enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
        # 使用编码器处理输入序列，获取上下文信息
        context, _ = self.enc(seq_in, **enc_kwargs)
        # 调用解码器生成目标序列
        return self.dec.generate(seq_out_start, max_seq_len, context = context, **{**dec_kwargs, **kwargs})

    # 前向传播，处理输入序列和目标序列，计算损失
    def forward(self, seq_in, seq_out, return_loss = False, randomly_truncate_sequence = False, **kwargs):
        # 提取并设置编码器和解码器的关键字参数
        enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
        # 使用编码器处理输入序列，获取上下文信息和编码器的辅助损失
        context, enc_aux_loss = self.enc(seq_in, **enc_kwargs)
        # 调用解码器计算损失
        loss = self.dec(seq_out, return_loss = return_loss, randomly_truncate_sequence = randomly_truncate_sequence, context = context, aux_loss = enc_aux_loss, **dec_kwargs)

        # 如果解码器可逆性开启，用户必须手动调用编码器辅助损失的反向传播
        if self.dec_reversible:
            return loss, enc_aux_loss

        # 初始化辅助损失为可求导的零张量
        aux_loss = torch.tensor(0., requires_grad = True)
        # 总损失为解码器损失加上编码器辅助损失
        loss = loss + enc_aux_loss
        return loss, aux_loss
```