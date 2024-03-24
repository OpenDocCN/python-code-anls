# `.\lucidrains\reformer-pytorch\reformer_pytorch\reformer_enc_dec.py`

```py
# 导入 re 模块，用于正则表达式操作
import re
# 从 torch 模块中导入 nn 类
from torch import nn
# 从 reformer_pytorch 模块中导入 ReformerLM 类
from reformer_pytorch.reformer_pytorch import ReformerLM
# 从 reformer_pytorch 模块中导入 TrainingWrapper 类
from reformer_pytorch.generative_tools import TrainingWrapper

# 定义编码器前缀
ENC_PREFIX = 'enc_'
# 定义解码器前缀
DEC_PREFIX = 'dec_'

# 根据条件将字典按键分组
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

# 根据键前缀将字典分组
def group_by_key_prefix(prefix, d):
    return group_dict_by_key(lambda x: string_begins_with(prefix, x), d)

# 根据键前缀并移除前缀将字典分组
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

# 定义 ReformerEncDec 类，继承自 nn.Module 类
class ReformerEncDec(nn.Module):
    def __init__(self, dim, ignore_index = 0, pad_value = 0, **kwargs):
        super().__init__()
        enc_kwargs, dec_kwargs, _ = extract_enc_dec_kwargs(kwargs)
        
        # 断言不能手动设置返回嵌入标志
        assert 'return_embedding' not in enc_kwargs, 'you cannot manually set the return embeddings flag for the encoder'
        # 断言必须为编码器和解码器设置维度
        assert 'dim' not in dec_kwargs and 'dim' not in enc_kwargs, 'you must set the dim for both encoder and decoder'

        # 设置编码器和解码器的维度
        enc_kwargs['dim'] = dec_kwargs['dim'] = dim
        enc_kwargs['return_embeddings'] = True
        dec_kwargs['causal'] = True

        # 设置编码器和解码器的 bucket_size
        enc_kwargs.setdefault('bucket_size', 64)
        dec_kwargs.setdefault('bucket_size', enc_kwargs['bucket_size'] * 2)

        # 创建 ReformerLM 编码器和解码器对象
        enc = ReformerLM(**enc_kwargs)
        dec = ReformerLM(**dec_kwargs)

        # 使用 TrainingWrapper 封装编码器和解码器对象
        self.enc = TrainingWrapper(enc, ignore_index = ignore_index, pad_value = pad_value)
        self.dec = TrainingWrapper(dec, ignore_index = ignore_index, pad_value = pad_value)

    # 生成序列
    def generate(self, seq_in, seq_out_start, seq_len, **kwargs):
        enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
        enc_keys = self.enc(seq_in, **enc_kwargs)
        return self.dec.generate(seq_out_start, seq_len, keys = enc_keys, **{**dec_kwargs, **kwargs})

    # 前向传播
    def forward(self, seq_in, seq_out, return_loss = False, **kwargs):
        enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
        enc_keys = self.enc(seq_in, **enc_kwargs)
        return self.dec(seq_out, return_loss = return_loss, keys = enc_keys, **dec_kwargs)
```