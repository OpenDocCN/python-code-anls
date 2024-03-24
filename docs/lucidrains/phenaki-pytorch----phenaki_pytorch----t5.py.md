# `.\lucidrains\phenaki-pytorch\phenaki_pytorch\t5.py`

```
# 导入 torch 库
import torch
# 导入 transformers 库
import transformers
# 从 transformers 库中导入 T5Tokenizer, T5EncoderModel, T5Config

# 减少警告信息，只使用编码器
transformers.logging.set_verbosity_error()

# 辅助函数
def exists(val):
    return val is not None

# 配置
MAX_LENGTH = 256
DEFAULT_T5_NAME = 'google/t5-v1_1-base'
T5_CONFIGS = {}

# 全局单例
# 获取指定名称的 tokenizer
def get_tokenizer(name):
    tokenizer = T5Tokenizer.from_pretrained(name)
    return tokenizer

# 获取指定名称的模型
def get_model(name):
    model = T5EncoderModel.from_pretrained(name)
    return model

# 获取指定名称的模型和 tokenizer
def get_model_and_tokenizer(name):
    global T5_CONFIGS

    if name not in T5_CONFIGS:
        T5_CONFIGS[name] = dict()

    if "model" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["model"] = get_model(name)

    if "tokenizer" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["tokenizer"] = get_tokenizer(name)

    return T5_CONFIGS[name]['model'], T5_CONFIGS[name]['tokenizer']

# 获取编码维度
def get_encoded_dim(name):
    if name not in T5_CONFIGS:
        config = T5Config.from_pretrained(name)
        T5_CONFIGS[name] = dict(config = config)

    elif "config" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["config"]

    elif "model" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["model"].config

    else:
        raise ValueError(f'unknown t5 name {name}')

    return config.d_model

# 编码文本
def t5_encode_text(
    texts,
    name = DEFAULT_T5_NAME,
    output_device = None
):
    # 获取模型和 tokenizer
    t5, tokenizer = get_model_and_tokenizer(name)

    # 如果 CUDA 可用，则将模型移至 CUDA
    if torch.cuda.is_available():
        t5 = t5.cuda()

    device = next(t5.parameters()).device

    # 对文本进行编码
    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors = 'pt',
        padding = 'longest',
        max_length = MAX_LENGTH,
        truncation = True
    )

    input_ids = encoded.input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)

    t5.eval()

    with torch.no_grad():
        output = t5(input_ids = input_ids, attention_mask = attn_mask)
        encoded_text = output.last_hidden_state.detach()

    attn_mask = attn_mask[..., None].bool()

    # 如果输出设备不存在，则返回编码文本
    if not exists(output_device):
        encoded_text = encoded_text.masked_fill(~attn_mask, 0.)
        return encoded_text

    encoded_text = encoded_text.to(output_device)
    attn_mask = attn_mask.to(output_device)

    encoded_text = encoded_text.masked_fill(~attn_mask, 0.)
    return encoded_text
```