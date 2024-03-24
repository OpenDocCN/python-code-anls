# `.\lucidrains\muse-maskgit-pytorch\muse_maskgit_pytorch\t5.py`

```py
# 导入日志、torch和transformers模块
import logging
import torch
import transformers
from transformers import T5Tokenizer, T5EncoderModel, T5Config

# 设置transformers日志级别为error
transformers.logging.set_verbosity_error()

# 检查值是否存在的函数
def exists(val):
    return val is not None

# 配置参数
MAX_LENGTH = 256
DEFAULT_T5_NAME = 'google/t5-v1_1-base'
T5_CONFIGS = {}

# 全局单例变量

# 获取指定模型的tokenizer
def get_tokenizer(name):
    tokenizer = T5Tokenizer.from_pretrained(name)
    return tokenizer

# 获取指定模型的encoder模型
def get_model(name):
    model = T5EncoderModel.from_pretrained(name)
    return model

# 获取指定模型的encoder模型和tokenizer
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
        # 避免加载模型，仅获取维度
        config = T5Config.from_pretrained(name)
        T5_CONFIGS[name] = dict(config=config)
    elif "config" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["config"]
    elif "model" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["model"].config
    else:
        assert False
    return config.d_model

# 编码文本

# 使用beartype装饰器，指定texts参数为字符串或字符串列表
@beartype
def t5_encode_text(
    texts: Union[str, List[str]],
    name = DEFAULT_T5_NAME,
    output_device = None
):
    if isinstance(texts, str):
        texts = [texts]

    # 获取指定模型的encoder模型和tokenizer
    t5, tokenizer = get_model_and_tokenizer(name)

    # 如果CUDA可用，则将模型移至CUDA
    if torch.cuda.is_available():
        t5 = t5.cuda()

    device = next(t5.parameters()).device

    # 对文本进行编码
    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors = "pt",
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

    attn_mask = attn_mask.bool()
    encoded_text = encoded_text.masked_fill(~attn_mask[..., None], 0.)

    # 如果output_device存在，则将编码后的文本移至指定设备
    if not exists(output_device):
        return encoded_text

    encoded_text.to(output_device)
    return encoded_text
```