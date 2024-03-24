# `.\lucidrains\video-diffusion-pytorch\video_diffusion_pytorch\text.py`

```py
# 导入 torch 库
import torch
# 从 einops 库中导入 rearrange 函数
from einops import rearrange

# 检查变量是否存在的函数
def exists(val):
    return val is not None

# 全局单例变量

# 模型和分词器的初始化为 None
MODEL = None
TOKENIZER = None
# BERT 模型的维度为 768
BERT_MODEL_DIM = 768

# 获取分词器函数
def get_tokenizer():
    global TOKENIZER
    # 如果 TOKENIZER 不存在，则加载 'bert-base-cased' 模型的分词器
    if not exists(TOKENIZER):
        TOKENIZER = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
    return TOKENIZER

# 获取 BERT 模型函数
def get_bert():
    global MODEL
    # 如果 MODEL 不存在，则加载 'bert-base-cased' 模型
    if not exists(MODEL):
        MODEL = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
        # 如果有 GPU 可用，则将模型移动到 GPU
        if torch.cuda.is_available():
            MODEL = MODEL.cuda()

    return MODEL

# 分词函数

def tokenize(texts, add_special_tokens = True):
    # 如果 texts 不是列表或元组，则转换为列表
    if not isinstance(texts, (list, tuple)):
        texts = [texts]

    # 获取分词器
    tokenizer = get_tokenizer()

    # 对文本进行编码
    encoding = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens = add_special_tokens,
        padding = True,
        return_tensors = 'pt'
    )

    # 获取 token_ids
    token_ids = encoding.input_ids
    return token_ids

# 嵌入函数

@torch.no_grad()
def bert_embed(
    token_ids,
    return_cls_repr = False,
    eps = 1e-8,
    pad_id = 0.
):
    # 获取 BERT 模型
    model = get_bert()
    # 创建 mask，标记不为 pad_id 的位置
    mask = token_ids != pad_id

    # 如果有 GPU 可用，则将 token_ids 和 mask 移动到 GPU
    if torch.cuda.is_available():
        token_ids = token_ids.cuda()
        mask = mask.cuda()

    # 使用 BERT 模型进行前向传播
    outputs = model(
        input_ids = token_ids,
        attention_mask = mask,
        output_hidden_states = True
    )

    # 获取最后一层的隐藏状态
    hidden_state = outputs.hidden_states[-1]

    # 如果需要返回 [cls] 的表示
    if return_cls_repr:
        return hidden_state[:, 0]               # 返回 [cls] 作为表示

    # 如果 mask 不存在，则返回所有 token 的平均值
    if not exists(mask):
        return hidden_state.mean(dim = 1)

    # 重新定义 mask，排除 [cls]，考虑长度
    mask = mask[:, 1:]                          # 平均所有 token，排除 [cls]
    mask = rearrange(mask, 'b n -> b n 1')

    # 计算加权平均值
    numer = (hidden_state[:, 1:] * mask).sum(dim = 1)
    denom = mask.sum(dim = 1)
    masked_mean =  numer / (denom + eps)
    return masked_mean
```