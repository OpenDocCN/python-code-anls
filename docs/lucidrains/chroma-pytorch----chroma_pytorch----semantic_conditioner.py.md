# `.\lucidrains\chroma-pytorch\chroma_pytorch\semantic_conditioner.py`

```py
# 导入所需的库
import torch
import os
import logging
from transformers import AutoTokenizer, AutoModelForMaskedLM, logging
from tf_bind_transformer.cache_utils import cache_fn, run_once

# 设置日志级别为错误
logging.set_verbosity_error()

# 检查值是否存在
def exists(val):
    return val is not None

# 对字典中的值应用函数
def map_values(fn, dictionary):
    return {k: fn(v) for k, v in dictionary.items()}

# 检查是否在环境变量中设置了使用 CPU 进行上下文嵌入
CONTEXT_EMBED_USE_CPU = os.getenv('CONTEXT_EMBED_USE_CPU', None) is not None

# 如果设置了使用 CPU 进行上下文嵌入，则打印提示信息
if CONTEXT_EMBED_USE_CPU:
    print('calculating context embed only on cpu')

# 预定义模型的维度和路径
MODELS = dict(
    pubmed = dict(
        dim = 768,
        path = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
    )
)

# 全局变量，用于存储模型和分词器
GLOBAL_VARIABLES = dict(model = None, tokenizer = None)

# 获取指定模型的上下文维度
def get_contextual_dim(model_name):
    assert model_name in MODELS
    return MODELS[model_name]['dim']

# 初始化模型和分词器，只运行一次
@run_once('init_transformer')
def init_transformer(model_name):
    path = MODELS[model_name]['path']
    GLOBAL_VARIABLES['tokenizer'] = AutoTokenizer.from_pretrained(path)

    model = AutoModelForMaskedLM.from_pretrained(path)

    # 如果未设置使用 CPU 进行上下文嵌入，则将模型移至 GPU
    if not CONTEXT_EMBED_USE_CPU:
        model = model.cuda()

    GLOBAL_VARIABLES['model'] = model

# 对文本进行分词和编码
@torch.no_grad()
def tokenize_text(
    text,
    max_length = 256,
    model_name = 'pubmed',
    hidden_state_index = -1,
    return_cls_token = True
):
    init_transformer(model_name)

    model = GLOBAL_VARIABLES['model']
    tokenizer = GLOBAL_VARIABLES['tokenizer']

    encoding = tokenizer.batch_encode_plus(
        [text],
        add_special_tokens = True,
        padding = True,
        truncation = True,
        max_length = max_length,
        return_attention_mask = True,
        return_tensors = 'pt'
    )

    # 如果未设置使用 CPU 进行上下文嵌入，则将编码移至 GPU
    if not CONTEXT_EMBED_USE_CPU:
        encoding = map_values(lambda t: t.cuda(), encoding)

    model.eval()
    with torch.no_grad():
        outputs = model(**encoding, output_hidden_states = True)

    hidden_state = outputs.hidden_states[hidden_state_index][0]

    if return_cls_token:
        return hidden_state[0]

    return hidden_state.mean(dim = 0)

# 获取文本表示
def get_text_repr(
    texts,
    *,
    device,
    max_length = 256,
    model_name = 'pubmed',
    hidden_state_index = -1,
    return_cls_token = True,
):
    assert model_name in MODELS, f'{model_name} not found in available text transformers to use'

    # 如果输入为字符串，则转换为列表
    if isinstance(texts, str):
        texts = [texts]

    # 缓存文本表示函数
    get_context_repr_fn = cache_fn(tokenize_text, path = f'contexts/{model_name}')

    # 获取文本的表示
    representations = [get_context_repr_fn(text, max_length = max_length, model_name = model_name, hidden_state_index = hidden_state_index, return_cls_token = return_cls_token) for text in texts]

    return torch.stack(representations).to(device)
```