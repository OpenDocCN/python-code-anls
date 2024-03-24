# `.\lucidrains\tf-bind-transformer\tf_bind_transformer\protein_utils.py`

```py
# 导入所需的库
import torch
import os
import re
from pathlib import Path
from functools import partial
import esm
from torch.nn.utils.rnn import pad_sequence
from transformers import AlbertTokenizer, AutoModelForMaskedLM, logging
from tf_bind_transformer.cache_utils import cache_fn, run_once, md5_hash_fn

# 定义函数，判断值是否存在
def exists(val):
    return val is not None

# 定义函数，对字典中的值应用给定函数
def map_values(fn, dictionary):
    return {k: fn(v) for k, v in dictionary.items()}

# 定义函数，将张量移动到指定设备
def to_device(t, *, device):
    return t.to(device)

# 定义函数，将输入转换为元组
def cast_tuple(t):
    return (t,) if not isinstance(t, tuple) else t

# 检查是否设置了环境变量 PROTEIN_EMBED_USE_CPU
PROTEIN_EMBED_USE_CPU = os.getenv('PROTEIN_EMBED_USE_CPU', None) is not None

# 如果设置了 PROTEIN_EMBED_USE_CPU，则打印提示信息
if PROTEIN_EMBED_USE_CPU:
    print('calculating protein embed only on cpu')

# 全局变量
GLOBAL_VARIABLES = {
    'model': None,
    'tokenizer': None
}

# 计算蛋白质表示与亚单位
def calc_protein_representations_with_subunits(proteins, get_repr_fn, *, device):
    representations = []

    for subunits in proteins:
        subunits = cast_tuple(subunits)
        subunits_representations = list(map(get_repr_fn, subunits))
        subunits_representations = list(map(partial(to_device, device=device), subunits_representations))
        subunits_representations = torch.cat(subunits_representations, dim=0)
        representations.append(subunits_representations)

    lengths = [seq_repr.shape[0] for seq_repr in representations]
    masks = torch.arange(max(lengths), device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]
    padded_representations = pad_sequence(representations, batch_first=True)

    return padded_representations.to(device), masks.to(device)

# ESM 相关函数
ESM_MAX_LENGTH = 1024
ESM_EMBED_DIM = 1280

# 映射整数到氨基酸字符串的字典
INT_TO_AA_STR_MAP = {
    0: 'A',
    1: 'C',
    2: 'D',
    3: 'E',
    4: 'F',
    5: 'G',
    6: 'H',
    7: 'I',
    8: 'K',
    9: 'L',
    10: 'M',
    11: 'N',
    12: 'P',
    13: 'Q',
    14: 'R',
    15: 'S',
    16: 'T',
    17: 'V',
    18: 'W',
    19: 'Y',
    20: '_'
}

# 将张量转换为氨基酸字符串
def tensor_to_aa_str(t):
    str_seqs = []
    for int_seq in t.unbind(dim=0):
        str_seq = list(map(lambda t: INT_TO_AA_STR_MAP[t] if t != 20 else '', int_seq.tolist()))
        str_seqs.append(''.join(str_seq))
    return str_seqs

# 初始化 ESM 模型
@run_once('init_esm')
def init_esm():
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()

    if not PROTEIN_EMBED_USE_CPU:
        model = model.cuda()

    GLOBAL_VARIABLES['model'] = (model, batch_converter)

# 获取单个蛋白质的 ESM 表示
def get_single_esm_repr(protein_str):
    init_esm()
    model, batch_converter = GLOBAL_VARIABLES['model']

    data = [('protein', protein_str)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    if batch_tokens.shape[1] > ESM_MAX_LENGTH:
        print(f'warning max length protein esm: {protein_str}')

    batch_tokens = batch_tokens[:, :ESM_MAX_LENGTH]

    if not PROTEIN_EMBED_USE_CPU:
        batch_tokens = batch_tokens.cuda()

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])

    token_representations = results['representations'][33]
    representation = token_representations[0][1: len(protein_str) + 1]
    return representation

# 获取多个蛋白质的 ESM 表示
def get_esm_repr(proteins, device):
    if isinstance(proteins, torch.Tensor):
        proteins = tensor_to_aa_str(proteins)

    get_protein_repr_fn = cache_fn(get_single_esm_repr, path='esm/proteins')

    return calc_protein_representations_with_subunits(proteins, get_protein_repr_fn, device=device)

# PROT-ALBERT 2048 上下文长度
PROT_ALBERT_PATH = 'Rostlab/prot_albert'
PROT_ALBERT_DIM = 4096
PROT_ALBERT_MAX_LENGTH = 2048

# 将蛋白质字符串中的特殊字符替换为空格
def protein_str_with_spaces(protein_str):
    protein_str = re.sub(r"[UZOB]", 'X', protein_str)
    return ' '.join([*protein_str])

# 初始化 PROT-ALBERT 模型
@run_once('init_prot_albert')
def init_prot_albert():
    GLOBAL_VARIABLES['tokenizer'] = AlbertTokenizer.from_pretrained(PROT_ALBERT_PATH, do_lower_case=False)
    # 从预训练的 ALBERT 模型中加载用于 Masked Language Modeling 的模型
    model = AutoModelForMaskedLM.from_pretrained(PROT_ALBERT_PATH)
    
    # 如果不使用 CPU 运行蛋白质嵌入模型，则将模型移动到 GPU 上
    if not PROTEIN_EMBED_USE_CPU:
        model = model.cuda()
    
    # 将加载的模型存储在全局变量中
    GLOBAL_VARIABLES['model'] = model
# 获取单个蛋白质的 ALBERT 表示
def get_single_prot_albert_repr(
    protein_str,
    max_length = PROT_ALBERT_MAX_LENGTH,
    hidden_state_index = -1
):
    # 初始化 ALBERT 模型
    init_prot_albert()
    # 获取全局变量中的模型和分词器
    model = GLOBAL_VARIABLES['model']
    tokenizer = GLOBAL_VARIABLES['tokenizer']

    # 对蛋白质字符串进行编码
    encoding = tokenizer.batch_encode_plus(
        [protein_str_with_spaces(protein_str)],
        add_special_tokens = True,
        padding = True,
        truncation = True,
        max_length = max_length,
        return_attention_mask = True,
        return_tensors = 'pt'
    )

    # 如果不使用 CPU 进行蛋白质嵌入
    if not PROTEIN_EMBED_USE_CPU:
        encoding = map_values(lambda t: t.cuda(), encoding)

    # 将模型设置为评估模式
    model.eval()
    # 禁用梯度计算
    with torch.no_grad():
        # 获取模型输出
        outputs = model(**encoding, output_hidden_states = True)

    # 获取隐藏状态
    hidden_state = outputs.hidden_states[hidden_state_index][0]
    return hidden_state

# 获取蛋白质 ALBERT 表示
def get_prot_albert_repr(
    proteins,
    device,
    max_length = PROT_ALBERT_MAX_LENGTH,
    hidden_state_index = -1
):
    # 如果输入为字符串，则转换为列表
    if isinstance(proteins, str):
        proteins = [proteins]

    # 如果输入为张量，则转换为氨基酸字符串
    if isinstance(proteins, torch.Tensor):
        proteins = tensor_to_aa_str(proteins)

    # 缓存单个蛋白质 ALBERT 表示的函数
    get_protein_repr_fn = cache_fn(get_single_prot_albert_repr, path = f'proteins/prot_albert')

    # 计算蛋白质表示
    return calc_protein_representations_with_subunits(proteins, get_protein_repr_fn, device = device)

# alphafold2 函数

# 定义最大长度和嵌入维度
AF2_MAX_LENGTH = 2500
AF2_EMBEDDING_DIM = 384

# 设置 AF2_DIRECTORY 路径
AF2_DIRECTORY = os.getenv('TF_BIND_AF2_DIRECTORY', os.path.expanduser('~/.cache.tf.bind.transformer/.af2_embeddings'))
AF2_DIRECTORY_PATH = Path(AF2_DIRECTORY)

# 获取单个 alphafold2 表示
def get_single_alphafold2_repr(
    protein_str,
    max_length = AF2_MAX_LENGTH,
):
    # 计算蛋白质字符串的 MD5 哈希值
    md5 = md5_hash_fn(protein_str)
    embedding_path = AF2_DIRECTORY_PATH / f'{md5}.pt'
    assert embedding_path.exists(), f'af2 embedding not found for {protein_str}'

    # 加载嵌入张量
    tensor = torch.load(str(embedding_path))
    return tensor[:max_length]

# 获取 alphafold2 表示
def get_alphafold2_repr(
    proteins,
    device,
    max_length = AF2_MAX_LENGTH,
    **kwargs
):
    representations = []

    for subunits in proteins:
        subunits = cast_tuple(subunits)
        subunits = list(map(lambda t: get_single_alphafold2_repr(t, max_length = max_length), subunits))
        subunits = torch.cat(subunits, dim = 0)
        representations.append(subunits)

    lengths = [seq_repr.shape[0] for seq_repr in representations]
    masks = torch.arange(max(lengths), device = device)[None, :] <  torch.tensor(lengths, device = device)[:, None]
    padded_representations = pad_sequence(representations, batch_first = True)

    return padded_representations.to(device), masks.to(device)

# 工厂函数

# 定义蛋白质表示配置
PROTEIN_REPR_CONFIG = {
    'esm': {
        'dim': ESM_EMBED_DIM,
        'fn': get_esm_repr
    },
    'protalbert': {
        'dim': PROT_ALBERT_DIM,
        'fn': get_prot_albert_repr
    },
    'alphafold2': {
        'dim': AF2_EMBEDDING_DIM,
        'fn': get_alphafold2_repr
    }
}

# 获取蛋白质嵌入器
def get_protein_embedder(name):
    allowed_protein_embedders = list(PROTEIN_REPR_CONFIG.keys())
    assert name in allowed_protein_embedders, f"must be one of {', '.join(allowed_protein_embedders)}"

    config = PROTEIN_REPR_CONFIG[name]
    return config
```