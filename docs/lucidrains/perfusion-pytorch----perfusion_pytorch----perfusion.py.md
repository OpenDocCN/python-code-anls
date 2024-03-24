# `.\lucidrains\perfusion-pytorch\perfusion_pytorch\perfusion.py`

```
# 从 math 模块中导入 ceil 函数
# 从 copy 模块中导入 deepcopy 函数
# 从 pathlib 模块中导入 Path 类
# 从 beartype 模块中导入 beartype 装饰器
# 从 beartype.typing 模块中导入 Union, List, Optional, Tuple 类型
# 从 torch 模块中导入 nn, einsum, Tensor 类
# 从 torch.nn 模块中导入 Module 类
# 从 torch.nn.functional 模块中导入 F 函数
# 从 einops 模块中导入 rearrange, reduce 函数
# 从 opt_einsum 模块中导入 contract 函数
# 从 perfusion_pytorch.open_clip 模块中导入 OpenClipAdapter 类

from math import ceil
from copy import deepcopy
from pathlib import Path

from beartype import beartype
from beartype.typing import Union, List, Optional, Tuple

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange, reduce

from opt_einsum import contract as opt_einsum

from perfusion_pytorch.open_clip import OpenClipAdapter

# 预先计算的协方差路径
# 如果论文验证通过，将为更多模型添加

CURRENT_DIR = Path(__file__).parents[0]
DATA_DIR = CURRENT_DIR / 'data'

assert DATA_DIR.is_dir()

COVARIANCE_FILENAME_BY_TEXT_IMAGE_MODEL = dict(
    SD15 = DATA_DIR / 'covariance_CLIP_ViT-L-14.pt'
)

assert all([filepath.exists() for filepath in COVARIANCE_FILENAME_BY_TEXT_IMAGE_MODEL.values()])

# 辅助函数

def exists(val):
    return val is not None

def is_all_unique(arr):
    return len(set(arr)) == len(arr)

# 用于计算 C - 输入协方差的函数

@beartype
@torch.no_grad()
def calculate_input_covariance(
    clip: OpenClipAdapter,
    texts: List[str],
    batch_size = 32,
    **cov_kwargs
):
    num_batches = ceil(len(texts) / batch_size)

    all_embeds = []

    length = len(texts)

    for batch_ind in range(num_batches):
        start_index = batch_ind * batch_size
        batch_texts = texts[start_index:(start_index + batch_size)]

        embeds, mask = clip.embed_texts(batch_texts)
        all_embeds.append(embeds[mask])

    all_embeds = torch.cat(all_embeds, dim = 0)

    return einsum('n d, n e -> d e', all_embeds, all_embeds) / length

# 由掩码加权的损失函数

@beartype
def loss_fn_weighted_by_mask(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    normalized_mask_min_value = 0.
):
    assert mask.shape[-2:] == pred.shape[-2:] == target.shape[-2:]
    assert mask.shape[0] == pred.shape[0] == target.shape[0]

    assert (mask.amin() >= 0.).all(), 'mask should not have values below 0'

    if mask.ndim == 4:
        assert mask.shape[1] == 1
        mask = rearrange(mask, 'b 1 h w -> b h w')

    loss = F.mse_loss(pred, target, reduction = 'none')
    loss = reduce(loss, 'b c h w -> b h w')

    # 通过最大值对掩码进行归一化

    normalized_mask = mask / mask.amax(dim = -1, keepdim = True).clamp(min = 1e-5)
    normalized_mask = normalized_mask.clamp(min = normalized_mask_min_value)

    loss = loss * normalized_mask

    return loss.mean()

# 一个模块，包装了交叉注意力的键和值投影到文本编码

class Rank1EditModule(Module):

    @beartype
    def __init__(
        self,
        key_or_values_proj: nn.Linear,
        *,
        num_concepts: int = 1,
        C: Optional[Tensor] = None,          # 输入的协方差，从 100K laion 文本中预先计算
        default_model = 'SD15',
        text_seq_len: int = 77,
        is_key_proj: bool = False,
        input_decay = 0.99,
        train_beta = 0.75,
        train_temperature = 0.1,
        eval_beta = 0.70,                    # 在论文中，指定了本地键锁定的范围 (0.6 - 0.75)，全局键锁定的范围 (0.4 -0.6)
        eval_temperature = 0.15,
        frac_gradient_concept_embed = 0.1,   # 他们使用一个较慢的学习率来嵌入 - 这可以通过一个技巧来减少反向传播的梯度
        multi_concepts_use_cholesky = False  # 对于多个概念，使用一种不需要 Cholesky 根的近似技术
        ):
        # 调用父类的构造函数
        super().__init__()
        # 断言在注意力中的键值投影不应该有偏置
        assert not exists(key_or_values_proj.bias), 'key value projection in attention should not have bias'

        # 初始化注意力模块的参数
        self.num_concepts = num_concepts
        self.multi_concepts_use_cholesky = multi_concepts_use_cholesky

        # 获取键值投影的权重
        self.weight = key_or_values_proj.weight
        dim_output, dim_input = self.weight.shape

        # 设置训练和评估时的温度和 beta 参数
        self.train_beta = train_beta
        self.train_temperature = train_temperature
        self.eval_beta = eval_beta
        self.eval_temperature = eval_temperature

        # 输入的衰减参数
        self.input_decay = input_decay

        # 文本序列的长度
        self.text_seq_len = text_seq_len

        # 降低概念嵌入学习率的参数
        assert 0 < frac_gradient_concept_embed <= 1.
        self.frac_gradient_concept_embed = frac_gradient_concept_embed

        # 初始化概念文本嵌入的指数平滑参数
        self.register_buffer('initted', torch.zeros(num_concepts, 1).bool())
        self.register_buffer('ema_concept_text_encs', torch.zeros(num_concepts, dim_input))

        # 概念输出 - 仅优化值，而不是键
        self.is_key_proj = is_key_proj # 锁定输出到超类，并关闭梯度

        self.concept_outputs = nn.Parameter(torch.zeros(num_concepts, dim_output), requires_grad = not is_key_proj)

        # 输入协方差 C 的逆矩阵，如果未传入协方差，则使用默认值
        if not exists(C):
            covariance_filepath = COVARIANCE_FILENAME_BY_TEXT_IMAGE_MODEL.get(default_model, None)

            assert exists(covariance_filepath), f'{default_model} not found in the list of precomputed covariances {tuple(COVARIANCE_FILENAME_BY_TEXT_IMAGE_MODEL.keys())}'

            C = torch.load(str(covariance_filepath))
            print(f'precomputed covariance loaded from {str(covariance_filepath)}')

        # 计算 C_inv
        C_inv = torch.inverse(C)
        self.register_buffer('C_inv', C_inv)

    @property
    def num_concepts(self):
        return self._num_concepts

    @num_concepts.setter
    def num_concepts(self, value):
        self._num_concepts = value

        if value == 1 or not self.multi_concepts_use_cholesky:
            return

        # 对于多个概念，需要 cholesky 分解 L_t_inv
        try:
            L = torch.linalg.cholesky(self.C_inv)
        except:
            print('unable to perform cholesky. please make sure input covariance matrix is properly calculated')
            exit()

        L_T = L.T
        L_T_inv = torch.inverse(L_T)

        self.register_buffer('L_T', L_T, persistent = False)
        self.register_buffer('L_T_inv', L_T_inv, persistent = False)

    @property
    def device(self):
        return next(self.buffers()).device

    # 返回参数
    def parameters(self):
        if not self.is_key_proj:
            return []

        return [self.concept_outputs]

    @beartype
    def forward(
        self,
        text_enc: Tensor,
        *,
        concept_indices: Optional[Tensor] = None,
        text_enc_with_superclass: Optional[Tensor] = None,
        concept_id: Union[int, Tuple[int, ...]] = 0
# 合并已训练的 Rank1EditModule(s) 的函数

@beartype
def merge_rank1_edit_modules(
    *modules: Rank1EditModule,  # 接受多个 Rank1EditModule 参数
    use_cholesky = False  # 是否使用 Cholesky 分解，默认为 False
) -> Rank1EditModule:  # 返回合并后的 Rank1EditModule 对象

    # 断言所有模块都已初始化并最好已训练
    assert all([m.initted.all() for m in modules]), 'all modules must be initialized and ideally trained'
    # 断言概念输出维度必须相同
    assert len(set([m.concept_outputs.shape[-1] for m in modules])) == 1, 'concept output dimension must be the same'
    # 断言所有模块必须为键或值。不能将键和值的 Rank1EditModule 合并在一起
    assert len(set([m.is_key_proj for m in modules])) == 1, 'all modules must be either for keys, or values. you cannot merge rank 1 edit modules of keys and values together'

    # 获取第一个模块
    first_module = modules[0]
    # 深拷贝第一个模块
    merged_module = deepcopy(first_module)
    # 设置是否使用 Cholesky 分解
    merged_module.multi_concepts_use_cholesky = use_cholesky

    # 计算总概念数
    total_concepts = sum([m.num_concepts for m in modules])
    merged_module.num_concepts = total_concepts

    # 拼接所有模块的概念输出
    concept_outputs = torch.cat(tuple(m.concept_outputs.data for m in modules), dim = 0)
    merged_module.concept_outputs = nn.Parameter(concept_outputs, requires_grad = not first_module.is_key_proj)

    # 拼接所有模块的 EMA 概念文本编码
    ema_concept_text_encs = torch.cat(tuple(m.ema_concept_text_encs.data for m in modules), dim = 0)
    merged_module.register_buffer('ema_concept_text_encs',  ema_concept_text_encs)

    # 注册初始化状态
    merged_module.register_buffer('initted', torch.ones(total_concepts, 1).bool())

    # 返回合并后的模块
    return merged_module

# 用于连接交叉注意力的函数

@beartype
def make_key_value_proj_rank1_edit_modules_(
    cross_attention: nn.Module,  # 交叉注意力模块
    *,
    input_covariance: Tensor,  # 输入协方差
    key_proj_name: str,  # 键投影名称
    value_proj_name: str,  # 值投影名称
    **rank1_edit_module_kwargs  # Rank1EditModule 的其他参数
):
    # 获取键投影和值投影线性层
    linear_key = getattr(cross_attention, key_proj_name, None)
    linear_values = getattr(cross_attention, value_proj_name, None)

    # 断言键投影和值投影必须是 nn.Linear 类型
    assert isinstance(linear_key, nn.Linear), f'{key_proj_name} must point to where the keys projection is (ex. self.to_keys = nn.Linear(in, out, bias = False) -> key_proj_name = "to_keys")'
    assert isinstance(linear_values, nn.Linear), f'{value_proj_name} must point to where the values projection is (ex. self.to_values = nn.Linear(in, out, bias = False) -> value_proj_name = "to_values")'

    # 创建键和值的 Rank1EditModule
    rank1_edit_module_keys = Rank1EditModule(linear_key, input_covariance = input_covariance, is_key_proj = True, **rank1_edit_module_kwargs)
    rank1_edit_module_values = Rank1EditModule(linear_values, input_covariance = input_covariance, is_key_proj = False, **rank1_edit_module_kwargs)

    # 将 Rank1EditModule 设置为键投影和值投影
    setattr(cross_attention, key_proj_name, rank1_edit_module_keys)
    setattr(cross_attention, value_proj_name, rank1_edit_module_values)
```