# `.\lucidrains\perfusion-pytorch\perfusion_pytorch\embedding.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn, Tensor
from torch import nn, Tensor
# 从 torch.nn 库中导入 Module
from torch.nn import Module

# 从 collections 库中导入 namedtuple
from collections import namedtuple

# 从 beartype 库中导入 beartype
from beartype import beartype
# 从 beartype.door 库中导入 is_bearable
from beartype.door import is_bearable
# 从 beartype.typing 库中导入 Optional, Tuple, Union, Callable, List
from beartype.typing import Optional, Tuple, Union, Callable, List

# 从 einops 库中导入 rearrange
from einops import rearrange

# 从 open_clip 库中导入 tokenizer
from open_clip import tokenizer

# 定义常量 EmbeddingReturn 为一个命名元组，包含 'embed_with_concept', 'embed_with_superclass', 'embed_mask', 'concept_indices' 四个字段
EmbeddingReturn = namedtuple('EmbeddingReturn', [
    'embed_with_concept',
    'embed_with_superclass',
    'embed_mask',
    'concept_indices'
])

# 定义辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    return val if exists(val) else d

# 判断列表中元素是否全部唯一
def is_all_unique(arr):
    return len(set(arr)) == len(arr)

# 根据给定的索引过滤元组中的元素
def filter_tuple_indices(tup, indices):
    return tuple(tup[i] for i in indices)

# 根据给定的 ids 创建一个 mask
@beartype
def get_mask(
    x: Tensor,
    ids: Tuple[int, ...]
):
    masks = tuple(x == i for i in ids)
    mask, *rest_masks = masks

    for rest_mask in rest_masks:
        mask = mask | rest_mask

    return mask

# 嵌入包装类

class EmbeddingWrapper(Module):

    # 初始化函数
    @beartype
    def __init__(
        self,
        embed: nn.Embedding,
        num_concepts = 1,
        superclass_embed_id: Optional[Union[int, Tuple[int, ...]]] = None,
        superclass_string: Optional[str] = None,
        tokenize: Callable[[List[str]], Tensor] = tokenizer.tokenize,
        tokenizer_pad_id: int = 0,
        tokenizer_sos_eos_id: Tuple[int, int] = (49406, 49407)
    ):
        super().__init__()
        self.embed = embed
        num_embeds, dim = embed.weight.shape

        self.num_embeds = num_embeds
        self.num_concepts = num_concepts
        self.concepts = nn.Parameter(torch.zeros(num_concepts, dim))

        assert not (exists(superclass_embed_id) and exists(superclass_string)), 'either superclass embed id is given, or the superclass string'

        self.pad_id = tokenizer_pad_id
        self.tokenize = None

        if exists(superclass_string):
            self.tokenize = tokenize

            ids = tokenize([superclass_string])[0]

            mask_for_ids = get_mask(ids, (tokenizer_pad_id, *tokenizer_sos_eos_id))
            ids = ids[~mask_for_ids]

            assert ids.shape[-1] == 1, f'your superclass concept string must map exactly one token id'
            superclass_embed_id = ids[0].item()

            print(f'super class embed for "{superclass_string}"" set as {superclass_embed_id}')
            print(f'you can now pass in a list of strings containing superclass concept, and this wrapper will return the embedding w/ concept and superclass required for finetuning')

        self.superclass_embed_id = superclass_embed_id

        assert not (exists(superclass_embed_id) and num_concepts > 1), 'cannot do multi concept with superclass embed id given'

        if exists(superclass_embed_id):
            # 作者发现将概念嵌入初始化为超类嵌入会获得更好的结果，允许这种选项

            if not isinstance(superclass_embed_id, tuple):
                superclass_embed_id = (superclass_embed_id,)

            superclass_embed_indices = torch.tensor(list(superclass_embed_id))
            superclass_embeds = embed(superclass_embed_indices)
            self.concepts.data.copy_(superclass_embeds)
        else:
            # 否则初始化为通常用于嵌入的小初始化值

            nn.init.normal_(self.concepts, std = 0.02)

        self.concept_embed_ids = tuple(range(num_embeds, num_embeds + num_concepts))

    # 返回参数
    def parameters(self):
        return [self.concepts]

    # 返回设备
    @property
    def device(self):
        return self.concepts.device

    # 前向传播函数
    @beartype
    def forward(
        self,
        x: Union[Tensor, List[str]],
        concept_id: Optional[Union[int, Tuple[int, ...]]] = None,
        return_embed_with_superclass = True,
        clip_transformer_fn: Optional[Callable[[Tensor], Tensor]] = None
# 一个用于 CLIP 的包装器
# 自动将令牌嵌入与新概念包装在一起
# 定义一个类 OpenClipEmbedWrapper，用于包装 CLIP 模型的嵌入层，并在前向传播中通过文本转换器和最终层归一化层传递概念嵌入和超类概念嵌入
# 同时，将 ids 和 superclass_ids 通过修改后的文本编码器传递两次（将尝试用 nn.Identity 替换 nn.Embedding）

class OpenClipEmbedWrapper(Module):
    @beartype
    def __init__(
        self,
        clip: Module,
        text_transformer_path = 'transformer',
        ln_final_path = 'ln_final',  # 在 CLIP 中，最终的层归一化层与转换器分开
        **embedding_wrapper_kwargs
    ):
        super().__init__()
        # 创建一个嵌入层包装器，用于包装 CLIP 模型的 token 嵌入
        self.wrapped_embed = EmbeddingWrapper(clip.token_embedding, **embedding_wrapper_kwargs)

        # 获取 CLIP 模型中各模块的路径和模块对象的字典
        path_to_modules = dict([(path, mod) for path, mod in clip.named_modules()])

        # 确保文本转换器路径在路径字典中
        assert text_transformer_path in path_to_modules

        # 获取文本转换器和最终层归一化层（如果存在）
        text_transformer = path_to_modules[text_transformer_path]
        ln_final = path_to_modules.get(ln_final_path, nn.Identity())

        # 将文本转换器和最终层归一化层组合成一个序列
        self.text_transformer = nn.Sequential(
            text_transformer,
            ln_final
        )

    # 前向传播函数，接收输入 x 和其他关键字参数，返回嵌入层包装器
    def forward(
        self,
        x,
        **kwargs
    ) -> EmbeddingWrapper:
        # 通过嵌入层包装器获取文本嵌入、超类文本嵌入、文本掩码和概念索引
        text_embeds, superclass_text_embeds, text_mask, concept_indices = self.wrapped_embed(x, **kwargs)

        # 将文本嵌入传递给文本转换器
        text_enc = self.text_transformer(text_embeds)

        superclass_text_enc = None

        # 如果超类文本嵌入存在，则将其传递给文本转换器
        if exists(superclass_text_embeds):
            superclass_text_enc = self.text_transformer(superclass_text_embeds)

        # 返回嵌入返回对象，包括文本嵌入、超类文本嵌入、文本掩码和概念索引
        return EmbeddingReturn(text_enc, superclass_text_enc, text_mask, concept_indices)

# 将多个嵌入层包装器（每个具有一个概念）合并为一个具有多个概念的合并嵌入层包装器

@beartype
def merge_embedding_wrappers(
    *embeds: EmbeddingWrapper
) -> EmbeddingWrapper:

    # 计算总概念数
    total_concepts = sum([embed.num_concepts for embed in embeds])

    # 确保所有嵌入层的权重形状相同
    assert len(set([tuple(embed.embed.weight.shape) for embed in embeds])) == 1

    # 获取第一个嵌入层的嵌入
    embed = embeds[0].embed

    # 创建一个合并的嵌入层包装器，包括总概念数
    merged_concepts = EmbeddingWrapper(
        embed = embed,
        num_concepts = total_concepts
    )

    # 将合并的嵌入层包装器设置为评估模式
    merged_concepts.eval()

    # 将所有嵌入层的概念连接起来
    concepts = torch.cat(tuple(embed.concepts.data for embed in embeds), dim = 0)

    # 将连接后的概念设置为合并的嵌入层包装器的概念
    merged_concepts.concepts = nn.Parameter(concepts)

    # 返回合并的嵌入层包装器
    return merged_concepts
```