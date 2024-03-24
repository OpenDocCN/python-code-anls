# `.\lucidrains\perfusion-pytorch\perfusion_pytorch\save_load.py`

```py
# 导入所需的模块
from pathlib import Path
import torch
from torch import nn
from torch.nn import Module
from beartype import beartype
from perfusion_pytorch.embedding import EmbeddingWrapper
from perfusion_pytorch.perfusion import Rank1EditModule

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 保存和加载必要的额外微调参数

# 保存函数，将模型的参数保存到指定路径
@beartype
def save(
    text_image_model: Module,
    path: str
):
    # 将路径转换为 Path 对象
    path = Path(path)
    # 创建路径的父目录，如果不存在则创建
    path.parents[0].mkdir(exist_ok=True, parents=True)

    embed_params = None
    key_value_params = []
    C_inv = None

    # 遍历模型的所有模块
    for module in text_image_model.modules():
        # 如果模块是 EmbeddingWrapper 类型
        if isinstance(module, EmbeddingWrapper):
            # 确保只有一个包装的 EmbeddingWrapper
            assert not exists(embed_params), 'there should only be one wrapped EmbeddingWrapper'
            embed_params = module.concepts.data

        # 如果模块是 Rank1EditModule 类型
        elif isinstance(module, Rank1EditModule):
            # 将模块的参数添加到列表中
            key_value_params.append([
                module.ema_concept_text_encs.data,
                module.concept_outputs.data
            ])

            C_inv = module.C_inv.data

    # 确保 C_inv 参数存在
    assert exists(C_inv), 'Rank1EditModule not found. you likely did not wire up the text to image model correctly'

    # 将参数打包成字典
    pkg = dict(
        embed_params=embed_params,
        key_value_params=key_value_params,
        C_inv=C_inv
    )

    # 保存参数到指定路径
    torch.save(pkg, f'{str(path)}')
    print(f'saved to {str(path)}')

# 加载函数，从指定路径加载参数到模型
@beartype
def load(
    text_image_model: Module,
    path: str
):
    # 将路径转换为 Path 对象
    path = Path(path)
    # 检查文件是否存在
    assert path.exists(), f'file not found at {str(path)}'

    # 加载保存的参数
    pkg = torch.load(str(path))

    embed_params = pkg['embed_params']
    key_value_params = pkg['key_value_params']
    C_inv = pkg['C_inv']

    # 遍历模型的所有模块
    for module in text_image_model.modules():
        # 如果模块是 EmbeddingWrapper 类型
        if isinstance(module, EmbeddingWrapper):
            # 将加载的参数复制到模块中
            module.concepts.data.copy_(embed_params)

        # 如果模块是 Rank1EditModule 类型
        elif isinstance(module, Rank1EditModule):
            # 确保保存的参数和加载的参数匹配
            assert len(key_value_params) > 0, 'mismatch between what was saved vs what is being loaded'
            concept_input, concept_output = key_value_params.pop(0)
            module.ema_concept_text_encs.data.copy_(concept_input)
            module.concept_outputs.data.copy_(concept_output)

            module.C_inv.copy_(C_inv)
            module.initted.copy_(torch.tensor([True]))

    print(f'loaded concept params from {str(path)}')
```