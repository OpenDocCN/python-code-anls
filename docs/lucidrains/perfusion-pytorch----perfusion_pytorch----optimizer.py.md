# `.\lucidrains\perfusion-pytorch\perfusion_pytorch\optimizer.py`

```py
# 从 torch.nn 模块中导入 Module 类
# 从 torch.optim 模块中导入 AdamW、Adam、Optimizer 类
from torch.nn import Module
from torch.optim import AdamW, Adam, Optimizer

# 从 beartype 模块中导入 beartype 装饰器
from beartype import beartype

# 从 perfusion_pytorch.embedding 模块中导入 EmbeddingWrapper 类
# 从 perfusion_pytorch.perfusion 模块中导入 Rank1EditModule 类

from perfusion_pytorch.embedding import EmbeddingWrapper
from perfusion_pytorch.perfusion import Rank1EditModule

# 定义一个函数，用于自动查找微调所需的所有参数
@beartype
def get_finetune_parameters(text_image_model: Module):
    # 初始化参数列表
    params = []
    # 遍历 text_image_model 模块中的所有子模块
    for module in text_image_model.modules():
        # 如果子模块是 EmbeddingWrapper 或 Rank1EditModule 类型
        if isinstance(module, (EmbeddingWrapper, Rank1EditModule)):
            # 将子模块的参数添加到参数列表中
            params.extend(module.parameters())

    # 返回参数列表
    return params

# 定义一个函数，用于获取微调优化器
@beartype
def get_finetune_optimizer(
    text_image_model: Module,
    lr = 1e-4,
    wd = 1e-2,
    betas = (0.9, 0.99),
    eps = 1e-8,
    **kwargs
) -> Optimizer:
    # 获取微调所需的参数
    params = get_finetune_parameters(text_image_model)

    # 断言参数列表长度大于0，否则抛出异常
    assert len(params) > 0, 'no finetuneable parameters found'
    # 计算总参数数量
    total_params = sum([p.numel() for p in params])
    # 打印优化的参数数量
    print(f'optimizing {total_params} parameters')

    # 判断是否有权重衰减
    has_weight_decay = wd > 0
    # 根据是否有权重衰减选择 AdamW 或 Adam 类
    adam_klass = AdamW if has_weight_decay else Adam
    # 初始化 Adam 的参数
    adam_kwargs = dict(lr = lr, betas = betas, eps = eps)

    # 如果有权重衰减，则更新参数字典
    if has_weight_decay:
        adam_kwargs.update(weight_decay = wd)

    # 返回根据参数和参数字典初始化的优化器
    return adam_klass(params, **adam_kwargs, **kwargs)
```