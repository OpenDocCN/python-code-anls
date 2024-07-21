# `.\pytorch\torch\ao\pruning\_experimental\data_sparsifier\lightning\callbacks\_data_sparstity_utils.py`

```py
# mypy: allow-untyped-defs
# 导入日志模块
import logging
# 从 Torch 的剪枝实验性模块中导入基础数据稀疏化类支持的类型
from torch.ao.pruning._experimental.data_sparsifier.base_data_sparsifier import SUPPORTED_TYPES

# 创建一个名为 __name__ 的日志记录器对象
logger: logging.Logger = logging.getLogger(__name__)


def _attach_model_to_data_sparsifier(module, data_sparsifier, config=None):
    """Attaches a data sparsifier to all the layers of the module.
    Essentially, loop over all the weight parameters in the module and
    attach it to the data sparsifier.
    Note::
        The '.' in the layer names are replaced with '_' (refer to _get_valid_name() below)
        before attaching to the sparsifier. This is because, the data
        sparsifier uses a dummy model inside to store the weight parameters.
    """
    # 如果未提供配置，将配置设为一个空字典
    if config is None:
        config = {}
    # 遍历模块中所有命名参数
    for name, parameter in module.named_parameters():
        # 如果参数的类型在支持的类型中
        if type(parameter) in SUPPORTED_TYPES:
            # 获取有效的名称，将名称中的 '.' 替换为 '_'
            valid_name = _get_valid_name(name)
            # 将参数添加到数据稀疏化器中，使用默认配置（如果有的话）
            data_sparsifier.add_data(name=valid_name, data=parameter, **config.get(valid_name, {}))


def _get_valid_name(name):
    # 将名称中的 '.' 替换为 '_'
    return name.replace('.', '_')  # . is not allowed as a name


def _log_sparsified_level(model, data_sparsifier) -> None:
    # 展示每一层在稀疏化后的稀疏程度：
    for name, parameter in model.named_parameters():
        # 如果参数的类型不在支持的类型中，跳过当前循环
        if type(parameter) not in SUPPORTED_TYPES:
            continue
        # 获取有效的名称，将名称中的 '.' 替换为 '_'
        valid_name = _get_valid_name(name)
        # 获取名为 valid_name 的层的稀疏化掩码
        mask = data_sparsifier.get_mask(name=valid_name)
        # 计算稀疏化水平，即稀疏化后的非零值比例
        sparsity_level = 1.0 - mask.float().mean()
        # 记录日志，显示层 name 中的稀疏度级别
        logger.info(
            "Sparsity in layer %s = % .2%", name, sparsity_level
        )
```