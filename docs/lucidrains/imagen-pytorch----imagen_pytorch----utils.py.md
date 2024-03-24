# `.\lucidrains\imagen-pytorch\imagen_pytorch\utils.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 functools 库中导入 reduce 函数
from functools import reduce
# 从 pathlib 库中导入 Path 类
from pathlib import Path

# 从 imagen_pytorch.configs 模块中导入 ImagenConfig 和 ElucidatedImagenConfig 类
from imagen_pytorch.configs import ImagenConfig, ElucidatedImagenConfig
# 从 ema_pytorch 模块中导入 EMA 类

from ema_pytorch import EMA

# 定义一个函数，用于检查变量是否存在
def exists(val):
    return val is not None

# 定义一个函数，用于安全获取字典中的值
def safeget(dictionary, keys, default = None):
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split('.'), dictionary)

# 加载模型和配置信息
def load_imagen_from_checkpoint(
    checkpoint_path,
    load_weights = True,
    load_ema_if_available = False
):
    # 创建 Path 对象
    model_path = Path(checkpoint_path)
    # 获取完整的模型路径
    full_model_path = str(model_path.resolve())
    # 断言模型路径存在
    assert model_path.exists(), f'checkpoint not found at {full_model_path}'
    # 加载模型参数
    loaded = torch.load(str(model_path), map_location='cpu')

    # 获取 imagen 参数和类型
    imagen_params = safeget(loaded, 'imagen_params')
    imagen_type = safeget(loaded, 'imagen_type')

    # 根据 imagen 类型选择对应的配置类
    if imagen_type == 'original':
        imagen_klass = ImagenConfig
    elif imagen_type == 'elucidated':
        imagen_klass = ElucidatedImagenConfig
    else:
        raise ValueError(f'unknown imagen type {imagen_type} - you need to instantiate your Imagen with configurations, using classes ImagenConfig or ElucidatedImagenConfig')

    # 断言 imagen 参数和类型存在
    assert exists(imagen_params) and exists(imagen_type), 'imagen type and configuration not saved in this checkpoint'

    # 根据配置类和参数创建 imagen 对象
    imagen = imagen_klass(**imagen_params).create()

    # 如果不加载权重，则直接返回 imagen 对象
    if not load_weights:
        return imagen

    # 检查是否存在 EMA 模型
    has_ema = 'ema' in loaded
    should_load_ema = has_ema and load_ema_if_available

    # 加载模型参数
    imagen.load_state_dict(loaded['model'])

    # 如果不需要加载 EMA 模型，则直接返回 imagen 对象
    if not should_load_ema:
        print('loading non-EMA version of unets')
        return imagen

    # 创建 EMA 模型列表
    ema_unets = nn.ModuleList([])
    # 遍历 imagen.unets，为每个 unet 创建一个 EMA 模型
    for unet in imagen.unets:
        ema_unets.append(EMA(unet))

    # 加载 EMA 模型参数
    ema_unets.load_state_dict(loaded['ema'])

    # 将 EMA 模型参数加载到对应的 unet 模型中
    for unet, ema_unet in zip(imagen.unets, ema_unets):
        unet.load_state_dict(ema_unet.ema_model.state_dict())

    # 打印信息并返回 imagen 对象
    print('loaded EMA version of unets')
    return imagen
```