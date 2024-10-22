# `.\diffusers\pipelines\stable_diffusion_safe\__init__.py`

```py
# 导入数据类装饰器，用于定义简单的类
from dataclasses import dataclass
# 导入枚举类，用于定义常量
from enum import Enum
# 导入类型检查相关工具
from typing import TYPE_CHECKING, List, Optional, Union

# 导入 NumPy 库
import numpy as np
# 导入 PIL 库用于图像处理
import PIL
# 从 PIL 导入图像模块
from PIL import Image

# 从 utils 模块导入多个工具和常量
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 慢导入的标志
    BaseOutput,  # 基础输出类
    OptionalDependencyNotAvailable,  # 可选依赖不可用的异常
    _LazyModule,  # 懒加载模块的类
    get_objects_from_module,  # 从模块获取对象的函数
    is_torch_available,  # 检查 Torch 是否可用的函数
    is_transformers_available,  # 检查 Transformers 是否可用的函数
)

# 定义安全配置的数据类
@dataclass
class SafetyConfig(object):
    # 弱安全级别配置
    WEAK = {
        "sld_warmup_steps": 15,  # 预热步骤数
        "sld_guidance_scale": 20,  # 引导比例
        "sld_threshold": 0.0,  # 阈值
        "sld_momentum_scale": 0.0,  # 动量比例
        "sld_mom_beta": 0.0,  # 动量 beta 值
    }
    # 中安全级别配置
    MEDIUM = {
        "sld_warmup_steps": 10,  # 预热步骤数
        "sld_guidance_scale": 1000,  # 引导比例
        "sld_threshold": 0.01,  # 阈值
        "sld_momentum_scale": 0.3,  # 动量比例
        "sld_mom_beta": 0.4,  # 动量 beta 值
    }
    # 强安全级别配置
    STRONG = {
        "sld_warmup_steps": 7,  # 预热步骤数
        "sld_guidance_scale": 2000,  # 引导比例
        "sld_threshold": 0.025,  # 阈值
        "sld_momentum_scale": 0.5,  # 动量比例
        "sld_mom_beta": 0.7,  # 动量 beta 值
    }
    # 最大安全级别配置
    MAX = {
        "sld_warmup_steps": 0,  # 预热步骤数
        "sld_guidance_scale": 5000,  # 引导比例
        "sld_threshold": 1.0,  # 阈值
        "sld_momentum_scale": 0.5,  # 动量比例
        "sld_mom_beta": 0.7,  # 动量 beta 值
    }

# 初始化空字典，存储虚拟对象
_dummy_objects = {}
# 初始化空字典，存储额外导入的对象
_additional_imports = {}
# 初始化导入结构的字典
_import_structure = {}

# 更新额外导入，添加 SafetyConfig 类
_additional_imports.update({"SafetyConfig": SafetyConfig})

# 尝试检查可选依赖
try:
    if not (is_transformers_available() and is_torch_available()):  # 检查 Transformers 和 Torch 是否可用
        raise OptionalDependencyNotAvailable()  # 若不可用，抛出异常
except OptionalDependencyNotAvailable:  # 捕获异常
    # 从 utils 导入虚拟对象以处理依赖问题
    from ...utils import dummy_torch_and_transformers_objects

    # 更新虚拟对象字典
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:  # 若依赖可用
    # 更新导入结构字典，添加相关模块
    _import_structure.update(
        {
            "pipeline_output": ["StableDiffusionSafePipelineOutput"],  # 管道输出模块
            "pipeline_stable_diffusion_safe": ["StableDiffusionPipelineSafe"],  # 稳定扩散安全管道
            "safety_checker": ["StableDiffusionSafetyChecker"],  # 安全检查器
        }
    )

# 如果进行类型检查或慢导入
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):  # 检查依赖可用性
            raise OptionalDependencyNotAvailable()  # 若不可用，抛出异常
    except OptionalDependencyNotAvailable:  # 捕获异常
        # 从 utils 导入虚拟对象
        from ...utils.dummy_torch_and_transformers_objects import *
    else:  # 若依赖可用
        # 从模块导入所需类
        from .pipeline_output import StableDiffusionSafePipelineOutput
        from .pipeline_stable_diffusion_safe import StableDiffusionPipelineSafe
        from .safety_checker import SafeStableDiffusionSafetyChecker

else:  # 若不是进行类型检查或慢导入
    import sys  # 导入 sys 模块

    # 使用懒加载模块更新当前模块
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    # 将虚拟对象添加到当前模块
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
    # 将额外导入的对象添加到当前模块
    for name, value in _additional_imports.items():
        setattr(sys.modules[__name__], name, value)
```