# `.\diffusers\pipelines\paint_by_example\__init__.py`

```py
# 从 dataclasses 模块导入 dataclass 装饰器，用于简化数据类的定义
from dataclasses import dataclass
# 从 typing 模块导入类型检查所需的类型提示
from typing import TYPE_CHECKING, List, Optional, Union

# 导入 numpy 库，通常用于数值计算
import numpy as np
# 导入 PIL 库，处理图像
import PIL
# 从 PIL 导入 Image 类，用于图像处理
from PIL import Image

# 从相对路径导入 utils 模块中的多个工具函数和常量
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 常量，指示是否需要慢速导入
    OptionalDependencyNotAvailable,  # 异常类，表示可选依赖不可用
    _LazyModule,  # 类，延迟加载模块的实现
    get_objects_from_module,  # 函数，从模块中获取对象
    is_torch_available,  # 函数，检查 PyTorch 是否可用
    is_transformers_available,  # 函数，检查 Transformers 是否可用
)

# 初始化一个空字典，存储虚拟对象
_dummy_objects = {}
# 初始化一个空字典，用于存储模块导入结构
_import_structure = {}

# 尝试检查可选依赖项是否可用
try:
    # 如果 Transformers 和 PyTorch 不可用，抛出异常
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖不可用的异常
except OptionalDependencyNotAvailable:
    # 从 utils 模块导入虚拟对象，避免错误使用
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    # 更新虚拟对象字典，添加从虚拟模块获取的对象
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
# 如果没有异常，更新导入结构字典
else:
    _import_structure["image_encoder"] = ["PaintByExampleImageEncoder"]  # 指定图像编码器模块
    _import_structure["pipeline_paint_by_example"] = ["PaintByExamplePipeline"]  # 指定图像处理管道模块

# 如果正在进行类型检查或需要慢速导入
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    # 尝试再次检查可选依赖项是否可用
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()

    # 捕获可选依赖不可用的异常
    except OptionalDependencyNotAvailable:
        # 从虚拟对象模块导入所有内容
        from ...utils.dummy_torch_and_transformers_objects import *
    # 如果没有异常，从实际模块导入所需的类
    else:
        from .image_encoder import PaintByExampleImageEncoder  # 导入图像编码器
        from .pipeline_paint_by_example import PaintByExamplePipeline  # 导入图像处理管道

# 如果不是类型检查且不需要慢速导入
else:
    import sys  # 导入 sys 模块以访问 Python 运行时

    # 使用 LazyModule 创建一个延迟加载的模块，减少启动时间
    sys.modules[__name__] = _LazyModule(
        __name__,  # 当前模块名
        globals()["__file__"],  # 当前模块的文件路径
        _import_structure,  # 导入结构
        module_spec=__spec__,  # 模块规范
    )

    # 将虚拟对象添加到当前模块中
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)  # 设置属性
```