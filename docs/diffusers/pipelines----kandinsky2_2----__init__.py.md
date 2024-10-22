# `.\diffusers\pipelines\kandinsky2_2\__init__.py`

```py
# 导入类型检查的常量
from typing import TYPE_CHECKING

# 从 utils 模块导入所需的工具和常量
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 用于判断是否进行慢速导入
    OptionalDependencyNotAvailable,  # 自定义异常类，用于处理缺失依赖
    _LazyModule,  # 用于延迟加载模块
    get_objects_from_module,  # 从模块中获取对象的工具函数
    is_torch_available,  # 检查 PyTorch 是否可用
    is_transformers_available,  # 检查 Transformers 是否可用
)

# 初始化一个空字典用于存储虚拟对象
_dummy_objects = {}
# 初始化一个空字典用于存储导入结构
_import_structure = {}

try:
    # 检查 Transformers 和 PyTorch 是否都可用
    if not (is_transformers_available() and is_torch_available()):
        # 如果不可用，抛出缺失依赖异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 导入虚拟对象，以防依赖缺失
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    # 更新虚拟对象字典
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    # 添加相关的管道到导入结构字典
    _import_structure["pipeline_kandinsky2_2"] = ["KandinskyV22Pipeline"]
    _import_structure["pipeline_kandinsky2_2_combined"] = [
        "KandinskyV22CombinedPipeline",  # 组合管道
        "KandinskyV22Img2ImgCombinedPipeline",  # 图像到图像的组合管道
        "KandinskyV22InpaintCombinedPipeline",  # 图像修复的组合管道
    ]
    _import_structure["pipeline_kandinsky2_2_controlnet"] = ["KandinskyV22ControlnetPipeline"]  # 控制网管道
    _import_structure["pipeline_kandinsky2_2_controlnet_img2img"] = ["KandinskyV22ControlnetImg2ImgPipeline"]  # 控制网图像到图像的管道
    _import_structure["pipeline_kandinsky2_2_img2img"] = ["KandinskyV22Img2ImgPipeline"]  # 图像到图像的管道
    _import_structure["pipeline_kandinsky2_2_inpainting"] = ["KandinskyV22InpaintPipeline"]  # 图像修复管道
    _import_structure["pipeline_kandinsky2_2_prior"] = ["KandinskyV22PriorPipeline"]  # 先验管道
    _import_structure["pipeline_kandinsky2_2_prior_emb2emb"] = ["KandinskyV22PriorEmb2EmbPipeline"]  # 先验嵌入到嵌入的管道

# 如果处于类型检查模式或需要慢速导入
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        # 再次检查 Transformers 和 PyTorch 是否可用
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()  # 抛出异常

    except OptionalDependencyNotAvailable:
        # 导入虚拟对象以避免依赖问题
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        # 从各个管道模块导入相应的类
        from .pipeline_kandinsky2_2 import KandinskyV22Pipeline
        from .pipeline_kandinsky2_2_combined import (
            KandinskyV22CombinedPipeline,
            KandinskyV22Img2ImgCombinedPipeline,
            KandinskyV22InpaintCombinedPipeline,
        )
        from .pipeline_kandinsky2_2_controlnet import KandinskyV22ControlnetPipeline
        from .pipeline_kandinsky2_2_controlnet_img2img import KandinskyV22ControlnetImg2ImgPipeline
        from .pipeline_kandinsky2_2_img2img import KandinskyV22Img2ImgPipeline
        from .pipeline_kandinsky2_2_inpainting import KandinskyV22InpaintPipeline
        from .pipeline_kandinsky2_2_prior import KandinskyV22PriorPipeline
        from .pipeline_kandinsky2_2_prior_emb2emb import KandinskyV22PriorEmb2EmbPipeline

else:
    # 如果不处于类型检查或慢速导入，则创建懒加载模块
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,  # 模块名称
        globals()["__file__"],  # 模块文件路径
        _import_structure,  # 导入结构
        module_spec=__spec__,  # 模块规格
    )

    # 将虚拟对象添加到当前模块
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
```