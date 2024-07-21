# `.\pytorch\torch\quantization\__init__.py`

```py
# mypy: allow-untyped-defs
# 导入quantize模块下所有内容，禁止F403错误
from .quantize import *  # noqa: F403
# 导入observer模块下所有内容，禁止F403错误
from .observer import *  # noqa: F403
# 导入qconfig模块下所有内容，禁止F403错误
from .qconfig import *  # noqa: F403
# 导入fake_quantize模块下所有内容，禁止F403错误
from .fake_quantize import *  # noqa: F403
# 导入fuse_modules模块中的fuse_modules函数
from .fuse_modules import fuse_modules
# 导入stubs模块下所有内容，禁止F403错误
from .stubs import *  # noqa: F403
# 导入quant_type模块下所有内容，禁止F403错误
from .quant_type import *  # noqa: F403
# 导入quantize_jit模块下所有内容，禁止F403错误
from .quantize_jit import *  # noqa: F403

# 从quantization_mappings模块导入所有内容，禁止F403错误
from .quantization_mappings import *  # noqa: F403
# 从fuser_method_mappings模块导入所有内容，禁止F403错误
from .fuser_method_mappings import *  # noqa: F403

# 默认的模型评估函数，接受一个模型和校准数据作为输入
def default_eval_fn(model, calib_data):
    r"""
    默认评估函数，接受torch.utils.data.Dataset或输入张量列表，并在数据集上运行模型
    """
    # 遍历校准数据的每个数据和目标
    for data, target in calib_data:
        # 在模型上应用数据
        model(data)


# __all__ 列表定义了模块中的公共接口和导出的符号列表
__all__ = [
    "QuantWrapper",
    "QuantStub",
    "DeQuantStub",
    # 顶层API，用于即时模式量化
    "quantize",
    "quantize_dynamic",
    "quantize_qat",
    "prepare",
    "convert",
    "prepare_qat",
    # 顶层API，用于图模式下的TorchScript量化
    "quantize_jit",
    "quantize_dynamic_jit",
    "_prepare_ondevice_dynamic_jit",
    "_convert_ondevice_dynamic_jit",
    "_quantize_ondevice_dynamic_jit",
    # 顶层API，用于图模式下的GraphModule(torch.fx)量化
    # 'fuse_fx', 'quantize_fx',  # TODO: 添加quantize_dynamic_fx
    # 'prepare_fx', 'prepare_dynamic_fx', 'convert_fx',
    "QuantType",  # 量化类型
    # 自定义模块API
    "get_default_static_quant_module_mappings",
    "get_static_quant_module_class",
    "get_default_dynamic_quant_module_mappings",
    "get_default_qat_module_mappings",
    "get_default_qconfig_propagation_list",
    "get_default_compare_output_module_list",
    "get_quantized_operator",
    "get_fuser_method",
    # 用于`prepare`和`swap_module`的子函数
    "propagate_qconfig_",
    "add_quant_dequant",
    "swap_module",
    "default_eval_fn",  # 默认评估函数
    # Observers（观察器）
    "ObserverBase",
    "WeightObserver",
    "HistogramObserver",
    "observer",
    "default_observer",
    "default_weight_observer",
    "default_placeholder_observer",
    "default_per_channel_weight_observer",
    # FakeQuantize（用于量化训练的假量化）
    "default_fake_quant",
    "default_weight_fake_quant",
    "default_fixed_qparams_range_neg1to1_fake_quant",
    "default_fixed_qparams_range_0to1_fake_quant",
    "default_per_channel_weight_fake_quant",
    "default_histogram_fake_quant",
    # QConfig（量化配置）
    "QConfig",
    "default_qconfig",
    "default_dynamic_qconfig",
    "float16_dynamic_qconfig",
    "float_qparams_weight_only_qconfig",
    # QAT（量化感知训练）工具
    "default_qat_qconfig",
    "prepare_qat",
    "quantize_qat",
    # 模块转换
    "fuse_modules",
]
```