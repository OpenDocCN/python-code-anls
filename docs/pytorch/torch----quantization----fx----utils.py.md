# `.\pytorch\torch\quantization\fx\utils.py`

```py
# flake8: noqa: F401
"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate files under `torch/ao/quantization/fx/`, while adding an import statement
here.
"""
# 从torch.ao.quantization.fx.utils模块中导入多个函数和类
from torch.ao.quantization.fx.utils import (
    all_node_args_have_no_tensors,  # 检查所有节点参数是否都没有张量
    assert_and_get_unique_device,    # 断言并获取唯一设备
    create_getattr_from_value,       # 从值创建getattr函数
    get_custom_module_class_keys,    # 获取自定义模块类的键
    get_linear_prepack_op_for_dtype, # 根据数据类型获取线性预打包操作
    get_new_attr_name_with_prefix,   # 使用前缀获取新的属性名
    get_non_observable_arg_indexes_and_types,  # 获取不可观察参数的索引和类型
    get_qconv_prepack_op,           # 获取量化卷积预打包操作
    graph_module_from_producer_nodes,  # 从生产者节点创建图模块
    maybe_get_next_module,           # 获取可能的下一个模块
)
```