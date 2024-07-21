# `.\pytorch\torch\ao\pruning\_mappings.py`

```py
# 设置模块级别变量 __all__，指定了可以被导出的符号列表
__all__ = [
    "get_static_sparse_quantized_mapping",
    "get_dynamic_sparse_quantized_mapping",
]

# 定义函数，返回静态稀疏量化映射的字典
def get_static_sparse_quantized_mapping():
    # 导入 torch.ao.nn.sparse 模块
    import torch.ao.nn.sparse
    # 定义静态稀疏量化映射字典，将 torch.nn.Linear 映射到 torch.ao.nn.sparse.quantized.Linear
    _static_sparse_quantized_mapping = {
        torch.nn.Linear: torch.ao.nn.sparse.quantized.Linear,
    }
    return _static_sparse_quantized_mapping

# 定义函数，返回动态稀疏量化映射的字典
def get_dynamic_sparse_quantized_mapping():
    # 导入 torch.ao.nn.sparse 模块
    import torch.ao.nn.sparse
    # 定义动态稀疏量化映射字典，将 torch.nn.Linear 映射到 torch.ao.nn.sparse.quantized.dynamic.Linear
    _dynamic_sparse_quantized_mapping = {
        torch.nn.Linear: torch.ao.nn.sparse.quantized.dynamic.Linear,
    }
    return _dynamic_sparse_quantized_mapping
```