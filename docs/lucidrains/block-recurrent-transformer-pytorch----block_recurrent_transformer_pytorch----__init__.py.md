# `.\lucidrains\block-recurrent-transformer-pytorch\block_recurrent_transformer_pytorch\__init__.py`

```
# 导入 torch 库
import torch
# 从 packaging 库中导入 version 模块
from packaging import version

# 检查 torch 版本是否大于等于 '2.0.0'，如果是则执行以下代码
if version.parse(torch.__version__) >= version.parse('2.0.0'):
    # 从 einops._torch_specific 模块中导入 allow_ops_in_compiled_graph 函数
    from einops._torch_specific import allow_ops_in_compiled_graph
    # 调用 allow_ops_in_compiled_graph 函数
    allow_ops_in_compiled_graph()

# 从 block_recurrent_transformer_pytorch 包中导入 BlockRecurrentTransformer 和 RecurrentTrainerWrapper 类
from block_recurrent_transformer_pytorch.block_recurrent_transformer_pytorch import BlockRecurrentTransformer, RecurrentTrainerWrapper
```