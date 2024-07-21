# `.\pytorch\test\custom_operator\my_custom_ops.py`

```py
# 从模型中导入自定义操作库路径获取函数
from model import get_custom_op_library_path

# 导入 PyTorch 库
import torch

# 使用获取的自定义操作库路径加载自定义操作库
torch.ops.load_library(get_custom_op_library_path())

# 定义一个抽象方法装饰器，声明为 Torch 库的抽象方法，实现自定义操作 "custom::nonzero"
@torch.library.impl_abstract("custom::nonzero")
def nonzero_abstract(x):
    # 获取输入张量 x 的维度数
    n = x.dim()
    # 获取当前 Torch 库的上下文
    ctx = torch.library.get_ctx()
    # 创建一个未支持的符号整数（unbacked symbolic integer）
    nnz = ctx.create_unbacked_symint()
    # 定义形状为 [nnz, n] 的空张量，数据类型为长整型（torch.long）
    shape = [nnz, n]
    # 返回具有指定形状和数据类型的新张量
    return x.new_empty(shape, dtype=torch.long)
```