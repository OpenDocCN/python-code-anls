# `.\pytorch\torch\fx\passes\dialect\common\cse_pass.py`

```
# mypy: allow-untyped-defs
# 引入必要的类型注解
from typing import Dict, Tuple, Any

# 引入 torch 库
import torch
# 导入 PassBase 和 PassResult 类
from torch.fx.passes.infra.pass_base import PassBase, PassResult
# 导入 tree_flatten 函数
from torch.utils._pytree import tree_flatten

# 导入 GraphModule、Graph 和 Node 类
from torch.fx import GraphModule, Graph, Node

# 获取 torch.ops.aten 模块的别名
aten = torch.ops.aten

# 禁止在 CSE 中进行共享的随机操作集合
rand_ops = {
    aten.dropout, aten._fused_dropout, aten._standard_gamma,
    aten.bernoulli, aten.multinomial, aten.native_dropout,
    aten.normal, aten.poisson, aten.binomial, aten.rrelu,
    aten.rand_like, aten.rand, aten.randint, aten.randn, aten.randperm
}  # noqa: E501,B950

# 禁止原地操作的集合
inplace_ops = {
    aten.add_, aten.sub_, aten.mul_, aten.div_, aten.pow_,
    aten.lerp_, aten.relu_, aten.sigmoid_, aten.tanh_
}  # noqa: E501

# 函数装饰器，标记此函数为不向后兼容的 FX 兼容性函数
@torch.fx._compatibility.compatibility(is_backward_compatible=False)
# 返回 CSE 过程中禁止共享的操作集合
def get_CSE_banned_ops():
    return rand_ops.union(inplace_ops)

# CSEPass 类，继承自 PassBase 类
@torch.fx._compatibility.compatibility(is_backward_compatible=False)
class CSEPass(PassBase):

    # 初始化方法
    def __init__(self, banned_ops=None):
        """
        This version of CSE Pass aims to be dialect agnostic, and it's implemented purely based on the connectivity between fx.Node.

        For functional dialects, user would only need to specify the random ops in ban list.

        Warning: CSE Pass cannot be safely applied on a FX graph in non-functional dialects.
        If your dialect contains stateful operators, please customized the banned_ops.

        """
        # 如果 banned_ops 参数为 None，则设为一个空集合
        if banned_ops is None:
            banned_ops = set()
        # 将传入的 banned_ops 参数赋值给实例变量 self.banned_ops
        self.banned_ops = banned_ops
        # 调用父类 PassBase 的初始化方法
        super().__init__()
```