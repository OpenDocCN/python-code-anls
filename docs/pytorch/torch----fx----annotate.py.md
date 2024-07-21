# `.\pytorch\torch\fx\annotate.py`

```py
# 引入类型提示允许未定义的函数
from torch.fx.proxy import Proxy
# 导入兼容性模块
from ._compatibility import compatibility

# 使用装饰器标记不向后兼容性
@compatibility(is_backward_compatible=False)
# 函数定义：给定一个值和类型，为 Proxy 对象添加类型注释
def annotate(val, type):
    """
    Annotates a Proxy object with a given type.

    This function annotates a val with a given type if a type of the val is a torch.fx.Proxy object
    Args:
        val (object): An object to be annotated if its type is torch.fx.Proxy.
        type (object): A type to be assigned to a given proxy object as val.
    Returns:
        The given val.
    Raises:
        RuntimeError: If a val already has a type in its node.
    """
    # 检查 val 是否为 Proxy 对象
    if isinstance(val, Proxy):
        # 如果 Proxy 对象已经有类型，则引发运行时错误
        if val.node.type:
            raise RuntimeError(f"Tried to annotate a value that already had a type on it!"
                               f" Existing type is {val.node.type} "
                               f"and new type is {type}. "
                               f"This could happen if you tried to annotate a function parameter "
                               f"value (in which case you should use the type slot "
                               f"on the function signature) or you called "
                               f"annotate on the same value twice")
        else:
            # 否则，为 Proxy 对象的节点添加新类型
            val.node.type = type
        # 返回已注释的 val
        return val
    else:
        # 如果 val 不是 Proxy 对象，则直接返回 val
        return val
```