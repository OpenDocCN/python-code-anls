# `transformer_vq\src\transformer_vq\utils\tree.py`

```
# 导入 flax 模块
import flax

# 定义一个函数，接受一个函数作为参数，并返回一个新的函数
def flattened_traversal(fn):
    """Returns function that is called with `(path, param)` instead of pytree."""
    # 定义内部函数 mask，接受一个树形结构作为参数
    def mask(tree):
        # 将树形结构展平为字典
        flat = flax.traverse_util.flatten_dict(tree)
        # 对展平后的字典中的每个键值对应用传入的函数 fn，并重新构建成树形结构
        return flax.traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})

    # 返回内部函数 mask
    return mask
```