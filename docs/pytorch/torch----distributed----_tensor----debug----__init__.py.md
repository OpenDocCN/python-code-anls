# `.\pytorch\torch\distributed\_tensor\debug\__init__.py`

```
# 引入类型提示允许未定义的函数或方法
# 从torch.distributed._tensor.api模块导入DTensor类
from torch.distributed._tensor.api import DTensor
# 从torch.distributed._tensor.debug.comm_mode模块导入CommDebugMode类

# 定义一个函数，获取分片传播缓存信息
def get_sharding_prop_cache_info():
    """
    Get the cache info for the sharding propagation cache, used for debugging purpose only.
    This would return a named tuple showing hits, misses, maxsize and cursize of the sharding
    propagator cache.
    """
    # 调用DTensor类的_op_dispatcher属性，然后访问其中的sharding_propagator属性，
    # 继续调用propagate_op_sharding方法，最后调用cache_info方法获取缓存信息
    return (
        DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding.cache_info()  # type:ignore[attr-defined]
    )
```