# `.\lucidrains\pytorch-custom-utils\pytorch_custom_utils\module_device.py`

```
# 导入必要的模块
from functools import wraps
from typing import List
from optree import tree_flatten, tree_unflatten

import torch
from torch import is_tensor
from torch.nn import Module

# 为模型提供一个 .device 属性
# 使用一个虚拟的标量张量

def module_device(
    device_property_name = 'device'
):
    # 装饰器函数，用于装饰类
    def decorator(klass):
        # 断言被装饰的类是 torch.nn.Module 的子类
        assert issubclass(klass, Module), 'should decorate a subclass of torch.nn.Module'

        # 保存原始的 __init__ 方法
        _orig_init = klass.__init__

        @wraps(_orig_init)
        def __init__(self, *args, **kwargs):
            # 调用原始的 __init__ 方法
            _orig_init(self, *args, **kwargs)

            # 在模型中注册一个名为 '_dummy' 的缓冲区，值为 torch.tensor(0)，不持久化
            self.register_buffer('_dummy', torch.tensor(0), persistent = False)

        @property
        def _device_property(self):
            # 返回 '_dummy' 缓冲区的设备信息
            return self._dummy.device

        # 替换类的 __init__ 方法为自定义的 __init__ 方法
        klass.__init__ = __init__
        # 设置类的属性 device_property_name 为 _device_property
        setattr(klass, device_property_name, _device_property)
        return klass

    return decorator

# 一个装饰器，自动将传入 .forward 方法的所有张量转换为正确的设备

def autocast_device(
    methods: List[str] = ['forward']
):
    # 装饰器函数，用于装饰类
    def decorator(klass):
        # 断言被装饰的类是 torch.nn.Module 的子类
        assert issubclass(klass, Module), 'should decorate a subclass of torch.nn.Module'

        # 获取要装饰的方法的原始函数
        orig_fns = [getattr(klass, method) for method in methods]

        for method, orig_fn in zip(methods, orig_fns):

            @wraps(orig_fn)
            def fn(self, *args, **kwargs):

                # 确定设备
                # 使用上面装饰器中的虚拟张量
                # 否则查找参数并使用参数上的设备

                if hasattr(self, '_dummy'):
                    device = self._dummy.device
                else:
                    device = next(self.parameters()).device

                # 展平参数

                flattened_args, tree_spec = tree_flatten([args, kwargs])

                # 转换参数

                maybe_transformed_args = []

                for flattened_arg in flattened_args:
                    if is_tensor(flattened_arg):
                        flattened_arg = flattened_arg.to(device)

                    maybe_transformed_args.append(flattened_arg)

                # 还原参数

                args, kwargs = tree_unflatten(tree_spec, maybe_transformed_args)

                # 调用原始函数

                orig_fn(self, *args, **kwargs)

            # 设置类的方法为新的 fn 函数
            setattr(klass, method, fn)

        return klass

    return decorator
```