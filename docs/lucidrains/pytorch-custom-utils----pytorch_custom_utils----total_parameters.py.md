# `.\lucidrains\pytorch-custom-utils\pytorch_custom_utils\total_parameters.py`

```
# 从 torch.nn 模块中导入 Module 类
from torch.nn import Module

# 为你的模型提供一个 .total_parameters 属性，该属性简单地对所有模块的参数求和

# 定义一个装饰器函数，用于为类添加 total_parameters 属性
def total_parameters(
    count_only_requires_grad = False,  # 是否只计算需要梯度的参数
    total_parameters_property_name = 'total_parameters'  # total_parameters 属性的名称
):
    # 装饰器函数
    def decorator(klass):
        # 断言 klass 是 torch.nn.Module 的子类
        assert issubclass(klass, Module), 'should decorate a subclass of torch.nn.Module'

        # 定义一个计算所有参数数量的属性
        @property
        def _total_parameters(self):
            return sum(p.numel() for p in self.parameters())

        # 定义一个计算需要梯度的参数数量的属性
        @property
        def _total_parameters_with_requires_grad(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        # 根据 count_only_requires_grad 的值选择计算哪种参数数量
        fn = _total_parameters_with_requires_grad if count_only_requires_grad else  _total_parameters

        # 将计算参数数量的函数设置为 klass 的属性
        setattr(klass, total_parameters_property_name, fn)
        return klass

    return decorator
```