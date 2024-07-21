# `.\pytorch\torch\testing\_internal\custom_tensor.py`

```
# 忽略 mypy 的错误，通常用于在类型检查工具中排除特定的错误或警告
# Import torch 模块，用于科学计算和机器学习任务
import torch
# 引入 PyTorch 内部模块，用于支持树形数据结构的操作
import torch.utils._pytree as pytree
# 从 torch.utils._python_dispatch 模块中导入 return_and_correct_aliasing 函数，用于处理别名问题

# 自定义张量子类，包含了张量、自定义元数据和自定义方法
class ConstantExtraMetadataTensor(torch.Tensor):
    # 静态方法：创建新的 ConstantExtraMetadataTensor 实例
    @staticmethod
    def __new__(cls, elem):
        # 获取张量的形状
        shape = elem.shape
        # 构建关键字参数字典
        kwargs = {}
        # 设置步幅
        kwargs["strides"] = elem.stride()
        # 设置存储偏移量
        kwargs["storage_offset"] = elem.storage_offset()
        # 设置设备信息
        kwargs["device"] = elem.device
        # 设置布局信息
        kwargs["layout"] = elem.layout
        # 设置是否需要梯度
        kwargs["requires_grad"] = elem.requires_grad
        # 设置数据类型
        kwargs["dtype"] = elem.dtype
        # 调用父类构造方法，创建张量实例
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    # 构造方法：初始化实例时执行的操作
    def __init__(self, elem):
        # 存储传入的元素
        self.elem = elem
        # 设置常量属性值为 4
        self.constant_attribute = 4

    # repr 方法：返回实例的字符串表示
    def __repr__(self):
        # 获取元素的字符串表示
        inner_repr = repr(self.elem)
        # 返回自定义格式化字符串
        return f"CustomTensor({inner_repr})"

    # __tensor_flatten__ 方法：扁平化张量的自定义方法
    def __tensor_flatten__(self):
        # 返回元素名称列表和常量属性值
        return ["elem"], self.constant_attribute

    # add_constant 方法：向常量属性添加值的方法
    def add_constant(self, a):
        # 将传入值累加到常量属性上
        self.constant_attribute += a

    # 静态方法：反扁平化张量的自定义方法
    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        # 断言：确保元数据不为空
        assert meta is not None
        # 获取元素
        elem = inner_tensors["elem"]
        # 创建新的 ConstantExtraMetadataTensor 实例
        out = ConstantExtraMetadataTensor(elem)
        # 设置常量属性为传入的元数据
        out.constant_attribute = meta
        # 返回实例
        return out

    # 类方法：处理 torch 分发的自定义方法
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # 如果 kwargs 为 None，则初始化为空字典
        if kwargs is None:
            kwargs = {}
        # 对 args 中的元素应用 ConstantExtraMetadataTensor 函数，构建新的列表
        args_inner = pytree.tree_map_only(ConstantExtraMetadataTensor, lambda x: x.elem, args)
        # 对 kwargs 中的元素应用 ConstantExtraMetadataTensor 函数，构建新的列表
        kwargs_inner = pytree.tree_map_only(ConstantExtraMetadataTensor, lambda x: x.elem, kwargs)
        # 执行传入的函数，得到内部结果
        out_inner = func(*args_inner, **kwargs_inner)
        # 扁平化内部结果，并返回规范
        out_inner_flat, spec = pytree.tree_flatten(out_inner)
        # 对于返回非张量的 aten 操作，假定我们的定制内部张量返回相同值
        # 构建扁平化的输出列表，根据内部张量类型选择 ConstantExtraMetadataTensor 或直接值
        out_flat = [
            ConstantExtraMetadataTensor(o_inner) if isinstance(o_inner, torch.Tensor) else o_inner
            for o_inner in out_inner_flat
        ]
        # 将扁平化的输出列表还原为树形结构，根据规范
        out = pytree.tree_unflatten(out_flat, spec)
        # 返回修正别名问题后的结果
        return return_and_correct_aliasing(func, args, kwargs, out)
```