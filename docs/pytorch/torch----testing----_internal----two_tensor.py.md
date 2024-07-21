# `.\pytorch\torch\testing\_internal\two_tensor.py`

```py
# 忽略 mypy 的错误检查

# 导入 PyTorch 库
import torch
# 导入 PyTorch 内部模块 _pytree
import torch.utils._pytree as pytree
# 从 torch.utils._python_dispatch 导入 return_and_correct_aliasing 函数

# 一个简单的张量子类，内部保存两个张量，并且对每个操作同时在两个张量上执行
class TwoTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, a, b):
        # 断言确保两个张量的设备、布局、梯度要求和数据类型相同
        assert (
            a.device == b.device
            and a.layout == b.layout
            and a.requires_grad == b.requires_grad
            and a.dtype == b.dtype
        )
        # 使用第一个张量 a 的形状创建新的张量
        shape = a.shape
        kwargs = {}
        kwargs["strides"] = a.stride()
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        # 使用给定参数创建一个新的张量子类
        out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

        # 断言确保两个张量具有相同的形状、步长和存储偏移
        assert a.shape == b.shape
        assert a.stride() == b.stride()
        assert a.storage_offset() == b.storage_offset()
        return out

    def __init__(self, a, b):
        # 初始化方法，保存两个输入张量 a 和 b
        self.a = a
        self.b = b

    def __repr__(self):
        # 返回对象的字符串表示，包括两个张量的 repr 字符串
        a_repr = repr(self.a)
        b_repr = repr(self.b)
        return f"TwoTensor({a_repr}, {b_repr})"

    def __tensor_flatten__(self):
        # 返回用于张量扁平化的属性名称列表和元数据（这里没有元数据）
        return ["a", "b"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        # 断言确保没有元数据传入
        assert meta is None
        # 从内部张量字典中取出张量 a 和 b
        a, b = inner_tensors["a"], inner_tensors["b"]
        # 返回一个新的 TwoTensor 对象
        return TwoTensor(a, b)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # 如果 kwargs 为 None，则初始化为空字典
        if kwargs is None:
            kwargs = {}
        # 对 args 和 kwargs 中所有的 TwoTensor 类型对象的属性 a 进行映射
        args_a = pytree.tree_map_only(TwoTensor, lambda x: x.a, args)
        # 对 args 和 kwargs 中所有的 TwoTensor 类型对象的属性 b 进行映射
        args_b = pytree.tree_map_only(TwoTensor, lambda x: x.b, args)

        # 对 args 和 kwargs 中所有的 TwoTensor 类型对象的属性 a 进行映射
        kwargs_a = pytree.tree_map_only(TwoTensor, lambda x: x.a, kwargs)
        # 对 args 和 kwargs 中所有的 TwoTensor 类型对象的属性 b 进行映射
        kwargs_b = pytree.tree_map_only(TwoTensor, lambda x: x.b, kwargs)

        # 分别调用 func 函数，并传入映射后的 args_a 和 kwargs_a，返回结果到 out_a
        out_a = func(*args_a, **kwargs_a)
        # 分别调用 func 函数，并传入映射后的 args_b 和 kwargs_b，返回结果到 out_b
        out_b = func(*args_b, **kwargs_b)
        # 断言确保 out_a 和 out_b 的类型相同
        assert type(out_a) == type(out_b)
        # 将 out_a 扁平化并返回结果与规范
        out_a_flat, spec = pytree.tree_flatten(out_a)
        # 将 out_b 展平并返回叶子节点列表
        out_b_flat = pytree.tree_leaves(out_b)
        # 对于返回非张量的 aten 操作，假设我们的两个内部张量返回相同的值
        # 如果 o_a 是 torch.Tensor 类型，则将其包装在 TwoTensor 中，否则直接使用 o_a
        out_flat = [
            TwoTensor(o_a, o_b) if isinstance(o_a, torch.Tensor) else o_a
            for o_a, o_b in zip(out_a_flat, out_b_flat)
        ]
        # 将扁平化的结果 out_flat 根据规范 spec 进行反扁平化，并返回结果
        out = pytree.tree_unflatten(out_flat, spec)
        # 调用 return_and_correct_aliasing 函数，处理别名问题，并返回结果
        return return_and_correct_aliasing(func, args, kwargs, out)


# 定义一个 TwoTensorMode 类，继承自 TorchDispatchMode 类
class TwoTensorMode(torch.utils._python_dispatch.TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # 调用 func 函数，并传入参数 args 和 kwargs，将结果保存到 out
        out = func(*args, **kwargs)
        # 如果 func 是 fake_tensor 的张量构造函数，则创建 TwoTensor 对象
        if torch._subclasses.fake_tensor._is_tensor_constructor(func):
            out = TwoTensor(out, out.clone())
        # 返回结果 out
        return out
```