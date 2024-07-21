# `.\pytorch\torch\testing\_internal\common_subclass.py`

```py
# mypy: ignore-errors  # 忽略类型检查错误，允许代码通过静态类型检查时忽略特定的错误

import torch  # 导入PyTorch库
from copy import deepcopy  # 导入深拷贝函数deepcopy
from torch.utils._pytree import tree_map  # 导入torch内部的tree_map函数

# TODO: Move LoggingTensor here.  # 待办事项：将LoggingTensor类移到这里，暂时未实现

from torch.testing._internal.logging_tensor import LoggingTensor  # 从torch内部的测试模块中导入LoggingTensor类


# Base class for wrapper-style tensors.
class WrapperTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, *args, **kwargs):
        # 获取包装属性，可能从输入参数中获得，生成一个新的子类
        t, kwargs = cls.get_wrapper_properties(*args, **kwargs)
        if "size" not in kwargs:
            size = t.size()
        else:
            size = kwargs["size"]
            del kwargs["size"]
        if "dtype" not in kwargs:
            kwargs["dtype"] = t.dtype
        if "layout" not in kwargs:
            kwargs["layout"] = t.layout
        if "device" not in kwargs:
            kwargs["device"] = t.device
        if "requires_grad" not in kwargs:
            kwargs["requires_grad"] = False
        # Ignore memory_format and pin memory for now as I don't know how to
        # safely access them on a Tensor (if possible??)
        
        # 创建一个包装子类，并校验方法
        wrapper = torch.Tensor._make_wrapper_subclass(cls, size, **kwargs)
        wrapper._validate_methods()  # 校验方法有效性
        return wrapper

    @classmethod
    def get_wrapper_properties(cls, *args, **kwargs):
        # 应当返回一个示例Tensor和一个用于覆盖该示例Tensor属性的kwargs字典
        # 这与`t.new_*(args)` API非常相似
        raise NotImplementedError("You need to implement get_wrapper_properties")  # 抛出未实现错误

    def _validate_methods(self):
        # 如果不是调试模式，则跳过此步骤
        # 在Python端更改这些方法是错误的，因为它们不会正确反映到C++端
        # 这不会捕捉在__init__中设置的属性
        forbidden_overrides = ["size", "stride", "dtype", "layout", "device", "requires_grad"]
        for el in forbidden_overrides:
            if getattr(self.__class__, el) is not getattr(torch.Tensor, el):
                raise RuntimeError(f"Subclass {self.__class__.__name__} is overwriting the "
                                   f"property {el} but this is not allowed as such change would "
                                   "not be reflected to c++ callers.")

class DiagTensorBelow(WrapperTensor):
    @classmethod
    def get_wrapper_properties(cls, diag, requires_grad=False):
        assert diag.ndim == 1  # 断言diag的维度为1
        return diag, {"size": diag.size() + diag.size(), "requires_grad": requires_grad}  # 返回diag和包含size和requires_grad的字典

    def __init__(self, diag, requires_grad=False):
        self.diag = diag  # 初始化self.diag为传入的diag

    handled_ops = {}  # 初始化handled_ops为空字典

    @classmethod
    # 定义特殊方法 __torch_dispatch__，用于处理特定的方法调度
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # 检查所有传入类型是否都是cls的子类，否则返回NotImplemented
        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

        # 根据函数名从handled_ops字典中获取对应的处理函数
        fn = cls.handled_ops.get(func.__name__, None)
        if fn:
            # 如果找到对应的处理函数，则调用该函数并传入参数
            return fn(*args, **(kwargs or {}))
        else:
            # 如果没有找到对应的处理函数，则执行默认的“回退”逻辑
            # 创建一个普通的Tensor，基于diag元素，并再次调用原始函数

            # 定义unwrap函数，根据情况返回对角线张量或者原始对象
            def unwrap(e):
                return e.diag.diag() if isinstance(e, DiagTensorBelow) else e

            # 定义wrap函数，根据情况包装成对角线张量DiagTensorBelow或者保持不变
            def wrap(e):
                if isinstance(e, torch.Tensor) and e.ndim == 1:
                    return DiagTensorBelow(e)
                if isinstance(e, torch.Tensor) and e.ndim == 2 and e.count_nonzero() == e.diag().count_nonzero():
                    return DiagTensorBelow(e.diag())
                return e

            # 使用tree_map函数将unwrap应用到所有参数上，并调用原始函数，再应用wrap函数到结果上
            rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {})))
            # 返回处理结果
            return rs

    # 定义特殊方法 __repr__，用于返回对象的字符串表示形式
    def __repr__(self):
        # 调用父类的__repr__方法，并传入自定义的tensor_contents参数，展示对角线元素的信息
        return super().__repr__(tensor_contents=f"diag={self.diag}")
# 定义一个稀疏张量类 SparseTensor，继承自 WrapperTensor
class SparseTensor(WrapperTensor):
    
    # 类方法，返回值张量和属性字典
    @classmethod
    def get_wrapper_properties(cls, size, values, indices, requires_grad=False):
        # 断言值张量和索引张量的设备相同
        assert values.device == indices.device
        # 返回值张量和属性字典，包括尺寸和是否需要梯度
        return values, {"size": size, "requires_grad": requires_grad}

    # 初始化方法，接受尺寸、值张量、索引张量和是否需要梯度作为参数
    def __init__(self, size, values, indices, requires_grad=False):
        # 将传入的值张量赋给对象的 values 属性
        self.values = values
        # 将传入的索引张量赋给对象的 indices 属性
        self.indices = indices

    # 字符串表示方法，返回对象的详细字符串表示，包括值张量和索引张量
    def __repr__(self):
        return super().__repr__(tensor_contents=f"values={self.values}, indices={self.indices}")

    # 稀疏到密集的转换方法，创建并返回一个与尺寸相同的零张量，然后填充值张量的值到相应的索引位置
    def sparse_to_dense(self):
        res = torch.zeros(self.size(), dtype=self.values.dtype)
        res[self.indices.unbind(1)] = self.values
        return res

    # 从密集张量转换为稀疏张量的静态方法，返回一个 SparseTensor 对象
    @staticmethod
    def from_dense(t):
        indices = t.nonzero()
        values = t[indices.unbind(1)]
        return SparseTensor(t.size(), values, indices)

    # 类方法，处理 torch 函数的分派，调用特定实现或使用默认的稠密张量构造方法
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # 获取函数的完整名称
        func_name = f"{func.__module__}.{func.__name__}"

        # 尝试调用特定实现方法
        res = cls._try_call_special_impl(func_name, args, kwargs)
        if res is not NotImplemented:
            return res

        # 否则，使用默认的方法构造稠密张量并计算值
        def unwrap(e):
            return e.sparse_to_dense() if isinstance(e, SparseTensor) else e

        def wrap(e):
            return SparseTensor.from_dense(e) if isinstance(e, torch.Tensor) else e

        # 使用 tree_map 将函数应用于参数并返回结果
        rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {})))
        return rs

    # 右乘操作符重载，继承自父类的实现
    def __rmul__(self, other):
        return super().__rmul__(other)

    # 特殊实现方法的字典，存储特定函数的实现方法
    _SPECIAL_IMPLS = {}

    # 类方法，尝试调用特定的实现方法
    @classmethod
    def _try_call_special_impl(cls, func, args, kwargs):
        if func not in cls._SPECIAL_IMPLS:
            return NotImplemented
        return cls._SPECIAL_IMPLS[func](args, kwargs)


# 非包装器子类示例，存储额外状态
class NonWrapperTensor(torch.Tensor):
    
    # 创建新对象的静态方法，添加额外状态 'last_func_called'
    def __new__(cls, data):
        t = torch.Tensor._make_subclass(cls, data)
        t.extra_state = {
            'last_func_called': None
        }
        return t

    # torch 函数的方法重载，处理额外状态的保存与复制
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        result = super().__torch_function__(func, types, args, kwargs)

        if isinstance(result, cls):
            # 更新额外状态，记录最后调用的函数名
            if func is torch.Tensor.__deepcopy__:
                result.extra_state = deepcopy(args[0].extra_state)
            else:
                result.extra_state = {
                    'last_func_called': func.__name__,
                }

        return result

    # 必须定义的方法，用于深复制操作
    # new_empty() must be defined for deepcopy to work
    # 定义一个方法 `new_empty`，接受一个 `shape` 参数，返回一个新对象，
    # 其中包含一个使用给定形状创建的空的 Torch 张量
    def new_empty(self, shape):
        # 调用当前对象的类型构造函数，使用 `torch.empty` 函数创建指定形状的空张量，
        # 然后返回该对象
        return type(self)(torch.empty(shape))
# 用于存储有关在测试中使用的子类张量的信息的类。
class SubclassInfo:
    
    __slots__ = ['name', 'create_fn', 'closed_under_ops']
    
    def __init__(self, name, create_fn, closed_under_ops=True):
        self.name = name
        self.create_fn = create_fn  # create_fn(shape) -> tensor instance
        self.closed_under_ops = closed_under_ops
        # 初始化函数，接受名称、创建函数和是否在操作下封闭的布尔值作为参数

subclass_db = {
    torch.Tensor: SubclassInfo(
        'base_tensor', create_fn=torch.randn
    ),
    NonWrapperTensor: SubclassInfo(
        'non_wrapper_tensor',
        create_fn=lambda shape: NonWrapperTensor(torch.randn(shape))
    ),
    LoggingTensor: SubclassInfo(
        'logging_tensor',
        create_fn=lambda shape: LoggingTensor(torch.randn(shape))
    ),
    SparseTensor: SubclassInfo(
        'sparse_tensor',
        create_fn=lambda shape: SparseTensor.from_dense(torch.randn(shape).relu())
    ),
    DiagTensorBelow: SubclassInfo(
        'diag_tensor_below',
        create_fn=lambda shape: DiagTensorBelow(torch.randn(shape)),
        closed_under_ops=False  # sparse semantics
    ),
}
# 子类张量类型作为键，SubclassInfo 的实例作为值的字典，每个实例包括名称、创建函数和操作封闭状态的信息
```