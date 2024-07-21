# `.\pytorch\torch\fx\tensor_type.py`

```
# 导入不带类型定义的 Var 类型（忽略类型检查）
from torch.fx.experimental.unification import Var  # type: ignore[attr-defined]

# 导入兼容性修饰器 compatibility
from ._compatibility import compatibility

# 使用 compatibility 修饰器定义一个类 TensorType，指定其不向后兼容
@compatibility(is_backward_compatible=False)
class TensorType:
    """
    TensorType 定义了张量的类型，由一个维度列表组成。
    示例:
        class M(torch.nn.Module):
            def forward(self, x:TensorType((1,2,3, Dyn)), y:TensorType((1,2,3, Dyn))):
                return torch.add(x, y)
    """

    # 初始化方法，接受一个维度 dim 的参数
    def __init__(self, dim):
        self.__origin__ = TensorType  # 设置类的原始类型为 TensorType
        self.__args__ = dim  # 将传入的维度参数保存在 __args__ 属性中

    # 返回 TensorType 对象的字符串表示形式
    def __repr__(self):
        return f'TensorType[{self.__args__}]'

    # 判断两个 TensorType 对象是否相等的方法
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return list(self.__args__) == list(other.__args__)
        else:
            return False

    # 静态方法，支持通过类似 TensorType[(1,2,3)] 的方式获取 TensorType 对象
    @staticmethod
    def __class_getitem__(*args):
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        return TensorType(tuple(args))


# 定义一个特殊类型 _DynType，表示缺少类型信息
class _DynType:
    """
    _DynType 定义了一个特殊类型，代表缺少类型信息。
    """
    def __init__(self):
        self.__name__ = '_DynType'

    # 判断两个 _DynType 对象是否相等的方法
    def __eq__(self, other):
        return isinstance(other, self.__class__)

    # 返回 _DynType 对象的字符串表示形式
    def __str__(self):
        return "Dyn"

    # 返回 _DynType 对象的字符串表示形式（用于调试）
    def __repr__(self):
        return "Dyn"


# 创建一个 _DynType 对象的实例，并命名为 Dyn
Dyn = _DynType()

# 定义一个函数 is_consistent，用于判断两个类型是否一致
@compatibility(is_backward_compatible=False)
def is_consistent(t1, t2):
    """
    由 ~ 表示的二元关系，确定 t1 是否与 t2 一致。
    该关系是自反的、对称的，但不是传递的。
    如果 t1 和 t2 一致，则返回 True，否则返回 False。
    示例:
        Dyn ~ TensorType((1,2,3))
        int ~ Dyn
        int ~ int
        TensorType((1,Dyn,3)) ~ TensorType((1,2,3))
    """

    # 如果 t1 和 t2 相等，则返回 True
    if t1 == t2:
        return True

    # 如果 t1 或 t2 是 Dyn 类型，或者是 Var 类型的实例，则返回 True
    if t1 == Dyn or t2 == Dyn or isinstance(t1, Var) or isinstance(t2, Var):
        return True

    # 如果 t1 和 t2 都是 TensorType 类型的实例，则比较它们的维度列表是否一致
    if isinstance(t1, TensorType) and isinstance(t2, TensorType):
        return len(t1.__args__) == len(t2.__args__) and \
            all(is_consistent(elem1, elem2) for elem1, elem2 in zip(t1.__args__, t2.__args__))
    else:
        return False


# 定义一个函数 is_more_precise，用于判断 t1 是否比 t2 更精确
@compatibility(is_backward_compatible=False)
def is_more_precise(t1, t2):
    """
    由 <= 表示的二元关系，确定 t1 是否比 t2 更精确。
    该关系是自反的和传递的。
    如果 t1 比 t2 更精确，则返回 True，否则返回 False。
    示例:
        Dyn >= TensorType((1,2,3))
        int >= Dyn
        int >= int
        TensorType((1,Dyn,3)) <= TensorType((1,2,3))
    """
    # 如果 t1 和 t2 相等，则返回 True
    if t1 == t2:
        return True

    # 如果 t2 是 _DynType 类型的实例，则返回 True
    if isinstance(t2, _DynType):
        return True

    # 如果 t1 和 t2 都是 TensorType 类型的实例，则比较它们的维度列表是否一致
    if isinstance(t1, TensorType) and isinstance(t2, TensorType):
        return len(t1.__args__) == len(t2.__args__) and \
            all(is_more_precise(elem1, elem2) for elem1, elem2 in zip(t1.__args__, t2.__args__))
    else:
        return False
```