# `D:\src\scipysrc\sympy\sympy\vector\kind.py`

```
# 导入必要的模块和类
from sympy.core.kind import Kind, _NumberKind, NumberKind
from sympy.core.mul import Mul

# 定义 VectorKind 类，继承自 Kind 类
class VectorKind(Kind):
    """
    Kind for all vector objects in SymPy.

    Parameters
    ==========

    element_kind : Kind
        Kind of the element. Default is
        :class:`sympy.core.kind.NumberKind`,
        which means that the vector contains only numbers.

    Examples
    ========

    Any instance of Vector class has kind ``VectorKind``:

    >>> from sympy.vector.coordsysrect import CoordSys3D
    >>> Sys = CoordSys3D('Sys')
    >>> Sys.i.kind
    VectorKind(NumberKind)

    Operations between instances of Vector keep also have the kind ``VectorKind``:

    >>> from sympy.core.add import Add
    >>> v1 = Sys.i * 2 + Sys.j * 3 + Sys.k * 4
    >>> v2 = Sys.i * Sys.x + Sys.j * Sys.y + Sys.k * Sys.z
    >>> v1.kind
    VectorKind(NumberKind)
    >>> v2.kind
    VectorKind(NumberKind)
    >>> Add(v1, v2).kind
    VectorKind(NumberKind)

    Subclasses of Vector also have the kind ``VectorKind``, such as
    Cross, VectorAdd, VectorMul or VectorZero.

    See Also
    ========

    sympy.core.kind.Kind
    sympy.matrices.kind.MatrixKind

    """

    # 构造函数，设置默认的 element_kind 为 NumberKind
    def __new__(cls, element_kind=NumberKind):
        # 调用父类的构造函数
        obj = super().__new__(cls, element_kind)
        # 设置实例变量 element_kind
        obj.element_kind = element_kind
        return obj

    # 重写 __repr__ 方法，返回类的字符串表示形式
    def __repr__(self):
        return "VectorKind(%s)" % self.element_kind

# 注册 Mul 类的 _kind_dispatcher 方法，处理 _NumberKind 与 VectorKind 类型之间的乘法
@Mul._kind_dispatcher.register(_NumberKind, VectorKind)
def num_vec_mul(k1, k2):
    """
    The result of a multiplication between a number and a Vector should be of VectorKind.
    The element kind is selected by recursive dispatching.
    """
    # 如果 k2 不是 VectorKind 类型，则交换 k1 和 k2
    if not isinstance(k2, VectorKind):
        k1, k2 = k2, k1
    # 递归调用 Mul 类的 _kind_dispatcher 方法，选择元素类型
    elemk = Mul._kind_dispatcher(k1, k2.element_kind)
    # 返回一个新的 VectorKind 实例，其 element_kind 为 elemk
    return VectorKind(elemk)
```