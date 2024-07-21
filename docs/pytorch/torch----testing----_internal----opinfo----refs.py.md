# `.\pytorch\torch\testing\_internal\opinfo\refs.py`

```
# mypy: ignore-errors

# 导入必要的模块和类
from torch.testing._internal.opinfo.core import (
    BinaryUfuncInfo,
    OpInfo,
    ReductionOpInfo,
    UnaryUfuncInfo,
)

# NOTE [Python References]
# Python References模拟现有的PyTorch操作，但最终可以用torch._prims的原始操作来表达。
#
# 这些引用是实验性的。
# 额外的背景信息请参见https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-0/577。
#
# Python Reference OpInfo应该添加到下面的python_ref_db列表中。
# 测试可以通过将该列表包含在@ops装饰器传递的Sequence中来选择运行这些引用。
#
# 在构造Python Reference OpInfo时，必须使用torch_opinfo_name关键字参数提供指向现有OpInfo的指针。
# 必须找到没有变体的具有该名称的现有OpInfo以继承。
#
# Python Reference OpInfos不仅继承现有OpInfo的元数据，还继承现有OpInfo的构造参数。
# 可以通过向构造函数添加kwargs来覆盖这些参数。

def _find_referenced_opinfo(referenced_name, variant_name, *, op_db=None):
    """
    根据给定的名称和变体名称查找OpInfo。
    """
    # 注意：当OpInfos分割到不同的模块时，全局op_db的搜索将不起作用，
    # 因为op_db可能尚未完全构建。因此，必须显式地传递本地的op_db。
    if op_db is None:
        from torch.testing._internal.common_methods_invocations import op_db

    # 遍历op_db中的OpInfo对象，寻找匹配指定名称和变体名称的OpInfo对象
    for opinfo in op_db:
        if opinfo.name == referenced_name and opinfo.variant_test_name == variant_name:
            return opinfo

def _inherit_constructor_args(name, op, inherited, overrides):
    """
    继承构造函数参数并进行必要的修正和覆盖。
    """
    # 继承的通用参数
    common_kwargs = {
        "name": name,
        "op": op,
        "aliases": None,  # TODO 添加别名覆盖的检查
        "method_variant": None,
        "inplace_variant": None,  # TODO：添加inplace覆盖的检查
        "supports_scripting": False,
    }

    # 复制已继承的kwargs
    kwargs = inherited.copy()

    # 修正metadata
    if "kwargs" in kwargs:
        kwargs.update(kwargs["kwargs"])
        del kwargs["kwargs"]
    if "self" in kwargs:
        del kwargs["self"]
    if "__class__" in kwargs:
        del kwargs["__class__"]
    if "skips" in kwargs:
        del kwargs["skips"]
    if "decorators" in kwargs:
        del kwargs["decorators"]

    # 覆盖metadata
    kwargs.update(common_kwargs)
    kwargs.update(overrides)

    # 目前没有原始支持自动求导，因此不应在测试dtype支持时运行自动求导测试。
    # 一旦开始为原始编写自动求导公式，可以删除此限制。
    kwargs["supports_autograd"] = False
    kwargs["supports_gradgrad"] = False
    kwargs["supports_fwgrad_bwgrad"] = False
    kwargs["supports_inplace_autograd"] = False
    # 设置关键字参数字典中的键 supports_forward_ad 的值为 False
    kwargs["supports_forward_ad"] = False

    # 返回更新后的关键字参数字典
    return kwargs
class PythonRefInfo(OpInfo):
    """
    An OpInfo for a Python reference of an OpInfo base class operation.
    """

    def __init__(
        self,
        name,  # the stringname of the callable Python reference
        *,
        op=None,  # the function variant of the operation, populated as torch.<name> if None
        op_db=None,  # The database of opinfos to search for the parent opinfo
        torch_opinfo_name,  # the string name of the corresponding torch opinfo
        torch_opinfo_variant_name="",  # the variant name for corresponding torch opinfo
        validate_view_consistency=True,
        **kwargs,
    ):  # additional kwargs override kwargs inherited from the torch opinfo
        # Initialize the base class OpInfo with inherited arguments
        self.torch_opinfo_name = torch_opinfo_name
        self.torch_opinfo_variant_name = torch_opinfo_variant_name
        # Find and assign the referenced OpInfo from the op_db
        self.torch_opinfo = _find_referenced_opinfo(
            torch_opinfo_name, torch_opinfo_variant_name, op_db=op_db
        )
        self.validate_view_consistency = validate_view_consistency
        # Ensure the referenced opinfo is an instance of OpInfo
        assert isinstance(self.torch_opinfo, OpInfo)

        # Inherit constructor arguments from the referenced opinfo
        inherited = self.torch_opinfo._original_opinfo_args
        ukwargs = _inherit_constructor_args(name, op, inherited, kwargs)
        # Call the parent class constructor with inherited and updated kwargs
        super().__init__(**ukwargs)


class ReductionPythonRefInfo(ReductionOpInfo):
    """
    An OpInfo for a Python reference of an elementwise unary operation.
    """

    def __init__(
        self,
        name,  # the stringname of the callable Python reference
        *,
        op=None,  # the function variant of the operation, populated as torch.<name> if None
        op_db=None,  # The database of opinfos to search for the parent opinfo
        torch_opinfo_name,  # the string name of the corresponding torch opinfo
        torch_opinfo_variant_name="",  # the variant name for corresponding torch opinfo
        **kwargs,
    ):  # additional kwargs override kwargs inherited from the torch opinfo
        # Initialize the base class ReductionOpInfo with inherited arguments
        self.torch_opinfo_name = torch_opinfo_name
        self.torch_opinfo_variant_name = torch_opinfo_variant_name
        # Find and assign the referenced OpInfo from the op_db
        self.torch_opinfo = _find_referenced_opinfo(
            torch_opinfo_name, torch_opinfo_variant_name, op_db=op_db
        )
        # Ensure the referenced opinfo is an instance of ReductionOpInfo
        assert isinstance(self.torch_opinfo, ReductionOpInfo)

        # Inherit constructor arguments from the referenced opinfo
        inherited = self.torch_opinfo._original_reduction_args
        ukwargs = _inherit_constructor_args(name, op, inherited, kwargs)

        # Set view consistency validation to False due to known issue
        self.validate_view_consistency = False

        # Call the parent class constructor with inherited and updated kwargs
        super().__init__(**ukwargs)


class ElementwiseUnaryPythonRefInfo(UnaryUfuncInfo):
    """
    An OpInfo for a Python reference of an elementwise unary operation.
    """
    # 初始化函数，用于创建一个新的对象实例
    def __init__(
        self,
        name,  # 可调用 Python 引用的字符串名称
        *,
        op=None,  # 操作的函数变体，如果为 None，则填充为 torch.<name>
        op_db=None,  # 用于搜索父 opinfo 的 opinfos 数据库
        torch_opinfo_name,  # 对应的 torch opinfo 的字符串名称
        torch_opinfo_variant_name="",  # 对应的 torch opinfo 的变体名称
        validate_view_consistency=True,  # 是否验证视图一致性的布尔标志
        **kwargs,  # 额外的关键字参数，用于覆盖从 torch opinfo 继承的 kwargs
    ):  # 构造函数参数的继承方式
        self.torch_opinfo_name = torch_opinfo_name
        self.torch_opinfo_variant_name = torch_opinfo_variant_name
        # 根据给定的 op_db 查找指定 torch_opinfo_name 和 torch_opinfo_variant_name 的 opinfo
        self.torch_opinfo = _find_referenced_opinfo(
            torch_opinfo_name, torch_opinfo_variant_name, op_db=op_db
        )
        self.validate_view_consistency = validate_view_consistency
        # 断言 self.torch_opinfo 是 UnaryUfuncInfo 类型的实例
        assert isinstance(self.torch_opinfo, UnaryUfuncInfo)

        # 获取继承的构造函数参数
        inherited = self.torch_opinfo._original_unary_ufunc_args
        # 继承构造函数参数，并覆盖由 kwargs 继承的参数
        ukwargs = _inherit_constructor_args(name, op, inherited, kwargs)

        # 调用父类的初始化方法，传入继承和覆盖后的参数 ukwargs
        super().__init__(**ukwargs)
# 定义 ElementwiseBinaryPythonRefInfo 类，继承自 BinaryUfuncInfo 类
class ElementwiseBinaryPythonRefInfo(BinaryUfuncInfo):
    """
    An OpInfo for a Python reference of an elementwise binary operation.
    """

    # 初始化方法，接受多个参数
    def __init__(
        self,
        name,  # 可调用 Python 引用的字符串名称
        *,
        op=None,  # 操作的函数变体，如果为 None，则填充为 torch.<name>
        op_db=None,  # OpInfo 数据库，用于搜索父 OpInfo
        torch_opinfo_name,  # 对应的 torch OpInfo 的字符串名称
        torch_opinfo_variant_name="",  # 对应 torch OpInfo 的变体名称
        **kwargs,  # 其他关键字参数，覆盖从 torch OpInfo 继承的参数
    ):
        # 设置 torch_opinfo_name 属性
        self.torch_opinfo_name = torch_opinfo_name
        # 设置 torch_opinfo_variant_name 属性
        self.torch_opinfo_variant_name = torch_opinfo_variant_name
        # 查找引用的 OpInfo，并赋值给 torch_opinfo 属性
        self.torch_opinfo = _find_referenced_opinfo(
            torch_opinfo_name, torch_opinfo_variant_name, op_db=op_db
        )
        # 断言 torch_opinfo 是 BinaryUfuncInfo 的实例
        assert isinstance(self.torch_opinfo, BinaryUfuncInfo)

        # 继承自 torch_opinfo 的原始二进制函数参数
        inherited = self.torch_opinfo._original_binary_ufunc_args
        # 继承构造函数参数，包括 name, op, 继承自 torch_opinfo 的参数以及当前的 kwargs
        ukwargs = _inherit_constructor_args(name, op, inherited, kwargs)

        # 调用父类 BinaryUfuncInfo 的初始化方法，传入继承的参数
        super().__init__(**ukwargs)
```