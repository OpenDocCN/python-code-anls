# `.\pytorch\torch\package\_mock.py`

```
# 定义一个列表，包含需要在 MockedObject 中添加的魔术方法名称
_magic_methods = [
    "__subclasscheck__",
    "__hex__",
    "__rmul__",
    "__float__",
    "__idiv__",
    "__setattr__",
    "__div__",
    "__invert__",
    "__nonzero__",
    "__rshift__",
    "__eq__",
    "__pos__",
    "__round__",
    "__rand__",
    "__or__",
    "__complex__",
    "__divmod__",
    "__len__",
    "__reversed__",
    "__copy__",
    "__reduce__",
    "__deepcopy__",
    "__rdivmod__",
    "__rrshift__",
    "__ifloordiv__",
    "__hash__",
    "__iand__",
    "__xor__",
    "__isub__",
    "__oct__",
    "__ceil__",
    "__imod__",
    "__add__",
    "__truediv__",
    "__unicode__",
    "__le__",
    "__delitem__",
    "__sizeof__",
    "__sub__",
    "__ne__",
    "__pow__",
    "__bytes__",
    "__mul__",
    "__itruediv__",
    "__bool__",
    "__iter__",
    "__abs__",
    "__gt__",
    "__iadd__",
    "__enter__",
    "__floordiv__",
    "__call__",
    "__neg__",
    "__and__",
    "__ixor__",
    "__getitem__",
    "__exit__",
    "__cmp__",
    "__getstate__",
    "__index__",
    "__contains__",
    "__floor__",
    "__lt__",
    "__getattr__",
    "__mod__",
    "__trunc__",
    "__delattr__",
    "__instancecheck__",
    "__setitem__",
    "__ipow__",
    "__ilshift__",
    "__long__",
    "__irshift__",
    "__imul__",
    "__lshift__",
    "__dir__",
    "__ge__",
    "__int__",
    "__ior__",
]

# 定义一个 MockedObject 类，用于模拟被包装的对象
class MockedObject:
    _name: str  # 类属性，用于存储对象名称

    def __new__(cls, *args, **kwargs):
        # __new__ 方法用于创建对象实例，在此处进行定制以确保对象的正确创建
        if not kwargs.get("_suppress_err"):
            # 如果没有传入 _suppress_err 参数或其值为 False，则抛出 NotImplementedError
            raise NotImplementedError(
                f"Object '{cls._name}' was mocked out during packaging "
                f"but it is being used in '__new__'. If this error is "
                "happening during 'load_pickle', please ensure that your "
                "pickled object doesn't contain any mocked objects."
            )
        # 否则，允许正常创建对象实例
        return super().__new__(cls)

    def __init__(self, name: str, _suppress_err: bool):
        # __init__ 方法用于初始化对象
        self.__dict__["_name"] = name  # 将传入的 name 参数作为对象的名称存储在 _name 属性中

    def __repr__(self):
        # __repr__ 方法用于返回对象的字符串表示形式
        return f"MockedObject({self._name})"

# 定义一个函数 install_method，用于向 MockedObject 类动态添加魔术方法
def install_method(method_name):
    def _not_implemented(self, *args, **kwargs):
        # _not_implemented 函数作为魔术方法的替代实现，抛出 NotImplementedError
        raise NotImplementedError(
            f"Object '{self._name}' was mocked out during packaging but it is being used in {method_name}"
        )

    # 使用 setattr 将 _not_implemented 函数绑定到 MockedObject 类的 method_name 属性上
    setattr(MockedObject, method_name, _not_implemented)

# 遍历 _magic_methods 列表，依次调用 install_method 函数，为 MockedObject 类添加所有魔术方法
for method_name in _magic_methods:
    install_method(method_name)
```