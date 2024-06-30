# `D:\src\scipysrc\sympy\sympy\core\tests\test_singleton.py`

```
# 导入 SymPy 库中的基础类 Basic 和有理数类 Rational
from sympy.core.basic import Basic
from sympy.core.numbers import Rational
# 导入 SymPy 中的单例类 Singleton 和常量 S
from sympy.core.singleton import S, Singleton

# 定义测试单例模式的函数
def test_Singleton():
    
    # 定义一个继承自 Basic 类并且使用 Singleton 元类的自定义单例类 MySingleton
    class MySingleton(Basic, metaclass=Singleton):
        pass
    
    MySingleton() # 强制实例化 MySingleton
    # 断言确保 MySingleton 的实例不是 Basic 的实例
    assert MySingleton() is not Basic()
    # 断言确保 MySingleton 的实例是同一个对象
    assert MySingleton() is MySingleton()
    # 断言确保 S 对象中的 MySingleton 与 MySingleton 的实例是同一个对象
    assert S.MySingleton is MySingleton()
    
    # 定义一个继承自 MySingleton 的子类 MySingleton_sub
    class MySingleton_sub(MySingleton):
        pass
    
    MySingleton_sub()
    # 断言确保 MySingleton_sub 的实例不是 MySingleton 的实例
    assert MySingleton_sub() is not MySingleton()
    # 断言确保 MySingleton_sub 的实例是同一个对象
    assert MySingleton_sub() is MySingleton_sub()

# 测试重新定义单例类的情况
def test_singleton_redefinition():
    # 定义一个继承自 Basic 并使用 Singleton 元类的 TestSingleton 类
    class TestSingleton(Basic, metaclass=Singleton):
        pass
    
    # 断言确保 TestSingleton 的实例与 S.TestSingleton 是同一个对象
    assert TestSingleton() is S.TestSingleton
    
    # 重新定义 TestSingleton 类
    class TestSingleton(Basic, metaclass=Singleton):
        pass
    
    # 断言确保重新定义后的 TestSingleton 的实例与 S.TestSingleton 仍然是同一个对象
    assert TestSingleton() is S.TestSingleton

# 测试在命名空间中的单例名称可访问性
def test_names_in_namespace():
    # 每个单例名称都应该能够从 'from sympy import *' 的命名空间中访问，
    # 除了 S 对象之外。不过，它们不一定要使用相同的名称（例如，oo 而不是 S.Infinity）。
    
    # 作为一般规则，只有当某些东西经常被使用时，才应该添加到单例注册表中，
    # 这样代码可以从使用 'is' 的性能优势中受益（这只在非常紧密的循环中才重要），
    # 或者从只有一个实例的内存节省中受益（这对数字单例有影响，但几乎不影响其他内容）。
    # 单例注册表已经有点过度填充，而且不能删除其中的内容，否则会破坏向后兼容性。
    # 因此，如果您通过向单例中添加新内容来到达这里，请问自己是否真的需要将其作为单例化。
    # 注意，SymPy 类彼此比较完全没有问题，因此即使每个 Class() 返回一个新实例，
    # Class() == Class() 也会返回 True。只有在上述注意到的性能收益方面，才需要唯一的实例。
    # 它不应该对任何行为目的需要唯一实例。
    
    # 如果您确定某些东西确实应该是单例的，它必须能够在不使用 'S' 的情况下被 sympify() 访问（因此这个测试）。
    # 此外，它的 str 打印器应该打印一个不使用 S 的形式。这是因为 sympify() 出于安全目的默认禁用属性查找。
    
    d = {}
    # 在命名空间中执行 'from sympy import *'
    exec('from sympy import *', d)
    # 对于 S 对象中的每个名称，包括 S 的属性和 _classes_to_install 列表中的元素
    for name in dir(S) + list(S._classes_to_install):
        # 如果名称以 '_' 开头，则跳过
        if name.startswith('_'):
            continue
        # 如果名称为 'register'，则跳过
        if name == 'register':
            continue
        # 如果 S 对象中该名称对应的属性是 Rational 类型，则跳过
        if isinstance(getattr(S, name), Rational):
            continue
        # 如果 S 对象中该名称对应的属性所在模块以 'sympy.physics' 开头，则跳过
        if getattr(S, name).__module__.startswith('sympy.physics'):
            continue
        # 如果名称在 ['MySingleton', 'MySingleton_sub', 'TestSingleton'] 中，则跳过
        if name in ['MySingleton', 'MySingleton_sub', 'TestSingleton']:
            # 从前面的测试中可以看出，这些名称不需要处理
            continue
        # 如果名称为 'NegativeInfinity'，则跳过
        if name == 'NegativeInfinity':
            # 可以通过 -oo 访问
            continue

        # 使用 'is' 操作符确保它是完全相同的对象
        assert any(getattr(S, name) is i for i in d.values()), name
```