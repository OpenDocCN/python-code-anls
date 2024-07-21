# `.\pytorch\test\dynamo\test_guard_manager.py`

```
# Owner(s): ["module: dynamo"]

# 导入 functools 和 weakref 模块
import functools
import weakref

# 导入 torch 相关模块
import torch
import torch._dynamo
import torch._dynamo.test_case
from torch._C._dynamo import guards
from torch._dynamo.convert_frame import GlobalStateGuard
from torch.testing._internal.common_utils import set_default_dtype

# 从 guards 中导入几个类和常量
RootGuardManager = guards.RootGuardManager
DictGuardManager = guards.DictGuardManager
DictSubclassGuardManager = guards.DictSubclassGuardManager
GetAttrGuardAccessor = guards.GetAttrGuardAccessor
GetItemGuardAccessor = guards.GetItemGuardAccessor
TypeGuardAccessor = guards.TypeGuardAccessor
TENSOR_ALIASING = guards.TENSOR_ALIASING
install_tensor_aliasing_guard = guards.install_tensor_aliasing_guard
NO_TENSOR_ALIASING = guards.NO_TENSOR_ALIASING
install_no_tensor_aliasing_guard = guards.install_no_tensor_aliasing_guard

# 创建一个张量 x
x = torch.tensor(4)
# 使用 weakref 创建对 x 的弱引用
weakref_x = weakref.ref(x)

# 定义一个枚举值 default_mgr_enum，指定为 GUARD_MANAGER
default_mgr_enum = torch._dynamo.guards.GuardManagerType.GUARD_MANAGER

# 定义一个简单的类 Pair，包含属性 x 和 y
class Pair:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 创建一个全局的 Pair 实例 global_pair，x 是一个张量，y 是整数 1
global_pair = Pair(torch.randn(4), 1)

# 定义函数 id_type，返回参数 x 的类型的 id
def id_type(x):
    return id(type(x))

# 定义函数 equals_match，比较 x 和 expected 是否相等
def equals_match(x, expected):
    return x == expected

# 定义函数 equals_match_verbose_code_parts，返回一个列表，包含字符串形式的 x 和 expected 比较
def equals_match_verbose_code_parts(expected):
    return [f"x == {expected}"]

# 定义函数 ge_match，判断 x 是否大于等于 expected
def ge_match(x, expected):
    return x >= expected

# 定义函数 ge_match_verbose_code_parts，返回一个字符串，描述 x 和 expected 的比较
def ge_match_verbose_code_parts(expected):
    return f"expected >= {expected}"

# 定义函数 less_match，判断 x 是否小于 expected
def less_match(x, expected):
    return x < expected

# 定义函数 less_match_verbose_code_parts，返回一个列表，包含字符串形式的 x 和 expected 比较
def less_match_verbose_code_parts(expected):
    return [f"expected < {expected}"]

# 测试 GuardManager 类的功能，继承自 torch._dynamo.test_case.TestCase
class GuardManagerTests(torch._dynamo.test_case.TestCase):
    # 测试 GlobalStateGuard 类的功能
    def test_global_state_guard(self):
        # 创建 GLOBAL_STATE 类型的 guard 对象
        guard = guards.GLOBAL_STATE(["global_state_check"])
        # 断言 guard(None) 返回 True
        self.assertTrue(guard(None))
        
        # 使用 set_default_dtype 修改默认数据类型为 torch.double
        with set_default_dtype(torch.double):
            # 断言 guard(None) 返回 False
            self.assertFalse(guard(None))
            # 断言 guard.check_verbose(None) 的输出符合预期
            self.assertExpectedInline(
                str(guard.check_verbose(None)),
                """\
GuardDebugInfo(
result=0,
verbose_code_parts=['GLOBAL_STATE changed: default_dtype '],
num_guards_executed=0)
""",
            )
        
        # 再次断言 guard(None) 返回 True
        self.assertTrue(guard(None))
        # 断言 guard.check_verbose(None).result 返回 True
        self.assertTrue(guard.check_verbose(None).result)
        
        # 保存当前 torch.are_deterministic_algorithms_enabled() 的原始值
        _orig = torch.are_deterministic_algorithms_enabled()
        try:
            # 设置 torch.are_deterministic_algorithms_enabled() 为非 _orig 值
            torch.use_deterministic_algorithms(not _orig)
            # 断言 guard(None) 返回 False
            self.assertFalse(guard(None))
            # 断言 guard.check_verbose(None) 的输出符合预期
            self.assertExpectedInline(
                str(guard.check_verbose(None)),
                """\
GuardDebugInfo(
result=0,
verbose_code_parts=['GLOBAL_STATE changed: deterministic_algorithms '],
num_guards_executed=0)
""",
            )
        finally:
            # 恢复 torch.are_deterministic_algorithms_enabled() 的原始值
            torch.use_deterministic_algorithms(_orig)
        
        # 再次断言 guard(None) 返回 True
        self.assertTrue(guard(None))
        # 断言 guard.check_verbose(None).result 返回 True
        self.assertTrue(guard.check_verbose(None).result)

    # 测试 GlobalStateGuard 类的 reason 方法
    def test_global_state_reason(self):
        # 启用梯度计算
        with torch.enable_grad():
            guards = GlobalStateGuard()
        # 禁用梯度计算
        with torch.no_grad():
            # 断言 guards.check() 返回 False
            self.assertIs(guards.check(), False)
            # 断言 guards.reason() 返回 "grad_mode "
            self.assertEqual(guards.reason(), "grad_mode ")
    # 定义一个测试函数，用于测试 Python Lambda 的 Leaf Guard
    def test_python_lambda_leaf_guard(self):
        # 创建一个常量 guard，用于检查是否等于 5 的 Lambda 函数
        const_guard = guards.LAMBDA_GUARD(
            functools.partial(equals_match, expected=5),
            equals_match_verbose_code_parts(5),
        )
        # 断言常量 guard 对于输入 5 返回 True
        self.assertTrue(const_guard(5))
        # 断言常量 guard 对于输入 4 返回 False
        self.assertFalse(const_guard(4))
        # 断言常量 guard 对于输入 "foo" 返回 False
        self.assertFalse(const_guard("foo"))

    # 定义一个测试函数，用于测试类型匹配的 guard
    def test_type_guard(self):
        # 初始化 foo 为整数 4
        foo = 4
        # 创建一个类型匹配的 guard，检查输入是否为整数
        guard = guards.TYPE_MATCH(id_type(foo), ["type(x) == int"])

        # 断言 guard 对于输入 5 返回 True
        self.assertTrue(guard(5))
        # 断言 guard 对于输入 4 返回 True
        self.assertTrue(guard(4))
        # 断言 guard 对于输入 "foo" 返回 False
        self.assertFalse(guard("foo"))

        # 将 foo 更新为字典 {"a": 1}
        foo = {"a": 1}
        # 更新 guard 为检查输入是否为字典的类型匹配 guard
        guard = guards.TYPE_MATCH(id_type(foo), ["type(x) == dict"])
        # 断言 guard 对于输入 foo 返回 True
        self.assertTrue(guard(foo))
        # 断言 guard 对于输入空字典返回 True
        self.assertTrue(guard({}))
        # 断言 guard 对于输入整数 5 返回 False
        self.assertFalse(guard(5))
        # 断言 guard 对于输入字符串 "foo" 返回 False
        self.assertFalse(guard("foo"))

        # 定义一个类 Foo，包含属性 x 和 y 的初始化函数
        class Foo:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        # 创建一个 Foo 类的实例 foo，传入参数 1 和 2
        foo = Foo(1, 2)

        # 更新 guard 为检查输入是否为 Foo 类型的 guard
        guard = guards.TYPE_MATCH(id_type(foo), ["type(x) == Foo"])
        # 断言 guard 对于输入 foo 返回 True
        self.assertTrue(guard(foo))
        # 断言 guard 对于输入空字典返回 False
        self.assertFalse(guard({}))
        # 断言 guard 对于输入整数 5 返回 False
        self.assertFalse(guard(5))
        # 断言 guard 对于输入字符串 "foo" 返回 False
        self.assertFalse(guard("foo"))

    # 定义一个测试函数，用于测试 id 匹配的 guard
    def test_id_guard(self):
        # 初始化 foo 为整数 4
        foo = 4
        # 创建一个 id 匹配的 guard，检查输入的 id 是否等于 foo 的 id
        guard = guards.ID_MATCH(id(foo), ["id(x) == id(foo)"])

        # 断言 guard 对于输入 foo 返回 True
        self.assertTrue(guard(foo))
        # 断言 guard 对于输入整数 5 返回 False
        self.assertFalse(guard(5))
        # 断言 guard 对于输入字符串 "foo" 返回 False
        self.assertFalse(guard("foo"))

        # 将 foo 更新为字典 {"a": 1}
        foo = {"a": 1}
        # 更新 guard 为检查输入的 id 是否等于 foo 的 id 的 guard
        guard = guards.ID_MATCH(id(foo), ["id(x) == id(foo)"])
        # 断言 guard 对于输入 foo 返回 True
        self.assertTrue(guard(foo))
        # 断言 guard 对于输入另一个具有相同内容的字典返回 False
        self.assertFalse(guard({"a": 1}))
        # 断言 guard 对于输入空字典返回 False
        self.assertFalse(guard({}))
        # 断言 guard 对于输入整数 5 返回 False
        self.assertFalse(guard(5))

    # 定义一个测试函数，用于测试相等匹配的 guard
    def test_equals_guard(self):
        # 初始化 foo 为整数 4
        foo = 4
        # 创建一个相等匹配的 guard，检查输入是否等于 4
        guard = guards.EQUALS_MATCH(foo, ["x == 4"])

        # 断言 guard 对于输入 4 返回 True
        self.assertTrue(guard(4))
        # 断言 guard 对于输入 5 返回 False
        self.assertFalse(guard(5))
        # 断言 guard 对于输入字符串 "foo" 返回 False
        self.assertFalse(guard("foo"))

        # 初始化 foo 为元组 (1, 2, 3)
        foo = (1, 2, 3)
        # 更新 guard 为检查输入是否等于 foo 的 guard
        guard = guards.EQUALS_MATCH(foo, ["x == foo"])
        # 断言 guard 对于输入 foo 返回 True
        self.assertTrue(guard(foo))
        # 断言 guard 对于输入具有相同内容的元组返回 True
        self.assertTrue(guard((1, 2, 3)))
        # 断言 guard 对于输入具有不同内容的元组返回 False
        self.assertFalse(guard((1, 2, 3, 4)))
        # 断言 guard 对于输入空字典返回 False
        self.assertFalse(guard({}))

        # 初始化 foo 为列表 [1, 2, 3]
        foo = [1, 2, 3]
        # 更新 guard 为检查输入是否等于 foo 的 guard
        guard = guards.EQUALS_MATCH(foo, ["x == foo"])
        # 断言 guard 对于输入 foo 返回 True
        self.assertTrue(guard(foo))
        # 断言 guard 对于输入具有相同内容的列表返回 True
        self.assertTrue(guard([1, 2, 3]))
        # 断言 guard 对于输入具有不同内容的列表返回 False
        self.assertFalse(guard([1, 2, 3, 4]))

        # 初始化 foo 为整型类型 int
        foo = int
        # 更新 guard 为检查输入是否等于 foo 的 guard
        guard = guards.EQUALS_MATCH(foo, ["x == foo"])
        # 断言 guard 对于输入 foo 返回 True
        self.assertTrue(guard(foo))
        # 断言 guard 对于输入整型类型 int 返回 True
        self.assertTrue(guard(int))
        # 断言 guard 对于输入浮点型类型 float 返回 False
        self.assertFalse(guard(float))

    # 定义一个测试函数，用于测试默认设备的 guard
    def test_default_device_guard(self):
        # 初始化 foo 为整数 1
        foo = 1
        # 创建一个默认设备的 guard，检查输入是否在 "cpu device" 中
        guard = guards.DEFAULT_DEVICE(["cpu device"])
        # 断言 guard 对于输入 foo 返回 True
        self.assertTrue(guard(foo))

        try:
            # 尝试将默认设备设置为 "cuda"
            torch.set_default_device("cuda")
            # 断言 guard 对于输入 foo 返回 False
            self.assertFalse(guard(foo))
        finally:
            # 最终将默认设备设置为 None
            torch.set_default_device(None)
    # 定义一个测试函数，用于验证 DATA_PTR_MATCH 守卫的功能
    def test_data_ptr_match_guard(self):
        # 创建一个 PyTorch 张量对象
        foo = torch.tensor([1, 2, 3])
        # 使用 DATA_PTR_MATCH 守卫创建一个保护对象，规定了一个匹配条件
        guard = guards.DATA_PTR_MATCH(foo, ["x.data_ptr() == foo.data_ptr()"])
        # 断言守卫函数能够成功通过参数 foo
        self.assertTrue(guard(foo))
        # 断言守卫函数不能通过不同的张量对象
        self.assertFalse(guard(torch.tensor([1, 2, 3])))

    # 定义一个测试函数，用于验证 LENGTH_CHECK 守卫的功能
    def test_length_check_guard(self):
        # 创建一个列表对象
        foo = [1, 2, 3]
        # 使用 LENGTH_CHECK 守卫创建一个保护对象，规定了一个长度匹配条件
        guard = guards.LENGTH_CHECK(len(foo), ["len(x) == len(foo)"])
        # 断言守卫函数能够成功通过参数 foo
        self.assertTrue(guard(foo))
        # 断言守卫函数不能通过空列表
        self.assertFalse(guard([]))

    # 定义一个测试函数，用于验证 NO_HASATTR 守卫的功能
    def test_no_hasattr_guard(self):
        # 定义一个包含属性的类
        class Bar:
            def __init__(self):
                self.bar = 2
        # 创建一个类实例
        bar = Bar()
        # 定义另一个包含属性的类
        class Foo:
            def __init__(self):
                self.foo = 2
        # 创建另一个类实例
        foo = Foo()
        # 使用 NO_HASATTR 守卫创建一个保护对象，规定了一个不存在属性的条件
        guard = guards.NO_HASATTR("foo", ["hasattr(x, 'foo') == False"])
        # 断言守卫函数能够成功通过参数 bar
        self.assertTrue(guard(bar))
        # 断言守卫函数不能通过参数 foo
        self.assertFalse(guard(foo))

    # 定义一个测试函数，用于验证张量别名守卫的功能
    def test_tensor_aliasing_guard(self):
        # 创建一个根守卫管理器对象
        guard_manager = RootGuardManager()
        # 创建一个 PyTorch 张量对象 a
        a = torch.randn(3, 4)
        # 定义一个包含两个张量属性的类
        class Foo:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        # 创建一个 Foo 类实例，两个属性值都是张量 a
        f_locals = Foo(a, a)
        # 获取 x 属性的守卫管理器
        x_guard_mgr = guard_manager.getattr_manager("x", "", a, default_mgr_enum)
        # 获取 y 属性的守卫管理器
        y_guard_mgr = guard_manager.getattr_manager("y", "", a, default_mgr_enum)
        # 安装张量别名守卫，规定了一个别名条件
        install_tensor_aliasing_guard(x_guard_mgr, y_guard_mgr, ["x is y"])

        # 检查结构
        x_guards = x_guard_mgr.get_leaf_guards()
        y_guards = y_guard_mgr.get_leaf_guards()
        self.assertEqual(len(x_guards), 1)
        self.assertEqual(len(y_guards), 1)
        self.assertTrue(isinstance(x_guards[0], TENSOR_ALIASING))
        self.assertTrue(isinstance(y_guards[0], TENSOR_ALIASING))
        # 检查两个守卫对象是否相同
        self.assertTrue(x_guards[0] is y_guards[0])

        # 创建一个新的 Foo 类实例，其属性值都是不同的张量对象
        f_locals_unaliased = Foo(torch.randn(3, 4), torch.randn(3, 4))
        # 再次检查守卫管理器的状态
        self.assertEqual(len(x_guard_mgr.get_leaf_guards()), 1)
        self.assertEqual(len(y_guard_mgr.get_leaf_guards()), 1)
        # 使用根守卫管理器检查 f_locals 是否符合所有的守卫条件
        self.assertTrue(guard_manager.check(f_locals))
        # 使用根守卫管理器检查 f_locals_unaliased 是否符合所有的守卫条件
        self.assertFalse(guard_manager.check(f_locals_unaliased))

    # 定义一个测试函数，用于验证 DICT_VERSION 守卫的功能
    def test_dict_version_guard(self):
        # 创建一个字典对象
        foo = {"a": 1, "b": 2}
        # 使用 DICT_VERSION 守卫创建一个保护对象，规定了一个版本匹配条件
        guard = guards.DICT_VERSION(foo, ["x.version == foo.version"])

        # 断言守卫函数能够成功通过参数 foo
        self.assertTrue(guard(foo))
        # 断言守卫函数不能通过相同内容的新字典对象
        self.assertFalse(guard(dict(foo)))
        # 修改原字典内容
        foo["a"] = 2
        # 断言守卫函数不能通过修改后的原字典
        self.assertFalse(guard(foo))
        # 断言守卫函数不能通过一个空字典
        self.assertFalse(guard({"a": 1, "b": 2}))
        # 断言守卫函数不能通过一个完全空的字典
        self.assertFalse(guard({}))
    # 定义测试动态索引守卫的方法
    def test_dynamic_indices_guard(self):
        # 创建两个动态索引守卫对象，一个空集合，一个包含索引0和1，对应的断言条件为"x.size(0) == y.size(0)"
        guard1 = guards.DYNAMIC_INDICES(set(), ["x.size(0) == y.size(0)"])
        guard2 = guards.DYNAMIC_INDICES(set({0, 1}), ["x.size(0) == y.size(0)"])

        # 生成一个形状为(4,)的随机张量x，并分别通过两个守卫进行断言
        x = torch.randn(4)
        self.assertTrue(guard1(x))
        self.assertTrue(guard2(x))

        # 设置x的动态索引为{0}，此时guard1返回False，guard2仍然返回True
        x._dynamo_dynamic_indices = set({0})
        self.assertFalse(guard1(x))
        self.assertTrue(guard2(x))

        # 设置x的动态索引为{2}，此时两个守卫均返回False
        x._dynamo_dynamic_indices = set({2})
        self.assertFalse(guard1(x))
        self.assertFalse(guard2(x))

    # 定义测试张量匹配守卫的方法
    def test_tensor_match_guard(self):
        # 创建根守卫管理器对象
        guard_manager = RootGuardManager()

        # 生成一个形状为(4, 4)的随机张量x，并获取其尺寸和步长
        x = torch.randn(4, 4)
        size = list(x.size())
        stride = list(x.stride())

        # 向守卫管理器添加张量匹配守卫，断言条件为"check_tensor(x)"
        guard_manager.add_tensor_match_guard(x, size, stride, "x", ["check_tensor(x)"])

        # 使用守卫管理器检查张量x，断言检查通过
        self.assertTrue(guard_manager.check(x))
        # 使用守卫管理器详细检查张量x，断言检查通过
        self.assertTrue(guard_manager.check_verbose(x).result)
        # 使用守卫管理器检查一个新的随机张量，断言检查通过
        self.assertTrue(guard_manager.check(torch.randn(4, 4)))
        # 使用守卫管理器详细检查一个新的随机张量，断言检查通过
        self.assertTrue(guard_manager.check_verbose(torch.randn(4, 4)).result)
        # 对x进行转置操作后，使用守卫管理器检查，断言检查不通过
        self.assertFalse(guard_manager.check(x.t_()))

        # 再次生成一个形状为(4, 4)的随机张量x，并对其进行转置操作
        x = torch.randn(4, 4)
        x.t_()

        # 使用守卫管理器详细检查转置后的张量x，并打印详细信息的第一个部分
        debug_info = guard_manager.check_verbose(x)
        print(debug_info.verbose_code_parts[0])

        # 断言详细信息的第一个部分包含"tensor 'x' stride mismatch"
        self.assertTrue(
            "tensor 'x' stride mismatch" in debug_info.verbose_code_parts[0]
        )
    # 定义一个测试方法，用于测试不允许张量别名的保护机制
    def test_no_tensor_aliasing_guard(self):
        # 创建根保护管理器的实例
        guard_manager = RootGuardManager()

        # 创建一个形状为 (3, 4) 的随机张量 a
        a = torch.randn(3, 4)

        # 定义一个名为 Foo 的类，初始化方法接收 x, y, z 三个参数并赋值给实例变量
        class Foo:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        # 创建 Foo 类的实例 f_locals，并传入三个相同的张量 a
        f_locals = Foo(a, a, a)

        # 获取属性管理器 x_guard_mgr, y_guard_mgr, z_guard_mgr，用于张量 a 的三个属性 x, y, z
        x_guard_mgr = guard_manager.getattr_manager("x", "", a, default_mgr_enum)
        y_guard_mgr = guard_manager.getattr_manager("y", "", a, default_mgr_enum)
        z_guard_mgr = guard_manager.getattr_manager("z", "", a, default_mgr_enum)

        # 安装不允许张量别名的保护，传入属性管理器列表、属性名称列表和保护描述列表
        install_no_tensor_aliasing_guard(
            [x_guard_mgr, y_guard_mgr, z_guard_mgr],
            ["x", "y", "z"],
            ["no_aliasing(x, y, z)"],
        )

        # 检查结构
        x_guards = x_guard_mgr.get_leaf_guards()
        y_guards = y_guard_mgr.get_leaf_guards()
        z_guards = z_guard_mgr.get_leaf_guards()

        # 断言每个属性的保护只有一个
        self.assertEqual(len(x_guards), 1)
        self.assertEqual(len(y_guards), 1)
        self.assertEqual(len(z_guards), 1)

        # 断言每个保护都是 NO_TENSOR_ALIASING 类型的实例
        self.assertTrue(isinstance(x_guards[0], NO_TENSOR_ALIASING))
        self.assertTrue(isinstance(y_guards[0], NO_TENSOR_ALIASING))
        self.assertTrue(isinstance(z_guards[0], NO_TENSOR_ALIASING))

        # 检查三个属性的保护对象是同一个实例
        self.assertTrue(x_guards[0] is y_guards[0] is z_guards[0])

        # 检查 guard_manager 对于 f_locals 的检查结果为 False
        self.assertFalse(guard_manager.check(f_locals))

        # 检查 guard_manager 对于 f_locals 的详细检查结果为 False
        self.assertFalse(guard_manager.check_verbose(f_locals).result)

        # 创建一个新的 Foo 类的实例 f_locals_unaliased，传入三个不同的随机张量
        f_locals_unaliased = Foo(
            torch.randn(3, 4),
            torch.randn(3, 4),
            torch.randn(3, 4),
        )

        # 检查 guard_manager 对于 f_locals_unaliased 的检查结果为 True
        self.assertTrue(guard_manager.check(f_locals_unaliased))

        # 检查 guard_manager 对于 f_locals_unaliased 的详细检查结果的结果字段为 True
        self.assertTrue(guard_manager.check_verbose(f_locals_unaliased).result)

        # 再次检查 guard_manager 对于 f_locals_unaliased 的检查结果为 True
        self.assertTrue(guard_manager.check(f_locals_unaliased))

        # 创建一个新的 Foo 类的实例 f_locals_unaliased，传入两个张量 a 和一个新的随机张量
        f_locals_unaliased = Foo(
            a,
            torch.randn(3, 4),
            a,
        )

        # 检查 guard_manager 对于 f_locals_unaliased 的检查结果为 False
        self.assertFalse(guard_manager.check(f_locals_unaliased))

        # 检查 guard_manager 对于 f_locals_unaliased 的详细检查结果的结果字段为 False
        self.assertFalse(guard_manager.check_verbose(f_locals_unaliased).result)

    # 定义一个测试方法，用于测试弱引用是否存活的保护机制
    def test_weakref_alive_guard(self):
        # 创建一个形状为 (3, 4) 的随机张量 x
        x = torch.rand(3, 4)

        # 创建 x 的弱引用 weakref_x
        weakref_x = weakref.ref(x)

        # 创建一个 NOT_NONE 的保护对象 guard，检查表达式 "weakref_x is not None"
        guard = guards.NOT_NONE(["weakref_x is not None"])

        # 断言 guard 对于 weakref_x 的检查结果为 True
        self.assertTrue(guard(weakref_x()))

        # 删除引用 x
        del x

        # 断言 guard 对于 weakref_x 的检查结果为 False
        self.assertFalse(guard(weakref_x()))
    def test_guard_manager_leaf_guard(self):
        # 创建一个 RootGuardManager 实例
        guard_manager = RootGuardManager()
        # 添加一个类型匹配的守卫，要求 id_type(5) 的类型为 int
        guard_manager.add_type_match_guard(id_type(5), ["type(x) == int"])
        # 添加一个基于 lambda 函数的守卫，使用 functools.partial 设置了 ge_match(expected=5) 函数
        guard_manager.add_lambda_guard(
            functools.partial(ge_match, expected=5),
            ge_match_verbose_code_parts(expected=5),
        )
        # 添加一个基于 lambda 函数的守卫，使用 functools.partial 设置了 less_match(expected=10) 函数
        guard_manager.add_lambda_guard(
            functools.partial(less_match, expected=10),
            less_match_verbose_code_parts(expected=10),
        )
        # 断言 Leaf Guards 的数量为 3
        self.assertEqual(len(guard_manager.get_leaf_guards()), 3)
        # 断言 Accessors 的数量为 0
        self.assertEqual(len(guard_manager.get_accessors()), 0)
        # 验证 guard_manager.check(6) 返回 True
        self.assertTrue(guard_manager.check(6))
        # 验证 guard_manager.check(4) 返回 False
        self.assertFalse(guard_manager.check(4))
        # 验证 guard_manager.check("foo") 返回 False
        self.assertFalse(guard_manager.check("foo"))

    def test_attr_guard_manager(self):
        # 定义一个名为 Foo 的类
        class Foo:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        # 创建一个 Foo 类的实例 foo，参数为 (1, 2)
        foo = Foo(1, 2)
        # 创建一个 RootGuardManager 实例
        guard_manager = RootGuardManager()
        # 添加一个类型匹配的守卫，要求 id_type(foo) 的类型为 Foo
        guard_manager.add_type_match_guard(id_type(foo), ["type(x) == Foo"])
        # 获取名为 "x" 的属性管理器，设置了一个基于 lambda 函数的守卫，要求值等于 foo.x
        guard_manager.getattr_manager("x", "x", 1, default_mgr_enum).add_lambda_guard(
            functools.partial(equals_match, expected=foo.x),
            equals_match_verbose_code_parts(foo.x),
        )
        # 获取名为 "y" 的属性管理器，设置了一个基于 lambda 函数的守卫，要求值等于 foo.y
        guard_manager.getattr_manager("y", "y", 2, default_mgr_enum).add_lambda_guard(
            functools.partial(equals_match, expected=foo.y),
            equals_match_verbose_code_parts(foo.y),
        )
        # 断言 Leaf Guards 的数量为 1
        self.assertEqual(len(guard_manager.get_leaf_guards()), 1)
        # 断言 Accessors 的数量为 2，分别对应属性 "x" 和 "y"
        self.assertEqual(len(guard_manager.get_accessors()), 2)
        # 验证第一个 Accessor 是 GetAttrGuardAccessor 类的实例
        self.assertTrue(
            isinstance(guard_manager.get_accessors()[0], GetAttrGuardAccessor)
        )
        # 验证第二个 Accessor 是 GetAttrGuardAccessor 类的实例
        self.assertTrue(
            isinstance(guard_manager.get_accessors()[1], GetAttrGuardAccessor)
        )
        # 验证对 "x" 属性的 Leaf Guards 数量为 1
        self.assertEqual(
            len(
                guard_manager.getattr_manager(
                    attr="x",
                    source="x",
                    example_value=None,
                    guard_manager_enum=default_mgr_enum,
                ).get_leaf_guards()
            ),
            1,
        )
        # 验证对 "y" 属性的 Leaf Guards 数量为 1
        self.assertEqual(
            len(
                guard_manager.getattr_manager(
                    "y", "y", None, default_mgr_enum
                ).get_leaf_guards()
            ),
            1,
        )
        # 验证 guard_manager.check(foo) 返回 True
        self.assertTrue(guard_manager.check(foo))
        # 验证 guard_manager.check(Foo(3, 4)) 返回 False
        self.assertFalse(guard_manager.check(Foo(3, 4)))
        # 验证 guard_manager.check("foo") 返回 False
        self.assertFalse(guard_manager.check("foo"))
    def test_item_guard_manager(self):
        # 创建一个列表 foo，包含元素 [1, 2]
        foo = [1, 2]
        # 创建一个 RootGuardManager 实例 guard_manager
        guard_manager = RootGuardManager()
        # 向 guard_manager 添加类型匹配的保护条件，针对 foo 的类型为 Foo
        guard_manager.add_type_match_guard(id_type(foo), ["type(x) == Foo"])
        # 获取 foo[0] 的管理器，并添加 lambda 函数作为保护条件，确保其等于 foo[0]
        guard_manager.getitem_manager(0, "", 1, default_mgr_enum).add_lambda_guard(
            functools.partial(equals_match, expected=foo[0]),
            equals_match_verbose_code_parts(foo[0]),
        )
        # 获取 foo[1] 的管理器，并添加 lambda 函数作为保护条件，确保其等于 foo[1]
        guard_manager.getitem_manager(1, "", 2, default_mgr_enum).add_lambda_guard(
            functools.partial(equals_match, expected=foo[1]),
            equals_match_verbose_code_parts(foo[1]),
        )
        # 断言 leaf guards 的数量为 1
        self.assertEqual(len(guard_manager.get_leaf_guards()), 1)
        # 断言 accessors 的数量为 2，分别对应 foo 的两个元素
        # 2 child managers, one for x and one for y
        self.assertEqual(len(guard_manager.get_accessors()), 2)
        # 断言第一个 accessor 是 GetItemGuardAccessor 类的实例
        self.assertTrue(
            isinstance(guard_manager.get_accessors()[0], GetItemGuardAccessor)
        )
        # 断言第二个 accessor 是 GetItemGuardAccessor 类的实例
        self.assertTrue(
            isinstance(guard_manager.get_accessors()[1], GetItemGuardAccessor)
        )
        # 检查子管理器上的 leaf guards
        # 检查索引为 0 的 getitem_manager 上的 leaf guards 数量为 1
        self.assertEqual(
            len(
                guard_manager.getitem_manager(
                    0, "", None, default_mgr_enum
                ).get_leaf_guards()
            ),
            1,
        )
        # 检查索引为 1 的 getitem_manager 上的 leaf guards 数量为 1
        self.assertEqual(
            len(
                guard_manager.getitem_manager(
                    1, "", None, default_mgr_enum
                ).get_leaf_guards()
            ),
            1,
        )

        # 断言 guard_manager 能通过 foo
        self.assertTrue(guard_manager.check(foo))
        # 断言 guard_manager 不能通过 [3, 4]
        self.assertFalse(guard_manager.check([3, 4]))
        # 断言 guard_manager 不能通过字符串 "foo"
        self.assertFalse(guard_manager.check("foo"))

    def test_dict_getitem_accessor(self):
        # 创建一个字典 foo，包含键值对 {"a": 1, "b": 2}
        foo = {
            "a": 1,
            "b": 2,
        }
        # 创建一个 RootGuardManager 实例 guards_manager
        guards_manager = RootGuardManager()
        # 向 guards_manager 添加类型匹配的保护条件，针对 foo 的类型为 Foo
        guards_manager.add_type_match_guard(id_type(foo), ["type(x) == Foo"])
        # 获取 key 为 "a" 的 dict_getitem_manager，并添加等值匹配的保护条件，确保其等于 1
        guards_manager.dict_getitem_manager(
            "a", "", 1, default_mgr_enum
        ).add_equals_match_guard(1, ["a == 1"])
        # 获取 key 为 "b" 的 dict_getitem_manager，并添加等值匹配的保护条件，确保其等于 2
        guards_manager.dict_getitem_manager(
            "b", "", 2, default_mgr_enum
        ).add_equals_match_guard(2, ["b == 2"])

        # 断言 guards_manager 能通过 foo
        self.assertTrue(guards_manager.check(foo))
        # 断言 guards_manager 不能通过 {"a": 1, "b": 3}
        self.assertFalse(guards_manager.check({"a": 1, "b": 3}))

    def test_globals(self):
        # 声明全局变量 global_pair 和 Pair
        global global_pair, Pair
        # 创建一个 RootGuardManager 实例 guard_manager
        guard_manager = RootGuardManager()
        # 获取 globals() 的 dict_manager，并获取 global_pair 的 getitem_manager，并将 global_pair 作为参数传入
        gpair_mgr = guard_manager.globals_dict_manager(
            globals(), "", None, default_mgr_enum
        ).getitem_manager("global_pair", "", global_pair, default_mgr_enum)
        # 在 gpair_mgr 上添加 lambda 函数作为保护条件，确保 global_pair 是 Pair 类的实例，
        # 并且 global_pair.x 是 torch.Tensor 类的实例，global_pair.y 是 int 类型
        gpair_mgr.add_lambda_guard(
            lambda x: isinstance(x, Pair)
            and isinstance(x.x, torch.Tensor)
            and isinstance(x.y, int),
            "global guard fail",
        )

        # 断言 guard_manager 能通过 global_pair
        self.assertTrue(guard_manager.check(global_pair))
        # 修改 global_pair 的 y 属性为 "foo"
        global_pair.y = "foo"
        # 断言 guard_manager 不能通过 global_pair
        self.assertFalse(guard_manager.check(global_pair))
    # 测试类型管理器的功能
    def test_type_manager(self):
        # 创建 RootGuardManager 的实例
        guard_manager = RootGuardManager()

        # 定义类 A
        class A:
            a = 4

        # 定义类 B，继承自 A，并添加方法 mul
        class B(A):
            def mul(self, x):
                # 调用父类 A 中的 mul 方法
                super().mul(x)

        # 创建类 B 的实例 foo
        foo = B()
        # 准备局部变量字典，包含 foo
        f_locals = {"foo": foo}

        # 使用 guard_manager 获取 foo 对象的管理器 foo_mgr
        foo_mgr = guard_manager.getitem_manager("foo", "", foo, default_mgr_enum)
        # 获取 foo 对象的类型管理器 type_manager
        type_manager = foo_mgr.type_manager("", type(foo), default_mgr_enum)
        # 断言 foo_mgr 的第一个访问器是 TypeGuardAccessor 类的实例
        self.assertTrue(isinstance(foo_mgr.get_accessors()[0], TypeGuardAccessor))
        
        # 获取 type_manager 中 __mro__ 属性的管理器 mro_manager
        mro_manager = type_manager.getattr_manager(
            "__mro__", "", type(foo).__mro__, default_mgr_enum
        )
        # 断言 type_manager 的第一个访问器是 GetAttrGuardAccessor 类的实例
        self.assertTrue(
            isinstance(type_manager.get_accessors()[0], GetAttrGuardAccessor)
        )
        
        # 给 mro_manager 添加长度检查的保护，期望 __mro__ 的长度为 3
        mro_manager.add_length_check_guard(
            3,
            "Expected len(type(foo).__mro__) == 3",
        )

        # 获取 mro_manager 中索引为 1 的项的管理器 item_manager
        item_manager = mro_manager.getitem_manager(
            1, "", type(foo).__mro__[1], default_mgr_enum
        )
        # 断言 mro_manager 的第一个访问器是 GetItemGuardAccessor 类的实例
        self.assertTrue(
            isinstance(mro_manager.get_accessors()[0], GetItemGuardAccessor)
        )
        
        # 获取 item_manager 中属性名为 "a" 的属性管理器 attr_manager
        attr_manager = item_manager.getattr_manager(
            "a", "", type(foo).__mro__[0].a, default_mgr_enum
        )
        # 断言 item_manager 的第一个访问器是 GetAttrGuardAccessor 类的实例
        self.assertTrue(
            isinstance(item_manager.get_accessors()[0], GetAttrGuardAccessor)
        )
        
        # 给 attr_manager 添加 lambda 表达式检查的保护，期望属性值为 4
        attr_manager.add_lambda_guard(
            lambda x: x == 4,
            "Expected value 4",
        )

        # 最终检查 guard_manager 是否通过所有保护条件
        self.assertTrue(guard_manager.check(f_locals))

    # 测试元组迭代器的索引访问功能
    def test_tuple_iterator_getitem(self):
        # 创建元组 a
        a = (1, 2, 3, 4, 5, 6)
        # 创建元组迭代器 foo
        foo = iter(a)
        # 获取 foo 的下一个元素，使 foo 指向索引 1

        # 创建 RootGuardManager 的实例
        guard_manager = RootGuardManager()
        
        # 给 guard_manager 添加元组迭代器长度检查的保护，期望长度为 5
        guard_manager.add_tuple_iterator_length_guard(
            5, id_type(iter(tuple())), ["len == 5"]
        )
        
        # 给 guard_manager 添加 tuple_iterator_getitem 的管理器，检查索引为 2 的项
        guard_manager.tuple_iterator_getitem_manager(
            2, "", foo, default_mgr_enum
        ).add_equals_match_guard(a[3], ["x==4"])

        # 断言类型匹配失败
        self.assertFalse(guard_manager.check(False))

        # 最终检查 guard_manager 是否通过所有保护条件
        self.assertTrue(guard_manager.check(foo))

        # 准备元组 b
        b = (1, 2)
        # 创建元组迭代器 b_foo
        b_foo = iter(b)
        # 检查索引错误是否优雅地失败
        self.assertFalse(guard_manager.check(b_foo))

    # 测试全局弱引用管理器的功能
    def test_global_weakref(self):
        # 创建 RootGuardManager 的实例
        guard_manager = RootGuardManager()
        # 获取全局字典的管理器 globals_manager
        globals_manager = guard_manager.globals_dict_manager(
            globals(), "", None, default_mgr_enum
        )
        # 获取全局弱引用的管理器 weakref_manager
        weakref_manager = globals_manager.global_weakref_manager(
            "weakref_x", "", None, default_mgr_enum
        )

        # 给 weakref_manager 添加 lambda 表达式检查的保护，期望 x 是 torch.Tensor 类型
        weakref_manager.add_lambda_guard(
            lambda x: isinstance(x, torch.Tensor),
            "global weakref fail",
        )

        # 最终检查 guard_manager 是否通过所有保护条件
        self.assertTrue(guard_manager.check(None))
        
        # 删除全局变量 x
        global x
        del x
        # 最终检查 guard_manager 是否通过所有保护条件
        self.assertFalse(guard_manager.check(None))
    def test_lambda_manager(self):
        # 定义一个元组
        a = (1, 1, 3, 4, 5, 6)

        # 创建 RootGuardManager 实例
        guard_manager = RootGuardManager()

        # 检查我们能否使用相同的访问器
        # 使用 lambda 表达式创建 foo_mgr，提取元组中第三个元素作为访问目标
        foo_mgr = guard_manager.lambda_manager(
            lambda x: x[2], "", None, default_mgr_enum
        )
        # 添加 lambda 守卫，确保 x 的值为 3
        foo_mgr.add_lambda_guard(
            lambda x: x == 3,
            "Expected value 3",
        )
        # 断言 guard_manager 能够通过元组 a 的验证
        self.assertTrue(guard_manager.check(a))

        # 测试异常情况
        guard_manager = RootGuardManager()

        # 定义一个会抛出异常的函数 fn
        def fn(x):
            raise AssertionError("Test")
            return x

        # 使用 fn 函数创建 foo_mgr
        foo_mgr = guard_manager.lambda_manager(fn, "", None, default_mgr_enum)

        # 断言 guard_manager 不能通过 None 的验证
        self.assertFalse(guard_manager.check(None))
        # 检查详细信息以确保异常被捕获
        debug_info = guard_manager.check_verbose(None)
        self.assertFalse(debug_info.result)
        self.assertTrue("Test" in debug_info.verbose_code_parts[0])

    def test_dict_contains_guard(self):
        # 创建一个字典 foo
        foo = {"a": 1, "b": 2}
        # 创建一个包含 "a" 键的 DICT_CONTAINS 守卫
        guard = guards.DICT_CONTAINS(True, "a", ["has a"])

        # 断言 guard 能够通过 foo 的验证
        self.assertTrue(guard(foo))
        self.assertTrue(guard({"a": 1, "b": 2}))
        self.assertFalse(guard({"b": 2, "c": 3}))
        self.assertFalse(guard({}))

        # 创建一个不包含 "c" 键的 DICT_CONTAINS 守卫
        guard = guards.DICT_CONTAINS(False, "c", ["not has c"])
        self.assertTrue(guard(foo))
        self.assertTrue(guard({"a": 1, "b": 2}))
        self.assertFalse(guard({"b": 2, "c": 3}))
        self.assertTrue(guard({}))
    def test_dict_guard_manager(self):
        # 创建一个 RootGuardManager 实例
        root = RootGuardManager()

        # 定义一个空函数 nothing
        def nothing():
            pass

        # 初始化 f_locals 字典，包含不同类型的值
        f_locals = {
            "d": {"a": 1, nothing: {"z": 3}, 100: torch.randn(4)},
        }

        # 获取 "d" 的 getitem_manager，期望其子 GuardManager 为 DictGuardManager 类型
        dict_mgr = root.getitem_manager(
            "d",
            "",
            f_locals["d"],
            torch._dynamo.guards.GuardManagerType.DICT_GUARD_MANAGER,
        )
        self.assertTrue(isinstance(dict_mgr, DictGuardManager))

        # 检查 root 对象是否通过检查 f_locals
        self.assertTrue(root.check(f_locals))

        # 检查是否能够添加 leaf guard，预期抛出 RuntimeError
        with self.assertRaises(RuntimeError):
            dict_mgr.add_id_match_guard(id_type(f_locals), "id match")

        # 检查是否能够添加任意访问器，预期抛出 RuntimeError
        with self.assertRaises(RuntimeError):
            dict_mgr.getitem_manager("a", "", f_locals["d"]["a"])

        # 检查不同长度的字典是否会导致检查失败
        f_locals_prime = {
            "d": {"a": 1, "b": 2},
        }
        self.assertFalse(root.check(f_locals_prime))

        # 添加 key-value manager ("a" : 1)，并验证检查通过
        self.assertTrue(root.check(f_locals))
        dict_mgr.get_key_manager(0, "", "a", default_mgr_enum).add_equals_match_guard(
            "a",
            ["dict.keys()[0] == a"],
        )
        self.assertTrue(root.check(f_locals))
        dict_mgr.get_value_manager(0, "", 1, default_mgr_enum).add_equals_match_guard(
            1, ["d[0] == 1"]
        )
        self.assertTrue(root.check(f_locals))

        # 添加 key-value manager (nothing : {"z" : 3})，并验证检查通过
        self.assertTrue(root.check(f_locals))
        dict_mgr.get_key_manager(1, "", nothing, default_mgr_enum).add_lambda_guard(
            lambda x: x is nothing, ["x is nothing"]
        )
        self.assertTrue(root.check(f_locals))
        
        # 获取 value manager，预期其子 GuardManager 为 DictGuardManager 类型，并验证检查通过
        value_mgr = dict_mgr.get_value_manager(
            1,
            "",
            f_locals["d"][nothing],
            torch._dynamo.guards.GuardManagerType.DICT_GUARD_MANAGER,
        )
        self.assertTrue(isinstance(value_mgr, DictGuardManager))
        self.assertTrue(root.check(f_locals))

        # 检查结构，确保只有两个 key-value 管理器
        self.assertEqual(len(dict_mgr.get_key_value_managers()), 2)

        # 修改 f_locals 中的值，检查是否通过检查
        f_locals["d"]["a"] = 2
        self.assertFalse(root.check(f_locals))
        self.assertFalse(root.check_verbose(f_locals).result)

        # 恢复 f_locals["d"]["a"] 的值，并验证检查通过
        f_locals["d"]["a"] = 1
        self.assertTrue(root.check(f_locals))

        # 移除 f_locals["d"] 中的 key 为 100 的项，检查因长度不符合而失败
        f_locals["d"].pop(100)
        self.assertFalse(root.check(f_locals))
# 如果当前脚本被直接执行（而非被作为模块导入），则执行以下代码块
if __name__ == "__main__":
    # 从 torch._dynamo.test_case 模块中导入 run_tests 函数
    from torch._dynamo.test_case import run_tests

    # 运行导入的 run_tests 函数，用于执行测试用例
    run_tests()
```