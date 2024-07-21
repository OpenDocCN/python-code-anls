# `.\pytorch\test\dynamo\test_global.py`

```
# Owner(s): ["module: dynamo"]
# 导入 PyTorch 库
import torch

# 导入 PyTorch 内部测试相关模块
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import same

# 尝试从当前包导入 utils 模块，如果失败则从全局导入
try:
    from . import utils
except ImportError:
    import utils

# 定义一个简单的类 Pair，用来存储两个值 x 和 y
class Pair:  # noqa: B903
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 定义函数 Foo，返回一个 Pair 对象
def Foo():
    return Pair(1, 1)

# 全局变量 g_counter，初始值为 1
g_counter = 1
# 全局变量 g_list，包含三个整数元素
g_list = [0, 1, 2]
# 全局变量 g_dict，包含两个键值对
g_dict = {"a": 0, "b": 1}
# 全局变量 g_object，调用 Foo 函数返回的 Pair 对象
g_object = Foo()
# 全局变量 g_tensor，初始化为包含 10 个零的 PyTorch 张量
g_tensor = torch.zeros(10)

# 全局变量 _name，用于生成唯一的变量名
_name: int = 0

# 定义函数 fresh_name，生成新的唯一变量名，形如 v0, v1, v2 ...
def fresh_name() -> str:
    """create a new unique name for a variable: v0, v1, v2"""
    global _name
    r = f"v{_name}"
    _name += 1
    return r

# 定义函数 reset_name，重置变量名生成器
def reset_name():
    global _name
    _name = 0

# 定义测试类 TestGlobals，继承自 torch._dynamo.test_case.TestCase
class TestGlobals(torch._dynamo.test_case.TestCase):
    
    # 定义测试方法 test_store_global_1
    def test_store_global_1(self):
        # 定义内部函数 fn，使用全局变量 g_counter
        def fn(x):
            global g_counter
            val = x + g_counter
            g_counter += 1
            return val
        
        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用编译计数器对 fn 进行优化
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res1 = opt_fn(x)
        res2 = fn(x)
        # 断言结果相等
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    # 定义测试方法 test_store_global_2
    def test_store_global_2(self):
        # 定义内部函数 fn，使用全局变量 g_counter
        def fn(x):
            global g_counter
            val = x + g_counter
            g_counter += 1
            g_counter += 1
            return val
        
        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用编译计数器对 fn 进行优化
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res1 = opt_fn(x)
        """Wrap the second call with torch._dynamo as well"""
        # 再次使用编译计数器对 fn 进行优化，实现第二次调用
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res2 = opt_fn(x)
        # 断言结果相等
        self.assertTrue(same(res2 - res1, 2 * torch.ones(10)))

    # 定义测试方法 test_store_global_new
    def test_store_global_new(self):
        # 定义内部函数 fn，测试创建新的全局变量 g_counter_new
        def fn(x):
            global g_counter_new
            g_counter_new = x + 1
            return x + g_counter_new
        
        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用编译计数器对 fn 进行优化
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res1 = opt_fn(x)
        # 断言结果相等
        self.assertTrue(same(res1, x + x + 1))

    # 定义测试方法 test_store_global_list
    def test_store_global_list(self):
        # 定义内部函数 fn，使用全局变量 g_list
        def fn(x):
            global g_list
            val = x + g_list[1]
            """
            Strictly speaking, we are not testing STORE_GLOBAL
            here, since STORE_SUBSCR is actually used to store.
            """
            g_list[1] += 1
            return val
        
        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用编译计数器对 fn 进行优化
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res1 = opt_fn(x)
        res2 = fn(x)
        # 断言结果相等
        self.assertTrue(same(res2 - res1, torch.ones(10)))
    def test_store_global_list_2(self):
        # 定义一个函数 fn，接受一个参数 x
        def fn(x):
            # 声明 g_list 为全局变量
            global g_list
            # 计算 val 的值，为 x 加上 g_list 列表的第二个元素
            val = x + g_list[1]
            # 更新 g_list 列表，将每个元素加一后重新赋值给 g_list
            g_list = [x + 1 for x in g_list]
            # 返回计算结果 val
            return val

        # 创建一个包含随机数的张量 x
        x = torch.randn(10)
        # 创建一个编译计数器 cnts 对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 通过优化函数对 fn 进行编译优化
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 分别调用优化后的函数 opt_fn 和原始函数 fn，获取结果 res1 和 res2
        res1 = opt_fn(x)
        res2 = fn(x)
        # 使用 assertTrue 断言 res2 减去 res1 的结果与一个张量全为1的张量相等
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_dict(self):
        # 定义一个函数 fn，接受一个参数 x
        def fn(x):
            # 声明 g_dict 为全局变量
            global g_dict
            # 计算 val 的值，为 x 加上 g_dict 中键为 "b" 的值
            val = x + g_dict["b"]
            """
            Strictly speaking, we are not testing STORE_GLOBAL
            here, since STORE_SUBSCR is actually used to store.
            """
            # 更新 g_dict 中键为 "b" 的值，加一
            g_dict["b"] += 1
            # 返回计算结果 val
            return val

        # 创建一个包含随机数的张量 x
        x = torch.randn(10)
        # 创建一个编译计数器 cnts 对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 通过优化函数对 fn 进行编译优化
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 分别调用优化后的函数 opt_fn 和原始函数 fn，获取结果 res1 和 res2
        res1 = opt_fn(x)
        res2 = fn(x)
        # 使用 assertTrue 断言 res2 减去 res1 的结果与一个张量全为1的张量相等
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_dict_2(self):
        # 定义一个函数 fn，接受一个参数 x
        def fn(x):
            # 声明 g_dict 为全局变量
            global g_dict
            # 使用字典推导式，对 g_dict 中每个键值对的值加一，并重新赋值给 g_dict
            g_dict = {key: value + 1 for key, value in g_dict.items()}
            # 计算 val 的值，为 x 加上 g_dict 中键为 "b" 的值
            val = x + g_dict["b"]
            # 返回计算结果 val
            return val

        # 创建一个包含随机数的张量 x
        x = torch.randn(10)
        # 创建一个编译计数器 cnts 对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 通过优化函数对 fn 进行编译优化
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 分别调用优化后的函数 opt_fn 和原始函数 fn，获取结果 res1 和 res2
        res1 = opt_fn(x)
        res2 = fn(x)
        # 使用 assertTrue 断言 res2 减去 res1 的结果与一个张量全为1的张量相等
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_object(self):
        # 定义一个函数 fn，接受一个参数 x
        def fn(x):
            # 声明 g_object 为全局变量
            global g_object
            # 计算 val 的值，为 x 加上 g_object 对象的属性 y 的值
            val = x + g_object.y
            # 更新 g_object 对象的属性 y，加一
            g_object.y += 1
            # 返回计算结果 val
            return val

        # 创建一个包含随机数的张量 x
        x = torch.randn(10)
        # 创建一个编译计数器 cnts 对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 通过优化函数对 fn 进行编译优化
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 分别调用优化后的函数 opt_fn 和原始函数 fn，获取结果 res1 和 res2
        res1 = opt_fn(x)
        res2 = fn(x)
        # 使用 assertTrue 断言 res2 减去 res1 的结果与一个张量全为1的张量相等
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_cross_file(self):
        # 定义一个函数 fn，接受一个参数 x
        def fn(x):
            # 计算 val 的值，为 x 加上 utils 模块中的全局变量 g_tensor_export
            val = x + utils.g_tensor_export
            # 更新 utils 模块中的全局变量 g_tensor_export，加一
            utils.g_tensor_export = utils.g_tensor_export + 1
            # 返回计算结果 val
            return val

        # 创建一个包含随机数的张量 x
        x = torch.randn(10)
        # 创建一个编译计数器 cnts 对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 通过优化函数对 fn 进行编译优化
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 分别调用优化后的函数 opt_fn 和原始函数 fn，获取结果 res1 和 res2
        res1 = opt_fn(x)
        res2 = fn(x)
        # 使用 assertTrue 断言 res2 减去 res1 的结果与一个张量全为1的张量相等
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_inline_1(self):
        # 从 test_python_autograd.py 借用的类 Variable 的定义
        class Variable:
            def __init__(self, value: torch.Tensor, name: str = None):
                self.value = value
                self.name = name or fresh_name()

        # 定义一个函数 fn，接受两个参数 a 和 b
        def fn(a, b):
            # 将参数 a 和 b 封装为 Variable 类的实例
            a = Variable(a)
            b = Variable(b)
            # 返回两个 Variable 实例的值和名称拼接的结果
            return a.value + b.value, a.name + b.name

        # 创建两个包含随机数的张量 a 和 b
        a = torch.randn(10)
        b = torch.randn(10)
        # 创建一个编译计数器 cnts 对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 通过优化函数对 fn 进行编译优化
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 调用优化后的函数 opt_fn，获取结果 v0 和 s0
        v0, s0 = opt_fn(a, b)
        # 使用 assertEqual 断言 s0 的值为 "v0v1"
        self.assertEqual(s0, "v0v1")
        # 调用 reset_name 函数，重置名称状态
        reset_name()
    def test_store_global_inline_2(self):
        # Borrowed from test_python_autograd.py
        # 定义一个内部类 Variable，用于包装 torch.Tensor 和名称
        class Variable:
            # 初始化方法，接受一个 torch.Tensor 对象和可选的名称参数
            def __init__(self, value: torch.Tensor, name: str = None):
                self.value = value
                self.name = name or fresh_name()  # 如果没有提供名称，则调用 fresh_name() 生成一个新名称

            # 静态方法，用于创建一个常量 Variable 对象
            @staticmethod
            def constant(value: torch.Tensor, name: str = None):
                return Variable(value, name)

        # 定义一个函数 fn，接受两个参数 a 和 b
        def fn(a, b):
            # 将参数 a 转换为 Variable 对象
            a = Variable.constant(a)
            # 将参数 b 转换为 Variable 对象
            b = Variable.constant(b)
            # 返回两个 Variable 对象的值和名称拼接的结果
            return a.value + b.value, a.name + b.name

        # 创建两个长度为 10 的随机张量 a 和 b
        a = torch.randn(10)
        b = torch.randn(10)
        # 创建一个 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 对函数 fn 进行优化处理，并返回优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 调用优化后的函数 opt_fn，并获取返回的值 v0 和 s0
        v0, s0 = opt_fn(a, b)
        # 断言 s0 的值应为 "v0v1"
        self.assertEqual(s0, "v0v1")
        # 调用 reset_name() 函数，重置名称状态
        reset_name()
# 如果当前脚本被直接执行（而非被导入为模块），则执行以下代码块
if __name__ == "__main__":
    # 从torch._dynamo.test_case模块中导入run_tests函数
    from torch._dynamo.test_case import run_tests

    # 运行导入的run_tests函数，用于执行测试用例
    run_tests()
```