# `.\pytorch\test\dynamo\test_comptime.py`

```
# Owner(s): ["module: dynamo"]

import collections  # 导入collections模块，用于创建命名元组和其他容器数据类型
import re  # 导入re模块，用于正则表达式操作
import sys  # 导入sys模块，提供对Python解释器的访问
from io import StringIO  # 从io模块导入StringIO类，用于在内存中操作文本数据流

import torch._dynamo.test_case  # 导入测试框架相关的torch模块
import torch._dynamo.testing  # 导入测试相关的torch模块
from torch._dynamo.comptime import comptime  # 从torch._dynamo.comptime模块导入comptime函数

# 因为当前不支持comptime中的自由变量，必须通过全局变量进行通信。
# 这也意味着这些测试不能在单个进程中并行运行（尽管你可能从不...希望这样做？）
FILE = None  # 初始化全局变量FILE为None
SELF = None  # 初始化全局变量SELF为None


class ComptimeTests(torch._dynamo.test_case.TestCase):
    def test_print_single(self):
        global FILE  # 声明全局变量FILE的使用

        FILE = StringIO()  # 将FILE设置为一个新的StringIO对象，用于捕获输出
        cnt = torch._dynamo.testing.CompileCounter()  # 创建CompileCounter对象cnt，用于计数

        # 定义一个在编译时打印函数comptime_print
        def comptime_print(e):
            @comptime
            def _(ctx):
                ctx.print(ctx.get_local("e"), file=FILE)

        Employee = collections.namedtuple("Employee", ["name", "id"])  # 创建命名元组Employee

        # 定义一个继承自list的子类mylist
        class mylist(list):
            pass

        # 优化函数f，并且在动态环境中进行
        @torch._dynamo.optimize(cnt, dynamic=True)
        def f(x):
            y = x * 2  # 计算输入x的两倍
            comptime_print(y)  # 在编译时打印y的值
            comptime_print(2)  # 在编译时打印数字2
            comptime_print([y, 2])  # 在编译时打印包含y和2的列表
            comptime_print((y, 2))  # 在编译时打印包含y和2的元组
            comptime_print({"foo": y})  # 在编译时打印包含键值对{'foo': y}的字典
            comptime_print(range(1, 3))  # 在编译时打印范围对象range(1, 3)
            comptime_print(Employee("foo", 2))  # 在编译时打印Employee命名元组
            comptime_print(mylist([1, 2]))  # 在编译时打印mylist类的实例
            comptime_print(collections.defaultdict(lambda: None))  # 在编译时打印默认值为None的defaultdict
            comptime_print(set())  # 在编译时打印空集合set()
            comptime_print({"a", "b"})  # 在编译时打印包含字符串'a'和'b'的集合
            comptime_print(x.size(0))  # 在编译时打印x的第一个维度大小
            return y + 3  # 返回y加3的结果

        f(torch.randn(2))  # 运行函数f，传入一个形状为(2,)的随机张量作为参数
        self.assertEqual(cnt.frame_count, 1)  # 断言编译帧数为1
        self.assertExpectedInline(
            FILE.getvalue().strip(),
            """\
FakeTensor(..., size=(s0,))
2
[FakeTensor(..., size=(s0,)), 2]
(FakeTensor(..., size=(s0,)), 2)
{'foo': FakeTensor(..., size=(s0,))}
range(1, 3, 1)
Employee(name='foo', id=2)
[1, 2]
defaultdict(NestedUserFunctionVariable(), {})
set()
{'a','b'}
s0""",
        )

    def test_print_graph(self):
        global FILE  # 声明全局变量FILE的使用

        FILE = StringIO()  # 将FILE设置为一个新的StringIO对象，用于捕获输出
        cnt = torch._dynamo.testing.CompileCounter()  # 创建CompileCounter对象cnt，用于计数

        # 优化函数f，并且在不同的环境中进行
        @torch._dynamo.optimize(cnt)
        def f(x):
            y = x * 2  # 计算输入x的两倍

            # 定义一个在编译时打印图形的函数
            @comptime
            def _(ctx):
                ctx.print_graph(verbose=False, file=FILE)

            # 测试紧凑表示法不会引发错误或破坏图形;
            # 你需要视觉检查以查看它是否打印了内容
            comptime.print_graph()

            return y + 3  # 返回y加3的结果

        f(torch.randn(2))  # 运行函数f，传入一个形状为(2,)的随机张量作为参数
        self.assertEqual(cnt.frame_count, 1)  # 断言编译帧数为1
        self.assertExpectedInline(
            FILE.getvalue().strip(),
            """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    y = l_x_ * 2;  l_x_ = None""",
        )
    # 定义一个测试方法，用于验证打印反汇编信息的功能
    def test_print_disas(self):
        # 声明全局变量 FILE，用于存储输出内容
        global FILE
        # 初始化 FILE 为一个字符串流对象
        FILE = StringIO()
        # 创建一个编译计数器对象
        cnt = torch._dynamo.testing.CompileCounter()

        # 定义一个装饰器函数，用于优化函数 f 的编译过程
        @torch._dynamo.optimize(cnt)
        def f(x):
            # 计算输入参数 x 的两倍
            y = x * 2

            # 嵌套定义一个编译期函数，用于打印反汇编指令到 FILE 中
            @comptime
            def _(ctx):
                ctx.print_disas(file=FILE)

            # 调用编译期函数的 print_disas 方法，打印反汇编指令到 FILE
            comptime.print_disas()

            # 返回计算结果 y + 3
            return y + 3

        # 定义一个函数，用于处理反汇编字符串的格式
        def munge_disas(s):
            # 使用正则表达式替换操作，调整反汇编指令的格式
            re.sub(
                r"^(?: +\d+)?(?: +(-->)) \+\d+ ([A-Za-z0-9_]+)",
                "\1 \3",
                s,
                flags=re.MULTILINE,
            )

        # 调用函数 f，传入参数并执行
        f(torch.randn(2))
        # 断言编译帧数量为 1
        self.assertEqual(cnt.frame_count, 1)
        # 获取 FILE 中的内容
        out = FILE.getvalue()
        # 检查输出中是否包含指定字符串，验证指令偏移量的工作是否正常
        self.assertIn("-->", out)
        # 检查输出中是否包含指定字符串，验证字节码是否符合预期
        self.assertIn("STORE_FAST", out)
        if sys.version_info < (3, 11):
            self.assertIn("BINARY_MULTIPLY", out)
        else:
            self.assertIn("BINARY_OP", out)

    # 定义一个测试方法，用于验证打印值栈信息的功能
    def test_print_value_stack(self):
        # 声明全局变量 FILE，用于存储输出内容
        global FILE
        # 初始化 FILE 为一个字符串流对象
        FILE = StringIO()
        # 创建一个编译计数器对象
        cnt = torch._dynamo.testing.CompileCounter()

        # 定义一个函数 g，用于打印值栈信息并返回输入参数 x
        def g(x):
            # 嵌套定义一个编译期函数，用于打印值栈信息到 FILE 中
            @comptime
            def _(ctx):
                ctx.print_value_stack(file=FILE, stacklevel=1)

            # 返回输入参数 x
            return x

        # 定义一个装饰器函数，用于优化函数 f 的编译过程
        @torch._dynamo.optimize(cnt)
        def f(x):
            # 计算 y = x + g(x)
            y = x + g(x)

            # 返回计算结果 y + comptime.print_value_stack_and_return(y * 2) 的值
            return y + comptime.print_value_stack_and_return(y * 2)

        # 调用函数 f，传入参数并执行
        f(torch.randn(2))
        # 断言编译帧数量为 1
        self.assertEqual(cnt.frame_count, 1)
        # 断言 FILE 中的输出内容符合预期
        self.assertExpectedInline(
            FILE.getvalue(),
            """\
    def test_print_locals(self):
        # 声明全局变量 FILE，并将其设为字符串输出流
        global FILE
        FILE = StringIO()
        # 创建计数器对象 cnt
        cnt = torch._dynamo.testing.CompileCounter()

        # 定义优化函数 f，接受参数 x
        @torch._dynamo.optimize(cnt)
        def f(x):
            # 计算 y = x * 2
            y = x * 2

            # 定义内部函数 comptime，用于打印局部变量到 FILE
            @comptime
            def _(ctx):
                ctx.print_locals(file=FILE)

            # 调用 comptime 对象的 print_locals 方法
            comptime.print_locals()

            # 返回 y + 3
            return y + 3

        # 调用 f 函数，传入参数 torch.randn(2)
        f(torch.randn(2))
        # 断言帧计数为 1
        self.assertEqual(cnt.frame_count, 1)
        # 断言 FILE 对象的输出符合预期格式
        self.assertExpectedInline(
            FILE.getvalue(),
            """\
x = TensorVariable()
y = TensorVariable()
""",
        )
    def test_print_guards(self):
        # 设置全局变量 FILE 为 StringIO 对象，用于存储输出内容
        global FILE
        FILE = StringIO()
        # 创建一个编译计数器对象
        cnt = torch._dynamo.testing.CompileCounter()

        # 优化函数 f，将其传入编译计数器
        @torch._dynamo.optimize(cnt)
        def f(x):
            # 计算 y = x * 2

            # 定义 comptime 函数，在上下文中打印守卫条件到 FILE
            @comptime
            def _(ctx):
                ctx.print_guards(file=FILE)

            # 调用 comptime 的 print_guards 方法

            # 返回计算结果 y + 3
            return y + 3

        # 调用函数 f，并传入参数 torch.randn(2)
        f(torch.randn(2))
        # 断言编译帧数为 1
        self.assertEqual(cnt.frame_count, 1)
        # 断言 FILE 的输出内容符合预期，去除多余空白和换行符
        self.assertExpectedInline(
            re.sub(r"\s+$", "", FILE.getvalue().rstrip(), flags=re.MULTILINE),
            """\
        local "L['x']" TENSOR_MATCH
        {
            'guard_types': None,
            'code': None,
            'obj_weakref': None
            'guarded_class': None
        }
        global '' GRAD_MODE
        {
            'guard_types': None,
            'code': None,
            'obj_weakref': None
            'guarded_class': None
        }
        global '' DETERMINISTIC_ALGORITHMS
        {
            'guard_types': None,
            'code': None,
            'obj_weakref': None
            'guarded_class': None
        }
        global '' TORCH_FUNCTION_STATE
        {
            'guard_types': None,
            'code': None,
            'obj_weakref': None
            'guarded_class': None
        }
        global '' DEFAULT_DEVICE
        {
            'guard_types': None,
            'code': None,
            'obj_weakref': None
            'guarded_class': None
        }
        shape_env '' SHAPE_ENV
        {
            'guard_types': None,
            'code': None,
            'obj_weakref': None
            'guarded_class': None
        }""",
        )

    def test_graph_break(self):
        # 创建编译计数器对象
        cnt = torch._dynamo.testing.CompileCounter()

        # 优化函数 f，将其传入编译计数器
        @torch._dynamo.optimize(cnt)
        def f(x):
            # 计算 y = x * 2

            # 定义 comptime 函数，但此处未执行任何操作
            @comptime
            def _(ctx):
                pass

            # 返回计算结果 y + 3
            return y + 3

        # 调用函数 f，并传入参数 torch.randn(2)
        f(torch.randn(2))
        # 断言编译帧数为 1
        self.assertEqual(cnt.frame_count, 1)
        
        # 将编译帧数重置为 0
        cnt.frame_count = 0

        # 优化函数 g，将其传入编译计数器
        @torch._dynamo.optimize(cnt)
        def g(x):
            # 计算 y = x * 2

            # 定义 comptime 函数，在上下文中调用 graph_break 方法
            @comptime
            def _(ctx):
                ctx.graph_break()

            # 计算 y = y + 2

            # 调用 comptime 的 graph_break 方法

            # 返回计算结果 y * 3
            return y * 3

        # 调用函数 g，并传入参数 torch.randn(2)
        g(torch.randn(2))
        # 断言编译帧数为 3
        self.assertEqual(cnt.frame_count, 3)
    # 定义一个测试方法，用于测试本地功能
    def test_get_local(self):
        # 声明全局变量 SELF 和 FILE，用于在嵌套函数中使用
        global SELF, FILE
        SELF = self
        FILE = StringIO()
        # 创建一个编译计数器实例
        cnt = torch._dynamo.testing.CompileCounter()

        # 使用 torch._dynamo.optimize 装饰器优化函数 f
        @torch._dynamo.optimize(cnt)
        def f(x):
            # 计算 x 的两倍，并将结果赋给 y
            y = x * 2
            # 声明一个常量 lit 并赋值为 2

            # 在编译时计算的函数装饰器
            @comptime
            def _(ctx):
                # 从上下文中获取变量 y 的本地版本，并进行断言检查其大小
                y = ctx.get_local("y")
                SELF.assertEqual(y.as_fake().size(0), 2)
                SELF.assertEqual(y.size(0), 2)
                # 触发图形写入（TODO: 目前还没有方法利用输出代理；也许将副作用操作插入图形可能会有用）
                y.as_proxy() + 4
                # 打印图形，将结果输出到 FILE 中，关闭详细信息
                ctx.print_graph(verbose=False, file=FILE)
                # 断言变量 y 的 Python 类型为 torch.Tensor
                SELF.assertIs(y.python_type(), torch.Tensor)
                # 从上下文中获取变量 lit 的本地版本，并进行断言检查其值
                lit = ctx.get_local("lit")
                SELF.assertEqual(lit.as_python_constant(), 2)

            # 返回 y + 3 的计算结果
            return y + 3

        # 调用函数 f，并传入 torch.randn(2) 作为参数
        f(torch.randn(2))
        # 断言编译帧计数为 1
        self.assertEqual(cnt.frame_count, 1)
        # 断言 FILE.getvalue().strip() 的输出与预期的内联字符串相匹配
        self.assertExpectedInline(
            FILE.getvalue().strip(),
            """\
# 定义一个方法，用于前向传播，接受一个名为 L_x_ 的 Torch 张量作为参数
def forward(self, L_x_ : torch.Tensor):
    # 将输入参数 L_x_ 复制给局部变量 l_x_
    l_x_ = L_x_
    # 计算新张量 y，其值为 l_x_ 的每个元素乘以2
    y = l_x_ * 2;  l_x_ = None
    # 计算新张量 add，其值为 y 的每个元素加上4
    add = y + 4;  y = None



if __name__ == "__main__":
    # 从 torch._dynamo.test_case 模块导入 run_tests 函数
    from torch._dynamo.test_case import run_tests

    # 运行测试函数
    run_tests()
```