# `.\pytorch\test\jit\test_await.py`

```
# Owner(s): ["oncall: jit"]

import io  # 导入 io 模块，用于处理输入输出流
from typing import List, Optional, Tuple  # 导入类型提示相关的模块

import torch  # 导入 PyTorch 模块
from torch import Tensor  # 导入 Tensor 类型
from torch._awaits import _Await as Await  # 导入 Await 类别名
from torch.testing._internal.jit_utils import JitTestCase, make_global  # 导入测试相关的工具类和全局变量设置


class TestAwait(JitTestCase):
    def test_await_python(self):
        def foo(x: int) -> int:
            return x + 13

        aw: Await[int] = torch.jit._awaitable(foo, 13)  # 创建一个 Await[int] 对象，使用 foo 函数和参数 13
        self.assertTrue(aw.fn()(*aw.args()) == torch.jit._awaitable_wait(aw))  # 断言调用 fn 和 args 返回值等于 awaitable_wait
        nw = torch.jit._awaitable_nowait(33)  # 创建一个 Nowait 对象，参数为 33
        self.assertTrue(nw.is_nowait())  # 断言 nw 是否是 nowait 类型
        self.assertTrue(nw.args() == (33,))  # 断言 nw 的参数是 (33,)

    def test_await_type_python(self):
        def foo() -> Tensor:
            return torch.randn()

        awaits = torch.jit.annotate(List[Await[Tensor]], [])  # 声明一个类型为 List[Await[Tensor]] 的空列表
        awaits.append(torch.jit._awaitable(foo))  # 向列表中添加一个 Await[Tensor] 对象，使用 foo 函数

    def test_script(self):
        def delayed(z: int) -> int:
            return z + 3

        def fn(x: Tensor):
            aw: Await[int] = torch.jit._awaitable(delayed, 99)  # 创建一个 Await[int] 对象，使用 delayed 函数和参数 99
            a = torch.eye(2)  # 创建一个 2x2 的单位矩阵 a
            b = torch.jit._awaitable_wait(aw)  # 调用 awaitable_wait 等待 aw 的结果
            return a + b + x  # 返回 a + b + x

        inp = torch.zeros(2)  # 创建一个 2 维全零 Tensor

        sm = torch.jit.script(fn)  # 对 fn 函数进行脚本化
        out = fn(inp)  # 调用 fn 函数并记录输出结果
        script_out = sm(inp)  # 使用脚本化模型 sm 处理输入 inp 并记录输出结果
        self.assertTrue(torch.allclose(torch.eye(2) + 102, script_out))  # 断言脚本化输出结果与预期值的接近程度
        self.assertTrue(torch.allclose(script_out, out))  # 断言脚本化输出结果与原始输出结果的接近程度

    def test_nowait(self):
        def fn(x: Tensor):
            aw = torch.jit._awaitable_nowait(13)  # 创建一个 Nowait 对象，参数为 13
            a = torch.eye(2)  # 创建一个 2x2 的单位矩阵 a
            b = torch.jit._awaitable_wait(aw)  # 调用 awaitable_wait 等待 aw 的结果
            return a + b + x  # 返回 a + b + x

        inp = torch.zeros(2)  # 创建一个 2 维全零 Tensor

        sm = torch.jit.script(fn)  # 对 fn 函数进行脚本化
        out = fn(inp)  # 调用 fn 函数并记录输出结果
        script_out = sm(inp)  # 使用脚本化模型 sm 处理输入 inp 并记录输出结果
        self.assertTrue(torch.allclose(torch.eye(2) + 13, script_out))  # 断言脚本化输出结果与预期值的接近程度
        self.assertTrue(torch.allclose(script_out, out))  # 断言脚本化输出结果与原始输出结果的接近程度

    def test_nowait_class(self):
        class C:
            def __init__(self, a: Tensor, b: Tensor):
                self._a = a
                self._b = b

            def a(self) -> Tensor:
                return self._a

        def fn(x: Tensor):
            aw = torch.jit._awaitable_nowait(C(torch.zeros(2), torch.ones(2)))  # 创建一个 Nowait 对象，参数为 C 类的实例
            _a = torch.eye(2)  # 创建一个 2x2 的单位矩阵 _a
            c = torch.jit._awaitable_wait(aw)  # 调用 awaitable_wait 等待 aw 的结果
            return _a + c.a() + x  # 返回 _a + c.a() + x

        make_global(C)  # 将 C 类设置为全局可用
        inp = torch.zeros(2)  # 创建一个 2 维全零 Tensor

        sm = torch.jit.script(fn)  # 对 fn 函数进行脚本化
        out = fn(inp)  # 调用 fn 函数并记录输出结果
        script_out = sm(inp)  # 使用脚本化模型 sm 处理输入 inp 并记录输出结果
        self.assertTrue(torch.allclose(torch.eye(2), script_out))  # 断言脚本化输出结果与预期值的接近程度
        self.assertTrue(torch.allclose(script_out, out))  # 断言脚本化输出结果与原始输出结果的接近程度
    def test_await_class_arg(self):
        # 定义一个内部类 C，接受两个 Tensor 类型参数 a 和 b
        class C:
            def __init__(self, a: Tensor, b: Tensor):
                # 初始化方法，将参数 a 和 b 分别赋给私有属性 __a 和 __b
                self.__a = a
                self.__b = b

            # 返回私有属性 __a 的方法
            def a(self) -> Tensor:
                return self.__a

        # 将类 C 注册为全局类
        make_global(C)

        # 定义一个函数 delayed，接受一个参数 c，返回 c 的属性 a 的值
        def delayed(c: C) -> Tensor:
            return c.a()

        # 定义函数 fn，接受一个参数 x，生成一个类 C 的实例 c，调用 torch.jit._awaitable 方法生成 aw
        def fn(x: Tensor):
            c = C(torch.zeros(2), torch.ones(2))
            aw = torch.jit._awaitable(delayed, c)
            _a = torch.eye(2)
            c2_t = torch.jit._awaitable_wait(aw)
            return _a + c2_t + x

        # 生成一个 Tensor 类型的零向量作为输入
        inp = torch.zeros(2)

        # 对函数 fn 进行脚本化，得到 sm
        sm = torch.jit.script(fn)
        # 分别用输入 inp 调用 fn 和 sm
        out = fn(inp)
        script_out = sm(inp)
        # 断言两个输出结果是否近似相等
        self.assertTrue(torch.allclose(torch.eye(2), script_out))
        self.assertTrue(torch.allclose(script_out, out))

    def test_awaitable_to_await(self):
        # 定义一个内部类 C，具有私有属性 _a 和 _b
        class C:
            __slots__ = ["_a", "_b"]

            def __init__(self, a: Tensor, b: Tensor):
                self._a = a
                self._b = b

        # 将类 C 注册为全局类
        make_global(C)

        # 定义函数 C_wait_impl，接受 self 参数，返回 self._a + self._b 的结果
        def C_wait_impl(self: C):
            return self._a + self._b

        # 定义函数 fn，接受一个参数 x，生成类 C 的实例并调用 torch.jit._awaitable 方法生成 aw
        def fn(x: Tensor):
            aw = torch.jit._awaitable(C_wait_impl, C(torch.zeros(2), torch.ones(2)))
            _a = torch.eye(2)
            c_wait_impl_res = torch.jit._awaitable_wait(aw)
            return _a + c_wait_impl_res + x

        # 生成一个 Tensor 类型的全为 1 的向量作为输入
        inp = torch.ones(2)

        # 对函数 fn 进行脚本化，得到 sm
        sm = torch.jit.script(fn)
        # 分别用输入 inp 调用 fn 和 sm
        out = fn(inp)
        script_out = sm(inp)
        # 断言两个输出结果是否近似相等
        self.assertTrue(torch.allclose(torch.eye(2) + 2 * torch.ones(2), script_out))
        self.assertTrue(torch.allclose(script_out, out))

    def test_await_class_return(self):
        # 定义一个内部类 C，具有属性 a 和 b
        class C:
            __slots__ = ["a", "b"]

            def __init__(self, a: Tensor, b: Tensor):
                self.a = a
                self.b = b

        # 将类 C 注册为全局类
        make_global(C)

        # 定义函数 C_wait_impl，接受 self 参数，返回类 C 的实例，属性分别为 self.a * 2 和 self.b * 3
        def C_wait_impl(self: C) -> C:
            return C(self.a * 2, self.b * 3)

        # 定义函数 fn_arg_C，接受一个参数 x，返回 x.a + x.b 的结果
        def fn_arg_C(x: C) -> Tensor:
            return x.a + x.b

        # 定义函数 fn，接受一个参数 x，生成类 C 的实例并调用 torch.jit._awaitable 方法生成 aw
        def fn(x: Tensor):
            aw: Await[C] = torch.jit._awaitable(C_wait_impl, C(x, x))
            _a = torch.eye(2)
            y = fn_arg_C(torch.jit._awaitable_wait(aw))
            return _a + y + x

        # 生成一个 Tensor 类型的全为 1 的向量作为输入
        inp = torch.ones(2)

        # 对函数 fn 进行脚本化，得到 sm
        sm = torch.jit.script(fn)
        # 分别用输入 inp 调用 fn 和 sm
        out = fn(inp)
        script_out = sm(inp)
        # 断言两个输出结果是否近似相等
        self.assertTrue(torch.allclose(torch.eye(2) + 6 * torch.ones(2), script_out))
        self.assertTrue(torch.allclose(script_out, out))
        self.assertGraphContainsExactly(
            sm.graph, kind="prim::awaitable_wait", num_kind_nodes=1
        )
    # 定义一个测试方法，用于验证在获取属性时的异步操作转换
    def test_await_getattr_implicit_convertion(self):
        # 定义一个类 C
        class C:
            # 类 C 的初始化方法，接受两个张量参数 a 和 b
            def __init__(self, a: Tensor, b: Tensor):
                self._a = a  # 将参数 a 存储在实例变量 _a 中
                self._b = b  # 将参数 b 存储在实例变量 _b 中

            # 定义一个方法 b，返回实例变量 _b
            def b(self):
                return self._b

        make_global(C)  # 将类 C 注册为全局类

        # 定义一个函数 C_wait_impl，接受参数 self，并返回类型为 C
        # 此函数用于等待实现，不能作为类内部函数存在，因为 Jit 不支持递归注解
        def C_wait_impl(self: C) -> C:
            return C(self._a * 2, self._b * 3)

        # 定义一个函数 fn_arg_C，接受参数 x 类型为 C，返回张量类型
        def fn_arg_C(x: C) -> Tensor:
            return x._a + x._b

        # 定义一个函数 fn，接受参数 x 为张量类型
        def fn(x: Tensor):
            # 创建一个等待对象 aw，类型为 Await[C]
            aw: Await[C] = torch.jit._awaitable(C_wait_impl, C(x, x))
            _a = torch.eye(2)  # 创建一个 2x2 的单位张量
            ai = aw._a  # 获取 aw 对象的 _a 属性
            awb = aw.b()  # 调用 aw 对象的 b 方法
            c = C(2 * x, 2 * x)  # 创建一个新的 C 类对象 c
            # 返回 _a、ai、x、c._a 和 c.b() 的和
            return _a + ai + x + c._a + c.b()

        inp = torch.ones(2)  # 创建一个包含两个元素的张量全为 1

        sm = torch.jit.script(fn)  # 对函数 fn 进行脚本化
        out = fn(inp)  # 调用 fn 函数，并传入 inp 作为参数
        script_out = sm(inp)  # 调用脚本化的 sm 函数，并传入 inp 作为参数
        # 断言脚本化的输出与原始输出的近似程度
        self.assertTrue(torch.allclose(torch.eye(2) + 7 * torch.ones(2), script_out))
        self.assertTrue(torch.allclose(script_out, out))
        # 断言脚本化图中包含两个 prim::awaitable_wait 类型的节点
        self.assertGraphContainsExactly(
            sm.graph, kind="prim::awaitable_wait", num_kind_nodes=2
        )

    # 定义一个测试方法，用于验证嵌套等待
    def test_await_nested(self):
        # 定义一个类 C
        class C:
            # 类 C 的初始化方法，接受两个张量参数 a 和 b
            def __init__(self, a: Tensor, b: Tensor):
                self.__a = a  # 将参数 a 存储在实例变量 __a 中
                self.__b = b  # 将参数 b 存储在实例变量 __b 中

            # 定义一个方法 a，返回实例变量 __a
            def a(self) -> Tensor:
                return self.__a

        make_global(C)  # 将类 C 注册为全局类

        # 定义一个函数 delayed，接受参数 c 类型为 C，并返回 Await[Tensor]
        def delayed(c: C) -> Await[Tensor]:
            return torch.jit._awaitable_nowait(3 * c.a())

        # 定义一个函数 fn，接受参数 x 为张量类型，并返回 Await[Await[Tensor]]
        def fn(x: Tensor) -> Await[Await[Tensor]]:
            return torch.jit._awaitable(delayed, C(2 * x, x))

        # 定义一个函数 main，接受参数 x 为张量类型，并返回张量类型
        def main(x: Tensor) -> Tensor:
            awaw = fn(x)  # 调用 fn 函数，并传入 x 作为参数
            # 对 awaw 进行两次等待操作
            return torch.jit._awaitable_wait(torch.jit._awaitable_wait(awaw))

        inp = torch.eye(2)  # 创建一个 2x2 的单位张量

        sm = torch.jit.script(main)  # 对函数 main 进行脚本化
        out = main(inp)  # 调用 main 函数，并传入 inp 作为参数
        script_out = sm(inp)  # 调用脚本化的 sm 函数，并传入 inp 作为参数
        # 断言脚本化的输出与原始输出的近似程度
        self.assertTrue(torch.allclose(6 * torch.eye(2), script_out))
        self.assertTrue(torch.allclose(script_out, out))

    # 定义一个测试方法，用于验证非脚本化的等待操作
    def test_eager_await_non_scriptable(self):
        # 定义一个树类 Tree，其初始化方法接受一个参数 v
        # 树类型不能编译（递归类型）
        class Tree:
            def __init__(self, v):
                # 将 parent 属性注释为可选的树类型，默认为 None
                self.parent = torch.jit.annotate(Optional[Tree], None)
                self.v = v  # 将参数 v 存储在实例变量 v 中

        make_global(Tree)  # 将类 Tree 注册为全局类

        # 定义一个函数 delayed，接受参数 t 类型为 Tree
        def delayed(t: Tree):
            t.v = t.v + 1  # 将 t 的 v 属性加 1
            return t

        aw = torch.jit._awaitable(delayed, Tree(2))  # 创建一个等待操作 aw
        t = torch.jit._awaitable_wait(aw)  # 对 aw 进行等待操作，并赋值给 t
        # 断言 t 的 v 属性是否等于 3
        self.assertTrue(t.v == 3)
    # 测试异步等待是否为特定类型的实例
    def test_await_isinstance(self):
        # 定义一个延迟函数，接受一个张量并返回处理后的张量
        def delayed(x: Tensor) -> Tensor:
            return 2 * (x + 1)

        # 主函数，接受一个张量并返回处理后的张量
        def main(x: Tensor) -> Tensor:
            # 创建一个可等待对象
            aw = torch.jit._awaitable(delayed, x)
            # 如果正在进行脚本化编译
            if torch.jit.is_scripting():
                # 断言可等待对象是 torch.jit._Await 类的实例
                assert isinstance(aw, torch.jit._Await)
            # 等待可等待对象的完成并返回结果
            return torch.jit._awaitable_wait(aw)

        # 输入是一个 2x2 的单位矩阵张量
        inp = torch.eye(2)

        # 对主函数进行脚本编译
        sm = torch.jit.script(main)
        # 执行主函数
        out = main(inp)
        # 使用脚本化版本执行主函数
        script_out = sm(inp)
        # 断言两个张量在数值上接近
        self.assertTrue(
            torch.allclose(2 * torch.eye(2) + 2 * torch.ones(2), script_out)
        )
        # 断言两个张量在数值上接近
        self.assertTrue(torch.allclose(script_out, out))

    # 测试异步等待在即时执行和延迟执行模式下的行为
    def test_await_eager_lazy(self):
        # 定义一个延迟函数，接受一个张量并返回处理后的张量
        def delayed(x: Tensor) -> Tensor:
            return 2 * (x + 1)

        # 创建一个张量，数据类型为 int64
        t = torch.ones(2, dtype=torch.int64)
        # 创建一个可等待对象
        aw = torch.jit._awaitable(delayed, t)
        # 断言可等待对象是 torch._C._Await 类的实例
        self.assertTrue(isinstance(aw, torch._C._Await))
        # 断言张量的数据类型与可等待对象的数据类型相同
        self.assertTrue(t.dtype == aw.dtype)

    # 测试异步等待在解释器之外的行为
    def test_await_out_of_interpreter(self):
        # 定义一个延迟函数，接受一个张量并返回处理后的张量
        def delayed(x: Tensor) -> Tensor:
            return 2 * (x + 1)

        # 主函数，接受一个张量并返回一个异步等待的张量
        def main(x: Tensor) -> Await[Tensor]:
            # 创建一个可等待对象
            aw = torch.jit._awaitable(delayed, x)
            # 返回可等待对象
            return aw

        # 输入是一个 2x2 的单位矩阵张量
        inp = torch.eye(2)

        # 对主函数进行脚本编译
        sm = torch.jit.script(main)
        # 执行主函数，获取异步等待对象
        out_aw = main(inp)
        # 等待异步等待对象完成，并获取结果张量
        out = torch.jit._awaitable_wait(out_aw)

        # 使用脚本化版本执行主函数，获取异步等待对象
        script_out_aw = sm(inp)
        # 等待异步等待对象完成，并获取结果张量
        script_out = torch.jit._awaitable_wait(script_out_aw)

        # 断言两个张量在数值上接近
        self.assertTrue(
            torch.allclose(2 * torch.eye(2) + 2 * torch.ones(2), script_out)
        )
        # 断言两个张量在数值上接近
        self.assertTrue(torch.allclose(script_out, out))

    # 测试即时跟踪（JIT trace）功能
    def test_jit_trace(self):
        # 定义一个函数，接受一个张量并返回对其进行一系列操作后的张量
        def gap(x: Tensor):
            return torch.relu(x) + torch.sin(x)

        # 定义一个延迟函数，接受一个张量并返回处理后的张量
        def delayed(x: Tensor) -> Tensor:
            return 2 * (torch.cos(x) + 1)

        # 主函数，接受两个张量并返回处理后的张量
        def main(x: Tensor, y: Tensor) -> Tensor:
            # 创建一个可等待对象
            aw = torch.jit._awaitable(delayed, x)
            # 对 y 执行一些操作
            z = gap(y)
            # 等待可等待对象的完成并获取结果
            k = torch.jit._awaitable_wait(aw)
            # 返回 y 加上 k 的结果
            return y + k

        # 输入是一个长度为 2 的随机张量
        inp = torch.randn(2)
        # 对主函数进行即时跟踪
        tm = torch.jit.trace(main, (inp, inp))
        # 创建一个全为 1 的张量作为输入
        inp_check = torch.ones(2)
        # 断言主函数直接调用和跟踪后的结果一致
        self.assertEqual(main(inp_check, inp_check), tm(inp_check, inp_check))
    # 定义一个测试方法，用于测试包含异步操作的函数 `main` 在保存和加载后的行为
    def test_await_multiout_save(self):
        # 定义一个函数 `gap`，接受一个张量 `x`，返回 `torch.relu(x) + torch.sin(x)`
        def gap(x: Tensor):
            return torch.relu(x) + torch.sin(x)

        # 定义一个函数 `delayed`，接受一个张量 `x`，返回一个元组 `(100 * x, [x * i for i in range(5)])`
        def delayed(x: Tensor) -> Tuple[Tensor, List[Tensor]]:
            l = [x * i for i in range(5)]  # 创建一个包含多个张量的列表
            return (100 * x, l)

        # 定义主函数 `main`，接受一个张量 `x`，返回计算结果
        def main(x: Tensor) -> Tensor:
            # 调用 `_awaitable` 函数创建一个异步操作对象 `aw`，等待 `delayed` 函数的执行结果
            aw = torch.jit._awaitable(delayed, x)
            # 计算 `gap(x)` 的结果并赋给 `z`
            z = gap(x)
            # 等待异步操作 `aw` 完成，并解包结果到变量 `(_, l)`
            (_, l) = torch.jit._awaitable_wait(aw)
            # 返回列表 `l` 中索引为 3 的张量与 `z` 的和
            return l[3] + z

        # 创建输入张量 `inp`，为单位矩阵
        inp = torch.eye(2)

        # 对主函数 `main` 进行脚本化编译，得到 `sm`
        sm = torch.jit.script(main)
        # 调用 `main` 函数并传入 `inp`，记录输出结果到 `out`
        out = main(inp)
        # 使用脚本化的 `sm` 对象调用并传入 `inp`，记录输出结果到 `script_out`
        script_out = sm(inp)
        # 预期的输出结果是 `4.8415 * torch.eye(2)`
        expected = 4.8415 * torch.eye(2)
        # 断言 `script_out` 与 `expected` 的所有元素近似相等
        self.assertTrue(torch.allclose(expected, script_out))
        # 断言 `script_out` 与 `out` 的所有元素近似相等
        self.assertTrue(torch.allclose(script_out, out))

        # 创建一个字节流对象 `iofile`
        iofile = io.BytesIO()
        # 将脚本化的 `sm` 对象保存到 `iofile` 中
        torch.jit.save(sm, iofile)
        # 将 `iofile` 的读取位置移动到开头
        iofile.seek(0)
        # 从 `iofile` 中加载模型并赋给 `sm`
        sm = torch.jit.load(iofile)
        # 使用加载后的 `sm` 对象调用并传入 `inp`，记录输出结果到 `script_out_load`
        script_out_load = sm(inp)
        # 断言 `script_out_load` 与 `expected` 的所有元素近似相等
        self.assertTrue(torch.allclose(expected, script_out_load))

    # 定义一个测试方法，用于测试异步函数作为参数传递的情况
    def test_await_func_arg(self):
        # 定义一个函数 `gap`，接受一个张量 `x`，返回 `torch.relu(x) + torch.sin(x)`
        def gap(x: Tensor):
            return torch.relu(x) + torch.sin(x)

        # 定义一个函数 `delayed`，接受一个张量 `x`，返回 `-1 * x`
        def delayed(x: Tensor) -> Tensor:
            return -1 * x

        # 定义一个函数 `fn`，接受一个异步张量 `aw`，返回 `3 * torch.jit._awaitable_wait(aw)`
        def fn(aw: Await[Tensor]) -> Tensor:
            return 3 * torch.jit._awaitable_wait(aw)

        # 定义主函数 `main`，接受一个张量 `x`，返回计算结果
        def main(x: Tensor) -> Tensor:
            # 调用 `_awaitable` 函数创建一个异步操作对象 `aw`，等待 `delayed` 函数的执行结果
            aw = torch.jit._awaitable(delayed, x)
            # 计算 `gap(x)` 的结果并赋给 `z`
            z = gap(x)
            # 调用函数 `fn` 并传入异步操作对象 `aw`，结果赋给 `y`
            y = fn(aw)
            # 返回 `y` 与 `x` 的和
            return y + x

        # 创建输入张量 `inp`，为单位矩阵
        inp = torch.eye(2)

        # 对主函数 `main` 进行脚本化编译，得到 `sm`
        sm = torch.jit.script(main)
        # 调用 `main` 函数并传入 `inp`，记录输出结果到 `out`
        out = main(inp)
        # 使用脚本化的 `sm` 对象调用并传入 `inp`，记录输出结果到 `script_out`
        script_out = sm(inp)
        # 预期的输出结果是 `-2 * torch.eye(2)`
        expected = -2 * torch.eye(2)
        # 断言 `script_out` 与 `expected` 的所有元素近似相等
        self.assertTrue(torch.allclose(expected, script_out))
        # 断言 `script_out` 与 `out` 的所有元素近似相等
        self.assertTrue(torch.allclose(script_out, out))

        # 创建一个字节流对象 `iofile`
        iofile = io.BytesIO()
        # 将脚本化的 `sm` 对象保存到 `iofile` 中
        torch.jit.save(sm, iofile)
        # 将 `iofile` 的读取位置移动到开头
        iofile.seek(0)
        # 从 `iofile` 中加载模型并赋给 `sm`
        sm = torch.jit.load(iofile)
        # 使用加载后的 `sm` 对象调用并传入 `inp`，记录输出结果到 `script_out_load`
        script_out_load = sm(inp)
        # 断言 `script_out_load` 与 `expected` 的所有元素近似相等
        self.assertTrue(torch.allclose(expected, script_out_load))
```