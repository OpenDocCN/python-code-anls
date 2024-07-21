# `.\pytorch\test\dynamo\test_exceptions.py`

```
# Owner(s): ["module: dynamo"]

# 引入 PyTorch 库
import torch
# 引入 PyTorch 的 Dynamo 配置模块
import torch._dynamo.config

# 引入 PyTorch 的 Dynamo 测试用例模块
import torch._dynamo.test_case
# 引入 PyTorch 的 Functorch 配置模块
import torch._functorch.config
# 引入 PyTorch 的 checkpoint 工具模块
import torch.utils.checkpoint

# 定义一个继承自 torch._dynamo.test_case.TestCase 的异常测试类
class ExceptionTests(torch._dynamo.test_case.TestCase):
    # 定义一个测试异常处理的方法
    def test_exception(self):
        # 定义一个函数 fn，接受输入 x
        def fn(x):
            # 对输入 x 求余弦
            x = torch.cos(x)
            # 尝试执行以下代码块
            try:
                # 对 x 求正弦
                x = torch.sin(x)
                # 抛出 NotImplementedError 异常
                raise NotImplementedError
            # 捕获任何异常
            except Exception:
                # 对 x 求 sigmoid
                x = torch.sigmoid(x)

            # 返回处理后的 x
            return x

        # 生成一个随机张量 x
        x = torch.randn(4)
        # 计算 fn 的预期输出
        ref = fn(x)
        # 编译 fn 函数，使用 "eager" 后端，并开启完整图形优化
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        # 执行优化后的 fn 函数
        res = opt_fn(x)
        # 断言优化后的结果与预期结果一致
        self.assertEqual(ref, res)

    # 定义第二个测试异常处理的方法
    def test_exception2(self):
        # 定义一个函数 fn，接受输入 x
        def fn(x):
            # 对输入 x 求余弦
            x = torch.cos(x)
            # 尝试执行以下代码块
            try:
                # 对 x 求正弦
                x = torch.sin(x)
                # 抛出 NotImplementedError 异常
                raise NotImplementedError
            # 捕获 NotImplementedError 或 AttributeError 异常
            except (NotImplementedError, AttributeError) as e:
                # 对 x 求 sigmoid
                x = torch.sigmoid(x)

            # 返回处理后的 x
            return x

        # 生成一个随机张量 x
        x = torch.randn(4)
        # 计算 fn 的预期输出
        ref = fn(x)
        # 编译 fn 函数，使用 "eager" 后端，并开启完整图形优化
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        # 执行优化后的 fn 函数
        res = opt_fn(x)
        # 断言优化后的结果与预期结果一致
        self.assertEqual(ref, res)

    # 定义第三个测试异常处理的方法
    def test_exception3(self):
        # 定义一个函数 fn，接受输入 x
        def fn(x):
            # 对输入 x 求余弦
            x = torch.cos(x)
            # 尝试执行以下代码块
            try:
                # 对 x 求正弦
                x = torch.sin(x)
                # 抛出自定义的 NotImplementedError 异常
                raise NotImplementedError("Not implemented")
            # 捕获 AssertionError 异常
            except AssertionError:
                # 对 x 求 sigmoid
                x = torch.sigmoid(x)
            # 捕获 NotImplementedError 异常
            except NotImplementedError:
                # 对 x 再次求余弦
                x = torch.cos(x)
            # 最终执行块，再次对 x 求余弦
            finally:
                x = torch.cos(x)

            # 返回处理后的 x
            return x

        # 生成一个随机张量 x
        x = torch.randn(4)
        # 计算 fn 的预期输出
        ref = fn(x)
        # 编译 fn 函数，使用 "eager" 后端，并开启完整图形优化
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        # 执行优化后的 fn 函数
        res = opt_fn(x)
        # 断言优化后的结果与预期结果一致
        self.assertEqual(ref, res)

    # 定义测试异常处理的方法，包含内部异常处理的测试
    def test_exception_with_another_exception(self):
        # 定义一个函数 fn，接受输入 x
        def fn(x):
            # 对输入 x 求余弦
            x = torch.cos(x)
            # 尝试执行以下代码块
            try:
                # 对 x 求正弦
                x = torch.sin(x)
                # 抛出自定义的 NotImplementedError 异常
                raise NotImplementedError("Not implemented")
            # 捕获 NotImplementedError 异常
            except NotImplementedError as e:
                # 对 x 求 sigmoid
                x = torch.sigmoid(x)
                # 再次尝试执行以下代码块
                try:
                    # 对 x 求余弦
                    x = torch.cos(x)
                    # 抛出 AssertionError 异常
                    raise AssertionError
                # 捕获 AssertionError 异常
                except AssertionError:
                    # 对 x 再次求余弦
                    x = torch.cos(x)

        # 生成一个随机张量 x
        x = torch.randn(4)
        # 执行 fn 函数，预期无返回值
        ref = fn(x)
        # 编译 fn 函数，使用 "eager" 后端，并开启完整图形优化
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        # 执行优化后的 fn 函数
        res = opt_fn(x)
        # 断言优化后的结果与预期结果一致（实际上这个测试函数预期无返回值，所以不进行断言）
        self.assertEqual(ref, res)

    # 定义测试异常处理的方法，包含 else 分支的测试
    def test_exception_else(self):
        # 定义一个函数 gn，接受输入 x，并返回 x 的余弦
        def gn(x):
            return torch.cos(x)

        # 定义一个函数 fn，接受输入 x
        def fn(x):
            # 对输入 x 求余弦
            x = torch.cos(x)
            # 尝试执行以下代码块
            try:
                # 对 x 求正弦
                x = torch.sin(x)
                # 对 x 应用 gn 函数
                x = gn(x)
            # 捕获任何异常
            except Exception:
                # 对 x 求 sigmoid
                x = torch.sigmoid(x)
            # 如果没有异常发生
            else:
                # 对 x 再次求余弦
                x = torch.cos(x)

            # 返回处理后的 x
            return x

        # 生成一个随机张量 x
        x = torch.randn(4)
        # 计算 fn 的预期输出
        ref = fn(x)
        # 编译 fn 函数，使用 "eager" 后端，并开启完整图形优化
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        # 执行优化后的 fn 函数
        res = opt_fn(x)
        # 断言优化后的结果与预期结果一致
        self.assertEqual(ref, res)

    # 待办事项：（anijain2305）- 不支持 fullgraph=True 的情况，暂无注释
    def test_exception_with_another_exception2(self):
        def gn(x):
            try:
                # 计算 x 的余弦值
                x = torch.cos(x)
                # 抛出未实现错误
                raise NotImplementedError("Not implemented")
            except NotImplementedError as e:
                # 计算 x 的 sigmoid 值
                x = torch.sigmoid(x)
                # 重新抛出当前异常
                raise

        def fn(x):
            try:
                # 计算 x 的余弦值
                x = torch.cos(x)
                # 调用 gn 函数，处理可能的异常
                gn(x)
            except Exception:
                pass
            return x

        x = torch.randn(4)
        # 调用普通版本的 fn 函数，并获取结果
        ref = fn(x)
        # 使用 eager 模式编译 fn 函数
        opt_fn = torch.compile(fn, backend="eager")
        # 执行编译后的函数并获取结果
        res = opt_fn(x)

    # TODO(anijain2305) - does not work with fullgraph=True
    def test_exception_with_ctx_manager(self):
        def fn(x):
            # 计算 x 的余弦值
            x = torch.cos(x)
            try:
                # 使用 torch.no_grad 上下文管理器
                with torch.no_grad():
                    # 计算 x 的正弦值
                    x = torch.sin(x)
                    # 抛出未实现错误
                    raise NotImplementedError("Not implemented")
            except NotImplementedError as e:
                # 计算 x 的 sigmoid 值
                x = torch.sigmoid(x)
            return x

        x = torch.randn(4)
        # 调用普通版本的 fn 函数，并获取结果
        ref = fn(x)
        # 使用 eager 模式编译 fn 函数
        opt_fn = torch.compile(fn, backend="eager")
        # 执行编译后的函数并获取结果
        res = opt_fn(x)
        # 断言编译前后的结果相等
        self.assertEqual(ref, res)

    def test_exception_raised_from_child(self):
        def gn():
            # 抛出未实现错误
            raise NotImplementedError("foo")

        def fn(x):
            # 计算 x 的余弦值
            x = torch.cos(x)
            try:
                # 计算 x 的正弦值
                x = torch.sin(x)
                # 调用 gn 函数，可能抛出异常
                gn()
                # 再次计算 x 的正弦值
                x = torch.sin(x)
            except Exception:
                # 计算 x 的 sigmoid 值
                x = torch.sigmoid(x)

            return x

        x = torch.randn(4)
        # 调用普通版本的 fn 函数，并获取结果
        ref = fn(x)
        # 使用 eager 模式编译 fn 函数，包括完整图形
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        # 执行编译后的函数并获取结果
        res = opt_fn(x)
        # 断言编译前后的结果相等
        self.assertEqual(ref, res)

    def test_nn_module_getattr(self):
        class A:
            def __init__(self):
                self._b = 20

            def __getattr__(self, name):
                # 如果属性名以 "_" 开头，返回对应属性值；否则抛出 AttributeError
                fixed_name = "_" + name
                if fixed_name in self.__dict__:
                    return self.__dict__[fixed_name]
                raise AttributeError(f"{name} absent")

        class B(A):
            def __init__(self):
                self.a = 10

            def __getattr__(self, name):
                try:
                    # 调用父类 A 的 __getattr__ 方法
                    return super().__getattr__(name)
                except AttributeError:
                    # 如果属性不存在，返回默认值 30
                    return 30

        obj = B()

        def fn(x):
            # 计算 x 与 obj 实例属性的乘积
            return x * obj.a * obj.b * obj.c

        x = torch.ones(4)
        # 调用普通版本的 fn 函数，并获取结果
        ref = fn(x)
        # 使用 eager 模式编译 fn 函数，包括完整图形
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        # 执行编译后的函数并获取结果
        res = opt_fn(x)
        # 断言编译前后的结果相等
        self.assertEqual(ref, res)

    @torch._dynamo.config.patch(inline_inbuilt_nn_modules=True)
    # 定义一个测试方法，用于测试自定义的 getattr 方法在模块异常时的行为
    def test_custom_getattr_on_module_exception(self):
        # 定义一个名为 Foo 的类，继承自 torch.nn.Module
        class Foo(torch.nn.Module):
            # 构造方法，初始化参数 a，默认为 3
            def __init__(self, a=3):
                super().__init__()
                # 注册参数 "a"，值为 torch.ones(4) * 2 的参数对象
                self.register_parameter("a", torch.nn.Parameter(torch.ones(4) * 2))

            # 自定义 getattr 方法
            def __getattr__(self, name):
                # 尝试调用父类的 getattr 方法处理逻辑
                try:
                    return super().__getattr__(name)  # 延迟至 nn.Module 的逻辑处理
                except AttributeError:
                    # 如果属性名为 "a_copy"，返回 self.a
                    if name == "a_copy":
                        return self.a
                    # 否则抛出 AttributeError 异常
                    raise

            # 前向传播方法
            def forward(self, x):
                # 返回 x 乘以 self.a 乘以 self.a_copy 的结果
                return x * self.a * self.a_copy

        # 创建一个 Foo 类的实例对象 mod
        mod = Foo()
        # 使用 torch.compile 方法编译 mod，使用 eager 后端和完整图模式
        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)

        # 创建一个包含全部元素为 1 的大小为 4 的张量 x
        x = torch.ones(4)
        # 断言 mod(x) 的结果与 opt_mod(x) 的结果相等
        self.assertEqual(mod(x), opt_mod(x))
# 如果该模块作为主程序执行（而不是被导入），则执行以下代码块
if __name__ == "__main__":
    # 从 torch._dynamo.test_case 模块导入 run_tests 函数
    from torch._dynamo.test_case import run_tests
    # 运行测试函数 run_tests()
    run_tests()
```