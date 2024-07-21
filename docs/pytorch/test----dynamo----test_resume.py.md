# `.\pytorch\test\dynamo\test_resume.py`

```py
# Owner(s): ["module: dynamo"]

# 导入 torch 库
import torch
# 导入 torch._dynamo.test_case 模块
import torch._dynamo.test_case

# 创建函数 fn_creator，返回一个函数 fn
def fn_creator():
    # 定义局部变量 var1 并赋值为 1
    var1 = 1

    # 定义函数 fn，接收参数 x
    def fn(x):
        # 将参数 x 增加 1
        x = x + 1
        # 定义并初始化局部变量 var2 为 1
        var2 = 1
        # 调用 torch._dynamo.graph_break() 方法
        torch._dynamo.graph_break()
        # 将 x 增加 var1 的值
        x = x + var1

        # 定义内部函数 inner_fn
        def inner_fn():
            # 返回 var2 的值
            return var2

        # 返回计算后的 x 值
        return x

    # 返回函数 fn
    return fn


# 定义测试类 ResumeFunctionTests，继承自 torch._dynamo.test_case.TestCase
class ResumeFunctionTests(torch._dynamo.test_case.TestCase):
    # 定义测试方法 test_freevars
    def test_freevars(self):
        # 调用 fn_creator() 函数，返回一个函数 fn
        fn = fn_creator()
        # 调用 torch.compile() 方法编译函数 fn，并指定后端为 "eager"
        opt_fn = torch.compile(fn, backend="eager")
        # 调用编译后的函数 opt_fn，并传入参数 torch.randn(10)
        opt_fn(torch.randn(10))
        # 获取全局变量中以 "__resume_at" 开头的键值对，并将值存入列表 codes
        codes = [v for k, v in list(globals().items()) if k.startswith("__resume_at")]
        # 断言 codes 的长度为 1
        self.assertEqual(len(codes), 1)
        # 断言 resume 函数的 co_freevars，它是原始函数 co_freevars 和 co_cellvars 的排序连接
        self.assertEqual(codes[0].co_freevars, ("var1", "var2"))


# 当脚本直接执行时
if __name__ == "__main__":
    # 从 torch._dynamo.test_case 模块导入 run_tests 函数，并执行
    from torch._dynamo.test_case import run_tests
    run_tests()
```