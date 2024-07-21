# `.\pytorch\test\dynamo\test_optimizers.py`

```
"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_adam in OptimizerTests)
"""
# 导入 functools 模块，用于函数式编程支持
import functools

# Owner(s): ["module: dynamo"]

# 导入 torch 库
import torch

# 导入 torch._dynamo 模块
import torch._dynamo
# 导入 torch._dynamo.test_case 模块
import torch._dynamo.test_case
# 导入 torch._dynamo.testing 模块
import torch._dynamo.testing
# 从 torch.nn 模块导入 Parameter 类
from torch.nn import Parameter

# 自定义优化器类 MyOptimizer，继承自 torch.optim.Optimizer
class MyOptimizer(torch.optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, {})

    # 初始化参数组的私有方法
    def _init_group(self, params, group):
        # 初始化任意复杂性标志为 False
        any_complex = False
        # 遍历组中的参数列表
        for p in group["params"]:
            # 将参数 p 添加到 params 列表中
            params.append(p)
            # 如果参数 p 是复数类型，则将 any_complex 设为 True
            any_complex |= p.is_complex()
        return any_complex

    # 梯度更新步骤
    def step(self):
        # 遍历所有参数组
        for group in self.param_groups:
            params = []
            # 初始化当前组的参数列表，并检查是否有复杂类型参数
            any_complex = self._init_group(params, group)
            # 如果存在复杂类型参数，将第一个参数减去 1；否则加上 1
            if any_complex:
                params[0] -= 1
            else:
                params[0] += 1


# End2EndTests 类，继承自 torch._dynamo.test_case.TestCase
class End2EndTests(torch._dynamo.test_case.TestCase):
    # https://github.com/pytorch/torchdynamo/issues/1604
    # 测试优化包含 requires_grad 的张量
    def test_optimizing_over_tensor_with_requires_grad(self):
        # 定义一个简单的神经网络模型类 Net
        class Net(torch.nn.Module):
            def forward(self, x, y):
                z = torch.bmm(x, y)
                z = torch.flatten(z, 1)
                return z

        # 训练迭代函数
        def training_iter_fn(batch, model, optimizer):
            optimizer.zero_grad()
            out = model(**batch)
            target = torch.tensor([0, 7])
            # 计算交叉熵损失
            loss = torch.nn.CrossEntropyLoss()(out, target)
            loss.backward()
            optimizer.step()
            return loss

        # 创建网络实例
        net = Net()
        # 创建两个张量作为输入
        input1 = torch.randn(2, 1, 4)
        input2 = torch.randn(2, 4, 8, requires_grad=True)
        # 使用 Adam 优化器优化 input2
        optimizer = torch.optim.Adam([input2], lr=0.1)

        # 编译计数器对象
        cnts = torch._dynamo.testing.CompileCounter()
        # 优化 training_iter_fn 函数
        opt_training_iter_fn = torch._dynamo.optimize(cnts)(training_iter_fn)
        batch = {"x": input1, "y": input2}
        # 进行两次优化迭代
        for _ in range(2):
            opt_training_iter_fn(batch, net, optimizer)
        # 断言编译帧数为 2
        self.assertEqual(cnts.frame_count, 2)

    # 测试状态字典的函数
    def test_state_dict(self):
        # 使用 torch.compile 注解，指定后端为 eager
        @torch.compile(backend="eager")
        def _test_state_dict(weight, bias, input):
            # 定义基础函数 fn_base
            def fn_base(optimizer, weight, bias):
                optimizer.zero_grad()
                i = input
                # 计算损失函数并反向传播
                loss = (weight.mv(i) + bias).pow(2).sum()
                loss.backward()
                return loss

            # 使用 Adagrad 优化 weight 和 bias
            optimizer = torch.optim.Adagrad([weight, bias])
            # 使用 functools.partial 创建 fn 函数
            fn = functools.partial(fn_base, optimizer, weight, bias)
            return optimizer, fn

        # 调用 _test_state_dict 函数，获取 optimizer 和 fn
        optimizer, fn = _test_state_dict(
            Parameter(torch.randn(10, 5)),
            Parameter(torch.randn(10)),
            torch.randn(5, requires_grad=True),
        )
        # 执行优化步骤
        optimizer.step(fn)
    # 定义测试函数 test_init_group，用于测试初始化参数组功能
    def test_init_group(self):
        # 遍历数据类型列表，包括 torch.float32 和 torch.cfloat
        for dtype in [torch.float32, torch.cfloat]:
            # 创建一个大小为 5x5 的张量 tensor，指定数据类型为当前循环的 dtype
            tensor = torch.randn(5, 5, dtype=dtype)
            # 使用 tensor 的克隆副本创建 Parameter 对象 params，并指定不需要梯度
            params = Parameter(tensor.detach().clone(), requires_grad=False)
            # 使用 tensor 的克隆副本创建另一个 Parameter 对象 opt_params，并同样指定不需要梯度
            opt_params = Parameter(tensor.detach().clone(), requires_grad=False)

            # 创建 MyOptimizer 对象 optim，用于优化 params 参数
            optim = MyOptimizer([params])
            # 调用优化器的 step 方法，执行优化步骤
            optim.step()

            # 创建另一个 MyOptimizer 对象 opt_optim，用于优化 opt_params 参数
            opt_optim = MyOptimizer([opt_params])
            # 使用 torch.compile 函数编译 opt_optim 对象的 step 方法，指定后端为 "eager"，生成完整图形
            opt_step = torch.compile(backend="eager", fullgraph=True)(opt_optim.step)
            # 执行编译后的优化步骤
            opt_step()

            # 使用断言方法 self.assertEqual 检查 params 和 opt_params 是否相等
            self.assertEqual(params, opt_params)
if __name__ == "__main__":
    # 如果当前模块是作为主程序运行时执行以下代码
    from torch._dynamo.test_case import run_tests
    # 从 torch._dynamo.test_case 模块导入 run_tests 函数

    # 运行测试函数
    run_tests()
```