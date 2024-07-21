# `.\pytorch\test\inductor\test_control_flow.py`

```py
# Owner(s): ["module: inductor"]

import itertools  # 导入 itertools 模块，用于生成迭代器的函数

import torch  # 导入 PyTorch 深度学习库
import torch._dynamo.testing  # 导入 PyTorch 的内部测试模块

from torch._inductor.test_case import TestCase  # 从 torch._inductor.test_case 模块导入 TestCase 类
from torch.testing._internal.common_utils import (  # 从 torch.testing._internal.common_utils 导入多个函数和类
    instantiate_parametrized_tests,  # 实例化参数化测试函数
    parametrize,  # 参数化装饰器
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU  # 从 torch.testing._internal.inductor_utils 导入 GPU 类型和硬件检测常量
from torch.testing._internal.triton_utils import requires_gpu  # 从 torch.testing._internal.triton_utils 导入需要 GPU 的装饰器


def _prepend_product_of_values(inputs, possible_values, num_to_prepend=1):
    result = []  # 初始化空列表用于存放结果
    device = inputs[0].device  # 获取输入张量的设备
    # 遍历谓词值的笛卡尔积
    for values in itertools.product(*([possible_values] * num_to_prepend)):
        prepended = [torch.tensor(v, device=device) for v in values]  # 根据值创建张量，并指定设备
        result.append((*prepended, *inputs))  # 将预置的张量与输入张量拼接，并添加到结果列表中
    return result  # 返回结果列表


def prepend_predicates(inputs, num_predicates=1):
    return _prepend_product_of_values(inputs, [False, True], num_predicates)  # 调用 _prepend_product_of_values 函数，用 False 和 True 预置输入张量


def prepend_counters(inputs, num_counters=1, counter_values=(0, 1, 5)):
    return _prepend_product_of_values(inputs, counter_values, num_counters)  # 调用 _prepend_product_of_values 函数，用指定的计数器值预置输入张量


class CondModels:
    class Simple(torch.nn.Module):
        def forward(self, p, a, b):
            def true_fn(x, y):
                return x + y  # 简单的加法操作

            def false_fn(x, y):
                return x - y  # 简单的减法操作

            return torch.cond(p, true_fn, false_fn, [a, b])  # 使用 torch.cond 根据 p 调用 true_fn 或 false_fn

    class Nested(torch.nn.Module):
        def forward(self, p0, p1, p2, a, b, c):
            def true_fn(x0, y0, z0):
                def true_true_fn(x1, y1, z1):
                    return (x1 - y1 * z1) * 3.14  # 复杂的数学运算

                def true_false_fn(x1, y1, z1):
                    def true_false_true_fn(x2, y2, z2):
                        return (x2 * y2 * z2) / 2.71  # 复杂的数学运算

                    def true_false_false_fn(x2, y2, z2):
                        return (x2 + y2 + z2) * 1.23  # 复杂的数学运算

                    return torch.cond(
                        p2, true_false_true_fn, true_false_false_fn, [x1, y1, z1]
                    )  # 使用 torch.cond 根据 p2 调用 true_false_true_fn 或 true_false_false_fn

                return torch.cond(p1, true_true_fn, true_false_fn, [x0, y0, z0])  # 使用 torch.cond 根据 p1 调用 true_true_fn 或 true_false_fn

            def false_fn(x0, y0, z0):
                def false_true_fn(x1, y1, z1):
                    def false_true_true_fn(x2, y2, z2):
                        return (x2 - y2 - z2) + 1.23  # 复杂的数学运算

                    def false_true_false_fn(x2, y2, z2):
                        return (x2 / y2 / z2) - 3.14  # 复杂的数学运算

                    return torch.cond(
                        p2, false_true_true_fn, false_true_false_fn, [x1, y1, z1]
                    )  # 使用 torch.cond 根据 p2 调用 false_true_true_fn 或 false_true_false_fn

                def false_false_fn(x1, y1, z1):
                    return (x1 - y1 * z1) / 2.71  # 复杂的数学运算

                return torch.cond(p1, false_true_fn, false_false_fn, [x0, y0, z0])  # 使用 torch.cond 根据 p1 调用 false_true_fn 或 false_false_fn

            return torch.cond(p0, true_fn, false_fn, [a, b, c])  # 使用 torch.cond 根据 p0 调用 true_fn 或 false_fn
    # 定义一个名为 Parameters 的神经网络模块
    class Parameters(torch.nn.Module):
        
        # 内部定义一个名为 InnerModel1 的神经网络模块
        class InnerModel1(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                # 使用线性层，输入大小为 20，输出大小为 30，设备为给定的设备
                self.layer = torch.nn.Linear(20, 30, device=device)

            # 前向传播函数
            def forward(self, x):
                # 对输入 x 进行操作：加 1，然后通过线性层并乘以 3.14
                return self.layer(x + 1) * 3.14

        # 内部定义另一个名为 InnerModel2 的神经网络模块
        class InnerModel2(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                # 使用两个线性层，输入大小为 20，输出大小为 10 和 30，设备为给定的设备
                self.layer1 = torch.nn.Linear(20, 10, device=device)
                self.layer2 = torch.nn.Linear(10, 30, device=device)

            # 前向传播函数
            def forward(self, x):
                # 对输入 x 进行操作：减 2，然后通过两个线性层并乘以 3.14
                return self.layer2(self.layer1(x - 2)) * 3.14

        # Parameters 类的初始化函数
        def __init__(self, device):
            super().__init__()
            # 创建 InnerModel1 实例，并存储为 true_fn
            self.true_fn = self.InnerModel1(device)
            # 创建 InnerModel2 实例，并存储为 false_fn
            self.false_fn = self.InnerModel2(device)

        # Parameters 类的前向传播函数
        def forward(self, p, a):
            # 根据条件 p，选择执行 true_fn 或 false_fn，并传递参数 [a]
            return torch.cond(p, self.true_fn, self.false_fn, [a])

    # 定义一个名为 ReinterpretView 的神经网络模块
    class ReinterpretView(torch.nn.Module):
        # ReinterpretView 类的前向传播函数
        def forward(self, p, a, b):
            # 定义一个返回 z1 和 z2 的函数 true_fn，接受 x 和 y 作为参数
            def true_fn(x, y):
                z1 = x + y
                z2 = x - y
                return z1[2:], z2[:, 4:]

            # 定义一个返回 z1 和 z2 的函数 false_fn，接受 x 和 y 作为参数
            def false_fn(x, y):
                z1 = x - y
                z2 = x + y
                return z1[2:], z2[:, 4:]

            # 根据条件 p，选择执行 true_fn 或 false_fn，并传递参数 [a[:-1], b[:-1]]
            return torch.cond(p, true_fn, false_fn, [a[:-1], b[:-1]])

    # 定义一个名为 MultipleOutputs 的神经网络模块
    class MultipleOutputs(torch.nn.Module):
        # MultipleOutputs 类的前向传播函数
        def forward(self, p, a, b, c):
            # 定义一个返回三个值的函数 true_fn，接受 x、y 和 z 作为参数
            def true_fn(x, y, z):
                return x * y, z / 2.71, (y - x).sum(dim=1)

            # 定义一个返回三个值的函数 false_fn，接受 x、y 和 z 作为参数
            def false_fn(x, y, z):
                return y / x, z * 3.14, (x + y).mean(dim=1)

            # 根据条件 p，选择执行 true_fn 或 false_fn，并传递参数 [a, b, c]
            return torch.cond(p, true_fn, false_fn, [a, b, c])

    # 定义一个名为 OuterCode 的神经网络模块
    class OuterCode(torch.nn.Module):
        # OuterCode 类的前向传播函数
        def forward(self, p, a, b):
            # 计算 c 和 d 的值
            c = a * b + 3.14
            d = a / b - 2.71

            # 定义一个返回 x + y 或 x - y 的函数 true_fn/false_fn，接受 x 和 y 作为参数
            def true_fn(x, y):
                return x + y

            def false_fn(x, y):
                return x - y

            # 根据条件 p，选择执行 true_fn 或 false_fn，并传递参数 [c, d]
            e = torch.cond(p, true_fn, false_fn, [c, d])

            # 返回 e*e 除以 1.41 的结果
            return e * e / 1.41

    # 定义一个名为 OuterBuffers 的神经网络模块
    class OuterBuffers(torch.nn.Module):
        # OuterBuffers 类的前向传播函数
        def forward(self, p, a, b, c):
            # 计算 d 和 e 的值
            d = a * 2
            e = b / 2

            # 定义一个返回 x+d 或 x-e 的函数 true_fn/false_fn，接受 x 作为参数
            def true_fn(x):
                return x + d

            def false_fn(x):
                return x - e

            # 根据条件 p，选择执行 true_fn 或 false_fn，并传递参数 [c]
            return torch.cond(p, true_fn, false_fn, [c])

    # 定义一个名为 WithNonTensorPredicate 的神经网络模块
    class WithNonTensorPredicate(torch.nn.Module):
        # WithNonTensorPredicate 类的前向传播函数
        def forward(self, a, b):
            # 定义一个返回 x.sum(0)/3.14 或 y.sum(0)*2.71 的函数 true_fn/false_fn，接受 x 和 y 作为参数
            def true_fn(x, y):
                return x.sum(0) / 3.14

            def false_fn(x, y):
                return y.sum(0) * 2.71

            # 根据条件 a.size(0) > b.size(0)，选择执行 true_fn 或 false_fn，并传递参数 [a, b]
            return torch.cond(a.size(0) > b.size(0), true_fn, false_fn, [a, b])
class CondTests(TestCase):
    def _run_test(
        self,
        model,
        inputs,
        device,
        dynamic=False,
        num_predicates=1,
    ):
        # 使用 torch._dynamo.testing.CompileCounterWithBackend 创建计数器 cnt
        cnt = torch._dynamo.testing.CompileCounterWithBackend("inductor")
        # 使用 torch.compile 将模型编译成 compiled_model
        compiled_model = torch.compile(backend=cnt, fullgraph=True)(model)

        # 将 inputs 中的每个张量移到指定设备上
        inputs = [inp.to(device=device) for inp in inputs]
        # 将 inputs 组成 input_sets 的列表
        input_sets = [inputs]
        # 如果 dynamic 为 True，则扩展输入数据集
        if dynamic:
            larger_inputs = []
            for inp in inputs:
                # 每个第一个维度扩展为 5 倍
                tiling = [5] + [1] * (inp.ndim - 1)
                larger_inputs.append(torch.tile(inp, tiling))
            input_sets.append(larger_inputs)
            # 对每个 input_sets 中的 inputs 进行动态标记
            for inputs in input_sets:
                for inp in inputs:
                    # 将每个第一个维度标记为动态
                    torch._dynamo.mark_dynamic(inp, 0)

        # 遍历 input_sets
        for inputs in input_sets:
            # 对每个输入添加谓词并进行预处理
            for inputs_with_predicates in prepend_predicates(inputs, num_predicates):
                # 克隆 inputs_with_predicates 中的每个输入张量
                cloned_inputs = [inp.clone() for inp in inputs_with_predicates]
                # 分别使用 model 和 compiled_model 计算结果
                result = model(*inputs_with_predicates)
                result_compiled = compiled_model(*inputs_with_predicates)
                # 断言克隆的输入与原始输入保持一致
                torch.testing.assert_close(cloned_inputs, inputs_with_predicates)
                # 断言 model 和 compiled_model 的结果一致
                torch.testing.assert_close(result, result_compiled)

        # 断言只进行了一次编译
        self.assertEqual(cnt.frame_count, 1, "only one compilation expected")

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_cond_simple_control_flow(self, device, dynamic):
        # 测试简单条件控制流，调用 _run_test 方法
        self._run_test(
            model=CondModels.Simple(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_cond_nested_control_flow(self, device, dynamic):
        # 测试嵌套条件控制流，调用 _run_test 方法
        self._run_test(
            model=CondModels.Nested(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
            num_predicates=3,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_cond_outer_code_before_after(self, device, dynamic):
        # 测试条件外部代码前后的情况，调用 _run_test 方法
        self._run_test(
            model=CondModels.OuterCode(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_cond_multiple_outputs(self, device, dynamic):
        # 使用@parametrize装饰器为测试方法添加参数化支持，dynamic参数分别测试False和True两种情况
        # 测试多个具有不同形状的输出
        self._run_test(
            model=CondModels.MultipleOutputs(),  # 创建CondModels.MultipleOutputs模型实例
            inputs=(
                torch.randn(10, 20),  # 输入张量1，大小为10x20
                torch.randn(10, 20),  # 输入张量2，大小为10x20
                torch.randn(30, 40),  # 输入张量3，大小为30x40
            ),
            device=device,  # 指定设备
            dynamic=dynamic,  # 指定dynamic参数的值
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    def test_cond_advanced_dynamic_shapes(self, device):
        # 使用@requires_gpu装饰器，表明需要GPU支持
        # 测试中包含符号表达式的子图输入形状
        class Model(torch.nn.Module):
            def forward(self, p, a, b):
                def true_fn(x, y):
                    return torch.cat([x - 3, y * 3], dim=1)  # 真条件函数

                def false_fn(x, y):
                    return torch.cat([x / 3, y - 3], dim=1)  # 假条件函数

                c = torch.cat([a, b], dim=0)  # 拼接输入a和b
                d = c * 2  # d为c的两倍
                e = c / 2  # e为c的一半

                return torch.cond(p, true_fn, false_fn, [d, e])  # 根据p条件选择执行true_fn或false_fn

        self._run_test(
            model=Model(),  # 创建Model模型实例
            inputs=(
                torch.randn(2, 3, 3),  # 输入张量1，大小为2x3x3
                torch.randn(4, 3, 3),  # 输入张量2，大小为4x3x3
            ),
            device=device,  # 指定设备
            dynamic=True,  # 指定dynamic参数为True
        )

    @requires_gpu
    def test_cond_use_buffers_from_outer_scope(self):
        # 使用@requires_gpu装饰器，表明需要GPU支持
        # 测试中包含符号表达式的子图输入形状
        self._run_test(
            model=CondModels.OuterBuffers(),  # 创建CondModels.OuterBuffers模型实例
            inputs=(
                torch.randn(10, 20),  # 输入张量1，大小为10x20
                torch.randn(10, 20),  # 输入张量2，大小为10x20
                torch.randn(10, 20),  # 输入张量3，大小为10x20
            ),
            device=GPU_TYPE,  # 指定设备为GPU_TYPE
            dynamic=False,  # 指定dynamic参数为False
        )

    @requires_gpu
    def test_cond_reintepret_view_inputs_outputs(self):
        # 使用@requires_gpu装饰器，表明需要GPU支持
        # 在子图的输入和输出中重新解释视图
        self._run_test(
            model=CondModels.ReinterpretView(),  # 创建CondModels.ReinterpretView模型实例
            inputs=(
                torch.randn(10, 20),  # 输入张量1，大小为10x20
                torch.randn(10, 20),  # 输入张量2，大小为10x20
            ),
            device=GPU_TYPE,  # 指定设备为GPU_TYPE
            dynamic=True,  # 指定dynamic参数为True
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_cond_subgraphs_with_parameters(self, device, dynamic):
        # 使用@requires_gpu装饰器，表明需要GPU支持
        # 使用@parametrize装饰器为测试方法添加参数化支持，device和dynamic参数均进行参数化测试
        # 包含参数的嵌套模块
        self._run_test(
            model=CondModels.Parameters(device),  # 创建CondModels.Parameters模型实例，使用device参数
            inputs=(torch.randn(10, 20),),  # 输入张量，大小为10x20
            device=device,  # 指定设备
            dynamic=dynamic,  # 指定dynamic参数的值
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    # 测试非张量谓词条件下的情况
    def test_cond_non_tensor_predicates(self, device, dynamic):
        # 使用不同的批量大小进行测试
        for b_size_0 in [5, 15]:
            # 重置 Torch Dynamo 状态
            torch._dynamo.reset()
            # 运行测试，传入模型、输入数据、设备和动态标志
            self._run_test(
                model=CondModels.WithNonTensorPredicate(),
                inputs=(
                    torch.randn(10, 20),
                    torch.randn(b_size_0, 20),
                ),
                device=device,
                dynamic=dynamic,
                num_predicates=0,
            )

    @requires_gpu
    # 测试子图中的输出别名问题
    def test_cond_aliasing_outputs(self):
        # 在子图中不支持输出别名
        class Model(torch.nn.Module):
            def forward(self, p, a, b):
                # 真条件函数
                def true_fn(x, y):
                    z = x + y
                    return z, z[1:]

                # 假条件函数
                def false_fn(x, y):
                    z = x - y
                    return z, z[1:]

                # 调用 torch.cond 来根据条件 p 执行不同的函数
                return torch.cond(p, true_fn, false_fn, [a, b])

        # 预期抛出 AssertionError: 输出别名目前不支持...
        with self.assertRaises(torch._dynamo.exc.BackendCompilerFailed):
            # 编译模型并调用
            torch.compile(Model())(
                torch.tensor(True),
                torch.randn(10, 20),
                torch.randn(10, 20),
            )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    # 测试在子图中分解操作的情况
    def test_cond_decompose_ops_in_subgraph(self, device):
        class Model(torch.nn.Module):
            def forward(self, p, a):
                # 真条件函数
                def true_fn(x):
                    return torch.zeros_like(x)

                # 假条件函数
                def false_fn(x):
                    return torch.ones_like(x)

                # 创建输入张量 b，并根据条件 p 调用 torch.cond 来执行不同的函数
                b = torch.ones_like(a)
                c = torch.cond(p, true_fn, false_fn, [b])
                return c

        # 运行测试，传入模型、输入数据和设备
        self._run_test(
            model=Model(),
            inputs=(torch.rand(10, 20),),
            device=device,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    # 测试在子图中递归分解操作的情况
    def test_cond_decompose_ops_in_subgraph_recursive(self, device):
        # 定义内部条件函数
        def inner_fn1(x):
            return torch.zeros_like(x)

        def inner_fn2(x):
            return torch.ones_like(x)

        class Model(torch.nn.Module):
            def forward(self, p, a):
                # 真条件函数
                def true_fn(x):
                    # 根据条件 p 调用不同的内部函数
                    return torch.cond(p, inner_fn2, inner_fn1, [x])

                # 假条件函数
                def false_fn(x):
                    # 根据条件 p 调用不同的内部函数
                    return torch.cond(p, inner_fn1, inner_fn2, [x])

                # 创建输入张量 b，并根据条件 p 调用 torch.cond 来执行不同的函数
                b = torch.ones_like(a)
                c = torch.cond(p, true_fn, false_fn, [b])
                return c

        # 运行测试，传入模型、输入数据和设备
        self._run_test(
            model=Model(),
            inputs=(torch.rand(10, 20),),
            device=device,
        )
    # 定义一个测试方法，用于验证条件感知图在递归应用时是否通过
    def test_cond_inductor_fx_passes_recursively_applied(self):
        # 初始化计数器字典，用于记录预处理和后处理方法的调用次数
        counters = {"pre_grad": 0, "post_grad": 0}

        # 定义预处理方法，每次调用计数器"pre_grad"加一
        def pre_grad_pass_counter(gm):
            counters["pre_grad"] += 1

        # 定义后处理方法，每次调用计数器"post_grad"加一
        def post_grad_pass_counter(gm):
            counters["post_grad"] += 1

        # 使用torch._inductor.config.patch方法进行上下文管理，设置定制的预处理和后处理方法
        with torch._inductor.config.patch(
            {
                "pre_grad_custom_pass": pre_grad_pass_counter,
                "post_grad_custom_pre_pass": post_grad_pass_counter,
                # 上述补丁不会被pickle序列化保存
                "fx_graph_cache": False,
            }
        ):
            # 运行测试方法_run_test，传入模型、输入数据、设备类型、是否动态计算、断言数量等参数
            self._run_test(
                model=CondModels.Nested(),
                inputs=(
                    torch.randn(10, 20),
                    torch.randn(10, 20),
                    torch.randn(10, 20),
                ),
                device=GPU_TYPE,
                dynamic=True,
                num_predicates=3,
            )

        # 断言预处理和后处理方法各自被调用11次
        self.assertEqual(counters["pre_grad"], 11)
        self.assertEqual(counters["post_grad"], 11)
class WhileLoopModels:
    class Simple(torch.nn.Module):
        def forward(self, ci, a, b):
            # 定义条件函数，返回 i > 0 条件结果
            def cond_fn(i, x, y):
                return i > 0

            # 定义循环体函数，返回更新后的 i, x+y, y-x
            def body_fn(i, x, y):
                return i - 1, x + y, y - x

            # 调用 PyTorch 提供的 while_loop 函数执行循环
            return torch._higher_order_ops.while_loop(cond_fn, body_fn, [ci, a, b])

    class Nested(torch.nn.Module):
        def forward(self, ci, cj, a, b):
            # 定义外层循环的条件函数，返回 i1 > 0 条件结果
            def cond_fn(i1, j1, x1, y1):
                return i1 > 0

            # 定义外层循环的循环体函数
            def body_fn(i1, j1, x1, y1):
                # 定义内层循环的条件函数，返回 j2 > 0 条件结果
                def cond_fn_nested(i2, j2, x2, y2):
                    return j2 > 0

                # 定义内层循环的循环体函数，返回更新后的 i2, j2-1, x2+3.14, y2-2.71
                def body_fn_nested(i2, j2, x2, y2):
                    return i2.clone(), j2 - 1, x2 + 3.14, y2 - 2.71

                # 调用 PyTorch 提供的 while_loop 函数执行内层循环
                i1, j1, x1, y1 = torch._higher_order_ops.while_loop(
                    cond_fn_nested, body_fn_nested, [i1, j1, x1, y1]
                )

                # 返回外层循环更新后的 i1-1, j1, x1*2, y1/2
                return i1 - 1, j1.clone(), x1 * 2, y1 / 2

            # 调用 PyTorch 提供的 while_loop 函数执行外层循环
            return torch._higher_order_ops.while_loop(cond_fn, body_fn, (ci, cj, a, b))

    class Parameters(torch.nn.Module):
        class InnerModel(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                # 定义神经网络模型的层
                self.layer1 = torch.nn.Linear(20, 30, device=device)
                self.layer2 = torch.nn.Linear(30, 20, device=device)

            def forward(self, c, x):
                # 返回更新后的 c-1, 对 layer2 对 layer1(x-2) 的输出乘以 3.14
                return c - 1, self.layer2(self.layer1(x - 2)) * 3.14

        def __init__(self, device):
            super().__init__()
            # 初始化内部模型
            self.body_fn = self.InnerModel(device)
            # 定义条件函数，lambda 表达式，返回 c > 0 条件结果
            self.cond_fn = lambda c, x: c > 0

        def forward(self, c, a):
            # 调用 PyTorch 提供的 while_loop 函数执行循环
            return torch._higher_order_ops.while_loop(
                self.cond_fn, self.body_fn, [c, a]
            )

    class OuterCode(torch.nn.Module):
        def forward(self, c, a, b):
            # 计算 d 和 e
            d = a * b + 3.14
            e = a / b - 2.71

            # 定义条件函数，返回 c > 0 条件结果
            def cond_fn(c, x, y):
                return c > 0

            # 定义循环体函数，返回更新后的 c-1, y-x, x+y
            def body_fn(c, x, y):
                return c - 1, y - x, x + y

            # 调用 PyTorch 提供的 while_loop 函数执行循环
            _, f, g = torch._higher_order_ops.while_loop(cond_fn, body_fn, [c, d, e])

            # 返回 f*g/1.41 的结果
            return f * g / 1.41

    # TODO(aakhundov): add while_loop test with outer buffers
    # with dynamic=True once dynamo / export allows while_loop
    # closure capture with mark_dynamic:
    # https://github.com/pytorch/pytorch/issues/123596
    class OuterBuffers(torch.nn.Module):
        def forward(self, c, a, b):
            # 计算 d 和 e
            d = a * 2
            e = b / 2

            # 定义条件函数，返回 c > 0 条件结果
            def cond_fn(c, x, y):
                return c > 0

            # 定义循环体函数，返回更新后的 c-1, x+d, y-e
            def body_fn(c, x, y):
                return c - 1, x + d, y - e

            # 调用 PyTorch 提供的 while_loop 函数执行循环
            return torch._higher_order_ops.while_loop(cond_fn, body_fn, [c, a, b])


class WhileLoopTests(TestCase):
    def _run_test(
        self,
        model,
        inputs,
        device,
        dynamic=False,
        num_counters=1,
    ):
        # 使用 CompileCounterWithBackend 类创建一个计数器对象 cnt，指定后端为 "inductor"
        cnt = torch._dynamo.testing.CompileCounterWithBackend("inductor")
        # 使用指定的后端编译模型，返回编译后的模型对象 compiled_model
        compiled_model = torch.compile(backend=cnt, fullgraph=True)(model)

        # 将输入数据列表中的每个张量移到指定设备上
        inputs = [inp.to(device=device) for inp in inputs]
        # 将 inputs 列表封装为 input_sets 列表的第一个元素
        input_sets = [inputs]
        if dynamic:
            # 如果 dynamic 为 True，则进行动态输入处理
            larger_inputs = []
            for inp in inputs:
                # 对每个输入张量在第一维度上进行 5 倍的复制
                tiling = [5] + [1] * (inp.ndim - 1)
                larger_inputs.append(torch.tile(inp, tiling))
            # 将复制后的 larger_inputs 添加到 input_sets 列表作为第二个元素
            input_sets.append(larger_inputs)
            for inputs in input_sets:
                for inp in inputs:
                    # 对每个输入张量的第一维度标记为动态
                    if inp.ndim:
                        torch._dynamo.mark_dynamic(inp, 0)

        # 遍历 input_sets 列表中的每个输入集合
        for inputs in input_sets:
            # 对每个 inputs_with_counters 在 prepend_counters 函数处理后的结果集合执行循环
            for inputs_with_counters in prepend_counters(inputs, num_counters):
                # 克隆 inputs_with_counters 中的每个张量，确保不改变原始输入
                cloned_inputs = [inp.clone() for inp in inputs_with_counters]
                # 使用模型对 inputs_with_counters 进行前向推断，返回结果 result
                result = model(*inputs_with_counters)
                # 使用编译后的模型对 inputs_with_counters 进行前向推断，返回结果 result_compiled
                with torch.no_grad():
                    result_compiled = compiled_model(*inputs_with_counters)
                # 使用 torch.testing.assert_close 检查 inputs_with_counters 和 cloned_inputs 的接近程度
                # 以及 result 和 result_compiled 的接近程度，使用绝对和相对误差为 1e-4
                torch.testing.assert_close(cloned_inputs, inputs_with_counters)
                torch.testing.assert_close(
                    result, result_compiled, atol=1e-4, rtol=1e-4
                )

        # 使用 self.assertEqual 检查 cnt.frame_count 是否为 1，确保只进行了一次编译
        self.assertEqual(cnt.frame_count, 1, "only one compilation expected")
    # 定义测试函数，测试带参数的 while 循环控制流
    def test_while_loop_with_parameters(self, device, dynamic):
        # 调用 _run_test 方法运行测试，使用 Parameters 类创建模型
        self._run_test(
            model=WhileLoopModels.Parameters(device),
            # 提供随机张量作为输入
            inputs=(torch.randn(10, 20),),
            # 设备参数
            device=device,
            # 是否动态执行的参数
            dynamic=dynamic,
        )

    # 使用装饰器设置 GPU 环境要求，设备参数为 "cpu" 或 GPU_TYPE
    # dynamic=True 现在无法工作，由于 https://github.com/pytorch/pytorch/issues/123596
    # 使用装饰器设置 dynamic 参数为 False
    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    def test_while_loop_with_outer_buffers(self, device, dynamic):
        # 调用 _run_test 方法运行测试，使用 OuterBuffers 类创建模型
        self._run_test(
            model=WhileLoopModels.OuterBuffers(),
            # 提供两个随机张量作为输入
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            # 设备参数
            device=device,
            # 是否动态执行的参数
            dynamic=dynamic,
        )
# 使用给定的参数化测试类实例化参数化测试，这里使用了 CondTests 类
instantiate_parametrized_tests(CondTests)

# 使用给定的参数化测试类实例化参数化测试，这里使用了 WhileLoopTests 类
instantiate_parametrized_tests(WhileLoopTests)

# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 从 torch._inductor.test_case 模块导入 run_tests 函数
    from torch._inductor.test_case import run_tests

    # 如果已经配置了 CPU 或 GPU 环境变量
    if HAS_CPU or HAS_GPU:
        # 运行测试，设置需要的依赖为 "filelock"
        run_tests(needs="filelock")
```