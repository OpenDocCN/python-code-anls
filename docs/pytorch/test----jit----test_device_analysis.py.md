# `.\pytorch\test\jit\test_device_analysis.py`

```py
# Owner(s): ["oncall: jit"]

# 引入单元测试模块
import unittest
# 引入 itertools 中的 product 函数
from itertools import product

# 引入 PyTorch 模块
import torch
# 引入 Torch JIT 中的属性传播模块
from torch.jit._passes._property_propagation import apply_input_props_using_example
# 引入 Torch 内部测试工具中的 TEST_CUDA
from torch.testing._internal.common_utils import TEST_CUDA
# 引入 Torch JIT 测试工具
from torch.testing._internal.jit_utils import JitTestCase

try:
    # 尝试引入 torchvision 中的 models 模块
    from torchvision import models
except ImportError:
    # 如果引入失败，设置 models 为 None
    models = None

# 如果当前脚本为主程序入口
if __name__ == "__main__":
    # 抛出运行时错误，提示不直接运行该测试文件
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义测试类 TestDeviceAnalysis，继承自 JitTestCase
class TestDeviceAnalysis(JitTestCase):
    @classmethod
    def setUpClass(cls):
        # 设置类变量 cpu, cuda, vulkan, mkldnn 分别为不同的 Torch 设备类型
        cls.cpu = torch.device("cpu")
        cls.cuda = torch.device("cuda")
        cls.vulkan = torch.device("vulkan")
        cls.mkldnn = torch.device(
            "mkldnn"
        )  # MKLDNN can't mix with other device types at all
        cls.device_types = [cls.cpu, cls.cuda, cls.vulkan]

    # 静态方法 node_output_device，用于获取图中输出节点的设备类型
    @staticmethod
    def node_output_device(graph):
        graph_out = list(graph.outputs())
        assert len(graph_out) == 1
        return graph_out[0].type().device()

    # 方法 prop_device_on_graph，用于在图中传播设备类型信息
    def prop_device_on_graph(self, graph, example_devices, in_shapes=None):
        graph_inputs = list(graph.inputs())
        # 擦除图中的形状信息
        torch._C._jit_pass_erase_shape_information(graph)

        # 断言图输入与示例设备数量相等
        self.assertEqual(len(graph_inputs), len(example_devices))
        # 遍历图输入与示例设备，设置输入节点的设备类型
        for graph_i, device_i in zip(graph_inputs, example_devices):
            if device_i is not None:
                graph_i.setType(graph_i.type().with_device(device_i))

        # 如果存在输入形状信息，则设置输入节点的形状信息
        if in_shapes:
            for graph_i, shapes_i in zip(graph_inputs, in_shapes):
                if shapes_i is not None:
                    graph_i.setType(graph_i.type().with_sizes(shapes_i))

            # 在图上传播形状信息
            torch._C._jit_pass_propagate_shapes_on_graph(graph)

        # 在图上传播设备类型信息
        torch._C._jit_pass_propagate_device(graph)

    # 方法 assert_device_equal，用于断言图的输出设备类型是否符合预期
    def assert_device_equal(
        self, fn, in_devices, expected_device, in_shapes=None, subtest_str=""
    ):
        with self.subTest(
            f"In device: {in_devices}, expected: {expected_device}, \n {subtest_str}"
        ):
            # 获取脚本化函数的图表示
            graph = torch.jit.script(fn).graph
            # 在图上传播设备类型信息并断言
            self.prop_device_on_graph(graph, in_devices, in_shapes)
            actual_device = self.node_output_device(graph)

            # 断言实际设备类型与预期设备类型相等
            if expected_device is None or actual_device is None:
                self.assertEqual(actual_device, expected_device)
            else:
                self.assertEqual(
                    actual_device.type, expected_device.type, "Failed Verification"
                )

    # 测试方法 test_device_apply，测试设备类型是否正确应用到输入
    def test_device_apply(self):
        # 定义一个简单的加法函数 add_self
        def add_self(x):
            return x + x

        # 获取加法函数的图表示
        graph = torch.jit.script(add_self).graph
        # 获取图的第一个输入节点
        graph_input = next(graph.inputs())
        # 设置输入节点的设备类型为 cpu
        graph_input.setType(graph_input.type().with_device(self.cpu))
        # 断言输入节点的设备类型为 cpu
        self.assertEqual(graph_input.type().device(), self.cpu)
    # 使用 unittest 模块中的装饰器 @unittest.skipIf 来标记该测试方法，如果 models 为 None，则跳过执行，需要 torchvision 库支持
    @unittest.skipIf(models is None, "Requires torchvision")
    def test_mobilenet(self):
        # 生成一个形状为 (1, 3, 224, 224) 的随机张量，设备为 self.cpu，并用它作为示例输入
        in_cpu = torch.randn(1, 3, 224, 224, device=self.cpu)
        in_example = in_cpu

        # 期望的设备是 self.cpu
        expected_device = self.cpu
        # 使用 torch.jit.script 将 mobilenet_v3_small 模型转换为 TorchScript 模型
        m = torch.jit.script(models.mobilenet_v3_small())
        # 将模型设置为评估模式
        m.eval()
        # 冻结 TorchScript 模型，获取其计算图
        graph = torch.jit.freeze(m).graph
        # 应用输入示例 in_example 到计算图中
        apply_input_props_using_example(graph, in_example)
        # 在计算图上传播形状信息
        torch._C._jit_pass_propagate_shapes_on_graph(graph)
        # 在计算图上传播设备信息
        torch._C._jit_pass_propagate_device(graph)

        # 获取节点的输出设备信息
        actual_device = self.node_output_device(graph)

        # 如果期望设备或实际设备有任何一个为 None，则断言它们相等
        if expected_device is None or actual_device is None:
            self.assertEqual(actual_device, expected_device)
        else:
            # 否则，断言实际设备的类型与期望设备的类型相同
            self.assertEqual(
                actual_device.type, expected_device.type, "Failed Verification"
            )

    # 测试简单的函数
    def test_simple(self):
        # 定义一个函数 add_self，将输入 x 加上自己返回
        def add_self(x):
            return x + x

        # 定义一个函数 relu_，应用 torch.nn.functional.relu_ 函数到输入 x 上返回
        def relu_(x):
            return torch.nn.functional.relu_(x)

        # 将函数 add_self 和 relu_ 存储在列表中
        functions = [add_self, relu_]

        # 对于每种输入设备类型和每个函数，使用 self.assert_device_equal 断言函数的设备等于输入设备
        for in_device, fn in product(self.device_types, functions):
            self.assert_device_equal(fn, [in_device], in_device)

    # 测试设置数据类型
    def test_set_dtype(self):
        # 定义一个函数 set_device，将输入张量 x 转换到 "cpu" 设备并返回
        def set_device(x):
            return x.to("cpu")

        # 对于每种输入设备类型，使用 self.assert_device_equal 断言 set_device 函数的设备等于 self.cpu
        for in_device in self.device_types:
            self.assert_device_equal(set_device, [in_device], self.cpu)

    # 测试带设备参数的函数
    def test_device_arg(self):
        # 定义一个函数 set_device，将输入张量 x 转换到指定的设备 device_name 并返回
        def set_device(x, device_name: torch.device):
            return x.to(device=device_name)

        # 对于每种输入设备类型，使用 self.assert_device_equal 断言 set_device 函数的设备等于 None
        for in_device in self.device_types:
            self.assert_device_equal(set_device, [in_device, None], None)

    # 测试张量作为函数参数
    def test_tensor_as_fns(self):
        # 定义函数 view_as_fn，将输入张量 x 作为输入 y 的视图返回
        def view_as_fn(x, y):
            return x.view_as(y)

        # 定义函数 expand_as_fn，将输入张量 x 扩展为输入 y 的形状返回
        def expand_as_fn(x, y):
            return x.expand_as(y)

        # 定义函数 reshape_as_fn，将输入张量 x 重塑为输入 y 的形状返回
        def reshape_as_fn(x, y):
            return x.reshape_as(y)

        # 对于每个测试函数 test_fn，使用 self.assert_device_equal 断言其设备等于预期设备
        for test_fn in [view_as_fn, expand_as_fn, reshape_as_fn]:
            # 在不同的设备组合下进行测试
            self.assert_device_equal(test_fn, [self.cpu, self.cpu], self.cpu)
            self.assert_device_equal(test_fn, [self.cuda, None], self.cuda)
            self.assert_device_equal(test_fn, [None, self.mkldnn], None)

        # 定义函数 type_as_fn，将输入张量 x 的类型设置为输入 y 的类型并返回
        def type_as_fn(x, y):
            return x.type_as(y)

        # 使用 self.assert_device_equal 断言 type_as_fn 函数的设备等于预期设备
        self.assert_device_equal(type_as_fn, [self.cpu, self.cpu], self.cpu)
        self.assert_device_equal(type_as_fn, [self.cuda, None], None)
        self.assert_device_equal(type_as_fn, [None, self.mkldnn], self.mkldnn)
    # 测试函数，用于测试支持非零维张量与零维张量的情况
    def zerodim_test_core(self, device_pairs):
        # 定义一个乘法函数
        def mul(x, y):
            return x * y

        # 定义一个加法函数
        def add(x, y):
            return x + y

        # 函数列表，包括乘法和加法
        fns = [mul, add]

        # 输入形状的列表，包括不同维度和零维张量的组合
        input_shapes = [
            ((1, 2, 2), (2, 2)),  # 不同维度，非零维张量
            ((1, 2, 2), ()),      # 一个零维张量
            ((), ()),             # 两个零维张量
        ]

        # 遍历函数、输入形状和设备对的笛卡尔积
        for fn, shapes, devices in product(fns, input_shapes, device_pairs):
            # 构建子测试字符串，包括函数名、输入形状和设备信息
            subtest_str = f"{fn.__name__} \n shapes: {shapes}, \n devices: {devices}"
            
            # 根据形状和设备创建随机张量
            in0 = torch.rand(shapes[0], device=devices[0])
            in1 = torch.rand(shapes[1], device=devices[1])

            try:
                # 调用函数计算输出
                out = fn(in0, in1)
            except Exception as e:
                # 对于 CPU 的零维张量，不期望立即失败
                for i in range(len(devices)):
                    if shapes[i] == () and devices[i] == self.cpu:
                        raise e
                
                # 仅期望在不同设备上出现的立即失败
                if devices[0] == devices[1]:
                    raise e
                
                # 预期失败时输出设备为 None
                self.assert_device_equal(fn, devices, None, shapes, subtest_str)
                continue
            
            # 断言输出设备与预期相等
            self.assert_device_equal(fn, devices, out.device, shapes, subtest_str)
            
            # 测试不带形状时，输出设备为相同设备或 None 的情况
            graph = torch.jit.script(fn).graph
            self.prop_device_on_graph(graph, devices)
            actual_device = self.node_output_device(graph)
            self.assertTrue(
                (actual_device is None) or (actual_device.type == out.device.type)
            )

    # 测试 CPU 上的零维张量
    def test_zerodim_cpu(self):
        # 在本地进行最小化测试
        self.zerodim_test_core([(self.cpu, self.cpu)])

    # 测试缺少设备时的零维张量情况
    def test_zerodim_no_device(self):
        # 定义一个乘法函数
        def mul(x, y):
            return x * y

        # 定义一个加法函数
        def add(x, y):
            return x + y

        # 函数列表，包括乘法和加法
        fns = [mul, add]

        # 设备对列表，包括其中一个设备为 None
        device_pairs = [
            (self.cpu, None),
            (None, self.cpu),
            (None, None),
        ]

        # 输入形状的列表，包括不同维度和零维张量的组合
        input_shapes = [
            ((1, 2, 2), (2, 2)),  # 不同维度，非零维张量
            ((1, 2, 2), ()),      # 一个零维张量
            ((), ()),             # 两个零维张量
        ]

        # 遍历函数、输入形状和设备对的笛卡尔积
        for fn, shapes, devices in product(fns, input_shapes, device_pairs):
            # 断言输出设备为 None
            self.assert_device_equal(fn, devices, None, shapes)

    # 如果支持 CUDA，则测试 GPU 上的零维张量
    @unittest.skipIf(not TEST_CUDA, "No CUDA")
    def test_zerodim_gpu(self):
        # GPU 设备对列表
        device_pairs = [
            (self.cpu, self.cuda),
            (self.cuda, self.cpu),
            (self.cuda, self.cuda),
        ]
        # 执行核心测试函数
        self.zerodim_test_core(device_pairs)
    def test_custom_device_op(self):
        # 测试自定义函数，检查正确应用设备类型

        # 定义将数据移动到 CUDA 设备的函数
        def set_cuda(x):
            return x.cuda()

        # 定义将数据移动到 CPU 设备的函数
        def set_cpu(x):
            return x.cpu()

        # 定义将数据移动到 MKLDNN 设备的函数
        def set_mkldnn(x):
            return x.to_mkldnn()

        # 定义不同设备对的函数对及其期望输出设备类型
        device_pairs = (
            (set_cuda, self.cuda),
            (set_cpu, self.cpu),
            (set_mkldnn, self.mkldnn),
        )

        # 遍历每个函数对及其对应的输入设备类型，检查输出设备类型是否符合预期
        for fn, out_device in device_pairs:
            for in_device in self.device_types:
                self.assert_device_equal(fn, [in_device], out_device)

    def test_device_if_propagation(self):
        # 测试条件语句对设备类型的影响

        # 定义带有条件语句的测试函数
        def test_fn(x, y, z: bool):
            if z:
                return x + 3
            else:
                return y * 2

        # 测试不同输入设备类型及条件对输出设备类型的影响
        self.assert_device_equal(test_fn, [self.cpu, self.cpu, None], self.cpu)
        self.assert_device_equal(test_fn, [self.mkldnn, self.mkldnn, None], self.mkldnn)
        self.assert_device_equal(test_fn, [self.cpu, self.cuda, None], None)

    def test_loop_simple(self):
        # 测试简单循环对设备类型的影响

        # 定义带有简单循环的测试函数
        def test_fn(x, y, z: int):
            for _ in range(z):
                y = x
            return y

        # 测试不同输入设备类型及循环次数对输出设备类型的影响
        self.assert_device_equal(test_fn, [self.cpu, self.cpu, None], self.cpu)
        self.assert_device_equal(test_fn, [self.cpu, self.cuda, None], None)
        self.assert_device_equal(test_fn, [self.cpu, None, None], None)

    def test_loop_device_change(self):
        # 测试循环中设备类型变更对输出设备类型的影响

        # 定义带有设备类型变更循环的测试函数
        def test_fn(x, z: int):
            for _ in range(z):
                x = x.cuda()
            return x

        # 测试不同输入设备类型及循环次数对输出设备类型的影响
        self.assert_device_equal(test_fn, [self.cpu, None], None)
        self.assert_device_equal(test_fn, [self.cuda, None], self.cuda)
        self.assert_device_equal(test_fn, [None, None], None)

    def test_while_change(self):
        # 测试 while 循环中设备类型变更对输出设备类型的影响

        # 定义带有设备类型变更 while 循环的测试函数
        def test_fn(x, z: int):
            while z > 0:
                x = x.cuda()
                z = 0
            return x

        # 测试不同输入设备类型及循环条件对输出设备类型的影响
        self.assert_device_equal(test_fn, [self.cpu, None], None)
        self.assert_device_equal(test_fn, [self.cuda, None], self.cuda)
        self.assert_device_equal(test_fn, [None, None], None)

    def test_nested_loops(self):
        # 测试嵌套循环对设备类型的影响

        # 定义带有嵌套循环的测试函数
        def test_fn(x, z: int):
            for i in range(z):
                x = x.cpu()
                for _ in range(i):
                    x = x + 1

            return x

        # 测试不同输入设备类型及循环嵌套深度对输出设备类型的影响
        self.assert_device_equal(test_fn, [self.cpu, None], self.cpu)
        self.assert_device_equal(test_fn, [self.cuda, None], None)
        self.assert_device_equal(test_fn, [None, None], None)
    # 定义一个测试函数，接受四个参数：x，y，z（布尔型），a（布尔型）
    def test_if_loop_mix(self):
        # 定义内部函数 test_fn，接受 x，y，z，a 四个参数
        def test_fn(x, y, z: bool, a: bool):
            # 将 x 赋值给 c
            c = x
            # 当 a 为真时进入循环
            while a:
                # 如果 z 为真，将 c 设为 x 加 3
                if z:
                    c = x + 3
                # 如果 z 为假，将 c 设为 y 乘以 2
                else:
                    c = y * 2
                # 将 a 设为假，退出循环
                a = False
            # 返回 c 的值
            return c

        # 调用 self.assert_device_equal 方法，测试 test_fn 函数
        self.assert_device_equal(test_fn, [self.cpu, self.cpu, None, None], self.cpu)
        # 再次调用 self.assert_device_equal 方法，测试 test_fn 函数
        self.assert_device_equal(
            test_fn, [self.mkldnn, self.mkldnn, None, None], self.mkldnn
        )
        # 第三次调用 self.assert_device_equal 方法，测试 test_fn 函数
        self.assert_device_equal(test_fn, [self.cpu, self.cuda, None, None], None)
```