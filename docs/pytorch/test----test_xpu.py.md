# `.\pytorch\test\test_xpu.py`

```
# Owner(s): ["module: intel"]

# 导入必要的库和模块
import collections  # 导入collections模块
import subprocess  # 导入subprocess模块
import sys  # 导入sys模块
import tempfile  # 导入tempfile模块
import unittest  # 导入unittest模块

import torch  # 导入torch库
import torch.xpu._gpu_trace as gpu_trace  # 导入gpu_trace模块
from torch.testing._internal.autocast_test_lists import AutocastTestLists  # 导入AutocastTestLists类
from torch.testing._internal.common_device_type import (  # 导入common_device_type模块的函数和类
    instantiate_device_type_tests,
    onlyXPU,
    OpDTypes,
    ops,
)
from torch.testing._internal.common_methods_invocations import ops_and_refs  # 导入ops_and_refs函数
from torch.testing._internal.common_utils import (  # 导入common_utils模块的函数和类
    NoTest,
    run_tests,
    suppress_warnings,
    TEST_WITH_UBSAN,
    TEST_XPU,
    TestCase,
)
from torch.utils.checkpoint import checkpoint_sequential  # 导入checkpoint_sequential函数

# 如果不测试XPU，打印信息并设置TestCase为NoTest
if not TEST_XPU:
    print("XPU not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811

# 检查是否有多个XPU设备
TEST_MULTIXPU = torch.xpu.device_count() > 1

# 定义CPU和XPU设备对象
cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

# 定义通用的CPU和XPU设备操作类型
any_common_cpu_xpu_one = OpDTypes.any_common_cpu_cuda_one

# 定义XPU计算操作列表
_xpu_computation_op_list = [
    "fill",
    "zeros",
    "zeros_like",
    "clone",
    "view_as_real",
    "view_as_complex",
    "view",
    "resize_",
    "resize_as_",
    "add",
    "sub",
    "mul",
    "div",
    "abs",
]

# 定义XPU张量工厂操作列表
_xpu_tensor_factory_op_list = [
    "as_strided",
    "empty",
    "empty_strided",
]

# 定义不测试dtype的XPU操作列表
_xpu_not_test_dtype_op_list = [
    "resize_",   # CPU不执行的操作
    "resize_as_",   # CPU不执行的操作
    "abs",   # 不支持的dtype
]

# 合并所有XPU操作列表
_xpu_all_op_list = _xpu_computation_op_list + _xpu_tensor_factory_op_list

# 根据操作和引用过滤XPU操作列表
_xpu_all_ops = [op for op in ops_and_refs if op.name in _xpu_all_op_list]

# 筛选出XPU计算操作列表
_xpu_computation_ops = [
    op for op in ops_and_refs if op.name in _xpu_computation_op_list
]


class TestXpu(TestCase):
    # 测试XPU设备行为
    def test_device_behavior(self):
        current_device = torch.xpu.current_device()  # 获取当前XPU设备
        torch.xpu.set_device(current_device)  # 设置当前设备为当前XPU设备
        self.assertEqual(current_device, torch.xpu.current_device())  # 断言当前设备与XPU当前设备相同

    # 测试多XPU设备行为（仅在检测到多个XPU时执行）
    @unittest.skipIf(not TEST_MULTIXPU, "only one GPU detected")
    def test_multi_device_behavior(self):
        current_device = torch.xpu.current_device()  # 获取当前XPU设备
        target_device = (current_device + 1) % torch.xpu.device_count()  # 计算目标XPU设备

        # 使用上下文管理器将操作切换到目标XPU设备，然后断言当前XPU设备为目标设备
        with torch.xpu.device(target_device):
            self.assertEqual(target_device, torch.xpu.current_device())

        # 再次断言当前XPU设备为之前的设备
        self.assertEqual(current_device, torch.xpu.current_device())

        # 使用_XPU_DeviceGuard上下文管理器将操作切换到目标XPU设备，然后断言当前XPU设备为目标设备
        with torch.xpu._DeviceGuard(target_device):
            self.assertEqual(target_device, torch.xpu.current_device())

        # 再次断言当前XPU设备为之前的设备
        self.assertEqual(current_device, torch.xpu.current_device())
    def test_get_device_properties(self):
        # 获取当前的 XPU 设备
        current_device = torch.xpu.current_device()
        # 获取当前设备的属性信息
        device_properties = torch.xpu.get_device_properties(current_device)
        # 断言获取的设备属性与 None 设备的属性相同
        self.assertEqual(device_properties, torch.xpu.get_device_properties(None))
        # 断言获取的设备属性与未指定设备的属性相同
        self.assertEqual(device_properties, torch.xpu.get_device_properties())

        # 获取当前设备的名称
        device_name = torch.xpu.get_device_name(current_device)
        # 断言获取的设备名称与 None 设备的名称相同
        self.assertEqual(device_name, torch.xpu.get_device_name(None))
        # 断言获取的设备名称与未指定设备的名称相同
        self.assertEqual(device_name, torch.xpu.get_device_name())

        # 获取当前设备的计算能力
        device_capability = torch.xpu.get_device_capability(current_device)
        # 断言最大工作组大小大于 0
        self.assertTrue(device_capability["max_work_group_size"] > 0)
        # 断言最大子组数大于 0
        self.assertTrue(device_capability["max_num_sub_groups"] > 0)
        # 断言设备属性中的驱动版本与计算能力中的驱动版本相同
        self.assertEqual(
            device_properties.driver_version, device_capability["driver_version"]
        )
        # 断言设备属性中是否支持 FP16 与计算能力中的是否支持 FP16 相同
        self.assertEqual(device_properties.has_fp16, device_capability["has_fp16"])
        # 断言设备属性中是否支持 FP64 与计算能力中的是否支持 FP64 相同
        self.assertEqual(device_properties.has_fp64, device_capability["has_fp64"])
        # 断言设备属性中是否支持原子操作（64位）与计算能力中的是否支持相同
        self.assertEqual(
            device_properties.has_atomic64, device_capability["has_atomic64"]
        )

    def test_wrong_xpu_fork(self):
        # 运行测试用例，并捕获标准错误流
        stderr = TestCase.runWithPytorchAPIUsageStderr(
            """\
import torch  # 导入 PyTorch 库
from torch.multiprocessing import Process  # 导入多进程模块 Process

def run(rank):
    torch.xpu.set_device(rank)  # 设置当前进程的 XPU 设备

if __name__ == "__main__":
    size = 2  # 定义进程数量
    processes = []  # 创建进程列表

    for rank in range(size):
        # 下面这行代码似乎是多余的，因为后面会为每个进程设置正确的设备
        torch.xpu.set_device(0)

        # 创建进程对象，指定运行函数为 run，传入进程编号作为参数
        p = Process(target=run, args=(rank,))
        p.start()  # 启动进程
        processes.append(p)  # 将进程对象加入进程列表

    for p in processes:
        p.join()  # 等待所有进程结束

"""
        )
        self.assertRegex(stderr, "Cannot re-initialize XPU in forked subprocess.")

    def test_lazy_init(self):
        """Validate that no XPU calls are made during `import torch` call"""

        def check_output(script: str) -> str:
            return (
                subprocess.check_output([sys.executable, "-c", script])
                .decode("ascii")
                .strip()
            )

        test_script = """\
import torch
from torch.multiprocessing import Process
import copy

def run_model(model, input):
    input_xpu = input.clone().to('xpu')
    model_xpu = copy.deepcopy(model).to('xpu')
    loss_xpu = model_xpu(input_xpu).sum()
    loss = model(input).sum()
    assert torch.allclose(loss_xpu.cpu(), loss)

def test_multi_process(model, input):
    p = Process(target=run_model, args=(model, input))
    p.start()
    p.join()
    assert p.exitcode == 0

input = torch.rand(1, 4, 16, 16)
model = torch.nn.Sequential(
    torch.nn.Conv2d(4, 2, 1, stride=2),
    torch.nn.BatchNorm2d(2, eps=1e-05, momentum=0.1),
)
test_multi_process(model, input)
test_multi_process(model, input)
print(torch.xpu.device_count())
"""
        rc = check_output(test_script)
        self.assertEqual(rc, str(torch.xpu.device_count()))

    def test_streams(self):
        s0 = torch.xpu.Stream()
        torch.xpu.set_stream(s0)
        s1 = torch.xpu.current_stream()
        self.assertEqual(s0, s1)
        s2 = torch.xpu.Stream()
        self.assertFalse(s0 == s2)
        torch.xpu.set_stream(s2)
        with torch.xpu.stream(s0):
            self.assertEqual(s0, torch.xpu.current_stream())
        self.assertEqual(s2, torch.xpu.current_stream())

    def test_stream_priority(self):
        low, high = torch.xpu.Stream.priority_range()
        s0 = torch.xpu.Stream(device=0, priority=low)

        self.assertEqual(low, s0.priority)
        self.assertEqual(torch.device("xpu:0"), s0.device)

        s1 = torch.xpu.Stream(device=0, priority=high)

        self.assertEqual(high, s1.priority)
        self.assertEqual(torch.device("xpu:0"), s1.device)

    def test_stream_event_repr(self):
        s = torch.xpu.current_stream()
        self.assertTrue("torch.xpu.Stream" in str(s))
        e = torch.xpu.Event()
        self.assertTrue("torch.xpu.Event(uninitialized)" in str(e))
        s.record_event(e)
        self.assertTrue("torch.xpu.Event" in str(e))
    # 测试事件操作函数
    def test_events(self):
        # 获取当前 XPU 流对象
        stream = torch.xpu.current_stream()
        # 创建新的 XPU 事件对象
        event = torch.xpu.Event()
        # 查询事件状态，断言为真
        self.assertTrue(event.query())
        # 在流对象上记录事件
        stream.record_event(event)
        # 同步事件
        event.synchronize()
        # 再次查询事件状态，断言为真
        self.assertTrue(event.query())

    # 测试通用流和事件操作函数
    def test_generic_stream_event(self):
        # 创建新的 XPU 流对象
        stream = torch.Stream("xpu")
        # 断言流对象的设备索引与当前设备索引相同
        self.assertEqual(stream.device_index, torch.xpu.current_device())
        # 创建对应的 XPU 流对象
        xpu_stream = torch.xpu.Stream(
            stream_id=stream.stream_id,
            device_index=stream.device_index,
            device_type=stream.device_type,
        )
        # 断言流对象的流 ID 与对应 XPU 流对象的流 ID 相同
        self.assertEqual(stream.stream_id, xpu_stream.stream_id)
        # 断言流对象的流 ID 与当前 XPU 流对象的流 ID 不同
        self.assertNotEqual(stream.stream_id, torch.xpu.current_stream().stream_id)

        # 创建两个新的 XPU 事件对象
        event1 = torch.Event("xpu")
        event2 = torch.Event("xpu")
        # 断言第一个事件对象的事件 ID 为 0
        self.assertEqual(event1.event_id, 0)
        # 创建张量 a 和 b
        a = torch.randn(1000)
        b = torch.randn(1000)
        # 在指定 XPU 流上下文中，将张量 a 和 b 分别移到 XPU 设备上
        with torch.xpu.stream(xpu_stream):
            a_xpu = a.to("xpu", non_blocking=True)
            b_xpu = b.to("xpu", non_blocking=True)
            # 断言当前流对象的流 ID 与当前 XPU 流对象的流 ID 相同
            self.assertEqual(stream.stream_id, torch.xpu.current_stream().stream_id)
        
        # 在指定流对象上记录事件 event1
        event1.record(stream)
        # 同步事件 event1
        event1.synchronize()
        # 查询事件 event1 的状态，断言为真
        self.assertTrue(event1.query())
        
        # 在 XPU 设备上计算张量 c_xpu
        c_xpu = a_xpu + b_xpu
        
        # 在默认流上记录事件 event2
        event2.record()
        # 同步事件 event2
        event2.synchronize()
        # 查询事件 event2 的状态，断言为真
        self.assertTrue(event2.query())
        
        # 断言两个事件对象的事件 ID 不相同
        self.assertNotEqual(event1.event_id, event2.event_id)
        # 断言 c_xpu 在 CPU 上的值与 a + b 相等
        self.assertEqual(c_xpu.cpu(), a + b)
        
        # 断言在 XPU 后端不支持 elapsedTime
        with self.assertRaisesRegex(
            NotImplementedError, "elapsedTime is not supported by XPU backend."
        ):
            event1.elapsed_time(event2)

    # 测试随机数生成器状态函数
    def test_generator(self):
        # 设置随机数种子为 2024
        torch.manual_seed(2024)
        # 获取当前 XPU 随机数生成器的状态
        g_state0 = torch.xpu.get_rng_state()
        # 设置新的随机数种子为 1234
        torch.manual_seed(1234)
        # 获取当前 XPU 随机数生成器的状态
        g_state1 = torch.xpu.get_rng_state()
        # 断言两次获取的随机数生成器状态不相等
        self.assertNotEqual(g_state0, g_state1)
        
        # 设置 XPU 随机数种子为 2024
        torch.xpu.manual_seed(2024)
        # 获取当前 XPU 随机数生成器的状态
        g_state2 = torch.xpu.get_rng_state()
        # 断言设置 XPU 随机数种子为 2024 后，状态与 g_state0 相同
        self.assertEqual(g_state0, g_state2)
        
        # 设置 XPU 随机数生成器的状态为 g_state1
        torch.xpu.set_rng_state(g_state1)
        # 断言设置后的状态与 g_state1 相同
        self.assertEqual(g_state1, torch.xpu.get_rng_state())
        
        # 设置 CPU 随机数种子为 1234
        torch.manual_seed(1234)
        # 设置 XPU 随机数生成器的状态为 g_state0
        torch.xpu.set_rng_state(g_state0)
        # 断言 XPU 初始种子为 2024
        self.assertEqual(2024, torch.xpu.initial_seed())

    # 测试装饰器
    @onlyXPU
    @suppress_warnings
    @ops(_xpu_computation_ops, dtypes=any_common_cpu_xpu_one)
    # 定义一个测试方法，用于比较在不同设备和数据类型下的操作结果
    def test_compare_cpu(self, device, dtype, op):
        # 定义一个内部函数，用于将输入数据转移到 CPU 设备上
        def to_cpu(arg):
            if isinstance(arg, torch.Tensor):
                return arg.to(device="cpu")
            return arg

        # 获取操作的参考输入样本
        samples = op.reference_inputs(device, dtype)

        # 遍历每个样本
        for sample in samples:
            # 将样本转移到 CPU 上
            cpu_sample = sample.transform(to_cpu)
            # 在当前设备上执行操作，获取结果
            xpu_results = op(sample.input, *sample.args, **sample.kwargs)
            # 在 CPU 上执行操作，获取结果
            cpu_results = op(cpu_sample.input, *cpu_sample.args, **cpu_sample.kwargs)

            # 对结果进行处理，以减小数值比较的容差，因为此测试标记为 @slowTest
            xpu_results = sample.output_process_fn_grad(xpu_results)
            cpu_results = cpu_sample.output_process_fn_grad(cpu_results)

            # 断言两种执行方式下的结果应相等，设置数值容差和相对容差
            self.assertEqual(xpu_results, cpu_results, atol=1e-4, rtol=1e-4)

    # 定义一个测试方法，用于测试非标准布尔值情况下的操作
    @onlyXPU
    @ops(_xpu_computation_ops, allowed_dtypes=(torch.bool,))
    @unittest.skipIf(TEST_WITH_UBSAN, "Test uses undefined behavior")
    def test_non_standard_bool_values(self, device, dtype, op):
        # 定义一个内部函数，用于转换布尔张量的值
        def convert_boolean_tensors(x):
            if not isinstance(x, torch.Tensor) or x.dtype != torch.bool:
                return x

            # 将 False 映射为 0，将 True 映射为区间 [2, 255] 中的随机值
            true_vals = torch.randint(
                2, 255, x.shape, dtype=torch.uint8, device=x.device
            )
            false_vals = torch.zeros((), dtype=torch.uint8, device=x.device)
            x_int = torch.where(x, true_vals, false_vals)

            # 将结果重新视为布尔类型张量，并断言转换后的结果应与原始输入相等
            ret = x_int.view(torch.bool)
            self.assertEqual(ret, x)
            return ret

        # 遍历操作的样本输入
        for sample in op.sample_inputs(device, dtype):
            # 获取预期输出结果
            expect = op(sample.input, *sample.args, **sample.kwargs)

            # 转换输入样本的布尔张量值
            transformed = sample.transform(convert_boolean_tensors)
            # 在转换后的输入上执行操作，获取实际输出结果
            actual = op(transformed.input, *transformed.args, **transformed.kwargs)

            # 断言预期输出和实际输出结果应相等
            self.assertEqual(expect, actual)

    # 定义一个测试方法，用于测试带有存储的数组序列化
    def test_serialization_array_with_storage(self):
        # 创建两个张量对象 x 和 y
        x = torch.randn(5, 5).xpu()
        y = torch.zeros(2, 5, dtype=torch.int, device="xpu")
        # 创建一个包含 x、y、x、y.storage() 的列表 q
        q = [x, y, x, y.storage()]
        
        # 使用临时文件存储 q 对象
        with tempfile.NamedTemporaryFile() as f:
            torch.save(q, f)
            f.seek(0)
            # 从文件中加载并复制 q 对象
            q_copy = torch.load(f)
        
        # 断言加载后的对象与原始对象 q 相等，设置数值容差和相对容差为 0
        self.assertEqual(q_copy, q, atol=0, rtol=0)
        
        # 修改复制对象 q_copy 中的第一个元素，将其所有元素填充为 5
        q_copy[0].fill_(5)
        # 断言修改后的第一个元素与第三个元素（原始的 x）相等，设置数值容差和相对容差为 0
        self.assertEqual(q_copy[0], q_copy[2], atol=0, rtol=0)
        # 断言修改后的第一个元素的数据类型应为 torch.float
        self.assertEqual(q_copy[0].dtype, torch.float)
        # 断言第二个元素的数据类型应为 torch.int
        self.assertEqual(q_copy[1].dtype, torch.int)
        # 断言第三个元素的数据类型应为 torch.float
        self.assertEqual(q_copy[2].dtype, torch.float)
        # 断言第四个元素应为 torch.storage.TypedStorage 类型对象
        self.assertTrue(isinstance(q_copy[3], torch.storage.TypedStorage))
        # 断言第四个元素的未命名存储类型应为 torch.UntypedStorage 类型对象
        self.assertTrue(isinstance(q_copy[3]._untyped_storage, torch.UntypedStorage))
        
        # 修改复制对象 q_copy 中的第二个元素，将其所有元素填充为 10
        q_copy[1].fill_(10)
        # 修改原始对象 y 中的所有元素，将其所有元素填充为 10
        y.fill_(10)
        # 断言修改后的第四个元素与修改后的第二个元素（原始的 y）相等
        self.assertEqual(q_copy[3], y.storage())
    # 定义一个测试函数，用于测试包含空数组的序列化功能
    def test_serialization_array_with_empty(self):
        # 创建一个包含两个元素的列表 x，其中包括一个随机张量和一个空的浮点数张量
        x = [
            torch.randn(4, 4).xpu(),  # 生成一个 4x4 的随机张量，设备类型为 xpu
            torch.tensor([], dtype=torch.float, device=torch.device("xpu")),  # 创建一个空的浮点数张量，设备类型为 xpu
        ]
        # 使用临时文件来保存序列化后的数据
        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)  # 将列表 x 序列化并保存到临时文件 f 中
            f.seek(0)  # 将文件指针移动到文件开头，以便读取数据
            x_copy = torch.load(f)  # 从临时文件 f 中加载并反序列化数据到 x_copy
        
        # 遍历原始列表 x 和反序列化后的列表 x_copy，并逐一比较其元素
        for original, copy in zip(x, x_copy):
            self.assertEqual(copy, original)  # 断言每个反序列化后的元素与原始元素相等
            self.assertIs(type(copy), type(original))  # 断言每个反序列化后的元素与原始元素类型相同
            self.assertEqual(copy.get_device(), original.get_device())  # 断言每个反序列化后的张量设备与原始张量设备相同
# 在全局范围中实例化设备类型测试，此处针对 TestXpu 类的测试，并将其添加到全局命名空间中
instantiate_device_type_tests(TestXpu, globals(), only_for="xpu")

# 定义 TestXpuAutocast 类，继承自 TestCase 类
class TestXpuAutocast(TestCase):
    # 这些操作在 XPU 后端上尚未实现，无法回退到 CPU 执行，因此在此阶段必须跳过它们
    # TODO: 当这些操作在 XPU 后端上实现后，从跳过列表中移除它们。
    skip_list = ["gru_cell"]

    # 设置测试前的准备工作
    def setUp(self):
        super().setUp()
        # 创建一个 AutocastTestLists 对象，针对 xpu 设备
        self.autocast_lists = AutocastTestLists(torch.device("xpu"))

    # 清理测试后的资源
    def tearDown(self):
        del self.autocast_lists
        super().tearDown()

    # 执行 out-of-place 的自动类型转换测试
    def _run_autocast_outofplace(
        self, op, args, run_as_type, out_type=None, module=torch, add_kwargs=None
    ):
        # 以下为具体的自动类型转换测试方法，逐一测试 torch.float16 类型
    def test_autocast_torch_fp16(self):
        for op_with_args in self.autocast_lists.torch_fp16:
            skip_test = False
            op, args = op_with_args[0], op_with_args[1]
            if op in self.skip_list:
                skip_test = True  # 跳过未实现的操作
            if len(op_with_args) == 3:
                skip_test = True  # 跳过 cudnn 操作
            if not skip_test:
                self._run_autocast_outofplace(op, args, torch.float16)

    # 类似 test_autocast_torch_fp16，但针对 torch.bfloat16 类型进行测试
    def test_autocast_torch_bf16(self):
        for op_with_args in self.autocast_lists.torch_fp16:
            skip_test = False
            op, args = op_with_args[0], op_with_args[1]
            if op in self.skip_list:
                skip_test = True  # 跳过未实现的操作
            if len(op_with_args) == 3:
                skip_test = True  # 跳过 cudnn 操作
            if not skip_test:
                self._run_autocast_outofplace(op, args, torch.bfloat16)

    # 执行需要自动类型转换提升的测试，测试 torch.float32 类型
    def test_autocast_torch_need_autocast_promote(self):
        for op, args in self.autocast_lists.torch_need_autocast_promote:
            self._run_autocast_outofplace(op, args, torch.float32)

    # 执行期望内建类型提升的测试，测试 torch.float32 类型，指定输出类型
    def test_autocast_torch_expect_builtin_promote(self):
        for op, args, out_type in self.autocast_lists.torch_expect_builtin_promote:
            self._run_autocast_outofplace(op, args, torch.float32, out_type=out_type)

    # 执行检查点检查的测试
    def test_autocast_checkpointing(self):
        # 创建一个包含线性层的简单模型，并指定其在 xpu 上执行
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)
        ).xpu()
        # 创建一个在 xpu 上的随机输入张量，数据类型为 torch.float16，需要梯度计算
        input = torch.rand(
            (8, 8), device="xpu", dtype=torch.float16, requires_grad=True
        )
        # 针对重进入性测试的两种情况（True 和 False），在 xpu 上自动类型转换执行检查点序列模型
        for reentrant in (True, False):
            with torch.autocast("xpu"):
                output = checkpoint_sequential(model, 2, input, use_reentrant=reentrant)
            self.assertTrue(output.requires_grad)
            self.assertTrue(output.dtype is torch.float16)
            output.sum().backward()
    # 定义一个测试方法，测试自动类型转换在 xpu 设备上的行为
    def test_xpu_autocast_dtype(self):
        # 获取 xpu 设备上的自动类型转换的数据类型
        dtype = torch.get_autocast_dtype("xpu")
        # 断言获取到的数据类型是 torch.float16
        self.assertEqual(dtype, torch.float16)
        # 创建两个在 xpu 设备上的随机数矩阵，数据类型为 torch.float32
        mat0_fp32 = torch.randn((10, 10), dtype=torch.float32, device="xpu")
        mat1_fp32 = torch.randn((10, 10), dtype=torch.float32, device="xpu")
        # 进入自动混合精度上下文环境 "xpu"
        with torch.amp.autocast("xpu"):
            # 执行矩阵乘法操作，期望结果的数据类型是 torch.float16
            result = torch.mm(mat0_fp32, mat1_fp32)
            # 断言实际结果的数据类型确实是 torch.float16
            self.assertEqual(result.dtype, torch.float16)
class TestXpuTrace(TestCase):
    # 测试用例类，用于测试 XPU 跟踪功能

    def setUp(self):
        # 设置测试环境，在测试开始前激活 GPU 跟踪功能
        torch._C._activate_gpu_trace()
        self.mock = unittest.mock.MagicMock()

    def test_event_creation_callback(self):
        # 测试事件创建回调函数注册与调用
        gpu_trace.register_callback_for_event_creation(self.mock)

        # 创建一个 XPU 事件并记录
        event = torch.xpu.Event()
        event.record()
        # 断言回调函数被调用，参数为事件的内部参数值
        self.mock.assert_called_once_with(event._as_parameter_.value)

    def test_event_deletion_callback(self):
        # 测试事件删除回调函数注册与调用
        gpu_trace.register_callback_for_event_deletion(self.mock)

        # 创建一个 XPU 事件并记录
        event = torch.xpu.Event()
        event.record()
        event_id = event._as_parameter_.value
        # 删除事件对象
        del event
        # 断言回调函数被调用，参数为事件的内部参数值
        self.mock.assert_called_once_with(event_id)

    def test_event_record_callback(self):
        # 测试事件记录回调函数注册与调用
        gpu_trace.register_callback_for_event_record(self.mock)

        # 创建一个 XPU 事件并记录
        event = torch.xpu.Event()
        event.record()
        # 断言回调函数被调用，参数为事件的内部参数值和当前流队列的 SYCL 队列
        self.mock.assert_called_once_with(
            event._as_parameter_.value, torch.xpu.current_stream().sycl_queue
        )

    def test_event_wait_callback(self):
        # 测试事件等待回调函数注册与调用
        gpu_trace.register_callback_for_event_wait(self.mock)

        # 创建一个 XPU 事件并记录，然后等待事件完成
        event = torch.xpu.Event()
        event.record()
        event.wait()
        # 断言回调函数被调用，参数为事件的内部参数值和当前流队列的 SYCL 队列
        self.mock.assert_called_once_with(
            event._as_parameter_.value, torch.xpu.current_stream().sycl_queue
        )

    def test_device_synchronization_callback(self):
        # 测试设备同步回调函数注册与调用
        gpu_trace.register_callback_for_device_synchronization(self.mock)

        # 执行设备同步操作
        torch.xpu.synchronize()
        # 断言回调函数被调用
        self.mock.assert_called()

    def test_stream_synchronization_callback(self):
        # 测试流同步回调函数注册与调用
        gpu_trace.register_callback_for_stream_synchronization(self.mock)

        # 创建一个 XPU 流并执行同步操作
        stream = torch.xpu.Stream()
        stream.synchronize()
        # 断言回调函数被调用，参数为流的 SYCL 队列
        self.mock.assert_called_once_with(stream.sycl_queue)

    def test_event_synchronization_callback(self):
        # 测试事件同步回调函数注册与调用
        gpu_trace.register_callback_for_event_synchronization(self.mock)

        # 创建一个 XPU 事件并记录，然后执行事件同步操作
        event = torch.xpu.Event()
        event.record()
        event.synchronize()
        # 断言回调函数被调用，参数为事件的内部参数值
        self.mock.assert_called_once_with(event._as_parameter_.value)


if __name__ == "__main__":
    run_tests()
```