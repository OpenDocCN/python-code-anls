# `.\pytorch\test\profiler\test_torch_tidy.py`

```py
# Owner(s): ["oncall: profiler"]

# 如果 tqdm 没有正确关闭，它会保持监视线程活动。
# 这会导致多线程测试中出现问题，因为我们在该测试中检查所有事件的线程 ID。
# 与这些持续存在的线程对应的事件的线程 ID 都是 (uint64_t)(-1)，即无效的。
# 解决方法是在加载 tqdm 时关闭监视线程。
# 由于这些是单元测试，关闭监视线程是安全的。
try:
    import tqdm

    # 设置 tqdm 的监视间隔为 0，即关闭监视线程
    tqdm.tqdm.monitor_interval = 0
except ImportError:
    None

import gc  # 导入垃圾回收模块
import re  # 导入正则表达式模块
import textwrap  # 导入文本包装模块
import unittest  # 导入单元测试框架
import weakref  # 导入弱引用模块
from typing import Any, Dict, List  # 导入类型提示模块

import torch  # 导入 PyTorch 模块
import torch.nn as nn  # 导入神经网络模块
import torch.optim  # 导入优化器模块
import torch.utils.data  # 导入数据处理模块
from torch._C._profiler import _TensorMetadata  # 导入 PyTorch 分析器相关模块
from torch.profiler import _utils, profile  # 导入分析器工具和分析模块
from torch.testing._internal.common_utils import run_tests, TestCase  # 导入测试工具和测试用例基类

Json = Dict[str, Any]  # 定义 Json 类型别名

from torch._C._profiler import _ExtraFields_PyCall  # 导入分析器额外字段相关模块


def find_node_with_name(nodes, name):
    # 根据节点名称在节点列表中查找节点
    for node in _utils.traverse_dfs(nodes):
        if node.name == name:
            return node


def find_node_with_regex(nodes, pattern):
    # 根据正则表达式模式在节点列表中查找节点
    for node in _utils.traverse_dfs(nodes):
        if re.search(pattern, node.name):
            return node


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义神经网络结构，包括两个全连接层
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        # 前向传播函数，实现网络的正向计算
        return self.fc2(self.fc1(x))


class TestTorchTidyProfiler(TestCase):
    def _get_tensor_fields(self, node, index):
        # 断言节点不为空
        self.assertIsNotNone(node)
        # 断言节点额外字段是 torch._C._profiler._ExtraFields_TorchOp 类型
        self.assertIsInstance(
            node.extra_fields, torch._C._profiler._ExtraFields_TorchOp
        )
        # 获取节点输入张量信息
        tensor_info = node.extra_fields.inputs[index]
        # 断言张量信息是 _TensorMetadata 类型
        self.assertIsInstance(tensor_info, _TensorMetadata)
        # 断言张量信息的实现指针不为空
        self.assertIsNotNone(tensor_info.impl_ptr)
        # 断言张量信息的存储数据指针不为空
        self.assertIsNotNone(tensor_info.storage_data_ptr)
        # 断言张量信息的 ID 不为空
        self.assertIsNotNone(tensor_info.id)
        # 返回张量信息的实现指针、存储数据指针和 ID
        return tensor_info.impl_ptr, tensor_info.storage_data_ptr, tensor_info.id
    # 定义测试方法 test_pointers_and_ids，使用 self 参数表示该方法属于一个类的实例方法
    def test_pointers_and_ids(self):
        # 创建一个形状为 (4, 3) 的随机张量 a
        a = torch.randn(4, 3)
        # 获取张量 a 的存储器的数据指针
        a_initial_storage_data = a.storage().data_ptr()

        # 视图张量可以共享相同的存储器，但具有不同的 TensorImpl
        b = a.view((1, 12))
        # 创建另一个形状为 (4, 1) 的随机张量 c
        c = torch.randn(4, 1)
        # 获取张量 c 的存储器的数据指针
        c_initial_storage_data = c.storage().data_ptr()
        # 创建另一个形状为 (4, 3) 的随机张量 d
        d = torch.randn(4, 3)

        # 使用 profile 进行性能分析，记录内存使用、调用栈和形状
        with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
            _ = a + c  # 执行张量 a 与 c 的加法操作
            _ = b * c  # 执行张量 b 与 c 的乘法操作

            # 调整张量 a 的大小，并保持 TensorImpl 相同但创建新的数据指针
            f = a.resize_(128, 129)
            _ = torch.relu(f)  # 执行张量 f 的 ReLU 操作

            # `.set_` 方法将张量 c 指向张量 d 的存储器
            _ = d.sin()
            c.set_(d.storage())
            _ = c.cos()

        # 从性能分析器中获取节点
        nodes = p.profiler.kineto_results.experimental_event_tree()

        # 定义一个函数 get_fields，用于获取指定操作名和索引的张量字段
        def get_fields(op_name, index):
            return self._get_tensor_fields(find_node_with_name(nodes, op_name), index)

        # 从性能分析结果中获取张量 a 的 Impl、存储器数据指针和 ID
        a_impl, a_storage_data, a_id = get_fields("aten::add", 0)
        # 从性能分析结果中获取张量 b 的 Impl、存储器数据指针和 ID
        b_impl, b_storage_data, b_id = get_fields("aten::mul", 0)

        # 断言：性能分析结果中的存储器数据指针与初始存储器数据指针相同
        self.assertEqual(a_storage_data, a_initial_storage_data)

        # 断言：处理视图张量的正确性
        self.assertEqual(a_storage_data, b_storage_data)
        self.assertNotEqual(a_impl, b_impl)

        # 断言：多次调用中相同的张量给出相同的结果
        c_impl, c_storage_data, c_id = get_fields("aten::add", 1)
        self.assertEqual((c_impl, c_storage_data, c_id), get_fields("aten::mul", 1))
        self.assertEqual(c_storage_data, c_initial_storage_data)

        # 断言：对底层存储器的变化会反映在性能分析中（ID 保持共享）
        f_impl, f_storage_data, f_id = get_fields("aten::relu", 0)
        self.assertEqual(a_impl, f_impl)
        self.assertNotEqual(a_storage_data, f_storage_data)
        self.assertEqual(a_id, f_id)

        # 断言：使用 `set_` 方法后，两个张量共享同一个 ID
        d_impl, d_storage_data, d_id = get_fields("aten::sin", 0)
        c_impl_new, c_storage_data_new, c_id_new = get_fields("aten::cos", 0)
        self.assertNotEqual(d_impl, c_impl_new)
        self.assertEqual(d_storage_data, c_storage_data_new)
        self.assertEqual(c_id, c_id_new)
        self.assertEqual(d_id, c_id_new)
    def _format_allocations(profiled_code):
        # 执行垃圾回收，确保内存状态清理
        gc.collect()
        # 使用 profile 上下文，记录内存分配和形状
        with profile(profile_memory=True, record_shapes=True) as prof:
            # 执行传入的 profiled_code 函数
            profiled_code()
            # 再次执行垃圾回收，确保记录完全
            gc.collect()

        # 从 profiler 中提取根事件，形成事件树
        root_events = prof.profiler.kineto_results.experimental_event_tree()
        # 深度优先遍历事件树并排序
        events = sorted(_utils.traverse_dfs(root_events), key=lambda x: x.start_time_ns)
        # 提取分配事件，并按照开始时间排序
        allocations = tuple(
            event.extra_fields
            for event in events
            if isinstance(
                event.extra_fields, torch._C._profiler._ExtraFields_Allocation
            )
        )

        # 格式化输出分配事件信息
        return textwrap.indent(
            "\n".join(
                f"{repr(i.id):>5}{' ' * 6}"
                f"{repr(i.allocation_id):>5}{' ' * 6}"
                f"{'Allocation' if i.alloc_size > 0 else 'Free'}"
                for i in allocations
            ),
            " " * 12,
        )

    def test_tensorimpl_invalidation_set(self) -> None:
        # 定义 profiled_code 函数，接受一个布尔参数 add_empty_set
        def profiled_code(add_empty_set: bool):
            # 创建一个包含单个元素的张量 x
            x = torch.ones((1,))

            # 判断是否在销毁旧存储之前或之后创建新存储
            if add_empty_set:
                # 若 add_empty_set 为真，则调用 set_() 方法
                x.set_()

            # 将 x 的存储设置为一个包含单个元素的张量的存储
            x.set_(torch.ones((1,)).storage())
            # 将 x 视图化为自身
            x.view_as(x)

        # 使用 self.assertExpectedInline 断言函数，验证 profiled_code 函数的输出
        self.assertExpectedInline(
            self._format_allocations(lambda: profiled_code(add_empty_set=False)),
            """\
                0          1      Allocation
                0          2      Allocation
                0          1      Free
                0          2      Free""",
        )

        # 使用 self.assertExpectedInline 断言函数，验证 profiled_code 函数在不同参数下的输出
        self.assertExpectedInline(
            self._format_allocations(lambda: profiled_code(add_empty_set=True)),
            """\
                0          1      Allocation
                0          1      Free
                0          2      Allocation
                0          2      Free""",
        )
    def test_tensorimpl_invalidation_keep_alive(self) -> None:
        # 定义内部函数 profiled_code，用于测试保持存活期间的张量实现无效化
        def profiled_code(add_empty_set: bool):
            # 创建一个包含一个元素的张量 x
            x = torch.ones((1,))
            # 初始化张量 x 的存储列表
            x_storages = [x.storage()]
            # 循环3次
            for _ in range(3):
                # 无效化张量 x
                x.set_()
                # 设置新的存储给张量 x
                x.set_(torch.ones((1,)).storage())

                # 这段代码保持 StorageImpl 存活并保持链的完整性
                # （尽管有 `set_()` 调用）
                x_storages.append(x.storage())
            # 将张量 x 视为自身，通常没有实际效果，仅为示例
            x.view_as(x)

            # 以确定性方式释放存储
            while x_storages:
                x_storages.pop()
                gc.collect()

            # 决定在旧存储销毁前后是否创建新存储
            if add_empty_set:
                x.set_()

            # 再次循环3次
            for _ in range(3):
                # 设置新的存储给张量 x
                x.set_(torch.ones((1,)).storage())
            # 将张量 x 视为自身，通常没有实际效果，仅为示例
            x.view_as(x)

            # 删除张量 x
            del x
            gc.collect()

        # 断言内联结果与预期结果一致，调用 profiled_code 函数，不添加空集合
        self.assertExpectedInline(
            self._format_allocations(lambda: profiled_code(add_empty_set=False)),
            """\
                0          1      Allocation
                0          2      Allocation
                0          4      Allocation
                0          5      Allocation
                0          4      Free
                0          2      Free
                0          1      Free
                0          6      Allocation
                0          5      Free
                0          7      Allocation
                0          6      Free
                0          8      Allocation
                0          7      Free
                0          8      Free""",
        )

        # 断言内联结果与预期结果一致，调用 profiled_code 函数，添加空集合
        self.assertExpectedInline(
            self._format_allocations(lambda: profiled_code(add_empty_set=True)),
            """\
                0          1      Allocation
                0          2      Allocation
                0          4      Allocation
                0          5      Allocation
                0          4      Free
                0          2      Free
                0          1      Free
                0          5      Free
                0          6      Allocation
                0          7      Allocation
                0          6      Free
                0          8      Allocation
                0          7      Free
                0          8      Free""",
        )
    def test_tensorimpl_invalidation_full(self) -> None:
        # 定义内部函数 profiled_code，用于测试张量操作的内存分配和释放情况
        def profiled_code():
            # 创建一个包含单个元素的张量 x
            x = torch.ones((1,))
            # 创建一个存储列表，将张量 x 的存储加入其中
            x_storages = [x.storage()]
            # 循环操作，重复以下步骤三次
            for _ in range(3):
                # 将张量 x 清零
                x.set_()
                # 重新设置张量 x 的存储为新的全 1 存储
                x.set_(torch.ones((1,)).storage())
                # 将新的存储加入存储列表
                x_storages.append(x.storage())
            # 对张量 x 进行视图操作
            x.view_as(x)

            # 以确定性方式释放存储
            while x_storages:
                # 从存储列表中弹出存储，并进行垃圾回收
                x_storages.pop()
                gc.collect()

            # 再次重复以下步骤三次
            for _ in range(3):
                # 将张量 x 的存储设置为新的全 1 存储
                x.set_(torch.ones((1,)).storage())

            # 再次重复以下步骤三次
            for _ in range(3):
                # 将张量 x 清零
                x.set_()
                # 重新设置张量 x 的存储为新的全 1 存储
                x.set_(torch.ones((1,)).storage())

            # 循环操作，调整张量 x 的大小，使其逐渐增加
            for i in range(4):
                x.resize_((1 + i,))
            # 对张量 x 进行视图操作
            x.view_as(x)

        # 调用断言方法，验证内部函数 profiled_code 的内存分配情况
        self.assertExpectedInline(
            self._format_allocations(profiled_code),
            """\
                0          1      Allocation
                0          2      Allocation
                0          4      Allocation
                0          5      Allocation
                0          4      Free
                0          2      Free
                0          1      Free
                0          6      Allocation
                0          5      Free
                0          7      Allocation
                0          6      Free
                0          8      Allocation
                0          7      Free
                0          8      Free
                0          9      Allocation
                0          9      Free
                0         10      Allocation
                0         10      Free
                0         11      Allocation
                0         12      Allocation
                0         11      Free
                0         13      Allocation
                0         12      Free
                0         14      Allocation
                0         13      Free
                0         14      Free""",
        )
    def test_tensorimpl_invalidation_scalar_args(self) -> None:
        # 定义测试方法，验证在无效化标量参数时的张量实现
        def profiled_code():
            # 使用 torch.no_grad() 上下文管理器，禁止梯度计算
            with torch.no_grad():
                # 创建一个包含单个元素的张量 x
                x = torch.ones((1,))
                # 执行 10 次操作，每次将 x 的值增加 2，使用 in-place 操作
                for _ in range(10):
                    x.add_(2)

        # 断言实际输出与预期输出一致
        self.assertExpectedInline(
            # 格式化并返回分配情况的字符串
            self._format_allocations(profiled_code),
            """\
                0          1      Allocation
                1          2      Allocation
                2          3      Allocation
                2          3      Free
                1          2      Free
                3          4      Allocation
                4          5      Allocation
                4          5      Free
                3          4      Free
                5          6      Allocation
                6          7      Allocation
                6          7      Free
                5          6      Free
                7          8      Allocation
                8          9      Allocation
                8          9      Free
                7          8      Free
                9         10      Allocation
               10         11      Allocation
               10         11      Free
                9         10      Free
               11         12      Allocation
               12         13      Allocation
               12         13      Free
               11         12      Free
               13         14      Allocation
               14         15      Allocation
               14         15      Free
               13         14      Free
               15         16      Allocation
               16         17      Allocation
               16         17      Free
               15         16      Free
               17         18      Allocation
               18         19      Allocation
               18         19      Free
               17         18      Free
               19         20      Allocation
               20         21      Allocation
               20         21      Free
               19         20      Free
                0          1      Free""",
        )
    # 测试函数，用于验证张量分配和释放过程中的标识符一致性
    def _test_allocation_ids(self, before_fn, after_fn) -> None:
        # 使用性能分析器，记录内存使用和操作形状
        with profile(profile_memory=True, record_shapes=True) as p:
            # 执行前置函数，引入其他操作和分配以检查鲁棒性
            _ = before_fn()

            # 创建一个大小为 (4, 3) 的随机张量 x
            x = torch.rand(4, 3)
            # 重新调整张量 x 的大小为 (4, 4)
            x.resize_(4, 4)

            # 对重新调整大小后的张量 x 执行正弦函数操作
            x.sin()

            # 执行后置函数，引入其他操作和分配以检查鲁棒性
            _ = after_fn()

            # 确保 x 是最后一个被收集的变量，以便更容易找到释放事件
            gc.collect()
            del x
            gc.collect()

        # 获取性能分析器的事件树
        nodes = p.profiler.kineto_results.experimental_event_tree()

        # 在事件树中查找特定链条的节点
        def find_chain(names: List[str]):
            out = []
            for name in names:
                # 如果 out 非空，将根节点设为上一个节点的后代
                root = [out[-1]] if out else nodes
                # 查找具有指定名称的节点，并将其添加到输出列表中
                out.append(find_node_with_name(root, name))
                # 断言找到的节点不为空，打印其名称
                self.assertIsNotNone(out[-1], name)
            return out

        # 查找并获取与 "aten::rand", "aten::empty", "[memory]" 相关的链条
        allocation = find_chain(["aten::rand", "aten::empty", "[memory]"])[-1].extra_fields
        _, uniform_node = find_chain(["aten::rand", "aten::uniform_"])

        # 从 uniform_node 中获取张量字段
        x_impl, x_storage_data, x_id = self._get_tensor_fields(uniform_node, 0)

        # 确保分配和操作输入之间的 ID 一致性
        self.assertEqual(allocation.ptr, x_storage_data)
        self.assertEqual(allocation.id, x_id)

        # 查找并获取 "aten::resize_" 的节点
        resize_node = find_node_with_name(nodes, "aten::resize_")
        self.assertIsNotNone(resize_node)
        self.assertEqual(len(resize_node.children), 2)

        # 获取 resize_node 的额外字段中分配新内存和释放旧内存的信息
        allocate_new = resize_node.children[0].extra_fields
        free_old = resize_node.children[1].extra_fields

        # 确保旧存储释放事件的 ID 和指针与分配事件一致
        self.assertEqual(free_old.id, allocation.id)
        self.assertEqual(free_old.ptr, allocation.ptr)

        # 确保 ID 在存储更改时保持一致
        self.assertEqual(allocate_new.id, allocation.id)
        self.assertNotEqual(allocate_new.ptr, allocation.ptr)

        # 查找并获取最后一个分配事件的额外字段，当 x 被释放时
        free_new = [
            i for i in nodes if i.tag == torch._C._profiler._EventType.Allocation
        ][-1].extra_fields

        # 断言新分配事件的 ID 和指针与分配新内存事件一致
        self.assertIsInstance(free_new, torch._C._profiler._ExtraFields_Allocation)
        self.assertEqual(free_new.id, allocate_new.id)
        self.assertEqual(free_new.ptr, allocate_new.ptr)

    # 测试函数：验证张量分配和释放过程中的标识符一致性
    def test_allocation_ids(self) -> None:
        # 使用空函数进行测试
        self._test_allocation_ids(lambda: None, lambda: None)

    # 测试函数：验证包含其他操作的张量分配和释放过程中的标识符一致性
    def test_allocation_ids_with_other_ops(self) -> None:
        # 创建一个包含单个元素的张量 x，执行包含其他操作的测试
        x = torch.ones((1,))
        self._test_allocation_ids(
            lambda: (x + 1).relu_(), lambda: torch.zeros((1,)).cos()
        )
    # 定义测试函数 test_impl_reuse，用于测试重复使用 tensor 实现的情况
    def test_impl_reuse(self) -> None:
        # 设置重复次数为 1000
        repeats = 1_000
        # 使用性能分析器 profile，并记录内存使用和形状信息
        with profile(profile_memory=True, record_shapes=True) as p:
            # 执行重复次数的 torch.ones((1,)) 操作
            for _ in range(repeats):
                torch.ones((1,))
            # 手动触发垃圾回收
            gc.collect()

        # 获取性能分析器的事件树根节点
        roots = p.profiler.kineto_results.experimental_event_tree()
        # 从事件树中查找所有 "aten::fill_" 操作对应的 tensor 实现指针
        tensor_impls = tuple(
            e.extra_fields.inputs[0].impl_ptr
            for e in _utils.traverse_dfs(roots)
            if e.name == "aten::fill_"
        )

        # 断言 tensor 实现指针的数量与重复次数相等
        self.assertEqual(len(tensor_impls), repeats)
        # 断言 tensor 实现指针的集合大小与重复次数相等，确保每个实现只出现一次
        self.assertEqual(len(set(tensor_impls)), repeats)

    # 定义测试函数 test_allocation_id_uniqueness，用于测试分配 ID 的唯一性
    def test_allocation_id_uniqueness(self) -> None:
        # 设置重复次数为 1000
        repeats = 1_000
        # 使用性能分析器 profile，并记录内存使用和形状信息
        with profile(profile_memory=True, record_shapes=True) as p:
            # 执行重复次数的 torch.ones((1,)) 操作
            for _ in range(repeats):
                torch.ones((1,))
            # 手动触发垃圾回收
            gc.collect()

        # 获取性能分析器的事件树根节点
        roots = p.profiler.kineto_results.experimental_event_tree()
        # 创建一个集合，用于存放分配 ID
        id_set = set()
        # 遍历事件树，提取所有 tensor 操作的分配 ID
        for e in _utils.traverse_dfs(roots):
            fields = e.extra_fields
            # 如果是 torch 操作，将其输入 tensor 的分配 ID 加入集合
            if isinstance(fields, torch._C._profiler._ExtraFields_TorchOp):
                id_set |= {
                    t.allocation_id
                    for t in fields.inputs
                    if isinstance(t, _TensorMetadata)
                }
            # 如果是分配信息，将分配 ID 加入集合
            elif isinstance(fields, torch._C._profiler._ExtraFields_Allocation):
                id_set.add(fields.allocation_id)

        # 移除集合中的 None 值
        id_set.difference_update([None])
        # 断言集合中分配 ID 的数量与重复次数相等，确保每个 ID 只出现一次
        self.assertEqual(repeats, len(id_set))

    # 定义测试函数 test_extra_fields，用于测试额外字段的存在和类型
    def test_extra_fields(self):
        # 使用性能分析器 profile，并记录堆栈信息和内存使用
        with profile(with_stack=True, profile_memory=True) as p:
            # 执行 torch.ones((1,)) 操作
            _ = torch.ones((1,))

        # 获取性能分析器的事件树根节点
        nodes = p.profiler.kineto_results.experimental_event_tree()
        # 查找特定名称的节点 "aten::ones"
        node = find_node_with_name(nodes, "aten::ones")
        # 断言找到符合条件的节点
        self.assertIsNotNone(node)

        # 断言节点的额外字段类型为 TorchOp 类型
        self.assertIsInstance(
            node.extra_fields, torch._C._profiler._ExtraFields_TorchOp
        )

        # 断言节点的父节点的额外字段类型为 PyCCall 类型
        self.assertIsInstance(
            node.parent.extra_fields, torch._C._profiler._ExtraFields_PyCCall
        )

        # 断言节点的第一个子节点的名称为 "aten::empty"
        self.assertEqual(node.children[0].name, "aten::empty")
        # 断言子节点的第一个子节点的名称为 "[memory]"
        self.assertEqual(node.children[0].children[0].name, "[memory]")
        # 断言子节点的第一个子节点的额外字段类型为 Allocation 类型
        self.assertIsInstance(
            node.children[0].children[0].extra_fields,
            torch._C._profiler._ExtraFields_Allocation,
        )
    def test_tensor_properties(self):
        # 创建一个 10x10 的张量，并使用 as_strided 方法改变其形状为 4x4，同时指定步长
        x = torch.ones(10, 10).as_strided([4, 4], [12, 3])
        # 创建一个 4x1 的张量，要求计算其梯度
        y = torch.ones(4, 1, requires_grad=True)

        # 使用 profiler 开始性能分析，记录堆栈信息、内存使用和张量形状
        with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
            # 执行张量加法
            _ = x + y
            # 执行张量乘法
            _ = x * y

        # 获取性能分析器的事件树
        nodes = p.profiler.kineto_results.experimental_event_tree()
        # 根据操作名称查找节点信息
        node = find_node_with_name(nodes, "aten::add")
        # 断言节点信息不为空
        self.assertIsNotNone(node)

        # 断言节点额外字段属于 Torch 操作的额外字段类型
        self.assertIsInstance(
            node.extra_fields, torch._C._profiler._ExtraFields_TorchOp
        )

        # 定义一个函数，用于获取节点的特定输入属性
        def getattr_inputs(name, default):
            return [getattr(i, name, default) for i in node.extra_fields.inputs]

        # 断言节点的输入大小属性
        self.assertEqual(getattr_inputs("sizes", []), [[4, 4], [4, 1], []])
        # 断言节点的步长属性
        self.assertEqual(getattr_inputs("strides", []), [[12, 3], [1, 1], []])
        # 断言节点的布局属性
        self.assertEqual(
            getattr_inputs("layout", None), [torch.strided, torch.strided, None]
        )
        # 断言节点的设备属性
        self.assertEqual(
            getattr_inputs("device", None),
            [torch.device("cpu"), torch.device("cpu"), None],
        )
        # 断言节点的数据类型属性
        self.assertEqual(
            getattr_inputs("dtype", None), [torch.float32, torch.float32, None]
        )
        # 断言节点的作用域属性为函数级别的性能记录
        self.assertEqual(node.extra_fields.scope, torch.profiler.RecordScope.FUNCTION)

        # 查找乘法操作节点
        mul_node = find_node_with_name(nodes, "aten::mul")
        # 断言乘法节点信息不为空
        self.assertIsNotNone(mul_node)
        # 断言乘法节点的序列号是加法节点序列号加一
        self.assertEqual(
            node.extra_fields.sequence_number + 1, mul_node.extra_fields.sequence_number
        )

    def test_sparse_tensors(self):
        # 定义稀疏张量的索引和值
        i = [[0, 1, 1], [2, 0, 2]]
        v = [3, 4, 5]
        # 创建一个 2x3 的稀疏 COO 张量
        s = torch.sparse_coo_tensor(i, v, (2, 3))

        # 使用 profiler 开始性能分析，记录堆栈信息、内存使用和张量形状
        with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
            # 执行稀疏张量的加法
            _ = s + s

        # 获取性能分析器的事件树
        nodes = p.profiler.kineto_results.experimental_event_tree()
        # 根据操作名称查找节点信息
        node = find_node_with_name(nodes, "aten::add")
        # 断言节点信息不为空
        self.assertIsNotNone(node)

        # 断言节点额外字段属于 Torch 操作的额外字段类型
        self.assertIsInstance(
            node.extra_fields, torch._C._profiler._ExtraFields_TorchOp
        )

        # 定义一个函数，用于获取节点的特定输入属性
        def getattr_inputs(name, default):
            return [getattr(i, name, default) for i in node.extra_fields.inputs]

        # 断言节点的输入大小属性
        self.assertEqual(getattr_inputs("sizes", []), [[2, 3], [2, 3], []])
        # 断言节点的步长属性
        self.assertEqual(getattr_inputs("strides", []), [[], [], []])
        # 断言节点的布局属性
        self.assertEqual(
            getattr_inputs("layout", None), [torch.sparse_coo, torch.sparse_coo, None]
        )
        # 断言节点的设备属性
        self.assertEqual(
            getattr_inputs("device", None),
            [torch.device("cpu"), torch.device("cpu"), None],
        )
    def test_mkldnn_tensors(self):
        # 创建一个 4x3 的全一张量，并将其转换为 MKLDNN 格式
        x = torch.ones(4, 3).to_mkldnn()

        # 使用 profile 对象进行性能分析，记录堆栈信息、内存使用和形状
        with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
            # 执行张量的加法操作
            _ = x + x

        # 从性能分析结果中获取事件树
        nodes = p.profiler.kineto_results.experimental_event_tree()
        # 在事件树中查找名称为 "aten::add" 的节点
        node = find_node_with_name(nodes, "aten::add")
        # 断言节点不为 None
        self.assertIsNotNone(node)

        # 断言节点的额外字段属于 TorchOp 类型
        self.assertIsInstance(
            node.extra_fields, torch._C._profiler._ExtraFields_TorchOp
        )

        # 定义一个函数，获取节点额外字段中名为 name 的属性，如果不存在返回 default
        def getattr_inputs(name, default):
            return [getattr(i, name, default) for i in node.extra_fields.inputs]

        # 断言节点的输入的 sizes 属性与预期值相等
        self.assertEqual(getattr_inputs("sizes", []), [[4, 3], [4, 3], []])
        # 断言节点的输入的 strides 属性与预期值相等
        self.assertEqual(getattr_inputs("strides", []), [[], [], []])
        # 断言节点的输入的 layout 属性与预期值相等
        self.assertEqual(
            getattr_inputs("layout", None), [torch._mkldnn, torch._mkldnn, None]
        )
        # 断言节点的输入的 device 属性与预期值相等
        self.assertEqual(
            getattr_inputs("device", None),
            [torch.device("cpu"), torch.device("cpu"), None],
        )

    def test_scalar_ins(self):
        # 创建一个 5x5 的全一张量
        x = torch.ones(5, 5)
        # 定义一个 alpha 值
        alpha = 0.9

        # 使用 profile 对象进行性能分析，记录堆栈信息、内存使用和形状
        with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
            # 执行张量的加法操作，指定 alpha 参数
            _ = torch.add(x, 9.1, alpha=alpha)

        # 从性能分析结果中获取事件树
        nodes = p.profiler.kineto_results.experimental_event_tree()
        # 在事件树中查找名称为 "aten::add" 的节点
        node = find_node_with_name(nodes, "aten::add")
        # 断言节点不为 None
        self.assertIsNotNone(node)

        # 定义一个函数，获取节点额外字段中名为 name 的属性，如果不存在返回 default
        def getattr_inputs(name, default):
            return [getattr(i, name, default) for i in node.extra_fields.inputs]

        # 断言节点的 dtype 属性与预期值相等
        # 第二个参数被提升为 zerodim 张量
        self.assertEqual(
            getattr_inputs("dtype", None), [torch.float32, torch.float64, None]
        )
        # 断言节点的输入的 sizes 属性与预期值相等
        self.assertEqual(getattr_inputs("sizes", []), [[5, 5], [], []])
        # 断言节点的输入的第三个输入与预期的 alpha 参数相等
        self.assertEqual(node.extra_fields.inputs[2], alpha)

    def test_tensor_lists(self):
        # 创建两个形状为 (1,) 的全一张量
        x = torch.ones((1,))
        y = torch.ones((1,))
        # 使用 profile 对象进行性能分析，记录堆栈信息、内存使用和形状
        with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
            # 执行张量的堆叠操作
            _ = torch.stack((x, y))

        # 从性能分析结果中获取事件树
        nodes = p.profiler.kineto_results.experimental_event_tree()
        # 在事件树中查找名称为 "aten::stack" 的节点
        node = find_node_with_name(nodes, "aten::stack")
        # 获取节点的额外字段中的输入列表
        inputs = node.extra_fields.inputs
        # 断言输入列表的长度为 2
        self.assertEqual(len(inputs), 2)
        # 断言输入列表中的第一个元素为列表类型
        self.assertIsInstance(inputs[0], list)
        # 断言列表的长度为 2
        self.assertEqual(len(inputs[0]), 2)
        # 断言 x 的存储数据指针与输入列表中第一个元素的存储数据指针相等
        self.assertEqual(x.storage().data_ptr(), inputs[0][0].storage_data_ptr)
        # 断言 y 的存储数据指针与输入列表中第二个元素的存储数据指针相等
        self.assertEqual(y.storage().data_ptr(), inputs[0][1].storage_data_ptr)
    def test_nnmodule_params(self):
        # 定义内部函数，用于扁平化提取节点的额外字段中的模块参数
        def flat_out_extrafields(nodes, out=None):
            # 如果输出列表为空，初始化为空列表
            if out is None:
                out = []
            # 遍历节点列表
            for node in nodes:
                # 检查节点的额外字段是否是 _ExtraFields_PyCall 类型，并且有模块存在
                if (
                    isinstance(node.extra_fields, _ExtraFields_PyCall)
                    and node.extra_fields.module
                ):
                    # 如果节点中的模块有参数，则将其添加到输出列表中
                    if node.extra_fields.module.parameters:
                        out.append(node.extra_fields.module)
                # 递归调用，处理节点的子节点
                flat_out_extrafields(node.children, out)
            return out

        # 创建输入张量
        inputs = torch.rand(10)
        # 实例化 SimpleNet 模型
        net = SimpleNet()
        # 将输入张量输入模型，获取输出
        out = net(inputs)
        # 计算输出与随机张量之间的交叉熵损失，并进行反向传播
        torch.nn.functional.cross_entropy(out, torch.rand(2)).backward()
        # 使用 torch.profiler 进行性能分析和内存分析
        with torch.profiler.profile(with_stack=True, profile_memory=True) as p:
            _ = net(inputs)

        # 从性能分析器中提取实验性事件树，获取模块列表
        modules = flat_out_extrafields(
            p.profiler.kineto_results.experimental_event_tree()
        )
        # 断言模块列表的长度为 2，验证预期的参数列表数量
        self.assertEqual(
            len(modules), 2, f"Expected two parameter list, but got {len(modules)}"
        )

        # 从模块列表中提取参数元组列表，包括参数名称、存储数据指针和梯度数据指针
        params = [
            (n, p.storage_data_ptr, g.storage_data_ptr)
            for module in modules
            for (n, p, g) in module.parameters
        ]

        # 获取预期的参数元组列表，针对网络模型的每个线性层（fc1 和 fc2）
        expected = [
            (name, val.storage().data_ptr(), val.grad.storage().data_ptr())
            for name, val in net.fc1._parameters.items()
        ]
        expected += [
            (name, val.storage().data_ptr(), val.grad.storage().data_ptr())
            for name, val in net.fc2._parameters.items()
        ]

        # 断言预期参数列表与实际提取的参数列表相等
        self.assertEqual(expected, params, f"{expected} vs. {params}")

    def _flat_out_extrafields(self, nodes, out=None):
        # 如果输出列表为空，初始化为空列表
        if out is None:
            out = []
        # 遍历节点列表
        for node in nodes:
            # 检查节点的额外字段是否是 _ExtraFields_PyCall 类型，并且有优化器和参数存在
            if (
                isinstance(node.extra_fields, _ExtraFields_PyCall)
                and node.extra_fields.optimizer
                and node.extra_fields.optimizer.parameters
            ):
                # 避免在迭代中出现重复的 OptInfo，获取优化器参数的存储数据指针
                addr = node.extra_fields.optimizer.parameters[0][0].storage_data_ptr
                # 如果输出列表中不存在具有相同存储数据指针的 OptInfo，则添加到输出列表中
                if not [o for o in out if addr == o.parameters[0][0].storage_data_ptr]:
                    out.append(node.extra_fields.optimizer)
            # 递归调用，处理节点的子节点
            self._flat_out_extrafields(node.children, out)
        return out
    # 检查优化器结果的私有方法。验证期望得到一个优化器，并输出相关信息如果与期望不符。
    def _check_results(self, opt, opts, check_items=False):
        # 断言期望得到一个优化器列表，如果不符则输出错误信息。
        self.assertEqual(len(opts), 1, f"Expected 1 optimizer: len(opts): {len(opts)}")
        # 断言给定的优化器对象与第一个优化器对象的身份标识符相同，如果不符则输出错误信息。
        self.assertEqual(
            id(opt),
            opts[0].self_ptr,
            f"Optimizer addr ({id(opt)}) vs. profiled addr ({opts[0].self_ptr})",
        )
        # 如果需要检查每个参数组，则进行以下验证。
        if check_items:
            # 断言优化器参数组的数量与优化器列表中的优化器数量相同。
            self.assertEqual(len(opt.param_groups), len(opts))
            # 逐个验证每个参数组中的参数的存储数据指针与优化器列表中对应优化器的参数的存储数据指针是否一致。
            for group, opt_ in zip(opt.param_groups, opts):
                self.assertEqual(
                    [(v.storage().data_ptr()) for v in group.get("params", [])],
                    [(o.storage_data_ptr) for (o, _, _) in opt_.parameters],
                )
            # 针对每个优化器对象，验证其观察到的状态字典是否正确收集和记录了优化器状态。
            for opt_ in opts:
                observed_state = {
                    p.storage_data_ptr: {name: s.storage_data_ptr for name, s in state}
                    for (p, _, state) in opt_.parameters
                }

                # 确保分析器收集了所有优化器状态，并检查分析器记录的地址是否正确。
                for parameter, parameter_state in opt.state.items():
                    self.assertEqual(
                        {
                            name: value.storage().data_ptr()
                            for name, value in parameter_state.items()
                        },
                        observed_state.get(parameter.storage().data_ptr(), []),
                    )

    # 测试优化器行为的方法。
    def test_optimizer(self):
        # 准备输入数据
        inputs = torch.rand(10)
        # 使用分析器进行性能和内存分析
        with torch.profiler.profile(with_stack=True, profile_memory=True) as p:
            # 创建简单神经网络模型
            net = SimpleNet()
            # 创建 SGD 优化器
            opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

            # 将梯度置零
            opt.zero_grad()
            # 将输入数据传递给网络
            out = net(inputs)
            # 计算损失
            loss = torch.nn.functional.cross_entropy(out, torch.rand(2))
            # 反向传播计算梯度
            loss.backward()
            # 执行优化步骤
            opt.step()
        # 使用私有方法检查优化器的结果
        self._check_results(
            opt,
            self._flat_out_extrafields(
                p.profiler.kineto_results.experimental_event_tree()
            ),
            False,
        )

    # 测试不同优化器参数行为的方法。
    def _test_optimizer_parameters(self, optimizer_factory):
        # 准备输入数据
        inputs = torch.rand(10)
        # 使用分析器进行性能和内存分析
        with torch.profiler.profile(with_stack=True, profile_memory=True) as p:
            # 创建简单神经网络模型
            net = SimpleNet()
            # 使用给定的优化器工厂函数创建优化器
            opt = optimizer_factory(net.parameters())
            # 执行两轮优化
            for _ in range(2):
                # 将梯度置零
                opt.zero_grad()
                # 将输入数据传递给网络
                out = net(inputs)
                # 计算损失
                loss = torch.nn.functional.cross_entropy(out, torch.rand(2))
                # 反向传播计算梯度
                loss.backward()
                # 执行优化步骤
                opt.step()
        # 使用私有方法检查优化器的结果
        self._check_results(
            opt,
            self._flat_out_extrafields(
                p.profiler.kineto_results.experimental_event_tree()
            ),
            True,
        )

    # 测试 SGD 优化器参数行为的方法。
    def test_optimizer_parameters_sgd(self):
        # 调用 _test_optimizer_parameters 方法，使用 lambda 表达式创建 SGD 优化器
        self._test_optimizer_parameters(
            lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9)
        )
    # 测试优化器参数是否正确设置为 Adam
    def test_optimizer_parameters_adam(self):
        # 调用通用的优化器参数测试函数，使用 Adam 优化器
        self._test_optimizer_parameters(
            lambda params: torch.optim.Adam(params, foreach=True)
        )

    # 测试内存分配和释放情况
    def test_allocations(self):
        # 手动触发垃圾回收
        gc.collect()
        # 使用性能分析器记录内存分配情况
        with profile(profile_memory=True) as p:
            x = torch.empty((3, 4))

        # 获取性能分析结果中的事件树
        nodes = p.profiler.kineto_results.experimental_event_tree()
        # 查找包含 "[memory]" 标记的节点
        node = find_node_with_name(nodes, "[memory]")
        # 确保找到了匹配的节点
        self.assertIsNotNone(node)

        # 计算预期的内存分配大小
        alloc_size = 3 * 4 * 4  # fp32 -> 4 字节
        # 获取节点的额外字段
        ptr = node.extra_fields.ptr
        # 断言指针大于零，即内存地址有效
        self.assertGreater(ptr, 0)
        # 断言节点的分配大小与预期相符
        self.assertEqual(node.extra_fields.alloc_size, alloc_size)
        # 断言节点的设备为 CPU
        self.assertEqual(node.extra_fields.device, torch.device("cpu"))
        # 获取节点的总分配量
        total_allocated = node.extra_fields.total_allocated

        # total_reserved 仅适用于 CUDACachingAllocator
        self.assertEqual(node.extra_fields.total_reserved, 0)

        # 再次使用性能分析器记录内存分配情况
        with profile(profile_memory=True) as p:
            del x
            gc.collect()

        # 获取更新后的性能分析结果中的事件树
        nodes = p.profiler.kineto_results.experimental_event_tree()
        # 再次查找包含 "[memory]" 标记的节点
        node = find_node_with_name(nodes, "[memory]")
        # 确保找到了匹配的节点
        self.assertIsNotNone(node)

        # 断言再次分配内存时的节点指针与之前相同
        self.assertEqual(node.extra_fields.ptr, ptr)
        # 断言再次分配内存时的节点分配大小为负的预期分配大小
        self.assertEqual(node.extra_fields.alloc_size, -alloc_size)
        # 断言再次分配内存时的节点设备为 CPU
        self.assertEqual(node.extra_fields.device, torch.device("cpu"))
        # 断言再次分配内存时的总分配量比之前减少了预期分配大小
        self.assertEqual(
            node.extra_fields.total_allocated, total_allocated - alloc_size
        )

    # 测试引用计数的情况
    def test_refcounts(self):
        # 定义一个 Sentinel 类来测试对象生命周期
        class Sentinel:
            pass

        # 创建一个工厂函数，确保测试范围内不会出现强引用
        def make():
            # 外部闭包内创建一个外部 Sentinel 实例
            outer_sentinel = Sentinel()

            def outer():
                # Python 只会闭合函数内部使用的变量
                _ = outer_sentinel
                # 内部函数内创建一个内部 Sentinel 实例
                inner_sentinel = Sentinel()

                def inner():
                    _ = inner_sentinel

                # 使用带堆栈信息的性能分析器记录内部函数调用
                with profile(with_stack=True):
                    inner()

                # 返回内部 Sentinel 实例的弱引用
                return weakref.ref(inner_sentinel)

            # 返回外部函数和外部 Sentinel 实例的弱引用
            return outer, weakref.ref(outer_sentinel)

        # 使用工厂函数创建外部函数和外部 Sentinel 实例的弱引用
        outer, outer_sentinel_ref = make()
        # 调用外部函数，获取内部 Sentinel 实例的弱引用
        inner_sentinel_ref = outer()

        # 断言内部 Sentinel 实例的弱引用为空
        self.assertIsNone(inner_sentinel_ref())

        # 断言外部 Sentinel 实例的弱引用不为空
        self.assertIsNotNone(outer_sentinel_ref())

        # 删除外部函数，断言外部 Sentinel 实例的弱引用为空
        del outer
        self.assertIsNone(outer_sentinel_ref())
# 如果当前脚本被直接执行（而不是被导入到其他脚本中执行），则执行下面的代码块
if __name__ == "__main__":
    # 调用函数 run_tests()，用于执行测试函数或者测试套件
    run_tests()
```