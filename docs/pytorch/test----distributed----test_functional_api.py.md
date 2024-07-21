# `.\pytorch\test\distributed\test_functional_api.py`

```
# Owner(s): ["oncall: distributed"]

import os  # 导入操作系统模块
import sys  # 导入系统相关的模块
import unittest  # 导入单元测试框架
from functools import partial, wraps  # 导入函数工具模块中的部分功能

import torch  # 导入PyTorch深度学习库
import torch.distributed as dist  # 导入PyTorch分布式模块
import torch.distributed._functional_collectives as ft_c  # 导入PyTorch分布式功能集合模块
import torch.distributed._tensor as dt  # 导入PyTorch分布式张量模块
import torch.distributed.distributed_c10d as c10d  # 导入PyTorch分布式C10d模块

from functorch import make_fx  # 导入functorch库中的make_fx函数
from torch._inductor.utils import run_and_get_code  # 导入torch._inductor.utils中的run_and_get_code函数
from torch.testing import FileCheck  # 导入PyTorch测试工具中的FileCheck模块
from torch.testing._internal.distributed.fake_pg import FakeStore  # 导入PyTorch内部分布式测试工具中的FakeStore类
from torch.utils._triton import has_triton  # 导入torch.utils._triton中的has_triton函数

if not dist.is_available():  # 如果分布式功能不可用
    print("Distributed not available, skipping tests", file=sys.stderr)  # 输出提示信息到标准错误流
    sys.exit(0)  # 退出程序，返回状态码0（表示正常退出）

from torch.testing._internal.common_distributed import (  # 导入PyTorch内部分布式通用测试工具中的
    MultiProcessTestCase,  # MultiProcessTestCase类
    MultiThreadedTestCase,  # MultiThreadedTestCase类
    requires_nccl,  # requires_nccl装饰器函数
    TEST_SKIPS,  # TEST_SKIPS变量
)
from torch.testing._internal.common_utils import (  # 导入PyTorch内部通用工具中的
    instantiate_parametrized_tests,  # instantiate_parametrized_tests函数
    parametrize,  # parametrize装饰器函数
    run_tests,  # run_tests函数
    TestCase,  # TestCase类
)


def new_subgroups(group_size: int, pg_tag=None):
    world_size = dist.get_world_size()  # 获取当前环境的进程数量
    subgroups = []  # 初始化子组列表
    cur_subgroup = None  # 初始化当前子组为None

    for subgroup_id in range(world_size // group_size):  # 遍历每个子组的ID范围
        start_rank = subgroup_id * group_size  # 计算子组内的起始进程排名
        end_rank = start_rank + group_size  # 计算子组内的结束进程排名
        ranks_in_subgroup = list(range(start_rank, end_rank))  # 生成子组内进程排名列表
        subgroup = c10d._new_group_with_tag(  # 使用C10d模块创建带有标签的新组
            ranks=ranks_in_subgroup,  # 指定组中的进程排名列表
            pg_tag=pg_tag,  # 指定组的标签
        )
        subgroups.append(subgroup)  # 将新创建的子组添加到子组列表中

        rank = dist.get_rank()  # 获取当前进程的排名
        if rank in ranks_in_subgroup:  # 如果当前进程在当前子组中
            cur_subgroup = subgroup  # 更新当前子组为当前创建的子组

    return cur_subgroup, subgroups  # 返回当前子组和所有子组列表


class TestExpand(MultiThreadedTestCase):
    @property
    def world_size(self):
        return 4  # 返回测试中使用的进程数量

    def setUp(self):
        super().setUp()  # 调用父类的setUp方法进行初始化
        self._spawn_threads()  # 启动多线程测试

    def test_expand_1d_rank_list(self):
        tag, rankset, group_size = ft_c._expand_group([0, 1, 2, 3])  # 执行分组扩展，返回标签、排名集合和组大小
        self.assertEqual("", tag)  # 断言标签为空字符串
        self.assertEqual([0, 1, 2, 3], rankset)  # 断言排名集合与预期相同
        self.assertEqual(4, group_size)  # 断言组大小为4

        tag, rankset, group_size = ft_c._expand_group([0, 1, 2, 3], "bla")  # 执行带标签的分组扩展
        self.assertEqual("bla", tag)  # 断言标签为"bla"

    def test_expand_2d_rank_list(self):
        tag, rankset, group_size = ft_c._expand_group([[0, 1], [2, 3]])  # 执行二维排名列表的分组扩展
        self.assertEqual("", tag)  # 断言标签为空字符串
        self.assertEqual([0, 1, 2, 3], rankset)  # 断言排名集合与预期相同
        self.assertEqual(2, group_size)  # 断言组大小为2

        tag, rankset, group_size = ft_c._expand_group([[0, 1], [2, 3]], "blu")  # 执行带标签的二维排名列表分组扩展
        self.assertEqual("blu", tag)  # 断言标签为"blu"

        with self.assertRaisesRegex(ValueError, "group sizes must be identical"):  # 断言引发值错误异常，异常消息为"group sizes must be identical"
            ft_c._expand_group([[0], [1, 2, 3]])  # 尝试扩展不同大小的二维排名列表
    # 定义测试方法：测试扩展处理组的功能
    def test_expand_process_group(self):
        # 调用 _expand_group 方法扩展 WORLD 组
        tag, rankset, group_size = ft_c._expand_group(dist.group.WORLD)
        # 断言：验证 WORLD 组的标签与预期相符
        self.assertEqual(c10d._get_group_tag(dist.group.WORLD), tag)
        # 断言：验证 WORLD 组的排名集合与预期相符
        self.assertEqual([0, 1, 2, 3], rankset)
        # 断言：验证 WORLD 组的大小与预期相符
        self.assertEqual(4, group_size)

        # 调用 _expand_group 方法扩展 WORLD 组，传入自定义标签 "bla"
        tag, rankset, group_size = ft_c._expand_group(dist.group.WORLD, "bla")
        # 断言：验证标签与预期的自定义标签相符
        self.assertEqual("bla", tag)

        # 创建新的子组并返回我的进程组和其他组
        my_pg, others = new_subgroups(group_size=2)
        # 调用 _expand_group 方法扩展我的进程组
        tag, rankset, group_size = ft_c._expand_group(my_pg)
        # 断言：验证我的进程组的标签与预期相符
        self.assertEqual(c10d._get_group_tag(my_pg), tag)
        # 断言：验证我的进程组的排名集合与预期相符
        self.assertEqual(dist.get_process_group_ranks(my_pg), rankset)
        # 断言：验证我的进程组的大小与预期相符
        self.assertEqual(2, group_size)

        # 将我的进程组设为 None
        my_pg = None
        # 遍历所有的世界大小
        for i in range(dist.get_world_size()):
            # 创建新的带有特定标签的组
            group = c10d._new_group_with_tag([i], pg_tag="my_pg")
            # 如果当前进程的排名等于 i
            if i == dist.get_rank():
                # 将我的进程组设为新创建的组
                my_pg = group
        # 调用 _expand_group 方法扩展我的进程组
        tag, rankset, group_size = ft_c._expand_group(my_pg)
        # 断言：验证我的进程组的标签与预期相符
        self.assertEqual("my_pg", tag)
        # 断言：验证我的进程组的排名集合与预期相符
        self.assertEqual([dist.get_rank()], rankset)
        # 断言：验证我的进程组的大小与预期相符
        self.assertEqual(1, group_size)

        # 调用 _expand_group 方法扩展我的进程组，传入自定义标签 "bla"
        tag, rankset, group_size = ft_c._expand_group(my_pg, "bla")
        # 断言：验证标签与预期的自定义标签相符
        self.assertEqual("bla", tag)

    # 定义测试方法：测试扩展设备网格的功能
    def test_expand_device_mesh(self):
        # 创建一个 CPU 设备网格
        mesh = dt.DeviceMesh("cpu", torch.arange(4))
        # 调用 _expand_group 方法扩展设备网格
        tag, rankset, group_size = ft_c._expand_group(mesh)
        # 断言：验证设备网格的标签与预期相符
        self.assertEqual(c10d._get_group_tag(mesh.get_group(mesh_dim=0)), tag)
        # 断言：验证设备网格的排名集合与预期相符
        self.assertEqual([0, 1, 2, 3], rankset)
        # 断言：验证设备网格的大小与预期相符
        self.assertEqual(4, group_size)

        # 创建一个 CPU 设备网格
        mesh = dt.DeviceMesh("cpu", torch.arange(4))
        # 调用 _expand_group 方法扩展设备网格
        tag, rankset, group_size = ft_c._expand_group(mesh)
        # 断言：验证设备网格的标签与预期相符
        self.assertEqual(c10d._get_group_tag(mesh.get_group(mesh_dim=0)), tag)
        # 断言：验证设备网格的排名集合与预期相符
        self.assertEqual([0, 1, 2, 3], rankset)
        # 断言：验证设备网格的大小与预期相符
        self.assertEqual(4, group_size)

    # 定义测试方法：测试扩展设备网格元组的功能
    def test_expand_device_mesh_tuple(self):
        # 创建一个 CPU 设备网格
        mesh = dt.DeviceMesh("cpu", torch.arange(4).view(2, 2))
        # 使用断言验证只能处理一维网格的情况
        with self.assertRaisesRegex(AssertionError, "Only 1D mesh"):
            tag, rankset, group_size = ft_c._expand_group(mesh)

        # 调用 _expand_group 方法扩展设备网格元组
        tag, rankset, group_size = ft_c._expand_group((mesh, 0))
        # 断言：验证设备网格的标签与预期相符
        self.assertEqual(c10d._get_group_tag(mesh.get_group(mesh_dim=0)), tag)
        # 预期的排名集合根据当前进程的排名确定
        expected_rankset = [0, 2] if dist.get_rank() in [0, 2] else [1, 3]
        # 断言：验证排名集合与预期相符
        self.assertEqual(expected_rankset, rankset)
        # 断言：验证设备网格的大小与预期相符
        self.assertEqual(2, group_size)

        # 调用 _expand_group 方法扩展设备网格元组
        tag, rankset, group_size = ft_c._expand_group((mesh, 1))
        # 预期的排名集合根据当前进程的排名确定
        expected_rankset = [0, 1] if dist.get_rank() in [0, 1] else [2, 3]
        # 断言：验证设备网格的标签与预期相符
        self.assertEqual(c10d._get_group_tag(mesh.get_group(mesh_dim=1)), tag)
        # 断言：验证排名集合与预期相符
        self.assertEqual(expected_rankset, rankset)
        # 断言：验证设备网格的大小与预期相符
        self.assertEqual(2, group_size)
class TestPgTag(MultiThreadedTestCase):
    @property
    # 返回固定的世界大小为4
    def world_size(self):
        return 4

    def setUp(self):
        super().setUp()
        # 调用父类的 setUp 方法初始化测试环境
        self._spawn_threads()

    """
    我们期望的行为如下：

    - rankset + tag 将始终导致相同的 PG。
    我们通过失败创建新的 PG 或返回现有的 PG 来实施这一点吗？
        返回现有的 PG。

    - 默认标签给出现有的行为。
    这意味着我们应该创建重复的 PG。
    - _expand_group 在具有 _default 标签的 PG 上应始终解析到它
    这意味着我们不能依赖于空标签 + rankset。
    """

    def test_pg_creation_with_tag(self):
        # 创建具有标签 "blu" 的新子组
        my_group, _ = new_subgroups(group_size=2, pg_tag="blu")
        my_group2, _ = new_subgroups(group_size=2, pg_tag="blu")
        # 断言两个具有相同标签的 PG 是同一个对象
        self.assertEqual(my_group, my_group2)

        # 创建具有不同标签 "blu2" 的新子组
        my_group3, _ = new_subgroups(group_size=2, pg_tag="blu2")
        # 断言具有不同标签的 PG 是不同的对象
        self.assertNotEqual(my_group, my_group3)

        # 创建没有指定标签的新子组
        my_group4, _ = new_subgroups(group_size=2)
        # 断言具有标签和没有标签的 PG 是不同的对象
        self.assertNotEqual(my_group, my_group4)

        # 再次创建没有指定标签的新子组
        my_group5, _ = new_subgroups(group_size=2)
        # 断言两个没有标签的 PG 是不同的对象
        self.assertNotEqual(my_group4, my_group5)

    def test_pg_lookup_roundtrip(self):
        # 创建具有标签 "blu" 的新子组
        pg_tag0, _ = new_subgroups(group_size=2, pg_tag="blu")
        # 创建具有标签 "blu2" 的新子组
        pg_tag1, _ = new_subgroups(group_size=2, pg_tag="blu2")
        # 创建没有指定标签的新子组
        pg_notag0, _ = new_subgroups(group_size=2)
        # 再次创建没有指定标签的新子组
        pg_notag1, _ = new_subgroups(group_size=2)

        def roundtrip(pg):
            # 调用 _expand_group 方法解析 PG
            tag, rankset, _ = ft_c._expand_group(pg)
            # 调用 _find_pg_by_ranks_and_tag 方法查找 PG
            return c10d._find_pg_by_ranks_and_tag(tag, rankset)

        # 断言 PG 查找结果正确
        self.assertEqual(pg_tag0, roundtrip(pg_tag0))
        self.assertEqual(pg_tag1, roundtrip(pg_tag1))
        self.assertEqual(pg_notag0, roundtrip(pg_notag0))
        self.assertEqual(pg_notag1, roundtrip(pg_notag1))

    def test_pg_lookup_with_tag(self):
        # 创建具有标签 "blu" 的新子组
        pg_tag0, _ = new_subgroups(group_size=2, pg_tag="blu")
        # 创建具有标签 "bla" 的新子组
        pg_tag1, _ = new_subgroups(group_size=2, pg_tag="bla")
        # 创建没有指定标签的新子组
        pg_notag0, _ = new_subgroups(group_size=2)

        def roundtrip(pg, pg_tag):
            # 调用 _expand_group 方法解析 PG 和标签
            tag, rankset, _ = ft_c._expand_group(pg, pg_tag)
            # 调用 _find_pg_by_ranks_and_tag 方法查找 PG
            return c10d._find_pg_by_ranks_and_tag(tag, rankset)

        # 断言 PG 查找结果正确
        self.assertEqual(pg_tag0, roundtrip(pg_tag1, "blu"))
        self.assertEqual(pg_tag0, roundtrip(pg_notag0, "blu"))
        # 不能擦除 PG 的标签
        self.assertEqual(pg_tag0, roundtrip(pg_tag0, ""))

    def test_find_or_create_pg(self):
        # 调用 _find_or_create_pg_by_ranks_and_tag 方法查找或创建 PG
        pg = c10d._find_or_create_pg_by_ranks_and_tag("blu", [0, 1, 2, 3], 2)
        # 创建具有标签 "blu" 的新子组
        pg_tag0, _ = new_subgroups(group_size=2, pg_tag="blu")
        # 断言 PG 查找或创建结果正确
        self.assertEqual(pg, pg_tag0)

    def test_find_root_pg(self):
        # 调用 _find_pg_by_ranks_and_tag 方法查找根 PG
        pg = c10d._find_pg_by_ranks_and_tag("", [0, 1, 2, 3])
        # 断言根 PG 是世界级分组
        self.assertEqual(dist.group.WORLD, pg)
    @parametrize("device", ["cpu", "cuda"])
    def test_broadcast(self, device):
        # 如果设备为 CUDA，则检查 CUDA 设备数量是否足够
        if device == "cuda":
            if torch.cuda.device_count() < self.world_size:
                self.skipTest("Not enough CUDA devices")
            # 设置当前进程的 CUDA 设备
            torch.cuda.set_device(dist.get_rank())

        # 如果当前进程的 rank 是 0，则创建一个全为 1 的张量
        if dist.get_rank() == 0:
            tensor = torch.ones([4], device=device)
        else:
            # 否则创建一个全为 0 的张量
            tensor = torch.zeros([4], device=device)

        # 创建设备网格对象，并传入当前设备和一个包含 [0, 1, 2, 3] 的张量
        mesh = dt.DeviceMesh(device, torch.arange(4))
        # 对张量 tensor 在网格 mesh 上进行广播操作，将结果保存在 res 中
        res = ft_c.broadcast(tensor, 0, mesh)
        # 断言广播后的结果与预期的全为 1 的张量相等
        self.assertEqual(res, torch.ones([4], device=device))

    @parametrize("device", ["cpu", "cuda"])
    def test_all_reduce_eager(self, device):
        # 如果设备为 CUDA，则检查 CUDA 设备数量是否足够
        if device == "cuda":
            if torch.cuda.device_count() < self.world_size:
                self.skipTest("Not enough CUDA devices")
            # 设置当前进程的 CUDA 设备
            torch.cuda.set_device(dist.get_rank())

        # 创建一个全为 1 的张量 tensor
        tensor = torch.ones([4], device=device)
        # 创建设备网格对象，并传入当前设备和一个包含 [0, 1, 2, 3] 的张量
        mesh = dt.DeviceMesh(device, torch.arange(4))

        # 对张量 tensor 在网格 mesh 上进行全局归约（求和），将结果保存在 res 中
        res = ft_c.all_reduce(tensor, "sum", mesh)
        # 断言归约后的结果与预期的全为 4 的张量相等
        self.assertEqual(res, torch.tensor([4, 4, 4, 4], dtype=torch.float))

        # 创建一个二维网格对象，并传入当前设备和一个形状为 [2, 2] 的张量
        mesh = dt.DeviceMesh(device, torch.arange(4).view(2, 2))
        # 对张量 tensor 在网格 mesh 上进行全局归约（求和），将结果保存在 res2 中
        res2 = ft_c.all_reduce(tensor, "sum", (mesh, 1))
        # 断言归约后的结果与预期的全为 2 的张量相等
        self.assertEqual(res2, torch.tensor([2, 2, 2, 2], dtype=torch.float))

    @parametrize("device", ["cpu", "cuda"])
    def test_all_reduce_coalesced_eager(self, device):
        # 如果设备为 CUDA，则检查 CUDA 设备数量是否足够
        if device == "cuda":
            if torch.cuda.device_count() < self.world_size:
                self.skipTest("Not enough CUDA devices")
            # 设置当前进程的 CUDA 设备
            torch.cuda.set_device(dist.get_rank())

        # 创建两个全为 1 的张量 t0 和 t1
        t0 = torch.ones([4], device=device)
        t1 = torch.ones([6], device=device) + 2
        # 创建设备网格对象，并传入当前设备和一个包含 [0, 1, 2, 3] 的张量
        mesh = dt.DeviceMesh(device, torch.arange(4))

        # 对张量列表 [t0, t1] 在网格 mesh 上进行全局归约（求和），将结果保存在 res 中
        res = ft_c.all_reduce_coalesced([t0, t1], "sum", mesh)
        # 断言归约后的第一个张量与预期的 t0 * 4 相等
        self.assertEqual(res[0], t0 * 4)
        # 断言归约后的第二个张量与预期的 t1 * 4 相等
        self.assertEqual(res[1], t1 * 4)

    @parametrize("device", ["cpu", "cuda"])
    def test_all_gather_tensor(self, device):
        # 如果设备为 CUDA，则检查 CUDA 设备数量是否足够
        if device == "cuda":
            if torch.cuda.device_count() < self.world_size:
                self.skipTest("Not enough CUDA devices")
            # 设置当前进程的 CUDA 设备
            torch.cuda.set_device(dist.get_rank())

        # 创建一个形状为 [3, 3, 3] 的全为 1 的本地张量 local_tensor
        # 进行全局聚集操作，收集维度为 dim，使用网格 mesh 和组标识为 0
        local_tensor = torch.ones([3, 3, 3], device=device)
        gathered_tensor = ft_c.all_gather_tensor(
            local_tensor, gather_dim=dim, group=(mesh, 0)
        )
        # 断言聚集后的张量与预期的全为 1 的张量相等
        self.assertEqual(gathered_tensor, torch.ones(output_size))

    @parametrize("device", ["cpu", "cuda"])
    # 定义一个测试方法，用于测试将多个张量按照协调的方式进行全局聚合
    def test_all_gather_into_tensor_coalesced(self, device):
        # 如果设备是 CUDA
        if device == "cuda":
            # 如果 CUDA 设备数量小于世界大小，跳过测试
            if torch.cuda.device_count() < self.world_size:
                self.skipTest("Not enough CUDA devices")
            # 设置当前进程的 CUDA 设备
            torch.cuda.set_device(dist.get_rank())

        # 创建两个张量列表，每个张量形状为 [4]，位于指定设备上
        tensors = [torch.ones([4], device=device), torch.ones([4], device=device) + 1]
        # 创建设备网格对象，使用指定设备和范围为 [0, 1, 2, 3] 的索引
        mesh = dt.DeviceMesh(device, torch.arange(4))

        # 调用函数 ft_c.all_gather_into_tensor_coalesced 进行张量的全局聚合
        res = ft_c.all_gather_into_tensor_coalesced(tensors, mesh)
        # 断言结果列表 res 的长度为 2
        self.assertEqual(2, len(res))
        # 断言 res[0] 的值为形状为 [4 * dist.get_world_size()] 的全 1 张量，位于指定设备上
        self.assertEqual(torch.ones([4 * dist.get_world_size()], device=device), res[0])
        # 断言 res[1] 的值为形状为 [4 * dist.get_world_size()] 的全 1 张量加 1，位于指定设备上
        self.assertEqual(
            torch.ones([4 * dist.get_world_size()], device=device) + 1, res[1]
        )

    # 使用参数化装饰器定义测试方法，用于测试张量的分散归约操作
    @parametrize("device", ["cpu", "cuda"])
    def test_reduce_scatter_tensor(self, device):
        # 如果设备是 CUDA
        if device == "cuda":
            # 如果 CUDA 设备数量小于世界大小，跳过测试
            if torch.cuda.device_count() < self.world_size:
                self.skipTest("Not enough CUDA devices")
            # 设置当前进程的 CUDA 设备
            torch.cuda.set_device(dist.get_rank())

        # 测试 1 维和 2 维设备网格
        # 创建 1 维设备网格对象，使用指定设备和范围为 [0, 1, ..., self.world_size-1] 的索引
        mesh_1d = dt.DeviceMesh(device, torch.arange(self.world_size))
        # 创建 2 维设备网格对象，使用指定设备和形状为 [2, 2] 的索引
        mesh_2d = dt.DeviceMesh(device, torch.arange(self.world_size).view(2, 2))
        # 对每个设备网格进行循环测试
        for mesh in [mesh_1d, mesh_2d]:
            # 要进行分散的维度列表
            dims_to_scatter = [0, 1]
            # 对每个维度进行循环测试
            for dim in dims_to_scatter:
                # 群体大小为设备网格的大小
                group_size = mesh.size(0)
                # 输入张量的形状
                input_size = [3, 3]
                # 输出张量的形状
                output_size = [3, 3]
                output_size[dim] *= group_size
                # 创建全 1 张量作为输入张量，位于指定设备上
                input_tensor = torch.ones(output_size, device=device)
                # 预期的结果数值
                res_num = 1 * group_size
                # 调用 ft_c.reduce_scatter_tensor 执行张量的分散归约操作
                rs_tensor = ft_c.reduce_scatter_tensor(
                    input_tensor, "sum", scatter_dim=dim, group=(mesh, 0)
                )
                # 断言分散归约后的张量 rs_tensor 的值等于全 1 张量乘以 res_num
                self.assertEqual(rs_tensor, torch.ones(input_size) * res_num)

    # 使用参数化装饰器定义测试方法，测试将多个张量按照协调的方式进行分散归约操作
    @parametrize("device", ["cpu", "cuda"])
    def test_reduce_scatter_into_tensor_coalesced(self, device):
        # 如果设备是 CUDA
        if device == "cuda":
            # 如果 CUDA 设备数量小于世界大小，跳过测试
            if torch.cuda.device_count() < self.world_size:
                self.skipTest("Not enough CUDA devices")
            # 设置当前进程的 CUDA 设备
            torch.cuda.set_device(dist.get_rank())
        # 创建两个张量列表，每个张量形状为 [4]，数据类型为 int64，位于指定设备上
        tensors = [
            torch.ones([4], dtype=torch.int64, device=device),
            torch.ones([4], dtype=torch.int64, device=device) + 1,
        ]
        # 创建设备网格对象，使用指定设备和范围为 [0, 1, 2, 3] 的索引
        mesh = dt.DeviceMesh(device, torch.arange(4))

        # 调用函数 ft_c.reduce_scatter_tensor_coalesced 进行张量的分散归约操作
        res = ft_c.reduce_scatter_tensor_coalesced(tensors, "sum", [0, 0], mesh)
        # 断言结果列表 res 的长度为 2
        self.assertEqual(2, len(res))
        # 断言 res[0] 的值为形状为 [1] 的张量，内容为 4，位于指定设备上
        self.assertEqual(torch.tensor([4], device=device), res[0])
        # 断言 res[1] 的值为形状为 [1] 的张量，内容为 8，位于指定设备上
        self.assertEqual(torch.tensor([8], device=device), res[1])
class TestMetaCollectives(TestCase):
    # 定义测试类 TestMetaCollectives，继承自 TestCase
    def test_all_reduce(self):
        # 测试函数 test_all_reduce，测试数据在设备 "meta" 上生成的随机张量 x
        x = torch.rand((2, 3, 4), device="meta")
        # 调用 ft_c.all_reduce 函数，对 x 进行全局归约操作，使用 "sum" 运算，指定目标为 "0"
        out = ft_c.all_reduce(x, "sum", "0")
        # 断言 x 和 out 的大小相同
        self.assertEqual(x.size(), out.size())


class TestGradCollectives(MultiThreadedTestCase):
    @property
    def world_size(self):
        # 返回测试所需的进程数量为 2
        return 2

    def setUp(self):
        # 设置测试环境
        super().setUp()
        # 启动线程
        self._spawn_threads()

    def test_all_reduce(self):
        # 测试函数 test_all_reduce，生成两个具有梯度的随机张量 x 和 y
        x = torch.rand([4], requires_grad=True)
        y = torch.rand([4], requires_grad=True)
        # 调用 ft_c.all_reduce 函数，对 x 在 dist.group.WORLD 组中进行全局归约操作，结果存储在 out 中
        out = ft_c.all_reduce(x, "sum", dist.group.WORLD)
        # 对 (out + y) 的和进行求和并反向传播梯度
        (out + y).sum().backward()
        # 断言 x 的梯度为 None
        self.assertIsNone(x.grad)


class TestMakeFx(MultiThreadedTestCase):
    @property
    def world_size(self):
        # 返回测试所需的进程数量为 2
        return 2

    def setUp(self):
        # 设置测试环境
        super().setUp()
        # 启动线程
        self._spawn_threads()

    def tearDown(self):
        # 清理测试环境
        super().tearDown()

        # 处理线程竞争条件，将 _is_fx_tracing_flag 标志设置为 False
        torch.fx._symbolic_trace._is_fx_tracing_flag = False
        # 断言不处于 FX 追踪状态
        self.assertFalse(torch.fx._symbolic_trace.is_fx_tracing())

    def test_all_reduce_tracing(self):
        # 定义一个函数 allred，对输入进行全局归约操作并返回结果加 1
        def allred(input):
            return ft_c.all_reduce(input, "sum", group=dist.group.WORLD) + 1

        # 使用 make_fx 将 allred 函数转换为图形化表示的图形对象
        graph = make_fx(allred)(torch.rand(4))
        # 使用 FileCheck 检查图中是否包含 "all_reduce" 和 "wait_tensor"，并运行检查
        FileCheck().check("all_reduce").check("wait_tensor").run(str(graph.graph))

        # 创建一个设备网格 mesh，类型为 DeviceMesh，包含 0 和 1
        mesh = dt.DeviceMesh("cpu", torch.arange(self.world_size))

        # 定义一个函数 allred_mesh，对输入进行全局归约操作并返回结果加 1
        def allred_mesh(input):
            return ft_c.all_reduce(input, "sum", mesh) + 1

        # 使用 make_fx 将 allred_mesh 函数转换为图形化表示的图形对象
        mesh_graph = make_fx(allred_mesh)(torch.rand(4))
        # 使用 FileCheck 检查图中是否不包含 "get_attr"，并包含 "wait_tensor"，并运行检查
        FileCheck().check_not("get_attr").check("wait_tensor").run(
            str(mesh_graph.graph)
        )

        # 定义一个函数 allred_mesh_dim，对输入进行全局归约操作并返回结果加 1
        def allred_mesh_dim(input):
            return ft_c.all_reduce(input, "sum", (mesh, 0)) + 1

        # 使用 make_fx 将 allred_mesh_dim 函数转换为图形化表示的图形对象
        mesh_dim_graph = make_fx(allred_mesh_dim)(torch.rand(4))
        # 使用 FileCheck 检查图中是否不包含 "get_attr"，并包含 "wait_tensor"，并运行检查
        FileCheck().check_not("get_attr").check("wait_tensor").run(
            str(mesh_dim_graph.graph)
        )
    def setUp(self):
        # 调用父类的 setUp 方法
        super().setUp()
        # 设置环境变量 WORLD_SIZE 为当前对象的 world_size 属性值
        os.environ["WORLD_SIZE"] = str(self.world_size)
        # 设置环境变量 BACKEND 为 NCCL 分布式后端
        os.environ["BACKEND"] = dist.Backend.NCCL
        # 将局部变量 BACKEND 设置为 NCCL 分布式后端
        BACKEND = dist.Backend.NCCL
        # 调用私有方法 _spawn_processes()，启动进程
        self._spawn_processes()

    @property
    def device(self):
        # 返回当前进程的设备，使用 self.rank 作为设备索引
        return torch.device(self.rank)

    @property
    def world_size(self):
        # 返回全局常量 WORLD_SIZE
        return WORLD_SIZE

    @property
    def process_group(self):
        # 返回分布式通信组对象 dist.group.WORLD
        return dist.group.WORLD

    def dist_init(self):
        # 初始化分布式进程组
        dist.init_process_group(
            backend=BACKEND,
            world_size=self.world_size,
            rank=self.rank,
            init_method=f"file://{self.file_name}",
        )

        # 如果使用的是 nccl 后端，设置当前 CUDA 设备为 self.rank 指定的设备
        if BACKEND == "nccl":
            torch.cuda.set_device(self.rank)

    def destroy_comms(self):
        # 等待所有进程到达此处，然后开始关闭通信
        dist.barrier()
        # 销毁分布式进程组
        dist.destroy_process_group()

    @requires_nccl()
    @with_comms()
    def test_all_gather_into_tensor_coalesced(self):
        # 如果 GPU 数量少于 self.world_size，退出测试
        exit_if_lt_x_gpu(self.world_size)

        # 创建两个张量，每个张量在当前进程的 CUDA 设备上
        tensors = [
            torch.ones([4], device=f"cuda:{self.rank}"),
            torch.ones([4], device=f"cuda:{self.rank}") + 1,
        ]
        # 创建一个 DeviceMesh 对象，绑定到当前进程的 CUDA 设备
        mesh = dt.DeviceMesh(f"cuda:{self.rank}", torch.arange(self.world_size))

        # 调用 ft_c.all_gather_into_tensor_coalesced 方法进行全局收集
        res = ft_c.all_gather_into_tensor_coalesced(tensors, mesh)
        # 断言收集到的结果数量为 2
        self.assertEqual(2, len(res))
        # 断言第一个收集到的张量与预期的全为 1
        self.assertEqual(torch.ones([4 * dist.get_world_size()]), res[0])
        # 断言第二个收集到的张量与预期的全为 2
        self.assertEqual(torch.ones([4 * dist.get_world_size()]) + 1, res[1])

    @with_comms()
    def test_all_to_all_single(self):
        # 根据后端类型确定设备类型
        device = "cuda" if BACKEND == dist.Backend.NCCL else "cpu"
        # 创建一个 DeviceMesh 对象，绑定到当前进程的设备
        mesh = dt.DeviceMesh(device, torch.arange(self.world_size))
        # 获取当前进程的排名
        rank = dist.get_rank()

        # 计算输出张量的行数
        row = self.world_size * (rank + 1) * (self.world_size + 1) / 2
        # 创建输入张量 x，全为当前进程排名加一，设备类型为 device
        x = torch.ones(int(row), 5, device=device) * (rank + 1)
        # 计算输入和输出分块的大小
        split_sizes = [(i + 1) * (rank + 1) for i in range(self.world_size)]
        # 调用 ft_c.all_to_all_single 方法进行单个元素的全对全通信
        y = ft_c.all_to_all_single(
            x, output_split_sizes=split_sizes, input_split_sizes=split_sizes, group=mesh
        )
        # 创建预期的输出张量列表，用于断言比较
        expected = []
        for idx, tensor in enumerate(torch.split(x, split_sizes)):
            expected.append(torch.full_like(tensor, (idx + 1)))
        expected = torch.cat(expected)
        # 断言 y 与预期的输出张量相等
        self.assertEqual(y, expected)

    @with_comms()
    # 定义测试方法，测试在单一1D输入上的全对全通信
    def test_all_to_all_single_1d_input(self):
        # 根据不同的后端选择设备，如果后端是NCCL，则选择cuda，否则选择cpu
        device = "cuda" if BACKEND == dist.Backend.NCCL else "cpu"
        # 创建设备网格对象，使用设备和从0到world_size-1的张量
        mesh = dt.DeviceMesh(device, torch.arange(self.world_size))
        # 获取当前进程的排名
        rank = dist.get_rank()

        # 计算输出张量的行数，行数是一个等差数列求和的结果
        row = self.world_size * (rank + 1) * (self.world_size + 1) / 2
        # 创建一个值为(rank+1)的张量x，长度为row，使用指定的设备
        x = torch.ones(int(row), device=device) * (rank + 1)
        # 根据输入分割大小和输出分割大小调用all_to_all_single函数，进行全对全通信
        split_sizes = [(i + 1) * (rank + 1) for i in range(self.world_size)]
        y = ft_c.all_to_all_single(
            x, output_split_sizes=split_sizes, input_split_sizes=split_sizes, group=mesh
        )
        # 期望的结果是按照分割大小将x分块，每块填充为(idx+1)的张量，最后连接起来
        expected = []
        for idx, tensor in enumerate(torch.split(x, split_sizes)):
            expected.append(torch.full_like(tensor, (idx + 1)))
        expected = torch.cat(expected)
        # 断言y与期望的结果相等
        self.assertEqual(y, expected)

    # 标记该测试方法需要通信支持
    @with_comms()
    def test_all_to_all_single_split_sizes_none(self):
        # 根据不同的后端选择设备，如果后端是NCCL，则选择cuda，否则选择cpu
        device = "cuda" if BACKEND == dist.Backend.NCCL else "cpu"
        # 创建设备网格对象，使用设备和从0到world_size-1的张量
        mesh = dt.DeviceMesh(device, torch.arange(self.world_size))
        # 获取当前进程的排名
        rank = dist.get_rank()

        # 创建一个值为(rank+1)的self.world_size x self.world_size的张量x，使用指定的设备
        x = torch.ones(self.world_size, self.world_size, device=device) * (rank + 1)
        # 调用all_to_all_single函数，输入分割大小和输出分割大小均为None，进行全对全通信
        y = ft_c.all_to_all_single(
            x, output_split_sizes=None, input_split_sizes=None, group=mesh
        )
        # 期望的结果是将x按照self.world_size分块，每块填充为(idx+1)的张量，最后连接起来
        expected = []
        for idx, tensor in enumerate(torch.chunk(x, self.world_size)):
            expected.append(torch.full_like(tensor, (idx + 1)))
        expected = torch.cat(expected)
        # 断言y与期望的结果相等
        self.assertEqual(y, expected)

    # 标记该测试方法需要Triton支持，并且需要通信支持
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @requires_nccl()
    @with_comms()
    def test_tracing(self):
        # 定义一个简单的全局归约函数allreduce，使用ft_c.all_reduce进行求和操作
        def allreduce(t, pg):
            return ft_c.all_reduce(t, "sum", pg)

        # 使用torch.compile编译allreduce函数，完全使用图模式
        compiled_allreduce = torch.compile(allreduce, fullgraph=True)
        # 调用编译后的函数，对一个8x8的随机张量进行全归约操作
        compiled_allreduce(torch.randn(8, device=self.device), self.process_group)

    # 标记该测试方法需要Triton支持
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_tracing_with_fakepg(self):
        # 如果当前的world_size小于8，直接退出测试
        exit_if_lt_x_gpu(self.world_size)

        # 定义一个简单的全局归约函数allreduce，使用ft_c.all_reduce进行求和操作
        def allreduce(t, pg):
            return ft_c.all_reduce(t, "sum", pg)

        # 使用torch.compile编译allreduce函数，完全使用图模式
        compiled_allreduce = torch.compile(allreduce, fullgraph=True)
        # 初始化一个虚拟的进程组，使用fake后端，8个进程，FakeStore存储
        dist.init_process_group(
            backend="fake",
            rank=0,
            world_size=8,
            store=FakeStore(),
        )
        # 调用编译后的函数，对一个8维的随机张量进行全归约操作，使用dist.group.WORLD作为进程组
        allreduce(torch.randn(8, device=self.device), pg=dist.group.WORLD)
# 创建一个测试类，继承自TestCollectivesWithNCCL，用于测试在world_size为4时的NCCL集合操作
class TestNCCLCollectivesWithWorldSize4(TestCollectivesWithNCCL):
    
    # 定义world_size属性，返回值为4，指定测试中的世界大小
    @property
    def world_size(self):
        return 4

    # 使用requires_nccl和with_comms装饰器，确保测试依赖于NCCL和通信上下文
    @requires_nccl()
    @with_comms()
    # 定义测试函数test_permute_tensor_with_sub_group，测试张量在子组内的置换操作
    def test_permute_tensor_with_sub_group(self):
        # 如果当前GPU数量少于world_size，立即退出测试
        exit_if_lt_x_gpu(self.world_size)

        # 指定设备为cuda，定义网格维度名称列表为["dp", "tp"]
        device = "cuda"
        mesh_dim_names = ["dp", "tp"]

        # 使用init_device_mesh初始化一个二维设备网格mesh_2d，网格大小为(2, self.world_size // 2)，指定网格维度名称
        mesh_2d = dt.init_device_mesh(
            device, (2, self.world_size // 2), mesh_dim_names=mesh_dim_names
        )

        # 遍历网格维度名称列表
        for mesh_name in mesh_dim_names:
            # 获取当前维度网格mesh的本地rank
            mesh = mesh_2d[mesh_name]
            rank = mesh.get_local_rank()

            # 根据rank生成发送张量send_tensor，使用torch.arange创建浮点数序列，dtype为torch.float32，设备为device
            send_tensor = torch.arange(2, dtype=torch.float32, device=device) + 2 * rank
            
            # 调用ft_c.permute_tensor函数对send_tensor进行置换操作，group参数指定为当前mesh
            recvd_tensor = ft_c.permute_tensor(send_tensor, [1, 0], group=mesh)

            # 根据rank计算预期张量expected，使用torch.arange创建浮点数序列，dtype为torch.float32，设备为device
            expected = torch.arange(2, dtype=torch.float32, device=device) + 2 * (
                (rank - 1 + 2) % 2
            )

            # 使用self.assertEqual断言recvd_tensor与expected相等，如果不等则输出详细的错误消息
            self.assertEqual(
                recvd_tensor,
                expected,
                msg=f"Expected {expected} on {self.rank=} (local_rank={rank}), "
                f"but received {recvd_tensor} instead.",
            )


# 使用instantiate_parametrized_tests装饰器实例化参数化测试类TestFunctionalAutograd
@instantiate_parametrized_tests
class TestFunctionalAutograd(MultiThreadedTestCase):

    # 在每个测试函数运行前执行setUp方法，调用父类的setUp方法并开启多线程
    def setUp(self):
        super().setUp()
        self._spawn_threads()

    # 定义world_size属性，返回值为2，指定测试中的世界大小
    @property
    def world_size(self):
        return 2

    # 使用parametrize装饰器，指定compile参数为[True, False]，对test_all_to_all_single进行参数化测试
    @parametrize("compile", [True, False])
    # 定义测试函数test_all_to_all_single，测试all_to_all_single_autograd函数的自动求导功能
    def test_all_to_all_single(self, compile: bool = True) -> None:
        # 指定通信组名称为dist.group.WORLD.group_name
        group = dist.group.WORLD.group_name

        # 创建形状为(self.world_size, 2)的全一张量t，requires_grad为True
        t = torch.ones((self.world_size, 2), requires_grad=True)

        # 定义函数my_func，接受torch.Tensor t和int world_size为参数，返回torch.Tensor
        def my_func(t: torch.Tensor, world_size: int) -> torch.Tensor:
            # 创建长度为world_size的大小列表sizes，每个元素为1
            sizes = [1] * world_size
            # 张量t每个元素乘以2
            t = t * 2
            # 断言张量t需要求导
            assert t.requires_grad
            # 调用ft_c.all_to_all_single_autograd函数，对张量t进行全对全的通信操作，使用sizes作为参数，group为指定组
            out = ft_c.all_to_all_single_autograd(t, sizes, sizes, group)
            # 对输出张量out每个元素加0
            out = out + 0
            return out

        # 根据compile参数判断是否编译my_func函数，使用torch.compile进行编译，backend为"aot_eager"
        if compile:
            compiled = torch.compile(my_func, fullgraph=True, backend="aot_eager")
        else:
            compiled = my_func

        # 调用compiled函数，传入参数t和self.world_size，得到输出张量out
        out = compiled(t, self.world_size)

        # 使用self.assertEqual断言out的形状与t相同
        self.assertEqual(out.shape, t.shape)
        # 使用self.assertEqual断言out与全1张量torch.full_like(t, 2.0)相等
        self.assertEqual(out, torch.full_like(t, 2.0))
        # 断言out需要梯度
        self.assertIsNotNone(out.grad_fn)
        self.assertTrue(out.requires_grad)
        
        # 计算loss，为out所有元素之和
        loss = out.sum()
        # 对loss进行反向传播
        loss.backward()
        # 使用self.assertEqual断言t的梯度与全1张量torch.full_like(t, 2.0)相等
        self.assertEqual(t.grad, torch.full_like(t, 2.0))
    # 定义一个单元测试方法，用于测试所有到所有通信的单电感情况
    def test_all_to_all_single_inductor(self) -> None:
        # 获取世界分组的名称
        group = dist.group.WORLD.group_name

        # 创建一个形状为 (self.world_size, 2) 的随机张量，需要梯度计算
        t = torch.rand((self.world_size, 2), requires_grad=True)

        # 定义一个自定义函数 my_func，接受一个张量 t 和世界大小作为输入，并返回一个张量
        def my_func(t: torch.Tensor, world_size: int) -> torch.Tensor:
            # 创建大小为 world_size 的一维数组，每个元素值为 1
            sizes = [1] * world_size
            # 将输入张量 t 的值乘以 10
            t = t * 10
            # 断言张量 t 需要梯度计算
            assert t.requires_grad
            # 调用 ft_c.all_to_all_single_autograd 函数进行所有到所有的单电感自动微分通信
            out = ft_c.all_to_all_single_autograd(t, sizes, sizes, group)
            # 将输出张量加上 2
            out = out + 2
            # 返回加和后的结果张量
            return out.sum()

        # 使用 torch.compile 将 my_func 编译成一个函数图，完整编译图模式
        compiled = torch.compile(my_func, fullgraph=True)

        # 定义一个运行带反向传播的函数
        def run_with_backward():
            # 调用编译后的函数 compiled 运行输入张量 t 和世界大小 self.world_size
            out = compiled(t, self.world_size)
            # 对输出结果进行反向传播
            out.backward()

        # 运行并获取代码和结果
        res, codes = run_and_get_code(run_with_backward)
        # 对每个代码进行检查，确保 _c10d_functional.all_to_all_single.default 和 _c10d_functional.wait_tensor.default 每个都只出现一次
        for code in codes:
            FileCheck().check_count(
                "_c10d_functional.all_to_all_single.default", 1, exactly=True
            ).check_count("_c10d_functional.wait_tensor.default", 1, exactly=True).run(
                code
            )

        # 断言输入张量 t 的梯度不为 None
        self.assertIsNotNone(t.grad)

    # 参数化测试方法，用于测试所有聚集张量的情况，参数 compile 表示是否编译
    @parametrize("compile", [True, False])
    def test_all_gather_tensor(self, compile: bool) -> None:
        # 获取世界分组的名称
        group = dist.group.WORLD.group_name

        # 定义一个自定义函数 my_func，接受一个张量 t 和一个维度 dim 作为输入，并返回一个张量
        def my_func(t: torch.Tensor, dim: int) -> torch.Tensor:
            # 断言张量 t 需要梯度计算
            assert t.requires_grad
            # 调用 ft_c.all_gather_tensor_autograd 函数进行聚集张量的自动微分通信
            out = ft_c.all_gather_tensor_autograd(
                t * 1.0,
                gather_dim=dim,
                group=group,
            )
            # 将输出张量乘以 1.0
            out = out * 1.0
            # 返回乘积后的结果张量
            return out

        # 根据参数 compile 决定是否进行编译
        if compile:
            compiled = torch.compile(my_func, fullgraph=True, backend="aot_eager")
        else:
            compiled = my_func

        # 定义需要聚集的维度列表
        dims_to_gather = [0, 1, 2]
        for dim in dims_to_gather:
            # 根据维度 dim 计算输出张量的大小
            output_size = [3, 3, 3]
            output_size[dim] *= self.world_size
            # 创建一个形状为 [3, 3, 3] 的全为 1 的本地张量，需要梯度计算
            local_tensor = torch.ones([3, 3, 3], requires_grad=True)
            # 调用编译后的函数 compiled 进行聚集张量操作
            gathered_tensor = compiled(local_tensor, dim)
            # 断言聚集后的张量与预期的全为 1 的输出张量相等
            self.assertEqual(gathered_tensor, torch.ones(output_size))

            # 对聚集后的张量进行求和并进行反向传播
            gathered_tensor.sum().backward()
            # 断言本地张量的梯度与全为 self.world_size 的张量相等
            self.assertEqual(
                local_tensor.grad,
                torch.full((3, 3, 3), fill_value=float(self.world_size)),
            )
    # 定义一个测试方法，用于测试 reduce_scatter_tensor 函数的行为
    def test_reduce_scatter_tensor(self, compile: bool) -> None:
        # 获取全局通信组的名称
        group = dist.group.WORLD.group_name

        # 定义一个函数 my_func，接受一个张量 t 和一个整数维度 dim，并返回 reduce_scatter_tensor 的结果张量
        def my_func(t: torch.Tensor, dim: int) -> torch.Tensor:
            # 断言输入张量 t 需要梯度计算
            assert t.requires_grad
            # 调用 ft_c.reduce_scatter_tensor_autograd 函数进行张量的 reduce scatter 操作，并乘以 1.0
            rs_tensor = (
                ft_c.reduce_scatter_tensor_autograd(
                    input_tensor * 1.0, "sum", scatter_dim=dim, group=group
                )
                * 1.0
            )
            return rs_tensor

        # 根据编译标志 compile，决定是否编译函数 my_func
        if compile:
            # 使用 torch.compile 编译 my_func 函数，使用完整图模式，后端为 "aot_eager"
            compiled = torch.compile(my_func, fullgraph=True, backend="aot_eager")
        else:
            # 否则，直接使用未编译的 my_func
            compiled = my_func

        # 需要进行 reduce scatter 操作的维度列表
        dims_to_scatter = [0, 1]
        for dim in dims_to_scatter:
            # 获取通信组的大小，通常是指进程数量
            group_size = self.world_size
            # 创建一个全为 1 的张量，形状为 output_size，需要梯度计算
            input_size = [3, 3]
            output_size = [3, 3]
            output_size[dim] *= group_size
            input_tensor = torch.ones(output_size, requires_grad=True)
            # 调用 compiled 函数进行 reduce scatter 操作
            rs_tensor = compiled(input_tensor, dim)
            # 预期的结果数值为 1 乘以进程数量
            res_num = 1 * group_size
            # 使用断言验证 reduce scatter 的结果是否正确
            self.assertEqual(rs_tensor, torch.ones(input_size) * res_num)
            # 对 rs_tensor 的和进行反向传播
            rs_tensor.sum().backward()
            # 使用断言验证输入张量的梯度是否正确计算
            self.assertEqual(input_tensor.grad, torch.full(output_size, fill_value=1.0))
# 继承自 MultiProcessTestCase 的测试类，用于测试基于 NCCL 的功能和自动求导
class TestFunctionalAutogradWithNCCL(MultiProcessTestCase):

    # 在每个测试方法运行前执行的设置方法
    def setUp(self):
        super().setUp()
        # 设置环境变量 WORLD_SIZE 为进程数目
        os.environ["WORLD_SIZE"] = str(self.world_size)
        # 设置环境变量 BACKEND 为 NCCL
        os.environ["BACKEND"] = dist.Backend.NCCL
        # 启动多进程
        self._spawn_processes()

    # 返回当前设备的属性
    @property
    def device(self):
        return torch.device(self.rank)

    # 返回世界大小的属性
    @property
    def world_size(self):
        return 2

    # 返回进程组的属性
    @property
    def process_group(self):
        return dist.group.WORLD

    # 初始化分布式进程组
    def dist_init(self):
        dist.init_process_group(
            backend=BACKEND,  # 使用预设的后端进行初始化
            world_size=self.world_size,  # 设置进程组的大小
            rank=self.rank,  # 设置当前进程的排名
            init_method=f"file://{self.file_name}",  # 使用文件初始化方法
        )

        # 为 NCCL 进程组设置当前设备
        if BACKEND == "nccl":
            torch.cuda.set_device(self.rank)

    # 销毁通信相关的组件
    def destroy_comms(self):
        # 等待所有进程到达此处再开始关闭
        dist.barrier()
        dist.destroy_process_group()

    # 装饰器，指示需要使用 NCCL
    @requires_nccl()
    # 装饰器，指示需要进行通信
    @with_comms()
    # 测试方法：测试 all_to_all_single_autograd 方法
    def test_all_to_all_single(self) -> None:
        group = self.process_group.group_name  # 获取进程组名称

        # 创建一个在当前设备上需要梯度的全 1 张量
        t = torch.ones((self.world_size, 2), requires_grad=True, device=self.device)

        sizes = [1] * self.world_size  # 设置通信大小
        assert t.requires_grad  # 断言张量需要梯度
        # 执行 all_to_all_single_autograd 方法，得到输出张量 out
        out = ft_c.all_to_all_single_autograd(t * 2, sizes, sizes, group) + 0

        # 断言输出张量的形状与 t 相同
        self.assertEqual(out.shape, t.shape)
        # 断言输出张量的值全为 2.0
        self.assertEqual(out, torch.full_like(t, 2.0))
        # 断言输出张量的梯度函数不为空
        self.assertIsNotNone(out.grad_fn)
        # 断言输出张量需要梯度
        self.assertTrue(out.requires_grad)
        # 计算损失，对损失进行反向传播
        loss = out.sum()
        loss.backward()
        # 断言输入张量 t 的梯度与全为 2.0 的张量相同
        self.assertEqual(t.grad, torch.full_like(t, 2.0))


if __name__ == "__main__":
    run_tests()
```