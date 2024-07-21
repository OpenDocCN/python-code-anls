# `.\pytorch\test\distributed\_tensor\test_view_ops.py`

```
# 导入所需模块和库
import itertools  # 提供迭代工具
from typing import cast, List  # 类型提示，用于类型检查和转换

import torch  # PyTorch核心库
import torch.distributed as dist  # 分布式通信模块
from torch import rand, randn, Tensor  # 张量生成函数
from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate, Shard  # 分布式张量相关类
from torch.distributed._tensor.debug import CommDebugMode  # 通信调试模式
from torch.distributed._tensor.ops.view_ops import (  # 张量视图操作函数
    Broadcast,
    dim_maps,
    Flatten,
    InputDim,
    Repeat,
    Singleton,
    Split,
    view_groups,
)
from torch.distributed._tensor.placement_types import Placement  # 张量放置类型
from torch.testing._internal.common_utils import run_tests  # 测试工具函数
from torch.testing._internal.distributed._tensor.common_dtensor import (  # 分布式张量测试基类
    DTensorTestBase,
    with_comms,
)
from torch.utils import _pytree as pytree  # PyTree相关工具函数


class TestViewOps(DTensorTestBase):  # 测试张量视图操作的测试类继承自DTensorTestBase

    @property
    def world_size(self) -> int:  # 返回世界大小（设定为6）
        return 6

    def call_dt_test(self, op, args, kwargs, device_mesh: DeviceMesh):  # 调用分布式张量测试函数
        dim_map = dim_maps[op]  # 获取操作对应的维度映射
        rules = dim_map(*args, **kwargs)  # 计算操作的规则
        outputs = op(*args, **kwargs)  # 执行操作得到输出张量
        flat_args = pytree.arg_tree_leaves(*args)  # 获取参数的扁平化列表
        in_shape = flat_args[0].shape  # 获取输入张量的形状

        no_shard_dims = set()  # 初始化无分片维度集合
        for rule in rules:  # 遍历规则列表
            if isinstance(rule, Repeat):  # 如果规则是Repeat类型
                if isinstance(rule.input_dim, InputDim):  # 如果重复规则的输入维度是InputDim类型
                    no_shard_dims.add(rule.input_dim.input_dim)  # 将输入维度加入无分片维度集合
            elif isinstance(rule, Flatten):  # 如果规则是Flatten类型
                for dim in rule.input_dims[1:]:  # 遍历除第一个维度外的所有输入维度
                    if isinstance(dim, InputDim):  # 如果维度是InputDim类型
                        no_shard_dims.add(dim.input_dim)  # 将维度加入无分片维度集合
            elif isinstance(rule, Split):  # 如果规则是Split类型
                if isinstance(rule.input_dim, Flatten):  # 如果分割规则的输入维度是Flatten类型
                    for dim in rule.input_dim.input_dims[1:]:  # 遍历除第一个维度外的所有输入维度
                        if isinstance(dim, InputDim):  # 如果维度是InputDim类型
                            no_shard_dims.add(dim.input_dim)  # 将维度加入无分片维度集合

        if op == torch.unbind:  # 如果操作是torch.unbind
            no_shard_dims.add(kwargs.get("dim", 0))  # 将指定维度加入无分片维度集合

        # 确定每个维度的分片选项
        sharding_choices = cast(List[Placement], [Replicate()]) + [
            Shard(i) for i, s in enumerate(in_shape) if s > 1 and i not in no_shard_dims
        ]

        # 生成所有可能的分片组合
        all_sharding_choices = itertools.product(
            *(device_mesh.ndim * [sharding_choices])
        )

        for in_shard in all_sharding_choices:  # 遍历所有分片组合
            in_dt = distribute_tensor(args[0], device_mesh, in_shard)  # 分布输入张量到指定设备

            comm_mode = CommDebugMode()  # 创建通信调试模式对象
            with comm_mode:  # 使用通信调试模式
                out_dt = op(in_dt, *args[1:], **kwargs)  # 执行操作得到输出张量

            self.assertEqual(
                comm_mode.get_total_counts(), 0, "Expected no redistribution."
            )  # 断言通信调试模式的总计数为0，即不期望重新分发

            full_out = out_dt.full_tensor()  # 获取完整的输出张量

            if dist.get_rank() == 0:  # 如果当前进程的排名为0
                self.assertEqual(outputs, full_out)  # 断言期望输出等于完整输出张量

    def dimmap_test(self, op, args, expected_rule_output):  # 维度映射测试函数
        rules = dim_maps[op](*args)  # 获取操作的维度映射规则
        self.assertEqual(rules, expected_rule_output)  # 断言规则等于期望的规则输出
        self.call_dt_test(op, args, {}, self.device_mesh)  # 调用分布式张量测试函数，传入操作和参数

    @with_comms  # 使用通信装饰器
    # 定义测试函数 test_complex_view_ops，用于测试复杂视图操作
    @with_comms
    def test_complex_view_ops(self):
        # 设置设备网格为 DeviceMesh 类的实例，包含设备类型和全局设备索引
        self.device_mesh = DeviceMesh(
            self.device_type, torch.arange(dist.get_world_size()).view(-1, 2)
        )
        # 创建输入张量 inp，形状为 (24, 13, 2)，值为随机生成的标准正态分布数
        inp = randn(24, 13, 2)
        # 将 inp 转换为复数形式，intermediate 是中间结果张量
        intermediate = torch.view_as_complex(inp)
        # 将 intermediate 张量转换回实数形式，out 是最终结果张量
        out = torch.view_as_real(intermediate)

        # 测试 dim_map 函数的正确性，对 torch.view_as_complex 的预期规则
        expected_view_as_complex_rule = (
            InputDim(0),
            Flatten((InputDim(1), InputDim(2))),
        )
        # 获取 torch.view_as_complex 在 inp 上的实际规则
        view_as_complex_rule = dim_maps[torch.view_as_complex](inp)
        # 断言实际规则与预期规则相等
        self.assertEqual(view_as_complex_rule, expected_view_as_complex_rule)

        # 测试 dim_map 函数的正确性，对 torch.view_as_real 的预期规则
        expected_view_as_real_rule = (
            InputDim(0),
            Split(InputDim(1), (13, 2), 0),
            Split(InputDim(1), (13, 2), 1),
        )
        # 获取 torch.view_as_real 在 intermediate 上的实际规则
        view_as_real_rule = dim_maps[torch.view_as_real](intermediate)
        # 断言实际规则与预期规则相等
        self.assertEqual(view_as_real_rule, expected_view_as_real_rule)

        # 测试分片计算的正确性
        # 注意: 对于输入到 torch.view_as_complex 的 inp，最后两个维度上的分片操作是不支持的。
        sharding_choices: List[Placement] = [Replicate(), Shard(0)]
        # 生成所有可能的分片选择组合
        all_sharding_choices = itertools.product(
            *(self.device_mesh.ndim * [sharding_choices])
        )

        # 遍历所有分片选择组合
        for inp_shard in all_sharding_choices:
            # 使用 distribute_tensor 函数将 inp 分布到设备网格上
            inp_dt = distribute_tensor(inp, self.device_mesh, inp_shard)

            # 使用 CommDebugMode 进行通信模式测试
            comm_mode = CommDebugMode()
            with comm_mode:
                # 将分布后的 inp_dt 转换为复数形式
                intermediate_dt = torch.view_as_complex(inp_dt)
                # 将 intermediate_dt 转换为实数形式
                out_dt = torch.view_as_real(intermediate_dt)

            # 断言通信模式的总计数为 0，即预期没有重新分布操作
            self.assertEqual(
                comm_mode.get_total_counts(), 0, "Expected no redistribution."
            )
            # 断言 out 与 out_dt 的全张量值相等
            self.assertEqual(out, out_dt.full_tensor())
# 如果当前脚本被作为主程序执行（而不是被导入到其他模块中）
if __name__ == "__main__":
    # 执行名为 run_tests() 的函数，用于执行程序的测试
    run_tests()
```