# `.\pytorch\test\distributed\_tensor\test_pointwise_ops.py`

```
# 导入必要的模块和类
from typing import Any, Callable, Dict, Optional, Sequence
from unittest import skip

# 导入 PyTorch 相关模块
import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch.distributed._tensor import DeviceMesh, distribute_tensor, DTensor
from torch.distributed._tensor.placement_types import Partial, Placement, Replicate, Shard
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorOpTestBase, skip_unless_torch_gpu

# 定义一个空操作函数
def no_op():
    return None

def deepcopy_convert_to_dtensor(
    val: Any,
    device_mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> Any:
    """
    递归地将 Tensors 转换为 DTensors（对于 Sequence 和 Dict 类型）。

    :param device_mesh: 使用的 DeviceMesh。
    :param placements: 使用的 Placement 列表。
    :return: 转换后的结构。
    """

    def f(x):
        if isinstance(x, Tensor) and not isinstance(x, DTensor):
            return distribute_tensor(
                x,
                device_mesh=device_mesh,
                placements=placements,
            )
        return x

    return pytree.tree_map(f, [val])[0]

def deepcopy_convert_from_dtensor(val: Any) -> Any:
    """
    递归地将任何 DTensor 转换为本地 Tensor。

    :param val: 要转换的结构。
    :return: 转换后的结构。
    """

    def f(x):
        if isinstance(x, DTensor):
            return x.full_tensor()
        return x

    return pytree.tree_map(f, [val])[0]

class DistElementwiseOpsTest(DTensorOpTestBase):
    def _compare_pairwise_ops(
        self,
        *,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement],
        op: Callable,
        pre_op_fn: Optional[Callable] = None,
        args: Sequence[Any] = tuple(),
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        比较两个操作在分布式环境中的执行结果。

        :param device_mesh: 使用的 DeviceMesh。
        :param placements: 使用的 Placement 列表。
        :param op: 进行比较的操作。
        :param pre_op_fn: 在执行操作前执行的函数。
        :param args: 操作的位置参数。
        :param kwargs: 操作的关键字参数。
        """

        if pre_op_fn is None:
            pre_op_fn = no_op

        if not kwargs:
            kwargs = {}

        # 将位置参数和关键字参数转换为 DTensor
        dargs = deepcopy_convert_to_dtensor(
            args,
            device_mesh=device_mesh,
            placements=placements,
        )
        dkwargs = deepcopy_convert_to_dtensor(
            kwargs,
            device_mesh=device_mesh,
            placements=placements,
        )

        pre_op_fn()

        # 首先运行参考实现，以确保调用正确；
        # 最好在此处调试不正确的调用。
        reference_result = op(*args, **kwargs)

        pre_op_fn()

        # 在分布式环境中运行操作
        dist_result = op(*dargs, **dkwargs)

        # 将分布式结果转换回本地 Tensor
        collected_result = deepcopy_convert_from_dtensor(dist_result)

        # 在相同的 rank 上进行断言比较
        self.assertEqualOnRank(reference_result, collected_result)

    # TODO: 未来需要为 CPU 上的操作添加测试。
    # 定义一个私有方法，用于在分片设备网格上运行逐元素操作
    def _run_sharded_elementwise_ops(
        self,
        *,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement],
        pre_op_fn: Optional[Callable] = None,
        input_size: Sequence[int],
        op: Callable,
        **kwargs,
    ):
        # 如果没有提供预处理函数，则默认使用空操作函数 no_op
        if pre_op_fn is None:
            pre_op_fn = no_op

        # 创建一个随机张量作为输入数据
        input_tensor = torch.randn(
            *input_size,
            device=self.device_type,
            requires_grad=True,
        )

        # 调用 _compare_pairwise_ops 方法，比较逐对操作的结果
        self._compare_pairwise_ops(
            device_mesh=device_mesh,
            placements=placements,
            pre_op_fn=pre_op_fn,
            op=op,
            args=(input_tensor,),
            kwargs=kwargs,
        )

    # 测试 DTensor 类的部分加法操作
    def test_partial_add(self):
        # 构建设备网格
        device_mesh = self.build_device_mesh()
        # 创建包含部分放置的 DTensor 对象 d_1 和 d_2，并执行加法操作
        d_1 = DTensor.from_local(torch.rand(2, 2), device_mesh, [Partial()])
        d_2 = DTensor.from_local(torch.rand(2, 2), device_mesh, [Partial()])
        d_3 = d_1 + d_2
        # 断言结果张量的第一个放置位置是部分放置
        self.assertTrue(d_3._spec.placements[0].is_partial())

    # 测试 DTensor 类的部分乘法操作
    def test_partial_mul(self):
        # 构建设备网格
        device_mesh = self.build_device_mesh()
        # 创建包含部分放置的 DTensor 对象 d_1 和 d_2，并执行乘法操作
        d_1 = DTensor.from_local(torch.ones(2, 2), device_mesh, [Partial()])
        d_2 = DTensor.from_local(torch.ones(2, 2), device_mesh, [Partial()])
        d_3 = d_1 * d_2
        # 断言结果张量的规范是复制
        self.assertTrue(d_3._spec.placements[0].is_replicate())
        # 断言结果张量的本地值等于全局大小的平方乘以1的张量
        self.assertEqual(d_3.to_local(), torch.ones(2, 2) * (self.world_size**2))

    # 测试激活函数在分片设备网格上的执行
    def test_activations(self):
        # 构建设备网格
        device_mesh = self.build_device_mesh()
        # 在第一个分片上运行 torch.nn.functional.gelu 激活函数
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Shard(0)],
            input_size=(8, 5),
            op=torch.nn.functional.gelu,
        )
        # 在复制的设备网格上运行 torch.nn.functional.gelu 激活函数
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Replicate()],
            input_size=(8, 5),
            op=torch.nn.functional.gelu,
        )
        # 在第二个分片上运行 torch.nn.functional.relu 激活函数
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Shard(1)],
            input_size=(3, 12),
            op=torch.nn.functional.relu,
        )
        # 在复制的设备网格上运行 torch.nn.functional.relu 激活函数
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Replicate()],
            input_size=(8, 5),
            op=torch.nn.functional.relu,
        )
        # 在第一个分片上运行 torch.sigmoid 激活函数
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Shard(0)],
            input_size=(8, 5),
            op=torch.sigmoid,
        )
        # 在复制的设备网格上运行 torch.sigmoid 激活函数
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Replicate()],
            input_size=(8, 5),
            op=torch.sigmoid,
        )

    # 标记此测试用例为跳过状态，因为基于随机数生成的操作存在问题
    @skip("testing RNG based ops is broken: https://github.com/pytorch/tau/issues/494")
    # 定义一个测试函数，用于测试 dropout 操作
    def test_dropout(self):
        # 构建设备网格
        device_mesh = self.build_device_mesh()

        # 定义内部函数 _reset_random_seed，用于重置随机种子
        def _reset_random_seed():
            torch.manual_seed(self.rank + 4)

        # 运行分片元素操作，对应于设备网格和特定分片，使用 torch.nn.functional.dropout 函数
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Shard(0)],
            input_size=(8, 5),
            op=torch.nn.functional.dropout,
            pre_op_fn=_reset_random_seed,
            p=0.4,
            training=False,
        )

        # 再次运行分片元素操作，对应于设备网格和不同的分片，使用 torch.nn.functional.dropout 函数
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Shard(1)],
            input_size=(3, 14),
            op=torch.nn.functional.dropout,
            pre_op_fn=_reset_random_seed,
            p=0.5,
            training=True,
        )

    # 标记为需要在 Torch GPU 环境下运行的测试函数
    @skip_unless_torch_gpu
    def test_dropout_backward(self):
        # 构建设备网格
        device_mesh = self.build_device_mesh()

        # 定义分片列表
        placements = [Shard(0)]

        # 定义输入张量的尺寸
        input_size = (8, 5)

        # 创建一个随机梯度输出张量，需要计算梯度
        grad_output = torch.rand(
            input_size,
            device=self.device_type,
            requires_grad=True,
        )

        # 创建一个掩码张量，用于控制 dropout 操作的保留概率
        mask = (
            torch.rand(
                input_size,
                device=self.device_type,
                requires_grad=False,
            )
            < 0.8
        )

        # 比较一对操作的结果，对应于设备网格和指定分片，使用 torch.ops.aten.native_dropout_backward 函数
        self._compare_pairwise_ops(
            device_mesh=device_mesh,
            placements=placements,
            op=torch.ops.aten.native_dropout_backward,
            kwargs=dict(
                grad_output=grad_output,
                mask=mask,
                scale=0.3,
            ),
        )

    # 测试 dropout 操作中可能出现的错误情况
    def test_dropout_errors(self):
        # 构建设备网格
        device_mesh = self.build_device_mesh()

        # 使用断言检查是否会抛出 RuntimeError，并包含特定的错误信息
        with self.assertRaisesRegex(RuntimeError, "supported"):
            self._run_sharded_elementwise_ops(
                device_mesh=device_mesh,
                placements=[Partial("sum")],
                input_size=(8, 5),
                op=torch.nn.functional.dropout,
            )

    # 测试张量乘法操作的正确性
    def test_mul_out(self):
        # 构建设备网格
        device_mesh = self.build_device_mesh()

        # 根据当前进程的排名设置随机种子
        torch.manual_seed(self.rank)

        # 定义分片规范
        shard_spec = [Shard(0)]

        # 定义输入张量的尺寸
        input_size = (8, 4)

        # 创建输入张量并转换为分布式张量
        input_tensor = torch.randn(*input_size, device=self.device_type)
        dtensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)

        # 创建另一个输入张量并转换为分布式张量
        other_tensor = torch.randn(*input_size, device=self.device_type)
        other_dtensor = DTensor.from_local(other_tensor, device_mesh, shard_spec)

        # 创建输出张量并转换为分布式张量
        output_tensor = torch.randn(*input_size, device=self.device_type)
        output_dtensor = DTensor.from_local(output_tensor, device_mesh, shard_spec)

        # 在分布式张量上执行乘法操作，结果输出到指定的输出张量
        dt = torch.mul(dtensor, other_dtensor, out=output_dtensor)

        # 计算预期的输出张量乘法结果
        expected = torch.mul(input_tensor, other_tensor, out=output_tensor)

        # 使用断言检查分布式张量的本地值是否与预期的输入张量匹配
        self.assertEqual(input_tensor, dtensor.to_local())

        # 使用断言检查分布式张量的本地值是否与预期的输出张量匹配
        self.assertEqual(expected, dt.to_local())
# 如果当前脚本作为主程序运行（而不是被导入），则执行以下代码
if __name__ == "__main__":
    # 调用函数 run_tests()，用于执行测试代码或者测试套件
    run_tests()
```