# `.\pytorch\test\distributed\_tensor\test_embedding_ops.py`

```
# Owner(s): ["oncall: distributed"]

# 导入必要的系统库
import sys

# 导入 PyTorch 库
import torch
from torch.distributed._tensor import (
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.distributed._tensor.debug import CommDebugMode
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

# 如果处于开发调试模式下的 ASAN 测试，输出警告并退出程序
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 获取 C10D 模块的函数接口
funcol = torch.ops.c10d_functional

# 测试用例类，继承自 DTensorTestBase
class TestEmbeddingOp(DTensorTestBase):
    # 应用分片操作的函数
    def _apply_sharding(self, embedding_mod, shard_dim, device_mesh):
        # 分片嵌入函数，为模块中的参数应用分布式张量
        def shard_embedding_fn(name, module, device_mesh):
            # 遍历模块中的参数并应用分布式张量
            for name, param in module.named_parameters():
                dist_param = torch.nn.Parameter(
                    distribute_tensor(param, device_mesh, [Shard(shard_dim)])
                )
                module.register_parameter(name, dist_param)

        # 使用 distribute_module 函数对嵌入模块进行分片
        sharded_embedding = distribute_module(
            embedding_mod, device_mesh, shard_embedding_fn
        )
        return sharded_embedding

    # 运行嵌入操作的测试函数
    def _run_embedding_op_test(
        self,
        device_mesh,
        shard_dim,
        input_size,
        num_embeddings,
        embedding_dim,
        **kwargs,
    ):
        # 使用相同的种子生成随机数，确保结果可重复性
        torch.manual_seed(0)
        # 创建本地嵌入层，用于单个节点的计算
        local_embedding = torch.nn.Embedding(
            num_embeddings,
            embedding_dim,
            device=self.device_type,
            **kwargs,
        )
        # 创建分片嵌入层，用于分布式计算
        sharded_embedding = torch.nn.Embedding(
            num_embeddings,
            embedding_dim,
            device=self.device_type,
            **kwargs,
        )

        # 将本地嵌入层的参数克隆并分片到分片嵌入层
        sharded_embedding.weight = torch.nn.Parameter(
            local_embedding.weight.clone().detach()
        )

        # 对分片嵌入层应用分片策略和设备网格
        sharded_embedding = self._apply_sharding(
            sharded_embedding, shard_dim, device_mesh
        )

        # 设置新的随机种子，以确保再次生成的随机数与之前不同
        torch.manual_seed(10)
        # 生成输入张量，用于计算
        inp = torch.randint(
            0, num_embeddings, tuple(input_size), device=self.device_type
        )
        # 生成目标张量，用于计算损失
        target = torch.empty(
            *inp.size(), embedding_dim, dtype=torch.float, device=self.device_type
        ).random_(0, 1)
        # 将输入张量分发到设备网格上
        dist_inp = distribute_tensor(inp, device_mesh, [Replicate()])

        # 在计算前向传播时启用通信调试模式，确保没有通信发生
        with CommDebugMode() as fwd_mode:
            dist_output = sharded_embedding(dist_inp)
            self.assertEqual(fwd_mode.get_total_counts(), 0)

        # 获取完整的输出张量
        output = dist_output.full_tensor()

        # 在本地节点上运行计算
        local_output = local_embedding(inp)

        # 验证本地输出与分布式输出是否一致
        self.assertEqual(local_output, output)

        # 使用交叉熵损失函数验证反向传播和梯度计算
        loss = torch.nn.CrossEntropyLoss()
        emb_loss = loss(
            output,
            target,
        )
        emb_dup_loss = loss(
            local_output,
            target,
        )

        # 对本地嵌入层进行反向传播
        emb_dup_loss.backward()

        # 在计算分片嵌入层的反向传播时启用通信调试模式，确保没有通信发生
        with CommDebugMode() as bwd_mode:
            emb_loss.backward()
            self.assertEqual(bwd_mode.get_total_counts(), 0)

        # 获取分片嵌入层的梯度张量
        gradient = sharded_embedding.weight.grad.full_tensor()

        # 获取本地嵌入层的梯度张量
        local_grad = local_embedding.weight.grad

        # 验证梯度是否一致
        self.assertEqual(gradient, local_grad)

        # 使用 torch.nn.functional.embedding 版本进行验证
        local_output = torch.nn.functional.embedding(
            inp,
            local_embedding.weight,
            **kwargs,
        )
        sharded_output = torch.nn.functional.embedding(
            DTensor.from_local(inp, device_mesh, [Replicate()], run_check=False),
            sharded_embedding.weight,
            **kwargs,
        )
        # 验证本地输出与分布式输出是否一致
        self.assertEqual(local_output, sharded_output.full_tensor())

    @with_comms
    def test_sharded_embedding_colwise(self):
        # 构建设备网格
        mesh = self.build_device_mesh()
        # 运行嵌入操作的测试，列方向分片
        self._run_embedding_op_test(mesh, 1, [5, 4], 17, 12)
        self._run_embedding_op_test(mesh, 1, [6, 7, 6], 21, 11)
        self._run_embedding_op_test(mesh, 1, [8, 6, 5, 4], 23, 13)
        self._run_embedding_op_test(mesh, 1, [8, 6, 5, 4, 7], 23, 16)
        self._run_embedding_op_test(mesh, 1, [4], 15, 14)
        self._run_embedding_op_test(mesh, 1, [34], 15, 14, padding_idx=10)
        self._run_embedding_op_test(mesh, 1, [8, 6, 5, 4], 23, 13, padding_idx=12)

    @with_comms
    def test_sharded_embedding_colwise_max_norm_errors(self):
        # 构建设备网格
        mesh = self.build_device_mesh()
        # 断言捕获 NotImplementedError 异常，检查是否注册了分片策略
        with self.assertRaisesRegex(
            NotImplementedError,
            "aten.embedding_renorm_.default does not have a sharding strategy registered.",
        ):
            # 运行嵌入操作的测试，列方向分片，带有最大范数限制异常测试
            self._run_embedding_op_test(
                mesh, 1, [8, 6, 5, 4], 23, 13, padding_idx=12, max_norm=2.0
            )

    @with_comms
    def test_sharded_embedding_rowwise(self):
        # 构建设备网格
        mesh = self.build_device_mesh()
        # 测试正确性
        self._run_embedding_op_test(mesh, 0, [5, 12], 16, 22)
        self._run_embedding_op_test(mesh, 0, [6, 7, 6], 13, 22)
        self._run_embedding_op_test(mesh, 0, [34], 15, 14, padding_idx=10)

        from torch.distributed._tensor.ops.embedding_ops import _MaskPartial

        # 测试集合操作
        embedding_mod = torch.nn.Embedding(10, 20, device=self.device_type)
        # 应用分片到嵌入模块上，行方向分片，使用设备网格
        sharded_embedding = self._apply_sharding(embedding_mod, 0, mesh)
        inp = torch.randint(0, 10, (8, 8), device=self.device_type)
        # 从本地张量创建分布式张量
        replicated_inp = DTensor.from_local(inp, mesh, [Replicate()], run_check=False)
        # 执行分片嵌入操作
        output = sharded_embedding(replicated_inp)
        # 断言输出的第一个位置是 _MaskPartial 类型的实例
        self.assertIsInstance(output.placements[0], _MaskPartial)

        comm_mode = CommDebugMode()

        # 使用通信调试模式
        with comm_mode:
            # 获取完整张量
            output.full_tensor()
            # 断言通信总数为 1
            self.assertEqual(comm_mode.get_total_counts(), 1)
            # 断言函数通信计数为 1
            self.assertEqual(comm_mode.get_comm_counts()[funcol.all_reduce], 1)
# 如果当前脚本作为主程序执行（而不是作为模块被导入），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```