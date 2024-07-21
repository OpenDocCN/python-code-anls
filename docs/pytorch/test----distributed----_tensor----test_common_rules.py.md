# `.\pytorch\test\distributed\_tensor\test_common_rules.py`

```py
# 版权声明及所有权信息
# 所有者：["oncall: distributed"]

# 导入PyTorch库
import torch
# 导入DeviceMesh类，用于处理分布式张量的设备网格
from torch.distributed._tensor import DeviceMesh
# 导入OpSchema类，用于操作模式的架构定义
from torch.distributed._tensor._op_schema import OpSchema

# 导入通用规则函数，如einop_rule和pointwise_rule
from torch.distributed._tensor.ops.common_rules import einop_rule, pointwise_rule
# 导入张量规格和元数据相关类，如DTensorSpec和TensorMeta
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta
# 导入测试运行函数，用于运行测试用例
from torch.testing._internal.common_utils import run_tests
# 导入分布式张量测试基础类DTensorTestBase及通信装饰器with_comms
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

# 获取aten命名空间下的操作函数
aten = torch.ops.aten

# CommonRulesTest类，继承自DTensorTestBase，用于测试通用规则
class CommonRulesTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        # 返回固定的世界大小为4，用于至少测试2D网格
        # 固定世界大小为4，因为我们需要测试至少2D网格
        return 4

    # 生成给定形状张量的元数据信息
    def _gen_tensor_meta(self, shape):
        # 创建一个空张量以获取其形状信息
        empty_tensor = torch.empty(shape)
        # 返回张量的元数据信息，包括形状、步长和数据类型
        return TensorMeta(
            empty_tensor.shape,
            empty_tensor.stride(),
            empty_tensor.dtype,
        )

    # 使用with_comms装饰器标记的测试方法，用于测试通信相关功能
    @with_comms
    def test_einop_basic_propagation(self):
        # 定义一个测试方法，用于测试 einsum 基本的传播特性

        # 创建一个 DeviceMesh 对象，使用设备类型和给定的世界大小范围
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # 设置 mm_call 为 aten.mm.default
        mm_call = aten.mm.default

        # 设置 mat1 和 mat2 的值，用于后续的矩阵操作
        mat1, mat2 = [-1, -1], [-1, 0]

        # 生成 mat1_tensor_meta 和 mat2_tensor_meta，分别表示 mat1 和 mat2 的张量元信息
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([4, 8]))

        # 使用给定的维度映射创建 mat1_spec 和 mat2_spec，表示 mat1 和 mat2 的张量规格
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )

        # 根据 einsum 规则 "mk,kn->mn" 和 OpSchema 创建输出分片
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(mm_call, (mat1_spec, mat2_spec), {})
        )

        # 获取输出的规格
        output_spec = output_sharding.output_spec

        # 断言输出规格不为 None
        self.assertIsNotNone(output_spec)

        # 断言输出规格的维度映射为 [-1, 0]
        self.assertEqual(output_spec.dim_map, [-1, 0])

        # 设置 mat1 和 mat2 的新值，用于下一次矩阵操作
        mat1, mat2 = [0, -1], [-1, -1]

        # 使用新的维度映射创建 mat1_spec 和 mat2_spec，表示 mat1 和 mat2 的张量规格
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )

        # 再次根据 einsum 规则 "mk,kn->mn" 和 OpSchema 创建输出分片
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(mm_call, (mat1_spec, mat2_spec), {})
        )

        # 获取输出的规格
        output_spec = output_sharding.output_spec

        # 断言输出规格不为 None
        self.assertIsNotNone(output_spec)

        # 断言输出规格的维度映射为 [0, -1]
        self.assertEqual(output_spec.dim_map, [0, -1])

        # 设置 mat1 和 mat2 的另一组新值，用于下一次矩阵操作
        mat1, mat2 = [-1, 0], [0, -1]

        # 使用新的维度映射创建 mat1_spec 和 mat2_spec，表示 mat1 和 mat2 的张量规格
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )

        # 再次根据 einsum 规则 "mk,kn->mn" 和 OpSchema 创建输出分片
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(mm_call, (mat1_spec, mat2_spec), {})
        )

        # 获取输出的规格
        output_spec = output_sharding.output_spec

        # 断言输出规格不为 None
        self.assertIsNotNone(output_spec)

        # 断言输出规格的第一个放置位置为部分放置
        self.assertTrue(output_spec.placements[0].is_partial())
    def test_einop_pointwise_propagation(self):
        # 创建设备网格对象，使用给定的设备类型和整数范围作为参数
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # 定义加法操作的 ATen 函数调用
        add_call = aten.add.Tensor

        # 创建一个大小为 [8, 8] 的张量元数据
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 8]))

        # 创建一个包含 [0, -1] 的列表作为第一个操作数的维度映射
        mat1 = [0, -1]

        # 使用设备网格对象和张量元数据创建第一个操作数的张量规范
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )

        # 使用 einstein 操作生成规则，将输出规范指定为 "ij,ij->ij"
        output_sharding = einop_rule(
            "ij,ij->ij", OpSchema(add_call, (mat1_spec, mat1_spec), {})
        )

        # 获取输出的张量规范
        output_spec = output_sharding.output_spec

        # 断言输出规范不为空
        self.assertIsNotNone(output_spec)

        # 断言输出规范的维度映射为 [0, -1]
        self.assertEqual(output_spec.dim_map, [0, -1])

        # 创建另一个大小为 [8, 8] 的张量元数据
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 8]))

        # 创建一个包含 [-1, 0, -1] 的列表作为第一个操作数的维度映射
        mat1 = [-1, 0, -1]

        # 使用设备网格对象和张量元数据创建第一个操作数的张量规范
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )

        # 创建一个大小为 [2] 的张量元数据
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([2]))

        # 创建一个包含 [-1] 的列表作为第二个操作数的维度映射
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, [-1], [], tensor_meta=mat2_tensor_meta
        )

        # 使用 einstein 操作生成规则，将输出规范指定为 "ijk,k->ijk"
        output_sharding = einop_rule(
            "ijk,k->ijk", OpSchema(add_call, (mat1_spec, mat2_spec), {})
        )

        # 获取输出的张量规范
        output_spec = output_sharding.output_spec

        # 断言输出规范不为空
        self.assertIsNotNone(output_spec)

        # 断言输出规范的维度映射为 [-1, 0, -1]
        self.assertEqual(output_spec.dim_map, [-1, 0, -1])

        # 创建一个大小为 [8, 8, 8] 的张量元数据
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 8, 8]))

        # 创建一个大小为 [1, 8] 的张量元数据
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([1, 8]))

        # 创建一个包含 [0, -1, -1] 的列表作为第一个操作数的维度映射
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, [0, -1, -1], [], tensor_meta=mat1_tensor_meta
        )

        # 创建一个包含 [-1, -1] 的列表作为第二个操作数的维度映射
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, [-1, -1], [], tensor_meta=mat2_tensor_meta
        )

        # 使用 einstein 操作生成规则，将输出规范指定为 "ijk,1k->ijk"
        output_sharding = einop_rule(
            "ijk,1k->ijk", OpSchema(add_call, (mat1_spec, mat2_spec), {})
        )

        # 获取输出的张量规范
        output_spec = output_sharding.output_spec

        # 断言输出规范不为空
        self.assertIsNotNone(output_spec)

        # 断言输出规范的维度映射为 [0, -1, -1]
        self.assertEqual(output_spec.dim_map, [0, -1, -1])
    def test_einop_merge_sharding(self):
        # 定义一个测试函数，用于测试 einsum 操作的合并和分片
        # 创建一个二维网格形状，其大小为 self.world_size，分为 self.world_size // 2 行和 self.world_size // 2 列
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        # 在指定设备类型上创建一个设备网格对象
        mesh = DeviceMesh(self.device_type, mesh_shape)

        # 定义矩阵乘法的默认调用函数
        mm_call = aten.mm.default

        # 定义两个矩阵 mat1 和 mat2
        mat1, mat2 = [0, -1], [-1, 1]
        # 生成 mat1 的张量元数据，大小为 [8, 4]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        # 生成 mat2 的张量元数据，大小为 [4, 8]
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([4, 8]))
        # 根据给定的网格和矩阵索引生成 mat1 的张量规范
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        # 根据给定的网格和矩阵索引生成 mat2 的张量规范
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        # 使用给定的操作模式和输入规范，调用 einsum 规则函数，生成输出的分片规范
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(mm_call, (mat1_spec, mat2_spec), {})
        )
        # 获取输出规范
        output_spec = output_sharding.output_spec
        # 断言输出规范不为空
        self.assertIsNotNone(output_spec)
        # 断言输出规范的维度映射为 [0, 1]
        self.assertEqual(output_spec.dim_map, [0, 1])

    @with_comms
    # 定义一个测试方法，用于测试 einsum 运算的线性性质
    def test_einop_linearity(self):
        # 创建一个表示网格形状的张量，并根据世界大小重新形成
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        # 使用 DeviceMesh 类创建一个网格对象
        mesh = DeviceMesh(self.device_type, mesh_shape)

        # 设置矩阵乘法调用的默认值
        mm_call = aten.mm.default

        # 定义两个矩阵 mat1 和 mat2 的索引
        mat1, mat2 = [0, -1], [-1, -1]
        # 生成 mat1 和 mat2 的张量元数据
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([4, 8]))
        # 根据维度映射和张量元数据创建 mat1_spec
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [1], tensor_meta=mat1_tensor_meta
        )
        # 根据维度映射和张量元数据创建 mat2_spec
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )

        # 当线性性关闭时，输出分片建议为空，返回重新划分输入的建议（即全局归约一个输入）
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(mm_call, (mat1_spec, mat2_spec), {})
        )
        self.assertIsNone(output_sharding.output_spec)
        suggestions = output_sharding.redistribute_schema
        self.assertIsNotNone(suggestions)
        suggested_spec = suggestions.args_schema[0]
        # 验证建议的规范中第二个位置不是部分（partial）的
        self.assertFalse(suggested_spec.placements[1].is_partial())

        # 当开启矩阵乘法的线性性时，应该返回转换成部分（partial）的位置的建议
        output_sharding = einop_rule(
            "mk,kn->mn",
            OpSchema(mm_call, (mat1_spec, mat2_spec), {}),
            linearity=True,
        )
        self.assertIsNone(output_sharding.output_spec)
        suggestions = output_sharding.redistribute_schema
        self.assertIsNotNone(suggestions)
        mat2_spec = suggestions.args_schema[1]
        # 验证 mat2 的网格维度 1 现在是部分（partial）
        self.assertTrue(mat2_spec.placements[1].is_partial())

        # 当开启点对点操作的线性性时，应该返回转换成部分（partial）的位置的建议
        add_call = aten.add.Tensor
        mat1, mat2 = [0, -1], [0, -1]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 6]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([8, 6]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [1], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )

        output_sharding = einop_rule(
            "ij,ij->ij",
            OpSchema(add_call, (mat1_spec, mat2_spec), {}),
            linearity=True,
        )
        self.assertIsNone(output_sharding.output_spec)
        suggestions = output_sharding.redistribute_schema
        self.assertIsNotNone(suggestions)
        mat2_spec = suggestions.args_schema[1]
        # 验证 mat2 的网格维度 1 现在是部分（partial）
        self.assertTrue(mat2_spec.placements[1].is_partial())

    @with_comms
    def test_einop_multi_sharding_on_mesh_dim(self):
        # einop prop with multi sharding on same mesh dim
        # 定义一个表示设备网格形状的张量，其长度为当前世界大小
        mesh_shape = torch.arange(self.world_size)
        # 使用设备类型和网格形状创建设备网格对象
        mesh = DeviceMesh(self.device_type, mesh_shape)

        # 定义一个矩阵乘法调用函数
        mm_call = aten.mm.default
        # 定义两个矩阵的索引，表示它们在设备网格中的位置
        mat1, mat2 = [0, -1], [0, -1]
        # 生成第一个矩阵的张量元信息，其大小为 [8, 12]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 12]))
        # 生成第二个矩阵的张量元信息，其大小为 [12, 4]
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([12, 4]))
        
        # 使用设备网格和矩阵索引创建第一个矩阵的规范化张量规范
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        # 使用设备网格和矩阵索引创建第二个矩阵的规范化张量规范
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        
        # 使用 einstein 操作规则，计算输出张量规范
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(mm_call, (mat1_spec, mat2_spec), {})
        )
        
        # 获取输出张量规范
        output_spec = output_sharding.output_spec
        # 断言输出张量规范为空
        self.assertIsNone(output_spec)
        # 断言输出分片模式建议不为空
        self.assertIsNotNone(output_sharding.redistribute_schema)

        # 确保建议重新分片第二个参数，通过全局聚合其张量维度分片
        schema_suggestion = output_sharding.redistribute_schema
        # 断言建议的参数模式中第一个参数的维度映射为 [0, -1]
        self.assertEqual(schema_suggestion.args_schema[0].dim_map, [0, -1])
        # 断言建议的参数模式中第二个参数的维度映射为 [-1, -1]
        self.assertEqual(schema_suggestion.args_schema[1].dim_map, [-1, -1])

    @with_comms
    def test_einop_errors(self):
        # 定义一个表示设备网格形状的张量，其形状是世界大小的一半
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        # 使用设备类型和网格形状创建设备网格对象
        mesh = DeviceMesh(self.device_type, mesh_shape)

        # 定义一个加法调用函数
        add_call = aten.add.Tensor
        # 定义两个矩阵的索引，表示它们在设备网格中的位置
        mat1, mat2 = [0, -1], [1, -1]
        # 生成第一个矩阵的张量元信息，其大小为 [8, 4]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        # 生成第二个矩阵的张量元信息，其大小为 [8, 4]
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        
        # 使用设备网格和矩阵索引创建第一个矩阵的规范化张量规范
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        # 使用设备网格和矩阵索引创建第二个矩阵的规范化张量规范
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )

        # 断言在运行时捕获到异常，异常信息包含 "sharded two different ways:"
        with self.assertRaisesRegex(RuntimeError, "sharded two different ways:"):
            # 使用 einstein 操作规则，计算输出张量规范
            einop_rule("ij,ij->ij", OpSchema(add_call, (mat1_spec, mat2_spec), {}))
    def test_pointwise_rules_broadcasting(self):
        # 创建设备网格对象，使用给定的设备类型和世界大小范围
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # 设置调用的操作为 aten.where.self
        where_call = aten.where.self
        # 定义输入参数
        inp1, inp2, inp3 = [0], [], [-1, -1]
        # 生成输入张量的元数据对象
        inp1_tensor_meta = self._gen_tensor_meta(torch.Size([8]))
        inp2_tensor_meta = self._gen_tensor_meta(torch.Size([]))
        inp3_tensor_meta = self._gen_tensor_meta(torch.Size([1, 1]))
        # 根据给定的参数和元数据创建条件张量规范对象
        condition = DTensorSpec.from_dim_map(
            mesh, inp1, [], tensor_meta=inp1_tensor_meta
        )
        # 根据给定的参数和元数据创建 self 张量规范对象
        self_tensor = DTensorSpec.from_dim_map(
            mesh, inp2, [], tensor_meta=inp2_tensor_meta
        )
        # 根据给定的参数和元数据创建 other 张量规范对象
        other_tensor = DTensorSpec.from_dim_map(
            mesh, inp3, [], tensor_meta=inp3_tensor_meta
        )
        # 使用 pointwise_rule 函数传播点对点分片并进行广播
        output_sharding = pointwise_rule(
            OpSchema(where_call, (condition, self_tensor, other_tensor), {})
        )
        # 获取输出规范对象
        output_spec = output_sharding.output_spec
        # 断言输出规范对象不为空
        self.assertIsNotNone(output_spec)
        # 断言输出规范对象的维度映射为 [-1, 0]

    @with_comms
    def test_pointwise_rules_suggestion(self):
        # 创建设备网格对象，使用给定的设备类型和世界大小范围
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # 设置调用的操作为 aten.lerp.Scalar
        lerp_call = aten.lerp.Scalar
        # 设置输入参数
        inp1, inp2 = [-1, -1], [-1, 0]
        # 生成输入张量的元数据对象
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        # 根据给定的参数和元数据创建 mat1 张量规范对象
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, inp1, [], tensor_meta=mat1_tensor_meta
        )
        # 根据给定的参数和元数据创建 mat2 张量规范对象
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, inp2, [], tensor_meta=mat2_tensor_meta
        )
        # 向 OpSchema 添加一个位置参数 -1
        output_sharding = pointwise_rule(
            OpSchema(lerp_call, (mat1_spec, mat2_spec, -1), {})
        )
        # 断言输出规范对象为空
        self.assertIsNone(output_sharding.output_spec)
        # 断言重新分配模式不为空
        self.assertIsNotNone(output_sharding.redistribute_schema)

        # 确保点对点规则的建议仍然具有非 DTensorSpec 的位置参数
        schema_suggestion = output_sharding.redistribute_schema
        # 断言参数模式列表长度为 3
        self.assertEqual(len(schema_suggestion.args_schema), 3)
        # 断言第三个参数为 -1

    @with_comms
    # 定义一个测试方法，用于测试点对点多重分片在网格维度上的应用
    def test_pointwise_multi_sharding_on_mesh_dim(self):
        # 创建一个二维网格形状，以测试点对点分片
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        # 创建一个设备网格对象，使用给定的设备类型和网格形状
        mesh = DeviceMesh(self.device_type, mesh_shape)

        # 指定一个张量运算加法的调用方式
        add_call = aten.add.Tensor

        # 基本情况，测试隐式广播的形状对齐
        mat1, mat2 = [-1, 0], [0]
        # 生成 mat1 的张量元信息
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([20, 6]))
        # 生成 mat2 的张量元信息
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([6]))
        # 根据映射创建 mat1 的张量规范
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        # 根据映射创建 mat2 的张量规范
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        # 应用点对点规则生成输出的分片规范
        output_sharding = pointwise_rule(OpSchema(add_call, (mat1_spec, mat2_spec), {}))
        # 获取输出的规范
        output_spec = output_sharding.output_spec
        # 断言输出规范不为空
        self.assertIsNotNone(output_spec)
        # 断言输出规范的维度映射为 [-1, 0]
        self.assertEqual(output_spec.dim_map, [-1, 0])

        # 更复杂的情况，需要重新分片一个输入以对齐分片
        mat1, mat2 = [0, -1, -1, 1], [0, -1, 1]
        # 生成 mat1 的张量元信息
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([12, 1, 1, 8]))
        # 生成 mat2 的张量元信息
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([12, 4, 8]))
        # 根据映射创建 mat1 的张量规范
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        # 根据映射创建 mat2 的张量规范
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        # 应用点对点规则生成输出的分片规范
        output_sharding = pointwise_rule(OpSchema(add_call, (mat1_spec, mat2_spec), {}))
        # 获取输出的规范
        output_spec = output_sharding.output_spec
        # 断言输出规范为空
        self.assertIsNone(output_spec)
        # 断言输出分片对象的重新分片方案不为空
        self.assertIsNotNone(output_sharding.redistribute_schema)

        # 确保建议重新分片第一个参数
        # 通过 all_gather 第一个张量维度的分片
        schema_suggestion = output_sharding.redistribute_schema
        # 断言重新分片方案的第一个参数的维度映射为 [-1, -1, -1, 1]
        self.assertEqual(schema_suggestion.args_schema[0].dim_map, [-1, -1, -1, 1])
        # 断言重新分片方案的第二个参数的维度映射与原始输入一致
        self.assertEqual(schema_suggestion.args_schema[1].dim_map, mat2)

    @with_comms
    def test_pointwise_enforce_sharding_multi_sharding_on_mesh_dim(self):
        # 定义测试方法：测试点对点操作在网格维度上多重分片

        # 创建二维网格形状，网格大小为 self.world_size
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        # 创建设备网格对象，使用 self.device_type 和定义的 mesh_shape
        mesh = DeviceMesh(self.device_type, mesh_shape)

        # 定义加法操作的调用
        add_call = aten.add_.Tensor

        # 需要对输入进行重分片以对齐分片的更高级情况
        mat1, mat2 = [0, -1, 1], [-1, -1, 0]
        # 生成 mat1 和 mat2 的张量元信息
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([12, 4, 8]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([12, 1, 8]))

        # 使用网格和张量元信息生成张量规格 mat1_spec
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        # 使用网格和张量元信息生成张量规格 mat2_spec
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )

        # 使用点对点操作规则生成输出分片
        output_sharding = pointwise_rule(OpSchema(add_call, (mat1_spec, mat2_spec), {}))
        # 获取输出规格
        output_spec = output_sharding.output_spec

        # 断言输出规格为空
        self.assertIsNone(output_spec)
        # 断言输出分片方案不为空
        self.assertIsNotNone(output_sharding.redistribute_schema)

        # 确保建议是对第二个参数进行重分片，
        # 因为我们应该强制执行第一个参数的分片
        schema_suggestion = output_sharding.redistribute_schema
        # 断言建议的第一个参数的维度映射与 mat1 相同
        self.assertEqual(schema_suggestion.args_schema[0].dim_map, mat1)
        # 断言建议的第二个参数的维度映射与 mat1 相同
        self.assertEqual(schema_suggestion.args_schema[1].dim_map, mat1)
# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 调用 run_tests 函数来执行测试
    run_tests()
```