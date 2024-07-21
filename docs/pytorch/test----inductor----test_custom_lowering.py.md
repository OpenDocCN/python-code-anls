# `.\pytorch\test\inductor\test_custom_lowering.py`

```py
# Owner(s): ["module: inductor"]

# 导入单元测试模块
import unittest

# 导入 PyTorch 库
import torch

# 从 torch._inductor.ir 模块导入 Pointwise 类
from torch._inductor.ir import Pointwise
# 从 torch._inductor.lowering 模块导入 register_lowering 函数
from torch._inductor.lowering import register_lowering
# 从 torch._inductor.test_case 模块导入 TestCase 类作为 InductorTestCase
from torch._inductor.test_case import TestCase as InductorTestCase
# 从 torch._inductor.virtualized 模块导入 ops
from torch._inductor.virtualized import ops

# 从 torch.testing._internal.inductor_utils 模块导入 HAS_CPU 和 HAS_CUDA
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA


# 这些测试用例检查主 PyTorch 仓库中不存在的降低问题
class TestCustomLowering(InductorTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # 创建 test_inductor_ops 的库对象，标识为 "DEF"
        cls.test_inductor_ops = torch.library.Library(
            "test_inductor_ops", "DEF"
        )
        # 创建 test_inductor_ops 的 CUDA 实现的库对象，标识为 "IMPL", "CUDA"
        cls.impl_cuda = torch.library.Library(
            "test_inductor_ops", "IMPL", "CUDA"
        )
        # 创建 test_inductor_ops 的 Meta 实现的库对象，标识为 "IMPL", "Meta"
        cls.impl_meta = torch.library.Library(
            "test_inductor_ops", "IMPL", "Meta"
        )
        # 注册 jagged_to_padded_dense 函数
        cls._register_jagged_to_padded_dense()

    @classmethod
    def tearDown(cls):
        super().tearDownClass()

    @classmethod
    def _register_jagged_to_padded_dense(cls):
        # 定义对应于 fbgemm.jagged_to_padded_dense_forward 的近似方法
        cls.test_inductor_ops.define(
            "jagged_to_padded_dense(Tensor input, Tensor offsets, SymInt max_seq_len, Scalar pad_value) -> Tensor"
        )

        # 定义在元数据处理中的 j2pd_meta 函数
        def j2pd_meta(inp, offsets, max_seq_len, pad_value):
            # 创建一个空的张量，用于存储结果
            return torch.empty(
                (offsets.shape[0] - 1, max_seq_len, inp.shape[1]),
                device=inp.device,
                dtype=inp.dtype,
            )

        # 定义在 CUDA 加速环境中的 j2pd_cuda 函数
        def j2pd_cuda(inp, offsets, max_seq_len, pad_value):
            # 创建一个填充指定值的张量
            res = torch.full(
                (offsets.shape[0] - 1, max_seq_len, inp.shape[1]),
                pad_value,
                device=inp.device,
                dtype=inp.dtype,
            )
            # 使用循环将输入数据填充到结果张量中
            for b in range(offsets.shape[0] - 1):
                for r in range(offsets[b + 1] - offsets[b]):
                    res[b][r] = inp[offsets[b] + r]
            return res

        # 定义在较低级别运行时的 j2pd_lowering 函数
        def j2pd_lowering(inp, offsets, max_seq_len, pad_value):
            # 使用偏移量和输入数据创建加载器对象
            offsets_loader = offsets.make_loader()
            inp_loader = inp.make_loader()
            jagged_len = inp.get_size()[0]  # 获取输入数据的大小信息
            offsets_dtype = offsets.get_dtype()  # 获取偏移量数据类型

            # 定义内部函数 inner_fn
            def inner_fn(index):
                batch_idx, seq_idx, emb_idx = index

                # 计算起始和结束索引
                begin_idx = ops.indirect_indexing(
                    offsets_loader([batch_idx]),
                    jagged_len + 1,
                )
                end_idx = offsets_loader([batch_idx + 1])
                jagged_idx = begin_idx + seq_idx

                # 使用掩码操作处理数据
                return ops.masked(
                    ops.lt(
                        ops.index_expr(jagged_idx, offsets_dtype),
                        end_idx,
                    ),
                    lambda: inp_loader([jagged_idx, emb_idx]),
                    pad_value,
                )

            # 创建 Pointwise 对象并返回
            return Pointwise.create(
                device=inp.get_device(),
                dtype=inp.get_dtype(),
                inner_fn=inner_fn,
                ranges=[offsets.get_size()[0] - 1, max_seq_len, inp.get_size()[1]],
            )

        # 将 j2pd_lowering 函数注册为降低操作的方法
        register_lowering(
            torch.ops.test_inductor_ops.jagged_to_padded_dense, type_promotion_kind=None
        )(j2pd_lowering)

        # 在元数据实现中注册 j2pd_meta 函数
        cls.impl_meta.impl("jagged_to_padded_dense", j2pd_meta)
        # 在 CUDA 实现中注册 j2pd_cuda 函数
        cls.impl_cuda.impl("jagged_to_padded_dense", j2pd_cuda)

    @unittest.skipIf(not HAS_CUDA, "CUDA needed")
    # 定义一个测试函数，用于测试 CUDA 下的 `jagged_to_padded_dense` 函数的正确性
    def test_jagged_to_padded_dense_sanity_cuda(self):
        
        # 定义内部函数 fn，接受输入数据、偏移量和最大序列长度作为参数，调用 `jagged_to_padded_dense` 函数并返回结果
        def fn(inp, offsets, max_seq_len):
            return torch.ops.test_inductor_ops.jagged_to_padded_dense(
                inp, offsets, max_seq_len, 60.0
            )

        # 创建一个大小为 (9, 96) 的随机张量 inp，设备为 CUDA
        inp = torch.rand((9, 96), device="cuda")
        
        # 创建偏移量张量 offsets，其值为 [0, 2, 5, 9]，数据类型为 torch.int32，设备为 CUDA
        offsets = torch.tensor([0, 2, 5, 9], dtype=torch.int32, device="cuda")
        
        # 设置最大序列长度为 4
        max_seq_len = 4

        # 调用 fn 函数，将输入数据 inp、偏移量 offsets 和最大序列长度 max_seq_len 作为参数
        res = fn(inp, offsets, max_seq_len)
        
        # 断言：验证 res 的第一个序列的第一个元素与输入 inp 的第一个元素相等
        self.assertEqual(inp[0], res[0][0])
        
        # 断言：验证 res 的第一个序列的第二个元素与输入 inp 的第二个元素相等
        self.assertEqual(inp[1], res[0][1])
        
        # 断言：验证 res 的第二个序列的第一个元素与输入 inp 的第三个元素相等
        self.assertEqual(inp[2], res[1][0])
        
        # 断言：验证 res 的第二个序列的第二个元素与输入 inp 的第四个元素相等
        self.assertEqual(inp[3], res[1][1])
        
        # 断言：验证 res 的第三个序列的第一个元素与输入 inp 的第六个元素相等
        self.assertEqual(inp[5], res[2][0])
        
        # 断言：验证 res 的第三个序列的第四个元素与输入 inp 的第九个元素相等
        self.assertEqual(inp[8], res[2][3])

        # 编译 fn 函数以获取优化版本 fn_opt
        fn_opt = torch.compile(fn)

        # 断言：比较原始 fn 函数和编译后的 fn_opt 函数的输出是否一致
        self.assertEqual(
            fn(inp, offsets, max_seq_len), fn_opt(inp, offsets, max_seq_len)
        )

    # 如果没有 CUDA 支持，跳过此测试
    @unittest.skipIf(not HAS_CUDA, "CUDA needed")
    def test_jagged_to_padded_dense_zero_size(self):
        # 之前，输入值的掩码被完全去除，导致尝试读取零大小张量的索引 0 而引发 IMA 错误
        # 定义内部函数 fn，接受输入数据、偏移量和最大序列长度作为参数，执行一系列操作后调用 `jagged_to_padded_dense` 函数并返回结果
        def fn(inp, offsets, max_seq_len):
            # 将输入 inp 与大小为 (1, 96, 1) 的全 1 张量进行批矩阵乘法，结果重塑为零大小张量
            inp = torch.bmm(inp, torch.ones((1, 96, 1), device="cuda")).view((0, 1))
            return torch.ops.test_inductor_ops.jagged_to_padded_dense(
                inp, offsets, max_seq_len, 60.0
            )

        # 创建一个大小为 (1, 0, 96) 的随机张量 inp，设备为 CUDA
        inp = torch.rand((1, 0, 96), device="cuda")
        
        # 创建零大小的偏移量张量 offsets，大小为 1025，数据类型为 torch.int32，设备为 CUDA
        offsets = torch.zeros(1025, device="cuda", dtype=torch.int32)
        
        # 设置最大序列长度为 20
        max_seq_len = 20

        # 编译 fn 函数以获取优化版本 fn_opt
        fn_opt = torch.compile(fn)

        # 断言：比较原始 fn 函数和编译后的 fn_opt 函数的输出是否一致
        self.assertEqual(
            fn(inp, offsets, max_seq_len), fn_opt(inp, offsets, max_seq_len)
        )
# 如果当前脚本被直接执行（而不是被导入到其他模块中执行）
if __name__ == "__main__":
    # 从 torch 库中导入测试用例运行函数 run_tests
    from torch._inductor.test_case import run_tests
    
    # 如果已经配置了 HAS_CPU 或者 HAS_CUDA 环境变量
    if HAS_CPU or HAS_CUDA:
        # 运行测试用例，并指定需要的依赖为 "filelock"
        run_tests(needs="filelock")
```