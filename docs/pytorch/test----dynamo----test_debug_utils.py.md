# `.\pytorch\test\dynamo\test_debug_utils.py`

```
# Owner(s): ["module: dynamo"]

# 引入单元测试模块
import unittest

# 引入 PyTorch 模块
import torch

# 引入 functorch 模块中的 make_fx 函数
from functorch import make_fx

# 引入 torch._dynamo.debug_utils 模块中的 debug_utils 和 aot_graph_input_parser 函数
from torch._dynamo import debug_utils
from torch._dynamo.debug_utils import aot_graph_input_parser

# 引入 torch._dynamo.test_case 模块中的 TestCase 类
from torch._dynamo.test_case import TestCase

# 引入 torch.testing._internal.inductor_utils 模块中的 HAS_CUDA 变量
from torch.testing._internal.inductor_utils import HAS_CUDA

# 如果没有 CUDA，跳过测试用例
requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")

# 定义常量 f32 和 i64，分别为 torch.float32 和 torch.int64 类型
f32 = torch.float32
i64 = torch.int64
i32 = torch.int32

# 定义测试类 TestDebugUtils，继承自 TestCase
class TestDebugUtils(TestCase):

    # 定义测试方法 test_cast_model_to_fp64_dtype_args
    def test_cast_model_to_fp64_dtype_args(self):
        # 测试确保 dtype 参数转换为 fp64

        # 定义函数 fn，接受参数 x
        def fn(x):
            # 返回四个值，分别是 x 转换为 torch.float16 后的结果，
            # x 转换为 torch.float16 后的结果，
            # 全部元素为 2 的 torch.float32 张量，
            # 和 x 相同形状的新空张量
            return (
                torch.ops.prims.convert_element_type(x, torch.float16),
                x.to(torch.float16),
                torch.full(x.shape, 2, dtype=torch.float32, device=x.device),
                x.new_empty(x.shape),
            )

        # 生成一个在 CPU 上的随机张量 x
        x = torch.randn(32, device="cpu")

        # 获取 torch._decomp.core_aten_decompositions() 的结果
        decomps = torch._decomp.core_aten_decompositions()

        # 使用 make_fx 函数对 fn 进行功能扩展，传入 decomposition_table 参数
        fx = make_fx(fn, decomposition_table=decomps)(x)

        # 断言 fx.code 去除首部空格后的内容与指定的字符串匹配
        self.assertExpectedInline(
            fx.code.lstrip(),
            """\
def forward(self, x_1):
    convert_element_type = torch.ops.prims.convert_element_type.default(x_1, torch.float16)
    _to_copy = torch.ops.aten._to_copy.default(x_1, dtype = torch.float16);  x_1 = None
    full = torch.ops.aten.full.default([32], 2, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    empty = torch.ops.aten.empty.memory_format([32], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    return (convert_element_type, _to_copy, full, empty)
    """,  # NOQA: B950
        )

        # 调用 debug_utils.cast_to_fp64 函数，对 fx 和 x 进行转换
        fp64_model, fp64_examples = debug_utils.cast_to_fp64(fx, (x,))

        # 断言 fp64_examples 的结果为 (x 转换为 torch.float64 后的结果,)
        self.assertEqual(fp64_examples, (x.to(torch.float64),))

        # 再次断言 fx.code 去除首部空格后的内容与指定的字符串匹配
        self.assertExpectedInline(
            fx.code.lstrip(),
            """\
def forward(self, x_1):
    convert_element_type = torch.ops.prims.convert_element_type.default(x_1, torch.float64)
    _to_copy = torch.ops.aten._to_copy.default(x_1, dtype = torch.float64);  x_1 = None
    full = torch.ops.aten.full.default([32], 2, dtype = torch.float64, device = device(type='cpu'), pin_memory = False)
    empty = torch.ops.aten.empty.memory_format([32], dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    return (convert_element_type, _to_copy, full, empty)
    """,  # NOQA: B950
        )

    # 标记测试方法需要 CUDA
    @requires_cuda
    @requires_cuda
    `
        # 测试函数，命名为 test_sym_aot_graph_parser，属于某个测试类的方法
        def test_sym_aot_graph_parser(self):
            # 定义一个名为 forward 的内部函数，接受多个参数，类型注解使用字符串表示
            def forward(
                self,
                primals_1: "f32[1001, 6]",  # 参数 primals_1 类型为 f32 的二维数组，形状为 [1001, 6]
                primals_2: "f32[s0]",  # 参数 primals_2 类型为 f32 的一维数组，形状由符号 s0 指定
                primals_3: "Sym(s0)",  # 参数 primals_3 类型为符号张量，形状由符号 s0 指定
                primals_4: "f32[s1]",  # 参数 primals_4 类型为 f32 的一维数组，形状由符号 s1 指定
                primals_5: "Sym(s1)",  # 参数 primals_5 类型为符号张量，形状由符号 s1 指定
            ):
                # 将一个张量常量赋值给 _tensor_constant0，类型为 i64 的一维数组，形状为 [4190]
                _tensor_constant0: "i64[4190]" = self._tensor_constant0
    
            # 调用 aot_graph_input_parser 函数，传入 forward 函数及其他参数，返回解析后的关键字参数字典
            kwargs = aot_graph_input_parser(
                forward, device="cuda", sym_shapes={"s0": 10}, default_sym_shape=5
            )
    
            # 断言 kwargs 字典中 primals_2 的形状为 [10]
            self.assertEqual(list(kwargs["primals_2"].shape), [10])
            # 断言 kwargs 字典中 primals_3 的值为 10
            self.assertEqual(kwargs["primals_3"], 10)
    
            # 断言 kwargs 字典中 primals_4 的形状为 [5]
            self.assertEqual(list(kwargs["primals_4"].shape), [5])
            # 断言 kwargs 字典中 primals_5 的值为 5
            self.assertEqual(kwargs["primals_5"], 5)
# 如果当前脚本作为主程序运行（而非被导入），则执行以下代码块
if __name__ == "__main__":
    # 导入名为 run_tests 的函数，来自 torch._dynamo.test_case 模块
    from torch._dynamo.test_case import run_tests

    # 运行导入的 run_tests 函数，通常用于执行测试用例
    run_tests()
```