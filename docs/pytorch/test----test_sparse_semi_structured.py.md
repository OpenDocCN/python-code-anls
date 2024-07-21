# `.\pytorch\test\test_sparse_semi_structured.py`

```
# Owner(s): ["module: sparse"]
# 引入必要的库和模块
import itertools  # 导入 itertools 库，用于生成迭代器的工具函数
import random  # 导入 random 库，用于生成随机数
import unittest  # 导入 unittest 框架，用于编写和运行单元测试

import torch  # 导入 PyTorch 库
from torch import nn  # 从 PyTorch 中导入神经网络模块 nn
import torch.nn.functional as F  # 导入 PyTorch 中的函数模块 F

# 导入稀疏相关的类和函数
from torch.sparse import (
    SparseSemiStructuredTensor,  # 稀疏半结构化张量
    SparseSemiStructuredTensorCUSPARSELT,  # 使用 CUSPARSELT 加速的稀疏半结构化张量
    SparseSemiStructuredTensorCUTLASS,  # 使用 CUTLASS 加速的稀疏半结构化张量
    to_sparse_semi_structured,  # 将普通张量转换为稀疏半结构化张量的函数
)

# 导入稀疏半结构化相关的转换函数
from torch.sparse._semi_structured_conversions import (
    sparse_semi_structured_from_dense_cutlass,  # 使用 CUTLASS 从密集张量转换为稀疏半结构化张量
    _sparse_semi_structured_tile,  # 稀疏半结构化张量的瓦片化函数
    _compute_compressed_swizzled_bitmask,  # 计算压缩交换位掩码的函数
)

# 导入用于测试的工具函数和类
from torch.testing import make_tensor  # 导入创建张量的测试函数

# 导入测试相关的设备类型和数据类型
from torch.testing._internal.common_device_type import (
    dtypes,  # 数据类型
    instantiate_device_type_tests,  # 实例化设备类型测试
)

# 导入所有数据类型和复杂类型
from torch.testing._internal.common_dtype import all_types_and_complex

# 导入与测试相关的动态模块
import torch._dynamo.test_case

# 导入测试相关的工具函数和类
from torch.testing._internal.common_utils import (
    parametrize,  # 参数化装饰器
    run_tests,  # 运行测试函数
    subtest,  # 测试子模块
    TestCase,  # 测试用例类
    TEST_WITH_ROCM,  # 是否在 ROCM 平台测试
    IS_WINDOWS,  # 是否在 Windows 环境下
)

# 导入 pytest 测试框架
import pytest

# 导入 Triton 框架的相关函数
from torch.utils._triton import has_triton

# 支持的稀疏半结构化张量的数据类型列表
SEMI_STRUCTURED_SUPPORTED_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.int8]
SEMI_STRUCTURED_SUPPORTED_BACKENDS = {}

# 检查是否支持 SM8X 架构
_IS_SM8X = False
if torch.cuda.is_available():
    _IS_SM8X = torch.cuda.get_device_capability(0)[0] == 8
    SEMI_STRUCTURED_SUPPORTED_BACKENDS["cutlass"] = SparseSemiStructuredTensorCUTLASS

    # 检查是否支持 cusparselt，用于加速
    try:
        torch._cslt_compress(torch.ones(128, 256).cuda())
        SEMI_STRUCTURED_SUPPORTED_BACKENDS["cusparselt"] = SparseSemiStructuredTensorCUSPARSELT
    except Exception:
        pass

# 推理数据类型
inference_dtypes = dtypes(torch.float16, torch.bfloat16, torch.float32, torch.int8)

# 训练数据类型
training_dtypes = dtypes(torch.float16, torch.bfloat16)

# 参数化测试支持的后端
parametrize_backends = parametrize("backend", SEMI_STRUCTURED_SUPPORTED_BACKENDS)

# 不同数据类型的容差和相对误差设置
atol_rtol_kw = {
    torch.float16: {
        "rtol": 1e-3,
        "atol": 1e-3,
    },
    torch.bfloat16: {
        "rtol": 1e-1,
        "atol": 1e-1,
    },
}

# 从原始密集张量生成稀疏半结构化张量，使用 CUTLASS 算法
def sparse24_largest_mask_2d(original):
    sparse = SparseSemiStructuredTensorCUTLASS.prune_dense_static_sort(original)
    return sparse.to_dense().bool()

# 从原始密集张量生成 24x24 稀疏半结构化张量
def sparsify24_dense(original):
    return sparse24_largest_mask_2d(original) * original

# 随机生成稀疏半结构化张量的掩码
def rand_sparse_semi_structured_mask(
    r, c, dtype=torch.float16, device="cuda", choice=None
):
    """
    This function returns a 1:2 sparse matrix of size (r, c).
    Note that this means this matrix will also be 2:4 and 4:8 sparse as well.
    """
    choices = [[0, 1], [1, 0]]
    mask_entries = [choice or random.choice(choices) for i in range(r * c // 2)]

    return (
        torch.tensor(mask_entries, dtype=dtype, device=device)
        .reshape(r, c)
        .contiguous()
    )

# 随机生成稀疏半结构化张量
def rand_sparse_semi_structured(r, c, dtype, device, choice=None):
    pattern = '2by4' if dtype != torch.float32 else '1by2'
    # 如果模式为 '1by2'，设置稀疏矩阵块的大小为 2
    if pattern == '1by2':
        ksparse = 2
        # 定义稀疏模式的选择列表，包含两种可能的稀疏矩阵块
        choices = [
            [0, 1],
            [1, 0]
        ]
    # 如果模式为 '2by4'，设置稀疏矩阵块的大小为 4
    elif pattern == '2by4':
        ksparse = 4
        # 定义稀疏模式的选择列表，包含六种可能的稀疏矩阵块
        choices = [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1]
        ]
    # 根据 r * c // ksparse 的大小，生成随机或预定义的稀疏矩阵块条目列表
    mask_entries = [choice or random.choice(choices) for i in range(r * c // ksparse)]
    # 创建布尔类型的张量 mask，用于表示稀疏矩阵块的位置
    mask = torch.tensor(mask_entries, dtype=torch.bool).view(r, c).to(device)
    # 创建指定形状、类型和设备的密集张量 dense
    dense = make_tensor(r, c, dtype=dtype, device=device)
    # 将 dense 中值为 0 的元素设为 1，以避免在稀疏矩阵块未应用时出现 0
    dense[dense == 0] = 1  # To prevent zeros except where mask applied.
    # 使用 mask 将 dense 中非稀疏矩阵块位置的元素置为 0
    dense = dense.masked_fill(~mask, 0)
    # 返回处理后的密集张量 dense
    return dense
# 定义一个生成稀疏半结构化张量模式的函数，返回两个稀疏张量
def rand_sparse_semi_structured_all_patterns(r, c, dtype, device):
    # 根据数据类型选择模式
    pattern = '2by4' if dtype != torch.float32 else '1by2'
    # 根据模式选择稀疏度
    if pattern == '1by2':
        ksparse = 2
        # 不同的稀疏模式选项
        choices = [
            [[0, 0], [0, 1]],
            [[0, 1], [0, 1]],
            [[1, 0], [1, 0]],
            [[1, 1], [1, 0]]
        ]
    elif pattern == '2by4':
        ksparse = 4
        # 不同的稀疏模式选项
        choices = [
            [[0, 0, 0, 0], [0, 0, 1, 1]],
            [[0, 0, 0, 1], [0, 0, 1, 1]],
            [[0, 0, 1, 0], [0, 0, 1, 1]],
            [[0, 0, 1, 1], [0, 0, 1, 1]],
            [[0, 1, 0, 0], [0, 1, 1, 0]],
            [[0, 1, 0, 1], [0, 1, 0, 1]],
            [[0, 1, 1, 0], [0, 1, 1, 0]],
            [[0, 1, 1, 1], [0, 1, 0, 1]],
            [[1, 0, 0, 0], [1, 0, 1, 0]],
            [[1, 0, 0, 1], [1, 0, 0, 1]],
            [[1, 0, 1, 0], [1, 0, 1, 0]],
            [[1, 0, 1, 1], [1, 0, 0, 1]],
            [[1, 1, 0, 0], [1, 1, 0, 0]],
            [[1, 1, 0, 1], [1, 1, 0, 0]],
            [[1, 1, 1, 0], [1, 1, 0, 0]],
            [[1, 1, 1, 1], [1, 1, 0, 0]],
        ]
    
    # 随机生成要遮罩的行索引
    mask_rows = [random.randint(0, len(choices) - 1) for i in range(r * c // ksparse)]

    # 定义遮罩中的两种条目：逆向和数值
    COL_INV, COL_VAL = 0, 1
    mask_entries_inv = [choices[i][COL_INV] for i in mask_rows]
    mask_entries_val = [choices[i][COL_VAL] for i in mask_rows]

    # 创建逆向和数值遮罩张量
    mask_inv = torch.tensor(mask_entries_inv, dtype=torch.bool).view(r, c).to(device)
    mask_val = torch.tensor(mask_entries_val, dtype=torch.bool).view(r, c).to(device)

    # 创建稠密张量并将所有零值替换为1，以防止遮罩中的零值被遮蔽
    dense = make_tensor(r, c, dtype=dtype, device=device)
    dense[dense == 0] = 1

    # 根据逆向和数值遮罩，创建稀疏张量的逆向和数值部分
    dense_inv = dense.masked_fill(~mask_inv, 0)
    dense_val = dense_inv.masked_fill(~mask_val, 0)

    # 返回稀疏张量的逆向部分和数值部分
    return dense_inv, dense_val


class SparseSemiStructuredTensorCompileTest(torch._dynamo.test_case.TestCase):

    def setUp(self):
        # 如果不在SM80架构下运行，则跳过测试
        if not _IS_SM8X:
            self.skipTest('Only runs on SM80')
        super().setUp()

    def tearDown(self):
        super().tearDown()

    @staticmethod
    def _test_mlp_contiguous_relu_compile(backend, dense_input_shape):
        """
        Test nn.Linear + .contiguous() + nn.ReLU with SparseSemiStructuredTensor + torch.compile
        We expect:
            (1) The sparse tensor subclass should turn nn.Linear into `aten._structured_sparse_addmm` + `aten.contiguous()`
            (2) Inductor should fuse the .contiguous() call into the relu
        """
        
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 128)  # 创建一个线性层，输入大小为128，输出大小为128

            def forward(self, x):
                x = self.linear(x)  # 将输入x通过线性层
                x = x.contiguous()  # 调用contiguous方法，保证张量在内存中是连续的
                return torch.nn.functional.relu(x)  # 对x应用ReLU激活函数

        input = torch.rand(dense_input_shape, device="cuda").half()  # 生成一个随机张量作为输入，放置在GPU上，数据类型为半精度浮点数
        model = Model().eval().cuda().half()  # 创建一个模型实例，设为评估模式，放置在GPU上，数据类型为半精度浮点数
        mod_linear = model.linear  # 获取模型中的线性层
        m, n = mod_linear.weight.shape  # 获取线性层权重的形状
        mask = torch.Tensor([1, 0, 0, 1]).tile((m, n // 4)).bool().cuda()  # 创建一个布尔类型的掩码张量，放置在GPU上，用来修改权重
        # 设置掩码后的权重
        mod_linear.weight = nn.Parameter(mod_linear.weight * mask)

        dense_result = model(input)  # 在稠密输入上运行模型
        mod_linear.weight = nn.Parameter(SEMI_STRUCTURED_SUPPORTED_BACKENDS[backend].from_dense(mod_linear.weight))  # 将稠密权重转换为稀疏形式
        sparse_result = model(input)  # 在稀疏输入上运行模型

        model = torch.compile(model, backend="inductor", fullgraph=True)  # 使用torch.compile编译模型，选择inductor后端，并使用完整图形模式
        sparse_compile_result = model(input)  # 在编译后的模型上运行稀疏输入

        # 测试编译后的稀疏结果和稠密结果在数值上是否接近
        torch.testing.assert_close(dense_result, sparse_compile_result, rtol=1e-3, atol=1e-3)
        # 断言稀疏结果和编译后的稀疏结果具有相同的步幅，
        # 因为元注册可能在输出转置时返回连续张量
        # https://github.com/pytorch/pytorch/pull/114477
        assert sparse_result.stride() == sparse_compile_result.stride()

    @unittest.skipIf(IS_WINDOWS, "torch.compile not supported on windows")
    @unittest.skipIf("cusparselt" not in SEMI_STRUCTURED_SUPPORTED_BACKENDS, "cusparselt not supported on this machine")
    def test_mlp_contiguous_relu_compile_cusparselt(self):
        """
        test for cuSPASRELt meta registrations (_cslt_sparse_mm) + torch.compile
        """
        for dense_input_shape in [(1, 128), (64, 128), (128, 128), (64, 128, 128)]:
            SparseSemiStructuredTensorCompileTest._test_mlp_contiguous_relu_compile("cusparselt", dense_input_shape)

    @unittest.skipIf(IS_WINDOWS, "torch.compile not supported on windows")
    def test_mlp_contiguous_relu_compile_cutlass(self):
        """
        test for CUTLASS meta registrations (_sparse_semi_structured_addmm) + torch.compile
        """
        for dense_input_shape in [(1, 128), (64, 128), (128, 128), (64, 128, 128)]:
            SparseSemiStructuredTensorCompileTest._test_mlp_contiguous_relu_compile("cutlass", dense_input_shape)


    @unittest.skipIf(IS_WINDOWS, "torch.compile not supported on windows")
    # 如果在当前机器上不支持"cusparselt"，则跳过此测试用例
    @unittest.skipIf("cusparselt" not in SEMI_STRUCTURED_SUPPORTED_BACKENDS, "cusparselt not supported on this machine")
    # 定义名为test_sp24_compile的测试函数，无返回值(None)
    def test_sp24_compile(self) -> None:
        # 在CUDA设备上生成一个大小为[1024, 512]的随机张量x，数据类型为torch.float16，需要梯度计算
        x = torch.randn([1024, 512], device="cuda", dtype=torch.float16, requires_grad=True)
        # 在CUDA设备上生成一个单位矩阵，大小与x相同，数据类型为torch.float16
        e = torch.eye(x.shape[0], x.shape[0], device="cuda", dtype=torch.float16)

        # 定义一个内部函数fn，接受参数x和e
        def fn(x, e):
            # 对稀疏半结构化张量使用CUSPARSELT算法进行静态排序和修剪
            y = SparseSemiStructuredTensorCUSPARSELT.prune_dense_static_sort(x)
            # 对y进行转置操作
            y = y.t()
            # 返回x与y的矩阵乘法结果
            return x @ y

        # 在"eager"模式下计算输出
        output = fn(x, e)
        # 对输出进行反向传播
        output.backward(output)
        
        # 使用Torch的编译功能对函数fn进行编译
        output = torch.compile(fn)(x, e)
        # 对编译后的输出进行反向传播
        output.backward(output)
    # 定义测试类 TestSparseSemiStructured，继承自 TestCase
class TestSparseSemiStructured(TestCase):

    # 在每个测试方法执行前设置预备条件
    def setUp(self):
        # 如果不是 SM80 架构，跳过测试
        if not _IS_SM8X:
            self.skipTest('Only runs on SM80')
        # 如果运行环境是 Windows，跳过测试（因为不支持在 Windows 上运行 torch.compile）
        if IS_WINDOWS:
            self.skipTest("torch.compile not supported on windows")

    # 标记为推断数据类型的测试方法，同时使用 @parametrize_backends 装饰器
    @inference_dtypes
    @parametrize_backends
    def test_to_sparse_semi_structured(self, dtype, backend):
        # 根据后端是否为 "cutlass"，设置 SparseSemiStructuredTensor 的强制 Cutlass 标志
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        # 生成一个随机稀疏半结构化掩码 A，数据类型为 dtype
        A = rand_sparse_semi_structured_mask(128, 256, dtype=dtype)
        # 将 A 转换为稀疏半结构化张量 A_sparse
        A_sparse = to_sparse_semi_structured(A)

        # 断言 A 和 A_sparse 的形状相同
        assert A.shape == A_sparse.shape
        # 断言 A 和 A_sparse 的设备（GPU 或 CPU）相同
        assert A.device == A_sparse.device
        # 断言 A 和 A_sparse 的数据类型相同
        assert A.dtype == A_sparse.dtype

        # 断言 A 是 torch.Tensor 类型
        assert isinstance(A, torch.Tensor)
        # 断言 A_sparse 是 SparseSemiStructuredTensor 类型
        assert isinstance(A_sparse, SparseSemiStructuredTensor)

    # 标记为推断数据类型的测试方法，同时使用 @parametrize_backends 和 @parametrize 装饰器
    @inference_dtypes
    @parametrize_backends
    @parametrize("dense_input_shape", [(128, 1), (128, 64), (128, 128)])
    def test_mm_sparse_first_NN(self, dense_input_shape, dtype, device, backend):
        """
        确保 torch.mm(A_sparse, B) 在 float16 下正确，并且在 int8 下会抛出错误
        """
        # 根据后端是否为 "cutlass"，设置 SparseSemiStructuredTensor 的强制 Cutlass 标志
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        # 生成一个随机稀疏半结构化掩码 A，数据类型为 dtype
        A = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        # 将 A 转换为稀疏半结构化张量 A_sparse
        A_sparse = to_sparse_semi_structured(A)

        # 生成一个随机张量 B，形状为 dense_input_shape，在 A_sparse 的设备上
        B = torch.rand(dense_input_shape, device=A_sparse.device).to(dtype)

        # 如果数据类型是 torch.int8
        if dtype is torch.int8:
            # 如果后端为 "cutlass"，断言在执行 torch.mm(A_sparse, B) 时会抛出特定的 RuntimeError
            if backend == "cutlass":
                with self.assertRaisesRegex(RuntimeError, "spgemm_cutlass_dispatch_layouts"):
                    sparse_result = torch.mm(A_sparse, B)
            else:
                # 否则，在执行 torch.mm(A_sparse, B) 时会抛出特定的 RuntimeError
                with self.assertRaisesRegex(RuntimeError,
                                            "CUDA error: operation not supported when calling `cusparseLtMatmulDescriptorInit"):
                    sparse_result = torch.mm(A_sparse, B)
        else:
            # 如果数据类型不是 torch.int8，计算稠密结果 dense_result
            dense_result = torch.mm(A, B)
            # 计算稀疏结果 sparse_result
            sparse_result = torch.mm(A_sparse, B)
            # 使用 torch.testing.assert_close 断言 dense_result 和 sparse_result 在指定的误差范围内相等
            torch.testing.assert_close(dense_result, sparse_result, rtol=1e-3, atol=1e-3)
    # 测试稀疏矩阵与转置稠密矩阵的乘积在特定条件下的正确性和错误处理
    def test_mm_sparse_first_NT(self, dense_input_shape, dtype, device, backend):
        """
        Ensure torch.mm(A_sparse, B.t()) is correct for float16/bfloat16
        and will throw an error for int8 + padding
        """
        # 设置强制使用 cutlass 后端
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        # 生成指定形状和数据类型的随机稀疏半结构化张量 A
        A = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        # 将 A 转换为稀疏半结构化张量
        A_sparse = to_sparse_semi_structured(A)

        # 生成指定形状和设备的随机稠密张量 B
        B = torch.rand(dense_input_shape, device=A_sparse.device).to(dtype)

        # 如果数据类型为 torch.int8 且 dense_input_shape 为 {(1, 128)}
        if dtype is torch.int8 and dense_input_shape in {(1, 128)}:
            # 当使用 cutlass 后端时，期望抛出特定错误信息
            if backend == "cutlass":
                with self.assertRaisesRegex(RuntimeError, "spgemm_cutlass_dispatch_layouts"):
                    # 执行稀疏矩阵 A_sparse 与 B 转置的矩阵乘积
                    sparse_result = torch.mm(A_sparse, B.t())
            else:
                with self.assertRaisesRegex(RuntimeError,
                                            "CUDA error: operation not supported when calling `cusparseLtMatmulDescriptorInit`"):
                    # 执行稀疏矩阵 A_sparse 与 B 转置的矩阵乘积
                    sparse_result = torch.mm(A_sparse, B.t())
        elif dtype is torch.int8:
            # 对转置后的稠密矩阵 B 执行乘法，并转换到指定设备和数据类型 torch.int8
            dense_result = torch.mm(A.cpu(), B.t().cpu()).to(device, dtype=torch.int8)
            # 执行稀疏矩阵 A_sparse 与 B 转置的矩阵乘积
            sparse_result = torch.mm(A_sparse, B.t())
            # 检验稠密结果和稀疏结果的接近程度
            torch.testing.assert_close(dense_result, sparse_result, rtol=1e-3, atol=1e-3)
        else:
            # 对转置后的稠密矩阵 B 执行乘法
            dense_result = torch.mm(A, B.t())
            # 执行稀疏矩阵 A_sparse 与 B 转置的矩阵乘积
            sparse_result = torch.mm(A_sparse, B.t())
            # 检验稠密结果和稀疏结果的接近程度
            torch.testing.assert_close(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    @inference_dtypes
    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128)])
    @parametrize_backends
    # 测试稀疏矩阵转置与稠密矩阵乘积的操作在特定条件下的错误处理
    def test_mm_sparse_first_TN(self, dtype, dense_input_shape, device, backend):
        """
        Ensure torch.mm(A_sparse.t(), B) throws error
        """
        # 设置强制使用 cutlass 后端
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        # 如果使用 cutlass 后端并且在 Windows 系统下，跳过测试
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        # 生成指定形状和数据类型的随机稀疏半结构化张量 A
        A = rand_sparse_semi_structured_mask(128, 256, dtype=dtype)
        # 将 A 转换为稀疏半结构化张量
        A_sparse = to_sparse_semi_structured(A)

        # 生成指定形状和设备的随机稠密张量 B
        B = torch.rand(dense_input_shape, device=A_sparse.device).to(dtype)

        # 期望抛出 NotImplementedError 错误，指明稀疏矩阵转置与稠密矩阵 B 的乘积操作不受支持
        with self.assertRaisesRegex(
            NotImplementedError,
            r"`SparseSemiStructuredTensor.*` matmul: operation is not supported",
        ):
            torch.mm(A_sparse.t(), B)
    def test_mm_sparse_second_NT(self, dense_input_shape, dtype, device, backend):
        """
        Ensure torch.mm(A, B_sparse.t()) is correct
        """
        # 设置是否强制使用 Cutlass 后端（仅限于 backend 为 "cutlass" 时）
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        # 如果使用 Cutlass 后端且在 Windows 上运行，则跳过测试
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        # 生成一个随机的稀疏半结构化 B 张量
        B = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        # 将 B 转换为稀疏半结构化张量 B_sparse
        B_sparse = to_sparse_semi_structured(B)

        # 生成一个随机的 dense_input_shape 形状的 A 张量
        A = torch.rand(dense_input_shape, device=B_sparse.device).to(dtype)

        # 如果 dtype 是 torch.int8，则当前不支持在 GPU 上进行整数矩阵乘法，需要在 CPU 上进行评估并复制到指定设备和 dtype
        if dtype is torch.int8:
            dense_result = torch.mm(A.cpu(), B.t().cpu()).to(device, dtype=torch.int8)
            sparse_result = torch.mm(A, B_sparse.t())
        else:
            dense_result = torch.mm(A, B.t())
            sparse_result = torch.mm(A, B_sparse.t())

        # 断言 dense_result 和 sparse_result 在相对误差（rtol）和绝对误差（atol）范围内近似相等
        torch.testing.assert_close(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    @inference_dtypes
    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128)])
    @parametrize_backends
    def test_mm_sparse_second_NN(self, dense_input_shape, dtype, device, backend):
        """
        Ensure torch.mm(A, B_sparse) throws error
        """
        # 设置是否强制使用 Cutlass 后端（仅限于 backend 为 "cutlass" 时）
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        # 如果使用 Cutlass 后端且在 Windows 上运行，则跳过测试
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        # 生成一个随机的稀疏半结构化 B 张量
        B = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        # 将 B 转换为稀疏半结构化张量 B_sparse
        B_sparse = to_sparse_semi_structured(B)

        # 生成一个随机的 dense_input_shape 形状的 A 张量
        A = torch.rand(dense_input_shape, device=B_sparse.device).to(dtype)

        # 使用断言检查 torch.mm(A, B_sparse) 是否抛出 NotImplementedError 异常，并包含指定的错误消息
        with self.assertRaisesRegex(
            NotImplementedError,
            r"`SparseSemiStructuredTensor.*` matmul: operation is not supported",
        ):
            sparse_result = torch.mm(A, B_sparse)

    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128), (64, 128, 128)])
    @parametrize("inference_mode", [subtest(True), subtest(False)])
    @parametrize_backends
    # 定义一个测试方法，用于验证 nn.Linear 的数值计算结果是否一致
    def test_linear(self, dense_input_shape, inference_mode, device, backend):
        """
        Test nn.Linear has the same numerics
        """
        # 根据后端选择是否使用 CUTLASS 库加速，仅在后端为 "cutlass" 并且不是在 Windows 平台时才设置为强制使用 CUTLASS
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        if backend == "cutlass" and IS_WINDOWS:
            # 如果后端为 "cutlass" 且在 Windows 平台上，跳过测试并返回
            self.skipTest("CUTLASS not supported on Windows")
        
        # 生成一个随机的半精度浮点数输入张量
        input = torch.rand((dense_input_shape), device=device).half()
        # 创建一个包含两层的线性模型，并将其转移到指定的设备上，并设置为半精度浮点数
        model = nn.Linear(128, 256).to(device).half()
        
        # 获取模型权重的形状 m 和 n
        m, n = model.weight.shape
        # 生成一个随机稀疏半结构化的掩码张量，用于权重稀疏化，数据类型为布尔型
        mask = rand_sparse_semi_structured_mask(m, n, device=device, dtype=torch.bool)
        # 设置被掩码后的权重
        model.weight = nn.Parameter(model.weight * mask)

        # 计算密集模式下的模型输出结果
        dense_result = model(input)

        # 将权重转换为稀疏半结构化表示
        model.weight = nn.Parameter(to_sparse_semi_structured(model.weight))

        # 根据推断模式选择计算稀疏模式或密集模式的模型输出结果
        if inference_mode:
            # 进入推断模式上下文，计算稀疏模式的模型输出结果
            with torch.inference_mode():
                sparse_result = model(input)
        else:
            # 计算稀疏模式的模型输出结果
            sparse_result = model(input)

        # 使用 torch.testing.assert_close 方法验证密集模式和稀疏模式的输出结果在指定的相对和绝对容差范围内是否一致
        torch.testing.assert_close(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    # 为多组输入形状和后端参数进行参数化测试
    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128), (64, 128, 128)])
    @parametrize_backends
    def test_mlp(self, device, dense_input_shape, backend):
        # 根据后端选择是否使用 CUTLASS 库加速
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        # 生成一个随机的半精度浮点数输入张量
        input = torch.rand(dense_input_shape, device=device).half()
        # 创建一个包含两层线性模型的序列模型，并将其转移到指定的设备上，并设置为半精度浮点数
        model = (
            nn.Sequential(
                nn.Linear(128, 256),
                nn.Linear(256, 128),
            )
            .half()
            .to(device)
        )

        # 循环处理每一层模型
        for i in range(2):
            # 获取当前层模型权重的形状 m 和 n
            m, n = model[i].weight.shape
            # 生成一个随机稀疏半结构化的掩码张量，用于权重稀疏化，数据类型为布尔型
            mask = rand_sparse_semi_structured_mask(
                m, n, device=device, dtype=torch.bool
            )
            # 设置被掩码后的权重
            model[i].weight = nn.Parameter(model[i].weight * mask)

        # 计算密集模式下的模型输出结果
        dense_result = model(input)

        # 循环处理每一层模型
        for i in range(2):
            # 将当前层模型的权重转换为稀疏半结构化表示
            model[i].weight = nn.Parameter(to_sparse_semi_structured(model[i].weight))

        # 计算稀疏模式的模型输出结果
        sparse_result = model(input)

        # 使用 torch.testing.assert_close 方法验证密集模式和稀疏模式的输出结果在指定的相对和绝对容差范围内是否一致
        torch.testing.assert_close(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    # 参数化测试后端类型
    @parametrize_backends
    def test_values(self, backend):
        # 根据后端选择是否使用 CUTLASS 库加速
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        # 如果后端为 "cutlass" 并且在 Windows 平台上，跳过测试并返回
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        
        # 生成一个随机稀疏半结构化的掩码张量
        A = rand_sparse_semi_structured_mask(128, 128)
        # 将稀疏半结构化的掩码张量转换为稀疏半结构化表示
        A_sparse = to_sparse_semi_structured(A)
        # 断言稀疏表示的值张量的形状是否为 (128, 64)
        assert A_sparse.values().shape == (128, 64)
        # 断言稀疏表示的值张量的所有元素是否均为 1
        assert (A_sparse.values() == 1).all()

    # 参数化测试后端类型
    @parametrize_backends
    def test_indices(self, backend):
        # 根据后端选择是否使用 CUTLASS 库加速
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        # 如果后端为 "cutlass" 并且在 Windows 平台上，跳过测试并返回
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        
        # 生成一个随机稀疏半结构化的掩码张量
        A = rand_sparse_semi_structured_mask(128, 128)
        # 将稀疏半结构化的掩码张量转换为稀疏半结构化表示
        A_sparse = to_sparse_semi_structured(A)
        # 断言稀疏表示的索引张量的形状是否为 (128, 8)
        assert A_sparse.indices().shape == (128, 8)

    # 标记一个推断数据类型的装饰器
    @inference_dtypes
    @parametrize_backends
    def test_min_sparse_shape(self, dtype, device, backend):
        # 设置全局变量，根据后端是否为"cutlass"来决定是否强制使用Cutlass库
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        # 获取当前后端的数据类型和形状约束配置
        config = SEMI_STRUCTURED_SUPPORTED_BACKENDS[backend]._DTYPE_SHAPE_CONSTRAINTS[dtype]
        # 创建稀疏半结构化张量 A
        A = rand_sparse_semi_structured_mask(config.sparse_min_rows, config.sparse_min_cols, dtype=dtype, device=device)
        # 将稀疏半结构化张量 A 转换为稀疏格式
        A_sparse = to_sparse_semi_structured(A)
        # 创建稠密张量 B
        B = torch.rand((config.sparse_min_cols, config.dense_min_cols), device=device).to(dtype)
        if dtype == torch.int8:
            # 对于 int8 类型，执行稠密矩阵乘法，并转换结果到设备和 int8 类型
            dense_res = torch.mm(A.cpu(), B.cpu()).to(device, dtype=torch.int8)
            # 对于 R/R -> R 布局，不支持 int8 稀疏矩阵乘法，因此需要转置一个参数以获得 R/C -> R 布局
            B_t = B.t().contiguous()
            sparse_res = torch.mm(A_sparse, B_t.t())
        else:
            # 对于其他数据类型，执行稠密矩阵乘法
            dense_res = torch.mm(A, B)
            sparse_res = torch.mm(A_sparse, B)
        # 使用测试框架断言稀疏矩阵乘法结果与稠密矩阵乘法结果的接近程度
        torch.testing.assert_close(sparse_res, dense_res, rtol=1e-3, atol=1e-3)
    
    @inference_dtypes
    @parametrize_backends
    def test_unsupported_shape(self, dtype, device, backend):
        # 设置全局变量，根据后端是否为"cutlass"来决定是否强制使用Cutlass库
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        # 如果后端为 cutlass 并且在 Windows 上运行，则跳过测试
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        # 创建形状不受支持的稀疏半结构化张量 A
        A = rand_sparse_semi_structured_mask(2, 2, dtype=dtype, device=device)
        # 使用断言检查异常是否包含特定字符串
        with self.assertRaisesRegex(RuntimeError, "Error original_tensor.shape"):
            A_sparse = to_sparse_semi_structured(A)
    
    @dtypes(*all_types_and_complex())
    @parametrize_backends
    def test_unsupported_dtype(self, dtype, device, backend):
        # 设置全局变量，根据后端是否为"cutlass"来决定是否强制使用Cutlass库
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        # 如果后端为 cutlass 并且在 Windows 上运行，则跳过测试
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        # 创建不受支持数据类型的稀疏半结构化张量 A
        A = rand_sparse_semi_structured_mask(128, 128, dtype=dtype, device=device)
        # 如果数据类型不在支持列表中，使用断言检查异常是否包含特定字符串
        if dtype not in SEMI_STRUCTURED_SUPPORTED_DTYPES:
            with self.assertRaisesRegex(RuntimeError, "Error original_tensor.dtype"):
                A_sparse = to_sparse_semi_structured(A)
        else:
            # 否则，正常执行转换为稀疏格式
            A_sparse = to_sparse_semi_structured(A)
    
    @parametrize_backends
    def test_unsupported_dim(self, device, backend):
        # 设置全局变量，根据后端是否为"cutlass"来决定是否强制使用Cutlass库
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        # 如果后端为 cutlass 并且在 Windows 上运行，则跳过测试
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        # 创建维度不受支持的张量 A
        A = torch.rand(128, 128, 128, device=device, dtype=torch.float16)
        # 使用断言检查异常是否包含特定字符串
        with self.assertRaisesRegex(RuntimeError, "Error original_tensor.dim"):
            A_sparse = to_sparse_semi_structured(A)
# 定义一个函数，生成指定形状的随机布尔掩码张量
def create_random_mask(shape) -> torch.Tensor:
    # 使用种子值 0 创建随机数生成器对象
    r = random.Random(0)
    # 创建一个形状为 shape 的零张量，数据类型为布尔型
    mask = torch.zeros(shape, dtype=torch.bool)
    # 遍历掩码张量的每一行
    for line in range(mask.shape[0]):
        # 遍历每行中以步长为 4 的列
        for col in range(0, mask.shape[1], 4):
            # 随机选择一个稀疏度模式
            sparsity = r.choice(
                [
                    [False, False, True, True],
                    [False, True, False, True],
                    [True, False, False, True],
                    [False, True, True, False],
                    [True, False, True, False],
                    [True, True, False, False],
                ]
            )
            # 将选择的稀疏度模式作为布尔张量赋值给掩码张量的一部分
            mask[line, col : col + 4] = torch.tensor(sparsity, dtype=torch.bool)
    # 返回生成的掩码张量
    return mask

class TestSparseSemiStructuredTraining(TestCase):

    def setUp(self):
        # 如果不是 SM80 架构，则跳过测试
        if not _IS_SM8X:
            self.skipTest('Only runs on SM80')
        # 如果是 Windows 系统，则跳过测试
        if IS_WINDOWS:
            self.skipTest('CUTLASS not supported on windows')

    # 下面应该有更多的测试方法或者测试装饰器，但在提供的代码片段中被省略了
    # 定义一个测试方法，用于测试稀疏半结构化张量的静态剪枝和排序功能，接受一个数据类型参数 dtype
    def test_prune_dense_static_sort(self, dtype) -> None:
        # 理想情况下，我们希望克隆并比较，但由于排序顺序不同，这种方法不适用
        # 取而代之的是将剪枝后的矩阵传递给 CUDA 实现，并保留稀疏模式
        dense = torch.randn(128, 128, device="cuda", dtype=dtype)
        
        # 对 dense 进行半结构化稀疏矩阵的剪枝处理
        pruned = _sparse_semi_structured_tile(dense)

        # 使用 CUTLASS 库进行静态排序剪枝
        reference_cutlass = SparseSemiStructuredTensorCUTLASS.prune_dense_static_sort(pruned, algorithm="largest_abs_values_greedy")
        # 断言 pruned 和 reference_cutlass 转换为稠密张量后的结果相似
        torch.testing.assert_close(pruned, reference_cutlass.to_dense())

        # 使用 CUTLASS 库将 pruned 转换为压缩格式的稀疏数据和元数据
        packed_cutlass, meta_cutlass = sparse_semi_structured_from_dense_cutlass(pruned)
        packed_t_cutlass, meta_t_cutlass = sparse_semi_structured_from_dense_cutlass(pruned.t().contiguous())
        
        # 将 meta_cutlass 和 meta_t_cutlass 调整为与 reference_cutlass 的形状和步幅相同
        meta_cutlass = meta_cutlass.as_strided(reference_cutlass.meta.shape, reference_cutlass.meta.stride())
        meta_t_cutlass = meta_t_cutlass.as_strided(reference_cutlass.meta_t.shape, reference_cutlass.meta_t.stride())
        
        # 计算压缩并重排的位掩码
        compressed_swizzled_bitmask = _compute_compressed_swizzled_bitmask(pruned)
        # 将位掩码调整为与 reference_cutlass 的压缩重排位掩码相同的形状和步幅
        compressed_swizzled_bitmask = compressed_swizzled_bitmask.as_strided(reference_cutlass.compressed_swizzled_bitmask.shape,
                                                                             reference_cutlass.compressed_swizzled_bitmask.stride())
        
        # 使用 CUTLASS 库创建 SparseSemiStructuredTensorCUTLASS 对象
        cutlass = SparseSemiStructuredTensorCUTLASS(dense.shape,
                                                    packed_cutlass,
                                                    meta_cutlass,
                                                    packed_t_cutlass,
                                                    meta_t_cutlass,
                                                    compressed_swizzled_bitmask)
        # 断言 reference_cutlass 和 cutlass 转换为稠密张量后的结果相似
        torch.testing.assert_close(reference_cutlass.to_dense(), cutlass.to_dense())

        # 使用 CUSPARSELT 库进行静态排序剪枝
        reference_cusparselt = SparseSemiStructuredTensorCUSPARSELT.prune_dense_static_sort(pruned,
                                                                                            algorithm="largest_abs_values_greedy")
        # 断言 pruned 和 reference_cusparselt 转换为稠密张量后的结果相似
        torch.testing.assert_close(pruned, reference_cusparselt.to_dense())

        # 使用 CUSPARSELT 库将 pruned 转换为压缩格式的稀疏数据
        packed_cusparselt = torch._cslt_compress(pruned)
        packed_t_cusparselt = torch._cslt_compress(pruned.t().contiguous())
        
        # 使用 CUSPARSELT 库创建 SparseSemiStructuredTensorCUSPARSELT 对象
        cusparselt = SparseSemiStructuredTensorCUSPARSELT(dense.shape,
                                                          packed_cusparselt,
                                                          None,
                                                          packed_t_cusparselt,
                                                          None,
                                                          compressed_swizzled_bitmask)
        # 断言 reference_cusparselt 和 cusparselt 转换为稠密张量后的结果相似
        torch.testing.assert_close(reference_cusparselt.to_dense(), cusparselt.to_dense())
    def test_pruning_algo_largest_abs_values_greedy(self, dtype, backend) -> None:
        # 创建输入张量，包含四个子张量，设备为 CUDA，指定数据类型为 dtype
        inp = torch.tensor(
            [[4, 3, 2, 1], [-1, -3, 0.6, 0.5], [1, 2, 3, 4], [10, 2, -1, 5]],
            device="cuda",
            dtype=dtype,
        )
        # 使用常数填充 inp，使其维度为 (128, 128)，填充值为 1
        inp = F.pad(inp, (0, 128 - 4, 0, 128 - 4), "constant", 1)
        # 使用指定的 backend 对 inp 进行静态稀疏化，并选择使用 largest_abs_values_greedy 算法
        sInp = SEMI_STRUCTURED_SUPPORTED_BACKENDS[backend].prune_dense_static_sort(inp, algorithm="largest_abs_values_greedy")

        # 创建稀疏掩码，计算 sInp 与 inp 的比值
        mask = sInp.to_dense() / inp
        # 断言前4行前4列的整数化列表等于指定的值
        assert mask[:4, :4].int().tolist() == [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
        ]

    @training_dtypes
    def test_gemm(self, dtype) -> None:
        # 定义矩阵乘法的维度
        M, N, K = 32, 32, 64
        # 创建随机张量 a 和 b，设备为 CUDA，数据类型为 dtype
        a = torch.randn([M, K], device="cuda", dtype=dtype)
        b = torch.randn([K, N], device="cuda", dtype=dtype)
        # 创建布尔类型的随机稀疏半结构化掩码
        mask = rand_sparse_semi_structured_mask(M, K, dtype=torch.bool)

        # 使用掩码填充 a，将非掩码部分置零
        a.masked_fill_(~mask, 0)

        # 将 a 转换为稀疏半结构化张量
        a_sparse = to_sparse_semi_structured(a)

        # 使用普通矩阵乘法计算参考输出
        masked_a = a * mask
        ref_out = masked_a @ b
        # 使用稀疏半结构化张量计算 sp24_out
        sp24_out = a_sparse @ b
        # 断言参考输出与 sp24_out 的近似相等性，根据数据类型选择相对/绝对误差容忍度
        torch.testing.assert_close(ref_out, sp24_out, **atol_rtol_kw[dtype])


    @training_dtypes
    @parametrize_backends
    def test_pack_both_ways_meta_correctness(self, dtype, backend) -> None:
        # 定义矩阵的尺寸
        M, N = 128, 256
        # 创建矩阵 a，确保每个 4x4 块中有 8 个元素
        a = (4 * torch.arange(8))[:, None] + torch.arange(8)[None, :]
        a = a.repeat(M // 8, N // 8)
        # 断言矩阵 a 的形状为 (M, N)
        assert a.shape == (M, N)
        # 将矩阵 a 移动到 CUDA 设备，并转换为指定数据类型 dtype
        a = a.cuda().to(dtype)
        # 创建随机张量 b，设备为 CUDA，数据类型为 dtype
        b = torch.randn([a.shape[1], 128], device="cuda", dtype=dtype)

        # 使用指定的 backend 对 a 进行静态稀疏化
        a_sparse = SEMI_STRUCTURED_SUPPORTED_BACKENDS[backend].prune_dense_static_sort(a)

        # 生成稀疏半结构化的最大掩码
        mask_dense = sparse24_largest_mask_2d(a).to(dtype)

        # 如果 backend 为 "cutlass"，执行以下断言和操作
        if backend == "cutlass":
            assert isinstance(a_sparse, SparseSemiStructuredTensorCUTLASS)
            # 使用 cutlass 加速方法对 mask_dense 进行矩阵打包
            (packed, meta, packed_t, meta_t, bitmask) = torch._sparse_semi_structured_tile(
                mask_dense, use_cutlass=True)

            # 创建 SparseSemiStructuredTensorCUTLASS 对象 sparse_mask
            sparse_mask = SparseSemiStructuredTensorCUTLASS(
                mask_dense.shape,
                packed=packed,
                meta=meta,
                packed_t=packed_t,
                meta_t=meta_t,
                compressed_swizzled_bitmask=bitmask,
            )
            # 断言 a_sparse 的 meta 视图等于 sparse_mask 的 meta，转换为 torch.short 类型
            torch.testing.assert_close(a_sparse.meta.view(torch.short), sparse_mask.meta)

        # 使用掩码 a_sparse 计算参考 gemm 输出和打包 gemm 输出
        ref_gemm = (mask_dense * a) @ b
        pack_gemm = a_sparse @ b
        # 断言参考 gemm 输出与打包 gemm 输出的近似相等性，根据数据类型选择相对/绝对误差容忍度
        torch.testing.assert_close(ref_gemm, pack_gemm, **atol_rtol_kw[dtype])

    @training_dtypes
    def test_pack_both_ways_id(self, dtype) -> None:
        # 设置随机种子确保结果可重复性
        N = 512
        torch.manual_seed(0)
        # 在CUDA设备上生成随机张量 a 和单位矩阵 b，数据类型为 dtype
        a = torch.randn([N, N], dtype=dtype, device="cuda")
        b = torch.eye(N, dtype=dtype, device="cuda")

        # 使用 _sparse_semi_structured_tile 函数处理张量 a，返回结果的前四个元素
        packed, meta, packed_t, meta_t = torch._sparse_semi_structured_tile(a)[:4]
        # 启发式方法确保打包后的值相同
        torch.testing.assert_close(
            packed.to(torch.float64).sum(), packed_t.to(torch.float64).sum()
        )

        # 计算稀疏掩码以用于稀疏矩阵乘法
        mask_dense = sparse24_largest_mask_2d(a.to(dtype))

        # 计算参考的稀疏 GEMM 运算结果
        ref_gemm = mask_dense * a
        # 测试 A@B 的结果
        pack_gemm = torch._sparse_semi_structured_linear(b.t(), packed, meta).t()
        # 计算最大差异的位置
        max_diff = (ref_gemm - pack_gemm).abs().argmax()
        torch.testing.assert_close(
            ref_gemm, pack_gemm,
            **atol_rtol_kw[dtype]
        ), f"packed is wrong at pos: ({max_diff // N}, {max_diff % N})"
        
        # 测试 A.t@B 的结果
        pack_gemm = torch._sparse_semi_structured_linear(b.t(), packed_t, meta_t)
        # 计算最大差异的位置
        max_diff = (ref_gemm - pack_gemm).abs().argmax()
        torch.testing.assert_close(
            ref_gemm, pack_gemm,
            **atol_rtol_kw[dtype]
        ), f"packed_t is wrong at pos: ({max_diff // N}, {max_diff % N})"

    @training_dtypes
    def test_pack_both_ways_edge_case1(self, dtype) -> None:
        # 在这种情况下，启发式方法将保留16个值中的7个，而不是8个，测试内核如何处理此情况
        quad = torch.tensor(
            [
                [2, -1, -2, -3],  # 应被打包为 `2 <null>`
                [-1, 8, -1, 6],
                [-1, -1, 4, 5],
                [-1, 3, 7, -1],
            ],
            dtype=dtype,
            device="cuda",
        )
        a = torch.randn([32, 64], dtype=dtype, device="cuda")
        a[:4, :4] = quad
        # 使用 _sparse_semi_structured_tile 函数处理张量 a，返回结果的前四个元素
        packed, meta, packed_t, meta_t = torch._sparse_semi_structured_tile(a)[:4]
        # 检查 A 的第一行
        assert packed[0, 0].item() == 2
        assert packed[0, 1].item() == 0
        # 检查 A.t 的第一列
        assert packed_t[0, 0].item() == 2
        assert packed_t[0, 1].item() == 0

    @training_dtypes
    def test_sp24_apply(self, dtype) -> None:
        # 定义张量 x 的形状 MxN，数据类型为 dtype，在 CUDA 设备上生成随机数据
        M, N = 256, 1024
        x = torch.randn([M, N], dtype=dtype, device="cuda")
        # 使用 _sparse_semi_structured_tile 函数处理张量 x，返回结果的前五个元素
        (
            packed,
            meta,
            packed_t,
            meta_t,
            bitmask,
        ) = torch._sparse_semi_structured_tile(x)
        # 使用 _sparse_semi_structured_apply 函数应用稀疏掩码并返回结果
        packed2, packed_t2 = torch._sparse_semi_structured_apply(x, bitmask)
        # 断言两次处理的结果应该相等
        torch.testing.assert_close(packed, packed2)
        torch.testing.assert_close(packed_t, packed_t2)
    # 定义一个测试函数，用于测试稀疏半结构化张量的稠密化操作
    def test_sp24_apply_dense(self, dtype) -> None:
        # 设置矩阵维度 M=256, N=1024
        M, N = 256, 1024
        # 生成一个在CUDA设备上的随机张量 x，数据类型为指定的 dtype
        x = torch.randn([M, N], dtype=dtype, device="cuda")
        # 调用 torch._sparse_semi_structured_tile 函数，获取稀疏半结构化张量的相关数据
        (
            packed,
            meta,
            packed_t,
            meta_t,
            bitmask,
        ) = torch._sparse_semi_structured_tile(x)

        # 创建 SparseSemiStructuredTensorCUTLASS 对象，将稀疏半结构化张量转换为密集张量，并期望值保存在 expected 中
        expected = SparseSemiStructuredTensorCUTLASS(
            x.shape,
            packed=packed,
            meta=meta,
            packed_t=packed_t,
            meta_t=meta_t,
            compressed_swizzled_bitmask=bitmask,
        ).to_dense()

        # 调用 torch._sparse_semi_structured_apply 函数，将稀疏半结构化张量应用于指定的 bitmask，得到 packed2 和 packed_t2
        packed2, packed_t2 = torch._sparse_semi_structured_apply(x, bitmask)
        # 创建 SparseSemiStructuredTensorCUTLASS 对象，用 packed2 和 packed_t2 初始化
        sparse = SparseSemiStructuredTensorCUTLASS(
            x.shape,
            packed=packed2,
            meta=meta,
            packed_t=packed_t2,
            meta_t=meta_t,
            compressed_swizzled_bitmask=bitmask,
        )

        # 调用 torch._sparse_semi_structured_apply_dense 函数，将稀疏半结构化张量应用于指定的 bitmask，得到 dense
        dense = torch._sparse_semi_structured_apply_dense(x, bitmask)

        # 断言 dense 与 expected 在数值上的接近程度
        torch.testing.assert_close(dense, expected)
        # 断言 sparse 转换为密集张量后与 expected 在数值上的接近程度
        torch.testing.assert_close(sparse.to_dense(), expected)


    # 使用 @training_dtypes 装饰器定义一个测试函数，用于测试稀疏半结构化张量的矩阵乘法操作
    def test_sp24_matmuls(self, dtype) -> None:
        # 设置矩阵维度 M=64, N=256, K=1024
        M, N, K = 64, 256, 1024
        # 生成两个在CUDA设备上的随机张量 a 和 b，数据类型为指定的 dtype
        a = torch.randn([M, K], device="cuda", dtype=dtype)
        b = torch.randn([K, N], device="cuda", dtype=dtype)
        # 使用 sparse24_largest_mask_2d 函数获取 a 和 b 的最大稀疏掩码
        a_m = sparse24_largest_mask_2d(a)
        b_m = sparse24_largest_mask_2d(b)
        # 调用 torch._sparse_semi_structured_tile 函数，获取张量 a 的稀疏半结构化数据
        (packed, meta, packed_t, meta_t, bitmask) = torch._sparse_semi_structured_tile(a)
        # 创建 SparseSemiStructuredTensorCUTLASS 对象，用于张量 a 的稀疏半结构化表示
        a_s = SparseSemiStructuredTensorCUTLASS(
            a.shape,
            packed=packed,
            meta=meta,
            packed_t=packed_t,
            meta_t=meta_t,
            compressed_swizzled_bitmask=bitmask,
        )
        # 调用 torch._sparse_semi_structured_tile 函数，获取张量 b 的稀疏半结构化数据
        (packed, meta, packed_t, meta_t, bitmask) = torch._sparse_semi_structured_tile(b)
        # 创建 SparseSemiStructuredTensorCUTLASS 对象，用于张量 b 的稀疏半结构化表示
        b_s = SparseSemiStructuredTensorCUTLASS(
            b.shape,
            packed=packed,
            meta=meta,
            packed_t=packed_t,
            meta_t=meta_t,
            compressed_swizzled_bitmask=bitmask,
        )

        # 断言 a_s 与 b 在稀疏半结构化张量乘法后与 a 和 b 乘法结果的接近程度
        torch.testing.assert_close(a_s @ b, (a * a_m) @ b, rtol=1e-1, atol=1.5e-1)
        # 断言 a 与 b_s 在稀疏半结构化张量乘法后与 a 和 b 乘法结果的接近程度
        torch.testing.assert_close(a @ b_s, a @ (b * b_m), rtol=1e-1, atol=1.5e-1)
        # 断言 a 与 a_s 转置乘积与 a 和 a 转置乘积结果的接近程度
        torch.testing.assert_close(
            a @ a_s.t(), a @ (a * a_m).t(), rtol=1e-1, atol=1.5e-1
        )
        # 断言 a_s 转置与 a 乘积与 a 转置乘积结果的接近程度
        torch.testing.assert_close(
            a_s.t() @ a, (a * a_m).t() @ a, rtol=1e-1, atol=1e-1
        )

    # 定义一个测试函数，用于测试稀疏半结构化张量的矩阵向量乘法操作
    def test_sp24_matmuls_mat_vec(self) -> None:
        # 生成在CUDA设备上的随机张量 a 和 b，数据类型为 torch.float16
        a = torch.randn([64, 128], device="cuda", dtype=torch.float16)
        b = torch.randn([128], device="cuda", dtype=torch.float16)
        # 使用 sparse24_largest_mask_2d 函数获取 a 的最大稀疏掩码
        a_m = sparse24_largest_mask_2d(a)
        # 将张量 a 转换为稀疏半结构化张量
        a_s = to_sparse_semi_structured(a)

        # 使用 pytest.raises 检测是否抛出 NotImplementedError 异常
        with pytest.raises(NotImplementedError):
            # 断言 a_s 与 b 在稀疏半结构化张量乘法后与 a 和 b 乘法结果的接近程度
            torch.testing.assert_close(a_s @ b, (a * a_m) @ b, **atol_rtol_kw[a.dtype])
    # 定义测试函数，用于测试稀疏矩阵乘法
    def test_sp24_matmuls_bmm(self) -> None:
        # 生成一个大小为 [64, 128] 的随机张量 a，存储在 GPU 上，数据类型为半精度浮点数
        a = torch.randn([64, 128], device="cuda", dtype=torch.float16)
        # 生成一个大小为 [5, 6, 128] 的随机张量 b，存储在 GPU 上，数据类型为半精度浮点数
        b = torch.randn([5, 6, 128], device="cuda", dtype=torch.float16)
        # 使用稀疏24算法生成张量 a 的最大值掩码
        a_m = sparse24_largest_mask_2d(a)
        # 将张量 a 转换为半结构化稀疏张量 a_s
        a_s = to_sparse_semi_structured(a)

        # 使用 pytest 断言捕获 NotImplementedError 异常
        with pytest.raises(NotImplementedError):
            # 断言稀疏张量 a_s 与张量 b 的矩阵乘积，与 (a * a_m) 与张量 b 的乘积在给定的容差范围内相等
            torch.testing.assert_close(a_s @ b, (a * a_m) @ b, **atol_rtol_kw[a.dtype])
class TestSparseSemiStructuredCUTLASS(TestCase):
    """
    This class defines unit tests specifically for CUTLASS operations in the context
    of torch._sparse_semi_structured_linear.
    """

    def setUp(self):
        # 检查是否为 SM80 架构，如果不是则跳过测试
        if not _IS_SM8X:
            self.skipTest('Only runs on SM80')
        
        # 检查是否已启用 CUTLASS 后端，如果没有则跳过测试
        if "cutlass" not in SEMI_STRUCTURED_SUPPORTED_BACKENDS:
            self.skipTest('CUTLASS not enabled')

    # 根据条件跳过测试：如果运行在 ROCm 或者 Windows 环境下，则跳过
    @unittest.skipIf(TEST_WITH_ROCM or IS_WINDOWS, "ROCm and Windows doesn't support CUTLASS")
    @inference_dtypes
    # 定义测试函数 test_linear_cutlass，接受设备和数据类型作为参数
    def test_linear_cutlass(self, device, dtype):

        # 定义内部函数 run_test，用于执行单个测试案例
        def run_test(batch_shape, m, n, k, device, dtype, dtype_out, add_bias, activation, rtol, atol):
            # 生成随机稀疏半结构化权重
            weight = rand_sparse_semi_structured(m, k, dtype, device)
            # 生成指定形状的输入张量
            input = make_tensor((*batch_shape, n, k), dtype=dtype, device=device)
            # 如果需要，生成偏置张量
            bias = make_tensor((m,), dtype=dtype_out, device=device) if add_bias else None

            # 将输入、权重和偏置张量转换为 torch.float32 类型的张量
            dtype_dense = torch.float32
            input_dense = input.to(dtype_dense)
            weight_dense = weight.to(dtype_dense)
            bias_dense = bias.to(dtype_dense) if add_bias else None

            # 执行 dense 矩阵乘法，并加上偏置，得到 output0
            output0 = torch.nn.functional.linear(input_dense, weight_dense, bias=bias_dense)

            # 根据激活函数类型，对 output0 应用激活函数
            if activation == "relu":
                relu = torch.nn.ReLU()
                output0 = relu(output0)
            elif activation == "silu":
                silu = torch.nn.SiLU()
                output0 = silu(output0)

            # 将稀疏半结构化的权重压缩为 compressed
            compressed = to_sparse_semi_structured(weight)
            # 提取稀疏权重和元数据
            weight_sparse = compressed.values()
            meta = compressed.indices()

            # 使用稀疏半结构化的权重和元数据执行稀疏乘法，得到 output1
            output1 = torch._sparse_semi_structured_linear(input, weight_sparse, meta, bias=bias, activation=activation,
                                                           out_dtype=dtype_out if dtype == torch.int8 else None)

            # 断言 output1 和 output0 在指定的误差范围内相等
            torch.testing.assert_close(output1.to(dtype_dense), output0, rtol=rtol, atol=atol)

        # 如果数据类型为 torch.float32
        if dtype == torch.float32:
            # 内部进行了 TF32 转换以便进行稀疏 GEMM，这里做 dense GEMM 以保证结果匹配
            orig = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True

        # 定义不同的批量形状、输出数据类型和激活函数
        batch_shapes = [[], [3], [3, 1]]
        dtype_out = {torch.int8: torch.int32, torch.half: torch.half, torch.bfloat16: torch.bfloat16, torch.float32: torch.float32}
        activations = [None, "relu", "silu"]
        rtol, atol = 1e-3, 1e-3
        # 根据数据类型调整比较的容差范围
        if dtype == torch.bfloat16:
            rtol, atol = 5e-3, 5e-3
        elif dtype == torch.float32:
            rtol, atol = 1e-3, 75e-2

        # 遍历所有可能的批量形状、m、n、k、是否添加偏置和激活函数的组合
        for batch_shape, m, n, k, add_bias, activation in \
                itertools.product(batch_shapes, range(3), range(3), range(3), (False, True), activations):
            # 如果激活函数是 "silu" 并且数据类型是 torch.int8，则跳过此组合
            if activation == "silu" and dtype == torch.int8:
                continue

            # 对 m、n、k 进行指数级的调整
            m = 2 ** m * 32
            n = 2 ** n * 32
            k = 2 ** k * 128
            # 运行单个测试案例
            run_test(batch_shape, m, n, k, device, dtype, dtype_out[dtype], add_bias, activation, rtol, atol)

        # 如果数据类型为 torch.float32，则恢复原始的 allow_tf32 设置
        if dtype == torch.float32:
            torch.backends.cuda.matmul.allow_tf32 = orig


    # 使用 unittest 装饰器，条件为 TEST_WITH_ROCM 或 IS_WINDOWS 为真，则跳过测试
    @unittest.skipIf(TEST_WITH_ROCM or IS_WINDOWS, "ROCm and Windows doesn't support CUTLASS")
    # 使用 parametrize 装饰器，指定 backend 参数为 "cutlass"
    @parametrize("backend", ["cutlass"])
    # 使用 dtypes 装饰器，参数为 SEMI_STRUCTURED_SUPPORTED_DTYPES 的所有元素
    @dtypes(*SEMI_STRUCTURED_SUPPORTED_DTYPES)
    def test_sparse_semi_structured_ops_cutlass(self, device, dtype, backend):
        # 设置一个标志，强制使用 CUTLASS 后端
        SparseSemiStructuredTensor._FORCE_CUTLASS = (backend == "cutlass")
        # 如果后端是 CUTLASS 且在 Windows 平台上，则跳过测试
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")

        # 定义一个运行测试的函数，参数包括矩阵维度和数据类型等
        def run_test(m, n, k, device, dtype, dtype_out, use_input, rtol, atol):
            # 创建稀疏半结构化张量 mat1
            mat1 = rand_sparse_semi_structured(m, k, dtype, device)
            # 创建 mat2 张量，并将其转置
            # 对于 int8 情况，仅支持行主序或列主序组合
            mat2 = make_tensor((n, k), dtype=dtype, device=device).t()
            # 根据 use_input 的值，创建输入张量 input
            input = make_tensor((m,), dtype=dtype_out, device=device) if use_input else None

            # 如果 use_input 为真，则设置 alpha 和 beta 的值
            if use_input:
                if dtype.is_floating_point:
                    alpha = 1.3
                    beta = -0.7
                else:
                    alpha = 2
                    beta = -3

            # 将 mat1 和 mat2 转换为 torch.float32 类型的稠密张量
            dtype_dense = torch.float32
            mat1_dense = mat1.to(dtype_dense)
            mat2_dense = mat2.to(dtype_dense)
            # 根据 use_input 的值选择执行 torch.mm 或 torch.addmm
            if not use_input:
                output0 = torch.mm(mat1_dense, mat2_dense)
            else:
                input_dense = input.to(dtype_dense)[:, None]
                output0 = torch.addmm(input_dense, mat1_dense, mat2_dense, alpha=alpha, beta=beta)

            # 将 mat1 转换为稀疏半结构化张量
            compressed = to_sparse_semi_structured(mat1)
            # 获取压缩后的 mat1 的值和索引信息
            mat1_sparse = compressed.values()
            mat1_meta = compressed.indices()

            # 根据 use_input 的值选择执行 torch._sparse_semi_structured_mm 或 torch._sparse_semi_structured_addmm
            if not use_input:
                output1 = torch._sparse_semi_structured_mm(mat1_sparse, mat1_meta, mat2, out_dtype=dtype_out)
            else:
                output1 = torch._sparse_semi_structured_addmm(
                    input, mat1_sparse, mat1_meta, mat2, alpha=alpha, beta=beta, out_dtype=dtype_out
                )
            # 使用 torch.testing.assert_close 断言 output1 和 output0 的近似相等性
            torch.testing.assert_close(output1.to(dtype_dense), output0, rtol=rtol, atol=atol)

        # 如果数据类型是 torch.float32
        if dtype == torch.float32:
            # 内部将输入转换为 TF32 以进行稀疏 GEMM，为了结果匹配，也进行稠密 GEMM
            orig = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True

        # 定义输出数据类型字典 dtype_out
        dtype_out = {torch.int8: torch.int32, torch.half: torch.half, torch.bfloat16: torch.bfloat16, torch.float32: torch.float32}
        # 设置相对误差和绝对误差的默认值
        rtol, atol = 1e-3, 1e-3
        # 根据数据类型调整误差容限
        if dtype == torch.bfloat16:
            rtol, atol = 5e-3, 5e-3
        elif dtype == torch.float32:
            rtol, atol = 1e-3, 75e-2
        # 遍历 m, n, k 的各种组合和 use_input 的两种情况，执行 run_test 函数
        for m, n, k, use_input in \
                itertools.product(range(3), range(3), range(3), (False, True)):
            m = 2 ** m * 32
            n = 2 ** n * 32
            k = 2 ** k * 128
            run_test(m, n, k, device, dtype, dtype_out[dtype], use_input, rtol, atol)

        # 如果数据类型是 torch.float32，恢复原始的 allow_tf32 设置
        if dtype == torch.float32:
            torch.backends.cuda.matmul.allow_tf32 = orig


    @unittest.skipIf(not has_triton(), "Test needs triton and recent GPU arch")
    @inference_dtypes
    # 定义一个测试方法，用于测试稀疏半结构化张量的转换操作
    def test_conversions(self, device, dtype):

        # 定义一个内部函数，用于运行单个测试
        def run_test(r, c, device, dtype):
            # 生成一个指定形状的随机稀疏半结构化张量作为参考密集张量
            dense_ref = rand_sparse_semi_structured(r, c, dtype, device)

            # 将参考密集张量转换为稀疏半结构化张量
            compressed = to_sparse_semi_structured(dense_ref)

            # 使用 torch.ops.aten._to_sparse_semi_structured 操作符，
            # 基于给定的密集矩阵执行转换，得到对应的稀疏矩阵和元数据矩阵，
            # 这里使用元数据矩阵作为参考，用于与由 SparseSemiStructuredTensor 类构造函数产生的元数据矩阵进行比较。
            _, meta_ref = torch.ops.aten._to_sparse_semi_structured(dense_ref)

            # 获取压缩后稀疏半结构化张量的索引
            meta = compressed.indices()

            # 断言压缩后的索引与参考元数据矩阵相等
            torch.testing.assert_close(meta, meta_ref, rtol=0, atol=0)

            # 将压缩后的稀疏半结构化张量转换回密集张量
            dense = compressed.to_dense()

            # 断言转换回的密集张量与参考密集张量相等
            torch.testing.assert_close(dense, dense_ref, rtol=0, atol=0)

        # 定义不同形状的测试用例
        shapes = [[32, 128], [32, 256], [64, 128], [64, 256]]
        for r, c in shapes:
            run_test(r, c, device, dtype)

    # 使用 unittest.skipIf 装饰器，当没有 Triton 或最近的 GPU 架构时跳过测试
    # 这个测试函数用于测试所有模式的稀疏半结构化张量的转换操作
    @unittest.skipIf(not has_triton(), "Test needs triton and recent GPU arch")
    @inference_dtypes
    def test_conversions_all_patterns(self, device, dtype):
        r, c = 32, 128

        # 生成所有模式的稀疏半结构化张量的随机逆矩阵和值矩阵作为参考密集张量
        dense_inv, dense_val = rand_sparse_semi_structured_all_patterns(r, c, dtype, device)

        # 将参考密集逆矩阵转换为稀疏半结构化张量
        compressed = to_sparse_semi_structured(dense_inv)

        # 将压缩后的稀疏半结构化张量转换回密集张量
        dense = compressed.to_dense()

        # 断言转换回的密集张量与参考值矩阵相等
        torch.testing.assert_close(dense, dense_val, rtol=0, atol=0)
CUSPARSELT_NUM_ALG_IDS = 4
CUSPARSELT_MIXED_DTYPE_SUPPORT = [torch.float16, torch.bfloat16, torch.int32]

# 定义测试类 TestSparseSemiStructuredCUSPARSELT，继承自 TestCase
class TestSparseSemiStructuredCUSPARSELT(TestCase):
    """
    This contains cuSPARSELt specific tests for
        torch._cslt_compress
        torch._cslt_sparse_mm
    """

    # 在每个测试方法运行前执行的设置方法
    def setUp(self):
        # 如果不是 SM8X 架构，则跳过测试
        if not _IS_SM8X:
            self.skipTest('Only runs on SM80')
        # 如果不支持 cuSPARSELt，则跳过测试
        if "cusparselt" not in SEMI_STRUCTURED_SUPPORTED_BACKENDS:
            self.skipTest('cuSPARSELt not enabled')

    # 参数化测试方法，测试不同输出数据类型的情况
    @parametrize("out_dtype", CUSPARSELT_MIXED_DTYPE_SUPPORT)
    @parametrize("dense_input_shape", [(128, 128)])
    def test_cslt_sparse_mm_mixed_dtype(self, dense_input_shape, out_dtype, device):
        # 创建稀疏半结构化掩码 A
        A = rand_sparse_semi_structured_mask(128, 128, dtype=torch.int8)
        # 对 A 进行压缩
        A_compressed = torch._cslt_compress(A)

        # 创建密集矩阵 B
        B = torch.rand(dense_input_shape, device=device).to(torch.int8)

        # 在 CPU 上进行矩阵乘法，转换输出数据类型为 out_dtype
        dense_result = torch.mm(A.cpu().to(torch.int64), B.t().cpu().to(torch.int64)).to(device, dtype=out_dtype)
        # 调用稀疏矩阵乘法函数，计算稀疏结果
        sparse_result = torch._cslt_sparse_mm(A_compressed, B.t(), out_dtype=out_dtype)
        # 断言稠密结果和稀疏结果的近似相等性
        torch.testing.assert_close(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    # 参数化测试方法，测试带有 alpha 参数的稀疏矩阵乘法
    @training_dtypes
    def test_cslt_sparse_mm_alpha(self, dtype, device):
        # 创建稀疏向量 A 和密集矩阵 B
        A = torch.Tensor([0, 0, 1, 1]).tile((128, 64)).to(dtype).cuda()
        B = torch.ones((256, 128), device=device).to(dtype)
        alpha = torch.Tensor([2**(-i) for i in range(128)]).cuda()

        # 对 A 进行压缩
        A_compressed = torch._cslt_compress(A)
        # 调用稀疏矩阵乘法函数，带有 alpha 参数
        sparse_result = torch._cslt_sparse_mm(A_compressed, B, alpha=alpha)

        # 计算 alpha 缩放后的密集结果
        alpha_scaled = torch.stack([alpha] * 128).t()
        dense_result = alpha_scaled * torch.mm(A.to(torch.float32), B.to(torch.float32))
        dense_result = dense_result.to(dtype)

        # 断言稀疏结果和密集结果的近似相等性
        torch.testing.assert_close(sparse_result, dense_result, rtol=1e-3, atol=1e-3)

    # 参数化测试方法，测试带有 alpha 参数和不同输出数据类型的稀疏矩阵乘法
    @parametrize("out_dtype", CUSPARSELT_MIXED_DTYPE_SUPPORT)
    def test_cslt_sparse_mm_alpha_mixed_dtype(self, out_dtype, device):
        # 创建稀疏向量 A 和密集矩阵 B
        A = torch.Tensor([0, 0, 10, 10]).tile((128, 64)).to(torch.int8).cuda()
        B = torch.ones((128, 256), device=device).to(torch.int8).t()
        alpha = torch.Tensor([2**(-i) if out_dtype is not torch.int32 else 1
                              for i in range(128)]).cuda()

        # 对 A 进行压缩
        A_compressed = torch._cslt_compress(A)
        # 调用稀疏矩阵乘法函数，带有 alpha 参数和指定输出数据类型
        sparse_result = torch._cslt_sparse_mm(A_compressed, B, alpha=alpha, out_dtype=out_dtype).cpu()

        # 计算 alpha 缩放后的密集结果
        alpha_scaled = torch.stack([alpha] * 128).t()
        dense_result = alpha_scaled.cpu() * torch.mm(A.to(torch.int64).cpu(), B.to(torch.int64).cpu())
        dense_result = dense_result.to(out_dtype)

        # 断言稀疏结果和密集结果的近似相等性
        torch.testing.assert_close(sparse_result, dense_result, rtol=1e-3, atol=1e-3)

    # 参数化测试方法，测试不同算法 ID 的情况
    @parametrize("alg_id", range(CUSPARSELT_NUM_ALG_IDS))
    @inference_dtypes
    # 定义测试函数，用于测试稀疏矩阵乘法算法的身份
    def test_cslt_sparse_mm_alg_id(self, device, dtype, alg_id):
        # 当数据类型为 torch.float32 且算法 ID 为 3 时，不支持
        if dtype == torch.float32 and alg_id == 3:
            return
        
        # 生成一个随机稀疏半结构化掩码矩阵 A，数据类型为指定的 dtype
        A = rand_sparse_semi_structured_mask(128, 128, dtype=dtype)
        # 对矩阵 A 进行压缩处理
        A_compressed = torch._cslt_compress(A)
        # 生成一个全为 1 的矩阵 B，设备为指定的 device，数据类型为 dtype
        B = torch.ones((128, 128), device=device).to(dtype)

        # 再次对 A 进行压缩处理
        A_compressed = torch._cslt_compress(A)
        # 使用稀疏矩阵乘法函数计算 sparse_result，传入压缩后的 A、B 的转置，以及指定的算法 ID
        sparse_result = torch._cslt_sparse_mm(A_compressed, B.t(), alg_id=alg_id)

        # 计算密集矩阵乘法的结果 dense_result，将 A 和 B 转换为 torch.float32 类型后计算
        dense_result = torch.mm(A.to(torch.float32), B.to(torch.float32))
        # 将 dense_result 转换为指定的 dtype
        dense_result = dense_result.to(dtype)

        # 使用测试断言检查 sparse_result 和 dense_result 是否足够接近
        torch.testing.assert_close(sparse_result, dense_result, rtol=1e-3, atol=1e-3)

    # 定义推理数据类型装饰器，用于测试稀疏矩阵乘法搜索
    @inference_dtypes
    def test_cslt_sparse_mm_search(self, device, dtype):
        # 生成一个随机稀疏半结构化掩码矩阵 A，数据类型为指定的 dtype
        A = rand_sparse_semi_structured_mask(128, 128, dtype=dtype)
        # 对矩阵 A 进行压缩处理
        A_compressed = torch._cslt_compress(A)
        # 生成一个全为 1 的矩阵 B，设备为指定的 device，数据类型为 dtype
        B = torch.ones((128, 128), device=device).to(dtype)

        # 再次对 A 进行压缩处理
        A_compressed = torch._cslt_compress(A)
        # 使用稀疏矩阵乘法搜索函数获取算法 ID
        alg_id = torch._cslt_sparse_mm_search(A_compressed, B.t())
        
        # 对于 cuSPARSELt v0.4.0，有一个 bug：尽管有 5 个算法 ID，但使用最后一个（4）会导致错误
        # 在 cuSPARSELt v0.5.0 中，总共只有 4 个算法 ID，因此在更新时应该移除这里的 +1
        assert alg_id in range(CUSPARSELT_NUM_ALG_IDS + 1)
# 使用给定的测试类实例化设备类型测试，适用于 CUDA 平台
instantiate_device_type_tests(TestSparseSemiStructured, globals(), only_for="cuda")
# 使用给定的测试类实例化设备类型测试，适用于 CUDA 平台，使用 CUTLASS 库
instantiate_device_type_tests(TestSparseSemiStructuredCUTLASS, globals(), only_for="cuda")
# 使用给定的测试类实例化设备类型测试，适用于 CUDA 平台，使用 CUSPARSELT 库
instantiate_device_type_tests(TestSparseSemiStructuredCUSPARSELT, globals(), only_for="cuda")
# 使用给定的测试类实例化设备类型训练测试，适用于 CUDA 平台
instantiate_device_type_tests(TestSparseSemiStructuredTraining, globals(), only_for="cuda")

# 如果当前脚本作为主程序运行，则执行所有测试
if __name__ == "__main__":
    run_tests()
```