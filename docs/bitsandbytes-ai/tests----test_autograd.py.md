# `bitsandbytes\tests\test_autograd.py`

```py
# 导入所需的模块
from typing import Tuple

import pytest
import torch

import bitsandbytes as bnb
from tests.helpers import (
    BOOLEAN_TRIPLES,
    BOOLEAN_TUPLES,
    TRUE_FALSE,
    describe_dtype,
    get_test_dims,
    id_formatter,
)

# 定义转置的取值范围
TRANSPOSE_VALS = [(False, True), (False, False)]

# 参数化测试用例，测试矩阵乘法
@pytest.mark.parametrize("dim1", get_test_dims(16, 64, n=1), ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", get_test_dims(32, 96, n=1), ids=id_formatter("dim2"))
@pytest.mark.parametrize("dim3", get_test_dims(32, 96, n=1), ids=id_formatter("dim3"))
@pytest.mark.parametrize("dim4", get_test_dims(32, 96, n=1), ids=id_formatter("dim4"))
@pytest.mark.parametrize("funcs", [(torch.bmm, bnb.bmm_cublas), (torch.matmul, bnb.matmul_cublas)], ids=["func=bmm", "func=matmul"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=describe_dtype)
@pytest.mark.parametrize("req_grad", BOOLEAN_TUPLES, ids=id_formatter("req_grad"))
@pytest.mark.parametrize("transpose", BOOLEAN_TUPLES, ids=id_formatter("transpose"))
def test_matmul(dim1, dim2, dim3, dim4, funcs, dtype, req_grad: Tuple[bool, bool], transpose: Tuple[bool, bool]):
    # 如果 dim2 大于 0，则将其调整为能被 16 整除的最大值
    if dim2 > 0:
        dim2 = dim2 - (dim2 % 16)
    # 将 dim3 和 dim4 调整为能被 16 整除的最大值
    dim3 = dim3 - (dim3 % 16)
    dim4 = dim4 - (dim4 % 16)

# 参数化测试用例，测试矩阵乘法
@pytest.mark.parametrize("dim1", get_test_dims(16, 64, n=1), ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", [*get_test_dims(32, 96, n=1), 0], ids=id_formatter("dim2"))
@pytest.mark.parametrize("dim3", get_test_dims(32, 96, n=1), ids=id_formatter("dim3"))
@pytest.mark.parametrize("dim4", get_test_dims(32, 96, n=1), ids=id_formatter("dim4"))
@pytest.mark.parametrize("decomp", [0.0, 6.0], ids=id_formatter("decomp"))
@pytest.mark.parametrize("funcs", [(torch.matmul, bnb.matmul), (torch.matmul, bnb.research.switchback_bnb)], ids=["func=matmul", "func=switchback_bnb"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=describe_dtype)
# 使用参数化测试，对 req_grad 进行测试，参数为 BOOLEAN_TRIPLES，使用 id_formatter 函数生成测试用例的标识符
@pytest.mark.parametrize("req_grad", BOOLEAN_TRIPLES, ids=id_formatter("req_grad"))
# 使用参数化测试，对 transpose 进行测试，参数为 TRANSPOSE_VALS，使用 id_formatter 函数生成测试用例的标识符
@pytest.mark.parametrize("transpose", TRANSPOSE_VALS, ids=id_formatter("transpose"))
# 使用参数化测试，对 has_fp16_weights 进行测试，参数为 TRUE_FALSE，使用 id_formatter 函数生成测试用例的标识符
@pytest.mark.parametrize("has_fp16_weights", TRUE_FALSE, ids=id_formatter("has_fp16_weights"))
# 使用参数化测试，对 has_bias 进行测试，参数为 TRUE_FALSE，使用 id_formatter 函数生成测试用例的标识符
@pytest.mark.parametrize("has_bias", TRUE_FALSE, ids=id_formatter("has_bias"))
# 定义测试函数 test_matmullt，接受多个参数
def test_matmullt(
    dim1,
    dim2,
    dim3,
    dim4,
    funcs,
    dtype,
    req_grad,
    transpose,
    decomp,
    has_fp16_weights,
    has_bias
):
    # 根据 transpose 的值确定 dimA 和 dimB 的维度
    dimA = (dim2, dim3) if not transpose[0] else (dim3, dim2)
    dimB = (dim3, dim4) if not transpose[1] else (dim4, dim3)
    # 在设备 "cuda" 上生成一个大小为 dimA[1] // 8 的随机整数张量，赋值给 outlier_dim
    outlier_dim = torch.randint(0, dimA[1], size=(dimA[1] // 8,), device="cuda")
    # 如果 has_bias 的值为 False，则将 req_grad 转换为列表，并将第三个元素设为 False
    if has_bias == False:
        req_grad = list(req_grad)
        req_grad[2] = False

# 使用参数化测试，对 dim1 进行测试，参数为 get_test_dims(16, 64, n=1)，使用 id_formatter 函数生成测试用例的标识符
@pytest.mark.parametrize("dim1", get_test_dims(16, 64, n=1), ids=id_formatter("dim1"))
# 使用参数化测试，对 dim2 进行测试，参数为 [*get_test_dims(32, 96, n=1), 0]，使用 id_formatter 函数生成测试用例的标识符
@pytest.mark.parametrize("dim2", [*get_test_dims(32, 96, n=1), 0], ids=id_formatter("dim2"))
# 使用参数化测试，对 dim3 进行测试，参数为 get_test_dims(32, 96, n=1)，使用 id_formatter 函数生成测试用例的标识符
@pytest.mark.parametrize("dim3", get_test_dims(32, 96, n=1), ids=id_formatter("dim3"))
# 使用参数化测试，对 dim4 进行测试，参数为 get_test_dims(32, 96, n=1)，使用 id_formatter 函数生成测试用例的标识符
@pytest.mark.parametrize("dim4", get_test_dims(32, 96, n=1), ids=id_formatter("dim4"))
# 使用参数化测试，对 funcs 进行测试，参数为 [(torch.matmul, bnb.matmul_4bit)]，使用 ids 指定标识符
@pytest.mark.parametrize("funcs", [(torch.matmul, bnb.matmul_4bit)], ids=["func=matmul"])
# 使用参数化测试，对 req_grad 进行测试，参数为 BOOLEAN_TRIPLES，使用 id_formatter 函数生成测试用例的标识符
@pytest.mark.parametrize("req_grad", BOOLEAN_TRIPLES, ids=id_formatter("req_grad"))
# 使用参数化测试，对 transpose 进行测试，参数为 TRANSPOSE_VALS，使用 id_formatter 函数生成测试用例的标识符
@pytest.mark.parametrize("transpose", TRANSPOSE_VALS, ids=id_formatter("transpose"))
# 使用参数化测试，对 has_bias 进行测试，参数为 TRUE_FALSE，使用 id_formatter 函数生成测试用例的标识符
@pytest.mark.parametrize("has_bias", TRUE_FALSE, ids=id_formatter("has_bias"))
# 使用参数化测试，对 dtype 进行测试，参数为 [torch.float16, torch.float32]，使用 describe_dtype 函数生成测试用例的标识符
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=describe_dtype)
# 使用参数化测试，对 compress_statistics 进行测试，参数为 TRUE_FALSE，使用 id_formatter 函数生成测试用例的标识符
@pytest.mark.parametrize("compress_statistics", TRUE_FALSE, ids=id_formatter("compress_statistics"))
# 使用参数化测试，对 quant_type 进行测试，参数为 ['fp4', 'nf4']，使用 id_formatter 函数生成测试用例的标识符
@pytest.mark.parametrize("quant_type", ['fp4', 'nf4'], ids=id_formatter("quant_type"))
# 定义测试函数 test_matmul_4bit，接受多个参数
def test_matmul_4bit(dim1, dim2, dim3, dim4, funcs, dtype, req_grad, transpose, has_bias, compress_statistics, quant_type):
    # 根据 transpose 的值确定 dimA 的维度
    dimA = (dim2, dim3) if not transpose[0] else (dim3, dim2)
    # 根据条件判断是否需要转置维度，如果不需要则保持原样，否则交换维度顺序
    dimB = (dim3, dim4) if not transpose[1] else (dim4, dim3)
    # 如果没有偏置项，则将梯度需求列表转换为可修改的列表，并将第三个元素设置为 False
    if has_bias == False:
        req_grad = list(req_grad)
        req_grad[2] = False
# 使用参数化测试，对 dim1 进行测试，范围为从 16 到 64，n=1
@pytest.mark.parametrize("dim1", get_test_dims(16, 64, n=1), ids=id_formatter("dim1"))
# 使用参数化测试，对 dim2 进行测试，范围为从 32 到 96，n=1，同时包含额外的值 0
@pytest.mark.parametrize("dim2", [*get_test_dims(32, 96, n=1), 0], ids=id_formatter("dim2"))
# 使用参数化测试，对 dim3 进行测试，范围为从 32 到 96，n=1
@pytest.mark.parametrize("dim3", get_test_dims(32, 96, n=1), ids=id_formatter("dim3"))
# 使用参数化测试，对 dim4 进行测试，范围为从 32 到 96，n=1
@pytest.mark.parametrize("dim4", get_test_dims(32, 96, n=1), ids=id_formatter("dim4"))
# 使用参数化测试，对 req_grad 进行测试，包含 True、False 组合
@pytest.mark.parametrize("req_grad", BOOLEAN_TRIPLES, ids=id_formatter("req_grad"))
# 使用参数化测试，对 transpose 进行测试，包含 True、False 组合
@pytest.mark.parametrize("transpose", TRANSPOSE_VALS, ids=id_formatter("transpose"))
# 使用参数化测试，对 dtype 进行测试，包含 torch.float16、torch.float32 两种数据类型
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=describe_dtype)
# 使用参数化测试，对 funcs 进行测试，包含两种不同的函数组合
@pytest.mark.parametrize("funcs", [(torch.matmul, bnb.research.matmul_fp8_mixed), (torch.matmul, bnb.research.matmul_fp8_global)], ids=["matmul_fp8_mixed", 'matmul_fp8_global'])
def test_matmul_fp8( dim1, dim2, dim3, dim4, funcs, dtype, req_grad, transpose):
    # 根据 transpose 的值确定 dimA 和 dimB 的维度
    dimA = (dim2, dim3) if not transpose[0] else (dim3, dim2)
    dimB = (dim3, dim4) if not transpose[1] else (dim4, dim3)
    # 将 req_grad 转换为列表，并将第三个元素设为 False
    req_grad = list(req_grad)
    req_grad[2] = False
```