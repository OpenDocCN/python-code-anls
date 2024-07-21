# `.\pytorch\test\distributed\_tensor\test_dtensor_ops.py`

```
# 载入单元测试框架
import unittest
# 载入警告模块
import warnings

# 载入PyTorch库
import torch
# 载入分布式操作模块
import torch.distributed as dist
# 载入测试用例中的常用方法和调用
import torch.testing._internal.common_methods_invocations as common_ops

# 从PyTorch分布式张量模块中导入设备网格和分布式张量类
from torch.distributed._tensor import DeviceMesh, DTensor

# 从PyTorch覆盖模块中解析名称
from torch.overrides import resolve_name
# 从测试中导入设备类型测试的实例化，操作
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
# 从测试中导入方法装饰信息，操作数据库
from torch.testing._internal.common_methods_invocations import DecorateInfo, op_db
# 从测试中导入运行测试用例和抑制警告的工具函数
from torch.testing._internal.common_utils import (
    run_tests,
    suppress_warnings,
    TEST_WITH_ASAN,
)
# 从测试中导入分布式张量的常见操作和测试基类
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorConverter,
    DTensorOpTestBase,
)
# 从PyTorch工具模块中导入_pytree相关函数
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map

# 重写常用大小变量为可均匀分片的数值
# 之后可以启用不均匀分片，但需要进一步调整样本输入（例如视图/重塑需要调整形状大小）
common_ops.L = 24
common_ops.M = 12
common_ops.S = 4
common_ops.XS = 2


# 从functorch中复制过来的函数，用于标记测试失败
def xfail(op_name, variant_name="", *, device_type=None, dtypes=None):
    return (op_name, variant_name, device_type, dtypes, True)


# 标记跳过测试的函数
def skip(op_name, variant_name="", *, device_type=None, dtypes=None):
    return (op_name, variant_name, device_type, dtypes, False)


# 跳过指定测试用例中的操作
def skipOps(test_case_name, base_test_name, to_skip):
    # 获取所有操作信息
    all_opinfos = op_db
    # 遍历要跳过的操作列表
    for xfail in to_skip:
        op_name, variant_name, device_type, dtypes, expected_failure = xfail
        # 查找匹配的操作信息
        matching_opinfos = [
            o
            for o in all_opinfos
            if o.name == op_name and o.variant_test_name == variant_name
        ]
        # 确保找到了对应的操作信息
        assert len(matching_opinfos) >= 1, f"Couldn't find OpInfo for {xfail}"
        # 遍历匹配的操作信息并添加装饰器
        for opinfo in matching_opinfos:
            decorators = list(opinfo.decorators)
            if expected_failure:
                # 如果期望失败，则添加预期失败装饰器
                decorator = DecorateInfo(
                    unittest.expectedFailure,
                    test_case_name,
                    base_test_name,
                    device_type=device_type,
                    dtypes=dtypes,
                )
            else:
                # 否则添加跳过装饰器
                decorator = DecorateInfo(
                    unittest.skip("Skipped!"),
                    test_case_name,
                    base_test_name,
                    device_type=device_type,
                    dtypes=dtypes,
                )
            decorators.append(decorator)
            opinfo.decorators = tuple(decorators)

    # 此装饰器不会改变函数fn
    def wrapped(fn):
        return fn

    return wrapped


# 重新生成这个失败列表，启用下面函数的dry_run
# check_dtensor_func(self, test, op, dry_run=True)，然后运行类似
# python test/distributed/_tensor/test_dtensor_ops.py > failed.expect
dtensor_fails = {
   `
# 声明一系列测试失败的函数名，准备在操作得到全面支持时从列表中移除
xfail("__getitem__"),
xfail("__rsub__"),
xfail("_chunk_cat"),
xfail("_native_batch_norm_legit"),
xfail("_upsample_bilinear2d_aa"),
xfail("addbmm"),
xfail("addmv"),
xfail("addr"),
xfail("all"),
xfail("allclose"),
xfail("alias_copy"),
xfail("amax"),
xfail("amin"),
xfail("aminmax"),
xfail("any"),
xfail("arange"),
xfail("argmax"),
xfail("argmin"),
xfail("argsort"),
xfail("as_strided"),
xfail("as_strided", "partial_views"),
xfail("as_strided_copy"),
xfail("as_strided_scatter"),
xfail("bernoulli"),
xfail("_batch_norm_with_update"),
xfail("block_diag"),
xfail("broadcast_shapes"),
xfail("cauchy"),
xfail("cdist"),
xfail("cholesky"),
xfail("cholesky_inverse"),
xfail("cholesky_solve"),
xfail("chunk"),
xfail("clamp"),
xfail("clamp_max"),
xfail("clamp_min"),
xfail("combinations"),
xfail("complex"),
xfail("constant_pad_nd"),
xfail("corrcoef"),
xfail("count_nonzero"),
xfail("cross"),
xfail("cummax"),
xfail("cummin"),
xfail("cumsum"),
xfail("cumulative_trapezoid"),
xfail("diag"),
xfail("diag_embed"),
xfail("diagflat"),
xfail("diagonal"),
xfail("diagonal_copy"),
xfail("diagonal_scatter"),
xfail("dist"),
xfail("dot"),
xfail("einsum"),
xfail("empty"),
xfail("empty_strided"),
xfail("empty_like"),
xfail("empty_permuted"),
xfail("exponential"),
xfail("equal"),
xfail("eye"),
xfail("fft.fft2"),
xfail("fft.fft"),
xfail("fft.fftn"),
xfail("fft.fftshift"),
xfail("fft.ifft2"),
xfail("fft.ifft"),
xfail("fft.ifftshift"),
xfail("fft.ihfft2"),
xfail("fft.ihfft"),
xfail("fft.ihfftn"),
xfail("fft.irfft2"),
xfail("fft.irfftn"),
xfail("fft.rfft2"),
xfail("fft.rfft"),
xfail("fft.rfftn"),
xfail("fill"),
xfail("flip"),
xfail("fliplr"),
xfail("flipud"),
xfail("floor_divide"),
xfail("fmax"),
xfail("fmin"),
xfail("frexp"),
xfail("full"),
xfail("full_like"),
xfail("gather"),
xfail("geometric"),
xfail("geqrf"),
xfail("grid_sampler_2d"),
xfail("gradient"),
xfail("heaviside"),
xfail("histc"),
xfail("histogram"),
xfail("histogramdd"),
xfail("index_add"),
xfail("index_copy"),
xfail("index_fill"),
xfail("index_put"),
xfail("index_reduce", "prod"),
xfail("index_reduce", "mean"),
xfail("index_reduce", "amax"),
xfail("index_reduce", "amin"),
xfail("index_select"),
xfail("isin"),
xfail("isinf"),
xfail("isneginf"),
xfail("isposinf"),
xfail("kthvalue"),
xfail("linalg.cholesky"),
xfail("linalg.cholesky_ex"),
xfail("linalg.cross"),
xfail("linalg.det"),
xfail("linalg.det", "singular"),
    # 标记为预期失败的测试用例：linalg.diagonal
    xfail("linalg.diagonal"),
    # 标记为预期失败的测试用例：linalg.eig
    xfail("linalg.eig"),
    # 标记为预期失败的测试用例：linalg.eigh
    xfail("linalg.eigh"),
    # 标记为预期失败的测试用例：linalg.eigvals
    xfail("linalg.eigvals"),
    # 标记为预期失败的测试用例：linalg.eigvalsh
    xfail("linalg.eigvalsh"),
    # 标记为预期失败的测试用例：linalg.householder_product
    xfail("linalg.householder_product"),
    # 标记为预期失败的测试用例：linalg.inv
    xfail("linalg.inv"),
    # 标记为预期失败的测试用例：linalg.inv_ex
    xfail("linalg.inv_ex"),
    # 标记为预期失败的测试用例：linalg.ldl_factor
    xfail("linalg.ldl_factor"),
    # 标记为预期失败的测试用例：linalg.ldl_factor_ex
    xfail("linalg.ldl_factor_ex"),
    # 标记为预期失败的测试用例：linalg.ldl_solve
    xfail("linalg.ldl_solve"),
    # 标记为预期失败的测试用例：linalg.lstsq
    xfail("linalg.lstsq"),
    # 标记为预期失败的测试用例：linalg.lstsq，针对梯度导向的测试
    xfail("linalg.lstsq", "grad_oriented"),
    # 标记为预期失败的测试用例：linalg.lu
    xfail("linalg.lu"),
    # 标记为预期失败的测试用例：linalg.lu_factor
    xfail("linalg.lu_factor"),
    # 标记为预期失败的测试用例：linalg.lu_factor_ex
    xfail("linalg.lu_factor_ex"),
    # 标记为预期失败的测试用例：linalg.lu_solve
    xfail("linalg.lu_solve"),
    # 标记为预期失败的测试用例：linalg.matrix_norm
    xfail("linalg.matrix_norm"),
    # 标记为预期失败的测试用例：linalg.matrix_power
    xfail("linalg.matrix_power"),
    # 标记为预期失败的测试用例：linalg.matrix_rank
    xfail("linalg.matrix_rank"),
    # 标记为预期失败的测试用例：linalg.matrix_rank，针对厄米特矩阵的测试
    xfail("linalg.matrix_rank", "hermitian"),
    # 标记为预期失败的测试用例：linalg.multi_dot
    xfail("linalg.multi_dot"),
    # 标记为预期失败的测试用例：linalg.norm
    xfail("linalg.norm"),
    # 标记为预期失败的测试用例：linalg.norm，针对零点的子梯度测试
    xfail("linalg.norm", "subgradients_at_zero"),
    # 标记为预期失败的测试用例：linalg.pinv
    xfail("linalg.pinv"),
    # 标记为预期失败的测试用例：linalg.pinv，针对厄米特矩阵的测试
    xfail("linalg.pinv", "hermitian"),
    # 标记为预期失败的测试用例：linalg.qr
    xfail("linalg.qr"),
    # 标记为预期失败的测试用例：linalg.slogdet
    xfail("linalg.slogdet"),
    # 标记为预期失败的测试用例：linalg.solve
    xfail("linalg.solve"),
    # 标记为预期失败的测试用例：linalg.solve_ex
    xfail("linalg.solve_ex"),
    # 标记为预期失败的测试用例：linalg.solve_triangular
    xfail("linalg.solve_triangular"),
    # 标记为预期失败的测试用例：linalg.tensorinv
    xfail("linalg.tensorinv"),
    # 标记为预期失败的测试用例：linalg.tensorsolve
    xfail("linalg.tensorsolve"),
    # 标记为预期失败的测试用例：linalg.vander
    xfail("linalg.vander"),
    # 标记为预期失败的测试用例：linalg.vecdot
    xfail("linalg.vecdot"),
    # 标记为预期失败的测试用例：linspace
    xfail("linspace"),
    # 标记为预期失败的测试用例：linspace，针对张量重载的测试
    xfail("linspace", "tensor_overload"),
    # 标记为预期失败的测试用例：log_normal
    xfail("log_normal"),
    # 标记为预期失败的测试用例：logcumsumexp
    xfail("logcumsumexp"),
    # 标记为预期失败的测试用例：logdet
    xfail("logdet"),
    # 标记为预期失败的测试用例：logspace
    xfail("logspace"),
    # 标记为预期失败的测试用例：logspace，针对张量重载的测试
    xfail("logspace", "tensor_overload"),
    # 标记为预期失败的测试用例：logsumexp
    xfail("logsumexp"),
    # 标记为预期失败的测试用例：lu
    xfail("lu"),
    # 标记为预期失败的测试用例：lu_solve
    xfail("lu_solve"),
    # 标记为预期失败的测试用例：lu_unpack
    xfail("lu_unpack"),
    # 标记为预期失败的测试用例：masked_fill
    xfail("masked_fill"),
    # 标记为预期失败的测试用例：masked_scatter
    xfail("masked_scatter"),
    # 标记为预期失败的测试用例：masked_select
    xfail("masked_select"),
    # 标记为预期失败的测试用例：masked.amax
    xfail("masked.amax"),
    # 标记为预期失败的测试用例：masked.amin
    xfail("masked.amin"),
    # 标记为预期失败的测试用例：masked.argmax
    xfail("masked.argmax"),
    # 标记为预期失败的测试用例：masked.argmin
    xfail("masked.argmin"),
    # 标记为预期失败的测试用例：masked.cumprod
    xfail("masked.cumprod"),
    # 标记为预期失败的测试用例：masked.cumsum
    xfail("masked.cumsum"),
    # 标记为预期失败的测试用例：masked.logsumexp
    xfail("masked.logsumexp"),
    # 标记为预期失败的测试用例：masked.median
    xfail("masked.median"),
    # 标记为预期失败的测试用例：matrix_exp
    xfail("matrix_exp"),
    # 标记为预期失败的测试用例：max，针对二进制操作的测试
    xfail("max", "binary"),
    # 标记为预期失败的测试用例：max，带有维度缩减的测试
    xfail("max", "reduction_with_dim"),
    # 标记为预期失败的测试用例：maximum
    xfail("maximum"),
    # 标记为预期失败的测试用例：median
    xfail("median"),
    # 标记为预期失败的测试用例：min，针对二进制操作的测试
    xfail("min", "binary"),
    # 标记为预期失败的测试用例：min，带有维度缩减的测试
    xfail("min", "reduction_with_dim"),
    # 标记为预期失败的测试用例：minimum
    xfail("minimum"),
    # 标记为预期失败的测试用例：mode
    xfail("mode"),
    # 标记为预期失败的测试用例：msort
    xfail("msort"),
    # 标记为预期失败的测试用例：multinomial
    xfail("multinomial"),
    # 标记为预期失败的测试用例：mv
    xfail("mv"),
    # 标记为预期失败的测试用例：max_pool2d_with_indices_backward
    xfail("max_pool2d_with_indices_backward", ""),
    # 标记为预期失败的测试用例：nanmean
    xfail("nanmean
    # 标记为不期望通过的测试函数：nn.functional.binary_cross_entropy_with_logits
    xfail("nn.functional.binary_cross_entropy_with_logits"),
    # 标记为不期望通过的测试函数：nn.functional.celu
    xfail("nn.functional.celu"),
    # 标记为不期望通过的测试函数：nn.functional.conv1d
    xfail("nn.functional.conv1d"),
    # 标记为不期望通过的测试函数：nn.functional.conv2d
    xfail("nn.functional.conv2d"),
    # 标记为不期望通过的测试函数：nn.functional.conv3d
    xfail("nn.functional.conv3d"),
    # 标记为不期望通过的测试函数：nn.functional.conv_transpose1d
    xfail("nn.functional.conv_transpose1d"),
    # 标记为不期望通过的测试函数：nn.functional.conv_transpose2d
    xfail("nn.functional.conv_transpose2d"),
    # 标记为不期望通过的测试函数：nn.functional.conv_transpose3d
    xfail("nn.functional.conv_transpose3d"),
    # 标记为不期望通过的测试函数：nn.functional.cosine_similarity
    xfail("nn.functional.cosine_similarity"),
    # 标记为不期望通过的测试函数：nn.functional.ctc_loss
    xfail("nn.functional.ctc_loss"),
    # 标记为不期望通过的测试函数：nn.functional.dropout
    xfail("nn.functional.dropout"),
    # 标记为不期望通过的测试函数：nn.functional.dropout2d
    xfail("nn.functional.dropout2d"),
    # 标记为不期望通过的测试函数：nn.functional.dropout3d
    xfail("nn.functional.dropout3d"),
    # 标记为不期望通过的测试函数：nn.functional.elu
    xfail("nn.functional.elu"),
    # 标记为不期望通过的测试函数：nn.functional.fractional_max_pool2d
    xfail("nn.functional.fractional_max_pool2d"),
    # 标记为不期望通过的测试函数：nn.functional.fractional_max_pool3d
    xfail("nn.functional.fractional_max_pool3d"),
    # 标记为不期望通过的测试函数：nn.functional.gaussian_nll_loss
    xfail("nn.functional.gaussian_nll_loss"),
    # 标记为不期望通过的测试函数：nn.functional.glu
    xfail("nn.functional.glu"),
    # 标记为不期望通过的测试函数：nn.functional.grid_sample
    xfail("nn.functional.grid_sample"),
    # 标记为不期望通过的测试函数：nn.functional.group_norm
    xfail("nn.functional.group_norm"),
    # 标记为不期望通过的测试函数：nn.functional.hardshrink
    xfail("nn.functional.hardshrink"),
    # 标记为不期望通过的测试函数：nn.functional.hardsigmoid
    xfail("nn.functional.hardsigmoid"),
    # 标记为不期望通过的测试函数：nn.functional.hardswish
    xfail("nn.functional.hardswish"),
    # 标记为不期望通过的测试函数：nn.functional.hardtanh
    xfail("nn.functional.hardtanh"),
    # 标记为不期望通过的测试函数：nn.functional.huber_loss
    xfail("nn.functional.huber_loss"),
    # 标记为不期望通过的测试函数：nn.functional.instance_norm
    xfail("nn.functional.instance_norm"),
    # 标记为不期望通过的测试函数：nn.functional.interpolate area
    xfail("nn.functional.interpolate", "area"),
    # 标记为不期望通过的测试函数：nn.functional.interpolate bicubic
    xfail("nn.functional.interpolate", "bicubic"),
    # 标记为不期望通过的测试函数：nn.functional.interpolate bilinear
    xfail("nn.functional.interpolate", "bilinear"),
    # 标记为不期望通过的测试函数：nn.functional.interpolate linear
    xfail("nn.functional.interpolate", "linear"),
    # 标记为不期望通过的测试函数：nn.functional.interpolate nearest
    xfail("nn.functional.interpolate", "nearest"),
    # 标记为不期望通过的测试函数：nn.functional.interpolate nearest-exact
    xfail("nn.functional.interpolate", "nearest-exact"),
    # 标记为不期望通过的测试函数：nn.functional.interpolate trilinear
    xfail("nn.functional.interpolate", "trilinear"),
    # 标记为不期望通过的测试函数：nn.functional.leaky_relu
    xfail("nn.functional.leaky_relu"),
    # 标记为不期望通过的测试函数：nn.functional.linear
    xfail("nn.functional.linear"),
    # 标记为不期望通过的测试函数：nn.functional.local_response_norm
    xfail("nn.functional.local_response_norm"),
    # 标记为不期望通过的测试函数：nn.functional.logsigmoid
    xfail("nn.functional.logsigmoid"),
    # 标记为不期望通过的测试函数：nn.functional.margin_ranking_loss
    xfail("nn.functional.margin_ranking_loss"),
    # 标记为不期望通过的测试函数：nn.functional.max_pool1d
    xfail("nn.functional.max_pool1d"),
    # 标记为不期望通过的测试函数：nn.functional.max_pool2d
    xfail("nn.functional.max_pool2d"),
    # 标记为不期望通过的测试函数：nn.functional.max_pool3d
    xfail("nn.functional.max_pool3d"),
    # 标记为不期望通过的测试函数：nn.functional.max_unpool1d
    xfail("nn.functional.max_unpool1d"),
    # 标记为不期望通过的测试函数：nn.functional.max_unpool1d grad
    xfail("nn.functional.max_unpool1d", "grad"),
    # 标记为不期望通过的测试函数：nn.functional.max_unpool2d
    xfail("nn.functional.max_unpool2d"),
    # 标记为不期望通过的测试函数：nn.functional.max_unpool2d grad
    xfail("nn.functional.max_unpool2d", "grad"),
    # 标记为不期望通过的测试函数：nn.functional.max_unpool3d
    xfail("nn.functional.max_unpool3d"),
    # 标记为不期望通过的测试函数：nn.functional.max_unpool3d grad
    xfail("nn.functional.max_unpool3d", "grad"),
    # 标记为不期望通过的测试函数：nn.functional.mish
    xfail("nn.functional.mish"),
    # 标记为不期望通过的测试函数：nn.functional.mse_loss
    xfail("nn.functional.mse_loss"),
    # 标记为不期望通过的测试函数：nn.functional.multi_margin_loss
    xfail("nn.functional.multi_margin_loss"),
    # 标记为不期望通过的测试函数：nn.functional.multi_head_attention_forward
    xfail("nn.functional.multi_head_attention_forward"),
    # 标记为不期望通过的测试函数：nn.functional.multilabel_margin_loss
    xfail("nn.functional.multilabel_margin_loss"),
    # 标记为不期望通过的测试函数：nn.functional.multilabel_soft_margin_loss
    xfail("nn.functional.multilabel_soft_margin_loss"),
    # 标记为不期望通过的测试函数：nn.functional.normalize
    xfail("nn.functional.normalize"),
    # 标记为不期望通过的测试函数：nn.functional.pad constant
    xfail("nn.functional.pad", "constant"),
    # 标记为不期望通过的测试函数：nn.functional.pad reflect
    xfail("nn.functional.pad", "reflect"),
    # 标记为不期望通过的测试函数：nn.functional.pad replicate
    xfail("nn.functional.pad", "replicate"),
    # 标记为不期望通过的测试函数：nn.functional.pad replicate_negative
    xfail("nn.functional.pad", "replicate_negative"),
    # 标记为不期望通过的测试函数：nn.functional.pairwise_distance
    xfail("nn.functional.pairwise_distance"),
    # 标记为不期望通过的测试函数：nn.functional.pdist
    xfail("nn.functional.pdist"),
    # 标记为不期望通过的测试函数：nn.functional.pixel_shuffle
    xfail("nn.functional.pixel_shuffle"),
    # 标记为不期望通过的测试函数：nn.functional.pixel_unshuffle
    xfail("nn.functional.pixel_unshuffle"),
    # 标记为不期望通过的测试函数：nn.functional.poisson_nll_loss
    xfail("nn.functional.poisson_nll_loss"),
    # 标记
    xfail("nn.functional.softshrink"),  # 标记 nn.functional.softshrink 函数为失败的测试用例
    xfail("nn.functional.threshold"),  # 标记 nn.functional.threshold 函数为失败的测试用例
    xfail("nn.functional.triplet_margin_loss"),  # 标记 nn.functional.triplet_margin_loss 函数为失败的测试用例
    xfail("nn.functional.triplet_margin_with_distance_loss"),  # 标记 nn.functional.triplet_margin_with_distance_loss 函数为失败的测试用例
    xfail("nn.functional.unfold"),  # 标记 nn.functional.unfold 函数为失败的测试用例
    xfail("nn.functional.upsample_bilinear"),  # 标记 nn.functional.upsample_bilinear 函数为失败的测试用例
    xfail("nn.functional.upsample_nearest"),  # 标记 nn.functional.upsample_nearest 函数为失败的测试用例
    xfail("nonzero"),  # 标记 nonzero 函数为失败的测试用例
    xfail("normal"),  # 标记 normal 函数为失败的测试用例
    xfail("normal", "number_mean"),  # 标记 normal 函数的 number_mean 参数组合为失败的测试用例
    xfail("normal", "in_place"),  # 标记 normal 函数的 in_place 参数组合为失败的测试用例
    xfail("ormqr"),  # 标记 ormqr 函数为失败的测试用例
    xfail("ones"),  # 标记 ones 函数为失败的测试用例
    xfail("pca_lowrank"),  # 标记 pca_lowrank 函数为失败的测试用例
    xfail("pinverse"),  # 标记 pinverse 函数为失败的测试用例
    xfail("polar"),  # 标记 polar 函数为失败的测试用例
    xfail("put"),  # 标记 put 函数为失败的测试用例
    xfail("qr"),  # 标记 qr 函数为失败的测试用例
    xfail("quantile"),  # 标记 quantile 函数为失败的测试用例
    xfail("rand_like"),  # 标记 rand_like 函数为失败的测试用例
    xfail("randint_like"),  # 标记 randint_like 函数为失败的测试用例
    xfail("randint"),  # 标记 randint 函数为失败的测试用例
    xfail("randn"),  # 标记 randn 函数为失败的测试用例
    xfail("randn_like"),  # 标记 randn_like 函数为失败的测试用例
    xfail("renorm"),  # 标记 renorm 函数为失败的测试用例
    xfail("repeat_interleave"),  # 标记 repeat_interleave 函数为失败的测试用例
    xfail("resize_"),  # 标记 resize_ 函数为失败的测试用例
    xfail("resize_as_"),  # 标记 resize_as_ 函数为失败的测试用例
    xfail("roll"),  # 标记 roll 函数为失败的测试用例
    xfail("rot90"),  # 标记 rot90 函数为失败的测试用例
    xfail("rsub"),  # 标记 rsub 函数为失败的测试用例
    xfail("scalar_tensor"),  # 标记 scalar_tensor 函数为失败的测试用例
    xfail("scatter_add"),  # 标记 scatter_add 函数为失败的测试用例
    xfail("scatter_reduce", "amax"),  # 标记 scatter_reduce 函数的 amax 参数组合为失败的测试用例
    xfail("scatter_reduce", "amin"),  # 标记 scatter_reduce 函数的 amin 参数组合为失败的测试用例
    xfail("scatter_reduce", "mean"),  # 标记 scatter_reduce 函数的 mean 参数组合为失败的测试用例
    xfail("scatter_reduce", "prod"),  # 标记 scatter_reduce 函数的 prod 参数组合为失败的测试用例
    xfail("scatter_reduce", "sum"),  # 标记 scatter_reduce 函数的 sum 参数组合为失败的测试用例
    xfail("searchsorted"),  # 标记 searchsorted 函数为失败的测试用例
    xfail("select"),  # 标记 select 函数为失败的测试用例
    xfail("select_scatter"),  # 标记 select_scatter 函数为失败的测试用例
    xfail("sort"),  # 标记 sort 函数为失败的测试用例
    xfail("sparse.sampled_addmm"),  # 标记 sparse.sampled_addmm 函数为失败的测试用例
    xfail("sparse.mm", "reduce"),  # 标记 sparse.mm 函数的 reduce 参数组合为失败的测试用例
    xfail("special.airy_ai"),  # 标记 special.airy_ai 函数为失败的测试用例
    xfail("special.bessel_j0"),  # 标记 special.bessel_j0 函数为失败的测试用例
    xfail("special.bessel_j1"),  # 标记 special.bessel_j1 函数为失败的测试用例
    xfail("special.bessel_y0"),  # 标记 special.bessel_y0 函数为失败的测试用例
    xfail("special.bessel_y1"),  # 标记 special.bessel_y1 函数为失败的测试用例
    xfail("special.chebyshev_polynomial_t"),  # 标记 special.chebyshev_polynomial_t 函数为失败的测试用例
    xfail("special.chebyshev_polynomial_u"),  # 标记 special.chebyshev_polynomial_u 函数为失败的测试用例
    xfail("special.entr"),  # 标记 special.entr 函数为失败的测试用例
    xfail("special.erfcx"),  # 标记 special.erfcx 函数为失败的测试用例
    xfail("special.hermite_polynomial_h"),  # 标记 special.hermite_polynomial_h 函数为失败的测试用例
    xfail("special.hermite_polynomial_he"),  # 标记 special.hermite_polynomial_he 函数为失败的测试用例
    xfail("special.i0e"),  # 标记 special.i0e 函数为失败的测试用例
    xfail("special.i1"),  # 标记 special.i1 函数为失败的测试用例
    xfail("special.i1e"),  # 标记 special.i1e 函数为失败的测试用例
    xfail("special.laguerre_polynomial_l"),  # 标记 special.laguerre_polynomial_l 函数为失败的测试用例
    xfail("special.log_ndtr"),  # 标记 special.log_ndtr 函数为失败的测试用例
    xfail("special.modified_bessel_i0"),  # 标记 special.modified_bessel_i0 函数为失败的测试用例
    xfail("special.modified_bessel_i1"),  # 标记 special.modified_bessel_i1 函数为失败的测试用例
    xfail("special.modified_bessel_k0"),  # 标记 special.modified_bessel_k0 函数为失败的测试用例
    xfail("special.modified_bessel_k1"),  # 标记 special.modified_bessel_k1 函数为失败的测试用例
    xfail("special.ndtri"),  # 标记 special.ndtri 函数为失败的测试用例
    xfail("special.scaled_modified_bessel_k0"),  # 标记 special.scaled_modified_bessel_k0 函数为失败的测试用例
    xfail("special.scaled_modified_bessel_k1"),  # 标记 special.scaled_modified_bessel_k1 函数为失败的测试用例
    xfail("special.spherical_bessel_j0"),  # 标记 special.spherical_bessel_j0 函数为失败的测试用例
    xfail("special.xlog1py"),  # 标记 special.xlog1py 函数为失败的测试用例
    xfail("special.zeta"),  # 标记 special.zeta 函数为失败的测试用例
    xfail("squeeze", "multiple"),  # 标记 squeeze 函数的 multiple 参数组合为失败的测试用例
    xfail("signal.windows.bartlett"),  # 标记 signal.windows.bartlett 函数为失败的测试用例
    xfail("signal.windows.blackman"),  # 标记 signal.windows.blackman 函数为失败的测试用例
    xfail("signal.windows.cosine"),  # 标记 signal.windows.cosine 函数为失败的测试用例
    xfail("signal.windows.exponential"),  # 标记 signal.windows.exponential
    # 标记为预期失败的测试用例："tril"
    xfail("tril"),
    # 标记为预期失败的测试用例："triu"
    xfail("triu"),
    # 标记为预期失败的测试用例："unbind"
    xfail("unbind"),
    # 标记为预期失败的测试用例："unfold"
    xfail("unfold"),
    # 标记为预期失败的测试用例："unfold_copy"
    xfail("unfold_copy"),
    # 标记为预期失败的测试用例："uniform"
    xfail("uniform"),
    # 标记为预期失败的测试用例："unflatten"
    xfail("unflatten"),
    # 标记为预期失败的测试用例："unique_consecutive"
    xfail("unique_consecutive"),
    # 标记为预期失败的测试用例："unique"
    xfail("unique"),
    # 标记为预期失败的测试用例："unsafe_split"
    xfail("unsafe_split"),
    # 标记为预期失败的测试用例："unsafe_chunk"
    xfail("unsafe_chunk"),
    # 标记为预期失败的测试用例："_unsafe_masked_index"
    xfail("_unsafe_masked_index"),
    # 标记为预期失败的测试用例："__unsafe_masked_index_put_accumulate"
    xfail("_unsafe_masked_index_put_accumulate"),
    # 标记为预期失败的测试用例："var_mean"
    xfail("var_mean"),
    # 标记为预期失败的测试用例："var_mean"，带有额外的参数："unbiased"
    xfail("var_mean", "unbiased"),
    # 标记为预期失败的测试用例："vdot"
    xfail("vdot"),
    # 标记为预期失败的测试用例："view_copy"
    xfail("view_copy"),
    # 标记为预期失败的测试用例："zeros"
    xfail("zeros"),
    
    # 下面是一组测试用例，由于缺少 dtensor 的支持，暂时标记为跳过
    # 在重新调整 op db 通用测试大小因子（例如 L、M、S）时，这些测试可能会失败
    # 生成的输入可能会错误，因此目前跳过这些测试，但应该稍后启用它们。
    # TODO: 需要清理此列表并删除所有这些情况
    
    # 标记为跳过的测试用例："argwhere"
    skip("argwhere"),
    # 标记为跳过的测试用例："cumprod"
    skip("cumprod"),
    # 标记为跳过的测试用例："__rmatmul__"
    skip("__rmatmul__"),
    # 标记为跳过的测试用例："meshgrid"，带有参数："list_of_tensors"
    skip("meshgrid", "list_of_tensors"),
    # 标记为跳过的测试用例："meshgrid"，带有参数："variadic_tensors"
    skip("meshgrid", "variadic_tensors"),
    # 标记为跳过的测试用例："nn.functional.scaled_dot_product_attention"
    skip("nn.functional.scaled_dot_product_attention"),
    # 标记为跳过的测试用例："nn.functional.softmin"
    skip("nn.functional.softmin"),
    # 标记为跳过的测试用例："nn.functional.embedding"
    skip("nn.functional.embedding"),
    # 标记为跳过的测试用例："nn.functional.embedding_bag"
    skip("nn.functional.embedding_bag"),
    # 标记为跳过的测试用例："nn.functional.feature_alpha_dropout"，带有参数："with_train"
    skip("nn.functional.feature_alpha_dropout", "with_train"),
    # 标记为跳过的测试用例："nn.functional.feature_alpha_dropout"，带有参数："without_train"
    skip("nn.functional.feature_alpha_dropout", "without_train"),
    # 标记为跳过的测试用例："nn.functional.hinge_embedding_loss"
    skip("nn.functional.hinge_embedding_loss"),
    # 标记为跳过的测试用例："nn.functional.cosine_embedding_loss"
    skip("nn.functional.cosine_embedding_loss"),
    # 标记为跳过的测试用例："fft.hfft"
    skip("fft.hfft"),
    # 标记为跳过的测试用例："fft.hfft2"
    skip("fft.hfft2"),
    # 标记为跳过的测试用例："fft.hfftn"
    skip("fft.hfftn"),
    # 标记为跳过的测试用例："fft.ifftn"
    skip("fft.ifftn"),
    # 标记为跳过的测试用例："fft.irfft"
    skip("fft.irfft"),
    # 标记为跳过的测试用例："istft"
    skip("istft"),
    # 标记为跳过的测试用例："isclose"
    skip("isclose"),
    # 标记为跳过的测试用例："isreal"
    skip("isreal"),
    # 标记为跳过的测试用例："matmul"
    skip("matmul"),
    # 标记为跳过的测试用例："masked.mean"
    skip("masked.mean"),
    # 标记为跳过的测试用例："masked.var"
    skip("masked.var"),
    # 标记为跳过的测试用例："masked.std"
    skip("masked.std"),
    # 标记为跳过的测试用例："masked.normalize"
    skip("masked.normalize"),
    # 标记为跳过的测试用例："prod"
    skip("prod"),
    # 标记为跳过的测试用例："__segment_reduce"，带有参数："lengths"
    skip("_segment_reduce", "lengths"),
    # 标记为跳过的测试用例："__segment_reduce"，带有参数："offsets"
    skip("_segment_reduce", "offsets"),
    
    # TODO: 需要修复以下操作
    # 标记为跳过的测试用例："squeeze"
    skip("squeeze"),
# Add a list of operations that are currently failing the backward pass (BW pass).
skip_bw = [
    None,  # corresponds to the transpose ops 'H' and 'T'
    "torch.bucketize",
    "torch.conj_physical",
    "torch.eq",
    "torch.isfinite",
    "torch.isnan",
]

# Set the world size for distributed operations to 4.
OP_DB_WORLD_SIZE = 4

# TODO: debug cuda illegal memory access issue and re-enable cuda tests
# Currently setting the device type to "cpu" due to debugging issues with CUDA.
DEVICE_TYPE = "cpu"

class TestDTensorOps(DTensorOpTestBase):
    @property
    def world_size(self) -> int:
        return OP_DB_WORLD_SIZE

    # Only allow float dtype for now; we can relax this constraint later.
    # This restriction is in place until quantization support is added.
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @suppress_warnings
    @ops(op_db, allowed_dtypes=(torch.float,))
    @skipOps("TestDTensorOps", "test_dtensor_op_db", dtensor_fails)
    def test_dtensor_op_db(self, dtype, op):
        # Initialize a device mesh with the specified DEVICE_TYPE and world size.
        self.mesh = DeviceMesh(DEVICE_TYPE, torch.arange(self.world_size))

        # Test each operation with distributed tensor inputs and normal inputs.
        def test():
            samples = op.sample_inputs(DEVICE_TYPE, dtype, requires_grad=True)
            for sample_input in samples:
                args = [sample_input.input] + list(sample_input.args)
                kwargs = sample_input.kwargs

                # Run the cross-reference test for the operation.
                self.run_dtensor_crossref(op.op, args, kwargs)

                # TODO: Implement testing for the out variant.
                # Testing the out variant is complex; it requires pre-allocation of dtensor out.
                # Some operations depend on known sharding placements (e.g., mm.out).

        self.check_dtensor_func(test, op)

    def assert_ref_dtensor_equal(self, dtensor_rs, rs):
        # Flatten the nested structure of dtensor_rs and rs.
        flat_dtensor_rs = pytree.tree_leaves(dtensor_rs)
        flat_rs = pytree.tree_leaves(rs)
        
        # Ensure the number of elements in flattened structures are equal.
        self.assertEqual(len(flat_dtensor_rs), len(flat_rs))
        
        # Compare each pair of tensors in flattened structures.
        for dtensor_r, r in zip(flat_dtensor_rs, flat_rs):
            if not isinstance(r, torch.Tensor):
                continue

            # Ensure both objects are instances of torch.Tensor.
            self.assertIsInstance(dtensor_r, torch.Tensor)
            
            # Check if shapes of tensors match.
            self.assertEqualOnRank(
                dtensor_r.shape,
                r.shape,
                f"Shape mismatch! original shape:{r.shape}, dtensor shape: {dtensor_r.shape}",
            )
            
            # Check if requires_grad flags of tensors match.
            self.assertEqualOnRank(
                dtensor_r.requires_grad,
                r.requires_grad,
                "op result requires_grad mismatch!"
                f"original requires_grad: {r.requires_grad}, "
                f"dtensor requires_grad: {dtensor_r.requires_grad}",
            )

            # Check if tensor values are equal on the same rank.
            self.assertEqualOnRank(dtensor_r, r)
    # 定义一个方法，用于检查指定测试函数的运行情况
    def check_dtensor_func(self, test_func, opinfo, dry_run=False):
        try:
            # 尝试运行测试函数
            test_func()
        except Exception:
            # 如果出现异常
            if not dry_run:
                # 如果不是干跑模式，直接抛出异常
                raise
            # 如果是干跑模式，并且当前进程的排名为0（即主进程）
            if dist.get_rank() == 0:
                # 如果测试信息中包含变体测试名称
                if opinfo.variant_test_name:
                    # 打印对应的xfail标记，包括操作信息和变体测试名称
                    print(f"xfail('{opinfo.name}', '{opinfo.variant_test_name}'),")
                else:
                    # 打印对应的xfail标记，只包括操作信息
                    print(f"xfail('{opinfo.name}'),")
# 仅为设备类型 DEVICE_TYPE 单独实例化测试 (即 CPU 或 GPU)
instantiate_device_type_tests(TestDTensorOps, globals(), only_for=(DEVICE_TYPE,))

# 如果作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```