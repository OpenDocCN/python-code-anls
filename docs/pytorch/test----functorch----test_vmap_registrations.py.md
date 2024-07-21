# `.\pytorch\test\functorch\test_vmap_registrations.py`

```
# Owner(s): ["module: functorch"]
import typing  # 导入 typing 模块，用于类型提示
import unittest  # 导入 unittest 模块，用于编写和运行单元测试

from torch._C import (  # 从 torch._C 模块中导入以下函数
    _dispatch_get_registrations_for_dispatch_key as get_registrations_for_dispatch_key,
)

from torch.testing._internal.common_utils import (  # 从 torch.testing._internal.common_utils 模块中导入以下函数和类
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)

xfail_functorch_batched = {  # 定义一个集合 xfail_functorch_batched，包含下列字符串元素
    "aten::is_nonzero",
    "aten::item",
    "aten::linalg_slogdet",
    "aten::masked_select_backward",
    "aten::one_hot",
    "aten::silu_backward",
    "aten::where",
}

xfail_functorch_batched_decomposition = {  # 定义一个集合 xfail_functorch_batched_decomposition，包含下列字符串元素
    "aten::alias_copy",
    "aten::as_strided_copy",
    "aten::diagonal_copy",
    "aten::is_same_size",
    "aten::unfold_copy",
}

xfail_not_implemented = {  # 定义一个集合 xfail_not_implemented，包含下列字符串元素
    "aten::affine_grid_generator_backward",
    "aten::align_as",
    "aten::align_tensors",
    "aten::align_to",
    "aten::align_to.ellipsis_idx",
    "aten::alpha_dropout",
    "aten::alpha_dropout_",
    "aten::argwhere",
    "aten::bilinear",
    "aten::can_cast",
    "aten::cat.names",
    "aten::chain_matmul",
    "aten::chalf",
    "aten::choose_qparams_optimized",
    "aten::clip_",
    "aten::clip_.Tensor",
    "aten::coalesce",
    "aten::column_stack",
    "aten::concat.names",
    "aten::concatenate.names",
    "aten::conj",
    "aten::conv_tbc_backward",
    "aten::ctc_loss.IntList",
    "aten::ctc_loss.Tensor",
    "aten::cudnn_is_acceptable",
    "aten::cummaxmin_backward",
    "aten::data",
    "aten::diagflat",
    "aten::divide.out_mode",
    "aten::divide_.Scalar",
    "aten::dropout",
    "aten::dropout_",
    "aten::embedding_bag",
    "aten::embedding_bag.padding_idx",
    "aten::feature_alpha_dropout",
    "aten::feature_alpha_dropout_",
    "aten::feature_dropout",
    "aten::feature_dropout_",
    "aten::fft_ihfft2",
    "aten::fft_ihfftn",
    "aten::fill_diagonal_",
    "aten::fix_",
    "aten::flatten.named_out_dim",
    "aten::flatten.using_names",
    "aten::flatten_dense_tensors",
    "aten::float_power_.Scalar",
    "aten::float_power_.Tensor",
    "aten::floor_divide_.Scalar",
    "aten::frobenius_norm",
    "aten::fused_moving_avg_obs_fake_quant",
    "aten::get_gradients",
    "aten::greater_.Scalar",
    "aten::greater_.Tensor",
    "aten::greater_equal_.Scalar",
    "aten::greater_equal_.Tensor",
    "aten::gru.data",
    "aten::gru.input",
    "aten::gru_cell",
    "aten::histogramdd",
    "aten::histogramdd.TensorList_bins",
    "aten::histogramdd.int_bins",
    "aten::infinitely_differentiable_gelu_backward",
    "aten::isclose",
    "aten::istft",
    "aten::item",  # 重复的元素，已在 xfail_functorch_batched 中出现
    "aten::kl_div",
    "aten::ldexp_",
    "aten::less_.Scalar",
    "aten::less_.Tensor",
    "aten::less_equal_.Scalar",
    "aten::less_equal_.Tensor",
    "aten::linalg_cond.p_str",
    "aten::linalg_eigh.eigvals",
    "aten::linalg_matrix_rank",
    "aten::linalg_matrix_rank.out_tol_tensor",
    "aten::linalg_matrix_rank.tol_tensor",
    "aten::linalg_pinv.out_rcond_tensor",
    "aten::linalg_pinv.rcond_tensor",
}
    # 列表包含多个字符串，每个字符串表示 PyTorch 操作的名称
    
    # 第一个操作：aten::linalg_slogdet
    # 第二个操作：aten::linalg_svd.U
    # 第三个操作：aten::linalg_tensorsolve
    # 第四个操作：aten::logsumexp.names
    # 第五个操作：aten::lstm.data
    # 第六个操作：aten::lstm.input
    # 第七个操作：aten::lstm_cell
    # 第八个操作：aten::lu_solve
    # 第九个操作：aten::margin_ranking_loss
    # 第十个操作：aten::masked_select_backward
    # 第十一个操作：aten::matrix_exp
    # 第十二个操作：aten::matrix_exp_backward
    # 第十三个操作：aten::max.names_dim
    # 第十四个操作：aten::max.names_dim_max
    # 第十五个操作：aten::mean.names_dim
    # 第十六个操作：aten::median.names_dim
    # 第十七个操作：aten::median.names_dim_values
    # 第十八个操作：aten::min.names_dim
    # 第十九个操作：aten::min.names_dim_min
    # 第二十个操作：aten::mish_backward
    # 第二十一个操作：aten::moveaxis.int
    # 第二十二个操作：aten::multilabel_margin_loss
    # 第二十三个操作：aten::nanmedian.names_dim
    # 第二十四个操作：aten::nanmedian.names_dim_values
    # 第二十五个操作：aten::nanquantile
    # 第二十六个操作：aten::nanquantile.scalar
    # 第二十七个操作：aten::narrow.Tensor
    # 第二十八个操作：aten::native_channel_shuffle
    # 第二十九个操作：aten::negative_
    # 第三十个操作：aten::nested_to_padded_tensor
    # 第三十一个操作：aten::nonzero_numpy
    # 第三十二个操作：aten::norm.names_ScalarOpt_dim
    # 第三十三个操作：aten::norm.names_ScalarOpt_dim_dtype
    # 第三十四个操作：aten::norm_except_dim
    # 第三十五个操作：aten::not_equal_.Scalar
    # 第三十六个操作：aten::not_equal_.Tensor
    # 第三十七个操作：aten::one_hot
    # 第三十八个操作：aten::output_nr
    # 第三十九个操作：aten::pad_sequence
    # 第四十个操作：aten::pdist
    # 第四十一个操作：aten::pin_memory
    # 第四十二个操作：aten::promote_types
    # 第四十三个操作：aten::qr.Q
    # 第四十四个操作：aten::quantile
    # 第四十五个操作：aten::quantile.scalar
    # 第四十六个操作：aten::refine_names
    # 第四十七个操作：aten::rename
    # 第四十八个操作：aten::rename_
    # 第四十九个操作：aten::requires_grad_
    # 第五十个操作：aten::retain_grad
    # 第五十一个操作：aten::retains_grad
    # 第五十二个操作：aten::rnn_relu.data
    # 第五十三个操作：aten::rnn_relu.input
    # 第五十四个操作：aten::rnn_relu_cell
    # 第五十五个操作：aten::rnn_tanh.data
    # 第五十六个操作：aten::rnn_tanh.input
    # 第五十七个操作：aten::rnn_tanh_cell
    # 第五十八个操作：aten::set_.source_Tensor_storage_offset
    # 第五十九个操作：aten::set_data
    # 第六十个操作：aten::silu_backward
    # 第六十一个操作：aten::slow_conv3d
    # 第六十二个操作：aten::smm
    # 第六十三个操作：aten::special_chebyshev_polynomial_t.n_scalar
    # 第六十四个操作：aten::special_chebyshev_polynomial_t.x_scalar
    # 第六十五个操作：aten::special_chebyshev_polynomial_u.n_scalar
    # 第六十六个操作：aten::special_chebyshev_polynomial_u.x_scalar
    # 第六十七个操作：aten::special_chebyshev_polynomial_v.n_scalar
    # 第六十八个操作：aten::special_chebyshev_polynomial_v.x_scalar
    # 第六十九个操作：aten::special_chebyshev_polynomial_w.n_scalar
    # 第七十个操作：aten::special_chebyshev_polynomial_w.x_scalar
    # 第七十一个操作：aten::special_hermite_polynomial_h.n_scalar
    # 第七十二个操作：aten::special_hermite_polynomial_h.x_scalar
    # 第七十三个操作：aten::special_hermite_polynomial_he.n_scalar
    # 第七十四个操作：aten::special_hermite_polynomial_he.x_scalar
    # 第七十五个操作：aten::special_laguerre_polynomial_l.n_scalar
    # 第七十六个操作：aten::special_laguerre_polynomial_l.x_scalar
    # 第七十七个操作：aten::special_legendre_polynomial_p.n_scalar
    # 第七十八个操作：aten::special_legendre_polynomial_p.x_scalar
    # 第七十九个操作：aten::special_shifted_chebyshev_polynomial_t.n_scalar
    # 第八十个操作：aten::special_shifted_chebyshev_polynomial_t.x_scalar
    # 第八十一个操作：aten::special_shifted_chebyshev_polynomial_u.n_scalar
    # 第八十二个操作：aten::special_shifted_chebyshev_polynomial_u.x_scalar
    # 第八十三个操作：aten::special_shifted_chebyshev_polynomial_v.n_scalar
    # 第八十四个操作：aten::special_shifted_chebyshev_polynomial_v.x_scalar
    # 创建包含特定字符串的常量变量
    "aten::special_shifted_chebyshev_polynomial_w.n_scalar",
    "aten::special_shifted_chebyshev_polynomial_w.x_scalar",
    
    # 创建常量变量，代表一个特定的操作或函数
    "aten::square_",
    
    # 创建常量变量，代表一个特定的操作或函数
    "aten::sspaddmm",
    
    # 创建包含特定字符串的常量变量
    "aten::std.correction_names",
    "aten::std.names_dim",
    
    # 创建包含特定字符串的常量变量
    "aten::std_mean.correction_names",
    "aten::std_mean.names_dim",
    
    # 创建常量变量，代表一个特定的操作或函数
    "aten::stft",
    
    # 创建常量变量，代表一个特定的操作或函数
    "aten::stft.center",
    
    # 创建常量变量，代表一个特定的操作或函数
    "aten::stride.int",
    
    # 创建常量变量，代表一个特定的操作或函数
    "aten::subtract.Scalar",
    
    # 创建常量变量，代表一个特定的操作或函数
    "aten::subtract_.Scalar",
    
    # 创建常量变量，代表一个特定的操作或函数
    "aten::subtract_.Tensor",
    
    # 创建常量变量，代表一个特定的操作或函数
    "aten::svd.U",
    
    # 创建常量变量，代表一个特定的操作或函数
    "aten::sym_size.int",
    
    # 创建常量变量，代表一个特定的操作或函数
    "aten::sym_stride.int",
    
    # 创建常量变量，代表一个特定的操作或函数
    "aten::sym_numel",
    
    # 创建常量变量，代表一个特定的操作或函数
    "aten::sym_storage_offset",
    
    # 创建常量变量，代表一个特定的操作或函数
    "aten::tensor_split.tensor_indices_or_sections",
    
    # 创建常量变量，代表一个特定的操作或函数
    "aten::thnn_conv2d",
    
    # 创建常量变量，代表一个特定的操作或函数
    "aten::to_dense",
    
    # 创建常量变量，代表一个特定的操作或函数
    "aten::to_dense_backward",
    
    # 创建常量变量，代表一个特定的操作或函数
    "aten::to_mkldnn_backward",
    
    # 创建常量变量，代表一个特定的操作或函数
    "aten::trace_backward",
    
    # 创建常量变量，代表一个特定的操作或函数
    "aten::triplet_margin_loss",
    
    # 创建常量变量，代表一个特定的操作或函数
    "aten::unflatten_dense_tensors",
    
    # 创建常量变量，代表一个特定的操作或函数
    "aten::vander",
    
    # 创建包含特定字符串的常量变量
    "aten::var.correction_names",
    "aten::var.names_dim",
    
    # 创建包含特定字符串的常量变量
    "aten::var_mean.correction_names",
    "aten::var_mean.names_dim",
    
    # 创建常量变量，代表一个特定的操作或函数
    "aten::where",
}


def dispatch_registrations(
    dispatch_key: str, xfails: set, filter_func: typing.Callable = lambda reg: True
):
    # 获取特定调度键对应的注册信息，并按名称排序
    registrations = sorted(get_registrations_for_dispatch_key(dispatch_key))
    # 创建子测试列表，每个注册信息对应一个子测试，名字格式为 [注册信息]，并根据条件添加装饰器
    subtests = [
        subtest(
            reg,
            name=f"[{reg}]",
            decorators=([unittest.expectedFailure] if reg in xfails else []),
        )
        for reg in registrations
        if filter_func(reg)
    ]
    # 返回参数化测试，每个子测试作为一个参数
    return parametrize("registration", subtests)


CompositeImplicitAutogradRegistrations = set(
    get_registrations_for_dispatch_key("CompositeImplicitAutograd")
)
FuncTorchBatchedRegistrations = set(
    get_registrations_for_dispatch_key("FuncTorchBatched")
)
FuncTorchBatchedDecompositionRegistrations = set(
    get_registrations_for_dispatch_key("FuncTorchBatchedDecomposition")
)


def filter_vmap_implementable(reg):
    # 将注册名称转换为小写，根据一系列规则过滤注册信息
    reg = reg.lower()
    if not reg.startswith("aten::"):
        return False
    if reg.startswith("aten::_"):
        return False
    if reg.endswith(".out"):
        return False
    if reg.endswith("_out"):
        return False
    if ".dimname" in reg:
        return False
    if "_dimname" in reg:
        return False
    if "fbgemm" in reg:
        return False
    if "quantize" in reg:
        return False
    if "sparse" in reg:
        return False
    if "::is_" in reg:
        return False
    # 符合条件的注册信息
    return True


class TestFunctorchDispatcher(TestCase):
    @dispatch_registrations("CompositeImplicitAutograd", xfail_functorch_batched)
    def test_register_a_batching_rule_for_composite_implicit_autograd(
        self, registration
    ):
        assert registration not in FuncTorchBatchedRegistrations, (
            f"You've added a batching rule for a CompositeImplicitAutograd operator {registration}. "
            "The correct way to add vmap support for it is to put it into BatchRulesDecomposition to "
            "reuse the CompositeImplicitAutograd decomposition"
        )

    @dispatch_registrations(
        "FuncTorchBatchedDecomposition", xfail_functorch_batched_decomposition
    )
    def test_register_functorch_batched_decomposition(self, registration):
        assert registration in CompositeImplicitAutogradRegistrations, (
            f"The registrations in BatchedDecompositions.cpp must be for CompositeImplicitAutograd "
            f"operations. If your operation {registration} is not CompositeImplicitAutograd, "
            "then please register it to the FuncTorchBatched key in another file."
        )

    @dispatch_registrations(
        "CompositeImplicitAutograd", xfail_not_implemented, filter_vmap_implementable
    )
    # 测试注册 CompositeImplicitAutograd 的函数，使用过滤器函数 filter_vmap_implementable 进行注册信息的过滤
    # 定义一个测试方法，用于检验未实现的批量注册
    def test_unimplemented_batched_registrations(self, registration):
        # 断言注册项是否在FuncTorchBatchedDecompositionRegistrations中，
        # 如果不在，提示用户检查是否有涵盖该运算符的OpInfo，并在BatchedDecompositions.cpp中添加注册。
        # 如果运算符不面向用户，请将其添加到xfail列表。
        assert registration in FuncTorchBatchedDecompositionRegistrations, (
            f"Please check that there is an OpInfo that covers the operator {registration} "
            "and add a registration in BatchedDecompositions.cpp. "
            "If your operator isn't user facing, please add it to the xfail list"
        )
# 实例化一个带参数的测试，使用 TestFunctorchDispatcher 类
instantiate_parametrized_tests(TestFunctorchDispatcher)

# 如果当前脚本作为主程序运行，则执行测试函数
if __name__ == "__main__":
    run_tests()
```