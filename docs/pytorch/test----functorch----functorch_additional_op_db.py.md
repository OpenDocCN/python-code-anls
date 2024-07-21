# `.\pytorch\test\functorch\functorch_additional_op_db.py`

```py
# 导入所需的模块和函数
import itertools
import unittest
from functools import partial

import torch

# 导入用于测试的内部模块和函数
from torch.testing._internal.common_dtype import (
    all_types_and_complex_and,
    floating_types,
    floating_types_and,
)
from torch.testing._internal.common_methods_invocations import (
    DecorateInfo,
    OpInfo,
    SampleInput,
)
from torch.testing._internal.common_utils import make_tensor

# 存储尚未合并到 PyTorch 核心中的操作信息列表
additional_op_db = []

# https://github.com/pytorch/pytorch/pull/61068

# 定义一个函数，生成用于 Conv2d 操作测试的输入样本
def sample_inputs_conv2d(
    has_bias, self, device, dtype, requires_grad, extra_args=(), groups=1
):
    # 输入通道数和输出通道数
    in_ch, out_ch = 6, 4
    # 生成输入张量
    inp = make_tensor(
        (2, in_ch * groups, 7, 5),
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        low=-1,
        high=1,
    )
    # 生成权重张量
    weight = make_tensor(
        (out_ch * groups, in_ch, 3, 2),
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        low=-1,
        high=1,
    )
    bias = None
    # 如果有偏置，则生成偏置张量
    if has_bias:
        bias = make_tensor(
            (out_ch * groups,),
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
            low=-1,
            high=1,
        )
    # 返回输入样本的列表，包括输入张量和可能的额外参数
    return [SampleInput(inp, args=((weight, bias) + extra_args))]

# 向操作信息数据库中添加样本输入生成函数
additional_op_db.extend(
    ]
)

# TODO: PyTorch 核心中有一个用于检查 requires_grad=True 的检查点。
# 我们实际上希望在这里对反向传播进行更多测试，这就是为什么我们有自己的函数

# 定义一个函数，生成用于嵌入操作测试的输入样本
def sample_inputs_embedding(op_info, device, dtype, requires_grad, **kwargs):
    # 定义生成输入张量的函数
    def make_input(shape):
        return make_tensor(
            shape, device=device, dtype=dtype, requires_grad=requires_grad
        )

    # 定义生成长整型输入张量的函数
    def make_long_input(shape, *, low, high):
        return make_tensor(shape, device=device, dtype=torch.long, low=low, high=high)

    M = 20
    S = 5
    def generator():
        # 定义一个生成器函数
        # 0-D 索引张量
        idx = make_long_input((), low=0, high=M)
        yield SampleInput(
            make_input((M, S)),
            args=(idx,),
        )

        # 1-D 索引张量
        idx = make_long_input((S,), low=0, high=M)
        yield SampleInput(
            make_input((M, S)),
            args=(idx,),
        )

        # 2-D 索引张量
        idx = make_long_input((S, S), low=0, high=M)
        yield SampleInput(
            make_input((M, S)),
            args=(idx,),
        )

        # 创建一个特定的 2x2 索引张量
        idx = make_long_input((2, 2), low=0, high=S)
        idx[0, 0] = 2
        idx[1, 1] = 2
        yield SampleInput(
            make_input((S, S)),
            args=(idx,),
            kwargs={"padding_idx": 2},
        )

        # 创建另一个特定的 2x2 索引张量
        idx = make_long_input((2, 2), low=0, high=S)
        idx[0, 0] = 4
        idx[1, 1] = 4
        yield SampleInput(
            make_input((S, S)),
            args=(idx,),
            kwargs={"padding_idx": -1},
        )

        # 根据特定索引的逆频率调整梯度的尺度
        idx = make_long_input((2, 2), low=0, high=S)
        idx[0, 0] = 1
        idx[0, 1] = 1
        weights = make_input((S, S))
        yield SampleInput(
            weights,
            args=(idx,),
            kwargs={"scale_grad_by_freq": True},
        )

    # 将生成器生成的所有样本收集为列表并返回
    return list(generator())
additional_op_db.append(
    OpInfo(
        "nn.functional.embedding",
        variant_test_name="functorch",
        # 使用 lambda 函数重新排列位置参数。
        # 这是因为当前只有 SampleInput 的 `input` 字段在梯度测试中被测试。
        # 定义了一个 lambda 函数，用于调用 torch.nn.functional.embedding 函数。
        op=lambda weight, idx, **kwargs: torch.nn.functional.embedding(
            idx, weight, **kwargs
        ),
        dtypes=floating_types_and(torch.bfloat16, torch.float16),  # 支持的数据类型，包括浮点数和特定的数据类型
        sample_inputs_func=sample_inputs_embedding,  # 获取样本输入的函数
        supports_forward_ad=True,  # 是否支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 是否支持前向后向自动微分
        supports_out=False,  # 是否支持输出
    )
)


def sample_inputs_mse_loss(op_info, device, dtype, requires_grad, **kwargs):
    def make_input(shape, requires_grad=requires_grad):
        return make_tensor(
            shape, device=device, dtype=dtype, requires_grad=requires_grad
        )

    rhs_requires_grad = kwargs.get("rhs_requires_grad", requires_grad)
    S = 5

    shapes = ((S, S), (S, S, S), (S, S, S, S))
    reductions = ("none", "mean", "sum")

    # 遍历所有形状和约简方式的组合
    for shape, reduction in itertools.product(shapes, reductions):
        yield SampleInput(
            make_input(shape),
            args=(make_input(shape, requires_grad=rhs_requires_grad),),
            kwargs={"reduction": reduction},
        )


additional_op_db.append(
    OpInfo(
        "nn.functional.mse_loss",
        variant_test_name="functorch",
        sample_inputs_func=sample_inputs_mse_loss,
        supports_out=False,  # 是否支持输出
        supports_forward_ad=True,  # 是否支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 是否支持前向后向自动微分
        dtypes=floating_types_and(torch.float16),  # 支持的数据类型，包括浮点数
        backward_dtypes=floating_types(),  # 反向传播时支持的数据类型
        dtypesIfCUDA=floating_types_and(torch.bfloat16, torch.float16),  # 在CUDA下支持的数据类型
        backward_dtypesIfCUDA=floating_types_and(torch.bfloat16, torch.float16),  # 在CUDA下反向传播支持的数据类型
    )
)


# TODO: upstream sample inputs to pytorch/pytorch.
# We are more comprehensive.
def sample_inputs_getitem(op_info, device, dtype, requires_grad, **kwargs):
    # Short for "advanced index"
    adv_idx = torch.LongTensor([[0, 1], [2, 3]])
    S = 5
    # self_dim, indices
    test_args = [
        (3, ([1, 2],)),  # 第一个测试参数：self_dim为3，args为([1, 2],)
        (3, (slice(0, 3),)),  # 第二个测试参数：self_dim为3，args为(slice(0, 3),)
        (3, ([slice(0, 3), 1],)),  # 第三个测试参数：self_dim为3，args为([slice(0, 3), 1],)
        (3, ([[0, 2, 3], [1, 3, 3], [0, 0, 2]],)),  # 第四个测试参数：self_dim为3，args为([[0, 2, 3], [1, 3, 3], [0, 0, 2]],)
        (3, ([[0, 0, 3], [1, 1, 3], [0, 0, 2]],)),  # 第五个测试参数：self_dim为3，args为([[0, 0, 3], [1, 1, 3], [0, 0, 2]],)
        (3, ([slice(None), slice(None), [0, 3]],)),  # 第六个测试参数：self_dim为3，args为([slice(None), slice(None), [0, 3]],)
        (3, ([slice(None), [0, 3], slice(None)],)),  # 第七个测试参数：self_dim为3，args为([slice(None), [0, 3], slice(None)],)
        (3, ([[0, 3], slice(None), slice(None)],)),  # 第八个测试参数：self_dim为3，args为([[0, 3], slice(None), slice(None)],)
        (3, ([[0, 3], [1, 2], slice(None)],)),  # 第九个测试参数：self_dim为3，args为([[0, 3], [1, 2], slice(None)],)
        (
            3,
            (
                [
                    [0, 3],
                ],
            ),
        ),  # 第十个测试参数：self_dim为3，args为([[0, 3],])
        (3, ([[0, 3], slice(None)],)),  # 第十一个测试参数：self_dim为3，args为([[0, 3], slice(None)],)
        (3, ([[0, 3], Ellipsis],)),  # 第十二个测试参数：self_dim为3，args为([[0, 3], Ellipsis],)
        (3, ([[0, 2, 3], [1, 3, 3], torch.LongTensor([0, 0, 2])],)),  # 第十三个测试参数：self_dim为3，args为([[0, 2, 3], [1, 3, 3], torch.LongTensor([0, 0, 2])],)
        (4, ([slice(None), adv_idx, adv_idx, slice(None)],)),  # 第十四个测试参数：self_dim为4，args为([slice(None), adv_idx, adv_idx, slice(None)],)
        (4, ([slice(None), adv_idx, slice(None), adv_idx],)),  # 第十五个测试参数：self_dim为4，args为([slice(None), adv_idx, slice(None), adv_idx],)
        (4, ([adv_idx, slice(None), slice(None), adv_idx],)),  # 第十六个测试参数：self_dim为4，args为([adv_idx, slice(None), slice(None), adv_idx],)
        (4, ([slice(None), slice(None), adv_idx, adv_idx],)),  # 第十七个测试参数：self_dim为4，args为([slice(None), slice(None), adv_idx, adv_idx],)
        (4, ([Ellipsis, adv_idx, adv_idx],)),  # 第十八个测试参数：self_dim为4，args为([Ellipsis, adv_idx, adv_idx],)
        (5, ([slice(None), slice(None), adv_idx, slice(None), adv_idx],)),  # 第十九个测试参数：self_dim为5，args为([slice(None), slice(None), adv_idx, slice(None), adv_idx],)
        (5, ([slice(None), slice(None), adv_idx, adv_idx, slice(None)],)),  # 第二十个测试参数：self_dim为5，args为([slice(None), slice(None), adv_idx, adv_idx, slice(None)],)
        (5, ([slice(None), slice(None), adv_idx, None, adv_idx, slice(None)],)),  # 第二十一个测试参数：self_dim为5，args为([slice(None), slice(None), adv_idx, None, adv_idx, slice(None)],)
        (6, ([slice(None), slice(None), slice(None), adv_idx, adv_idx],)),  # 第二十二个测试参数：self_dim为6，args为([slice(None), slice(None), slice(None), adv_idx, adv_idx],)
        (6, ([slice(None), slice(None), adv_idx, adv_idx, adv_idx],)),  # 第二十三个测试参数：self_dim为6，args为([slice(None), slice(None), adv_idx, adv_idx, adv_idx],)
        (6, ([slice(None), slice(None), None, adv_idx, adv_idx, adv_idx],)),  # 第二十四个测试参数：self_dim为6，args为([slice(None), slice(None), None, adv_idx, adv_idx, adv_idx],)
    ]

    def get_shape(dim):
        return tuple(S + i for i in range(dim))  # 返回一个元组，元素为S + 0, S + 1, ..., S + (dim - 1)

    return tuple(
        SampleInput(  # 返回一个SampleInput对象的元组
            make_tensor(  # 调用make_tensor函数
                get_shape(self_dim),  # 使用get_shape函数获取形状，self_dim为维度参数
                device=device,  # 设备参数
                dtype=dtype,  # 数据类型参数
                low=None,  # 最小值参数
                high=None,  # 最大值参数
                requires_grad=requires_grad,  # 是否需要梯度参数
            ),
            args=args,  # 将args作为SampleInput的参数
        )
        for self_dim, args in test_args  # 对于test_args中的每组self_dim和args，生成一个SampleInput对象
    )
# 添加一个操作信息对象到额外操作信息数据库中，描述 PyTorch 的 __getitem__ 方法的特性和约束条件。
# 此处在 functorch 变体下测试，支持的数据类型包括所有标准类型和 torch.bool、torch.float16、torch.bfloat16。
# 不支持输出张量（supports_out=False）、不支持原地自动求导（supports_inplace_autograd=False）、不支持脚本化（supports_scripting=False）。
# 使用 torch.Tensor.__getitem__ 作为操作的实现函数。
additional_op_db.append(
    OpInfo(
        "__getitem__",
        variant_test_name="functorch",
        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
        supports_out=False,
        supports_inplace_autograd=False,
        supports_scripting=False,
        op=torch.Tensor.__getitem__,
        assert_jit_shape_analysis=False,  # TODO: support index.Tensor()
        supports_forward_ad=True,
        sample_inputs_func=sample_inputs_getitem,
    )
)


# 发现 at::index_put 与 torch.index_put 不同...
# TODO: 弄清楚如何将这个功能贡献给上游
# 定义一个函数，生成用于 at::index_put 操作的样本输入数据
def sample_inputs_aten_index_put(op_info, device, dtype, requires_grad, **kwargs):
    # 创建一个函数，生成指定设备、数据类型、是否需要梯度的张量
    make_arg = partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )
    inputs = []
    adv_idx = torch.LongTensor([[0, 1], [2, 3]])
    # 添加额外的测试样本：每个测试样本包含张量的形状和索引的组合
    additional = [
        ((5, 6, 7, 8), [None, adv_idx, adv_idx, None]),
        ((5, 6, 7, 8), [None, adv_idx, None, adv_idx]),
        ((5, 6, 7, 8), [adv_idx, None, None, adv_idx]),
        ((5, 6, 7, 8), [None, None, adv_idx, adv_idx]),
        ((5, 6, 7, 8, 9), [None, None, adv_idx, None, adv_idx]),
        ((5, 6, 7, 8, 9), [None, None, adv_idx, adv_idx, None]),
        ((5, 6, 7, 8, 9, 10), [None, None, None, adv_idx, adv_idx]),
        ((5, 6, 7, 8, 9, 10), [None, None, adv_idx, adv_idx, adv_idx]),
    ]
    for self_shape, indices in additional:
        for broadcast_value in [False, True]:
            # 生成输入张量和操作参数
            inp = make_arg(self_shape)
            # 处理索引，为 None 的位置使用 slice(None)，以便正确选择张量的子集
            tmp_indices = [slice(None) if idx is None else idx for idx in indices]
            values_shape = inp[tmp_indices].shape
            # 如果需要广播值，则调整值的形状
            if broadcast_value:
                values_shape = values_shape[3:]
            values = make_arg(values_shape)
            # 将样本输入添加到输入列表中
            inputs.append(SampleInput(inp, args=(tuple(indices), values)))
    return inputs


# 定义一个函数，生成用于 torch.index_put 操作的样本输入数据
def sample_inputs_index_put(op_info, device, dtype, requires_grad, **kwargs):
    # 创建一个函数，生成指定设备、数据类型、是否需要梯度的张量
    make_arg = partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )
    # 创建一个函数，生成长整型张量索引
    make_idx = partial(
        make_tensor, dtype=torch.long, device=device, requires_grad=False
    )
    S = 5
    inputs = []
    for accumulate in [False, True]:
        # 对于每个布尔值累积的情况，创建样本输入对象并添加到输入列表中
        inputs.append(
            SampleInput(
                make_arg((S, S)),  # 创建一个大小为 (S, S) 的参数
                args=((make_idx((2,), low=0, high=4),), make_arg((2, S))),  # 指定位置参数和大小为 (2, S) 的参数
                kwargs=dict(accumulate=accumulate),  # 使用累积标志作为关键字参数
            )
        )

        # 将多维张量放置在索引位置
        inputs.append(
            SampleInput(
                make_arg((S, S, 2)),  # 创建一个大小为 (S, S, 2) 的参数
                args=((make_idx((3,), low=0, high=4),), make_arg((3, S, 2))),  # 指定位置参数和大小为 (3, S, 2) 的参数
                kwargs=dict(accumulate=accumulate),  # 使用累积标志作为关键字参数
            )
        )

        # 大小为 `0` 维度的值
        inputs.append(
            SampleInput(
                make_arg((S, 0)),  # 创建一个大小为 (S, 0) 的参数
                args=((make_idx((3,), low=0, high=4),), make_arg((3, 0))),  # 指定位置参数和大小为 (3, 0) 的参数
                kwargs=dict(accumulate=accumulate),  # 使用累积标志作为关键字参数
            )
        )

        # 标量值
        inputs.append(
            SampleInput(
                make_arg((S,)),  # 创建一个大小为 (S,) 的参数
                args=((make_idx((), low=0, high=S),), make_arg(())),  # 指定位置参数和大小为 () 的参数
                kwargs=dict(accumulate=accumulate),  # 使用累积标志作为关键字参数
            )
        )

        # 当累积为假且设备为 "cuda" 时，CUDA 和累积不兼容
        # 参考: https://github.com/pytorch/pytorch/issues/72053
        if not accumulate and device == "cuda":
            # 广播 `values`
            inputs.append(
                SampleInput(
                    make_arg((S, S)),  # 创建一个大小为 (S, S) 的参数
                    args=((make_idx((2,), low=0, high=S),), make_arg((S,))),  # 指定位置参数和大小为 (S,) 的参数
                    kwargs=dict(accumulate=accumulate),  # 使用累积标志作为关键字参数
                )
            )

    return inputs
additional_op_db.append(
    OpInfo(
        "index_put",
        variant_test_name="functorch",
        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
        supports_out=False,
        sample_inputs_func=sample_inputs_index_put,
        supports_forward_ad=True,
    )
)
additional_op_db.append(
    OpInfo(
        "ops.aten.index_put",
        variant_test_name="functorch",
        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
        supports_out=False,
        sample_inputs_func=sample_inputs_aten_index_put,
        supports_forward_ad=True,
    )
)

# 定义生成 `masked_fill` 操作的输入样本的函数
def sample_inputs_masked_fill(op_info, device, dtype, requires_grad, **kwargs):
    S = 3
    make_arg = partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )

    # 生成不同的样本输入
    yield SampleInput(make_arg((S, S)), args=(torch.randn(S, S, device=device) > 0, 10))
    yield SampleInput(make_arg((S, S)), args=(torch.randn(S, device=device) > 0, 10))
    yield SampleInput(make_arg(()), args=(torch.randn((), device=device) > 0, 10))
    yield SampleInput(make_arg((S, S)), args=(torch.randn((), device=device) > 0, 10))
    yield SampleInput(
        make_arg((S,)),
        args=(torch.randn(S, S, device=device) > 0, 10),
        broadcasts_input=True,
    )

additional_op_db.append(
    OpInfo(
        "masked_fill",
        variant_test_name="functorch_Scalar_only",
        dtypes=all_types_and_complex_and(
            torch.bool, torch.half, torch.bfloat16, torch.chalf
        ),
        sample_inputs_func=sample_inputs_masked_fill,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        check_batched_forward_grad=False,
        supports_out=False,
    )
)

# 定义生成 `new_zeros_with_same_feature_meta` 操作的输入样本的函数
def sample_inputs_new_zeros_with_same_feature_meta(
    op_info, device, dtype, requires_grad, **kwargs
):
    make_arg = partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )
    # 定义不同的输入形状组合
    matrix = [
        # tangent, base, num_tangent_bdims
        ([5], [2, 3], 0),
        ([2, 3], [2, 3], 0),
        ([5], [2], 0),
        ([1, 0, 2], [1, 2], 0),
        ([], [1, 2], 0),
        ([8, 7, 5], [2, 3, 11], 1),
        ([6, 7, 5], [2, 3, 4], 2),
        ([6, 4], [3], 2),
    ]
    results = []
    # 遍历矩阵，生成样本输入
    for tangent_shape, base_shape, num_tangent_bdims in matrix:
        tangent = make_arg(tangent_shape)
        base = make_arg(base_shape)
        results.append(
            SampleInput(
                tangent,
                args=(base,),
                kwargs=dict(self_num_batch_dims=num_tangent_bdims),
            )
        )
    return results

additional_op_db.append(
    OpInfo(
        "ops.aten._new_zeros_with_same_feature_meta",  # 定义一个 OpInfo 对象，指定操作名称为 "_new_zeros_with_same_feature_meta"
        variant_test_name="functorchonly",  # 设置变体测试名称为 "functorchonly"
        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),  # 设置数据类型，包括所有类型、复杂类型以及 torch.bool、torch.float16、torch.bfloat16
        supports_out=False,  # 指示不支持输出参数
        supports_autograd=False,  # 指示不支持自动求导
        supports_forward_ad=False,  # 指示不支持前向自动求导
        sample_inputs_func=sample_inputs_new_zeros_with_same_feature_meta,  # 设置样本输入函数为 sample_inputs_new_zeros_with_same_feature_meta
    )
)

def sample_inputs_conversion(op_info, device, dtype, requires_grad, **kwargs):
    # 使用偏函数构造一个创建张量的函数 make_arg，固定了 dtype、device 和 requires_grad 参数
    make_arg = partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )
    # 定义张量的形状和内存格式选项的组合
    shapes = ((), (2, 3))
    memory_format_options = [None, torch.contiguous_format]
    # 使用 itertools.product 生成形状和内存格式的所有可能组合
    for shape, memory_format in itertools.product(shapes, memory_format_options):
        # 生成 SampleInput 对象，其中包含根据 shape 创建的张量和可能的内存格式参数
        yield SampleInput(
            make_arg(shape),
            kwargs={"memory_format": memory_format} if memory_format else {},
        )


additional_op_db.extend(
    ]
)
```