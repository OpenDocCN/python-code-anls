# `.\pytorch\test\functorch\common_utils.py`

```
# 导入所需的模块和库
import itertools  # 导入 itertools 模块，用于迭代操作
import os  # 导入 os 模块，用于操作系统相关功能
import unittest  # 导入 unittest 模块，用于编写和运行单元测试
from collections import namedtuple  # 从 collections 模块导入 namedtuple 类型

# 导入 functorch_additional_op_db 模块中的 additional_op_db
from functorch_additional_op_db import additional_op_db

import torch  # 导入 PyTorch 深度学习框架
import torch.utils._pytree as pytree  # 导入 pytree 模块中的 _pytree 功能

# 导入 functorch 模块中的 vmap 函数
from functorch import vmap

# 导入 torch.testing._internal.autograd_function_db 模块中的 autograd_function_db
from torch.testing._internal.autograd_function_db import autograd_function_db

# 导入 torch.testing._internal.common_device_type 模块中的 toleranceOverride
from torch.testing._internal.common_device_type import toleranceOverride

# 导入 torch.testing._internal.common_methods_invocations 模块中的 DecorateInfo, op_db
from torch.testing._internal.common_methods_invocations import DecorateInfo, op_db

# 导入 torch.testing._internal.common_modules 模块中的 module_db
from torch.testing._internal.common_modules import module_db

# 检查是否在 FBCODE 环境中
IS_FBCODE = os.getenv("FUNCTORCH_TEST_FBCODE") == "1"


# 定义一个循环函数，用于批处理操作
def loop(op, in_dims, out_dim, batch_size, *batched_args, **kwarg_values):
    outs = []  # 存储每个批次计算的结果
    out_spec = None  # 初始化输出的规范

    # 遍历每个批次
    for idx in range(batch_size):
        flat_args, args_spec = pytree.tree_flatten(batched_args)  # 扁平化输入参数
        flat_dims, dims_spec = pytree.tree_flatten(in_dims)  # 扁平化输入维度
        assert args_spec == dims_spec  # 断言参数规范与维度规范相同

        # 根据索引选择参数
        new_args = [
            a.select(in_dim, idx) if in_dim is not None else a
            for a, in_dim in zip(flat_args, flat_dims)
        ]
        
        # 执行操作并获取输出结果
        out = op(*pytree.tree_unflatten(new_args, args_spec), **kwarg_values)

        flat_out, out_spec = pytree.tree_flatten(out)  # 扁平化输出结果
        outs.append(flat_out)  # 将扁平化的输出结果添加到 outs 中

    # 将 outs 转置为每批次的结果列表，并转换为张量后返回
    outs = zip(*outs)
    result = [torch.stack(out_lst) for out_lst in outs]
    return pytree.tree_unflatten(result, out_spec)


# 类似于 loop 函数的辅助函数，但用于两层 vmap 操作
def loop2(
    op,
    in_dims1,
    in_dims2,
    out_dim1,
    out_dim2,
    batch_size1,
    batch_size2,
    *batched_args,
    **kwarg_values,
):
    outs = []  # 存储每个批次计算的结果
    flat_args, args_spec = pytree.tree_flatten(batched_args)  # 扁平化输入参数
    flat_dims1, dims_spec1 = pytree.tree_flatten(in_dims1)  # 扁平化第一组输入维度
    flat_dims2, dims_spec2 = pytree.tree_flatten(in_dims2)  # 扁平化第二组输入维度
    assert args_spec == dims_spec1  # 断言参数规范与第一组维度规范相同
    assert args_spec == dims_spec2  # 断言参数规范与第二组维度规范相同
    assert len(flat_dims1) == len(flat_dims2)  # 断言第一组维度与第二组维度长度相同

    # 遍历第一组批次
    for idx1 in range(batch_size1):
        out_split = []  # 存储第二层循环的输出结果
        arg_split = [
            a.select(in_dim1, idx1) if in_dim1 is not None else a
            for a, in_dim1 in zip(flat_args, flat_dims1)
        ]

        # 遍历第二组批次
        for idx2 in range(batch_size2):
            # 根据索引选择参数
            new_args = [
                a.select(in_dim, idx2) if in_dim is not None else a
                for a, in_dim in zip(arg_split, flat_dims2)
            ]
            
            # 执行操作并获取输出结果
            out = op(*pytree.tree_unflatten(new_args, args_spec), **kwarg_values)
            out_split.append(out)  # 将输出结果添加到 out_split 中
        
        outs.append(out_split)  # 将每一层循环的输出结果添加到 outs 中

    loop_out = []
    # 遍历输出列表中的每个元素 out_split
    for out_split in outs:
        # 检查 out_split 的第一个元素是否为 torch.Tensor 类型
        if isinstance(out_split[0], torch.Tensor):
            # 如果是张量，则沿着第一维度 out_dim1 进行堆叠，添加到 loop_out 中
            loop_out.append(torch.stack(out_split, out_dim1))
        else:
            # 如果不是张量，则对 out_split 中的每个索引进行遍历
            new_out = []
            for idx in range(len(out_split[0])):
                # 将 out_split 中每个索引位置的张量按照 out_dim1 进行堆叠，添加到 new_out 中
                new_out.append(torch.stack([i[idx] for i in out_split], out_dim1))
            # 将 new_out 添加到 loop_out 中
            loop_out.append(new_out)

    # 初始化新列表 new_out
    new_out = []
    # 检查 loop_out 是否为 torch.Tensor 类型
    if isinstance(loop_out, torch.Tensor):
        # 如果是张量，则沿着第二维度 out_dim2 进行堆叠，赋值给 new_out
        new_out = torch.stack(loop_out, out_dim2)
    else:
        # 如果不是张量，则对 loop_out 中的每个索引进行遍历
        for idx in range(len(loop_out[0])):
            # 将 loop_out 中每个索引位置的张量按照 out_dim2 进行堆叠，添加到 new_out 中
            new_out.append(torch.stack([i[idx] for i in loop_out], out_dim2))
    # 返回堆叠后的结果 new_out
    return new_out
# 检查是否提供了 inplace_variant 参数，如果未提供则返回 False
def is_valid_inplace_sample_input(sample_input, op, inplace_variant):
    if inplace_variant is None:
        return False
    # 如果 sample_input 允许广播，则返回 False
    if sample_input.broadcasts_input:
        return False
    # 如果 sample_input.input 不是 torch.Tensor 对象，则返回 False
    if not isinstance(sample_input.input, torch.Tensor):
        return False

    # 检查输入的数据类型是否与操作的输出数据类型匹配
    args = (sample_input.input,) + sample_input.args
    kwargs = sample_input.kwargs
    output_dtype = op(*args, **kwargs).dtype
    return sample_input.input.dtype == output_dtype


# 这个函数实现了一个记忆化（memoization）功能，用来缓存函数的计算结果，避免重复计算
# 注意事项：
# - memo 字典用来存储参数与对应结果的键值对
# - wrapped 函数包装了原始函数 fn，并在需要时将结果存入 memo 中
def memoize(fn):
    memo = {}

    def wrapped(*args):
        if args not in memo:
            memo[args] = fn(*args)
        return memo[args]

    return wrapped


# 这个函数生成了所有可能的 batch 维度选择，返回一个元组
# 注意事项：
# - 时间复杂度为 O(2 ** num_tensors)，因为它生成了所有可能的组合
# - 返回的元组中不包含最后一个全为 None 的选择
# - 使用 memoize 装饰器进行了记忆化，避免重复计算
@memoize
def get_bdim_choices(num_tensors):
    choices = []

    # 添加全为 0 的选择
    choices.append((0,) * num_tensors)

    # 添加所有 (-1, None) 的排列组合
    options = (-1, None)
    choices.extend(itertools.product(options, repeat=num_tensors))

    # 断言最后一个选择是全为 None
    assert choices[-1] == (None,) * num_tensors
    return tuple(choices[:-1])


# 这个函数生成了所有可能的 batch 维度选择，返回一个元组，专门用于 batch normalization
# 注意事项：
# - 时间复杂度为 O(2 ** num_tensors)，因为它生成了所有可能的组合
# - 参数 running_mean 和 running_var 用于限制选择的条件
# - 如果 running_mean 或 running_var 为 None，则排除 (0,) * num_tensors 的选择
def get_bdim_choices_batch_norm(
    num_tensors, _, running_mean=None, running_var=None, *args
):
    choices = []
    options = (-1, None)

    if running_mean is None or running_var is None:
        # 如果 running_mean 或 running_var 为 None，则添加特定选择
        choices.append((None,) + (0,) * (num_tensors - 1))
        for choice in itertools.product(options, repeat=num_tensors - 1):
            choices.append((None,) + choice)

    else:
        # 否则，测试所有其他情况的选择
        choices.append((0,) * num_tensors)
        for choice in itertools.product(options, repeat=num_tensors):
            input_bdim = choice[0]
            running_mean_bdim = choice[1]
            running_var_bdim = choice[2]
            # 如果 input_bdim 为真而 running_mean_bdim 或 running_var_bdim 任一为假，则跳过此选择
            if input_bdim and (not running_mean_bdim or not running_var_bdim):
                continue
            choices.append(choice)

    # 断言最后一个选择是全为 None
    assert choices[-1] == (None,) * num_tensors
    return tuple(choices[:-1])


# 给定一个张量和 batch 维度，添加 batch 维度并返回结果
# 注意事项：
# - 确保 bdim 只能是 0 或 -1
# - 确保 arg 是 torch.Tensor 类型
# - 如果 bdim 为 0，则在 shape 中插入 batch_size，然后使用 repeat 方法进行重复
# - 返回结果是一个元组，包含添加 batch 维度后的张量和 bdim 的值
def add_batch_dim(arg, bdim, batch_size=3):
    assert bdim == 0 or bdim == -1
    assert isinstance(arg, torch.Tensor)
    if bdim == 0:
        shape = [1] * len(arg.shape)
        shape.insert(bdim, batch_size)
        return (arg.repeat(shape), bdim)
    # 如果 bdim 的值为 -1，执行以下操作
    if bdim == -1:
        # 将 arg 张量在最后一个维度上添加一个维度，并扩展为与 batch_size 相同维度的张量，使其连续存储
        arg = arg.unsqueeze(-1).expand(*arg.shape, batch_size).contiguous()
        # 返回调整后的 arg 张量和 bdim 值
        return (arg, bdim)
# 构建输入维度的函数，根据给定的维度选择和是否是张量，生成结果元组
def construct_in_dims(bdim_choice_for_tensors, is_tensors):
    result = []  # 初始化结果列表
    bdim = iter(bdim_choice_for_tensors)  # 迭代器，用于逐个获取维度选择
    for is_tensor in is_tensors:  # 遍历每个输入是否是张量
        if not is_tensor:  # 如果不是张量
            result.append(None)  # 添加 None 到结果列表
            continue
        result.append(next(bdim))  # 添加下一个维度选择到结果列表
    return tuple(result)  # 返回结果列表转换为元组


# 检查操作名和关键字参数值，判断是否是批量归一化训练
def is_batch_norm_training(op_name, kwarg_values):
    batch_norm_fns = (  # 定义批量归一化函数名元组
        "nn.functional.batch_norm",
        "nn.functional.instance_norm",
    )
    if op_name not in batch_norm_fns:  # 如果操作名不在批量归一化函数名元组中
        return False  # 返回 False

    # 批量归一化和实例归一化需要值为普通布尔类型
    default_training = (op_name == "nn.functional.instance_norm")  # 实例归一化默认为训练状态，批量归一化不是
    is_training = tuple(  # 获取所有布尔类型参数值的元组
        arg for arg in tuple(kwarg_values.values()) if isinstance(arg, bool)
    )
    if len(is_training) == 0:  # 如果没有布尔类型参数值
        return default_training  # 返回默认训练状态
    else:
        assert len(is_training) == 1  # 确保只有一个布尔类型参数值
        return is_training[0]  # 返回布尔类型参数值


# 生成输入映射的函数，根据参数值和是否批量归一化和训练，生成批量化的输入组合
def generate_vmap_inputs(
    arg_values, kwarg_values, is_batch_norm_and_training=False, batch_size=2
):
    flat_args, arg_spec = pytree.tree_flatten(tuple(arg_values))  # 扁平化输入参数并获取参数规范
    is_tensors = [isinstance(a, torch.Tensor) for a in flat_args]  # 判断每个参数是否是张量
    num_tensors = sum(is_tensors)  # 统计张量参数个数

    # 对于批量归一化，如果只有一个输入张量，不能进行批量处理
    if num_tensors == 1 and is_batch_norm_and_training:
        return  # 如果满足条件，直接返回

    # 根据是否批量归一化和训练状态，选择不同的维度选择函数
    bdim_choices = (
        get_bdim_choices_batch_norm(num_tensors, *arg_values)
        if is_batch_norm_and_training
        else get_bdim_choices(num_tensors)
    )

    # 缓存函数，用于获取批量化的参数
    @memoize
    def get_batched_arg(arg, bdim):
        assert isinstance(arg, torch.Tensor)  # 断言参数必须是张量
        assert bdim is not None  # 断言维度选择不为 None
        result, _ = add_batch_dim(arg, bdim, batch_size)  # 添加批量维度到参数中
        return result  # 返回批量化后的参数

    # 遍历所有维度选择
    for bdim_choice in bdim_choices:
        # 构建输入维度的扁平化列表
        flat_in_dims = construct_in_dims(bdim_choice, is_tensors)

        # 将每个参数批量化或保持原样，生成扁平化的批量化参数列表
        flat_batched_args = tuple(
            arg if in_dim is None else get_batched_arg(arg, in_dim)
            for arg, in_dim in zip(flat_args, flat_in_dims)
        )

        # 将扁平化的批量化参数列表恢复成原始结构
        batched_args = pytree.tree_unflatten(flat_batched_args, arg_spec)
        in_dims = pytree.tree_unflatten(flat_in_dims, arg_spec)

        # 返回批量化的参数、输入维度和关键字参数值
        yield batched_args, in_dims, kwarg_values


# 如果输入是张量，进行克隆操作，否则返回原始输入
def clone_if_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.clone()  # 返回张量的克隆
    return x  # 返回原始输入


# 用于比较 `vmap` 和 `for-loop` 版本输出的辅助函数
def _compute_quantities_for_vmap_test(
    op,
    orig_batched_args,
    orig_kwarg_values,
    in_dims,
    out_dim,
    batch_size,
    compute_loop_out=True,
    clone_inputs=False,
):
    # 如果需要克隆输入，则克隆所有输入参数
    def maybe_clone_inputs():
        if clone_inputs:
            batched_args = pytree.tree_map(clone_if_tensor, orig_batched_args)
            kwarg_values = pytree.tree_map(clone_if_tensor, orig_kwarg_values)
            return batched_args, kwarg_values
        return orig_batched_args, orig_kwarg_values  # 否则返回原始输入参数
    # 从 maybe_clone_inputs 函数中获取批处理参数和关键字参数值
    batched_args, kwarg_values = maybe_clone_inputs()

    # 如果 compute_loop_out 为真，调用 loop 函数进行循环计算
    if compute_loop_out:
        loop_out = loop(op, in_dims, out_dim, batch_size, *batched_args, **kwarg_values)
    else:
        loop_out = None

    # 用于调试生成的操作结果
    # from functorch import make_fx
    # def f(a):
    #     return op(a)
    # t = make_fx(vmap(f, in_dims=in_dims, out_dims=out_dim))(*batched_args, **kwarg_values)
    # print(in_dims, [arg.shape for arg in batched_args], kwarg_values)

    # 重新获取批处理参数和关键字参数值，可能因为上述代码被调试而重新获取
    batched_args, kwarg_values = maybe_clone_inputs()

    # 使用 vmap 函数对 op 进行批处理映射计算，并得到批处理结果
    batched_out = vmap(op, in_dims=in_dims, out_dims=out_dim)(
        *batched_args, **kwarg_values
    )

    # 测试情况，其中我们分发到一个没有 bdims 的批处理规则
    # 这应该由自动生成的基础设施处理。对于手动添加的 vmap 支持，您可能需要特别处理这种情况。
    # 定义一个函数，在输入是张量时添加一个新的维度
    def add_bdim_if_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.unsqueeze(1)
        return x

    # 定义一个函数 f，对参数进行操作 op 的调用
    def f(dummy, *args, **kwargs):
        return op(*args, **kwargs)

    # 创建一个大小为 (batch_size, 1) 的张量 dummy
    dummy = torch.ones(batch_size, 1)

    # 使用 pytree.tree_map 将 add_bdim_if_tensor 函数映射到 batched_out 的每个元素
    vmapvmap_expected = pytree.tree_map(add_bdim_if_tensor, batched_out)

    # 设置内部和外部输入维度
    inner_in_dims = (0,) + pytree.tree_map(lambda x: None, in_dims)
    outer_in_dims = (0,) + in_dims

    # 再次获取批处理参数和关键字参数值
    batched_args, kwarg_values = maybe_clone_inputs()

    # 使用 vmap 函数对 vmap(f, inner_in_dims) 进行批处理映射计算，并得到最终的批处理结果
    vmapvmap_output = vmap(vmap(f, inner_in_dims), outer_in_dims)(
        dummy, *batched_args, **kwarg_values
    )

    # 生成器函数的返回值，包括批处理结果、循环计算结果、双重批处理映射计算的输出和期望输出
    yield (batched_out, loop_out, vmapvmap_output, vmapvmap_expected)
# 定义一个生成器函数，用于测试 vmap 功能时计算数量
def compute_quantities_for_vmap_test(
    op,
    orig_batched_args,
    orig_kwarg_values,
    in_dims,
    out_dim=0,
    batch_size=2,
    compute_loop_out=True,
    clone_inputs=False,
):
    # 使用 `_compute_quantities_for_vmap_test` 函数生成的结果进行迭代
    for quantities in _compute_quantities_for_vmap_test(
        op,
        orig_batched_args,
        orig_kwarg_values,
        in_dims,
        out_dim,
        batch_size,
        compute_loop_out,
        clone_inputs,
    ):
        # 生成两个元组，每个元组包含两个量
        yield (quantities[0], quantities[1])
        yield (quantities[2], quantities[3])


# 获取 fallback 和 vmap exhaustive 的结果
def get_fallback_and_vmap_exhaustive(
    op,
    arg_values,
    kwarg_values,
    is_batch_norm_and_training=False,
    compute_loop_out=True,
):
    # 初始化输出维度和批处理大小
    out_dim = 0
    batch_size = 2

    # 定义一个函数，用于将张量进行批处理
    def make_batched(t):
        if isinstance(t, torch.Tensor):
            shape = list(t.shape)
            shape.insert(out_dim, batch_size)
            return t.expand(*shape)
        return t

    # 计算未批处理和批处理后的预期值
    expected_unbatched = op(*arg_values, **kwarg_values)
    expected_batched = pytree.tree_map(make_batched, expected_unbatched)
    
    # 生成 vmap 输入的生成器，根据 is_batch_norm_and_training 决定是否进行批处理
    generator = generate_vmap_inputs(
        arg_values, kwarg_values, is_batch_norm_and_training
    )
    
    # 遍历生成器产生的批处理参数、输入维度和关键字参数
    for batched_args, in_dims, kwarg_values in generator:
        # 对于 `_compute_quantities_for_vmap_test` 生成的每个结果数量进行迭代
        for quantities in _compute_quantities_for_vmap_test(
            op,
            batched_args,
            kwarg_values,
            in_dims,
            out_dim,
            batch_size,
            compute_loop_out=False,
        ):
            # 断言第二个量为 None
            assert quantities[1] is None
            # 生成两个元组，分别包含第一个量和预期的批处理结果
            yield (quantities[0], expected_batched)
            yield (quantities[2], quantities[3])


# 检查 opinfo 是否在字典 d 中
def opinfo_in_dict(opinfo, d):
    return (opinfo.name in d) or (f"{opinfo.name}.{opinfo.variant_test_name}" in d)


# 定义一个命名元组 DecorateMeta，包含几种装饰信息
DecorateMeta = namedtuple(
    "DecorateMeta",
    [
        "op_name",
        "variant_name",
        "decorator",
        "device_type",
        "dtypes",
    ],
)


# 装饰函数，返回一个 DecorateMeta 元组
def decorate(
    op_name, variant_name="", *, decorator=None, device_type=None, dtypes=None
):
    # 断言 decorator 参数不为 None
    assert decorator is not None
    # 返回一个 DecorateMeta 元组
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=decorator,
        device_type=device_type,
        dtypes=dtypes,
    )


# 返回一个装饰为 unittest.expectedFailure 的 DecorateMeta 元组
def xfail(op_name, variant_name="", *, device_type=None, dtypes=None):
    return decorate(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.expectedFailure,
        device_type=device_type,
        dtypes=dtypes,
    )


# 跳过测试用例的装饰函数
def skip(op_name, variant_name="", *, device_type=None, dtypes=None):
    # 调用decorate函数，并传入以下参数：
    # op_name: 操作名称
    # variant_name: 变体名称
    # decorator: 使用unittest.skip("Skipped!")装饰器，表示测试跳过
    # device_type: 设备类型
    # dtypes: 数据类型列表
    return decorate(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.skip("Skipped!"),
        device_type=device_type,
        dtypes=dtypes,
    )
# 定义函数 `skipOps`，用于为指定测试用例名和基本测试名跳过一些操作。
def skipOps(test_case_name, base_test_name, to_skip):
    # 将三个操作信息列表合并成一个大列表
    all_opinfos = op_db + additional_op_db + autograd_function_db
    # 遍历要跳过的每个修饰元信息
    for decorate_meta in to_skip:
        # 找到所有符合条件的操作信息对象
        matching_opinfos = [
            o
            for o in all_opinfos
            if o.name == decorate_meta.op_name
            and o.variant_test_name == decorate_meta.variant_name
        ]
        # 确保找到了至少一个匹配的操作信息对象
        assert len(matching_opinfos) > 0, f"Couldn't find OpInfo for {decorate_meta}"
        # 确保只有一个唯一的操作信息对象符合条件
        assert len(matching_opinfos) == 1, (
            "OpInfos should be uniquely determined by their (name, variant_name). "
            f"Got more than one result for ({decorate_meta.op_name}, {decorate_meta.variant_name})"
        )
        # 取出唯一的操作信息对象
        opinfo = matching_opinfos[0]
        # 将修饰信息添加到操作信息对象的修饰器列表中
        decorators = list(opinfo.decorators)
        new_decorator = DecorateInfo(
            decorate_meta.decorator,
            test_case_name,
            base_test_name,
            device_type=decorate_meta.device_type,
            dtypes=decorate_meta.dtypes,
        )
        decorators.append(new_decorator)
        opinfo.decorators = tuple(decorators)

    # 定义一个装饰器函数 `wrapped`，不对 fn 进行任何修改
    def wrapped(fn):
        return fn

    # 返回装饰后的函数 `wrapped`
    return wrapped


# 定义函数 `decorateForModules`，用于为给定的模块类列表添加装饰器。
def decorateForModules(decorator, module_classes, device_type=None, dtypes=None):
    # 定义装饰器函数 `wrapped`，不对 fn 进行任何修改
    def wrapped(
        fn,
        module_classes=module_classes,
        decorator=decorator,
        device_type=device_type,
        dtypes=dtypes,
    ):
        # 将函数名分割成两部分，确保函数只应用于测试类的测试函数
        name_parts = fn.__qualname__.split(".")
        assert (
            len(name_parts) == 2
        ), "Decorator only applies to a test function of a test class"
        test_case_name, base_test_name = name_parts
        # 遍历每个模块类，为其找到匹配的模块信息对象并添加装饰器信息
        for module_cls in module_classes:
            matching_module_infos = [m for m in module_db if m.module_cls == module_cls]
            assert (
                len(matching_module_infos) == 1
            ), f"Couldn't find single ModuleInfo for {module_cls}"
            module_info = matching_module_infos[0]
            decorators = list(module_info.decorators)
            new_decorator = DecorateInfo(
                decorator,
                test_case_name,
                base_test_name,
                device_type=device_type,
                dtypes=dtypes,
            )
            decorators.append(new_decorator)
            module_info.decorators = tuple(decorators)
        return fn

    # 返回装饰后的函数 `wrapped`
    return wrapped


# 定义装饰器工厂函数 `expectedFailureIf`，根据条件决定是否标记为预期的测试失败。
def expectedFailureIf(condition):
    def decorator(fn):
        if condition:
            return unittest.expectedFailure(fn)
        return fn

    # 返回装饰器函数 `decorator`
    return decorator


# 定义函数 `tol2`，返回一个包含操作名、变体名、覆盖字典和设备类型的元组。
def tol2(op_name, variant_name, override_dct, *, device_type=None):
    return (op_name, variant_name, override_dct, device_type)


# 定义函数 `tol1`，调用 `tol2` 函数并传递空字符串作为变体名。
def tol1(op_name, override_dct, *, device_type=None):
    return tol2(op_name, "", override_dct, device_type=device_type)


# 定义函数 `opsToleranceOverride`，用于操作容差覆盖。
def opsToleranceOverride(test_case_name, base_test_name, overrides):
    # 将两个操作信息列表合并成一个大列表
    all_opinfos = op_db + additional_op_db
    # 对于每个覆盖项，解构覆盖信息：操作名、变体名、覆盖值、设备类型
    for override in overrides:
        # 解构操作名、变体名、覆盖值、设备类型
        op_name, variant_name, override, device_type = override
        
        # 在所有操作信息中查找匹配的操作信息对象
        matching_opinfos = [
            o
            for o in all_opinfos
            if o.name == op_name and o.variant_test_name == variant_name
        ]
        
        # 断言只能找到一个匹配的操作信息对象，否则抛出异常
        assert len(matching_opinfos) == 1, f"Couldn't find OpInfo for {override}"
        
        # 获取匹配的操作信息对象
        opinfo = matching_opinfos[0]
        
        # 复制操作信息对象的装饰器列表，添加新的装饰器信息
        decorators = list(opinfo.decorators)
        decorators.append(
            DecorateInfo(
                toleranceOverride(override),  # 调用 toleranceOverride 函数创建装饰信息
                test_case_name,
                base_test_name,
                device_type=device_type,
            )
        )
        
        # 将更新后的装饰器列表重新赋值给操作信息对象
        opinfo.decorators = tuple(decorators)

    # 定义一个装饰器函数 wrapped，它不会对传入的函数 fn 进行任何修改
    def wrapped(fn):
        return fn

    # 返回装饰器函数 wrapped，用于装饰函数或方法
    return wrapped
class DisableVmapFallback:
    # 进入上下文管理器时调用的方法，禁用 vmap 回退功能
    def __enter__(self):
        # 保存当前 vmap 回退功能的状态，并设置为禁用
        self.prev_state = torch._C._functorch._is_vmap_fallback_enabled()
        torch._C._functorch._set_vmap_fallback_enabled(False)

    # 退出上下文管理器时调用的方法
    def __exit__(self, *ignored):
        # 恢复之前保存的 vmap 回退功能状态
        torch._C._functorch._set_vmap_fallback_enabled(self.prev_state)


# 检查 vmap 回退功能的辅助函数
def check_vmap_fallback(test_case, thunk, opinfo, dry_run=False):
    try:
        # 使用 DisableVmapFallback 上下文管理器禁用 vmap 回退功能
        with DisableVmapFallback():
            # 执行 thunk 函数，该函数可能触发异常
            thunk()
    except Exception:
        # 如果不是 dry_run 模式，将异常重新抛出
        if not dry_run:
            raise
        # 在 dry_run 模式下，根据 opinfo 的信息输出 xfail 信息
        if opinfo.variant_test_name:
            print(f"xfail('{opinfo.name}', '{opinfo.variant_test_name}'),")
        else:
            print(f"xfail('{opinfo.name}'),")
```