# `.\pytorch\torch\testing\_internal\composite_compliance.py`

```py
# 忽略 mypy 错误检查
# mypy: ignore-errors

import torch  # 导入 torch 库
from torch import Tensor  # 从 torch 导入 Tensor 类
import itertools  # 导入 itertools 模块

from torch.utils._python_dispatch import TorchDispatchMode  # 从 torch.utils._python_dispatch 导入 TorchDispatchMode
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten  # 从 torch.utils._pytree 导入 tree_map, tree_flatten, tree_unflatten
from torch.utils import _pytree as pytree  # 从 torch.utils 导入 _pytree 并重命名为 pytree
from functools import partial  # 从 functools 导入 partial
from torch.utils._mode_utils import no_dispatch, all_same_mode  # 从 torch.utils._mode_utils 导入 no_dispatch, all_same_mode
import torch.autograd.forward_ad as fwAD  # 导入 torch.autograd.forward_ad 模块作为 fwAD
from typing import Callable  # 从 typing 导入 Callable
import re  # 导入正则表达式模块 re


def check_attr_consistency(wrapper_tensor, metadata_name, metadata_accessor):
    # 获取包装张量的元素
    elem = wrapper_tensor.elem
    # 使用元数据访问器获取包装张量的元数据
    metadata_wrapper_tensor = metadata_accessor(wrapper_tensor)
    # 使用元数据访问器获取元素的元数据
    metadata_elem = metadata_accessor(elem)
    # 如果包装张量的元数据与元素的元数据相同，返回
    if metadata_wrapper_tensor == metadata_elem:
        return
    # 否则，抛出运行时错误，说明该操作符未遵循 CompositeCompliant 标准
    raise RuntimeError(
        f"This operator is not Composite Compliant: the "
        f"{metadata_name} of the tensor was modified directly without "
        f"going through the PyTorch dispatcher.")

def check_metadata_consistency(wrapper_tensor, CCT):
    # CCT: CompositeCompliantTensor 类，通过 generate_cct 生成
    if not isinstance(wrapper_tensor, CCT):
        return
    # 定义需要检查的元数据及其访问器
    things_to_check = {
        'shape': Tensor.size,
        'dtype': lambda x: x.dtype,
        'device': lambda x: x.device,
        'numel': Tensor.numel,
        'stride': Tensor.stride,
        'storage_offset': Tensor.storage_offset,
    }
    # 遍历所有需要检查的元数据
    for metadata_name, metadata_accessor in things_to_check.items():
        # 调用 check_attr_consistency 检查每个元数据的一致性
        check_attr_consistency(wrapper_tensor, metadata_name, metadata_accessor)

def is_view_fn(func):
    # 判断函数是否为视图函数，基于其重载包名称
    return func.overloadpacket.__name__ in {
        'as_strided',
        'detach',
        'diagonal',
        'expand',
        'expand_as',
        'movedim',
        'narrow',
        'permute',
        'select',
        'squeeze',
        'transpose',
        't',
        'real',
        'imag',
        'view_as_real',
        'view_as_complex',
        'unflatten',
        'unfold',
        'unsqueeze',
        'view',
        'view_as',
        'unbind',
        'split',
        'split_with_sizes',
        'vsplit',
        'hsplit',
        'tensor_split',
        'chunk',
        'swapaxes',
        'slice',
        '_reshape_alias',
        '_unsafe_view',
        '_conj',
        'alias',
    }

# 手动填充，来自 native_functions 的 inplace_view: True 的函数。
# 将来我们可能能够直接获取该列表
def is_inplace_view_fn(func):
    return func.overloadpacket.__name__ in {
        'as_strided_',
        'detach_',
        'squeeze_',
        'swapaxes_',
        'swapdims_',
        't_',
        'transpose_',
        'unsqueeze_',
    }


# 深入剖析，帮助我们了解函数的特性
def is_inplace(func):
    name = func.overloadpacket.__name__
    # 正则表达式匹配以 '__i' 开头的函数名称，认为其为原位函数
    if re.match('__i.+__', name):
        return True
    # 正则表达式匹配以 '__' 开头的函数名称，认为其不是原位函数
    if re.match('__.+__', name):
        return False
    # 其他情况下，判断函数名的最后一个字符是否为 '_'，如果是，则认为其为原位函数
    return name[-1] == '_'


def generate_cct_and_mode(autograd_view_consistency=True):
    # 此函数返回一个新的类 CompositeCompliantTensor
    # The two arguments control the behaviour described below.

    # autograd_view_consistency:
    #   If True, alias result using `set_` if func returns a view
    #   (See Note [Alias Result]).
    #   Since Forward AD doesn't work with `set_`
    #   we disable it by setting alias to False.

    # 定义一个新的子类 CompositeCompliantTensor，继承自 torch.Tensor
    class CompositeCompliantTensor(torch.Tensor):
        elem: torch.Tensor

        __slots__ = ['elem']

        @staticmethod
        # 定义静态方法 __new__，用于创建新的 CompositeCompliantTensor 实例
        def __new__(cls, elem, mode, *args, **kwargs):
            # 断言 elem 的类型不是 cls，即不是 CompositeCompliantTensor
            assert type(elem) is not cls, \
                "Wrapping a CompositeCompliantTensor in a CompositeCompliantTensor is not supported"

            # CompositeCompliantTensor 的 storage 不应该被 Composite 操作直接使用；
            # 如果 Composite 操作尝试直接从 storage 读取而不是派发，则会引发 RuntimeError，因为它是一个元存储。
            r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
                cls, elem.size(),
                dtype=elem.dtype, layout=elem.layout,
                device=elem.device, requires_grad=elem.requires_grad,
                strides=elem.stride(), storage_offset=elem.storage_offset())

            if elem.requires_grad:
                # 如果 elem 需要梯度，则复制一份 elem，并且不要求梯度
                # 这是因为有时 OpInfo 在测试之间共享输入...
                tmp = torch.empty_strided(elem.shape, elem.stride(), dtype=elem.dtype,
                                          device=elem.device, layout=elem.layout,
                                          requires_grad=False)
                tmp.copy_(elem.detach())
                r.elem = tmp
            else:
                # 否则直接使用 elem
                r.elem = elem

            # 断言 r 的步长与 r.elem 的步长相同
            assert r.stride() == r.elem.stride()

            # 将共轭位传播到包装张量
            torch._C._set_conj(r, r.elem.is_conj())
            torch._C._set_neg(r, r.elem.is_neg())

            r.mode = mode  # 设置模式
            return r

        # 返回 CompositeCompliantTensor 的字符串表示形式
        def __repr__(self):
            return f"CompositeCompliantTensor({self.elem})"

        @classmethod
        # 定义类方法 __torch_dispatch__，用于分发函数调用
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            all_args = pytree.arg_tree_leaves(*args, **(kwargs or {}))
            # 获取所有参数中的模式，并确保它们都是相同的模式
            modes = tuple(e.mode for e in all_args if isinstance(e, CompositeCompliantTensor))
            if not all_same_mode(modes):
                raise RuntimeError("Multiple CompositeCompliantTensorModes NYI")
            # 使用第一个模式进行上下文管理
            with modes[0]:
                return func(*args, **kwargs)

    # 返回 CompositeCompliantTensor 类及其模式
    return CompositeCompliantTensor, CompositeCompliantTensorMode()
# 检查给定的列表或元组是否不是列表或元组类型，如果不是，则返回 False
def is_tensorlist(lst):
    if not isinstance(lst, list) and not isinstance(lst, tuple):
        return False
    # 如果列表或元组长度为 0，则返回 False
    if len(lst) == 0:
        return False
    # 检查列表或元组中所有元素是否都是 torch.Tensor 类型，如果是，则返回 True
    all_tensors = all(isinstance(elt, torch.Tensor) for elt in lst)
    if all_tensors:
        return True
    # 如果列表或元组中存在至少一个元素是 torch.Tensor 类型，则抛出 RuntimeError 异常
    exists_one_tensor = all(isinstance(elt, torch.Tensor) for elt in lst)
    if exists_one_tensor:
        raise RuntimeError('This test assumes that PyTorch APIs cannot take '
                           'mixed lists of Tensor and other things')
    # 其他情况返回 False
    return False


# 根据 should_map 的值决定是否对 arg 应用函数 fn，然后返回结果
def maybe_map(fn, should_map, arg):
    return fn(arg) if should_map else arg


# 根据输入的 arg 对象和 CCT 类型创建相应的 CCT 对象
def wrap(arg, CCT, cct_mode):
    # CCT: 通过 generate_cct_and_mode 生成的 CompositeCompliantTensor 类
    if isinstance(arg, torch.Tensor):
        # 如果 arg 是 torch.Tensor 类型，则使用 CCT 和 cct_mode 创建 CCT 对象并返回
        return CCT(arg, cct_mode)
    if is_tensorlist(arg):
        # 如果 arg 是由 torch.Tensor 组成的列表或元组，则分别使用 CCT 和 cct_mode 创建对应的 CCT 对象并返回列表
        return [CCT(a, cct_mode) for a in arg]
    # 如果 arg 既不是 torch.Tensor 也不是由 torch.Tensor 组成的列表或元组，则抛出 RuntimeError 异常
    raise RuntimeError("wrap assumes that the input can be wrapped")


# 给定一组扁平化的参数 flat_args，有些参数可能是 Tensors，返回所有可能的 CompositeCompliantTensor (CCT) 组合方式
def generate_subclass_choices(flat_args, CCT, cct_mode):
    # CCT: 通过 generate_cct_and_mode 生成的 CompositeCompliantTensor 类
    # 检查每个 flat_args 中的参数是否类似于 Tensor
    is_tensor_likes = [isinstance(arg, torch.Tensor) or is_tensorlist(arg) for arg in flat_args]
    # 根据每个参数是否类似于 Tensor，生成对应的选项列表
    subclass_options = [[False, True] if is_tensor_like else [False] for is_tensor_like in is_tensor_likes]

    # 使用 itertools 生成所有 subclass_options 组合的迭代器
    for which_args_are_wrapped in itertools.product(*subclass_options):
        # 对于每种组合，应用 wrap 函数并返回结果列表
        result = [maybe_map(partial(wrap, CCT=CCT, cct_mode=cct_mode), should_wrap_arg, arg)
                  for should_wrap_arg, arg in zip(which_args_are_wrapped, flat_args)]
        # 返回生成的结果列表以及用于调试的元数据信息
        yield result, which_args_are_wrapped


# 对于操作 f(*args, **kwargs)，每个 Tensor 参数可以是常规 Tensor 或 Tensor 子类，迭代所有这些选项
def generate_subclass_choices_args_kwargs(args, kwargs, CCT, cct_mode):
    # CCT: 通过 generate_cct_and_mode 生成的 CompositeCompliantTensor 类
    # 将 kwargs 扁平化并保存其结构
    flat_kwargs, spec = tree_flatten(kwargs)
    # 将 args 和 flat_kwargs 合并成一个扁平化的参数列表
    flat_args_kwargs = list(args) + list(flat_kwargs)
    # 使用 generate_subclass_choices 生成所有可能的 CCT 组合方式
    for choice, debug_metadata in generate_subclass_choices(flat_args_kwargs, CCT, cct_mode):
        # 根据 debug_metadata，将生成的结果列表拆分成新的 args 和 kwargs
        new_args = choice[:len(args)]
        new_kwargs = tree_unflatten(choice[len(args):], spec)
        # 将 debug_metadata 拆分成 args 和 kwargs 对应的 which_args_are_wrapped
        which_args_are_wrapped = debug_metadata[:len(args)]
        which_kwargs_are_wrapped = tree_unflatten(debug_metadata[len(args):], spec)
        # 返回新的 args、kwargs 和其是否被包装的信息
        yield new_args, new_kwargs, which_args_are_wrapped, which_kwargs_are_wrapped


# 抛出 CompositeCompliance 错误，可选择附加额外信息
def raise_composite_compliance_error(err, additional_info=''):
    pass  # 这个函数仅声明了抛出错误，但没有实际内容，可以在需要时补充具体实现
    # 抛出运行时异常，指示复合一致性检查失败，并包含详细的错误信息
    raise RuntimeError(
        "Composite compliance check failed with "
        "the above error.\n"
        f"{additional_info}"
        "If you are adding an OpInfo of an "
        "existing operator, please feel free to skip this test "
        "because the problem was pre-existing and file an issue. "
        "Otherwise, if you added a new operator, please read "
        "through the Composite Compliance section in "
        "aten/src/ATen/native/README.md for how to resolve this. "
    ) from err
# 检查使用 `op` 函数的所有可能参数排列组合，包括普通 Tensor 或 Tensor 子类。
#
# 主要策略是将一些 Tensor 参数和关键字参数包装在 CompositeCompliantTensor 包装器中，并调用操作。

# 如果某些复合操作执行了非兼容行为，CompositeCompliantTensor 将会引发错误。
def check_all_permutations(op, args, kwargs, assert_equal_fn):
    # 生成 CompositeCompliantTensor 和模式
    CCT, cct_mode = generate_cct_and_mode()
    # 计算预期结果，调用 op 函数
    expected = op(*args, **kwargs)
    # 对于每一种 Tensor 子类选择，生成参数和关键字参数的组合
    for choice in generate_subclass_choices_args_kwargs(args, kwargs, CCT, cct_mode):
        new_args, new_kwargs, which_args_are_wrapped, which_kwargs_are_wrapped = choice

        try:
            # 调用 op 函数，传入新的参数和关键字参数
            actual = op(*new_args, **new_kwargs)
        # 注释：[Composite Compliance 需要捕获什么错误？]
        #
        # 我们希望捕获两种类型的错误：
        # - 在 torch_dispatch 实现中可能引发的错误
        # - 访问 data_ptr 的操作
        # 第一种错误可以通过过滤不同的错误类来捕获，而第二种总是会由于其实现方式（如果尝试访问包装 Tensor 的 data_ptr，则会引发内部 RuntimeError）而引发 RuntimeError。
        #
        # 因此，这里捕获最一般的 RuntimeError。如果您在这里调试为什么测试失败，可能操作本身有问题，并且可能还有其他测试也失败了。
        except RuntimeError as err:
            raise_composite_compliance_error(
                err,
                f"- wrapped_args: {which_args_are_wrapped}\n"
                f"- wrapped_kwargs: {which_kwargs_are_wrapped}\n"
            )

        # 定义解包函数，用于解包 CompositeCompliantTensor
        def unwrap(e):
            return e.elem if isinstance(e, CCT) else e

        # 断言实际结果与预期结果相等
        assert_equal_fn(tree_map(unwrap, actual), expected)

# 使用 torch dispatch 模式检查特定反模式，这些反模式不符合复合要求。
#
# 特别是，我们试图防止的反模式是用户创建一个空张量，然后调用 resize_ 进行调整大小。Torch Dispatch 模式在这里有所帮助，因为所有工厂函数将创建符合复合要求的张量。
#
# 主要策略是将所有 Tensor 参数和关键字参数包装在 CompositeCompliantTensor 包装器中。如果复合操作执行了任何非兼容行为，CompositeCompliantTensor 将会引发错误。
def check_with_mode(op, args, kwargs, assert_equal_fn):
    # 生成 CompositeCompliantTensor 和模式
    CCT, cct_mode = generate_cct_and_mode()

    # 包装函数，用于将 Tensor 包装在 CompositeCompliantTensor 中
    def wrap(e):
        return CCT(e, cct_mode) if isinstance(e, torch.Tensor) else e

    # 计算预期结果，调用 op 函数
    expected = op(*args, **kwargs)

    # 将所有参数和关键字参数都进行包装
    args = tree_map(wrap, args)
    kwargs = tree_map(wrap, kwargs)

    try:
        # 使用 cct_mode 上下文管理器调用 op 函数
        with cct_mode:
            actual = op(*args, **kwargs)
    # 参见注释：[Composite Compliance 需要捕获什么错误？]
    # 捕获 RuntimeError 异常，并将其重新抛出为一个复合一致性错误
    except RuntimeError as err:
        raise_composite_compliance_error(err)

    # 定义一个 unwrap 函数，用于从 CCT 实例中获取元素，否则直接返回给定的参数 e
    def unwrap(e):
        return e.elem if isinstance(e, CCT) else e

    # 使用 tree_map 函数对 actual 进行映射操作，将其中的每个元素通过 unwrap 函数处理后与 expected 进行相等性断言
    assert_equal_fn(tree_map(unwrap, actual), expected)
# 收集所有叶子张量，即不需要梯度的张量列表
def gather_leaf_tensors(args, kwargs):
    # 将args展平为列表，同时获取其结构信息
    args, args_spec = tree_flatten(args)
    # 将kwargs展平为列表，同时获取其结构信息
    kwargs, kwargs_spec = tree_flatten(kwargs)
    # 将kwargs的内容添加到args列表中
    args = args + kwargs
    # 遍历args中的每个元素
    for arg in args:
        # 如果arg不是torch.Tensor类型，则继续下一个循环
        if not isinstance(arg, torch.Tensor):
            continue
        # 如果arg需要梯度，则将其添加到leaf_tensors列表中
        if arg.requires_grad:
            leaf_tensors.append(arg)
    # 返回包含所有需要梯度的张量的列表
    return leaf_tensors


# 计算预期的梯度
def compute_expected_grads(op, args, kwargs, output_process_fn_grad=None, gradcheck_wrapper=None):
    # 如果gradcheck_wrapper为None，则直接调用op函数
    if gradcheck_wrapper is None:
        results = op(*args, **kwargs)
    else:
        # 否则使用gradcheck_wrapper对op函数进行包装调用
        results = gradcheck_wrapper(op, *args, **kwargs)

    # 如果指定了output_process_fn_grad函数，则对results应用该函数
    if output_process_fn_grad is not None:
        results = output_process_fn_grad(results)

    # 将results展平为列表
    flat_results = pytree.tree_leaves(results)
    # 筛选出flat_results中的torch.Tensor对象
    flat_results = [r for r in flat_results if isinstance(r, torch.Tensor)]
    # 筛选出需要梯度的torch.Tensor对象
    flat_diff_results = [r for r in flat_results if r.requires_grad]
    # 断言至少存在一个需要梯度的torch.Tensor对象
    assert len(flat_diff_results) > 0

    # 为flat_diff_results中的每个张量创建全为1的梯度张量列表
    grads = [torch.ones(r.shape, device=r.device, dtype=r.dtype) for r in flat_diff_results]
    # 收集所有叶子张量，即不需要梯度的张量列表
    leaf_tensors = gather_leaf_tensors(args, kwargs)
    # 断言至少存在一个叶子张量
    assert len(leaf_tensors) > 0
    # 返回所有需要计算的张量的梯度
    return torch.autograd.grad(flat_diff_results, leaf_tensors,
                               grads, allow_unused=True, retain_graph=True)


# 检查后向传播公式是否符合复合兼容性
# 通过测试所有可能的{inputs, grad_outputs}排列组合，这些可以是CompositeCompliantTensor或常规张量
#
# 注意：op参数被接受为Callable而不是OpInfo，这意味着我们可以将check_backward_formula应用于不是OpInfos的东西，用于调试。
def check_backward_formula(op: Callable, args, kwargs,
                           output_process_fn_grad=None,
                           gradcheck_wrapper=None, assert_equal_fn=None):
    # 生成CompositeCompliantTensor及其模式
    CCT, cct_mode = generate_cct_and_mode()

    # 计算预期的梯度
    expected = compute_expected_grads(op, args, kwargs, output_process_fn_grad, gradcheck_wrapper)
    # 对于给定的参数和关键字参数生成子类选择
    for choice in generate_subclass_choices_args_kwargs(args, kwargs, CCT, cct_mode):
        # 解包选择结果
        new_args, new_kwargs, which_args_are_wrapped, which_kwargs_are_wrapped = choice
        # 收集所有叶子张量
        leaf_tensors = gather_leaf_tensors(new_args, new_kwargs)
        # 断言叶子张量的数量大于0
        assert len(leaf_tensors) > 0

        try:
            # 如果梯度检查函数为None，则调用操作函数
            if gradcheck_wrapper is None:
                results = op(*new_args, **new_kwargs)
            else:
                results = gradcheck_wrapper(op, *new_args, **new_kwargs)
            # 如果有输出处理函数，则对结果进行处理
            if output_process_fn_grad is not None:
                results = output_process_fn_grad(results)
        # 见注释: [Composite Compliance试图捕获哪些错误?]
        except RuntimeError as err:
            # 抛出复合合规性错误，包含错误信息和相关包装参数
            raise_composite_compliance_error(
                err,
                f"- wrapped_args: {which_args_are_wrapped}\n"
                f"- wrapped_kwargs: {which_kwargs_are_wrapped}\n"
            )

        # 扁平化结果张量列表
        flat_results = pytree.tree_leaves(results)
        # 过滤出张量类型的结果
        flat_results = [r for r in flat_results if isinstance(r, torch.Tensor)]
        # 过滤出需要梯度的结果
        flat_diff_results = [r for r in flat_results if r.requires_grad]
        # 断言需要梯度的结果数量大于0
        assert len(flat_diff_results) > 0

        # 注意: 这里使用torch.ones而不是torch.ones_like，以便得到常规张量
        # 创建与梯度张量形状和设备匹配的全1张量列表
        grads = [torch.ones(r.shape, device=r.device, dtype=r.dtype)
                 for r in flat_diff_results]
        
        # 对于生成的梯度张量和子类选择生成批次的子类选择
        for flat_new_grads, which_grad_is_batched in generate_subclass_choices(grads, CCT, cct_mode):
            try:
                # 执行梯度计算，允许未使用变量，并保留计算图
                actual = torch.autograd.grad(flat_diff_results, leaf_tensors, flat_new_grads,
                                             allow_unused=True, retain_graph=True)
            # 见注释: [Composite Compliance试图捕获哪些错误?]
            except RuntimeError as err:
                # 抛出复合合规性错误，包含错误信息和相关包装参数以及梯度信息
                raise_composite_compliance_error(
                    err,
                    f"- wrapped_args: {which_args_are_wrapped}\n"
                    f"- wrapped_kwargs: {which_kwargs_are_wrapped}\n"
                    f"- wrapped_grads: {which_grad_is_batched}\n"
                )

            # 定义一个解包函数，根据是否为CCT类型来解包元素
            def unwrap(e):
                return e.elem if isinstance(e, CCT) else e

            # 断言两个元组中的元素相等，考虑NaN的情况
            assert_equal_fn(tuple(map(unwrap, actual)), expected, equal_nan=True)
# 检查前向自动微分公式是否符合复合要求，通过测试
# 所有可能的 {primals, tangents} 排列组合，这些可以是 CompositeCompliantTensor 或普通的 Tensors。
#
# 注意：op 必须被接受为 Callable 而不是 OpInfo，
# 这意味着我们可以将 check_forward_ad_formula 应用于不是 OpInfos 的东西，用于调试时。
def check_forward_ad_formula(op: Callable, args, kwargs, gradcheck_wrapper=None, assert_equal_fn=None):
    # 生成 CCT 和 cct_mode，autograd_view_consistency 设为 False
    CCT, cct_mode = generate_cct_and_mode(autograd_view_consistency=False)

    def maybe_tangent(t):
        # 断言 t 的类型不是 CCT
        assert type(t) is not CCT
        # 如果给定的对象是一个 Tensor 并且需要梯度
        # 生成 `tangent` 张量
        if isinstance(t, torch.Tensor) and t.requires_grad:
            return torch.randn_like(t)
        elif is_tensorlist(t):
            # 如果 t 是一个 tensorlist，则生成每个元素的随机张量，如果需要梯度的话
            return [torch.randn_like(e) if e.requires_grad else None for e in t]
        return None

    # 对于 args 中的每个参数生成对应的 tangent 参数
    tangent_args = tuple(maybe_tangent(arg) for arg in args)
    
    # 对 kwargs 进行扁平化处理并获取其结构信息
    flat_kwargs, spec = tree_flatten(kwargs)
    # 对扁平化的 kwargs 中的每个参数生成对应的 tangent 参数
    flat_tangent_kwargs = tuple(maybe_tangent(arg) for arg in flat_kwargs)
    # 将生成的扁平化 tangent 参数恢复成原始的 kwargs 结构
    tangent_kwargs = tree_unflatten(flat_tangent_kwargs, spec)
```