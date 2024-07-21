# `.\pytorch\test\test_masked.py`

```
# Owner(s): ["module: masked operators"]

"""Tests for masked operations.
"""

# 导入所需的模块和库
import itertools  # 导入 itertools 库，用于迭代操作
import torch  # 导入 PyTorch 库
from typing import List, Any  # 导入类型提示相关的模块
from functools import wraps  # 导入 wraps 函数，用于装饰器
import unittest  # 导入 unittest 模块，用于单元测试框架
from torch.testing._internal.common_utils import skipIfTorchDynamo  # 从内部测试工具导入 skipIfTorchDynamo 函数


from torch.testing._internal.common_utils import \
    (TestCase, parametrize, suppress_warnings, _TestParametrizer, run_tests)
# 从内部测试工具中导入多个函数和类：TestCase, parametrize, suppress_warnings, _TestParametrizer, run_tests

from torch.testing._internal.common_methods_invocations import \
    (op_db, SampleInput)
# 从内部测试方法调用中导入 op_db 和 SampleInput 函数

from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, ops, onlyNativeDeviceTypes, precisionOverride)
# 从内部设备类型中导入多个函数和对象：instantiate_device_type_tests, ops, onlyNativeDeviceTypes, precisionOverride


def apply_masked_reduction_along_dim(op, input, *args, **kwargs):
    """Applies reduction op along given dimension to strided x
    elements that are valid according to mask tensor.

    The op is applied to each elementary slice of input with args and
    kwargs with the following constraints:

    1. Prior applying the op:

      A. if kwargs contains an item with key 'dim_position' then it is
         removed from kwargs. The value of 'dim_position' is an
         integer that describes the dim argument position: while
         typically the dim argument appears at the 0-th position of
         the op arguments (excluding input), for instance, sum(input,
         dim), then there exists reductions that have extra arguments
         prior the dim argument, for instance, norm(input, ord, dim).

      B. if args or kwargs contains dim or keepdim arguments, these
         will be removed or replaced with None so that the op is
         applied to elementary slice using the default dim and keepdim
         value.

    2. The elementary slice of the input is defined as the flattened
      slice that has no masked out elements and when op is applied,
      the result will be a scalar value (assuming keepdim=False). For
      example, an input tensor to a reduction operation op having
      dim=0 and keepdim=True argument:

       [[1 * 2 * *]
        [* 3 4 * 5]]

      (* denotes masked out elements) has the following elementary
      slices: [1, 2] and [3, 4, 5]. The result of
      apply_masked_reduction_along_dim is

       [[op([1, 2], *args0, **kwargs, dim=None, keepdim=False)]
        [op([3, 4, 5], *args0, **kwargs, dim=None, keepdim=False)]]

      where args0 is args where dim value is replased with None if
      present.

      Using the same example data, if the op is called with dim=(0, 1)
      and keepdim=False, there is one elementary slice: [1, 2, 3, 4,
      5]; and the corresponding result of the op is:

        op([1, 2, 3, 4, 5], *args0, **kwargs, dim=None, keepdim=False)
    """
    # 对输入的每个元素切片应用带有约束条件的 op 操作
    pass  # 占位符，表示函数当前无实际执行内容
    """
    3. If the elementary slice is empty, the corresponding output
      value is nan if dtype is float, otherwise, 0.  An empty
      elementary slice corresponds to fully masked-out output, so, the
      corresponding specific value of the output will not be important
      because we used masked equality check for comparing the results
      of masked operations.
    """
    # 移除 mask 和 dim_position 关键字参数：
    mask = kwargs.pop('mask', None)
    dim_pos = kwargs.pop('dim_position', 0)

    # 获取 dtype 参数，若未指定则使用 input 的 dtype
    dtype = kwargs.get('dtype', input.dtype)

    if input.ndim == 0:
        # 如果输入是标量，即一个 elementary slice
        return op(input, *args, **kwargs).to(dtype=dtype)

    # 移除 keepdim 关键字参数（如果指定）
    keepdim = kwargs.pop('keepdim', False)

    # 如果 dim 在 args 中指定
    if dim_pos < len(args):
        # args 中指定了 dim
        assert 'dim' not in kwargs, (args, kwargs)
        dim = args[dim_pos]
        args0 = args[:dim_pos] + (None,) + args[dim_pos + 1:]
    else:
        # kwargs 中可能指定了 dim
        dim = kwargs.pop('dim', None)
        args0 = args

    # 标准化 dim 参数，确保在合理范围内
    dim_ = torch.masked._canonical_dim(dim, input.ndim)

    # 构建所有 elementary slices 的索引范围
    ranges: List[Any] = []
    shape = []
    for i in range(input.ndim):
        if i in dim_:
            ranges.append((slice(None),))
            shape.append(1)
        else:
            ranges.append(range(input.shape[i]))
            shape.append(input.shape[i])

    # 创建一个填充了 nan 或 0 的 output 数组，用于 keepdim=True 情况
    output = input.new_full(shape, float('nan') if dtype.is_floating_point else 0, dtype=dtype)

    # 对所有 elementary slices 应用操作 op
    if mask is None:
        inpmask = input.new_ones([], dtype=torch.bool).expand(input.shape)
    else:
        inpmask = torch.masked._input_mask(input, mask=mask)
    for s in itertools.product(*ranges):
        # 获取 elementary slice 的数据，只包含 masked-in 的元素
        data = input[s].flatten()[inpmask[s].flatten().nonzero()]
        if not data.numel():
            # 空的 elementary slice
            continue
        output[s][0] = op(data, *args0, **kwargs)

    if not keepdim:
        # 对于 keepdim=False 情况，重新调整 output 的形状
        shape = [shape[i] for i in range(len(shape)) if i not in dim_]
        output = output.reshape(shape)
    return output
# 定义一个函数，沿指定维度应用标准化操作到按掩码张量有效的分块 x 元素
def apply_masked_normalization_along_dim(op, input, *args, **kwargs):
    # 从 kwargs 中取出掩码张量（如果有），默认为 None
    mask = kwargs.pop('mask', None)
    # 从 kwargs 中取出维度位置参数，默认为 0
    dim_pos = kwargs.pop('dim_position', 0)
    
    # 如果输入是标量（0 维）
    if input.ndim == 0:
        # 直接对输入应用操作 op 并返回结果
        return op(input, *args, **kwargs)
    
    # 取输入的数据类型，默认为输入的数据类型
    dtype = kwargs.get('dtype', input.dtype)
    # 取出指定维度的大小
    dim = args[dim_pos]
    # 构建新的参数元组，将第 dim_pos 位置的参数替换为 0
    args0 = args[:dim_pos] + (0,) + args[dim_pos + 1:]
    
    # 根据输入创建一个与其同类型和形状的零张量
    output = torch.zeros_like(input, dtype=dtype)
    
    # 如果没有指定掩码张量，创建一个与输入同形状的全 1 张量作为掩码
    if mask is None:
        inpmask = input.new_ones([], dtype=torch.bool).expand(input.shape)
    else:
        # 使用输入和指定的掩码创建掩码张量
        inpmask = torch.masked._input_mask(input, mask=mask)
    
    # 计算实际维度位置，取余以避免超出维度范围
    dim_ = dim % input.ndim
    
    # 构建左侧和右侧维度范围的迭代器元组
    left_ranges = tuple(map(range, input.shape[:dim_]))
    right_ranges = tuple(map(range, input.shape[dim_ + 1:]))
    
    # 使用 itertools 生成块状索引迭代器，迭代每个块的索引
    for s in itertools.product(*(left_ranges + ((slice(None),),) + right_ranges)):
        # 找到当前块内非零掩码位置的索引
        indices = inpmask[s].argwhere()
        # 对输出中当前块的对应位置应用操作 op，并将结果保存到输出张量中
        output[s][indices] = op(input[s][indices], *args0, **kwargs)
    
    # 返回应用操作后的输出张量
    return output


# 引用函数字典，包含不同的标准化函数及其对应的调用
reference_functions = dict(
    norm=lambda *args, **kwargs: apply_masked_reduction_along_dim(torch.linalg.vector_norm, *args, **dict(kwargs, dim_position=1)),
    var=lambda *args, **kwargs: apply_masked_reduction_along_dim(torch.var, *args, **dict(kwargs, dim_position=0)),
    std=lambda *args, **kwargs: apply_masked_reduction_along_dim(torch.std, *args, **dict(kwargs, dim_position=0)),
    softmax=lambda *args, **kwargs: apply_masked_normalization_along_dim(torch.softmax, *args, **kwargs),
    log_softmax=lambda *args, **kwargs: apply_masked_normalization_along_dim(torch.log_softmax, *args, **kwargs),
    softmin=lambda *args, **kwargs: apply_masked_normalization_along_dim(torch.nn.functional.softmin, *args, **kwargs),
    normalize=lambda *args, **kwargs: apply_masked_normalization_along_dim(
        torch.nn.functional.normalize, *args, **dict(kwargs, dim_position=1)),
)

# 从操作数据库中筛选出以 'masked.' 开头的操作列表
masked_ops = [op for op in op_db if op.name.startswith('masked.')]

# 筛选出带有参考函数的操作列表，参考函数是函数字典中的函数
masked_ops_with_references = [op for op in masked_ops if op.name.rsplit('.', 1)[-1] in reference_functions]

# 筛选出支持稀疏张量或稀疏 CSR 格式的操作列表
masked_ops_with_non_strided_support = [op for op in masked_ops if op.supports_sparse or op.supports_sparse_csr]


# 定义一个函数，将对象的张量内容转换为分块张量内容
def _tensor_to_strided(obj):
    # 一旦 gh-59958 解决，将使用 torch.Tensor.to_dense 替代这个函数的使用
    if torch.is_tensor(obj):
        # 如果张量布局是 strided，直接返回该张量
        if obj.layout == torch.strided:
            return obj
        # 否则将稀疏张量转换为密集张量
        return obj.to_dense()
    # 对于非张量对象，直接返回原对象
    return obj


# 定义一个函数，将对象的张量内容转换为稀疏 COO 格式张量内容
def to_strided(obj):
    """Convert the tensor content of object to strided tensor content.
    """
    return torch.utils._pytree.tree_map(_tensor_to_strided, obj)


# 定义一个函数，将对象的张量内容转换为稀疏 CSR 格式张量内容
def to_sparse_coo(obj):
    """Convert the tensor content of object to sparse coo tensor content.
    """
    return torch.utils._pytree.tree_map(torch.Tensor.to_sparse, obj)


# 定义一个函数，将对象的张量内容转换为稀疏 CSR 格式张量内容
def to_sparse_csr(obj):
    """Convert the tensor content of object to sparse csr tensor content.
    """
    # 使用 PyTorch 的 tree_map 函数，对对象 obj 进行递归映射操作
    return torch.utils._pytree.tree_map(torch.Tensor.to_sparse_csr, obj)
# mask_layouts 类，继承自 _TestParametrizer 类
class mask_layouts(_TestParametrizer):
    """Decorator class for parametrization of test function with an input
    layout argument and an extra argument of sample inputs generator.
    The sample_inputs generator provides samples with all supported
    layouts for the mask argument.
    """
    # 定义一个参数化测试的装饰器函数，接受三个参数：test、generic_cls 和 device_cls
    def _parametrize_test(self, test, generic_cls, device_cls):
    
        # 定义一个包装函数 wrap，用于实际执行参数化测试
        @wraps(test)
        def wrap(self, layout, device, dtype, op):
            # 将 layout 转换为字符串，并去除前缀 'torch.'
            layout_name = str(layout).lstrip('torch.')
            
            # 根据 layout 的不同情况选择合适的 sample_inputs_func 函数
            if layout == torch.strided:
                # 对于 strided 布局，始终支持
                sample_inputs_func = op.sample_inputs
            elif layout == torch.sparse_coo:
                # 对于 sparse_coo 布局，检查是否支持稀疏输入，若不支持则跳过测试
                if not op.supports_sparse:
                    raise unittest.SkipTest(f"{op.name} does not support inputs with {layout_name} layout")
                sample_inputs_func = op.sample_inputs_sparse_coo
            elif layout == torch.sparse_csr:
                # 对于 sparse_csr 布局，检查是否支持稀疏输入和 CSR 格式，若不支持则跳过测试
                if not op.supports_sparse_csr:
                    raise unittest.SkipTest(f"{op.name} does not support inputs with {layout_name} layout")
                sample_inputs_func = op.sample_inputs_sparse_csr
            else:
                # 若 layout 不是上述三种之一，则抛出 NotImplementedError
                raise NotImplementedError(f'{layout}')
            
            # 定义一个生成器函数 sample_inputs_generator，用于生成适合当前布局的输入样本
            def sample_inputs_generator():
                for sample_input in sample_inputs_func(device, dtype):
                    mask = sample_input.kwargs.get('mask')
                    if mask is None:
                        yield sample_input
                    else:
                        # 如果输入样本有 mask 属性，并且当前布局与样本布局相同，则保留原样本
                        if layout == sample_input.input.layout:
                            yield sample_input
                        # 如果当前布局不是 strided，则生成一个带有转换后 mask 的新样本
                        if layout != torch.strided:
                            sample_input_kwargs = sample_input.kwargs.copy()
                            sample_input_kwargs.update(mask=mask.to_dense())
                            yield SampleInput(sample_input.input.clone(),
                                              args=sample_input.args,
                                              kwargs=sample_input_kwargs)
                        # 如果当前布局不是 sparse_coo 并且 op 支持稀疏输入，则生成一个稀疏格式 mask 的新样本
                        if layout != torch.sparse_coo and op.supports_sparse:
                            sample_input_kwargs = sample_input.kwargs.copy()
                            sample_input_kwargs.update(mask=mask.to_sparse())
                            yield SampleInput(sample_input.input.clone(),
                                              args=sample_input.args,
                                              kwargs=sample_input_kwargs)
                        # 如果当前布局不是 sparse_csr 并且 op 支持稀疏输入和 CSR 格式，并且输入样本是二维的，则生成一个 CSR 格式 mask 的新样本
                        if layout != torch.sparse_csr and op.supports_sparse_csr and sample_input.input.ndim == 2:
                            sample_input_kwargs = sample_input.kwargs.copy()
                            sample_input_kwargs.update(mask=mask.to_sparse_csr())
                            yield SampleInput(sample_input.input.clone(),
                                              args=sample_input.args,
                                              kwargs=sample_input_kwargs)
            
            # 调用测试函数 test，并传入 layout、device、dtype、op 和 sample_inputs_generator
            test(self, layout, device, dtype, op, sample_inputs_generator())
        
        # 对三种布局进行循环迭代，每次迭代生成一个参数化测试
        for layout in (torch.strided, torch.sparse_coo, torch.sparse_csr):
            # 返回一个元组，包含 wrap 函数、去除 'torch.' 后的布局名字符串和布局字典参数
            yield (wrap, str(layout).lstrip('torch.'), {'layout': layout}, lambda _: [])
class TestMasked(TestCase):

    # 自定义断言方法，用于比较带有遮罩的张量操作的实际和期望结果
    def assertEqualMasked(self, actual, expected, mask):
        # 将实际张量转换为步进张量
        strided = to_strided(actual)
        # 如果存在遮罩，根据遮罩条件选择是否保留步进张量中的元素或置零
        if mask is not None:
            strided = torch.where(mask, strided, strided.new_zeros([]))
            expected = torch.where(mask, expected, expected.new_zeros([]))
        # 使用自定义的不精确设备比较方法进行断言比较
        self.assertEqual(strided, expected, exact_device=False)

    # 测试带有引用的遮罩操作，使用特定设备类型进行测试，忽略警告
    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(masked_ops_with_references)
    @precisionOverride({torch.bfloat16: 5e-4, torch.float16: 5e-4})
    def test_reference_masked(self, device, dtype, op):
        # 获取操作的名称
        op_name = op.name.rsplit('.', 1)[-1]
        # 获取参考函数的操作
        ref_op = reference_functions[op_name]
        # 获取操作的样本输入
        sample_inputs = op.sample_inputs(device, dtype)
        # 遍历每个样本输入
        for sample_input in sample_inputs:
            t_inp, t_args, t_kwargs = sample_input.input, sample_input.args, sample_input.kwargs
            # 如果操作是 'var' 或 'std' 且输入不是浮点数或复数类型，则跳过
            if op_name in {'var', 'std'} and not (t_inp.dtype.is_floating_point or t_inp.dtype.is_complex):
                continue
            # 执行操作并获取实际结果
            actual = op.op(t_inp, *t_args, **t_kwargs)
            # 获取预期结果
            expected = ref_op(t_inp, *t_args, **t_kwargs)
            # 如果存在遮罩，则获取输出遮罩
            if t_kwargs.get('mask') is None:
                outmask = None
            else:
                outmask = torch.masked._output_mask(op.op, t_inp, *t_args, **t_kwargs)
            # 使用自定义的断言方法比较实际和预期结果
            self.assertEqualMasked(actual, expected, outmask)

    # 测试遮罩布局，使用不同的布局和设备类型进行测试，忽略警告
    @mask_layouts()
    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(masked_ops_with_non_strided_support)
    @precisionOverride({torch.bfloat16: 5e-3, torch.float16: 5e-3})
    def test_mask_layout(self, layout, device, dtype, op, sample_inputs):
        # 遍历每个样本输入
        for sample in sample_inputs:
            t_inp, t_args, t_kwargs = sample.input, sample.args, sample.kwargs
            # 执行操作并获取实际结果
            actual = op.op(t_inp, *t_args, **t_kwargs)
            # 断言实际结果的布局与指定布局一致
            assert actual.layout == layout

            # 检查遮罩不变性:
            #  op(inp, mask).to_dense() == op(inp.to_dense(), mask.to_dense()) at outmask
            #
            # 转换输入和参数到步进形式
            r_inp, r_args, r_kwargs = to_strided((t_inp, t_args, t_kwargs))
            # 如果存在遮罩，则获取输出遮罩
            if r_kwargs.get('mask') is None:
                outmask = None
            else:
                outmask = torch.masked._output_mask(op.op, r_inp, *r_args, **r_kwargs)
            # 获取预期结果
            expected = op.op(r_inp, *r_args, **r_kwargs)
            # 使用自定义的断言方法比较实际和预期结果
            self.assertEqualMasked(actual, expected, outmask)

    # 如果存在 Torch Dynamo 的问题，则跳过相关测试
    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1992")
    @parametrize("sparse_kind,fill_value", [('coo', 0), ('hybrid_coo', 0),
                                            ('coo', 123), ('hybrid_coo', 123),
                                            ('csr', 0), ('csr', 123)],
                 name_fn=lambda sparse_kind, fill_value: f'{sparse_kind}_fill_value_{fill_value}')
instantiate_device_type_tests(TestMasked, globals(), except_for='meta')

# 如果当前文件为主文件，则执行测试
if __name__ == "__main__":
    run_tests()
```