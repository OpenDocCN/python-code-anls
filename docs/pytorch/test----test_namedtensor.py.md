# `.\pytorch\test\test_namedtensor.py`

```py
# Owner(s): ["module: named tensor"]

# 引入单元测试模块
import unittest
# 引入测试框架的基类、运行测试方法、测试的NumPy相关工具
from torch.testing._internal.common_utils import TestCase, run_tests, TEST_NUMPY
# 引入如果使用Torch Dynamo则跳过测试的方法
from torch.testing._internal.common_utils import skipIfTorchDynamo
# 引入CUDA相关测试工具
from torch.testing._internal.common_cuda import TEST_CUDA
# 引入设备类型管理工具
from torch.testing._internal.common_device_type import get_all_device_types
# 引入命名元组和有序字典
from collections import namedtuple, OrderedDict
# 引入迭代工具和函数工具
import itertools
import functools
# 引入PyTorch主库
import torch
# 引入PyTorch的Tensor类
from torch import Tensor
# 引入PyTorch的函数库
import torch.nn.functional as F
# 引入进程间数据传递工具
from multiprocessing.reduction import ForkingPickler
# 引入序列化工具
import pickle
# 引入输入输出工具
import io
# 引入系统相关工具
import sys
# 引入警告处理工具
import warnings

# 定义函数，创建带有命名的Tensor
def pass_name_to_python_arg_parser(name):
    x = torch.empty(2, names=(name,))

# 定义函数，将嵌套列表展开成一维列表
def flatten(lst):
    return [item for sublist in lst for item in sublist]

# 定义命名元组的别名为Function，包含名称和Lambda表达式
Function = namedtuple('TestCase', ['name', 'lambd'])

# 解析压缩的命名形状描述字符串
def parse_compressed_namedshape(string):
    # 函数内部定义：解析命名，返回None或字符串名称
    def parse_name(maybe_name):
        maybe_name = maybe_name.strip()
        if maybe_name == 'None':
            return None
        return maybe_name

    string = string.strip()

    # 如果字符串为空，返回None和空列表
    if len(string) == 0:
        return None, []

    # 如果字符串中不包含冒号，则按逗号分割，并转换为整数列表，名称为None
    if ':' not in string:
        return None, [int(size) for size in string.split(',')]

    # 分割字符串，形成维度的列表
    dims = string.split(',')
    # 列表推导式：将维度分割成名称和大小的元组，调用解析命名函数
    tuples = [dim.split(':') for dim in dims]
    return zip(*[(parse_name(name), int(size)) for name, size in tuples])

# 根据压缩的命名形状字符串创建Tensor
def create(namedshape, factory=torch.randn):
    # 解析命名形状字符串，获取名称和形状
    names, shape = parse_compressed_namedshape(namedshape)
    # 使用指定工厂函数创建Tensor
    return factory(shape, names=names)

# 定义装饰器函数，用于处理输出的函数操作符
def out_fn(operator):
    @functools.wraps(operator)
    def fn(*inputs):
        return operator(*inputs[1:], out=inputs[0])
    return fn

# 测试类：测试命名Tensor
class TestNamedTensor(TestCase):
    # 测试：必须首先运行的检查实验性警告
    def test_aaa_must_run_first_check_experimental_warning(self):
        # TODO(rzou): It would be nice for this to be a "real" python warning.
        # Right now this error message only prints once and doesn't respect
        # warnings.simplefilter behavior (where python users can control whether
        # or not to display warnings once, all the time, or never).
        # 捕获警告，确保仅出现一次警告，并验证警告消息的开头部分
        with warnings.catch_warnings(record=True) as warns:
            x = torch.randn(3, 3, names=('N', 'C'))
            self.assertEqual(len(warns), 1)
            self.assertTrue(str(warns[0].message).startswith(
                'Named tensors and all their associated APIs are an experimental feature'))

    # 测试：简单的占位测试
    def test_trivial(self):
        pass
    # 定义测试方法 _test_name_inference，用于测试操作 op 在给定参数 args 下的命名推断
    def _test_name_inference(self, op, args=(), expected_names=(), device='cpu',
                             maybe_raises_regex=None):
        # 将参数 args 中的张量转移到指定设备 device 上，非张量参数保持不变
        casted_args = [arg.to(device) if isinstance(arg, torch.Tensor) else arg
                       for arg in args]
        # 如果定义了异常正则表达式 maybe_raises_regex，则期望操作 op 在 args 上引发异常
        if maybe_raises_regex is not None:
            with self.assertRaisesRegex(RuntimeError, maybe_raises_regex):
                result = op(*args)
            return
        # 否则，执行操作 op 在 args 上的计算，得到结果 result
        result = op(*args)
        # 断言操作返回的张量命名与预期命名 expected_names 相符
        self.assertEqual(result.names, expected_names,
                         msg=f'Name inference for {op.__name__} on device {device} failed')

    # TODO(rzou): 应该将某种形式的这个检查添加到 self.assertEqual 中。
    # 目前我不知道它应该是什么样子。

    # 定义断言方法 assertTensorDataAndNamesEqual，用于比较两个张量 x 和 y 的数据及命名是否相等
    def assertTensorDataAndNamesEqual(self, x, y):
        # 断言张量 x 和 y 的命名相等
        self.assertEqual(x.names, y.names)
        # 将张量 x 和 y 的命名重命名为 None，再次断言它们的数据是否相等
        unnamed_x = x.rename(None)
        unnamed_y = y.rename(None)
        self.assertEqual(unnamed_x, unnamed_y)

    # 定义测试工厂方法 _test_factory，用于测试创建张量的工厂函数 factory 在不同参数情况下的行为
    def _test_factory(self, factory, device):
        # 测试不同参数情况下，使用工厂函数 factory 创建张量 x 后的命名情况
        x = factory([], device=device)
        self.assertEqual(x.names, ())

        x = factory(1, 2, 3, device=device)
        self.assertEqual(x.names, (None, None, None))

        x = factory(1, 2, 3, names=None, device=device)
        self.assertEqual(x.names, (None, None, None))

        x = factory(1, 2, 3, names=('N', 'T', 'D'), device=device)
        self.assertEqual(x.names, ('N', 'T', 'D'))

        x = factory(1, 2, 3, names=('N', None, 'D'), device=device)
        self.assertEqual(x.names, ('N', None, 'D'))

        x = factory(1, 2, 3, names=('_1', 'batch9', 'BATCH_5'), device=device)
        self.assertEqual(x.names, ('_1', 'batch9', 'BATCH_5'))

        # 测试工厂函数 factory 在指定不合法命名下是否会引发异常
        with self.assertRaisesRegex(RuntimeError,
                                    'a valid identifier contains only'):
            x = factory(2, names=('1',), device=device)

        with self.assertRaisesRegex(RuntimeError,
                                    'a valid identifier contains only'):
            x = factory(2, names=('?',), device=device)

        # 测试工厂函数 factory 在命名数量不匹配的情况下是否会引发异常
        with self.assertRaisesRegex(RuntimeError, 'Number of names'):
            x = factory(2, 1, names=('N',), device=device)

        # 测试工厂函数 factory 在命名参数类型错误的情况下是否会引发异常
        with self.assertRaisesRegex(TypeError, 'invalid combination of arguments'):
            x = factory(2, 1, names='N', device=device)

        # 测试工厂函数 factory 在存在重复命名的情况下是否会引发异常
        with self.assertRaisesRegex(RuntimeError, 'construct a tensor with duplicate names'):
            x = factory(2, 1, 1, names=('N', 'C', 'N'), device=device)

        # 测试工厂函数 factory 在超过64维命名的情况下是否会引发异常
        names64 = ['A' * i for i in range(1, 65)]
        x = factory([1] * 64, names=names64, device=device)
        self.assertEqual(x.names, names64)

        with self.assertRaisesRegex(
                RuntimeError,
                'only support up to 64 dims'):
            names65 = ['A' * i for i in range(1, 66)]
            x = factory([1] * 65, names=names64, device=device)

    # 标记该测试方法为跳过，当 Torch Dynamo 激活时不运行该测试
    @skipIfTorchDynamo("not a bug: Dynamo causes the refcounts to be different")
    def test_none_names_refcount(self, N=10):
        # 定义内部函数 scope，用于测试作用域内的操作
        def scope():
            # 创建一个未命名的 tensor，形状为 (2, 3)
            unnamed = torch.empty(2, 3)
            # 获取未命名 tensor 的命名信息，将其实例化为 [None, None]
            unnamed.names  # materialize [None, None]

        # 获取当前 Py_None 对象的引用计数
        prev_none_refcnt = sys.getrefcount(None)
        # 运行 scope 函数 N 次，以减少不确定性
        [scope() for i in range(N)]
        # 获取运行 scope 函数后 Py_None 对象的引用计数
        after_none_refcnt = sys.getrefcount(None)
        # 断言运行 scope 函数后 Py_None 对象的引用计数变化不超过 N/2
        self.assertTrue(after_none_refcnt - prev_none_refcnt < N / 2,
                        msg='Using tensor.names should not change '
                            'the refcount of Py_None')

    def test_has_names(self):
        # 创建一个未命名的 tensor，形状为 (2, 3)
        unnamed = torch.empty(2, 3)
        # 创建一个两个维度均为 None 命名的 tensor，形状为 (2, 3)
        none_named = torch.empty(2, 3, names=(None, None))
        # 创建一个部分命名的 tensor，形状为 (2, 3)，其中一个维度命名为 'N'
        partially_named = torch.empty(2, 3, names=('N', None))
        # 创建一个完全命名的 tensor，形状为 (2, 3)，两个维度分别命名为 'N' 和 'C'
        fully_named = torch.empty(2, 3, names=('N', 'C'))

        # 断言未命名 tensor 是否有命名
        self.assertFalse(unnamed.has_names())
        # 断言两个维度均为 None 命名的 tensor 是否有命名
        self.assertFalse(none_named.has_names())
        # 断言部分命名的 tensor 是否有命名
        self.assertTrue(partially_named.has_names())
        # 断言完全命名的 tensor 是否有命名
        self.assertTrue(fully_named.has_names())

    def test_py3_ellipsis(self):
        # 创建一个形状为 (2, 3, 5, 7) 的 tensor
        tensor = torch.randn(2, 3, 5, 7)
        # 使用 refine_names 方法给 tensor 命名维度 'N'、'...'、'C'
        output = tensor.refine_names('N', ..., 'C')
        # 断言输出的 tensor 的命名信息为 ['N', None, None, 'C']
        self.assertEqual(output.names, ['N', None, None, 'C'])

    def test_refine_names(self):
        # 测试用例：未命名 tensor -> 未命名 tensor
        self._test_name_inference(Tensor.refine_names,
                                  [create('None:1,None:2,None:3'), 'N', 'C', 'H'],
                                  ['N', 'C', 'H'])

        # 测试用例：命名 tensor -> 命名 tensor
        self._test_name_inference(Tensor.refine_names,
                                  [create('N:1,C:2,H:3'), 'N', 'C', 'H'],
                                  ['N', 'C', 'H'])

        # 测试用例：部分命名 tensor -> 命名 tensor
        self._test_name_inference(Tensor.refine_names,
                                  [create('None:1,C:2,None:3'), None, 'C', 'H'],
                                  [None, 'C', 'H'])

        # 测试用例：维度数不匹配
        self._test_name_inference(Tensor.refine_names,
                                  [create('None:2,None:3'), 'N', 'C', 'H'],
                                  maybe_raises_regex="different number of dims")

        # 测试用例：无法从 'D' 类型的 tensor 改变为 'N' 类型的 tensor
        self._test_name_inference(Tensor.refine_names,
                                  [create('D:3'), 'N'],
                                  maybe_raises_regex="is different from")

        # 测试用例：无法从 'D' 类型的 tensor 改变为 'None' 类型的 tensor
        self._test_name_inference(Tensor.refine_names,
                                  [create('D:3'), None],
                                  maybe_raises_regex="'D' is more specific than None")

        # 测试用例：存在通配符行为
        self._test_name_inference(Tensor.refine_names,
                                  [create('None:1,None:1,None:2,None:3'), '...', 'C', 'H'],
                                  [None, None, 'C', 'H'])
    def test_detach(self):
        names = ['N']
        # 调用 Tensor.detach_ 方法，断开梯度关联，并验证命名推断是否正确
        self._test_name_inference(
            Tensor.detach_,
            [torch.randn(3, requires_grad=True, names=names)],
            names)
        # 调用 Tensor.detach 方法，返回不需要梯度的张量，并验证命名推断是否正确
        self._test_name_inference(
            Tensor.detach,
            [torch.randn(3, requires_grad=True, names=names)],
            names)

    def test_index_fill(self):
        for device in get_all_device_types():
            expected_names = ('N', 'C')
            # 创建具有指定设备和命名的张量
            x = torch.randn(3, 5, device=device, names=expected_names)

            # 使用指定索引填充张量的指定维度，并验证填充后的张量命名是否正确
            output = x.index_fill_('C', torch.tensor([0, 1], device=device), 5)
            self.assertEqual(output.names, expected_names)

            # 使用指定索引填充张量的指定维度，并验证填充后的张量命名是否正确
            output = x.index_fill_('C', torch.tensor([0, 1], device=device), torch.tensor(4.))
            self.assertEqual(output.names, expected_names)

            # 创建使用指定索引填充张量的指定维度，并验证填充后的张量命名是否正确
            output = x.index_fill('C', torch.tensor([0, 1], device=device), 5)
            self.assertEqual(output.names, expected_names)

            # 创建使用指定索引填充张量的指定维度，并验证填充后的张量命名是否正确
            output = x.index_fill('C', torch.tensor([0, 1], device=device), torch.tensor(4.))
            self.assertEqual(output.names, expected_names)

    def test_equal(self):
        for device in get_all_device_types():
            # 创建指定设备的随机张量和其克隆张量
            tensor = torch.randn(2, 3, device=device)
            other = tensor.clone()

            # 测试张量重命名后是否相等，并验证结果
            self.assertTrue(torch.equal(tensor.rename('N', 'C'), other.rename('N', 'C')))
            self.assertFalse(torch.equal(tensor.rename('M', 'C'), other.rename('N', 'C')))
            self.assertFalse(torch.equal(tensor.rename(None, 'C'), other.rename('N', 'C')))

    def test_squeeze(self):
        # 创建具有命名的张量并进行维度压缩操作，并验证结果张量的命名是否正确
        x = create('N:3,C:1,H:1,W:1')
        output = x.squeeze('C')
        self.assertEqual(output.names, ['N', 'H', 'W'])

        # 创建未命名的张量并进行维度压缩操作，并验证结果张量的命名是否正确
        output = x.squeeze()
        self.assertEqual(output.names, ['N'])

    def test_repr(self):
        # 创建带有命名的全零张量并验证其字符串表示是否正确
        named_tensor = torch.zeros(2, 3).rename_('N', 'C')
        expected = "tensor([[0., 0., 0.],\n        [0., 0., 0.]], names=('N', 'C'))"
        self.assertEqual(repr(named_tensor), expected)

        # 创建未命名的全零张量并验证其字符串表示是否正确
        unnamed_tensor = torch.zeros(2, 3)
        expected = "tensor([[0., 0., 0.],\n        [0., 0., 0.]])"
        self.assertEqual(repr(unnamed_tensor), expected)

        # 创建无命名的全零张量并验证其字符串表示是否正确
        none_named_tensor = torch.zeros(2, 3).rename_(None, None)
        self.assertEqual(repr(none_named_tensor), expected)

    def test_diagonal(self):
        # 创建具有命名的张量并进行对角线操作，并验证结果张量的命名是否正确
        named_tensor = torch.zeros(2, 3, 5, 7, names=list('ABCD'))
        self.assertEqual(named_tensor.diagonal().names, ['C', 'D', None])
        self.assertEqual(named_tensor.diagonal(1, 3).names, ['A', 'C', None])

        # 创建具有命名的张量并进行指定输出维度、维度1和维度2的对角线操作，并验证结果张量的命名是否正确
        self.assertEqual(named_tensor.diagonal(outdim='E', dim1='B', dim2='D').names,
                         ['A', 'C', 'E'])
    # 测试最大池化操作的方法
    def test_max_pooling(self):
        # 定义检查返回元组的函数
        def check_tuple_return(op, inputs, expected_names):
            # 调用操作函数，获取返回的值和索引
            values, indices = op(*inputs)
            # 断言返回的值和索引的命名与期望的命名一致
            self.assertEqual(values.names, expected_names)
            self.assertEqual(indices.names, expected_names)

        # 遍历所有设备类型
        for device in get_all_device_types():

            # 创建命名张量，指定设备和命名
            named_tensor_1d = torch.zeros(2, 3, 5, device=device, names=list('ABC'))
            named_tensor_2d = torch.zeros(2, 3, 5, 7, device=device, names=list('ABCD'))
            named_tensor_3d = torch.zeros(2, 3, 5, 7, 9, device=device, names=list('ABCDE'))

            # 断言对应维度的最大池化操作的输出命名与输入命名一致
            self.assertEqual(F.max_pool1d(named_tensor_1d, 2).names, named_tensor_1d.names)
            self.assertEqual(F.max_pool2d(named_tensor_2d, [2, 2]).names, named_tensor_2d.names)
            self.assertEqual(F.max_pool3d(named_tensor_3d, [2, 2, 2]).names, named_tensor_3d.names)

            # 调用检查元组返回函数，验证带索引的最大池化操作的输出命名与输入命名一致
            check_tuple_return(F.max_pool1d_with_indices, [named_tensor_1d, 2], named_tensor_1d.names)
            check_tuple_return(F.max_pool2d_with_indices, [named_tensor_2d, [2, 2]], named_tensor_2d.names)
            check_tuple_return(F.max_pool3d_with_indices, [named_tensor_3d, [2, 2, 2]], named_tensor_3d.names)

    # 测试最大池化操作在没有命名时不会产生警告
    def test_max_pooling_without_names_does_not_warn(self):
        # 遍历所有设备类型
        for device in get_all_device_types():
            # 创建需要梯度的命名张量
            tensor_2d = torch.zeros(2, 3, 5, 7, device=device, requires_grad=True)
            # 使用警告上下文捕获所有警告
            with warnings.catch_warnings(record=True) as warns:
                warnings.simplefilter("always")
                # 执行最大池化操作，并进行梯度反向传播
                result = F.max_pool2d(tensor_2d, [2, 2])
                result.sum().backward()
                # 断言警告数量为零
                self.assertEqual(len(warns), 0)

    # 测试不支持保存命名张量的情况
    def test_no_save_support(self):
        # 创建带命名的零张量
        named_tensor = torch.zeros(2, 3, names=('N', 'C'))
        buf = io.BytesIO()
        # 断言在尝试保存时抛出特定的运行时错误
        with self.assertRaisesRegex(RuntimeError, "NYI"):
            torch.save(named_tensor, buf)

    # 测试不支持使用 pickle 序列化命名张量的情况
    def test_no_pickle_support(self):
        # 创建带命名的零张量
        named_tensor = torch.zeros(2, 3, names=('N', 'C'))
        # 断言在尝试使用 pickle 序列化时抛出特定的运行时错误
        with self.assertRaisesRegex(RuntimeError, "NYI"):
            serialized = pickle.dumps(named_tensor)

    # 测试不支持使用多进程 pickle 序列化命名张量的情况
    def test_no_multiprocessing_support(self):
        # 创建带命名的零张量
        named_tensor = torch.zeros(2, 3, names=('N', 'C'))
        buf = io.BytesIO()
        # 断言在尝试使用多进程 pickle 序列化时抛出特定的运行时错误
        with self.assertRaisesRegex(RuntimeError, "NYI"):
            ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(named_tensor)

    # 测试大张量的字符串表示是否包含命名信息
    def test_big_tensor_repr_has_names(self):
        # 定义检查张量字符串表示的函数
        def check_repr(named_tensor):
            # 创建不带命名的张量
            unnamed_tensor = named_tensor.rename(None)
            # 构建包含命名信息的标签
            names_tag = f'names={named_tensor.names}'
            # 断言张量的字符串表示包含命名信息的标签
            self.assertIn(names_tag, repr(named_tensor))

        # 调用检查张量字符串表示的函数，验证大张量的字符串表示是否包含命名信息
        check_repr(torch.randn(128, 3, 64, 64, names=('N', 'C', 'H', 'W')))

    # 测试非连续张量转换为连续张量时是否保留命名信息
    def test_noncontig_contiguous(self):
        # 对特定情况下的非连续张量进行连续化操作的测试
        # 遍历所有设备类型
        for device in get_all_device_types():
            # 创建随机张量，并转置后重命名
            x = torch.randn(2, 3, device=device).t().rename_('N', 'C')
            # 断言连续化后张量的命名信息仍然存在且正确
            self.assertEqual(x.contiguous().names, ('N', 'C'))
    def test_copy_transpose(self):
        # This type of copy is special-cased and therefore needs its own test
        # 定义内部函数 _test，用于测试特定情况下的张量复制和转置操作
        def _test(self_names, other_names, expected_names):
            # 创建一个空的张量 x，指定其维度名称为 self_names
            x = torch.empty(2, 5, names=self_names)
            # 创建一个空的张量 y，形状为 (5, 2)，进行转置，并设置其维度名称为 other_names
            y = torch.empty(5, 2).t().rename_(*other_names)
            # 将 y 的数据复制到 x 中
            x.copy_(y)
            # 断言复制后 x 的维度名称与期望的 expected_names 相同
            self.assertEqual(x.names, expected_names)

        # 测试两种不同的维度名称情况下的 _test 函数
        _test(('N', 'C'), ('N', 'C'), ('N', 'C'))
        _test(None, ('N', 'C'), ('N', 'C'))

    def test_rename_(self):
        # 创建一个空的张量 tensor，指定其维度名称为 ('N', 'C')
        tensor = torch.empty(1, 1, names=('N', 'C'))
        # 使用 rename_ 方法将张量的维度名称设置为 None，并断言修改后的维度名称为 (None, None)
        self.assertEqual(tensor.rename_(None).names, (None, None))
        # 使用 rename_ 方法将张量的维度名称从 ('N', 'C') 修改为 ('H', 'W')，并断言修改后的维度名称为 ('H', 'W')
        self.assertEqual(tensor.rename_('H', 'W').names, ('H', 'W'))
        # 使用 rename_ 方法传入不合法的维度名称参数，断言会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, 'Number of names'):
            tensor.rename_('N', 'C', 'W')
        # 使用 rename_ 方法传入重复的维度名称参数，断言会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, 'duplicate names'):
            tensor.rename_('N', 'N')

    def test_rename(self):
        # 创建一个空的张量 tensor，指定其维度名称为 ('N', 'C')
        tensor = torch.empty(1, 1, names=('N', 'C'))

        # 使用 rename 方法将张量的维度名称设置为 None，并断言修改后的维度名称为 (None, None)
        self.assertEqual(tensor.rename(None).names, (None, None))
        # 使用 rename 方法将张量的维度名称从 ('N', 'C') 修改为 ('H', 'W')，并断言修改后的维度名称为 ('H', 'W')
        self.assertEqual(tensor.rename('H', 'W').names, ('H', 'W'))

        # 检查修改维度名称后，原始张量 tensor 的维度名称并未改变
        self.assertEqual(tensor.names, ('N', 'C'))

        # 使用 rename 方法传入不合法的维度名称参数，断言会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, 'Number of names'):
            tensor.rename('N', 'C', 'W')
        # 使用 rename 方法传入重复的维度名称参数，断言会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, 'duplicate names'):
            tensor.rename('N', 'N')

        # 使用 rename 方法同时传入位置参数和关键字参数，断言会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, 'either positional args or keyword args'):
            tensor.rename(None, N='batch')

        # rename 方法返回张量的视图，因此断言修改维度名称后的数据指针与原张量相同
        self.assertEqual(tensor.rename('H', 'W').data_ptr(), tensor.data_ptr())
        # 使用 rename 方法将维度名称设置为 None，断言修改维度名称后的数据指针与原张量相同
        self.assertEqual(tensor.rename(None).data_ptr(), tensor.data_ptr())
    def test_rename_globber(self):
        # 创建一个标量张量
        scalar = torch.randn([])
        # 创建一个未命名的张量
        unnamed_tensor = torch.empty(1, 1, 1, 1)
        # 创建一个命名的张量
        named_tensor = torch.empty(1, 1, 1, 1, names=('N', 'C', 'H', 'W'))

        # 测试标量张量的重命名，期望结果是空的维度名列表
        self.assertEqual(scalar.rename(None).names, [])
        # 再次测试标量张量的重命名，期望结果是空的维度名列表
        self.assertEqual(scalar.rename('...').names, [])

        # 检查未命名张量的重命名操作
        self.assertEqual(unnamed_tensor.rename('...').names, unnamed_tensor.names)
        # 检查未命名张量的多维度重命名操作
        self.assertEqual(unnamed_tensor.rename('...', 'H', 'W').names,
                         [None, None, 'H', 'W'])
        # 检查未命名张量的多维度重命名操作
        self.assertEqual(unnamed_tensor.rename('N', '...', 'W').names,
                         ['N', None, None, 'W'])
        # 检查未命名张量的多维度重命名操作
        self.assertEqual(unnamed_tensor.rename('N', 'C', '...').names,
                         ['N', 'C', None, None])

        # 检查命名张量的重命名操作
        self.assertEqual(named_tensor.rename('...').names, named_tensor.names)
        # 检查命名张量的多维度重命名操作
        self.assertEqual(named_tensor.rename('...', 'width').names,
                         ['N', 'C', 'H', 'width'])
        # 检查命名张量的多维度重命名操作
        self.assertEqual(named_tensor.rename('batch', 'channels', '...', 'width').names,
                         ['batch', 'channels', 'H', 'width'])
        # 检查命名张量的多维度重命名操作
        self.assertEqual(named_tensor.rename('batch', '...').names,
                         ['batch', 'C', 'H', 'W'])

        # 测试空的全局重命名
        self.assertEqual(unnamed_tensor.rename('...', None, None, None, None).names,
                         [None, None, None, None])
        # 检查命名张量的多维度重命名操作
        self.assertEqual(named_tensor.rename('N', 'C', 'H', '...', 'W').names,
                         ['N', 'C', 'H', 'W'])

        # 测试多个全局重命名引发异常
        with self.assertRaisesRegex(RuntimeError, 'More than one '):
            named_tensor.rename('...', 'channels', '...')

    def test_rename_rename_map(self):
        # 创建一个标量张量
        scalar = torch.randn([])
        # 创建一个未命名的张量
        unnamed_tensor = torch.empty(1, 1, 1, 1)
        # 创建一个命名的张量
        named_tensor = torch.empty(1, 1, 1, 1, names=('N', 'C', 'H', 'W'))

        # 测试标量张量尝试重命名不存在的维度时引发异常
        with self.assertRaisesRegex(RuntimeError, "dim 'N' does not exist"):
            scalar.rename(N='batch')
        # 测试未命名张量尝试重命名不存在的维度时引发异常
        with self.assertRaisesRegex(RuntimeError, "dim 'N' does not exist"):
            unnamed_tensor.rename(N='batch')
        # 测试命名张量尝试重命名不存在的维度时引发异常
        with self.assertRaisesRegex(RuntimeError, "dim 'B' does not exist"):
            named_tensor.rename(B='batch')
        # 测试命名张量尝试重命名多个不存在的维度时引发异常
        with self.assertRaisesRegex(RuntimeError, "dim 'B' does not exist"):
            named_tensor.rename(H='height', B='batch')

        # 检查命名张量的单维度重命名操作，验证数据指针未发生变化
        self.assertEqual(named_tensor.rename(N='batch').data_ptr(),
                         named_tensor.data_ptr())
        # 检查命名张量的单维度重命名操作，验证维度名变更后的结果
        self.assertEqual(named_tensor.rename(N='batch').names,
                         ['batch', 'C', 'H', 'W'])
        # 检查命名张量的多维度重命名操作，验证维度名变更后的结果
        self.assertEqual(named_tensor.rename(N='batch', H='height').names,
                         ['batch', 'C', 'height', 'W'])
    # 定义测试方法，验证设置张量名称属性的行为
    def test_set_names_property(self):
        # 创建一个空张量，指定维度名称为 ('N', 'C')
        tensor = torch.empty(1, 1, names=('N', 'C'))

        # 设置张量名称为 None，并断言名称变为 (None, None)
        tensor.names = None
        self.assertEqual(tensor.names, (None, None))

        # 设置张量名称为 ('N', 'W')，并断言名称变为 ('N', 'W')
        tensor.names = ('N', 'W')
        self.assertEqual(tensor.names, ('N', 'W'))

        # 使用断言捕获 RuntimeError 异常，确保设置 ['N', 'C', 'W'] 时抛出异常
        with self.assertRaisesRegex(RuntimeError, 'Number of names'):
            tensor.names = ['N', 'C', 'W']
        # 使用断言捕获 RuntimeError 异常，确保设置 ['N', 'N'] 时抛出异常
        with self.assertRaisesRegex(RuntimeError, 'duplicate names'):
            tensor.names = ['N', 'N']

    # 定义测试方法，测试张量工厂函数的边缘情况
    def test_factory_edge_cases(self):
        # 遍历所有设备类型，对每种设备调用 _test_factory 方法
        for device in get_all_device_types():
            self._test_factory(torch.empty, device)

    # 定义测试方法，覆盖张量工厂函数的不同情况
    def test_factory_coverage(self):
        # 定义内部方法 _test，测试特定工厂函数和设备的行为
        def _test(factory, device):
            # 定义维度名称 ('N', 'T', 'D')
            names = ('N', 'T', 'D')

            # 设置随机种子，创建张量并指定维度名称和设备
            torch.manual_seed(0)
            result = factory(1, 2, 3, names=names, device=device)

            # 重新设置随机种子，创建预期的张量并重命名维度名称
            torch.manual_seed(0)
            expected = factory(1, 2, 3, device=device).rename_(*names)

            # 断言结果张量与预期张量的数据和名称相等
            self.assertTensorDataAndNamesEqual(result, expected)

        # 定义支持的工厂函数列表和所有设备类型的笛卡尔积
        supported = [
            torch.ones,
            torch.rand,
            torch.randn,
            torch.zeros,
        ]

        # 遍历每种工厂函数和每种设备类型，调用 _test 方法进行测试
        for op, device in itertools.product(supported, get_all_device_types()):
            _test(op, device)

        # 测试 torch.full 函数在每种设备类型下的行为
        for device in get_all_device_types():
            # 定义维度名称 ('N', 'T', 'D')
            names = ('N', 'T', 'D')

            # 创建填充值为 2.0 的张量，指定维度名称和设备
            result = torch.full([1, 2, 3], 2., names=names, device=device)

            # 创建预期的填充值为 2.0 的张量，并重命名维度名称
            expected = torch.full([1, 2, 3], 2., device=device).rename_(*names)

            # 断言结果张量与预期张量的数据和名称相等
            self.assertTensorDataAndNamesEqual(result, expected)

    # 定义测试方法，测试从列表创建张量的行为
    def test_tensor_from_lists(self):
        # 定义维度名称 ('N', 'C')，从列表 [ [1] ] 创建张量，并断言维度名称与定义相等
        names = ('N', 'C')
        tensor = torch.tensor([[1]], names=names)
        self.assertEqual(tensor.names, names)

        # 定义维度名称 ('N')，从列表 [1] 创建张量，并断言维度名称与定义相等
        names = ('N',)
        tensor = torch.tensor([1], names=names)
        self.assertEqual(tensor.names, names)

        # 使用断言捕获 RuntimeError 异常，确保从列表 [1] 创建时抛出异常
        with self.assertRaisesRegex(RuntimeError, 'Number of names'):
            names = ('N', 'C')
            tensor = torch.tensor([1], names=names)

    # 根据测试条件跳过测试，如果未安装 numpy
    @unittest.skipIf(not TEST_NUMPY, "no numpy")
    # 定义测试方法，测试从 numpy 数组创建张量的行为
    def test_tensor_from_numpy(self):
        import numpy as np
        # 创建 numpy 数组 [[1]]，定义维度名称 ('N', 'C')，从中创建张量，并断言维度名称与定义相等
        arr = np.array([[1]])
        names = ('N', 'C')
        tensor = torch.tensor([[1]], names=names)
        self.assertEqual(tensor.names, names)

    # 定义测试方法，测试从另一个张量创建张量的行为
    def test_tensor_from_tensor(self):
        # 创建随机张量 x，定义维度名称 ('N', 'C')，从中创建张量，并断言维度名称与定义相等
        x = torch.randn(1, 1)
        names = ('N', 'C')
        tensor = torch.tensor(x, names=names)
        self.assertEqual(tensor.names, names)
    # 定义一个测试方法，用于验证从命名张量创建张量的不同方式
    def test_tensor_from_named_tensor(self):
        # 创建一个具有命名维度的随机张量
        x = torch.randn(1, 1, names=('N', 'D'))
        # 使用torch.tensor复制张量x，并验证复制后张量的命名维度是否与原始张量相同
        tensor = torch.tensor(x)
        self.assertEqual(tensor.names, ('N', 'D'))

        # 第二次创建具有命名维度的随机张量x
        x = torch.randn(1, 1, names=('N', 'D'))
        # 使用torch.tensor复制张量x，传入names=None，此处注释提示无法区分names=None和不传入names参数的情况
        tensor = torch.tensor(x, names=None)
        self.assertEqual(tensor.names, ('N', 'D'))

        # 第三次创建具有命名维度的随机张量x
        x = torch.randn(1, 1, names=('N', 'D'))
        # 使用torch.tensor复制张量x，并传入与原始张量不同的names参数，此处预期引发运行时错误
        with self.assertRaisesRegex(RuntimeError, "Name mismatch"):
            tensor = torch.tensor(x, names=('N', 'C'))

    # 定义一个测试方法，用于验证命名张量的尺寸操作
    def test_size(self):
        # 创建一个未命名维度的空张量t，并指定其中的命名维度
        t = torch.empty(2, 3, 5, names=('N', None, 'C'))
        # 验证张量t中命名维度'N'和'C'的尺寸是否正确
        self.assertEqual(t.size('N'), 2)
        self.assertEqual(t.size('C'), 5)
        # 验证在张量t中查找未命名的维度'channels'时是否引发预期的运行时错误
        with self.assertRaisesRegex(RuntimeError, 'Name \'channels\' not found in '):
            t.size('channels')
        # 验证在创建未命名维度的空张量时，尝试获取不存在的命名维度'N'是否引发预期的运行时错误
        with self.assertRaisesRegex(RuntimeError, 'Name \'N\' not found in '):
            torch.empty(2, 3, 4).size('N')

    # 定义一个测试方法，用于验证命名张量的步幅操作
    def test_stride(self):
        # 创建一个未命名维度的空张量t，并指定其中的命名维度
        t = torch.empty(2, 3, 5, names=('N', None, 'C'))
        # 验证张量t中命名维度'N'和'C'的步幅是否正确
        self.assertEqual(t.stride('N'), 3 * 5)
        self.assertEqual(t.stride('C'), 1)
        # 验证在张量t中查找未命名的维度'channels'时是否引发预期的运行时错误
        with self.assertRaisesRegex(RuntimeError, 'Name \'channels\' not found in '):
            t.stride('channels')
        # 验证在创建未命名维度的空张量时，尝试获取不存在的命名维度'N'是否引发预期的运行时错误
        with self.assertRaisesRegex(RuntimeError, 'Name \'N\' not found in '):
            torch.empty(2, 3, 4).stride('N')

    # 定义一个测试方法，用于验证张量的转置操作
    def test_transpose_variants(self):
        # 创建一个具有多个命名维度的随机张量t
        t = torch.randn(2, 3, 5, 7, names=('N', 'C', 'H', 'W'))
        # 验证张量t按指定的维度转置后命名维度的顺序是否正确
        self.assertEqual(t.transpose('N', 'C').names, ['C', 'N', 'H', 'W'])
        # 验证张量t按指定的维度转置后命名维度的顺序是否正确
        self.assertEqual(t.transpose(1, 3).names, ['N', 'W', 'H', 'C'])

        # 创建一个具有两个命名维度的随机张量t
        t = torch.randn(2, 3, names=('N', 'C'))
        # 验证张量t的转置操作是否正确修改了命名维度的顺序
        self.assertEqual(t.t().names, ['C', 'N'])

    # 定义一个测试方法，用于验证命名张量的尺寸调整操作
    def test_resize(self):
        # 遍历所有设备类型并执行以下操作
        for device in get_all_device_types():
            # 在指定设备上创建一个具有命名维度的随机张量named
            named = torch.randn(2, names=('N',), device=device)
            # 调整张量named的尺寸为指定大小，并验证调整后的命名维度是否正确
            named.resize_([2])
            self.assertEqual(named.names, ['N'])

            # 尝试对张量named调整为不同大小时，验证是否引发预期的运行时错误
            with self.assertRaisesRegex(RuntimeError, "Cannot resize named tensor"):
                named.resize_([3])

            # 在同一设备上创建另一个具有命名维度的随机张量other_named
            other_named = torch.randn(2, names=('N',), device=device)
            # 使用other_named的尺寸调整张量named，并验证调整后的命名维度是否正确
            named.resize_as_(other_named)
            self.assertEqual(other_named.names, ['N'])

            # 在同一设备上创建一个未命名维度的随机张量unnamed
            unnamed = torch.randn(2, device=device)
            # 尝试将unnamed调整为与named相同的尺寸时，验证是否引发预期的运行时错误
            with self.assertRaisesRegex(
                    RuntimeError, r'names .* are not the same as the computed output names'):
                named.resize_as_(unnamed)

            # 在同一设备上创建一个未命名维度的随机张量unnamed
            unnamed = torch.randn(1, device=device)
            # 将unnamed调整为与named相同的尺寸，并验证调整后的命名维度是否正确
            unnamed.resize_as_(named)
            self.assertEqual(unnamed.names, ['N'])
    # 定义测试函数 test_cdist，用于测试 torch.cdist 函数的命名张量功能
    def test_cdist(self):
        # 遍历所有设备类型
        for device in get_all_device_types():
            # 创建具有命名维度的随机张量 tensor
            tensor = torch.randn(3, 1, 2, 7, names=('M', 'N', 'first_group', 'features'),
                                 device=device)
            # 创建具有命名维度的随机张量 other
            other = torch.randn(5, 11, 7, names=('N', 'second_group', 'features'),
                                device=device)
            # 计算 tensor 和 other 之间的距离张量
            result = torch.cdist(tensor, other)
            # 断言结果张量的命名维度顺序
            self.assertEqual(result.names, ['M', 'N', 'first_group', 'second_group'])

    # 定义测试函数 test_info_smoke，用于测试命名张量的信息函数、方法和属性
    def test_info_smoke(self):
        # 创建空的命名张量 tensor
        tensor = torch.empty(1, 1, names=('N', 'D'))

        # 获取张量的设备属性
        tensor.device
        # 获取张量的数据类型
        tensor.dtype
        # 获取张量所在的设备索引
        tensor.get_device()
        # 检查张量是否为复数类型
        tensor.is_complex()
        # 检查张量是否为浮点数类型
        tensor.is_floating_point()
        # 检查张量是否非零
        tensor.is_nonzero()
        # 检查两个张量是否具有相同的大小
        torch.is_same_size(tensor, tensor)
        # 检查张量元素是否带符号
        torch.is_signed(tensor)
        # 获取张量的布局类型
        tensor.layout
        # 获取张量元素的数量
        tensor.numel()
        # 获取张量的维度数
        tensor.dim()
        # 获取张量中每个元素的字节大小
        tensor.element_size()
        # 检查张量是否是连续存储的
        tensor.is_contiguous()
        # 检查张量是否在 CUDA 设备上
        tensor.is_cuda
        # 检查张量是否为叶子节点
        tensor.is_leaf
        # 检查张量是否固定在内存中
        tensor.is_pinned()
        # 检查张量是否在共享内存上
        tensor.is_shared()
        # 检查张量是否为稀疏张量
        tensor.is_sparse
        # 获取张量的维度数
        tensor.ndimension()
        # 获取张量中元素的总数
        tensor.nelement()
        # 获取张量的形状
        tensor.shape
        # 获取张量的大小
        tensor.size()
        # 获取张量在指定维度的大小
        tensor.size(1)
        # 获取张量的存储对象
        tensor.storage()
        # 获取张量在存储中的偏移量
        tensor.storage_offset()
        # 获取张量的存储类型
        tensor.storage_type()
        # 获取张量的步长
        tensor.stride()
        # 获取张量在指定维度的步长
        tensor.stride(1)
        # 获取张量的数据部分
        tensor.data
        # 获取张量数据的指针
        tensor.data_ptr()
        # 获取张量的维度数
        tensor.ndim
        # 获取张量的标量值
        tensor.item()
        # 获取张量的数据类型
        tensor.type()
        # 检查张量是否在共享内存上
        tensor.is_shared()
        # 检查张量元素是否带符号
        tensor.is_signed()

    # 定义测试函数 test_autograd_smoke，用于测试自动求导功能对命名张量的影响
    def test_autograd_smoke(self):
        # 创建具有命名维度和需求梯度属性的随机张量 x
        x = torch.randn(3, 3, names=('N', 'D'), requires_grad=True)

        # 克隆张量 x 并保留其梯度
        y = x.clone()
        y.retain_grad()
        # 注册一个 hook 函数以接收梯度信息
        y.register_hook(lambda x: x)

        # 计算 y 的和，并进行反向传播
        y.sum().backward()

        # 测试与自动求导相关的属性
        tensor = torch.empty(1, 1, names=('N', 'D'), requires_grad=True)
        # 应用 ReLU 激活函数
        tensor = tensor.relu()
        # 获取输出编号
        tensor.output_nr
        # 获取梯度函数
        tensor.grad_fn
        # 检查张量是否需要梯度
        tensor.requires_grad

    # 定义测试函数 test_split_fns_propagates_names，测试张量分割函数在命名维度上传播的情况
    def test_split_fns_propagates_names(self):
        # 定义需要测试的分割函数列表
        fns = [
            lambda x: x.split(1, 0),
            lambda x: x.split([1, 1], 1),
            lambda x: x.chunk(2, 0),
        ]

        # 遍历所有设备类型
        for device in get_all_device_types():
            # 创建具有命名维度的空张量 orig_tensor
            orig_tensor = torch.empty(2, 2, names=('N', 'D'), device=device)
            # 遍历所有分割函数
            for fn in fns:
                # 对 orig_tensor 应用分割函数 fn
                splits = fn(orig_tensor)
                # 遍历分割结果
                for split in splits:
                    # 断言分割后的张量保持与原始张量相同的命名维度
                    self.assertEqual(split.names, orig_tensor.names)

    # 定义测试函数 test_any_all，测试张量的 any 和 all 方法对命名维度的影响
    def test_any_all(self):
        # 遍历所有设备类型
        for device in get_all_device_types():
            # 创建具有命名维度的零张量 x，数据类型为布尔型
            x = torch.zeros(3, dtype=torch.bool, device=device, names=('C',))
            # 断言 any 方法返回的张量没有命名维度
            self.assertEqual(x.any().names, [])
            # 断言 all 方法返回的张量没有命名维度
            self.assertEqual(x.all().names, [])
    def test_addcmul_addcdiv(self):
        # 遍历所有设备类型进行测试
        for device in get_all_device_types():
            names = ['N']
            # 在指定设备上创建具有指定名称的随机张量 a 和 b
            a = torch.rand(3, device=device, names=names)
            b = torch.rand(3, device=device, names=names)
            # 创建具有指定名称的随机张量 c，并确保不会出现除以 0 的情况
            c = torch.rand(3, device=device, names=names).clamp_min_(0.1)
            # 创建具有指定名称的随机张量 out
            out = torch.randn(3, device=device, names=names)

            # 使用 addcmul 函数进行张量操作，并验证输出张量的名称是否符合预期
            self.assertEqual(torch.addcmul(a, b, c).names, names)
            # 使用 addcmul 函数进行张量操作，将结果存储到指定的输出张量 out，并验证输出张量的名称是否符合预期
            self.assertEqual(torch.addcmul(a, b, c, out=out).names, names)
            # 使用 addcmul_ 函数进行张量操作（inplace），并验证原始张量 a 的名称是否符合预期
            self.assertEqual(a.addcmul_(b, c).names, names)

            # 使用 addcdiv 函数进行张量操作，并验证输出张量的名称是否符合预期
            self.assertEqual(torch.addcdiv(a, b, c).names, names)
            # 使用 addcdiv 函数进行张量操作，将结果存储到指定的输出张量 out，并验证输出张量的名称是否符合预期
            self.assertEqual(torch.addcdiv(a, b, c, out=out).names, names)
            # 使用 addcdiv_ 函数进行张量操作（inplace），并验证原始张量 a 的名称是否符合预期
            self.assertEqual(a.addcdiv_(b, c).names, names)

    def test_logical_ops(self):
        # 使用 TensorIterator 实现逻辑运算，验证每个版本（out-of-place、inplace、out=）是否正确传播名称
        def zeros(*args, **kwargs):
            return torch.zeros(*args, dtype=torch.bool, **kwargs)

        # 遍历逻辑运算符（logical_xor、logical_and、logical_or）
        for op in ('logical_xor', 'logical_and', 'logical_or'):
            # 使用 _test_name_inference 函数验证逻辑运算符的名称推断
            self._test_name_inference(
                getattr(torch, op),
                # 创建具有指定名称的张量，类型为 bool
                (create('N:2,C:3', zeros), create('N:2,C:3', zeros)),
                expected_names=['N', 'C'])

            # 使用 _test_name_inference 函数验证 inplace 版本的逻辑运算符的名称推断
            self._test_name_inference(
                getattr(Tensor, op + '_'),
                # 创建具有指定名称的张量，类型为 bool
                (create('N:2,C:3', zeros), create('N:2,C:3', zeros)),
                expected_names=['N', 'C'])

            # 使用 lambda 函数和 out 参数验证逻辑运算符的名称推断
            self._test_name_inference(
                lambda out, x, y: getattr(torch, op)(x, y, out=out),
                # 创建具有指定名称的张量，类型为 bool
                (create('0', zeros), create('N:2,C:3', zeros), create('N:2,C:3', zeros)),
                expected_names=['N', 'C'])

    def test_pow_special(self):
        # 测试一些不通过 TensorIterator 处理的特殊 pow 情况
        for device in get_all_device_types():
            # 创建具有指定设备和名称的随机张量 named 和 unnamed
            named = torch.randn(2, 3, names=('N', 'C'), device=device)
            unnamed = torch.randn([0], device=device)

            # 使用指定的 unnamed 张量进行 pow 操作，并验证结果张量的名称是否与 named 张量的名称相同
            result = torch.pow(named, 0, out=unnamed.clone())
            self.assertEqual(result.names, named.names)

            # 使用指定的 unnamed 张量进行 pow 操作，并验证结果张量的名称是否与 named 张量的名称相同
            result = torch.pow(named, 1, out=unnamed.clone())
            self.assertEqual(result.names, named.names)

            # 使用指定的 unnamed 张量进行 pow 操作，并验证结果张量的名称是否与 named 张量的名称相同
            result = torch.pow(1, named, out=unnamed.clone())
            self.assertEqual(result.names, named.names)
    def test_out_fn_semantics(self):
        # 使用 torch.abs 函数作为测试对象
        out_fn = torch.abs
        # 创建一个没有命名的张量
        unnamed_tensor = torch.randn(3, 2)
        # 创建一个完全没有命名的张量
        none_named_tensor = torch.randn(3, 2, names=(None, None))
        # 创建一个命名为 ('N', 'C') 的张量
        named_tensor = torch.randn(3, 2, names=('N', 'C'))
        # 创建一个部分命名为 ('N', None) 的张量
        partially_named_tensor = torch.randn(3, 2, names=('N', None))

        # 测试部分命名的张量作为输出和命名张量的匹配情况
        with self.assertRaisesRegex(RuntimeError, "Name mismatch"):
            out_fn(partially_named_tensor, out=named_tensor)
        # 测试命名张量作为输出和部分命名张量的匹配情况
        with self.assertRaisesRegex(RuntimeError, "Name mismatch"):
            out_fn(named_tensor, out=partially_named_tensor)
        # 测试没有命名的张量作为输出和命名张量的匹配情况
        with self.assertRaisesRegex(RuntimeError, "Name mismatch"):
            out_fn(none_named_tensor, out=named_tensor)
        # 测试没有命名的张量作为输出和没有命名的张量的匹配情况
        with self.assertRaisesRegex(RuntimeError, "Name mismatch"):
            out_fn(unnamed_tensor, out=named_tensor)

        # 创建一个用于接收输出的张量，并对未命名的张量进行操作
        output = torch.randn(3, 2)
        out_fn(unnamed_tensor, out=output)
        # 断言输出张量没有命名
        self.assertFalse(output.has_names())

        # 创建一个用于接收输出的张量，并对命名张量进行操作
        output = torch.randn(3, 2, names=(None, None))
        out_fn(named_tensor, out=output)
        # 断言输出张量的命名与原始命名张量相同
        self.assertEqual(output.names, named_tensor.names)

        # 创建一个用于接收输出的张量，并对命名张量进行操作
        output = torch.randn(3, 2)
        out_fn(named_tensor, out=output)
        # 断言输出张量的命名与原始命名张量相同
        self.assertEqual(output.names, named_tensor.names)

        # 创建一个用于接收输出的张量，并对未命名的张量进行操作
        output = torch.randn(3, 2, names=(None, None))
        out_fn(unnamed_tensor, out=output)
        # 断言输出张量没有命名
        self.assertFalse(output.has_names())
    # 定义测试 Bernoulli 分布方法的单元测试函数
    def test_bernoulli(self):
        # 遍历所有设备类型并执行以下操作
        for device in get_all_device_types():
            # 定义张量的维度名称
            names = ('N', 'D')
            # 创建指定维度名称的随机张量
            tensor = torch.rand(2, 3, names=names)
            # 创建一个空张量作为输出
            result = torch.empty(0)
            # 断言张量的 Bernoulli 操作后维度名称不变
            self.assertEqual(tensor.bernoulli().names, names)

            # 在指定张量上执行 Bernoulli 操作，结果存储到 result 张量中
            torch.bernoulli(tensor, out=result)
            # 断言结果张量的维度名称与预期相同
            self.assertEqual(result.names, names)

    # 定义测试张量展平操作的单元测试函数
    def test_flatten(self):
        # 创建指定维度名称的随机张量
        tensor = torch.randn(2, 3, 5, 7, 11, names=('N', 'C', 'D', 'H', 'W'))

        # 基本的展平操作
        out = tensor.flatten('D', 'W', 'features')
        # 断言输出张量的维度名称与预期相同
        self.assertEqual(out.names, ['N', 'C', 'features'])
        # 断言展平后张量的重塑视图与原张量的重塑视图一致
        self.assertEqual(out.rename(None), tensor.rename(None).view(2, 3, -1))

        # 使用整数索引进行展平操作
        out = tensor.flatten(2, 4, 'features')
        # 断言输出张量的维度名称与预期相同
        self.assertEqual(out.names, ['N', 'C', 'features'])
        # 断言展平后张量的重塑视图与原张量的重塑视图一致
        self.assertEqual(out.rename(None), tensor.rename(None).view(2, 3, -1))

        # 使用列表形式的维度名称进行展平操作
        out = tensor.flatten(['D', 'H', 'W'], 'features')
        # 断言输出张量的维度名称与预期相同
        self.assertEqual(out.names, ['N', 'C', 'features'])
        # 断言展平后张量的重塑视图与原张量的重塑视图一致
        self.assertEqual(out.rename(None), tensor.rename(None).view(2, 3, -1))

        # 非连续的展平操作：在内存中，'N' 和 'H' 维度不相邻
        sentences = torch.randn(2, 3, 5, 7, names=('N', 'T', 'H', 'D'))
        # 转置张量，使 'T' 和 'H' 维度相邻
        sentences = sentences.transpose('T', 'H')
        # 执行展平操作，并命名结果维度为 'N_H'
        out = sentences.flatten('N', 'H', 'N_H')
        # 断言输出张量的维度名称与预期相同
        self.assertEqual(out.names, ['N_H', 'T', 'D'])

        # 测试未知维度名称引发的异常
        with self.assertRaisesRegex(RuntimeError, "Name 'L' not found in"):
            tensor.flatten(['D', 'L'], 'features')

        # 测试非连续维度引发的异常
        with self.assertRaisesRegex(RuntimeError, "must be consecutive in"):
            tensor.flatten(['D', 'W'], 'features')

        # 测试非连续维度引发的异常
        with self.assertRaisesRegex(RuntimeError, "must be consecutive in"):
            tensor.flatten(['H', 'D', 'W'], 'features')

    # 定义测试在不指定维度时展平操作引发的异常的单元测试函数
    def test_flatten_nodims(self):
        # 创建空张量
        tensor = torch.empty((2, 3))
        # 断言对空张量执行展平操作引发的异常消息
        with self.assertRaisesRegex(RuntimeError, "cannot be empty"):
            tensor.flatten((), 'abcd')

    # 定义测试在指定维度超出范围时引发的异常的单元测试函数
    def test_flatten_index_error(self):
        # 创建指定维度名称的随机张量
        tensor = torch.randn(1, 2)
        # 断言在指定维度超出范围时引发的索引异常消息
        with self.assertRaisesRegex(IndexError,
                                    r"Dimension out of range \(expected to be in range of \[-2, 1\], but got 2\)"):
            tensor.flatten(0, 2)
        # 断言在指定维度超出范围时引发的索引异常消息
        with self.assertRaisesRegex(IndexError,
                                    r"Dimension out of range \(expected to be in range of \[-2, 1\], but got 2\)"):
            tensor.flatten(0, 2, 'N')
        # 断言在不正确指定起始和结束维度时引发的运行时异常消息
        with self.assertRaisesRegex(RuntimeError,
                                    r"flatten\(\) has invalid args: start_dim cannot come after end_dim"):
            tensor.flatten(1, 0)
        # 断言在不正确指定起始和结束维度时引发的运行时异常消息
        with self.assertRaisesRegex(RuntimeError,
                                    r"flatten\(\) has invalid args: start_dim cannot come after end_dim"):
            tensor.flatten(1, 0, 'N')
    # 测试在使用命名张量时，torch.pdist 抛出错误信息
    def test_unsupported_op_error_msg(self):
        # 创建一个带有名称的3x3的张量
        named = torch.randn(3, 3, names=('N', 'C'))
        # 使用 assertRaisesRegex 确保运行时错误包含特定的错误消息
        with self.assertRaisesRegex(
                RuntimeError, r"pdist.+is not yet supported with named tensors"):
            # 调用 torch.pdist 函数，预期会抛出特定的运行时错误
            torch.pdist(named)
        # 同样的测试，检查另一个不支持的操作
        with self.assertRaisesRegex(
                RuntimeError, r"as_strided_.+is not yet supported with named tensors"):
            # 尝试在命名张量上调用 as_strided_ 方法，预期会抛出运行时错误
            named.as_strided_((3, 3), (3, 1))

    # 测试 torch.masked_select 函数的名称推断
    def test_masked_select(self):
        # 简单情况的名称推断测试
        self._test_name_inference(
            torch.masked_select,
            # 创建输入张量和掩码，并使用 rename 方法重命名维度
            (create('N:2,C:3'), (create('2,3') > 0).rename('N', 'C')),
            expected_names=[None])

        # 左边广播情况的名称推断测试
        self._test_name_inference(
            torch.masked_select,
            # 创建输入张量和左边广播的掩码，并使用 rename 方法重命名维度
            (create('C:3'), (create('2,3') > 0).rename('N', 'C')),
            expected_names=[None])

        # 右边广播情况的名称推断测试
        self._test_name_inference(
            torch.masked_select,
            # 创建输入张量和右边广播的掩码，并使用 rename 方法重命名维度
            (create('N:2,C:3'), (create('3') > 0).rename('C')),
            expected_names=[None])

        # 错误情况的名称推断测试
        self._test_name_inference(
            torch.masked_select,
            # 创建输入张量和不匹配的命名掩码，预期可能会抛出特定错误
            (create('N:2,C:3'), (create('3') > 0).rename('D')),
            maybe_raises_regex='do not match')

        # out= 参数情况的名称推断测试
        self._test_name_inference(
            out_fn(torch.masked_select),
            # 创建输入张量和掩码，并测试 out= 参数的名称推断
            (create('0'), create('N:2,C:3'), (create('2,3') > 0).rename('N', 'C')),
            expected_names=[None])

    # 测试 torch.cat 函数的名称推断
    def test_cat(self):
        # 简单情况的名称推断测试
        self._test_name_inference(
            torch.cat,
            # 创建一个包含两个相同形状张量的列表，并进行拼接
            [[create('N:2,C:3'), create('N:2,C:3')]],
            expected_names=['N', 'C'])

        # 错误情况：零维张量的名称推断测试
        self._test_name_inference(
            torch.cat,
            # 创建一个包含零维张量的列表，预期会抛出特定错误
            [[create(''), create('')]],
            maybe_raises_regex='zero-dim')

        # 错误情况：名称不匹配的名称推断测试
        self._test_name_inference(
            torch.cat,
            # 创建一个包含形状不匹配的两个张量的列表，预期会抛出特定错误
            [[create('N:2,C:3'), create('C:3,N:2')]],
            maybe_raises_regex='do not match')

        # 错误情况：维度数不同的名称推断测试
        self._test_name_inference(
            torch.cat,
            # 创建一个包含维度数不同的两个张量的列表，预期会抛出特定错误
            [[create('N:2,C:3'), create('C:3')]],
            maybe_raises_regex='must have same number of dimensions')

        # out= 参数情况的名称推断测试
        self._test_name_inference(
            out_fn(torch.cat),
            # 创建一个包含0维张量和两个相同形状张量的列表，并测试 out= 参数的名称推断
            [create('0'), [create('N:2,C:3'), create('N:2,C:3')]],
            expected_names=['N', 'C'])
    # 定义测试方法，用于测试 Tensor 对象的 masked_fill 方法
    def test_masked_fill(self):
        # 测试用例：简单情况下的名称推断
        self._test_name_inference(
            Tensor.masked_fill,  # 调用 Tensor 对象的 masked_fill 方法
            (create('N:2,C:3'), (create('2,3') > 0).rename('N', 'C'), 3.14),  # 参数包括创建的张量和掩码条件
            expected_names=['N', 'C'])  # 期望的输出张量维度名称

        # 测试用例：左侧广播
        self._test_name_inference(
            Tensor.masked_fill,
            (create('C:3'), (create('2,3') > 0).rename('N', 'C'), 3.14),
            maybe_raises_regex="must be less than or equal to")  # 可能会引发异常的正则表达式

        # 测试用例：右侧广播
        self._test_name_inference(
            Tensor.masked_fill,
            (create('N:2,C:3'), (create('3') > 0).rename('C'), 3.14),
            expected_names=['N', 'C'])

        # 测试用例：错误情况
        self._test_name_inference(
            Tensor.masked_fill,
            (create('N:2,C:3'), (create('3') > 0).rename('D'), 3.14),
            maybe_raises_regex='do not match')  # 可能会引发异常的字符串

        # 测试用例：原地操作
        self._test_name_inference(
            Tensor.masked_fill_,
            (create('N:2,C:3'), (create('2,3') > 0).rename('N', 'C'), 3.14),
            expected_names=['N', 'C'])

        # 测试用例：原地操作，计算得到的名称与输出张量名称不匹配
        self._test_name_inference(
            Tensor.masked_fill_,
            (create('N:2,None:3'), (create('2,3') > 0).rename('N', 'C'), 3.14),
            maybe_raises_regex="not the same as the computed output names")


    # 测试函数：测试使用已见过的内部字符串是否会增加引用计数
    def test_using_seen_interned_string_doesnt_bump_refcount(self):
        def see_name():
            seen_name = 'N'  # 见过的字符串
            pass_name_to_python_arg_parser(seen_name)

        see_name()
        seen_name = 'N'
        old_refcnt = sys.getrefcount(seen_name)

        pass_name_to_python_arg_parser(seen_name)

        new_refcnt = sys.getrefcount(seen_name)
        self.assertEqual(new_refcnt, old_refcnt)  # 断言引用计数未增加

    # 这个测试在 Python 3.12 上失败：https://github.com/pytorch/pytorch/issues/119464
    @unittest.skipIf(sys.version_info >= (3, 12), "Failing on python 3.12+")
    # 测试函数：测试使用未见过的内部字符串是否会永久增加引用计数
    def test_using_unseen_interned_string_bumps_refcount_permanently(self):
        # 请不要在其他测试中使用此名称。
        unseen_name = 'abcdefghi'  # 未见过的字符串
        old_refcnt = sys.getrefcount(unseen_name)

        pass_name_to_python_arg_parser(unseen_name)

        new_refcnt = sys.getrefcount(unseen_name)
        self.assertEqual(new_refcnt, old_refcnt + 1)  # 断言引用计数增加了一次
    def test_using_unseen_uninterned_string_refcounts(self):
        # 在测试中使用未见过且未被国际化的字符串引用计数
        # 非编译时常量不会被国际化
        unseen_name = ''.join(['abc', 'def', 'ghi', 'jkl'])
        interned_unseen_name = 'abcdefghijkl'
        self.assertFalse(unseen_name is interned_unseen_name)

        # 获取未见过字符串的引用计数
        old_uninterned_refcnt = sys.getrefcount(unseen_name)
        # 获取已国际化字符串的引用计数
        old_interned_refcnt = sys.getrefcount(interned_unseen_name)

        # 将未见过的字符串传递给 Python 参数解析器
        pass_name_to_python_arg_parser(unseen_name)

        # 获取传递后未见过字符串的新引用计数
        new_uninterned_refcnt = sys.getrefcount(unseen_name)
        # 获取传递后已国际化字符串的新引用计数
        new_interned_refcnt = sys.getrefcount(interned_unseen_name)

        # 内部应该不会持有未国际化的字符串引用
        self.assertEqual(new_uninterned_refcnt, old_uninterned_refcnt)

        # 相反，应该会持有已国际化版本的新引用
        self.assertEqual(new_interned_refcnt, old_interned_refcnt + 1)

    def _test_select(self, device):
        # 创建一个形状为 (2, 3, 4, 5) 的空张量，并指定维度名称
        x = torch.empty(2, 3, 4, 5, names=('N', 'C', 'H', 'W'), device=device)
        # 选择指定维度的切片，并断言切片的维度名称
        y = x.select(1, 1)
        self.assertEqual(y.names, ('N', 'H', 'W'))

        # 使用维度名称选择切片，并断言切片的维度名称
        y = x.select('C', 1)
        self.assertEqual(y.names, ('N', 'H', 'W'))

        # 使用 None 作为维度选择参数，断言引发 RuntimeError 异常
        with self.assertRaisesRegex(
                RuntimeError, 'Please look up dimensions by name'):
            y = x.select(None, 1)

    def test_select(self):
        # 在 CPU 上运行 _test_select 方法
        self._test_select('cpu')

    @unittest.skipIf(not TEST_CUDA, 'no CUDA')
    def test_select_cuda(self):
        # 在 CUDA 上运行 _test_select 方法
        self._test_select('cuda')

    def _test_as_strided(self, device):
        # 创建一个形状为 (2, 3, 4, 5) 的空张量，并指定维度名称
        x = torch.empty(2, 3, 4, 5, names=('N', 'C', 'H', 'W'), device=device)
        # 使用 as_strided 方法创建一个视图，并断言其维度名称
        y = x.as_strided([2 * 3 * 4 * 5], [1])
        self.assertEqual(y.names, (None,))

    def test_as_strided(self):
        # 在 CPU 上运行 _test_as_strided 方法
        self._test_as_strided('cpu')

    @unittest.skipIf(not TEST_CUDA, 'no CUDA')
    def test_as_strided_cuda(self):
        # 在 CUDA 上运行 _test_as_strided 方法
        self._test_as_strided('cuda')

    def test_no_jit_tracer_support(self):
        # 定义一个函数 foo，返回一个指定形状和名称的全 2 张量
        def foo(x):
            return torch.full(x.shape, 2., names=('N',))

        # 断言在使用跟踪器时引发 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, 'not supported with the tracer'):
            x = torch.randn(3)
            torch.jit.trace(foo, example_inputs=x)

        # 定义一个函数 bar，使用维度名称 'N' 进行选择
        def bar(x):
            return x.select('N', 1)

        # 断言在使用跟踪器时引发 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, 'not supported with the tracer'):
            x = torch.randn(3)
            torch.jit.trace(bar, example_inputs=x)
    def test_no_jit_script_support(self):
        # 定义一个使用了 Torch JIT 脚本装饰器的函数 foo，实现对输入张量加一的操作
        @torch.jit.script
        def foo(x):
            return x + 1

        # 测试在具有不支持的命名维度的张量输入时，是否会抛出 RuntimeError 异常，并包含 'NYI' 字符串
        with self.assertRaisesRegex(RuntimeError, 'NYI'):
            foo(torch.randn(2, 3, names=('N', 'C')))

        # 定义一个使用了 Torch JIT 忽略装饰器的函数 add_names，用于设置张量的命名维度
        @torch.jit.ignore
        def add_names(x):
            x.names = ('N', 'C')

        # 定义一个使用了 Torch JIT 脚本装饰器的函数 return_named_tensor，调用 add_names 函数并返回输入张量
        @torch.jit.script
        def return_named_tensor(input):
            add_names(input)
            return input

        # 测试在调用 return_named_tensor 函数时，是否会抛出 RuntimeError 异常，并包含 "NYI" 字符串
        with self.assertRaisesRegex(RuntimeError, "NYI"):
            return_named_tensor(torch.randn(1, 1))

    def test_align_to(self):
        # 创建一个带有命名维度的张量 tensor
        tensor = create('N:3')
        # 使用 align_to 方法对张量进行对齐，使其仅包含 'N' 维度
        output = tensor.align_to('N')
        # 断言输出张量的命名维度为 ['N']
        self.assertEqual(output.names, ['N'])
        # 断言输出张量的形状为 [3]
        self.assertEqual(output.shape, [3])

        # 创建一个带有命名维度的张量 tensor
        tensor = create('N:3')
        # 使用 align_to 方法对张量进行对齐，使其包含 'N' 和 'D' 两个命名维度
        output = tensor.align_to('N', 'D')
        # 断言输出张量的命名维度为 ['N', 'D']
        self.assertEqual(output.names, ['N', 'D'])
        # 断言输出张量的形状为 [3, 1]

        # 创建一个带有命名维度的张量 tensor
        tensor = create('N:3,C:2')
        # 使用 align_to 方法对张量进行对齐，使其命名维度变为 'C', 'N'
        output = tensor.align_to('C', 'N')
        # 断言输出张量的命名维度为 ['C', 'N']
        self.assertEqual(output.names, ['C', 'N'])
        # 断言输出张量的形状为 [2, 3]

        # 创建一个带有命名维度的张量 tensor
        tensor = create('C:2,N:3,H:5')
        # 使用 align_to 方法对张量进行对齐，使其命名维度变为 'N', 'H', 'W', 'C'
        output = tensor.align_to('N', 'H', 'W', 'C')
        # 断言输出张量的命名维度为 ['N', 'H', 'W', 'C']
        self.assertEqual(output.names, ['N', 'H', 'W', 'C'])
        # 断言输出张量的形状为 [3, 5, 1, 2]

        # 测试当输入张量存在未命名维度时，是否会抛出 RuntimeError 异常，并包含指定字符串
        with self.assertRaisesRegex(RuntimeError, "All input dims must be named. Found unnamed dim at index 0"):
            create('None:2,C:3').align_to('N', 'C')

        # 测试当输入张量缺少命名维度时，是否会抛出 RuntimeError 异常，并包含指定字符串
        with self.assertRaisesRegex(RuntimeError, "Cannot find dim 'N'"):
            create('N:2,C:3').align_to('C')

        # 测试当输入张量中找不到指定的命名维度时，是否会抛出 RuntimeError 异常，并包含指定字符串
        with self.assertRaisesRegex(RuntimeError, "Cannot find dim 'C'"):
            create('N:2,C:3').align_to('D', 'N')
    def test_align_to_ellipsis(self):
        # 创建一个名为 tensor 的张量，形状为 'N:7,H:3,W:5,C:2'
        tensor = create('N:7,H:3,W:5,C:2')

        # 使用省略号 '...' 对张量进行维度对齐
        output = tensor.align_to('...')
        # 断言输出的维度名称为 ['N', 'H', 'W', 'C']
        self.assertEqual(output.names, ['N', 'H', 'W', 'C'])
        # 断言输出的形状为 [7, 3, 5, 2]
        self.assertEqual(output.shape, [7, 3, 5, 2])

        # 使用省略号 '...'，以及额外的维度 'W', 'N' 对张量进行维度对齐
        output = tensor.align_to('...', 'W', 'N')
        # 断言输出的维度名称为 ['H', 'C', 'W', 'N']
        self.assertEqual(output.names, ['H', 'C', 'W', 'N'])
        # 断言输出的形状为 [3, 2, 5, 7]

        # 使用维度 'H', 'C'，以及省略号 '...' 对张量进行维度对齐
        output = tensor.align_to('H', 'C', '...')
        # 断言输出的维度名称为 ['H', 'C', 'N', 'W']
        self.assertEqual(output.names, ['H', 'C', 'N', 'W'])
        # 断言输出的形状为 [3, 2, 7, 5]

        # 使用维度 'W', 省略号 '...', 'N' 对张量进行维度对齐
        output = tensor.align_to('W', '...', 'N')
        # 断言输出的维度名称为 ['W', 'H', 'C', 'N']
        self.assertEqual(output.names, ['W', 'H', 'C', 'N'])
        # 断言输出的形状为 [5, 3, 2, 7]

        # 使用维度 'N', 'C'，省略号 '...', 'D', 'H', 'W' 对张量进行维度对齐
        output = tensor.align_to('N', '...', 'C', 'D', 'H', 'W')
        # 断言输出的维度名称为 ['N', 'C', 'D', 'H', 'W']
        self.assertEqual(output.names, ['N', 'C', 'D', 'H', 'W'])
        # 断言输出的形状为 [7, 2, 1, 3, 5]

        # 输入张量部分命名
        partially_named = create('None:2,None:3,None:5,C:7')
        # 使用维度 'C', 省略号 '...' 对部分命名的张量进行维度对齐
        output = partially_named.align_to('C', '...')
        # 断言输出的维度名称为 ['C', None, None, None]
        self.assertEqual(output.names, ['C', None, None, None])
        # 断言输出的形状为 [7, 2, 3, 5]

        # 引发异常，因为维度顺序中包含了 None
        with self.assertRaisesRegex(RuntimeError, "order of dimensions cannot contain a None"):
            partially_named.align_to('C', None, '...')

        # 输入顺序部分命名
        with self.assertRaisesRegex(RuntimeError, "cannot contain a None name"):
            # 引发异常，因为输入顺序中包含重复的维度名称 'N'
            tensor.align_to('...', 'N', None)

        # 输入顺序重复的维度名称
        with self.assertRaisesRegex(RuntimeError, "duplicate names"):
            # 引发异常，因为输入顺序中包含重复的维度名称 'N'
            tensor.align_to('...', 'N', 'N')
    # 定义测试方法 test_mm，用于测试 torch.mm 函数的名称推断功能
    def test_mm(self):
        # 遍历所有设备类型
        for device in get_all_device_types():
            # 测试名称推断函数 _test_name_inference 的结果，期望左右参数的名称为 'N' 和 'H'
            self._test_name_inference(
                torch.mm, device=device,
                args=(create('N:3,C:2'), create('W:2,H:5')),
                expected_names=('N', 'H'))

            # 左参数未命名的情况下进行名称推断测试
            self._test_name_inference(
                torch.mm, device=device,
                args=(create('3,2'), create('W:2,H:5')),
                expected_names=(None, 'H'))

            # 右参数未命名的情况下进行名称推断测试
            self._test_name_inference(
                torch.mm, device=device,
                args=(create('N:3,C:2'), create('2,5')),
                expected_names=('N', None))

            # 测试输出参数为命名的情况下的名称推断
            self._test_name_inference(
                out_fn(torch.mm), device=device,
                args=(create('0'), create('N:3,C:2'), create('W:2,H:5')),
                expected_names=('N', 'H'))

            # 测试参数中含有重复名称时是否会触发异常，期望抛出带有 'with duplicate names' 字符串的异常
            self._test_name_inference(
                torch.mm, device=device,
                args=(create('N:3,C:2'), create('W:2,N:5')),
                maybe_raises_regex='with duplicate names')

    # 定义测试方法 test_expand，用于测试 Tensor.expand 方法的名称推断功能
    def test_expand(self):
        # 遍历所有设备类型
        for device in get_all_device_types():
            # 测试名称推断函数 _test_name_inference 的结果，期望扩展后张量的名称为 'D'
            self._test_name_inference(
                Tensor.expand, device=device,
                args=(create('D:1'), [3]), expected_names=('D',))

            # 测试名称推断函数 _test_name_inference 的结果，期望扩展后张量的名称为 'H' 和 'W'，其它维度不命名
            self._test_name_inference(
                Tensor.expand, device=device,
                args=(create('H:3,W:2'), [10, 3, 3, 2]),
                expected_names=(None, None, 'H', 'W'))

            # 测试名称推断函数 _test_name_inference 的结果，所有维度均不命名
            self._test_name_inference(
                Tensor.expand, device=device,
                args=(create('3, 2'), [10, 3, 3, 2]),
                expected_names=(None, None, None, None))
    # 定义一个名为 test_addmm 的测试方法，使用了 self 参数表示这是一个类方法
    def test_addmm(self):
        # 对所有设备类型进行迭代测试
        for device in get_all_device_types():
            # 测试 torch.addmm 函数的名称推断，指定设备类型和参数
            self._test_name_inference(
                torch.addmm, device=device,
                args=(create('N:3,H:5'), create('N:3,C:2'), create('W:2,H:5')),
                expected_names=('N', 'H'))

            # 测试 torch.addmm 函数，不给偏置参数命名
            self._test_name_inference(
                torch.addmm, device=device,
                args=(create('3,5'), create('N:3,C:2'), create('W:2,H:5')),
                expected_names=('N', 'H'))

            # 测试 torch.addmm 函数，部分命名偏置参数
            self._test_name_inference(
                torch.addmm, device=device,
                args=(create('N:3,None:5'), create('N:3,C:2'), create('W:2,H:5')),
                expected_names=('N', 'H'))

            # 测试 torch.addmm 函数使用 out 参数
            self._test_name_inference(
                out_fn(torch.addmm), device=device,
                args=(create('0'), create('N:3,None:5'), create('N:3,C:2'), create('W:2,H:5')),
                expected_names=('N', 'H'))

            # 测试 torch.Tensor.addmm_ 方法的名称推断，使用 inplace 操作
            self._test_name_inference(
                torch.Tensor.addmm_, device=device,
                args=(create('N:3,H:5'), create('N:3,C:2'), create('W:2,H:5')),
                expected_names=('N', 'H'))

            # 测试 torch.addmm 函数，传入具有重复名称的参数，期待引发异常
            self._test_name_inference(
                torch.addmm, device=device,
                args=(create('N:3,H:5'), create('N:3,C:2'), create('W:2,N:5')),
                maybe_raises_regex='with duplicate names')
    # 定义测试函数 test_bmm，用于测试 torch.bmm 函数的参数推断
    def test_bmm(self):
        # 对所有设备类型进行迭代测试
        for device in get_all_device_types():
            # 测试完整名称的推断
            self._test_name_inference(
                torch.bmm, device=device,
                args=(create('N:7,A:3,B:2'), create('N:7,A:2,B:5')),
                expected_names=('N', 'A', 'B'))

            # 左张量无名称
            self._test_name_inference(
                torch.bmm, device=device,
                args=(create('7,3,2'), create('N:7,A:2,B:5')),
                expected_names=('N', None, 'B'))

            # 右张量无名称
            self._test_name_inference(
                torch.bmm, device=device,
                args=(create('N:7,A:3,B:2'), create('7,2,5')),
                expected_names=('N', 'A', None))

            # out= 的推断
            self._test_name_inference(
                out_fn(torch.bmm), device=device,
                args=(create('0'), create('N:7,A:3,B:2'), create('N:7,A:2,B:5')),
                expected_names=('N', 'A', 'B'))

            # mm 后出现重复的名称
            self._test_name_inference(
                torch.bmm, device=device,
                args=(create('N:7,A:3,B:2'), create('N:7,B:2,A:5')),
                maybe_raises_regex='with duplicate names')

            # 匹配错误（批次维度必须对齐）
            self._test_name_inference(
                torch.bmm, device=device,
                args=(create('N:3,A:3,B:3'), create('M:3,A:3,B:3')),
                maybe_raises_regex='do not match')

            # 不对齐（批次维度被压缩）
            self._test_name_inference(
                torch.bmm, device=device,
                args=(create('N:3,A:3,B:3'), create('None:3,N:3,B:3')),
                maybe_raises_regex='misaligned')

    # 定义测试函数 test_mv，用于测试 torch.mv 函数的参数推断
    def test_mv(self):
        # 对所有设备类型进行迭代测试
        for device in get_all_device_types():
            # 测试推断
            self._test_name_inference(
                torch.mv, device=device,
                args=(create('N:3,C:2'), create('W:2')),
                expected_names=('N',))

            # 左参数没有名称
            self._test_name_inference(
                torch.mv, device=device,
                args=(create('3,2'), create('W:2')),
                expected_names=(None,))

            # 右参数没有名称
            self._test_name_inference(
                torch.mv, device=device,
                args=(create('N:3,C:2'), create('2')),
                expected_names=('N',))

            # out= 的推断
            self._test_name_inference(
                out_fn(torch.mv), device=device,
                args=(create('0'), create('N:3,C:2'), create('W:2')),
                expected_names=('N',))
    def test_addmv(self):
        for device in get_all_device_types():
            # 针对所有设备类型进行测试

            # 使用 torch.addmv 函数进行名称推断测试，指定设备为当前设备
            self._test_name_inference(
                torch.addmv, device=device,
                args=(create('N:3'), create('N:3,C:2'), create('H:2')),
                expected_names=['N'])

            # 测试时不传递偏置的名称
            self._test_name_inference(
                torch.addmv, device=device,
                args=(create('3'), create('N:3,C:2'), create('H:2')),
                expected_names=('N',))

            # 测试使用 out 参数的名称推断
            self._test_name_inference(
                out_fn(torch.addmv), device=device,
                args=(create('0'), create('N:3'), create('N:3,C:2'), create('H:2')),
                expected_names=('N',))

            # 测试 inplace 操作的名称推断
            self._test_name_inference(
                torch.Tensor.addmv_, device=device,
                args=(create('N:3'), create('N:3,C:2'), create('H:2')),
                expected_names=('N',))

    def test_autograd_ignores_names(self):
        # sigmoid 前向传播支持命名张量，但 sigmoid 反向传播不支持
        # 测试 autograd 是否忽略名称，并且 sigmoid 反向传播成功执行
        x = torch.randn(3, 3, names=('N', 'C'), requires_grad=True)
        x.sigmoid().sum().backward()

    def test_tensor_grad_is_unnamed(self):
        # 测试张量乘积的梯度是否没有名称
        x = torch.randn(3, 3, names=(None, None), requires_grad=True)
        y = torch.randn(3, 3, names=('N', 'C'), requires_grad=True)
        (x * y).sum().backward()

        # 检查梯度是否没有传播名称
        self.assertEqual(y.grad.names, [None, None])
        self.assertEqual(x.grad.names, [None, None])

    def test_autograd_warns_named_grad(self):
        # 测试 autograd 是否会警告命名的梯度张量
        base = torch.randn(3, 3, names=('N', 'C'))
        named_grad = base.clone()
        base.requires_grad_()

        with warnings.catch_warnings(record=True) as warns:
            # 让所有警告都被触发
            warnings.simplefilter("always")
            base.clone().backward(named_grad)
            self.assertEqual(len(warns), 1)
            self.assertTrue(
                str(warns[0].message).startswith('Autograd was passed a named grad tensor'))

    def test_nyi_dimname_overload_msg(self):
        # 测试当传递 dimname 给 squeeze 函数时的异常消息
        x = torch.randn(3, 3)
        with self.assertRaisesRegex(RuntimeError, "squeeze: You passed a dimname"):
            x.squeeze_("N")

    def test_dot(self):
        for device in get_all_device_types():
            # torch.dot 忽略两个张量的名称
            self._test_name_inference(
                torch.dot, device=device,
                args=(create('C:2'), create('W:2')),
                expected_names=[])
    # 定义测试函数，用于测试张量的比较运算符
    def test_comparison_ops(self):
        # 遍历获取所有设备类型的函数返回结果
        for device in get_all_device_types():
            # 创建形状为 (3, 3) 的张量 a 和 b，张量具有命名维度 'N' 和 'C'，使用指定设备
            a = torch.randn(3, 3, names=('N', 'C'), device=device)
            b = torch.randn(3, 3, names=('N', 'C'), device=device)
            # 创建一个随机标量张量，使用指定设备
            scalar = torch.randn([], device=device)

            # 断言张量 a 和 b 的相等性，检查结果的命名维度是否为 ['N', 'C']
            self.assertEqual((a == b).names, ['N', 'C'])
            # 断言张量 a 和 b 的不等性，检查结果的命名维度是否为 ['N', 'C']
            self.assertEqual((a != b).names, ['N', 'C'])
            # 断言张量 a 大于 b 的逐元素比较结果的命名维度是否为 ['N', 'C']
            self.assertEqual((a > b).names, ['N', 'C'])
            # 断言张量 a 小于 b 的逐元素比较结果的命名维度是否为 ['N', 'C']
            self.assertEqual((a < b).names, ['N', 'C'])
            # 断言张量 a 大于等于 b 的逐元素比较结果的命名维度是否为 ['N', 'C']
            self.assertEqual((a >= b).names, ['N', 'C'])
            # 断言张量 a 小于等于 b 的逐元素比较结果的命名维度是否为 ['N', 'C']

            self.assertEqual((a <= b).names, ['N', 'C'])

            # 断言张量 a 等于标量 1 的逐元素比较结果的命名维度是否为 ['N', 'C']
            self.assertEqual((a == 1).names, ['N', 'C'])
            # 断言张量 a 不等于标量 1 的逐元素比较结果的命名维度是否为 ['N', 'C']
            self.assertEqual((a != 1).names, ['N', 'C'])
            # 断言张量 a 大于标量 1 的逐元素比较结果的命名维度是否为 ['N', 'C']
            self.assertEqual((a > 1).names, ['N', 'C'])
            # 断言张量 a 小于标量 1 的逐元素比较结果的命名维度是否为 ['N', 'C']
            self.assertEqual((a < 1).names, ['N', 'C'])
            # 断言张量 a 大于等于标量 1 的逐元素比较结果的命名维度是否为 ['N', 'C']
            self.assertEqual((a >= 1).names, ['N', 'C'])
            # 断言张量 a 小于等于标量 1 的逐元素比较结果的命名维度是否为 ['N', 'C']

            self.assertEqual((a <= 1).names, ['N', 'C'])

            # 断言张量 a 等于标量 scalar 的逐元素比较结果的命名维度是否为 ['N', 'C']
            self.assertEqual((a == scalar).names, ['N', 'C'])
            # 断言张量 a 不等于标量 scalar 的逐元素比较结果的命名维度是否为 ['N', 'C']
            self.assertEqual((a != scalar).names, ['N', 'C'])
            # 断言张量 a 大于标量 scalar 的逐元素比较结果的命名维度是否为 ['N', 'C']
            self.assertEqual((a > scalar).names, ['N', 'C'])
            # 断言张量 a 小于标量 scalar 的逐元素比较结果的命名维度是否为 ['N', 'C']
            self.assertEqual((a < scalar).names, ['N', 'C'])
            # 断言张量 a 大于等于标量 scalar 的逐元素比较结果的命名维度是否为 ['N', 'C']
            self.assertEqual((a >= scalar).names, ['N', 'C'])
            # 断言张量 a 小于等于标量 scalar 的逐元素比较结果的命名维度是否为 ['N', 'C']

            self.assertEqual((a <= scalar).names, ['N', 'C'])

            # 创建一个形状为 (3, 3) 的布尔型张量 res，使用指定设备
            res = torch.empty(3, 3, dtype=torch.bool, device=device)
            # 使用 torch.eq 计算张量 a 和 b 的逐元素相等性，并将结果存入 res
            torch.eq(a, b, out=res)
            # 断言 res 的命名维度是否为 ['N', 'C']
            self.assertEqual(res.names, ['N', 'C'])
            # 使用 torch.ne 计算张量 a 和 b 的逐元素不等性，并将结果存入 res
            torch.ne(a, b, out=res)
            # 断言 res 的命名维度是否为 ['N', 'C']
            self.assertEqual(res.names, ['N', 'C'])
            # 使用 torch.lt 计算张量 a 小于 b 的逐元素结果，并将结果存入 res
            torch.lt(a, b, out=res)
            # 断言 res 的命名维度是否为 ['N', 'C']
            self.assertEqual(res.names, ['N', 'C'])
            # 使用 torch.gt 计算张量 a 大于 b 的逐元素结果，并将结果存入 res
            torch.gt(a, b, out=res)
            # 断言 res 的命名维度是否为 ['N', 'C']
            self.assertEqual(res.names, ['N', 'C'])
            # 使用 torch.le 计算张量 a 小于等于 b 的逐元素结果，并将结果存入 res
            torch.le(a, b, out=res)
            # 断言 res 的命名维度是否为 ['N', 'C']
            self.assertEqual(res.names, ['N', 'C'])
            # 使用 torch.ge 计算张量 a 大于等于 b 的逐元素结果，并将结果存入 res
            torch.ge(a, b, out=res)
            # 断言 res 的命名维度是否为 ['N', 'C']

            self.assertEqual(res.names, ['N', 'C'])

            # 计算张量 a 中的 NaN 值，并将结果存入 res
            res = torch.isnan(a)
            # 断言 res 的命名维度是否为 ['N', 'C']
            self.assertEqual(res.names, ['N', 'C'])

            # 计算张量 a 中的无穷大值，并将结果存入 res
            res = torch.isinf(a)
            # 断言 res 的命名维度是否为 ['N', 'C']

            self.assertEqual(res.names, ['N', 'C'])

    # 定义测试函数，用于测试支持带名称梯度的设备
    def test_support_device_named_grad(self):
        # 创建一个具有 'meta' 设备的随机张量 named_tensor
        named_tensor = torch.randn(3, 3, device='meta')
        # 使用断言检查在使用命名张量时是否引发了预期的运行时错误
        with self.assertRaisesRegex(RuntimeError, 'NYI: named tensors only support CPU, CUDA'):
            # 尝试使用 rename_ 方法重命名命名张量的维度 'N' 为 'C'
            named_tensor.rename_('N', 'C')
            # 尝试设置命名张量的维度名称为 ['N', 'C']
            named_tensor.names = ['N', 'C']
            # 重新创建一个具有 'meta' 设备和命名维度 ['N', 'C'] 的随机张量 named_tensor
            named_tensor = torch.randn(3, 3, device='meta', names=['N', 'C'])
# 如果当前脚本被直接执行（而非被导入为模块），则执行以下代码块
if __name__ == '__main__':
    # 调用名为 run_tests 的函数来运行测试
    run_tests()
```