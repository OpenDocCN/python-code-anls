# `.\pytorch\test\test_maskedtensor.py`

```
# Owner(s): ["module: masked operators"]

# 导入 PyTorch 库
import torch
# 导入测试相关的模块和函数
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    make_tensor,
    parametrize,
    instantiate_parametrized_tests,
)
# 导入设备相关的测试函数和对象
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
# 导入常见方法和调用函数
from torch.testing._internal.common_methods_invocations import (
    SampleInput,
    binary_ufuncs,
    reduction_ops,
    unary_ufuncs,
)

# 导入 torch.masked 模块中的相关函数和类
from torch.masked import as_masked_tensor, masked_tensor, _combine_input_and_mask
# 导入 torch.masked.maskedtensor.core 模块中的函数
from torch.masked.maskedtensor.core import _masks_match, _tensors_match
# 导入 torch.masked.maskedtensor.unary 模块中的函数和名称列表
from torch.masked.maskedtensor.unary import NATIVE_INPLACE_UNARY_FNS, NATIVE_UNARY_FNS, UNARY_NAMES
# 导入 torch.masked.maskedtensor.binary 模块中的函数和名称列表
from torch.masked.maskedtensor.binary import NATIVE_BINARY_FNS, NATIVE_INPLACE_BINARY_FNS, BINARY_NAMES
# 导入 torch.masked.maskedtensor.reductions 模块中的名称列表
from torch.masked.maskedtensor.reductions import REDUCE_NAMES

# 比较 MaskedTensor 和普通 Tensor 的数据是否匹配
def _compare_mt_t(mt_result, t_result, rtol=1e-05, atol=1e-05):
    # 获取 MaskedTensor 的掩码和数据
    mask = mt_result.get_mask()
    mt_result_data = mt_result.get_data()
    # 如果掩码是稀疏格式，则转换为稠密格式
    if mask.layout in {torch.sparse_coo, torch.sparse_csr}:
        mask = mask.to_dense()
    # 如果数据是稀疏格式，则转换为稠密格式
    if mt_result_data.layout in {torch.sparse_coo, torch.sparse_csr}:
        mt_result_data = mt_result_data.to_dense()
    # 使用掩码填充数据，生成新的 Tensor a 和 b
    a = mt_result_data.detach().masked_fill_(~mask, 0)
    b = t_result.detach().masked_fill_(~mask, 0)
    # 比较 Tensor a 和 Tensor b 的数据是否匹配
    if not _tensors_match(a, b, exact=False, rtol=rtol, atol=atol):
        raise ValueError("The data in MaskedTensor a and Tensor b do not match")

# 比较两个 MaskedTensor 对象的数据是否匹配
def _compare_mts(mt1, mt2, rtol=1e-05, atol=1e-08):
    # 获取两个 MaskedTensor 对象的数据
    mt_data1 = mt1.get_data()
    mt_data2 = mt2.get_data()
    # 如果两个对象的数据布局不同，抛出数值错误
    if mt_data1.layout != mt_data2.layout:
        raise ValueError("mt1's data and mt2's data do not have the same layout. "
                         f"mt1.get_data().layout = {mt_data1.layout} while mt2.get_data().layout = {mt_data2.layout}")

    # 获取两个 MaskedTensor 对象的掩码
    mask = mt1.get_mask()
    mask2 = mt2.get_mask()
    # 如果两个对象的掩码不匹配，抛出数值错误
    if not _masks_match(mt1, mt2):
        raise ValueError("mt1 and mt2 must have matching masks")
    # 如果两个掩码的布局不同，抛出数值错误
    if mask.layout != mask2.layout:
        raise ValueError("mt1's mask and mt2's mask do not have the same layout. "
                         f"mt1.get_mask().layout = {mask.layout} while mt2.get_mask().layout = {mask2.layout}")
    # 如果掩码是稀疏格式，则转换为稠密格式
    if mask.layout in {torch.sparse_coo, torch.sparse_csr}:
        mask = mask.to_dense()

    # 如果数据是稀疏格式，则转换为稠密格式
    if mt_data1.layout in {torch.sparse_coo, torch.sparse_csr}:
        mt_data1 = mt_data1.to_dense()
        mt_data2 = mt_data2.to_dense()
    # 使用掩码填充数据，生成新的 Tensor a 和 b
    a = mt_data1.detach().masked_fill_(~mask, 0)
    b = mt_data2.detach().masked_fill_(~mask, 0)

    # 比较 Tensor a 和 Tensor b 的数据是否匹配
    if not _tensors_match(a, b, exact=False, rtol=rtol, atol=atol):
        raise ValueError("The data in MaskedTensor mt1 and MaskedTensor mt2 do not match")

# 创建一个随机掩码，形状由参数 shape 决定，存储在指定设备上
def _create_random_mask(shape, device):
    return make_tensor(shape, device=device, dtype=torch.bool)

# 生成样本数据的辅助函数，可以指定设备、数据类型、是否需要梯度和布局
def _generate_sample_data(
    device="cpu", dtype=torch.float, requires_grad=True, layout=torch.strided
):
    # 确保布局参数在支持的范围内，否则引发断言错误
    assert layout in {
        torch.strided,
        torch.sparse_coo,
        torch.sparse_csr,
    }, "Layout must be strided/sparse_coo/sparse_csr"
    
    # 定义不同形状的张量结构列表
    shapes = [
        [],
        [2],
        [3, 5],
        [3, 2, 1, 2],
    ]
    
    # 初始化输入列表
    inputs = []
    
    # 遍历各种形状
    for s in shapes:
        # 创建具有指定参数的张量数据
        data = make_tensor(s, device=device, dtype=dtype, requires_grad=requires_grad)  # type: ignore[arg-type]
        
        # 创建随机掩码
        mask = _create_random_mask(s, device)
        
        # 根据布局类型处理稀疏张量
        if layout == torch.sparse_coo:
            # 转换掩码为稀疏 COO 格式并压缩
            mask = mask.to_sparse_coo().coalesce()
            # 应用稀疏掩码到数据张量并设置是否需要梯度
            data = data.sparse_mask(mask).requires_grad_(requires_grad)
        elif layout == torch.sparse_csr:
            # 如果数据和掩码维度不是二维，则跳过当前循环
            if data.ndim != 2 and mask.ndim != 2:
                continue
            # 将掩码转换为稀疏 CSR 格式
            mask = mask.to_sparse_csr()
            # 应用稀疏掩码到数据张量
            data = data.sparse_mask(mask)
        
        # 将处理后的数据和掩码作为 SampleInput 对象的参数添加到输入列表
        inputs.append(SampleInput(data, kwargs={"mask": mask}))
    
    # 返回生成的输入列表
    return inputs
# 定义一个函数，用于修正函数名，去除末尾的下划线（如果有的话）
def _fix_fn_name(fn_name):
    if fn_name[-1] == "_":
        fn_name = fn_name[:-1]
    return fn_name


# 定义一个测试类 TestBasics，继承自 TestCase
class TestBasics(TestCase):
    
    # 测试无效的张量输入
    def test_invalid_tensor_inputs(self, device):
        # 创建一个随机张量数据，设备为指定的 device
        data = torch.randn((3, 4), device=device)
        # 创建一个随机掩码(mask)张量，设备为指定的 device
        mask = _create_random_mask((3, 4), device=device)
        # 创建一个 MaskedTensor 对象 mt，用于测试
        mt = masked_tensor(data, mask)

        # 使用断言验证是否抛出了 TypeError 异常，异常信息为 "data must be a Tensor"
        with self.assertRaisesRegex(TypeError, "data must be a Tensor"):
            # 传递 mt 和 mask 作为参数调用 masked_tensor 函数
            masked_tensor(mt, mask)
        with self.assertRaisesRegex(TypeError, "data must be a Tensor"):
            # 传递整数 0 和 mask 作为参数调用 masked_tensor 函数
            masked_tensor(0, mask)
        with self.assertRaisesRegex(TypeError, "mask must be a Tensor"):
            # 传递 data 和 mt 作为参数调用 masked_tensor 函数
            masked_tensor(data, mt)
        with self.assertRaisesRegex(TypeError, "mask must be a Tensor"):
            # 传递 data 和整数 0 作为参数调用 masked_tensor 函数
            masked_tensor(data, 0)

    # 测试不同的布局(layout)
    def test_diff_layouts(self, device):
        # 创建一个稀疏 COO 格式的随机张量数据，设备为指定的 device
        data = torch.randn((3, 4), device=device).to_sparse_coo()
        # 创建一个随机掩码(mask)张量，设备为指定的 device
        mask = _create_random_mask((3, 4), device=device)
        # 使用断言验证是否抛出了 TypeError 异常，异常信息为 "data and mask must have the same layout"
        with self.assertRaisesRegex(TypeError, "data and mask must have the same layout"):
            # 传递 data 和 mask 作为参数调用 masked_tensor 函数
            masked_tensor(data, mask)

    # 测试不同的维度(dim)
    def test_diff_dim(self, device):
        # 创建一个三维的随机张量数据，设备为指定的 device
        data = torch.randn((3, 4, 5), device=device)
        # 创建一个随机掩码(mask)张量，设备为指定的 device
        mask = _create_random_mask((3, 4), device=device)
        # 使用断言验证是否抛出了 ValueError 异常，异常信息为 "data.dim() must equal mask.dim()"
        with self.assertRaisesRegex(ValueError, "data.dim\\(\\) must equal mask.dim\\(\\)"):
            # 传递 data 和 mask 作为参数调用 masked_tensor 函数
            masked_tensor(data, mask)

    # 测试不同的大小(size)
    def test_diff_sizes(self, device):
        # 创建一个随机张量数据，大小为 (3, 4)，设备为指定的 device
        data = torch.randn((3, 4), device=device)
        # 创建一个大小为 (3, 3) 的随机掩码(mask)张量，设备为指定的 device
        mask = _create_random_mask((3, 3), device=device)
        # 使用断言验证是否抛出了 ValueError 异常，异常信息为 "data.size() must equal mask.size()"
        with self.assertRaisesRegex(ValueError, "data.size\\(\\) must equal mask.size\\(\\)"):
            # 传递 data 和 mask 作为参数调用 masked_tensor 函数
            masked_tensor(data, mask)

    # 测试梯度警告
    def test_grad_warning(self, device):
        # 创建一个带梯度的随机张量数据，大小为 (3, 4)，设备为指定的 device
        data = torch.randn((3, 4), device=device, requires_grad=True)
        # 创建一个随机掩码(mask)张量，设备为指定的 device
        mask = _create_random_mask((3, 4), device=device)
        # 设置警告消息
        msg = "It is not recommended to create a MaskedTensor with a tensor that requires_grad."
        # 使用断言验证是否发出了 UserWarning 警告，警告消息为 msg
        with self.assertWarnsRegex(UserWarning, msg):
            # 创建一个 MaskedTensor 对象 mt，用于测试
            mt = masked_tensor(data, mask)

    # 测试加法操作
    def test_add(self, device):
        # 创建一个设备为指定的 device 的张量数据，从 0 到 4
        data = torch.arange(5.0, device=device)
        # 创建一个设备为指定的 device 的掩码(mask)张量，包含 True, True, False, True, False
        mask = torch.tensor([True, True, False, True, False], device=device)
        # 创建一个 m0 MaskedTensor 对象，使用 data 和 mask
        m0 = masked_tensor(data, mask)
        # 创建一个 m1 MaskedTensor 对象，使用 data 和 ~mask (取反)
        m1 = masked_tensor(data, ~mask)
        # 使用断言验证是否抛出了 ValueError 异常，异常信息为 "Input masks must match."
        with self.assertRaisesRegex(ValueError, "Input masks must match."):
            # 对 m0 和 m1 进行加法操作
            m0 + m1
        # 调用 _compare_mts 函数，比较 m0 + m0 和特定的 MaskedTensor 对象
        _compare_mts(m0 + m0, masked_tensor(torch.tensor([0., 2, 0, 6, 0], device=device), mask))
   `
    # 测试 softmax 函数在给定设备上的行为
    def test_softmax(self, device):
        # 创建一个形状为 (3, 4) 的随机张量，乘以 0.1
        data = torch.randn((3, 4), device=device) * 0.1
        # 创建一个布尔掩码张量，指定哪些元素要参与 softmax 运算
        mask = torch.tensor(
            [
                [True, True, True, False],
                [False, True, False, True],
                [True, True, False, False],
            ],
            device=device
        )
        # 使用自定义函数 masked_tensor 创建一个被掩码包装的张量，设置 requires_grad 为 True
        mt = masked_tensor(data, mask, requires_grad=True)
        # 对 masked_res 应用 softmax 操作
        masked_res = torch.softmax(mt, -1)
        # 计算 masked_res 的总和并反向传播梯度
        masked_res.sum().backward()
        # 创建一个数据张量的副本，并在不符合掩码条件的位置填充负无穷，同时保留梯度信息
        xinf = data.masked_fill(~mask, float("-inf")).detach().clone().requires_grad_()
        # 对 xinf 应用 softmax 操作
        tensor_res = torch.softmax(xinf, -1)
        # 计算 tensor_res 的总和并反向传播梯度
        tensor_res.sum().backward()

        # 比较 masked_res 和 tensor_res 的结果
        _compare_mt_t(masked_res, tensor_res)
        # 比较 mt.grad 和 xinf.grad，设置允许的绝对误差为 1e-06
        _compare_mt_t(mt.grad, xinf.grad, atol=1e-06)

    # 测试 where 函数在给定设备上的行为
    def test_where(self, device):
        # 创建一个数据张量，包含一组负数
        data = torch.tensor([-10.0, -5, 0, 5, 10, 50, 60, 70, 80, 90, 100], device=device)
        # 创建一个布尔掩码，标识小于零的元素
        mask = data < 0

        # 使用 masked_tensor 函数创建一个被掩码包装的张量，设置 requires_grad 为 True
        mx = masked_tensor(data, mask, requires_grad=True)
        # 创建一个与 data 大小相同的张量，其非掩码部分为 1，并设置 requires_grad 为 True
        my = masked_tensor(torch.ones_like(data), ~mask, requires_grad=True)
        # 根据掩码应用 torch.where 函数
        masked_res = torch.where(mask, torch.exp(mx), my)
        # 计算 masked_res 的总和并反向传播梯度
        masked_res.sum().backward()

        # 创建 data 的一个副本，并保留梯度信息
        x = data.detach().clone().requires_grad_()
        # 创建一个与 x 大小相同的张量，其元素为 1，并设置 requires_grad 为 True
        y = torch.ones_like(x, device=device, requires_grad=True)
        # 根据掩码应用 torch.where 函数
        tensor_res = torch.where(mask, torch.exp(x), y)
        # 计算 tensor_res 的总和并反向传播梯度
        tensor_res.sum().backward()

        # 比较 masked_res 和 tensor_res 的结果
        _compare_mt_t(masked_res, tensor_res)
        # 比较 mx.grad 和 x.grad
        _compare_mt_t(mx.grad, x.grad)
        # 比较 my.grad 和 y.grad
        _compare_mt_t(my.grad, y.grad)

    # 测试 to_sparse 方法在给定设备上的行为
    def test_to_sparse(self, device):
        # 遍历生成的样本数据
        for sample in _generate_sample_data(device=device):
            # 获取输入数据
            data = sample.input
            # 获取掩码
            mask = sample.kwargs["mask"]
            # 使用 masked_tensor 函数创建一个被掩码包装的张量，设置 requires_grad 为 True
            mt = masked_tensor(data.clone().detach(), mask, requires_grad=True)

            # 将 mt 转换为稀疏张量
            sparse_mt = mt.to_sparse()
            # 对原始数据应用 to_sparse 后再 to_dense，计算其总和并反向传播梯度
            data.to_sparse().to_dense().sum().backward()
            # 对 sparse_mt 应用 to_dense，计算其总和并反向传播梯度
            sparse_mt.to_dense().sum().backward()

            # 比较 sparse_mt 和 data 的结果
            _compare_mt_t(sparse_mt, data)
            # 比较 mt.grad 和 data.grad
            _compare_mt_t(mt.grad, data.grad)

    # 测试 to_dense 方法在给定设备上的行为
    def test_to_dense(self, device):
        # 生成两种布局样本数据：COO 和 CSR
        samples = _generate_sample_data(
            device=device,
            layout=torch.sparse_coo
        ) + _generate_sample_data(device=device, layout=torch.sparse_csr)
        # 遍历样本数据
        for sample in samples:
            # 获取输入数据
            data = sample.input
            # 获取掩码
            mask = sample.kwargs["mask"]
            # 使用 masked_tensor 函数创建一个被掩码包装的张量，设置 requires_grad 为 True
            mt = masked_tensor(data, mask, requires_grad=True)

            # 将 data 转换为稠密张量的副本，并设置 requires_grad 为 True
            dense_data = data.to_dense().detach().clone().requires_grad_(True)
            # 将 mt 转换为稠密张量
            dense_mt = mt.to_dense()
            # 计算 dense_data 的总和并反向传播梯度
            dense_data.sum().backward()
            # 计算 dense_mt 的总和并反向传播梯度
            dense_mt.sum().backward()

            # 比较 dense_mt 和 dense_data 的结果
            _compare_mt_t(dense_mt, dense_data)
            # 比较 mt.grad.to_dense() 和 dense_data.grad
            _compare_mt_t(mt.grad.to_dense(), dense_data.grad)
    # 测试函数，用于验证稠密和稀疏 COO 格式之间的转换
    def test_to_dense_and_sparse_coo(self, device):
        # 遍历生成的样本数据
        for sample in _generate_sample_data(device=device, layout=torch.strided):
            # 获取输入数据和掩码
            data = sample.input
            mask = sample.kwargs["mask"]
            # 将掩码转换为稀疏 COO 格式，并进行合并
            ms = mask.to_sparse_coo().coalesce()

            # 创建带掩码的张量对象，要求梯度跟踪
            mt = masked_tensor(data, mask, requires_grad=True)
            # 使用稀疏掩码创建带掩码的张量对象，要求梯度跟踪
            mts = masked_tensor(data.sparse_mask(ms), ms, requires_grad=True)

            # 将稠密张量转换为稀疏张量，然后再转换回稠密张量
            converted = mt.to_sparse().to_dense()
            # 对转换后的张量求和并反向传播梯度
            converted.sum().backward()

            # 直接将稀疏带掩码的张量转换为稠密张量，并进行反向传播梯度
            converted2 = mts.to_dense()
            converted2.sum().backward()

            # 比较两个张量对象
            _compare_mts(converted, converted2)
            # 比较原始张量和带掩码张量的梯度
            _compare_mts(mt.grad, mts.grad.to_dense())

    # 测试函数，用于验证稠密和稀疏 CSR 格式之间的转换
    def test_to_dense_and_sparse_csr(self, device):
        # 遍历生成的样本数据
        for sample in _generate_sample_data(device=device, layout=torch.strided):
            # 获取输入数据和掩码
            data = sample.input
            mask = sample.kwargs["mask"]
            # 如果数据维度不是二维，则跳过当前循环
            if data.ndim != 2:
                continue
            # 将掩码转换为稀疏 CSR 格式
            ms = mask.to_sparse_csr()

            # 创建带掩码的张量对象，要求梯度跟踪
            mt = masked_tensor(data, mask, requires_grad=True)
            # 使用稀疏掩码创建带掩码的张量对象，要求梯度跟踪
            mts = masked_tensor(data.sparse_mask(ms), ms, requires_grad=True)

            # 将稠密张量转换为稀疏 CSR 格式，然后再转换回稠密张量
            converted = mt.to_sparse_csr().to_dense()
            # 对转换后的张量求和并反向传播梯度
            converted.sum().backward()

            # 直接将稀疏带掩码的张量转换为稠密张量，并进行反向传播梯度
            converted2 = mts.to_dense()
            converted2.sum().backward()

            # 比较两个张量对象
            _compare_mts(converted, converted2)
            # 比较原始张量和带掩码张量的梯度
            _compare_mts(mt.grad, mts.grad.to_dense())

    # 测试函数，用于验证无效的稀疏布局异常情况
    def test_invalid_sparse_layout(self, device):
        # 创建随机生成的稀疏 CSC 格式数据和掩码
        data = torch.randn((3, 4), device=device).to_sparse_csc()
        mask = _create_random_mask((3, 4), device=device).to_sparse_csc()
        # 断言捕获预期的类型错误异常
        with self.assertRaisesRegex(TypeError, "data layout of torch.sparse_csc is not supported"):
            # 调用函数，期望抛出异常
            masked_tensor(data, mask)

    # 测试函数，用于验证无效的稀疏 COO 值异常情况
    def test_invalid_sparse_coo_values(self, device):
        # 创建稀疏 COO 格式张量的值和索引
        v = torch.tensor([3, 4, 5], dtype=torch.float32)
        i1 = torch.tensor([[0, 1, 1], [2, 0, 2]])
        i2 = torch.tensor([[0, 1, 1], [2, 1, 2]])

        # 创建稀疏 COO 张量对象
        t = torch.sparse_coo_tensor(i1, v, (2, 4), device=device)
        # 创建稀疏 COO 张量掩码
        mask = torch.sparse_coo_tensor(i2, torch.tensor([True, True, True]), (2, 4), device=device)

        # 断言捕获预期的值错误异常
        msg = "data and mask are both sparse COO tensors but do not have the same indices."
        with self.assertRaisesRegex(ValueError, msg):
            # 调用函数，期望抛出异常
            masked_tensor(t, mask)
    def test_invalid_sparse_csr_values(self, device):
        # 定义第一个稀疏 CSR 张量的行索引
        crow_indices1 = [0, 2, 3]
        # 定义第二个稀疏 CSR 张量的行索引
        crow_indices2 = [0, 1, 3]
        # 定义列索引，用于两个稀疏 CSR 张量
        col_indices1 = [0, 1, 2]
        col_indices2 = [1, 2, 3]

        # 定义第一个稀疏 CSR 张量的数值
        values = [2, 3, 4]
        # 定义掩码值，用于第一个稀疏 CSR 张量的掩码
        mask_values = [True, True, True]

        # 创建第一个稀疏 CSR 张量 t1
        t1 = torch.sparse_csr_tensor(
            torch.tensor(crow_indices1, dtype=torch.int64),
            torch.tensor(col_indices1, dtype=torch.int64),
            torch.tensor(values),
            size=(2, 4)
        )
        # 创建第一个稀疏 CSR 张量的掩码 mask1
        mask1 = torch.sparse_csr_tensor(
            torch.tensor(crow_indices2, dtype=torch.int64),
            torch.tensor(col_indices1, dtype=torch.int64),
            torch.tensor(mask_values),
            dtype=torch.bool,
            size=(2, 4),
        )

        # 创建第二个稀疏 CSR 张量 t2
        t2 = torch.sparse_csr_tensor(
            torch.tensor(crow_indices2, dtype=torch.int64),
            torch.tensor(col_indices1, dtype=torch.int64),
            torch.tensor(values),
            size=(2, 4),
        )
        # 创建第二个稀疏 CSR 张量的掩码 mask2
        mask2 = torch.sparse_csr_tensor(
            torch.tensor(crow_indices2, dtype=torch.int64),
            torch.tensor(col_indices2, dtype=torch.int64),
            torch.tensor(mask_values),
            dtype=torch.bool,
            size=(2, 4),
        )

        # 错误消息，指示数据和掩码都是稀疏 CSR 张量，但未共享行或列索引
        msg = "data and mask are both sparse CSR tensors but do not share either crow or col indices."
        # 断言异常信息中包含预期的错误消息
        with self.assertRaisesRegex(ValueError, msg):
            # 调用被测函数 masked_tensor，预期抛出异常
            masked_tensor(t1, mask1)
        # 断言异常信息中包含预期的错误消息
        with self.assertRaisesRegex(ValueError, msg):
            # 调用被测函数 masked_tensor，预期抛出异常
            masked_tensor(t2, mask2)
    # 定义一个测试方法，用于测试连续性功能，接受一个设备参数
    def test_contiguous(self, device):
        # 创建一个在指定设备上的随机张量数据
        data = torch.randn((3, 3), device=device)

        # 克隆数据以保留原始数据
        contiguous_data = data.clone()
        # 创建一个布尔掩码，标记大于零的元素
        mask1 = (contiguous_data > 0).bool()
        # 创建一个非连续的张量数据，使用指定的尺寸和步长
        not_contiguous_data = torch.as_strided(data.clone(), (2, 2), (1, 2))
        # 创建一个布尔掩码，标记大于零的元素
        mask2 = (not_contiguous_data > 0).bool()

        # 使用标记张量类创建连续和非连续张量对象
        contiguous_mt = masked_tensor(contiguous_data, mask1)
        not_contiguous_mt = masked_tensor(not_contiguous_data, mask2)

        # 使用稀疏 COO 格式的数据创建连续和非连续标记张量对象
        contiguous_mt_sparse = masked_tensor(
            contiguous_data.to_sparse_coo(), mask1.to_sparse_coo()
        )
        not_contiguous_mt_sparse = masked_tensor(
            not_contiguous_data.to_sparse_coo(), mask2.to_sparse_coo()
        )

        # 断言连续性检查结果
        self.assertEqual(contiguous_data.is_contiguous(), True)
        self.assertEqual(not_contiguous_data.is_contiguous(), False)

        # 断言标记张量对象的连续性检查结果
        self.assertEqual(contiguous_mt.is_contiguous(), True)
        self.assertEqual(not_contiguous_mt.is_contiguous(), False)

        # 针对稀疏数据的标记张量对象进行异常断言，验证其连续性
        error_msg = "MaskedTensors with sparse data do not have is_contiguous"
        for t in [contiguous_mt_sparse, not_contiguous_mt_sparse]:
            with self.assertRaisesRegex(ValueError, error_msg):
                t.is_contiguous()
            with self.assertRaisesRegex(ValueError, error_msg):
                t.contiguous()

        # 将非连续的标记张量对象转换为连续的张量对象
        now_contiguous_mt = not_contiguous_mt.contiguous()

        # 比较非连续和现在连续的标记张量对象
        _compare_mts(not_contiguous_mt, now_contiguous_mt)

        # 断言现在连续的标记张量对象的连续性
        self.assertEqual(now_contiguous_mt.is_contiguous(), True)
        self.assertEqual(now_contiguous_mt.get_data().is_contiguous(), True)
        self.assertEqual(now_contiguous_mt.is_contiguous(), True)
class TestUnary(TestCase):
    def _get_test_data(self, fn_name):
        # 生成一个 10x10 的随机张量
        data = torch.randn(10, 10)
        # 生成一个 10x10 的随机掩码张量，元素大于0.5的为True
        mask = torch.rand(10, 10) > 0.5
        # 调整函数名以适应内部使用
        fn_name = _fix_fn_name(fn_name)
        # 根据函数名调整数据
        if fn_name in ["log", "log10", "log1p", "log2", "sqrt"]:
            data = data.mul(0.5).abs()  # 对数据进行缩放和取绝对值
        if fn_name in ["rsqrt"]:
            data = data.abs() + 1  # 避免除零错误
        if fn_name in ["acos", "arccos", "asin", "arcsin", "logit"]:
            data = data.abs().mul(0.5).clamp(0, 1)  # 对数据进行缩放并限制范围
        if fn_name in ["atanh", "arctanh", "erfinv"]:
            data = data.mul(0.5).clamp(-1, 1)  # 对数据进行缩放并限制范围
        if fn_name in ["acosh", "arccosh"]:
            data = data.abs() + 1  # 对数据进行缩放
        if fn_name in ["bitwise_not"]:
            data = data.mul(128).to(torch.int8)  # 将数据乘以128并转换为int8类型
        return data, mask

    def _get_sample_kwargs(self, fn_name):
        # 调整函数名以适应内部使用
        fn_name = _fix_fn_name(fn_name)
        kwargs = {}  # 初始化空的关键字参数字典
        if fn_name in ["clamp", "clip"]:
            kwargs["min"] = -0.5  # 设置关键字参数min为-0.5
            kwargs["max"] = 0.5  # 设置关键字参数max为0.5
        return kwargs  # 返回关键字参数字典

    def _get_sample_args(self, fn_name, data, mask):
        # 调整函数名以适应内部使用
        fn_name = _fix_fn_name(fn_name)
        # 创建一个根据掩码进行遮蔽的张量
        mt = masked_tensor(data, mask)
        t_args = [data]  # 创建张量参数列表
        mt_args = [mt]  # 创建遮蔽张量参数列表
        if fn_name in ["pow"]:
            t_args += [2.0]  # 对张量参数列表添加额外的浮点数参数
            mt_args += [2.0]  # 对遮蔽张量参数列表添加额外的浮点数参数
        return t_args, mt_args  # 返回张量参数列表和遮蔽张量参数列表

    @parametrize("fn", NATIVE_UNARY_FNS)
    def test_unary(self, fn):
        torch.random.manual_seed(0)  # 设置随机种子
        fn_name = fn.__name__  # 获取函数名
        data, mask = self._get_test_data(fn_name)  # 获取测试数据和掩码
        kwargs = self._get_sample_kwargs(fn_name)  # 获取关键字参数

        t_args, mt_args = self._get_sample_args(fn_name, data, mask)  # 获取参数列表和遮蔽参数列表

        mt_result = fn(*mt_args, **kwargs)  # 使用遮蔽参数计算结果
        t_result = fn(*t_args, **kwargs)  # 使用参数列表计算结果
        _compare_mt_t(mt_result, t_result)  # 比较遮蔽结果和普通结果

    @parametrize("fn", NATIVE_INPLACE_UNARY_FNS)
    def test_inplace_unary(self, fn):
        torch.random.manual_seed(0)  # 设置随机种子
        fn_name = fn.__name__  # 获取函数名
        data, mask = self._get_test_data(fn_name)  # 获取测试数据和掩码
        kwargs = self._get_sample_kwargs(fn_name)  # 获取关键字参数

        t_args, mt_args = self._get_sample_args(fn_name, data, mask)  # 获取参数列表和遮蔽参数列表

        mt_result = fn(*mt_args, **kwargs)  # 使用遮蔽参数计算结果
        t_result = fn(*t_args, **kwargs)  # 使用参数列表计算结果
        _compare_mt_t(mt_result, t_result)  # 比较遮蔽结果和普通结果

class TestBinary(TestCase):
    def _get_test_data(self, fn_name):
        fn_name = _fix_fn_name(fn_name)  # 调整函数名以适应内部使用
        data0 = torch.randn(10, 10)  # 生成一个 10x10 的随机张量
        data1 = torch.randn(10, 10)  # 生成一个 10x10 的随机张量
        mask = torch.rand(10, 10) > 0.5  # 生成一个 10x10 的随机掩码张量，元素大于0.5的为True
        if fn_name in ["bitwise_and", "bitwise_or", "bitwise_xor"]:
            data0 = data0.mul(128).to(torch.int8)  # 将数据0乘以128并转换为int8类型
            data1 = data1.mul(128).to(torch.int8)  # 将数据1乘以128并转换为int8类型
        if fn_name in ["bitwise_left_shift", "bitwise_right_shift"]:
            data0 = data0.abs().to(torch.int64)  # 将数据0取绝对值并转换为int64类型
            data1 = data1.abs().to(torch.int64)  # 将数据1取绝对值并转换为int64类型
        return data0, data1, mask  # 返回数据0，数据1和掩码

    def _get_sample_kwargs(self, fn_name):
        fn_name = _fix_fn_name(fn_name)  # 调整函数名以适应内部使用
        kwargs = {}  # 初始化空的关键字参数字典
        return kwargs  # 返回关键字参数字典
    def _yield_sample_args(self, fn_name, data0, data1, mask):
        """ 
        为二元函数生成两组 Tensor 和 MaskedTensor 参数来进行计算。
        Tensor 参数是相同的（即提供的两个数据张量），
        而 MaskedTensor 参数测试了 (MaskedTensor, MaskedTensor) 和 (MaskedTensor, Tensor) 两种情况。
        """
        fn_name = _fix_fn_name(fn_name)  # 调用内部函数 _fix_fn_name 处理函数名

        # 创建两个 MaskedTensor 对象，使用给定的数据和掩码
        mt0 = masked_tensor(data0, mask)
        mt1 = masked_tensor(data1, mask)

        t_args = [data0, data1]  # 创建 Tensor 参数列表
        mt_args = [mt0, mt1]  # 创建 MaskedTensor 参数列表
        yield t_args, mt_args  # 生成一组 Tensor 和 MaskedTensor 参数的迭代器

        t_args = [data0, data1]  # 再次创建 Tensor 参数列表
        mt_args = [mt0, data1]   # 创建另一组包含一个 Tensor 参数的 MaskedTensor 参数列表
        yield t_args, mt_args    # 生成另一组 Tensor 和 MaskedTensor 参数的迭代器

    @parametrize("fn", NATIVE_BINARY_FNS)
    def test_binary(self, fn):
        """
        使用 parametrize 装饰器，为 NATIVE_BINARY_FNS 中的每个函数 fn 执行测试。
        设置随机种子，并获取测试数据 data0, data1, mask 和样本关键字参数 kwargs。
        """
        torch.random.manual_seed(0)  # 设置随机种子
        fn_name = fn.__name__  # 获取函数名
        data0, data1, mask = self._get_test_data(fn_name)  # 获取测试数据
        kwargs = self._get_sample_kwargs(fn_name)  # 获取样本关键字参数

        # 遍历通过 _yield_sample_args 生成的每对参数组合 (t_args, mt_args)
        for (t_args, mt_args) in self._yield_sample_args(fn_name, data0, data1, mask):
            mt_result = fn(*mt_args, **kwargs)  # 调用 fn 函数，传入 MaskedTensor 参数
            t_result = fn(*t_args, **kwargs)    # 调用 fn 函数，传入 Tensor 参数
            _compare_mt_t(mt_result, t_result)  # 比较 MaskedTensor 和 Tensor 结果

    @parametrize("fn", NATIVE_INPLACE_BINARY_FNS)
    def test_inplace_binary(self, fn):
        """
        使用 parametrize 装饰器，为 NATIVE_INPLACE_BINARY_FNS 中的每个函数 fn 执行原地操作测试。
        设置随机种子，并获取测试数据 data0, data1, mask 和样本关键字参数 kwargs。
        """
        torch.random.manual_seed(0)  # 设置随机种子
        fn_name = fn.__name__  # 获取函数名
        data0, data1, mask = self._get_test_data(fn_name)  # 获取测试数据
        kwargs = self._get_sample_kwargs(fn_name)  # 获取样本关键字参数

        # 遍历通过 _yield_sample_args 生成的每对参数组合 (t_args, mt_args)
        for (t_args, mt_args) in self._yield_sample_args(fn_name, data0, data1, mask):
            mt_result = fn(*mt_args, **kwargs)  # 调用 fn 函数，传入 MaskedTensor 参数
            t_result = fn(*t_args, **kwargs)    # 调用 fn 函数，传入 Tensor 参数
            _compare_mt_t(mt_result, t_result)  # 比较 MaskedTensor 和 Tensor 结果

    @parametrize("fn_name", ["add", "add_"])
    def test_masks_match(self, fn_name):
        """
        使用 parametrize 装饰器，为 "add" 和 "add_" 函数名执行测试，检验输入的掩码是否匹配。
        设置随机种子，并获取测试数据 data0, data1, mask0 和样本关键字参数 kwargs。
        """
        torch.random.manual_seed(0)  # 设置随机种子
        fn = getattr(torch.ops.aten, fn_name)  # 获取对应函数名的函数对象
        data0, data1, mask = self._get_test_data(fn_name)  # 获取测试数据
        mask0 = mask  # 第一个掩码为原始掩码
        mask1 = torch.rand(mask.size()) > 0.5  # 创建一个随机掩码 mask1
        mt0 = masked_tensor(data0, mask0)  # 创建第一个 MaskedTensor 对象
        mt1 = masked_tensor(data1, mask1)  # 创建第二个 MaskedTensor 对象

        try:
            fn(mt0, mt1)  # 尝试调用函数 fn，传入两个 MaskedTensor 参数
            raise AssertionError  # 如果没有抛出异常，抛出断言错误
        except ValueError as e:
            assert (
                "Input masks must match. If you need support for this, please open an issue on Github."
                == str(e)
            )  # 检查异常消息是否为掩码不匹配的错误信息
class TestReductions(TestCase):
    # 定义测试类 TestReductions，继承自 TestCase

    def test_max_not_implemented(self):
        # 定义测试方法 test_max_not_implemented，测试 max 方法未实现异常情况
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        # 创建张量 d，包含两个子列表，数据类型为 torch.Tensor
        m = torch.tensor([[True, False, False], [False, True, False]])
        # 创建张量 m，包含两个子列表，数据类型为 torch.Tensor
        mt = masked_tensor(d, m)
        # 调用 masked_tensor 函数创建 mt，输入参数为 d, m
        with self.assertRaisesRegex(TypeError, "torch._ops.aten.max.default"):
            # 使用 self.assertRaisesRegex 检查是否引发 TypeError 异常，异常信息包含 "torch._ops.aten.max.default"
            mt.max()
            # 调用 mt 的 max 方法

    def test_sum(self):
        # 定义测试方法 test_sum，测试 sum 方法
        d = torch.tensor([[0, 1, 2, 6], [3, 4, 5.0, 7]])
        # 创建张量 d，包含两个子列表，数据类型为 torch.Tensor
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        # 创建张量 m，包含两个子列表，数据类型为 torch.Tensor
        mt = masked_tensor(d, m)
        # 调用 masked_tensor 函数创建 mt，输入参数为 d, m
        _compare_mts(masked_tensor(torch.tensor(17.0), torch.tensor(True)), mt.sum())
        # 调用 _compare_mts 函数比较结果，输入参数为 masked_tensor(torch.tensor(17.0), torch.tensor(True)) 和 mt.sum()
        _compare_mts(
            masked_tensor(
                torch.tensor([0.0, 4.0, 1.0, 13]),
                torch.tensor([True, True, False, True]),
            ),
            mt.sum(dim=0),
        )
        # 调用 _compare_mts 函数比较结果，输入参数为两个 masked_tensor 对象

    def test_sum_grad(self):
        # 定义测试方法 test_sum_grad，测试带梯度的 sum 方法
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        # 创建张量 d，包含两个子列表，数据类型为 torch.Tensor
        m = torch.tensor([[True, False, False], [False, True, False]])
        # 创建张量 m，包含两个子列表，数据类型为 torch.Tensor
        mt = masked_tensor(d, m, requires_grad=True)
        # 调用 masked_tensor 函数创建 mt，输入参数为 d, m，并指定 requires_grad=True
        mt.sum().backward()
        # 调用 mt 的 sum 方法并进行反向传播
        _compare_mts(mt.grad, masked_tensor(torch.tensor(1.0).expand_as(m), m))
        # 调用 _compare_mts 函数比较 mt 的梯度和 masked_tensor(torch.tensor(1.0).expand_as(m), m) 的结果

    def test_mean(self):
        # 定义测试方法 test_mean，测试 mean 方法
        d = torch.tensor([[0, 1, 3, 2], [3, 4, 1.0, 4]])
        # 创建张量 d，包含两个子列表，数据类型为 torch.Tensor
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        # 创建张量 m，包含两个子列表，数据类型为 torch.Tensor
        mt = masked_tensor(d, m)
        # 调用 masked_tensor 函数创建 mt，输入参数为 d, m
        _compare_mts(masked_tensor(torch.tensor(2.5), torch.tensor(True)), mt.mean())
        # 调用 _compare_mts 函数比较结果，输入参数为 masked_tensor(torch.tensor(2.5), torch.tensor(True)) 和 mt.mean()
        _compare_mts(
            masked_tensor(
                torch.tensor([0.0, 4.0, 1.0, 3]),
                torch.tensor([True, True, False, True]),
            ),
            mt.mean(dim=0),
        )
        # 调用 _compare_mts 函数比较结果，输入参数为两个 masked_tensor 对象
    """
    The following block of tests "test_mean_grad_case_1[a through e] are used to test the functionality of
    the two different ways of constructing MaskedTensors:
        masked_tensor(data, mask, requires_grad=True/False) -- NO differentiable constructor and always a leaf
        as_masked_tensor(data, mask) -- differentiable constructor
    
    Like torch.tensor(data), masked_tensor(data, mask) will provide a UserWarning if data.requires_grad=True
    as_masked_tensor does not take in requires_grad -- it just takes on the requires_grad from data
    
    Therefore, there are 6 cases to test and we use `mean` as a proxy to test the different combinations
    
    Assuming mt.mean().backward() is run after each constructor:
    
    Case 1a:
        values.requires_grad = True
        mt = masked_tensor(values, mask, requires_grad=True)
    yields
        - Provide a UserWarning because values.requires_grad=True
        - values.grad = None
        - mt.grad is a MaskedTensor with the correct gradient
    
    Case 1b:
        values.requires_grad = False
        mt = masked_tensor(values, mask, requires_grad=True)
    yields
        - values.grad = None
        - mt.grad is a MaskedTensor with the correct gradient
    
    Case 2a/2b:
        values.requires_grad = True/False
        mt = masked_tensor(values, mask, requires_grad=False)
    
        will both yield a RuntimeError of "element 0 of tensors does not require grad and does not have a grad_fn"
        as expected. When values.requires_grad=True, we will also get a UserWarning
    
    Case 3a:
        values.requires_grad = True
        mt = as_masked_tensor(values, mask)
    yields
        - values.grad is a MaskedTensor with the correct gradient
        - mt.grad is None and gives a UserWarning that
          "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad"
    
    Case 3b:
        values.requires_grad = False
        mt = as_masked_tensor(values, mask)
    
        will yield a RuntimeError of "element 0 of tensors does not require grad and does not have a grad_fn"
        as expected.
    """
    def test_mean_grad_case_1a(self):
        """ values.requires_grad = True
            mt = masked_tensor(values, mask, requires_grad=True)
        """
        # 创建一个二维张量 d，设置 requires_grad=True，表示它需要梯度计算
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]], requires_grad=True)
        # 创建一个二维张量 m，表示掩码
        m = torch.tensor([[True, False, False], [False, True, False]])
        # 使用 assertWarnsRegex 来检查是否会产生 UserWarning，警告内容包含 "It is not recommended to create a MaskedTensor"
        with self.assertWarnsRegex(UserWarning, "It is not recommended to create a MaskedTensor"):
            # 使用 masked_tensor 函数创建 MaskedTensor mt，传入数据 d、掩码 m 和 requires_grad=True
            mt = masked_tensor(d, m, requires_grad=True)
        # 对 mt 调用 mean() 函数后进行反向传播
        mt.mean().backward()
        # 断言 d 的梯度为 None，即 d.grad 为空
        self.assertIsNone(d.grad)
        # 调用 _compare_mts 函数比较 mt.grad 和使用 masked_tensor 创建的预期 MaskedTensor
        _compare_mts(mt.grad, masked_tensor(torch.tensor([[0.5, 0, 0], [0, 0.5, 0]]), m))
    def test_mean_grad_case_1b(self):
        """
        values.requires_grad = False
        mt = masked_tensor(values, mask, requires_grad=True)
        """
        # 创建输入张量 d 和掩码张量 m
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        # 创建需要梯度的 MaskedTensor 对象 mt
        mt = masked_tensor(d, m, requires_grad=True)
        # 计算 mt 的均值并进行反向传播
        mt.mean().backward()
        # 验证输入张量 d 的梯度应为 None
        self.assertIsNone(d.grad)
        # 比较 mt 的梯度与预期的 MaskedTensor 对象
        _compare_mts(mt.grad, masked_tensor(torch.tensor([[0.5, 0, 0], [0, 0.5, 0]]), m))

    def test_mean_grad_case_1c(self):
        """
        values.requires_grad = True
        mt = masked_tensor(values, mask, requires_grad=False)
        """
        # 创建输入张量 d，并标记需要梯度
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]], requires_grad=True)
        m = torch.tensor([[True, False, False], [False, True, False]])
        # 使用断言检测警告信息
        with self.assertWarnsRegex(UserWarning, "It is not recommended to create a MaskedTensor"):
            # 创建不需要梯度的 MaskedTensor 对象 mt
            mt = masked_tensor(d, m, requires_grad=False)
        # 对 mt 的均值进行计算，预期引发 RuntimeError
        result = mt.mean()
        msg = "element 0 of tensors does not require grad and does not have a grad_fn"
        with self.assertRaisesRegex(RuntimeError, msg):
            result.backward()

    def test_mean_grad_case_1d(self):
        """
        values.requires_grad = False
        mt = masked_tensor(values, mask, requires_grad=False)
        """
        # 创建输入张量 d 和掩码张量 m
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        # 创建不需要梯度的 MaskedTensor 对象 mt
        mt = masked_tensor(d, m, requires_grad=False)
        # 对 mt 的均值进行计算，预期引发 RuntimeError
        result = mt.mean()
        msg = "element 0 of tensors does not require grad and does not have a grad_fn"
        with self.assertRaisesRegex(RuntimeError, msg):
            result.backward()

    def test_mean_grad_case_1e(self):
        """
        values.requires_grad = True
        mt = as_masked_tensor(values, mask)
        """
        # 创建输入张量 d，并标记需要梯度
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]], requires_grad=True)
        m = torch.tensor([[True, False, False], [False, True, False]])
        # 创建不需要梯度的 MaskedTensor 对象 mt
        mt = as_masked_tensor(d, m)
        # 计算 mt 的均值并进行反向传播
        mt.mean().backward()
        # 比较 d 的梯度与预期的 MaskedTensor 对象
        _compare_mts(d.grad, masked_tensor(torch.tensor([[0.5, 0, 0], [0, 0.5, 0]]), m))
        # 使用断言检测警告信息
        msg = "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad"
        with self.assertWarnsRegex(UserWarning, msg):
            self.assertIsNone(mt.grad)

    def test_mean_grad_case_1f(self):
        """
        values.requires_grad = False
        mt = as_masked_tensor(values, mask)
        """
        # 创建输入张量 d 和掩码张量 m
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        # 创建不需要梯度的 MaskedTensor 对象 mt
        mt = as_masked_tensor(d, m)
        # 对 mt 的均值进行计算，预期引发 RuntimeError
        result = mt.mean()
        msg = "element 0 of tensors does not require grad and does not have a grad_fn"
        with self.assertRaisesRegex(RuntimeError, msg):
            result.backward()
    # 定义一个测试方法，用于测试带有梯度的 masked_tensor 对象的 mean(dim) 方法的梯度计算
    def test_mean_dim_grad(self):
        # 创建一个包含浮点数和整数的 PyTorch 张量
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        # 创建一个布尔类型的 PyTorch 张量，用于标记哪些元素要参与计算
        m = torch.tensor([[True, True, False], [False, True, False]])
        # 创建一个带有掩码的 masked_tensor 对象，并要求计算梯度
        mt = masked_tensor(d, m, requires_grad=True)
        # 对 masked_tensor 对象调用 mean(dim) 方法后，计算其结果的和，然后进行反向传播
        mt.mean(1).sum().backward()
        # 比较 masked_tensor 对象的梯度与给定的参考值的 masked_tensor 对象
        _compare_mts(mt.grad, masked_tensor(torch.tensor([[0.5, 0.5, 0], [0, 1, 0]]), m))

    # 定义一个测试方法，用于测试 masked_tensor 对象的 amax 方法
    def test_amax(self):
        # 创建一个包含浮点数和整数的 PyTorch 张量
        d = torch.tensor([[0, 1, 3, -3], [3, -4, 1.0, 3]])
        # 创建一个布尔类型的 PyTorch 张量，用于标记哪些元素要参与计算
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        # 创建一个带有掩码的 masked_tensor 对象
        mt = masked_tensor(d, m)
        # 比较 masked_tensor 对象的 amax 方法计算的结果与给定的参考值的 masked_tensor 对象
        _compare_mts(masked_tensor(torch.tensor(3.0), torch.tensor(True)), mt.amax())
        # 比较 masked_tensor 对象的 amax(dim) 方法计算的结果与给定的参考值的 masked_tensor 对象
        _compare_mts(
            masked_tensor(
                torch.tensor([0.0, -4.0, 1.0, 3]),
                torch.tensor([True, True, False, True]),
            ),
            mt.amax(dim=0),
        )

    # 定义一个测试方法，用于测试带有梯度的 masked_tensor 对象的 amax 方法的梯度计算
    def test_amax_grad(self):
        # 创建一个包含浮点数和整数的 PyTorch 张量
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        # 创建一个布尔类型的 PyTorch 张量，用于标记哪些元素要参与计算
        m = torch.tensor([[True, False, False], [False, True, False]])
        # 创建一个带有掩码的 masked_tensor 对象，并要求计算梯度
        mt = masked_tensor(d, m, requires_grad=True)
        # 对 masked_tensor 对象调用 amax 方法后，计算其结果，然后进行反向传播
        mt.amax().backward()
        # 比较 masked_tensor 对象的梯度与给定的参考值的 masked_tensor 对象
        _compare_mts(mt.grad, masked_tensor(torch.tensor([[0.0, 0, 0], [0, 1, 0]]), m))

    # 定义一个测试方法，用于测试 masked_tensor 对象的 amin 方法
    def test_amin(self):
        # 创建一个包含浮点数和整数的 PyTorch 张量
        d = torch.tensor([[0, 1, 3, -3], [3, -4, 1.0, 3]])
        # 创建一个布尔类型的 PyTorch 张量，用于标记哪些元素要参与计算
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        # 创建一个带有掩码的 masked_tensor 对象
        mt = masked_tensor(d, m)
        # 比较 masked_tensor 对象的 amin 方法计算的结果与给定的参考值的 masked_tensor 对象
        _compare_mts(masked_tensor(torch.tensor(-4.0), torch.tensor(True)), mt.amin())
        # 比较 masked_tensor 对象的 amin(dim) 方法计算的结果与给定的参考值的 masked_tensor 对象
        _compare_mts(
            masked_tensor(
                torch.tensor([0.0, -4.0, 1.0, -3]),
                torch.tensor([True, True, False, True]),
            ),
            mt.amin(dim=0),
        )

    # 定义一个测试方法，用于测试带有梯度的 masked_tensor 对象的 amin 方法的梯度计算
    def test_amin_grad(self):
        # 创建一个包含浮点数和整数的 PyTorch 张量
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        # 创建一个布尔类型的 PyTorch 张量，用于标记哪些元素要参与计算
        m = torch.tensor([[True, False, False], [False, True, False]])
        # 创建一个带有掩码的 masked_tensor 对象，并要求计算梯度
        mt = masked_tensor(d, m, requires_grad=True)
        # 对 masked_tensor 对象调用 amin 方法后，计算其结果，然后进行反向传播
        mt.amin().backward()
        # 比较 masked_tensor 对象的梯度与给定的参考值的 masked_tensor 对象
        _compare_mts(mt.grad, masked_tensor(torch.tensor([[1.0, 0, 0], [0, 0, 0]]), m))

    # 定义一个测试方法，用于测试 masked_tensor 对象的 prod 方法
    def test_prod(self):
        # 创建一个包含浮点数和整数的 PyTorch 张量，其中包含 NaN
        d = torch.tensor([[0, 1, 3, 0.0], [float("nan"), 4, 1.0, 5.0]])
        # 创建一个布尔类型的 PyTorch 张量，用于标记哪些元素要参与计算
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        # 创建一个带有掩码的 masked_tensor 对象
        mt = masked_tensor(d, m)
        # 比较 masked_tensor 对象的 prod 方法计算的结果与给定的参考值的 masked_tensor 对象
        _compare_mts(masked_tensor(torch.tensor(0.0), torch.tensor(True)), mt.prod())
        # 比较 masked_tensor 对象的 prod(dim) 方法计算的结果与给定的参考值的 masked_tensor 对象
        _compare_mts(
            masked_tensor(
                torch.tensor([0.0, 4.0, 1.0, 0.0]),
                torch.tensor([True, True, False, True]),
            ),
            mt.prod(dim=0),
        )

    # 定义一个测试方法，用于测试带有梯度的 masked_tensor 对象的 prod 方法的梯度计算
    def test_prod_grad(self):
        # 创建一个包含浮点数和整数的 PyTorch 张量，其中包含 NaN
        d = torch.tensor([[2, float("nan"), 2], [3, 4, 5.0]])
        # 创建一个布尔类型的 PyTorch 张量，用于标记哪些元素要参与计算
        m = torch.tensor([[True, False, False], [False, True, False]])
        # 创建一个带有掩码的 masked_tensor 对象，并要求计算梯度
        mt = masked_tensor(d, m, requires_grad=True)
        # 对 masked_tensor 对象调用 prod 方法后，计算其结果，然后进行反向传播
        mt.prod().backward()
        # 比较 masked_tensor 对象的梯度与给定的参考值的 masked_tensor 对象
        _compare_mts(mt.grad, masked_tensor(torch.tensor([[4.0, 0, 0], [0, 2, 0]]), m))
    # 定义测试方法，用于测试 masked_tensor 函数的不同输入情况
    def test_all(self):
        # 创建一个二维张量 d，包含布尔值 True 和 False
        d = torch.tensor([[True, True, False, False], [False, True, True, True]])
        # 创建一个二维张量 m，包含布尔值 True 和 False
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        # 使用 masked_tensor 函数创建一个被掩码的张量 mt，传入 d 和 m
        mt = masked_tensor(d, m)
        # 调用 _compare_mts 函数，比较两个 masked_tensor 对象
        _compare_mts(masked_tensor(torch.tensor(False), torch.tensor(True)), mt.all())
        # 调用 _compare_mts 函数，比较两个 masked_tensor 对象
        _compare_mts(
            # 使用 masked_tensor 函数创建一个被掩码的张量，传入两个张量作为参数
            masked_tensor(
                torch.tensor([True, True, True, False]),
                torch.tensor([True, True, False, True]),
            ),
            # 调用 mt.all(dim=0)，对 mt 进行按列维度的 all 操作
            mt.all(dim=0),
        )

        # 重新定义 m 为一个新的二维张量，包含布尔值 True 和 False
        m = torch.tensor([[True, False, True, False], [False, True, False, False]])
        # 使用 masked_tensor 函数创建一个被掩码的张量 mt，传入 d 和新的 m
        mt = masked_tensor(d, m)
        # 调用 _compare_mts 函数，比较两个 masked_tensor 对象
        _compare_mts(
            # 使用 masked_tensor 函数创建一个被掩码的张量，传入两个张量作为参数
            masked_tensor(
                torch.tensor([True, True, False, True]),
                torch.tensor([True, True, True, False]),
            ),
            # 调用 mt.all(dim=0)，对 mt 进行按列维度的 all 操作
            mt.all(dim=0),
        )

    # 定义测试梯度和数据类型的方法
    def test_grad_dtype(self):
        # 创建一个二维张量 d，包含布尔值 True 和 False
        d = torch.tensor([[True, True, False], [False, True, True]])
        # 创建一个二维张量 m，包含布尔值 True 和 False
        m = torch.tensor([[True, False, False], [False, True, False]])
        # 定义错误消息字符串，用于验证是否抛出期望的 RuntimeError
        msg = "Only Tensors of floating point and complex dtype can require gradients"
        # 使用 self.assertRaisesRegex 检查是否抛出期望的 RuntimeError，并验证错误消息
        with self.assertRaisesRegex(RuntimeError, msg):
            # 调用 masked_tensor 函数，传入 d、m 和 requires_grad=True 参数
            masked_tensor(d, m, requires_grad=True)
# 检查给定操作是否为一元操作
def is_unary(op):
    return op.name in UNARY_NAMES

# 检查给定操作是否为二元操作
def is_binary(op):
    return op.name in BINARY_NAMES

# 检查给定操作是否为约简操作，并排除部分例外
def is_reduction(op):
    return op.name in REDUCE_NAMES and op.name not in {"all", "mean", "std", "var"}

# 从一元操作列表中筛选出满足 is_unary 函数条件的操作列表
mt_unary_ufuncs = [op for op in unary_ufuncs if is_unary(op)]

# 从二元操作列表中筛选出满足 is_binary 函数条件的操作列表
mt_binary_ufuncs = [op for op in binary_ufuncs if is_binary(op)]

# 从约简操作列表中筛选出满足 is_reduction 函数条件的操作列表
mt_reduction_ufuncs = [op for op in reduction_ops if is_reduction(op)]

# 定义支持的浮点类型集合
MASKEDTENSOR_FLOAT_TYPES = {
    torch.float16,
    torch.float32,
    torch.float64,
}

# 测试用例类，用于测试操作符功能
class TestOperators(TestCase):
    # 将参数转换为 MaskedTensor 类型的参数列表
    def _convert_mt_args(self, args, mask, layout):
        return [
            masked_tensor(
                arg.sparse_mask(mask) if layout != torch.strided else arg, mask
            )
            if torch.is_tensor(arg)
            else arg
            for arg in args
        ]

    # 测试一元和二元操作的相等性
    def _test_unary_binary_equality(self, device, dtype, op, layout=torch.strided):
        # 获取操作的输入样本
        samples = op.sample_inputs(device, dtype, requires_grad=True)

        for sample in samples:
            input = sample.input
            sample_args, sample_kwargs = sample.args, sample.kwargs
            # 获取掩码，如果 sample_kwargs 中有 "mask" 则使用它，否则创建随机掩码
            mask = (
                _create_random_mask(input.shape, device)
                if "mask" not in sample_kwargs
                else sample_kwargs.pop("mask")
            )

            # 如果布局为稀疏 COO，则将掩码转换为稀疏 COO 格式并重新组织输入
            if layout == torch.sparse_coo:
                mask = mask.to_sparse_coo().coalesce()
                input = input.sparse_mask(mask)
            # 如果布局为稀疏 CSR，则将输入和掩码转换为稀疏 CSR 格式并重新组织输入
            elif layout == torch.sparse_csr:
                if input.ndim != 2 or mask.ndim != 2:
                    continue
                mask = mask.to_sparse_csr()
                input = input.sparse_mask(mask)

            # 对于二元操作，当前仅支持相同大小的掩码
            if is_binary(op):
                if input.shape != sample_args[0].shape:
                    continue
                else:
                    # 目前二元操作不支持 kwargs
                    sample_kwargs = {}

            # 创建 MaskedTensor 对象
            mt = masked_tensor(input, mask)
            # 将样本参数转换为 MaskedTensor 类型的参数列表
            mt_args = self._convert_mt_args(sample_args, mask, layout)

            # 对 MaskedTensor 和普通张量进行操作
            mt_result = op(mt, *mt_args, **sample_kwargs)
            t_result = op(sample.input, *sample_args, **sample_kwargs)

            # 比较 MaskedTensor 结果和普通张量结果
            _compare_mt_t(mt_result, t_result)

            # 如果操作是二元操作且布局为 torch.strided，则检查 lhs=masked、rhs=regular tensor 也能正常工作
            if is_binary(op) and layout == torch.strided:
                mt_result2 = op(mt, *sample_args, **sample_kwargs)
                _compare_mt_t(mt_result2, t_result)
    # 定义一个测试方法，用于测试降维操作的正确性
    def _test_reduction_equality(self, device, dtype, op, layout=torch.strided):
        # 从操作对象中获取样本输入，要求梯度为True
        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # 遍历每个样本
        for sample in samples:
            # 获取样本的输入数据
            input = sample.input
            # 提示：当前不支持更高级的参数和关键字参数
            sample_args, sample_kwargs = (), {}

            # 如果输入数据维度为0或元素个数为0，则跳过当前循环
            if input.dim() == 0 or input.numel() == 0:
                continue

            # 创建一个随机掩码
            mask = _create_random_mask(input.shape, device)

            # 如果掩码中非零元素个数为0，则跳过当前循环
            if torch.count_nonzero(mask) == 0:
                continue

            # 结合输入数据和掩码，形成张量输入
            tensor_input = _combine_input_and_mask(op.op, input, mask)

            # 根据布局类型进行特定处理
            if layout == torch.sparse_coo:
                # 将掩码转换为稀疏 COO 格式并合并
                mask = mask.to_sparse_coo().coalesce()
                input = input.sparse_mask(mask)
            elif layout == torch.sparse_csr:
                # 如果输入数据或掩码维度不为2，则跳过当前循环
                if input.ndim != 2 or mask.ndim != 2:
                    continue
                # 将掩码转换为稀疏 CSR 格式
                mask = mask.to_sparse_csr()
                input = input.sparse_mask(mask)

            # 创建 MaskedTensor 对象
            mt = masked_tensor(input, mask)
            # 转换 MaskedTensor 的参数
            mt_args = self._convert_mt_args(sample_args, mask, layout)

            # 使用操作对象对 MaskedTensor 进行操作，并记录结果
            mt_result = op(mt, *mt_args, **sample_kwargs)
            # 对比使用普通张量进行操作的结果
            t_result = op(tensor_input, *sample_args, **sample_kwargs)

            # 比较 MaskedTensor 和普通张量操作的结果
            _compare_mt_t(mt_result, t_result)

    # 使用装饰器定义一元核心测试方法
    @ops(mt_unary_ufuncs, allowed_dtypes=MASKEDTENSOR_FLOAT_TYPES)  # type: ignore[arg-type]
    @parametrize("layout", [torch.strided, torch.sparse_coo, torch.sparse_csr])
    def test_unary_core(self, device, dtype, op, layout):
        # 跳过某些不符合条件的测试变种
        skip_variants = {
            "decimals_0",
            "decimals_3",
            "decimals_neg_3",
        }
        # 如果操作是 round 且其测试变种在跳过列表中，则直接返回
        if op.name == "round" and op.variant_test_name in skip_variants:
            return
        # 执行一元测试方法
        self._test_unary_binary_equality(device, dtype, op)

    # 使用装饰器定义二元核心测试方法
    @ops(mt_binary_ufuncs, allowed_dtypes=MASKEDTENSOR_FLOAT_TYPES)  # type: ignore[arg-type]
    @parametrize("layout", [torch.strided, torch.sparse_coo, torch.sparse_csr])
    def test_binary_core(self, device, dtype, op, layout):
        # 执行二元测试方法
        self._test_unary_binary_equality(device, dtype, op, layout)

    # 使用装饰器定义全部降维测试方法
    @ops(mt_reduction_ufuncs, allowed_dtypes=MASKEDTENSOR_FLOAT_TYPES)  # type: ignore[arg-type]
    @parametrize("layout", [torch.strided, torch.sparse_coo, torch.sparse_csr])
    def test_reduction_all(self, device, dtype, op, layout):
        # 当操作是 argmin 或 argmax 且布局是稀疏 CSR 时，不支持当前操作，直接返回
        if op.name in {"argmin", "argmax"} and layout == torch.sparse_csr:
            return
        # 执行全部降维测试方法
        self._test_reduction_equality(device, dtype, op, layout)
# 定义一个包含字符串 "cpu" 和 "cuda" 的元组，限定仅适用于这两种设备类型
only_for = ("cpu", "cuda")

# 根据给定的测试类 TestOperators，以及当前全局命名空间和设备类型的限制，实例化设备类型相关的测试
instantiate_device_type_tests(TestOperators, globals(), only_for=only_for)

# 根据给定的测试类 TestBasics，以及当前全局命名空间和设备类型的限制，实例化设备类型相关的测试
instantiate_device_type_tests(TestBasics, globals(), only_for=only_for)

# 实例化参数化测试，使用 TestUnary 类
instantiate_parametrized_tests(TestUnary)

# 实例化参数化测试，使用 TestBinary 类
instantiate_parametrized_tests(TestBinary)

# 实例化参数化测试，使用 TestReductions 类
instantiate_parametrized_tests(TestReductions)

# 如果当前脚本作为主程序执行，则运行所有测试
if __name__ == '__main__':
    run_tests()
```