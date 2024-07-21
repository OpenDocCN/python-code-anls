# `.\pytorch\test\quantization\core\test_quantized_tensor.py`

```
# Owner(s): ["oncall: quantization"]

import numpy as np  # 导入 NumPy 库，用于数值计算
import math  # 导入数学函数库
import random  # 导入随机数生成模块
import torch  # 导入 PyTorch 深度学习库
import io  # 导入用于处理字节流的模块
import unittest  # 导入单元测试框架模块
from copy import deepcopy  # 导入深度复制函数
from hypothesis import given  # 导入用于属性测试的假设模块
from hypothesis import strategies as st  # 导入假设策略模块
from torch.testing._internal.common_utils import TemporaryFileName  # 导入临时文件名生成工具
from torch.testing._internal.common_cuda import TEST_CUDA  # 导入测试 CUDA 是否可用的变量
from torch.testing._internal.common_utils import TestCase, DeterministicGuard  # 导入测试用例和确定性保护工具
import torch.testing._internal.hypothesis_utils as hu  # 导入假设测试工具
from torch.testing._internal.common_quantization import get_supported_device_types  # 导入获取支持的设备类型函数

hu.assert_deadline_disabled()  # 禁用假设测试的执行时间限制

import itertools  # 导入迭代工具模块
import tempfile  # 导入临时文件处理模块

class Foo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qscheme = torch.per_tensor_symmetric  # 初始化量化方案为每张量对称量化

def _calculate_dynamic_qparams(X, dtype, reduce_range=False):
    """Calculate the dynamic quantization parameters (scale, zero_point)
    according to the min and max element of the tensor"""
    if isinstance(X, torch.Tensor):
        X = X.cpu().data.numpy()  # 如果输入是 PyTorch 张量，则转换为 NumPy 数组
    if dtype == torch.qint8:
        if reduce_range:
            qmin, qmax = -64, 63  # 如果是 qint8 类型并且需要减少范围，则设置量化范围
        else:
            qmin, qmax = -128, 127  # 否则设置默认 qint8 量化范围
    else:  # dtype == torch.quint8
        if reduce_range:
            qmin, qmax = 0, 127  # 如果是 quint8 类型并且需要减少范围，则设置量化范围
        else:
            qmin, qmax = 0, 255  # 否则设置默认 quint8 量化范围

    min_val = X.min().astype(dtype=np.float32)  # 计算输入张量的最小值，并转换为 float32 类型
    max_val = X.max().astype(dtype=np.float32)  # 计算输入张量的最大值，并转换为 float32 类型
    min_val = min(0.0, min_val)  # 将最小值限制为 0.0 或更小
    max_val = max(0.0, max_val)  # 将最大值限制为 0.0 或更大
    scale = (np.float64(max_val) - min_val) / (qmax - qmin)  # 计算量化比例因子
    if scale == 0.0 or math.isinf(1.0 / scale):
        scale = np.float64(0.1)  # 处理异常情况，如果 scale 为零或无穷大，则设置为默认值
        zero_point = 0

    zero_point_from_min = qmin - min_val / float(scale)  # 从最小值计算初始零点
    zero_point_from_max = qmax - max_val / float(scale)  # 从最大值计算初始零点
    zero_point_from_min_error = abs(qmin) - abs(min_val / float(scale))
    zero_point_from_max_error = abs(qmax) - abs(max_val / float(scale))
    if zero_point_from_min_error < zero_point_from_max_error:
        initial_zero_point = zero_point_from_min  # 根据误差大小确定初始零点
    else:
        initial_zero_point = zero_point_from_max

    nudged_zero_point = 0

    if initial_zero_point < qmin:
        nudged_zero_point = qmin  # 如果初始零点小于最小量化值，则设置为最小量化值
    elif initial_zero_point > qmax:
        nudged_zero_point = qmax  # 如果初始零点大于最大量化值，则设置为最大量化值
    else:
        nudged_zero_point = int(round(initial_zero_point))  # 否则将初始零点四舍五入为整数

    return [scale.astype(np.float32), int(nudged_zero_point)]  # 返回量化比例因子和零点值

# Note we explicitly cast variables to np.float32 in a couple of places to avoid
# the default casting in Python often resulting in double precision and to make
# sure we're doing the same numerics as C++ code.
def param_search_greedy(x, bit_rate, n_bins=200, ratio=0.16):
    xmin, xmax = np.min(x), np.max(x)  # 计算输入数组的最小和最大值
    stepsize = (xmax - xmin) / np.float32(n_bins)  # 计算步长
    min_bins = np.float32(n_bins) * (np.float32(1) - np.float32(ratio))  # 计算最小 bin 数量
    xq, loss = _compress_uniform_simplified(x, bit_rate, xmin, xmax)  # 压缩输入数组并计算损失值

    solutions = []  # 存储局部最优解的列表，格式为 [(left, right, loss)]

    cur_min, cur_max, cur_loss = xmin, xmax, loss  # 初始化当前最小、最大值和损失值
    thr = min_bins * stepsize  # 计算阈值
    # 当前最小值加上阈值小于当前最大值时进行循环
    while cur_min + thr < cur_max:
        # 向左移动
        xq, loss1 = _compress_uniform_simplified(
            x, bit_rate, cur_min + stepsize, cur_max
        )
        # 向右移动
        xq, loss2 = _compress_uniform_simplified(
            x, bit_rate, cur_min, cur_max - stepsize
        )

        # 如果当前损失小于 loss1 和 loss2，则找到一个局部最优解
        if cur_loss < loss1 and cur_loss < loss2:
            solutions.append((cur_min, cur_max, cur_loss))
        
        # 如果 loss1 小于 loss2，则更新当前最小值、最大值和当前损失为 loss1
        if loss1 < loss2:
            cur_min, cur_max, cur_loss = cur_min + stepsize, cur_max, loss1
        else:
            # 否则更新当前最小值、最大值和当前损失为 loss2
            cur_min, cur_max, cur_loss = cur_min, cur_max - stepsize, loss2
    
    # 如果找到了解决方案，则从中选择最优解
    if len(solutions):
        best = solutions[0]
        for solution in solutions:
            if solution[-1] < best[-1]:
                best = solution
        return best[1], best[0]  # 返回最佳的 xmax, xmin
    # 如果未找到解决方案，则返回原始的 xmax, xmin
    return xmax, xmin
# 定义一个函数用于对输入数据 X 进行简化的均匀量化压缩
def _compress_uniform_simplified(X, bit_rate, xmin, xmax, fp16_scale_bias=True):
    # 如果开启了 fp16_scale_bias 参数，将 xmin 转换为 np.float16 类型再转为 np.float32 类型
    if fp16_scale_bias:
        xmin = xmin.astype(np.float16).astype(np.float32)
    
    # 计算数据范围
    data_range = xmax - xmin
    
    # 计算缩放比例，处理数据范围为 0 的情况
    scale = np.where(
        data_range == 0, np.float32(1), data_range / np.float32(2 ** bit_rate - 1)
    )
    
    # 如果开启了 fp16_scale_bias 参数，将 scale 缩放比例转换为 np.float16 类型再转为 np.float32 类型
    if fp16_scale_bias:
        scale = scale.astype(np.float16).astype(np.float32)
    
    # 计算反向缩放比例
    inverse_scale = np.float32(1) / scale
    
    # 对 X 进行量化压缩，并根据缩放比例和 xmin 进行反量化操作
    Xq = np.clip(np.round((X - xmin) * inverse_scale), 0, np.float32(2 ** bit_rate - 1))
    Xq = Xq * scale + xmin
    
    # 手动计算损失，而不使用 np.linalg.norm，以保持与 C++ 代码相同的累积顺序
    vlen = 8  # 定义向量长度
    loss_v = np.zeros(vlen).astype(np.float32)  # 初始化损失向量
    for i in range(len(Xq) // vlen * vlen):
        # 计算每个向量元素的损失并累加到 loss_v 中
        loss_v[i % vlen] += (X[i] - Xq[i]) * (X[i] - Xq[i])
    
    # 计算总损失
    loss = np.float32(0)
    for i in range(vlen):
        loss += loss_v[i]
    for i in range(len(Xq) // vlen * vlen, len(Xq)):
        loss += (X[i] - Xq[i]) * (X[i] - Xq[i])
    
    # 对总损失进行平方根处理
    loss = np.sqrt(loss)
    
    # 返回压缩后的量化数据 Xq 和损失值 loss
    return Xq, loss


class TestQuantizedTensor(TestCase):
    def test_qtensor_equal(self):
        # ASAN 回归测试，详情见 https://github.com/pytorch/pytorch/issues/116087
        x = torch.rand(5)  # 创建一个包含 5 个随机数的张量 x
        x_q = torch.quantize_per_tensor(x, 0.1, 10, torch.quint4x2)  # 对张量 x 进行量化处理
        y_q = torch.quantize_per_tensor(x, 0.1, 10, torch.quint4x2)  # 对张量 x 进行相同的量化处理
        self.assertTrue(torch.equal(x_q, y_q))  # 断言 x_q 和 y_q 是否相等
    def test_per_tensor_qtensor_to_memory_format(self):
        # 随机生成张量的维度大小
        n = np.random.randint(1, 10)
        c = np.random.randint(2, 10)
        h = np.random.randint(2, 10)
        w = np.random.randint(2, 10)
        # 创建一个随机数值的张量
        x = torch.rand(n, c, h, w)
        # 随机生成量化参数
        scale = np.random.uniform(0.1, 1.0)
        zero_point = np.random.randint(0.0, 10)
        # 选择量化类型
        qints = [torch.qint8, torch.quint8, torch.qint32]
        dtype = qints[np.random.randint(0, len(qints))]
        # 对张量进行量化
        qx = torch.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=dtype)
        # 将张量格式转换为通道在最后的格式
        x_nhwc = x.to(memory_format=torch.channels_last)
        # 使用to方法将量化张量转换为通道在最后的格式
        qx_nhwc_using_to = qx.to(memory_format=torch.channels_last)
        # 使用contiguous方法将量化张量转换为通道在最后的格式
        qx_nhwc_using_contiguous = qx.contiguous(memory_format=torch.channels_last)
        # 断言两种方式的内存布局是否相同
        self.assertEqual(qx_nhwc_using_to.stride(), qx_nhwc_using_contiguous.stride())
        # 断言量化张量和非量化张量在通道在最后格式下的内存布局是否相同

        # 当4D张量的最后两个维度都是1，或者c等于1时，会出现退化情况
        # 参考：https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html
        # 在这种情况下，torch.Tensor.to和torch.Tensor.contiguous的输出应该不相同
        x = torch.rand(10, 2, 1, 1)
        qx = torch.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=dtype)
        qx_nhwc_using_to = qx.to(memory_format=torch.channels_last)
        qx_nhwc_using_contiguous = qx.contiguous(memory_format=torch.channels_last)
        self.assertNotEqual(qx_nhwc_using_to.stride(), qx_nhwc_using_contiguous.stride())

        x = torch.rand(10, 1, 2, 2)
        qx = torch.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=dtype)
        qx_nhwc_using_to = qx.to(memory_format=torch.channels_last)
        qx_nhwc_using_contiguous = qx.contiguous(memory_format=torch.channels_last)
        self.assertNotEqual(qx_nhwc_using_to.stride(), qx_nhwc_using_contiguous.stride())

    def test_per_channel_qtensor_to_memory_format(self):
        # 随机生成张量的维度大小
        n = np.random.randint(1, 10)
        c = np.random.randint(2, 10)
        h = np.random.randint(2, 10)
        w = np.random.randint(2, 10)
        # 创建一个随机数值的张量
        x = torch.rand(n, c, h, w)
        # 将张量格式转换为通道在最后的格式
        x_nhwc = x.to(memory_format=torch.channels_last)
        # 随机生成量化参数
        scale = np.random.uniform(0.1, 1.0)
        zero_point = np.random.randint(0.0, 10)
        # 选择量化类型
        qints = [torch.qint8, torch.quint8, torch.qint32]
        dtype = qints[np.random.randint(0, len(qints))]
        # 对每一个维度轴进行遍历
        for axis in range(x.ndim):
            # 随机生成每个维度轴的量化参数
            scales = torch.rand(x.size(axis)) + 0.00001
            zero_points = torch.randint(low=0, high=10, size=(x.size(axis), ))
            # 对张量进行通道量化
            qx = torch.quantize_per_channel(x, scales=scales, zero_points=zero_points, dtype=dtype, axis=axis)
            # 使用to方法将量化张量转换为通道在最后的格式
            qx_nhwc_using_to = qx.to(memory_format=torch.channels_last)
            # 断言量化张量在通道在最后的格式下的内存布局是否与非量化张量相同
            self.assertEqual(qx_nhwc_using_to.stride(), x_nhwc.stride())

    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    # 定义一个测试方法，用于在 CUDA 设备上测试量化张量的相关功能
    def test_qtensor_cuda(self):
        # 调用 _test_qtensor 方法，传入 CUDA 设备作为参数
        self._test_qtensor(torch.device('cuda'))
        # 调用 _test_qtensor_dynamic 方法，传入 CUDA 设备作为参数
        self._test_qtensor_dynamic(torch.device('cuda'))

    # 定义一个测试方法，用于在 CPU 设备上测试量化张量的相关功能
    def test_qtensor_cpu(self):
        # 调用 _test_qtensor 方法，传入 CPU 设备作为参数
        self._test_qtensor(torch.device('cpu'))
        # 调用 _test_qtensor_dynamic 方法，传入 CPU 设备作为参数
        self._test_qtensor_dynamic(torch.device('cpu'))

    # 定义一个私有方法，用于动态测试量化张量的功能，接受一个设备参数
    def _test_qtensor_dynamic(self, device):
        # 设置张量的最大维度数
        max_tensor_order = 4
        # 设置每个张量维度的最大尺寸
        max_dim_sz = 20

        # 随机生成张量的维度数量
        num_dim = np.random.randint(low=1, high=max_tensor_order)
        # 随机生成张量的各个维度尺寸
        dims = np.random.randint(low=1, high=max_dim_sz, size=num_dim)
        # 生成一个随机张量，指定数据类型和设备
        mat2quant = torch.randn(*dims, dtype=torch.float, device=device)
        # 是否进行减少标志位
        reduce_flag = False

        # 遍历两种量化数据类型
        for dtype in [torch.qint8, torch.quint8]:
            # 使用动态量化方法对张量进行量化
            q_d = torch.quantize_per_tensor_dynamic(mat2quant, dtype, reduce_flag)
            # 计算动态量化参数（缩放因子和零点）
            scale, zero_pt = _calculate_dynamic_qparams(mat2quant, dtype, reduce_flag)
            # 使用给定的缩放因子和零点对张量进行量化
            q_s = torch.quantize_per_tensor(mat2quant, scale, zero_pt, dtype)

            # 断言两种量化方法得到的结果应该相等
            self.assertEqual(q_d, q_s)

    # 定义一个私有方法，用于测试量化张量的功能，接受一个设备参数
    def _test_qtensor(self, device):
        # 将设备参数转换为字符串格式
        device = str(device)
        # 设置张量中元素的数量
        num_elements = 10
        # 设置张量的缩放因子
        scale = 1.0
        # 设置张量的零点
        zero_point = 2

        # 遍历三种量化数据类型
        for dtype in [torch.qint8, torch.quint8, torch.qint32]:
            # 创建一个元素全为1的张量，指定数据类型和设备
            r = torch.ones(num_elements, dtype=torch.float, device=device)
            # 对该张量进行量化
            qr = torch.quantize_per_tensor(r, scale, zero_point, dtype)
            # 断言量化后的张量的缩放因子等于预期值
            self.assertEqual(qr.q_scale(), scale)
            # 断言量化后的张量的零点等于预期值
            self.assertEqual(qr.q_zero_point(), zero_point)
            # 断言量化后的张量已被量化
            self.assertTrue(qr.is_quantized)
            # 断言原始张量未被量化
            self.assertFalse(r.is_quantized)
            # 断言量化方案为每个张量的仿射量化
            self.assertEqual(qr.qscheme(), torch.per_tensor_affine)
            # 断言量化方案类型为 torch.qscheme
            self.assertTrue(isinstance(qr.qscheme(), torch.qscheme))
            # 对量化后的张量进行切片和整数表示的测试
            int_repr = qr.int_repr()
            for num in int_repr:
                self.assertEqual(num, 3)
            for num in qr[2:].int_repr():
                self.assertEqual(num, 3)
            # 对量化后的张量进行反量化操作的测试
            rqr = qr.dequantize()
            for i in range(num_elements):
                self.assertEqual(r[i], rqr[i])
            # 测试空张量的量化和打印
            empty_r = torch.ones((0, 1), dtype=torch.float, device=device)
            empty_qr = torch.quantize_per_tensor(empty_r, scale, zero_point, dtype)

            device_msg = "" if device == 'cpu' else "device='" + device + ":0', "
            dtype_msg = str(dtype) + ", "
            # 断言打印的空张量的格式符合预期
            self.assertEqual(' '.join(str(empty_qr).split()),
                             "tensor([], " + device_msg + "size=(0, 1), dtype=" + dtype_msg +
                             "quantization_scheme=torch.per_tensor_affine, " +
                             "scale=1.0, zero_point=2)")
    def test_qtensor_int_repr(self):
        # 当元素数量 * 比特率 < 8 时，确保至少分配一个字节来保存整数表示
        num_elements = 1
        # 设备选择为 CPU
        device = torch.device('cpu')
        # 缩放比例为 1.0
        scale = 1.0
        # 零点偏移为 2
        zero_point = 2
        # 数据类型为 torch.quint2x4
        dtype = torch.quint2x4
        # 创建一个包含1个元素的张量，数据类型为 float，在指定设备上
        r = torch.ones(num_elements, dtype=torch.float, device=device)
        # 对张量进行量化
        qr = torch.quantize_per_tensor(r, scale, zero_point, dtype)
        # 获取量化后的整数表示
        int_repr = qr.int_repr()
        # 断言整数表示的元素数量为 1
        self.assertEqual(int_repr.numel(), 1)
        # 第一个元素的整数表示应为 3
        self.assertEqual(int_repr[0], 3)

    def test_qtensor_sub_byte_aligned_cols(self):
        # 对于 torch.quint4x2 数据类型，打包的4个值每个为3，看起来像 00110011, 00110011；
        # 对于 torch.quint2x4 数据类型，打包的一个字节为 11111111
        self._test_qtensor_sub_byte(1, 4, torch.quint4x2, 2, [51, 51])
        self._test_qtensor_sub_byte(1, 4, torch.quint2x4, 4, [255])

    def test_qtensor_sub_byte_not_aligned_cols(self):
        # 对于 torch.quint4x2 数据类型，打包的5个值每个为3，看起来像 00110011, 00110011, 00000011；
        # 对于 torch.quint2x4 数据类型，打包的两个字节为 11111111, 00000011
        self._test_qtensor_sub_byte(1, 5, torch.quint4x2, 2, [51, 51, 3])
        self._test_qtensor_sub_byte(1, 5, torch.quint2x4, 4, [255, 3])

    def _test_qtensor_sub_byte(self, rows, cols, dtype, elements_per_byte, expected_packed_vals):
        # 计算元素总数
        num_elements = rows * cols
        # 缩放比例为 1.0
        scale = 1.0
        # 零点偏移为 2
        zero_point = 2

        # 创建一个所有元素为1的张量，数据类型为 float
        r = torch.ones((rows, cols), dtype=torch.float)
        # 对张量进行量化
        qr = torch.quantize_per_tensor(r, scale, zero_point, dtype)
        # 断言量化的缩放比例与零点偏移
        self.assertEqual(qr.q_scale(), scale)
        self.assertEqual(qr.q_zero_point(), zero_point)
        # 断言张量已经被量化
        self.assertTrue(qr.is_quantized)
        # 断言原始张量未量化
        self.assertFalse(r.is_quantized)
        # 断言量化后张量的存储大小符合预期
        self.assertEqual(qr.storage().size(), rows * math.ceil(cols / elements_per_byte), f"with {dtype}, {elements_per_byte}")

        # 获取量化后的整数表示
        int_repr = qr.int_repr()
        # 断言整数表示的元素数量与预期的打包值数量相同
        self.assertEqual(int_repr.numel(), len(expected_packed_vals))
        # 逐个比较每个整数表示的值与预期的打包值
        for num, expected in zip(int_repr, expected_packed_vals):
            self.assertEqual(num, expected, f"with dtype={dtype}, elements_per_byte={elements_per_byte}, rows={rows}, cols={cols}")

        # 测试张量的创建
        q = torch._empty_affine_quantized([num_elements], scale=scale, zero_point=zero_point, dtype=dtype)
        # 断言张量的存储大小符合预期
        self.assertEqual(q.storage().size(), math.ceil(num_elements / elements_per_byte), f"with {dtype}, {elements_per_byte}")

        # 测试保存和加载
        with tempfile.NamedTemporaryFile() as f:
            torch.save(qr, f)
            for weights_only in [True, False]:
                f.seek(0)
                loaded_q = torch.load(f, weights_only=weights_only)
                loaded_int_repr = loaded_q.int_repr()
                # 断言加载的整数表示与原始的整数表示相同
                self.assertEqual(int_repr, loaded_int_repr)
    # 定义一个测试方法，用于测试量化张量的通道级浮点赋值
    def test_qtensor_channel_float_assignment(self):
        # 创建两个随机张量 t1 和 t2，形状为 (2, 3, 5, 5)
        t1 = torch.rand(2, 3, 5, 5)
        t2 = torch.rand(2, 3, 5, 5)
        # 遍历张量 t1 的每个维度
        for axis in range(t1.ndim):
            # 随机生成每个维度上的标度和零点
            scales = np.random.rand(t1.size()[axis])
            zero_points = np.random.randint(low=0, high=50, size=t1.size()[axis])
            # 对于每种数据类型，进行量化通道级张量的测试
            for dtype in [torch.qint8, torch.quint8, torch.qint32]:
                # 使用 torch.quantize_per_channel 方法量化 t1 和 t2
                qt1 = torch.quantize_per_channel(t1, scales=torch.tensor(scales),
                                                 zero_points=torch.tensor(zero_points), dtype=dtype, axis=axis)
                qt2 = torch.quantize_per_channel(t2, scales=torch.tensor(scales),
                                                 zero_points=torch.tensor(zero_points), dtype=dtype, axis=axis)
                # 设定索引变量 i, j, k, l
                i = 0
                j = 1
                k = 2
                l = 4
                # 标量赋值验证
                qt1[i][j][k][l] = t2[i][j][k][l]
                # 断言量化后的值与 t2 相同
                self.assertEqual(qt1[i][j][k][l], qt2[i][j][k][l])
                # 1D 张量赋值验证
                qt1[i][j][k][2:l] = t2[i][j][k][2:l]
                self.assertEqual(qt1[i][j][k][2:l], qt2[i][j][k][2:l])
                qt1[i][j][k] = t2[i][j][k]
                self.assertEqual(qt1[i][j][k], qt2[i][j][k])
                # 2D 张量赋值验证
                qt1[i][j][k:] = t2[i][j][k:]
                self.assertEqual(qt1[i][j][k:], qt2[i][j][k:])
                qt1[i][j] = t2[i][j]
                self.assertEqual(qt1[i][j], qt2[i][j])
                # 3D 张量赋值验证
                qt1[i][j:] = t2[i][j:]
                self.assertEqual(qt1[i][j:], qt2[i][j:])
                qt1[i] = t2[i]
                self.assertEqual(qt1[i], qt2[i])
                # 4D 张量赋值验证
                qt1[:1] = t2[:1]
                self.assertEqual(qt1[:1], qt2[:1])
                qt1[:] = t2[:]
                self.assertEqual(qt1[:], qt2[:])
                # 非连续情况，应该引发异常
                with self.assertRaisesRegex(RuntimeError, "Quantized copy only works with contiguous and NHWC Tensors"):
                    qt1[:, 0] = t2[:, 0]
    # 定义测试方法，用于测试量化张量的浮点赋值操作
    def test_qtensor_float_assignment(self):
        # 标量张量的赋值，初始化 scale 和 zero_point
        scale = 1.0
        zero_point = 2
        # 根据 GPU 是否可用选择设备
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        # 遍历设备列表
        for device in devices:
            # 创建一个值为 1.0 的张量，并移动到指定设备
            r = torch.ones(1, dtype=torch.float).to(device=device)
            # 遍历数据类型列表：torch.qint8, torch.quint8, torch.qint32
            for dtype in [torch.qint8, torch.quint8, torch.qint32]:
                # 对张量 r 进行量化操作，得到量化后的张量 qr
                qr = torch.quantize_per_tensor(r, scale, zero_point, dtype=dtype)
                # 断言量化后张量的数值为 1
                self.assertEqual(qr.item(), 1)
                # 断言量化后张量的第一个元素为 1
                self.assertEqual(qr[0].item(), 1)
                # 断言量化后张量的第一个元素已经被量化
                self.assertTrue(qr[0].is_quantized)
                # 将浮点张量 [11.3] 赋值给量化张量 qr 的第一个元素
                qr[0] = torch.Tensor([11.3]).to(device=device)  # float assignment
                # 断言量化后张量的数值为 11
                self.assertEqual(qr.item(), 11)
                # 创建一个浮点张量 x，值为 15.3，并移动到指定设备
                x = torch.ones(1, dtype=torch.float).to(device=device) * 15.3
                # 将浮点张量 x 的值复制到量化张量 qr 中
                qr[:] = x
                # 断言量化后张量的数值为 15
                self.assertEqual(qr.item(), 15)

                # 生成描述数据类型的消息字符串
                dtype_msg = str(dtype) + ", "
                # 检查如果设备为 "cuda"，则进行特定的字符串断言
                if device == "cuda":
                    self.assertEqual(' '.join(str(qr).split()),
                                     "tensor([15.], device='" + str(qr.device) + "', size=(1,), dtype=" + dtype_msg +
                                     "quantization_scheme=torch.per_tensor_affine, " +
                                     "scale=1.0, zero_point=2)")
                else:
                    self.assertEqual(' '.join(str(qr).split()),
                                     "tensor([15.], size=(1,), dtype=" + dtype_msg +
                                     "quantization_scheme=torch.per_tensor_affine, " +
                                     "scale=1.0, zero_point=2)")

    # 测试量化张量的量化和反量化操作
    def test_qtensor_quant_dequant(self):
        # 设置量化的 scale 和 zero_point
        scale = 0.02
        zero_point = 2
        # 遍历支持的设备类型
        for device in get_supported_device_types():
            # 创建一个随机浮点张量 r，形状为 (3, 2, 4, 5)，数值范围为 [-2, 2)
            r = torch.rand(3, 2, 4, 5, dtype=torch.float, device=device) * 4 - 2
            # 遍历内存布局格式：torch.contiguous_format, torch.channels_last
            for memory_format in [torch.contiguous_format, torch.channels_last]:
                # 将张量 r 转换为指定内存布局格式
                r = r.contiguous(memory_format=memory_format)
                # 遍历数据类型列表：torch.qint8, torch.quint8, torch.qint32
                for dtype in [torch.qint8, torch.quint8, torch.qint32]:
                    # 对浮点张量 r 进行量化操作，得到量化后的张量 qr
                    qr = torch.quantize_per_tensor(r, scale, zero_point, dtype)
                    # 将量化张量 qr 进行反量化操作，得到反量化后的张量 rqr
                    rqr = qr.dequantize()
                    # 断言量化和反量化后的张量 r 和 rqr 数值相近，精度为 2/scale
                    self.assertTrue(np.allclose(r.cpu().numpy(), rqr.cpu().numpy(), atol=2 / scale))
        
        # 测试也支持 5 维张量的量化和反量化操作
        for device in get_supported_device_types():
            # 创建一个随机浮点张量 r，形状为 (3, 2, 4, 5, 6)，数值范围为 [-2, 2)
            r = torch.rand(3, 2, 4, 5, 6, dtype=torch.float, device=device) * 4 - 2
            # 遍历数据类型列表：torch.qint8, torch.quint8, torch.qint32
            for dtype in [torch.qint8, torch.quint8, torch.qint32]:
                # 对浮点张量 r 进行量化操作，得到量化后的张量 qr
                qr = torch.quantize_per_tensor(r, scale, zero_point, dtype)
                # 将量化张量 qr 进行反量化操作，得到反量化后的张量 rqr
                rqr = qr.dequantize()
                # 断言量化和反量化后的张量 r 和 rqr 数值相近，精度为 2/scale
                self.assertTrue(np.allclose(r.cpu().numpy(), rqr.cpu().numpy(), atol=2 / scale))

    # 旧的构造方法/新的构造方法不支持量化张量
    # 定义测试函数，用于测试量化张量的新建操作失败的情况
    def test_qtensor_legacy_new_failure(self):
        # 创建一个随机张量 r，形状为 (3, 2)，数据类型为 torch.float，数值范围在 [-2, 2)
        r = torch.rand(3, 2, dtype=torch.float) * 4 - 2
        # 设置量化参数：缩放因子为 0.02，零点为 2
        scale = 0.02
        zero_point = 2
        # 对随机张量 r 进行量化操作，得到量化后的张量 qr
        qr = torch.quantize_per_tensor(r, scale, zero_point, torch.quint8)
        # 测试以下操作是否会引发 RuntimeError 异常
        self.assertRaises(RuntimeError, lambda: qr.new(device='cpu'))
        self.assertRaises(RuntimeError, lambda: qr.new(r.storage()))
        self.assertRaises(RuntimeError, lambda: qr.new(r))
        self.assertRaises(RuntimeError, lambda: qr.new(torch.Size([2, 3])))
        self.assertRaises(RuntimeError, lambda: qr.new([6]))

    # 测试在 CPU 上创建通道量化张量的函数
    def test_per_channel_qtensor_creation_cpu(self):
        self._test_per_channel_qtensor_creation(torch.device('cpu'))

    # 测试在指定设备上进行 FP16 解量化操作的函数
    def _test_dequantize_fp16(self, device):
        # 创建一个随机张量 data_orig，形状为 (1, 2, 4, 4)，数据类型为 torch.float，在指定设备上
        data_orig = torch.randn(1, 2, 4, 4, dtype=torch.float, device=device)
        # 将 data_orig 转换为 torch.float16 类型
        data_fp16 = data_orig.to(torch.float16)
        # 对 data_fp16 进行解量化操作，得到 data_fp16_dequant
        data_fp16_dequant = data_fp16.dequantize()
        # 将 data_fp16 转换为 torch.float32 类型
        data_fp16_fp32 = data_fp16.to(torch.float)
        # 断言 data_fp16_dequant 的数据类型为 torch.float
        self.assertTrue(data_fp16_dequant.dtype == torch.float)
        # 断言 data_fp16_fp32 与 data_fp16_dequant 在数值上的近似性
        self.assertTrue(torch.allclose(data_fp16_fp32, data_fp16_dequant))

    # 测试在 CPU 上进行 FP16 解量化操作的函数
    def test_dequantize_fp16_cpu(self):
        self._test_dequantize_fp16(torch.device('cpu'))

    # 如果没有可用的 GPU，则跳过测试 CUDA 上的 FP16 解量化操作
    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def test_dequantize_fp16_cuda(self):
        self._test_dequantize_fp16(torch.device('cuda'))

    # 如果没有可用的 GPU，则跳过测试 CUDA 上创建通道量化张量的函数
    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def test_per_channel_qtensor_creation_cuda(self):
        self._test_per_channel_qtensor_creation(torch.device('cuda'))

    # 在指定设备上测试创建通道量化张量的函数
    def _test_per_channel_qtensor_creation(self, device):
        # 定义张量的元素数量为 10
        numel = 10
        # 通道轴的索引为 0
        ch_axis = 0
        # 创建随机的缩放因子张量 scales，形状为 (numel,)，在指定设备上
        scales = torch.rand(numel, device=device)
        # 创建随机的零点张量 zero_points_int，元素值在 [0, 10) 范围内，形状为 (numel,)，在指定设备上
        zero_points_int = torch.randint(0, 10, size=(numel,), device=device)
        # 创建随机的浮点数零点张量 zero_points_float，形状为 (numel,)，在指定设备上
        zero_points_float = torch.randn(numel, device=device)
        # 遍历量化类型和零点张量的组合，使用 torch._empty_per_channel_affine_quantized 创建通道量化张量 q
        for dtype, zero_points in itertools.product([torch.qint8, torch.quint8], [zero_points_float, zero_points_int]):
            q = torch._empty_per_channel_affine_quantized(
                [numel], scales=scales, zero_points=zero_points, axis=ch_axis, dtype=dtype, device=device)
            # 断言张量的缩放因子与预期值相等（忽略精确的数据类型）
            self.assertEqual(scales, q.q_per_channel_scales(), exact_dtype=False)
            # 断言张量的零点张量与预期值相等
            self.assertEqual(zero_points, q.q_per_channel_zero_points())
            # 断言张量的通道轴与预期值相等
            self.assertEqual(ch_axis, q.q_per_channel_axis())

        # 从 uint8_t 类型的张量、缩放因子和零点张量创建通道量化张量
        for zero_points in [zero_points_float, zero_points_int]:
            int_tensor = torch.randint(0, 100, size=(numel,), dtype=torch.uint8, device=device)
            q = torch._make_per_channel_quantized_tensor(int_tensor, scales, zero_points, ch_axis)
            # 断言张量的整数表示与预期值相等
            self.assertEqual(int_tensor, q.int_repr())
            # 断言张量的缩放因子与预期值相等（忽略精确的数据类型）
            self.assertEqual(scales, q.q_per_channel_scales(), exact_dtype=False)
            # 断言张量的零点张量与预期值相等
            self.assertEqual(zero_points, q.q_per_channel_zero_points())
            # 断言张量的通道轴与预期值相等
            self.assertEqual(ch_axis, q.q_per_channel_axis())
    # 定义测试量化张量创建的方法
    def test_qtensor_creation(self):
        # 设置量化参数：缩放因子为0.5，零点为10，张量元素个数为10
        scale = 0.5
        zero_point = 10
        numel = 10
        # 遍历支持的设备类型列表
        for device in get_supported_device_types():
            # 使用 torch._empty_affine_quantized 创建一个量化张量 q，指定缩放因子、零点、设备类型和数据类型为 torch.quint8
            q = torch._empty_affine_quantized([numel], scale=scale, zero_point=zero_point,
                                              device=device, dtype=torch.quint8)
            # 断言量化张量 q 的缩放因子与预期的 scale 相等
            self.assertEqual(scale, q.q_scale())
            # 断言量化张量 q 的零点与预期的 zero_point 相等
            self.assertEqual(zero_point, q.q_zero_point())

            # 创建一个整型张量 int_tensor，其元素为在指定设备上，dtype 为 torch.uint8，数值范围在 [0, 100) 之间的随机整数
            int_tensor = torch.randint(0, 100, size=(10,), device=device, dtype=torch.uint8)
            # 使用 torch._make_per_tensor_quantized_tensor 创建一个量化张量 q，基于 int_tensor、scale 和 zero_point
            q = torch._make_per_tensor_quantized_tensor(int_tensor, scale, zero_point)
            # 断言量化张量 q 的整数表示（int_repr）与原始整型张量 int_tensor 相等
            self.assertEqual(int_tensor, q.int_repr())
            # 断言量化张量 q 的缩放因子与预期的 scale 相等
            self.assertEqual(scale, q.q_scale())
            # 断言量化张量 q 的零点与预期的 zero_point 相等
            self.assertEqual(zero_point, q.q_zero_point())

            # 使用 torch._empty_affine_quantized 创建一个量化张量 q，指定缩放因子、零点、设备类型和数据类型为 torch.quint8
            q = torch._empty_affine_quantized([numel], scale=scale, zero_point=zero_point,
                                              device=device, dtype=torch.quint8)
            # 使用 torch.empty_like 创建一个与 q 相同形状的量化张量 q_el
            q_el = torch.empty_like(q)
            # 断言量化张量 q_el 的缩放因子与 q 相同
            self.assertEqual(q.q_scale(), q_el.q_scale())
            # 断言量化张量 q_el 的零点与 q 相同
            self.assertEqual(q.q_zero_point(), q_el.q_zero_point())
            # 断言量化张量 q_el 的数据类型与 q 相同
            self.assertEqual(q.dtype, q_el.dtype)

            # 使用 torch.empty_like 创建一个与 q 相同形状的量化张量 q，但尝试更改其数据类型（目前不支持）
            with self.assertRaises(RuntimeError):
                torch.empty_like(q, dtype=torch.qint8)

    # 定义测试量化张量数据类型的方法
    def test_qtensor_dtypes(self):
        # 创建一个随机浮点张量 r，形状为 (3, 2)，数值范围在 [-2, 2) 之间
        r = torch.rand(3, 2, dtype=torch.float) * 4 - 2
        # 设置量化参数：缩放因子为0.2，零点为2
        scale = 0.2
        zero_point = 2
        # 遍历不同的量化数据类型：torch.qint8、torch.quint8、torch.qint32、torch.quint4x2、torch.quint2x4
        for dtype in [torch.qint8, torch.quint8, torch.qint32, torch.quint4x2, torch.quint2x4]:
            # 使用 torch.quantize_per_tensor 将浮点张量 r 量化为 qr，使用指定的缩放因子、零点和数据类型 dtype
            qr = torch.quantize_per_tensor(r, scale, zero_point, dtype)
            # 将量化张量 qr 还原成浮点张量 rqr
            rqr = qr.dequantize()
            # 断言浮点张量 r 与还原的浮点张量 rqr 在给定的容差范围内相等
            self.assertTrue(np.allclose(r.numpy(), rqr.numpy(), atol=2 / scale))

    # 定义测试量化张量在设备间迁移的方法，如果没有 GPU 可用则跳过测试
    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def test_per_tensor_to_device(self):
        # 定义量化数据类型列表
        dtypes = [
            torch.quint8,
            torch.qint8,
            torch.qint32,
        ]
        # 将设备类型设置为 CUDA
        device = torch.device('cuda')
        # 遍历不同的量化数据类型
        for dtype in dtypes:
            # 创建一个随机浮点张量 r，形状为 (2, 2)，数值范围在 [0, 10) 之间
            r = torch.rand(2, 2, dtype=torch.float) * 10
            # 计算随机生成的缩放因子和零点
            scale = torch.rand(2).abs().max().item()
            zero_point = (torch.rand(2) * 10).round().to(torch.long).max().item()

            # 使用 torch.quantize_per_tensor 将浮点张量 r 量化为 qr，使用指定的缩放因子、零点和数据类型 dtype
            qr = torch.quantize_per_tensor(r, scale, zero_point, dtype)
            # 将量化张量 qr 移动到 CUDA 设备
            qr = qr.to(device)
            # 将 CUDA 设备上的量化张量 qr 移动回 CPU
            qr_cuda = torch.quantize_per_tensor(r.to(device), scale, zero_point, dtype)
            qr_cuda = qr_cuda.to('cpu')
            # 断言量化张量 qr 的设备类型为 CUDA
            self.assertEqual('cuda', qr.device.type)
            # 断言量化张量 qr_cuda 的设备类型为 CPU
            self.assertEqual('cpu', qr_cuda.device.type)
    def test_per_channel_to_device(self):
        # 定义不同的量化类型和零点类型组合
        dtype_and_zero_types = [
            (torch.quint8, torch.float),
            (torch.qint8, torch.float),
            #  (torch.qint32, torch.float) not supported for quantize_per_channel
            (torch.quint8, torch.long),
            (torch.qint8, torch.long),
            (torch.qint32, torch.long),
        ]
        # 设置量化的轴向和设备为CUDA
        axis = 1
        device = torch.device('cuda')
        # 遍历每一种量化类型和零点类型的组合
        for dtype, zero_type in dtype_and_zero_types:
            # 创建一个随机张量 r，并乘以 10
            r = torch.rand(2, 2, dtype=torch.float) * 10
            # 创建随机的量化尺度
            scales = torch.rand(2).abs()
            # 创建随机的零点，并四舍五入为指定类型
            zero_points = (torch.rand(2) * 10).round().to(zero_type)

            # 在每个量化通道上对张量 r 进行量化
            dqr = torch.quantize_per_channel(r, scales, zero_points, axis, dtype)
            # 将量化后的张量移动到指定设备上
            dqr = dqr.to(device)
            # 在CUDA设备上对张量 r 进行量化
            dqr_cuda = torch.quantize_per_channel(r.to(device), scales.to(
                device), zero_points.to(device), axis, dtype)
            # 将CUDA上的量化张量移动回CPU
            dqr_cuda = dqr_cuda.to('cpu')

            # 断言量化后的张量 dqr 在CUDA设备上
            self.assertEqual('cuda', dqr.device.type)
            # 断言量化后的每个通道的尺度在CUDA设备上
            self.assertEqual('cuda', dqr.q_per_channel_scales().device.type)
            # 断言量化后的每个通道的零点在CUDA设备上
            self.assertEqual('cuda', dqr.q_per_channel_zero_points().device.type)

            # 断言量化后的张量 dqr_cuda 在CPU上
            self.assertEqual('cpu', dqr_cuda.device.type)
            # 断言量化后的每个通道的尺度在CPU上
            self.assertEqual('cpu', dqr_cuda.q_per_channel_scales().device.type)
            # 断言量化后的每个通道的零点在CPU上
            self.assertEqual('cpu', dqr_cuda.q_per_channel_zero_points().device.type)

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_compare_per_tensor_device_numerics(self):
        # 定义不同的量化类型
        dtypes = [
            torch.quint8,
            torch.qint8,
            torch.qint32,
        ]
        # 设置设备为CUDA
        device = torch.device('cuda')
        # 遍历每一种量化类型
        for dtype in dtypes:
            # 创建一个随机张量 r，并乘以 10
            r = torch.rand(2, 2) * 10
            # 修改张量的特定元素值
            r[0, 0] = 2.5
            # 计算张量的最大值的绝对值作为尺度
            scale = torch.rand(2).abs().max().item()
            # 创建随机的零点，并四舍五入为长整型后取最大值
            zero_point = (torch.rand(2) * 10).round().to(torch.long).max().item()

            # 对张量 r 进行全局量化
            qtr = torch.quantize_per_tensor(r, scale, zero_point, dtype)
            # 对全局量化的张量进行反量化
            dqtr = qtr.dequantize()
            # 在CUDA设备上对张量 r 进行全局量化
            qtr_cuda = torch.quantize_per_tensor(r.to(device), scale, zero_point, dtype)
            # 在CUDA设备上对全局量化的张量进行反量化
            dqtr_cuda = qtr_cuda.dequantize()
            
            # 断言全局量化后的整数表示在CUDA设备上一致
            self.assertEqual(qtr.int_repr(), qtr_cuda.int_repr())
            # 断言反量化后的值在CPU上近似一致
            self.assertTrue(np.allclose(dqtr, dqtr_cuda.cpu()))

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_compare_per_channel_device_numerics(self):
        # 定义要测试的数据类型组合
        dtype_and_zero_types = [
            (torch.quint8, torch.float),
            (torch.qint8, torch.float),
            #  (torch.qint32, torch.float) not supported for quantize_per_channel
            (torch.quint8, torch.long),
            (torch.qint8, torch.long),
            (torch.qint32, torch.long),
        ]
        # 设置通道轴的索引
        axis = 1
        # 指定设备为 CUDA
        device = torch.device('cuda')
        # 外层循环，重复20次
        for i in range(20):
            # 遍历每种数据类型组合
            for dtype, zero_type in dtype_and_zero_types:
                # 创建一个 2x2 的随机张量并扩展范围到 [0, 25)
                r = torch.rand(2, 2) * 10
                r[0, 0] = 2.5  # 设置特定值
                # 随机生成绝对值后的两个比例因子
                scales = torch.rand(2).abs()
                # 将随机生成的 [0, 10) 范围内的值四舍五入，并转换为指定类型
                zero_points = (torch.rand(2) * 10).round().to(zero_type)

                # 在给定轴上进行通道量化操作
                qr = torch.quantize_per_channel(r, scales, zero_points, axis, dtype)
                # 对量化结果进行反量化
                dqr = qr.dequantize()
                # 将张量移到 CUDA 设备上并进行通道量化
                qr_cuda = torch.quantize_per_channel(r.to(device), scales.to(
                    device), zero_points.to(device), axis, dtype)
                # 对 CUDA 上的量化结果进行反量化
                dqr_cuda = qr_cuda.dequantize()
                # 断言两个量化结果的整数表示是否相等
                self.assertEqual(qr.int_repr(), qr_cuda.int_repr())
                # 断言两个反量化结果是否非常接近
                self.assertTrue(np.allclose(dqr, dqr_cuda.cpu()))

    def _test_quantize_per_channel(self, r, scales, zero_points, axis, float_params):

        def _quantize_per_channel_ref_nd(data, scales, zero_points, float_params):
            # 获取数据的维度
            dims = data.size()
            # 对数据进行维度变换，以便对通道进行量化
            data = data.view(-1, dims[axis], np.prod(dims[axis + 1:]))
            # 创建一个空的张量，与输入数据大小相同
            res = torch.empty_like(data)
            # 设置量化的最小值和最大值
            quant_min, quant_max = 0, 255
            # 遍历数据的每个元素并进行量化
            for i in range(res.size()[0]):
                for j in range(res.size()[1]):
                    for k in range(res.size()[2]):
                        if float_params:
                            # 如果是浮点参数，使用浮点运算量化
                            inv_scale = 1.0 / scales[j]
                            res[i][j][k] = np.clip(
                                np.round(data[i][j][k] * inv_scale + zero_points[j]), quant_min, quant_max)
                        else:
                            # 否则使用整数运算量化
                            res[i][j][k] = np.clip(
                                np.round(data[i][j][k] / scales[j]) + zero_points[j], quant_min, quant_max)
            # 将结果变换回原始数据的维度
            res = res.view(*dims)
            return res

        # 根据输入张量的维度，选择合适的内存格式进行测试
        contig_format = torch.channels_last if r.ndim == 4 else torch.channels_last_3d
        # 遍历测试使用的内存格式
        for memory_format in [torch.contiguous_format, contig_format]:
            # 使用参考函数进行量化
            ref_res = _quantize_per_channel_ref_nd(r, scales, zero_points, float_params)
            # 使用指定的内存格式使输入张量连续
            r_contig = r.contiguous(memory_format=memory_format)
            # 对连续化后的张量进行通道量化
            qr = torch.quantize_per_channel(r_contig, scales, zero_points, axis, torch.quint8)
            # 反量化量化后的张量
            rqr = qr.dequantize()
            # 断言量化后的结果与参考结果是否非常接近
            self.assertTrue(np.allclose(qr.int_repr(), ref_res))
            # 断言反量化后的结果与原始张量是否非常接近，考虑到量化误差
            self.assertTrue(np.allclose(r.numpy(), rqr.numpy(), atol=2 / np.min(scales.numpy())))
    # 定义一个测试函数，用于测试量化每个通道的操作
    def test_qtensor_quantize_per_channel(self):
        # 创建一个形状为 (3, 2) 的随机张量，数据类型为 float，值范围在 [-2, 2)
        r = torch.rand(3, 2, dtype=torch.float) * 4 - 2
        # 设置每个通道的量化比例因子，数据类型为 double
        scales = torch.tensor([0.2, 0.03], dtype=torch.double)
        # 设置每个通道的零点，数据类型为 long
        zero_points = torch.tensor([5, 10], dtype=torch.long)
        # 指定量化操作的轴向为 1
        axis = 1

        # 定义一个内部函数 quantize_c，用于实现张量的自定义量化过程
        def quantize_c(data, scales, zero_points):
            # 创建一个形状为 (3, 2) 的空张量 res，用于存储量化后的结果
            res = torch.empty((3, 2))
            # 设置量化的最小值和最大值
            quant_min, quant_max = 0, 255
            # 循环遍历张量的每个元素，进行量化计算
            for i in range(3):
                for j in range(2):
                    # 计算每个元素的量化值，使用 np.clip 进行范围约束
                    res[i][j] = np.clip(np.round(data[i][j] / scales[j]) + zero_points[j], quant_min, quant_max)
            # 返回量化后的结果张量
            return res
        
        # 对张量 r 进行每通道量化操作，返回量化后的结果 qr
        qr = torch.quantize_per_channel(r, scales, zero_points, axis, torch.quint8)
        # 对量化后的结果 qr 进行反量化操作，得到反量化后的结果 rqr
        rqr = qr.dequantize()
        # 使用 numpy 函数 np.allclose 检查量化后的整数表示是否与自定义量化函数 quantize_c 计算的结果相近
        self.assertTrue(np.allclose(qr.int_repr(), quantize_c(r, scales, zero_points)))
        # 使用 numpy 函数 np.allclose 检查反量化后的结果是否与原始张量 r 相近，设置允许的误差范围为 2 / scales 的最小值
        self.assertTrue(np.allclose(r.numpy(), rqr.numpy(), atol=2 / np.min(scales.numpy())))

        # 检查形状为 (3, 2, 4, 5) 的张量，使用自定义测试函数 _test_quantize_per_channel 进行每通道量化测试
        r = torch.rand(3, 2, 4, 5, dtype=torch.float) * 4 - 2
        # 设置量化比例因子和零点，进行每通道量化测试
        scales = torch.tensor([0.2, 0.03], dtype=torch.double)
        zero_points = torch.tensor([5, 10], dtype=torch.long)
        self._test_quantize_per_channel(r, scales, zero_points, 1, False)

        # 设置不同的量化比例因子和零点，进行每通道量化测试
        scales = torch.tensor([0.2, 0.03, 0.5], dtype=torch.double)
        zero_points = torch.tensor([5, 10, 7], dtype=torch.long)
        self._test_quantize_per_channel(r, scales, zero_points, 0, False)

        # 检查形状为 (3, 2, 4, 5, 7) 的张量，使用自定义测试函数 _test_quantize_per_channel 进行每通道量化测试
        r = torch.rand(3, 2, 4, 5, 7, dtype=torch.float) * 4 - 2
        # 设置量化比例因子和零点，进行每通道量化测试
        scales = torch.tensor([0.2, 0.03], dtype=torch.double)
        zero_points = torch.tensor([5, 10], dtype=torch.long)
        self._test_quantize_per_channel(r, scales, zero_points, 1, False)

        # 设置不同的量化比例因子和零点，进行每通道量化测试
        scales = torch.tensor([0.2, 0.03, 0.5], dtype=torch.double)
        zero_points = torch.tensor([5, 10, 7], dtype=torch.long)
        self._test_quantize_per_channel(r, scales, zero_points, 0, False)
    # 定义测试函数，用于测试按通道量化的浮点参数
    def test_quantize_per_channel_float_qparams(self):
        # 创建一个形状为 (3, 2) 的随机浮点张量，元素范围在 [0, 4) 之间
        r = torch.rand(3, 2, dtype=torch.float) * 4
        # 定义按通道的量化参数：缩放因子
        scales = torch.tensor([0.2, 0.03], dtype=torch.float)
        # 定义按通道的量化参数：零点
        zero_points = torch.tensor([0.1, 0.2], dtype=torch.float)
        # 指定量化操作的轴
        axis = 1

        # 参考量化函数，使用浮点数零点
        def quantize_ref(data, scales, zero_points):
            # 创建一个形状为 (3, 2) 的空张量用于存放结果
            res = torch.empty((3, 2))
            # 定义量化的最小和最大值
            quant_min, quant_max = 0, 255
            # 循环遍历张量中的每个元素
            for i in range(3):
                for j in range(2):
                    # 计算逆缩放因子
                    inv_scale = 1.0 / scales[j]
                    # 对数据进行量化，并使用 np.clip 限制在 [quant_min, quant_max] 范围内
                    res[i][j] = np.clip(np.round(data[i][j] * inv_scale + zero_points[j]), quant_min, quant_max)
            return res

        # 对随机张量 r 进行按通道量化，得到量化后的张量 qr
        qr = torch.quantize_per_channel(r, scales, zero_points, axis, torch.quint8)
        # 将量化后的张量反量化为浮点数张量
        dequant_tensor = qr.dequantize()
        # 使用参考量化函数 quantize_ref 对 r 进行量化作为参考值 ref
        ref = quantize_ref(r, scales, zero_points)
        # 使用 np.allclose 检查 qr 的整数表示是否与参考值 ref 相近
        self.assertTrue(np.allclose(qr.int_repr(), ref))
        # 使用 np.allclose 检查反量化后的张量是否与原始浮点张量 r 相近，允许误差为 1
        self.assertTrue(np.allclose(r.numpy(), dequant_tensor.numpy(), atol=1))

        # 检查具有两种不同内存格式的形状为 4D 的张量
        r = torch.rand(3, 2, 4, 5, dtype=torch.float) * 4
        # 使用不同的缩放因子和零点参数进行按通道量化测试，并指定轴为 1
        scales = torch.tensor([0.2, 0.03], dtype=torch.float)
        zero_points = torch.tensor([0.1, 0.2], dtype=torch.float)
        self._test_quantize_per_channel(r, scales, zero_points, 1, True)

        # 使用不同的缩放因子和零点参数进行按通道量化测试，并指定轴为 0
        scales = torch.tensor([0.2, 0.03, 0.5], dtype=torch.float)
        zero_points = torch.tensor([0.1, 0.2, 1.], dtype=torch.float)
        self._test_quantize_per_channel(r, scales, zero_points, 0, True)

        # 检查形状为 5D 的张量
        r = torch.rand(3, 2, 4, 5, 7, dtype=torch.float) * 4 - 2
        # 使用不同的缩放因子和零点参数进行按通道量化测试，并指定轴为 1
        scales = torch.tensor([0.2, 0.03], dtype=torch.float)
        zero_points = torch.tensor([0.1, 0.2], dtype=torch.float)
        self._test_quantize_per_channel(r, scales, zero_points, 1, True)

        # 使用不同的缩放因子和零点参数进行按通道量化测试，并指定轴为 0
        scales = torch.tensor([0.2, 0.03, 0.5], dtype=torch.float)
        zero_points = torch.tensor([0.1, 0.2, 1.], dtype=torch.float)
        self._test_quantize_per_channel(r, scales, zero_points, 0, True)
    def test_quantize_per_channel_sub_byte(self):
        """ Tests the per channel quantization scheme for 4-bit qtensors.
        The scale and zero point for this have to be in floating point. """
        
        # 生成一个3x2的随机张量r，数据类型为float，值在0到4之间
        r = torch.rand(3, 2, dtype=torch.float) * 4
        
        # 创建长度为3的张量，存储浮点类型的scale值
        scales = torch.tensor([0.2, 0.3, 0.1], dtype=torch.float)
        
        # 创建长度为3的张量，存储浮点类型的zero point值
        zero_points = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float)
        
        # 对r进行按通道量化，使用给定的scales和zero points，使用4位量化方案
        qr = torch.quantize_per_channel(r, scales, zero_points, 0, torch.quint4x2)
        
        # 对量化后的张量进行反量化操作
        dequant_tensor = qr.dequantize()

        def _get_qranges(bit_width):
            # 根据比特宽度返回量化范围的最小值和最大值
            if bit_width == 4:
                return 0, 15

        def _quantize_per_channel_sub_byte_ref(data, scales, zero_points, axis, bit_width):
            # 将数据按指定轴进行重塑为二维张量
            dims = data.size()
            data = data.view(-1, dims[axis], np.prod(dims[axis + 1:]))
            
            # 计算需要的输出张量大小，以uint8类型存储
            qtensor_size = math.ceil(data.numel() / 2)
            res = torch.empty(qtensor_size, dtype=torch.uint8)
            
            # 计算每字节存储的元素个数
            elem_per_byte = 8 // bit_width
            
            # 获取量化的最小值和最大值
            quant_min, quant_max = _get_qranges(bit_width)
            
            # 遍历数据进行量化
            for i in range(data.size()[0]):
                for j in range(data.size()[1]):
                    for k in range(data.size()[2]):
                        inv_scale = 1.0 / scales[j]
                        index = i * data.size()[1] * data.size()[2] + j * data.size()[2] + k
                        
                        # 对数据进行量化和截断操作，转换为torch.int类型
                        qvalue = np.clip(
                            np.round(data[i][j][k] * inv_scale + zero_points[j]), quant_min, quant_max).to(dtype=torch.int)
                        
                        # 计算存储结果的索引
                        res_idx = int(index / elem_per_byte)
                        
                        # 按照元素在字节中的位置存储量化结果
                        if (index % elem_per_byte == 0):
                            res[res_idx] = qvalue
                        else:
                            res[res_idx] |= (qvalue << ((index % elem_per_byte) * bit_width))
            return res

        # 使用参考函数计算量化后的参考结果
        ref_res = _quantize_per_channel_sub_byte_ref(r, scales, zero_points, 0, 4)
        
        # 断言量化后的结果和参考结果在数值上的接近程度
        self.assertTrue(np.allclose(qr.int_repr(), ref_res))
        
        # 断言反量化后的结果与原始数据在给定精度下的接近程度
        self.assertTrue(np.allclose(r.numpy(), dequant_tensor.numpy(), atol=1 / np.min(scales.numpy())))

        # 检查具有非零轴的4D张量
        r = torch.rand(3, 2, 4, 5, dtype=torch.float) * 4
        scales = torch.tensor([0.2, 0.03], dtype=torch.float)
        zero_points = torch.tensor([0.1, 0.2], dtype=torch.float)
        
        # 对具有非零轴的张量进行按通道量化，使用4位量化方案
        qr = torch.quantize_per_channel(r, scales, zero_points, axis=1, dtype=torch.quint4x2)
        
        # 使用参考函数计算量化后的参考结果
        ref_res = _quantize_per_channel_sub_byte_ref(r, scales, zero_points, 1, 4)
        
        # 断言量化后的结果和参考结果在数值上的接近程度
        self.assertTrue(np.allclose(qr.int_repr(), ref_res))
    # 定义一个测试量化张量置换的测试函数
    def test_qtensor_permute(self):
        # 定义量化的缩放因子和零点
        scale = 0.02
        zero_point = 1
        # 遍历支持的设备类型
        for device in get_supported_device_types():
            # 创建一个形状为 (10, 30, 2, 2) 的随机张量 r，数据类型为 torch.float
            r = torch.rand(10, 30, 2, 2, device=device, dtype=torch.float) * 4 - 2
            # 遍历需要测试的量化数据类型
            for dtype in [torch.qint8, torch.quint8, torch.qint32]:
                # 对张量 r 进行量化
                qr = torch.quantize_per_tensor(r, scale, zero_point, dtype=dtype)
                # 将量化后的张量 qr 进行转置操作
                qr = qr.transpose(0, 1)
                # 对转置后的张量进行反量化操作
                rqr = qr.dequantize()
                # 断言转置后的反量化结果与原始转置结果在指定误差范围内相等
                self.assertTrue(np.allclose(r.cpu().numpy().transpose([1, 0, 2, 3]), rqr.cpu().numpy(), atol=2 / scale))

                # 重新量化张量 r 并进行维度置换
                qr = torch.quantize_per_tensor(r, scale, zero_point, dtype=dtype)
                qr1 = qr.permute([1, 0, 2, 3])
                qr2 = qr.transpose(0, 1)
                # 断言两种转换后的整数表示相等
                self.assertEqual(qr1.int_repr(), qr2.int_repr())
                # 断言两种转换后的量化缩放因子相等
                self.assertEqual(qr1.q_scale(), qr2.q_scale())
                # 断言两种转换后的量化零点相等
                self.assertEqual(qr1.q_zero_point(), qr2.q_zero_point())
                # 比较置换后的反量化结果与原始转置结果是否相等
                self.assertTrue(np.allclose(qr2.dequantize().cpu().numpy(),
                                            r.cpu().numpy().transpose([1, 0, 2, 3]), atol=2 / scale))
                # 确保置换后的结果是连续的
                self.assertEqual(qr2.contiguous().int_repr(), qr2.int_repr())

                # 更改内存格式为通道优先
                qlast = qr.contiguous(memory_format=torch.channels_last)
                # 断言原始张量和通道优先后张量的步长排序是否相反
                self.assertEqual(qr.stride(), sorted(qr.stride(), reverse=True))
                self.assertNotEqual(qlast.stride(), sorted(qlast.stride(), reverse=True))
                # 断言两种内存格式下的整数表示相等
                self.assertEqual(qr.int_repr(), qlast.int_repr())
                # 断言两种内存格式下的量化缩放因子相等
                self.assertEqual(qr.q_scale(), qlast.q_scale())
                # 断言两种内存格式下的量化零点相等
                self.assertEqual(qr.q_zero_point(), qlast.q_zero_point())
                # 比较通道优先内存格式下的反量化结果与原始量化结果的反量化结果是否相等
                self.assertEqual(qlast.dequantize(), qr.dequantize())

                # 测试置换较大的张量
                x = torch.randn(64, 64, device=device)
                qx = torch.quantize_per_tensor(x, 1.0, 0, dtype)
                # 确保置换操作能够正常执行
                qx.permute([1, 0])
    # 定义一个测试函数，用于测试按通道量化张量的轴置换操作
    def test_qtensor_per_channel_permute(self):
        # 遍历所有支持的设备类型
        for device in get_supported_device_types():
            # 生成一个形状为 (20, 10, 2, 2) 的随机张量 r，数据类型为 float，范围在 [-2, 2) 之间
            r = torch.rand(20, 10, 2, 2, dtype=torch.float, device=device) * 4 - 2
            # 设置量化后的数据类型为 qint8
            dtype = torch.qint8
            # 随机生成长度为 10 的张量 scales，范围在 [0.01, 0.03) 之间
            scales = torch.rand(10, device=device) * 0.02 + 0.01
            # 随机生成长度为 10 的 zero_points 张量，范围在 [-1, 1) 之间并四舍五入为 long 类型
            zero_points = torch.round(torch.rand(10, device=device) * 2 - 1).to(torch.long)
            # 对张量 r 进行按通道量化，使用 scales 和 zero_points，通道数为 1，数据类型为 dtype
            qr = torch.quantize_per_channel(r, scales, zero_points, 1, dtype)

            # 尝试对 qr 执行轴置换，预期抛出 RuntimeError 异常
            with self.assertRaises(RuntimeError):
                qr.transpose(0, 1)

            # 将 qr 转换为内存格式为 channels_last 的连续张量 qlast
            qlast = qr.contiguous(memory_format=torch.channels_last)
            # 断言 qr 的步幅（stride）是逆序排列后的结果
            self.assertEqual(qr.stride(), sorted(qr.stride(), reverse=True))
            # 断言 qlast 的步幅不是逆序排列后的结果
            self.assertNotEqual(qlast.stride(), sorted(qlast.stride(), reverse=True))
            # 断言 qr 的整数表示等于 qlast 的整数表示
            self.assertEqual(qr.int_repr(), qlast.int_repr())
            # 断言 scales 转换为 float64 后与 qlast 的量化比例张量相等
            self.assertEqual(scales.to(dtype=torch.float64), qlast.q_per_channel_scales())
            # 断言 zero_points 与 qlast 的量化零点张量相等
            self.assertEqual(zero_points, qlast.q_per_channel_zero_points())
            # 断言 qlast 的量化通道轴为 1
            self.assertEqual(1, qlast.q_per_channel_axis())
            # 断言 qlast 的反量化结果与 qr 的反量化结果相等
            self.assertEqual(qlast.dequantize(), qr.dequantize())

    # 定义一个测试函数，用于测试按张量量化的加载和保存操作
    def test_qtensor_load_save(self):
        # 设置量化比例为 0.2 和量化零点为 10
        scale = 0.2
        zero_point = 10
        # 指定设备为 "cpu"，因为 CUDA 当前不支持存储操作
        device = "cpu"
        # 生成形状为 (15, 2) 的随机张量 r，数据类型为 float32，范围在 [0, 2) 之间
        r = torch.rand(15, 2, dtype=torch.float32, device=device) * 2
        # 遍历三种数据类型：qint8, quint8, qint32
        for dtype in [torch.qint8, torch.quint8, torch.qint32]:
            # 对张量 r 进行按张量量化，使用 scale 和 zero_point，数据类型为 dtype
            qr = torch.quantize_per_tensor(r, scale, zero_point, dtype=dtype)
            # 提取 qr 的第二列 qrv 作为新的量化张量
            qrv = qr[:, 1]
            # 使用临时文件来保存和加载张量的序列化操作
            with tempfile.NamedTemporaryFile() as f:
                # 将 qr 和 qrv 序列化保存到文件 f 中
                torch.save((qr, qrv), f)
                # 遍历两种 weights_only 标志：True 和 False
                for weights_only in [True, False]:
                    # 将文件指针定位到文件开头
                    f.seek(0)
                    # 从文件 f 中加载 qr2 和 qrv2
                    qr2, qrv2 = torch.load(f, weights_only=weights_only)
                    # 断言 qr 和 qr2 相等
                    self.assertEqual(qr, qr2)
                    # 断言 qrv 和 qrv2 相等
                    self.assertEqual(qrv, qrv2)
                    # 断言 qr2 的存储数据指针与 qrv2 的存储数据指针相等
                    self.assertEqual(qr2.storage().data_ptr(), qrv2.storage().data_ptr())

    # 定义一个测试函数，用于测试按通道量化的加载和保存操作
    def test_qtensor_per_channel_load_save(self):
        # 生成形状为 (20, 10) 的随机张量 r，数据类型为 float，范围在 [-2, 2) 之间
        r = torch.rand(20, 10, dtype=torch.float) * 4 - 2
        # 随机生成长度为 10 的张量 scales，范围在 [0.01, 0.03) 之间
        scales = torch.rand(10, dtype=torch.double) * 0.02 + 0.01
        # 随机生成长度为 10 的 zero_points 张量，范围在 [1, 21) 之间并转换为 long 类型
        zero_points = torch.round(torch.rand(10) * 20 + 1).to(torch.long)
        # 遍历三种数据类型：quint8, qint8, quint4x2
        for dtype in [torch.quint8, torch.qint8, torch.quint4x2]:
            # 如果数据类型为 quint4x2，则将 zero_points 设置为全为 1 的张量
            if dtype == torch.quint4x2:
                zero_points = torch.ones(10, dtype=torch.float)
            # 对张量 r 进行按通道量化，使用 scales 和 zero_points，通道数为 1，数据类型为 dtype
            qr = torch.quantize_per_channel(r, scales, zero_points, 1, dtype)
            # 使用临时文件来保存和加载张量的序列化操作
            with tempfile.NamedTemporaryFile() as f:
                # 将 qr 序列化保存到文件 f 中
                torch.save(qr, f)
                # 遍历两种 weights_only 标志：True 和 False
                for weights_only in [True, False]:
                    # 将文件指针定位到文件开头
                    f.seek(0)
                    # 从文件 f 中加载 qr2
                    qr2 = torch.load(f, weights_only=weights_only)
                    # 断言 qr 和 qr2 相等
                    self.assertEqual(qr, qr2)
    # 定义测试函数 test_qtensor_copy，用于测试量化张量的复制操作
    def test_qtensor_copy(self):
        # 设置量化参数的初始值
        scale = 0.5
        zero_point = 10
        numel = 10
        # 迭代不同的量化数据类型
        for dtype in [torch.qint8, torch.quint8, torch.qint32]:
            # 迭代支持的设备类型
            for device in get_supported_device_types():
                # 创建具有相同scale和zero_point的量化张量q和q2，并进行复制操作
                q = torch._empty_affine_quantized([numel], scale=scale,
                                                  zero_point=zero_point, device=device, dtype=dtype)
                q2 = torch._empty_affine_quantized([numel], scale=scale,
                                                   zero_point=zero_point, device=device, dtype=dtype)
                q.copy_(q2)
                # 断言量化张量q和q2的整数表示相等
                self.assertEqual(q.int_repr(), q2.int_repr())
                # 断言量化张量q和q2的量化参数scale相等
                self.assertEqual(q.q_scale(), q2.q_scale())
                # 断言量化张量q和q2的量化参数zero_point相等
                self.assertEqual(q.q_zero_point(), q2.q_zero_point())
                # 创建具有不同scale和zero_point的量化张量q，并进行复制操作
                new_scale = 3.2
                new_zero_point = 5
                q = torch._empty_affine_quantized([numel], scale=new_scale,
                                                  zero_point=new_zero_point, device=device, dtype=dtype)
                # 断言设置的量化参数new_scale和new_zero_point已正确应用到量化张量q中
                self.assertEqual(q.q_scale(), new_scale)
                self.assertEqual(q.q_zero_point(), new_zero_point)
                # 再次进行复制操作，检查量化张量的scale和zero_point是否被正确复制
                q.copy_(q2)
                self.assertEqual(q, q2)
                # 不能从量化张量复制到非量化张量，验证这种情况会抛出RuntimeError
                r = torch.empty([numel], dtype=torch.float)
                q = torch._empty_affine_quantized([numel], scale=scale, zero_point=zero_point, dtype=dtype)
                with self.assertRaisesRegex(RuntimeError, "please use dequantize"):
                    r.copy_(q)
            # 对于不支持cuda的情况，设备类型设为'cpu'
            device = 'cpu'
            # 创建非量化张量r和量化张量q，并进行复制操作
            r = torch.randn([numel], dtype=torch.float, device=device)
            q = torch._empty_affine_quantized([numel], scale=scale, zero_point=zero_point, dtype=dtype, device=device)
            q.copy_(r)
            # 将非量化张量r量化为qr，再与量化张量q比较是否相等
            qr = torch.quantize_per_tensor(r, scale=scale, zero_point=zero_point, dtype=dtype)
            self.assertEqual(q, qr)

    # 定义测试函数 test_torch_qtensor_deepcopy，用于测试深度复制量化张量的操作
    def test_torch_qtensor_deepcopy(self):
        # 暂时不支持cuda，设备类型设为"cpu"
        device = "cpu"
        # 创建具有随机整数值的量化张量q_int
        q_int = torch.randint(0, 100, [3, 5], device=device, dtype=torch.uint8)
        scale, zero_point = 2.0, 3
        # 使用指定的scale和zero_point创建量化张量q
        q = torch._make_per_tensor_quantized_tensor(q_int, scale=scale, zero_point=zero_point)
        # 对量化张量q进行深度复制操作，得到qc
        qc = deepcopy(q)
        # 断言深度复制后的qc与原始量化张量q相等
        self.assertEqual(qc, q)
    # 定义一个测试方法 test_clone，用于测试张量的克隆功能
    def test_clone(self):
        numel = 10  # 张量元素个数
        scale = 0.5  # 量化比例因子
        zero_point = 10  # 量化零点

        # 生成所有可能的设备类型和数据类型的组合
        options = itertools.product(
            get_supported_device_types(),
            [torch.qint8, torch.quint8, torch.qint32])

        # 遍历每一种设备和数据类型组合
        for device, dtype in options:
            # 创建一个元素个数为 numel 的量化张量，使用张量级别的量化
            per_tensor_quantized = torch._empty_affine_quantized(
                [numel], scale=scale, zero_point=zero_point,
                device=device, dtype=dtype)
            
            # 创建一个元素个数为 numel 的通道级别的量化张量
            per_channel_quantized = torch._empty_per_channel_affine_quantized(
                [numel],
                scales=torch.tensor([scale] * numel, device=device),
                zero_points=torch.tensor([zero_point] * numel, device=device),
                axis=0,
                device=device,
                dtype=dtype
            )
            
            # 将两种量化张量存入列表 qtensors
            qtensors = [per_tensor_quantized, per_channel_quantized]

            # 遍历 qtensors 中的每一个量化张量
            for q in qtensors:
                # 克隆当前量化张量 q，生成新的量化张量 q2
                q2 = q.clone()
                
                # 断言新克隆的量化张量 q2 与原量化张量 q 在内容上的一致性
                # 检查量化比例因子和量化零点是否已被复制
                self.assertEqual(q, q2)

    # 定义一个测试方法 test_qtensor_fill_per_tensor，测试张量按元素填充功能
    def test_qtensor_fill_per_tensor(self):
        numel = 10  # 张量元素个数
        scale = 0.5  # 量化比例因子
        zero_point = 10  # 量化零点

        ones = torch.ones(numel).to(torch.float)  # 创建元素全为 1 的张量，并转换为浮点类型

        qtypes = [torch.qint8, torch.quint8, torch.qint32]  # 量化类型的列表
        vals2fill = [-1, 1, 2**32]  # 填充值列表，包括正数、负数、溢出值

        devices = get_supported_device_types()  # 获取支持的设备类型列表
        # 遍历量化类型、填充值、设备类型的所有组合
        for qtype, val2fill, device in itertools.product(qtypes, vals2fill, devices):
            ones = ones.to(device)  # 将 ones 张量移动到当前设备

            # 创建一个元素个数为 numel 的量化张量，并按 val2fill 填充
            q_filled = torch._empty_affine_quantized(
                [numel], scale=scale, zero_point=zero_point, device=device,
                dtype=qtype)
            q_filled.fill_(val2fill)  # 使用 val2fill 填充量化张量 q_filled

            # 创建一个参考张量 q_ref，用于与 q_filled 进行比较
            q_ref = torch.quantize_per_tensor(ones * val2fill, scale,
                                              zero_point, qtype)
            
            # 断言量化张量 q_filled 与参考张量 q_ref 在整数表示上的一致性
            self.assertEqual(q_filled.int_repr(), q_ref.int_repr())
            # 断言量化张量 q_filled 与参考张量 q_ref 在反量化后的值上的一致性
            self.assertEqual(q_filled.dequantize(), q_ref.dequantize())
            # 确保量化比例因子和量化零点没有改变
            self.assertEqual(q_filled.q_scale(), scale)
            self.assertEqual(q_filled.q_zero_point(), zero_point)

            # 适用于 NHWC 张量的测试方法，基于 test_qtensor_fill_per_tensor 进行调整
    # 定义测试函数，用于测试按张量填充的量化张量，使用 NHWC 内存布局
    def test_qtensor_fill_per_tensor_nhwc(self):
        # 随机生成长度为 4 的整数列表，作为张量的维度
        dims = torch.randint(low=1, high=10, size=(4, )).tolist()
        scale = 0.5  # 设置量化比例
        zero_point = 10  # 设置量化零点

        ones = torch.ones(dims).to(torch.float)  # 创建一个全为1的张量

        qtypes = [torch.qint8, torch.quint8, torch.qint32]  # 量化类型的列表
        vals2fill = [-1, 1, 2**32]  # 需要填充的值：负数、正数、溢出值
        memory_formats = [torch.contiguous_format, torch.channels_last]  # 内存布局格式列表
        devices = get_supported_device_types()  # 获取支持的设备类型列表
        # 对每一种量化类型、填充值、内存布局格式和设备类型进行组合测试
        for qtype, val2fill, memory_format, device in itertools.product(qtypes, vals2fill, memory_formats, devices):
            # 创建一个空的仿射量化张量，按照指定参数
            q_filled = torch._empty_affine_quantized(
                dims, scale=scale, zero_point=zero_point, device=device,
                dtype=qtype, memory_format=memory_format)
            # 使用指定值填充量化张量
            q_filled.fill_(val2fill)
            # 创建一个参考张量，用于与填充后的量化张量比较
            q_ref = torch.quantize_per_tensor(ones * val2fill, scale,
                                              zero_point, qtype)
            # 断言填充后的量化张量的整数表示与参考张量相同
            self.assertEqual(q_filled.int_repr(), q_ref.int_repr())
            # 断言填充后的量化张量的反量化结果与参考张量相同
            self.assertEqual(q_filled.dequantize(), q_ref.dequantize())
            # 断言量化比例和零点不变
            self.assertEqual(q_filled.q_scale(), scale)
            self.assertEqual(q_filled.q_zero_point(), zero_point)

    # 从 test_qtensor_fill_per_tensor 适配而来，测试按通道填充的量化张量
    def test_qtensor_fill_per_channel(self):
        dims = [4, 5]  # 张量的维度
        axis = 0  # 指定按通道量化的轴
        # 随机生成每个通道的量化比例，加上一个常数以避免过小的比例
        scales = torch.rand(dims[axis], dtype=torch.float64) + 0.1
        zero_points = torch.randint(low=0, high=10, size=(dims[axis], ))  # 随机生成每个通道的量化零点

        ones = torch.ones(dims).to(torch.float)  # 创建一个全为1的张量

        qtypes = [torch.qint8, torch.quint8, torch.qint32]  # 量化类型的列表
        vals2fill = [-1, 1, 2**32]  # 需要填充的值：负数、正数、溢出值
        devices = get_supported_device_types()  # 获取支持的设备类型列表
        # 对每一种量化类型、填充值和设备类型进行组合测试
        for qtype, val2fill, device in itertools.product(qtypes, vals2fill, devices):
            scales = scales.to(device)  # 将量化比例移到指定设备上
            zero_points = zero_points.to(device)  # 将量化零点移到指定设备上
            ones = ones.to(device)  # 将全为1的张量移到指定设备上
            # 创建一个空的按通道仿射量化张量，按照指定参数
            q_filled = torch._empty_per_channel_affine_quantized(
                dims, scales=scales, zero_points=zero_points, device=device,
                axis=axis, dtype=qtype)
            # 使用指定值填充量化张量
            q_filled.fill_(val2fill)
            # 创建一个参考张量，用于与填充后的量化张量比较
            q_ref = torch.quantize_per_channel(ones * val2fill, scales=scales,
                                               zero_points=zero_points, axis=axis, dtype=qtype)
            # 断言填充后的量化张量的整数表示与参考张量相同
            self.assertEqual(q_filled.int_repr(), q_ref.int_repr())
            # 断言填充后的量化张量的反量化结果与参考张量相同
            self.assertEqual(q_filled.dequantize(), q_ref.dequantize())
            # 断言每个通道的量化比例和零点不变
            self.assertEqual(q_filled.q_per_channel_scales(), scales)
            self.assertEqual(q_filled.q_per_channel_zero_points(), zero_points)

    # 测试按 CPU 进行的量化张量掩码填充操作
    def test_qtensor_masked_fill_cpu(self):
        self._test_qtensor_masked_fill('cpu')
    # 如果未启用 CUDA 测试，则跳过该测试函数
    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def test_qtensor_masked_fill_cuda(self):
        # 使用 CUDA 测试 _test_qtensor_masked_fill 函数
        self._test_qtensor_masked_fill('cuda')

    # 从 test_qtensor_fill_per_tensor 适配而来
    def _test_qtensor_masked_fill(self, device):
        # 定义张量的元素数量、缩放因子和零点
        numel = 10
        scale = 0.5
        zero_point = 10

        # 创建一个由 torch.ones 构成的张量，数据类型为 float，设备为指定的 device
        ones = torch.ones(numel, dtype=torch.float, device=device)

        # 定义量化类型列表和填充值列表
        types = [torch.qint8, torch.quint8, torch.qint32]
        fills = [-1, 1, 2**32]  # 正数、负数、溢出值

        # 遍历量化类型和填充值的笛卡尔积
        for qtype, fill_with in itertools.product(types, fills):
            # 创建一个空的仿射量化张量，指定大小、缩放因子、零点、设备和数据类型
            q_filled = torch._empty_affine_quantized(
                [numel], scale=scale, zero_point=zero_point, device=device,
                dtype=qtype)
            # 使用 fill_ 方法填充量化张量
            q_filled.fill_(fill_with)

            # 创建一个空的仿射量化张量，与 q_filled 具有相同的参数
            q_masked_fill = torch._empty_affine_quantized(
                [numel], scale=scale, zero_point=zero_point, device=device,
                dtype=qtype)
            # 使用 masked_fill_ 方法填充量化张量，相当于普通填充操作
            mask = torch.tensor(True, device=device)
            q_masked_fill.masked_fill_(mask, fill_with)

            # 将 ones 乘以 fill_with 进行量化，并获取其整数表示
            int_repr = torch.quantize_per_tensor(ones * fill_with, scale,
                                                 zero_point, qtype)
            fill_with = int_repr.dequantize()
            int_repr = int_repr.int_repr()

            # 断言填充后的量化张量与预期相等
            self.assertEqual(q_filled, q_masked_fill)
            self.assertEqual(q_masked_fill.int_repr(), int_repr)
            self.assertEqual(q_masked_fill.dequantize(), fill_with)
            # 确保缩放因子和零点未发生变化
            self.assertEqual(q_masked_fill.q_scale(), scale)
            self.assertEqual(q_masked_fill.q_zero_point(), zero_point)

        # 上述循环执行与 test_qtensor_fill 相同的测试
        # 现在我们将检查索引放置的子集的 masked_fill
        mask = torch.randint(0, 2, (numel, ), device=device)
        mask = mask.bool()
        x = torch.rand(numel, device=device)
        qx = torch.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=qtype)
        
        # 再次遍历量化类型和填充值的笛卡尔积
        for qtype, fill_with in itertools.product(types, fills):
            # 克隆 qx 张量
            q_masked_fill = qx.clone()
            # 使用 masked_fill_ 方法填充量化张量的子集
            q_masked_fill.masked_fill_(mask, fill_with)
            ref = qx.clone()

            # 手动填充 ref 张量的子集，以便与 masked_fill 的实现进行比较
            for i in range(numel):
                if mask[i]:
                    ref[i] = torch.tensor([fill_with], device=device, dtype=torch.float)

            # 断言 masked_fill 的结果与手动填充的结果相等
            self.assertEqual(q_masked_fill, ref)
            self.assertEqual(q_masked_fill.int_repr(), ref.int_repr())
            self.assertEqual(q_masked_fill.dequantize(), ref.dequantize())

    # 在 CPU 上测试 qtensor_index_put
    def test_qtensor_index_put_cpu(self):
        self._test_qtensor_index_put('cpu')
        self._test_qtensor_index_put_non_accumulate_deterministic('cpu')

    # 如果未启用 CUDA 测试，则跳过该测试函数
    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    # 测试函数，用于在 CUDA 设备上执行 _test_qtensor_index_put 和 _test_qtensor_index_put_non_accumulate_deterministic
    def test_qtensor_index_put_cuda(self):
        self._test_qtensor_index_put('cuda')  # 调用 _test_qtensor_index_put 方法，传递 'cuda' 参数
        self._test_qtensor_index_put_non_accumulate_deterministic('cuda')  # 调用 _test_qtensor_index_put_non_accumulate_deterministic 方法，传递 'cuda' 参数

    # 核心测试函数，用于测试量化张量的索引放置功能
    def _test_qtensor_index_put(self, device):
        n = 10
        m = 10
        x_orig = torch.rand(n, m, device=device)  # 在指定设备上生成随机张量 x_orig
        indices = tuple(torch.tensor([[0, 0], [1, 1], [5, 5], [7, 3], [0, 5], [6, 9], [-1, -1]], device=device).t())  # 生成索引元组，指定设备
        # 对于标量张量情况，index_put 路由到 masked_fill
        values_list = [torch.tensor(2.5, device=device), torch.rand(len(indices[0]), device=device) * 1000]  # 在指定设备上生成值列表
        scale = 0.5  # 设置量化的尺度
        zero_point = 10  # 设置量化的零点
        types = [torch.qint8, torch.quint8, torch.qint32]  # 量化类型列表
        # 遍历量化类型和值列表的笛卡尔积
        for qtype, values in itertools.product(types, values_list):
            x_ref = x_orig.clone()  # 克隆原始张量 x_orig
            x_ref[indices] = values.to(dtype=x_ref.dtype)  # 在指定索引处使用指定类型的值进行替换
            qx_ref = torch.quantize_per_tensor(x_ref, scale=scale, zero_point=zero_point, dtype=qtype)  # 对替换后的张量进行量化

            x = x_orig.clone()  # 克隆原始张量 x_orig
            qx = torch.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=qtype)  # 对克隆的张量进行量化
            qx[indices] = values  # 在指定索引处放置值

            self.assertEqual(qx_ref, qx)  # 断言量化结果与参考量化结果相等

    # 非累积确定性测试函数，用于在指定设备上测试量化张量的索引放置功能
    def _test_qtensor_index_put_non_accumulate_deterministic(self, device):
        with DeterministicGuard(True):  # 启用确定性模式保护
            scale = 0.5  # 设置量化的尺度
            zero_point = 10  # 设置量化的零点
            types = [torch.qint8, torch.quint8, torch.qint32]  # 量化类型列表
            # 遍历量化类型
            for qtype in types:
                # 遍历三次
                for i in range(3):
                    m = random.randint(10, 20)  # 随机生成 m，范围在 10 到 20 之间
                    elems = random.randint(20000, 30000)  # 随机生成 elems，范围在 20000 到 30000 之间
                    values = torch.rand(elems, device=device)  # 在指定设备上生成随机值张量
                    indices = torch.randint(m, (elems,), device=device)  # 在指定设备上生成随机索引张量，范围在 0 到 m-1 之间
                    x_orig = torch.rand(m, device=device)  # 在指定设备上生成 m 维随机张量

                    x = x_orig.clone()  # 克隆原始张量 x_orig
                    qx = torch.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=qtype)  # 对克隆的张量进行量化
                    output = qx.index_put((indices,), values, accumulate=False)  # 在指定索引处使用指定值进行放置，不累积

                    x_ref = x_orig.clone()  # 克隆原始张量 x_orig
                    output_ref = x_ref.index_put((indices,), values, accumulate=False)  # 在原始张量上进行相同的操作，不累积
                    qx_ref = torch.quantize_per_tensor(output_ref, scale=scale, zero_point=zero_point, dtype=qtype)  # 对结果张量进行量化

                    self.assertEqual(output, qx_ref)  # 断言输出结果与参考量化结果相等

    # 自 test_qtensor_fill_per_channel 和 test_qtensor_fill_per_tensor_nhwc 调整而来的
    # 定义测试函数，用于测试按通道填充的量化张量（NHWC格式）
    def test_qtensor_fill_per_channel_nhwc(self):
        # 随机生成一个四维张量的维度列表，每个维度的取值范围是[1, 10)
        dims = torch.randint(low=1, high=10, size=(4,)).tolist()
        # 指定操作的轴向为0
        axis = 0
        # 生成一个包含随机数的张量作为尺度，数据类型为torch.float64，并加上0.1以避免尺度过小
        scales = torch.rand(dims[axis], dtype=torch.float64) + 0.1
        # 生成一个随机整数张量作为零点，取值范围是[0, 10)
        zero_points = torch.randint(low=0, high=10, size=(dims[axis],))

        # 创建一个全为1的张量，数据类型转换为torch.float
        ones = torch.ones(dims).to(torch.float)

        # 定义量化类型的列表
        qtypes = [torch.qint8, torch.quint8, torch.qint32]
        # 定义要填充的值的列表，包括负数、正数和溢出情况
        vals2fill = [-1, 1, 2**32]
        # 定义内存格式的列表，包括连续格式和通道最后格式
        memory_formats = [torch.contiguous_format, torch.channels_last]
        # 获得支持的设备类型列表
        devices = get_supported_device_types()

        # 使用itertools的product函数，遍历所有量化类型、填充值、内存格式和设备类型的组合
        for qtype, val2fill, memory_format, device in itertools.product(qtypes, vals2fill, memory_formats, devices):
            # 将尺度、零点、全1张量分别移动到指定的设备上
            scales = scales.to(device)
            zero_points = zero_points.to(device)
            ones = ones.to(device)

            # 使用torch._empty_per_channel_affine_quantized函数创建按通道仿射量化的空张量
            q_filled = torch._empty_per_channel_affine_quantized(
                dims, scales=scales, zero_points=zero_points, device=device,
                axis=axis, dtype=qtype, memory_format=memory_format)
            
            # 使用给定的值填充量化张量q_filled
            q_filled.fill_(val2fill)
            
            # 创建参考张量q_ref，用于与填充后的量化张量q_filled进行比较
            q_ref = torch.quantize_per_channel(ones * val2fill, scales=scales,
                                               zero_points=zero_points, axis=axis, dtype=qtype)
            
            # 使用self.assertEqual断言检查填充后的量化张量q_filled与参考张量q_ref的整数表示是否相等
            self.assertEqual(q_filled.int_repr(), q_ref.int_repr())
            
            # 使用self.assertEqual断言检查填充后的量化张量q_filled与参考张量q_ref的反量化结果是否相等
            self.assertEqual(q_filled.dequantize(), q_ref.dequantize())
            
            # 使用self.assertEqual断言检查填充后的量化张量q_filled的通道尺度是否与预期的尺度scales相等
            self.assertEqual(q_filled.q_per_channel_scales(), scales)
            
            # 使用self.assertEqual断言检查填充后的量化张量q_filled的通道零点是否与预期的零点zero_points相等
            self.assertEqual(q_filled.q_per_channel_zero_points(), zero_points)

    # 当没有GPU可用时，跳过此测试
    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def test_qtensor_index_select_cuda(self):
        # 调用_test_qtensor_index_select函数，在CUDA设备上进行测试
        self._test_qtensor_index_select('cuda')

    # 在CPU上执行量化张量的索引选择测试
    def test_qtensor_index_select_cpu(self):
        # 调用_test_qtensor_index_select函数，在CPU设备上进行测试
        self._test_qtensor_index_select('cpu')

    # 辅助函数，用于测试量化张量的索引选择功能
    def _test_qtensor_index_select(self, device):
        # 遍历torch.quint8和torch.qint8两种量化类型
        for quant_type in [torch.quint8, torch.qint8]:
            # 指定维度为3
            dims = 3
            # 随机生成一个索引，范围是[0, 3)之间的整数
            index = torch.randint(dims, [1]).item()
            # 生成一个随机选择的索引列表，包含两个元素，选择的设备是device
            selected = torch.randperm(dims)[:2].to(device)
            # 设置尺度为1，零点为0
            scale = 1
            zp = 0
            # 生成一个在指定设备上的随机张量x，其形状为[3, 3, 3]
            x = torch.randn([3] * dims, device=device) * 10

            # 使用torch.index_select函数，根据指定的索引index和selected选出子集x_selected
            x_selected = torch.index_select(x, index, selected)
            # 对选出的子集x_selected进行量化，使用指定的尺度scale、零点zp和量化类型quant_type
            x_selected_quantized = torch.quantize_per_tensor(x_selected, scale, zp, quant_type)

            # 对张量x进行量化，使用指定的尺度scale、零点zp和量化类型quant_type
            x_quantized = torch.quantize_per_tensor(x, scale, zp, quant_type)
            # 对量化后的张量x_quantized，根据指定的索引index和selected选出子集x_quantized_selected
            x_quantized_selected = torch.index_select(x_quantized, index, selected)

            # 使用self.assertEqual断言检查量化后的子集x_quantized_selected与量化后的选出子集x_selected_quantized是否相等
            self.assertEqual(x_quantized_selected, x_selected_quantized)
    def test_qtensor_view(self):
        # 定义量化张量的缩放因子、零点和数据类型
        scale, zero_point, dtype = 1.0, 2, torch.uint8
        # 遍历支持的设备类型列表
        for device in get_supported_device_types():
            # 生成在指定设备上的随机整数张量，使用指定的数据类型和量化参数创建量化张量
            q_int = torch.randint(0, 100, [1, 2, 3], device=device, dtype=dtype)
            q = torch._make_per_tensor_quantized_tensor(q_int, scale=scale, zero_point=zero_point)
            # 对量化张量进行形状变换
            q2 = q.view(1, 3, 2)
            # 断言两个量化张量的元素个数相同
            self.assertEqual(q.numel(), q2.numel())
            # 测试使用-1的形状参数进行形状变换
            self.assertEqual(q, q2.view(1, -1, 3))

            # 生成更高维度的随机整数张量，使用相同的量化参数创建量化张量
            a_int = torch.randint(0, 100, [1, 2, 3, 4], device=device, dtype=dtype)
            a = torch._make_per_tensor_quantized_tensor(a_int, scale=scale, zero_point=zero_point)
            # 对张量进行维度交换操作
            b = a.transpose(1, 2)  # 交换第2和第3个维度
            c = a.view(1, 3, 2, 4)  # 在内存中不改变张量的布局
            # 断言b和c的形状相同
            self.assertEqual(b.size(), c.size())
            # 断言b和c的量化缩放因子相同
            self.assertEqual(b.q_scale(), c.q_scale())
            # 断言b和c的量化零点相同
            self.assertEqual(b.q_zero_point(), c.q_zero_point())
            # 断言b和c的步长不同
            self.assertNotEqual(b.stride(), c.stride())
            # 断言b和c的整数表示不同
            self.assertNotEqual(b.int_repr(), c.int_repr())
            # 对于CPU设备，断言b和c不相等（因为torch.equal在cuda后端不支持）
            if device == 'cpu':
                self.assertFalse(torch.equal(b, c))

            # 生成更高维度的随机整数张量，使用相同的量化参数创建量化张量
            a_int = torch.randint(0, 100, [1, 2, 3, 4], device=device, dtype=dtype)
            a = torch._make_per_tensor_quantized_tensor(a_int, scale=scale, zero_point=zero_point)
            # 对张量进行维度交换操作
            b = a.transpose(1, 2)  # 交换第2和第3个维度
            # 出错信息字符串
            err_str = "view size is not compatible with input tensor's size and stride*"
            # 断言在出错信息中捕获运行时错误异常
            with self.assertRaisesRegex(RuntimeError, err_str):
                b.view(1, 4, 2, 3)
            # 对连续张量进行形状变换操作
            b.contiguous().view(1, 4, 2, 3)
    # 定义一个测试函数，用于测试量化张量的 resize 操作
    def test_qtensor_resize(self):
        # 遍历支持的设备类型列表
        for device in get_supported_device_types():
            # 设置量化张量的缩放因子、零点和数据类型
            scale, zero_point, dtype = 1.0, 2, torch.uint8
            # 定义不同大小的张量尺寸列表
            sizes1 = [1, 2, 3, 4]
            sizes2 = [1 * 2, 3 * 4]
            sizes3 = [1, 2 * 3, 4]
            sizes4 = [1 * 2 * 3 * 4]
            sizes5 = [1, 2, 1, 3, 1, 4]

            # 创建随机整数张量并进行量化处理
            q1_int = torch.randint(0, 100, sizes1, dtype=dtype, device=device)
            q1 = torch._make_per_tensor_quantized_tensor(q1_int, scale=scale, zero_point=zero_point)
            
            # 调整张量尺寸并生成新的量化张量
            q2 = q1.resize(*sizes2)
            q3 = q2.resize(*sizes3)
            q4 = q3.resize(*sizes4)
            q5 = q4.resize(*sizes5)

            # 断言各个量化张量的元素数量相同
            self.assertEqual(q1.numel(), q2.numel())
            self.assertEqual(q1.numel(), q3.numel())
            self.assertEqual(q1.numel(), q4.numel())
            self.assertEqual(q1.numel(), q5.numel())

            # 创建另一个随机整数张量并进行量化处理
            a_int = torch.randint(0, 100, sizes1, dtype=dtype, device=device)
            a = torch._make_per_tensor_quantized_tensor(a_int, scale=scale, zero_point=zero_point)
            
            # 转置张量 b，交换第二和第三维度
            b = a.transpose(1, 2)
            # 将张量 b 调整尺寸为原始尺寸
            c = b.resize(*sizes1)

            # 断言调整尺寸后张量 c 的属性与张量 b 相同
            self.assertEqual(a.size(), c.size())
            self.assertEqual(b.q_scale(), c.q_scale())
            self.assertEqual(b.q_zero_point(), c.q_zero_point())
            self.assertNotEqual(b.stride(), c.stride())
            # 断言张量 b 和 c 的整数表示不同（底层数据不同）
            self.assertNotEqual(b.int_repr(), c.int_repr())
            # 如果设备为 CPU，则断言张量 b 和 c 不相等（torch.equal 在 CUDA 后端不支持）
            if device == 'cpu':
                self.assertFalse(torch.equal(b, c))

            # 抛出异常，如果张量尺寸调整错误
            q1_int = torch.randint(0, 100, sizes1, dtype=dtype, device=device)
            q1 = torch._make_per_tensor_quantized_tensor(a_int, scale=scale, zero_point=zero_point)
            err_str = "requested resize to*"
            with self.assertRaisesRegex(RuntimeError, err_str):
                q2 = q1.resize(*sizes1[:-1])
            
            # 调整连续和非连续张量尺寸应该正常
            q3 = q1.resize(*sizes2)
            q4 = q1.contiguous().resize(*sizes2)
    # 定义一个测试方法，用于测试量化张量的 reshape 功能
    def test_qtensor_reshape(self):
        # 定义量化张量的缩放因子、零点和数据类型
        scale, zero_point, dtype = 1.0, 2, torch.uint8
        # 遍历所有支持的设备类型
        for device in get_supported_device_types():
            # 创建一个随机整数张量，并转换为量化张量 q_int，指定缩放因子和零点
            q_int = torch.randint(0, 100, [3, 5], dtype=dtype, device=device)
            q = torch._make_per_tensor_quantized_tensor(q_int, scale=scale, zero_point=zero_point)
            # 对量化张量进行 reshape 操作，变成一维张量 q2
            q2 = q.reshape([15])
            # 断言量化张量 q 和 q2 的元素数量相等
            self.assertEqual(q.numel(), q2.numel())
            # 断言 q2 的尺寸为 [15]
            self.assertEqual(q2.size(), [15])
            # 测试使用 -1 的 reshape 操作
            self.assertEqual(q, q2.reshape([3, -1]))

            # 创建一个更高维度的随机整数张量 a_int，并转换为量化张量 a
            a_int = torch.randint(0, 100, [1, 2, 3, 4], dtype=dtype, device=device)
            a = torch._make_per_tensor_quantized_tensor(a_int, scale=scale, zero_point=zero_point)
            # 对 a 进行维度转置，得到 b，交换第二和第三维度
            b = a.transpose(1, 2)
            # 对 a 进行 reshape 操作，保持张量布局不变，得到 c
            c = a.reshape(1, 3, 2, 4)
            # 断言 b 和 c 的尺寸相等
            self.assertEqual(b.size(), c.size())
            # 断言 b 和 c 的缩放因子相同
            self.assertEqual(b.q_scale(), c.q_scale())
            # 断言 b 和 c 的零点相同
            self.assertEqual(b.q_zero_point(), c.q_zero_point())
            # 断言 b 和 c 的步长不同
            self.assertNotEqual(b.stride(), c.stride())
            # 断言 b 和 c 的整数表示不同
            self.assertNotEqual(b.int_repr(), c.int_repr())
            # 对于 CPU 设备，断言 b 和 c 不相等（CUDA 不支持 torch.equal）
            if device == 'cpu':
                self.assertFalse(torch.equal(b, c))

            # 对非连续的张量 a 进行 reshape 操作，得到 c
            a_int = torch.randint(0, 100, [1, 2, 3, 4], dtype=dtype, device=device)
            a = torch._make_per_tensor_quantized_tensor(a_int, scale=scale, zero_point=zero_point)
            b = a.transpose(1, 2)  # 交换第二和第三维度
            c = b.reshape(1, 4, 2, 3)
    # 定义测试函数 test_qtensor_unsqueeze，用于测试量化张量的 unsqueeze 操作
    def test_qtensor_unsqueeze(self):
        # 遍历支持的设备类型列表
        for device in get_supported_device_types():
            # 创建形状为 (1, 3, 4) 的随机张量 x，并将其量化为 qx
            x = torch.randn((1, 3, 4), device=device)
            qx = torch.quantize_per_tensor(x, scale=1.0, zero_point=0, dtype=torch.quint8)
            # 对 qx 进行 unsqueeze 操作，在第2维插入维度
            qy = qx.unsqueeze(2)
            # 断言 qy 的形状为 (1, 3, 1, 4)
            self.assertEqual(qy.size(), (1, 3, 1, 4))
            # 对 qy 进行 squeeze 操作，消除第2维
            qy = qy.squeeze(2)
            # 断言 squeeze 后 qy 的形状与 qx 相同
            self.assertEqual(qy.size(), qx.size())

            # 创建形状为 (1, 3, 4) 的随机张量 x，并使用 per channel 的量化方式量化为 qx
            scales = torch.tensor([1.0], device=device)
            zero_points = torch.tensor([0], device=device)
            qx = torch.quantize_per_channel(x, scales=scales, zero_points=zero_points, dtype=torch.quint8, axis=0)
            # 对 qx 进行 unsqueeze 操作，在第0维插入维度
            qy = qx.unsqueeze(0)
            # 断言 qy 的形状为 (1, 1, 3, 4)
            self.assertEqual(qy.size(), (1, 1, 3, 4))
            # 断言 qy 的 per channel 轴为 1
            self.assertEqual(qy.q_per_channel_axis(), 1)

            # 对 qy 进行 squeeze 操作，消除第0维
            qz = qy.squeeze(0)
            # 断言 squeeze 后 qz 的形状与 x 相同
            self.assertEqual(qz.size(), x.size())
            # 断言 qz 的 per channel 轴为 0
            self.assertEqual(qz.q_per_channel_axis(), 0)
            # 使用断言检查尝试在 per-channel 数据上对 qy 进行 squeeze 操作时是否引发 RuntimeError 异常
            with self.assertRaisesRegex(RuntimeError, "Squeeze is only possible on non-axis dimension for Per-Channel"):
                qz = qy.squeeze(1)

            # 创建形状为 (3, 1, 2, 1, 4) 的随机张量 x，并使用 per channel 的量化方式量化为 qx
            scales = torch.tensor([1.0, 1.0], device=device)
            zero_points = torch.tensor([0, 0], device=device)
            qx = torch.quantize_per_channel(x, scales=scales, zero_points=zero_points, dtype=torch.quint8, axis=2)
            # 对 qx 进行 squeeze 操作，消除所有维度中的尺寸为 1 的维度
            qz = qx.squeeze()
            # 断言 squeeze 后 qz 的形状为 (3, 2, 4)
            self.assertEqual(qz.size(), (3, 2, 4))
            # 断言 qz 的 per channel 轴为 1
            self.assertEqual(qz.q_per_channel_axis(), 1)
            # 使用断言检查尝试在 per-channel 数据上对 qy 进行 squeeze 操作时是否引发 RuntimeError 异常
            with self.assertRaisesRegex(RuntimeError, "Squeeze is only possible on non-axis dimension for Per-Channel"):
                qz = qy.squeeze()

    # 定义测试函数 test_repeat，用于测试张量的 repeat 操作
    def test_repeat(self):
        scale, zero_point, dtype = 1.0, 2, torch.uint8
        # 遍历支持的设备类型列表
        for device in get_supported_device_types():
            # 创建形状为 [3] 的随机整数张量 q_int，并指定 dtype 和 device
            q_int = torch.randint(0, 100, [3], dtype=dtype, device=device)
            # 使用 repeat 操作对 q_int 进行重复 [4, 2] 次
            q_int_repeat = q_int.repeat(4, 2)
            # 创建用于比较的参考量化张量 q_ref
            q_ref = torch._make_per_tensor_quantized_tensor(q_int_repeat, scale=scale, zero_point=zero_point)

            # 使用 _make_per_tensor_quantized_tensor 创建量化张量 q，并使用 repeat 操作对其重复 [4, 2] 次
            q = torch._make_per_tensor_quantized_tensor(q_int, scale=scale, zero_point=zero_point)
            q_repeat = q.repeat(4, 2)
            # 断言重复后的 q_repeat 与参考张量 q_ref 相等
            self.assertEqual(q_ref, q_repeat)

    # 定义测试函数 test_qscheme_pickle，用于测试量化方案的 pickle 操作
    def test_qscheme_pickle(self):
        # 创建 Foo 类的实例 f
        f = Foo()
        buf = io.BytesIO()
        # 将 f 序列化保存到 buf 中
        torch.save(f, buf)

        buf.seek(0)
        # 从 buf 中加载保存的模型到 f2，不进行 weights_only 测试，因为这是加载 Module（传统方式）
        f2 = torch.load(buf)

        # 断言 f2 的量化方案为 torch.per_tensor_symmetric
        self.assertEqual(f2.qscheme, torch.per_tensor_symmetric)
    # 测试函数：选择量化参数
    def test_choose_qparams(self, X, reduce_range):
        # 从输入元组 X 中解包得到数据 X，以及 scale、zero_point 和 torch_type
        X, (scale, zero_point, torch_type) = X
        # 将 X 转换为 Torch 的张量
        X = torch.from_numpy(X)
        # 计算动态量化参数 X_scale 和 X_zp
        X_scale, X_zp = _calculate_dynamic_qparams(X, torch.quint8, reduce_range=reduce_range)
        # 选择张量 X 的量化参数 qparams
        qparams = torch._choose_qparams_per_tensor(X, reduce_range)
        # 使用 numpy 测试，断言 X_scale 与 qparams[0] 数组几乎相等（精度为 3 位小数）
        np.testing.assert_array_almost_equal(X_scale, qparams[0], decimal=3)
        # 断言 X_zp 与 qparams[1] 相等
        self.assertEqual(X_zp, qparams[1])

    # 如果 CUDA 不可用，则跳过测试 CUDA 量化不会固定内存的情况
    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_cuda_quantization_does_not_pin_memory(self):
        # 创建随机张量 x
        x = torch.randn(3)
        # 断言张量 x 没有固定内存
        self.assertEqual(x.is_pinned(), False)

        # 在 CUDA 设备上创建随机整数张量 q_int，并量化为 q 张量
        q_int = torch.randint(0, 100, [1, 2, 3], device="cuda", dtype=torch.uint8)
        q = torch._make_per_tensor_quantized_tensor(q_int, scale=0.1, zero_point=0)

        # 重新创建随机张量 x
        x = torch.randn(3)
        # 断言张量 x 没有固定内存
        self.assertEqual(x.is_pinned(), False)

    # 没有方法可以实际固定量化张量的内存
    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_quant_pin_memory(self):
        # 创建随机张量 x 并固定其内存
        x = torch.randn(3).pin_memory()
        # 断言张量 x 已固定内存
        self.assertEqual(x.is_pinned(), True)
        
        # 对 x 进行量化得到 x_q
        x_q = torch.quantize_per_tensor(x, 1, 0, torch.quint8)
        # 断言 x_q 没有固定内存
        self.assertEqual(x_q.is_pinned(), False)
        
        # 使用 x_q 创建一个固定内存的量化张量 x_pin
        x_pin = torch.empty_quantized([3], x_q, pin_memory=True, dtype=torch.quint8)
        # 断言 x_pin 没有固定内存
        self.assertEqual(x_pin.is_pinned(), False)
        
        # 引发 RuntimeError，因为无法对量化张量 x_q 固定内存
        self.assertRaises(RuntimeError, lambda: x_q.pin_memory())

    # 测试函数：FP16 饱和操作
    def test_fp16_saturate_op(self):
        # 创建全为 65532 的浮点张量 x
        x = torch.ones(5, 5, dtype=torch.float32) * 65532
        # 将第一行的值设为全为 -65532
        x[0] = torch.ones(5) * -65532
        # 创建参考的 FP16 饱和值张量 ref
        # FP16 值的范围为 [-65504, +65504]
        ref = torch.ones(5, 5) * 65504
        ref[0] = torch.ones(5) * -65504
        # 对张量 x 进行 FP16 饱和操作得到 y
        y = torch._saturate_weight_to_fp16(x)
        # 断言 y 等于 ref
        self.assertEqual(y, ref)

    # 测试函数：优化选择量化参数
    def test_choose_qparams_optimized(self):
        # 对于每个比特宽度 bit_width
        for bit_width in [4, 2]:
            # 创建包含 64 个随机浮点数的张量 x
            x = torch.randn(64, dtype=torch.float)
            # 使用优化方法选择量化参数 y
            y = torch.choose_qparams_optimized(x, numel=64, n_bins=200, ratio=0.16, bit_width=bit_width)
            # 使用贪婪搜索得到参考的量化参数 ref
            ref = param_search_greedy(x.numpy(), bit_rate=bit_width)
            # 断言 y[0] 与 ref[0] 数组相等
            self.assertEqual(y[0].numpy(), ref[0])
            # 断言 y[1] 与 ref[1] 数组相等
            self.assertEqual(y[1].numpy(), ref[1])
    # 测试通过 pickle 对象保存和加载量化张量
    def _test_pickle_checkpoint_qtensor(self, device):
        # 使用临时文件名上下文管理器
        with TemporaryFileName() as fname:
            # 定义一个继承自 ScriptModule 的类 M
            class M(torch.jit.ScriptModule):
                # 定义类变量 __constants__
                __constants__ = ['fname']

                # 初始化方法
                def __init__(self):
                    super().__init__()
                    # 设置实例变量 fname
                    self.fname = fname

                # 前向传播方法的 Torch 脚本版本
                @torch.jit.script_method
                def forward(self, x, y):
                    # 将 x 和 y 保存到 fname 指定的文件中
                    torch.save((x, y), self.fname)
                    # 返回 y
                    return y

            # 创建一个量化后的张量 q
            q = torch.quantize_per_tensor(
                torch.rand(2, 3, dtype=torch.float), scale=0.1, zero_point=10, dtype=torch.quint8).to(device)
            # 创建一个通道量化的张量 qc
            qc = torch.quantize_per_channel(
                torch.rand(2, 3, dtype=torch.float),
                scales=torch.tensor([0.1, 0.5, 0.01]),
                zero_points=torch.tensor([10, 0, 20]),
                axis=1, dtype=torch.quint8).to(device)
            # 创建 M 类的实例 m
            m = M()
            # 调用 m 的前向传播方法
            m(q, qc)
            # 使用 with 语句打开 fname 指定的文件，读取数据
            with open(fname, "rb") as handle:
                # 遍历 weights_only 参数的两种取值
                for weights_only in [True, False]:
                    # 从文件中加载数据
                    loaded_q, loaded_qc = torch.load(fname, weights_only=weights_only)
                    # 使用断言检查加载的量化张量与原始张量 q 和 qc 是否相等
                    self.assertEqual(loaded_q, q)
                    self.assertEqual(loaded_qc, qc)

    # 测试 CPU 下的 pickle 量化张量检查点
    def test_pickle_checkpoint_qtensor(self):
        self._test_pickle_checkpoint_qtensor('cpu')

    # 测试 Torch 脚本的序列化
    def test_jit_serialization(self):
        # 定义 SimpleQTensor 类，继承自 ScriptModule
        class SimpleQTensor(torch.jit.ScriptModule):
            # 初始化方法
            def __init__(self, per_channel):
                super().__init__()
                # 创建一个浮点型张量 x
                x = torch.rand(5, 5).float()
                # 根据 per_channel 参数判断是否使用通道量化
                if not per_channel:
                    x_q = torch.quantize_per_tensor(x, 0.2, 10, torch.quint8)
                else:
                    s = torch.rand(5, dtype=torch.float64) + 0.1
                    zp = torch.randint(5, 15, (5,))
                    x_q = torch.quantize_per_channel(x, s, zp, 1, torch.quint8)
                # 将量化后的张量 x_q 注册为 buffer
                self.register_buffer('x', x_q)

            # Torch 脚本版本的前向传播方法
            @torch.jit.script_method
            def forward(self):
                # 返回注册的量化张量 x
                return self.x

        # 遍历 per_channel 参数的两种取值
        for per_channel in [False, True]:
            # 创建 SimpleQTensor 的实例 model
            model = SimpleQTensor(per_channel)
            # 创建一个字节流缓冲区 buffer
            buffer = io.BytesIO()
            # 将 model 保存到 buffer 中
            torch.jit.save(model, buffer)
            buffer.seek(0)
            # 从 buffer 中加载模型
            model_loaded = torch.jit.load(buffer)
            # 使用断言检查加载的模型的输出是否与原始模型的输出相等
            self.assertEqual(model_loaded(), model())

    # 测试 bfloat16 张量的量化与反量化
    def test_bfp16_quantize(self):
        # 创建一个随机张量 X
        X = torch.randn(5 , 10)
        # 将 X 转换为 bfloat16 类型的量化张量 quantized_X
        quantized_X = X.to(torch.bfloat16)
        # 将 quantized_X 转换回 float32 类型的张量 dedequantized_X
        dedequantized_X = quantized_X.to(torch.float32)
        # 使用断言检查 dedequantized_X 是否与原始张量 X 的值接近
        torch.testing.assert_close(X, dedequantized_X, rtol=1e-4, atol=5e-3)
    def test_decomposed_quantize_per_tensor(self):
        # 导入 torch.ao.quantization.fx._decomposed 模块，注册运算符
        import torch.ao.quantization.fx._decomposed
        # 创建一个形状为 (5, 10) 的随机张量 X
        X = torch.randn(5, 10)
        # 定义测试用例列表，每个元素是一个四元组 (输出量化类型, 输出数据类型, 量化的最小值, 量化的最大值)
        test_cases = [
            (torch.quint8, torch.uint8, 0, 255),
            (torch.qint8, torch.int8, -128, 127),
            (torch.qint32, torch.int32, -2**31, 2**31 - 1),
        ]
        # 对于每个测试用例进行以下操作
        for qdtype, dtype, quant_min, quant_max in test_cases:
            # 计算 X 的动态量化参数 scale 和 zero_point
            scale, zero_point = _calculate_dynamic_qparams(X, qdtype)
            # 对张量 X 进行量化，使用 scale 和 zero_point，生成量化后的张量 quantized_X
            quantized_X = torch.quantize_per_tensor(X, scale, zero_point, qdtype)
            # 调用自定义的 quantize_per_tensor 运算符，对张量 X 进行分解量化
            quantized_decomposed_X = \
                torch.ops.quantized_decomposed.quantize_per_tensor(
                    X, scale, zero_point, quant_min, quant_max, dtype)
            # 断言分解量化后的张量的数据类型为 dtype
            self.assertEqual(quantized_decomposed_X.dtype, dtype)
            # 断言 quantized_X 的整数表示等于 quantized_decomposed_X
            self.assertEqual(quantized_X.int_repr(), quantized_decomposed_X)

    def test_decomposed_quantize_per_tensor_bfloat16_input(self):
        # 导入 torch.ao.quantization.fx._decomposed 模块，注册运算符
        import torch.ao.quantization.fx._decomposed
        # 创建一个形状为 (5, 5) 的随机整数张量 X，并将其转换为 float32 类型
        X = torch.randint(1, 10, (5, 5)).to(torch.float32)
        # 计算 X 的动态量化参数 scale 和 zero_point，使用 quint8 作为输出量化类型
        scale, zero_point = _calculate_dynamic_qparams(X, torch.quint8)
        # 对张量 X 进行量化，使用 scale 和 zero_point，生成量化后的张量 quantized_X
        quantized_X = torch.quantize_per_tensor(X, scale, zero_point, torch.quint8)
        # 将张量 X 转换为 bfloat16 类型，并调用自定义的 quantize_per_tensor 运算符进行分解量化
        quantized_decomposed_X = \
            torch.ops.quantized_decomposed.quantize_per_tensor(
                X.to(torch.bfloat16), scale, zero_point, 0, 255, torch.uint8)
        # 断言分解量化后的张量的数据类型为 uint8
        self.assertEqual(quantized_decomposed_X.dtype, torch.uint8)
        # 断言 quantized_X 的整数表示等于 quantized_decomposed_X
        self.assertEqual(quantized_X.int_repr(), quantized_decomposed_X)

    def test_decomposed_dequantize_per_tensor(self):
        # 导入 torch.ao.quantization.fx._decomposed 模块，注册运算符
        import torch.ao.quantization.fx._decomposed
        # 创建一个形状为 (5, 10) 的随机张量 X
        X = torch.randn(5, 10)
        # 定义测试用例列表，每个元素是一个四元组 (输出量化类型, 输出数据类型, 量化的最小值, 量化的最大值)
        test_cases = [
            (torch.quint8, torch.uint8, 0, 255),
            (torch.qint8, torch.int8, -128, 127),
            (torch.qint32, torch.int32, -2**31, 2**31 - 1),
        ]
        # 对于每个测试用例进行以下操作
        for qdtype, dtype, quant_min, quant_max in test_cases:
            # 计算 X 的动态量化参数 scale 和 zero_point
            scale, zero_point = _calculate_dynamic_qparams(X, qdtype)
            # 对张量 X 进行量化，使用 scale 和 zero_point，生成量化后的张量 quantized_X
            quantized_X = torch.quantize_per_tensor(X, scale, zero_point, qdtype)
            # 对量化后的张量进行反量化操作，生成浮点数张量 dequantized_X
            dequantized_X = torch.dequantize(quantized_X)

            # 调用自定义的 quantize_per_tensor 和 dequantize_per_tensor 运算符，分别对张量 X 进行量化和反量化操作
            quantized_decomposed_X = torch.ops.quantized_decomposed.quantize_per_tensor(
                X, scale, zero_point, quant_min, quant_max, dtype)
            dequantized_decomposed_X = torch.ops.quantized_decomposed.dequantize_per_tensor(
                quantized_decomposed_X, scale, zero_point, quant_min, quant_max, dtype
            )
            # 断言 quantized_X 的整数表示等于 quantized_decomposed_X
            self.assertEqual(quantized_X.int_repr(), quantized_decomposed_X)
            # 断言 dequantized_X 等于 dequantized_decomposed_X
            self.assertEqual(dequantized_X, dequantized_decomposed_X)
    def test_decomposed_dynamic_quant_pattern(self):
        # 导入需要的模块和函数
        import torch.ao.quantization.fx._decomposed
        # 创建一个形状为 (5, 10) 的随机张量 X
        X = torch.randn(5, 10)
        # 定义数据类型为 uint8 和相应的量化数据类型 quint8
        dtype = torch.uint8
        qdtype = torch.quint8
        # 根据张量 X 选择量化参数的比例和零点
        scale, zero_point = torch._choose_qparams_per_tensor(X, False)
        # 定义量化的最小值和最大值
        quant_min, quant_max = 0, 255

        # 对张量 X 进行量化
        quantized_X = torch.quantize_per_tensor(X, scale, zero_point, qdtype)
        # 对量化后的张量进行反量化
        dequantized_X = torch.dequantize(quantized_X)

        # 尝试使用分解模式进行量化
        # 选择使用分解模式的量化参数（比例和零点）
        (scale_decomposed, zero_point_decomposed) = torch.ops.quantized_decomposed.choose_qparams.tensor(
            X, quant_min, quant_max, torch.Tensor([torch.finfo(torch.float32).eps]), dtype)
        # 使用分解模式对张量 X 进行量化
        quantized_decomposed_X = torch.ops.quantized_decomposed.quantize_per_tensor.tensor(
            X, scale_decomposed, zero_point_decomposed, quant_min, quant_max, dtype)
        # 使用分解模式对量化后的张量进行反量化
        dequantized_decomposed_X = torch.ops.quantized_decomposed.dequantize_per_tensor.tensor(
            quantized_decomposed_X, scale_decomposed, zero_point_decomposed, quant_min, quant_max, dtype
        )
        # 断言量化结果和反量化结果与原始量化结果相同
        self.assertEqual(quantized_X.int_repr(), quantized_decomposed_X)
        self.assertEqual(dequantized_X, dequantized_decomposed_X)

    def test_decomposed_quantize_per_channel(self):
        # 注册相关操作
        import torch.ao.quantization.fx._decomposed
        # 创建一个形状为 (5, 10) 的随机张量 X
        X = torch.randn(5, 10)
        # 定义量化的数据类型 quint8 和数据类型 uint8
        qdtype = torch.quint8
        dtype = torch.uint8
        # 随机生成长度为 5 的比例数组和长度为 5 的零点数组
        scales = torch.randn(5,)
        zero_points = torch.randint(0, 100, (5,))
        # 定义量化的最小值和最大值
        quant_min, quant_max = 0, 255
        # 定义量化的轴向
        axis = 0

        # 对张量 X 进行通道级别的量化
        quantized_X = torch.quantize_per_channel(X, scales, zero_points, axis, qdtype)
        # 使用分解模式对张量 X 进行通道级别的量化
        quantized_decomposed_X = \
            torch.ops.quantized_decomposed.quantize_per_channel(
                X, scales, zero_points, axis, quant_min, quant_max, dtype)
        # 断言使用分解模式量化后的结果数据类型为指定的 uint8 类型
        self.assertEqual(quantized_decomposed_X.dtype, dtype)
        # 断言量化后的结果与使用分解模式量化后的结果在整数表示上相等
        self.assertEqual(quantized_X.int_repr(), quantized_decomposed_X)

    def test_decomposed_quantize_per_channel_bfloat16_input(self):
        # 注册相关操作
        import torch.ao.quantization.fx._decomposed
        # 创建一个形状为 (5, 5) 的随机整数张量 X，并将其转换为 float32 类型
        X = torch.randint(1, 10, (5, 5)).to(torch.float32)
        # 定义量化的数据类型 quint8 和数据类型 uint8
        qdtype = torch.quint8
        dtype = torch.uint8
        # 随机生成长度为 5 的比例数组和长度为 5 的零点数组
        scales = torch.randn(5,)
        zero_points = torch.randint(0, 100, (5,))
        # 定义量化的最小值和最大值
        quant_min, quant_max = 0, 255
        # 定义量化的轴向
        axis = 0

        # 对张量 X 进行通道级别的量化
        quantized_X = torch.quantize_per_channel(X, scales, zero_points, axis, qdtype)
        # 使用分解模式对张量 X（转换为 bfloat16 类型）进行通道级别的量化
        quantized_decomposed_X = \
            torch.ops.quantized_decomposed.quantize_per_channel(
                X.to(torch.bfloat16), scales, zero_points, axis, quant_min, quant_max, dtype)
        # 断言使用分解模式量化后的结果数据类型为指定的 uint8 类型
        self.assertEqual(quantized_decomposed_X.dtype, dtype)
        # 断言量化后的结果与使用分解模式量化后的结果在整数表示上相等
        self.assertEqual(quantized_X.int_repr(), quantized_decomposed_X)
    def test_decomposed_dequantize_per_channel(self):
        # 导入必要的运算符模块
        import torch.ao.quantization.fx._decomposed
        # 创建一个形状为 (5, 10) 的随机张量 X
        X = torch.randn(5, 10)
        # 设置量化后的数据类型为 quint8
        qdtype = torch.quint8
        # 设置量化前的数据类型为 uint8
        dtype = torch.uint8
        # 创建长度为 5 的随机张量 scales
        scales = torch.randn(5,)
        # 创建长度为 5 的随机整数张量 zero_points，值域为 [0, 100)
        zero_points = torch.randint(0, 100, (5,))
        # 设置量化的最小值和最大值
        quant_min, quant_max = 0, 255
        # 设置量化的轴向
        axis = 0

        # 使用 torch.quantize_per_channel 对张量 X 进行分通道量化
        quantized_X = torch.quantize_per_channel(X, scales, zero_points, axis, qdtype)
        # 对分通道量化后的张量进行反量化操作
        dequantized_X = torch.dequantize(quantized_X)

        # 使用自定义的 quantized_decomposed.quantize_per_channel 函数进行分通道量化
        quantized_decomposed_X = \
            torch.ops.quantized_decomposed.quantize_per_channel(
                X, scales, zero_points, axis, quant_min, quant_max, dtype)
        # 使用自定义的 quantized_decomposed.dequantize_per_channel 函数进行反量化
        dequantized_decomposed_X = \
            torch.ops.quantized_decomposed.dequantize_per_channel(
                quantized_decomposed_X, scales, zero_points, axis, quant_min, quant_max, dtype)

        # 断言分通道量化后的张量整数表示与自定义量化函数结果一致
        self.assertEqual(quantized_X.int_repr(), quantized_decomposed_X)
        # 断言反量化后的张量与自定义反量化函数结果一致
        self.assertEqual(dequantized_X, dequantized_decomposed_X)

    def test_decomposed_choose_qparams_per_token_asymmetric_backward(self):
        # 导入必要的运算符模块
        import torch.ao.quantization.fx._decomposed
        # 创建一个形状为 (2, 3) 的张量 x，并设置其需要梯度
        x = torch.randn(2, 3).requires_grad_()
        # 调用自定义的 _choose_qparams_per_token_asymmetric_impl 函数选择每个 token 的非对称量化参数
        (s, zp) = torch.ops.quantized_decomposed._choose_qparams_per_token_asymmetric_impl(x, torch.int8)
        # 使用选择的参数进行伪量化操作
        out = x.div(s).add(zp).round()
        # 计算输出的和，并进行反向传播
        out.sum().backward()

    def test_decomposed_quantize_per_channel_group(self):
        # 导入必要的运算符模块
        import torch.ao.quantization.fx._decomposed
        # 设置分通道组量化的最小值和最大值
        qmin, qmax = (-8, 7)
        # 设置每组的大小
        group_size = 128
        # 创建一个形状为 (100, 256) 的随机张量 x
        x = torch.randn(100, 256)
        # 创建形状为 (100, 2) 的随机张量 s，用于分通道组量化
        s = torch.randn(100, 2)
        # 创建形状为 (100, 2) 的随机整数张量 zp，表示每组的零点
        zp = torch.randint(qmax, size=(100, 2), dtype=torch.int32)

        # 使用自定义的 quantized_decomposed.quantize_per_channel_group 函数进行分通道组量化
        q = torch.ops.quantized_decomposed.quantize_per_channel_group(
            x, s, zp, qmin, qmax, torch.int8, group_size,
        )
        # 使用自定义的 quantized_decomposed.dequantize_per_channel_group 函数进行反量化
        dq = torch.ops.quantized_decomposed.dequantize_per_channel_group(
            q, s, zp, qmin, qmax, torch.int8, group_size, torch.float32
        )

        # 使用 torch.fake_quantize_per_channel_affine 函数进行分组伪量化
        x_grouped = x.reshape(-1, group_size)
        s_flattened = s.flatten()
        zp_flattened = zp.flatten()
        fq = torch.fake_quantize_per_channel_affine(
            x_grouped, s_flattened, zp_flattened, 0, qmin, qmax,
        )
        # 将伪量化的结果 reshape 成与原始张量 x 相同的形状
        fq = fq.reshape_as(x)
        # 断言反量化结果与伪量化结果在给定的误差范围内相等
        torch.testing.assert_close(dq, fq, rtol=0, atol=0)
    # 定义测试函数，用于测试分解量化每个标记的操作
    def test_decomposed_quantize_per_token(self):
        # 导入 torch.ao.quantization.fx._decomposed 模块，注册相关操作
        import torch.ao.quantization.fx._decomposed
        # 定义量化范围的最小值和最大值
        qmin, qmax = (-8, 7)
        # 生成一个大小为 (100, 256) 的随机张量 x
        x = torch.randn(100, 256)
        # 生成一个大小为 (100, 1) 的随机张量 s，用于量化的比例因子
        s = torch.randn(100, 1)
        # 生成一个大小为 (100, 1) 的随机张量 zp，用于量化的零点偏移
        zp = torch.randint(qmax, size=(100, 1), dtype=torch.int32)

        # 使用 quantize_per_token 函数进行每个标记的伪量化
        q = torch.ops.quantized_decomposed.quantize_per_token(
            x, s, zp, qmin, qmax, torch.int8,
        )
        # 使用 dequantize_per_token 函数进行每个标记的反量化
        dq = torch.ops.quantized_decomposed.dequantize_per_token(
            q, s, zp, qmin, qmax, torch.int8, torch.float32
        )

        # 使用 `torch.fake_quantize_per_channel_affine` 实现每个标记的伪量化
        s_flattened = s.flatten()
        zp_flattened = zp.flatten()
        fq = torch.fake_quantize_per_channel_affine(
            x, s_flattened, zp_flattened, 0, qmin, qmax,
        )
        # 断言反量化结果 dq 与伪量化结果 fq 在相对误差 rtol=0 和绝对误差 atol=0 的情况下相似
        torch.testing.assert_close(dq, fq, rtol=0, atol=0)
# 如果当前脚本作为主程序运行，则抛出运行时错误并显示一条错误消息
if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_quantization.py TESTNAME\n\n"
                       "instead.")
```