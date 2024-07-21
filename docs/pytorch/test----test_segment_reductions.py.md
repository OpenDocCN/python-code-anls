# `.\pytorch\test\test_segment_reductions.py`

```
# Owner(s): ["module: scatter & gather ops"]

# 导入 itertools 库中的 product 函数，用于生成迭代器的笛卡尔积
from itertools import product
# 导入 functools 库中的 partial 函数，用于部分应用函数
from functools import partial

# 导入 numpy 库并简写为 np，用于数值计算
import numpy as np
# 导入 torch 库，用于深度学习任务
import torch
# 从 torch.testing._internal.common_device_type 导入需要的测试函数和数据类型
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    dtypes,
)
# 从 torch.testing._internal.common_utils 导入测试基类 TestCase 和运行测试的函数 run_tests，以及其他测试辅助函数
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    gradcheck,
    parametrize,
    skipIfRocm,
)

# 定义一个列表，包含几种归约操作的字符串表示
reductions = ["max", "mean", "min", "sum", "prod"]


class TestSegmentReductions(TestCase):
    # 测试基类，用于组织测试用例

    # 定义一个函数，根据初始值和归约类型返回默认值
    def get_default_value(initial_value, reduction):
        if initial_value is not None:
            return initial_value
        if reduction == "max":
            return -float("Inf")  # 如果是最大值归约，则返回负无穷大
        elif reduction == "mean":
            return float("nan")  # 如果是均值归约，则返回 NaN
        elif reduction == "min":
            return float("Inf")  # 如果是最小值归约，则返回正无穷大
        elif reduction == "sum":
            return 0.0  # 如果是求和归约，则返回 0.0
        elif reduction == "prod":
            return 1.0  # 如果是乘积归约，则返回 1.0
    ):
        # 将 lengths_arr 转换为 Torch 张量，并指定设备和数据类型
        lengths = torch.tensor(lengths_arr, device=device, dtype=lengths_dtype)
        
        # 根据 lengths 生成 offsets
        zeros_shape = list(lengths.shape)
        zeros_shape[-1] = 1
        offsets = torch.cat((lengths.new_zeros(zeros_shape), lengths), -1).cumsum_(-1)

        # 将 data_arr 转换为 Torch 张量，指定设备、数据类型，并启用梯度计算
        data = torch.tensor(
            data_arr,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        
        # 将 expected_arr 转换为 Torch 张量，并指定设备和数据类型
        expected_result = torch.tensor(expected_arr, device=device, dtype=dtype)
        
        # 将 expected_grad_arr 转换为 Torch 张量，并指定设备和数据类型
        expected_grad = torch.tensor(expected_grad_arr, device=device, dtype=dtype)
        
        # 遍历 ['lengths', 'offsets'] 列表
        for mode in ['lengths', 'offsets']:
            # 准备传递给 torch._segment_reduce 的关键字参数字典
            segment_reduce_kwargs = dict(
                axis=axis,
                unsafe=unsafe,
                initial=initial_value)
            
            # 根据当前 mode 设置相应的关键字参数
            if (mode == 'lengths'):
                segment_reduce_kwargs['lengths'] = lengths
            else:
                segment_reduce_kwargs['offsets'] = offsets
            
            # 调用 torch._segment_reduce 函数计算实际结果
            actual_result = torch._segment_reduce(
                data=data,
                reduce=reduction,
                **segment_reduce_kwargs
            )
            
            # 断言实际结果与预期结果相等，设置相对和绝对容差，允许 NaN 相等
            self.assertEqual(
                expected_result, actual_result, rtol=1e-02, atol=1e-05, equal_nan=True
            )

            # 如果不需要检查反向传播，则直接返回
            if not check_backward:
                return

            # 测试反向传播
            actual_result.sum().backward()
            
            # 断言计算得到的梯度与预期梯度相等，设置相对和绝对容差，允许 NaN 相等
            self.assertEqual(
                expected_grad, data.grad, rtol=1e-02, atol=1e-05, equal_nan=True
            )
            
            # 克隆数据张量，分离计算图，并设置需要梯度计算
            data = data.clone().detach().requires_grad_(True)

            # gradcheck 不适用于 bfloat16 或 fp16 的 CPU 类型
            # 同时 fp32 存在小的数值差异
            if dtype not in [torch.half, torch.bfloat16, torch.float]:
                # gradcheck 不喜欢 "nan" 输入，将其设置为随机数 10
                d_non_nan = np.nan_to_num(data_arr, nan=10)
                new_data = torch.tensor(
                    d_non_nan,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
                
                # 使用 gradcheck 函数验证 torch._segment_reduce 的梯度计算
                self.assertTrue(
                    gradcheck(
                        lambda x: torch._segment_reduce(
                            data=x,
                            reduce=reduction,
                            **segment_reduce_kwargs
                        ),
                        (new_data,),
                    )
                )

    @dtypes(
        *product(
            (torch.half, torch.bfloat16, torch.float, torch.double),
            (torch.int, torch.int64),
        )
    )
    # 定义一个测试方法，测试简单的一维情况，接受设备和数据类型作为参数
    def test_simple_1d(self, device, dtypes):
        # 从 dtypes 中获取值类型和长度类型
        val_dtype, length_type = dtypes
        # 定义长度列表
        lengths = [1, 2, 3, 0]
        # 定义数据列表，包括整数和浮点数
        data = [1, float("nan"), 3, 4, 5, 5]

        # 遍历各种减少（reduction）的方法
        for reduction in reductions:
            # 遍历初始值选项
            for initial in [0, None]:
                # 根据初始值是否为 None 决定是否进行反向检查
                check_backward = True if initial is not None else False
                # 设置初始值和默认值
                initial_value = initial
                default_value = get_default_value(initial_value, reduction)
                # 根据不同的减少方法设定预期结果和梯度
                if reduction == "max":
                    expected_result = [1, float("nan"), 5, default_value]
                    expected_grad = [1, 1, 0, 0, 0.5, 0.5]
                elif reduction == "mean":
                    expected_result = [1, float("nan"), 4.666, default_value]
                    expected_grad = [1.0, 0.5, 0.5, 0.333, 0.333, 0.333]
                elif reduction == "min":
                    # 如果初始值不为 None，则设定一个高数值作为初始值
                    if initial is not None:
                        initial_value = 1000  # some high number
                        default_value = get_default_value(initial_value, reduction)
                    expected_result = [1, float("nan"), 4, default_value]
                    expected_grad = [1.0, 1.0, 0, 1, 0, 0]
                elif reduction == "sum":
                    expected_result = [1, float("nan"), 14, default_value]
                    expected_grad = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                elif reduction == "prod":
                    # 如果初始值不为 None，则设定一个初始值为 2，否则使用默认值
                    if initial is not None:
                        initial_value = 2  # 0 initial_value will zero out everything for prod
                        default_value = get_default_value(initial_value, reduction)
                        expected_result = [2, float("nan"), 200, default_value]
                        expected_grad = [2.0, 6.0, float("nan"), 50.0, 40.0, 40.0]
                    else:
                        expected_result = [1, float("nan"), 100, default_value]
                        expected_grad = [1.0, 3.0, float("nan"), 25.0, 20.0, 20.0]
                
                # 遍历轴和不安全标志选项
                for axis in [0, -1]:
                    for unsafe in [True, False]:
                        # 调用通用测试方法，传入各种参数进行测试
                        self._test_common(
                            reduction,
                            device,
                            val_dtype,
                            unsafe,
                            axis,
                            initial_value,
                            data,
                            lengths,
                            expected_result,
                            expected_grad,
                            check_backward,
                            length_type,
                        )

    @dtypes(
        *product(
            (torch.half, torch.bfloat16, torch.float, torch.double),
            (torch.int, torch.int64),
        )
    )
    # 定义一个测试方法，用于测试长度为零的简单情况
    def test_simple_zero_length(self, device, dtypes):
        # 解构 dtypes 元组，获取数值和长度类型
        val_dtype, length_type = dtypes
        # 创建长度为零的列表
        lengths = [0, 0]
        # 创建一个空的 torch 张量
        data = torch.ones(0)
    
        # 遍历不同的 reduction 类型
        for reduction in reductions:
            # 遍历不同的初始值选项
            for initial in [0, None]:
                # 根据初始值是否为 None 决定是否进行反向检查
                check_backward = True if initial is not None else False
                # 设置初始值
                initial_value = initial
                # 根据 reduction 和初始值获取默认值
                default_value = get_default_value(initial_value, reduction)
                
                # 根据 reduction 类型设置预期结果和梯度
                if reduction == "max":
                    expected_result = [default_value, default_value]
                    expected_grad = []
                elif reduction == "mean":
                    expected_result = [default_value, default_value]
                    expected_grad = []
                elif reduction == "min":
                    # 如果初始值不为 None，则设置一个较大的初始值
                    if initial is not None:
                        initial_value = 1000  # 一些较大的数字
                        default_value = get_default_value(initial_value, reduction)
                    expected_result = [default_value, default_value]
                    expected_grad = []
                elif reduction == "sum":
                    expected_result = [default_value, default_value]
                    expected_grad = []
                elif reduction == "prod":
                    # 如果初始值不为 None，则设置一个初始值为 2
                    if initial is not None:
                        initial_value = 2  # 初始值为 0 将导致 prod 的所有元素为零
                        default_value = get_default_value(initial_value, reduction)
                        expected_result = [default_value, default_value]
                        expected_grad = []
                    else:
                        expected_result = [default_value, default_value]
                        expected_grad = []
    
                # 遍历轴向选项
                for axis in [0]:
                    # 遍历不安全选项
                    for unsafe in [True, False]:
                        # 调用内部方法 _test_common 进行测试
                        self._test_common(
                            reduction,
                            device,
                            val_dtype,
                            unsafe,
                            axis,
                            initial_value,
                            data,
                            lengths,
                            expected_result,
                            expected_grad,
                            check_backward,
                            length_type,
                        )
    
    # 使用 skipIfRocm 装饰器标记该测试方法，跳过在 Rocm 平台下的运行
    @skipIfRocm
    # 使用 dtypes 装饰器标记该测试方法，指定不同的数据类型组合
    @dtypes(
        *product(
            (torch.half, torch.bfloat16, torch.float, torch.double),
            (torch.int, torch.int64),
        )
    )
    # 使用 parametrize 装饰器标记该测试方法，指定 reduce 参数的多个取值
    @parametrize("reduce", ['sum', 'prod', 'min', 'max', 'mean'])
    # 使用 dtypes 装饰器标记该测试方法，再次指定不同的数据类型组合
    @dtypes(
        *product(
            (torch.half, torch.bfloat16, torch.float, torch.double),
            (torch.int, torch.int64),
        )
    )
    # 使用 dtypes 装饰器标记该测试方法，再次指定不同的数据类型组合
    @dtypes(
        *product(
            (torch.half, torch.bfloat16, torch.float, torch.double),
            (torch.int, torch.int64),
        )
    )
    # 定义一个测试方法，用于测试多维张量的不同降维操作
    def test_multi_d(self, device, dtypes):
        # 解包 dtypes 元组，获取数值数据类型和长度类型
        val_dtype, length_type = dtypes
        # 设置默认的降维轴为 0
        axis = 0
        # 定义一个长度列表
        lengths = [0, 2, 3, 0]
        # 创建一个二维数组，其元素为 0 到 49 的连续整数，并将其转换为列表形式
        data = np.arange(50).reshape(5, 2, 5).tolist()
        # 初始化预期的梯度为空列表
        expected_grad = []

        # TODO: 计算梯度并检查正确性的标志，初始设为 False
        check_backward = False

        # 遍历每种降维操作类型
        for reduction in reductions:
            # 初始化值设为 0
            initial_value = 0
            # 根据不同的降维操作类型生成预期结果
            if reduction == "max":
                expected_result = [
                    np.full((2, 5), initial_value).tolist(),
                    np.max(data[:2], axis=0).tolist(),
                    np.max(data[2:], axis=0).tolist(),
                    np.full((2, 5), initial_value).tolist(),
                ]
            elif reduction == "mean":
                expected_result = [
                    np.full((2, 5), initial_value).tolist(),
                    np.mean(data[:2], axis=0).tolist(),
                    np.mean(data[2:], axis=0).tolist(),
                    np.full((2, 5), initial_value).tolist(),
                ]
            elif reduction == "min":
                # 设置初始值为一个较大的数
                initial_value = 1000  # some high number
                expected_result = [
                    np.full((2, 5), initial_value).tolist(),
                    np.min(data[:2], axis=0).tolist(),
                    np.min(data[2:], axis=0).tolist(),
                    np.full((2, 5), initial_value).tolist(),
                ]
            elif reduction == "sum":
                expected_result = [
                    np.full((2, 5), initial_value).tolist(),
                    np.sum(data[:2], axis=0).tolist(),
                    np.sum(data[2:], axis=0).tolist(),
                    np.full((2, 5), initial_value).tolist(),
                ]
            elif reduction == "prod":
                # 设置初始值为 1
                initial_value = 1
                expected_result = [
                    np.full((2, 5), initial_value).tolist(),
                    np.prod(data[:2], axis=0).tolist(),
                    np.prod(data[2:], axis=0).tolist(),
                    np.full((2, 5), initial_value).tolist(),
                ]
            # 遍历是否允许不安全操作的情况
            for unsafe in [True, False]:
                # 调用内部方法 _test_common 进行公共测试
                self._test_common(
                    reduction,
                    device,
                    val_dtype,
                    unsafe,
                    axis,
                    initial_value,
                    data,
                    lengths,
                    expected_result,
                    expected_grad,
                    check_backward,
                )

    # 装饰器，设置测试方法的数据类型为 torch.int 和 torch.int64
    @dtypes(torch.int, torch.int64)
    # 定义一个测试方法，用于测试不安全标志的行为
    def test_unsafe_flag(self, device, dtype):
        # 将 dtype 参数赋值给 length_type 变量
        length_type = dtype
        # 创建一个张量 lengths，表示各段的长度，使用指定设备和数据类型
        lengths = torch.tensor([0, 2, 3, 0], device=device, dtype=length_type)
        # 创建一个数据张量 data，包含浮点数数据，使用指定设备
        data = torch.arange(6, dtype=torch.float, device=device)

        # 在下面的代码块中，测试在长度为1-D时的错误情况
        with self.assertRaisesRegex(RuntimeError, "Expected all rows of lengths along axis"):
            # 调用 torch._segment_reduce 函数，执行求和操作，传入数据、长度、轴向参数和不安全标志
            torch._segment_reduce(data, 'sum', lengths=lengths, axis=0, unsafe=False)

        # 在下面的代码块中，测试在多维长度时的错误情况
        # 创建一个多维长度张量 nd_lengths，使用指定设备和数据类型
        nd_lengths = torch.tensor([[0, 3, 3, 0], [2, 3, 0, 0]], dtype=length_type, device=device)
        # 创建一个多维数据张量 nd_data，包含浮点数数据，使用指定设备并重塑为二维形状
        nd_data = torch.arange(12, dtype=torch.float, device=device).reshape(2, 6)
        with self.assertRaisesRegex(RuntimeError, "Expected all rows of lengths along axis"):
            # 调用 torch._segment_reduce 函数，执行求和操作，传入多维数据、多维长度、轴向参数和不安全标志
            torch._segment_reduce(nd_data, 'sum', lengths=nd_lengths, axis=1, unsafe=False)
# 实例化设备类型测试，使用 TestSegmentReductions 类，并将其添加到全局变量中
instantiate_device_type_tests(TestSegmentReductions, globals())

# 如果当前脚本作为主程序运行，则执行测试函数
if __name__ == "__main__":
    run_tests()
```