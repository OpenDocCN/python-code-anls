# `.\pytorch\test\quantization\core\test_workflow_module.py`

```py
# Owner(s): ["oncall: quantization"]

# Torch模块导入
import torch
from torch.ao.quantization import (
    MinMaxObserver,  # 导入MinMaxObserver类，用于量化观察器
    PerChannelMinMaxObserver,  # 导入PerChannelMinMaxObserver类，用于通道间量化观察器
    MovingAverageMinMaxObserver,  # 导入MovingAverageMinMaxObserver类，用于移动平均最小最大量化观察器
    MovingAveragePerChannelMinMaxObserver,  # 导入MovingAveragePerChannelMinMaxObserver类，用于通道间移动平均最小最大量化观察器
    HistogramObserver,  # 导入HistogramObserver类，用于直方图量化观察器
    RecordingObserver,  # 导入RecordingObserver类，用于记录量化观察器
    PlaceholderObserver,  # 导入PlaceholderObserver类，用于占位符量化观察器
    NoopObserver,  # 导入NoopObserver类，用于空操作量化观察器
    FakeQuantize,  # 导入FakeQuantize类，用于伪量化
    FixedQParamsObserver,  # 导入FixedQParamsObserver类，用于固定量化参数量化观察器
    default_debug_qconfig,  # 导入default_debug_qconfig常量，用于默认的调试量化配置
    default_observer,  # 导入default_observer常量，用于默认的量化观察器
    default_histogram_observer,  # 导入default_histogram_observer常量，用于默认的直方图量化观察器
    default_per_channel_weight_observer,  # 导入default_per_channel_weight_observer常量，用于默认的通道权重量化观察器
    prepare,  # 导入prepare函数，用于量化准备
    prepare_qat,  # 导入prepare_qat函数，用于量化训练中的准备
    convert,  # 导入convert函数，用于模型量化转换
    QConfig,  # 导入QConfig类，用于量化配置
    FusedMovingAvgObsFakeQuantize,  # 导入FusedMovingAvgObsFakeQuantize类，用于融合移动平均伪量化观察器
    get_embedding_qat_module_mappings,  # 导入get_embedding_qat_module_mappings函数，用于获取量化训练嵌入模块映射
    get_embedding_static_quant_module_mappings,  # 导入get_embedding_static_quant_module_mappings函数，用于获取静态量化嵌入模块映射
)
from torch.ao.quantization.quantize import _get_observer_dict  # 导入_get_observer_dict函数，用于获取量化观察器字典

import torch.nn as nn

# 标准库导入
import copy
import io
import itertools
import unittest
import math
import numpy as np

# 测试工具导入
from hypothesis import given, settings
from hypothesis import strategies as st
import torch.testing._internal.hypothesis_utils as hu
hu.assert_deadline_disabled()  # 禁用测试工具的截止时间检查
from torch.testing._internal.common_cuda import TEST_MULTIGPU, TEST_CUDA
from torch.testing._internal.common_utils import TestCase, skipIfTorchDynamo
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    AnnotatedSingleLayerLinearModel,
    test_only_eval_fn,
    SingleLayerLinearModel,
)

from torch.testing._internal.common_quantized import (
    override_quantized_engine,
    supported_qengines,
    override_qengines,
    _fake_quantize_per_channel_affine_reference,
    _fake_quantize_per_channel_affine_grad_reference,
    to_tensor,
)

from torch.testing._internal.common_quantization import (
    DeFusedEmbeddingBagLinear,
)

NP_RANDOM_SEED = 19  # 设置随机数种子
tolerance = 1e-6  # 设置公差

class TestObserver(QuantizationTestCase):
    @given(qdtype=st.sampled_from((torch.qint8, torch.quint8, torch.qint32)),
           qscheme=st.sampled_from((torch.per_tensor_affine, torch.per_tensor_symmetric)),
           reduce_range=st.booleans())
    @given(qdtype=st.sampled_from((torch.qint8, torch.quint8)),
           qscheme=st.sampled_from((torch.per_channel_affine, torch.per_channel_symmetric, torch.per_channel_affine_float_qparams)),
           ch_axis=st.sampled_from((0, 1, 2, 3)), reduce_range=st.booleans())
    def test_observer_scriptable(self):
        obs_list = [MinMaxObserver(), MovingAverageMinMaxObserver()]  # 创建量化观察器实例列表
        for obs in obs_list:
            scripted = torch.jit.script(obs)  # 对观察器进行脚本化

            x = torch.rand(3, 4)  # 创建随机张量
            obs(x)  # 应用观察器到张量上
            scripted(x)  # 应用脚本化后的观察器到张量上
            self.assertEqual(obs.calculate_qparams(), scripted.calculate_qparams())  # 比较计算后的量化参数是否一致

            buf = io.BytesIO()  # 创建一个字节流缓冲区
            torch.jit.save(scripted, buf)  # 将脚本化后的模型保存到字节流中
            buf.seek(0)  # 将字节流指针移动到起始位置
            loaded = torch.jit.load(buf)  # 从字节流中加载模型
            self.assertEqual(obs.calculate_qparams(), loaded.calculate_qparams())  # 比较加载后的模型计算的量化参数是否一致

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @override_qengines
    def test_state_dict_respects_device_affinity(self):
        """
        Tests that loading from a state dict loads buffers to the correct
        device.
        """
        # 定义 CPU 和 CUDA 设备
        device_cpu = torch.device('cpu')
        device_cuda = torch.device('cuda:0')
        # 生成测试用例的所有可能组合
        test_cases = itertools.product(
            [device_cpu, device_cuda],
            [device_cpu, device_cuda],
            [MinMaxObserver, MovingAverageMinMaxObserver,
             PerChannelMinMaxObserver,
             MovingAveragePerChannelMinMaxObserver,
             # TODO: enable this (separate PR)
             # HistogramObserver,
             PlaceholderObserver, RecordingObserver, NoopObserver,
             FakeQuantize])

        for device_source, device_target, obs_cls in test_cases:
            # 创建一个观察器类的实例
            model = obs_cls()
            # 将模型放置在指定的设备上
            model.to(device_source)
            # 对模型进行一次前向传播
            model(torch.randn(4, 1, 4, 4, device=device_source))
            # 创建另一个相同类型的观察器类的实例
            model2 = obs_cls()
            # 将第二个模型放置在目标设备上
            model2.to(device_target)
            # 加载第一个模型的状态字典到第二个模型中
            model2.load_state_dict(model.state_dict())
            # 验证模型参数和缓冲区是否仍然在model2的设备上
            model_devices = {p.device for p in model2.parameters()} | \
                {p.device for p in model2.buffers()}
            # 由于某些观察器可能没有任何缓冲区，所以使用“小于等于”而不是“等于”
            self.assertLessEqual(len(model_devices), 1)
            if len(model_devices) == 1:
                model_device = next(iter(model_devices))
                # 断言模型的设备与目标设备相同
                self.assertEqual(model_device, device_target)

    def test_histogram_observer_consistent_buffer_shape(self):
        """
        Ensures that the buffer shapes do not change from uninitialized to
        initialized states for HistogramObserver.
        """
        # 创建一个直方图观察器实例
        obs = HistogramObserver()
        # 记录初始化前的最小值和最大值形状
        min_shape_before = obs.min_val.shape
        max_shape_before = obs.max_val.shape
        for _ in range(2):
            # 对观察器传入随机张量进行观测
            obs(torch.randn(4, 4, 4, 4))
        # 断言初始化前后最小值和最大值的形状未改变
        self.assertEqual(min_shape_before, obs.min_val.shape)
        self.assertEqual(max_shape_before, obs.max_val.shape)

    def test_histogram_observer_ignore_infinity(self):
        """
        Ensures that HistogramObserver doesn't record values of infinity
        """
        # 创建两个直方图观察器实例
        obs = HistogramObserver()
        obs2 = HistogramObserver()
        x = torch.randn(4, 4, 4, 4)
        # 对第一个观察器传入含有无穷大的张量
        obs(x * torch.inf)
        obs(x)
        obs2(x)
        obs(x * torch.inf)
        # 断言观察器未记录无穷大的最小值和最大值
        self.assertTrue(obs.min_val != -torch.inf and obs.max_val != torch.inf)
        # 断言两个观察器的直方图相等
        self.assertEqual(obs.histogram, obs2.histogram)
    # 测试直方图观察器处理接近无穷大情况
    def test_histogram_observer_handle_close_to_infinity(self):
        # 对于正负符号循环
        for sign in [-1, 1]:
            # 使用不缩减范围的参数创建直方图观察器对象
            obser = HistogramObserver.with_args(reduce_range=False)()
            # 创建一个张量 mask，其中的最小值接近于负无穷大
            mask = torch.tensor([-3.4028234663852886 * 10**30, 0, 0, 0]) * sign
            # 将 mask 输入到观察器中
            obser(mask)
            # 输入 mask 减去 sign 后的结果到观察器中
            obser(mask - sign)
            # 计算量化参数的缩放因子和零点
            scale, zp = obser.calculate_qparams()

            # 创建一个随机输入张量
            input = torch.randn(1, 4)
            # 计算参考结果，softmax 应用于输入加上 mask 的结果
            ref_result = torch.softmax(input + mask, dim=1)

            # 将 mask 进行量化，使用之前计算得到的 scale 和 zp，将量化后的 mask 进行反量化
            quant_mask = torch.quantize_per_tensor(mask, scale, zp, torch.quint8)
            dequant_mask = quant_mask.dequantize()
            # 计算结果，softmax 应用于输入加上反量化的 mask 的结果
            result = torch.softmax(input + dequant_mask, dim=1)
            # 断言结果与参考结果相等
            self.assertEqual(result, ref_result)

    # 测试直方图观察器由于接近最小和最大值导致的内存溢出
    def test_histogram_observer_handle_OOM_due_to_close_min_max_value(self):
        # 使用默认参数创建直方图观察器对象
        obser = HistogramObserver.with_args(reduce_range=False)()
        # 在观察器中输入 x1 张量，包含接近最小值和零的值
        x1 = torch.tensor([0, 1e-9])
        obser(x1)

        # 在观察器中输入 x2 张量，包含较大的值
        x2 = torch.tensor([2.0, 3.0])
        obser(x2)

    # 测试直方图观察器由于大的上采样率导致的内存溢出
    def test_histogram_observer_handle_OOM_due_to_large_upsample_rate(self):
        # 使用大的上采样率参数创建直方图观察器对象
        obser = HistogramObserver.with_args(upsample_rate=(8000**2), reduce_range=False)()

        # 在观察器中输入 x1 张量，包含接近零和一的值
        x1 = torch.tensor([0, 1.0])
        obser(x1)

        # 在观察器中输入 x2 张量，包含较大的值
        x2 = torch.tensor([2, 2 + 1e-9])
        obser(x2)

    # 测试直方图观察器保存和加载状态字典的功能
    def test_histogram_observer_save_load_state_dict(self):
        """
        Smoke test on saving/loading state_dict
        """
        # 创建一个直方图观察器对象 obs1
        obs1 = HistogramObserver()
        # 将随机张量输入到 obs1 中
        obs1(torch.randn(4, 4, 4, 4))
        # 创建另一个直方图观察器对象 obs2
        obs2 = HistogramObserver()
        # 加载 obs1 的状态字典到 obs2
        obs2.load_state_dict(obs1.state_dict())
        # 断言 obs2 的 min_val 和 max_val 的形状与预期相等
        self.assertEqual(obs2.min_val.shape, torch.Size([]))
        self.assertEqual(obs2.max_val.shape, torch.Size([]))
    def test_save_load_state_dict_script(self):
        """
        Tests that we can save and load state_dict for observers that are scripted
        in a quantized model.
        """
        # 定义观察器类列表，用于测试
        obs_list = [MinMaxObserver, MovingAverageMinMaxObserver, HistogramObserver]

        for obs in obs_list:
            # 创建一个单层线性模型并设置为评估模式
            model = SingleLayerLinearModel().eval()
            # 定义量化配置，其中激活函数使用默认观察器，权重使用当前观察器类
            qconfig = QConfig(activation=default_observer, weight=obs)
            qconfig_dict = {'' : qconfig}
            # 对模型进行脚本化
            scripted = torch.jit.script(model)
            # 准备脚本化模型以便量化
            scripted = torch.ao.quantization.prepare_jit(scripted, qconfig_dict)
            # 创建一个随机张量作为输入
            x = torch.rand(5, 5)
            # 在脚本化模型上执行前向传播
            scripted(x)
            # 获取观察器的状态字典
            obs_dict = torch.ao.quantization.get_observer_state_dict(scripted)

            # 加载统计信息
            # 再次对模型进行脚本化
            scripted_2 = torch.jit.script(model)
            # 准备另一个脚本化模型以便量化
            scripted_2 = torch.ao.quantization.prepare_jit(scripted_2, qconfig_dict)
            # 加载观察器的状态字典到模型中
            torch.ao.quantization.load_observer_state_dict(scripted_2, obs_dict)
            # 验证状态字典是否与原始模型完全匹配
            self.assertEqual(scripted.state_dict(), scripted_2.state_dict())


    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_observer_qparams_respects_device_affinity(self):
        """
        Ensure that the scale and zero_point returned by the observer
        are on the same device as the input tensor.
        """
        # 定义观察器列表
        observerList = [MinMaxObserver(),
                        MovingAverageMinMaxObserver(),
                        PerChannelMinMaxObserver(),
                        MovingAveragePerChannelMinMaxObserver()]
        for obs in observerList:
            # 将观察器移动到指定设备（cuda:1）
            device = torch.device('cuda:1')
            x = torch.randn(1, 2, device=device)
            obs.to(device)
            # 对输入张量进行观察
            result = obs(x)
            # 计算量化参数（scale 和 zero_point）
            scale, zero_point = obs.calculate_qparams()

            # 验证 scale 和 zero_point 是否与输入张量在相同设备上
            self.assertEqual(x.device, scale.device)
            self.assertEqual(x.device, zero_point.device)

    def test_zero_numel(self):
        # 定义观察器类列表
        obs_list = [MinMaxObserver, MovingAverageMinMaxObserver,
                    PerChannelMinMaxObserver,
                    MovingAveragePerChannelMinMaxObserver, HistogramObserver,
                    FakeQuantize, FixedQParamsObserver]
        for obs_cls in obs_list:
            if obs_cls is FixedQParamsObserver:
                # 如果是 FixedQParamsObserver 类，使用指定的参数创建观察器
                obs = obs_cls(0.1, 0)
            else:
                # 否则，使用默认构造函数创建观察器
                obs = obs_cls()
            x = torch.tensor([])
            # 验证不会因为输入张量为空而导致崩溃
            x = obs(x)

    def test_dynamic_quant_observer(self):
        # 创建一个动态观察器，其计算动态量化参数
        obs = MovingAverageMinMaxObserver(averaging_constant=1, is_dynamic=True)
        x = torch.randn((3, 3))
        obs(x)
        # 获取初始的量化参数
        params = obs.calculate_qparams()
        for _ in range(20):
            # 多次对不同输入进行观察
            obs(10 * torch.randn((3, 3)))
            # 验证动态量化参数是否发生变化
            self.assertNotEqual(params, obs.calculate_qparams())
            obs(x)
            # 验证量化参数是否与初始相同
            self.assertEqual(params, obs.calculate_qparams())
    # 定义测试方法，用于验证动态量化观察器匹配选择量化参数的行为
    def test_dynamic_quant_observer_matching_choose_qparams(self):
        # 创建一个动态移动平均最小最大观察器对象，设置平均常数为1
        obs = MovingAverageMinMaxObserver(averaging_constant=1, is_dynamic=True)
        
        # 对不同形状的张量进行迭代
        for x in [torch.randn(3, 3), torch.rand(3, 3, 3), torch.randn(3, 3, 3, 3)]:
            # 使用观察器对象处理当前张量
            obs(x)
            
            # 计算当前观察器的量化参数
            params = obs.calculate_qparams()
            
            # 根据当前张量选择量化参数（缩放因子和零点）
            scale, zero_point = torch._choose_qparams_per_tensor(x)
            
            # 断言当前量化参数与观察器计算的量化参数一致
            self.assertEqual(scale, params[0])
            self.assertEqual(zero_point, params[1])

    # 定义测试方法，用于验证通道级别观察器加载状态字典的行为
    def test_per_channel_observers_load_state_dict(self):
        # 定义观察器类列表
        observer_list = [PerChannelMinMaxObserver, MovingAveragePerChannelMinMaxObserver]

        # 对每种观察器类进行迭代
        for obs_cls in observer_list:
            # 创建当前观察器类的观察器对象
            obs = obs_cls()
            
            # 对一个随机形状为 (32, 32) 的张量进行观察
            obs(torch.randn((32, 32)))
            
            # 创建新的观察器对象
            new_obs = obs_cls()
            
            # 确保可以成功加载状态字典
            new_obs.load_state_dict(obs.state_dict())
            
            # 断言原观察器对象的最小值与新观察器对象的最小值相等
            self.assertTrue(torch.equal(obs.min_val, new_obs.min_val))
            
            # 断言原观察器对象的最大值与新观察器对象的最大值相等
            self.assertTrue(torch.equal(obs.max_val, new_obs.max_val))
# 创建一个继承自HistogramObserver的_ReferenceHistogramObserver类，用于实现与主程序相同的直方图观察功能
class _ReferenceHistogramObserver(HistogramObserver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # 使用torch.jit.ignore装饰器，标记为不需要进行 Torch JIT 脚本化的类或方法
    @torch.jit.ignore
    # 继承自QuantizationTestCase的TestRecordHistogramObserver类，用于记录直方图观察器的测试
class TestRecordHistogramObserver(QuantizationTestCase):
    # TODO: 将此方法移到quantize.py文件中
    def test_record_observer(self):
        # 遍历支持的量化引擎列表
        for qengine in supported_qengines:
            # 使用override_quantized_engine上下文管理器，设置当前量化引擎为qengine
            with override_quantized_engine(qengine):
                # 创建AnnotatedSingleLayerLinearModel模型实例
                model = AnnotatedSingleLayerLinearModel()
                # 设置模型的量化配置为default_debug_qconfig
                model.qconfig = default_debug_qconfig
                # 对模型进行准备，以便量化
                model = prepare(model)
                # 执行评估并记录所有张量数据
                test_only_eval_fn(model, self.calib_data)
                test_only_eval_fn(model, self.calib_data)
                # 创建空的观察器字典
                observer_dict = {}
                # 调用_get_observer_dict函数，将模型的观察器信息填充到observer_dict中
                _get_observer_dict(model, observer_dict)

                # 断言'fc1.module.activation_post_process'在observer_dict的键中，
                # 并输出错误信息'observer is not recorded in the dict'，如果不存在的话
                self.assertTrue('fc1.module.activation_post_process' in observer_dict.keys(),
                                'observer is not recorded in the dict')
                # 断言'fc1.module.activation_post_process'观察器的张量值列表长度为
                # 2倍于self.calib_data的长度
                self.assertEqual(len(observer_dict['fc1.module.activation_post_process'].get_tensor_value()),
                                 2 * len(self.calib_data))
                # 断言'fc1.module.activation_post_process'观察器的第一个张量值与
                # 模型对self.calib_data[0][0]的输出相等
                self.assertEqual(observer_dict['fc1.module.activation_post_process'].get_tensor_value()[0],
                                 model(self.calib_data[0][0]))

    # 使用hypothesis库的given装饰器，参数qdtype从torch.qint8和torch.quint8中抽样选择
    def test_observer_scriptable(self, qdtype):
        # 创建RecordingObserver实例obs，使用指定的数据类型qdtype
        obs = RecordingObserver(dtype=qdtype)
        # 对obs进行Torch JIT脚本化
        scripted = torch.jit.script(obs)

        # 创建一个3x4的随机张量x
        x = torch.rand(3, 4)
        # 对obs和scripted分别执行张量x的观察
        obs(x)
        scripted(x)
        # 断言obs和scripted的第一个张量值相等
        self.assertTrue(torch.equal(obs.get_tensor_value()[0], scripted.get_tensor_value()[0]))
        # 创建一个BytesIO对象buf，用于保存Torch JIT脚本化后的模型
        buf = io.BytesIO()
        torch.jit.save(scripted, buf)
        buf.seek(0)
        # 加载buf中的模型，得到loaded
        loaded = torch.jit.load(buf)
        # 断言obs和loaded的第一个张量值相等
        self.assertTrue(torch.equal(obs.get_tensor_value()[0], loaded.get_tensor_value()[0]))

# 继承自QuantizationTestCase的TestHistogramObserver类，用于直方图观察器的测试
class TestHistogramObserver(QuantizationTestCase):
    # 使用hypothesis库的given装饰器，参数qdtype从torch.qint8和torch.quint8中抽样选择，
    # qscheme从torch.per_tensor_affine和torch.per_tensor_symmetric中抽样选择
    def test_observer_scriptable(self, qdtype, qscheme):
        # 创建包含两种观察器的列表ob_list
        ob_list = [
            HistogramObserver(dtype=qdtype, qscheme=qscheme),
            default_histogram_observer()
        ]
        # 遍历ob_list中的每个观察器obs
        for obs in ob_list:
            # 对obs进行Torch JIT脚本化
            scripted = torch.jit.script(obs)

            # 创建一个3x4的随机张量x
            x = torch.rand(3, 4)
            # 对obs和scripted分别执行张量x的观察
            obs(x)
            scripted(x)
            # 断言obs和scripted的直方图值相等
            self.assertTrue(torch.equal(obs.histogram, scripted.histogram))
            # 创建一个BytesIO对象buf，用于保存Torch JIT脚本化后的模型
            buf = io.BytesIO()
            torch.jit.save(scripted, buf)
            buf.seek(0)
            # 加载buf中的模型，得到loaded
            loaded = torch.jit.load(buf)
            # 断言obs和loaded的直方图值相等
            self.assertTrue(torch.equal(obs.histogram, scripted.histogram))
    @given(qdtype=st.sampled_from((torch.qint8, torch.quint8)),
           qscheme=st.sampled_from((torch.per_tensor_affine, torch.per_tensor_symmetric)),
           reduce_range=st.booleans())
    @settings(max_examples=10)
    # 定义测试函数，用于测试 HistogramObserver 类的功能
    def test_histogram_observer(self, qdtype, qscheme, reduce_range):
        # 创建 HistogramObserver 实例，设置初始参数
        myobs = HistogramObserver(bins=3, dtype=qdtype, qscheme=qscheme, reduce_range=reduce_range)
        # 计算并获取量化参数，用于空观察者情况
        qparams = myobs.calculate_qparams()
        # 创建包含梯度的张量
        x = torch.tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
        # 创建普通张量
        y = torch.tensor([5.0, 6.0, 7.0, 8.0])
        # 对张量 x 应用量化观察器
        out_x = myobs(x)
        # 断言输出张量需要梯度计算
        self.assertTrue(out_x.requires_grad)
        # 对张量 y 应用量化观察器
        myobs(y)
        # 断言观察器记录的最小值和最大值
        self.assertEqual(myobs.min_val, 2.0)
        self.assertEqual(myobs.max_val, 8.0)
        # 断言观察器记录的直方图
        self.assertEqual(myobs.histogram, [2., 3., 3.])

        # 重新计算量化参数
        qparams = myobs.calculate_qparams()

        # 根据 reduce_range 和 qscheme 设置参考的量化参数
        if reduce_range:
            if qscheme == torch.per_tensor_symmetric:
                ref_scale = 0.0470588 * 255 / 127
                ref_zero_point = 0 if qdtype is torch.qint8 else 128
            else:
                ref_scale = 0.0235294 * 255 / 127
                ref_zero_point = -64 if qdtype is torch.qint8 else 0
        else:
            if qscheme == torch.per_tensor_symmetric:
                ref_scale = 0.0470588
                ref_zero_point = 0 if qdtype is torch.qint8 else 128
            else:
                ref_scale = 0.0235294
                ref_zero_point = -128 if qdtype is torch.qint8 else 0

        # 断言计算得到的量化参数与预期值一致
        self.assertEqual(qparams[1].item(), ref_zero_point)
        self.assertEqual(qparams[0].item(), ref_scale, atol=1e-5, rtol=0)
        
        # 测试对象的可序列化性
        state_dict = myobs.state_dict()
        b = io.BytesIO()
        # 将状态字典保存到字节流中
        torch.save(state_dict, b)
        b.seek(0)
        # 从字节流中加载状态字典
        loaded_dict = torch.load(b)
        # 验证加载后的状态字典与原始状态字典相等
        for key in state_dict:
            self.assertEqual(state_dict[key], loaded_dict[key])
        # 创建新的 HistogramObserver 实例，加载加载后的状态字典
        loaded_obs = HistogramObserver(bins=3, dtype=qdtype, qscheme=qscheme, reduce_range=reduce_range)
        loaded_obs.load_state_dict(loaded_dict)
        # 重新计算加载后实例的量化参数
        loaded_qparams = loaded_obs.calculate_qparams()
        # 断言加载前后的对象属性一致
        self.assertEqual(myobs.min_val, loaded_obs.min_val)
        self.assertEqual(myobs.max_val, loaded_obs.max_val)
        self.assertEqual(myobs.histogram, loaded_obs.histogram)
        self.assertEqual(myobs.bins, loaded_obs.bins)
        self.assertEqual(myobs.calculate_qparams(), loaded_obs.calculate_qparams())

    # 测试仅支持单边量化的情况
    def test_histogram_observer_one_sided(self):
        # 创建 HistogramObserver 实例，设置单边量化所需参数
        myobs = HistogramObserver(bins=8, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=True)
        # 创建张量 x 和 y
        x = torch.tensor([0.0, 0.3, 1.2, 1.7])
        y = torch.tensor([0.1, 1.3, 2.0, 2.7])
        # 对张量 x 和 y 应用量化观察器
        myobs(x)
        myobs(y)
        # 断言观察器记录的最小值为 0
        self.assertEqual(myobs.min_val, 0)
        # 重新计算并验证量化参数
        qparams = myobs.calculate_qparams()
        self.assertEqual(qparams[1].item(), 0)
    def test_histogram_observer_same_inputs(self):
        # 创建一个 HistogramObserver 实例，设置参数
        myobs = HistogramObserver(bins=3, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False)
        
        # 创建四个张量，每个都要求梯度跟踪
        w = torch.ones(4, requires_grad=True)
        x = torch.zeros(4, requires_grad=True)
        y = torch.tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
        z = torch.tensor([5.0, 6.0, 7.0, 8.0])
        
        # 对每个张量应用 HistogramObserver 实例
        myobs(w)
        myobs(x)
        myobs(x)  # 再次应用 x
        myobs(y)
        myobs(z)
        
        # 计算并断言最小值、最大值和直方图的期望值
        qparams = myobs.calculate_qparams()
        self.assertEqual(myobs.min_val, 2.0)
        self.assertEqual(myobs.max_val, 8.0)
        self.assertEqual(myobs.histogram, [2., 3., 3.])

    @skipIfTorchDynamo("too slow")
    @given(N=st.sampled_from([10, 1000]),
           bins=st.sampled_from([256, 512, 1024, 2048]),
           dtype=st.sampled_from([torch.qint8, torch.quint8]),
           qscheme=st.sampled_from([torch.per_tensor_affine, torch.per_tensor_symmetric]),
           reduce_range=st.booleans())
    def test_histogram_observer_against_reference(self, N, bins, dtype, qscheme, reduce_range):
        # 创建参考的 _ReferenceHistogramObserver 和待测试的 HistogramObserver 实例
        ref_obs = _ReferenceHistogramObserver(bins=bins, dtype=dtype, qscheme=qscheme, reduce_range=reduce_range)
        my_obs = HistogramObserver(bins=bins, dtype=dtype, qscheme=qscheme, reduce_range=reduce_range)

        # 迭代多次，进行直方图观察
        for _ in range(10):
            X = torch.randn(N)
            my_obs(X)
            ref_obs(X)
            # 断言待测试直方图与参考直方图的一致性
            self.assertEqual(my_obs.histogram, ref_obs.histogram)
            self.assertEqual(my_obs.min_val, ref_obs.min_val)
            self.assertEqual(my_obs.max_val, ref_obs.max_val)

        # 计算并断言量化参数的一致性
        ref_qparams = ref_obs.calculate_qparams()
        my_qparams = my_obs.calculate_qparams()

        # 迭代计算量化误差并进行断言
        for i in range(0, bins, 200):
            for j in range(i + 5, bins, 200):
                ref_qe = ref_obs._compute_quantization_error(i, j)
                qe = my_obs._compute_quantization_error(i, j)
                self.assertEqual(ref_qe, qe)

        # 最终断言量化参数的一致性
        self.assertEqual(ref_qparams, my_qparams)

    def test_histogram_observer_extreme_inputs(self):
        """
        确保 HistogramObserver 能够在极端情况下正确工作
        """
        # 创建 HistogramObserver 实例
        obs = HistogramObserver()
        
        # 极端输入示例
        test_input = torch.tensor(
            [0.0, 0.0, 4.58e-41, 4.58e-41]
        )
        
        # 运行两次观察，基于 forward 函数的行为，第一次初始化 min_val 和 max_val，第二次调用 _adjust_min_max
        obs(test_input)
        obs(test_input)

    def test_histogram_observer_correct_numel(self):
        # 迭代测试不同大小的输入张量
        for i in range(1, 10):
            # 创建 HistogramObserver 实例
            obs = HistogramObserver()
            
            # 对大小为 i x i 的随机张量进行观察
            obs(torch.randn(i, i))
            
            # 断言直方图的总和是否与张量元素数的平方相等
            self.assertEqual(obs.histogram.sum().item(), i**2)
class TestFakeQuantize(TestCase):
    @given(device=st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']),
           X=hu.per_channel_tensor(shapes=hu.array_shapes(2, 5,),
           qparams=hu.qparams(dtypes=torch.qint8)))
    def test_fq_module_per_channel(self, device, X):
        np.random.seed(NP_RANDOM_SEED)  # 设置随机种子
        X, (scale, zero_point, axis, torch_type) = X  # 解包测试数据和量化参数
        quant_min = torch.iinfo(torch_type).min  # 获取量化类型的最小值
        quant_max = torch.iinfo(torch_type).max  # 获取量化类型的最大值

        X = to_tensor(X, device)  # 将测试数据转换为张量，并放到指定设备上
        X.requires_grad_()  # 设置张量需要梯度信息
        fq_module = FakeQuantize(default_per_channel_weight_observer, quant_min, quant_max, ch_axis=axis).to(device)
        Y_prime = fq_module(X)  # 使用量化模块进行前向传播
        assert fq_module.scale is not None  # 断言量化模块的尺度参数不为空
        assert fq_module.zero_point is not None  # 断言量化模块的零点参数不为空
        Y = _fake_quantize_per_channel_affine_reference(X, fq_module.scale,
                                                        fq_module.zero_point, axis, quant_min, quant_max)  # 使用参考函数计算参考输出
        np.testing.assert_allclose(Y.cpu().detach().numpy(), Y_prime.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)  # 断言前向传播的输出与参考输出相近

        # Test backward
        dout = torch.rand_like(X, dtype=torch.float, device=device)  # 创建与 X 相同形状的随机梯度张量
        Y_prime.backward(dout)  # 对前向传播的结果进行反向传播
        dX = _fake_quantize_per_channel_affine_grad_reference(dout, X, fq_module.scale,
                                                              fq_module.zero_point, axis, quant_min, quant_max)  # 使用参考函数计算参考梯度
        np.testing.assert_allclose(dX.cpu().numpy(), X.grad.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)  # 断言计算得到的梯度与张量的实际梯度相近

    def test_fq_serializable_per_channel(self):
        observer = default_per_channel_weight_observer  # 获取默认的通道权重观察器
        quant_min = -128  # 设置量化的最小值
        quant_max = 127  # 设置量化的最大值
        fq_module = FakeQuantize(observer, quant_min, quant_max)  # 创建量化模块
        X = torch.tensor([[-5, -3.5, -2, 0, 3, 5, 7], [1, 3, 2, 5, 6.5, 8, 10]], dtype=torch.float32)  # 创建张量 X
        y_ref = fq_module(X)  # 使用量化模块对 X 进行量化
        state_dict = fq_module.state_dict()  # 获取量化模块的状态字典
        self.assertEqual(state_dict['scale'], [0.054902, 0.078431])  # 断言尺度参数的值
        self.assertEqual(state_dict['zero_point'], [0, 0])  # 断言零点参数的值
        b = io.BytesIO()  # 创建字节流对象
        torch.save(state_dict, b)  # 将状态字典保存到字节流中
        b.seek(0)  # 将字节流指针移到起始位置
        loaded_dict = torch.load(b)  # 从字节流中加载状态字典
        for key in state_dict:
            self.assertEqual(state_dict[key], loaded_dict[key])  # 断言保存前后状态字典的一致性

    def test_quant_min_max_override(self):
        observer = default_per_channel_weight_observer  # 获取默认的通道权重观察器
        # test no override
        fq_module = FakeQuantize(observer)  # 创建未覆盖量化范围的量化模块
        self.assertEqual(fq_module.activation_post_process.quant_min, -128)  # 断言量化模块的最小量化值
        self.assertEqual(fq_module.activation_post_process.quant_max, 127)  # 断言量化模块的最大量化值
        # test quant_min/quant_max override
        fq_module = FakeQuantize(observer, quant_min=0, quant_max=127)  # 创建覆盖了量化范围的量化模块
        self.assertEqual(fq_module.activation_post_process.quant_min, 0)  # 断言量化模块的最小量化值
        self.assertEqual(fq_module.activation_post_process.quant_max, 127)  # 断言量化模块的最大量化值

def _get_buffer_ids(module):
    """
    Object addresses stay constant if and only if all modifications are in-place
    """
    # 返回一个列表，列表中每个元素是参数 module._buffers.items() 中每个值 v 的 id()
    return [id(v) for k, v in module._buffers.items()]
    # 定义一个测试类 TestDistributed，继承自 QuantizationTestCase，用于测试分布式量化功能
class TestDistributed(QuantizationTestCase):

    # 定义测试方法 test_observers_preserve_buffers，用于验证观察器仅原地修改缓冲区
    def test_observers_preserve_buffers(self):
        """
        Tests that observers only modify buffers in place. Note: this is important
        because nn.DataParallel depends on this assumption to work correctly.
        However, DataParallel does not expose IDs of the replicas, so we test it
        without DataParallel in order to easily access the object IDs.
        """
        # 定义不同类型的观察器对象列表
        observer_types = [
            torch.ao.quantization.MinMaxObserver.with_args(dtype=torch.qint8),
            torch.ao.quantization.MovingAverageMinMaxObserver.with_args(dtype=torch.qint8),
            torch.ao.quantization.PerChannelMinMaxObserver.with_args(dtype=torch.qint8),
            torch.ao.quantization.MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8),
            torch.ao.quantization.HistogramObserver.with_args(dtype=torch.qint8),
            torch.ao.quantization.RecordingObserver.with_args(dtype=torch.qint8),
            torch.ao.quantization.PlaceholderObserver.with_args(dtype=torch.float16),
        ]

        # 遍历不同类型的观察器
        for observer_type in observer_types:
            # 创建观察器对象
            observer = observer_type()
            # 获取观察器修改前的缓冲区 IDs
            buffer_ids_before = _get_buffer_ids(observer)
            # 对观察器进行5次输入观察
            for _i in range(5):
                inputs = torch.rand((4, 4, 4))
                observer(inputs)
            # 获取观察器修改后的缓冲区 IDs
            buffer_ids_after = _get_buffer_ids(observer)
            # 断言观察器的缓冲区 IDs 在修改前后保持不变
            self.assertEqual(
                buffer_ids_before,
                buffer_ids_after,
                msg=f"{str(observer)}: Buffers must be modified in place")

    # 定义测试方法 test_fake_quant_preserves_buffers，用于验证伪量化仅原地修改缓冲区
    def test_fake_quant_preserves_buffers(self):
        """
        Tests that fake quant only modifies buffers in place. Note: this is important
        because nn.DataParallel depends on this assumption to work correctly.
        However, DataParallel does not expose IDs of the replicas, so we test it
        without DataParallel in order to easily access the object IDs.
        """
        # 创建 FakeQuantize 模型对象
        model = torch.ao.quantization.FakeQuantize()
        # 获取模型修改前的缓冲区 IDs
        buffer_ids_before = _get_buffer_ids(model)
        # 对模型进行5次输入观察
        for _i in range(5):
            inputs = torch.rand((4, 4, 4))
            model(inputs)
        # 应用伪量化操作
        model.apply(torch.ao.quantization.enable_fake_quant)
        # 取消伪量化操作
        model.apply(torch.ao.quantization.disable_fake_quant)
        # 启用观察器
        model.apply(torch.ao.quantization.enable_observer)
        # 禁用观察器
        model.apply(torch.ao.quantization.disable_observer)
        # 获取模型修改后的缓冲区 IDs
        buffer_ids_after = _get_buffer_ids(model)
        # 断言模型的缓冲区 IDs 在修改前后保持不变
        self.assertEqual(
            buffer_ids_before,
            buffer_ids_after,
            msg="FakeQuant: Buffers must be modified in place")

    # 使用条件装饰器跳过多GPU测试
    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    # 使用条件装饰器跳过CUDA测试
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_qat_data_parallel(self):
        """
        Tests that doing QAT in nn.DataParallel does not crash.
        测试在 nn.DataParallel 中进行量化感知训练（QAT）不会导致崩溃。
        """
        # 检查是否支持 fbgemm 引擎，如果不支持则直接返回
        if 'fbgemm' not in torch.backends.quantized.supported_engines:
            return
        
        # 使用 fbgemm 引擎进行量化操作
        with override_quantized_engine('fbgemm'):
            # 设置使用 CUDA 设备
            device = torch.device('cuda')

            # 定义量化感知训练的模型
            model = nn.Sequential(
                torch.ao.quantization.QuantStub(),  # 添加量化输入的量化桩
                nn.Conv2d(3, 1, 1, bias=False),     # 添加卷积层，无偏置
                nn.BatchNorm2d(1),                  # 添加批归一化层
                nn.ReLU(),                          # 添加ReLU激活层
                nn.Conv2d(1, 2, 3, stride=2, padding=1, bias=False),  # 添加卷积层，无偏置
                nn.BatchNorm2d(2),                  # 添加批归一化层
                nn.AvgPool2d(14),                   # 添加平均池化层
                nn.Sigmoid(),                       # 添加Sigmoid激活层
                torch.ao.quantization.DeQuantStub(), # 添加量化输出的去量化桩
            )

            # 融合指定的模块以进行量化感知训练
            torch.ao.quantization.fuse_modules_qat(model, [['1', '2', '3'], ['4', '5']], inplace=True)

            # 设置模型的量化配置为默认的 QAT fbgemm 配置
            model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
            
            # 准备模型以进行量化感知训练
            torch.ao.quantization.prepare_qat(model, inplace=True)
            
            # 将模型包装成 DataParallel 模型，并指定使用的设备
            model = nn.DataParallel(model, device_ids=[0, 1])
            model.to(device)  # 将模型移动到指定设备上
            model.train()     # 设置模型为训练模式

            # 执行多轮训练
            for epoch in range(3):
                inputs = torch.rand(2, 3, 28, 28).to(device)  # 生成随机输入张量并移动到设备上
                model(inputs)   # 使用模型进行前向传播

                # 在第二轮之后，禁用量化观察者
                if epoch >= 1:
                    model.apply(torch.ao.quantization.disable_observer)
                
                # 在第三轮之后，冻结批归一化统计信息
                if epoch >= 2:
                    model.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)
                
                # 深度复制模型并进行量化转换
                quant_model = copy.deepcopy(model.module)
                quant_model = torch.ao.quantization.convert(quant_model.eval().cpu(), inplace=False)

                # 使用无梯度计算环境执行量化模型的推理
                with torch.no_grad():
                    out = quant_model(torch.rand(1, 3, 28, 28))
    def test_qat_convbn_fused_syncbn_replacement(self):
        """
        Tests that SyncBatchNorm replacement works for fused ConvBN.
        """
        # 检查是否支持 fbgemm 引擎，如果不支持则退出测试
        if 'fbgemm' not in torch.backends.quantized.supported_engines:
            return
        # 使用 fbgemm 引擎进行量化替换
        with override_quantized_engine('fbgemm'):
            # 创建包含融合 Conv-BN 的模型
            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    # 定义 Conv2d 层和 BatchNorm2d 层
                    self.conv = nn.Conv2d(4, 1, 3, padding=1)
                    self.bn = nn.BatchNorm2d(1)

                def forward(self, x):
                    # 前向传播过程：Conv -> BN
                    x = self.conv(x)
                    x = self.bn(x)
                    return x

            model = Model()
            # 对模型进行融合操作
            fused_model = torch.ao.quantization.fuse_modules_qat(
                model,
                [['conv', 'bn']],
            )
            # 将模型转换为 QAT 模式
            fused_model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
            torch.ao.quantization.prepare_qat(fused_model, inplace=True)
            # 将 BatchNorm 层替换为 SyncBatchNorm 层
            fused_model = nn.SyncBatchNorm.convert_sync_batchnorm(fused_model)
            self.assertTrue(
                isinstance(fused_model.conv.bn, nn.SyncBatchNorm),
                "Expected BN to be converted to SyncBN")

    def test_syncbn_preserves_qconfig(self):
        """
        Makes sure that if a BatchNorm is not fused and a qconfig exists,
        convering the module to SyncBatchNorm preserves the qconfig.
        """
        # 创建包含单独 Conv2d 和 BatchNorm2d 的 Sequential 模型
        m = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
        )
        # 设置 BatchNorm 层的量化配置
        m[1].qconfig = torch.ao.quantization.default_qconfig
        # 将模型转换为 SyncBatchNorm 模式
        m = torch.nn.SyncBatchNorm.convert_sync_batchnorm(m)
        # 断言：转换后的 SyncBatchNorm 层仍然保留了原有的量化配置
        self.assertTrue(
            hasattr(m[1], "qconfig"),
            "missing qconfig after SyncBatchNorm conversion")

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @override_qengines
    def test_device_affinity(self):
        """
        Tests that converting a model to QAT respects device affinity
        """
        # 定义一个简单的神经网络模型
        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                # 定义一个2维卷积层，输入通道数为1，输出通道数为1，卷积核大小为1x1
                self.conv = nn.Conv2d(1, 1, 1)
                # 定义一个批归一化层，输入通道数为1
                self.bn = nn.BatchNorm2d(1)
                # 定义一个ReLU激活函数层
                self.relu = nn.ReLU()

            def forward(self, x):
                # 模型的前向传播过程
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                return x

        # 创建一个模型实例
        model = Model()
        # 设置模型的量化配置为默认的量化训练配置
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig(torch.backends.quantized.engine)
        # 指定模型运行在cuda:0设备上
        device = torch.device('cuda:0')
        # 将模型移动到指定设备上
        model.to(device)
        # 准备模型进行量化训练
        torch.ao.quantization.prepare_qat(model, inplace=True)
        # 收集模型参数和缓冲区所在的设备
        model_devices = {p.device for p in model.parameters()} | \
            {p.device for p in model.buffers()}
        # 断言模型参数和缓冲区只在一个设备上
        self.assertEqual(len(model_devices), 1)
        # 获取模型参数和缓冲区所在的设备
        model_device = next(iter(model_devices))
        # 断言模型参数和缓冲区所在的设备与预期设备相同
        self.assertEqual(model_device, device)

        # 确保在CUDA上运行输入数据时无需任何额外的变化
        input = torch.randn(4, 1, 4, 4, device=device)
        # 运行模型
        model(input)
# 定义一个 TestCase 类，用于测试 FusedObsFakeQuantModule
class TestFusedObsFakeQuantModule(TestCase):
    
    # 使用 Hypothesis 的 given 装饰器，设定测试参数 device 为 cpu 或 cuda
    @given(
        device=st.sampled_from(
            ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        )
    )
    # 设置测试的设置，deadline=None 表示没有执行期限限制
    @settings(deadline=None)
    # 定义测试方法 test_fused_obs_fq_module，接受参数 device
    def test_fused_obs_fq_module(self, device):
        
        # 设置测试中的参数
        
        # 创建一个形状为 (5, 5) 的随机张量 x，并将其放置在指定的设备上
        x = torch.randn(5, 5, device=device)
        
        # 创建一个初始值为正无穷大的张量 running_min_op，并将其放置在指定的设备上
        running_min_op = torch.tensor(float("inf"), device=device)
        
        # 创建一个初始值为负无穷大的张量 running_max_op，并将其放置在指定的设备上
        running_max_op = torch.tensor(float("-inf"), device=device)
        
        # 设置平均常数值为 0.01
        avg_const = 0.01
        
        # 创建一个值为 1.0 的张量 scale，并将其放置在指定的设备上
        scale = torch.tensor([1.0], device=device)
        
        # 创建一个值为 0 的整数张量 zero_point，并将其数据类型设为 torch.int，放置在指定的设备上
        zero_point = torch.tensor([0], dtype=torch.int, device=device)

        # 创建 FusedMovingAvgObsFakeQuantize 的实例 mod
        mod = FusedMovingAvgObsFakeQuantize()
        
        # 启用 mod 的 fake quantization 功能
        torch.ao.quantization.enable_fake_quant(mod)
        
        # 启用 mod 的 observer 功能
        torch.ao.quantization.enable_observer(mod)
        
        # 将 mod 移动到指定的设备上
        mod.to(device)
        
        # 对输入张量 x 运行 mod 的前向传播，得到输出 out
        out = mod(x)

        # 直接运行 torch.fused_moving_avg_obs_fake_quant 运算符
        pt_op = torch.fused_moving_avg_obs_fake_quant
        
        # 使用 pt_op 运算符计算参考输出 out_ref
        out_ref = pt_op(
            x,
            mod.observer_enabled,
            mod.fake_quant_enabled,
            running_min_op,
            running_max_op,
            scale,
            zero_point,
            avg_const,
            0,
            255,
            0,
            False,
        )

        # 使用 torch.testing.assert_close 检验 out 是否接近 out_ref
        torch.testing.assert_close(out, out_ref)
        
        # 使用 torch.testing.assert_close 检验 running_min_op 是否接近 mod.activation_post_process.min_val
        torch.testing.assert_close(
            running_min_op, mod.activation_post_process.min_val
        )
        
        # 使用 torch.testing.assert_close 检验 running_max_op 是否接近 mod.activation_post_process.max_val
        torch.testing.assert_close(
            running_max_op, mod.activation_post_process.max_val
        )

    # 使用 Hypothesis 的 given 装饰器，设定测试参数 device 为 cpu 或 cuda
    @given(
        device=st.sampled_from(
            ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        )
    )
    # 设置测试的设置，deadline=None 表示没有执行期限限制
    @settings(deadline=None)
    # 定义一个测试方法，用于测试融合的观察者和量化器的移动平均模块
    def test_fused_obs_fq_moving_avg_module(self, device):
        # 设置参数
        running_min_op = torch.tensor(float("inf"), device=device)  # 初始化运行时最小值为正无穷
        running_max_op = torch.tensor(float("-inf"), device=device)  # 初始化运行时最大值为负无穷
        avg_const = 0.001  # 移动平均的常数
        scale = torch.tensor([1.0], device=device)  # 缩放因子
        zero_point = torch.tensor([0], dtype=torch.int, device=device)  # 零点

        # 创建 FusedMovingAvgObsFakeQuantize 模块实例
        mod = FusedMovingAvgObsFakeQuantize(averaging_constant=0.001)
        mod.to(device)  # 将模块移动到指定设备
        mod.observer_enabled[0] = 0  # 禁用观察者
        mod.fake_quant_enabled[0] = 0  # 禁用量化器

        # 循环执行模块的前向传播
        for i in range(10):
            x = torch.randn(5, 5, device=device)  # 生成随机输入张量
            if i > 2:
                mod.observer_enabled[0] = 1  # 开启观察者
            if i > 4:
                mod.fake_quant_enabled[0] = 1  # 开启量化器

            # 在模块上运行前向传播
            out = mod(x)

            # 直接运行算子
            pt_op = torch.fused_moving_avg_obs_fake_quant
            out_ref = pt_op(
                x,
                mod.observer_enabled,
                mod.fake_quant_enabled,
                running_min_op,
                running_max_op,
                scale,
                zero_point,
                avg_const,
                0,
                255,
                0,
                False,
            )

            # 比较模块输出和参考值
            torch.testing.assert_close(out, out_ref)
            torch.testing.assert_close(
                running_min_op, mod.activation_post_process.min_val
            )
            torch.testing.assert_close(
                running_max_op, mod.activation_post_process.max_val
            )

    # 使用 hypothesis 的 given 和 settings 运行比较融合的观察者和量化器的 OSS 模块测试
    @given(
        device=st.sampled_from(
            ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        )
    )
    @settings(deadline=None)
    def test_compare_fused_obs_fq_oss_module(self, device):
        # 创建 FusedMovingAvgObsFakeQuantize 模块实例
        mod = FusedMovingAvgObsFakeQuantize()
        torch.ao.quantization.enable_fake_quant(mod)  # 启用模块的伪量化
        torch.ao.quantization.enable_observer(mod)  # 启用模块的观察者
        mod.to(device)  # 将模块移动到指定设备

        # 创建 FakeQuantize 模块作为参考
        mod_ref = FakeQuantize()
        torch.ao.quantization.enable_fake_quant(mod_ref)  # 启用参考模块的伪量化
        torch.ao.quantization.enable_observer(mod_ref)  # 启用参考模块的观察者
        mod_ref.to(device)  # 将参考模块移动到指定设备

        # 循环执行模块的前向传播
        for i in range(10):
            x = torch.randn(5, 5, device=device)  # 生成随机输入张量
            out = mod(x)  # 在模块上运行前向传播
            out_ref = mod_ref(x)  # 在参考模块上运行前向传播
            torch.testing.assert_close(out, out_ref)  # 比较模块输出和参考值
            torch.testing.assert_close(
                mod_ref.activation_post_process.min_val,
                mod.activation_post_process.min_val,
            )  # 比较模块和参考模块的最小值
            torch.testing.assert_close(
                mod_ref.activation_post_process.max_val,
                mod.activation_post_process.max_val,
            )  # 比较模块和参考模块的最大值
    def test_fused_mod_per_channel(self):
        # 检查当前环境是否支持 CUDA，选择设备为 CPU 或 CUDA
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        m = 5
        n = 10
        # 遍历每个设备
        for device in devices:
            # 创建一个空的张量，用于记录运行中的最小值和最大值，并填充为无穷大和负无穷大
            running_min_op = torch.empty(m, device=device).fill_(float("inf"))
            running_max_op = torch.empty(m, device=device).fill_(float("-inf"))
            avg_const = 0.001
            # 创建一个空的张量，填充为0.1，表示缩放因子
            scale = torch.empty(m, device=device).fill_(0.1)
            # 创建一个空的整型张量，填充为0，表示零点
            zero_point = torch.empty(m, dtype=torch.int, device=device).fill_(0)
            # 创建一个具有特定参数的伪量化观察器
            obs = FusedMovingAvgObsFakeQuantize.with_args(
                averaging_constant=avg_const,
                observer=MovingAveragePerChannelMinMaxObserver,
            )
            mod = obs()
            # 将模型转换为脚本模式
            mod = torch.jit.script(mod)
            # 将模型移动到指定的设备
            mod.to(device)

            # 进行10次迭代
            for i in range(10):
                # 生成指定形状的随机张量
                x = torch.randn(m, n, device=device)
                if i > 2:
                    # 激活观察器
                    mod.observer_enabled[0] = 1
                if i > 4:
                    # 启用伪量化
                    mod.fake_quant_enabled[0] = 1
                # 在模块上运行前向传播
                out = mod(x)

                # 直接运行操作符
                pt_op = torch.fused_moving_avg_obs_fake_quant

                # 使用操作符的参考输出
                out_ref = pt_op(
                    x,
                    mod.observer_enabled,
                    mod.fake_quant_enabled,
                    running_min_op,
                    running_max_op,
                    scale,
                    zero_point,
                    avg_const,
                    0,
                    255,
                    0,
                    True,
                    False,
                )
                # 比较输出和参考输出
                torch.testing.assert_close(out, out_ref)
                if mod.observer_enabled[0]:
                    # 检查运行中的最小值和最大值是否与模型中的值匹配
                    torch.testing.assert_close(
                        running_min_op, mod.activation_post_process.min_val
                    )
                    torch.testing.assert_close(
                        running_max_op, mod.activation_post_process.max_val
                    )
                if mod.fake_quant_enabled:
                    # 检查缩放因子和零点是否与模型中的值匹配
                    torch.testing.assert_close(scale, mod.scale)
                    torch.testing.assert_close(zero_point, mod.zero_point)

            # 检查模型状态字典中的最小值和最大值是否与运行中的值匹配
            torch.testing.assert_close(mod.state_dict()['activation_post_process.min_val'], running_min_op)
            torch.testing.assert_close(mod.state_dict()['activation_post_process.max_val'], running_max_op)

    def test_fused_mod_reduce_range(self):
        # 创建一个具有减少范围的伪量化观察器
        obs = FusedMovingAvgObsFakeQuantize(quant_min=0, quant_max=255, dtype=torch.quint8, reduce_range=True)
        # 断言量化的最小值和最大值是否与预期匹配
        self.assertEqual(obs.activation_post_process.quant_min, 0)
        self.assertEqual(obs.activation_post_process.quant_max, 127)
    def test_embedding_bag_qat_config(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化第一个嵌入层，使用EmbeddingBag类，设置参数：总嵌入数为10，每个嵌入维度为12，
                # 包括最后一个偏移量，不按频率缩放梯度，求和模式
                self.emb1 = torch.nn.EmbeddingBag(num_embeddings=10, embedding_dim=12,
                                                  include_last_offset=True, scale_grad_by_freq=False, mode='sum')
                # 初始化第二个嵌入层，使用EmbeddingBag类，设置参数同上
                self.emb2 = torch.nn.EmbeddingBag(num_embeddings=10, embedding_dim=12,
                                                  include_last_offset=True, scale_grad_by_freq=False, mode='sum')

            def forward(self, indices):
                # 在前向传播中，计算两个嵌入层在给定索引下的输出，并拼接结果
                return torch.cat((self.emb1(indices), self.emb2(indices)))


        qconfigs = [torch.ao.quantization.default_embedding_qat_qconfig,
                    torch.ao.quantization.default_embedding_qat_qconfig_4bit]
        # 遍历量化配置列表
        for qconfig in qconfigs:
            # 创建并训练模型
            model = Model().train()
            # 生成随机索引，形状为(5, 12)，范围在0到9之间
            indices = torch.randint(0, 10, (5, 12))

            # 设置当前模型的量化配置
            model.qconfig = qconfig

            # 准备量化训练模型，使用指定的嵌入量化模块映射
            quant_model = prepare_qat(model,
                                      mapping=get_embedding_qat_module_mappings())

            count_fake_quant = 0
            # 统计模型中的伪量化操作数量
            for name, mod in quant_model.named_modules():
                if name.endswith('weight_fake_quant'):
                    count_fake_quant += 1
                    # 断言每个伪量化操作都是FakeQuantize类型
                    self.assertEqual(type(mod), FakeQuantize)
            # 断言伪量化操作的数量为2
            self.assertEqual(count_fake_quant, 2)

            # 在量化模型上执行推理
            quant_model(indices)

            # 确保嵌入层的权重伪量化的零点值的数据类型为torch.float32
            self.assertEqual(quant_model.emb1.weight_fake_quant.zero_point.dtype, torch.float32)
            self.assertEqual(quant_model.emb2.weight_fake_quant.zero_point.dtype, torch.float32)

            # 转换为静态量化的推理图，并使用特定的嵌入静态量化模块映射
            inference_gm = convert(quant_model.eval().cpu(),
                                   mapping=get_embedding_static_quant_module_mappings())

            # 确保嵌入层现在以适当的位宽进行量化
            self.assertEqual(type(inference_gm.emb1), torch.ao.nn.quantized.EmbeddingBag)
            self.assertEqual(type(inference_gm.emb2), torch.ao.nn.quantized.EmbeddingBag)
            # 确保嵌入层的数据类型与量化配置中的权重数据类型匹配
            self.assertEqual(inference_gm.emb1.dtype, qconfig.weight().dtype)
            self.assertEqual(inference_gm.emb2.dtype, qconfig.weight().dtype)
    # 定义测试嵌入量化训练配置的方法
    def test_embedding_qat_config(self):
        # 对于每种支持的量化引擎循环测试
        for qengine in supported_qengines:
            # 使用当前量化引擎覆盖上下文环境
            with override_quantized_engine(qengine):
                # 创建 DeFusedEmbeddingBagLinear 模型实例
                model = DeFusedEmbeddingBagLinear()
                # 生成一个形状为 (5, 12) 的随机整数张量作为索引
                indices = torch.randint(0, 10, (5, 12))
                # 准备量化训练模型，使用嵌入量化模块映射
                quant_model = prepare_qat(model,
                                          mapping=get_embedding_qat_module_mappings())

                # 初始化计数器：用于统计 weight_fake_quant 模块的数量
                count_fake_quant = 0
                # 初始化计数器：用于统计 activation_post_process 模块的数量
                count_activation_postproc = 0

                # 遍历量化模型的所有模块
                for name, mod in quant_model.named_modules():
                    # 如果模块名以 'weight_fake_quant' 结尾，增加 fake_quant 计数器
                    if name.endswith('weight_fake_quant'):
                        count_fake_quant += 1
                    # 如果模块名中包含一次 'activation_post_process' 且不含 'weight_fake_quant'，增加 activation_postproc 计数器
                    if name.count('activation_post_process') == 1 and 'weight_fake_quant' not in name:
                        count_activation_postproc += 1

                # 断言：weight_fake_quant 应有两个（一个用于嵌入，一个用于线性层）
                self.assertEqual(count_fake_quant, 2)
                # 断言：activation_post_process 应有三个（一个用于嵌入，一个用于量化，一个用于线性层）
                self.assertEqual(count_activation_postproc, 3)

                # 断言：嵌入权重的 fake_quant 应为 FakeQuantize 类型
                self.assertEqual(type(quant_model.emb.weight_fake_quant), FakeQuantize)
                # 断言：嵌入权重的 zero_point 属性数据类型应为 torch.float32
                self.assertEqual(quant_model.emb.weight_fake_quant.zero_point.dtype, torch.float32)
                # 断言：嵌入激活后处理应为 NoopObserver 类型
                self.assertEqual(type(quant_model.emb.activation_post_process), NoopObserver)
                # 断言：线性层权重的 fake_quant 应为 FusedMovingAvgObsFakeQuantize 类型
                self.assertEqual(type(quant_model.linear.weight_fake_quant), FusedMovingAvgObsFakeQuantize)
                # 断言：线性层激活后处理应为 FusedMovingAvgObsFakeQuantize 类型
                self.assertEqual(type(quant_model.linear.activation_post_process), FusedMovingAvgObsFakeQuantize)

                # 对量化模型进行推理
                quant_model(indices)
                # 将量化模型转换为推理图模型，使用静态量化模块映射
                inference_gm = convert(quant_model,
                                       mapping=get_embedding_static_quant_module_mappings())
                # 断言：嵌入层现在应为 torch.ao.nn.quantized.Embedding 类型
                self.assertEqual(type(inference_gm.emb), torch.ao.nn.quantized.Embedding)
                # 断言：线性层现在应为 torch.ao.nn.quantized.Linear 类型
                self.assertEqual(type(inference_gm.linear), torch.ao.nn.quantized.Linear)
    def test_default_fused_qat_config(self):
        # 定义一个内部类 Model，继承自 nn.Module
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)  # 添加线性层，输入输出都是2维
                self.relu = nn.ReLU()  # 添加ReLU激活函数

            def forward(self, x):
                x = self.linear(x)  # 线性层前向传播
                x = self.relu(x)  # ReLU激活函数前向传播
                return x

        # 遍历两种量化引擎
        for qengine in ["fbgemm", "qnnpack"]:
            model = Model()  # 创建模型对象
            model.linear.weight = torch.nn.Parameter(torch.randn(2, 2))  # 初始化线性层权重
            sample_input = torch.randn(2, 2)  # 创建一个2x2的随机输入样本
            model.qconfig = torch.ao.quantization.get_default_qat_qconfig(qengine, version=1)  # 获取量化配置
            ref_model = torch.ao.quantization.QuantWrapper(model)  # 创建量化封装模型
            ref_model = torch.ao.quantization.prepare_qat(ref_model)  # 准备模型进行量化训练
            ref_model(sample_input)  # 对输入样本进行模型前向传播
            count_fake_quant = 0  # 初始化伪量化计数器

            # 遍历模型中的所有模块
            for name, mod in ref_model.named_modules():
                if name.endswith('weight_fake_quant'):  # 如果模块名以'weight_fake_quant'结尾
                    count_fake_quant += 1  # 增加伪量化计数器
                    self.assertEqual(type(mod), FusedMovingAvgObsFakeQuantize)  # 断言模块类型为 FusedMovingAvgObsFakeQuantize

                # 如果模块名中包含'activation_post_process'，并且不含'weight_fake_quant'
                if name.count('activation_post_process') == 1 and 'weight_fake_quant' not in name:
                    count_fake_quant += 1  # 增加伪量化计数器
                    self.assertEqual(type(mod), FusedMovingAvgObsFakeQuantize)  # 断言模块类型为 FusedMovingAvgObsFakeQuantize

            self.assertEqual(count_fake_quant, 3)  # 断言伪量化计数器为3

            # 根据不同的量化引擎设置下限、上限和观察器类型
            if qengine == "fbgemm":
                lower_bnd = 0
                upper_bnd = 127
                obs2match = MovingAveragePerChannelMinMaxObserver
            else:
                lower_bnd = 0
                upper_bnd = 255
                obs2match = MovingAverageMinMaxObserver

            # 断言量化后处理过程的量化下限和上限设置正确
            self.assertEqual(ref_model.quant.activation_post_process.activation_post_process.quant_min, lower_bnd)
            self.assertEqual(ref_model.quant.activation_post_process.activation_post_process.quant_max, upper_bnd)
            # 断言线性层权重的量化后处理类型正确
            self.assertEqual(type(ref_model.module.linear.weight_fake_quant.activation_post_process),
                             obs2match)
# 如果当前脚本被直接运行，则抛出运行时错误并显示以下消息，指导正确的使用方法
if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_quantization.py TESTNAME\n\n"
                       "instead.")
```