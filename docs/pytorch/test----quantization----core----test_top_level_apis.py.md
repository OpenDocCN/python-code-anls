# `.\pytorch\test\quantization\core\test_top_level_apis.py`

```
# Owner(s): ["oncall: quantization"]

import torch  # 导入PyTorch库
import torch.ao.quantization  # 导入PyTorch的量化模块
from torch.testing._internal.common_utils import TestCase  # 导入测试用例类


class TestDefaultObservers(TestCase):  # 定义测试类TestDefaultObservers，继承自TestCase类

    # 定义观察器列表
    observers = [
        "default_affine_fixed_qparams_observer",
        "default_debug_observer",
        "default_dynamic_quant_observer",
        "default_placeholder_observer",
        "default_fixed_qparams_range_0to1_observer",
        "default_fixed_qparams_range_neg1to1_observer",
        "default_float_qparams_observer",
        "default_float_qparams_observer_4bit",
        "default_histogram_observer",
        "default_observer",
        "default_per_channel_weight_observer",
        "default_reuse_input_observer",
        "default_symmetric_fixed_qparams_observer",
        "default_weight_observer",
        "per_channel_weight_observer_range_neg_127_to_127",
        "weight_observer_range_neg_127_to_127",
    ]

    # 定义伪量化器列表
    fake_quants = [
        "default_affine_fixed_qparams_fake_quant",
        "default_dynamic_fake_quant",
        "default_embedding_fake_quant",
        "default_embedding_fake_quant_4bit",
        "default_fake_quant",
        "default_fixed_qparams_range_0to1_fake_quant",
        "default_fixed_qparams_range_neg1to1_fake_quant",
        "default_fused_act_fake_quant",
        "default_fused_per_channel_wt_fake_quant",
        "default_fused_wt_fake_quant",
        "default_histogram_fake_quant",
        "default_per_channel_weight_fake_quant",
        "default_symmetric_fixed_qparams_fake_quant",
        "default_weight_fake_quant",
        "fused_per_channel_wt_fake_quant_range_neg_127_to_127",
        "fused_wt_fake_quant_range_neg_127_to_127",
    ]

    def _get_observer_ins(self, observer):  # 定义获取观察器实例的方法
        obs_func = getattr(torch.ao.quantization, observer)  # 获取量化模块中对应名称的方法
        return obs_func()  # 调用方法，返回观察器实例

    def test_observers(self) -> None:  # 定义测试观察器的方法
        t = torch.rand(1, 2, 3, 4)  # 创建随机张量
        for observer in self.observers:  # 遍历观察器列表
            obs = self._get_observer_ins(observer)  # 获取观察器实例
            obs.forward(t)  # 对随机张量进行前向传播

    def test_fake_quants(self) -> None:  # 定义测试伪量化器的方法
        t = torch.rand(1, 2, 3, 4)  # 创建随机张量
        for observer in self.fake_quants:  # 遍历伪量化器列表
            obs = self._get_observer_ins(observer)  # 获取伪量化器实例
            obs.forward(t)  # 对随机张量进行前向传播


class TestQConfig(TestCase):  # 定义测试类TestQConfig，继承自TestCase类

    # 定义减少范围的字典
    REDUCE_RANGE_DICT = {
        'fbgemm': (True, False),
        'qnnpack': (False, False),
        'onednn': (False, False),
        'x86': (True, False),
    }

    def test_reduce_range_qat(self) -> None:  # 定义测试减少范围的方法
        for backend, reduce_ranges in self.REDUCE_RANGE_DICT.items():  # 遍历减少范围字典
            for version in range(2):  # 遍历版本号范围
                qconfig = torch.ao.quantization.get_default_qat_qconfig(backend, version)  # 获取默认的量化训练配置

                # 获取激活函数的伪量化器并断言减少范围是否符合预期
                fake_quantize_activ = qconfig.activation()
                self.assertEqual(fake_quantize_activ.activation_post_process.reduce_range, reduce_ranges[0])

                # 获取权重的伪量化器并断言减少范围是否符合预期
                fake_quantize_weight = qconfig.weight()
                self.assertEqual(fake_quantize_weight.activation_post_process.reduce_range, reduce_ranges[1])
    # 定义测试方法 test_reduce_range，测试量化配置的减少范围
    def test_reduce_range(self) -> None:
        # 遍历 REDUCE_RANGE_DICT 字典，其中存储了不同后端的减少范围配置
        for backend, reduce_ranges in self.REDUCE_RANGE_DICT.items():
            # 对于每个后端，遍历版本号，这里只遍历一次版本号为 0
            for version in range(1):
                # 获取默认的量化配置 qconfig，根据指定的后端和版本号
                qconfig = torch.ao.quantization.get_default_qconfig(backend, version)

                # 创建激活函数的假量化对象 fake_quantize_activ
                fake_quantize_activ = qconfig.activation()
                # 断言假量化对象的激活函数减少范围是否与 REDUCE_RANGE_DICT 中的第一个值相等
                self.assertEqual(fake_quantize_activ.reduce_range, reduce_ranges[0])

                # 创建权重的假量化对象 fake_quantize_weight
                fake_quantize_weight = qconfig.weight()
                # 断言假量化对象的权重减少范围是否与 REDUCE_RANGE_DICT 中的第二个值相等
                self.assertEqual(fake_quantize_weight.reduce_range, reduce_ranges[1])
```