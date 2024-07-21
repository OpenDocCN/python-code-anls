# `.\pytorch\test\quantization\core\test_backend_config.py`

```
# Owner(s): ["oncall: quantization"]

import torch  # 导入PyTorch库
import torch.ao.nn.intrinsic as nni  # 导入PyTorch AO库中的intrinsic模块
import torch.ao.nn.qat as nnqat  # 导入PyTorch AO库中的qat模块
import torch.ao.nn.quantized.reference as nnqr  # 导入PyTorch AO库中的quantized.reference模块
from torch.testing._internal.common_quantization import QuantizationTestCase  # 导入测试中的量化测试用例

from torch.ao.quantization.backend_config import (  # 导入PyTorch AO量化配置的后端配置模块中的各类配置
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    DTypeWithConstraints,
    ObservationType,
)
from torch.ao.quantization.fuser_method_mappings import _sequential_wrapper2  # 导入PyTorch AO量化中的方法映射模块的特定函数
from torch.ao.quantization.fx.quantize_handler import _default_root_node_getter  # 导入PyTorch AO量化中的量化处理器模块的特定函数

class TestBackendConfig(QuantizationTestCase):
    """
    测试类：TestBackendConfig，继承自QuantizationTestCase用于量化测试
    """

    # =============
    #  DTypeConfig
    # =============

    dtype_config1 = DTypeConfig(
        input_dtype=torch.quint8,  # 输入数据类型设置为8位无符号整数
        output_dtype=torch.quint8,  # 输出数据类型设置为8位无符号整数
        weight_dtype=torch.qint8,   # 权重数据类型设置为8位有符号整数
        bias_dtype=torch.float      # 偏置数据类型设置为浮点数
    )

    dtype_config2 = DTypeConfig(
        input_dtype=torch.float16,  # 输入数据类型设置为16位浮点数
        output_dtype=torch.float,   # 输出数据类型设置为浮点数
        is_dynamic=True             # 数据类型是动态的标志设置为True
    )

    activation_dtype_with_constraints = DTypeWithConstraints(
        dtype=torch.quint8,         # 数据类型设置为8位无符号整数
        quant_min_lower_bound=0,     # 量化最小下界设置为0
        quant_max_upper_bound=127,   # 量化最大上界设置为127
        scale_min_lower_bound=2 ** -12,  # 缩放最小下界设置为2的负12次方
    )

    weight_dtype_with_constraints = DTypeWithConstraints(
        dtype=torch.qint8,          # 数据类型设置为8位有符号整数
        quant_min_lower_bound=-128,  # 量化最小下界设置为-128
        quant_max_upper_bound=127,   # 量化最大上界设置为127
        scale_min_lower_bound=2 ** -12,  # 缩放最小下界设置为2的负12次方
    )

    dtype_config3 = DTypeConfig(
        input_dtype=activation_dtype_with_constraints,  # 输入数据类型设置为带有约束的激活数据类型
        output_dtype=activation_dtype_with_constraints, # 输出数据类型设置为带有约束的激活数据类型
        weight_dtype=weight_dtype_with_constraints,     # 权重数据类型设置为带有约束的权重数据类型
    )

    dtype_config_dict1_legacy = {
        "input_dtype": torch.quint8,    # 输入数据类型设置为8位无符号整数
        "output_dtype": torch.quint8,   # 输出数据类型设置为8位无符号整数
        "weight_dtype": torch.qint8,    # 权重数据类型设置为8位有符号整数
        "bias_dtype": torch.float,      # 偏置数据类型设置为浮点数
    }

    dtype_config_dict2_legacy = {
        "input_dtype": torch.float16,   # 输入数据类型设置为16位浮点数
        "output_dtype": torch.float,    # 输出数据类型设置为浮点数
        "is_dynamic": True,             # 数据类型是动态的标志设置为True
    }

    dtype_config_dict1 = {
        "input_dtype": DTypeWithConstraints(dtype=torch.quint8),  # 输入数据类型设置为带有约束的8位无符号整数数据类型
        "output_dtype": DTypeWithConstraints(torch.quint8),       # 输出数据类型设置为带有约束的8位无符号整数数据类型
        "weight_dtype": DTypeWithConstraints(torch.qint8),        # 权重数据类型设置为带有约束的8位有符号整数数据类型
        "bias_dtype": torch.float,                                # 偏置数据类型设置为浮点数
    }

    dtype_config_dict2 = {
        "input_dtype": DTypeWithConstraints(dtype=torch.float16),  # 输入数据类型设置为带有约束的16位浮点数数据类型
        "output_dtype": DTypeWithConstraints(dtype=torch.float),   # 输出数据类型设置为带有约束的浮点数数据类型
        "is_dynamic": True,                                        # 数据类型是动态的标志设置为True
    }

    dtype_config_dict3 = {
        "input_dtype": activation_dtype_with_constraints,          # 输入数据类型设置为带有约束的激活数据类型
        "output_dtype": activation_dtype_with_constraints,         # 输出数据类型设置为带有约束的激活数据类型
        "weight_dtype": weight_dtype_with_constraints,             # 权重数据类型设置为带有约束的权重数据类型
    }
    # 测试从字典创建 DTypeConfig 对象的方法
    def test_dtype_config_from_dict(self):
        # 断言从旧版字典创建的 DTypeConfig 对象等于预期的 self.dtype_config1
        self.assertEqual(DTypeConfig.from_dict(self.dtype_config_dict1_legacy), self.dtype_config1)
        # 断言从旧版字典创建的 DTypeConfig 对象等于预期的 self.dtype_config2
        self.assertEqual(DTypeConfig.from_dict(self.dtype_config_dict2_legacy), self.dtype_config2)
        # 断言从新版字典创建的 DTypeConfig 对象等于预期的 self.dtype_config1
        self.assertEqual(DTypeConfig.from_dict(self.dtype_config_dict1), self.dtype_config1)
        # 断言从新版字典创建的 DTypeConfig 对象等于预期的 self.dtype_config2
        self.assertEqual(DTypeConfig.from_dict(self.dtype_config_dict2), self.dtype_config2)
        # 断言从新版字典创建的 DTypeConfig 对象等于预期的 self.dtype_config3
        self.assertEqual(DTypeConfig.from_dict(self.dtype_config_dict3), self.dtype_config3)

    # 测试将 DTypeConfig 对象转换为字典的方法
    def test_dtype_config_to_dict(self):
        # 断言 DTypeConfig 对象转换为字典等于预期的 self.dtype_config_dict1
        self.assertEqual(self.dtype_config1.to_dict(), self.dtype_config_dict1)
        # 断言 DTypeConfig 对象转换为字典等于预期的 self.dtype_config_dict2
        self.assertEqual(self.dtype_config2.to_dict(), self.dtype_config_dict2)
        # 断言 DTypeConfig 对象转换为字典等于预期的 self.dtype_config_dict3
        self.assertEqual(self.dtype_config3.to_dict(), self.dtype_config_dict3)

    # ======================
    #  BackendPatternConfig
    # ======================

    # 预定义一个用于后端模式配置的方法，使用线性和ReLU作为模式
    _fuser_method = _sequential_wrapper2(nni.LinearReLU)

    # 定义一个映射，将张量参数的数量映射到观察类型
    _num_tensor_args_to_observation_type = {
        0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
        2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    }
    
    # 定义一个映射，将输入类型映射到索引位置
    _input_type_to_index = {
        "bias": 0,
        "input": 1,
        "weight": 2,
    }

    # 定义一个方法，用于获取额外输入
    def _extra_inputs_getter(self, p):
        return (torch.rand(3, 3),)

    # 获取第一个后端操作配置对象的方法
    def _get_backend_op_config1(self):
        return BackendPatternConfig((torch.nn.Linear, torch.nn.ReLU)) \
            .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
            .add_dtype_config(self.dtype_config1) \
            .add_dtype_config(self.dtype_config2) \
            .set_root_module(torch.nn.Linear) \
            .set_qat_module(nnqat.Linear) \
            .set_reference_quantized_module(nnqr.Linear) \
            .set_fused_module(nni.LinearReLU) \
            .set_fuser_method(self._fuser_method)

    # 获取第二个后端操作配置对象的方法
    def _get_backend_op_config2(self):
        return BackendPatternConfig(torch.add) \
            .add_dtype_config(self.dtype_config2) \
            ._set_root_node_getter(_default_root_node_getter) \
            ._set_extra_inputs_getter(self._extra_inputs_getter) \
            ._set_num_tensor_args_to_observation_type(self._num_tensor_args_to_observation_type) \
            ._set_input_type_to_index(self._input_type_to_index)

    # 获取第一个后端模式配置字典的方法
    def _get_backend_pattern_config_dict1(self):
        return {
            "pattern": (torch.nn.Linear, torch.nn.ReLU),
            "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
            "dtype_configs": [self.dtype_config_dict1, self.dtype_config_dict2],
            "root_module": torch.nn.Linear,
            "qat_module": nnqat.Linear,
            "reference_quantized_module_for_root": nnqr.Linear,
            "fused_module": nni.LinearReLU,
            "fuser_method": self._fuser_method,
        }
    # 返回一个包含预设后端模式配置的字典
    def _get_backend_pattern_config_dict2(self):
        return {
            "pattern": torch.add,  # 定义模式为 torch.add 函数
            "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,  # 观察类型设定为 OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
            "dtype_configs": [self.dtype_config_dict2],  # 使用预设的 dtype 配置列表
            "root_node_getter": _default_root_node_getter,  # 根节点获取器设定为默认的 _default_root_node_getter 函数
            "extra_inputs_getter": self._extra_inputs_getter,  # 额外输入获取器设定为当前对象的 _extra_inputs_getter 方法
            "num_tensor_args_to_observation_type": self._num_tensor_args_to_observation_type,  # 将张量参数数目映射到观察类型的映射方法
            "input_type_to_index": self._input_type_to_index,  # 输入类型到索引的映射方法
        }

    # 测试设置后端操作配置的观察类型
    def test_backend_op_config_set_observation_type(self):
        conf = BackendPatternConfig(torch.nn.Linear)  # 创建一个线性层模型的后端模式配置对象
        self.assertEqual(conf.observation_type, ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # 断言观察类型为 OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
        conf.set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)  # 设置观察类型为 OUTPUT_SHARE_OBSERVER_WITH_INPUT
        self.assertEqual(conf.observation_type, ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)  # 断言观察类型已经被正确设置

    # 测试向后端操作配置添加 dtype 配置
    def test_backend_op_config_add_dtype_config(self):
        conf = BackendPatternConfig(torch.nn.Linear)  # 创建一个线性层模型的后端模式配置对象
        self.assertEqual(len(conf.dtype_configs), 0)  # 断言当前 dtype 配置列表为空
        conf.add_dtype_config(self.dtype_config1)  # 添加第一个 dtype 配置
        conf.add_dtype_config(self.dtype_config2)  # 添加第二个 dtype 配置
        self.assertEqual(len(conf.dtype_configs), 2)  # 断言现在 dtype 配置列表中有两个配置项
        self.assertEqual(conf.dtype_configs[0], self.dtype_config1)  # 断言第一个 dtype 配置已经正确设置
        self.assertEqual(conf.dtype_configs[1], self.dtype_config2)  # 断言第二个 dtype 配置已经正确设置

    # 测试设置后端操作配置的根模块
    def test_backend_op_config_set_root_module(self):
        conf = BackendPatternConfig(nni.LinearReLU)  # 创建一个线性ReLU模块的后端模式配置对象
        self.assertTrue(conf.root_module is None)  # 断言根模块当前为 None
        conf.set_root_module(torch.nn.Linear)  # 设置根模块为 torch.nn.Linear
        self.assertEqual(conf.root_module, torch.nn.Linear)  # 断言根模块已经正确设置为 torch.nn.Linear

    # 测试设置后端操作配置的量化感知训练（QAT）模块
    def test_backend_op_config_set_qat_module(self):
        conf = BackendPatternConfig(torch.nn.Linear)  # 创建一个线性层模型的后端模式配置对象
        self.assertTrue(conf.qat_module is None)  # 断言量化感知训练模块当前为 None
        conf.set_qat_module(nnqat.Linear)  # 设置量化感知训练模块为 nnqat.Linear
        self.assertEqual(conf.qat_module, nnqat.Linear)  # 断言量化感知训练模块已经正确设置为 nnqat.Linear

    # 测试设置后端操作配置的参考量化模块
    def test_backend_op_config_set_reference_quantized_module(self):
        conf = BackendPatternConfig(torch.nn.Linear)  # 创建一个线性层模型的后端模式配置对象
        self.assertTrue(conf.reference_quantized_module is None)  # 断言参考量化模块当前为 None
        conf.set_reference_quantized_module(nnqr.Linear)  # 设置参考量化模块为 nnqr.Linear
        self.assertEqual(conf.reference_quantized_module, nnqr.Linear)  # 断言参考量化模块已经正确设置为 nnqr.Linear

    # 测试设置后端操作配置的融合模块
    def test_backend_op_config_set_fused_module(self):
        conf = BackendPatternConfig((torch.nn.Linear, torch.nn.ReLU))  # 创建包含线性层和ReLU激活函数的后端模式配置对象
        self.assertTrue(conf.fused_module is None)  # 断言融合模块当前为 None
        conf.set_fused_module(nni.LinearReLU)  # 设置融合模块为 nni.LinearReLU
        self.assertEqual(conf.fused_module, nni.LinearReLU)  # 断言融合模块已经正确设置为 nni.LinearReLU

    # 测试设置后端操作配置的融合方法
    def test_backend_op_config_set_fuser_method(self):
        conf = BackendPatternConfig((torch.nn.Linear, torch.nn.ReLU))  # 创建包含线性层和ReLU激活函数的后端模式配置对象
        self.assertTrue(conf.fuser_method is None)  # 断言融合方法当前为 None
        conf.set_fuser_method(self._fuser_method)  # 设置融合方法为当前对象的 _fuser_method 方法
        self.assertEqual(conf.fuser_method, self._fuser_method)  # 断言融合方法已经正确设置为 _fuser_method
    # 测试设置后端模式配置的根节点获取器函数
    def test_backend_op_config_set_root_node_getter(self):
        # 创建一个包含线性层和ReLU激活函数的后端模式配置对象
        conf = BackendPatternConfig((torch.nn.Linear, torch.nn.ReLU))
        # 断言根节点获取器函数初始为None
        self.assertTrue(conf._root_node_getter is None)
        # 设置根节点获取器函数为默认的根节点获取器函数
        conf._set_root_node_getter(_default_root_node_getter)
        # 断言根节点获取器函数已经被正确设置
        self.assertEqual(conf._root_node_getter, _default_root_node_getter)

    # 测试设置后端模式配置的额外输入获取器函数
    def test_backend_op_config_set_extra_inputs_getter(self):
        # 创建一个包含线性层的后端模式配置对象
        conf = BackendPatternConfig(torch.nn.Linear)
        # 断言额外输入获取器函数初始为None
        self.assertTrue(conf._extra_inputs_getter is None)
        # 设置额外输入获取器函数为指定的额外输入获取器函数
        conf._set_extra_inputs_getter(self._extra_inputs_getter)
        # 断言额外输入获取器函数已经被正确设置
        self.assertEqual(conf._extra_inputs_getter, self._extra_inputs_getter)

    # 测试设置后端模式配置的张量参数数量到观察类型的映射
    def test_backend_op_config_set_num_tensor_args_to_observation_type(self):
        # 创建一个包含torch.add操作的后端模式配置对象
        conf = BackendPatternConfig(torch.add)
        # 断言张量参数数量到观察类型映射的初始长度为0
        self.assertEqual(len(conf._num_tensor_args_to_observation_type), 0)
        # 设置张量参数数量到观察类型的映射为指定的映射
        conf._set_num_tensor_args_to_observation_type(self._num_tensor_args_to_observation_type)
        # 断言张量参数数量到观察类型的映射已经被正确设置
        self.assertEqual(conf._num_tensor_args_to_observation_type, self._num_tensor_args_to_observation_type)

    # 测试设置后端模式配置的输入类型到索引的映射
    def test_backend_op_config_set_input_type_to_index(self):
        # 创建一个包含torch.addmm操作的后端模式配置对象
        conf = BackendPatternConfig(torch.addmm)
        # 断言输入类型到索引映射的初始长度为0
        self.assertEqual(len(conf._input_type_to_index), 0)
        # 设置输入类型到索引的映射为指定的映射
        conf._set_input_type_to_index(self._input_type_to_index)
        # 断言输入类型到索引的映射已经被正确设置
        self.assertEqual(conf._input_type_to_index, self._input_type_to_index)
    # 定义测试方法，用于测试从字典创建后端模式配置对象
    def test_backend_op_config_from_dict(self):
        # 获取第一个后端模式配置字典
        conf_dict1 = self._get_backend_pattern_config_dict1()
        # 使用字典创建后端模式配置对象
        conf1 = BackendPatternConfig.from_dict(conf_dict1)
        # 断言配置对象的模式为线性层和ReLU激活函数的元组
        self.assertEqual(conf1.pattern, (torch.nn.Linear, torch.nn.ReLU))
        # 断言观察类型为使用不同观察者作为输入的输出类型
        self.assertEqual(conf1.observation_type, ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        # 断言根模块为torch.nn.Linear
        self.assertEqual(conf1.root_module, torch.nn.Linear)
        # 断言量化感知训练模块为nnqat.Linear
        self.assertEqual(conf1.qat_module, nnqat.Linear)
        # 断言参考量化模块为nnqr.Linear
        self.assertEqual(conf1.reference_quantized_module, nnqr.Linear)
        # 断言融合模块为nni.LinearReLU
        self.assertEqual(conf1.fused_module, nni.LinearReLU)
        # 断言融合方法为_fuser_method属性值
        self.assertEqual(conf1.fuser_method, self._fuser_method)
        # 断言根节点获取器为None
        self.assertTrue(conf1._root_node_getter is None)
        # 断言额外输入获取器为None
        self.assertTrue(conf1._extra_inputs_getter is None)
        # 断言_num_tensor_args_to_observation_type字典长度为0
        self.assertEqual(len(conf1._num_tensor_args_to_observation_type), 0)
        # 断言_input_type_to_index字典长度为0

        self.assertEqual(len(conf1._input_type_to_index), 0)

        # 测试临时/内部键
        # 获取第二个后端模式配置字典
        conf_dict2 = self._get_backend_pattern_config_dict2()
        # 使用字典创建第二个后端模式配置对象
        conf2 = BackendPatternConfig.from_dict(conf_dict2)
        # 断言配置对象的模式为torch.add
        self.assertEqual(conf2.pattern, torch.add)
        # 断言观察类型为使用不同观察者作为输入的输出类型
        self.assertEqual(conf2.observation_type, ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        # 断言根模块为None
        self.assertTrue(conf2.root_module is None)
        # 断言量化感知训练模块为None
        self.assertTrue(conf2.qat_module is None)
        # 断言参考量化模块为None
        self.assertTrue(conf2.reference_quantized_module is None)
        # 断言融合模块为None
        self.assertTrue(conf2.fused_module is None)
        # 断言融合方法为None
        self.assertTrue(conf2.fuser_method is None)
        # 断言根节点获取器为默认的_root_node_getter
        self.assertEqual(conf2._root_node_getter, _default_root_node_getter)
        # 断言额外输入获取器为self._extra_inputs_getter
        self.assertEqual(conf2._extra_inputs_getter, self._extra_inputs_getter)
        # 断言_num_tensor_args_to_observation_type字典内容与self._num_tensor_args_to_observation_type相同
        self.assertEqual(conf2._num_tensor_args_to_observation_type, self._num_tensor_args_to_observation_type)
        # 断言_input_type_to_index字典内容与self._input_type_to_index相同

        self.assertEqual(conf2._input_type_to_index, self._input_type_to_index)

    # ===============
    #  BackendConfig
    # ===============

    # 定义测试方法，用于测试设置后端配置名称
    def test_backend_config_set_name(self):
        # 创建名称为"name1"的后端配置对象
        conf = BackendConfig("name1")
        # 断言配置对象的名称为"name1"
        self.assertEqual(conf.name, "name1")
        # 设置配置对象的名称为"name2"
        conf.set_name("name2")
        # 断言配置对象的名称为"name2"
        self.assertEqual(conf.name, "name2")
    # 测试设置后端配置的方法，验证添加模式配置后的正确性
    def test_backend_config_set_backend_pattern_config(self):
        # 创建一个后端配置对象，名为 "name1"
        conf = BackendConfig("name1")
        # 验证初始时配置列表的长度为0
        self.assertEqual(len(conf.configs), 0)
        # 获取第一个后端操作配置
        backend_op_config1 = self._get_backend_op_config1()
        # 获取第二个后端操作配置
        backend_op_config2 = self._get_backend_op_config2()
        # 设置第一个后端操作配置为模式配置
        conf.set_backend_pattern_config(backend_op_config1)
        # 验证设置后的复杂格式到配置的映射是否正确
        self.assertEqual(conf._pattern_complex_format_to_config, {
            (torch.nn.ReLU, torch.nn.Linear): backend_op_config1,
        })
        # 设置第二个后端操作配置为模式配置
        conf.set_backend_pattern_config(backend_op_config2)
        # 再次验证复杂格式到配置的映射是否正确，包括第一个和第二个配置
        self.assertEqual(conf._pattern_complex_format_to_config, {
            (torch.nn.ReLU, torch.nn.Linear): backend_op_config1,
            torch.add: backend_op_config2
        })

    # 测试从字典创建后端配置的方法
    def test_backend_config_from_dict(self):
        # 获取第一个后端操作配置
        op1 = self._get_backend_op_config1()
        # 获取第二个后端操作配置
        op2 = self._get_backend_op_config2()
        # 获取第一个后端模式配置字典
        op_dict1 = self._get_backend_pattern_config_dict1()
        # 获取第二个后端模式配置字典
        op_dict2 = self._get_backend_pattern_config_dict2()
        # 构建包含配置信息的字典
        conf_dict = {
            "name": "name1",
            "configs": [op_dict1, op_dict2],
        }
        # 根据字典创建后端配置对象
        conf = BackendConfig.from_dict(conf_dict)
        # 验证配置对象的名称是否正确
        self.assertEqual(conf.name, "name1")
        # 验证配置对象中配置列表的长度是否为2
        self.assertEqual(len(conf.configs), 2)
        # 设置第一个和第二个配置的键
        key1 = (torch.nn.ReLU, torch.nn.Linear)
        key2 = torch.add
        # 验证第一个键是否在复杂格式到配置的映射中
        self.assertTrue(key1 in conf._pattern_complex_format_to_config)
        # 验证第二个键是否在复杂格式到配置的映射中
        self.assertTrue(key2 in conf._pattern_complex_format_to_config)
        # 验证映射中第一个键对应的配置字典是否正确
        self.assertEqual(conf._pattern_complex_format_to_config[key1].to_dict(), op_dict1)
        # 验证映射中第二个键对应的配置字典是否正确
        self.assertEqual(conf._pattern_complex_format_to_config[key2].to_dict(), op_dict2)

    # 测试将后端配置对象转换为字典的方法
    def test_backend_config_to_dict(self):
        # 获取第一个后端操作配置
        op1 = self._get_backend_op_config1()
        # 获取第二个后端操作配置
        op2 = self._get_backend_op_config2()
        # 获取第一个后端模式配置字典
        op_dict1 = self._get_backend_pattern_config_dict1()
        # 获取第二个后端模式配置字典
        op_dict2 = self._get_backend_pattern_config_dict2()
        # 创建一个后端配置对象，并设置两个后端模式配置
        conf = BackendConfig("name1").set_backend_pattern_config(op1).set_backend_pattern_config(op2)
        # 预期的后端配置字典
        conf_dict = {
            "name": "name1",
            "configs": [op_dict1, op_dict2],
        }
        # 验证转换后的配置字典是否和预期一致
        self.assertEqual(conf.to_dict(), conf_dict)
if __name__ == '__main__':
    # 检查当前模块是否作为主程序直接运行
    raise RuntimeError("This _test file is not meant to be run directly, use:\n\n"
                       "\tpython _test/_test_quantization.py TESTNAME\n\n"
                       "instead.")
    # 如果是，抛出运行时错误，提示不应直接运行此测试文件，应该通过指定的方式运行
```