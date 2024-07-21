# `.\pytorch\test\quantization\ao_migration\test_quantization.py`

```
# Owner(s): ["oncall: quantization"]

from .common import AOMigrationTestCase


class TestAOMigrationQuantization(AOMigrationTestCase):
    r"""Modules and functions related to the
    `torch/quantization` migration to `torch/ao/quantization`.
    """

    def test_function_import_quantize(self):
        # 定义要测试的函数列表，用于验证 quantize 模块中的函数导入
        function_list = [
            "_convert",  # 函数：_convert
            "_observer_forward_hook",  # 函数：_observer_forward_hook
            "_propagate_qconfig_helper",  # 函数：_propagate_qconfig_helper
            "_remove_activation_post_process",  # 函数：_remove_activation_post_process
            "_remove_qconfig",  # 函数：_remove_qconfig
            "_add_observer_",  # 函数：_add_observer_
            "add_quant_dequant",  # 函数：add_quant_dequant
            "convert",  # 函数：convert
            "_get_observer_dict",  # 函数：_get_observer_dict
            "_get_unique_devices_",  # 函数：_get_unique_devices_
            "_is_activation_post_process",  # 函数：_is_activation_post_process
            "prepare",  # 函数：prepare
            "prepare_qat",  # 函数：prepare_qat
            "propagate_qconfig_",  # 函数：propagate_qconfig_
            "quantize",  # 函数：quantize
            "quantize_dynamic",  # 函数：quantize_dynamic
            "quantize_qat",  # 函数：quantize_qat
            "_register_activation_post_process_hook",  # 函数：_register_activation_post_process_hook
            "swap_module",  # 函数：swap_module
        ]
        self._test_function_import("quantize", function_list)

    def test_function_import_stubs(self):
        # 定义要测试的函数列表，用于验证 stubs 模块中的函数导入
        function_list = [
            "QuantStub",  # 类：QuantStub
            "DeQuantStub",  # 类：DeQuantStub
            "QuantWrapper",  # 类：QuantWrapper
        ]
        self._test_function_import("stubs", function_list)

    def test_function_import_quantize_jit(self):
        # 定义要测试的函数列表，用于验证 quantize_jit 模块中的函数导入
        function_list = [
            "_check_is_script_module",  # 函数：_check_is_script_module
            "_check_forward_method",  # 函数：_check_forward_method
            "script_qconfig",  # 变量：script_qconfig
            "script_qconfig_dict",  # 变量：script_qconfig_dict
            "fuse_conv_bn_jit",  # 函数：fuse_conv_bn_jit
            "_prepare_jit",  # 函数：_prepare_jit
            "prepare_jit",  # 函数：prepare_jit
            "prepare_dynamic_jit",  # 函数：prepare_dynamic_jit
            "_convert_jit",  # 函数：_convert_jit
            "convert_jit",  # 函数：convert_jit
            "convert_dynamic_jit",  # 函数：convert_dynamic_jit
            "_quantize_jit",  # 函数：_quantize_jit
            "quantize_jit",  # 函数：quantize_jit
            "quantize_dynamic_jit",  # 函数：quantize_dynamic_jit
        ]
        self._test_function_import("quantize_jit", function_list)

    def test_function_import_fake_quantize(self):
        # 定义要测试的函数列表，用于验证 fake_quantize 模块中的函数导入
        function_list = [
            "_is_per_channel",  # 函数：_is_per_channel
            "_is_per_tensor",  # 函数：_is_per_tensor
            "_is_symmetric_quant",  # 函数：_is_symmetric_quant
            "FakeQuantizeBase",  # 类：FakeQuantizeBase
            "FakeQuantize",  # 类：FakeQuantize
            "FixedQParamsFakeQuantize",  # 类：FixedQParamsFakeQuantize
            "FusedMovingAvgObsFakeQuantize",  # 类：FusedMovingAvgObsFakeQuantize
            "default_fake_quant",  # 函数：default_fake_quant
            "default_weight_fake_quant",  # 函数：default_weight_fake_quant
            "default_fixed_qparams_range_neg1to1_fake_quant",  # 函数：default_fixed_qparams_range_neg1to1_fake_quant
            "default_fixed_qparams_range_0to1_fake_quant",  # 函数：default_fixed_qparams_range_0to1_fake_quant
            "default_per_channel_weight_fake_quant",  # 函数：default_per_channel_weight_fake_quant
            "default_histogram_fake_quant",  # 函数：default_histogram_fake_quant
            "default_fused_act_fake_quant",  # 函数：default_fused_act_fake_quant
            "default_fused_wt_fake_quant",  # 函数：default_fused_wt_fake_quant
            "default_fused_per_channel_wt_fake_quant",  # 函数：default_fused_per_channel_wt_fake_quant
            "_is_fake_quant_script_module",  # 函数：_is_fake_quant_script_module
            "disable_fake_quant",  # 函数：disable_fake_quant
            "enable_fake_quant",  # 函数：enable_fake_quant
            "disable_observer",  # 函数：disable_observer
            "enable_observer",  # 函数：enable_observer
        ]
        self._test_function_import("fake_quantize", function_list)
    # 测试导入函数列表，用于测试是否成功导入了指定模块中的函数
    def test_function_import_fuse_modules(self):
        # 定义要测试导入的函数列表
        function_list = [
            "_fuse_modules",
            "_get_module",
            "_set_module",
            "fuse_conv_bn",
            "fuse_conv_bn_relu",
            "fuse_known_modules",
            "fuse_modules",
            "get_fuser_method",
        ]
        # 调用测试导入函数，传入模块名和函数列表
        self._test_function_import("fuse_modules", function_list)

    # 测试导入量化类型相关的函数
    def test_function_import_quant_type(self):
        # 定义要测试导入的函数列表
        function_list = [
            "QuantType",
            "_get_quant_type_to_str",
        ]
        # 调用测试导入函数，传入模块名和函数列表
        self._test_function_import("quant_type", function_list)

    # 测试导入观察者模块相关的函数
    def test_function_import_observer(self):
        # 定义要测试导入的函数列表
        function_list = [
            "_PartialWrapper",
            "_with_args",
            "_with_callable_args",
            "ABC",
            "ObserverBase",
            "_ObserverBase",
            "MinMaxObserver",
            "MovingAverageMinMaxObserver",
            "PerChannelMinMaxObserver",
            "MovingAveragePerChannelMinMaxObserver",
            "HistogramObserver",
            "PlaceholderObserver",
            "RecordingObserver",
            "NoopObserver",
            "_is_activation_post_process",
            "_is_per_channel_script_obs_instance",
            "get_observer_state_dict",
            "load_observer_state_dict",
            "default_observer",
            "default_placeholder_observer",
            "default_debug_observer",
            "default_weight_observer",
            "default_histogram_observer",
            "default_per_channel_weight_observer",
            "default_dynamic_quant_observer",
            "default_float_qparams_observer",
        ]
        # 调用测试导入函数，传入模块名和函数列表
        self._test_function_import("observer", function_list)

    # 测试导入量化配置相关的函数
    def test_function_import_qconfig(self):
        # 定义要测试导入的函数列表
        function_list = [
            "QConfig",
            "default_qconfig",
            "default_debug_qconfig",
            "default_per_channel_qconfig",
            "QConfigDynamic",
            "default_dynamic_qconfig",
            "float16_dynamic_qconfig",
            "float16_static_qconfig",
            "per_channel_dynamic_qconfig",
            "float_qparams_weight_only_qconfig",
            "default_qat_qconfig",
            "default_weight_only_qconfig",
            "default_activation_only_qconfig",
            "default_qat_qconfig_v2",
            "get_default_qconfig",
            "get_default_qat_qconfig",
            "_assert_valid_qconfig",
            "QConfigAny",
            "_add_module_to_qconfig_obs_ctr",
            "qconfig_equals",
        ]
        # 调用测试导入函数，传入模块名和函数列表
        self._test_function_import("qconfig", function_list)
    # 定义一个测试方法，用于导入和测试量化映射函数模块中的函数列表
    def test_function_import_quantization_mappings(self):
        # 函数列表，包含需要测试的函数名称
        function_list = [
            "no_observer_set",
            "get_default_static_quant_module_mappings",
            "get_static_quant_module_class",
            "get_dynamic_quant_module_class",
            "get_default_qat_module_mappings",
            "get_default_dynamic_quant_module_mappings",
            "get_default_qconfig_propagation_list",
            "get_default_compare_output_module_list",
            "get_default_float_to_quantized_operator_mappings",
            "get_quantized_operator",
            "_get_special_act_post_process",
            "_has_special_act_post_process",
        ]
        # 字典列表，包含需要导入和测试的字典名称
        dict_list = [
            "DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS",
            "DEFAULT_STATIC_QUANT_MODULE_MAPPINGS",
            "DEFAULT_QAT_MODULE_MAPPINGS",
            "DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS",
            # "_INCLUDE_QCONFIG_PROPAGATE_LIST",
            "DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS",
            "DEFAULT_MODULE_TO_ACT_POST_PROCESS",
        ]
        # 调用类内方法，测试导入的函数列表是否正确
        self._test_function_import("quantization_mappings", function_list)
        # 调用类内方法，测试导入的字典列表是否正确
        self._test_dict_import("quantization_mappings", dict_list)

    # 定义一个测试方法，用于导入和测试融合方法映射函数模块中的函数列表
    def test_function_import_fuser_method_mappings(self):
        # 函数列表，包含需要测试的函数名称
        function_list = [
            "fuse_conv_bn",
            "fuse_conv_bn_relu",
            "fuse_linear_bn",
            "get_fuser_method",
        ]
        # 字典列表，包含需要导入和测试的字典名称
        dict_list = ["_DEFAULT_OP_LIST_TO_FUSER_METHOD"]
        # 调用类内方法，测试导入的函数列表是否正确
        self._test_function_import("fuser_method_mappings", function_list)
        # 调用类内方法，测试导入的字典列表是否正确
        self._test_dict_import("fuser_method_mappings", dict_list)

    # 定义一个测试方法，用于导入和测试实用函数模块中的函数列表
    def test_function_import_utils(self):
        # 函数列表，包含需要测试的函数名称
        function_list = [
            "activation_dtype",
            "activation_is_int8_quantized",
            "activation_is_statically_quantized",
            "calculate_qmin_qmax",
            "check_min_max_valid",
            "get_combined_dict",
            "get_qconfig_dtypes",
            "get_qparam_dict",
            "get_quant_type",
            "get_swapped_custom_module_class",
            "getattr_from_fqn",
            "is_per_channel",
            "is_per_tensor",
            "weight_dtype",
            "weight_is_quantized",
            "weight_is_statically_quantized",
        ]
        # 调用类内方法，测试导入的函数列表是否正确
        self._test_function_import("utils", function_list)
```