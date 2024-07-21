# `.\pytorch\test\quantization\ao_migration\test_quantization_fx.py`

```
# Owner(s): ["oncall: quantization"]

# 引入 AOMigrationTestCase 类，用于测试迁移情况
from .common import AOMigrationTestCase

# TestAOMigrationQuantizationFx 类，继承自 AOMigrationTestCase 类，用于测试量化功能迁移情况
class TestAOMigrationQuantizationFx(AOMigrationTestCase):

    # 测试 quantize_fx 函数的导入情况
    def test_function_import_quantize_fx(self):
        # 函数名称列表，包含待测试的函数名
        function_list = [
            "_check_is_graph_module",
            "_swap_ff_with_fxff",
            "_fuse_fx",
            "QuantizationTracer",
            "_prepare_fx",
            "_prepare_standalone_module_fx",
            "fuse_fx",
            "Scope",
            "ScopeContextManager",
            "prepare_fx",
            "prepare_qat_fx",
            "_convert_fx",
            "convert_fx",
            "_convert_standalone_module_fx",
        ]
        # 调用父类方法进行函数导入测试
        self._test_function_import("quantize_fx", function_list)

    # 测试 fx 函数的导入情况
    def test_function_import_fx(self):
        # 函数名称列表，包含待测试的函数名
        function_list = [
            "prepare",
            "convert",
            "fuse",
        ]
        # 调用父类方法进行函数导入测试
        self._test_function_import("fx", function_list)

    # 测试 fx.graph_module 函数的导入情况
    def test_function_import_fx_graph_module(self):
        # 函数名称列表，包含待测试的函数名
        function_list = [
            "FusedGraphModule",
            "ObservedGraphModule",
            "_is_observed_module",
            "ObservedStandaloneGraphModule",
            "_is_observed_standalone_module",
            "QuantizedGraphModule",
        ]
        # 调用父类方法进行函数导入测试
        self._test_function_import("fx.graph_module", function_list)

    # 测试 fx.pattern_utils 函数的导入情况
    def test_function_import_fx_pattern_utils(self):
        # 函数名称列表，包含待测试的函数名
        function_list = [
            "QuantizeHandler",
            "_register_fusion_pattern",
            "get_default_fusion_patterns",
            "_register_quant_pattern",
            "get_default_quant_patterns",
            "get_default_output_activation_post_process_map",
        ]
        # 调用父类方法进行函数导入测试
        self._test_function_import("fx.pattern_utils", function_list)

    # 测试 fx._equalize 函数的导入情况
    def test_function_import_fx_equalize(self):
        # 函数名称列表，包含待测试的函数名
        function_list = [
            "reshape_scale",
            "_InputEqualizationObserver",
            "_WeightEqualizationObserver",
            "calculate_equalization_scale",
            "EqualizationQConfig",
            "input_equalization_observer",
            "weight_equalization_observer",
            "default_equalization_qconfig",
            "fused_module_supports_equalization",
            "nn_module_supports_equalization",
            "node_supports_equalization",
            "is_equalization_observer",
            "get_op_node_and_weight_eq_obs",
            "maybe_get_weight_eq_obs_node",
            "maybe_get_next_input_eq_obs",
            "maybe_get_next_equalization_scale",
            "scale_input_observer",
            "scale_weight_node",
            "scale_weight_functional",
            "clear_weight_quant_obs_node",
            "remove_node",
            "update_obs_for_equalization",
            "convert_eq_obs",
            "_convert_equalization_ref",
            "get_layer_sqnr_dict",
            "get_equalization_qconfig_dict",
        ]
        # 调用父类方法进行函数导入测试
        self._test_function_import("fx._equalize", function_list)
    # 测试函数：test_function_import_fx_quantization_patterns
    def test_function_import_fx_quantization_patterns(self):
        # 函数列表，包含要导入的函数名称
        function_list = [
            "QuantizeHandler",
            "BinaryOpQuantizeHandler",
            "CatQuantizeHandler",
            "ConvReluQuantizeHandler",
            "LinearReLUQuantizeHandler",
            "BatchNormQuantizeHandler",
            "EmbeddingQuantizeHandler",
            "RNNDynamicQuantizeHandler",
            "DefaultNodeQuantizeHandler",
            "FixedQParamsOpQuantizeHandler",
            "CopyNodeQuantizeHandler",
            "CustomModuleQuantizeHandler",
            "GeneralTensorShapeOpQuantizeHandler",
            "StandaloneModuleQuantizeHandler",
        ]
        # 调用测试函数_import，导入指定模块的函数列表
        self._test_function_import(
            "fx.quantization_patterns",  # 模块路径：fx.quantization_patterns
            function_list,  # 要导入的函数列表
            new_package_name="fx.quantize_handler",  # 新的包名称
        )
    
    # 测试函数：test_function_import_fx_match_utils
    def test_function_import_fx_match_utils(self):
        # 函数列表，包含要导入的函数名称
        function_list = ["_MatchResult", "MatchAllNode", "_is_match", "_find_matches"]
        # 调用测试函数_import，导入指定模块的函数列表
        self._test_function_import("fx.match_utils", function_list)
    
    # 测试函数：test_function_import_fx_prepare
    def test_function_import_fx_prepare(self):
        # 函数列表，包含要导入的函数名称
        function_list = ["prepare"]
        # 调用测试函数_import，导入指定模块的函数列表
        self._test_function_import("fx.prepare", function_list)
    
    # 测试函数：test_function_import_fx_convert
    def test_function_import_fx_convert(self):
        # 函数列表，包含要导入的函数名称
        function_list = ["convert"]
        # 调用测试函数_import，导入指定模块的函数列表
        self._test_function_import("fx.convert", function_list)
    
    # 测试函数：test_function_import_fx_fuse
    def test_function_import_fx_fuse(self):
        # 函数列表，包含要导入的函数名称
        function_list = ["fuse"]
        # 调用测试函数_import，导入指定模块的函数列表
        self._test_function_import("fx.fuse", function_list)
    
    # 测试函数：test_function_import_fx_fusion_patterns
    def test_function_import_fx_fusion_patterns(self):
        # 函数列表，包含要导入的函数名称
        function_list = ["FuseHandler", "DefaultFuseHandler"]
        # 调用测试函数_import，导入指定模块的函数列表
        self._test_function_import(
            "fx.fusion_patterns",  # 模块路径：fx.fusion_patterns
            function_list,  # 要导入的函数列表
            new_package_name="fx.fuse_handler",  # 新的包名称
        )
    
    # 测试函数：test_function_import_fx_utils
    def test_function_import_fx_utils(self):
        # 函数列表，包含要导入的函数名称
        function_list = [
            "get_custom_module_class_keys",
            "get_linear_prepack_op_for_dtype",
            "get_qconv_prepack_op",
            "get_new_attr_name_with_prefix",
            "graph_module_from_producer_nodes",
            "assert_and_get_unique_device",
            "create_getattr_from_value",
            "all_node_args_have_no_tensors",
            "get_non_observable_arg_indexes_and_types",
            "maybe_get_next_module",
        ]
        # 调用测试函数_import，导入指定模块的函数列表
        self._test_function_import("fx.utils", function_list)
```