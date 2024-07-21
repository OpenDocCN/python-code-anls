# `.\pytorch\test\quantization\ao_migration\test_ao_migration.py`

```py
# Owner(s): ["oncall: quantization"]

from .common import AOMigrationTestCase  # 导入AOMigrationTestCase类，来自common模块


class TestAOMigrationNNQuantized(AOMigrationTestCase):
    def test_functional_import(self):
        r"""Tests the migration of the torch.nn.quantized.functional"""
        # 测试torch.nn.quantized.functional的迁移
        function_list = [
            "avg_pool2d",
            "avg_pool3d",
            "adaptive_avg_pool2d",
            "adaptive_avg_pool3d",
            "conv1d",
            "conv2d",
            "conv3d",
            "interpolate",
            "linear",
            "max_pool1d",
            "max_pool2d",
            "celu",
            "leaky_relu",
            "hardtanh",
            "hardswish",
            "threshold",
            "elu",
            "hardsigmoid",
            "clamp",
            "upsample",
            "upsample_bilinear",
            "upsample_nearest",
        ]
        self._test_function_import("functional", function_list, base="nn.quantized")

    def test_modules_import(self):
        module_list = [
            # Modules
            "BatchNorm2d",
            "BatchNorm3d",
            "Conv1d",
            "Conv2d",
            "Conv3d",
            "ConvTranspose1d",
            "ConvTranspose2d",
            "ConvTranspose3d",
            "DeQuantize",
            "ELU",
            "Embedding",
            "EmbeddingBag",
            "GroupNorm",
            "Hardswish",
            "InstanceNorm1d",
            "InstanceNorm2d",
            "InstanceNorm3d",
            "LayerNorm",
            "LeakyReLU",
            "Linear",
            "MaxPool2d",
            "Quantize",
            "ReLU6",
            "Sigmoid",
            "Softmax",
            "Dropout",
            # Wrapper modules
            "FloatFunctional",
            "FXFloatFunctional",
            "QFunctional",
        ]
        self._test_function_import("modules", module_list, base="nn.quantized")

    def test_modules_activation(self):
        function_list = [
            "ReLU6",
            "Hardswish",
            "ELU",
            "LeakyReLU",
            "Sigmoid",
            "Softmax",
        ]
        self._test_function_import(
            "activation", function_list, base="nn.quantized.modules"
        )

    def test_modules_batchnorm(self):
        function_list = [
            "BatchNorm2d",
            "BatchNorm3d",
        ]
        self._test_function_import(
            "batchnorm", function_list, base="nn.quantized.modules"
        )

    def test_modules_conv(self):
        function_list = [
            "_reverse_repeat_padding",
            "Conv1d",
            "Conv2d",
            "Conv3d",
            "ConvTranspose1d",
            "ConvTranspose2d",
            "ConvTranspose3d",
        ]
        self._test_function_import("conv", function_list, base="nn.quantized.modules")
    # 测试模块的Dropout函数导入情况
    def test_modules_dropout(self):
        # 函数列表包含"Dropout"
        function_list = [
            "Dropout",
        ]
        # 调用测试函数，验证dropout模块中的函数导入情况
        self._test_function_import(
            "dropout", function_list, base="nn.quantized.modules"
        )

    # 测试模块的嵌入操作函数导入情况
    def test_modules_embedding_ops(self):
        # 函数列表包含"EmbeddingPackedParams", "Embedding", "EmbeddingBag"
        function_list = [
            "EmbeddingPackedParams",
            "Embedding",
            "EmbeddingBag",
        ]
        # 调用测试函数，验证embedding_ops模块中的函数导入情况
        self._test_function_import(
            "embedding_ops", function_list, base="nn.quantized.modules"
        )

    # 测试模块的功能模块函数导入情况
    def test_modules_functional_modules(self):
        # 函数列表包含"FloatFunctional", "FXFloatFunctional", "QFunctional"
        function_list = [
            "FloatFunctional",
            "FXFloatFunctional",
            "QFunctional",
        ]
        # 调用测试函数，验证functional_modules模块中的函数导入情况
        self._test_function_import(
            "functional_modules", function_list, base="nn.quantized.modules"
        )

    # 测试模块的线性操作函数导入情况
    def test_modules_linear(self):
        # 函数列表包含"Linear", "LinearPackedParams"
        function_list = [
            "Linear",
            "LinearPackedParams",
        ]
        # 调用测试函数，验证linear模块中的函数导入情况
        self._test_function_import("linear", function_list, base="nn.quantized.modules")

    # 测试模块的归一化操作函数导入情况
    def test_modules_normalization(self):
        # 函数列表包含"LayerNorm", "GroupNorm", "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d"
        function_list = [
            "LayerNorm",
            "GroupNorm",
            "InstanceNorm1d",
            "InstanceNorm2d",
            "InstanceNorm3d",
        ]
        # 调用测试函数，验证normalization模块中的函数导入情况
        self._test_function_import(
            "normalization", function_list, base="nn.quantized.modules"
        )

    # 测试模块的工具函数导入情况
    def test_modules_utils(self):
        # 函数列表包含"_ntuple_from_first", "_pair_from_first", "_quantize_weight", "_hide_packed_params_repr", "WeightedQuantizedModule"
        function_list = [
            "_ntuple_from_first",
            "_pair_from_first",
            "_quantize_weight",
            "_hide_packed_params_repr",
            "WeightedQuantizedModule",
        ]
        # 调用测试函数，验证utils模块中的函数导入情况
        self._test_function_import("utils", function_list, base="nn.quantized.modules")

    # 测试nn.quantized动态导入模块函数导入情况
    def test_import_nn_quantized_dynamic_import(self):
        module_list = [
            # 模块列表包含线性、循环神经网络等
            "Linear",
            "LSTM",
            "GRU",
            "LSTMCell",
            "RNNCell",
            "GRUCell",
            "Conv1d",
            "Conv2d",
            "Conv3d",
            "ConvTranspose1d",
            "ConvTranspose2d",
            "ConvTranspose3d",
        ]
        # 调用测试函数，验证nn.quantized.dynamic模块中的动态导入情况
        self._test_function_import("dynamic", module_list, base="nn.quantized")

    # 测试nn.quantizable.activation模块函数导入情况
    def test_import_nn_quantizable_activation(self):
        module_list = [
            # 模块列表包含"MultiheadAttention"
            "MultiheadAttention",
        ]
        # 调用测试函数，验证nn.quantizable.activation模块中的函数导入情况
        self._test_function_import(
            "activation", module_list, base="nn.quantizable.modules"
        )

    # 测试nn.quantizable.rnn模块函数导入情况
    def test_import_nn_quantizable_rnn(self):
        module_list = [
            # 模块列表包含"LSTM", "LSTMCell"
            "LSTM",
            "LSTMCell",
        ]
        # 调用测试函数，验证nn.quantizable.rnn模块中的函数导入情况
        self._test_function_import("rnn", module_list, base="nn.quantizable.modules")

    # 测试nn.qat.conv模块函数导入情况
    def test_import_nn_qat_conv(self):
        module_list = [
            # 模块列表包含"Conv1d", "Conv2d", "Conv3d"
            "Conv1d",
            "Conv2d",
            "Conv3d",
        ]
        # 调用测试函数，验证nn.qat.conv模块中的函数导入情况
        self._test_function_import("conv", module_list, base="nn.qat.modules")
    # 测试导入 nn.qat.embedding_ops 模块中的函数
    def test_import_nn_qat_embedding_ops(self):
        # 定义包含需要测试的模块名称列表
        module_list = [
            "Embedding",
            "EmbeddingBag",
        ]
        # 调用测试函数，检查函数是否成功导入
        self._test_function_import("embedding_ops", module_list, base="nn.qat.modules")
    
    # 测试导入 nn.qat.linear 模块中的函数
    def test_import_nn_qat_linear(self):
        # 定义包含需要测试的模块名称列表
        module_list = [
            "Linear",
        ]
        # 调用测试函数，检查函数是否成功导入
        self._test_function_import("linear", module_list, base="nn.qat.modules")
    
    # 测试导入 nn.qat.dynamic.linear 模块中的函数
    def test_import_nn_qat_dynamic_linear(self):
        # 定义包含需要测试的模块名称列表
        module_list = [
            "Linear",
        ]
        # 调用测试函数，检查函数是否成功导入
        self._test_function_import("linear", module_list, base="nn.qat.dynamic.modules")
class TestAOMigrationNNIntrinsic(AOMigrationTestCase):
    # 定义测试类 TestAOMigrationNNIntrinsic，继承自 AOMigrationTestCase

    def test_modules_import_nn_intrinsic(self):
        # 定义测试方法 test_modules_import_nn_intrinsic，测试导入 nn.intrinsic 模块的功能
        module_list = [
            # 定义要测试导入的模块列表
            "_FusedModule",
            "ConvBn1d",
            "ConvBn2d",
            "ConvBn3d",
            "ConvBnReLU1d",
            "ConvBnReLU2d",
            "ConvBnReLU3d",
            "ConvReLU1d",
            "ConvReLU2d",
            "ConvReLU3d",
            "LinearReLU",
            "BNReLU2d",
            "BNReLU3d",
            "LinearBn1d",
        ]
        # 调用私有方法 _test_function_import，测试函数导入情况，base 参数为 nn
        self._test_function_import("intrinsic", module_list, base="nn")

    def test_modules_nn_intrinsic_fused(self):
        # 定义测试方法 test_modules_nn_intrinsic_fused，测试 nn.intrinsic.modules 下的融合函数的导入
        function_list = [
            "_FusedModule",
            "ConvBn1d",
            "ConvBn2d",
            "ConvBn3d",
            "ConvBnReLU1d",
            "ConvBnReLU2d",
            "ConvBnReLU3d",
            "ConvReLU1d",
            "ConvReLU2d",
            "ConvReLU3d",
            "LinearReLU",
            "BNReLU2d",
            "BNReLU3d",
            "LinearBn1d",
        ]
        # 调用私有方法 _test_function_import，测试函数导入情况，base 参数为 nn.intrinsic.modules
        self._test_function_import("fused", function_list, base="nn.intrinsic.modules")

    def test_modules_import_nn_intrinsic_qat(self):
        # 定义测试方法 test_modules_import_nn_intrinsic_qat，测试 nn.intrinsic 下的量化训练模块的导入
        module_list = [
            "LinearReLU",
            "LinearBn1d",
            "ConvReLU1d",
            "ConvReLU2d",
            "ConvReLU3d",
            "ConvBn1d",
            "ConvBn2d",
            "ConvBn3d",
            "ConvBnReLU1d",
            "ConvBnReLU2d",
            "ConvBnReLU3d",
            "update_bn_stats",
            "freeze_bn_stats",
        ]
        # 调用私有方法 _test_function_import，测试函数导入情况，base 参数为 nn.intrinsic
        self._test_function_import("qat", module_list, base="nn.intrinsic")

    def test_modules_intrinsic_qat_conv_fused(self):
        # 定义测试方法 test_modules_intrinsic_qat_conv_fused，测试 nn.intrinsic.qat.modules 下卷积融合函数的导入
        function_list = [
            "ConvBn1d",
            "ConvBnReLU1d",
            "ConvReLU1d",
            "ConvBn2d",
            "ConvBnReLU2d",
            "ConvReLU2d",
            "ConvBn3d",
            "ConvBnReLU3d",
            "ConvReLU3d",
            "update_bn_stats",
            "freeze_bn_stats",
        ]
        # 调用私有方法 _test_function_import，测试函数导入情况，base 参数为 nn.intrinsic.qat.modules
        self._test_function_import(
            "conv_fused", function_list, base="nn.intrinsic.qat.modules"
        )

    def test_modules_intrinsic_qat_linear_fused(self):
        # 定义测试方法 test_modules_intrinsic_qat_linear_fused，测试 nn.intrinsic.qat.modules 下线性融合函数的导入
        function_list = [
            "LinearBn1d",
        ]
        # 调用私有方法 _test_function_import，测试函数导入情况，base 参数为 nn.intrinsic.qat.modules
        self._test_function_import(
            "linear_fused", function_list, base="nn.intrinsic.qat.modules"
        )

    def test_modules_intrinsic_qat_linear_relu(self):
        # 定义测试方法 test_modules_intrinsic_qat_linear_relu，测试 nn.intrinsic.qat.modules 下线性 ReLU 函数的导入
        function_list = [
            "LinearReLU",
        ]
        # 调用私有方法 _test_function_import，测试函数导入情况，base 参数为 nn.intrinsic.qat.modules
        self._test_function_import(
            "linear_relu", function_list, base="nn.intrinsic.qat.modules"
        )

    def test_modules_import_nn_intrinsic_quantized(self):
        # 定义测试方法 test_modules_import_nn_intrinsic_quantized，测试 nn.intrinsic 下量化模块的导入
        module_list = [
            "BNReLU2d",
            "BNReLU3d",
            "ConvReLU1d",
            "ConvReLU2d",
            "ConvReLU3d",
            "LinearReLU",
        ]
        # 调用私有方法 _test_function_import，测试函数导入情况，base 参数为 nn.intrinsic
        self._test_function_import("quantized", module_list, base="nn.intrinsic")
    # 定义一个测试方法，用于测试量化（quantized）模块中带有 BNReLU 的函数导入情况
    def test_modules_intrinsic_quantized_bn_relu(self):
        # 定义包含函数名称的列表
        function_list = [
            "BNReLU2d",
            "BNReLU3d",
        ]
        # 调用测试函数 _test_function_import，测试导入 bn_relu 函数列表中的函数，
        # 导入基础路径为 nn.intrinsic.quantized.modules
        self._test_function_import(
            "bn_relu", function_list, base="nn.intrinsic.quantized.modules"
        )
    
    # 定义一个测试方法，用于测试量化（quantized）模块中带有 ConvReLU 的函数导入情况
    def test_modules_intrinsic_quantized_conv_relu(self):
        # 定义包含函数名称的列表
        function_list = [
            "ConvReLU1d",
            "ConvReLU2d",
            "ConvReLU3d",
        ]
        # 调用测试函数 _test_function_import，测试导入 conv_relu 函数列表中的函数，
        # 导入基础路径为 nn.intrinsic.quantized.modules
        self._test_function_import(
            "conv_relu", function_list, base="nn.intrinsic.quantized.modules"
        )
    
    # 定义一个测试方法，用于测试量化（quantized）模块中带有 LinearReLU 的函数导入情况
    def test_modules_intrinsic_quantized_linear_relu(self):
        # 定义包含函数名称的列表
        function_list = [
            "LinearReLU",
        ]
        # 调用测试函数 _test_function_import，测试导入 linear_relu 函数列表中的函数，
        # 导入基础路径为 nn.intrinsic.quantized.modules
        self._test_function_import(
            "linear_relu", function_list, base="nn.intrinsic.quantized.modules"
        )
    
    # 定义一个测试方法，用于验证在未来的 Pull Request 中一般化这个测试
    def test_modules_no_import_nn_intrinsic_quantized_dynamic(self):
        # 导入 torch 模块
        import torch
    
        # 尝试获取 torch.ao.nn.intrinsic.quantized.dynamic 的引用
        _ = torch.ao.nn.intrinsic.quantized.dynamic
        # 尝试获取 torch.nn.intrinsic.quantized.dynamic 的引用
        _ = torch.nn.intrinsic.quantized.dynamic
```