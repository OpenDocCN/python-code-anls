# `.\pytorch\test\quantization\eager\test_numeric_suite_eager.py`

```py
# Owner(s): ["oncall: quantization"]

# 引入单元测试模块
import unittest
# 引入 PyTorch 库
import torch
import torch.nn as nn
# 引入量化后的神经网络模块
import torch.ao.nn.quantized as nnq
# 引入量化相关操作
from torch.ao.quantization import (
    DeQuantStub,
    QuantStub,
    convert,
    default_qconfig,
    prepare,
    quantize,
    quantize_dynamic,
)
# 引入数值套件相关模块
from torch.ao.ns._numeric_suite import (
    OutputLogger,
    Shadow,
    ShadowLogger,
    compare_model_outputs,
    compare_model_stub,
    compare_weights,
    prepare_model_outputs,
    get_matching_activations,
)
# 引入通用量化测试相关模块和类
from torch.testing._internal.common_quantization import (
    AnnotatedConvBnReLUModel,
    AnnotatedConvModel,
    AnnotatedConvTransposeModel,
    AnnotatedSingleLayerLinearModel,
    LSTMwithHiddenDynamicModel,
    AnnotatedTwoLayerLinearModel,
    QuantizationTestCase,
    SingleLayerLinearDynamicModel,
    test_only_eval_fn,
    skip_if_no_torchvision,
)
# 引入通用量化测试相关方法
from torch.testing._internal.common_quantized import override_qengines
# 引入通用工具方法
from torch.testing._internal.common_utils import IS_ARM64

# 定义子模块类，继承自 torch.nn.Module
class SubModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 设置默认量化配置
        self.qconfig = default_qconfig
        # 创建一个没有偏置的二维卷积层，输入通道数、输出通道数、卷积核大小
        self.mod1 = torch.nn.Conv2d(3, 3, 3, bias=False).to(dtype=torch.float)
        # 创建一个 ReLU 激活层
        self.mod2 = nn.ReLU()
        # 创建量化和反量化存根
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    # 前向传播函数
    def forward(self, x):
        # 对输入 x 进行量化
        x = self.quant(x)
        # 经过 mod1 进行卷积操作
        x = self.mod1(x)
        # 经过 mod2 进行 ReLU 激活
        x = self.mod2(x)
        # 对输出 x 进行反量化
        x = self.dequant(x)
        # 返回处理后的 x
        return x


# 定义带子模块的模型类，继承自 torch.nn.Module
class ModelWithSubModules(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 创建一个 SubModule 实例
        self.mod1 = SubModule()
        # 创建一个带偏置的二维卷积层，输入通道数 3，输出通道数 5，卷积核大小 3
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)

    # 前向传播函数
    def forward(self, x):
        # 通过 mod1 处理输入 x
        x = self.mod1(x)
        # 经过 conv 进行卷积操作
        x = self.conv(x)
        # 返回处理后的 x
        return x


# 定义带功能模块的模型类，继承自 torch.nn.Module
class ModelWithFunctionals(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 创建 FloatFunctional 实例用于各种功能操作
        self.mycat = nnq.FloatFunctional()
        self.myadd = nnq.FloatFunctional()
        self.mymul = nnq.FloatFunctional()
        self.myadd_relu = nnq.FloatFunctional()
        self.my_scalar_add = nnq.FloatFunctional()
        self.my_scalar_mul = nnq.FloatFunctional()
        # 创建量化和反量化存根
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    # 前向传播函数
    def forward(self, x):
        # 对输入 x 进行量化
        x = self.quant(x)
        # 使用 mycat 进行张量拼接操作
        x = self.mycat.cat([x, x, x])
        # 使用 myadd 进行张量相加操作
        x = self.myadd.add(x, x)
        # 使用 mymul 进行张量相乘操作
        x = self.mymul.mul(x, x)
        # 使用 myadd_relu 进行张量相加并应用 ReLU 激活操作
        x = self.myadd_relu.add_relu(x, x)
        # 使用 my_scalar_add 对张量加上标量 -0.5
        w = self.my_scalar_add.add_scalar(x, -0.5)
        # 使用 my_scalar_mul 对张量乘上标量 0.5
        w = self.my_scalar_mul.mul_scalar(w, 0.5)
        # 对输出 w 进行反量化
        w = self.dequant(w)
        # 返回处理后的 w
        return w


# 定义测试数值套件的测试类，继承自 QuantizationTestCase
class TestNumericSuiteEager(QuantizationTestCase):
    @override_qengines
    def test_compare_weights_conv_static(self):
        r"""Compare the weights of float and static quantized conv layer"""

        qengine = torch.backends.quantized.engine  # 获取当前的量化引擎

        def compare_and_validate_results(float_model, q_model):
            weight_dict = compare_weights(
                float_model.state_dict(), q_model.state_dict()
            )  # 比较浮点模型和量化模型的权重
            self.assertEqual(len(weight_dict), 1)  # 断言权重字典长度为1
            for v in weight_dict.values():
                self.assertTrue(v["float"].shape == v["quantized"].shape)  # 断言浮点权重和量化权重的形状一致

        model_list = [AnnotatedConvModel(qengine), AnnotatedConvBnReLUModel(qengine)]
        for model in model_list:
            model.eval()  # 将模型设置为评估模式
            if hasattr(model, "fuse_model"):
                model.fuse_model()  # 若有融合模型方法，则执行融合模型
            q_model = quantize(model, test_only_eval_fn, [self.img_data_2d])  # 对模型进行量化，仅用于评估
            compare_and_validate_results(model, q_model)  # 比较并验证权重

    @override_qengines
    def test_compare_weights_linear_static(self):
        r"""Compare the weights of float and static quantized linear layer"""

        qengine = torch.backends.quantized.engine  # 获取当前的量化引擎

        def compare_and_validate_results(float_model, q_model):
            weight_dict = compare_weights(
                float_model.state_dict(), q_model.state_dict()
            )  # 比较浮点模型和量化模型的权重
            self.assertEqual(len(weight_dict), 1)  # 断言权重字典长度为1
            for v in weight_dict.values():
                self.assertTrue(v["float"].shape == v["quantized"].shape)  # 断言浮点权重和量化权重的形状一致

        model_list = [AnnotatedSingleLayerLinearModel(qengine)]
        for model in model_list:
            model.eval()  # 将模型设置为评估模式
            if hasattr(model, "fuse_model"):
                model.fuse_model()  # 若有融合模型方法，则执行融合模型
            q_model = quantize(model, test_only_eval_fn, [self.calib_data])  # 对模型进行量化，用于评估和校准
            compare_and_validate_results(model, q_model)  # 比较并验证权重

    @override_qengines
    def test_compare_weights_linear_dynamic(self):
        r"""Compare the weights of float and dynamic quantized linear layer"""

        qengine = torch.backends.quantized.engine  # 获取当前的量化引擎

        def compare_and_validate_results(float_model, q_model):
            weight_dict = compare_weights(
                float_model.state_dict(), q_model.state_dict()
            )  # 比较浮点模型和动态量化模型的权重
            self.assertEqual(len(weight_dict), 1)  # 断言权重字典长度为1
            for v in weight_dict.values():
                self.assertTrue(len(v["float"]) == len(v["quantized"]))  # 断言浮点权重和量化权重的数量一致
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)  # 断言每个权重张量的形状一致

        model_list = [SingleLayerLinearDynamicModel(qengine)]
        for model in model_list:
            model.eval()  # 将模型设置为评估模式
            if hasattr(model, "fuse_model"):
                model.fuse_model()  # 若有融合模型方法，则执行融合模型
            q_model = quantize_dynamic(model)  # 对模型进行动态量化
            compare_and_validate_results(model, q_model)  # 比较并验证权重

    @override_qengines
    def test_compare_weights_lstm_dynamic(self):
        r"""Compare the weights of float and dynamic quantized LSTM layer"""

        # 获取当前量化引擎
        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model):
            # 比较和验证浮点和量化模型的权重结果
            weight_dict = compare_weights(
                float_model.state_dict(), q_model.state_dict()
            )
            # 断言结果字典长度为1
            self.assertEqual(len(weight_dict), 1)
            for v in weight_dict.values():
                # 断言浮点权重和量化权重列表长度相等
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    # 断言每对浮点和量化权重的形状相同
                    self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)

        # 创建包含动态量化 LSTM 模型的列表
        model_list = [LSTMwithHiddenDynamicModel(qengine)]
        for model in model_list:
            model.eval()
            # 如果模型有融合模型方法，执行融合操作
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            # 对模型进行动态量化
            q_model = quantize_dynamic(model)
            # 比较和验证模型权重
            compare_and_validate_results(model, q_model)

    @override_qengines
    def test_compare_model_stub_conv_static(self):
        r"""Compare the output of static quantized conv layer and its float shadow module"""

        # 获取当前量化引擎
        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model, module_swap_list, data):
            # 比较和验证静态量化卷积层和其浮点阴影模块的输出
            ob_dict = compare_model_stub(float_model, q_model, module_swap_list, data)
            # 断言结果字典长度为1
            self.assertEqual(len(ob_dict), 1)
            for v in ob_dict.values():
                # 断言浮点输出和量化输出列表长度相等
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    # 断言每对浮点和量化输出的形状相同
                    self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)

        # 创建包含静态量化卷积模型的列表
        model_list = [AnnotatedConvModel(qengine),
                      AnnotatedConvTransposeModel("qnnpack"),  # ConvT cannot use per channel weights
                      AnnotatedConvBnReLUModel(qengine)]
        # 模块替换列表，包含卷积层、融合的卷积ReLU层和转置卷积层
        module_swap_list = [nn.Conv2d, nn.intrinsic.modules.fused.ConvReLU2d, nn.ConvTranspose2d]
        for model in model_list:
            model.eval()
            # 如果模型有融合模型方法，执行融合操作
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            # 对模型进行静态量化
            q_model = quantize(model, test_only_eval_fn, [self.img_data_2d])
            # 比较和验证模型输出
            compare_and_validate_results(
                model, q_model, module_swap_list, self.img_data_2d[0][0]
            )

    @override_qengines
    def test_compare_model_stub_linear_static(self):
        r"""Compare the output of static quantized linear layer and its float shadow module"""

        # 获取当前的量化引擎
        qengine = torch.backends.quantized.engine

        # 定义比较并验证结果的函数，用于比较浮点模型和量化模型的输出
        def compare_and_validate_results(float_model, q_model, module_swap_list, data):
            # 调用 compare_model_stub 函数比较浮点模型和量化模型的输出
            ob_dict = compare_model_stub(float_model, q_model, module_swap_list, data)
            # 断言字典 ob_dict 的长度为1
            self.assertEqual(len(ob_dict), 1)
            for v in ob_dict.values():
                # 断言每个值中 "float" 和 "quantized" 的长度相等
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    # 断言每个量化值的形状与对应的浮点值相同
                    self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)

        # 从校准数据中获取线性数据
        linear_data = self.calib_data[0][0]
        # 指定要替换的模块列表，这里只包含 nn.Linear
        module_swap_list = [nn.Linear]
        # 创建单层线性模型的列表，并进行初始化
        model_list = [AnnotatedSingleLayerLinearModel(qengine)]
        for model in model_list:
            # 设置模型为评估模式
            model.eval()
            # 如果模型具有 fuse_model 方法，则调用该方法
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            # 对模型进行量化，使用 test_only_eval_fn 函数进行评估
            q_model = quantize(model, test_only_eval_fn, [self.calib_data])
            # 调用比较函数，比较浮点模型和量化模型的输出结果
            compare_and_validate_results(model, q_model, module_swap_list, linear_data)

    @override_qengines
    def test_compare_model_stub_partial(self):
        r"""Compare the output of static quantized linear layer and its float shadow module"""

        # 获取当前的量化引擎
        qengine = torch.backends.quantized.engine
        # TODO: Rebase on top of PR to remove compare and validate results here

        # 定义比较并验证结果的函数，用于比较浮点模型和量化模型的输出
        def compare_and_validate_results(float_model, q_model, module_swap_list, data):
            # 调用 compare_model_stub 函数比较浮点模型和量化模型的输出
            ob_dict = compare_model_stub(float_model, q_model, module_swap_list, data)
            # 断言字典 ob_dict 的长度为1
            self.assertEqual(len(ob_dict), 1)
            for v in ob_dict.values():
                # 断言每个值中 "float" 和 "quantized" 的长度相等
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    # 断言每个量化值的形状与对应的浮点值相同
                    self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)

        # 从校准数据中获取线性数据
        linear_data = self.calib_data[0][0]
        # 指定要替换的模块列表，这里只包含 nn.Linear
        module_swap_list = [nn.Linear]
        # 创建双层线性模型的列表，并进行初始化
        model_list = [AnnotatedTwoLayerLinearModel()]
        for model in model_list:
            # 设置模型为评估模式
            model.eval()
            # 如果模型具有 fuse_model 方法，则调用该方法
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            # 对模型进行量化，使用 test_only_eval_fn 函数进行评估
            q_model = quantize(model, test_only_eval_fn, [self.calib_data])
            # 调用比较函数，比较浮点模型和量化模型的输出结果
            compare_and_validate_results(model, q_model, module_swap_list, linear_data)

    @override_qengines
    def test_compare_model_stub_submodule_static(self):
        r"""Compare the output of static quantized submodule and its float shadow module"""

        # 获取当前的量化引擎
        qengine = torch.backends.quantized.engine

        # 创建一个包含子模块的模型，并设置为评估模式
        model = ModelWithSubModules().eval()
        # 对模型进行量化，使用指定的量化函数和数据进行测试
        q_model = quantize(model, test_only_eval_fn, [self.img_data_2d])
        # 指定需要交换的模块列表，用于模型比较
        module_swap_list = [SubModule, nn.Conv2d]
        # 比较原始模型和量化后模型的输出，返回结果字典
        ob_dict = compare_model_stub(
            model, q_model, module_swap_list, self.img_data_2d[0][0]
        )
        # 因为 conv 模块没有被量化，所以不会插入影子模块
        self.assertTrue(isinstance(q_model.mod1, Shadow))
        self.assertFalse(isinstance(q_model.conv, Shadow))


    @override_qengines
    def test_compare_model_stub_functional_static(self):
        r"""Compare the output of static quantized functional layer and its float shadow module"""

        # 获取当前的量化引擎
        qengine = torch.backends.quantized.engine

        # 创建一个包含功能层的模型，并设置为评估模式
        model = ModelWithFunctionals().eval()
        # 设置模型的量化配置为默认配置 "fbgemm"
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        # 准备模型以进行量化，不改变原始模型
        q_model = prepare(model, inplace=False)
        # 使用模型对数据进行前向传播，以便获得量化参数
        q_model(self.img_data_2d[0][0])
        # 将准备好的量化模型进行转换，使其成为量化后的模型
        q_model = convert(q_model)
        # 指定需要交换的模块列表，用于模型比较
        module_swap_list = [nnq.FloatFunctional]
        # 比较原始模型和量化后模型的输出，返回结果字典
        ob_dict = compare_model_stub(
            model, q_model, module_swap_list, self.img_data_2d[0][0]
        )
        # 断言结果字典的长度为6
        self.assertEqual(len(ob_dict), 6)
        # 断言各功能层是否插入了影子模块
        self.assertTrue(isinstance(q_model.mycat, Shadow))
        self.assertTrue(isinstance(q_model.myadd, Shadow))
        self.assertTrue(isinstance(q_model.mymul, Shadow))
        self.assertTrue(isinstance(q_model.myadd_relu, Shadow))
        self.assertTrue(isinstance(q_model.my_scalar_add, Shadow))
        self.assertTrue(isinstance(q_model.my_scalar_mul, Shadow))
        # 遍历结果字典的值，断言每对比的量化输出与原始浮点输出的形状一致
        for v in ob_dict.values():
            self.assertTrue(len(v["float"]) == len(v["quantized"]))
            for i, val in enumerate(v["quantized"]):
                self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)

    @override_qengines
    def test_compare_model_stub_linear_dynamic(self):
        r"""Compare the output of dynamic quantized linear layer and its float shadow module"""

        # 获取当前的量化引擎
        qengine = torch.backends.quantized.engine

        # 定义一个比较和验证结果的函数
        def compare_and_validate_results(float_model, q_model, module_swap_list, data):
            # 使用比较模型的 stub 函数比较浮点模型和量化模型
            ob_dict = compare_model_stub(float_model, q_model, module_swap_list, data)
            # 断言返回的结果字典长度为1
            self.assertEqual(len(ob_dict), 1)
            for v in ob_dict.values():
                # 断言浮点数结果和量化结果的长度相同
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    # 断言浮点数结果和量化结果的形状相同
                    self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)

        # 从校准数据中获取线性层的数据
        linear_data = self.calib_data[0][0]

        # 创建线性动态模型列表
        model_list = [SingleLayerLinearDynamicModel(qengine)]
        # 模块替换列表，用于量化过程
        module_swap_list = [nn.Linear, nn.LSTM]
        for model in model_list:
            # 将模型设置为评估模式
            model.eval()
            # 如果模型有融合模型的方法，则调用融合模型方法
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            # 对模型进行动态量化
            q_model = quantize_dynamic(model)
            # 比较并验证动态量化后的结果
            compare_and_validate_results(model, q_model, module_swap_list, linear_data)

    @override_qengines
    def test_compare_model_stub_lstm_dynamic(self):
        r"""Compare the output of dynamic quantized LSTM layer and its float shadow module"""

        # 获取当前的量化引擎
        qengine = torch.backends.quantized.engine

        # 定义一个比较和验证结果的函数
        def compare_and_validate_results(
            float_model, q_model, module_swap_list, input, hidden
        ):
            # 使用比较模型的 stub 函数比较浮点模型和量化模型
            ob_dict = compare_model_stub(
                float_model, q_model, module_swap_list, input, hidden
            )
            # 断言返回的结果字典长度为1
            self.assertEqual(len(ob_dict), 1)
            for v in ob_dict.values():
                # 断言浮点数结果和量化结果的长度相同
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    # 断言浮点数结果和量化结果的形状相同
                    self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)

        # 创建随机输入和隐藏状态作为 LSTM 的输入
        lstm_input = torch.rand((1, 1, 2))
        lstm_hidden = (torch.rand(1, 1, 2), torch.rand(1, 1, 2))

        # 创建包含动态隐藏状态 LSTM 模型的列表
        model_list = [LSTMwithHiddenDynamicModel(qengine)]
        # 模块替换列表，用于量化过程
        module_swap_list = [nn.Linear, nn.LSTM]
        for model in model_list:
            # 将模型设置为评估模式
            model.eval()
            # 如果模型有融合模型的方法，则调用融合模型方法
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            # 对模型进行动态量化
            q_model = quantize_dynamic(model)
            # 比较并验证动态量化后的结果
            compare_and_validate_results(
                model, q_model, module_swap_list, lstm_input, lstm_hidden
            )

    @override_qengines
    def test_compare_model_outputs_conv_static(self):
        r"""Compare the output of conv layer in stataic quantized model and corresponding
        output of conv layer in float model
        """
        # 获取当前的量化引擎
        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model, data):
            # 比较浮点模型和量化模型的输出，并验证结果
            act_compare_dict = compare_model_outputs(float_model, q_model, data)
            # 预期的比较结果应包含以下键值
            expected_act_compare_dict_keys = {"conv.stats", "quant.stats"}

            # 确保实际比较结果的键与预期一致
            self.assertTrue(act_compare_dict.keys() == expected_act_compare_dict_keys)
            for v in act_compare_dict.values():
                # 确保浮点数和量化数的张量形状一致
                self.assertTrue(v["float"][0].shape == v["quantized"][0].shape)

        # 量化模型列表
        model_list = [AnnotatedConvModel(qengine), AnnotatedConvBnReLUModel(qengine)]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            # 对模型进行量化，仅用于测试评估函数，传入数据为二维图像数据
            q_model = quantize(model, test_only_eval_fn, [self.img_data_2d])
            # 比较并验证结果
            compare_and_validate_results(model, q_model, self.img_data_2d[0][0])

    @override_qengines
    def test_compare_model_outputs_linear_static(self):
        r"""Compare the output of linear layer in static quantized model and corresponding
        output of conv layer in float model
        """
        # 获取当前的量化引擎
        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model, data):
            # 比较浮点模型和量化模型的输出，并验证结果
            act_compare_dict = compare_model_outputs(float_model, q_model, data)
            # 预期的比较结果应包含以下键值
            expected_act_compare_dict_keys = {"fc1.quant.stats", "fc1.module.stats"}

            # 确保实际比较结果的键与预期一致
            self.assertTrue(act_compare_dict.keys() == expected_act_compare_dict_keys)
            for v in act_compare_dict.values():
                # 确保浮点数和量化数的张量形状一致
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)

        # 线性层的校准数据
        linear_data = self.calib_data[0][0]
        # 量化模型列表
        model_list = [AnnotatedSingleLayerLinearModel(qengine)]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            # 对模型进行量化，仅用于测试评估函数，传入校准数据
            q_model = quantize(model, test_only_eval_fn, [self.calib_data])
            # 比较并验证结果
            compare_and_validate_results(model, q_model, linear_data)

    @override_qengines
    def test_compare_model_outputs_functional_static(self):
        r"""Compare the output of functional layer in static quantized model and corresponding
        output of conv layer in float model
        """
        # 获取当前量化引擎
        qengine = torch.backends.quantized.engine

        # 创建一个使用函数式层的模型，并设为评估模式
        model = ModelWithFunctionals().eval()
        # 设置模型的量化配置为默认的 "fbgemm" 配置
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        # 准备模型以进行量化，不在原地修改
        q_model = prepare(model, inplace=False)
        # 将输入数据传递给量化模型
        q_model(self.img_data_2d[0][0])
        # 将准备好的量化模型转换为量化表示
        q_model = convert(q_model)
        # 比较原始模型和量化模型的输出结果
        act_compare_dict = compare_model_outputs(model, q_model, self.img_data_2d[0][0])
        # 断言比较结果字典的长度为5
        self.assertEqual(len(act_compare_dict), 5)
        # 预期的比较结果字典的键集合
        expected_act_compare_dict_keys = {
            "mycat.stats",
            "myadd.stats",
            "mymul.stats",
            "myadd_relu.stats",
            "quant.stats",
        }
        # 断言实际比较结果字典的键与预期一致
        self.assertTrue(act_compare_dict.keys() == expected_act_compare_dict_keys)
        # 遍历比较结果字典的值
        for v in act_compare_dict.values():
            # 断言每个值的 "float" 和 "quantized" 部分长度相等
            self.assertTrue(len(v["float"]) == len(v["quantized"]))
            # 遍历每个量化值，断言其形状与对应的浮点值形状相同
            for i, val in enumerate(v["quantized"]):
                self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)

    @override_qengines
    def test_compare_model_outputs_linear_dynamic(self):
        r"""Compare the output of linear layer in dynamic quantized model and corresponding
        output of conv layer in float model
        """
        # 获取当前量化引擎
        qengine = torch.backends.quantized.engine

        # 定义一个比较和验证结果的函数，接受浮点模型、量化模型和数据作为输入
        def compare_and_validate_results(float_model, q_model, data):
            # 比较模型的输出结果
            act_compare_dict = compare_model_outputs(float_model, q_model, data)
            # 预期的比较结果字典的键集合
            expected_act_compare_dict_keys = {"fc1.stats"}

            # 断言实际比较结果字典的键与预期一致
            self.assertTrue(act_compare_dict.keys() == expected_act_compare_dict_keys)
            # 遍历比较结果字典的值
            for v in act_compare_dict.values():
                # 断言每个值的 "float" 和 "quantized" 部分长度相等
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                # 遍历每个量化值，断言其形状与对应的浮点值形状相同
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)

        # 从校准数据中获取线性数据
        linear_data = self.calib_data[0][0]

        # 创建动态量化模型列表，包含单层线性动态模型
        model_list = [SingleLayerLinearDynamicModel(qengine)]
        # 遍历模型列表
        for model in model_list:
            # 设为评估模式
            model.eval()
            # 如果模型具有融合模型的能力，则进行融合
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            # 对模型进行动态量化
            q_model = quantize_dynamic(model)
            # 使用定义的函数比较并验证结果
            compare_and_validate_results(model, q_model, linear_data)

    @override_qengines
    def test_compare_model_outputs_lstm_dynamic(self):
        r"""Compare the output of LSTM layer in dynamic quantized model and corresponding
        output of conv layer in float model
        """
        # 获取当前的量化引擎
        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model, input, hidden):
            # 比较和验证 LSTM 模型的输出
            act_compare_dict = compare_model_outputs(
                float_model, q_model, input, hidden
            )
            # 预期的比较结果字典的键
            expected_act_compare_dict_keys = {"lstm.stats"}

            # 断言实际比较结果的键与预期一致
            self.assertTrue(act_compare_dict.keys() == expected_act_compare_dict_keys)
            for v in act_compare_dict.values():
                # 断言每个输出的长度相同
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    # 断言每个输出的形状相同
                    self.assertTrue(len(v["float"][i]) == len(v["quantized"][i]))
                    if i == 0:
                        self.assertTrue(v["float"][i][0].shape == v["quantized"][i][0].shape)
                    else:
                        self.assertTrue(
                            v["float"][i][0].shape == v["quantized"][i][0].shape
                        )
                        self.assertTrue(
                            v["float"][i][1].shape == v["quantized"][i][1].shape
                        )

        # 随机生成 LSTM 的输入和隐藏层状态
        lstm_input = torch.rand((1, 1, 2))
        lstm_hidden = (torch.rand(1, 1, 2), torch.rand(1, 1, 2))

        # 创建包含动态量化 LSTM 模型的列表
        model_list = [LSTMwithHiddenDynamicModel(qengine)]
        for model in model_list:
            model.eval()
            # 如果模型具有融合功能，则执行模型融合
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            # 对模型进行动态量化
            q_model = quantize_dynamic(model)
            # 比较和验证结果
            compare_and_validate_results(model, q_model, lstm_input, lstm_hidden)

    @override_qengines
    def test_output_logger(self):
        r"""Compare output from OutputLogger with the expected results"""
        # 创建两个随机张量
        x = torch.rand(2, 2)
        y = torch.rand(2, 1)

        # 将张量 x 和 y 添加到列表 l 中
        l = []
        l.append(x)
        l.append(y)

        # 创建输出日志记录器
        logger = OutputLogger()
        # 记录张量 x 和 y 的前向传播输出
        logger.forward(x)
        logger.forward(y)

        # 断言列表 l 等于记录的张量值
        self.assertEqual(l, logger.stats["tensor_val"])

    @override_qengines
    def test_shadow_logger(self):
        r"""Compare output from ShawdowLogger with the expected results"""
        # 创建两组随机浮点数和量化后的张量
        a_float = torch.rand(2, 2)
        a_quantized = torch.rand(2, 2)

        b_float = torch.rand(3, 2, 2)
        b_quantized = torch.rand(3, 2, 2)

        # 创建影子日志记录器
        logger = ShadowLogger()
        # 记录浮点数和量化张量的前向传播输出
        logger.forward(a_float, a_quantized)
        logger.forward(b_float, b_quantized)

        # 断言记录的浮点数和量化张量数量为 2
        self.assertEqual(len(logger.stats["float"]), 2)
        self.assertEqual(len(logger.stats["quantized"]), 2)

    @skip_if_no_torchvision
    # 测试视觉模型的量化效果，输入一个浮点数模型（float_model）
    def _test_vision_model(self, float_model):
        # 将浮点数模型移动到CPU上
        float_model.to('cpu')
        # 设置模型为评估模式，即不启用训练相关的功能
        float_model.eval()
        # 尝试将模型中的一些操作融合以优化性能
        float_model.fuse_model()
        # 设置量化配置为默认的量化配置
        float_model.qconfig = torch.ao.quantization.default_qconfig
        # 创建一个包含随机图像数据和标签的列表
        img_data = [(torch.rand(2, 3, 224, 224, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long)) for _ in range(2)]
        # 对浮点数模型进行量化，生成一个量化模型（qmodel），并不在原地操作
        qmodel = quantize(float_model, torch.ao.quantization.default_eval_fn, [img_data], inplace=False)

        # 比较浮点数模型和量化模型的权重，返回一个包含权重差异的字典
        wt_compare_dict = compare_weights(float_model.state_dict(), qmodel.state_dict())

        # 定义一个函数，用于计算两个张量之间的误差
        def compute_error(x, y):
            Ps = torch.norm(x)
            Pn = torch.norm(x - y)
            return 20 * torch.log10(Ps / Pn)

        # 获取图像数据的第一个样本
        data = img_data[0][0]
        
        # 比较浮点数模型和量化模型在给定输入数据下的输出，返回一个包含输出差异的字典
        act_compare_dict = compare_model_outputs(float_model, qmodel, data)

        # 遍历输出差异字典中的每个模块
        for key in act_compare_dict:
            # 计算浮点数模型和量化模型输出之间的误差
            compute_error(act_compare_dict[key]['float'][0], act_compare_dict[key]['quantized'][0].dequantize())

        # 准备模型输出，可能会修改模型状态以准备进行后续的评估
        prepare_model_outputs(float_model, qmodel)

        # 遍历图像数据，对浮点数模型和量化模型分别进行推理
        for data in img_data:
            float_model(data[0])
            qmodel(data[0])

        # 查找浮点数模型和量化模型之间匹配的激活值，返回一个包含匹配激活值的字典
        act_compare_dict = get_matching_activations(float_model, qmodel)

    # 如果没有torchvision库，则跳过测试
    @skip_if_no_torchvision
    # 如果运行环境是ARM64架构，则跳过测试
    @unittest.skipIf(IS_ARM64, "Not working on arm right now")
    # 测试MobileNet V2模型的量化效果
    def test_mobilenet_v2(self):
        # 导入torchvision库中的MobileNet V2模型，并调用_test_vision_model进行测试
        from torchvision.models.quantization import mobilenet_v2
        self._test_vision_model(mobilenet_v2(pretrained=True, quantize=False))

    # 如果没有torchvision库，则跳过测试
    @skip_if_no_torchvision
    # 如果运行环境是ARM64架构，则跳过测试
    @unittest.skipIf(IS_ARM64, "Not working on arm right now")
    # 测试MobileNet V3模型的量化效果
    def test_mobilenet_v3(self):
        # 导入torchvision库中的MobileNet V3 Large模型，并调用_test_vision_model进行测试
        from torchvision.models.quantization import mobilenet_v3_large
        self._test_vision_model(mobilenet_v3_large(pretrained=True, quantize=False))
```