# `.\pytorch\test\quantization\jit\test_ondevice_quantization.py`

```
# Owner(s): ["oncall: quantization"]

import io  # 导入 io 模块，用于处理文件流
from typing import Dict  # 导入 Dict 类型提示，用于指定字典类型的变量声明

import torch  # 导入 PyTorch 深度学习框架
import torch._C  # 导入 PyTorch 的 C++ 扩展模块

from torch.ao.quantization import default_dynamic_qconfig, per_channel_dynamic_qconfig  # 从量化模块导入默认和按通道动态量化配置

from torch.ao.quantization.quantize_jit import (  # 从 JIT 量化模块导入相关函数
    _prepare_ondevice_dynamic_jit,
    _quantize_ondevice_dynamic_jit,
    convert_dynamic_jit,
    prepare_dynamic_jit,
)

from torch.jit.mobile import _load_for_lite_interpreter, LiteScriptModule  # 导入 JIT 移动相关模块和 LiteScriptModule

from torch.testing import FileCheck  # 导入用于测试的文件检查工具

from torch.testing._internal.common_quantization import (  # 导入内部通用量化测试模块
    get_script_module,
    LinearAddModel,
)

from torch.testing._internal.common_utils import TestCase  # 导入用于测试的通用测试用例模块
from torch.utils import bundled_inputs as bundled_inputs  # 导入捆绑输入数据的模块


class myMod(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 5).float()  # 定义一个全连接层，输入输出维度都为 5
        self.fc1.weight = weight  # 设置全连接层的权重为给定的权重参数
        self.fc2 = torch.nn.Linear(5, 5).float()  # 定义另一个全连接层，输入输出维度都为 5

    def forward(self, x):
        return self.fc2(self.fc1(x))  # 模型的前向传播，先经过 fc1 再经过 fc2


class MyConvLinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3)  # 定义一个二维卷积层，输入通道 3，输出通道 5，卷积核大小 3x3
        weight = torch.nn.Parameter(torch.ones(5, 5))  # 创建一个可学习的权重参数矩阵 5x5
        self.weight1 = torch.nn.Parameter(torch.ones(5, 5))  # 创建另一个可学习的权重参数矩阵 5x5
        self.mymod = myMod(weight)  # 创建 myMod 类的实例，传入前面定义的权重参数

    def forward(self, x):
        conv_output = self.conv(x)  # 对输入 x 进行卷积操作，得到卷积输出
        y = self.mymod(conv_output)  # 将卷积输出传入 myMod 模块进行处理，得到 y
        z = torch.nn.functional.linear(y, self.weight1)  # 对 y 应用线性变换，权重为 self.weight1，得到 z
        return z  # 返回线性变换后的结果 z

    def get_example_inputs(self):
        return (torch.rand(1, 3, 12, 7),)  # 返回一个示例输入，随机生成的 1x3x12x7 的张量


class OnDevicePTQUtils:
    observer_module_name = ["MinMaxObserver", "PerChannelMinMaxObserver"]  # 观察器模块的名称列表

    @staticmethod
    def insert_observers(model, qconfig_dict):
        inputs = model.get_example_inputs()  # 获取模型的示例输入
        scripted_model = get_script_module(model, False, inputs)  # 获取模型的脚本化表示
        scripted_model = _prepare_ondevice_dynamic_jit(scripted_model, qconfig_dict)  # 准备在设备上进行动态 JIT 量化
        return scripted_model  # 返回量化后的脚本化模型

    @staticmethod
    def ptq_dynamic_quantize(model, qconfig_dict):
        inputs = model.get_example_inputs()  # 获取模型的示例输入
        m = get_script_module(model, False, inputs)  # 获取模型的脚本化表示
        m = _quantize_ondevice_dynamic_jit(m, qconfig_dict, "forward", True)  # 在设备上进行动态 JIT 量化
        return m  # 返回量化后的模型

    @staticmethod
    def find_observer_modules(m):
        observer_modules = []  # 初始化观察器模块列表
        for child_module in m.children():  # 遍历模型的子模块
            if child_module.original_name in OnDevicePTQUtils.observer_module_name:  # 如果子模块名称在观察器模块名称列表中
                observer_modules.append(child_module)  # 将该子模块添加到观察器模块列表中
        return observer_modules  # 返回找到的观察器模块列表

    @staticmethod
    def is_value_type_observer(value):
        type_name = value.type()  # 获取值的类型名称
        for observer_type in OnDevicePTQUtils.observer_module_name:  # 遍历观察器模块名称列表
            if observer_type in type_name.str():  # 如果观察器类型名称出现在值的类型名称中
                return True  # 返回 True
        return False  # 否则返回 False

    @staticmethod
    def is_calculate_qparam(node):
        if node.kind() == "prim::CallMethod":  # 如果节点类型是方法调用
            if node.s("name") == "calculate_qparams":  # 如果方法名是 calculate_qparams
                return True  # 返回 True
        return False  # 否则返回 False

    @staticmethod
    # 获取线性量化打包参数的浮点权重名称
    def get_linear_packed_param_fp_weight(node):
        # 获取输入节点的第一个节点作为权重节点
        weight = node.inputsAt(0).node()
        # 如果权重节点的类型既不是"aten::quantize_per_tensor"也不是"aten::quantize_per_channel"
        if (
            weight.kind() != "aten::quantize_per_tensor"
            and weight.kind() != "aten::quantize_per_channel"
        ):
            # 抛出数值错误，说明量化的权重必须已经生成
            raise ValueError("Quantized weight must be produced.")
        # 获取浮点权重节点
        fp_weight = weight.inputsAt(0).node()
        # 断言浮点权重节点的类型为"prim::GetAttr"，即权重必须是模块的属性
        assert (
            fp_weight.kind() == "prim::GetAttr"
        ), "Weight must be an attribute of the module."
        # 获取浮点权重的名称
        fp_weight_name = fp_weight.s("name")
        # 返回浮点权重的名称
        return fp_weight_name

    @staticmethod
    # 判断节点是否为通道量化的打包参数
    def is_per_channel_quantized_packed_param(node):
        # 断言节点的类型必须是"quantized::linear_prepack"，即节点必须对应线性预打包
        assert (
            node.kind() == "quantized::linear_prepack"
        ), "Node must corresponds to linear_prepack."
        # 获取节点的输入权重节点
        weight = node.inputsAt(0).node()
        # 断言权重节点的类型既不是"aten::quantize_per_tensor"也不是"aten::quantize_per_channel"
        assert (
            weight.kind() != "aten::quantize_per_tensor"
            or weight.kind() != "aten::quantize_per_channel"
        )
        # 返回权重节点的类型是否不是"aten::quantize_per_tensor"，即判断是否为通道量化
        return weight.kind() != "aten::quantize_per_tensor"
# 定义测试类 TestOnDeviceDynamicPTQInsertObservers，继承自 TestCase
class TestOnDeviceDynamicPTQInsertObservers(TestCase):

    # 检查观察者数量和类型的私有方法
    def _check_num_and_type_of_observers(self, model, num_observers):
        # 默认的动态量化配置字典
        qconfig_dict = {"": default_dynamic_qconfig}
        # 插入观察者并返回脚本化的模型
        scripted_model = OnDevicePTQUtils.insert_observers(model, qconfig_dict)
        # 查找脚本化模型中的观察者模块
        observer_modules = OnDevicePTQUtils.find_observer_modules(scripted_model)
        # 断言观察者模块的数量与期望值相等
        self.assertTrue(len(observer_modules) == num_observers)
        # 遍历观察者模块列表，断言每个观察者的原始名称为 "MinMaxObserver"
        for observer in observer_modules:
            self.assertTrue(observer.original_name == "MinMaxObserver")

        # 每通道的动态量化配置字典
        qconfig_dict = {"": per_channel_dynamic_qconfig}
        # 再次插入观察者并返回脚本化的模型
        scripted_model = OnDevicePTQUtils.insert_observers(model, qconfig_dict)
        # 再次查找脚本化模型中的观察者模块
        observer_modules = OnDevicePTQUtils.find_observer_modules(scripted_model)
        # 再次断言观察者模块的数量与期望值相等
        self.assertTrue(len(observer_modules) == num_observers)
        # 遍历观察者模块列表，断言每个观察者的原始名称为 "PerChannelMinMaxObserver"
        for observer in observer_modules:
            self.assertTrue(observer.original_name == "PerChannelMinMaxObserver")

    # 检查观察者方法的私有方法
    def _check_observer_method(self, model, num_observers):
        # 默认的动态量化配置字典
        qconfig_dict = {"": default_dynamic_qconfig}
        # 获取模型的示例输入
        inputs = model.get_example_inputs()
        # 获取脚本化模型，不进行内联处理
        orig_scripted_model = get_script_module(model, False, inputs)
        # 内联处理原始脚本化模型的图形
        torch._C._jit_pass_inline(orig_scripted_model.graph)
        # 获取原始前向图的字符串表示
        orig_forward_graph = orig_scripted_model.graph.str()
        # 插入观察者并返回脚本化的模型
        scripted_model = OnDevicePTQUtils.insert_observers(model, qconfig_dict)
        # 获取量化后的前向图的字符串表示
        quant_forward_graph = scripted_model.graph.str()
        # 断言原始前向图和量化后前向图的行数相等
        # 由于确切的图形匹配很困难，这里只比较行数
        self.assertEqual(
            len(orig_forward_graph.splitlines()), len(quant_forward_graph.splitlines())
        )
        # 获取 observe_forward 方法的图形表示
        observe_method = scripted_model.observe_forward.graph
        # 使用 FileCheck 检查 forward 方法调用的次数
        FileCheck().check_count(
            'prim::CallMethod[name="forward"](%_observer', num_observers, exactly=True
        ).run(observe_method)
        # 获取 reset_observers_forward 方法的图形表示
        reset_observers_method = scripted_model.reset_observers_forward.graph
        # 使用 FileCheck 检查 reset_min_max_vals 方法调用的次数
        FileCheck().check_count(
            'prim::CallMethod[name="reset_min_max_vals"](%_observer',
            num_observers,
            exactly=True,
        ).run(reset_observers_method)

    # 检查节点是否是仅权重的观察者的私有方法
    def _observer_is_weight_only(self, node):
        # 如果节点类型为 "prim::CallMethod" 并且名称为 "forward"
        if (node.kind() == "prim::CallMethod") and node.s("name") == "forward":
            # 如果是值类型观察者，则返回输入第二个节点是否为 "prim::GetAttr"
            if OnDevicePTQUtils.is_value_type_observer(node.inputsAt(0)):
                return node.inputsAt(1).node().kind() == "prim::GetAttr"
        # 默认返回 False
        return False

    # 测试观察者数量的方法
    def test_num_observers(self):
        # 创建线性加法模型
        model = LinearAddModel()
        # 检查观察者数量和类型是否符合预期
        self._check_num_and_type_of_observers(model, 2)
        # 创建自定义卷积线性模块
        model = MyConvLinearModule()
        # 检查观察者数量和类型是否符合预期
        self._check_num_and_type_of_observers(model, 3)

    # 测试观察者方法的方法
    def test_observe_method(self):
        # 创建自定义卷积线性模块
        model = MyConvLinearModule()
        # 检查观察者方法是否符合预期
        self._check_observer_method(model, 3)
    # 定义一个测试方法，测试仅包含权重观察器的情况
    def test_weight_only_observers(self):
        # 创建一个自定义的卷积线性模块实例
        model = MyConvLinearModule()
        # 定义量化配置字典，其中使用默认的动态量化配置
        qconfig_dict = {"": default_dynamic_qconfig}
        # 获取模型的示例输入
        inputs = model.get_example_inputs()
        # 将观察器插入到模型中，并返回脚本化的模型
        scripted_model = OnDevicePTQUtils.insert_observers(model, qconfig_dict)
        # 获取插入观察器后的前向观察图
        observe_forward_graph = scripted_model.observe_forward.graph
        # 初始化权重观察器计数器
        num_weight_only_observers = 0
        # 遍历前向观察图中的每个节点
        for node in observe_forward_graph.nodes():
            # 检查当前节点是否是权重观察器
            if self._observer_is_weight_only(node):
                # 如果是权重观察器，则增加计数器
                num_weight_only_observers += 1
        # 断言权重观察器的数量是否为3
        self.assertEqual(num_weight_only_observers, 3)
# 定义测试类 TestOnDeviceDynamicPTQInsertQuantDequant，继承自 TestCase
class TestOnDeviceDynamicPTQInsertQuantDequant(TestCase):

    # 验证模型中量化和反量化节点的数量
    def _validate_quant_dequant_nodes(self, model, num_nodes, per_channel=0):
        # 获取模型的量化前向图
        quantize_forward_graph = model.quantize_forward.graph
        quantize_per_tensor = quantize_per_channel = 0
        # 遍历图中的节点
        for n in quantize_forward_graph.nodes():
            # 检查节点类型是否包含 "aten::quantize_per_tensor"
            if "aten::quantize_per_tensor" in n.kind():
                quantize_per_tensor += 1
            # 检查节点类型是否包含 "aten::quantize_per_channel"
            if "aten::quantize_per_channel" in n.kind():
                quantize_per_channel += 1
        # 断言量化节点和反量化节点的总数等于指定的节点数
        self.assertEqual(quantize_per_tensor + quantize_per_channel, num_nodes)

    # 验证模型中计算量化参数的节点数量
    def _validate_calculate_qparams(self, model, num_nodes):
        # 获取模型的量化前向图
        quantize_forward_graph = model.quantize_forward.graph
        num_calculate_qparams = 0
        # 遍历图中的节点
        for n in quantize_forward_graph.nodes():
            # 如果节点是计算量化参数的节点
            if OnDevicePTQUtils.is_calculate_qparam(n):
                num_calculate_qparams += 1
        # 断言计算量化参数节点的数量等于指定的节点数
        self.assertEqual(num_calculate_qparams, num_nodes)

    # 验证模型中是否没有观察者前向方法
    def _validate_no_observer_forward(self, model):
        # 获取模型的量化前向图
        quantize_forward_graph = model.quantize_forward.graph
        # 遍历图中的节点
        for n in quantize_forward_graph.nodes():
            # 如果节点的类型是 "prim::CallMethod" 并且方法名是 "forward"
            if (n.kind() == "prim::CallMethod") and n.s("name") == "forward":
                # 检查输入节点是否是值类型的观察者
                if OnDevicePTQUtils.is_value_type_observer(n.inputsAt(0)):
                    return False
        return True

    # 检查模型的量化、反量化和计算量化参数的节点数是否正确
    def _check_quant_dequant_and_calc_qparams(self, model, num_nodes):
        # 默认配置字典
        qconfig_dict = {"": default_dynamic_qconfig}
        # 进行动态量化
        m = OnDevicePTQUtils.ptq_dynamic_quantize(model, qconfig_dict)
        # 验证量化和反量化节点的数量
        self._validate_quant_dequant_nodes(m, num_nodes)
        # 验证计算量化参数的节点数量
        self._validate_calculate_qparams(m, num_nodes)
        # 验证是否没有观察者前向方法
        self._validate_no_observer_forward(m)

        # 通道动态配置字典
        qconfig_dict = {"": per_channel_dynamic_qconfig}
        # 进行动态量化
        m = OnDevicePTQUtils.ptq_dynamic_quantize(model, qconfig_dict)
        # 验证量化和反量化节点的数量
        self._validate_quant_dequant_nodes(m, num_nodes, num_nodes)
        # 验证计算量化参数的节点数量
        self._validate_calculate_qparams(m, num_nodes)
        # 验证是否没有观察者前向方法
        self._validate_no_observer_forward(m)

    # 检查模型的量化前向运行是否正常
    def _check_quantize_forward_runs(self, model):
        # 获取示例输入
        inputs = model.get_example_inputs()
        # 默认配置字典
        qconfig_dict = {"": default_dynamic_qconfig}
        # 进行动态量化
        m = OnDevicePTQUtils.ptq_dynamic_quantize(model, qconfig_dict)
        # 观察前向运行以记录统计数据，以产生正确的缩放因子和零点
        m.observe_forward(*inputs)
        # 运行量化前向
        m.quantize_forward(*inputs)

        # 通道动态配置字典
        qconfig_dict = {"": per_channel_dynamic_qconfig}
        # 进行动态量化
        m = OnDevicePTQUtils.ptq_dynamic_quantize(model, qconfig_dict)
        # 首先必须运行观察前向以记录统计数据，以产生正确的缩放因子和零点
        m.observe_forward(*inputs)
        # 运行量化前向
        m.quantize_forward(*inputs)

    # 测试函数，验证量化和反量化节点的数量
    def test_num_quant_dequant_nodes(self):
        # 创建 LinearAddModel 模型
        model = LinearAddModel()
        # 验证模型中的量化和反量化节点数量为 2
        self._check_quant_dequant_and_calc_qparams(model, 2)
        # 创建 MyConvLinearModule 模型
        model = MyConvLinearModule()
        # 验证模型中的量化和反量化节点数量为 3
        self._check_quant_dequant_and_calc_qparams(model, 3)
    # 定义测试函数 test_quantize_forward_runs，用于测试量化前向运行
    def test_quantize_forward_runs(self):
        # 创建一个 LinearAddModel 实例，用于测试其量化前向运行
        model = LinearAddModel()
        # 调用内部方法 _check_quantize_forward_runs，验证模型的量化前向运行
        self._check_quantize_forward_runs(model)
        # 创建一个 MyConvLinearModule 实例，用于测试其量化前向运行
        model = MyConvLinearModule()
        # 再次调用内部方法 _check_quantize_forward_runs，验证模型的量化前向运行
        self._check_quantize_forward_runs(model)
class TestOnDeviceDynamicPTQFinalize(TestCase):
    # 定义一个测试类 TestOnDeviceDynamicPTQFinalize，继承自 TestCase

    def _validate_packed_params(self, model, num_nodes, per_channel=0):
        # 定义一个方法 _validate_packed_params，用于验证模型的打包参数
        # model: 输入的模型对象
        # num_nodes: 预期的节点数量
        # per_channel: 是否为通道量化，默认为0（否）
        
        quantize_forward_graph = model.quantize_forward.graph
        # 获取模型的量化前向图

        quantize_per_tensor = quantize_per_channel = 0
        # 初始化量化为每张量和每通道的计数器为0
        linear_prepack = 0
        # 初始化线性预打包的计数器为0
        linear_prepack_uses = 0
        # 初始化线性预打包的使用次数计数器为0
        
        for n in quantize_forward_graph.nodes():
            # 遍历量化前向图的所有节点
            if n.kind() == "prim::SetAttr":
                # 如果节点类型是 "prim::SetAttr"
                maybe_packed_param_value = n.inputsAt(1)
                # 获取第二个输入（也就是 SetAttr 的值）

                maybe_packed_param = maybe_packed_param_value.node()
                # 获取这个值的节点对象
                
                if maybe_packed_param.kind() == "quantized::linear_prepack":
                    # 如果节点类型是 "quantized::linear_prepack"
                    linear_prepack += 1
                    # 线性预打包计数器加1
                    linear_prepack_uses += len(maybe_packed_param_value.uses())
                    # 增加线性预打包节点使用次数计数器
                    if OnDevicePTQUtils.is_per_channel_quantized_packed_param(
                        maybe_packed_param
                    ):
                        # 检查是否为通道量化的打包参数
                        quantize_per_channel += 1
                    else:
                        quantize_per_tensor += 1
                    # 否则为每张量量化的打包参数计数器加1

        self.assertEqual(quantize_per_tensor + quantize_per_channel, num_nodes)
        # 断言每张量和每通道的打包参数总数等于预期的节点数量
        self.assertEqual(quantize_per_channel, per_channel)
        # 断言通道量化的打包参数数量等于预期的 per_channel
        self.assertEqual(linear_prepack, num_nodes)
        # 断言线性预打包节点数量等于预期的节点数量
        self.assertEqual(linear_prepack_uses, num_nodes)
        # 断言线性预打包节点使用次数等于预期的节点数量

    def _validate_no_linear_unpack(self, model):
        # 定义一个方法 _validate_no_linear_unpack，用于验证模型中是否没有线性解包节点
        quantize_forward_graph = model.quantize_forward.graph
        # 获取模型的量化前向图

        for n in quantize_forward_graph.nodes():
            # 遍历量化前向图的所有节点
            if n.kind() == "quantized::linear_unpack":
                # 如果节点类型是 "quantized::linear_unpack"
                return False
                # 返回 False，表示存在线性解包节点
        
        return True
        # 如果未发现线性解包节点，返回 True

    def _validate_setattr_fp_weights(self, model, num_nodes):
        # 定义一个方法 _validate_setattr_fp_weights，用于验证设置浮点权重的属性
        quantize_forward_graph = model.quantize_forward.graph
        # 获取模型的量化前向图

        fp_weights_setattr = 0
        # 初始化浮点权重属性设置计数器为0
        fp_weight_names = []
        # 初始化浮点权重名称列表为空列表
        
        for n in quantize_forward_graph.nodes():
            # 第一轮遍历：遍历量化前向图的所有节点
            if n.kind() == "prim::SetAttr":
                # 如果节点类型是 "prim::SetAttr"
                maybe_packed_param = n.inputsAt(1).node()
                # 获取第二个输入的节点对象

                if maybe_packed_param.kind() == "quantized::linear_prepack":
                    # 如果节点类型是 "quantized::linear_prepack"
                    weight_name = OnDevicePTQUtils.get_linear_packed_param_fp_weight(
                        maybe_packed_param
                    )
                    # 获取线性打包参数的浮点权重名称
                    fp_weight_names.append(weight_name)
                    # 将浮点权重名称添加到列表中

        for n in quantize_forward_graph.nodes():
            # 第二轮遍历：再次遍历量化前向图的所有节点
            # 此部分基本上是在检测
            # %x = prim::Constant
            # = prim::SetAttr(<weight_name>)(module_value, x)
            # 以确保原始浮点权重被重置
            if n.kind() == "prim::SetAttr":
                # 如果节点类型是 "prim::SetAttr"
                weight_name = n.s("name")
                # 获取属性的名称
                if weight_name in fp_weight_names:
                    # 如果属性名称在浮点权重名称列表中
                    maybe_constant = n.inputsAt(1).node()
                    # 获取第二个输入的节点对象

                    if maybe_constant.kind() == "prim::Constant":
                        # 如果节点类型是 "prim::Constant"
                        fp_weights_setattr += 1
                        # 浮点权重设置属性计数器加1
        
        self.assertEqual(fp_weights_setattr, num_nodes)
        # 断言浮点权重设置属性的数量等于预期的节点数量
    # 验证量化前向图的有效性
    def _validate_quantized_forward(self, model, num_nodes):
        # 获取模型的量化前向图
        quantized_forward_graph = model.quantized_forward.graph
        # 初始化量化类型统计变量
        quantize_per_tensor = quantize_per_channel = 0
        # 初始化动态量化线性层统计变量和线性层参数打包统计变量
        quantized_linear_dynamic = 0
        linear_packed_params = 0
        # 初始化设置属性节点数量计数变量
        num_setattr = 0
        # 遍历量化前向图的所有节点
        for n in quantized_forward_graph.nodes():
            # 检查节点类型是否包含量化为每个张量的操作
            if "aten::quantize_per_tensor" in n.kind():
                quantize_per_tensor += 1
            # 检查节点类型是否包含量化为每个通道的操作
            if "aten::quantize_per_channel" in n.kind():
                quantize_per_channel += 1
            # 检查节点类型是否包含动态量化线性层操作
            if "quantized::linear_dynamic" in n.kind():
                quantized_linear_dynamic += 1
            # 检查节点是否为获取属性操作
            if n.kind() == "prim::GetAttr":
                # 获取输出节点
                output = n.outputsAt(0)
                # 获取输出节点类型
                output_type = output.type()
                # 检查输出节点类型是否为线性层参数打包基类
                if "LinearPackedParamsBase" in output_type.str():
                    linear_packed_params += 1
            # 检查节点是否为设置属性操作
            if n.kind() == "prim::SetAttr":
                num_setattr += 1
        # 断言：每种量化类型的节点数量应为零
        self.assertEqual(quantize_per_tensor, 0)
        self.assertEqual(quantize_per_channel, 0)
        # 断言：动态量化线性层节点数量应与给定节点数量相等
        self.assertEqual(quantized_linear_dynamic, num_nodes)
        # 断言：线性层参数打包节点数量应与给定节点数量相等
        self.assertEqual(linear_packed_params, num_nodes)
        # self.assertEqual(num_setattr, 0)

    # 检查动态量化前向操作
    def _check_quantize_forward(self, model, num_nodes):
        # 设置默认的动态量化配置字典
        qconfig_dict = {"": default_dynamic_qconfig}
        # 对模型进行动态量化
        m = OnDevicePTQUtils.ptq_dynamic_quantize(model, qconfig_dict)
        # 验证打包参数的有效性
        self._validate_packed_params(m, num_nodes)
        # 验证不进行线性层解包操作
        self._validate_no_linear_unpack(m)
        # 验证设置属性为浮点权重
        self._validate_setattr_fp_weights(m, num_nodes)

        # 设置每通道动态量化配置字典
        qconfig_dict = {"": per_channel_dynamic_qconfig}
        # 对模型进行动态量化
        m = OnDevicePTQUtils.ptq_dynamic_quantize(model, qconfig_dict)
        # 验证打包参数的有效性
        self._validate_packed_params(m, num_nodes, num_nodes)
        # 验证不进行线性层解包操作
        self._validate_no_linear_unpack(m)
        # 验证设置属性为浮点权重
        self._validate_setattr_fp_weights(m, num_nodes)

    # 检查量化后向操作
    def _check_quantized_forward(self, model, num_nodes):
        # 设置默认的动态量化配置字典
        qconfig_dict = {"": default_dynamic_qconfig}
        # 对模型进行动态量化
        m = OnDevicePTQUtils.ptq_dynamic_quantize(model, qconfig_dict)
        # 验证量化后向操作的有效性
        self._validate_quantized_forward(m, num_nodes)

        # 设置每通道动态量化配置字典
        qconfig_dict = {"": per_channel_dynamic_qconfig}
        # 对模型进行动态量化
        m = OnDevicePTQUtils.ptq_dynamic_quantize(model, qconfig_dict)
        # 验证量化后向操作的有效性
        self._validate_quantized_forward(m, num_nodes)
    # 定义一个内部方法，用于检查模型对动态量化后的输出是否与参考输出相近
    def _check_against_ref_dynamic_ptq(self, model):
        # 将模型设置为评估模式
        model.eval()
        # 获取模型的示例输入
        inputs = model.get_example_inputs()
        # 使用 Torch JIT 将模型转换为脚本模型
        ref_m = torch.jit.script(model)
        # 对脚本模型的计算图进行内联优化
        torch._C._jit_pass_inline(ref_m.graph)
        # 定义量化配置字典，使用默认的动态量化配置
        qconfig_dict = {"": default_dynamic_qconfig}
        # 准备动态 JIT 模型，应用动态量化配置
        ref_m = prepare_dynamic_jit(ref_m, qconfig_dict)
        # 将 JIT 模型转换为动态量化模型
        ref_m = convert_dynamic_jit(ref_m)
        # 调用动态量化模型的前向推理，得到参考输出
        ref_output = ref_m(*inputs)

        # 使用 OnDevicePTQUtils 中的方法对模型进行动态量化
        m = OnDevicePTQUtils.ptq_dynamic_quantize(model, qconfig_dict)
        # 观察模型前向传播输出，用于量化
        m.observe_forward(*inputs)
        # 对模型进行前向量化
        m.quantize_forward(*inputs)
        # 调用量化后的模型进行前向推理，得到输出
        output = m.quantized_forward(*inputs)
        # 断言量化后的输出与参考输出相近
        self.assertTrue(torch.allclose(ref_output, output))
        
        # 初始化异常抛出标志
        thrown = False
        # 尝试调用量化后的模型进行前向推理，捕获异常
        try:
            m(*inputs)
        except Exception as e:
            thrown = True
        # 断言已经抛出异常
        self.assertTrue(thrown)

        # 测试使用每通道量化配置的情况
        ref_m = torch.jit.script(model)
        torch._C._jit_pass_inline(ref_m.graph)
        qconfig_dict = {"": per_channel_dynamic_qconfig}
        ref_m = prepare_dynamic_jit(ref_m, qconfig_dict)
        ref_m = convert_dynamic_jit(ref_m)
        ref_output = ref_m(*inputs)

        m = OnDevicePTQUtils.ptq_dynamic_quantize(model, qconfig_dict)
        m.observe_forward(*inputs)
        m.quantize_forward(*inputs)
        output = m.quantized_forward(*inputs)
        self.assertTrue(torch.allclose(ref_output, output))
        
        thrown = False
        try:
            m(*inputs)
        except Exception as e:
            thrown = True
        self.assertTrue(thrown)

    # 定义一个内部辅助方法，用于测试模型的序列化和反序列化功能
    def _check_serdes_and_device_side_api_helper(
        self, model, check_device_side_api=False
    ):
        # 调用 _check_serdes_and_device_side_api_helper 方法，不检查设备端 API
        self._check_serdes_and_device_side_api_helper(model, False)

    # 定义一个内部方法，用于测试模型的设备端 API
    def _check_device_side_api(self, model):
        # 调用 _check_serdes_and_device_side_api_helper 方法，检查设备端 API
        self._check_serdes_and_device_side_api_helper(model, True)

    # 定义测试方法，用于测试量化前向推理
    def test_quantize_forward(self):
        # 创建 LinearAddModel 模型并测试其量化前向推理
        model = LinearAddModel()
        self._check_quantize_forward(model, 2)
        # 创建 MyConvLinearModule 模型并测试其量化前向推理
        model = MyConvLinearModule()
        self._check_quantize_forward(model, 3)

    # 定义测试方法，用于测试量化后的前向推理
    def test_quantized_forward(self):
        # 创建 LinearAddModel 模型并测试其量化后的前向推理
        model = LinearAddModel()
        self._check_quantized_forward(model, 2)
        # 创建 MyConvLinearModule 模型并测试其量化后的前向推理
        model = MyConvLinearModule()
        self._check_quantized_forward(model, 3)

    # 定义测试方法，用于测试动态量化与离线动态量化对比
    def test_against_offdevice_dynamic_ptq(self):
        # 创建 LinearAddModel 模型并测试其与参考动态量化模型的对比
        model = LinearAddModel()
        self._check_against_ref_dynamic_ptq(model)
        # 创建 MyConvLinearModule 模型并测试其与参考动态量化模型的对比
        model = MyConvLinearModule()
        self._check_against_ref_dynamic_ptq(model)

    # 定义测试方法，用于测试序列化和反序列化功能
    def test_serialization_deserialization(self):
        # 创建 MyConvLinearModule 模型并测试其序列化和反序列化功能
        model = MyConvLinearModule()
        self._check_serialization_deserialization(model)

    # 定义测试方法，用于测试设备端 API
    def test_device_side_api(self):
        # 创建 MyConvLinearModule 模型并测试其设备端 API
        model = MyConvLinearModule()
        self._check_device_side_api(model)
```