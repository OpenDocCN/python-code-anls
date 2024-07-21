# `.\pytorch\test\onnx\test_autograd_funs.py`

```
# Owner(s): ["module: onnx"]

# 导入测试相关的公共函数和类
import pytorch_test_common
from onnx_test_common import run_model_test

# 导入PyTorch相关模块
import torch
from torch.onnx import OperatorExportTypes
from torch.onnx._globals import GLOBALS
from torch.onnx.utils import _model_to_graph
from torch.testing._internal import common_utils

# 定义一个继承自ExportTestCase的测试类
class TestAutogradFuns(pytorch_test_common.ExportTestCase):
    # 设置ONNX导出操作的版本号
    opset_version = GLOBALS.export_onnx_opset_version
    # 设置是否将初始化器保留为输入
    keep_initializers_as_inputs = False
    # 设置是否进行ONNX图形状推断
    onnx_shape_inference = True

    # 定义测试单一输出情况下的函数
    def test_single_output(self):
        # 定义一个继承自torch.autograd.Function的单一输出类
        class SingleOut(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                # 计算输入张量的指数
                result = i.exp()
                # 计算结果的自然对数
                result = result.log()
                # 保存结果用于反向传播
                ctx.save_for_backward(result)
                return result

            @staticmethod
            def backward(ctx, grad_output):
                # 获取保存的结果张量
                (result,) = ctx.saved_tensors
                # 返回梯度乘以结果张量
                return grad_output * result

        # 定义一个调用SingleOut的神经网络模块
        class Caller(torch.nn.Module):
            def forward(self, input):
                # 对输入加5，并调用SingleOut进行处理后再加3
                result = input + 5
                return SingleOut.apply(result) + 3

        # 创建Caller类的实例
        model = Caller()
        # 创建输入张量
        input = torch.ones(1)
        # 运行模型测试
        run_model_test(self, model, input_args=(input,))

    # 定义测试多输出情况下的函数
    def test_multi_output(self):
        # 定义一个继承自torch.autograd.Function的多输出类
        class MultiOut(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                # 计算输入张量的指数和自然对数
                result_exp = i.exp()
                result_log = result_exp.log()
                # 保存结果用于反向传播
                ctx.save_for_backward(result_exp, result_log)
                return result_exp, result_log

            @staticmethod
            def backward(ctx, grad_output):
                # 获取保存的结果张量
                (result,) = ctx.saved_tensors
                # 返回梯度乘以结果张量
                return grad_output * result

        # 定义一个调用MultiOut的神经网络模块
        class Caller(torch.nn.Module):
            def forward(self, input):
                # 调用MultiOut处理输入张量
                return MultiOut.apply(input)

        # 创建Caller类的实例
        model = Caller()
        # 创建输入张量
        input = torch.ones(1, 5)
        # 运行模型测试
        run_model_test(self, model, input_args=(input,))

    # 定义测试部分输出情况下的函数
    def test_partial_output(self):
        # 定义一个继承自torch.autograd.Function的部分输出类
        class PartialOut(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                # 保存输入张量用于反向传播
                ctx.save_for_backward(input)
                # 获取输入张量中的前三个最大值
                values, indices = torch.topk(input, 3)
                return values

        # 定义一个调用PartialOut的神经网络模块
        class Caller(torch.nn.Module):
            def forward(self, input):
                # 调用PartialOut处理输入张量
                return PartialOut.apply(input)

        # 创建Caller类的实例
        model = Caller()
        # 创建输入张量
        input = torch.ones(1, 5)
        # 运行模型测试
        run_model_test(self, model, input_args=(input,))
    # 定义一个测试方法，测试嵌套的自动求导功能
    def test_nested_autograd(self):
        # 定义一个继承自torch.autograd.Function的子类Child
        class Child(torch.autograd.Function):
            # 前向传播函数的静态方法
            @staticmethod
            def forward(ctx, i):
                # 计算输入张量i的对数
                result = i.log()
                # 对结果再次计算对数
                result_log = result.log()
                # 保存结果以便反向传播使用
                ctx.save_for_backward(result_log)
                return result_log

            # 反向传播函数的静态方法
            @staticmethod
            def backward(ctx, grad_output):
                # 从上下文中加载保存的张量
                (result,) = ctx.saved_tensors
                # 返回梯度
                return grad_output * result

        # 定义一个继承自torch.autograd.Function的父类Parent
        class Parent(torch.autograd.Function):
            # 前向传播函数的静态方法
            @staticmethod
            def forward(ctx, i):
                # 计算输入张量i的指数
                result_exp = i.exp()
                # 调用Child类的apply方法计算对数并保存结果
                result_log = Child.apply(result_exp)
                # 保存结果以便反向传播使用
                ctx.save_for_backward(result_exp, result_log)
                return result_exp, result_log

            # 反向传播函数的静态方法
            @staticmethod
            def backward(ctx, grad_output):
                # 从上下文中加载保存的张量
                (result,) = ctx.saved_tensors
                # 返回梯度
                return grad_output * result

        # 定义一个继承自torch.nn.Module的调用者类Caller
        class Caller(torch.nn.Module):
            # 前向传播方法
            def forward(self, input):
                # 调用Parent类的apply方法
                return Parent.apply(input)

        # 创建Caller类的实例model
        model = Caller()
        # 创建一个输入张量，全为1，形状为(1, 5)
        input = torch.ones(1, 5)
        # 调用run_model_test函数测试模型，传入输入参数input
        run_model_test(self, model, input_args=(input,))

    # 以ONNX_FALLTHROUGH模式运行导出，因为torch.erf()不被支持
    def test_aten_unsupported(self):
        # 定义一个继承自torch.autograd.Function的Erf类
        class Erf(torch.autograd.Function):
            # 前向传播函数的静态方法
            @staticmethod
            def forward(ctx, x):
                # 计算输入张量x的误差函数
                erf_out = torch.special.erf(x)
                # 保存结果以便反向传播使用
                ctx.save_for_backward(erf_out)
                return erf_out

            # 反向传播函数的静态方法
            @staticmethod
            def backward(ctx, grad_output):
                # 从上下文中加载保存的张量
                result = ctx.saved_tensors
                # 返回特殊函数erfinv的反向传播结果和None
                return torch.special.erfinv(result), None

        # 定义一个继承自torch.nn.Module的调用者类Caller
        class Caller(torch.nn.Module):
            # 前向传播方法
            def forward(self, input):
                # 调用Erf类的apply方法
                return Erf.apply(input)

        # 创建Caller类的实例model
        model = Caller()
        # 创建一个输入张量，全为1，形状为(1, 5)
        input = torch.ones(1, 5)

        # 测试ONNX_FALLTHROUGH_MODE模式
        graph, _, _ = _model_to_graph(
            model,
            (input,),
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
        )
        # 获取图的迭代器，比较第一个节点的类型是否为"prim::PythonOp"
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "prim::PythonOp")

        # 测试ATEN_FALLBACK_MODE模式
        graph, _, _ = _model_to_graph(
            model,
            (input,),
            operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )
        # 获取图的迭代器，比较第一个节点的类型是否为"aten::ATen"
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "aten::ATen")
    def test_inline_and_symbolic(self):
        # 定义一个继承自torch.autograd.Function的Exp类，用于计算输入的指数函数
        class Exp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                ctx.save_for_backward(input)  # 保存输入到上下文中
                return i.exp()  # 返回输入的指数值

            @staticmethod
            def symbolic(g, input):
                return g.op("Exp", input)  # 使用符号化方法创建一个指数操作的图形表示

        # 定义一个继承自torch.autograd.Function的LogLog类，用于计算输入的双重对数
        class LogLog(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                ctx.save_for_backward(input)  # 保存输入到上下文中
                return i.log().log()  # 返回输入的双重对数值

        # 定义一个继承自torch.nn.Module的Caller类，实现前向传播函数
        class Caller(torch.nn.Module):
            def forward(self, input):
                exp_result = Exp.apply(input)  # 应用Exp类计算输入的指数函数
                return LogLog.apply(exp_result)  # 应用LogLog类计算指数函数的双重对数

        model = Caller()  # 创建Caller类的实例作为模型
        input = torch.ones(1)  # 创建输入张量，全为1

        # 设置模型的追踪映射，将模型及其类型映射为字符串表示
        torch.jit._trace._trace_module_map = {
            _m: torch.typename(type(_m)) for _m in model.modules()
        }
        run_model_test(self, model, input_args=(input,))  # 运行模型测试
        torch.jit._trace._trace_module_map = None  # 清空追踪映射
# 如果当前脚本作为主程序执行（而非被导入到其他脚本中），则执行以下代码块
if __name__ == "__main__":
    # 调用 common_utils 模块中的 run_tests 函数，通常用于执行测试套件
    common_utils.run_tests()
```