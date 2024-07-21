# `.\pytorch\test\onnx\test_pytorch_jit_onnx.py`

```py
# Owner(s): ["module: onnx"]
import onnxruntime                     # 导入 ONNX 运行时库
import pytorch_test_common              # 导入 PyTorch 测试公共模块
from pytorch_test_common import skipIfNoCuda  # 从公共模块导入条件跳过装饰器

import torch                            # 导入 PyTorch 库
from torch.onnx import verification    # 导入 PyTorch ONNX 模型验证模块
from torch.onnx._globals import GLOBALS  # 导入全局设置
from torch.testing._internal import common_utils  # 导入内部测试公共工具

def _jit_graph_to_onnx_model(graph, operator_export_type, opset_version):
    r"""
    This function exports torch::jit::Graph object
    to serialized ONNX ModelProto.
    This function is for testing purpose.
    It only keeps the essential parts for IR graph conversions.
    It also does not interact with actual PyTorch modules nor
    PyTorch tensor inputs.
    """

    GLOBALS.export_onnx_opset_version = opset_version  # 设置全局 ONNX 操作集版本

    # 优化图形结构以便导出为 ONNX
    graph = torch.onnx.utils._optimize_graph(
        graph, operator_export_type, params_dict={}
    )

    # 将图形导出为 ONNX 格式
    proto, _, _, _ = graph._export_onnx(
        {},
        opset_version,
        {},
        False,
        operator_export_type,
        False,
        False,
        {},
        True,
        "",
        {},
    )
    return proto  # 返回序列化的 ONNX ModelProto 对象


class _TestJITIRToONNX:
    """Abstract base class for test cases.

    Intentionally not a sub-class of unittest.TestCase so that unittest / pytest
    don't run it directly. unitest.TestCase is mixed in as another base class when
    creating concrete sub-types. See MakeTestCase().
    """

    opset_version = -1  # Sub-classes must override
    ort_providers = ["CPUExecutionProvider"]  # ONNX 运行时提供者为 CPUExecutionProvider
    check_shape = True  # 检查输出形状
    check_dtype = True  # 检查输出数据类型
    ignore_none = True  # 对于跟踪（tracing）为 True，对于脚本化（scripting）为 False

    def run_test(self, graph_ir, example_inputs, parse_tensor_constants=False):
        # 解析输入的图形 IR，创建图形对象
        graph = torch._C.parse_ir(graph_ir, parse_tensor_constants)
        # 使用 JIT 解释图形对象，得到 JIT 输出
        jit_outs = torch._C._jit_interpret_graph(graph, example_inputs)

        # 将 JIT 图形导出为 ONNX 模型
        onnx_proto = _jit_graph_to_onnx_model(
            graph, torch.onnx.OperatorExportTypes.ONNX, self.opset_version
        )

        # 创建 ONNX 运行时推理会话
        ort_sess = onnxruntime.InferenceSession(
            onnx_proto, providers=self.ort_providers
        )

        # 在 ONNX 运行时上运行推理
        ort_outs = verification._run_onnx(ort_sess, example_inputs)

        # 设置验证选项
        options = verification.VerificationOptions(
            rtol=1e-3,
            atol=1e-7,
            check_shape=self.check_shape,
            check_dtype=self.check_dtype,
            ignore_none=self.ignore_none,
            acceptable_error_percentage=None,
        )

        # 比较 ONNX 和 JIT 的输出结果
        verification._compare_onnx_pytorch_outputs(
            ort_outs,
            jit_outs,
            options,
        )

    def test_example_ir(self):
        # 示例图形 IR
        graph_ir = """
        graph(%1 : Float(2, 3),
              %2 : Float(2, 3)):
          %3 : int = prim::Constant[value=1]()
          %4 : Float(2, 3) = aten::add(%1, %2, %3)
          return (%4)
        """
        # 创建随机输入张量
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        # 运行测试
        self.run_test(graph_ir, (a, b))
    def test_where_constants(self):
        # 定义一个描述图形计算的字符串，包含输入数据类型和设备信息
        graph_ir = """
        graph(%0 : Bool(8, device=cpu),
              %1 : Float(8, device=cpu)):
          %3 : Double(device=cpu) = prim::Constant[value={0.}]()
          %4 : Float(8) = aten::where(%0, %1, %3)
          return (%4)
        """
        # 创建两个张量，a 是大小为 8 的全零布尔型张量，b 是大小为 8 的全零浮点型张量
        a = torch.zeros(8, dtype=bool)
        b = torch.zeros(8)
        # 运行测试函数 run_test，传入 graph_ir 和参数 (a, b)，并指定解析张量常量为 True
        self.run_test(graph_ir, (a, b), parse_tensor_constants=True)

    def test_add_sub_with_graph_inputs(self):
        # 对于每个操作 op，创建描述图形计算的字符串
        for op in ["add", "sub", "rsub"]:
            graph_ir = f"""
            graph(%1 : Float(2, 3),
                  %2 : Float(2, 3),
                  %3 : int):
              %4 : Float(2, 3) = aten::{op}(%1, %2, %3)
              return (%4)
            """
            # 创建两个随机初始化的 2x3 浮点型张量 a 和 b
            a = torch.randn(2, 3)
            b = torch.randn(2, 3)
            # 运行测试函数 run_test，传入 graph_ir 和参数 (a, b, 2)
            self.run_test(graph_ir, (a, b, 2))

    def test_native_layer_norm(self):
        # 定义一个描述图形计算的字符串，包含输入数据类型和设备信息
        graph_ir = """
        graph(%x : Float(2, 3, 2),
              %w : Float(3, 2),
              %b : Float(3, 2)):
          %5 : int = prim::Constant[value=3]()
          %6 : int = prim::Constant[value=2]()
          %7 : int[] = prim::ListConstruct(%5, %6)
          %10 : float = prim::Constant[value=1.0000000000000001e-05]()
          %11 : Float(2, 3, 2), %12 : Float(2, 1, 1), %13 : Float(2, 1, 1) = aten::native_layer_norm(%x, %7, %w, %b, %10)
          return (%11, %12, %13)
        """
        # 创建三个随机初始化的张量 x、w、b
        x = torch.randn(2, 3, 2)
        w = torch.randn(3, 2)
        b = torch.randn(3, 2)
        # 运行测试函数 run_test，传入 graph_ir 和参数 (x, w, b)
        self.run_test(graph_ir, (x, w, b))

    def test_convolution(self):
        # 定义一个描述图形计算的字符串，包含输入数据类型和设备信息
        graph_ir = """
        graph(%1 : Tensor,
              %2 : Tensor):
          %3 : NoneType = prim::Constant()
          %4 : int[] = prim::Constant[value=[1, 1]]()
          %5 : int[] = prim::Constant[value=[0, 0]]()
          %6 : bool = prim::Constant[value=0]()
          %7 : int = prim::Constant[value=1]()
          %8 : Tensor = aten::convolution(%1, %2, %3, %4, %5, %4, %6, %5, %7)
          return (%8)
        """
        # 创建两个随机初始化的张量 x 和 w
        x = torch.randn(8, 1, 5, 5)
        w = torch.randn(4, 1, 3, 3)
        # 运行测试函数 run_test，传入 graph_ir 和参数 (x, w)
        self.run_test(graph_ir, (x, w))

    def test_log_softmax(self):
        # 定义一个描述图形计算的字符串，只包含一个输入张量 x
        graph_ir = """
        graph(%x: Tensor):
          %half_to_float: bool = prim::Constant[value=0]()
          %dim: int = prim::Constant[value=1]()
          %y = aten::_log_softmax(%x, %dim, %half_to_float)
          return (%y)
        """
        # 创建一个随机初始化的大小为 5x2 的张量 x
        x = torch.randn(5, 2)
        # 运行测试函数 run_test，传入 graph_ir 和参数 (x,)
        self.run_test(graph_ir, (x,))

    @skipIfNoCuda
    def test_log_softmax_half_to_float(self):
        # 定义一个描述图形计算的字符串，只包含一个输入张量 x，且指定 half_to_float 为 True
        graph_ir = """
        graph(%x: Tensor):
          %half_to_float: bool = prim::Constant[value=1]()
          %dim: int = prim::Constant[value=1]()
          %y = aten::_log_softmax(%x, %dim, %half_to_float)
          return (%y)
        """
        # 创建一个随机初始化的大小为 5x2 的半精度张量 x，转移到 CUDA 设备上
        x = torch.randn(5, 2).half().to("cuda")
        # 运行测试函数 run_test，传入 graph_ir 和参数 (x,)
        self.run_test(graph_ir, (x,))
    # 定义一个测试方法，测试本地的 dropout 函数
    def test_native_dropout(self):
        # 定义一个包含图形 IR 的字符串，描述了一个计算图
        graph_ir = """
        graph(%1 : Float(2, 3)):
          %2 : float = prim::Constant[value=0.0]()
          %training : bool = prim::Constant[value=1]()
          %3 : Tensor, %4 : Tensor = aten::native_dropout(%1, %2, %training)
          return (%3, %4)
        """
        # 生成一个 2x3 大小的随机张量
        a = torch.randn(2, 3)
        # 运行测试方法，传入生成的张量作为输入参数
        self.run_test(graph_ir, (a,))
# 定义一个函数MakeTestCase，用于动态创建测试用例类
def MakeTestCase(opset_version: int) -> type:
    # 根据给定的opset_version生成测试用例类的名称
    name = f"TestJITIRToONNX_opset{opset_version}"
    # 使用type()函数动态创建一个新的类，继承自pytorch_test_common.ExportTestCase
    # 并且继承自_TestJITIRToONNX类的所有属性和方法，同时添加opset_version属性
    return type(
        str(name),
        (pytorch_test_common.ExportTestCase,),
        dict(_TestJITIRToONNX.__dict__, opset_version=opset_version),
    )

# 创建一个名为TestJITIRToONNX_opset14的测试用例类，opset_version为14
TestJITIRToONNX_opset14 = MakeTestCase(14)

# 如果当前脚本被直接执行
if __name__ == "__main__":
    # 运行通用测试工具函数，用于执行所有测试用例
    common_utils.run_tests()
```