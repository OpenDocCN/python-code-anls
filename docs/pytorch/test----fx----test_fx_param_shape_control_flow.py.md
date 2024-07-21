# `.\pytorch\test\fx\test_fx_param_shape_control_flow.py`

```py
# Owner(s): ["module: fx"]

# 导入unittest模块，用于编写和运行测试用例
import unittest

# 导入PyTorch相关模块
import torch
import torch.fx

# 导入测试框架中的TestCase类，用于编写测试用例
from torch.testing._internal.common_utils import TestCase

# 定义一个自定义的PyTorch模块类MyModuleBase，继承自torch.nn.Module
class MyModuleBase(torch.nn.Module):
    # 定义模块的前向传播函数
    def forward(self, x):
        # 调用get_mul_matrix方法获取乘法矩阵
        matrx = self.get_mul_matrix()
        # 如果no_relu方法返回True，则直接返回矩阵乘法结果
        if self.no_relu():
            return torch.mm(x, matrx)
        # 否则，应用ReLU激活函数后返回结果
        else:
            return torch.relu(torch.mm(x, matrx))

    # 获取乘法矩阵的方法，由子类实现
    def get_mul_matrix(self):
        return self.param

    # 检查是否不应用ReLU的方法，由子类实现，这里抛出一个异常，提示未实现
    def no_relu(self):
        raise Exception("not implemented")  # noqa: TRY002

# 定义一个继承自MyModuleBase的子类MyModuleParamShape
class MyModuleParamShape(MyModuleBase):
    # 初始化方法，接受输入通道数，并调用父类构造函数
    def __init__(self, in_channels):
        super().__init__()
        # 初始化一个形状为(in_channels, 3)的参数矩阵
        self.param = torch.nn.Parameter(torch.randn(in_channels, 3))

    # 重写父类的no_relu方法，返回是否不应用ReLU的布尔值
    def no_relu(self):
        # 判断参数矩阵的行数是否小于10，是则返回True，否则返回False
        return self.param.shape[0] < 10

# 定义另一个继承自MyModuleBase的子类MyModuleParamSize
class MyModuleParamSize(MyModuleBase):
    # 初始化方法，接受输入通道数，并调用父类构造函数
    def __init__(self, in_channels):
        super().__init__()
        # 初始化一个形状为(in_channels, 3)的参数矩阵
        self.param = torch.nn.Parameter(torch.randn(in_channels, 3))

    # 重写父类的no_relu方法，返回是否不应用ReLU的布尔值
    def no_relu(self):
        # 判断参数矩阵的尺寸信息中的第一个维度是否小于10，是则返回True，否则返回False
        return self.param.size()[0] < 10

# 定义另一个继承自MyModuleBase的子类MyModuleParamDim
class MyModuleParamDim(MyModuleBase):
    # 初始化方法，接受一个参数，并调用父类构造函数
    def __init__(self, param):
        super().__init__()
        # 将传入的参数作为本类的param属性
        self.param = param

    # 重写父类的get_mul_matrix方法，根据参数维度返回不同的乘法矩阵
    def get_mul_matrix(self):
        return self.param[0] if (self.param.dim() == 3) else self.param

    # 重写父类的no_relu方法，返回是否不应用ReLU的布尔值
    def no_relu(self):
        # 判断参数的维度是否为3，是则返回True，否则返回False
        return self.param.dim() == 3

# 定义另一个继承自MyModuleBase的子类MyModuleParamNDim
class MyModuleParamNDim(MyModuleBase):
    # 初始化方法，接受一个参数，并调用父类构造函数
    def __init__(self, param):
        super().__init__()
        # 将传入的参数作为本类的param属性
        self.param = param

    # 重写父类的get_mul_matrix方法，根据参数维度返回不同的乘法矩阵
    def get_mul_matrix(self):
        return self.param[0] if (self.param.ndim == 3) else self.param

    # 重写父类的no_relu方法，返回是否不应用ReLU的布尔值
    def no_relu(self):
        # 判断参数的维度是否为3，是则返回True，否则返回False
        return self.param.ndim == 3

# 定义另一个继承自MyModuleBase的子类MyModuleParamNumEl
class MyModuleParamNumEl(MyModuleBase):
    # 初始化方法，接受输入通道数，并调用父类构造函数
    def __init__(self, in_channels):
        super().__init__()
        # 初始化一个形状为(in_channels, 3)的参数矩阵
        self.param = torch.nn.Parameter(torch.randn(in_channels, 3))

    # 重写父类的no_relu方法，返回是否不应用ReLU的布尔值
    def no_relu(self):
        # 判断参数矩阵的元素个数是否小于10*3，是则返回True，否则返回False
        return self.param.numel() < 10 * 3

# 定义另一个继承自MyModuleBase的子类MyModuleParamNElement
class MyModuleParamNElement(MyModuleBase):
    # 初始化方法，接受输入通道数，并调用父类构造函数
    def __init__(self, in_channels):
        super().__init__()
        # 初始化一个形状为(in_channels, 3)的参数矩阵
        self.param = torch.nn.Parameter(torch.randn(in_channels, 3))

    # 重写父类的no_relu方法，返回是否不应用ReLU的布尔值
    def no_relu(self):
        # 判断参数矩阵的元素个数是否小于10*3，是则返回True，否则返回False
        return self.param.nelement() < 10 * 3

# 定义一个测试类TestConstParamShapeInControlFlow，继承自TestCase
class TestConstParamShapeInControlFlow(TestCase):
    def verify_mm_relu_mods(self, mm_only_mod, relu_mod):
        """
        Verify one module only does a mm op while the other
        performs both mm and relu ops in cascade
        """
        # 创建一个大小为 (10, 5) 的随机张量
        x = torch.randn(10, 5)
        # 使用 torch.testing.assert_close 检验 mm_only_mod 的输出与 torch.mm(x, mm_only_mod.get_mul_matrix()) 的近似性
        torch.testing.assert_close(
            mm_only_mod(x), torch.mm(x, mm_only_mod.get_mul_matrix())
        )
        # 使用 Tracer 对 mm_only_mod 进行跟踪
        tracer = torch.fx.Tracer(param_shapes_constant=True)
        traced_graph = tracer.trace(mm_only_mod)

        # 验证图模块计算相同的结果
        graph_mod_mm = torch.fx.GraphModule(mm_only_mod, traced_graph)
        torch.testing.assert_close(
            graph_mod_mm(x), torch.mm(x, mm_only_mod.get_mul_matrix())
        )

        # 创建一个大小为 (10, 15) 的随机张量
        x = torch.randn(10, 15)
        # 使用 torch.testing.assert_close 检验 relu_mod 的输出与 torch.relu(torch.mm(x, relu_mod.get_mul_matrix())) 的近似性
        torch.testing.assert_close(
            relu_mod(x), torch.relu(torch.mm(x, relu_mod.get_mul_matrix()))
        )

        # 使用 Tracer 对 relu_mod 进行跟踪
        tracer2 = torch.fx.Tracer(param_shapes_constant=True)
        traced_graph2 = tracer2.trace(relu_mod)

        # 验证图模块计算相同的结果
        graph_mod_relu = torch.fx.GraphModule(relu_mod, traced_graph2)
        torch.testing.assert_close(
            graph_mod_relu(x), torch.relu(torch.mm(x, relu_mod.get_mul_matrix()))
        )

        # 获取 traced_graph 和 traced_graph2 中所有节点的目标函数
        graph1_node_targets = [n.target for n in traced_graph.nodes]
        graph2_node_targets = [n.target for n in traced_graph2.nodes]

        # 第二个图中有一个额外的 relu 函数调用节点
        assert torch.mm in graph1_node_targets and torch.mm in graph2_node_targets
        assert (
            torch.relu not in graph1_node_targets and torch.relu in graph2_node_targets
        )

    def test_param_shape_const(self):
        mymod = MyModuleParamShape(in_channels=5)
        mymod2 = MyModuleParamShape(in_channels=15)
        self.verify_mm_relu_mods(mymod, mymod2)

    def test_param_size_const(self):
        mymod = MyModuleParamSize(in_channels=5)
        mymod2 = MyModuleParamSize(in_channels=15)
        self.verify_mm_relu_mods(mymod, mymod2)

    def test_param_dim_const(self):
        mymod = MyModuleParamDim(torch.nn.Parameter(torch.randn(2, 5, 3)))
        mymod2 = MyModuleParamDim(torch.nn.Parameter(torch.randn(15, 3)))
        self.verify_mm_relu_mods(mymod, mymod2)

    def test_param_ndim_const(self):
        mymod = MyModuleParamNDim(torch.nn.Parameter(torch.randn(2, 5, 3)))
        mymod2 = MyModuleParamNDim(torch.nn.Parameter(torch.randn(15, 3)))
        self.verify_mm_relu_mods(mymod, mymod2)

    def test_param_numel_const(self):
        mymod = MyModuleParamNumEl(in_channels=5)
        mymod2 = MyModuleParamNumEl(in_channels=15)
        self.verify_mm_relu_mods(mymod, mymod2)

    def test_param_nelement_const(self):
        mymod = MyModuleParamNElement(in_channels=5)
        mymod2 = MyModuleParamNElement(in_channels=15)
        self.verify_mm_relu_mods(mymod, mymod2)
# 如果当前脚本被直接执行（而不是被导入到其它模块中执行），则执行以下代码
if __name__ == "__main__":
    # 运行单元测试的主函数，启动测试框架执行所有测试用例
    unittest.main()
```