# `.\pytorch\test\jit\test_optimize_for_mobile_preserve_debug_info.py`

```
# Owner(s): ["oncall: mobile"]

import torch  # 导入 PyTorch 库
import torch._C  # 导入 PyTorch 的 C++ 扩展模块
import torch.nn.functional as F  # 导入 PyTorch 的函数模块
from torch.testing._internal.common_utils import skipIfNoXNNPACK  # 导入测试时的条件跳过装饰器
from torch.testing._internal.jit_utils import JitTestCase  # 导入用于 JIT 测试的基类


class TestOptimizeForMobilePreserveDebugInfo(JitTestCase):
    def check_replacement(
        self,
        model,
        replacements,
        jit_pass,
    ):
        """
        model: 要进行优化的模型
        replacements: 将优化后模型中节点类型映射到原始模型中被替换节点类型的字典
        jit_pass: 执行优化的函数
        """

        original_kinds = set(replacements.values())  # 获取被替换节点类型的集合
        original_source_ranges = {
            node.kind(): node.sourceRange()
            for node in model.graph.nodes()
            if node.kind() in original_kinds
        }  # 构建原始模型中被替换节点类型与其源代码范围的字典

        jit_pass(model._c)  # 调用 JIT 优化传递函数

        for node in model.graph.nodes():  # 遍历优化后模型的所有节点
            if node.kind() in replacements:  # 如果节点类型在替换字典中
                self.assertEqual(
                    node.sourceRange(),
                    original_source_ranges[replacements[node.kind()]],
                )  # 断言节点的源代码范围与对应原始模型中被替换节点的源代码范围一致

    @skipIfNoXNNPACK  # 如果没有 XNNPACK，则跳过测试用例
    def test_replace_conv1d_with_conv2d(self):
        class TestConv1d(torch.nn.Module):
            def __init__(self, weight, bias):
                super().__init__()
                self.weight = weight
                self.bias = bias

            def forward(self, x):
                return F.conv1d(x, self.weight, self.bias)  # 使用 F.conv1d 执行前向传播

        self.check_replacement(
            model=torch.jit.script(  # 对 TestConv1d 类进行 JIT 编译
                TestConv1d(
                    weight=torch.rand(3, 3, 3),
                    bias=torch.rand(3),
                ),
            ),
            replacements={  # 定义节点类型替换字典
                "prim::ListUnpack": "aten::conv1d",
                "prim::ListConstruct": "aten::conv1d",
                "aten::unsqueeze": "aten::conv1d",
                "aten::conv2d": "aten::conv1d",
                "aten::squeeze": "aten::conv1d",
            },
            jit_pass=torch._C._jit_pass_transform_conv1d_to_conv2d,  # 传递进行 conv1d 转换为 conv2d 的 JIT 优化函数
        )
    def test_insert_pre_packed_linear_before_inline_and_conv_2d_op(self):
        # 定义一个测试类，用于测试预打包线性层和二维卷积操作
        class TestPrepackedLinearBeforeInlineAndConv2dOp(torch.nn.Module):
            def __init__(
                self,
                linear_weight,
                linear_bias,
                conv2d_weight,
                conv2d_bias,
                conv_transpose2d_weight,
                conv_transpose2d_bias,
            ):
                super(
                    TestPrepackedLinearBeforeInlineAndConv2dOp,
                    self,
                ).__init__()
                # 初始化线性层的权重和偏置
                self.linear_weight = linear_weight.float()
                self.linear_bias = linear_bias.float()
                # 初始化二维卷积层的权重和偏置
                self.conv2d_weight = conv2d_weight.float()
                self.conv2d_bias = conv2d_bias.float()
                # 初始化二维转置卷积层的权重和偏置
                self.conv_transpose2d_weight = conv_transpose2d_weight.float()
                self.conv_transpose2d_bias = conv_transpose2d_bias.float()

            def forward(self, x):
                # 执行线性层操作
                linear_res = F.linear(
                    x.float(),
                    self.linear_weight,
                    self.linear_bias,
                )
                # 执行二维卷积操作
                conv2d_res = F.conv2d(
                    input=linear_res.unsqueeze(dim=0).float(),
                    weight=self.conv2d_weight,
                    bias=self.conv2d_bias,
                )
                # 返回二维转置卷积操作的结果
                return F.conv_transpose2d(
                    input=conv2d_res,
                    weight=self.conv_transpose2d_weight,
                    bias=self.conv_transpose2d_bias,
                )

        # 设置测试用的小批量输入数据
        minibatch = 1
        # 输入数据的通道数
        in_channels = 6
        # 输入数据的高度和宽度
        iH = 4
        iW = 5
        # 输出数据的通道数
        out_channels = 6
        # 卷积核的高度和宽度
        kH = 2
        kW = 3

        # 调用自定义的测试方法，验证预打包操作的插入情况
        self.check_replacement(
            # 使用 Torch 的 JIT 脚本编译测试类的实例
            model=torch.jit.script(
                TestPrepackedLinearBeforeInlineAndConv2dOp(
                    linear_weight=torch.rand(iW, 3),
                    linear_bias=torch.rand(iW),
                    conv2d_weight=torch.rand(out_channels, in_channels, kH, kW),
                    conv2d_bias=torch.rand(out_channels),
                    conv_transpose2d_weight=torch.rand(
                        out_channels,
                        in_channels,
                        kH,
                        kW,
                    ),
                    conv_transpose2d_bias=torch.rand(out_channels),
                ),
            ),
            # 定义预打包操作的替换映射
            replacements={
                "prepacked::linear_clamp_prepack": "aten::linear",
                "prepacked::linear_clamp_run": "aten::linear",
                "prepacked::conv2d_clamp_prepack": "aten::conv2d",
                "prepacked::conv2d_clamp_run": "aten::conv2d",
                "prepacked::conv2d_transpose_clamp_prepack": "aten::conv_transpose2d",
                "prepacked::conv2d_transpose_clamp_run": "aten::conv_transpose2d",
            },
            # 执行 JIT 传递的预打包操作插入
            jit_pass=torch._C._jit_pass_insert_prepacked_ops,
        )

    @skipIfNoXNNPACK


这段代码是一个测试方法，用于验证预打包线性层和二维卷积操作在 Torch 中的实现情况，并包含了相应的注释解释每行代码的作用。
    # 定义一个名为 test_insert_pre_packed_linear_op 的测试方法
    def test_insert_pre_packed_linear_op(self):
        # 调用 self.check_replacement 方法，检查替换逻辑
        self.check_replacement(
            # 使用 torch.jit.trace 方法对一个包含5个输入和4个输出的线性模型进行跟踪
            model=torch.jit.trace(torch.nn.Linear(5, 4), torch.rand(3, 2, 5)),
            # 指定替换字典，将 prepacked::linear_clamp_prepack 替换为 aten::linear
            replacements={
                "prepacked::linear_clamp_prepack": "aten::linear",
                # 将 prepacked::linear_clamp_run 替换为 aten::linear
                "prepacked::linear_clamp_run": "aten::linear",
            },
            # 指定 JIT pass，用于插入预打包操作
            jit_pass=torch._C._jit_pass_insert_prepacked_ops,
        )

    # 定义一个名为 run_test_fuse_activation_with_pack_ops_linear_conv2d 的测试方法
    def run_test_fuse_activation_with_pack_ops_linear_conv2d(
        self,
        linear_activation,
        linear_activation_kind,
        conv2d_activation,
        conv2d_activation_kind,
        ):
            class TestFuseActivationLinearConv2d(torch.nn.Module):
                def __init__(
                    self,
                    linear_weight,
                    linear_bias,
                    conv2d_weight,
                    conv2d_bias,
                ):
                    super().__init__()
                    self.linear_weight = linear_weight
                    self.linear_bias = linear_bias
                    self.conv2d_weight = conv2d_weight
                    self.conv2d_bias = conv2d_bias

                def forward(self, x):
                    # 执行线性变换操作
                    x = F.linear(
                        input=x,
                        weight=self.linear_weight,
                        bias=self.linear_bias,
                    )
                    # 应用线性激活函数
                    x = linear_activation(x)
                    # 执行二维卷积操作
                    x = F.conv2d(
                        input=x.unsqueeze(dim=0),
                        weight=self.conv2d_weight,
                        bias=self.conv2d_bias,
                    )
                    # 应用二维卷积激活函数
                    return conv2d_activation(x)

            # 设置线性层的输入和输出特征数
            linear_in_features = 5
            linear_out_features = 4
            # 设置二维卷积层的输入通道数、输出通道数和卷积核大小
            conv2d_in_channels = 3
            conv2d_out_channels = 4
            conv2d_kernel = 2
            # 设置输入张量的形状
            x_shape = (3, 2, 5)

            # 创建模型，并用输入张量进行 JIT 跟踪
            model = torch.jit.trace(
                TestFuseActivationLinearConv2d(
                    linear_weight=torch.nn.Parameter(
                        data=torch.rand(
                            linear_out_features,
                            linear_in_features,
                        ),
                        requires_grad=False,
                    ),
                    linear_bias=torch.nn.Parameter(
                        data=torch.rand(linear_out_features),
                        requires_grad=False,
                    ),
                    conv2d_weight=torch.rand(
                        conv2d_out_channels,
                        conv2d_in_channels,
                        conv2d_kernel,
                        conv2d_kernel,
                    ),
                    conv2d_bias=torch.rand(conv2d_out_channels),
                ),
                torch.rand(x_shape),
            )

            # 插入预打包操作
            torch._C._jit_pass_insert_prepacked_ops(model._c)

            # 检查替换情况
            self.check_replacement(
                model=model,
                replacements={
                    "prepacked::linear_clamp_prepack": "prepacked::linear_clamp_prepack",
                    "prepacked::linear_clamp_run": linear_activation_kind,
                    "prepacked::conv2d_clamp_prepack": "prepacked::conv2d_clamp_prepack",
                    "prepacked::conv2d_clamp_run": conv2d_activation_kind,
                },
                jit_pass=torch._C._jit_pass_fuse_clamp_w_prepacked_linear_conv,
            )

        @skipIfNoXNNPACK
        def test_fuse_activation_with_pack_ops_linear_conv2d_1(self):
            # 运行融合激活函数和打包操作的线性二维卷积测试
            self.run_test_fuse_activation_with_pack_ops_linear_conv2d(
                linear_activation=F.hardtanh,
                linear_activation_kind="aten::hardtanh",
                conv2d_activation=F.hardtanh_,
                conv2d_activation_kind="aten::hardtanh_",
            )

        @skipIfNoXNNPACK
    # 调用名为 test_fuse_activation_with_pack_ops_linear_conv2d_2 的测试方法，用于测试线性和二维卷积操作中的激活函数融合
    def test_fuse_activation_with_pack_ops_linear_conv2d_2(self):
        # 调用 run_test_fuse_activation_with_pack_ops_linear_conv2d 方法，设置线性激活函数为 hardtanh_，并指定其种类为 "aten::hardtanh_"
        self.run_test_fuse_activation_with_pack_ops_linear_conv2d(
            linear_activation=F.hardtanh_,
            linear_activation_kind="aten::hardtanh_",
            # 设置二维卷积激活函数为 hardtanh，并指定其种类为 "aten::hardtanh"
            conv2d_activation=F.hardtanh,
            conv2d_activation_kind="aten::hardtanh",
        )

    # 如果没有 XNNPACK，则跳过测试
    @skipIfNoXNNPACK
    def test_fuse_activation_with_pack_ops_linear_conv2d_3(self):
        # 调用 run_test_fuse_activation_with_pack_ops_linear_conv2d 方法，设置线性激活函数为 relu，并指定其种类为 "aten::relu"
        self.run_test_fuse_activation_with_pack_ops_linear_conv2d(
            linear_activation=F.relu,
            linear_activation_kind="aten::relu",
            # 设置二维卷积激活函数为 relu_，并指定其种类为 "aten::relu_"
            conv2d_activation=F.relu_,
            conv2d_activation_kind="aten::relu_",
        )

    # 如果没有 XNNPACK，则跳过测试
    @skipIfNoXNNPACK
    def test_fuse_activation_with_pack_ops_linear_conv2d_4(self):
        # 调用 run_test_fuse_activation_with_pack_ops_linear_conv2d 方法，设置线性激活函数为 relu_，并指定其种类为 "aten::relu_"
        self.run_test_fuse_activation_with_pack_ops_linear_conv2d(
            linear_activation=F.relu_,
            linear_activation_kind="aten::relu_",
            # 设置二维卷积激活函数为 relu，并指定其种类为 "aten::relu"
            conv2d_activation=F.relu,
            conv2d_activation_kind="aten::relu",
        )
```