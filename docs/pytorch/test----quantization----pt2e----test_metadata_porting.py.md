# `.\pytorch\test\quantization\pt2e\test_metadata_porting.py`

```py
# Owner(s): ["oncall: quantization"]
# 导入必要的模块和库
import copy  # 导入copy模块，用于复制对象

import unittest  # 导入unittest模块，用于编写和运行单元测试
from typing import List  # 导入List类型提示，用于类型注解

import torch  # 导入PyTorch深度学习框架
import torch._export  # 导入torch._export模块，用于导出模型
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e  # 导入量化相关函数
from torch.ao.quantization.quantizer import QuantizationAnnotation, Quantizer  # 导入量化注释和量化器类
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,  # 导入对称量化配置函数
)
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import OP_TO_ANNOTATOR  # 导入运算符到注释转换器映射表

from torch.fx import Node  # 导入Node类，用于构建和操作计算图节点

from torch.testing._internal.common_quantization import QuantizationTestCase  # 导入量化测试用例基类
from torch.testing._internal.common_utils import IS_WINDOWS  # 导入用于判断是否为Windows系统的常量

# 定义测试辅助模块类
class TestHelperModules:
    # 包含卷积、硬均值截断、自适应平均池化和全连接层的模块
    class Conv2dWithObsSharingOps(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)  # 创建3通道到3通道的卷积层
            self.hardtanh = torch.nn.Hardtanh()  # 创建硬均值截断激活函数
            self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))  # 创建自适应平均池化层
            self.linear = torch.nn.Linear(3, 3)  # 创建输入3，输出3的全连接层

        # 前向传播方法
        def forward(self, x):
            x = self.conv(x)  # 卷积操作
            x = self.adaptive_avg_pool2d(x)  # 自适应平均池化操作
            x = self.hardtanh(x)  # 硬均值截断激活函数操作
            x = x.view(-1, 3)  # 改变张量形状
            x = self.linear(x)  # 全连接操作
            return x


# 标记分区的函数
def _tag_partitions(
    backend_name: str, op_name: str, annotated_partitions: List[List[Node]]
):
    # 遍历所有的分区
    for index, partition_nodes in enumerate(annotated_partitions):
        tag_name = backend_name + "_" + op_name + "_" + str(index)  # 构建标记名
        # 遍历分区内的每个节点
        for node in partition_nodes:
            assert "quantization_tag" not in node.meta, f"{node} is already tagged"  # 检查节点是否已经标记
            node.meta["quantization_tag"] = tag_name  # 添加量化标记到节点元数据中


# 定义一组量化操作集合
_QUANT_OPS = {
    torch.ops.quantized_decomposed.quantize_per_tensor.default,  # 默认张量量化
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,  # 默认张量反量化
    torch.ops.quantized_decomposed.quantize_per_tensor.tensor,  # 张量量化
    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,  # 张量反量化
    torch.ops.quantized_decomposed.quantize_per_channel.default,  # 默认通道量化
    torch.ops.quantized_decomposed.dequantize_per_channel.default,  # 默认通道反量化
    torch.ops.quantized_decomposed.choose_qparams.tensor,  # 选择量化参数的张量版本
}


# TODO: 将名称更改为TestPortMetadataPass以与工具名称对齐？
# 元数据传输测试类，继承自QuantizationTestCase
@unittest.skipIf(IS_WINDOWS, "Windows暂不支持torch.compile")
class TestMetaDataPorting(QuantizationTestCase):
    # 测试量化标记在分解过程中的保留
    def _test_quant_tag_preservation_through_decomp(
        self, model, example_inputs, from_node_to_tags
    ):
        # 使用 Torch 的 export 模块导出模型，传入示例输入
        ep = torch.export.export(model, example_inputs)
        # 初始化标志变量为 True
        found_tags = True
        # 初始化未找到的节点为空字符串
        not_found_nodes = ""
        # 遍历 from_node_to_tags 字典中的每个键值对
        for from_node, tag in from_node_to_tags.items():
            # 遍历导出模型的图中的每个节点
            for n in ep.graph_module.graph.nodes:
                # 获取当前节点的 "from_node" 元数据
                from_node_meta = n.meta.get("from_node", None)
                # 如果元数据为 None，则继续下一个节点
                if from_node_meta is None:
                    continue
                # 如果元数据不是列表类型，则抛出值错误异常
                if not isinstance(from_node_meta, list):
                    raise ValueError(
                        f"from_node metadata is of type {type(from_node_meta)}, but expected list"
                    )
                # 遍历元数据列表中的每个元素
                for meta in from_node_meta:
                    # 获取目标节点
                    node_target = meta[1]
                    # 如果目标节点与当前 from_node 匹配
                    if node_target == from_node:
                        # 获取节点的量化标记
                        node_tag = n.meta.get("quantization_tag", None)
                        # 如果节点没有量化标记或者标记与预期不符，则添加到未找到节点的字符串中，并设置 found_tags 为 False
                        if node_tag is None or tag != node_tag:
                            not_found_nodes += str(n.target) + ", "
                            found_tags = False
                            break
                # 如果已经发现不符合预期的标记，则结束当前循环
                if not found_tags:
                    break
        # 断言 found_tags 为 True，否则输出错误信息
        self.assertTrue(
            found_tags,
            f"Decomposition did not preserve quantization tag for {not_found_nodes}",
        )

    def _test_metadata_porting(
        self,
        model,
        example_inputs,
        quantizer,
        node_tags=None,
    ) -> torch.fx.GraphModule:
        # 将模型设置为评估模式
        m_eager = model.eval()

        # 深拷贝模型的预自动微分图
        m = copy.deepcopy(m_eager)
        m = torch._export.capture_pre_autograd_graph(
            m,
            example_inputs,
        )

        # 准备将 PyTorch 模型转换为 TorchScript
        m = prepare_pt2e(m, quantizer)
        # 进行校准
        m(*example_inputs)
        # 执行 PyTorch 到 TorchScript 的转换
        m = convert_pt2e(m)

        # 使用示例输入再次运行转换后的模型，并获取量化输出
        pt2_quant_output = m(*example_inputs)
        # 记录节点的标记
        recorded_node_tags = {}
        # 遍历转换后模型的图中的每个节点
        for n in m.graph.nodes:
            # 如果节点的元数据中没有 "quantization_tag"，则继续下一个节点
            if "quantization_tag" not in n.meta:
                continue
            # 根据节点的操作类型和目标确定键值
            if n.op == "call_function" and n.target in _QUANT_OPS:
                key = n.target
            elif n.op == "get_attr":
                key = "get_attr"
            else:
                continue

            # 如果键不存在于记录的节点标记中，则初始化为集合
            if key not in recorded_node_tags:
                recorded_node_tags[key] = set()

            # 如果当前节点的标记已经在记录中，则引发值错误异常
            if (
                n.op == "call_function"
                and n.meta["quantization_tag"] in recorded_node_tags[key]
            ):
                raise ValueError(
                    f"{key} {n.format_node()} has tag {n.meta['quantization_tag']} that "
                    "is associated with another node of the same type"
                )
            # 将当前节点的标记添加到记录中
            recorded_node_tags[key].add(n.meta["quantization_tag"])

        # 断言记录的节点标记的键集合与给定的节点标记的键集合相等
        self.assertEqual(set(recorded_node_tags.keys()), set(node_tags.keys()))
        # 遍历记录的节点标记，断言其值与给定的节点标记相等
        for k, v in recorded_node_tags.items():
            self.assertEqual(v, node_tags[k])
        # 返回转换后的模型
        return m
        """
        Model under test
        conv2d -> avgpool -> hardtanh -> linear
        Check quantization tags on conv2d, avgpool and linear are correctly set
        """
        
        # 定义一个名为BackendAQuantizer的类，继承自Quantizer类
        class BackendAQuantizer(Quantizer):
            
            # annotate方法用于在图模块上进行标注
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                # 设置后端字符串为"BackendA"
                backend_string = "BackendA"
                # 获取对称量化配置，设置为按通道量化
                quantization_config = get_symmetric_quantization_config(
                    is_per_channel=True
                )
                # 对线性操作进行标注，并返回标注后的分区
                annotated_partitions = OP_TO_ANNOTATOR["linear"](
                    gm, quantization_config
                )
                # 使用后端字符串和操作类型"linear"对分区进行标记
                _tag_partitions(backend_string, "linear", annotated_partitions)
                
                # 对卷积操作进行标注，并返回标注后的分区
                annotated_partitions = OP_TO_ANNOTATOR["conv"](gm, quantization_config)
                # 使用后端字符串和操作类型"conv2d"对分区进行标记
                _tag_partitions(backend_string, "conv2d", annotated_partitions)
                
                # 对自适应平均池化操作进行标注，并返回标注后的分区
                annotated_partitions = OP_TO_ANNOTATOR["adaptive_avg_pool2d"](
                    gm, quantization_config
                )
                # 使用后端字符串和操作类型"adaptive_avg_pool2d"对分区进行标记
                _tag_partitions(
                    backend_string, "adaptive_avg_pool2d", annotated_partitions
                )
            
            # validate方法用于验证模型，但在此处未实际执行验证操作
            def validate(self, model: torch.fx.GraphModule) -> None:
                pass
        
        # 定义一个示例输入
        example_inputs = (torch.randn(1, 3, 5, 5),)
        
        # 设置获取属性标签的集合
        get_attr_tags = {
            "BackendA_conv2d_0",
            "BackendA_linear_0",
        }
        
        # 设置按张量量化的标签集合
        quantize_per_tensor_tags = {
            "BackendA_conv2d_0",
            "BackendA_adaptive_avg_pool2d_0",
            "BackendA_linear_0",
        }
        
        # 设置按张量去量化的标签集合
        dequantize_per_tensor_tags = {
            "BackendA_adaptive_avg_pool2d_0",
            "BackendA_conv2d_0",
            "BackendA_linear_0",
        }
        
        # 设置按通道去量化的标签集合
        dequantize_per_channel_tags = {"BackendA_conv2d_0", "BackendA_linear_0"}
        
        # 设置节点标签的字典
        node_tags = {
            "get_attr": get_attr_tags,
            torch.ops.quantized_decomposed.quantize_per_tensor.default: quantize_per_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: dequantize_per_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: dequantize_per_channel_tags,
        }
        
        # 调用self._test_metadata_porting方法，测试元数据的传递
        m = self._test_metadata_porting(
            TestHelperModules.Conv2dWithObsSharingOps(),
            example_inputs,
            BackendAQuantizer(),
            node_tags,
        )
        
        # 设置从节点到标签的映射关系
        from_node_to_tags = {
            torch.ops.aten.adaptive_avg_pool2d.default: "BackendA_adaptive_avg_pool2d_0",
            torch.ops.aten.linear.default: "BackendA_linear_0",
        }
        
        # 调用self._test_quant_tag_preservation_through_decomp方法，测试量化标签在分解过程中的保留
        self._test_quant_tag_preservation_through_decomp(
            m, example_inputs, from_node_to_tags
        )
    # 定义一个测试方法，用于验证在不加量化中间步骤的情况下元数据传输的功能
    def test_metadata_porting_with_no_quant_inbetween(self):
        """
        Model under test
        conv2d -> avgpool -> hardtanh -> linear
        Dont quantize avgpool
        Check quantization tags on conv2d and linear are correctly set
        """
        
        # 定义一个名为BackendAQuantizer的类，继承自Quantizer类
        class BackendAQuantizer(Quantizer):
            # 实现Quantizer类中的annotate方法，用于注解图模块gm
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                # 指定后端字符串为"BackendA"
                backend_string = "BackendA"
                # 获取对称量化配置，每个通道独立量化
                quantization_config = get_symmetric_quantization_config(
                    is_per_channel=True
                )
                # 对线性层进行注解，并将注解后的分区保存到annotated_partitions中
                annotated_partitions = OP_TO_ANNOTATOR["linear"](
                    gm, quantization_config
                )
                # 给注解后的分区打标签，标签为"BackendA_linear"
                _tag_partitions(backend_string, "linear", annotated_partitions)
                # 对卷积层进行注解，并将注解后的分区保存到annotated_partitions中
                annotated_partitions = OP_TO_ANNOTATOR["conv"](gm, quantization_config)
                # 给注解后的分区打标签，标签为"BackendA_conv2d"
                _tag_partitions(backend_string, "conv2d", annotated_partitions)

            # 实现Quantizer类中的validate方法，用于验证模型，但此处仅pass，未实现具体验证逻辑
            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        # 定义示例输入数据
        example_inputs = (torch.randn(1, 3, 5, 5),)
        
        # 定义预期的属性标签集合，包含"BackendA_conv2d_0"和"BackendA_linear_0"
        get_attr_tags = {"BackendA_conv2d_0", "BackendA_linear_0"}
        
        # 定义预期的按张量量化标签集合，包含"BackendA_conv2d_0"和"BackendA_linear_0"
        quantize_per_tensor_tags = {"BackendA_conv2d_0", "BackendA_linear_0"}
        
        # 定义预期的按张量反量化标签集合，包含"BackendA_conv2d_0"和"BackendA_linear_0"
        dequantize_per_tensor_tags = {"BackendA_conv2d_0", "BackendA_linear_0"}
        
        # 定义预期的按通道反量化标签集合，包含"BackendA_conv2d_0"和"BackendA_linear_0"
        dequantize_per_channel_tags = {"BackendA_conv2d_0", "BackendA_linear_0"}
        
        # 定义节点标签字典，指定不同操作对应的标签集合
        node_tags = {
            "get_attr": get_attr_tags,
            torch.ops.quantized_decomposed.quantize_per_tensor.default: quantize_per_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: dequantize_per_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: dequantize_per_channel_tags,
        }
        
        # 调用私有方法_test_metadata_porting，传入参数进行元数据传输测试
        self._test_metadata_porting(
            TestHelperModules.Conv2dWithObsSharingOps(),  # 测试用的模型
            example_inputs,  # 示例输入数据
            BackendAQuantizer(),  # 自定义的量化器
            node_tags,  # 节点标签字典
        )

    # 标记该测试方法为暂时禁用状态，unittest.skip用于跳过测试执行
    @unittest.skip("Temporarily disabled")
    def test_metadata_porting_for_two_dq(self):
        """
        Model under test
        conv2d -> avgpool -> hardtanh -> linear
        Quantize linear and conv with dynamic quantization
        Check quantization tags on conv2d, avgpool and linear are correctly set
        """

        class BackendAQuantizer(Quantizer):
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                backend_string = "BackendA"

                # 定义动态量化的配置，对于每个通道使用对称量化
                quantization_config_dynamic = get_symmetric_quantization_config(
                    is_per_channel=True, is_dynamic=True
                )
                # 对 conv 操作进行标注
                annotated_partitions = OP_TO_ANNOTATOR["conv"](
                    gm, quantization_config_dynamic
                )
                # 添加标签到 conv2d 的标注分区
                _tag_partitions(backend_string, "conv2d_dynamic", annotated_partitions)
                # 对 linear 操作进行标注
                annotated_partitions = OP_TO_ANNOTATOR["linear"](
                    gm, quantization_config_dynamic
                )
                # 添加标签到 linear 的标注分区
                _tag_partitions(backend_string, "linear_dynamic", annotated_partitions)

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        example_inputs = (torch.randn(1, 3, 5, 5),)
        # 定义获取属性标签
        get_attr_tags = {
            "BackendA_conv2d_dynamic_0",
            "BackendA_linear_dynamic_0",
        }
        # 定义选择量化参数张量标签
        choose_qparams_tensor_tags = {
            "BackendA_conv2d_dynamic_0",
            "BackendA_linear_dynamic_0",
        }
        # 定义对张量进行量化的张量标签
        quantize_per_tensor_tensor_tags = {
            "BackendA_conv2d_dynamic_0",
            "BackendA_linear_dynamic_0",
        }
        # 定义对张量进行反量化的张量标签
        dequantize_per_tensor_tensor_tags = {
            "BackendA_conv2d_dynamic_0",
            "BackendA_linear_dynamic_0",
        }
        # 定义对通道进行反量化的标签
        dequantize_per_channel_tags = {
            "BackendA_conv2d_dynamic_0",
            "BackendA_linear_dynamic_0",
        }
        # 定义节点标签
        node_tags = {
            "get_attr": get_attr_tags,
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: quantize_per_tensor_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: dequantize_per_tensor_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: dequantize_per_channel_tags,
            torch.ops.quantized_decomposed.choose_qparams.tensor: choose_qparams_tensor_tags,
        }
        # 执行元数据迁移测试
        self._test_metadata_porting(
            TestHelperModules.Conv2dWithObsSharingOps(),
            example_inputs,
            BackendAQuantizer(),
            node_tags,
        )
    def test_metadata_porting_for_dq_no_static_q(self):
        """
        Model under test
        conv2d -> avgpool -> hardtanh -> linear
        Dont quantize anything except linear.
        Quantize linear with dynamic quantization
        Check quantization tags on conv2d, avgpool and linear are correctly set
        """

        # 定义一个自定义的量化器类BackendAQuantizer，继承自Quantizer
        class BackendAQuantizer(Quantizer):
            
            # 实现annotate方法，对输入的torch.fx.GraphModule进行标注
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                backend_string = "BackendA"
                # 设置动态量化的配置，针对每个通道进行对称量化
                quantization_config_dynamic = get_symmetric_quantization_config(
                    is_per_channel=True, is_dynamic=True
                )
                # 调用OP_TO_ANNOTATOR字典中的"linear"标记函数，对gm进行标记，并返回标记后的模块列表
                annotated_partitions = OP_TO_ANNOTATOR["linear"](
                    gm, quantization_config_dynamic
                )
                # 给标记后的模块列表添加后端标签和动态线性量化标签
                _tag_partitions(backend_string, "linear_dynamic", annotated_partitions)

            # 实现validate方法，用于验证模型，但在此处为空实现
            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        # 准备测试所需的示例输入
        example_inputs = (torch.randn(1, 3, 5, 5),)

        # 定义各个节点标签的预期标签集合
        get_attr_tags = {"BackendA_linear_dynamic_0"}
        choose_qparams_tensor_tags = {"BackendA_linear_dynamic_0"}
        quantize_per_tensor_tensor_tags = {"BackendA_linear_dynamic_0"}
        dequantize_per_tensor_tensor_tags = {"BackendA_linear_dynamic_0"}
        dequantize_per_channel_tags = {"BackendA_linear_dynamic_0"}

        # 构建节点标签字典，将各个操作映射到其对应的预期标签集合
        node_tags = {
            "get_attr": get_attr_tags,
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: quantize_per_tensor_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: dequantize_per_tensor_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: dequantize_per_channel_tags,
            torch.ops.quantized_decomposed.choose_qparams.tensor: choose_qparams_tensor_tags,
        }

        # 调用测试函数_test_metadata_porting，验证量化标签的传递
        self._test_metadata_porting(
            TestHelperModules.Conv2dWithObsSharingOps(),  # 使用带有观察共享操作的Conv2d作为测试模型
            example_inputs,  # 使用示例输入
            BackendAQuantizer(),  # 使用自定义的BackendAQuantizer作为量化器
            node_tags,  # 期望的节点标签字典
        )
    # 定义一个测试函数，用于测试没有元数据传输的情况
    def test_no_metadata_porting(self):
        # 定义一个名为BackendAQuantizer的类，继承自Quantizer类
        class BackendAQuantizer(Quantizer):
            # 重写annotate方法，用于向图模块添加注释
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                # 设置后端字符串为"BackendA"
                backend_string = "BackendA"
                # 获取对称量化配置，设置为按通道量化
                quantization_config = get_symmetric_quantization_config(
                    is_per_channel=True
                )
                # 对线性操作应用量化注释器
                OP_TO_ANNOTATOR["linear"](gm, quantization_config)
                # 对卷积操作应用量化注释器
                OP_TO_ANNOTATOR["conv"](gm, quantization_config)
                # 对自适应平均池化2D操作应用量化注释器
                OP_TO_ANNOTATOR["adaptive_avg_pool2d"](gm, quantization_config)

            # 定义一个验证方法，用于验证模型，但未实现具体逻辑
            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        # 准备一个示例输入数据元组
        example_inputs = (torch.randn(1, 3, 5, 5),)
        # 初始化一个空的节点标签字典
        node_tags = {}
        # 调用_test_metadata_porting方法，测试元数据的传输情况，并获取返回的模型对象m
        m = self._test_metadata_porting(
            TestHelperModules.Conv2dWithObsSharingOps(),  # 传入带有操作共享的Conv2d测试模块
            example_inputs,  # 使用示例输入数据
            BackendAQuantizer(),  # 使用BackendAQuantizer量化器
            node_tags,  # 传入节点标签字典
        )

        # 初始化一个空的节点到标签的字典
        from_node_to_tags = {}
        # 调用_test_quant_tag_preservation_through_decomp方法，测试量化标签通过分解的保留情况
        self._test_quant_tag_preservation_through_decomp(
            m,  # 使用之前返回的模型对象m
            example_inputs,  # 使用示例输入数据
            from_node_to_tags  # 传入空的节点到标签字典
        )
        """
        Model under test
        matmul -> add -> relu
        matmul has get_attr as first input, but the quantization_tag should not be
        propagated to add even if it's part of a chain that ends at get_attr
        """
        
        # 定义一个继承自 torch.nn.Module 的模型类 MatmulWithConstInput
        class MatmulWithConstInput(torch.nn.Module):
            # 初始化函数，创建一个参数 w，其形状为 (8, 16)
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.rand(8, 16)))

            # 前向传播函数，接收输入 x 和 y
            def forward(self, x, y):
                # 执行矩阵乘法操作，self.w 是模型的参数
                x = torch.matmul(self.w, x)
                # 执行加法操作
                z = x + y
                # 对结果 z 应用 relu 激活函数
                return torch.nn.functional.relu(z)

        # 定义一个量化器类 BackendAQuantizer，继承自 Quantizer
        class BackendAQuantizer(Quantizer):
            # annotate 方法，用于在图模块 gm 上进行标注
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                backend_string = "BackendA"
                qconfig = get_symmetric_quantization_config()
                # 遍历图中的节点
                for n in gm.graph.nodes:
                    # 如果节点的操作不是 "call_function"，则跳过
                    if n.op != "call_function":
                        continue

                    # 在节点的 meta 属性中添加量化标注信息
                    n.meta["quantization_annotation"] = QuantizationAnnotation(
                        input_qspec_map={n.args[0]: qconfig.input_activation},
                        output_qspec=qconfig.output_activation,
                    )

                    tag = str(n.target)
                    # 在节点的 meta 属性中添加量化标签
                    n.meta["quantization_tag"] = tag
                    # 遍历节点的参数
                    for arg in n.args:
                        # 如果参数的操作是 "get_attr"
                        if arg.op == "get_attr":
                            # 在参数的 meta 属性中添加量化标签
                            arg.meta["quantization_tag"] = tag

            # validate 方法，用于验证模型
            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        # 定义示例输入
        example_inputs = (torch.randn(16, 24), torch.randn(8, 24))
        # 定义 get_attr 标签集合
        get_attr_tags = {"aten.matmul.default"}
        # 定义量化为每个张量的张量标签集合
        quantize_per_tensor_tensor_tags = {
            "aten.matmul.default",
            "aten.add.Tensor",
            "aten.relu.default",
        }
        # 定义每个张量的张量反量化标签集合
        dequantize_per_tensor_tensor_tags = {
            "aten.matmul.default",
            "aten.add.Tensor",
            "aten.relu.default",
        }
        # 定义节点标签字典
        node_tags = {
            "get_attr": get_attr_tags,
            torch.ops.quantized_decomposed.quantize_per_tensor.default: quantize_per_tensor_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: dequantize_per_tensor_tensor_tags,
        }
        # 调用测试方法 _test_metadata_porting，用于测试元数据的传递
        m = self._test_metadata_porting(
            MatmulWithConstInput(),  # 使用 MatmulWithConstInput 模型进行测试
            example_inputs,  # 使用示例输入
            BackendAQuantizer(),  # 使用 BackendAQuantizer 量化器
            node_tags,  # 使用节点标签
        )
```