# `.\pytorch\test\onnx\test_pytorch_onnx_onnxruntime.py`

```
# 所有者: ["模块: onnx"]

# 导入必要的模块和库
from __future__ import annotations  # 导入将来版本的特性支持
import functools
import io
import itertools
import os
import unittest
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import onnx  # 导入 ONNX 模块
import onnx_test_common
import parameterized
import torchvision  # 导入 PyTorch 的视觉模块
from model_defs import (
    lstm_flattening_result,
    rnn_model_with_packed_sequence,
    word_language_model,
)
from pytorch_test_common import (
    BATCH_SIZE,
    RNN_BATCH_SIZE,
    RNN_HIDDEN_SIZE,
    RNN_INPUT_SIZE,
    RNN_SEQUENCE_LENGTH,
    skipDtypeChecking,
    skipIfQuantizationBackendQNNPack,
    skipIfUnsupportedMaxOpsetVersion,
    skipIfUnsupportedMinOpsetVersion,
    skipIfUnsupportedOpsetVersion,
    skipScriptTest,
    skipShapeChecking,
    skipTraceTest,
)
import torch  # 导入 PyTorch 模块

from torch import Tensor  # 导入 PyTorch 的张量类型
from torch.nn.utils import rnn as rnn_utils
from torch.onnx import errors, verification
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import skipIfNoLapack


# 初始化函数，用于设置通用的 RCNN 变换参数
def _init_test_generalized_rcnn_transform():
    min_size = 100  # 最小尺寸
    max_size = 200  # 最大尺寸
    image_mean = [0.485, 0.456, 0.406]  # 图像均值
    image_std = [0.229, 0.224, 0.225]  # 图像标准差
    # 创建通用 RCNN 变换对象
    transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(
        min_size, max_size, image_mean, image_std
    )
    return transform


# 初始化函数，用于设置 RPN 网络的参数和对象
def _init_test_rpn():
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))  # 锚点大小
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)  # 长宽比
    # 创建 RPN 锚点生成器
    rpn_anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
        anchor_sizes, aspect_ratios
    )
    out_channels = 256  # 输出通道数
    # 创建 RPN 头部
    rpn_head = torchvision.models.detection.rpn.RPNHead(
        out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
    )
    rpn_fg_iou_thresh = 0.7  # RPN 前景 IOU 阈值
    rpn_bg_iou_thresh = 0.3  # RPN 背景 IOU 阈值
    rpn_batch_size_per_image = 256  # 每张图像的 RPN 批量大小
    rpn_positive_fraction = 0.5  # RPN 正样本比例
    rpn_pre_nms_top_n = dict(training=2000, testing=1000)  # RPN NMS 前的保留个数
    rpn_post_nms_top_n = dict(training=2000, testing=1000)  # RPN NMS 后的保留个数
    rpn_nms_thresh = 0.7  # RPN NMS 阈值
    rpn_score_thresh = 0.0  # RPN 分数阈值

    # 创建区域建议网络（RPN）对象
    rpn = torchvision.models.detection.rpn.RegionProposalNetwork(
        rpn_anchor_generator,
        rpn_head,
        rpn_fg_iou_thresh,
        rpn_bg_iou_thresh,
        rpn_batch_size_per_image,
        rpn_positive_fraction,
        rpn_pre_nms_top_n,
        rpn_post_nms_top_n,
        rpn_nms_thresh,
        score_thresh=rpn_score_thresh,
    )
    return rpn


# 辅助函数，用于生成量化测试的张量输入
def _construct_tensor_for_quantization_test(
    shape: Tuple[int, ...],  # 张量形状
    offset: Optional[Union[int, float]] = None,  # 偏移量（可选）
    max_val: Optional[Union[int, float]] = None,  # 最大值（可选）
) -> Tensor:
    """Helper function to generate weights and test inputs in a deterministic way.

    Due to difference in implementation details between PyTorch and ONNXRuntime, randomly generated
    test data for quantization tests can be flaky. To help stabilize the test, this helper function is
    """
    def generate_deterministic_tensor(shape, offset=None, max_val=None):
        """
        Generate a tensor with deterministic weights and test inputs.
    
        Args:
            shape (Tuple[int]): Shape for tensor to construct.
            offset (Optional[Union[int, float]]): Offset to be added to the generated tensor.
            max_val (Optional[Union[int, float]]): If any element within the tensor has a larger absolute value than
                max_val, the tensor will be scaled by max_val / tensor.abs().max(). This step is done after
                applying the offset.
        """
        # Create a tensor filled with values from 0 to the product of shape elements - 1
        tensor = torch.arange(np.prod(shape), dtype=torch.float).view(shape)
        # Add offset to the tensor if offset is provided
        if offset is not None:
            tensor = tensor + offset
        # Scale the tensor if max_val is provided and any element's absolute value exceeds max_val
        if max_val is not None and tensor.abs().max() > max_val:
            tensor = tensor * max_val / tensor.abs().max()
        return tensor
# 定义一个函数，用于生成带参数化类的属性和值的字典
def _parameterized_class_attrs_and_values(
    min_opset_version: int, max_opset_version: int
):
    # 定义属性列表
    attrs = ("opset_version", "is_script", "keep_initializers_as_inputs")
    # 初始化输入值列表为空
    input_values = []
    # 扩展输入值列表，使用 itertools.product 生成组合参数
    input_values.extend(itertools.product((7, 8), (True, False), (True,)))
    # 检查最小操作集版本是否小于9，如果是，则抛出异常
    if min_opset_version < 9:
        raise ValueError("min_opset_version must be >= 9")
    # 继续扩展输入值列表，生成指定范围内的参数组合
    input_values.extend(
        itertools.product(
            range(min_opset_version, max_opset_version + 1),
            (True, False),
            (True, False),
        )
    )
    # 返回包含属性和输入值的字典
    return {"attrs": attrs, "input_values": input_values}


# 定义一个函数，根据参数名称返回相应的参数化选项字典
def _parametrize_rnn_args(arg_name):
    # 定义参数名称和其对应的选项字典
    options = {
        "layers": {1: "unilayer", 3: "trilayer"},
        "bidirectional": {True: "bidirectional", False: "forward"},
        "initial_state": {True: "with_initial_state", False: "no_initial_state"},
        "packed_sequence": {
            0: "without_sequence_lengths",
            1: "with_variable_length_sequences",
            2: "with_batch_first_sequence_lengths",
        },
        "dropout": {0.2: "with_dropout", 0.0: "without_dropout"},
    }

    # 返回包含参数化选项的字典
    return {
        "arg_str": arg_name,
        "arg_values": options[arg_name].keys(),
        "name_fn": lambda val: options[arg_name][val],
    }


# 使用 parameterized.parameterized_class 装饰器和给定的参数化类属性和值，
# 以及类名生成函数和实例化参数化测试的装饰器来定义测试类
@parameterized.parameterized_class(
    **_parameterized_class_attrs_and_values(
        onnx_test_common.MIN_ONNX_OPSET_VERSION, onnx_test_common.MAX_ONNX_OPSET_VERSION
    ),
    class_name_func=onnx_test_common.parameterize_class_name,
)
@common_utils.instantiate_parametrized_tests
class TestONNXRuntime(onnx_test_common._TestONNXRuntime):
    # 定义测试方法，测试融合 Conv1d 和 BatchNorm1d 的情况
    def test_fuse_conv_bn1d(self):
        # 定义一个模型类，包含 Conv1d 和 BatchNorm1d 层
        class Fuse(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(16, 33, 3, stride=2)
                self.bn = torch.nn.BatchNorm1d(33)

            def forward(self, x):
                out = self.conv(x)
                return self.bn(out)

        # 创建模型实例和随机输入数据
        model = Fuse()
        x = torch.randn(20, 16, 50, requires_grad=True)
        # 运行测试方法
        self.run_test(model, (x,))

    # 定义测试方法，测试融合 Conv2d 和 BatchNorm2d 的情况
    def test_fuse_conv_bn2d(self):
        # 定义一个模型类，包含 Conv2d 和 BatchNorm2d 层
        class Fuse(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 2, kernel_size=1, stride=2, padding=3, bias=False
                )
                self.bn = torch.nn.BatchNorm2d(2)

            def forward(self, x):
                out = self.conv(x)
                return self.bn(out)

        # 创建模型实例和随机输入数据
        model = Fuse()
        x = torch.randn(2, 3, 2, 2, requires_grad=True)
        # 运行测试方法
        self.run_test(model, (x,))
    def test_fuse_conv_bn3d(self):
        # 定义一个名为 Fuse 的子类模块，用于融合卷积和 BatchNorm3d 操作
        class Fuse(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个 3D 卷积层，输入通道数为 3，输出通道数为 2，核大小为 (3, 5, 2)，步幅为 (2, 1, 1)，填充为 (3, 2, 0)，无偏置
                self.conv = torch.nn.Conv3d(
                    3, 2, (3, 5, 2), stride=(2, 1, 1), padding=(3, 2, 0), bias=False
                )
                # 创建一个 3D BatchNorm 层，输入通道数为 2
                self.bn = torch.nn.BatchNorm3d(2)

            def forward(self, x):
                # 执行卷积操作
                out = self.conv(x)
                # 对卷积结果执行 BatchNorm 操作
                return self.bn(out)

        # 创建 Fuse 类的实例 model
        model = Fuse()
        # 创建一个形状为 (2, 3, 10, 50, 100) 的张量 x，元素服从标准正态分布，需要梯度计算
        x = torch.randn(2, 3, 10, 50, 100, requires_grad=True)
        # 运行测试函数 run_test，传入 model 和输入 x，设置相对误差容差为 1e-3，绝对误差容差为 1e-6
        self.run_test(model, (x,), rtol=1e-3, atol=1e-6)

    def test_fuse_conv_in_block(self):
        # 定义一个名为 Fuse 的子类模块，用于融合卷积和 BatchNorm1d 操作
        class Fuse(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个 1D 卷积层，输入通道数为 5，输出通道数为 5，核大小为 3，步幅为 1，填充为 2，扩张为 1
                self.conv = torch.nn.Conv1d(
                    in_channels=5,
                    out_channels=5,
                    kernel_size=3,
                    stride=1,
                    padding=2,
                    dilation=1,
                )
                # 创建一个 1D BatchNorm 层，输入通道数为 5
                self.bn = torch.nn.BatchNorm1d(5)

            def forward(self, x):
                results_available = True

                if x.sum() > -1:
                    results_available = False

                # 若结果可用，则执行卷积操作和 BatchNorm 操作
                if results_available:
                    x = self.conv(x)
                    x = self.bn(x)

                return x

        # 创建 Fuse 类的实例 model
        model = Fuse()
        # 创建一个形状为 (2, 5, 9) 的张量 x，元素服从标准正态分布，需要梯度计算
        x = torch.randn(2, 5, 9, requires_grad=True)
        # 运行 torch.jit.script 将 model 脚本化，并传入输入 x，设置输入名称为 "x"，动态轴设置为 [0, 2]，相对误差容差为 1e-3，绝对误差容差为 1e-6
        self.run_test(
            torch.jit.script(model),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 2]},
            rtol=1e-3,
            atol=1e-6,
        )

    def test_conv_tbc(self):
        # 导入 torch.nn.modules.utils 中的 _single 函数
        from torch.nn.modules.utils import _single

        # 定义一个名为 ConvTBC 的子类模块，用于时间扩展一维卷积操作
        class ConvTBC(torch.nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, padding=0):
                super().__init__()
                # 设置输入通道数、输出通道数、核大小、填充，使用 _single 处理核大小和填充，使其可接受单一值或元组
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.kernel_size = _single(kernel_size)
                self.padding = _single(padding)

                # 创建一个权重参数，形状为 (kernel_size[0], in_channels, out_channels)，初始化为 Xavier 正态分布
                self.weight = torch.nn.Parameter(
                    Tensor(self.kernel_size[0], in_channels, out_channels)
                )
                # 创建一个偏置参数，形状为 (out_channels)，初始化为零
                self.bias = torch.nn.Parameter(Tensor(out_channels))
                # 重置模型参数
                self.reset_parameters()

            def reset_parameters(self):
                # 使用 Xavier 正态分布初始化权重参数
                torch.nn.init.xavier_normal_(self.weight)
                # 使用零初始化偏置参数
                torch.nn.init.zeros_(self.bias)

            def conv_tbc(self, input):
                # 调用 torch.conv_tbc 函数执行时间扩展的一维卷积操作，使用连续内存的输入、权重和偏置参数，以及填充
                return torch.conv_tbc(
                    input.contiguous(), self.weight, self.bias, self.padding[0]
                )

            def forward(self, input):
                # 执行时间扩展的一维卷积操作
                return self.conv_tbc(input)

        # 设置输入通道数为 3，输出通道数为 5，核大小为 5
        in_channels = 3
        out_channels = 5
        kernel_size = 5
        # 创建 ConvTBC 类的实例 model
        model = ConvTBC(in_channels, out_channels, kernel_size, padding=0)
        # 创建一个形状为 (10, 7, in_channels) 的张量 x，元素服从标准正态分布，需要梯度计算
        x = torch.randn(10, 7, in_channels, requires_grad=True)
        # 运行测试函数 run_test，传入 model 和输入 x，设置绝对误差容差为 1e-5
        self.run_test(model, (x,), atol=1e-5)
    # 定义一个测试函数，用于测试常量折叠后的重塑操作
    def test_reshape_constant_fold(self):
        # 定义一个继承自 torch.nn.Module 的类 Reshape
        class Reshape(torch.nn.Module):
            # 初始化函数
            def __init__(
                self,
            ):
                # 调用父类的初始化函数
                super().__init__()
                # 注册一个名为 weight 的缓冲区，值为全为1的大小为5的张量
                self.register_buffer("weight", torch.ones(5))

            # 前向传播函数
            def forward(self, x):
                # 将 weight 重塑为大小为 (1, -1, 1, 1) 的张量
                scale_1 = self.weight.reshape(1, -1, 1, 1)
                # 返回 x 与 scale_1 的乘积
                return x * scale_1

        # 生成一个大小为 (4, 5) 的随机张量 x
        x = torch.randn(4, 5)
        # 运行测试，传入 Reshape 类的实例和输入张量 x，设置相对误差和绝对误差的阈值
        self.run_test(Reshape(), (x,), rtol=1e-3, atol=1e-5)

    # 运行单词语言模型测试
    def run_word_language_model(self, model_name):
        # 定义模型参数
        ntokens = 50
        emsize = 5
        nhid = 5
        nlayers = 5
        dropout = 0.2
        tied = False
        batchsize = 5
        # 根据模型名称选择不同的模型
        if model_name == "GRU":
            model = word_language_model.RNNModelWithTensorHidden(
                model_name, ntokens, emsize, nhid, nlayers, dropout, tied, batchsize
            )
        elif model_name == "LSTM":
            model = word_language_model.RNNModelWithTupleHidden(
                model_name, ntokens, emsize, nhid, nlayers, dropout, tied, batchsize
            )
        else:
            model = word_language_model.RNNModel(
                model_name, ntokens, emsize, nhid, nlayers, dropout, tied, batchsize
            )
        # 生成一个大小为 (ntokens, batchsize) 的长整型张量 x
        x = torch.arange(0, ntokens).long().view(-1, batchsize)
        # 只支持 CPU 版本，因为 GPU RNN 中的追踪器不起作用
        self.run_test(model, (x, model.hidden))

    # 获取指定路径的图像，并调整大小后返回张量
    def get_image(self, rel_path: str, size: Tuple[int, int]) -> Tensor:
        # 导入必要的库
        from PIL import Image
        from torchvision import transforms

        # 获取数据目录路径
        data_dir = os.path.join(os.path.dirname(__file__), "assets")
        # 拼接图像路径
        path = os.path.join(data_dir, *rel_path.split("/"))
        # 打开图像，转换为 RGB 模式，调整大小后返回张量
        image = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)

        return transforms.ToTensor()(image)

    # 获取测试图像的张量列表
    def get_test_images(self) -> Tuple[List[Tensor], List[Tensor]:
        return (
            # 返回包含两个图像张量的元组
            [self.get_image("grace_hopper_517x606.jpg", (100, 320))],
            [self.get_image("rgb_pytorch.png", (250, 380))],
        )
    def test_paste_mask_in_image(self):
        # 创建随机的掩码数据，形状为 (10, 1, 26, 26)
        masks = torch.rand(10, 1, 26, 26)
        # 创建随机的边界框数据，形状为 (10, 4)
        boxes = torch.rand(10, 4)
        # 将边界框的宽高部分增加随机数值，并扩大 50 倍
        boxes[:, 2:] += torch.rand(10, 2)
        boxes *= 50
        # 输出图像的大小设定为 (100, 100)
        o_im_s = (100, 100)
        # 导入 torchvision 的 paste_masks_in_image 函数
        from torchvision.models.detection.roi_heads import paste_masks_in_image

        # 使用 paste_masks_in_image 函数处理掩码和边界框，得到输出
        out = paste_masks_in_image(masks, boxes, o_im_s)
        # 使用 torch.jit.trace 追踪 paste_masks_in_image 函数的运行
        jit_trace = torch.jit.trace(
            paste_masks_in_image,
            (masks, boxes, [torch.tensor(o_im_s[0]), torch.tensor(o_im_s[1])]),
        )
        # 使用追踪后的模型执行函数，得到追踪输出
        out_trace = jit_trace(
            masks, boxes, [torch.tensor(o_im_s[0]), torch.tensor(o_im_s[1])]
        )

        # 断言追踪输出和原始输出相等
        assert torch.all(out.eq(out_trace))

        # 创建另一组随机的掩码数据，形状为 (20, 1, 26, 26)
        masks2 = torch.rand(20, 1, 26, 26)
        # 创建另一组随机的边界框数据，形状为 (20, 4)
        boxes2 = torch.rand(20, 4)
        # 将边界框的宽高部分增加随机数值，并扩大 100 倍
        boxes2[:, 2:] += torch.rand(20, 2)
        boxes2 *= 100
        # 输出图像的大小设定为 (200, 200)
        o_im_s2 = (200, 200)

        # 使用 paste_masks_in_image 函数处理第二组掩码和边界框，得到输出
        out2 = paste_masks_in_image(masks2, boxes2, o_im_s2)
        # 使用之前追踪的 jit_trace 模型执行函数，得到追踪输出
        out_trace2 = jit_trace(
            masks2, boxes2, [torch.tensor(o_im_s2[0]), torch.tensor(o_im_s2[1])]
        )

        # 断言追踪输出和原始输出相等
        assert torch.all(out2.eq(out_trace2))

    def test_heatmaps_to_keypoints(self):
        # 创建随机的热图数据，形状为 (10, 1, 26, 26)
        maps = torch.rand(10, 1, 26, 26)
        # 创建随机的感兴趣区域 (ROIs)，形状为 (10, 4)
        rois = torch.rand(10, 4)
        # 导入 torchvision 的 heatmaps_to_keypoints 函数
        from torchvision.models.detection.roi_heads import heatmaps_to_keypoints

        # 使用 heatmaps_to_keypoints 函数处理热图和 ROIs，得到输出
        out = heatmaps_to_keypoints(maps, rois)
        # 使用 torch.jit.trace 追踪 heatmaps_to_keypoints 函数的运行
        jit_trace = torch.jit.trace(heatmaps_to_keypoints, (maps, rois))
        # 使用追踪后的模型执行函数，得到追踪输出
        out_trace = jit_trace(maps, rois)

        # 断言追踪输出和原始输出的第一个元素相等
        assert torch.all(out[0].eq(out_trace[0]))
        # 断言追踪输出和原始输出的第二个元素相等
        assert torch.all(out[1].eq(out_trace[1]))

        # 创建另一组随机的热图数据，形状为 (20, 2, 21, 21)
        maps2 = torch.rand(20, 2, 21, 21)
        # 创建另一组随机的 ROIs，形状为 (20, 4)
        rois2 = torch.rand(20, 4)

        # 使用 heatmaps_to_keypoints 函数处理第二组热图和 ROIs，得到输出
        out2 = heatmaps_to_keypoints(maps2, rois2)
        # 使用之前追踪的 jit_trace 模型执行函数，得到追踪输出
        out_trace2 = jit_trace(maps2, rois2)

        # 断言追踪输出和原始输出的第一个元素相等
        assert torch.all(out2[0].eq(out_trace2[0]))
        # 断言追踪输出和原始输出的第二个元素相等
        assert torch.all(out2[1].eq(out_trace2[1]))

    def test_word_language_model_RNN_TANH(self):
        # 运行名为 "RNN_TANH" 的词语言模型
        self.run_word_language_model("RNN_TANH")

    def test_word_language_model_RNN_RELU(self):
        # 运行名为 "RNN_RELU" 的词语言模型
        self.run_word_language_model("RNN_RELU")

    @skipScriptTest()  # scripting prim::unchecked_cast prim::setattr
    def test_word_language_model_LSTM(self):
        # 跳过脚本测试，原因是包含脚本化的 unchecked_cast 和 setattr 操作
        self.run_word_language_model("LSTM")

    def test_word_language_model_GRU(self):
        # 运行名为 "GRU" 的词语言模型
        self.run_word_language_model("GRU")

    def test_index_1d(self):
        # 定义一个简单的模型，用于返回输入的第一个元素
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[0]

        # 创建一个随机的张量 m1，形状为 (3, 4, 5, 6, 7)
        m1 = torch.randn(3, 4, 5, 6, 7)
        # 使用 self.run_test 方法运行定义的模型 MyModel，并传入 m1 作为输入
        self.run_test(MyModel(), m1)

    def test_index_2d_1dimslice(self):
        # 定义一个简单的模型，用于返回输入的第一行，所有列
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[0:1, :]

        # 创建一个随机的张量 m1，形状为 (3, 4, 5, 6, 7)
        m1 = torch.randn(3, 4, 5, 6, 7)
        # 使用 self.run_test 方法运行定义的模型 MyModel，并传入 m1 作为输入
        self.run_test(MyModel(), m1)
    # 定义一个测试函数，用于测试二维索引和切片操作
    def test_index_2d_sliceint(self):
        # 定义一个简单的 PyTorch 模型类 MyModel，重写 forward 方法实现索引操作
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[1, :]  # 返回输入张量的第一行所有列数据

        # 创建一个形状为 (3, 4, 5, 6, 7) 的随机张量 m1
        m1 = torch.randn(3, 4, 5, 6, 7)
        # 调用 run_test 方法，传入 MyModel 实例和随机张量 m1 进行测试
        self.run_test(MyModel(), m1)

    # 定义一个测试函数，用于测试带负索引的二维切片操作
    def test_index_2d_neg_slice(self):
        # 定义一个简单的 PyTorch 模型类 MyModel，重写 forward 方法实现带负索引的切片操作
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[0:-1, :]  # 返回输入张量的从第一行到倒数第二行的所有列数据

        # 创建一个形状为 (3, 4, 5, 6, 7) 的随机张量 m1
        m1 = torch.randn(3, 4, 5, 6, 7)
        # 调用 run_test 方法，传入 MyModel 实例和随机张量 m1 进行测试
        self.run_test(MyModel(), m1)

    # 定义一个测试函数，用于测试使用索引掩码的操作
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_mask(self):
        # 定义一个简单的 PyTorch 模型类 MyModel，重写 forward 方法实现使用索引掩码的操作
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[torch.tensor([0, 1, 0], dtype=torch.uint8)]  # 使用索引掩码选择第 0、1、0 行的数据

        # 创建一个形状为 (3, 4, 5, 6, 7) 的随机张量 m1
        m1 = torch.randn(3, 4, 5, 6, 7)
        # 调用 run_test 方法，传入 MyModel 实例和随机张量 m1 进行测试
        self.run_test(MyModel(), m1)

        # 定义另一个简单的 PyTorch 模型类 MyModel，重写 forward 方法实现使用布尔类型的索引掩码的操作
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[torch.tensor([0, 1, 0], dtype=torch.bool)]  # 使用布尔类型的索引掩码选择第 0、1、0 行的数据

        # 创建一个形状为 (3, 4, 5, 6, 7) 的随机张量 m1
        m1 = torch.randn(3, 4, 5, 6, 7)
        # 调用 run_test 方法，传入 MyModel 实例和随机张量 m1 进行测试
        self.run_test(MyModel(), m1)

    # 定义一个测试函数，用于测试数据处理的操作
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_data(self):
        # 定义一个简单的 TorchScript 模块类 Data，重写 forward 方法实现数据处理操作
        class Data(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return x.new_zeros(x.data.size())  # 返回形状与输入 x 相同的零张量

        # 创建一个形状为 (3, 4) 的随机张量 x
        x = torch.randn(3, 4)
        # 调用 run_test 方法，传入 Data 类的实例和随机张量 x 进行测试，并指定输入名称和动态轴信息
        self.run_test(Data(), x, input_names=["x"], dynamic_axes={"x": [0, 1]})
        # 再次调用 run_test 方法，传入 Data 类的实例和随机张量 x 进行测试，并指定 remained_onnx_input_idx 为空列表
        self.run_test(Data(), x, remained_onnx_input_idx=[])

    # 定义一个测试函数，用于测试多维索引掩码的操作
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_mask_nd(self):
        # 定义一个简单的 PyTorch 模型类 MyModel，重写 forward 方法实现多维索引掩码的操作
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[input > 0]  # 返回输入张量中大于 0 的所有元素

        # 创建一个形状为 (3, 4, 5, 6, 7) 的随机张量 m1
        m1 = torch.randn(3, 4, 5, 6, 7)
        # 调用 run_test 方法，传入 MyModel 实例和随机张量 m1 进行测试
        self.run_test(MyModel(), m1)

    # 定义一个测试函数，用于测试字典操作
    @skipScriptTest()  # 用户定义的类不受支持
    def test_dict(self):
        # 定义一个简单的 PyTorch 模型类 MyModel，重写 forward 方法实现字典操作
        class MyModel(torch.nn.Module):
            def forward(self, x_in):
                x_out = {}
                # 将输入字典中的第一个键对应的值加上这个键，存入输出字典的 "test_key_out" 键中
                x_out["test_key_out"] = torch.add(
                    x_in[list(x_in.keys())[0]], list(x_in.keys())[0]  # noqa: RUF015
                )
                return x_out

        # 创建一个键为 tensor(1.0)，值为形状为 (1, 2, 3) 的随机张量的输入字典 x
        x = {torch.tensor(1.0): torch.randn(1, 2, 3)}
        # 调用 run_test 方法，传入 MyModel 实例和输入元组 (x,) 进行测试
        self.run_test(MyModel(), (x,))

    # 定义一个测试函数，用于测试字典操作（字符串键）
    @skipScriptTest()  # 用户定义的类不受支持
    def test_dict_str(self):
        # 定义一个简单的 PyTorch 模型类 MyModel，重写 forward 方法实现字典操作
        class MyModel(torch.nn.Module):
            def forward(self, x_in):
                x_out = {}
                # 将输入字典中 "test_key_in" 键对应的值加上 2.0，存入输出字典的 "test_key_out" 键中
                x_out["test_key_out"] = torch.add(x_in["test_key_in"], 2.0)
                return x_out

        # 创建一个键为 "test_key_in"，值为形状为 (1, 2, 3) 的随机张量的输入字典 x
        x = {"test_key_in": torch.randn(1, 2, 3)}
        # 调用 run_test 方法，传入 MyModel 实例和输入元组 (x,) 进行测试
        self.run_test(MyModel(), (x,))
    def test_dict_output(self):
        # 定义一个继承自OrderedDict的类，用于输出模型的字典格式结果
        class DictModelOutput(OrderedDict):
            tensor_out: Tensor  # 输出的张量
            tuple_out: Optional[Tuple[Tensor]] = None  # 可选的元组输出
            list_out: Optional[List[Tensor]] = None  # 可选的列表输出

        class MyModel(torch.nn.Module):
            def forward(self, a, b, c, d):
                # 返回一个DictModelOutput对象，包含tensor_out, tuple_out, list_out三个字段
                return DictModelOutput(
                    tensor_out=a,
                    tuple_out=(b, c),
                    list_out=[d],
                )

        # 生成四个随机张量
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)
        d = torch.randn(2, 3)
        # 运行测试，验证模型输出
        self.run_test(MyModel(), (a, b, c, d))

    def test_tuple_output(self):
        class MyModel(torch.nn.Module):
            def forward(self, a, b, c, d):
                # 直接返回一个元组，包含a, (b, c), d三个元素
                return a, (b, c), d

        # 生成四个随机张量
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)
        d = torch.randn(2, 3)
        # 运行测试，验证模型输出
        self.run_test(MyModel(), (a, b, c, d))

    def test_nested_tuple_output(self):
        class MyModel(torch.nn.Module):
            def forward(self, a, b, c, d):
                # 返回一个嵌套的元组结构
                return a, ((b,), (c, d))

        # 生成四个随机张量
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)
        d = torch.randn(2, 3)
        # 运行测试，验证模型输出
        self.run_test(MyModel(), (a, b, c, d))

    def test_tuple_input(self):
        class TupleModel(torch.nn.Module):
            def forward(self, a: Tuple[Tensor, Tensor]):
                # 直接返回输入的元组a
                return a

        # 定义一个包含两个张量的元组
        x = (torch.randn(3, 4), torch.randn(4, 3))
        # 运行测试，验证模型输出
        self.run_test(TupleModel(), input_args=(x,))

    def test_tuple_primitive_input(self):
        class TupleModel(torch.nn.Module):
            def forward(self, a: Tuple[int, Tensor], b):
                # 返回元组a中的第一个元素和a的第二个元素与b的和
                return a[0], a[1] + b

        # 定义一个包含一个整数和一个张量的元组
        x = (3, torch.randn(4, 3))
        y = torch.randn(4, 3)
        # 运行测试，验证模型输出
        self.run_test(TupleModel(), input_args=(x, y))

    def test_nested_tuple_input(self):
        class NestedTupleModel(torch.nn.Module):
            def forward(self, a, b: Tuple[Tensor, Tuple[Tensor, Tensor]]):
                # 返回a与b的第一个元素，以及b的第一个和第二个元素的第一个和第二个元素的和
                return a + b[0] + b[1][0] + b[1][1]

        # 定义一个张量和一个包含两层嵌套元组的元组
        x = torch.randn(4, 5)
        y = (torch.randn(4, 5), (torch.randn(1, 5), torch.randn(4, 1)))
        # 运行测试，验证模型输出
        self.run_test(NestedTupleModel(), input_args=(x,))

    @skipScriptTest()  # 需要 https://github.com/pytorch/rfcs/pull/21 的支持
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_mixed_optional_default_none(self):
        class Model(torch.nn.Module):
            def forward(
                self,
                x,
                y: Optional[Tensor] = None,
                z: Optional[Tensor] = None,
            ):
                # 如果 y 不为 None，则返回 x + y
                if y is not None:
                    return x + y
                # 如果 z 不为 None，则返回 x + z
                if z is not None:
                    return x + z
                # 否则返回 x
                return x

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)
        model = Model()
        # 使用 run_test 方法测试模型，传入 x, y 和 None
        self.run_test(model, (x, y, None))
        # 使用 run_test 方法测试模型，传入 x, None 和 z
        self.run_test(model, (x, None, z))
        # 使用 run_test 方法测试模型，传入 x 和 kwargs 字典 {"y": y, "z": None}
        self.run_test(model, (x,), {"y": y, "z": None})
        # 使用 run_test 方法测试模型，传入 x 和 kwargs 字典 {"y": None, "z": z}
        self.run_test(model, (x,), {"y": None, "z": z})
        # 使用 run_test 方法测试模型，传入 x 和 kwargs 字典 {"z": z}
        self.run_test(model, (x,), {"z": z})
        # 使用 run_test 方法测试模型，传入 x 和 kwargs 字典 {"y": y}
        self.run_test(model, (x,), {"y": y})

    @skipScriptTest()  # 脚本测试被跳过，因为 None 输入在跟踪时会被消除，表现会不同。请参考下面的 _script 版本。
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_mixed_optional_default_tensor(self):
        class Model(torch.nn.Module):
            def forward(
                self,
                x,
                y: Optional[Tensor] = torch.ones(2, 3),
                z: Optional[Tensor] = torch.zeros(2, 3),
            ):
                # 如果 y 不为 None，则返回 x + y
                if y is not None:
                    return x + y
                # 如果 z 不为 None，则返回 x + z
                if z is not None:
                    return x + z
                # 否则返回 x
                return x

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)
        model = Model()

        # 使用 run_test 方法测试模型，传入 x, y 和 None
        self.run_test(model, (x, y, None))
        # 使用 run_test 方法测试模型，传入 x, None 和 z
        self.run_test(model, (x, None, z))

    @skipTraceTest()  # 跟踪测试被跳过，验证时使用不同的输入集合。请参考上面的说明。
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_mixed_optional_default_tensor_script(self):
        # 定义一个模型类，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            # 前向传播方法
            def forward(
                self,
                x,  # 输入张量 x
                y: Optional[Tensor] = torch.ones(2, 3),  # 可选参数 y，默认为全1张量
                z: Optional[Tensor] = torch.zeros(2, 3),  # 可选参数 z，默认为全0张量
            ):
                # 如果 y 不为 None，则返回 x + y
                if y is not None:
                    return x + y
                # 如果 y 为 None 且 z 不为 None，则返回 x + z
                if z is not None:
                    return x + z
                # 如果 y 和 z 都为 None，则返回 x
                return x

        # 创建输入张量 x
        x = torch.randn(2, 3)
        # 创建不同的张量 y 和 z
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)
        # 使用 torch.jit.script 将模型转换为脚本模型
        model = torch.jit.script(Model())

        # 运行测试，传入 x, y, z，指定输入名称为 "x", "y", "z"
        self.run_test(model, (x, y, z), input_names=("x", "y", "z"))
        # 运行测试，传入 x 和 y，并将 y 作为关键字参数，指定输入名称为 "x", "y", "z"
        self.run_test(model, (x,), {"y": y, "z": z}, input_names=("x", "y", "z"))
        # 运行测试，传入 x 和 y，并将 y 作为关键字参数，指定输入名称为 "x", "y"
        self.run_test(model, (x,), {"y": y}, input_names=("x", "y"))

        # 遍历不同的例子输入和关键字参数组合
        for example_inputs, example_kwargs in (
            ((x, y, None), {}),
            ((x, None, z), {}),
            ((x,), {"y": y, "z": None}),
            ((x,), {"y": None, "z": z}),
        ):
            # 使用 self.assertRaisesRegex 检查是否抛出 ValueError 异常，并验证异常消息
            with self.assertRaisesRegex(
                ValueError, "args contained 1 None's after flattening."
            ):
                # 运行测试，传入 example_inputs 和 example_kwargs，指定输入名称为 "x", "y", "z"
                self.run_test(
                    model, example_inputs, example_kwargs, input_names=("x", "y", "z")
                )

    @skipScriptTest()  # 需要 https://github.com/pytorch/rfcs/pull/21
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_all_optional_default_none(self):
        # 定义一个模型类，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            # 前向传播方法
            def forward(self, x: Optional[Tensor] = None, y: Optional[Tensor] = None):
                # 如果 x 不为 None，则返回 x
                if x is not None:
                    return x
                # 如果 y 不为 None，则返回 y
                if y is not None:
                    return y
                else:
                    # 如果 x 和 y 都为 None，则返回一个包含值 -1.0 的张量
                    return torch.tensor(-1.0)

        # 创建输入张量 x
        x = torch.randn(2, 3)
        # 创建模型实例
        model = Model()
        # 运行测试，传入 x 和 None
        self.run_test(model, (x, None))
        # 运行测试，传入空元组，并将 x 作为关键字参数，y 为 None，指定输入名称为 "x"
        self.run_test(
            model,
            (),
            {"x": x, "y": None},
            # 在追踪过程中，y 将被忽略
            input_names=("x",),
        )

    @skipScriptTest()  # 追踪过程中会消除 None 输入，因此行为不同。参见下面的 _script 版本。
    @skipIfUnsupportedMinOpsetVersion(15)
    # 定义一个测试方法，测试所有可选的默认张量参数情况
    def test_all_optional_default_tensor(self):
        # 定义一个继承自 torch.nn.Module 的模型类
        class Model(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(
                self,
                x: Optional[Tensor] = torch.ones(2, 3),  # 默认为一个2x3的全1张量，可选参数
                y: Optional[Tensor] = torch.zeros(2, 3),  # 默认为一个2x3的全0张量，可选参数
            ):
                # 如果 x 不为 None，则返回 x
                if x is not None:
                    return x
                # 如果 x 为 None 且 y 不为 None，则返回 y
                elif y is not None:
                    return y
                # 如果 x 和 y 都为 None，则返回一个值为 -1.0 的张量
                else:
                    return torch.tensor(-1.0)

        # 创建两个随机的2x3张量 x 和 y
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        # 创建模型对象
        model = Model()
        # 使用给定的输入 x 和 None 测试模型
        self.run_test(model, (x, None))
        # 使用给定的输入 None 和 y 测试模型
        self.run_test(model, (None, y))
        # 使用 x 和 y 作为输入，预期会出现异常，因为位置输入过多
        with self.assertRaisesRegex(ValueError, "got too many positional inputs"):
            self.run_test(model, (x, y))

    # 跳过追踪测试，这些测试在不同输入情况下已验证过。参见上文。
    @skipTraceTest()
    # 如果操作集的最小版本不支持15，则跳过测试
    @skipIfUnsupportedMinOpsetVersion(15)
    # 定义一个测试方法，测试所有可选的默认张量参数情况（脚本化版本）
    def test_all_optional_default_tensor_script(self):
        # 定义一个继承自 torch.nn.Module 的模型类
        class Model(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(
                self,
                x: Optional[Tensor] = torch.ones(2, 3),  # 默认为一个2x3的全1张量，可选参数
                y: Optional[Tensor] = torch.zeros(2, 3),  # 默认为一个2x3的全0张量，可选参数
            ):
                # 如果 x 不为 None，则返回 x
                if x is not None:
                    return x
                # 如果 x 为 None 且 y 不为 None，则返回 y
                elif y is not None:
                    return y
                # 如果 x 和 y 都为 None，则返回一个值为 -1.0 的张量
                else:
                    return torch.tensor(-1.0)

        # 创建两个随机的2x3张量 x 和 y
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        # 对模型进行脚本化（转换为 Torch 脚本）
        model = torch.jit.script(Model())

        # 使用给定的输入 x 测试模型，y 使用 None 表示可选输入
        self.run_test(model, (x,))
        # 注意：默认值在 ONNX 中不受支持，因此 Torch 和 ONNX 的行为不同
        with self.assertRaisesRegex(AssertionError, "Tensor-likes are not close!"):
            # 使用空输入和给定的 y 输入测试模型，指定输入名称为 "y"
            self.run_test(model, (), {"y": y}, input_names=["y"])

        # 使用给定的输入 x 和 y 测试模型
        self.run_test(model, (x, y))
        # 使用空输入和给定的 x、y 输入测试模型，指定输入名称为 "x" 和 "y"
        self.run_test(model, (), {"x": x, "y": y}, input_names=("x", "y"))

    # 如果操作集的最小版本不支持9，则跳过测试
    @skipIfUnsupportedMinOpsetVersion(9)
    # 定义一个测试方法，测试对数函数
    def test_logit(self):
        # 定义一个继承自 torch.nn.Module 的模型类
        class Logit(torch.nn.Module):
            # 初始化方法，设置 eps 参数
            def __init__(self, eps):
                super().__init__()
                self.eps = eps

            # 定义模型的前向传播方法
            def forward(self, x):
                # 返回 x 的对数函数，带有 eps 参数
                return x.logit(self.eps)

        # 创建 Logit 模型对象，设置 eps 为 1e-6
        model = Logit(eps=1e-6)
        # 使用随机生成的1x3x640x640张量输入测试模型
        self.run_test(model, torch.randn(1, 3, 640, 640))

    # 定义一个继承自 torch.nn.Module 的类，实现至少1维、2维和3维张量的转换
    class Atleast1d(torch.nn.Module):
        # 定义模型的前向传播方法，接受 t, w, x, y, z 五个参数
        def forward(self, t, w, x, y, z):
            # 返回至少1维的张量
            return torch.atleast_1d((t, w, x, y, z))

    class Atleast2d(torch.nn.Module):
        # 定义模型的前向传播方法，接受 t, w, x, y, z 五个参数
        def forward(self, t, w, x, y, z):
            # 返回至少2维的张量
            return torch.atleast_2d((t, w, x, y, z))

    class Atleast3d(torch.nn.Module):
        # 定义模型的前向传播方法，接受 t, w, x, y, z 五个参数
        def forward(self, t, w, x, y, z):
            # 返回至少3维的张量
            return torch.atleast_3d((t, w, x, y, z))

    # 定义一个继承自 torch.nn.Module 的类，实现至少1维张量转换
    class Atleast1dTensor(torch.nn.Module):
        # 定义模型的前向传播方法，接受 x 一个参数
        def forward(self, x):
            # 返回至少1维的张量
            return torch.atleast_1d(x)
    @skipScriptTest()  # 跳过脚本测试，因为跟踪使用 prim::ListUnpack 来避免 onnx::SequenceConstruct
    @skipIfUnsupportedMinOpsetVersion(11)  # 如果不支持最小 Opset 版本 11 则跳过测试

    @common_utils.parametrize("module_class", (Atleast1d, Atleast2d, Atleast3d))
    # 使用 common_utils.parametrize 装饰器，参数化 module_class，可以是 Atleast1d、Atleast2d 或 Atleast3d 中的一个
    def test_atleast_nd_list_input(self, module_class: torch.nn.Module):
        inputs = (
            torch.tensor(1.0),  # 创建一个标量张量
            torch.randn(2),     # 创建一个形状为 (2,) 的张量
            torch.randn(2, 3),  # 创建一个形状为 (2, 3) 的张量
            torch.randn(2, 3, 4),  # 创建一个形状为 (2, 3, 4) 的张量
            torch.randn(2, 3, 4, 5),  # 创建一个形状为 (2, 3, 4, 5) 的张量
        )
        self.run_test(module_class(), inputs)  # 运行测试函数，传入 module_class 和 inputs 作为参数

    @skipScriptTest()  # 跳过脚本测试，因为跟踪使用 prim::ListUnpack 来避免 onnx::SequenceConstruct
    @skipIfUnsupportedMinOpsetVersion(11)  # 如果不支持最小 Opset 版本 11 则跳过测试

    @common_utils.parametrize(
        "module_class", (Atleast1dTensor, Atleast2dTensor, Atleast3dTensor)
    )
    # 使用 common_utils.parametrize 装饰器，参数化 module_class，可以是 Atleast1dTensor、Atleast2dTensor 或 Atleast3dTensor 中的一个
    @common_utils.parametrize(
        "inputs",
        [
            torch.tensor(1.0),      # 创建一个标量张量
            torch.randn(2),         # 创建一个形状为 (2,) 的张量
            torch.randn(2, 3),      # 创建一个形状为 (2, 3) 的张量
            torch.randn(2, 3, 4),   # 创建一个形状为 (2, 3, 4) 的张量
            torch.randn(2, 3, 4, 5),  # 创建一个形状为 (2, 3, 4, 5) 的张量
        ],
    )
    def test_atleast_nd_single_tensor_input(
        self, module_class: torch.nn.Module, inputs: torch.Tensor
    ):
        self.run_test(module_class(), inputs)  # 运行测试函数，传入 module_class 和 inputs 作为参数

    @skipScriptTest()  # 需要 https://github.com/pytorch/rfcs/pull/21 才能运行
    @skipIfUnsupportedMinOpsetVersion(15)  # 如果不支持最小 Opset 版本 15 则跳过测试
    def test_mixed_optional(self):
        class Model(torch.nn.Module):
            def forward(self, x, y: Optional[Tensor]):
                if y is not None:
                    return x + y
                return x

        x = torch.randn(2, 3)  # 创建一个形状为 (2, 3) 的张量
        model = Model()
        self.run_test(model, (x, None))  # 运行测试函数，传入 model 和 (x, None) 作为参数
        self.run_test(model, (x, x))    # 运行测试函数，传入 model 和 (x, x) 作为参数

    @skipScriptTest()  # 需要 https://github.com/pytorch/rfcs/pull/21 才能运行
    @skipIfUnsupportedMinOpsetVersion(15)  # 如果不支持最小 Opset 版本 15 则跳过测试
    def test_tuple_of_optional(self):
        class Model(torch.nn.Module):
            def forward(self, x, y: Tuple[Optional[Tensor], Optional[Tensor]]):
                if y[0] is not None:
                    return x + y[0]
                if y[1] is not None:
                    return x + y[1]
                return x

        x = torch.randn(2, 3)  # 创建一个形状为 (2, 3) 的张量
        y1 = torch.randn(2, 3)  # 创建一个形状为 (2, 3) 的张量
        self.run_test(Model(), (x, (None, y1)))  # 运行测试函数，传入 Model() 和 (x, (None, y1)) 作为参数

    @skipScriptTest()  # 跟踪消除了 None 输入，因此工作方式不同。参见下面的 _script 版本。
    @skipIfUnsupportedMinOpsetVersion(15)  # 如果不支持最小 Opset 版本 15 则跳过测试
    # 定义一个测试方法，测试包含可选默认张量的元组参数的模型
    def test_tuple_of_optional_default_tensor(self):
        # 定义一个继承自torch.nn.Module的模型类
        class Model(torch.nn.Module):
            # 前向传播方法，接受输入参数x和可选元组参数y，默认为两个2x3的零张量
            def forward(
                self,
                x,
                y: Tuple[Optional[Tensor], Optional[Tensor]] = (
                    torch.zeros(2, 3),
                    torch.zeros(2, 3),
                ),
            ):
                # 解包元组y
                y0, y1 = y
                # 如果y0不为None，返回x + y0
                if y0 is not None:
                    return x + y0
                # 如果y1不为None，返回x + y1
                if y1 is not None:
                    return x + y1
                # 如果y0和y1都为None，返回x
                return x

        # 生成一个形状为2x3的随机张量x
        x = torch.randn(2, 3)
        # 生成一个形状为2x3的随机张量y1
        y1 = torch.randn(2, 3)
        # 使用self.run_test方法测试Model类的实例，传入x和元组(None, y1)
        self.run_test(Model(), (x, (None, y1)))

    # 使用skipTraceTest装饰器，用于跳过跟踪测试
    # 通过不同的输入集合验证跟踪。
    # 参见上文。
    @skipTraceTest()
    # 如果操作集版本小于15，则跳过该测试
    @skipIfUnsupportedMinOpsetVersion(15)
    # 定义一个测试方法，测试包含可选默认张量的元组参数的脚本化模型
    def test_tuple_of_optional_default_tensor_script(self):
        # 定义一个继承自torch.nn.Module的模型类
        class Model(torch.nn.Module):
            # 前向传播方法，接受输入参数x和可选元组参数y，默认为两个2x3的零张量
            def forward(
                self,
                x,
                y: Tuple[Optional[Tensor], Optional[Tensor]] = (
                    torch.zeros(2, 3),
                    torch.zeros(2, 3),
                ),
            ):
                # 解包元组y
                y0, y1 = y
                # 如果y0不为None，返回x + y0
                if y0 is not None:
                    return x + y0
                # 如果y1不为None，返回x + y1
                if y1 is not None:
                    return x + y1
                # 如果y0和y1都为None，返回x
                return x

        # 生成一个形状为2x3的随机张量x
        x = torch.randn(2, 3)
        # 生成一个形状为2x3的随机张量y0
        y0 = torch.randn(2, 3)
        # 生成一个形状为2x3的随机张量y1
        y1 = torch.randn(2, 3)
        # 将Model类实例化为脚本模型
        model = torch.jit.script(Model())
        # 使用self.assertRaisesRegex断言捕获ValueError异常，检查是否包含特定消息
        # 错误消息为"args contained 1 None's after flattening."
        with self.assertRaisesRegex(
            ValueError, "args contained 1 None's after flattening."
        ):
            # 使用self.run_test方法测试model，传入x和元组(None, y1)
            self.run_test(model, (x, (None, y1)))
        # 使用self.run_test方法测试model，传入x和元组(y0, y1)
        self.run_test(model, (x, (y0, y1)))
        # 输出ONNX格式的导出结果，但通过run_test运行ORT会失败，
        # 因为导出的模型将输入展平为3个输入。
        torch.onnx.export(
            # 导出model模型
            model,
            # 传入元组(x, {"y": (y0, y1)})
            (x, {"y": (y0, y1)}),
            # 导出到字节流
            io.BytesIO(),
            # 设置操作集版本为self.opset_version
            opset_version=self.opset_version
        )

    # 定义一个测试方法，测试包含整数类型输入的模型
    def test_primitive_input_integer(self):
        # 定义一个继承自torch.nn.Module的模型类
        class Model(torch.nn.Module):
            # 前向传播方法，接受整数类型输入x和任意类型输入y，返回x + y
            def forward(self, x: int, y):
                return x + y

        # 设置x为整数值3
        x = 3
        # 生成一个形状为(2, 3, 4)的随机整数张量y
        y = torch.randint(10, (2, 3, 4))
        # 使用self.run_test方法测试Model类的实例，传入x和y
        self.run_test(Model(), (x, y))

    # 使用skipDtypeChecking装饰器，用于跳过数据类型检查
    # 定义一个测试方法，测试包含浮点数类型输入的模型
    def test_primitive_input_floating(self):
        # 定义一个继承自torch.nn.Module的模型类
        class Model(torch.nn.Module):
            # 前向传播方法，接受浮点数类型输入x和任意类型输入y，返回x + y
            def forward(self, x: float, y):
                return x + y

        # 设置x为浮点数值3.0
        x = 3.0
        # 生成一个形状为(2, 3, 4)的随机浮点数张量y
        y = torch.randn(2, 3, 4)
        # 使用self.run_test方法测试Model类的实例，传入x和y
        self.run_test(Model(), (x, y))

    # 定义一个测试方法，测试包含布尔类型输入的模型
    def test_primitive_input_bool(self):
        # 定义一个继承自torch.nn.Module的模型类
        class Model(torch.nn.Module):
            # 前向传播方法，接受布尔类型输入flag、任意类型输入x和y，
            # 如果flag为True，返回x；否则，返回y
            def forward(self, flag: bool, x, y):
                if flag:
                    return x
                else:
                    return y

        # 设置flag为True
        flag = True
        # 生成一个形状为(2, 3, 4)的随机浮点数张量x
        x = torch.randn(2, 3, 4)
        # 生成一个形状为(2, 3, 4)的随机浮点数张量y
        y = torch.randn(2, 3, 4)
        # 使用torch.jit.script方法将Model类实例化为脚本模型，
        # 并使用self.run_test方法测试该脚本模型，传入flag、x和y
        self.run_test(torch.jit.script(Model()), (flag, x, y))

    # 如果操作集版本小于9，则跳过该测试
    @skipIfUnsupportedMinOpsetVersion(9)
    # 定义一个测试函数，用于测试自定义的 ScriptModule
    def test_cste_script(self):
        # 定义一个继承自 torch.jit.ScriptModule 的自定义模型类 MyModel
        class MyModel(torch.jit.ScriptModule):
            # 使用 @torch.jit.script_method 装饰器定义前向传播方法
            @torch.jit.script_method
            def forward(self, x):
                # 返回一个大小与 x 第一维相同的全零张量，和一个全一张量
                return torch.zeros(x.size(0)), torch.ones(
                    (x.size(1), x.size(0)), dtype=torch.int64
                )

        # 创建一个 3x4 的随机张量 x
        x = torch.randn(3, 4)
        # 运行测试函数 run_test，测试 MyModel 的前向传播，指定输入名为 "x"，动态轴为第 0 和第 1 维
        self.run_test(MyModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1]})
        # 再次运行测试函数 run_test，测试 MyModel 的前向传播，保持 ONNX 输入索引为空列表
        self.run_test(MyModel(), x, remained_onnx_input_idx=[])

    # 定义另一个测试函数，用于测试返回标量张量的自定义模型
    def test_scalar_tensor(self):
        # 定义一个继承自 torch.nn.Module 的自定义模型类 test
        class test(torch.nn.Module):
            # 实现模型的前向传播方法
            def forward(self, input):
                # 返回输入的第一维大小作为标量张量，和输入的第二维大小作为整型标量张量
                return torch.scalar_tensor(input.size(0)), torch.scalar_tensor(
                    input.size(1), dtype=torch.int64
                )

        # 创建两个随机张量 x 和 y
        x = torch.randn(2, 3, 4)
        y = torch.randn(7, 8, 9)
        # 创建 test 类的实例 model
        model = test()
        # 运行测试函数 run_test，测试 model 的前向传播，附加额外的输入 y，输入名为 "input_1"，动态轴为第 0、1、2 维
        self.run_test(
            model,
            x,
            additional_test_inputs=[y],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2]},
        )

    # 定义测试函数，用于测试返回张量的自定义 ScriptModule
    def test_tensor(self):
        # 定义一个继承自 torch.jit.ScriptModule 的自定义模型类 ScalarInputModel
        class ScalarInputModel(torch.jit.ScriptModule):
            # 使用 @torch.jit.script_method 装饰器定义前向传播方法
            @torch.jit.script_method
            def forward(self, input):
                # 返回输入张量的第二维大小作为张量
                return torch.tensor(input.shape[1])

        # 创建一个 3x4 的随机张量 x
        x = torch.randn(3, 4)
        # 运行测试函数 run_test，测试 ScalarInputModel 的前向传播，指定输入名为 "x"，动态轴为第 0 和第 1 维
        self.run_test(
            ScalarInputModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        # 再次运行测试函数 run_test，测试 ScalarInputModel 的前向传播，保持 ONNX 输入索引为空列表
        self.run_test(ScalarInputModel(), x, remained_onnx_input_idx=[])

        # 定义一个继承自 torch.jit.ScriptModule 的自定义模型类 TensorInputModel
        class TensorInputModel(torch.jit.ScriptModule):
            # 使用 @torch.jit.script_method 装饰器定义前向传播方法
            @torch.jit.script_method
            def forward(self, input):
                # 返回一个包含输入张量的形状的两个元素的张量
                return torch.tensor([input.shape[0], input.shape[1]])

        # 重新创建一个 3x4 的随机张量 x
        x = torch.randn(3, 4)
        # 运行测试函数 run_test，测试 TensorInputModel 的前向传播，指定输入名为 "x"，动态轴为第 0 和第 1 维
        self.run_test(
            TensorInputModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        # 再次运行测试函数 run_test，测试 TensorInputModel 的前向传播，保持 ONNX 输入索引为空列表
        self.run_test(TensorInputModel(), x, remained_onnx_input_idx=[])

        # 定义一个继承自 torch.jit.ScriptModule 的自定义模型类 FloatInputModel
        class FloatInputModel(torch.jit.ScriptModule):
            # 使用 @torch.jit.script_method 装饰器定义前向传播方法
            @torch.jit.script_method
            def forward(self, input):
                # 返回一个包含输入张量的浮点数形式的张量
                return torch.tensor([float(input)])

        # 创建一个包含一个随机数的张量 x
        x = torch.randn(1)
        # 运行测试函数 run_test，测试 FloatInputModel 的前向传播
        self.run_test(FloatInputModel(), x)

        # 定义一个继承自 torch.jit.ScriptModule 的自定义模型类 InputWithDtypeModel
        class InputWithDtypeModel(torch.jit.ScriptModule):
            # 使用 @torch.jit.script_method 装饰器定义前向传播方法
            @torch.jit.script_method
            def forward(self, input):
                # 返回输入张量的第二维大小作为长整型张量
                return torch.tensor(input.shape[1], dtype=torch.long)

        # 重新创建一个 3x4 的随机张量 x
        x = torch.randn(3, 4)
        # 运行测试函数 run_test，测试 InputWithDtypeModel 的前向传播，指定输入名为 "x"，动态轴为第 0 和第 1 维
        self.run_test(
            InputWithDtypeModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        # 再次运行测试函数 run_test，测试 InputWithDtypeModel 的前向传播，保持 ONNX 输入索引为空列表
        self.run_test(InputWithDtypeModel(), x, remained_onnx_input_idx=[])

        # 定义一个继承自 torch.jit.ScriptModule 的自定义模型类 MixedInputModel
        class MixedInputModel(torch.jit.ScriptModule):
            # 使用 @torch.jit.script_method 装饰器定义前向传播方法
            @torch.jit.script_method
            def forward(self, input):
                # 返回一个包含输入张量第一维大小和整型化的输入的张量
                return torch.tensor([input.shape[0], int(input)])

        # 创建一个包含一个随机数的张量 x
        x = torch.randn(1)
        # 运行测试函数 run_test，测试 MixedInputModel 的前向传播
        self.run_test(MixedInputModel(), x)

    # 定义测试函数，用于测试 torch.nn.Hardtanh 激活函数模型
    def test_hardtanh(self):
        # 创建一个 Hardtanh 激活函数模型，限制输出在 -1.5 到 2.5 之间
        model = torch.nn.Hardtanh(-1.5, 2.5)
        # 创建一个从 -5 到 5 的序列张量 x，并将其类型转换为 float32
        x = torch.arange(-5, 5).to(dtype=torch.float32)
        # 运行测试函数 run_test，测试 Hardtanh 模型的前向传播
        self.run_test(model, x)
    # 定义一个测试方法，测试使用默认数值的 hardtanh 函数的脚本模型
    def test_hardtanh_script_with_default_values(self):
        # 定义一个继承自 torch.jit.ScriptModule 的自定义模型类 MyModel
        class MyModel(torch.jit.ScriptModule):
            # 使用 torch.jit.script_method 装饰器定义前向传播方法
            @torch.jit.script_method
            def forward(self, x):
                # 返回 torch.nn.functional.hardtanh(x) 的结果
                return torch.nn.functional.hardtanh(x)

        # 创建一个从 -5 到 4 的浮点数张量 x
        x = torch.arange(-5, 5).to(dtype=torch.float32)
        # 调用 run_test 方法，运行 MyModel 的测试
        self.run_test(MyModel(), x)

    # 定义一个测试方法，测试 Hardswish 激活函数
    def test_hardswish(self):
        # 创建一个 Hardswish 模型实例
        model = torch.nn.Hardswish()

        # 创建一个 3x3 的随机浮点数张量 x
        x = torch.rand(3, 3).to(dtype=torch.float32)
        # 调用 run_test 方法，运行 Hardswish 模型的测试
        self.run_test(model, x)

        # 测试边界情况
        # 创建一个数值为 3 的浮点数张量 x
        x = torch.tensor(3).to(dtype=torch.float32)
        # 再次调用 run_test 方法，运行 Hardswish 模型的测试
        self.run_test(model, x)
        # 创建一个数值为 -3 的浮点数张量 x
        x = torch.tensor(-3).to(dtype=torch.float32)
        # 再次调用 run_test 方法，运行 Hardswish 模型的测试
        self.run_test(model, x)

    # 定义一个测试方法，测试使用脚本模型的 Hardswish 激活函数
    def test_hardswish_script(self):
        # 定义一个继承自 torch.jit.ScriptModule 的自定义模型类 MyModel
        class MyModel(torch.jit.ScriptModule):
            # 使用 torch.jit.script_method 装饰器定义前向传播方法
            @torch.jit.script_method
            def forward(self, x):
                # 返回 torch.nn.functional.hardswish(x) 的结果
                return torch.nn.functional.hardswish(x)

        # 创建一个 3x3 的随机浮点数张量 x
        x = torch.rand(3, 3).to(dtype=torch.float32)
        # 调用 run_test 方法，运行 MyModel 的测试
        self.run_test(MyModel(), x)

    # 定义一个测试方法，测试 Hardsigmoid 激活函数
    def test_hardsigmoid(self):
        # 创建一个 Hardsigmoid 模型实例
        model = torch.nn.Hardsigmoid()

        # 创建一个 3x3 的随机浮点数张量 x
        x = torch.rand(3, 3).to(dtype=torch.float32)
        # 调用 run_test 方法，运行 Hardsigmoid 模型的测试
        self.run_test(model, x)

        # 测试边界情况
        # 创建一个数值为 3 的浮点数张量 x
        x = torch.tensor(3).to(dtype=torch.float32)
        # 再次调用 run_test 方法，运行 Hardsigmoid 模型的测试
        self.run_test(model, x)
        # 创建一个数值为 -3 的浮点数张量 x
        x = torch.tensor(-3).to(dtype=torch.float32)
        # 再次调用 run_test 方法，运行 Hardsigmoid 模型的测试
        self.run_test(model, x)

    # 定义一个测试方法，测试 Tanhshrink 激活函数
    def test_tanhshrink(self):
        # 创建一个 Tanhshrink 模型实例
        model = torch.nn.Tanhshrink()

        # 创建一个 3x3 的随机浮点数张量 x
        x = torch.rand(3, 3).to(dtype=torch.float32)
        # 调用 run_test 方法，运行 Tanhshrink 模型的测试
        self.run_test(model, x)

    # 使用 skipIfUnsupportedMinOpsetVersion(9) 装饰器定义一个测试方法，测试 Hardshrink 激活函数
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_hardshrink(self):
        # 创建一个 Hardshrink 模型实例
        model = torch.nn.Hardshrink()

        # 创建一个 3x3 的随机浮点数张量 x
        x = torch.rand(3, 3).to(dtype=torch.float32)
        # 调用 run_test 方法，运行 Hardshrink 模型的测试
        self.run_test(model, x)

        # 测试边界情况
        # 创建一个数值为 0.5 的浮点数张量 x
        x = torch.tensor(0.5).to(dtype=torch.float32)
        # 再次调用 run_test 方法，运行 Hardshrink 模型的测试
        self.run_test(model, x)
        # 创建一个数值为 -0.5 的浮点数张量 x
        x = torch.tensor(-0.5).to(dtype=torch.float32)
        # 再次调用 run_test 方法，运行 Hardshrink 模型的测试
        self.run_test(model, x)

    # 使用 skipIfUnsupportedMinOpsetVersion(9) 装饰器定义一个测试方法，测试 Hardshrink 激活函数的数据类型
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_hardshrink_dtype(self):
        # 创建一个 3x3 的随机浮点数张量 x，数据类型为 torch.float64
        x = torch.rand(3, 3).to(dtype=torch.float64)
        # 调用 run_test 方法，运行 Hardshrink 模型的测试
        self.run_test(torch.nn.Hardshrink(), x)

    # 使用 skipIfUnsupportedMinOpsetVersion(9) 装饰器定义一个测试方法，测试 Softshrink 激活函数
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_softshrink(self):
        # 创建一个 Softshrink 模型实例
        model = torch.nn.Softshrink()

        # 创建一个 3x3 的随机浮点数张量 x
        x = torch.rand(3, 3).to(dtype=torch.float32)
        # 调用 run_test 方法，运行 Softshrink 模型的测试
        self.run_test(model, x)

        # 测试边界情况
        # 创建一个数值为 0.5 的浮点数张量 x
        x = torch.tensor(0.5).to(dtype=torch.float32)
        # 再次调用 run_test 方法，运行 Softshrink 模型的测试
        self.run_test(model, x)
        # 创建一个数值为 -0.5 的浮点数张量 x
        x = torch.tensor(-0.5).to(dtype=torch.float32)
        # 再次调用 run_test 方法，运行 Softshrink 模型的测试
        self.run_test(model, x)

    # 使用 skipIfUnsupportedMinOpsetVersion(9) 装饰器定义一个测试方法，测试 Softshrink 激活函数的数据类型
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_softshrink_dtype(self):
        # 创建一个 3x3 的随机浮点数张量 x，数据类型为 torch.float64
        x = torch.rand(3, 3).to(dtype=torch.float64)
        # 调用 run_test 方法，运行 Softshrink 模型的测试
        self.run_test(torch.nn.Softshrink(), x)
    # 定义测试方法 test_clamp，用于测试不同的 clamp 函数调用
    def test_clamp(self):
        # 定义 ClampModel 类，继承自 torch.nn.Module
        class ClampModel(torch.nn.Module):
            # 实现 forward 方法，对输入张量 x 进行 clamp 操作
            def forward(self, x):
                return x.clamp(-0.5, 0.5)

        # 生成一个形状为 (3, 4) 的随机张量 x
        x = torch.randn(3, 4)
        # 运行测试，使用 ClampModel 实例作为模型，输入为 x
        self.run_test(ClampModel(), x)

        # 定义 ClampMinModel 类，继承自 torch.nn.Module
        class ClampMinModel(torch.nn.Module):
            # 实现 forward 方法，对输入张量 x 进行 clamp 操作，指定最小值为 -0.5
            def forward(self, x):
                return x.clamp(min=-0.5)

        # 生成一个形状为 (3, 4) 的随机张量 x
        x = torch.randn(3, 4)
        # 运行测试，使用 ClampMinModel 实例作为模型，输入为 x
        self.run_test(ClampMinModel(), x)

        # 定义 ClampMaxModel 类，继承自 torch.nn.Module
        class ClampMaxModel(torch.nn.Module):
            # 实现 forward 方法，对输入张量 x 进行 clamp 操作，指定最大值为 0.5
            def forward(self, x):
                return x.clamp(max=0.5)

        # 生成一个形状为 (3, 4) 的随机张量 x
        x = torch.randn(3, 4)
        # 运行测试，使用 ClampMaxModel 实例作为模型，输入为 x
        self.run_test(ClampMaxModel(), x)

    # 根据 Opset 版本跳过不支持的测试，适用于 Opset >= 8
    @skipIfUnsupportedMinOpsetVersion(8)
    # 定义测试方法 test_clamp_dyn，测试动态 clamp 操作
    def test_clamp_dyn(self):
        # 定义 ClampMaxModel 类，继承自 torch.jit.ScriptModule
        class ClampMaxModel(torch.jit.ScriptModule):
            # 使用 torch.jit.script_method 装饰的 forward 方法
            @torch.jit.script_method
            # 实现 forward 方法，对输入张量 x 进行 clamp 操作，最大值为 x 的第一个维度大小
            def forward(self, x):
                return x.clamp(None, x.size(0))

        # 生成一个张量 x，从 0 到 15，reshape 成 (4, 4)，转换为 float 类型
        x = torch.arange(16).view(4, 4).float()
        # 运行测试，使用 ClampMaxModel 实例作为模型，输入为 x
        self.run_test(ClampMaxModel(), x)

        # 定义 ClampMinModel 类，继承自 torch.jit.ScriptModule
        class ClampMinModel(torch.jit.ScriptModule):
            # 使用 torch.jit.script_method 装饰的 forward 方法
            @torch.jit.script_method
            # 实现 forward 方法，对输入张量 x 进行 clamp 操作，最小值为 x 的第一个维度大小
            def forward(self, x):
                return x.clamp(x.size(0), None)

        # 生成一个张量 x，从 0 到 15，reshape 成 (4, 4)，转换为 float 类型
        x = torch.arange(16).view(4, 4).float()
        # 运行测试，使用 ClampMinModel 实例作为模型，输入为 x
        self.run_test(ClampMinModel(), x)

        # 定义 ClampMinMaxModel 类，继承自 torch.jit.ScriptModule
        class ClampMinMaxModel(torch.jit.ScriptModule):
            # 使用 torch.jit.script_method 装饰的 forward 方法
            @torch.jit.script_method
            # 实现 forward 方法，对输入张量 x 进行 clamp 操作，最小值为 x 的第一个维度大小，最大值为 x 的第二个维度大小
            def forward(self, x):
                return x.clamp(x.size(0), x.size(1))

        # 生成一个张量 x，从 0 到 15，reshape 成 (2, 8)，转换为 float 类型
        x = torch.arange(16).view(2, 8).float()
        # 运行测试，使用 ClampMinMaxModel 实例作为模型，输入为 x
        self.run_test(ClampMinMaxModel(), x)

        # 定义 ClampTensorModel 类，继承自 torch.nn.Module
        class ClampTensorModel(torch.nn.Module):
            # 实现 forward 方法，对输入张量 x 进行 clamp 操作，指定最小值为 min，最大值为 max
            def forward(self, x, min, max):
                return x.clamp(min, max)

        # 生成一个形状为 (3, 4) 的随机张量 x，y，z
        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        z = torch.randn(3, 4)
        # 运行测试，使用 ClampTensorModel 实例作为模型，输入为 x，min，max
        self.run_test(ClampTensorModel(), (x, y, z))

        # 定义 ClampTensorMinModel 类，继承自 torch.nn.Module
        class ClampTensorMinModel(torch.nn.Module):
            # 实现 forward 方法，对输入张量 x 进行 clamp 操作，指定最小值为 min
            def forward(self, x, min):
                return x.clamp(min=min)

        # 运行测试，使用 ClampTensorMinModel 实例作为模型，输入为 x，y
        self.run_test(ClampTensorMinModel(), (x, y))

        # 定义 ClampTensorMaxModel 类，继承自 torch.nn.Module
        class ClampTensorMaxModel(torch.nn.Module):
            # 实现 forward 方法，对输入张量 x 进行 clamp 操作，指定最大值为 max
            def forward(self, x, max):
                return x.clamp(max=max)

        # 运行测试，使用 ClampTensorMaxModel 实例作为模型，输入为 x，z
        self.run_test(ClampTensorMaxModel(), (x, z))

    # 根据 Opset 版本跳过不支持的测试，适用于 Opset >= 9
    @skipIfUnsupportedMinOpsetVersion(9)
    # 定义测试方法 test_full_trace，测试 torch.full 在 trace 模式下的使用
    def test_full_trace(self):
        # 定义 FullModel 类，继承自 torch.nn.Module
        class FullModel(torch.nn.Module):
            # 实现 forward 方法，返回形状为 (3, 4)，元素值为 x 的长整型张量
            def forward(self, x):
                return torch.full((3, 4), x, dtype=torch.long)

        # 生成一个张量 x，值为 12
        x = torch.tensor(12)
        # 运行测试，使用 FullModel 实例作为模型，输入为 x
        self.run_test(FullModel(), x)

    # 根据 Opset 版本跳过不支持的测试，适用于 Opset >= 9
    @skipIfUnsupportedMinOpsetVersion(9)
    # 定义测试方法 test_full_script，测试 torch.full 在 script 模式下的使用
    def test_full_script(self):
        # 定义 FullModelScripting 类，继承自 torch.jit.ScriptModule
        class FullModelScripting(torch.jit.ScriptModule):
            # 使用 torch.jit.script_method 装饰的 forward 方法
            @torch.jit.script_method
            # 实现 forward 方法，返回形状为 (3, 4)，元素值为 x 的长整型张量
            def forward(self, x):
                return torch.full((3, 4), x, dtype=torch.long)

        # 生成一个张量 x，值为 12
        x = torch.tensor(12)
        # 运行测试，使用 FullModelScripting 实例作为模型，输入为 x
        self.run_test(FullModelScripting(), x)

    # 定义测试方法 test_fuse_addmm，测试 torch.mm 与 torch.add 的融合
    def test_fuse_addmm(self):
        # 定义 AddmmModel 类，继承自 torch.nn.Module
        class AddmmModel(torch.nn.Module):
            # 实现 forward 方法，对输入张量 x 进行矩阵
    # 定义一个测试函数，测试 MaxPool1d 模型的行为
    def test_maxpool(self):
        # 创建一个 MaxPool1d 模型，池化窗口大小为2，步长为1
        model = torch.nn.MaxPool1d(2, stride=1)
        # 生成一个形状为 (20, 16, 50) 的随机张量作为输入
        x = torch.randn(20, 16, 50)
        # 运行测试函数，传入模型和输入张量 x
        self.run_test(model, x)

    # 定义一个测试函数，测试包含不同维度卷积操作的行为
    def test_conv(self):
        # 定义一个继承自 Module 的子类 TraceModel
        class TraceModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个 Conv1d 层，输入通道数为 16，输出通道数为 33，卷积核大小为 3，步长为 2
                self.conv1 = torch.nn.Conv1d(16, 33, 3, stride=2)
                # 添加一个 Conv2d 层，输入通道数为 16，输出通道数为 33，卷积核大小为 (3, 5)，步长为 (2, 1)，填充为 (4, 2)，膨胀率为 (3, 1)
                self.conv2 = torch.nn.Conv2d(
                    16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)
                )
                # 添加一个 Conv3d 层，输入通道数为 16，输出通道数为 33，卷积核大小为 (3, 5, 2)，步长为 (2, 1, 1)，填充为 (4, 2, 0)
                self.conv3 = torch.nn.Conv3d(
                    16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0)
                )

            # 定义 forward 方法，接受三个输入，并依次对其进行卷积操作
            def forward(self, input1, input2, input3):
                return self.conv1(input1), self.conv2(input2), self.conv3(input3)

        # 生成三个随机张量作为输入
        x1 = torch.randn(20, 16, 50)
        x2 = torch.randn(20, 16, 50, 50)
        x3 = torch.randn(20, 16, 10, 50, 50)

        # 运行测试函数，传入 TraceModel 实例和三个输入张量，设置容差为 10e-5
        self.run_test(TraceModel(), (x1, x2, x3), atol=10e-5)

    # 定义一个测试函数，测试包含字符串填充方式的卷积操作的行为
    def test_conv_str_padding(self):
        # 定义一个继承自 Module 的子类 TraceModel
        class TraceModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个 Conv1d 层，输入通道数为 16，输出通道数为 33，卷积核大小为 3，填充方式为 "valid"
                self.conv1 = torch.nn.Conv1d(16, 33, 3, padding="valid")
                # 添加一个 Conv2d 层，输入通道数为 16，输出通道数为 33，卷积核大小为 (3, 5)，步长为 1，填充方式为 "valid"，膨胀率为 (3, 1)
                self.conv2 = torch.nn.Conv2d(
                    16, 33, (3, 5), stride=1, padding="valid", dilation=(3, 1)
                )
                # 添加一个 Conv3d 层，输入通道数为 16，输出通道数为 33，卷积核大小为 (3, 5, 2)，步长为 1，填充方式为 "same"
                self.conv3 = torch.nn.Conv3d(
                    16, 33, (3, 5, 2), stride=1, padding="same"
                )

            # 定义 forward 方法，接受三个输入，并依次对其进行卷积操作
            def forward(self, input1, input2, input3):
                return self.conv1(input1), self.conv2(input2), self.conv3(input3)

        # 生成三个随机张量作为输入
        x1 = torch.randn(20, 16, 50)
        x2 = torch.randn(20, 16, 50, 50)
        x3 = torch.randn(20, 16, 10, 50, 50)

        # 运行测试函数，传入 TraceModel 实例和三个输入张量，设置容差为 10e-5
        self.run_test(TraceModel(), (x1, x2, x3), atol=10e-5)

    # 定义一个测试函数，测试包含形状推断的卷积操作的行为
    def test_conv_shape_inference(self):
        # 定义一个继承自 Module 的子类 Model
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个 Conv2d 层，输入通道数为 16，输出通道数为 33，卷积核大小为 (3, 5)，步长为 (2, 1)，填充为 (4, 2)，膨胀率为 (3, 1)
                self.conv2 = torch.nn.Conv2d(
                    16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)
                )

            # 定义 forward 方法，接受一个输入，并在卷积后加上常数 2
            def forward(self, input):
                return self.conv2(input) + 2

        # 生成一个随机张量作为输入
        x = torch.randn(20, 16, 50, 100)

        # 运行测试函数，传入 Model 实例和输入张量 x，设置容差为 10e-5，指定输入名称和动态轴
        self.run_test(
            Model(), x, atol=10e-5, input_names=["x"], dynamic_axes={"x": [0]}
        )
    # 定义一个测试类，用于测试转置卷积操作
    def test_conv_transpose(self):
        # 定义一个模型类，继承自torch.nn.Module
        class TraceModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义三个转置卷积层，分别是1维、2维和3维的
                self.conv1 = torch.nn.ConvTranspose1d(16, 33, 3, stride=2)
                self.conv2 = torch.nn.ConvTranspose2d(
                    16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)
                )
                self.conv3 = torch.nn.ConvTranspose3d(
                    16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0)
                )

            # 定义前向传播函数，接受三个输入，分别经过三个转置卷积层处理后返回
            def forward(self, input1, input2, input3):
                return self.conv1(input1), self.conv2(input2), self.conv3(input3)

        # 创建三个不同形状的输入张量
        x1 = torch.randn(20, 16, 10)
        x2 = torch.randn(20, 16, 10, 10)
        x3 = torch.randn(20, 16, 10, 10, 10)

        # 运行测试，传入TraceModel实例和三个输入张量，并设置绝对误差tolerance
        self.run_test(TraceModel(), (x1, x2, x3), atol=10e-5)

    # 定义一个测试类，用于测试Numpy数组的转置
    def test_numpy_T(self):
        # 定义一个模型类，继承自torch.nn.Module
        class NumpyTranspose(torch.nn.Module):
            # 定义前向传播函数，对输入张量进行转置操作并返回
            def forward(self, x):
                return x.T

        # 运行测试，传入NumpyTranspose实例和一个随机形状的输入张量
        self.run_test(NumpyTranspose(), torch.randn(4, 7))

    # 转置操作依赖于输入形状的已知情况。
    # 当启用onnx形状推断时，以下测试才能正常工作。
    def test_transpose_infer_shape(self):
        # 定义一个脚本模块类，继承自torch.jit.ScriptModule
        class TransposeModule(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 定义一个2维卷积层
                self.conv = torch.nn.Conv2d(3, 1, 3, stride=2)

            # 定义前向传播的脚本方法
            @torch.jit.script_method
            def forward(self, x):
                # 对输入张量进行卷积操作
                x = self.conv(x)
                # 对卷积结果进行转置操作并返回
                return x.transpose(0, 1)

        # 创建两个不同形状的输入张量
        x = torch.randn(32, 3, 64, 64)
        y = torch.randn(16, 3, 8, 64)

        # 运行测试，传入TransposeModule实例和输入张量x，同时设置输入名和动态轴
        self.run_test(
            TransposeModule(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 2]},
            additional_test_inputs=[y],
        )

    # 定义一个测试类，用于测试挤压操作
    def squeeze_model_tests(self, d, x1, x2):
        # 定义一个挤压模型类，继承自torch.nn.Module
        class Squeeze(torch.nn.Module):
            def __init__(self, d):
                super().__init__()
                self.d = d

            # 定义前向传播函数，根据维度d对输入张量进行挤压操作并返回
            def forward(self, x):
                if self.d is not None:
                    return torch.squeeze(x, dim=self.d)
                else:
                    return torch.squeeze(x)

        # 如果x2为None，则将其设置为空列表
        x2 = [] if x2 is None else [x2]
        # 如果x2非空，则运行测试，传入Squeeze实例、输入张量x1、输入名和动态轴信息，并加入额外测试输入x2
        if len(x2) > 0:
            self.run_test(
                Squeeze(d),
                x1,
                input_names=["input"],
                dynamic_axes={"input": {0: "0", 1: "1", 2: "2"}},
                additional_test_inputs=x2,
            )
        # 如果x2为空，则仅运行基本的挤压操作测试
        else:
            self.run_test(Squeeze(d), x1)

    # 定义一个测试类，测试不进行无操作的挤压操作
    def test_squeeze_without_no_op(self):
        # 创建一个形状为(2, 1, 4)的随机张量
        x = torch.randn(2, 1, 4)
        # 运行挤压操作测试，传入维度1、输入张量x和None
        self.squeeze_model_tests(1, x, None)

    # 使用skipIfUnsupportedMinOpsetVersion(11)装饰器，定义一个测试动态挤压操作的方法
    def test_squeeze_dynamic(self):
        # 创建两个不同形状的随机张量
        x_squeeze = torch.randn(2, 1, 4)
        x_noop = torch.randn(2, 2, 3)
        # 运行挤压操作测试，传入维度1、输入张量x_squeeze和x_noop
        self.squeeze_model_tests(1, x_squeeze, x_noop)
    # 测试在不执行空操作的情况下对负数维度进行挤压操作
    def test_squeeze_neg_without_no_op(self):
        # 创建一个形状为 (2, 1, 4) 的张量，其中元素为随机数
        x = torch.randn(2, 1, 4)
        # 调用 squeeze_model_tests 方法，对张量 x 执行维度为 -2 的挤压操作，无空操作输入
        self.squeeze_model_tests(-2, x, None)

    @skipIfUnsupportedMinOpsetVersion(11)
    # 测试对负数维度进行挤压操作
    def test_squeeze_neg(self):
        # 创建形状为 (2, 1, 4) 的张量 x_squeeze 和形状为 (2, 2, 3) 的张量 x_noop，分别包含随机数
        x_squeeze = torch.randn(2, 1, 4)
        x_noop = torch.randn(2, 2, 3)
        # 调用 squeeze_model_tests 方法，对张量 x_squeeze 和 x_noop 分别执行维度为 -2 的挤压操作
        self.squeeze_model_tests(-2, x_squeeze, x_noop)

    # 测试对所有维度进行挤压操作
    def test_squeeze_all_dims(self):
        # 创建形状为 (2, 1, 4) 的张量 x_squeeze 和形状为 (2, 2, 3) 的张量 x_noop，分别包含随机数
        x_squeeze = torch.randn(2, 1, 4)
        x_noop = torch.randn(2, 2, 3)
        # 调用 squeeze_model_tests 方法，对张量 x_squeeze 和 x_noop 执行挤压操作，无指定维度
        self.squeeze_model_tests(None, x_squeeze, x_noop)

    @skipIfUnsupportedMinOpsetVersion(11)
    # 测试在不执行挤压操作的情况下对张量进行空操作
    def test_squeeze_no_op(self):
        # 创建形状为 (2, 1, 4) 的张量 x_noop 和形状为 (2, 2, 1) 的张量 x_squeeze，分别包含随机数
        x_noop = torch.randn(2, 1, 4)
        x_squeeze = torch.randn(2, 2, 1)
        # 调用 squeeze_model_tests 方法，对张量 x_noop 和 x_squeeze 执行维度为 2 的挤压操作
        self.squeeze_model_tests(2, x_noop, x_squeeze)

    @skipIfUnsupportedMinOpsetVersion(11)
    # 测试在运行时对维度进行挤压操作
    def test_squeeze_runtime_dim(self):
        # 定义 Squeeze 类，其 forward 方法接受两个参数 d1 和 d2，创建一个指定形状的零张量并挤压维度
        class Squeeze(torch.nn.Module):
            def forward(self, d1, d2):
                t = torch.zeros(d1[0], d2[0])
                return t.squeeze(0)

        d1 = torch.tensor([1])
        d3 = torch.tensor([3])
        d4 = torch.tensor([4])
        # 使用 run_test 方法分别测试 Squeeze 模块，输入为 (d1, d4) 和 (d3, d4)
        self.run_test(Squeeze(), (d1, d4), additional_test_inputs=[(d3, d4)])
        # 再次测试 Squeeze 模块，输入为 (d3, d4) 和 (d1, d3)
        self.run_test(Squeeze(), (d3, d4), additional_test_inputs=[(d1, d3)])

    # 测试对张量进行挤压操作
    def test_squeeze(self):
        # 定义 Squeeze 类，其 forward 方法接受一个参数 x，并对其进行维度为 -2 的挤压操作
        class Squeeze(torch.nn.Module):
            def forward(self, x):
                return torch.squeeze(x, dim=-2)

        # 创建形状为 (2, 1, 4) 的张量 x，包含随机数
        x = torch.randn(2, 1, 4)
        # 使用 run_test 方法测试 Squeeze 模块
        self.run_test(Squeeze(), x)

    @skipIfUnsupportedMinOpsetVersion(13)
    # 测试在运行时指定维度对张量进行挤压操作
    def test_squeeze_dynamic_dim(self):
        # 定义 Squeeze 类，其 forward 方法接受两个参数 x 和 dim，对 x 进行指定维度的挤压操作
        class Squeeze(torch.nn.Module):
            def forward(self, x, dim: int):
                return torch.squeeze(x, dim)

        # 创建形状为 (2, 1, 4) 的张量 x，包含随机数，和整数 dim = 1
        x = torch.randn(2, 1, 4)
        dim = 1
        # 使用 run_test 方法测试 Squeeze 模块
        self.run_test(Squeeze(), (x, dim))

    # 测试对张量进行展开操作
    def test_unsqueeze(self):
        # 定义 Unsqueeze 类，其 forward 方法接受一个参数 x，并对其进行维度为 -2 的展开操作
        class Unsqueeze(torch.nn.Module):
            def forward(self, x):
                return torch.unsqueeze(x, dim=-2)

        # 创建形状为 (2, 3, 4) 的张量 x，包含随机数
        x = torch.randn(2, 3, 4)
        # 使用 run_test 方法测试 Unsqueeze 模块
        self.run_test(Unsqueeze(), x)

    @skipIfUnsupportedMinOpsetVersion(13)
    # 测试在运行时指定维度对张量进行展开操作
    def test_unsqueeze_dynamic_dim(self):
        # 定义 Unsqueeze 类，其 forward 方法接受两个参数 x 和 dim，对 x 进行指定维度的展开操作
        class Unsqueeze(torch.nn.Module):
            def forward(self, x, dim: int):
                return torch.unsqueeze(x, dim)

        # 创建形状为 (2, 1, 4) 的张量 x，包含随机数，和整数 dim = -1
        x = torch.randn(2, 1, 4)
        dim = -1
        # 使用 run_test 方法测试 Unsqueeze 模块
        self.run_test(Unsqueeze(), (x, dim))

    # 测试默认步长的最大池化操作
    def test_maxpool_default_stride(self):
        # 定义 MaxPoolModel 类，其 forward 方法接受一个参数 x，并对其进行默认步长为 2 的最大池化操作
        class MaxPoolModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.max_pool2d(x, 2)

        # 创建形状为 (10, 20, 16, 50) 的张量 x，包含随机数
        model = MaxPoolModel()
        x = torch.randn(10, 20, 16, 50)
        # 使用 run_test 方法测试 MaxPoolModel 模块
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(8)
    # 当 Opset 版本至少为 8 时，执行以下测试（未完整）
    # 定义一个测试方法，测试 AdaptiveMaxPool1d 模块的功能
    def test_maxpool_adaptive(self):
        # 创建一个 AdaptiveMaxPool1d 模块，输出为长度为 5 的自适应最大池化，不返回索引
        model = torch.nn.AdaptiveMaxPool1d((5), return_indices=False)
        # 创建一个大小为 (20, 16, 50) 的随机张量 x，并要求计算梯度
        x = torch.randn(20, 16, 50, requires_grad=True)
        # 创建一个大小为 (32, 16, 50) 的随机张量 y，并要求计算梯度
        y = torch.randn(32, 16, 50, requires_grad=True)
        # 运行测试函数，对模型进行测试，传入输入名称 "x"，设置动态轴 {"x": [0]}，额外测试输入为 y
        self.run_test(
            model,
            x,
            input_names=["x"],
            dynamic_axes={"x": [0]},
            additional_test_inputs=[y],
        )

    # 定义一个测试方法，测试 MaxPool2d 模块的功能
    def test_maxpool_2d(self):
        # 创建一个 MaxPool2d 模块，池化核大小为 5，填充为 (1, 2)
        model = torch.nn.MaxPool2d(5, padding=(1, 2))
        # 创建一个大小为 (1, 20, 16, 50) 的随机张量 x，并要求计算梯度
        x = torch.randn(1, 20, 16, 50, requires_grad=True)
        # 运行测试函数，对模型进行测试，传入输入 x
        self.run_test(model, x)

    # 定义一个测试方法，测试 MaxPool1d 模块的功能，使用 ceil_mode=True
    def test_maxpool_1d_ceil(self):
        # 创建一个 MaxPool1d 模块，池化核大小为 3，步幅为 2，使用 ceil_mode=True
        model = torch.nn.MaxPool1d(3, 2, ceil_mode=True)
        # 创建一个大小为 (20, 16, 50) 的随机张量 x
        x = torch.randn(20, 16, 50)
        # 运行测试函数，对模型进行测试，传入输入 x
        self.run_test(model, x)

    # 定义一个测试方法，测试 MaxPool2d 模块的功能，使用 ceil_mode=True
    def test_maxpool_2d_ceil(self):
        # 创建一个 MaxPool2d 模块，池化核大小为 3，步幅为 2，使用 ceil_mode=True
        model = torch.nn.MaxPool2d(3, 2, ceil_mode=True)
        # 创建一个大小为 (20, 16, 50, 32) 的随机张量 x
        x = torch.randn(20, 16, 50, 32)
        # 运行测试函数，对模型进行测试，传入输入 x
        self.run_test(model, x)

    # 定义一个测试方法，测试 MaxPool3d 模块的功能，使用 ceil_mode=True
    def test_maxpool_3d_ceil(self):
        # 创建一个 MaxPool3d 模块，池化核大小为 3，步幅为 2，使用 ceil_mode=True
        model = torch.nn.MaxPool3d(3, 2, ceil_mode=True)
        # 创建一个大小为 (20, 16, 50, 44, 31) 的随机张量 x
        x = torch.randn(20, 16, 50, 44, 31)
        # 运行测试函数，对模型进行测试，传入输入 x
        self.run_test(model, x)

    # 使用自定义装饰器，在不支持的最小 Opset 版本下跳过该测试
    @skipIfUnsupportedMinOpsetVersion(10)
    # 定义一个测试方法，测试自定义模块的功能
    def test_maxpool_dynamic(self):
        # 定义一个继承自 torch.nn.Module 的测试类
        class test(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                # 使用 functools.partial 创建带有指定参数的部分函数 norm_layer
                norm_layer = functools.partial(torch.nn.BatchNorm2d, eps=0.0009)
                # 创建一个 MaxPool2d 模块，池化核大小为 (2, 2)，步幅为 2，使用 ceil_mode=True
                self.avgpool = torch.nn.MaxPool2d((2, 2), stride=2, ceil_mode=True)
                # 创建一个 2D 卷积模块，输入通道数为 in_channels，输出通道数为 out_channels
                self.conv = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, bias=False
                )
                # 使用 norm_layer 对输出进行归一化
                self.norm = norm_layer(out_channels)

            def forward(self, x):
                # 模型的前向传播过程，先经过平均池化层，再经过卷积层，最后经过归一化层
                return self.norm(self.conv(self.avgpool(x)))

        # 创建一个 test 类的实例，输入通道数为 8，输出通道数为 16
        model = test(8, 16)
        # 创建一个大小为 (2, 8, 64, 64) 的随机张量 inputs
        inputs = torch.randn(2, 8, 64, 64)
        # 运行测试函数，对模型进行测试，传入输入名称 "input_0"，设置动态轴的映射关系
        self.run_test(
            model,
            inputs,
            input_names=["input_0"],
            dynamic_axes={"input_0": {3: "x", 2: "y"}, "output_0": {3: "x", 2: "y"}},
            output_names=["output_0"],
        )

    # TODO: 在 ONNX 1.15.1+ 版本支持后启用 maxpool-ceil 系列测试
    @skipIfUnsupportedMaxOpsetVersion(9)
    # 定义一个测试方法，测试 MaxPool1d 模块的功能，使用 ceil_mode=True
    def test_maxpool_1d_ceil_corner(self):
        # 创建一个 MaxPool1d 模块，池化核大小为 1，膨胀率为 1，步幅为 2，使用 ceil_mode=True，不返回索引
        model = torch.nn.MaxPool1d(
            kernel_size=1, dilation=1, stride=2, ceil_mode=True, return_indices=False
        )
        # 创建一个大小为 (1, 3, 32) 的随机张量 x
        x = torch.randn(1, 3, 32)
        # 运行测试函数，对模型进行测试，传入输入 x
        self.run_test(model, x)

    # TODO: 在 ONNX 1.15.1+ 版本支持后启用 maxpool-ceil 系列测试
    @skipIfUnsupportedMaxOpsetVersion(9)
    # 定义一个测试方法，测试 MaxPool2d 模块的功能，使用 ceil_mode=True
    def test_maxpool_2d_ceil_corner(self):
        # 创建一个 MaxPool2d 模块，池化核大小为 [1, 1]，膨胀率为 [1, 1]，步幅为 [2, 2]，使用 ceil_mode=True，不返回索引
        model = torch.nn.MaxPool2d(
            kernel_size=[1, 1],
            dilation=[1, 1],
            stride=[2, 2],
            ceil_mode=True,
            return_indices=False,
        )
        # 创建一个大小为 (1, 3, 32, 32) 的随机张量 x
        x = torch.randn(1, 3, 32, 32)
        # 运行测试函数，对模型进行测试，传入输入 x
        self.run_test(model, x)

    # TODO: 在 ONNX 1.15.1+ 版本支持后启用 maxpool-ceil 系列测试
    @skipIfUnsupportedMaxOpsetVersion(9)
    def test_maxpool_3d_ceil_corner(self):
        # 创建一个 3D 最大池化层模型
        model = torch.nn.MaxPool3d(
            kernel_size=[7, 8, 4],        # 池化窗口大小为 [7, 8, 4]
            dilation=[1, 1, 1],           # 空洞卷积的膨胀率为 [1, 1, 1]
            stride=[10, 11, 3],           # 池化操作的步长为 [10, 11, 3]
            padding=[2, 2, 2],            # 输入的每个维度两侧各填充 2 个单位
            ceil_mode=True,               # 使用向上取整模式
            return_indices=False,         # 不返回池化结果的索引
        )
        x = torch.randn(1, 3, 51, 52, 45)  # 创建一个形状为 (1, 3, 51, 52, 45) 的随机张量
        self.run_test(model, x)           # 执行测试

    @skipIfUnsupportedMaxOpsetVersion(9)
    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_1d_ceil_corner_with_indices(self):
        # 创建一个带索引的 1D 最大池化层模型
        model = torch.nn.MaxPool1d(
            kernel_size=1,                # 池化窗口大小为 1
            dilation=1,                   # 空洞卷积的膨胀率为 1
            stride=2,                     # 池化操作的步长为 2
            ceil_mode=True,               # 使用向上取整模式
            return_indices=True           # 返回池化结果的索引
        )
        x = torch.randn(1, 3, 32)          # 创建一个形状为 (1, 3, 32) 的随机张量
        self.run_test(model, x)            # 执行测试

    @skipIfUnsupportedMaxOpsetVersion(9)
    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_2d_ceil_corner_with_indices(self):
        # 创建一个带索引的 2D 最大池化层模型
        model = torch.nn.MaxPool2d(
            kernel_size=[1, 1],           # 池化窗口大小为 [1, 1]
            dilation=[1, 1],              # 空洞卷积的膨胀率为 [1, 1]
            stride=[2, 2],                # 池化操作的步长为 [2, 2]
            ceil_mode=True,               # 使用向上取整模式
            return_indices=True           # 返回池化结果的索引
        )
        x = torch.randn(1, 3, 32, 32)      # 创建一个形状为 (1, 3, 32, 32) 的随机张量
        self.run_test(model, x)            # 执行测试

    @skipIfUnsupportedMaxOpsetVersion(9)
    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_3d_ceil_corner_with_indices(self):
        # 创建一个带索引的 3D 最大池化层模型
        model = torch.nn.MaxPool3d(
            kernel_size=[7, 8, 4],        # 池化窗口大小为 [7, 8, 4]
            dilation=[1, 1, 1],           # 空洞卷积的膨胀率为 [1, 1, 1]
            stride=[10, 11, 3],           # 池化操作的步长为 [10, 11, 3]
            padding=[2, 2, 2],            # 输入的每个维度两侧各填充 2 个单位
            ceil_mode=True,               # 使用向上取整模式
            return_indices=True           # 返回池化结果的索引
        )
        x = torch.randn(1, 3, 51, 52, 45)  # 创建一个形状为 (1, 3, 51, 52, 45) 的随机张量
        self.run_test(model, x)           # 执行测试

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_with_indices(self):
        # 创建一个带索引的 1D 最大池化层模型
        model = torch.nn.MaxPool1d(2, stride=1, return_indices=True)
        x = torch.randn(20, 16, 50)        # 创建一个形状为 (20, 16, 50) 的随机张量
        self.run_test(model, x)            # 执行测试

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_maxpool_dilation(self):
        # 创建一个带膨胀的 1D 最大池化层模型
        model = torch.nn.MaxPool1d(2, stride=1, dilation=2)
        x = torch.randn(20, 16, 50)        # 创建一个形状为 (20, 16, 50) 的随机张量
        self.run_test(model, x)            # 执行测试

    def test_avgpool_default_stride(self):
        # 创建一个自定义类的平均池化模型
        class AvgPoolModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.avg_pool2d(x, 2)

        model = AvgPoolModel()             # 实例化 AvgPoolModel 类
        x = torch.randn(10, 20, 16, 50)    # 创建一个形状为 (10, 20, 16, 50) 的随机张量
        self.run_test(model, x)            # 执行测试

    def test_avgpool(self):
        # 创建一个 1D 平均池化层模型
        model = torch.nn.AvgPool1d(2, stride=1)
        x = torch.randn(20, 16, 50)        # 创建一个形状为 (20, 16, 50) 的随机张量
        self.run_test(model, x)            # 执行测试

    def test_avgpool_1d_ceil(self):
        # 创建一个带向上取整的 1D 平均池化层模型
        model = torch.nn.AvgPool1d(3, 2, ceil_mode=True)
        x = torch.randn(1, 1, 7)           # 创建一个形状为 (1, 1, 7) 的随机张量
        self.run_test(model, x)            # 执行测试

    # TODO: ceil_mode is not included in the test, because of
    # https://github.com/microsoft/onnxruntime/issues/16203
    # The ORT and PyTorch has different calculation for ceil_mode (the last value).
    @common_utils.parametrize(
        "padding",
        (0, 1),
    )
    @common_utils.parametrize(
        "count_include_pad",
        (True, False),
    )
    # 定义一个测试方法，用于测试 AvgPool2d 的功能
    def test_avgpool_2d(self, padding, count_include_pad):
        # 创建 AvgPool2d 模型，设置池化窗口大小为 3x3，步幅为 3，填充方式由参数 padding 决定
        model = torch.nn.AvgPool2d(
            3,
            3,
            padding=padding,
            count_include_pad=count_include_pad,
        )
        # 生成一个随机张量作为输入数据，大小为 [20, 16, 50, 32]
        x = torch.randn(20, 16, 50, 32)
        # 运行测试函数，测试模型对输入数据的处理
        self.run_test(model, x)

    # 由于已知的问题，测试中不包括 ceil_mode 的情况
    # 原因是 https://github.com/microsoft/onnxruntime/issues/16203
    # ORT 和 PyTorch 在 ceil_mode 计算上存在差异（最后一个值）
    # 问题需要在 onnx (https://github.com/onnx/onnx/issues/5711) 版本 21 中修复
    # ORT 的修复计划中。修复完成后，可以将 ceil_mode 添加到测试中。
    @skipIfUnsupportedMinOpsetVersion(21)
    def test_avgpool_3d_ceil(self):
        # 创建 AvgPool3d 模型，池化窗口大小为 3x3x3，步幅为 2，启用 ceil_mode
        model = torch.nn.AvgPool3d(3, 2, ceil_mode=True)
        # 生成两个随机张量作为输入数据，大小分别为 [20, 16, 50, 44, 31] 和 [32, 8, 50, 44, 31]
        x = torch.randn(20, 16, 50, 44, 31)
        y = torch.randn(32, 8, 50, 44, 31)
        # 运行测试函数，测试模型对输入数据的处理
        self.run_test(
            model,
            x,
            input_names=["x"],  # 指定输入的名称
            dynamic_axes={"x": [0, 1]},  # 动态轴定义，此处设置维度 0 和 1 为动态轴
            additional_test_inputs=[y],  # 添加额外的测试输入
        )

    # 根据最小 Opset 版本检查是否支持，如果不支持则跳过测试
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_avgpool_dynamic(self):
        # 定义一个测试类，继承自 torch.nn.Module
        class test(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                # 设置 BatchNorm2d 的部分参数，eps 设为 0.0009
                norm_layer = functools.partial(torch.nn.BatchNorm2d, eps=0.0009)
                # 创建 AvgPool2d 模型，池化窗口大小为 (2, 2)，步幅为 2，启用 ceil_mode，不包含填充
                self.avgpool = torch.nn.AvgPool2d(
                    (2, 2), stride=2, ceil_mode=True, count_include_pad=False
                )
                # 创建 2D 卷积层，输入通道数为 in_channels，输出通道数为 out_channels
                self.conv = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, bias=False
                )
                # 添加 Batch normalization 层
                self.norm = norm_layer(out_channels)

            # 定义模型的前向传播方法
            def forward(self, x):
                return self.norm(self.conv(self.avgpool(x)))

        # 创建测试模型对象，输入通道数为 8，输出通道数为 16
        model = test(8, 16)
        # 生成一个随机张量作为输入数据，大小为 [2, 8, 64, 64]
        inputs = torch.randn(2, 8, 64, 64)
        # 运行测试函数，测试模型对输入数据的处理
        self.run_test(
            model,
            inputs,
            input_names=["input_0"],  # 指定输入的名称
            dynamic_axes={  # 定义动态轴，维度 2 和 3 命名为 "x" 和 "y"
                "input_0": {3: "x", 2: "y"},
                "output_0": {3: "x", 2: "y"}
            },
            output_names=["output_0"],  # 指定输出的名称
        )

    # 根据最小 Opset 版本检查是否支持，如果不支持则跳过测试
    @skipIfUnsupportedMinOpsetVersion(9)
    # 定义一个测试方法，用于测试处理浮点数的情况
    def test_floating_point(self):
        # 定义一个继承自torch.jit.ScriptModule的类FloatingPoint
        class FloatingPoint(torch.jit.ScriptModule):
            # 定义前向传播方法，使用装饰器torch.jit.script_method进行静态图编译
            @torch.jit.script_method
            def forward(self, x):
                # 如果输入张量x是浮点数类型
                if x.is_floating_point():
                    # 返回与x相同形状的全零张量
                    return x.new_zeros(x.shape)
                # 否则返回与x相同形状的全零张量
                return x.new_zeros(x.shape)

        # 创建一个2x3x4的随机张量x
        x = torch.randn(2, 3, 4)
        # 运行测试函数run_test，传入FloatingPoint类的实例、输入张量x，
        # input_names指定输入名称为"x"，dynamic_axes指定动态维度为[0, 1, 2]
        self.run_test(
            FloatingPoint(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        # 再次运行测试函数，传入FloatingPoint类的实例和输入张量x，指定remained_onnx_input_idx为空列表
        self.run_test(FloatingPoint(), x, remained_onnx_input_idx=[])

        # 定义另一个类FloatingPoint，与前一个类同名但实现不同
        class FloatingPoint(torch.jit.ScriptModule):
            # 定义前向传播方法，使用torch.jit.script_method进行静态图编译
            @torch.jit.script_method
            def forward(self, x):
                # 如果输入张量x的第一维大小大于1
                if x.size(0) > 1:
                    # 对x加2，并赋值给变量a
                    a = x + 2
                    # 如果a是浮点数类型
                    if a.is_floating_point():
                        # 返回x加1的结果
                        return x + 1
                    # 否则返回x加1的结果
                    return x + 1
                # 如果x的第一维大小不大于1，则直接返回x
                return x

        # 创建一个2x3x4的随机张量x
        x = torch.randn(2, 3, 4)
        # 运行测试函数，传入FloatingPoint类的实例和输入张量x
        self.run_test(FloatingPoint(), x)

    # 在opset版本小于11的情况下，跳过该测试
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_floating_point_infer_dtype(self):
        # 定义一个继承自torch.jit.ScriptModule的类FloatingPoint
        class FloatingPoint(torch.jit.ScriptModule):
            # 定义前向传播方法，使用torch.jit.script_method进行静态图编译
            @torch.jit.script_method
            def forward(self, x):
                # 如果输入张量x的第一维大小大于1
                if x.size(0) > 1:
                    # 对x加2，并赋值给变量a
                    a = x + 2
                    # 如果a是浮点数类型
                    if a.is_floating_point():
                        # 返回一个与x形状的1到最后维度的全零张量
                        return x.new_zeros(x.shape[1:])
                    # 否则返回一个与x形状相同的全零张量
                    return x.new_zeros(x.shape)
                # 如果x的第一维大小不大于1，则直接返回x
                return x

        # 创建一个2x3x4的随机张量x
        x = torch.randn(2, 3, 4)
        # 运行测试函数，传入FloatingPoint类的实例、输入张量x，
        # input_names指定输入名称为"x"，dynamic_axes指定动态维度为[0, 1, 2]
        self.run_test(
            FloatingPoint(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        # 再次运行测试函数，传入FloatingPoint类的实例和输入张量x，指定remained_onnx_input_idx为空列表
        self.run_test(FloatingPoint(), x, remained_onnx_input_idx=[])

        # 定义另一个类FloatingPoint，与前一个类同名但实现不同
        class FloatingPoint(torch.jit.ScriptModule):
            # 定义前向传播方法，使用torch.jit.script_method进行静态图编译
            @torch.jit.script_method
            def forward(self, x):
                # 如果输入张量x的第一维大小大于1
                if x.size(0) > 1:
                    # 对x加2，并赋值给变量a
                    a = x + 2
                    # 如果a是浮点数类型
                    if a.is_floating_point():
                        # 返回x加1的结果
                        return x + 1
                    # 否则返回x本身
                    return x
                # 如果x的第一维大小不大于1，则直接返回x
                return x

        # 创建一个2x3x4的随机张量x，并将其类型转换为torch.int32
        x = torch.randn(2, 3, 4).to(torch.int32)
        # 运行测试函数，传入FloatingPoint类的实例和输入张量x
        self.run_test(FloatingPoint(), x)

    # 在opset版本小于12的情况下，跳过该测试
    @skipIfUnsupportedMinOpsetVersion(12)
    def test_prim_min(self):
        # 定义一个torch.jit.script装饰的函数list_append，接受一个张量列表boxes作为输入
        @torch.jit.script
        def list_append(boxes: List[Tensor]):
            # 初始化一个空列表temp
            temp = []
            # 对boxes中的每个张量b进行遍历，使用enumerate创建prim::min操作
            for i, b in enumerate(
                boxes
            ):  # enumerate is creating a prim::min op in torch graph
                # 将每个b[:, 1]填充为值为i的张量，并添加到temp列表中
                temp.append(torch.full_like(b[:, 1], i))
            # 返回temp列表中的第一个元素
            return temp[0]

        # 定义一个继承自torch.nn.Module的类Min
        class Min(torch.nn.Module):
            # 定义前向传播方法，接受一个输入x
            def forward(self, x):
                # 创建一个包含三个x副本的列表boxes
                boxes = [x for _ in range(3)]
                # 调用list_append函数处理boxes列表并返回结果
                return list_append(boxes)

        # 创建一个5x5的随机张量x
        x = torch.rand(5, 5)
        # 运行测试函数，传入Min类的实例和输入元组(x,)
        self.run_test(Min(), (x,))

        # 定义一个继承自torch.jit.ScriptModule的类M
        class M(torch.jit.ScriptModule):
            # 定义前向传播方法，使用torch.jit.script_method进行静态图编译
            @torch.jit.script_method
            def forward(self, x):
                # 定义变量i为3
                i = 3
                # 返回x[i]和i中较小的值
                return min(x[i], i)

        # 创建一个包含0到5的整数张量x
        x = torch.arange(6, dtype=torch.int64)
        # 运行测试函数，传入M类的实例和输入元组(x,)
        self.run_test(M(), (x,))
    # 定义一个测试方法，用于测试基本的算术操作
    def test_arithmetic(self):
        # 定义一个继承自torch.nn.Module的算术模块
        class ArithmeticModule(torch.nn.Module):
            def forward(self, x):
                # 加法操作：每个元素加2
                x = x + 2
                # 减法操作：每个元素减4
                x = x - 4
                # 乘法操作：每个元素乘以6
                x = x * 6
                # 除法操作：每个元素除以8
                x = x / 8
                return x

        # 生成一个形状为(2, 3, 4)的随机张量x
        x = torch.randn(2, 3, 4)
        # 运行测试，验证ArithmeticModule的输出是否正确
        self.run_test(ArithmeticModule(), x)

    # 定义一个测试方法，测试包含整型参数的算术操作
    def test_arithmetic_prim_long(self):
        # 定义一个继承自torch.nn.Module的算术模块
        class ArithmeticModule(torch.nn.Module):
            def forward(self, x, y: int):
                # 加法操作：每个元素加y
                x = x + y
                # 减法操作：每个元素减y
                x = x - y
                # 乘法操作：每个元素乘以y的3倍
                x = x * (y * 3)
                # 除法操作：每个元素除以y的4倍
                x = x / (y * 4)
                return x

        # 生成一个形状为(2, 3, 4)的随机张量x
        x = torch.randn(2, 3, 4)
        # 设置y为整数值2
        y = 2
        # 运行测试，验证ArithmeticModule的输出是否正确
        self.run_test(ArithmeticModule(), (x, y))

        # 定义一个新的算术模块，不同于上面的算术模块
        class ArithmeticModule(torch.nn.Module):
            def forward(self, x):
                # 加法操作：每个元素加2
                x = x + 2
                # 减法操作：每个元素减3
                x = x - 3
                # 返回张量x的形状的第一个维度的大小
                return x.shape[0]

        # 生成一个形状为(2, 3, 4)的随机张量x
        x = torch.randn(2, 3, 4)
        # 运行测试，验证ArithmeticModule的输出是否正确，并指定不保留ONNX输入索引
        self.run_test(ArithmeticModule(), x, remained_onnx_input_idx=[])

    @skipDtypeChecking
    # 定义一个测试方法，测试包含浮点型参数的算术操作，跳过数据类型检查
    def test_arithmetic_prim_float(self):
        # 定义一个继承自torch.nn.Module的算术模块
        class ArithmeticModule(torch.nn.Module):
            def forward(self, x, y: float):
                # 加法操作：每个元素加y
                x = x + y
                # 减法操作：每个元素减y
                x = x - y
                # 乘法操作：每个元素乘以y的3倍
                x = x * (y * 3)
                # 除法操作：每个元素除以y的4倍
                x = x / (y * 4)
                return x

        # 生成一个形状为(2, 3, 4)的随机张量x
        x = torch.randn(2, 3, 4)
        # 设置y为浮点数值2.5
        y = 2.5
        # 运行测试，验证ArithmeticModule的输出是否正确
        self.run_test(ArithmeticModule(), (x, y))

        # 定义一个新的算术模块，不同于上面的算术模块
        class ArithmeticModule(torch.nn.Module):
            def forward(self, x):
                # 加法操作：每个元素加2
                x = x + 2
                # 减法操作：每个元素减3
                x = x - 3
                # 返回张量x形状的第二个维度的大小，再除以2
                return x.shape[1] / 2

        # 生成一个形状为(2, 3, 4)的随机张量x
        x = torch.randn(2, 3, 4)
        # 运行测试，验证ArithmeticModule的输出是否正确，并指定不保留ONNX输入索引
        self.run_test(ArithmeticModule(), x, remained_onnx_input_idx=[])

    @skipDtypeChecking
    # 定义一个测试方法，测试包含布尔型参数的算术操作，跳过数据类型检查
    def test_arithmetic_prim_bool(self):
        # 定义一个继承自torch.nn.Module的算术模块
        class ArithmeticModule(torch.nn.Module):
            def forward(self, x, y: int, z: bool, t: float):
                # 加法操作：每个元素加y
                x = x + y
                # 减法操作：每个元素减y
                x = x - y
                # 如果z为True，则进行乘法和除法操作
                if z:
                    # 乘法操作：每个元素乘以y的3倍
                    x = x * (y * 3)
                    # 除法操作：每个元素除以y的4倍
                    x = x / (y * 4)
                # 返回x除以t的结果和z的值
                return x / t, z

        # 生成一个形状为(2, 3, 4)的随机张量x
        x = torch.randn(2, 3, 4)
        # 设置y为整数值2，z为False，t为浮点数值2.5
        y = 2
        z = False
        t = 2.5
        # 运行测试，验证ArithmeticModule的输出是否正确
        self.run_test(ArithmeticModule(), (x, y, z, t))

        # 定义一个新的算术模块，不同于上面的算术模块
        class ArithmeticModule(torch.nn.Module):
            def forward(self, x: int, y: int):
                # 返回x是否等于y的布尔值结果
                return x == y

        # 设置x为整数值3，y为整数值2
        x = 3
        y = 2
        # 运行测试，验证ArithmeticModule的输出是否正确
        self.run_test(ArithmeticModule(), (x, y))

    @skipScriptTest(
        15,
        reason="In trace: Outputs that are always None are removed. \
                In script: Outputs that are always None are removed before opset 15. \
                After opset 15, we replace the None in output with Optional node.",
    )
    # 定义一个测试方法，测试包含None输出的元组情况，跳过脚本测试
    def test_tuple_with_none_outputs(self):
        # 定义一个继承自torch.nn.Module的元组模型
        class TupleModel(torch.nn.Module):
            def forward(self, x):
                # 返回一个元组，包含x本身、(x, None)、(x, None)
                return (x, (x, None, (x, None)))

        # 生成一个形状为(3, 4)的随机张量x
        x = torch.randn(3, 4)
        # 运行测试，验证TupleModel的输出是否正确
        self.run_test(TupleModel(), (x,))
        
    # 在脚本化中，第一个转置节点不携带形状和数据类型信息。
    # 当 ONNX 形状推断被启用时才会执行的测试
    def test_arithmetic_infer_dtype(self):
        # 定义一个继承自 torch.jit.ScriptModule 的算术运算模块
        class ArithmeticModule(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                # 转置输入张量 x
                x = x.t()
                # 加 2
                x = x + 2
                # 减 4
                x = x - 4
                # 乘 6
                x = x * 6
                # 除以 8
                x = x / 8
                return x

        # 创建一个形状为 (2, 3) 的随机张量 x
        x = torch.randn(2, 3)
        # 运行测试，使用定义的 ArithmeticModule 进行计算
        self.run_test(ArithmeticModule(), x)

    @unittest.skip("Floor division on ONNX is inconsistent with eager (see #78411)")
    # 测试函数，跳过测试，并注明原因
    def test_floor_div(self):
        # 定义一个继承自 torch.nn.Module 的整数除法模块
        class FloorDivModule(torch.nn.Module):
            # 实现前向传播方法
            def forward(self, x, y):
                return (
                    x // 3,  # x 除以 3，结果向下取整
                    x // 2.0,  # x 除以 2.0，结果向下取整
                    x.to(dtype=torch.float64) // 3,  # 将 x 转换为 float64 类型后除以 3，结果向下取整
                    x.to(dtype=torch.float64) // 2.0,  # 将 x 转换为 float64 类型后除以 2.0，结果向下取整
                    x.to(dtype=torch.int64) // 3,  # 将 x 转换为 int64 类型后除以 3，结果向下取整
                    x.to(dtype=torch.int64) // 2.0,  # 将 x 转换为 int64 类型后除以 2.0，结果向下取整
                    x // (y + 1.0).to(dtype=torch.int64),  # x 除以 (y + 1.0) 转换为 int64 类型，结果向下取整
                    x // y,  # x 除以 y，结果向下取整
                    x.to(dtype=torch.float64) // y.to(dtype=torch.int64),  # 将 x 和 y 分别转换为 float64 和 int64 类型后做除法，结果向下取整
                    x.to(dtype=torch.float64) // y.to(dtype=torch.float64),  # 将 x 和 y 转换为 float64 类型后做除法，结果向下取整
                    x.to(dtype=torch.int64) // y.to(dtype=torch.int64),  # 将 x 和 y 转换为 int64 类型后做除法，结果向下取整
                    x.to(dtype=torch.int64) // y,  # 将 x 转换为 int64 类型后除以 y，结果向下取整
                )

        # 创建输入张量 x，形状为 (2, 3, 1)，值为 -2 到 3 的序列
        x = torch.arange(-2, 4).reshape(2, 3, 1)
        # 创建输入张量 y，形状为 (2, 3, 4)，值为 1 到 2*3*4 的序列
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4)
        # 运行测试，使用定义的 FloorDivModule 进行计算
        self.run_test(FloorDivModule(), (x, y))

    @unittest.skip("Floor division on ONNX is inconsistent with eager (see #78411)")
    # 测试函数，跳过测试，并注明原因
    def test_floor_div_script(self):
        # 定义一个继承自 torch.jit.ScriptModule 的整数除法模块
        class FloorDivModule(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x, y):
                return x // 3, x // 2.0, x // y

        # 创建输入张量 x，形状为 (2, 3, 1)，值为 -2 到 3 的序列
        x = torch.arange(-2, 4).reshape(2, 3, 1)
        # 创建输入张量 y，形状为 (2, 3, 4)，随机值
        y = torch.randn(2, 3, 4)
        # 运行测试，使用定义的 FloorDivModule 进行计算
        self.run_test(FloorDivModule(), (x, y))

    @unittest.skip("Floor division on ONNX is inconsistent with eager (see #78411)")
    @skipIfUnsupportedMinOpsetVersion(9)
    # 测试函数，跳过测试，并注明原因
    def test_floordiv(self):
        # 定义一个继承自 torch.nn.Module 的整数除法模块
        class FloordivModule(torch.nn.Module):
            # 实现前向传播方法
            def forward(self, x):
                return x.new_zeros(x.size(2) // x.size(1))

        # 创建输入张量 x，形状为 (2, 3, 4)，随机值
        x = torch.randn(2, 3, 4)
        # 运行测试，使用定义的 FloordivModule 进行计算
        self.run_test(
            FloordivModule(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(FloordivModule(), (x,), remained_onnx_input_idx=[])

    # 测试函数，测试除法运算
    def test_div(self):
        # 定义一个继承自 torch.nn.Module 的除法模块
        class DivModule(torch.nn.Module):
            # 实现前向传播方法
            def forward(self, x, y):
                return x / y, torch.true_divide(x, y)

        # 创建输入张量 x，形状为 (2, 3, 4)，类型为整数
        x = torch.randn(2, 3, 4).to(torch.int)
        # 创建输入张量 y，形状为 (2, 3, 4)，序列值，类型为整数
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)
        # 运行测试，使用定义的 DivModule 进行计算
        self.run_test(DivModule(), (x, y))
        # 运行测试，将输入张量转换为 float 类型后进行计算
        self.run_test(DivModule(), (x.float(), y.float()))

    # 注意：除法运算通常无法通过脚本导出，因为其类型提升逻辑依赖于知道标量类型
    # 的信息
    # 输入张量的数据类型在 ONNX 图中是重要的，因为 ONNX 图依赖于输入的数据类型。
    # 这使得它适合于追踪（trace）模式。
    def test_div_promotion_trace(self):
        # 定义一个继承自 torch.nn.Module 的类 DivModule
        class DivModule(torch.nn.Module):
            # 前向传播函数，接受两个参数 x 和 y
            def forward(self, x, y):
                # 返回 x 除以 y 的结果和 torch.true_divide(x, y) 的结果
                return x / y, torch.true_divide(x, y)

        # 创建一个大小为 (2, 3, 4) 的随机张量 x，并转换为整型
        x = torch.randn(2, 3, 4).to(torch.int)
        # 创建一个张量 y，其值为从 1 到 2*3*4 的整数，reshape 成 (2, 3, 4)，并转换为整型
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)

        # 使用默认数据类型设置为 float，运行追踪模式下的测试
        with common_utils.set_default_dtype(torch.float):
            self.run_test(torch.jit.trace(DivModule(), (x, y)), (x, y))

        # 使用默认数据类型设置为 double，运行追踪模式下的测试
        with common_utils.set_default_dtype(torch.double):
            self.run_test(torch.jit.trace(DivModule(), (x, y)), (x, y))

    # 在脚本模式下，x 和 y 不包含形状和数据类型信息。
    # 下面的测试仅在启用了 ONNX 形状推断时有效。
    def test_div_promotion_script(self):
        # 定义一个继承自 torch.nn.Module 的类 DivModule
        class DivModule(torch.nn.Module):
            # 前向传播函数，接受两个参数 x 和 y
            def forward(self, x, y):
                # 添加转置操作以隐藏形状和类型信息
                x = x.transpose(1, 2)
                y = y.transpose(1, 2)
                # 返回 x 除以 y 的结果和 torch.true_divide(x, y) 的结果
                return x / y, torch.true_divide(x, y)

        # 创建一个大小为 (2, 3, 4) 的随机张量 x，并转换为整型
        x = torch.randn(2, 3, 4).to(torch.int)
        # 创建一个张量 y，其值为从 1 到 2*3*4 的整数，reshape 成 (2, 3, 4)，并转换为整型
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)

        # Case 1: x 和 y 是整型，输出是浮点型。
        #        这可以通过默认情况处理，其中 x 和 y 都转换为浮点型。
        #        即使 x 和 y 的类型未知，也可以正常工作。
        with common_utils.set_default_dtype(torch.float):
            self.run_test(torch.jit.script(DivModule()), (x, y))

        # Case 2: x 和 y 是整型，输出是双精度浮点型。
        #        这可以通过默认情况处理，其中 x 和 y 都转换为双精度浮点型。
        #        即使 x 和 y 的类型未知，也可以正常工作。
        with common_utils.set_default_dtype(torch.double):
            self.run_test(torch.jit.script(DivModule()), (x, y))

        # Case 3: x 是整型，y 是双精度浮点型，输出是双精度浮点型。
        #        只有当 x 和 y 的类型都已知时才能处理。
        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.double)
        self.run_test(torch.jit.script(DivModule()), (x, y))

    @skipDtypeChecking
    def test_div_rounding_mode(self):
        # 定义一个测试方法，用于测试不同的除法舍入模式
        class TrueDivModule(torch.nn.Module):
            def forward(self, x, y):
                # 返回使用真实除法的结果，torch.div 也被用于真实除法
                return (
                    x.div(y, rounding_mode=None),  # 使用 None 舍入模式进行真实除法
                    torch.div(x, y, rounding_mode=None),  # 同上，使用 torch.div
                )

        class TruncDivModule(torch.nn.Module):
            def forward(self, x, y):
                # 返回使用截断舍入模式的结果
                return (
                    x.div(y, rounding_mode="trunc"),  # 使用截断舍入模式进行除法
                    torch.div(x, y, rounding_mode="trunc"),  # 同上，使用 torch.div
                )

        class FloorDivModule(torch.nn.Module):
            def forward(self, x, y):
                # 返回使用向下舍入模式的结果
                return (
                    x.div(y, rounding_mode="floor"),  # 使用向下舍入模式进行除法
                    torch.div(x, y, rounding_mode="floor"),  # 同上，使用 torch.div
                )

        modules = [TrueDivModule(), TruncDivModule(), FloorDivModule()]

        x = (torch.randn(2, 3, 4) * 100).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)

        for module in modules:
            self.run_test(module, (x, y))  # 运行测试函数，传入不同的模块和参数
            self.run_test(torch.jit.trace(module, (x, y)), (x, y))  # 对模块进行追踪编译并测试
            self.run_test(torch.jit.script(module), (x, y))  # 对模块进行脚本化编译并测试

        x = torch.randn(2, 3, 4)
        y = torch.rand(2, 3, 4) * 10.0 + 0.1

        for module in modules:
            self.run_test(module, (x, y))  # 同上，测试浮点数情况
            self.run_test(torch.jit.trace(module, (x, y)), (x, y))
            self.run_test(torch.jit.script(module), (x, y))

    def test_slice_trace(self):
        # 定义一个测试方法，测试切片操作对追踪模型的影响
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x[0:1]  # 返回输入张量的第一个元素切片

        x = torch.randn(3)
        self.run_test(MyModule(), x)  # 运行测试，传入模块和参数

    def test_slice_neg(self):
        # 定义一个测试方法，测试负索引的切片操作
        class NegSlice(torch.nn.Module):
            def forward(self, x):
                return x[-1:]  # 返回输入张量的最后一个元素切片

        x = torch.randn(3, 4, 5)
        self.run_test(NegSlice(), x)  # 运行测试，传入模块和参数

    def test_slice_neg_large(self):
        # 定义一个测试方法，测试大规模负索引的切片操作
        class NegSlice(torch.nn.Module):
            def forward(self, x):
                return x[:, :, -3:-1, :, -1]  # 返回输入张量的指定位置切片

        x = torch.randn(3, 4, 5, 6, 7)
        self.run_test(NegSlice(), x)  # 运行测试，传入模块和参数

    def test_slice_neg_large_negone(self):
        # 定义一个测试方法，测试全部使用负索引的切片操作
        class NegSlice(torch.nn.Module):
            def forward(self, x):
                return x[:, :, :, :, -1]  # 返回输入张量的最后一个元素切片

        x = torch.randn(3, 4, 5, 6, 7)
        self.run_test(NegSlice(), x)  # 运行测试，传入模块和参数

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_slice_with_input_index(self):
        # 定义一个测试方法，测试带输入索引的切片操作
        class InputIndexSlice(torch.nn.Module):
            def forward(self, x, y):
                x[: y.size(0), 0, :] = y  # 使用输入张量的索引对部分张量进行赋值
                return x

        x = torch.zeros((56, 6, 256))
        y = torch.rand((22, 256))
        self.run_test(InputIndexSlice(), (x, y))  # 运行测试，传入模块和参数

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()  # Torchscript 不支持一维索引。
    def test_slice_with_1d_input_index(self):
        # 定义一个继承自 torch.nn.Module 的内部类 InputIndexSlice
        class InputIndexSlice(torch.nn.Module):
            # 实现 forward 方法，接受 x 和 y 两个参数
            def forward(self, x, y):
                # 对输入 x 进行切片操作，将前 y 个元素的第一维，所有元素的第二维，所有元素的第三维设置为 y
                x[:y, 0, :] = y
                # 返回处理后的 x
                return x

        # 创建一个形状为 (56, 6, 256) 的全零张量 x
        x = torch.zeros((56, 6, 256))
        # 创建一个包含单个元素值为 5 的整数张量 y
        y = torch.tensor([5], dtype=torch.int64)
        # 使用 self.run_test 方法运行 InputIndexSlice 类的实例
        self.run_test(InputIndexSlice(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_slice_with_input_step_size(self):
        # 定义一个继承自 torch.nn.Module 的内部类 InputIndexSlice
        class InputIndexSlice(torch.nn.Module):
            # 实现 forward 方法，接受 x、y、z 三个参数
            def forward(self, x, y, z):
                # 对输入 x 进行切片操作，以步长 z 对前 y 个元素的第一维，步长 z 对所有元素的第二维，所有元素的第三维设置为 1
                x[:y:z, 0::z, :] = 1
                # 返回处理后的 x
                return x

        # 创建一个形状为 (56, 6, 256) 的全零张量 x
        x = torch.zeros((56, 6, 256))
        # 创建整数张量 y 和 z，分别包含单个元素值为 5 和 2
        y = torch.tensor(5, dtype=torch.int64)
        z = torch.tensor(2, dtype=torch.int64)
        # 使用 self.run_test 方法运行 InputIndexSlice 类的实例
        self.run_test(InputIndexSlice(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipScriptTest()  # scripting tuple/list append
    def test_slice_dynamic(self):
        # 定义一个继承自 torch.nn.Module 的内部类 DynamicSliceExportMod
        class DynamicSliceExportMod(torch.nn.Module):
            # 实现 forward 方法，接受 x 一个参数
            def forward(self, x):
                # 初始化一个空列表 results 用于存储结果
                results = []
                # 循环迭代 4 次
                for i in range(4):
                    # 将 x 进行切片操作，从第一维去除 i 个元素到结尾，从第二维去除 i 到 x 的第三维，去除 i 到 3
                    results.append(x[: x.size(0) - i, i : x.size(2), i:3])
                # 返回结果列表的元组
                return tuple(results)

        # 创建一个形状为 (5, 5, 5) 的随机张量 x 和一个形状为 (6, 7, 8) 的随机张量 y
        x = torch.rand(5, 5, 5)
        y = torch.randn(6, 7, 8)
        # 使用 self.run_test 方法运行 DynamicSliceExportMod 类的实例
        self.run_test(
            DynamicSliceExportMod(),
            x,
            additional_test_inputs=[y],
            input_names=["input_1"],
            output_names=["output_1"],
            dynamic_axes={"input_1": [0, 1, 2], "output_1": [0, 1, 2]},
        )

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_slice_dynamic_script(self):
        # 定义一个继承自 torch.jit.ScriptModule 的内部类 DynamicSliceModel
        class DynamicSliceModel(torch.jit.ScriptModule):
            # 定义一个脚本方法 forward，接受 x 一个参数
            @torch.jit.script_method
            def forward(self, x):
                # 返回 x 的第一维从第一个元素到末尾的切片
                return x[1 : x.size(1)]

        # 创建一个形状为 (1, 2) 的随机张量 x
        x = torch.rand(1, 2)
        # 使用 self.run_test 方法运行 DynamicSliceModel 类的实例
        self.run_test(DynamicSliceModel(), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_slice_dynamic_shape_script(self):
        # 定义一个继承自 torch.nn.Module 的内部类 DynamicSliceModel
        class DynamicSliceModel(torch.nn.Module):
            # 实现 forward 方法，接受 x 一个参数
            def forward(self, x):
                # 返回与 x 形状相同的全零张量
                return x.new_zeros(x.shape[1 : x.size(2)])

        # 创建一个形状为 (1, 2, 3, 4) 的随机张量 x
        x = torch.rand(1, 2, 3, 4)
        # 使用 self.run_test 方法运行 DynamicSliceModel 类的实例，设置输入名为 "x"，并声明 "x" 的动态轴为 [0, 1, 2, 3]
        self.run_test(
            DynamicSliceModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2, 3]}
        )
        # 使用 self.run_test 方法再次运行 DynamicSliceModel 类的实例，不保留 ONNX 输入索引
        self.run_test(DynamicSliceModel(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipScriptTest()  # scripting tuple/list append
    def test_slice_dynamic_to_end(self):
        # 定义一个继承自 torch.nn.Module 的内部类 DynamicSliceExportMod
        class DynamicSliceExportMod(torch.nn.Module):
            # 实现 forward 方法，接受 x 一个参数
            def forward(self, x):
                # 初始化一个空列表 results 用于存储结果
                results = []
                # 循环迭代 4 次
                for i in range(4):
                    # 将 x 进行切片操作，选取所有第一维的所有元素，从第二维去除 i 个元素到结尾，选取第三维的倒数第 5 个元素
                    results.append(x[:, i:, x.size(2) - 5])
                # 返回结果列表的元组
                return tuple(results)

        # 创建一个形状为 (5, 5, 5) 的随机张量 x
        x = torch.rand(5, 5, 5)
        # 使用 self.run_test 方法运行 DynamicSliceExportMod 类的实例，设置动态轴
        self.run_test(
            DynamicSliceExportMod(),
            x,
            dynamic_axes={"input_1": [0, 1, 2], "output_1": [0, 1, 2]},
        )
    # 定义一个测试用例，测试 Square 类的功能
    def test_square(self):
        # 定义一个简单的 PyTorch 模块 Square，用于计算输入张量的平方
        class Square(torch.nn.Module):
            def forward(self, x):
                return torch.square(x)

        # 创建一个形状为 (2, 3, 4) 的随机张量
        x = torch.randn(2, 3, 4)
        # 运行测试，验证 Square 模块对 x 的输出
        self.run_test(Square(), x)

    # 如果不支持最小的 Opset 版本 9，则跳过该测试
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_dynamic(self):
        # 定义一个 ArangeModel 类，继承自 torch.nn.Module，实现了 forward 方法
        class ArangeModel(torch.nn.Module):
            def forward(self, input):
                # 返回三个张量：
                # 1. 根据 input 的第一个维度创建的张量
                # 2. 从 0 到 11 的张量
                # 3. 从 input.shape[0] 到 input.shape[0] + 4 的张量
                return (
                    torch.arange(input.shape[0]),
                    torch.arange(12),
                    torch.arange(start=input.shape[0], end=input.shape[0] + 5),
                )

        # 创建形状为 (5, 3, 2) 和 (8, 3, 2) 的两个随机张量 x 和 y
        x = torch.randn(5, 3, 2)
        y = torch.randn(8, 3, 2)
        # 运行测试，验证 ArangeModel 对 x 的输出，并设置附加的测试输入 y
        # 指定输入和输出的名称，以及动态轴的定义
        self.run_test(
            ArangeModel(),
            x,
            additional_test_inputs=[y],
            input_names=["input_1"],
            output_names=["output_1", "output_2", "output_3"],
            dynamic_axes={"input_1": [0], "output_1": [0]},
        )
        # 对 ArangeModel 进行 Torch 脚本化，再次运行测试
        self.run_test(
            torch.jit.script(ArangeModel()),
            x,
            additional_test_inputs=[y],
            input_names=["input_1"],
            output_names=["output_1", "output_2", "output_3"],
            dynamic_axes={"input_1": [0], "output_1": [0]},
        )

    # 如果不支持最小的 Opset 版本 9，则跳过该测试
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_dynamic_arange_out(self):
        # 定义一个 ArangeOutModel 类，继承自 torch.nn.Module，实现了 forward 方法
        class ArangeOutModel(torch.nn.Module):
            def forward(self, end):
                # 创建一个形状为 [1] 的 int64 类型张量 out_t，并使用 torch.arange 返回从 0 到 end 的张量
                out_t = torch.tensor([1], dtype=torch.int64)
                return torch.arange(end, out=out_t)

        # 创建一个值为 8 的 int64 类型张量 x
        x = torch.tensor(8)
        # 运行测试，验证 ArangeOutModel 对 x 的输出
        self.run_test(ArangeOutModel(), (x))

    # 如果不支持最小的 Opset 版本 9，则跳过该测试
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_dynamic_arange_start_out(self):
        # 定义一个 ArangeStartOutModel 类，继承自 torch.nn.Module，实现了 forward 方法
        class ArangeStartOutModel(torch.nn.Module):
            def forward(self, start, end):
                # 创建一个形状为 [1] 的 int64 类型张量 out_t，并使用 torch.arange 返回从 start 到 end 的张量
                out_t = torch.tensor([1], dtype=torch.int64)
                return torch.arange(start.size(0), end, out=out_t)

        # 创建形状为 (2, 3, 4) 的随机张量 x 和一个值为 8 的 int64 类型张量 y
        x = torch.randn(2, 3, 4)
        y = torch.tensor(8)
        # 运行测试，验证 ArangeStartOutModel 对 x 和 y 的输出
        # 指定输入名称为 "x" 和 "y"，以及动态轴的定义
        self.run_test(
            ArangeStartOutModel(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        # 再次运行测试，但是保留 ONNX 输入索引为 1 的输入
        self.run_test(ArangeStartOutModel(), (x, y), remained_onnx_input_idx=[1])

    # 如果不支持最小的 Opset 版本 9，则跳过该测试
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_linspace(self):
        # 定义一个 LinspaceModel 类，继承自 torch.nn.Module，实现了 forward 方法
        class LinspaceModel(torch.nn.Module):
            def forward(self, start, end, steps):
                # 返回一个从 start 到 end 的等间隔数字的张量，步长由 steps 决定
                return torch.linspace(start, end, steps)

        # 创建三个数值型张量 x, y, z
        x = torch.tensor(3, dtype=torch.float)
        y = torch.tensor(10, dtype=torch.float)
        z = torch.tensor(5, dtype=torch.int)
        # 运行测试，验证 LinspaceModel 对 x, y, z 的输出
        self.run_test(LinspaceModel(), (x, y, z))

    # 如果不支持最小的 Opset 版本 9，则跳过该测试
    def test_linspace_negative_start(self):
        # 定义一个测试方法，用于测试 torch.linspace 方法对负起始值的处理
        class LinspaceModel(torch.nn.Module):
            def forward(self, start, end, steps):
                # 模型的前向传播方法，返回 torch.linspace 的结果
                return torch.linspace(start, end, steps)

        # 设置测试用例的输入值
        x = torch.tensor(-1, dtype=torch.float)
        y = torch.tensor(1, dtype=torch.float)
        z = torch.tensor(6, dtype=torch.int)
        # 运行测试方法，并传入测试数据
        self.run_test(LinspaceModel(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_with_floats_out(self):
        # 定义测试 arange 方法使用浮点数参数，并要求输出到指定的张量
        class ArangeModelEnd(torch.nn.Module):
            def forward(self, end):
                # 在前向传播中创建一个输出张量
                out_t = torch.tensor([1], dtype=torch.float)
                # 返回 torch.arange 的结果，指定输出到 out_t
                return torch.arange(end, out=out_t)

        # 设置测试用例的输入值
        y = torch.tensor(8.5, dtype=torch.float)
        # 运行测试方法，并传入测试数据
        self.run_test(ArangeModelEnd(), (y))

        # 定义测试 arange 方法使用浮点数步长，并要求输出到指定的张量
        class ArangeModelStep(torch.nn.Module):
            def forward(self, start, end):
                # 在前向传播中创建一个输出张量
                out_t = torch.tensor([1], dtype=torch.float)
                # 返回 torch.arange 的结果，指定输出到 out_t，使用浮点数步长
                return torch.arange(start.size(0), end, 1.5, out=out_t)

        # 设置测试用例的输入值
        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        # 运行测试方法，并传入测试数据，同时指定输入名称和动态轴
        self.run_test(
            ArangeModelStep(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        # 再次运行测试方法，并传入测试数据，指定保留的 ONNX 输入索引
        self.run_test(ArangeModelStep(), (x, y), remained_onnx_input_idx=[1])

    @skipIfUnsupportedMinOpsetVersion(9)
    # 测试 torch.arange 函数处理浮点数的情况

    class ArangeModelEnd(torch.nn.Module):
        def forward(self, end):
            # 返回从 0 开始到 end-1 的张量
            return torch.arange(end)

    # 创建一个浮点数张量 y
    y = torch.tensor(8.5, dtype=torch.float)
    # 运行测试，输入模型 ArangeModelEnd() 和张量 y
    self.run_test(ArangeModelEnd(), (y))

    class ArangeModelStep(torch.nn.Module):
        def forward(self, start, end):
            # 返回从 start.size(0) 开始，以步长 1.5 直到 end-1 的张量
            return torch.arange(start.size(0), end, 1.5)

    # 创建一个形状为 (2, 3, 4) 的随机张量 x 和浮点数张量 y
    x = torch.randn(2, 3, 4)
    y = torch.tensor(8.5, dtype=torch.float)
    # 运行测试，输入模型 ArangeModelStep()、张量 x 和 y，并指定输入名称和动态轴
    self.run_test(
        ArangeModelStep(),
        (x, y),
        input_names=["x", "y"],
        dynamic_axes={"x": [0, 1, 2]},
    )
    # 再次运行测试，同上，但保留输入索引为 1 的张量
    self.run_test(ArangeModelStep(), (x, y), remained_onnx_input_idx=[1])

    class ArangeModelStepNeg(torch.nn.Module):
        def forward(self, start, end):
            # 返回从 end 开始，以步长 -1.5 直到 start.size(0) 的张量
            return torch.arange(end, start.size(0), -1.5)

    # 创建一个形状为 (2, 3, 4) 的随机张量 x 和浮点数张量 y
    x = torch.randn(2, 3, 4)
    y = torch.tensor(8.5, dtype=torch.float)
    # 运行测试，输入模型 ArangeModelStepNeg()、张量 x 和 y，并指定输入名称和动态轴
    self.run_test(
        ArangeModelStepNeg(),
        (x, y),
        input_names=["x", "y"],
        dynamic_axes={"x": [0, 1, 2]},
    )
    # 再次运行测试，同上，但保留输入索引为 1 的张量
    self.run_test(ArangeModelStepNeg(), (x, y), remained_onnx_input_idx=[1])

    class ArangeModelStart(torch.nn.Module):
        def forward(self, start, end):
            # 返回从 start.size(0) 开始到 end-1 的张量
            return torch.arange(start.size(0), end)

    # 创建一个形状为 (2, 3, 4) 的随机张量 x 和浮点数张量 y
    x = torch.randn(2, 3, 4)
    y = torch.tensor(8.5, dtype=torch.float)
    # 运行测试，输入模型 ArangeModelStart()、张量 x 和 y，并指定输入名称和动态轴
    self.run_test(
        ArangeModelStart(),
        (x, y),
        input_names=["x", "y"],
        dynamic_axes={"x": [0, 1, 2]},
    )
    # 再次运行测试，同上，但保留输入索引为 1 的张量
    self.run_test(ArangeModelStart(), (x, y), remained_onnx_input_idx=[1])
    def test_arange_start_out(self):
        # 定义一个继承自torch.nn.Module的模型类ArangeStartOutModel
        class ArangeStartOutModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, start, end):
                # 创建一个包含单个元素的浮点张量out_t
                out_t = torch.tensor([1], dtype=torch.float)
                # 返回一个从start到end的张量，结果保存在out_t中
                return torch.arange(start.size(0), end, out=out_t)

        # 创建一个2x3x4的张量x，元素服从标准正态分布
        x = torch.randn(2, 3, 4)
        # 创建一个值为8.5的浮点张量y
        y = torch.tensor(8.5, dtype=torch.float)
        # 运行测试函数run_test，测试ArangeStartOutModel模型
        self.run_test(
            ArangeStartOutModel(),
            (x, y),
            input_names=["x", "y"],  # 输入参数的名称
            dynamic_axes={"x": [0, 1, 2]},  # 动态轴的定义
        )
        # 再次运行测试函数run_test，测试ArangeStartOutModel模型，指定remained_onnx_input_idx为[1]
        self.run_test(ArangeStartOutModel(), (x, y), remained_onnx_input_idx=[1])

    @skipIfUnsupportedMinOpsetVersion(11)  # 如果opset版本小于11则跳过该测试
    def test_arange_no_type(self):
        # 定义一个继承自torch.nn.Module的模型类ArangeModel
        class ArangeModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, end):
                # 返回两个张量，分别是从0到end的整数张量，以及从0到end的整数张量
                return torch.arange(end), torch.arange(0, end)

        # 创建一个值为6.2的浮点张量x
        x = torch.tensor(6.2, dtype=torch.float)
        # 运行测试函数run_test，测试ArangeModel模型
        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)  # 如果opset版本小于9则跳过该测试
    def test_size(self):
        # 定义一个继承自torch.nn.Module的模型类SizeModel
        class SizeModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, input):
                # 返回三个张量，分别是input张量的第一个维度大小的整数张量，
                # input张量的最后一个维度大小的整数张量，以及元素全为1的张量，形状与input相同
                return (
                    torch.arange(input.size(0)),
                    torch.arange(input.size(-1)),
                    torch.ones(input.shape),
                )

        # 创建一个5x3x2的张量x，元素服从标准正态分布
        x = torch.randn(5, 3, 2)
        # 运行测试函数run_test，测试SizeModel模型
        self.run_test(SizeModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        # 再次运行测试函数run_test，测试SizeModel模型，不保留任何ONNX输入索引
        self.run_test(SizeModel(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)  # 如果opset版本小于9则跳过该测试
    @skipScriptTest()  # 不支持脚本化测试，因为x.stride()不可脚本化
    def test_as_strided(self):
        # 定义一个继承自torch.nn.Module的模型类Model
        class Model(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 计算chunk_size和chunk_stride
                chunk_size = list(x.size())
                chunk_size[1] = chunk_size[1] * 2 - 1
                chunk_stride = list(x.stride())
                chunk_stride[1] = chunk_stride[1] // 2
                # 返回两个张量，分别是以给定参数创建的新张量
                return x.as_strided(
                    (3, 3, 3), (1, 4, 2), storage_offset=2
                ), x.as_strided(chunk_size, chunk_stride)

        # 创建一个5x8x7的张量x，元素服从标准正态分布
        x = torch.randn(5, 8, 7)
        # 运行测试函数run_test，测试Model模型
        self.run_test(Model(), x)

    @skipScriptTest()  # 不支持脚本化测试，因为Ellipses后面跟随张量索引不可脚本化
    def test_tensor_index_advanced_indexing_ellipsis(self):
        # 定义一个继承自torch.nn.Module的模型类MyModel
        class MyModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, input):
                # 返回对输入张量进行高级索引的结果
                return input[..., torch.tensor([2, 1]), torch.tensor([0, 3])]

        # 创建一个形状为3x4x5x6x7的张量m1，元素服从标准正态分布
        m1 = torch.randn(3, 4, 5, 6, 7)
        # 运行测试函数run_test，测试MyModel模型
        self.run_test(MyModel(), (m1,))
    def test_tensor_index_advanced_indexing(self):
        # 定义一个内部模型类，继承自torch.nn.Module
        class MyModel(torch.nn.Module):
            # 定义模型的前向传播函数
            def forward(self, input):
                # 使用高级索引操作从输入张量中选择特定的元素
                return input[
                    :,  # 选择所有第一维的元素
                    torch.tensor([[0, 2], [1, 1]]),  # 选择指定位置的元素
                    :,  # 选择所有第三维的元素
                    torch.tensor([2, 1]),  # 选择指定位置的元素
                    torch.tensor([0, 3]),  # 选择指定位置的元素
                ]

        # 创建一个形状为(3, 4, 5, 6, 7)的随机张量m1
        m1 = torch.randn(3, 4, 5, 6, 7)
        # 运行测试，使用MyModel实例和m1作为输入
        self.run_test(MyModel(), (m1,))

        # 定义一个新的内部模型类，继承自torch.nn.Module
        class MyModel(torch.nn.Module):
            # 定义模型的前向传播函数
            def forward(self, input):
                # 使用高级索引操作从输入张量中选择特定的元素
                return input[
                    :,  # 选择所有第一维的元素
                    torch.tensor([0, 2]),  # 选择指定位置的元素
                    None,  # 插入一个新的维度
                    2:4,  # 选择第四维的特定范围的元素
                    torch.tensor([[1, 3], [4, 0]])  # 选择指定位置的元素
                ]

        # 运行测试，使用MyModel实例和m1作为输入
        self.run_test(MyModel(), (m1,))

        # 定义一个新的内部模型类，继承自torch.nn.Module
        class MyModel(torch.nn.Module):
            # 定义模型的前向传播函数
            def forward(self, input):
                # 使用高级索引操作从输入张量中选择特定的元素
                return input[
                    :,  # 选择所有第一维的元素
                    torch.tensor([0, 2]),  # 选择指定位置的元素
                    torch.tensor([1]),  # 选择指定位置的元素
                    2:4,  # 选择第四维的特定范围的元素
                    torch.tensor([[1], [4]]),  # 选择指定位置的元素
                ]

        # 运行测试，使用MyModel实例和m1作为输入
        self.run_test(MyModel(), (m1,))

    def test_tensor_index_advanced_indexing_consecutive(self):
        # 定义一个内部模型类，继承自torch.nn.Module
        class MyModel(torch.nn.Module):
            # 定义模型的前向传播函数
            def forward(self, input):
                # 使用高级索引操作从输入张量中选择特定的元素
                return input[
                    :,  # 选择所有第一维的元素
                    torch.tensor([0, 2]),  # 选择指定位置的元素
                    torch.tensor([[1, 3], [4, 0]]),  # 选择指定位置的元素
                    None  # 插入一个新的维度
                ]

        # 创建一个形状为(3, 4, 5, 6, 7)的随机张量m1
        m1 = torch.randn(3, 4, 5, 6, 7)
        # 运行测试，使用MyModel实例和m1作为输入
        self.run_test(MyModel(), (m1,))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put(self):
        # 定义一个内部模型类，继承自torch.nn.Module
        class IndexPutModel(torch.nn.Module):
            # 定义模型的前向传播函数，x是输入张量，ind是索引，update是更新的值
            def forward(self, x, ind, update):
                # 在输入张量x的指定索引ind处进行赋值操作，更新为update的值
                x[ind] = update
                return x

        # 创建一个形状为(3, 4)的随机张量x
        x = torch.randn(3, 4)
        # 创建一个索引张量ind，包含一个长整型的索引值
        ind = torch.tensor([1], dtype=torch.long)
        # 创建一个形状为(4,)的张量update，包含所有元素为1的值
        update = torch.ones(4)
        # 运行测试，使用IndexPutModel实例、x、ind和update作为输入
        self.run_test(IndexPutModel(), (x, ind, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_singular(self):
        # 定义一个内部模型类，继承自torch.nn.Module
        class IndexPutBoolModel(torch.nn.Module):
            # 定义模型的前向传播函数，mask是布尔类型的掩码，indices是索引
            def forward(self, mask, indices):
                # 根据索引indices，将mask中对应位置的元素赋值为True
                mask[indices] = True
                return mask

        # 创建一个形状为(100,)、元素类型为布尔型、所有元素为False的张量mask
        mask = torch.zeros(100, dtype=torch.bool)
        # 创建一个形状为(25,)、整型索引类型的张量indices，其值为在mask.shape[0]范围内的随机索引
        indices = (torch.rand(25) * mask.shape[0]).to(torch.int64)
        # 运行测试，使用IndexPutBoolModel实例、mask和indices作为输入
        self.run_test(IndexPutBoolModel(), (mask, indices))

        # 定义一个内部模型类，继承自torch.nn.Module
        class IndexPutFloatModel(torch.nn.Module):
            # 定义模型的前向传播函数，mask是浮点型张量，indices是索引
            def forward(self, mask, indices):
                # 根据索引indices，将mask中对应位置的元素赋值为5.5
                mask[indices] = torch.tensor(5.5)
                return mask

        # 创建一个形状为(100,)、元素类型为浮点型的随机张量mask
        mask = torch.rand(100, dtype=torch.float)
        # 创建一个形状为(50,)、整型索引类型的张量indices，其值为在mask.shape[0]范围内的随机索引
        indices = (torch.rand(50) * mask.shape[0]).to(torch.int64)
        # 运行测试，使用IndexPutFloatModel实例、mask和indices作为输入
        self.run_test(IndexPutFloatModel(), (mask, indices))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_accumulate(self):
        # 定义一个内部模型类，继承自torch.nn.Module
        class IndexPutModel(torch.nn.Module):
            # 定义模型的前向传播函数，x是输入张量，ind是索引，update是更新的值
            def forward(self, x, ind, update):
                # 在输入张量x的指定索引ind处进行累加赋值操作，更新为update的值
                return x.index_put((ind,), update, accumulate=True)

        # 创建一个形状为(3, 4)的随机张量x
        x = torch.randn(3, 4)
        # 创建一个索引张量ind，包含一个长整型的索引值
        ind = torch.tensor([2], dtype=torch.long)
        # 创建一个形状为(4,)的张量update，包含所有元素为1的值
        update = torch.ones(4)
        # 运行测试，使用IndexPutModel实例、x、ind和update作为输入
        self.run_test(IndexPutModel(), (x, ind, update))
    @skipIfUnsupportedMinOpsetVersion(11)
    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()  # 脚本化测试跳过，因为省略号后跟张量索引不可脚本化
    def test_index_put_ellipsis(self):
        class IndexPutModel(torch.nn.Module):
            def forward(self, x, update):
                # 在张量 x 上执行索引操作，使用省略号扩展维度，以及张量索引和切片
                x[..., torch.tensor([2, 1, 3]), 2:4] += update
                return x

        x = torch.randn(3, 4, 5, 6, 7)
        update = torch.randn(3, 1, 1, 3, 2)
        self.run_test(IndexPutModel(), (x, update))

        class IndexPutModel2(torch.nn.Module):
            def forward(self, x, update):
                # 在张量 x 上执行索引操作，指定具体的索引值
                x[2, ..., torch.tensor([2, 1, 3]), 2:4] += update
                return x

        x = torch.randn(3, 4, 5, 6, 7)
        update = torch.randn(4, 1, 3, 2)
        self.run_test(IndexPutModel2(), (x, update))

    @unittest.skip(
        "regression in 1.18: https://github.com/microsoft/onnxruntime/issues/20855"
    )
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_loop(self):
        @torch.jit.script
        def ngram_attention_bias(
            sequence_length: int, ngram: int, device: torch.device, dtype: torch.dtype
        ):
            # 创建一个指定形状的全为负无穷大的张量 bias
            bias = torch.ones(
                (ngram, sequence_length), device=device, dtype=dtype
            ) * float("-inf")
            for stream_idx in range(ngram):
                for i in range(sequence_length):
                    bias = bias * 2  # 张量乘以 2
                    bias[stream_idx, i] = 5  # 设置特定位置的值为 5
                    bias = bias * 5  # 张量乘以 5
                    bias[0, 0] = 5  # 设置特定位置的值为 5

            for stream_idx in range(ngram):
                for i in range(sequence_length):
                    bias[stream_idx, i] = 5  # 设置特定位置的值为 5
                    bias[0, i] = 5  # 设置特定位置的值为 5
            return bias

        class ScriptModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ngram = 2  # 初始化 ngram 为 2
                self.max_target_positions = 512  # 最大目标位置设为 512

            def forward(self, hidden_states):
                # 获取隐藏状态的序列长度和批量大小
                seq_length, batch_size = hidden_states.shape[:2]
                # 调用 ngram_attention_bias 函数生成预测的因果掩码
                predict_causal_mask = ngram_attention_bias(
                    self.max_target_positions,
                    self.ngram,
                    hidden_states.device,
                    hidden_states.dtype,
                )
                # 限制预测的因果掩码的尺寸范围
                predict_causal_mask = predict_causal_mask[:, :seq_length]
                return predict_causal_mask

        x = torch.randn(6, 2)  # 随机初始化输入张量 x
        y = torch.randn(4, 1)  # 随机初始化输入张量 y
        # 运行测试，传入 ScriptModel 实例以及输入参数
        self.run_test(
            ScriptModel(),
            x,
            input_names=["x"],
            dynamic_axes={"x": {0: "seq_length", 1: "batch_size"}},
            additional_test_inputs=[y],
        )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_copy_(self):
        # 定义一个简单的 PyTorch 模型，用于复制数据到指定的索引范围
        class CopyModel(torch.nn.Module):
            def forward(self, x, data):
                # 将数据复制到 x 的指定索引范围
                x[1:3] = data
                return x

        # 创建一个随机张量 x
        x = torch.randn(3, 4)
        # 创建一个随机更新数据的张量
        update = torch.randn(2, 4)
        # 运行测试，验证模型对给定输入的正确性
        self.run_test(CopyModel(), (x, update))

        # mixed slice and select
        # 定义另一个 PyTorch 模型，支持混合的切片和选择操作
        class CopyModel2(torch.nn.Module):
            def forward(self, x, data):
                # 将数据复制到 x 的指定索引范围和列上
                x[1:3, 0] = data
                return x

        # 创建一个随机张量 x
        x = torch.randn(3, 4)
        # 创建一个标量更新数据的张量
        update = torch.tensor([0], dtype=torch.float32)
        # 运行测试，验证模型对给定输入的正确性
        self.run_test(CopyModel2(), (x, update))

        # 创建一个向量更新数据的张量
        update = torch.tensor([2, 3], dtype=torch.float32)
        # 运行测试，验证模型对给定输入的正确性
        self.run_test(CopyModel2(), (x, update))

        # 创建一个随机向量更新数据的张量
        update = torch.randn(2)
        # 运行测试，验证模型对给定输入的正确性
        self.run_test(CopyModel2(), (x, update))

        # 定义另一个 PyTorch 模型，支持在指定行和范围内的切片操作
        class CopyModel3(torch.nn.Module):
            def forward(self, x, data):
                # 将数据复制到 x 的指定行和范围内的切片位置
                x[1, 1:3] = data
                return x

        # 创建一个随机张量 x
        x = torch.randn(3, 4)
        # 创建一个标量更新数据的张量
        update = torch.tensor([0], dtype=torch.float32)
        # 运行测试，验证模型对给定输入的正确性
        self.run_test(CopyModel3(), (x, update))

        # 创建一个向量更新数据的张量
        update = torch.tensor([2, 3], dtype=torch.float32)
        # 运行测试，验证模型对给定输入的正确性
        self.run_test(CopyModel3(), (x, update))

        # 创建一个随机向量更新数据的张量
        update = torch.randn(2)
        # 运行测试，验证模型对给定输入的正确性
        self.run_test(CopyModel3(), (x, update))

        # 定义另一个 PyTorch 模型，支持使用索引直接复制数据
        class CopyModel4(torch.nn.Module):
            def forward(self, x, ind, data):
                # 将数据复制到 x 的指定索引位置
                x[ind] = data
                return x

        # 创建一个随机张量 x
        x = torch.randn(3, 4)
        # 创建一个索引张量 ind 和随机数据张量 data
        ind = torch.tensor(2)
        data = torch.randn(4)
        # 运行测试，验证模型对给定输入的正确性
        self.run_test(CopyModel4(), (x, ind, data))

        # 定义另一个 PyTorch 模型，支持使用掩码复制数据
        class CopyModel5(torch.nn.Module):
            def forward(self, x, mask):
                if mask is not None:
                    # 使用掩码 mask 复制数据到 x
                    x.copy_(mask)
                    return x

        # 创建一个随机张量 x
        x = torch.randn(3, 4)
        # 创建一个随机掩码张量 mask
        mask = torch.randn(3, 1)
        # 运行测试，验证模型对给定输入的正确性
        self.run_test(CopyModel5(), (x, mask))
    def test_copy_ellipsis_script(self):
        # 定义一个名为 CopyModel 的内部类，继承自 torch.nn.Module
        class CopyModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x, update):
                # 在脚本化过程中插入重塑节点，确保 x 在没有 ONNX 形状推断的情况下无形状/类型信息。
                x = x.reshape(4, 3, 5, 6)
                # 使用切片操作更新 x 的部分内容
                x[2, ..., 1:3] = update
                return x

        # 创建一个形状为 (3, 4, 5, 6) 的随机张量 x
        x = torch.randn(3, 4, 5, 6)

        # 创建一个值为 1 的张量 update
        update = torch.ones(1)
        # 运行测试，使用自定义的 CopyModel 类，并传入 x 和 update 作为参数
        self.run_test(CopyModel(), (x, update))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_flip(self):
        # 定义一个名为 MyModule 的内部类，继承自 torch.nn.Module
        class MyModule(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 返回在维度 0 上翻转的张量 x
                return torch.flip(x, dims=[0])

        # 创建一个形状为 (2, 3) 的张量 x，包含从 0 到 5 的浮点数
        x = torch.tensor(np.arange(6.0).reshape(2, 3))
        # 运行测试，使用自定义的 MyModule 类，并传入 x 作为参数
        self.run_test(MyModule(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_randint(self):
        # 定义一个名为 RandInt 的内部类，继承自 torch.nn.Module
        class RandInt(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 创建一个与 x 相同形状的随机整数张量 randint，取值范围为 [1, 10)
                randint = torch.randint(1, 10, x.shape)
                # 将 x 与 randint 相乘并返回
                x = 0 * randint + x
                return x

        # 创建一个形状为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 运行测试，使用自定义的 RandInt 类，并传入 x 作为参数
        self.run_test(RandInt(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_randint_value(self):
        # 定义一个名为 RandInt 的内部类，继承自 torch.nn.Module
        class RandInt(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 这个 randint 调用始终返回值为 3 的张量
                return torch.randint(3, 4, x.shape) + x

        # 创建一个形状为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 运行测试，使用自定义的 RandInt 类，并传入 x 作为参数
        self.run_test(RandInt(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_randint_like(self):
        # 定义一个名为 RandInt 的内部类，继承自 torch.nn.Module
        class RandInt(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 这个 randint 调用始终返回值为 3 的张量
                return torch.randint_like(x, 3, 4) + x

        # 创建一个形状为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 运行测试，使用自定义的 RandInt 类，并传入 x 作为参数
        self.run_test(RandInt(), x)

    def test_randn(self):
        # 定义一个名为 RandN 的内部类，继承自 torch.nn.Module
        class RandN(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 返回 x 与 (形状为 (2, 3, 4) 的随机张量 x 的元素数量大小) 的乘积
                return torch.mul(x, (torch.randn(2, 3, 4) + x).size(0))

        # 创建一个形状为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 运行测试，使用自定义的 RandN 类，并传入 x 作为参数
        self.run_test(RandN(), x)

    def test_rand(self):
        # 定义一个名为 Rand 的内部类，继承自 torch.nn.Module
        class Rand(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 返回 x 与 (形状为 (2, 3, 4) 的随机张量 x 的元素数量大小) 的乘积
                return torch.mul(x, (torch.rand(2, 3, 4) + x).size(0))

        # 创建一个形状为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 运行测试，使用自定义的 Rand 类，并传入 x 作为参数
        self.run_test(Rand(), x)

    def test_randn_dtype(self):
        # 定义一个名为 RandN 的内部类，继承自 torch.nn.Module
        class RandN(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 结果节点的 dtype 应该是 double。
                return (
                    x.to(torch.float32)
                    * torch.randn(2, 3, 4, dtype=torch.double)
                    * torch.tensor(0, dtype=torch.float32)
                )

        # 创建一个形状为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 运行测试，使用自定义的 RandN 类，并传入 x 作为参数
        self.run_test(RandN(), x)
    def test_rand_dtype(self):
        # 定义一个名为 Rand 的 PyTorch 模块
        class Rand(torch.nn.Module):
            # 前向传播函数
            def forward(self, x):
                # 返回 x 乘以一个随机生成的张量，其中元素类型为双精度浮点数
                return (
                    x.to(torch.float32)
                    * torch.rand(2, 3, 4, dtype=torch.double)
                    * torch.tensor(0, dtype=torch.float32)
                )

        # 创建一个形状为 (2, 3, 4) 的张量 x
        x = torch.randn(2, 3, 4)
        # 运行测试，使用定义的 Rand 模块和输入 x
        self.run_test(Rand(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_randn_dynamic_size(self):
        # 定义一个名为 RandN 的 PyTorch 模块
        class RandN(torch.nn.Module):
            # 前向传播函数
            def forward(self, x):
                # 返回 x 乘以一个形状与 x 相同的随机张量，并取张量形状的第一个维度的大小作为结果
                return torch.mul(x, torch.randn(x.size()).size(1))

        # 创建一个形状为 (2, 3, 4) 的张量 x
        x = torch.randn(2, 3, 4)
        # 运行测试，使用定义的 RandN 模块和输入 x
        self.run_test(RandN(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_rand_dynamic_size(self):
        # 定义一个名为 Rand 的 PyTorch 模块
        class Rand(torch.nn.Module):
            # 前向传播函数
            def forward(self, x):
                # 返回 x 乘以一个形状与 x 相同的随机张量，并取张量形状的第一个维度的大小作为结果
                return torch.mul(x, torch.rand(x.size()).size(1))

        # 创建一个形状为 (2, 3, 4) 的张量 x
        x = torch.randn(2, 3, 4)
        # 运行测试，使用定义的 Rand 模块和输入 x
        self.run_test(Rand(), x)

    def test_randn_like(self):
        # 定义一个名为 RandNLike 的 PyTorch 模块
        class RandNLike(torch.nn.Module):
            # 前向传播函数
            def forward(self, x):
                # 返回 x 乘以一个与 x 形状相同的随机张量，并取张量形状的第一个维度的大小作为结果
                return torch.mul(x, torch.randn_like(x).size(0))

        # 创建一个形状为 (2, 3, 4) 的张量 x
        x = torch.randn(2, 3, 4)
        # 运行测试，使用定义的 RandNLike 模块和输入 x
        self.run_test(RandNLike(), x)
        # 对使用 Torch 脚本编译的 RandNLike 模块进行测试，使用相同的输入 x
        self.run_test(torch.jit.script(RandNLike()), x)

    def test_rand_like(self):
        # 定义一个名为 RandLike 的 PyTorch 模块
        class RandLike(torch.nn.Module):
            # 前向传播函数
            def forward(self, x):
                # 返回 x 乘以一个与 x 形状相同的随机张量，并取张量形状的第一个维度的大小作为结果
                return torch.mul(x, torch.rand_like(x).size(0))

        # 创建一个形状为 (2, 3, 4) 的张量 x
        x = torch.randn(2, 3, 4)
        # 运行测试，使用定义的 RandLike 模块和输入 x
        self.run_test(RandLike(), x)
        # 对使用 Torch 脚本编译的 RandLike 模块进行测试，使用相同的输入 x
        self.run_test(torch.jit.script(RandLike()), x)

    def test_randn_like_dtype(self):
        # 定义一个名为 RandNLike 的 PyTorch 模块
        class RandNLike(torch.nn.Module):
            # 前向传播函数
            def forward(self, x):
                # 返回 x 乘以一个与 x 形状相同且元素类型为双精度浮点数的随机张量，并取张量形状的第一个维度的大小作为结果
                return (
                    x.to(torch.float32)
                    * torch.randn_like(x, dtype=torch.double)
                    * torch.tensor(0, dtype=torch.float32)
                )

        # 创建一个形状为 (2, 3, 4) 的张量 x
        x = torch.randn(2, 3, 4)
        # 运行测试，使用定义的 RandNLike 模块和输入 x
        self.run_test(RandNLike(), x)

    def test_rand_like_dtype(self):
        # 定义一个名为 RandLike 的 PyTorch 模块
        class RandLike(torch.nn.Module):
            # 前向传播函数
            def forward(self, x):
                # 返回 x 乘以一个与 x 形状相同且元素类型为双精度浮点数的随机张量，并取张量形状的第一个维度的大小作为结果
                return (
                    x.to(torch.float32)
                    * torch.rand_like(x, dtype=torch.double)
                    * torch.tensor(0, dtype=torch.float32)
                )

        # 创建一个形状为 (2, 3, 4) 的张量 x
        x = torch.randn(2, 3, 4)
        # 运行测试，使用定义的 RandLike 模块和输入 x
        self.run_test(RandLike(), x)

    def test_bernoulli(self):
        # 定义一个名为 Bernoulli 的 PyTorch 模块
        class Bernoulli(torch.nn.Module):
            # 前向传播函数
            def forward(self, x):
                # 返回 x 乘以一个随机生成的伯努利分布张量，并取张量形状的第一个维度的大小作为结果
                return torch.mul(x, torch.bernoulli(x).size(0))

        # 创建一个形状为 (3, 3) 的空张量 x，元素在区间 [0, 1] 均匀分布
        x = torch.empty(3, 3).uniform_(0, 1)
        # 运行测试，使用定义的 Bernoulli 模块和输入 x
        self.run_test(Bernoulli(), x)

        # 创建一个形状为 (2, 3, 3) 且元素类型为双精度浮点数的空张量 x，元素在区间 [0, 1] 均匀分布
        x = torch.empty(2, 3, 3, dtype=torch.double).uniform_(0, 1)
        # 运行测试，使用定义的 Bernoulli 模块和输入 x
        self.run_test(Bernoulli(), x)
    # 定义一个测试函数，用于测试 Bernoulli 分布的生成和张量操作
    def test_bernoulli_p(self):
        # 定义一个继承自 torch.nn.Module 的类 Bernoulli_float，用于生成 Bernoulli 分布的浮点数结果
        class Bernoulli_float(torch.nn.Module):
            # 重写 forward 方法，接受输入张量 x，并返回 x 与 Bernoulli 生成的张量元素相乘后的结果
            def forward(self, x):
                return torch.mul(x, torch.bernoulli(x, 0.2).size(0))

        # 定义一个继承自 torch.nn.Module 的类 Bernoulli_tensor，用于生成 Bernoulli 分布的张量结果
        class Bernoulli_tensor(torch.nn.Module):
            # 重写 forward 方法，接受输入张量 x，并返回 x 与随机生成的 Bernoulli 张量元素相乘后的结果
            def forward(self, x):
                return torch.mul(x, torch.rand_like(x).bernoulli_(x).size(0))

        # 创建一个 3x3 的随机张量 x
        x = torch.rand(3, 3)
        # 分别使用 Bernoulli_float 类和 Bernoulli_tensor 类对 x 运行测试
        self.run_test(Bernoulli_float(), x)
        self.run_test(Bernoulli_tensor(), x)

        # 创建一个 dtype 为 torch.double，形状为 2x3x3 的随机张量 x
        x = torch.rand(2, 3, 3, dtype=torch.double)
        # 分别使用 Bernoulli_float 类和 Bernoulli_tensor 类对 x 运行测试
        self.run_test(Bernoulli_float(), x)
        self.run_test(Bernoulli_tensor(), x)

    # 使用 unittest.skip 装饰器标记此测试用例跳过，因为 ORT 中存在 bug，等待 rel-1.11 修复后再执行
    @unittest.skip("Bug in ORT, skip test until rel-1.11.")
    # 使用 skipIfUnsupportedMinOpsetVersion 装饰器，要求最小 opset 版本为 14，否则跳过测试
    @skipIfUnsupportedMinOpsetVersion(14)
    # 定义一个测试函数，用于测试 reshape 操作允许零维度的情况
    def test_reshape_allowzero(self):
        # 定义一个继承自 torch.nn.Module 的类 ReshapeModel，重写 forward 方法执行张量的 reshape 操作
        class ReshapeModel(torch.nn.Module):
            def forward(self, x):
                # 将输入张量 x 重塑为维度为 (3, 4, 0) 的张量
                x = x.reshape(3, 4, 0)
                return x

        # 创建一个形状为 (0, 3, 4) 的随机张量 x
        x = torch.randn(0, 3, 4)
        # 使用 ReshapeModel 类对 x 运行测试
        self.run_test(ReshapeModel(), x)

    # 定义一个测试函数，用于测试不同维度张量的 reshape 操作
    def test_reshape_different_rank(self):
        # 定义一个继承自 torch.nn.Module 的类 ReshapeModel，重写 forward 方法执行张量的 reshape 操作
        class ReshapeModel(torch.nn.Module):
            def forward(self, x):
                # 将输入张量 x 重塑为维度为 (-1, 2, 4, 4, 5, 5) 的张量
                x = x.reshape(-1, 2, 4, 4, 5, 5)
                return x

        # 创建一个形状为 (1, 32, 5, 5) 的随机张量 x
        x = torch.randn(1, 32, 5, 5)
        # 使用 ReshapeModel 类对 x 运行测试
        self.run_test(ReshapeModel(), x)
    def _interpolate(self, x, mode, use_size, is_upsample, align_corners=False):
        # 定义一个内部的 PyTorch 模型类，用于插值操作
        class MyModel(torch.nn.Module):
            # 模型类的常量定义
            __constants__ = [
                "mode",
                "use_size",
                "is_upsample",
                "size",
                "scale",
                "size_array",
                "scale_array",
                "align_corners",
            ]

            def __init__(self, mode, use_size, is_upsample, align_corners):
                super().__init__()
                # 初始化模型参数
                self.mode = mode  # 插值模式
                self.use_size = use_size  # 是否使用指定大小进行插值
                self.is_upsample = is_upsample  # 是否上采样
                self.align_corners = align_corners  # 是否按角落对齐
                self.scale = 2.0 if self.is_upsample else 0.5  # 根据是否上采样确定缩放比例
                self.size = 24 if self.is_upsample else 2  # 根据是否上采样确定大小
                # 根据输入张量的维度选择合适的缩放和大小数组
                if x.dim() == 3:
                    self.scale_array = [2.3]
                    self.size_array = [16]
                elif x.dim() == 4:
                    self.scale_array = [2.3, 3.1]
                    self.size_array = [16, 32]
                else:
                    self.scale_array = [2.3, 3.1, 4.6]
                    self.size_array = [16, 32, 64]

            def forward(self, x):
                # 根据 use_size 参数决定是否使用指定大小进行插值
                if self.use_size:
                    # 如果 align_corners 为 True，则使用指定大小和角落对齐进行插值
                    if self.align_corners:
                        return torch.nn.functional.interpolate(
                            x, mode=self.mode, size=self.size, align_corners=True
                        ), torch.nn.functional.interpolate(
                            x, mode=self.mode, size=self.size_array, align_corners=True
                        )
                    # 否则，使用指定大小进行插值，但不进行角落对齐
                    return torch.nn.functional.interpolate(
                        x, mode=self.mode, size=self.size
                    ), torch.nn.functional.interpolate(
                        x, mode=self.mode, size=self.size_array
                    )
                # 如果 align_corners 为 True，则使用缩放因子和角落对齐进行插值
                if self.align_corners:
                    return torch.nn.functional.interpolate(
                        x,
                        mode=self.mode,
                        scale_factor=self.scale,
                        recompute_scale_factor=False,
                    ), torch.nn.functional.interpolate(
                        x,
                        mode=self.mode,
                        scale_factor=self.scale_array,
                        recompute_scale_factor=False,
                    )
                # 否则，使用缩放因子进行插值，但不进行角落对齐
                return torch.nn.functional.interpolate(
                    x,
                    mode=self.mode,
                    scale_factor=self.scale,
                    recompute_scale_factor=False,
                ), torch.nn.functional.interpolate(
                    x,
                    mode=self.mode,
                    scale_factor=self.scale_array,
                    recompute_scale_factor=False,
                )

        # 创建 MyModel 类的实例
        model = MyModel(mode, use_size, is_upsample, align_corners)
        # 使用创建的模型实例运行测试
        self.run_test(model, x, atol=1e-6)
    # 定义一个用于测试插值方法的私有方法，根据是否需要上采样确定测试条件
    def _interpolate_tests(self, is_upsample):
        # 对于 opset 版本低于 11，不支持 cubic 插值模式；
        # 对于 opset 版本低于 11，不匹配 linear 插值模式；
        modes = ["nearest", "linear", "bicubic"]
        if self.opset_version < 11:
            modes = ["nearest"]
        x = [
            torch.randn(1, 2, 6, requires_grad=True),        # 创建一个形状为 (1, 2, 6) 的随机张量，需要梯度
            torch.randn(1, 2, 4, 6, requires_grad=True),    # 创建一个形状为 (1, 2, 4, 6) 的随机张量，需要梯度
            torch.randn(1, 2, 4, 4, 6, requires_grad=True),  # 创建一个形状为 (1, 2, 4, 4, 6) 的随机张量，需要梯度
        ]

        for mode in modes:
            for xi in x:
                mode_i = mode
                # 如果 mode 是 "bicubic" 且张量 xi 的维度不是 4，则跳过当前循环
                if mode == "bicubic" and xi.dim() != 4:
                    continue
                elif mode == "linear":
                    if xi.dim() == 3:
                        # TODO: 当 ORT 修复精度损失问题时启用线性模式用于 1D 输入
                        continue
                    elif xi.dim() == 4:
                        mode_i = "bilinear"  # 将 mode_i 设为 "bilinear"
                    elif xi.dim() == 5:
                        # TODO: 当 ORT 实现线性模式用于 3D 输入时启用三线性模式
                        mode_i = "trilinear"
                        continue
                # 调用 _interpolate 方法，传入张量 xi、mode_i、True（表示测试模式），is_upsample
                self._interpolate(xi, mode_i, True, is_upsample)
                # 如果 mode 不是 "nearest"，再次调用 _interpolate 方法，测试 align_corners 是否支持
                if mode != "nearest":
                    self._interpolate(xi, mode_i, True, is_upsample, True)
                # 下列情况需要动态大小/比例，对于 opset_version < 9 不支持
                if self.opset_version >= 9:
                    self._interpolate(xi, mode_i, True, is_upsample)
                    # 如果 mode 不是 "nearest"，再次测试 align_corners 是否支持
                    if mode != "nearest":
                        self._interpolate(xi, mode_i, False, is_upsample, True)
                    self._interpolate(xi, mode_i, False, is_upsample)

    # 当 opset 版本 >= 9 时，跳过测试插值方法上采样的情况
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_interpolate_upsample(self):
        self._interpolate_tests(True)

    # 当 opset 版本 <= 8 时，跳过测试插值方法上采样的情况（使用脚本测试）
    @skipIfUnsupportedMaxOpsetVersion(8)
    @skipScriptTest()  # 对于 opset > 8 支持脚本测试。参见 test_interpolate_upsample
    def test_interpolate_upsample_trace(self):
        self._interpolate_tests(True)

    # 当 opset 版本 >= 9 时，跳过当前测试
    @skipIfUnsupportedMinOpsetVersion(9)
    @skipIfUnsupportedMinOpsetVersion(11)
    # 标记为一个测试方法，用于检查是否支持指定的最小操作集版本（Opset Version 11）
    def test_interpolate_half_pixel(self):
        # 测试是否使用 "half_pixel" 或 "pytorch_half_pixel"
        # 参考 https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize

        # 定义一个模型类 MyModel 继承自 torch.nn.Module
        class MyModel(torch.nn.Module):
            def __init__(self, mode, size):
                super().__init__()
                self.mode = mode  # 初始化模式
                self.size = size  # 初始化大小

            # 前向传播函数，接受输入 x，使用指定的插值模式和尺寸进行插值操作
            def forward(self, x):
                return torch.nn.functional.interpolate(
                    x, mode=self.mode, size=self.size
                )

        # 定义插值的模式列表
        modes = ["linear", "bicubic"]

        # 定义输入张量列表 x，每个张量都需要梯度计算
        x = [
            torch.randn(1, 2, 6, requires_grad=True),
            torch.randn(1, 2, 4, 6, requires_grad=True),
            torch.randn(1, 2, 4, 4, 6, requires_grad=True),
        ]

        # 遍历模式列表和输入张量列表
        for mode in modes:
            for xi in x:
                mode_i = mode

                # 如果模式为 "bicubic" 且输入张量的维度不是 4，则跳过当前循环
                if mode == "bicubic" and xi.dim() != 4:
                    continue
                # 如果模式为 "linear"
                elif mode == "linear":
                    # 如果输入张量的维度是 4，则更新模式为 "bilinear"
                    if xi.dim() == 4:
                        mode_i = "bilinear"
                    # 如果输入张量的维度是 5，则更新模式为 "trilinear"
                    elif xi.dim() == 5:
                        mode_i = "trilinear"

                # 遍历去除前两个维度后的所有维度
                for i in range(xi.dim() - 2):
                    size = list(xi.shape[2:])
                    size[i] = 1
                    # 运行测试，使用 MyModel 类的实例作为模型，传入输入张量 xi
                    self.run_test(MyModel(mode_i, size), xi)
    def test_interpolate_no_shape(self):
        # 定义一个继承自 torch.jit.ScriptModule 的模型类 MyModel
        class MyModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            # 定义模型的前向传播方法
            def forward(self, x, y):
                # 对输入张量 x 执行加法操作，等效于 x + x
                x = torch.add(x, x)
                # 使用双线性插值法对输入 x 进行大小调整，输出大小为 (16, 16)
                out1 = torch.nn.functional.interpolate(
                    x, mode="bilinear", size=(16, 16), align_corners=False
                )
                # 使用最近邻插值法对输入 x 进行大小调整，输出大小由张量 y 的大小确定
                out2 = torch.nn.functional.interpolate(
                    x, mode="nearest", size=(int(y.size(0)), int(y.size(1)))
                )
                # 返回两个插值操作的结果
                return out1, out2

        # 创建随机张量 x 和 y，用于测试模型
        x = torch.randn(1, 2, 4, 4, requires_grad=True)
        y = torch.randn(16, 16, requires_grad=True)
        # 运行测试函数，验证模型输出是否符合预期
        self.run_test(
            MyModel(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2, 3], "y": [0, 1]},
        )
        # 再次运行测试函数，验证模型是否能够生成符合预期的 ONNX 输入
        self.run_test(MyModel(), (x, y), remained_onnx_input_idx=[0])

    @skipScriptTest()  # scripting raises OnnxRuntimeError
    # 测试插值操作在使用自适应池化时是否能够正确引发 RuntimeError 异常
    def test_interpolate_adaptive_pooling_error(self):
        # 创建形状为 (1, 2, 6) 的随机张量 x，用于测试自适应池化时的异常情况
        x = torch.randn(1, 2, 6, requires_grad=True)
        with self.assertRaises(RuntimeError) as cm:
            # 调用 _interpolate 方法，使用 area 模式，进行自适应池化操作
            self._interpolate(x, "area", True, True)

        with self.assertRaises(RuntimeError) as cm:
            # 再次调用 _interpolate 方法，使用 area 模式，不进行自适应池化操作
            self._interpolate(x, "area", False, True)

    # 测试 GroupNorm 模块的正常工作情况
    def test_groupnorm(self):
        # 创建 GroupNorm 模块，组数为 3，每组包含 6 个特征，归一化参数为 0.002
        model = torch.nn.GroupNorm(3, 6, 0.002)
        # 创建形状为 (4, 6, 36, 36, 18) 的随机张量 x，用于测试模型
        x = torch.randn(4, 6, 36, 36, 18)
        # 运行测试函数，验证 GroupNorm 模块是否能够正常工作
        self.run_test(model, x)

        # 创建 GroupNorm 模块，组数为 1，每组包含 6 个特征，归一化参数为 0.002
        model = torch.nn.GroupNorm(1, 6, 0.002)
        # 创建形状为 (4, 6, 180, 180) 的随机张量 x，用于测试模型
        x = torch.randn(4, 6, 180, 180)
        # 运行测试函数，验证 GroupNorm 模块是否能够正常工作
        self.run_test(model, x)

        # 创建 GroupNorm 模块，组数为 6，每组包含 6 个特征，归一化参数为 0.002
        model = torch.nn.GroupNorm(6, 6, 0.002)
        # 创建形状为 (4, 6, 180, 180) 的随机张量 x，用于测试模型
        x = torch.randn(4, 6, 180, 180)
        # 运行测试函数，验证 GroupNorm 模块是否能够正常工作
        self.run_test(model, x)

    # 测试不带仿射参数的 GroupNorm 模块
    def test_groupnorm_noaffine(self):
        # 创建不带仿射参数的 GroupNorm 模块，组数为 4，每组包含 8 个特征，归一化参数为 0.002
        model = torch.nn.GroupNorm(4, 8, 0.002, affine=False)
        # 创建形状为 (3, 8, 224, 224) 的随机张量 x，用于测试模型
        x = torch.randn(3, 8, 224, 224)
        # 运行测试函数，验证 GroupNorm 模块是否能够正常工作
        self.run_test(model, x)

        # 创建不带仿射参数的 GroupNorm 模块，组数为 1，每组包含 6 个特征，归一化参数为 0.002
        model = torch.nn.GroupNorm(1, 6, 0.002, affine=False)
        # 创建形状为 (4, 6, 180, 180) 的随机张量 x，用于测试模型
        x = torch.randn(4, 6, 180, 180)
        # 运行测试函数，验证 GroupNorm 模块是否能够正常工作
        self.run_test(model, x)

        # 创建不带仿射参数的 GroupNorm 模块，组数为 6，每组包含 6 个特征，归一化参数为 0.002
        model = torch.nn.GroupNorm(6, 6, 0.002, affine=False)
        # 创建形状为 (4, 6, 180, 180) 的随机张量 x，用于测试模型
        x = torch.randn(4, 6, 180, 180)
        # 运行测试函数，验证 GroupNorm 模块是否能够正常工作
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    # 测试脚本化运行时是否能够正确解包列表输入
    def test_list_unpack_scripted(self):
        # 定义 ListUnpack 类，继承自 torch.nn.Module
        class ListUnpack(torch.nn.Module):
            # 定义前向传播方法，用于解包输入 x 的形状并生成全零张量
            def forward(self, x):
                a, b = x.shape
                return x.new_zeros((a, b))

        # 创建形状为 (2, 3) 的随机张量 x，用于测试 ListUnpack 模型
        x = torch.randn(2, 3)
        # 运行测试函数，验证 ListUnpack 模型是否能够正常工作
        self.run_test(
            torch.jit.script(ListUnpack()),
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
        )
        # 再次运行测试函数，验证 ListUnpack 模型是否能够生成符合预期的 ONNX 输入
        self.run_test(torch.jit.script(ListUnpack()), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    # 测试脚本化运行时是否能够处理构造的列表作为输入而不引发错误
    def test_list_unpack_scripted_runs_without_error_with_constructed_list_as_input(
        self,
    ):
        class PackUnpack(torch.nn.Module):
            """Create and unpack a list of tensors.

            When scripted, it should produce a graph similar to

            ```
            graph(%self : __torch__.PackUnpack,
                %a.1 : Tensor,
                %b.1 : Tensor):
            %packed.1 : Tensor[] = prim::ListConstruct(%a.1, %b.1)
            %c.1 : Tensor, %8 : Tensor = prim::ListUnpack(%packed.1)
            return (%c.1)
            ```
            """

            def forward(self, a, b):
                packed = [a, b]  # 创建包含张量 a 和 b 的列表 packed
                c, _ = packed  # 解包列表 packed 到变量 c 和 _
                return c  # 返回解包后的变量 c

        self.run_test(
            torch.jit.script(PackUnpack()),  # 对 PackUnpack 模块进行脚本化
            (torch.tensor(0), torch.tensor([42])),  # 提供输入参数 (0, [42])
            remained_onnx_input_idx=[0],  # 指定 ONNX 输入的索引为 [0]
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_list_unpack_slice_scripted(self):
        class ListUnpackSlice(torch.nn.Module):
            def forward(self, x):
                a, b = x.shape[2:]  # 解包输入张量 x 的形状的第三个和第四个维度到变量 a 和 b
                return x.new_zeros((a, b))  # 返回与输入张量 x 形状相关的全零张量

        x = torch.randn(2, 3, 4, 5)  # 创建一个随机张量 x
        self.run_test(
            torch.jit.script(ListUnpackSlice()),  # 对 ListUnpackSlice 模块进行脚本化
            x,  # 提供输入参数 x
            input_names=["x"],  # 指定输入参数的名称为 "x"
            dynamic_axes={"x": [0, 1, 2, 3]},  # 指定 "x" 的动态轴
        )
        self.run_test(
            torch.jit.script(ListUnpackSlice()),  # 再次对 ListUnpackSlice 模块进行脚本化
            x,  # 提供输入参数 x
            remained_onnx_input_idx=[],  # 没有保留的 ONNX 输入索引
        )

    @skipDtypeChecking
    def test_pow(self):
        class PowModule(torch.nn.Module):
            def forward(self, x, y):
                return x.pow(y)  # 返回 x 的 y 次幂的张量

        x = torch.randn(2, 3, 4)  # 创建一个随机张量 x
        y = torch.randn(2, 3, 4)  # 创建一个随机张量 y
        self.run_test(PowModule(), (x, y))  # 对 PowModule 模块进行测试

        x = torch.randint(10, (2, 3, 4))  # 创建一个随机整数张量 x
        y = torch.randint(10, (2, 3, 4)).to(dtype=torch.int32)  # 创建一个随机整数张量 y，并转换为 torch.int32 类型
        self.run_test(PowModule(), (x, y))  # 再次对 PowModule 模块进行测试

        x = torch.randint(10, (2, 3, 4))  # 创建一个随机整数张量 x
        y = torch.randint(10, (2, 3, 4))  # 创建一个随机整数张量 y
        self.run_test(PowModule(), (x, y))  # 第三次对 PowModule 模块进行测试

        x = torch.randn(2, 3, 4).to(dtype=torch.float64)  # 创建一个随机张量 x，并转换为 torch.float64 类型
        y = torch.randint(10, (2, 3, 4))  # 创建一个随机整数张量 y
        self.run_test(PowModule(), (x, y))  # 第四次对 PowModule 模块进行测试

        class PowModule2(torch.nn.Module):
            def forward(self, x):
                return torch.pow(2, x)  # 返回 2 的 x 次幂的张量

        x = torch.randn(1, 10)  # 创建一个随机张量 x
        self.run_test(PowModule2(), (x,))  # 对 PowModule2 模块进行测试

        x = torch.randint(10, (2, 3, 4))  # 创建一个随机整数张量 x
        self.run_test(PowModule2(), (x,))  # 再次对 PowModule2 模块进行测试

        x = torch.randn(1, 10).to(dtype=torch.float64)  # 创建一个随机张量 x，并转换为 torch.float64 类型
        self.run_test(PowModule2(), (x,))  # 第三次对 PowModule2 模块进行测试

        class PowModule3(torch.nn.Module):
            def forward(self, x, y):
                return y[torch.pow(2, x)]  # 返回 y 中 2 的 x 次幂对应的元素

        x = torch.randint(5, (2, 3, 4))  # 创建一个随机整数张量 x
        y = torch.rand(100)  # 创建一个长度为 100 的随机浮点数张量 y
        self.run_test(PowModule3(), (x, y))  # 对 PowModule3 模块进行测试

    # the arithmeticOps(Add\Sub\Mul\Div\Gemm\Pow\Mod) with low precision include unit8 will be failed in ORT
    # add to(dtype=torch.long) to avoid ORT output type does not match expected type.
    # will be fixed in ONNX version 14.
    @skipIfUnsupportedMaxOpsetVersion(13)
    @skipDtypeChecking
    # 定义一个测试方法，用于测试低精度下的算术运算
    def test_arithmeticOps_with_low_precision(self):
        # 定义一个加法模块，继承自torch.nn.Module
        class AddModule(torch.nn.Module):
            # 定义模块的前向传播方法，实现加法运算
            def forward(self, x, y):
                return x + y

        # 定义一个减法模块，继承自torch.nn.Module
        class SubModule(torch.nn.Module):
            # 定义模块的前向传播方法，实现减法运算
            def forward(self, x, y):
                return x - y

        # 定义一个乘法模块，继承自torch.nn.Module
        class MulModule(torch.nn.Module):
            # 定义模块的前向传播方法，实现乘法运算
            def forward(self, x, y):
                return x * y

        # 定义一个除法模块，继承自torch.nn.Module
        class DivModule(torch.nn.Module):
            # 定义模块的前向传播方法，实现除法运算
            def forward(self, x, y):
                return x / y

        # 定义一个幂运算模块，继承自torch.nn.Module
        class PowModule(torch.nn.Module):
            # 定义模块的前向传播方法，实现幂运算
            def forward(self, x, y):
                return x.pow(y)

        # 创建一个uint8类型的张量x，包含元素[2, 3, 5]
        x = torch.tensor([2, 3, 5], dtype=torch.uint8)
        # 创建一个uint8类型的张量y，包含元素[2, 3, 5]
        y = torch.tensor([2, 3, 5], dtype=torch.uint8)
        # 创建一个uint8类型的张量z，包含元素[1]
        z = torch.tensor([1], dtype=torch.uint8)
        # 使用self.run_test方法运行AddModule模块，传入参数(x, y)
        self.run_test(AddModule(), (x, y))
        # 使用self.run_test方法运行SubModule模块，传入参数(x, y)
        self.run_test(SubModule(), (x, y))
        # 使用self.run_test方法运行MulModule模块，传入参数(x, y)
        self.run_test(MulModule(), (x, y))
        # 使用self.run_test方法运行DivModule模块，传入参数(x, y)
        self.run_test(DivModule(), (x, y))
        # 使用self.run_test方法运行PowModule模块，传入参数(x, z)
        self.run_test(PowModule(), (x, z))

        # 创建一个int8类型的张量x，包含元素[2, 3, 5]
        x = torch.tensor([2, 3, 5], dtype=torch.int8)
        # 创建一个int8类型的张量y，包含元素[2, 3, 5]
        y = torch.tensor([2, 3, 5], dtype=torch.int8)
        # 创建一个int8类型的张量z，包含元素[1]
        z = torch.tensor([1], dtype=torch.int8)
        # 使用self.run_test方法运行AddModule模块，传入参数(x, y)
        self.run_test(AddModule(), (x, y))
        # 使用self.run_test方法运行SubModule模块，传入参数(x, y)
        self.run_test(SubModule(), (x, y))
        # 使用self.run_test方法运行MulModule模块，传入参数(x, y)
        self.run_test(MulModule(), (x, y))
        # 使用self.run_test方法运行DivModule模块，传入参数(x, y)
        self.run_test(DivModule(), (x, y))
        # 使用self.run_test方法运行PowModule模块，传入参数(x, z)
        self.run_test(PowModule(), (x, z))

        # 创建一个int16类型的张量x，包含元素[2, 3, 5]
        x = torch.tensor([2, 3, 5], dtype=torch.int16)
        # 创建一个int16类型的张量y，包含元素[2, 3, 5]
        y = torch.tensor([2, 3, 5], dtype=torch.int16)
        # 创建一个int16类型的张量z，包含元素[1]
        z = torch.tensor([1], dtype=torch.int16)
        # 使用self.run_test方法运行AddModule模块，传入参数(x, y)
        self.run_test(AddModule(), (x, y))
        # 使用self.run_test方法运行SubModule模块，传入参数(x, y)
        self.run_test(SubModule(), (x, y))
        # 使用self.run_test方法运行MulModule模块，传入参数(x, y)
        self.run_test(MulModule(), (x, y))
        # 使用self.run_test方法运行DivModule模块，传入参数(x, y)
        self.run_test(DivModule(), (x, y))
        # 使用self.run_test方法运行PowModule模块，传入参数(x, z)
        self.run_test(PowModule(), (x, z))

        # 创建一个uint8类型的张量x，包含元素[2, 3, 5]
        x = torch.tensor([2, 3, 5], dtype=torch.uint8)
        # 创建一个float32类型的张量y，包含元素[2, 3, 5]
        y = torch.tensor([2, 3, 5], dtype=torch.float32)
        # 创建一个float64类型的张量z，包含元素[1]
        z = torch.tensor([1], dtype=torch.float64)
        # 使用self.run_test方法运行AddModule模块，传入参数(x, y)
        self.run_test(AddModule(), (x, y))
        # 使用self.run_test方法运行SubModule模块，传入参数(x, y)
        self.run_test(SubModule(), (x, y))
        # 使用self.run_test方法运行MulModule模块，传入参数(x, y)
        self.run_test(MulModule(), (x, y))
        # 使用self.run_test方法运行DivModule模块，传入参数(x, y)
        self.run_test(DivModule(), (x, y))
        # 使用self.run_test方法运行PowModule模块，传入参数(x, z)
        self.run_test(PowModule(), (x, z))

        # 创建一个uint8类型的张量x，包含元素[2, 3, 5]
        x = torch.tensor([2, 3, 5], dtype=torch.uint8)
        # 创建一个int64类型的张量y，包含元素[2, 3, 5]
        y = torch.tensor([2, 3, 5], dtype=torch.int64)
        # 创建一个int32类型的张量z，包含元素[1]
        z = torch.tensor([1], dtype=torch.int32)
        # 使用self.run_test方法运行AddModule模块，传入参数(x, y)
        self.run_test(AddModule(), (x, y))
        # 使用self.run_test方法运行SubModule模块，传入参数(x, y)
        self.run_test(SubModule(), (x, y))
        # 使用self.run_test方法运行MulModule模块，传入参数(x, y)
        self.run_test(MulModule(), (x, y))
        # 使用self.run_test方法运行DivModule模块，传入参数(x, y)
        self.run_test(DivModule(), (x, y))
        # 使用self.run_test方法运行PowModule模块，传入参数(x, z)
        self.run_test(PowModule(), (x, z))

    # 定义一个测试方法，测试布尔乘法运算
    def test_mul_bool(self):
        # 定义一个模型MyModel，继承自torch.nn.Module，实现前向传播为布尔乘法运算
        class MyModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.mul(x, y)

        # 创建一个包含布尔值的张量x，[True, False, True, False]
        x_t = torch.tensor([True, False, True, False])
        # 创建一个包含布尔值的张量y，[True, True, False, False]
        y_t = torch.tensor([True, True, False, False])
        # 创建一个包含浮点数的张量z，[1.0, 2.0, 3.0, 0.0]
        z_t = torch.tensor
    # 使用装饰器跳过不支持 Opset 版本大于等于 13 的测试
    @skipIfUnsupportedMaxOpsetVersion(13)
    def test_mod_with_low_precision(self):
        # 定义一个继承自 torch.nn.Module 的模块 ModModule
        class ModModule(torch.nn.Module):
            # 模块的前向传播方法，计算 x 和 y 的 torch.fmod，转换为 torch.long 类型
            def forward(self, x, y):
                return torch.fmod(x, y).to(dtype=torch.long)

        # 测试数据类型为 torch.uint8 的 x 和 y
        x = torch.tensor([2, 3, 5], dtype=torch.uint8)
        y = torch.tensor([2, 3, 5], dtype=torch.uint8)
        # 运行测试，使用 self.run_test 方法
        self.run_test(ModModule(), (x, y))

        # 测试数据类型为 torch.int8 的 x 和 y
        x = torch.tensor([2, 3, 5], dtype=torch.int8)
        y = torch.tensor([2, 3, 5], dtype=torch.int8)
        self.run_test(ModModule(), (x, y))

        # 测试数据类型为 torch.int16 的 x 和 y
        x = torch.tensor([2, 3, 5], dtype=torch.int16)
        y = torch.tensor([2, 3, 5], dtype=torch.int16)
        self.run_test(ModModule(), (x, y))

        # 测试数据类型为 torch.uint8 的 x 和 torch.int32 的 y
        x = torch.tensor([2, 3, 5], dtype=torch.uint8)
        y = torch.tensor([2, 3, 5], dtype=torch.int32)
        self.run_test(ModModule(), (x, y))

        # 测试数据类型为 torch.uint8 的 x 和 torch.float64 的 y
        x = torch.tensor([2, 3, 5], dtype=torch.uint8)
        y = torch.tensor([2, 3, 5], dtype=torch.float64)
        self.run_test(ModModule(), (x, y))

    # 使用装饰器跳过不支持 Opset 版本小于等于 9 的测试
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_empty_constant_shape(self):
        # 定义一个继承自 torch.nn.Module 的模块 Zeros
        class Zeros(torch.nn.Module):
            # 模块的前向传播方法，创建一个形状为空的全零张量 y，并加上输入张量 x
            def forward(self, x):
                y = torch.zeros(())
                y += x
                return y

        # 测试数据类型为 torch.tensor(42.0)
        x = torch.tensor(42.0)
        self.run_test(Zeros(), x)

        # 定义一个继承自 torch.nn.Module 的模块 Ones
        class Ones(torch.nn.Module):
            # 模块的前向传播方法，创建一个形状为空的全一张量 y，并加上输入张量 x
            def forward(self, x):
                y = torch.ones(())
                y += x
                return y

        x = torch.tensor(42.0)
        self.run_test(Ones(), x)

        # 定义一个继承自 torch.nn.Module 的模块 Full
        class Full(torch.nn.Module):
            # 模块的前向传播方法，创建一个形状为空的全一张量 y，并加上输入张量 x
            def forward(self, x):
                y = torch.full((), 1.0)
                y += x
                return y

        x = torch.tensor(42.0)
        self.run_test(Full(), x)

        # 定义一个继承自 torch.nn.Module 的模块 Empty
        class Empty(torch.nn.Module):
            # 模块的前向传播方法，创建一个形状为空的张量 y，填充为 0，然后加上输入张量 x
            def forward(self, x):
                y = torch.empty(()).fill_(0)
                y += x
                return y

        x = torch.tensor(42.0)
        self.run_test(Empty(), x)

    # 定义测试标准差的方法
    def test_std(self):
        # 定义一个继承自 torch.nn.Module 的模块 StandardDeviation
        class StandardDeviation(torch.nn.Module):
            # 模块的前向传播方法，计算输入张量的标准差，不使用无偏估计
            def forward(self, input):
                return torch.std(input, unbiased=False)

        # 生成一个形状为 (2, 3, 4) 的正态分布随机张量 x
        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        # 运行测试，使用 self.run_test 方法
        self.run_test(model, x)

        # 定义一个继承自 torch.nn.Module 的模块 StandardDeviationUnbiased
        class StandardDeviationUnbiased(torch.nn.Module):
            # 模块的前向传播方法，计算输入张量的标准差，使用无偏估计
            def forward(self, input):
                return torch.std(input, unbiased=True)

        model = StandardDeviationUnbiased()
        self.run_test(model, x)
    def test_std_along_dims(self):
        # 定义计算标准差的模块，对输入数据沿着第0和第1维度计算，使用有偏估计
        class StandardDeviation(torch.nn.Module):
            def forward(self, input):
                return torch.std(input, dim=(0, 1), unbiased=False)

        # 创建一个形状为(2, 3, 4)的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化标准差模块
        model = StandardDeviation()
        # 运行测试函数，验证模型的输出与预期是否一致
        self.run_test(model, x)

        # 定义计算标准差的模块，对输入数据沿着第0和第1维度计算，使用无偏估计
        class StandardDeviationUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.std(input, dim=(0, 1), unbiased=True)

        # 创建一个形状为(2, 3, 4)的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化无偏标准差模块
        model = StandardDeviationUnbiased()
        # 运行测试函数，验证模型的输出与预期是否一致
        self.run_test(model, x)

    def test_std_keepdim(self):
        # 定义计算标准差的模块，对输入数据沿着第0和第1维度计算，使用有偏估计，并保持维度
        class StandardDeviation(torch.nn.Module):
            def forward(self, input):
                return torch.std(input, dim=(0, 1), unbiased=False, keepdim=True)

        # 创建一个形状为(2, 3, 4)的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化保持维度的标准差模块
        model = StandardDeviation()
        # 运行测试函数，验证模型的输出与预期是否一致
        self.run_test(model, x)

        # 定义计算标准差的模块，对输入数据沿着第0和第1维度计算，使用无偏估计，并保持维度
        class StandardDeviationUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.std(input, dim=(0, 1), unbiased=True, keepdim=True)

        # 创建一个形状为(2, 3, 4)的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化保持维度的无偏标准差模块
        model = StandardDeviationUnbiased()
        # 运行测试函数，验证模型的输出与预期是否一致
        self.run_test(model, x)

    def test_std_correction(self):
        # 定义计算标准差的模块，对输入数据沿着第0和第1维度计算，使用有偏估计，并保持维度，校正系数设置为3
        class StandardDeviation(torch.nn.Module):
            def forward(self, input):
                return torch.std(input, dim=(0, 1), correction=3, keepdim=True)

        # 创建一个形状为(2, 3, 4)的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化带校正系数的标准差模块
        model = StandardDeviation()
        # 运行测试函数，验证模型的输出与预期是否一致
        self.run_test(model, x)

    def test_var(self):
        # 定义计算方差的模块，对输入数据使用有偏估计
        class Variance(torch.nn.Module):
            def forward(self, input):
                return torch.var(input, unbiased=False)

        # 创建一个形状为(2, 3, 4)的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化有偏方差模块
        model = Variance()
        # 运行测试函数，验证模型的输出与预期是否一致
        self.run_test(model, x)

        # 定义计算方差的模块，对输入数据使用无偏估计
        class VarianceUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.var(input, unbiased=True)

        # 创建一个形状为(2, 3, 4)的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化无偏方差模块
        model = VarianceUnbiased()
        # 运行测试函数，验证模型的输出与预期是否一致
        self.run_test(model, x)

        # 定义计算方差的模块，对输入数据沿着第1维度使用有偏估计，并对结果进行平方根处理，避免除以0的情况
        class VarianceSqrt(torch.nn.Module):
            def forward(self, input):
                y = torch.var(input, 1)
                return torch.sqrt(y + 1e-8)

        # 创建一个形状为(1, 2, 3, 300, 300)的随机张量
        x = torch.randn(1, 2, 3, 300, 300)
        # 实例化方差开根号模块
        model = VarianceSqrt()
        # 运行测试函数，验证模型的输出与预期是否一致
        self.run_test(model, x)

    def test_var_along_dims(self):
        # 定义计算方差的模块，对输入数据沿着第0和第1维度使用有偏估计
        class Variance(torch.nn.Module):
            def forward(self, input):
                return torch.var(input, dim=(0, 1), unbiased=False)

        # 创建一个形状为(2, 3, 4)的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化沿指定维度计算有偏方差模块
        model = Variance()
        # 运行测试函数，验证模型的输出与预期是否一致
        self.run_test(model, x)

        # 定义计算方差的模块，对输入数据沿着第0和第1维度使用无偏估计
        class VarianceUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.var(input, dim=(0, 1), unbiased=True)

        # 创建一个形状为(2, 3, 4)的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化沿指定维度计算无偏方差模块
        model = VarianceUnbiased()
        # 运行测试函数，验证模型的输出与预期是否一致
        self.run_test(model, x)
    def test_var_keepdim(self):
        # 定义计算方差的模块，保持维度为1
        class Variance(torch.nn.Module):
            def forward(self, input):
                return torch.var(input, dim=(0, 1), unbiased=False, keepdim=True)

        # 创建一个 2x3x4 的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化方差计算模块
        model = Variance()
        # 运行测试
        self.run_test(model, x)

        # 定义计算无偏方差的模块，保持维度为1
        class VarianceUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.var(input, dim=(0, 1), unbiased=True, keepdim=True)

        # 创建另一个 2x3x4 的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化无偏方差计算模块
        model = VarianceUnbiased()
        # 运行测试
        self.run_test(model, x)

    def test_var_correction(self):
        # 定义计算方差的模块，使用修正值为3，保持维度为1
        class Variance(torch.nn.Module):
            def forward(self, input):
                return torch.var(input, dim=(0, 1), correction=3, keepdim=True)

        # 创建一个 2x3x4 的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化方差计算模块
        model = Variance()
        # 运行测试
        self.run_test(model, x)

    def test_var_mean(self):
        # 定义计算方差均值的模块，不使用无偏估计
        class Variance(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, unbiased=False)

        # 创建一个 2x3x4 的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化方差均值计算模块
        model = Variance()
        # 运行测试
        self.run_test(model, x)

        # 定义计算无偏方差均值的模块
        class VarianceUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, unbiased=True)

        # 实例化无偏方差均值计算模块
        model = VarianceUnbiased()
        # 运行测试
        self.run_test(model, x)

    def test_var_mean_along_dims(self):
        # 定义沿指定维度计算方差均值的模块，不使用无偏估计
        class Variance(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, dim=(0, 1), unbiased=False)

        # 创建一个 2x3x4 的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化沿指定维度计算方差均值的模块
        model = Variance()
        # 运行测试
        self.run_test(model, x)

        # 定义沿指定维度计算无偏方差均值的模块
        class VarianceUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, dim=(0, 1), unbiased=True)

        # 创建另一个 2x3x4 的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化沿指定维度计算无偏方差均值的模块
        model = VarianceUnbiased()
        # 运行测试
        self.run_test(model, x)

    def test_var_mean_mixed_dims(self):
        # 定义沿多个混合维度计算方差均值的模块，不使用无偏估计
        class ReverseDims(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, dim=(2, 1), unbiased=False)

        # 创建一个 2x3x4 的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化沿多个混合维度计算方差均值的模块
        model = ReverseDims()
        # 运行测试
        self.run_test(model, x)

        # 定义沿指定维度计算方差均值的模块，跳过某些维度，不使用无偏估计
        class SkipDims(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, dim=(0, 2), unbiased=False)

        # 创建另一个 2x3x4 的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化沿指定维度计算方差均值的模块，跳过某些维度
        model = SkipDims()
        # 运行测试
        self.run_test(model, x)

        # 定义沿指定维度计算方差均值的模块，保留非零维度，不使用无偏估计
        class NonZeroDims(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, dim=(1, 2), unbiased=False)

        # 创建另一个 2x3x4 的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化沿指定维度计算方差均值的模块，保留非零维度
        model = NonZeroDims()
        # 运行测试
        self.run_test(model, x)
    def test_var_mean_keepdim(self):
        # 定义计算方差和均值的模型
        class Variance(torch.nn.Module):
            def forward(self, input):
                # 调用 torch.var_mean 函数计算输入张量的方差和均值
                return torch.var_mean(input, dim=(0, 1), unbiased=False, keepdim=True)

        # 创建一个形状为 (2, 3, 4) 的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化 Variance 类
        model = Variance()
        # 运行测试函数，测试模型对输入 x 的输出
        self.run_test(model, x)

        # 定义计算无偏方差和均值的模型
        class VarianceUnbiased(torch.nn.Module):
            def forward(self, input):
                # 调用 torch.var_mean 函数计算输入张量的无偏方差和均值
                return torch.var_mean(input, dim=(0, 1), unbiased=True, keepdim=True)

        # 创建一个形状为 (2, 3, 4) 的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化 VarianceUnbiased 类
        model = VarianceUnbiased()
        # 运行测试函数，测试模型对输入 x 的输出
        self.run_test(model, x)

    def test_var_mean_correction(self):
        # 定义计算带修正的方差和均值的模型
        class Variance(torch.nn.Module):
            def forward(self, input):
                # 调用 torch.var_mean 函数计算输入张量的方差和均值，使用修正值为 3
                return torch.var_mean(input, dim=(0, 1), correction=3, keepdim=True)

        # 创建一个形状为 (2, 3, 4) 的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化 Variance 类
        model = Variance()
        # 运行测试函数，测试模型对输入 x 的输出
        self.run_test(model, x)

    def test_std_mean(self):
        # 定义计算标准差和均值的模型
        class StandardDeviation(torch.nn.Module):
            def forward(self, input):
                # 调用 torch.std_mean 函数计算输入张量的标准差和均值，不使用无偏估计
                return torch.std_mean(input, unbiased=False)

        # 创建一个形状为 (2, 3, 4) 的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化 StandardDeviation 类
        model = StandardDeviation()
        # 运行测试函数，测试模型对输入 x 的输出
        self.run_test(model, x)

        # 定义计算无偏标准差和均值的模型
        class StandardDeviationUnbiased(torch.nn.Module):
            def forward(self, input):
                # 调用 torch.std_mean 函数计算输入张量的无偏标准差和均值
                return torch.std_mean(input, unbiased=True)

        # 实例化 StandardDeviationUnbiased 类
        model = StandardDeviationUnbiased()
        # 运行测试函数，测试模型对输入 x 的输出
        self.run_test(model, x)

    def test_std_mean_along_dims(self):
        # 定义沿指定维度计算标准差和均值的模型
        class StandardDeviation(torch.nn.Module):
            def forward(self, input):
                # 调用 torch.std_mean 函数沿 (0, 1) 维度计算输入张量的标准差和均值，不使用无偏估计
                return torch.std_mean(input, dim=(0, 1), unbiased=False)

        # 创建一个形状为 (2, 3, 4) 的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化 StandardDeviation 类
        model = StandardDeviation()
        # 运行测试函数，测试模型对输入 x 的输出
        self.run_test(model, x)

        # 定义沿指定维度计算无偏标准差和均值的模型
        class VarianceUnbiased(torch.nn.Module):
            def forward(self, input):
                # 调用 torch.std_mean 函数沿 (0, 1) 维度计算输入张量的无偏标准差和均值
                return torch.std_mean(input, dim=(0, 1), unbiased=True)

        # 创建一个形状为 (2, 3, 4) 的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化 VarianceUnbiased 类
        model = VarianceUnbiased()
        # 运行测试函数，测试模型对输入 x 的输出
        self.run_test(model, x)

    def test_std_mean_keepdim(self):
        # 定义沿指定维度计算标准差和均值并保持维度的模型
        class StandardDeviation(torch.nn.Module):
            def forward(self, input):
                # 调用 torch.std_mean 函数沿 (0, 1) 维度计算输入张量的标准差和均值，不使用无偏估计，并保持维度
                return torch.std_mean(input, dim=(0, 1), unbiased=False, keepdim=True)

        # 创建一个形状为 (2, 3, 4) 的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化 StandardDeviation 类
        model = StandardDeviation()
        # 运行测试函数，测试模型对输入 x 的输出
        self.run_test(model, x)

        # 定义沿指定维度计算无偏标准差和均值并保持维度的模型
        class StandardDeviationUnbiased(torch.nn.Module):
            def forward(self, input):
                # 调用 torch.std_mean 函数沿 (0, 1) 维度计算输入张量的无偏标准差和均值，并保持维度
                return torch.std_mean(input, dim=(0, 1), unbiased=True, keepdim=True)

        # 创建一个形状为 (2, 3, 4) 的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化 StandardDeviationUnbiased 类
        model = StandardDeviationUnbiased()
        # 运行测试函数，测试模型对输入 x 的输出
        self.run_test(model, x)

    def test_std_mean_correction(self):
        # 定义计算带修正的标准差和均值的模型
        class StandardDeviation(torch.nn.Module):
            def forward(self, input):
                # 调用 torch.var_mean 函数计算输入张量的方差和均值，使用修正值为 3，并保持维度
                return torch.var_mean(input, dim=(0, 1), correction=3, keepdim=True)

        # 创建一个形状为 (2, 3, 4) 的随机张量
        x = torch.randn(2, 3, 4)
        # 实例化 StandardDeviation 类
        model = StandardDeviation()
        # 运行测试函数，测试模型对输入 x 的输出
        self.run_test(model, x)
    def test_bitshift(self):
        # 定义一个内部的 BitshiftModel 类，继承自 torch.nn.Module
        class BitshiftModel(torch.nn.Module):
            # 定义前向传播函数
            def forward(self, input):
                # 返回输入按位右移1位，左移3位，以及按照张量 [1, 2] 右移和左移4位的结果
                return (
                    input >> 1,
                    input << 3,
                    input >> torch.tensor([1, 2]),
                    input << 4,
                )

        # 创建一个 torch 的 int64 类型的张量，形状为 (3, 4, 2)，数值为 0 到 23
        input = torch.arange(24, dtype=torch.int64).reshape(3, 4, 2)
        # 运行测试，传入 BitshiftModel 的实例和 input 张量
        self.run_test(BitshiftModel(), input)

    @skipIfUnsupportedMinOpsetVersion(18)
    def test_bitwise_and(self):
        # 定义一个内部的 BitwiseAndModel 类，继承自 torch.nn.Module
        class BitwiseAndModel(torch.nn.Module):
            # 定义前向传播函数，接受 input 和 other 两个参数
            def forward(self, input, other):
                # 返回 input 按位与 20 的结果，input 和 other 按位与的结果，以及 other 按位与张量 [1, 2] 的结果
                return (
                    input & 20,
                    torch.bitwise_and(input, other),
                    other & torch.tensor([1, 2], dtype=torch.int32),
                )

        # 创建一个 uint8 类型的张量 input，形状为 (3, 4, 2)，数值为 0 到 254 之间的随机数
        input = torch.randint(0, 255, (3, 4, 2), dtype=torch.uint8)
        # 创建一个 int8 类型的张量 other，形状为 (3, 4, 2)，数值为 -128 到 126 之间的随机数
        other = torch.randint(-128, 127, (3, 4, 2), dtype=torch.int8)
        # 运行测试，传入 BitwiseAndModel 的实例和 (input, other) 元组
        self.run_test(BitwiseAndModel(), (input, other))

    # uint8 not implemented in ORT for Mul used in
    # exporting bitshift for opset_version < 10
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_bitshift_uint8(self):
        # 定义一个内部的 BitshiftModel 类，继承自 torch.nn.Module
        class BitshiftModel(torch.nn.Module):
            # 定义前向传播函数，接受 input 和 input2 两个参数
            def forward(self, input, input2):
                # 返回 input 按位右移1位，左移3位，input2 按位右移和左移4位的结果
                return (
                    input >> 1,
                    input << 3,
                    input2 >> torch.tensor([1, 2], dtype=torch.uint8),
                    input2 << 4,
                )

        # 创建一个 uint8 类型的张量 input 和 input2，形状为 (3, 4, 2)，数值为 0 到 23 之间的随机数
        input = torch.arange(24, dtype=torch.uint8).reshape(3, 4, 2)
        input2 = torch.arange(24, dtype=torch.uint8).reshape(3, 4, 2)
        # 运行测试，传入 BitshiftModel 的实例和 (input, input2) 元组
        self.run_test(BitshiftModel(), (input, input2))

    def test_narrow(self):
        # 定义一个内部的 NarrowModel 类，继承自 torch.nn.Module
        class NarrowModel(torch.nn.Module):
            # 定义前向传播函数，接受 input 一个参数
            def forward(self, input):
                # 返回对 input 张量的第0维进行缩窄操作，从索引0开始，长度为2
                return torch.narrow(input, 0, 0, 2)

        # 创建一个形状为 (3, 3) 的浮点数张量 x，用于测试
        x = torch.randn(3, 3, requires_grad=True)
        # 运行测试，传入 NarrowModel 的实例和张量 x
        self.run_test(NarrowModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_narrow_dynamic(self):
        # 定义一个内部的 NarrowModel 类，继承自 torch.nn.Module
        class NarrowModel(torch.nn.Module):
            # 定义前向传播函数，接受 input 一个参数
            def forward(self, input):
                # 返回对 input 张量的第0维进行缩窄操作，从索引0开始，长度为 input.shape[0] - 1
                return torch.narrow(input, 0, 0, input.shape[0] - 1)

        # 创建一个形状为 (3, 3) 的浮点数张量 x，用于测试
        x = torch.randn(3, 3, requires_grad=True)
        # 运行测试，传入 NarrowModel 的实例和张量 x
        self.run_test(NarrowModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_fill(self):
        # 定义一个内部的 IndexFillModel 类，继承自 torch.nn.Module
        class IndexFillModel(torch.nn.Module):
            # 定义前向传播函数，接受 input 一个参数
            def forward(self, input):
                # 创建一个索引张量 [2, 0]，用 -1 填充 input 张量的第2维
                index = torch.tensor([2, 0])
                return input.index_fill(2, index, -1)

        # 创建一个形状为 (3, 4, 5) 的浮点数张量 x，用于测试
        x = torch.randn(3, 4, 5, requires_grad=True)
        # 运行测试，传入 IndexFillModel 的实例和张量 x
        self.run_test(IndexFillModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_copy(self):
        # 定义一个继承自 torch.nn.Module 的模型类 IndexCopyModel，用于测试索引复制操作
        class IndexCopyModel(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, input):
                # 定义索引 tensor
                index = torch.tensor([2, 0])
                # 创建一个尺寸为 (3, 2, 5) 的全为 1 的张量作为源数据
                source = torch.ones(3, 2, 5)
                # 对输入张量执行索引复制操作，返回结果
                return input.index_copy(self.dim, index, source)

        # 创建一个尺寸为 (3, 4, 5) 的随机张量，并标记为需要梯度计算
        x = torch.randn(3, 4, 5, requires_grad=True)
        # 遍历维度值 1 和 -2，对 IndexCopyModel 进行测试
        for dim in (1, -2):
            self.run_test(IndexCopyModel(dim), x)

    def test_select(self):
        # 定义一个继承自 torch.nn.Module 的模型类 Select，用于测试选择操作
        class Select(torch.nn.Module):
            def forward(self, x):
                # 返回输入张量的所有行，但只选择第二列的数据
                return x[:, 1]

        # 创建一个尺寸为 (3, 4) 的随机张量 x
        x = torch.randn(3, 4)
        # 对 Select 模型进行测试
        self.run_test(Select(), x)

    def test_select_negative_index(self):
        # 定义一个继承自 torch.nn.Module 的模型类 Select，用于测试选择操作（使用负索引）
        class Select(torch.nn.Module):
            def forward(self, x):
                # 返回输入张量的所有行，但只选择倒数第一列的数据
                return x[:, -1]

        # 创建一个尺寸为 (3, 4) 的随机张量 x
        x = torch.randn(3, 4)
        # 对 Select 模型进行测试
        self.run_test(Select(), x)

    def test_index_select_constant_scaler_index(self):
        # 定义一个继承自 torch.nn.Module 的模型类 IndexSelectScalerIndexModel，用于测试索引选择操作（常量标量索引）
        class IndexSelectScalerIndexModel(torch.nn.Module):
            def forward(self, x):
                # 定义索引为 2 的常量标量
                index = 2
                # 对输入张量 x 执行索引选择操作，选择第 1 维度的第 index 列数据
                return torch.index_select(x, 1, torch.tensor(index))

        # 创建一个尺寸为 (3, 4) 的随机张量 x
        x = torch.randn(3, 4)
        # 对 IndexSelectScalerIndexModel 进行测试
        self.run_test(IndexSelectScalerIndexModel(), x)

    def test_index_select_scaler_index(self):
        # 定义一个继承自 torch.nn.Module 的模型类 IndexSelectScalerIndexModel，用于测试索引选择操作（标量索引）
        class IndexSelectScalerIndexModel(torch.nn.Module):
            def __init__(self, index_base):
                super().__init__()
                self.index_base = torch.tensor(index_base)

            def forward(self, x, index_offset):
                # 将基础索引和偏移索引相加，得到最终索引
                index = self.index_base + index_offset
                # 对输入张量 x 执行索引选择操作，选择第 1 维度的 index 列数据
                return torch.index_select(x, 1, index)

        # 创建一个尺寸为 (3, 4) 的随机张量 x 和一个偏移索引为 2 的张量 index_offset
        x = torch.randn(3, 4)
        offset = 2
        index_offset = torch.tensor(offset)
        base = 1
        # 对 IndexSelectScalerIndexModel 进行测试
        self.run_test(IndexSelectScalerIndexModel(base), (x, index_offset))

    def test_take(self):
        # 定义一个继承自 torch.nn.Module 的模型类 TakeModel，用于测试 take 操作
        class TakeModel(torch.nn.Module):
            def forward(self, x, y):
                # 在张量 x 中根据张量 y 中的索引取值
                return torch.take(x, y)

        # 创建一个尺寸为 (6, 4, 3, 3) 的随机张量 x 和一个索引张量 y
        x = torch.randn(6, 4, 3, 3)
        y = torch.tensor([4, 1, 7, 15, 63])
        # 对 TakeModel 进行测试
        self.run_test(TakeModel(), (x, y))

    def test_topk(self):
        # 定义一个继承自 torch.nn.Module 的模型类 MyModule，用于测试 topk 操作
        class MyModule(torch.nn.Module):
            def forward(self, x):
                # 返回张量 x 中的前 3 个最大值及其索引
                return torch.topk(x, 3)

        # 创建一个从 1 到 6 的张量 x，并标记为需要梯度计算
        x = torch.arange(1.0, 6.0, requires_grad=True)
        # 对 MyModule 进行测试
        self.run_test(MyModule(), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_topk_int32_k(self):
        # 定义一个继承自 torch.nn.Module 的模型类 Model，用于测试 topk 操作（使用 int32 类型的 k）
        class Model(torch.nn.Module):
            def forward(self, x, k):
                # 返回张量 x 中的前 k 个最大值及其索引
                return torch.topk(x, k)

        # 创建一个从 1 到 6 的张量 x 和一个 k 值为 3 的 int32 类型张量 k
        x = torch.arange(1.0, 6.0)
        k = torch.tensor(3, dtype=torch.int32)
        # 对 Model 进行测试
        self.run_test(Model(), (x, k))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_topk_smallest_unsorted(self):
        # 定义一个继承自torch.nn.Module的内部类MyModule，用于测试topk函数的最小k个非排序输出
        class MyModule(torch.nn.Module):
            def forward(self, x, k):
                # 使用torch.topk函数获取张量x中最小的k个值，largest=False表示取最小的k个，sorted=False表示不排序
                topk_unsorted = torch.topk(x, k, largest=False, sorted=False)
                # 使用torch.topk函数获取张量x中最小的k个值，largest=False表示取最小的k个，sorted=True表示排序
                topk_sorted = torch.topk(x, k, largest=False, sorted=True)
                # 返回排序后的topk值和未排序的topk值按值排序后的结果
                return topk_sorted, torch.sort(topk_unsorted.values).values

        # 创建一个包含1.0到5.0的张量x，要求梯度
        x = torch.arange(1.0, 6.0, requires_grad=True)
        # 创建一个张量k，值为3
        k = torch.tensor(3)
        # 运行测试函数run_test，传入MyModule的实例和(x, k)作为参数
        self.run_test(MyModule(), (x, k))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_topk_script(self):
        # 定义一个继承自torch.jit.ScriptModule的内部类MyModuleDynamic，用于测试torch.topk函数的脚本化版本
        class MyModuleDynamic(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x, k):
                # 调用torch.topk函数获取张量x中最大的k个值
                return torch.topk(x, k)

        # 创建一个包含1.0到5.0的张量x，要求梯度
        x = torch.arange(1.0, 6.0, requires_grad=True)
        # 创建一个张量k，值为3
        k = torch.tensor(3)
        # 运行测试函数run_test，传入MyModuleDynamic的实例和(x, k)作为参数
        self.run_test(MyModuleDynamic(), (x, k))

    @skipScriptTest()  # Python builtin apply of FunctionMeta object is currently not supported in Torchscript.
    @skipIfUnsupportedMinOpsetVersion(11)  # Clip op min is an input since opset 11.
    def test_auto_grad(self):
        # 定义一个自定义的torch.autograd.Function类MyClip，用于实现自定义的梯度计算操作
        class MyClip(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, scalar):
                # 保存输入张量input到上下文中，并返回输入张量input的按照scalar最小值截断后的结果
                ctx.save_for_backward(input)
                return input.clamp(min=scalar)

        # 定义一个自定义的torch.autograd.Function类MyRelu，用于实现自定义的ReLU激活函数操作
        class MyRelu(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                # 保存输入张量input到上下文中，并返回输入张量input的ReLU激活函数结果
                ctx.save_for_backward(input)
                return input.clamp(min=0)

        # 定义一个符号化Python操作函数symbolic_python_op，用于在ONNX导出中处理自定义操作
        def symbolic_python_op(
            ctx: torch.onnx.SymbolicContext, g: torch._C.Graph, *args, **kwargs
        ):
            n = ctx.cur_node
            name = kwargs["name"]
            if name == "MyClip":
                # 对于操作名称为"MyClip"，使用ONNX图中的Clip操作处理输入张量args[0]和标量args[1]
                return g.op("Clip", args[0], args[1], outputs=n.outputsSize())
            elif name == "MyRelu":
                # 对于操作名称为"MyRelu"，使用ONNX图中的Relu操作处理输入张量args[0]
                return g.op("Relu", args[0], outputs=n.outputsSize())
            else:
                # 对于未知的操作名称，返回未实现的错误
                return torch.onnx.symbolic_helper._unimplemented(
                    "prim::PythonOp", "unknown node kind: " + name
                )

        # 在ONNX导出注册自定义操作符"prim::PythonOp"和相应的符号化Python操作函数symbolic_python_op
        torch.onnx.register_custom_op_symbolic("prim::PythonOp", symbolic_python_op, 1)
        # 添加清理函数，用于在测试结束时取消注册的自定义操作符"prim::PythonOp"
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, "prim::PythonOp", 1)

        # 定义一个MyClipModule类，继承自torch.nn.Module，用于将MyClip函数作为模块封装
        class MyClipModule(torch.nn.Module):
            def forward(self, x, min):
                # 调用自定义的MyClip函数，对输入张量x进行截断操作，截断最小值为min
                return MyClip.apply(x, min)

        # 创建一个形状为(3, 3)的随机张量x
        x = torch.randn(3, 3)
        # 创建一个标量张量min，值为0.0
        min = torch.tensor([0.0])
        # 运行测试函数run_test，传入MyClipModule的实例和(x, min)作为参数
        self.run_test(MyClipModule(), (x, min))

        # 定义一个MyReluModule类，继承自torch.nn.Module，用于将MyRelu函数作为模块封装
        class MyReluModule(torch.nn.Module):
            def forward(self, x):
                # 调用自定义的MyRelu函数，对输入张量x进行ReLU激活函数操作
                return MyRelu.apply(x)

        # 创建一个形状为(3, 3)的随机张量x
        x = torch.randn(3, 3)
        # 运行测试函数run_test，传入MyReluModule的实例和x作为参数
        self.run_test(MyReluModule(), x)
    # 定义一个测试用例，测试 torch.clamp 函数对整型张量的操作
    def test_clip_int(self):
        # 定义一个简单的 torch.nn.Module 子类，实现 forward 方法来执行 torch.clamp 操作
        class MyClipInt(torch.nn.Module):
            def forward(self, x):
                return torch.clamp(x, 0, 1)

        # 运行测试，使用随机生成的大小为 3x3 的整型张量作为输入
        self.run_test(MyClipInt(), torch.randn(3, 3).to(torch.int64))

    # 定义一个测试用例，测试 torch.nn.ReLU 对整型张量的操作
    def test_relu_int(self):
        # 运行测试，使用随机生成的大小为 3x3 的整型张量作为输入
        self.run_test(torch.nn.ReLU(), torch.randn(3, 3).to(torch.int32))

    # 定义一个测试用例，测试 torch.nn.functional.pad 对整型张量的操作
    def test_pad_int(self):
        # 定义一个简单的 torch.nn.Module 子类，实现 forward 方法来执行 torch.nn.functional.pad 操作
        class MyPadInt(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.pad(x, (1, 1))

        # 运行测试，使用随机生成的大小为 3x3 的整型张量作为输入
        self.run_test(MyPadInt(), torch.randn(3, 3).to(torch.int32))

    # 定义一个测试用例，测试 torch.min 对整型张量的操作
    def test_min_int(self):
        # 定义一个简单的 torch.nn.Module 子类，实现 forward 方法来执行 torch.min 操作
        class MyMinInt(torch.nn.Module):
            def forward(self, x):
                return torch.min(x, x + 1)

        # 运行测试，使用随机生成的大小为 3x3 的整型张量作为输入
        self.run_test(MyMinInt(), torch.randn(3, 3).to(torch.int32))

    # 定义一个测试用例，测试 torch.max 对整型张量的操作
    def test_max_int(self):
        # 定义一个简单的 torch.nn.Module 子类，实现 forward 方法来执行 torch.max 操作
        class MyMaxnInt(torch.nn.Module):
            def forward(self, x):
                return torch.max(x, x + 1)

        # 运行测试，使用随机生成的大小为 3x3 的整型张量作为输入
        self.run_test(MyMaxnInt(), torch.randn(3, 3).to(torch.int32))

    # 根据 opset 版本进行条件跳过，测试 torch.nn.functional.normalize 对张量的操作
    @skipIfUnsupportedOpsetVersion([7])
    def test_normalize(self):
        # 定义一个简单的 torch.nn.Module 子类，实现 forward 方法来执行 torch.nn.functional.normalize 操作
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.normalize(x)

        # 生成大小为 3x3 的随机张量作为输入，并运行测试
        x = torch.randn(3, 3)
        self.run_test(Model(), x)

    # 定义一个测试用例，测试 torch.ops.aten.norm 对张量的操作，包含类型声明
    def test_norm_with_dtype(self):
        # 定义一个简单的 torch.nn.Module 子类，实现 forward 方法来执行 torch.ops.aten.norm 操作
        class Model(torch.nn.Module):
            def forward(self, x):
                # TODO(bowbao): 在当前的测试基础设施中，测试 aten 操作存在一些小问题
                # OpInfo `torch.norm` 在 `common_methods_invocations.py` 中将不会分解到下面的 aten 操作。
                return torch.ops.aten.norm(
                    x, p=2, dim=[1], keepdim=True, dtype=torch.float64
                )

        # 生成大小为 3x3 的随机张量作为输入，并运行测试
        x = torch.randn(3, 3)
        self.run_test(Model(), x)

    # 定义一个测试用例，测试 torch.nn.LayerNorm 对张量的操作
    def test_layer_norm(self):
        # 由于 layer_norm 在最后一个维度上工作，请保持输入至少三维，以避免 axis=2 映射到与 axis=-2 相同的轴
        for elementwise_affine in (True, False):
            for bias in (True, False):
                # 创建 LayerNorm 模型，并生成大小为 20x5x10x10x10 的随机张量作为输入，并运行测试
                model = torch.nn.LayerNorm(
                    [10, 10, 10], elementwise_affine=elementwise_affine, bias=bias
                )
                x = torch.randn(20, 5, 10, 10, 10)
                self.run_test(model, x)

    # 定义一个测试用例，测试 torch.nn.BatchNorm1d 对张量的操作（包含 affine 参数）
    def test_batchnorm1d(self):
        # 生成大小为 10x10 的随机张量作为输入，并运行 BatchNorm1d 测试
        x = torch.randn(10, 10)
        model = torch.nn.BatchNorm1d(10, affine=True)
        self.run_test(model, x)

        # 生成大小为 10x10x128 的随机张量作为输入，并再次运行 BatchNorm1d 测试
        x = torch.randn(10, 10, 128)
        self.run_test(model, x)

    # 定义一个测试用例，测试 torch.nn.BatchNorm1d 对张量的操作（不包含 affine 参数）
    def test_batchnorm1d_noaffine(self):
        # 生成大小为 10x10 的随机张量作为输入，并运行 BatchNorm1d 测试（不使用 affine 参数）
        x = torch.randn(10, 10)
        model = torch.nn.BatchNorm1d(10, affine=False)
        self.run_test(model, x)

        # 生成大小为 10x10x128 的随机张量作为输入，并再次运行 BatchNorm1d 测试（不使用 affine 参数）
        x = torch.randn(10, 10, 128)
        self.run_test(model, x)
    # 测试 BatchNorm1d 层，不使用运行时统计信息
    def test_batchnorm1d_norunningstats(self):
        # 生成一个大小为 (10, 10) 的随机张量
        x = torch.randn(10, 10)
        # 创建一个 BatchNorm1d 模型，关闭运行时统计信息
        model = torch.nn.BatchNorm1d(10, track_running_stats=False)
        # 运行测试函数，对模型进行测试
        self.run_test(model, x)

        # 生成一个大小为 (10, 10, 128) 的随机张量
        x = torch.randn(10, 10, 128)
        # 再次运行测试函数，对同一模型进行测试
        self.run_test(model, x)

    # 测试 BatchNorm2d 层
    def test_batchnorm2d(self):
        # 生成一个大小为 (10, 3, 128, 128) 的随机张量
        x = torch.randn(10, 3, 128, 128)
        # 创建一个具有仿射变换的 BatchNorm2d 模型
        model = torch.nn.BatchNorm2d(3, affine=True)
        # 运行测试函数，对模型进行测试
        self.run_test(model, x)

    # 测试 BatchNorm2d 层，不使用仿射变换
    def test_batchnorm2d_noaffine(self):
        # 生成一个大小为 (10, 3, 128, 128) 的随机张量
        x = torch.randn(10, 3, 128, 128)
        # 创建一个不具有仿射变换的 BatchNorm2d 模型
        model = torch.nn.BatchNorm2d(3, affine=False)
        # 运行测试函数，对模型进行测试
        self.run_test(model, x)

    # 测试 BatchNorm2d 层，不使用运行时统计信息
    def test_batchnorm2d_norunningstats(self):
        # 生成一个大小为 (10, 3, 128, 128) 的随机张量
        x = torch.randn(10, 3, 128, 128)
        # 创建一个不使用运行时统计信息的 BatchNorm2d 模型
        model = torch.nn.BatchNorm2d(3, track_running_stats=False)
        # 运行测试函数，对模型进行测试
        self.run_test(model, x)

    # 测试 BatchNorm3d 层
    def test_batchnorm3d(self):
        # 生成一个大小为 (10, 3, 64, 64, 64) 的随机张量
        x = torch.randn(10, 3, 64, 64, 64)
        # 创建一个具有仿射变换的 BatchNorm3d 模型
        model = torch.nn.BatchNorm3d(3, affine=True)
        # 运行测试函数，对模型进行测试
        self.run_test(model, x)

    # 测试 BatchNorm3d 层，不使用仿射变换
    def test_batchnorm3d_noaffine(self):
        # 生成一个大小为 (10, 3, 64, 64, 64) 的随机张量
        x = torch.randn(10, 3, 64, 64, 64)
        # 创建一个不具有仿射变换的 BatchNorm3d 模型
        model = torch.nn.BatchNorm3d(3, affine=False)
        # 运行测试函数，对模型进行测试
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(
        9
    )  # 因为 ConstantOfShape 操作在 opset < 9 中不受支持
    # 测试 InstanceNorm1d 层，使用运行时统计信息
    def test_instancenorm1d_runningstats(self):
        # 生成一个大小为 (10, 5, 128) 的随机张量
        x = torch.randn(10, 5, 128)
        # 创建一个具有仿射变换且使用运行时统计信息的 InstanceNorm1d 模型
        model = torch.nn.InstanceNorm1d(5, affine=True, track_running_stats=True)
        # 运行测试函数，对模型进行测试
        self.run_test(model, x)

        # 创建一个不具有仿射变换但使用运行时统计信息的 InstanceNorm1d 模型
        model = torch.nn.InstanceNorm1d(5, affine=False, track_running_stats=True)
        # 再次运行测试函数，对模型进行测试
        self.run_test(model, x)

    # 测试 InstanceNorm1d 层，不使用运行时统计信息
    def test_instancenorm1d_norunningstats(self):
        # 生成一个大小为 (10, 5, 128) 的随机张量
        x = torch.randn(10, 5, 128)
        # 创建一个不具有仿射变换且不使用运行时统计信息的 InstanceNorm1d 模型
        model = torch.nn.InstanceNorm1d(5, affine=True, track_running_stats=False)
        # 运行测试函数，对模型进行测试
        self.run_test(model, x)

        # 创建一个不具有仿射变换且不使用运行时统计信息的 InstanceNorm1d 模型
        model = torch.nn.InstanceNorm1d(5, affine=False, track_running_stats=False)
        # 再次运行测试函数，对模型进行测试
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(
        9
    )  # 因为 ConstantOfShape 操作在 opset < 9 中不受支持
    # 测试 InstanceNorm2d 层，使用运行时统计信息
    def test_instancenorm2d_runningstats(self):
        # 生成一个大小为 (10, 3, 128, 128) 的随机张量
        x = torch.randn(10, 3, 128, 128)
        # 创建一个具有仿射变换且使用运行时统计信息的 InstanceNorm2d 模型
        model = torch.nn.InstanceNorm2d(3, affine=True, track_running_stats=True)
        # 运行测试函数，对模型进行测试
        self.run_test(model, x)

        # 创建一个不具有仿射变换但使用运行时统计信息的 InstanceNorm2d 模型
        model = torch.nn.InstanceNorm2d(3, affine=False, track_running_stats=True)
        # 再次运行测试函数，对模型进行测试
        self.run_test(model, x)

    # 测试 InstanceNorm2d 层，不使用运行时统计信息
    def test_instancenorm2d_norunningstats(self):
        # 生成一个大小为 (10, 3, 128, 128) 的随机张量
        x = torch.randn(10, 3, 128, 128)
        # 创建一个不具有仿射变换且不使用运行时统计信息的 InstanceNorm2d 模型
        model = torch.nn.InstanceNorm2d(3, affine=True, track_running_stats=False)
        # 运行测试函数，对模型进行测试
        self.run_test(model, x)

        # 创建一个不具有仿射变换且不使用运行时统计信息的 InstanceNorm2d 模型
        model = torch.nn.InstanceNorm2d(3, affine=False, track_running_stats=False)
        # 再次运行测试函数，对模型进行测试
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(
        9
    )  # 因为 ConstantOfShape 操作在 opset < 9 中不受支持
    def test_instancenorm3d_runningstats(self):
        # 创建一个大小为 (10, 3, 64, 64, 64) 的随机张量作为输入数据
        x = torch.randn(10, 3, 64, 64, 64)
        # 创建一个 InstanceNorm3d 模型，affine=True 表示使用可学习参数，track_running_stats=True 表示追踪运行时统计信息
        model = torch.nn.InstanceNorm3d(3, affine=True, track_running_stats=True)
        # 运行测试，将模型和输入数据传递给 self.run_test 方法
        self.run_test(model, x)

        # 创建一个 InstanceNorm3d 模型，affine=False 表示不使用可学习参数，track_running_stats=True 表示追踪运行时统计信息
        model = torch.nn.InstanceNorm3d(3, affine=False, track_running_stats=True)
        # 再次运行测试，将模型和输入数据传递给 self.run_test 方法
        self.run_test(model, x)

    def test_instancenorm3d_norunningstats(self):
        # 创建一个大小为 (10, 3, 64, 64, 64) 的随机张量作为输入数据
        x = torch.randn(10, 3, 64, 64, 64)
        # 创建一个 InstanceNorm3d 模型，affine=True 表示使用可学习参数，track_running_stats=False 表示不追踪运行时统计信息
        model = torch.nn.InstanceNorm3d(3, affine=True, track_running_stats=False)
        # 运行测试，将模型和输入数据传递给 self.run_test 方法
        self.run_test(model, x)

        # 创建一个 InstanceNorm3d 模型，affine=False 表示不使用可学习参数，track_running_stats=False 表示不追踪运行时统计信息
        model = torch.nn.InstanceNorm3d(3, affine=False, track_running_stats=False)
        # 再次运行测试，将模型和输入数据传递给 self.run_test 方法
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scatter_with_scalar(self):
        # 定义一个 ScatterModel 类，用于实现输入数据的 scatter 操作
        class ScatterModel(torch.nn.Module):
            def forward(self, input, indices):
                # 定义要分散的值为 1.0
                values = 1.0
                # 执行 input 的 scatter 操作，将 values 根据 indices 分散到 input 中
                return input.scatter(1, indices, values)

        # 创建一个浮点型张量 input，形状为 (3, 3)，元素全为 0.0
        input = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
        )
        # 创建一个索引张量 indices，形状为 (3, 2)，指定了 input 中每行要更新的位置
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        # 运行测试，将 ScatterModel 实例和 input、indices 作为输入参数传递给 self.run_test 方法
        self.run_test(ScatterModel(), input_args=(input, indices))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scatter_with_scalar_different_types(self):
        # 测试当标量值的类型与 input 张量的类型不同时的情况
        # 这种情况只会出现在标量值为 src 且 src 是一个张量时
        class ScatterModel(torch.nn.Module):
            def forward(self, input, indices):
                # 定义要分散的值为 1.0
                values = 1.0
                # 执行 input 的 scatter 操作，将 values 根据 indices 分散到 input 中
                return input.scatter(1, indices, values)

        # 创建一个浮点型张量 input，形状为 (3, 3)，元素全为 0.0
        input = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32
        )
        # 创建一个索引张量 indices，形状为 (3, 2)，指定了 input 中每行要更新的位置
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        # 运行测试，将 ScatterModel 实例和 input、indices 作为输入参数传递给 self.run_test 方法
        self.run_test(ScatterModel(), input_args=(input, indices))
    # 定义一个名为 test_scatter 的测试方法
    def test_scatter(self):
        # 定义一个内嵌的 ScatterModel 类，继承自 torch.nn.Module
        class ScatterModel(torch.nn.Module):
            # 定义模型的前向传播方法，接受 input, indices, values 三个参数，执行 scatter 操作后返回结果
            def forward(self, input, indices, values):
                return input.scatter(1, indices, values)

        # 创建一个形状为 (3, 3) 的浮点数张量 input，初始值为 0.0
        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        # 创建一个形状为 (3, 2) 的整数张量 indices，表示 scatter 操作的索引
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        # 创建一个形状为 (3, 2) 的浮点数张量 values，表示 scatter 操作的值
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        # 调用外部方法 run_test，以 ScatterModel 实例和输入参数进行测试
        self.run_test(ScatterModel(), input_args=(input, indices, values))

        # 创建另一个形状为 (3, 3) 的浮点数张量 input，初始值为 0.0
        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        # 创建另一个形状为 (3, 2) 的整数张量 indices，表示 scatter 操作的索引
        indices = torch.tensor([[1, 0], [0, 2], [0, 1]], dtype=torch.int64)
        # 创建另一个形状为 (3, 2) 的浮点数张量 values，表示 scatter 操作的值
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        # 调用外部方法 run_test，以 ScatterModel 实例和输入参数进行测试
        self.run_test(ScatterModel(), (input, indices, values))

        # 创建一个形状为 (3, 4, 5, 6) 的全零张量 input
        input = torch.zeros(3, 4, 5, 6)
        # 创建一个形状为 (3, 2) 的整数张量 indices，表示 scatter 操作的索引
        indices = torch.tensor([[1, 0], [0, 2], [0, 1]], dtype=torch.int64)
        # 将 indices 重塑为 (3, 2, 1, 1)，并扩展为 (3, 2, 5, 6)
        indices = indices.view(3, 2, 1, 1).expand(3, 2, 5, 6)
        # 创建一个形状为 (3, 2, 5, 6) 的浮点数张量 values，表示 scatter 操作的值
        values = torch.arange(3 * 2 * 5 * 6, dtype=torch.float32).view(3, 2, 5, 6)
        # 调用外部方法 run_test，以 ScatterModel 实例和输入参数进行测试
        self.run_test(ScatterModel(), (input, indices, values))

        # 创建一个形状为 (3, 4, 2) 的全零张量 input
        input = torch.zeros(3, 4, 2)
        # 创建一个形状为 (3, 2, 2) 的整数张量 indices，表示 scatter 操作的索引
        indices = torch.tensor([[[1, 0], [0, 2]], [[1, 1], [0, 1]], [[2, 1], [2, 2]]])
        # 创建一个形状为 (3, 2, 2) 的浮点数张量 values，表示 scatter 操作的值
        values = torch.arange(3 * 2 * 2, dtype=torch.float32).view(3, 2, 2)
        # 调用外部方法 run_test，以 ScatterModel 实例和输入参数进行测试
        self.run_test(ScatterModel(), (input, indices, values))

    # 根据支持的最小操作集版本跳过测试，要求最小支持版本为 9
    @skipIfUnsupportedMinOpsetVersion(9)
    # 定义一个名为 test_scatter_add 的测试方法
    def test_scatter_add(self):
        # 定义一个内嵌的 ScatterModel 类，继承自 torch.nn.Module
        class ScatterModel(torch.nn.Module):
            # 定义模型的前向传播方法，接受 input, indices, values 三个参数，执行 scatter_add 操作后返回结果
            def forward(self, input, indices, values):
                return input.scatter_add(1, indices, values)

        # 创建一个形状为 (3, 3) 的浮点数张量 input，初始值为 0.0
        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        # 创建一个形状为 (3, 2) 的整数张量 indices，表示 scatter_add 操作的索引
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        # 创建一个形状为 (3, 2) 的浮点数张量 values，表示 scatter_add 操作的值
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        # 调用外部方法 run_test，以 ScatterModel 实例和输入参数进行测试
        self.run_test(ScatterModel(), input_args=(input, indices, values))

        # 定义一个 Torch 脚本函数 scatter_sum，用于执行 scatter_add 操作
        @torch.jit.script
        def scatter_sum(src: Tensor, index: Tensor):
            # 获取 src 的尺寸
            size = src.size()
            # 创建一个与 src 相同尺寸的全零张量 out，数据类型与 src 相同
            out = torch.zeros(size, dtype=src.dtype)
            # 执行 scatter_add 操作，将 src 的值按 index 的索引累加到 out 上，并返回 out
            return out.scatter_add_(1, index, src)

        # 定义一个内嵌的 ScatterModel 类，继承自 torch.nn.Module
        class ScatterModel(torch.nn.Module):
            # 定义模型的前向传播方法，接受 src, index 两个参数，调用 scatter_sum 执行 scatter_add 操作后返回结果
            def forward(self, src, index):
                return scatter_sum(src, index)

        # 创建一个形状为 (3, 2) 的随机数张量 src
        src = torch.rand(3, 2)
        # 创建一个形状为 (3, 2) 的整数张量 index，表示 scatter_add 操作的索引
        index = torch.tensor([[0, 1], [0, 1], [0, 1]], dtype=torch.int64)
        # 调用外部方法 run_test，以 ScatterModel 实例和输入参数进行测试
        self.run_test(ScatterModel(), (src, index))

    # 根据支持的最小操作集版本跳过测试，要求最小支持版本为 16
    @skipIfUnsupportedMinOpsetVersion(16)
    @skipIfUnsupportedMinOpsetVersion(16)
    # 装饰器：跳过不支持的最小运算集版本号为16的测试用例
    def test_scatter_add_different_size_index_src(self):
        # 定义一个名为 ScatterModel 的内部类，继承自 torch.nn.Module
        class ScatterModel(torch.nn.Module):
            # 前向传播方法，接受 input、indices、src 三个参数
            def forward(self, input, indices, src):
                # 对 input 执行 scatter_add 操作，按照 0 维度（行）进行聚合，用 indices 和 src 进行加和
                return input.scatter_add(0, indices, src)

        # 创建一个全为1的 2x5 张量 src
        src = torch.ones((2, 5))
        # 创建一个全为0的 3x5 张量 input，数据类型与 src 相同
        input = torch.zeros(3, 5, dtype=src.dtype)
        # 创建一个 1x5 的索引张量 indices，其中包含 [0, 1, 2, 0, 0]
        indices = torch.tensor([[0, 1, 2, 0, 0]])
        # 运行测试，使用 ScatterModel 类的实例，传入 input、indices、src 作为参数
        self.run_test(ScatterModel(), input_args=(input, indices, src))

    @common_utils.parametrize(
        "src, indices",
        [
            common_utils.subtest(
                [torch.ones((1, 5)), torch.tensor([[0, 1, 2, 0, 0]])],
                name="src_indices_dynamic_combination1",
            ),
            common_utils.subtest(
                [torch.ones((2, 5)), torch.tensor([[0, 1, 2, 0, 0], [1, 0, 2, 1, 2]])],
                name="src_indices_dynamic_combination2",
            ),
            common_utils.subtest(
                [torch.ones((3, 5)), torch.tensor([[0, 1, 2, 0, 0], [1, 0, 2, 1, 2]])],
                name="src_indices_dynamic_combination3",
            ),
            common_utils.subtest(
                [torch.ones((3, 5)), torch.tensor([[0, 1, 2, 0], [1, 0, 2, 1]])],
                name="src_indices_dynamic_combination4",
            ),
        ],
    )
    @skipIfUnsupportedMinOpsetVersion(16)
    # 装饰器：跳过不支持的最小运算集版本号为16的测试用例
    def test_scatter_add_dynamic_index(self, src, indices):
        # 定义一个名为 ScatterModel 的内部类，继承自 torch.nn.Module
        class ScatterModel(torch.nn.Module):
            # 前向传播方法，接受 input、indices、src 三个参数
            def forward(self, input, indices, src):
                # 对 input 执行 scatter_add 操作，按照 0 维度（行）进行聚合，用 indices 和 src 进行加和
                return input.scatter_add(0, indices, src)

        # 创建一个全为0的 3x5 张量 input，数据类型与 src 相同
        input = torch.zeros(3, 5, dtype=src.dtype)
        # 运行测试，使用 ScatterModel 类的实例，传入 input、indices、src 作为参数
        self.run_test(
            ScatterModel(),
            input_args=(input, indices, src),
            input_names=["input", "indices", "src"],
            dynamic_axes={"indices": {0: "a", 1: "b"}, "src": {0: "c", 1: "d"}},
        )

    @skipIfUnsupportedMinOpsetVersion(16)
    # 装饰器：跳过不支持的最小运算集版本号为16的测试用例
    def test_scatter_reduce(self):
        # 定义一个内嵌的 PyTorch 模型类 Model，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            # 构造函数，初始化模型
            def __init__(self):
                super().__init__()

            # 前向传播函数，接受参数 x, index, input
            def forward(self, x, index, input):
                # 使用 input 的 scatter_reduce 方法，对输入进行 reduce 操作（最大值）
                y_max = input.scatter_reduce(0, index, x, reduce="amax")
                # 使用 input 的 scatter_reduce 方法，对输入进行 reduce 操作（求和）
                y_sum = input.scatter_reduce(0, index, x, reduce="sum")
                # 使用 input 的 scatter_reduce 方法，对输入进行 reduce 操作（最小值）
                y_min = input.scatter_reduce(0, index, x, reduce="amin")
                # 使用 input 的 scatter_reduce 方法，对输入进行 reduce 操作（乘积）
                y_mul = input.scatter_reduce(0, index, x, reduce="prod")
                # 返回四种 reduce 操作的结果
                return y_max, y_sum, y_min, y_mul

        # 创建 Model 类的实例 model
        model = Model()
        # 将模型设置为评估模式
        model.eval()

        # 创建源张量 src，包含浮点数 1.0 到 6.0
        src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        # 创建索引张量 index，指定了每个元素的分组
        index = torch.tensor([0, 1, 0, 1, 2, 1])
        # 创建输入张量 input，包含浮点数 1.0, 2.0, 3.0, 8.0
        input = torch.tensor([1.0, 2.0, 3.0, 8.0])

        # 运行自定义的测试函数 run_test，传入模型和参数元组
        self.run_test(model, (src, index, input))

    @skipIfUnsupportedMinOpsetVersion(16)
    def test_scatter_reduce_self_rank_zero(self):
        # 定义一个内嵌的 PyTorch 模型类 Model，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            # 构造函数，初始化模型
            def __init__(self):
                super().__init__()

            # 前向传播函数，接受参数 x, index, input
            def forward(self, x, index, input):
                # 使用 input 的 scatter_reduce 方法，对输入进行 reduce 操作（最大值）
                y_max = input.scatter_reduce(0, index, x, reduce="amax")
                # 使用 input 的 scatter_reduce 方法，对输入进行 reduce 操作（求和）
                y_sum = input.scatter_reduce(0, index, x, reduce="sum")
                # 使用 input 的 scatter_reduce 方法，对输入进行 reduce 操作（最小值）
                y_min = input.scatter_reduce(0, index, x, reduce="amin")
                # 使用 input 的 scatter_reduce 方法，对输入进行 reduce 操作（乘积）
                y_mul = input.scatter_reduce(0, index, x, reduce="prod")
                # 返回四种 reduce 操作的结果
                return y_max, y_sum, y_min, y_mul

        # 创建 Model 类的实例 model
        model = Model()
        # 将模型设置为评估模式
        model.eval()

        # 创建空的张量 empty_tensor
        empty_tensor = torch.tensor([])
        # 创建空的索引张量 empty_idx，数据类型为 torch.int64
        empty_idx = torch.tensor([], dtype=torch.int64)

        # 运行自定义的测试函数 run_test，传入模型和参数元组
        self.run_test(model, (empty_tensor, empty_idx, empty_tensor))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_bucketize(self):
        # 定义一个内嵌的 PyTorch 模型类 BucketModel，继承自 torch.nn.Module
        class BucketModel(torch.nn.Module):
            # 前向传播函数，接受输入 input 和边界 boundaries
            def forward(self, input, boundaries):
                # 使用 torch.bucketize 函数进行分桶操作，并返回结果
                return torch.bucketize(input, boundaries), torch.bucketize(
                    input, boundaries, right=True
                )

        # 创建输入张量 input，包含两个子张量，每个子张量包含整数
        input = torch.tensor([[2, 5, 10], [6, 8, 3]])
        # 创建边界张量 boundaries，指定分桶的边界
        boundaries = torch.tensor([1, 5, 7, 8, 10])

        # 运行自定义的测试函数 run_test，传入 BucketModel 类的实例和参数元组
        self.run_test(BucketModel(), (input, boundaries))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_one_hot(self):
        # 定义一个内嵌的 PyTorch 模型类 OneHot，继承自 torch.nn.Module
        class OneHot(torch.nn.Module):
            # 构造函数，初始化模型并指定类别数 num_classes
            def __init__(self, num_classes):
                super().__init__()
                self.num_classes = num_classes

            # 前向传播函数，接受输入张量 x，返回 one-hot 编码结果
            def forward(self, x):
                return torch.nn.functional.one_hot(x, self.num_classes)

        # 创建整数张量 x，包含从 0 到 9 的整数
        x = torch.arange(10)
        # 运行自定义的测试函数 run_test，传入 OneHot 类的实例和参数元组
        self.run_test(OneHot(15), (x))

        # 定义一个内嵌的 PyTorch 模型类 OneHot，继承自 torch.nn.Module
        class OneHot(torch.nn.Module):
            # 前向传播函数，接受输入张量 x 和类别数 num_classes
            def forward(self, x, num_classes):
                # 将类别数转换为 torch.int32 类型
                num_classes = num_classes.to(torch.int32)
                # 返回 x 的 one-hot 编码结果
                return torch.nn.functional.one_hot(x, num_classes[0])

        # 创建整数张量 x，包含从 0 到 9 的整数
        x = torch.arange(10)
        # 创建类别数张量 num_classes，包含数值 15，数据类型为 torch.float32
        num_classes = 15 * torch.ones(1)

        # 运行自定义的测试函数 run_test，传入 OneHot 类的实例和参数元组
        self.run_test(OneHot(), (x, num_classes))

    @skipIfUnsupportedMinOpsetVersion(9)
    # 定义一个测试方法 test_gather，用于测试 gather 操作的功能
    def test_gather(self):
        # 定义一个内部的 PyTorch 模块 GatherModel
        class GatherModel(torch.nn.Module):
            # 定义模块的前向传播方法
            def forward(self, input, indices):
                return input.gather(1, indices)

        # 创建一个输入张量 input，包含三个子列表
        input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        # 创建一个索引张量 indices，包含三个子列表
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        # 运行测试函数 run_test，测试 GatherModel 的输出
        self.run_test(GatherModel(), input_args=(input, indices))

    @skipScriptTest()  # 脚本化测试跳过注解：无法实例化 nn 模块
    # 定义一个测试方法 test_gather_constant_fold，并跳过测试
    def test_gather_constant_fold(self):
        # 定义一个内部的 PyTorch 模块 GatherModule
        class GatherModule(torch.nn.Module):
            # 模块的初始化方法
            def __init__(self):
                super().__init__()
                # 注册一个名为 weight 的缓冲区，初始化为全 1 的张量
                self.register_buffer("weight", torch.ones(5))
                # 使用 torch.nn.Embedding 转换为 ONNX::Gather
                # 常量输入将触发常量折叠
                # 这种模式在 transformer 模型中的常量掩码输入中很常见
                self.embed = torch.nn.Embedding(8, 3)

            # 模块的前向传播方法
            def forward(self, x):
                # 计算 weight 张量的形状维度
                shape = self.weight.shape[0]
                # 计算 m 的值
                m = 5 - shape
                # 创建一个全 1 的长整型张量 y
                y = torch.ones(1, 4, dtype=torch.long)
                # 返回 x 的最小值受限于 m，并且使用 embed 方法处理 y
                return x.clamp(min=m), self.embed(y)

        # 创建一个随机张量 x
        x = torch.randn(1)
        # 运行测试函数 run_test，测试 GatherModule 的输出
        self.run_test(GatherModule(), (x,))

        # 定义一个内部的 PyTorch 模块 GatherModule
        class GatherModule(torch.nn.Module):
            # 模块的初始化方法
            def __init__(self):
                super().__init__()
                # 注册一个名为 weight 的缓冲区，初始化为全 1 的长度为 2 的张量
                self.register_buffer("weight", torch.ones(2))

            # 模块的前向传播方法
            def forward(self, x):
                # 计算 weight 张量的形状维度
                shape = self.weight.shape[0]
                # 定义 pad 为列表 [1, shape, shape, shape]
                pad = [1, shape, shape, shape]
                # 创建一个 2D 的零填充层 zero_pad
                zero_pad = torch.nn.ZeroPad2d(pad)
                # 返回 zero_pad 处理后的 x
                return zero_pad(x)

        # 创建一个随机张量 x，包含三个子列表
        x = torch.randn(1, 3, 2)
        # 运行测试函数 run_test，测试 GatherModule 的输出
        self.run_test(GatherModule(), (x,))

        # 定义一个内部的 PyTorch 模块 GatherModule
        class GatherModule(torch.nn.Module):
            # 模块的初始化方法
            def __init__(self):
                super().__init__()
                # 注册一个名为 rb 的缓冲区，初始化为大小为 [1, 1, 3, 1, 1] 的随机张量
                self.register_buffer("rb", torch.randn(1, 1, 3, 1, 1))

            # 模块的前向传播方法
            def forward(self, x):
                # 将 x 加上缓冲区 rb 的第一个元素
                x += self.rb[0]
                # 返回更新后的 x
                return x

        # 创建一个随机张量 x，包含三个子列表
        x = torch.randn(1, 3, 224, 224)
        # 运行测试函数 run_test，测试 GatherModule 的输出
        self.run_test(
            GatherModule(),
            (x,),
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 1: "class", 2: "height", 3: "width"},
            },
            input_names=["input"],
            output_names=["output"],
        )

    @skipIfUnsupportedOpsetVersion([13])  # 跳过不支持的 Opset 版本测试
    @skipIfUnsupportedMinOpsetVersion(9)  # 跳过不支持的最小 Opset 版本测试
    # 定义测试方法 test_expand，用于测试 Torch 模型的数据扩展功能
    def test_expand(self):
        # 定义一个简单的 Torch 模型 ExpandModel，实现数据扩展操作
        class ExpandModel(torch.nn.Module):
            def forward(self, input):
                return input.expand(2, 3, -1)

        # 创建输入张量 input，形状为 (2, 1, 4)，用于测试 ExpandModel
        input = torch.randn(2, 1, 4)
        # 运行测试，使用自定义的 run_test 方法，传入 ExpandModel 和 input 作为参数
        self.run_test(ExpandModel(), input_args=(input))

        # 定义另一个 Torch 模型 ExpandInferDimModel，演示推断维度的数据扩展操作
        class ExpandInferDimModel(torch.nn.Module):
            def forward(self, input):
                return input.expand(-1, input.size(0))

        # 创建输入张量 input，形状为 (3, 1)，用于测试 ExpandInferDimModel
        input = torch.randn(3, 1)
        # 运行测试，使用自定义的 run_test 方法，传入 ExpandInferDimModel 和 input 作为参数
        self.run_test(ExpandInferDimModel(), input_args=(input))

        # 定义另一个 Torch 模型 ExpandTensorSizeModel，接受额外的 size 参数进行数据扩展
        class ExpandTensorSizeModel(torch.nn.Module):
            def forward(self, input, size):
                return input.expand(size)

        # 创建输入张量 input，形状为 (3,)，用于测试 ExpandTensorSizeModel
        input = torch.randn(
            3,
        )
        # 创建大小为 -1 的 size 张量，用于测试 ExpandTensorSizeModel
        size = torch.tensor(-1)
        # 运行测试，使用自定义的 run_test 方法，传入 ExpandTensorSizeModel、input 和 size 作为参数
        self.run_test(ExpandTensorSizeModel(), input_args=(input, size))

    # 跳过不支持的最小操作集版本，测试动态维度下的数据扩展操作
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dynamic_expand_as(self):
        # 定义一个 Torch 模型 Model，实现动态维度下的数据扩展操作
        class Model(torch.nn.Module):
            def forward(self, x):
                # 将输入张量 x 的一部分维度扩展为相同大小的零张量
                x[:, x.size(0) :] = 0
                return x

        # 创建输入张量 x，形状为 (2, 5)，用于测试 Model
        x = torch.ones(2, 5)
        # 创建形状为 (3, 4) 的 x2 张量，用于测试 Model
        x2 = torch.randn(3, 4)
        # 运行测试，使用自定义的 run_test 方法，传入 Model 和输入参数
        self.run_test(
            Model(),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
            additional_test_inputs=[x2],
        )

        # 定义另一个 Torch 模型 Model，实现动态维度下的数据扩展操作，扩展为指定的张量
        class Model(torch.nn.Module):
            def forward(self, x):
                # 将输入张量 x 的一部分维度扩展为具有相同维度的固定值张量
                x[:, x.size(0) :] = torch.tensor([1, 2, 3])
                return x

        # 创建输入张量 x，形状为 (2, 5, 3)，用于测试 Model
        x = torch.ones(2, 5, 3)
        # 创建形状为 (3, 4, 3) 的 x2 张量，用于测试 Model
        x2 = torch.randn(3, 4, 3)
        # 运行测试，使用自定义的 run_test 方法，传入 Model 和输入参数
        self.run_test(
            Model(),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2]},
            additional_test_inputs=[x2],
        )

        # 定义另一个 Torch 模型 Model，实现数据扩展操作，根据 aa 张量的形状进行扩展
        class Model(torch.nn.Module):
            def forward(self, x):
                aa = torch.tensor([[0], [1], [2]])
                return aa.expand_as(x)

        # 创建输入张量 x，形状为 (3, 2)，用于测试 Model
        x = torch.ones(3, 2)
        # 创建形状为 (3, 5) 的 x2 张量，用于测试 Model
        x2 = torch.randn(3, 5)
        # 运行测试，使用自定义的 run_test 方法，传入 Model 和输入参数
        self.run_test(
            Model(),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
            additional_test_inputs=[x2],
        )

    # 定义测试方法 test_multinomial，测试 Torch 中的多项分布采样功能
    def test_multinomial(self):
        # 定义 Torch 模型 Multinomial，实现带替换的多项分布采样
        class Multinomial(torch.nn.Module):
            def forward(self, weight):
                return torch.multinomial(weight, 3, replacement=True)

        # 定义 Torch 模型 MultinomialNoReplacement，实现不带替换的多项分布采样
        class MultinomialNoReplacement(torch.nn.Module):
            def forward(self, weight):
                return torch.multinomial(weight, 1)

        # 创建权重张量 weight，形状为 (2, 4)，用于测试 Multinomial 和 MultinomialNoReplacement
        weight = torch.tensor([[0, 10, 0, 0], [0, 0, 100, 0]], dtype=torch.float)
        # 运行测试，使用自定义的 run_test 方法，传入 Multinomial 和权重张量作为参数
        self.run_test(Multinomial(), (weight,))
        # 运行测试，使用自定义的 run_test 方法，传入 MultinomialNoReplacement 和权重张量作为参数
        self.run_test(MultinomialNoReplacement(), (weight,))
    # 定义一个测试函数，用于测试特定的 reduce 操作 op
    def _test_reduced_ops(self, op):
        # 定义一个内部类 ReducedOpModule，继承自 torch.nn.Module，用于测试特定的 reduce 操作
        class ReducedOpModule(torch.nn.Module):
            # 定义前向传播函数，对输入进行 reduce 操作
            def forward(self, input):
                return op(input, dim=-1)

        # 如果 op 不是 torch.mean，则执行以下测试
        if op != torch.mean:  # torch.mean 只支持 float 类型
            # 创建一个 dtype 为 torch.uint8 的 4x4 随机整数张量 x，并运行测试
            x = torch.randint(10, (4, 4), dtype=torch.uint8)
            self.run_test(ReducedOpModule(), x)

            # 创建一个 dtype 为 torch.int8 的 4x4 随机整数张量 x，并运行测试
            x = torch.randint(10, (4, 4), dtype=torch.int8)
            self.run_test(ReducedOpModule(), x)

            # 创建一个 dtype 为 torch.int16 的 4x4 随机整数张量 x，并运行测试
            x = torch.randint(10, (4, 4), dtype=torch.int16)
            self.run_test(ReducedOpModule(), x)

            # 创建一个 dtype 为 torch.int32 的 4x4 随机整数张量 x，并运行测试
            x = torch.randint(10, (4, 4), dtype=torch.int32)
            self.run_test(ReducedOpModule(), x)

            # 创建一个 dtype 为 torch.int64 的 4x4 随机整数张量 x，并运行测试
            x = torch.randint(10, (4, 4), dtype=torch.int64)
            self.run_test(ReducedOpModule(), x)

        # 如果 op 不是 torch.prod 和 torch.mean，则执行以下测试
        # torch.mean 只支持 float 类型，ORT 不支持 double ReduceProd
        if op != torch.prod and op != torch.mean:
            # 创建一个 dtype 为 torch.double 的 4x5 随机张量 x，并运行测试
            x = torch.randn(4, 5, dtype=torch.double)
            self.run_test(ReducedOpModule(), x)

        # 如果 op 不是 torch.prod，则执行以下测试
        # torch.prod 不支持 dtype 为 torch.half 的张量
        if op != torch.prod:
            # 创建一个 dtype 为 torch.half 的 4x4 随机张量 x，并运行测试
            x = torch.randn(4, 4, dtype=torch.half)
            self.run_test(ReducedOpModule(), x)

        # 创建一个 dtype 为 torch.float 的 4x5 随机张量 x，并运行测试
        x = torch.randn(4, 5, dtype=torch.float)
        self.run_test(ReducedOpModule(), x)

    # 测试 torch.sum 操作的函数
    def test_reduced_sum(self):
        return self._test_reduced_ops(op=torch.sum)

    # 测试 torch.mean 操作的函数
    def test_reduced_mean(self):
        return self._test_reduced_ops(op=torch.mean)

    # 测试 torch.prod 操作的函数
    def test_reduced_prod(self):
        return self._test_reduced_ops(op=torch.prod)

    # 测试不同数据类型下的 reduce sum 操作
    def test_reduced_sum_dtypes(self):
        # 定义一个不带维度参数的模型 NoDimModel
        class NoDimModel(torch.nn.Module):
            # 定义前向传播函数，对输入进行 reduce sum 操作，dtype 为 torch.float
            def forward(self, input):
                return input.sum(dtype=torch.float)

        # 定义一个带维度参数的模型 DimModel
        class DimModel(torch.nn.Module):
            # 定义前向传播函数，对输入在指定维度进行 reduce sum 操作，dtype 为 torch.float
            def forward(self, input):
                return input.sum(dim=-1, dtype=torch.float)

        # 创建一个 dtype 为 torch.half 的 4x4 随机张量 input，并分别运行两个模型的测试
        input = torch.randn((4, 4), dtype=torch.half)
        self.run_test(NoDimModel(), input)
        self.run_test(DimModel(), input)

    # 测试 reduce min 和 reduce max 操作的函数
    def test_reduced_min_max(self):
        # 定义一个 ReducedMinMaxModule 模型
        class ReducedMinMaxModule(torch.nn.Module):
            # 定义前向传播函数，分别对输入进行 reduce min 和 reduce max 操作
            def forward(self, input):
                return torch.min(input, dim=-1)[0], torch.max(input, dim=0)[0]

        # 创建一个 dtype 为 torch.int32 的 4x4 随机整数张量 x，并运行测试
        x = torch.randint(10, (4, 4), dtype=torch.int32)
        self.run_test(ReducedMinMaxModule(), x)

        # 创建一个 dtype 为 torch.int64 的 4x4 随机整数张量 x，并运行测试
        x = torch.randint(10, (4, 4), dtype=torch.int64)
        self.run_test(ReducedMinMaxModule(), x)

        # 创建一个 dtype 为 torch.float 的 4x5 随机张量 x，并运行测试
        x = torch.randn(4, 5, dtype=torch.float)
        self.run_test(ReducedMinMaxModule(), x)

    # 测试 reduce logsumexp 操作的函数
    def test_reduce_log_sum_exp(self):
        # 定义一个 ReduceLogSumExpModel 模型
        class ReduceLogSumExpModel(torch.nn.Module):
            # 定义前向传播函数，对输入进行 reduce logsumexp 操作
            def forward(self, input):
                # 在 dim=0 上进行 logsumexp 操作，并在 dim=(0, 1) 上进行 logsumexp 操作，最后返回两者之和
                a = torch.logsumexp(input, dim=0)
                b = torch.logsumexp(input, dim=(0, 1))
                return a + b

        # 创建一个 4x4 随机张量 input，并运行测试
        x = torch.randn(4, 4, requires_grad=True)
        self.run_test(ReduceLogSumExpModel(), x)
    # 定义测试 softmax 函数的方法
    def test_softmax(self):
        # 对指定范围的维度进行迭代
        for i in range(-4, 3):
            # 创建 softmax 模型，指定维度为 i
            model = torch.nn.Softmax(dim=i)
            # 生成随机输入张量
            input = torch.randn(3, 4, 5, 6)
            # 运行测试函数，评估模型在给定输入上的表现
            self.run_test(model, input)

            # 定义一个未知维度的 softmax 模型类
            class SoftmaxUnknownRank(torch.nn.Module):
                def __init__(self, i):
                    super().__init__()
                    self.softmax = torch.nn.Softmax(dim=i)

                # 实现前向传播，对输入张量重新形状后应用 softmax
                def forward(self, x):
                    return self.softmax(x.reshape(3, 4, 5, 6))

            # 使用 JIT 编译并创建 SoftmaxUnknownRank 类的模型实例
            model = torch.jit.script(SoftmaxUnknownRank(i))
            # 运行测试函数，评估 JIT 编译后的模型在给定输入上的表现
            self.run_test(model, input)

    # 定义测试 softmax 在处理大数值时的情况的方法
    def test_softmax_large_values(self):
        # 创建输入张量，包含极大负数、极大正数和一些其他数值
        input = torch.tensor(
            [[-1e12, -1e12, -1e12], [1e12, 0.0, -5.0], [3.0, 4.0, 5.0]]
        )
        # 对指定范围的维度进行迭代
        for i in range(-2, 1):
            # 创建 softmax 模型，指定维度为 i
            model = torch.nn.Softmax(dim=i)
            # 运行测试函数，评估模型在给定输入上的表现
            self.run_test(model, input)

            # 定义一个未知维度的 softmax 模型类
            class SoftmaxUnknownRank(torch.nn.Module):
                def __init__(self, i):
                    super().__init__()
                    self.softmax = torch.nn.Softmax(dim=i)

                # 实现前向传播，对输入张量重新形状后应用 softmax
                def forward(self, x):
                    return self.softmax(x.reshape(3, 3))

            # 使用 JIT 编译并创建 SoftmaxUnknownRank 类的模型实例
            model = torch.jit.script(SoftmaxUnknownRank(i))
            # 运行测试函数，评估 JIT 编译后的模型在给定输入上的表现
            self.run_test(model, input)

    # 定义测试 logsoftmax 函数的方法
    def test_logsoftmax(self):
        # 对指定范围的维度进行迭代
        for i in range(7)[2:]:
            # 创建 logsoftmax 模型，指定维度为 i-1
            model = torch.nn.LogSoftmax(dim=i - 1)
            # 创建全为 1 的输入张量，形状由维度 i 决定
            dims = [2] * (i - 2) + [3, 4]
            input = torch.ones(*dims, requires_grad=True)
            # 运行测试函数，评估模型在给定输入上的表现
            self.run_test(model, input)

    # 定义测试 logsoftmax 在处理不同维度时的方法
    def test_logsoftmax_dim(self):
        # 对指定范围的维度进行迭代
        for i in range(-4, 3):
            # 创建 logsoftmax 模型，指定维度为 i
            model = torch.nn.LogSoftmax(dim=i)
            # 生成随机输入张量
            input = torch.randn(3, 4, 5, 6)
            # 运行测试函数，评估模型在给定输入上的表现
            self.run_test(model, input)

    # 定义测试 logsoftmax 在处理数据类型时的方法
    def test_logsoftmax_dtype(self):
        # 定义一个模型类，实现 log_softmax 的功能，指定维度为 1 和数据类型为 float64
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.log_softmax(x, dim=1, dtype=torch.float64)

        # 创建一个随机输入张量，形状为 (3, 4, 5)，并标记为需要梯度计算
        x = torch.randn(3, 4, 5, requires_grad=True)
        # 运行测试函数，评估模型在给定输入上的表现
        self.run_test(Model(), x)

    # 定义测试 softplus 函数的方法
    def test_softplus(self):
        # 定义一个类，实现 softplus 的功能，无额外参数
        class BetaOneModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.softplus(x)

        # 创建一个随机输入张量，形状为 (3, 4, 5)，并标记为需要梯度计算
        x = torch.randn(3, 4, 5, requires_grad=True)
        # 运行测试函数，评估模型在给定输入上的表现
        self.run_test(BetaOneModel(), x)

        # 定义一个类，实现 softplus 的功能，指定额外参数 beta 为 2
        class BetaModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.softplus(x, beta=2)

        # 创建一个随机输入张量，形状为 (3, 4, 5)，并标记为需要梯度计算
        x = torch.randn(3, 4, 5, requires_grad=True)
        # 运行测试函数，评估模型在给定输入上的表现
        self.run_test(BetaModel(), x)

        # 定义一个类，实现 softplus 的功能，指定额外参数 beta 为 1.7
        class BetaFloatModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.softplus(x, beta=1.7)

        # 创建一个随机输入张量，形状为 (3, 4, 5)，并标记为需要梯度计算
        x = torch.randn(3, 4, 5, requires_grad=True)
        # 运行测试函数，评估模型在给定输入上的表现
        self.run_test(BetaFloatModel(), x)

    # 标记当前测试用例需要支持最小操作集版本为 9，用于跳过不支持该版本的测试
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_no_hidden(self):
        # 定义一个简单的 LSTM 模型类，没有隐藏层
        class LSTMModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个 LSTM 层，输入维度为16，隐藏层维度为16
                self.rnn = torch.nn.LSTM(input_size=16, hidden_size=16)

            def forward(self, x):
                # 前向传播函数，直接返回 LSTM 层的输出
                return self.rnn(x)

        # 生成一个大小为(10, 16, 16)的随机输入张量
        input = torch.randn((10, 16, 16))
        # 运行测试函数，传入上面定义的 LSTMModel 实例和输入张量
        self.run_test(LSTMModel(), (input,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_proj_no_hidden(self):
        # 定义一个带有投影层但没有隐藏层的 LSTM 模型类
        class LSTMModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个带有投影层的 LSTM 层，输入维度为16，隐藏层维度为16，投影层维度为8
                self.rnn = torch.nn.LSTM(input_size=16, hidden_size=16, proj_size=8)

            def forward(self, x):
                # 前向传播函数，直接返回 LSTM 层的输出
                return self.rnn(x)

        # 生成一个大小为(10, 16, 16)的随机输入张量
        input = torch.randn((10, 16, 16))
        # 断言运行测试时会抛出 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            self.run_test(LSTMModel(), (input,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm(self):
        # 定义一个带有初始化隐藏状态的 LSTM 模型类
        class LSTMModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个 LSTM 层，输入维度为 RNN_INPUT_SIZE，隐藏层维度为 RNN_HIDDEN_SIZE，单向
                self.rnn = torch.nn.LSTM(
                    RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False
                )

            def forward(self, x, h0, c0):
                # 前向传播函数，传入输入张量 x 和初始化的隐藏状态 h0, c0
                return self.rnn(x, (h0, c0))

        # 生成一个大小为 (RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE) 的随机输入张量
        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        # 生成大小为 (1, BATCH_SIZE, RNN_HIDDEN_SIZE) 的随机隐藏状态 h0 和 c0
        h0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        c0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        # 运行测试函数，传入上面定义的 LSTMModel 实例、输入张量和隐藏状态
        self.run_test(LSTMModel(), (input, h0, c0))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_cell(self):
        # 定义一个带有偏置的 LSTMCell 模型类
        class LSTMCellModel(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                # 定义一个 LSTMCell，输入维度为 RNN_INPUT_SIZE，隐藏层维度为 RNN_HIDDEN_SIZE，是否使用偏置根据参数 bias 决定
                self.lstm_cell = torch.nn.LSTMCell(
                    RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, bias=bias
                )

            def forward(self, x, h0, c0):
                # 前向传播函数，传入输入张量 x 和初始化的隐藏状态 h0, c0
                return self.lstm_cell(x, (h0, c0))

        # 生成一个大小为 (BATCH_SIZE, RNN_INPUT_SIZE) 的随机输入张量
        input = torch.randn(BATCH_SIZE, RNN_INPUT_SIZE)
        # 生成大小为 (BATCH_SIZE, RNN_HIDDEN_SIZE) 的随机隐藏状态 h0 和 c0
        h0 = torch.randn(BATCH_SIZE, RNN_HIDDEN_SIZE)
        c0 = torch.randn(BATCH_SIZE, RNN_HIDDEN_SIZE)
        
        # 循环测试带有和不带有偏置的 LSTMCellModel
        for bias in [True, False]:
            self.run_test(LSTMCellModel(bias), (input, h0, c0))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_default_init_state(self):
        # 定义一个带有默认初始化状态的 LSTM 模型类
        class LSTMModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个 LSTM 层，输入维度为 RNN_INPUT_SIZE，隐藏层维度为 RNN_HIDDEN_SIZE，单向
                self.rnn = torch.nn.LSTM(
                    RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False
                )

            def forward(self, x):
                # 前向传播函数，传入输入张量 x
                return self.rnn(x)

        # 生成一个大小为 (RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE) 的随机输入张量
        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        # 运行测试函数，传入上面定义的 LSTMModel 实例和输入张量
        self.run_test(LSTMModel(), input)
    def test_lstm_fixed_batch_size(self):
        # 定义一个名为 test_lstm_fixed_batch_size 的测试方法
        class LSTMModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化 LSTM 模型，指定输入大小、隐藏层大小和层数等参数
                self.lstm = torch.nn.LSTM(
                    RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False
                )
                self.RNN_HIDDEN_SIZE = RNN_HIDDEN_SIZE

            def forward(self, input):
                # 获取输入数据的批量大小
                batch_size = input.size()[1]
                # 创建初始隐藏状态 h0 和细胞状态 c0，大小为 [1, batch_size, RNN_HIDDEN_SIZE]
                h0 = torch.ones([1, batch_size, self.RNN_HIDDEN_SIZE])
                c0 = torch.ones([1, batch_size, self.RNN_HIDDEN_SIZE])
                # 执行 LSTM 模型的前向传播，返回输出和最终状态
                return self.lstm(input, (h0, c0))

        # 生成随机输入数据，大小为 [RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE]
        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        # 使用相同批量大小的不同输入进行验证
        input2 = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        # 运行测试，传入 LSTMModel 实例、输入数据和额外的测试输入数据
        self.run_test(
            LSTMModel(), input, fixed_batch_size=True, additional_test_inputs=[input2]
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_post_fix_init_state(self):
        # 定义一个名为 test_lstm_post_fix_init_state 的测试方法，条件是 Opset 版本不低于 9
        class LSTMModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化 LSTM 模型，指定输入大小、隐藏层大小和层数等参数
                self.lstm = torch.nn.LSTM(
                    RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False
                )
                self.RNN_HIDDEN_SIZE = RNN_HIDDEN_SIZE

            def forward(self, input):
                # 获取输入数据的批量大小
                batch_size = input.size()[1]
                # 创建初始隐藏状态 h0 和细胞状态 c0，大小为 [1, batch_size, RNN_HIDDEN_SIZE]
                h0 = torch.ones([1, batch_size, self.RNN_HIDDEN_SIZE])
                c0 = torch.ones([1, batch_size, self.RNN_HIDDEN_SIZE])
                # 执行 LSTM 模型的前向传播，返回输出和最终状态
                return self.lstm(input, (h0, c0))

        # 创建 LSTMModel 实例
        model = LSTMModel()
        # 生成随机输入数据，大小为 [RNN_SEQUENCE_LENGTH, 1, RNN_INPUT_SIZE]
        input = torch.randn(RNN_SEQUENCE_LENGTH, 1, RNN_INPUT_SIZE)
        # 使用不同批量大小的输入进行验证
        input2 = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        # 运行测试，传入 LSTMModel 实例、输入数据、指定输入名称和动态轴信息以及额外的测试输入数据
        self.run_test(
            model,
            input,
            input_names=["input.1"],
            dynamic_axes={"input.1": {0: "seq", 1: "batch"}},
            additional_test_inputs=[input2],
        )
    def test_lstm_constant_folding(self):
        # 定义一个测试函数，测试 LSTM 网络在常量折叠时的行为
        class LstmNet(torch.nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, bidirectional):
                super().__init__()
                # 初始化一个 LSTM 层
                self.lstm = torch.nn.LSTM(
                    input_size, hidden_size, num_layers, bidirectional=bidirectional
                )

            def forward(self, input, initial_state: Tuple[Tensor, Tensor]):
                # LSTM 网络的前向传播
                return self.lstm(input, initial_state)

        def get_LstmNet_model_and_inputs(
            input_size, hidden_size, num_layers, batch_size, seq_len, bidirectional
        ):
            # 根据参数返回一个 LstmNet 模型和输入数据
            num_directions = 2 if bidirectional else 1
            model = LstmNet(input_size, hidden_size, num_layers, bidirectional)
            input = torch.randn(seq_len, batch_size, input_size)
            h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
            c0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
            return model, (input, (h0, c0))

        batch_size1 = 3
        # 创建第一个模型和输入数据，进行测试，开启常量折叠功能
        model1, input1 = get_LstmNet_model_and_inputs(7, 3, 2, batch_size1, 5, True)
        self.run_test(model1, input1, do_constant_folding=True)

        batch_size2 = 4
        # 创建第二个模型和输入数据，进行测试，开启常量折叠功能
        model2, input2 = get_LstmNet_model_and_inputs(5, 4, 3, batch_size2, 7, False)
        self.run_test(model2, input2, do_constant_folding=True)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_no_bias(self):
        # 定义一个测试函数，测试没有偏置的 LSTM 网络的行为
        class LstmNet(torch.nn.Module):
            def __init__(self, num_layers, bidirectional):
                super().__init__()
                # 初始化一个没有偏置的 LSTM 层
                self.lstm = torch.nn.LSTM(
                    RNN_INPUT_SIZE,
                    RNN_HIDDEN_SIZE,
                    num_layers,
                    bias=False,
                    bidirectional=bidirectional,
                )

            def forward(self, input, initial_state: Tuple[Tensor, Tensor]):
                # LSTM 网络的前向传播
                return self.lstm(input, initial_state)

        def get_LstmNet_model_and_inputs(num_layers, bidirectional):
            # 根据参数返回一个没有偏置的 LstmNet 模型和输入数据
            input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
            num_directions = 2 if bidirectional else 1
            model = LstmNet(num_layers, bidirectional)
            h0 = torch.randn(num_layers * num_directions, BATCH_SIZE, RNN_HIDDEN_SIZE)
            c0 = torch.randn(num_layers * num_directions, BATCH_SIZE, RNN_HIDDEN_SIZE)
            return model, (input, (h0, c0))

        num_layers = [1, 1, 2, 3]
        bidirectional = [True, False, True, False]
        # 根据不同的层数和方向创建多个模型和输入数据的组合
        models_and_inputs = [
            get_LstmNet_model_and_inputs(n, b)
            for n, b in zip(num_layers, bidirectional)
        ]
        for model, input in models_and_inputs:
            # 对每个模型和输入数据进行测试
            self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(9)
    # 定义一个测试方法，用于测试 LSTM 网络的序列处理功能
    def test_lstm_sequence(self):
        # 定义一个继承自 torch.nn.Module 的 LSTM 网络类
        class LstmNet(torch.nn.Module):
            # 网络初始化方法
            def __init__(self):
                super().__init__()
                # 第一个双向 LSTM 层，输入维度为8，输出维度也为8，batch_first=True 表示输入数据的第一个维度是 batch_size
                self.rnn1 = torch.nn.LSTM(8, 8, bidirectional=True, batch_first=True)
                # 第一个线性层，输入维度是16（因为双向 LSTM 输出维度相加），输出维度是8
                self.linear1 = torch.nn.Linear(8 * 2, 8)
                # 第二个双向 LSTM 层，输入维度是8，输出维度也是8，batch_first=True 同上
                self.rnn2 = torch.nn.LSTM(8, 8, bidirectional=True, batch_first=True)
                # 第二个线性层，输入维度是16，输出维度是8
                self.linear2 = torch.nn.Linear(8 * 2, 8)

            # 前向传播方法
            def forward(self, input):
                # 第一次 LSTM 计算，得到 rnn_output1 是 LSTM 层的输出，_ 表示 LSTM 的隐藏状态
                rnn_output1, _ = self.rnn1(input)
                # 经过第一个线性层计算得到 linear_output1
                linear_output1 = self.linear1(rnn_output1)
                # 第二次 LSTM 计算，得到 rnn_output2 是 LSTM 层的输出，_ 同样表示隐藏状态
                rnn_output2, _ = self.rnn2(linear_output1)
                # 经过第二个线性层计算得到 linear_output2，作为最终的输出
                linear_output2 = self.linear2(rnn_output2)
                return linear_output2

        # 创建一个输入张量，形状为 (1, 100, 8)，数据类型为 torch.float32，所有元素都为0
        input = torch.zeros((1, 100, 8), dtype=torch.float32)
        # 调用自定义的测试运行函数 run_test，测试 LstmNet 的前向传播功能
        self.run_test(
            LstmNet(),  # 创建 LstmNet 类的实例
            input,  # 输入数据
            input_names=["input"],  # 输入数据的名称
            output_names=["output"],  # 输出数据的名称
            dynamic_axes={  # 指定输入和输出数据的动态维度映射
                "input": {0: "batch_size", 1: "w", 2: "h"},  # input 的动态维度映射
                "output": {0: "batch_size", 1: "w", 2: "h"},  # output 的动态维度映射
            },
        )

    # 跳过脚本测试的装饰器，用于标记当前测试不执行
    @skipScriptTest()
    # 定义一个测试方法，用于测试不带偏置的循环神经网络（RNN）模型
    def test_rnn_no_bias(self):
        # 定义一个内部函数，用于创建RNN模型
        def make_model(layers, packed_sequence):
            # 根据 packed_sequence 参数确定是否按批次优先(batch_first=True)构建模型
            batch_first = True if packed_sequence == 2 else False
            # 创建一个不带偏置的 RNN 模型对象
            model = torch.nn.RNN(
                RNN_INPUT_SIZE,            # 输入数据的维度大小
                RNN_HIDDEN_SIZE,           # 隐藏层的单元数量
                layers,                    # RNN 层的数量
                bidirectional=False,       # 不使用双向RNN
                batch_first=batch_first,   # 按需设置批次优先或时间步优先
                bias=False,                # 不使用偏置
            )

            # 根据 packed_sequence 参数选择是否使用带有打包序列功能的 RNN 模型包装器
            if packed_sequence == 1:
                model = rnn_model_with_packed_sequence.RnnModelWithPackedSequence(
                    model, False
                )
            if packed_sequence == 2:
                model = rnn_model_with_packed_sequence.RnnModelWithPackedSequence(
                    model, True
                )
            return model

        # 定义一个内部函数，用于生成模型的输入数据
        def make_input(batch_size, layers, packed_sequence):
            # 根据 packed_sequence 参数确定是否按批次优先(batch_first=True)处理输入数据
            batch_first = True if packed_sequence == 2 else False
            # 生成随机长度的序列
            seq_lengths = np.random.randint(1, RNN_SEQUENCE_LENGTH + 1, size=batch_size)
            seq_lengths = sorted(map(int, seq_lengths), reverse=True)
            # 生成随机数据作为输入
            inputs = [torch.randn(l, RNN_INPUT_SIZE) for l in seq_lengths]
            # 对输入数据进行填充，确保它们具有相同的长度
            inputs = rnn_utils.pad_sequence(inputs, batch_first=batch_first)
            inputs = [inputs]

            # 生成初始隐藏状态 h0
            h0 = torch.randn(layers, batch_size, RNN_HIDDEN_SIZE)
            inputs.append(h0)
            # 如果 packed_sequence 不为 0，则添加序列长度信息作为输入的一部分
            if packed_sequence != 0:
                inputs.append(torch.IntTensor(seq_lengths))
            # 根据输入的数量，选择合适的输入格式
            if len(inputs) == 1:
                input = inputs[0]
            else:
                input = tuple(inputs)
            return input

        # 定义不同的 RNN 层数和 packed_sequence 类型的列表
        layers = [1, 3, 1, 3, 1, 3]
        packed_sequence = [0, 0, 1, 1, 2, 2]
        # 使用 make_model 函数创建一组 RNN 模型
        models = [make_model(l, p) for l, p in zip(layers, packed_sequence)]
        # 使用 make_input 函数创建一组输入数据
        inputs = [
            make_input(RNN_BATCH_SIZE, l, p) for l, p in zip(layers, packed_sequence)
        ]

        # 遍历模型和输入数据，运行测试方法 run_test
        for model, input in zip(models, inputs):
            self.run_test(model, input)
    def test_gru_no_bias(self):
        # 定义一个名为 test_gru_no_bias 的测试函数
        class GruNet(torch.nn.Module):
            # 定义 GruNet 类，继承自 torch.nn.Module
            def __init__(self, input_size, hidden_size, num_layers, bidirectional):
                super().__init__()
                # 调用父类构造函数初始化模型
                self.mygru = torch.nn.GRU(
                    input_size,
                    hidden_size,
                    num_layers,
                    bidirectional=bidirectional,
                    bias=False,
                )
                # 初始化一个不带偏置的 GRU 层

            def forward(self, input, initial_state):
                # 定义模型的前向传播方法
                out = self.mygru(input, initial_state)
                return out
                # 使用定义好的 GRU 层进行前向计算并返回结果

        def get_GruNet_model_and_inputs(
            input_size, hidden_size, num_layers, batch_size, seq_len, bidirectional
        ):
            # 定义一个辅助函数，用于生成 GruNet 模型和输入数据
            num_directions = 2 if bidirectional else 1
            # 计算双向与单向 GRU 的方向数
            model = GruNet(input_size, hidden_size, num_layers, bidirectional)
            # 创建一个 GruNet 模型实例
            input = torch.randn(seq_len, batch_size, input_size)
            # 生成随机输入数据
            h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
            # 生成随机初始状态数据
            return model, (input, h0)
            # 返回模型和输入数据的元组

        input_size = [7, 5]
        hidden_size = [3, 4]
        num_layers = [2, 3]
        batch_size = [3, 4]
        seq_len = [5, 7]
        bidirectional = [True, False]
        # 定义不同参数组合的列表
        models_and_inputs = [
            get_GruNet_model_and_inputs(i, h, n, b, s, bi)
            for i, h, n, b, s, bi in zip(
                input_size, hidden_size, num_layers, batch_size, seq_len, bidirectional
            )
        ]
        # 使用 zip 将参数列表打包成元组列表
        for model, input in models_and_inputs:
            self.run_test(model, input, do_constant_folding=True)
            # 对每个模型和输入组合运行测试，并开启常量折叠优化

    def test_gru_constant_folding(self):
        # 定义一个名为 test_gru_constant_folding 的测试函数
        class GruNet(torch.nn.Module):
            # 定义 GruNet 类，继承自 torch.nn.Module
            def __init__(self, input_size, hidden_size, num_layers, bidirectional):
                super().__init__()
                # 调用父类构造函数初始化模型
                self.mygru = torch.nn.GRU(
                    input_size, hidden_size, num_layers, bidirectional=bidirectional
                )
                # 初始化一个带有可能的偏置的 GRU 层

            def forward(self, input, initial_state):
                # 定义模型的前向传播方法
                out = self.mygru(input, initial_state)
                return out
                # 使用定义好的 GRU 层进行前向计算并返回结果

        def get_GruNet_model_and_inputs(
            input_size, hidden_size, num_layers, batch_size, seq_len, bidirectional
        ):
            # 定义一个辅助函数，用于生成 GruNet 模型和输入数据
            num_directions = 2 if bidirectional else 1
            # 计算双向与单向 GRU 的方向数
            model = GruNet(input_size, hidden_size, num_layers, bidirectional)
            # 创建一个 GruNet 模型实例
            input = torch.randn(seq_len, batch_size, input_size)
            # 生成随机输入数据
            h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
            # 生成随机初始状态数据
            return model, (input, h0)
            # 返回模型和输入数据的元组

        batch_size1 = 3
        # 设置批量大小为 3
        model1, input1 = get_GruNet_model_and_inputs(7, 3, 2, batch_size1, 5, True)
        # 获取第一个 GruNet 模型和输入数据
        self.run_test(model1, input1, do_constant_folding=True)
        # 对第一个模型和输入数据运行测试，并开启常量折叠优化

        batch_size2 = 4
        # 设置批量大小为 4
        model2, input2 = get_GruNet_model_and_inputs(5, 4, 3, batch_size2, 7, False)
        # 获取第二个 GruNet 模型和输入数据
        self.run_test(model2, input2, do_constant_folding=True)
        # 对第二个模型和输入数据运行测试，并开启常量折叠优化

    @skipIfUnsupportedMinOpsetVersion(8)
    # 标记为跳过测试，如果运行的操作集版本低于 8
    def test_max_tensors(self):
        # 定义一个继承自torch.nn.Module的最大值模型类
        class MaxModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, input, other):
                # 返回输入张量和另一个张量的最大值
                return torch.max(input, other)

        # 创建一个MaxModel实例
        model = MaxModel()
        # 生成一个4x4的随机张量，并标记为需要梯度计算
        x = torch.randn(4, 4, requires_grad=True)
        # 生成一个4x1的随机张量，并标记为需要梯度计算
        y = torch.randn(4, 1, requires_grad=True)
        # 运行测试，传入模型实例和输入张量元组
        self.run_test(model, (x, y))

    def test_amax_amin(self):
        # 定义一个继承自torch.nn.Module的模型类
        class Model(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 返回张量在指定维度上的最大值和最小值
                return torch.amax(x, dim=0, keepdim=True), torch.amin(
                    x, dim=[0, 1], keepdim=False
                )

        # 创建一个Model实例
        model = Model()
        # 生成一个4x4的随机张量
        x = torch.randn(4, 4)
        # 运行测试，传入模型实例和输入张量
        self.run_test(model, x)

    def test_aminmax(self):
        # 定义一个继承自torch.nn.Module的模型类
        class Model(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 返回张量在指定维度上的最小值和最大值
                return torch.aminmax(x, dim=1, keepdim=True), torch.aminmax(
                    x, keepdim=False
                )

        # 创建一个Model实例
        model = Model()
        # 生成一个3x4的随机张量
        x = torch.randn(3, 4)
        # 运行测试，传入模型实例和输入张量
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_end(self):
        # 定义一个继承自torch.jit.ScriptModule的arange脚本模型类
        class ArangeScript(torch.jit.ScriptModule):
            # 定义脚本模型的前向传播方法
            @torch.jit.script_method
            def forward(self, a):
                # 返回从0开始到张量a的长度减1的浮点数arange，并将其视图改为列向量后加上张量a
                return torch.arange(a.size(0), dtype=torch.float).view(-1, 1) + a

        # 生成一个3x4的随机张量，并标记为需要梯度计算
        x = torch.randn(3, 4, requires_grad=True)
        # 对ArangeScript模型进行前向传播，得到输出张量
        outputs = ArangeScript()(x)
        # 运行测试，传入ArangeScript模型实例和输入张量
        self.run_test(ArangeScript(), x)

        # 定义一个继承自torch.nn.Module的arange模型类
        class ArangeModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, a):
                # 返回从0开始到张量a的长度减1的浮点数arange，并将其视图改为列向量后加上张量a
                return torch.arange(a.size(0), dtype=torch.float).view(-1, 1) + a

        # 运行测试，传入ArangeModel模型实例和输入张量
        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_end_notype(self):
        # 定义一个继承自torch.jit.ScriptModule的arange脚本模型类
        class ArangeScript(torch.jit.ScriptModule):
            # 定义脚本模型的前向传播方法
            @torch.jit.script_method
            def forward(self, a):
                # 返回从0开始到张量a的长度减1的整数arange
                return torch.arange(a.size(0))

        # 生成一个3x4的随机张量，并标记为需要梯度计算
        x = torch.randn(3, 4, requires_grad=True)
        # 对ArangeScript模型进行前向传播，得到输出张量
        outputs = ArangeScript()(x)
        # 运行测试，传入ArangeScript模型实例、输入张量、输入名称和动态轴
        self.run_test(ArangeScript(), x, input_names=["x"], dynamic_axes={"x": [0, 1]})
        # 运行测试，传入ArangeScript模型实例、输入张量、保留的ONNX输入索引
        self.run_test(ArangeScript(), x, remained_onnx_input_idx=[])

        # 定义一个继承自torch.nn.Module的arange模型类
        class ArangeModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, a):
                # 返回从0开始到张量a的长度减1的整数arange
                return torch.arange(a.size(0))

        # 运行测试，传入ArangeModel模型实例、输入张量、输入名称和动态轴
        self.run_test(ArangeModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1]})
        # 运行测试，传入ArangeModel模型实例、输入张量、保留的ONNX输入索引
        self.run_test(ArangeModel(), x, remained_onnx_input_idx=[])
    # 定义测试函数 test_arange_start_end_notype，测试不同类型下的 torch.arange() 函数使用
    def test_arange_start_end_notype(self):
        # 定义 ArangeScript 类，继承自 torch.jit.ScriptModule，用于 JIT 脚本化模型
        class ArangeScript(torch.jit.ScriptModule):
            # 定义前向方法，使用 JIT 脚本方法装饰器
            @torch.jit.script_method
            def forward(self, a):
                # 返回 torch.arange() 函数生成的张量与输入张量 a 相加的结果
                return torch.arange(2.7, a.size(0) + 2).view(-1, 1) + a

        # 生成一个形状为 (3, 4) 的随机张量 x，并要求梯度计算
        x = torch.randn(3, 4, requires_grad=True)
        # 运行测试，传入 ArangeScript 实例和随机张量 x
        self.run_test(ArangeScript(), x)

        # 定义 ArangeModel 类，继承自 torch.nn.Module，用于标准模型定义
        class ArangeModel(torch.nn.Module):
            # 定义前向方法
            def forward(self, a):
                # 返回 torch.arange() 函数生成的张量与输入张量 a 相加的结果
                return torch.arange(2.7, a.size(0) + 2).view(-1, 1) + a

        # 运行测试，传入 ArangeModel 实例和随机张量 x
        self.run_test(ArangeModel(), x)

    # 标记为需要跳过的测试函数，要求最低支持 Opset 版本为 9
    @skipIfUnsupportedMinOpsetVersion(9)
    # 定义测试函数 test_arange_start_end_step，测试带步长参数的 torch.arange() 函数使用
    def test_arange_start_end_step(self):
        # 定义 ArangeScript 类，继承自 torch.jit.ScriptModule，用于 JIT 脚本化模型
        class ArangeScript(torch.jit.ScriptModule):
            # 定义前向方法，使用 JIT 脚本方法装饰器
            @torch.jit.script_method
            def forward(self, a):
                # 返回 torch.arange() 函数生成的张量与输入张量 a 相加的结果
                return (
                    torch.arange(
                        2, a.size(0) * a.size(1) + 2, a.size(1), dtype=torch.float
                    ).view(-1, 1)
                    + a
                )

        # 生成一个形状为 (3, 4) 的随机张量 x，并要求梯度计算
        x = torch.randn(3, 4, requires_grad=True)
        # 运行测试，传入 ArangeScript 实例和随机张量 x
        self.run_test(ArangeScript(), x)

        # 定义 ArangeModel 类，继承自 torch.nn.Module，用于标准模型定义
        class ArangeModel(torch.nn.Module):
            # 定义前向方法
            def forward(self, a):
                # 返回 torch.arange() 函数生成的张量与输入张量 a 相加的结果
                return (
                    torch.arange(
                        2, a.size(0) * a.size(1) + 2, a.size(1), dtype=torch.float
                    ).view(-1, 1)
                    + a
                )

        # 运行测试，传入 ArangeModel 实例和随机张量 x
        self.run_test(ArangeModel(), x)

    # 标记为需要跳过的测试函数，要求最低支持 Opset 版本为 11
    @skipIfUnsupportedMinOpsetVersion(11)
    # 定义测试函数 test_arange_start_end_step_notype，测试带步长参数但不带数据类型的 torch.arange() 函数使用
    def test_arange_start_end_step_notype(self):
        # 定义 ArangeScript 类，继承自 torch.jit.ScriptModule，用于 JIT 脚本化模型
        class ArangeScript(torch.jit.ScriptModule):
            # 定义前向方法，使用 JIT 脚本方法装饰器
            @torch.jit.script_method
            def forward(self, a):
                # 返回 torch.arange() 函数生成的张量与输入张量 a 相加的结果
                return (
                    torch.arange(2.7, a.size(0) * a.size(1) + 2, a.size(1)).view(-1, 1)
                    + a
                )

        # 生成一个形状为 (3, 4) 的随机张量 x，并要求梯度计算
        x = torch.randn(3, 4, requires_grad=True)
        # 运行测试，传入 ArangeScript 实例和随机张量 x
        self.run_test(ArangeScript(), x)

        # 定义 ArangeModel 类，继承自 torch.nn.Module，用于标准模型定义
        class ArangeModel(torch.nn.Module):
            # 定义前向方法
            def forward(self, a):
                # 返回 torch.arange() 函数生成的张量与输入张量 a 相加的结果
                return (
                    torch.arange(2.7, a.size(0) * a.size(1) + 2, a.size(1)).view(-1, 1)
                    + a
                )

        # 运行测试，传入 ArangeModel 实例和随机张量 x
        self.run_test(ArangeModel(), x)

    # 标记为需要跳过的测试函数，要求最低支持 Opset 版本为 9
    @skipIfUnsupportedMinOpsetVersion(9)
    # 定义测试函数 test__dim_arange，测试 torch._dim_arange() 函数的使用
    def test__dim_arange(self):
        # 定义 DimArange 类，继承自 torch.nn.Module，用于测试 _dim_arange 函数
        class DimArange(torch.nn.Module):
            # 定义前向方法
            def forward(self, input):
                # 返回 torch._dim_arange() 函数对输入张量 input 进行维度为 1 的操作的结果
                return torch._dim_arange(input, 1)

        # 创建一个形状为 (5, 6) 的全一张量 x
        x = torch.ones(5, 6)
        # 运行测试，传入 DimArange 实例和全一张量 x，并设置输入名称和动态轴
        self.run_test(DimArange(), x, input_names=["x"], dynamic_axes={"x": [0, 1]})
        # 如果 Opset 版本小于 11，则将 remained_onnx_input_idx 设为 None；否则设为空列表
        remained_onnx_input_idx = None if self.opset_version < 11 else []
        # 再次运行测试，传入 DimArange 实例和全一张量 x，同时传入 remained_onnx_input_idx
        self.run_test(DimArange(), x, remained_onnx_input_idx=remained_onnx_input_idx)
    # 测试比较运算符的功能，用于测试模型对不同输入的处理
    def _test_compare_ops(self, model, num_inputs):
        # 生成一个随机的浮点数张量，并设置其需要梯度
        x_float = torch.randn(1, 2, 3, 4, requires_grad=True)
        # 生成一个随机的整数张量
        x_int = torch.randint(10, (3, 4), dtype=torch.int32)
        
        # 根据输入数量选择不同的测试方式
        if num_inputs > 1:
            # 生成第二个随机的浮点数张量，并设置其需要梯度
            y_float = torch.randn(1, 2, 3, 4, requires_grad=True)
            # 生成第二个随机的整数张量
            y_int = torch.randint(10, (3, 4), dtype=torch.int32)
            
            # 分别使用不同的输入对模型进行测试
            self.run_test(model, (x_float, y_float))
            self.run_test(model, (x_float, y_int))
            self.run_test(model, (x_int, y_float))
            self.run_test(model, (x_int, y_int))
        else:
            # 如果只有一个输入，直接使用该输入对模型进行测试
            self.run_test(model, x_float)
            self.run_test(model, x_int)

    # 如果不支持最小的操作集版本9，则跳过测试
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_and_or_xor(self):
        # 定义一个简单的模型，实现按位异或、按位或、按位与以及按位取反操作
        class MyModel(torch.nn.Module):
            def forward(self, x, y):
                return x ^ y, x | y, x & y, ~x

        # 生成两个随机的布尔类型张量作为输入
        x = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        y = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        
        # 运行测试，验证模型对布尔类型输入的处理
        self.run_test(MyModel(), input_args=(x, y))

    # 如果不支持最小的操作集版本9，则跳过测试
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_logical_and(self):
        # 定义一个逻辑与的模型，对两个输入进行逻辑与运算
        class AndModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.logical_and(x, y)

        # 生成两个随机的布尔类型张量作为输入
        x = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        y = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        
        # 运行测试，验证模型对布尔类型输入的处理
        self.run_test(AndModel(), input_args=(x, y))

        # 使用不同类型的张量作为输入，分别进行测试
        x = torch.randint(10, (5, 5), dtype=torch.int32)
        y = torch.randint(10, (5, 5), dtype=torch.int32)
        self.run_test(AndModel(), input_args=(x, y))

        x = torch.randint(10, (5, 5), dtype=torch.double)
        y = torch.randint(10, (5, 5), dtype=torch.double)
        self.run_test(AndModel(), input_args=(x, y))

        x = torch.randint(10, (2, 3, 5), dtype=torch.float32)
        y = torch.randint(10, (2, 3, 5), dtype=torch.long)
        self.run_test(AndModel(), input_args=(x, y))

    # 如果不支持最小的操作集版本9，则跳过测试
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_logical_or(self):
        # 定义一个逻辑或的模型，对两个输入进行逻辑或运算
        class OrModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.logical_or(x, y)

        # 生成两个随机的布尔类型张量作为输入
        x = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        y = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        
        # 运行测试，验证模型对布尔类型输入的处理
        self.run_test(OrModel(), input_args=(x, y))

        # 使用不同类型的张量作为输入，分别进行测试
        x = torch.randint(10, (5, 5), dtype=torch.int32)
        y = torch.randint(10, (5, 5), dtype=torch.int32)
        self.run_test(OrModel(), input_args=(x, y))

        x = torch.randint(10, (5, 5), dtype=torch.double)
        y = torch.randint(10, (5, 5), dtype=torch.double)
        self.run_test(OrModel(), input_args=(x, y))

        x = torch.randint(10, (2, 3, 5), dtype=torch.float32)
        y = torch.randint(10, (2, 3, 5), dtype=torch.long)
        self.run_test(OrModel(), input_args=(x, y))
    # 定义一个测试函数，用于测试逻辑异或操作
    def test_logical_xor(self):
        # 定义一个内部类 XorModel，继承自 torch.nn.Module，用于执行逻辑异或操作
        class XorModel(torch.nn.Module):
            # 定义 forward 方法，接收两个输入 x 和 y，并返回它们的逻辑异或结果
            def forward(self, x, y):
                return torch.logical_xor(x, y)

        # 创建两个布尔类型的张量 x 和 y，形状为 (5, 5)，值为 0 或 1
        x = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        y = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        # 调用 self.run_test 方法，测试 XorModel 的 forward 方法，传入参数 x 和 y
        self.run_test(XorModel(), input_args=(x, y))

        # 创建两个 int32 类型的张量 x 和 y，形状为 (5, 5)，值为 0 到 9 之间的随机整数
        x = torch.randint(10, (5, 5), dtype=torch.int32)
        y = torch.randint(10, (5, 5), dtype=torch.int32)
        # 再次调用 self.run_test 方法，测试 XorModel 的 forward 方法，传入参数 x 和 y
        self.run_test(XorModel(), input_args=(x, y))

        # 创建两个 double 类型的张量 x 和 y，形状为 (5, 5)，值为 0 到 9 之间的随机浮点数
        x = torch.randint(10, (5, 5), dtype=torch.double)
        y = torch.randint(10, (5, 5), dtype=torch.double)
        # 再次调用 self.run_test 方法，测试 XorModel 的 forward 方法，传入参数 x 和 y
        self.run_test(XorModel(), input_args=(x, y))

        # 创建两个张量 x 和 y，形状为 (2, 3, 5)，分别为 float32 和 long 类型，值为 0 到 9 之间的随机数
        x = torch.randint(10, (2, 3, 5), dtype=torch.float32)
        y = torch.randint(10, (2, 3, 5), dtype=torch.long)
        # 再次调用 self.run_test 方法，测试 XorModel 的 forward 方法，传入参数 x 和 y
        self.run_test(XorModel(), input_args=(x, y))

    # 标记为在不支持的最小 opset 版本下跳过执行的测试函数
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_logical_not(self):
        # 定义一个内部类 NotModel，继承自 torch.nn.Module，用于执行逻辑非操作
        class NotModel(torch.nn.Module):
            # 定义 forward 方法，接收输入 x，并返回其逻辑非结果
            def forward(self, x):
                return torch.logical_not(x)

        # 创建一个布尔类型的张量 x，形状为 (5, 5)，值为 0 或 1
        x = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        # 调用 self.run_test 方法，测试 NotModel 的 forward 方法，传入参数 x
        self.run_test(NotModel(), input_args=(x,))

        # 创建一个 int32 类型的张量 x，形状为 (5, 5)，值为 0 到 9 之间的随机整数
        x = torch.randint(10, (5, 5), dtype=torch.int32)
        # 再次调用 self.run_test 方法，测试 NotModel 的 forward 方法，传入参数 x
        self.run_test(NotModel(), input_args=(x,))

        # 创建一个 double 类型的张量 x，形状为 (5, 5)，值为 0 到 9 之间的随机浮点数
        x = torch.randint(10, (5, 5), dtype=torch.double)
        # 再次调用 self.run_test 方法，测试 NotModel 的 forward 方法，传入参数 x
        self.run_test(NotModel(), input_args=(x,))

        # 创建一个张量 x，形状为 (2, 3, 5)，为 float32 类型，值为 0 到 9 之间的随机数
        x = torch.randint(10, (2, 3, 5), dtype=torch.float32)
        # 再次调用 self.run_test 方法，测试 NotModel 的 forward 方法，传入参数 x
        self.run_test(NotModel(), input_args=(x,))

    # 标记为在不支持的最小 opset 版本下跳过执行的测试函数，并添加了注释说明 float equal 操作在 opset 11 之后添加
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_eq(self):
        # 定义一个内部类 EqualModel，继承自 torch.nn.Module，用于执行相等比较操作
        class EqualModel(torch.nn.Module):
            # 定义 forward 方法，接收输入 input 和 other，并返回它们的相等比较结果
            def forward(self, input, other):
                return input == other

        # 调用 _test_compare_ops 方法，测试 EqualModel 的 forward 方法，传入参数 2
        self._test_compare_ops(EqualModel(), 2)

    # 定义一个测试函数，用于测试大于比较操作
    def test_gt(self):
        # 定义一个内部类 GreaterModel，继承自 torch.nn.Module，用于执行大于比较操作
        class GreaterModel(torch.nn.Module):
            # 定义 forward 方法，接收输入 input 和 other，并返回 input 是否大于 other 的比较结果
            def forward(self, input, other):
                return input > other

        # 调用 _test_compare_ops 方法，测试 GreaterModel 的 forward 方法，传入参数 2
        self._test_compare_ops(GreaterModel(), 2)

    # 标记为在不支持的最小 opset 版本下跳过执行的测试函数
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_ge(self):
        # 定义一个内部类 GreaterOrEqualModel，继承自 torch.nn.Module，用于执行大于等于比较操作
        class GreaterOrEqualModel(torch.nn.Module):
            # 定义 forward 方法，接收输入 input 和 other，并返回 input 是否大于等于 other 的比较结果
            def forward(self, input, other):
                return input >= other

        # 调用 _test_compare_ops 方法，测试 GreaterOrEqualModel 的 forward 方法，传入参数 2
        self._test_compare_ops(GreaterOrEqualModel(), 2)

    # 定义一个测试函数，用于测试大于比较操作（与 test_gt 类似，但只有一个参数）
    def test_gt_scalar(self):
        # 定义一个内部类 GreaterModel，继承自 torch.nn.Module，用于执行大于比较操作
        class GreaterModel(torch.nn.Module):
            # 定义 forward 方法，接收输入 input，并返回 input 是否大于 1 的比较结果
            def forward(self, input):
                return input > 1

        # 调用 _test_compare_ops 方法，测试 GreaterModel 的 forward 方法，传入参数 1
        self._test_compare_ops(GreaterModel(), 1)

    # 定义一个测试函数，用于测试大于比较操作（与 test_gt_scalar 类似，但使用了类成员变量）
    def test_gt_primitive(self):
        # 定义一个内部类 GreaterModel，继承自 torch.nn.Module，用于执行大于比较操作
        class GreaterModel(torch.nn.Module):
            # 定义构造函数，初始化一个整型成员变量 y 为 2
            def __init__(self):
                super().__init__()
                self.y: int = 2

            # 定义 forward 方法，接收输入 x，返回成员变量 y 是否大于 x 的比较结果
            def forward(self, x: int):
                return self.y > x

        # 创建一个整型变量 x，赋值为 3
        x = 3
        # 调用 self.run_test 方法，测试 GreaterModel 的 forward 方法，传入参数 x
        self.run_test(GreaterModel(), (x,))

    # 标记为在不支持的最小 opset 版本下跳过执行的测试函数
    @skipIfUnsupportedMinOpsetVersion(9
    # 定义测试方法，测试输入是否小于给定值
    def test_lt(self):
        # 定义一个继承自 torch.nn.Module 的类 LessModel
        class LessModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, input, other):
                # 返回输入是否大于给定值的布尔值
                return input > other

        # 调用 _test_compare_ops 方法，测试 LessModel 类
        self._test_compare_ops(LessModel(), 2)

    # 根据条件跳过不支持 Opset 版本低于 9 的测试
    @skipIfUnsupportedMinOpsetVersion(9)
    # 定义测试方法，测试输入是否小于等于给定值
    def test_le(self):
        # 定义一个继承自 torch.nn.Module 的类 LessOrEqualModel
        class LessOrEqualModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, input, other):
                # 返回输入是否小于等于给定值的布尔值
                return input <= other

        # 调用 _test_compare_ops 方法，测试 LessOrEqualModel 类
        self._test_compare_ops(LessOrEqualModel(), 2)

    # 定义测试方法，测试输入是否小于给定标量值
    def test_lt_scalar(self):
        # 定义一个继承自 torch.nn.Module 的类 LessModel
        class LessModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, input):
                # 返回输入是否小于给定标量值的布尔值
                return input < 1

        # 调用 _test_compare_ops 方法，测试 LessModel 类
        self._test_compare_ops(LessModel(), 1)

    # 根据条件跳过不支持 Opset 版本低于 9 的测试
    @skipIfUnsupportedMinOpsetVersion(9)
    # 定义测试方法，测试输入是否小于等于给定标量值
    def test_le_scalar(self):
        # 定义一个继承自 torch.nn.Module 的类 LessOrEqualModel
        class LessOrEqualModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, input):
                # 返回输入是否小于等于给定标量值的布尔值
                return input <= 1

        # 调用 _test_compare_ops 方法，测试 LessOrEqualModel 类
        self._test_compare_ops(LessOrEqualModel(), 1)

    # 定义测试矩阵相乘方法
    def test_matmul(self):
        # 定义一个继承自 torch.nn.Module 的类 MatmulModel
        class MatmulModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, input, other):
                # 返回输入张量的矩阵乘积
                return torch.matmul(input, other)

        # 创建两个随机张量 x 和 y，测试 MatmulModel 类
        x = torch.randn(3, 4, requires_grad=True)
        y = torch.randn(4, 5, requires_grad=True)
        self.run_test(MatmulModel(), (x, y))

        # 创建两个整数张量 x 和 y，测试 MatmulModel 类
        x = torch.randint(10, (3, 4))
        y = torch.randint(10, (4, 5))
        self.run_test(MatmulModel(), (x, y))

    # 定义测试批量矩阵相乘方法
    def test_matmul_batch(self):
        # 定义一个继承自 torch.nn.Module 的类 MatmulModel
        class MatmulModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, input, other):
                # 返回输入张量的批量矩阵乘积
                return torch.matmul(input, other)

        # 创建两个随机张量 x 和 y，测试 MatmulModel 类
        x = torch.randn(2, 3, 4, requires_grad=True)
        y = torch.randn(2, 4, 5, requires_grad=True)
        self.run_test(MatmulModel(), (x, y))

        # 创建两个整数张量 x 和 y，测试 MatmulModel 类
        x = torch.randint(10, (2, 3, 4))
        y = torch.randint(10, (2, 4, 5))
        self.run_test(MatmulModel(), (x, y))

    # 定义测试 argmin 和 argmax 方法
    def _argmin_argmax_model(self, input):
        # 定义一个继承自 torch.nn.Module 的类 ArgminArgmaxModel
        class ArgminArgmaxModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, input):
                # 返回输入张量的最小和最大值索引
                return (
                    torch.argmin(input),
                    torch.argmax(input),
                    torch.argmin(input, keepdim=True),
                    torch.argmax(input, keepdim=True),
                    torch.argmin(input, dim=0, keepdim=True),
                    torch.argmax(input, dim=1, keepdim=True),
                )

        # 调用 run_test 方法，测试 ArgminArgmaxModel 类
        self.run_test(ArgminArgmaxModel(), input)

    # 根据条件跳过不支持 Opset 版本低于 9 的测试
    @skipIfUnsupportedMinOpsetVersion(9)
    # 定义测试 argmin 和 argmax 方法
    def test_argmin_argmax(self):
        # 创建一个随机张量 input，测试 _argmin_argmax_model 方法
        input = torch.randn(7, 3, 5)
        self._argmin_argmax_model(input)

    # 在 Opset 版本低于 12 时，跳过带有 "select_last_index" 的 argmin 和 argmax 测试
    # "select_last_index" 在 Opset 版本 12 中添加，用于处理张量中同一值出现多次的情况
    @skipIfUnsupportedMinOpsetVersion(12)
    # 定义测试 argmin 和 argmax 方法，测试带有 "select_last_index" 的情况
    def test_argmin_argmax_select_last_index(self):
        # 创建两个张量 input，测试 _argmin_argmax_model 方法
        input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 2.0]])
        self._argmin_argmax_model(input)

        # 创建一个全为 1 的张量 input，测试 _argmin_argmax_model 方法
        input = torch.ones(7, 3, 5)
        self._argmin_argmax_model(input)
    def test_repeat(self):
        # 定义一个内部模型类 RepeatModel，继承自 torch.nn.Module
        class RepeatModel(torch.nn.Module):
            # 定义前向传播函数，接受两个参数 x 和 y
            def forward(self, x, y):
                # 将 x 沿着第一维重复 y.shape[0] 次，第二维重复 1 次
                x2 = x.repeat(y.shape[0], 1)
                # 将 y 变形为 (-1, 1) 的形状
                y1 = y.view(-1, 1)
                # 返回 x2 和 y1 按元素相加的结果
                return x2 + y1

        # 创建输入张量 x 和 y
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 8, 9])
        # 运行测试，使用 RepeatModel() 作为模型，传入参数 (x, y)
        self.run_test(RepeatModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_repeat_interleave(self):
        # 定义一个内部模型类 FlattenModel，继承自 torch.nn.Module
        class FlattenModel(torch.nn.Module):
            # 定义前向传播函数，接受一个参数 x
            def forward(self, x):
                # 对输入张量 x 进行维度上的重复插入，每个元素重复 2 次
                return x.repeat_interleave(2)

        # 遍历不同形状的输入张量
        for shape in ([3], [3, 4], [2, 3, 4]):
            # 创建随机张量 x，形状由 shape 决定
            x = torch.randn(shape)
            # 运行测试，使用 FlattenModel() 作为模型，传入参数 (x,)
            self.run_test(FlattenModel(), (x,))

        # 定义一个内部模型类 DimsModel，继承自 torch.nn.Module
        class DimsModel(torch.nn.Module):
            # 定义前向传播函数，接受一个参数 x
            def forward(self, x):
                # 对输入张量 x 进行维度为 1 的重复插入，每个元素重复 4 次
                return x.repeat_interleave(4, dim=1)

        # 创建输入张量 x
        x = torch.tensor([[1, 2], [3, 4]])
        # 运行测试，使用 DimsModel() 作为模型，传入参数 (x,)
        self.run_test(DimsModel(), (x,))

        # 定义一个内部模型类 DimsModel2，继承自 torch.nn.Module
        class DimsModel2(torch.nn.Module):
            # 定义前向传播函数，接受一个参数 x
            def forward(self, x):
                # 创建重复次数张量 repeats，指定为 [4]
                repeats = torch.tensor([4])
                # 对输入张量 x 进行重复插入，重复次数由 repeats 决定，维度为 1
                return torch.repeat_interleave(x, repeats, dim=1)

        # 创建输入张量 x
        x = torch.tensor([[1, 2], [3, 4]])
        # 运行测试，使用 DimsModel2() 作为模型，传入参数 (x,)
        self.run_test(DimsModel2(), (x,))

        # 定义一个内部模型类 RepeatsDimsModel，继承自 torch.nn.Module
        class RepeatsDimsModel(torch.nn.Module):
            # 定义前向传播函数，接受一个参数 x
            def forward(self, x):
                # 创建重复次数张量 repeats，指定为 [1, 2]
                repeats = torch.tensor([1, 2])
                # 对输入张量 x 进行重复插入，重复次数由 repeats 决定，维度为 0
                return torch.repeat_interleave(x, repeats, dim=0)

        # 创建输入张量 x
        x = torch.tensor([[1, 2], [3, 4]])
        # 运行测试，使用 RepeatsDimsModel() 作为模型，传入参数 (x,)
        self.run_test(RepeatsDimsModel(), (x,))

        # 定义一个内部模型类 RepeatsDimsModel2，继承自 torch.nn.Module
        class RepeatsDimsModel2(torch.nn.Module):
            # 定义前向传播函数，接受一个参数 x
            def forward(self, x):
                # 创建重复次数张量 repeats，指定为 [1, 2]
                repeats = torch.tensor([1, 2])
                # 对输入张量 x 进行重复插入，重复次数由 repeats 决定，维度为 1
                return torch.repeat_interleave(x, repeats, dim=1)

        # 创建输入张量 x
        x = torch.tensor([[1, 2], [3, 4]])
        # 运行测试，使用 RepeatsDimsModel2() 作为模型，传入参数 (x,)
        self.run_test(RepeatsDimsModel2(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_repeat_interleave_noop(self):
        # 定义一个内部模型类 Model，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            # 定义前向传播函数，接受一个参数 x
            def forward(self, x):
                # 对输入张量 x 进行维度为 1 的重复插入，每个元素重复 1 次（无变化）
                return x.repeat_interleave(1, dim=1)

        # 创建输入张量 x
        x = torch.randn(4, 1, 8)
        # 运行测试，使用 Model() 作为模型，传入参数 (x,)
        self.run_test(Model(), (x,))
    # 定义一个测试函数，用于测试动态重复和交错的功能
    def test_multiple_dynamic_repeat_interleave(self):
        # 定义一个继承自torch.nn.Module的模型类DynamicRepeatsModel
        class DynamicRepeatsModel(torch.nn.Module):
            # 模型的前向传播方法，接受输入x和重复次数repeats，使用torch.repeat_interleave函数在维度1上进行重复
            def forward(self, x, repeats):
                return torch.repeat_interleave(x, repeats, dim=1)

        # 创建一个输入张量x
        x = torch.tensor([[1, 2, 4], [3, 4, 7]])
        # 创建一个重复次数张量repeats
        repeats = torch.tensor([2, 3, 4])
        # 创建另一个重复次数张量another_repeats
        another_repeats = torch.tensor([4, 3, 2])
        # 运行测试方法self.run_test，传入DynamicRepeatsModel模型实例及其输入参数
        self.run_test(
            DynamicRepeatsModel(),
            (x, repeats),
            additional_test_inputs=[(x, another_repeats)],
            input_names=["input_1", "repeats_1"],
            dynamic_axes={"repeats_1": {0: "r"}},
        )

        # 定义另一个继承自torch.nn.Module的模型类DynamicRepeatsModel2
        class DynamicRepeatsModel2(torch.nn.Module):
            # 模型的前向传播方法，接受输入x和重复次数repeats，使用torch.repeat_interleave函数在维度0上进行重复
            def forward(self, x, repeats):
                return torch.repeat_interleave(x, repeats, dim=0)

        # 重新定义输入张量x和重复次数张量repeats
        x = torch.tensor([[1, 2, 4], [3, 4, 7]])
        repeats = torch.tensor([2, 3])
        another_repeats = torch.tensor([4, 3])
        # 运行测试方法self.run_test，传入DynamicRepeatsModel2模型实例及其输入参数
        self.run_test(
            DynamicRepeatsModel2(),
            (x, repeats),
            additional_test_inputs=[(x, another_repeats)],
            input_names=["input_1", "repeats_1"],
            dynamic_axes={"repeats_1": {0: "r"}},
        )

    # 定义一个测试函数，用于测试张量视图操作
    def test_view(self):
        # 定义一个继承自torch.nn.Module的模型类ViewModel
        class ViewModel(torch.nn.Module):
            # 模型的前向传播方法，接受输入input，使用input.view方法将输入重塑为4行24列的张量
            def forward(self, input):
                return input.view(4, 24)

        # 创建一个形状为(4, 2, 3, 4)的随机整数张量x
        x = torch.randint(10, (4, 2, 3, 4), dtype=torch.int32)
        # 运行测试方法self.run_test，传入ViewModel模型实例及其输入参数x
        self.run_test(ViewModel(), x)

    # 定义一个测试函数，用于测试动态形状的张量视图操作
    def test_view_dynamic(self):
        # 定义一个继承自torch.nn.Module的模型类ViewModel
        class ViewModel(torch.nn.Module):
            # 模型的前向传播方法，接受输入input和other，使用input.view方法将input重塑为other张量的形状
            def forward(self, input, other):
                return input.view(other.shape)

        # 创建一个形状为(2, 3, 4)的随机张量x和形状为(6, 4)的随机张量shape
        x = torch.randn(2, 3, 4)
        shape = torch.randn(6, 4)
        # 运行测试方法self.run_test，传入ViewModel模型实例及其输入参数x和shape
        self.run_test(
            ViewModel(),
            (x, shape),
            input_names=["x", "shape"],
            dynamic_axes={"x": [0, 1, 2], "shape": [0, 1]},
        )
        # 再次运行测试方法self.run_test，传入ViewModel模型实例及其输入参数x和shape，保留ONNX输入索引为0
        self.run_test(ViewModel(), (x, shape), remained_onnx_input_idx=[0])

    # 定义一个测试函数，用于测试包含零维度张量的视图操作
    def test_view_dynamic_zero_dim(self):
        # 定义一个继承自torch.nn.Module的模型类ViewModel
        class ViewModel(torch.nn.Module):
            # 模型的前向传播方法，接受输入input，使用input.view方法将输入重塑为形状为(-1, 2)的张量，再次重塑为形状为(1, -1)的张量
            def forward(self, input):
                input = input.view(-1, 2)
                return input.view(1, -1)

        # 创建一个包含两个元素的全1张量x和一个空张量another_x
        x = torch.ones(2)
        another_x = torch.empty((0,))
        # 运行测试方法self.run_test，传入ViewModel模型实例及其输入参数x和additional_test_inputs中的another_x
        self.run_test(
            ViewModel(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={
                "input_1": [
                    0,
                ]
            },
        )

    # 定义一个测试函数，用于测试张量的view_as方法
    def test_view_as(self):
        # 定义一个继承自torch.nn.Module的模型类ViewModel
        class ViewModel(torch.nn.Module):
            # 模型的前向传播方法，接受输入input和other，使用input.view_as方法将input重塑为other的形状
            def forward(self, input, other):
                return input.view_as(other)

        # 创建一个形状为(2, 3, 4)的随机张量x和形状为(6, 4)的随机张量y
        x = torch.randn(2, 3, 4)
        y = torch.randn(6, 4)
        # 运行测试方法self.run_test，传入ViewModel模型实例及其输入参数x和y
        self.run_test(ViewModel(), (x, y))
    # 定义测试线性模型的方法
    def test_linear(self):
        # 定义一个简单的线性模型类
        class LinearModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个线性层，输入维度为16，输出维度为16
                self.fc = torch.nn.Linear(16, 16)

            def forward(self, x):
                # 模型的前向传播方法，将输入数据通过线性层计算输出
                out = self.fc(x)
                out = self.fc(out)
                return out

        # 生成一个形状为(3, 16)的随机输入数据
        x = torch.randn(3, 16)
        # 运行测试，测试上面定义的LinearModel类
        self.run_test(LinearModel(), (x,))

        # 定义另一个线性模型类，使用torch.nn.functional中的linear函数计算输出
        class LinearModel(torch.nn.Module):
            def forward(self, input, weight, bias):
                return torch.nn.functional.linear(input, weight, bias)

        # 生成形状为(2, 2)的随机输入数据x和y，形状为(1,)的随机输入数据z
        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        z = torch.randn(1)
        # 运行测试，测试上面定义的LinearModel类
        self.run_test(LinearModel(), (x, y, z))

        # 生成形状为(3, 3, 3)的随机输入数据x和y，形状为(1,)的随机输入数据z
        x = torch.randn(3, 3, 3)
        y = torch.randn(3, 3)
        z = torch.randn(1)
        # 运行测试，测试上面定义的LinearModel类
        self.run_test(LinearModel(), (x, y, z))

    # 跳过脚本测试的装饰器，用于指定下面的测试方法不参与脚本测试
    @skipScriptTest()
    def test_weight_norm(self):
        # 对torch.nn.Linear模型应用权重归一化，dim=1表示在第一维度上进行归一化
        model = torch.nn.utils.weight_norm(torch.nn.Linear(5, 10), dim=1)
        # 生成形状为(3, 4, 5)的随机输入数据x
        x = torch.randn(3, 4, 5, requires_grad=True)
        # 运行测试，测试应用了权重归一化的模型model
        self.run_test(model, x)

        # 对torch.nn.Linear模型应用权重归一化，dim=1表示在第一维度上进行归一化
        model = torch.nn.utils.weight_norm(torch.nn.Linear(5, 10), dim=1)
        # 生成形状为(4, 5)的随机输入数据x
        x = torch.randn(4, 5, requires_grad=True)
        # 运行测试，测试应用了权重归一化的模型model
        self.run_test(model, x)

        # 对torch.nn.Conv1d模型应用权重归一化，没有指定dim，默认在最后一个维度上进行归一化
        model = torch.nn.utils.weight_norm(torch.nn.Conv1d(1, 1, 3))
        # 生成形状为(1, 1, 5)的随机输入数据x
        x = torch.randn(1, 1, 5, requires_grad=True)
        # 运行测试，测试应用了权重归一化的模型model
        self.run_test(model, x)

        # 对torch.nn.Conv1d模型应用权重归一化，指定dim=-2，在倒数第二个维度上进行归一化
        model = torch.nn.utils.weight_norm(torch.nn.Conv1d(1, 1, 3), dim=-2)
        # 生成形状为(1, 1, 5)的随机输入数据x
        x = torch.randn(1, 1, 5, requires_grad=True)
        # 运行测试，测试应用了权重归一化的模型model
        self.run_test(model, x)

        # 对torch.nn.Conv1d模型应用权重归一化，指定name="weight"，在weight参数上进行归一化
        model = torch.nn.utils.weight_norm(torch.nn.Conv1d(3, 6, 3), name="weight")
        # 生成形状为(3, 3, 5)的随机输入数据x
        x = torch.randn(3, 3, 5, requires_grad=True)
        # 运行测试，测试应用了权重归一化的模型model
        self.run_test(model, x)

    # 跳过脚本测试的装饰器，用于指定下面的测试方法不参与脚本测试
    @skipScriptTest()
    def test_weight_norm_nodim(self):
        # 对torch.nn.Linear模型应用权重归一化，dim=None表示在所有维度上进行归一化
        model = torch.nn.utils.weight_norm(torch.nn.Linear(5, 10), dim=None)
        # 生成形状为(3, 4, 5)的随机输入数据x
        x = torch.randn(3, 4, 5, requires_grad=True)
        # 运行测试，测试应用了权重归一化的模型model
        self.run_test(model, x)

        # 对torch.nn.Linear模型应用权重归一化，dim=None表示在所有维度上进行归一化
        model = torch.nn.utils.weight_norm(torch.nn.Linear(5, 10), dim=None)
        # 生成形状为(4, 5)的随机输入数据x
        x = torch.randn(4, 5, requires_grad=True)
        # 运行测试，测试应用了权重归一化的模型model
        self.run_test(model, x)

    # 定义测试展平模型的方法
    def test_flatten(self):
        # 定义一个展平模型类
        class FlattenModel(torch.nn.Module):
            def forward(self, input):
                # 使用torch.flatten函数将输入展平
                return torch.flatten(input)

        # 创建一个FlattenModel实例
        model = FlattenModel()

        # 使用形状为(1, 2, 3, 4)的随机整数张量x进行测试
        x = torch.randint(10, (1, 2, 3, 4))
        # 运行测试，测试展平模型
        self.run_test(model, x)

        # 使用形状为()的随机张量x进行测试
        x = torch.randn([])
        # 运行测试，测试展平模型
        self.run_test(model, x)

        # 使用形状为(4,)的随机张量x进行测试
        x = torch.randn(4)
        # 运行测试，测试展平模型
        self.run_test(model, x)
    # 定义一个测试方法，用于测试二维张量的展平操作
    def test_flatten2d(self):
        # 定义一个简单的神经网络模型，重写了 forward 方法来执行展平操作
        class FlattenModel(torch.nn.Module):
            def forward(self, input):
                return torch.flatten(input, 1)

        # 创建一个形状为 (1, 2, 3, 4) 的随机整数张量
        x = torch.randint(10, (1, 2, 3, 4))
        # 调用 run_test 方法，运行 FlattenModel 模型并传入 x 作为输入
        self.run_test(FlattenModel(), x)

    # 定义另一个测试方法，用于测试带有负数参数的二维张量展平操作
    def test_flatten2d_neg(self):
        # 定义一个神经网络模型，重写了 forward 方法，执行多个带负数参数的展平操作
        class FlattenModel(torch.nn.Module):
            def forward(self, x):
                return (
                    torch.flatten(x, 1, -1),
                    torch.flatten(x, 0, -2),
                    torch.flatten(x, 1, -2),
                )

        # 创建一个形状为 (1, 2, 3, 4) 的随机整数张量
        x = torch.randint(10, (1, 2, 3, 4))
        # 调用 run_test 方法，运行 FlattenModel 模型并传入 x 作为输入
        self.run_test(FlattenModel(), x)

    # 根据 Opset 版本进行条件跳过，测试动态轴展平操作
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_flatten_dynamic_axes(self):
        # 定义一个神经网络模型，重写了 forward 方法，执行动态轴的展平操作
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return torch.flatten(x, start_dim=2, end_dim=3)

        # 创建形状为 (batch_size, 5, 4, 5) 的随机正态分布张量 x 和形状为 (5, 5, 4, 5) 的随机正态分布张量 y
        batch_size = 3
        x = torch.randn(batch_size, 5, 4, 5)
        y = torch.randn(5, 5, 4, 5)
        model = MyModule()
        # 调用 run_test 方法，运行 MyModule 模型并传入 x 作为输入
        self.run_test(
            model,
            x,
            additional_test_inputs=[y],  # 额外的测试输入 y
            input_names=["input"],  # 输入张量的名称为 "input"
            output_names=["output"],  # 输出张量的名称为 "output"
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # 动态轴定义
        )

    # 根据 Opset 版本进行条件跳过，测试带有 __getitem__ 方法的模型
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_getitem(self):
        # 定义一个带有 __getitem__ 方法的 ScriptModule
        class GetItemModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x, y, z, ind):
                # 创建一个列表 arr，包含 x, y, z，并使用 ind 进行索引
                # 这将生成 prim::ListConstruct(x, y, z) + aten::__getitem__
                arr = [x, y, z]
                return arr[ind]

        # 创建形状为 (3, 4, 5) 的随机正态分布张量 x, y, z 和索引张量 ind
        x = torch.randn(3, 4, 5)
        y = torch.randn(1, 4, 5)
        z = torch.randn(2, 4, 5)
        ind = torch.tensor(1, dtype=torch.long)
        # 调用 run_test 方法，运行 GetItemModel 模型并传入 x, y, z, ind 作为输入
        self.run_test(GetItemModel(), (x, y, z, ind))

        # 创建形状为 (3, 4, 5) 的随机正态分布张量 x, y, z 和索引张量 ind（负数索引）
        ind = torch.tensor(-2, dtype=torch.long)
        # 再次调用 run_test 方法，运行 GetItemModel 模型并传入 x, y, z, ind 作为输入
        self.run_test(GetItemModel(), (x, y, z, ind))

    # 跳过类型检查，测试返回张量元素的整数值
    @skipDtypeChecking
    def test_item(self):
        # 定义一个带有整数返回值的神经网络模型
        class M(torch.nn.Module):
            def forward(self, x, y, i: int):
                return int(x[y[i]].item())

        # 创建一个浮点型序列张量 x 和一个整型张量 y 和整数 i
        x = torch.arange(6, dtype=torch.float)
        y = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        i = 3
        # 调用 run_test 方法，运行 M 模型并传入 x, y, i 作为输入
        self.run_test(torch.jit.script(M()), (x, y, i))

    # 跳过脚本测试，因为 torch.nonzero(x, as_tuple=True) 无法脚本化
    # 根据 Opset 版本进行条件跳过，测试 torch.nonzero 方法
    @skipScriptTest()  # torch.nonzero(x, as_tuple=True) is not scriptable.
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_nonzero(self):
        # 定义一个神经网络模型，重写了 forward 方法，执行 torch.nonzero 方法
        class NonzeroModel(torch.nn.Module):
            def forward(self, x):
                return x.nonzero(), x.nonzero(as_tuple=True)

        # 创建形状为 (3, 4, 5) 的随机正态分布张量 x，并对其进行处理
        x = torch.randn(60).index_fill_(0, torch.randint(0, 60, (20,)), 0).view(3, 4, 5)
        # 调用 run_test 方法，运行 NonzeroModel 模型并传入 x 作为输入
        self.run_test(NonzeroModel(), (x,))
    def test_unbind(self):
        # 定义一个继承自 torch.nn.Module 的内部类 UnbindModel
        class UnbindModel(torch.nn.Module):
            # 定义 forward 方法，处理输入数据的解绑操作
            def forward(self, input):
                # 对输入数据进行解绑操作，获取第二个元素作为输出
                _, out, _ = input.unbind()
                return out

        # 创建一个形状为 (3, 4, 5) 的随机张量 x
        x = torch.randn(3, 4, 5)
        # 运行测试，使用 UnbindModel 实例和 x 作为输入
        self.run_test(UnbindModel(), x)

        # 定义另一个继承自 torch.nn.Module 的内部类 UnbindModel2
        class UnbindModel2(torch.nn.Module):
            # 定义 forward 方法，处理输入数据的按维度解绑操作
            def forward(self, input):
                # 对输入数据按第一个维度进行解绑操作，获取第二个元素作为输出
                _, out, _, _ = input.unbind(1)
                return out

        # 再次创建形状为 (3, 4, 5) 的随机张量 x
        x = torch.randn(3, 4, 5)
        # 运行测试，使用 UnbindModel2 实例和 x 作为输入
        self.run_test(UnbindModel2(), x)

        # 定义第三个继承自 torch.nn.Module 的内部类 UnbindModel3
        class UnbindModel3(torch.nn.Module):
            # 定义 forward 方法，处理输入数据的负索引解绑操作
            def forward(self, input):
                # 对输入数据按倒数第二个维度进行解绑操作，获取第二个元素作为输出
                _, out, _, _ = input.unbind(-2)
                return out

        # 再次创建形状为 (3, 4, 5) 的随机张量 x
        x = torch.randn(3, 4, 5)
        # 运行测试，使用 UnbindModel3 实例和 x 作为输入
        self.run_test(UnbindModel3(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_len(self):
        # 定义一个继承自 torch.jit.ScriptModule 的内部类 LenModel
        class LenModel(torch.jit.ScriptModule):
            # 定义 forward 方法，使用 Torch Script 处理输入数据的长度和加法操作
            @torch.jit.script_method
            def forward(self, input):
                # 返回输入数据解绑后的长度加上输入数据本身
                return len(input.unbind()) + input

        # 创建一个形状为 (4, 5) 的随机张量 x
        x = torch.randn(4, 5)
        # 运行测试，使用 LenModel 实例和 x 作为输入，同时设置输入名称和动态轴
        self.run_test(
            LenModel(),
            x,
            input_names=["input"],
            dynamic_axes={"input": {0: "seq"}},
            additional_test_inputs=(torch.randn(5, 5),),
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_len_list(self):
        # 定义一个继承自 torch.jit.ScriptModule 的内部类 LenListModel
        class LenListModel(torch.jit.ScriptModule):
            # 定义 forward 方法，使用 Torch Script 返回输入张量形状长度全为 1 的张量
            @torch.jit.script_method
            def forward(self, input):
                return torch.ones(len(input.shape))

        # 创建一个形状为 (4, 5) 的随机张量 x
        x = torch.randn(4, 5)
        # 运行测试，使用 LenListModel 实例和 x 作为输入，不保留任何额外的 ONNX 输入索引
        self.run_test(LenListModel(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_unbind_dynamic(self):
        # 定义一个继承自 torch.jit.ScriptModule 的内部类 UnbindModel
        class UnbindModel(torch.jit.ScriptModule):
            # 定义 forward 方法，使用 Torch Script 处理输入数据的解绑操作
            @torch.jit.script_method
            def forward(self, input):
                # 返回输入数据解绑后的第二个元素作为输出
                return input.unbind()[1]

        # 创建一个形状为 (3, 4, 5) 的随机张量 x
        x = torch.randn(3, 4, 5)
        # 运行测试，使用 UnbindModel 实例和 x 作为输入
        self.run_test(UnbindModel(), x)

        # 定义第二个继承自 torch.jit.ScriptModule 的内部类 UnbindModel2
        class UnbindModel2(torch.jit.ScriptModule):
            # 定义 forward 方法，使用 Torch Script 处理输入数据的负索引解绑操作
            @torch.jit.script_method
            def forward(self, input):
                # 返回输入数据按倒数第一维度解绑后的第二个元素作为输出
                return input.unbind(-1)[1]

        # 再次创建形状为 (3, 4, 5) 的随机张量 x
        x = torch.randn(3, 4, 5)
        # 运行测试，使用 UnbindModel2 实例和 x 作为输入
        self.run_test(UnbindModel2(), x)

    @skipScriptTest()  # scripting tests run for opsets > 11. See: test_split_script
    def test_split(self):
        # 定义一个继承自 torch.nn.Module 的内部类 SplitModel
        class SplitModel(torch.nn.Module):
            # 定义 forward 方法，处理输入数据的分割操作
            def forward(self, input):
                # 返回输入数据按指定分割大小分割后的结果及另一种分割结果
                return input.split([2, 1, 2]), input.split([3, 2])[0]

        # 创建一个形状为 (5, 4, 3) 的随机张量 x
        x = torch.randn(5, 4, 3)
        # 运行测试，使用 SplitModel 实例和 x 作为输入
        self.run_test(SplitModel(), x)

        # 定义第二个继承自 torch.nn.Module 的内部类 SplitModel2
        class SplitModel2(torch.nn.Module):
            # 定义 forward 方法，处理输入数据的按维度分割操作
            def forward(self, input):
                # 返回输入数据按倒数第二维度进行分割后的结果及另一种分割结果的最后一部分
                return input.split([2, 1, 1], -2), input.split([2, 2], -2)[-1]

        # 再次创建形状为 (5, 4, 3) 的随机张量 x
        x = torch.randn(5, 4, 3)
        # 运行测试，使用 SplitModel2 实例和 x 作为输入
        self.run_test(SplitModel2(), x)

        # 定义第三个继承自 torch.nn.Module 的内部类 SplitModel3
        class SplitModel3(torch.nn.Module):
            # 定义 forward 方法，处理输入数据的默认分割操作
            def forward(self, input):
                # 返回输入数据按默认分割大小分割后的结果
                return input.split([2, 1, 2])

        # 再次创建形状为 (5, 4, 3) 的随机张量 x
        x = torch.randn(5, 4, 3)
        # 运行测试，使用 SplitModel3 实例和 x 作为输入
        self.run_test(SplitModel3(), x)
    def test_split_script(self):
        # 定义一个名为 test_split_script 的测试方法
        class SplitModel(torch.nn.Module):
            # 定义一个名为 SplitModel 的内部类，继承自 torch.nn.Module
            def forward(self, input):
                # 定义 forward 方法，接受一个输入 input
                return input.split([2, 1, 2]), input.split([3, 2])[0]

        x = torch.randn(5, 4, 3)
        # 生成一个形状为 (5, 4, 3) 的随机张量 x
        self.run_test(SplitModel(), x)
        # 调用 self.run_test 方法，测试 SplitModel 类的 forward 方法对 x 的输出

        class SplitModel2(torch.nn.Module):
            # 定义另一个名为 SplitModel2 的内部类，继承自 torch.nn.Module
            def forward(self, input):
                # 定义 forward 方法，接受一个输入 input
                return input.split([2, 1, 1], -2), input.split([2, 2], -2)[-1]

        x = torch.randn(5, 4, 3)
        # 生成一个形状为 (5, 4, 3) 的随机张量 x
        self.run_test(SplitModel2(), x)
        # 调用 self.run_test 方法，测试 SplitModel2 类的 forward 方法对 x 的输出

        class SplitModel3(torch.nn.Module):
            # 定义另一个名为 SplitModel3 的内部类，继承自 torch.nn.Module
            def forward(self, input):
                # 定义 forward 方法，接受一个输入 input
                return input.split([2, 1, 2])

        x = torch.randn(5, 4, 3)
        # 生成一个形状为 (5, 4, 3) 的随机张量 x
        self.run_test(SplitModel3(), x)
        # 调用 self.run_test 方法，测试 SplitModel3 类的 forward 方法对 x 的输出

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_split_size_as_list(self):
        # 定义一个名为 test_split_size_as_list 的测试方法
        class SplitModel(torch.nn.Module):
            # 定义一个名为 SplitModel 的内部类，继承自 torch.nn.Module
            def forward(self, input, split_sizes: List[int]):
                # 定义 forward 方法，接受 input 和 split_sizes 作为参数
                out = []
                # 创建一个空列表 out
                split_list: List[Tensor] = input.split(split_sizes)
                # 使用 input.split(split_sizes) 将 input 按照 split_sizes 切分，得到一个张量列表 split_list

                for ob in split_list:
                    out.append(ob)  # noqa: PERF402
                    # 将 split_list 中的每个张量 ob 添加到 out 列表中
                return torch.cat(out, dim=0)
                # 返回通过 torch.cat 在维度 0 上拼接 out 列表中的张量

        x = torch.randn(6, 4, 3)
        # 生成一个形状为 (6, 4, 3) 的随机张量 x
        split_sizes = [torch.tensor(2), torch.tensor(4)]
        # 创建一个包含两个张量的列表 split_sizes
        self.run_test(SplitModel(), (x, split_sizes))
        # 调用 self.run_test 方法，测试 SplitModel 类的 forward 方法对 (x, split_sizes) 的输出

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_split_size_with_slice(self):
        # 定义一个名为 test_split_size_with_slice 的测试方法
        class SplitModule(torch.nn.Module):
            # 定义一个名为 SplitModule 的内部类，继承自 torch.nn.Module
            def forward(self, x, y, t):
                # 定义 forward 方法，接受 x, y, t 三个输入参数
                splits = (x.size(1), y.size(1))
                # 计算 splits 为 (x 的第二维大小, y 的第二维大小)
                out, out2 = torch.split(t, splits, dim=1)
                # 使用 torch.split 将 t 按照 splits 在维度 1 上切分，得到 out 和 out2
                return out, out2
                # 返回 out 和 out2

        x = torch.randn(2, 3)
        # 生成一个形状为 (2, 3) 的随机张量 x
        y = torch.randn(2, 4)
        # 生成一个形状为 (2, 4) 的随机张量 y
        t = torch.randn(2, 7)
        # 生成一个形状为 (2, 7) 的随机张量 t
        self.run_test(
            SplitModule(),
            (x, y, t),
            input_names=["x", "y", "t"],
            dynamic_axes={"x": [0, 1], "y": [0, 1], "t": [0, 1]},
        )
        # 调用 self.run_test 方法，测试 SplitModule 类的 forward 方法对 (x, y, t) 的输出，并指定输入名和动态轴

        self.run_test(SplitModule(), (x, y, t), remained_onnx_input_idx=[2])
        # 再次调用 self.run_test 方法，测试 SplitModule 类的 forward 方法对 (x, y, t) 的输出，并指定 remained_onnx_input_idx

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_split_dynamic(self):
        # 定义一个名为 test_split_dynamic 的测试方法
        class SplitModel(torch.jit.ScriptModule):
            # 定义一个名为 SplitModel 的脚本模块类
            @torch.jit.script_method
            def forward(self, input):
                # 定义 forward 方法，接受一个输入 input
                return input.split(2)[1]
                # 返回 input.split(2) 的第二个分片

        x = torch.randn(5, 4, 3)
        # 生成一个形状为 (5, 4, 3) 的随机张量 x
        self.run_test(SplitModel(), x)
        # 调用 self.run_test 方法，测试 SplitModel 类的 forward 方法对 x 的输出

        class SplitModel2(torch.jit.ScriptModule):
            # 定义另一个名为 SplitModel2 的脚本模块类
            @torch.jit.script_method
            def forward(self, input):
                # 定义 forward 方法，接受一个输入 input
                return input.split(2, -3)[1]
                # 返回 input.split(2, -3) 的第二个分片

        x = torch.randn(5, 4, 3)
        # 生成一个形状为 (5, 4, 3) 的随机张量 x
        self.run_test(SplitModel2(), x)
        # 调用 self.run_test 方法，测试 SplitModel2 类的 forward 方法对 x 的输出

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_split_dynamic_axes(self):
        # 定义一个名为 test_split_dynamic_axes 的测试方法
        class Split(torch.nn.Module):
            # 定义一个名为 Split 的内部类，继承自 torch.nn.Module
            def forward(self, x):
                # 定义 forward 方法，接受一个输入 x
                return x.split(1, dim=-1)
                # 返回在维度 -1 上将 x 切分的结果列表

        x = torch.randn(4, 384, 2)
        # 生成一个形状为 (4, 384, 2) 的随机张量 x
        input_names = ["logits"]
        # 定义一个名为 input_names 的列表，包含字符串 "logits"
        self.run_test(
            Split(),
            x,
            input_names=input_names,
            dynamic_axes={input_names[0]: {0: "batch"}},
        )
        # 调用 self.run_test 方法，测试 Split 类的 forward 方法对 x 的输出，并指定输入名和动态轴

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_chunk(self):
        # 定义一个用于测试的 ChunkModel 类，继承自 torch.nn.Module
        class ChunkModel(torch.nn.Module):
            def __init__(self, dim=1):
                super().__init__()
                self.dim = dim

            # 定义前向传播方法，使用 torch.chunk 对输入 x 进行分块操作
            def forward(self, x):
                return torch.chunk(x, 3, dim=self.dim)

        # 创建一个 ChunkModel 实例 model，并设置为评估模式
        model = ChunkModel()
        model.eval()
        
        # 创建另一个 ChunkModel 实例 model_neg_dim，传入参数 dim=-1，并设置为评估模式
        model_neg_dim = ChunkModel(-1)
        model_neg_dim.eval()
        
        # 生成一个形状为 (1, 18) 的随机张量 x
        x = torch.randn(1, 18)

        # 遍历 dim_size_ 在范围 [13, 16) 中的整数
        for dim_size_ in range(13, 16):
            # 生成一个形状为 (1, dim_size_) 的随机张量 y
            y = torch.randn(1, dim_size_)
            
            # 运行测试函数 run_test，对 model 进行测试，传入参数 x 和 y，并设定输入名称和动态轴
            self.run_test(
                model,
                x,
                additional_test_inputs=[y],
                input_names=["x"],
                dynamic_axes={"x": {0: "batch_size", 1: "dims"}},
            )

            # 运行测试函数 run_test，对 model_neg_dim 进行测试，传入参数 x 和 y，并设定输入名称和动态轴
            self.run_test(
                model_neg_dim,
                x,
                additional_test_inputs=[y],
                input_names=["x"],
                dynamic_axes={"x": {0: "batch_size", 1: "dims"}},
            )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dynamic_chunk(self):
        # 定义一个用于测试的 ChunkModel 类，继承自 torch.nn.Module
        class ChunkModel(torch.nn.Module):
            def __init__(self, dim=1):
                super().__init__()
                self.dim = dim

            # 定义前向传播方法，使用 torch.chunk 对输入 x 进行动态分块操作
            def forward(self, x):
                return torch.chunk(x, x.size(0), dim=self.dim)

        # 创建一个 ChunkModel 实例 model，并设置为评估模式
        model = ChunkModel()
        model.eval()
        
        # 创建另一个 ChunkModel 实例 model_neg_dim，传入参数 dim=-1，并设置为评估模式
        model_neg_dim = ChunkModel(-1)
        model_neg_dim.eval()
        
        # 生成一个形状为 (3, 18) 的随机张量 x
        x = torch.randn(3, 18)

        # 遍历 dim_size_ 在范围 [13, 16) 中的整数
        for dim_size_ in range(13, 16):
            # 生成一个形状为 (3, dim_size_) 的随机张量 y
            y = torch.randn(3, dim_size_)
            
            # 运行测试函数 run_test，对 model 进行测试，传入参数 x 和 y，并设定输入名称和动态轴
            self.run_test(
                model,
                x,
                additional_test_inputs=[y],
                input_names=["x"],
                dynamic_axes={"x": {0: "batch_size", 1: "dims"}},
            )

            # 运行测试函数 run_test，对 model_neg_dim 进行测试，传入参数 x 和 y，并设定输入名称和动态轴
            self.run_test(
                model_neg_dim,
                x,
                additional_test_inputs=[y],
                input_names=["x"],
                dynamic_axes={"x": {0: "batch_size", 1: "dims"}},
            )

    def test_concat(self):
        # 定义一个用于测试的 ConcatModel 类，继承自 torch.nn.Module
        class ConcatModel(torch.nn.Module):
            # 定义前向传播方法，对输入 x, y, z 进行 torch.cat 连接操作
            def forward(self, x, y, z):
                return torch.cat((x, y, z))

        # 生成形状分别为 (3, 4, 5), (1, 4, 5), (2, 4, 5) 的随机张量 x, y, z
        x = torch.randn(3, 4, 5)
        y = torch.randn(1, 4, 5)
        z = torch.randn(2, 4, 5)
        
        # 运行测试函数 run_test，对 ConcatModel 进行测试，传入参数 (x, y, z)
        self.run_test(ConcatModel(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_concat_dynamic(self):
        # 定义一个用于测试的 ConcatDynamicModel 类，继承自 torch.jit.ScriptModule
        class ConcatDynamicModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            # 定义 torch.jit 脚本方法的前向传播，使用 torch.cat 和 unbind 对输入 x 进行连接操作
            def forward(self, x):
                return torch.cat(x.unbind())

        # 生成形状为 (4, 5, 6) 的随机张量 x
        x = torch.randn(4, 5, 6)
        
        # 运行测试函数 run_test，对 ConcatDynamicModel 进行测试，传入参数 x
        self.run_test(ConcatDynamicModel(), x)

    def test_stack(self):
        # 定义一个用于测试的 StackModel 类，继承自 torch.nn.Module
        class StackModel(torch.nn.Module):
            # 定义前向传播方法，对输入 x, y, z 进行 torch.stack 操作，维度为 1
            def forward(self, x, y, z):
                return torch.stack((x, y, z), 1)

        # 生成形状分别为 (3, 4, 5) 的随机张量 x, y, z
        x = torch.randn(3, 4, 5)
        y = torch.randn(3, 4, 5)
        z = torch.randn(3, 4, 5)
        
        # 运行测试函数 run_test，对 StackModel 进行测试，传入参数 (x, y, z)
        self.run_test(StackModel(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_stack_dynamic(self):
        # 定义一个继承自 torch.jit.ScriptModule 的动态堆叠模型类
        class StackDynamicModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            # 实现模型的前向传播方法
            def forward(self, x):
                # 使用 torch.stack 将输入张量 x 沿着第一维度堆叠
                return torch.stack(x.unbind(), 1)

        # 创建一个形状为 (4, 5, 6) 的随机张量 x
        x = torch.randn(4, 5, 6)
        # 运行测试，评估 StackDynamicModel 对象在输入 x 上的表现
        self.run_test(StackDynamicModel(), x)

    def test_loop_dynamic(self):
        # 定义一个继承自 torch.jit.ScriptModule 的动态循环模型类
        class LoopModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            # 实现模型的前向传播方法
            def forward(self, x):
                # 对输入张量 x 的第三维度进行循环
                for i in range(x.size(2)):
                    # 在每次迭代中，将 i 加到 x 上
                    x = x + i
                return x

        # 创建一个形状为 (1, 2, 3)、数据类型为 long 的零张量 inputs
        model = LoopModel()
        inputs = torch.zeros(1, 2, 3, dtype=torch.long)
        # 运行测试，评估 LoopModel 对象在输入 inputs 上的表现
        self.run_test(model, inputs)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_loop_nested(self):
        # 定义一个继承自 torch.jit.ScriptModule 的嵌套循环模型类
        class NestedLoopsModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            # 实现模型的前向传播方法
            def forward(self, x):
                # 外层循环，共 5 次
                for i in range(5):
                    a = 0
                    # 内层循环，直到 a 小于 4
                    while a < 4:
                        a += 1
                    # 在每次迭代中，将 a 加到 x 上
                    x = x + a
                return x

        # 创建一个形状为 (1, 2, 3)、数据类型为 long 的零张量 inputs
        model = NestedLoopsModel()
        inputs = torch.zeros(1, 2, 3, dtype=torch.long)
        # 运行测试，评估 NestedLoopsModel 对象在输入 inputs 上的表现
        self.run_test(model, inputs)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_loop_with_list(self):
        # 定义一个继承自 torch.jit.ScriptModule 的列表循环模型类
        class ListLoopModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            # 实现模型的前向传播方法
            def forward(self, x):
                # 初始化空列表 res、res1、res3、res4
                res = []
                res1 = []
                res3 = []
                res4 = []
                # 使用给定的分割参数对输入张量 x 进行分割
                arr = x.split([3, 4, 1, 1, 2, 3, 2], 0)
                # 创建一个形状为 (3, 4)、数据类型为 long 的零张量 res2
                res2 = torch.zeros(3, 4, dtype=torch.long)
                # 遍历分割后的张量列表 arr
                for i in range(len(arr)):
                    # 将 arr[i] 沿第一维度求和，结果加入 res 列表
                    res.append(arr[i].sum(0, False))
                    # 将 arr 中倒数第 i 个张量沿第一维度求和，结果加入 res1 列表
                    res1.append(arr[-1 - i].sum(0, False))
                    # res2 中的每个元素加 1
                    res2 += 1
                    # 将 arr[i] 沿第一维度求和，结果加入 res3 列表
                    res3 = res3 + [arr[i].sum(0, False)]
                    # 将 arr 中倒数第 i 个张量沿第一维度求和，结果加入 res4 列表
                    res4 += [arr[-1 - i].sum(0, False)]
                # 返回结果列表 res、res1、res2、res3 和 res4 组成的元组
                return res, res1, res2, torch.stack(res3), torch.stack(res4)

        # 创建一个形状为 (16,) 的随机张量 inputs
        model = ListLoopModel()
        inputs = torch.randn(16)
        # 运行测试，评估 ListLoopModel 对象在输入 inputs 上的表现
        self.run_test(model, inputs)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_loop_transpose(self):
        # 定义一个继承自 torch.nn.Module 的循环转置模型类
        class LoopModel(torch.nn.Module):
            # 实现模型的前向传播方法
            def forward(self, x):
                # 创建一个形状与 x[0] 相同的零张量 res
                res = torch.zeros_like(x[0])
                # 对输入张量 x 的第一维度进行循环
                for i in range(x.size(0)):
                    # 将 x[0] 在第一维度与第二维度进行转置后加到 res 上
                    res += x[0].transpose(0, 1)
                return res

        # 将 LoopModel 类实例化为 torch.jit.ScriptModule 类型的对象 model
        model = torch.jit.script(LoopModel())
        # 创建一个形状为 (5, 3, 3) 的随机张量 x
        x = torch.randn(5, 3, 3)
        # 运行测试，评估 LoopModel 对象在输入 x 上的表现
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_loop_multi_dim(self):
        # 定义一个继承自 torch.jit.ScriptModule 的模型类
        class LoopMultiDimModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            # 定义模型的前向传播方法，接受两个参数 x 和 y
            def forward(self, x, y):
                # 使用 torch.flip 对 x 在第一个维度上的前 7 个元素进行翻转
                for x_ in torch.flip(x.narrow(0, 0, 7), [0]):
                    # 更新 y 为 x_ 的第一个元素的 y 索引处的值
                    y = x_[0][y]
                return y

        # 创建 LoopMultiDimModel 的实例
        model = LoopMultiDimModel()
        # 生成一个 shape 为 (8, 1, 17)，数值在 [0, 5) 范围内的长整型张量 x
        x = torch.randint(0, 5, (8, 1, 17), dtype=torch.long)
        # 创建一个值为 1 的长整型张量 y
        y = torch.ones(1, dtype=torch.long)
        # 运行测试，验证模型的行为是否符合预期
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list(self):
        # 定义一个继承自 torch.jit.ScriptModule 的模型类
        class ListModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            # 定义模型的前向传播方法，接受一个参数 x
            def forward(self, x):
                # 使用 unbind() 方法解绑 x 的所有张量，得到一个张量列表 tensors
                tensors = x.unbind()
                # 初始化一个空列表 res
                res = []
                # 将 tensors 的第一个张量添加到 res 中
                res.append(tensors[0])
                # 将 res 中索引为 1 的元素弹出
                res.pop(1)

                # 在 res 的开头插入 tensors 的索引为 1 的张量
                res.insert(0, tensors[1])
                # 将 tensors 的索引为 2 的张量追加到 res 的末尾
                res.append(tensors[2])
                # 将 tensors 的索引为 3 和 4 的张量扩展到 res 的末尾
                res += [tensors[3], tensors[4]]
                # 将 res 和包含 tensors 索引为 5 的张量的列表相加
                res = res + [tensors[5]]
                # 返回一个长度为 res 的全为 1 的张量
                return torch.ones(len(res))

        # 创建 ListModel 的实例
        model = ListModel()
        # 生成一个 shape 为 (16, 1) 的随机张量 inputs
        inputs = torch.randn(16, 1)
        # 运行测试，验证模型的行为是否符合预期
        self.run_test(model, inputs)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_append(self):
        # 定义一个继承自 torch.nn.Module 的模型类
        class ListModel(torch.nn.Module):
            # 定义模型的前向传播方法，接受两个参数 x 和 y
            def forward(self, x, y):
                # 初始化一个空列表 res
                res = []
                # 对 x 的第一个维度进行迭代，范围是 x 的大小
                for i in range(x.size(0)):
                    # 将 torch.matmul(x[i], y) 的结果追加到 res 中
                    res += [torch.matmul(x[i], y)]
                # 返回 res 列表
                return res

        # 使用 torch.jit.script 将 ListModel 转换为脚本模式
        model = torch.jit.script(ListModel())
        # 生成一个 shape 为 (16, 3, 4) 的随机张量 x
        x = torch.randn(16, 3, 4)
        # 生成一个 shape 为 (4, 5) 的随机张量 y
        y = torch.randn(4, 5)
        # 运行测试，验证模型的行为是否符合预期
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_append_nested(self):
        # 定义一个继承自 torch.nn.Module 的模型类
        class ListModel(torch.nn.Module):
            # 定义模型的前向传播方法，接受两个参数 x 和 y
            def forward(self, x, y):
                # 初始化一个空列表 res
                res = []
                # 对 x 的第一个和第二个维度进行嵌套迭代，范围是 x 的大小
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        # 将 torch.matmul(x[i][j], y) 的结果追加到 res 中
                        res += [torch.matmul(x[i][j], y)]
                # 返回 res 列表
                return res

        # 使用 torch.jit.script 将 ListModel 转换为脚本模式
        model = torch.jit.script(ListModel())
        # 生成一个 shape 为 (4, 4, 3, 4) 的随机张量 x
        x = torch.randn(4, 4, 3, 4)
        # 生成一个 shape 为 (4, 5) 的随机张量 y
        y = torch.randn(4, 5)
        # 运行测试，验证模型的行为是否符合预期
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(14)  # Need onnx::Identity of sequence in opset 14
    def test_list_append_nested_2(self):
        # 定义一个继承自 torch.nn.Module 的模型类
        class ListModel(torch.nn.Module):
            # 定义模型的前向传播方法，接受一个参数 x
            def forward(self, x):
                # 初始化一个空列表 res
                res = []
                # 初始化一个空列表 res_replicate
                res_replicate = []
                # 对 x 的第一个维度进行迭代，范围是 x 的大小
                for i in range(x.size(0)):
                    # 如果 res 的长度大于 2
                    if len(res) > 2:
                        # 对 x 的第二个维度进行迭代，范围是 x 的大小
                        for j in range(x.size(1)):
                            # 将 x[i][j] 追加到 res 中
                            res.append(x[i][j])
                        # 将 res 的最后一个元素追加到 res_replicate 中
                        res_replicate.append(res[-1])
                        # 将 res_replicate 的最后一个元素追加到 res 中
                        res.append(res_replicate[-1])
                # 返回 res 和 res_replicate 列表
                return res, res_replicate

        # 使用 torch.jit.script 将 ListModel 转换为脚本模式
        model = torch.jit.script(ListModel())
        # 生成一个 shape 为 (4, 4, 3, 4) 的随机张量 x
        x = torch.randn(4, 4, 3, 4)
        # 运行测试，验证模型的行为是否符合预期
        self.run_test(model, (x,))
    def test_list_append_nested_mixed_dtype(self):
        # 定义一个继承自 torch.nn.Module 的内嵌类 ListModel
        class ListModel(torch.nn.Module):
            # 定义模型的前向传播方法，接受两个参数 x 和 y
            def forward(self, x, y):
                # 初始化一个空列表 res，用于存储计算结果
                res = []
                # 循环遍历 x 的第一维
                for i in range(x.size(0)):
                    # 再次循环遍历 x 的第二维
                    for j in range(x.size(1)):
                        # 如果 i 等于 j，执行 x == y 的计算并添加到 res
                        if i == j:
                            res.append(x == y)
                        # 否则执行 x != y 的计算并添加到 res
                        else:
                            res.append(x != y)
                # 返回最终的结果列表 res
                return res

        # 使用 torch.jit.script 方法对 ListModel 进行脚本化
        model = torch.jit.script(ListModel())
        # 生成随机张量 x 和 y，作为输入参数进行测试
        x = torch.randn(4, 4, 3, 4)
        y = torch.randn(3, 4)
        # 调用 self.run_test 方法执行模型测试
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_pop(self):
        # 定义一个继承自 torch.nn.Module 的内嵌类 ListModel
        class ListModel(torch.nn.Module):
            # 定义模型的前向传播方法，接受两个参数 x 和 y
            def forward(self, x, y):
                # 初始化一个空列表 res，用于存储计算结果
                res = []
                # 循环遍历 x 的第一维
                for i in range(x.size(0)):
                    # 将 torch.matmul(x[i], y) 的结果添加到 res 中
                    res += [torch.matmul(x[i], y)]
                # 移除 res 中的最后一个元素
                res.pop()
                # 返回最终的结果列表 res
                return res

        # 使用 torch.jit.script 方法对 ListModel 进行脚本化
        model = torch.jit.script(ListModel())
        # 生成随机张量 x 和 y，作为输入参数进行测试
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        # 调用 self.run_test 方法执行模型测试
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_pop_nested(self):
        # 定义一个继承自 torch.nn.Module 的内嵌类 ListModel
        class ListModel(torch.nn.Module):
            # 定义模型的前向传播方法，接受两个参数 x 和 y
            def forward(self, x, y):
                # 初始化一个空列表 res，用于存储计算结果
                res = []
                # 循环遍历 x 的第一维
                for i in range(x.size(0)):
                    # 再次循环遍历 x 的第二维
                    for j in range(x.size(1)):
                        # 将 torch.matmul(x[i][j], y) 的结果添加到 res 中
                        res += [torch.matmul(x[i][j], y)]
                        # 移除 res 中的最后一个元素
                        res.pop()
                    # 将 torch.matmul(x[i][0], y) 的结果添加到 res 中
                    res += [torch.matmul(x[i][0], y)]
                # 返回最终的结果列表 res
                return res

        # 使用 torch.jit.script 方法对 ListModel 进行脚本化
        model = torch.jit.script(ListModel())
        # 生成随机张量 x 和 y，作为输入参数进行测试
        x = torch.randn(4, 4, 3, 4)
        y = torch.randn(4, 5)
        # 调用 self.run_test 方法执行模型测试
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_del(self):
        # 定义一个继承自 torch.nn.Module 的内嵌类 ListModel
        class ListModel(torch.nn.Module):
            # 定义模型的前向传播方法，接受两个参数 x 和 y
            def forward(self, x, y):
                # 初始化一个空列表 res，用于存储计算结果
                res = []
                # 循环遍历 x 的第一维
                for i in range(x.size(0)):
                    # 将 torch.matmul(x[i], y) 的结果添加到 res 中
                    res += [torch.matmul(x[i], y)]
                # 删除 res 中索引为 2 的元素
                del res[2]
                # 返回最终的结果列表 res
                return res

        # 使用 torch.jit.script 方法对 ListModel 进行脚本化
        model = torch.jit.script(ListModel())
        # 生成随机张量 x 和 y，作为输入参数进行测试
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        # 调用 self.run_test 方法执行模型测试
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_del_nested(self):
        # 定义一个继承自 torch.nn.Module 的内嵌类 ListModel
        class ListModel(torch.nn.Module):
            # 定义模型的前向传播方法，接受两个参数 x 和 y
            def forward(self, x, y):
                # 初始化一个空列表 res，用于存储计算结果
                res = []
                # 循环遍历 x 的第一维
                for i in range(x.size(0)):
                    # 再次循环遍历 x 的第二维
                    for j in range(x.size(1)):
                        # 将 torch.matmul(x[i][j], y) 的结果添加到 res 中
                        res += [torch.matmul(x[i][j], y)]
                        # 删除 res 中索引为 i 的元素
                        del res[i]
                    # 将 torch.matmul(x[i][0], y) 的结果添加到 res 中
                    res += [torch.matmul(x[i][0], y)]
                # 返回最终的结果列表 res
                return res

        # 使用 torch.jit.script 方法对 ListModel 进行脚本化
        model = torch.jit.script(ListModel())
        # 生成随机张量 x 和 y，作为输入参数进行测试
        x = torch.randn(4, 4, 3, 4)
        y = torch.randn(4, 5)
        # 调用 self.run_test 方法执行模型测试
        self.run_test(model, (x, y))
    def test_list_set(self):
        # 定义一个继承自 torch.nn.Module 的内部类 ListModel
        class ListModel(torch.nn.Module):
            # 定义模型的前向传播函数
            def forward(self, x, y):
                # 初始化一个空列表 res
                res = []
                # 遍历 x 的第一维
                for i in range(x.size(0)):
                    # 将 x 中第 i 行的数据添加到 res 中
                    res.append(x[i])
                # 将 x 中第 y 行的数据覆盖 res 中索引为 y 的位置
                res[y] = x[y]
                # 返回结果列表 res
                return res

        # 使用 torch.jit.script 方法将 ListModel 转换为 TorchScript
        model = torch.jit.script(ListModel())
        # 生成一个大小为 (12, 4) 的随机张量 x
        x = torch.randn(12, 4)
        # 创建一个值为 2 的长整型张量 y
        y = torch.tensor(2, dtype=torch.long)
        # 调用 run_test 方法运行模型测试
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_idx_sum(self):
        # 定义一个继承自 torch.nn.Module 的内部类 ListModel
        class ListModel(torch.nn.Module):
            # 定义模型的前向传播函数
            def forward(self, x, y):
                # 使用 torch.arange 生成从 0 到 x 的第一维大小的张量 indices
                indices = torch.arange(x.size(0))
                # 初始化一个空列表 res
                res = []
                # 遍历 x 的第一维
                for i in range(x.size(0)):
                    # 将 x 中第 i 行的数据添加到 res 中
                    res.append(x[i])
                # 返回 res 中索引为 torch.sum(indices[:y]) 的元素
                return res[torch.sum(indices[:y])]

        # 使用 torch.jit.script 方法将 ListModel 转换为 TorchScript
        model = torch.jit.script(ListModel())
        # 生成一个大小为 (12, 4) 的随机张量 x
        x = torch.randn(12, 4)
        # 创建一个值为 2 的长整型张量 y
        y = torch.tensor(2, dtype=torch.long)
        # 调用 run_test 方法运行模型测试
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_tensor_factories(self):
        # 定义一个继承自 torch.nn.Module 的内部类 TensorFactory
        class TensorFactory(torch.nn.Module):
            # 定义模型的前向传播函数
            def forward(self, x):
                # 返回一个与 x 大小相同的全零张量加上全一张量
                return torch.zeros(x.size()) + torch.ones(x.size())

        # 生成一个大小为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 调用 run_test 方法运行模型测试，指定输入名称和动态轴
        self.run_test(
            TensorFactory(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        # 再次调用 run_test 方法运行模型测试，不保留 ONNX 输入索引
        self.run_test(TensorFactory(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_tensor_factories_script(self):
        # 定义一个继承自 torch.jit.ScriptModule 的内部类 TensorFactory
        class TensorFactory(torch.jit.ScriptModule):
            @torch.jit.script_method
            # 定义 TorchScript 方法的前向传播函数
            def forward(self, x):
                # 返回一个与 x 相同形状和数据类型的全零张量加上全一张量
                return torch.zeros(x.shape, dtype=torch.float) + torch.ones(
                    x.shape, dtype=torch.float
                )

        # 生成一个大小为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 调用 run_test 方法运行模型测试，指定输入名称和动态轴
        self.run_test(
            TensorFactory(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        # 再次调用 run_test 方法运行模型测试，不保留 ONNX 输入索引
        self.run_test(TensorFactory(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_tensor_like_factories_script(self):
        # 定义一个继承自 torch.jit.ScriptModule 的内部类 TensorFactory
        class TensorFactory(torch.jit.ScriptModule):
            @torch.jit.script_method
            # 定义 TorchScript 方法的前向传播函数
            def forward(self, x):
                # 使用 torch.zeros_like 和 torch.ones_like 分别生成与 x 相同大小和数据类型的全零和全一张量
                zeros = torch.zeros_like(
                    x,
                    dtype=torch.float,
                    layout=torch.strided,
                    device=torch.device("cpu"),
                )
                ones = torch.ones_like(
                    x,
                    dtype=torch.float,
                    layout=torch.strided,
                    device=torch.device("cpu"),
                )
                # 返回 zeros 和 ones 张量的和
                return zeros + ones

        # 生成一个大小为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 调用 run_test 方法运行模型测试，指定输入名称和动态轴
        self.run_test(
            TensorFactory(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        # 再次调用 run_test 方法运行模型测试，不保留 ONNX 输入索引
        self.run_test(TensorFactory(), x, remained_onnx_input_idx=[])
    # 定义一个名为 test_tensor_split 的测试方法
    def test_tensor_split(self):
        # 定义一个内部类 TensorSplitModel，继承自 torch.nn.Module
        class TensorSplitModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, input):
                # 返回多个张量分割的结果
                return (
                    input.tensor_split([1, 3]),  # 对输入进行张量分割，分割点为 [1, 3]
                    # 在输出索引上进行测试
                    input.tensor_split([2, 4])[0],  # 对输入进行张量分割，并取索引为 0 的结果
                    # 在特定维度上进行张量分割
                    input.tensor_split([1, 3, 4], dim=-2),  # 对输入在指定维度 -2 进行张量分割
                    # 在特定维度上进行张量分割，并取索引为 -1 的结果
                    input.tensor_split([0, 2], dim=-2)[-1],  # 对输入在指定维度 -2 进行张量分割，并取索引为 -1 的结果
                    # 测试超出边界的结束索引 (5)
                    input.tensor_split([2, 3, 5]),  # 对输入进行张量分割，分割点为 [2, 3, 5]
                )

        # 运行 TensorSplitModel 的测试
        self.run_test(TensorSplitModel(), torch.randn(5, 4, 3))

    # 带有条件跳过装饰器的测试方法，最小支持操作集版本为 13
    @skipIfUnsupportedMinOpsetVersion(13)
    def test_tensor_split_scalar(self):
        # 定义一个内部类 TensorSplitModel，继承自 torch.nn.Module
        class TensorSplitModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 返回张量 x 按照其第二维度进行等分
                return torch.tensor_split(x, x.size(1))

        # 运行 TensorSplitModel 的测试
        self.run_test(TensorSplitModel(), torch.randn(1, 2, 3))

    # 带有条件跳过装饰器的测试方法，最小支持操作集版本为 13
    @skipIfUnsupportedMinOpsetVersion(13)
    def test_tensor_split_dynamic_axes(self):
        # 定义一个内部类 TensorSplitModel，继承自 torch.nn.Module
        class TensorSplitModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 返回张量 x 按照其最后一个维度进行等分
                return x.tensor_split(1, dim=-1)

        x = torch.randn(4, 384, 2)
    # 定义一个测试方法，测试返回全零的张量
    def test_new_zeros(self):
        # 定义一个继承自torch.nn.Module的内部类Zero_
        class Zero_(torch.nn.Module):
            # 重写forward方法，接受输入x并返回x的零张量
            def forward(self, x):
                # 返回x形状第1维到第2维的全零张量，以及x形状第2维到最后的全零张量（数据类型为torch.long）
                return x.new_zeros(x.shape[1:2]), x.new_zeros(
                    x.shape[2:], dtype=torch.long
                )

        # 创建一个形状为(2, 3, 4)的随机张量x
        x = torch.randn(2, 3, 4)
        # 运行测试函数run_test，测试Zero_模型在输入x上的表现，指定输入名称"x"和动态轴"x"的维度
        self.run_test(Zero_(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        # 再次运行测试函数，测试Zero_模型在输入x上的表现，指定remained_onnx_input_idx为空列表
        self.run_test(Zero_(), x, remained_onnx_input_idx=[])

    # 根据不支持的最小Opset版本（版本号>=9），跳过该测试方法
    @skipIfUnsupportedMinOpsetVersion(9)
    # 定义一个测试方法，测试返回全零的张量，并指定数据类型
    def test_new_zeros_with_dtype(self):
        # 定义一个继承自torch.nn.Module的内部类MyModel
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个包含50个单词，每个单词64维的嵌入层
                self.emb = torch.nn.Embedding(50, 64)

            # 重写forward方法，接受输入x并返回嵌入inp的结果
            def forward(self, x):
                # 返回与输入x相同形状的全零张量inp，并使用该张量作为嵌入层的输入
                inp = x.new_zeros(x.shape)
                return self.emb(inp)

        # 创建MyModel的实例model
        model = MyModel()
        # 创建一个形状为(2, 3)的整型张量x，并将其转换为torch.int64类型
        x = torch.Tensor([[2, 5, 6], [3, 2, 5]]).to(torch.int64)
        # 运行测试函数run_test，测试model在输入x上的表现，指定输入名称"x"和动态轴"x"的前两维
        self.run_test(model, x, input_names=["x"], dynamic_axes={"x": [0, 1]})

    # 根据不支持的最小Opset版本（版本号>=9），跳过该测试方法
    @skipIfUnsupportedMinOpsetVersion(9)
    # 定义一个测试方法，测试返回全一的张量
    def test_new_ones(self):
        # 定义一个继承自torch.nn.Module的内部类OnesModel
        class OnesModel(torch.nn.Module):
            # 重写forward方法，接受输入x并返回x形状的全一张量
            def forward(self, x):
                # 返回x形状第1维到第2维的全一张量，以及x形状第2维到最后的全一张量（数据类型为torch.long）
                return x.new_ones(x.shape[1:2]), x.new_ones(
                    x.shape[2:], dtype=torch.long
                )

        # 创建一个形状为(2, 3, 4)的随机张量x
        x = torch.randn(2, 3, 4)
        # 运行测试函数run_test，测试OnesModel模型在输入x上的表现，指定输入名称"x"和动态轴"x"的维度
        self.run_test(OnesModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        # 再次运行测试函数，测试OnesModel模型在输入x上的表现，指定remained_onnx_input_idx为空列表
        self.run_test(OnesModel(), x, remained_onnx_input_idx=[])

    # 根据不支持的最小Opset版本（版本号>=9），跳过该测试方法
    @skipIfUnsupportedMinOpsetVersion(9)
    @skipScriptTest()  # torch.zeros/torch.ones with size tensor of dim != 0 not scriptable.
    # 定义一个测试方法，测试根据输入张量返回全零和全一张量
    def test_zeros_ones_with_tensor_input(self):
        # 定义一个继承自torch.nn.Module的内部类ZeroAndOnes
        class ZeroAndOnes(torch.nn.Module):
            # 重写forward方法，接受输入x并返回全零和全一张量
            def forward(self, x):
                # 返回根据输入x的全零张量和全一张量
                return torch.zeros(x, 1), torch.ones(x, 1)

        # 创建一个形状为[2]的整型张量x
        x = torch.tensor([2])
        # 运行测试函数run_test，测试ZeroAndOnes模型在输入(x,)上的表现
        self.run_test(ZeroAndOnes(), (x,))

    # 根据不支持的最小Opset版本（版本号>=9），跳过该测试方法
    @skipIfUnsupportedMinOpsetVersion(9)
    @skipShapeChecking
    # 定义一个测试方法，测试将张量转换为Python列表
    def test_tolist(self):
        # 定义一个继承自torch.jit.ScriptModule的内部类List
        class List(torch.jit.ScriptModule):
            @torch.jit.script_method
            # 定义一个torch.jit脚本方法forward，接受输入input并返回转换为列表的结果
            def forward(self, input):
                # 将输入input转换为整型列表res
                res: List[int] = input.tolist()
                return res

        # 运行测试函数run_test，测试List模型在输入torch.randint(100, (1,))上的表现
        self.run_test(List(), (torch.randint(100, (1,)),))
    def test_list_pass(self):
        # 定义一个名为 Slice 的 Torch 模块，实现了 forward 方法
        class Slice(torch.nn.Module):
            def forward(self, x, y):
                # 返回一个与 x 形状相关的全零张量，形状为 x 的后三个维度与 y 的后两个维度
                return x.new_zeros(x.shape[2:] + y.shape[1:])

        # 创建两个随机张量 x 和 y
        x = torch.randn(2, 3, 4, 5)
        y = torch.randn(1, 2, 3, 4)
        # 运行测试，传入 Slice 模块，输入为 x 和 y，定义输入名和动态轴
        self.run_test(
            Slice(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2, 3], "y": [0, 1, 2, 3]},
        )
        # 再次运行测试，传入 Slice 模块和输入 x、y，不指定保留的 ONNX 输入索引
        self.run_test(Slice(), (x, y), remained_onnx_input_idx=[])

        # 定义一个名为 Size 的 Torch 模块，实现了 forward 方法
        class Size(torch.nn.Module):
            def forward(self, x, y):
                # 返回一个与 x 和 y 形状相关的全零张量，形状为 x 和 y 的形状拼接
                return x.new_zeros(x.shape + y.shape)

        # 创建两个随机张量 x 和 y
        x = torch.randn(2, 3, 4)
        y = torch.randn(1, 2, 3)
        # 运行测试，传入 Size 模块，输入为 x 和 y，定义输入名和动态轴
        self.run_test(
            Size(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2], "y": [0, 1, 2]},
        )
        # 再次运行测试，传入 Size 模块和输入 x、y，不指定保留的 ONNX 输入索引
        self.run_test(Size(), (x, y), remained_onnx_input_idx=[])

        # 定义一个名为 Array 的 Torch 模块，实现了 forward 方法
        class Array(torch.nn.Module):
            def forward(self, x, y):
                # 计算数组 arr1 和 arr2，然后返回一个与 x 和 y 形状相关的全零张量
                arr1 = [x.shape[0], x.shape[1], 2]
                arr2 = [y.shape[0], y.shape[1]]
                return x.new_zeros(arr1 + arr2)

        # 创建两个随机张量 x 和 y
        x = torch.randn(2, 3, 4)
        y = torch.randn(1, 2, 3)
        # 运行测试，传入 Array 模块，输入为 x 和 y，定义输入名和动态轴
        self.run_test(
            Array(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2], "y": [0, 1, 2]},
        )
        # 再次运行测试，传入 Array 模块和输入 x、y，不指定保留的 ONNX 输入索引
        self.run_test(Array(), (x, y), remained_onnx_input_idx=[])

        # 定义一个名为 List 的 Torch 模块，实现了 forward 方法
        class List(torch.nn.Module):
            def forward(self, x, y):
                # 获取 x 和 y 的形状作为列表 l1 和 l2，返回一个与 l1 和 l2 拼接后形状相关的全零张量
                l1 = list(x.shape)
                l2 = list(y.shape)
                return x.new_zeros(l1 + l2)

        # 创建两个随机张量 x 和 y
        x = torch.randn(2, 3, 4)
        y = torch.randn(1, 2, 3)
        # 运行测试，传入 List 模块，输入为 x 和 y，定义输入名和动态轴
        self.run_test(
            List(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2], "y": [0, 1, 2]},
        )
        # 再次运行测试，传入 List 模块和输入 x、y，不指定保留的 ONNX 输入索引
        self.run_test(List(), (x, y), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_new_empty(self):
        # 定义一个名为 Emtpy 的 Torch 模块，实现了 forward 方法
        class Emtpy(torch.nn.Module):
            def forward(self, x):
                # 返回两个新的空张量，一个用零填充，一个用零乘以零填充（类型为长整型）
                return (
                    x.new_empty(x.shape[0]).fill_(0),
                    x.new_empty(x.shape[0], dtype=torch.long) * 0,
                )

        # 创建一个随机张量 x
        x = torch.randn(2, 3, 4)
        # 运行测试，传入 Emtpy 模块和输入 x，定义输入名和动态轴
        self.run_test(Emtpy(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        # 再次运行测试，传入 Emtpy 模块和输入 x，不指定保留的 ONNX 输入索引
        self.run_test(Emtpy(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_new_full(self):
        # 定义一个名为 Full 的 Torch 模块，实现了 forward 方法
        class Full(torch.nn.Module):
            def forward(self, x):
                # 返回两个新的填充张量，一个用 5 填充，一个用 1.3（类型为长整型）填充
                return x.new_full(x.shape[1:2], 5), x.new_full(
                    x.shape[0:1], 1.3, dtype=torch.long
                )

        # 创建一个随机张量 x
        x = torch.randn(2, 3, 4)
        # 运行测试，传入 Full 模块和输入 x，定义输入名和动态轴
        self.run_test(Full(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        # 再次运行测试，传入 Full 模块和输入 x，不指定保留的 ONNX 输入索引
        self.run_test(Full(), x, remained_onnx_input_idx=[])
    def test_inplace_list(self):
        class Arithmetic(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x, y):
                # 使用 torch.cat 将 x 增加 3 后与 y.fill_(0) 进行拼接并返回
                return torch.cat([x.add_(3), y.fill_(0)])

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        self.run_test(
            Arithmetic(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1], "y": [0, 1]},
        )
        # 调用 run_test 方法进行测试，验证 Arithmetic 模型的输出
        self.run_test(Arithmetic(), (x, y), remained_onnx_input_idx=[0])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_inplace_fill(self):
        class Fill_(torch.nn.Module):
            def forward(self, x):
                # 使用 x.fill_(3) 在原地填充张量 x，并返回填充后的结果以及未改变的 x
                return x.fill_(3), x

        x = torch.randn(2, 3, 4)
        # 调用 run_test 方法测试 Fill_ 模型的输出
        self.run_test(Fill_(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        # 再次调用 run_test 方法，验证不保留任何 ONNX 输入索引的情况
        self.run_test(Fill_(), x, remained_onnx_input_idx=[])

    def test_inplace_arithmetic(self):
        class Arithmetic(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x, y):
                # 在原地对张量 x 添加 3，并将 y 乘以修改后的 x 返回
                x.add_(3)
                y.mul_(x)
                return x, y

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        # 调用 run_test 方法测试 Arithmetic 模型的输出
        self.run_test(Arithmetic(), (x, y))

    def test_inplace_arithmetic_half(self):
        class InplaceAddModel(torch.nn.Module):
            def forward(self, x, y):
                # 在原地将 x 加上 y 并返回
                return x.add_(y)

        class InplaceMulModel(torch.nn.Module):
            def forward(self, x, y):
                # 在原地将 x 乘以 y 并返回
                return x.mul_(y)

        x = torch.randn(2, 2, dtype=torch.half)
        y = torch.randn(2, 2, dtype=torch.float)
        # 调用 run_test 方法测试 InplaceAddModel 和 InplaceMulModel 模型的输出，设置相对和绝对误差容限
        self.run_test(InplaceAddModel(), (x, y), rtol=1e-2, atol=1e-2)
        self.run_test(InplaceMulModel(), (x, y), rtol=1e-2, atol=1e-2)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_inplace_with_loop(self):
        class M(torch.nn.Module):
            def forward(self, x):
                a = torch.ones(
                    12,
                )
                for i in range(10):
                    # 在原地将 a 加上一个全为 1 的张量，并返回结果
                    a.add_(
                        torch.ones(
                            12,
                        )
                    )
                return a + x

        m = M()
        x = torch.randn(
            12,
        )
        # 调用 torch.jit.script 将模型 M 脚本化，然后调用 run_test 方法测试其输出
        self.run_test(torch.jit.script(M()), (x))
    def test_inplace_with_loop_2(self):
        # 定义一个内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 定义 forward 方法，接受输入 x
            def forward(self, x):
                # 初始化一个包含 12 个元素的全为 1 的张量 _bias
                _bias = torch.ones(
                    12,
                )
                # 初始化一个包含 12 个元素的全为 1 的张量 a，并在循环中被改变
                a = torch.ones(
                    12,
                )  # 在循环中被使用，并且会被修改
                # 创建 a 的引用 a_ref，虽然未在循环中使用，但应该被改变
                a_ref = a  # 未在循环中使用，应该被修改
                # 克隆输入张量 x，并赋值给 b，在循环中使用但不被修改
                b = x.clone()  # 在循环中使用，但不会被修改
                # 创建 b 的引用 b_ref，虽然未在循环中使用，但不应该被修改
                b_ref = b  # 未在循环中使用，不应该被修改
                # 循环 10 次
                for i in range(10):
                    # 当 i 等于 3 时执行以下操作
                    if i == 3:
                        # 内部循环 5 次
                        for j in range(5):
                            # a 加上 _bias
                            a += _bias
                            # _bias 自身加上一个全为 1 的张量
                            _bias.add_(
                                torch.ones(
                                    12,
                                )
                            )
                            # b 加上一个全为 1 的张量
                            b = b + torch.ones(
                                12,
                            )

                    # _bias 自身加上一个全为 1 的张量
                    _bias.add_(
                        torch.ones(
                            12,
                        )
                    )
                    # a 加上 _bias
                    a += _bias

                # TODO: value for a_ref is incorrect.
                # a_ref += torch.ones(12,)
                # b_ref 加上一个全为 1 的张量
                b_ref += torch.ones(
                    12,
                )
                # 返回 _bias 与 x 的和，以及 a、b、b_ref
                return _bias + x, a, b, b_ref

        # 创建 M 类的实例 m
        m = M()
        # 创建一个全为 0 的张量 x，作为输入
        x = torch.zeros(
            12,
        )
        # 运行测试函数 run_test，对 torch.jit.script(M()) 进行测试
        self.run_test(torch.jit.script(M()), (x))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_inplace_attr_with_loop(self):
        # 定义一个内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 构造函数，初始化时生成一个从 0 到 11 的张量 _bias
            def __init__(self):
                super().__init__()
                self._bias = torch.arange(
                    12,
                )

            # 定义 forward 方法，接受输入 x
            def forward(self, x):
                # 重新生成一个从 0 到 11 的张量 _bias
                self._bias = torch.arange(
                    12,
                )
                # 循环 10 次
                for i in range(10):
                    # 当 i 等于 3 时执行以下操作
                    if i == 3:
                        # 内部循环 5 次
                        for j in range(5):
                            # self._bias 加上一个从 0 到 11 的张量
                            self._bias += torch.arange(
                                12,
                            )
                # 返回 self._bias 与 x 的和
                return self._bias + x

        # 创建 M 类的实例 m
        m = M()
        # 创建一个全为 0 的张量 x，作为输入
        x = torch.zeros(
            12,
        )
        # 运行测试函数 run_test，对 torch.jit.script(M()) 进行测试
        self.run_test(torch.jit.script(M()), (x))
    # 定义一个测试函数，用于测试具有循环的原位属性复制
    def test_inplace_attr_copy_with_loop(self):
        # 定义一个继承自 torch.nn.Module 的模型类 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个名为 _bias 的张量，其值为从0开始的12个元素
                self._bias = torch.arange(
                    12,
                )

            # 前向传播函数，接受输入 x
            def forward(self, x):
                # 重新设置 _bias 为从0开始的12个元素
                self._bias = torch.arange(
                    12,
                )
                # 开始循环，重复10次
                for i in range(10):
                    # 如果 i 等于 3，则执行以下操作
                    if i == 3:
                        # 再次进入循环，重复5次
                        for j in range(5):
                            # 将 _bias 的副本更新为从0开始的12个元素
                            self._bias.copy_(
                                torch.arange(
                                    12,
                                )
                            )
                        # 将 _bias 的副本更新为其自身加上从0开始的12个元素
                        self._bias.copy_(
                            self._bias
                            + torch.arange(
                                12,
                            )
                        )

                    # 将 _bias 的副本更新为其自身加上从0开始的12个元素
                    self._bias.copy_(
                        self._bias
                        + torch.arange(
                            12,
                        )
                    )
                # 返回 _bias 加上输入 x 的结果
                return self._bias + x

        # 创建 M 类的实例 m
        m = M()
        # 创建一个形状为 (12,) 的全零张量 x
        x = torch.zeros(
            12,
        )
        # 运行测试，使用 torch.jit.script 将模型 M 编译为 TorchScript，并传入参数 (x)
        self.run_test(torch.jit.script(M()), (x))

    # 声明一个需要跳过的测试函数，因为在 opset 14 中需要 onnx::Identity 来处理序列
    @skipIfUnsupportedMinOpsetVersion(14)
    # 定义一个测试函数，测试具有循环的原位序列操作
    def test_inplace_sequence_with_loop(self):
        # 定义一个继承自 torch.nn.Module 的模型类 M
        class M(torch.nn.Module):
            # 处理函数，接受 beam_hyps（张量列表）、done（布尔张量）和输入 x
            def process(self, beam_hyps: List[Tensor], done: Tensor, x):
                # 获取输入 x 的批处理大小
                batch_size = x.shape[0]
                # 开始循环，重复 batch_size 次
                for i in range(batch_size):
                    # 如果 done[i] 为真，则跳过本次循环
                    if done[i]:
                        continue

                    # 初始化 beam_idx 为 0
                    beam_idx = 0
                    # 遍历 x[i] 中的每个元素 token
                    for _, token in enumerate(x[i]):
                        # 将 token 添加到 beam_hyps 列表中
                        beam_hyps.append(token)
                        # beam_idx 加一
                        beam_idx += 1

                        # 如果 beam_idx 达到 6，则跳出内层循环
                        if beam_idx == 6:
                            break

                    # 更新 done[i]，判断 beam_hyps 列表的长度是否大于 4
                    done[i] = len(beam_hyps) > 4

                # 返回更新后的 beam_hyps 列表和 done 张量
                return beam_hyps, done

            # 前向传播函数，接受输入 x
            def forward(self, x):
                # 初始化 beam_hyps 为空的张量列表
                beam_hyps: List[Tensor] = []
                # 获取输入 x 的批处理大小
                batch_size = x.shape[0]
                # 初始化当前长度 cur_len 为 0，最大长度 max_len 为 x 的第一维长度
                cur_len = 0
                max_len = x.shape[1]
                # 初始化 done 张量为全零布尔张量，形状为 (batch_size,)
                done = torch.zeros(batch_size, dtype=torch.bool)
                # 当当前长度 cur_len 小于最大长度 max_len 时循环
                while cur_len < max_len:
                    # 调用 process 函数处理 beam_hyps、done 和 x[:, 0, :]，更新 beam_hyps 和 done
                    beam_hyps, done = self.process(beam_hyps, done, x[:, 0, :])
                    # 当前长度 cur_len 加一
                    cur_len = cur_len + 1

                # 返回 beam_hyps 列表作为输出
                return beam_hyps

        # 创建 M 类的 TorchScript 实例 m
        m = torch.jit.script(M())
        # 创建一个形状为 (8, 4, 3) 的随机张量 x
        x = torch.randn(8, 4, 3)
        # 运行测试，使用 torch.jit.script 将模型 M 编译为 TorchScript，并传入参数 (x)
        self.run_test(torch.jit.script(M()), (x))

    # 声明一个需要跳过的测试函数，因为在 ONNX 中不支持动态维度排序
    @skipScriptTest()
    # 定义一个测试函数，测试排序模型
    def test_sort(self):
        # 定义一个继承自 torch.nn.Module 的排序模型类 SortModel
        class SortModel(torch.nn.Module):
            # 前向传播函数，接受输入 x
            def forward(self, x):
                # 初始化一个空列表 out
                out = []
                # 开始循环，重复4次，范围为 -2 到 1
                for i in range(-2, 2):
                    # 将 x 按维度 i 进行降序排序，结果添加到 out 列表中
                    out.append(torch.sort(x, dim=i, descending=True))
                # 返回排序结果列表 out
                return out

        # 创建一个形状为 (3, 4) 的随机张量 x
        x = torch.randn(3, 4)
        # 运行测试，使用 SortModel 类测试排序，并传入参数 x
        self.run_test(SortModel(), x)

    # 声明一个需要跳过的测试函数，因为在 opset 11 中不支持某些操作
    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()  # 跳过脚本测试，因为 ONNX 不支持动态维度排序
    def test_sort_ascending(self):
        class SortModel(torch.nn.Module):
            def forward(self, x):
                out = []
                for i in range(-2, 2):
                    out.append(torch.sort(x, dim=i, descending=False))
                return out
        # 创建随机输入张量
        x = torch.randn(3, 4)
        # 运行测试，验证排序模型的输出
        self.run_test(SortModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_argsort(self):
        class ArgSortModel(torch.nn.Module):
            def forward(self, x):
                # 返回张量在指定维度上的升序索引
                return torch.argsort(x, dim=1, descending=False)
        # 创建随机输入张量
        x = torch.randn(3, 4)
        # 运行测试，验证 argsort 模型的输出
        self.run_test(ArgSortModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_masked_fill(self):
        class MaskedFillModel(torch.nn.Module):
            def forward(self, x):
                # 创建一个布尔掩码
                mask = torch.tensor([[0, 0, 1], [1, 1, 0]], dtype=torch.bool)
                # 使用指定值填充张量中被掩盖的位置
                return x.masked_fill(mask, 2)

        # 创建全零的张量，并设置其为可导
        x = torch.zeros(4, 2, 3, requires_grad=True)
        # 运行测试，验证 masked_fill 模型的输出
        self.run_test(MaskedFillModel(), x)

        class MaskedFillModel2(torch.nn.Module):
            def forward(self, x):
                # 使用条件语句掩盖张量中大于指定值的元素，并填充为新值
                return x.masked_fill(x > 3, -1)

        # 创建一个从 0 到 15 的张量，并转换为浮点型
        x = torch.arange(16).view(2, 2, 4).to(torch.float32)
        # 运行测试，验证 masked_fill 模型的输出
        self.run_test(MaskedFillModel2(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_masked_fill_inplace(self):
        class MaskedFillModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                # 创建一个布尔掩码并直接在原张量上进行填充操作
                mask = torch.tensor([[0, 0, 1], [1, 1, 0]], dtype=torch.bool)
                x.masked_fill_(mask, 2)
                return x

        # 创建全零的张量，并设置其为可导
        x = torch.zeros(4, 2, 3, requires_grad=True)
        # 运行测试，验证 inplace masked_fill 模型的输出
        self.run_test(MaskedFillModel(), x)

        class MaskedFillModel2(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                # 使用条件语句掩盖张量中大于指定值的元素，并直接在原张量上进行填充操作
                x.masked_fill_(x > 3, -1)
                return x

        # 创建一个从 0 到 15 的张量，并转换为浮点型
        x = torch.arange(16).view(2, 2, 4).to(torch.float32)
        # 运行测试，验证 inplace masked_fill 模型的输出
        self.run_test(MaskedFillModel2(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_masked_scatter(self):
        class MaskedScatterModel(torch.nn.Module):
            def forward(self, x):
                # 使用掩码将指定张量的选定位置替换为指定值
                return torch.masked_scatter(x, x.ge(0.5), torch.ones(100, 100) * 5)

        # 创建随机输入张量
        x = torch.randn(3, 4, 5, requires_grad=True)
        # 运行测试，验证 masked_scatter 模型的输出
        self.run_test(MaskedScatterModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_masked_select(self):
        class MaskedSelectModel(torch.nn.Module):
            def forward(self, x):
                # 根据掩码选取张量中满足条件的元素
                return torch.masked_select(x, x.ge(0.5))

        # 创建随机输入张量
        x = torch.randn(3, 4, 5, requires_grad=True)
        # 运行测试，验证 masked_select 模型的输出
        self.run_test(MaskedSelectModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_to_masked_fill(self):
        # 定义一个继承自 torch.nn.Module 的 MaskedFillModel 类
        class MaskedFillModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, input_mask, some_const):
                # 克隆输入的掩码数据
                mask = input_mask.clone()
                # 将掩码中不等于 some_const 的元素置为 1
                mask[mask != some_const] = 1
                # 将掩码中等于 some_const 的元素置为 0
                mask[mask == some_const] = 0
                # 返回处理后的掩码
                return mask

        # 创建一个形状为 (2, 2, 2) 的随机张量，并且要求梯度计算
        mask = torch.randn(2, 2, 2, requires_grad=True)
        # 创建一个常量张量，值为 5，数据类型为 torch.float
        constant = torch.tensor(5, dtype=torch.float)
        # 运行测试函数，测试 MaskedFillModel 的前向传播方法
        self.run_test(MaskedFillModel(), (mask, constant))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_to_masked_scatter(self):
        # 定义一个继承自 torch.nn.Module 的 MaskedScatterModel 类
        class MaskedScatterModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, input_mask, some_const):
                # 克隆输入的掩码数据
                mask = input_mask.clone()
                # 将掩码中不等于 some_const 的元素替换为一个全为 1 的张量
                mask[mask != some_const] = torch.ones(8)
                # 返回处理后的掩码
                return mask

        # 创建一个形状为 (2, 2, 2) 的随机张量，并且要求梯度计算
        mask = torch.randn(2, 2, 2, requires_grad=True)
        # 创建一个常量张量，值为 5，数据类型为 torch.float
        constant = torch.tensor(5, dtype=torch.float)
        # 运行测试函数，测试 MaskedScatterModel 的前向传播方法
        self.run_test(MaskedScatterModel(), (mask, constant))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_with_1d_mask_to_masked_scatter(self):
        # 定义一个继承自 torch.nn.Module 的 MaskedScatterModel 类
        class MaskedScatterModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, tensor, mask, some_const):
                # 根据 1 维掩码 mask 将 tensor 中对应位置的元素替换为 some_const
                tensor[mask] = some_const
                # 返回处理后的 tensor
                return tensor

        # 创建一个 1 维布尔类型的掩码张量
        mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.bool)
        # 创建一个形状为 (8, 4, 5) 的随机张量，并且要求梯度计算
        tensor = torch.randn(8, 4, 5, requires_grad=True)
        # 创建一个形状为 (4, 4, 5) 的随机张量，数据类型为 torch.float
        some_const = torch.randn(4, 4, 5, dtype=torch.float)
        # 运行测试函数，测试 MaskedScatterModel 的前向传播方法
        self.run_test(MaskedScatterModel(), (tensor, mask, some_const))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_pixel_shuffle(self):
        # 定义一个继承自 torch.nn.Module 的 PixelShuffle 类
        class PixelShuffle(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 对输入 x 进行像素混洗操作，放大因子为 2
                return torch.pixel_shuffle(x, upscale_factor=2)

        # 创建一个形状为 (2, 16, 4, 3) 的随机张量，并且要求梯度计算
        x = torch.randn(2, 16, 4, 3, requires_grad=True)
        # 创建一个形状为 (4, 32, 8, 4) 的随机张量，并且要求梯度计算
        y = torch.randn(4, 32, 8, 4, requires_grad=True)
        # 运行测试函数，测试 PixelShuffle 的前向传播方法
        self.run_test(PixelShuffle(), x)
        # 再次运行测试函数，测试 PixelShuffle 的前向传播方法，并传入额外的测试输入
        self.run_test(
            PixelShuffle(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
            additional_test_inputs=[y],
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_pixel_unshuffle(self):
        # 定义一个继承自 torch.nn.Module 的 PixelUnshuffle 类
        class PixelUnshuffle(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 对输入 x 进行像素解混洗操作，降低因子为 2
                return torch.pixel_unshuffle(x, downscale_factor=2)

        # 创建一个形状为 (2, 16, 4, 6) 的随机张量，并且要求梯度计算
        x = torch.randn(2, 16, 4, 6, requires_grad=True)
        # 创建一个形状为 (4, 32, 8, 4) 的随机张量，并且要求梯度计算
        y = torch.randn(4, 32, 8, 4, requires_grad=True)
        # 运行测试函数，测试 PixelUnshuffle 的前向传播方法
        self.run_test(PixelUnshuffle(), x)
        # 再次运行测试函数，测试 PixelUnshuffle 的前向传播方法，并传入额外的测试输入
        self.run_test(
            PixelUnshuffle(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
            additional_test_inputs=[y],
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_reciprocal(self):
        # 定义一个简单的 torch 模型，计算输入张量的倒数
        class ReciprocalModel(torch.nn.Module):
            def forward(self, x):
                return torch.reciprocal(x)

        # 创建 ReciprocalModel 实例
        model = ReciprocalModel()
        # 创建一个整型张量
        x = torch.tensor([2, 4])
        # 分别运行测试，将输入张量转换为 long 类型、float 类型和 double 类型
        self.run_test(model, x.to(torch.long))
        self.run_test(model, x.to(torch.float))
        self.run_test(model, x.to(torch.double))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scalar_type(self):
        # 定义一个 torch 模型，进行一些基本的算术运算
        class ArithmeticModel(torch.nn.Module):
            def forward(self, x):
                return x.size(0) * 2 * x, 2 - x

        # 创建一个 float32 类型的全 1 张量
        x = torch.ones(2, 3, dtype=torch.float32)
        # 运行测试
        self.run_test(ArithmeticModel(), x)

        # 定义一个 torch 模型，进行比较运算
        class ComparisonModel(torch.nn.Module):
            def forward(self, x, y):
                # 创建一个 float 类型的张量 [12.0]
                a = torch.tensor([12.0])
                return x.lt(1.5) & y.le(2) & x.le(1), x.gt(y), x.lt(y), a.ge(x.size(0))

        # 创建一个 int32 类型的全 1 张量
        x = torch.ones(2, 3, dtype=torch.int32)
        # 创建一个 float32 类型的全 1 张量
        y = torch.ones(2, 3, dtype=torch.float32)
        # 运行测试
        self.run_test(ComparisonModel(), (x, y))

        # 定义一个 torch 模型，进行矩阵乘法和加法运算
        class MatMulModel(torch.nn.Module):
            def forward(self, x):
                return torch.mm(x, x) + x + torch.mm(x, x) + x

        # 创建一个全 1 的 3x3 张量
        x = torch.ones(3, 3)
        # 运行测试
        self.run_test(MatMulModel(), x)

        # 定义一个 torch 模型，进行矩阵乘法和加法运算
        class AddMMModel(torch.nn.Module):
            def forward(self, x):
                return torch.mm(x, x) + x

        # 创建一个全 1 的 3x3 张量
        x = torch.ones(3, 3)
        # 运行测试
        self.run_test(AddMMModel(), x)

        # 定义一个 torch 模型，使用 torch.full 创建一个指定大小的张量
        class FullModel(torch.nn.Module):
            # 在导出完整模型时使用 add
            def forward(self, x):
                return torch.full((3, 4), x)

        # 创建一个值为 12.0 的标量张量
        x = torch.tensor(12.0)
        # 运行测试
        self.run_test(FullModel(), x)

        # 定义一个 torch 模型，进行张量拼接
        class CatModel(torch.nn.Module):
            def forward(self, fp16, fp32):
                return torch.cat([fp16, fp32])

        # 创建一个值为 0.5 的张量，转换为半精度浮点数
        fp16 = Tensor([0.5])
        fp16 = fp16.half()
        # 创建一个值为 1.5 的张量
        fp32 = Tensor([1.5])
        # 运行测试
        self.run_test(CatModel(), (fp16, fp32))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scalar_type_does_not_trigger_upcast_type_promotion(self):
        # 定义一个 torch 模型，确保不会触发类型提升
        class DoNotUpcastModel(torch.nn.Module):
            def forward(self, x):
                # 计算缩放因子，不会触发类型提升
                scale = x.size()[-1] ** -0.5
                # 'scale' 导出为 ONNX float32 等级的标量张量。
                # 以下 'Mul' 不应被提升为 float32。
                return x * scale

        # 创建一个 float16 类型的全 1 张量
        x = torch.ones(2, 3, dtype=torch.float16)
        # 运行测试
        self.run_test(DoNotUpcastModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scalar_type_promotion_onnx_where_two_prim_const(self):
        # 定义一个 torch 模型，其中有两个基本常量的类型转换
        class TwoPrimConstCastWhereModel(torch.nn.Module):
            def forward(self, c):
                return torch.where(c, 0, 1.0)

        # 创建一个布尔类型的全 1 张量
        c = torch.ones(8, dtype=torch.bool)
        # 运行测试
        self.run_test(TwoPrimConstCastWhereModel(), (c))
    # 定义一个测试方法，测试在 ONNX 中使用 torch.where 函数时的标量类型提升
    def test_scalar_type_promotion_onnx_where_one_prim_const(self):
        # 定义一个继承自 torch.nn.Module 的模型类 OnePrimConstCastWhereModel
        class OnePrimConstCastWhereModel(torch.nn.Module):
            # 实现模型的前向传播方法
            def forward(self, c, x):
                # 使用 torch.where 函数根据条件 c 选择 x 或 1.0
                return torch.where(c, x, 1.0)

        # 创建一个元素全为 True 的张量 c，数据类型为 torch.bool
        c = torch.ones(8, dtype=torch.bool)
        # 创建一个元素全为 1.0 的张量 x，数据类型为 torch.float16
        x = torch.ones(8, dtype=torch.float16)
        # 运行测试，传入 OnePrimConstCastWhereModel 的实例和参数元组 (c, x)
        self.run_test(OnePrimConstCastWhereModel(), (c, x))

    # 根据最小的运算集版本跳过不支持的测试
    @skipIfUnsupportedMinOpsetVersion(9)
    # 测试在 ONNX 中使用 torch.where 函数时的标量类型提升，其中条件为张量常量
    def test_scalar_type_promotion_onnx_where_one_tensor_const(self):
        # 定义一个继承自 torch.nn.Module 的模型类 OneTensorConstCastWhereModel
        class OneTensorConstCastWhereModel(torch.nn.Module):
            # 实现模型的前向传播方法
            def forward(self, c, x):
                # 使用 torch.where 函数根据条件 c 选择 x 或 torch.ones 创建的浮点64位张量常量
                return torch.where(c, x, torch.ones(size=(), dtype=torch.float64))

        # 创建一个元素全为 True 的张量 c，数据类型为 torch.bool
        c = torch.ones(8, dtype=torch.bool)
        # 创建一个元素全为 1.0 的张量 x，数据类型为 torch.float16
        x = torch.ones(8, dtype=torch.float16)
        # 运行测试，传入 OneTensorConstCastWhereModel 的实例和参数元组 (c, x)
        self.run_test(OneTensorConstCastWhereModel(), (c, x))

    # 根据最小的运算集版本跳过不支持的测试
    @skipIfUnsupportedMinOpsetVersion(9)
    # 测试在 ONNX 中使用 torch.where 函数时的标量类型提升，其中条件和选择的张量类型不同
    def test_scalar_type_upcast_type_promotion_onnx_where_no_const(self):
        # 定义一个继承自 torch.nn.Module 的模型类 OnnxWhereUpcastModel
        class OnnxWhereUpcastModel(torch.nn.Module):
            # 实现模型的前向传播方法
            def forward(self, c, x, y):
                # 使用 torch.where 函数根据条件 c 选择 x 或 y
                return torch.where(c, x, y)

        # 创建一个元素全为 True 的张量 c，数据类型为 torch.bool
        c = torch.ones(8, dtype=torch.bool)
        # 创建一个元素全为 1.0 的张量 x，数据类型为 torch.float16
        x = torch.ones(8, dtype=torch.float16)
        # 创建一个元素全为 1.0 的张量 y，数据类型为 torch.float32
        y = torch.ones(8, dtype=torch.float32)
        # 运行测试，传入 OnnxWhereUpcastModel 的实例和参数元组 (c, x, y)
        self.run_test(OnnxWhereUpcastModel(), (c, x, y))

    # 根据最小的运算集版本跳过不支持的测试
    @skipIfUnsupportedMinOpsetVersion(9)
    # 测试 torch.full_like 函数的使用
    def test_full_like(self):
        # 定义一个继承自 torch.nn.Module 的模型类 FullLikeModel
        class FullLikeModel(torch.nn.Module):
            # 实现模型的前向传播方法
            def forward(self, x):
                # 使用 torch.full_like 函数创建一个与输入张量 x 形状一致的张量，元素值为 1.3，数据类型为 torch.int
                return torch.full_like(x, 1.3, dtype=torch.int)

        # 创建一个值为 12 的整数张量 x
        x = torch.tensor(12)
        # 运行测试，传入 FullLikeModel 的实例和参数 x
        self.run_test(FullLikeModel(), x)

    # 根据最小的运算集版本跳过不支持的测试
    @skipIfUnsupportedMinOpsetVersion(9)
    # 测试 torch.full_like 函数的使用，其中填充值为 y + 2
    @skipDtypeChecking
    def test_full_like_value(self):
        # 定义一个继承自 torch.nn.Module 的模型类 FullLikeModel
        class FullLikeModel(torch.nn.Module):
            # 实现模型的前向传播方法
            def forward(self, x, y):
                # 计算 y + 2，并使用 torch.full_like 函数创建一个与输入张量 x 形状一致的张量
                out = y + 2
                return torch.full_like(x, out)

        # 创建一个值为 12 的整数张量 x 和一个值为 2 的整数张量 y
        x = torch.tensor(12)
        y = torch.tensor(2)
        # 运行测试，传入 FullLikeModel 的实例和参数元组 (x, y)
        self.run_test(FullLikeModel(), (x, y))

    # 测试 torch.norm 函数计算向量的 L1 范数
    def test_l1_norm(self):
        # 定义一个继承自 torch.nn.Module 的模型类 NormModel
        class NormModel(torch.nn.Module):
            # 实现模型的前向传播方法
            def forward(self, x):
                # 使用 torch.norm 函数计算输入张量 x 按照 L1 范数在最后一个维度上的范数，不保持维度
                return torch.norm(x, p=1, dim=-1, keepdim=False)

        # 创建一个形状为 (4, 2, 3) 的随机张量 x，需要计算梯度
        x = torch.randn(4, 2, 3, requires_grad=True)
        # 运行测试，传入 NormModel 的实例和参数 x
        self.run_test(NormModel(), x)

    # 测试 torch.norm 函数计算向量的 L2 范数
    def test_l2_norm(self):
        # 定义一个继承自 torch.nn.Module 的模型类 NormModel
        class NormModel(torch.nn.Module):
            # 实现模型的前向传播方法
            def forward(self, x):
                # 使用 torch.norm 函数计算输入张量 x 按照 L2 范数在倒数第二个维度上的范数，不保持维度
                return torch.norm(x, p=2, dim=-2, keepdim=False)

        # 创建一个形状为 (4, 2, 3) 的随机张量 x，需要计算梯度
        x = torch.randn(4, 2, 3, requires_grad=True)
        # 运行测试，传入 NormModel 的实例和参数 x
        self.run_test(NormModel(), x)

    # 测试 torch.norm 函数计算矩阵的 Frobenius 范数
    def test_frobenius_norm(self):
        # 定义一个继承自 torch.nn.Module 的模型类 NormModel
        class NormModel(torch.nn.Module):
            # 实现模型的前向传播方法
            def forward(self, x):
                # 使用 torch.norm 函数计算输入张量 x 的 Frobenius 范数，沿第一个维度，不保持维度
                return torch.norm(x, p="fro", dim=0, keepdim=False)

        # 创建一个形状为 (4, 2, 3) 的随机张量 x，需要计算梯度
        x = torch.randn(4, 2, 3, requires_grad=True)
        # 运行测试，传入 NormModel 的实例和参数 x
        self.run_test(NormModel(), x)

    # 测试 torch.norm 函数计算矩阵的 Frobenius 范数，并保持维度
    def test_frobenius_norm_keepdim(self):
        #
    # 定义一个测试用例，测试 torch 的 unfold 方法
    def test_unfold(self):
        # 定义一个继承自 torch.nn.Module 的类 UnfoldModel
        class UnfoldModel(torch.nn.Module):
            # 重写 forward 方法，对输入 x 执行 unfold 操作
            def forward(self, x):
                return x.unfold(dimension=2, size=2, step=2)

        # 生成一个随机张量 x，形状为 (4, 2, 3)，需要计算梯度
        x = torch.randn(4, 2, 3, requires_grad=True)
        # 生成一个随机张量 y，形状为 (2, 1, 3)，需要计算梯度
        y = torch.randn(2, 1, 3, requires_grad=True)
        # 运行测试，传入 UnfoldModel 实例、x 作为输入，动态轴设置为 x 的前两个维度，输入名称为 "x"，额外的测试输入为 y
        self.run_test(
            UnfoldModel(),
            x,
            dynamic_axes={"x": [0, 1]},
            input_names=["x"],
            additional_test_inputs=[y],
        )

    # 定义一个测试用例，测试 torch.jit.ScriptModule 中的 unfold 方法推断形状
    def test_unfold_infer_shape(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 UnfoldModule
        class UnfoldModule(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 添加一个 Conv1d 层，输入通道为 3，输出通道为 1，卷积核大小为 3，步长为 2
                self.conv = torch.nn.Conv1d(3, 1, 3, stride=2)

            @torch.jit.script_method
            # 重写 forward 方法，对输入 x 执行卷积操作后再执行 unfold 操作
            def forward(self, x):
                x = self.conv(x)
                return x.unfold(dimension=2, size=2, step=2)

        # 生成一个随机张量 x，形状为 (32, 3, 64)
        x = torch.randn(32, 3, 64)
        # 运行测试，传入 UnfoldModule 实例和 x 作为输入
        self.run_test(UnfoldModule(), x)

    # 定义一个测试用例，测试动态输入下的 unfold 方法
    @skipIfUnsupportedMinOpsetVersion(12)
    def test_unfold_dynamic_inputs(self):
        # 定义一个继承自 torch.nn.Module 的类 UnfoldModel
        class UnfoldModel(torch.nn.Module):
            # 重写 forward 方法，对输入 x 执行动态大小和步长的 unfold 操作
            def forward(self, x):
                return x.unfold(dimension=2, size=x.shape[1], step=x.shape[1] - 1)

        # 生成一个随机张量 x，形状为 (4, 2, 4)，需要计算梯度
        x = torch.randn(4, 2, 4, requires_grad=True)
        # 运行测试，传入 UnfoldModel 实例和 x 作为输入
        self.run_test(UnfoldModel(), x)

        # 再次定义一个类 UnfoldModel（类名重复，需要注意），重写 forward 方法，执行动态大小为 x 第二维度，步长为 1 的 unfold 操作
        class UnfoldModel(torch.nn.Module):
            def forward(self, x):
                return x.unfold(dimension=2, size=x.shape[1], step=1)

        # 生成一个随机张量 x，形状为 (4, 2, 4)，需要计算梯度
        x = torch.randn(4, 2, 4, requires_grad=True)
        # 运行测试，传入重定义后的 UnfoldModel 实例和 x 作为输入
        self.run_test(UnfoldModel(), x)

    # 定义一个测试用例，测试 torch.mv 方法
    @skipIfUnsupportedMinOpsetVersion(9)  # MatMul long inputs is added in ONNX opset 9.
    def test_mv(self):
        # 定义一个继承自 torch.nn.Module 的类 MatmulModel
        class MatmulModel(torch.nn.Module):
            # 重写 forward 方法，执行 torch.mv 运算
            def forward(self, input, other):
                return torch.mv(input, other)

        # 生成一个随机张量 x，形状为 (4, 5)，需要计算梯度
        x = torch.randn(4, 5, requires_grad=True)
        # 生成一个随机张量 y，形状为 (5,)，需要计算梯度
        y = torch.randn(5, requires_grad=True)
        # 运行测试，传入 MatmulModel 实例、(x, y) 作为输入
        self.run_test(MatmulModel(), (x, y))

        # 生成一个随机整数张量 x，形状为 (4, 5)
        x = torch.randint(10, (4, 5))
        # 生成一个随机整数张量 y，形状为 (5,)
        y = torch.randint(10, (5,))
        # 运行测试，传入 MatmulModel 实例、(x, y) 作为输入
        self.run_test(MatmulModel(), (x, y))

    # 定义一个测试用例，测试 torch.dot 方法
    @skipIfUnsupportedMinOpsetVersion(9)  # MatMul long inputs is added in ONNX opset 9.
    def test_dot(self):
        # 定义一个继承自 torch.nn.Module 的类 MatmulModel
        class MatmulModel(torch.nn.Module):
            # 重写 forward 方法，执行 torch.dot 运算
            def forward(self, input, other):
                return torch.dot(input, other)

        # 生成一个随机张量 x，形状为 (5,)，需要计算梯度
        x = torch.randn(5, requires_grad=True)
        # 生成一个随机张量 y，形状为 (5,)，需要计算梯度
        y = torch.randn(5, requires_grad=True)
        # 运行测试，传入 MatmulModel 实例、(x, y) 作为输入
        self.run_test(MatmulModel(), (x, y))

        # 生成一个随机整数张量 x，形状为 (5,)
        x = torch.randint(10, (5,))
        # 生成一个随机整数张量 y，形状为 (5,)
        y = torch.randint(10, (5,))
        # 运行测试，传入 MatmulModel 实例、(x, y) 作为输入
        self.run_test(MatmulModel(), (x, y))

    # 定义一个测试用例，测试 torch.nn.utils.spectral_norm 方法
    @skipScriptTest()  # SpectralNorm not TorchScript compatible.
    def test_spectral_norm(self):
        # 使用 torch.nn.utils.spectral_norm 对 torch.nn.Linear(2, 4) 进行谱归一化
        m = torch.nn.utils.spectral_norm(torch.nn.Linear(2, 4))

        # 生成一个随机张量 x，形状为 (6, 2)
        x = torch.randn(6, 2)
        # 运行测试，传入谱归一化后的线性层 m 和输入 x
        self.run_test(m, (x,))
    # 定义一个测试 PReLU 激活函数的方法
    def test_prelu(self):
        # 定义一个内部的 PReLU 模型类，继承自 torch.nn.Module
        class PReluModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.prelu = torch.nn.PReLU()

            # 前向传播方法，应用 PReLU 激活函数
            def forward(self, x):
                return self.prelu(x)

        # 生成一个大小为 (2, 3, 4) 的随机张量 x 和 (2, 4, 5) 的随机张量 y
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 4, 5)
        # 运行测试，传入 PReluModel 实例、输入 x、指定输入名称为 "x"、动态轴设置、额外的测试输入 y
        self.run_test(
            PReluModel(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [1, 2]},
            additional_test_inputs=[y],
        )

    # 定义一个测试 PReLU 激活函数对标量输入的方法
    def test_prelu_scalar(self):
        # 生成一个标量张量 x，值为 1.0
        x = torch.scalar_tensor(1.0)
        # 运行测试，传入 PReLU 实例、输入 x、指定输入名称为 "x"
        self.run_test(torch.nn.PReLU(), x, input_names=["x"])

    # 定义一个测试 ReLU6 激活函数的方法
    def test_relu6(self):
        # 定义一个内部的 Relu6 模型类，继承自 torch.nn.Module
        class Relu6Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu6 = torch.nn.ReLU6()

            # 前向传播方法，应用 ReLU6 激活函数
            def forward(self, x):
                return self.relu6(x)

        # 生成一个符合要求的随机张量 x 和 y，大小为 (2, 3, 4)
        x = torch.randn(2, 3, 4) * 100.0
        y = torch.randn(2, 4, 5) * 100.0
        # 运行测试，传入 Relu6Model 实例、输入 x、指定输入名称为 "x"、动态轴设置、额外的测试输入 y
        self.run_test(
            Relu6Model(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [1, 2]},
            additional_test_inputs=[y],
        )

    # 定义一个测试 SiLU 激活函数的方法
    def test_silu(self):
        # 定义一个内部的 SiLU 模型类，继承自 torch.nn.Module
        class SiLUModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.silu = torch.nn.SiLU()

            # 前向传播方法，应用 SiLU 激活函数
            def forward(self, x):
                return self.silu(x)

        # 生成一个大小为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 运行测试，传入 SiLUModel 实例和输入 x
        self.run_test(SiLUModel(), (x))

    # 使用装饰器跳过不支持的最小操作集版本来定义测试 tril 函数
    @skipIfUnsupportedMinOpsetVersion(14)
    def test_tril(self):
        # 定义一个返回下三角矩阵的模型类
        class trilModel(torch.nn.Module):
            def forward(self, x):
                return torch.tril(x)

        # 生成一个大小为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 运行测试，传入 trilModel 实例和输入 x
        self.run_test(trilModel(), (x))

        # 定义一个带对角线参数的返回下三角矩阵的模型类
        class trilModelwithDiagonal(torch.nn.Module):
            def forward(self, x):
                return torch.tril(x, diagonal=1)

        # 生成一个大小为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 运行测试，传入 trilModelwithDiagonal 实例和输入 x
        self.run_test(trilModelwithDiagonal(), (x))

        # 定义一个带负对角线参数的返回下三角矩阵的模型类
        class trilModelwithNegDiagonal(torch.nn.Module):
            def forward(self, x):
                return torch.tril(x, diagonal=-1)

        # 生成一个大小为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 运行测试，传入 trilModelwithNegDiagonal 实例和输入 x
        self.run_test(trilModelwithNegDiagonal(), (x))

        # 定义一个带对角线输入参数的返回下三角矩阵的模型类
        class trilModelWithDiagonalInput(torch.nn.Module):
            def forward(self, x, diagnonal: int):
                return torch.tril(x, diagonal=diagnonal)

        # 生成一个大小为 (2, 3, 4) 的随机张量 x 和对角线参数值 5
        x = torch.randn(2, 3, 4)
        # 运行测试，传入 trilModelWithDiagonalInput 实例、输入 x 和对角线参数值 5
        self.run_test(trilModelWithDiagonalInput(), (x, 5))
    # 定义一个测试方法，用于测试 torch.triu 函数的不同用法
    def test_triu(self):
        # 定义一个简单的 PyTorch 模型，将输入张量的上三角部分返回
        class triuModel(torch.nn.Module):
            def forward(self, x):
                return torch.triu(x)

        # 创建一个形状为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 使用自定义的 run_test 方法来运行 triuModel 模型的测试
        self.run_test(triuModel(), (x))

        # 定义另一个 PyTorch 模型，将输入张量的指定对角线以上的部分返回
        class triuModelwithDiagonal(torch.nn.Module):
            def forward(self, x):
                return torch.triu(x, diagonal=1)

        # 再次创建形状为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 使用 run_test 方法来运行 triuModelwithDiagonal 模型的测试
        self.run_test(triuModelwithDiagonal(), (x))

        # 定义另一个 PyTorch 模型，将输入张量的指定对角线以下的部分返回
        class triuModelwithNegDiagonal(torch.nn.Module):
            def forward(self, x):
                return torch.triu(x, diagonal=-1)

        # 再次创建形状为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 使用 run_test 方法来运行 triuModelwithNegDiagonal 模型的测试
        self.run_test(triuModelwithNegDiagonal(), (x))

        # 定义一个接受对角线参数的 PyTorch 模型，根据输入的对角线位置返回张量的上三角部分
        class triuModelWithDiagonalInput(torch.nn.Module):
            def forward(self, x, diagnonal: int):
                return torch.triu(x, diagonal=diagnonal)

        # 再次创建形状为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 使用 run_test 方法来运行 triuModelWithDiagonalInput 模型的测试，传入对角线参数 5
        self.run_test(triuModelWithDiagonalInput(), (x, 5))

    # 定义一个测试方法，用于测试 torch.nn.Mish 激活函数的使用
    def test_mish(self):
        # 定义一个简单的 PyTorch 模型，使用 Mish 激活函数作为其一部分
        class MishModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mish = torch.nn.Mish()

            def forward(self, x):
                return self.mish(x)

        # 创建一个形状为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 使用 run_test 方法来运行 MishModel 模型的测试
        self.run_test(MishModel(), (x))

    # 定义一个测试方法，用于测试 torch.remainder 函数的不同用法
    def test_remainder(self):
        # 定义一个 PyTorch 模型，返回输入张量对另一个张量的求余结果
        class RemainderModel(torch.nn.Module):
            def forward(self, input, other):
                return torch.remainder(input, other)

        # 创建形状为 (4, 2, 3) 的随机张量 x 和形状为 (1, 2, 1) 的随机张量 y
        x = torch.randn(4, 2, 3)
        y = torch.randn(1, 2, 1)
        # 使用 run_test 方法来运行 RemainderModel 模型的测试，传入 x 和 y
        self.run_test(RemainderModel(), (x, y))

        # 创建一个包含整数的张量 x 和 y
        x = torch.tensor([7, 6, -7, -6], dtype=torch.long)
        y = torch.tensor([2], dtype=torch.long)
        # 将 x 转换为浮点型，并使用 run_test 方法来运行 RemainderModel 模型的测试，传入 x 和 y
        x = x.to(torch.float)
        self.run_test(RemainderModel(), (x, y))

        # 将 y 也转换为浮点型，并使用 run_test 方法来运行 RemainderModel 模型的测试，传入转换后的 x 和 y
        y = y.to(torch.float)
        self.run_test(RemainderModel(), (x, y))

        # 将 x 转换为 int32 类型，并使用 run_test 方法来运行 RemainderModel 模型的测试，传入转换后的 x 和 y
        x = x.to(torch.int32)
        self.run_test(RemainderModel(), (x, y))

    # 定义一个测试方法，用于测试 torch.remainder 函数在标量情况下的使用
    def test_remainder_scalar(self):
        # 定义一个带有标量参数的 PyTorch 模型，对输入张量的每个元素求余
        class RemainderModel(torch.nn.Module):
            def __init__(self, scalar=2.55):
                super().__init__()
                self.scalar = scalar

            def forward(self, input):
                return torch.remainder(input, self.scalar)

        # 创建一个形状为 (2, 3) 的随机整数张量 x
        x = torch.randint(10, (2, 3))
        # 使用 run_test 方法来运行 RemainderModel 模型的测试，传入 x
        self.run_test(RemainderModel(), x)

        # 创建一个包含整数的张量 x，并使用 run_test 方法来运行 RemainderModel 模型的测试，传入 x 和标量参数 2
        x = torch.tensor([7, 6, -7, -6], dtype=torch.long)
        self.run_test(RemainderModel(2), x)

    # 使用 skipIfUnsupportedMinOpsetVersion 装饰器，标记以下测试方法在 Opset 版本小于 10 时不执行
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_fmod(self):
        # 定义一个 PyTorch 模型，返回输入张量对另一个张量的按元素 fmod 结果
        class FModModel(torch.nn.Module):
            def forward(self, input, other):
                return torch.fmod(input, other)

        # 创建形状为 (4, 2, 3) 的随机张量 x 和形状为 (1, 2, 1) 的随机张量 y
        x = torch.randn(4, 2, 3)
        y = torch.randn(1, 2, 1)
        # 使用 run_test 方法来运行 FModModel 模型的测试，传入 x 和 y
        self.run_test(FModModel(), (x, y))
    def test_fmod_scalar(self):
        # 定义一个继承自 torch.nn.Module 的模型类 FModModel
        class FModModel(torch.nn.Module):
            # 定义模型的前向传播函数，对输入的张量 input 求模运算
            def forward(self, input):
                return torch.fmod(input, 2.55)

        # 生成一个形状为 (2, 3) 的整数张量 x，用于测试
        x = torch.randint(10, (2, 3))
        # 调用自定义的测试函数 run_test，测试 FModModel 模型在输入 x 上的运行结果
        self.run_test(FModModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_glu(self):
        # 定义一个继承自 torch.nn.Module 的模型类 GluModel
        class GluModel(torch.nn.Module):
            # 定义模型的前向传播函数，使用 torch.nn.functional.glu 对输入 x 进行运算
            def forward(self, x):
                return torch.nn.functional.glu(x)

        # 生成一个形状为 (2, 4, 5, 6) 的随机张量 x，要求梯度计算
        x = torch.randn(2, 4, 5, 6, requires_grad=True)
        # 调用自定义的测试函数 run_test，测试 GluModel 模型在输入 x 上的运行结果
        self.run_test(GluModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_gelu(self):
        # 定义一个继承自 torch.nn.Module 的模型类 GeluModel
        class GeluModel(torch.nn.Module):
            # 定义模型的前向传播函数，使用 torch.nn.functional.gelu 对输入 x 进行运算，选择"none"方法
            def forward(self, x):
                return torch.nn.functional.gelu(x, approximate="none")

        # 生成一个形状为 (2, 4, 5, 6) 的随机张量 x，要求梯度计算
        x = torch.randn(2, 4, 5, 6, requires_grad=True)
        # 调用自定义的测试函数 run_test，测试 GeluModel 模型在输入 x 上的运行结果
        self.run_test(GeluModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_tanh_gelu(self):
        # 定义一个继承自 torch.nn.Module 的模型类 GeluModel
        class GeluModel(torch.nn.Module):
            # 定义模型的前向传播函数，使用 torch.nn.functional.gelu 对输入 x 进行运算，选择"tanh"方法
            def forward(self, x):
                return torch.nn.functional.gelu(x, approximate="tanh")

        # 生成一个形状为 (2, 4, 5, 6) 的随机张量 x，要求梯度计算
        x = torch.randn(2, 4, 5, 6, requires_grad=True)
        # 调用自定义的测试函数 run_test，测试 GeluModel 模型在输入 x 上的运行结果
        self.run_test(GeluModel(), x)

    def test_add_inplace(self):
        # 定义一个继承自 torch.nn.Module 的模型类 InplaceAddModel
        class InplaceAddModel(torch.nn.Module):
            # 定义模型的前向传播函数，对输入 x 进行原地加法操作
            def forward(self, x):
                x += 12
                return x

        # 生成一个形状为 (4, 2, 3) 的随机张量 x，要求梯度计算
        x = torch.randn(4, 2, 3, requires_grad=True)
        # 调用自定义的测试函数 run_test，测试 InplaceAddModel 模型在输入 x 上的运行结果
        self.run_test(InplaceAddModel(), x)

    def test_addcmul(self):
        # 定义一个继承自 torch.nn.Module 的模型类 AddcmulModel
        class AddcmulModel(torch.nn.Module):
            # 定义模型的前向传播函数，对输入 x、t1、t2 进行 torch.addcmul 运算
            def forward(self, x, t1, t2):
                return torch.addcmul(x, t1, t2), torch.addcmul(x, t1, t2, value=2.2)

        # 生成形状分别为 (1, 3)、(3, 1)、(1, 3) 的随机张量 x、t1、t2，用于测试
        x = torch.randn(1, 3)
        t1 = torch.randn(3, 1)
        t2 = torch.randn(1, 3)
        # 调用自定义的测试函数 run_test，测试 AddcmulModel 模型在输入 x、t1、t2 上的运行结果
        self.run_test(AddcmulModel(), (x, t1, t2))

    def test_rsqrt(self):
        # 定义一个继承自 torch.nn.Module 的模型类 RsqrtModel
        class RsqrtModel(torch.nn.Module):
            # 定义模型的前向传播函数，对输入 x 进行 rsqrt 运算
            def forward(self, x):
                return x.rsqrt()

        # 生成一个形状为 (4, 2, 3)、数据类型为 torch.float64 的随机张量 x，要求梯度计算
        x = torch.randn(4, 2, 3, requires_grad=True, dtype=torch.float64)
        # 调用自定义的测试函数 run_test，测试 RsqrtModel 模型在输入 x 上的运行结果
        self.run_test(RsqrtModel(), x)

    def test_rsqrt_zeros(self):
        # 定义一个继承自 torch.nn.Module 的模型类 RsqrtModel
        class RsqrtModel(torch.nn.Module):
            # 定义模型的前向传播函数，对输入 x 进行 rsqrt 运算
            def forward(self, x):
                return x.rsqrt()

        # 生成一个形状为 (4, 2, 3)、数据类型为 torch.float64 的零张量 x，要求梯度计算
        x = torch.zeros(4, 2, 3, requires_grad=True, dtype=torch.float64)
        # 调用自定义的测试函数 run_test，测试 RsqrtModel 模型在输入 x 上的运行结果
        self.run_test(RsqrtModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_unique(self):
        # 定义一个继承自 torch.nn.Module 的模型类 UniqueModel
        class UniqueModel(torch.nn.Module):
            # 定义模型的前向传播函数，使用 torch.unique 对输入 x 进行运算
            def forward(self, x):
                return torch.unique(
                    x, sorted=True, return_inverse=False, return_counts=True
                )

        # 生成一个 dtype=torch.long 的整数张量 x，包含元素 [1, 3, 2, 3]
        x = torch.tensor([1, 3, 2, 3], dtype=torch.long)
        # 调用自定义的测试函数 run_test，测试 UniqueModel 模型在输入 x 上的运行结果
        self.run_test(UniqueModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_unique_along_dim(self):
        # 定义一个继承自 torch.nn.Module 的内部类 UniqueModel
        class UniqueModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 使用 torch.unique 函数对输入张量 x 进行唯一化操作
                return torch.unique(
                    x, dim=0, sorted=True, return_inverse=True, return_counts=False
                )

        # 创建一个输入张量 x，包含四个长整型数值
        x = torch.tensor([1, 3, 2, 3], dtype=torch.long)
        # 调用 self.run_test 方法运行 UniqueModel 类，并传入输入张量 x
        self.run_test(UniqueModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_cumsum(self):
        # 定义一个继承自 torch.nn.Module 的内部类 CumSum
        class CumSum(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, input):
                # 使用 torch.cumsum 函数对输入张量 input 进行累加求和操作，沿着 dim=0 的维度
                return torch.cumsum(input, dim=0)

        # 创建一个形状为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 创建 CumSum 类的实例 model
        model = CumSum()
        # 调用 self.run_test 方法运行 model，并传入输入张量 x
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_cumsum_with_cast(self):
        # 定义一个继承自 torch.nn.Module 的内部类 CumSum
        class CumSum(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, input):
                # 使用 torch.cumsum 函数对输入张量 input 进行累加求和操作，沿着 dim=0 的维度，并将结果转换为 torch.float32 类型
                return torch.cumsum(input, dim=0, dtype=torch.float32)

        # 创建 CumSum 类的实例 model
        model = CumSum()
        # 创建一个包含三个整型数值的张量 x
        x = torch.tensor([2, 3, 4], dtype=torch.int32)
        # 调用 self.run_test 方法运行 model，并传入输入张量 x
        self.run_test(model, x)
        # 创建一个包含三个布尔值的张量 x
        x = torch.tensor([False, True, True])
        # 再次调用 self.run_test 方法运行 model，并传入输入张量 x
        self.run_test(model, x)

    @skipScriptTest()  # error in propagate as assign input shape
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_embedding_bag(self):
        # 创建一个具有 10 个单词和 5 维度的嵌入袋模型，使用"sum"模式和按频率缩放梯度
        model = torch.nn.EmbeddingBag(10, 5, mode="sum", scale_grad_by_freq=True)
        # 创建一个形状为 (7,) 的整型张量 input，包含随机整数
        input = torch.randint(10, (7,))
        # 创建一个偏移张量 offset，指定嵌入袋中每个样本的起始偏移量
        offset = torch.tensor([0, 2, 5, 6])
        # 调用 self.run_test 方法运行 model，并传入输入张量 input 和偏移张量 offset
        self.run_test(model, (input, offset))

        # 创建一个具有 10 个单词和 5 维度的嵌入袋模型，使用"sum"模式和包含最后一个偏移量
        model = torch.nn.EmbeddingBag(10, 5, mode="sum", include_last_offset=True)
        # 再次创建一个形状为 (7,) 的整型张量 input，包含随机整数
        input = torch.randint(10, (7,))
        # 再次创建一个偏移张量 offset，指定嵌入袋中每个样本的起始偏移量
        offset = torch.tensor([0, 2, 5, 6])
        # 再次调用 self.run_test 方法运行 model，并传入输入张量 input 和偏移张量 offset
        self.run_test(model, (input, offset))

        # 创建一个具有 10 个单词和 5 维度的嵌入袋模型，使用"max"模式
        model = torch.nn.EmbeddingBag(10, 5, mode="max")
        # 创建一个形状为 (7, 5) 的整型张量 input，包含随机整数
        input = torch.randint(10, (7, 5))
        # 调用 self.run_test 方法运行 model，并传入输入张量 input
        self.run_test(model, (input))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_embedding_bag_1d_per_sample_weights(self):
        # 定义一个继承自 torch.nn.Module 的内部类 EmbeddingModel
        class EmbeddingModel(torch.nn.Module):
            # 定义模型的前向传播方法，使用 torch.nn.functional.embedding_bag 函数
            def forward(self, embedding_matrix, input, offset, weights):
                return torch.nn.functional.embedding_bag(
                    input,
                    embedding_matrix,
                    offsets=offset,
                    mode="sum",
                    per_sample_weights=weights,
                )

        # 创建 EmbeddingModel 类的实例 model
        model = EmbeddingModel()
        # 创建一个形状为 (6,) 的整型张量 x，包含随机整数
        x = torch.randint(7, (6,))
        # 创建一个形状为 (6,) 的随机张量 w
        w = torch.randn(
            6,
        )
        # 创建一个偏移张量 offset，指定嵌入袋中每个样本的起始偏移量
        offset = torch.tensor([0, 2, 5])
        # 创建一个形状为 (10, 15) 的随机嵌入矩阵 embedding_matrix
        embedding_matrix = torch.rand(10, 15)
        # 调用 self.run_test 方法运行 model，并传入嵌入矩阵 embedding_matrix、输入张量 x、偏移张量 offset 和权重张量 w
        self.run_test(model, (embedding_matrix, x, offset, w))

    @skipIfUnsupportedMinOpsetVersion(11)
    @unittest.skip(
        "This test is broken with ONNXRuntime(17): "
        "when running with onnxruntime 1.17.0 this test fails with the following error:"
        "FAIL : Non-zero status code returned while running If node. "
        "Name:'/If' Status Message: if.cc:253 Compute "
        "If nodes condition input must have exactly one element"
        "https://github.com/pytorch/pytorch/issues/119442"
    )
    # 定义一个嵌入模型类，继承自torch.nn.Module
    def test_embedding_bag_2d_per_sample_weights(self):
        # 定义嵌入模型的内部类EmbeddingModel，继承自torch.nn.Module
        class EmbeddingModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, embedding_matrix, input, weights):
                # 使用torch.nn.functional.embedding_bag进行嵌入操作，计算加权嵌入的和
                return torch.nn.functional.embedding_bag(
                    input, embedding_matrix, mode="sum", per_sample_weights=weights
                )

        # 生成一个形状为(10, 15)的随机张量作为嵌入矩阵
        embedding_matrix = torch.rand(10, 15)
        # 创建EmbeddingModel类的实例
        model = EmbeddingModel()
        # 生成一个形状为(2, 3)的整数张量作为输入
        x = torch.randint(7, (2, 3))
        # 生成一个形状为(2, 3)的随机张量作为权重
        w = torch.randn(2, 3)

        # 生成一个形状为(4, 3)的整数张量作为输入
        x2 = torch.randint(7, (4, 3))
        # 生成一个形状为(4, 3)的随机张量作为权重
        w2 = torch.randn(4, 3)

        # 运行测试方法self.run_test，传入模型、输入参数、输入名称和动态轴
        self.run_test(
            model,
            (embedding_matrix, x, w),
            input_names=["embed", "x", "w"],
            dynamic_axes={"x": [0], "w": [0]},
            additional_test_inputs=[(embedding_matrix, x2, w2)],
        )

    # 使用skipScriptTest装饰器跳过该测试，原因是存在未初始化的操作和类型转换
    @skipScriptTest()  # scripting prim::Uninitialized, prim::dtype, prim::unchecked_cast
    # 当Opset版本小于11时，跳过测试
    @skipIfUnsupportedMinOpsetVersion(11)
    # 使用unittest.skip装饰器跳过测试，原因是ONNX循环形状推断问题
    @unittest.skip(
        "Due to ONNX Loop shape inference issue. "
        "https://msdata.visualstudio.com/Vienna/_workitems/edit/1352001"
    )
    def test_embedding_bag_dynamic_input(self):
        # 定义一个用于测试动态输入的嵌入模型类
        class EmbeddingModel1D(torch.nn.Module):
            def forward(self, embedding_matrix, input, weights, offsets):
                # 使用 PyTorch 的 embedding_bag 函数进行嵌入操作，按照 sum 模式
                return torch.nn.functional.embedding_bag(
                    input,
                    embedding_matrix,
                    offsets=offsets,
                    mode="sum",
                    per_sample_weights=weights,
                )

        # 创建 EmbeddingModel1D 类的实例
        model = EmbeddingModel1D()
        # 生成一个大小为 (6,) 的整数张量 x
        x = torch.randint(7, (6,))
        # 生成一个大小为 (6,) 的随机张量 w
        w = torch.randn(
            6,
        )
        # 创建一个长整型张量 offsets，其值为 [0, 2, 5]
        offsets = torch.tensor([0, 2, 5], dtype=torch.long)
        # 创建一个大小为 (10, 15) 的随机张量 embedding_matrix
        embedding_matrix = torch.rand(10, 15)
        # 生成一个大小为 (2,) 的整数张量 x2
        x2 = torch.randint(7, (2,))
        # 生成一个大小为 (2,) 的随机张量 w2
        w2 = torch.randn(
            2,
        )
        # 创建一个长整型张量 offsets2，其值为 [0]
        offsets2 = torch.tensor(
            [
                0,
            ],
            dtype=torch.long,
        )
        # 运行测试函数 run_test，传入模型、输入数据以及其他参数
        self.run_test(
            model,
            (embedding_matrix, x, w, offsets),
            additional_test_inputs=[(embedding_matrix2, x2, w2, offsets2)],
            input_names=["embedding_matrix", "x", "offsets", "w"],
            dynamic_axes={
                "embedding_matrix": [0, 1],
                "x": [0],
                "offsets": [0],
                "w": [0],
            },
        )

        # 定义另一个用于测试动态输入的嵌入模型类，这次输入为二维张量
        class EmbeddingModel2D(torch.nn.Module):
            def forward(self, embedding_matrix, input, weights):
                # 使用 PyTorch 的 embedding_bag 函数进行嵌入操作，按照 sum 模式
                return torch.nn.functional.embedding_bag(
                    input, embedding_matrix, mode="sum", per_sample_weights=weights
                )

        # 创建 EmbeddingModel2D 类的实例
        model = EmbeddingModel2D()
        # 生成一个大小为 (2, 3) 的整数张量 x
        x = torch.randint(7, (2, 3))
        # 生成一个大小为 (2, 3) 的随机张量 w
        w = torch.randn(2, 3)
        # 创建一个大小为 (10, 15) 的随机张量 embedding_matrix
        embedding_matrix = torch.rand(10, 15)
        # 生成一个大小为 (3, 5) 的整数张量 x2
        x2 = torch.randint(7, (3, 5))
        # 生成一个大小为 (3, 5) 的随机张量 w2
        w2 = torch.randn(3, 5)
        # 创建一个大小为 (12, 25) 的随机张量 embedding_matrix2
        embedding_matrix2 = torch.rand(12, 25)
        # 运行测试函数 run_test，传入模型、输入数据以及其他参数
        self.run_test(
            model,
            (embedding_matrix, x, w),
            additional_test_inputs=[(embedding_matrix2, x2, w2)],
            input_names=["embedding_matrix", "x", "w"],
            dynamic_axes={"embedding_matrix": [0, 1], "x": [0, 1], "w": [0, 1]},
        )

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_meshgrid(self):
        # 定义一个用于测试 meshgrid 函数的模型类
        class Meshgrid(torch.nn.Module):
            def forward(self, x, y, z):
                # 使用 torch 的 meshgrid 函数生成三个张量 output1, output2, output3
                output1, output2, output3 = torch.meshgrid(x, y, z)
                return output1, output2, output3

        # 生成一个形状为 (3,) 的随机张量 x，并设置 requires_grad=True
        x = torch.randn(3, requires_grad=True)
        # 生成一个形状为 (4,) 的零张量 y，并设置 requires_grad=True
        y = torch.zeros(4, requires_grad=True)
        # 生成一个形状为 (5,) 的随机张量 z，并设置 requires_grad=True
        z = torch.randn(5, requires_grad=True)
        # 运行测试函数 run_test，传入 Meshgrid 类的实例及输入数据
        self.run_test(Meshgrid(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(8)
    # 定义一个测试类，用于测试 torch.meshgrid 的索引行为
    def test_meshgrid_indexing(self):
        # 定义一个内部类 Meshgrid，继承自 torch.nn.Module
        class Meshgrid(torch.nn.Module):
            # 初始化方法，接受一个 indexing 参数
            def __init__(self, indexing):
                super().__init__()
                self.indexing = indexing

            # 前向传播方法，接受 x, y, z 三个参数
            def forward(self, x, y, z):
                # 使用 torch.meshgrid 生成网格
                output1, output2, output3 = torch.meshgrid(
                    x, y, z, indexing=self.indexing
                )
                return output1, output2, output3

        # 创建测试用的 tensor 对象 x, y, z
        x = torch.randn(5, requires_grad=True)
        y = torch.zeros(6, requires_grad=True)
        z = torch.randn(7, requires_grad=True)
        # 遍历不同的 indexing 方式进行测试
        for indexing in ("xy", "ij"):
            # 调用 self.run_test 方法执行 Meshgrid 类的实例化和测试
            self.run_test(Meshgrid(indexing), (x, y, z))

    # 跳过不支持最低 Opset 版本 8 的测试方法装饰器
    @skipIfUnsupportedMinOpsetVersion(8)
    def test_meshgrid_scalar(self):
        # 定义一个内部类 Meshgrid，继承自 torch.nn.Module
        class Meshgrid(torch.nn.Module):
            # 前向传播方法，接受 x, y, z 三个参数
            def forward(self, x, y, z):
                # 使用 torch.meshgrid 生成网格
                output1, output2, output3 = torch.meshgrid(x, y, z)
                return output1, output2, output3

        # 创建测试用的 tensor 对象 x, y, z
        x = torch.ones(3, requires_grad=True)
        y = torch.zeros(4, requires_grad=True)
        z = torch.tensor(2.0)
        # 调用 self.run_test 方法执行 Meshgrid 类的实例化和测试
        self.run_test(Meshgrid(), (x, y, z))

    # 测试 torch.baddbmm 方法的函数
    def test_baddbmm(self):
        # 定义一个内部类 MyModule，继承自 torch.nn.Module
        class MyModule(torch.nn.Module):
            # 前向传播方法，接受 input, batch1, batch2 三个参数
            def forward(self, input, batch1, batch2):
                # 使用 torch.baddbmm 方法执行批量矩阵相加乘操作
                return torch.baddbmm(
                    input, batch1, batch2, alpha=torch.tensor(5), beta=3.5
                )

        # 创建测试用的 tensor 对象 x, batch1, batch2
        x = torch.randn(10, 3, 5)
        batch1 = torch.randn(10, 3, 4)
        batch2 = torch.randn(10, 4, 5)
        model = MyModule()
        # 调用 self.run_test 方法执行 MyModule 类的实例化和测试
        self.run_test(model, (x, batch1, batch2))

    # 测试带动态参数的 torch.baddbmm 方法的函数
    def test_baddbmm_dynamic(self):
        # 定义一个内部类 MyModule，继承自 torch.nn.Module
        class MyModule(torch.nn.Module):
            # 前向传播方法，接受 input, batch1, batch2, alpha, beta 五个参数
            def forward(self, input, batch1, batch2, alpha, beta):
                # 使用 torch.baddbmm 方法执行批量矩阵相加乘操作，接受动态的 alpha 和 beta 参数
                return torch.baddbmm(input, batch1, batch2, alpha=alpha, beta=beta)

        # 创建测试用的 tensor 对象 x, batch1, batch2, alpha, beta
        x = torch.randn(10, 3, 5)
        batch1 = torch.randn(10, 3, 4)
        batch2 = torch.randn(10, 4, 5)
        alpha = torch.tensor(5)
        beta = torch.tensor(3.5)
        model = MyModule()
        # 调用 self.run_test 方法执行 MyModule 类的实例化和测试
        self.run_test(model, (x, batch1, batch2, alpha, beta))

    # 测试 torch.numel 方法的函数
    def test_numel(self):
        # 定义一个内部类 MyModule，继承自 torch.nn.Module
        class MyModule(torch.nn.Module):
            # 前向传播方法，接受 input 参数
            def forward(self, input):
                # 返回输入 tensor 的元素个数乘以输入 tensor 本身
                return input.numel() * input

        # 创建测试用的 tensor 对象 x, 并为输入和附加测试输入指定参数名和动态轴
        x = torch.randn(2, 3, 5)
        x2 = torch.randn(4, 5, 6)
        model = MyModule()
        self.run_test(
            model,
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2]},
            additional_test_inputs=[(x2,)],
        )

    # 测试 torch.numel 方法对空 tensor 的函数
    def test_numel_empty(self):
        # 定义一个内部类 MyModule，继承自 torch.nn.Module
        class MyModule(torch.nn.Module):
            # 前向传播方法，接受 input 参数
            def forward(self, input):
                # 返回输入 tensor 的元素个数乘以输入 tensor 本身
                return input.numel() * input

        # 创建测试用的空 tensor 对象 x, 并为输入和附加测试输入指定参数名和动态轴
        x = torch.randn(0)
        x2 = torch.randn(4)
        model = MyModule()
        self.run_test(
            model,
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0]},
            additional_test_inputs=[(x2,)],
        )
    # 定义一个测试类，用于测试数据类型处理
    def test_dtype(self):
        # 定义一个继承自torch.jit.ScriptModule的模型类
        class MyModel(torch.jit.ScriptModule):
            # 定义一个脚本方法，实现模型的前向传播
            @torch.jit.script_method
            def forward(self, input, other):
                # 将输入张量的数据类型转换为`other`的数据类型后相加，并返回结果
                return input.to(dtype=other.dtype) + other

        # 创建两个随机张量作为输入
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        # 运行测试函数，验证MyModel类的行为是否符合预期
        self.run_test(MyModel(), (x, y))

    # 定义一个测试类，用于测试数据类型相等性判断
    def test_dtype_eq(self):
        # 定义一个继承自torch.jit.ScriptModule的模型类
        class MyModel(torch.jit.ScriptModule):
            # 定义一个脚本方法，实现模型的前向传播
            @torch.jit.script_method
            def forward(self, input, other):
                # 如果输入张量和`other`张量的数据类型相同，返回它们的加法结果；否则返回输入张量
                if input.dtype == other.dtype:
                    return input + other
                return input

        # 创建两个随机张量作为输入
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        # 运行测试函数，验证MyModel类的行为是否符合预期
        self.run_test(MyModel(), (x, y))

    # 定义一个测试类，用于测试张量类型转换
    def test_cast_to(self):
        # 定义一个继承自torch.jit.ScriptModule的模型类
        class MyModule(torch.jit.ScriptModule):
            # 定义一个脚本方法，实现模型的前向传播
            @torch.jit.script_method
            def forward(self, input, other):
                # 将输入张量转换为`other`的类型后相加，并返回结果
                return input.to(other) + other

        # 创建一个随机张量和一个torch.tensor类型的张量作为输入
        x = torch.randn(2, 3, 4)
        y = torch.tensor([1], dtype=torch.int64)
        # 创建模型实例
        model = MyModule()
        # 运行测试函数，验证MyModule类的行为是否符合预期
        self.run_test(model, (x, y))

    # 定义一个测试类，用于测试将张量转换为布尔类型
    def test_cast_to_bool(self):
        # 定义一个继承自torch.nn.Module的模型类
        class MyModule(torch.nn.Module):
            # 实现模型的前向传播
            def forward(self, input, other):
                # 将输入张量转换为`other`的布尔类型后连接，并返回结果
                return torch.cat((input.to(other), other), 0)

        # 创建一个随机张量和一个全零张量作为输入
        x = torch.randn(2, 3, 4)
        y = torch.zeros([2, 3, 4], dtype=torch.bool)
        # 创建模型实例
        model = MyModule()
        # 运行测试函数，验证MyModule类的行为是否符合预期
        self.run_test(model, (x, y))

    # 当ONNX支持的操作版本大于等于13时，测试张量类型转换为bfloat16
    @skipIfUnsupportedMinOpsetVersion(13)
    def test_cast_type_as_with_bfloat16(self):
        # 定义一个继承自torch.nn.Module的模型类
        class MyModule(torch.nn.Module):
            # 实现模型的前向传播
            def forward(self, x):
                # 创建一个数据类型为bfloat16的全1张量
                y = torch.ones((3, 4), dtype=torch.bfloat16)
                # 将输入张量转换为y的数据类型后返回，并将结果转换为torch.float16类型
                x = x.type_as(y)
                return x.to(dtype=torch.float16)

        # 创建一个全1张量作为输入
        x = torch.ones(3, 4, dtype=torch.float16)
        # 创建模型实例
        model = MyModule()
        # 运行测试函数，验证MyModule类的行为是否符合预期
        self.run_test(model, x)

    # 当ONNX支持的操作版本大于等于9时，测试张量类型转换
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_type_as(self):
        # 定义一个继承自torch.nn.Module的模型类
        class MyModule(torch.nn.Module):
            # 实现模型的前向传播
            def forward(self, x):
                # 创建一个浮点数张量
                y = torch.tensor([1.0])
                # 返回输入张量转换为y的数据类型后的结果
                return x.type_as(y)

        # 创建三个不同数据类型的张量作为输入
        a = torch.tensor([True, False], dtype=torch.bool)
        b = torch.randn(3, 4, dtype=torch.double)
        c = torch.ones((2, 2), dtype=torch.int64)
        # 创建模型实例
        model = MyModule()
        # 运行测试函数，验证MyModule类的行为是否符合预期
        self.run_test(model, a)
        self.run_test(model, b)
        self.run_test(model, c)

    # 当ONNX支持的操作版本大于等于9时，测试布尔类型全1张量的逻辑运算
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_ones_bool(self):
        # 定义一个继承自torch.nn.Module的模型类
        class MyModule(torch.nn.Module):
            # 实现模型的前向传播
            def forward(self, input):
                # 创建一个与输入张量形状相同的全1布尔类型张量
                true = torch.ones(input.shape, dtype=torch.bool)
                # 返回输入张量与true的按位与运算结果
                return input.to(true) & true

        # 创建一个随机张量作为输入
        x = torch.randn(2, 3, 4)
        # 创建模型实例
        model = MyModule()
        # 运行测试函数，验证MyModule类的行为是否符合预期
        self.run_test(model, x)

    # 定义一个测试类，用于测试对数运算
    def test_log(self):
        # 定义一个继承自torch.nn.Module的模型类
        class Log(torch.nn.Module):
            # 实现模型的前向传播
            def forward(self, input):
                # 返回输入张量的对数
                return torch.log(input)

        # 创建一个随机张量作为输入
        x = torch.rand(2, 3, 4)
        # 创建模型实例
        model = Log()
        # 运行测试函数，验证Log类的行为是否符合预期
        self.run_test(model, x)
    # 定义一个测试函数，用于测试 torch.log1p 函数
    def test_log1p(self):
        # 定义 Log1p 类，继承自 torch.nn.Module
        class Log1p(torch.nn.Module):
            # 重写 forward 方法，计算输入张量的对数(1 + x)
            def forward(self, input):
                return torch.log1p(input)

        # 创建一个形状为 (2, 3, 4) 的随机张量
        x = torch.rand(2, 3, 4)
        # 实例化 Log1p 模型
        model = Log1p()
        # 运行测试，验证模型输出与输入是否一致
        self.run_test(model, x)

    # 定义一个测试函数，用于测试 torch.log10 函数
    def test_log10(self):
        # 定义 Log10 类，继承自 torch.nn.Module
        class Log10(torch.nn.Module):
            # 重写 forward 方法，计算输入张量的以 10 为底的对数
            def forward(self, input):
                return torch.log10(input)

        # 创建一个形状为 (2, 3, 4) 的随机张量
        x = torch.rand(2, 3, 4)
        # 实例化 Log10 模型
        model = Log10()
        # 运行测试，验证模型输出与输入是否一致
        self.run_test(model, x)

    # 定义一个测试函数，用于测试 torch.log2 函数
    def test_log2(self):
        # 定义 Log2 类，继承自 torch.nn.Module
        class Log2(torch.nn.Module):
            # 重写 forward 方法，计算输入张量的以 2 为底的对数
            def forward(self, input):
                return torch.log2(input)

        # 创建一个值为 1.0 的张量
        x = torch.tensor(1.0)
        # 实例化 Log2 模型
        model = Log2()
        # 运行测试，验证模型输出与输入是否一致
        self.run_test(model, x)

    # 跳过不支持 Opset 版本小于 11 的测试
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_round(self):
        # 定义 Round 类，继承自 torch.nn.Module
        class Round(torch.nn.Module):
            # 重写 forward 方法，对输入张量进行四舍五入
            def forward(self, x):
                return torch.round(x)

        # 创建一个包含浮点数的张量，要求梯度计算
        x = torch.tensor([0.9920, -1.0362, -1.5000, 3.5000], requires_grad=True)
        # 运行测试，验证模型输出与输入是否一致
        self.run_test(Round(), x)

        # 创建一个包含整数的张量
        int_x = torch.tensor([9920, 1036, -1500, 35], dtype=torch.int32)
        # 运行测试，验证模型输出与输入是否一致
        self.run_test(Round(), int_x)

    # 跳过不支持 Opset 版本小于 11 的测试
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_round_with_decimals(self):
        # 定义 Round 类，继承自 torch.nn.Module
        class Round(torch.nn.Module):
            # 初始化方法，接受一个 decimals 参数
            def __init__(self, decimals):
                super().__init__()
                self.decimals = decimals

            # 重写 forward 方法，对输入张量进行指定小数位数的四舍五入
            def forward(self, x):
                return torch.round(x, decimals=self.decimals)

        # 创建一个包含浮点数的张量
        x = torch.tensor([0.9920, -1234.0362, -1.58960, 3.5000])
        # 遍历不同的小数位数进行测试
        for decimals in (0, -2, 3):
            # 实例化 Round 模型
            self.run_test(Round(decimals), x)

    # 跳过不支持 Opset 版本小于 17 的测试
    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_default(self):
        # 定义 STFT 类，继承自 torch.nn.Module
        class STFT(torch.nn.Module):
            # 重写 forward 方法，计算输入张量的短时傅里叶变换
            def forward(self, x):
                n_fft = 16
                return torch.stft(x, n_fft=n_fft, center=False, return_complex=False)

        # 创建一个形状为 (1, 32) 的随机张量，要求梯度计算
        x = torch.randn((1, 32), requires_grad=True)
        # 运行测试，验证模型输出与输入是否一致，指定允许的绝对误差
        self.run_test(STFT(), x, atol=1e-6)

    # 跳过不支持 Opset 版本小于 17 的测试
    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_hop_length(self):
        # 定义 STFT 类，继承自 torch.nn.Module
        class STFT(torch.nn.Module):
            # 重写 forward 方法，计算输入张量的短时傅里叶变换
            def forward(self, x):
                n_fft = 16
                hop_length = 4
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    hop_length=hop_length,
                    return_complex=False,
                )

        # 创建一个形状为 (1, 32) 的随机张量，要求梯度计算
        x = torch.randn((1, 32), requires_grad=True)
        # 运行测试，验证模型输出与输入是否一致，指定允许的绝对误差
        self.run_test(STFT(), x, atol=1e-6)
    @skipIfUnsupportedMinOpsetVersion(17)
    # 使用装饰器，跳过不支持的最小 Opset 版本为 17 的测试用例

    def test_stft_non_divisible_hop_length(self):
        # 定义一个测试用例，测试当 hop_length 不是 n_fft 的整数倍时的情况

        class STFT(torch.nn.Module):
            def forward(self, x):
                n_fft = 16
                hop_length = 5
                # 执行短时傅里叶变换（STFT），设置参数 n_fft 为 16，hop_length 为 5
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    hop_length=hop_length,
                    return_complex=False,
                )

        x = torch.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    @skipIfUnsupportedMinOpsetVersion(17)
    # 使用装饰器，跳过不支持的最小 Opset 版本为 17 的测试用例

    def test_stft_window_int_same_size(self):
        # 定义一个测试用例，测试当 win_length 与 n_fft 相同时的情况

        class STFT(torch.nn.Module):
            def forward(self, x):
                n_fft = 16
                win_length = 16
                # 执行短时傅里叶变换（STFT），设置参数 n_fft 和 win_length 均为 16
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    win_length=win_length,
                    return_complex=False,
                )

        x = torch.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    @skipIfUnsupportedMinOpsetVersion(17)
    # 使用装饰器，跳过不支持的最小 Opset 版本为 17 的测试用例

    def test_stft_window_int_different_size(self):
        # 定义一个测试用例，测试当 win_length 小于 n_fft 时的情况

        class STFT(torch.nn.Module):
            def forward(self, x):
                n_fft = 16
                win_length = 9
                # 执行短时傅里叶变换（STFT），设置参数 n_fft 为 16，win_length 为 9
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    win_length=win_length,
                    return_complex=False,
                )

        x = torch.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    @skipIfUnsupportedMinOpsetVersion(17)
    # 使用装饰器，跳过不支持的最小 Opset 版本为 17 的测试用例

    def test_stft_window_custom(self):
        # 定义一个测试用例，测试自定义窗函数时的情况

        class STFT(torch.nn.Module):
            def forward(self, x):
                n_fft = 16
                window = torch.hann_window(16)
                # 执行短时傅里叶变换（STFT），设置参数 n_fft 为 16，window 使用汉宁窗口长度为 16
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    window=window,
                    return_complex=False,
                )

        x = torch.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    @skipIfUnsupportedMinOpsetVersion(17)
    # 使用装饰器，跳过不支持的最小 Opset 版本为 17 的测试用例

    def test_stft_wrong_custom_window_size(self):
        # 定义一个测试用例，测试当自定义窗口大小小于 n_fft 时的情况

        class STFT(torch.nn.Module):
            def forward(self, x):
                n_fft = 16
                window = torch.hann_window(10)
                # 执行短时傅里叶变换（STFT），设置参数 n_fft 为 16，使用长度为 10 的汉宁窗口
                return torch.stft(
                    x, n_fft=n_fft, window=window, center=False, return_complex=False
                )

        x = torch.randn((1, 32), requires_grad=True)
        # 确保测试会抛出 AssertionError 或 RuntimeError 异常
        with self.assertRaises((AssertionError, RuntimeError)):
            self.run_test(STFT(), x)
    def test_stft_wrong_window_length(self):
        # 定义一个测试函数，用于测试当窗口长度设置不正确时的情况
        class STFT(torch.nn.Module):
            def forward(self, x):
                n_fft = 16
                win_len = 17
                # 使用 torch.stft 函数计算短时傅里叶变换
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    win_length=win_len,
                    center=False,
                    return_complex=False,
                )

        # 生成一个随机张量 x，形状为 (1, 32)，要求梯度计算
        x = torch.randn((1, 32), requires_grad=True)
        # 使用 assertRaises 断言捕获 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            # 调用 self.run_test 方法执行 STFT 类的实例化对象和 x 作为参数的测试
            self.run_test(STFT(), x)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_window_size_with_win_len(self):
        # 定义一个测试函数，用于测试设置了窗口长度参数时的短时傅里叶变换
        class STFT(torch.nn.Module):
            def forward(self, x):
                n_fft = 16
                # 创建长度为 10 的汉宁窗口
                window = torch.hann_window(10)
                win_len = 10
                # 使用 torch.stft 函数计算短时傅里叶变换
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    window=window,
                    win_length=win_len,
                    center=False,
                    return_complex=False,
                )

        # 生成一个随机张量 x，形状为 (1, 32)，要求梯度计算
        x = torch.randn((1, 32), requires_grad=True)
        # 调用 self.run_test 方法执行 STFT 类的实例化对象和 x 作为参数的测试，设定允许误差为 1e-6
        self.run_test(STFT(), x, atol=1e-6)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_one_dimension(self):
        # 定义一个测试函数，用于测试一维输入的短时傅里叶变换
        class STFT(torch.nn.Module):
            def forward(self, x):
                n_fft = 16
                # 使用 torch.stft 函数计算短时傅里叶变换
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    return_complex=False,
                )

        # 生成一个随机张量 x，形状为 (32)，要求梯度计算
        x = torch.randn((32), requires_grad=True)
        # 调用 self.run_test 方法执行 STFT 类的实例化对象和 x 作为参数的测试，设定允许误差为 1e-6
        self.run_test(STFT(), x, atol=1e-6)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_wrong_input_size(self):
        # 定义一个测试函数，用于测试输入尺寸不正确的情况下的短时傅里叶变换
        class STFT(torch.nn.Module):
            def forward(self, x):
                n_fft = 16
                # 使用 torch.stft 函数计算短时傅里叶变换
                return torch.stft(x, n_fft=n_fft, center=False, return_complex=False)

        # 生成一个随机张量 x，形状为 (1, 1, 32)，要求梯度计算
        x = torch.randn((1, 1, 32), requires_grad=True)
        # 使用 assertRaises 断言捕获 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            # 调用 self.run_test 方法执行 STFT 类的实例化对象和 x 作为参数的测试
            self.run_test(STFT(), x)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_wrong_return_complex(self):
        # 定义一个测试函数，用于测试返回复数值时的短时傅里叶变换
        class STFT(torch.nn.Module):
            def forward(self, x):
                n_fft = 16
                # 使用 torch.stft 函数计算短时傅里叶变换
                return torch.stft(x, n_fft=n_fft, center=False, return_complex=True)

        # 生成一个随机张量 x，形状为 (1, 32)，要求梯度计算
        x = torch.randn((1, 32), requires_grad=True)
        # 使用 assertRaises 断言捕获 errors.SymbolicValueError 异常
        with self.assertRaises(errors.SymbolicValueError):
            # 调用 self.run_test 方法执行 STFT 类的实例化对象和 x 作为参数的测试
            self.run_test(STFT(), x)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_normalize(self):
        # 定义一个测试函数，用于测试启用归一化参数时的短时傅里叶变换
        class STFT(torch.nn.Module):
            def forward(self, x):
                n_fft = 16
                # 使用 torch.stft 函数计算短时傅里叶变换
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    normalized=True,
                    return_complex=False,
                )

        # 生成一个随机张量 x，形状为 (32)，要求梯度计算
        x = torch.randn((32), requires_grad=True)
        # 调用 self.run_test 方法执行 STFT 类的实例化对象和 x 作为参数的测试，设定允许误差为 1e-6
        self.run_test(STFT(), x, atol=1e-6)
    def test_stft_not_onesided(self):
        # 定义一个内部类 STFT，继承自 torch.nn.Module
        class STFT(torch.nn.Module):
            # 定义 forward 方法，接收输入 x
            def forward(self, x):
                # 设置 STFT 的窗口大小为 n_fft = 16，并执行 STFT 变换
                n_fft = 16
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    onesided=False,
                    return_complex=False,
                )

        # 生成一个形状为 (32,) 的随机张量 x，并标记为需要计算梯度
        x = torch.randn((32), requires_grad=True)
        # 调用 run_test 方法，测试 STFT 模型对输入 x 的输出，设置容差为 1e-6
        self.run_test(STFT(), x, atol=1e-6)

    def test_constant_pad(self):
        # 创建一个 ConstantPad1d 模型，向每侧填充宽度为 2，填充值为 3.5
        model = torch.nn.ConstantPad1d(2, 3.5)
        # 生成一个形状为 (2, 4, 4) 的随机张量 x
        x = torch.randn(2, 4, 4)
        # 使用 run_test 方法测试 ConstantPad1d 模型对输入 x 的输出

        self.run_test(model, x)

        # 创建一个 ConstantPad2d 模型，向上、下、左、右侧分别填充 (3, 0, 2, 1) 个元素，填充值为 3.5
        model = torch.nn.ConstantPad2d((3, 0, 2, 1), 3.5)
        # 生成一个形状为 (2, 2, 4, 4) 的随机张量 x
        x = torch.randn(2, 2, 4, 4)
        # 使用 run_test 方法测试 ConstantPad2d 模型对输入 x 的输出
        self.run_test(model, x)

    @common_utils.parametrize(
        "pad",
        [
            common_utils.subtest([2, 4], name="scalar_list"),  # 测试标量列表形式的填充参数
            common_utils.subtest(
                [
                    torch.tensor(2, dtype=torch.int64),
                    torch.tensor(4, dtype=torch.int64),
                ],
                name="scalar_tensor_list",  # 测试张量标量列表形式的填充参数
            ),
        ],
    )
    @skipIfUnsupportedMinOpsetVersion(11)  # 如果当前 Opset 版本小于 11，则跳过测试
    def test_pad_types(self, pad):
        # 定义一个内部类 Pad，继承自 torch.nn.Module
        class Pad(torch.nn.Module):
            # 定义 forward 方法，接收输入 x 和填充参数 pad（列表形式）
            def forward(self, x, pad: List[int]):
                # 对输入 x 进行填充操作，使用 torch.nn.functional.pad
                return torch.nn.functional.pad(x, pad)

        # 生成一个形状为 (2, 2, 4, 4) 的随机张量 x
        x = torch.randn(2, 2, 4, 4)
        # 使用 run_test 方法测试 Pad 模型对输入 x 和填充参数 pad 的输出
        self.run_test(Pad(), (x, pad))

    @skipIfUnsupportedMinOpsetVersion(11)  # 如果当前 Opset 版本小于 11，则跳过测试
    def test_pad_circular(self):
        # 定义一个内部类 PadModel，继承自 torch.nn.Module
        class PadModel(torch.nn.Module):
            # 定义 forward 方法，接收输入 x
            def forward(self, x):
                # 对输入 x 进行循环填充操作，填充参数为 (1, 2, 1, 2)，填充模式为 "circular"
                out = torch.nn.functional.pad(x, (1, 2, 1, 2), mode="circular")
                return out

        # 生成一个形状为 (2, 3, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 3, 4)
        # 使用 run_test 方法测试 PadModel 模型对输入 x 的输出
        self.run_test(PadModel(), (x))

    @skipIfUnsupportedMinOpsetVersion(11)  # 如果当前 Opset 版本小于 11，则跳过测试
    def test_pad_circular_negative(self):
        # 定义一个内部类 PadModel，继承自 torch.nn.Module
        class PadModel(torch.nn.Module):
            # 定义 forward 方法，接收输入 x
            def forward(self, x):
                # 对输入 x 进行循环填充操作，填充参数为 (-1, -2)，填充模式为 "circular"
                out = torch.nn.functional.pad(x, (-1, -2), mode="circular")
                return out

        # 生成一个形状为 (2, 3, 6) 的随机张量 x
        x = torch.randn(2, 3, 6)
        # 使用 run_test 方法测试 PadModel 模型对输入 x 的输出
        self.run_test(PadModel(), (x))

    @skipIfUnsupportedMinOpsetVersion(11)  # 如果当前 Opset 版本小于 11，则跳过测试
    def test_pad_circular_dynamic_axes(self):
        # 定义一个内部类 PadModel，继承自 torch.nn.Module
        class PadModel(torch.nn.Module):
            # 定义 forward 方法，接收输入 x
            def forward(self, x):
                # 对输入 x 进行循环填充操作，填充参数为 (2, 1, 2, 1)，填充模式为 "circular"
                out = torch.nn.functional.pad(x, (2, 1, 2, 1), mode="circular")
                return out

        # 生成一个形状为 (4, 3, 5, 6) 的随机张量 x
        x = torch.randn(4, 3, 5, 6)
        # 使用 run_test 方法测试 PadModel 模型对输入 x 的输出，
        # 并指定输入名为 "input_1"，动态轴为 {"input_1": [0, 1, 2, 3]}
        self.run_test(
            PadModel(),
            x,
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2, 3]},
        )

    @skipIfUnsupportedMaxOpsetVersion(10)  # 如果当前 Opset 版本大于 10，则跳过测试
    @skipScriptTest()  # TODO: symbolic_opset9 中的逻辑不处理脚本
    # 定义一个名为 test_unsupported_pad 的测试方法
    def test_unsupported_pad(self):
        # 定义一个名为 Pad 的内部类，继承自 torch.nn.Module
        class Pad(torch.nn.Module):
            # 实现前向传播方法，接受输入 x 和一个整数列表 pad
            def forward(self, x, pad: List[int]):
                # 调用 PyTorch 的函数式 API 对输入 x 进行填充操作
                return torch.nn.functional.pad(x, pad)

        # 创建一个形状为 (2, 2, 4, 4) 的随机张量 x
        x = torch.randn(2, 2, 4, 4)
        # 创建一个整数列表 y，包含 [2, 4]
        y = [2, 4]

        # 使用断言检测是否抛出指定异常，并包含特定错误消息
        with self.assertRaisesRegex(
            RuntimeError,
            (
                "Unsupported: ONNX export of Pad.*"
                + "The sizes of the padding must be constant"
            ),
        ):
            # 调用 self.run_test 方法，传入 Pad 实例和输入参数 (x, y)
            self.run_test(Pad(), (x, y))

    # 标记为跳过不支持 Opset 版本小于 9 的测试方法
    @skipIfUnsupportedMinOpsetVersion(9)
    # 标记为跳过不支持 Opset 版本小于 11 的测试方法
    @skipIfUnsupportedMinOpsetVersion(11)
    # 定义一个名为 test_uninitialized 的测试方法
    def test_uninitialized(self):
        # 定义一个名为 UninitializedModel 的内部类，继承自 torch.nn.Module
        class UninitializedModel(torch.nn.Module):
            # 实现前向传播方法，接受输入 y
            def forward(self, y):
                # 如果输入 y 的第二维度小于 5
                if y.shape[1] < 5:
                    # 如果 y 的第一维度为 1
                    if y.size(0) == 1:
                        # 对 y 进行加法操作
                        y = y + 4
                    else:
                        # 否则直接返回 y
                        return y
                # 返回处理后的 y
                return y

        # 创建一个形状为 (3, 4)、数据类型为 torch.int 的全 1 张量 x
        x = torch.ones((3, 4), dtype=torch.int)
        # 调用 self.run_test 方法，传入 UninitializedModel 实例和输入参数 x
        self.run_test(UninitializedModel(), x)

    # 标记为跳过不支持 Opset 版本小于 11 的测试方法
    @skipIfUnsupportedMinOpsetVersion(11)
    # 定义一个名为 test_uninitialized_dynamic 的测试方法
    def test_uninitialized_dynamic(self):
        # 定义一个名为 UninitializedModel 的内部类，继承自 torch.nn.Module
        class UninitializedModel(torch.nn.Module):
            # 实现前向传播方法，接受输入 y
            def forward(self, y):
                # 如果输入 y 的第二维度小于 5
                if y.shape[1] < 5:
                    # 如果 y 的第一维度为 1
                    if y.size(0) == 1:
                        # 对 y 进行加法操作
                        y = y + 4
                    else:
                        # 否则直接返回 y
                        return y
                # 返回处理后的 y
                return y

        # 创建一个形状为 (3, 4)、数据类型为 torch.int 的全 1 张量 x
        x = torch.ones((3, 4), dtype=torch.int)
        # 创建一个形状为 (6, 7)、数据类型为 torch.int 的全 1 张量 y
        y = torch.ones((6, 7), dtype=torch.int)
        # 调用 self.run_test 方法，传入 UninitializedModel 实例和输入参数 x，
        # 同时传入额外的测试输入 y、设置输入名称和动态轴信息
        self.run_test(
            UninitializedModel(),
            x,
            additional_test_inputs=[y],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1]},
        )

    # 标记为跳过不支持 Opset 版本小于 14 的测试方法
    @skipIfUnsupportedMinOpsetVersion(14)
    # 定义一个名为 test_uninitialized_tensorList 的测试方法
    def test_uninitialized_tensorList(self):
        # 定义一个名为 UninitializedTensorListModel 的内部类，继承自 torch.nn.Module
        class UninitializedTensorListModel(torch.nn.Module):
            # 实现前向传播方法，接受输入 x
            def forward(self, x):
                # 如果输入列表的第一个元素的第一维度小于 5
                if x[0].shape[0] < 5:
                    # 如果 x 的第一维度为 1
                    if x.size(0) == 1:
                        # 对 x 进行加法操作
                        x = x + 4
                    else:
                        # 否则将 x 包装为列表返回
                        return [x]
                # 返回包含 x 的列表
                return [x]

        # 创建一个形状为 (3, 4)、数据类型为 torch.int 的全 1 张量 x
        x = torch.ones((3, 4), dtype=torch.int)
        # 使用 torch.jit.script 方法对 UninitializedTensorListModel 类进行脚本化
        # 并调用 self.run_test 方法，传入脚本化的 UninitializedTensorListModel 实例和输入参数 x
        self.run_test(torch.jit.script(UninitializedTensorListModel()), x)

    # 标记为跳过不支持 Opset 版本小于 14 的测试方法
    @skipIfUnsupportedMinOpsetVersion(14)
    # 定义一个名为 test_uninitialized_tensorList_dynamic 的测试方法
    def test_uninitialized_tensorList_dynamic(self):
        # 定义一个名为 UninitializedTensorListModel 的内部类，继承自 torch.nn.Module
        class UninitializedTensorListModel(torch.nn.Module):
            # 实现前向传播方法，接受输入 x
            def forward(self, x):
                # 如果输入列表的第一个元素的第一维度小于 5
                if x[0].shape[0] < 5:
                    # 如果 x 的第一维度为 1
                    if x.size(0) == 1:
                        # 对 x 进行加法操作
                        x += x
                    else:
                        # 否则将 x 转换为列表返回
                        return list(x)
                # 返回包含 x 的列表
                return list(x)

        # 创建一个形状为 (3, 4)、数据类型为 torch.double 的全 1 张量 x
        x = torch.ones((3, 4), dtype=torch.double)
        # 使用 torch.jit.script 方法对 UninitializedTensorListModel 类进行脚本化
        # 并调用 self.run_test 方法，传入脚本化的 UninitializedTensorListModel 实例和输入参数 x，
        # 同时传入输入名称和动态轴信息
        self.run_test(
            torch.jit.script(UninitializedTensorListModel()),
            x,
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1]},
        )
    # 使用装饰器，检查当前的运行时是否支持最小 opset 版本为 14，若不支持则跳过测试
    @skipIfUnsupportedMinOpsetVersion(14)
    def test_uninitialized_intList(self):
        # 定义一个继承自 torch.nn.Module 的类 UninitializedListModel
        class UninitializedListModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 创建一个列表 y，其中元素为从 0 到 x 的行数的整数
                y = list(range(x.size(0)))
                # 如果列表 y 的第一个元素小于 5
                if y[0] < 5:
                    # 如果输入 x 的行数等于 3，则在列表 y 的末尾添加元素 10
                    if x.size(0) == 3:
                        y.append(10)
                    else:
                        return y
                return y

        # 创建一个形状为 (3, 4) 的全为 1 的整数张量 x
        x = torch.ones((3, 4), dtype=torch.int)
        # 运行测试，对 torch.jit.script(UninitializedListModel()) 进行测试
        self.run_test(
            torch.jit.script(UninitializedListModel()),
            x,
            input_names=["input_1"],  # 输入名称为 "input_1"
            dynamic_axes={"input_1": [0, 1]},  # "input_1" 的动态维度为 [0, 1]
        )

    # 在 opset >= 14 的情况下，支持对序列的 Identity 操作
    @skipIfUnsupportedMinOpsetVersion(14)
    def test_uninitialized_tensorList_shape(self):
        # 定义一个继承自 torch.nn.Module 的类 UninitializedModel
        class UninitializedModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 如果输入 x 的第二个维度小于 5
                if x.shape[1] < 5:
                    # 如果 x 的行数等于 1，则将 x 的每个元素加上 4
                    if x.size(0) == 1:
                        x = x + 4
                    else:
                        # 否则将 x 转换为列表，并在末尾添加 x 自身，然后返回
                        x_list = list(x)
                        x_list.append(x)
                        return x_list
                # 返回包含两个 x 张量的列表
                return [x, x]

        # 创建两个形状为 (3, 4) 的全为 1 的整数张量 x 和 y
        x = torch.ones((3, 4), dtype=torch.int)
        y = torch.ones((4, 6), dtype=torch.int)
        # 运行测试，对 torch.jit.script(UninitializedModel()) 进行测试
        self.run_test(
            torch.jit.script(UninitializedModel()),
            x,
            additional_test_inputs=[y],  # 额外的测试输入为 y
            input_names=["input_1"],  # 输入名称为 "input_1"
            dynamic_axes={"input_1": [0, 1]},  # "input_1" 的动态维度为 [0, 1]
        )

    # 在 opset >= 13 的情况下，支持循环中的序列类型作为循环传递的依赖项
    @skipIfUnsupportedMinOpsetVersion(13)
    def test_sequance_loopcarried(self):
        # 定义一个继承自 torch.nn.Module 的类 SequanceLoopModel
        class SequanceLoopModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 初始化一个空列表 outputs
                outputs = []
                # 循环三次，每次将 x 添加到 outputs 中
                for i in range(3):
                    outputs += [x]
                # 将 outputs 列表转置，并将维度 0 和 1 互换后返回
                return torch.stack(outputs).transpose(0, 1)

        # 创建一个形状为 (3, 4) 的全为 1 的整数张量 x
        x = torch.ones((3, 4), dtype=torch.int)
        # 运行测试，对 torch.jit.script(SequanceLoopModel()) 进行测试
        self.run_test(torch.jit.script(SequanceLoopModel()), x)

    # 测试反射填充（ReflectionPad1d 和 ReflectionPad2d）
    def test_reflection_pad(self):
        # 创建一个 ReflectionPad1d 模型，填充数为 2
        model = torch.nn.ReflectionPad1d(2)
        # 创建一个形状为 (2, 4, 4) 的随机张量 x
        x = torch.randn(2, 4, 4)
        # 运行测试，对 model 进行测试
        self.run_test(model, x)

        # 创建一个 ReflectionPad2d 模型，填充数为 (3, 0, 2, 1)
        model = torch.nn.ReflectionPad2d((3, 0, 2, 1))
        # 创建一个形状为 (2, 2, 4, 4) 的随机张量 x
        x = torch.randn(2, 2, 4, 4)
        # 运行测试，对 model 进行测试
        self.run_test(model, x)

    # 测试复制填充（ReplicationPad1d 和 ReplicationPad2d）
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_replication_pad(self):
        # 创建一个 ReplicationPad1d 模型，填充数为 2
        model = torch.nn.ReplicationPad1d(2)
        # 创建一个形状为 (2, 4, 4) 的随机张量 x
        x = torch.randn(2, 4, 4)
        # 运行测试，对 model 进行测试
        self.run_test(model, x)

        # 创建一个 ReplicationPad2d 模型，填充数为 (3, 0, 2, 1)
        model = torch.nn.ReplicationPad2d((3, 0, 2, 1))
        # 创建一个形状为 (2, 2, 4, 4) 的随机张量 x
        x = torch.randn(2, 2, 4, 4)
        # 运行测试，对 model 进行测试
        self.run_test(model, x)
    # 定义一个测试函数 test_im2col，用于测试图像转换列矩阵的功能
    def test_im2col(self):
        # 定义一个内部类 Unfold，继承自 torch.nn.Module
        class Unfold(torch.nn.Module):
            # 实现 forward 方法，用于前向传播
            def forward(self, input):
                # 调用 torch.nn.functional.unfold 函数，将输入张量按照指定参数展开
                return (
                    torch.nn.functional.unfold(
                        input, kernel_size=(10, 15), dilation=2, padding=5, stride=3
                    ),
                    torch.nn.functional.unfold(
                        input, kernel_size=(2, 2), dilation=1, padding=0, stride=3
                    ),
                    torch.nn.functional.unfold(
                        input, kernel_size=(1, 1), dilation=5, padding=2, stride=3
                    ),
                )

        # 创建一个形状为 (1, 1, 200, 100) 的随机张量 x
        x = torch.rand(1, 1, 200, 100)
        # 调用 self.run_test 方法，执行 Unfold 类的前向传播，并传入随机张量 x 进行测试

    # 装饰器 @skipIfNoLapack：如果没有 LAPACK 库，则跳过该测试用例
    # 装饰器 @skipIfUnsupportedMinOpsetVersion(11)：如果不支持最小操作集版本 11，则跳过该测试用例
    def test_det(self):
        # 定义一个内部类 Det，继承自 torch.nn.Module
        class Det(torch.nn.Module):
            # 实现 forward 方法，用于前向传播
            def forward(self, x):
                # 调用 torch.linalg.det 函数，计算输入张量 x 的行列式
                return torch.linalg.det(x)

        # 创建一个形状为 (2, 3, 5, 5) 的标准正态分布张量 x
        x = torch.randn(2, 3, 5, 5)
        # 调用 self.run_test 方法，执行 Det 类的前向传播，并传入标准正态分布张量 x 进行测试
    # 定义一个测试方法，用于测试 torch.linalg.norm 函数在不同情况下的行为
    def test_linalg_norm(self):
        # 定义一个单维度模型，继承自 torch.nn.Module
        class LinalgSingleDimModel(torch.nn.Module):
            # 初始化方法，接收一个 ord_val 参数
            def __init__(self, ord_val):
                super().__init__()
                # 将 ord_val 参数赋值给实例的 ord 属性
                self.ord = ord_val

            # 前向传播方法，接收输入 x
            def forward(self, x):
                # 调用 torch.linalg.norm 计算输入 x 的范数，使用 self.ord 参数指定范数类型，对第一维进行操作
                return torch.linalg.norm(x, ord=self.ord, dim=1)

        # 生成一个形状为 (2, 3, 5, 5) 的随机张量 x
        x = torch.randn(2, 3, 5, 5)
        # 使用 self.run_test 方法测试 LinalgSingleDimModel 的实例，传入随机张量 x
        self.run_test(LinalgSingleDimModel(None), x)
        self.run_test(LinalgSingleDimModel(2), x)
        self.run_test(LinalgSingleDimModel(float("inf")), x)
        self.run_test(LinalgSingleDimModel(-float("inf")), x)
        self.run_test(LinalgSingleDimModel(-4), x)
        self.run_test(LinalgSingleDimModel(1.5), x)

        # 定义一个多维度模型，继承自 torch.nn.Module
        class LinalgMultiDimModel(torch.nn.Module):
            # 初始化方法，接收一个 ord_val 参数
            def __init__(self, ord_val):
                super().__init__()
                # 将 ord_val 参数赋值给实例的 ord 属性
                self.ord = ord_val

            # 前向传播方法，接收输入 x
            def forward(self, x):
                # 调用 torch.linalg.norm 计算输入 x 的范数，使用 self.ord 参数指定范数类型，对 (0, 2) 维度进行操作
                return torch.linalg.norm(x, ord=self.ord, dim=(0, 2))

        # 生成一个形状为 (2, 3, 5, 5) 的随机张量 x
        x = torch.randn(2, 3, 5, 5)
        # 使用 self.run_test 方法测试 LinalgMultiDimModel 的实例，传入随机张量 x
        self.run_test(LinalgMultiDimModel("fro"), x)
        self.run_test(LinalgMultiDimModel(float("inf")), x)
        self.run_test(LinalgMultiDimModel(-float("inf")), x)
        self.run_test(LinalgMultiDimModel(1), x)
        self.run_test(LinalgMultiDimModel(-1), x)

        # 定义一个无指定维度和范数类型的模型，继承自 torch.nn.Module
        class LinalgNoDimNoOrdModel(torch.nn.Module):
            # 前向传播方法，接收输入 x
            def forward(self, x):
                # 调用 torch.linalg.norm 计算输入 x 的范数，未指定范数类型和维度
                return torch.linalg.norm(x)

        # 生成一个形状为 (2, 3, 5, 5) 的随机张量 x
        x = torch.randn(2, 3, 5, 5)
        # 使用 self.run_test 方法测试 LinalgNoDimNoOrdModel 的实例，传入随机张量 x
        self.run_test(LinalgNoDimNoOrdModel(), x)
        # 生成一个形状为 (2, 3) 的随机张量 y
        y = torch.randn(2, 3)
        # 使用 self.run_test 方法测试 LinalgNoDimNoOrdModel 的实例，传入随机张量 y
        self.run_test(LinalgNoDimNoOrdModel(), y)
        # 生成一个形状为 (2) 的随机张量 z
        z = torch.randn(2)
        # 使用 self.run_test 方法测试 LinalgNoDimNoOrdModel 的实例，传入随机张量 z
        self.run_test(LinalgNoDimNoOrdModel(), z)

        # 定义一个单维度模型，继承自 torch.nn.Module
        class LinalgNoDim1DModel(torch.nn.Module):
            # 初始化方法，接收一个 ord_val 参数
            def __init__(self, ord_val):
                super().__init__()
                # 将 ord_val 参数赋值给实例的 ord 属性
                self.ord = ord_val

            # 前向传播方法，接收输入 x
            def forward(self, x):
                # 调用 torch.linalg.norm 计算输入 x 的范数，使用 self.ord 参数指定范数类型
                return torch.linalg.norm(x, ord=self.ord)

        # 生成一个形状为 (2) 的随机张量 x
        x = torch.randn(2)
        # 使用 self.run_test 方法测试 LinalgNoDim1DModel 的实例，传入随机张量 x
        self.run_test(LinalgNoDim1DModel(None), x)
        self.run_test(LinalgNoDim1DModel(2), x)
        self.run_test(LinalgNoDim1DModel(float("inf")), x)
        self.run_test(LinalgNoDim1DModel(-float("inf")), x)
        self.run_test(LinalgNoDim1DModel(-4), x)
        self.run_test(LinalgNoDim1DModel(1.5), x)

        # 定义一个二维模型，继承自 torch.nn.Module
        class LinalgNoDim2DModel(torch.nn.Module):
            # 初始化方法，接收一个 ord_val 参数
            def __init__(self, ord_val):
                super().__init__()
                # 将 ord_val 参数赋值给实例的 ord 属性
                self.ord = ord_val

            # 前向传播方法，接收输入 x
            def forward(self, x):
                # 调用 torch.linalg.norm 计算输入 x 的范数，使用 self.ord 参数指定范数类型
                return torch.linalg.norm(x, ord=self.ord)

        # 生成一个形状为 (2, 3) 的随机张量 x
        x = torch.randn(2, 3)
        # 使用 self.run_test 方法测试 LinalgNoDim2DModel 的实例，传入随机张量 x
        self.run_test(LinalgNoDim2DModel("fro"), x)
        self.run_test(LinalgNoDim2DModel(float("inf")), x)
        self.run_test(LinalgNoDim2DModel(-float("inf")), x)
        self.run_test(LinalgNoDim2DModel(1), x)
        self.run_test(LinalgNoDim2DModel(-1), x)
    # 定义测试用例：计算向量的零范数
    def test_linalg_vector_norm_zero(self):
        # 定义模型类，用于计算向量范数
        class LinalgVectorNormModel(torch.nn.Module):
            def __init__(self, ord_val):
                super().__init__()
                self.ord = ord_val

            # 前向传播函数，调用 torch.linalg.vector_norm 计算向量的指定范数
            def forward(self, x):
                return torch.linalg.vector_norm(x, ord=self.ord)

        # 生成随机输入张量
        x = torch.randn(2, 3, 5, 5)
        # 运行测试，传入零范数模型和输入张量
        self.run_test(LinalgVectorNormModel(0), x)

    # 定义测试用例：计算向量的指定范数
    def test_linalg_vector_norm(self):
        # 定义模型类，用于计算向量范数
        class LinalgVectorNormModel(torch.nn.Module):
            def __init__(self, ord_val, dim_info):
                super().__init__()
                self.ord = ord_val
                self.dim, self.keepdim = dim_info

            # 前向传播函数，调用 torch.linalg.vector_norm 计算向量的指定范数
            def forward(self, x):
                return torch.linalg.vector_norm(
                    x, ord=self.ord, dim=self.dim, keepdim=self.keepdim
                )

        # 生成随机输入张量
        x = torch.randn(2, 3, 5, 5)
        # 定义范数和维度的多种组合
        ord_options = [2, float("inf"), -float("inf"), -4, 1.5]
        dim_options = [(None, False), (1, False), ((1, 2), False), ((1, 2), True)]
        # 遍历不同的范数和维度组合，运行测试
        for ord_val in ord_options:
            for dim_info in dim_options:
                self.run_test(LinalgVectorNormModel(ord_val, dim_info), x)

    # 定义测试用例：计算矩阵的指定范数
    def test_linalg_matrix_norm(self):
        # 定义模型类，用于计算矩阵范数
        class LinalgMatrixNormModel(torch.nn.Module):
            def __init__(self, ord_val, dim_val=(-2, -1), keepdim_val=False):
                super().__init__()
                self.ord = ord_val
                self.dim = dim_val
                self.keepdim = keepdim_val

            # 前向传播函数，调用 torch.linalg.matrix_norm 计算矩阵的指定范数
            def forward(self, x):
                return torch.linalg.matrix_norm(
                    x, ord=self.ord, dim=self.dim, keepdim=self.keepdim
                )

        # 生成随机输入张量
        x = torch.randn(2, 3, 5, 5)
        # 定义范数的多种选项
        ord_options = ["fro", float("inf"), -float("inf"), 1, -1]
        # 遍历不同的范数选项，运行测试
        for ord_val in ord_options:
            self.run_test(LinalgMatrixNormModel(ord_val), x)
            self.run_test(LinalgMatrixNormModel(ord_val, (0, 2)), x)
            self.run_test(LinalgMatrixNormModel(ord_val, (0, 2), True), x)

    # 跳过不支持的最小操作集版本（条件为 Opset 9）
    @skipIfUnsupportedMinOpsetVersion(9)
    # 定义测试用例：计算向量的叉积
    def test_linalg_cross(self):
        # 定义模型类，用于计算向量的叉积
        class Cross(torch.nn.Module):
            # 前向传播函数，调用 torch.linalg.cross 计算向量的叉积
            def forward(self, x, y):
                return torch.linalg.cross(x, y, dim=1), torch.linalg.cross(x, y)

        # 生成随机输入张量
        x = torch.randn(5, 3, 2, 3)
        y = torch.randn(1, 3, 1, 3)
        # 运行测试，传入叉积模型和输入参数
        self.run_test(Cross(), input_args=(x, y))

    # 此测试检查 ONNX 图中的输出标量类型不应为 null
    # 参考：https://github.com/pytorch/pytorch/issues/28607
    @skipIfUnsupportedMinOpsetVersion(10)
    # 定义测试用例：对脚本进行追踪
    def test_trace_script(self):
        # 定义 Torch 脚本函数，用于从指定偏移量开始切片输入张量
        @torch.jit.script
        def center_slice_helper(input, h_offset):
            return input[:, h_offset:]

        # 定义模型类，用于对输入进行中心裁剪操作
        class CenterCrop(torch.nn.Module):
            # 前向传播函数，调用 center_slice_helper 函数
            def forward(self, input):
                return center_slice_helper(input, torch.tensor(input.shape[1] - 1))

        # 生成随机输入张量
        x = torch.randn(3, 4)
        # 运行测试，传入中心裁剪模型和输入张量
        self.run_test(CenterCrop(), x)

    # 跳过没有 LAPACK 支持的情况
    @skipIfNoLapack
    # 跳过不支持的最小操作集版本（条件为 Opset 11）
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_logdet(self):
        # 定义一个继承自 torch.nn.Module 的类 LogDet
        class LogDet(torch.nn.Module):
            # 实现 forward 方法，返回输入张量 x 的对数行列式
            def forward(self, x):
                return torch.logdet(x)

        # 创建一个形状为 (2, 3, 5, 5) 的随机张量 x
        x = torch.randn(2, 3, 5, 5)
        # 使用 self.run_test 方法运行 LogDet 模型的测试
        self.run_test(LogDet(), x)

    def test_dim(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 DimModel
        class DimModel(torch.jit.ScriptModule):
            # 实现 forward 方法作为脚本方法
            @torch.jit.script_method
            def forward(self, input):
                # 计算输入张量 input 的每个元素乘以 2 的结果
                out = input * 2
                # 将 out 乘以 out 的维度数
                out *= out.dim()
                return out

        # 创建一个空输入的随机张量 empty_input，需要梯度计算
        empty_input = torch.randn(0, requires_grad=True)
        # 创建一个多维输入的随机张量 multi_dim_input，需要梯度计算
        multi_dim_input = torch.randn(1, 2, 3, requires_grad=True)
        # 使用 self.run_test 方法分别运行 DimModel 模型对 empty_input 和 multi_dim_input 的测试
        self.run_test(DimModel(), empty_input)
        self.run_test(DimModel(), multi_dim_input)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dim_1(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 M
        class M(torch.jit.ScriptModule):
            # 实现 forward 方法作为脚本方法
            @torch.jit.script_method
            def forward(self, poses):
                # 创建一个全零张量 boxes，形状为 [poses.shape[0], 2, 4]
                boxes = torch.zeros([poses.shape[0], 2, 4])
                batch_boxes = []
                # 遍历 boxes 的每个 kp_boxes
                for kp_boxes in boxes:
                    # 调用 torchvision.ops.clip_boxes_to_image 函数，将 kp_boxes 裁剪到图像边界 (2, 3)
                    kp_boxes = torchvision.ops.clip_boxes_to_image(kp_boxes, (2, 3))
                    # 将裁剪后的 kp_boxes 添加到 batch_boxes 列表中
                    batch_boxes.append(kp_boxes)
                return batch_boxes

        # 创建一个形状为 (2, 2, 3) 的随机张量 dummy_inputs
        dummy_inputs = torch.rand(2, 2, 3)
        # 使用 self.run_test 方法运行 M 模型的测试，并指定输入名称和动态轴
        self.run_test(M(), (dummy_inputs,), input_names=["x"], dynamic_axes={"x": [0]})

    @skipIfUnsupportedMinOpsetVersion(12)
    @skipDtypeChecking
    def test_outer(self):
        # 定义一个继承自 torch.nn.Module 的类 Outer
        class Outer(torch.nn.Module):
            # 实现 forward 方法，返回输入张量 x 和 y 的外积
            def forward(self, x, y):
                return torch.outer(x, y)

        # 创建不同类型和形状的输入张量 x 和 y，分别运行 Outer 模型的测试
        x = torch.arange(1, 5)
        y = torch.arange(1, 4)
        self.run_test(Outer(), input_args=(x, y))

        x = torch.arange(1, 6).to(dtype=torch.float32)
        y = torch.arange(1, 4).to(dtype=torch.long)
        self.run_test(Outer(), input_args=(x, y))

        x = torch.arange(2, 5).to(dtype=torch.float32)
        y = torch.arange(2, 4).to(dtype=torch.float64)
        self.run_test(Outer(), input_args=(x, y))

        x = torch.arange(3, 6).to(dtype=torch.int32)
        y = torch.arange(4, 7).to(dtype=torch.long)
        self.run_test(Outer(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_movedim(self):
        # 定义一个继承自 torch.nn.Module 的类 MovedimModel
        class MovedimModel(torch.nn.Module):
            # 实现 forward 方法，对输入张量 x 执行多种 movedim 操作
            def forward(self, x):
                return (
                    x.movedim(1, 3),
                    x.movedim(2, 0),
                    x.movedim(1, 1),
                    x.movedim((1, 2, 3), (3, 0, 1)),
                    x.movedim((0, 1, 2), (1, 2, 3)),
                    x.movedim((1, 3, 2), (1, 3, 2)),
                )

        # 创建一个形状为 (5, 3, 4, 2) 的随机张量 x
        x = torch.randn(5, 3, 4, 2)
        # 使用 self.run_test 方法运行 MovedimModel 模型的测试
        self.run_test(MovedimModel(), x)
    def test_moveaxis(self):
        # `test_moveaxis` function tests Torch's `moveaxis` functionality.
        # moveaxis is an alias of movedim; thus, mostly copied from `test_movedim`.
        # Define a nested class `MoveaxisModel` inheriting from `torch.nn.Module`.
        class MoveaxisModel(torch.nn.Module):
            # Define the forward method for the `MoveaxisModel` class.
            def forward(self, x):
                # Return tuple of tensor operations using `moveaxis`.
                return (
                    x.moveaxis(1, 3),          # Move axis 1 to 3
                    x.moveaxis(2, 0),          # Move axis 2 to 0
                    x.moveaxis(1, 1),          # No change as both axes are 1
                    x.moveaxis((1, 2, 3), (3, 0, 1)),  # Reorder axes
                    x.moveaxis((0, 1, 2), (1, 2, 3)),  # Reorder axes
                    x.moveaxis((1, 3, 2), (1, 3, 2)),  # No change as axes match
                )

        # Generate a random tensor `x` of shape (5, 3, 4, 2)
        x = torch.randn(5, 3, 4, 2)

        # Run the `run_test` method with `MoveaxisModel` instance and tensor `x`
        self.run_test(MoveaxisModel(), x)

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_einsum(self):
        # `test_einsum` function tests Torch's `einsum` functionality.
        # Define a nested class `EinsumModelBatchDiagonal` inheriting from `torch.nn.Module`.
        class EinsumModelBatchDiagonal(torch.nn.Module):
            # Define the forward method for the `EinsumModelBatchDiagonal` class.
            def forward(self, x):
                # Define the einsum equation
                eqn = "...ii ->...i"
                # Perform einsum operation with equation `eqn` on tensor `x`
                return torch.einsum(eqn, x)

        # Iterate over tensors `x` with shapes [3, 5, 5] and [3, 5, 5] (boolean type)
        for x in [torch.randn(3, 5, 5), torch.randn(3, 5, 5).to(dtype=torch.bool)]:
            # Run the `run_test` method with `EinsumModelBatchDiagonal` instance and `x`
            self.run_test(EinsumModelBatchDiagonal(), input_args=(x,))

        # Define a nested class `EinsumModelBatchMatmul` inheriting from `torch.nn.Module`.
        class EinsumModelBatchMatmul(torch.nn.Module):
            # Define the forward method for the `EinsumModelBatchMatmul` class.
            def forward(self, x, y):
                # Define the einsum equation
                eqn = "bij, bjk -> bik"
                # Perform einsum operation with equation `eqn` on tensors `x` and `y`
                return torch.einsum(eqn, x, y)

        # Generate random tensors `x` of shape (5, 2, 3) and `y` of shape (5, 3, 4)
        x = torch.randn(5, 2, 3)
        y = torch.randn(5, 3, 4)
        # Run the `run_test` method with `EinsumModelBatchMatmul` instance, `x`, and `y`
        self.run_test(EinsumModelBatchMatmul(), input_args=(x, y))

        # Define a nested class `EinsumModelInnerProd` inheriting from `torch.nn.Module`.
        class EinsumModelInnerProd(torch.nn.Module):
            # Define the forward method for the `EinsumModelInnerProd` class.
            def forward(self, x, y):
                # Define the einsum equation
                eqn = "i,i"
                # Perform einsum operation with equation `eqn` on tensors `x` and `y`
                return torch.einsum(eqn, x, y)

        # Generate random tensors `x` and `y` of shape (5)
        x = torch.randn(5)
        y = torch.randn(5)
        # Run the `run_test` method with `EinsumModelInnerProd` instance, `x`, and `y`
        self.run_test(EinsumModelInnerProd(), input_args=(x, y))

        # Define a nested class `EinsumModelTranspose` inheriting from `torch.nn.Module`.
        class EinsumModelTranspose(torch.nn.Module):
            # Define the forward method for the `EinsumModelTranspose` class.
            def forward(self, x):
                # Define the einsum equation
                eqn = "ij->ji"
                # Perform einsum operation with equation `eqn` on tensor `x`
                return torch.einsum(eqn, x)

        # Iterate over tensors `x` with shapes [3, 4] and [3, 4] (boolean type)
        for x in [torch.randn(3, 4), torch.randn(3, 4).to(dtype=torch.bool)]:
            # Run the `run_test` method with `EinsumModelTranspose` instance and `x`
            self.run_test(EinsumModelTranspose(), input_args=(x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_cosine_similarity(self):
        # `test_cosine_similarity` function tests Torch's `CosineSimilarity` module.
        # Generate random tensors `x` and `y` of shape (5, 3, 2)
        x = torch.randn(5, 3, 2)
        y = torch.randn(5, 3, 2)
        # Run the `run_test` method with `CosineSimilarity` instance, `x`, and `y`
        self.run_test(torch.nn.CosineSimilarity(dim=2), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_pairwise_distance(self):
        # `test_pairwise_distance` function tests Torch's `PairwiseDistance` module.
        # Generate random tensors `x` and `y` of shape (5, 3, 2)
        x = torch.randn(5, 3, 2)
        y = torch.randn(5, 3, 2)
        # Run the `run_test` method with `PairwiseDistance` instance, `x`, and `y`
        self.run_test(torch.nn.PairwiseDistance(p=2.0), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_cross(self):
        # `test_cross` function tests Torch's `cross` function.
        # Define a nested class `Cross` inheriting from `torch.nn.Module`.
        class Cross(torch.nn.Module):
            # Define the forward method for the `Cross` class.
            def forward(self, x, y):
                # Perform cross product operation with tensors `x` and `y`
                return torch.cross(x, y, dim=3), torch.cross(x, y)

        # Generate random tensors `x` and `y` of shape (5, 3, 2, 3)
        x = torch.randn(5, 3, 2, 3)
        y = torch.randn(5, 3, 2, 3)
        # Run the `run_test` method with `Cross` instance and tensors `x` and `y`
        self.run_test(Cross(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_cdist(self):
        # `test_cdist` function tests Torch's `cdist` function.
        # Define a nested class `Cdist` inheriting from `torch.nn.Module`.
        class Cdist(torch.nn.Module):
            # Define the forward method for the `Cdist` class.
            def forward(self, x, y):
                # Compute the pairwise distance between tensors `x` and `y`
                return torch.cdist(x, y)

        # Generate random tensors `x` of shape (5, 3, 3) and `y` of shape (5, 2, 3)
        x = torch.randn(5, 3, 3)
        y = torch.randn(5, 2, 3)
        # Run the `run_test` method with `Cdist` instance and tensors `x` and `y`
        self.run_test(Cdist(), input_args=(x, y))
    # 标记为需要支持最小Opset版本为12，跳过测试如果版本不符合
    @skipIfUnsupportedMinOpsetVersion(12)
    def test_crossentropyloss(self):
        # 遍历两个忽略索引，生成测试数据
        for ignore_index in [-100, 1]:
            # 生成随机输入张量x，形状为(3, 5)，数据类型为torch.float32
            x = torch.randn(3, 5)
            # 生成随机标签张量y，形状为(3,)，数据类型为torch.int64
            y = torch.empty(3, dtype=torch.long).random_(5)
            # 将y中等于1的元素替换为ignore_index
            y[y == 1] = ignore_index
    
            # 调用测试对象的_crossentropyloss方法进行测试
            self._crossentropyloss(x, y, ignore_index)
    
            # 生成随机输入张量x，形状为(3, 5, 2)，数据类型为torch.float32
            x = torch.randn(3, 5, 2)
            # 生成随机标签张量y，形状为(3, 2)，数据类型为torch.int64
            y = torch.empty(3, 2, dtype=torch.long).random_(5)
            # 将y中等于1的元素替换为ignore_index
            y[y == 1] = ignore_index
            # 调用测试对象的_crossentropyloss方法进行测试
            self._crossentropyloss(x, y, ignore_index)
    
            # 生成随机输入张量x，形状为(3, 5, 2, 7)，数据类型为torch.float32
            x = torch.randn(3, 5, 2, 7)
            # 生成随机标签张量y，形状为(3, 2, 7)，数据类型为torch.int64
            y = torch.empty(3, 2, 7, dtype=torch.long).random_(5)
            # 将y中等于1的元素替换为ignore_index
            y[y == 1] = ignore_index
            # 调用测试对象的_crossentropyloss方法进行测试
            self._crossentropyloss(x, y, ignore_index)
    
    # 标记为需要支持最小Opset版本为9，跳过测试如果版本不符合
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_MSELoss(self):
        # 定义一个MSELoss类，继承自torch.nn.Module
        class MSELoss(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 定义三种MSELoss的实例，分别设置不同的reduction模式
                self.loss1 = torch.nn.MSELoss(reduction="none")
                self.loss2 = torch.nn.MSELoss(reduction="sum")
                self.loss3 = torch.nn.MSELoss(reduction="mean")
    
            # 前向传播方法
            def forward(self, input, target):
                # 返回三种MSELoss的计算结果
                return (
                    self.loss1(input, target),
                    self.loss2(input, target),
                    self.loss3(input, target),
                )
    
        # 生成随机输入张量x，形状为(2, 3, 5)，数据类型为torch.float32
        x = torch.randn(2, 3, 5)
        # 生成随机标签张量y，形状为(2, 3, 5)，数据类型为torch.float32
        y = torch.randn(2, 3, 5)
        # 使用self.run_test方法运行MSELoss类的实例
        self.run_test(MSELoss(), input_args=(x, y))
    
    # 标记为需要支持最小Opset版本为9，跳过测试如果版本不符合
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_kldiv_loss(self):
        # 生成随机输入张量x，形状为(5,)，数据类型为torch.float32，取对数
        x = torch.rand(5).log()
        # 生成随机标签张量y，形状为(5,)，数据类型为torch.float32
        y = torch.rand(5)
        # 调用测试对象的_kldiv_loss方法进行测试
        self._kldiv_loss(x, y)
    
        # 生成随机输入张量x，形状为(2, 3, 5)，数据类型为torch.float32，取对数
        x = torch.rand(2, 3, 5).log()
        # 生成随机标签张量y，形状为(2, 3, 5)，数据类型为torch.float32
        y = torch.rand(2, 3, 5)
        # 调用测试对象的_kldiv_loss方法进行测试
        self._kldiv_loss(x, y)
    
        # 生成随机输入张量x，形状为(2, 3, 5, 7)，数据类型为torch.float32，取对数
        x = torch.rand(2, 3, 5, 7).log()
        # 生成随机标签张量y，形状为(2, 3, 5, 7)，数据类型为torch.float32
        y = torch.rand(2, 3, 5, 7)
        # 调用测试对象的_kldiv_loss方法进行测试
        self._kldiv_loss(x, y)
    # 定义私有方法 _kldiv_loss，用于计算 KL 散度损失
    def _kldiv_loss(self, x, y):
        
        # 定义 KLDivLossNone 类，继承自 torch.nn.Module，用于计算 KL 散度损失（无减少）
        class KLDivLossNone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 使用 torch.nn.KLDivLoss 初始化损失函数，设置为无减少（reduction="none"）并且对目标使用对数转换
                self.loss = torch.nn.KLDivLoss(reduction="none", log_target=True)

            def forward(self, input, target):
                # 计算 KL 散度损失
                return self.loss(input, target.log())

        # 运行测试函数，测试 KLDivLossNone 类
        self.run_test(KLDivLossNone(), input_args=(x, y))

        # 定义 KLDivLossMean 类，继承自 torch.nn.Module，用于计算 KL 散度损失（平均）
        class KLDivLossMean(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 使用 torch.nn.KLDivLoss 初始化损失函数，设置为平均（reduction="mean"）并且不对目标使用对数转换
                self.loss = torch.nn.KLDivLoss(reduction="mean", log_target=False)

            def forward(self, input, target):
                # 计算 KL 散度损失
                return self.loss(input, target)

        # 运行测试函数，测试 KLDivLossMean 类
        self.run_test(KLDivLossMean(), input_args=(x, y))

        # 定义 KLDivLossSum 类，继承自 torch.nn.Module，用于计算 KL 散度损失（总和）
        class KLDivLossSum(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 使用 torch.nn.KLDivLoss 初始化损失函数，设置为总和（reduction="sum"）并且对目标使用对数转换
                self.loss = torch.nn.KLDivLoss(reduction="sum", log_target=True)

            def forward(self, input, target):
                # 计算 KL 散度损失
                return self.loss(input, target.log())

        # 运行测试函数，测试 KLDivLossSum 类
        self.run_test(KLDivLossSum(), input_args=(x, y))

        # 定义 KLDivLossBatchMean 类，继承自 torch.nn.Module，用于计算 KL 散度损失（批次均值）
        class KLDivLossBatchMean(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 使用 torch.nn.KLDivLoss 初始化损失函数，设置为批次均值（reduction="batchmean"）并且不对目标使用对数转换
                self.loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)

            def forward(self, input, target):
                # 计算 KL 散度损失
                return self.loss(input, target)

        # 运行测试函数，测试 KLDivLossBatchMean 类
        self.run_test(KLDivLossBatchMean(), input_args=(x, y))

        # 定义 KLDivLossMiniBatchMean 类，继承自 torch.nn.Module，用于计算 KL 散度损失（迷你批次均值）
        class KLDivLossMiniBatchMean(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 使用 torch.nn.KLDivLoss 初始化损失函数，设置为迷你批次均值（reduction="batchmean"，size_average=False）并且对目标使用对数转换
                self.loss = torch.nn.KLDivLoss(
                    reduction="batchmean", size_average=False, log_target=True
                )

            def forward(self, input, target):
                # 计算 KL 散度损失
                return self.loss(input, target.log())

        # 运行测试函数，测试 KLDivLossMiniBatchMean 类
        self.run_test(KLDivLossMiniBatchMean(), input_args=(x, y))
    def test_nllloss_2d_none(self):
        # 定义一个继承自 torch.nn.Module 的内部模型类 NLLModel
        class NLLModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 使用 torch.nn.NLLLoss(reduction="none") 定义不带降维的负对数似然损失函数
                self.loss = torch.nn.NLLLoss(reduction="none")
                # 定义一个二维卷积层，输入通道数为 16，输出通道数为 C，卷积核大小为 (3, 3)
                self.conv = torch.nn.Conv2d(16, C, (3, 3))
                # 定义对数softmax函数，沿第一个维度进行计算
                self.m = torch.nn.LogSoftmax(dim=1)

            # 前向传播函数，接受输入 input 和目标 target
            def forward(self, input, target):
                # 对输入进行卷积操作，然后对卷积结果进行对数softmax操作，再计算损失
                output = self.loss(self.m(self.conv(input)), target)
                return output

        # 定义常量 N 和 C 分别为 5 和 4
        N, C = 5, 4
        # 生成随机输入数据 input，大小为 (N, 16, 10, 10)
        input = torch.randn(N, 16, 10, 10)
        # 生成随机目标数据 target，大小为 (N, 8, 8)，数据类型为 long 类型，取值范围在 [0, C) 之间
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)

        # 使用默认的 ignore_index=-100 处理测试数据 target
        target[target == 1] = -100
        # 运行测试函数 self.run_test，测试 NLLModel 类
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_mean(self):
        # 定义一个继承自 torch.nn.Module 的内部模型类 NLLModel
        class NLLModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 使用 torch.nn.NLLLoss(reduction="mean") 定义带平均降维的负对数似然损失函数
                self.loss = torch.nn.NLLLoss(reduction="mean")
                # 定义一个二维卷积层，输入通道数为 16，输出通道数为 C，卷积核大小为 (3, 3)
                self.conv = torch.nn.Conv2d(16, C, (3, 3))
                # 定义对数softmax函数，沿第一个维度进行计算
                self.m = torch.nn.LogSoftmax(dim=1)

            # 前向传播函数，接受输入 input 和目标 target
            def forward(self, input, target):
                # 对输入进行卷积操作，然后对卷积结果进行对数softmax操作，再计算损失
                output = self.loss(self.m(self.conv(input)), target)
                return output

        # 定义常量 N 和 C 分别为 5 和 4
        N, C = 5, 4
        # 生成随机输入数据 input，大小为 (N, 16, 10, 10)
        input = torch.randn(N, 16, 10, 10)
        # 生成随机目标数据 target，大小为 (N, 8, 8)，数据类型为 long 类型，取值范围在 [0, C) 之间
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)

        # 使用默认的 ignore_index=-100 处理测试数据 target
        target[target == 1] = -100
        # 运行测试函数 self.run_test，测试 NLLModel 类
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_sum(self):
        # 定义一个继承自 torch.nn.Module 的内部模型类 NLLModel
        class NLLModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 使用 torch.nn.NLLLoss(reduction="sum") 定义带求和降维的负对数似然损失函数
                self.loss = torch.nn.NLLLoss(reduction="sum")
                # 定义一个二维卷积层，输入通道数为 16，输出通道数为 C，卷积核大小为 (3, 3)
                self.conv = torch.nn.Conv2d(16, C, (3, 3))
                # 定义对数softmax函数，沿第一个维度进行计算
                self.m = torch.nn.LogSoftmax(dim=1)

            # 前向传播函数，接受输入 input 和目标 target
            def forward(self, input, target):
                # 对输入进行卷积操作，然后对卷积结果进行对数softmax操作，再计算损失
                output = self.loss(self.m(self.conv(input)), target)
                return output

        # 定义常量 N 和 C 分别为 5 和 4
        N, C = 5, 4
        # 生成随机输入数据 input，大小为 (N, 16, 10, 10)
        input = torch.randn(N, 16, 10, 10)
        # 生成随机目标数据 target，大小为 (N, 8, 8)，数据类型为 long 类型，取值范围在 [0, C) 之间
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)

        # 使用默认的 ignore_index=-100 处理测试数据 target
        target[target == 1] = -100
        # 运行测试函数 self.run_test，测试 NLLModel 类
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_mean_weights(self):
        # 定义一个继承自torch.nn.Module的类NLLModel，用于测试NLLLoss在二维情况下的加权平均损失
        class NLLModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化NLLLoss损失函数，设置为均值模式，权重为随机生成的长度为C的张量
                self.loss = torch.nn.NLLLoss(reduction="mean", weight=torch.randn(C))
                # 初始化一个二维卷积层，输入通道为16，输出通道数为C，卷积核大小为(3, 3)
                self.conv = torch.nn.Conv2d(16, C, (3, 3))
                # 初始化对数softmax层，指定对第一个维度进行softmax操作
                self.m = torch.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                # 计算网络前向传播后的损失，其中包括卷积操作、对数softmax操作以及NLLLoss损失计算
                output = self.loss(self.m(self.conv(input)), target)
                return output

        N, C = 5, 4
        # 生成随机输入数据，形状为(N, 16, 10, 10)
        input = torch.randn(N, 16, 10, 10)
        # 生成随机目标数据，形状为(N, 8, 8)，数据类型为长整型，范围在0到C之间
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)

        # 使用默认的ignore_index=-100对目标数据进行处理，将值为1的位置设置为-100
        target[target == 1] = -100
        # 运行测试函数，测试定义的NLLModel类
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_mean_ignore_index(self):
        # 定义一个继承自torch.nn.Module的类NLLModel，用于测试NLLLoss在二维情况下的忽略指定索引的平均损失
        class NLLModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化NLLLoss损失函数，设置为均值模式，忽略指定索引值为1
                self.loss = torch.nn.NLLLoss(reduction="mean", ignore_index=1)
                # 初始化一个二维卷积层，输入通道为16，输出通道数为C，卷积核大小为(3, 3)
                self.conv = torch.nn.Conv2d(16, C, (3, 3))
                # 初始化对数softmax层，指定对第一个维度进行softmax操作
                self.m = torch.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                # 计算网络前向传播后的损失，其中包括卷积操作、对数softmax操作以及NLLLoss损失计算
                output = self.loss(self.m(self.conv(input)), target)
                return output

        N, C = 5, 4
        # 生成随机输入数据，形状为(N, 16, 10, 10)
        input = torch.randn(N, 16, 10, 10)
        # 生成随机目标数据，形状为(N, 8, 8)，数据类型为长整型，范围在0到C之间
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
        # 运行测试函数，测试定义的NLLModel类
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_dynamic_ignore_index(self):
        import torch.nn.functional as F

        # 定义一个线性组合函数，根据给定的epsilon进行线性组合
        def linear_combination(x, y, epsilon):
            return epsilon * x + (1 - epsilon) * y

        # 定义一个函数，根据reduction参数减少损失值
        def reduce_loss(loss, reduction="mean"):
            return (
                loss.mean()
                if reduction == "mean"
                else loss.sum()
                if reduction == "sum"
                else loss
            )

        # 定义一个继承自torch.nn.Module的类LabelSmoothingCrossEntropy，实现标签平滑交叉熵损失函数
        class LabelSmoothingCrossEntropy(torch.nn.Module):
            def __init__(self, epsilon: float = 0.1, reduction="mean"):
                super().__init__()
                self.epsilon = epsilon
                self.reduction = reduction

            def forward(self, preds, target, start_position):
                # 计算预测值的对数softmax
                log_preds = F.log_softmax(preds, dim=-1)
                # 获取开始位置的大小
                ignore_index = start_position.size(1)
                # 使用NLLLoss函数计算损失，根据指定的reduction和ignore_index参数
                nll = F.nll_loss(
                    log_preds,
                    target,
                    reduction=self.reduction,
                    ignore_index=ignore_index,
                )
                # 返回损失加上开始位置的浮点数
                return nll + start_position.float()

        N = 5
        # 生成随机预测数据，形状为(N, 16)
        preds = torch.randn(N, 16)
        # 生成随机目标数据，范围在0到5之间
        target = torch.randint(5, (N,))
        # 生成随机开始位置数据，范围在0到10之间
        start_position = torch.randint(10, (N, N))
        # 运行测试函数，测试定义的LabelSmoothingCrossEntropy类
        self.run_test(LabelSmoothingCrossEntropy(), (preds, target, start_position))

    @skipIfUnsupportedMinOpsetVersion(12)
    # 定义一个测试方法，用于测试 NLLLoss 的二维均值忽略索引权重
    def test_nllloss_2d_mean_ignore_index_weights(self):
        # 定义一个继承自 torch.nn.Module 的模型类 NLLModel
        class NLLModel(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 使用 NLLLoss 来定义损失函数，设置为均值，随机初始化权重并忽略索引为 1 的类别
                self.loss = torch.nn.NLLLoss(
                    reduction="mean", weight=torch.randn(C), ignore_index=1
                )
                # 定义一个二维卷积层，输入通道数为 16，输出通道数为 C，卷积核大小为 (3, 3)
                self.conv = torch.nn.Conv2d(16, C, (3, 3))
                # 定义一个对第一维进行 log softmax 操作的层
                self.m = torch.nn.LogSoftmax(dim=1)

            # 前向传播方法
            def forward(self, input, target):
                # 将输入经过卷积和 log softmax 操作后，计算损失值
                output = self.loss(self.m(self.conv(input)), target)
                return output

        # 设置常量 N 和 C 的值分别为 5 和 4
        N, C = 5, 4
        # 生成一个形状为 (N, 16, 10, 10) 的随机张量作为输入
        input = torch.randn(N, 16, 10, 10)
        # 生成一个形状为 (N, 8, 8) 的长整型张量，其元素在 [0, C) 范围内随机分布，作为目标张量
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
        # 运行测试，传入 NLLModel 的实例和输入元组 (input, target)
        self.run_test(NLLModel(), (input, target))

    # 如果不支持最小的运算版本为 12，则跳过此测试
    @skipIfUnsupportedMinOpsetVersion(12)
    def test_binary_cross_entropy_with_logits(self):
        # 生成一个形状为 (5,) 的随机张量 x
        x = torch.randn(5)
        # 生成一个形状为 (5,) 的长整型张量 y，其元素在 [0, 2) 范围内随机分布
        y = torch.empty(5).random_(2)
        # 调用 _bce_logits 方法，传入 x 和 y 作为输入参数
        self._bce_logits(x, y)

        # 生成一个形状为 (3, 4) 的随机张量 x
        x = torch.randn(3, 4)
        # 生成一个形状为 (3, 4) 的长整型张量 y，其元素在 [0, 2) 范围内随机分布
        y = torch.empty(3, 4).random_(2)
        # 生成一个张量 weight，其元素为 [3]
        weight = torch.tensor([3])
        # 调用 _bce_logits_wegiht 方法，传入 x、y 和 weight 作为输入参数
        self._bce_logits_wegiht(x, y, weight)

        # 生成一个形状为 (3, 2, 4) 的随机张量 x
        x = torch.randn(3, 2, 4)
        # 生成一个形状为 (3, 2, 4) 的长整型张量 y，其元素在 [0, 2) 范围内随机分布
        y = torch.empty(3, 2, 4).random_(2)
        # 生成一个形状为 [2, 4] 的随机张量 pos_weight
        pos_weight = torch.empty([2, 4]).random_(2)
        # 调用 _bce_logits_posweight 方法，传入 x、y 和 pos_weight 作为输入参数
        self._bce_logits_posweight(x, y, pos_weight)

        # 生成一个形状为 (3, 3, 4) 的随机张量 x
        x = torch.randn(3, 3, 4)
        # 生成一个形状为 (3, 3, 4) 的长整型张量 y，其元素在 [0, 2) 范围内随机分布
        y = torch.empty(3, 3, 4).random_(2)
        # 生成一个张量 weight，其元素为 [3]
        weight = torch.tensor([3])
        # 生成一个形状为 [3, 4] 的随机张量 pos_weight
        pos_weight = torch.empty([3, 4]).random_(2)
        # 调用 _bce_logits_loss_weight_posweight 方法，传入 x、y、weight 和 pos_weight 作为输入参数
        self._bce_logits_loss_weight_posweight(x, y, weight, pos_weight)

    # 定义一个方法 _bce_logits，用于计算二进制交叉熵损失
    def _bce_logits(self, x, y):
        # 定义一个继承自 torch.nn.Module 的模型类 BCEWithLogitsLossNone
        class BCEWithLogitsLossNone(torch.nn.Module):
            # 前向传播方法
            def forward(self, input, target):
                # 使用 torch.nn.functional.binary_cross_entropy_with_logits 计算二进制交叉熵损失，reduction 设为 "none"
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    input, target, reduction="none"
                )

        # 运行测试，传入 BCEWithLogitsLossNone 的实例和输入参数元组 (x, y)
        self.run_test(BCEWithLogitsLossNone(), input_args=(x, y))

        # 定义一个继承自 torch.nn.Module 的模型类 BCEWithLogitsLossMean
        class BCEWithLogitsLossMean(torch.nn.Module):
            # 前向传播方法
            def forward(self, input, target):
                # 使用 torch.nn.functional.binary_cross_entropy_with_logits 计算二进制交叉熵损失，reduction 设为 "mean"
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    input, target, reduction="mean"
                )

        # 运行测试，传入 BCEWithLogitsLossMean 的实例和输入参数元组 (x, y)
        self.run_test(BCEWithLogitsLossMean(), input_args=(x, y))

        # 定义一个继承自 torch.nn.Module 的模型类 BCEWithLogitsLossSum
        class BCEWithLogitsLossSum(torch.nn.Module):
            # 前向传播方法
            def forward(self, input, target):
                # 使用 torch.nn.functional.binary_cross_entropy_with_logits 计算二进制交叉熵损失，reduction 设为 "sum"
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    input, target, reduction="sum"
                )

        # 运行测试，传入 BCEWithLogitsLossSum 的实例和输入参数元组 (x, y)
        self.run_test(BCEWithLogitsLossSum(), input_args=(x, y))
    # 定义一个私有方法 `_bce_logits_wegiht`，用于测试不同的二元交叉熵损失函数（带权重）
    def _bce_logits_wegiht(self, x, y, weight):
        # 定义一个内部类 BCEWithLogitsLossWegihtNone，继承自 torch.nn.Module
        class BCEWithLogitsLossWegihtNone(torch.nn.Module):
            # 定义 forward 方法，接收输入 input, target, weight，并返回带权重的二元交叉熵损失
            def forward(self, input, target, weight):
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    input, target, weight=weight, reduction="none"
                )
        
        # 运行测试，使用 BCEWithLogitsLossWegihtNone 类作为模型，传入 x, y, weight 作为输入参数
        self.run_test(BCEWithLogitsLossWegihtNone(), input_args=(x, y, weight))

        # 定义一个内部类 BCEWithLogitsLossWegihtMean，继承自 torch.nn.Module
        class BCEWithLogitsLossWegihtMean(torch.nn.Module):
            # 定义 forward 方法，接收输入 input, target, weight，并返回带权重的二元交叉熵损失
            def forward(self, input, target, weight):
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    input, target, weight=weight, reduction="mean"
                )
        
        # 运行测试，使用 BCEWithLogitsLossWegihtMean 类作为模型，传入 x, y, weight 作为输入参数
        self.run_test(BCEWithLogitsLossWegihtMean(), input_args=(x, y, weight))

        # 定义一个内部类 BCEWithLogitsLossWegihtSum，继承自 torch.nn.Module
        class BCEWithLogitsLossWegihtSum(torch.nn.Module):
            # 定义 forward 方法，接收输入 input, target, weight，并返回带权重的二元交叉熵损失
            def forward(self, input, target, weight):
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    input, target, weight=weight, reduction="sum"
                )
        
        # 运行测试，使用 BCEWithLogitsLossWegihtSum 类作为模型，传入 x, y, weight 作为输入参数
        self.run_test(BCEWithLogitsLossWegihtSum(), input_args=(x, y, weight))

    # 定义一个私有方法 `_bce_logits_posweight`，用于测试不同的二元交叉熵损失函数（带正样本权重）
    def _bce_logits_posweight(self, x, y, pos_weight):
        # 定义一个内部类 BCEWithLogitsLossPosWegihtNone，继承自 torch.nn.Module
        class BCEWithLogitsLossPosWegihtNone(torch.nn.Module):
            # 定义 forward 方法，接收输入 input, target, pos_weight，并返回带正样本权重的二元交叉熵损失
            def forward(self, input, target, pos_weight):
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    input, target, pos_weight=pos_weight, reduction="none"
                )
        
        # 运行测试，使用 BCEWithLogitsLossPosWegihtNone 类作为模型，传入 x, y, pos_weight 作为输入参数
        self.run_test(BCEWithLogitsLossPosWegihtNone(), input_args=(x, y, pos_weight))

        # 定义一个内部类 BCEWithLogitsLossPosWegihtMean，继承自 torch.nn.Module
        class BCEWithLogitsLossPosWegihtMean(torch.nn.Module):
            # 定义 forward 方法，接收输入 input, target, pos_weight，并返回带正样本权重的二元交叉熵损失
            def forward(self, input, target, pos_weight):
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    input, target, pos_weight=pos_weight, reduction="mean"
                )
        
        # 运行测试，使用 BCEWithLogitsLossPosWegihtMean 类作为模型，传入 x, y, pos_weight 作为输入参数
        self.run_test(BCEWithLogitsLossPosWegihtMean(), input_args=(x, y, pos_weight))

        # 定义一个内部类 BCEWithLogitsLossPosWegihtSum，继承自 torch.nn.Module
        class BCEWithLogitsLossPosWegihtSum(torch.nn.Module):
            # 定义 forward 方法，接收输入 input, target, pos_weight，并返回带正样本权重的二元交叉熵损失
            def forward(self, input, target, pos_weight):
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    input, target, pos_weight=pos_weight, reduction="sum"
                )
        
        # 运行测试，使用 BCEWithLogitsLossPosWegihtSum 类作为模型，传入 x, y, pos_weight 作为输入参数
        self.run_test(BCEWithLogitsLossPosWegihtSum(), input_args=(x, y, pos_weight))
    # 定义一个名为 _bce_logits_loss_weight_posweight 的方法，接受四个参数 x, y, weight, pos_weight
    def _bce_logits_loss_weight_posweight(self, x, y, weight, pos_weight):
        # 定义内部类 BCEWithLogitsLossWeightPosweightNone，继承自 torch.nn.Module
        class BCEWithLogitsLossWeightPosweightNone(torch.nn.Module):
            # 定义 forward 方法，接受 input, target, weight, pos_weight 参数
            def forward(self, input, target, weight, pos_weight):
                # 调用 PyTorch 提供的 binary_cross_entropy_with_logits 函数，使用 "none" reduction
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    input,
                    target,
                    weight=weight,
                    pos_weight=pos_weight,
                    reduction="none",
                )

        # 运行测试，使用 BCEWithLogitsLossWeightPosweightNone 类进行测试，传入 input_args 参数
        self.run_test(
            BCEWithLogitsLossWeightPosweightNone(),
            input_args=(x, y, weight, pos_weight),
        )

        # 定义内部类 BCEWithLogitsLossWeightPosweightMean，继承自 torch.nn.Module
        class BCEWithLogitsLossWeightPosweightMean(torch.nn.Module):
            # 定义 forward 方法，接受 input, target, weight, pos_weight 参数
            def forward(self, input, target, weight, pos_weight):
                # 调用 PyTorch 提供的 binary_cross_entropy_with_logits 函数，使用 "mean" reduction
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    input,
                    target,
                    weight=weight,
                    pos_weight=pos_weight,
                    reduction="mean",
                )

        # 运行测试，使用 BCEWithLogitsLossWeightPosweightMean 类进行测试，传入 input_args 参数
        self.run_test(
            BCEWithLogitsLossWeightPosweightMean(),
            input_args=(x, y, weight, pos_weight),
        )

        # 定义内部类 BCEWithLogitsLossWeightPosweightSum，继承自 torch.nn.Module
        class BCEWithLogitsLossWeightPosweightSum(torch.nn.Module):
            # 定义 forward 方法，接受 input, target, weight, pos_weight 参数
            def forward(self, input, target, weight, pos_weight):
                # 调用 PyTorch 提供的 binary_cross_entropy_with_logits 函数，使用 "sum" reduction
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    input, target, weight=weight, pos_weight=pos_weight, reduction="sum"
                )

        # 运行测试，使用 BCEWithLogitsLossWeightPosweightSum 类进行测试，传入 input_args 参数
        self.run_test(
            BCEWithLogitsLossWeightPosweightSum(), input_args=(x, y, weight, pos_weight)
        )

    # 定义测试方法 test_torch_mm
    def test_torch_mm(self):
        # 定义内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 定义 forward 方法，接受 mat1, mat2 参数
            def forward(self, mat1, mat2):
                # 执行 torch.mm 运算，计算 mat1 和 mat2 的矩阵乘积
                mm = torch.mm(mat1, mat2)
                return mm

        # 创建两个随机的张量 mat1 和 mat2
        mat1 = torch.randn(2, 3)
        mat2 = torch.randn(3, 3)
        # 运行测试，使用 M 类进行测试，传入 input_args 参数
        self.run_test(M(), input_args=(mat1, mat2))

    # 根据条件跳过不支持 opset < 9 的测试
    @skipIfUnsupportedMinOpsetVersion(
        9
    )  # 因为 where 操作在 opset < 9 不支持。
    # 定义测试方法 test_where_with_bool_tensor
    def test_where_with_bool_tensor(self):
        # 定义内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 定义 forward 方法，接受 mat1, mat2 参数
            def forward(self, mat1, mat2):
                # 使用 torch.where 根据条件 mat1 > 0 在 mat1 和 mat2 之间选择
                out = torch.where(mat1 > 0, mat1, mat2)
                return out

        # 创建一个随机张量 mat1 和一个全为 1 的张量 mat2
        mat1 = torch.randn(2, 3)
        mat2 = torch.ones(2, 3)
        # 运行测试，使用 M 类进行测试，传入 input_args 参数
        self.run_test(M(), input_args=(mat1, mat2))

    # 根据条件跳过不支持 opset < 9 的测试
    @skipIfUnsupportedMinOpsetVersion(
        9
    )  # 因为 where 操作在 opset < 9 不支持。
    # 定义测试方法 test_where_with_byte_tensor
    def test_where_with_byte_tensor(self):
        # 定义内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 定义 forward 方法，接受 cond, mat1, mat2 参数
            def forward(self, cond, mat1, mat2):
                # 使用 torch.where 根据条件 cond 在 mat1 和 mat2 之间选择
                out = torch.where(cond, mat1, mat2)
                return out

        # 创建一个全为 1 的 uint8 类型张量 cond，修改其中一个元素为 0，同时创建一个随机张量 mat1 和一个全为 1 的张量 mat2
        cond = torch.ones(2, 3, dtype=torch.uint8)
        cond[1, 2] = 0
        mat1 = torch.randn(2, 3)
        mat2 = torch.ones(2, 3)
        # 运行测试，使用 M 类进行测试，传入 input_args 参数
        self.run_test(M(), input_args=(cond, mat1, mat2))

    # 根据条件跳过不支持 opset < 10 的测试
    @skipIfUnsupportedMinOpsetVersion(10)  # 因为 ONNX IsInf 操作在 opset >= 10 才被支持。
    def test_isinf(self):
        # 定义一个继承自 torch.nn.Module 的模型类 M，重写 forward 方法
        class M(torch.nn.Module):
            def forward(self, x):
                return x.isinf()

        # 创建一个包含 inf 和 nan 的张量 x
        x = torch.tensor([[1, 2, float("inf")], [2, float("nan"), float("inf")]])
        # 运行测试函数 run_test，传入模型 M 的实例和张量 x
        self.run_test(M(), (x,))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_isfinite(self):
        # 定义一个继承自 torch.nn.Module 的模型类 M，重写 forward 方法
        class M(torch.nn.Module):
            def forward(self, x):
                return x.isfinite()

        # 创建一个包含 inf 和 nan 的张量 x
        x = torch.tensor([[1, 2, float("inf")], [2, float("nan"), -float("inf")]])
        # 运行测试函数 run_test，传入模型 M 的实例和张量 x
        self.run_test(M(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)  # ONNX IsNaN op is added in opset 9.
    def test_isnan(self):
        # 定义一个继承自 torch.nn.Module 的模型类 M，重写 forward 方法
        class M(torch.nn.Module):
            def forward(self, x):
                return x.isnan()

        # 创建一个包含 inf 和 nan 的张量 x
        x = torch.tensor([[1, 2, float("inf")], [2, float("nan"), float("inf")]])
        # 运行测试函数 run_test，传入模型 M 的实例和张量 x
        self.run_test(M(), (x,))

    @skipIfUnsupportedMinOpsetVersion(
        10
    )  # ONNX IsNaN, IsInf op is added in opset 9, 10 respectively.
    def test_nan_to_num(self):
        # 定义一个无参数的继承自 torch.nn.Module 的模型类 NoParams，重写 forward 方法
        class NoParams(torch.nn.Module):
            def forward(self, x):
                return x.nan_to_num()

        # 创建一个包含 inf 和 nan 的张量 x
        x = torch.tensor([[1, 2, float("inf")], [2, float("nan"), -float("inf")]])
        # 创建不同数据类型的张量：整数类型 xint 和半精度类型 xhalf
        xint = torch.ones((2, 4), dtype=torch.int)
        xhalf = torch.ones((2, 4), dtype=torch.half)
        # 运行测试函数 run_test，分别传入模型 NoParams 的实例和不同类型的张量
        self.run_test(NoParams(), (x,))
        self.run_test(NoParams(), (xint,))
        self.run_test(NoParams(), (xhalf,))

        # 定义一个带参数的继承自 torch.nn.Module 的模型类 WithParams，重写 forward 方法
        class WithParams(torch.nn.Module):
            def forward(self, x):
                return x.nan_to_num(nan=2.3, posinf=4.5, neginf=6.7)

        # 创建一个包含 inf 和 nan 的张量 x
        x = torch.tensor([[1, 2, float("inf")], [2, float("nan"), -float("inf")]])
        # 运行测试函数 run_test，传入模型 WithParams 的实例和张量 x
        self.run_test(WithParams(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_maximum_minimum(self):
        # 定义一个带 nan 的继承自 torch.nn.Module 的模型类 ModelWithNan，重写 forward 方法
        class ModelWithNan(torch.nn.Module):
            def forward(self, x, y):
                return torch.maximum(x, y), torch.minimum(x, y)

        # 创建一个包含 nan 的张量 x 和一个随机张量 y
        x = torch.tensor([-2, -2, float("nan")])
        y = torch.rand(1, 3)
        # 运行测试函数 run_test，传入模型 ModelWithNan 的实例和张量 x、y
        self.run_test(ModelWithNan(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_minimum_dtypes(self):
        # 定义一个带两个参数的继承自 torch.nn.Module 的模型类 MinimumModel，重写 forward 方法
        class MinimumModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.minimum(x, y)

        # 创建不同数据类型的张量 x 和 y，运行测试函数 run_test，传入模型 MinimumModel 的实例和张量
        x = torch.randn((5, 5), dtype=torch.float16)
        y = torch.randn((5, 5), dtype=torch.float)
        self.run_test(MinimumModel(), (x, y))

        x = torch.randn((5, 5), dtype=torch.float16)
        y = torch.randint(10, (5, 5), dtype=torch.int16)
        self.run_test(MinimumModel(), (x, y))

        x = torch.randint(10, (5, 5), dtype=torch.int16)
        y = torch.randint(10, (5, 5), dtype=torch.int32)
        self.run_test(MinimumModel(), (x, y))

        x = torch.randint(10, (5, 5), dtype=torch.int)
        y = torch.full_like(x, True)
        self.run_test(MinimumModel(), (x, y))
    # 定义一个测试方法，用于测试支持不同数据类型的最大值计算
    def test_maximum_dtypes(self):
        # 定义一个简单的神经网络模型，实现最大值计算的前向传播
        class MaximumModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.maximum(x, y)

        # 创建一个 float16 类型的随机张量 x 和 float 类型的随机张量 y
        x = torch.randn((5, 5), dtype=torch.float16)
        y = torch.randn((5, 5), dtype=torch.float)
        # 运行测试函数，测试最大值模型的前向传播
        self.run_test(MaximumModel(), (x, y))

        # 创建一个 float16 类型的随机张量 x 和 int16 类型的随机张量 y
        x = torch.randn((5, 5), dtype=torch.float16)
        y = torch.randint(10, (5, 5), dtype=torch.int16)
        # 运行测试函数，测试最大值模型的前向传播
        self.run_test(MaximumModel(), (x, y))

        # 创建一个 int16 类型的随机张量 x 和 int32 类型的随机张量 y
        x = torch.randint(10, (5, 5), dtype=torch.int16)
        y = torch.randint(10, (5, 5), dtype=torch.int32)
        # 运行测试函数，测试最大值模型的前向传播
        self.run_test(MaximumModel(), (x, y))

        # 创建一个 int 类型的随机张量 x 和与 x 同维度的全为 True 的张量 y
        x = torch.randint(10, (5, 5), dtype=torch.int)
        y = torch.full_like(x, True)
        # 运行测试函数，测试最大值模型的前向传播
        self.run_test(MaximumModel(), (x, y))

    # 根据支持的最小操作集版本，跳过不支持的测试
    @skipIfUnsupportedMinOpsetVersion(9)
    # 定义一个测试方法，测试张量的任一元素是否为 True
    def test_any(self):
        # 定义一个简单的神经网络模型，实现任一元素是否为 True 的判断
        class M(torch.nn.Module):
            def forward(self, x):
                return x.any()

        # 创建一个布尔类型张量 x，测试其任一元素是否为 True
        x = torch.tensor([[True, False], [False, False]])
        # 运行测试函数，测试 any 模型的前向传播
        self.run_test(M(), (x,))

        # 定义一个带维度参数的神经网络模型，实现指定维度上任一元素是否为 True 的判断
        class MDim(torch.nn.Module):
            def forward(self, x):
                return x.any(dim=1)

        # 创建一个随机布尔类型张量 x，测试指定维度上任一元素是否为 True
        x = torch.rand(3, 4).bool()
        # 运行测试函数，测试 MDim 模型的前向传播
        self.run_test(MDim(), (x,))

        # 定义一个带保持维度参数的神经网络模型，实现指定维度上任一元素是否为 True 的判断并保持维度
        class MKeepdim(torch.nn.Module):
            def forward(self, x):
                return x.any(dim=1, keepdim=True)

        # 创建一个随机布尔类型张量 x，测试指定维度上任一元素是否为 True 并保持维度
        x = torch.rand(3, 4).bool()
        # 运行测试函数，测试 MKeepdim 模型的前向传播
        self.run_test(MKeepdim(), (x,))

    # 根据支持的最小操作集版本，跳过不支持的测试
    @skipIfUnsupportedMinOpsetVersion(9)
    # 定义一个测试方法，测试张量的所有元素是否为 True
    def test_all(self):
        # 定义一个简单的神经网络模型，实现所有元素是否为 True 的判断
        class M(torch.nn.Module):
            def forward(self, x):
                return x.all()

        # 创建一个布尔类型张量 x，测试其所有元素是否为 True
        x = torch.tensor([[True, False], [False, False]])
        # 运行测试函数，测试 all 模型的前向传播
        self.run_test(M(), (x,))

        # 定义一个带维度参数的神经网络模型，实现指定维度上所有元素是否为 True 的判断
        class MDim(torch.nn.Module):
            def forward(self, x):
                return x.all(dim=1)

        # 创建一个随机布尔类型张量 x，测试指定维度上所有元素是否为 True
        x = torch.rand(3, 4).bool()
        # 运行测试函数，测试 MDim 模型的前向传播
        self.run_test(MDim(), (x,))

        # 定义一个带保持维度参数的神经网络模型，实现指定维度上所有元素是否为 True 的判断并保持维度
        class MKeepdim(torch.nn.Module):
            def forward(self, x):
                return x.all(dim=1, keepdim=True)

        # 创建一个随机布尔类型张量 x，测试指定维度上所有元素是否为 True 并保持维度
        x = torch.rand(3, 4).bool()
        # 运行测试函数，测试 MKeepdim 模型的前向传播
        self.run_test(MKeepdim(), (x,))

    # 定义一个测试方法，测试 Dropout 层的功能
    def test_dropout(self):
        # 定义一个包含 Dropout 层的神经网络模型
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = torch.nn.Dropout(0.3)

            def forward(self, x):
                dropout = self.dropout(x)
                return dropout

        # 创建一个随机张量 x，测试 Dropout 模型的前向传播
        x = torch.randn(10, 3, 53)
        self.run_test(M(), (x))

    # 定义一个测试方法，测试 RReLU 激活函数在评估模式下的行为
    def test_rrelu_eval(self):
        # 创建一个包含两个元素的张量 x，测试 RReLU 激活函数在评估模式下的输出
        x = torch.tensor([0.5, -0.5])
        self.run_test(torch.nn.RReLU(0.1, 0.3).eval(), x)

    # 定义一个测试方法，测试在计算图中对常数形状的折叠
    def test_shape_constant_fold(self):
        # 定义一个包含固定形状张量的神经网络模型
        class ShapeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 注册一个固定值为全 1 的张量作为模型的权重
                self.register_buffer("weight", torch.ones(5))

            def forward(self, x):
                # 获取权重张量的长度作为形状常量
                shape = self.weight.shape[0]
                # 返回输入张量 x 加上形状常量的结果
                return x + shape

        # 创建一个随机张量 x，测试 ShapeModule 模型的前向传播
        x = torch.randn(2, 5)
        # 运行测试函数，测试 ShapeModule 模型的前向传播
        self.run_test(ShapeModule(), (x,), rtol=1e-3, atol=1e-5)
    @skipIfUnsupportedMinOpsetVersion(12)
    def test_celu(self):
        # 定义 CELU 类，继承自 torch.nn.Module
        class Celu(torch.nn.Module):
            # 初始化函数，设置 alpha 值为 1.0 的 CELU 激活函数
            def __init__(self):
                super().__init__()
                self.celu = torch.nn.CELU(alpha=1.0)
    
            # 前向传播函数，应用 CELU 激活函数到输入上
            def forward(self, input):
                return self.celu(input)
    
        # 生成一个大小为 2 的随机张量作为输入
        input = torch.randn(2)
        # 运行测试函数，验证 CELU 模块的正确性
        self.run_test(Celu(), (input,))
    
    @skipIfUnsupportedMinOpsetVersion(12)
    def test_celu_default(self):
        # 定义 CELU 类，继承自 torch.nn.Module
        class Celu(torch.nn.Module):
            # 初始化函数，使用默认参数创建 CELU 激活函数
            def __init__(self):
                super().__init__()
                self.celu = torch.nn.CELU()
    
            # 前向传播函数，应用 CELU 激活函数到输入上
            def forward(self, input):
                return self.celu(input)
    
        # 生成一个大小为 2 的随机张量作为输入
        input = torch.randn(2)
        # 运行测试函数，验证 CELU 模块的正确性
        self.run_test(Celu(), (input,))
    
    @skipIfUnsupportedMinOpsetVersion(12)
    def test_celu_alpha(self):
        # 定义 CELU 类，继承自 torch.nn.Module
        class Celu(torch.nn.Module):
            # 初始化函数，设置 alpha 值为 2.0 的 CELU 激活函数
            def __init__(self):
                super().__init__()
                self.celu = torch.nn.CELU(alpha=2.0)
    
            # 前向传播函数，应用 CELU 激活函数到输入上
            def forward(self, input):
                return self.celu(input)
    
        # 生成一个大小为 2 的随机张量作为输入
        input = torch.randn(2)
        # 运行测试函数，验证 CELU 模块的正确性
        self.run_test(Celu(), (input,))
    
    @skipIfUnsupportedMinOpsetVersion(12)
    def test_celu_cast(self):
        # 定义 CELU 类，继承自 torch.nn.Module
        class Celu(torch.nn.Module):
            # 初始化函数，使用默认参数创建 CELU 激活函数
            def __init__(self):
                super().__init__()
                self.celu = torch.nn.CELU()
    
            # 前向传播函数，应用 CELU 激活函数到输入上
            def forward(self, input):
                return self.celu(input)
    
        # 生成一个大小为 (2, 5, 7) 的随机张量，数据类型为 torch.float64
        input = torch.randn(2, 5, 7, dtype=torch.float64)
        # 运行测试函数，验证 CELU 模块的正确性
        self.run_test(Celu(), (input,))
    
    def test_lower_tuple(self):
        # 定义 TupleModule 类，继承自 torch.nn.Module
        class TupleModule(torch.nn.Module):
            # 前向传播函数，接受三个张量输入并返回一个张量
            def forward(self, input1: Tensor, input2: Tensor, input3: Tensor) -> Tensor:
                # 创建包含 input1 和 input2 的元组 a
                a = (input1, input2)
                # 将 a 赋值给 b
                b = a
                # 创建包含 input1、input2 和 input3 的元组 c
                c = (input1, input2, input3)
                # 循环5次
                for i in range(5):
                    # 将 a 的第一个元素赋值给 d
                    d = a[0]
                    # 遍历元组 a
                    for j in range(2):
                        # 将元组 a 解包为 e 和 f
                        e, f = a
                        # 更新元组 a 的值为 (d, f)
                        a = (d, f)
                        # 将 c 的第三个元素赋值给 f
                        f = c[2]
                        # 检查 f 的第一个维度大小是否与 input1 的最后一个维度大小相等
                        if f.size(0) != input1.size(-1):
                            # 将 b 的第二个元素赋值给 g
                            g = b[1]
                            # 更新元组 b 的值为 (g, f)
                            b = (g, f)
                        else:
                            # 创建包含 c 的第二个到最后一个元素的元组 k
                            k = c[1:]
                            # 更新元组 b 的值为 (f, k 的第一个元素)
                            b = (f, k[0])
                    # 将 b 解包为 m 和 n
                    m, n = b
                    # 更新元组 c 的值为 (input1, n, m)
                    c = (input1, n, m)
                # 将元组 c 解包为 p, q, r
                p, q, r = c
                # 返回 p + q + r 的结果
                return p + q + r
    
        # 生成三个大小为 2 的随机张量作为输入
        input1 = torch.randn(2)
        input2 = torch.randn(2)
        input3 = torch.randn(2)
        # 运行测试函数，验证 TupleModule 模块的正确性
        self.run_test(TupleModule(), (input1, input2, input3))
    
    def test_lower_tuple_2(self):
        # 定义 TupleModule 类，继承自 torch.nn.Module
        class TupleModule(torch.nn.Module):
            # 前向传播函数，接受两个张量输入并返回一个元组
            def forward(self, input1: Tensor, input2: Tensor) -> Tuple[Tensor, Tensor]:
                # 创建包含 input1 和 input2 的元组 a
                a = (input1, input2)
                # 循环5次
                for x in range(5):
                    # 将元组 a 解包为 c 和 d
                    c, d = a
                    # 更新元组 a 的值为 (c, d)
                    a = (c, d)
                # 返回更新后的元组 a
                return a
    
        # 生成两个大小为 2 的随机张量作为输入
        input1 = torch.randn(2)
        input2 = torch.randn(2)
        # 运行测试函数，验证 TupleModule 模块的正确性
        self.run_test(TupleModule(), (input1, input2))
    def test_lower_tuple_3(self):
        # 定义一个内部类 TupleModule，继承自 torch.nn.Module
        class TupleModule(torch.nn.Module):
            # 实现 Module 类的 forward 方法
            def forward(
                self,
                input1: Tuple[Tensor, Tensor],  # 输入参数 input1 是一个 Tuple，包含两个 Tensor 类型的元素
                input2: Tuple[Tensor, Tensor],  # 输入参数 input2 是一个 Tuple，包含两个 Tensor 类型的元素
            ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
                # 将 input1 赋值给变量 a
                a = input1
                # 将 input2 赋值给变量 b
                b = input2
                # 循环执行 5 次
                for x in range(5):
                    # 将 a 解包为两个 Tensor，分别赋值给 c 和 d
                    c, d = a
                    # 将 b 解包为两个 Tensor，分别赋值给 e 和 f
                    e, f = b
                    # 检查 c 的形状的第一个维度是否与 e 的第一个维度相同
                    if c.shape[0] == e.shape[0]:
                        # 如果相同，将 e 更新为 e 加上 c
                        e = e + c
                    else:
                        # 如果不同，将 f 更新为 f 加上 d
                        f = f + d
                    # 更新 a 为包含更新后的 e 和 f 的 Tuple
                    a = (e, f)
                    # 更新 b 为包含原始的 c 和 d 的 Tuple
                    b = (c, d)
                # 返回包含更新后的 a 和 b 的 Tuple
                return a, b

        # 创建两个随机的 Tensor，作为输入
        input1 = (torch.randn(2), torch.randn(2))
        input2 = (torch.randn(2), torch.randn(2))
        # 运行测试，使用 TupleModule 类的实例和输入作为参数
        self.run_test(TupleModule(), (input1, input2))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_where(self):
        # 定义一个内部类 Model，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            # 实现 Module 类的 forward 方法
            def forward(self, cond, input, other):
                # 使用 torch.where 函数，根据 cond 条件选择 input 或者 other 的元素
                return torch.where(cond, input, other)

        # 创建一个随机的布尔型 Tensor x
        x = torch.randint(0, 1, (2, 3, 4), dtype=torch.bool)
        # 创建一个随机的浮点型 Tensor y
        y = torch.randn(2, 1, 4)
        # 创建一个全为 1 的浮点型 Tensor z
        z = torch.ones(2, 3, 1)
        # 运行测试，使用 Model 类的实例和输入作为参数
        self.run_test(Model(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipScriptTest()  # scripting tests run for opsets > 11. See: test_where_condition_script
    def test_where_condition(self):
        # 定义一个内部类 Model1，继承自 torch.nn.Module
        class Model1(torch.nn.Module):
            # 实现 Module 类的 forward 方法
            def forward(self, input):
                # 使用 torch.where 函数，根据 input 中大于 0.5 的元素的索引，进行堆叠
                return torch.stack(torch.where(input > 0.5), dim=1)

        # 创建一个随机的布尔型 Tensor x
        x = torch.randint(0, 2, (2, 3, 4), dtype=bool)
        # 运行测试，使用 Model1 类的实例和输入作为参数
        self.run_test(Model1(), (x))

        # 定义一个内部类 Model2，继承自 torch.nn.Module
        class Model2(torch.nn.Module):
            # 实现 Module 类的 forward 方法
            def forward(self, input, other):
                # 使用 torch.where 函数，根据 input 中大于 other 的元素的索引，进行堆叠
                return torch.stack(torch.where(input > other), dim=1)

        # 创建两个随机的布尔型 Tensor x 和 y
        x = torch.randint(0, 1, (2, 3, 4), dtype=bool)
        y = torch.randint(1, 2, (2, 3, 4), dtype=bool)
        # 运行测试，使用 Model2 类的实例和输入作为参数
        self.run_test(Model2(), (x, y))

    @skipIfUnsupportedOpsetVersion([13])
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_where_condition_script(self):
        # 定义一个内部类 Model1，继承自 torch.nn.Module
        class Model1(torch.nn.Module):
            # 实现 Module 类的 forward 方法
            def forward(self, input):
                # 使用 torch.where 函数，根据 input 中大于 0.5 的元素的索引，进行堆叠
                return torch.stack(torch.where(input > 0.5), dim=1)

        # 创建一个随机的布尔型 Tensor x
        x = torch.randint(0, 2, (2, 3, 4), dtype=bool)
        # 运行测试，使用 Model1 类的实例和输入作为参数
        self.run_test(Model1(), (x))

        # 定义一个内部类 Model2，继承自 torch.nn.Module
        class Model2(torch.nn.Module):
            # 实现 Module 类的 forward 方法
            def forward(self, input, other):
                # 使用 torch.where 函数，根据 input 中大于 other 的元素的索引，进行堆叠
                return torch.stack(torch.where(input > other), dim=1)

        # 创建两个随机的布尔型 Tensor x 和 y
        x = torch.randint(0, 1, (2, 3, 4), dtype=bool)
        y = torch.randint(1, 2, (2, 3, 4), dtype=bool)
        # 运行测试，使用 Model2 类的实例和输入作为参数
        self.run_test(Model2(), (x, y))
    def test_empty_branch(self):
        # 定义一个继承自torch.jit.ScriptModule的空分支模型类
        class EmptyBranchModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            # 定义前向传播方法
            def forward(self, input):
                # 计算 input + 1
                out = input + 1
                # 检查 out 的维度是否大于 2
                if out.dim() > 2:
                    # 如果 out 的维度大于 3
                    if out.dim() > 3:
                        # 对 out 进行加法操作
                        out += 3
                    else:
                        # 如果 out 的维度不大于 3，则什么也不做
                        pass
                else:
                    # 如果 out 的维度不大于 2，则什么也不做
                    pass
                # 返回计算结果 out
                return out

        # 创建一个形状为 (1, 2, 3) 的张量，用于测试
        x = torch.randn(1, 2, 3, requires_grad=True)
        # 运行测试，验证空分支模型类的前向传播方法
        self.run_test(EmptyBranchModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_derive_index_scripting(self):
        # 定义一个继承自torch.nn.Module的模型类，用于测试索引推导在脚本模式下的行为
        class MyModule(torch.nn.Module):
            def forward(self, x: Tensor):
                # 初始化一个空列表 j
                j = []
                # 使用 range 函数从后向前遍历 x 的索引
                for idx in range(len(x) - 1, -len(x), -2):
                    # 获取索引为 idx 的元素 y
                    y = x[idx]
                    # 将 x 与 y 的乘积添加到列表 j 中
                    j += [x * y]
                # 返回列表 j
                return j

        # 创建一个形状为 (5, 13) 的张量 x，用于测试
        x = torch.randn(5, 13)
        # 运行测试，验证 MyModule 类的前向传播方法
        self.run_test(MyModule(), x)

        # 定义另一个 MyModule 类，测试另一种索引推导方式
        class MyModule(torch.nn.Module):
            def forward(self, x: Tensor):
                # 初始化一个空列表 j
                j = []
                # 使用 range 函数从前向后遍历 x 的索引
                for idx in range(-len(x), len(x) - 1, 2):
                    # 获取索引为 idx 的元素 y
                    y = x[idx]
                    # 将 x 与 y 的乘积添加到列表 j 中
                    j += [x * y]
                # 返回列表 j
                return j

        # 再次使用形状为 (5, 13) 的张量 x 运行测试
        self.run_test(MyModule(), x)

        # 定义另一个 MyModule 类，测试不同步长的索引推导方式
        class MyModule(torch.nn.Module):
            def forward(self, x: Tensor):
                # 初始化一个空列表 j
                j = []
                # 使用 range 函数从后向前遍历 x 的索引，步长为 -3
                for idx in range(len(x) - 1, -len(x), -3):
                    # 获取索引为 idx 的元素 y
                    y = x[idx]
                    # 将 x 与 y 的乘积添加到列表 j 中
                    j += [x * y]
                # 返回列表 j
                return j

        # 最后一次使用形状为 (5, 13) 的张量 x 运行测试
        self.run_test(MyModule(), x)

        # 定义另一个 MyModule 类，测试不同步长的索引推导方式
        class MyModule(torch.nn.Module):
            def forward(self, x: Tensor):
                # 初始化一个空列表 j
                j = []
                # 使用 range 函数从前向后遍历 x 的索引，步长为 3
                for idx in range(-len(x), len(x) - 1, 3):
                    # 获取索引为 idx 的元素 y
                    y = x[idx]
                    # 将 x 与 y 的乘积添加到列表 j 中
                    j += [x * y]
                # 返回列表 j
                return j

        # 最后一次使用形状为 (5, 13) 的张量 x 运行测试
        self.run_test(MyModule(), x)

    @skipScriptTest()  # 对于 opset 版本小于 11，列表相加的脚本化会失败。请查看 test_derive_index_scripting
    def test_derive_index(self):
        # 定义一个测试函数，测试在不同条件下的自定义模块
        class MyModule(torch.nn.Module):
            def forward(self, x: Tensor):
                # 初始化一个空列表 j 用于存储结果
                j = []
                # 遍历 x 的索引范围，步长为 -2
                for idx in range(len(x) - 1, -len(x), -2):
                    # 获取 x 中指定索引处的值 y
                    y = x[idx]
                    # 将 x 和 y 的乘积添加到 j 中
                    j += [x * y]
                return j

        # 生成一个大小为 (5, 13) 的随机张量 x
        x = torch.randn(5, 13)
        # 运行测试函数，传入自定义模块 MyModule 和输入张量 x
        self.run_test(MyModule(), x)

        # 定义另一个自定义模块 MyModule，与前一个模块结构类似但是索引范围不同
        class MyModule(torch.nn.Module):
            def forward(self, x: Tensor):
                j = []
                # 遍历 x 的索引范围，步长为 2
                for idx in range(-len(x), len(x) - 1, 2):
                    y = x[idx]
                    j += [x * y]
                return j

        # 再次生成一个大小为 (5, 13) 的随机张量 x
        x = torch.randn(5, 13)
        # 运行测试函数，传入自定义模块 MyModule 和输入张量 x
        self.run_test(MyModule(), x)

        # 定义另一个自定义模块 MyModule，与前一个模块结构类似但是步长为 -3
        class MyModule(torch.nn.Module):
            def forward(self, x: Tensor):
                j = []
                # 遍历 x 的索引范围，步长为 -3
                for idx in range(len(x) - 1, -len(x), -3):
                    y = x[idx]
                    j += [x * y]
                return j

        # 运行测试函数，传入自定义模块 MyModule 和相同的输入张量 x
        self.run_test(MyModule(), x)

        # 定义另一个自定义模块 MyModule，与前一个模块结构类似但是步长为 3
        class MyModule(torch.nn.Module):
            def forward(self, x: Tensor):
                j = []
                # 遍历 x 的索引范围，步长为 3
                for idx in range(-len(x), len(x) - 1, 3):
                    y = x[idx]
                    j += [x * y]
                return j

        # 运行测试函数，传入自定义模块 MyModule 和相同的输入张量 x
        self.run_test(MyModule(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_if_transpose(self):
        # 定义一个测试函数，测试条件语句中的转置操作
        class IfModel(torch.nn.Module):
            def forward(self, x):
                # 对输入张量 x 进行维度转置操作
                x = x.transpose(0, 1)
                # 如果转置后的张量 x 的第一个维度大小为 2
                if x.size(0) == 2:
                    # 再次对 x 进行维度转置操作
                    return x.transpose(0, 1)
                else:
                    # 否则返回原始转置后的张量 x
                    return x

        # 生成一个大小为 (2, 3) 的随机张量 x
        x = torch.randn(2, 3)
        # 运行测试函数，传入使用 torch.jit.script 脚本化的 IfModel 模块和输入张量 x
        self.run_test(
            torch.jit.script(IfModel()),
            x,
            output_names=["output_1"],
            dynamic_axes={"output_1": [0, 1]},
        )

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_if_list(self):
        # 定义一个测试函数，测试条件语句中的列表操作
        class IfModel(torch.nn.Module):
            def forward(self, x, y, cond):
                res = []
                # 如果 cond 为 True，则将 x 添加到 res 列表中
                if cond:
                    res = res + [x]
                else:
                    # 否则将 y 添加到 res 列表中
                    res = res + [y]
                return res

        # 生成大小分别为 (2, 3) 和 (3, 3) 的两个随机张量 x 和 y，以及条件张量 cond
        x = torch.randn(2, 3)
        y = torch.randn(3, 3)
        cond = torch.tensor(1, dtype=torch.bool)
        # 运行测试函数，传入使用 torch.jit.script 脚本化的 IfModel 模块和输入张量 x, y, cond
        self.run_test(torch.jit.script(IfModel()), (x, y, cond))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_if_view(self):
        # 定义一个测试函数，测试条件语句中的视图操作和维度转置
        class IfModel(torch.nn.Module):
            def forward(self, x, y, cond):
                # 获取 y 的形状的前两个维度 bs 和 seq
                bs, seq = y.shape[:2]
                # 如果 cond 为 True，则对 x 进行形状重塑操作
                if cond:
                    res = x.view(bs, seq, -1)
                else:
                    # 否则 res 为 y
                    res = y
                # 返回 res 的转置结果
                return res.transpose(1, 2)

        # 生成大小分别为 (2, 16, 2, 2) 和 (2, 16, 8) 的两个随机张量 x 和 y，以及条件张量 cond
        x = torch.randn(2, 16, 2, 2)
        y = torch.randn(2, 16, 8)
        cond = torch.tensor(1, dtype=torch.bool)
        # 运行测试函数，传入使用 torch.jit.script 脚本化的 IfModel 模块和输入张量 x, y, cond
        self.run_test(
            torch.jit.script(IfModel()),
            (x, y, cond),
            output_names=["output_1"],
            dynamic_axes={"output_1": [1]},
        )
    # 装饰器：跳过脚本测试，设置跳过条件为 opset 版本小于 11，原因是在 opset 版本 11 中添加了动态拆分的支持
    @skipScriptTest(
        skip_before_opset_version=11, reason="dynamic split support added in 11"
    )
    def test_split_tensor_scalar(self):
        # 定义一个名为 SplitModel 的类，继承自 torch.nn.Module
        class SplitModel(torch.nn.Module):
            # 定义前向传播函数
            def forward(self, x):
                # 使用 torch.split 函数对张量 x 进行分割，分割成与 x 的第二维度大小相同的若干部分
                return torch.split(x, x.size(1))
    
        # 创建一个形状为 (1, 2, 3) 的随机张量 x，要求计算梯度
        x = torch.randn(1, 2, 3, requires_grad=True)
        # 运行测试，验证 SplitModel 在输入 x 上的表现
        self.run_test(SplitModel(), x)
    
    # 定义一个名为 test_split_tensor_multi 的测试函数
    def test_split_tensor_multi(self):
        # 定义一个名为 SplitModel 的类，继承自 torch.nn.Module
        class SplitModel(torch.nn.Module):
            # 定义前向传播函数
            def forward(self, x):
                # 使用 torch.split 函数对张量 x 进行分割，分割成由 torch.ones(3) 决定的若干部分
                return torch.split(x, torch.ones(3))
    
        # 创建一个形状为 (1, 2, 3) 的随机张量 x，要求计算梯度
        x = torch.randn(1, 2, 3, requires_grad=True)
    
        # 定义一个内部函数 run_model，用于测试实例化 SplitModel 时是否会引发 TypeError 异常
        def run_model():
            SplitModel(x)
    
        # 断言运行 run_model 会引发 TypeError 异常
        self.assertRaises(TypeError, run_model)
    
    # 装饰器：跳过脚本测试，条件为不支持 opset 版本小于 9
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_embedding(self):
        # 定义一个名为 EmbedModel 的类，继承自 torch.nn.Module
        class EmbedModel(torch.nn.Module):
            # 定义前向传播函数，接受 input 和 emb 两个参数
            def forward(self, input, emb):
                # 使用 torch.nn.functional.embedding 函数对 input 应用嵌入 emb，设置填充索引为 1
                return torch.nn.functional.embedding(input, emb, padding_idx=1)
    
        # 创建 EmbedModel 的实例
        model = EmbedModel()
        # 创建一个形状为 (4,) 的整数张量 x，元素值在 [0, 4) 范围内随机选择
        x = torch.randint(4, (4,))
        # 将 x 的第 2 和第 0 个元素设置为 1
        x[2] = x[0] = 1
        # 创建一个形状为 (10, 3) 的随机浮点张量作为嵌入矩阵 embedding_matrix
        embedding_matrix = torch.rand(10, 3)
        # 运行测试，验证 EmbedModel 在输入 x 和 embedding_matrix 上的表现
        self.run_test(model, (x, embedding_matrix))
    
        # 创建一个形状为 (4, 3, 2) 的整数张量 x
        x = torch.randint(4, (4, 3, 2))
        # 将 x 的第 2 行设置为 1
        x[2] = 1
        # 将 x 的第 0 行、第 1 列设置为 1
        x[0][1] = 1
        # 运行测试，验证 EmbedModel 在输入 x 和 embedding_matrix 上的表现
        self.run_test(model, (x, embedding_matrix))
        # 运行测试，设置训练模式为 torch.onnx.TrainingMode.TRAINING，验证 EmbedModel 的表现
        self.run_test(
            model, (x, embedding_matrix), training=torch.onnx.TrainingMode.TRAINING
        )
    
        # 定义一个名为 EmbedModelWithoutPaddingIdx 的类，继承自 torch.nn.Module
        class EmbedModelWithoutPaddingIdx(torch.nn.Module):
            # 定义初始化函数
            def __init__(self):
                super().__init__()
                # 创建一个维度为 (4, 3) 的嵌入层 emb，不设置填充索引
                self.emb = torch.nn.Embedding(4, 3)
    
            # 定义前向传播函数，接受 input 作为参数
            def forward(self, input):
                # 使用嵌入层 emb 对 input 进行嵌入
                return self.emb(input)
    
        # 创建 EmbedModelWithoutPaddingIdx 的实例
        model = EmbedModelWithoutPaddingIdx()
        # 创建一个形状为 (4, 3, 2) 的整数张量 x
        x = torch.randint(4, (4, 3, 2))
        # 运行测试，验证 EmbedModelWithoutPaddingIdx 在输入 x 上的表现
        self.run_test(model, (x,))
    
    # 装饰器：跳过脚本测试，条件为不支持 opset 版本小于 9
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_embedding_module(self):
        # 定义一个名为 EmbedModel 的类，继承自 torch.nn.Module
        class EmbedModel(torch.nn.Module):
            # 定义初始化函数
            def __init__(self):
                super().__init__()
                # 创建一个维度为 (4, 3) 的嵌入层 emb，设置填充索引为 1
                self.emb = torch.nn.Embedding(4, 3, padding_idx=1)
                # 创建一个维度为 (4, 3) 的嵌入层 emb2，设置填充索引为 1，并将其权重设置为全为 1
                self.emb2 = torch.nn.Embedding(4, 3, padding_idx=1)
                with torch.no_grad():
                    self.emb2.weight[1] = torch.ones(3)
    
            # 定义前向传播函数，接受 input 作为参数
            def forward(self, input):
                # 返回 input 经过 emb 和 emb2 嵌入层的结果
                return self.emb(input), self.emb2(input)
    
        # 创建 EmbedModel 的实例
        model = EmbedModel()
        # 创建一个形状为 (4,) 的整数张量 x，元素值在 [0, 4) 范围内随机选择
        x = torch.randint(4, (4,))
        # 将 x 的第 2 和第 0 个元素设置为 1
        x[2] = x[0] = 1
        # 运行测试，验证 EmbedModel 在输入 x 上的表现
        self.run_test(model, (x,))
    
        # 创建一个形状为 (4, 3, 2) 的整数张量 x
        x = torch.randint(4, (4, 3, 2))
        # 将 x 的第 2 行设置为 1
        x[2] = 1
        # 将 x 的第 0 行、第 1 列设置为 1
        x[0][1] = 1
        # 运行测试，验证 EmbedModel 在输入 x 上的表现
        self.run_test(model, (x,))
    
        # 定义一个名为 EmbedModelWithoutPaddingIdx 的类，继承自 torch.nn.Module
        class EmbedModelWithoutPaddingIdx(torch.nn.Module):
            # 定义初始化函数
            def __init__(self):
                super().__init__()
                # 创建一个维度为 (4, 3) 的嵌入层 emb，不设置填充索引
                self.emb = torch.nn.Embedding(4, 3)
    
            # 定义前向传播函数，接受 input 作为参数
            def forward(self, input):
                # 使用嵌入层 emb 对 input 进行嵌入
                return self.emb(input)
    
        # 创建 EmbedModelWithoutPaddingIdx 的实例
        model = EmbedModelWithoutPaddingIdx()
        # 创建一个形状为 (4, 3, 2) 的整数张量 x
        x
    # 定义一个测试嵌入层重新归一化的方法
    def test_embedding_renorm(self):
        # 设置嵌入层的维度和词汇表大小
        n, d = 7, 5
        # 创建一个嵌入层对象，限制每个嵌入向量的最大范数为0.2
        embedding = torch.nn.Embedding(n, d, max_norm=0.2)
        # 创建包含索引的张量，表示要查询的嵌入向量
        idx = torch.tensor([2, 1])
        # 调用自定义的运行测试方法，测试嵌入层的功能
        self.run_test(embedding, idx)

        # 创建另一个嵌入层对象，限制每个嵌入向量的最大范数为0.5，范数类型为L1
        embedding = torch.nn.Embedding(n, d, max_norm=0.5, norm_type=1.0)
        # 创建包含索引的张量，表示要查询的嵌入向量
        idx = torch.tensor([4, 3, 4, 2])
        # 再次调用自定义的运行测试方法，测试嵌入层的功能
        self.run_test(embedding, idx)

    # 根据名称分派不同的循环神经网络测试方法
    def _dispatch_rnn_test(self, name, *args, **kwargs):
        # 如果名称为"elman"，则调用 Elman 循环神经网络测试方法
        if name == "elman":
            self._elman_rnn_test(*args, **kwargs)
        # 如果名称为"lstm"，则调用 LSTM 循环神经网络测试方法
        if name == "lstm":
            self._lstm_test(*args, **kwargs)
        # 如果名称为"gru"，则调用 GRU 循环神经网络测试方法
        if name == "gru":
            self._gru_test(*args, **kwargs)

    # 定义 Elman 循环神经网络测试方法，测试 Elman RNN 的各种参数组合
    def _elman_rnn_test(
        self,
        layers,
        nonlinearity,
        bidirectional,
        initial_state,
        packed_sequence,
        dropout,
        **extra_kwargs,
    ):
        # 实现 Elman RNN 测试方法的具体逻辑，参数包括层数、非线性函数、双向性等

    # 定义 LSTM 循环神经网络测试方法，测试 LSTM 的各种参数组合
    def _lstm_test(
        self,
        layers,
        bidirectional,
        initial_state,
        packed_sequence,
        dropout,
        **extra_kwargs,
    ):
        # 实现 LSTM 测试方法的具体逻辑，参数包括层数、双向性等
    ):
        # 根据 packed_sequence 是否等于 2 确定是否批处理优先
        batch_first = packed_sequence == 2

        # 如果使用了 packed_sequence
        if packed_sequence:
            # 创建带有序列长度的 LSTM 模型
            model = lstm_flattening_result.LstmFlatteningResultWithSeqLength(
                RNN_INPUT_SIZE,
                RNN_HIDDEN_SIZE,
                layers,
                bidirectional,
                dropout,
                batch_first,
            )
            # 如果需要初始状态，创建带有状态的 RNN 模型
            if initial_state:
                model = (
                    rnn_model_with_packed_sequence.RnnModelWithPackedSequenceWithState(
                        model, batch_first
                    )
                )
            # 如果不需要初始状态，创建不带状态的 RNN 模型
            else:
                model = rnn_model_with_packed_sequence.RnnModelWithPackedSequenceWithoutState(
                    model, batch_first
                )
        # 如果未使用 packed_sequence
        else:
            # 创建不带序列长度的 LSTM 模型
            model = lstm_flattening_result.LstmFlatteningResultWithoutSeqLength(
                RNN_INPUT_SIZE,
                RNN_HIDDEN_SIZE,
                layers,
                bidirectional,
                dropout,
                batch_first,
            )

        # 定义生成输入数据的函数
        def make_input(batch_size):
            # 随机生成序列长度
            seq_lengths = np.random.randint(1, RNN_SEQUENCE_LENGTH + 1, size=batch_size)
            seq_lengths = sorted(map(int, seq_lengths), reverse=True)
            # 生成随机输入数据
            inputs = [torch.randn(l, RNN_INPUT_SIZE) for l in seq_lengths]
            # 使用 rnn_utils.pad_sequence 进行填充
            inputs = rnn_utils.pad_sequence(inputs, batch_first=batch_first)
            # 将输入数据组成列表
            inputs = [inputs]
            # 定义输入数据的名称
            input_names = ["input"]
            # 如果需要初始状态
            directions = 2 if bidirectional else 1
            if initial_state:
                # 随机生成初始隐藏状态和细胞状态
                h0 = torch.randn(directions * layers, batch_size, RNN_HIDDEN_SIZE)
                c0 = torch.randn(directions * layers, batch_size, RNN_HIDDEN_SIZE)
                inputs.append((h0, c0))
                input_names.append("h0")
                input_names.append("c0")
            # 如果使用了 packed_sequence
            if packed_sequence != 0:
                # 将序列长度作为输入之一
                inputs.append(torch.IntTensor(seq_lengths))
                input_names.append("seq_lengths")
            # 如果输入数据只有一个，则直接使用该输入
            if len(inputs) == 1:
                input = inputs[0]
            else:
                input = tuple(inputs)
            return input, input_names

        # 生成指定批次大小的输入数据和输入名称
        input, input_names = make_input(RNN_BATCH_SIZE)
        # 定义动态轴的映射关系
        dynamic_axes = {"input": [0, 1], "seq_lengths": [0]}
        if initial_state:
            dynamic_axes.update({"h0": [1], "c0": [1]})
        # 定义导出选项
        export_options = {"input_names": input_names, "dynamic_axes": dynamic_axes}

        # 测试模型在不同批次大小下是否仍然可运行
        other_input, _ = make_input(RNN_BATCH_SIZE + 1)
        self.run_test(
            model, input, additional_test_inputs=[other_input], **export_options
        )

    # 定义用于测试 GRU 的函数
    def _gru_test(
        self,
        layers,
        bidirectional,
        initial_state,
        packed_sequence,
        dropout,
        **extra_kwargs,
    @skipIfUnsupportedMinOpsetVersion(10)
    # 定义一个测试方法，用于测试 FakeQuantizePerTensorModel 类
    def test_fake_quantize_per_tensor(self):
        # 定义一个模型类 FakeQuantizePerTensorModel，继承自 torch.nn.Module
        class FakeQuantizePerTensorModel(torch.nn.Module):
            # 重写 forward 方法，定义模型的前向传播逻辑
            def forward(self, input):
                # 定义量化的比例因子
                scale = 1.0 / 127
                # 定义量化的零点
                zero_point = 0
                # 定义量化的最小值
                quant_min = -128
                # 定义量化的最大值
                quant_max = 127
                # 返回按张量进行仿真量化的结果
                return torch.fake_quantize_per_tensor_affine(
                    input, scale, zero_point, quant_min, quant_max
                )

        # 生成一个随机张量作为输入
        x = torch.randn(6, 4, 3, 3)
        # 运行测试，传入 FakeQuantizePerTensorModel 类实例和输入张量
        self.run_test(FakeQuantizePerTensorModel(), (x))

    # 根据 Opset 版本检查是否支持，如果不支持则跳过测试
    @skipIfUnsupportedMinOpsetVersion(13)
    # 定义一个测试方法，用于测试带动态比例和零点的 FakeQuantizePerTensorModel 类
    def test_fake_quantize_per_tensor_dynamic_scale_zeropoint(self):
        # 定义一个模型类 FakeQuantizePerTensorModel，继承自 torch.nn.Module
        class FakeQuantizePerTensorModel(torch.nn.Module):
            # 重写 forward 方法，定义模型的前向传播逻辑，接受额外的 scale 和 zero_point 参数
            def forward(self, input, scale, zero_point):
                # 定义量化的最小值
                quant_min = -128
                # 定义量化的最大值
                quant_max = 127
                # 返回按张量进行仿真量化的结果，使用传入的 scale 和 zero_point 参数
                return torch.fake_quantize_per_tensor_affine(
                    input, scale, zero_point, quant_min, quant_max
                )

        # 生成一个随机张量作为输入
        x = torch.randn(6, 4, 3, 3)
        # 定义量化的比例因子张量
        scale = torch.tensor(1.0 / 127)
        # 定义量化的零点张量
        zero_point = torch.tensor(0)
        # 运行测试，传入 FakeQuantizePerTensorModel 类实例、输入张量、比例因子张量和零点张量
        self.run_test(FakeQuantizePerTensorModel(), (x, scale, zero_point))

    # 根据 Opset 版本检查是否支持，如果不支持则跳过测试
    @skipIfUnsupportedMinOpsetVersion(13)
    # 定义一个测试方法，用于测试 FakeQuantizePerChannelModel 类
    def test_fake_quantize_per_channel(self):
        # 定义一个模型类 FakeQuantizePerChannelModel，继承自 torch.nn.Module
        class FakeQuantizePerChannelModel(torch.nn.Module):
            # 重写 forward 方法，定义模型的前向传播逻辑
            def forward(self, input):
                # 创建一个全为 1 的张量作为每个通道的最大值
                amax = torch.ones(4)
                # 计算每个通道的量化比例因子
                scale = amax / 127.0
                # 创建一个与 amax 张量相同大小和类型的零点张量
                zero_point = torch.zeros_like(amax, dtype=torch.int)
                # 第一次量化以测试不同的分支
                y = torch.fake_quantize_per_channel_affine(
                    input, scale, zero_point, 1, 0, 255
                )
                # 第二次量化，使用不同的最小和最大值
                return torch.fake_quantize_per_channel_affine(
                    y, scale, zero_point, 1, -128, 127
                )

        # 生成一个随机张量作为输入
        x = torch.randn(6, 4, 3, 3)
        # 运行测试，传入 FakeQuantizePerChannelModel 类实例和输入张量
        self.run_test(FakeQuantizePerChannelModel(), (x))

    # 根据 Opset 版本检查是否支持，如果不支持则跳过测试
    @skipIfUnsupportedMinOpsetVersion(13)
    # 注解：测试被跳过，因为运行时发生了 RuntimeError 错误
    # RuntimeError: Can't redefine method:
    # forward on class: __torch__.torch.nn.modules.linear.Linear
    # 注解：跳过脚本测试
    @skipScriptTest()
    # 定义一个测试方法，用于测试假量化激活函数
    def test_fake_quantize_activation(self):
        # 导入 torch.quantization 模块中的 quantization 子模块
        from torch.ao import quantization

        # 创建一个具有单个输入和输出的线性层
        m = torch.nn.Linear(1, 1)
        # 配置量化参数，包括激活量化和权重量化
        m.qconfig = quantization.QConfig(
            activation=quantization.default_fake_quant,
            weight=quantization.default_per_channel_weight_fake_quant,
        )
        # 准备量化训练模型
        quantization.prepare_qat(m.train(), inplace=True)
        # 启用量化观察器
        m.apply(quantization.enable_observer)
        # 启用假量化
        m.apply(quantization.enable_fake_quant)
        # 遍历模型的所有模块，计算假量化参数
        for module in m.modules():
            if isinstance(module, quantization.FakeQuantize):
                module.calculate_qparams()

        # 禁用量化观察器
        m.apply(quantization.disable_observer)
        # 将模型设置为评估模式
        m.eval()

        # 假量化激活函数是一个特例，其限制量化范围为 (0, 127)，
        # 而标准的8位量化范围为 (-128, 127) 或 (0, 255)。
        # 设置固定的权重、偏置和输入，以测试 ONNX 是否正确处理溢出。
        m.weight = torch.nn.Parameter(torch.tensor([[1.0], [1.0], [1.0]]))
        m.bias = torch.nn.Parameter(torch.tensor([0.0]))
        x = torch.tensor([[150.0], [127.0], [-5.0]])
        # 运行测试方法，验证模型在给定输入下的行为
        self.run_test(m, x)

    # 定义一个测试方法，用于测试批量归一化在训练模式下的行为
    def test_batchnorm_training(self):
        # 定义一个包含批量归一化、卷积和其他模块的自定义模型类
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 第一个批量归一化层，不包含可学习参数
                self.bn1 = torch.nn.BatchNorm2d(3, affine=False)
                # 第一个卷积层
                self.cv1 = torch.nn.Conv2d(3, 3, 10)
                # 第二个批量归一化层，包含可学习的拉伸和偏移参数
                self.bn2 = torch.nn.BatchNorm2d(3, affine=True)
                # 第二个卷积层
                self.cv2 = torch.nn.Conv2d(3, 3, 10)
                # 第三个批量归一化层，不包含可学习参数
                self.bn3 = torch.nn.BatchNorm2d(3, affine=False)

            # 定义前向传播方法
            def forward(self, x):
                x = self.bn1(x)  # 应用第一个批量归一化层
                x = self.cv1(x)  # 应用第一个卷积层
                x = self.bn2(x)  # 应用第二个批量归一化层
                x = self.cv2(x)  # 应用第二个卷积层
                x = self.bn3(x)  # 应用第三个批量归一化层
                return x

        # 创建一个形状为 (10, 3, 20, 20) 的张量，并乘以2来生成随机输入
        x = torch.randn(10, 3, 20, 20) * 2
        # 实例化自定义模型类
        model_export = MyModule()
        # 运行测试方法，验证模型在训练模式下的行为
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.TRAINING,
            rtol=1e-3,
            atol=1e-5,
        )
        # 将模型设置为训练模式
        model_export.train()
        # 再次运行测试方法，验证模型在保持模式下的行为
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.PRESERVE,
            rtol=1e-3,
            atol=1e-5,
        )
    def test_batchnorm_training_mode_fix_layer(self):
        # 定义一个继承自torch.nn.Module的模块类MyModule，用于测试批量归一化在训练模式下的行为
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 第一个批量归一化层，输出通道数为3，affine参数为True，即使用可学习的仿射变换
                self.bn1 = torch.nn.BatchNorm2d(3, affine=True)
                # 第一个卷积层，输入输出通道数均为3，卷积核大小为10
                self.cv1 = torch.nn.Conv2d(3, 3, 10)
                # 第二个批量归一化层，输出通道数为3，affine参数为False，即不使用可学习的仿射变换
                self.bn2 = torch.nn.BatchNorm2d(3, affine=False)
                # 第二个卷积层，输入输出通道数均为3，卷积核大小为10
                self.cv2 = torch.nn.Conv2d(3, 3, 10)
                # 第三个批量归一化层，输出通道数为3，affine参数为True，即使用可学习的仿射变换
                self.bn3 = torch.nn.BatchNorm2d(3, affine=True)
                # 将第三个批量归一化层设置为评估模式
                self.bn3.eval()

            # 定义前向传播函数，接受输入x，按顺序执行归一化和卷积操作，并返回处理后的输出x
            def forward(self, x):
                x = self.bn1(x)  # 应用第一个批量归一化层
                x = self.cv1(x)  # 应用第一个卷积层
                x = self.bn2(x)  # 应用第二个批量归一化层
                x = self.cv2(x)  # 应用第二个卷积层
                x = self.bn3(x)  # 应用第三个批量归一化层
                return x

        # 创建一个输入张量x，形状为(10, 3, 128, 128)，包含随机数据
        x = torch.randn(10, 3, 128, 128)
        # 实例化MyModule类得到模型model_export
        model_export = MyModule()
        # 运行测试，模型处于训练模式下，期望的相对和绝对误差容差分别为1e-3和1e-5
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.TRAINING,
            rtol=1e-3,
            atol=1e-5,
        )
        # 将模型设置为训练模式
        model_export.train()
        # 再次运行测试，模型处于保留模式下，期望的相对和绝对误差容差分别为1e-3和1e-5
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.PRESERVE,
            rtol=1e-3,
            atol=1e-5,
        )

    def test_batchnorm_eval_mode_train_layer(self):
        # 定义一个继承自torch.nn.Module的模块类MyModule，用于测试批量归一化在评估模式下的行为
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 第一个批量归一化层，输出通道数为3，affine参数为True，即使用可学习的仿射变换
                self.bn1 = torch.nn.BatchNorm2d(3, affine=True)
                # 第一个卷积层，输入输出通道数均为3，卷积核大小为10
                self.cv1 = torch.nn.Conv2d(3, 3, 10)
                # 第二个批量归一化层，输出通道数为3，affine参数为False，即不使用可学习的仿射变换
                self.bn2 = torch.nn.BatchNorm2d(3, affine=False)
                # 第二个卷积层，输入输出通道数均为3，卷积核大小为10
                self.cv2 = torch.nn.Conv2d(3, 3, 10)
                # 第三个批量归一化层，输出通道数为3，affine参数为True，即使用可学习的仿射变换
                self.bn3 = torch.nn.BatchNorm2d(3, affine=True)
                # 将第三个批量归一化层设置为训练模式
                self.bn3.train()

            # 定义前向传播函数，接受输入x，按顺序执行归一化和卷积操作，并返回处理后的输出x
            def forward(self, x):
                x = self.bn1(x)  # 应用第一个批量归一化层
                x = self.cv1(x)  # 应用第一个卷积层
                x = self.bn2(x)  # 应用第二个批量归一化层
                x = self.cv2(x)  # 应用第二个卷积层
                x = self.bn3(x)  # 应用第三个批量归一化层
                return x

        # 创建一个输入张量x，形状为(10, 3, 128, 128)，包含随机数据
        x = torch.randn(10, 3, 128, 128)
        # 实例化MyModule类得到模型model_export
        model_export = MyModule()
        # 运行测试，模型处于评估模式下，期望的相对和绝对误差容差分别为1e-3和1e-5
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.EVAL,
            rtol=1e-3,
            atol=1e-5,
        )
        # 将模型设置为评估模式
        model_export.eval()
        # 再次运行测试，模型处于保留模式下，期望的相对和绝对误差容差分别为1e-3和1e-5
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.PRESERVE,
            rtol=1e-3,
            atol=1e-5,
        )
    def test_instancenorm_training(self):
        # 定义一个继承自 torch.nn.Module 的模块类 MyModule
        class MyModule(torch.nn.Module):
            # 构造函数，初始化模块的各个组件
            def __init__(self):
                super().__init__()
                # 第一个 InstanceNorm2d 层，3 个输入通道，启用仿射变换
                self.in1 = torch.nn.InstanceNorm2d(3, affine=True)
                # 第一个卷积层，输入和输出都是 3 个通道，卷积核大小为 10x10
                self.cv1 = torch.nn.Conv2d(3, 3, 10)
                # 第二个 InstanceNorm2d 层，3 个输入通道，不启用仿射变换
                self.in2 = torch.nn.InstanceNorm2d(3, affine=False)
                # 第二个卷积层，输入和输出都是 3 个通道，卷积核大小为 10x10
                self.cv2 = torch.nn.Conv2d(3, 3, 10)
                # 第三个 InstanceNorm2d 层，3 个输入通道，启用仿射变换
                self.in3 = torch.nn.InstanceNorm2d(3, affine=True)

            # 前向传播函数，定义了模型的计算流程
            def forward(self, x):
                x = self.in1(x)  # 使用第一个 InstanceNorm2d 层处理输入
                x = self.cv1(x)  # 使用第一个卷积层处理输入
                x = self.in2(x)  # 使用第二个 InstanceNorm2d 层处理输入
                x = self.cv2(x)  # 使用第二个卷积层处理输入
                x = self.in3(x)  # 使用第三个 InstanceNorm2d 层处理输入
                return x

        # 创建一个 10x3x128x128 的张量作为输入
        x = torch.randn(10, 3, 128, 128)
        # 实例化 MyModule 类，得到模型实例 model_export
        model_export = MyModule()
        # 运行测试，验证模型在训练模式下的行为
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.TRAINING,
            rtol=1e-3,
            atol=1e-5,
        )
        # 将模型设置为训练模式
        model_export.train()
        # 再次运行测试，验证模型在保持模式下的行为
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.PRESERVE,
            rtol=1e-3,
            atol=1e-5,
        )

    def test_instancenorm_training_mode_fix_layer(self):
        # 定义一个继承自 torch.nn.Module 的模块类 MyModule
        class MyModule(torch.nn.Module):
            # 构造函数，初始化模块的各个组件
            def __init__(self):
                super().__init__()
                # 第一个 InstanceNorm2d 层，3 个输入通道，启用仿射变换
                self.in1 = torch.nn.InstanceNorm2d(3, affine=True)
                # 第一个卷积层，输入和输出都是 3 个通道，卷积核大小为 10x10
                self.cv1 = torch.nn.Conv2d(3, 3, 10)
                # 第二个 InstanceNorm2d 层，3 个输入通道，不启用仿射变换
                self.in2 = torch.nn.InstanceNorm2d(3, affine=False)
                # 第二个卷积层，输入和输出都是 3 个通道，卷积核大小为 10x10
                self.cv2 = torch.nn.Conv2d(3, 3, 10)
                # 第三个 InstanceNorm2d 层，3 个输入通道，启用仿射变换
                self.in3 = torch.nn.InstanceNorm2d(3, affine=True)
                # 将第三个 InstanceNorm2d 层设置为评估模式
                self.in3.eval()

            # 前向传播函数，定义了模型的计算流程
            def forward(self, x):
                x = self.in1(x)  # 使用第一个 InstanceNorm2d 层处理输入
                x = self.cv1(x)  # 使用第一个卷积层处理输入
                x = self.in2(x)  # 使用第二个 InstanceNorm2d 层处理输入
                x = self.cv2(x)  # 使用第二个卷积层处理输入
                x = self.in3(x)  # 使用第三个 InstanceNorm2d 层处理输入
                return x

        # 创建一个 10x3x128x128 的张量作为输入
        x = torch.randn(10, 3, 128, 128)
        # 实例化 MyModule 类，得到模型实例 model_export
        model_export = MyModule()
        # 运行测试，验证模型在训练模式下的行为
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.TRAINING,
            rtol=1e-3,
            atol=1e-5,
        )
        # 将模型设置为训练模式
        model_export.train()
        # 再次运行测试，验证模型在保持模式下的行为
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.PRESERVE,
            rtol=1e-3,
            atol=1e-5,
        )
    def test_instancenorm_eval_mode_train_layer(self):
        # 定义一个继承自 torch.nn.Module 的子类 MyModule
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个包含 8 个特征的 InstanceNorm2d 层，并启用可学习的仿射变换
                self.in1 = torch.nn.InstanceNorm2d(8, affine=True)
                # 创建一个输入输出均为 8 通道的二维卷积层
                self.cv1 = torch.nn.Conv2d(8, 8, 10)
                # 创建一个包含 8 个特征的 InstanceNorm2d 层，并禁用仿射变换
                self.in2 = torch.nn.InstanceNorm2d(8, affine=False)
                # 再次创建一个输入输出均为 8 通道的二维卷积层
                self.cv2 = torch.nn.Conv2d(8, 8, 10)
                # 创建第三个 InstanceNorm2d 层，包含 8 个特征并启用仿射变换，并设为训练模式
                self.in3 = torch.nn.InstanceNorm2d(8, affine=True)
                self.in3.train()

            # 定义前向传播方法
            def forward(self, x):
                # 使用第一个 InstanceNorm2d 层处理输入数据 x
                x = self.in1(x)
                # 使用第一个卷积层处理处理后的数据 x
                x = self.cv1(x)
                # 使用第二个 InstanceNorm2d 层处理数据 x
                x = self.in2(x)
                # 使用第二个卷积层处理处理后的数据 x
                x = self.cv2(x)
                # 使用第三个 InstanceNorm2d 层处理数据 x
                x = self.in3(x)
                return x

        # 创建输入 tensor x，大小为 (10, 8, 128, 128)
        x = torch.randn(10, 8, 128, 128)
        # 实例化 MyModule 类
        model_export = MyModule()
        # 运行测试，评估模式下训练图层
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.EVAL,
            rtol=1e-3,
            atol=1e-5,
        )
        # 将模型设为评估模式
        model_export.eval()
        # 再次运行测试，保持训练图层的评估模式
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.PRESERVE,
            rtol=1e-3,
            atol=1e-5,
        )

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_dropout_training(self):
        # 定义一个继承自 torch.nn.Module 的子类 MyModule
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个丢弃率为 0.4 的 Dropout 层
                self.dropout = torch.nn.Dropout(0.4)

            # 定义前向传播方法
            def forward(self, x):
                # 对输入 x 应用 Dropout 层
                dropout = self.dropout(x)
                return dropout

        # 创建 MyModule 类的实例 model
        model = MyModule()
        # 创建一个大小为 10 的随机输入 tensor x
        x = torch.randn(10)
        # 设置模型为训练模式
        model.train()

        # 创建一个字节流对象 model_onnx
        model_onnx = io.BytesIO()
        # 使用 Torch ONNX 导出功能将模型导出到 model_onnx
        torch.onnx.export(
            model,
            x,
            model_onnx,
            opset_version=self.opset_version,
            do_constant_folding=False,
            training=torch.onnx.TrainingMode.TRAINING,
        )
        # 创建一个 ONNX 运行时会话对象 ort_sess
        ort_sess = verification._ort_session(model_onnx)
        # 使用 ONNX 运行时会话对象 ort_sess 运行 ONNX 模型，并获取输出 ort_outs
        ort_outs = verification._run_onnx(ort_sess, (x,))
        # 断言，检查是否所有元素都不相等
        assert not torch.all(torch.eq(x, torch.from_numpy(ort_outs[0])))

        # 将模型转换为 Torch 脚本模型 script_model
        script_model = torch.jit.script(model)
        # 再次使用 Torch ONNX 导出功能将模型导出到 model_onnx
        model_onnx = io.BytesIO()
        torch.onnx.export(
            model,
            x,
            model_onnx,
            opset_version=self.opset_version,
            do_constant_folding=False,
            training=torch.onnx.TrainingMode.TRAINING,
        )
        # 使用 ONNX 运行时会话对象 ort_sess 运行 ONNX 模型，并获取输出 ort_outs
        ort_outs = verification._run_onnx(ort_sess, (x,))
        # 断言，检查是否所有元素都不相等
        assert not torch.all(torch.eq(x, torch.from_numpy(ort_outs[0])))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_dropout_training_zero(self):
        # 定义一个名为test_dropout_training_zero的测试函数
        class MyModule(torch.nn.Module):
            # 定义一个继承自torch.nn.Module的子类MyModule
            def __init__(self):
                # 初始化函数
                super().__init__()
                self.dropout = torch.nn.Dropout(0.5)
                # 创建一个Dropout层对象，丢弃概率为0.5

            def forward(self, x):
                # 前向传播函数定义
                dropout = self.dropout(x)
                # 对输入x应用Dropout层
                return dropout
                # 返回Dropout后的结果

        model = MyModule()
        # 创建MyModule类的实例model

        # ensure there are no zeros in the input
        # 确保输入中没有零值
        x = torch.randn(10, 3, 128, 128)
        # 生成一个形状为(10, 3, 128, 128)的随机张量x
        y = x.numpy()
        # 将张量x转换为NumPy数组y
        y_mask = np.where(y == 0, 1, y)
        # 使用np.where将y中等于0的元素替换为1，否则保持原值
        input = torch.from_numpy(y_mask)
        # 将NumPy数组y_mask转换为PyTorch张量input
        nb_elements = torch.numel(input)
        # 计算输入张量input中元素的总数

        model.train()
        # 将模型设置为训练模式
        model_onnx = io.BytesIO()
        # 创建一个BytesIO对象model_onnx，用于存储导出的ONNX模型
        torch.onnx.export(
            model,
            x,
            model_onnx,
            opset_version=self.opset_version,
            do_constant_folding=False,
            training=torch.onnx.TrainingMode.TRAINING,
        )
        # 将PyTorch模型导出为ONNX格式，并存储到model_onnx中

        ort_sess = verification._ort_session(model_onnx)
        # 使用ONNX Runtime创建一个会话ort_sess，加载model_onnx
        ort_outs = verification._run_onnx(ort_sess, (x,))
        # 在ONNX Runtime会话ort_sess上运行输入x，并获取输出ort_outs

        y = model(input)
        # 使用模型对输入input进行预测
        output = y.cpu().numpy()
        # 将模型输出y转换为NumPy数组output
        ort_mask = np.where(ort_outs[0] != 0, 1, 0)
        # 使用np.where将ort_outs中不为0的元素替换为1，否则为0
        pyt_mask = np.where(output != 0, 1, 0)
        # 使用np.where将output中不为0的元素替换为1，否则为0

        ratio_pytorch = np.sum(pyt_mask) / nb_elements
        # 计算PyTorch模型输出中非零元素的比例
        ratio_ort = np.sum(ort_mask) / nb_elements
        # 计算ONNX Runtime模型输出中非零元素的比例

        np.testing.assert_allclose(ratio_pytorch, ratio_ort, rtol=0.01, atol=0.01)
        # 使用NumPy测试函数断言ratio_pytorch和ratio_ort的近似性

        script_model = torch.jit.script(model)
        # 使用Torch脚本化将模型转换为脚本模型script_model
        y = model(input)
        # 再次使用模型对输入input进行预测
        output = y.cpu().numpy()
        # 将模型输出y转换为NumPy数组output
        model_onnx = io.BytesIO()
        # 创建一个新的BytesIO对象model_onnx，用于存储重新导出的ONNX模型
        torch.onnx.export(
            model,
            x,
            model_onnx,
            opset_version=self.opset_version,
            do_constant_folding=False,
            training=torch.onnx.TrainingMode.TRAINING,
        )
        # 将更新后的PyTorch模型再次导出为ONNX格式，并存储到model_onnx中

        ort_sess = verification._ort_session(model_onnx)
        # 使用ONNX Runtime创建一个会话ort_sess，加载更新后的model_onnx
        ort_outs = verification._run_onnx(ort_sess, (x,))
        # 在更新后的ONNX Runtime会话ort_sess上运行输入x，并获取输出ort_outs
        ort_mask = np.where(ort_outs[0] != 0, 1, 0)
        # 使用np.where将ort_outs中不为0的元素替换为1，否则为0
        pyt_mask = np.where(output != 0, 1, 0)
        # 使用np.where将output中不为0的元素替换为1，否则为0

        ratio_pytorch = np.sum(pyt_mask) / nb_elements
        # 计算更新后的PyTorch模型输出中非零元素的比例
        ratio_ort = np.sum(ort_mask) / nb_elements
        # 计算更新后的ONNX Runtime模型输出中非零元素的比例

        np.testing.assert_allclose(ratio_pytorch, ratio_ort, rtol=0.01, atol=0.01)
        # 使用NumPy测试函数断言更新后的ratio_pytorch和ratio_ort的近似性

    def test_conv_bn(self):
        # 定义一个名为test_conv_bn的测试函数
        class MyModule(torch.nn.Module):
            # 定义一个继承自torch.nn.Module的子类MyModule
            def __init__(self):
                # 初始化函数
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 16, kernel_size=1, stride=2, padding=3, bias=True
                )
                # 创建一个卷积层对象conv，输入通道数为3，输出通道数为16，卷积核大小为1x1，步长为2，填充为3，包含偏置项
                self.bn = torch.nn.BatchNorm2d(16, affine=True)
                # 创建一个批归一化层对象bn，输入通道数为16，affine参数设置为True表示包含可学习的仿射变换参数

            def forward(self, x):
                # 前向传播函数定义
                x = self.conv(x)
                # 对输入x应用卷积层
                bn = self.bn(x)
                # 对卷积层输出x应用批归一化层bn
                return bn
                # 返回批归一化后的结果

        model_export = MyModule()
        # 创建MyModule类的实例model_export
        x = torch.randn(10, 3, 128, 128)
        # 生成一个形状为(10, 3, 128, 128)的随机张量x
        self.run_test(model_export, (x,), training=torch.onnx.TrainingMode.EVAL)
        # 调用self.run_test方法，评估模型model_export在输入x上的表现

        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.TRAINING,
            rtol=1e-3,
            atol=1e-5,
        )
        # 再次调用self.run_test方法，训练模型model_export在输入x上的表现，设置相对和绝对误差容差
    def test_multiple_conv_bn(self):
        # 定义一个测试函数，测试包含多个卷积层和批归一化的模块
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 第一个卷积层：输入通道数为3，输出通道数为64，卷积核大小为7x7，步长为2，填充为3，无偏置项
                self.conv1 = torch.nn.Conv2d(
                    3, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
                # 第二个卷积层：输入通道数为64，输出通道数为2，卷积核大小为1x1，步长为1，填充为0，无偏置项
                self.conv2 = torch.nn.Conv2d(
                    64, 2, kernel_size=1, stride=1, padding=0, bias=False
                )
                # 第三个卷积层：输入通道数为2，输出通道数为2，卷积核大小为3x3，步长为1，填充为1，无偏置项
                self.conv3 = torch.nn.Conv2d(
                    2, 2, kernel_size=3, stride=1, padding=1, bias=False
                )
                # 第一个批归一化层，输入通道数为64
                self.bn = torch.nn.BatchNorm2d(64)
                # 第二个批归一化层，输入通道数为2
                self.bn2 = torch.nn.BatchNorm2d(2)
                # ReLU 激活函数，inplace=True 表示原地操作
                self.relu = torch.nn.ReLU(inplace=True)
                # 最大池化层，池化核大小为3x3，步长为2，填充为1
                self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            def forward(self, x):
                # 前向传播函数
                x = self.conv1(x)   # 第一层卷积
                x = self.bn(x)      # 第一层批归一化
                x = self.relu(x)    # 第一层ReLU激活
                x = self.maxpool(x) # 第一层最大池化
                x = self.conv2(x)   # 第二层卷积
                x = self.bn2(x)     # 第二层批归一化
                x = self.relu(x)    # 第二层ReLU激活
                x = self.conv3(x)   # 第三层卷积
                x = self.bn2(x)     # 第三层批归一化
                x = self.relu(x)    # 第三层ReLU激活
                return x            # 返回输出

        # 实例化模块对象
        model_export = MyModule()
        # 生成输入数据，形状为[2, 3, 224, 224]
        x = torch.randn(2, 3, 224, 224)
        # 运行测试，训练模式，设置相对误差容差为1e-3，绝对误差容差为1e-5
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.TRAINING,
            rtol=1e-3,
            atol=1e-5,
        )
        # 运行测试，评估模式
        self.run_test(model_export, (x,), training=torch.onnx.TrainingMode.EVAL)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_nms(self):
        # 定义测试非极大值抑制（NMS）功能
        num_boxes = 100
        # 生成随机边界框，形状为[100, 4]
        boxes = torch.rand(num_boxes, 4)
        # 将边界框右下角坐标转换为边界框的宽高坐标
        boxes[:, 2:] += boxes[:, :2]
        # 生成随机得分，形状为[100]
        scores = torch.randn(num_boxes)

        # 定义模块类，执行NMS操作
        class Module(torch.nn.Module):
            def forward(self, boxes, scores):
                return torchvision.ops.nms(boxes, scores, 0.5)

        # 运行测试，传入边界框和得分
        self.run_test(Module(), (boxes, scores))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_batched_nms(self):
        # 定义测试批量NMS功能
        num_boxes = 100
        # 生成随机边界框，形状为[100, 4]
        boxes = torch.rand(num_boxes, 4)
        # 将边界框右下角坐标转换为边界框的宽高坐标
        boxes[:, 2:] += boxes[:, :2]
        # 生成随机得分，形状为[100]
        scores = torch.randn(num_boxes)
        # 生成随机索引，形状为[100]
        idxs = torch.randint(0, 5, size=(num_boxes,))

        # 定义模块类，执行批量NMS操作
        class Module(torch.nn.Module):
            def forward(self, boxes, scores, idxs):
                return torchvision.ops.batched_nms(boxes, scores, idxs, 0.5)

        # 运行测试，传入边界框、得分和索引
        self.run_test(Module(), (boxes, scores, idxs))

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    # 定义测试函数 test_clip_boxes_to_image
    def test_clip_boxes_to_image(self):
        # 创建一个 5x4 的张量 boxes，其值为标准正态分布随机数乘以500
        boxes = torch.randn(5, 4) * 500
        # 将每个框的右下角坐标调整为相对左上角的偏移量
        boxes[:, 2:] += boxes[:, :2]
        # 创建一个大小为 200x300 的随机张量 size
        size = torch.randn(200, 300)

        # 创建一个大小为 300x400 的随机张量 size_2
        size_2 = torch.randn(300, 400)

        # 定义内部类 Module，继承自 torch.nn.Module
        class Module(torch.nn.Module):
            # 定义 forward 方法，接收 boxes 和 size 作为参数
            def forward(self, boxes, size):
                # 获取 size 张量的形状，并保存在 shape 变量中
                shape = (size.shape[0], size.shape[1])
                # 调用 torchvision 库中的 boxes.clip_boxes_to_image 函数，将 boxes 裁剪到图像边界内
                return torchvision.ops.boxes.clip_boxes_to_image(boxes, shape)

        # 运行测试，传入 Module 的实例、boxes 和 size 作为参数
        self.run_test(
            Module(),
            (boxes, size),
            input_names=["boxes", "size"],
            dynamic_axes={"size": [0, 1]},
            additional_test_inputs=[(boxes, size), (boxes, size_2)],
        )

    # 装饰器函数 @skipScriptTest，跳过脚本测试，原因是在 ONNX 中不支持通过 prim::isinstance 条件判断
    @skipScriptTest(
        reason="Conditioning on input type via prim::isinstance unsupported in ONNX"
    )
    # 装饰器函数 @skipIfUnsupportedMinOpsetVersion，跳过不支持的最小操作集版本（Opset 11）
    @skipIfUnsupportedMinOpsetVersion(11)
    # 定义测试函数 test_roi_align
    def test_roi_align(self):
        # 创建一个形状为 (1, 1, 10, 10) 的随机张量 x，数据类型为 float32
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        # 创建一个包含单个 ROI 的张量 single_roi，数据类型为 float32
        single_roi = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32)
        # 创建 RoIAlign 模型，输出大小为 (5, 5)，尺度因子为 1.0，采样点个数为 2
        model = torchvision.ops.RoIAlign((5, 5), 1.0, 2)
        # 运行测试，传入 model 和 (x, single_roi) 作为参数
        self.run_test(model, (x, single_roi))

    # 装饰器函数 @skipScriptTest，跳过脚本测试，原因是在 ONNX 中不支持通过 prim::isinstance 条件判断
    @skipScriptTest(
        reason="Conditioning on input type via prim::isinstance unsupported in ONNX"
    )
    # 装饰器函数 @skipIfUnsupportedMinOpsetVersion，跳过不支持的最小操作集版本（Opset 16）
    @skipIfUnsupportedMinOpsetVersion(16)
    # 定义测试函数 test_roi_align_aligned
    def test_roi_align_aligned(self):
        # 创建一个形状为 (1, 1, 10, 10) 的随机张量 x，数据类型为 float32
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        # 创建一个包含单个 ROI 的张量 single_roi，数据类型为 float32
        single_roi = torch.tensor([[0, 1.5, 1.5, 3, 3]], dtype=torch.float32)
        # 创建 RoIAlign 模型，输出大小为 (5, 5)，尺度因子为 1.0，采样点个数为 2，启用对齐模式
        model1 = torchvision.ops.RoIAlign((5, 5), 1.0, 2, aligned=True)
        # 运行测试，传入 model1 和 (x, single_roi) 作为参数
        self.run_test(model1, (x, single_roi))

        # 创建其他 RoIAlign 模型，用不同的参数进行测试
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0.2, 0.3, 4.5, 3.5]], dtype=torch.float32)
        model2 = torchvision.ops.RoIAlign((5, 5), 0.5, 3, aligned=True)
        self.run_test(model2, (x, single_roi))

        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0.2, 0.3, 4.5, 3.5]], dtype=torch.float32)
        model3 = torchvision.ops.RoIAlign((5, 5), 1.8, 2, aligned=True)
        self.run_test(model3, (x, single_roi))

        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0.2, 0.3, 4.5, 3.5]], dtype=torch.float32)
        model4 = torchvision.ops.RoIAlign((2, 2), 2.5, 0, aligned=True)
        self.run_test(model4, (x, single_roi))

    # 装饰器函数 @skipScriptTest，跳过脚本测试，原因是在 ONNX 中不支持通过 prim::isinstance 条件判断
    @skipScriptTest(
        reason="Conditioning on input type via prim::isinstance unsupported in ONNX"
    )
    # 装饰器函数 @skipIfUnsupportedMinOpsetVersion，跳过不支持的最小操作集版本（Opset 11）
    @skipIfUnsupportedMinOpsetVersion(11)
    # 定义测试函数 test_roi_pool
    def test_roi_pool(self):
        # 创建一个形状为 (1, 1, 10, 10) 的随机张量 x，数据类型为 float32
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        # 创建一个包含单个 ROI 的张量 rois，数据类型为 float32
        rois = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32)
        # 创建 RoIPool 模型，池化输出大小为 (5, 5)，尺度因子为 2.0
        model = torchvision.ops.RoIPool((5, 5), 2.0)
        # 运行测试，传入 model 和 (x, rois) 作为参数
        self.run_test(model, (x, rois))

    # 装饰器函数 @skipIfUnsupportedMinOpsetVersion，跳过不支持的最小操作集版本（Opset 11）
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_resize_images(self):
        # 定义一个继承自 torch.nn.Module 的测试转换模块
        class TransformModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化通用 RCNN 转换
                self.transform = _init_test_generalized_rcnn_transform()

            def forward(self, images):
                # 调用转换对象的 resize 方法，将输入的图像 images 进行尺寸调整，并返回调整后的第一个图像
                return self.transform.resize(images, None)[0]

        # 创建输入数据
        input = torch.rand(3, 10, 20)
        input_test = torch.rand(3, 100, 150)
        # 运行测试
        self.run_test(
            TransformModule(),  # 使用定义好的 TransformModule 进行测试
            (input,),  # 输入数据
            input_names=["input1"],  # 输入数据的名称
            dynamic_axes={"input1": [0, 1, 2]},  # 动态轴定义
            additional_test_inputs=[(input,), (input_test,)],  # 额外的测试输入数据
        )

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_transform_images(self):
        # 定义一个继承自 torch.nn.Module 的测试转换模块
        class TransformModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化通用 RCNN 转换
                self.transform = _init_test_generalized_rcnn_transform()

            def forward(self, images: List[Tensor]):
                # 调用转换对象的 transform 方法，将输入的图像列表 images 进行转换，并返回转换后的第一个张量
                return self.transform(images)[0].tensors

        # 创建输入数据
        input = torch.rand(3, 100, 200), torch.rand(3, 200, 200)
        input_test = torch.rand(3, 100, 200), torch.rand(3, 200, 200)
        # 运行测试
        self.run_test(
            TransformModule(),  # 使用定义好的 TransformModule 进行测试
            (input,),  # 输入数据
            additional_test_inputs=[(input,), (input_test,)],  # 额外的测试输入数据
        )

    def get_features(self, images):
        # 获取输入图像的高度和宽度
        s0, s1 = images.shape[-2:]
        # 定义特征列表，包括不同尺度的特征图
        features = [
            ("0", torch.rand(2, 256, s0 // 4, s1 // 4)),
            ("1", torch.rand(2, 256, s0 // 8, s1 // 8)),
            ("2", torch.rand(2, 256, s0 // 16, s1 // 16)),
            ("3", torch.rand(2, 256, s0 // 32, s1 // 32)),
            ("4", torch.rand(2, 256, s0 // 64, s1 // 64)),
        ]
        # 将特征列表转换为有序字典
        features = OrderedDict(features)
        return features

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_rpn(self):
        # 定义一个测试用的 RPN 模块
        class RPNModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化 RPN 模块
                self.rpn = _init_test_rpn()

            def forward(self, images, features: Dict[str, Tensor]):
                # 创建 ImageList 对象，用于处理输入的图像数据
                images_m = torchvision.models.detection.image_list.ImageList(
                    images, [(i.shape[-1], i.shape[-2]) for i in images]
                )
                # 调用 RPN 模块的前向传播方法
                return self.rpn(images_m, features)

        # 创建随机的图像数据
        images = torch.rand(2, 3, 150, 150)
        # 获取图像特征
        features = self.get_features(images)
        # 创建第二组随机图像数据
        images2 = torch.rand(2, 3, 80, 80)
        # 获取第二组图像的特征
        test_features = self.get_features(images2)

        # 创建 RPN 模块的实例
        model = RPNModule()
        # 设置模型为评估模式
        model.eval()
        # 运行 RPN 模块的前向传播
        model(images, features)
        # 运行测试
        self.run_test(
            model,
            (images, features),
            input_names=["input1", "input2", "input3", "input4", "input5", "input6"],
            dynamic_axes={
                "input1": [0, 1, 2, 3],
                "input2": [0, 1, 2, 3],
                "input3": [0, 1, 2, 3],
                "input4": [0, 1, 2, 3],
                "input5": [0, 1, 2, 3],
                "input6": [0, 1, 2, 3],
            },
            additional_test_inputs=[(images, features), (images2, test_features)],
            # 禁用字典检查
            # dict_check=False,
        )

    @skipIfUnsupportedMaxOpsetVersion(15)  # 如果 Opset 版本不支持最大版本号为 15，则跳过测试
    @skipIfUnsupportedMinOpsetVersion(11)  # 如果 Opset 版本不支持最小版本号为 11，则跳过测试
    @skipScriptTest()  # 跳过脚本测试
    def test_multi_scale_roi_align(self):
        # 定义一个测试多尺度 RoI Align 的模块
        class TransformModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化多尺度 RoI Align 模块
                self.model = torchvision.ops.MultiScaleRoIAlign(
                    ["feat1", "feat2"], 3, 2
                )
                # 设置图像大小
                self.image_sizes = [(512, 512)]

            def forward(self, input: Dict[str, Tensor], boxes: List[Tensor]) -> Tensor:
                # 执行多尺度 RoI Align 操作
                return self.model(input, boxes, self.image_sizes)

        # 创建输入字典 i，包含特征 "feat1" 和 "feat2"
        i = OrderedDict()
        i["feat1"] = torch.rand(1, 5, 64, 64)
        i["feat2"] = torch.rand(1, 5, 16, 16)
        # 创建随机框框，用于 RoI Align 操作
        boxes = torch.rand(6, 4) * 256
        boxes[:, 2:] += boxes[:, :2]

        # 创建第二组输入字典 i1，包含特征 "feat1" 和 "feat2"
        i1 = OrderedDict()
        i1["feat1"] = torch.rand(1, 5, 64, 64)
        i1["feat2"] = torch.rand(1, 5, 16, 16)
        # 创建第二组随机框框，用于 RoI Align 操作
        boxes1 = torch.rand(6, 4) * 256
        boxes1[:, 2:] += boxes1[:, :2]

        # 运行测试
        self.run_test(
            TransformModule(),
            (
                i,
                [boxes],
            ),
            additional_test_inputs=[
                (
                    i,
                    [boxes],
                ),
                (
                    i1,
                    [boxes1],
                ),
            ],
        )
    # 定义一个测试方法 test_set_，用于测试模型类 M 的行为
    def test_set_(self):
        # 定义一个内部模型类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x, y):
                # 使用 y 的值来设置 x 的值
                x.set_(y)
                # 返回设置后的 x
                return x

        # 创建一个 2x3 全一张量 x
        x = torch.ones(2, 3)
        # 创建一个形状为 4x6 的随机张量 y
        y = torch.randn(4, 6)

        # 运行测试方法 run_test，传入模型 M 的实例，输入为 (x, y)，期望保留的 ONNX 输入索引为 [1]
        self.run_test(M(), (x, y), remained_onnx_input_idx=[1])

        # 创建一个形状为 5x2 的随机张量 y2
        y2 = torch.randn(5, 2)

        # 再次运行测试方法 run_test，传入模型 M 的实例，输入为 (x, y)
        # 期望保留的 ONNX 输入索引为 [1]，指定输入名称为 ["x", "y"]，动态轴为 {"x": [0, 1], "y": [0, 1]}
        # 并额外提供测试输入 (y, y2)
        self.run_test(
            M(),
            (x, y),
            remained_onnx_input_idx=[1],
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1], "y": [0, 1]},
            additional_test_inputs=[(y, y2)],
        )

    # 标记装饰器，如果不支持最小的 Opset 版本 9，则跳过执行该测试方法
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_set_attr_modules(self):
        # 定义内部模块 InnerModule2
        class InnerModule2(torch.nn.Module):
            # 内部模块构造函数，接收嵌入维度参数 embedding_dim
            def __init__(self, embedding_dim):
                super().__init__()
                # 初始化权重为嵌入维度相关的嵌入向量
                self.weights = InnerModule2.get_embedding(embedding_dim)
                # 注册一个缓冲区 _float_tensor，包含一个浮点数张量
                self.register_buffer("_float_tensor", torch.FloatTensor(1))
                # 设置一个常数值为 2
                self.const = 2

            # 静态方法：根据嵌入维度计算并返回嵌入向量
            @staticmethod
            def get_embedding(embedding_dim: int):
                emb = 4 / ((embedding_dim // 2) - 1)
                emb = torch.exp(
                    torch.arange((embedding_dim // 2), dtype=torch.float) * -emb
                )
                return emb

            # 前向传播函数，接收输入和可选的增量状态张量 incremental_state
            def forward(self, input, incremental_state: Optional[Tensor] = None):
                bsz, seq_len = input.shape[0], input.shape[1]
                # 更新常数值为 3
                self.const = 3
                # 如果权重为 None，则重新获取嵌入向量
                if self.weights is None:
                    self.weights = InnerModule.get_embedding(self.embedding_dim)
                # 将权重转移到 _float_tensor 上
                self.weights = self.weights.to(self._float_tensor)
                # 权重乘以常数值
                self.weights = self.weights * self.const
                # 如果存在增量状态，则返回部分权重的扩展
                if incremental_state is not None:
                    pos = seq_len
                    return self.weights[1 + pos, :].expand(bsz, 1, -1)
                # 否则，按索引选择权重，并重新视图重塑
                return self.weights.index_select(
                    0, torch.ones((bsz * seq_len), dtype=torch.int64)
                ).view(bsz, seq_len, -1)

        # 定义内部模块 InnerModule
        class InnerModule(torch.nn.Module):
            # 内部模块构造函数，接收嵌入维度参数 embedding_dim
            def __init__(self, embedding_dim):
                super().__init__()
                # 初始化权重为嵌入维度相关的嵌入向量
                self.weights = InnerModule.get_embedding(embedding_dim)
                # 初始化内部模块2
                self.module = InnerModule2(embedding_dim=8)

            # 静态方法：根据嵌入维度计算并返回嵌入向量
            @staticmethod
            def get_embedding(embedding_dim: int):
                emb = 4 / ((embedding_dim // 2) - 1)
                emb = torch.exp(
                    torch.arange((embedding_dim // 2), dtype=torch.float) * -emb
                )
                return emb

            # 前向传播函数，接收输入 x
            def forward(self, x):
                # 返回内部模块2的输出加上权重
                return self.module(x) + self.weights

        # 定义模块类 Module
        class Module(torch.nn.Module):
            # 模块构造函数
            def __init__(self):
                super().__init__()
                # 初始化内部模块
                self.module = InnerModule(embedding_dim=8)

            # 前向传播函数，接收输入 x
            def forward(self, x):
                # 返回内部模块的输出
                return self.module(x)

        # 生成一个形状为 (3, 256) 的随机张量 x
        x = torch.randn(3, 256)
        # 运行测试，传入模块实例 Module，输入 x，指定输入名称 "x" 和动态轴
        self.run_test(Module(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]})
        # 运行测试，传入模块实例 Module，输入 x，指定空的保留的 ONNX 输入索引
        self.run_test(Module(), (x,), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_set_attr_modules_2(self):
        # 定义内部模块 InnerModule，继承自 torch.nn.Module
        class InnerModule(torch.nn.Module):
            # 初始化方法，接受 embedding_dim 参数
            def __init__(self, embedding_dim):
                super().__init__()
                # 设置模块的 embedding 维度和常数值
                self.embedding_dim = embedding_dim
                self.const = 2.5
                # 调用静态方法 get_embedding 初始化权重
                self.weights = InnerModule.get_embedding(self.embedding_dim)
                # 注册一个名为 "_float_tensor" 的缓冲区
                self.register_buffer("_float_tensor", torch.FloatTensor(1))

            # 静态方法，根据 embedding_dim 计算并返回初始化的权重张量
            @staticmethod
            def get_embedding(embedding_dim: int):
                emb = 4 / ((embedding_dim // 2) - 1)
                emb = torch.exp(
                    torch.arange((embedding_dim // 2), dtype=torch.float) * -emb
                )
                return emb

            # 前向传播方法，接受 input 和 incremental_state 参数
            def forward(self, input, incremental_state: Optional[Tensor] = None):
                # 获取输入张量的 batch size 和序列长度
                bsz, seq_len = input.shape[0], input.shape[1]
                # 修改常数值为 1.5
                self.const = 1.5
                # 重新获取权重
                self.weights = InnerModule.get_embedding(self.embedding_dim)
                # 返回经过索引选择和视图变换后的权重张量乘以常数值的结果
                return (
                    self.weights.index_select(
                        0, torch.ones((bsz * seq_len), dtype=torch.int64)
                    ).view(bsz, seq_len, -1)
                ) * self.const

        # 定义外部模块 Module，继承自 torch.nn.Module
        class Module(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建 InnerModule 类的实例，设置 embedding_dim 为 8
                self.module = InnerModule(embedding_dim=8)

            # 前向传播方法，接受 x 参数
            def forward(self, x):
                # 调用内部模块的 forward 方法
                return self.module(x)

        # 生成输入张量 x，形状为 (3, 256)
        x = torch.randn(3, 256)
        # 运行测试方法 run_test，传入 Module 的实例和输入张量 x，指定输入名称和动态轴
        self.run_test(Module(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]})
        # 运行测试方法 run_test，传入 Module 的实例和输入张量 x，指定空的 remained_onnx_input_idx
        self.run_test(Module(), (x,), remained_onnx_input_idx=[])

    def test_set_attr(self):
        # 定义 MyModule 类，继承自 torch.nn.Module
        class MyModule(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个卷积层 conv，输入通道数为 3，输出通道数为 10，卷积核大小为 2
                self.conv = torch.nn.Conv1d(3, 10, 2)
                # 设置布尔类型属性 b 初始值为 False
                self.b = False

            # 前向传播方法，接受 box_regression 和 weight 参数
            def forward(self, box_regression, weight):
                # 修改布尔属性 b 为 True
                self.b = True
                # 将卷积层的权重设置为 weight
                self.conv.weight = weight
                # 对卷积层的权重进行 softmax 操作
                w = torch.softmax(self.conv.weight, dim=0)
                # 将卷积层的权重设置为自身权重的和
                self.conv.weight = w + w
                # 如果属性 b 为 True，则返回 box_regression 加上卷积层权重的结果，否则返回差值
                if self.b:
                    return box_regression + self.conv.weight
                else:
                    return box_regression - self.conv.weight

        # 使用 torch.jit.script 方法将 MyModule 实例化为脚本模块 model
        model = torch.jit.script(MyModule())
        # 创建一个形状为 (3, 2) 的全 1 张量 weight
        weight = torch.ones(3, 2)
        # 创建一个形状为 (3, 2) 的随机张量 box_regression
        box_regression = torch.randn(3, 2)
        # 运行测试方法 run_test，传入脚本模块 model，输入为 box_regression 和 weight
        self.run_test(model, (box_regression, weight))

    # 使用装饰器 skipIfUnsupportedMinOpsetVersion(11) 跳过不支持 Opset 版本小于 11 的测试用例
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_set_attr_3(self):
        # 定义一个继承自torch.nn.Module的子类MyModule
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个1维卷积层，输入通道为10，输出通道为3，卷积核大小为3
                self.conv = torch.nn.Conv1d(10, 3, 3)
                # 设置卷积层的权重为3x10的零张量，并转换为参数
                self.conv.weight = torch.nn.Parameter(torch.zeros(3, 10))
                # 设置卷积层的偏置为3x10x3的零张量，并转换为参数
                self.conv.bias = torch.nn.Parameter(torch.zeros(3, 10, 3))

            # 定义一个设置单元锚点的方法，接受anchors和boxes两个参数
            def set_cell_anchors(self, anchors, boxes):
                # 将卷积层的权重设置为3x10的全1张量
                self.conv.weight = torch.ones(3, 10)
                # 如果卷积层的偏置不为None
                if self.conv.bias is not None:
                    # 将卷积层的偏置设置为3x10x3的标准正态分布随机张量
                    self.conv.bias = torch.randn(3, 10, 3)
                    # 将卷积层的权重更新为anchors与当前权重的和
                    self.conv.weight = anchors + self.conv.weight
                    # 将boxes张量的内容设置为2x3的全零张量
                    boxes[:] = torch.zeros(2, 3)

            # 定义前向传播方法，接受anchors参数，返回类型为元组的张量
            def forward(self, anchors) -> Tuple[Tensor, Tensor]:
                # 创建一个全1的2x2x3张量boxes
                boxes = torch.ones(2, 2, 3)
                # 调用设置单元锚点的方法，传入anchors和boxes作为参数
                self.set_cell_anchors(anchors, boxes)
                # 如果卷积层的偏置不为None
                if self.conv.bias is not None:
                    # 返回卷积层的权重和boxes张量
                    return self.conv.weight, boxes
                # 否则返回anchors和boxes张量
                return anchors, boxes

        # 使用torch.jit.script方法将MyModule类实例化为模型
        model = torch.jit.script(MyModule())
        # 创建一个大小为3x10的随机张量anchors
        anchors = torch.rand(3, 10)
        # 调用run_test方法，传入模型和anchors作为参数进行测试
        self.run_test(model, (anchors))
    def test_set_attr_5(self):
        # 定义一个名为 MyModule 的自定义 Torch 模块
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个 Conv1d 层，输入通道数为 10，输出通道数为 3，卷积核大小为 3
                self.conv = torch.nn.Conv1d(10, 3, 3)
                # 初始化 Conv1d 层的偏置为一个 3x10x3 的 Parameter 张量，值全为零
                self.conv.bias = torch.nn.Parameter(torch.zeros(3, 10, 3))

            # 设置单元的锚点
            def set_cell_anchors(self, anchors):
                # 设置 Conv1d 层的权重为一个从 0 到 9 的张量
                self.conv.weight = torch.arange(10)
                # 循环处理权重张量中的每个元素
                for i in range(10):
                    # 如果当前索引 i 等于 3
                    if i == 3:
                        # 再次循环处理权重张量中的每个元素
                        for j in range(10):
                            # 获取 Conv1d 层的权重
                            w = self.conv.weight
                            # 更新 Conv1d 层的权重，加上一个从 0 到 9 的张量
                            self.conv.weight = torch.arange(10) + w

                    # 更新 Conv1d 层的权重，加上一个从 0 到 9 的张量
                    self.conv.weight = self.conv.weight + torch.arange(10)
                    # NOTE: `is not None` and `assert` is for passing torchscript.
                    # 如果 Conv1d 层的偏置不为空
                    if self.conv.bias is not None:
                        # 获取 Conv1d 层的偏置
                        a = self.conv.bias
                        # 断言确保偏置 a 不为空
                        assert a is not None
                        # 更新 Conv1d 层的偏置，加上给定的锚点张量和偏置张量 a
                        self.conv.bias = anchors + a

            # 前向传播函数，接收锚点作为输入
            def forward(self, anchors):
                # 调用 set_cell_anchors 方法设置单元的锚点
                self.set_cell_anchors(anchors)
                # 返回 Conv1d 层的权重和偏置
                return self.conv.weight, self.conv.bias

        # 使用 Torch 的脚本化功能对 MyModule 模块进行脚本化
        model = torch.jit.script(MyModule())
        # 初始化一个大小为 (5, 11, 30) 的张量 x
        x = torch.rand(5, 11, 30)
        # 初始化一个大小为 (3, 10, 3) 的张量 anchors，所有元素值为 1
        anchors = torch.ones(3, 10, 3)
        # 运行测试函数 run_test，测试脚本化后的模块 model，输入参数为 x 和 anchors
        self.run_test(model, (anchors))
    def test_set_attr_in_loop_with_list(self):
        # 定义一个继承自 torch.nn.Module 的自定义模块 MyModule
        class MyModule(torch.nn.Module):
            # 初始化函数
            def __init__(self):
                super().__init__()
                # 创建一个 1 维卷积层，输入通道数为 10，输出通道数为 3，卷积核大小为 3
                self.conv = torch.nn.Conv1d(10, 3, 3)
                # 初始化卷积核权重为 3x10 的全零张量，并转换为可学习参数
                self.conv.weight = torch.nn.Parameter(torch.zeros(3, 10))
                # 初始化卷积层的偏置为 3x10x3 的全零张量，并转换为可学习参数
                self.conv.bias = torch.nn.Parameter(torch.zeros(3, 10, 3))
                # 定义一个列表 boxes 作为 TorchScript 的占位符
                self.boxes: List[Tensor] = [
                    torch.ones(1)
                ]  # Workaround placeholder for TorchScript

            # 设置细胞锚点函数，接受锚点 anchors 作为参数
            def set_cell_anchors(self, anchors):
                # 随机初始化卷积核权重为 3x10 的张量
                self.conv.weight = torch.randn(3, 10)
                # 遍历卷积核权重的第一维度
                for i in range(self.conv.weight.size(0)):
                    # 遍历 10 个节点
                    for j in range(10):
                        # 随机初始化卷积层的偏置为 3x10x3 的张量
                        self.conv.bias = torch.randn(3, 10, 3)
                        # 更新卷积核权重为 anchors 乘以当前索引 i 的结果
                        self.conv.weight = anchors * i
                        # 向列表 boxes 中添加一个 3x3 的全一张量
                        self.boxes.append(torch.ones(3, 3))

            # 前向传播函数，接受锚点 anchors 作为参数，返回权重和列表 boxes
            def forward(self, anchors) -> Tuple[Tensor, List[Tensor]]:
                # 清空列表 boxes
                self.boxes = []
                # 调用 set_cell_anchors 函数设置细胞锚点
                self.set_cell_anchors(anchors)
                # 如果卷积层的偏置不为空
                if self.conv.bias is not None:
                    # 返回卷积核权重和列表 boxes
                    return self.conv.weight, self.boxes
                # 否则返回锚点和列表 boxes
                return anchors, self.boxes

        # 使用 torch.jit.script 将 MyModule 脚本化
        model = torch.jit.script(MyModule())
        # 随机生成长度为 10 的锚点
        anchors = torch.rand(10)
        # 运行测试函数 run_test，传入模型和锚点
        self.run_test(model, anchors)
    # 定义一个测试方法，用于测试索引、修改和条件赋值
    def test_index_put_if(self):
        # 使用 Torch Script 注解装饰器，将函数 check_init 转换为 Torch 脚本
        @torch.jit.script
        def check_init(
            input_data: Tensor, hidden_size: int, prev_state: Tensor
        ) -> Tuple[Tensor, Tensor]:
            # 获取输入数据的批量大小
            batch_size = input_data.size(0)
            # 获取输入数据的第二和第三维度大小，作为空间尺寸
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # 如果未提供 prev_state，则生成一个空的 prev_state
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            state_copy = torch.zeros(state_size, device=input_data.device)
            if prev_state.size(0) == 0:
                # 如果 prev_state 的大小为 0，则使用全零张量填充 state
                state[:] = (
                    torch.zeros(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    + state[:]
                )
                # 使用全一张量乘以 2 填充 state_copy
                state_copy[:] = (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 2
                )
                # 使用全零张量乘以 2 填充 state_copy
                state_copy[:] = (
                    torch.zeros(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 2
                )
            else:
                # 如果 prev_state 的大小不为 0，则使用全一张量乘以 4 填充 state
                state[:] = (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 4
                )
            # 返回初始化后的 state 和 state_copy
            return state, state_copy

        # 定义一个继承自 torch.nn.Module 的示例类 Example
        class Example(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            # 实现 forward 方法，接受 input_data 和 prev_state 作为输入
            def forward(self, input_data, prev_state):
                # 调用 check_init 函数初始化 prev_state
                prev_state = check_init(input_data, self.hidden_size, prev_state)
                # 返回初始化后的 prev_state 的第一个和第二个元素
                return prev_state[0], prev_state[1]

        # 创建 Example 类的一个实例 model，设置 hidden_size 为 10
        model = Example(10)
        # 创建一个随机数据张量 random_data，形状为 (1, 5, 30, 30)
        random_data = torch.rand((1, 5, 30, 30))
        # 创建一个空的张量 empty_tensor，数据类型为 float，形状为空
        empty_tensor = torch.tensor([], dtype=torch.float).view(0, 0, 0, 0, 0)
        # 调用 self.run_test 方法，测试 model
        self.run_test(
            model,
            (random_data, empty_tensor),
            input_names=["random_data", "empty_tensor"],
            dynamic_axes={"random_data": [0, 1, 2, 3], "empty_tensor": [0, 1, 2, 3, 4]},
        )
        # 再次调用 self.run_test 方法，测试 model，传入 remained_onnx_input_idx=[]
        self.run_test(model, (random_data, empty_tensor), remained_onnx_input_idx=[])

    # 使用 skipIfUnsupportedMinOpsetVersion 装饰器，条件是 Opset 版本最低支持到 11
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_if_2(self):
        @torch.jit.script
        def check_init(
            input_data: Tensor, hidden_size: int, prev_state: Tensor
        ) -> Tuple[Tensor, Tensor]:
            batch_size = input_data.size(0)  # 获取输入数据的批次大小
            spatial_size_0 = input_data.size(2)  # 获取输入数据的空间尺寸维度0大小
            spatial_size_1 = input_data.size(3)  # 获取输入数据的空间尺寸维度1大小
            
            # 如果未提供 prev_state，则生成空的 prev_state
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)  # 创建全零的状态张量
            state_copy = torch.zeros(state_size, device=input_data.device)  # 创建全零的状态拷贝张量
            
            if prev_state.size(0) == 0:
                # 如果 prev_state 的第一个维度大小为 0，则初始化状态和状态拷贝
                for i in range(2):
                    state[:] = (
                        torch.ones(
                            batch_size, hidden_size, spatial_size_0, spatial_size_1
                        )
                        * i
                    )  # 将状态张量设置为全一乘以 i
                    state_copy[:] = (
                        torch.ones(
                            batch_size, hidden_size, spatial_size_0, spatial_size_1
                        )
                        * i
                    )  # 将状态拷贝张量设置为全一乘以 i
            elif prev_state.size(0) == 1:
                # 如果 prev_state 的第一个维度大小为 1，则将状态张量更新为 prev_state 加上当前状态
                s = state[:]
                state[:] = prev_state + s
            elif prev_state.size(0) == 2:
                # 如果 prev_state 的第一个维度大小为 2，则将状态张量设置为全四
                state[:] = (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 4
                )
            
            return state, state_copy  # 返回状态张量和状态拷贝张量

        class Example(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data, prev_state):
                prev_state = check_init(input_data, self.hidden_size, prev_state)  # 调用 check_init 初始化或更新 prev_state
                return prev_state[0], prev_state[1]  # 返回初始化或更新后的状态张量和状态拷贝张量的元组

        model = Example(10)  # 创建 Example 类的实例
        random_data = torch.rand((1, 5, 30, 30))  # 创建随机数据张量
        empty_tensor = torch.tensor([], dtype=torch.float).view(0, 0, 0, 0, 0)  # 创建空张量
        random_state = torch.rand((1, 1, 10, 30, 30))  # 创建随机状态张量

        self.run_test(
            model,
            (random_data, empty_tensor),
            input_names=["data", "state"],  # 设置输入张量的名称
            dynamic_axes={"data": [0, 1, 2], "state": [0, 1, 2, 3, 4]},  # 设置动态轴
            additional_test_inputs=[(random_data, random_state)],  # 添加额外的测试输入
        )
        self.run_test(
            model,
            (random_data, empty_tensor),
            input_names=["data", "state"],  # 设置输入张量的名称
            dynamic_axes={"state": [0, 1, 2, 3, 4]},  # 设置动态轴
            additional_test_inputs=[(random_data, random_state)],  # 添加额外的测试输入
            remained_onnx_input_idx=[1],  # 设置保留在 ONNX 输入中的索引
        )
        self.run_test(model, (random_data, empty_tensor), remained_onnx_input_idx=[])  # 运行测试，不保留任何 ONNX 输入

    @skipIfUnsupportedMinOpsetVersion(11)  # 如果不支持的最小 Opset 版本高于 11，则跳过测试
    def test_index_put_if_3(self):
        @torch.jit.script
        # 定义一个 TorchScript 函数，用于检查初始化并返回状态张量
        def check_init(
            input_data: Tensor, hidden_size: int, prev_state: Tensor
        ) -> Tensor:
            # 获取输入数据的批量大小
            batch_size = input_data.size(0)
            # 获取输入数据的空间维度大小
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # 如果未提供 prev_state，则生成一个空的状态张量
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            if prev_state.size(0) < 2:
                # 如果 prev_state 的大小小于 2，则对状态张量进行填充
                state = state * 3
                if prev_state.size(0) == 0:
                    # 如果 prev_state 的大小为 0，则将状态张量设置为全 3 的张量
                    state[:] = (
                        torch.ones(
                            batch_size, hidden_size, spatial_size_0, spatial_size_1
                        )
                        * 3
                    )
                else:
                    # 否则，对状态张量进行加法操作
                    state = state + 2

            return state

        # 定义一个示例模型类 Example，继承自 torch.nn.Module
        class Example(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            # 定义模型的前向传播方法
            def forward(self, input_data, prev_state):
                prev_state = check_init(input_data, self.hidden_size, prev_state)
                return prev_state

        # 创建 Example 类的实例 model，hidden_size 为 4
        model = Example(4)
        # 创建随机数据 random_data，形状为 (1, 5, 4, 4)
        random_data = torch.rand((1, 5, 4, 4))
        # 创建空张量 empty_tensor，形状为 (0, 0, 0, 0, 0)
        empty_tensor = torch.tensor([], dtype=torch.float).view(0, 0, 0, 0, 0)
        # 运行测试函数 run_test，传入模型、数据和相关参数
        self.run_test(
            model,
            (random_data, empty_tensor),
            input_names=["random_data", "empty_tensor"],
            dynamic_axes={"random_data": [0, 1, 2, 3], "empty_tensor": [0, 1, 2, 3, 4]},
        )
        # 再次运行测试函数 run_test，传入模型、数据和空列表
        self.run_test(model, (random_data, empty_tensor), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_if_4(self):
        # 定义一个使用 Torch 脚本的函数，用于检查初始化状态
        @torch.jit.script
        def check_init(
            input_data: Tensor, hidden_size: int, prev_state: Tensor
        ) -> Tensor:
            # 获取输入数据的批次大小、空间大小
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # 如果没有提供 prev_state，则生成一个空的 prev_state
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            if prev_state.size(0) == 0:
                # 对 state 进行初始化
                state = state + 3
                state[:] = (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 3
                )
                state = state + 3
                state[:] = (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 4
                )
            else:
                # 如果已有 prev_state，则对 state 进行加法操作
                state = state + 2
            return state

        # 定义一个示例的 PyTorch 模块
        class Example(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            # 前向传播方法
            def forward(self, input_data, prev_state):
                prev_state = check_init(input_data, self.hidden_size, prev_state)
                return prev_state

        # 创建 Example 类的一个实例
        model = Example(4)
        # 创建一个随机数据张量
        random_data = torch.rand((1, 5, 4, 4))
        # 创建一个空的张量
        empty_tensor = torch.tensor([], dtype=torch.float).view(0, 0, 0, 0, 0)
        # 运行测试方法，测试模型的前向传播
        self.run_test(
            model,
            (random_data, empty_tensor),
            input_names=["random_data", "empty_tensor"],
            dynamic_axes={"random_data": [0, 1, 2, 3], "empty_tensor": [0, 1, 2, 3, 4]},
        )
        # 再次运行测试方法，用于特定的 ONNX 输入索引
        self.run_test(model, (random_data, empty_tensor), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipIfUnsupportedMinOpsetVersion(11)
    # 使用装饰器，检查当前的运行环境是否支持最低 Opset 版本为 11，如果不支持则跳过测试
    def test_list_append_in_block(self):
        # 定义一个继承自 torch.nn.Module 的类 ListModel
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                # 初始化一个空列表 res
                res = []
                # 遍历 x 的第一维，范围是 x.size(0)
                for i in range(x.size(0)):
                    # 对每个 i，计算 torch.matmul(x[i], y) 并将结果添加到 res 中
                    res.append(torch.matmul(x[i], y))
                # 返回列表 res
                return res

        # 使用 torch.jit.script 将 ListModel 类型的实例转换为 TorchScript
        model = torch.jit.script(ListModel())
        # 创建输入张量 x 和 y，形状分别为 (16, 3, 4) 和 (4, 5)，内容为随机数
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        # 运行测试，验证模型在输入 x 和 y 上的输出
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    # 使用装饰器，检查当前的运行环境是否支持最低 Opset 版本为 13，如果不支持则跳过测试
    def test_list_append_in_nested_block(self):
        # 定义一个继承自 torch.nn.Module 的类 ListModel
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                # 初始化一个空列表 res
                res = []
                # 双重循环遍历 x 的前两维，范围分别是 x.size(0) 和 x.size(1)
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        # 对每个 (i, j)，计算 torch.matmul(x[i][j], y) 并将结果添加到 res 中
                        res.append(torch.matmul(x[i][j], y))
                # 返回列表 res
                return res

        # 使用 torch.jit.script 将 ListModel 类型的实例转换为 TorchScript
        model = torch.jit.script(ListModel())
        # 创建输入张量 x 和 y，形状分别为 (4, 4, 3, 4) 和 (4, 5)，内容为随机数
        x = torch.randn(4, 4, 3, 4)
        y = torch.randn(4, 5)
        # 运行测试，验证模型在输入 x 和 y 上的输出
        self.run_test(model, (x, y))
    def test_list_pop_in_block(self):
        # 定义一个继承自 torch.nn.Module 的列表模型类 ListModel
        class ListModel(torch.nn.Module):
            # 定义模型的前向传播函数，接受输入参数 x 和 y
            def forward(self, x, y):
                # 初始化一个空列表 res
                res = []
                # 计算 x[0] 和 y 的矩阵乘积，并赋值给 elem
                elem = torch.matmul(x[0], y)
                # 遍历 x 的第一维，将每个元素与 y 的矩阵乘积结果追加到 res 中
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                # 再次遍历 x 的第一维，依次弹出 res 中的元素赋值给 elem
                for i in range(x.size(0)):
                    elem = res.pop()
                # 再次遍历 x 的第一维，将每个元素与 y 的矩阵乘积结果追加到 res 中
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                    # 弹出 res 中的元素赋值给 elem
                    elem = res.pop()
                # 将 elem 添加到 res 的末尾，但由于 append 方法返回 None，因此此处没有实际返回
                return res.append(elem)

        # 使用 torch.jit.script 方法将 ListModel 类转换为脚本化模型
        model = torch.jit.script(ListModel())
        # 生成随机张量 x 和 y，分别形状为 (16, 3, 4) 和 (4, 5)
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        # 调用 self.run_test 方法运行模型测试
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_del_in_block(self):
        # 定义一个继承自 torch.nn.Module 的列表模型类 ListModel
        class ListModel(torch.nn.Module):
            # 定义模型的前向传播函数，接受输入参数 x 和 y
            def forward(self, x, y):
                # 初始化一个空列表 res
                res = []
                # 计算 x[0] 和 y 的矩阵乘积，并赋值给 elem
                elem = torch.matmul(x[0], y)
                # 遍历 x 的第一维，将每个元素与 y 的矩阵乘积结果追加到 res 中
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                # 再次遍历 x 的第一维，删除 res 中的第一个元素
                for i in range(x.size(0)):
                    del res[0]
                # 再次遍历 x 的第一维，将每个元素与 y 的矩阵乘积结果追加到 res 中
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                    # 删除 res 中的第一个元素
                    del res[0]
                # 将 elem 添加到 res 的末尾，但由于 append 方法返回 None，因此此处没有实际返回
                return res.append(elem)

        # 使用 torch.jit.script 方法将 ListModel 类转换为脚本化模型
        model = torch.jit.script(ListModel())
        # 生成随机张量 x 和 y，分别形状为 (16, 3, 4) 和 (4, 5)
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        # 调用 self.run_test 方法运行模型测试
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_unpack(self):
        # 定义一个继承自 torch.nn.Module 的列表模型类 ListModel
        class ListModel(torch.nn.Module):
            # 定义模型的前向传播函数，接受输入参数 x 和 y
            def forward(self, x, y):
                # 初始化一个空列表 res
                res = []
                # 计算 x[0] 和 y 的矩阵乘积，并赋值给 elem
                elem = torch.matmul(x[0], y)
                # 遍历 x 的第一维，将每个元素与 y 的矩阵乘积结果追加到 res 中
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                # 将 res 中的元素拆分为 a, b, c 三个变量
                a, b, c = res
                # 返回 a 和 b，c 没有使用到
                return a, b

        # 使用 torch.jit.script 方法将 ListModel 类转换为脚本化模型
        model = torch.jit.script(ListModel())
        # 生成随机张量 x 和 y，分别形状为 (3, 3, 4) 和 (4, 5)
        x = torch.randn(3, 3, 4)
        y = torch.randn(4, 5)
        # 调用 self.run_test 方法运行模型测试
        self.run_test(model, (x, y))
    # 定义一个测试函数，用于测试索引放置操作
    def test_index_put_inplace_ops(self):
        # 使用 TorchScript 装饰器将函数编译为 TorchScript
        @torch.jit.script
        def check_init(input_data: Tensor, hidden_size: int) -> Tensor:
            # 获取输入数据的批量大小
            batch_size = input_data.size(0)
            # 获取输入数据的空间大小
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # 如果未提供 prev_state，则生成空的 prev_state
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            # 如果输入数据的批量大小为1
            if input_data.size(0) == 1:
                # 对 state[1] 进行操作
                state[1] += (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 2
                )
                state[1] /= (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 3
                )
            # 遍历输入数据的批量大小
            for i in range(input_data.size(0)):
                state[1] += torch.ones(
                    batch_size, hidden_size, spatial_size_0, spatial_size_1
                )
                state[1] /= (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * i
                )
            return state

        # 定义一个示例类
        class Example(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data):
                # 调用 check_init 函数初始化 state
                state = check_init(input_data, self.hidden_size)
                return state

        # 创建 Example 类的实例
        model = Example(10)
        # 生成随机数据
        random_data = torch.rand((1, 5, 30, 30))
        # 运行测试，设置输入名称和动态轴
        self.run_test(
            model,
            (random_data),
            input_names=["random_data"],
            dynamic_axes={"random_data": [0, 1, 2, 3]},
        )
        # 运行测试，设置输入索引为空
        self.run_test(model, (random_data), remained_onnx_input_idx=[])

    # 如果不支持最小的 Opset 版本为 11，则跳过测试
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_input_mask_model(self):
        # 定义一个测试函数，测试输入掩码模型的功能
        class InputMaskModel(torch.nn.Module):
            def __init__(self, output_size):
                super().__init__()
                # 初始化偏置参数为指定大小的空张量
                self.bias = torch.nn.Parameter(
                    torch.empty(output_size, dtype=torch.float)
                )
                # 将偏置参数初始化为零，无需梯度
                with torch.no_grad():
                    self.bias.zero_()

            def forward(self, model_input, y):
                # 创建输入掩码，将不符合条件的元素置为零
                input_mask = (model_input <= 0) | (model_input > 25)
                # 将不符合输入掩码条件的输出置为零
                y[input_mask, :] = 0.0
                # 输出结果为处理后的 y 加上偏置
                output = y + self.bias
                return output

        # 定义输出大小
        output_size = 4
        # 创建输入掩码模型实例
        m = InputMaskModel(output_size)
        # 创建模型输入张量 x
        x = torch.tensor([0, 4, 24, 25], dtype=torch.int64)
        # 创建模型输入张量 y
        y = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
            ],
            dtype=torch.float,
        )
        # 运行测试函数
        self.run_test(m, (x, y))

        # 定义另一个输入掩码模型
        class InputMaskModel(torch.nn.Module):
            def __init__(self, output_size):
                super().__init__()

            def forward(self, model_input_1, model_input_2, y):
                # 创建两个输入掩码，将不符合条件的元素置为零
                input_mask_1 = (model_input_1 <= 0) | (model_input_1 > 25)
                input_mask_2 = (model_input_2 < 1) | (model_input_2 >= 12)
                # 将不符合输入掩码条件的输出置为零
                y[input_mask_1, input_mask_2] = 0.0
                return y

        # 定义输出大小
        output_size = 4
        # 创建输入掩码模型实例
        m = InputMaskModel(output_size)
        # 创建模型输入张量 x1 和 x2
        x1 = torch.tensor([0, 4, 24, 25], dtype=torch.int64)
        x2 = torch.tensor([0, 3, 12, 15], dtype=torch.int64)
        # 创建模型输入张量 y
        y = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
            ],
            dtype=torch.float,
        )
        # 运行测试函数
        self.run_test(m, (x1, x2, y))

    @skipScriptTest()
    def test_unsafe_chunk(self):
        # 跳过此测试函数，不执行测试
        class ChunkModel(torch.nn.Module):
            def forward(self, x):
                # 使用 torch.unsafe_chunk 在维度 1 上对张量 x 进行分块操作
                return torch.unsafe_chunk(x, 3, dim=1)

        # 创建 ChunkModel 实例
        model = ChunkModel()
        # 将模型设置为评估模式
        model.eval()
        # 创建输入张量 x，形状为 (1, 18)
        x = torch.randn(1, 18)
        # 运行测试函数，对模型进行测试，输入名称为 "x"
        self.run_test(model, x, input_names=["x"])
    # 测试符号形状推断
    def test_symbolic_shape_inference(self):
        # 在 test_embedding_bag 中测试 ConstantOfShape
        # 在 test_repeat 中测试 Tile
        # 测试 Shape, Reshape, Transpose, Gather
        class ShapeModel(torch.nn.Module):
            def forward(self, x, y):
                # 获取 x 的前三个维度大小，最后一个维度设为 -1
                shape = x.size()[:3] + (-1,)  # shape [4], ("batch", 3, 4, -1)
                y = y.reshape(shape)  # batch, 3, 4, 10/batch
                return y.transpose(1, 2)

        model = ShapeModel()
        model.eval()
        x = torch.ones(2, 3, 4, 5)
        y = torch.ones(3, 4, 5, 2)
        # 运行测试
        self.run_test(
            model,
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2, 3], "y": [0, 1, 2, 3]},
        )
        self.run_test(model, (x, y), remained_onnx_input_idx=[1])

        class ViewModel(torch.nn.Module):
            def forward(self, x):
                return x.view(-1)

        model = ViewModel()
        model.eval()
        x = torch.tensor(2.0)
        self.run_test(model, (x,))

        # 测试 prim::ListConstruct 用于 Reshape 输入 1
        class ViewModel_2(torch.nn.Module):
            def forward(self, x):
                N, C, H, W = x.shape[0], x.shape[2], x.shape[3], x.shape[4]
                x1 = x.view(N, -1, C, H, W)
                x2 = x1.permute(0, 3, 4, 1, 2)
                return x2.reshape(N, -1, C)

        model = ViewModel_2()
        model.eval()
        x = torch.ones(2, 3, 4, 5, 6)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_symbolic_shape_inference_arange(self):
        # 测试 Range
        class ArangeModel(torch.nn.Module):
            def forward(self, signal):
                frame_step = 2
                outer_dimensions = signal.size()[:-2]
                frames, frame_length = signal.size()[-2:]

                subframe_length = signal.size()[0]
                subframe_step = frame_step // subframe_length
                subframes_per_frame = frame_length // subframe_length
                output_size = frame_step * (frames - 1) + frame_length
                output_subframes = output_size // subframe_length

                frame = torch.arange(0, output_subframes)
                return frame

        model = ArangeModel()
        model.eval()
        M, C, K, N = 1, 2, 3, 4
        x = torch.randint(5, (M, C, K, N))
        y = torch.randint(5, (M, C + 1, K + 1, N + 1))
        self.run_test(model, x, input_names=["x"], dynamic_axes={"x": [0, 1, 2, 3]})
        self.run_test(model, x, remained_onnx_input_idx=[])
        self.run_test(
            model,
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
            additional_test_inputs=[(x,), (y,)],
        )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_symbolic_shape_inference_box(self):
        # test NonZero
        # 定义一个名为BoxModel的torch模块
        class BoxModel(torch.nn.Module):
            # 定义模块的前向传播方法
            def forward(self, boxes):
                # 定义最小尺寸阈值
                min_size = 1e-2
                # 计算每个框的宽度和高度
                ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
                # 筛选出宽度和高度均大于等于最小尺寸的框的索引
                keep = (ws >= min_size) & (hs >= min_size)
                # 获取符合条件的框的索引
                keep = torch.where(keep)[0]
                # 返回符合条件的框的索引
                return keep

        # 创建BoxModel的实例
        model = BoxModel()
        # 设置模型为评估模式
        model.eval()
        # 创建一个形状为(2, 4)的张量
        x = torch.ones(2, 4)
        # 创建一个形状为(3, 5)的张量
        y = torch.ones(3, 5)
        # 运行测试函数，传入模型和输入张量x
        self.run_test(model, x)
        # 运行测试函数，传入模型、输入张量x，以及其他参数和输入
        self.run_test(
            model,
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
            additional_test_inputs=[(x,), (y,)],
        )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_symbolic_shape_inference_box_if(self):
        # test If
        # 定义一个名为BoxIfModel的torch模块
        class BoxIfModel(torch.nn.Module):
            # 定义模块的前向传播方法
            def forward(self, boxes, scores):
                # 定义分数阈值
                score_thresh = 0.0
                # 筛选出分数大于阈值的框的索引
                inds = torch.where(scores > score_thresh)[0]
                # 根据索引获取符合条件的框
                boxes_1 = boxes[inds]
                # 如果符合条件的框的元素数量大于3，返回这些框，否则返回这些框的两倍
                if boxes_1.numel() > 3:
                    return boxes_1
                else:
                    return boxes_1 * 2

        # 创建BoxIfModel的实例
        model = BoxIfModel()
        # 设置模型为评估模式
        model.eval()
        # 创建形状为(2, 4)的框张量和形状为(1, 4)的分数张量
        boxes = torch.ones(2, 4)
        scores = torch.ones(1, 4)
        # 运行测试函数，传入模型和输入张量组(boxes, scores)
        self.run_test(model, (boxes, scores))

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipDtypeChecking
    def test_symbolic_shape_inference_arange_2(self):
        # test Range
        # 定义一个名为ArangeModel的torch模块
        class ArangeModel(torch.nn.Module):
            # 定义模块的前向传播方法
            def forward(self, start):
                # 返回一个torch.arange对象，从start.size(0)开始到8.5结束，步长为1.5，数据类型为torch.int64
                return torch.arange(start.size(0), 8.5, 1.5, dtype=torch.int64)

        # 创建形状为(2, 3, 4)的随机张量x
        x = torch.randn(2, 3, 4)
        # 运行测试函数，传入ArangeModel的实例和输入张量x，设置输入张量x的动态轴
        self.run_test(
            ArangeModel(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        # 运行测试函数，传入ArangeModel的实例和输入张量x，设置不保留ONNX输入索引
        self.run_test(ArangeModel(), (x,), remained_onnx_input_idx=[])

        # 定义一个名为ArangeModel2的torch模块
        class ArangeModel2(torch.nn.Module):
            # 定义模块的前向传播方法
            def forward(self, start):
                # 返回一个torch.arange对象，从start.size(0)开始到8.5结束，步长为1.5，数据类型为torch.double
                return torch.arange(start.size(0), 8.5, 1.5, dtype=torch.double)

        # 创建形状为(2, 3, 4)的随机张量x
        x = torch.randn(2, 3, 4)
        # 运行测试函数，传入ArangeModel2的实例和输入张量x，设置输入张量x的动态轴
        self.run_test(
            ArangeModel2(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        # 运行测试函数，传入ArangeModel2的实例和输入张量x，设置不保留ONNX输入索引
        self.run_test(ArangeModel2(), (x,), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_symbolic_shape_inference_nonzero(self):
        # 定义一个继承自 torch.nn.Module 的内部类 OneLikeModel
        class OneLikeModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 使用 torch.ones_like 创建一个与输入 x 形状相同的全为 1 的张量
                ones = torch.ones_like(
                    x,
                    dtype=torch.float,
                    layout=torch.strided,
                    device=torch.device("cpu"),
                )
                # 返回 ones 张量中非零元素的索引
                return torch.nonzero(ones)

        # 创建一个形状为 (2,) 的随机张量 x
        x = torch.randn(2)
        # 运行测试，输入 OneLikeModel 实例和张量 x，指定输入名为 "x"，动态维度轴为 {"x": [0]}
        self.run_test(OneLikeModel(), x, input_names=["x"], dynamic_axes={"x": [0]})
        # 再次运行测试，输入 OneLikeModel 实例和张量 x，不保留 ONNX 输入索引
        self.run_test(OneLikeModel(), x, remained_onnx_input_idx=[])
        
        # 创建一个形状为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 运行测试，输入 OneLikeModel 实例和张量 x，指定输入名为 "x"，动态维度轴为 {"x": [0, 1, 2]}
        self.run_test(
            OneLikeModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        # 再次运行测试，输入 OneLikeModel 实例和张量 x，不保留 ONNX 输入索引
        self.run_test(OneLikeModel(), x, remained_onnx_input_idx=[])

        # 定义一个继承自 torch.nn.Module 的内部类 ZeroLikeModel
        class ZeroLikeModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 使用 torch.zeros_like 创建一个与输入 x 形状相同的全为 0 的张量
                zeros = torch.zeros_like(
                    x,
                    dtype=torch.float,
                    layout=torch.strided,
                    device=torch.device("cpu"),
                )
                # 返回 zeros 张量中非零元素的索引
                return torch.nonzero(zeros)

        # 创建一个形状为 (2,) 的随机张量 x
        x = torch.randn(2)
        # 运行测试，输入 ZeroLikeModel 实例和张量 x，指定输入名为 "x"，动态维度轴为 {"x": [0]}
        self.run_test(ZeroLikeModel(), x, input_names=["x"], dynamic_axes={"x": [0]})
        # 再次运行测试，输入 ZeroLikeModel 实例和张量 x，不保留 ONNX 输入索引
        self.run_test(ZeroLikeModel(), x, remained_onnx_input_idx=[])
        
        # 创建一个形状为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 运行测试，输入 ZeroLikeModel 实例和张量 x，指定输入名为 "x"，动态维度轴为 {"x": [0, 1, 2]}
        self.run_test(
            ZeroLikeModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        # 再次运行测试，输入 ZeroLikeModel 实例和张量 x，不保留 ONNX 输入索引
        self.run_test(ZeroLikeModel(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_symbolic_shape_inference_expand_1(self):
        # 定义一个继承自 torch.nn.Module 的内部类 ExpandModel
        class ExpandModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 使用 x.expand(4, 6, 2) 对输入张量 x 进行扩展操作
                return x.expand(4, 6, 2)

        # 创建一个形状为 (6, 1) 的随机张量 x，并指定其需要梯度
        x = torch.randn(6, 1, requires_grad=True)
        # 运行测试，输入 ExpandModel 实例和张量 x
        self.run_test(ExpandModel(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_symbolic_shape_inference_expand_2(self):
        # 定义一个继承自 torch.nn.Module 的内部类 M
        class M(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 获取输入 x 的形状信息
                input_shape = x.size()
                # 分别获取批量大小和序列长度
                batch_size, seq_length = input_shape
                # 创建一个序列长度的张量 seq_ids
                seq_ids = torch.arange(seq_length)
                # 创建一个因果掩码 causal_mask
                causal_mask = (
                    # 使用广播将 seq_ids 扩展成与输入 x 相同的形状
                    seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                    # 判断 seq_ids 是否小于等于 seq_ids 的转置
                    <= seq_ids[None, :, None]
                )
                # 返回 causal_mask 的转置
                return causal_mask.transpose(0, 1)

        # 创建一个形状为 (3, 16) 的随机张量 x
        x = torch.randn(3, 16)
        # 运行测试，输入 M 实例和张量 x，指定输入名为 "x"，动态维度轴为 {"x": [0, 1]}
        self.run_test(M(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]})
        # 再次运行测试，输入 M 实例和张量 x，不保留 ONNX 输入索引
        self.run_test(M(), (x,), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_symbolic_shape_inference_slice(self):
        # 定义一个测试函数，用于测试符号推断中的切片操作

        class M(torch.nn.Module):
            def forward(self, x, position_bias):
                # 获取输入张量 x 的形状
                input_shape = x.size()
                # 解包输入形状，得到批量大小和序列长度
                batch_size, seq_length = input_shape
                # 对位置偏置进行切片操作，保留最后 seq_length 列
                position_bias = position_bias[:, :, -seq_length:, :]
                # 返回转置后的位置偏置张量
                return position_bias.transpose(0, 1)

        # 创建一个形状为 (3, 16) 的随机张量 x
        x = torch.randn(3, 16)
        # 创建一个形状为 (1, 3, 20, 8) 的随机张量 position_bias
        position_bias = torch.randn(1, 3, 20, 8)
        # 运行测试函数，测试模块 M 的前向传播函数
        self.run_test(
            M(),
            (x, position_bias),
            input_names=["x", "position_bias"],
            dynamic_axes={"x": [0, 1], "position_bias": [0, 1, 2, 3]},
        )
        # 再次运行测试函数，测试模块 M 的前向传播函数，保留输入索引为 1 的 ONNX 输入
        self.run_test(M(), (x, position_bias), remained_onnx_input_idx=[1])

    def test_symbolic_shape_inference_slice_2(self):
        # 定义一个测试函数，用于测试符号推断中的切片操作

        class M(torch.nn.Module):
            def forward(self, position_bias):
                # 对位置偏置进行切片操作，保留最后两列
                position_bias = position_bias[:, :, -2:, :]
                # 返回转置后的位置偏置张量
                return position_bias.transpose(0, 1)

        # 创建一个形状为 (1, 3, 20, 8) 的随机张量 position_bias
        position_bias = torch.randn(1, 3, 20, 8)
        # 运行测试函数，测试模块 M 的前向传播函数
        self.run_test(M(), (position_bias,))

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipScriptTest()
    def test_symbolic_shape_inference_time(self):
        # 定义一个测试函数，用于测试符号推断中的时间相关操作

        # 创建形状为 (RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE) 的随机输入张量 input
        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        # 创建形状为 (1, BATCH_SIZE, RNN_HIDDEN_SIZE) 的随机隐藏状态张量 h0
        h0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        # 创建形状为 (1, BATCH_SIZE, RNN_HIDDEN_SIZE) 的随机细胞状态张量 c0
        c0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)

        # 创建一个单向 LSTM 模型 model_lstm
        model_lstm = torch.nn.LSTM(
            RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False
        )
        # 运行测试函数，测试 LSTM 模型的前向传播函数
        self.run_test(
            model_lstm,
            (input, (h0, c0)),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1]},
        )

        # 创建一个单向 GRU 模型 model_gru
        model_gru = torch.nn.GRU(
            RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False, bias=False
        )
        # 运行测试函数，测试 GRU 模型的前向传播函数
        self.run_test(
            model_gru, (input, h0), input_names=["x", "y"], dynamic_axes={"x": [0, 1]}
        )

        # 创建一个单向 RNN 模型 model_rnn
        model_rnn = torch.nn.RNN(
            RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False, bias=False
        )
        # 运行测试函数，测试 RNN 模型的前向传播函数
        self.run_test(
            model_rnn, (input, h0), input_names=["x", "y"], dynamic_axes={"x": [0, 1]}
        )

    def test_symbolic_shape_inference_dynamic_axes(self):
        # 定义一个测试函数，用于测试符号推断中的动态轴定义

        class M(torch.nn.Module):
            def forward(self, input_ids):
                # 获取输入张量 input_ids 的形状
                input_shape = input_ids.size()
                # 重新调整 input_ids 的形状，将其视为二维张量
                input_ids = input_ids.view(-1, input_shape[-1])
                # 返回转置后的 input_ids 张量
                return input_ids.transpose(0, 1)

        # 创建一个形状为 (3, 16) 的随机张量 x
        x = torch.randn(3, 16)
        # 运行测试函数，测试模块 M 的前向传播函数
        self.run_test(
            M(),
            (x,),
            input_names=["input_ids"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}},
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_hann_window_periodic(self):
        # 定义一个名为 HannWindowModule_Periodic 的子类，继承自 torch.nn.Module
        class HannWindowModule_Periodic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.window_length = 0

            # 重写 forward 方法，计算带汉宁窗的输入张量 x
            def forward(self, x, window_length: int):
                # 设置窗口长度属性
                self.window_length = window_length
                # 返回带有周期性汉宁窗的张量 x
                return torch.add(
                    x,
                    torch.hann_window(
                        self.window_length, periodic=True, dtype=torch.float
                    ),
                )

        win_length = 100
        x = torch.randn(win_length)

        # 创建 HannWindowModule_Periodic 的实例
        module = HannWindowModule_Periodic()
        # 运行测试
        self.run_test(module, (x, win_length))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_hann_window_not_periodic(self):
        # 定义一个名为 HannWindowModule_NotPeriodic 的子类，继承自 torch.nn.Module
        class HannWindowModule_NotPeriodic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.window_length = 0

            # 重写 forward 方法，计算带汉宁窗的输入张量 x
            def forward(self, x, window_length: int):
                # 设置窗口长度属性
                self.window_length = window_length
                # 返回不带周期性汉宁窗的张量 x
                return torch.add(
                    x,
                    torch.hann_window(
                        self.window_length, periodic=False, dtype=torch.float
                    ),
                )

        win_length = 100
        x = torch.randn(win_length)

        # 创建 HannWindowModule_NotPeriodic 的实例
        module = HannWindowModule_NotPeriodic()
        # 运行测试
        self.run_test(module, (x, win_length))

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipScriptTest()
    def test_hann_window_default_values(self):
        # 定义一个名为 HannWindowModule 的子类，继承自 torch.nn.Module
        class HannWindowModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.window_length = 0

            # 重写 forward 方法，计算带汉宁窗的输入张量 x
            def forward(self, x, window_length: int):
                import torch.nn.functional as F

                # 设置窗口长度属性
                self.window_length = window_length
                # 返回带有汉宁窗的张量 x
                return torch.add(x, F.relu(torch.hann_window(self.window_length)))

        win_length = 100
        x = torch.randn(win_length, dtype=torch.float)
        module = HannWindowModule()

        # 计算模块输出
        output = module(x, win_length)
        # 运行测试
        self.run_test(module, (x, win_length))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_tensordot_dim_count(self):
        # 定义一个名为 M 的子类，继承自 torch.nn.Module
        class M(torch.nn.Module):
            def forward(self, x, y):
                # 执行张量点积，维度为 2
                output = torch.tensordot(x, y, 2)
                return output

        x = torch.randint(6, (7, 5, 3, 4))
        y = torch.randint(6, (3, 4, 9, 2))

        # 运行测试
        self.run_test(M(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_tensordot_dim_list(self):
        # 定义一个名为 M 的子类，继承自 torch.nn.Module
        class M(torch.nn.Module):
            def forward(self, x, y):
                # 执行张量点积，指定维度列表为 ([1, -2, -1], [1, 0, 3])
                output = torch.tensordot(x, y, ([1, -2, -1], [1, 0, 3]))
                return output

        x = torch.randint(6, (7, 4, 3, 5, 2))
        y = torch.randint(6, (5, 4, 4, 2, 6))

        # 运行测试
        self.run_test(M(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(12)
    # 定义一个测试方法，用于测试动态维度下的 torch.tensordot 函数
    def test_tensordot_dynamic_dim(self):
        # 定义一个简单的 PyTorch 模型类
        class M(torch.nn.Module):
            # 前向传播方法，接受输入 x 和 y，执行 torch.tensordot 操作
            def forward(self, x, y):
                output = torch.tensordot(x, y, 2)
                return output

        # 创建两个随机张量 x 和 y，形状分别为 (7, 5, 3, 4) 和 (3, 4, 9, 2)
        x = torch.randint(6, (7, 5, 3, 4))
        y = torch.randint(6, (3, 4, 9, 2))

        # 创建两个新的随机张量 new_x 和 new_y，用于额外的测试输入
        new_x = torch.randint(6, (8, 6, 2, 5))
        new_y = torch.randint(6, (2, 5, 3, 4))

        # 运行测试，传入模型 M，输入 x 和 y，以及额外的测试输入 new_x 和 new_y
        self.run_test(
            M(),
            (x, y),
            additional_test_inputs=[(new_x, new_y)],
            input_names=["input_x", "input_y"],
            dynamic_axes={"input_x": [0, 1, 2, 3], "input_y": [0, 1, 2, 3]},
        )

    # 使用装饰器 skipIfUnsupportedMinOpsetVersion(9)，标记以下测试方法为仅在 Opset 版本 >= 9 支持时才运行
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_to_device(self):
        # 定义一个简单的 PyTorch 模型类 M_ToDevice，将输入 x 移动到 y 所在的设备上
        class M_ToDevice(torch.nn.Module):
            def forward(self, x, y):
                return x.to(y.device), y

        # 定义另一个 PyTorch 模型类 M_ToDeviceDtype，将输入 x 移动到 y 所在的设备上，并将其数据类型转换为 torch.long
        class M_ToDeviceDtype(torch.nn.Module):
            def forward(self, x, y):
                return x.to(y.device, dtype=torch.long), y

        # 创建两个随机张量 x 和 y
        x = torch.randn(6)
        y = torch.randn(6)

        # 分别对 M_ToDevice 和 M_ToDeviceDtype 模型运行测试
        self.run_test(M_ToDevice(), (x, y))
        self.run_test(M_ToDeviceDtype(), (x, y))

    # 使用装饰器 skipIfUnsupportedMinOpsetVersion(9)，标记以下测试方法为仅在 Opset 版本 >= 9 支持时才运行
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_fill(self):
        # 定义一个填充指定值的 PyTorch 模型类 FillModule
        class FillModule(torch.nn.Module):
            def forward(self, x, filled_value: int):
                return x.fill_(filled_value)

        # 创建一个形状为 (4, 5, 6) 的随机张量 x，以及填充值为 7
        x = torch.randn((4, 5, 6))
        filled_value = 7
        # 运行 FillModule 模型的测试，填充输入张量 x 的所有元素为填充值 filled_value
        self.run_test(FillModule(), (x, filled_value))

        # 定义一个填充指定浮点数值的 PyTorch 模型类 FillFloatModule
        class FillFloatModule(torch.nn.Module):
            def forward(self, x, filled_value: float):
                return x.fill_(filled_value)

        # 创建一个形状为 (4, 5, 6) 的随机张量 x，以及填充值为 7.5
        x = torch.randn((4, 5, 6))
        filled_value = 7.5
        # 运行 FillFloatModule 模型的测试，填充输入张量 x 的所有元素为填充值 filled_value
        self.run_test(FillFloatModule(), (x, filled_value))

        # 定义一个只填充标量值的 PyTorch 模型类 FillScalarModule
        class FillScalarModule(torch.nn.Module):
            def forward(self, x):
                # 对输入张量 x 执行加法运算并填充结果为 2.5，然后返回填充后的结果和原始输入 x
                res = x + 2
                res.fill_(2.5)
                return res, x

        # 创建一个数据类型为 torch.long，形状为 (2, 3, 4) 的张量 x
        x = torch.ones(2, 3, 4, dtype=torch.long)
        # 运行 FillScalarModule 模型的测试，填充输入张量 x 的所有元素为标量值 2.5
        self.run_test(FillScalarModule(), x)
    def test_index_add_normal(self):
        # 定义一个简单的模块类 M，用于测试 index_add_ 方法
        class M(torch.nn.Module):
            def __init__(self, dim, index, updates):
                super().__init__()
                self.dim = dim
                self.index = index
                self.updates = updates

            def forward(self, x):
                # 在张量 x 的指定维度 dim 上使用 index 指定的索引位置，添加 updates 的值
                x.index_add_(self.dim, self.index, self.updates)
                return x

        # 创建一个形状为 (5, 1) 的张量 x，元素全为 1
        x = torch.ones(5, 1)
        # 更新数据 updates，形状为 (5, 1)
        updates = torch.tensor([[1], [4], [7], [3], [2]], dtype=torch.float)
        # 更新索引 index，形状为 (5,)
        index = torch.tensor([0, 2, 3, 1, 4])
        # 运行测试，使用定义的模块 M 进行测试
        self.run_test(M(0, index, updates), (x,))

        # 创建一个形状为 (1, 4, 3) 的张量 x，元素全为 1
        x = torch.ones(1, 4, 3)
        # 更新数据 updates，形状为 (1, 1, 4, 3)
        updates = torch.tensor(
            [[[1, 5, 7], [2, 4, 5], [5, 5, 6], [2, 3, 4]]], dtype=torch.float
        )
        # 更新索引 index，形状为 (4,)
        index = torch.tensor([0, 2, 3, 1])
        # 运行测试，使用定义的模块 M 进行测试
        self.run_test(M(1, index, updates), (x,))

        # 更新数据 updates，形状为 (1, 1, 4, 3)
        updates = torch.tensor(
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4]]], dtype=torch.float
        )
        # 更新索引 index，形状为 (3,)
        index = torch.tensor([0, 2, 1])
        # 运行测试，使用定义的模块 M 进行测试
        self.run_test(M(2, index, updates), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_add_dim_size_differ(self):
        # 定义一个带有维度大小不匹配测试的模块类 M
        class M(torch.nn.Module):
            def __init__(self, dim, index, updates):
                super().__init__()
                self.dim = dim
                self.index = index
                self.updates = updates

            def forward(self, x):
                # 在张量 x 的指定维度 dim 上使用 index 指定的索引位置，添加 updates 的值
                x.index_add_(self.dim, self.index, self.updates)
                return x

        # 创建一个形状为 (1, 4, 3) 的张量 x，元素全为 1
        x = torch.ones(1, 4, 3)
        # 更新数据 updates，形状为 (1, 1, 3)
        updates = torch.tensor([[[1, 5, 7], [2, 4, 5], [5, 5, 6]]], dtype=torch.float)
        # 更新索引 index，形状为 (3,)
        index = torch.tensor([0, 2, 1])
        # 运行测试，使用定义的模块 M 进行测试
        self.run_test(M(1, index, updates), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_add_in_loop(self):
        # 定义一个循环中使用 index_add_ 方法的模块类 M
        class M(torch.nn.Module):
            def __init__(self, dim, index, updates, loop_count):
                super().__init__()
                self.dim = dim
                self.index = index
                self.updates = updates
                self.loop_count = loop_count

            def forward(self, x):
                # 循环执行 self.loop_count 次，每次在张量 x 的指定维度 dim 上使用 index 指定的索引位置，添加 updates 的值
                for i in range(self.loop_count):
                    x.index_add_(self.dim, self.index, self.updates)
                return x

        # 创建一个形状为 (1, 4, 3) 的张量 x，元素全为 1
        x = torch.ones(1, 4, 3)
        # 更新数据 updates，形状为 (1, 1, 4, 3)
        updates = torch.tensor(
            [[[1, 5, 7], [2, 4, 5], [5, 5, 6], [2, 3, 4]]], dtype=torch.float
        )
        # 更新索引 index，形状为 (4,)
        index = torch.tensor([0, 2, 3, 1])
        # 随机生成一个循环次数 loop_count
        loop_count = torch.randint(20, (1,))[0].item()
        # 运行测试，使用定义的模块 M 进行测试
        self.run_test(M(1, index, updates, loop_count), (x,))
    # 定义一个测试方法，用于测试索引添加操作的行为
    def test_index_add_if(self):
        # 定义一个内部模块 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 初始化方法，接受维度 dim，更新数据 updates，以及真实索引 index_true 和假索引 index_false
            def __init__(self, dim, updates, index_true, index_false):
                super().__init__()
                self.dim = dim
                self.updates = updates
                self.index_true = index_true
                self.index_false = index_false

            # 前向传播方法，根据条件 cond 执行不同的索引添加操作
            def forward(self, x, cond):
                # 如果 cond 为真，使用 index_true 执行索引添加操作
                if cond:
                    x.index_add_(self.dim, self.index_true, self.updates)
                # 否则，使用 index_false 执行索引添加操作
                else:
                    x.index_add_(self.dim, self.index_false, self.updates)
                return x

        # 创建一个形状为 (1, 4, 3) 的全一张量 x
        x = torch.ones(1, 4, 3)
        # 创建一个更新数据的张量 updates
        updates = torch.tensor(
            [[[1, 5, 7], [2, 4, 5], [5, 5, 6], [2, 3, 4]]], dtype=torch.float
        )
        # 创建一个真实索引的张量 index_true
        index_true = torch.tensor([0, 2, 3, 1])
        # 创建一个假索引的张量 index_false
        index_false = torch.tensor([1, 0, 2, 3])
        # 创建一个条件张量 cond，作为条件传递给模块 M
        cond = torch.tensor(1, dtype=torch.bool)
        # 运行测试方法，使用 torch.jit.script 将模块 M 脚本化，并传入输入 x 和 cond
        self.run_test(
            torch.jit.script(M(1, updates, index_true, index_false)), (x, cond)
        )

    # 标记为仅在最小操作集版本大于等于 9 时才执行的测试方法
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_add_dynamic_axes(self):
        # 定义一个内部模块 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 初始化方法，接受维度 dim，索引 index，以及更新数据 updates
            def __init__(self, dim, index, updates):
                super().__init__()
                self.dim = dim
                self.index = index
                self.updates = updates

            # 前向传播方法，使用给定的索引 index 执行索引添加操作
            def forward(self, x):
                x.index_add_(self.dim, self.index, self.updates)
                return x

        # 创建一个形状为 (1, 4, 3) 的全一张量 x
        x = torch.ones(1, 4, 3)
        # 创建一个更新数据的张量 updates
        updates = torch.tensor(
            [[[1, 5, 7], [2, 4, 5], [5, 5, 6], [2, 3, 4]]], dtype=torch.float
        )
        # 创建一个索引的张量 index
        index = torch.tensor([0, 2, 3, 1])

        # 运行测试方法，创建模块 M 的实例并传入输入 x，同时指定输入名称和动态轴信息
        self.run_test(
            M(1, index, updates),
            (x,),
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1]},
        )

    # 定义一个测试方法，测试张量的滚动操作
    def test_roll(self):
        # 定义一个内部模块 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 初始化方法，接受滚动量 shifts 和维度 dims
            def __init__(self, shifts, dims):
                super().__init__()
                self.shifts = shifts
                self.dims = dims

            # 前向传播方法，对输入张量 x 执行滚动操作
            def forward(self, x):
                return torch.roll(x, self.shifts, self.dims)

        # 创建一个形状为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 分别运行滚动操作测试，传入不同的 shifts 和 dims 参数
        self.run_test(M([1, 1], [1, 0]), (x,))
        self.run_test(M([0, 1, 2], [1, 0, 2]), (x,))
        self.run_test(M(2, 1), (x,))
        self.run_test(M([-1, 3], [-2, -1]), (x,))

    # 定义一个测试方法，测试张量的求和操作
    def test_sum(self):
        # 定义一个内部模块 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 前向传播方法，对输入张量 x 执行求和操作
            def forward(self, x):
                return torch.sum(x)

        # 创建一个形状为 (12, 3) 的全一张量 x
        x = torch.ones(12, 3)
        # 运行测试方法，传入模块 M 的实例和输入 x，同时指定输入名称和动态轴信息
        self.run_test(M(), (x,), input_names=["x"], dynamic_axes={"x": [0]})

    # 标记为跳过形状检查的测试方法
    @skipShapeChecking
    def test_sum_empty_tensor(self):
        # 定义一个内部模块 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 前向传播方法，对输入张量 x 执行从索引 0 开始到索引 0 结束的切片求和，以及整体求和
            def forward(self, x):
                return x[0:0].sum(), x.sum()

        # 创建一个形状为 (12) 的全一张量 x，运行测试方法
        x = torch.ones(12)
        self.run_test(M(), (x,))

        # 创建一个形状为 (2, 0, 3) 的全一张量 x，运行测试方法
        x = torch.ones(2, 0, 3)
        self.run_test(M(), (x,))

        # 创建一个形状为 (0) 的全一张量 x，运行测试方法
        x = torch.ones(0)
        self.run_test(M(), (x,))

    # 标记为仅在最小操作集版本大于等于 11 时才执行的测试方法
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_broad_cast_tensors(self):
        # 定义一个测试函数，用于测试广播张量操作
        class M(torch.nn.Module):
            def forward(self, x, y):
                # 在 forward 方法中，使用 torch.broadcast_tensors 函数广播输入张量 x 和 y
                m = torch.broadcast_tensors(x, y)
                return m

        # 创建两个张量 x 和 y，形状分别为 (1,) 和 (5,)
        x = torch.randint(5, (1,))
        y = torch.randint(5, (5,))

        # 运行测试，调用 self.run_test 方法，传入模型 M 和张量 (x, y)
        self.run_test(M(), (x, y))

        # 创建两个张量 x 和 y，形状分别为 (4, 2, 1, 4) 和 (2, 3, 1)
        x = torch.randint(5, (4, 2, 1, 4))
        y = torch.randint(5, (2, 3, 1))

        # 再次运行测试，传入模型 M 和张量 (x, y)
        self.run_test(M(), (x, y))

        # 创建两个形状为 (2, 1, 4) 和 (5, 2, 3, 1) 的张量 x 和 y
        x = torch.randn(2, 1, 4)
        y = torch.randn(5, 2, 3, 1)

        # 再次运行测试，传入模型 M 和张量 (x, y)
        self.run_test(M(), (x, y))

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dist_normal(self):
        # 定义一个测试函数，测试 torch.distributions.Normal 类
        class M(torch.nn.Module):
            def forward(self, x, y):
                # 在 forward 方法中，使用 Normal 分布以 x 为均值，y 为标准差生成样本，并返回样本大小及输入 x 和 y
                return torch.distributions.Normal(x, y).sample().size(0), x, y

        # 运行测试，调用 self.run_test 方法，传入模型 M 和具体的输入参数组合
        self.run_test(M(), (torch.tensor([0.0]), torch.tensor([[1.0], [2.0]])))
        self.run_test(M(), (torch.tensor([0.0]), torch.tensor([1.0])))

        # 运行测试，传入模型 M 和更复杂的输入参数组合
        self.run_test(
            M(),
            (
                torch.tensor([[[0.0], [10.0]], [[2.0], [8.0]], [[2.0], [8.0]]]),
                torch.tensor([[1.0], [3.0]]),
            ),
        )

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dist_normal_correctness(self):
        # 定义一个测试函数，验证 Normal 分布模型输出的准确性
        class M(torch.nn.Module):
            def forward(self, x, y):
                # 在 forward 方法中，使用 Normal 分布以 x 为均值，y 为标准差生成 20000 个样本
                return torch.distributions.Normal(x, y).sample([20000])

        # 预期的均值和标准差
        expected_mean = 5.0
        expected_std = 10.0

        # 实例化模型 M
        model_export = M()
        # 创建一个包含预期均值和标准差的输入 dummy_input
        dummy_input = (torch.tensor([expected_mean]), torch.tensor([expected_std]))
        # 创建一个用于存储 ONNX 模型的 BytesIO 对象
        model_onnx = io.BytesIO()
        # 导出 ONNX 模型，使用 opset_version 参数指定运算集版本
        torch.onnx.export(
            model_export, dummy_input, model_onnx, opset_version=self.opset_version
        )
        # 创建一个 ONNX 推理会话
        ort_sess = verification._ort_session(model_onnx)
        # 在 ONNX 推理会话上运行模型，传入 dummy_input 作为输入
        ort_out = verification._run_onnx(ort_sess, inputs=dummy_input)

        # 计算 ONNX 输出的均值和标准差
        actual_std = np.std(ort_out)
        actual_mean = np.mean(ort_out)

        # 断言均值的绝对差值小于预期均值的 10%，否则输出异常信息
        assert (
            abs(abs(actual_mean) - expected_mean) <= expected_mean * 0.1
        ), "the gap of mean between ort outputs and expected one is unacceptable."
        # 断言标准差的绝对差值小于预期标准差的 10%，否则输出异常信息
        assert (
            abs(abs(actual_std) - expected_std) <= expected_std * 0.1
        ), "the gap of variance between ort outputs and expected one is unacceptable."

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_nn_init_normal_correctness(self):
        # 设置期望的均值和标准差
        expected_mean = 5.0
        expected_std = 10.0

        class M(torch.nn.Module):
            def forward(self):
                # 创建一个形状为 [1, 400, 50] 的张量，并填充为全1
                x = torch.ones([]).new_empty(1, 400, 50)
                # 对张量 x 进行正态分布初始化
                torch.nn.init.normal_(x, expected_mean, expected_std)
                return x

        # 实例化模型 M
        model_export = M()
        # 创建一个字节流对象，用于存储导出的 ONNX 模型
        model_onnx = io.BytesIO()
        # 准备空的输入数据元组
        test_inputs = tuple()
        # 将模型导出为 ONNX 格式
        torch.onnx.export(
            model_export, test_inputs, model_onnx, opset_version=self.opset_version
        )
        # 创建一个 ONNX 运行会话
        ort_sess = verification._ort_session(model_onnx)
        # 在 ONNX 运行时执行模型，并获取输出结果
        ort_out = verification._run_onnx(ort_sess, inputs=test_inputs)

        # 计算 ONNX 输出的标准差和均值
        actual_std = np.std(ort_out)
        actual_mean = np.mean(ort_out)

        # 断言检查均值的差距是否在可接受范围内
        assert (
            abs(abs(actual_mean) - expected_mean) <= expected_mean * 0.1
        ), "the gap of mean between ort outputs and expected one is unacceptable."
        # 断言检查方差的差距是否在可接受范围内
        assert (
            abs(abs(actual_std) - expected_std) <= expected_std * 0.1
        ), "the gap of variance between ort outputs and expected one is unacceptable."

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dist_uniform(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                # 返回在指定范围内均匀分布的样本数量及输入的 x 和 y
                return torch.distributions.Uniform(x, y).sample().size(0), x, y

        # 分别测试不同的输入情况
        self.run_test(M(), (torch.tensor([0.0]), torch.tensor([10.0])))
        self.run_test(M(), (torch.tensor([[0.0], [6.0]]), torch.tensor([[1.0], [7.0]])))
        self.run_test(
            M(), (torch.tensor([1.0]), torch.tensor([[10.0], [7.0], [9.0], [20.0]]))
        )

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dist_uniform_correctness(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                # 返回在指定范围内均匀分布的样本，数量为 10000
                return torch.distributions.Uniform(x, y).sample([10000])

        # 设置期望的最小值、最大值和均值
        expected_min = 5.0
        expected_max = 10.0
        expected_mean = (expected_min + expected_max) / 2

        # 创建模型实例
        model_export = M()
        # 创建一个包含期望最小值和最大值的虚拟输入
        dummy_input = (torch.tensor([expected_min]), torch.tensor([expected_max]))
        # 创建一个字节流对象，用于存储导出的 ONNX 模型
        model_onnx = io.BytesIO()
        # 将模型导出为 ONNX 格式
        torch.onnx.export(
            model_export, dummy_input, model_onnx, opset_version=self.opset_version
        )
        # 创建一个 ONNX 运行会话
        ort_sess = verification._ort_session(model_onnx)

        # 在 ONNX 运行时执行模型，并获取输出结果
        ort_out = verification._run_onnx(ort_sess, inputs=dummy_input)
        # 计算 ONNX 输出的最小值、最大值和均值
        actual_min = np.min(ort_out)
        actual_max = np.max(ort_out)
        actual_mean = np.mean(ort_out)

        # 断言检查最小值是否在期望范围内
        assert (
            actual_min >= expected_min
        ), "the minimum value of ort outputs is out of scope."
        # 断言检查最大值是否在期望范围内
        assert (
            actual_max <= expected_max
        ), "the maximum value of ort outputs is out of scope."
        # 断言检查均值是否在期望范围内
        assert (
            abs(actual_mean - expected_mean) <= expected_mean * 0.05
        ), "the mean value of ort outputs is out of scope."

    @skipIfUnsupportedMinOpsetVersion(13)
    # 定义一个测试方法，将序列映射为整数类型张量
    def test_sequence_to_int(self):
        # 定义一个简单的神经网络模块，接受输入并返回原始输入和整数类型结果张量
        class M(torch.nn.Module):
            def forward(self, x):
                # 创建一个元素为2的张量，长度与输入张量的行数相同
                result = torch.tensor([2 for i in range(x.size()[0])], dtype=torch.int)
                return x, result

        # 创建一个10行5列的随机张量
        x = torch.randn(10, 5)
        # 使用自定义的运行测试方法，对上面定义的模块进行测试
        self.run_test(M(), (x,))

    # 根据支持的最小运算集版本号为13，定义一个测试方法，将序列映射为浮点数类型张量
    @skipIfUnsupportedMinOpsetVersion(13)
    def test_sequence_to_float(self):
        # 定义一个简单的神经网络模块，接受输入并返回原始输入和浮点数类型结果张量
        class M(torch.nn.Module):
            def forward(self, x):
                # 创建一个元素为1.1的张量，长度与输入张量的行数相同
                result = torch.tensor(
                    [1.1 for i in range(x.size()[0])], dtype=torch.float
                )
                return x, result

        # 创建一个10行5列的随机张量
        x = torch.randn(10, 5)
        # 使用自定义的运行测试方法，对上面定义的模块进行测试
        self.run_test(M(), (x,))

    # 根据支持的最小运算集版本号为13，定义一个测试方法，将序列映射为布尔类型张量
    @skipIfUnsupportedMinOpsetVersion(13)
    def test_sequence_to_bool(self):
        # 定义一个简单的神经网络模块，接受输入并返回原始输入和布尔类型结果张量
        class M(torch.nn.Module):
            def forward(self, x):
                # 创建一个元素为False的张量，长度与输入张量的行数相同
                result = torch.tensor(
                    [False for i in range(x.size()[0])], dtype=torch.bool
                )
                return x, result

        # 创建一个10行5列的随机张量
        x = torch.randn(10, 5)
        # 使用自定义的运行测试方法，对上面定义的模块进行测试
        self.run_test(M(), (x,))

    # 定义一个测试方法，测试在引发异常的情况下，从if语句返回元组输出
    def test_tuple_output_from_if_with_raised_exception(self):
        # 定义一个简单的神经网络模块，接受张量输入并返回两个长度为5的零张量
        class M(torch.nn.Module):
            def forward(self, t: Tensor) -> Tuple[Tensor, Tensor]:
                if float(t) < 0:
                    raise Exception("Negative input")  # noqa: TRY002
                else:
                    return torch.zeros(5), torch.zeros(5)

        # 创建一个长度为1的零张量
        x = torch.zeros(1)
        # 使用Torch JIT编译模块，运行自定义的测试方法
        self.run_test(torch.jit.script(M()), (x,))

    # 注意：用于量化测试时，需谨慎选择缩放因子和零点，
    #       以确保输入和输出不会总是溢出/下溢。
    #       否则，测试结果可能不准确。
    # 根据支持的最小运算集版本号为10，定义一个测试方法，测试量化线性层
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_linear(self):
        # 创建一个具有4个输入和8个输出的量化线性模型
        model = torch.ao.nn.quantized.Linear(4, 8)
        # 设置固定权重以避免测试不稳定
        weight = torch.quantize_per_tensor(
            torch.arange(32, dtype=torch.float).view(8, 4), 0.5, 0, torch.qint8
        )
        # 设置非零偏置
        bias = torch.arange(8, dtype=torch.float)
        model.set_weight_bias(weight, bias)
        # 设置固定输入以避免测试不稳定
        input = torch.randn(4, 4)
        input = torch.arange(16, dtype=torch.float).view(4, 4) - 8
        input_tensor = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        # 使用自定义的运行测试方法，对上面定义的模型进行测试
        self.run_test(model, input_tensor)

    # 根据支持的最小运算集版本号为10，定义一个测试方法，测试量化一维卷积层
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_conv1d(self):
        # 创建一个具有16个输入通道，33个输出通道，3个内核大小和步幅为2的量化一维卷积模型
        model = torch.ao.nn.quantized.Conv1d(16, 33, 3, stride=2)
        # 手动初始化模型的权重和偏置为随机数，默认为全零
        q_weight = torch.quantize_per_tensor(
            torch.randn(33, 16, 3), 0.5, 0, torch.qint8
        )
        bias = torch.arange(33).to(torch.float) - 16
        model.set_weight_bias(q_weight, bias)
        # 创建一个随机输入张量
        input = torch.randn(3, 16, 32)
        # 将输入张量量化
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        # 使用自定义的运行测试方法，对上面定义的模型进行测试
        self.run_test(model, q_input)
    # 跳过不支持的最小 Opset 版本为 10 的测试函数修饰器
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_conv2d(self):
        # 创建一个 2D 量化卷积模型，输入通道数为 16，输出通道数为 33，卷积核大小为 3，步长为 2
        model = torch.ao.nn.quantized.Conv2d(16, 33, 3, stride=2)
        # 手动初始化模型的权重和偏置为随机数，初始值为零
        q_weight = torch.quantize_per_tensor(
            torch.randn(33, 16, 3, 3), 0.5, 0, torch.qint8
        )
        # 设置偏置，偏置值为从 -16 到 16 的序列
        bias = torch.arange(33).to(torch.float) - 16
        model.set_weight_bias(q_weight, bias)
        # 创建输入数据，形状为 (3, 16, 32, 32)
        input = torch.randn(3, 16, 32, 32)
        # 对输入数据进行量化，量化参数为 0.5，零点值为 128，数据类型为 quint8
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        # 运行测试函数，对量化卷积模型进行测试
        self.run_test(model, q_input)
    
    # 跳过不支持的最小 Opset 版本为 10 的测试函数修饰器
    @skipIfUnsupportedMinOpsetVersion(10)
    @skipIfQuantizationBackendQNNPack
    def test_quantized_conv3d(self):
        # 创建一个 3D 量化卷积模型，输入通道数为 16，输出通道数为 33，卷积核大小为 [2, 3, 4]，步长为 [3, 1, 2]
        model = torch.ao.nn.quantized.Conv3d(16, 33, [2, 3, 4], stride=[3, 1, 2])
        # 手动初始化模型的权重和偏置为随机数，初始值为零
        q_weight = torch.quantize_per_tensor(
            torch.randn(33, 16, 2, 3, 4), 0.5, 0, torch.qint8
        )
        # 设置偏置，偏置值为从 -16 到 16 的序列
        bias = torch.arange(33).to(torch.float) - 16
        model.set_weight_bias(q_weight, bias)
        # 创建输入数据，形状为 (3, 16, 8, 8, 8)
        input = torch.randn(3, 16, 8, 8, 8)
        # 对输入数据进行量化，量化参数为 0.5，零点值为 128，数据类型为 quint8
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        # 运行测试函数，对量化卷积模型进行测试
        self.run_test(model, q_input)
    
    # 跳过不支持的最小 Opset 版本为 10 的测试函数修饰器
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_adaptive_avg_pool2d(self):
        # 创建一个自适应平均池化模型，输出形状为 (5, 7)
        model = torch.nn.AdaptiveAvgPool2d((5, 7))
        # 创建输入数据，形状为 (4, 3, 10, 14)
        input = torch.randn(4, 3, 10, 14)
        # 对输入数据进行量化，量化参数为 0.2，零点值为 128，数据类型为 quint8
        q_input = torch.quantize_per_tensor(input, 0.2, 128, torch.quint8)
        # 运行测试函数，对自适应平均池化模型进行测试
        self.run_test(model, q_input)
    
    # 跳过不支持的最小 Opset 版本为 10 的测试函数修饰器
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_conv1d_relu(self):
        # 创建一个带有 ReLU 的 1D 量化卷积模型，输入通道数为 16，输出通道数为 33，卷积核大小为 3，步长为 2
        model = torch.ao.nn.intrinsic.quantized.ConvReLU1d(16, 33, 3, stride=2)
        # 手动初始化模型的权重和偏置为随机数，初始值为零
        q_weight = torch.quantize_per_tensor(
            torch.randn(33, 16, 3), 0.5, 0, torch.qint8
        )
        # 设置偏置，偏置值为从 -16 到 16 的序列
        bias = torch.arange(33).to(torch.float) - 16
        model.set_weight_bias(q_weight, bias)
        # 创建输入数据，形状为 (3, 16, 32)
        input = torch.randn(3, 16, 32)
        # 对输入数据进行量化，量化参数为 0.5，零点值为 128，数据类型为 quint8
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        # 运行测试函数，对量化卷积模型进行测试
        self.run_test(model, q_input)
    
    # 跳过不支持的最小 Opset 版本为 10 的测试函数修饰器
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_conv2d_relu(self):
        # 创建一个带有 ReLU 的 2D 量化卷积模型，输入通道数为 16，输出通道数为 33，卷积核大小为 3，步长为 2
        model = torch.ao.nn.intrinsic.quantized.ConvReLU2d(16, 33, 3, stride=2)
        # 手动初始化模型的权重和偏置为随机数，初始值为零
        q_weight = torch.quantize_per_tensor(
            torch.randn(33, 16, 3, 3), 0.5, 0, torch.qint8
        )
        # 设置偏置，偏置值为从 -16 到 16 的序列
        bias = torch.arange(33).to(torch.float) - 16
        model.set_weight_bias(q_weight, bias)
        # 创建输入数据，形状为 (3, 16, 32, 32)
        input = torch.randn(3, 16, 32, 32)
        # 对输入数据进行量化，量化参数为 0.5，零点值为 128，数据类型为 quint8
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        # 运行测试函数，对量化卷积模型进行测试
        self.run_test(model, q_input)
    
    # 跳过不支持的最小 Opset 版本为 10 的测试函数修饰器
    @skipIfUnsupportedMinOpsetVersion(10)
    @skipIfQuantizationBackendQNNPack
    # 定义一个测试函数，用于测试量化的 3D 卷积层与ReLU激活函数的组合
    def test_quantized_conv3d_relu(self):
        # 创建一个量化的 ConvReLU3d 模型，设置输入通道为16，输出通道为33，卷积核大小为[2, 3, 4]，步长为[3, 1, 2]
        model = torch.ao.nn.intrinsic.quantized.ConvReLU3d(
            16, 33, [2, 3, 4], stride=[3, 1, 2]
        )
        # 手动初始化模型的权重和偏置为随机数，默认为全零
        q_weight = torch.quantize_per_tensor(
            torch.randn(33, 16, 2, 3, 4), 0.5, 0, torch.qint8
        )
        # 创建一个偏置张量，其值为从0到32的序列，并转换为浮点型后减去16
        bias = torch.arange(33).to(torch.float) - 16
        # 设置模型的权重和偏置
        model.set_weight_bias(q_weight, bias)
        # 创建一个随机输入张量，形状为[3, 16, 8, 8, 8]
        input = torch.randn(3, 16, 8, 8, 8)
        # 对输入张量进行量化，量化参数为0.5和128，类型为quint8
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        # 运行测试函数，传入量化后的输入张量
        self.run_test(model, q_input)

    # 在 Opset 版本小于10时跳过该测试函数
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_conv_transpose1d(self):
        # 创建一个量化的 ConvTranspose1d 模型，设置输入通道为16，输出通道为33，卷积核大小为3，输出填充为1，步长为2
        model = torch.ao.nn.quantized.ConvTranspose1d(
            16, 33, 3, output_padding=1, stride=2
        )
        # 手动初始化模型的权重和偏置为随机数，默认为全零
        q_weight = torch.quantize_per_tensor(
            torch.randn(16, 33, 3), 0.5, 0, torch.qint8
        )
        # 创建一个偏置张量，其值为从0到32的序列，并转换为浮点型后减去16
        bias = torch.arange(33).to(torch.float) - 16
        # 设置模型的权重和偏置
        model.set_weight_bias(q_weight, bias)
        # 创建一个随机输入张量，形状为[3, 16, 32]
        input = torch.randn(3, 16, 32)
        # 对输入张量进行量化，量化参数为0.5和128，类型为quint8
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        # 运行测试函数，传入量化后的输入张量
        self.run_test(model, q_input)

    # 在 Opset 版本小于10时跳过该测试函数
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_conv_transpose2d(self):
        # 创建一个量化的 ConvTranspose2d 模型，设置输入通道为16，输出通道为33，卷积核大小为[3, 3]，输出填充为(0, 1)，步长为2
        model = torch.ao.nn.quantized.ConvTranspose2d(
            16, 33, 3, output_padding=(0, 1), stride=2
        )
        # 手动初始化模型的权重和偏置为随机数，默认为全零
        q_weight = torch.quantize_per_tensor(
            torch.randn(16, 33, 3, 3), 0.5, 0, torch.qint8
        )
        # 创建一个偏置张量，其值为从0到32的序列，并转换为浮点型后减去16
        bias = torch.arange(33).to(torch.float) - 16
        # 设置模型的权重和偏置
        model.set_weight_bias(q_weight, bias)
        # 创建一个随机输入张量，形状为[3, 16, 32, 32]
        input = torch.randn(3, 16, 32, 32)
        # 对输入张量进行量化，量化参数为0.5和128，类型为quint8
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        # 运行测试函数，传入量化后的输入张量
        self.run_test(model, q_input)

    # 在 Opset 版本小于10时跳过该测试函数，并且不使用 QNNPack 量化后端
    @skipIfUnsupportedMinOpsetVersion(10)
    @skipIfQuantizationBackendQNNPack
    def test_quantized_conv_transpose3d(self):
        # 创建一个量化的 ConvTranspose3d 模型，设置输入通道为16，输出通道为33，卷积核大小为[2, 3, 4]，输出填充为(0, 1, 2)，步长为[3, 1, 2]
        model = torch.ao.nn.quantized.ConvTranspose3d(
            16, 33, [2, 3, 4], output_padding=(0, 1, 2), stride=[3, 1, 2]
        )
        # 手动初始化模型的权重和偏置为随机数，默认为全零
        q_weight = torch.quantize_per_tensor(
            torch.randn(16, 33, 2, 3, 4), 0.5, 0, torch.qint8
        )
        # 创建一个偏置张量，其值为从0到32的序列，并转换为浮点型后减去16
        bias = torch.arange(33).to(torch.float) - 16
        # 设置模型的权重和偏置
        model.set_weight_bias(q_weight, bias)
        # 创建一个随机输入张量，形状为[3, 16, 8, 8, 8]
        input = torch.randn(3, 16, 8, 8, 8)
        # 对输入张量进行量化，量化参数为0.5和128，类型为quint8
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        # 运行测试函数，传入量化后的输入张量
        self.run_test(model, q_input)
    # 使用 common_utils.parametrize 装饰器，为每个子测试函数或模块提供参数化支持
    @common_utils.parametrize(
        "function_or_module",
        [
            # 定义测试函数或模块，以及其对应的名称
            common_utils.subtest(
                torch.nn.ReLU(),
                name="relu",
            ),
            common_utils.subtest(
                torch.nn.LeakyReLU(),
                name="leaky_relu",
            ),
            common_utils.subtest(
                torch.ao.nn.quantized.LeakyReLU(2.0, 1),
                name="quantized_leaky_relu",
            ),
            common_utils.subtest(
                torch.ao.nn.quantized.Hardswish(2.0, 1),
                name="quantized_hardswish",
            ),
            common_utils.subtest(
                torch.nn.Sigmoid(),
                name="sigmoid",
            ),
            common_utils.subtest(
                torch.ao.nn.quantized.Sigmoid(2.0, 1),
                name="quantized_sigmoid",
            ),
            common_utils.subtest(
                torch.nn.Hardsigmoid(),
                name="hardsigmoid",
            ),
            common_utils.subtest(
                torch.nn.Tanh(),
                name="tanh",
            ),
            common_utils.subtest(
                torch.nn.Hardtanh(),
                name="hardtanh",
            ),
            common_utils.subtest(
                # 使用 lambda 表达式定义匿名函数，对输入进行转置操作
                lambda x: torch.transpose(x, 0, 1),
                name="transpose",
            ),
            common_utils.subtest(
                # 使用 lambda 表达式定义匿名函数，对输入进行扩展操作
                lambda x: x.expand(2, 4, 2, 3),
                name="expand",
            ),
            common_utils.subtest(
                # 使用 lambda 表达式定义匿名函数，对输入进行视图重塑操作
                lambda x: x.view(1, 4, 6),
                name="view",
            ),
            common_utils.subtest(
                # 使用 lambda 表达式定义匿名函数，从输入中选择指定维度的切片
                lambda x: x.select(1, 1),
                name="select",
            ),
            common_utils.subtest(
                # 实例化一个带有指定参数的量化 LayerNorm 模块
                torch.ao.nn.quantized.LayerNorm(
                    [4, 2, 3],
                    torch.nn.Parameter(torch.ones([4, 2, 3])),
                    torch.nn.Parameter(torch.zeros([4, 2, 3])),
                    2.0,
                    1,
                ),
                name="layer_norm",
            ),
            common_utils.subtest(
                # 实例化一个带有指定参数的一维量化 InstanceNorm 模块
                torch.ao.nn.quantized.InstanceNorm1d(
                    2,
                    torch.nn.Parameter(torch.ones(4)),
                    torch.nn.Parameter(torch.zeros(4)),
                    2.0,
                    1,
                ),
                name="instance_norm",
            ),
            common_utils.subtest(
                # 实例化一个带有指定参数的量化 GroupNorm 模块
                torch.ao.nn.quantized.GroupNorm(
                    2,
                    4,
                    torch.nn.Parameter(torch.zeros(4)),
                    torch.nn.Parameter(torch.zeros(4)),
                    2.0,
                    1,
                ),
                name="group_norm",
            ),
            common_utils.subtest(
                # 使用 lambda 表达式定义匿名函数，对输入进行 as_strided 操作
                lambda x: torch.as_strided(x, (2, 2), (1, 2)),
                name="as_strided",
            ),
        ],
    )
    # 使用 skipScriptTest 装饰器，跳过这些测试函数或模块的脚本测试
    @skipScriptTest()
    # 使用装饰器跳过不支持的最小操作集版本为10的测试函数
    @skipIfUnsupportedMinOpsetVersion(10)
    # 定义测试量化一元操作的方法，接受一个参数 function_or_module
    def test_quantized_unary_ops(self, function_or_module):
        # 创建一个形状为 (1, 4, 2, 3) 的随机张量作为输入
        input = torch.randn(1, 4, 2, 3)
        # 对输入张量进行量化，设置比例因子为0.26，零点为128，数据类型为torch.quint8
        q_input = torch.quantize_per_tensor(input, 0.26, 128, torch.quint8)

        # 定义一个继承自torch.nn.Module的模型类
        class Model(torch.nn.Module):
            def __init__(self, function_or_module):
                super().__init__()
                self.function_or_module = function_or_module

            # 定义模型的前向传播方法
            def forward(self, x):
                return self.function_or_module(x)

        # 运行测试函数，传入实例化后的Model对象和量化后的输入数据
        self.run_test(Model(function_or_module), q_input)

    # 使用装饰器跳过不支持的最小操作集版本为10的测试函数
    @skipIfUnsupportedMinOpsetVersion(10)
    # 定义测试量化展平操作的方法
    def test_quantized_flatten(self):
        # 定义一个继承自torch.nn.Module的展平模型类
        class FlattenModel(torch.nn.Module):
            # 定义模型的前向传播方法，对输入进行展平
            def forward(self, input):
                return torch.flatten(input)

        # 创建一个形状为 (1, 2, 3, 4) 的随机张量，并对其进行量化
        x = torch.quantize_per_tensor(torch.randn(1, 2, 3, 4), 1, 0, torch.quint8)
        # 运行测试函数，传入实例化后的FlattenModel对象和量化后的输入数据
        self.run_test(FlattenModel(), x)

    # 使用装饰器跳过不支持的最小操作集版本为10的测试函数
    @skipIfUnsupportedMinOpsetVersion(10)
    # 使用装饰器跳过脚本测试，因为在脚本函数中无法实例化'QFunctional'类
    def test_quantized_cat_when_concatinating_the_same_tensor(self):
        # 定义一个继承自torch.nn.Module的量化自我连接模型类
        class QuantizedSelfConcatenationModel(torch.nn.Module):
            # 定义模型的前向传播方法，使用QFunctional.cat方法连接相同张量
            def forward(self, x):
                return torch.ao.nn.quantized.QFunctional().cat((x, x), dim=1)

        # 创建一个形状为 (2, 3) 的全1张量，并对其进行量化
        q_input = torch.quantize_per_tensor(torch.ones(2, 3), 0.26, 128, torch.quint8)
        # 运行测试函数，传入实例化后的QuantizedSelfConcatenationModel对象和量化后的输入数据
        self.run_test(QuantizedSelfConcatenationModel(), q_input)

    # 使用common_utils.parametrize装饰器，为测试函数提供参数化测试数据
    @common_utils.parametrize(
        "x, y",
        [
            # 参数化子测试，测试不同形状的量化张量
            common_utils.subtest(
                [
                    torch.quantize_per_tensor(
                        torch.ones(2, 3), 0.26, 128, torch.quint8
                    ),
                    torch.quantize_per_tensor(
                        torch.zeros(1, 3), 0.26, 128, torch.quint8
                    ),
                ],
                name="different_shape",
            ),
            # 参数化子测试，测试不同比例因子的量化张量
            common_utils.subtest(
                [
                    torch.quantize_per_tensor(
                        torch.ones(2, 3), 0.26, 128, torch.quint8
                    ),
                    torch.quantize_per_tensor(torch.ones(2, 3), 42, 1, torch.quint8),
                ],
                name="different_scale",
            ),
            # 参数化子测试，测试不同零点的量化张量
            common_utils.subtest(
                [
                    torch.quantize_per_tensor(
                        torch.ones(2, 3), 0.26, 128, torch.quint8
                    ),
                    torch.quantize_per_tensor(torch.ones(2, 3), 0.26, 63, torch.quint8),
                ],
                name="different_zero_point",
            ),
            # 参数化子测试，测试不同比例因子和零点的量化张量
            common_utils.subtest(
                [
                    torch.quantize_per_tensor(
                        torch.ones(2, 3), 0.26, 128, torch.quint8
                    ),
                    torch.quantize_per_tensor(torch.ones(2, 3), 0.1, 63, torch.quint8),
                ],
                name="different_zero_point_and_scale",
            ),
        ],
    )
    @skipIfUnsupportedMinOpsetVersion(10)
    @skipScriptTest()  # 跳过不支持的最小操作集版本，并且不在脚本测试中使用
    
    def test_quantized_cat(self, x: torch.Tensor, y: torch.Tensor):
        class QuantizedConcatenationModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.ao.nn.quantized.QFunctional().cat((x, y), dim=0)
        
        self.run_test(QuantizedConcatenationModel(), (x, y))
    
    @skipIfUnsupportedMinOpsetVersion(10)
    # torch.jit.frontend.FrontendError:
    # 在脚本函数中无法实例化 'QFunctional' 类
    @skipScriptTest()
    def test_quantized_arithmetic_qfunctional(self):
        x = torch.quantize_per_tensor(torch.randn(3, 4), 0.2, 128, torch.quint8)
        y = torch.quantize_per_tensor(torch.randn(3, 4), 0.2, 128, torch.quint8)
    
        class ArithmeticModel(torch.nn.Module):
            def forward(self, x, y):
                o = torch.ao.nn.quantized.QFunctional().add(x, y)
                o = torch.ao.nn.quantized.QFunctional().mul(o, x)
                return o
        
        self.run_test(ArithmeticModel(), (x, y))
    
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_arithmetic(self):
        x = torch.quantize_per_tensor(torch.randn(3, 4), 0.2, 128, torch.quint8)
        y = torch.quantize_per_tensor(torch.randn(3, 4), 0.2, 128, torch.quint8)
    
        class ArithmeticModel2(torch.nn.Module):
            def forward(self, x, y):
                o = torch.ops.quantized.add(x, y, 0.4, 100)
                o = torch.ops.quantized.mul(o, x, 0.4, 100)
                return o
        
        self.run_test(ArithmeticModel2(), (x, y))
    
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantize_per_tensor(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return (
                    torch.quantize_per_tensor(x, 0.2, 0, torch.qint8),
                    torch.quantize_per_tensor(x, 0.2, 128, torch.quint8),
                )
        
        x = torch.randn(4, 6)
        self.run_test(Module(), x)
    
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_dequantize(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return torch.dequantize(x)
        
        x = torch.quantize_per_tensor(torch.randn(3, 4), 0.2, 0, torch.qint8)
        self.run_test(Module(), x)
    
    @skipIfUnsupportedMinOpsetVersion(13)
    def test_qat_linear_per_channel(self):
        # 定义一个包含量化和反量化操作的简单模型类
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()  # 添加量化前处理器
                self.linear = torch.nn.Linear(4, 3)  # 添加线性层
                self.dequant = torch.ao.quantization.DeQuantStub()  # 添加反量化后处理器

            def forward(self, x):
                x = self.quant(x)  # 对输入进行量化
                x = self.linear(x)  # 线性层处理
                x = self.dequant(x)  # 对输出进行反量化
                return x

        model = M()  # 创建模型实例
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")  # 获取默认的量化配置
        model = torch.ao.quantization.prepare_qat(model)  # 准备用于训练的量化模型
        # 设置固定的权重和偏置以避免不稳定的测试结果
        model.linear.weight = torch.nn.Parameter(
            _construct_tensor_for_quantization_test((3, 4))
        )
        model.linear.bias = torch.nn.Parameter(torch.arange(3, dtype=torch.float))
        model = torch.ao.quantization.convert(model)  # 将模型转换为量化模型

        # 设置固定的输入以避免不稳定的测试结果
        input = _construct_tensor_for_quantization_test((4, 4), offset=-8)
        self.run_test(model, input)  # 执行测试

    @unittest.skip(
        "ORT fails with Validating no unexpected access using an invalid node_index on torch converted model"
    )
    @skipIfUnsupportedMinOpsetVersion(13)
    def test_quantized_list_of_inputs_with_cat(self):
        # 定义一个简单的模型类，包含量化和反量化操作，并使用torch.cat操作
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()  # 添加量化前处理器
                self.dequant = torch.ao.quantization.DeQuantStub()  # 添加反量化后处理器

            def forward(self, x):
                x = self.quant(x)  # 对输入进行量化
                x = torch.cat([x, x], 1)  # 使用torch.cat进行张量拼接
                x = self.dequant(x)  # 对输出进行反量化
                return x

        model = TestModel()  # 创建模型实例
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")  # 获取默认的量化配置
        model = torch.ao.quantization.prepare_qat(model)  # 准备用于训练的量化模型
        model = torch.ao.quantization.convert(model)  # 将模型转换为量化模型
        x = torch.randn(2, 4, 6)  # 创建随机输入张量
        self.run_test(model, x)  # 执行测试

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_qat_relu(self):
        # 定义一个包含ReLU激活函数的简单模型类
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()  # 添加量化前处理器
                self.relu = torch.nn.ReLU()  # 添加ReLU激活函数
                self.dequant = torch.ao.quantization.DeQuantStub()  # 添加反量化后处理器

            def forward(self, x):
                x = self.quant(x)  # 对输入进行量化
                x = self.relu(x)  # 使用ReLU激活函数
                x = self.dequant(x)  # 对输出进行反量化
                return x

        model = M()  # 创建模型实例
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")  # 获取默认的量化配置
        model = torch.ao.quantization.prepare_qat(model)  # 准备用于训练的量化模型
        model = torch.ao.quantization.convert(model)  # 将模型转换为量化模型
        input = torch.randn(8, 4)  # 创建随机输入张量
        self.run_test(model, input)  # 执行测试

    @skipIfUnsupportedMinOpsetVersion(13)
    @skipIfUnsupportedMinOpsetVersion(13)
    # 装饰器：如果当前的最小操作集版本小于13，则跳过测试
    def test_qat_conv2d_relu(self):
        # 定义一个名为 M 的子类，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 添加量化存根
                self.quant = torch.ao.quantization.QuantStub()
                # 添加二维卷积层，输入通道数为4，输出通道数为2，卷积核大小为3x3，步长为2
                self.conv = torch.nn.Conv2d(4, 2, 3, stride=2)
                # 添加 ReLU 激活函数
                self.relu = torch.nn.ReLU()
                # 添加反量化存根
                self.dequant = torch.ao.quantization.DeQuantStub()

            # 前向传播方法
            def forward(self, x):
                # 对输入 x 进行量化
                x = self.quant(x)
                # 对量化后的 x 进行卷积操作
                x = self.conv(x)
                # 对卷积结果应用 ReLU 激活函数
                x = self.relu(x)
                # 对激活后的结果进行反量化
                x = self.dequant(x)
                # 返回处理后的结果 x
                return x

        # 创建 M 类的实例 model
        model = M()
        # 设置模型的量化配置为默认配置 "fbgemm"
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        # 准备模型进行量化感知训练
        model = torch.ao.quantization.prepare_qat(model)
        # 设置固定的权重和偏置，以避免测试结果不稳定
        model.conv.weight = torch.nn.Parameter(
            _construct_tensor_for_quantization_test((2, 4, 3, 3), max_val=2)
        )
        model.conv.bias = torch.nn.Parameter(torch.tensor([0.0, 1.0]))
        # 将模型转换为量化后的模型
        model = torch.ao.quantization.convert(model)

        # 设置固定的输入，以避免测试结果不稳定
        input = _construct_tensor_for_quantization_test(
            (3, 4, 8, 8), offset=-384, max_val=12
        )
        # 运行测试方法 self.run_test，传入量化后的模型和固定输入
        self.run_test(model, input)
    # 定义一个测试用例函数，用于测试量化训练后的 Conv2d 和 ReLU 融合的模型
    def test_qat_conv2d_relu_fused(self):
        # 定义一个简单的神经网络模型类
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加量化辅助模块，用于量化和反量化操作
                self.quant = torch.ao.quantization.QuantStub()
                # 添加一个 2D 卷积层，输入通道数为 4，输出通道数为 2，卷积核大小为 3x3，步长为 2
                self.conv = torch.nn.Conv2d(4, 2, 3, stride=2)
                # 添加 ReLU 激活函数
                self.relu = torch.nn.ReLU()
                # 添加反量化辅助模块
                self.dequant = torch.ao.quantization.DeQuantStub()

            # 定义前向传播函数
            def forward(self, x):
                x = self.quant(x)  # 对输入进行量化
                x = self.conv(x)   # 卷积操作
                x = self.relu(x)   # ReLU 激活函数
                x = self.dequant(x)  # 反量化操作
                return x

        model = M()  # 创建模型实例
        # 设置模型的量化配置为默认配置 "fbgemm"
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        # 对模型进行模块融合，将 Conv2d 和 ReLU 层融合为单个操作
        model = torch.ao.quantization.fuse_modules(model.eval(), [["conv", "relu"]])
        # 准备模型进行量化训练
        model = torch.ao.quantization.prepare_qat(model.train())
        # 为了避免测试结果不稳定，设置固定的权重和偏置值
        model.conv.weight = torch.nn.Parameter(
            _construct_tensor_for_quantization_test((2, 4, 3, 3), max_val=2)
        )
        model.conv.bias = torch.nn.Parameter(torch.tensor([0.0, 1.0]))
        # 将模型转换为量化表示
        model = torch.ao.quantization.convert(model)

        # 设置固定的输入数据，以避免测试结果不稳定
        input = _construct_tensor_for_quantization_test(
            (3, 4, 8, 8), offset=-384, max_val=12
        )
        # 运行测试函数，传入模型和输入数据
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(13)
    # 定义一个测试函数，用于测试量化训练后的 Linear 和 ReLU 融合的模型
    def test_qat_linear_relu_fused(self):
        # 定义一个简单的神经网络模型类
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加量化辅助模块，用于量化和反量化操作
                self.quant = torch.ao.quantization.QuantStub()
                # 添加一个全连接层，输入特征数为 4，输出特征数为 2
                self.linear = torch.nn.Linear(4, 2)
                # 添加 ReLU 激活函数
                self.relu = torch.nn.ReLU()
                # 添加反量化辅助模块
                self.dequant = torch.ao.quantization.DeQuantStub()

            # 定义前向传播函数
            def forward(self, x):
                x = self.quant(x)  # 对输入进行量化
                x = self.linear(x)  # 线性变换
                x = self.relu(x)   # ReLU 激活函数
                x = self.dequant(x)  # 反量化操作
                return x

        model = M()  # 创建模型实例
        # 设置模型的量化配置为默认配置 "fbgemm"
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        # 对模型进行模块融合，将 Linear 和 ReLU 层融合为单个操作
        model = torch.ao.quantization.fuse_modules(model.eval(), [["linear", "relu"]])
        # 准备模型进行量化训练
        model = torch.ao.quantization.prepare_qat(model.train())
        # 为了避免测试结果不稳定，设置固定的权重和偏置值
        model.linear.weight = torch.nn.Parameter(
            _construct_tensor_for_quantization_test((2, 4), max_val=2)
        )
        model.linear.bias = torch.nn.Parameter(torch.tensor([0.0, 1.0]))
        # 将模型转换为量化表示
        model = torch.ao.quantization.convert(model)

        # 设置固定的输入数据，以避免测试结果不稳定
        input = _construct_tensor_for_quantization_test((3, 4), offset=-384, max_val=12)
        # 运行测试函数，传入模型和输入数据
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_qat_maxpool2d(self):
        # 定义一个包含量化和反量化模块的神经网络模型类
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加量化模块
                self.quant = torch.ao.quantization.QuantStub()
                # 添加最大池化层，指定内核大小为3，步长为2，填充为1
                self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                # 添加反量化模块
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                # 前向传播函数，依次调用量化、池化和反量化模块
                x = self.quant(x)
                x = self.pool(x)
                x = self.dequant(x)
                return x

        # 创建 M 类的实例
        model = M()
        # 设置量化配置为默认的 fbgemm 配置
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        # 准备模型以进行量化感知训练
        model = torch.ao.quantization.prepare_qat(model.train())
        # 将模型转换为量化版本
        model = torch.ao.quantization.convert(model)

        # 设置固定输入以避免测试不稳定性
        input = _construct_tensor_for_quantization_test((4, 4, 3, 2))
        # 运行测试函数，验证模型行为
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipScriptTest()  # Scale and Zero-point must be a scalar in ORT:optimization
    def test_qat_avg_pool2d(self):
        # 创建包含量化模块、平均池化层和反量化模块的序列化模型
        model = torch.nn.Sequential(
            torch.ao.quantization.QuantStub(),
            torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            torch.ao.quantization.DeQuantStub(),
        )
        # 设置量化配置为默认的 fbgemm 配置
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        # 准备模型以进行量化感知训练
        model = torch.ao.quantization.prepare_qat(model.train())
        # 将模型转换为量化版本
        model = torch.ao.quantization.convert(model)
        # 设置固定输入以避免测试不稳定性
        input = _construct_tensor_for_quantization_test((4, 4, 3, 2))
        # 运行测试函数，验证模型行为
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_qat_upsample_nearest2d(self):
        # 创建包含量化模块、最近邻插值上采样层和反量化模块的序列化模型
        model = torch.nn.Sequential(
            torch.ao.quantization.QuantStub(),
            torch.nn.UpsamplingNearest2d(scale_factor=1.5),
            torch.ao.quantization.DeQuantStub(),
        )
        # 设置量化配置为默认的 fbgemm 配置
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        # 准备模型以进行量化感知训练
        model = torch.ao.quantization.prepare_qat(model.train())
        # 将模型转换为量化版本
        model = torch.ao.quantization.convert(model)
        # 设置固定输入以避免测试不稳定性
        input = _construct_tensor_for_quantization_test((4, 3, 2, 2))
        # 运行测试函数，验证模型行为
        self.run_test(model, input)

    def test_0d_tensor_broadcast(self):
        # 定义一个函数类，实现输入 x 和 y 的加法和乘法操作
        class fn(torch.nn.Module):
            def forward(self, x, y):
                # 计算 x 和 y 的加法
                a = torch.add(x, y)
                # 计算 y 的平方
                b = torch.mul(y, y)
                # 返回 a 和 b 的和作为输出
                return a + b

        # 创建长度为 0 的全一张量 x 和长度为 1 的全一张量 y
        x = torch.ones(0)
        y = torch.ones(1)
        # 运行测试函数，验证 fn 类的行为，指定输入输出的名称
        self.run_test(fn(), (x, y), input_names=["x", "y"], output_names=["output"])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_convolution_allow_tf32(self):
        # 定义一个测试函数，测试卷积操作是否支持TF32模式
        class Module(torch.nn.Module):
            def __init__(self, allow_tf32):
                super().__init__()

                # 初始化模块，设置是否支持TF32模式
                self.allow_tf32 = allow_tf32
                # 随机生成权重张量
                weight = torch.rand(32, 3, 3, 3)
                self.weight = torch.nn.Parameter(weight)

            def forward(self, x):
                # 根据是否支持TF32模式选择不同的卷积操作
                if self.allow_tf32:
                    return torch._convolution(
                        x,
                        self.weight,
                        None,
                        [2, 2],     # 卷积核大小
                        [0, 0],     # 填充大小
                        [1, 1],     # 步幅大小
                        False,      # 是否进行dilated卷积
                        [0, 0],     # 输出padding大小
                        1,          # 输出填充
                        False,      # 是否需要计算输出pad
                        False,      # 是否进行transposed卷积
                        True,       # 是否使用混合精度
                        True,       # 是否支持torchscript
                    )
                else:
                    return torch._convolution(
                        x,
                        self.weight,
                        None,
                        [2, 2],     # 卷积核大小
                        [0, 0],     # 填充大小
                        [1, 1],     # 步幅大小
                        False,      # 是否进行dilated卷积
                        [0, 0],     # 输出padding大小
                        1,          # 输出填充
                        False,      # 是否需要计算输出pad
                        False,      # 是否进行transposed卷积
                        True,       # 是否使用混合精度
                        True,       # 是否支持torchscript
                    )

        x = torch.randn(1, 3, 224, 224)
        # 运行测试，分别测试不支持和支持TF32模式的卷积操作
        self.run_test(Module(False), x, rtol=1e-3, atol=1e-6)
        self.run_test(Module(True), x, rtol=1e-3, atol=1e-6)

    class AffineGridModule(torch.nn.Module):
        def __init__(self, align_corners) -> None:
            super().__init__()
            # 初始化模块，设置是否按照角点对齐
            self.align_corners = align_corners

        def forward(self, theta, size):
            # 调用PyTorch中的affine_grid函数生成仿射网格
            return torch.nn.functional.affine_grid(theta, size, self.align_corners)

    @skipIfUnsupportedMinOpsetVersion(20)
    @skipScriptTest()
    @common_utils.parametrize(
        "align_corners",
        (True, False),
    )
    @common_utils.parametrize(
        "theta_params",
        (
            (
                10,
                np.array([0.3, -0.5]),
                np.array([1.5, 0.5]),
            ),
            (
                60,
                np.array([-0.5, -0.5]),
                np.array([3.0, 5.5]),
            ),
        ),
    )
    @common_utils.parametrize(
        "size",
        ([1, 1, 3, 2], [2, 10, 2, 3]),
    )
    # 定义测试函数，用于测试二维仿射网格生成
    def test_affine_grid_2d(self, align_corners, theta_params, size):
        # 解包 theta_params 元组
        angle, translation, scale = theta_params
        # 创建空的 theta 数组，数据类型为 np.float32
        theta = np.array([], dtype=np.float32)
        # 循环 size[0] 次
        for _ in range(size[0]):
            # 将角度转换为弧度
            angle_radian = (angle / 180.0) * np.pi
            # 向 theta 数组追加一组值，表示仿射变换矩阵
            theta = np.append(
                theta,
                [
                    np.cos(angle_radian) * scale[0],
                    -np.sin(angle_radian),
                    translation[0],
                    np.sin(angle_radian),
                    np.cos(angle_radian) * scale[1],
                    translation[1],
                ],
            )
        # 将 theta 重新reshape为 size[0] x 2 x 3 的数组
        theta = theta.reshape(size[0], 2, 3)
        # 转换为 Torch 张量类型
        theta = torch.Tensor(theta)
        # 运行 AffineGridModule 测试
        self.run_test(TestONNXRuntime.AffineGridModule(align_corners), (theta, size))

    # 标记为 Opset 版本不支持跳过测试
    @skipIfUnsupportedMinOpsetVersion(20)
    # 跳过脚本测试装饰器
    @skipScriptTest()
    # 参数化测试，测试 align_corners 参数为 True 和 False 两种情况
    @common_utils.parametrize(
        "align_corners",
        (True, False),
    )
    # 参数化测试，测试 theta_params 参数的不同组合
    @common_utils.parametrize(
        "theta_params",
        (
            (
                [10, 20],
                np.array([0.3, -0.5, 1.8]),
                np.array([1.5, 2.0, 0.5]),
            ),
            (
                [60, -30],
                np.array([-0.5, -0.5, 0.3]),
                np.array([0.3, 3.0, 5.5]),
            ),
        ),
    )
    # 参数化测试，测试 size 参数的不同组合
    @common_utils.parametrize(
        "size",
        ([1, 1, 3, 2, 2], [2, 10, 2, 2, 3]),
    )
    # 定义测试函数，用于测试三维仿射网格生成
    def test_affine_grid_3d(self, align_corners, theta_params, size):
        # 解包 theta_params 元组
        angle, translation, scale = theta_params
        # 创建空的 theta 数组，数据类型为 np.float32
        theta = np.array([], dtype=np.float32)
        # 循环 size[0] 次
        for _ in range(size[0]):
            # 将角度转换为弧度
            angle_radian_x = (angle[0] / 180.0) * np.pi
            angle_radian_y = (angle[1] / 180.0) * np.pi
            # 创建 X 轴和 Y 轴的旋转矩阵
            rot_matrix_x = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(angle_radian_x), -np.sin(angle_radian_x)],
                    [0, np.sin(angle_radian_x), np.cos(angle_radian_x)],
                ]
            )
            rot_matrix_y = np.array(
                [
                    [np.cos(angle_radian_y), 0, np.sin(angle_radian_y)],
                    [0, 1, 0],
                    [-np.sin(angle_radian_y), 0, np.cos(angle_radian_y)],
                ]
            )
            # 计算整体旋转矩阵
            rot_matrix = np.matmul(rot_matrix_x, rot_matrix_y)
            # 缩放旋转矩阵
            rot_matrix = rot_matrix * scale.reshape(3, 1)
            # 添加平移向量到旋转矩阵
            rot_matrix = np.append(rot_matrix, np.reshape(translation, (3, 1)), axis=1)
            # 向 theta 数组追加一组值，表示仿射变换矩阵
            theta = np.append(theta, rot_matrix.flatten())

        # 将 theta 重新reshape为 size[0] x 3 x 4 的数组
        theta = theta.reshape(size[0], 3, 4)
        # 转换为 Torch 张量类型
        theta = torch.Tensor(theta)
        # 运行 AffineGridModule 测试
        self.run_test(TestONNXRuntime.AffineGridModule(align_corners), (theta, size))

    # 标记为 Opset 版本不支持跳过测试
    @skipIfUnsupportedMinOpsetVersion(16)
    # 参数化测试，测试 mode 参数的不同取值
    @common_utils.parametrize(
        "mode",
        ("bilinear", "nearest", "bicubic"),
    )
    # 参数化测试，测试 padding_mode 参数的不同取值
    @common_utils.parametrize(
        "padding_mode",
        ("zeros", "border", "reflection"),
    )
    @common_utils.parametrize(
        "align_corners",
        (True, False),
        name_fn=lambda align_corners: str(align_corners),
    )
    # 使用 common_utils.parametrize 装饰器，用于参数化测试，测试 align_corners 参数为 True 和 False 两种情况
    def test_grid_sample(self, mode, padding_mode, align_corners):
        # 设置输入数据的维度
        n, c, d_in, h_in, w_in, d_out, h_out, w_out = 1, 1, 2, 3, 2, 3, 2, 4

        # 初始化容差和相对容差字典
        atol_rtol = {}
        # 如果模式和填充模式为 "bicubic" 和 "border"
        if (mode, padding_mode) == ("bicubic", "border"):
            # 如果 align_corners 为 True，则设置较大的容差和相对容差
            if align_corners:
                atol_rtol.update({"atol": 0.3, "rtol": 0.4})
            else:
                # 否则设置较小的容差和相对容差
                atol_rtol.update({"atol": 0.02, "rtol": 0.02})
        
        # 随机生成输入数据 input 和 grid
        input, grid = torch.randn(n, c, h_in, w_in), torch.randn(n, h_out, w_out, 2)

        # 定义 GridSampleModule 类，继承自 torch.nn.Module
        class GridSampleModule(torch.nn.Module):
            def __init__(self, mode, padding_mode, align_corners) -> None:
                super().__init__()
                # 初始化模式、填充模式和 align_corners 参数
                self.mode, self.padding_mode, self.align_corners = (
                    mode,
                    padding_mode,
                    align_corners,
                )

            # 定义前向传播函数
            def forward(self, input, grid):
                # 调用 torch.nn.functional.grid_sample 执行网格采样操作
                return torch.nn.functional.grid_sample(
                    input, grid, self.mode, self.padding_mode, self.align_corners
                )

        # 运行测试函数 run_test，测试 GridSampleModule 的输出结果
        self.run_test(
            GridSampleModule(mode, padding_mode, align_corners),
            (input, grid),
            **atol_rtol,
        )

        # 提示信息：ONNX Opset 16 不支持 5D 体积输入的 GridSample 操作
        volumetric_input_tensor = torch.randn(n, c, d_in, h_in, w_in)
        volumetric_grid_tensor = torch.randn(n, d_out, h_out, w_out, 3)
        # 遍历不同的模式、填充模式和 align_corners 参数组合
        for mode, padding_mode, align_corners in itertools.product(
            (
                "bilinear",
                "nearest",
            ),  # PyTorch grid_sample "bicubic" mode does not support 5D volumetric input.
            (
                "zeros",
                "border",
                "reflection",
            ),
            (
                True,
                False,
            ),
        ):
            # 如果当前 opset_version 小于 20
            if self.opset_version < 20:
                # 使用 assertRaises 检查是否抛出 OnnxExporterError 异常
                with self.assertRaises(
                    torch.onnx.errors.OnnxExporterError,
                ):
                    # 运行测试函数 run_test，测试 GridSampleModule 的输出结果
                    self.run_test(
                        GridSampleModule(mode, padding_mode, align_corners),
                        (volumetric_input_tensor, volumetric_grid_tensor),
                        **atol_rtol,
                    )
            else:
                # 否则，正常运行测试函数 run_test，测试 GridSampleModule 的输出结果
                self.run_test(
                    GridSampleModule(mode, padding_mode, align_corners),
                    (volumetric_input_tensor, volumetric_grid_tensor),
                    **atol_rtol,
                )

    # 定义 IfNoneInput 类，继承自 torch.nn.Module
    class IfNoneInput(torch.nn.Module):
        # 定义前向传播函数，返回类型为 Optional[Tensor]
        def forward(self, x) -> Optional[Tensor]:
            # 初始化 y 为 None
            y: Optional[Tensor] = None
            # 如果输入张量 x 的第一个维度大于 1
            if x.size(0) > 1:
                # 将 y 赋值为输入张量 x
                y = x
            # 返回 y
            return y
    # 定义一个继承自 torch.nn.Module 的模块 IfNoneOutput，用于处理可选的输出
    class IfNoneOutput(torch.nn.Module):
        # 定义前向传播函数，接受一个张量 x 作为输入，返回一个可选的张量（Tensor 或 None）
        def forward(self, x) -> Optional[Tensor]:
            # 将输入 x 赋给 y，y 是一个可选的张量
            y: Optional[Tensor] = x
            # 如果输入张量 x 的第一维大于 1，将 y 设为 None
            if x.size(0) > 1:
                y = None
            return y

    # 定义一个继承自 torch.nn.Module 的模块 LoopNoneInput，用于处理可选的输入
    class LoopNoneInput(torch.nn.Module):
        # 定义前向传播函数，接受一个张量 x 作为输入，返回一个可选的张量（Tensor 或 None）
        def forward(self, x) -> Optional[Tensor]:
            # 初始化 y 为 None
            y: Optional[Tensor] = None
            # 循环 x.size(0) 次，每次将 x 赋给 y
            for _ in range(x.size(0)):
                y = x
            return y

    # 定义一个继承自 torch.nn.Module 的模块 LoopNoneOutput，用于处理可选的输出
    class LoopNoneOutput(torch.nn.Module):
        # 定义前向传播函数，接受一个张量 x 作为输入，返回一个可选的张量（Tensor 或 None）
        def forward(self, x) -> Optional[Tensor]:
            # 将输入张量 x 赋给 y，y 是一个可选的张量
            y: Optional[Tensor] = x
            # 循环 x.size(0) 次，每次将 y 设为 None
            for _ in range(x.size(0)):
                y = None
            return y

    # 使用 common_utils.parametrize 装饰器，为测试函数 test_optional_output 参数化设置
    @common_utils.parametrize(
        "module_class",
        (IfNoneOutput, IfNoneInput, LoopNoneOutput, LoopNoneInput),
        name_fn=lambda module_class: module_class.__name__,
    )
    # 参数化测试的输入大小 x_size，可以为 0 或 1
    @common_utils.parametrize("x_size", (0, 1), name_fn=lambda x_size: str(x_size))
    # 跳过堆栈跟踪的测试
    @skipTraceTest()
    # 如果不支持最小的操作集版本 16，则跳过测试
    @skipIfUnsupportedMinOpsetVersion(16)
    # 定义测试函数 test_optional_output，接受模块类 module_class 和输入大小 x_size 作为参数
    def test_optional_output(self, module_class: Type[torch.nn.Module], x_size: int):
        # 需要使用脚本化以保留此测试的控制流的含义
        model = torch.jit.script(module_class())
        # 创建一个字节流对象 f
        f = io.BytesIO()
        # 创建一个大小为 x_size 的全为 1 的张量 x
        x = torch.ones(x_size)
        # 动态轴名称为 "condition"，导出模型到 ONNX 格式
        dynamic_axis_name = "condition"
        torch.onnx.export(
            model,
            x,
            f,
            opset_version=self.opset_version,
            # 确保条件不是常量
            dynamic_axes={"x": {0: dynamic_axis_name}},
            input_names=["x"],
        )
        # 从导出的 ONNX 字符串中加载模型
        exported = onnx.load_from_string(f.getvalue())
        # 获取期望的元素类型
        expected_elem_type = torch.onnx.JitScalarType.from_value(x).onnx_type()
        # 创建期望的输出类型为可选类型的张量
        expected_output_type = onnx.helper.make_optional_type_proto(
            onnx.helper.make_tensor_type_proto(expected_elem_type, (dynamic_axis_name,))
        )
        # 断言导出的输出类型与期望的输出类型相同
        self.assertEqual(expected_output_type, exported.graph.output[0].type)
        # 遍历导出的图中的节点
        for node in exported.graph.node:
            # 如果节点类型为 "If"
            if node.op_type == "If":
                # 遍历节点的属性
                for attr in node.attribute:
                    # 如果属性名为 "then_branch" 或 "else_branch"
                    if attr.name in ("then_branch", "else_branch"):
                        # 断言两个分支的输出类型相同
                        self.assertEqual(expected_output_type, attr.g.output[0].type)

        # 运行测试，验证模块类 module_class 的输出
        self.run_test(
            module_class(),
            x,
            # 确保条件不是常量
            dynamic_axes={"x": {0: dynamic_axis_name}},
            input_names=["x"],
        )

    # 跳过堆栈跟踪的测试
    @skipTraceTest()
    # 如果不支持最小的操作集版本 16，则跳过测试
    @skipIfUnsupportedMinOpsetVersion(16)
    def test_uninitialized_optional(self):
        # 定义一个继承自 torch.nn.Module 的类 Module
        class Module(torch.nn.Module):
            # 定义模型的前向传播函数 forward，接受一个可选的 Tensor 类型参数 y，并返回一个可选的 Tensor 类型结果
            def forward(self, y: Optional[Tensor]) -> Optional[Tensor]:
                # 如果 y 不为 None，则进行以下操作
                if y is not None:
                    # 如果 y 的第二维度小于 5，则执行以下操作
                    if y.shape[1] < 5:
                        # 如果 y 的第一维度为 1，则将 y 的每个元素加 4
                        if y.size(0) == 1:
                            y = y + 4
                        else:
                            # 否则直接返回 y
                            return y
                # 返回处理后的 y 或者原始的 y（如果没有进行处理）
                return y

        # 运行测试函数 run_test，测试 Module 的行为
        self.run_test(
            Module(),  # 创建 Module 类的实例
            torch.ones((3, 4), dtype=torch.int),  # 创建一个形状为 (3, 4) 的全 1 Tensor，数据类型为 torch.int
            dynamic_axes={"y": {0: "y0", 1: "y1"}},  # 指定动态轴信息，用于 ONNX 转换
            input_names=["y"],  # 指定输入名称为 "y"
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_device_eq(self):
        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            # 定义模型的前向传播函数 forward，接受一个参数 a
            def forward(self, a):
                # 检查参数 a 的设备是否不是 CPU
                if a.device != torch.device("cpu"):
                    # 如果不是 CPU，则直接返回 a
                    return a
                # 如果是 CPU，则返回一个与 a 形状相同的全 0 Tensor
                return torch.zeros_like(a)

        # 将类 M 实例化为 TorchScript 模块 mod，保持控制流
        mod = torch.jit.script(M())

        # 运行测试函数 run_test，测试 mod 的行为
        self.run_test(
            mod,  # 输入模型 mod
            torch.randn(3, 3, device="cpu"),  # 创建一个在 CPU 上的随机数 Tensor
            input_names=["a"],  # 指定输入名称为 "a"
            dynamic_axes={"a": {0: "a0"}},  # 指定动态轴信息，用于 ONNX 转换
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lerp(self):
        # 定义一个继承自 torch.nn.Module 的类 LerpModel
        class LerpModel(torch.nn.Module):
            # 定义模型的前向传播函数 forward，接受一个参数 x
            def forward(self, x):
                # 使用 x.lerp 对输入 x 进行多个线性插值操作，并返回结果的元组
                return (
                    x.lerp(torch.full_like(x, 10), 0.4),  # 线性插值操作，插值到全 10 的 Tensor
                    x.lerp(torch.full_like(x, 20), 0.7),  # 线性插值操作，插值到全 20 的 Tensor
                    x.lerp(torch.full_like(x, 30), torch.tensor(0.4)),  # 线性插值操作，插值到全 30 的 Tensor
                    x.lerp(torch.full_like(x, 40), x / 10.0),  # 线性插值操作，插值到 x / 10.0 的 Tensor
                    x.lerp(torch.tensor(10.0), x / 10.0),  # 线性插值操作，插值到全 10.0 的 Tensor
                    x.lerp(torch.tensor(10.0), 0.4),  # 线性插值操作，插值到全 10.0 的 Tensor
                    x.lerp(torch.tensor(10.0), torch.tensor(0.4)),  # 线性插值操作，插值到全 10.0 的 Tensor
                )

        # 运行测试函数 run_test，测试 LerpModel 的行为
        self.run_test(LerpModel(), torch.rand(5, 4, 3))

    @common_utils.parametrize("input_dtype", [torch.cfloat, torch.float])
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_print_tensor_within_torch_nn_module(self, input_dtype: torch.dtype):
        # 定义一个继承自torch.nn.Module的内部类PrintTensorOnMyModel，用于测试打印张量的功能
        class PrintTensorOnMyModel(torch.nn.Module):
            def forward(self, x):
                # 获取张量x的第一列数据
                x_firsts = x[:, 0]
                # 打印张量x的第一列数据
                print(f"x_firsts: {x_firsts}")
                # 将张量x转换为Python列表，并声明其类型为List[float]
                _: List[float] = x.tolist()
                return x_firsts

        # 创建PrintTensorOnMyModel类的实例m
        m = PrintTensorOnMyModel()
        # 生成一个随机张量x，大小为10x5，数据类型为input_dtype
        x = torch.randn(10, 5, dtype=input_dtype)
        # 如果input_dtype是torch.cfloat类型，则期望引发RuntimeError异常
        if input_dtype == torch.cfloat:
            with self.assertRaises(RuntimeError):
                self.run_test(
                    m,
                    x,
                )
        else:
            # 否则运行测试函数self.run_test，传入模型m和张量x
            self.run_test(
                m,
                x,
            )

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(16)
    @unittest.skipIf(
        not torch.hub._check_module_exists("torch_geometric"),
        "torch_geometric not installed.",
    )
    # 定义测试函数test_sage_conv，测试SAGEConv模型在给定输入下的运行情况
    def test_sage_conv(self):
        from torch_geometric import nn as torch_geometric_nn

        # 输入数据准备
        coords0 = torch.randn(1, 6)  # 生成随机张量coords0，大小为1x6
        coords1 = torch.randn(1, 6)  # 生成随机张量coords1，大小为1x6
        coords = torch.transpose(torch.cat((coords0, coords1), dim=0), 0, 1)  # 拼接和转置coords0和coords1得到coords张量
        adj = torch_geometric_nn.knn_graph(coords, k=2, batch=None, loop=True)  # 根据coords生成KNN图的邻接矩阵adj
        edge_from = adj[0:1, :]  # 从adj中获取部分边的起始节点信息
        edge_to = adj[1:, :]  # 从adj中获取部分边的终止节点信息
        inputs = (coords0, coords1, edge_from, edge_to)  # 将所有输入数据组合成一个tuple

        # 定义MySAGEConv模型类，继承自torch.nn.Module，用于SAGEConv模型的测试
        class MySAGEConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.SAGEConvBlock1 = torch_geometric_nn.SAGEConv(
                    2, 512, normalize=True
                )  # 使用torch_geometric_nn中的SAGEConv定义SAGEConvBlock1层
                self.bano1 = torch_geometric_nn.BatchNorm(512)  # BatchNorm层
                self.relu = torch.nn.ReLU()  # ReLU激活函数
                self.dense1 = torch.nn.Seq(Lin(512, 1))  # 使用Lin层定义dense1层（此处忽略F821错误）
                self.sigmoid = torch.nn.Sigmoid()  # Sigmoid激活函数

            def forward(self, coords0, coords1, edge_from, edge_to):
                adj = torch.cat((edge_from, edge_to), dim=0)  # 拼接起始和终止节点得到新的邻接矩阵adj
                gra = torch.transpose(torch.cat((coords0, coords1), dim=0), 0, 1)  # 拼接和转置coords0和coords1得到gra张量
                x1 = self.SAGEConvBlock1(gra, edge_index=adj)  # 使用SAGEConvBlock1处理输入数据
                x = torch.unsqueeze(torch.sum(x1), dim=0)  # 对x1进行求和并添加新维度
                return x  # 返回处理后的张量x

        input_names = ["coords0", "coords1", "edge_from", "edge_to"]  # 定义输入数据的名称列表
        output_names = ["outputs"]  # 定义输出数据的名称列表
        dynamic_axes = {
            "coords0": {0: "batch_size", 1: "features"},
            "coords1": {0: "batch_size", 1: "features"},
            "edge_from": {0: "batch_size", 1: "features"},
            "edge_to": {0: "batch_size", 1: "features"},
            "outputs": {0: "batch_size"},
        }  # 定义动态轴映射，用于模型转换
        self.run_test(
            MySAGEConv(),  # 运行测试函数，传入MySAGEConv模型的实例
            inputs,  # 输入数据
            input_names=input_names,  # 输入数据的名称
            output_names=output_names,  # 输出数据的名称
            dynamic_axes=dynamic_axes,  # 动态轴映射
        )
    # 由于旧的 opsets 不支持 "ConstantFill" 操作，导致无法导出
    # ConstantFill 是一个临时操作，在 opset 8 中被移除，因此不再被 onnxruntime 支持
    # 仍然存在一些问题阻止我们为这些场景启用脚本测试：
    # test_gru_*:
    #   运算符 aten::as_tensor 目前导出器尚不支持。
    #       - https://msdata.visualstudio.com/Vienna/_workitems/edit/1055382
    #   运算符 aten::_pack_padded_sequence 目前导出器尚不支持。
    #       - https://msdata.visualstudio.com/Vienna/_workitems/edit/1055384
    # test_elman_*:
    #   在脚本模式下编译失败，出现如下错误：
    #   torch.jit.frontend.UnsupportedNodeError: annotated assignments
    #   without assigned value aren't supported
    #       - https://msdata.visualstudio.com/Vienna/_workitems/edit/1160723
    # test_lstm_*:
    #   在脚本模式下编译失败，出现如下错误：
    #   RuntimeError: Arguments for call are not valid.
    #       - https://msdata.visualstudio.com/Vienna/_workitems/edit/1160723
    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(9)
    @common_utils.parametrize(
        "name, nonlinearity",
        [
            ("elman", "relu"),
            ("elman", "tanh"),
            ("lstm", None),
            ("gru", None),
        ],
    )
    @common_utils.parametrize(**_parametrize_rnn_args("layers"))
    @common_utils.parametrize(**_parametrize_rnn_args("bidirectional"))
    @common_utils.parametrize(**_parametrize_rnn_args("initial_state"))
    @common_utils.parametrize(**_parametrize_rnn_args("packed_sequence"))
    @common_utils.parametrize(**_parametrize_rnn_args("dropout"))
    # 测试函数装饰器，用于测试各种 RNN 模型的不同参数组合
    def test_rnn(self, *args, **kwargs):
        # 调用内部方法来分派 RNN 测试
        self._dispatch_rnn_test(*args, **kwargs)
# 如果这个脚本被直接执行（而非被作为模块导入），则执行以下代码
if __name__ == "__main__":
    # 启用通用测试工具中的默认数据类型检查
    common_utils.TestCase._default_dtype_check_enabled = True
    # 运行通用测试工具中的测试函数
    common_utils.run_tests()
```