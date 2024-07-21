# `.\pytorch\test\onnx\test_models_onnxruntime.py`

```
# Owner(s): ["module: onnx"]

# 引入必要的库和模块
import os
import unittest
from collections import OrderedDict
from typing import List, Mapping, Tuple

# 引入自定义的测试工具和库
import onnx_test_common
import parameterized
import PIL
import pytorch_test_common
import test_models
import torchvision
from pytorch_test_common import skipIfUnsupportedMinOpsetVersion, skipScriptTest
from torchvision import ops
from torchvision.models.detection import (
    faster_rcnn,
    image_list,
    keypoint_rcnn,
    mask_rcnn,
    roi_heads,
    rpn,
    transform,
)

# 引入 PyTorch 库和模块
import torch
from torch import nn
from torch.testing._internal import common_utils

# 定义一个函数用于导出测试
def exportTest(
    self,
    model,
    inputs,
    rtol=1e-2,
    atol=1e-7,
    opset_versions=None,
    acceptable_error_percentage=None,
):
    # 默认测试的 opset 版本范围
    opset_versions = opset_versions if opset_versions else [7, 8, 9, 10, 11, 12, 13, 14]

    # 遍历所有 opset 版本进行测试
    for opset_version in opset_versions:
        self.opset_version = opset_version
        self.onnx_shape_inference = True
        # 运行模型测试，包括形状推断
        onnx_test_common.run_model_test(
            self,
            model,
            input_args=inputs,
            rtol=rtol,
            atol=atol,
            acceptable_error_percentage=acceptable_error_percentage,
        )

        # 如果脚本测试启用且 opset 版本大于 11，则创建脚本化模型进行测试
        if self.is_script_test_enabled and opset_version > 11:
            script_model = torch.jit.script(model)
            # 运行脚本化模型测试
            onnx_test_common.run_model_test(
                self,
                script_model,
                input_args=inputs,
                rtol=rtol,
                atol=atol,
                acceptable_error_percentage=acceptable_error_percentage,
            )


# 创建一个定制的测试类，继承自 ExportTestCase，用于导出模型测试
TestModels = type(
    "TestModels",
    (pytorch_test_common.ExportTestCase,),
    dict(
        test_models.TestModels.__dict__,  # 继承 TestModels 中的属性和方法
        is_script_test_enabled=False,  # 禁用脚本测试
        is_script=False,  # 不是脚本化模型
        exportTest=exportTest,  # 使用定义的导出测试函数
    ),
)


# 创建一个定制的测试类，用于测试新的 JIT API 和形状推断
TestModels_new_jit_API = type(
    "TestModels_new_jit_API",
    (pytorch_test_common.ExportTestCase,),
    dict(
        TestModels.__dict__,  # 继承 TestModels 的属性和方法
        exportTest=exportTest,  # 使用定义的导出测试函数
        is_script_test_enabled=True,  # 启用脚本测试
        is_script=True,  # 是脚本化模型
        onnx_shape_inference=True,  # 使用形状推断
    ),
)


# 辅助函数：从相对路径中获取图像数据
def _get_image(rel_path: str, size: Tuple[int, int]) -> torch.Tensor:
    # 数据目录路径
    data_dir = os.path.join(os.path.dirname(__file__), "assets")
    # 图像文件完整路径
    path = os.path.join(data_dir, *rel_path.split("/"))
    # 打开图像文件，转换为 RGB 模式，并调整大小
    image = PIL.Image.open(path).convert("RGB").resize(size, PIL.Image.BILINEAR)

    # 将图像转换为 PyTorch 张量并返回
    return torchvision.transforms.ToTensor()(image)


# 辅助函数：获取测试图像数据
def _get_test_images() -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    # 返回包含测试图像数据的元组
    return (
        [_get_image("grace_hopper_517x606.jpg", (100, 320))],  # 返回 Grace Hopper 图像数据
        [_get_image("rgb_pytorch.png", (250, 380))],  # 返回 PyTorch 图像数据
    )


# 辅助函数：获取图像特征
def _get_features(images):
    s0, s1 = images.shape[-2:]  # 获取图像的宽度和高度
    # 定义一个特征列表，包含五个元素，每个元素是一个包含随机数的张量
    features = [
        ("0", torch.rand(2, 256, s0 // 4, s1 // 4)),    # 特征 "0"，尺寸为 s0//4 x s1//4
        ("1", torch.rand(2, 256, s0 // 8, s1 // 8)),    # 特征 "1"，尺寸为 s0//8 x s1//8
        ("2", torch.rand(2, 256, s0 // 16, s1 // 16)),  # 特征 "2"，尺寸为 s0//16 x s1//16
        ("3", torch.rand(2, 256, s0 // 32, s1 // 32)),  # 特征 "3"，尺寸为 s0//32 x s1//32
        ("4", torch.rand(2, 256, s0 // 64, s1 // 64)),  # 特征 "4"，尺寸为 s0//64 x s1//64
    ]
    # 将特征列表转换为有序字典，以确保顺序一致性
    features = OrderedDict(features)
    # 返回有序字典作为函数的输出结果
    return features
# 初始化通用的 RCNN 变换对象
def _init_test_generalized_rcnn_transform():
    min_size = 100  # 设定最小尺寸为 100
    max_size = 200  # 设定最大尺寸为 200
    image_mean = [0.485, 0.456, 0.406]  # 图像均值数组
    image_std = [0.229, 0.224, 0.225]   # 图像标准差数组
    # 返回通用 RCNN 变换对象
    return transform.GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)


# 初始化 RPN（区域建议网络）对象
def _init_test_rpn():
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))  # 锚框尺寸元组
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)  # 锚框长宽比元组
    # 创建 RPN 锚框生成器对象
    rpn_anchor_generator = rpn.AnchorGenerator(anchor_sizes, aspect_ratios)
    out_channels = 256  # 输出通道数为 256
    # 创建 RPN 头部对象
    rpn_head = rpn.RPNHead(
        out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
    )
    # 设定 RPN 正样本 IOU 阈值为 0.7，负样本 IOU 阈值为 0.3
    rpn_fg_iou_thresh = 0.7
    rpn_bg_iou_thresh = 0.3
    rpn_batch_size_per_image = 256  # 每张图像的 RPN 批处理大小
    rpn_positive_fraction = 0.5  # 正样本占比为 0.5
    # RPN 预 NMS 保留框数目设定
    rpn_pre_nms_top_n = dict(training=2000, testing=1000)
    # RPN 后 NMS 保留框数目设定
    rpn_post_nms_top_n = dict(training=2000, testing=1000)
    rpn_nms_thresh = 0.7  # RPN NMS 阈值为 0.7
    rpn_score_thresh = 0.0  # RPN 分数阈值为 0.0

    # 返回区域建议网络对象
    return rpn.RegionProposalNetwork(
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


# 初始化 Faster RCNN 中的 ROI 头部对象
def _init_test_roi_heads_faster_rcnn():
    out_channels = 256  # 输出通道数为 256
    num_classes = 91  # 类别数为 91

    box_fg_iou_thresh = 0.5  # 边界框前景 IOU 阈值为 0.5
    box_bg_iou_thresh = 0.5  # 边界框背景 IOU 阈值为 0.5
    box_batch_size_per_image = 512  # 每张图像的边界框批处理大小为 512
    box_positive_fraction = 0.25  # 正样本占比为 0.25
    bbox_reg_weights = None  # 边界框回归权重设为 None
    box_score_thresh = 0.05  # 边界框得分阈值为 0.05
    box_nms_thresh = 0.5  # 边界框 NMS 阈值为 0.5
    box_detections_per_img = 100  # 每张图像的边界框检测数目为 100

    # 创建多尺度 ROI 对齐对象
    box_roi_pool = ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
    )

    resolution = box_roi_pool.output_size[0]  # ROI 池化层输出尺寸的分辨率
    representation_size = 1024  # 表征特征大小为 1024
    # 创建 Faster RCNN 中的两层 MLP 头部对象
    box_head = faster_rcnn.TwoMLPHead(
        out_channels * resolution**2, representation_size
    )

    representation_size = 1024  # 更新表征特征大小为 1024
    # 创建 Faster RCNN 预测器对象
    box_predictor = faster_rcnn.FastRCNNPredictor(representation_size, num_classes)

    # 返回 ROI 头部对象
    return roi_heads.RoIHeads(
        box_roi_pool,
        box_head,
        box_predictor,
        box_fg_iou_thresh,
        box_bg_iou_thresh,
        box_batch_size_per_image,
        box_positive_fraction,
        bbox_reg_weights,
        box_score_thresh,
        box_nms_thresh,
        box_detections_per_img,
    )


@parameterized.parameterized_class(
    ("is_script",),
    [(True,), (False,)],
    class_name_func=onnx_test_common.parameterize_class_name,
)
class TestModelsONNXRuntime(onnx_test_common._TestONNXRuntime):
    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()  # Faster RCNN 模型不支持脚本化
    # 定义测试函数 test_faster_rcnn，用于测试 Faster R-CNN 模型的转换和推理
    def test_faster_rcnn(self):
        # 使用 faster_rcnn 模块中的 fasterrcnn_resnet50_fpn 函数创建模型实例
        model = faster_rcnn.fasterrcnn_resnet50_fpn(
            pretrained=False, pretrained_backbone=True, min_size=200, max_size=300
        )
        # 将模型设置为评估模式
        model.eval()
        # 创建随机张量 x1 和 x2 作为输入数据，形状为 (3, 200, 300)，并设置其 requires_grad 为 True
        x1 = torch.randn(3, 200, 300, requires_grad=True)
        x2 = torch.randn(3, 200, 300, requires_grad=True)
        # 使用自定义的辅助函数 run_test 运行模型测试，输入参数为 x1 和 x2 张量
        self.run_test(model, ([x1, x2],), rtol=1e-3, atol=1e-5)
        # 再次使用 run_test 函数进行模型测试，同时指定输入和输出的名称以及动态维度
        self.run_test(
            model,
            ([x1, x2],),
            input_names=["images_tensors"],
            output_names=["outputs"],
            dynamic_axes={"images_tensors": [0, 1, 2], "outputs": [0, 1, 2]},
            rtol=1e-3,
            atol=1e-5,
        )
        # 创建虚拟的 dummy_image 作为输入，形状为 (3, 100, 100) 的全 0.3 张量
        dummy_image = [torch.ones(3, 100, 100) * 0.3]
        # 调用 _get_test_images 函数获取 images 和 test_images 作为输入
        images, test_images = _get_test_images()
        # 使用 run_test 函数测试模型，同时传入 images 和其他附加的测试输入数据
        self.run_test(
            model,
            (images,),
            additional_test_inputs=[(images,), (test_images,), (dummy_image,)],
            input_names=["images_tensors"],
            output_names=["outputs"],
            dynamic_axes={"images_tensors": [0, 1, 2], "outputs": [0, 1, 2]},
            rtol=1e-3,
            atol=1e-5,
        )
        # 再次使用 run_test 函数测试模型，传入 dummy_image 作为输入，并提供其他附加的测试数据
        self.run_test(
            model,
            (dummy_image,),
            additional_test_inputs=[(dummy_image,), (images,)],
            input_names=["images_tensors"],
            output_names=["outputs"],
            dynamic_axes={"images_tensors": [0, 1, 2], "outputs": [0, 1, 2]},
            rtol=1e-3,
            atol=1e-5,
        )

    # 标记当前测试函数为跳过状态，因为在 ONNX 1.13.0 之后出现问题导致测试失败
    @unittest.skip("Failing after ONNX 1.13.0")
    # 如果当前的操作集版本小于 11，则跳过这个测试
    @skipIfUnsupportedMinOpsetVersion(11)
    # 标记当前脚本测试为跳过状态
    @skipScriptTest()
    # 定义测试函数 test_mask_rcnn，用于测试 Mask R-CNN 模型的功能
    def test_mask_rcnn(self):
        # 创建 Mask R-CNN 模型，使用 maskrcnn_resnet50_fpn 函数初始化，设置参数如下：
        # pretrained=False: 不使用预训练模型
        # pretrained_backbone=True: 使用预训练的骨干网络
        # min_size=200: 图像最小尺寸为200
        # max_size=300: 图像最大尺寸为300
        model = mask_rcnn.maskrcnn_resnet50_fpn(
            pretrained=False, pretrained_backbone=True, min_size=200, max_size=300
        )
        
        # 调用 _get_test_images 函数获取测试用的图像数据
        images, test_images = _get_test_images()
        
        # 对模型执行测试，使用 self.run_test 方法，传入模型和图像数据 images，设置相对和绝对误差容忍度
        self.run_test(model, (images,), rtol=1e-3, atol=1e-5)
        
        # 再次调用 self.run_test 方法测试模型，传入模型和图像数据 images，并设置输入和输出的名称以及动态维度
        self.run_test(
            model,
            (images,),
            input_names=["images_tensors"],
            output_names=["boxes", "labels", "scores", "masks"],
            dynamic_axes={
                "images_tensors": [0, 1, 2],
                "boxes": [0, 1],
                "labels": [0],
                "scores": [0],
                "masks": [0, 1, 2],
            },
            rtol=1e-3,
            atol=1e-5,
        )
        
        # 使用 dummy_image 创建一个形状为 (3, 100, 100) 的张量列表，并将其传入模型进行测试
        dummy_image = [torch.ones(3, 100, 100) * 0.3]
        self.run_test(
            model,
            (images,),
            additional_test_inputs=[(images,), (test_images,), (dummy_image,)],
            input_names=["images_tensors"],
            output_names=["boxes", "labels", "scores", "masks"],
            dynamic_axes={
                "images_tensors": [0, 1, 2],
                "boxes": [0, 1],
                "labels": [0],
                "scores": [0],
                "masks": [0, 1, 2],
            },
            rtol=1e-3,
            atol=1e-5,
        )
        
        # 对模型使用 dummy_image 进行额外的测试，传入 dummy_image 数据并设置输入输出名称和动态维度
        self.run_test(
            model,
            (dummy_image,),
            additional_test_inputs=[(dummy_image,), (images,)],
            input_names=["images_tensors"],
            output_names=["boxes", "labels", "scores", "masks"],
            dynamic_axes={
                "images_tensors": [0, 1, 2],
                "boxes": [0, 1],
                "labels": [0],
                "scores": [0],
                "masks": [0, 1, 2],
            },
            rtol=1e-3,
            atol=1e-5,
        )

    # 标记下面的测试函数为跳过测试，原因是存在已知的问题
    @unittest.skip("Failing, see https://github.com/pytorch/pytorch/issues/66528")
    # 根据最小操作集版本是否支持，跳过脚本测试
    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    # 定义一个测试方法，用于测试 Keypoint R-CNN 模型
    def test_keypoint_rcnn(self):
        # 使用 ResNet-50 FPN 架构创建 Keypoint R-CNN 模型，不使用预训练权重
        model = keypoint_rcnn.keypointrcnn_resnet50_fpn(
            pretrained=False, pretrained_backbone=False, min_size=200, max_size=300
        )
        # 获取测试图像数据
        images, test_images = _get_test_images()
        # 运行测试，检查模型在 images 数据上的输出是否符合预期，设置数值容差
        self.run_test(model, (images,), rtol=1e-3, atol=1e-5)
        # 再次运行测试，这次指定输入输出名称和动态轴设置
        self.run_test(
            model,
            (images,),
            input_names=["images_tensors"],
            output_names=["outputs1", "outputs2", "outputs3", "outputs4"],
            dynamic_axes={"images_tensors": [0, 1, 2]},
            rtol=1e-3,
            atol=1e-5,
        )
        # 使用虚拟的 dummy_images 运行测试，测试额外的输入组合
        dummy_images = [torch.ones(3, 100, 100) * 0.3]
        self.run_test(
            model,
            (images,),
            additional_test_inputs=[(images,), (test_images,), (dummy_images,)],
            input_names=["images_tensors"],
            output_names=["outputs1", "outputs2", "outputs3", "outputs4"],
            dynamic_axes={"images_tensors": [0, 1, 2]},
            rtol=5e-3,
            atol=1e-5,
        )
        # 再次使用 dummy_images 运行测试，测试另一组额外输入
        self.run_test(
            model,
            (dummy_images,),
            additional_test_inputs=[(dummy_images,), (test_images,)],
            input_names=["images_tensors"],
            output_names=["outputs1", "outputs2", "outputs3", "outputs4"],
            dynamic_axes={"images_tensors": [0, 1, 2]},
            rtol=5e-3,
            atol=1e-5,
        )

    # 如果当前 Opset 版本小于 11，则跳过该测试
    @skipIfUnsupportedMinOpsetVersion(11)
    # 标记为不适合脚本测试
    @skipScriptTest()
    def test_roi_heads(self):
        # 定义一个测试函数，用于测试 RoIHeads 模块
        class RoIHeadsModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化通用的 RCNN 变换、RPN 和 Faster R-CNN 的 RoIHeads
                self.transform = _init_test_generalized_rcnn_transform()
                self.rpn = _init_test_rpn()
                self.roi_heads = _init_test_roi_heads_faster_rcnn()

            def forward(self, images, features: Mapping[str, torch.Tensor]):
                # 获取原始图像尺寸列表
                original_image_sizes = [
                    (img.shape[-1], img.shape[-2]) for img in images
                ]

                # 创建图像列表对象，包含图像和它们的尺寸
                images_m = image_list.ImageList(
                    images, [(i.shape[-1], i.shape[-2]) for i in images]
                )
                # 使用 RPN 模型生成建议框
                proposals, _ = self.rpn(images_m, features)
                # 使用 RoIHeads 模型进行检测
                detections, _ = self.roi_heads(
                    features, proposals, images_m.image_sizes
                )
                # 对检测结果进行后处理，恢复原始图像尺寸
                detections = self.transform.postprocess(
                    detections, images_m.image_sizes, original_image_sizes
                )
                # 返回检测结果
                return detections

        # 创建两个随机图像作为输入
        images = torch.rand(2, 3, 100, 100)
        features = _get_features(images)
        images2 = torch.rand(2, 3, 150, 150)
        test_features = _get_features(images2)

        # 实例化 RoIHeadsModule 模型
        model = RoIHeadsModule()
        model.eval()
        # 使用模型进行推理
        model(images, features)

        # 运行测试函数，验证模型的输出
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
        )

    @skipScriptTest()  # 标记：跳过脚本测试，待解决 issue #75625
    @skipIfUnsupportedMinOpsetVersion(20)
    def test_transformer_encoder(self):
        # 定义一个测试函数，用于测试 Transformer 编码器
        class MyModule(torch.nn.Module):
            def __init__(self, ninp, nhead, nhid, dropout, nlayers):
                super().__init__()
                # 创建 Transformer 编码器层
                encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
                # 创建 Transformer 编码器
                self.transformer_encoder = nn.TransformerEncoder(
                    encoder_layers, nlayers
                )

            def forward(self, input):
                # 执行 Transformer 编码器的前向传播
                return self.transformer_encoder(input)

        # 创建输入数据
        x = torch.rand(10, 32, 512)
        # 运行测试，验证 Transformer 编码器的输出
        self.run_test(MyModule(512, 8, 2048, 0.0, 3), (x,), atol=1e-5)

    @skipScriptTest()  # 标记：跳过脚本测试
    def test_mobilenet_v3(self):
        # 测试 MobileNetV3 模型
        model = torchvision.models.quantization.mobilenet_v3_large(pretrained=False)
        # 创建一个随机输入
        dummy_input = torch.randn(1, 3, 224, 224)
        # 运行测试，验证 MobileNetV3 模型的输出
        self.run_test(model, (dummy_input,))

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()  # 标记：跳过脚本测试
    def test_shufflenet_v2_dynamic_axes(self):
        # 使用 torchvision 提供的 shufflenet_v2_x0_5 模型，不加载预训练权重
        model = torchvision.models.shufflenet_v2_x0_5(weights=None)
        # 创建一个形状为 (1, 3, 224, 224) 的随机张量作为模型的输入，并允许梯度计算
        dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
        # 创建一个形状为 (3, 3, 224, 224) 的随机张量作为测试输入，并允许梯度计算
        test_inputs = torch.randn(3, 3, 224, 224, requires_grad=True)
        # 运行测试函数，传入模型、dummy_input作为元组的元素、额外的测试输入、输入名称、输出名称、动态轴信息、相对误差、绝对误差
        self.run_test(
            model,
            (dummy_input,),  # 模型输入的元组
            additional_test_inputs=[(dummy_input,), (test_inputs,)],  # 额外的测试输入，作为元组列表
            input_names=["input_images"],  # 输入的名称列表
            output_names=["outputs"],  # 输出的名称列表
            dynamic_axes={  # 定义动态轴的字典
                "input_images": {0: "batch_size"},  # "input_images" 的第 0 维作为 "batch_size"
                "output": {0: "batch_size"},  # "output" 的第 0 维作为 "batch_size"
            },
            rtol=1e-3,  # 相对误差容忍度
            atol=1e-5,  # 绝对误差容忍度
        )
# 如果当前脚本被直接执行（而不是被导入为模块），则执行以下代码块
if __name__ == "__main__":
    # 调用 common_utils 模块中的 run_tests 函数，用于运行测试用例
    common_utils.run_tests()
```