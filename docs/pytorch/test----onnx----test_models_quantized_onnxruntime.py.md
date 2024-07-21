# `.\pytorch\test\onnx\test_models_quantized_onnxruntime.py`

```
# Owner(s): ["module: onnx"]

# 引入标准库和第三方库
import os
import unittest

# 引入自定义测试工具和参数化库
import onnx_test_common
import parameterized
import PIL
import torchvision

# 引入PyTorch相关模块
import torch
from torch import nn


# 获取测试用的图片张量
def _get_test_image_tensor():
    # 数据目录为当前文件所在目录的assets子目录
    data_dir = os.path.join(os.path.dirname(__file__), "assets")
    # 图片路径为assets子目录下的grace_hopper_517x606.jpg
    img_path = os.path.join(data_dir, "grace_hopper_517x606.jpg")
    # 打开并加载图片为PIL对象
    input_image = PIL.Image.open(img_path)
    
    # 图像预处理，参考自https://pytorch.org/hub/pytorch_vision_resnet/
    preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),  # 调整大小至256x256
            torchvision.transforms.CenterCrop(224),  # 中心裁剪为224x224
            torchvision.transforms.ToTensor(),  # 转换为张量
            torchvision.transforms.Normalize(  # 标准化
                mean=[0.485, 0.456, 0.406],  # 均值
                std=[0.229, 0.224, 0.225]  # 标准差
            ),
        ]
    )
    return preprocess(input_image).unsqueeze(0)  # 返回预处理后的张量，增加批量维度


# 由于量化误差，只检查顶部预测是否匹配
class _TopPredictor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        x = self.base_model(x)
        _, topk_id = torch.topk(x[0], 1)  # 获取最高概率的类别索引
        return topk_id


# TODO: 所有torchvision的量化模型测试可以作为单个参数化测试用例编写，
# 在支持每个参数测试装饰时，通过＃79979，或在它们全部启用后，
# 先到达哪个。


@parameterized.parameterized_class(
    ("is_script",),  # 参数化类，根据是否是脚本模式进行分类
    [(True,), (False,)],  # 参数：True和False两种情况
    class_name_func=onnx_test_common.parameterize_class_name,  # 类名生成函数
)
class TestQuantizedModelsONNXRuntime(onnx_test_common._TestONNXRuntime):
    def run_test(self, model, inputs, *args, **kwargs):
        model = _TopPredictor(model)  # 使用_TopPredictor封装模型
        return super().run_test(model, inputs, *args, **kwargs)

    def test_mobilenet_v3(self):
        # 获取预训练的并且量化的mobilenet_v3_large模型
        model = torchvision.models.quantization.mobilenet_v3_large(
            pretrained=True, quantize=True
        )
        self.run_test(model, _get_test_image_tensor())  # 运行测试

    @unittest.skip("quantized::cat not supported")  # 跳过测试：quantized::cat不支持
    def test_inception_v3(self):
        # 获取预训练的并且量化的inception_v3模型
        model = torchvision.models.quantization.inception_v3(
            pretrained=True, quantize=True
        )
        self.run_test(model, _get_test_image_tensor())  # 运行测试

    @unittest.skip("quantized::cat not supported")  # 跳过测试：quantized::cat不支持
    def test_googlenet(self):
        # 获取预训练的并且量化的googlenet模型
        model = torchvision.models.quantization.googlenet(
            pretrained=True, quantize=True
        )
        self.run_test(model, _get_test_image_tensor())  # 运行测试

    @unittest.skip("quantized::cat not supported")  # 跳过测试：quantized::cat不支持
    def test_shufflenet_v2_x0_5(self):
        # 获取预训练的并且量化的shufflenet_v2_x0_5模型
        model = torchvision.models.quantization.shufflenet_v2_x0_5(
            pretrained=True, quantize=True
        )
        self.run_test(model, _get_test_image_tensor())  # 运行测试

    def test_resnet18(self):
        # 获取预训练的并且量化的resnet18模型
        model = torchvision.models.quantization.resnet18(pretrained=True, quantize=True)
        self.run_test(model, _get_test_image_tensor())  # 运行测试
    # 定义测试 ResNet-50 模型的方法
    def test_resnet50(self):
        # 使用 torchvision 提供的预训练和量化后的 ResNet-50 模型
        model = torchvision.models.quantization.resnet50(pretrained=True, quantize=True)
        # 运行测试方法，传入模型和测试图像张量
        self.run_test(model, _get_test_image_tensor())

    # 定义测试 ResNeXt-101 32x8d 模型的方法
    def test_resnext101_32x8d(self):
        # 使用 torchvision 提供的预训练和量化后的 ResNeXt-101 32x8d 模型
        model = torchvision.models.quantization.resnext101_32x8d(
            pretrained=True, quantize=True
        )
        # 运行测试方法，传入模型和测试图像张量
        self.run_test(model, _get_test_image_tensor())
```