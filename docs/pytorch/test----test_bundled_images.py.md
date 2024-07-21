# `.\pytorch\test\test_bundled_images.py`

```
#!/usr/bin/env python3
# Owner(s): ["oncall: mobile"]
# mypy: allow-untyped-defs

# 导入必要的库
import io  # 导入用于创建字节流的模块
import cv2  # 导入 OpenCV 库

import torch  # 导入 PyTorch 深度学习框架
import torch.utils.bundled_inputs  # 导入用于捆绑输入数据的工具
from torch.testing._internal.common_utils import TestCase  # 导入 PyTorch 测试框架中的测试用例类

# 加载自定义的 C++ 库中的操作符
torch.ops.load_library("//caffe2/torch/fb/operators:decode_bundled_image")

# 定义函数：计算模型大小并返回
def model_size(sm):
    buffer = io.BytesIO()  # 创建一个字节流缓冲区
    torch.jit.save(sm, buffer)  # 将模型 sm 保存到字节流缓冲区中
    return len(buffer.getvalue())  # 返回字节流缓冲区的大小

# 定义函数：保存模型到字节流并加载返回
def save_and_load(sm):
    buffer = io.BytesIO()  # 创建一个字节流缓冲区
    torch.jit.save(sm, buffer)  # 将模型 sm 保存到字节流缓冲区中
    buffer.seek(0)  # 将读写位置移动到字节流的开头
    return torch.jit.load(buffer)  # 从字节流中加载并返回模型

"""返回一个 InflatableArg 对象，包含压缩图像的张量及其解码方式

    关键字参数:
    img_tensor -- HWC 或 NCHW 格式的原始图像张量，像素值为无符号整数类型
                  如果是 NCHW 格式，N 应为 1
    quality -- 压缩图像所需的质量

"""


def bundle_jpeg_image(img_tensor, quality):
    # 将 NCHW 格式转换为 HWC 格式
    if img_tensor.dim() == 4:
        assert img_tensor.size(0) == 1  # 断言 NCHW 格式的维度中 N 为 1
        img_tensor = img_tensor[0].permute(1, 2, 0)  # 转换为 HWC 格式
    pixels = img_tensor.numpy()  # 将图像张量转换为 NumPy 数组
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]  # 设置 JPEG 编码参数
    _, enc_img = cv2.imencode(".JPEG", pixels, encode_param)  # 使用 OpenCV 编码为 JPEG 格式
    enc_img_tensor = torch.from_numpy(enc_img)  # 将编码后的图像转换为 PyTorch 张量
    enc_img_tensor = torch.flatten(enc_img_tensor).byte()  # 将图像张量展平并转换为字节类型
    obj = torch.utils.bundled_inputs.InflatableArg(
        enc_img_tensor, "torch.ops.fb.decode_bundled_image({})"
    )  # 创建 InflatableArg 对象，用于包含图像张量及其解码方式
    return obj


# 定义函数：从原始 BGR 图像获取张量
def get_tensor_from_raw_BGR(im) -> torch.Tensor:
    raw_data = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # 将 BGR 格式转换为 RGB 格式
    raw_data = torch.from_numpy(raw_data).float()  # 将 NumPy 数组转换为 PyTorch 浮点张量
    raw_data = raw_data.permute(2, 0, 1)  # 调整张量维度顺序为 CxHxW
    raw_data = torch.div(raw_data, 255).unsqueeze(0)  # 归一化并增加一维作为批处理维度
    return raw_data  # 返回处理后的张量

# 定义测试类：用于测试捆绑图像处理相关功能
class TestBundledImages(TestCase):
    # 定义一个测试函数，用于测试处理单个张量的功能
    def test_single_tensors(self):
        # 定义一个简单的 PyTorch 模型，将输入直接返回作为输出
        class SingleTensorModel(torch.nn.Module):
            def forward(self, arg):
                return arg
        
        # 使用 OpenCV 读取图像文件作为 numpy 数组
        im = cv2.imread("caffe2/test/test_img/p1.jpg")
        
        # 将 numpy 数组转换为 PyTorch 张量
        tensor = torch.from_numpy(im)
        
        # 将张量打包成可充气的参数，即在 JPEG 格式下进行打包
        inflatable_arg = bundle_jpeg_image(tensor, 90)
        
        # 将充气的参数放入一个列表中
        input = [(inflatable_arg,)]
        
        # 使用 TorchScript 对定义的模型进行脚本化
        sm = torch.jit.script(SingleTensorModel())
        
        # 使用 bundled_inputs 工具为模型添加打包输入
        torch.utils.bundled_inputs.augment_model_with_bundled_inputs(sm, input)
        
        # 将模型保存并重新加载
        loaded = save_and_load(sm)
        
        # 获取重新加载模型的所有打包输入
        inflated = loaded.get_all_bundled_inputs()
        
        # 从重新加载模型的打包输入中提取解码后的数据
        decoded_data = inflated[0][0]

        # 从原始的 BGR 图像数据中获取张量表示
        raw_data = get_tensor_from_raw_BGR(im)
        
        # 断言确保打包后的输入列表长度为 1
        self.assertEqual(len(inflated), 1)
        
        # 断言确保每个打包后的输入列表中包含的元素数量为 1
        self.assertEqual(len(inflated[0]), 1)
        
        # 断言确保原始数据和解码后数据的形状相同
        self.assertEqual(raw_data.shape, decoded_data.shape)
        
        # 断言确保原始数据和解码后数据在给定的误差范围内相等
        self.assertEqual(raw_data, decoded_data, atol=0.1, rtol=1e-01)

        # 检查 fb::image_decode_to_NCHW 函数是否按预期工作
        with open("caffe2/test/test_img/p1.jpg", "rb") as fp:
            # 创建权重和偏置张量，用于解码图像数据
            weight = torch.full((3,), 1.0 / 255.0).diag()
            bias = torch.zeros(3)
            
            # 从文件对象中读取字节数据，并将其转换为 torch 字节张量
            byte_tensor = torch.tensor(list(fp.read())).byte()
            
            # 使用 Torch 自定义运算符 fb::image_decode_to_NCHW 解码图像到 NCHW 格式的张量
            im2_tensor = torch.ops.fb.image_decode_to_NCHW(byte_tensor, weight, bias)
            
            # 断言确保原始数据和解码后数据的形状相同
            self.assertEqual(raw_data.shape, im2_tensor.shape)
            
            # 断言确保原始数据和解码后数据在给定的误差范围内相等
            self.assertEqual(raw_data, im2_tensor, atol=0.1, rtol=1e-01)
```